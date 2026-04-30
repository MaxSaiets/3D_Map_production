from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union

from services.building_geometry_pipeline import prepare_building_geometry
from services.building_processor import get_building_height
from services.canonical_2d_pipeline import _expand_building_mask_for_roads, _subtract_masks
from services.detail_layer_pipeline import _build_canonical_road_masks
from services.detail_layer_utils import MIN_LAND_WIDTH_MODEL_MM, prepare_green_areas_for_processing
from services.data_loader import fetch_city_data
from services.extras_loader import fetch_extras
from services.full_generation_pipeline import _prepare_zone_stage, run_canonical_preview_pipeline, run_full_generation_pipeline
from services.generation_runtime_context import prepare_generation_runtime_context
from services.generation_task import GenerationTask
from services.geometry_preclip_pipeline import prepare_preclipped_geometry
from services.global_center import GlobalCenter
from services.green_processor import process_green_areas
from services.printer_profile import get_printer_profile_for_request
from services.processing_results import SourceDataResult, ZonePreparationResult
from services.road_geometry_pipeline import prepare_road_geometry
from services.road_processor import normalize_drivable_highway_tag
from services.terrain_generator import create_terrain_mesh
from services.terrain_pipeline_utils import resolve_generation_source_crs
from services.water_layer_pipeline import _prepare_water_polygons
from services.zone_context_pipeline import build_zone_context
from services.zone_geometry_pipeline import prepare_zone_geometry


PREVIEW_CACHE_DIR = Path(os.getenv("PREVIEW_CACHE_DIR", "cache/site_previews"))
PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_CANONICAL_DIR = PREVIEW_CACHE_DIR / "canonical"
PREVIEW_CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_JOBS_DIR = PREVIEW_CACHE_DIR / "jobs"
PREVIEW_JOBS_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_OUTPUT_DIR = Path(os.getenv("PREVIEW_OUTPUT_DIR", "output")).resolve()
PREVIEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_WORKER_TIMEOUT_SECONDS = int(os.getenv("PREVIEW_WORKER_TIMEOUT_SECONDS", "1800"))
PREVIEW_MAX_ACTIVE_WORKERS = int(os.getenv("PREVIEW_MAX_ACTIVE_WORKERS", "1"))
PREVIEW_FAILED_RETRY_DELAY_SECONDS = int(os.getenv("PREVIEW_FAILED_RETRY_DELAY_SECONDS", "90"))

MAX_FEATURES = {
    "buildings": 220,
    "roads": 260,
    "water": 60,
    "parks": 70,
}

ROAD_WIDTH_MAP_M = {
    "motorway": 4.8,
    "motorway_link": 4.2,
    "trunk": 4.2,
    "trunk_link": 3.7,
    "primary": 3.7,
    "primary_link": 3.2,
    "secondary": 3.2,
    "secondary_link": 2.8,
    "tertiary": 2.6,
    "tertiary_link": 2.3,
    "residential": 2.1,
    "living_street": 1.8,
    "service": 1.6,
    "unclassified": 1.9,
}


def _cache_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def _empty_collection() -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": []}


def _bbox_from_bounds(bounds: dict[str, float]) -> Polygon:
    return box(float(bounds["west"]), float(bounds["south"]), float(bounds["east"]), float(bounds["north"]))


def _selection_geometry(bounds: dict[str, float], polygon_geojson: Optional[dict[str, Any]]) -> BaseGeometry:
    bbox = _bbox_from_bounds(bounds)
    if not polygon_geojson:
        return bbox
    try:
        geom = shape(polygon_geojson)
        if geom.is_empty:
            return bbox
        clipped = geom.intersection(bbox)
        return clipped if not clipped.is_empty else bbox
    except Exception:
        return bbox


def _features_from_bbox(bounds: dict[str, float], tags: dict[str, Any]) -> gpd.GeoDataFrame:
    north = float(bounds["north"])
    south = float(bounds["south"])
    east = float(bounds["east"])
    west = float(bounds["west"])
    try:
        return ox.features_from_bbox(north=north, south=south, east=east, west=west, tags=tags)
    except TypeError:
        try:
            return ox.features_from_bbox((west, south, east, north), tags)
        except TypeError:
            return ox.features_from_bbox(north, south, east, west, tags)
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def _filter_by_tag(gdf: gpd.GeoDataFrame, tag: str, values: Any = True) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty or tag not in gdf.columns:
        return gpd.GeoDataFrame(geometry=[], crs=getattr(gdf, "crs", "EPSG:4326"))
    series = gdf[tag]
    if values is True:
        mask = series.notna()
    else:
        allowed = set(values if isinstance(values, list) else [values])
        mask = series.apply(lambda value: any(str(item) in allowed for item in value) if isinstance(value, list) else str(value) in allowed)
    return gdf[mask].copy()


def _merge_frames(*frames: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    valid = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    merged = gpd.GeoDataFrame(
        pd.concat(valid, ignore_index=True),
        geometry="geometry",
        crs=valid[0].crs or "EPSG:4326",
    )
    try:
        merged["_preview_wkb"] = merged.geometry.apply(lambda geom: geom.wkb_hex if geom is not None else "")
        merged = merged.drop_duplicates(subset=["_preview_wkb"]).drop(columns=["_preview_wkb"])
    except Exception:
        pass
    return merged.copy()


def _clip_to_selection(
    gdf: gpd.GeoDataFrame,
    bounds: dict[str, float],
    polygon_geojson: Optional[dict[str, Any]],
) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    try:
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        elif str(gdf.crs).lower() not in {"epsg:4326", "4326"}:
            gdf = gdf.to_crs("EPSG:4326")
        selection = _selection_geometry(bounds, polygon_geojson)
        clipped = gpd.clip(gdf, gpd.GeoSeries([selection], crs="EPSG:4326"))
        clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()].copy()
        return clipped
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def _numeric_width(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        text = str(value).replace(",", ".")
        token = ""
        for char in text:
            if char.isdigit() or char == ".":
                token += char
            elif token:
                break
        if not token:
            return None
        width = float(token)
        if 0.8 <= width <= 80.0:
            return width
    except Exception:
        return None
    return None


def _road_buffer_radius_m(row: Any, road_width_multiplier: float = 0.8) -> float:
    highway = normalize_drivable_highway_tag(row.get("highway"))
    fallback = ROAD_WIDTH_MAP_M.get(highway or "", 1.6)
    explicit = _numeric_width(row.get("width"))
    width = explicit if explicit is not None else fallback
    if explicit is not None:
        width = min(max(explicit, 0.8), max(fallback * 1.15, fallback + 0.35))
    width *= max(0.1, min(float(road_width_multiplier or 0.8), 5.0))
    return max(width / 2.0, 0.35)


def _build_preview_road_mask(
    roads: gpd.GeoDataFrame,
    bounds: dict[str, float],
    polygon_geojson: Optional[dict[str, Any]],
    road_width_multiplier: float,
) -> gpd.GeoDataFrame:
    clipped = _clip_to_selection(roads, bounds, polygon_geojson)
    if clipped.empty:
        return clipped

    try:
        projected = clipped.to_crs("EPSG:3857")
        geoms = []
        records = []
        for _, row in projected.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if isinstance(geom, (LineString, MultiLineString)):
                mask = geom.buffer(_road_buffer_radius_m(row, road_width_multiplier), cap_style=2, join_style=1, resolution=3)
            elif isinstance(geom, (Polygon, MultiPolygon)):
                mask = geom.buffer(0)
            else:
                continue
            if mask.is_empty:
                continue
            geoms.append(mask)
            records.append({"highway": row.get("highway"), "name": row.get("name")})
        if not geoms:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        masked = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:3857").to_crs("EPSG:4326")
        return _clip_to_selection(masked, bounds, polygon_geojson)
    except Exception:
        return clipped


def _subtract_buildings_from_roads(roads: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if roads is None or roads.empty or buildings is None or buildings.empty:
        return roads
    try:
        road_proj = roads.to_crs("EPSG:3857")
        building_proj = buildings.to_crs("EPSG:3857")
        building_union = unary_union([geom for geom in building_proj.geometry if geom is not None and not geom.is_empty])
        if building_union.is_empty:
            return roads
        road_proj["geometry"] = road_proj.geometry.apply(lambda geom: geom.difference(building_union) if geom is not None else geom)
        road_proj = road_proj[road_proj.geometry.notna() & ~road_proj.geometry.is_empty].copy()
        road_proj = road_proj.explode(index_parts=False, ignore_index=True)
        road_proj = road_proj[road_proj.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        return road_proj.to_crs("EPSG:4326")
    except Exception:
        return roads


def _height_from_row(row: Any, building_min_height: float = 5.0, building_height_multiplier: float = 1.8) -> float:
    raw = row.get("height")
    levels = row.get("building:levels")
    min_height = max(1.0, float(building_min_height or 5.0))
    multiplier = max(0.1, float(building_height_multiplier or 1.8))
    height = _numeric_width(raw)
    if height is not None:
        return max(min_height, min(height * multiplier, 120.0))
    try:
        if levels is not None and not (isinstance(levels, float) and math.isnan(levels)):
            return max(min_height, min(float(levels) * 3.0 * multiplier, 100.0))
    except Exception:
        pass
    return min_height


def _feature_collection(
    gdf: gpd.GeoDataFrame,
    limit: int,
    simplify: float,
    layer: str,
    *,
    building_min_height: float = 5.0,
    building_height_multiplier: float = 1.8,
    scale_factor_mm_per_m: float = 1.0,
) -> dict[str, Any]:
    features = []
    if gdf is None or gdf.empty:
        return {"type": "FeatureCollection", "features": []}
    for _, row in gdf.head(limit).iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            if simplify > 0:
                geom = geom.simplify(simplify, preserve_topology=True)
            props = {"layer": layer}
            if layer == "buildings":
                height_m = _height_from_row(row, building_min_height, building_height_multiplier)
                props["height_m"] = height_m
                props["height_mm"] = max(0.4, min(height_m * scale_factor_mm_per_m, 45.0))
            if row.get("name") is not None:
                props["name"] = str(row.get("name"))[:80]
            features.append({"type": "Feature", "properties": props, "geometry": mapping(geom)})
        except Exception:
            continue
    return {"type": "FeatureCollection", "features": features}


class _PreviewTask:
    def update_status(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _request_namespace(
    *,
    bounds: dict[str, float],
    road_width_multiplier: float,
    building_min_height: float,
    building_height_multiplier: float,
    model_size_mm: float,
    terrain_z_scale: float,
    terrain_resolution: int,
    terrain_base_thickness_mm: float,
    road_height_mm: float,
    road_embed_mm: float,
    building_foundation_mm: float,
    building_embed_mm: float,
    water_depth: float,
    parks_height_mm: float,
    parks_embed_mm: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        north=float(bounds["north"]),
        south=float(bounds["south"]),
        east=float(bounds["east"]),
        west=float(bounds["west"]),
        road_width_multiplier=float(road_width_multiplier),
        road_height_mm=float(road_height_mm),
        road_embed_mm=float(road_embed_mm),
        building_min_height=float(building_min_height),
        building_height_multiplier=float(building_height_multiplier),
        # The full 3D building layer reads these names. Keep them aligned so
        # preview and generated model use the same recipe values.
        buildings_height_scale=float(building_height_multiplier),
        building_foundation_mm=float(building_foundation_mm),
        building_embed_mm=float(building_embed_mm),
        building_max_foundation_mm=2.5,
        buildings_embed_mm=float(building_embed_mm),
        water_depth=float(water_depth),
        include_parks=True,
        parks_height_mm=float(parks_height_mm),
        parks_embed_mm=float(parks_embed_mm),
        terrain_enabled=True,
        terrain_z_scale=float(terrain_z_scale),
        terrain_base_thickness_mm=float(terrain_base_thickness_mm),
        terrain_resolution=int(terrain_resolution),
        terrain_subdivide=True,
        terrain_subdivide_levels=1,
        terrarium_zoom=15,
        terrain_smoothing_sigma=2.0,
        flatten_buildings_on_terrain=True,
        flatten_roads_on_terrain=False,
        export_format="3mf",
        model_size_mm=float(model_size_mm),
        preview_include_base=True,
        preview_include_roads=True,
        preview_include_buildings=True,
        preview_include_water=True,
        preview_include_parks=True,
        context_padding_m=80.0,
        terrain_only=False,
        elevation_ref_m=None,
        baseline_offset_m=0.0,
        preserve_global_xy=False,
        grid_step_m=None,
        hex_size_m=300.0,
        is_ams_mode=False,
        include_buildings=True,
        canonical_mask_bundle_dir=None,
        auto_canonicalize_masks=True,
    )


def _zone_polygon_coords_from_geojson(polygon_geojson: Optional[dict[str, Any]]) -> Optional[list]:
    if not polygon_geojson:
        return None
    try:
        geom = shape(polygon_geojson)
        if geom is None or geom.is_empty:
            return None
        if geom.geom_type == "Polygon":
            return [[float(x), float(y)] for x, y in list(geom.exterior.coords)]
        if geom.geom_type == "MultiPolygon":
            largest = max(list(geom.geoms), key=lambda part: float(part.area))
            return [[float(x), float(y)] for x, y in list(largest.exterior.coords)]
    except Exception:
        return None
    return None


def _local_to_wgs84_transformer(global_center: GlobalCenter):
    def _convert(x: float, y: float, z: Optional[float] = None):
        x_utm, y_utm = global_center.from_local(float(x), float(y))
        lon, lat = global_center.to_wgs84(x_utm, y_utm)
        if z is None:
            return lon, lat
        return lon, lat, z

    return _convert


def _geometry_feature_collection(
    geometry: BaseGeometry | None,
    *,
    limit: int,
    layer: str,
    global_center: GlobalCenter,
    props: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return {"type": "FeatureCollection", "features": []}
    features = []
    converter = _local_to_wgs84_transformer(global_center)
    parts = [geometry] if getattr(geometry, "geom_type", "") == "Polygon" else list(getattr(geometry, "geoms", []))
    for geom in parts[:limit]:
        if geom is None or getattr(geom, "is_empty", True):
            continue
        try:
            wgs84_geom = transform_geometry(converter, geom)
            feature_props = {"layer": layer}
            if props:
                feature_props.update(props)
            features.append({"type": "Feature", "properties": feature_props, "geometry": mapping(wgs84_geom)})
        except Exception:
            continue
    return {"type": "FeatureCollection", "features": features}


def _clip_mask_to_zone(geometry: BaseGeometry | None, zone_polygon: BaseGeometry | None) -> BaseGeometry | None:
    if geometry is None or getattr(geometry, "is_empty", True):
        return None
    if zone_polygon is None or getattr(zone_polygon, "is_empty", True):
        return geometry
    try:
        clipped = geometry.intersection(zone_polygon)
        if clipped is None or getattr(clipped, "is_empty", True):
            return None
        if "Polygon" in str(getattr(clipped, "geom_type", "")):
            return clipped.buffer(0)
        return clipped
    except Exception:
        return geometry


def _canonical_building_collection(
    *,
    gdf_buildings_local: Optional[gpd.GeoDataFrame],
    building_mask: BaseGeometry | None,
    global_center: GlobalCenter,
    scale_factor_mm_per_m: float,
    height_scale_factor: float,
    limit: int,
) -> dict[str, Any]:
    if gdf_buildings_local is None or gdf_buildings_local.empty:
        return _geometry_feature_collection(
            building_mask,
            limit=limit,
            layer="buildings",
            global_center=global_center,
            props={"height_m": 4.0, "height_mm": max(0.4, 4.0 * scale_factor_mm_per_m)},
        )

    converter = _local_to_wgs84_transformer(global_center)
    features = []
    min_height_m = (1.0 / scale_factor_mm_per_m) if scale_factor_mm_per_m > 0 else 2.0
    mask = building_mask
    for _, row in gdf_buildings_local.head(limit * 3).iterrows():
        if len(features) >= limit:
            break
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            if mask is not None and not getattr(mask, "is_empty", True):
                geom = geom.intersection(mask)
            if geom is None or geom.is_empty:
                continue
            height_m = get_building_height(row, min_height_m) * float(height_scale_factor)
            props = {
                "layer": "buildings",
                "height_m": float(height_m),
                "height_mm": max(0.4, min(float(height_m) * scale_factor_mm_per_m, 45.0)),
            }
            if row.get("name") is not None:
                props["name"] = str(row.get("name"))[:80]
            wgs84_geom = transform_geometry(converter, geom)
            features.append({"type": "Feature", "properties": props, "geometry": mapping(wgs84_geom)})
        except Exception:
            continue

    if features:
        return {"type": "FeatureCollection", "features": features}
    return _geometry_feature_collection(
        building_mask,
        limit=limit,
        layer="buildings",
        global_center=global_center,
        props={"height_m": 4.0, "height_mm": max(0.4, 4.0 * scale_factor_mm_per_m)},
    )


def _build_preview_canonical_masks(
    *,
    request_ns: SimpleNamespace,
    source: SourceDataResult,
    zone: ZonePreparationResult,
    global_center: GlobalCenter,
    zone_prefix: str,
) -> SimpleNamespace:
    """Build the same canonical 2D source masks as full generation, without print/export audit.

    Full generation continues from these masks into runtime printability audits,
    grooves, terrain booleans, Blender/mesh export, and slicer checks. Preview
    deliberately stops here so it stays interactive while still showing the same
    road/building/water/park partition logic.
    """
    printer_profile = get_printer_profile_for_request(request_ns)
    building_geometry = prepare_building_geometry(
        gdf_buildings=source.gdf_buildings,
        global_center=global_center,
        zone_prefix=zone_prefix,
    )
    building_exclusion_for_roads = _expand_building_mask_for_roads(
        building_geometry.building_union_local,
        scale_factor=zone.scale_factor,
        clearance_mm=max(float(printer_profile.groove_side_clearance_mm), 0.25),
    )
    preclip_result = prepare_preclipped_geometry(
        gdf_buildings_local=building_geometry.gdf_buildings_local,
        building_geometries_for_flatten=building_geometry.building_geometries_for_flatten,
        gdf_water=source.gdf_water,
        global_center=global_center,
        zone_polygon_local=zone.zone_polygon_local,
        zone_prefix=zone_prefix,
    )
    road_gap_fill_mm_effective = 1.0
    road_geometry = prepare_road_geometry(
        G_roads=source.G_roads,
        scale_factor=zone.scale_factor,
        road_width_multiplier_effective=zone.road_width_multiplier_effective,
        min_printable_gap_mm=float(road_gap_fill_mm_effective),
        tiny_feature_threshold_mm=0.5,
        road_gap_fill_threshold_mm=float(road_gap_fill_mm_effective),
        enforce_printable_min_width=True,
        min_gap_fill_floor_mm=0.5,
        global_center=global_center,
        zone_polygon_local=zone.zone_polygon_local,
        zone_prefix=zone_prefix,
    )
    road_insert_source = road_geometry.merged_roads_geom_local
    if road_insert_source is None or getattr(road_insert_source, "is_empty", True):
        road_insert_source = road_geometry.merged_roads_geom_local_raw

    canonical_road_masks = _build_canonical_road_masks(
        road_insert_source=road_insert_source,
        building_union_local=building_geometry.building_union_local,
        scale_factor=zone.scale_factor,
        groove_clearance_mm=float(printer_profile.groove_side_clearance_mm),
        tiny_feature_threshold_mm=0.5,
        road_gap_fill_threshold_mm=float(road_gap_fill_mm_effective),
        zone_polygon_local=zone.zone_polygon_local,
        zone_prefix=zone_prefix,
    )
    road_insert = canonical_road_masks.road_insert_mask
    road_groove = canonical_road_masks.road_groove_mask or road_insert
    water_polygons = _prepare_water_polygons(
        preclip_result.gdf_water_local,
        road_polygons=road_groove,
        building_polygons=building_exclusion_for_roads,
        scale_factor=zone.scale_factor,
        fit_clearance_mm=float(printer_profile.groove_side_clearance_mm) * 0.5,
    )
    prepared_green = prepare_green_areas_for_processing(
        source.gdf_green,
        global_center=global_center,
        zone_polygon_local=zone.zone_polygon_local,
    )
    parks_result = process_green_areas(
        prepared_green,
        height_m=0.01,
        embed_m=0.0,
        terrain_provider=None,
        global_center=global_center,
        scale_factor=float(zone.scale_factor),
        zone_polygon_local=zone.zone_polygon_local,
        min_feature_mm=float(max(float(printer_profile.min_printable_feature_mm), float(MIN_LAND_WIDTH_MODEL_MM))),
        fit_clearance_mm=float(printer_profile.groove_side_clearance_mm) * 0.5,
        road_polygons=road_groove,
        water_polygons=water_polygons,
        building_polygons=building_exclusion_for_roads,
        return_result=True,
    )
    zone_polygon = zone.zone_polygon_local
    parks_final = _subtract_masks(
        parks_result.processed_polygons if parks_result is not None else None,
        road_insert,
        road_groove,
        water_polygons,
        building_exclusion_for_roads,
    )
    water_final = _subtract_masks(
        water_polygons,
        road_insert,
        road_groove,
        parks_final,
        building_exclusion_for_roads,
    )
    buildings_final = _subtract_masks(
        building_geometry.building_union_local,
        road_groove,
    )
    return SimpleNamespace(
        zone_polygon=zone_polygon,
        roads_final=_clip_mask_to_zone(road_insert, zone_polygon),
        road_groove_mask=_clip_mask_to_zone(road_groove, zone_polygon),
        parks_final=_clip_mask_to_zone(parks_final, zone_polygon),
        water_final=_clip_mask_to_zone(water_final, zone_polygon),
        buildings_footprints=_clip_mask_to_zone(buildings_final, zone_polygon),
        building_geometry=building_geometry,
        road_geometry=road_geometry,
        preclip_result=preclip_result,
    )


def _fetch_preview_source_data(
    *,
    request_ns: SimpleNamespace,
    global_center: GlobalCenter,
    zone_prefix: str,
) -> SourceDataResult:
    print(
        f"[DEBUG] {zone_prefix} Loading preview source data: "
        f"north={request_ns.north}, south={request_ns.south}, east={request_ns.east}, west={request_ns.west}"
    )
    # Full generation uses a topological OSMnx graph and a generous road context.
    # For browser preview we still run the same canonical geometry processors, but
    # feed them clipped OSM feature lines so Overpass cannot hold the request for
    # minutes before the user sees a model.
    context_deg = 0.0008
    preview_bbox = (
        float(request_ns.west) - context_deg,
        float(request_ns.south) - context_deg,
        float(request_ns.east) + context_deg,
        float(request_ns.north) + context_deg,
    )
    target_bbox = box(
        float(request_ns.west),
        float(request_ns.south),
        float(request_ns.east),
        float(request_ns.north),
    )

    def fetch_features(label: str, tags: dict[str, Any]) -> gpd.GeoDataFrame:
        original_timeout = int(getattr(ox.settings, "timeout", 180) or 180)
        try:
            ox.settings.timeout = min(original_timeout, 25)
            try:
                gdf = ox.features_from_bbox(bbox=preview_bbox, tags=tags)
            except TypeError:
                gdf = ox.features_from_bbox(
                    preview_bbox[0],
                    preview_bbox[1],
                    preview_bbox[2],
                    preview_bbox[3],
                    tags=tags,
                )
        except Exception as exc:
            print(f"[WARN] {zone_prefix} Preview {label} fetch failed: {exc}")
            return gpd.GeoDataFrame()
        finally:
            ox.settings.timeout = original_timeout

        if gdf is None or gdf.empty:
            return gpd.GeoDataFrame()
        gdf = gdf[gdf.geometry.notna()].copy()
        if gdf.empty:
            return gpd.GeoDataFrame()
        try:
            gdf = gdf[gdf.geometry.intersects(target_bbox)].copy()
        except Exception:
            pass
        try:
            if global_center is not None and getattr(global_center, "utm_crs", None) is not None:
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326", allow_override=True)
                gdf = gdf.to_crs(global_center.utm_crs)
        except Exception as exc:
            print(f"[WARN] {zone_prefix} Preview {label} projection skipped: {exc}")
        print(f"[DEBUG] {zone_prefix} Preview {label} features: {len(gdf)}")
        return gdf

    def get_buildings():
        return fetch_features("buildings", {"building": True})

    def get_water():
        return fetch_features(
            "water",
            {
                "natural": "water",
                "water": True,
                "waterway": "riverbank",
                "landuse": "reservoir",
            },
        )

    def get_roads():
        roads = fetch_features("roads", {"highway": True})
        if roads is None or roads.empty:
            return gpd.GeoDataFrame()
        try:
            roads = roads[
                roads.geometry.geom_type.isin(["LineString", "MultiLineString"])
            ].copy()
        except Exception:
            pass
        if "highway" in roads.columns:
            roads["_normalized_highway"] = roads["highway"].apply(normalize_drivable_highway_tag)
            roads = roads[roads["_normalized_highway"].notna()].copy()
        print(f"[DEBUG] {zone_prefix} Preview road centerlines kept: {len(roads)}")
        return roads

    def get_extras():
        return fetch_extras(
            float(request_ns.north),
            float(request_ns.south),
            float(request_ns.east),
            float(request_ns.west),
            target_crs=global_center.utm_crs if global_center else None,
        )

    gdf_buildings = gpd.GeoDataFrame()
    gdf_water = gpd.GeoDataFrame()
    gdf_green = gpd.GeoDataFrame()
    G_roads = None
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    futures = {
        executor.submit(get_buildings): "buildings",
        executor.submit(get_water): "water",
        executor.submit(get_roads): "roads",
        executor.submit(get_extras): "parks",
    }
    done, pending = concurrent.futures.wait(futures, timeout=32)
    for future in done:
        label = futures[future]
        try:
            value = future.result()
        except Exception as exc:
            print(f"[WARN] {zone_prefix} Preview {label} result failed: {exc}")
            continue
        if label == "buildings":
            gdf_buildings = value
        elif label == "water":
            gdf_water = value
        elif label == "roads":
            G_roads = value
        elif label == "parks":
            gdf_green = value
    for future in pending:
        label = futures[future]
        future.cancel()
        print(f"[WARN] {zone_prefix} Preview {label} fetch timed out; continuing without this layer")
    executor.shutdown(wait=False, cancel_futures=True)

    road_count = 0
    if G_roads is not None:
        if hasattr(G_roads, "edges"):
            road_count = len(G_roads.edges)
        elif hasattr(G_roads, "__len__"):
            road_count = len(G_roads)
    print(
        f"[DEBUG] {zone_prefix} Preview source loaded: "
        f"{len(gdf_buildings) if gdf_buildings is not None and not gdf_buildings.empty else 0} buildings, "
        f"{len(gdf_water) if gdf_water is not None and not gdf_water.empty else 0} water, "
        f"{road_count} roads"
    )
    return SourceDataResult(
        gdf_buildings=gdf_buildings,
        gdf_water=gdf_water,
        G_roads=G_roads,
        gdf_green=gdf_green,
    )


def _terrain_heightfield_layer(
    *,
    request_ns: SimpleNamespace,
    latlon_bbox: tuple[float, float, float, float],
    source: SourceDataResult,
    zone: ZonePreparationResult,
    global_center: GlobalCenter,
    building_geometries_for_flatten: Any,
    road_groove_mask: Optional[BaseGeometry],
    zone_prefix: str,
) -> dict[str, Any]:
    if not bool(getattr(request_ns, "terrain_enabled", True)):
        return {"enabled": False}

    try:
        source_crs = resolve_generation_source_crs(
            gdf_buildings=source.gdf_buildings,
            G_roads=source.G_roads,
            global_center=global_center,
            allow_global_center_fallback=True,
            zone_prefix=zone_prefix,
        )
        scale_factor = float(zone.scale_factor or 0.0)
        base_thickness = (
            float(getattr(request_ns, "terrain_base_thickness_mm", 2.0) or 2.0) / scale_factor
            if scale_factor > 0
            else 5.0
        )
        preview_resolution = int(
            max(
                24,
                min(
                    56,
                    float(getattr(request_ns, "preview_terrain_resolution", 42) or 42),
                    float(getattr(request_ns, "terrain_resolution", 180) or 180),
                ),
            )
        )
        terrain_mesh, terrain_provider = create_terrain_mesh(
            zone.bbox_meters,
            z_scale=float(getattr(request_ns, "terrain_z_scale", 0.5) or 0.5),
            resolution=preview_resolution,
            latlon_bbox=latlon_bbox,
            source_crs=source_crs,
            terrarium_zoom=int(getattr(request_ns, "terrarium_zoom", 15) or 15),
            elevation_ref_m=getattr(request_ns, "elevation_ref_m", None),
            baseline_offset_m=float(getattr(request_ns, "baseline_offset_m", 0.0) or 0.0),
            base_thickness=base_thickness,
            flatten_buildings=bool(getattr(request_ns, "flatten_buildings_on_terrain", True)),
            building_geometries=building_geometries_for_flatten,
            flatten_roads=False,
            road_geometries=road_groove_mask,
            smoothing_sigma=float(getattr(request_ns, "terrain_smoothing_sigma", 0.0) or 0.0),
            water_geometries=None,
            water_depth_m=0.0,
            global_center=global_center,
            bbox_is_local=True,
            subdivide=False,
            subdivide_levels=0,
            zone_polygon=zone.zone_polygon_local,
            grid_step_m=None,
            road_polygons_for_cutting=None,
        )
        if terrain_provider is None:
            return {"enabled": True, "heightfield": None, "source": "terrain_provider_unavailable"}

        z_grid = getattr(terrain_provider, "z_grid", None)
        x_axis = getattr(terrain_provider, "x_axis", None)
        y_axis = getattr(terrain_provider, "y_axis", None)
        if z_grid is None or x_axis is None or y_axis is None:
            return {"enabled": True, "heightfield": None, "source": "terrain_grid_unavailable"}

        def rounded_list(values: Any, digits: int = 3) -> list[float]:
            return [round(float(v), digits) for v in list(values)]

        z_values = [
            [round(float(v), 3) for v in row]
            for row in getattr(z_grid, "tolist", lambda: z_grid)()
        ]
        mesh_vertices = 0
        mesh_faces = 0
        if terrain_mesh is not None:
            try:
                mesh_vertices = int(len(terrain_mesh.vertices))
                mesh_faces = int(len(terrain_mesh.faces))
            except Exception:
                mesh_vertices = 0
                mesh_faces = 0

        return {
            "enabled": True,
            "heightfield": {
                "x": rounded_list(x_axis),
                "y": rounded_list(y_axis),
                "z": z_values,
                "z_min_m": round(float(getattr(terrain_provider, "min_z", 0.0)), 4),
                "z_max_m": round(float(getattr(terrain_provider, "max_z", 0.0)), 4),
                "mesh_vertices": mesh_vertices,
                "mesh_faces": mesh_faces,
            },
            "source": "create_terrain_mesh_provider_preview",
        }
    except Exception as exc:
        print(f"[WARN] {zone_prefix} Preview terrain heightfield failed: {exc}")
        return {"enabled": True, "heightfield": None, "source": "terrain_preview_failed", "error": str(exc)}


def _build_canonical_preview(
    *,
    preview_id: str,
    bounds: dict[str, float],
    polygon_geojson: Optional[dict[str, Any]],
    include_terrain: bool,
    include_roads: bool,
    include_buildings: bool,
    include_water: bool,
    include_parks: bool,
    model_logic: dict[str, Any],
    request_ns: SimpleNamespace,
    started: float,
) -> dict[str, Any]:
    latlon_bbox = (
        float(bounds["north"]),
        float(bounds["south"]),
        float(bounds["east"]),
        float(bounds["west"]),
    )
    center_lat = (float(bounds["north"]) + float(bounds["south"])) / 2.0
    center_lon = (float(bounds["east"]) + float(bounds["west"])) / 2.0
    global_center = GlobalCenter(center_lat, center_lon)
    zone = _prepare_zone_stage(
        request=request_ns,
        global_center=global_center,
        zone_polygon_coords=_zone_polygon_coords_from_geojson(polygon_geojson),
        grid_bbox_latlon=latlon_bbox,
        zone_row=None,
        zone_col=None,
        hex_size_m=getattr(request_ns, "hex_size_m", 300.0),
        zone_prefix="[preview] ",
    )
    if zone.zone_polygon_local is None:
        zone.zone_polygon_local = box(*zone.bbox_meters)
    source_started = time.perf_counter()
    source = _fetch_preview_source_data(
        request_ns=request_ns,
        global_center=global_center,
        zone_prefix="[preview] ",
    )
    bundle = _build_preview_canonical_masks(
        request_ns=request_ns,
        source=source,
        zone=zone,
        global_center=global_center,
        zone_prefix="[preview] ",
    )
    display_zone = zone.zone_polygon_local
    if display_zone is None or getattr(display_zone, "is_empty", True):
        display_zone = box(*zone.bbox_meters)
    roads_final = _clip_mask_to_zone(getattr(bundle, "roads_final", None), display_zone)
    parks_final = _clip_mask_to_zone(getattr(bundle, "parks_final", None), display_zone)
    water_final = _clip_mask_to_zone(getattr(bundle, "water_final", None), display_zone)
    buildings_footprints = _clip_mask_to_zone(getattr(bundle, "buildings_footprints", None), display_zone)
    model_logic = {
        **model_logic,
        "scale_factor_mm_per_m": float(zone.scale_factor),
        "model_width_mm": float(zone.bbox_meters[2] - zone.bbox_meters[0]) * float(zone.scale_factor),
        "model_height_mm": float(zone.bbox_meters[3] - zone.bbox_meters[1]) * float(zone.scale_factor),
        "preview_source": "canonical_fast_3d_preview",
        "canonical_note": "same canonical mask processors as full 3D generation with preview-scoped OSM fetch; displayed before terrain booleans, Blender and export",
        "canonical_elapsed_ms": int((time.perf_counter() - source_started) * 1000),
    }
    building_geometry = getattr(bundle, "building_geometry", None)
    gdf_buildings_local = getattr(building_geometry, "gdf_buildings_local", None)
    terrain_layer = _terrain_heightfield_layer(
        request_ns=request_ns,
        latlon_bbox=latlon_bbox,
        source=source,
        zone=zone,
        global_center=global_center,
        building_geometries_for_flatten=getattr(building_geometry, "building_geometries_for_flatten", None),
        road_groove_mask=getattr(bundle, "road_groove_mask", None),
        zone_prefix="[preview] ",
    ) if include_terrain else {"enabled": False}

    result = {
        "preview_id": preview_id,
        "cached": False,
        "bounds": bounds,
        "center": {"lat": center_lat, "lng": center_lon},
        "selection": mapping(transform_geometry(_local_to_wgs84_transformer(global_center), display_zone))
        if display_zone is not None and not getattr(display_zone, "is_empty", True)
        else mapping(_selection_geometry(bounds, polygon_geojson)),
        "model_logic": model_logic,
        "layers": {
            "terrain": terrain_layer,
            "buildings": _canonical_building_collection(
                gdf_buildings_local=gdf_buildings_local if include_buildings else None,
                building_mask=buildings_footprints if include_buildings else None,
                global_center=global_center,
                scale_factor_mm_per_m=float(zone.scale_factor),
                height_scale_factor=float(getattr(request_ns, "buildings_height_scale", 1.0) or 1.0),
                limit=MAX_FEATURES["buildings"],
            ),
            "roads": _geometry_feature_collection(
                roads_final if include_roads else None,
                limit=MAX_FEATURES["roads"],
                layer="roads",
                global_center=global_center,
            ),
            "water": _geometry_feature_collection(
                water_final if include_water else None,
                limit=MAX_FEATURES["water"],
                layer="water",
                global_center=global_center,
            ),
            "parks": _geometry_feature_collection(
                parks_final if include_parks else None,
                limit=MAX_FEATURES["parks"],
                layer="parks",
                global_center=global_center,
            ),
        },
        "metrics": {
            "buildings": int(len(gdf_buildings_local)) if gdf_buildings_local is not None and not gdf_buildings_local.empty else 0,
            "roads": int(len(source.G_roads.edges)) if hasattr(source.G_roads, "edges") else int(len(source.G_roads)) if source.G_roads is not None and hasattr(source.G_roads, "__len__") else 0,
            "water": int(len(source.gdf_water)) if source.gdf_water is not None and not source.gdf_water.empty else 0,
            "parks": int(len(source.gdf_green)) if source.gdf_green is not None and not source.gdf_green.empty else 0,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        },
    }
    return result


def _pending_preview_response(
    *,
    preview_id: str,
    bounds: dict[str, float],
    polygon_geojson: Optional[dict[str, Any]],
    model_logic: dict[str, Any],
    status: str = "processing",
    message: str = "Точне preview готується з тієї ж pipeline-логіки, що й 3D-модель.",
) -> dict[str, Any]:
    center = {
        "lat": (float(bounds["north"]) + float(bounds["south"])) / 2.0,
        "lng": (float(bounds["east"]) + float(bounds["west"])) / 2.0,
    }
    return {
        "preview_id": preview_id,
        "preview_status": status,
        "cached": False,
        "bounds": bounds,
        "center": center,
        "selection": mapping(_selection_geometry(bounds, polygon_geojson)),
        "model_logic": {
            **model_logic,
            "preview_source": "full_generation_pipeline_pending",
            "preview_message": message,
        },
        "layers": {
            "terrain": {"enabled": True},
            "buildings": _empty_collection(),
            "roads": _empty_collection(),
            "water": _empty_collection(),
            "parks": _empty_collection(),
        },
        "metrics": {"buildings": 0, "roads": 0, "water": 0, "parks": 0, "elapsed_ms": 0},
    }


def _read_preview_job_status(preview_id: str) -> dict[str, Any] | None:
    status_file = PREVIEW_JOBS_DIR / f"{preview_id}.status.json"
    try:
        if status_file.exists():
            return json.loads(status_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _pid_is_running(pid: Any) -> bool:
    try:
        pid_int = int(pid)
        if pid_int <= 0:
            return False
        os.kill(pid_int, 0)
        return True
    except Exception:
        return False


def _active_preview_worker_count(now: float) -> int:
    active = 0
    for status_file in PREVIEW_JOBS_DIR.glob("*.status.json"):
        try:
            status = json.loads(status_file.read_text(encoding="utf-8"))
            if status.get("status") != "running":
                continue
            if not _pid_is_running(status.get("pid")):
                continue
            started_at = float(status.get("started_at", now))
            if now - started_at < PREVIEW_WORKER_TIMEOUT_SECONDS:
                active += 1
        except Exception:
            continue
    return active


def _start_preview_worker_if_needed(
    *,
    preview_id: str,
    worker_payload: dict[str, Any],
) -> dict[str, Any]:
    status_file = PREVIEW_JOBS_DIR / f"{preview_id}.status.json"
    input_file = PREVIEW_JOBS_DIR / f"{preview_id}.input.json"
    now = time.time()
    try:
        if status_file.exists():
            status = json.loads(status_file.read_text(encoding="utf-8"))
            state = status.get("status")
            started_at = float(status.get("started_at", now))
            finished_at = float(status.get("finished_at", now))
            if state == "running" and now - started_at < PREVIEW_WORKER_TIMEOUT_SECONDS:
                if _pid_is_running(status.get("pid")):
                    return status
            if state == "queued" and _active_preview_worker_count(now) >= PREVIEW_MAX_ACTIVE_WORKERS:
                return status
            if state == "failed" and now - finished_at < PREVIEW_FAILED_RETRY_DELAY_SECONDS:
                return status
    except Exception:
        pass

    if _active_preview_worker_count(now) >= PREVIEW_MAX_ACTIVE_WORKERS:
        queued_status = {
            "status": "queued",
            "queued_at": now,
            "preview_id": preview_id,
            "message": "Інше точне preview ще рахується. Цей запит запуститься автоматично.",
        }
        return queued_status

    input_file.write_text(json.dumps(worker_payload, ensure_ascii=False), encoding="utf-8")
    backend_root = Path(__file__).resolve().parents[1]
    stdout_file = PREVIEW_JOBS_DIR / f"{preview_id}.worker.out.log"
    stderr_file = PREVIEW_JOBS_DIR / f"{preview_id}.worker.err.log"
    with stdout_file.open("ab") as stdout_handle, stderr_file.open("ab") as stderr_handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "services.preview_worker", str(input_file)],
            cwd=str(backend_root),
            stdout=stdout_handle,
            stderr=stderr_handle,
            close_fds=(os.name != "nt"),
            start_new_session=(os.name != "nt"),
        )
    running_status = {
        "status": "running",
        "started_at": now,
        "preview_id": preview_id,
        "pid": process.pid,
        "stdout_log": str(stdout_file),
        "stderr_log": str(stderr_file),
    }
    status_file.write_text(json.dumps(running_status, ensure_ascii=False), encoding="utf-8")
    return running_status


def _static_file_url(path_str: str | None) -> str | None:
    if not path_str:
        return None
    return f"/files/{Path(path_str).name}"


def _build_full_pipeline_preview(
    *,
    preview_id: str,
    bounds: dict[str, float],
    polygon_geojson: Optional[dict[str, Any]],
    include_terrain: bool,
    include_roads: bool,
    include_buildings: bool,
    include_water: bool,
    include_parks: bool,
    model_logic: dict[str, Any],
    request_ns: SimpleNamespace,
    started: float,
) -> dict[str, Any]:
    request_ns.preview_include_base = bool(include_terrain)
    request_ns.preview_include_roads = bool(include_roads)
    request_ns.preview_include_buildings = bool(include_buildings)
    request_ns.preview_include_water = bool(include_water)
    request_ns.preview_include_parks = bool(include_parks)
    request_ns.include_parks = bool(include_parks)
    request_ns.terrain_enabled = bool(include_terrain)
    request_ns.export_format = "3mf"
    request_ns.skip_road_hole_audit = True
    request_ns.skip_layer_overlap_audit = True
    request_ns.skip_canonical_printability_audit = True

    task = GenerationTask(task_id=preview_id, request=request_ns)
    runtime_context = prepare_generation_runtime_context(
        request=request_ns,
        zone_prefix="[preview-full] ",
    )
    latlon_bbox = (
        float(bounds["north"]),
        float(bounds["south"]),
        float(bounds["east"]),
        float(bounds["west"]),
    )
    workflow_result = run_full_generation_pipeline(
        task=task,
        request=request_ns,
        task_id=preview_id,
        output_dir=PREVIEW_OUTPUT_DIR,
        global_center=runtime_context.global_center,
        latlon_bbox=runtime_context.latlon_bbox,
        zone_polygon_coords=_zone_polygon_coords_from_geojson(polygon_geojson),
        grid_bbox_latlon=latlon_bbox,
        zone_row=None,
        zone_col=None,
        hex_size_m=getattr(request_ns, "hex_size_m", 300.0),
        zone_prefix="[preview-full] ",
        require_groove_success=False,
        require_print_acceptance=False,
        apply_grooves=True,
    )
    output_files = getattr(task, "output_files", {}) or {}
    primary_url = _static_file_url(str(workflow_result.output_file_abs))
    stl_url = _static_file_url(output_files.get("stl"))
    preview_3mf_url = _static_file_url(output_files.get("preview_3mf"))
    result = {
        "preview_id": preview_id,
        "preview_status": "ready",
        "cached": False,
        "bounds": bounds,
        "center": {
            "lat": (float(bounds["north"]) + float(bounds["south"])) / 2.0,
            "lng": (float(bounds["east"]) + float(bounds["west"])) / 2.0,
        },
        "selection": mapping(_selection_geometry(bounds, polygon_geojson)),
        "model_file_url": stl_url or primary_url,
        "preview_stl": stl_url,
        "preview_3mf": preview_3mf_url,
        "download_url": primary_url,
        "download_url_3mf": _static_file_url(output_files.get("3mf")) or primary_url,
        "download_url_stl": stl_url,
        "task_outputs": {key: _static_file_url(value) for key, value in output_files.items()},
        "model_logic": {
            **model_logic,
            "preview_source": "run_full_generation_pipeline_with_grooves",
            "canonical_note": "preview is generated by the same full 3D pipeline as production model generation; groove/paz boolean cutting is attempted and export continues if Blender rejects an unsafe groove result",
            "pipeline_task_id": preview_id,
            "pipeline_elapsed_ms": int((time.perf_counter() - started) * 1000),
            "task_message": task.message,
        },
        "layers": {
            "terrain": {"enabled": include_terrain},
            "buildings": _empty_collection(),
            "roads": _empty_collection(),
            "water": _empty_collection(),
            "parks": _empty_collection(),
        },
        "metrics": {
            "buildings": 0,
            "roads": 0,
            "water": 0,
            "parks": 0,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        },
    }
    return result


def build_preview_cache_from_worker_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = _build_full_pipeline_preview(
        preview_id=str(payload["preview_id"]),
        bounds=payload["bounds"],
        polygon_geojson=payload.get("polygon_geojson"),
        include_terrain=bool(payload.get("include_terrain", True)),
        include_roads=bool(payload.get("include_roads", True)),
        include_buildings=bool(payload.get("include_buildings", True)),
        include_water=bool(payload.get("include_water", True)),
        include_parks=bool(payload.get("include_parks", True)),
        model_logic=payload["model_logic"],
        request_ns=SimpleNamespace(**payload["request"]),
        started=time.perf_counter(),
    )
    result["preview_status"] = "ready"
    cache_file = Path(payload["cache_file"])
    cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def build_fast_preview(
    *,
    bounds: dict[str, float],
    polygon_geojson: Optional[dict[str, Any]] = None,
    include_terrain: bool = True,
    include_roads: bool = True,
    include_buildings: bool = True,
    include_water: bool = True,
    include_parks: bool = True,
    road_width_multiplier: float = 0.8,
    building_min_height: float = 5.0,
    building_height_multiplier: float = 1.8,
    model_size_mm: float = 180.0,
    terrain_z_scale: float = 3.0,
    terrain_resolution: int = 350,
    road_height_mm: float = 0.5,
    road_embed_mm: float = 0.3,
    building_foundation_mm: float = 0.6,
    building_embed_mm: float = 0.2,
    water_depth: float = 1.2,
    parks_height_mm: float = 0.6,
    parks_embed_mm: float = 1.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    center = {
        "lat": (float(bounds["north"]) + float(bounds["south"])) / 2,
        "lng": (float(bounds["east"]) + float(bounds["west"])) / 2,
    }
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lng = 111_320.0 * math.cos((center["lat"] * math.pi) / 180.0)
    width_m = max(1.0, abs(float(bounds["east"]) - float(bounds["west"])) * meters_per_deg_lng)
    height_m = max(1.0, abs(float(bounds["north"]) - float(bounds["south"])) * meters_per_deg_lat)
    scale_factor_mm_per_m = float(model_size_mm or 180.0) / max(width_m, height_m)
    terrain_base_thickness_mm = max(0.3, float(road_embed_mm or 0.3), float(water_depth or 1.2), float(parks_embed_mm or 1.0)) + 0.5
    model_logic = {
        "road_width_multiplier": road_width_multiplier,
        "road_height_mm": road_height_mm,
        "road_embed_mm": road_embed_mm,
        "building_min_height": building_min_height,
        "building_height_multiplier": building_height_multiplier,
        "building_foundation_mm": building_foundation_mm,
        "building_embed_mm": building_embed_mm,
        "water_depth": water_depth,
        "parks_height_mm": parks_height_mm,
        "parks_embed_mm": parks_embed_mm,
        "model_size_mm": model_size_mm,
        "terrain_z_scale": terrain_z_scale,
        "terrain_resolution": terrain_resolution,
        "scale_factor_mm_per_m": scale_factor_mm_per_m,
        "model_width_mm": width_m * scale_factor_mm_per_m,
        "model_height_mm": height_m * scale_factor_mm_per_m,
        "terrain_base_thickness_mm": terrain_base_thickness_mm,
    }
    request_ns = _request_namespace(
        bounds=bounds,
        road_width_multiplier=road_width_multiplier,
        building_min_height=building_min_height,
        building_height_multiplier=building_height_multiplier,
        model_size_mm=model_size_mm,
        terrain_z_scale=terrain_z_scale,
        terrain_resolution=terrain_resolution,
        terrain_base_thickness_mm=terrain_base_thickness_mm,
        road_height_mm=road_height_mm,
        road_embed_mm=road_embed_mm,
        building_foundation_mm=building_foundation_mm,
        building_embed_mm=building_embed_mm,
        water_depth=water_depth,
        parks_height_mm=parks_height_mm,
        parks_embed_mm=parks_embed_mm,
    )
    payload = {
        "v": 23,
        "mode": "full_generation_pipeline_with_grooves_preview",
        "bounds": bounds,
        "polygon_geojson": polygon_geojson,
        "model_logic": model_logic,
        "layers": {
            "terrain": include_terrain,
            "roads": include_roads,
            "buildings": include_buildings,
            "water": include_water,
            "parks": include_parks,
        },
    }
    preview_id = _cache_key(payload)
    cache_file = PREVIEW_CACHE_DIR / f"{preview_id}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            cached["cached"] = True
            cached["preview_status"] = "ready"
            return cached
        except Exception:
            pass

    status = _read_preview_job_status(preview_id)
    if status and status.get("status") == "failed":
        return _pending_preview_response(
            preview_id=preview_id,
            bounds=bounds,
            polygon_geojson=polygon_geojson,
            model_logic=model_logic,
            status="failed",
            message=str(status.get("error") or "Повна генерація preview з пазами завершилась помилкою."),
        )

    worker_payload = {
        "preview_id": preview_id,
        "bounds": bounds,
        "polygon_geojson": polygon_geojson,
        "include_terrain": include_terrain,
        "include_roads": include_roads,
        "include_buildings": include_buildings,
        "include_water": include_water,
        "include_parks": include_parks,
        "model_logic": model_logic,
        "request": vars(request_ns),
        "cache_file": str(cache_file),
    }
    worker_status = _start_preview_worker_if_needed(preview_id=preview_id, worker_payload=worker_payload)
    return _pending_preview_response(
        preview_id=preview_id,
        bounds=bounds,
        polygon_geojson=polygon_geojson,
        model_logic=model_logic,
        status="processing",
        message=str(
            worker_status.get("message")
            or "Генеруємо повну 3D-модель тим самим pipeline, що й основна генерація: рельєф, шари, пази та export. Це може зайняти кілька хвилин."
        ),
    )

    selection = _selection_geometry(bounds, polygon_geojson)
    all_features = _features_from_bbox(
        bounds,
        {
            "building": True,
            "highway": True,
            "natural": ["water", "wood"],
            "waterway": True,
            "leisure": "park",
            "landuse": ["grass", "recreation_ground", "forest"],
        },
    )

    buildings = _clip_to_selection(
        _filter_by_tag(all_features, "building") if include_buildings else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
        bounds,
        polygon_geojson,
    )
    raw_roads = _filter_by_tag(all_features, "highway") if include_roads else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    roads = _build_preview_road_mask(raw_roads, bounds, polygon_geojson, road_width_multiplier) if include_roads else raw_roads
    if include_roads and include_buildings:
        roads = _clip_to_selection(_subtract_buildings_from_roads(roads, buildings), bounds, polygon_geojson)
    water = _clip_to_selection(
        _merge_frames(_filter_by_tag(all_features, "natural", "water"), _filter_by_tag(all_features, "waterway")) if include_water else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
        bounds,
        polygon_geojson,
    )
    parks = _clip_to_selection(
        _merge_frames(
            _filter_by_tag(all_features, "leisure", "park"),
            _filter_by_tag(all_features, "landuse", ["grass", "recreation_ground", "forest"]),
            _filter_by_tag(all_features, "natural", "wood"),
        ) if include_parks else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
        bounds,
        polygon_geojson,
    )

    result = {
        "preview_id": preview_id,
        "cached": False,
        "bounds": bounds,
        "center": center,
        "selection": mapping(selection),
        "model_logic": {**payload["model_logic"], "preview_source": "legacy_osm_fallback"},
        "layers": {
            "terrain": {"enabled": include_terrain},
            "buildings": _feature_collection(
                buildings,
                MAX_FEATURES["buildings"],
                0.00001,
                "buildings",
                building_min_height=building_min_height,
                building_height_multiplier=building_height_multiplier,
                scale_factor_mm_per_m=scale_factor_mm_per_m,
            ),
            "roads": _feature_collection(roads, MAX_FEATURES["roads"], 0.000004, "roads"),
            "water": _feature_collection(water, MAX_FEATURES["water"], 0.00001, "water"),
            "parks": _feature_collection(parks, MAX_FEATURES["parks"], 0.00001, "parks"),
        },
        "metrics": {
            "buildings": int(len(buildings)),
            "roads": int(len(raw_roads)),
            "water": int(len(water)),
            "parks": int(len(parks)),
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        },
    }
    cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result
