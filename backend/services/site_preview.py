from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry

from services.road_processor import normalize_drivable_highway_tag


PREVIEW_CACHE_DIR = Path(os.getenv("PREVIEW_CACHE_DIR", "cache/site_previews"))
PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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


def _road_buffer_radius_m(row: Any) -> float:
    highway = normalize_drivable_highway_tag(row.get("highway"))
    fallback = ROAD_WIDTH_MAP_M.get(highway or "", 1.6)
    explicit = _numeric_width(row.get("width"))
    width = explicit if explicit is not None else fallback
    if explicit is not None:
        width = min(max(explicit, 0.8), max(fallback * 1.15, fallback + 0.35))
    return max(width / 2.0, 0.35)


def _build_preview_road_mask(
    roads: gpd.GeoDataFrame,
    bounds: dict[str, float],
    polygon_geojson: Optional[dict[str, Any]],
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
                mask = geom.buffer(_road_buffer_radius_m(row), cap_style=2, join_style=1, resolution=3)
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


def _height_from_row(row: Any) -> float:
    raw = row.get("height")
    levels = row.get("building:levels")
    height = _numeric_width(raw)
    if height is not None:
        return max(3.0, min(height, 80.0))
    try:
        if levels is not None and not (isinstance(levels, float) and math.isnan(levels)):
            return max(3.0, min(float(levels) * 3.0, 60.0))
    except Exception:
        pass
    return 8.0


def _feature_collection(gdf: gpd.GeoDataFrame, limit: int, simplify: float, layer: str) -> dict[str, Any]:
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
                props["height_m"] = _height_from_row(row)
            if row.get("name") is not None:
                props["name"] = str(row.get("name"))[:80]
            features.append({"type": "Feature", "properties": props, "geometry": mapping(geom)})
        except Exception:
            continue
    return {"type": "FeatureCollection", "features": features}


def build_fast_preview(
    *,
    bounds: dict[str, float],
    polygon_geojson: Optional[dict[str, Any]] = None,
    include_terrain: bool = True,
    include_roads: bool = True,
    include_buildings: bool = True,
    include_water: bool = True,
    include_parks: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    payload = {
        "v": 7,
        "bounds": bounds,
        "polygon_geojson": polygon_geojson,
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
            return cached
        except Exception:
            pass

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
    roads = _build_preview_road_mask(raw_roads, bounds, polygon_geojson) if include_roads else raw_roads
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

    center = {
        "lat": (float(bounds["north"]) + float(bounds["south"])) / 2,
        "lng": (float(bounds["east"]) + float(bounds["west"])) / 2,
    }
    result = {
        "preview_id": preview_id,
        "cached": False,
        "bounds": bounds,
        "center": center,
        "selection": mapping(selection),
        "layers": {
            "terrain": {"enabled": include_terrain},
            "buildings": _feature_collection(buildings, MAX_FEATURES["buildings"], 0.00001, "buildings"),
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
