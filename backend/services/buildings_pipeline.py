from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from services.building_supports import (
    build_building_supports,
    concatenate_meshes,
    merge_building_and_support_meshes,
)
from services.building_processor import process_buildings
from services.detail_layer_utils import MICRO_REGION_THRESHOLD_MM, model_mm_to_world_m
from services.printable_3d_validator import validate_building_mesh
from services.processing_results import BuildingLayerResult


def _is_building_part_row(row: Any) -> bool:
    try:
        if bool(row.get("__is_building_part", False)):
            return True
    except Exception:
        pass
    try:
        value = row.get("building:part", None)
    except Exception:
        value = None
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in ("", "0", "false", "no", "none", "nan")


def _clean_building_geometry(geometry: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return None
    try:
        clean = geometry.buffer(0)
        if clean is not None and not getattr(clean, "is_empty", True):
            return clean
    except Exception:
        pass
    return geometry


def split_building_parts_from_parent_footprints(
    buildings: GeoDataFrame,
    *,
    min_part_overlap_ratio: float = 0.5,
    full_coverage_ratio: float = 0.985,
    min_remainder_area_m2: float = 1.0,
) -> GeoDataFrame:
    """Let OSM building:part geometry win over the parent building footprint.

    OSM Simple 3D Buildings often stores one coarse `building=yes` outline plus
    several `building:part` polygons with their own heights. If we extrude the
    parent as-is, a tower height on the parent turns the whole block into one
    tall slab. This normalizer keeps part rows, subtracts their union from the
    parent outline, and drops the parent when parts cover it fully.
    """
    if buildings is None or buildings.empty or "geometry" not in buildings.columns:
        return buildings

    part_flags = buildings.apply(_is_building_part_row, axis=1)
    if not bool(part_flags.any()):
        return buildings

    gdf = buildings.copy()
    parts = gdf[part_flags & gdf.geometry.notna()].copy()
    parents = gdf[~part_flags & gdf.geometry.notna()].copy()
    passthrough = gdf[gdf.geometry.isna()].copy()
    if parents.empty or parts.empty:
        return gdf

    parts["geometry"] = parts.geometry.apply(_clean_building_geometry)
    parts = parts[parts.geometry.notna() & ~parts.geometry.is_empty].copy()
    parents["geometry"] = parents.geometry.apply(_clean_building_geometry)
    parents = parents[parents.geometry.notna() & ~parents.geometry.is_empty].copy()
    if parents.empty or parts.empty:
        return gdf

    try:
        sindex = parts.sindex
    except Exception:
        sindex = None

    output_rows = []
    parent_split_count = 0
    parent_drop_count = 0
    for parent_idx, parent_row in parents.iterrows():
        parent_geom = _clean_building_geometry(parent_row.geometry)
        if parent_geom is None or getattr(parent_geom, "is_empty", True):
            continue
        try:
            parent_area = float(parent_geom.area)
        except Exception:
            parent_area = 0.0
        if parent_area <= 0.0:
            output_rows.append(parent_row)
            continue

        try:
            candidate_indices = list(sindex.query(parent_geom, predicate="intersects")) if sindex is not None else []
            candidate_parts = parts.iloc[candidate_indices] if candidate_indices else parts[parts.intersects(parent_geom)]
        except Exception:
            candidate_parts = parts[parts.intersects(parent_geom)]

        cutters = []
        for _, part_row in candidate_parts.iterrows():
            part_geom = _clean_building_geometry(part_row.geometry)
            if part_geom is None or getattr(part_geom, "is_empty", True):
                continue
            try:
                part_area = float(part_geom.area)
                if part_area <= 0.0:
                    continue
                inter = parent_geom.intersection(part_geom)
                inter_area = float(inter.area) if inter is not None and not getattr(inter, "is_empty", True) else 0.0
            except Exception:
                continue
            if inter_area <= 0.0:
                continue
            if (inter_area / max(part_area, 1e-9)) >= float(min_part_overlap_ratio):
                cutters.append(inter)

        if not cutters:
            output_rows.append(parent_row)
            continue

        try:
            cutter_union = unary_union(cutters).buffer(0)
            coverage_ratio = float(parent_geom.intersection(cutter_union).area) / max(parent_area, 1e-9)
            remainder = parent_geom.difference(cutter_union).buffer(0)
        except Exception:
            output_rows.append(parent_row)
            continue

        min_area = max(float(min_remainder_area_m2), parent_area * 0.002)
        if coverage_ratio >= float(full_coverage_ratio) or remainder is None or getattr(remainder, "is_empty", True):
            parent_drop_count += 1
            continue
        if float(getattr(remainder, "area", 0.0) or 0.0) < min_area:
            parent_drop_count += 1
            continue

        new_row = parent_row.copy()
        new_row["geometry"] = remainder
        new_row["__has_building_parts_cut"] = True
        output_rows.append(new_row)
        parent_split_count += 1

    if not output_rows and parts.empty and passthrough.empty:
        return buildings.iloc[0:0].copy()

    frames = []
    if output_rows:
        frames.append(GeoDataFrame(output_rows, crs=buildings.crs))
    frames.append(parts)
    if not passthrough.empty:
        frames.append(passthrough)

    result = GeoDataFrame(
        pd.concat(frames, ignore_index=True, sort=False),
        crs=buildings.crs,
    )
    print(
        "[BUILDINGS] building:part normalization: "
        f"parents_split={parent_split_count}, parents_dropped={parent_drop_count}, "
        f"parts_kept={len(parts)}, total={len(buildings)}->{len(result)}"
    )
    return result


def process_building_layer(
    *,
    task: Any,
    request: Any,
    scale_factor: Optional[float],
    terrain_provider: Any,
    global_center: Any,
    stl_extra_embed_m: float,
    gdf_buildings_local: Optional[GeoDataFrame],
    support_exclusion_polygons: Optional[BaseGeometry] = None,
    road_insert_mask: Optional[BaseGeometry] = None,
    road_groove_mask: Optional[BaseGeometry] = None,
    building_exclusion_polygons: Optional[BaseGeometry] = None,
) -> BuildingLayerResult:
    if not (scale_factor and scale_factor > 0 and (terrain_provider is not None or request.is_ams_mode)):
        return BuildingLayerResult(meshes=None)

    if not getattr(request, "include_buildings", True):
        return BuildingLayerResult(meshes=None)

    if gdf_buildings_local is None or gdf_buildings_local.empty:
        return BuildingLayerResult(meshes=None)

    task.update_status("processing", 50, "Generating 3D buildings...")
    height_scale_factor = float(
        getattr(request, "buildings_height_scale", None)
        or getattr(request, "building_height_multiplier", 1.0)
    )
    requested_min_height_m = float(getattr(request, "building_min_height", 2.0) or 2.0)
    printable_min_height_m = (1.0 / scale_factor) if scale_factor > 0 else 2.0
    min_building_height_m = max(requested_min_height_m, printable_min_height_m)
    building_embed_m = stl_extra_embed_m if not request.is_ams_mode else 0.0
    gdf_buildings_for_mesh = split_building_parts_from_parent_footprints(gdf_buildings_local)

    building_records = process_buildings(
        gdf_buildings_for_mesh,
        terrain_provider=terrain_provider,
        global_center=global_center,
        height_multiplier=height_scale_factor,
        min_height=min_building_height_m,
        embed_depth=building_embed_m + float(getattr(request, "buildings_embed_mm", 0.0)) / scale_factor,
        coordinates_already_local=True,
        return_records=True,
        exclusion_polygons=building_exclusion_polygons,
        min_feature_m=model_mm_to_world_m(MICRO_REGION_THRESHOLD_MM, scale_factor) if scale_factor and scale_factor > 0 else 0.0,
        scale_factor=scale_factor,
    )
    meshes = [record.mesh for record in building_records if getattr(record, "mesh", None) is not None]
    footprints = None
    try:
        footprint_parts = [
            record.footprint
            for record in building_records
            if getattr(record, "footprint", None) is not None and not getattr(record.footprint, "is_empty", True)
        ]
        if footprint_parts:
            footprints = unary_union(footprint_parts)
            if footprints is not None and not getattr(footprints, "is_empty", True):
                footprints = footprints.buffer(0)
    except Exception:
        footprints = None
    support_bottom_z = float(getattr(terrain_provider, "min_z", 0.0)) if terrain_provider is not None else None
    support_inset_m = 0.01
    support_min_feature_m = 0.0
    support_meshes = []
    foundation_depth_mm = float(getattr(request, "building_foundation_mm", 0.0) or 0.0)
    embed_depth_mm = float(getattr(request, "buildings_embed_mm", 0.0) or 0.0)
    if scale_factor and scale_factor > 0:
        try:
            support_inset_m = max(float((0.7 / 1000.0) / float(scale_factor)), 0.01)
        except Exception:
            support_inset_m = 0.01
        support_min_feature_m = model_mm_to_world_m(MICRO_REGION_THRESHOLD_MM, scale_factor)
    support_exclusion = support_exclusion_polygons
    if (
        support_exclusion is not None
        and not getattr(support_exclusion, "is_empty", True)
        and building_exclusion_polygons is not None
        and not getattr(building_exclusion_polygons, "is_empty", True)
    ):
        try:
            support_exclusion = support_exclusion.union(building_exclusion_polygons).buffer(0)
        except Exception:
            pass
    elif building_exclusion_polygons is not None and not getattr(building_exclusion_polygons, "is_empty", True):
        support_exclusion = building_exclusion_polygons
    should_build_supports = bool(building_records) and foundation_depth_mm <= 0.0 and embed_depth_mm <= 0.0
    if should_build_supports:
        support_meshes = build_building_supports(
            building_records,
            support_bottom_z=support_bottom_z,
            top_overlap_m=0.03,
            footprint_inset_m=float(support_inset_m),
            exclusion_polygons=support_exclusion,
            min_feature_m=float(support_min_feature_m),
        )
    merged_meshes = merge_building_and_support_meshes(meshes, support_meshes)
    merged_mesh = concatenate_meshes(merged_meshes)
    if merged_mesh is not None and scale_factor and float(scale_factor) > 0:
        merged_mesh = validate_building_mesh(merged_mesh, scale_factor=float(scale_factor))
    export_meshes = merged_meshes
    return BuildingLayerResult(
        meshes=export_meshes,
        support_meshes=support_meshes,
        merged_mesh=merged_mesh,
        footprints=footprints,
    )
