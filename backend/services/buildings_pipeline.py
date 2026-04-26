from __future__ import annotations

from typing import Any, Optional

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
    height_scale_factor = float(getattr(request, "buildings_height_scale", 1.0))
    min_building_height_m = (1.0 / scale_factor) if scale_factor > 0 else 2.0
    building_embed_m = stl_extra_embed_m if not request.is_ams_mode else 0.0

    building_records = process_buildings(
        gdf_buildings_local,
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
