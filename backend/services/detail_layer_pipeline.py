from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Optional

import trimesh
from geopandas import GeoDataFrame
from shapely.geometry import GeometryCollection
from shapely.geometry.base import BaseGeometry

from services.boolean_backends import BooleanBackend
from services.buildings_pipeline import process_building_layer
from services.canonical_mask_bundle import CanonicalMaskBundle
from services.detail_layer_utils import MIN_LAND_WIDTH_MODEL_MM, MICRO_REGION_THRESHOLD_MM, model_mm_to_world_m
from services.groove_pipeline import GrooveCutResult, cut_inlay_grooves, prepare_road_cut_mask
from services.inlay_fit import InlayFitConfig
from services.parks_pipeline import process_park_layer
from services.processing_results import GreenAreaProcessingResult, RoadProcessingResult
from services.roads_pipeline import process_road_layer
from services.water_layer_pipeline import process_water_layer


@dataclass
class DetailLayerPipelineResult:
    terrain_mesh: Optional[trimesh.Trimesh]
    road_mesh: Optional[trimesh.Trimesh]
    road_result: Optional[RoadProcessingResult]
    road_cut_source: Any
    road_groove_mask: Optional[BaseGeometry]
    building_meshes: Any
    water_mesh: Optional[trimesh.Trimesh]
    water_cut_polygons: Optional[BaseGeometry]
    parks_mesh: Optional[trimesh.Trimesh]
    parks_result: Optional[GreenAreaProcessingResult]
    groove_result: Optional[GrooveCutResult]
    merged_building_mesh: Optional[trimesh.Trimesh]
    support_meshes: Any
    building_footprints: Optional[BaseGeometry]


@dataclass
class CanonicalRoadMasks:
    road_insert_mask: Optional[BaseGeometry]
    road_groove_mask: Optional[BaseGeometry]
    building_exclusion_mask: Optional[BaseGeometry]
    support_exclusion_mask: Optional[BaseGeometry]


def _build_canonical_road_masks(
    *,
    road_insert_source: Optional[BaseGeometry],
    building_union_local: Optional[BaseGeometry],
    scale_factor: Optional[float],
    groove_clearance_mm: float,
    tiny_feature_threshold_mm: float,
    road_gap_fill_threshold_mm: float,
    zone_polygon_local: Optional[BaseGeometry],
    zone_prefix: str,
    road_groove_mask_override: Optional[BaseGeometry] = None,
    preserve_exact_masks: bool = False,
) -> CanonicalRoadMasks:
    # ── groove clearance in world metres ─────────────────────────────────────
    # This is used to compute building_exclusion_mask as road_insert.buffer(clearance).
    # Both paths (preserve-exact and normal) use the same formula so buildings are
    # always clipped at exactly groove_clearance_m from the road surface edge.
    groove_clearance_m = (
        float(groove_clearance_mm) / float(scale_factor)
        if scale_factor and float(scale_factor) > 0
        else 0.0
    )

    road_insert_mask = road_insert_source
    building_exclusion_mask = road_insert_mask
    support_exclusion_mask = road_insert_mask

    def _make_building_exclusion(road_mask, groove_mask):
        """
        building_exclusion = groove_mask

        groove_mask = road_insert.buffer(groove_clearance) extends
        groove_clearance INTO building footprints.  building.difference(groove_mask)
        clips building by groove_clearance from road-adjacent edges.
        """
        base = groove_mask if groove_mask is not None else road_mask
        if base is None or getattr(base, "is_empty", True):
            return road_mask
        return base

    # ── preserve-exact path ───────────────────────────────────────────────────
    if (
        preserve_exact_masks
        and road_insert_mask is not None
        and not getattr(road_insert_mask, "is_empty", True)
        and road_groove_mask_override is not None
        and not getattr(road_groove_mask_override, "is_empty", True)
    ):
        road_groove_mask = road_groove_mask_override
        try:
            support_exclusion_mask = road_groove_mask.union(road_insert_mask).buffer(0)
        except Exception:
            support_exclusion_mask = road_groove_mask
        building_exclusion_mask = _make_building_exclusion(road_insert_mask, road_groove_mask)
        return CanonicalRoadMasks(
            road_insert_mask=road_insert_mask,
            road_groove_mask=road_groove_mask,
            building_exclusion_mask=building_exclusion_mask,
            support_exclusion_mask=support_exclusion_mask,
        )

    # ── normal path ───────────────────────────────────────────────────────────
    if road_insert_mask is not None and not getattr(road_insert_mask, "is_empty", True):
        try:
            road_insert_mask = road_insert_mask.buffer(0)
        except Exception:
            pass

    # Road keeps its full shape — buildings yield to roads (buildings are clipped
    # by building_exclusion_mask = road_groove_mask, which is computed below).
    # DO NOT subtract buildings from road here.
    road_for_groove = road_insert_mask

    if road_groove_mask_override is not None and not getattr(road_groove_mask_override, "is_empty", True):
        road_groove_mask = road_groove_mask_override
    else:
        road_groove_mask = prepare_road_cut_mask(
            merged_roads_geom_local=road_for_groove,
            building_union_local=building_union_local,
            scale_factor=scale_factor,
            groove_clearance_mm=float(groove_clearance_mm),
            building_clearance_mm=0.0,
            zone_polygon_local=zone_polygon_local,
            min_printable_mm=float(tiny_feature_threshold_mm),
            road_gap_fill_threshold_mm=float(road_gap_fill_threshold_mm),
            zone_prefix=zone_prefix,
        )

    road_mask_for_support = road_insert_mask or road_for_groove
    if road_groove_mask is not None and not getattr(road_groove_mask, "is_empty", True):
        try:
            support_exclusion_mask = road_groove_mask.union(road_mask_for_support).buffer(0) if road_mask_for_support is not None else road_groove_mask
        except Exception:
            support_exclusion_mask = road_groove_mask
    else:
        support_exclusion_mask = road_mask_for_support

    building_exclusion_mask = _make_building_exclusion(road_insert_mask, road_groove_mask)

    return CanonicalRoadMasks(
        road_insert_mask=road_insert_mask,
        road_groove_mask=road_groove_mask,
        building_exclusion_mask=building_exclusion_mask,
        support_exclusion_mask=support_exclusion_mask,
    )


def process_detail_layers(
    *,
    task: Any,
    request: Any,
    scale_factor: Optional[float],
    terrain_provider: Any,
    terrain_mesh: Optional[trimesh.Trimesh],
    global_center: Any,
    G_roads: Any,
    water_geoms_for_bridges: Any,
    road_width_multiplier_effective: float,
    zone_polygon_local: Any,
    building_union_local: Any,
    merged_roads_geom_local: Any,
    road_cut_mask: Any,
    road_height_m: float,
    road_embed_m: float,
    stl_extra_embed_m: float,
    gdf_buildings_local: Optional[GeoDataFrame],
    gdf_water: Optional[GeoDataFrame],
    water_depth_m: float,
    gdf_green: Optional[GeoDataFrame],
    groove_clearance_mm: float,
    apply_grooves: bool = True,
    boolean_backend: Optional[BooleanBackend] = None,
    zone_prefix: str = "",
    canonical_mask_bundle: Optional[CanonicalMaskBundle] = None,
) -> DetailLayerPipelineResult:
    pipeline_start = time.perf_counter()

    def _log_stage(name: str, started_at: float) -> None:
        elapsed = time.perf_counter() - started_at
        total = time.perf_counter() - pipeline_start
        print(f"[TIMING] {zone_prefix}detail.{name}: {elapsed:.2f}s (detail total {total:.2f}s)")

    micro_region_threshold_mm = float(MICRO_REGION_THRESHOLD_MM)
    tiny_feature_threshold_mm = max(float(getattr(request, "tiny_feature_threshold_mm", 0.2)), 0.2)
    road_gap_fill_threshold_mm = max(
        float(getattr(request, "road_gap_fill_threshold_mm", 0.45)),
        float(tiny_feature_threshold_mm),
    )
    fit_config = InlayFitConfig(
        insert_side_clearance_mm=0.0,
        groove_side_clearance_mm=float(groove_clearance_mm),
    )
    # Roads keep the full groove clearance, but neighbouring inlays should not
    # inset by the same full amount or the shared white gap becomes visually
    # doubled. Use a split-fit for parks/water inserts.
    shared_inlay_fit_clearance_mm = float(fit_config.groove_side_clearance_mm) * 0.5
    stage_start = time.perf_counter()
    road_layer = process_road_layer(
        task=task,
        request=request,
        scale_factor=scale_factor,
        terrain_provider=terrain_provider,
        terrain_mesh=terrain_mesh,
        global_center=global_center,
        G_roads=G_roads,
        water_geoms_for_bridges=water_geoms_for_bridges,
        road_width_multiplier_effective=road_width_multiplier_effective,
        zone_polygon_local=zone_polygon_local,
        building_union_local=building_union_local,
        merged_roads_geom_local=merged_roads_geom_local,
        road_height_m=road_height_m,
        road_embed_m=road_embed_m,
        stl_extra_embed_m=stl_extra_embed_m,
        fit_config=fit_config,
        road_polygons_override=(
            getattr(canonical_mask_bundle, "roads_final", None) if canonical_mask_bundle is not None else None
        ),
    )
    _log_stage("roads", stage_start)
    road_mesh = road_layer.mesh
    road_result = road_layer.road_result
    road_cut_source = road_layer.road_cut_source
    road_insert_source = (
        getattr(canonical_mask_bundle, "roads_final", None)
        if canonical_mask_bundle is not None and getattr(canonical_mask_bundle, "roads_final", None) is not None
        else (
            road_result.source_polygons
            if road_result is not None and getattr(road_result, "source_polygons", None) is not None
            else road_cut_source
        )
    )
    canonical_masks = _build_canonical_road_masks(
        road_insert_source=road_insert_source,
        building_union_local=building_union_local,
        scale_factor=scale_factor,
        groove_clearance_mm=float(fit_config.groove_side_clearance_mm),
        tiny_feature_threshold_mm=float(tiny_feature_threshold_mm),
        road_gap_fill_threshold_mm=float(road_gap_fill_threshold_mm),
        zone_polygon_local=zone_polygon_local,
        zone_prefix=zone_prefix,
        road_groove_mask_override=(
            getattr(canonical_mask_bundle, "road_groove_mask", None) if canonical_mask_bundle is not None else None
        ),
        preserve_exact_masks=canonical_mask_bundle is not None,
    )
    road_insert_exclusion_polygons = canonical_masks.road_insert_mask
    canonical_road_groove_mask = canonical_masks.road_groove_mask
    support_exclusion_mask = canonical_masks.support_exclusion_mask
    building_exclusion_mask = canonical_masks.building_exclusion_mask

    canonical_water_for_detail = None
    if canonical_mask_bundle is not None:
        canonical_water_for_detail = getattr(canonical_mask_bundle, "water_final", None)
        if canonical_water_for_detail is None:
            canonical_water_for_detail = GeometryCollection()

    stage_start = time.perf_counter()
    building_layer = process_building_layer(
        task=task,
        request=request,
        scale_factor=scale_factor,
        terrain_provider=terrain_provider,
        global_center=global_center,
        stl_extra_embed_m=stl_extra_embed_m,
        gdf_buildings_local=gdf_buildings_local,
        support_exclusion_polygons=support_exclusion_mask,
        road_insert_mask=road_insert_exclusion_polygons,
        road_groove_mask=canonical_road_groove_mask,
        building_exclusion_polygons=building_exclusion_mask,  # buildings clipped by road+groove; road mesh stays whole
    )
    _log_stage("buildings", stage_start)
    building_meshes = building_layer.meshes
    merged_building_mesh = building_layer.merged_mesh
    support_meshes = building_layer.support_meshes

    stage_start = time.perf_counter()
    water_layer = process_water_layer(
        task=task,
        request=request,
        scale_factor=scale_factor,
        terrain_provider=terrain_provider,
        global_center=global_center,
        gdf_water=gdf_water,
        water_depth_m=water_depth_m,
        road_polygons=road_insert_exclusion_polygons or canonical_road_groove_mask,
        building_polygons=building_union_local,
        coordinates_already_local=True,
        zone_prefix=zone_prefix,
        water_polygons_override=canonical_water_for_detail,
        fit_clearance_mm=float(shared_inlay_fit_clearance_mm),
    )
    _log_stage("water", stage_start)
    water_mesh = water_layer.mesh
    water_cut_polygons = water_layer.cutting_polygons

    stage_start = time.perf_counter()
    park_layer = process_park_layer(
        task=task,
        request=request,
        scale_factor=scale_factor,
        terrain_provider=terrain_provider,
        terrain_mesh=terrain_mesh,
        global_center=global_center,
        zone_polygon_local=zone_polygon_local,
        # Park inserts must respect the actual road groove footprint; otherwise the
        # final park inlay can compete with the road groove at shared boundaries.
        road_cut_source=canonical_road_groove_mask or road_insert_exclusion_polygons,
        building_union_local=building_union_local,
        water_polygons=water_cut_polygons,
        road_exclusion_clearance_mm=0.0,
        fit_clearance_mm=float(shared_inlay_fit_clearance_mm),
        gdf_green=gdf_green,
        zone_prefix=zone_prefix,
        park_polygons_override=(
            getattr(canonical_mask_bundle, "parks_final", None) if canonical_mask_bundle is not None else None
        ),
    )
    _log_stage("parks", stage_start)
    parks_mesh = park_layer.mesh
    parks_result = park_layer.parks_result

    has_road_grooves = (
        apply_grooves
        and not request.is_ams_mode
        and terrain_mesh is not None
        and road_mesh is not None
        and scale_factor
        and scale_factor > 0
    )
    has_park_grooves = (
        apply_grooves
        and not request.is_ams_mode
        and terrain_mesh is not None
        and parks_mesh is not None
        and scale_factor
        and scale_factor > 0
    )
    has_water_grooves = (
        apply_grooves
        and not request.is_ams_mode
        and terrain_mesh is not None
        and water_mesh is not None
        and scale_factor
        and scale_factor > 0
    )

    groove_result = None
    if not apply_grooves:
        print(f"[INFO] {zone_prefix} Groove cutting skipped: preview exports the real pipeline mesh before groove booleans")
    if has_road_grooves or has_park_grooves or has_water_grooves:
        try:
            stage_start = time.perf_counter()
            groove_result = cut_inlay_grooves(
                terrain_mesh=terrain_mesh,
                road_mesh=road_mesh,
                parks_mesh=parks_mesh,
                water_mesh=water_mesh,
                road_cut_mask=canonical_road_groove_mask or road_cut_mask,
                merged_roads_geom_local=road_insert_exclusion_polygons or road_cut_source,
                parks_polygons=(
                    getattr(canonical_mask_bundle, "parks_final", None)
                    if canonical_mask_bundle is not None and getattr(canonical_mask_bundle, "parks_final", None) is not None
                    else (parks_result.processed_polygons if parks_result is not None else None)
                ),
                water_polygons=(
                    canonical_water_for_detail
                    if canonical_mask_bundle is not None
                    else water_cut_polygons
                ),
                building_polygons=building_union_local,
                scale_factor=float(scale_factor),
                groove_clearance_mm=float(fit_config.groove_side_clearance_mm),
                road_embed_m=road_embed_m,
                parks_embed_mm=float(request.parks_embed_mm),
                water_depth_m=water_depth_m,
                boolean_backend=boolean_backend,
                zone_prefix=zone_prefix,
                zone_polygon_local=zone_polygon_local,
                min_printable_mm=max(float(MIN_LAND_WIDTH_MODEL_MM), tiny_feature_threshold_mm),
                parks_groove_override=(
                    getattr(canonical_mask_bundle, "parks_groove_mask", None) if canonical_mask_bundle is not None else None
                ),
                water_groove_override=(
                    (
                        getattr(canonical_mask_bundle, "water_groove_mask", None)
                        or getattr(canonical_mask_bundle, "water_final", None)
                    )
                    if canonical_mask_bundle is not None
                    else None
                ),
                use_exact_masks=canonical_mask_bundle is not None,
            )
            terrain_mesh = groove_result.terrain_mesh
            _log_stage("grooves", stage_start)
        except Exception as exc:
            print(f"[WARN] {zone_prefix} Failed to cut grooves: {exc}")
            import traceback

            traceback.print_exc()

    return DetailLayerPipelineResult(
        terrain_mesh=terrain_mesh,
        road_mesh=road_mesh,
        road_result=road_result,
        road_cut_source=road_cut_source,
        road_groove_mask=canonical_road_groove_mask or road_cut_mask,
        building_meshes=building_meshes,
        water_mesh=water_mesh,
        water_cut_polygons=water_cut_polygons,
        parks_mesh=parks_mesh,
        parks_result=parks_result,
        groove_result=groove_result,
        merged_building_mesh=merged_building_mesh,
        support_meshes=support_meshes,
        building_footprints=(
            getattr(canonical_mask_bundle, "buildings_footprints", None)
            if canonical_mask_bundle is not None and getattr(canonical_mask_bundle, "buildings_footprints", None) is not None
            else getattr(building_layer, "footprints", None)
        ),
    )
