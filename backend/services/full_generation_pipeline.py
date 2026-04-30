from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import trimesh
from services.canonical_2d_pipeline import prepare_canonical_2d_stage
from services.bridge_water_pipeline import prepare_bridge_water_geometries
from services.building_supports import union_mesh_collection
from services.canonical_mask_bundle import load_canonical_mask_bundle
from services.data_fetch_pipeline import fetch_generation_data
from services.debug_bundle_pipeline import create_debug_bundle
from services.detail_layer_pipeline import process_detail_layers
from services.export_pipeline import export_generation_outputs
from services.detail_layer_utils import MICRO_REGION_THRESHOLD_MM
from services.firebase_publish_pipeline import publish_outputs_to_firebase
from services.mesh_clip_pipeline import clip_generated_meshes
from services.mesh_postprocess_pipeline import postprocess_generated_meshes
from services.processing_results import SourceDataResult, TerrainStageResult, ZonePreparationResult
from services.road_groove_validation import print_road_groove_validation_report
from services.runtime_canonical_masks import build_runtime_canonical_bundle
from services.terrain_only_pipeline import TerrainOnlyPipelineResult, run_terrain_only_pipeline
from services.zone_context_pipeline import build_zone_context
from services.zone_geometry_pipeline import prepare_zone_geometry
from services.building_geometry_pipeline import prepare_building_geometry
from services.generation_pipeline import process_generation_stage
from services.geometry_preclip_pipeline import prepare_preclipped_geometry
from services.inlay_fit import InlayFitConfig
from services.print_acceptance import summarize_export_print_failures, write_export_print_acceptance_report
from services.printer_profile import get_printer_profile_for_request
from services.road_geometry_pipeline import prepare_road_geometry
from services.terrain_pipeline_utils import compute_water_depth_m, resolve_generation_source_crs
from services.terrain_building_merge_pipeline import merge_terrain_and_buildings
from services.stage_snapshot_pipeline import create_stage_snapshot_collector
from services.model_exporter import (
    _filter_mesh_components_for_export,
    _normalize_part_for_print_export,
    repair_base_export_mesh_aggressive,
    repair_road_export_mesh,
    export_stl_safe,
)


@dataclass
class FullGenerationPipelineResult:
    output_file_abs: Path
    primary_format: str
    terrain_only_result: Optional[TerrainOnlyPipelineResult] = None


@dataclass
class CanonicalPreviewPipelineResult:
    zone: ZonePreparationResult
    source: SourceDataResult
    canonical_2d_stage: Any
    canonical_mask_bundle: Any
    elapsed_seconds: float


def _geometry_area(geom: Any) -> float:
    if geom is None or getattr(geom, "is_empty", True):
        return 0.0
    try:
        return float(getattr(geom, "area", 0.0) or 0.0)
    except Exception:
        return 0.0


def _symdiff_area(lhs: Any, rhs: Any) -> float:
    if lhs is None and rhs is None:
        return 0.0
    if lhs is None or getattr(lhs, "is_empty", True):
        return _geometry_area(rhs)
    if rhs is None or getattr(rhs, "is_empty", True):
        return _geometry_area(lhs)
    try:
        return float(getattr(lhs.symmetric_difference(rhs), "area", 0.0) or 0.0)
    except Exception:
        return float("inf")


def _validate_canonical_mask_handoff(
    *,
    canonical_mask_bundle: Any,
    terrain_stage: Any,
    detail_layers: Any,
    zone_prefix: str,
) -> None:
    """Hard-check that 3D stages consume canonical 2D masks without drift."""
    if canonical_mask_bundle is None:
        return

    problems: list[str] = []

    def _check_match(name: str, canonical_geom: Any, runtime_geom: Any) -> None:
        c_area = _geometry_area(canonical_geom)
        r_area = _geometry_area(runtime_geom)
        if c_area <= 0.0 and r_area <= 0.0:
            return
        delta = _symdiff_area(canonical_geom, runtime_geom)
        denom = max(c_area, r_area, 1.0)
        rel = delta / denom
        # Allow tiny boolean noise, reject material drift.
        if not (delta == delta) or rel > 0.001:
            problems.append(f"{name}:symdiff={delta:.6f}m2 rel={rel:.6f}")

    canonical_roads = getattr(canonical_mask_bundle, "roads_final", None)
    canonical_road_groove = getattr(canonical_mask_bundle, "road_groove_mask", None)
    canonical_parks = getattr(canonical_mask_bundle, "parks_final", None)
    canonical_water = getattr(canonical_mask_bundle, "water_final", None)

    terrain_road_cut = getattr(terrain_stage, "road_cut_mask", None)
    detail_road_insert = (
        getattr(getattr(detail_layers, "road_result", None), "source_polygons", None)
        or getattr(detail_layers, "road_cut_source", None)
    )
    detail_road_groove = getattr(detail_layers, "road_groove_mask", None)
    detail_parks = getattr(getattr(detail_layers, "parks_result", None), "processed_polygons", None)
    detail_water = getattr(detail_layers, "water_cut_polygons", None)

    _check_match("terrain.road_cut_mask_vs_canonical_road_groove", canonical_road_groove, terrain_road_cut)
    _check_match("detail.roads_final_vs_canonical_roads_final", canonical_roads, detail_road_insert)
    _check_match("detail.road_groove_vs_canonical_road_groove", canonical_road_groove, detail_road_groove)
    _check_match("detail.parks_final_vs_canonical_parks_final", canonical_parks, detail_parks)
    _check_match("detail.water_final_vs_canonical_water_final", canonical_water, detail_water)

    if problems:
        raise RuntimeError(
            "Canonical 2D -> 3D handoff drift detected: " + "; ".join(problems[:8])
        )

    print(f"[INFO] {zone_prefix}Canonical 2D -> 3D handoff verified (mask parity: OK)")


def _is_printable_water_export_mesh(mesh: Any, *, min_face_count: int = 12) -> bool:
    """Validate a water mesh would survive slicing.

    Why: post-filter the water mesh may still have zero volume or non-manifold
    edges that make PrusaSlicer abort (`slicer:water:slice_failed`). Checking
    here lets recovery drop water cleanly instead of retrying a broken file.
    """
    if mesh is None:
        return False
    faces = getattr(mesh, "faces", None)
    if faces is None or len(faces) < int(min_face_count):
        return False
    try:
        volume = float(getattr(mesh, "volume", 0.0) or 0.0)
    except Exception:
        volume = 0.0
    if not (volume == volume) or volume <= 1e-9:
        return False
    try:
        bounds = mesh.bounds
        extents = [float(bounds[1][i] - bounds[0][i]) for i in range(3)]
    except Exception:
        return False
    if any(e != e for e in extents) or min(extents) <= 1e-6:
        return False
    return True


def _repair_print_part_file(
    *,
    part_name: str,
    path: Path,
    report: dict[str, Any],
) -> bool:
    if not path.exists():
        return False
    try:
        mesh = trimesh.load(path, force="mesh")
    except Exception:
        return False
    if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
        return False

    updated = mesh.copy()
    changed = False

    try:
        updated = _normalize_part_for_print_export(updated, part_key=part_name) or updated
    except Exception:
        pass

    if part_name in ("base", "terrain"):
        repaired = repair_base_export_mesh_aggressive(updated)
        if repaired is not None:
            updated = repaired
            changed = True
    elif part_name == "roads":
        repaired = repair_road_export_mesh(updated)
        if repaired is not None:
            updated = repaired
            changed = True
    elif part_name in ("parks", "green"):
        filtered = _filter_mesh_components_for_export(
            updated,
            min_feature_mm=0.45,
            min_area_mm2=0.08,
        )
        if filtered is not None:
            updated = filtered
            changed = True
    elif part_name == "water":
        failing_checks = list(report.get("failing_checks") or [])
        # If the slicer itself rejected the water mesh, component filtering
        # won't rescue it — drop water entirely since it's an optional layer.
        slicer_rejected = any(
            str(check).startswith("slicer:water:") for check in failing_checks
        )
        filtered = None
        if not slicer_rejected:
            filtered = _filter_mesh_components_for_export(
                updated,
                min_feature_mm=0.45,
                min_area_mm2=0.12,
            )
        mesh_survives = (
            not slicer_rejected
            and filtered is not None
            and filtered.faces is not None
            and len(filtered.faces) > 0
            and _is_printable_water_export_mesh(filtered)
        )
        if mesh_survives:
            updated = filtered
            changed = True
        else:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            return True

    if not changed:
        return False

    try:
        export_stl_safe(updated, str(path))
        return True
    except Exception:
        return False


def _attempt_print_recovery(
    *,
    task_id: str,
    output_dir: Path,
    task: Any,
    parts_for_print: dict[str, str],
    expected_parts: dict[str, bool],
    printer_profile: Any,
    initial_report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], dict[str, bool]]:
    failing_checks = set(initial_report.get("failing_checks") or [])
    if not failing_checks:
        return initial_report, parts_for_print, expected_parts

    repaired_any = False
    normalized_parts = {str(k).lower(): str(v) for k, v in (parts_for_print or {}).items()}
    adjusted_expected = dict(expected_parts)

    for part_name in ("base", "roads", "parks", "water"):
        path_str = normalized_parts.get(part_name)
        if not path_str:
            continue
        path = Path(path_str)
        should_try = any(
            check.startswith(f"{part_name}:") or check.startswith(f"slicer:{part_name}:")
            for check in failing_checks
        )
        if not should_try:
            continue
        if _repair_print_part_file(part_name=part_name, path=path, report=initial_report):
            repaired_any = True
            if part_name == "water" and not path.exists():
                normalized_parts.pop("water", None)
                adjusted_expected["water"] = False

    if not repaired_any:
        return initial_report, normalized_parts, adjusted_expected

    recovery_path = write_export_print_acceptance_report(
        task_id=task_id,
        output_dir=output_dir,
        parts_for_print=normalized_parts,
        expected_parts=adjusted_expected,
        printer_profile=printer_profile,
        require_slicer_validation=True,
        fail_on_slicer_warnings=True,
        rotate_x_deg=0,
    )
    task.set_output("print_acceptance", str(recovery_path.resolve()))
    recovery_report = json.loads(recovery_path.read_text(encoding="utf-8"))
    return recovery_report, normalized_parts, adjusted_expected


def _collect_print_part_paths(task: Any, export_result: Any) -> dict[str, str]:
    parts: dict[str, str] = {}
    output_files = getattr(task, "output_files", {}) or {}
    for part_name in ("base", "roads", "parks", "water", "buildings"):
        task_key = f"{part_name}_stl"
        part_path = output_files.get(task_key)
        if part_path:
            parts[part_name] = str(part_path)
    if not parts and getattr(export_result, "parts_from_main", None):
        for part_name, part_path in (export_result.parts_from_main or {}).items():
            normalized = str(part_name).lower()
            if normalized in ("base", "roads", "parks", "water", "buildings") and part_path:
                parts[normalized] = str(part_path)
    return parts


def _compute_stl_extra_embed(scale_factor: Optional[float], export_format: str) -> float:
    try:
        if str(export_format).lower() == "stl" and scale_factor and scale_factor > 0:
            return 0.6 / float(scale_factor)
    except Exception:
        pass
    return 0.0


def _validate_groove_stage(
    *,
    detail_layers: Any,
    task: Any,
    zone_prefix: str,
    require_success: bool = True,
) -> None:
    groove_result = getattr(detail_layers, "groove_result", None)
    if groove_result is None:
        return
    grooves_expected = getattr(groove_result, "grooves_expected", None)
    if grooves_expected is None:
        grooves_expected = any(
            geometry is not None and not getattr(geometry, "is_empty", True)
            for geometry in (
                getattr(groove_result, "road_polygons_used", None),
                getattr(groove_result, "parks_polygons_used", None),
                getattr(groove_result, "water_polygons_used", None),
            )
        )
    if not bool(grooves_expected):
        return

    if bool(getattr(groove_result, "rejected", False)):
        reason = getattr(groove_result, "rejection_reason", None) or getattr(groove_result, "failure_reason", None) or "unknown_rejection"
        message = f"Groove stage failed: unsafe groove cut was rejected ({reason})"
        if not require_success:
            print(f"[WARN] {zone_prefix}{message}; continuing because groove success is optional for this run")
            return
        if hasattr(task, "fail"):
            task.fail(message)
        raise RuntimeError(message)

    if not bool(getattr(groove_result, "change_applied", False)):
        reason = getattr(groove_result, "failure_reason", None) or "boolean_noop"
        message = (
            f"Groove stage failed: canonical groove masks existed but no groove cut was applied "
            f"({reason})"
        )
        if not require_success:
            print(f"[WARN] {zone_prefix}{message}; continuing because groove success is optional for this run")
            return
        if hasattr(task, "fail"):
            task.fail(message)
        raise RuntimeError(message)

    print(
        f"[INFO] {zone_prefix}Groove stage verified: backend="
        f"{getattr(groove_result, 'boolean_backend_name', 'unknown')} "
        f"volume_removed_m3={getattr(groove_result, 'volume_removed_m3', None)}"
    )


def _validate_groove_result(
    *,
    detail_layers: Any,
    task: Any,
    zone_prefix: str,
) -> None:
    _validate_groove_stage(detail_layers=detail_layers, task=task, zone_prefix=zone_prefix)


def _prepare_zone_stage(
    *,
    request: Any,
    global_center: Any,
    zone_polygon_coords: Optional[list],
    grid_bbox_latlon: Any,
    zone_row: Any,
    zone_col: Any,
    hex_size_m: Any,
    zone_prefix: str,
) -> ZonePreparationResult:
    zone_geometry = prepare_zone_geometry(
        global_center=global_center,
        grid_bbox_latlon=grid_bbox_latlon,
        zone_row=zone_row,
        zone_col=zone_col,
        hex_size_m=hex_size_m,
        zone_polygon_coords=zone_polygon_coords,
        zone_prefix=zone_prefix,
    )
    zone_context = build_zone_context(
        request=request,
        global_center=global_center,
        zone_polygon_local=zone_geometry.zone_polygon_local,
        reference_xy_m=zone_geometry.reference_xy_m,
        zone_prefix=zone_prefix,
    )
    scale_factor = zone_context.scale_factor
    return ZonePreparationResult(
        zone_polygon_local=zone_geometry.zone_polygon_local,
        reference_xy_m=zone_geometry.reference_xy_m,
        bbox_meters=zone_context.bbox_meters,
        scale_factor=scale_factor,
        # Keep road width multiplier faithful to request.
        # The legacy x3 expansion over-merges neighboring roads into blobs and
        # destroys canonical mask topology.
        road_width_multiplier_effective=float(request.road_width_multiplier),
        stl_extra_embed_m=_compute_stl_extra_embed(scale_factor, getattr(request, "export_format", "")),
    )


def _fetch_source_stage(
    *,
    task: Any,
    request: Any,
    global_center: Any,
    zone_prefix: str,
) -> SourceDataResult:
    data_result = fetch_generation_data(
        request=request,
        global_center=global_center,
        task=task,
        zone_prefix=zone_prefix,
    )
    return SourceDataResult(
        gdf_buildings=data_result.gdf_buildings,
        gdf_water=data_result.gdf_water,
        G_roads=data_result.G_roads,
        gdf_green=data_result.gdf_green,
    )


def _validate_source_stage(
    *,
    source: SourceDataResult,
    zone_prefix: str,
) -> None:
    building_count = len(source.gdf_buildings) if getattr(source.gdf_buildings, "empty", True) is False else 0
    water_count = len(source.gdf_water) if getattr(source.gdf_water, "empty", True) is False else 0
    green_count = len(source.gdf_green) if getattr(source.gdf_green, "empty", True) is False else 0
    road_count = 0
    if source.G_roads is not None:
        if hasattr(source.G_roads, "edges"):
            try:
                road_count = len(list(source.G_roads.edges()))
            except Exception:
                road_count = 0
        elif hasattr(source.G_roads, "__len__"):
            road_count = len(source.G_roads)

    if road_count == 0 and building_count == 0:
        print(
            f"[WARN] {zone_prefix}Source data is sparse after API fetch; continuing in sparse-zone mode "
            f"(roads={road_count}, buildings={building_count}, water={water_count}, green={green_count})"
        )


def run_canonical_preview_pipeline(
    *,
    task: Any,
    request: Any,
    task_id: str,
    output_dir: Path,
    global_center: Any,
    zone_polygon_coords: Optional[list],
    grid_bbox_latlon: Any,
    zone_row: Any = None,
    zone_col: Any = None,
    hex_size_m: Any = None,
    zone_prefix: str = "[preview] ",
) -> CanonicalPreviewPipelineResult:
    """Run the real generation pipeline until the canonical 2D handoff.

    This is the latest point before terrain booleans, Blender/mesh construction,
    export, and slicer validation become expensive. The returned canonical masks
    are exactly the geometry source consumed by the 3D stages in
    `run_full_generation_pipeline`.
    """
    pipeline_start = time.perf_counter()
    zone = _prepare_zone_stage(
        request=request,
        global_center=global_center,
        zone_polygon_coords=zone_polygon_coords,
        grid_bbox_latlon=grid_bbox_latlon,
        zone_row=zone_row,
        zone_col=zone_col,
        hex_size_m=hex_size_m,
        zone_prefix=zone_prefix,
    )
    source = _fetch_source_stage(
        task=task,
        request=request,
        global_center=global_center,
        zone_prefix=zone_prefix,
    )
    _validate_source_stage(source=source, zone_prefix=zone_prefix)
    canonical_2d_stage = prepare_canonical_2d_stage(
        task_id=task_id,
        request=request,
        source=source,
        zone=zone,
        global_center=global_center,
        debug_generated_dir=(output_dir.parent / "debug" / "generated"),
        zone_prefix=zone_prefix,
    )
    return CanonicalPreviewPipelineResult(
        zone=zone,
        source=source,
        canonical_2d_stage=canonical_2d_stage,
        canonical_mask_bundle=canonical_2d_stage.canonical_mask_bundle,
        elapsed_seconds=time.perf_counter() - pipeline_start,
    )


def _run_terrain_stage(
    *,
    task: Any,
    request: Any,
    latlon_bbox: Any,
    global_center: Any,
    zone_prefix: str,
    zone: ZonePreparationResult,
    source: SourceDataResult,
    min_printable_gap_mm: float,
    groove_clearance_mm: float,
    canonical_mask_bundle: Any = None,
) -> TerrainStageResult:
    building_geometry = prepare_building_geometry(
        gdf_buildings=source.gdf_buildings,
        global_center=global_center,
        zone_prefix=zone_prefix,
    )

    source_crs = resolve_generation_source_crs(
        gdf_buildings=source.gdf_buildings,
        G_roads=source.G_roads,
        global_center=global_center,
        allow_global_center_fallback=True,
        zone_prefix=zone_prefix,
    )
    road_mask_cleanup_mm = max(float(getattr(request, "tiny_feature_threshold_mm", 0.0) or 0.0), 0.0)
    road_gap_fill_threshold_mm = max(float(min_printable_gap_mm or 0.0), 0.5)
    road_geometry = prepare_road_geometry(
        G_roads=source.G_roads,
        scale_factor=zone.scale_factor,
        road_width_multiplier_effective=zone.road_width_multiplier_effective,
        min_printable_gap_mm=min_printable_gap_mm,
        tiny_feature_threshold_mm=float(max(road_mask_cleanup_mm, 0.5)),
        road_gap_fill_threshold_mm=float(road_gap_fill_threshold_mm),
        enforce_printable_min_width=True,
        min_gap_fill_floor_mm=0.5,
        global_center=global_center,
        zone_polygon_local=zone.zone_polygon_local,
        zone_prefix=zone_prefix,
    )
    water_depth_m = compute_water_depth_m(
        water_depth_mm=float(request.water_depth),
        scale_factor=zone.scale_factor,
    )

    elevation_ref_m = getattr(request, "elevation_ref_m", None)
    baseline_offset_m = getattr(request, "baseline_offset_m", 0.0)
    if elevation_ref_m is not None:
        print(
            f"[INFO] {zone_prefix} Using global elevation_ref_m: {elevation_ref_m:.2f}m "
            "for elevation synchronization"
        )
        print(f"[INFO] {zone_prefix} Using global baseline_offset_m: {baseline_offset_m:.3f}m")
    else:
        print(f"[INFO] {zone_prefix} elevation_ref_m not provided, local normalization will be used")

    preclip_result = prepare_preclipped_geometry(
        gdf_buildings_local=building_geometry.gdf_buildings_local,
        building_geometries_for_flatten=building_geometry.building_geometries_for_flatten,
        gdf_water=source.gdf_water,
        global_center=global_center,
        zone_polygon_local=zone.zone_polygon_local,
        zone_prefix=zone_prefix,
    )

    road_height_m = None
    road_embed_m = None
    if zone.scale_factor and zone.scale_factor > 0:
        road_height_m = float(request.road_height_mm) / float(zone.scale_factor)
        road_embed_m = float(request.road_embed_mm) / float(zone.scale_factor)
        if zone.stl_extra_embed_m > 0:
            road_embed_m += float(zone.stl_extra_embed_m)
    if request.is_ams_mode and zone.scale_factor and zone.scale_factor > 0:
        road_height_m = 0.4 / zone.scale_factor
        road_embed_m = 0.0

    fit_config = InlayFitConfig(
        insert_side_clearance_mm=0.0,
        groove_side_clearance_mm=float(groove_clearance_mm),
    )

    bundle_road_groove = None
    if canonical_mask_bundle is not None:
        bundle_road_groove = getattr(canonical_mask_bundle, "road_groove_mask", None)

    generation_result = process_generation_stage(
        task=task,
        request=request,
        scale_factor=zone.scale_factor,
        bbox_meters=zone.bbox_meters,
        latlon_bbox=latlon_bbox,
        source_crs=source_crs,
        elevation_ref_m=elevation_ref_m,
        baseline_offset_m=baseline_offset_m,
        building_geometries_for_flatten=preclip_result.building_geometries_for_flatten,
        merged_roads_geom_local=road_geometry.merged_roads_geom_local,
        merged_roads_geom=road_geometry.merged_roads_geom,
        building_union_local=building_geometry.building_union_local,
        gdf_water_local=preclip_result.gdf_water_local,
        global_center=global_center,
        zone_polygon_local=zone.zone_polygon_local,
        groove_clearance_mm=float(fit_config.groove_side_clearance_mm),
        stl_extra_embed_m=zone.stl_extra_embed_m,
        zone_prefix=zone_prefix,
        road_height_m=road_height_m,
        road_embed_m=road_embed_m,
        road_cut_mask_override=bundle_road_groove,
    )

    final_road_height_m = generation_result.road_height_m
    final_road_embed_m = generation_result.road_embed_m
    if final_road_height_m is None and zone.scale_factor and zone.scale_factor > 0:
        final_road_height_m = float(request.road_height_mm) / float(zone.scale_factor)
    if final_road_embed_m is None and zone.scale_factor and zone.scale_factor > 0:
        final_road_embed_m = float(request.road_embed_mm) / float(zone.scale_factor)
        if zone.stl_extra_embed_m > 0:
            final_road_embed_m += float(zone.stl_extra_embed_m)

    return TerrainStageResult(
        terrain_mesh=generation_result.terrain_mesh,
        terrain_provider=generation_result.terrain_provider,
        road_cut_mask=generation_result.road_cut_mask,
        road_height_m=final_road_height_m,
        road_embed_m=final_road_embed_m,
        water_depth_m=water_depth_m,
        gdf_buildings_local=preclip_result.gdf_buildings_local,
        building_geometries_for_flatten=preclip_result.building_geometries_for_flatten,
        building_union_local=building_geometry.building_union_local,
        gdf_water_local=preclip_result.gdf_water_local,
        merged_roads_geom=road_geometry.merged_roads_geom,
        merged_roads_geom_local=road_geometry.merged_roads_geom_local,
        preclipped_to_zone=preclip_result.preclipped_to_zone,
    )


def run_full_generation_pipeline(
    *,
    task: Any,
    request: Any,
    task_id: str,
    output_dir: Path,
    global_center: Any,
    latlon_bbox: Any,
    zone_polygon_coords: Optional[list],
    grid_bbox_latlon: Any,
    zone_row: Any,
    zone_col: Any,
    hex_size_m: Any,
    zone_prefix: str = "",
    min_printable_gap_mm: float = 1.0,
    groove_clearance_mm: float = 0.15,
    require_groove_success: bool = True,
) -> FullGenerationPipelineResult:
    pipeline_start = time.perf_counter()
    stage_snapshot_collector = None
    stage_snapshot_manifest_path: Optional[Path] = None
    try:
        stage_snapshot_collector = create_stage_snapshot_collector(
            task_id=task_id,
            debug_root=(output_dir.parent / "debug"),
            zone_prefix=zone_prefix,
        )
    except Exception as exc:
        print(f"[WARN] {zone_prefix}Failed to initialize stage snapshot collector: {exc}")

    def _log_stage(name: str, started_at: float) -> None:
        elapsed = time.perf_counter() - started_at
        total = time.perf_counter() - pipeline_start
        print(f"[TIMING] {zone_prefix}{name}: {elapsed:.2f}s (total {total:.2f}s)")

    stage_start = time.perf_counter()
    zone = _prepare_zone_stage(
        request=request,
        global_center=global_center,
        zone_polygon_coords=zone_polygon_coords,
        grid_bbox_latlon=grid_bbox_latlon,
        zone_row=zone_row,
        zone_col=zone_col,
        hex_size_m=hex_size_m,
        zone_prefix=zone_prefix,
    )
    _log_stage("prepare_zone", stage_start)

    canonical_mask_bundle = None

    stage_start = time.perf_counter()
    source = _fetch_source_stage(
        task=task,
        request=request,
        global_center=global_center,
        zone_prefix=zone_prefix,
    )
    _validate_source_stage(source=source, zone_prefix=zone_prefix)
    _log_stage("fetch_source", stage_start)

    task.update_status("processing", 20, "Генерація рельєфу...")

    if request.terrain_only:
        terrain_only_result = run_terrain_only_pipeline(
            task=task,
            request=request,
            task_id=task_id,
            output_dir=output_dir,
            bbox_meters=zone.bbox_meters,
            latlon_bbox=latlon_bbox,
            scale_factor=zone.scale_factor,
            gdf_buildings=source.gdf_buildings,
            G_roads=source.G_roads,
            gdf_water=source.gdf_water,
            global_center=global_center,
            reference_xy_m=zone.reference_xy_m,
        )
        print(f"[OK] Terrain-only task {task_id} completed. File: {terrain_only_result.export_result.output_file_abs}")
        return FullGenerationPipelineResult(
            output_file_abs=terrain_only_result.export_result.output_file_abs,
            primary_format=terrain_only_result.export_result.primary_format,
            terrain_only_result=terrain_only_result,
        )

    stage_start = time.perf_counter()
    canonical_2d_stage = prepare_canonical_2d_stage(
        task_id=task_id,
        request=request,
        source=source,
        zone=zone,
        global_center=global_center,
        debug_generated_dir=(output_dir.parent / "debug" / "generated"),
        zone_prefix=zone_prefix,
    )
    canonical_mask_bundle = canonical_2d_stage.canonical_mask_bundle
    _log_stage("canonical_2d", stage_start)
    if stage_snapshot_collector is not None:
        try:
            stage_snapshot_collector.capture_canonical(canonical_mask_bundle)
        except Exception as exc:
            print(f"[WARN] {zone_prefix}Stage snapshot failed at canonical_2d: {exc}")

    stage_start = time.perf_counter()
    terrain_stage = _run_terrain_stage(
        task=task,
        request=request,
        latlon_bbox=latlon_bbox,
        global_center=global_center,
        zone_prefix=zone_prefix,
        zone=zone,
        source=source,
        min_printable_gap_mm=min_printable_gap_mm,
        groove_clearance_mm=groove_clearance_mm,
        canonical_mask_bundle=canonical_mask_bundle,
    )
    _log_stage("terrain_stage", stage_start)
    if stage_snapshot_collector is not None:
        try:
            stage_snapshot_collector.capture_terrain_stage(terrain_stage)
        except Exception as exc:
            print(f"[WARN] {zone_prefix}Stage snapshot failed at terrain_stage: {exc}")

    water_geoms_for_bridges = prepare_bridge_water_geometries(
        request=request,
        gdf_water=source.gdf_water,
        zone_prefix=zone_prefix,
    )

    stage_start = time.perf_counter()
    detail_layers = process_detail_layers(
        task=task,
        request=request,
        scale_factor=zone.scale_factor,
        terrain_provider=terrain_stage.terrain_provider,
        terrain_mesh=terrain_stage.terrain_mesh,
        global_center=global_center,
        G_roads=source.G_roads,
        water_geoms_for_bridges=water_geoms_for_bridges,
        road_width_multiplier_effective=zone.road_width_multiplier_effective,
        zone_polygon_local=zone.zone_polygon_local,
        building_union_local=terrain_stage.building_union_local,
        merged_roads_geom_local=terrain_stage.merged_roads_geom_local,
        road_cut_mask=terrain_stage.road_cut_mask,
        road_height_m=terrain_stage.road_height_m,
        road_embed_m=terrain_stage.road_embed_m,
        stl_extra_embed_m=zone.stl_extra_embed_m,
        gdf_buildings_local=terrain_stage.gdf_buildings_local,
        gdf_water=terrain_stage.gdf_water_local,
        water_depth_m=terrain_stage.water_depth_m,
        gdf_green=source.gdf_green,
        groove_clearance_mm=groove_clearance_mm,
        zone_prefix=zone_prefix,
        canonical_mask_bundle=canonical_mask_bundle,
    )
    _log_stage("detail_layers", stage_start)
    if stage_snapshot_collector is not None:
        try:
            stage_snapshot_collector.capture_detail_stage(detail_layers)
        except Exception as exc:
            print(f"[WARN] {zone_prefix}Stage snapshot failed at detail_layers: {exc}")
    _validate_canonical_mask_handoff(
        canonical_mask_bundle=canonical_mask_bundle,
        terrain_stage=terrain_stage,
        detail_layers=detail_layers,
        zone_prefix=zone_prefix,
    )
    _validate_groove_stage(
        detail_layers=detail_layers,
        task=task,
        zone_prefix=zone_prefix,
        require_success=require_groove_success,
    )

    terrain_mesh = detail_layers.terrain_mesh
    road_mesh = detail_layers.road_mesh
    building_meshes = detail_layers.building_meshes
    water_mesh = detail_layers.water_mesh
    parks_mesh = detail_layers.parks_mesh

    stage_start = time.perf_counter()
    postprocess_result = postprocess_generated_meshes(
        task=task,
        request=request,
        scale_factor=zone.scale_factor,
        terrain_mesh=terrain_mesh,
        road_mesh=road_mesh,
        building_meshes=building_meshes,
        water_mesh=water_mesh,
        parks_mesh=parks_mesh,
    )
    _log_stage("postprocess_meshes", stage_start)
    if stage_snapshot_collector is not None:
        try:
            stage_snapshot_collector.capture_postprocess_stage(postprocess_result)
        except Exception as exc:
            print(f"[WARN] {zone_prefix}Stage snapshot failed at postprocess: {exc}")
    terrain_mesh = postprocess_result.terrain_mesh
    road_mesh = postprocess_result.road_mesh
    building_meshes = postprocess_result.building_meshes
    water_mesh = postprocess_result.water_mesh
    parks_mesh = postprocess_result.parks_mesh

    if not request.is_ams_mode and zone.scale_factor and zone.scale_factor > 0:
        try:
            print_road_groove_validation_report(
                road_mesh=road_mesh,
                terrain_mesh=terrain_mesh,
                road_polygons=getattr(detail_layers, "road_groove_mask", None) or terrain_stage.road_cut_mask,
                scale_factor=float(zone.scale_factor),
                groove_clearance_mm=groove_clearance_mm,
                zone_prefix=zone_prefix,
            )
        except Exception as exc:
            print(f"[WARN] {zone_prefix} Road/groove validation report failed: {exc}")

    task.update_status("processing", 80, "Обрізання мешів по bbox...")
    stage_start = time.perf_counter()
    clip_result = clip_generated_meshes(
        terrain_mesh=terrain_mesh,
        road_mesh=road_mesh,
        building_meshes=building_meshes,
        water_mesh=water_mesh,
        parks_mesh=parks_mesh,
        bbox_meters=zone.bbox_meters,
        zone_polygon_coords=zone_polygon_coords,
        global_center=global_center,
        preclipped_to_zone=terrain_stage.preclipped_to_zone,
        clip_tolerance=0.1,
    )
    _log_stage("clip_meshes", stage_start)
    if stage_snapshot_collector is not None:
        try:
            stage_snapshot_collector.capture_clip_stage(clip_result)
        except Exception as exc:
            print(f"[WARN] {zone_prefix}Stage snapshot failed at clip: {exc}")
    terrain_mesh = clip_result.terrain_mesh
    road_mesh = clip_result.road_mesh
    building_meshes = clip_result.building_meshes
    water_mesh = clip_result.water_mesh
    parks_mesh = clip_result.parks_mesh

    stage_start = time.perf_counter()
    merge_result = merge_terrain_and_buildings(
        terrain_mesh=terrain_mesh,
        building_meshes=building_meshes,
        merged_building_mesh=union_mesh_collection(building_meshes, label="clipped_building_layer"),
        support_meshes=detail_layers.support_meshes,
    )
    _log_stage("merge_terrain_buildings", stage_start)
    if stage_snapshot_collector is not None:
        try:
            stage_snapshot_collector.capture_merge_stage(merge_result)
        except Exception as exc:
            print(f"[WARN] {zone_prefix}Stage snapshot failed at merge: {exc}")
    terrain_mesh = merge_result.terrain_mesh
    building_meshes = merge_result.building_meshes

    task.update_status("processing", 82, "Експорт моделі...")
    stage_start = time.perf_counter()
    export_result = export_generation_outputs(
        task=task,
        request=request,
        task_id=task_id,
        output_dir=output_dir,
        terrain_mesh=terrain_mesh,
        road_mesh=road_mesh,
        building_meshes=building_meshes,
        water_mesh=water_mesh,
        parks_mesh=parks_mesh,
        reference_xy_m=zone.reference_xy_m,
    )
    _log_stage("export_outputs", stage_start)
    if stage_snapshot_collector is not None:
        try:
            stage_snapshot_collector.capture_export_stage(export_result)
        except Exception as exc:
            print(f"[WARN] {zone_prefix}Stage snapshot failed at export: {exc}")
        # Finalize manifest eagerly so it persists even if the downstream
        # print_acceptance gate rejects the bundle. The manifest is the primary
        # debug artefact — we never want to lose it to a slicer-level failure.
        try:
            stage_snapshot_manifest_path = stage_snapshot_collector.finalize()
            task.set_output("stage_snapshots_manifest", str(stage_snapshot_manifest_path.resolve()))
            print(f"[DEBUG] {zone_prefix}Stage snapshots manifest: {stage_snapshot_manifest_path}")
        except Exception as exc:
            print(f"[WARN] {zone_prefix}Failed to finalize stage snapshots: {exc}")

    stage_start = time.perf_counter()
    printer_profile = get_printer_profile_for_request(request)
    parts_for_print = _collect_print_part_paths(task, export_result)
    expected_parts = {
        "base": terrain_mesh is not None,
        "roads": road_mesh is not None,
        "parks": parks_mesh is not None,
        "water": water_mesh is not None,
        "buildings": bool(building_meshes),
    }
    print_acceptance_path = write_export_print_acceptance_report(
        task_id=task_id,
        output_dir=output_dir,
        parts_for_print=parts_for_print,
        expected_parts=expected_parts,
        printer_profile=printer_profile,
        require_slicer_validation=True,
        fail_on_slicer_warnings=True,
        rotate_x_deg=0,
    )
    task.set_output("print_acceptance", str(print_acceptance_path.resolve()))
    print_acceptance_report = json.loads(print_acceptance_path.read_text(encoding="utf-8"))
    if print_acceptance_report.get("status") != "pass":
        print_acceptance_report, parts_for_print, expected_parts = _attempt_print_recovery(
            task_id=task_id,
            output_dir=output_dir,
            task=task,
            parts_for_print=parts_for_print,
            expected_parts=expected_parts,
            printer_profile=printer_profile,
            initial_report=print_acceptance_report,
        )
    if print_acceptance_report.get("status") != "pass":
        raise RuntimeError(summarize_export_print_failures(print_acceptance_report))
    _log_stage("print_acceptance", stage_start)

    stage_start = time.perf_counter()
    try:
        debug_bundle_dir = create_debug_bundle(
            task_id=task_id,
            request=request,
            output_dir=output_dir,
            zone=zone,
            source=source,
            terrain_stage=terrain_stage,
            detail_layers=detail_layers,
            postprocess_result=postprocess_result,
            clip_result=clip_result,
            merge_result=merge_result,
            export_result=export_result,
            global_center=global_center,
            canonical_mask_bundle=canonical_mask_bundle,
        )
        if debug_bundle_dir is not None:
            print(f"[DEBUG] Debug bundle created: {debug_bundle_dir}")
    except Exception as exc:
        print(f"[WARN] Failed to create debug bundle for {task_id}: {exc}")
    _log_stage("debug_bundle", stage_start)

    print("[INFO] Running garbage collection before upload...")
    gc.collect()
    stage_start = time.perf_counter()
    publish_outputs_to_firebase(
        task=task,
        output_file_abs=export_result.output_file_abs,
        primary_format=export_result.primary_format,
    )
    _log_stage("firebase_publish", stage_start)
    print(f"[TIMING] {zone_prefix}full_generation_pipeline total: {time.perf_counter() - pipeline_start:.2f}s")

    return FullGenerationPipelineResult(
        output_file_abs=export_result.output_file_abs,
        primary_format=export_result.primary_format,
        terrain_only_result=None,
    )
