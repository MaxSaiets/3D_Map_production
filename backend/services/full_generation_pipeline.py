from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import trimesh
from shapely.geometry import box

from services.canonical_2d_pipeline import prepare_canonical_2d_stage
from services.bridge_water_pipeline import prepare_bridge_water_geometries
from services.building_supports import union_mesh_collection
from services.canonical_mask_bundle import load_canonical_mask_bundle
from services.data_fetch_pipeline import fetch_generation_data
from services.debug_bundle_pipeline import create_debug_bundle
from services.detail_layer_pipeline import process_detail_layers
from services.export_pipeline import export_generation_outputs
from services.flat_plate_pipeline import run_flat_plate_pipeline
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
        msg = "Canonical 2D -> 3D handoff drift detected: " + "; ".join(problems[:8])
        # Strict by default in production; can be relaxed for asset/showcase
        # generation (e.g. flat preview maps) where minor 2D<->3D mask drift is
        # cosmetic and must not abort the build. Set HANDOFF_DRIFT_STRICT=0.
        import os as _os
        if _os.getenv("HANDOFF_DRIFT_STRICT", "1") != "0":
            raise RuntimeError(msg)
        print(f"[WARN] {zone_prefix}{msg} (non-fatal: HANDOFF_DRIFT_STRICT=0)")
    else:
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

    _rec_strict = os.environ.get("PRINT_QA_STRICT", "").lower() in ("1", "true", "yes", "on")
    recovery_path = write_export_print_acceptance_report(
        task_id=task_id,
        output_dir=output_dir,
        parts_for_print=normalized_parts,
        expected_parts=adjusted_expected,
        printer_profile=printer_profile,
        require_slicer_validation=_rec_strict,
        fail_on_slicer_warnings=_rec_strict,
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
) -> None:
    # Preview mode runs with BOOLEAN_BACKEND=noop on purpose — grooves stay
    # uncut by design, so the "did the boolean actually change terrain?"
    # check must not fail the task. Just log and return.
    if os.environ.get("PREVIEW_MODE", "").lower() in ("1", "true", "yes"):
        print(f"[INFO] {zone_prefix}PREVIEW_MODE: skipping groove validation (noop backend)")
        return
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

    # GRACEFUL FALLBACK на крутому рельєфі (гори): boolean-груви можуть не
    # врізатись / бути відхиленими через катастрофічний z-зсув на дуже високому
    # перепаді. Замість того щоб ВАЛИТИ всю модель — деградуємо до рельєфу без
    # рецесованих грувів (поверхня лишається валідною, дороги просто не втоплені).
    # Вимкнути (суворий режим) можна GROOVE_FALLBACK_ON_FAIL=0.
    _groove_graceful = os.getenv("GROOVE_FALLBACK_ON_FAIL", "1").lower() in ("1", "true", "yes")

    if bool(getattr(groove_result, "rejected", False)):
        reason = getattr(groove_result, "rejection_reason", None) or getattr(groove_result, "failure_reason", None) or "unknown_rejection"
        message = f"Groove stage failed: unsafe groove cut was rejected ({reason})"
        if _groove_graceful:
            print(f"[WARN] {zone_prefix}{message} — продовжуємо БЕЗ грувів (рельєф зберігається, дороги не втоплені)")
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
        if _groove_graceful:
            print(f"[WARN] {zone_prefix}{message} — продовжуємо БЕЗ грувів (рельєф зберігається)")
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
    zone_polygon_local = zone_geometry.zone_polygon_local
    if zone_polygon_local is None or getattr(zone_polygon_local, "is_empty", True):
        try:
            minx, miny, maxx, maxy = zone_context.bbox_meters
            zone_polygon_local = box(float(minx), float(miny), float(maxx), float(maxy))
            print(f"[DEBUG] {zone_prefix} Built local bbox zone polygon for single-area generation")
        except Exception as exc:
            print(f"[WARN] {zone_prefix} Failed to build bbox zone polygon: {exc}")
            zone_polygon_local = zone_geometry.zone_polygon_local
    scale_factor = zone_context.scale_factor
    return ZonePreparationResult(
        zone_polygon_local=zone_polygon_local,
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
    # Drop tiny kiosk/stall buildings (OSM market kiosks, ларки) by real AREA.
    # They pass the min-WIDTH filter (one side can be 5-8m) but have a tiny area,
    # so on the print they stand as thin "pillars" in the park/street. A real
    # building is hundreds of m2; a kiosk is < ~30m2. Filtering at the source
    # removes them from BOTH the masks and the 3D mesh. Env: BUILDING_MIN_AREA_M2
    # (default 25; 0 disables).
    _gdf_b = data_result.gdf_buildings
    try:
        _min_area = float(os.environ.get("BUILDING_MIN_AREA_M2", "25"))
    except (TypeError, ValueError):
        _min_area = 25.0
    if _min_area > 0 and _gdf_b is not None and not getattr(_gdf_b, "empty", True):
        try:
            import math as _math

            _is_geographic = bool(getattr(getattr(_gdf_b, "crs", None), "is_geographic", False))
            _keep = []
            for _g in _gdf_b.geometry.values:
                if _g is None or getattr(_g, "is_empty", True):
                    _keep.append(True)
                    continue
                try:
                    if _is_geographic:
                        _lat = _g.centroid.y
                        _m2 = _g.area * (111320.0 ** 2) * _math.cos(_math.radians(_lat))
                    else:
                        _m2 = _g.area
                    _keep.append(_m2 >= _min_area)
                except Exception:
                    _keep.append(True)
            _dropped = sum(1 for _k in _keep if not _k)
            if _dropped:
                _gdf_b = _gdf_b[_keep]
                print(
                    f"[BUILDINGS] dropped {_dropped} tiny (<{_min_area:.0f} m2) "
                    f"kiosk/stall buildings"
                )
        except Exception as _exc:
            print(f"[BUILDINGS] kiosk-area filter failed: {_exc}")

    # Drop SMALL buildings that sit (almost) entirely INSIDE a green park. Buildings
    # share the base/ivory colour, so a small building inside the green renders as a
    # pale "white blob in the park" (owner complaint). A building whose footprint is
    # >= PARK_BUILDING_DROP_RATIO inside the green AND below PARK_BUILDING_DROP_M2 is
    # removed at the source, so the green reads solid (no carved hole, no speck). Large
    # structures (a pavilion/palace genuinely in a park) stay because of the area cap.
    # Env: PARK_BUILDING_DROP_M2 (default 250 m2 world; 0 = off — owner-tunable:
    #      raise to drop larger in-park structures, lower to keep more),
    #      PARK_BUILDING_DROP_RATIO (default 0.85).
    try:
        _park_drop_m2 = float(os.environ.get("PARK_BUILDING_DROP_M2", "250"))
    except (TypeError, ValueError):
        _park_drop_m2 = 0.0
    _grn = data_result.gdf_green
    # Only REAL parks count for the in-park drop. A building inside grass / meadow /
    # forest / cemetery / allotments / orchard is legitimate (a chapel, a shed, a
    # forest cabin) and must NOT be removed — those broad landuse polygons are where
    # using ALL green over-dropped real buildings. Restrict to park-like OSM types via
    # the green `type` column; if it's absent, fall back to all green (old behaviour).
    # Park-LIKE types only (NOT grass/meadow/forest/cemetery/allotments/orchard — a
    # building inside those is legitimate, e.g. a chapel/shed/cabin). 'garden' is
    # included but the green polygon AREA cap below (PARK_MAX_AREA_M2) protects against
    # mis-tagged huge "garden" polygons (seen: 20-hectare gardens over residential
    # blocks) — a building inside a manicured park/square is a visible blob, but a
    # building inside a landuse-scale polygon is a normal city block, never dropped.
    _PARK_TYPES = {
        "park", "garden", "recreation_ground", "playground",
        "village_green", "common", "dog_park",
    }
    try:
        _park_max_area = float(os.environ.get("PARK_MAX_AREA_M2", "50000"))
    except (TypeError, ValueError):
        _park_max_area = 50000.0
    if (
        _park_drop_m2 > 0
        and _gdf_b is not None and not getattr(_gdf_b, "empty", True)
        and _grn is not None and not getattr(_grn, "empty", True)
    ):
        try:
            import math as _math
            from shapely.ops import unary_union as _unary_union
            try:
                _drop_ratio = float(os.environ.get("PARK_BUILDING_DROP_RATIO", "0.85"))
            except (TypeError, ValueError):
                _drop_ratio = 0.85
            _is_geo2 = bool(getattr(getattr(_gdf_b, "crs", None), "is_geographic", False))

            def _world_m2(g):
                if _is_geo2:
                    return g.area * (111320.0 ** 2) * _math.cos(_math.radians(g.centroid.y))
                return g.area

            _grn_for_drop = _grn
            try:
                if "type" in list(getattr(_grn, "columns", [])):
                    _ptype = _grn["type"].astype(str).str.strip().str.lower()
                    _pmask = _ptype.isin(_PARK_TYPES)
                    if bool(_pmask.any()):
                        _grn_for_drop = _grn[_pmask]
            except Exception:
                _grn_for_drop = _grn
            # Only park-like polygons SMALL ENOUGH to be a manicured park/square — a
            # building inside a huge (mis-tagged) green polygon is a normal block.
            _green_u = _unary_union(
                [g for g in _grn_for_drop.geometry.values
                 if g is not None and not getattr(g, "is_empty", True)
                 and _world_m2(g) <= _park_max_area]
            ).buffer(0)

            _keep_park = []
            for _g in _gdf_b.geometry.values:
                if _g is None or getattr(_g, "is_empty", True) or _green_u.is_empty:
                    _keep_park.append(True)
                    continue
                try:
                    if _g.area <= 0 or not _g.intersects(_green_u):
                        _keep_park.append(True)
                        continue
                    _frac = _g.intersection(_green_u).area / _g.area
                    _keep_park.append(not (_frac >= _drop_ratio and _world_m2(_g) < _park_drop_m2))
                except Exception:
                    _keep_park.append(True)
            _dropped_park = sum(1 for _k in _keep_park if not _k)
            try:
                _gtot = len(_grn); _gpark = len(_grn_for_drop)
            except Exception:
                _gtot = _gpark = -1
            if _dropped_park:
                _gdf_b = _gdf_b[_keep_park]
                print(
                    f"[BUILDINGS] dropped {_dropped_park} small (<{_park_drop_m2:.0f} m2) "
                    f"buildings >={_drop_ratio:.0%} inside a park "
                    f"(park-like green {_gpark}/{_gtot} polys)"
                )
            else:
                print(f"[BUILDINGS] park-drop: 0 dropped (park-like green {_gpark}/{_gtot} polys)")
        except Exception as _exc:
            print(f"[BUILDINGS] park-building drop failed: {_exc}")

    return SourceDataResult(
        gdf_buildings=_gdf_b,
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


def cut_relief_connector_notch(terrain_mesh, *, zone, request, sf_c, zone_prefix=""):
    """Ріже конектор-паз (ластівчин хвіст) у ДНІ рельєф-мешу + повертає ключ-метелик.
    ВИКЛИКАТИ на ГЕРМЕТИЧНОМУ (ДО-грувовому) рельєфі → manifold ріже чисто (drift≈0).
    Паз — у дні, груви — на версі → не конфліктують. Раніше різали ПІСЛЯ грувів на
    негерметичному мешеві → boolean руйнував/відкидав («рельєф: нема зʼєднання»).
    Повертає (terrain_mesh, connector_key_mesh, carved)."""
    if not getattr(request, "map_connector", False) or terrain_mesh is None or sf_c <= 0:
        return terrain_mesh, None, False
    try:
        from services.flat_plate_pipeline import (
            build_map_connector_geometry, build_flat_layer_mesh_from_mask,
            parse_connector_azimuths,
        )
        import trimesh as _t
        floor_z = float(terrain_mesh.bounds[0][2])
        model_h = float(terrain_mesh.bounds[1][2]) - floor_z
        depth_mm = float(getattr(request, "map_connector_depth_mm", 2.0) or 2.0)
        depth_m = min(depth_mm / sf_c, max(model_h * 0.6, 0.0))
        ntc, keyc = build_map_connector_geometry(
            zone.zone_polygon_local,
            edges=str(getattr(request, "map_connector_edges", "NSEW") or "NSEW"),
            span_mm=float(getattr(request, "map_connector_span_mm", 10.0) or 10.0),
            length_mm=float(getattr(request, "map_connector_length_mm", 15.0) or 15.0),
            waist_frac=0.5,
            clearance_mm=float(getattr(request, "map_connector_clearance_mm", 0.03) or 0.03),
            export_scale_factor=sf_c,
            key_edges=(str(getattr(request, "map_connector_key_edges", "") or "") or None),
            edge_dirs=parse_connector_azimuths(getattr(request, "map_connector_edge_az", "")),
            key_dirs=parse_connector_azimuths(getattr(request, "map_connector_key_az", "")),
        )
        if ntc is None or depth_m <= 1e-6:
            return terrain_mesh, None, False
        eps = max(model_h * 0.01, 0.5 / sf_c)
        cutter = build_flat_layer_mesh_from_mask(
            ntc, bottom_z_m=floor_z - eps, thickness_m=depth_m + eps,
            color=[128, 128, 128], min_area_m2=1e-12,
        )
        if cutter is None or not bool(getattr(cutter, "is_volume", False)):
            print(f"[CONNECTOR] {zone_prefix}cutter not a volume — skip pre-groove notch")
            return terrain_mesh, None, False
        b0 = terrain_mesh.bounds
        faces0 = len(getattr(terrain_mesh, "faces", []))
        isv = bool(getattr(terrain_mesh, "is_volume", False))
        print(f"[CONNECTOR] {zone_prefix}pre-groove terrain is_volume={isv}")
        cutc, via = None, None
        if isv:
            try:
                cutc = _t.boolean.difference([terrain_mesh, cutter], engine="manifold")
                via = "manifold"
            except Exception as e:
                print(f"[CONNECTOR] {zone_prefix}manifold notch failed ({e})")
                cutc = None
        if cutc is None or len(getattr(cutc, "faces", [])) == 0:
            try:
                import trimesh.repair as _rr
                rep = terrain_mesh.copy()
                rep.merge_vertices()
                try:
                    rep.update_faces(rep.nondegenerate_faces())
                    rep.update_faces(rep.unique_faces())
                    rep.remove_unreferenced_vertices()
                except Exception:
                    pass
                for _ in range(3):
                    if bool(getattr(rep, "is_volume", False)):
                        break
                    rep.fill_holes()
                    try:
                        _rr.fix_winding(rep)
                    except Exception:
                        pass
                try:
                    rep.fix_normals()
                except Exception:
                    pass
                if bool(getattr(rep, "is_volume", False)):
                    cutc = _t.boolean.difference([rep, cutter], engine="manifold")
                    via = "repair+manifold"
            except Exception as e:
                print(f"[CONNECTOR] {zone_prefix}repair+manifold notch failed ({e})")
        if cutc is None or len(getattr(cutc, "faces", [])) == 0:
            print(f"[CONNECTOR] {zone_prefix}pre-groove notch boolean produced nothing — skip")
            return terrain_mesh, None, False
        # ЛЕГКА очистка: manifold-difference вже дає ГЕРМЕТИЧНИЙ том із ЧИСТИМ
        # пазом (вертикальні стінки пласкі, виїмка-метелик рівна). Тут лише
        # зливаємо дублі-вершини та прибираємо ВИРОДЖЕНІ (нульової площі) грані —
        # БЕЗ порогу висоти. (Старий код різав nondegenerate_faces(height=0.9/sf):
        # це видаляло ВСІ дрібні грід-трикутники рельєфу → лишалась 40-гранна
        # «шкаралупа» з барами/стінками. САМЕ це бачив користувач як «гребінець».)
        try:
            cutc.merge_vertices()
            cutc.update_faces(cutc.unique_faces())
            cutc.update_faces(cutc.nondegenerate_faces())  # лише нульова площа
            cutc.remove_unreferenced_vertices()
            cutc.fix_normals()
        except Exception as _ce:
            print(f"[CONNECTOR] {zone_prefix}notch cleanup skipped (non-fatal): {_ce}")
        b1 = cutc.bounds
        drift = (max(abs(b1[0][i] - b0[0][i]) for i in range(2))
                 + max(abs(b1[1][i] - b0[1][i]) for i in range(2)))
        changed = abs(len(cutc.faces) - faces0) > 0
        ext0 = (float(b0[1][0] - b0[0][0]), float(b0[1][1] - b0[0][1]))
        ext1 = (float(b1[1][0] - b1[0][0]), float(b1[1][1] - b1[0][1]))
        ext_ok = ((ext0[0] <= 1e-6 or ext1[0] >= 0.6 * ext0[0])
                  and (ext0[1] <= 1e-6 or ext1[1] >= 0.6 * ext0[1]))
        if not (drift < (5.0 / sf_c) and changed and ext_ok):
            print(f"[CONNECTOR] {zone_prefix}pre-groove notch rejected "
                  f"(via={via} drift={drift:.1f} changed={changed} ext_ok={ext_ok})")
            return terrain_mesh, None, False
        key_mesh = None
        if keyc is not None and not getattr(keyc, "is_empty", True):
            clip_h = max(min(1.7 / sf_c, depth_m), 0.4 / sf_c)
            key_mesh = build_flat_layer_mesh_from_mask(
                keyc, bottom_z_m=floor_z, thickness_m=clip_h,
                color=[242, 242, 242], min_area_m2=1e-12,
            )
        print(f"[CONNECTOR] {zone_prefix}PRE-GROOVE dovetail notch carved "
              f"({via}, depth {depth_m * sf_c:.2f}mm, drift {drift:.2f})")
        return cutc, key_mesh, True
    except Exception as e:
        print(f"[CONNECTOR] {zone_prefix}pre-groove connector failed (non-fatal): {e}")
        return terrain_mesh, None, False


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

    # КОНТЕНТ (будівлі/вода) обрізаємо до ПОВНОЇ зони плитки (як рельєф-підложку) —
    # будинки доходять до самого краю, НЕ зрізаються всередину. (Раніше тут був 5мм-
    # інсет «щоб не було стінок на шві», але він РІЗАВ будинки біля країв → погано.
    # Чистий край тепер робить edge-flush для доріг/води/парків, а будинки лишаємо
    # цілими до межі.)
    _preclip_zone = zone.zone_polygon_local
    preclip_result = prepare_preclipped_geometry(
        gdf_buildings_local=building_geometry.gdf_buildings_local,
        building_geometries_for_flatten=building_geometry.building_geometries_for_flatten,
        gdf_water=source.gdf_water,
        global_center=global_center,
        zone_polygon_local=_preclip_zone,
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
        semantic_centerlines_local=getattr(road_geometry, "semantic_centerlines_local", None),
    )


def _flush_raised_content_at_tile_edge(
    meshes,
    *,
    terrain_provider,
    zone_polygon_local,
    scale_factor,
    edge_mm: float = 2.5,
    zone_prefix: str = "",
):
    """СЕРІЯ: підняте покриття (дороги/вода/парки) сидить на ~0.6–1.3мм ВИЩЕ рельєфу.
    Коли його ріже по краю плитки — торець зрізу стирчить угору над рельєфом = «стінки/
    зубці» на шві (річка — найгірша). Тут ОПУСКАЄМО верх контенту до поверхні рельєфу
    у тонкому кільці (~edge_mm) біля межі зони: контент ДОХОДИТЬ до краю, але ВРІВЕНЬ
    (без торців-стінок). Внутрішня (піднята) частина не чіпається. Не критично —
    при будь-якій помилці лишаємо меш як є.
    """
    try:
        import numpy as np
        if terrain_provider is None or zone_polygon_local is None:
            return
        if not scale_factor or float(scale_factor) <= 0:
            return
        edge_world = float(edge_mm) / float(scale_factor)
        inner = zone_polygon_local.buffer(-edge_world).buffer(0)
        if inner is None or inner.is_empty:
            return
        drop_world = 0.3 / float(scale_factor)  # верх контенту трохи НИЖЧЕ рельєфу на краю → без z-fight і без торця
        _cxy = None
        try:
            from shapely import contains_xy as _cxy  # shapely 2.x — векторизовано
        except Exception:
            _cxy = None
        inner_prep = None
        if _cxy is None:
            from shapely.prepared import prep as _prep
            from shapely.geometry import Point as _Pt
            inner_prep = _prep(inner)
        for mesh in meshes:
            if mesh is None or not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
                continue
            v = np.asarray(mesh.vertices, dtype=float)
            xy = v[:, :2]
            try:
                tz = np.asarray(terrain_provider.get_heights_for_points(xy), dtype=float)
            except Exception as _sx:
                print(f"[CONNECTOR] {zone_prefix}edge-flush: terrain sample failed: {_sx}")
                continue
            if _cxy is not None:
                inside = np.asarray(_cxy(inner, xy[:, 0], xy[:, 1]), dtype=bool)
            else:
                inside = np.array([inner_prep.contains(_Pt(float(x), float(y))) for x, y in xy], dtype=bool)
            # near = у кільці біля межі; above = верхні вершини що стирчать над рельєфом
            sel = (~inside) & (v[:, 2] > tz) & np.isfinite(tz)
            if np.any(sel):
                v[sel, 2] = tz[sel] - drop_world
                mesh.vertices = v
                print(f"[CONNECTOR] {zone_prefix}edge-flush: опущено {int(np.sum(sel))} вершин контенту до рельєфу на краю плитки")
    except Exception as _exc:
        print(f"[CONNECTOR] {zone_prefix}edge-flush skipped (non-fatal): {_exc}")


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
    file_basename: Optional[str] = None,
) -> FullGenerationPipelineResult:
    pipeline_start = time.perf_counter()
    stage_snapshot_collector = None
    stage_snapshot_manifest_path: Optional[Path] = None
    # Stage snapshots are DEBUG artifacts (per-stage captures written to debug/).
    # On real OSM data each capture is expensive (the clip-stage capture alone
    # measured ~170s), so they are OFF by default and opt-in via PIPELINE_DEBUG=1.
    # Always skipped in preview mode.
    _pipeline_debug = os.environ.get("PIPELINE_DEBUG", "").lower() in ("1", "true", "yes")
    if os.environ.get("PREVIEW_MODE", "").lower() in ("1", "true", "yes"):
        print(f"[INFO] {zone_prefix}PREVIEW_MODE: skipping stage snapshots")
    elif not _pipeline_debug:
        print(f"[INFO] {zone_prefix}Stage snapshots OFF (set PIPELINE_DEBUG=1 to enable)")
    else:
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

    task.update_status("processing", 15, "Завантажую дані OSM (дороги, будівлі, вода)...")

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

    if bool(getattr(request, "flat_plate_mode", False)):
        print(f"[INFO] {zone_prefix}FLAT_PLATE_MODE: skipping DEM terrain, grooves, inlays, and print-fit gates")
        stage_start = time.perf_counter()
        export_result = run_flat_plate_pipeline(
            task=task,
            request=request,
            task_id=task_id,
            output_dir=output_dir,
            zone=zone,
            source=source,
            canonical_2d_stage=canonical_2d_stage,
            global_center=global_center,
            file_basename=file_basename,
        )
        _log_stage("flat_plate_pipeline", stage_start)
        return FullGenerationPipelineResult(
            output_file_abs=export_result.output_file_abs,
            primary_format=export_result.primary_format,
        )

    task.update_status("processing", 20, "Будую рельєф місцевості (~3-5 хв)...")
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

    # ── ЗʼЄДНУВАЧ-ПАЗ у ДНІ рельєфу — ДО грувів (рельєф ще ГЕРМЕТИЧНИЙ → manifold
    # ріже чисто, drift≈0). Паз у дні, груви потім на версі → не конфліктують.
    # Раніше різали ПІСЛЯ грувів на негерметичному → boolean руйнував меш. Ключ-
    # метелик зберігаємо у connector_key_mesh; late-блок пропускається якщо вдалось.
    connector_key_mesh = None
    _relief_connector_done = False
    _conn_terrain = terrain_stage.terrain_mesh
    if getattr(request, "map_connector", False):
        # Коли CDT-грувы АКТИВНІ для цього тайла — паз НЕ ріжемо тут (до грувів): CDT
        # перебудовує терен з нуля й зітре паз. Натомість ПІЗНІЙ блок (нижче) вріже паз
        # у ЧИСТИЙ герметичний CDT-терен (manifold, drift≈0) → стінки паза чисті, а грувы
        # доріг/парків/води теж чисті (CDT) замість boolean-комба. Коли CDT вимкнено
        # (CDT_GROOVE off — як на проді) або зона завелика → старий надійний pre-groove.
        import os as _osc
        _cdt_on = _osc.environ.get("CDT_GROOVE", "").strip().lower() in ("1", "true", "yes", "on")
        _zarea_c = float(getattr(zone.zone_polygon_local, "area", 0.0) or 0.0) \
            if getattr(zone, "zone_polygon_local", None) is not None else 0.0
        _cdt_conn_elig = (_cdt_on and 0.0 < _zarea_c <= 2.0e6
                          and terrain_stage.terrain_mesh is not None)
        if _cdt_conn_elig:
            print(f"[CONNECTOR] {zone_prefix}CDT active → відкладаю паз на POST-groove "
                  f"(чисті стінки; area={_zarea_c:.0f})")
        else:
            _conn_terrain, connector_key_mesh, _relief_connector_done = cut_relief_connector_notch(
                terrain_stage.terrain_mesh,
                zone=zone,
                request=request,
                sf_c=float(getattr(zone, "scale_factor", 0.0) or 0.0),
                zone_prefix=zone_prefix,
            )

    water_geoms_for_bridges = prepare_bridge_water_geometries(
        request=request,
        gdf_water=source.gdf_water,
        zone_prefix=zone_prefix,
    )

    task.update_status("processing", 40, "Генерую дороги, воду, будівлі...")
    stage_start = time.perf_counter()
    # ДОРОГИ/парки теж обрізаємо до ПОВНОЇ зони (доходять до краю). Тонкі торці-стінки
    # на краю прибирає edge-flush (опускає верх до рельєфу в кільці біля межі), а не
    # інсет — щоб контент НЕ зникав біля країв.
    _content_zone = zone.zone_polygon_local
    detail_layers = process_detail_layers(
        task=task,
        request=request,
        scale_factor=zone.scale_factor,
        terrain_provider=terrain_stage.terrain_provider,
        terrain_mesh=_conn_terrain,
        global_center=global_center,
        G_roads=source.G_roads,
        water_geoms_for_bridges=water_geoms_for_bridges,
        road_width_multiplier_effective=zone.road_width_multiplier_effective,
        zone_polygon_local=_content_zone,
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
    # Canonical 2D->3D handoff verification does heavy polygon symmetric-difference
    # math (~170s on real OSM) purely to *check* mask parity — it does not change
    # the model. Off by default; enable with PIPELINE_DEBUG=1 or HANDOFF_DRIFT_STRICT=1.
    if (os.environ.get("PIPELINE_DEBUG", "").lower() in ("1", "true", "yes")
            or os.environ.get("HANDOFF_DRIFT_STRICT", "").lower() in ("1", "true", "yes")):
        _validate_canonical_mask_handoff(
            canonical_mask_bundle=canonical_mask_bundle,
            terrain_stage=terrain_stage,
            detail_layers=detail_layers,
            zone_prefix=zone_prefix,
        )
    _validate_groove_stage(detail_layers=detail_layers, task=task, zone_prefix=zone_prefix)

    terrain_mesh = detail_layers.terrain_mesh
    road_mesh = detail_layers.road_mesh
    building_meshes = detail_layers.building_meshes
    water_mesh = detail_layers.water_mesh
    parks_mesh = detail_layers.parks_mesh

    # СЕРІЯ: робимо КРАЙ плитки чистим — підняте покриття (дороги/вода/парки) на
    # краю опускаємо ВРІВЕНЬ з рельєфом, щоб торці зрізу не стирчали як стінки на шві.
    # Контент доходить до краю, але без вертикальних торців-плавників.
    # ВАЖЛИВО: тригер = БУДЬ-ЯКА плитка серії (elevation_ref_m задано — як у preserve_z),
    # а НЕ лише коли увімкнено зʼєднувачі. Власник часто генерує серію БЕЗ пазів →
    # раніше flush не запускався → саме там і стирчали «стінки на краях». Одиночна
    # мапа (ref=None, без конектора) — поведінка незмінна, golden недоторканий.
    _flush_series_edges = (
        bool(getattr(request, "map_connector", False))
        or getattr(request, "elevation_ref_m", None) is not None
    )
    if _flush_series_edges:
        _flush_raised_content_at_tile_edge(
            [road_mesh, water_mesh, parks_mesh],
            terrain_provider=terrain_stage.terrain_provider,
            zone_polygon_local=zone.zone_polygon_local,
            scale_factor=zone.scale_factor,
            edge_mm=2.5,
            zone_prefix=zone_prefix,
        )

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

    # PREVIEW: turn roads / parks / water 3D inlays into thin coloured decals
    # draped on top of terrain. MUST run AFTER postprocess_generated_meshes —
    # postprocess strips "thin mesh components < 0.7mm" which would otherwise
    # delete our flat decal sheets entirely (this happened on the previous
    # render: water=None, parks=None despite OSM providing both).
    if os.environ.get("PREVIEW_MODE", "").lower() in ("1", "true", "yes"):
        from services.terrain_cutter import build_terrain_decal_from_2d_mask, flatten_inlay_to_terrain_decal
        try:
            terrain_provider = getattr(terrain_stage, "terrain_provider", None)
        except Exception:
            terrain_provider = None
        try:
            def _first_nonempty_geometry(*items):
                for item in items:
                    if item is not None and not getattr(item, "is_empty", False):
                        return item
                return None

            def _first_surface_geometry(*items):
                for item in items:
                    if item is None or getattr(item, "is_empty", False):
                        continue
                    if getattr(item, "geom_type", "") in ("Polygon", "MultiPolygon", "GeometryCollection"):
                        return item
                return None

            road_mask = _first_surface_geometry(
                getattr(canonical_mask_bundle, "roads_final", None)
                if canonical_mask_bundle is not None else None,
                getattr(detail_layers, "road_groove_mask", None),
                terrain_stage.road_cut_mask,
            )
            if road_mask is not None or road_mesh is not None:
                rebuilt = build_terrain_decal_from_2d_mask(
                    road_mask,
                    terrain_provider,
                    offset_m=0.02,
                    target_edge_len_m=3.0,
                    simplify_tolerance_m=0.05,
                )
                road_mesh = rebuilt if rebuilt is not None else (
                    flatten_inlay_to_terrain_decal(road_mesh, terrain_provider, offset_m=0.02)
                    if road_mesh is not None else None
                )
            parks_mask = _first_nonempty_geometry(
                getattr(canonical_mask_bundle, "parks_final", None)
                if canonical_mask_bundle is not None else None,
                getattr(getattr(detail_layers, "parks_result", None), "processed_polygons", None),
            )
            if parks_mask is not None or parks_mesh is not None:
                rebuilt = build_terrain_decal_from_2d_mask(
                    parks_mask,
                    terrain_provider,
                    offset_m=0.02,
                    target_edge_len_m=3.0,
                    simplify_tolerance_m=0.05,
                )
                parks_mesh = rebuilt if rebuilt is not None else (
                    flatten_inlay_to_terrain_decal(parks_mesh, terrain_provider, offset_m=0.02)
                    if parks_mesh is not None else None
                )
            water_mask = _first_nonempty_geometry(
                getattr(canonical_mask_bundle, "water_final", None)
                if canonical_mask_bundle is not None else None,
                getattr(detail_layers, "water_cut_polygons", None),
            )
            if water_mask is not None or water_mesh is not None:
                rebuilt = build_terrain_decal_from_2d_mask(
                    water_mask,
                    terrain_provider,
                    offset_m=0.02,
                    target_edge_len_m=3.0,
                    simplify_tolerance_m=0.05,
                )
                water_mesh = rebuilt if rebuilt is not None else (
                    flatten_inlay_to_terrain_decal(water_mesh, terrain_provider, offset_m=0.02)
                    if water_mesh is not None else None
                )
            print(
                f"[INFO] {zone_prefix}PREVIEW_MODE: built roads/parks/water from 2D masks as terrain decals"
            )
        except Exception as exc:
            print(f"[WARN] {zone_prefix}PREVIEW_MODE flatten_inlay failed: {exc}")

    # Road/groove validation report is a DIAGNOSTIC print (~165s on real OSM) that
    # doesn't alter the model. Off by default; enable with PIPELINE_DEBUG=1.
    if (os.environ.get("PIPELINE_DEBUG", "").lower() in ("1", "true", "yes")
            and not request.is_ams_mode and zone.scale_factor and zone.scale_factor > 0):
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

    task.update_status("processing", 75, "Збираю фінальну модель...")
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

    # ── ВИДІЛЕННЯ ДОМУ (РЕЛЬЄФ): ВІДБІР + червона деталь ДО merge ──────────────
    # КРИТИЧНО робити ДО merge_terrain_and_buildings, бо: (а) merge зливає будинки у
    # рельєф і ОБНУЛЯЄ building_meshes → пізніше їх уже нема (червоний дім губився у
    # 3MF попри показ у preview = preview≠print баг); (б) треба ВИКЛЮЧИТИ обраний
    # будинок з merge, інакше він і у сірому рельєфі, і червоною вставкою (дубль).
    # Тут лише ВІДБІР+деталь+запамʼятовування footprint/Z; сам ПАЗ ріжемо у ФІНАЛЬНИЙ
    # terrain_mesh ПІСЛЯ merge. Без кліку — fallback на будинок біля центру зони (як flat).
    highlight_meshes = []
    _hl_pockets = []
    _sf_c = float(getattr(zone, "scale_factor", 0.0) or 0.0)
    if getattr(request, "map_highlight_building", False) and building_meshes and _sf_c > 0:
        try:
            from services.flat_plate_pipeline import (
                _select_highlight_building_index, build_highlight_insert,
            )
            _hl_raw = list(getattr(request, "highlight_points", None) or [])
            _hl_single = getattr(request, "highlight_point", None)
            if not _hl_raw and _hl_single and len(_hl_single) >= 2:
                _hl_raw = [_hl_single]
            _hl_targets = []
            for _hp in _hl_raw[:8]:
                if not _hp or len(_hp) < 2:
                    continue
                try:
                    _ux, _uy = global_center.to_utm(float(_hp[0]), float(_hp[1]))
                    _lx, _ly = global_center.to_local(_ux, _uy)
                    _hl_targets.append((float(_lx), float(_ly)))
                except Exception:
                    pass
            _hl_chosen = []
            for _t in _hl_targets:
                _i = _select_highlight_building_index(building_meshes, target_xy=_t, exclude=set(_hl_chosen))
                if _i is not None and _i not in _hl_chosen:
                    _hl_chosen.append(_i)
            if not _hl_chosen:  # NO-CLICK FALLBACK: будинок біля центру зони (як flat-гілка)
                try:
                    _zc = zone.zone_polygon_local.centroid
                    _i = _select_highlight_building_index(building_meshes, target_xy=(float(_zc.x), float(_zc.y)))
                    if _i is not None:
                        _hl_chosen.append(_i)
                except Exception:
                    pass
            if _hl_chosen:
                from services.flat_plate_pipeline import _mesh_xy_footprint, clip_buildings_around_highlight
                from services.highlight_relief_groove import carve_highlight_groove
                _hl_foots = []
                for _i in _hl_chosen:
                    _bm = building_meshes[_i]
                    _bm_top = float(_bm.bounds[1][2])
                    _bm_bot = float(_bm.bounds[0][2])   # НАЙНИЖЧА точка мешу = пласке дно/підлога пазу
                    try:
                        _foot = _mesh_xy_footprint(_bm)
                    except Exception:
                        _foot = None
                    if _foot is None or getattr(_foot, "is_empty", True):
                        continue
                    _hl_foots.append(_foot)   # для обрізки сусідів з-під вставки
                    # ПАЗ ЯК У ДОРІГ, ЧИСТО: ріжемо ТУТ (рельєф ще ГЕРМЕТИЧНИЙ том, ДО merge)
                    # boolean-ом → рівні стінки (не обривчасті). Будинок СТОЇТЬ на пласкому
                    # дні (доробленому до найнижчої точки мешу). НЕ пізній vertex-set.
                    terrain_mesh, _red = carve_highlight_groove(
                        terrain_mesh, _foot, _bm_top, _bm_bot,
                        scale_factor=_sf_c, clearance_mm=0.15, zone_prefix=zone_prefix,
                    )
                    if _red is not None:
                        highlight_meshes.append(_red)
                # ВИКЛЮЧАЄМО обрані будинки з merge (стають окремою червоною вставкою)
                _hl_cs = set(_hl_chosen)
                building_meshes = [b for j, b in enumerate(building_meshes) if j not in _hl_cs]
                # ОБРІЗАЄМО СУСІДІВ із зони вставки ДО merge — інакше їхня стіна (злита
                # в рельєф) фізично блокує посадку обраної червоної деталі.
                if _hl_foots:
                    try:
                        building_meshes = clip_buildings_around_highlight(
                            building_meshes, _hl_foots,
                            export_scale_factor=_sf_c, label=f"HIGHLIGHT {zone_prefix}",
                        )
                    except Exception as _ncx:
                        print(f"[HIGHLIGHT] {zone_prefix}neighbor clip failed (non-fatal): {_ncx}")
                print(f"[HIGHLIGHT] {zone_prefix}{len(highlight_meshes)} highlight building(s) -> CLEAN pre-merge pocket")
        except Exception as _hexc:
            print(f"[HIGHLIGHT] {zone_prefix}relief highlight select failed (non-fatal): {_hexc}")
            highlight_meshes = []
            _hl_pockets = []

    # ВИЗНАЧНІ МІСЦЯ (церкви/вежі/історичні/музеї) → окрема БРОНЗОВА «Landmark» деталь.
    # Як highlight: вилучаємо їхні будинки ДО boolean-merge (інакше зливаються у рельєф
    # і колір губиться) і віддаємо окремою частиною — export фарбує за назвою у бронзу.
    # Без landmark-даних (старий DB / порожньо) блок no-op → вивід байт-ідентичний.
    landmark_meshes = []
    _landmark_centroids = getattr(detail_layers, "landmark_centroids", None) or []
    if _landmark_centroids and building_meshes:
        try:
            from services.flat_plate_pipeline import _select_highlight_building_index, _mesh_xy_footprint
            from shapely.geometry import Point as _LmPt
            _lm_chosen = []
            for (_cx, _cy) in _landmark_centroids:
                _i = _select_highlight_building_index(building_meshes, target_xy=(_cx, _cy), exclude=set(_lm_chosen))
                if _i is None or _i in _lm_chosen:
                    continue
                # центроїд орієнтира МАЄ лежати ВСЕРЕДИНІ обраного будинку — інакше це
                # nearest-fallback (будинок орієнтира відсутній) → пропускаємо, щоб не
                # пофарбувати у бронзу чужий будинок.
                try:
                    _foot = _mesh_xy_footprint(building_meshes[_i])
                    if _foot is None or getattr(_foot, "is_empty", True) or not _foot.contains(_LmPt(_cx, _cy)):
                        continue
                except Exception:
                    continue
                _lm_chosen.append(_i)
            if _lm_chosen:
                _lm_set = set(_lm_chosen)
                landmark_meshes = [building_meshes[j] for j in _lm_chosen]
                building_meshes = [b for j, b in enumerate(building_meshes) if j not in _lm_set]
                print(f"[LANDMARK] {zone_prefix}{len(landmark_meshes)} building(s) -> bronze Landmark part (relief, pre-merge)")
        except Exception as _lmexc:
            print(f"[LANDMARK] {zone_prefix}relief landmark extract failed (non-fatal): {_lmexc}")
            landmark_meshes = []

    stage_start = time.perf_counter()
    merge_result = None  # defined for downstream debug_bundle/return value
    if os.environ.get("PREVIEW_MODE", "").lower() in ("1", "true", "yes"):
        # Preview mode: skip the manifold3d boolean union (~30-60s on real
        # OSM data). Buildings stay as separate meshes alongside terrain.
        #
        # ВАЖЛИВО: НЕ викликаємо extend_buildings_mesh_to_uniform_bottom
        # у preview. Раніше викликали з target_z = terrain_mesh.bounds[0][2]
        # (~-17м), і building walls простягались до самого дна підложки.
        # Без boolean union ці стіни проходили КРІЗЬ terrain — видно з кутів
        # як темні sliver-зрізи. У preview building лишається там де його
        # поставив process_buildings (flat_base_z = ground_max - 0.1mm)
        # — sit-on-terrain look без z-fighting артефактів.
        print(
            f"[INFO] {zone_prefix}PREVIEW_MODE: skipped boolean union AND extend "
            f"(buildings sit on terrain as-is to avoid sliver artifacts)"
        )
        _log_stage("merge_terrain_buildings (preview-skip)", stage_start)
    else:
        # ЗʼЄДНУВАЧ: будинки опускаємо лише до (дно + глибина_пазу + 0.6мм запас),
        # а НЕ до самого дна → нижня смуга, де ріжеться паз, лишається ЧИСТОЮ
        # підложкою без будинків (інакше у пазу «частинки будинків»). Без конектора
        # clearance=0 → стара поведінка (будинки до дна) недоторкана.
        _merge_bottom_clr = 0.0
        if getattr(request, "map_connector", False) and _sf_c > 0:
            # Зазор знизу потрібен коли ріжемо паз (щоб будинки не лізли у виїмку).
            # Без конектора clearance=0 → стара поведінка (будинки до дна).
            _cd_mm = float(getattr(request, "map_connector_depth_mm", 2.0) or 2.0)
            _merge_bottom_clr = (_cd_mm + 0.6) / _sf_c
        _merged_bmesh = union_mesh_collection(building_meshes, label="clipped_building_layer")
        merge_result = merge_terrain_and_buildings(
            terrain_mesh=terrain_mesh,
            building_meshes=building_meshes,
            merged_building_mesh=_merged_bmesh,
            support_meshes=detail_layers.support_meshes,
            bottom_clearance_m=_merge_bottom_clr,
        )
        _log_stage("merge_terrain_buildings", stage_start)
        if stage_snapshot_collector is not None:
            try:
                stage_snapshot_collector.capture_merge_stage(merge_result)
            except Exception as exc:
                print(f"[WARN] {zone_prefix}Stage snapshot failed at merge: {exc}")
        terrain_mesh = merge_result.terrain_mesh
        building_meshes = merge_result.building_meshes

    # D4 GPX-ТРЕК: ВРІЗАНИЙ маршрут (інлей) — червона вставка, верх якої flush з
    # поверхнею (НЕ виступає), + жолоб вирізаний у рельєфі boolean-ом (graceful:
    # якщо boolean впав — вставка лишається flush без жолоба). Трек спершу
    # спрощується+згладжується (друкований) і притягується до доріг міста.
    gpx_mesh = None
    if getattr(request, "gpx_track", None):
        try:
            from services.gpx_track import (
                build_gpx_track_inlay_on_terrain,
                build_gpx_track_polygon,
                TRACK_COLOR,
            )

            # ОСЬОВІ ЛІНІЇ доріг для snap-to-roads. ПАСТКА (виправлено): беремо їх з
            # terrain_stage, а НЕ з road_geometry — road_geometry присвоюється у
            # _run_terrain_stage() і поза його областю видимості тут undefined →
            # NameError мовчки ловився except'ом і road-snap НЕ працював.
            road_lines_local = None
            try:
                _rc = getattr(terrain_stage, "semantic_centerlines_local", None)
                if _rc is not None and not getattr(_rc, "is_empty", True):
                    road_lines_local = list(_rc.geoms) if hasattr(_rc, "geoms") else [_rc]
            except Exception:
                road_lines_local = None

            _tp = getattr(terrain_stage, "terrain_provider", None)
            _sf = float(zone.scale_factor or 1.0)
            _recess_mm = float(getattr(request, "gpx_raise_mm", 0.6) or 0.6)
            if _tp is not None:
                # Реалістичний рельєф: інлей (flush-вставка) + жолоб boolean-ом.
                _insert, _cutter = build_gpx_track_inlay_on_terrain(
                    gpx_track=request.gpx_track,
                    global_center=global_center,
                    zone_polygon_local=zone.zone_polygon_local,
                    terrain_provider=_tp,
                    scale_factor=_sf,
                    width_mm=float(getattr(request, "gpx_width_mm", 1.2) or 1.2),
                    recess_mm=_recess_mm,
                    road_lines_local=road_lines_local,
                )
                gpx_mesh = _insert
                # Вирізаємо жолоб у рельєфі під вставку (manifold, швидко). Graceful:
                # якщо boolean впав/зсунувся — лишаємо рельєф як є (вставка flush).
                # Жолоб ріжемо ЛИШЕ якщо обидва меші — чисті обʼєми (manifold інакше
                # падає «not a volume»). Рельєф ПІСЛЯ грувів доріг/парків часто не
                # watertight → пропускаємо boolean (без 6с марно), лишаємо flush-вставку.
                _try_groove = (
                    _insert is not None and _cutter is not None and terrain_mesh is not None
                    and bool(getattr(_cutter, "is_volume", False))
                    and bool(getattr(terrain_mesh, "is_volume", False))
                )
                if _try_groove:
                    try:
                        import trimesh as _tm
                        _b0 = terrain_mesh.bounds
                        _cut = _tm.boolean.difference([terrain_mesh, _cutter], engine="manifold")
                        if (_cut is not None and len(getattr(_cut, "faces", [])) > 0):
                            _b1 = _cut.bounds
                            # sanity: bounds не «втекли» (catastrophic boolean shift)
                            _drift = max(abs(_b1[0][i] - _b0[0][i]) for i in range(2)) + \
                                     max(abs(_b1[1][i] - _b0[1][i]) for i in range(2))
                            if _drift < (5.0 / _sf):
                                terrain_mesh = _cut
                                print(f"[GPX] {zone_prefix}track groove carved into terrain (manifold)")
                            else:
                                print(f"[GPX] {zone_prefix}track groove rejected (drift {_drift:.1f}) — flush insert kept")
                    except Exception as _bexc:
                        print(f"[GPX] {zone_prefix}track groove boolean failed (flush insert kept): {_bexc}")
            elif _sf > 0:
                # БЕЗ terrain_provider (AMS, або плоска мапа з вимкненим рельєфом,
                # або preview) → плаский flush-трек: верх на рівні землі, втоплений
                # у базу. Раніше умова вимагала саме is_ams_mode → у не-AMS плоских
                # картах трек МОВЧКИ зникав. Тепер fallback працює завжди.
                _poly = build_gpx_track_polygon(
                    gpx_track=request.gpx_track,
                    global_center=global_center,
                    zone_polygon_local=zone.zone_polygon_local,
                    scale_factor=_sf,
                    width_mm=float(getattr(request, "gpx_width_mm", 1.2) or 1.2),
                    road_lines_local=road_lines_local,
                )
                if _poly is not None and not getattr(_poly, "is_empty", True):
                    from services.flat_plate_pipeline import build_flat_layer_mesh_from_mask
                    _land = 1.0 / _sf
                    _recess = max(_recess_mm, 0.4) / _sf
                    gpx_mesh = build_flat_layer_mesh_from_mask(
                        _poly, bottom_z_m=max(_land - _recess, 0.0), thickness_m=_recess,
                        color=TRACK_COLOR, min_area_m2=1e-12,
                    )
                    if gpx_mesh is not None:
                        print(f"[GPX] {zone_prefix}AMS flush track (inset): {len(gpx_mesh.faces)} faces")
        except Exception as exc:
            print(f"[GPX] {zone_prefix}track build failed (non-fatal): {exc}")
            gpx_mesh = None

    # ── З'ЄДНУВАЧ-ПАЗИ на РЕЛЬЄФНІЙ карті (opt-in, map_connector) ──────────────
    # Раніше рельєф і конектори були взаємовиключні: конектор жив ЛИШЕ у плоскому
    # конвеєрі (паз = дві пласкі призми). Тут ріжемо той самий «ластівчин хвіст» у
    # ПЛАСКЕ ДНО рельєфної бази 3D-булеаном — той самий manifold + graceful-guard,
    # що GPX-жолоб вище. Строго за map_connector → звичайна рельєфна генерація
    # лишається БАЙТ-В-БАЙТ (golden не чіпається). Ключ-метелик = окрема деталь на
    # рівні дна (floor_z), товщина = глибина пазу. Будь-який сумнів → лишаємо базу.
    # connector_key_mesh уже міг бути виставлений PRE-GROOVE різом (вище). Цей late-
    # блок — FALLBACK лише якщо pre-groove не вдався (_relief_connector_done=False).
    _sf_c = float(getattr(zone, "scale_factor", 0.0) or 0.0)
    if (getattr(request, "map_connector", False) and not _relief_connector_done
            and terrain_mesh is not None and _sf_c > 0):
        try:
            from services.flat_plate_pipeline import (
                build_map_connector_geometry, build_flat_layer_mesh_from_mask,
                parse_connector_azimuths,
            )
            import trimesh as _tmc
            _floor_z = float(terrain_mesh.bounds[0][2])
            _model_h = float(terrain_mesh.bounds[1][2]) - _floor_z
            _depth_mm = float(getattr(request, "map_connector_depth_mm", 2.0) or 2.0)
            # Глибина пазу: не глибше 60% висоти моделі (лишаємо суцільний матеріал).
            _depth_m = min(_depth_mm / _sf_c, max(_model_h * 0.6, 0.0))
            # ── КООРДИНАТНА РАМКА (фікс «рельєф+серія: паз відкидається drift~офсет») ──
            # У СЕРІЇ рельєф-меш будується ЦЕНТРОВАНИМ у локально-тайловій рамці, а
            # zone_polygon_local — у ГЛОБАЛЬНІЙ (зміщений на офсет плитки від центру
            # сітки). Різні рамки → cutter не перетинає меш → boolean дає дрейф ≈
            # офсет плитки і паз відкидається (скарга «рельєф: жодного зʼєднання»).
            # Вирівнюємо полігон конектора до центру terrain_mesh (для single-zone
            # офсет≈0 → без змін; flat-пайплайн сюди не заходить).
            from shapely import affinity as _affc
            _tb = terrain_mesh.bounds
            _tcx = (float(_tb[0][0]) + float(_tb[1][0])) / 2.0
            _tcy = (float(_tb[0][1]) + float(_tb[1][1])) / 2.0
            _zb = zone.zone_polygon_local.bounds
            _zcx = (float(_zb[0]) + float(_zb[2])) / 2.0
            _zcy = (float(_zb[1]) + float(_zb[3])) / 2.0
            _off_x, _off_y = _tcx - _zcx, _tcy - _zcy
            _conn_poly = zone.zone_polygon_local
            if abs(_off_x) > 0.5 or abs(_off_y) > 0.5:
                _conn_poly = _affc.translate(zone.zone_polygon_local, xoff=_off_x, yoff=_off_y)
                print(f"[CONNECTOR] {zone_prefix}frame-align cutter by offset "
                      f"({_off_x:.1f},{_off_y:.1f}) terrain_c=({_tcx:.1f},{_tcy:.1f}) zone_c=({_zcx:.1f},{_zcy:.1f})")
            _ntc, _keyc = build_map_connector_geometry(
                _conn_poly,
                edges=str(getattr(request, "map_connector_edges", "NSEW") or "NSEW"),
                span_mm=float(getattr(request, "map_connector_span_mm", 10.0) or 10.0),
                length_mm=float(getattr(request, "map_connector_length_mm", 15.0) or 15.0),
                waist_frac=0.5,
                clearance_mm=float(getattr(request, "map_connector_clearance_mm", 0.03) or 0.03),
                export_scale_factor=_sf_c,
                key_edges=(str(getattr(request, "map_connector_key_edges", "") or "") or None),
                edge_dirs=parse_connector_azimuths(getattr(request, "map_connector_edge_az", "")),
                key_dirs=parse_connector_azimuths(getattr(request, "map_connector_key_az", "")),
            )
            _notch_carved = False
            if _ntc is not None and _depth_m > 1e-6:
                _eps = max(_model_h * 0.01, 0.5 / _sf_c)
                _cutterc = build_flat_layer_mesh_from_mask(
                    _ntc, bottom_z_m=_floor_z - _eps, thickness_m=_depth_m + _eps,
                    color=[128, 128, 128], min_area_m2=1e-12,
                )
                if _cutterc is not None and bool(getattr(_cutterc, "is_volume", False)):
                    # РОБАСТНИЙ ВИРІЗ ПАЗУ. Раніше вимагали, щоб terrain_mesh був
                    # герметичним (is_volume) — але рельєф ПІСЛЯ грувів доріг/парків
                    # майже ЗАВЖДИ НЕ watertight (груви приймають до 512 відкритих
                    # ребер), тож manifold-гілка тихо скипала → користувач НЕ діставав
                    # ні пазу, ні ключа («рельєф + з'єднувачі не працюють»). Тепер три
                    # рівні: (A) manifold коли герметично (швидко/точно); (B) ремонт
                    # копії (fill_holes+fix_normals)→manifold (лише trimesh, працює без
                    # Blender — важливо для прод-сервера); (C) Blender-boolean (той
                    # самий рушій, що груви; ріже й негерметичні меші). Беремо перший
                    # результат, що реально змінив геометрію в межах дрейф-ліміту.
                    _b0 = terrain_mesh.bounds
                    _faces0 = len(getattr(terrain_mesh, "faces", []))
                    _cutc = None
                    _via = None
                    # (A) manifold — ПРОБУЄМО ЗАВЖДИ (рушій manifold терпить помірну
                    # негерметичність і ЗБЕРІГАЄ координати → drift≈0). Blender-шлях у
                    # СЕРІЇ дрейфував на ~офсет плитки (~700м) і коректний паз хибно
                    # відкидався — тож manifold має пріоритет навіть на не-watertight.
                    try:
                        _cutc = _tmc.boolean.difference([terrain_mesh, _cutterc], engine="manifold")
                        _via = "manifold"
                    except Exception as _mexc:
                        print(f"[CONNECTOR] {zone_prefix}manifold notch failed ({_mexc})")
                        _cutc = None
                    # (B) РОБАСТНИЙ ремонт копії → watertight → manifold (без Blender).
                    # Рельєф після грувів доріг/парків НЕ volume (сотні відкритих ребер
                    # від каналів). fill_holes сам не закриває → manifold кидав «not all
                    # volumes» і падали у Blender, який на негерметичному мешwe РУЙНУЄ
                    # геометрію (лишав тонку Y-смужку). Тут: merge+nondegenerate+unique+
                    # КІЛЬКА проходів fill_holes + fix_winding/fix_normals → зазвичай
                    # робить volume. manifold зберігає координати (drift≈0).
                    if _cutc is None or len(getattr(_cutc, "faces", [])) == 0:
                        try:
                            import trimesh.repair as _rr
                            _rep = terrain_mesh.copy()
                            _rep.merge_vertices()
                            try:
                                _rep.update_faces(_rep.nondegenerate_faces())
                                _rep.update_faces(_rep.unique_faces())
                                _rep.remove_unreferenced_vertices()
                            except Exception:
                                pass
                            for _ in range(3):
                                if bool(getattr(_rep, "is_volume", False)):
                                    break
                                _rep.fill_holes()
                                try:
                                    _rr.fix_winding(_rep)
                                except Exception:
                                    pass
                            try:
                                _rep.fix_normals()
                            except Exception:
                                pass
                            _isv = bool(getattr(_rep, "is_volume", False))
                            print(f"[CONNECTOR] {zone_prefix}repair→volume={_isv} "
                                  f"(watertight={bool(getattr(_rep,'is_watertight',False))})")
                            if _isv:
                                _cutc = _tmc.boolean.difference([_rep, _cutterc], engine="manifold")
                                _via = "repair+manifold"
                        except Exception as _rexc:
                            print(f"[CONNECTOR] {zone_prefix}repair+manifold notch failed ({_rexc})")
                    # (C) Blender-boolean (терпить негерметичний рельєф)
                    if _cutc is None or len(getattr(_cutc, "faces", [])) == 0:
                        try:
                            from services.terrain_cutter import _run_blender_boolean
                            _bres = _run_blender_boolean(terrain_mesh, _cutterc, label="connector")
                            # повертає ВХІДНИЙ меш при невдачі → приймаємо лише якщо
                            # це інший об'єкт із гранями.
                            if (_bres is not None and _bres is not terrain_mesh
                                    and len(getattr(_bres, "faces", [])) > 0):
                                _cutc = _bres
                                _via = "blender"
                        except Exception as _bexc:
                            print(f"[CONNECTOR] {zone_prefix}Blender notch failed (non-fatal): {_bexc}")
                    # Blender boolean може лишити дрібні ВІДʼЄДНАНІ уламки далеко
                    # (артефакт різання негерметичного рельєфу) → bounds стрибають на
                    # ~офсет плитки і коректний паз ХИБНО відкидався (drift~600). Як у
                    # road-грувах: беремо НАЙБІЛЬШУ звʼязну компоненту, відкидаємо стрес.
                    if _cutc is not None and len(getattr(_cutc, "faces", [])) > 0:
                        try:
                            _parts = _cutc.split(only_watertight=False)
                            if _parts is not None and len(_parts) > 1:
                                _cutc = max(_parts, key=lambda m: len(getattr(m, "faces", [])) if m is not None else 0)
                        except Exception:
                            pass
                    if _cutc is not None and len(getattr(_cutc, "faces", [])) > 0:
                        _bb = _cutc.bounds
                        try:
                            _nparts = len(_cutc.split(only_watertight=False))
                        except Exception:
                            _nparts = -1
                        print(f"[CONNECTOR] {zone_prefix}DIAG via={_via} "
                              f"terr_xy=[{_b0[0][0]:.0f},{_b0[0][1]:.0f}..{_b0[1][0]:.0f},{_b0[1][1]:.0f}] "
                              f"res_xy=[{_bb[0][0]:.0f},{_bb[0][1]:.0f}..{_bb[1][0]:.0f},{_bb[1][1]:.0f}] parts={_nparts}")
                    # Blender-boolean на НЕгерметичному рельєфі лишає ВИРОДЖЕНУ
                    # геометрію (вершина/грань у origin (0,0)) → XY-bounds стрибають на
                    # ~позицію плитки (у серії ~700м) і коректний паз ХИБНО відкидався.
                    # Відсікаємо грані, що виходять за межі плитки (+50м запас) — стрес
                    # зникає, реальний паз (у межах плитки) лишається. manifold цього
                    # не потребує (зберігає координати), тож лише для blender-шляху.
                    if _via == "blender" and _cutc is not None and len(getattr(_cutc, "faces", [])) > 0:
                        try:
                            import numpy as _npc
                            _lo0, _lo1 = _b0[0][0] - 50.0, _b0[0][1] - 50.0
                            _hi0, _hi1 = _b0[1][0] + 50.0, _b0[1][1] + 50.0
                            _v = _npc.asarray(_cutc.vertices)
                            _inb = ((_v[:, 0] >= _lo0) & (_v[:, 0] <= _hi0)
                                    & (_v[:, 1] >= _lo1) & (_v[:, 1] <= _hi1))
                            _fm = _inb[_cutc.faces].all(axis=1)
                            if _fm.any() and not _fm.all():
                                _cutc.update_faces(_fm)
                                _cutc.remove_unreferenced_vertices()
                                print(f"[CONNECTOR] {zone_prefix}clipped {int((~_fm).sum())} stray faces outside tile bbox")
                        except Exception as _clx:
                            print(f"[CONNECTOR] {zone_prefix}stray-clip failed (non-fatal): {_clx}")
                    # Blender-виріз на НЕгерметичному вході інколи РУЙНУЄ меш (нищить
                    # дно/паз, «пази не створились, підложки немає»), але габарити
                    # лишаються ОК → стара валідація пропускала. Детект руйнування =
                    # РІЗКЕ ЗРОСТАННЯ відкритих ребер (у юзер-кейсі 3→80). manifold
                    # зберігає топологію — перевірка лише для blender-шляху.
                    if _via == "blender" and _cutc is not None and len(getattr(_cutc, "faces", [])) > 0:
                        try:
                            from trimesh.grouping import group_rows as _grc
                            _oe_in = len(_grc(terrain_mesh.edges_sorted, require_count=1))
                            _oe_out = len(_grc(_cutc.edges_sorted, require_count=1))
                            if _oe_out > _oe_in + 40:
                                print(f"[CONNECTOR] {zone_prefix}blender notch DAMAGED mesh "
                                      f"(openEdges {_oe_in}→{_oe_out}) → reject, base kept intact")
                                _cutc = None
                        except Exception:  # noqa: BLE001
                            pass
                    # СУВОРА ВАЛІДАЦІЯ «паз РЕАЛЬНО вирізаний»: у footprint різака
                    # мають зʼявитись донні грані на рівні floor+depth (стеля паза).
                    # Blender-виріз на негерметичному вході писав «carved» при
                    # незмінних габаритах, хоча паза НЕМАЄ (юзер-кейс ce68afde).
                    if _cutc is not None and len(getattr(_cutc, "faces", [])) > 0:
                        try:
                            import numpy as _npn
                            _nb = _ntc.bounds  # (minx,miny,maxx,maxy) маски пазів
                            _cN = _cutc.face_normals
                            _cc = _cutc.triangles_center
                            _dn = (_cN[:, 2] < -0.5) \
                                & (_cc[:, 0] > _nb[0] - 1) & (_cc[:, 0] < _nb[2] + 1) \
                                & (_cc[:, 1] > _nb[1] - 1) & (_cc[:, 1] < _nb[3] + 1) \
                                & (_cc[:, 2] > _floor_z + _depth_m * 0.4) \
                                & (_cc[:, 2] < _floor_z + _depth_m * 1.6)
                            if not bool(_dn.any()):
                                print(f"[CONNECTOR] {zone_prefix}notch NOT actually cut "
                                      f"(no ceiling faces at floor+depth in cutter bbox, via={_via}) → reject")
                                _cutc = None
                        except Exception as _nvx:  # noqa: BLE001
                            print(f"[CONNECTOR] {zone_prefix}notch-exists check failed (keep): {_nvx}")
                    # Валідація: меш існує, реально змінився, межі не «втекли».
                    if _cutc is not None and len(getattr(_cutc, "faces", [])) > 0:
                        _b1 = _cutc.bounds
                        _driftc = max(abs(_b1[0][i] - _b0[0][i]) for i in range(2)) + \
                                  max(abs(_b1[1][i] - _b0[1][i]) for i in range(2))
                        _changed = abs(len(_cutc.faces) - _faces0) > 0
                        # Захист від ЗРУЙНОВАНОГО boolean (Blender на не-volume лишав
                        # тонку смужку): відхиляємо, якщо XY-протяжність впала >40%.
                        _ext0 = (float(_b0[1][0] - _b0[0][0]), float(_b0[1][1] - _b0[0][1]))
                        _ext1 = (float(_b1[1][0] - _b1[0][0]), float(_b1[1][1] - _b1[0][1]))
                        _ext_ok = (_ext0[0] <= 1e-6 or _ext1[0] >= 0.6 * _ext0[0]) and \
                                  (_ext0[1] <= 1e-6 or _ext1[1] >= 0.6 * _ext0[1])
                        if _driftc < (5.0 / _sf_c) and _changed and _ext_ok:
                            terrain_mesh = _cutc
                            _notch_carved = True
                            print(f"[CONNECTOR] {zone_prefix}dovetail notch carved into relief base "
                                  f"({_via}, depth {_depth_m * _sf_c:.2f}mm, drift {_driftc:.2f})")
                        else:
                            print(f"[CONNECTOR] {zone_prefix}notch rejected (drift {_driftc:.1f}, "
                                  f"changed={_changed}, ext_ok={_ext_ok}) — base kept")
                    else:
                        print(f"[CONNECTOR] {zone_prefix}notch boolean produced nothing — base kept")
                else:
                    print(f"[CONNECTOR] {zone_prefix}connector cutter not a volume — skip notch")
                # Ключ-метелик ЛИШЕ якщо паз РЕАЛЬНО вирізано (інакше юзер отримає
                # ключ без слоту — на не-watertight рельєфі паз пропускається).
                if _notch_carved and _keyc is not None and not getattr(_keyc, "is_empty", True):
                    # Висота скрепки = 1.7мм (за вимогою), паз лишається 2мм → 0.3мм
                    # вертикальний зазор. min(): не вища за паз; floor 0.4мм проти виродження.
                    _clip_h = max(min(1.7 / _sf_c, _depth_m), 0.4 / _sf_c)
                    connector_key_mesh = build_flat_layer_mesh_from_mask(
                        _keyc, bottom_z_m=_floor_z, thickness_m=_clip_h,
                        color=[242, 242, 242], min_area_m2=1e-12,
                    )
                elif not _notch_carved:
                    print(f"[CONNECTOR] {zone_prefix}notch not carved -> key part skipped (no orphan key)")
        except Exception as _cexc:
            print(f"[CONNECTOR] {zone_prefix}relief connector failed (non-fatal): {_cexc}")
            connector_key_mesh = None

    # ── ВИДІЛЕННЯ ДОМУ (РЕЛЬЄФ): паз+будинок зроблено ВИЩЕ, ДО merge (ЧИСТА boolean-кишеня
    # на герметичному рельєфі). Тут більше нічого не ріжемо (пізній виріз давав обривчасті
    # стінки на негерметичному мерджі). highlight_meshes уже заповнено.
    highlight_part = None
    if highlight_meshes:
        try:
            import trimesh as _tmh2
            highlight_part = (highlight_meshes[0] if len(highlight_meshes) == 1
                              else _tmh2.util.concatenate(highlight_meshes))
        except Exception:
            highlight_part = highlight_meshes[0]

    landmark_part = None
    if landmark_meshes:
        try:
            import trimesh as _tmh3
            landmark_part = (landmark_meshes[0] if len(landmark_meshes) == 1
                             else _tmh3.util.concatenate([m for m in landmark_meshes if m is not None]))
        except Exception:
            landmark_part = landmark_meshes[0] if landmark_meshes else None

    # WATER seated CORRECTLY on the relief. A water polygon on a DEM spike is seated
    # against the terrain PROVIDER (which still has the spike) while the exported
    # terrain MESH was despiked/smoothed — so the water floats far above the land as a
    # vertical blue column (owner's screenshot). Here we have the ACTUAL final terrain
    # mesh: if the water surface pokes above it, RE-DRAPE the body as a thin layer that
    # hugs the real terrain (top a hair above grade so it reads as water, bottom just
    # below) — a visible flat water patch, never a pillar, never deleted. Correctly
    # seated water (surface already at/below the terrain mesh) is left untouched.
    if water_mesh is not None and terrain_mesh is not None and len(getattr(water_mesh, "vertices", [])) > 0:
        try:
            import numpy as _wnp
            _wv = water_mesh.vertices.copy()
            _wsf = float(getattr(zone, "scale_factor", 0.0) or 0.0)
            _orig = _wnp.column_stack([_wv[:, 0], _wv[:, 1],
                                       _wnp.full(len(_wv), float(terrain_mesh.bounds[1][2]) + 10.0)])
            _loc, _ridx, _ = terrain_mesh.ray.intersects_location(
                _orig, _wnp.tile([0.0, 0.0, -1.0], (len(_wv), 1)), multiple_hits=True)
            _tz = _wnp.full(len(_wv), _wnp.nan)
            for _i in range(len(_wv)):
                _h = _loc[_ridx == _i]
                if len(_h):
                    _tz[_i] = _h[:, 2].max()
            _valid = _wnp.isfinite(_tz)
            _terr_max = float(_wnp.max(_tz[_valid])) if int(_valid.sum()) > 0 else None
            _poke_thr = (0.3 / _wsf) if _wsf > 0 else 0.3
            if _terr_max is not None and float(_wv[:, 2].max()) > _terr_max + _poke_thr and int(_valid.sum()) >= 3:
                # Re-drape: thin water layer hugging the terrain, top +0.15mm, bottom
                # -0.45mm (model) about the local grade. Per-vertex terrain so it
                # follows the surface; fills the despike gap where the water groove was
                # smoothed away. Vertices without a terrain hit fall back to the median.
                # Верх води НАД грейдом (+0.15мм — ПЕРЕВІРЕНИЙ стан): спроба −0.06мм
                # «під грейд» дала КАТАСТРОФУ на великій воді (7a2412e1): там грув
                # у терені не вирізаний → терен пробивав воду → синьо-біла рябизна
                # на всю ріку. +0.15 накриває терен чисто.
                _raise = (0.15 / _wsf) if _wsf > 0 else 0.15
                _sink = (0.45 / _wsf) if _wsf > 0 else 0.45
                _med = float(_wnp.median(_tz[_valid]))
                _tzf = _wnp.where(_valid, _tz, _med)
                _zmin, _zmax = float(_wv[:, 2].min()), float(_wv[:, 2].max())
                _span = max(_zmax - _zmin, 1e-9)
                _ratio = (_wv[:, 2] - _zmin) / _span          # 0=floor, 1=surface
                _wv[:, 2] = (_tzf - _sink) + _ratio * (_sink + _raise)
                water_mesh.vertices = _wv
                print(f"[WATER] re-draped DEM-spike water onto the relief as a thin "
                      f"layer (was poking {float(_wv[:,2].max()):.2f}m, terrain {_terr_max:.2f}m)")
        except Exception as _wexc:
            print(f"[WARN] water re-drape failed: {_wexc}")

    task.update_status("processing", 85, "Експорт 3MF-файлу...")
    stage_start = time.perf_counter()
    # СЕРІЯ ЗОН: коли задано глобальний elevation_ref (серія з generate-zones) —
    # зберігаємо АБСОЛЮТНУ висоту Z (preserve_z) у кожній плитці, інакше експортер
    # обнуляє кожну плитку по власному min-Z (model_exporter ~1189/1822) → спільний
    # baseline рельєфу (який вже спечено у меш) ВТРАЧАЄТЬСЯ → сусіди стають на різні
    # висоти (сходинка/злам на шві у композит-превʼю). Одиночні мапи (ref=None) —
    # стара поведінка (обнулення на 0), golden недоторканий. XY НЕ чіпаємо (превʼю
    # розкладає плитки за метаданими; preserve_xy зламав би друк — плитка поза столом).
    _series_preserve_z = bool(getattr(request, "elevation_ref_m", None) is not None)
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
        preserve_z=_series_preserve_z,
        extra_mesh_items=(
            ([("Track", gpx_mesh)] if gpx_mesh is not None else [])
            + ([("Connector", connector_key_mesh)] if connector_key_mesh is not None else [])
            + ([("Highlight", highlight_part)] if highlight_part is not None else [])
            + ([("Landmark", landmark_part)] if landmark_part is not None else [])
        ) or None,
        reference_xy_m=zone.reference_xy_m,
        file_basename=file_basename,
        # Власник (2026-07-02): основа+рельєф+будинки = ОДИН шар (один колір у
        # палітрі й так; злиття прибирає окремий обʼєкт Buildings у 3MF/превʼю).
        # Лише relief-шлях; брелки/магніти (flat_plate) не зачеплені.
        merge_buildings_into_base=True,
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

    # Preview mode: skip the slicer-based print_acceptance gate entirely.
    # It runs a real slicer (PrusaSlicer/Bambu) on every part — easily 60-120s
    # on its own, and the preview model isn't going to be printed anyway.
    if os.environ.get("PREVIEW_MODE", "").lower() in ("1", "true", "yes"):
        print(f"[INFO] {zone_prefix}PREVIEW_MODE: skipping print_acceptance gate")
    else:
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
        # NON-BLOCKING QA: на проді немає слайсера (PrusaSlicer/Bambu), тож
        # require_slicer_validation=True давало slicer:not_found і гейт зривав
        # КОЖНУ мапу (юзер: "не створюються як до цього"). Тепер гейт лише
        # діагностичний; суворий режим вмикається через PRINT_QA_STRICT=1.
        _qa_strict = os.environ.get("PRINT_QA_STRICT", "").lower() in ("1", "true", "yes", "on")
        print_acceptance_path = write_export_print_acceptance_report(
            task_id=task_id,
            output_dir=output_dir,
            parts_for_print=parts_for_print,
            expected_parts=expected_parts,
            printer_profile=printer_profile,
            require_slicer_validation=_qa_strict,
            fail_on_slicer_warnings=_qa_strict,
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
        _qa_report_path = str(print_acceptance_path.resolve())
        if print_acceptance_report.get("status") != "pass":
            _qa_summary = summarize_export_print_failures(print_acceptance_report)
            if _qa_strict:
                if hasattr(task, "print_quality"):
                    task.print_quality = {"status": "failed", "warnings": [_qa_summary], "report": _qa_report_path}
                raise RuntimeError(_qa_summary)
            # НЕ-блокуюче: модель уже експортована (terrain+base+шари) — віддаємо
            # її попри QA-зауваження, лише логуємо. Так мапи знову створюються.
            print(f"[WARN] {zone_prefix}print_acceptance NOT passed (non-blocking): {_qa_summary}")
            if hasattr(task, "print_quality"):
                task.print_quality = {"status": "warning", "warnings": [_qa_summary], "report": _qa_report_path}
        else:
            if hasattr(task, "print_quality"):
                task.print_quality = {"status": "ok", "warnings": [], "report": _qa_report_path}
        _log_stage("print_acceptance", stage_start)

    stage_start = time.perf_counter()
    _preview_mode_on = os.environ.get("PREVIEW_MODE", "").lower() in ("1", "true", "yes")
    # debug_bundle = пост-ген дамп (~42-57с + диск), лише для розслідувань. OFF за
    # замовч. (оптимізація). Увімкнути: SAVE_DEBUG_BUNDLE=1. canonical-маски окремо.
    _save_debug_bundle = os.environ.get("SAVE_DEBUG_BUNDLE", "").lower() in ("1", "true", "yes")
    if _preview_mode_on or not _save_debug_bundle:
        print(f"[INFO] {zone_prefix}debug bundle skipped (~50s saved; SAVE_DEBUG_BUNDLE=1 to enable)")
    else:
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

    if _preview_mode_on:
        print(f"[INFO] {zone_prefix}PREVIEW_MODE: skipping Firebase upload (preview not shared)")
    else:
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
