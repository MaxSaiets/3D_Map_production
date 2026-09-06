from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from services.building_geometry_pipeline import prepare_building_geometry
from services.canonical_mask_bundle import CanonicalMaskBundle, load_canonical_mask_bundle
from services.detail_layer_pipeline import _build_canonical_road_masks
from services.detail_layer_utils import (
    MICRO_REGION_THRESHOLD_MM,
    MIN_LAND_WIDTH_MODEL_MM,
    model_mm_to_world_m,
    prepare_green_areas_for_processing,
)
from services.green_processor import process_green_areas
from services.geometry_preclip_pipeline import prepare_preclipped_geometry
from services.groove_pipeline import _prepare_parks_groove_mask
from services.print_acceptance import (
    build_mask_printability_report,
    summarize_mask_printability_failures,
    write_mask_printability_report,
)
from services.printer_profile import PrinterProfile, get_printer_profile_for_request
from services.road_geometry_pipeline import RoadGeometryPreparationResult, prepare_road_geometry
from services.runtime_canonical_masks import build_runtime_canonical_bundle
from services.water_layer_pipeline import _prepare_water_polygons


@dataclass
class Canonical2DStageResult:
    canonical_mask_bundle: CanonicalMaskBundle
    printer_profile: PrinterProfile
    printability_report: dict[str, Any]
    source_bundle_dir: Path
    bundle_origin: str
    road_geometry: Optional[RoadGeometryPreparationResult] = None
    building_geometry: Any = None
    preclip_result: Any = None


def _has_blocking_mask_failures(report: dict[str, Any] | None) -> bool:
    if not report:
        return False
    return bool(report.get("failing_layers")) or bool(report.get("failing_overlaps"))


def _is_road_only_debt(report: dict[str, Any] | None) -> bool:
    if not report:
        return False
    failing_overlaps = [str(x) for x in (report.get("failing_overlaps") or [])]
    if failing_overlaps:
        return False
    failing_layers = {str(x) for x in (report.get("failing_layers") or [])}
    if not failing_layers:
        return bool(report.get("failing_road_holes"))
    allowed = {"roads_final", "road_groove_mask"}
    return failing_layers.issubset(allowed)


def _fill_orphan_holes(
    geometry: Any,
    *,
    backing_mask: Any,
) -> Any:
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry

    polygons = [geometry] if getattr(geometry, "geom_type", "") == "Polygon" else list(getattr(geometry, "geoms", []))
    rebuilt = []
    for poly in polygons:
        if getattr(poly, "geom_type", "") != "Polygon" or poly.is_empty:
            continue
        kept_holes = []
        for ring in poly.interiors:
            try:
                hole = Polygon(ring.coords)
            except Exception:
                kept_holes.append(ring)
                continue
            if hole.is_empty:
                continue
            overlap_area = 0.0
            if backing_mask is not None and not getattr(backing_mask, "is_empty", True):
                try:
                    overlap_area = float(getattr(hole.intersection(backing_mask), "area", 0.0) or 0.0)
                except Exception:
                    overlap_area = 0.0
            if overlap_area > 1e-8:
                kept_holes.append(ring)
        try:
            rebuilt_poly = Polygon(poly.exterior.coords, holes=kept_holes).buffer(0)
        except Exception:
            rebuilt_poly = poly
        if rebuilt_poly is not None and not getattr(rebuilt_poly, "is_empty", True):
            rebuilt.append(rebuilt_poly)
    if not rebuilt:
        return None
    try:
        from shapely.ops import unary_union

        return unary_union(rebuilt).buffer(0)
    except Exception:
        return rebuilt[0]


def _fill_small_enclosed_holes(geometry: Any, *, max_hole_area_m2: float) -> Any:
    """Fill tiny fully-enclosed interior holes (road-surrounded terrain pads, the
    "white diamonds" at junctions) below max_hole_area_m2 — merge them into the
    mask. A hole is enclosed by the mask by definition, so this NEVER absorbs
    edge terrain or large courtyards (those exceed the cap or are not holes).
    Topology-only: keeps the exterior + the large holes, drops the small ones."""
    if geometry is None or getattr(geometry, "is_empty", True) or max_hole_area_m2 <= 0:
        return geometry
    polygons = [geometry] if getattr(geometry, "geom_type", "") == "Polygon" else list(getattr(geometry, "geoms", []))
    rebuilt = []
    changed = False
    for poly in polygons:
        if getattr(poly, "geom_type", "") != "Polygon" or poly.is_empty:
            if not getattr(poly, "is_empty", True):
                rebuilt.append(poly)
            continue
        kept_holes = []
        for ring in poly.interiors:
            try:
                if float(Polygon(ring.coords).area) > float(max_hole_area_m2):
                    kept_holes.append(ring)
                else:
                    changed = True
            except Exception:
                kept_holes.append(ring)
        if len(kept_holes) != len(list(poly.interiors)):
            try:
                rebuilt.append(Polygon(poly.exterior.coords, holes=kept_holes).buffer(0))
            except Exception:
                rebuilt.append(poly)
        else:
            rebuilt.append(poly)
    if not changed or not rebuilt:
        return geometry
    try:
        return unary_union(rebuilt).buffer(0)
    except Exception:
        return rebuilt[0] if len(rebuilt) == 1 else geometry


def _smooth_sharp_corners(geometry: Any, *, scale_factor: Optional[float], radius_mm: float = 0.15) -> Any:
    """Round convex and concave corners sharper than the printer can resolve.

    Why: slicers concentrate stress at sharp corners — they chip off (outside)
    or collapse into stringy first layers (inside). A tiny round-buffer /
    debuffer pair smooths both without changing topology.

    How to apply: call this just before baking masks into the canonical bundle,
    after all difference ops (difference creates the worst corners).
    """
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    if not scale_factor or scale_factor <= 0 or radius_mm <= 0:
        return geometry
    try:
        radius_m = model_mm_to_world_m(float(radius_mm), float(scale_factor))
    except Exception:
        return geometry
    if radius_m <= 0:
        return geometry
    try:
        smoothed = geometry.buffer(radius_m, join_style=1).buffer(-radius_m, join_style=1)
        smoothed = smoothed.buffer(0)
    except Exception:
        return geometry
    if smoothed is None or getattr(smoothed, "is_empty", True):
        # Smoothing destroyed the geometry (too thin) — keep original rather
        # than losing the feature entirely.
        return geometry
    return smoothed


def _collapse_acute_corners(
    geometry: Any,
    *,
    scale_factor: Optional[float],
    collapse_mm: float,
) -> Any:
    """Erode+dilate (morphological opening) to drop acute spikes and necks.

    Why: slicer can't resolve corners sharper than the nozzle line width. Pure
    `buffer(+r).buffer(-r)` (round smoothing) preserves the outline but leaves
    sub-nozzle necks; the opposite order `buffer(-r).buffer(+r)` removes those
    necks entirely — the outline is regenerated from the "thick" interior only.

    How to apply: call on every canonical mask before the bundle freeze,
    with r = acute_corner_collapse_mm / 2. Keeps inlays and grooves fitting
    because both are processed with the same radius.
    """
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    if not scale_factor or scale_factor <= 0 or collapse_mm <= 0:
        return geometry
    try:
        radius_m = model_mm_to_world_m(float(collapse_mm) * 0.5, float(scale_factor))
    except Exception:
        return geometry
    if radius_m <= 0:
        return geometry
    try:
        opened = geometry.buffer(-radius_m, join_style=2).buffer(radius_m, join_style=2)
        opened = opened.buffer(0)
    except Exception:
        return geometry
    if opened is None or getattr(opened, "is_empty", True):
        # Mask was entirely sub-acute-threshold — keep original rather than
        # deleting the layer. Downstream filters will catch it.
        return geometry
    return opened


def _prune_tiny_fragments(
    geometry: Any,
    *,
    scale_factor: Optional[float],
    min_feature_mm: float = MICRO_REGION_THRESHOLD_MM,
    min_area_mm2: float = 0.08,
) -> Any:
    """Drop tiny islands that survived boolean ops but won't print cleanly.

    Why: difference/intersection chains leave sub-millimetre slivers that the
    slicer renders as single-extrusion needles — they warp, detach, and jam
    the nozzle. Pruning here is cheaper than repairing later.
    """
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    if not scale_factor or scale_factor <= 0:
        return geometry
    try:
        min_dim_m = model_mm_to_world_m(float(min_feature_mm), float(scale_factor))
        min_area_m2 = (float(min_area_mm2) / 1e6) / (float(scale_factor) ** 2)
    except Exception:
        return geometry
    if min_dim_m <= 0 and min_area_m2 <= 0:
        return geometry

    polys: list[Polygon] = []
    if isinstance(geometry, Polygon):
        polys = [geometry]
    elif isinstance(geometry, MultiPolygon) or hasattr(geometry, "geoms"):
        polys = [g for g in getattr(geometry, "geoms", []) if isinstance(g, Polygon)]
    else:
        return geometry

    kept = []
    for poly in polys:
        if poly is None or poly.is_empty:
            continue
        try:
            area = float(getattr(poly, "area", 0.0) or 0.0)
            minx, miny, maxx, maxy = poly.bounds
            min_dim = float(min(maxx - minx, maxy - miny))
        except Exception:
            continue
        if area < min_area_m2:
            continue
        if min_dim_m > 0 and min_dim < min_dim_m:
            continue
        kept.append(poly)

    if not kept:
        return None
    if len(kept) == 1:
        return kept[0]
    try:
        return unary_union(kept).buffer(0)
    except Exception:
        return MultiPolygon(kept)


def _subtract_masks(geometry: Any, *masks: Any) -> Any:
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    result = geometry
    for mask in masks:
        if mask is None or getattr(mask, "is_empty", True):
            continue
        result_clean = result
        mask_clean = mask
        try:
            result_clean = result.buffer(0)
        except Exception:
            pass
        try:
            mask_clean = mask.buffer(0)
        except Exception:
            pass
        try:
            result = result_clean.difference(mask_clean)
            if result is None or getattr(result, "is_empty", True):
                return None
            result = result.buffer(0)
        except Exception:
            # Retry once with robust unioned operands to survive occasional
            # self-touching rings produced by earlier smoothing/pruning.
            try:
                from shapely.ops import unary_union

                lhs = unary_union([result_clean]).buffer(0)
                rhs = unary_union([mask_clean]).buffer(0)
                result = lhs.difference(rhs).buffer(0)
                if result is None or getattr(result, "is_empty", True):
                    return None
            except Exception:
                continue
    return result


def _expand_building_mask_for_roads(
    buildings: Any,
    *,
    scale_factor: float | None,
    clearance_mm: float,
) -> Any:
    if buildings is None or getattr(buildings, "is_empty", True):
        return buildings
    if not scale_factor or scale_factor <= 0 or clearance_mm <= 0:
        return buildings
    try:
        clearance_m = model_mm_to_world_m(float(clearance_mm), float(scale_factor))
    except Exception:
        return buildings
    if clearance_m <= 0:
        return buildings
    try:
        expanded = buildings.buffer(float(clearance_m), join_style=2).buffer(0)
        if expanded is None or getattr(expanded, "is_empty", True):
            return buildings
        return expanded
    except Exception:
        return buildings


def _audit_bundle_or_none(
    bundle_dir: Path,
    *,
    printer_profile: PrinterProfile,
) -> dict[str, Any] | None:
    if not bundle_dir.exists():
        return None
    try:
        report = build_mask_printability_report(  # perf-2026-09-03: build once
            bundle_dir,
            min_feature_mm=float(printer_profile.min_printable_feature_mm),
        )
        write_mask_printability_report(bundle_dir, printer_profile=printer_profile, report=report)
        return report
    except Exception:
        return None


def _bundle_matches_zone(
    *,
    bundle: CanonicalMaskBundle,
    zone_polygon_local: Any,
) -> bool:
    """Return True only if external canonical bundle belongs to current zone.

    We often keep prebuilt bundles for one control zone. Reusing them for a
    different zone silently produces valid but spatially wrong 3D geometry.
    """
    if zone_polygon_local is None or getattr(zone_polygon_local, "is_empty", True):
        return False
    bundle_zone = getattr(bundle, "zone_polygon", None)
    if bundle_zone is None or getattr(bundle_zone, "is_empty", True):
        return False
    try:
        lhs = zone_polygon_local.buffer(0)
    except Exception:
        lhs = zone_polygon_local
    try:
        rhs = bundle_zone.buffer(0)
    except Exception:
        rhs = bundle_zone
    try:
        inter = lhs.intersection(rhs)
        inter_area = float(getattr(inter, "area", 0.0) or 0.0)
        lhs_area = float(getattr(lhs, "area", 0.0) or 0.0)
        rhs_area = float(getattr(rhs, "area", 0.0) or 0.0)
        if lhs_area <= 0.0 or rhs_area <= 0.0:
            return False
        overlap_ratio = inter_area / max(lhs_area, rhs_area)
        # Same zone polygons should overlap almost completely.
        if overlap_ratio < 0.995:
            return False
        # Additional centroid sanity check in local meters.
        lc = lhs.centroid
        rc = rhs.centroid
        dx = float(lc.x) - float(rc.x)
        dy = float(lc.y) - float(rc.y)
        return (dx * dx + dy * dy) ** 0.5 <= 1.0
    except Exception:
        return False


def _attempt_runtime_overlap_self_heal(
    *,
    task_id: str,
    debug_generated_dir: Path,
    zone_polygon_local: Any,
    scale_factor: float | None,
    groove_side_clearance_mm: float,
    runtime_bundle: CanonicalMaskBundle,
    failing_overlaps: list[str],
    zone_prefix: str = "",
) -> CanonicalMaskBundle | None:
    overlap_keys = set(str(name) for name in (failing_overlaps or []))
    if not overlap_keys:
        return None
    buildings = runtime_bundle.buildings_footprints
    buildings_for_roads = _expand_building_mask_for_roads(
        buildings,
        scale_factor=scale_factor,
        clearance_mm=max(float(groove_side_clearance_mm), 0.25),
    ) if buildings is not None and not getattr(buildings, "is_empty", True) else buildings

    # Let build_runtime_canonical_bundle resolve road-vs-building precedence
    # itself. Do not pre-notch roads here; this fallback is only meant to
    # resanitize downstream park/water masks after a runtime overlap failure.
    roads_fixed = runtime_bundle.roads_final
    road_groove_fixed = runtime_bundle.road_groove_mask
    parks_fixed = _subtract_masks(
        runtime_bundle.parks_final,
        roads_fixed,
        road_groove_fixed,
        buildings_for_roads,
    )
    parks_groove_fixed = _subtract_masks(
        runtime_bundle.parks_groove_mask,
        roads_fixed,
        road_groove_fixed,
        buildings_for_roads,
    )
    water_fixed = _subtract_masks(
        runtime_bundle.water_final,
        roads_fixed,
        road_groove_fixed,
        parks_fixed,
        parks_groove_fixed,
        buildings_for_roads,
    )
    water_groove_fixed = _subtract_masks(
        runtime_bundle.water_groove_mask,
        roads_fixed,
        road_groove_fixed,
        parks_fixed,
        parks_groove_fixed,
        buildings_for_roads,
    )
    if roads_fixed is None or getattr(roads_fixed, "is_empty", True):
        roads_fixed = runtime_bundle.roads_final
    if road_groove_fixed is None or getattr(road_groove_fixed, "is_empty", True):
        road_groove_fixed = runtime_bundle.road_groove_mask
    if parks_fixed is None:
        parks_fixed = runtime_bundle.parks_final
    if parks_groove_fixed is None:
        parks_groove_fixed = runtime_bundle.parks_groove_mask
    if water_fixed is None:
        water_fixed = runtime_bundle.water_final
    if water_groove_fixed is None:
        water_groove_fixed = runtime_bundle.water_groove_mask

    print(
        f"[WARN] {zone_prefix}Canonical overlaps detected "
        f"({','.join(sorted(overlap_keys))}); running runtime self-heal"
    )
    try:
        return build_runtime_canonical_bundle(
            task_id=task_id,
            debug_generated_dir=debug_generated_dir,
            zone_polygon=zone_polygon_local,
            roads_final=roads_fixed,
            road_groove_mask=road_groove_fixed,
            parks_final=parks_fixed,
            parks_groove_mask=parks_groove_fixed,
            water_final=water_fixed,
            water_groove_mask=water_groove_fixed,
            buildings_footprints=buildings,
            scale_factor=scale_factor,
            roads_semantic_preview=getattr(runtime_bundle, "roads_semantic_preview", None),
            groove_clearance_mm=float(groove_side_clearance_mm),
        )
    except Exception as exc:
        print(f"[WARN] {zone_prefix}Runtime overlap self-heal failed: {exc}")
        return None


def _attempt_drop_water_overlap_fallback(
    *,
    task_id: str,
    debug_generated_dir: Path,
    zone_polygon_local: Any,
    scale_factor: float | None,
    groove_side_clearance_mm: float,
    runtime_bundle: CanonicalMaskBundle,
    zone_prefix: str = "",
) -> CanonicalMaskBundle | None:
    try:
        print(f"[WARN] {zone_prefix}Water masks still overlap roads; dropping water layer for this zone")
        return build_runtime_canonical_bundle(
            task_id=task_id,
            debug_generated_dir=debug_generated_dir,
            zone_polygon=zone_polygon_local,
            roads_final=runtime_bundle.roads_final,
            road_groove_mask=runtime_bundle.road_groove_mask,
            parks_final=runtime_bundle.parks_final,
            parks_groove_mask=runtime_bundle.parks_groove_mask,
            water_final=None,
            water_groove_mask=None,
            buildings_footprints=runtime_bundle.buildings_footprints,
            scale_factor=scale_factor,
            roads_semantic_preview=getattr(runtime_bundle, "roads_semantic_preview", None),
            groove_clearance_mm=float(groove_side_clearance_mm),
        )
    except Exception as exc:
        print(f"[WARN] {zone_prefix}Water-drop fallback failed: {exc}")
        return None


def _attempt_runtime_road_hole_self_heal(
    *,
    task_id: str,
    debug_generated_dir: Path,
    zone_polygon_local: Any,
    scale_factor: float | None,
    groove_side_clearance_mm: float,
    runtime_bundle: CanonicalMaskBundle,
    zone_prefix: str = "",
) -> CanonicalMaskBundle | None:
    """Patch orphan holes in road masks and rebuild runtime bundle once."""
    try:
        buildings = runtime_bundle.buildings_footprints
        roads_fixed = _fill_orphan_holes(
            runtime_bundle.roads_final,
            backing_mask=buildings,
        )
        road_groove_fixed = _fill_orphan_holes(
            runtime_bundle.road_groove_mask,
            backing_mask=buildings,
        )
        print(f"[WARN] {zone_prefix}Canonical road-hole debt detected; running runtime road-hole self-heal")
        return build_runtime_canonical_bundle(
            task_id=task_id,
            debug_generated_dir=debug_generated_dir,
            zone_polygon=zone_polygon_local,
            roads_final=roads_fixed,
            road_groove_mask=road_groove_fixed,
            parks_final=runtime_bundle.parks_final,
            parks_groove_mask=runtime_bundle.parks_groove_mask,
            water_final=runtime_bundle.water_final,
            water_groove_mask=runtime_bundle.water_groove_mask,
            buildings_footprints=buildings,
            scale_factor=scale_factor,
            roads_semantic_preview=getattr(runtime_bundle, "roads_semantic_preview", None),
            groove_clearance_mm=float(groove_side_clearance_mm),
        )
    except Exception as exc:
        print(f"[WARN] {zone_prefix}Runtime road-hole self-heal failed: {exc}")
        return None


def prepare_canonical_2d_stage(
    *,
    task_id: str,
    request: Any,
    source: Any,
    zone: Any,
    global_center: Any,
    debug_generated_dir: Path,
    zone_prefix: str = "",
) -> Canonical2DStageResult:
    printer_profile = get_printer_profile_for_request(request)

    canonical_mask_bundle_dir = getattr(request, "canonical_mask_bundle_dir", None)
    # By default we prioritize runtime canonicalization from current zone data.
    # External prebuilt bundles are useful for debugging, but can inject stale
    # topology and make all masks look broken for the current zone.
    if bool(getattr(request, "auto_canonicalize_masks", True)):
        if canonical_mask_bundle_dir:
            print(f"[INFO] {zone_prefix}Ignoring external canonical bundle; rebuilding canonical masks from source")
        canonical_mask_bundle_dir = None
    bundle_zone_mismatch = False
    if canonical_mask_bundle_dir:
        bundle_dir = Path(canonical_mask_bundle_dir).resolve()
        audit_report = _audit_bundle_or_none(bundle_dir, printer_profile=printer_profile)
        if audit_report is not None and not _has_blocking_mask_failures(audit_report):
            bundle = load_canonical_mask_bundle(bundle_dir)
            if not _bundle_matches_zone(bundle=bundle, zone_polygon_local=zone.zone_polygon_local):
                bundle_zone_mismatch = True
                print(
                    f"[WARN] {zone_prefix}Canonical bundle zone mismatch "
                    f"({bundle.source_dir}); rebuilding runtime canonical bundle"
                )
            else:
                print(f"[INFO] {zone_prefix}Using canonical 2D bundle: {bundle.source_dir}")
                bundle_buildings_for_roads = _expand_building_mask_for_roads(
                    bundle.buildings_footprints,
                    scale_factor=zone.scale_factor,
                    clearance_mm=max(float(printer_profile.groove_side_clearance_mm), 0.25),
                )
                # Road keeps full shape — buildings clipped downstream by building_exclusion_mask.
                # DO NOT subtract buildings from road here.
                bundle_roads = bundle.roads_final
                # groove intentionally overlaps buildings — do NOT subtract
                bundle_road_groove = bundle.road_groove_mask
                bundle_parks = _subtract_masks(
                    bundle.parks_final,
                    bundle_roads,
                    bundle_road_groove,
                    bundle_buildings_for_roads,
                )
                bundle_parks_groove = _subtract_masks(
                    bundle.parks_groove_mask,
                    bundle_roads,
                    bundle_road_groove,
                    bundle_buildings_for_roads,
                )
                bundle_water = _subtract_masks(
                    bundle.water_final,
                    bundle_roads,
                    bundle_road_groove,
                    bundle_parks,
                    bundle_parks_groove,
                    bundle_buildings_for_roads,
                )
                bundle_water_groove = _subtract_masks(
                    bundle.water_groove_mask,
                    bundle_roads,
                    bundle_road_groove,
                    bundle_parks,
                    bundle_parks_groove,
                    bundle_buildings_for_roads,
                )
                sanitized_bundle = build_runtime_canonical_bundle(
                    task_id=task_id,
                    debug_generated_dir=debug_generated_dir,
                    zone_polygon=zone.zone_polygon_local,
                    roads_final=bundle_roads if bundle_roads is not None else bundle.roads_final,
                    road_groove_mask=bundle_road_groove if bundle_road_groove is not None else bundle.road_groove_mask,
                    parks_final=bundle_parks if bundle_parks is not None else bundle.parks_final,
                    parks_groove_mask=bundle_parks_groove if bundle_parks_groove is not None else bundle.parks_groove_mask,
                    water_final=bundle_water if bundle_water is not None else bundle.water_final,
                    water_groove_mask=bundle_water_groove if bundle_water_groove is not None else bundle.water_groove_mask,
                    buildings_footprints=bundle.buildings_footprints,
                    scale_factor=zone.scale_factor,
                    roads_semantic_preview=getattr(bundle, "roads_semantic_preview", None),
                    groove_clearance_mm=float(printer_profile.groove_side_clearance_mm),
                )
                sanitized_report = build_mask_printability_report(  # perf-2026-09-03: build once
                    sanitized_bundle.source_dir,
                    min_feature_mm=float(printer_profile.min_printable_feature_mm),
                )
                write_mask_printability_report(
                    sanitized_bundle.source_dir, printer_profile=printer_profile, report=sanitized_report
                )
                if _has_blocking_mask_failures(sanitized_report):
                    healed_bundle = _attempt_runtime_overlap_self_heal(
                        task_id=task_id,
                        debug_generated_dir=debug_generated_dir,
                        zone_polygon_local=zone.zone_polygon_local,
                        scale_factor=zone.scale_factor,
                        groove_side_clearance_mm=float(printer_profile.groove_side_clearance_mm),
                        runtime_bundle=sanitized_bundle,
                        failing_overlaps=list(sanitized_report.get("failing_overlaps") or []),
                        zone_prefix=zone_prefix,
                    )
                    if healed_bundle is not None:
                        sanitized_bundle = healed_bundle
                        sanitized_report = build_mask_printability_report(  # perf-2026-09-03: build once
                            sanitized_bundle.source_dir,
                            min_feature_mm=float(printer_profile.min_printable_feature_mm),
                        )
                        write_mask_printability_report(
                            sanitized_bundle.source_dir, printer_profile=printer_profile, report=sanitized_report
                        )
                if _has_blocking_mask_failures(sanitized_report):
                    failing_overlaps = [str(name) for name in (sanitized_report.get("failing_overlaps") or [])]
                    if failing_overlaps and all(name.startswith("water") for name in failing_overlaps):
                        dropped_water_bundle = _attempt_drop_water_overlap_fallback(
                            task_id=task_id,
                            debug_generated_dir=debug_generated_dir,
                            zone_polygon_local=zone.zone_polygon_local,
                            scale_factor=zone.scale_factor,
                            groove_side_clearance_mm=float(printer_profile.groove_side_clearance_mm),
                            runtime_bundle=sanitized_bundle,
                            zone_prefix=zone_prefix,
                        )
                        if dropped_water_bundle is not None:
                            sanitized_bundle = dropped_water_bundle
                            sanitized_report = build_mask_printability_report(  # perf-2026-09-03: build once
                                sanitized_bundle.source_dir,
                                min_feature_mm=float(printer_profile.min_printable_feature_mm),
                            )
                            write_mask_printability_report(
                                sanitized_bundle.source_dir, printer_profile=printer_profile, report=sanitized_report
                            )
                if _has_blocking_mask_failures(sanitized_report):
                    if _is_road_only_debt(sanitized_report):
                        summary = summarize_mask_printability_failures(sanitized_report)
                        print(
                            f"[WARN] {zone_prefix}Runtime canonical bundle accepted with road-only debt "
                            f"({summary})"
                        )
                    else:
                        summary = summarize_mask_printability_failures(sanitized_report)
                        raise RuntimeError(f"Canonical 2D stage failed printability audit: {summary}")
                if sanitized_report.get("failing_road_holes"):
                    summary = summarize_mask_printability_failures(sanitized_report)
                    print(
                        f"[WARN] {zone_prefix}Runtime canonical bundle accepted with road-hole debt "
                        f"({summary})"
                    )
                return Canonical2DStageResult(
                    canonical_mask_bundle=sanitized_bundle,
                    printer_profile=printer_profile,
                    printability_report=sanitized_report,
                    source_bundle_dir=sanitized_bundle.source_dir,
                    bundle_origin="prebuilt_sanitized",
                )
        if audit_report is not None and not bundle_zone_mismatch:
            summary = summarize_mask_printability_failures(audit_report)
            print(
                f"[WARN] {zone_prefix}Canonical bundle failed 2D printability audit "
                f"({summary}); rebuilding runtime canonical bundle"
            )

    # perf-2026-09-03: [TIMING][C2D] sub-phase breakdown of the single
    # "[TIMING] canonical_2d: X.XXs" figure. Print-only, no behaviour change.
    import time as _t_c2d
    _c2d_marks = [_t_c2d.perf_counter()]

    def _c2d_mark(_name):
        _now = _t_c2d.perf_counter()
        print(f"[TIMING][C2D] {_name}: {_now - _c2d_marks[-1]:.2f}s "
              f"(total {_now - _c2d_marks[0]:.2f}s)")
        _c2d_marks.append(_now)

    building_geometry = prepare_building_geometry(
        gdf_buildings=source.gdf_buildings,
        global_center=global_center,
        zone_prefix=zone_prefix,
    )
    # Canonical 2D printability floor for roads:
    # - all road strokes must remain at least 0.5mm printable in model space
    # - voids narrower than 1.0mm between adjacent roads are merged into the road
    #   mask (merge_close_road_gaps is guarded by equiv_width < 1.1×threshold so
    #   wide city-block interiors are never swallowed — only narrow inter-lane
    #   strips / endpoint gaps are bridged).
    # 1.0mm model == 5.0m world at scale 0.2 (standard 1 km zone):
    #   • fills junction triangle gaps and parallel-lane gaps ≤ 5m
    #   • orphan_hole (0.5mm = 2.5m) fills small interior intersection holes
    # KEYCHAIN: НЕ агресивно merge'имо — користувач хоче чисту мережу без
    # «заливки» junction-трикутників. Використовуємо мінімальний gap-fill.
    road_gap_fill_mm_effective = 1.0
    _c2d_mark("building_geometry")
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
    _c2d_mark("preclip")
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

    _c2d_mark("road_geometry")
    road_insert_source = road_geometry.merged_roads_geom_local
    if road_insert_source is None or getattr(road_insert_source, "is_empty", True):
        road_insert_source = road_geometry.merged_roads_geom_local_raw
        if road_insert_source is not None and not getattr(road_insert_source, "is_empty", True):
            print(f"[WARN] {zone_prefix}Using raw local road mask fallback for canonical roads")

    canonical_road_masks = _build_canonical_road_masks(
        road_insert_source=road_insert_source,
        # Use RAW building union for road clipping. Expanded building exclusion
        # (used later for park/water and building safety margins) is too
        # aggressive for road seed masks and fragments the road network into
        # dashed micro-segments in dense zones.
        building_union_local=building_geometry.building_union_local,
        scale_factor=zone.scale_factor,
        groove_clearance_mm=float(printer_profile.groove_side_clearance_mm),
        tiny_feature_threshold_mm=0.5,
        road_gap_fill_threshold_mm=float(road_gap_fill_mm_effective),
        zone_polygon_local=zone.zone_polygon_local,
        zone_prefix=zone_prefix,
    )
    water_polygons = _prepare_water_polygons(
        preclip_result.gdf_water_local,
        road_polygons=canonical_road_masks.road_groove_mask or canonical_road_masks.road_insert_mask,
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
        min_feature_mm=float(
            max(float(printer_profile.min_printable_feature_mm), float(MIN_LAND_WIDTH_MODEL_MM))
        ),
        fit_clearance_mm=float(printer_profile.groove_side_clearance_mm) * 0.5,
        road_polygons=canonical_road_masks.road_groove_mask or canonical_road_masks.road_insert_mask,
        water_polygons=water_polygons,
        building_polygons=building_exclusion_for_roads,
        return_result=True,
    )

    # ЗЕЛЕНЬ ВИГРАЄ ВСЕРЕДИНІ ЗЕЛЕНІ (кладовища/парки): раніше parks_final =
    # parks − roads (дороги різали парк) → щільна мережа алей кладовища
    # «заливала» його дорогою (скарга юзера). Тепер зелень лишається суцільною
    # (мінус лише вода/будівлі), а дороги навпаки віднімаються зеленню нижче.
    parks_final = _subtract_masks(
        parks_result.processed_polygons if parks_result is not None else None,
        water_polygons,
        building_exclusion_for_roads,
    )

    # Прибрати тонкі ШИЙКИ/ПАЛЬЦІ парку («стовпи всередині зеленої зони»): parks_final =
    # parks − roads − buildings лишає тонкі перешийки, що екструдуються у тонкі
    # стіни-стовпи. Морфологічне ВІДКРИТТЯ (erode+dilate) прибирає шийки <~1.6мм ТУТ —
    # ДО деривації parks_groove нижче (рядок ~861), тож рівчак слідує відкритому
    # інсерту → БЕЗ orphan-grooves (інакше різало б рівчак у базі без зеленої вставки).
    # ЛИШЕ parks (дороги НЕ чіпаємо — там opening рве смуги, документована регресія).
    # Гард: якщо площа падає >5% або стає порожньо — лишаємо як було.
    try:
        if (parks_final is not None and not getattr(parks_final, "is_empty", True)
                and zone.scale_factor and float(zone.scale_factor) > 0
                and os.environ.get("PREVIEW_MODE", "").lower() not in ("1", "true", "yes")):
            _pf_a0 = float(getattr(parks_final, "area", 0.0) or 0.0)
            _pf_open = _collapse_acute_corners(
                parks_final,
                scale_factor=zone.scale_factor,
                collapse_mm=float(os.environ.get("PARK_NECK_COLLAPSE_MM", "1.6")),
            )
            _pf_a1 = float(getattr(_pf_open, "area", 0.0) or 0.0)
            if (_pf_open is not None and not getattr(_pf_open, "is_empty", True)
                    and _pf_a0 > 0 and _pf_a1 >= 0.95 * _pf_a0):
                parks_final = _pf_open
                print(f"[INFO] {zone_prefix}park neck-collapse: area {_pf_a0:.0f}->{_pf_a1:.0f} m2 "
                      f"(thin park fingers/pillars removed)")
            else:
                print(f"[INFO] {zone_prefix}park neck-collapse skipped "
                      f"(area {_pf_a0:.0f}->{_pf_a1:.0f}, >5% loss guard)")
    except Exception as _pexc:
        print(f"[WARN] {zone_prefix}park neck-collapse failed: {_pexc}")

    # Дороги НЕ залазять у зелень: віднімаємо зелену зону від road-масок, тож
    # службові алеї всередині кладовища/парку зникають, а зелень читається суцільною.
    # ВАЖЛИВО: ЗАПОВНЮЄМО building-holes у парку (parks−buildings лишає дірки де будинки)
    # перед відніманням — інакше у дірці лишається service-дорога, яка ріже будинок
    # (building_exclusion=road_groove) → будинок-у-парку НЕ рендериться → темна пляма
    # «будинок став дорогою» (скарга власника). З заповненими дірками footprint будинку
    # чистий від road_groove → будинок малюється на зеленому, дорога під ним прибрана.
    _green_src = parks_result.processed_polygons if parks_result is not None else None
    if _green_src is not None and not getattr(_green_src, "is_empty", True):
        try:
            from shapely.geometry import Polygon as _GPoly
            _gp = [_green_src] if _green_src.geom_type == "Polygon" else list(getattr(_green_src, "geoms", []))
            _green_fill = unary_union(
                [_GPoly(p.exterior) for p in _gp if p.geom_type == "Polygon"]
            ).buffer(0)
            if _green_fill is None or getattr(_green_fill, "is_empty", True):
                _green_fill = _green_src
        except Exception:
            _green_fill = _green_src
        for _attr in ("road_insert_mask", "road_groove_mask"):
            _m = getattr(canonical_road_masks, _attr, None)
            if _m is not None and not getattr(_m, "is_empty", True):
                try:
                    setattr(canonical_road_masks, _attr, _m.difference(_green_fill).buffer(0))
                except Exception:
                    pass

    parks_groove_mask = _prepare_parks_groove_mask(
        parks_final,
        road_groove_mask=canonical_road_masks.road_groove_mask,
        water_polygons=water_polygons,
        building_polygons=building_exclusion_for_roads,
        groove_clearance_m=float(printer_profile.groove_side_clearance_mm) / float(zone.scale_factor),
        boundary_snap_m=0.0,
        zone_prefix=zone_prefix,
    )
    if parks_groove_mask is not None and not getattr(parks_groove_mask, "is_empty", True):
        for exclusion_mask in (water_polygons, canonical_road_masks.road_insert_mask, canonical_road_masks.road_groove_mask):
            if exclusion_mask is None or getattr(exclusion_mask, "is_empty", True):
                continue
            try:
                parks_groove_mask = parks_groove_mask.difference(exclusion_mask).buffer(0)
            except Exception:
                pass

    # Stage 4 hygiene: after all boolean ops, smooth sharp corners and drop
    # tiny slivers before the masks are frozen into the canonical bundle.
    # Groove masks share the source polygon with their inlay so the same
    # smoothing radius is applied on both — the inlay still fits the groove.
    smoothing_radius_mm = max(
        float(printer_profile.groove_side_clearance_mm),
        0.20,
    )
    prune_min_feature_mm = float(printer_profile.min_printable_feature_mm)
    prune_min_area_mm2 = max(float(prune_min_feature_mm) ** 2 * 0.9, 0.12)

    acute_collapse_mm = float(printer_profile.acute_corner_collapse_mm)

    def _finalize_mask(mask: Any, *, label: str) -> Any:
        # Roads are topology-critical: aggressive smoothing/pruning here can
        # cut thin connectors and leave dashed/dot artifacts. Keep road masks
        # as-is at this stage; runtime canonical bundle applies its own
        # printability cleanup later.
        if label in ("road_insert", "road_groove"):
            return mask

        # Morphological opening (_collapse_acute_corners) was too aggressive at
        # 0.3mm radius — thin road strips disappeared, leaving orphan grooves
        # and sparse road networks. _enforce_min_width in the bundle builder
        # handles the min-width floor with gentler join_style=1.
        # ПРЕВʼЮ: пропускаємо дороге згладжування контуру (buffer +r/-r ×2 на маску) —
        # це чисто косметика, а у превʼю не друкуємо. Прунінг дрібних фрагментів ЛИШАЄМО
        # (він прибирає слівери, а не додає). Друк/golden (PREVIEW_MODE не виставлено) —
        # повна гігієна без змін. Економить основну частину canonical_2d-часу у превʼю.
        _preview = os.environ.get("PREVIEW_MODE", "").lower() in ("1", "true", "yes")
        if _preview:
            cleaned = mask
        else:
            cleaned = _smooth_sharp_corners(
                mask,
                scale_factor=zone.scale_factor,
                radius_mm=smoothing_radius_mm,
            )
        cleaned = _prune_tiny_fragments(
            cleaned,
            scale_factor=zone.scale_factor,
            min_feature_mm=prune_min_feature_mm,
            min_area_mm2=prune_min_area_mm2,
        )
        if cleaned is None or getattr(cleaned, "is_empty", True):
            if mask is not None and not getattr(mask, "is_empty", True):
                print(
                    f"[WARN] {zone_prefix}canonical {label} mask collapsed after "
                    f"printability hygiene; keeping pre-clean geometry"
                )
                return mask
            return None
        return cleaned

    canonical_road_masks.road_insert_mask = _finalize_mask(
        canonical_road_masks.road_insert_mask, label="road_insert"
    )
    canonical_road_masks.road_groove_mask = _finalize_mask(
        canonical_road_masks.road_groove_mask, label="road_groove"
    )
    # Fill tiny fully-enclosed terrain islands (the "white diamond" junction pads,
    # interior holes < ~3mm model) in the canonical road mask so they print as
    # road instead of a fragile sub-printable pad. Safe: only holes (road-enclosed
    # by definition) below the cap are filled — edge terrain and real courtyards
    # (bigger, or not holes) are untouched. Insert+groove both, to stay consistent.
    try:
        if zone.scale_factor and float(zone.scale_factor) > 0:
            _pad = model_mm_to_world_m(3.0, float(zone.scale_factor))
            _hole_cap = float(_pad * _pad)
            canonical_road_masks.road_insert_mask = _fill_small_enclosed_holes(
                canonical_road_masks.road_insert_mask, max_hole_area_m2=_hole_cap
            )
            canonical_road_masks.road_groove_mask = _fill_small_enclosed_holes(
                canonical_road_masks.road_groove_mask, max_hole_area_m2=_hole_cap
            )
    except Exception as exc:
        print(f"[WARN] {zone_prefix}small road-hole fill failed: {exc}")
    # Hard invariant: if road insert survived, road groove MUST also exist so
    # the 3D stages can cut terrain and the handoff validator sees matching
    # masks. If the upstream groove was lost to topology noise or an aggressive
    # filter, resynthesize by buffering the insert mask with groove clearance.
    if (
        canonical_road_masks.road_insert_mask is not None
        and not getattr(canonical_road_masks.road_insert_mask, "is_empty", True)
        and (
            canonical_road_masks.road_groove_mask is None
            or getattr(canonical_road_masks.road_groove_mask, "is_empty", True)
        )
    ):
        try:
            clearance_m = float(printer_profile.groove_side_clearance_mm) / float(zone.scale_factor)
            synthesized_groove = canonical_road_masks.road_insert_mask.buffer(
                float(clearance_m), join_style=2
            ).buffer(0)
            if zone.zone_polygon_local is not None and not getattr(zone.zone_polygon_local, "is_empty", True):
                synthesized_groove = synthesized_groove.intersection(zone.zone_polygon_local).buffer(0)
            if synthesized_groove is not None and not getattr(synthesized_groove, "is_empty", True):
                canonical_road_masks.road_groove_mask = synthesized_groove
                print(
                    f"[WARN] {zone_prefix}canonical road_groove was empty after finalize; "
                    f"synthesized from road_insert + {printer_profile.groove_side_clearance_mm:.2f}mm clearance"
                )
        except Exception as exc:
            print(f"[WARN] {zone_prefix}canonical road_groove resynthesis failed: {exc}")

    # Road keeps full shape — buildings are clipped by road_groove_mask downstream
    # in detail_layer_pipeline (building_exclusion_polygons=building_exclusion_mask).
    # DO NOT subtract buildings from road_insert here.
    # road_groove_mask intentionally overlaps buildings by groove_clearance so a
    # visible channel is cut in the terrain between the road insert and the building
    # wall. DO NOT subtract buildings here.
    parks_final = _finalize_mask(parks_final, label="parks")
    parks_groove_mask = _finalize_mask(parks_groove_mask, label="parks_groove")
    water_polygons = _finalize_mask(water_polygons, label="water")

    # ── Merge thin terrain slivers between GREEN and a road INTO the green ────
    # User's case: a <0.3mm line of terrain left standing between a park edge and
    # a road. Grow the park into that thin terrain up to whatever bounds it (road/
    # building/water) so the green meets the road and no standing sliver remains.
    # Why this is clean (unlike absorbing into the ROAD, which raggedizes the road
    # edge and just relocates the sliver — measured 27.7->29.7mm2): the park is a
    # large area, its new edge simply becomes the road edge, and nothing else moves
    # (road/building/water untouched, park's far edges untouched) -> no new sliver.
    # Only thin terrain that TOUCHES a park is absorbed, so road-road slivers are
    # left alone. Env: GREEN_SLIVER_MERGE_MM (default 0.3; 0 disables).
    try:
        _sf = float(zone.scale_factor or 0.0)
        try:
            _gm = float(os.environ.get("GREEN_SLIVER_MERGE_MM", "0.3"))
        except (TypeError, ValueError):
            _gm = 0.3
        _ri = canonical_road_masks.road_insert_mask
        _zone_poly = zone.zone_polygon_local
        if (
            _sf > 0
            and _gm > 0
            and parks_final is not None
            and not getattr(parks_final, "is_empty", True)
            and _zone_poly is not None
            and not getattr(_zone_poly, "is_empty", True)
        ):
            _r = model_mm_to_world_m(_gm / 2.0, _sf)
            # Standing terrain = zone minus every recessed/raised feature. Use the
            # road GROOVE (insert + clearance), not the insert: the strip that
            # actually stands is between the groove edges, so the park must reach
            # the groove edge or the clearance band is left as a standing sliver.
            _rg = canonical_road_masks.road_groove_mask
            _road_ref = (
                _rg if (_rg is not None and not getattr(_rg, "is_empty", True)) else _ri
            )
            _occupied = [
                _g
                for _g in (
                    _road_ref,
                    parks_final,
                    water_polygons,
                    getattr(building_geometry, "building_union_local", None),
                )
                if _g is not None and not getattr(_g, "is_empty", True)
            ]
            _terrain = _zone_poly
            if _occupied:
                _terrain = _terrain.difference(unary_union(_occupied)).buffer(0)
            # Thin terrain = terrain minus its morphological opening (width < gm).
            _core = _terrain.buffer(-_r, join_style=2).buffer(_r, join_style=2).buffer(0)
            _thin = _terrain.difference(_core).buffer(0)
            # Keep only thin terrain adjacent to a park -> absorb it into the park.
            if _thin is not None and not getattr(_thin, "is_empty", True):
                _sliver = _thin.intersection(parks_final.buffer(_r * 3.0)).buffer(0)
                if _sliver is not None and not getattr(_sliver, "is_empty", True):
                    _before = float(getattr(parks_final, "area", 0.0) or 0.0)
                    # The sliver is the standing terrain between the green and the
                    # road. Absorbing only the terrain leaves the groove-clearance
                    # gap (green reaches the road GROOVE, not the road itself). To
                    # close it fully, also bridge a sub-0.3mm margin of the road
                    # INSERT into the green and then let green win there (subtract
                    # the bridge from the road). Result: green meets the road with
                    # no standing terrain and no clearance gap. Bridge is < nozzle,
                    # so the road only recedes invisibly.
                    _bridge = None
                    if _ri is not None and not getattr(_ri, "is_empty", True):
                        _bridge = (
                            _sliver.buffer(_r * 1.2, join_style=2).intersection(_ri).buffer(0)
                        )
                    _absorb = _sliver if (_bridge is None or _bridge.is_empty) else _sliver.union(_bridge)
                    parks_final = parks_final.union(_absorb).intersection(_zone_poly).buffer(0)
                    if parks_groove_mask is not None and not getattr(
                        parks_groove_mask, "is_empty", True
                    ):
                        parks_groove_mask = (
                            parks_groove_mask.union(_absorb).intersection(_zone_poly).buffer(0)
                        )
                    # Green wins over road in the bridged margin (road recedes).
                    if _bridge is not None and not getattr(_bridge, "is_empty", True):
                        canonical_road_masks.road_insert_mask = _ri.difference(_bridge).buffer(0)
                        _rg2 = canonical_road_masks.road_groove_mask
                        if _rg2 is not None and not getattr(_rg2, "is_empty", True):
                            canonical_road_masks.road_groove_mask = _rg2.difference(_bridge).buffer(0)
                    _after = float(getattr(parks_final, "area", 0.0) or 0.0)
                    print(
                        f"[INFO] {zone_prefix}green-sliver merge (<{_gm}mm wide): "
                        f"absorbed {_after - _before:.1f} m2 into green ({_before:.0f}->{_after:.0f})"
                    )
    except Exception as exc:
        print(f"[WARN] {zone_prefix}green-sliver merge failed: {exc}")

    # ── Fill small COMPACT terrain islands enclosed by road INTO the road (①) ─────
    # User's case: a small terrain pocket left standing between two roads that is
    # neither cut nor replaced by road (a sub-printable island). Absorb ONLY pockets
    # that are SMALL, COMPACT (low aspect) and MOSTLY bounded by road, and NOT
    # adjacent to park/water. Filling a compact, road-surrounded blob just grows the
    # road by a hair — it does NOT create a new thin strip. Long high-aspect strips
    # between PARALLEL roads are deliberately LEFT: filling those welds the two roads
    # into a courtyard plate (the documented black-plate / whack-a-mole regression).
    # NOTE (2026-06-28): gated OFF by default. Verified on Vuhledar that absorbing
    # the islands into the canonical road_insert/groove mask here does NOT propagate
    # to the exported Roads inlay mesh (that mesh is built from road centerlines, not
    # this mask), so the islands survive in the print while road components fragment
    # (8->15). Same mask->mesh disconnect as the relief-buildings drop. Left in place
    # for a future fix that makes the road inlay mesh consume the canonical mask;
    # enable with ROAD_ISLAND_FILL_MM=3.
    # Env: ROAD_ISLAND_FILL_MM (default 0 = off; model-mm cap side),
    #      ROAD_ISLAND_FILL_MAX_ASPECT (default 3.0),
    #      ROAD_ISLAND_FILL_MIN_ROAD_FRAC (default 0.6).
    try:
        # GATED OFF (=0): empirically REGRESSES. Filling open compact islands into the
        # insert DOES now propagate to the print (inlay+cut rebuild from roads_final),
        # but absorbing them SHIFTS the road boundary and FRAGMENTS the surrounding
        # terrain into MORE slivers (Vuhledar: absorbed 12 but islands 17 -> 30). This
        # is the documented whack-a-mole: only FULLY-ENCLOSED holes are safe to fill
        # (handled separately by road_geometry._fill_small_road_holes), and open
        # pockets between narrowed roads cannot be removed without redesigning the
        # road-clearance model. Left env-enablable for experiments only.
        _rif_mm = float(os.environ.get("ROAD_ISLAND_FILL_MM", "0"))
        _rif_aspect = float(os.environ.get("ROAD_ISLAND_FILL_MAX_ASPECT", "3.0"))
        _rif_frac = float(os.environ.get("ROAD_ISLAND_FILL_MIN_ROAD_FRAC", "0.6"))
        _sf3 = float(zone.scale_factor or 0.0)
        _ri3 = canonical_road_masks.road_insert_mask
        _rg3 = canonical_road_masks.road_groove_mask
        _zp3 = zone.zone_polygon_local
        if (
            _rif_mm > 0 and _sf3 > 0
            and _ri3 is not None and not getattr(_ri3, "is_empty", True)
            and _zp3 is not None and not getattr(_zp3, "is_empty", True)
        ):
            _cap3 = float(model_mm_to_world_m(_rif_mm, _sf3)) ** 2          # world m^2
            _touch3 = float(model_mm_to_world_m(0.4, _sf3))                 # adjacency band
            # Detect against the INSERT (not the groove): the island then physically
            # TOUCHES the insert, so filling it into the insert CONNECTS it (no
            # clearance gap) and it survives the runtime-bundle filters instead of
            # becoming a dropped fragment. roads_final (=insert) feeds inlay + cut.
            _road_ref3 = _ri3
            _bu3 = getattr(building_geometry, "building_union_local", None)
            _occ3 = [g for g in (_road_ref3, parks_final, water_polygons, _bu3)
                     if g is not None and not getattr(g, "is_empty", True)]
            _terr3 = _zp3
            if _occ3:
                _terr3 = _terr3.difference(unary_union(_occ3)).buffer(0)
            _pieces3 = [_terr3] if getattr(_terr3, "geom_type", "") == "Polygon" else list(getattr(_terr3, "geoms", []))
            _fill3 = []
            for _p in _pieces3:
                if getattr(_p, "geom_type", "") != "Polygon" or _p.is_empty or _p.area > _cap3:
                    continue
                _minx, _miny, _maxx, _maxy = _p.bounds
                _dx, _dy = (_maxx - _minx), (_maxy - _miny)
                if max(_dx, _dy) / max(min(_dx, _dy), 1e-6) > _rif_aspect:
                    continue  # long strip -> leave (would weld parallel roads)
                _bnd = _p.boundary
                if getattr(_bnd, "length", 0.0) <= 0:
                    continue
                try:
                    _rlen = _bnd.intersection(_road_ref3.buffer(_touch3)).length
                except Exception:
                    continue
                if (_rlen / _bnd.length) < _rif_frac:
                    continue  # not mostly road-bounded -> leave
                _grow = _p.buffer(_touch3)
                if parks_final is not None and not getattr(parks_final, "is_empty", True) and _grow.intersects(parks_final):
                    continue
                if water_polygons is not None and not getattr(water_polygons, "is_empty", True) and _grow.intersects(water_polygons):
                    continue
                _fill3.append(_p)
            if _fill3:
                _fillu3 = unary_union(_fill3)
                canonical_road_masks.road_insert_mask = _ri3.union(_fillu3).intersection(_zp3).buffer(0)
                if _rg3 is not None and not getattr(_rg3, "is_empty", True):
                    canonical_road_masks.road_groove_mask = _rg3.union(_fillu3).intersection(_zp3).buffer(0)
                print(
                    f"[INFO] {zone_prefix}road-island fill: absorbed {len(_fill3)} compact "
                    f"road-bounded terrain islands ({_fillu3.area:.1f} m2) into road"
                )
    except Exception as exc:
        print(f"[WARN] {zone_prefix}road-island fill failed: {exc}")

    # runtime_canonical_masks now resolves building-vs-road precedence in one
    # place and rebuilds road_groove from the final road insert. Pass the raw
    # building union in and let that resolver decide which buildings yield,
    # which roads are cut, and what the final building footprint mask is.
    _c2d_mark("mask_build_finalize_prune")
    # PROFILE_BUNDLE=1 → cProfile навколо резолвера (діагностика R-6, без зміни логіки)
    _prof = None
    if os.getenv("PROFILE_BUNDLE") == "1":
        import cProfile as _cp
        _prof = _cp.Profile(); _prof.enable()
    runtime_bundle = build_runtime_canonical_bundle(
        task_id=task_id,
        debug_generated_dir=debug_generated_dir,
        zone_polygon=zone.zone_polygon_local,
        roads_final=canonical_road_masks.road_insert_mask,
        road_groove_mask=canonical_road_masks.road_groove_mask,
        parks_final=parks_final,
        parks_groove_mask=parks_groove_mask,
        water_final=water_polygons,
        water_groove_mask=water_polygons,
        buildings_footprints=building_geometry.building_union_local,
        scale_factor=zone.scale_factor,
        roads_semantic_preview=getattr(road_geometry, "semantic_centerlines_local", None),
        groove_clearance_mm=float(printer_profile.groove_side_clearance_mm),
    )
    if _prof is not None:
        _prof.disable()
        import pstats as _ps, io as _io
        _buf = _io.StringIO(); _ps.Stats(_prof, stream=_buf).sort_stats("cumulative").print_stats(28)
        print("[PROFILE_BUNDLE]" + chr(10) + _buf.getvalue()[:6000])
    _c2d_mark("bundle_geojson_write")
    # perf-2026-09-03 B-8: PREVIEW skips the mask printability audit (build+write
    # report) and every self-heal branch it drives. The report is print-only: no
    # downstream stage reads Canonical2DStageResult.printability_report, and the
    # heal branches only re-engineer masks for printability, never for the GLB.
    audit_report: dict[str, Any] = {}
    if os.environ.get("PREVIEW_MODE", "").lower() not in ("1", "true", "yes"):
        audit_report = build_mask_printability_report(  # perf-2026-09-03: build once
            runtime_bundle.source_dir,
            min_feature_mm=float(printer_profile.min_printable_feature_mm),
        )
        write_mask_printability_report(
            runtime_bundle.source_dir, printer_profile=printer_profile, report=audit_report
        )
        _c2d_mark("audit")
        if _has_blocking_mask_failures(audit_report):
            healed_bundle = _attempt_runtime_overlap_self_heal(
                task_id=task_id,
                debug_generated_dir=debug_generated_dir,
                zone_polygon_local=zone.zone_polygon_local,
                scale_factor=zone.scale_factor,
                groove_side_clearance_mm=float(printer_profile.groove_side_clearance_mm),
                runtime_bundle=runtime_bundle,
                failing_overlaps=list(audit_report.get("failing_overlaps") or []),
                zone_prefix=zone_prefix,
            )
            if healed_bundle is not None:
                runtime_bundle = healed_bundle
                audit_report = build_mask_printability_report(  # perf-2026-09-03: build once
                    runtime_bundle.source_dir,
                    min_feature_mm=float(printer_profile.min_printable_feature_mm),
                )
                write_mask_printability_report(
                    runtime_bundle.source_dir, printer_profile=printer_profile, report=audit_report
                )
            if _has_blocking_mask_failures(audit_report):
                failing_overlaps = [str(name) for name in (audit_report.get("failing_overlaps") or [])]
                if failing_overlaps and all(name.startswith("water") for name in failing_overlaps):
                    dropped_water_bundle = _attempt_drop_water_overlap_fallback(
                        task_id=task_id,
                        debug_generated_dir=debug_generated_dir,
                        zone_polygon_local=zone.zone_polygon_local,
                        scale_factor=zone.scale_factor,
                        groove_side_clearance_mm=float(printer_profile.groove_side_clearance_mm),
                        runtime_bundle=runtime_bundle,
                        zone_prefix=zone_prefix,
                    )
                    if dropped_water_bundle is not None:
                        runtime_bundle = dropped_water_bundle
                        audit_report = build_mask_printability_report(  # perf-2026-09-03: build once
                            runtime_bundle.source_dir,
                            min_feature_mm=float(printer_profile.min_printable_feature_mm),
                        )
                        write_mask_printability_report(
                            runtime_bundle.source_dir, printer_profile=printer_profile, report=audit_report
                        )
        if audit_report.get("failing_road_holes"):
            summary = summarize_mask_printability_failures(audit_report)
            print(
                f"[WARN] {zone_prefix}Runtime canonical bundle accepted with road-hole debt "
                f"({summary})"
            )

    _c2d_mark("self_heal")
    print(f"[INFO] {zone_prefix}Canonical 2D stage ready: {runtime_bundle.source_dir}")
    return Canonical2DStageResult(
        canonical_mask_bundle=runtime_bundle,
        printer_profile=printer_profile,
        printability_report=audit_report,
        source_bundle_dir=runtime_bundle.source_dir,
        bundle_origin="runtime",
        road_geometry=road_geometry,
        building_geometry=building_geometry,
        preclip_result=preclip_result,
    )
