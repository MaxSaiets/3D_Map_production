from __future__ import annotations

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
        write_mask_printability_report(bundle_dir, printer_profile=printer_profile)
        return build_mask_printability_report(
            bundle_dir,
            min_feature_mm=float(printer_profile.min_printable_feature_mm),
        )
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
    mask_report_kwargs = {
        "min_feature_mm": float(printer_profile.min_printable_feature_mm),
        "check_road_holes": not bool(getattr(request, "skip_road_hole_audit", False)),
        "check_layer_overlaps": not bool(getattr(request, "skip_layer_overlap_audit", False)),
    }
    write_report_kwargs = {
        key: value for key, value in mask_report_kwargs.items() if key != "min_feature_mm"
    }
    skip_runtime_printability_audit = bool(getattr(request, "skip_canonical_printability_audit", False))

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
                write_mask_printability_report(sanitized_bundle.source_dir, printer_profile=printer_profile, **write_report_kwargs)
                sanitized_report = build_mask_printability_report(
                    sanitized_bundle.source_dir,
                    **mask_report_kwargs,
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
                        write_mask_printability_report(sanitized_bundle.source_dir, printer_profile=printer_profile, **write_report_kwargs)
                        sanitized_report = build_mask_printability_report(
                            sanitized_bundle.source_dir,
                            **mask_report_kwargs,
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
                            write_mask_printability_report(sanitized_bundle.source_dir, printer_profile=printer_profile, **write_report_kwargs)
                            sanitized_report = build_mask_printability_report(
                                sanitized_bundle.source_dir,
                                **mask_report_kwargs,
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
    road_gap_fill_mm_effective = 1.0
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

    parks_final = _subtract_masks(
        parks_result.processed_polygons if parks_result is not None else None,
        canonical_road_masks.road_insert_mask,
        canonical_road_masks.road_groove_mask,
        water_polygons,
        building_exclusion_for_roads,
    )

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

    # runtime_canonical_masks now resolves building-vs-road precedence in one
    # place and rebuilds road_groove from the final road insert. Pass the raw
    # building union in and let that resolver decide which buildings yield,
    # which roads are cut, and what the final building footprint mask is.
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
    if skip_runtime_printability_audit:
        audit_report = {
            "bundle_dir": str(runtime_bundle.source_dir.resolve()),
            "status": "pass",
            "skipped": True,
            "reason": "canonical printability audit skipped for full-pipeline preview",
            "failing_layers": [],
            "failing_overlaps": [],
            "failing_road_holes": [],
        }
        try:
            (runtime_bundle.source_dir / "printability_audit.json").write_text(
                json.dumps(audit_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        print(f"[INFO] {zone_prefix}Canonical 2D printability audit skipped for preview")
    else:
        write_mask_printability_report(runtime_bundle.source_dir, printer_profile=printer_profile, **write_report_kwargs)
        audit_report = build_mask_printability_report(
            runtime_bundle.source_dir,
            **mask_report_kwargs,
        )
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
                write_mask_printability_report(runtime_bundle.source_dir, printer_profile=printer_profile, **write_report_kwargs)
                audit_report = build_mask_printability_report(
                    runtime_bundle.source_dir,
                    **mask_report_kwargs,
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
                        write_mask_printability_report(runtime_bundle.source_dir, printer_profile=printer_profile, **write_report_kwargs)
                        audit_report = build_mask_printability_report(
                            runtime_bundle.source_dir,
                            **mask_report_kwargs,
                        )
    if audit_report.get("failing_road_holes"):
        summary = summarize_mask_printability_failures(audit_report)
        print(
            f"[WARN] {zone_prefix}Runtime canonical bundle accepted with road-hole debt "
            f"({summary})"
        )

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
