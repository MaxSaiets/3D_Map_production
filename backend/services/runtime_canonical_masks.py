from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from shapely.geometry import Polygon, box, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from services.canonical_mask_bundle import CanonicalMaskBundle, load_canonical_mask_bundle
from services.detail_layer_utils import MIN_LAND_WIDTH_MODEL_MM

CANONICAL_PARTITION_MIN_FEATURE_MM = 0.6
CANONICAL_AREA_CLUSTER_MERGE_MM = 2.0
CANONICAL_PARK_ISOLATED_DETAIL_MIN_AREA_MM2 = 8.0
CANONICAL_WATER_ISOLATED_DETAIL_MIN_AREA_MM2 = 4.0
CANONICAL_TERRAIN_SLIVER_MAX_AREA_MM2 = 30.0
CANONICAL_TERRAIN_SLIVER_MAX_WIDTH_MM = 3.0
CANONICAL_ROAD_CORNER_ROUND_MM = 0.32
CANONICAL_ROAD_CORNER_CLOSE_MM = 0.80
CANONICAL_ROAD_CORNER_MAX_AREA_CHANGE_RATIO = 0.025


def _iter_polygons(geometry: BaseGeometry | None) -> list[Polygon]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return []
    if getattr(geometry, "geom_type", "") == "Polygon":
        return [geometry]
    return [geom for geom in getattr(geometry, "geoms", []) if getattr(geom, "geom_type", "") == "Polygon"]


def _polygon_equivalent_width(poly: Polygon) -> float:
    if poly is None or poly.is_empty:
        return 0.0
    try:
        area = float(getattr(poly, "area", 0.0) or 0.0)
        perimeter = float(getattr(poly, "length", 0.0) or 0.0)
        if perimeter <= 0.0:
            return 0.0
        return float((2.0 * area) / perimeter)
    except Exception:
        return 0.0


def _land_min_feature_m(scale_factor: float | None) -> float:
    if scale_factor is None or scale_factor <= 0.0:
        return 0.0
    return max(float(MIN_LAND_WIDTH_MODEL_MM), CANONICAL_PARTITION_MIN_FEATURE_MM) / float(scale_factor)


def _model_mm_to_world_m(model_mm: float, scale_factor: float | None) -> float:
    if scale_factor is None or scale_factor <= 0.0:
        return 0.0
    return float(model_mm) / float(scale_factor)


def _clip_to_zone(geometry: BaseGeometry | None, zone_geometry: BaseGeometry | None) -> BaseGeometry | None:
    if geometry is None or getattr(geometry, "is_empty", True):
        return None
    if zone_geometry is None or getattr(zone_geometry, "is_empty", True):
        return geometry
    try:
        clipped = geometry.intersection(zone_geometry)
        if clipped is None or getattr(clipped, "is_empty", True):
            return None
        geom_type = str(getattr(clipped, "geom_type", "") or "")
        if "Polygon" in geom_type:
            return clipped.buffer(0)
        return clipped
    except Exception:
        return geometry


def _drop_outlier_components(
    geometry: BaseGeometry | None,
    *,
    min_ratio_to_largest: float = 1e-3,
) -> BaseGeometry | None:
    """Drop components whose area is a fraction of the largest component.

    Why: a 0.5cm² islet next to a 30m² park is visual noise — the slicer
    renders it as a single-extrusion speck that detaches on the first layer.
    The width-based filter alone doesn't catch squarish outliers.

    How to apply: after _filter_tiny_polygon_parts, on layers that naturally
    come in distinct patches (parks, water). Don't use on roads (one network).
    """
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    polys = _iter_polygons(geometry)
    if len(polys) <= 1:
        return geometry
    areas = [float(getattr(p, "area", 0.0) or 0.0) for p in polys]
    max_area = max(areas) if areas else 0.0
    if max_area <= 0.0:
        return geometry
    kept = [p for p, a in zip(polys, areas) if (a / max_area) >= float(min_ratio_to_largest)]
    if len(kept) == len(polys):
        return geometry
    if not kept:
        return None
    try:
        return unary_union(kept).buffer(0)
    except Exception:
        return kept[0]


def _drop_isolated_area_details(
    geometry: BaseGeometry | None,
    *,
    scale_factor: float | None,
    min_area_mm2: float,
) -> BaseGeometry | None:
    """Remove printable but visually noisy isolated green/water specks."""
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    if scale_factor is None or scale_factor <= 0.0 or min_area_mm2 <= 0.0:
        return geometry
    polys = _iter_polygons(geometry)
    if len(polys) <= 1:
        return geometry

    kept: list[Polygon] = []
    for poly in polys:
        try:
            area_mm2 = float(poly.area) * float(scale_factor) * float(scale_factor)
        except Exception:
            kept.append(poly)
            continue
        if area_mm2 >= float(min_area_mm2):
            kept.append(poly)

    if len(kept) == len(polys):
        return geometry
    if not kept:
        return None
    try:
        return unary_union(kept).buffer(0)
    except Exception:
        return kept[0]


def _drop_small_road_components(
    geometry: BaseGeometry | None,
    *,
    min_feature_m: float,
    min_area_factor: float = 8.0,
    min_ratio_to_largest: float = 0.003,
) -> BaseGeometry | None:
    """Drop isolated road specks that are technically printable but meaningless.

    Roads are often one large network plus tiny detached fragments from clipping
    artifacts (small stubs, dots, self-intersection leftovers). Those fragments
    visually pollute 2D masks and become noisy detached inserts in 3D.

    Keep a road polygon only when:
    - it is not tiny in absolute area (>= min_feature^2 * factor), or
    - it is meaningful relative to the largest road component.
    """
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    polys = _iter_polygons(geometry)
    if len(polys) <= 1:
        return geometry

    min_area_m2 = max((float(min_feature_m) ** 2) * float(min_area_factor), 1e-6)
    areas = [float(getattr(poly, "area", 0.0) or 0.0) for poly in polys]
    largest = max(areas) if areas else 0.0
    if largest <= 0.0:
        return geometry

    kept: list[Polygon] = []
    for poly, area in zip(polys, areas):
        rel = float(area / largest) if largest > 0 else 0.0
        if area >= min_area_m2 or rel >= float(min_ratio_to_largest):
            kept.append(poly)

    if not kept:
        return None
    if len(kept) == len(polys):
        return geometry
    try:
        return unary_union(kept).buffer(0)
    except Exception:
        return kept[0]


def _component_count(geometry: BaseGeometry | None) -> int:
    if geometry is None or getattr(geometry, "is_empty", True):
        return 0
    return len(_iter_polygons(geometry))


def _round_road_corners_for_print(
    geometry: BaseGeometry | None,
    *,
    scale_factor: float | None,
    radius_mm: float = CANONICAL_ROAD_CORNER_ROUND_MM,
    close_radius_mm: float = CANONICAL_ROAD_CORNER_CLOSE_MM,
    max_area_change_ratio: float = CANONICAL_ROAD_CORNER_MAX_AREA_CHANGE_RATIO,
) -> BaseGeometry | None:
    """Round acute road tips without running broad road gap-fill.

    Road faithful mode keeps OSM topology intact, so the normal min-width open
    is disabled for roads. That also leaves razor mitre tips at clipped road
    turns and zone/building boundaries. A tiny road-only open rounds those tips
    while guarded area/component checks prevent deleting the actual network.
    """
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    if scale_factor is None or scale_factor <= 0.0 or radius_mm <= 0.0:
        return geometry
    before_area = float(getattr(geometry, "area", 0.0) or 0.0)
    before_components = _component_count(geometry)
    if before_area <= 0.0:
        return geometry

    candidate_pairs_mm = [
        (float(radius_mm), float(close_radius_mm)),
        (min(float(radius_mm), 0.30), min(float(close_radius_mm), 0.65)),
        (min(float(radius_mm), 0.25), min(float(close_radius_mm), 0.50)),
        (min(float(radius_mm), 0.20), min(float(close_radius_mm), 0.35)),
    ]
    seen: set[tuple[float, float]] = set()
    for open_mm, close_mm in candidate_pairs_mm:
        open_mm = round(float(open_mm), 4)
        close_mm = round(float(close_mm), 4)
        key = (open_mm, close_mm)
        if key in seen or open_mm <= 0.0 or close_mm <= 0.0:
            continue
        seen.add(key)
        try:
            open_radius_m = _model_mm_to_world_m(open_mm, float(scale_factor))
            close_radius_m = _model_mm_to_world_m(close_mm, float(scale_factor))
        except Exception:
            continue
        if open_radius_m <= 0.0 or close_radius_m <= 0.0:
            continue

        try:
            opened = geometry.buffer(-open_radius_m, join_style=1).buffer(open_radius_m, join_style=1)
            rounded = opened.buffer(close_radius_m, join_style=1).buffer(-close_radius_m, join_style=1).buffer(0)
        except Exception:
            continue
        if rounded is None or getattr(rounded, "is_empty", True):
            continue

        after_area = float(getattr(rounded, "area", 0.0) or 0.0)
        if after_area <= 0.0:
            continue
        area_change_ratio = abs(after_area - before_area) / before_area
        if area_change_ratio > float(max_area_change_ratio):
            continue

        # Opening may split or delete narrow connectors/components. Rounding is
        # allowed to shave tips, but not to change road topology.
        after_components = _component_count(rounded)
        if before_components > 0 and after_components != before_components:
            continue

        return rounded

    return geometry


def _apply_mask_difference(
    geometry: BaseGeometry | None,
    exclusion: BaseGeometry | None,
) -> BaseGeometry | None:
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    if exclusion is None or getattr(exclusion, "is_empty", True):
        return geometry
    lhs = geometry
    rhs = exclusion
    try:
        lhs = geometry.buffer(0)
    except Exception:
        pass
    try:
        rhs = exclusion.buffer(0)
    except Exception:
        pass
    try:
        clipped = lhs.difference(rhs).buffer(0)
    except Exception:
        try:
            clipped = unary_union([lhs]).buffer(0).difference(unary_union([rhs]).buffer(0)).buffer(0)
        except Exception:
            return geometry
    if clipped is None or getattr(clipped, "is_empty", True):
        return None
    return clipped


def _bounds_overlap(
    a: BaseGeometry | None,
    b: BaseGeometry | None,
    *,
    padding_m: float = 0.0,
) -> bool:
    if a is None or b is None or getattr(a, "is_empty", True) or getattr(b, "is_empty", True):
        return False
    try:
        ax0, ay0, ax1, ay1 = a.bounds
        bx0, by0, bx1, by1 = b.bounds
        pad = max(0.0, float(padding_m))
        return not (
            ax1 + pad < bx0
            or bx1 + pad < ax0
            or ay1 + pad < by0
            or by1 + pad < ay0
        )
    except Exception:
        return True


def _clip_conflict_to_building_window(
    conflict: BaseGeometry | None,
    building: BaseGeometry | None,
    *,
    padding_m: float,
) -> BaseGeometry | None:
    if conflict is None or building is None or getattr(conflict, "is_empty", True) or getattr(building, "is_empty", True):
        return None
    if not _bounds_overlap(conflict, building, padding_m=padding_m):
        return None
    try:
        minx, miny, maxx, maxy = building.bounds
        pad = max(0.0, float(padding_m))
        window = box(minx - pad, miny - pad, maxx + pad, maxy + pad)
        clipped = conflict.intersection(window)
        if clipped is None or getattr(clipped, "is_empty", True):
            return None
        return clipped
    except Exception:
        return conflict


def _polygon_min_dimension(poly: Polygon | None) -> float:
    if poly is None or poly.is_empty:
        return 0.0
    try:
        minx, miny, maxx, maxy = poly.bounds
        return float(min(maxx - minx, maxy - miny))
    except Exception:
        return 0.0


def _overlap_spans_building(
    building: Polygon | None,
    overlap: BaseGeometry | None,
    *,
    span_ratio: float = 0.82,
    min_area_ratio: float = 0.08,
) -> bool:
    if building is None or building.is_empty or overlap is None or getattr(overlap, "is_empty", True):
        return False
    try:
        bx0, by0, bx1, by1 = building.bounds
        ox0, oy0, ox1, oy1 = overlap.bounds
        b_width = float(bx1 - bx0)
        b_height = float(by1 - by0)
        if b_width <= 0.0 or b_height <= 0.0:
            return False
        o_width = float(ox1 - ox0)
        o_height = float(oy1 - oy0)
        width_ratio = o_width / b_width
        height_ratio = o_height / b_height
        overlap_area = float(getattr(overlap, "area", 0.0) or 0.0)
        building_area = float(getattr(building, "area", 0.0) or 0.0)
        area_ratio = (overlap_area / building_area) if building_area > 0.0 else 0.0
        return area_ratio >= float(min_area_ratio) and (
            width_ratio >= float(span_ratio) or height_ratio >= float(span_ratio)
        )
    except Exception:
        return False


def _build_road_groove_from_insert(
    roads_geom: BaseGeometry | None,
    *,
    groove_clearance_m: float,
    zone_geom: BaseGeometry | None,
) -> BaseGeometry | None:
    if roads_geom is None or getattr(roads_geom, "is_empty", True):
        return None
    try:
        groove = (
            roads_geom.buffer(float(groove_clearance_m), join_style=1)
            if groove_clearance_m > 0.0
            else roads_geom.buffer(0)
        )
    except Exception:
        groove = roads_geom
    groove = _clip_to_zone(groove, zone_geom)
    if groove is None or getattr(groove, "is_empty", True):
        return None
    try:
        return groove.buffer(0)
    except Exception:
        return groove


def _semantic_support_length(
    roads_semantic_geom: BaseGeometry | None,
    region: BaseGeometry | None,
) -> float:
    if (
        roads_semantic_geom is None
        or getattr(roads_semantic_geom, "is_empty", True)
        or region is None
        or getattr(region, "is_empty", True)
    ):
        return 0.0
    try:
        return float(getattr(roads_semantic_geom.intersection(region), "length", 0.0) or 0.0)
    except Exception:
        return 0.0


def _semantic_centerline_crosses_building(
    building: Polygon | None,
    roads_semantic_geom: BaseGeometry | None,
    *,
    groove_clearance_m: float,
    span_ratio: float = 0.72,
    min_length_ratio: float = 0.55,
) -> bool:
    if (
        building is None
        or building.is_empty
        or roads_semantic_geom is None
        or getattr(roads_semantic_geom, "is_empty", True)
    ):
        return False
    try:
        support_window = building.buffer(max(float(groove_clearance_m) * 0.5, 0.25), join_style=2).buffer(0)
    except Exception:
        support_window = building
    try:
        centerline_local = roads_semantic_geom.intersection(support_window)
    except Exception:
        return False
    if centerline_local is None or getattr(centerline_local, "is_empty", True):
        return False
    try:
        bx0, by0, bx1, by1 = building.bounds
        cx0, cy0, cx1, cy1 = centerline_local.bounds
        b_width = float(bx1 - bx0)
        b_height = float(by1 - by0)
        if b_width <= 0.0 or b_height <= 0.0:
            return False
        width_ratio = float(cx1 - cx0) / b_width
        height_ratio = float(cy1 - cy0) / b_height
        local_length = float(getattr(centerline_local, "length", 0.0) or 0.0)
        min_dim = float(min(b_width, b_height))
        return local_length >= max(min_dim * float(min_length_ratio), 1.0) and (
            width_ratio >= float(span_ratio) or height_ratio >= float(span_ratio)
        )
    except Exception:
        return False


def _road_overlap_reaches_building_core(
    building: Polygon | None,
    overlap: BaseGeometry | None,
    *,
    groove_clearance_m: float,
    large_building_min_dim_m: float = 40.0,
    core_margin_ratio: float = 0.22,
    min_core_margin_m: float = 6.0,
    max_core_margin_ratio: float = 0.35,
) -> bool:
    if (
        building is None
        or building.is_empty
        or overlap is None
        or getattr(overlap, "is_empty", True)
    ):
        return False
    try:
        min_dim = _polygon_min_dimension(building)
        if min_dim <= 0.0:
            return False
        # Small buildings do not need a special "core" rule; for them the
        # semantic crossing + survival tests are already strict enough.
        if min_dim < float(large_building_min_dim_m):
            return True
        core_margin_m = max(
            float(min_core_margin_m),
            float(groove_clearance_m) * 6.0,
            float(min_dim) * float(core_margin_ratio),
        )
        core_margin_m = min(core_margin_m, float(min_dim) * float(max_core_margin_ratio))
        core = building.buffer(-core_margin_m, join_style=2).buffer(0)
        if core is None or getattr(core, "is_empty", True):
            return True
        return not getattr(overlap.intersection(core), "is_empty", True)
    except Exception:
        return True


def _build_large_building_edge_clip_exclusion(
    *,
    building: Polygon | None,
    conflict: BaseGeometry | None,
    roads_geom: BaseGeometry | None,
    groove_clearance_m: float,
    large_building_min_dim_m: float = 40.0,
    edge_depth_factor: float = 1.2,
    min_edge_depth_m: float = 2.0,
    max_edge_depth_ratio: float = 0.12,
) -> BaseGeometry | None:
    if (
        building is None
        or building.is_empty
        or conflict is None
        or getattr(conflict, "is_empty", True)
    ):
        return None
    try:
        min_dim = _polygon_min_dimension(building)
        if min_dim < float(large_building_min_dim_m):
            return None

        conflict_overlap = building.intersection(conflict).buffer(0)
        if conflict_overlap is None or getattr(conflict_overlap, "is_empty", True):
            return None

        # If the conflict already reaches the building core, this is not an
        # edge-only case; fall back to the normal full clip / road-cut logic.
        if _road_overlap_reaches_building_core(
            building,
            conflict_overlap,
            groove_clearance_m=float(groove_clearance_m),
            large_building_min_dim_m=float(large_building_min_dim_m),
        ):
            return None

        if roads_geom is not None and not getattr(roads_geom, "is_empty", True):
            try:
                road_overlap = building.intersection(roads_geom).buffer(0)
            except Exception:
                road_overlap = None
            if (
                road_overlap is not None
                and not getattr(road_overlap, "is_empty", True)
                and _road_overlap_reaches_building_core(
                    building,
                    road_overlap,
                    groove_clearance_m=float(groove_clearance_m),
                    large_building_min_dim_m=float(large_building_min_dim_m),
                )
            ):
                return None

        edge_depth_m = max(float(groove_clearance_m) * float(edge_depth_factor), float(min_edge_depth_m))
        edge_depth_m = min(edge_depth_m, float(min_dim) * float(max_edge_depth_ratio))
        try:
            core = building.buffer(-edge_depth_m, join_style=1).buffer(0)
        except Exception:
            core = None
        if core is None or getattr(core, "is_empty", True):
            edge_band = building.buffer(0)
        else:
            try:
                edge_band = building.difference(core).buffer(0)
            except Exception:
                edge_band = building.buffer(0)
        if edge_band is None or getattr(edge_band, "is_empty", True):
            return None
        try:
            limited = conflict_overlap.intersection(edge_band).buffer(0)
        except Exception:
            limited = conflict_overlap
        if limited is None or getattr(limited, "is_empty", True):
            return None
        return limited
    except Exception:
        return None


def _buffer_building_for_road_cut(
    building: Polygon | None,
    *,
    groove_clearance_m: float,
) -> BaseGeometry | None:
    if building is None or building.is_empty:
        return None
    cut_radius_m = max(float(groove_clearance_m), 0.0)
    epsilon_m = max(cut_radius_m * 0.01, 0.01) if cut_radius_m > 0.0 else 0.0
    try:
        buffered = building.buffer(cut_radius_m + epsilon_m, join_style=2).buffer(0)
        return buffered if buffered is not None and not getattr(buffered, "is_empty", True) else building.buffer(0)
    except Exception:
        try:
            return building.buffer(0)
        except Exception:
            return building


def _build_local_road_cut_geometry(
    *,
    building: Polygon | None,
    road_overlap: BaseGeometry | None,
    groove_clearance_m: float,
) -> BaseGeometry | None:
    if (
        building is None
        or building.is_empty
        or road_overlap is None
        or getattr(road_overlap, "is_empty", True)
    ):
        return None
    try:
        limit = building.buffer(
            max(float(groove_clearance_m), 0.0) + 0.01,
            join_style=2,
        ).buffer(0)
    except Exception:
        limit = building
    try:
        cut = road_overlap.buffer(
            max(float(groove_clearance_m), 0.0) + 0.01,
            join_style=2,
        ).buffer(0)
    except Exception:
        cut = road_overlap
    try:
        cut = cut.intersection(limit).buffer(0)
    except Exception:
        pass
    if cut is None or getattr(cut, "is_empty", True):
        return _buffer_building_for_road_cut(building, groove_clearance_m=float(groove_clearance_m))
    return cut


def _road_cut_creates_subprintable_corridor(
    *,
    roads_geom: BaseGeometry | None,
    cut_geom: BaseGeometry | None,
    building: Polygon | None,
    min_feature_m: float,
    groove_clearance_m: float,
    loss_ratio_threshold: float = 0.12,
) -> bool:
    if (
        roads_geom is None
        or getattr(roads_geom, "is_empty", True)
        or cut_geom is None
        or getattr(cut_geom, "is_empty", True)
        or building is None
        or building.is_empty
        or min_feature_m <= 0.0
    ):
        return False
    try:
        outer = building.buffer(
            max(float(min_feature_m) * 1.5, float(groove_clearance_m) + float(min_feature_m)),
            join_style=2,
        ).buffer(0)
        inner = building.buffer(max(float(groove_clearance_m) * 0.25, 0.0), join_style=2).buffer(0)
        ring_window = outer.difference(inner).buffer(0)
    except Exception:
        return False
    if ring_window is None or getattr(ring_window, "is_empty", True):
        return False

    try:
        local_after = roads_geom.difference(cut_geom).intersection(ring_window).buffer(0)
    except Exception:
        return False
    if local_after is None or getattr(local_after, "is_empty", True):
        return False

    after_area = float(getattr(local_after, "area", 0.0) or 0.0)
    if after_area <= 1e-8:
        return False

    printable_after = _enforce_min_width(local_after, min_feature_m=float(min_feature_m))
    printable_after = _filter_tiny_polygon_parts(printable_after, min_feature_m=float(min_feature_m))
    printable_area = float(getattr(printable_after, "area", 0.0) or 0.0) if printable_after is not None else 0.0

    removed_ratio = max(0.0, after_area - printable_area) / after_area if after_area > 0.0 else 0.0
    return removed_ratio >= float(loss_ratio_threshold)


def _resolve_building_road_conflicts(
    *,
    roads_geom: BaseGeometry | None,
    buildings_geom: BaseGeometry | None,
    roads_semantic_geom: BaseGeometry | None,
    scale_factor: float | None,
    groove_clearance_m: float,
    zone_geom: BaseGeometry | None,
) -> tuple[BaseGeometry | None, BaseGeometry | None, BaseGeometry | None]:
    """Resolve building-vs-road conflicts before final canonical bundle freeze.

    Rules implemented here:
    - Roads stay authoritative by default; groove is rebuilt from the final road insert.
    - Buildings usually yield to roads + groove clearance.
    - Exception: if a valid building is essentially traversed by a true road
      centerline corridor, preserve the building and cut only the local road
      segment that passes through it.
    """
    if buildings_geom is None or getattr(buildings_geom, "is_empty", True):
        road_groove_geom = _build_road_groove_from_insert(
            roads_geom,
            groove_clearance_m=float(groove_clearance_m),
            zone_geom=zone_geom,
        )
        return roads_geom, road_groove_geom, buildings_geom

    if roads_geom is None or getattr(roads_geom, "is_empty", True):
        return roads_geom, None, buildings_geom

    building_parts = [part.buffer(0) for part in _iter_polygons(buildings_geom) if part is not None and not part.is_empty]
    if not building_parts:
        road_groove_geom = _build_road_groove_from_insert(
            roads_geom,
            groove_clearance_m=float(groove_clearance_m),
            zone_geom=zone_geom,
        )
        return roads_geom, road_groove_geom, None

    raw_road_groove = _build_road_groove_from_insert(
        roads_geom,
        groove_clearance_m=float(groove_clearance_m),
        zone_geom=zone_geom,
    )
    raw_conflict_masks = [geom for geom in (roads_geom, raw_road_groove) if geom is not None and not getattr(geom, "is_empty", True)]
    try:
        raw_conflict = unary_union(raw_conflict_masks).buffer(0) if raw_conflict_masks else None
    except Exception:
        raw_conflict = raw_conflict_masks[0] if raw_conflict_masks else None

    min_feature_m = 0.0
    if scale_factor is not None and scale_factor > 0.0:
        min_feature_m = _land_min_feature_m(scale_factor)
    min_preserve_area_m2 = max((float(min_feature_m) ** 2) * 0.5, 3.0)
    min_preserve_dim_m = max(float(min_feature_m) * 0.75, 1.4)
    survival_ratio_threshold = 0.35

    road_cut_parts: list[BaseGeometry] = []
    road_cut_indices: set[int] = set()

    for idx, building in enumerate(building_parts):
        if building is None or building.is_empty:
            continue
        local_roads = _clip_conflict_to_building_window(
            roads_geom,
            building,
            padding_m=max(float(groove_clearance_m) * 2.0, 1.0),
        )
        if local_roads is None or getattr(local_roads, "is_empty", True):
            continue
        try:
            road_overlap = building.intersection(local_roads)
        except Exception:
            road_overlap = None
        if road_overlap is None or getattr(road_overlap, "is_empty", True):
            continue

        before_area = float(getattr(building, "area", 0.0) or 0.0)
        min_dim = _polygon_min_dimension(building)
        if before_area <= 0.0:
            continue

        road_overlap_area = float(getattr(road_overlap, "area", 0.0) or 0.0)
        road_overlap_ratio = (road_overlap_area / before_area) if before_area > 0.0 else 0.0

        if before_area < float(min_preserve_area_m2) or min_dim < float(min_preserve_dim_m):
            continue

        local_raw_conflict = _clip_conflict_to_building_window(
            raw_conflict,
            building,
            padding_m=max(float(groove_clearance_m) * 2.0, 1.0),
        )
        clipped_building = _apply_mask_difference(building, local_raw_conflict)
        after_area = float(getattr(clipped_building, "area", 0.0) or 0.0) if clipped_building is not None else 0.0
        survival_ratio = (after_area / before_area) if before_area > 0.0 else 0.0

        semantic_cross = _semantic_centerline_crosses_building(
            building,
            roads_semantic_geom,
            groove_clearance_m=float(groove_clearance_m),
            span_ratio=0.72,
            min_length_ratio=0.55,
        )
        reaches_core = _road_overlap_reaches_building_core(
            building,
            road_overlap,
            groove_clearance_m=float(groove_clearance_m),
            large_building_min_dim_m=40.0,
            core_margin_ratio=0.22,
            min_core_margin_m=6.0,
            max_core_margin_ratio=0.35,
        )
        semantic_support_m = _semantic_support_length(roads_semantic_geom, building)
        min_semantic_crossing_m = max(min_dim * 0.9, 25.0)

        should_cut_road = (
            road_overlap_area > 1e-8
            and (
                road_overlap_ratio >= 0.95
                or (
                    reaches_core
                    and
                    semantic_cross
                    and semantic_support_m >= float(min_semantic_crossing_m)
                    and road_overlap_ratio >= 0.25
                    and (
                        clipped_building is None
                        or getattr(clipped_building, "is_empty", True)
                        or survival_ratio < 0.45
                    )
                )
            )
        )
        if not should_cut_road:
            continue

        cut_geom = _build_local_road_cut_geometry(
            building=building,
            road_overlap=road_overlap,
            groove_clearance_m=float(groove_clearance_m),
        )
        if cut_geom is None or getattr(cut_geom, "is_empty", True):
            continue
        if _road_cut_creates_subprintable_corridor(
            roads_geom=roads_geom,
            cut_geom=cut_geom,
            building=building,
            min_feature_m=float(min_feature_m),
            groove_clearance_m=float(groove_clearance_m),
            loss_ratio_threshold=0.12,
        ):
            # If cutting the road under this building would leave only a
            # sub-printable road ring / side strip around the building, keep
            # the road authoritative and let the building yield instead.
            continue
        road_cut_parts.append(cut_geom)
        road_cut_indices.add(idx)

    if road_cut_parts:
        try:
            road_cut_union = unary_union(road_cut_parts).buffer(0)
        except Exception:
            road_cut_union = road_cut_parts[0]
        roads_geom = _apply_mask_difference(roads_geom, road_cut_union)
        roads_geom = _clip_to_zone(roads_geom, zone_geom)

    road_groove_geom = _build_road_groove_from_insert(
        roads_geom,
        groove_clearance_m=float(groove_clearance_m),
        zone_geom=zone_geom,
    )
    final_conflict_masks = [geom for geom in (roads_geom, road_groove_geom) if geom is not None and not getattr(geom, "is_empty", True)]
    try:
        final_conflict = unary_union(final_conflict_masks).buffer(0) if final_conflict_masks else None
    except Exception:
        final_conflict = final_conflict_masks[0] if final_conflict_masks else None

    kept_buildings: list[BaseGeometry] = []
    for idx, building in enumerate(building_parts):
        if building is None or building.is_empty:
            continue
        building_conflict = _clip_conflict_to_building_window(
            final_conflict,
            building,
            padding_m=max(float(groove_clearance_m) * 2.0, 1.0),
        )
        if building_conflict is None or getattr(building_conflict, "is_empty", True):
            kept_buildings.append(building)
            continue
        edge_clip_exclusion = _build_large_building_edge_clip_exclusion(
            building=building,
            conflict=building_conflict,
            roads_geom=roads_geom,
            groove_clearance_m=float(groove_clearance_m),
            large_building_min_dim_m=40.0,
            edge_depth_factor=1.2,
            min_edge_depth_m=2.0,
            max_edge_depth_ratio=0.12,
        )
        if edge_clip_exclusion is not None and not getattr(edge_clip_exclusion, "is_empty", True):
            building_conflict = edge_clip_exclusion
        clipped_building = _apply_mask_difference(building, building_conflict)
        if clipped_building is not None and not getattr(clipped_building, "is_empty", True):
            kept_buildings.append(clipped_building)
            continue
        if idx in road_cut_indices:
            kept_buildings.append(building)

    if kept_buildings:
        try:
            buildings_geom = unary_union(kept_buildings).buffer(0)
        except Exception:
            buildings_geom = kept_buildings[0]
    else:
        buildings_geom = None

    return roads_geom, road_groove_geom, buildings_geom


def _filter_tiny_polygon_parts(
    geometry: BaseGeometry | None,
    *,
    min_feature_m: float,
) -> BaseGeometry | None:
    """Drop polygons that fail the printability test used by the audit.

    The audit (``_component_stats`` → ``_survives_printable_erosion`` in
    ``print_acceptance.py``) flags a polygon as unprintable if either:
      * its area is below ``min_feature_m**2 * 0.5``, or
      * it disappears when eroded by ``min_feature_m * 0.5``.

    We deliberately mirror those two tests here instead of using
    ``equivalent_width = 2 * area / perimeter``. Equivalent-width is a poor
    proxy for a BRANCHED network (e.g. a whole road mask is often a single
    connected polygon; its perimeter is dominated by side branches and
    eq_width collapses to well below the threshold even though the trunk is
    wide). The old eq_width filter wiped the entire road network; the
    fallback then restored the raw pre-filter geometry with all the needle
    branches still attached, and the downstream audit failed with
    ``layers=roads_final`` because the needles did not survive erosion.

    With the erosion-based test a polygon that passes this filter is
    guaranteed to pass the audit. Morphological opening via
    ``_enforce_min_width`` (applied before this filter for road/groove
    layers) removes needle branches PERMANENTLY so the remaining lobes
    comfortably pass the erosion check.
    """
    if geometry is None or getattr(geometry, "is_empty", True) or min_feature_m <= 0:
        return geometry
    kept: list[Polygon] = []
    min_area_m2 = max(float(min_feature_m) ** 2 * 0.5, 1e-8)
    erosion = float(min_feature_m) * 0.5
    for poly in _iter_polygons(geometry):
        if poly is None or poly.is_empty:
            continue
        area = float(getattr(poly, "area", 0.0) or 0.0)
        if area <= 1e-8 or area < min_area_m2:
            continue
        try:
            shrunken = poly.buffer(-erosion, join_style=1)
        except Exception:
            # Defensive: if erosion errors, keep the polygon — the audit
            # uses the same operation and will flag it if truly degenerate.
            kept.append(poly)
            continue
        if shrunken is None or getattr(shrunken, "is_empty", True):
            continue
        kept.append(poly)
    if not kept:
        return None
    try:
        return unary_union(kept).buffer(0)
    except Exception:
        return kept[0]


def _fill_tiny_holes(
    geometry: BaseGeometry | None,
    *,
    min_feature_m: float,
) -> BaseGeometry | None:
    """Fill any interior ring that the audit would flag as an unprintable hole.

    The audit (``_component_stats``) flags a hole if its area is below
    ``min_feature_m**2 * 0.5`` OR it disappears under erosion by
    ``min_feature_m * 0.5``. We mirror those tests here so every hole that
    would fail the audit gets filled before the bundle is written. The old
    logic also required ``area <= 8*min_feature_m^2``, which meant a long
    thin hole (e.g. a 2 m × 100 m canal at scale 1:4500) slipped through —
    too big to be "tiny" by area, too narrow to print. Now the erosion
    test catches it regardless of absolute area.
    """
    if geometry is None or getattr(geometry, "is_empty", True) or min_feature_m <= 0:
        return geometry

    min_area_m2 = max(float(min_feature_m) ** 2 * 0.5, 1e-8)
    erosion = float(min_feature_m) * 0.5

    def _fill_poly(poly: Polygon) -> Polygon | None:
        if poly is None or poly.is_empty:
            return None
        kept_holes = []
        for interior in poly.interiors:
            try:
                hole = Polygon(interior)
            except Exception:
                kept_holes.append(interior)
                continue
            if hole.is_empty:
                continue
            try:
                area = float(getattr(hole, "area", 0.0) or 0.0)
            except Exception:
                kept_holes.append(interior)
                continue

            # Area-based test: holes smaller than half a min-feature square
            # are always unprintable — fill them.
            if area < min_area_m2:
                continue
            # Keep non-tiny holes intact to preserve map topology.
            # The erosion-based branch was too aggressive for ring roads:
            # long/narrow interior holes collapsed and entire districts were
            # filled into one black road blob.
            kept_holes.append(interior)

        try:
            return Polygon(poly.exterior.coords, holes=kept_holes).buffer(0)
        except Exception:
            return poly

    kept = [_fill_poly(poly) for poly in _iter_polygons(geometry)]
    kept = [poly for poly in kept if poly is not None and not poly.is_empty]
    if not kept:
        return None
    try:
        return unary_union(kept).buffer(0)
    except Exception:
        return kept[0]


def _survives_printable_erosion(
    geometry: BaseGeometry | None,
    *,
    min_feature_m: float,
) -> bool:
    if geometry is None or getattr(geometry, "is_empty", True):
        return False
    if min_feature_m <= 0.0:
        return True
    erosion = float(min_feature_m) * 0.5
    try:
        shrunken = geometry.buffer(-erosion, join_style=1)
        return shrunken is not None and not getattr(shrunken, "is_empty", True)
    except Exception:
        return True


def _fill_unprintable_holes(
    geometry: BaseGeometry | None,
    *,
    min_feature_m: float,
    max_area_factor: float | None = 8.0,
) -> BaseGeometry | None:
    """Fill only conservative, audit-failing holes.

    Why this exists:
    - `road_map_faithful_mode` intentionally preserves real road topology, so
      we avoid broad orphan-hole healing on roads.
    - Some final road/groove holes are still boolean leftovers that the audit
      will reject as unprintable (`small_hole_count`), even though filling
      them does not alter semantic street blocks.

    Safety rule:
    - fill the hole only when it fails the same printable-erosion test used by
      the audit, and its area stays below a conservative cap relative to the
      printable threshold.
    """
    if geometry is None or getattr(geometry, "is_empty", True) or min_feature_m <= 0.0:
        return geometry

    min_area_m2 = max(float(min_feature_m) ** 2 * 0.5, 1e-8)
    max_fill_area_m2 = (
        None
        if max_area_factor is None
        else max(float(min_feature_m) ** 2 * float(max_area_factor), min_area_m2)
    )

    def _fill_poly(poly: Polygon) -> Polygon | None:
        if poly is None or poly.is_empty:
            return None
        kept_holes = []
        for interior in poly.interiors:
            try:
                hole = Polygon(interior)
            except Exception:
                kept_holes.append(interior)
                continue
            if hole.is_empty:
                continue

            try:
                hole_area = float(getattr(hole, "area", 0.0) or 0.0)
            except Exception:
                kept_holes.append(interior)
                continue

            if hole_area <= 0.0:
                continue

            if hole_area < min_area_m2:
                continue

            if max_fill_area_m2 is None:
                try:
                    minx, miny, maxx, maxy = hole.bounds
                    min_dim = min(float(maxx - minx), float(maxy - miny))
                    equiv_width = _polygon_equivalent_width(hole)
                except Exception:
                    min_dim = float("inf")
                    equiv_width = float("inf")
                if min_dim < float(min_feature_m) or equiv_width < float(min_feature_m):
                    continue

            if (max_fill_area_m2 is None or hole_area <= max_fill_area_m2) and not _survives_printable_erosion(
                hole,
                min_feature_m=float(min_feature_m),
            ):
                continue

            kept_holes.append(interior)

        try:
            return Polygon(poly.exterior.coords, holes=kept_holes).buffer(0)
        except Exception:
            return poly

    kept = [_fill_poly(poly) for poly in _iter_polygons(geometry)]
    kept = [poly for poly in kept if poly is not None and not poly.is_empty]
    if not kept:
        return None
    try:
        return unary_union(kept).buffer(0)
    except Exception:
        return kept[0]


def _fill_orphan_holes(
    geometry: BaseGeometry | None,
    *,
    backing_mask: BaseGeometry | None,
) -> BaseGeometry | None:
    # Map-faithful mode: never auto-fill interior road holes.
    # Earlier "orphan hole" healing turned ring-like and parallel roads into
    # solid blobs in dense zones. Keep the source topology as-is.
    return geometry


def _enforce_min_width(
    geometry: BaseGeometry | None,
    *,
    min_feature_m: float,
) -> BaseGeometry | None:
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    if min_feature_m <= 0.0:
        return geometry
    erosion = float(min_feature_m) * 0.5
    if erosion <= 0.0:
        return geometry
    try:
        opened = geometry.buffer(-erosion, join_style=1)
    except Exception:
        return geometry
    if opened is None or getattr(opened, "is_empty", True):
        return None
    try:
        restored = opened.buffer(erosion, join_style=1).buffer(0)
    except Exception:
        return opened
    if restored is None or getattr(restored, "is_empty", True):
        return None
    return restored


def _write_geojson(path: Path, geometry: BaseGeometry | None) -> Path:
    payload = {"type": "FeatureCollection", "features": []}
    if geometry is not None and not getattr(geometry, "is_empty", True):
        payload["features"].append(
            {
                "type": "Feature",
                "properties": {},
                "geometry": mapping(geometry),
            }
        )
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _prepare_water_groove_mask(
    water_insert_mask: BaseGeometry | None,
    *,
    road_groove_mask: BaseGeometry | None,
    parks_groove_mask: BaseGeometry | None,
    building_polygons: BaseGeometry | None,
    groove_clearance_m: float,
) -> BaseGeometry | None:
    if water_insert_mask is None or getattr(water_insert_mask, "is_empty", True):
        return None
    if building_polygons is not None and not getattr(building_polygons, "is_empty", True):
        try:
            water_insert_mask = water_insert_mask.difference(building_polygons).buffer(0)
        except Exception:
            pass
    try:
        water_groove_mask = (
            water_insert_mask.buffer(float(groove_clearance_m), join_style=2)
            if groove_clearance_m > 0.0
            else water_insert_mask
        )
        water_groove_mask = water_groove_mask.buffer(0)
    except Exception:
        water_groove_mask = water_insert_mask

    for exclusion_mask in (building_polygons,):
        if exclusion_mask is None or getattr(exclusion_mask, "is_empty", True):
            continue
        try:
            water_groove_mask = water_groove_mask.difference(exclusion_mask).buffer(0)
        except Exception:
            pass
    return water_groove_mask


def _build_inlay_groove_from_insert(
    insert_mask: BaseGeometry | None,
    *,
    groove_clearance_m: float,
    zone_geom: BaseGeometry | None,
    hard_exclusions: list[BaseGeometry | None] | None = None,
) -> BaseGeometry | None:
    if insert_mask is None or getattr(insert_mask, "is_empty", True):
        return None
    try:
        groove = (
            insert_mask.buffer(float(groove_clearance_m), join_style=2)
            if groove_clearance_m > 0.0
            else insert_mask.buffer(0)
        )
    except Exception:
        groove = insert_mask
    groove = _clip_to_zone(groove, zone_geom)
    if groove is None or getattr(groove, "is_empty", True):
        return None
    for exclusion in hard_exclusions or []:
        groove = _apply_mask_difference(groove, exclusion)
    if groove is None or getattr(groove, "is_empty", True):
        return None
    try:
        return groove.buffer(0)
    except Exception:
        return groove


def _building_fit_exclusion_for_insert(
    *,
    buildings_geom: BaseGeometry | None,
    insert_geom: BaseGeometry | None,
    groove_clearance_m: float,
    min_overlap_ratio: float = 0.85,
) -> BaseGeometry | None:
    """Expand only buildings that are actually embedded in an insert mask."""
    if buildings_geom is None or getattr(buildings_geom, "is_empty", True):
        return buildings_geom
    if (
        insert_geom is None
        or getattr(insert_geom, "is_empty", True)
        or groove_clearance_m <= 0.0
    ):
        return buildings_geom

    exclusions: list[BaseGeometry] = []
    for building in _iter_polygons(buildings_geom):
        try:
            building_area = float(building.area or 0.0)
        except Exception:
            building_area = 0.0
        if building_area <= 0.0:
            continue
        try:
            overlap_area = float(building.intersection(insert_geom).area or 0.0)
        except Exception:
            overlap_area = 0.0
        near_insert_area = 0.0
        if groove_clearance_m > 0.0:
            try:
                probe = building.buffer(float(groove_clearance_m) * 1.5, join_style=2)
                probe = probe.difference(building).buffer(0)
                near_insert_area = float(probe.intersection(insert_geom).area or 0.0)
            except Exception:
                near_insert_area = 0.0
        if (
            overlap_area >= building_area * float(min_overlap_ratio)
            or near_insert_area > max(1e-8, building_area * 1e-4)
        ):
            try:
                exclusions.append(building.buffer(float(groove_clearance_m), join_style=2))
            except Exception:
                exclusions.append(building)
        else:
            exclusions.append(building)

    if not exclusions:
        return buildings_geom
    try:
        return unary_union(exclusions).buffer(0)
    except Exception:
        return buildings_geom


def _close_unprintable_bays(
    geometry: BaseGeometry | None,
    *,
    min_feature_m: float,
) -> BaseGeometry | None:
    """Fill concave bays/necks narrower than the printable floor.

    This is safe only for non-road inserts. Roads are semantic networks where a
    close can merge neighbouring streets into blobs; parks/water are area masks
    where sub-0.6mm bites become ragged, non-printable edges.
    """
    if geometry is None or getattr(geometry, "is_empty", True) or min_feature_m <= 0.0:
        return geometry
    radius_m = float(min_feature_m) * 0.5
    try:
        closed = geometry.buffer(radius_m, join_style=1).buffer(-radius_m, join_style=1).buffer(0)
    except Exception:
        return geometry
    if closed is None or getattr(closed, "is_empty", True):
        return geometry
    return closed


def _sanitize_nonroad_insert(
    geometry: BaseGeometry | None,
    *,
    min_feature_m: float,
    close_bays: bool = True,
) -> BaseGeometry | None:
    if geometry is None or getattr(geometry, "is_empty", True) or min_feature_m <= 0.0:
        return geometry
    cleaned = _fill_unprintable_holes(
        geometry,
        min_feature_m=float(min_feature_m),
        max_area_factor=None,
    )
    if close_bays:
        cleaned = _close_unprintable_bays(cleaned, min_feature_m=float(min_feature_m))
    cleaned = _enforce_min_width(cleaned, min_feature_m=float(min_feature_m))
    cleaned = _filter_tiny_polygon_parts(cleaned, min_feature_m=float(min_feature_m))
    cleaned = _fill_unprintable_holes(
        cleaned,
        min_feature_m=float(min_feature_m),
        max_area_factor=None,
    )
    return cleaned


def _fill_unbacked_insert_holes(
    geometry: BaseGeometry | None,
    *,
    backing_masks: list[BaseGeometry | None],
    min_overlap_ratio: float = 0.01,
) -> BaseGeometry | None:
    """Fill insert holes that are not occupied by another final layer."""
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    blockers = [mask for mask in backing_masks if mask is not None and not getattr(mask, "is_empty", True)]
    try:
        backing = unary_union(blockers).buffer(0) if blockers else None
    except Exception:
        backing = None

    filled_parts: list[Polygon] = []
    for poly in _iter_polygons(geometry):
        kept_holes = []
        for interior in poly.interiors:
            try:
                hole = Polygon(interior)
                hole_area = float(getattr(hole, "area", 0.0) or 0.0)
            except Exception:
                kept_holes.append(interior.coords)
                continue
            if hole_area <= 0.0:
                continue
            backed_area = 0.0
            if backing is not None and not getattr(backing, "is_empty", True):
                try:
                    backed_area = float(hole.intersection(backing).area or 0.0)
                except Exception:
                    backed_area = 0.0
            if backed_area >= max(1e-8, hole_area * float(min_overlap_ratio)):
                kept_holes.append(interior.coords)
        try:
            filled = Polygon(poly.exterior.coords, holes=kept_holes).buffer(0)
        except Exception:
            filled = poly
        if filled is not None and not getattr(filled, "is_empty", True):
            filled_parts.extend(_iter_polygons(filled))
    if not filled_parts:
        return None
    try:
        return unary_union(filled_parts).buffer(0)
    except Exception:
        return geometry


def _is_unprintable_remainder(poly: Polygon | None, *, min_feature_m: float) -> bool:
    if poly is None or poly.is_empty or min_feature_m <= 0.0:
        return False
    try:
        area = float(getattr(poly, "area", 0.0) or 0.0)
        minx, miny, maxx, maxy = poly.bounds
        min_dim = min(float(maxx - minx), float(maxy - miny))
    except Exception:
        return False
    min_area_m2 = max(float(min_feature_m) ** 2 * 0.5, 1e-8)
    return (
        area < min_area_m2
        or min_dim < float(min_feature_m)
        or not _survives_printable_erosion(poly, min_feature_m=float(min_feature_m))
    )


def _is_visual_sliver_remainder(
    poly: Polygon | None,
    *,
    min_feature_m: float,
    scale_factor: float | None,
) -> bool:
    if poly is None or poly.is_empty or min_feature_m <= 0.0:
        return False
    if scale_factor is None or scale_factor <= 0.0:
        return False
    try:
        area_mm2 = float(poly.area or 0.0) * (float(scale_factor) ** 2)
        equivalent_width_mm = _polygon_equivalent_width(poly) * float(scale_factor)
        minx, miny, maxx, maxy = poly.bounds
        min_dim_mm = min(float(maxx - minx), float(maxy - miny)) * float(scale_factor)
    except Exception:
        return False
    width_mm = min(equivalent_width_mm, min_dim_mm)
    return (
        area_mm2 <= CANONICAL_TERRAIN_SLIVER_MAX_AREA_MM2
        and width_mm <= CANONICAL_TERRAIN_SLIVER_MAX_WIDTH_MM
    )


def _absorb_unprintable_remainders_into_inlays(
    *,
    zone_geom: BaseGeometry | None,
    roads_geom: BaseGeometry | None,
    road_groove_geom: BaseGeometry | None,
    parks_geom: BaseGeometry | None,
    water_geom: BaseGeometry | None,
    buildings_geom: BaseGeometry | None,
    min_feature_m: float,
    scale_factor: float | None = None,
) -> tuple[BaseGeometry | None, BaseGeometry | None]:
    """Assign tiny leftover terrain scraps to nearby park/water inlays.

    Boolean differences often leave beige terrain crumbs between green/water
    inserts and their neighbours. If such a crumb cannot survive the printable
    erosion test and it sits next to a park or water mask, merge it into that
    inlay before grooves/terrain masks are frozen. Isolated scraps with no
    nearby inlay are left as terrain rather than inventing a semantic feature.
    """
    if zone_geom is None or getattr(zone_geom, "is_empty", True) or min_feature_m <= 0.0:
        return parks_geom, water_geom

    blockers = [
        geom
        for geom in (roads_geom, road_groove_geom, parks_geom, water_geom, buildings_geom)
        if geom is not None and not getattr(geom, "is_empty", True)
    ]
    if not blockers:
        return parks_geom, water_geom

    try:
        remainder = zone_geom.difference(unary_union(blockers)).buffer(0)
    except Exception:
        return parks_geom, water_geom
    if remainder is None or getattr(remainder, "is_empty", True):
        return parks_geom, water_geom

    boundary = None
    try:
        boundary = zone_geom.boundary
    except Exception:
        boundary = None

    proximity_m = max(float(min_feature_m) * 1.25, 1e-4)
    park_absorb: list[Polygon] = []
    water_absorb: list[Polygon] = []

    def _candidate_score(part: Polygon, candidate: BaseGeometry | None, priority: int) -> tuple[float, int] | None:
        if candidate is None or getattr(candidate, "is_empty", True):
            return None
        try:
            distance = float(part.distance(candidate))
        except Exception:
            return None
        if distance > proximity_m:
            return None
        return (distance, priority)

    for part in _iter_polygons(remainder):
        if part is None or part.is_empty:
            continue
        is_unprintable = _is_unprintable_remainder(part, min_feature_m=float(min_feature_m))
        is_visual_sliver = _is_visual_sliver_remainder(
            part,
            min_feature_m=float(min_feature_m),
            scale_factor=scale_factor,
        )
        if not is_unprintable and not is_visual_sliver:
            continue
        if boundary is not None:
            try:
                if part.intersects(boundary):
                    continue
            except Exception:
                pass

        candidates: list[tuple[tuple[float, int], str]] = []
        park_score = _candidate_score(part, parks_geom, 0)
        water_score = _candidate_score(part, water_geom, 1)
        if park_score is not None:
            candidates.append((park_score, "parks"))
        if water_score is not None:
            candidates.append((water_score, "water"))
        if not candidates:
            continue
        _, target = min(candidates, key=lambda item: item[0])
        if target == "parks":
            park_absorb.append(part)
        else:
            water_absorb.append(part)

    if park_absorb:
        try:
            parks_geom = unary_union([geom for geom in (parks_geom, *park_absorb) if geom is not None]).buffer(0)
        except Exception:
            pass
    if water_absorb:
        try:
            water_geom = unary_union([geom for geom in (water_geom, *water_absorb) if geom is not None]).buffer(0)
        except Exception:
            pass

    return parks_geom, water_geom


def build_runtime_canonical_bundle(
    *,
    task_id: str,
    debug_generated_dir: Path,
    zone_polygon: BaseGeometry | None,
    roads_final: BaseGeometry | None,
    road_groove_mask: BaseGeometry | None,
    parks_final: BaseGeometry | None,
    parks_groove_mask: BaseGeometry | None,
    water_final: BaseGeometry | None,
    water_groove_mask: BaseGeometry | None,
    buildings_footprints: BaseGeometry | None,
    scale_factor: Optional[float],
    roads_semantic_preview: BaseGeometry | None = None,
    groove_clearance_mm: float = 0.15,
) -> CanonicalMaskBundle:
    bundle_dir = (debug_generated_dir / "runtime_canonical_bundles" / task_id).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    zone_geom = zone_polygon
    roads_geom = _clip_to_zone(roads_final, zone_geom)
    road_groove_geom = _clip_to_zone(road_groove_mask, zone_geom)
    parks_geom = _clip_to_zone(parks_final, zone_geom)
    parks_groove_geom = _clip_to_zone(parks_groove_mask, zone_geom)
    water_geom = _clip_to_zone(water_final, zone_geom)
    buildings_geom = _clip_to_zone(buildings_footprints, zone_geom)
    roads_semantic_geom = _clip_to_zone(roads_semantic_preview, zone_geom)
    # Map-faithful mode is ON: roads keep full OSM topology. The canonical
    # slicer passes use morphological OPEN (buffer(-r).buffer(+r)), which
    # DELETES features narrower than `min_road_feature_m` instead of widening
    # them to meet the printable threshold. Turning faithful mode OFF wiped
    # the entire road network (reported 2026-04-19). Min-width enforcement is
    # instead applied downstream in `process_roads` via `normalize_road_mask_for_print`,
    # which handles narrow roads without deleting them wholesale.
    road_map_faithful_mode = True
    groove_clearance_m = 0.0
    if scale_factor is not None and scale_factor > 0:
        groove_clearance_m = float(groove_clearance_mm) / float(scale_factor)

    roads_geom, road_groove_geom, buildings_geom = _resolve_building_road_conflicts(
        roads_geom=roads_geom,
        buildings_geom=buildings_geom,
        roads_semantic_geom=roads_semantic_geom,
        scale_factor=scale_factor,
        groove_clearance_m=float(groove_clearance_m),
        zone_geom=zone_geom,
    )

    water_groove_geom = _prepare_water_groove_mask(
        water_geom,
        road_groove_mask=road_groove_geom,
        parks_groove_mask=parks_groove_geom,
        building_polygons=buildings_geom,
        groove_clearance_m=float(groove_clearance_m),
    )
    water_groove_geom = _clip_to_zone(water_groove_geom, zone_geom)

    # Bundle-level safety: buildings take precedence. Any road/groove/park/water
    # that overlaps a building footprint must lose, so 3D extrude can't produce
    # walls clipping through a building mesh. Do this BEFORE the filter so the
    # filter only sees the clipped (final) masks.
    #
    # Important: do NOT subtract the building-groove ring from road inserts.
    # Doing so creates "moat" holes around buildings inside roads, which are
    # reported by the canonical audit as road_holes orphan_holes and appear as
    # empty cutouts in the final model.
    building_groove_geom = None
    parks_building_fit_source = parks_geom
    water_building_fit_source = water_geom
    parks_building_exclusion = buildings_geom
    water_building_exclusion = buildings_geom
    if buildings_geom is not None and not getattr(buildings_geom, "is_empty", True):
        if groove_clearance_m > 0.0:
            try:
                ring = buildings_geom.buffer(float(groove_clearance_m), join_style=2)
                ring = ring.difference(buildings_geom).buffer(0)
                if zone_geom is not None and not getattr(zone_geom, "is_empty", True):
                    ring = ring.intersection(zone_geom).buffer(0)
                if ring is not None and not getattr(ring, "is_empty", True):
                    building_groove_geom = ring
            except Exception:
                building_groove_geom = None

        hard_exclusion = buildings_geom
        parks_building_exclusion = _building_fit_exclusion_for_insert(
            buildings_geom=buildings_geom,
            insert_geom=parks_building_fit_source,
            groove_clearance_m=float(groove_clearance_m),
        )
        water_building_exclusion = _building_fit_exclusion_for_insert(
            buildings_geom=buildings_geom,
            insert_geom=water_building_fit_source,
            groove_clearance_m=float(groove_clearance_m),
        )

        # Roads remain authoritative. Never notch road insert or road groove
        # with buildings here; only downstream non-road layers yield.
        # Buildings fully embedded in parks/water need a real fit clearance in
        # the insert mask; edge-touching buildings still use the footprint only
        # to avoid broad tan halos along roads and parcel boundaries.
        parks_geom = _apply_mask_difference(parks_geom, parks_building_exclusion)
        parks_groove_geom = _apply_mask_difference(parks_groove_geom, parks_building_exclusion)
        water_geom = _apply_mask_difference(water_geom, water_building_exclusion)
        water_groove_geom = _apply_mask_difference(water_groove_geom, water_building_exclusion)

    if scale_factor is not None and scale_factor > 0.0:
        # World-space threshold equivalent to MIN_LAND_WIDTH_MODEL_MM on the
        # printed model. scale_factor is defined as print_mm / world_m (see
        # services/generator.py: scale_factor = model_size_mm / zone_size_m),
        # so `world_m = print_mm / scale_factor`. The previous formula
        # (`print_mm * scale_factor / 1000`) was wrong by factor ~20,000 —
        # the filter threshold came out to 0.12 world-mm instead of 2.47 m,
        # so nothing below the nozzle diameter was ever filtered.
        min_land_feature_m = _land_min_feature_m(scale_factor)
        # Road printability thresholds are DISABLED in faithful mode: the
        # morphological OPEN inside `_enforce_min_width` uses
        # `buffer(-r).buffer(+r)` which DELETES narrow roads instead of
        # widening them. At typical zone scale_factor = 0.2 (1km → 200mm),
        # 0.55mm model == 2.75m world erosion radius, which wipes all
        # footways and service roads and merges nearby parallel roads into
        # patches (user report 2026-04-19: "roads replaced by patches").
        # Keep these at 0.0 so `_enforce_min_width` and `_fill_tiny_holes`
        # early-return and the road network stays intact.
        min_road_feature_m = 0.0
        road_hole_fill_m = 0.0
        area_cluster_merge_m = max(
            min_land_feature_m,
            _model_mm_to_world_m(CANONICAL_AREA_CLUSTER_MERGE_MM, scale_factor),
        )

        # Preserve originals so we can fall back if the filter chain over-reaches
        # and wipes out a still-valid road network. The downstream handoff
        # validator compares canonical.road_groove to terrain/detail masks — if
        # bundle.road_groove is accidentally None, the 3D stages resynthesize
        # their own version and drift is reported as an error.
        roads_pre_filter = roads_geom
        road_groove_pre_filter = road_groove_geom

        # Order matters for branchy networks (roads/road_groove). Applying the
        # per-polygon speck filter BEFORE morphological opening meant the
        # filter saw the whole road network as one connected polygon and
        # sometimes wiped it (branchy networks fail the legacy equivalent-width
        # test). Opening first permanently removes needle branches < 2*erosion
        # wide, naturally splitting the network into clean lobes; the filter
        # then only drops true specks.
        roads_geom = _fill_tiny_holes(roads_geom, min_feature_m=road_hole_fill_m)
        roads_geom = _enforce_min_width(roads_geom, min_feature_m=min_road_feature_m)
        roads_geom = _filter_tiny_polygon_parts(roads_geom, min_feature_m=min_road_feature_m)
        if (
            (roads_geom is None or getattr(roads_geom, "is_empty", True))
            and roads_pre_filter is not None
            and not getattr(roads_pre_filter, "is_empty", True)
        ):
            # Filter chain collapsed a real road network — fall back to the
            # opened pre-filter geometry (still passes audit, but may include
            # wider feeder branches we'd otherwise have pruned).
            try:
                fallback = _enforce_min_width(roads_pre_filter, min_feature_m=min_road_feature_m)
            except Exception:
                fallback = roads_pre_filter
            roads_geom = fallback if fallback is not None and not getattr(fallback, "is_empty", True) else roads_pre_filter

        road_groove_geom = _fill_tiny_holes(road_groove_geom, min_feature_m=road_hole_fill_m)
        road_groove_geom = _enforce_min_width(road_groove_geom, min_feature_m=min_road_feature_m)
        road_groove_geom = _filter_tiny_polygon_parts(road_groove_geom, min_feature_m=min_road_feature_m)
        if (
            (road_groove_geom is None or getattr(road_groove_geom, "is_empty", True))
            and road_groove_pre_filter is not None
            and not getattr(road_groove_pre_filter, "is_empty", True)
        ):
            try:
                fallback = _enforce_min_width(road_groove_pre_filter, min_feature_m=min_road_feature_m)
            except Exception:
                fallback = road_groove_pre_filter
            road_groove_geom = fallback if fallback is not None and not getattr(fallback, "is_empty", True) else road_groove_pre_filter

        # NOTE: a morphological CLOSE on the canonical road mask here was too
        # aggressive in dense urban areas. At scale_factor=0.2 a 0.6 mm model
        # gap = 3 m world, and adjacent streets in city centres are routinely
        # 2–4 m apart (with sidewalks / medians between). The guarded CLOSE
        # joined them into large black blobs (user report 2026-04-19, follow-
        # up). Road gap-fill, if needed, should happen ONCE on the raw
        # merged_roads in process_roads (where it is driven by the user's
        # `road_gap_fill_threshold_mm` setting) — not here on the finished
        # polygon mask.
        parks_geom = _fill_tiny_holes(parks_geom, min_feature_m=min_land_feature_m)
        # Parks/water are area inserts, not semantic networks. Use a larger
        # visual clustering threshold than the hard 0.6mm print floor so nearby
        # green islands become one clean insert instead of printable confetti.
        parks_geom = _close_unprintable_bays(parks_geom, min_feature_m=area_cluster_merge_m)
        parks_geom = _enforce_min_width(parks_geom, min_feature_m=min_land_feature_m)
        parks_geom = _filter_tiny_polygon_parts(parks_geom, min_feature_m=min_land_feature_m)
        parks_groove_geom = _fill_tiny_holes(parks_groove_geom, min_feature_m=min_land_feature_m)
        parks_groove_geom = _close_unprintable_bays(parks_groove_geom, min_feature_m=area_cluster_merge_m)
        parks_groove_geom = _enforce_min_width(parks_groove_geom, min_feature_m=min_land_feature_m)
        parks_groove_geom = _filter_tiny_polygon_parts(parks_groove_geom, min_feature_m=min_land_feature_m)
        water_geom = _close_unprintable_bays(water_geom, min_feature_m=area_cluster_merge_m)
        water_geom = _enforce_min_width(water_geom, min_feature_m=min_land_feature_m)
        water_geom = _filter_tiny_polygon_parts(water_geom, min_feature_m=min_land_feature_m)
        water_groove_geom = _close_unprintable_bays(water_groove_geom, min_feature_m=area_cluster_merge_m)
        water_groove_geom = _enforce_min_width(water_groove_geom, min_feature_m=min_land_feature_m)
        water_groove_geom = _filter_tiny_polygon_parts(water_groove_geom, min_feature_m=min_land_feature_m)
        # Parks/water: drop outlier components that are a fraction of the
        # largest patch — they survive the width filter but print as specks.
        parks_geom = _drop_outlier_components(parks_geom)
        parks_groove_geom = _drop_outlier_components(parks_groove_geom)
        water_geom = _drop_outlier_components(water_geom)
        water_groove_geom = _drop_outlier_components(water_groove_geom)
        parks_geom = _drop_isolated_area_details(
            parks_geom,
            scale_factor=scale_factor,
            min_area_mm2=CANONICAL_PARK_ISOLATED_DETAIL_MIN_AREA_MM2,
        )
        water_geom = _drop_isolated_area_details(
            water_geom,
            scale_factor=scale_factor,
            min_area_mm2=CANONICAL_WATER_ISOLATED_DETAIL_MIN_AREA_MM2,
        )
        # Drop buildings that are too small to print — they create 3D extrusions
        # with non-manifold edges and punch unprintable voids through roads.
        # Same open-then-filter ordering: opening trims building stubs that are
        # an artifact of clip-against-roads, then the filter drops any tiny
        # remaining specks.
        buildings_geom = _enforce_min_width(buildings_geom, min_feature_m=min_land_feature_m)
        buildings_geom = _filter_tiny_polygon_parts(buildings_geom, min_feature_m=min_land_feature_m)
        if buildings_geom is None or getattr(buildings_geom, "is_empty", True):
            parks_building_exclusion = buildings_geom
            water_building_exclusion = buildings_geom

    # Pin every groove to the vicinity of its surviving final mask. Grooves and
    # final masks were filtered independently above — a road polygon might fall
    # below min-width and disappear while its outward-ring groove survived. The
    # result is a depression in the terrain with no insert to seat in it. Clip
    # each groove to the dilated footprint of its own final layer (drop it
    # entirely if the final layer was emptied).
    if groove_clearance_m > 0.0:
        proximity_m = max(float(groove_clearance_m) * 3.0, 1e-4)

        def _pin_groove(final_mask, groove_mask):
            if final_mask is None or getattr(final_mask, "is_empty", True):
                return None
            if groove_mask is None or getattr(groove_mask, "is_empty", True):
                return groove_mask
            try:
                proximity = final_mask.buffer(proximity_m, join_style=2).buffer(0)
                clipped = groove_mask.intersection(proximity).buffer(0)
                return clipped if clipped is not None and not getattr(clipped, "is_empty", True) else None
            except Exception:
                return groove_mask

        road_groove_geom = _pin_groove(roads_geom, road_groove_geom)
        parks_groove_geom = _pin_groove(parks_geom, parks_groove_geom)
        water_groove_geom = _pin_groove(water_geom, water_groove_geom)

    roads_geom = _fill_orphan_holes(roads_geom, backing_mask=None)
    road_groove_geom = _fill_orphan_holes(road_groove_geom, backing_mask=None)
    # Filling road holes is required for printability, but it can recreate
    # areas under building footprints / building-groove as solid road. Enforce
    # strict precedence one more time after hole-fill, including the groove.
    if buildings_geom is not None and not getattr(buildings_geom, "is_empty", True):
        parks_building_exclusion = _building_fit_exclusion_for_insert(
            buildings_geom=buildings_geom,
            insert_geom=parks_building_fit_source,
            groove_clearance_m=float(groove_clearance_m),
        )
        water_building_exclusion = _building_fit_exclusion_for_insert(
            buildings_geom=buildings_geom,
            insert_geom=water_building_fit_source,
            groove_clearance_m=float(groove_clearance_m),
        )
        # Roads remain authoritative. Never subtract buildings from roads here.
        parks_geom = _apply_mask_difference(parks_geom, parks_building_exclusion)
        parks_groove_geom = _apply_mask_difference(parks_groove_geom, parks_building_exclusion)
        water_geom = _apply_mask_difference(water_geom, water_building_exclusion)
        water_groove_geom = _apply_mask_difference(water_groove_geom, water_building_exclusion)

    # Strict layer precedence post-holefill: buildings > roads > parks > water.
    # Both inlay masks AND their groove rings must respect the precedence, else
    # two grooves overlap and we get a double-dig through the terrain base —
    # producing a through-hole in the print. The audit flags overlaps at any
    # pair (inlay/inlay, inlay/groove, groove/groove).
    if roads_geom is not None and not getattr(roads_geom, "is_empty", True):
        parks_geom = _apply_mask_difference(parks_geom, roads_geom)
        parks_groove_geom = _apply_mask_difference(parks_groove_geom, roads_geom)
        water_geom = _apply_mask_difference(water_geom, roads_geom)
        water_groove_geom = _apply_mask_difference(water_groove_geom, roads_geom)
    if road_groove_geom is not None and not getattr(road_groove_geom, "is_empty", True):
        parks_geom = _apply_mask_difference(parks_geom, road_groove_geom)
        parks_groove_geom = _apply_mask_difference(parks_groove_geom, road_groove_geom)
        water_geom = _apply_mask_difference(water_geom, road_groove_geom)
        water_groove_geom = _apply_mask_difference(water_groove_geom, road_groove_geom)
    if parks_geom is not None and not getattr(parks_geom, "is_empty", True):
        water_geom = _apply_mask_difference(water_geom, parks_geom)
        water_groove_geom = _apply_mask_difference(water_groove_geom, parks_geom)
    if parks_groove_geom is not None and not getattr(parks_groove_geom, "is_empty", True):
        water_geom = _apply_mask_difference(water_geom, parks_groove_geom)
        water_groove_geom = _apply_mask_difference(water_groove_geom, parks_groove_geom)

    # FINAL CLEANUP after all precedence subtractions.
    #
    # Subtracting buildings/roads/groove masks from already-filtered layers can
    # leave thin L-shaped slivers where a building cuts into a road, a road
    # crosses a park edge, etc. These slivers are narrower than min_feature_m
    # and the audit correctly flags them as small_components. We have to run
    # the morphological opening + per-polygon erosion filter once more on
    # every printable layer so the bundle written to disk is audit-clean by
    # construction. Without this pass: roads_final still contained 10 slivers
    # and road_groove_mask 15 slivers even after the earlier cleanup — the
    # per-zone audit then failed with `layers=roads_final,road_groove_mask`.
    if scale_factor is not None and scale_factor > 0.0:
        final_min_feature_m = _land_min_feature_m(scale_factor)

        def _cleanup_after_precedence(geom: BaseGeometry | None) -> BaseGeometry | None:
            if geom is None or getattr(geom, "is_empty", True):
                return geom
            opened = _enforce_min_width(geom, min_feature_m=final_min_feature_m)
            if opened is None or getattr(opened, "is_empty", True):
                return None
            filtered = _filter_tiny_polygon_parts(opened, min_feature_m=final_min_feature_m)
            return filtered

        roads_geom = _cleanup_after_precedence(roads_geom) if min_road_feature_m > 0.0 else roads_geom
        road_groove_geom = _cleanup_after_precedence(road_groove_geom) if min_road_feature_m > 0.0 else road_groove_geom
        parks_geom = _cleanup_after_precedence(parks_geom)
        parks_groove_geom = _cleanup_after_precedence(parks_groove_geom)
        water_geom = _cleanup_after_precedence(water_geom)
        water_groove_geom = _cleanup_after_precedence(water_groove_geom)
        buildings_geom = _cleanup_after_precedence(buildings_geom)

        # Re-pin grooves to (possibly shrunken) final masks so groove-only
        # remnants with no backing inlay are dropped.
        if groove_clearance_m > 0.0:
            proximity_m = max(float(groove_clearance_m) * 3.0, 1e-4)

            def _pin_groove_final(final_mask, groove_mask):
                if final_mask is None or getattr(final_mask, "is_empty", True):
                    return None
                if groove_mask is None or getattr(groove_mask, "is_empty", True):
                    return groove_mask
                try:
                    proximity = final_mask.buffer(proximity_m, join_style=2).buffer(0)
                    clipped = groove_mask.intersection(proximity).buffer(0)
                    return clipped if clipped is not None and not getattr(clipped, "is_empty", True) else None
                except Exception:
                    return groove_mask

            road_groove_geom = _pin_groove_final(roads_geom, road_groove_geom)
            parks_groove_geom = _pin_groove_final(parks_geom, parks_groove_geom)
            water_groove_geom = _pin_groove_final(water_geom, water_groove_geom)

        # Re-enforce strict layer precedence since the cleanup may have
        # marginally grown a layer at its boundaries (the buffer(+erosion)
        # step in opening rounds convex corners outward).
        if roads_geom is not None and not getattr(roads_geom, "is_empty", True):
            parks_geom = _apply_mask_difference(parks_geom, roads_geom)
            parks_groove_geom = _apply_mask_difference(parks_groove_geom, roads_geom)
            water_geom = _apply_mask_difference(water_geom, roads_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, roads_geom)
        if road_groove_geom is not None and not getattr(road_groove_geom, "is_empty", True):
            parks_geom = _apply_mask_difference(parks_geom, road_groove_geom)
            parks_groove_geom = _apply_mask_difference(parks_groove_geom, road_groove_geom)
            water_geom = _apply_mask_difference(water_geom, road_groove_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, road_groove_geom)
        if buildings_geom is not None and not getattr(buildings_geom, "is_empty", True):
            # Buildings already yielded or won locally inside
            # _resolve_building_road_conflicts near bundle assembly start.
            # Never notch the authoritative road insert here.
            parks_geom = _apply_mask_difference(parks_geom, parks_building_exclusion)
            parks_groove_geom = _apply_mask_difference(parks_groove_geom, parks_building_exclusion)
            water_geom = _apply_mask_difference(water_geom, water_building_exclusion)
            water_groove_geom = _apply_mask_difference(water_groove_geom, water_building_exclusion)

        # One more filter pass — the precedence re-enforcement above may have
        # created fresh slivers at subtraction boundaries. Using just the
        # per-polygon erosion filter here (no opening) avoids unnecessarily
        # reshaping the layers again; opening already ran once in
        # _cleanup_after_precedence.
        if not road_map_faithful_mode:
            roads_geom = _filter_tiny_polygon_parts(roads_geom, min_feature_m=min_road_feature_m)
            road_groove_geom = _filter_tiny_polygon_parts(road_groove_geom, min_feature_m=min_road_feature_m)
        parks_geom = _filter_tiny_polygon_parts(parks_geom, min_feature_m=final_min_feature_m)
        parks_groove_geom = _filter_tiny_polygon_parts(parks_groove_geom, min_feature_m=final_min_feature_m)
        water_geom = _filter_tiny_polygon_parts(water_geom, min_feature_m=final_min_feature_m)
        water_groove_geom = _filter_tiny_polygon_parts(water_groove_geom, min_feature_m=final_min_feature_m)
        buildings_geom = _filter_tiny_polygon_parts(buildings_geom, min_feature_m=final_min_feature_m)

        # Final post-precedence hole heal.
        #
        # The last subtraction round (roads/buildings precedence) can re-create
        # tiny and orphan interior rings inside road layers, especially in
        # branchy junctions near clipped building outlines. Those holes fail the
        # canonical printability audit as:
        #   - layers=road_groove_mask (small_hole_count)
        #   - road_holes=road_groove_orphan_holes
        #
        # Re-apply the same hole cleanup at the very end so the written bundle
        # is audit-clean by construction.
        road_hole_fill_m = 0.0
        if not road_map_faithful_mode:
            roads_geom = _fill_tiny_holes(roads_geom, min_feature_m=road_hole_fill_m)
            road_groove_geom = _fill_tiny_holes(road_groove_geom, min_feature_m=road_hole_fill_m)
            roads_geom = _fill_orphan_holes(roads_geom, backing_mask=None)
            road_groove_geom = _fill_orphan_holes(road_groove_geom, backing_mask=None)

        # Final sliver prune after post-holefill building subtraction.
        if not road_map_faithful_mode:
            roads_geom = _enforce_min_width(roads_geom, min_feature_m=min_road_feature_m)
            road_groove_geom = _enforce_min_width(road_groove_geom, min_feature_m=min_road_feature_m)
            roads_geom = _filter_tiny_polygon_parts(roads_geom, min_feature_m=min_road_feature_m)
            road_groove_geom = _filter_tiny_polygon_parts(road_groove_geom, min_feature_m=min_road_feature_m)
        # NOTE: keep road components here. If roads still appear as detached
        # dots, root cause is usually under-merged source topology (gap-fill
        # threshold too low) rather than speck filtering. We avoid aggressive
        # post-filter pruning to prevent deleting valid short road segments.

        # Absolute last topology cleanup after all precedence differences.
        #
        # Small interior holes can be recreated by the final building/road
        # subtraction and min-width opening passes. Those holes are not a real
        # map feature anymore: they are tiny boolean leftovers that later fail
        # the canonical audit (`small_hole_count` / `*_orphan_holes`).
        #
        # We heal them here, then immediately re-apply strict precedence so a
        # filled hole can never resurrect overlap with roads/buildings/water.
        if not road_map_faithful_mode:
            roads_geom = _fill_tiny_holes(roads_geom, min_feature_m=min_road_feature_m)
            road_groove_geom = _fill_tiny_holes(road_groove_geom, min_feature_m=min_road_feature_m)
        parks_geom = _fill_tiny_holes(parks_geom, min_feature_m=final_min_feature_m)
        parks_groove_geom = _fill_tiny_holes(parks_groove_geom, min_feature_m=final_min_feature_m)
        water_geom = _fill_tiny_holes(water_geom, min_feature_m=final_min_feature_m)
        water_groove_geom = _fill_tiny_holes(water_groove_geom, min_feature_m=final_min_feature_m)

        if not road_map_faithful_mode:
            roads_geom = _fill_orphan_holes(roads_geom, backing_mask=buildings_geom)
            road_groove_geom = _fill_orphan_holes(road_groove_geom, backing_mask=buildings_geom)

        if roads_geom is not None and not getattr(roads_geom, "is_empty", True):
            parks_geom = _apply_mask_difference(parks_geom, roads_geom)
            parks_groove_geom = _apply_mask_difference(parks_groove_geom, roads_geom)
            water_geom = _apply_mask_difference(water_geom, roads_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, roads_geom)
        if road_groove_geom is not None and not getattr(road_groove_geom, "is_empty", True):
            parks_geom = _apply_mask_difference(parks_geom, road_groove_geom)
            parks_groove_geom = _apply_mask_difference(parks_groove_geom, road_groove_geom)
            water_geom = _apply_mask_difference(water_geom, road_groove_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, road_groove_geom)
        if parks_geom is not None and not getattr(parks_geom, "is_empty", True):
            water_geom = _apply_mask_difference(water_geom, parks_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, parks_geom)
        if parks_groove_geom is not None and not getattr(parks_groove_geom, "is_empty", True):
            water_geom = _apply_mask_difference(water_geom, parks_groove_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, parks_groove_geom)
        if buildings_geom is not None and not getattr(buildings_geom, "is_empty", True):
            parks_geom = _apply_mask_difference(parks_geom, parks_building_exclusion)
            parks_groove_geom = _apply_mask_difference(parks_groove_geom, parks_building_exclusion)
            water_geom = _apply_mask_difference(water_geom, water_building_exclusion)
            water_groove_geom = _apply_mask_difference(water_groove_geom, water_building_exclusion)

        # Absolute terminal topology heal.
        #
        # The final precedence subtraction above can re-introduce microscopic
        # numeric holes at polygon seams (for example ~1e-8 m^2 rings created
        # by buffer/difference round-trips). Those are not semantic map
        # features, but the canonical audit still sees them as orphan road
        # holes and fails the zone. Heal them after ALL precedence ops so no
        # later boolean can recreate them.
        if not road_map_faithful_mode:
            roads_geom = _fill_tiny_holes(roads_geom, min_feature_m=min_road_feature_m)
            road_groove_geom = _fill_tiny_holes(road_groove_geom, min_feature_m=min_road_feature_m)
        parks_geom = _fill_tiny_holes(parks_geom, min_feature_m=final_min_feature_m)
        parks_groove_geom = _fill_tiny_holes(parks_groove_geom, min_feature_m=final_min_feature_m)
        water_geom = _fill_tiny_holes(water_geom, min_feature_m=final_min_feature_m)
        water_groove_geom = _fill_tiny_holes(water_groove_geom, min_feature_m=final_min_feature_m)
        if not road_map_faithful_mode:
            roads_geom = _fill_orphan_holes(roads_geom, backing_mask=buildings_geom)
            road_groove_geom = _fill_orphan_holes(road_groove_geom, backing_mask=buildings_geom)

        # Even in faithful road mode, heal only very small audit-failing holes
        # left behind by late-stage booleans. This keeps ring roads and real
        # city blocks intact while removing holes that can never print anyway.
        roads_geom = _fill_unprintable_holes(roads_geom, min_feature_m=final_min_feature_m)
        road_groove_geom = _fill_unprintable_holes(road_groove_geom, min_feature_m=final_min_feature_m)

        # Road faithful mode preserves topology, but raw mitre tips at clipped
        # turns still create razor corners that chip or vanish in FDM. Round the
        # road insert itself with a tiny guarded open, then derive the groove
        # from that final insert so the 0.15mm fit remains consistent.
        roads_geom = _round_road_corners_for_print(
            roads_geom,
            scale_factor=scale_factor,
        )

        # Final road groove must be derived from the final road insert so the
        # 0.15mm fit is guaranteed even after late-stage numeric cleanup.
        road_groove_geom = _build_road_groove_from_insert(
            roads_geom,
            groove_clearance_m=float(groove_clearance_m),
            zone_geom=zone_geom,
        )
        if buildings_geom is not None and not getattr(buildings_geom, "is_empty", True):
            final_building_conflict = [
                geom
                for geom in (roads_geom, road_groove_geom)
                if geom is not None and not getattr(geom, "is_empty", True)
            ]
            if final_building_conflict:
                try:
                    buildings_geom = _apply_mask_difference(
                        buildings_geom,
                        unary_union(final_building_conflict).buffer(0),
                    )
                except Exception:
                    pass
            parks_building_exclusion = _building_fit_exclusion_for_insert(
                buildings_geom=buildings_geom,
                insert_geom=parks_building_fit_source,
                groove_clearance_m=float(groove_clearance_m),
            )
            water_building_exclusion = _building_fit_exclusion_for_insert(
                buildings_geom=buildings_geom,
                insert_geom=water_building_fit_source,
                groove_clearance_m=float(groove_clearance_m),
            )
            parks_geom = _apply_mask_difference(parks_geom, parks_building_exclusion)
            parks_groove_geom = _apply_mask_difference(parks_groove_geom, parks_building_exclusion)
            water_geom = _apply_mask_difference(water_geom, water_building_exclusion)
            water_groove_geom = _apply_mask_difference(water_groove_geom, water_building_exclusion)
        if road_groove_geom is not None and not getattr(road_groove_geom, "is_empty", True):
            parks_geom = _apply_mask_difference(parks_geom, road_groove_geom)
            parks_groove_geom = _apply_mask_difference(parks_groove_geom, road_groove_geom)
            water_geom = _apply_mask_difference(water_geom, road_groove_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, road_groove_geom)

    if scale_factor is not None and scale_factor > 0.0:
        partition_min_feature_m = _land_min_feature_m(scale_factor)
        area_cluster_merge_m = max(
            float(partition_min_feature_m),
            _model_mm_to_world_m(CANONICAL_AREA_CLUSTER_MERGE_MM, scale_factor),
        )

        parks_geom, water_geom = _absorb_unprintable_remainders_into_inlays(
            zone_geom=zone_geom,
            roads_geom=roads_geom,
            road_groove_geom=road_groove_geom,
            parks_geom=parks_geom,
            water_geom=water_geom,
            buildings_geom=buildings_geom,
            min_feature_m=float(partition_min_feature_m),
            scale_factor=scale_factor,
        )

        parks_geom = _close_unprintable_bays(parks_geom, min_feature_m=area_cluster_merge_m)
        water_geom = _close_unprintable_bays(water_geom, min_feature_m=area_cluster_merge_m)
        parks_geom = _sanitize_nonroad_insert(
            parks_geom,
            min_feature_m=float(partition_min_feature_m),
            close_bays=False,
        )
        water_geom = _sanitize_nonroad_insert(
            water_geom,
            min_feature_m=float(partition_min_feature_m),
            close_bays=False,
        )
        parks_geom = _drop_isolated_area_details(
            parks_geom,
            scale_factor=scale_factor,
            min_area_mm2=CANONICAL_PARK_ISOLATED_DETAIL_MIN_AREA_MM2,
        )
        water_geom = _drop_isolated_area_details(
            water_geom,
            scale_factor=scale_factor,
            min_area_mm2=CANONICAL_WATER_ISOLATED_DETAIL_MIN_AREA_MM2,
        )
        buildings_geom = _sanitize_nonroad_insert(
            buildings_geom,
            min_feature_m=float(partition_min_feature_m),
            close_bays=False,
        )

        # Rebuild inlay grooves from the final, absorbed insert masks. This
        # prevents groove-only leftovers and keeps the groove/insert fit from
        # drifting after small terrain scraps are absorbed into parks/water.
        parks_groove_geom = _build_inlay_groove_from_insert(
            parks_geom,
            groove_clearance_m=float(groove_clearance_m),
            zone_geom=zone_geom,
            hard_exclusions=[parks_building_exclusion, roads_geom, road_groove_geom],
        )
        water_groove_geom = _build_inlay_groove_from_insert(
            water_geom,
            groove_clearance_m=float(groove_clearance_m),
            zone_geom=zone_geom,
            hard_exclusions=[water_building_exclusion, roads_geom, road_groove_geom, parks_geom, parks_groove_geom],
        )

        if roads_geom is not None and not getattr(roads_geom, "is_empty", True):
            parks_geom = _apply_mask_difference(parks_geom, roads_geom)
            parks_groove_geom = _apply_mask_difference(parks_groove_geom, roads_geom)
            water_geom = _apply_mask_difference(water_geom, roads_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, roads_geom)
        if road_groove_geom is not None and not getattr(road_groove_geom, "is_empty", True):
            parks_geom = _apply_mask_difference(parks_geom, road_groove_geom)
            parks_groove_geom = _apply_mask_difference(parks_groove_geom, road_groove_geom)
            water_geom = _apply_mask_difference(water_geom, road_groove_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, road_groove_geom)
        if buildings_geom is not None and not getattr(buildings_geom, "is_empty", True):
            parks_geom = _apply_mask_difference(parks_geom, parks_building_exclusion)
            parks_groove_geom = _apply_mask_difference(parks_groove_geom, parks_building_exclusion)
            water_geom = _apply_mask_difference(water_geom, water_building_exclusion)
            water_groove_geom = _apply_mask_difference(water_groove_geom, water_building_exclusion)
        if parks_geom is not None and not getattr(parks_geom, "is_empty", True):
            water_geom = _apply_mask_difference(water_geom, parks_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, parks_geom)
        if parks_groove_geom is not None and not getattr(parks_groove_geom, "is_empty", True):
            water_geom = _apply_mask_difference(water_geom, parks_groove_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, parks_groove_geom)

        parks_geom = _sanitize_nonroad_insert(
            parks_geom,
            min_feature_m=float(partition_min_feature_m),
            close_bays=False,
        )
        water_geom = _sanitize_nonroad_insert(
            water_geom,
            min_feature_m=float(partition_min_feature_m),
            close_bays=False,
        )

        parks_groove_geom = _build_inlay_groove_from_insert(
            parks_geom,
            groove_clearance_m=float(groove_clearance_m),
            zone_geom=zone_geom,
            hard_exclusions=[parks_building_exclusion, roads_geom, road_groove_geom],
        )
        water_groove_geom = _build_inlay_groove_from_insert(
            water_geom,
            groove_clearance_m=float(groove_clearance_m),
            zone_geom=zone_geom,
            hard_exclusions=[water_building_exclusion, roads_geom, road_groove_geom, parks_geom, parks_groove_geom],
        )

        parks_groove_geom = _sanitize_nonroad_insert(
            parks_groove_geom,
            min_feature_m=float(partition_min_feature_m),
            close_bays=False,
        )
        water_groove_geom = _sanitize_nonroad_insert(
            water_groove_geom,
            min_feature_m=float(partition_min_feature_m),
            close_bays=False,
        )

        if roads_geom is not None and not getattr(roads_geom, "is_empty", True):
            parks_geom = _apply_mask_difference(parks_geom, roads_geom)
            parks_groove_geom = _apply_mask_difference(parks_groove_geom, roads_geom)
            water_geom = _apply_mask_difference(water_geom, roads_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, roads_geom)
        if road_groove_geom is not None and not getattr(road_groove_geom, "is_empty", True):
            parks_geom = _apply_mask_difference(parks_geom, road_groove_geom)
            parks_groove_geom = _apply_mask_difference(parks_groove_geom, road_groove_geom)
            water_geom = _apply_mask_difference(water_geom, road_groove_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, road_groove_geom)
        if buildings_geom is not None and not getattr(buildings_geom, "is_empty", True):
            parks_geom = _apply_mask_difference(parks_geom, parks_building_exclusion)
            parks_groove_geom = _apply_mask_difference(parks_groove_geom, parks_building_exclusion)
            water_geom = _apply_mask_difference(water_geom, water_building_exclusion)
            water_groove_geom = _apply_mask_difference(water_groove_geom, water_building_exclusion)
        if parks_geom is not None and not getattr(parks_geom, "is_empty", True):
            water_geom = _apply_mask_difference(water_geom, parks_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, parks_geom)
        if parks_groove_geom is not None and not getattr(parks_groove_geom, "is_empty", True):
            water_geom = _apply_mask_difference(water_geom, parks_groove_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, parks_groove_geom)

        parks_geom = _fill_unbacked_insert_holes(
            parks_geom,
            backing_masks=[roads_geom, road_groove_geom, buildings_geom, water_geom, water_groove_geom],
        )
        water_geom = _fill_unbacked_insert_holes(
            water_geom,
            backing_masks=[roads_geom, road_groove_geom, buildings_geom, parks_geom, parks_groove_geom],
        )
        parks_geom = _filter_tiny_polygon_parts(parks_geom, min_feature_m=float(partition_min_feature_m))
        parks_groove_geom = _filter_tiny_polygon_parts(parks_groove_geom, min_feature_m=float(partition_min_feature_m))
        water_geom = _filter_tiny_polygon_parts(water_geom, min_feature_m=float(partition_min_feature_m))
        water_groove_geom = _filter_tiny_polygon_parts(water_groove_geom, min_feature_m=float(partition_min_feature_m))
        parks_geom = _drop_isolated_area_details(
            parks_geom,
            scale_factor=scale_factor,
            min_area_mm2=CANONICAL_PARK_ISOLATED_DETAIL_MIN_AREA_MM2,
        )
        water_geom = _drop_isolated_area_details(
            water_geom,
            scale_factor=scale_factor,
            min_area_mm2=CANONICAL_WATER_ISOLATED_DETAIL_MIN_AREA_MM2,
        )
        parks_geom = _fill_unprintable_holes(
            parks_geom,
            min_feature_m=float(partition_min_feature_m),
            max_area_factor=None,
        )
        parks_groove_geom = _fill_unprintable_holes(
            parks_groove_geom,
            min_feature_m=float(partition_min_feature_m),
            max_area_factor=None,
        )
        water_geom = _fill_unprintable_holes(
            water_geom,
            min_feature_m=float(partition_min_feature_m),
            max_area_factor=None,
        )
        water_groove_geom = _fill_unprintable_holes(
            water_groove_geom,
            min_feature_m=float(partition_min_feature_m),
            max_area_factor=None,
        )

        # Terminal fill can legitimately close visual pinholes, but it must
        # never resurrect overlap with authoritative masks. Re-apply
        # precedence after the final fill, then only prune disconnected specks
        # (no more opening/closing that could grow boundaries again).
        if roads_geom is not None and not getattr(roads_geom, "is_empty", True):
            parks_geom = _apply_mask_difference(parks_geom, roads_geom)
            parks_groove_geom = _apply_mask_difference(parks_groove_geom, roads_geom)
            water_geom = _apply_mask_difference(water_geom, roads_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, roads_geom)
        if road_groove_geom is not None and not getattr(road_groove_geom, "is_empty", True):
            parks_geom = _apply_mask_difference(parks_geom, road_groove_geom)
            parks_groove_geom = _apply_mask_difference(parks_groove_geom, road_groove_geom)
            water_geom = _apply_mask_difference(water_geom, road_groove_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, road_groove_geom)
        if buildings_geom is not None and not getattr(buildings_geom, "is_empty", True):
            parks_geom = _apply_mask_difference(parks_geom, parks_building_exclusion)
            parks_groove_geom = _apply_mask_difference(parks_groove_geom, parks_building_exclusion)
            water_geom = _apply_mask_difference(water_geom, water_building_exclusion)
            water_groove_geom = _apply_mask_difference(water_groove_geom, water_building_exclusion)
        if parks_geom is not None and not getattr(parks_geom, "is_empty", True):
            water_geom = _apply_mask_difference(water_geom, parks_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, parks_geom)
        if parks_groove_geom is not None and not getattr(parks_groove_geom, "is_empty", True):
            water_geom = _apply_mask_difference(water_geom, parks_groove_geom)
            water_groove_geom = _apply_mask_difference(water_groove_geom, parks_groove_geom)

        parks_geom = _fill_unbacked_insert_holes(
            parks_geom,
            backing_masks=[roads_geom, road_groove_geom, buildings_geom, water_geom, water_groove_geom],
        )
        water_geom = _fill_unbacked_insert_holes(
            water_geom,
            backing_masks=[roads_geom, road_groove_geom, buildings_geom, parks_geom, parks_groove_geom],
        )
        parks_geom = _filter_tiny_polygon_parts(parks_geom, min_feature_m=float(partition_min_feature_m))
        parks_groove_geom = _filter_tiny_polygon_parts(parks_groove_geom, min_feature_m=float(partition_min_feature_m))
        water_geom = _filter_tiny_polygon_parts(water_geom, min_feature_m=float(partition_min_feature_m))
        water_groove_geom = _filter_tiny_polygon_parts(water_groove_geom, min_feature_m=float(partition_min_feature_m))
        parks_geom = _drop_isolated_area_details(
            parks_geom,
            scale_factor=scale_factor,
            min_area_mm2=CANONICAL_PARK_ISOLATED_DETAIL_MIN_AREA_MM2,
        )
        water_geom = _drop_isolated_area_details(
            water_geom,
            scale_factor=scale_factor,
            min_area_mm2=CANONICAL_WATER_ISOLATED_DETAIL_MIN_AREA_MM2,
        )
        parks_groove_geom = _build_inlay_groove_from_insert(
            parks_geom,
            groove_clearance_m=float(groove_clearance_m),
            zone_geom=zone_geom,
            hard_exclusions=[parks_building_exclusion, roads_geom, road_groove_geom],
        )
        water_groove_geom = _build_inlay_groove_from_insert(
            water_geom,
            groove_clearance_m=float(groove_clearance_m),
            zone_geom=zone_geom,
            hard_exclusions=[water_building_exclusion, roads_geom, road_groove_geom, parks_geom, parks_groove_geom],
        )
        parks_groove_geom = _sanitize_nonroad_insert(
            parks_groove_geom,
            min_feature_m=float(partition_min_feature_m),
            close_bays=False,
        )
        water_groove_geom = _sanitize_nonroad_insert(
            water_groove_geom,
            min_feature_m=float(partition_min_feature_m),
            close_bays=False,
        )

    # Last-mile precedence after all hole-fill/sanitize passes: those passes can
    # legally close pinholes, but they must not refill the building fit voids
    # inside park/water inserts or their terrain grooves.
    if buildings_geom is not None and not getattr(buildings_geom, "is_empty", True):
        final_parks_building_exclusion = _building_fit_exclusion_for_insert(
            buildings_geom=buildings_geom,
            insert_geom=parks_geom,
            groove_clearance_m=float(groove_clearance_m),
        )
        final_water_building_exclusion = _building_fit_exclusion_for_insert(
            buildings_geom=buildings_geom,
            insert_geom=water_geom,
            groove_clearance_m=float(groove_clearance_m),
        )
        parks_geom = _apply_mask_difference(parks_geom, final_parks_building_exclusion)
        parks_groove_geom = _apply_mask_difference(parks_groove_geom, final_parks_building_exclusion)
        water_geom = _apply_mask_difference(water_geom, final_water_building_exclusion)
        water_groove_geom = _apply_mask_difference(water_groove_geom, final_water_building_exclusion)

    terrain_bare_geom = zone_geom
    terrain_land_geom = zone_geom
    blockers = [
        geom
        for geom in (roads_geom, parks_geom, water_geom, buildings_geom)
        if geom is not None and not getattr(geom, "is_empty", True)
    ]
    land_blockers = [
        geom
        for geom in (roads_geom, water_geom, buildings_geom)
        if geom is not None and not getattr(geom, "is_empty", True)
    ]
    if terrain_bare_geom is not None and blockers:
        try:
            terrain_bare_geom = terrain_bare_geom.difference(unary_union(blockers)).buffer(0)
        except Exception:
            pass
    if terrain_land_geom is not None and land_blockers:
        try:
            terrain_land_geom = terrain_land_geom.difference(unary_union(land_blockers)).buffer(0)
        except Exception:
            pass

    if scale_factor is not None and scale_factor > 0.0:
        # Same scale_factor=print_mm/world_m convention as above; correct
        # world-meters conversion is print_mm / scale_factor (NOT * / 1000).
        min_land_feature_m = _land_min_feature_m(scale_factor)
        # Terrain is the "everything else" layer. Do not run morphological
        # opening on it (that can delete legitimate strips between roads), but
        # do fill any interior cut-out that cannot survive the 0.6mm printable
        # erosion test. Those cut-outs become visible beige pits/slivers in
        # the final render and do not print cleanly as separate detail.
        terrain_bare_geom = _fill_unprintable_holes(
            terrain_bare_geom,
            min_feature_m=min_land_feature_m,
            max_area_factor=None,
        )
        terrain_bare_geom = _filter_tiny_polygon_parts(terrain_bare_geom, min_feature_m=min_land_feature_m)
        terrain_bare_geom = _drop_outlier_components(terrain_bare_geom)
        terrain_land_geom = _fill_unprintable_holes(
            terrain_land_geom,
            min_feature_m=min_land_feature_m,
            max_area_factor=None,
        )
        terrain_land_geom = _filter_tiny_polygon_parts(terrain_land_geom, min_feature_m=min_land_feature_m)
        terrain_land_geom = _drop_outlier_components(terrain_land_geom)

    _write_geojson(bundle_dir / "zone_polygon.geojson", zone_geom)
    _write_geojson(bundle_dir / "roads_final.geojson", roads_geom)
    _write_geojson(bundle_dir / "road_groove_mask.geojson", road_groove_geom)
    _write_geojson(bundle_dir / "parks_final.geojson", parks_geom)
    _write_geojson(bundle_dir / "parks_groove_mask.geojson", parks_groove_geom)
    _write_geojson(bundle_dir / "water_final.geojson", water_geom)
    _write_geojson(bundle_dir / "water_groove_mask.geojson", water_groove_geom)
    _write_geojson(bundle_dir / "buildings_footprints.geojson", buildings_geom)
    _write_geojson(bundle_dir / "roads_semantic_preview.geojson", roads_semantic_geom)
    _write_geojson(bundle_dir / "terrain_bare_mask.geojson", terrain_bare_geom)
    _write_geojson(bundle_dir / "terrain_land_mask.geojson", terrain_land_geom)

    manifest = {
        "task_id": task_id,
        "source": "runtime_canonical_masks",
        "scale_factor": float(scale_factor) if scale_factor is not None else None,
        "bundle_dir": str(bundle_dir),
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_canonical_mask_bundle(bundle_dir)
