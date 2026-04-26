from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from services.detail_layer_utils import (
    MIN_LAND_WIDTH_MODEL_MM,
    MIN_ROAD_WIDTH_MODEL_MM,
    model_mm_to_world_m,
)


MIN_PRINTABLE_GAP_MM = 0.6
DEFAULT_CLEARANCE_MM = 0.2


@dataclass
class SlicerConfig:
    scale_factor: float
    min_width_mm: float = MIN_ROAD_WIDTH_MODEL_MM
    gap_fill_mm: float = MIN_PRINTABLE_GAP_MM
    corner_smooth_mm: float = 0.3
    tiny_feature_mm: float = 0.7


def _safe_buffer(geom: BaseGeometry, distance: float, **kwargs) -> Optional[BaseGeometry]:
    if geom is None or getattr(geom, "is_empty", True):
        return geom
    try:
        result = geom.buffer(distance, **kwargs)
        if result is None or result.is_empty:
            return None
        return result
    except Exception:
        return geom


def _clean(geom: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    if geom is None or getattr(geom, "is_empty", True):
        return None
    try:
        return geom.buffer(0)
    except Exception:
        return geom


def enforce_min_width(geom: Optional[BaseGeometry], min_width_m: float) -> Optional[BaseGeometry]:
    """Morphological open with half min_width, then dilate back to preserve edges.

    Removes features narrower than `min_width_m` and widens sub-threshold regions
    so the printed width is at least `min_width_m` everywhere.
    """
    if geom is None or getattr(geom, "is_empty", True) or min_width_m <= 0:
        return geom
    half = float(min_width_m) / 2.0
    opened = _safe_buffer(geom, -half, join_style=2, resolution=8)
    if opened is None or getattr(opened, "is_empty", True):
        return None
    grown = _safe_buffer(opened, half, join_style=2, resolution=8)
    return _clean(grown)


def merge_close_gaps(geom: Optional[BaseGeometry], gap_m: float) -> Optional[BaseGeometry]:
    if geom is None or getattr(geom, "is_empty", True) or gap_m <= 0:
        return geom
    half = float(gap_m) / 2.0
    closed = _safe_buffer(geom, half, join_style=2, resolution=8)
    if closed is None or getattr(closed, "is_empty", True):
        return geom
    shrunk = _safe_buffer(closed, -half, join_style=2, resolution=8)
    return _clean(shrunk) or geom


def smooth_corners(geom: Optional[BaseGeometry], radius_m: float) -> Optional[BaseGeometry]:
    if geom is None or getattr(geom, "is_empty", True) or radius_m <= 0:
        return geom
    inflated = _safe_buffer(geom, radius_m, join_style=1, resolution=8)
    if inflated is None or getattr(inflated, "is_empty", True):
        return geom
    deflated = _safe_buffer(inflated, -radius_m, join_style=1, resolution=8)
    return _clean(deflated) or geom


def remove_tiny_features(geom: Optional[BaseGeometry], min_area_m2: float) -> Optional[BaseGeometry]:
    if geom is None or getattr(geom, "is_empty", True) or min_area_m2 <= 0:
        return geom
    try:
        if hasattr(geom, "geoms"):
            kept = [g for g in geom.geoms if g is not None and not g.is_empty and g.area >= min_area_m2]
            if not kept:
                return None
            return unary_union(kept)
        if geom.area < min_area_m2:
            return None
        return geom
    except Exception:
        return geom


def slice_mask(
    geom: Optional[BaseGeometry],
    *,
    config: SlicerConfig,
    subtract: Optional[BaseGeometry] = None,
    clearance_mm: float = 0.0,
) -> Optional[BaseGeometry]:
    """Apply a full slicer-style cleanup pass to a 2D mask.

    Order of operations mirrors a FDM slicer's inset logic:
      1. Merge close gaps below the printable threshold.
      2. Smooth sharp corners.
      3. Enforce minimum printable width.
      4. Subtract higher-precedence layers (with optional clearance).
      5. Drop tiny unprintable features.
    """
    if geom is None or getattr(geom, "is_empty", True):
        return None
    scale = float(config.scale_factor)
    if scale <= 0:
        return geom

    min_width_m = model_mm_to_world_m(config.min_width_mm, scale)
    gap_m = model_mm_to_world_m(config.gap_fill_mm, scale)
    corner_m = model_mm_to_world_m(config.corner_smooth_mm, scale)
    tiny_m = model_mm_to_world_m(config.tiny_feature_mm, scale)

    result = _clean(geom)
    if result is None:
        return None

    result = merge_close_gaps(result, gap_m)
    result = smooth_corners(result, corner_m)
    result = enforce_min_width(result, min_width_m)
    if result is None or getattr(result, "is_empty", True):
        return None

    if subtract is not None and not getattr(subtract, "is_empty", True):
        clearance_m = model_mm_to_world_m(clearance_mm, scale) if clearance_mm > 0 else 0.0
        expanded = subtract
        if clearance_m > 0:
            buffered = _safe_buffer(subtract, clearance_m, join_style=2, resolution=8)
            if buffered is not None and not getattr(buffered, "is_empty", True):
                expanded = buffered
        try:
            result = result.difference(expanded)
            result = _clean(result)
        except Exception:
            pass

    if result is None or getattr(result, "is_empty", True):
        return None

    result = remove_tiny_features(result, tiny_m * tiny_m)
    return result


def slice_road_mask(
    roads: Optional[BaseGeometry],
    *,
    scale_factor: float,
    buildings: Optional[BaseGeometry] = None,
    building_clearance_mm: float = DEFAULT_CLEARANCE_MM,
) -> Optional[BaseGeometry]:
    cfg = SlicerConfig(
        scale_factor=scale_factor,
        min_width_mm=MIN_ROAD_WIDTH_MODEL_MM,
        gap_fill_mm=MIN_PRINTABLE_GAP_MM,
        corner_smooth_mm=0.3,
        tiny_feature_mm=0.7,
    )
    return slice_mask(roads, config=cfg, subtract=buildings, clearance_mm=building_clearance_mm)


def slice_land_mask(
    land: Optional[BaseGeometry],
    *,
    scale_factor: float,
    subtract: Optional[BaseGeometry] = None,
    clearance_mm: float = DEFAULT_CLEARANCE_MM,
) -> Optional[BaseGeometry]:
    cfg = SlicerConfig(
        scale_factor=scale_factor,
        min_width_mm=MIN_LAND_WIDTH_MODEL_MM,
        gap_fill_mm=MIN_PRINTABLE_GAP_MM,
        corner_smooth_mm=0.3,
        tiny_feature_mm=0.7,
    )
    return slice_mask(land, config=cfg, subtract=subtract, clearance_mm=clearance_mm)
