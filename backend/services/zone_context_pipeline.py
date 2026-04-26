from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from shapely.geometry.base import BaseGeometry

from services.crs_utils import bbox_latlon_to_utm


@dataclass
class ZoneContextResult:
    bbox_meters: tuple[float, float, float, float]
    scale_factor: float


def build_zone_context(
    *,
    request: Any,
    global_center: Any,
    zone_polygon_local: Optional[BaseGeometry],
    reference_xy_m: Optional[tuple[float, float]],
    zone_prefix: str = "",
) -> ZoneContextResult:
    if zone_polygon_local is not None and not zone_polygon_local.is_empty:
        bounds = zone_polygon_local.bounds
        bbox_meters = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
        print(f"[DEBUG] {zone_prefix} Zone bbox from polygon (local): {bbox_meters}")
    else:
        bbox_utm_result = bbox_latlon_to_utm(request.north, request.south, request.east, request.west)
        minx_utm, miny_utm, maxx_utm, maxy_utm = bbox_utm_result[:4]
        minx_local, miny_local = global_center.to_local(minx_utm, miny_utm)
        maxx_local, maxy_local = global_center.to_local(maxx_utm, maxy_utm)
        bbox_meters = (float(minx_local), float(miny_local), float(maxx_local), float(maxy_local))
        print(f"[DEBUG] {zone_prefix} Zone bbox from request bounds (local): {bbox_meters}")

    scale_factor: Optional[float] = None
    try:
        if reference_xy_m is not None:
            size_x, size_y = float(reference_xy_m[0]), float(reference_xy_m[1])
            if size_x > 0 and size_y > 0:
                avg_xy = (size_x + size_y) / 2.0
                if avg_xy > 0:
                    scale_factor = float(request.model_size_mm) / float(avg_xy)
                    print(
                        f"[DEBUG] {zone_prefix} Scale factor (polygon): {scale_factor:.6f} mm/m "
                        f"(reference: {size_x:.1f} x {size_y:.1f} m)"
                    )
            elif size_x > 0:
                scale_factor = float(request.model_size_mm) / float(size_x)
                print(f"[DEBUG] {zone_prefix} Scale factor (polygon, X only): {scale_factor:.6f} mm/m")
            elif size_y > 0:
                scale_factor = float(request.model_size_mm) / float(size_y)
                print(f"[DEBUG] {zone_prefix} Scale factor (polygon, Y only): {scale_factor:.6f} mm/m")

        if scale_factor is None:
            size_x = float(bbox_meters[2] - bbox_meters[0])
            size_y = float(bbox_meters[3] - bbox_meters[1])
            if size_x > 0 and size_y > 0:
                avg_xy = (size_x + size_y) / 2.0
                if avg_xy > 0:
                    scale_factor = float(request.model_size_mm) / float(avg_xy)
                    print(
                        f"[DEBUG] {zone_prefix} Scale factor (bbox): {scale_factor:.6f} mm/m "
                        f"(zone size: {size_x:.1f} x {size_y:.1f} m)"
                    )
            elif size_x > 0:
                scale_factor = float(request.model_size_mm) / float(size_x)
                print(f"[DEBUG] {zone_prefix} Scale factor (bbox, X only): {scale_factor:.6f} mm/m")
            elif size_y > 0:
                scale_factor = float(request.model_size_mm) / float(size_y)
                print(f"[DEBUG] {zone_prefix} Scale factor (bbox, Y only): {scale_factor:.6f} mm/m")
    except Exception as exc:
        print(f"[ERROR] {zone_prefix} Failed to compute scale factor: {exc}")
        import traceback

        print(f"[ERROR] Traceback: {traceback.format_exc()}")

    if scale_factor is None or scale_factor <= 0:
        print(f"[ERROR] {zone_prefix} Failed to compute scale_factor, using fallback value.")
        scale_factor = float(request.model_size_mm) / 400.0
        print(f"[WARN] {zone_prefix} Fallback scale_factor: {scale_factor:.6f} mm/m")

    return ZoneContextResult(
        bbox_meters=bbox_meters,
        scale_factor=scale_factor,
    )
