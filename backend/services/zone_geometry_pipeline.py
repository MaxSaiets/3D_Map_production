from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from services.crs_utils import bbox_latlon_to_utm
from services.hexagonal_grid import hexagon_center_to_corner


@dataclass
class ZoneGeometryResult:
    zone_polygon_local: Optional[BaseGeometry]
    reference_xy_m: Optional[tuple[float, float]]


def prepare_zone_geometry(
    *,
    global_center: Any,
    grid_bbox_latlon: Any,
    zone_row: Any,
    zone_col: Any,
    hex_size_m: Any,
    zone_polygon_coords: Optional[list],
    zone_prefix: str = "",
) -> ZoneGeometryResult:
    zone_polygon_local = None
    reference_xy_m = None

    if zone_polygon_coords is not None and global_center is not None:
        try:
            local_coords = []
            for coord in zone_polygon_coords:
                if len(coord) < 2:
                    continue
                lon, lat = float(coord[0]), float(coord[1])
                if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                    print(f"[WARN] {zone_prefix} Invalid zone coordinate: lon={lon}, lat={lat}")
                    continue
                x_utm, y_utm = global_center.to_utm(lon, lat)
                x_local, y_local = global_center.to_local(x_utm, y_utm)
                local_coords.append((x_local, y_local))

            if len(local_coords) >= 3:
                zone_polygon_local = Polygon(local_coords)
                if not zone_polygon_local.is_valid:
                    print(f"[WARN] {zone_prefix} Zone polygon invalid, applying buffer(0)...")
                    zone_polygon_local = zone_polygon_local.buffer(0)
                if zone_polygon_local is not None and not zone_polygon_local.is_empty:
                    bounds = zone_polygon_local.bounds
                    reference_xy_m = (float(bounds[2] - bounds[0]), float(bounds[3] - bounds[1]))
                    print(
                        f"[DEBUG] {zone_prefix} Zone polygon converted to local coords "
                        f"({len(local_coords)} points), reference_xy_m={reference_xy_m[0]:.2f}x{reference_xy_m[1]:.2f}m"
                    )
                    return ZoneGeometryResult(
                        zone_polygon_local=zone_polygon_local,
                        reference_xy_m=reference_xy_m,
                    )
                else:
                    print(f"[WARN] {zone_prefix} Zone polygon became empty after buffer(0)")
            else:
                print(f"[WARN] {zone_prefix} Not enough coordinates for polygon: {len(local_coords)}")
        except Exception as exc:
            print(f"[WARN] {zone_prefix} Failed to build zone polygon from coordinates: {exc}")
            import traceback

            print(f"[WARN] Traceback: {traceback.format_exc()}")

    if (
        global_center is not None
        and grid_bbox_latlon is not None
        and zone_row is not None
        and zone_col is not None
        and hex_size_m is not None
    ):
        try:
            import math

            north, south, east, west = grid_bbox_latlon
            minx_utm_grid, miny_utm_grid, _, _, _, _, _ = bbox_latlon_to_utm(
                float(north), float(south), float(east), float(west)
            )

            hs = float(hex_size_m)
            hex_width = math.sqrt(3.0) * hs
            hex_height = 1.5 * hs
            row = int(zone_row)
            col = int(zone_col)

            center_x = float(minx_utm_grid + col * hex_width + (hex_width / 2.0 if (row % 2) == 1 else 0.0))
            center_y = float(miny_utm_grid + row * hex_height)

            corners_utm = hexagon_center_to_corner(center_x, center_y, hs)
            local_coords = []
            for x_utm, y_utm in corners_utm:
                x_local, y_local = global_center.to_local(float(x_utm), float(y_utm))
                local_coords.append((float(x_local), float(y_local)))

            zone_polygon_local = Polygon(local_coords)
            if not zone_polygon_local.is_valid:
                zone_polygon_local = zone_polygon_local.buffer(0)

            if zone_polygon_local is not None and not zone_polygon_local.is_empty:
                reference_xy_m = (float(hex_width), float(2.0 * hs))
                print(
                    f"[DEBUG] {zone_prefix} Reconstructed hex zone polygon from row/col ({row},{col}) in local coords; "
                    f"reference_xy_m={reference_xy_m[0]:.2f}x{reference_xy_m[1]:.2f}m"
                )
                return ZoneGeometryResult(
                    zone_polygon_local=zone_polygon_local,
                    reference_xy_m=reference_xy_m,
                )
        except Exception as exc:
            print(f"[WARN] {zone_prefix} Failed to reconstruct hex polygon from row/col: {exc}")

    return ZoneGeometryResult(
        zone_polygon_local=zone_polygon_local,
        reference_xy_m=reference_xy_m,
    )
