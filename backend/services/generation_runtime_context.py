from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.global_center import get_global_center, get_or_create_global_center


@dataclass
class GenerationRuntimeContext:
    latlon_bbox: tuple[float, float, float, float]
    global_center: Any


def prepare_generation_runtime_context(
    *,
    request: Any,
    zone_prefix: str = "",
) -> GenerationRuntimeContext:
    try:
        from services.global_center import get_global_dem_bbox_latlon

        latlon_bbox = get_global_dem_bbox_latlon() or (
            request.north,
            request.south,
            request.east,
            request.west,
        )
    except Exception:
        latlon_bbox = (request.north, request.south, request.east, request.west)

    existing_global_center = get_global_center()
    if existing_global_center is not None:
        global_center = existing_global_center
        print(
            f"[INFO] {zone_prefix} Using existing global center "
            f"(grid mode): lat={global_center.center_lat:.6f}, lon={global_center.center_lon:.6f}"
        )
    else:
        global_center = get_or_create_global_center(bbox_latlon=latlon_bbox)
        print(
            f"[INFO] {zone_prefix} Created new global center for zone: "
            f"lat={global_center.center_lat:.6f}, lon={global_center.center_lon:.6f}"
        )

    return GenerationRuntimeContext(
        latlon_bbox=latlon_bbox,
        global_center=global_center,
    )
