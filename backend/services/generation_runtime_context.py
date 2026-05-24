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

    # КРИТИЧНО: для keychain (single zone) ЗАВЖДИ створюємо новий global_center
    # від bbox запиту. Інакше при перемиканні міст (Київ→Львів) використовується
    # старий center і всі координати летять на десятки км, ламається проекція.
    is_keychain = bool(getattr(request, "keychain_mode", False))
    request_bbox_center_lat = (request.north + request.south) / 2.0
    request_bbox_center_lon = (request.east + request.west) / 2.0
    existing_global_center = get_global_center()

    # Перевірка: чи поточний center "далеко" від bbox запиту (>50км)
    needs_reset = False
    if existing_global_center is not None:
        dlat = abs(existing_global_center.center_lat - request_bbox_center_lat)
        dlon = abs(existing_global_center.center_lon - request_bbox_center_lon)
        # ~50км по широті = 0.45°, по довготі ~0.7° на середніх широтах
        if dlat > 0.45 or dlon > 0.7:
            needs_reset = True
            print(
                f"[INFO] {zone_prefix} Global center is FAR from request bbox "
                f"(Δlat={dlat:.2f}°, Δlon={dlon:.2f}°) — resetting to local"
            )

    if existing_global_center is not None and not needs_reset and not is_keychain:
        global_center = existing_global_center
        print(
            f"[INFO] {zone_prefix} Using existing global center "
            f"(grid mode): lat={global_center.center_lat:.6f}, lon={global_center.center_lon:.6f}"
        )
    else:
        # Для keychain — завжди новий center від поточної зони
        from services.global_center import set_global_center
        global_center = set_global_center(request_bbox_center_lat, request_bbox_center_lon)
        print(
            f"[INFO] {zone_prefix} Created NEW global center for zone (keychain={is_keychain}): "
            f"lat={global_center.center_lat:.6f}, lon={global_center.center_lon:.6f}"
        )

    return GenerationRuntimeContext(
        latlon_bbox=latlon_bbox,
        global_center=global_center,
    )
