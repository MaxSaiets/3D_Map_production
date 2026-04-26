from __future__ import annotations

from typing import Any, Optional


def resolve_generation_source_crs(
    *,
    gdf_buildings: Any,
    G_roads: Any,
    global_center: Any,
    allow_global_center_fallback: bool = True,
    zone_prefix: str = "",
) -> Any:
    source_crs = None
    try:
        if gdf_buildings is not None and not gdf_buildings.empty:
            return gdf_buildings.crs
        if G_roads is not None and hasattr(G_roads, "crs"):
            source_crs = getattr(G_roads, "crs", None)
            if source_crs is not None:
                return source_crs
    except Exception:
        pass

    if allow_global_center_fallback and global_center is not None:
        try:
            source_crs = global_center.get_utm_crs()
            print(f"[INFO] {zone_prefix} Using UTM CRS from global_center for terrain generation fallback")
            return source_crs
        except Exception:
            pass

    return None


def compute_water_depth_m(
    *,
    water_depth_mm: float,
    scale_factor: Optional[float],
) -> Optional[float]:
    if water_depth_mm is None:
        return None
    if scale_factor and scale_factor > 0:
        return float(water_depth_mm) / float(scale_factor)
    return float(water_depth_mm) / 1000.0


def compute_water_surface_thickness_m(
    *,
    water_depth_mm: float,
    water_depth_m: Optional[float],
    scale_factor: Optional[float],
    min_thickness_mm: float = 1.5,
    max_cap_mm: float = 3.0,
    ratio: float = 0.4,
) -> Optional[float]:
    if water_depth_mm is None:
        return None

    max_thickness_mm = min(float(water_depth_mm) * 0.5, max_cap_mm)
    surface_mm = float(max(min_thickness_mm, min(max_thickness_mm, float(water_depth_mm) * ratio)))
    if scale_factor and scale_factor > 0:
        return float(surface_mm) / float(scale_factor)
    if water_depth_m is not None:
        return float(water_depth_m) * ratio
    return float(surface_mm) / 1000.0
