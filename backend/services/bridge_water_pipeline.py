from __future__ import annotations

from typing import Any

def prepare_bridge_water_geometries(
    *,
    request: Any,
    gdf_water: Any,
    zone_prefix: str = "",
) -> Any:
    water_geoms_for_bridges = None
    if gdf_water is not None and not gdf_water.empty:
        try:
            water_geoms_for_bridges = list(gdf_water.geometry.values)
        except Exception:
            water_geoms_for_bridges = None

    try:
        city_cache_key = getattr(request, "city_cache_key", None)
        if not city_cache_key:
            print(
                f"[DEBUG] {zone_prefix} city_cache_key not set, using only local water for bridge detection"
            )
            return water_geoms_for_bridges

        print(
            f"[DEBUG] {zone_prefix} Loading city water cache for bridge detection (key={city_cache_key})..."
        )
        try:
            from services.data_loader import load_city_cache  # legacy optional entrypoint
        except ImportError:
            print(
                f"[WARN] {zone_prefix} load_city_cache is unavailable, using only local water for bridge detection"
            )
            return water_geoms_for_bridges

        city_data = load_city_cache(city_cache_key)
        if not city_data or "water" not in city_data:
            print(
                f"[DEBUG] {zone_prefix} City cache does not contain water, using only local water"
            )
            return water_geoms_for_bridges

        gdf_water_city = city_data["water"]
        if gdf_water_city is None or gdf_water_city.empty:
            return water_geoms_for_bridges

        if water_geoms_for_bridges is None:
            water_geoms_for_bridges = list(gdf_water_city.geometry.values)
        else:
            existing_bounds = {geometry.bounds for geometry in water_geoms_for_bridges if geometry is not None}
            for geometry in gdf_water_city.geometry.values:
                if geometry is not None and geometry.bounds not in existing_bounds:
                    water_geoms_for_bridges.append(geometry)

        print(
            f"[DEBUG] {zone_prefix} Added {len(gdf_water_city)} city-cache water objects for bridge detection"
        )
    except Exception as exc:
        print(f"[WARN] {zone_prefix} Failed to load city water cache: {exc}")

    return water_geoms_for_bridges
