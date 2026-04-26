from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Any

import geopandas as gpd

from services.data_loader import fetch_city_data
from services.extras_loader import fetch_extras
from services.osm_source import resolve_osm_source


@dataclass
class DataFetchPipelineResult:
    gdf_buildings: gpd.GeoDataFrame
    gdf_water: gpd.GeoDataFrame
    G_roads: Any
    gdf_green: gpd.GeoDataFrame


def fetch_generation_data(
    *,
    request: Any,
    global_center: Any,
    task: Any,
    zone_prefix: str = "",
) -> DataFetchPipelineResult:
    task.update_status("processing", 10, "Завантаження даних OSM для зони...")
    print(
        f"[DEBUG] {zone_prefix} Loading data for zone: "
        f"north={request.north}, south={request.south}, east={request.east}, west={request.west}"
    )
    print(f"[DEBUG] {zone_prefix} OSM source mode: {resolve_osm_source()}")
    print(f"[DEBUG] {zone_prefix} Starting parallel data fetch for zone...")

    def get_city_data():
        road_padding = 0.01
        return fetch_city_data(
            request.north + road_padding,
            request.south - road_padding,
            request.east + road_padding,
            request.west - road_padding,
            padding=0.005,
            target_crs=global_center.utm_crs if global_center else None,
        )

    def get_extras():
        return fetch_extras(
            request.north,
            request.south,
            request.east,
            request.west,
            target_crs=global_center.utm_crs if global_center else None,
        )

    gdf_buildings = gpd.GeoDataFrame()
    gdf_water = gpd.GeoDataFrame()
    G_roads = None
    gdf_green = gpd.GeoDataFrame()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_city = executor.submit(get_city_data)
            future_extras = executor.submit(get_extras)
            gdf_buildings, gdf_water, G_roads = future_city.result()
            gdf_green = future_extras.result()
    except Exception as exc:
        print(f"[WARN] {zone_prefix} Parallel fetch failed: {exc}")
        try:
            gdf_buildings, gdf_water, G_roads = get_city_data()
        except Exception:
            pass
        try:
            gdf_green = get_extras()
        except Exception:
            pass

    num_buildings = len(gdf_buildings) if gdf_buildings is not None and not gdf_buildings.empty else 0
    num_water = len(gdf_water) if gdf_water is not None and not gdf_water.empty else 0
    num_roads = 0
    if G_roads is not None:
        if hasattr(G_roads, "edges"):
            num_roads = len(G_roads.edges)
        elif isinstance(G_roads, gpd.GeoDataFrame) and not G_roads.empty:
            num_roads = len(G_roads)

    print(
        f"[DEBUG] {zone_prefix} Loaded: {num_buildings} buildings, {num_water} water objects, {num_roads} roads"
    )
    print(
        f"[DEBUG] {zone_prefix} Data scope: roads are fetched with padded bbox and clipped later to zone polygon"
    )

    return DataFetchPipelineResult(
        gdf_buildings=gdf_buildings,
        gdf_water=gdf_water,
        G_roads=G_roads,
        gdf_green=gdf_green,
    )
