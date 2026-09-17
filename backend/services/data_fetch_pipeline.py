from __future__ import annotations

import concurrent.futures
import os
from dataclasses import dataclass
from typing import Any

import geopandas as gpd

from services.data_loader import fetch_city_data
from services.extras_loader import fetch_extras
from services.osm_source import resolve_osm_source


# Ті самі набори тегів, що запитують шари в data_loader/extras_loader. Пакет
# перевіряє на кожному зверненні, що шар був у запиті (інакше BundleMiss і
# старий шлях) — тож розбіжність тут не зіпсує геометрію, лише зробить
# повільніше і напише [BUNDLE] … → окремий запит у лог.
_BUNDLE_CITY_TAGS = (
    {"building": True},
    {"building:part": True},
    {
        "natural": "water",
        "water": True,
        "waterway": ["riverbank", "dock", "canal"],
        "landuse": ["reservoir", "basin"],
        "man_made": ["water_well", "reservoir_covered"],
    },
    {"bridge": True},
    {"railway": ["rail", "light_rail", "narrow_gauge", "tram", "subway", "funicular"]},
)
_BUNDLE_EXTRAS_TAGS = (
    {
        "leisure": ["park", "garden", "playground", "recreation_ground", "pitch", "nature_reserve", "golf_course"],
        "landuse": ["grass", "meadow", "forest", "village_green", "cemetery", "allotments", "orchard", "recreation_ground"],
        "natural": ["wood", "grassland", "scrub", "heath"],
    },
)


def _env_float(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        return float(v) if v not in (None, "") else default
    except ValueError:
        return default


def _source_wait_callback(task):
    """Пакет чекає на джерело → задача про це знає (`source_wait_s` іде у
    /api/status, фронт показує людині чесний стан замість «зависло»)."""
    if task is None:
        return None

    def _cb(message: str, pause_s):
        try:
            task.source_wait_s = int(round(pause_s)) if pause_s else None
            if message:
                task.update_status("processing", 10, message)
        except Exception:  # noqa: BLE001
            pass

    return _cb


def _make_bundle(*, request, road_padding, loader_padding, keychain_mode, zone_prefix="", status_cb=None):
    """LazyBundle для зони або None (вимкнено / джерело не Overpass / помилка)."""
    try:
        from services import overpass_bundle
        if not overpass_bundle.enabled():
            return None
        if resolve_osm_source() in ("pbf", "geofabrik", "local"):
            return None
        n = request.north + road_padding
        s = request.south - road_padding
        e = request.east + road_padding
        w = request.west - road_padding
        city_bbox = (w - loader_padding, s - loader_padding, e + loader_padding, n + loader_padding)
        tags = list(_BUNDLE_CITY_TAGS) if not keychain_mode else [t for t in _BUNDLE_CITY_TAGS if "building:part" not in t]
        return overpass_bundle.LazyBundle(
            city_bbox=city_bbox,
            extras_bbox=(request.west, request.south, request.east, request.north),
            feature_tags=tags,
            extras_tags=_BUNDLE_EXTRAS_TAGS,
            label=zone_prefix.strip(),
            status_cb=status_cb,
        )
    except Exception as exc:  # noqa: BLE001 — пакет лише пришвидшує; без нього все працює як раніше
        print(f"[BUNDLE] не створено ({exc}) — шари окремими запитами", flush=True)
        return None


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

    keychain_mode = bool(getattr(request, "keychain_mode", False))
    if keychain_mode:
        padding_m = max(float(getattr(request, "context_padding_m", 35.0) or 35.0), 0.0)
        # Keychains are standalone crops, so we only need enough outside
        # context for roads touching the edge. The normal map workflow
        # fetches a much larger neighborhood for bridges/stitching, which
        # makes tiny keychain previews spend most time on irrelevant roads.
        road_padding = min(padding_m / 111_000.0, 0.001)
        loader_padding = 0.0
    else:
        # 16.09.2026: було 0.01 + 0.005 (≈1.65 км з кожного боку, ×25 площі зони —
        # «сусідство для мостів/зшивання»). Заміряно на Відні 80 мм через живий
        # бекенд (scripts/bench_padding.py): з 0.004 + 0.002 (≈660 м) і превʼю-GLB,
        # і друкарський 3MF байт-ідентичні (4 частини, той самий хеш), а елементів
        # з Overpass 226k → 58k, час задачі 512 → 119 с (превʼю). Ці буфери
        # стосуються лише Overpass-шляху (закордон): DuckDB для України читає
        # незбуферений bbox. Env-ручки лишаються для вимірів.
        road_padding = _env_float("OSM_ROAD_PADDING_DEG", 0.004)
        loader_padding = _env_float("OSM_LOADER_PADDING_DEG", 0.002)

    # Один Overpass-запит на всі шари цієї генерації (зони поза ukraine.duckdb).
    # Лінивий: качається лише коли якийсь шар реально йде в мережу (parquet-кеш і
    # DuckDB — локальні), і спільний для обох потоків нижче. Тому ж bbox, що
    # `fetch_city_data` рахує сама (та сама арифметика, той самий порядок дій).
    bundle = _make_bundle(
        request=request,
        road_padding=road_padding,
        loader_padding=loader_padding,
        keychain_mode=keychain_mode,
        zone_prefix=zone_prefix,
        status_cb=_source_wait_callback(task),
    )

    def get_city_data():
        return fetch_city_data(
            request.north + road_padding,
            request.south - road_padding,
            request.east + road_padding,
            request.west - road_padding,
            padding=loader_padding,
            target_crs=global_center.utm_crs if global_center else None,
            include_building_parts=not keychain_mode,
            bundle=bundle,
        )

    def get_extras():
        return fetch_extras(
            request.north,
            request.south,
            request.east,
            request.west,
            target_crs=global_center.utm_crs if global_center else None,
            bundle=bundle,
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

    if bundle is not None:
        bundle.release()

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
