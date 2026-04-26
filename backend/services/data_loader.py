"""
Сервіс для завантаження даних з OpenStreetMap
Використовує osmnx для отримання структурованих даних
Підтримка кешування для швидкого повторного доступу
"""
import osmnx as ox
import geopandas as gpd
import pandas as pd
import warnings
from typing import Tuple, Optional
import os
import hashlib
import math
import time
from pathlib import Path
from osmnx._errors import InsufficientResponseError
import networkx as nx
from services.osm_source import resolve_osm_source

# Придушення deprecation warnings від pandas/geopandas
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pandas')

# Налаштування кешування
_CACHE_DIR = Path(os.getenv("OSM_DATA_CACHE_DIR") or "cache/osm/overpass_cache")
_CACHE_VERSION = "v2"  # Версія кешу (збільшити при зміні формату)
_OVERPASS_ENDPOINTS_DEFAULT = (
    "https://overpass-api.de/api",
    "https://overpass.private.coffee/api",
)


def _cache_enabled() -> bool:
    """Перевіряє, чи увімкнено кешування"""
    return (os.getenv("OSM_DATA_CACHE_ENABLED") or "1").lower() in ("1", "true", "yes")


def _overpass_endpoints() -> list[str]:
    raw = os.getenv("OSM_OVERPASS_ENDPOINTS", "").strip()
    if raw:
        endpoints = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
        if endpoints:
            return endpoints
    return [item.rstrip("/") for item in _OVERPASS_ENDPOINTS_DEFAULT]


def _run_overpass_with_retries(label: str, fetch_fn):
    original_endpoint = getattr(ox.settings, "overpass_endpoint", "https://overpass-api.de/api")
    original_timeout = int(getattr(ox.settings, "timeout", 180) or 180)
    last_error: Optional[Exception] = None
    endpoints = _overpass_endpoints()
    try:
        for attempt_index, endpoint in enumerate(endpoints, start=1):
            try:
                ox.settings.overpass_endpoint = endpoint
                ox.settings.timeout = max(original_timeout, 180)
                result = fetch_fn()
                if result is None:
                    raise InsufficientResponseError(f"{label}: empty result from {endpoint}")
                if hasattr(result, "empty") and bool(result.empty):
                    raise InsufficientResponseError(f"{label}: empty result from {endpoint}")
                if hasattr(result, "edges") and len(list(result.edges())) == 0:
                    raise InsufficientResponseError(f"{label}: empty road graph from {endpoint}")
                if attempt_index > 1:
                    print(f"[INFO] Overpass retry succeeded for {label} via {endpoint}")
                return result
            except InsufficientResponseError as exc:
                last_error = exc
                print(f"[WARN] Overpass empty response for {label} via {endpoint}: {exc}")
            except Exception as exc:
                last_error = exc
                print(f"[WARN] Overpass request failed for {label} via {endpoint}: {exc}")

            if attempt_index < len(endpoints):
                time.sleep(min(attempt_index, 2))
    finally:
        ox.settings.overpass_endpoint = original_endpoint
        ox.settings.timeout = original_timeout

    if last_error is not None:
        raise last_error
    raise InsufficientResponseError(f"{label}: all Overpass endpoints failed")


def _estimate_bbox_radius_m(
    north: float,
    south: float,
    east: float,
    west: float,
) -> float:
    center_lat = (float(north) + float(south)) * 0.5
    lat_span_m = abs(float(north) - float(south)) * 111_320.0
    lon_span_m = abs(float(east) - float(west)) * max(1.0, 111_320.0 * math.cos(math.radians(center_lat)))
    diagonal_m = math.hypot(lat_span_m, lon_span_m)
    return float(max(900.0, diagonal_m * 0.75))


def _cache_key(north: float, south: float, east: float, west: float, padding: float) -> str:
    """Створює ключ кешу на основі bbox та padding"""
    # Round to avoid cache fragmentation due to tiny float diffs
    s = f"{_CACHE_VERSION}|overpass|{round(float(north), 6)}|{round(float(south), 6)}|{round(float(east), 6)}|{round(float(west), 6)}|{round(float(padding), 6)}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _clean_gdf_for_parquet(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Очищує GeoDataFrame від колонок зі складними типами даних для збереження в Parquet"""
    df = gdf.copy()
    
    # 1. Явно видаляємо відомі проблемні колонки (але НЕ u/v!)
    problematic_cols = ['nodes', 'ways', 'relations', 'members', 'restrictions']
    cols_to_drop = [c for c in problematic_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 2. Конвертуємо всі object-колонки в рядки (КРІМ u, v, geometry)
    protected_cols = ['geometry', 'u', 'v', 'key']  # Захищені колонки для графу
    for col in df.columns:
        if col in protected_cols:
            continue
            
        if df[col].dtype == 'object':
            try:
                # Перевіряємо, чи містить колонка списки/словники
                has_complex = False
                sample = df[col].dropna().head(20)
                for val in sample:
                    if isinstance(val, (list, dict, set, tuple)):
                        has_complex = True
                        break
                
                # Конвертуємо в string для безпечного збереження
                df[col] = df[col].astype(str)
            except Exception:
                # Якщо конвертація не вдалася - видаляємо колонку (але не u/v!)
                if col not in protected_cols and col in df.columns:
                    df = df.drop(columns=[col])
                    
    return df


def _save_to_cache(north: float, south: float, east: float, west: float, padding: float,
                   buildings: gpd.GeoDataFrame, water: gpd.GeoDataFrame, roads_graph) -> None:
    """Зберігає дані в кеш"""
    if not _cache_enabled():
        return
    
    try:
        key = _cache_key(north, south, east, west, padding)
        cache_base = _CACHE_DIR / key
        cache_base.mkdir(parents=True, exist_ok=True)
        
        # Зберігаємо будівлі
        if buildings is not None and not buildings.empty:
            try:
                buildings_clean = _clean_gdf_for_parquet(buildings)
                buildings_clean.to_parquet(cache_base / "buildings.parquet", index=False)
            except Exception as e:
                 print(f"[WARN] Помилка збереження будівель в кеш: {e}")
        
        # Зберігаємо воду
        if water is not None and not water.empty:
            try:
                water_clean = _clean_gdf_for_parquet(water)
                water_clean.to_parquet(cache_base / "water.parquet", index=False)
            except Exception as e:
                print(f"[WARN] Помилка збереження води в кеш: {e}")
        
        # Зберігаємо дороги як GeoDataFrame edges
        if roads_graph is not None:
            try:
                # Перевіряємо, чи граф не порожній
                edges_list = list(roads_graph.edges()) if hasattr(roads_graph, 'edges') else []
                if len(edges_list) > 0:
                    print(f"[CACHE] Конвертація {len(edges_list)} edges в GeoDataFrame...")
                    gdf_edges = ox.graph_to_gdfs(roads_graph, nodes=False)
                    if not gdf_edges.empty:
                        print(f"[CACHE] GeoDataFrame має {len(gdf_edges.columns)} колонок.")
                        
                        gdf_edges = _clean_gdf_for_parquet(gdf_edges)
                        
                        # Перевіряємо наявність 'u' та 'v' (потрібні для відновлення графу)
                        if 'u' not in gdf_edges.columns or 'v' not in gdf_edges.columns:
                            print(f"[WARN] GeoDataFrame не містить 'u' та 'v' колонок після очищення")
                            # Спробуємо відновити з індексів, якщо можливо
                            if hasattr(gdf_edges.index, 'names') and len(gdf_edges.index.names) >= 2:
                                gdf_edges = gdf_edges.reset_index()
                                # Ще раз чистимо, бо reset_index може повернути index-колонки як object
                                gdf_edges = _clean_gdf_for_parquet(gdf_edges)
                        
                        try:
                            gdf_edges.to_parquet(cache_base / "roads_edges.parquet", index=False)
                            
                            # Зберігаємо CRS графу для подальшого відновлення
                            import json
                            graph_metadata = {}
                            if hasattr(roads_graph, 'graph') and 'crs' in roads_graph.graph:
                                graph_metadata['crs'] = str(roads_graph.graph['crs'])
                            # Також зберігаємо CRS з GeoDataFrame, якщо є
                            if hasattr(gdf_edges, 'crs') and gdf_edges.crs is not None:
                                graph_metadata['gdf_crs'] = str(gdf_edges.crs)
                            
                            if graph_metadata:
                                with open(cache_base / "roads_metadata.json", 'w') as f:
                                    json.dump(graph_metadata, f)
                            
                            print(f"[CACHE] OK: Збережено {len(roads_graph.edges())} доріг в кеш: {cache_base}")
                        except Exception as parquet_error:
                            print(f"[WARN] Помилка збереження доріг в Parquet: {parquet_error}")
                            # Спробуємо зберегти тільки основні колонки
                            try:
                                basic_cols = ['geometry', 'u', 'v'] + [c for c in gdf_edges.columns if c not in ['geometry', 'u', 'v'] and gdf_edges[c].dtype in ['int64', 'float64', 'object']]
                                basic_cols = [c for c in basic_cols if c in gdf_edges.columns]
                                gdf_basic = gdf_edges[basic_cols].copy()
                                # Конвертуємо object колонки в string, якщо можливо
                                for col in gdf_basic.columns:
                                    if gdf_basic[col].dtype == 'object' and col not in ['geometry']:
                                        try:
                                            gdf_basic[col] = gdf_basic[col].astype(str)
                                        except:
                                            gdf_basic = gdf_basic.drop(columns=[col])
                                gdf_basic.to_parquet(cache_base / "roads_edges.parquet", index=False)
                                print(f"[CACHE] OK: Збережено {len(gdf_basic)} доріг в кеш (спрощена версія): {cache_base}")
                            except Exception as e2:
                                print(f"[ERROR] Не вдалося зберегти дороги навіть у спрощеному форматі: {e2}")
                                import traceback
                                traceback.print_exc()
                    else:
                        print(f"[WARN] Граф доріг має {len(edges_list)} edges, але gdf_edges порожній після конвертації")
                else:
                    print(f"[WARN] Граф доріг порожній ({len(edges_list)} edges), не зберігаємо в кеш")
            except Exception as e:
                print(f"[WARN] Помилка збереження доріг в кеш: {e}")
                import traceback
                print(f"[DEBUG] Traceback для доріг:")
                traceback.print_exc()
        else:
            print(f"[CACHE] roads_graph is None, дороги не зберігаються в кеш")
    except Exception as e:
        print(f"[WARN] Помилка збереження в кеш (загальна): {e}")
        import traceback
        print(f"[DEBUG] Повний traceback:")
        traceback.print_exc()


def _load_from_cache(north: float, south: float, east: float, west: float, padding: float) -> Optional[Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, object]]:
    """Завантажує дані з кешу"""
    if not _cache_enabled():
        return None
    
    try:
        key = _cache_key(north, south, east, west, padding)
        cache_base = _CACHE_DIR / key
        
        bpath = cache_base / "buildings.parquet"
        wpath = cache_base / "water.parquet"
        rpath = cache_base / "roads_edges.parquet"
        
        # Перевіряємо наявність файлів (хоча б один має існувати)
        if not (bpath.exists() or wpath.exists() or rpath.exists()):
            print(f"[CACHE] Кеш не знайдено: {cache_base} (ключ: {key})")
            return None
        
        print(f"[CACHE] Кеш знайдено: {cache_base}")
        
        # Завантажуємо будівлі
        buildings = gpd.GeoDataFrame()
        if bpath.exists():
            buildings = gpd.read_parquet(bpath)
        
        # Завантажуємо воду
        water = gpd.GeoDataFrame()
        if wpath.exists():
            water = gpd.read_parquet(wpath)
        
        # Завантажуємо дороги та перетворюємо в граф
        roads_graph = None
        if rpath.exists():
            try:
                gdf_edges = gpd.read_parquet(rpath)
                if not gdf_edges.empty:
                    print(f"[CACHE] Завантажено {len(gdf_edges)} доріг з кешу")
                    # Перетворюємо GeoDataFrame edges назад в NetworkX граф
                    # ВАЖЛИВО: Зберігаємо всі атрибути, включаючи геометрію, для коректної роботи з road_processor
                    roads_graph = nx.MultiDiGraph()
                    edges_added = 0
                    for idx, row in gdf_edges.iterrows():
                        u = row.get('u')
                        v = row.get('v')
                        if u is not None and v is not None:
                            # Копіюємо всі атрибути з рядка, включаючи геометрію
                            # 'u' та 'v' використовуються тільки для додавання edges
                            attrs = {k: val for k, val in row.items() if k not in ['u', 'v']}
                            roads_graph.add_edge(u, v, **attrs)
                            edges_added += 1
                            # osmnx 2.x потребує x/y атрибутів на вузлах для graph_to_gdfs
                            geom = attrs.get('geometry')
                            if geom is not None and hasattr(geom, 'coords'):
                                coords = list(geom.coords)
                                if coords:
                                    if 'x' not in roads_graph.nodes[u]:
                                        roads_graph.nodes[u]['x'] = coords[0][0]
                                        roads_graph.nodes[u]['y'] = coords[0][1]
                                    if 'x' not in roads_graph.nodes[v]:
                                        roads_graph.nodes[v]['x'] = coords[-1][0]
                                        roads_graph.nodes[v]['y'] = coords[-1][1]
                    if edges_added == 0:
                        print(f"[WARN] Не вдалося додати жодної дороги з кешу (проблема з даними)")
                        roads_graph = None
                    else:
                        # Відновлюємо CRS з метаданих
                        import json
                        metadata_path = cache_base / "roads_metadata.json"
                        if metadata_path.exists():
                            try:
                                with open(metadata_path, 'r') as f:
                                    graph_metadata = json.load(f)
                                    # Відновлюємо CRS у графі (потрібно для osmnx)
                                    if 'crs' in graph_metadata:
                                        roads_graph.graph['crs'] = graph_metadata['crs']
                                    elif 'gdf_crs' in graph_metadata:
                                        roads_graph.graph['crs'] = graph_metadata['gdf_crs']
                            except Exception as e:
                                print(f"[WARN] Не вдалося завантажити метадані графу: {e}")
                        
                        # Якщо CRS не знайдено в метаданих, спробуємо використати CRS з GeoDataFrame
                        if 'crs' not in roads_graph.graph and hasattr(gdf_edges, 'crs') and gdf_edges.crs is not None:
                            roads_graph.graph['crs'] = str(gdf_edges.crs)
                        
                        print(f"[CACHE] Створено граф з {edges_added} edges" + (f" (CRS: {roads_graph.graph.get('crs', 'не встановлено')})" if 'crs' in roads_graph.graph else ""))
                else:
                    print(f"[CACHE] Файл доріг існує, але порожній")
            except Exception as e:
                print(f"[WARN] Помилка завантаження доріг з кешу: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"[CACHE] Дані завантажено з кешу: {cache_base}")
        return buildings, water, roads_graph
    except Exception as e:
        print(f"[WARN] Помилка завантаження з кешу: {e}")
        return None


def fetch_city_data(
    north: float,
    south: float,
    east: float,
    west: float,
    padding: float = 0.002,  # Буфер для коректної обробки країв (~200 метрів)
    target_crs: Optional[str] = None  # Цільова система координат (UTM zone)
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, object]:
    """
    Завантажує дані OSM для вказаної області з буферизацією для коректної обробки країв
    
    Args:
        north: Північна межа (широта)
        south: Південна межа (широта)
        east: Східна межа (довгота)
        west: Західна межа (довгота)
        padding: Буфер для розширення зони запиту (в градусах, ~200м за замовчуванням)
    
    Returns:
        Tuple з (buildings_gdf, water_gdf, roads_graph) - обрізані до оригінального bbox
    """
    # Зберігаємо оригінальні координати для обрізки
    target_north, target_south, target_east, target_west = north, south, east, west
    
    # Розширюємо зону запиту (буферизація)
    padded_north = north + padding
    padded_south = south - padding
    padded_east = east + padding
    padded_west = west - padding
    
    # Створюємо target_bbox в WGS84 (для обрізки до проекції)
    from shapely.geometry import box as shapely_box
    target_bbox_wgs84 = shapely_box(target_west, target_south, target_east, target_north)
    
    # Визначаємо джерело даних (потрібно для перевірки кешу та збереження)
    source = resolve_osm_source()
    
    # Перевіряємо кеш (для Overpass режиму)
    # PBF режим має власний кеш в pbf_loader
    if source not in ("pbf", "geofabrik", "local"):
        if _cache_enabled():
            print(f"[CACHE] Перевірка кешу для bbox: north={target_north:.6f}, south={target_south:.6f}, east={target_east:.6f}, west={target_west:.6f}, padding={padding}")
            cached_data = _load_from_cache(target_north, target_south, target_east, target_west, padding)
            if cached_data is not None:
                buildings_cached, water_cached, roads_cached = cached_data
                # Перевіряємо, чи дані не порожні
                if (buildings_cached is not None or water_cached is not None or roads_cached is not None):
                    # Підрахунок доріг
                    roads_count = 0
                    if roads_cached is not None:
                        if hasattr(roads_cached, 'edges'):
                            try:
                                roads_count = len(list(roads_cached.edges()))
                            except:
                                roads_count = 0
                        elif hasattr(roads_cached, '__len__'):
                            roads_count = len(roads_cached)
                    
                    print(f"[CACHE] OK: Використано кешовані дані: {len(buildings_cached) if buildings_cached is not None and not buildings_cached.empty else 0} будівель, "
                          f"{len(water_cached) if water_cached is not None and not water_cached.empty else 0} водних об'єктів, "
                          f"{roads_count} доріг")
                    # Виправлено: використовуємо перевірку is None замість or (GeoDataFrame не можна використовувати в булевих контекстах)
                    return (
                        buildings_cached if buildings_cached is not None and not buildings_cached.empty else gpd.GeoDataFrame(),
                        water_cached if water_cached is not None and not water_cached.empty else gpd.GeoDataFrame(),
                        roads_cached
                    )
                    
                    # If we have target_crs, ensure cached data matches it
                    if target_crs:
                        if not buildings_cached.empty and str(buildings_cached.crs) != str(target_crs):
                             try: buildings_cached = buildings_cached.to_crs(target_crs)
                             except: pass
                        if not water_cached.empty and str(water_cached.crs) != str(target_crs):
                             try: water_cached = water_cached.to_crs(target_crs)
                             except: pass
                        if roads_cached is not None and hasattr(roads_cached, "graph"):
                             try: 
                                 # Re-project graph if needed
                                 current_crs = roads_cached.graph.get('crs')
                                 if str(current_crs) != str(target_crs):
                                     roads_cached = ox.project_graph(roads_cached, to_crs=target_crs)
                             except: pass

                    return (
                        buildings_cached,
                        water_cached,
                        roads_cached
                    )
                else:
                    print("[CACHE] WARN: Кеш знайдено, але дані порожні, завантажую з Overpass...")
            else:
                print("[CACHE] MISS: Кеш не знайдено, завантажую з Overpass API...")
        else:
            print("[CACHE] Кешування вимкнено (OSM_DATA_CACHE_ENABLED=0), завантажую з Overpass API...")
    
    # Optional best-data mode: local Geofabrik PBF extraction by bbox
    if source in ("pbf", "geofabrik", "local"):
        print("[INFO] 📁 ДЖЕРЕЛО ДАНИХ: PBF файл (cache/osm/ukraine-latest.osm.pbf)")
        print(f"[INFO] Буферизація: розширено bbox на {padding} градусів (~{padding * 111000:.0f}м) для коректної обробки країв")
        from services.pbf_loader import fetch_city_data_from_pbf
        # Завантажуємо дані для розширеної зони
        buildings, water, roads_edges = fetch_city_data_from_pbf(padded_north, padded_south, padded_east, padded_west)
        # Optional: replace building outlines with footprints (better detail), while keeping OSM heights where possible.
        try:
            from services.footprints_loader import is_footprints_enabled, load_footprints_bbox, transfer_osm_attributes_to_footprints

            if is_footprints_enabled():
                fp = load_footprints_bbox(north, south, east, west, target_crs=getattr(buildings, "crs", None))
                if fp is not None and not fp.empty:
                    fp = transfer_osm_attributes_to_footprints(fp, buildings)
                    # Keep OSM building parts (extra detail) if present
                    if "__is_building_part" in buildings.columns:
                        parts = buildings[buildings["__is_building_part"].fillna(False)]
                        if not parts.empty:
                            buildings = gpd.GeoDataFrame(
                                pd.concat([fp, parts], ignore_index=True),
                                crs=fp.crs or parts.crs,
                            )
                        else:
                            buildings = fp
                    else:
                        buildings = fp
        except Exception as e:
            print(f"[WARN] Footprints integration skipped: {e}")

        # Обрізаємо дані до оригінального bbox
        from shapely.geometry import box as shapely_box
        target_bbox = shapely_box(target_west, target_south, target_east, target_north)
        
        if buildings is not None and not buildings.empty:
            try:
                buildings = buildings[buildings.geometry.intersects(target_bbox)]
            except Exception:
                pass
        if water is not None and not water.empty:
            try:
                water = water[water.geometry.intersects(target_bbox)]
            except Exception:
                pass
        if roads_edges is not None and not roads_edges.empty:
            try:
                roads_edges = roads_edges[roads_edges.geometry.intersects(target_bbox)]
            except Exception:
                pass
        
        return buildings, water, roads_edges

    # Використовуємо розширені координати для завантаження
    padded_bbox = (padded_west, padded_south, padded_east, padded_north)  # osmnx 2.x: (left, bottom, right, top)
    bbox = (target_west, target_south, target_east, target_north)  # Для обрізки
    
    print("[INFO] 🌐 ДЖЕРЕЛО ДАНИХ: Overpass API (онлайн)")
    print(f"[INFO] Буферизація: розширено bbox на {padding} градусів (~{padding * 111000:.0f}м) для коректної обробки країв")
    print(f"[INFO] Завантаження даних для розширеного bbox: north={padded_north}, south={padded_south}, east={padded_east}, west={padded_west}")
    
    # Налаштування osmnx: кеш ВИМКНЕНО для меншого використання пам'яті
    ox.settings.use_cache = False
    ox.settings.log_console = False
    
    # Helper functions for parallel execution
    def _fetch_buildings():
        print("Завантаження будівель...")
        tags_buildings = {'building': True}
        tags_building_parts = {'building:part': True}
        try:
            def _load_buildings_once():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    try:
                        gdf_base = ox.features_from_bbox(bbox=padded_bbox, tags=tags_buildings)
                    except TypeError:
                        gdf_base = ox.features_from_bbox(padded_bbox[0], padded_bbox[1], padded_bbox[2], padded_bbox[3], tags=tags_buildings)
                return gdf_base

            gdf_b = _run_overpass_with_retries("buildings", _load_buildings_once)
            try:
                def _load_building_parts_once():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        try:
                            return ox.features_from_bbox(bbox=padded_bbox, tags=tags_building_parts)
                        except TypeError:
                            return ox.features_from_bbox(padded_bbox[0], padded_bbox[1], padded_bbox[2], padded_bbox[3], tags=tags_building_parts)
                gdf_p = _run_overpass_with_retries("building_parts", _load_building_parts_once)
            except Exception:
                gdf_p = gpd.GeoDataFrame()
            
            # Фільтрація невалідних геометрій
            if not gdf_b.empty:
                gdf_b = gdf_b[gdf_b.geometry.notna()]
                print(f"[DEBUG] Raw/Raw filtered buildings found: {len(gdf_b)}")
            else:
                print(f"[DEBUG] No raw buildings found by OSMnx.")

            if not gdf_p.empty:
                gdf_p = gdf_p[gdf_p.geometry.notna()]
            
            # ОБРІЗКА ДО ПРОЕКЦІЇ (в WGS84 координатах)
            if not gdf_b.empty:
                try:
                    before_clip = len(gdf_b)
                    gdf_b = gdf_b[gdf_b.geometry.intersects(target_bbox_wgs84)]
                    print(f"[DEBUG] Buildings after WGS84 clip: {len(gdf_b)} (was {before_clip})")
                except Exception as e:
                    print(f"[WARN] Error clipping buildings: {e}")
                    pass
            if not gdf_p.empty:
                try:
                    gdf_p = gdf_p[gdf_p.geometry.intersects(target_bbox_wgs84)]
                except Exception:
                    pass
            
            # Проекція в метричну систему (UTM автоматично) - після обрізки
            if not gdf_b.empty:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    if target_crs:
                         gdf_b = gdf_b.to_crs(target_crs)
                    else:
                         gdf_b = ox.project_gdf(gdf_b)
            if not gdf_p.empty:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    if target_crs:
                        gdf_p = gdf_p.to_crs(target_crs)
                    else:
                        gdf_p = ox.project_gdf(gdf_p)

            # Позначаємо parts і додаємо до buildings тільки ті, що мають висотні теги
            if not gdf_p.empty:
                gdf_p = gdf_p.copy()
                gdf_p["__is_building_part"] = True
                # Якщо part не має height/levels — часто дублює "корпус" без користі → пропускаємо
                has_height = None
                for col in [
                    "height",
                    "building:height",
                    "building:levels",
                    "building:levels:aboveground",
                    "roof:height",
                    "roof:levels",
                ]:
                    if col in gdf_p.columns:
                        s = gdf_p[col].notna()
                        has_height = s if has_height is None else (has_height | s)
                if has_height is not None:
                    gdf_p = gdf_p[has_height]
                if not gdf_p.empty:
                    gdf_b = gpd.GeoDataFrame(
                        pd.concat([gdf_b, gdf_p], ignore_index=True),
                        crs=gdf_b.crs or gdf_p.crs,
                    )
            return gdf_b
        except Exception as e:
            print(f"Помилка завантаження будівель: {e}")
            return gpd.GeoDataFrame()

    def _fetch_water():
        print("Завантаження водних об'єктів...")
        tags_water = {
            'natural': 'water',
            'water': True,
            'waterway': 'riverbank',
            'landuse': 'reservoir',
        }
        try:
            def _load_water_once():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    try:
                        return ox.features_from_bbox(bbox=padded_bbox, tags=tags_water)
                    except TypeError:
                        return ox.features_from_bbox(padded_bbox[0], padded_bbox[1], padded_bbox[2], padded_bbox[3], tags=tags_water)

            gdf_w = _run_overpass_with_retries("water", _load_water_once)
            if not gdf_w.empty:
                gdf_w = gdf_w[gdf_w.geometry.notna()]
                # ОБРІЗКА ДО ПРОЕКЦІЇ (в WGS84 координатах)
                try:
                    gdf_w = gdf_w[gdf_w.geometry.intersects(target_bbox_wgs84)]
                except Exception:
                    pass
                # Проекція в метричну систему
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    try:
                        if target_crs:
                            gdf_w = gdf_w.to_crs(target_crs)
                        else:
                            gdf_w = ox.project_gdf(gdf_w)
                    except AttributeError:
                        # OSMnx v2.0+
                        if target_crs:
                            gdf_w = gdf_w.to_crs(target_crs)
                        else:
                            gdf_w = ox.projection.project_gdf(gdf_w)
                return gdf_w
        except InsufficientResponseError:
            return gpd.GeoDataFrame()
        except Exception as e:
            print(f"[WARN] Завантаження води не вдалося: {e}")
            return gpd.GeoDataFrame()
        return gpd.GeoDataFrame()

    def _fetch_roads():
        print("Завантаження дорожньої мережі...")
        try:
            def _load_roads_once():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    try:
                        return ox.graph_from_bbox(bbox=padded_bbox, network_type='all', simplify=True, retain_all=True)
                    except TypeError:
                        return ox.graph_from_bbox(padded_bbox[0], padded_bbox[1], padded_bbox[2], padded_bbox[3], network_type='all', simplify=True, retain_all=True)

            G = _run_overpass_with_retries("roads", _load_roads_once)

            if G is None or not hasattr(G, 'edges') or len(list(G.edges())) == 0:
                center_point = (
                    (float(padded_north) + float(padded_south)) * 0.5,
                    (float(padded_east) + float(padded_west)) * 0.5,
                )
                search_radius_m = _estimate_bbox_radius_m(
                    north=padded_north,
                    south=padded_south,
                    east=padded_east,
                    west=padded_west,
                )

                def _load_roads_from_point_once():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        return ox.graph_from_point(
                            center_point=center_point,
                            dist=search_radius_m,
                            network_type='all',
                            simplify=True,
                            retain_all=True,
                        )

                try:
                    G = _run_overpass_with_retries("roads_point", _load_roads_from_point_once)
                    if G is not None and hasattr(G, 'edges') and len(list(G.edges())) > 0:
                        print(
                            f"[INFO] Roads bbox-query was empty; recovered via point-query "
                            f"(dist={search_radius_m:.0f}m, edges={len(list(G.edges()))})"
                        )
                except InsufficientResponseError:
                    G = None
                except Exception as exc:
                    print(f"[WARN] Point-based road fallback failed: {exc}")
                    G = None

            if G is None or not hasattr(G, 'edges') or len(list(G.edges())) == 0:
                print("[WARN] Граф доріг порожній (G is None or no edges)")
                return None
            
            print(f"[DEBUG] Raw road graph edges: {len(list(G.edges()))}")

            # Проекція графа
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                try:
                    if target_crs:
                        print(f"[DEBUG] Projecting graph to {target_crs}...")
                        G = ox.project_graph(G, to_crs=target_crs)
                    else:
                        G = ox.project_graph(G)
                except AttributeError:
                    # OSMnx v2.0+
                    if target_crs:
                        G = ox.projection.project_graph(G, to_crs=target_crs)
                    else:
                         G = ox.projection.project_graph(G)
            
            print(f"[DEBUG] Road graph projected. Edges: {len(list(G.edges()))}")
            return G
        except InsufficientResponseError:
            print("[WARN] Дороги не знайдено (InsufficientResponseError) - Overpass returned empty result.")
            return None
        except Exception as e:
            print(f"[ERROR] Помилка завантаження доріг: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Execute in parallel
    import concurrent.futures
    gdf_buildings = gpd.GeoDataFrame()
    gdf_water = gpd.GeoDataFrame()
    G_roads = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_buildings = executor.submit(_fetch_buildings)
        future_water = executor.submit(_fetch_water)
        future_roads = executor.submit(_fetch_roads)

        gdf_buildings = future_buildings.result()
        gdf_water = future_water.result()
        G_roads = future_roads.result()

    # Optional: footprints replacement
    try:
        from services.footprints_loader import is_footprints_enabled, load_footprints_bbox, transfer_osm_attributes_to_footprints

        if is_footprints_enabled() and gdf_buildings is not None and not gdf_buildings.empty:
            fp = load_footprints_bbox(north, south, east, west, target_crs=getattr(gdf_buildings, "crs", None))
            if fp is not None and not fp.empty:
                fp = transfer_osm_attributes_to_footprints(fp, gdf_buildings)
                # keep parts if present
                if "__is_building_part" in gdf_buildings.columns:
                    parts = gdf_buildings[gdf_buildings["__is_building_part"].fillna(False)]
                    if not parts.empty:
                        gdf_buildings = gpd.GeoDataFrame(
                            pd.concat([fp, parts], ignore_index=True),
                            crs=fp.crs or parts.crs,
                        )
                    else:
                        gdf_buildings = fp
                else:
                    gdf_buildings = fp
    except Exception as e:
        print(f"[WARN] Footprints integration skipped: {e}")
    
    # Save to cache logic remains unchanged...
    # Save to cache logic
    if source not in ("pbf", "geofabrik", "local"):
        if _cache_enabled():
            print(f"[CACHE] Збереження даних в кеш...")
            # TODO: We might want to save the CRS used in cache key? 
            # Current cache implementation assumes data is valid for the bbox.
            # If we project to different zones, we might get different Coordinates for same BBox key.
            # BUT: _save_to_cache saves parquet. Parquet preserves CRS.
            # So next load will have correct CRS.
            _save_to_cache(target_north, target_south, target_east, target_west, padding, gdf_buildings, gdf_water, G_roads)
        else:
            print("[CACHE] Кешування вимкнено, дані не збережено в кеш")
    
    return gdf_buildings, gdf_water, G_roads
