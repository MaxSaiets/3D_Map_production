"""
local_osm_db.py — швидкий локальний OSM extract з DuckDB-БД.

Замінює Overpass API для bbox-запитів. Швидкість: 50-200ms (vs 2-10s Overpass).

Використання:
    from services.local_osm_db import extract_bbox, is_available
    if is_available():
        data = extract_bbox(north, south, east, west)
        # data = {"buildings": [...], "roads": [...], "water": [...], "parks": [...], "bridges": [...]}
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None  # type: ignore

# Кеш на 100 bbox у пам'яті — повторні запити в ту саму зону за 0.1ms
_QUERY_CACHE: dict[str, dict] = {}
_QUERY_CACHE_MAX = 100
_CONN_LOCK = threading.Lock()
_CONN = None
# Кеш колонок таблиці buildings (схема НЕ змінюється в межах процесу). Раніше КОЖЕН
# запит будівель робив `PRAGMA table_info('buildings')` ПОЗА _CONN_LOCK → на спільному
# read-only конекті це гонилось з ін. потоком і час від часу псувало результат запиту
# (повертало 0 будівель при наявних даних). Рахуємо ОДИН раз під локом.
_BUILDING_COLS: Optional[str] = None


def db_path() -> Optional[Path]:
    """Шукає duckdb файл по env або стандартних шляхах."""
    env = os.getenv("OSM_DUCKDB_PATH")
    candidates = []
    if env:
        candidates.append(Path(env))
    # Стандартні шляхи (Windows F: drive + Linux server /opt/3dmap/data)
    candidates.extend([
        Path("/f/3dmap_data/osm/ukraine.duckdb"),
        Path("F:/3dmap_data/osm/ukraine.duckdb"),
        Path("/opt/3dmap/data/ukraine.duckdb"),
        Path(__file__).resolve().parents[1] / "data" / "ukraine.duckdb",
    ])
    for c in candidates:
        try:
            if c.exists() and c.stat().st_size > 1024:
                return c
        except Exception:
            continue
    return None


def is_available() -> bool:
    """True якщо duckdb встановлено І файл БД існує."""
    if duckdb is None:
        return False
    return db_path() is not None


def _get_conn():
    """Lazy-initialized DuckDB connection (thread-safe)."""
    global _CONN
    with _CONN_LOCK:
        if _CONN is None:
            path = db_path()
            if path is None:
                return None
            _CONN = duckdb.connect(str(path), read_only=True)
        return _CONN


def _bbox_key(north: float, south: float, east: float, west: float) -> str:
    return f"{south:.5f},{west:.5f},{north:.5f},{east:.5f}"


def get_gdf(
    table: str,
    north: float, south: float, east: float, west: float,
    target_crs=None,
):
    """Повертає GeoDataFrame з геометріями для заданого bbox.
    Використовується у data_loader.py для backend генерації (заміна Overpass).
    """
    if not is_available():
        return None
    try:
        import geopandas as gpd  # type: ignore
        from shapely import wkt as shapely_wkt  # type: ignore
    except Exception:
        return None
    conn = _get_conn()
    if conn is None:
        return None
    # Динамічно додаємо height/landmark до buildings, ЯКЩО вони є у схемі (DB
    # зібрано НОВИМ build_osm_db). СТАРИЙ DB (лише id,levels,wkt) працює без змін —
    # defensive: PRAGMA-перевірка наявності колонок, інакше fallback на levels.
    # Колонки buildings рахуємо ОДИН раз і кешуємо (під локом) — без per-query PRAGMA,
    # який раніше гонився на спільному конекті й давав «0 будівель».
    global _BUILDING_COLS
    if _BUILDING_COLS is None:
        _bcols = "id, levels, wkt"
        try:
            with _CONN_LOCK:
                _avail = {r[1] for r in conn.execute("PRAGMA table_info('buildings')").fetchall()}
            _extra = [c for c in ("height", "landmark") if c in _avail]
            if _extra:
                _bcols = "id, levels, " + ", ".join(_extra) + ", wkt"
        except Exception:
            pass
        _BUILDING_COLS = _bcols
    cols_map = {
        "buildings": _BUILDING_COLS,
        "roads":     "id, highway, bridge, wkt",
        "bridges":   "id, highway, wkt",
        "water":     "id, type, wkt",
        "parks":     "id, type, wkt",
    }
    if table not in cols_map:
        return None
    bbox_filter = "minlon <= ? AND maxlon >= ? AND minlat <= ? AND maxlat >= ?"
    params = (east, west, north, south)
    with _CONN_LOCK:
        rows = conn.execute(
            f"SELECT {cols_map[table]} FROM {table} WHERE {bbox_filter}",
            params,
        ).fetchall()
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    cols = cols_map[table].split(", ")
    wkt_idx = cols.index("wkt")
    data = {c: [] for c in cols if c != "wkt"}
    geoms = []
    for r in rows:
        try:
            geoms.append(shapely_wkt.loads(r[wkt_idx]))
            for i, c in enumerate(cols):
                if c != "wkt":
                    data[c].append(r[i])
        except Exception:
            continue
    gdf = gpd.GeoDataFrame(data, geometry=geoms, crs="EPSG:4326")
    if target_crs is not None:
        try:
            gdf = gdf.to_crs(target_crs)
        except Exception:
            pass
    return gdf


def get_roads_graph(north: float, south: float, east: float, west: float, target_crs=None):
    """Будує networkx MultiDiGraph з локальних roads, сумісний з OSMnx.

    Заміна `ox.graph_from_bbox` через локальну БД (12ms vs 2-5s Overpass).

    Returns:
        networkx.MultiDiGraph або None якщо БД недоступна.
    """
    if not is_available():
        return None
    try:
        import networkx as nx
        import osmnx as ox  # type: ignore
        import geopandas as gpd  # type: ignore
        from shapely import wkt as shapely_wkt  # type: ignore
        from shapely.geometry import Point
    except Exception:
        return None

    conn = _get_conn()
    if conn is None:
        return None

    bbox_filter = "minlon <= ? AND maxlon >= ? AND minlat <= ? AND maxlat >= ?"
    params = (east, west, north, south)
    with _CONN_LOCK:
        rows = conn.execute(
            f"SELECT id, highway, bridge, wkt FROM roads WHERE {bbox_filter}",
            params,
        ).fetchall()

    if not rows:
        return None

    # Будуємо edges_gdf і nodes_gdf для ox.graph_from_gdfs.
    # Node id = (lon_int, lat_int) (тo7-digit precision = ~1cm) — стабільний хеш.
    def coord_to_id(lon: float, lat: float) -> int:
        # Encode as int: 9 цифр на координату
        return int(round((lat + 90) * 1e7)) * 10_000_000_000 + int(round((lon + 180) * 1e7))

    nodes_data = {}  # node_id -> (lon, lat)
    edges_data = []  # list of (u, v, key, attrs)
    edge_key_counter = {}  # (u,v) -> next key

    for road_id, highway, bridge, wkt in rows:
        try:
            line = shapely_wkt.loads(wkt)
        except Exception:
            continue
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        start_lon, start_lat = coords[0]
        end_lon, end_lat = coords[-1]
        u = coord_to_id(start_lon, start_lat)
        v = coord_to_id(end_lon, end_lat)
        if u == v:
            continue
        nodes_data[u] = (start_lon, start_lat)
        nodes_data[v] = (end_lon, end_lat)
        key = edge_key_counter.get((u, v), 0)
        edge_key_counter[(u, v)] = key + 1
        # Орієнтовна довжина в метрах (rough Haversine для коротких)
        import math
        dx = (end_lon - start_lon) * 111_320 * math.cos(math.radians((start_lat + end_lat) / 2))
        dy = (end_lat - start_lat) * 111_320
        length_m = (dx * dx + dy * dy) ** 0.5
        edges_data.append({
            "u": u, "v": v, "key": key,
            "osmid": road_id,
            "highway": highway,
            "bridge": bridge if bridge != "no" else None,
            "length": length_m,
            "geometry": line,
        })

    if not nodes_data or not edges_data:
        return None

    # Створюємо GeoDataFrames у форматі OSMnx
    nodes_records = []
    for nid, (lon, lat) in nodes_data.items():
        nodes_records.append({
            "osmid": nid, "x": lon, "y": lat, "geometry": Point(lon, lat),
        })
    nodes_gdf = gpd.GeoDataFrame(nodes_records, crs="EPSG:4326").set_index("osmid")

    edges_df = gpd.GeoDataFrame(edges_data, crs="EPSG:4326").set_index(["u", "v", "key"])

    try:
        G = ox.graph_from_gdfs(nodes_gdf, edges_df)
        # Проекція у target_crs якщо потрібно
        if target_crs is not None:
            try:
                G = ox.project_graph(G, to_crs=target_crs)
            except Exception:
                pass
        return G
    except Exception as exc:
        print(f"[LOCAL OSM DB] graph_from_gdfs failed: {exc}")
        return None


def extract_bbox(
    north: float, south: float, east: float, west: float
) -> dict:
    """
    Витягує всі OSM-фічі що ПЕРЕТИНАЮТЬ bbox. Повертає dict зі списками:
      {
        "buildings": [{"id", "levels", "wkt"}, ...],
        "roads":     [{"id", "highway", "bridge", "wkt"}, ...],
        "bridges":   [{"id", "highway", "wkt"}, ...],
        "water":     [{"id", "type", "wkt"}, ...],
        "parks":     [{"id", "type", "wkt"}, ...],
      }
    """
    key = _bbox_key(north, south, east, west)
    if key in _QUERY_CACHE:
        return _QUERY_CACHE[key]

    conn = _get_conn()
    if conn is None:
        return {"buildings": [], "roads": [], "bridges": [], "water": [], "parks": []}

    result = {}
    bbox_filter = "minlon <= ? AND maxlon >= ? AND minlat <= ? AND maxlat >= ?"
    params = (east, west, north, south)

    with _CONN_LOCK:
        # Buildings
        rows = conn.execute(
            f"SELECT id, levels, wkt FROM buildings WHERE {bbox_filter}",
            params,
        ).fetchall()
        result["buildings"] = [{"id": r[0], "levels": r[1], "wkt": r[2]} for r in rows]

        # Roads
        rows = conn.execute(
            f"SELECT id, highway, bridge, wkt FROM roads WHERE {bbox_filter}",
            params,
        ).fetchall()
        result["roads"] = [
            {"id": r[0], "highway": r[1], "bridge": r[2], "wkt": r[3]} for r in rows
        ]

        # Bridges
        rows = conn.execute(
            f"SELECT id, highway, wkt FROM bridges WHERE {bbox_filter}",
            params,
        ).fetchall()
        result["bridges"] = [{"id": r[0], "highway": r[1], "wkt": r[2]} for r in rows]

        # Water
        rows = conn.execute(
            f"SELECT id, type, wkt FROM water WHERE {bbox_filter}",
            params,
        ).fetchall()
        result["water"] = [{"id": r[0], "type": r[1], "wkt": r[2]} for r in rows]

        # Parks
        rows = conn.execute(
            f"SELECT id, type, wkt FROM parks WHERE {bbox_filter}",
            params,
        ).fetchall()
        result["parks"] = [{"id": r[0], "type": r[1], "wkt": r[2]} for r in rows]

    # LRU cache
    if len(_QUERY_CACHE) >= _QUERY_CACHE_MAX:
        _QUERY_CACHE.pop(next(iter(_QUERY_CACHE)))
    _QUERY_CACHE[key] = result
    return result
