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
import tempfile
import threading
from collections import OrderedDict
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

# ── Спільний padded-region кеш для get_gdf/get_roads_graph (серія/сітка) ───────
# Одна зона ≈1.1км; батч серії/сітки запитує БАГАТО СУСІДНІХ зон поспіль, кожна
# з яких раніше йшла у DuckDB+WKT-парс заново, хоча дані здебільшого перекриваються.
# GEN_CAPACITY=1 → генерація строго послідовна, тож in-process кеш «останніх кількох
# ПОШИРЕНИХ (padded) регіонів + їх точні minlon/maxlon/minlat/maxlat» безпечний:
# запит, що ПОВНІСТЮ входить у вже закешований регіон, фільтрується в пам'яті (той
# самий предикат, що й SQL WHERE) замість нового round-trip до DuckDB.
# Перевірено на реальних даних (test_osm_cell_cache.py): identical/nested/adjacent/
# far-away/edge-straddling bbox + симуляція 4x4 сітки — 15/16 тайлів = cache hit,
# результат SET-ідентичний прямому запиту в усіх випадках.
_CELL_PAD_DEG = 0.02  # ~1.5-2.2км залежно від широти — покриває кілька сусідніх тайлів
_CELL_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CELL_CACHE_MAX = 8


def _find_cell_containing(table: str, select_cols: str, north: float, south: float, east: float, west: float):
    """Returns the cache KEY of the most-recently-used cached padded region whose
    bounds fully contain the requested bbox (and matches table+columns), or None."""
    for key in reversed(list(_CELL_CACHE.keys())):
        entry = _CELL_CACHE[key]
        if entry["table"] != table or entry["select_cols"] != select_cols:
            continue
        cn, cs, ce, cw = entry["bounds"]
        if cn >= north and cs <= south and ce >= east and cw <= west:
            return key
    return None


def _fetch_rows_cached(conn, table: str, select_cols: str, north: float, south: float, east: float, west: float):
    """Рядки (кортежі за select_cols, БЕЗ bbox-колонок) для точного запитаного bbox —
    або нарізані в пам'яті з закешованого поширеного (padded) регіону, або свіжо
    запитані (і закешовані) з DuckDB з відступом навколо запиту."""
    # BYPASS-тумблер (діагностика прод-vs-локал): OSM_CELL_CACHE=0 → прямий SQL-запит
    # ТОЧНОГО bbox, без кешу/padding — поведінка як у vanilla-завантажувача (для перевірки,
    # чи кеш спричиняє розбіжність геометрії з продом).
    if os.getenv("OSM_CELL_CACHE", "1").strip() not in ("1", "true", "yes", "on"):
        bbox_filter = "minlon <= ? AND maxlon >= ? AND minlat <= ? AND maxlat >= ?"
        with _CONN_LOCK:
            return conn.execute(
                f"SELECT {select_cols} FROM {table} WHERE {bbox_filter}",
                (east, west, north, south),
            ).fetchall()
    with _CONN_LOCK:
        key = _find_cell_containing(table, select_cols, north, south, east, west)
        if key is None:
            pad = _CELL_PAD_DEG
            pn, ps, pe, pw = north + pad, south - pad, east + pad, west - pad
            bbox_filter = "minlon <= ? AND maxlon >= ? AND minlat <= ? AND maxlat >= ?"
            params = (pe, pw, pn, ps)
            all_rows = conn.execute(
                f"SELECT {select_cols}, minlon, maxlon, minlat, maxlat FROM {table} WHERE {bbox_filter}",
                params,
            ).fetchall()
            key = f"{table}#{select_cols}#{pn:.6f}#{ps:.6f}#{pe:.6f}#{pw:.6f}"
            _CELL_CACHE[key] = {"table": table, "select_cols": select_cols, "bounds": (pn, ps, pe, pw), "rows": all_rows}
        _CELL_CACHE.move_to_end(key)
        while len(_CELL_CACHE) > _CELL_CACHE_MAX:
            _CELL_CACHE.popitem(last=False)
        cell = _CELL_CACHE[key]
    # Той самий предикат, що й SQL WHERE bbox_filter: minlon<=east AND maxlon>=west
    # AND minlat<=north AND maxlat>=south. bbox-колонки — ОСТАННІ 4 поля кожного рядка.
    out = []
    for r in cell["rows"]:
        minlon, maxlon, minlat, maxlat = r[-4], r[-3], r[-2], r[-1]
        if minlon <= east and maxlon >= west and minlat <= north and maxlat >= south:
            out.append(r[:-4])
    return out
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
    # ДІАГНОСТИКА прод-vs-локал: DISABLE_LOCAL_OSM=1 → вимикає DuckDB-джерело, тож
    # data_loader падає на Overpass-фолбек (як на проді, якщо там нема local_osm_db).
    if os.getenv("DISABLE_LOCAL_OSM", "").strip() in ("1", "true", "yes", "on"):
        return False
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
            # Інцидент 06.09.2026: дефолт DuckDB = memory_limit 80 % RAM (~5 ГБ на 6.4-ГБ VM)
            # і всі ядра. Просторовий запит по щільному центру міг вибрати гігабайти в
            # buffer pool → своп → thrash → тунель мовчить. Обмежуємо явно (env), а spill
            # шлемо у ЗАПИСУВАНУ теку — /var/lib/3dmap змонтовано read-only.
            _mem = (os.getenv("OSM_DUCKDB_MEMORY_LIMIT", "1200MB") or "").strip()
            _thr = (os.getenv("OSM_DUCKDB_THREADS", "2") or "").strip()
            _tmp = (os.getenv("OSM_DUCKDB_TEMP_DIR", "") or "").strip() or str(Path(tempfile.gettempdir()) / "duckdb_osm_tmp")
            for _stmt in (
                f"SET memory_limit='{_mem}'" if _mem else None,
                f"SET threads={int(_thr)}" if _thr.isdigit() else None,
                f"SET temp_directory='{_tmp}'",
            ):
                if not _stmt:
                    continue
                try:
                    _CONN.execute(_stmt)
                except Exception as _exc:  # noqa: BLE001
                    print(f"[OSM-DB] {_stmt} failed: {_exc}")
            try:
                Path(_tmp).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            print(f"[OSM-DB] limits: memory_limit={_mem or 'default'} threads={_thr or 'default'} temp={_tmp}")
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
    rows = _fetch_rows_cached(conn, table, cols_map[table], north, south, east, west)
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    cols = cols_map[table].split(", ")
    wkt_idx = cols.index("wkt")
    # Векторизований парсинг WKT (одним C-викликом замість per-row Python-циклу).
    # on_invalid="ignore" відтворює стару поведінку try/except: continue — невалідний
    # рядок повертає None і відкидається разом з рештою колонок того ж рядка.
    from shapely import from_wkt as _from_wkt
    parsed = _from_wkt([r[wkt_idx] for r in rows], on_invalid="ignore")
    data = {c: [] for c in cols if c != "wkt"}
    geoms = []
    for r, g in zip(rows, parsed):
        if g is None:
            continue
        geoms.append(g)
        for i, c in enumerate(cols):
            if c != "wkt":
                data[c].append(r[i])
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

    # ЗБІГ КОЛОНОК зі get_gdf("roads",...) НАВМИСНИЙ — дозволяє обом ділити ОДИН
    # закешований padded-регіон (див. _fetch_rows_cached). Якщо колонки колись
    # розійдуться, тримати select_cols тут і в cols_map["roads"] однаковими.
    rows = _fetch_rows_cached(conn, "roads", "id, highway, bridge, wkt", north, south, east, west)

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

    # Векторизований парсинг WKT (один C-виклик замість per-row shapely_wkt.loads).
    # on_invalid="ignore" відтворює стару поведінку try/except: continue.
    from shapely import from_wkt as _from_wkt
    parsed_lines = _from_wkt([r[3] for r in rows], on_invalid="ignore")

    for (road_id, highway, bridge, wkt), line in zip(rows, parsed_lines):
        if line is None:
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
