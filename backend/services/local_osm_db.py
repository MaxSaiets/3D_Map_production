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
