"""
Extra layers loader:
- parks/green areas (polygons)
- POIs (benches etc.)

Works in two modes:
- OSM_SOURCE=pbf -> read from local Geofabrik PBF via pyrosm
- otherwise -> fetch from Overpass via OSMnx (best-effort)
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Tuple
from pathlib import Path
import hashlib
import pandas as pd

import geopandas as gpd
import osmnx as ox
from services.osm_source import resolve_osm_source


_CACHE_DIR = Path(os.getenv("OSM_DATA_CACHE_DIR") or "cache/osm/overpass_cache/extras")
_CACHE_VERSION = "v1"
_OVERPASS_ENDPOINTS_DEFAULT = (
    "https://overpass-api.de/api",
    "https://overpass.private.coffee/api",
)


def _bbox_key(north: float, south: float, east: float, west: float) -> tuple[float, float, float, float]:
    return (round(float(north), 6), round(float(south), 6), round(float(east), 6), round(float(west), 6))


def _cache_enabled() -> bool:
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
    last_error: Exception | None = None
    endpoints = _overpass_endpoints()
    try:
        for attempt_index, endpoint in enumerate(endpoints, start=1):
            try:
                ox.settings.overpass_endpoint = endpoint
                ox.settings.timeout = max(original_timeout, 180)
                result = fetch_fn()
                if result is None:
                    raise RuntimeError(f"{label}: empty result from {endpoint}")
                if hasattr(result, "empty") and bool(result.empty):
                    raise RuntimeError(f"{label}: empty result from {endpoint}")
                if attempt_index > 1:
                    print(f"[INFO] Overpass retry succeeded for {label} via {endpoint}")
                return result
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
    raise RuntimeError(f"{label}: all Overpass endpoints failed")


def _clean_gdf_for_parquet(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = gdf.copy()
    problematic_cols = ['nodes', 'ways', 'relations', 'members', 'restrictions']
    cols_to_drop = [c for c in problematic_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    protected_cols = ['geometry']
    for col in list(df.columns):
        if col in protected_cols:
            continue
        if df[col].dtype == 'object':
            try:
                sample = df[col].dropna().head(20)
                has_complex = any(isinstance(val, (list, dict, set, tuple)) for val in sample)
                if has_complex:
                    df[col] = df[col].astype(str)
            except Exception:
                if col in df.columns:
                    df = df.drop(columns=[col])
    return df


def _cache_key(
    source: str,
    north: float,
    south: float,
    east: float,
    west: float,
    target_crs: str | None,
) -> str:
    payload = "|".join(
        [
            _CACHE_VERSION,
            source,
            f"{round(float(north), 6)}",
            f"{round(float(south), 6)}",
            f"{round(float(east), 6)}",
            f"{round(float(west), 6)}",
            str(target_crs or ""),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def _load_from_cache(
    *,
    source: str,
    north: float,
    south: float,
    east: float,
    west: float,
    target_crs: str | None,
) -> gpd.GeoDataFrame | None:
    if not _cache_enabled():
        return None
    try:
        key = _cache_key(source, north, south, east, west, target_crs)
        path = _CACHE_DIR / f"{key}.parquet"
        if not path.exists():
            return None
        gdf = gpd.read_parquet(path)
        if gdf is None or getattr(gdf, "empty", False):
            print(f"[extras_loader] empty cache ignored: {path}")
            return None
        print(f"[extras_loader] cache hit: {path}")
        return gdf
    except Exception as exc:
        print(f"[WARN] extras cache read failed: {exc}")
        return None


def _save_to_cache(
    *,
    source: str,
    north: float,
    south: float,
    east: float,
    west: float,
    target_crs: str | None,
    gdf_green: gpd.GeoDataFrame,
) -> None:
    if not _cache_enabled() or gdf_green is None:
        return
    if getattr(gdf_green, "empty", False):
        # Do not persist empty green results. Overpass can return an empty GDF
        # for transient endpoint/tag failures, and caching that turns parks into
        # stable terrain voids on later runs.
        return
    try:
        key = _cache_key(source, north, south, east, west, target_crs)
        path = _CACHE_DIR / f"{key}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        cleaned = _clean_gdf_for_parquet(gdf_green)
        cleaned.to_parquet(path, index=False)
    except Exception as exc:
        print(f"[WARN] extras cache write failed: {exc}")


def fetch_extras(
    north: float,
    south: float,
    east: float,
    west: float,
    target_crs: str | None = None
) -> gpd.GeoDataFrame:
    # Перевіряємо чи є preloaded дані (пріоритет)
    try:
        from services.preloaded_data import is_loaded, get_extras_for_bbox
        if is_loaded():
            print("[extras_loader] Використовую preloaded дані")
            green, _ = get_extras_for_bbox(north, south, east, west)
            # Preloaded data might be in generic UTM. Re-project if target_crs is set.
            if target_crs and not green.empty:
                try: green = green.to_crs(target_crs)
                except: pass
            return green
    except Exception as e:
        print(f"[WARN] Помилка використання preloaded даних для extras: {e}, використовуємо звичайний режим")
    
    source = resolve_osm_source()

    cached = _load_from_cache(
        source=source,
        north=north,
        south=south,
        east=east,
        west=west,
        target_crs=target_crs,
    )
    if cached is not None:
        return cached

    if source in ("pbf", "geofabrik", "local"):
        from services.pbf_loader import fetch_extras_from_pbf

        green, _ = fetch_extras_from_pbf(north, south, east, west)
        if target_crs and not green.empty:
             try: green = green.to_crs(target_crs)
             except: pass
        _save_to_cache(
            source=source,
            north=north,
            south=south,
            east=east,
            west=west,
            target_crs=target_crs,
            gdf_green=green,
        )
        return green

    bbox = (west, south, east, north)  # osmnx 2.x: (left, bottom, right, top)

    # Вимкнення кешу OSMnx для меншого використання пам'яті
    ox.settings.use_cache = False
    ox.settings.log_console = False

    # Parks/green polygons
    tags_green = {
        "leisure": ["park", "garden", "playground", "recreation_ground", "pitch"],
        "landuse": ["grass", "meadow", "forest", "village_green"],
        "natural": ["wood"],
    }
    # POIs - REMOVED per user request
    # tags_pois = { ... }

    gdf_green = gpd.GeoDataFrame()

    try:
        def _load_green_once():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                try:
                    return ox.features_from_bbox(bbox=bbox, tags=tags_green)
                except TypeError:
                    return ox.features_from_bbox(bbox[3], bbox[1], bbox[2], bbox[0], tags=tags_green)

        gdf_green = _run_overpass_with_retries("green", _load_green_once)
        if not gdf_green.empty:
            gdf_green = gdf_green[gdf_green.geometry.notna()]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                try:
                    if target_crs:
                         gdf_green = gdf_green.to_crs(target_crs)
                    else:
                         gdf_green = ox.project_gdf(gdf_green)
                except AttributeError:
                    if target_crs:
                         gdf_green = gdf_green.to_crs(target_crs)
                    else:
                         gdf_green = ox.projection.project_gdf(gdf_green)
            # Keep polygons only
            gdf_green = gdf_green[gdf_green.geom_type.isin(["Polygon", "MultiPolygon"])]
    except Exception:
        gdf_green = gpd.GeoDataFrame()

    _save_to_cache(
        source=source,
        north=north,
        south=south,
        east=east,
        west=west,
        target_crs=target_crs,
        gdf_green=gdf_green,
    )
    return gdf_green
