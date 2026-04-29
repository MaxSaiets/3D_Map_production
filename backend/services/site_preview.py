from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import osmnx as ox
from shapely.geometry import mapping, shape


PREVIEW_CACHE_DIR = Path(os.getenv("PREVIEW_CACHE_DIR", "cache/site_previews"))
ORDERS_DIR = Path(os.getenv("ORDERS_DIR", "output/orders"))


@dataclass(frozen=True)
class PreviewBounds:
    north: float
    south: float
    east: float
    west: float

    @property
    def area_hint(self) -> float:
        lat_m = max(0.0, (self.north - self.south) * 111_320.0)
        mid_lat = (self.north + self.south) / 2.0
        lon_m = max(0.0, (self.east - self.west) * 111_320.0 * math.cos(math.radians(mid_lat)))
        return lat_m * lon_m

    @property
    def center(self) -> dict[str, float]:
        return {
            "lat": (self.north + self.south) / 2.0,
            "lon": (self.east + self.west) / 2.0,
        }


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _cache_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=_json_default)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        if value is None:
            return fallback
        if isinstance(value, str):
            cleaned = value.lower().replace("m", "").replace(",", ".").strip()
            if ";" in cleaned:
                cleaned = cleaned.split(";", 1)[0]
            return float(cleaned)
        return float(value)
    except Exception:
        return fallback


def _feature_collection(gdf: gpd.GeoDataFrame | None, *, limit: int, simplify: float) -> dict[str, Any]:
    if gdf is None or gdf.empty:
        return {"type": "FeatureCollection", "features": []}

    gdf = gdf.copy()
    if len(gdf) > limit:
        gdf = gdf.head(limit)
    try:
        gdf["geometry"] = gdf.geometry.simplify(simplify, preserve_topology=True)
    except Exception:
        pass

    features: list[dict[str, Any]] = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if getattr(geom, "geom_type", "") not in {"LineString", "MultiLineString", "Polygon", "MultiPolygon"}:
            continue
        props: dict[str, Any] = {}
        for key in ("name", "highway", "building", "waterway", "natural", "leisure", "amenity"):
            value = row.get(key)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                props[key] = str(value)
        height = _safe_float(row.get("height"), 0.0)
        levels = _safe_float(row.get("building:levels"), 0.0)
        if height <= 0 and levels > 0:
            height = levels * 3.0
        if height > 0:
            props["height_m"] = min(height, 120.0)
        features.append({"type": "Feature", "geometry": mapping(geom), "properties": props})

    return {"type": "FeatureCollection", "features": features}


def _fetch_layer(bounds: PreviewBounds, tags: dict[str, Any]) -> gpd.GeoDataFrame:
    try:
        return ox.features_from_bbox(bounds.north, bounds.south, bounds.east, bounds.west, tags)
    except TypeError:
        # OSMnx 2.x accepts a bbox tuple.
        try:
            return ox.features_from_bbox((bounds.west, bounds.south, bounds.east, bounds.north), tags)
        except Exception as exc:
            if "No matching features" in str(exc):
                return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            raise
    except Exception as exc:
        if "No matching features" in str(exc):
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        raise


def _clip_to_shape(gdf: gpd.GeoDataFrame | None, polygon_geojson: dict[str, Any] | None) -> gpd.GeoDataFrame | None:
    if gdf is None or gdf.empty or not polygon_geojson:
        return gdf
    try:
        clip_geom = shape(polygon_geojson)
        if clip_geom.is_empty:
            return gdf
        return gpd.clip(gdf, gpd.GeoDataFrame(geometry=[clip_geom], crs="EPSG:4326"))
    except Exception:
        return gdf


def build_fast_preview(
    *,
    bounds: PreviewBounds,
    polygon_geojson: dict[str, Any] | None = None,
    include_terrain: bool = True,
) -> dict[str, Any]:
    if bounds.north <= bounds.south or bounds.east <= bounds.west:
        raise ValueError("Invalid preview bounds")
    if bounds.area_hint > 2_000_000:
        raise ValueError("Preview area is too large. Select up to about 2 sq km for fast preview.")

    cache_payload = {
        "v": 4,
        "bounds": bounds.__dict__,
        "polygon": polygon_geojson,
        "include_terrain": include_terrain,
    }
    PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = PREVIEW_CACHE_DIR / f"{_cache_key(cache_payload)}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    buildings = _clip_to_shape(_fetch_layer(bounds, {"building": True}), polygon_geojson)
    roads = _clip_to_shape(_fetch_layer(bounds, {"highway": True}), polygon_geojson)
    water = _clip_to_shape(_fetch_layer(bounds, {"natural": "water", "waterway": True}), polygon_geojson)
    parks = _clip_to_shape(
        _fetch_layer(bounds, {"leisure": ["park", "garden"], "natural": ["wood", "grassland"]}),
        polygon_geojson,
    )

    result = {
        "preview_id": cache_path.stem,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "fast_preview",
        "bounds": bounds.__dict__,
        "center": bounds.center,
        "polygon": polygon_geojson,
        "layers": {
            "buildings": _feature_collection(buildings, limit=900, simplify=0.000006),
            "roads": _feature_collection(roads, limit=1600, simplify=0.00001),
            "water": _feature_collection(water, limit=120, simplify=0.000018),
            "parks": _feature_collection(parks, limit=160, simplify=0.000018),
        },
        "terrain": {
            "enabled": bool(include_terrain),
            "mode": "browser_approximation",
        },
        "metrics": {
            "buildings": 0 if buildings is None else int(len(buildings)),
            "roads": 0 if roads is None else int(len(roads)),
            "water": 0 if water is None else int(len(water)),
            "parks": 0 if parks is None else int(len(parks)),
            "area_m2": round(bounds.area_hint, 1),
        },
    }
    cache_path.write_text(json.dumps(result, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return result


def create_order_record(payload: dict[str, Any]) -> dict[str, Any]:
    ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    order_id = f"ord_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid_like()}"
    record = {
        "id": order_id,
        "status": "new",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    (ORDERS_DIR / f"{order_id}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return record


def list_order_records() -> list[dict[str, Any]]:
    ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for path in sorted(ORDERS_DIR.glob("*.json"), reverse=True):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return records


def uuid_like() -> str:
    return hashlib.sha1(os.urandom(16)).hexdigest()[:8]
