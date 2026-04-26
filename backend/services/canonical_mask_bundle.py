from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def _normalize_loaded_geometry(geometry: BaseGeometry | None) -> Optional[BaseGeometry]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return None
    geom_type = str(getattr(geometry, "geom_type", "") or "")
    if "Polygon" in geom_type:
        try:
            return geometry.buffer(0)
        except Exception:
            return geometry
    return geometry


def _load_geojson_union(path: Path) -> Optional[BaseGeometry]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    geoms = []
    for feature in data.get("features") or []:
        geom = feature.get("geometry")
        if not geom:
            continue
        try:
            geoms.append(shape(geom))
        except Exception:
            continue
    if not geoms:
        return None
    try:
        merged = unary_union(geoms)
        return _normalize_loaded_geometry(merged)
    except Exception:
        try:
            return _normalize_loaded_geometry(geoms[0])
        except Exception:
            return geoms[0]


@dataclass
class CanonicalMaskBundle:
    source_dir: Path
    zone_polygon: Optional[BaseGeometry]
    roads_final: Optional[BaseGeometry]
    road_groove_mask: Optional[BaseGeometry]
    parks_final: Optional[BaseGeometry]
    parks_groove_mask: Optional[BaseGeometry]
    water_final: Optional[BaseGeometry]
    water_groove_mask: Optional[BaseGeometry]
    buildings_footprints: Optional[BaseGeometry]
    roads_semantic_preview: Optional[BaseGeometry] = None


def load_canonical_mask_bundle(bundle_dir: str | Path) -> CanonicalMaskBundle:
    source_dir = Path(bundle_dir).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Canonical mask bundle not found: {source_dir}")

    return CanonicalMaskBundle(
        source_dir=source_dir,
        zone_polygon=_load_geojson_union(source_dir / "zone_polygon.geojson"),
        roads_final=_load_geojson_union(source_dir / "roads_final.geojson"),
        road_groove_mask=_load_geojson_union(source_dir / "road_groove_mask.geojson"),
        parks_final=_load_geojson_union(source_dir / "parks_final.geojson"),
        parks_groove_mask=_load_geojson_union(source_dir / "parks_groove_mask.geojson"),
        water_final=_load_geojson_union(source_dir / "water_final.geojson"),
        water_groove_mask=_load_geojson_union(source_dir / "water_groove_mask.geojson"),
        buildings_footprints=_load_geojson_union(source_dir / "buildings_footprints.geojson"),
        roads_semantic_preview=_load_geojson_union(source_dir / "roads_semantic_preview.geojson"),
    )
