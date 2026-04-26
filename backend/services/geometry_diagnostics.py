from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import trimesh
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry


def ensure_valid_geometry(geometry: Any) -> Optional[BaseGeometry]:
    if geometry is None:
        return None
    if not hasattr(geometry, "is_empty"):
        return None
    if geometry.is_empty:
        return None
    geom_type = str(getattr(geometry, "geom_type", "") or "")
    if "Polygon" in geom_type:
        try:
            geometry = geometry.buffer(0)
        except Exception:
            pass
    if geometry is None or getattr(geometry, "is_empty", True):
        return None
    return geometry


def iter_polygons(geometry: Any) -> list[Polygon]:
    geometry = ensure_valid_geometry(geometry)
    if geometry is None:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [poly for poly in geometry.geoms if poly is not None and not poly.is_empty]
    if isinstance(geometry, GeometryCollection):
        polygons: list[Polygon] = []
        for item in geometry.geoms:
            polygons.extend(iter_polygons(item))
        return polygons
    if hasattr(geometry, "geoms"):
        polygons = []
        for item in geometry.geoms:
            polygons.extend(iter_polygons(item))
        return polygons
    return []


def geometry_stats(geometry: Any) -> dict[str, Any]:
    geometry = ensure_valid_geometry(geometry)
    if geometry is None:
        return {
            "present": False,
            "geometry_type": None,
            "polygon_count": 0,
            "hole_count": 0,
            "area": 0.0,
            "total_hole_area": 0.0,
            "bounds": None,
        }

    polygons = iter_polygons(geometry)
    hole_count = 0
    total_hole_area = 0.0
    for polygon in polygons:
        for ring in polygon.interiors:
            try:
                hole_count += 1
                total_hole_area += float(Polygon(ring.coords).area)
            except Exception:
                continue

    bounds = None
    try:
        bounds = [float(v) for v in geometry.bounds]
    except Exception:
        bounds = None

    return {
        "present": True,
        "geometry_type": geometry.geom_type,
        "polygon_count": len(polygons),
        "hole_count": hole_count,
        "area": float(getattr(geometry, "area", 0.0)),
        "total_hole_area": total_hole_area,
        "bounds": bounds,
    }


def overlap_report(lhs: Any, rhs: Any, *, lhs_name: str, rhs_name: str) -> dict[str, Any]:
    lhs = ensure_valid_geometry(lhs)
    rhs = ensure_valid_geometry(rhs)
    if lhs is None or rhs is None:
        return {
            "lhs": lhs_name,
            "rhs": rhs_name,
            "has_overlap": False,
            "overlap_area": 0.0,
            "overlap_bounds": None,
        }
    try:
        overlap = lhs.intersection(rhs)
    except Exception:
        overlap = None
    overlap = ensure_valid_geometry(overlap)
    if overlap is None:
        return {
            "lhs": lhs_name,
            "rhs": rhs_name,
            "has_overlap": False,
            "overlap_area": 0.0,
            "overlap_bounds": None,
        }
    bounds = None
    try:
        bounds = [float(v) for v in overlap.bounds]
    except Exception:
        bounds = None
    return {
        "lhs": lhs_name,
        "rhs": rhs_name,
        "has_overlap": True,
        "overlap_area": float(getattr(overlap, "area", 0.0)),
        "overlap_bounds": bounds,
    }


def concatenate_meshes(meshes: Any) -> Optional[trimesh.Trimesh]:
    if meshes is None:
        return None
    if isinstance(meshes, trimesh.Trimesh):
        return meshes if len(meshes.vertices) > 0 else None
    try:
        items = [mesh for mesh in meshes if mesh is not None and len(mesh.vertices) > 0]
    except Exception:
        return None
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    try:
        return trimesh.util.concatenate(items)
    except Exception:
        return items[0]


def mesh_stats(mesh: Any, *, label: str) -> dict[str, Any]:
    mesh = concatenate_meshes(mesh)
    if mesh is None:
        return {
            "label": label,
            "present": False,
            "vertices": 0,
            "faces": 0,
            "component_count": 0,
            "watertight": False,
            "bounds": None,
            "extents": None,
        }

    try:
        components = list(mesh.split(only_watertight=False))
    except Exception:
        components = [mesh]

    bounds = None
    extents = None
    try:
        bounds_arr = np.asarray(mesh.bounds, dtype=float)
        bounds = bounds_arr.tolist()
        extents = (bounds_arr[1] - bounds_arr[0]).tolist()
    except Exception:
        bounds = None
        extents = None

    return {
        "label": label,
        "present": True,
        "path_hint": str(Path(label)),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "component_count": int(len(components)),
        "watertight": bool(mesh.is_watertight),
        "bounds": bounds,
        "extents": extents,
    }
