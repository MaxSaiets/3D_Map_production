from __future__ import annotations

from typing import Optional

import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from services.global_center import GlobalCenter


def looks_like_projected_meters(geometry: Optional[BaseGeometry], threshold: float = 100000.0) -> bool:
    if geometry is None or getattr(geometry, "is_empty", True):
        return False
    try:
        bounds = geometry.bounds
        return max(abs(float(bounds[0])), abs(float(bounds[1])), abs(float(bounds[2])), abs(float(bounds[3]))) > threshold
    except Exception:
        return False


def make_to_local_transformer(global_center: Optional[GlobalCenter]):
    if global_center is None:
        return None

    def _to_local(x, y, z=None):
        x_local, y_local = global_center.to_local(x, y)
        return (x_local, y_local) if z is None else (x_local, y_local, z)

    return _to_local


def to_local_geometry_if_needed(
    geometry: Optional[BaseGeometry],
    global_center: Optional[GlobalCenter],
    *,
    force: bool = False,
) -> Optional[BaseGeometry]:
    if geometry is None or getattr(geometry, "is_empty", True) or global_center is None:
        return geometry
    if not force and not looks_like_projected_meters(geometry):
        return geometry

    transformer = make_to_local_transformer(global_center)
    if transformer is None:
        return geometry

    try:
        return transform(transformer, geometry)
    except Exception:
        return geometry


def to_local_geodataframe_if_needed(
    gdf: Optional[gpd.GeoDataFrame],
    global_center: Optional[GlobalCenter],
    *,
    force: bool = False,
) -> Optional[gpd.GeoDataFrame]:
    if gdf is None or gdf.empty or global_center is None:
        return gdf

    try:
        sample_bounds = gdf.total_bounds if hasattr(gdf, "total_bounds") else None
        if not force and sample_bounds is not None:
            threshold = 100000.0
            needs_transform = max(abs(float(sample_bounds[0])), abs(float(sample_bounds[1])),
                                  abs(float(sample_bounds[2])), abs(float(sample_bounds[3]))) > threshold
            if not needs_transform:
                return gdf
    except Exception:
        pass

    transformer = make_to_local_transformer(global_center)
    if transformer is None:
        return gdf

    try:
        gdf_local = gdf.copy()
        gdf_local["geometry"] = gdf_local["geometry"].apply(
            lambda geom: transform(transformer, geom) if geom is not None and not geom.is_empty else geom
        )
        return gdf_local
    except Exception:
        return gdf


def clean_geometry(geometry: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    try:
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
    except Exception:
        return geometry
    return geometry


def clip_geometry(geometry: Optional[BaseGeometry], clipper: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    if geometry is None or getattr(geometry, "is_empty", True) or clipper is None or getattr(clipper, "is_empty", True):
        return geometry
    try:
        return geometry.intersection(clipper)
    except Exception:
        return geometry
