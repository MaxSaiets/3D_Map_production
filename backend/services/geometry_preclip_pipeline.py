from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import geopandas as gpd
from shapely.geometry import MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry

from services.geometry_context import make_to_local_transformer


@dataclass
class GeometryPreclipResult:
    gdf_buildings_local: Optional[gpd.GeoDataFrame]
    building_geometries_for_flatten: Any
    gdf_water_local: Optional[gpd.GeoDataFrame]
    water_geometries_local_for_bridges: Any
    preclipped_to_zone: bool


def _keep_polygons(geometry: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    if geometry is None or geometry.is_empty:
        return None

    geom_type = getattr(geometry, "geom_type", None)
    if geom_type in ("Polygon", "MultiPolygon"):
        return geometry
    if geom_type == "GeometryCollection":
        polygons = [part for part in geometry.geoms if getattr(part, "geom_type", None) == "Polygon"]
        if not polygons:
            return None
        return MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
    return None


def _clip_geometry_to_zone(
    geometry: Optional[BaseGeometry],
    zone_polygon_local: Optional[BaseGeometry],
) -> Optional[BaseGeometry]:
    if geometry is None or geometry.is_empty or zone_polygon_local is None or zone_polygon_local.is_empty:
        return geometry

    try:
        clipped = geometry.intersection(zone_polygon_local)
    except Exception:
        return geometry

    clipped = _keep_polygons(clipped)
    if clipped is None or clipped.is_empty:
        return None

    try:
        if hasattr(clipped, "area") and float(clipped.area) < 1e-6:
            return None
    except Exception:
        pass

    return clipped


def prepare_preclipped_geometry(
    *,
    gdf_buildings_local: Optional[gpd.GeoDataFrame],
    building_geometries_for_flatten: Any,
    gdf_water: Optional[gpd.GeoDataFrame],
    global_center: Any,
    zone_polygon_local: Optional[BaseGeometry],
    zone_prefix: str = "",
) -> GeometryPreclipResult:
    gdf_water_local = None
    water_geometries_local_for_bridges = None
    preclipped_to_zone = False

    if zone_polygon_local is None or zone_polygon_local.is_empty:
        return GeometryPreclipResult(
            gdf_buildings_local=gdf_buildings_local,
            building_geometries_for_flatten=building_geometries_for_flatten,
            gdf_water_local=None,
            water_geometries_local_for_bridges=None,
            preclipped_to_zone=False,
        )

    try:
        if gdf_buildings_local is not None and not gdf_buildings_local.empty:
            gdf_buildings_local = gdf_buildings_local.copy()
            gdf_buildings_local["geometry"] = gdf_buildings_local["geometry"].apply(
                lambda geom: _clip_geometry_to_zone(geom, zone_polygon_local)
            )
            gdf_buildings_local = gdf_buildings_local[gdf_buildings_local.geometry.notna()]
            gdf_buildings_local = gdf_buildings_local[~gdf_buildings_local.geometry.is_empty]
            building_geometries_for_flatten = [
                geometry
                for geometry in list(gdf_buildings_local.geometry.values)
                if geometry is not None and not geometry.is_empty
            ]

        if gdf_water is not None and not gdf_water.empty and global_center is not None:
            transformer = make_to_local_transformer(global_center)
            if transformer is not None:
                gdf_water_local_raw = gdf_water.copy()
                gdf_water_local_raw["geometry"] = gdf_water_local_raw["geometry"].apply(
                    lambda geom: (
                        None
                        if geom is None
                        else geom
                        if geom.is_empty
                        else transform_geometry(transformer, geom)
                    )
                )

                try:
                    water_geometries_local_for_bridges = list(gdf_water_local_raw.geometry.values)
                except Exception:
                    water_geometries_local_for_bridges = None

                gdf_water_local = gdf_water_local_raw.copy()
                gdf_water_local["geometry"] = gdf_water_local["geometry"].apply(
                    lambda geom: _clip_geometry_to_zone(geom, zone_polygon_local)
                )
                gdf_water_local = gdf_water_local[gdf_water_local.geometry.notna()]
                gdf_water_local = gdf_water_local[~gdf_water_local.geometry.is_empty]

        preclipped_to_zone = True
    except Exception as exc:
        print(f"[WARN] {zone_prefix} Failed to preclip source geometries to zone: {exc}")
        preclipped_to_zone = False
        gdf_water_local = None
        water_geometries_local_for_bridges = None

    return GeometryPreclipResult(
        gdf_buildings_local=gdf_buildings_local,
        building_geometries_for_flatten=building_geometries_for_flatten,
        gdf_water_local=gdf_water_local,
        water_geometries_local_for_bridges=water_geometries_local_for_bridges,
        preclipped_to_zone=preclipped_to_zone,
    )
