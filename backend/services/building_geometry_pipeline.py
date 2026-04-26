from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union

from services.geometry_context import make_to_local_transformer


@dataclass
class BuildingGeometryPreparationResult:
    gdf_buildings_local: Optional[gpd.GeoDataFrame]
    building_geometries_for_flatten: Any
    building_union_local: Optional[BaseGeometry]


def prepare_building_geometry(
    *,
    gdf_buildings: Optional[gpd.GeoDataFrame],
    global_center: Any,
    zone_prefix: str = "",
) -> BuildingGeometryPreparationResult:
    gdf_buildings_local = None
    building_geometries_for_flatten = None
    building_union_local = None

    if gdf_buildings is None or gdf_buildings.empty or global_center is None:
        return BuildingGeometryPreparationResult(
            gdf_buildings_local=gdf_buildings_local,
            building_geometries_for_flatten=building_geometries_for_flatten,
            building_union_local=building_union_local,
        )

    try:
        transformer = make_to_local_transformer(global_center)
        if transformer is None:
            raise ValueError("Global center transformer is unavailable")

        print(
            f"[DEBUG] {zone_prefix} Transforming building geometries once for flatten and building processing"
        )
        gdf_buildings_local = gdf_buildings.copy()
        gdf_buildings_local["geometry"] = gdf_buildings_local["geometry"].apply(
            lambda geom: (
                geom
                if geom is None or geom.is_empty
                else transform_geometry(transformer, geom)
            )
        )

        building_geometries_for_flatten = [
            geom for geom in gdf_buildings_local.geometry.values if geom is not None and not geom.is_empty
        ]
        print(
            f"[DEBUG] {zone_prefix} Converted {len(building_geometries_for_flatten)} building geometries to local coordinates"
        )

        if building_geometries_for_flatten:
            try:
                building_union_local = unary_union(building_geometries_for_flatten)
            except Exception as exc:
                print(f"[WARN] {zone_prefix} Failed to create local building union: {exc}")
                building_union_local = None
    except Exception as exc:
        print(f"[WARN] {zone_prefix} Failed to convert building geometries to local coordinates: {exc}")
        import traceback

        traceback.print_exc()
        gdf_buildings_local = gdf_buildings
        building_geometries_for_flatten = list(gdf_buildings.geometry.values) if not gdf_buildings.empty else None
        building_union_local = None

    return BuildingGeometryPreparationResult(
        gdf_buildings_local=gdf_buildings_local,
        building_geometries_for_flatten=building_geometries_for_flatten,
        building_union_local=building_union_local,
    )
