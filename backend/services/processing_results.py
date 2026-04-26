from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import trimesh
from shapely.geometry.base import BaseGeometry


@dataclass
class RoadProcessingResult:
    mesh: Optional[trimesh.Trimesh]
    source_polygons: Optional[BaseGeometry]
    cutting_polygons: Optional[BaseGeometry]


@dataclass
class GreenAreaProcessingResult:
    mesh: Optional[trimesh.Trimesh]
    processed_polygons: Optional[BaseGeometry]


@dataclass
class RoadLayerResult:
    mesh: Optional[trimesh.Trimesh]
    road_result: Optional[RoadProcessingResult]
    road_cut_source: Optional[BaseGeometry]


@dataclass
class BuildingLayerResult:
    meshes: Any
    support_meshes: Any = None
    merged_mesh: Optional[trimesh.Trimesh] = None
    footprints: Optional[BaseGeometry] = None


@dataclass
class WaterLayerResult:
    mesh: Optional[trimesh.Trimesh]
    cutting_polygons: Optional[BaseGeometry] = None


@dataclass
class ParkLayerResult:
    mesh: Optional[trimesh.Trimesh]
    parks_result: Optional[GreenAreaProcessingResult]


@dataclass
class ZonePreparationResult:
    zone_polygon_local: Optional[BaseGeometry]
    reference_xy_m: Optional[tuple[float, float]]
    bbox_meters: tuple[float, float, float, float]
    scale_factor: float
    road_width_multiplier_effective: float
    stl_extra_embed_m: float


@dataclass
class SourceDataResult:
    gdf_buildings: Any
    gdf_water: Any
    G_roads: Any
    gdf_green: Any


@dataclass
class TerrainStageResult:
    terrain_mesh: Optional[trimesh.Trimesh]
    terrain_provider: Any
    road_cut_mask: Optional[BaseGeometry]
    road_height_m: Optional[float]
    road_embed_m: Optional[float]
    water_depth_m: float
    gdf_buildings_local: Any
    building_geometries_for_flatten: Any
    building_union_local: Optional[BaseGeometry]
    gdf_water_local: Any
    merged_roads_geom: Optional[BaseGeometry]
    merged_roads_geom_local: Optional[BaseGeometry]
    preclipped_to_zone: bool


@dataclass
class TerrainBuildingMergeResult:
    terrain_mesh: Optional[trimesh.Trimesh]
    building_meshes: Any
    merged_building_mesh: Optional[trimesh.Trimesh]
    support_meshes: Any = None
