from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import trimesh

from services.mesh_clipper import clip_mesh_to_bbox, clip_mesh_to_polygon


@dataclass
class MeshClipPipelineResult:
    terrain_mesh: Optional[trimesh.Trimesh]
    road_mesh: Optional[trimesh.Trimesh]
    building_meshes: Any
    water_mesh: Optional[trimesh.Trimesh]
    parks_mesh: Optional[trimesh.Trimesh]


def clip_generated_meshes(
    *,
    terrain_mesh: Optional[trimesh.Trimesh],
    road_mesh: Optional[trimesh.Trimesh],
    building_meshes: Any,
    water_mesh: Optional[trimesh.Trimesh],
    parks_mesh: Optional[trimesh.Trimesh],
    bbox_meters: Any,
    zone_polygon_coords: Any,
    global_center: Any,
    preclipped_to_zone: bool,
    clip_tolerance: float = 0.1,
) -> MeshClipPipelineResult:
    if terrain_mesh is not None:
        if zone_polygon_coords is None:
            clipped_terrain = clip_mesh_to_bbox(terrain_mesh, bbox_meters, tolerance=clip_tolerance)
            if clipped_terrain is not None and len(clipped_terrain.vertices) > 0:
                terrain_mesh = clipped_terrain
            else:
                print("[WARN] Terrain mesh became empty after clipping; keeping the original mesh")

    # Roads are already clipped on the canonical 2D polygon stage.
    # Re-clipping the final mesh with face/plane slicing can introduce diagonal artifacts
    # on tile borders that should be straight 90-degree cuts.

    if building_meshes is not None and not preclipped_to_zone:
        clipped_buildings = []
        for building_mesh in building_meshes:
            if building_mesh is None:
                continue
            if zone_polygon_coords is not None:
                clipped = clip_mesh_to_polygon(
                    building_mesh,
                    zone_polygon_coords,
                    global_center=global_center,
                    tolerance=clip_tolerance,
                )
            else:
                clipped = clip_mesh_to_bbox(building_mesh, bbox_meters, tolerance=clip_tolerance)
            if clipped is not None and len(clipped.vertices) > 0 and len(clipped.faces) > 0:
                clipped_buildings.append(clipped)
        building_meshes = clipped_buildings if clipped_buildings else None

    if water_mesh is not None and not preclipped_to_zone:
        if zone_polygon_coords is not None:
            clipped_water = clip_mesh_to_polygon(
                water_mesh,
                zone_polygon_coords,
                global_center=global_center,
                tolerance=clip_tolerance,
            )
        else:
            clipped_water = clip_mesh_to_bbox(water_mesh, bbox_meters, tolerance=clip_tolerance)

        if clipped_water is not None and len(clipped_water.vertices) > 0 and len(clipped_water.faces) > 0:
            water_mesh = clipped_water
        else:
            water_mesh = None

    if parks_mesh is not None:
        if zone_polygon_coords is None:
            clipped_parks = clip_mesh_to_bbox(parks_mesh, bbox_meters, tolerance=clip_tolerance)
            if clipped_parks is not None and len(clipped_parks.vertices) > 0 and len(clipped_parks.faces) > 0:
                parks_mesh = clipped_parks
            else:
                parks_mesh = None

    return MeshClipPipelineResult(
        terrain_mesh=terrain_mesh,
        road_mesh=road_mesh,
        building_meshes=building_meshes,
        water_mesh=water_mesh,
        parks_mesh=parks_mesh,
    )
