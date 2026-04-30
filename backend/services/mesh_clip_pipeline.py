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
    # ── Terrain ───────────────────────────────────────────────────────────────
    # Clip terrain to zone polygon when available, otherwise to bbox.
    # terrain_generator already clips internally, but this handles edge cases
    # where terrain extends slightly beyond zone due to grid snapping.
    if terrain_mesh is not None:
        if zone_polygon_coords is not None:
            clipped_terrain = clip_mesh_to_polygon(
                terrain_mesh,
                zone_polygon_coords,
                global_center=global_center,
                tolerance=clip_tolerance,
            )
            if clipped_terrain is not None and len(clipped_terrain.vertices) > 0:
                terrain_mesh = clipped_terrain
            else:
                print("[WARN] Terrain mesh became empty after polygon clipping; keeping the original mesh")
        else:
            clipped_terrain = clip_mesh_to_bbox(terrain_mesh, bbox_meters, tolerance=clip_tolerance)
            if clipped_terrain is not None and len(clipped_terrain.vertices) > 0:
                terrain_mesh = clipped_terrain
            else:
                print("[WARN] Terrain mesh became empty after bbox clipping; keeping the original mesh")

    # ── Roads ─────────────────────────────────────────────────────────────────
    # Always clip roads to bbox. Roads are pre-clipped to zone polygon during
    # canonical 2D geometry stage, but the final 3D mesh can extend slightly
    # beyond bbox due to road width buffering near zone borders.
    # Bbox clipping uses axis-aligned plane cuts — no diagonal artifacts.
    # Without this clip, unbound roads dominate combined export bounds and
    # cause everything else to appear tiny (scale distortion bug).
    if road_mesh is not None:
        clipped_roads = clip_mesh_to_bbox(road_mesh, bbox_meters, tolerance=clip_tolerance)
        if clipped_roads is not None and len(clipped_roads.vertices) > 0 and len(clipped_roads.faces) > 0:
            road_mesh = clipped_roads
        else:
            print("[WARN] Road mesh became empty after bbox clipping; keeping the original road mesh")

    # ── Buildings ─────────────────────────────────────────────────────────────
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

    # ── Water ─────────────────────────────────────────────────────────────────
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

    # ── Parks ─────────────────────────────────────────────────────────────────
    # Fix: previously parks were only clipped when zone_polygon_coords was None
    # (inverted condition). Now clip to zone polygon when available, else bbox.
    if parks_mesh is not None:
        if zone_polygon_coords is not None:
            clipped_parks = clip_mesh_to_polygon(
                parks_mesh,
                zone_polygon_coords,
                global_center=global_center,
                tolerance=clip_tolerance,
            )
        else:
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
