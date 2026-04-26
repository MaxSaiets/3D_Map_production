from __future__ import annotations

from typing import Any, Optional

import trimesh

from services.building_supports import union_mesh_collection
from services.processing_results import TerrainBuildingMergeResult


def _component_count(mesh: Optional[trimesh.Trimesh]) -> int:
    if mesh is None:
        return 0
    try:
        return len(mesh.split(only_watertight=False))
    except Exception:
        return 0


def merge_terrain_and_buildings(
    *,
    terrain_mesh: Optional[trimesh.Trimesh],
    building_meshes: Any,
    merged_building_mesh: Optional[trimesh.Trimesh] = None,
    support_meshes: Any = None,
) -> TerrainBuildingMergeResult:
    if terrain_mesh is None:
        return TerrainBuildingMergeResult(
            terrain_mesh=terrain_mesh,
            building_meshes=building_meshes,
            merged_building_mesh=merged_building_mesh,
            support_meshes=support_meshes,
        )

    if merged_building_mesh is None:
        return TerrainBuildingMergeResult(
            terrain_mesh=terrain_mesh,
            building_meshes=building_meshes,
            merged_building_mesh=merged_building_mesh,
            support_meshes=support_meshes,
        )

    terrain_components = _component_count(terrain_mesh)
    building_components = _component_count(merged_building_mesh)

    base_mesh = None
    try:
        base_mesh = union_mesh_collection(
            [terrain_mesh, merged_building_mesh],
            label="terrain_buildings_base",
        )
    except Exception as exc:
        print(f"[WARN] terrain/building boolean merge failed: {exc}")
        base_mesh = None

    if base_mesh is not None:
        merged_components = _component_count(base_mesh)
        allowed_components = max(terrain_components + 8, terrain_components * 4)
        if merged_components > allowed_components and merged_components >= (terrain_components + building_components):
            print(
                "[WARN] terrain/building merge produced fragmented base "
                f"({merged_components} components, terrain={terrain_components}, buildings={building_components}); "
                "keeping terrain as base and exporting buildings separately"
            )
            base_mesh = None

    exported_buildings = building_meshes
    if base_mesh is None:
        base_mesh = terrain_mesh
    else:
        exported_buildings = None

    return TerrainBuildingMergeResult(
        terrain_mesh=base_mesh,
        building_meshes=exported_buildings,
        merged_building_mesh=merged_building_mesh,
        support_meshes=support_meshes,
    )
