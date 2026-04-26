from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import trimesh

from services.detail_layer_utils import (
    MICRO_REGION_THRESHOLD_MM,
    MIN_ROAD_WIDTH_MODEL_MM,
    MIN_LAND_WIDTH_MODEL_MM,
    filter_mesh_components_by_printability,
)
from services.mesh_quality import improve_mesh_for_3d_printing, validate_mesh_for_3d_printing


@dataclass
class MeshPostprocessResult:
    terrain_mesh: Optional[trimesh.Trimesh]
    road_mesh: Optional[trimesh.Trimesh]
    building_meshes: Any
    water_mesh: Optional[trimesh.Trimesh]
    parks_mesh: Optional[trimesh.Trimesh]


def _remove_tiny_detached_faces(
    mesh: Optional[trimesh.Trimesh],
    *,
    label: str,
    max_faces: int = 4,
    max_vertices: int = 4,
) -> Optional[trimesh.Trimesh]:
    if mesh is None or len(mesh.vertices) == 0:
        return mesh
    try:
        components = list(mesh.split(only_watertight=False))
    except Exception:
        return mesh
    if len(components) <= 1:
        return mesh

    kept = []
    removed = 0
    for component in components:
        if component is None or len(component.vertices) == 0:
            removed += 1
            continue
        if len(component.faces) <= max_faces or len(component.vertices) <= max_vertices:
            removed += 1
            continue
        kept.append(component)

    if removed <= 0:
        return mesh
    if not kept:
        return None

    print(f"[{label}] Removed {removed} tiny detached mesh components")
    try:
        return trimesh.util.concatenate(kept)
    except Exception:
        return kept[0]


def _sanitize_mesh_topology_no_drift(
    mesh: Optional[trimesh.Trimesh],
    *,
    label: str,
) -> Optional[trimesh.Trimesh]:
    """Remove degenerate/duplicate topology without moving vertices."""
    if mesh is None or len(mesh.vertices) == 0 or mesh.faces is None or len(mesh.faces) == 0:
        return mesh
    cleaned = mesh.copy()
    before_faces = int(len(cleaned.faces))
    try:
        # Keep original coordinates: no smoothing, no vertex relocation.
        cleaned.update_faces(cleaned.unique_faces())
        cleaned.remove_unreferenced_vertices()
        try:
            cleaned.update_faces(cleaned.nondegenerate_faces())
            cleaned.remove_unreferenced_vertices()
        except Exception:
            pass
    except Exception:
        return mesh

    after_faces = int(len(cleaned.faces)) if cleaned.faces is not None else 0
    if after_faces <= 0:
        return mesh
    removed = max(0, before_faces - after_faces)
    if removed > 0:
        print(f"[{label}] Removed {removed} duplicate/degenerate faces (no-drift topology cleanup)")
    return cleaned


def _keep_dominant_component_if_fragmented(
    mesh: Optional[trimesh.Trimesh],
    *,
    label: str,
    min_dominant_face_ratio: float = 0.985,
    min_fragment_count: int = 3,
) -> Optional[trimesh.Trimesh]:
    if mesh is None or len(mesh.vertices) == 0:
        return mesh
    try:
        components = [
            c for c in mesh.split(only_watertight=False)
            if c is not None and getattr(c, "faces", None) is not None and len(c.faces) > 0
        ]
    except Exception:
        return mesh
    if len(components) < int(min_fragment_count):
        return mesh

    face_counts = [int(len(c.faces)) for c in components]
    total_faces = int(sum(face_counts))
    if total_faces <= 0:
        return mesh
    dominant_idx = max(range(len(components)), key=lambda idx: face_counts[idx])
    dominant_faces = int(face_counts[dominant_idx])
    dominant_ratio = float(dominant_faces) / float(total_faces)
    if dominant_ratio < float(min_dominant_face_ratio):
        return mesh

    dropped = len(components) - 1
    print(
        f"[{label}] Keeping dominant component only "
        f"(ratio={dominant_ratio:.4f}, dropped={dropped})"
    )
    return components[dominant_idx]


def postprocess_generated_meshes(
    *,
    task: Any,
    request: Any,
    scale_factor: Optional[float],
    terrain_mesh: Optional[trimesh.Trimesh],
    road_mesh: Optional[trimesh.Trimesh],
    building_meshes: Any,
    water_mesh: Optional[trimesh.Trimesh],
    parks_mesh: Optional[trimesh.Trimesh],
) -> MeshPostprocessResult:
    task.update_status("processing", 75, "Покращення якості mesh для 3D принтера...")

    print(f"\n{'='*60}")
    print("ПОКРАЩЕННЯ ЯКОСТІ MESH ДЛЯ 3D ДРУКУ")
    print(f"{'='*60}")

    micro_region_threshold_mm = float(MICRO_REGION_THRESHOLD_MM)
    land_region_threshold_mm = float(MIN_LAND_WIDTH_MODEL_MM)

    if terrain_mesh is not None:
        print("\n--- TERRAIN MESH ---")
        has_grooves = road_mesh is not None or parks_mesh is not None
        terrain_mesh = improve_mesh_for_3d_printing(
            terrain_mesh,
            aggressive=True,
            verbose=True,
            skip_fix_normals=has_grooves,
        )
        _, mesh_warnings = validate_mesh_for_3d_printing(
            terrain_mesh,
            scale_factor=scale_factor,
            model_size_mm=request.model_size_mm,
        )
        if mesh_warnings:
            print("[INFO] Terrain mesh quality warnings:")
            for warning in mesh_warnings:
                print(f"  - {warning}")
        terrain_mesh = filter_mesh_components_by_printability(
            terrain_mesh,
            scale_factor=scale_factor,
            min_feature_mm=land_region_threshold_mm,
            label="TERRAIN",
        )
        terrain_mesh = _remove_tiny_detached_faces(terrain_mesh, label="TERRAIN")
        terrain_mesh = _keep_dominant_component_if_fragmented(
            terrain_mesh,
            label="TERRAIN",
        )
    if road_mesh is not None:
        print("\n--- ROAD MESH ---")
        print("[ROAD] Preserving generated road mesh to avoid postprocess footprint/top-surface drift")
        road_mesh = _sanitize_mesh_topology_no_drift(road_mesh, label="ROAD")
        _, mesh_warnings = validate_mesh_for_3d_printing(
            road_mesh,
            scale_factor=scale_factor,
            model_size_mm=request.model_size_mm,
        )
        if mesh_warnings:
            print("[INFO] Road mesh quality warnings:")
            for warning in mesh_warnings:
                print(f"  - {warning}")
        road_mesh = filter_mesh_components_by_printability(
            road_mesh,
            scale_factor=scale_factor,
            min_feature_mm=MIN_ROAD_WIDTH_MODEL_MM,
            label="ROAD",
        )
        road_mesh = _remove_tiny_detached_faces(
            road_mesh,
            label="ROAD",
            max_faces=2,
            max_vertices=3,
        )

    if building_meshes is not None:
        print(f"\n--- BUILDING MESHES ({len(building_meshes)} buildings) ---")
        improved_buildings = []
        for index, building_mesh in enumerate(building_meshes):
            if building_mesh is None:
                continue
            verbose = index < 3 or (index % 50 == 0)
            if verbose:
                print(f"  Improving building {index + 1}/{len(building_meshes)}...")
            improved = improve_mesh_for_3d_printing(building_mesh, aggressive=True, verbose=verbose)
            improved = filter_mesh_components_by_printability(
                improved,
                scale_factor=scale_factor,
                min_feature_mm=micro_region_threshold_mm,
                label=f"BUILDING {index + 1}",
            )
            if improved is not None and len(improved.vertices) > 0:
                improved_buildings.append(improved)
        building_meshes = improved_buildings
        print(f"  Improved {len(building_meshes)} buildings")

    if water_mesh is not None:
        print("\n--- WATER MESH ---")
        water_mesh = improve_mesh_for_3d_printing(water_mesh, aggressive=True, verbose=True)
        water_mesh = filter_mesh_components_by_printability(
            water_mesh,
            scale_factor=scale_factor,
            min_feature_mm=micro_region_threshold_mm,
            label="WATER",
        )
        water_mesh = _remove_tiny_detached_faces(
            water_mesh,
            label="WATER",
            max_faces=12,
            max_vertices=12,
        )

    if parks_mesh is not None:
        print("\n--- PARKS MESH ---")
        parks_mesh = improve_mesh_for_3d_printing(parks_mesh, aggressive=True, verbose=True)
        parks_mesh = filter_mesh_components_by_printability(
            parks_mesh,
            scale_factor=scale_factor,
            min_feature_mm=land_region_threshold_mm,
            label="PARKS",
        )
        parks_mesh = _remove_tiny_detached_faces(
            parks_mesh,
            label="PARKS",
            max_faces=8,
            max_vertices=8,
        )

    print(f"\n{'='*60}")
    print("ПОКРАЩЕННЯ ЯКОСТІ ЗАВЕРШЕНО")
    print(f"{'='*60}\n")

    return MeshPostprocessResult(
        terrain_mesh=terrain_mesh,
        road_mesh=road_mesh,
        building_meshes=building_meshes,
        water_mesh=water_mesh,
        parks_mesh=parks_mesh,
    )
