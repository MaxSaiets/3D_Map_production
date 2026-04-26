from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh

from services.detail_layer_utils import (
    MICRO_REGION_THRESHOLD_MM,
    MIN_ROAD_WIDTH_MODEL_MM,
    filter_mesh_components_by_printability,
    model_mm_to_world_m,
)


@dataclass
class ValidatorConfig:
    scale_factor: float
    min_feature_mm: float = MICRO_REGION_THRESHOLD_MM
    min_wall_mm: float = MIN_ROAD_WIDTH_MODEL_MM
    fill_holes: bool = True
    repair_normals: bool = True


def _footprint_extents(mesh: trimesh.Trimesh) -> tuple[float, float]:
    try:
        mins = mesh.vertices[:, :2].min(axis=0)
        maxs = mesh.vertices[:, :2].max(axis=0)
        ext = np.asarray(maxs - mins, dtype=float)
        return float(np.min(ext)), float(np.max(ext))
    except Exception:
        return 0.0, 0.0


def _reject_thin_walled_components(
    mesh: Optional[trimesh.Trimesh],
    *,
    min_wall_m: float,
    min_feature_m: float,
    label: str,
) -> Optional[trimesh.Trimesh]:
    if mesh is None or len(mesh.vertices) == 0 or min_wall_m <= 0:
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
        min_xy, max_xy = _footprint_extents(component)
        if min_xy < min_wall_m and max_xy < min_wall_m * 2.0:
            removed += 1
            continue
        try:
            footprint_area = float(component.area) if hasattr(component, "area") else 0.0
        except Exception:
            footprint_area = 0.0
        if footprint_area > 0 and footprint_area < min_feature_m * min_feature_m:
            removed += 1
            continue
        kept.append(component)

    if removed == 0:
        return mesh
    if not kept:
        print(f"[{label}] All {removed} components rejected by wall/feature threshold")
        return None
    print(f"[{label}] Rejected {removed} thin/tiny component(s)")
    try:
        return trimesh.util.concatenate(kept)
    except Exception:
        return kept[0]


def _repair_mesh(mesh: Optional[trimesh.Trimesh], *, fill_holes: bool, repair_normals: bool, label: str) -> Optional[trimesh.Trimesh]:
    if mesh is None or len(mesh.vertices) == 0:
        return mesh
    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass
    if fill_holes:
        try:
            mesh.fill_holes()
        except Exception:
            pass
    if repair_normals:
        try:
            mesh.fix_normals()
        except Exception:
            pass
    if not getattr(mesh, "is_winding_consistent", True):
        print(f"[{label}] Winding not consistent after repair")
    return mesh


def validate_mesh(
    mesh: Optional[trimesh.Trimesh],
    *,
    config: ValidatorConfig,
    label: str = "3D_VALIDATOR",
) -> Optional[trimesh.Trimesh]:
    """Slicer-style validation: drop unprintable components, repair, fill holes.

    This is intentionally conservative — we only remove isolated thin/tiny
    components that can never print, and we repair trivial mesh defects. We do
    NOT modify overall geometry.
    """
    if mesh is None or len(mesh.vertices) == 0:
        return mesh
    scale = float(config.scale_factor)
    if scale <= 0:
        return mesh

    min_feature_m = model_mm_to_world_m(config.min_feature_mm, scale)
    min_wall_m = model_mm_to_world_m(config.min_wall_mm, scale)

    cleaned = filter_mesh_components_by_printability(
        mesh,
        scale_factor=scale,
        min_feature_mm=config.min_feature_mm,
        label=label,
    )
    cleaned = _reject_thin_walled_components(
        cleaned,
        min_wall_m=min_wall_m,
        min_feature_m=min_feature_m,
        label=label,
    )
    cleaned = _repair_mesh(
        cleaned,
        fill_holes=config.fill_holes,
        repair_normals=config.repair_normals,
        label=label,
    )

    if cleaned is None or len(cleaned.vertices) == 0:
        print(f"[{label}] Mesh removed entirely by printability pass")
        return None

    try:
        verts = len(cleaned.vertices)
        faces = len(cleaned.faces)
        print(f"[{label}] Validated mesh: {verts} verts, {faces} faces")
    except Exception:
        pass
    return cleaned


def validate_road_mesh(mesh: Optional[trimesh.Trimesh], *, scale_factor: float) -> Optional[trimesh.Trimesh]:
    cfg = ValidatorConfig(
        scale_factor=scale_factor,
        min_feature_mm=MICRO_REGION_THRESHOLD_MM,
        min_wall_mm=MIN_ROAD_WIDTH_MODEL_MM,
        fill_holes=True,
        repair_normals=True,
    )
    return validate_mesh(mesh, config=cfg, label="ROAD_VALIDATOR")


def validate_building_mesh(mesh: Optional[trimesh.Trimesh], *, scale_factor: float) -> Optional[trimesh.Trimesh]:
    cfg = ValidatorConfig(
        scale_factor=scale_factor,
        min_feature_mm=MICRO_REGION_THRESHOLD_MM,
        min_wall_mm=MIN_ROAD_WIDTH_MODEL_MM,
        fill_holes=True,
        repair_normals=True,
    )
    return validate_mesh(mesh, config=cfg, label="BUILDING_VALIDATOR")
