"""
3D Model Export Service
Formats: STL (binary, slicer-safe), 3MF
Guarantees:
- no empty geometry
- no broken STL headers
- no Scene→STL corruption
- Cura / Prusa / Bambu safe
"""

from typing import List, Tuple, Optional, Union, Dict
import os
import numpy as np
import trimesh
import trimesh.visual
from services.mesh_quality import improve_mesh_for_3d_printing, detect_nonmanifold_edges
from services.detail_layer_utils import MICRO_REGION_THRESHOLD_MM
# from trimesh.exchange.stl import export_stl_binary


def _boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    try:
        edges = mesh.edges_sorted
        if len(edges) == 0:
            return 0
        unique = trimesh.grouping.group_rows(edges, require_count=1)
        return int(len(unique))
    except Exception:
        return 10**9


def _road_export_candidate_score(mesh: Optional[trimesh.Trimesh]) -> tuple:
    if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
        return (-1, -10**9, -10**9, -1)
    return (
        1 if mesh.is_watertight else 0,
        -_boundary_edge_count(mesh),
        -len(list(mesh.split(only_watertight=False))),
        int(len(mesh.faces)),
    )


def _base_export_candidate_score(mesh: Optional[trimesh.Trimesh]) -> tuple:
    if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
        return (-1, -10**9, -10**9, -1)
    return (
        1 if bool(getattr(mesh, "is_watertight", False)) else 0,
        -_boundary_edge_count(mesh),
        -len(list(mesh.split(only_watertight=False))),
        int(len(mesh.faces)),
    )


def _dominant_component_if_safe(
    mesh: Optional[trimesh.Trimesh],
    *,
    min_face_ratio: float = 0.995,
) -> Optional[trimesh.Trimesh]:
    if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
        return None
    try:
        components = list(mesh.split(only_watertight=False))
    except Exception:
        return None
    if len(components) <= 1:
        return None
    ranked = sorted(
        [component for component in components if component is not None and component.faces is not None and len(component.faces) > 0],
        key=lambda item: len(item.faces),
        reverse=True,
    )
    if not ranked:
        return None
    dominant = ranked[0]
    total_faces = sum(len(component.faces) for component in ranked)
    if total_faces <= 0:
        return None
    dominant_ratio = float(len(dominant.faces)) / float(total_faces)
    if dominant_ratio < float(min_face_ratio):
        return None
    return dominant.copy()


def repair_base_export_mesh(mesh: Optional[trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
    if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
        return mesh

    original = mesh.copy()
    candidate = mesh.copy()
    try:
        trimesh.repair.fill_holes(candidate)
    except Exception:
        return original
    try:
        trimesh.repair.fix_winding(candidate)
    except Exception:
        pass
    try:
        candidate.fix_normals()
    except Exception:
        pass
    try:
        candidate.update_faces(candidate.unique_faces())
        candidate.remove_unreferenced_vertices()
    except Exception:
        pass

    if _base_export_candidate_score(candidate) >= _base_export_candidate_score(original):
        return candidate
    return original


def repair_base_export_mesh_aggressive(mesh: Optional[trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
    if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
        return mesh

    original = mesh.copy()
    dominant = _dominant_component_if_safe(original, min_face_ratio=0.985)
    if dominant is not None:
        # Print-fit rule: when 98.5%+ faces belong to one component, treat the
        # rest as boolean debris and force-export only the dominant body.
        original = dominant
    candidate = repair_base_export_mesh(original.copy())
    dominant_candidate = _dominant_component_if_safe(candidate, min_face_ratio=0.985)
    if dominant_candidate is not None:
        candidate = dominant_candidate
    try:
        trimesh.repair.fill_holes(candidate)
    except Exception:
        pass
    try:
        trimesh.repair.fix_winding(candidate)
    except Exception:
        pass
    try:
        candidate.fix_normals()
    except Exception:
        pass
    try:
        candidate.update_faces(candidate.unique_faces())
        candidate.remove_unreferenced_vertices()
    except Exception:
        pass
    if _base_export_candidate_score(candidate) >= _base_export_candidate_score(original):
        return candidate
    return original


def _base_plane_profile(mesh: trimesh.Trimesh) -> Optional[dict]:
    if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
        return None
    vertices = np.asarray(mesh.vertices, dtype=float)
    extents = np.asarray(getattr(mesh, "extents", np.zeros(3)), dtype=float)
    if vertices.shape[0] == 0 or extents.shape[0] != 3:
        return None

    z_values = vertices[:, 2]
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    z_extent = float(max(z_max - z_min, 1e-9))
    plane_tol = max(z_extent * 0.002, 0.02)

    def _plane_stats(mask: np.ndarray) -> tuple[int, float]:
        if not np.any(mask):
            return 0, 0.0
        plane_vertices = vertices[mask][:, :2]
        mins = plane_vertices.min(axis=0)
        maxs = plane_vertices.max(axis=0)
        span = np.maximum(maxs - mins, 0.0)
        return int(plane_vertices.shape[0]), float(span[0] * span[1])

    bottom_mask = z_values <= (z_min + plane_tol)
    top_mask = z_values >= (z_max - plane_tol)
    bottom_count, bottom_area = _plane_stats(bottom_mask)
    top_count, top_area = _plane_stats(top_mask)
    xy_area = float(max(extents[0], 1e-9) * max(extents[1], 1e-9))

    return {
        "z_min": z_min,
        "z_max": z_max,
        "z_extent": z_extent,
        "xy_area": xy_area,
        "bottom_count": bottom_count,
        "bottom_area": bottom_area,
        "bottom_ratio": float(bottom_area / xy_area) if xy_area > 1e-9 else 0.0,
        "top_count": top_count,
        "top_area": top_area,
        "top_ratio": float(top_area / xy_area) if xy_area > 1e-9 else 0.0,
        "extents": extents,
    }


def _base_print_orientation_score(mesh: Optional[trimesh.Trimesh]) -> tuple:
    if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
        return (-1.0, -1.0, -1.0, -1.0)
    profile = _base_plane_profile(mesh)
    if not profile:
        return (-1.0, -1.0, -1.0, -1.0)

    extents = np.asarray(profile["extents"], dtype=float)
    z_extent = float(profile["z_extent"])
    xy_area = float(profile["xy_area"])
    bottom_ratio = float(profile["bottom_ratio"])
    top_ratio = float(profile["top_ratio"])
    bottom_count = int(profile["bottom_count"])
    top_count = int(profile["top_count"])

    # Prefer orientations where Z is the thinnest dimension and the bottom plane
    # covers most of the XY footprint. This matches a printable terrain tile.
    z_is_min = 1.0 if z_extent <= float(np.min(extents)) + 1e-6 else 0.0
    support_bias = float(bottom_ratio - max(top_ratio - bottom_ratio, 0.0))
    return (
        z_is_min,
        support_bias,
        float(bottom_ratio),
        float(bottom_count - top_count),
        -z_extent,
        xy_area,
    )


def _generic_print_orientation_score(mesh: Optional[trimesh.Trimesh]) -> tuple:
    if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
        return (-1.0, -1.0, -1.0, -1.0)
    profile = _base_plane_profile(mesh)
    if not profile:
        return (-1.0, -1.0, -1.0, -1.0)
    extents = np.asarray(profile["extents"], dtype=float)
    z_extent = float(profile["z_extent"])
    bottom_ratio = float(profile["bottom_ratio"])
    top_ratio = float(profile["top_ratio"])
    bottom_count = int(profile["bottom_count"])
    z_is_min = 1.0 if z_extent <= float(np.min(extents)) + 1e-6 else 0.0
    return (
        z_is_min,
        float(bottom_ratio),
        float(bottom_count),
        -z_extent,
        -max(top_ratio - bottom_ratio, 0.0),
    )


def _flip_mesh_z(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    flipped = mesh.copy()
    vertices = np.asarray(flipped.vertices, dtype=float).copy()
    z_min = float(np.min(vertices[:, 2]))
    z_max = float(np.max(vertices[:, 2]))
    vertices[:, 2] = z_max - (vertices[:, 2] - z_min)
    flipped.vertices = vertices
    try:
        trimesh.repair.fix_winding(flipped)
    except Exception:
        pass
    try:
        flipped.fix_normals()
    except Exception:
        pass
    return flipped


def _select_print_orientation(mesh: Optional[trimesh.Trimesh]) -> tuple[np.ndarray, str]:
    if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
        return np.eye(4), "identity"
    try:
        rotation_candidates: dict[str, np.ndarray] = {
            "identity": np.eye(4),
            "rot_x_pos_90": trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0]),
            "rot_x_neg_90": trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1, 0, 0]),
            "rot_y_pos_90": trimesh.transformations.rotation_matrix(np.pi / 2.0, [0, 1, 0]),
            "rot_y_neg_90": trimesh.transformations.rotation_matrix(-np.pi / 2.0, [0, 1, 0]),
        }

        best_name = "identity"
        best_matrix = np.eye(4)
        best_mesh = mesh.copy()
        best_score = _base_print_orientation_score(best_mesh)

        for name, matrix in rotation_candidates.items():
            rotated = mesh.copy()
            rotated.apply_transform(matrix)
            score = _base_print_orientation_score(rotated)
            if score > best_score:
                best_name = name
                best_matrix = matrix
                best_mesh = rotated
                best_score = score

        profile = _base_plane_profile(best_mesh)
        if profile and profile["top_ratio"] > (profile["bottom_ratio"] + 0.1):
            flipped = _flip_mesh_z(best_mesh)
            flipped_score = _base_print_orientation_score(flipped)
            if flipped_score >= best_score:
                flip_matrix = np.eye(4)
                z_min = float(np.min(np.asarray(best_mesh.vertices, dtype=float)[:, 2]))
                z_max = float(np.max(np.asarray(best_mesh.vertices, dtype=float)[:, 2]))
                flip_matrix[2, 2] = -1.0
                flip_matrix[2, 3] = z_max + z_min
                best_matrix = flip_matrix @ best_matrix
                best_name = f"{best_name}+flip_z"
        return best_matrix, best_name
    except Exception:
        return np.eye(4), "identity"


def _select_generic_part_orientation(mesh: Optional[trimesh.Trimesh]) -> tuple[np.ndarray, str]:
    if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
        return np.eye(4), "identity"
    try:
        rotation_candidates: dict[str, np.ndarray] = {
            "identity": np.eye(4),
            "rot_x_pos_90": trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0]),
            "rot_x_neg_90": trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1, 0, 0]),
            "rot_y_pos_90": trimesh.transformations.rotation_matrix(np.pi / 2.0, [0, 1, 0]),
            "rot_y_neg_90": trimesh.transformations.rotation_matrix(-np.pi / 2.0, [0, 1, 0]),
        }
        best_name = "identity"
        best_matrix = np.eye(4)
        best_score = _generic_print_orientation_score(mesh)
        best_mesh = mesh.copy()
        for name, matrix in rotation_candidates.items():
            rotated = mesh.copy()
            rotated.apply_transform(matrix)
            score = _generic_print_orientation_score(rotated)
            if score > best_score:
                best_name = name
                best_matrix = matrix
                best_score = score
                best_mesh = rotated

        profile = _base_plane_profile(best_mesh)
        if profile and profile["top_ratio"] > (profile["bottom_ratio"] + 0.1):
            flip_matrix = np.eye(4)
            z_min = float(np.min(np.asarray(best_mesh.vertices, dtype=float)[:, 2]))
            z_max = float(np.max(np.asarray(best_mesh.vertices, dtype=float)[:, 2]))
            flip_matrix[2, 2] = -1.0
            flip_matrix[2, 3] = z_max + z_min
            flipped = best_mesh.copy()
            flipped.apply_transform(flip_matrix)
            if _generic_print_orientation_score(flipped) >= best_score:
                best_name = f"{best_name}+flip_z"
                best_matrix = flip_matrix @ best_matrix
        return best_matrix, best_name
    except Exception:
        return np.eye(4), "identity"


def _normalize_part_for_print_export(
    mesh: Optional[trimesh.Trimesh],
    *,
    part_key: str,
) -> Optional[trimesh.Trimesh]:
    if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
        return mesh
    normalized = mesh.copy()
    try:
        min_z = float(normalized.bounds[0][2])
        normalized.apply_translation([0.0, 0.0, -min_z])
    except Exception:
        pass
    return normalized


def _normalize_source_part_axes(
    mesh: Optional[trimesh.Trimesh],
    *,
    part_key: str,
) -> Optional[trimesh.Trimesh]:
    if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
        return mesh
    # Keep all incoming parts in one coordinate frame. Per-part axis rotation
    # makes insert parts (roads/water/parks/buildings) stop matching the base.
    return mesh.copy()


def _sanitize_mesh_visual(mesh: Optional[trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
    if mesh is None:
        return mesh
    try:
        visual = getattr(mesh, "visual", None)
        if visual is None:
            return mesh
        if hasattr(visual, "face_colors") and visual.face_colors is not None:
            colors = np.asarray(visual.face_colors)
            if colors.ndim != 2 or colors.shape[0] not in (0, len(mesh.faces)) or colors.shape[1] < 3:
                mesh.visual = trimesh.visual.ColorVisuals()
    except Exception:
        try:
            mesh.visual = trimesh.visual.ColorVisuals()
        except Exception:
            pass
    return mesh


def _build_assembly_preview_parts(
    parts: Dict[str, trimesh.Trimesh],
) -> Dict[str, trimesh.Trimesh]:
    preview_parts: Dict[str, trimesh.Trimesh] = {}
    for key, mesh in (parts or {}).items():
        if mesh is None:
            continue
        try:
            copied = mesh.copy()
        except Exception:
            copied = mesh
        preview_parts[key] = _sanitize_mesh_visual(copied) or copied
    # IMPORTANT: keep original transformed coordinates for all parts.
    # Any extra preview-time Z alignment breaks inlay fit on realistic terrain
    # and causes severe z-fighting artifacts in 3MF viewers.
    return preview_parts


def orient_base_for_print(mesh: Optional[trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
    if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
        return mesh
    matrix, _ = _select_print_orientation(mesh)
    if np.allclose(matrix, np.eye(4)):
        return mesh
    oriented = mesh.copy()
    oriented.apply_transform(matrix)
    return oriented


def _rotation_align_vector_to_z(source_vec: np.ndarray) -> np.ndarray:
    source = np.asarray(source_vec, dtype=float)
    norm = float(np.linalg.norm(source))
    if norm <= 1e-12:
        return np.eye(4)
    source = source / norm
    target = np.array([0.0, 0.0, 1.0], dtype=float)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot >= 1.0 - 1e-9:
        return np.eye(4)
    if dot <= -1.0 + 1e-9:
        return trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])

    axis = np.cross(source, target)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        return np.eye(4)
    axis = axis / axis_norm
    angle = float(np.arccos(dot))
    return trimesh.transformations.rotation_matrix(angle, axis)


def _estimate_bottom_plane_normal(mesh: Optional[trimesh.Trimesh]) -> Optional[np.ndarray]:
    if mesh is None or mesh.vertices is None or len(mesh.vertices) < 16:
        return None
    vertices = np.asarray(mesh.vertices, dtype=float)
    z_values = vertices[:, 2]
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    z_extent = float(max(z_max - z_min, 1e-9))
    plane_tol = max(z_extent * 0.002, 0.02)
    bottom_points = vertices[z_values <= (z_min + plane_tol)]
    if len(bottom_points) < 12:
        q = float(np.quantile(z_values, 0.02))
        bottom_points = vertices[z_values <= q]
    if len(bottom_points) < 12:
        return None
    centered = bottom_points - bottom_points.mean(axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return None
    normal = np.asarray(vh[-1], dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return None
    normal = normal / norm
    if normal[2] < 0.0:
        normal = -normal
    return normal


def _stabilize_scene_bottom_plane(parts: Dict[str, trimesh.Trimesh]) -> tuple[Dict[str, trimesh.Trimesh], float]:
    if not parts:
        return parts, 0.0
    reference = parts.get("base") or parts.get("terrain")
    normal = _estimate_bottom_plane_normal(reference)
    if normal is None:
        return parts, 0.0
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    angle_rad = float(np.arccos(np.clip(np.dot(normal, z_axis), -1.0, 1.0)))
    angle_deg = float(np.degrees(angle_rad))
    if angle_deg <= 0.05:
        return parts, angle_deg

    rotation = _rotation_align_vector_to_z(normal)
    if np.allclose(rotation, np.eye(4)):
        return parts, angle_deg

    rotated: Dict[str, trimesh.Trimesh] = {}
    for key, mesh in parts.items():
        if mesh is None:
            continue
        moved = mesh.copy()
        moved.apply_transform(rotation)
        rotated[key] = moved
    return rotated, angle_deg


def _filter_mesh_components_for_export(
    mesh: Optional[trimesh.Trimesh],
    *,
    min_feature_mm: float = MICRO_REGION_THRESHOLD_MM,
    min_area_mm2: float = 0.05,
) -> Optional[trimesh.Trimesh]:
    if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
        return mesh

    try:
        components = list(mesh.split(only_watertight=False))
    except Exception:
        components = [mesh]

    kept = []
    for component in components:
        if component is None or component.faces is None or len(component.faces) == 0:
            continue
        part = component.copy()
        try:
            part.update_faces(part.unique_faces())
            part.update_faces(part.nondegenerate_faces())
            part.remove_unreferenced_vertices()
            part.merge_vertices(digits_vertex=6)
            part.remove_unreferenced_vertices()
            part.fix_normals()
        except Exception:
            pass
        if part.faces is None or len(part.faces) == 0:
            continue
        try:
            bounds = part.bounds
            extents = np.asarray(bounds[1] - bounds[0], dtype=float)
            max_xy = float(np.max(extents[:2])) if len(extents) >= 2 else 0.0
            min_xy = float(np.min(extents[:2])) if len(extents) >= 2 else 0.0
            area = float(getattr(part, "area", 0.0) or 0.0)
            if len(part.faces) <= 8:
                continue
            if area < min_area_mm2:
                continue
            if max_xy < float(min_feature_mm):
                continue
            if min_xy < float(min_feature_mm) and area < 15.0:
                continue
        except Exception:
            pass
        kept.append(part)

    if not kept:
        return mesh

    try:
        filtered = trimesh.util.concatenate(kept)
        filtered.update_faces(filtered.unique_faces())
        filtered.update_faces(filtered.nondegenerate_faces())
        filtered.remove_unreferenced_vertices()
        filtered.merge_vertices(digits_vertex=6)
        filtered.remove_unreferenced_vertices()
        filtered.fix_normals()
        return filtered
    except Exception:
        return mesh


def _quantize_mesh_vertices(
    mesh: Optional[trimesh.Trimesh],
    *,
    digits: int = 4,
) -> Optional[trimesh.Trimesh]:
    if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
        return mesh
    try:
        quantized = mesh.copy()
        quantized.vertices = np.round(np.asarray(quantized.vertices, dtype=float), digits)
        quantized.update_faces(quantized.unique_faces())
        quantized.update_faces(quantized.nondegenerate_faces())
        quantized.remove_unreferenced_vertices()
        quantized.merge_vertices(digits_vertex=digits)
        quantized.remove_unreferenced_vertices()
        quantized.fix_normals()
        return quantized
    except Exception:
        return mesh


def repair_road_export_mesh(mesh: Optional[trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
    if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
        return mesh
    original = _quantize_mesh_vertices(_filter_mesh_components_for_export(mesh.copy()), digits=4)
    try:
        components = list(original.split(only_watertight=False))
    except Exception:
        components = [original]

    repaired = []
    for component in components:
        if component is None or component.faces is None or len(component.faces) == 0:
            continue
        part = component.copy()
        try:
            part = _quantize_mesh_vertices(part, digits=4)
            part.fill_holes()
        except Exception:
            pass
        try:
            part.update_faces(part.unique_faces())
            part.update_faces(part.nondegenerate_faces())
            part.remove_unreferenced_vertices()
            part.merge_vertices(digits_vertex=6)
            trimesh.repair.fix_winding(part)
            part.fix_normals()
        except Exception:
            pass
        if part.faces is not None and len(part.faces) > 0:
            repaired.append(part)

    if not repaired:
        return original

    try:
        candidate = trimesh.util.concatenate(repaired)
        candidate = _quantize_mesh_vertices(candidate, digits=4)
        candidate = _filter_mesh_components_for_export(candidate)
        candidate.update_faces(candidate.unique_faces())
        candidate.update_faces(candidate.nondegenerate_faces())
        candidate.remove_unreferenced_vertices()
        candidate.merge_vertices(digits_vertex=6)
        candidate.remove_unreferenced_vertices()
        candidate.fix_normals()
    except Exception:
        return original

    if _road_export_candidate_score(candidate) >= _road_export_candidate_score(original):
        return candidate
    return original


# ============================================================
# GEOMETRY NORMALIZATION (SAFE, NON-DESTRUCTIVE)
# ============================================================

def to_trimesh(obj: Union[trimesh.Trimesh, trimesh.Scene], repair: bool = False) -> Optional[trimesh.Trimesh]:
    if obj is None:
        return None

    # Scene → merge valid meshes
    if isinstance(obj, trimesh.Scene):
        meshes = [
            g for g in obj.geometry.values()
            if isinstance(g, trimesh.Trimesh) and g.faces is not None and len(g.faces) > 0
        ]
        if not meshes:
            return None
        obj = trimesh.util.concatenate(meshes)

    if not isinstance(obj, trimesh.Trimesh):
        return None

    if obj.vertices is None or obj.faces is None:
        return None

    if len(obj.vertices) == 0 or len(obj.faces) == 0:
        return None

    m = obj.copy()

    # minimal safe cleanup (NO aggressive processing)
    try:
        m.update_faces(m.nondegenerate_faces())
        m.remove_unreferenced_vertices()
        m.merge_vertices(digits_vertex=8)
    except Exception:
        pass

    if len(m.faces) == 0:
        return None
    
    # Optional mesh repair for 3D printing
    if repair:
        try:
            # If mesh is already watertight, avoid aggressive repairs which can flip faces
            if m.is_watertight:
                try:
                    trimesh.repair.fix_winding(m)
                except Exception:
                    pass
                m.fix_normals()
            else:
                nm_count_before, _ = detect_nonmanifold_edges(m)
                if nm_count_before > 0:
                    print(f"[INFO] Detected {nm_count_before} non-manifold edges before repair")
                    m = improve_mesh_for_3d_printing(m, aggressive=True, verbose=True)
                    nm_count_after, _ = detect_nonmanifold_edges(m)
                    print(f"[INFO] Non-manifold edges after repair: {nm_count_after} (fixed {nm_count_before - nm_count_after})")
        except Exception as e:
            print(f"[WARN] Mesh repair failed: {e}")

    return m


def fallback_box() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=[5.0, 5.0, 1.0])


# ============================================================
# TRANSFORMS (SINGLE SOURCE OF TRUTH)
# ============================================================

def get_transform_matrix(
    bounds: np.ndarray,
    scale_factor: float, # Pass explicit scale factor
    reference_xy_m: Optional[Tuple[float, float]], # Kept for signature compatibility if needed, but unused for scale
    preserve_xy: bool,
    preserve_z: bool,
    rotate_to_ground: bool,
) -> np.ndarray:
    """
    Calculates the 4x4 transformation matrix to apply to all meshes.
    """
    matrix = np.eye(4)
    
    # 1. Center XY
    if not preserve_xy:
        # Calculate centroid of the bounding box
        c = (bounds[0] + bounds[1]) / 2.0
        
        # Translation to center: [-cx, -cy, 0]
        t_center = np.eye(4)
        t_center[0, 3] = -c[0]
        t_center[1, 3] = -c[1]
        
        matrix = np.dot(t_center, matrix)

    # 2. Scale
    if abs(scale_factor - 1.0) > 1e-9:
        s_matrix = np.eye(4)
        s_matrix[0, 0] = scale_factor
        s_matrix[1, 1] = scale_factor
        s_matrix[2, 2] = scale_factor
        
        matrix = np.dot(s_matrix, matrix)


    # 3. Rotate
    if rotate_to_ground:
        # Rotate -90 deg around X (вертикальна орієнтація для стіни/3D друку)
        # MODEL_ORIENTATION_FLIP=1 — інвертувати поворот (+90° замість -90°)
        angle = np.pi / 2 if os.environ.get("MODEL_ORIENTATION_FLIP", "").lower() in ("1", "true", "yes") else -np.pi / 2
        rot = trimesh.transformations.rotation_matrix(angle, [1, 0, 0])
        matrix = np.dot(rot, matrix)

    # 4. Ground Z (This is tricky because rotation might have changed Z)
    # Ideally, we ground Z *after* all other transforms.
    # But to do that in one matrix, we need the bounds *after* the previous transforms.
    # For simplicity, we'll return the matrix so far, apply it, check bounds, then apply Z translation.
    # OR, we assume standard pipeline order.
    
    return matrix


# ============================================================
# BASE GENERATION
# ============================================================

def create_flat_base(bounds: np.ndarray, thickness_mm: float) -> trimesh.Trimesh:
    """
    Creates a base box matching the bounds (pre-scaling? No, this function assumes
    we are generating the base matching the mesh provided).
    
    WAIT: If we want the base to be part of the "Base" component, we should generate it
    relative to the input coordinates (meters), OR generate it relative to output (mm).
    
    Original code used `add_flat_base` on the Combined mesh (which was still in meters usually, 
    then scaled? logic in `prepare_mesh`: 
    combined -> add_flat_base -> apply_transforms.
    So base is added in METERS (or input units).
    """
    size = bounds[1] - bounds[0]
    
    # Thickness needs to be in input units (meters) to match the mesh scale?
    # NO. The arguments to `prepare_mesh` say `base_thickness_mm`. 
    # But `apply_transforms` scales everything by `model_size_mm / avg_xy`.
    # If we add base *before* scaling, we must calculate what `thickness_mm` corresponds to in meters.
    # This is complicated. exact thickness is desired in MM.
    
    # ALTERNATIVE:
    # 1. Transform all meshes to MM.
    # 2. Calculate bounds of combined MM meshes.
    # 3. Generate base in MM.
    # 4. Add to Base component.
    
    # Let's try to stick to this: Transform -> Add Base.
    # BUT, `apply_transforms` handles "Ground Z". If we add base after, we need to respect that.
    
    base = trimesh.creation.box(
        extents=[
            max(size[0], 1e-3),
            max(size[1], 1e-3),
            max(thickness_mm, 0.1), # This is tricky. If input is meters, this is wrong.
        ]
    )
    
    # Position logic...
    return base


def create_base_in_mesh_space(bounds: np.ndarray, target_thickness_mm: float, scale_factor: float) -> trimesh.Trimesh:
    """
    Creates the base geometry in the same coordinate space as the input mesh (Meters),
    calculating the required thickness in Meters so that after scaling it becomes target_thickness_mm.
    """
    size = bounds[1] - bounds[0]
    
    # Calculate thickness in mesh units
    # scale_factor = mm / meter
    # thickness_m = thickness_mm / scale_factor
    thickness_mesh_units = target_thickness_mm / scale_factor if scale_factor > 0 else 1.0
    
    base = trimesh.creation.box(
        extents=[
            max(size[0], 1e-3),
            max(size[1], 1e-3),
            max(thickness_mesh_units, 1e-6),
        ]
    )
    
    # Align to bottom of bounds
    base.apply_translation([
        (bounds[0][0] + bounds[1][0]) / 2.0,
        (bounds[0][1] + bounds[1][1]) / 2.0,
        bounds[0][2] - base.extents[2] / 2.0, # Put entirely below the mesh
    ])
    
    return base



# ============================================================
# SMART MESH COMBINATION (PREVENTS OVERLAPS)
# ============================================================

def smart_combine_meshes(
    mesh_items: List[Tuple[str, trimesh.Trimesh]],
    aggressive_cleanup: bool = True
) -> trimesh.Trimesh:
    """
    Intelligently combines mesh components with proper Z-ordering and cleanup.
    
    Order of addition (bottom to top):
    1. Terrain (base)
    2. Water (if exists)
    3. Roads (at ground level)
    4. Parks (above roads)
    5. Buildings (highest)
    
    Args:
        mesh_items: List of (name, mesh) tuples
        aggressive_cleanup: Apply final repair and cleanup
    
    Returns:
        Combined watertight mesh
    """
    from services.mesh_quality import detect_nonmanifold_edges
    
    # Organize by component type
    components = {
        'terrain': None,
        'base': None,
        'roads': None,
        'buildings': None,
        'water': None,
        'parks': None,
        'green': None,  # Alias for parks
    }
    
    for name, mesh in mesh_items:
        if mesh is None:
            continue
        key = name.lower()
        if key in components:
            if components[key] is None:
                components[key] = mesh
            else:
                # Merge same types
                components[key] = trimesh.util.concatenate([components[key], mesh])
    
    # Combine in correct Z-order (bottom to top)
    # Track (mesh, is_terrain) to know which ones to skip fill_holes on
    meshes_to_combine = []

    # 1. Terrain/Base (lowest)
    terrain_mesh = components['terrain'] or components['base']
    if terrain_mesh is not None:
        meshes_to_combine.append((terrain_mesh, True))  # is_terrain=True

    # 2. Water (at or below ground)
    if components['water'] is not None:
        meshes_to_combine.append((components['water'], False))

    # 3. Roads (at ground level)
    if components['roads'] is not None:
        meshes_to_combine.append((components['roads'], False))

    # 4. Parks (slightly above ground)
    park_mesh = components['parks'] or components['green']
    if park_mesh is not None:
        meshes_to_combine.append((park_mesh, False))

    # 5. Buildings (highest)
    if components['buildings'] is not None:
        meshes_to_combine.append((components['buildings'], False))

    if not meshes_to_combine:
        return fallback_box()

    # Per-component cleanup before combining
    cleaned = []
    for i, (mesh, is_terrain) in enumerate(meshes_to_combine):
        try:
            # Basic cleanup
            mesh.update_faces(mesh.unique_faces())
            mesh.merge_vertices()
            mesh.remove_unreferenced_vertices()

            # Try to fill holes — but NOT on terrain (grooves would be closed!)
            if not mesh.is_watertight and not is_terrain:
                try:
                    mesh.fill_holes()
                except Exception:
                    pass

            cleaned.append(mesh)
        except Exception as e:
            print(f"[WARN] Component {i} cleanup failed: {e}")
            cleaned.append(mesh)  # Use uncleaned if cleanup fails
    
    # Combine all components
    combined = trimesh.util.concatenate(cleaned)
    
    # Final aggressive cleanup for watertight mesh
    if aggressive_cleanup:
        try:
            print("[INFO] Applying final aggressive cleanup...")
            
            # Remove duplicates
            combined.update_faces(combined.unique_faces())
            combined.merge_vertices(digits_vertex=6)  # 0.001mm precision
            combined.remove_unreferenced_vertices()
            
            # Fix winding order
            try:
                trimesh.repair.fix_winding(combined)
            except Exception:
                pass
            
            # Fill remaining holes
            if not combined.is_watertight:
                try:
                    combined.fill_holes()
                except Exception:
                    pass
            
            # Remove degenerate/broken faces
            try:
                trimesh.repair.broken_faces(combined, color=None)
            except Exception:
                pass
            
            # Final check and report
            nm_count, _ = detect_nonmanifold_edges(combined)
            print(f"[INFO] Final non-manifold edges: {nm_count}")
            
            if combined.is_watertight:
                print("[INFO] Mesh is WATERTIGHT!")
            else:
                print("[WARN] Mesh is not watertight, but improved")
            
        except Exception as e:
            print(f"[WARN] Final cleanup failed: {e}")
    
    return combined

# ============================================================
# CORE PREPARATION (SPLIT AWARE)
# ============================================================

def prepare_scene_parts(
    mesh_items: List[Tuple[str, trimesh.Trimesh]],
    model_size_mm: float,
    add_base: bool,
    base_thickness_mm: float,
    rotate_to_ground: bool,
    reference_xy_m: Optional[Tuple[float, float]],
    preserve_xy: bool,
    preserve_z: bool,
    repair_meshes: bool = True,
) -> Dict[str, trimesh.Trimesh]:

    # 1. Validate and copy inputs
    valid_items = {}
    
    # First pass: collect valid meshes with optional repair
    for name, m in mesh_items:
        tm = to_trimesh(m, repair=repair_meshes)
        if tm:
            tm = _sanitize_mesh_visual(tm) or tm
            # If multiple items have same name, we merge them or append suffix?
            # Standard parts: Base, Roads, Buildings, Water...
            # We'll assume list input is unique-ish or we merge same keys.
            key = name.lower()
            try:
                tm = _normalize_source_part_axes(tm, part_key=key) or tm
            except Exception:
                pass
            if key in valid_items:
                 valid_items[key] = trimesh.util.concatenate([valid_items[key], tm])
            else:
                 valid_items[key] = tm # stored as lowercase keys? or keep original?
                 valid_items[key].metadata["original_name"] = name
    
    if not valid_items:
        fb = fallback_box()
        fb.metadata["original_name"] = "Fallback"
        return {"fallback": fb}

    # 2. Calculate Global Bounds using smart combination
    # ВИПРАВЛЕННЯ: використовуємо smart_combine для правильного порядку компонентів
    all_meshes = list(valid_items.values())
    combined_raw = smart_combine_meshes(
        [(k,  v) for k, v in valid_items.items()],
        aggressive_cleanup=True
    )
    bounds = combined_raw.bounds
    size = bounds[1] - bounds[0]
    
    # 3. Calculate Scale Factor early (needed for base thickness)
    # MODEL_SQUARE_OUTPUT=1 за замовчуванням — рівні X та Y (довжина=ширина)
    # MODEL_SQUARE_OUTPUT=0 — вимкнути (зберегти форму шестикутника)
    square_output = os.environ.get("MODEL_SQUARE_OUTPUT", "1").lower() in ("1", "true", "yes")
    avg_xy = (size[0] + size[1]) / 2.0
    if square_output:
        avg_xy = max(size[0], size[1])
        print(f"[DEBUG] Square output: using ref={avg_xy:.4f}m (max of X,Y)")
    
    # Validation for reference_xy_m
    if reference_xy_m and not square_output:
        ref_avg = (reference_xy_m[0] + reference_xy_m[1]) / 2.0
        if ref_avg > 1e-3: # Must be at least 1mm size
             avg_xy = ref_avg
        else:
             print(f"[WARN] Ignored tiny reference_xy_m: {reference_xy_m}")
    elif reference_xy_m and square_output:
        avg_xy = max(avg_xy, max(reference_xy_m[0], reference_xy_m[1]))

    scale_factor = 1.0
    if avg_xy > 1e-6:
        scale_factor = model_size_mm / avg_xy
        
    # Sanity check for scale factor (prevent astronomical scaling if bounds are tiny)
    # If scale factor is > 10000 (1m -> 10km), something is wrong unless user asked for it.
    if scale_factor > 100000:
        print(f"[WARN] Huge scale factor detected: {scale_factor:.2f}. Clamping to 1.0 to prevent explosion.")
        scale_factor = 1.0 
        # Better to warn and maybe let it proceed if user really has 1nm model?
        # But for 3D map, this is definitely a bug.

    print(f"[DEBUG] Export Transform: Bounds={size}, AvgXY={avg_xy:.4f}, Scale={scale_factor:.6f}")
    
    # 4. Add Base (тільки коли add_base=True — terrain відсутній)
    # Якщо terrain є — він вже має підложку (hex), не додаємо прямокутну навіть при square_output
    if add_base:
        base_bounds = bounds  # Використовуємо фактичні межі контенту (terrain clipped to zone)
        base_geo = create_base_in_mesh_space(base_bounds, base_thickness_mm, scale_factor)
        
        # Find key for 'Base'
        base_key = "base"
        
        if base_key in valid_items:
            valid_items[base_key] = trimesh.util.concatenate([valid_items[base_key], base_geo])
        else:
            # Maybe it's called 'terrain'?
            if "terrain" in valid_items:
                valid_items["terrain"] = trimesh.util.concatenate([valid_items["terrain"], base_geo])
            else:
                # Create new base item - use the name from first item's list or just "base"
                # If we modify keys, we should be consistent.
                # Let's check original names from input list to match casing.
                # The input `mesh_items` usually comes from `export_scene` with keys: "Base", "Roads", etc.
                valid_items["base"] = base_geo
                valid_items["base"].metadata["original_name"] = "Base"

        # Re-calculate bounds using smart_combine (CRITICAL: don't bypass previous smart combination!)
        combined_raw = smart_combine_meshes(
            [(k, v) for k, v in valid_items.items()],
            aggressive_cleanup=True
        )
        bounds = combined_raw.bounds
        size = bounds[1] - bounds[0]  # оновлюємо size після додавання base

    # 5. Get base transform matrix without legacy print rotation.
    # Scene print orientation is resolved later once from the canonical base/terrain
    # reference and then applied to every part consistently.
    matrix = get_transform_matrix(bounds, scale_factor, reference_xy_m, preserve_xy, preserve_z, False)
    
    # 5b. Square output: додаємо padding для рівних X та Y (довжина=ширина)
    if square_output:
        scaled_size = size * scale_factor
        target_size = model_size_mm
        pad_x = (target_size - scaled_size[0]) / 2.0 if scaled_size[0] < target_size - 1e-6 else 0.0
        pad_y = (target_size - scaled_size[1]) / 2.0 if scaled_size[1] < target_size - 1e-6 else 0.0
        if abs(pad_x) > 1e-6 or abs(pad_y) > 1e-6:
            t_pad = np.eye(4)
            t_pad[0, 3] = pad_x
            t_pad[1, 3] = pad_y
            matrix = np.dot(t_pad, matrix)
            print(f"[DEBUG] Square padding: pad_x={pad_x:.2f}mm, pad_y={pad_y:.2f}mm")
    
    # 6. Apply Transform to ALL parts
    transformed_parts = {}
    for key, mesh in valid_items.items():
        safe_mesh = _sanitize_mesh_visual(mesh) or mesh
        try:
            m_trans = safe_mesh.copy()
        except Exception:
            try:
                safe_mesh.visual = trimesh.visual.ColorVisuals()
            except Exception:
                pass
            m_trans = safe_mesh.copy()
        # Зберігаємо кольори перед трансформацією
        has_visual = hasattr(m_trans, 'visual') and m_trans.visual is not None
        visual_backup = None
        if has_visual:
            try:
                visual_backup = m_trans.visual.copy() if hasattr(m_trans.visual, 'copy') else m_trans.visual
            except Exception:
                has_visual = False
        m_trans.apply_transform(matrix)
        # Відновлюємо кольори після трансформації
        if has_visual and hasattr(m_trans, 'visual'):
            try:
                m_trans.visual = visual_backup
            except:
                pass  # Якщо не вдалося відновити, продовжуємо
        
        # Restore original name if possible
        orig_name = mesh.metadata.get("original_name", key)
        transformed_parts[key] = m_trans

    # 6b. Optional auto orientation.
    # Disabled by default because it can rotate geographically-correct models
    # away from the 2D mask orientation users expect in previews.
    # Keep XY orientation consistent with canonical 2D masks by default.
    # Auto-reorientation is now opt-in only via explicit force flag.
    auto_orient = os.environ.get("MODEL_AUTO_REORIENT_PRINT", "0").lower() in ("force", "forced")
    if auto_orient:
        orientation_ref = transformed_parts.get("base") or transformed_parts.get("terrain")
        orientation_matrix = np.eye(4)
        orientation_name = "identity"
        if orientation_ref is not None:
            try:
                orientation_matrix, orientation_name = _select_print_orientation(orientation_ref)
                if not np.allclose(orientation_matrix, np.eye(4)):
                    for mesh in transformed_parts.values():
                        mesh.apply_transform(orientation_matrix)
                    ref_profile = _base_plane_profile(transformed_parts.get("base") or transformed_parts.get("terrain"))
                    if ref_profile:
                        print(
                            f"[EXPORT] Reoriented scene for print: {orientation_name} "
                            f"(bottom_ratio={ref_profile['bottom_ratio']:.3f}, "
                            f"top_ratio={ref_profile['top_ratio']:.3f}, "
                            f"z_extent={ref_profile['z_extent']:.3f})"
                        )
            except Exception:
                pass

    # 7. Ground Z (Common for all)
    # If not preserving Z, we want the LOWEST point of the ENTIRE model to be at Z=0.
    if not preserve_z:
        stabilized_parts, tilt_before_deg = _stabilize_scene_bottom_plane(transformed_parts)
        if stabilized_parts is not transformed_parts:
            transformed_parts = stabilized_parts
            print(f"[EXPORT] Bottom plane stabilized before grounding (tilt={tilt_before_deg:.4f}°)")
        # Check global bounds of transformed meshes using smart combination
        c_trans = smart_combine_meshes(
            [(k, v) for k, v in transformed_parts.items()],
            aggressive_cleanup=False  # Already cleaned before
        )
        min_z = c_trans.bounds[0][2]
        
        # Apply translation to all
        trans_z = [0, 0, -min_z]
        for m in transformed_parts.values():
            m.apply_translation(trans_z)
            # Final cleanup
            m.update_faces(m.nondegenerate_faces())
            # m.remove_unreferenced_vertices() 

    from services.detail_layer_utils import MIN_LAND_WIDTH_MODEL_MM

    for key in ("water", "parks", "green"):
        if key not in transformed_parts:
            continue
        try:
            min_feature_mm = max(float(MICRO_REGION_THRESHOLD_MM), 0.2)
            if key in ("parks", "green"):
                min_feature_mm = max(float(MIN_LAND_WIDTH_MODEL_MM), 0.2)
            transformed_parts[key] = _filter_mesh_components_for_export(
                transformed_parts[key],
                min_feature_mm=min_feature_mm,
                min_area_mm2=0.02,
            )
        except Exception:
            pass

    for key in ("base", "terrain"):
        if key not in transformed_parts:
            continue
        try:
            transformed_parts[key] = repair_base_export_mesh(transformed_parts[key])
        except Exception:
            pass

    # 8. CRITICAL: Final aggressive repair after combination
    # Even if each component is watertight, concatenation can create gaps
    if repair_meshes:
        print(f"\n[FINAL REPAIR] Applying aggressive repair to combined mesh...")
        for key, mesh in transformed_parts.items():
            try:
                # Зберігаємо кольори перед ремонтом
                has_visual = hasattr(mesh, 'visual') and mesh.visual is not None
                visual_backup = None
                if has_visual:
                    try:
                        if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
                            visual_backup = mesh.visual.face_colors.copy()
                    except:
                        pass

                nm_before, _ = detect_nonmanifold_edges(mesh)
                if nm_before > 0:
                    print(f"  [{key}] {nm_before} non-manifold edges detected")
                    # CRITICAL: skip fix_normals for terrain/base — grooved meshes have
                    # non-convex geometry that causes ray-casting heuristic to INVERT normals.
                    # fill_holes() also CLOSES groove openings. Both must be skipped.
                    is_terrain = key in ("base", "terrain")
                    if key in ("roads", "water", "parks", "green"):
                        print(f"  [{key}] Skipping aggressive exporter repair to preserve canonical layer geometry")
                        continue
                    repaired = improve_mesh_for_3d_printing(mesh, aggressive=True, verbose=False, skip_fix_normals=is_terrain)
                    nm_after, _ = detect_nonmanifold_edges(repaired)
                    if nm_after < nm_before:
                        print(f"  [{key}] Fixed {nm_before - nm_after} edges → {nm_after} remaining")
                        # Відновлюємо кольори після ремонту
                        if visual_backup is not None and len(visual_backup) > 0:
                            try:
                                # Перевіряємо, чи кількість граней збігається
                                if len(repaired.faces) == len(visual_backup):
                                    repaired.visual = trimesh.visual.ColorVisuals(face_colors=visual_backup)
                                elif len(repaired.faces) < len(visual_backup):
                                    # Якщо граней менше, беремо перші кольори
                                    repaired.visual = trimesh.visual.ColorVisuals(face_colors=visual_backup[:len(repaired.faces)])
                                else:
                                    # Якщо граней більше, повторюємо останній колір
                                    last_color = visual_backup[-1] if len(visual_backup) > 0 else [150, 150, 150, 255]
                                    extended_colors = np.vstack([
                                        visual_backup,
                                        np.tile(last_color, (len(repaired.faces) - len(visual_backup), 1))
                                    ])
                                    repaired.visual = trimesh.visual.ColorVisuals(face_colors=extended_colors)
                            except Exception as e:
                                print(f"  [{key}] Failed to restore colors after repair: {e}")
                        transformed_parts[key] = repaired
                    else:
                        print(f"  [{key}] No improvement, keeping original")
            except Exception as e:
                print(f"  [{key}] Repair failed: {e}")
        
        print(f"[FINAL REPAIR] Complete\n")

    return transformed_parts


# ============================================================
# SAFE STL EXPORT (BINARY, GUARANTEED VALID)
# ============================================================


def export_stl_safe(mesh: trimesh.Trimesh, filename: str) -> None:
    if mesh.faces is None or len(mesh.faces) == 0:
        return

    # Use the standard export method which returns bytes for STL
    data = mesh.export(file_type="stl")
    
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    with open(filename, "wb") as f:
        f.write(data)



def export_stl(
    filename: str,
    mesh_items: List[Tuple[str, trimesh.Trimesh]],
    model_size_mm: float = 100.0,
    add_flat_base: bool = True,
    base_thickness_mm: float = 2.0,
    rotate_to_ground: bool = False,
    reference_xy_m: Optional[Tuple[float, float]] = None,
    preserve_z: bool = False,
    preserve_xy: bool = False,
    repair_meshes: bool = True,
) -> Dict[str, str]:

    print(f"[STL EXPORT] Starting export with repair_meshes={repair_meshes}")
    
    parts = prepare_scene_parts(
        mesh_items,
        model_size_mm,
        add_flat_base,
        base_thickness_mm,
        rotate_to_ground,
        reference_xy_m,
        preserve_xy,
        preserve_z,
        repair_meshes,
    )

    if "roads" in parts:
        try:
            parts["roads"] = repair_road_export_mesh(parts["roads"])
        except Exception:
            pass
    
    result_files = {}
    
    # 1. Export Main Combined (Legacy behavior + convenience)
    # Merge all parts
    all_meshes = list(parts.values())
    if all_meshes:
        # Strip visuals before concatenation: STL has no color support and mixed
        # ColorVisuals/VertexColorVisuals across meshes causes "face colors incorrect shape".
        for _m in all_meshes:
            try:
                _m.visual = trimesh.visual.ColorVisuals()
            except Exception:
                pass
        try:
            combined = trimesh.util.concatenate(all_meshes)
        except Exception as _exc:
            print(f"[WARN] STL concatenate failed: {_exc}")
            combined = None
        if combined is not None:
            try:
                export_stl_safe(combined, filename)
                result_files["stl"] = filename
            except ValueError:
                pass  # Empty
            
    # 2. Export Parts
    # Filename format: {basename}_{part}.stl
    # e.g. "output/123.stl" -> "output/123_base.stl", "output/123_roads.stl"
    
    base_path = os.path.splitext(filename)[0]
    
    for key, mesh in parts.items():
        # key is usually lowercase like 'base', 'roads'
        part_filename = f"{base_path}_{key}.stl"
        
        # For 'Base', we might want to map it to 'base' (it is already if prepare_scene_parts used keys)
        # Verify the key usage in main: expects "base_stl", "roads_stl"
        # We return dict with keys "base_stl": "path/to/file.stl"
        
        try:
            printable_mesh = _normalize_part_for_print_export(mesh, part_key=key)
            if key in ("base", "terrain"):
                printable_mesh = repair_base_export_mesh_aggressive(printable_mesh)
            elif key == "roads":
                printable_mesh = repair_road_export_mesh(printable_mesh)
            export_stl_safe(printable_mesh, part_filename)
            result_files[key] = part_filename
        except Exception:
            pass
            
    return result_files


# ============================================================
# 3MF EXPORT (SCENE IS OK HERE)
# ============================================================

def export_3mf(
    filename: str,
    mesh_items: List[Tuple[str, trimesh.Trimesh]],
    model_size_mm: float = 100.0,
    add_flat_base: bool = True,
    base_thickness_mm: float = 2.0,
    rotate_to_ground: bool = False,
    reference_xy_m: Optional[Tuple[float, float]] = None,
    preserve_z: bool = False,
    preserve_xy: bool = False,
    repair_meshes: bool = True,
) -> Dict[str, str]:

    print(f"[3MF EXPORT] Starting export with repair_meshes={repair_meshes}")
    
    parts = prepare_scene_parts(
        mesh_items,
        model_size_mm,
        add_flat_base,
        base_thickness_mm,
        rotate_to_ground,
        reference_xy_m,
        preserve_xy,
        preserve_z,
        repair_meshes,
    )

    if "roads" in parts:
        try:
            parts["roads"] = repair_road_export_mesh(parts["roads"])
        except Exception:
            pass

    scene = trimesh.Scene()
    preview_parts = _build_assembly_preview_parts(parts)

    # Add items to scene with names and colors
    for key, mesh in preview_parts.items():
        # Clean name for 3MF metadata?
        # Trimesh handles scene nodes.
        # metadata["original_name"] might have "Base", "Roads"
        name = key
        if hasattr(mesh, 'metadata') and 'original_name' in mesh.metadata:
            name = mesh.metadata['original_name']
        
        # Перевіряємо, чи є кольори в меші
        has_colors = False
        if hasattr(mesh, 'visual') and mesh.visual is not None:
            if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
                if len(mesh.visual.face_colors) > 0:
                    has_colors = True
        
        # Якщо кольорів немає, застосовуємо кольори на основі імені
        if not has_colors:
            color_map = {
                "base": [200, 180, 140, 255],      # Бежевий для рельєфу
                "terrain": [200, 180, 140, 255],  # Бежевий для рельєфу
                "roads": [60, 60, 60, 255],        # Темно-сірий для доріг
                "buildings": [120, 120, 120, 255], # Сірий для будівель
                "water": [100, 150, 200, 255],     # Блакитний для води
                "parks": [100, 150, 100, 255],     # Зелений для парків
                "green": [100, 150, 100, 255],     # Зелений для парків
            }
            
            key_lower = key.lower()
            color = color_map.get(key_lower, [150, 150, 150, 255])  # Сірий за замовчуванням
            
            if len(mesh.faces) > 0:
                face_colors = np.tile(color, (len(mesh.faces), 1))
                mesh.visual = trimesh.visual.ColorVisuals(face_colors=face_colors)
                print(f"[3MF EXPORT] Applied color to {name}: {color[:3]}")
            
        scene.add_geometry(mesh, node_name=name, geom_name=name)

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    scene.export(filename)
    print(f"[3MF EXPORT] Exported scene with {len(parts)} parts to {filename}")

    # 3MF is a single container, so we just return the file
    # But for consistency, maybe we could say we support parts?
    # No, 3MF is one file internally containing parts.
    return {"3mf": filename}


# ============================================================
# PUBLIC API
# ============================================================

def export_scene(
    terrain_mesh: Optional[trimesh.Trimesh],
    road_mesh: Optional[trimesh.Trimesh],
    building_meshes: List[trimesh.Trimesh],
    water_mesh: Optional[trimesh.Trimesh],
    filename: str,
    format: str = "3mf",
    model_size_mm: float = 100.0,
    add_flat_base: bool = True,
    base_thickness_mm: float = 2.0,
    parks_mesh: Optional[trimesh.Trimesh] = None,
    poi_mesh: Optional[trimesh.Trimesh] = None,
    reference_xy_m: Optional[Tuple[float, float]] = None,
    preserve_z: bool = False,
    preserve_xy: bool = False,
    rotate_to_ground: bool = True,  # Default True for vertical orientation
):

    items: List[Tuple[str, trimesh.Trimesh]] = []

    def add(name, mesh):
        tm = to_trimesh(mesh)
        if tm:
            items.append((name, tm))

    add("Base", terrain_mesh)
    add("Roads", road_mesh)
    add("Water", water_mesh)
    add("Parks", parks_mesh)
    add("POI", poi_mesh)

    if building_meshes:
        valid = [to_trimesh(b) for b in building_meshes if to_trimesh(b)]
        if valid:
            items.append(("Buildings", trimesh.util.concatenate(valid)))

    # If no items, fallback
    if not items:
        # Pass empty list, prepare_scene_parts will handle fallback
        pass

    # Застосовуємо кольори для 3MF
    if format.lower() == "3mf":
        color_map = {
            "base": [200, 180, 140, 255],      # Бежевий для рельєфу
            "terrain": [200, 180, 140, 255],  # Бежевий для рельєфу
            "roads": [60, 60, 60, 255],        # Темно-сірий для доріг
            "buildings": [120, 120, 120, 255], # Сірий для будівель
            "water": [100, 150, 200, 255],     # Блакитний для води
            "parks": [100, 150, 100, 255],     # Зелений для парків
            "green": [100, 150, 100, 255],     # Зелений для парків
            "poi": [255, 200, 100, 255],       # Помаранчевий для POI
        }
        
        colored_items = []
        for name, mesh in items:
            if mesh is None:
                continue
            
            # Копіюємо меш для безпеки
            colored_mesh = mesh.copy()
            
            # Застосовуємо колір, якщо він не встановлений
            key = name.lower()
            # Перевіряємо, чи є вже встановлені кольори
            has_colors = False
            if hasattr(colored_mesh.visual, 'face_colors') and colored_mesh.visual.face_colors is not None:
                if len(colored_mesh.visual.face_colors) > 0:
                    # Перевіряємо, чи не всі кольори однакові (білі/сірі за замовчуванням)
                    colors = np.array(colored_mesh.visual.face_colors)
                    if len(colors.shape) == 2 and colors.shape[1] >= 3:
                        # Якщо не всі кольори однакові, вважаємо що кольори вже встановлені
                        unique_colors = np.unique(colors[:, :3], axis=0)
                        if len(unique_colors) > 1:
                            has_colors = True
            
            if not has_colors:
                color = color_map.get(key, [150, 150, 150, 255])  # Сірий за замовчуванням
                if len(colored_mesh.faces) > 0:
                    face_colors = np.tile(color, (len(colored_mesh.faces), 1))
                    colored_mesh.visual = trimesh.visual.ColorVisuals(face_colors=face_colors)
            
            colored_items.append((name, colored_mesh))
        
        items = colored_items

    if format.lower() == "stl":
        return export_stl(
            filename,
            items,
            model_size_mm,
            add_flat_base,
            base_thickness_mm,
            rotate_to_ground,
            reference_xy_m,
            preserve_z,
            preserve_xy,
        )

    if format.lower() == "3mf":
        return export_3mf(
            filename,
            items,
            model_size_mm,
            add_flat_base,
            base_thickness_mm,
            rotate_to_ground,  # True для вертикальної орієнтації
            reference_xy_m,
            preserve_z,
            preserve_xy,
        )

    raise ValueError(f"Unsupported format: {format}")


def export_preview_parts_stl(
    output_prefix: str,
    mesh_items: List[Tuple[str, trimesh.Trimesh]],
    model_size_mm: float = 100.0,
    add_flat_base: bool = True,
    base_thickness_mm: float = 2.0,
    rotate_to_ground: bool = False,
    reference_xy_m: Optional[Tuple[float, float]] = None,
    preserve_z: bool = False,
    preserve_xy: bool = False,
) -> Dict[str, str]:

    out_file = f"{output_prefix}.stl"

    return export_stl(
        out_file,
        mesh_items,
        model_size_mm,
        add_flat_base,
        base_thickness_mm,
        rotate_to_ground,
        reference_xy_m,
        preserve_z,
        preserve_xy,
    )


def export_preview_parts_3mf(
    output_prefix: str,
    mesh_items: List[Tuple[str, trimesh.Trimesh]],
    model_size_mm: float = 100.0,
    add_flat_base: bool = True,
    base_thickness_mm: float = 2.0,
    rotate_to_ground: bool = True,  # Default True for vertical orientation
    reference_xy_m: Optional[Tuple[float, float]] = None,
    preserve_z: bool = False,
    preserve_xy: bool = False,
    include_components: Optional[Dict[str, bool]] = None,  # {"base": True, "roads": True, ...}
) -> Dict[str, str]:
    """
    Експортує прев'ю частини в форматі 3MF з кольорами та можливістю виключення компонентів.
    
    Args:
        include_components: Словник з можливістю виключення компонентів
            {"base": True, "roads": True, "buildings": True, "water": True, "parks": True}
    """
    # Фільтруємо компоненти, якщо вказано
    if include_components is not None:
        filtered_items = []
        for name, mesh in mesh_items:
            key = name.lower()
            # Перевіряємо, чи компонент включений
            if include_components.get(key, True):  # За замовчуванням включено
                filtered_items.append((name, mesh))
        mesh_items = filtered_items
    
    # Застосовуємо кольори до мешів
    color_map = {
        "base": [200, 180, 140, 255],      # Бежевий для рельєфу
        "terrain": [200, 180, 140, 255],  # Бежевий для рельєфу
        "roads": [60, 60, 60, 255],        # Темно-сірий для доріг
        "buildings": [120, 120, 120, 255], # Сірий для будівель
        "water": [100, 150, 200, 255],     # Блакитний для води
        "parks": [100, 150, 100, 255],     # Зелений для парків
        "green": [100, 150, 100, 255],     # Зелений для парків
    }
    
    colored_items = []
    for name, mesh in mesh_items:
        if mesh is None:
            continue

        # Копіюємо меш для безпеки
        safe_mesh = _sanitize_mesh_visual(mesh) or mesh
        try:
            colored_mesh = safe_mesh.copy()
        except Exception:
            colored_mesh = safe_mesh
            colored_mesh = _sanitize_mesh_visual(colored_mesh) or colored_mesh

        # Застосовуємо колір, якщо він не встановлений
        key = name.lower()
        # Перевіряємо, чи є вже встановлені кольори
        has_colors = False
        if hasattr(colored_mesh.visual, 'face_colors') and colored_mesh.visual.face_colors is not None:
            if len(colored_mesh.visual.face_colors) > 0:
                # Перевіряємо, чи не всі кольори однакові (білі/сірі за замовчуванням)
                colors = np.array(colored_mesh.visual.face_colors)
                if len(colors.shape) == 2 and colors.shape[1] >= 3:
                    # Якщо не всі кольори однакові, вважаємо що кольори вже встановлені
                    unique_colors = np.unique(colors[:, :3], axis=0)
                    if len(unique_colors) > 1:
                        has_colors = True
        
        if not has_colors:
            color = color_map.get(key, [150, 150, 150, 255])  # Сірий за замовчуванням
            if len(colored_mesh.faces) > 0:
                face_colors = np.tile(color, (len(colored_mesh.faces), 1))
                colored_mesh.visual = trimesh.visual.ColorVisuals(face_colors=face_colors)
        
        colored_items.append((name, colored_mesh))
    
    out_file = f"{output_prefix}.3mf"

    result = export_3mf(
        out_file,
        colored_items,
        model_size_mm,
        add_flat_base,
        base_thickness_mm,
        rotate_to_ground,  # True для вертикальної орієнтації
        reference_xy_m,
        preserve_z,
        preserve_xy,
    )
    
    # Перейменовуємо ключ з "3mf" на "preview_3mf" для консистентності
    if "3mf" in result:
        result["preview_3mf"] = result.pop("3mf")
    
    return result
