from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from typing import Iterable, List, Optional

import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from services.building_processor import BuildingMeshRecord
from services.detail_layer_utils import MICRO_REGION_THRESHOLD_MM, model_mm_to_world_m


def _polygon_min_dimension(poly: Polygon) -> float:
    try:
        minx, miny, maxx, maxy = poly.bounds
        return float(min(maxx - minx, maxy - miny))
    except Exception:
        return 0.0


def _normalize_support_footprint(
    footprint: Optional[BaseGeometry],
    *,
    inset_m: float = 0.02,
    simplify_m: float = 0.02,
    exclusion_geom: Optional[BaseGeometry] = None,
    min_feature_m: float = 0.0,
) -> list[Polygon]:
    if footprint is None or getattr(footprint, "is_empty", True):
        return []

    try:
        geom = footprint.buffer(0)
    except Exception:
        geom = footprint

    polygons: list[Polygon] = []
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = [poly for poly in geom.geoms if isinstance(poly, Polygon) and not poly.is_empty]
    elif hasattr(geom, "geoms"):
        polygons = [poly for poly in geom.geoms if isinstance(poly, Polygon) and not poly.is_empty]

    normalized: list[Polygon] = []
    for poly in polygons:
        candidate = poly
        if abs(float(inset_m)) > 1e-6:
            try:
                inset_poly = poly.buffer(-float(inset_m), join_style=2)
                if inset_poly is not None and not inset_poly.is_empty and getattr(inset_poly, "area", 0.0) > 0:
                    candidate = inset_poly
                else:
                    continue
            except Exception:
                continue
        allowed_footprint = candidate
        if simplify_m > 0:
            try:
                candidate = candidate.simplify(float(simplify_m), preserve_topology=True)
            except Exception:
                pass
        if exclusion_geom is not None and not getattr(exclusion_geom, "is_empty", True):
            try:
                candidate = candidate.difference(exclusion_geom)
            except Exception:
                pass
        try:
            candidate = candidate.buffer(0)
        except Exception:
            pass
        try:
            if allowed_footprint is not None and not getattr(allowed_footprint, "is_empty", True):
                candidate = candidate.intersection(allowed_footprint)
        except Exception:
            pass
        parts: list[Polygon] = []
        if isinstance(candidate, Polygon) and not candidate.is_empty and candidate.area > 0:
            parts = [candidate]
        elif isinstance(candidate, MultiPolygon):
            parts = [p for p in candidate.geoms if isinstance(p, Polygon) and not p.is_empty and p.area > 0]

        for part in parts:
            if min_feature_m > 0:
                try:
                    min_dim = _polygon_min_dimension(part)
                    area = float(getattr(part, "area", 0.0) or 0.0)
                    min_area_m2 = max(float(min_feature_m) ** 2, 1e-8)
                    if min_dim < float(min_feature_m) or area < min_area_m2:
                        continue
                except Exception:
                    pass
            normalized.append(part)
    return normalized


def _cleanup_support_mesh(
    mesh: Optional[trimesh.Trimesh],
    *,
    min_feature_m: float = 0.0,
) -> Optional[trimesh.Trimesh]:
    if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
        return None

    try:
        mesh = mesh.copy()
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices(digits_vertex=6)
    except Exception:
        pass

    try:
        components = list(mesh.split(only_watertight=False))
    except Exception:
        components = [mesh]

    kept: list[trimesh.Trimesh] = []
    min_area = max(float(min_feature_m) ** 2, 1e-8) if min_feature_m > 0 else 0.0
    for component in components:
        if component is None or component.faces is None or len(component.faces) < 4:
            continue
        try:
            bounds = component.bounds
            extents = bounds[1] - bounds[0]
            if min_feature_m > 0:
                xy_min = float(min(extents[0], extents[1]))
                if xy_min < float(min_feature_m):
                    continue
            if min_area > 0 and float(component.area) < min_area * 2.0:
                continue
        except Exception:
            pass
        kept.append(component)

    if not kept:
        return None

    cleaned = trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0].copy()
    try:
        cleaned.update_faces(cleaned.unique_faces())
        cleaned.update_faces(cleaned.nondegenerate_faces())
        cleaned.remove_unreferenced_vertices()
        cleaned.merge_vertices(digits_vertex=6)
        cleaned.fix_normals()
    except Exception:
        pass
    return cleaned


def build_building_supports(
    building_records: Iterable[BuildingMeshRecord] | None,
    *,
    support_bottom_z: float | None,
    top_overlap_m: float = 0.12,
    footprint_inset_m: float = 0.01,
    exclusion_polygons: Optional[BaseGeometry] = None,
    min_feature_m: float = 0.0,
) -> List[trimesh.Trimesh]:
    if building_records is None or support_bottom_z is None:
        return []

    supports: List[trimesh.Trimesh] = []
    base_z = float(support_bottom_z)

    for record in building_records:
        if record is None or record.mesh is None or len(record.mesh.vertices) == 0:
            continue

        top_z = float(record.base_z) + float(top_overlap_m)
        height = top_z - base_z
        if height <= 1e-4:
            continue

        footprints = _normalize_support_footprint(
            record.footprint,
            inset_m=float(footprint_inset_m),
            simplify_m=min(float(min_feature_m) * 0.2, 0.02) if min_feature_m > 0 else 0.02,
            exclusion_geom=exclusion_polygons,
            min_feature_m=float(min_feature_m),
        )
        if not footprints:
            continue

        for footprint in footprints:
            try:
                support = trimesh.creation.extrude_polygon(footprint, height=height)
            except Exception:
                continue

            if support is None or len(support.vertices) == 0 or len(support.faces) == 0:
                continue

            support.apply_translation([0.0, 0.0, base_z])
            support = _cleanup_support_mesh(support, min_feature_m=float(min_feature_m))
            if support is None or len(support.vertices) == 0 or len(support.faces) == 0:
                continue
            supports.append(support)

    return supports


def merge_building_and_support_meshes(
    building_meshes: Iterable[trimesh.Trimesh] | None,
    support_meshes: Iterable[trimesh.Trimesh] | None,
) -> List[trimesh.Trimesh]:
    merged: List[trimesh.Trimesh] = []
    if building_meshes:
        merged.extend([mesh for mesh in building_meshes if mesh is not None])
    if support_meshes:
        merged.extend([mesh for mesh in support_meshes if mesh is not None])
    return merged


def _find_blender() -> Optional[str]:
    blender_exe = shutil.which("blender")
    if blender_exe:
        return blender_exe
    custom_path = r"D:\Soft\Blender\blender.exe"
    if os.path.exists(custom_path):
        return custom_path
    return None


def _run_blender_union(meshes: list[trimesh.Trimesh], *, label: str) -> Optional[trimesh.Trimesh]:
    blender_exe = _find_blender()
    if blender_exe is None or len(meshes) < 2:
        return None

    session_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join(os.getcwd(), f"temp_building_union_{session_id}")
    os.makedirs(temp_dir, exist_ok=True)

    mesh_paths = []
    result_path = os.path.abspath(os.path.join(temp_dir, "result.obj"))
    script_path = os.path.abspath(os.path.join(temp_dir, "union.py"))

    try:
        for idx, mesh in enumerate(meshes):
            mesh_path = os.path.abspath(os.path.join(temp_dir, f"mesh_{idx}.obj"))
            mesh.export(mesh_path)
            mesh_paths.append(mesh_path)

        imports = []
        imports.append(
            f"""
try:
    bpy.ops.wm.obj_import(filepath=r\"{mesh_paths[0]}\")
except:
    bpy.ops.import_scene.obj(filepath=r\"{mesh_paths[0]}\")
base = bpy.context.selected_objects[0]
base.name = "BaseUnion"
bpy.ops.object.select_all(action='DESELECT')
"""
        )
        for idx, mesh_path in enumerate(mesh_paths[1:], start=1):
            imports.append(
                f"""
try:
    bpy.ops.wm.obj_import(filepath=r\"{mesh_path}\")
except:
    bpy.ops.import_scene.obj(filepath=r\"{mesh_path}\")
cutter = bpy.context.selected_objects[0]
cutter.name = "UnionPart{idx}"
bpy.ops.object.select_all(action='DESELECT')
bpy.context.view_layer.objects.active = base
bpy.ops.object.modifier_add(type='BOOLEAN')
mod = base.modifiers[-1]
mod.object = cutter
mod.operation = 'UNION'
mod.solver = 'EXACT'
bpy.ops.object.modifier_apply(modifier=mod.name)
bpy.data.objects.remove(cutter, do_unlink=True)
"""
            )

        blender_script = f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
{''.join(imports)}
bpy.context.view_layer.objects.active = base
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')
try:
    bpy.ops.wm.obj_export(filepath=r\"{result_path}\", export_selected_objects=True)
except:
    bpy.ops.export_scene.obj(filepath=r\"{result_path}\", use_selection=True)
print("UNION_SUCCESS")
"""
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(blender_script)

        proc = subprocess.run(
            [blender_exe, "--background", "--python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
        stdout_text = proc.stdout.decode("utf-8", errors="replace")
        if "UNION_SUCCESS" not in stdout_text or not os.path.exists(result_path):
            return None

        result_mesh = trimesh.load(result_path)
        if isinstance(result_mesh, trimesh.Scene):
            geoms = list(result_mesh.geometry.values())
            if not geoms:
                return None
            result_mesh = max(geoms, key=lambda m: len(m.vertices))
        return result_mesh if result_mesh is not None and len(result_mesh.vertices) > 0 else None
    except Exception:
        return None
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def union_building_layer_meshes(
    building_meshes: Iterable[trimesh.Trimesh] | None,
    support_meshes: Iterable[trimesh.Trimesh] | None,
    *,
    label: str = "buildings",
) -> Optional[trimesh.Trimesh]:
    building_mesh = concatenate_meshes(building_meshes)
    support_mesh = concatenate_meshes(support_meshes)
    if building_mesh is None and support_mesh is None:
        return None
    if building_mesh is None:
        return support_mesh
    if support_mesh is None:
        return building_mesh

    try:
        result = trimesh.boolean.union([building_mesh, support_mesh], engine="manifold")
        if isinstance(result, list):
            result = trimesh.util.concatenate([m for m in result if m is not None])
        if result is not None and len(result.vertices) > 0:
            return result
    except Exception:
        pass

    blender_result = _run_blender_union([building_mesh, support_mesh], label=label)
    if blender_result is not None and len(blender_result.vertices) > 0:
        return blender_result

    return concatenate_meshes([building_mesh, support_mesh])


def union_mesh_collection(
    meshes: Iterable[trimesh.Trimesh] | None,
    *,
    label: str = "mesh_collection",
) -> Optional[trimesh.Trimesh]:
    valid = [mesh for mesh in (meshes or []) if mesh is not None and len(mesh.vertices) > 0]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]

    try:
        result = trimesh.boolean.union(valid, engine="manifold")
        if isinstance(result, list):
            result = trimesh.util.concatenate([m for m in result if m is not None])
        if result is not None and len(result.vertices) > 0:
            return result
    except Exception:
        pass

    blender_result = _run_blender_union(valid, label=label)
    if blender_result is not None and len(blender_result.vertices) > 0:
        return blender_result

    return concatenate_meshes(valid)


def concatenate_meshes(meshes: Iterable[trimesh.Trimesh] | None) -> Optional[trimesh.Trimesh]:
    valid = [mesh for mesh in (meshes or []) if mesh is not None and len(mesh.vertices) > 0]
    if not valid:
        return None
    try:
        return trimesh.util.concatenate(valid)
    except Exception:
        return None
