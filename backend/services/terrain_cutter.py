"""
Terrain Cutter — вирізання пазів (grooves) у рельєфі для вставки мешів (дороги, парки, вода).

Використовує Blender Boolean DIFFERENCE для створення точних пазів.
Кожна деталь (дорога, парк) вставляється в свій паз як у пазл.
"""

import time
import trimesh
import shutil
import numpy as np
import os
import uuid
import subprocess
from typing import Optional, Tuple, List
from shapely.geometry.base import BaseGeometry
from shapely.geometry import MultiPoint, Polygon as ShapelyPolygon
from shapely.ops import unary_union


# ---------------------------------------------------------------------------
# Допоміжні функції
# ---------------------------------------------------------------------------

def manifold_repair_subtract(terrain_mesh, cutter_mesh):
    """РОБАСТНИЙ виріз: пропускає НЕгерметичний терен через manifold3d-конструктор
    (топологічний РЕМОНТ у валідний том — чого fill_holes/blender НЕ вміли), тоді
    boolean-різниця. Доведено на щільній юзер-зоні: 290138 граней wt=False →
    manifold3d → wt=True → виріз → wt=True (паз врізано чисто, без нівечення).
    Повертає trimesh.Trimesh або None. Це ГОЛОВНИЙ шлях різу конектор-паза —
    працює навіть коли trimesh manifold-engine відмовляє («not all volumes»)."""
    try:
        from manifold3d import Manifold, Mesh as _MMesh
        from trimesh.grouping import group_rows as _grp
        def _open_edges(m):
            try:
                return len(_grp(m.edges_sorted, require_count=1))
            except Exception:
                return -1
        def _to_man(m):
            _mm = _MMesh(vert_properties=np.asarray(m.vertices, dtype=np.float32),
                         tri_verts=np.asarray(m.faces, dtype=np.uint32))
            return Manifold(_mm)
        # ЗВАРЮВАННЯ ЗАЗОРІВ — КРИТИЧНИЙ КРОК. Рельєф після грувів має жменю
        # coincident-вершинних зазорів (юзер-zone_1: лише 6 відкритих ребер), через
        # які manifold3d кидає NotManifold і виріз падав ТИЖДЕНЬ. merge_vertices на
        # правильній точності зводить дублі → 0 відкритих ребер (лишається тільки
        # winding-дефект, який manifold3d ремонтує сам). Пробуємо кілька рівнів
        # точності й беремо перший, що дає герметичну топологію.
        _terr = terrain_mesh
        if _open_edges(_terr) != 0:
            for _dg in (6, 5, 4, 7):
                _w = terrain_mesh.copy()
                try:
                    _w.merge_vertices(digits_vertex=_dg)
                except Exception:
                    continue
                if _open_edges(_w) == 0:
                    _terr = _w
                    break
        _t = _to_man(_terr)
        if _t.is_empty() or _t.num_tri() == 0:
            return None
        _c = _to_man(cutter_mesh)
        _res = _t - _c
        if _res.is_empty() or _res.num_tri() == 0:
            return None
        _mo = _res.to_mesh()
        _out = trimesh.Trimesh(
            vertices=np.asarray(_mo.vert_properties)[:, :3].astype(np.float64),
            faces=np.asarray(_mo.tri_verts), process=False)
        return _out if len(_out.faces) > 0 else None
    except Exception as _e:  # noqa: BLE001
        print(f"[MANIFOLD-CUT] repair-subtract failed: {_e}")
        return None


def _find_blender() -> Optional[str]:
    """Знаходить шлях до Blender."""
    blender_exe = shutil.which("blender")
    if blender_exe:
        return blender_exe
    custom_path = r"D:\Soft\Blender\blender.exe"
    if os.path.exists(custom_path):
        return custom_path
    return None


def _run_blender_boolean(
    terrain_mesh: trimesh.Trimesh,
    cutter_mesh: trimesh.Trimesh,
    label: str = "mesh",
    clearance_m: float = 0.0,
    pre_seal: bool = False,
) -> Optional[trimesh.Trimesh]:
    """
    Запускає Blender Boolean DIFFERENCE: terrain - cutter.
    
    Спільна функція для всіх типів пазів (дороги, парки, вода).
    Логує stderr Blender для діагностики.
    """
    blender_exe = _find_blender()
    if blender_exe is None:
        print(f"[TERRAIN CUT] WARNING: Blender not found. Skipping {label} grooves.")
        return terrain_mesh

    # Унікальна temp dir для кожного виклику (уникає колізій)
    session_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join(os.getcwd(), f"temp_boolean_{session_id}")
    os.makedirs(temp_dir, exist_ok=True)

    terrain_path = os.path.abspath(os.path.join(temp_dir, "terrain.obj"))
    cutter_path = os.path.abspath(os.path.join(temp_dir, "cutter.obj"))
    result_path = os.path.abspath(os.path.join(temp_dir, "result.stl"))
    script_path = os.path.abspath(os.path.join(temp_dir, "boolean.py"))

    try:
        # НЕ викликаємо fix_normals() — Blender сам нормалізує нормалі у скрипті.
        # fix_normals() trimesh використовує ray casting heuristic який інвертує
        # нормалі на складних не-convex мешах (кутери з дірками, рельєф з пазами).

        terrain_mesh.export(terrain_path)
        cutter_mesh.export(cutter_path)

        # PRE-SEAL лише для КОНЕКТОР-паза (pre_seal=True). Для доріг/парків/all_grooves
        # fill_holes ЗАЛИВ БИ грувы — тому там pre_seal=False (порожній блок).
        _seal = ("""
bpy.context.view_layer.objects.active = terrain
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=2e-4)
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_non_manifold()
bpy.ops.mesh.fill_holes(sides=0)
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_non_manifold()
bpy.ops.mesh.edge_face_add()
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')
""") if pre_seal else ""

        blender_script = f"""
import bpy
import sys

bpy.ops.wm.read_factory_settings(use_empty=True)

# Import Terrain
try:
    bpy.ops.wm.obj_import(filepath=r"{terrain_path}")
except:
    bpy.ops.import_scene.obj(filepath=r"{terrain_path}")
terrain = bpy.context.selected_objects[0]
terrain.name = "Terrain"
bpy.ops.object.select_all(action='DESELECT')

# Import Cutter
try:
    bpy.ops.wm.obj_import(filepath=r"{cutter_path}")
except:
    bpy.ops.import_scene.obj(filepath=r"{cutter_path}")
cutter = bpy.context.selected_objects[0]
cutter.name = "Cutter"
bpy.ops.object.select_all(action='DESELECT')

# Fix normals for both
for obj in [terrain, cutter]:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

# PRE-SEAL (лише конектор-паз): зашити дірки ДО EXACT-різу → не нівечить негерметичний
# меш (openEdges 18→514 РАНІШЕ). Для грувів порожньо (fill_holes залив би жолоби).
{_seal}
# Apply Displace for clearance (Minkowski sum expansion)
if {clearance_m} > 0.0:
    bpy.context.view_layer.objects.active = cutter
    bpy.ops.object.modifier_add(type='DISPLACE')
    mod_disp = cutter.modifiers[-1]
    mod_disp.strength = {clearance_m}
    bpy.ops.object.modifier_apply(modifier=mod_disp.name)

# Boolean DIFFERENCE
bpy.context.view_layer.objects.active = terrain
bpy.ops.object.modifier_add(type='BOOLEAN')
mod = terrain.modifiers[-1]
mod.object = cutter
mod.operation = 'DIFFERENCE'
mod.solver = 'EXACT'
bpy.ops.object.modifier_apply(modifier=mod.name)

# Post-cleanup
bpy.context.view_layer.objects.active = terrain
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=0.0001)
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')

# Remove cutter
bpy.data.objects.remove(cutter, do_unlink=True)

# Export result
bpy.ops.object.select_all(action='DESELECT')
terrain.select_set(True)
bpy.context.view_layer.objects.active = terrain
try:
    bpy.ops.wm.stl_export(filepath=r"{result_path}", export_selected_objects=True)
except:
    bpy.ops.export_mesh.stl(filepath=r"{result_path}", use_selection=True)

print("BOOLEAN_SUCCESS")
"""
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(blender_script)

        print(f"[TERRAIN CUT] Running Blender for {label}...")
        t0 = time.time()
        cmd = [blender_exe, "--background", "--python", script_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        elapsed = time.time() - t0

        # Логуємо stderr для діагностики
        stderr_text = proc.stderr.decode("utf-8", errors="replace").strip()
        stdout_text = proc.stdout.decode("utf-8", errors="replace").strip()
        
        if stderr_text:
            # Фільтруємо шум Blender (AL lib, mem stats) — лише реальні помилки
            real_errors = [
                line for line in stderr_text.split("\n")
                if not any(skip in line for skip in ["AL lib", "Read prefs", "found bundled", "ALSA", "memleak"])
            ]
            if real_errors:
                print(f"[TERRAIN CUT] Blender stderr ({label}):")
                for line in real_errors[:10]:  # Перші 10 рядків
                    print(f"  {line}")

        if "BOOLEAN_SUCCESS" not in stdout_text:
            print(f"[TERRAIN CUT] WARNING: Blender did not report success for {label}")

        # Завантажуємо результат
        if os.path.exists(result_path):
            result_mesh = trimesh.load(result_path, file_type="stl", force="mesh")

            if isinstance(result_mesh, trimesh.Scene):
                geoms = list(result_mesh.geometry.values())
                if geoms:
                    result_mesh = max(geoms, key=lambda m: len(m.vertices))
                else:
                    print(f"[TERRAIN CUT] WARNING: {label}: Blender returned empty scene")
                    return terrain_mesh

            # AUTO-FIX осьового свопу: Blender 4.x wm.obj_import робить Y-up→Z-up
            # ротацію, яку STL-export не скасовує → результат повернутий (Y↔Z; саме
            # це ламало груви ЛОКАЛЬНО на 4.3.2). Детект: чи -90° навколо X наближає
            # bounds результату до вхідного terrain. На проді (не свопнуто) — no-op.
            try:
                import numpy as _np
                _tb = _np.array(terrain_mesh.bounds)
                def _bdiff(_m):
                    return float(_np.max(_np.abs(_np.array(_m.bounds) - _tb)))
                _d0 = _bdiff(result_mesh)
                if _d0 > 5.0:
                    _Rx = trimesh.transformations.rotation_matrix(-_np.pi / 2.0, [1, 0, 0])
                    _rot = result_mesh.copy()
                    _rot.apply_transform(_Rx)
                    if _bdiff(_rot) < _d0 * 0.5:
                        result_mesh = _rot
                        print(f"[TERRAIN CUT] {label}: fixed Blender Y/Z axis-swap (rot -90 X)")
            except Exception:
                pass

            if len(result_mesh.vertices) >= 4:
                # КРИТИЧНО: НЕ викликаємо fix_normals() на результаті Boolean!
                # Ray casting heuristic інвертує нормалі на нековпуклих мешах з пазами.
                # Blender вже робить normals_make_consistent() у скрипті.
                print(f"[TERRAIN CUT] OK {label} grooves cut: {len(result_mesh.vertices)} verts ({elapsed:.1f}s)")
                return result_mesh
            else:
                print(f"[TERRAIN CUT] WARNING: {label}: result mesh too small ({len(result_mesh.vertices)} verts)")
        else:
            print(f"[TERRAIN CUT] FAIL: {label}: no result file created")

    except subprocess.TimeoutExpired:
        print(f"[TERRAIN CUT] FAIL: {label}: Blender timeout (300s)")
    except Exception as e:
        print(f"[TERRAIN CUT] Error cutting {label}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception:
            pass

    return terrain_mesh


def _mesh_component_count(mesh: Optional[trimesh.Trimesh]) -> int:
    if mesh is None or len(mesh.vertices) == 0:
        return 0
    try:
        return len(list(mesh.split(only_watertight=False)))
    except Exception:
        return 0


def _boundary_edge_count(mesh: Optional[trimesh.Trimesh]) -> int:
    if mesh is None or len(mesh.vertices) == 0:
        return 10**9
    try:
        edges = mesh.edges_sorted
        if len(edges) == 0:
            return 0
        unique = trimesh.grouping.group_rows(edges, require_count=1)
        return int(len(unique))
    except Exception:
        return 10**9


def _xy_bounds_delta(lhs: Optional[trimesh.Trimesh], rhs: Optional[trimesh.Trimesh]) -> float:
    if lhs is None or rhs is None:
        return float("inf")
    try:
        lhs_bounds = np.asarray(lhs.bounds, dtype=float)[:, :2]
        rhs_bounds = np.asarray(rhs.bounds, dtype=float)[:, :2]
        delta = float(np.max(np.abs(lhs_bounds - rhs_bounds)))
        if not np.isfinite(delta):
            return float("inf")
        return delta
    except Exception:
        return float("inf")


def _z_bounds_delta(lhs: Optional[trimesh.Trimesh], rhs: Optional[trimesh.Trimesh]) -> float:
    if lhs is None or rhs is None:
        return float("inf")
    try:
        lhs_bounds = np.asarray(lhs.bounds, dtype=float)[:, 2]
        rhs_bounds = np.asarray(rhs.bounds, dtype=float)[:, 2]
        delta = float(np.max(np.abs(lhs_bounds - rhs_bounds)))
        if not np.isfinite(delta):
            return float("inf")
        return delta
    except Exception:
        return float("inf")


def _z_span_ratio(lhs: Optional[trimesh.Trimesh], rhs: Optional[trimesh.Trimesh]) -> float:
    if lhs is None or rhs is None:
        return float("inf")
    try:
        lhs_bounds = np.asarray(lhs.bounds, dtype=float)[:, 2]
        rhs_bounds = np.asarray(rhs.bounds, dtype=float)[:, 2]
        lhs_span = float(lhs_bounds[1] - lhs_bounds[0])
        rhs_span = float(rhs_bounds[1] - rhs_bounds[0])
        if lhs_span <= 0.0 or rhs_span <= 0.0:
            return float("inf")
        ratio = max(rhs_span / lhs_span, lhs_span / rhs_span)
        if not np.isfinite(ratio):
            return float("inf")
        return float(ratio)
    except Exception:
        return float("inf")


def _dominant_component(mesh: Optional[trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
    if mesh is None or len(mesh.vertices) == 0:
        return None
    try:
        components = list(mesh.split(only_watertight=False))
    except Exception:
        return None
    if not components:
        return None
    return max(components, key=lambda item: len(item.faces) if item is not None and item.faces is not None else 0)


def _boolean_candidate_score(mesh: Optional[trimesh.Trimesh]) -> tuple:
    if mesh is None or len(mesh.vertices) == 0:
        return (-1, -10**9, -10**9, -1)
    return (
        1 if bool(getattr(mesh, "is_watertight", False)) else 0,
        -_boundary_edge_count(mesh),
        -_mesh_component_count(mesh),
        len(mesh.faces) if mesh.faces is not None else -1,
    )


def _cleanup_boolean_candidate(mesh: Optional[trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
    if mesh is None or len(mesh.vertices) == 0:
        return mesh

    original = mesh.copy()
    candidate = mesh.copy()
    try:
        candidate.merge_vertices()
        candidate.update_faces(candidate.unique_faces())
        candidate.remove_unreferenced_vertices()
    except Exception:
        return original

    # `nondegenerate_faces()` can remove a tiny number of faces and turn an
    # already watertight boolean result into a non-watertight one. Only keep
    # that extra cleanup if it does not regress topology.
    try:
        extra_clean = candidate.copy()
        extra_clean.update_faces(extra_clean.nondegenerate_faces())
        extra_clean.remove_unreferenced_vertices()
        if _boolean_candidate_score(extra_clean) >= _boolean_candidate_score(candidate):
            candidate = extra_clean
    except Exception:
        pass

    if _boolean_candidate_score(candidate) >= _boolean_candidate_score(original):
        return candidate
    return original


def _accept_boolean_stage_result(
    *,
    previous_mesh: trimesh.Trimesh,
    candidate_mesh: Optional[trimesh.Trimesh],
    label: str,
) -> trimesh.Trimesh:
    if candidate_mesh is None or len(candidate_mesh.vertices) == 0:
        print(f"[TERRAIN CUT] WARNING: {label}: boolean returned empty mesh, keeping previous terrain")
        return previous_mesh

    candidate_mesh = _cleanup_boolean_candidate(candidate_mesh)

    if bool(getattr(candidate_mesh, "is_watertight", False)):
        return candidate_mesh

    candidate_components = _mesh_component_count(candidate_mesh)
    candidate_boundary_edges = _boundary_edge_count(candidate_mesh)
    candidate_xy_drift = _xy_bounds_delta(previous_mesh, candidate_mesh)
    candidate_z_drift = _z_bounds_delta(previous_mesh, candidate_mesh)
    candidate_z_span_ratio = _z_span_ratio(previous_mesh, candidate_mesh)
    if (
        candidate_components <= 2
        and candidate_boundary_edges <= 64
        and candidate_xy_drift <= 0.25
        and candidate_z_drift <= 2.0
        and candidate_z_span_ratio <= 3.0
    ):
        print(
            f"[TERRAIN CUT] {label}: accepted near-watertight boolean result "
            f"(components={candidate_components}, boundary_edges={candidate_boundary_edges}, "
            f"xy_drift={candidate_xy_drift:.6f}m, z_drift={candidate_z_drift:.6f}m, "
            f"z_span_ratio={candidate_z_span_ratio:.3f})"
        )
        return candidate_mesh

    dominant = _dominant_component(candidate_mesh)
    if dominant is not None:
        try:
            total_faces = sum(
                len(component.faces)
                for component in candidate_mesh.split(only_watertight=False)
                if component is not None and component.faces is not None
            )
        except Exception:
            total_faces = len(candidate_mesh.faces)
        dominant_faces = len(dominant.faces) if dominant.faces is not None else 0
        extra_ratio = float(total_faces - dominant_faces) / float(total_faces) if total_faces > 0 else 1.0
        xy_drift = _xy_bounds_delta(previous_mesh, dominant)
        z_drift = _z_bounds_delta(previous_mesh, dominant)
        z_span_ratio = _z_span_ratio(previous_mesh, dominant)
        if (
            bool(getattr(dominant, "is_watertight", False))
            and extra_ratio <= 0.02
            and xy_drift <= 0.25
            and z_drift <= 2.0
            and z_span_ratio <= 3.0
        ):
            print(
                f"[TERRAIN CUT] {label}: accepted dominant watertight component "
                f"(extra_ratio={extra_ratio:.4f}, xy_drift={xy_drift:.6f}m, "
                f"z_drift={z_drift:.6f}m, z_span_ratio={z_span_ratio:.3f})"
            )
            return dominant
        # Practical fallback: some Blender booleans produce tiny detached shards
        # and leave the dominant terrain shell non-watertight. Keep the dominant
        # component when it preserves geometry and only sheds a small face share.
        dominant_boundary_edges = _boundary_edge_count(dominant)
        if (
            dominant_faces > 0
            and extra_ratio <= 0.08
            and xy_drift <= 0.25
            and z_drift <= 2.0
            and z_span_ratio <= 3.0
            and dominant_boundary_edges <= 512
        ):
            print(
                f"[TERRAIN CUT] {label}: accepted dominant near-watertight component "
                f"(extra_ratio={extra_ratio:.4f}, boundary_edges={dominant_boundary_edges}, "
                f"xy_drift={xy_drift:.6f}m, z_drift={z_drift:.6f}m, "
                f"z_span_ratio={z_span_ratio:.3f})"
            )
            return dominant

    print(
        f"[TERRAIN CUT] WARNING: {label}: result remained non-watertight "
        f"(components={_mesh_component_count(candidate_mesh)}, "
        f"xy_drift={candidate_xy_drift:.6f}m, z_drift={candidate_z_drift:.6f}m, "
        f"z_span_ratio={candidate_z_span_ratio:.3f}), keeping previous terrain"
    )
    return previous_mesh


def _extract_2d_footprint(mesh: trimesh.Trimesh) -> Optional[ShapelyPolygon]:
    """
    Витягує 2D footprint (проекцію на XY) з 3D мешу.
    
    Використовує alpha shape для точного контуру складних форм парків.
    Fallback на convex hull якщо alpha shape не працює.
    """
    if mesh is None or len(mesh.vertices) == 0:
        return None
    
    xy_points = mesh.vertices[:, :2]
    
    # Спочатку пробуємо через trimesh (alpha shape)
    try:
        # outline() дає контур мешу в 2D (проекція)
        outline = mesh.outline()
        if outline is not None:
            # Конвертуємо Path2D/Path3D у полігон
            polys = []
            for entity in outline.entities:
                pts = outline.vertices[entity.points][:, :2]
                if len(pts) >= 3:
                    try:
                        p = ShapelyPolygon(pts)
                        if p.is_valid and not p.is_empty:
                            polys.append(p)
                    except Exception:
                        pass
            if polys:
                result = unary_union(polys)
                if result and not result.is_empty:
                    return result
    except Exception:
        pass
    
    # Fallback: convex hull через Shapely
    try:
        mp = MultiPoint(xy_points.tolist())
        hull = mp.convex_hull
        if hull and not hull.is_empty and hull.geom_type == "Polygon":
            return hull
    except Exception:
        pass
    
    return None


# ---------------------------------------------------------------------------
# Extend bottom functions
# ---------------------------------------------------------------------------

def extend_road_mesh_to_uniform_bottom(road_mesh: "trimesh.Trimesh") -> None:
    """
    Продовжує дно доріг вниз до найнижчої точки мешу.
    Усі пази мають однакову висоту (плоске дно на road_min_z).
    Вершини нижніх граней опускаються до road_min_z.
    """
    if road_mesh is None or len(road_mesh.vertices) == 0:
        return
    road_min_z = float(road_mesh.vertices[:, 2].min())
    faces = road_mesh.faces
    vertices = road_mesh.vertices.copy()
    norms = road_mesh.face_normals
    bottom_face_mask = norms[:, 2] < -0.5
    bottom_vertex_indices = np.unique(faces[bottom_face_mask].ravel())
    vertices[bottom_vertex_indices, 2] = road_min_z
    road_mesh.vertices = vertices
    road_mesh.fix_normals()


def extend_buildings_mesh_to_uniform_bottom(
    building_meshes: Optional[List["trimesh.Trimesh"]],
    target_z: float,
) -> None:
    """Опускає нижні грані КОЖНОЇ будівлі до спільного `target_z`.

    Аналогічно `extend_road_mesh_to_uniform_bottom`: знаходимо downward-facing
    faces (normal.z < -0.5) і переносимо їх vertices у `target_z`. Бокові
    стіни автоматично стають довшими — building solid тепер з'єднується з
    підложкою terrain, перекриття/overhang з рельєфом зникає (саме те, на що
    скаржиться Bambu Studio: "плаваюча консоль" біля будівель на схилах).

    Виклик у pipeline ПІСЛЯ `process_buildings(...)` коли terrain_mesh готовий:
        target_z = float(terrain_mesh.bounds[0][2])  # дно підложки
        extend_buildings_mesh_to_uniform_bottom(building_meshes, target_z=target_z)
    """
    if not building_meshes:
        return
    for mesh in building_meshes:
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue
        try:
            faces = mesh.faces
            vertices = mesh.vertices.copy()
            norms = mesh.face_normals
            bottom_face_mask = norms[:, 2] < -0.5
            if not np.any(bottom_face_mask):
                continue
            bottom_vertex_indices = np.unique(faces[bottom_face_mask].ravel())
            vertices[bottom_vertex_indices, 2] = float(target_z)
            mesh.vertices = vertices
            try:
                mesh.fix_normals()
            except Exception:
                pass
        except Exception as e:
            print(f"[WARN] extend_buildings_mesh_to_uniform_bottom failed for one mesh: {e}")
            continue


def flatten_inlay_to_terrain_decal(
    inlay_mesh: Optional["trimesh.Trimesh"],
    terrain_provider,
    offset_m: float = 0.05,
) -> Optional["trimesh.Trimesh"]:
    """Convert a 3D inlay (road/park/water) into a thin coloured decal hugging
    the terrain surface — for the FAST PREVIEW path.

    Approach: extract the inlay's 2D footprint (its highest-z vertices per
    XY position), keep the triangulation between those vertices, then drape
    them onto terrain_provider's surface at `terrain_height + offset_m`.

    Why per-vertex max-Z instead of "top face by normal":
      The previous version filtered faces by normal.z > 0.5 which discarded
      top faces sitting on slopes (their normal tilts as the road follows
      the hill). After draping every vertex to terrain anyway, the precise
      original Z doesn't matter — we just need ONE vertex per XY position.

    Result: a single-layer triangulated sheet sitting strictly on terrain,
    no side walls, no embedded bottom, no slivers possible.
    """
    if inlay_mesh is None or len(inlay_mesh.vertices) == 0 or len(inlay_mesh.faces) == 0:
        return inlay_mesh
    try:
        V = np.asarray(inlay_mesh.vertices, dtype=float)
        F = np.asarray(inlay_mesh.faces, dtype=np.int64)

        # Step 1: collapse duplicate XY columns — keep the vertex with the
        # highest Z (= top surface). This effectively flattens stacks of
        # top/bottom verts at the same (x,y) into one decal vertex.
        # Quantize to 1mm in model space to merge ~equal coordinates.
        keys = np.round(V[:, :2], 3)
        # Use structured-array trick for unique-by-row + which-index
        view = np.ascontiguousarray(keys).view(
            np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))
        )
        _, unique_inv = np.unique(view, return_inverse=True)
        n_unique = int(unique_inv.max() + 1) if len(unique_inv) else 0
        if n_unique == 0:
            return inlay_mesh
        new_verts = np.zeros((n_unique, 3), dtype=float)
        # representative XY = average of duplicates (close enough since
        # quantized to 1mm), Z = max over duplicates
        np.add.at(new_verts[:, :2], unique_inv, V[:, :2])
        counts = np.bincount(unique_inv, minlength=n_unique).astype(float)
        counts[counts == 0] = 1.0
        new_verts[:, :2] = new_verts[:, :2] / counts[:, None]
        z_max = np.full(n_unique, -np.inf, dtype=float)
        np.maximum.at(z_max, unique_inv, V[:, 2])
        new_verts[:, 2] = z_max

        # Step 2: remap faces, drop degenerate ones (where two or more
        # corners collapsed to the same XY).
        new_F = unique_inv[F]
        non_deg = (
            (new_F[:, 0] != new_F[:, 1]) &
            (new_F[:, 1] != new_F[:, 2]) &
            (new_F[:, 0] != new_F[:, 2])
        )
        new_F = new_F[non_deg]
        if len(new_F) == 0:
            return inlay_mesh

        # Step 3: drape every vertex to terrain + offset. After this Z values
        # come from the terrain provider, not the original inlay solid.
        if terrain_provider is not None and len(new_verts) > 0:
            try:
                heights = terrain_provider.get_surface_heights_for_points(new_verts[:, :2])
                if heights is not None and len(heights) == len(new_verts):
                    heights = np.where(np.isfinite(heights), heights, new_verts[:, 2])
                    new_verts[:, 2] = heights + float(offset_m)
            except Exception as e:
                print(f"[WARN] flatten_inlay_to_terrain_decal: draping failed: {e}")

        return trimesh.Trimesh(vertices=new_verts, faces=new_F, process=False)
    except Exception as e:
        print(f"[WARN] flatten_inlay_to_terrain_decal failed: {e}")
        import traceback
        traceback.print_exc()
        return inlay_mesh


def build_terrain_decal_from_2d_mask(
    mask_geometry,
    terrain_provider,
    offset_m: float = 0.75,
    target_edge_len_m: float = 4.0,
    simplify_tolerance_m: float = 0.0,
) -> Optional["trimesh.Trimesh"]:
    """Build a preview-only surface decal from a canonical 2D mask.

    This intentionally ignores the source inlay mesh triangulation. Some park
    and road solids are triangulated as large fans; once draped onto terrain,
    those long triangles cut through hills and show visible rays.

    Use constrained polygon triangulation first so the boundary follows the
    canonical mask exactly, then subdivide long edges so the surface samples
    terrain often enough to stay visibly above it.
    """
    if mask_geometry is None or getattr(mask_geometry, "is_empty", True) or terrain_provider is None:
        return None

    def _iter_polys(geom):
        if geom is None or getattr(geom, "is_empty", True):
            return []
        if getattr(geom, "geom_type", "") == "Polygon":
            return [geom]
        return [
            part for part in getattr(geom, "geoms", [])
            if getattr(part, "geom_type", "") == "Polygon" and not getattr(part, "is_empty", True)
        ]

    def _clean_poly(poly):
        try:
            clean = poly.buffer(0)
            tolerance = max(float(simplify_tolerance_m or 0.0), 0.0)
            if tolerance > 0.0 and clean is not None and not getattr(clean, "is_empty", True):
                simplified = clean.simplify(tolerance, preserve_topology=True)
                if simplified is not None and not getattr(simplified, "is_empty", True):
                    clean = simplified
            if clean is not None and not getattr(clean, "is_empty", True):
                return clean
        except Exception:
            pass
        return poly

    def _surface_mesh_from_polygon(poly) -> Optional[trimesh.Trimesh]:
        try:
            max_edge = max(float(target_edge_len_m or 0.0), 0.5)
            max_area = max((max_edge * max_edge) * 0.45, 0.25)
            vertices_2d = None
            faces = None

            for engine in ("triangle", "earcut", None):
                try:
                    kwargs = {"engine": engine} if engine is not None else {}
                    if engine == "triangle":
                        kwargs["triangle_args"] = f"pq30a{max_area:.6f}"
                    vertices_2d, faces = trimesh.creation.triangulate_polygon(poly, **kwargs)
                    if vertices_2d is not None and faces is not None and len(vertices_2d) >= 3 and len(faces) > 0:
                        break
                except Exception:
                    vertices_2d = None
                    faces = None
                    continue

            if vertices_2d is None or faces is None or len(vertices_2d) < 3 or len(faces) == 0:
                return None

            vertices_3d = np.column_stack(
                [np.asarray(vertices_2d, dtype=float), np.zeros(len(vertices_2d), dtype=float)]
            )
            mesh = trimesh.Trimesh(vertices=vertices_3d, faces=np.asarray(faces, dtype=np.int64), process=False)
            mesh.remove_unreferenced_vertices()

            try:
                verts, subdiv_faces = trimesh.remesh.subdivide_to_size(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    max_edge=max_edge,
                    max_iter=4,
                )
                if len(verts) <= 120000 and len(subdiv_faces) > 0:
                    mesh = trimesh.Trimesh(vertices=verts, faces=subdiv_faces, process=False)
                    mesh.remove_unreferenced_vertices()
            except Exception:
                pass

            return mesh if len(mesh.vertices) > 0 and len(mesh.faces) > 0 else None
        except Exception as exc:
            print(f"[WARN] build_terrain_decal_from_2d_mask: constrained triangulation failed: {exc}")
            return None

    meshes: list[trimesh.Trimesh] = []
    try:
        cleaned_polys = [_clean_poly(poly) for poly in _iter_polys(mask_geometry)]
        valid_polys = [poly for poly in cleaned_polys if poly is not None and not getattr(poly, "is_empty", True)]
        if valid_polys:
            mask_geometry = unary_union(valid_polys).buffer(0)
    except Exception:
        pass

    for poly in _iter_polys(mask_geometry):
        if getattr(poly, "area", 0.0) <= 1e-6:
            continue
        clean = _clean_poly(poly)
        mesh = _surface_mesh_from_polygon(clean)

        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue

        try:
            new_verts = np.asarray(mesh.vertices, dtype=float).copy()
            heights = terrain_provider.get_surface_heights_for_points(new_verts[:, :2])
            if heights is not None and len(heights) == len(new_verts):
                heights = np.where(np.isfinite(heights), heights, new_verts[:, 2])
                new_verts[:, 2] = heights + float(offset_m)

            decal = trimesh.Trimesh(vertices=new_verts, faces=np.asarray(mesh.faces, dtype=np.int64), process=False)
            decal.remove_unreferenced_vertices()
            if len(decal.vertices) > 0 and len(decal.faces) > 0:
                meshes.append(decal)
        except Exception as exc:
            print(f"[WARN] build_terrain_decal_from_2d_mask failed for polygon: {exc}")
            continue

    if not meshes:
        return None

    try:
        combined = trimesh.util.concatenate(meshes)
        combined.remove_unreferenced_vertices()
        return combined
    except Exception:
        return meshes[0]


def extend_parks_mesh_to_uniform_bottom(parks_mesh: "trimesh.Trimesh") -> None:
    """
    Продовжує дно парків вниз до найнижчої точки мешу.
    
    УВАГА: Тепер це вирівнювання виконується безпосередньо в process_green_areas
    за допомогою збереженої точної маски дна (_bottom_mask), 
    оскільки нормалі або статичний Z-поріг не працюють для парків на схилах.
    Тому ця функція залишається як no-op для сумісності з main.py.
    """
    pass


# ---------------------------------------------------------------------------
# Groove cutting functions
# ---------------------------------------------------------------------------

def cut_parks_from_solid_terrain(
    terrain_mesh: trimesh.Trimesh,
    parks_polygons: Optional[BaseGeometry] = None,
    clearance_m: float = 0.0,
    scale_factor: float = 1.0,
    parks_mesh: Optional[trimesh.Trimesh] = None,
) -> Optional[trimesh.Trimesh]:
    """
    Вирізає ПАЗИ (Grooves) для парків у рельєфі.

    АЛГОРИТМ (Inlay):
      1. Беремо оригінальний 3D меш парків (з урахуванням draping та embed_m).
      2. Передаємо його в Blender як cutter.
      3. Blender застосовує модифікатор DISPLACE (clearance_m) для точного 3D-розширення.
      4. Boolean DIFFERENCE створює паз, який ідеально повторює форму парку.
    """
    # Preview mode: skip ~30-60s of Blender boolean. Parks just sit on terrain.
    if os.environ.get("PREVIEW_MODE", "").lower() in ("1", "true", "yes"):
        print("[TERRAIN CUT] PREVIEW_MODE: skipping parks groove cutting")
        return terrain_mesh
    if terrain_mesh is None:
        return terrain_mesh
    
    if parks_mesh is None or len(parks_mesh.vertices) == 0:
        print("[TERRAIN CUT] WARNING: Parks mesh is None/empty, skipping groove cutting")
        return terrain_mesh

    print(f"[TERRAIN CUT] === Starting parks groove cutting ===")
    print(f"[TERRAIN CUT] Parks mesh: {len(parks_mesh.vertices)} verts, "
          f"Z=[{parks_mesh.bounds[0][2]:.4f}, {parks_mesh.bounds[1][2]:.4f}]")
    print(f"[TERRAIN CUT] Terrain mesh: {len(terrain_mesh.vertices)} verts, "
          f"Z=[{terrain_mesh.bounds[0][2]:.4f}, {terrain_mesh.bounds[1][2]:.4f}]")
    print(f"[TERRAIN CUT] Clearance: {clearance_m:.4f}m ({clearance_m * scale_factor:.2f}mm) via Blender DISPLACE")

    # ВИПРАВЛЕННЯ: Щоб рельєф не утворював "дах" над зеленими зонами (оскільки верх парку
    # лежить під землею на глибині embed_m), ми не можемо використовувати 3D parks_mesh як різак напряму.
    # Натомість ми витягуємо 2D полігони парків від найнижчої точки парку і далеко вгору (вище рельєфу).
    
    terrain_max_z = terrain_mesh.bounds[1][2]
    terrain_floor_z = terrain_mesh.bounds[0][2]
    park_min_z = float(parks_mesh.vertices[:, 2].min())
    # Захист: не ріжемо нижче підлоги рельєфу
    base_preserve = max(0.3, (terrain_max_z - terrain_floor_z) * 0.1)
    park_min_z = max(park_min_z, terrain_floor_z + base_preserve)
    cutter_height = (terrain_max_z - park_min_z) + 2.0  # 2м запас зверху
    
    cutter_mesh = None
    
    if parks_polygons is not None and not parks_polygons.is_empty:
        parks_expanded = parks_polygons
        if clearance_m > 0:
            try:
                parks_expanded = parks_polygons.buffer(clearance_m, join_style=2)
            except Exception:
                pass
                
        cutter_parts = []
        geoms = parks_expanded.geoms if hasattr(parks_expanded, 'geoms') else [parks_expanded]

        # Фільтруємо занадто малі полігони
        valid_geoms = [g for g in geoms if not g.is_empty and g.area > 0.01]
        print(f"[TERRAIN CUT] Extruding {len(valid_geoms)} park polygons (filtered from {len(geoms)})")

        for poly in valid_geoms:
            try:
                # Спрощуємо полігон для стабільності Boolean
                simple = poly.simplify(0.05, preserve_topology=True)
                if simple.is_empty or simple.area < 0.01:
                    continue
                part = trimesh.creation.extrude_polygon(simple, height=cutter_height)
                curr_min = part.bounds[0][2]
                shift_z = park_min_z - curr_min
                part.apply_translation([0, 0, shift_z])
                cutter_parts.append(part)
            except Exception as e:
                print(f"[WARN] Failed to extrude park polygon for cutter: {e}")
                continue

        if cutter_parts:
            cutter_mesh = trimesh.util.concatenate(cutter_parts)
            cutter_mesh.fix_normals()
            print(f"[TERRAIN CUT] Using EXTRUDED 2D POLYGONS as parks cutter: "
                  f"{len(cutter_mesh.vertices)} verts, Z=[{park_min_z:.4f}, {park_min_z + cutter_height:.4f}]")
    
    # Fallback, якщо полігони не передані (може залишати дах з рельєфу)
    if cutter_mesh is None:
        print("[TERRAIN CUT] WARNING: No valid parks_polygons provided, falling back to 3D mesh as cutter (may leave terrain roof).")
        cutter_mesh = parks_mesh.copy()
        cutter_mesh.fix_normals()

    # Зберігаємо проміжні моделі для діагностики
    try:
        debug_dir = os.path.join(os.getcwd(), "debug_grooves")
        os.makedirs(debug_dir, exist_ok=True)
        terrain_mesh.export(os.path.join(debug_dir, "terrain_before_park_cut.stl"))
        cutter_mesh.export(os.path.join(debug_dir, "park_cutter.stl"))
        print(f"[TERRAIN CUT] Debug: park cutter meshes saved to {debug_dir}/")
    except Exception:
        pass

    # 4. Boolean DIFFERENCE через Blender
    # clearance_m вже враховано при буферізації 2D полігонів (parks_expanded.buffer)
    result = _run_blender_boolean(terrain_mesh, cutter_mesh, label="parks")

    try:
        if result is not None:
            result.export(os.path.join(debug_dir, "terrain_after_park_cut.stl"))
    except Exception:
        pass

    return result

def cut_mesh_from_terrain(
    terrain_mesh: "trimesh.Trimesh",
    cutter_mesh: "trimesh.Trimesh",
    clearance_m: float = 0.0,
    scale_factor: float = 1.0,
    label: str = "mesh",
) -> Optional["trimesh.Trimesh"]:
    """
    Вирізає ПАЗ (Groove) у рельєфі для вставки мешу (вода, парки тощо).
    
    ВИПРАВЛЕННЯ: Замість масштабування від центроїда (що ламає non-circular форми),
    витягуємо 2D footprint і розширюємо через Shapely buffer.
    """
    if terrain_mesh is None or cutter_mesh is None:
        return terrain_mesh
    if len(cutter_mesh.vertices) == 0:
        return terrain_mesh

    # Делегуємо до cut_parks_from_solid_terrain (однакова логіка)
    return cut_parks_from_solid_terrain(
        terrain_mesh=terrain_mesh,
        parks_polygons=None,
        clearance_m=clearance_m,
        scale_factor=scale_factor,
        parks_mesh=cutter_mesh,
    )


def cut_roads_from_solid_terrain(
    terrain_mesh: trimesh.Trimesh,
    road_polygons: BaseGeometry,
    clearance_m: float,
    scale_factor: float,
    road_height_m: Optional[float] = None,
    road_mesh: Optional[trimesh.Trimesh] = None
) -> Optional[trimesh.Trimesh]:
    # Preview mode: skip Blender groove cutting (~30-90s saved).
    if os.environ.get("PREVIEW_MODE", "").lower() in ("1", "true", "yes"):
        print("[TERRAIN CUT] PREVIEW_MODE: skipping roads groove cutting")
        return terrain_mesh
    """
    Вирізає ПАЗИ (Grooves) для доріг у рельєфі.

    АЛГОРИТМ (polygon extrusion):
      1. Беремо 2D полігони доріг (road_polygons), розширюємо buffer(clearance_m)
      2. Визначаємо Z-діапазон пазу: від (terrain_floor + base_preserve) до (terrain_top + запас)
      3. Екструдуємо полігони у 3D cutter
      4. Boolean DIFFERENCE через Blender

    ЧОМУ НЕ road_mesh як cutter:
      - DISPLACE розширює по Z вниз → прорізає дно рельєфу
      - Road mesh може бути нижче terrain floor → вирізає стінки наскрізь
      - Polygon extrusion з контрольованим Z дає чистий результат
    """
    if terrain_mesh is None:
        return terrain_mesh

    has_polygons = road_polygons is not None and not road_polygons.is_empty

    if not has_polygons:
        print("[TERRAIN CUT] WARNING: No road_polygons provided, skipping road groove cutting")
        return terrain_mesh

    print(f"[TERRAIN CUT] Starting road GROOVE cutting (polygon extrusion)...")

    bounds = terrain_mesh.bounds
    terrain_max_z = bounds[1][2]
    terrain_floor_z = bounds[0][2]
    terrain_height = terrain_max_z - terrain_floor_z

    # Z-діапазон cutter:
    # Bottom: залишаємо 20% висоти рельєфу як базу (мінімум 0.5м)
    base_preserve = max(0.5, terrain_height * 0.2)
    target_cut_z = terrain_floor_z + base_preserve
    # Top: вище рельєфу на 1м
    cutter_top_z = terrain_max_z + 1.0
    cutter_height = cutter_top_z - target_cut_z

    print(f"[TERRAIN CUT] Terrain Z=[{terrain_floor_z:.4f}, {terrain_max_z:.4f}], height={terrain_height:.4f}")
    print(f"[TERRAIN CUT] Cutter Z=[{target_cut_z:.4f}, {cutter_top_z:.4f}], base_preserve={base_preserve:.4f}")

    # Clearance: розширюємо 2D полігони (лише XY, не Z!)
    road_expanded = road_polygons
    if clearance_m > 0:
        try:
            road_expanded = road_polygons.buffer(clearance_m, join_style=2)
            print(f"[TERRAIN CUT] Applied 2D clearance buffer: {clearance_m:.4f}m")
        except Exception:
            pass

    # Екструдуємо полігони
    cutter_parts = []
    geoms = road_expanded.geoms if hasattr(road_expanded, 'geoms') else [road_expanded]
    valid_geoms = [g for g in geoms if not g.is_empty and g.area > 0.01]
    print(f"[TERRAIN CUT] Extruding {len(valid_geoms)} road polygons (filtered from {len(list(geoms))})")

    for poly in valid_geoms:
        try:
            simple = poly.simplify(0.05, preserve_topology=True)
            if simple.is_empty or simple.area < 0.01:
                continue
            part = trimesh.creation.extrude_polygon(simple, height=cutter_height)
            curr_min = part.bounds[0][2]
            shift_z = target_cut_z - curr_min
            part.apply_translation([0, 0, shift_z])
            cutter_parts.append(part)
        except Exception:
            continue

    if not cutter_parts:
        print("[TERRAIN CUT] WARNING: No valid cutter parts created")
        return terrain_mesh

    cutter_mesh = trimesh.util.concatenate(cutter_parts)
    cutter_mesh.fix_normals()
    print(f"[TERRAIN CUT] Polygon cutter: {len(cutter_mesh.vertices)} verts, {len(cutter_mesh.faces)} faces")
    print(f"[TERRAIN CUT] Cutter bounds: Z=[{cutter_mesh.bounds[0][2]:.4f}, {cutter_mesh.bounds[1][2]:.4f}]")

    # Зберігаємо проміжні моделі для діагностики
    try:
        debug_dir = os.path.join(os.getcwd(), "debug_grooves")
        os.makedirs(debug_dir, exist_ok=True)
        terrain_mesh.export(os.path.join(debug_dir, "terrain_before_road_cut.stl"))
        cutter_mesh.export(os.path.join(debug_dir, "road_cutter.stl"))
        if road_mesh is not None:
            road_mesh.export(os.path.join(debug_dir, "road_mesh_original.stl"))
        print(f"[TERRAIN CUT] Debug meshes saved to {debug_dir}/")
    except Exception as e:
        print(f"[TERRAIN CUT] Failed to save debug meshes: {e}")

    # Boolean БЕЗ DISPLACE (clearance вже в 2D buffer)
    result = _run_blender_boolean(terrain_mesh, cutter_mesh, label="roads", clearance_m=0.0)

    # Зберігаємо результат для діагностики
    try:
        if result is not None:
            result.export(os.path.join(debug_dir, "terrain_after_road_cut.stl"))
            print(f"[TERRAIN CUT] Debug: terrain_after_road_cut.stl saved")
    except Exception:
        pass

    return result


def cut_all_grooves(
    terrain_mesh: trimesh.Trimesh,
    road_polygons: Optional[BaseGeometry] = None,
    road_clearance_m: float = 0.0,
    parks_polygons: Optional[BaseGeometry] = None,
    parks_clearance_m: float = 0.0,
    parks_mesh: Optional[trimesh.Trimesh] = None,
    water_polygons: Optional[BaseGeometry] = None,
    water_clearance_m: float = 0.0,
    water_mesh: Optional[trimesh.Trimesh] = None,
    scale_factor: float = 1.0,
    road_mesh: Optional[trimesh.Trimesh] = None,
    groove_depth_m: Optional[float] = None,
) -> Optional[trimesh.Trimesh]:
    """
    Вирізає ВСІ пази (дороги + парки) ОДНИМ Blender Boolean.

    КЛЮЧОВИЙ ПРИНЦИП: Спочатку об'єднуємо ВСІ 2D полігони в одну чисту
    геометрію через unary_union, потім екструдуємо ОДИН раз.

    groove_depth_m: глибина пазу від поверхні рельєфу (в метрах світу).
                    Визначається з embed глибини інлей-деталей.
    """
    if terrain_mesh is None:
        return terrain_mesh

    has_roads = road_polygons is not None and not road_polygons.is_empty
    has_parks = parks_polygons is not None and not parks_polygons.is_empty
    has_water = water_polygons is not None and not water_polygons.is_empty

    if not has_roads and not has_parks and not has_water:
        print("[TERRAIN CUT] No road, park or water polygons provided, skipping groove cutting")
        return terrain_mesh

    bounds = terrain_mesh.bounds
    terrain_max_z = bounds[1][2]
    terrain_floor_z = bounds[0][2]
    terrain_height = terrain_max_z - terrain_floor_z

    print(f"[TERRAIN CUT] === Unified groove cutting ===")
    print(f"[TERRAIN CUT] Terrain Z=[{terrain_floor_z:.4f}, {terrain_max_z:.4f}], height={terrain_height:.4f}")

    # Z-діапазон cutter:
    # Bottom: визначається РЕАЛЬНИМ нижнім Z інлей-деталей (road_mesh, parks_mesh)
    # Вирівнюємо дно ВСІХ інлеїв до єдиного найнижчого Z
    mesh_min_z = float('inf')
    mesh_bottoms = {}
    if road_mesh is not None:
        road_bot_z = float(road_mesh.bounds[0][2])
        mesh_min_z = min(mesh_min_z, road_bot_z)
        mesh_bottoms['road'] = road_bot_z
        print(f"[TERRAIN CUT] Road mesh bottom Z: {road_bot_z:.4f}")
    if parks_mesh is not None:
        parks_bot_z = float(parks_mesh.bounds[0][2])
        mesh_min_z = min(mesh_min_z, parks_bot_z)
        mesh_bottoms['parks'] = parks_bot_z
        print(f"[TERRAIN CUT] Parks mesh bottom Z: {parks_bot_z:.4f}")
    if water_mesh is not None:
        water_bot_z = float(water_mesh.bounds[0][2])
        mesh_min_z = min(mesh_min_z, water_bot_z)
        mesh_bottoms['water'] = water_bot_z
        print(f"[TERRAIN CUT] Water mesh bottom Z: {water_bot_z:.4f}")

    # Вирівнюємо дно ВСІХ інлеїв до єдиного mesh_min_z
    # Використовуємо нормалі граней: грані з нормаллю вниз (Z < -0.5) = дно
    if mesh_min_z < float('inf'):
        for label, mesh_obj in [('road', road_mesh), ('parks', parks_mesh), ('water', water_mesh)]:
            if mesh_obj is None:
                continue
            bot_z = mesh_bottoms.get(label)
            if bot_z is None:
                continue

            verts = mesh_obj.vertices.copy()
            face_normals = mesh_obj.face_normals
            faces = mesh_obj.faces

            # Грані з нормаллю вниз = нижня поверхня
            bottom_faces = face_normals[:, 2] < -0.5
            # Вершини цих граней
            bottom_vert_ids = np.unique(faces[bottom_faces].ravel())
            count = len(bottom_vert_ids)

            if count > 0:
                old_min = float(verts[bottom_vert_ids, 2].min())
                old_max = float(verts[bottom_vert_ids, 2].max())
                verts[bottom_vert_ids, 2] = mesh_min_z
                mesh_obj.vertices = verts
                print(f"[TERRAIN CUT] {label} bottom aligned: Z[{old_min:.4f}..{old_max:.4f}] -> {mesh_min_z:.4f} ({count} verts)")
            else:
                print(f"[TERRAIN CUT] {label}: no bottom faces found")

    if mesh_min_z < float('inf'):
        # Паз трохи нижче єдиного дна інлеїв
        cut_bottom_z = mesh_min_z - 0.1
    elif groove_depth_m is not None and groove_depth_m > 0:
        # Fallback: groove_depth_m від поверхні
        z_values = terrain_mesh.vertices[:, 2]
        surface_min_z = float(np.percentile(z_values, 25))
        cut_bottom_z = surface_min_z - groove_depth_m
    else:
        cut_bottom_z = terrain_floor_z + terrain_height * 0.3

    # Ніколи не різати нижче підлоги (з мінімальним запасом)
    min_floor = terrain_floor_z + 0.5
    cut_bottom_z = max(cut_bottom_z, min_floor)

    # Top: вище рельєфу — гарантуємо повне вирізання поверхні
    cut_top_z = terrain_max_z + 5.0
    cutter_height = cut_top_z - cut_bottom_z

    print(f"[TERRAIN CUT] Cutter Z=[{cut_bottom_z:.4f}, {cut_top_z:.4f}], h={cutter_height:.4f}")

    # === КРОК 1: Збираємо ВСІ 2D полігони в один список ===
    all_2d_polygons = []

    def _collect_polygons(geom, clearance_m, label):
        """Збирає чисті Polygon об'єкти з геометрії."""
        if geom is None or geom.is_empty:
            return []

        expanded = geom
        if clearance_m > 0:
            try:
                expanded = geom.buffer(clearance_m, join_style=2)
            except Exception:
                pass

        # buffer(0) для cleanup самоперетинів
        try:
            expanded = expanded.buffer(0)
        except Exception:
            pass

        polygons = []
        if hasattr(expanded, 'geoms'):
            raw_geoms = list(expanded.geoms)
        else:
            raw_geoms = [expanded]

        for g in raw_geoms:
            if g is None or g.is_empty:
                continue
            if g.geom_type == 'Polygon' and g.area > 0.001:
                polygons.append(g)
            elif g.geom_type == 'MultiPolygon':
                for p in g.geoms:
                    if not p.is_empty and p.area > 0.001:
                        polygons.append(p)

        print(f"[TERRAIN CUT] {label}: collected {len(polygons)} polygons")
        return polygons

    if has_roads:
        road_polys = _collect_polygons(road_polygons, road_clearance_m, "Roads")
        all_2d_polygons.extend(road_polys)

    if has_parks:
        park_polys = _collect_polygons(parks_polygons, parks_clearance_m, "Parks")
        all_2d_polygons.extend(park_polys)
    if has_water:
        water_polys = _collect_polygons(water_polygons, water_clearance_m, "Water")
        all_2d_polygons.extend(water_polys)

    if not all_2d_polygons:
        print("[TERRAIN CUT] WARNING: No valid polygons collected!")
        return terrain_mesh

    # === КРОК 2: unary_union — об'єднуємо ВСІ полігони в одну чисту геометрію ===
    # Це КРИТИЧНО: усуває перетини між road і park cutters
    try:
        combined_2d = unary_union(all_2d_polygons)
        # Clean up
        combined_2d = combined_2d.buffer(0)
        print(f"[TERRAIN CUT] Combined 2D: type={combined_2d.geom_type}, area={combined_2d.area:.2f} m2")
    except Exception as e:
        print(f"[TERRAIN CUT] WARNING: unary_union failed: {e}, falling back to individual polygons")
        combined_2d = None

    # === КРОК 3: Екструдуємо об'єднану 2D геометрію ===
    cutter_parts = []

    if combined_2d is not None and not combined_2d.is_empty:
        # Розбираємо на окремі Polygon для екструзії
        if combined_2d.geom_type == 'Polygon':
            final_polys = [combined_2d]
        elif combined_2d.geom_type == 'MultiPolygon':
            final_polys = list(combined_2d.geoms)
        else:
            # GeometryCollection — витягуємо полігони
            final_polys = [g for g in combined_2d.geoms if g.geom_type == 'Polygon' and g.area > 0.001]

        print(f"[TERRAIN CUT] Extruding {len(final_polys)} unified polygons...")

        for i, poly in enumerate(final_polys):
            if poly.is_empty or poly.area < 0.001:
                continue
            try:
                # Спрощуємо для стабільності Boolean (але обережно)
                simple = poly.simplify(0.02, preserve_topology=True)
                if simple.is_empty or simple.area < 0.001 or simple.geom_type != 'Polygon':
                    simple = poly

                part = trimesh.creation.extrude_polygon(simple, height=cutter_height)
                curr_min = part.bounds[0][2]
                part.apply_translation([0, 0, cut_bottom_z - curr_min])
                cutter_parts.append(part)
            except Exception as e:
                print(f"[TERRAIN CUT] WARN: Failed to extrude polygon {i} (area={poly.area:.2f}): {e}")
                # Fallback: спробуємо convex hull
                try:
                    hull = poly.convex_hull
                    if hull.geom_type == 'Polygon' and hull.area > 0.001:
                        part = trimesh.creation.extrude_polygon(hull, height=cutter_height)
                        curr_min = part.bounds[0][2]
                        part.apply_translation([0, 0, cut_bottom_z - curr_min])
                        cutter_parts.append(part)
                        print(f"[TERRAIN CUT] Used convex hull fallback for polygon {i}")
                except Exception:
                    pass
    else:
        # Fallback: екструдуємо кожний полігон окремо
        print("[TERRAIN CUT] Falling back to individual polygon extrusion...")
        for poly in all_2d_polygons:
            try:
                simple = poly.simplify(0.02, preserve_topology=True)
                if simple.is_empty or simple.geom_type != 'Polygon':
                    continue
                part = trimesh.creation.extrude_polygon(simple, height=cutter_height)
                curr_min = part.bounds[0][2]
                part.apply_translation([0, 0, cut_bottom_z - curr_min])
                cutter_parts.append(part)
            except Exception:
                continue

    if not cutter_parts:
        print("[TERRAIN CUT] WARNING: No cutter parts created!")
        return terrain_mesh

    # === КРОК 4: Об'єднуємо 3D частини ===
    cutter_mesh = trimesh.util.concatenate(cutter_parts)
    # НЕ викликаємо fix_normals() — trimesh ray-casting heuristic ламає
    # нормалі на складних не-convex мешах. Blender сам нормалізує.
    print(f"[TERRAIN CUT] Final cutter: {len(cutter_mesh.vertices)} verts, {len(cutter_mesh.faces)} faces")
    print(f"[TERRAIN CUT] Cutter bounds: X=[{cutter_mesh.bounds[0][0]:.2f}, {cutter_mesh.bounds[1][0]:.2f}], "
          f"Y=[{cutter_mesh.bounds[0][1]:.2f}, {cutter_mesh.bounds[1][1]:.2f}], "
          f"Z=[{cutter_mesh.bounds[0][2]:.4f}, {cutter_mesh.bounds[1][2]:.4f}]")

    # Debug meshes
    try:
        debug_dir = os.path.join(os.getcwd(), "debug_grooves")
        os.makedirs(debug_dir, exist_ok=True)
        terrain_mesh.export(os.path.join(debug_dir, "terrain_before_all_cuts.stl"))
        cutter_mesh.export(os.path.join(debug_dir, "combined_cutter.stl"))
        if road_mesh is not None:
            road_mesh.export(os.path.join(debug_dir, "road_mesh_original.stl"))
        if parks_mesh is not None:
            parks_mesh.export(os.path.join(debug_dir, "parks_mesh_original.stl"))
        print(f"[TERRAIN CUT] Debug meshes saved to {debug_dir}/")
    except Exception:
        pass

    # === КРОК 5: ОДИН Boolean DIFFERENCE ===
    result = _run_blender_boolean(terrain_mesh, cutter_mesh, label="all_grooves", clearance_m=0.0)

    try:
        if result is not None:
            debug_dir = os.path.join(os.getcwd(), "debug_grooves")
            result.export(os.path.join(debug_dir, "terrain_after_all_cuts.stl"))
    except Exception:
        pass

    return result


def cut_grooves_sequentially(
    terrain_mesh: trimesh.Trimesh,
    road_polygons: Optional[BaseGeometry] = None,
    road_clearance_m: float = 0.0,
    parks_polygons: Optional[BaseGeometry] = None,
    parks_clearance_m: float = 0.0,
    parks_mesh: Optional[trimesh.Trimesh] = None,
    water_polygons: Optional[BaseGeometry] = None,
    water_clearance_m: float = 0.0,
    water_mesh: Optional[trimesh.Trimesh] = None,
    scale_factor: float = 1.0,
    road_mesh: Optional[trimesh.Trimesh] = None,
    groove_depth_m: Optional[float] = None,
) -> Optional[trimesh.Trimesh]:
    if terrain_mesh is None:
        return terrain_mesh

    result = terrain_mesh

    if road_polygons is not None and not road_polygons.is_empty:
        road_cut = cut_roads_from_solid_terrain(
            terrain_mesh=result,
            road_polygons=road_polygons,
            clearance_m=road_clearance_m,
            scale_factor=scale_factor,
            road_height_m=None,
            road_mesh=road_mesh,
        )
        result = _accept_boolean_stage_result(
            previous_mesh=result,
            candidate_mesh=road_cut,
            label="roads",
        )

    if parks_polygons is not None and not parks_polygons.is_empty and parks_mesh is not None and len(parks_mesh.vertices) > 0:
        parks_cut = cut_parks_from_solid_terrain(
            terrain_mesh=result,
            parks_polygons=parks_polygons,
            clearance_m=parks_clearance_m,
            scale_factor=scale_factor,
            parks_mesh=parks_mesh,
        )
        result = _accept_boolean_stage_result(
            previous_mesh=result,
            candidate_mesh=parks_cut,
            label="parks",
        )

    if water_polygons is not None and not water_polygons.is_empty and water_mesh is not None and len(water_mesh.vertices) > 0:
        water_cut = cut_parks_from_solid_terrain(
            terrain_mesh=result,
            parks_polygons=water_polygons,
            clearance_m=water_clearance_m,
            scale_factor=scale_factor,
            parks_mesh=water_mesh,
        )
        result = _accept_boolean_stage_result(
            previous_mesh=result,
            candidate_mesh=water_cut,
            label="water",
        )

    return result
