"""
Blender-очистка дегенеративних мешів («частокіл»/слівери у пазах зʼєднувачів і
на стінках рельєфу).

ПРОБЛЕМА: булеві операції (паз-notch + груви + злиття будинків) на густому
грід-рельєфі лишають тисячі тонких-вироджених трикутників (zero-area + слівери +
вертикальні спайки), які trimesh НЕ може прибрати (merge_vertices не бере тонкі
ВИСОКІ слівери; видалення рве герметичність — перевірено). Blender бере це
надійно: weld(merge by distance) + dissolve_degenerate + dissolve_limited
(злиття майже-компланарних) збиває спайки на ~97% і ЗБЕРІГАЄ будинки/рельєф/пази.

Сервер має Blender 4.0.2 (/usr/bin/blender), як і terrain_cutter.
"""

from __future__ import annotations

import os
import time
import uuid
import shutil
import subprocess
from typing import Optional

import trimesh

from services.terrain_cutter import _find_blender


def blender_cleanup_mesh(
    mesh: Optional[trimesh.Trimesh],
    *,
    scale_factor: float = 1.0,
    merge_mm: float = 0.06,
    degen_mm: float = 0.06,
    planar_deg: float = 3.0,
    delete_loose: bool = True,
    label: str = "base",
    timeout_s: int = 240,
) -> Optional[trimesh.Trimesh]:
    """Прогнати меш через Blender-очистку вироджених граней.

    Пороги задаються у МОДЕЛІ-мм і конвертуються у одиниці меша через
    scale_factor (пайплайн тримає геометрію у світових метрах; для вже-
    масштабованого у мм меша передавай scale_factor=1.0).

    Безпечний fallback: якщо Blender відсутній / помилка / результат гірший —
    повертає ОРИГІНАЛ (ніколи не повертає None, якщо на вході був меш).
    """
    if mesh is None or len(mesh.faces) == 0:
        return mesh
    blender_exe = _find_blender()
    if blender_exe is None:
        print(f"[MESH CLEANUP] Blender not found — skip {label} cleanup (mesh unchanged)")
        return mesh

    sf = float(scale_factor) if scale_factor and scale_factor > 0 else 1.0
    merge_w = float(merge_mm) / sf
    degen_w = float(degen_mm) / sf
    planar_rad = float(planar_deg) * 3.141592653589793 / 180.0

    session_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join(os.getcwd(), f"temp_cleanup_{session_id}")
    os.makedirs(temp_dir, exist_ok=True)
    in_path = os.path.abspath(os.path.join(temp_dir, "in.stl"))
    out_path = os.path.abspath(os.path.join(temp_dir, "out.stl"))
    script_path = os.path.abspath(os.path.join(temp_dir, "cleanup.py"))

    try:
        mesh.export(in_path)
        script = f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
try:
    bpy.ops.wm.stl_import(filepath=r"{in_path}")
except Exception:
    bpy.ops.import_mesh.stl(filepath=r"{in_path}")
o = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = o
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold={merge_w})
bpy.ops.mesh.dissolve_degenerate(threshold={degen_w})
{"bpy.ops.mesh.dissolve_limited(angle_limit=%f)" % planar_rad if planar_deg and planar_deg > 0 else "# dissolve_limited skipped (planar_deg<=0): keeps top terrain surface fine so roads/parks dont poke through"}
{"bpy.ops.mesh.select_all(action='SELECT'); bpy.ops.mesh.delete_loose()" if delete_loose else ""}
# Close any open edges left by dissolve so the base stays watertight (no Bambu repair prompts).
bpy.ops.mesh.select_all(action='SELECT')
try:
    bpy.ops.mesh.fill_holes(sides=0)
except Exception:
    pass
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')
o.select_set(True)
bpy.context.view_layer.objects.active = o
try:
    bpy.ops.wm.stl_export(filepath=r"{out_path}", export_selected_objects=True)
except Exception:
    bpy.ops.export_mesh.stl(filepath=r"{out_path}", use_selection=True)
print("CLEANUP_SUCCESS")
"""
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        t0 = time.time()
        proc = subprocess.run(
            [blender_exe, "--background", "--python", script_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s,
        )
        ok = b"CLEANUP_SUCCESS" in proc.stdout
        if not ok or not os.path.exists(out_path):
            print(f"[MESH CLEANUP] {label}: Blender did not finish cleanly — keeping original")
            return mesh
        cleaned = trimesh.load(out_path)
        if cleaned is None or len(getattr(cleaned, "faces", [])) == 0:
            return mesh
        # Guard: не приймати результат, що схлопнув геометрію (захист від over-clean)
        if len(cleaned.faces) < 0.15 * len(mesh.faces):
            print(f"[MESH CLEANUP] {label}: result collapsed ({len(mesh.faces)}->{len(cleaned.faces)}) — keeping original")
            return mesh
        print(f"[MESH CLEANUP] {label}: faces {len(mesh.faces)}->{len(cleaned.faces)} in {time.time()-t0:.1f}s")
        return cleaned
    except Exception as exc:
        print(f"[MESH CLEANUP] {label}: error ({exc}) — keeping original")
        return mesh
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
