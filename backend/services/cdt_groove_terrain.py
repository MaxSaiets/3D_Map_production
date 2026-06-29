"""CDT-based CLEAN groove-wall terrain construction for RELIEF maps.

Замінює «boolean-виріз жолобів у дрібну сітку» (який шматує стінки слотів на
тонкі смужки → «гребінець/дегенеративні стінки») на КОНСТРУКЦІЮ через
Constrained Delaunay Triangulation: межі доріг/парків/води = ОБМЕЖУВАЛЬНІ ребра
тріангуляції, тож стінки слотів = чисті бордюр-вирівняні квад-смужки.

Доведено прототипом (4-агентний judge-воркфлоу, 2026-06-28): combWalls 93→1,
watertight, рельєф цілий (reliefErr95 0.1). Див. [[thin-terrain-walls-stovbi]].

Ключові ідеї (НЕ ламати при правках):
  * `triangle` СЕГФОЛТИТЬ → запускаємо у СУБПРОЦЕСІ (цей же файл з --worker) з
    ретраями; падіння субпроцесу НЕ валить пайплайн. [[triangle-lib-segfault]]
  * `-YY` прапор = БЕЗ Steiner-точок на обмежувальних сегментах → ущільнені
    кільця меж лишаються вершина-в-вершину однакові між terrain/walls/floor →
    шви закриваються під merge_vertices (герметично за побудовою).
  * Стінки беремо з ВЛАСНИХ boundary_loops меша terrain-top (а не з незалежно
    тріангульованих кілець) → верх стінки = шов terrain точно → нема розривів.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from typing import Callable, Optional

import numpy as np

try:  # важкі залежності лише коли реально будуємо (не у --worker гілці)
    import trimesh
    from shapely.geometry import (LineString, MultiPolygon, Point, Polygon)
    from shapely.ops import unary_union
except Exception:  # pragma: no cover
    trimesh = None


# ──────────────────────────────────────────────────────────────────────────
#  SUBPROCESS WORKER  (python services/cdt_groove_terrain.py --worker spec out)
#  Ізолюємо `triangle` (може сегфолтити) у власному процесі.
# ──────────────────────────────────────────────────────────────────────────
def _worker(spec_path: str, out_path: str) -> None:
    import triangle as tr  # локальний імпорт — лише у субпроцесі
    with open(spec_path) as f:
        spec = json.load(f)
    A = {"vertices": np.asarray(spec["vertices"], dtype=float)}
    if spec.get("segments"):
        A["segments"] = np.asarray(spec["segments"], dtype=np.int32)
    if spec.get("holes"):
        A["holes"] = np.asarray(spec["holes"], dtype=float)
    B = tr.triangulate(A, spec.get("triangle_args", "pq30a8YY"))
    with open(out_path, "w") as f:
        json.dump({
            "vertices": np.asarray(B["vertices"], dtype=float).tolist(),
            "triangles": np.asarray(B["triangles"], dtype=int).tolist(),
        }, f)


def _run_triangle(vertices, segments, holes, triangle_args, tag, work_dir):
    spec = {
        "vertices": [list(map(float, v)) for v in vertices],
        "segments": [list(map(int, s)) for s in segments],
        "holes": [list(map(float, h)) for h in holes] if holes else [],
        "triangle_args": triangle_args,
    }
    spec_path = os.path.join(work_dir, f"spec_{tag}.json")
    out_path = os.path.join(work_dir, f"out_{tag}.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f)
    last = None
    for _ in range(3):
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", spec_path, out_path],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode == 0 and os.path.exists(out_path):
            with open(out_path) as f:
                res = json.load(f)
            return np.asarray(res["vertices"]), np.asarray(res["triangles"], dtype=np.int64)
        last = f"rc={proc.returncode} stderr={(proc.stderr or '')[-400:]}"
    raise RuntimeError(f"triangle subprocess failed [{tag}]: {last}")


# ──────────────────────────────────────────────────────────────────────────
#  GEOMETRY HELPERS
# ──────────────────────────────────────────────────────────────────────────
def _poly_to_pslg(poly):
    pts, segs, holes = [], [], []

    def add_ring(coords):
        coords = list(coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        start = len(pts)
        n = len(coords)
        for c in coords:
            pts.append((c[0], c[1]))
        for i in range(n):
            segs.append((start + i, start + ((i + 1) % n)))

    geoms = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
    for g in geoms:
        add_ring(g.exterior.coords)
        for interior in g.interiors:
            add_ring(interior.coords)
            ip = Polygon(interior).representative_point()
            holes.append((ip.x, ip.y))
    return pts, segs, holes


def _points_pslg(points2d, closed_loops):
    pts = [tuple(p) for p in points2d]
    segs = []
    for loop in closed_loops:
        n = len(loop)
        for i in range(n):
            segs.append((loop[i], loop[(i + 1) % n]))
    return pts, segs


def _ring_chains(poly):
    chains = []
    geoms = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
    for g in geoms:
        for ring in [g.exterior, *g.interiors]:
            coords = list(ring.coords)
            if len(coords) > 1 and coords[0] == coords[-1]:
                coords = coords[:-1]
            chains.append([(c[0], c[1]) for c in coords])
    return chains


def _densify_ring(coords, max_seg):
    out = []
    n = len(coords)
    for i in range(n):
        a = coords[i]
        b = coords[(i + 1) % n]
        out.append(a)
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        if d > max_seg:
            k = int(d // max_seg)
            for j in range(1, k + 1):
                t = j / (k + 1)
                out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def _densify_polygon(poly, max_seg):
    geoms = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
    newgeoms = []
    for g in geoms:
        ext = _densify_ring(list(g.exterior.coords)[:-1], max_seg)
        ints = [_densify_ring(list(r.coords)[:-1], max_seg) for r in g.interiors]
        newgeoms.append(Polygon(ext, ints))
    return MultiPolygon(newgeoms) if len(newgeoms) > 1 else newgeoms[0]


_BLENDER_FILL = '''
import bpy, sys
inp, outp = sys.argv[-2], sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
try: bpy.ops.wm.stl_import(filepath=inp)
except: bpy.ops.import_mesh.stl(filepath=inp)
o=bpy.context.selected_objects[0]; bpy.context.view_layer.objects.active=o
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=1e-4)
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_non_manifold()
bpy.ops.mesh.edge_face_add()
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')
try: bpy.ops.wm.stl_export(filepath=outp)
except: bpy.ops.export_mesh.stl(filepath=outp)
print("FILL_OK")
'''


def _blender_fill_watertight(mesh, work_dir):
    """Закриває СТРУКТУРНІ відкриті ребра швів (кап-стінка) через Blender
    edge_face_add — надійніше за trimesh fill_holes на складній road-топології.
    Стінки лишаються чисті (філи — лише пласкі грані на дні/підлозі). Без Blender
    → повертає як є (на проді Blender завжди є)."""
    try:
        from services.terrain_cutter import _find_blender
        blender = _find_blender()
    except Exception:  # noqa: BLE001
        blender = None
    if not blender:
        return mesh
    inp = os.path.join(work_dir, "fill_in.stl")
    outp = os.path.join(work_dir, "fill_out.stl")
    scr = os.path.join(work_dir, "fill.py")
    try:
        mesh.export(inp)
        with open(scr, "w", encoding="utf-8") as f:
            f.write(_BLENDER_FILL)
        proc = subprocess.run([blender, "--background", "--python", scr, "--", inp, outp],
                              capture_output=True, text=True, timeout=180)
        if os.path.exists(outp):
            filled = trimesh.load(outp, force="mesh")
            if filled is not None and len(filled.faces) > 0:
                return filled
    except Exception as exc:  # noqa: BLE001
        print(f"[CDT] blender fill skipped ({exc})")
    return mesh


def _boundary_loops(mesh):
    from trimesh.grouping import group_rows
    g = group_rows(mesh.edges_sorted, require_count=1)
    edges = mesh.edges[g]
    nxt = {}
    for a, b in edges:
        nxt.setdefault(int(a), []).append(int(b))
    loops = []
    visited = set()
    for a0, b0 in edges:
        a0 = int(a0); b0 = int(b0)
        if (a0, b0) in visited:
            continue
        loop = [a0]
        cur = b0
        visited.add((a0, b0))
        guard = 0
        while cur != a0 and guard < len(edges) + 5:
            loop.append(cur)
            picked = None
            for nb in nxt.get(cur, []):
                if (cur, nb) not in visited:
                    picked = nb
                    break
            if picked is None:
                break
            visited.add((cur, picked))
            cur = picked
            guard += 1
        loops.append(loop)
    return loops


# ──────────────────────────────────────────────────────────────────────────
#  PUBLIC: build the grooved terrain solid via CDT
# ──────────────────────────────────────────────────────────────────────────
def build_cdt_grooved_terrain(
    zone_polygon,
    inlay_mask,
    height_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    slab_z: float,
    floor_z: float,
    seg_len: float = 3.0,
    tri_area: float = 8.0,
):
    """Будує ГЕРМЕТИЧНИЙ рельєф-солід із ЧИСТИМИ стінками слотів інлеїв.

    zone_polygon  — межа плитки (shapely Polygon, локальні метри).
    inlay_mask    — об'єднана 2D-маска жолобів (roads ∪ parks ∪ water), shapely.
    height_fn(x,y)→z — семплер висоти поверхні рельєфу (terrain_provider).
    slab_z        — дно слотів інлеїв (єдине, як mesh_min_z у cut_all_grooves).
    floor_z       — низ плити.
    Повертає trimesh.Trimesh (top+walls+floor+bottom+bridges, merge_vertices).

    Гейтнути на наявність рельєфу + інлеїв; flat/golden не використовують.
    """
    if trimesh is None:
        raise RuntimeError("trimesh/shapely unavailable")
    work_dir = tempfile.mkdtemp(prefix="cdt_groove_")
    try:
        zone = zone_polygon.buffer(0)
        inlays = inlay_mask.buffer(0) if inlay_mask is not None else None
        if inlays is not None and not inlays.is_empty:
            inlays = inlays.intersection(zone)
        has_inlays = inlays is not None and not inlays.is_empty

        # ── 1) TERRAIN top: CDT of (zone − inlays) ──────────────────────────
        terrain_poly = zone.difference(inlays) if has_inlays else zone
        terrain_poly_d = _densify_polygon(terrain_poly, seg_len)
        pts, segs, holes = _poly_to_pslg(terrain_poly_d)
        Vt2, Ft = _run_triangle(pts, segs, holes, f"pq30a{tri_area:g}YY", "terrain", work_dir)
        zt = np.asarray(height_fn(Vt2[:, 0], Vt2[:, 1]), dtype=float)
        Vt = np.column_stack([Vt2[:, 0], Vt2[:, 1], zt])
        terrain_mesh = trimesh.Trimesh(vertices=Vt, faces=Ft, process=False)
        terrain_mesh.fix_normals()
        if terrain_mesh.face_normals[:, 2].mean() < 0:
            terrain_mesh.faces = terrain_mesh.faces[:, ::-1]

        # perimeter test (zone border edges)
        zminx, zminy, zmaxx, zmaxy = zone.bounds

        def edge_is_perim(pa, pb):
            for axis, val in ((0, zmaxx), (0, zminx), (1, zmaxy), (1, zminy)):
                if abs(pa[axis] - val) < 1e-4 and abs(pb[axis] - val) < 1e-4:
                    return True
            return False

        # corner posts = border∩inlay junctions (need a SLAB_Z split level)
        corner_posts = set()
        if has_inlays:
            for ch in _ring_chains(_densify_polygon(inlays, seg_len)):
                for (x, y) in ch:
                    if (abs(x - zmaxx) < 1e-4 or abs(x - zminx) < 1e-4
                            or abs(y - zmaxy) < 1e-4 or abs(y - zminy) < 1e-4):
                        corner_posts.add((round(x, 4), round(y, 4)))

        def col_levels(x, y, z_hi, z_lo):
            levels = [z_hi]
            if (round(x, 4), round(y, 4)) in corner_posts and z_lo < slab_z < z_hi:
                levels.append(slab_z)
            levels.append(z_lo)
            return levels

        wall_verts, wall_faces = [], []

        def vidx(x, y, z):
            wall_verts.append((x, y, z))
            return len(wall_verts) - 1

        for loop in _boundary_loops(terrain_mesh):
            P = terrain_mesh.vertices[loop]
            n = len(loop)
            for i in range(n):
                a = P[i]
                b = P[(i + 1) % n]
                if edge_is_perim(a, b):
                    z_lo, outward_neg = floor_z, True
                else:
                    z_lo, outward_neg = slab_z, False
                la = col_levels(a[0], a[1], a[2], z_lo)
                lb = col_levels(b[0], b[1], b[2], z_lo)
                colA = [vidx(a[0], a[1], z) for z in la]
                colB = [vidx(b[0], b[1], z) for z in lb]
                poly = colA + list(reversed(colB))
                apex = 0 if len(la) <= len(lb) else len(la)
                seq = list(range(len(poly)))
                rot = seq[apex:] + seq[:apex]
                for k in range(1, len(rot) - 1):
                    p, q, r = poly[rot[0]], poly[rot[k]], poly[rot[k + 1]]
                    wall_faces.append((p, r, q) if not outward_neg else (p, q, r))

        wall_mesh = trimesh.Trimesh(vertices=np.asarray(wall_verts),
                                    faces=np.asarray(wall_faces), process=False)

        def _z_loops(mesh, zval, tol=1e-3):
            """Впорядковані петлі ВІДКРИТИХ ребер меша, чиї вершини всі на z≈zval.
            Дають ТОЧНІ бордюр-вершини стінок → каже (floor/bottom) шиються без шва."""
            if mesh is None or len(mesh.faces) == 0:
                return []
            from trimesh.grouping import group_rows
            g = group_rows(mesh.edges_sorted, require_count=1)
            ed = mesh.edges[g]
            V = mesh.vertices
            keep = [(int(a), int(b)) for a, b in ed
                    if abs(V[a][2] - zval) < tol and abs(V[b][2] - zval) < tol]
            nxt = {}
            for a, b in keep:
                nxt.setdefault(a, []).append(b)
            loops, visited = [], set()
            for a0, b0 in keep:
                if (a0, b0) in visited:
                    continue
                loop = [a0]
                cur = b0
                visited.add((a0, b0))
                guard = 0
                while cur != a0 and guard < len(keep) + 5:
                    loop.append(cur)
                    picked = None
                    for nb in nxt.get(cur, []):
                        if (cur, nb) not in visited:
                            picked = nb
                            break
                    if picked is None:
                        break
                    visited.add((cur, picked))
                    cur = picked
                    guard += 1
                if len(loop) >= 3:
                    loops.append([(float(V[i][0]), float(V[i][1])) for i in loop])
            return loops

        def _cap_from_loops(loops, zval, tag, faces_up):
            """Тріангулює ОБЛАСТЬ, обмежену петлями `loops` (точні вершини стінок),
            на висоті zval. -YY зберігає бордюр → шов закривається."""
            from shapely.geometry import Polygon as _P
            from shapely.ops import unary_union as _uu
            polys = []
            for lp in loops:
                try:
                    p = _P(lp).buffer(0)
                    if not p.is_empty and p.area > 1e-9:
                        polys.append(p)
                except Exception:  # noqa: BLE001
                    pass
            if not polys:
                return None
            # вершини = ВСІ вершини петель (точні, зі стінок); сегменти = петлі
            vlist, vmap, seg = [], {}, []
            for lp in loops:
                idx = []
                for (x, y) in lp:
                    key = (round(x, 5), round(y, 5))
                    if key not in vmap:
                        vmap[key] = len(vlist)
                        vlist.append((x, y))
                    idx.append(vmap[key])
                for i in range(len(idx)):
                    seg.append((idx[i], idx[(i + 1) % len(idx)]))
            # дірки = представницькі точки внутрішніх петель (якщо кільце в кільці)
            holes = []
            uni = _uu(polys)
            geoms = uni.geoms if hasattr(uni, "geoms") else [uni]
            for gpoly in geoms:
                for ring in gpoly.interiors:
                    ip = _P(ring).representative_point()
                    holes.append((ip.x, ip.y))
            Vc2, Fc = _run_triangle(vlist, seg, holes, f"pq30a{tri_area:g}YY", tag, work_dir)
            Vc = np.column_stack([Vc2[:, 0], Vc2[:, 1], np.full(len(Vc2), zval)])
            cap = trimesh.Trimesh(vertices=Vc, faces=Fc, process=False)
            cap.fix_normals()
            up = cap.face_normals[:, 2].mean() > 0
            if up != faces_up:
                cap.faces = cap.faces[:, ::-1]
            return cap

        # ── 2b) BORDER BRIDGE walls (inlay arm meets the zone border) ───────
        bridge_mesh = None
        if has_inlays:
            inlays_d = _densify_polygon(inlays, seg_len)
            bverts, bfaces = [], []

            def bvidx(x, y, z):
                bverts.append((x, y, z))
                return len(bverts) - 1

            for ch in _ring_chains(inlays_d):
                m = len(ch)
                for i in range(m):
                    a = ch[i]
                    b = ch[(i + 1) % m]
                    if edge_is_perim(a, b):
                        ta = bvidx(a[0], a[1], slab_z)
                        tb = bvidx(b[0], b[1], slab_z)
                        ba = bvidx(a[0], a[1], floor_z)
                        bb = bvidx(b[0], b[1], floor_z)
                        bfaces.append((ta, ba, bb))
                        bfaces.append((ta, bb, tb))
            if bverts:
                bridge_mesh = trimesh.Trimesh(vertices=np.asarray(bverts),
                                              faces=np.asarray(bfaces), process=False)

        # ── 2+3) FLOOR (slab_z) + BOTTOM (floor_z) з БОРДЮР-ПЕТЕЛЬ СТІНОК ────
        # Беремо ТОЧНІ бордюр-вершини зі стінок (+мостів) → каже шиються без шва.
        wb_parts = [m for m in (wall_mesh, bridge_mesh) if m is not None and len(m.faces) > 0]
        wb = trimesh.util.concatenate(wb_parts) if len(wb_parts) > 1 else wb_parts[0]
        wb.merge_vertices(digits_vertex=5)
        slab_loops = _z_loops(wb, slab_z)
        floor_z_loops = _z_loops(wb, floor_z)
        # підлога слотів на slab_z (нормаль ВГОРУ — дивимось у слот зверху)
        floor_mesh = _cap_from_loops(slab_loops, slab_z, "floor", faces_up=True) if slab_loops else None
        # дно плити на floor_z (нормаль ВНИЗ)
        bottom_mesh = _cap_from_loops(floor_z_loops, floor_z, "bottom", faces_up=False) if floor_z_loops else None

        parts = [m for m in (terrain_mesh, wall_mesh, floor_mesh, bottom_mesh, bridge_mesh)
                 if m is not None and len(m.faces) > 0]
        result = trimesh.util.concatenate(parts)
        result.merge_vertices(digits_vertex=5)
        try:
            result.update_faces(result.unique_faces())
            result.update_faces(result.nondegenerate_faces())
            result.remove_unreferenced_vertices()
            trimesh.repair.fix_winding(result)
            trimesh.repair.fix_inversion(result)
        except Exception:  # noqa: BLE001
            pass
        # Закрити структурні шви кап-стінка → герметично (Blender надійніший).
        if not bool(getattr(result, "is_volume", False)):
            result = _blender_fill_watertight(result, work_dir)
        return result
    finally:
        try:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        _worker(sys.argv[2], sys.argv[3])
