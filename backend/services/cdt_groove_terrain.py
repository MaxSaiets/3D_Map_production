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
    for _attempt in range(4):
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
        # rc=0xC0000142/0xC0000005 без stderr = субпроцес НЕ СТАРТУВАВ (DLL init /
        # тиск памʼяті під час важкої генерації) — ТРАНЗІЄНТНО. Пауза + gc дає ОС
        # звільнити памʼять/хендли перед новою спробою (без паузи всі ретраї падали
        # підряд за мс → CDT хибно валився у boolean-гребінець на реальних зонах).
        import gc as _gc
        import time as _time
        _gc.collect()
        _time.sleep(1.5 * (_attempt + 1))
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
bpy.ops.mesh.remove_doubles(threshold=1e-3)
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_non_manifold()
bpy.ops.mesh.fill_holes(sides=0)
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


def _rings_pslg(rings, snap=1e-4):
    """Manual PSLG from a list of closed rings [[(x,y),...],...] → (pts, segs).
    РОБАСТНО для `triangle` (інакше «invalid geometry on input» на реальних масках):
    snap координат до сітки + ГЛОБАЛЬНИЙ dedup вершин (ті самі координати між
    кільцями = ОДИН індекс, не дубль) + викид вироджених (нульова довжина) сегментів.
    Спільні вершини між викликами зберігаються детерміновано → шви закриваються."""
    pts, segs = [], []
    index = {}

    def _key(c):
        return (round(float(c[0]) / snap), round(float(c[1]) / snap))

    for ring in rings:
        # 1) зняти послідовні дублі + замикаючу копію (snap-простір)
        cleaned = []
        for c in ring:
            k = _key(c)
            if cleaned and cleaned[-1][0] == k:
                continue
            cleaned.append((k, c))
        if len(cleaned) > 1 and cleaned[0][0] == cleaned[-1][0]:
            cleaned.pop()
        if len(cleaned) < 3:
            continue  # вироджене кільце — пропустити
        # 2) глобальний dedup → індекси
        ids = []
        for k, c in cleaned:
            j = index.get(k)
            if j is None:
                j = len(pts)
                index[k] = j
                pts.append((float(c[0]), float(c[1])))
            ids.append(j)
        # 3) сегменти, без нульової довжини
        n = len(ids)
        for i in range(n):
            a, b = ids[i], ids[(i + 1) % n]
            if a != b:
                segs.append((a, b))
    return pts, segs


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

        # ── САНІТАЦІЯ inlays (КРИТИЧНО проти triangle «invalid geometry on input» на
        # реальних OSM-масках) ── реальні дорожні маски = сотні полігонів, що
        # самоперетинаються / торкаються одне одного / торкаються межі зони → сирий
        # PSLG невалідний → triangle падає → fallback boolean (старий гребінець).
        # Фікс: unary_union+buffer(0) (злити дотичні, прибрати самоперетини) + СТРОГИЙ
        # інсет від межі зони (розірвати збіг з zone_ext) + simplify (зняти майже-дублі
        # колінеарні вершини). Дороги біля краю клипляться на ~0.3м (тонка смужка терену).
        if has_inlays:
            try:
                _geoms = list(inlays.geoms) if isinstance(inlays, MultiPolygon) else [inlays]
                _clean = unary_union([g.buffer(0) for g in _geoms
                                      if g is not None and not g.is_empty]).buffer(0)
                _area_before = float(getattr(_clean, "area", 0.0) or 0.0)
                _inset = max(seg_len * 0.15, 0.25)
                _inset_res = _clean.intersection(zone.buffer(-_inset))
                # ЗАПОБІЖНИК: якщо інсет стер >85% площі (інлеї здебільшого вздовж/поперек
                # межі тайла — напр. річка чи парк уздовж краю) → НЕ інсетимо (лишаємо
                # у межах zone), інакше груви біля краю зникають / шов відкривається.
                _area_after = float(getattr(_inset_res, "area", 0.0) or 0.0)
                if _area_before > 1e-6 and _area_after < 0.15 * _area_before:
                    print(f"[CDT] inset collapsed inlays ({_area_after:.0f}<15% of {_area_before:.0f}) "
                          f"→ skip inset (clip to zone only)")
                    inlays = _clean.intersection(zone)
                else:
                    inlays = _inset_res
                if inlays is not None and not inlays.is_empty:
                    inlays = inlays.simplify(max(seg_len * 0.2, 0.3)).buffer(0)
                has_inlays = inlays is not None and not inlays.is_empty
            except Exception as _sanex:  # noqa: BLE001
                print(f"[CDT] inlay sanitize failed (continue raw): {_sanex}")
                has_inlays = inlays is not None and not inlays.is_empty

        # ── ЩІЛЬНІСТЬ масштабована до площі зони (інакше великі зони (1.5км/model_150)
        # дають ~929к-гранний терен → 8хв ген + boolean парків/води ламає герметичність
        # на велетенському меші + ризик OOM на 4ГБ). Терен обмежуємо ~150к трикутників;
        # ПЛОСКІ капи (bottom/floor) грубо (їм щільність не потрібна) ~6к. Малі зони
        # лишаються як були (max з tri_area). ──
        _zone_area = max(float(zone.area), 1.0)
        _top_area = max(float(tri_area), _zone_area / 150000.0)
        _cap_area = max(float(tri_area) * 6.0, _zone_area / 6000.0)

        # ── 1) TERRAIN top: CDT з ЯВНИМИ обмежувальними кільцями (НЕ difference) ──
        # Спільна densified геометрія: zone_ext + inlay_rings використовуються І для
        # terrain-дірок, І для floor/bottom-капів → ТІ САМІ вершини → герметично за
        # побудовою (надійніше за loop-extraction, який фрагментує на щільних мережах).
        SP_zone = zone
        zone_ext = _densify_ring(list(SP_zone.exterior.coords)[:-1], seg_len)
        inlay_rings, inlay_hole_pts, block_hole_pts = [], [], []
        inlays_d = None
        if has_inlays:
            # densify ОДИН раз → ТІ САМІ кільця для terrain-дірок, floor-капу та bridge.
            inlays_d = _densify_polygon(inlays, seg_len)
            for gp in (inlays_d.geoms if isinstance(inlays_d, MultiPolygon) else [inlays_d]):
                if getattr(gp, "is_empty", True) or gp.area < 1e-9:
                    continue
                ext = [(c[0], c[1]) for c in list(gp.exterior.coords)[:-1]]
                if len(ext) < 3:
                    continue
                inlay_rings.append(ext)
                # ⭐КРИТИЧНО (root cause «в центрі рельєфу немає»): зв'язна дорожня сітка
                # після unary_union = ОДИН полігон, чий exterior ≈ вся зона, а МІСЬКІ
                # КВАРТАЛИ = ДІРКИ (interiors). Якщо interiors НЕ додати як обмежувальні
                # кільця — hole-point заливає ВЕСЬ інтер'єр екстер'єра → усі квартали
                # зникають у slab-плиту, рельєф лишається лише тонкою смужкою по краю.
                # Interior-кільця = constraints (терен зупиняється на межі кварталу);
                # для floor запам'ятовуємо точку в КОЖНОМУ кварталі (щоб slab не капив його).
                for it in gp.interiors:
                    ir = [(c[0], c[1]) for c in list(it.coords)[:-1]]
                    if len(ir) >= 3:
                        inlay_rings.append(ir)
                        try:
                            _bp = Polygon(it).representative_point()
                            block_hole_pts.append((_bp.x, _bp.y))
                        except Exception:  # noqa: BLE001
                            pass
                rp = gp.representative_point()  # ГАРАНТОВАНО у дорожній смузі (не в кварталі)
                inlay_hole_pts.append((rp.x, rp.y))
        t_pts, t_segs = _rings_pslg([zone_ext] + inlay_rings)
        Vt2, Ft = _run_triangle(t_pts, t_segs, inlay_hole_pts, f"pq30a{_top_area:g}YY", "terrain", work_dir)
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

        # ── 2) FLOOR at slab_z: ЛИШЕ дорожня смуга (НЕ квартали) → ТІ САМІ вершини ──
        floor_mesh = None
        if has_inlays and inlay_rings:
            f_pts, f_segs = _rings_pslg(inlay_rings)  # snapped → збіг з terrain/walls
            # hole-точки у кварталах (interiors) → slab НЕ капить квартали (вони = терен)
            Vf2, Ff = _run_triangle(f_pts, f_segs, block_hole_pts, f"pq30a{_cap_area:g}YY", "floor", work_dir)
            # ЗРІЗ: triangle лишає трикутники у ввігнутостях між exterior і опуклою
            # оболонкою → тримаємо ЛИШЕ ті, чий центроїд у дорожній смузі inlays_d,
            # інакше slab-плита вилазить за межі доріг (відкриті ребра / не-герметично).
            if inlays_d is not None and len(Ff):
                try:
                    from shapely.prepared import prep as _prep
                    _pin = _prep(inlays_d)
                    _cen = Vf2[Ff].mean(axis=1)
                    _keep = np.fromiter(
                        (_pin.contains(Point(float(x), float(y))) for x, y in _cen[:, :2]),
                        dtype=bool, count=len(Ff))
                    if _keep.any():
                        Ff = Ff[_keep]
                except Exception:  # noqa: BLE001
                    pass
            Vf = np.column_stack([Vf2[:, 0], Vf2[:, 1], np.full(len(Vf2), slab_z)])
            floor_mesh = trimesh.Trimesh(vertices=Vf, faces=Ff, process=False)
            try:
                floor_mesh.remove_unreferenced_vertices()
            except Exception:  # noqa: BLE001
                pass
            floor_mesh.fix_normals()
            if floor_mesh.face_normals[:, 2].mean() < 0:  # підлога слота дивиться ВГОРУ
                floor_mesh.faces = floor_mesh.faces[:, ::-1]

        # ── 3) BOTTOM at floor_z: CDT zone_ext (= terrain-периметр → ТІ САМІ вершини) ──
        b_pts, b_segs = _rings_pslg([zone_ext])
        Vb2, Fb = _run_triangle(b_pts, b_segs, [], f"pq30a{_cap_area:g}YY", "bottom", work_dir)
        Vb = np.column_stack([Vb2[:, 0], Vb2[:, 1], np.full(len(Vb2), floor_z)])
        bottom_mesh = trimesh.Trimesh(vertices=Vb, faces=Fb, process=False)
        bottom_mesh.fix_normals()
        if bottom_mesh.face_normals[:, 2].mean() > 0:  # дно плити дивиться ВНИЗ
            bottom_mesh.faces = bottom_mesh.faces[:, ::-1]

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
        # 1) Закрити структурні шви кап-стінка СПЕРШУ (Blender зшиває стінки↔дно↔підлогу).
        if not bool(getattr(result, "is_volume", False)):
            result = _blender_fill_watertight(result, work_dir)
        # 2) ГОЛОВНИЙ солід: на складних масках/межах ПІСЛЯ зшивання інколи лишається
        # ОКРЕМИЙ компонент (напр. плоский bottom-аркуш vol≈0, що не злився, або дрібний
        # острівець після інсету) → беремо найбільший ГЕРМЕТИЧНИЙ компонент (повний
        # рельєф-солід); якщо герметичних нема — найбільший за гранями.
        try:
            comps = result.split(only_watertight=False)
        except Exception:  # noqa: BLE001
            comps = [result]
        if len(comps) > 1:
            def _topface_area(_m):
                try:
                    _n = _m.face_normals
                    return float(_m.area_faces[_n[:, 2] > 0.5].sum())
                except Exception:  # noqa: BLE001
                    return 0.0
            _full_top = _topface_area(result)
            wt_comps = [c for c in comps if bool(getattr(c, "is_volume", False))]
            _cand = (max(wt_comps, key=lambda c: abs(float(c.volume)))
                     if wt_comps else max(comps, key=lambda c: len(c.faces)))
            # ЗАПОБІЖНИК: keep-largest НЕ має викинути РЕЛЬЄФ (терен-острови у парк-важких
            # зонах). ЕТАЛОН = ПЛОЩА ЗОНИ (не сумарний top-area результату!): Blender-fill
            # інколи заливає ДРУГУ «кришку» поверх усього (top-area → 2× зони) → порівняння
            # з full_top хибно блокувало вибір ПРАВИЛЬНОГО герметичного компонента → далі
            # негерметичний меш ламав manifold-виріз паза (Blender-виріз нищив дно/паз:
            # «пази не створились, підложки немає»). Кандидат ОК, якщо покриває ≥55% зони.
            _zone_ref = float(zone.area)
            _cand_top = _topface_area(_cand)
            # ⚠️КРИТИЧНО: кандидат мусить СЯГАТИ ДНА ПЛИТИ (floor_z). Коли шов
            # стінка↔низ не злився, меш = ВЕРХНЄ тіло (терен+стінки, zmin=slab) +
            # НИЖНЯ коробка slab→floor. Вибір верхнього ВИКИДАВ ПІДЛОЖКУ → модель
            # без низу, дно=slab → «будинки в повітрі» (extend-до-floor брав slab),
            # дороги над пусткою (юзер-кейс fe979dea, 40/215 будинків плавали).
            try:
                _cand_reaches_floor = float(_cand.bounds[0][2]) <= float(floor_z) + 0.05
            except Exception:  # noqa: BLE001
                _cand_reaches_floor = False
            if not _cand_reaches_floor:
                print(f"[CDT] keep-largest candidate misses the base plate "
                      f"(zmin={float(_cand.bounds[0][2]):.2f} > floor {float(floor_z):.2f}) "
                      f"→ keep full + seal (не викидати підложку)")
            elif _zone_ref > 1e-6 and _cand_top >= 0.55 * _zone_ref:
                result = _cand
                try:
                    result.remove_unreferenced_vertices()
                except Exception:  # noqa: BLE001
                    pass
            elif _full_top > 1e-6 and _cand_top >= 0.8 * _full_top:
                result = _cand
                try:
                    result.remove_unreferenced_vertices()
                except Exception:  # noqa: BLE001
                    pass
            else:
                print(f"[CDT] keep-largest would drop terrain "
                      f"(top-area {_cand_top:.0f} < 55% zone {_zone_ref:.0f}) → keep full + seal")
        # 3) Якщо все ще не герметично (вибрали негерметичний найбільший) — досшити.
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
