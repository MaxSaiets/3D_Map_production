# -*- coding: utf-8 -*-
"""Геометрія режиму «Гори»: поле висот → герметичне тіло без булевих.

Одна функція будує все тіло: верх (сітка рельєфу), бічні стінки (гладкі або скельні),
ободок (плаский з фаскою / заокруглений / без ободка), низ. Спільні вершини по контуру ⇒
герметично за побудовою; плитки — зріз того самого тіла площинами, тому шви збігаються.

Похилі боки («slope») робляться НЕ стінкою, а полем висот: зовнішня смуга рельєфу опускається
до ободка профілем (1−t)^1.7 зі справжнім мікрорельєфом і скельними фасетами (без нависань).
"""
from __future__ import annotations

import struct
import zipfile
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter


# ── шуми ─────────────────────────────────────────────────────────────────────
def _value_noise(s, z, cell, seed):
    rng = np.random.default_rng(seed)
    gi, gj = np.asarray(s) / cell, np.asarray(z) / cell
    i0, j0 = np.floor(gi).astype(int), np.floor(gj).astype(int); fi, fj = gi - i0, gj - j0
    fi, fj = fi * fi * (3 - 2 * fi), fj * fj * (3 - 2 * fj)
    i0 -= i0.min(); j0 -= j0.min()
    grid = rng.random((int(i0.max()) + 3, int(j0.max()) + 3))
    return (grid[i0, j0] * (1 - fi) * (1 - fj) + grid[i0 + 1, j0] * fi * (1 - fj)
            + grid[i0, j0 + 1] * (1 - fi) * fj + grid[i0 + 1, j0 + 1] * fi * fj)


def _ridged(s, z, cs, cz, seed):
    n = _value_noise(np.asarray(s) / cs * cz, z, cz, seed)
    return 1 - np.abs(2 * n - 1)


# ── скельні стінки (вертикальні боки) ─────────────────────────────────────────
def wall_depth_matrix(P, s_arc, Zrows, z_top, z_bot, seed=7, depth_mm=6.0):
    """Глибина заглиблення стінки всередину (rows × m): ребра/жолоби, витягнуті по вертикалі,
    великі випуклості, рвані уступи. Нависання обмежені 45° (Δd ≤ Δz знизу вгору)."""
    rows, m = Zrows.shape
    S = np.broadcast_to(s_arc, (rows, m)); Zf = Zrows
    x = np.broadcast_to(P[:, 0], (rows, m)); y = np.broadcast_to(P[:, 1], (rows, m))
    crag = (4.0 * _ridged(S, Zf, 14.0, 34.0, seed + 1) + 1.8 * _ridged(S, Zf, 6.0, 14.0, seed + 2)
            + 0.6 * _value_noise(S, Zf, 2.0, seed + 3))
    swell = 2.2 * _value_noise(S, Zf, 45.0, seed + 4)
    u = Zf + 0.08 * x - 0.08 * y
    rng = np.random.default_rng(seed); th = rng.uniform(12.0, 22.0, 400); bounds = np.cumsum(th) - 800.0
    k = np.searchsorted(bounds, u); du = u - bounds[np.clip(k - 1, 0, 399)]
    brk = np.clip((_value_noise(S, Zf, 18.0, seed + 5) - 0.35) / 0.4, 0, 1)
    ledge = 1.6 * np.clip(du / 6.0, 0, 1) ** 0.7 * rng.uniform(0.3, 1.0, 400)[np.clip(k, 0, 399)] * brk
    d = crag + swell + ledge
    d = d - np.percentile(d, 3); d = np.clip(d * (depth_mm / max(np.percentile(d, 97), 1e-6)), 0, depth_mm * 1.15)
    top_t = np.clip((z_top[None, :] - Zf) / 6.0, 0, 1); top_t = top_t * top_t * (3 - 2 * top_t)
    bot_t = np.clip((Zf - z_bot[None, :]) / 3.0, 0, 1); bot_t = bot_t * bot_t * (3 - 2 * bot_t)
    d = d * top_t * bot_t
    for r in range(rows - 2, -1, -1):
        d[r] = np.maximum(d[r], d[r + 1] - np.abs(Zf[r] - Zf[r + 1]))
    return d


# ── тіло ─────────────────────────────────────────────────────────────────────
def frame_solid(xs, ys, Z, fw=10.0, fh=40.0, ch=4.0, fillet: Optional[float] = None, segs=12, foot=0.6,
                wall_rows=120, wall_rock=False, wall_depth_mm=6.0):
    """→ (V, F, info). Координати: плита 0..S+2fw, рельєф у центрі, зсунутий на fw.
    fw=0 → без ободка (fh тоді = товщина під рельєфом, стінка від краю рельєфу до z=0)."""
    ny, nx = Z.shape; X, Y = np.meshgrid(xs, ys); S = xs[-1] - xs[0]
    top = np.column_stack([X.ravel() + fw, Y.ravel() + fw, Z.ravel() + fh])
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    faces = [np.column_stack([a, b, c]), np.column_stack([a, c, d])]
    ring = np.concatenate([idx[0, :-1], idx[:-1, -1], idx[-1, ::-1][:-1], idx[::-1, 0][:-1]]); m = len(ring)
    P = top[ring][:, :2]; cxy = np.array([S / 2 + fw, S / 2 + fw])
    scale = lambda dd: cxy + (P - cxy) * ((S / 2 + dd) / (S / 2))  # noqa: E731
    rings = [top[ring]]
    z_wall_bottom = fh if fw > 0 else 0.0
    K = wall_rows if wall_rock else 1
    if wall_rock:
        eps = 1e-6; inward = np.zeros((m, 2))
        inward[np.abs(P[:, 1] - fw) < eps] += (0, 1); inward[np.abs(P[:, 0] - (S + fw)) < eps] += (-1, 0)
        inward[np.abs(P[:, 1] - (S + fw)) < eps] += (0, -1); inward[np.abs(P[:, 0] - fw) < eps] += (1, 0)
        inward /= np.linalg.norm(inward, axis=1, keepdims=True)
        s_arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        z_top = top[ring][:, 2]
        T = (1 - np.arange(1, K) / K)[:, None]
        Zrows = z_wall_bottom + T * (z_top[None, :] - z_wall_bottom)
        D = wall_depth_matrix(P, s_arc, Zrows, z_top, np.full(m, z_wall_bottom), depth_mm=wall_depth_mm)
        for r in range(K - 1):
            rings.append(np.column_stack([P + inward * D[r][:, None], Zrows[r]]))
    if fw > 0:
        rings.append(np.column_stack([P, np.full(m, fh)]))
        if fillet:
            for t in np.linspace(0, np.pi / 2, segs + 1):
                rings.append(np.column_stack([scale(fw - fillet + fillet * np.sin(t)), np.full(m, fh - fillet + fillet * np.cos(t))]))
        else:
            rings.append(np.column_stack([scale(fw - ch), np.full(m, fh)]))
            rings.append(np.column_stack([scale(fw), np.full(m, fh - ch)]))
        if foot:
            rings.append(np.column_stack([scale(fw), np.full(m, foot)]))
            rings.append(np.column_stack([scale(fw - foot), np.zeros(m)]))
        else:
            rings.append(np.column_stack([scale(fw), np.zeros(m)]))
    else:
        if foot:
            rings.append(np.column_stack([P, np.full(m, foot)]))
            rings.append(np.column_stack([scale(-foot), np.zeros(m)]))
        else:
            rings.append(np.column_stack([P, np.zeros(m)]))
    base = nx * ny; V = [top]; ids = [ring]
    for k, R in enumerate(rings[1:]):
        V.append(R); ids.append(base + k * m + np.arange(m))
    ctr = base + (len(rings) - 1) * m; V.append(np.array([[S / 2 + fw, S / 2 + fw, 0.0]]))
    for r0, r1 in zip(ids[:-1], ids[1:]):
        r0n, r1n = np.roll(r0, -1), np.roll(r1, -1)
        faces.append(np.column_stack([r0, r1, r1n])); faces.append(np.column_stack([r0, r1n, r0n]))
    last = ids[-1]; faces.append(np.column_stack([last, np.full(m, ctr), np.roll(last, -1)]))
    info = {"terrain_faces": 2 * (nx - 1) * (ny - 1), "wall_faces": 2 * m * K}
    return np.vstack(V), np.vstack(faces), info


# ── похилі боки як поле висот ────────────────────────────────────────────────
def apply_slope_band(Zmm: np.ndarray, cell: float, band_mm: float, base_mm: float, seed: int = 7) -> np.ndarray:
    """Зовнішня смуга band_mm опускається до base_mm профілем (1−t)^1.7 зі збереженням дрібного
    рельєфу + скельні фасети вздовж нормалі. Все лишається полем висот ⇒ без нависань."""
    G = Zmm.shape[0]; inner = (G - 1) * cell; core = inner - 2 * band_mm
    xs = np.arange(G) * cell; X, Y = np.meshgrid(xs, xs)
    dx = np.maximum(np.abs(X - inner / 2) - core / 2, 0); dy = np.maximum(np.abs(Y - inner / 2) - core / 2, 0)
    tt = np.clip(np.hypot(dx, dy) / band_mm, 0, 1); band = tt > 0
    ang0 = np.arctan2(Y - inner / 2, X - inner / 2); s_edge = (ang0 + np.pi) / (2 * np.pi) * (4 * core)
    warp = (3.2 * (_ridged(s_edge, tt * 22.0, 11.0, 9.0, seed + 21) - 0.5) + 1.3 * (_ridged(s_edge * 1.3 + 77, tt * 22.0, 4.5, 4.0, seed + 22) - 0.5)
            + 0.8 * (_value_noise(s_edge + 500, tt * 22.0, 2.0, seed + 23) - 0.5))
    tt_w = np.clip(tt + warp * 4 * tt * (1 - tt) / band_mm, 0, 1); wprof = (1 - tt_w) ** 1.7
    LP = gaussian_filter(Zmm, max(8.0 / cell, 1.0)); HP = Zmm - LP
    crag = (2.6 * _ridged(X, Y, 10.0, 10.0, seed + 11) + 1.2 * _ridged(X * 1.7 + 50, Y * 0.6, 4.0, 4.0, seed + 12) + 0.5 * _value_noise(X, Y, 1.5, seed + 13))
    crag = (crag - crag.mean()) * 0.5 * 4 * wprof * (1 - wprof)
    talus = 1.2 * (_value_noise(X + 300, Y + 300, 3.0, seed + 14) - 0.5) * (1 - wprof)
    Zs = base_mm + (LP - base_mm) * wprof + HP * (0.55 + 0.45 * wprof) + crag + talus
    gy0, gx0 = np.gradient(gaussian_filter(Zs, 2.0), cell); cosA = 1 / np.sqrt(1 + gx0 ** 2 + gy0 ** 2)
    facet = (2.5 * (_ridged(X + 900, Y + 900, 9.0, 9.0, seed + 31) - 0.5) + 1.0 * (_ridged(X * 0.7 + 400, Y * 1.4 + 200, 4.0, 4.0, seed + 32) - 0.5)
             + 0.4 * (_value_noise(X + 50, Y + 70, 1.5, seed + 33) - 0.5))
    wgt = np.clip(tt / 0.15, 0, 1) * np.clip((1 - tt) / 0.25, 0, 1)
    Zs = Zs + facet * np.minimum(1 / cosA, 6.0) * wgt
    out = np.where(band, np.maximum(Zs, base_mm), Zmm)
    sh = np.clip((core / 2 - np.maximum(np.abs(X - inner / 2), np.abs(Y - inner / 2))) / 4.0, 0, 1)
    sh = np.where(band, 1.0, sh); sh = sh * sh * (3 - 2 * sh)
    return np.where(band, out, out - 1.5 * (1 - sh))


# ── потокові записи ───────────────────────────────────────────────────────────
def write_3mf(V, F, path, name="Model"):
    """Один обʼєкт, мм; XML пишеться шматками (без збирання 150-МБ рядка в памʼяті)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        with z.open("3D/3dmodel.model", "w") as f:
            f.write(('<?xml version="1.0" encoding="UTF-8"?><model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
                     f'<metadata name="Title">{name}</metadata><resources><object id="1" type="model" name="{name}"><mesh><vertices>').encode())
            for i in range(0, len(V), 100000):
                f.write("".join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z_:.4f}"/>' for x, y, z_ in V[i:i + 100000]).encode())
            f.write(b"</vertices><triangles>")
            for i in range(0, len(F), 100000):
                f.write("".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in F[i:i + 100000]).encode())
            f.write(b'</triangles></mesh></object></resources><build><item objectid="1"/></build></model>')


def write_3mf_multi(objects, path):
    """Кілька обʼєктів (name, V, F) в одному 3MF потоково — для збірки «гора + фігурки» без
    trimesh.Scene.export (той збирає весь XML у памʼяті: +1 ГБ на 900k граней)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        with z.open("3D/3dmodel.model", "w") as f:
            f.write(b'<?xml version="1.0" encoding="UTF-8"?><model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>')
            for oid, (name, V, F) in enumerate(objects, start=1):
                f.write(f'<object id="{oid}" type="model" name="{name}"><mesh><vertices>'.encode())
                for i in range(0, len(V), 100000):
                    f.write("".join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z_:.4f}"/>' for x, y, z_ in V[i:i + 100000]).encode())
                f.write(b"</vertices><triangles>")
                for i in range(0, len(F), 100000):
                    f.write("".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in F[i:i + 100000]).encode())
                f.write(b"</triangles></mesh></object>")
            f.write(b"</resources><build>" + "".join(f'<item objectid="{i}"/>' for i in range(1, len(objects) + 1)).encode() + b"</build></model>")


def write_stl(V, F, path):
    V = np.asarray(V, dtype=np.float32); F = np.asarray(F)
    with open(path, "wb") as f:
        f.write(b"\0" * 80); f.write(struct.pack("<I", len(F)))
        for i in range(0, len(F), 200000):
            tri = V[F[i:i + 200000]]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-12)
            rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]); rec["n"] = n; rec["v"] = tri; f.write(rec.tobytes())


# ── плитки ───────────────────────────────────────────────────────────────────
def tile_cuts(size_mm: float, bed_mm: float) -> list[float]:
    """Лінії розрізу (мм) так, щоб кожна плитка ≤ bed_mm, а центральна містила середину
    (вершина зазвичай у центрі): 1 плитка ≤ bed; інакше 3 з цілою серединою; далі рівні."""
    if size_mm <= bed_mm:
        return []
    mid = min(bed_mm * 0.95, size_mm * 0.5)
    side = (size_mm - mid) / 2
    if side <= bed_mm:
        return [round(side, 1), round(size_mm - side, 1)]
    n = int(np.ceil(size_mm / (bed_mm * 0.95)))
    return [round(size_mm * k / n, 1) for k in range(1, n)]


def split_tiles(mesh, cuts: list[float], size_mm: float):
    """Ріже trimesh площинами x/y = cuts → dict name → Trimesh (кожен зсунутий у нуль)."""
    if not cuts:
        mesh.metadata["origin"] = np.zeros(3); return {"FULL": mesh}
    def cut(m, axis, pos):
        n = np.zeros(3); n[axis] = 1; o = [pos if axis == 0 else 0, pos if axis == 1 else 0, 0]
        lo = m.slice_plane(o, -n, cap=True); hi = m.slice_plane(o, n, cap=True)
        for q in (lo, hi):
            q.merge_vertices(); q.fix_normals()
        return lo, hi
    def strips(m, axis):
        out, rest = [], m
        for c in cuts:
            lo, nxt = cut(rest, axis, c); out.append(lo)
            if rest is not m:
                del rest
            rest = nxt
        out.append(rest); return out
    ncol = len(cuts) + 1
    names_x = ["W", "", "E"] if ncol == 3 else [f"c{i+1}" for i in range(ncol)]
    names_y = ["S", "", "N"] if ncol == 3 else [f"r{i+1}" for i in range(ncol)]
    tiles = {}
    for ix, col in enumerate(strips(mesh, 0)):
        for iy, t in enumerate(strips(col, 1)):
            nm = (names_y[iy] + names_x[ix]) or "CENTER"
            if ncol != 3:
                nm = f"{names_y[iy]}{names_x[ix]}"
            origin = t.bounds[0].copy(); t.apply_translation(-origin)
            t.metadata["origin"] = origin; tiles[nm] = t
    return tiles


# ── превʼю GLB (легкий, з текстурою) ─────────────────────────────────────────
def preview_glb(V, F, path, size_mm: float, fw: float, texture_rgb: Optional[np.ndarray] = None, max_faces: int = 160000, mirror_x: bool = True):
    """Спрощений GLB для вʼюера: рельєф децимується, текстура (рядок 0 = південь) проєктується зверху."""
    import trimesh
    m = trimesh.Trimesh(V, F, process=False)
    if len(m.faces) > max_faces:
        try:
            import open3d as o3d  # noqa: F401
            om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
            om = om.simplify_quadric_decimation(max_faces); m = trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles), process=False)
        except Exception:
            m = m.simplify_quadric_decimation(face_count=max_faces) if hasattr(m, "simplify_quadric_decimation") else m
    m.fix_normals()
    if texture_rgb is not None:
        from PIL import Image
        img = Image.fromarray(np.ascontiguousarray(texture_rgb[::-1]))          # PIL: рядок 0 = верх = північ
        inner = size_mm - 2 * fw
        uv = np.column_stack([(m.vertices[:, 0] - fw) / inner, (m.vertices[:, 1] - fw) / inner])
    if mirror_x:
        # Model3DViewer сайту дзеркалить X кожного GLB (спадок мап) — заздалегідь дзеркалимо,
        # щоб у вʼюері схід лишився сходом; UV уже пораховані до дзеркалення.
        m.vertices[:, 0] = size_mm - m.vertices[:, 0]; m.invert()
    if texture_rgb is not None:
        # явний PBR: metallic=0, інакше вʼюер без HDRI показує текстуру майже чорною
        mat = trimesh.visual.material.PBRMaterial(baseColorTexture=img, metallicFactor=0.0, roughnessFactor=0.9, doubleSided=True)
        m.visual = trimesh.visual.TextureVisuals(uv=np.clip(uv, 0, 1), material=mat)
    m.export(path)
    return len(m.faces)
