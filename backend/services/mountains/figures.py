# -*- coding: utf-8 -*-
"""Фігурки (люди, хатина) на рельєфі: бібліотека + посадка.

Правила посадки (перевірені на Матергорні власника, scene_v10):
  * standing  — стоїть вертикально; ґрунт під підошвами мікровирівнюється (радіус ≈ 0.35·ширини,
                нахил ≤ 25°), підошви на 0.35 мм у ґрунті, обличчям униз по схилу (до глядача);
  * building  — ґрунт вирівнюється до ≤ 11° у радіусі ≈ 0.6·ширини з плавним переходом і
                збереженою дрібною фактурою; цоколь (plinth_frac) у землі; фасад (−Y) униз по схилу;
  * climbing  — шукає найкрутішу стінку в радіусі 6 % плити від точки; тіло вздовж стіни,
                обличчям (−Y) до скелі; зсув по нормалі, доки найглибша точка = −0.3 мм (зрощення);
                канат: від маківки вгору по стіні до анкера + униз від рук.
Усі зміни рельєфу робляться у ПОЛІ ВИСОТ до побудови тіла ⇒ герметичність не страждає.
Фігурки експортуються окремими обʼєктами; у плитки зрощуються manifold-union (з відкатом на
окремі обʼєкти, якщо union не вдався).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter

ASSETS = Path(os.getenv("MOUNTAIN_FIGURES_DIR", Path(__file__).resolve().parents[2] / "assets" / "figures"))
_smooth = lambda t: t * t * (3 - 2 * t)  # noqa: E731


def library() -> list[dict]:
    p = ASSETS / "manifest.json"
    if not p.exists():
        return []
    return [f for f in json.loads(p.read_text(encoding="utf-8"))["figures"] if (ASSETS / f["file"]).exists()]


def get(fid: str) -> Optional[dict]:
    return next((f for f in library() if f["id"] == fid), None)


def zat_on(Z, x, y, cell):
    G = Z.shape[0]; fx = np.clip(x / cell, 0, G - 1.001); fy = np.clip(y / cell, 0, G - 1.001)
    i, j = int(fx), int(fy); a, b = fx - i, fy - j
    return float(Z[j, i] * (1 - a) * (1 - b) + Z[j, i + 1] * a * (1 - b) + Z[j + 1, i] * (1 - a) * b + Z[j + 1, i + 1] * a * b)


def grade(Z, cell, cx, cy, r_flat, r_blend, clamp_deg, detail_k=0.6, pct=55):
    """Локальна площина (ухил ≤ clamp) у радіусі r_flat, перехід r_blend, дрібна фактура лишається.
    → (Zs, down_dir(2), slope_deg, max_change)."""
    G = Z.shape[0]; xs = np.arange(G) * cell; X, Y = np.meshgrid(xs, xs)
    sel = (X - cx) ** 2 + (Y - cy) ** 2 <= r_flat ** 2
    if sel.sum() < 6:
        return Z, np.array([0.0, -1.0]), 0.0, 0.0
    A = np.column_stack([X[sel] - cx, Y[sel] - cy, np.ones(sel.sum())])
    coef, *_ = np.linalg.lstsq(A, Z[sel], rcond=None)
    g = coef[:2]; gn = float(np.linalg.norm(g)) + 1e-9; gs = g / gn * min(gn, np.tan(np.radians(clamp_deg)))
    z0 = float(np.percentile(Z[sel], pct)); plane = z0 + gs[0] * (X - cx) + gs[1] * (Y - cy)
    detail = Z - gaussian_filter(Z, max(2.0 / cell, 0.5))
    d = np.hypot(X - cx, Y - cy); w = 1 - _smooth(np.clip((d - r_flat) / r_blend, 0, 1))
    Zs = w * (plane + detail_k * detail) + (1 - w) * Z
    return Zs, -g / gn, float(np.degrees(np.arctan(gn))), float(np.abs(Zs - Z).max())


def _rotz(deg):
    import trimesh
    return trimesh.transformations.rotation_matrix(np.radians(deg), [0, 0, 1])


def _load(f: dict, height_mm: float):
    import trimesh
    m = trimesh.load(ASSETS / f["file"]); m.apply_scale(height_mm / 100.0)
    return m


def place_standing(Zmm, cell, f, x, y, height_mm, log):
    m = _load(f, height_mm); w = max(m.extents[0], m.extents[1])
    Zs, down, sl, ch = grade(Zmm, cell, x, y, 0.35 * w + 1.0, 0.4 * w + 1.0, 25.0, detail_k=0.4, pct=60)
    ang = np.degrees(np.arctan2(down[0], -down[1]))                      # −Y (обличчя) → униз по схилу
    m.apply_transform(_rotz(ang))
    soles = m.vertices[m.vertices[:, 2] < 0.02 * height_mm]
    zf = max(zat_on(Zs, x + p[0], y + p[1], cell) for p in soles[::max(1, len(soles) // 400)]) - 0.35
    m.apply_translation([x, y, zf])
    log(f"[FIG] {f['id']} стоїть на ({x:.0f},{y:.0f}) z={zf:.1f}: схил {sl:.0f}°→≤25°, зміна ґрунту {ch:.1f} мм")
    return Zs, m, None


def place_building(Zmm, cell, f, x, y, height_mm, log):
    m = _load(f, height_mm); w = max(m.extents[0], m.extents[1]); d = m.extents[1]
    Zs, down, sl, ch = grade(Zmm, cell, x, y, 0.55 * w, 0.4 * w, 11.0)
    ang = np.degrees(np.arctan2(down[0], -down[1])); m.apply_transform(_rotz(ang))
    edge_dn = np.array([x, y]) + down * (d / 2 - 1.0)
    floor_z = zat_on(Zs, *edge_dn, cell) - float(f.get("plinth_frac", 0.1)) * height_mm
    m.apply_translation([x, y, floor_z])
    log(f"[FIG] {f['id']} на ({x:.0f},{y:.0f}): схил {sl:.0f}°→≤11°, поворот {ang:.0f}°, зміна ґрунту {ch:.1f} мм")
    return Zs, m, None


def _steepest_near(Zmm, cell, x, y, radius):
    G = Zmm.shape[0]; xs = np.arange(G) * cell; X, Y = np.meshgrid(xs, xs)
    gy, gx = np.gradient(gaussian_filter(Zmm, max(1.5 / cell, 0.5)), cell); s = np.hypot(gx, gy)
    sel = ((X - x) ** 2 + (Y - y) ** 2 <= radius ** 2) & (X > 0.06 * xs[-1]) & (X < 0.94 * xs[-1]) & (Y > 0.06 * xs[-1]) & (Y < 0.94 * xs[-1])
    if not sel.any():
        return x, y, np.array([0.0, 1.0]), 30.0
    k = np.argmax(np.where(sel, s, -1)); j, i = np.unravel_index(k, s.shape)
    up = np.array([gx[j, i], gy[j, i]]); n = np.linalg.norm(up) + 1e-9
    return float(X[j, i]), float(Y[j, i]), up / n, float(np.degrees(np.arctan(n)))


def place_climbing(Zmm, cell, f, x, y, height_mm, log):
    import trimesh
    from scipy.interpolate import splev, splprep
    from shapely.geometry import Point
    m = _load(f, height_mm); v = m.vertices; H = height_mm
    cx, cy, up_dir, slope = _steepest_near(Zmm, cell, x, y, 0.06 * Zmm.shape[0] * cell)
    zr = zat_on(Zmm, cx, cy, cell); s = np.radians(min(slope, 80.0))
    t3 = np.array([up_dir[0] * np.cos(s), up_dir[1] * np.cos(s), np.sin(s)]); n3 = np.array([-up_dir[0] * np.sin(s), -up_dir[1] * np.sin(s), np.cos(s)])
    feet = v[v[:, 2] < 0.06 * H].mean(0); head = v[v[:, 2] > 0.9 * H].mean(0); body = head - feet; body /= np.linalg.norm(body)
    center = (m.bounds[0] + m.bounds[1]) / 2
    R1 = trimesh.geometry.align_vectors(body, t3); best = None
    for psi in np.arange(0, 360, 4):
        R = trimesh.transformations.rotation_matrix(np.radians(psi), t3) @ R1
        fr = R[:3, :3] @ np.array([0, -1.0, 0]); sc = fr @ (-n3)
        if best is None or sc > best[0]:
            best = (sc, psi)
    R = trimesh.transformations.rotation_matrix(np.radians(best[1]), t3) @ R1
    P0 = np.array([cx, cy, zr]); vsub = v[::max(1, len(v) // 3000)]
    Tof = lambda sh: trimesh.transformations.translation_matrix(P0 + n3 * sh) @ R @ trimesh.transformations.translation_matrix(-center)  # noqa: E731
    def min_gap(sh):
        Pw = trimesh.transform_points(vsub, Tof(sh)); return float(np.min(Pw[:, 2] - np.array([zat_on(Zmm, px, py, cell) for px, py in Pw[:, :2]])))
    lo, hi = -5.0, 3 * H
    for _ in range(28):
        mid = (lo + hi) / 2
        if min_gap(mid) < -0.3:
            lo = mid
        else:
            hi = mid
    T = Tof((lo + hi) / 2); m.apply_transform(T)
    rope = None
    if f.get("rope"):
        RR = max(0.35, 0.028 * H); knot = trimesh.transform_points([v[v[:, 2] > 0.95 * H].mean(0)], T)[0]
        low = trimesh.transform_points([v[(v[:, 2] > 0.84 * H) & (v[:, 2] < 0.9 * H)].mean(0)], T)[0]
        side = np.array([up_dir[1], -up_dir[0]]); cs = np.cos(s)
        surf = lambda sd, lat=0.0: (lambda p: np.array([p[0], p[1], zat_on(Zmm, *p, cell) + RR - 0.15]))(np.array([cx, cy]) + up_dir * sd + side * lat)  # noqa: E731
        along = lambda p: float((p[:2] - np.array([cx, cy])) @ up_dir)  # noqa: E731
        anchor = surf(along(knot) + 0.55 * H * cs)
        up_ctrl = np.array([knot, (knot + anchor) / 2 + n3 * 0.3, anchor]); tck, _ = splprep(up_ctrl.T, s=0.0, k=2)
        p_up = np.array(splev(np.linspace(0, 1, 40), tck)).T
        dn_ctrl = [low] + [surf(sd, 0.05 * H * np.sin(k * 0.8)) for k, sd in enumerate(np.arange(along(low) - 0.2 * H * cs, along(low) - 1.5 * H * cs, -0.15 * H * cs))]
        tck, _ = splprep(np.array(dn_ctrl).T, s=0.2, k=3); p_dn = np.array(splev(np.linspace(0, 1, 90), tck)).T
        for P in (p_up, p_dn):
            for p in P:
                p[2] = max(p[2], zat_on(Zmm, p[0], p[1], cell) + RR - 0.15)
        parts = [trimesh.creation.sweep_polygon(Point(0, 0).buffer(RR, resolution=6), p_up), trimesh.creation.sweep_polygon(Point(0, 0).buffer(RR, resolution=6), p_dn),
                 trimesh.creation.icosphere(subdivisions=1, radius=2 * RR).apply_translation(anchor - n3 * 0.3)]
        rope = trimesh.util.concatenate(parts)
    log(f"[FIG] {f['id']} лізе на ({cx:.0f},{cy:.0f}) z={zr:.0f}, стіна {slope:.0f}°, обличчям до скелі")
    return Zmm, m, rope


PLACERS = {"standing": place_standing, "building": place_building, "climbing": place_climbing}


def place_all(Zmm: np.ndarray, cell: float, requests: list[dict], log=print) -> tuple[np.ndarray, list[dict]]:
    """requests: [{"id","x_mm","y_mm","height_mm"}] у координатах РЕЛЬЄФУ (без ободка).
    → (Zmm після вирівнювань, [{"id","mesh","extra"(канат)}])."""
    out = []
    G = Zmm.shape[0]; size = (G - 1) * cell
    for r in requests:
        f = get(str(r.get("id", "")))
        if not f:
            log(f"[FIG] невідома фігурка {r.get('id')} — пропускаю"); continue
        h = float(np.clip(float(r.get("height_mm") or f["default_height_mm"]), f["min_height_mm"], f["max_height_mm"]))
        x = float(np.clip(float(r.get("x_mm", size / 2)), 0.04 * size, 0.96 * size)); y = float(np.clip(float(r.get("y_mm", size / 2)), 0.04 * size, 0.96 * size))
        Zmm, mesh, extra = PLACERS[f["kind"]](Zmm, cell, f, x, y, h, log)
        out.append({"id": f["id"], "mesh": mesh, "extra": extra, "x_mm": x, "y_mm": y, "height_mm": h})
    return Zmm, out


def fuse_into(tile_mesh, parts: list, log=print):
    """Зрощує фігурки в плитку (manifold). Якщо не вдалося — повертає (плитка, окремі частини)."""
    import trimesh
    if not parts:
        return tile_mesh, []
    try:
        u = trimesh.boolean.union([tile_mesh] + parts, engine="manifold")
        u.merge_vertices()
        bodies = u.split(only_watertight=False)
        if len(bodies) > 1:                                          # крихти від зрізу площинами
            big = [b for b in bodies if b.volume >= 5.0]
            u = trimesh.util.concatenate(big) if big else u
        if abs(u.volume) > 0.9 * abs(tile_mesh.volume):
            if not u.is_watertight:
                # дотик по ребру (T-стик) — для слайсера нешкідливо; справжні дірки відкидаємо
                e = u.edges_sorted; _, c = np.unique(e, axis=0, return_counts=True)
                if (c == 1).sum() == 0:
                    log(f"[FIG] union: {(c > 2).sum()} ребер-дотиків, дірок немає — приймаю")
                    return u, []
            else:
                return u, []
        log("[FIG] union дав негерметичне тіло — лишаю фігурки окремими обʼєктами")
    except Exception as exc:  # noqa: BLE001
        log(f"[FIG] union не вдався ({exc}) — лишаю фігурки окремими обʼєктами")
    return tile_mesh, parts
