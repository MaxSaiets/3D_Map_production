# -*- coding: utf-8 -*-
"""Пайплайн «Гори»: spec → DEM → поле висот (масштаб, ободок, боки, фігурки) → файли.

Виходи (у OUTPUT_DIR, basename = mountain_<task8>):
  *_print.3mf   — головний друк-файл: одна плита (якщо влазить у стіл) або збірка обʼєктів;
  *_tiles.zip   — плитки під стіл (лише коли плита більша за стіл) + фігурки зрощено в плитки;
  *.glb         — легке превʼю з текстурою для вʼюера;
  *_preview.png — «фото»-вид зверху (знімок × тіні), *_paint.png — гайд розпису без тіней;
  *_meta.json   — усі числа (масштаб, перебільшення, джерела висот, розміри плиток).
Памʼять: сітка ≤ 601² (≈0.72 млн граней), потокові записи, без Blender/Cycles.
"""
from __future__ import annotations

import gc
import io
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np

from . import figures as FIG
from .agent import normalize_spec
from .dem import fetch_dem
from .mesh import apply_slope_band, frame_solid, preview_glb, split_tiles, tile_cuts, write_3mf, write_3mf_multi
from .texture import compose_plate_texture, hillshade_png, paint_texture, satellite_texture

MAX_GRID = int(os.getenv("MOUNTAIN_MAX_GRID", "481"))   # 481² ≈ 0.5 млн граней рельєфу: пік ~1.6 ГБ; 601² давав 2.8 ГБ
DEFAULT_AREA_KM = 4.0


def _grid_for(inner_mm: float) -> int:
    return int(min(MAX_GRID, max(161, round(inner_mm / 0.5) + 1)))


def _summit(Zmm, core_mask):
    k = np.argmax(np.where(core_mask, Zmm, -np.inf)); j, i = np.unravel_index(k, Zmm.shape); return i, j


def _figure_xy(where: str, f: dict, Zmm: np.ndarray, cell: float, core_mask, inner: float, band: float, taken: list):
    """→ (x_mm, y_mm) у координатах рельєфу."""
    from scipy.ndimage import gaussian_filter
    G = Zmm.shape[0]; si, sj = _summit(Zmm, core_mask); sx, sy = si * cell, sj * cell
    if where == "point":
        return band + float(f.get("fx", 0.5)) * (inner - 2 * band), band + float(f.get("fy", 0.5)) * (inner - 2 * band)
    if where in ("summit", "steepest"):
        # кілька фігурок «на вершині» — розводимо по колу, щоб не злипались
        n = sum(1 for t in taken if t[0] == where); r = 0.0 if n == 0 else 0.03 * inner
        ang = n * 2.4
        return float(np.clip(sx + r * np.cos(ang), band, inner - band)), float(np.clip(sy + r * np.sin(ang), band, inner - band))
    # slope: пологий майданчик нижче вершини — шукаємо по всьому ядру на 0.10–0.45 inner від вершини,
    # оцінка = пологість передусім, висота — другорядно (щоб хатина не «висіла» на ребрі піраміди)
    xs = np.arange(G) * cell; X, Y = np.meshgrid(xs, xs)
    gy, gx = np.gradient(gaussian_filter(Zmm, max(3.0 / cell, 1)), cell); slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    d = np.hypot(X - sx, Y - sy)
    ok = core_mask & (d > 0.10 * inner) & (d < 0.6 * inner)
    for t in taken:
        ok &= np.hypot(X - t[1], Y - t[2]) > 0.06 * inner
    if not ok.any():
        ok = core_mask & (d > 0.08 * inner)
    zn = (Zmm - Zmm[core_mask].min()) / max(Zmm[core_mask].ptp(), 1e-6)
    score = -np.minimum(slope, 45) / 45.0 * 2.0 + zn * 0.6 - d / inner * 0.4
    k = np.argmax(np.where(ok, score, -np.inf)); j, i = np.unravel_index(k, Zmm.shape)
    return float(X[j, i]), float(Y[j, i])


def quick_preview(lat: float, lon: float, area_km: float, size_mm: float = 200.0, height_mm: float | None = None, with_sat: bool = True) -> dict:
    """Миттєве превʼю для мапи: hillshade (+знімок) 384 px + цифри масштабу. Грубa сітка 181."""
    d = fetch_dem(lat, lon, area_km * 1000, 181)
    Z = d["Z"]; relief_m = d["elev_max"] - d["elev_min"]; scale = area_km * 1000 / (size_mm / 1000)
    relief_mm = relief_m / scale * 1000
    sat = None
    if with_sat:
        try:
            sat = satellite_texture(lat, lon, area_km * 1000, tex=384)
        except Exception as exc:  # noqa: BLE001
            print(f"[MNT] sat preview skipped: {exc}", flush=True)
    png = hillshade_png(Z, d["step_m"], 384, sat)
    import base64
    return {"png": "data:image/png;base64," + base64.b64encode(png).decode(), "elev_min": round(d["elev_min"]), "elev_max": round(d["elev_max"]),
            "relief_m": round(relief_m), "scale": round(scale), "relief_mm_natural": round(relief_mm, 1),
            "zexag_for_height": (round((height_mm) / relief_mm, 2) if height_mm and relief_mm > 0 else None), "sources": d["sources"]}


def run(spec_in: dict, out_dir: Path, basename: str, progress=lambda p, m: None, log=print) -> dict:
    t0 = time.time()
    spec, _ = normalize_spec(spec_in)
    pl = spec.get("place")
    if not pl:
        raise ValueError("Не вказано місце гори")
    size = spec["size_mm"]; fr = spec["frame"]; fw, fh = fr["width_mm"], (fr["height_mm"] if fr["width_mm"] > 0 else 0.0)
    base = spec["base_mm"]; sides = spec["sides"]
    inner = size - 2 * fw
    band = float(np.clip(0.055 * inner, 6.0, 24.0)) if sides == "slope" else 0.0
    core = inner - 2 * band
    area_km = spec.get("area_km") or pl.get("area_km") or DEFAULT_AREA_KM
    scale = area_km * 1000 / (core / 1000)                       # ядро = область
    G = _grid_for(inner); cell = inner / (G - 1)
    progress(8, "Завантажую висоти…")
    dem = fetch_dem(pl["lat"], pl["lon"], area_km * 1000 * inner / core, G, progress=lambda m: progress(12, m))
    Z = dem["Z"]
    xs = np.arange(G) * cell; X, Y = np.meshgrid(xs, xs)
    core_mask = (X >= band) & (X <= inner - band) & (Y >= band) & (Y <= inner - band)
    zmin, zmax = float(Z[core_mask].min()), float(Z[core_mask].max())
    relief_mm = (zmax - zmin) / scale * 1000
    fig_top = max([f["height_mm"] for f in spec.get("figures", []) if f["where"] in ("summit", "steepest")] or [0.0])
    top_budget = (spec["height_mm"] - fh - base) if spec.get("height_mm") else None
    if top_budget is not None and spec["height_mm"] + fig_top > spec["bed_mm"] and spec["height_mm"] <= spec["bed_mm"]:
        top_budget -= (spec["height_mm"] + fig_top - spec["bed_mm"])   # фігурка на вершині має влізти під стелю стола
        log(f"[MNT] висоту рельєфу зменшено на {spec['height_mm'] + fig_top - spec['bed_mm']:.0f} мм: фігурка на вершині + стеля стола {spec['bed_mm']:.0f} мм")
    if top_budget is None:
        zexag = min(1.0, (0.6 * size - fh - base) / max(relief_mm, 1e-6))
    else:
        zexag = max(top_budget, 5.0) / max(relief_mm, 1e-6)
    Zmm = (Z - zmin) / scale * 1000 * zexag + base
    log(f"[MNT] {pl['name']}: {area_km:.1f} км → 1:{scale:.0f}, рельєф {zmax-zmin:.0f} м = {relief_mm:.1f} мм, ×{zexag:.3f}; сітка {G}², ободок {fr['style']} {fw}×{fh}, боки {sides}")
    progress(30, "Формую рельєф і боки…")
    if sides == "slope":
        Zmm = apply_slope_band(Zmm, cell, band, base)
    # фігурки
    figs_req = []
    taken = []
    for f in spec.get("figures", []):
        x, y = _figure_xy(f["where"], f, Zmm, cell, core_mask, inner, band, taken)
        taken.append((f["where"], x, y)); figs_req.append({"id": f["id"], "x_mm": x, "y_mm": y, "height_mm": f["height_mm"]})
    progress(38, "Саджу фігурки…" if figs_req else "Будую тіло…")
    Zmm, placed = FIG.place_all(Zmm, cell, figs_req, log=log)
    for p in placed:                                             # у координати плити
        p["mesh"].apply_translation([fw, fw, fh])
        if p["extra"] is not None:
            p["extra"].apply_translation([fw, fw, fh])
    progress(45, "Будую тіло…")
    V, F, info = frame_solid(xs, xs, Zmm, fw=fw, fh=fh, ch=4.0, fillet=(min(5.0, fw * 0.5) if fr["style"] == "rounded" and fw > 0 else None),
                             wall_rock=(sides == "rock"), wall_depth_mm=min(6.0, 0.03 * size), wall_rows=int(np.clip(200 * 200 / size, 60, 120)))
    import trimesh
    full = trimesh.Trimesh(V, F, process=False)
    if full.volume < 0:
        full.invert()
    full.fix_normals()
    height_total = float(full.bounds[1][2])
    log(f"[MNT] тіло: {len(full.faces):,} граней, герметичне={full.is_watertight}, {np.round(full.extents,1)} мм, {time.time()-t0:.0f} с")
    out_dir = Path(out_dir); out = {}
    fig_parts = [p["mesh"] for p in placed] + [p["extra"] for p in placed if p["extra"] is not None]
    # ── друк-файл ────────────────────────────────────────────────────────────
    bed = spec["bed_mm"]; cuts = tile_cuts(size, bed)
    progress(62, "Записую друк-файли…")
    p_print = out_dir / f"{basename}_print.3mf"
    if not cuts:
        body, loose = (FIG.fuse_into(full, fig_parts, log) if fig_parts and len(full.faces) < 900_000 else (full, fig_parts))
        if loose:
            write_3mf_multi([("mountain", body.vertices, body.faces)] + [(f"figure_{i+1}", m.vertices, m.faces) for i, m in enumerate(loose)], p_print)
        else:
            write_3mf(body.vertices, body.faces, p_print, name=basename)
        del body; out["3mf"] = str(p_print); out["tiles"] = []
    else:
        zpath = out_dir / f"{basename}_tiles.zip"; info_t = []
        tiles = split_tiles(full, cuts, size)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for nm, t in tiles.items():
                org = t.metadata["origin"]; ext = t.extents
                mine = []
                for m in fig_parts:                                   # фігурки, чий центр у цій плитці
                    c = m.bounds.mean(0)
                    if org[0] - 1e-6 <= c[0] <= org[0] + ext[0] + 1e-6 and org[1] - 1e-6 <= c[1] <= org[1] + ext[1] + 1e-6:
                        mm = m.copy(); mm.apply_translation(-org); mine.append(mm)
                body, loose = (FIG.fuse_into(t, mine, log) if mine else (t, []))
                parts = [(nm, body)]
                if body.extents[2] > bed:                              # вища за стіл → низ + «шапка» зі стиком на z_cut
                    z_cut = round(bed * 0.55 / 10) * 10.0
                    low = body.slice_plane([0, 0, z_cut], [0, 0, -1], cap=True); high = body.slice_plane([0, 0, z_cut], [0, 0, 1], cap=True)
                    high.apply_translation([0, 0, -z_cut])
                    for q in (low, high):
                        q.merge_vertices(); q.fix_normals()
                    parts = [(f"{nm}_low", low), (f"{nm}_top", high)]; log(f"[MNT] плитка {nm} {body.extents[2]:.0f} мм > стіл → розріз на z={z_cut:.0f}")
                for pn, pm in parts:
                    tmp = out_dir / f"{basename}_tile_{pn}.3mf"
                    if loose and pn == nm:
                        write_3mf_multi([(pn, pm.vertices, pm.faces)] + [(f"figure_{i+1}", m.vertices, m.faces) for i, m in enumerate(loose)], tmp)
                    else:
                        write_3mf(pm.vertices, pm.faces, tmp, name=pn)
                    z.write(tmp, tmp.name); tmp.unlink()
                    info_t.append({"name": pn, "size_mm": [round(float(v), 1) for v in pm.extents], "faces": len(pm.faces), "watertight": bool(pm.is_watertight), "figures": len(mine) if pn == nm or pn.endswith("_top") else 0})
                    del pm
        del tiles; gc.collect()
        out["tiles"] = info_t; out["tiles_zip"] = str(zpath)
        # збірка для перегляду/друку на великому столі: гора + фігурки окремими обʼєктами (потоково)
        write_3mf_multi([("mountain", V, F)] + [(f"figure_{i+1}", m.vertices, m.faces) for i, m in enumerate(fig_parts)], p_print)
        out["3mf"] = str(p_print)
    # ── текстури й превʼю ────────────────────────────────────────────────────
    progress(78, "Готую фото-превʼю…")
    sat = None; paint = None
    if spec["texture"] == "satellite":
        try:
            sat = satellite_texture(pl["lat"], pl["lon"], area_km * 1000 * inner / core, tex=1024)
            if sat is not None:
                paint = paint_texture(Z, dem["step_m"], sat)
        except Exception as exc:  # noqa: BLE001
            log(f"[MNT] texture failed: {exc}")
    from PIL import Image
    tex_plate = compose_plate_texture(paint if paint is not None else (sat if sat is not None else (np.full((64, 64, 3), 200, np.uint8))), size, fw)
    # вʼюер сайту світить слабко для текстурованих матеріалів — піднімаємо гаму лише для GLB
    tex_glb = (255.0 * (tex_plate.astype(np.float32) / 255.0) ** 0.72).astype(np.uint8)
    p_glb = out_dir / f"{basename}.glb"
    # превʼю з ГРУБОЇ сітки (≤ 241²) — без децимації і без другої копії повного тіла (памʼять прода)
    k = max(1, int(np.ceil((G - 1) / 240))); Zc = Zmm[::k, ::k]; xc = xs[::k]
    if (G - 1) % k:
        Zc = np.vstack([np.hstack([Zc, Zmm[::k, -1:]]), np.hstack([Zmm[-1:, ::k], Zmm[-1:, -1:]])]); xc = np.append(xc, xs[-1])
    Vc, Fc, _ = frame_solid(xc, xc, Zc, fw=fw, fh=fh, ch=4.0, fillet=(min(5.0, fw * 0.5) if fr["style"] == "rounded" and fw > 0 else None), wall_rock=False)
    Vp, Fp = Vc, Fc
    if fig_parts:
        offs = len(Vc); Vp = np.vstack([Vc] + [m.vertices for m in fig_parts]); Fl = [Fc]
        for m in fig_parts:
            Fl.append(m.faces + offs); offs += len(m.vertices)
        Fp = np.vstack(Fl)
    nf = preview_glb(Vp, Fp, p_glb, size, 0.0, texture_rgb=tex_glb, max_faces=10**9)
    del Vc, Fc, Vp, Fp
    out["glb"] = str(p_glb)
    (out_dir / f"{basename}_preview.png").write_bytes(hillshade_png(Zmm, cell, 768, sat)); out["preview_png"] = str(out_dir / f"{basename}_preview.png")
    if paint is not None:
        Image.fromarray(np.ascontiguousarray(paint[::-1])).save(out_dir / f"{basename}_paint.jpg", quality=90); out["paint_jpg"] = str(out_dir / f"{basename}_paint.jpg")
    try:
        import resource  # Linux (прод)
        peak_mb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except Exception:
        try:
            import ctypes, ctypes.wintypes as wt
            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wt.DWORD), ("pf", wt.DWORD), ("peak", ctypes.c_size_t), ("ws", ctypes.c_size_t)] + [(f"x{i}", ctypes.c_size_t) for i in range(6)]
            c = _PMC(); c.cb = ctypes.sizeof(_PMC); fn = ctypes.windll.psapi.GetProcessMemoryInfo; fn.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]
            fn(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb); peak_mb = int(c.peak // 2**20)
        except Exception:
            peak_mb = None
    log(f"[MNT] пік памʼяті процесу: {peak_mb} МБ")
    meta = {"place": pl, "peak_mb": peak_mb, "area_km": area_km, "scale": round(scale), "zexag": round(zexag, 3), "size_mm": size, "height_mm": round(height_total, 1),
            "inner_mm": inner, "band_mm": band, "frame": fr, "sides": sides, "grid": G, "cell_mm": round(cell, 3), "elev_min": round(zmin), "elev_max": round(zmax),
            "sources": dem["sources"], "coverage": dem["coverage"], "faces": len(F), "watertight": bool(full.is_watertight), "tiles": out.get("tiles", []),
            "figures": [{"id": p["id"], "x_mm": round(p["x_mm"], 1), "y_mm": round(p["y_mm"], 1), "height_mm": p["height_mm"]} for p in placed],
            "preview_faces": nf, "seconds": round(time.time() - t0)}
    (out_dir / f"{basename}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"); out["meta"] = meta
    del full, V, F; gc.collect()
    progress(96, "Готово")
    return out
