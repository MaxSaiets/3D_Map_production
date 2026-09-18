# -*- coding: utf-8 -*-
"""Джерела висот для режиму «Гори».

Пріоритет (кращий → гірший):
  1. swissALTI3D 2 м (swisstopo, відкриті дані) — лише Швейцарія (+буфер за кордон);
  2. Copernicus GLO-30 (ESA, 30 м, глобально; COG-тайли 1°×1° з AWS open data) — основне;
  3. Terrarium z13 (той самий провайдер, що й для мап на сайті) — запасний.

Усе кешується на диску (cache/mountains/{copernicus,swissalti}). Виходить сітка висот у
метрах на регулярній UTM-сітці навколо центру (рядок 0 = ПІВДЕНЬ, як у всьому пайплайні гір).
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests

CACHE_ROOT = Path(os.getenv("MOUNTAIN_CACHE_DIR", "cache/mountains")).resolve()
COP_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
SWISS_STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items"
UA = {"User-Agent": "monadruk-mountains/1.0 (+https://monadruk.com)"}


def utm_epsg(lat: float, lon: float) -> int:
    zone = int(math.floor((lon + 180) / 6) % 60) + 1
    return (32600 if lat >= 0 else 32700) + zone


def make_grid(lat: float, lon: float, area_m: float, grid: int):
    """Квадратна UTM-сітка grid×grid зі стороною area_m навколо (lat, lon).
    → (lons, lats) плоскі масиви, step_m, epsg, (cx, cy)."""
    from pyproj import Transformer
    epsg = utm_epsg(lat, lon)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    cx, cy = to_utm.transform(lon, lat)
    half = area_m / 2
    ex = np.linspace(cx - half, cx + half, grid); ey = np.linspace(cy - half, cy + half, grid)
    EX, EY = np.meshgrid(ex, ey)
    lons, lats = to_ll.transform(EX.ravel(), EY.ravel())
    return np.asarray(lons), np.asarray(lats), float(ex[1] - ex[0]), epsg, (float(cx), float(cy))


# ── Copernicus GLO-30 ────────────────────────────────────────────────────────
def _cop_name(lat_floor: int, lon_floor: int) -> str:
    ns = "N" if lat_floor >= 0 else "S"; ew = "E" if lon_floor >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00_DEM"


def _download(url: str, dest: Path, timeout: float = 300.0, attempts: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for a in range(attempts):
        try:
            with requests.get(url, headers=UA, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            return True
        except requests.RequestException as exc:  # noqa: PERF203
            print(f"[DEM] download retry {a+1}/{attempts} {url}: {exc}", flush=True)
            time.sleep(2 * (a + 1))
    return False


def sample_copernicus(lats: np.ndarray, lons: np.ndarray) -> Optional[np.ndarray]:
    """Білінійна вибірка Copernicus GLO-30. NaN там, де тайла немає (океан)."""
    import rasterio
    from scipy.ndimage import map_coordinates
    out = np.full(lats.shape, np.nan, dtype=np.float32)
    tiles = {(int(math.floor(la)), int(math.floor(lo))) for la, lo in zip(lats[::97], lons[::97])}
    for la in (lats.min(), lats.max()):
        for lo in (lons.min(), lons.max()):
            tiles.add((int(math.floor(la)), int(math.floor(lo))))
    for la0, lo0 in sorted(tiles):
        name = _cop_name(la0, lo0); p = CACHE_ROOT / "copernicus" / f"{name}.tif"
        if not p.exists() and not _download(COP_URL.format(n=name), p):
            print(f"[DEM] Copernicus tile missing: {name}", flush=True); continue
        with rasterio.open(p) as ds:
            a = ds.read(1).astype(np.float32); t = ds.transform; nd = ds.nodata
            if nd is not None:
                a[a == nd] = np.nan
        sel = (lats >= la0) & (lats < la0 + 1) & (lons >= lo0) & (lons < lo0 + 1)
        if not sel.any():
            continue
        col = (lons[sel] - t.c) / t.a - 0.5; row = (lats[sel] - t.f) / t.e - 0.5
        filled = np.where(np.isfinite(a), a, 0.0); valid = np.isfinite(a).astype(np.float32)
        z = map_coordinates(filled, [row, col], order=1, mode="nearest")
        v = map_coordinates(valid, [row, col], order=1, mode="nearest")
        out[sel] = np.where(v > 0.99, z / np.maximum(v, 1e-6), np.nan)
        del a, filled, valid
    return out if np.isfinite(out).any() else None


# ── swissALTI3D 2 м (лише в межах швейцарських даних) ───────────────────────
def _swiss_items(bbox_ll):
    out, url, params = [], SWISS_STAC, {"bbox": ",".join(f"{v:.5f}" for v in bbox_ll), "limit": 100}
    while url:
        r = requests.get(url, params=params, timeout=60, headers=UA); r.raise_for_status()
        d = r.json(); out += d.get("features", []); params = None
        url = next((l["href"] for l in d.get("links", []) if l.get("rel") == "next"), None)
    return out


def sample_swissalti(lats: np.ndarray, lons: np.ndarray, target_step_m: float, max_tiles: int = 80) -> Optional[np.ndarray]:
    """→ висоти (NaN поза швейцарськими даними) або None, якщо область не в Швейцарії."""
    bbox = (float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max()))
    if not (5.9 <= bbox[2] and bbox[0] <= 10.6 and 45.7 <= bbox[3] and bbox[1] <= 47.9):
        return None
    try:
        items = _swiss_items(bbox)
    except Exception as exc:  # noqa: BLE001
        print(f"[DEM] swiss STAC failed: {exc}", flush=True); return None
    hrefs = []
    for it in items:
        for a in it.get("assets", {}).values():
            if a.get("eo:gsd") == 2.0 and a["href"].endswith(".tif"):
                hrefs.append(a["href"]); break
    if not hrefs:
        return None
    if len(hrefs) > max_tiles:
        print(f"[DEM] swiss: {len(hrefs)} тайлів > {max_tiles} — беру Copernicus", flush=True); return None
    import rasterio
    from pyproj import Transformer
    from scipy.ndimage import gaussian_filter, map_coordinates
    rasters = []
    for h in hrefs:
        p = CACHE_ROOT / "swissalti" / h.rsplit("/", 1)[-1]
        if not p.exists() and not _download(h, p, timeout=120):
            continue
        with rasterio.open(p) as ds:
            a = ds.read(1).astype(np.float32); nd = ds.nodata
            if nd is not None:
                a[a == nd] = np.nan
            a[a < -1000] = np.nan
            rasters.append((ds.bounds, ds.res[0], a))
    if not rasters:
        return None
    res = rasters[0][1]
    x0 = min(b.left for b, _, _ in rasters); x1 = max(b.right for b, _, _ in rasters)
    y0 = min(b.bottom for b, _, _ in rasters); y1 = max(b.top for b, _, _ in rasters)
    W, H = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    mosaic = np.full((H, W), np.nan, dtype=np.float32)
    for b, _, a in rasters:
        c = int(round((b.left - x0) / res)); r = int(round((y1 - b.top) / res))
        mosaic[r:r + a.shape[0], c:c + a.shape[1]] = a
    del rasters
    if target_step_m > 1.5 * res:                                   # анти-аліасинг до кроку сітки
        sig = 0.5 * target_step_m / res; ok = np.isfinite(mosaic)
        num = gaussian_filter(np.where(ok, mosaic, 0.0), sig); den = gaussian_filter(ok.astype(np.float32), sig)
        mosaic = np.where(den > 0.5, num / np.maximum(den, 1e-6), np.nan).astype(np.float32)
    to_lv = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    ex, ny = to_lv.transform(lons, lats)
    col = (np.asarray(ex) - x0) / res - 0.5; row = (y1 - np.asarray(ny)) / res - 0.5
    filled = np.where(np.isfinite(mosaic), mosaic, 0.0); valid = np.isfinite(mosaic).astype(np.float32)
    z = map_coordinates(filled, [row, col], order=1, mode="constant", cval=0.0)
    v = map_coordinates(valid, [row, col], order=1, mode="constant", cval=0.0)
    return np.where(v > 0.999, z / np.maximum(v, 1e-6), np.nan).astype(np.float32)


# ── Terrarium (запасний) ─────────────────────────────────────────────────────
def sample_terrarium(lats: np.ndarray, lons: np.ndarray, zoom: int = 13) -> Optional[np.ndarray]:
    try:
        from services.terrarium_tiles import TerrariumTileProvider
        prov = TerrariumTileProvider(base_url=os.getenv("TERRARIUM_URL", "https://s3.amazonaws.com/elevation-tiles-prod/terrarium"),
                                     cache_dir=os.getenv("TERRARIUM_CACHE_DIR", "cache/terrarium"), timeout=30.0)
        z = prov.sample_points(lats, lons, z=zoom)
        return None if z is None else np.asarray(z, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        print(f"[DEM] terrarium failed: {exc}", flush=True); return None


# ── Головний вхід ────────────────────────────────────────────────────────────
def fetch_dem(lat: float, lon: float, area_m: float, grid: int, blend_m: float = 300.0, progress=None) -> dict:
    """→ {"Z": (grid×grid) м, рядок 0 = південь; "step_m"; "sources": [..]; "coverage": {..};
          "epsg"; "center_utm"; "elev_min"; "elev_max"; "lat"; "lon"; "area_m"}"""
    from scipy.ndimage import distance_transform_edt
    lons, lats, step_m, epsg, cutm = make_grid(lat, lon, area_m, grid)
    sources, cover = [], {}
    if progress:
        progress("Завантажую висоти (Copernicus 30 м)…")
    Z = sample_copernicus(lats, lons)
    if Z is not None:
        Z = Z.reshape(grid, grid); ok = np.isfinite(Z); cover["copernicus"] = float(ok.mean()); sources.append("Copernicus GLO-30")
        if not ok.all():                                             # океан/прогалини → Terrarium, далі мінімум
            Zt = sample_terrarium(lats, lons)
            if Zt is not None:
                Zt = Zt.reshape(grid, grid); Z = np.where(ok, Z, Zt); sources.append("Terrarium (прогалини)")
            Z = np.where(np.isfinite(Z), Z, np.nanmin(Z) if np.isfinite(Z).any() else 0.0)
    else:
        if progress:
            progress("Copernicus недоступний — беру Terrarium…")
        Zt = sample_terrarium(lats, lons)
        if Zt is None:
            raise RuntimeError("Жодне джерело висот не відповіло (Copernicus, Terrarium)")
        Z = Zt.reshape(grid, grid); Z = np.where(np.isfinite(Z), Z, np.nanmin(Z)); sources.append("Terrarium z13"); cover["terrarium"] = 1.0
    if os.getenv("MOUNTAIN_SWISS", "1") == "1":
        try:
            if progress:
                progress("Перевіряю swissALTI3D 2 м…")
            Zs = sample_swissalti(lats, lons, step_m)
        except Exception as exc:  # noqa: BLE001
            print(f"[DEM] swiss failed: {exc}", flush=True); Zs = None
        if Zs is not None:
            Zs = Zs.reshape(grid, grid); ok = np.isfinite(Zs)
            if ok.mean() > 0.02:
                bias = float(np.nanmedian(Zs[ok] - Z[ok]))
                w = np.clip(distance_transform_edt(ok) * step_m / max(blend_m, step_m), 0, 1)
                Z = np.where(ok, w * np.nan_to_num(Zs) + (1 - w) * (Z + bias), Z)
                cover["swissalti3d"] = float(ok.mean()); sources.insert(0, f"swissALTI3D 2 м ({ok.mean()*100:.0f} %)")
    Z = Z.astype(np.float64)
    return {"Z": Z, "step_m": step_m, "sources": sources, "coverage": cover, "epsg": epsg, "center_utm": cutm,
            "elev_min": float(Z.min()), "elev_max": float(Z.max()), "lat": lat, "lon": lon, "area_m": area_m}
