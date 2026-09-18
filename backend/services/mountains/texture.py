# -*- coding: utf-8 -*-
"""Текстури й превʼю для режиму «Гори».

* satellite_texture — ESRI World Imagery (z ≤ 16) на UTM-сітку моделі (рядок 0 = південь);
  кеш тайлів на диску. Використовується для «фото»-превʼю у 3D-вʼюері та карток пресетів.
* hillshade_png — швидка тіньова карта висот (matplotlib, без тайлів) для миттєвого превʼю,
  коли користувач тягне рамку по мапі.
* paint_texture — «денне світло без тіней»: знімок із математично прибраними тінями
  (оцінка сонця за кореляцією яскравості з косинусом падіння) — базовий гайд для розпису.
"""
from __future__ import annotations

import io
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests

CACHE = Path(os.getenv("MOUNTAIN_CACHE_DIR", "cache/mountains")).resolve() / "imagery"
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
UA = {"User-Agent": "monadruk-mountains/1.0 (+https://monadruk.com)"}


def _get(url: str, key: str) -> Optional[bytes]:
    CACHE.mkdir(parents=True, exist_ok=True); p = CACHE / key
    if p.exists():
        return p.read_bytes() if p.stat().st_size else None
    for a in range(3):
        try:
            r = requests.get(url, timeout=30, headers=UA)
            if r.status_code == 200 and r.content[:3] in (b"\xff\xd8\xff", b"\x89PN"):
                p.write_bytes(r.content); return r.content
            if r.status_code in (204, 404):
                p.write_bytes(b""); return None
        except requests.RequestException:
            time.sleep(1 + a)
    return None


def _gpx(lat, lon, z):
    n = 2 ** z; lr = np.radians(lat)
    return (lon + 180) / 360 * n * 256, (1 - np.log(np.tan(lr) + 1 / np.cos(lr)) / math.pi) / 2 * n * 256


def satellite_texture(lat: float, lon: float, area_m: float, tex: int = 1024, epsg: Optional[int] = None) -> Optional[np.ndarray]:
    """→ uint8 (tex×tex×3), рядок 0 = ПІВДЕНЬ (як сітка моделі). None, якщо тайли недоступні."""
    from PIL import Image
    from pyproj import Transformer
    from scipy.ndimage import map_coordinates
    from .dem import utm_epsg
    epsg = epsg or utm_epsg(lat, lon)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True); to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    cx, cy = to_utm.transform(lon, lat); half = area_m / 2
    ex = np.linspace(cx - half, cx + half, tex); EX, EY = np.meshgrid(ex, np.linspace(cy - half, cy + half, tex))
    lons, lats = to_ll.transform(EX.ravel(), EY.ravel()); lons, lats = np.asarray(lons), np.asarray(lats)
    # zoom: ~1 тайловий піксель на піксель текстури, але не вище 16 (і не більше ~120 тайлів)
    mpp = area_m / tex; z = int(np.clip(round(math.log2(156543.03 * math.cos(math.radians(lat)) / mpp)), 8, 16))
    while True:
        gx, gy = _gpx(lats, lons, z)
        tx0, tx1, ty0, ty1 = int(gx.min() // 256), int(gx.max() // 256), int(gy.min() // 256), int(gy.max() // 256)
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) <= 120 or z <= 8:
            break
        z -= 1
    mos = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), np.float32); got = 0
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            b = _get(ESRI.format(z=z, x=tx, y=ty), f"esri_{z}_{tx}_{ty}.jpg")
            if b is None:
                mos[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = 128; continue
            mos[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = np.asarray(Image.open(io.BytesIO(b)).convert("RGB")); got += 1
    if got == 0:
        return None
    rgb = np.stack([map_coordinates(mos[..., c], [gy - ty0 * 256, gx - tx0 * 256], order=1, mode="nearest") for c in range(3)], -1)
    return np.clip(rgb, 0, 255).astype(np.uint8).reshape(tex, tex, 3)


def hillshade_rgb(Z: np.ndarray, step_m: float, az: float = 315, alt: float = 45, cmap: str = "gray") -> np.ndarray:
    """Тіньова карта (рядок 0 = південь) → uint8 RGB."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=az, altdeg=alt)
    rgb = ls.shade(Z, cmap=getattr(cm, cmap), blend_mode="overlay", vert_exag=2.2, dx=step_m, dy=step_m)
    return (np.clip(rgb[..., :3], 0, 1) * 255).astype(np.uint8)


def hillshade_png(Z: np.ndarray, step_m: float, size_px: int = 512, texture: Optional[np.ndarray] = None) -> bytes:
    """PNG «північ угорі»: hillshade (за бажанням змішаний із знімком)."""
    from PIL import Image
    from scipy.ndimage import zoom
    shade = hillshade_rgb(Z, step_m).astype(np.float32) / 255.0
    if texture is not None:
        t = texture.astype(np.float32) / 255.0
        if t.shape[0] != shade.shape[0]:
            t = zoom(t, (shade.shape[0] / t.shape[0], shade.shape[1] / t.shape[1], 1), order=1)
        lum = shade.mean(2, keepdims=True); shade = np.clip(t * (0.30 + 1.45 * lum), 0, 1)
    img = Image.fromarray((shade[::-1] * 255).astype(np.uint8)).resize((size_px, size_px), Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, "PNG", optimize=True); return buf.getvalue()


def paint_texture(Z: np.ndarray, step_m: float, sat: np.ndarray) -> np.ndarray:
    """Знімок без тіней: оцінюємо сонце знімка (перебір азимут/висота за кореляцією), ділимо
    на модель освітлення, у глибоких тінях підставляємо колір околу."""
    from scipy.ndimage import gaussian_filter, zoom
    T = sat.shape[0]
    E = zoom(Z, T / Z.shape[0], order=1) if Z.shape[0] != T else Z
    st = step_m * Z.shape[0] / T
    dzdy, dzdx = np.gradient(gaussian_filter(E, 1.0), st)
    nrm = np.dstack([-dzdx, -dzdy, np.ones_like(E)]); nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    tex = sat.astype(np.float32) / 255.0; lum = tex @ np.array([0.299, 0.587, 0.114])
    slope = np.hypot(dzdx, dzdy); m = slope > 0.3
    if m.sum() < 500:
        return sat
    best = (-2, 180, 50)
    for az in range(90, 271, 15):
        for el in (30, 45, 60):
            a, e = np.radians(az), np.radians(el); sv = np.array([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])
            ci = np.clip(nrm @ sv, 0, 1); c = np.corrcoef(ci[m][::5], lum[m][::5])[0, 1]
            if c > best[0]:
                best = (c, az, el)
    if best[0] < 0.25:
        return sat
    a, e = np.radians(best[1]), np.radians(best[2]); sv = np.array([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])
    cosi = np.clip(nrm @ sv, 0, 1); shade = 0.45 + 0.55 * cosi
    de = np.clip(tex / shade[..., None] * 0.88, 0, 1)
    fill = gaussian_filter(de, (12, 12, 0)); w = np.clip((cosi - 0.12) / 0.3, 0, 1)[..., None]
    out = de * w + fill * (1 - w)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def compose_plate_texture(inner: np.ndarray, size_mm: float, fw: float, tex: int = 1024, frame_rgb=(150, 105, 62), wall_rgb=(160, 158, 152)) -> np.ndarray:
    """Текстура на всю плиту (для UV = xy/size): усередині — знімок/розпис, по краю — ободок."""
    from scipy.ndimage import zoom
    out = np.empty((tex, tex, 3), np.uint8); out[:] = np.array(frame_rgb if fw > 0 else wall_rgb, np.uint8)
    k0 = int(round(fw / size_mm * tex)); k1 = tex - k0
    n = max(k1 - k0, 1)
    src = zoom(inner, (n / inner.shape[0], n / inner.shape[1], 1), order=1) if inner.shape[0] != n else inner
    out[k0:k0 + n, k0:k0 + n] = np.clip(src, 0, 255).astype(np.uint8)
    return out
