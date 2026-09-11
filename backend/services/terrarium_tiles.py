"""
Terrarium elevation tiles loader (no API key).

Tile source (commonly used Mapzen terrarium):
https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png

Terrarium encoding:
elevation_m = (R * 256 + G + B / 256) - 32768

This module downloads required tiles on demand, caches them on disk, and provides
fast sampling for arrays of (lat, lon) points.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import requests


@dataclass(frozen=True)
class TileKey:
    z: int
    x: int
    y: int


def _latlon_to_global_pixel(lon: float, lat: float, z: int) -> Tuple[float, float]:
    # WebMercator global pixel coordinates at zoom z
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 256.0 * (2**z)
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n
    return x, y


def _global_pixel_to_tile(x: float, y: float) -> Tuple[int, int, float, float]:
    tx = int(math.floor(x / 256.0))
    ty = int(math.floor(y / 256.0))
    px = x - tx * 256.0
    py = y - ty * 256.0
    return tx, ty, px, py


def _bilinear_sample(img: np.ndarray, px: float, py: float) -> float:
    # img: (H,W) float32
    h, w = img.shape
    x = float(np.clip(px, 0.0, w - 1.0))
    y = float(np.clip(py, 0.0, h - 1.0))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0
    v00 = img[y0, x0]
    v10 = img[y0, x1]
    v01 = img[y1, x0]
    v11 = img[y1, x1]
    return float((v00 * (1 - dx) + v10 * dx) * (1 - dy) + (v01 * (1 - dx) + v11 * dx) * dy)


def _decode_terrarium_png(png_bytes: bytes) -> np.ndarray:
    # Decode PNG bytes -> elevation array float32 (256x256)
    from PIL import Image
    import io

    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.asarray(im, dtype=np.float32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    elev = (r * 256.0 + g + b / 256.0) - 32768.0
    return elev.astype(np.float32)


class TerrariumTileProvider:
    def __init__(
        self,
        base_url: str = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium",
        cache_dir: str = "cache/terrarium",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

        # in-memory decoded tiles cache
        self._mem: Dict[TileKey, np.ndarray] = {}

    def _tile_path(self, key: TileKey) -> Path:
        return self.cache_dir / str(key.z) / str(key.x) / f"{key.y}.png"

    def _fetch_tile_png(self, key: TileKey) -> Optional[bytes]:
        # disk cache first
        p = self._tile_path(key)
        if p.exists():
            return p.read_bytes()

        url = f"{self.base_url}/{key.z}/{key.x}/{key.y}.png"
        try:
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            png = resp.content
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(png)
            return png
        except Exception:
            return None

    def get_tile(self, key: TileKey) -> Optional[np.ndarray]:
        if key in self._mem:
            return self._mem[key]
        png = self._fetch_tile_png(key)
        if png is None:
            return None
        elev = _decode_terrarium_png(png)
        self._mem[key] = elev
        return elev

    def prefetch(self, keys: "list[TileKey]", workers: int = 8) -> None:
        """Паралельно тягне відсутні тайли (диск/памʼять — миттєво)."""
        missing = [k for k in keys if k not in self._mem and not self._tile_path(k).exists()]
        if len(missing) < 2:
            return
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(missing)))) as pool:
            list(pool.map(self._fetch_tile_png, missing))

    def sample_points(self, lats: np.ndarray, lons: np.ndarray, z: int) -> Optional[np.ndarray]:
        """Білінійна вибірка висот для масиву точок.

        12.09.2026: було — Python-цикл по кожній точці (для сітки рельєфу це
        ~500 тис. ітерацій: 24.6 с «Отримання з API» на проді, з них саме
        завантаження — секунди). Тепер numpy по тайлах + паралельне
        завантаження. Формула та сама, що в `_bilinear_sample`.
        """
        lats = np.asarray(lats, dtype=np.float64).ravel()
        lons = np.asarray(lons, dtype=np.float64).ravel()
        out = np.full(lats.shape, np.nan, dtype=np.float32)
        if lats.size == 0:
            return out

        n = 256.0 * (2 ** z)
        lat_c = np.clip(lats, -85.05112878, 85.05112878)
        gx = (lons + 180.0) / 360.0 * n
        lat_rad = np.radians(lat_c)
        gy = (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / math.pi) / 2.0 * n
        tx = np.floor(gx / 256.0).astype(np.int64)
        ty = np.floor(gy / 256.0).astype(np.int64)
        px = gx - tx * 256.0
        py = gy - ty * 256.0

        pair = tx * (1 << 32) + ty
        uniq, inverse = np.unique(pair, return_inverse=True)
        keys = [TileKey(z=z, x=int(u >> 32), y=int(u & 0xFFFFFFFF)) for u in uniq]
        self.prefetch(keys)

        for j, key in enumerate(keys):
            tile = self.get_tile(key)
            if tile is None:
                continue
            sel = np.nonzero(inverse == j)[0]
            h, w = tile.shape
            x = np.clip(px[sel], 0.0, w - 1.0)
            y = np.clip(py[sel], 0.0, h - 1.0)
            x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
            x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
            dx = x - x0; dy = y - y0
            v = ((tile[y0, x0] * (1 - dx) + tile[y0, x1] * dx) * (1 - dy)
                 + (tile[y1, x0] * (1 - dx) + tile[y1, x1] * dx) * dy)
            out[sel] = v.astype(np.float32)

        return out


