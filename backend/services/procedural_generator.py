"""
procedural_generator.py — генерує ДРУКОВАНУ 3D-модель «світу» зі специфікації
(spec), яку дає llm_orchestrator (Claude) або rule-based парсер промту.

Це MVP режиму «опиши світ → отримай 3D» (задача #5). БЕЗ зовнішнього text-to-3D API:
будуємо heightfield (шумове поле + форма-маска) → watertight-solid (верх+стіни+дно),
готовий до друку (manifold by construction, мін-фіча через згладження).

spec = {
  "shape": "mountain"|"island"|"valley"|"plateau"|"ridges"|"crater"|"rolling",
  "width_mm": float,            # розмір моделі по X/Y (мм)
  "max_height_mm": float,       # макс. висота рельєфу над базою (мм)
  "base_thickness_mm": float,   # товщина суцільної бази (мм)
  "roughness": float,           # 0..1 деталізація/шум
  "seed": int,                  # для відтворюваності
}
generate_world_mesh(spec) -> trimesh.Trimesh (у МІЛІМЕТРАХ, дно на z=0).
"""
from __future__ import annotations

from typing import Any
import numpy as np
import trimesh

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None

_SHAPES = {"mountain", "island", "valley", "plateau", "ridges", "crater", "rolling"}
_RES = 140  # роздільність сітки heightfield (N×N вершин) — баланс деталь/розмір 3MF


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_spec(spec: dict | None) -> dict:
    """Приводить spec у безпечні, друковані межі."""
    spec = dict(spec or {})
    shape = str(spec.get("shape", "mountain")).lower().strip()
    if shape not in _SHAPES:
        shape = "mountain"
    out = {
        "shape": shape,
        "width_mm": float(_clamp(float(spec.get("width_mm", 120) or 120), 40.0, 220.0)),
        "max_height_mm": float(_clamp(float(spec.get("max_height_mm", 18) or 18), 2.0, 40.0)),
        "base_thickness_mm": float(_clamp(float(spec.get("base_thickness_mm", 3) or 3), 1.0, 8.0)),
        "roughness": float(_clamp(float(spec.get("roughness", 0.5) or 0.5), 0.0, 1.0)),
        "seed": int(spec.get("seed", 0) or 0) & 0x7FFFFFFF,
        "label": str(spec.get("label", "") or "")[:40],
    }
    return out


def _fractal_noise(n: int, rng: np.random.Generator, roughness: float) -> np.ndarray:
    """Багатооктавний згладжений шум у [0,1] (fractal Brownian motion)."""
    field = np.zeros((n, n), dtype=np.float64)
    amp = 1.0
    total = 0.0
    octaves = 5
    for o in range(octaves):
        scale = max(1, int(n / (2 ** (o + 1))))
        coarse = rng.random((scale + 2, scale + 2))
        # білінійний апсемпл до n×n
        ys = np.linspace(0, scale, n)
        xs = np.linspace(0, scale, n)
        y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
        y1 = np.minimum(y0 + 1, scale + 1); x1 = np.minimum(x0 + 1, scale + 1)
        fy = (ys - y0)[:, None]; fx = (xs - x0)[None, :]
        c = coarse
        top = c[np.ix_(y0, x0)] * (1 - fx) + c[np.ix_(y0, x1)] * fx
        bot = c[np.ix_(y1, x0)] * (1 - fx) + c[np.ix_(y1, x1)] * fx
        field += amp * (top * (1 - fy) + bot * fy)
        total += amp
        amp *= (0.35 + 0.35 * roughness)  # вищий roughness → більше дрібних деталей
    field /= max(total, 1e-9)
    if gaussian_filter is not None:
        field = gaussian_filter(field, sigma=max(0.6, (1.0 - roughness) * 2.0))
    fmin, fmax = float(field.min()), float(field.max())
    return (field - fmin) / max(fmax - fmin, 1e-9)


def _shape_field(shape: str, n: int, noise: np.ndarray) -> np.ndarray:
    """Комбінує шум із формою-маскою → нормований heightfield [0,1]."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    cx = cy = (n - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (n / 2.0)  # 0 у центрі, ~1 на краю
    r = np.clip(r, 0.0, 1.0)
    if shape == "mountain":
        mask = np.clip(1.0 - r ** 1.4, 0.0, 1.0)
        h = (0.45 + 0.55 * noise) * mask
    elif shape == "island":
        coast = np.clip(1.0 - (r * 1.15) ** 3, 0.0, 1.0)  # різкий спад до моря
        h = (0.3 + 0.7 * noise) * coast
    elif shape == "valley":
        ridge = np.clip(r ** 1.3, 0.0, 1.0)               # низько в центрі, високо по краях
        h = (0.4 + 0.6 * noise) * ridge
    elif shape == "plateau":
        flat = np.clip(1.0 - r ** 6, 0.0, 1.0)            # плоский верх, обриви
        h = 0.55 * flat + 0.25 * noise * flat
    elif shape == "ridges":
        wave = 0.5 + 0.5 * np.sin((xx / n) * np.pi * 7.0 + noise * 4.0)  # паралельні хребти
        h = (0.3 + 0.7 * wave) * np.clip(1.0 - r ** 2.2, 0.0, 1.0)
    elif shape == "crater":
        crater = np.clip((r - 0.35) * 2.2, -1.0, 1.0)     # вал по колу, западина в центрі
        crater = np.clip(0.5 + 0.5 * np.cos(crater * np.pi) * (r < 0.7), 0.0, 1.0)
        h = (0.4 + 0.6 * noise) * crater
    else:  # rolling
        h = 0.25 + 0.75 * noise
    fmin, fmax = float(h.min()), float(h.max())
    return (h - fmin) / max(fmax - fmin, 1e-9)


def generate_world_mesh(spec: dict) -> trimesh.Trimesh:
    """spec → watertight-solid trimesh у мм (дно z=0). Manifold by construction."""
    s = normalize_spec(spec)
    n = _RES
    rng = np.random.default_rng(s["seed"] or 12345)
    field = _shape_field(s["shape"], n, _fractal_noise(n, rng, s["roughness"]))

    w = s["width_mm"]
    base = s["base_thickness_mm"]
    top_z = base + field * s["max_height_mm"]   # (n,n) висоти верхньої поверхні

    xs = np.linspace(0.0, w, n)
    ys = np.linspace(0.0, w, n)
    gx, gy = np.meshgrid(xs, ys)

    # Вершини: спершу верхня поверхня (n*n), потім нижня площина z=0 (n*n).
    top_v = np.column_stack([gx.ravel(), gy.ravel(), top_z.ravel()])
    bot_v = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(n * n)])
    verts = np.vstack([top_v, bot_v])
    NB = n * n  # зсув індексів нижніх вершин

    def vid(i, j):  # індекс верхньої вершини (рядок i, стовпець j)
        return i * n + j

    faces = []
    # Верхня поверхня (2 трикутники на клітинку)
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)
            faces.append((a, b, c)); faces.append((a, c, d))
    # Нижня поверхня (реверс нормалей)
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = NB + vid(i, j), NB + vid(i, j + 1), NB + vid(i + 1, j + 1), NB + vid(i + 1, j)
            faces.append((a, c, b)); faces.append((a, d, c))
    # Бічні стіни по 4 краях (зшиваємо верх із низом)
    for i in range(n - 1):
        # ліва (j=0) і права (j=n-1)
        for j in (0, n - 1):
            t0, t1 = vid(i, j), vid(i + 1, j)
            b0, b1 = NB + t0, NB + t1
            if j == 0:
                faces.append((t0, b0, b1)); faces.append((t0, b1, t1))
            else:
                faces.append((t0, b1, b0)); faces.append((t0, t1, b1))
    for j in range(n - 1):
        # верх (i=0) і низ (i=n-1)
        for i in (0, n - 1):
            t0, t1 = vid(i, j), vid(i, j + 1)
            b0, b1 = NB + t0, NB + t1
            if i == 0:
                faces.append((t0, b1, b0)); faces.append((t0, t1, b1))
            else:
                faces.append((t0, b0, b1)); faces.append((t0, b1, t1))

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    try:
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        mesh.fix_normals()
    except Exception:
        pass
    # колір як основа карт (білий) — друкується нейтрально, AMS можна перефарбувати
    mesh.visual.face_colors = np.tile([242, 242, 242, 255], (len(mesh.faces), 1))
    return mesh
