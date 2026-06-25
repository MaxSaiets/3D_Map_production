"""
Highlight-будинок на РЕЛЬЄФІ — паз ЯК У ДОРІГ, але будинок СТОЇТЬ (пласке дно).

Власник: «низ будинку доробляється до НАЙНИЖЧОЇ точки свого мешу щоб він СТОЯВ, і
вирізаються пази як у доріг із зазором».
  • Будинок = пласка призма footprint від building_bottom (найнижча z мешу) до
    building_top — пласке дно, СТОЇТЬ (не драпіруємо за рельєфом).
  • Паз = пласка кишеня з підлогою на building_bottom у footprint+зазор: верхні
    вершини рельєфу у зоні опускаємо до підлоги → будинок сідає рівно, з тим самим
    боковим зазором (GROOVE_CLEARANCE_MM=0.15), що дорожня вставка.
Надійно (опускання вершин, БЕЗ boolean — працює і на негерметичному мерджі).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import trimesh
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry


def _contains_mask(poly: BaseGeometry, xy: np.ndarray) -> np.ndarray:
    """Векторизований point-in-polygon (shapely 2.x contains_xy) з prepared-fallback."""
    if len(xy) == 0:
        return np.zeros(0, dtype=bool)
    try:
        from shapely import contains_xy
        return np.asarray(contains_xy(poly, xy[:, 0], xy[:, 1]), dtype=bool)
    except Exception:
        from shapely.prepared import prep
        pr = prep(poly)
        return np.array([pr.contains(Point(float(x), float(y))) for x, y in xy], dtype=bool)


def carve_highlight_groove(
    terrain_mesh: Optional[trimesh.Trimesh],
    footprint: Optional[BaseGeometry],
    building_top_m: float,
    building_bottom_m: float,
    *,
    scale_factor: float,
    clearance_mm: float = 0.15,
    zone_prefix: str = "",
) -> Tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh]]:
    """Пласка кишеня (паз) під будинок, що СТОЇТЬ + червоний будинок з пласким дном.

    Повертає (terrain_mesh_з_пазом, red_building) або (terrain_mesh, None) на невдачі
    (тоді будинок просто не вставний — fallback, нічого не ламає).

    Одиниці: terrain_mesh/footprint/building_*_m у СВІТОВИХ метрах (локальний кадр);
    clearance — модель-мм, /scale_factor у світові метри (як _model_mm_to_world_m).
    """
    if terrain_mesh is None or footprint is None or getattr(footprint, "is_empty", True):
        return terrain_mesh, None
    try:
        sf = float(scale_factor)
        if sf <= 0:
            return terrain_mesh, None
        clear = float(clearance_mm) / sf
        floor = float(building_bottom_m)           # найнижча точка мешу будинку = підлога пазу
        thick = float(building_top_m) - floor       # повна висота будинку
        if thick <= 1e-6:
            return terrain_mesh, None

        # Будинок: ПЛАСКА призма (footprint, дно=floor, верх=building_top) — СТОЇТЬ.
        from services.flat_plate_pipeline import build_flat_layer_mesh_from_mask, _with_color, LAYER_COLORS
        red = build_flat_layer_mesh_from_mask(
            footprint, bottom_z_m=floor, thickness_m=thick,
            color=LAYER_COLORS["highlight"], min_area_m2=1e-12,
        )
        if red is None or len(red.faces) == 0:
            return terrain_mesh, None

        # Паз: ПЛАСКА підлога на floor у footprint+зазор — опускаємо туди верхні
        # вершини рельєфу (що ВИЩЕ floor); нижчі не чіпаємо. Будинок сідає рівно.
        groove = footprint.buffer(clear, join_style=2)
        N = terrain_mesh.face_normals
        up = N[:, 2] > 0.3
        tv = terrain_mesh.vertices.copy()
        topverts = np.unique(terrain_mesh.faces[up].ravel())
        tvp = tv[topverts]
        gminx, gminy, gmaxx, gmaxy = groove.bounds
        bbok = (tvp[:, 0] >= gminx) & (tvp[:, 0] <= gmaxx) & (tvp[:, 1] >= gminy) & (tvp[:, 1] <= gmaxy)
        cand = topverts[bbok]
        if len(cand) == 0:
            return terrain_mesh, None
        inside = _contains_mask(groove, tv[cand][:, :2])
        ids = cand[inside]
        ids = ids[tv[ids, 2] > floor]   # опускаємо лише ті, що ВИЩЕ підлоги
        if len(ids) == 0:
            return terrain_mesh, None
        tv[ids, 2] = floor
        terrain_mesh.vertices = tv

        _with_color(red, LAYER_COLORS["highlight"])
        print(f"[HIGHLIGHT] {zone_prefix}road-like flat pocket: {len(ids)} верш. рельєфу -> підлога "
              f"{floor:.2f}; будинок СТОЇТЬ ({len(red.faces)} граней, дно={floor:.2f} верх={building_top_m:.2f})")
        return terrain_mesh, red
    except Exception as exc:
        print(f"[HIGHLIGHT] {zone_prefix}road-like flat pocket failed (non-fatal): {exc}")
        return terrain_mesh, None
