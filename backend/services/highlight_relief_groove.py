"""
Highlight-будинок на РЕЛЬЄФІ — паз ЯК У ДОРІГ (ЧИСТІ стінки) + будинок СТОЇТЬ.

Власник: «низ будинку доробляється до НАЙНИЖЧОЇ точки свого мешу щоб він СТОЯВ, і
вирізаються пази як у доріг із зазором — стінки мають бути ЧИСТІ, не обривчасті».
  • Будинок = пласка призма footprint від building_bottom (найнижча z мешу) до
    building_top — пласке дно, СТОЇТЬ.
  • Паз = ЧИСТА boolean-кишеня (як дороги): прямокутний різак footprint+зазор від
    дна будинку вгору крізь рельєф, manifold-difference. Стінки = рівно по контуру
    (зазор 0.15мм), БЕЗ сходинок. Викликати ДО merge (рельєф ще герметичний том!).
  • FALLBACK (якщо рельєф негерметичний → boolean fail): опускання вершин (надійно,
    але стінки обривчасті — лише як запасний варіант).
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


def _vertex_set_fallback(terrain_mesh, groove, floor, zone_prefix):
    """Запасний паз опусканням вершин (стінки обривчасті — лише коли boolean недоступний)."""
    N = terrain_mesh.face_normals
    up = N[:, 2] > 0.3
    tv = terrain_mesh.vertices.copy()
    topverts = np.unique(terrain_mesh.faces[up].ravel())
    tvp = tv[topverts]
    gminx, gminy, gmaxx, gmaxy = groove.bounds
    bbok = (tvp[:, 0] >= gminx) & (tvp[:, 0] <= gmaxx) & (tvp[:, 1] >= gminy) & (tvp[:, 1] <= gmaxy)
    cand = topverts[bbok]
    if len(cand) == 0:
        return terrain_mesh
    inside = _contains_mask(groove, tv[cand][:, :2])
    ids = cand[inside]
    ids = ids[tv[ids, 2] > floor]
    if len(ids) == 0:
        return terrain_mesh
    tv[ids, 2] = floor
    terrain_mesh.vertices = tv
    print(f"[HIGHLIGHT] {zone_prefix}fallback vertex-set pocket: {len(ids)} верш. -> {floor:.2f}")
    return terrain_mesh


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
    """ЧИСТА boolean-кишеня (паз) під будинок, що СТОЇТЬ + червоний будинок з пласким дном.

    Викликати ДО terrain_building_merge (рельєф ще герметичний → boolean ЧИСТИЙ).
    Повертає (terrain_mesh_з_пазом, red_building) або (terrain_mesh, None) на невдачі.
    Одиниці: світові метри; clearance — модель-мм /scale_factor.
    """
    if terrain_mesh is None or footprint is None or getattr(footprint, "is_empty", True):
        return terrain_mesh, None
    try:
        sf = float(scale_factor)
        if sf <= 0:
            return terrain_mesh, None
        clear = float(clearance_mm) / sf
        floor = float(building_bottom_m)
        thick = float(building_top_m) - floor
        if thick <= 1e-6:
            return terrain_mesh, None

        from services.flat_plate_pipeline import build_flat_layer_mesh_from_mask, _with_color, LAYER_COLORS

        # Будинок: ПЛАСКА призма (footprint, дно=floor, верх=building_top) — СТОЇТЬ.
        red = build_flat_layer_mesh_from_mask(
            footprint, bottom_z_m=floor, thickness_m=thick,
            color=LAYER_COLORS["highlight"], min_area_m2=1e-12,
        )
        if red is None or len(red.faces) == 0:
            return terrain_mesh, None
        _with_color(red, LAYER_COLORS["highlight"])

        groove = footprint.buffer(clear, join_style=2)

        # ЧИСТА boolean-кишеня: різак footprint+зазор від floor вгору крізь рельєф.
        terr_top = float(terrain_mesh.bounds[1][2])
        cut_thick = (terr_top - floor) + max(2.0 / sf, 1.0)
        cutter = build_flat_layer_mesh_from_mask(
            groove, bottom_z_m=floor, thickness_m=cut_thick,
            color=[128, 128, 128], min_area_m2=1e-12,
        )
        import os
        cutter_ok = (cutter is not None and len(cutter.faces) > 0
                     and bool(getattr(cutter, "is_volume", False)))
        force_blender = os.environ.get("HL_FORCE_BLENDER") == "1"

        def _drift_ok(res, b0):
            if res is None or len(getattr(res, "faces", [])) == 0:
                return False
            b1 = res.bounds
            drift = (max(abs(b1[0][i] - b0[0][i]) for i in (0, 1))
                     + max(abs(b1[1][i] - b0[1][i]) for i in (0, 1)))
            return drift < (5.0 / sf)

        # 1) manifold — швидко, ЧИСТІ стінки, але потребує ГЕРМЕТИЧНИЙ рельєф.
        if cutter_ok and not force_blender and bool(getattr(terrain_mesh, "is_volume", False)):
            b0 = terrain_mesh.bounds
            try:
                res = trimesh.boolean.difference([terrain_mesh, cutter], engine="manifold")
            except Exception:
                res = None
            if res is not None and bool(getattr(res, "is_volume", False)) and _drift_ok(res, b0):
                print(f"[HIGHLIGHT] {zone_prefix}CLEAN manifold pocket (стінки рівні, зазор "
                      f"{clearance_mm}мм); будинок СТОЇТЬ {len(red.faces)} граней")
                return res, red

        # 2) Blender boolean — ріже ЧИСТО і на НЕгерметичному рельєфі (саме твої зони,
        #    де manifold не проходив і падав на кривий vertex-set).
        if cutter_ok:
            try:
                from services.terrain_cutter import _run_blender_boolean
                b0 = terrain_mesh.bounds
                bres = _run_blender_boolean(terrain_mesh, cutter, label="highlight")
                if (bres is not None and len(getattr(bres, "faces", [])) > 0
                        and abs(len(bres.faces) - len(terrain_mesh.faces)) > 4  # реально вирізало
                        and _drift_ok(bres, b0)):
                    print(f"[HIGHLIGHT] {zone_prefix}CLEAN Blender pocket (рівні стінки на "
                          f"негерметичному рельєфі); будинок СТОЇТЬ {len(red.faces)} граней")
                    return bres, red
            except Exception as _bexc:
                print(f"[HIGHLIGHT] {zone_prefix}blender pocket failed (non-fatal): {_bexc}")

        # 3) ОСТАННІЙ резерв (обидва boolean не вийшли): vertex-set — обривчасто.
        terrain_mesh = _vertex_set_fallback(terrain_mesh, groove, floor, zone_prefix)
        return terrain_mesh, red
    except Exception as exc:
        print(f"[HIGHLIGHT] {zone_prefix}highlight pocket failed (non-fatal): {exc}")
        return terrain_mesh, None
