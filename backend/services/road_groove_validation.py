"""
Валідація доріг та пазів для 3D друку.
Виводить точні розміри після генерації моделі: ширина доріг, пазів, перевірка сумісності.
"""
from typing import Optional, Any
import numpy as np


# Мінімальна ширина для надійного друку (мм) — діаметр сопла FDM
MIN_PRINTABLE_WIDTH_MM = 0.45


def _estimate_road_widths_from_mesh(road_mesh, scale_factor: float) -> dict:
    """Оцінка ширини доріг з mesh (bounding box та апроксимація по вершинах)."""
    out = {
        "bounds_mm": None,
        "min_extent_mm": None,
        "max_extent_mm": None,
        "road_area_mm2": None,
    }
    if road_mesh is None or len(road_mesh.vertices) == 0 or not scale_factor:
        return out
    try:
        b = road_mesh.bounds
        dx_m = float(b[1][0] - b[0][0])
        dy_m = float(b[1][1] - b[0][1])
        out["bounds_mm"] = (dx_m * scale_factor, dy_m * scale_factor)
        out["min_extent_mm"] = min(dx_m, dy_m) * scale_factor
        out["max_extent_mm"] = max(dx_m, dy_m) * scale_factor
        # Орієнтовна площа доріг у mm2 (проекція на XY)
        if hasattr(road_mesh, "area"):
            out["road_area_mm2"] = road_mesh.area * (scale_factor ** 2)
    except Exception:
        pass
    return out


def _estimate_groove_from_polygons(road_polygons, scale_factor: float) -> dict:
    """Оцінка пазів з полігонів (road_cut_mask = roads.buffer(clearance) — вже з зазором)."""
    out = {
        "groove_area_m2": None,
        "groove_area_mm2": None,
    }
    if road_polygons is None or road_polygons.is_empty or not scale_factor:
        return out
    try:
        out["groove_area_m2"] = float(road_polygons.area)
        out["groove_area_mm2"] = out["groove_area_m2"] * (scale_factor ** 2)
    except Exception:
        pass
    return out


def print_road_groove_validation_report(
    road_mesh=None,
    terrain_mesh=None,
    road_polygons=None,
    scale_factor: Optional[float] = None,
    groove_clearance_mm: float = 0.4,
    zone_prefix: str = "",
) -> None:
    """
    Виводить звіт валідації доріг та пазів після генерації моделі.
    Перевіряє ширини, зазори та сумісність для 3D друку.
    """
    if scale_factor is None or scale_factor <= 0:
        return
    groove_total_extra_mm = groove_clearance_mm * 2

    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"{zone_prefix} ЗВІТ ВАЛІДАЦІЇ ДОРІГ ТА ПАЗІВ (після генерації)")
    lines.append("=" * 70)

    # Параметри масштабу
    lines.append(f"  Масштаб: scale_factor = {scale_factor:.4f} мм/м")
    lines.append(f"  Зазор пазу: {groove_clearance_mm} мм з кожного боку (всього +{groove_total_extra_mm} мм)")
    lines.append(f"  Мін. друкована ширина: {MIN_PRINTABLE_WIDTH_MM} мм")
    lines.append("")

    # Дороги з mesh
    if road_mesh is not None and len(road_mesh.vertices) > 0:
        rw = _estimate_road_widths_from_mesh(road_mesh, scale_factor)
        lines.append("  --- ДОРІГИ (mesh) ---")
        if rw["bounds_mm"]:
            lines.append(f"    Bounds XY (мм): {rw['bounds_mm'][0]:.3f} x {rw['bounds_mm'][1]:.3f}")
        if rw["min_extent_mm"] is not None:
            ok = "[OK]" if rw["min_extent_mm"] >= MIN_PRINTABLE_WIDTH_MM else "[WARN]"
            lines.append(f"    Мін. розмір дороги: {rw['min_extent_mm']:.3f} мм {ok}")
        if rw["road_area_mm2"] is not None:
            lines.append(f"    Орієнтовна площа: {rw['road_area_mm2']:.1f} mm2")
        lines.append("")
    else:
        lines.append("  --- ДОРІГИ: mesh відсутній ---")
        lines.append("")

    # Паз (groove) з полігонів
    if road_polygons is not None and not getattr(road_polygons, "is_empty", True):
        gw = _estimate_groove_from_polygons(road_polygons, scale_factor)
        lines.append("  --- ПАЗ (groove) ---")
        if gw["groove_area_mm2"] is not None:
            lines.append(f"    Площа пазу: {gw['groove_area_mm2']:.1f} mm2")
        lines.append(f"    Зазор (clearance): {groove_clearance_mm} мм з кожного боку")
        lines.append("")
    else:
        lines.append("  --- ПАЗ: полігони відсутні ---")
        lines.append("")

    # Перевірка сумісності
    lines.append("  --- ПЕРЕВІРКА СУМІСНОСТІ ---")
    if road_mesh is not None and len(road_mesh.vertices) > 0:
        rw = _estimate_road_widths_from_mesh(road_mesh, scale_factor)
        if rw["min_extent_mm"] is not None:
            road_plus_clearance = rw["min_extent_mm"] + groove_total_extra_mm
            lines.append(f"    Дорога + зазор з обох боків: {rw['min_extent_mm']:.3f} + {groove_total_extra_mm} = {road_plus_clearance:.3f} мм")
            lines.append(f"    Паз має бути >= {road_plus_clearance:.3f} мм для коректної вставки")
            if rw["min_extent_mm"] >= MIN_PRINTABLE_WIDTH_MM:
                lines.append(f"    [OK] Ширина дороги достатня для друку (>= {MIN_PRINTABLE_WIDTH_MM} мм)")
            else:
                lines.append(f"    [WARN] Ширина дороги < {MIN_PRINTABLE_WIDTH_MM} мм - може не друкуватися")
    lines.append("")
    lines.append("=" * 70)
    lines.append("")

    for line in lines:
        print(line)
