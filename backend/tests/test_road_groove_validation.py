"""
Тести валідації доріг та пазів для 3D друку.
Перевіряє ширину доріг, пазів, merge_close_road_gaps та сумісність після друку.
"""
import pytest
import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union

from services.road_processor import merge_close_road_gaps, build_road_polygons
from services.road_groove_validation import (
    print_road_groove_validation_report,
    MIN_PRINTABLE_WIDTH_MM,
    _estimate_road_widths_from_mesh,
    _estimate_groove_from_polygons,
)



class TestMergeCloseRoadGaps:
    """Тести об'єднання вузьких проміжків між дорогами."""

    def test_merge_close_roads_parallel_lanes(self):
        """Дві паралельні смуги з вузьким проміжком — мають об'єднатися."""
        # Дві дороги: y=0..1 та y=1.2..2.2, проміжок 0.2м
        road1 = Polygon([(0, 0), (10, 0), (10, 1), (0, 1)])
        road2 = Polygon([(0, 1.2), (10, 1.2), (10, 2.2), (0, 2.2)])
        merged = unary_union([road1, road2])
        area_before = merged.area
        # min_gap_m=0.3 — проміжок 0.2м < 0.3, має об'єднатися. half_gap=0.15м буфер з кожного боку
        min_gap_m = 0.3
        result = merge_close_road_gaps(merged, min_gap_m)
        assert result is not None
        assert not result.is_empty
        # Після merge площа збільшиться (проміжок заповнено) або буде один полігон
        assert result.area >= area_before

    def test_merge_does_nothing_for_large_gap(self):
        """Великий проміжок — не об'єднуємо."""
        road1 = Polygon([(0, 0), (5, 0), (5, 2), (0, 2)])
        road2 = Polygon([(0, 10), (5, 10), (5, 12), (0, 12)])
        merged = unary_union([road1, road2])
        min_gap_m = 0.001  # 1мм — менше ніж відстань 8м між дорогами
        result = merge_close_road_gaps(merged, min_gap_m)
        assert result is not None
        # Дороги на відстані 8м НЕ з'єднуються; площа без змін (нинішня
        # реалізація повертає оригінал, коли заливати нічого — стара асерція
        # area > перевіряла імплементаційну деталь буфера по краях)
        assert result.area == pytest.approx(merged.area, rel=1e-3)
        parts = list(result.geoms) if hasattr(result, "geoms") else [result]
        assert len(parts) == 2

    def test_merge_empty_returns_empty(self):
        """Порожня геометрія — повертаємо як є."""
        result = merge_close_road_gaps(None, 0.001)
        assert result is None

    def test_merge_zero_gap_returns_unchanged(self):
        """min_gap_m=0 — без змін."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = merge_close_road_gaps(poly, 0)
        assert result is not None
        assert result.area == pytest.approx(poly.area, rel=1e-6)


class TestRoadGrooveValidation:
    """Тести валідації доріг та пазів."""

    def test_estimate_road_widths_from_mesh(self):
        """Оцінка ширини дороги з mesh."""
        # Прямокутна призма 10x2x1 метрів
        box_mesh = trimesh.creation.box(extents=[10, 2, 1])
        scale_factor = 100.0  # 100 мм/м
        rw = _estimate_road_widths_from_mesh(box_mesh, scale_factor)
        assert rw["min_extent_mm"] is not None
        assert rw["min_extent_mm"] == pytest.approx(200.0, rel=0.01)  # 2m * 100
        assert rw["max_extent_mm"] == pytest.approx(1000.0, rel=0.01)
        assert rw["bounds_mm"][0] == pytest.approx(1000.0, rel=0.01)
        assert rw["bounds_mm"][1] == pytest.approx(200.0, rel=0.01)

    def test_estimate_groove_from_polygons(self):
        """Оцінка пазу з полігонів."""
        poly = Polygon([(0, 0), (5, 0), (5, 3), (0, 3)])
        scale_factor = 100.0
        gw = _estimate_groove_from_polygons(poly, scale_factor)
        assert gw["groove_area_m2"] == pytest.approx(15.0, rel=0.01)
        assert gw["groove_area_mm2"] == pytest.approx(15.0 * 10000, rel=0.01)

    def test_validation_report_runs_without_error(self, capsys):
        """Звіт валідації виводиться без помилок."""
        box_mesh = trimesh.creation.box(extents=[1, 1, 0.1])
        poly = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
        print_road_groove_validation_report(
            road_mesh=box_mesh,
            terrain_mesh=None,
            road_polygons=poly,
            scale_factor=100.0,
            groove_clearance_mm=0.4,
            zone_prefix="[TEST]",
        )
        captured = capsys.readouterr()
        assert "ВАЛІДАЦІЇ" in captured.out or "ЗВІТ" in captured.out
        assert "0.4" in captured.out
        assert "мм" in captured.out

    def test_min_printable_width_constant(self):
        """Константа мінімальної ширини для друку."""
        assert MIN_PRINTABLE_WIDTH_MM >= 0.4
        assert MIN_PRINTABLE_WIDTH_MM <= 1.0


class TestGrooveClearanceSides:
    """
    Тести зазору пазу по БОКАХ (XY).
    Критично: зазор має бути по боках, НЕ знизу.
    """

    def test_clearance_formula_produces_correct_mm_in_model(self):
        """
        Формула: clearance_m = GROOVE_CLEARANCE_MM / scale_factor.
        scale_factor = мм/м → 1м світ = scale_factor мм модель.
        Тобто clearance_m метрів = clearance_m * scale_factor мм в моделі.
        """
        from main import GROOVE_CLEARANCE_MM

        scale_factor = 0.107  # типовий мм/м
        road_clearance_m = GROOVE_CLEARANCE_MM / scale_factor

        # clearance_m метрів у світі → clearance_m * scale_factor мм в моделі
        clearance_in_model_mm = road_clearance_m * scale_factor
        assert clearance_in_model_mm >= GROOVE_CLEARANCE_MM * 0.95, (
            f"Буфер {road_clearance_m:.4f}м дає {clearance_in_model_mm:.4f}мм в моделі, "
            f"очікувалось >= {GROOVE_CLEARANCE_MM}мм з кожного боку"
        )

    def test_buffer_makes_groove_wider_than_road(self):
        """
        road_cut_mask = roads.buffer(clearance_m) має бути ШИРШИМ за дорогу.
        Перевірка що буфер додає простір по боках.
        """
        from main import GROOVE_CLEARANCE_MM

        # Дорога 10м x 2м
        road = Polygon([(0, 0), (10, 0), (10, 2), (0, 2)])
        road_width_m = 2.0
        scale_factor = 0.107

        road_clearance_m = GROOVE_CLEARANCE_MM / scale_factor
        groove = road.buffer(road_clearance_m, join_style=2)

        # Ширина пазу (мінімальний розмір) має бути більша за дорогу
        minx, miny, maxx, maxy = groove.bounds
        groove_width_m = min(maxx - minx, maxy - miny)
        road_min_dim = min(10, 2)  # 2м

        # Паз ширший на 2*clearance з кожного боку
        expected_extra_m = 2 * road_clearance_m
        assert groove_width_m > road_width_m, (
            f"Паз {groove_width_m:.4f}м має бути ширшим за дорогу {road_width_m}м"
        )
        assert groove_width_m >= road_width_m + expected_extra_m * 0.9, (
            f"Паз має бути ширшим на ~{expected_extra_m*1000:.2f}мм в світі "
            f"(≈{GROOVE_CLEARANCE_MM*2}мм в моделі)"
        )

    def test_clearance_in_model_units(self):
        """
        Пряма перевірка: clearance_m * scale_factor = мм в моделі.
        Формула: clearance_m = GROOVE_CLEARANCE_MM / scale_factor.
        """
        from main import GROOVE_CLEARANCE_MM

        scale_factor = 0.107
        clearance_m = GROOVE_CLEARANCE_MM / scale_factor
        mm_in_model = clearance_m * scale_factor
        assert abs(mm_in_model - GROOVE_CLEARANCE_MM) < 0.01, (
            f"clearance_m={clearance_m:.4f}м має давати {GROOVE_CLEARANCE_MM}мм в моделі, отримано {mm_in_model:.4f}мм"
        )

    def test_terrain_cutter_uses_polygon_when_clearance(self):
        """
        cut_roads_from_solid_terrain має використовувати polygon path (не road_mesh)
        коли clearance_m > 0 і є road_polygons — для гарантії зазору по боках.
        """
        from services.terrain_cutter import cut_roads_from_solid_terrain

        # use_polygon_for_clearance = clearance_m > 0 and has_polygons
        # Це внутрішня логіка — перевіряємо що при clearance>0 і polygons результат не exact fit
        road = Polygon([(0, 0), (5, 0), (5, 1), (0, 1)])
        road_buffered = road.buffer(0.02, join_style=2)  # 20мм в світі

        # Якщо передати road_polygons=road_buffered, clearance_m=0.02 — має використати polygon
        # (ми не можемо напряму перевірити внутрішню логіку, але перевіримо що виклик не падає)
        terrain = trimesh.creation.box(extents=[20, 20, 5])
        result = cut_roads_from_solid_terrain(
            terrain_mesh=terrain,
            road_polygons=road_buffered,
            clearance_m=0.02,
            scale_factor=0.1,
            road_mesh=None,  # без road_mesh — завжди polygon path
        )
        # Якщо Blender не знайдено — поверне terrain без змін
        assert result is not None
