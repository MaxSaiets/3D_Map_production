"""GPX Phase 2 (backend-геометрія) — фокусні unit-тести road-snap / nature-simplify /
validity. Без мережі/БД: синтетичні road_lines_local (shapely LineString у локальних
метрах) + синтетичний трек. global_center — тривіальний identity-стаб, що трактує
вхідні (x, y) як уже-локальні метри, тож тест керує локальними координатами напряму.

Запуск:  cd backend && python -m pytest tests/test_gpx_phase2.py -v
"""
from __future__ import annotations

import math

import pytest
from shapely.geometry import LineString, Point

from services.gpx_track import (
    GPX_PHASE2_ENABLED,
    _ensure_valid_polygon,
    _flatten_road_lines,
    _snap_coords_to_roads,
    build_gpx_track_polygon,
    gpx_track_to_local_geometry,
)


class _IdentityCenter:
    """Стаб GlobalCenter: координати ВЖЕ локальні. У продакшні шлях lon/lat→UTM→local,
    тут to_utm/to_local — тотожність, тож тест подає локальні метри напряму як [x, y]."""

    def to_utm(self, lon, lat):
        return (float(lon), float(lat))

    def to_local(self, x_utm, y_utm):
        return (float(x_utm), float(y_utm))


# ── (a) трек паралельно-але-зі-зсувом від прямої дороги → snap НА дорогу ──────────
def test_offset_track_snaps_onto_parallel_road():
    # Дорога: пряма по y=0 від x=0 до x=100.
    road = LineString([(0.0, 0.0), (100.0, 0.0)])
    road_lines = [road]
    # Трек: паралельний, зсунутий на 3м угору (в межах порогу ~18м).
    track = [(10.0, 3.0), (30.0, 3.0), (50.0, 3.0), (70.0, 3.0), (90.0, 3.0)]

    snapped = _snap_coords_to_roads(track, road_lines, snap_threshold_m=18.0)

    assert len(snapped) == len(track)
    # Кожна точка має лягти НА вісь дороги (y≈0), тобто притягнутись.
    for x, y in snapped:
        assert abs(y) < 1e-6, f"очікувалось y≈0 (на дорозі), отримано {y}"
        assert -1e-6 <= x <= 100.0 + 1e-6


# ── (b) далекий від доріг трек (природа/парк) лишається БЕЗ змін ──────────────────
def test_far_from_road_track_is_unchanged():
    road = LineString([(0.0, 0.0), (100.0, 0.0)])
    road_lines = [road]
    # Трек далеко (y=200м) — поза будь-яким розумним порогом → НЕ чіпати.
    track = [(10.0, 200.0), (30.0, 205.0), (50.0, 198.0)]

    snapped = _snap_coords_to_roads(track, road_lines, snap_threshold_m=18.0)

    assert snapped == [(10.0, 200.0), (30.0, 205.0), (50.0, 198.0)]


def test_mixed_track_snaps_only_near_vertices():
    """Змішаний трек: частина біля дороги (snap), частина далеко (без змін)."""
    road = LineString([(0.0, 0.0), (100.0, 0.0)])
    track = [(10.0, 2.0), (50.0, 2.0), (50.0, 150.0), (60.0, 150.0)]
    snapped = _snap_coords_to_roads(track, [road], snap_threshold_m=18.0)
    # Перші дві близькі → на дорогу (y≈0)
    assert abs(snapped[0][1]) < 1e-6
    assert abs(snapped[1][1]) < 1e-6
    # Останні дві далеко → незмінні
    assert snapped[2] == (50.0, 150.0)
    assert snapped[3] == (60.0, 150.0)


# ── (c) фінальний полігон треку валідний і непорожній ────────────────────────────
def test_track_polygon_is_valid_and_nonempty_city():
    road = LineString([(0.0, 0.0), (100.0, 0.0)])
    track = [(10.0, 3.0), (30.0, 2.0), (50.0, 4.0), (70.0, 3.0), (90.0, 2.0)]
    poly = build_gpx_track_polygon(
        gpx_track=track,
        global_center=_IdentityCenter(),
        zone_polygon_local=None,
        scale_factor=0.1,  # 0.1мм/м → world half-width відчутна
        width_mm=1.2,
        road_lines_local=road,  # передаємо одну LineString (каллер інколи так)
    )
    assert poly is not None
    assert poly.is_valid
    assert not poly.is_empty
    assert poly.area > 0


def test_track_polygon_is_valid_and_nonempty_nature():
    # Без доріг (природа): має лишитись валідним після simplify/smooth/buffer.
    track = [
        (0.0, 0.0), (20.0, 15.0), (40.0, -10.0), (60.0, 20.0),
        (80.0, -5.0), (100.0, 10.0),
    ]
    poly = build_gpx_track_polygon(
        gpx_track=track,
        global_center=_IdentityCenter(),
        zone_polygon_local=None,
        scale_factor=0.1,
        width_mm=1.5,
        road_lines_local=None,
    )
    assert poly is not None
    assert poly.is_valid
    assert not poly.is_empty
    assert poly.area > 0


def test_self_crossing_track_yields_valid_polygon():
    """Самоперетинний (вісімка) трек → буфер+make_valid дає валідний полігон."""
    track = [
        (0.0, 0.0), (50.0, 50.0), (50.0, 0.0), (0.0, 50.0), (0.0, 0.0),
    ]
    poly = build_gpx_track_polygon(
        gpx_track=track,
        global_center=_IdentityCenter(),
        zone_polygon_local=None,
        scale_factor=0.2,
        width_mm=1.5,
        road_lines_local=None,
    )
    assert poly is not None
    assert poly.is_valid
    assert poly.area > 0


# ── Захисні/regression тести ─────────────────────────────────────────────────────
def test_phase2_default_path_no_roads_unchanged():
    """Регресія: без road_lines_local snap-функція повертає вхід БЕЗ змін
    (дефолтний шлях не повинен мінятись)."""
    track = [(1.0, 1.0), (2.0, 2.0)]
    assert _snap_coords_to_roads(track, None, 18.0) == track
    assert _snap_coords_to_roads(track, [], 18.0) == track
    # поріг 0 → теж no-op
    road = LineString([(0.0, 0.0), (10.0, 0.0)])
    assert _snap_coords_to_roads(track, [road], 0.0) == track


def test_flatten_road_lines_handles_multilinestring():
    from shapely.geometry import MultiLineString

    mls = MultiLineString([[(0, 0), (10, 0)], [(0, 5), (10, 5)]])
    single = LineString([(0, 10), (10, 10)])
    flat = _flatten_road_lines([mls, single, None])
    assert len(flat) == 3
    assert all(isinstance(g, LineString) for g in flat)


def test_flatten_road_lines_empty_inputs():
    assert _flatten_road_lines(None) == []
    assert _flatten_road_lines([]) == []
    # порожня лінія відкидається
    assert _flatten_road_lines([LineString()]) == []


def test_ensure_valid_polygon_repairs_bowtie():
    # «Метелик» (bowtie) — класичний невалідний self-intersect полігон.
    from shapely.geometry import Polygon

    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    assert not bowtie.is_valid
    fixed = _ensure_valid_polygon(bowtie)
    assert fixed is not None
    assert fixed.is_valid
    assert fixed.area > 0


def test_ensure_valid_polygon_empty_returns_none():
    from shapely.geometry import Polygon

    assert _ensure_valid_polygon(None) is None
    assert _ensure_valid_polygon(Polygon()) is None


def test_snap_projects_to_nearest_point_on_segment():
    """Проєкція має давати найближчу ТОЧКУ осі (нога перпендикуляра), не вершину."""
    road = LineString([(0.0, 0.0), (100.0, 0.0)])
    # точка над серединою дороги
    snapped = _snap_coords_to_roads([(50.0, 5.0)], [road], snap_threshold_m=18.0)
    assert snapped[0] == pytest.approx((50.0, 0.0), abs=1e-6)


def test_zone_clip_outside_returns_none():
    """Трек повністю поза зоною → шар пропускається (None)."""
    from shapely.geometry import box

    track = [(0.0, 0.0), (10.0, 0.0)]
    zone = box(1000.0, 1000.0, 1100.0, 1100.0)  # далеко від треку
    poly = build_gpx_track_polygon(
        gpx_track=track,
        global_center=_IdentityCenter(),
        zone_polygon_local=zone,
        scale_factor=0.1,
        width_mm=1.2,
        road_lines_local=None,
    )
    assert poly is None


def test_gpx_to_local_geometry_basic_line():
    track = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    geom = gpx_track_to_local_geometry(track, _IdentityCenter())
    assert geom is not None
    assert geom.geom_type in ("LineString", "MultiLineString")
    assert geom.length > 0


def test_phase2_toggle_present():
    # kill-switch існує (булевий модуль-рівень прапор).
    assert isinstance(GPX_PHASE2_ENABLED, bool)
