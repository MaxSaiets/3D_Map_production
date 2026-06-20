import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from services.flat_plate_pipeline import (
    build_flat_building_meshes,
    build_flat_layer_mesh_from_mask,
    build_flat_zone_base_mesh,
    build_keychain_label_mesh,
    build_keychain_layout,
    _build_keychain_base_parts,
    _keychain_body_shape,
    _stretch_geometry_into_bounds,
)


def _square(x0, y0, x1, y1):
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def test_flat_layer_mesh_starts_at_base_top_without_floating():
    scale_factor = 0.2
    base_top_m = 1.0 / scale_factor
    thickness_m = 0.4 / scale_factor

    mesh = build_flat_layer_mesh_from_mask(
        _square(0, 0, 10, 10),
        bottom_z_m=base_top_m,
        thickness_m=thickness_m,
        color=[1, 2, 3, 255],
    )

    assert mesh is not None
    assert float(mesh.bounds[0][2]) == pytest.approx(base_top_m)
    assert float(mesh.bounds[1][2]) == pytest.approx(base_top_m + thickness_m)


def test_flat_zone_base_is_solid_plate_with_bottom_and_top():
    mesh = build_flat_zone_base_mesh(
        _square(0, 0, 10, 10),
        bbox_meters=(0, 0, 10, 10),
        thickness_m=2.5,
    )

    assert mesh is not None
    assert float(mesh.bounds[0][2]) == pytest.approx(0.0)
    assert float(mesh.bounds[1][2]) == pytest.approx(2.5)
    assert mesh.is_watertight


def test_flat_buildings_keep_building_part_heights_and_sit_on_base_top():
    parent = _square(0, 0, 10, 10)
    low = _square(0, 0, 5, 10)
    tower = _square(5, 0, 10, 10)
    gdf = gpd.GeoDataFrame(
        {
            "building": ["yes", None, None],
            "building:levels": [30, 10, 30],
            "building:part": [None, "yes", "yes"],
            "__is_building_part": [False, True, True],
            "geometry": [parent, low, tower],
        }
    )

    request = type(
        "Request",
        (),
        {
            "building_height_multiplier": 1.0,
            "building_min_height": 2.0,
            "include_buildings": True,
        },
    )()
    base_top_m = 3.0

    meshes, _landmarks = build_flat_building_meshes(
        request=request,
        scale_factor=0.2,
        gdf_buildings_local=gdf,
        base_top_m=base_top_m,
    )

    assert len(meshes) == 2
    assert sorted(round(float(mesh.bounds[0][2]), 6) for mesh in meshes) == [base_top_m, base_top_m]
    heights = sorted(round(float(mesh.bounds[1][2] - mesh.bounds[0][2]), 6) for mesh in meshes)
    assert heights == [30.0, 90.0]


def test_flat_buildings_landmark_render_disabled_no_split():
    """Landmark-рендер ВИМКНЕНО власником (LANDMARK_RENDERING_ENABLED=False):
    визначні місця НЕ виокремлюються в бронзову деталь — усі будинки звичайні,
    landmark-список порожній."""
    ordinary = _square(0, 0, 10, 10)
    church = _square(40, 40, 50, 50)
    gdf = gpd.GeoDataFrame(
        {
            "building": ["yes", "church"],
            "building:levels": [3, 2],
            "landmark": ["", "worship"],
            "geometry": [ordinary, church],
        }
    )
    request = type(
        "Request",
        (),
        {
            "building_height_multiplier": 1.0,
            "building_min_height": 2.0,
            "include_buildings": True,
        },
    )()

    meshes, landmarks = build_flat_building_meshes(
        request=request,
        scale_factor=0.2,
        gdf_buildings_local=gdf,
        base_top_m=1.0,
    )

    assert len(landmarks) == 0   # рендер вимкнено → жодного окремого landmark-меша
    assert len(meshes) == 2      # обидва будинки лишаються звичайними


def test_keychain_layout_adds_reinforced_loop_and_reserved_label_band():
    layout = build_keychain_layout(
        bbox_meters=(0, 0, 100, 80),
        scale_factor=1.0,
        model_size_mm=80,
        body_width_mm=78,
        body_height_mm=76,
        loop_center_x_mm=8.5,
        loop_center_y_mm=-4.0,
        loop_outer_radius_mm=6.5,
        loop_inner_radius_mm=3.0,
        corner_radius_mm=4.0,
        label_band_height_mm=9.0,
    )

    assert layout["base"].bounds[3] > 80
    assert layout["base"].contains(layout["content_area"].representative_point())
    assert not layout["base"].covers(layout["loop_hole"].representative_point())
    # ЗА ДИЗАЙНОМ (див. коментар у build_keychain_layout): слот мапи = ВЕСЬ body,
    # label_band НЕ відрізає місце від карти — лише позиціонує текст поверх неї.
    band = layout["label_band"]
    assert band is not None and not band.is_empty
    assert layout["base"].buffer(1e-6).covers(band)
    assert layout["export_size_mm"] > 80


def test_keychain_label_is_fast_separate_raised_mesh():
    body = _square(0, 0, 100, 80)
    band = _square(0, 0, 100, 12)
    mesh = build_keychain_label_mesh(
        "KYIV MAP",
        body_geometry=body,
        label_band_geometry=band,
        bottom_z_m=2.0,
        thickness_m=0.5,
        text_height_m=5.0,
        color=[255, 255, 255, 255],
    )

    assert mesh is not None
    assert float(mesh.bounds[0][2]) == pytest.approx(2.0)
    assert float(mesh.bounds[1][2]) == pytest.approx(2.5)


def test_keychain_label_uses_printable_connected_strokes():
    body = _square(0, 0, 100, 80)
    band = _square(0, 0, 100, 12)
    mesh = build_keychain_label_mesh(
        "KYIV MAP",
        body_geometry=body,
        label_band_geometry=band,
        bottom_z_m=2.0,
        thickness_m=0.5,
        text_height_m=3.5,
        min_stroke_m=0.45,
        color=[255, 255, 255, 255],
    )

    assert mesh is not None
    assert float(mesh.extents[0]) > 10.0
    assert float(mesh.extents[2]) == pytest.approx(0.5)


def test_keychain_map_transform_fills_target_slot_without_letterboxing():
    source = _square(0, 0, 100, 40)
    transformed = _stretch_geometry_into_bounds(
        source,
        source_bounds=(0, 0, 100, 40),
        target_bounds=(10, 20, 50, 100),
    )

    assert transformed is not None
    assert transformed.bounds == pytest.approx((10, 20, 50, 100))


def test_keychain_buildings_are_clamped_to_keychain_height():
    gdf = gpd.GeoDataFrame(
        {
            "building": ["yes"],
            "building:levels": [30],
            "geometry": [_square(0, 0, 10, 10)],
        }
    )
    request = type(
        "Request",
        (),
        {
            "building_height_multiplier": 1.0,
            "building_min_height": 2.0,
            "include_buildings": True,
            "keychain_mode": True,
            "flat_max_building_height_mm": 2.0,
        },
    )()

    meshes, _landmarks = build_flat_building_meshes(
        request=request,
        scale_factor=0.2,
        export_scale_factor=0.2,
        gdf_buildings_local=gdf,
        base_top_m=1.0,
    )

    assert len(meshes) == 1
    assert float(meshes[0].bounds[1][2] - meshes[0].bounds[0][2]) == pytest.approx(10.0)


# ── Нові форми основи (серце/будиночок) ─────────────────────────────────────────
# Система координат: maxy = ВЕРХ брелка (бік петлі), див. build_keychain_layout
# (loop_center_y = body_maxy - y_mm*scale).

def test_keychain_body_shape_heart_is_concave_with_lobes_at_loop_side():
    heart = _keychain_body_shape(0, 0, 46, 42, radius_m=0, shape="heart")
    bbox_area = 46 * 42
    # Серце помітно менше за bbox і строго неопукле (це НЕ прямокутник)
    assert bbox_area * 0.45 < heart.area < bbox_area * 0.80
    assert heart.convex_hull.area > heart.area * 1.04
    # Лоби (важча половина) — вгорі, до maxy = бік петлі; вістря — внизу
    top_half = heart.intersection(_square(0, 21, 46, 42)).area
    bottom_half = heart.intersection(_square(0, 0, 46, 21)).area
    assert top_half > bottom_half


def test_keychain_body_shape_house_has_roof_apex_at_loop_side():
    house = _keychain_body_shape(0, 0, 44, 48, radius_m=0, shape="house")
    roof_h = 48 * 0.38
    expected_area = 44 * 48 - (44 * roof_h / 2)  # bbox мінус два трикутники даху
    assert house.area == pytest.approx(expected_area, rel=0.01)
    assert house.bounds[3] == pytest.approx(48)  # вершина даху сягає maxy
    # Біля верхівки силует вузький (дах сходиться у точку)
    apex_strip = house.intersection(_square(0, 46, 44, 48))
    assert apex_strip.area < 44 * 2 * 0.5


# ── Гравіювання на звороті: розщеплення бази на два watertight-екструди ─────────

def test_keychain_base_split_engraves_letters_into_bottom_layer():
    base = _square(0, 0, 46, 42)
    letters = _square(10, 18, 36, 24)  # «текст» 26×6 мм по центру
    upper, lower = _build_keychain_base_parts(
        base, base_top_m=1.5, back_text_poly=letters, engrave_m=0.5,
    )
    assert upper is not None and lower is not None
    assert upper.is_watertight and lower.is_watertight
    # Шари стикуються рівно: 0..0.5 (гравіювання) + 0.5..1.5 (суцільний)
    assert float(lower.bounds[0][2]) == pytest.approx(0.0)
    assert float(lower.bounds[1][2]) == pytest.approx(0.5)
    assert float(upper.bounds[0][2]) == pytest.approx(0.5)
    assert float(upper.bounds[1][2]) == pytest.approx(1.5)
    # Обʼєм нижнього шару = повний шар мінус обʼєм літер
    assert float(lower.volume) == pytest.approx(46 * 42 * 0.5 - 26 * 6 * 0.5, rel=0.01)
    assert float(upper.volume) == pytest.approx(46 * 42 * 1.0, rel=0.01)


def test_keychain_body_shape_tag_cut_is_at_loop_side():
    # Великий зріз кута «бірки» — біля maxy (бік петлі), як у дизайнер-превʼю.
    # Зріз (0.16·min) більший за радіуси кутів → центроїд зміщується ВНИЗ.
    tag = _keychain_body_shape(0, 0, 50, 30, radius_m=2.0, shape="tag")
    assert tag.centroid.y < 15.0
    # Верхній правий кут зрізаний сильніше за нижній правий
    top_right = tag.intersection(_square(45, 25, 50, 30)).area
    bottom_right = tag.intersection(_square(45, 0, 50, 5)).area
    assert top_right < bottom_right


def test_magnet_pocket_carves_circle_from_bottom_layer():
    from shapely.geometry import Point
    base = _square(0, 0, 60, 60)
    pocket = Point(30, 30).buffer(5.2, resolution=48)
    upper, lower = _build_keychain_base_parts(
        base, base_top_m=3.0, back_text_poly=pocket, engrave_m=2.0,
    )
    assert upper is not None and lower is not None
    assert upper.is_watertight and lower.is_watertight
    import math
    expected_lower = 60 * 60 * 2.0 - math.pi * 5.2 ** 2 * 2.0
    assert float(lower.volume) == pytest.approx(expected_lower, rel=0.01)
    assert float(upper.volume) == pytest.approx(60 * 60 * 1.0, rel=0.01)


def test_keychain_base_split_without_back_text_stays_single_solid_plate():
    upper, lower = _build_keychain_base_parts(
        _square(0, 0, 40, 40), base_top_m=1.5, back_text_poly=None, engrave_m=0.5,
    )
    assert lower is None
    assert upper is not None and upper.is_watertight
    assert float(upper.bounds[0][2]) == pytest.approx(0.0)
    assert float(upper.bounds[1][2]) == pytest.approx(1.5)


# ===== C3 ТОПО-БРЕЛОК: heightfield-рельєф на базі жетона =====

def _make_topo_layout():
    # 350×550м зона → слот 31мм (масштаб ~11.3 м/мм), стандартний брелок 35×55
    return build_keychain_layout(
        bbox_meters=(0.0, 0.0, 350.0, 550.0),
        scale_factor=0.1,
        model_size_mm=55.0,
        body_width_mm=35.0,
        body_height_mm=55.0,
        map_width_mm=31.0,
        map_height_mm=40.0,
        loop_outer_radius_mm=6.5,
        loop_inner_radius_mm=3.0,
        corner_radius_mm=4.0,
        label_band_height_mm=9.0,
    )


class _RadialPeakProvider:
    """Висота — пік у центрі source-зони (175, 275), спадає до країв на 100м."""

    def get_heights_for_points(self, pts):
        import numpy as np
        p = np.asarray(pts, dtype=float)
        d2 = (p[:, 0] - 175.0) ** 2 + (p[:, 1] - 275.0) ** 2
        return 100.0 * np.exp(-d2 / (2 * 120.0 ** 2))


def test_keychain_topo_base_builds_clipped_watertight_relief(monkeypatch):
    import numpy as np
    from services import flat_plate_pipeline as fpp

    monkeypatch.setattr(
        fpp, "_fetch_zone_heightfield_provider", lambda **kw: _RadialPeakProvider()
    )
    layout = _make_topo_layout()
    scale_m_per_mm = float(layout["layout_scale_m_per_mm"])
    export_scale = 1.0 / scale_m_per_mm
    base_top_m = 1.5 * scale_m_per_mm
    relief_m = 2.2 * scale_m_per_mm
    src_b = tuple(float(v) for v in layout["source_bbox"].bounds)
    tgt_b = tuple(float(v) for v in layout["map_target_bounds"])
    # unwrap_params як у run_flat_plate_pipeline (angle=0, COVER scale)
    unwrap = {
        "cx_src": 175.0, "cy_src": 275.0,
        "rect_w": 350.0, "rect_h": 550.0,
        "tgt_cx": (tgt_b[0] + tgt_b[2]) / 2.0, "tgt_cy": (tgt_b[1] + tgt_b[3]) / 2.0,
        "tgt_w": tgt_b[2] - tgt_b[0], "tgt_h": tgt_b[3] - tgt_b[1],
        "angle": 0.0,
    }
    topo, bottom = fpp._build_keychain_topo_base(
        request=object(),
        zone=object(),
        global_center=object(),
        base_mask=layout["base"],
        relief_zone=layout["content_area"],
        base_top_m=base_top_m,
        relief_m=relief_m,
        feather_m=1.5 * scale_m_per_mm,
        unwrap_params=unwrap,
        source_bounds=src_b,
        target_bounds=tgt_b,
        map_rotation_deg=0.0,
        back_text_poly=None,
        engrave_m=0.0,
        export_scale_factor=export_scale,
    )
    assert bottom is None
    assert topo is not None
    assert topo.is_watertight
    z_min = float(topo.bounds[0][2])
    z_max = float(topo.bounds[1][2])
    assert z_min == pytest.approx(0.0, abs=1e-6)
    # Пік у центрі (feather=1 там) → верх ≈ base_top + relief (p98-нормалізація
    # дає невеликий overshoot кліпнутий до 1.0)
    assert z_max == pytest.approx(base_top_m + relief_m, rel=0.08)
    # Вушко (вище за body_maxy) лишається ПЛОСКИМ на base_top
    body_maxy = float(layout["body"].bounds[3])
    verts = topo.vertices
    loop_verts = verts[verts[:, 1] > body_maxy + 0.05 * scale_m_per_mm]
    assert len(loop_verts) > 0
    assert float(loop_verts[:, 2].max()) <= base_top_m + 1e-6


def test_keychain_topo_base_flat_terrain_falls_back_to_none(monkeypatch):
    from services import flat_plate_pipeline as fpp

    class _FlatProvider:
        def get_heights_for_points(self, pts):
            import numpy as np
            return np.full(len(pts), 123.0)

    monkeypatch.setattr(
        fpp, "_fetch_zone_heightfield_provider", lambda **kw: _FlatProvider()
    )
    layout = _make_topo_layout()
    scale_m_per_mm = float(layout["layout_scale_m_per_mm"])
    src_b = tuple(float(v) for v in layout["source_bbox"].bounds)
    tgt_b = tuple(float(v) for v in layout["map_target_bounds"])
    topo, bottom = fpp._build_keychain_topo_base(
        request=object(),
        zone=object(),
        global_center=object(),
        base_mask=layout["base"],
        relief_zone=layout["content_area"],
        base_top_m=1.5 * scale_m_per_mm,
        relief_m=2.2 * scale_m_per_mm,
        feather_m=1.5 * scale_m_per_mm,
        unwrap_params=None,
        source_bounds=src_b,
        target_bounds=tgt_b,
        map_rotation_deg=0.0,
        back_text_poly=None,
        engrave_m=0.0,
        export_scale_factor=1.0 / scale_m_per_mm,
    )
    # Рівнина (range < 0.5м) → None → пайплайн лишає стандартну плоску базу
    assert topo is None and bottom is None


# ===== C2 ПАЗЛ-ПАРА: L-виступ входить у R-паз із клиренсом =====

def test_puzzle_pair_tab_fits_into_notch_with_clearance():
    from shapely.affinity import translate
    from shapely.geometry import box as _box
    from services.flat_plate_pipeline import _keychain_body_shape

    bbox = (0.0, 0.0, 35.0, 42.0)
    rect = _box(*bbox)
    left = _keychain_body_shape(*bbox, radius_m=4.0, shape="puzzle-l")
    right = _keychain_body_shape(*bbox, radius_m=4.0, shape="puzzle-r")

    # L: виступ додає площу і стирчить за праву грань
    assert left.area > rect.area * 0.99
    assert left.bounds[2] > 35.0 + 1.0
    # R: паз зменшує площу, лівий край body не покриває зону паза
    assert right.area < left.area
    k = 35.0 * 0.13
    from shapely.geometry import Point as _Pt
    assert not right.covers(_Pt(0.95 * k, 21.0))

    # Сполучність: tab (частина L за межами rect), перенесений впритул
    # (L.maxx → R.minx), мусить ПОВНІСТЮ влізти в паз (R його не перекриває)
    tab = left.difference(rect).buffer(0)
    assert tab.area > 1.0
    tab_moved = translate(tab, xoff=-35.0)
    overlap = tab_moved.intersection(right).area
    assert overlap < tab.area * 0.01, f"tab перетинає тіло R на {overlap:.3f}мм² — клиренсу нема"


# ===== З'ЄДНУВАЧ-ПАЗИ (метелик/bowtie) для стикування плоских карт =====

def test_map_connector_notches_carved_into_bottom_keep_face_solid():
    from services.flat_plate_pipeline import build_map_connector_geometry
    zone = _square(-400.0, -400.0, 400.0, 400.0)
    export_scale = 80.0 / 800.0  # 80мм модель 800м зони → 0.1 мм/м
    base_top_m = 3.0 / export_scale
    depth_m = 2.0 / export_scale

    notches, keys = build_map_connector_geometry(
        zone, edges="NSEW", span_mm=10.0, length_mm=15.0, waist_frac=0.5,
        clearance_mm=0.2, export_scale_factor=export_scale,
    )
    # 4 пази (по грані) + 4 ключі
    assert len(getattr(notches, "geoms", [notches])) == 4
    assert len(getattr(keys, "geoms", [keys])) == 4
    # ключі лежать ПІД картою (поза footprint основи) — не перетинають базу
    assert keys.bounds[3] <= zone.bounds[1] + 1e-6

    upper, lower = _build_keychain_base_parts(
        zone, base_top_m=base_top_m, back_text_poly=notches, engrave_m=depth_m,
    )
    assert upper is not None and lower is not None
    assert upper.is_watertight and lower.is_watertight
    # ЛИЦЕ суцільне: верхній шар (depth..base_top) покриває всю зону, 1мм цілого
    assert float(upper.bounds[1][2]) == pytest.approx(base_top_m)
    assert float(lower.bounds[0][2]) == pytest.approx(0.0)
    assert float(lower.bounds[1][2]) == pytest.approx(depth_m)
    # паз справді відняв обʼєм з нижнього шару (менше за суцільну плиту)
    assert float(lower.volume) < 800.0 * 800.0 * depth_m * 0.999


def test_map_connector_key_fits_combined_slot_of_two_tiles():
    # Складений шов: плитка A ріже СХІДНУ грань, плитка B — ЗАХІДНУ; повний
    # метелик-ключ мусить влізти в обʼєднання двох пазів (з кліренс-кільцем).
    from shapely.affinity import translate
    from services.flat_plate_pipeline import build_map_connector_geometry
    export_scale = 1.0  # mm == world для прямої перевірки геометрії
    tA = _square(0, 0, 40, 40)
    tB = _square(40, 0, 80, 40)
    nA, kA = build_map_connector_geometry(tA, edges="E", span_mm=10.0, length_mm=15.0,
                                          waist_frac=0.5, clearance_mm=0.2, export_scale_factor=export_scale)
    nB, _ = build_map_connector_geometry(tB, edges="W", span_mm=10.0, length_mm=15.0,
                                         waist_frac=0.5, clearance_mm=0.2, export_scale_factor=export_scale)
    cavity = nA.union(nB).buffer(0)
    # ключ A збудовано біля грані A (без кліренсу) → переносимо його на шов x=40.
    # Простіше: ключ метелика центрований на шві — будуємо прямо.
    from shapely.geometry import Polygon as _P
    w, h, hw = 5.0, 7.5, 7.5 * 0.5
    key = _P([(-w, -h), (-w, h), (0, hw), (w, h), (w, -h), (0, -hw)])
    key = translate(key, xoff=40.0, yoff=20.0)
    assert cavity.contains(key), "ключ не влазить у спільний паз двох плиток"
    assert cavity.area - key.area > 0.05, "немає кліренс-кільця — клинитиме"


def test_map_connector_off_leaves_base_byte_identical_solid():
    # OPT-IN: без map_connector база ІДЕНТИЧНА старій суцільній плиті (golden ОК).
    zone = _square(0, 0, 80, 80)
    solid = build_flat_zone_base_mesh(zone, bbox_meters=zone.bounds, thickness_m=3.0)
    upper, lower = _build_keychain_base_parts(zone, base_top_m=3.0)
    assert lower is None
    assert len(upper.faces) == len(solid.faces)
    assert float(upper.volume) == pytest.approx(float(solid.volume), rel=1e-6)


# ===== ПРЕМІУМ-РАМКА: компас + масштабна лінійка + координати =====

def test_map_frame_overlay_has_compass_scale_coords_inside_base():
    from services.flat_plate_pipeline import build_map_frame_overlay
    zone = _square(-400.0, -400.0, 400.0, 400.0)
    es = 80.0 / 800.0
    overlay = build_map_frame_overlay(
        zone, north=50.4550, south=50.4480, east=30.5270, west=30.5180,
        export_scale_factor=es, want_compass=True, want_scale=True, want_coords=True,
    )
    assert overlay is not None and not overlay.is_empty
    # все в межах плити
    assert zone.buffer(1e-6).contains(overlay)
    # три кластери елементів (компас NE, лінійка SW, координати SE) → багато частин
    parts = list(getattr(overlay, "geoms", [overlay]))
    assert len(parts) >= 3
    # компас угорі-праворуч, лінійка внизу-ліворуч
    cx, cy = 0.0, 0.0
    ne = [p for p in parts if p.centroid.x > cx and p.centroid.y > cy]
    sw = [p for p in parts if p.centroid.x < cx and p.centroid.y < cy]
    assert ne, "немає елементів у верхньому-правому куті (компас)"
    assert sw, "немає елементів у нижньому-лівому куті (лінійка)"


def test_map_frame_each_subfeature_can_be_disabled():
    from services.flat_plate_pipeline import build_map_frame_overlay
    zone = _square(-400.0, -400.0, 400.0, 400.0)
    es = 80.0 / 800.0
    kw = dict(north=50.45, south=50.448, east=30.527, west=30.518, export_scale_factor=es)
    only_compass = build_map_frame_overlay(zone, want_compass=True, want_scale=False, want_coords=False, **kw)
    only_scale = build_map_frame_overlay(zone, want_compass=False, want_scale=True, want_coords=False, **kw)
    none = build_map_frame_overlay(zone, want_compass=False, want_scale=False, want_coords=False, **kw)
    assert only_compass is not None and not only_compass.is_empty
    assert only_scale is not None and not only_scale.is_empty
    assert none is None
    # компас угорі-праворуч, лінійка внизу-ліворуч → центроїди в різних кутах
    assert only_compass.centroid.y > 0 and only_compass.centroid.x > 0
    assert only_scale.centroid.y < 0 and only_scale.centroid.x < 0


# ===== ВИДІЛЕНА БУДІВЛЯ: окрема червона вставна деталь (паз + peg, counterbore) =====

def _building_mesh(poly, base_top, height):
    import trimesh
    m = trimesh.creation.extrude_polygon(poly, height=height)
    m.apply_translation([0, 0, base_top])
    return m


def test_highlight_insert_counterbore_peg_below_building_above():
    from services.flat_plate_pipeline import build_highlight_insert, _mesh_xy_footprint
    es = 80.0 / 800.0
    base_top = 3.0 / es
    # L-shaped (concave) building — convex hull would over-cover the notch
    Lpoly = Polygon([(-30, -24), (30, -24), (30, 0), (0, 0), (0, 24), (-30, 24)])
    b = _building_mesh(Lpoly, base_top, 2.0 / es)
    hi, pocket, depth = build_highlight_insert(b, base_top_m=base_top, export_scale_factor=es)
    assert hi is not None and pocket is not None and depth > 0
    assert hi.is_watertight
    # peg dips below base_top (plugs into pocket); building stays above
    assert float(hi.bounds[0][2]) < base_top - 1e-6
    assert float(hi.bounds[1][2]) > base_top + 1e-6
    # counterbore: pocket opening is SMALLER than the real footprint (shoulder to rest on)
    foot = _mesh_xy_footprint(b, simplify_m=0.05 / es)
    assert pocket.area < foot.area
    assert foot.buffer(1e-6).contains(pocket)


def test_highlight_pocket_carves_into_base_top_watertight():
    from services.flat_plate_pipeline import build_highlight_insert
    es = 80.0 / 800.0
    base_top = 3.0 / es
    b = _building_mesh(_square(-20, -20, 20, 20), base_top, 2.0 / es)
    _, pocket, depth = build_highlight_insert(b, base_top_m=base_top, export_scale_factor=es)
    zone = _square(-400, -400, 400, 400)
    top, bottom = _build_keychain_base_parts(zone, base_top_m=base_top, top_cut_poly=pocket, top_cut_depth_m=depth)
    solid = build_flat_zone_base_mesh(zone, bbox_meters=zone.bounds, thickness_m=base_top)
    assert top is not None and top.is_watertight
    assert bottom is None  # top-cut only → no BaseBack
    assert len(top.faces) > len(solid.faces)  # pocket walls carved in


def test_highlight_select_by_point_and_tiny_glue_on():
    from services.flat_plate_pipeline import build_highlight_insert, _select_highlight_building_index
    es = 80.0 / 800.0
    base_top = 3.0 / es
    far = _building_mesh(_square(100, 100, 140, 140), base_top, 2.0 / es)
    near = _building_mesh(_square(-20, -20, 20, 20), base_top, 2.0 / es)
    # point inside `near` selects index 1, not the far one
    assert _select_highlight_building_index([far, near], target_xy=(0.0, 0.0)) == 1
    # tiny building → glue-on (pocket None), no crash, mesh still returned
    tiny = _building_mesh(_square(-1, -1, 1, 1), base_top, 2.0 / es)
    hi, pocket, depth = build_highlight_insert(tiny, base_top_m=base_top, export_scale_factor=es)
    assert hi is not None and pocket is None
    # SMALL building (~0.7mm, too small for the lip shoulder) still gets a peg via the
    # no-lip tier 2 (key for typical city-map houses) — pocket present, not glue-on
    small = _building_mesh(_square(-3.5, -3.5, 3.5, 3.5), base_top, 2.0 / es)
    hi2, pocket2, depth2 = build_highlight_insert(small, base_top_m=base_top, export_scale_factor=es)
    assert hi2 is not None and pocket2 is not None and depth2 > 0


# ===== ПАРА ДЛЯ ЗАКОХАНИХ: серце-половинки з замком =====

def test_heart_tip_is_rounded():
    # Вістря серця заокруглене (_round_polygon_tip): у смужці 1мм над самою
    # нижньою точкою контур уже помітно ШИРОКИЙ (у гострого — голка <1мм).
    # 2026-06-14: радіус знижено 0.16→0.11 (чіткіший, але не гострий низ) —
    # смужка ~2.5мм; голка дала б <1мм, тож поріг 2.0 підтверджує заокруглення.
    heart = _keychain_body_shape(0, 0, 40, 42, radius_m=4.0, shape="heart")
    miny = heart.bounds[1]
    strip = heart.intersection(_square(0, miny, 40, miny + 1.0))
    assert strip.bounds[2] - strip.bounds[0] > 2.0


def test_heart_pair_halves_assemble_into_full_heart():
    import math
    from shapely.affinity import translate
    from shapely.geometry import Polygon as _Poly

    W, H = 30.0, 44.0
    left = _keychain_body_shape(0, 0, W, H, radius_m=4.0, shape="heart-l")
    right = _keychain_body_shape(0, 0, W, H, radius_m=4.0, shape="heart-r")
    # Еталон = ГОСТРЕ повне серце на 2W (як будує пара всередині — БЕЗ заокруглення
    # вістря, інакше клиповані половинки давали 90°-«гачок» біля шва).
    _raw = []
    for _i in range(160):
        _t = 2.0 * math.pi * _i / 160
        _raw.append((16.0 * math.sin(_t) ** 3,
                     13.0 * math.cos(_t) - 5.0 * math.cos(2 * _t) - 2.0 * math.cos(3 * _t) - math.cos(4 * _t)))
    _xs = [p[0] for p in _raw]; _ys = [p[1] for p in _raw]
    _x0, _x1 = min(_xs), max(_xs); _y0, _y1 = min(_ys), max(_ys)
    full = _Poly([(0 + (px - _x0) / (_x1 - _x0) * 2 * W, 0 + (py - _y0) / (_y1 - _y0) * H) for px, py in _raw]).buffer(0)

    assert left.is_valid and right.is_valid
    # L: замок стирчить за грань розрізу, але лишається в контурі повного серця
    assert left.bounds[2] > W + 1.0
    assert full.buffer(0.2).covers(left)
    # Кожна половинка сходить на шві у МАЛЕНЬКИЙ ПЛАСКИЙ кінчик (~1.7мм зрізано):
    # 0-ширинна голка не друкувалась/відламувалась → зрізаємо у друкований флет,
    # серце все одно читається гострим. Низ біля шва (x≈W), піднятий на _tip_flat.
    lowest = min(left.exterior.coords, key=lambda c: c[1])
    assert abs(lowest[0] - W) < 0.8 and 0.3 < lowest[1] < 2.2, f"низ L не на пласкому шві: {lowest}"
    # Стиковка: жодного перетину тіл при складанні
    overlap = translate(right, xoff=W).intersection(left).area
    assert overlap < 0.5, f"половинки перетинаються на {overlap:.3f}мм²"
    # Складене серце покриває ≥97% площі повного (мінус кліренс замка)
    union_area = translate(right, xoff=W).union(left).area
    assert union_area > full.area * 0.97
    # R: паз реально вирізаний — площа R менша за чисту праву половину серця
    from shapely.geometry import box as _box
    right_half_clean = full.intersection(_box(W, -H, 2 * W, 2 * H))
    assert right.area < right_half_clean.area - 10.0


# ===== МАГНІТ: кілька кишень під шайби Ø4×2мм =====

def test_magnet_pockets_four_corner_ring_inside_square():
    from services.flat_plate_pipeline import build_magnet_pocket_geometry

    zone = _square(0, 0, 60, 60)
    pockets = build_magnet_pocket_geometry(
        zone, diameter_mm=4.4, count=4, inset_mm=8.0, export_scale_factor=1.0,
    )
    parts = list(pockets.geoms) if hasattr(pockets, "geoms") else [pockets]
    assert len(parts) == 4
    import math
    for p in parts:
        # r = diameter/2 + 0.05мм кліренс/бік (легша посадка шайби) → 2.25мм
        assert p.area == pytest.approx(math.pi * 2.25 ** 2, rel=0.02)
        # кишеня + бічна стінка цілком у тілі
        assert zone.contains(p.buffer(0.5))


def test_magnet_pockets_fallback_to_single_centroid_when_shape_too_tight():
    from shapely.geometry import Point as _Pt
    from services.flat_plate_pipeline import build_magnet_pocket_geometry

    # Маленьке коло Ø20: кільце з inset 8 не вміщує кишені → 1 у центрі
    zone = _Pt(10, 10).buffer(10, resolution=48)
    pockets = build_magnet_pocket_geometry(
        zone, diameter_mm=4.4, count=4, inset_mm=8.0, export_scale_factor=1.0,
    )
    parts = list(pockets.geoms) if hasattr(pockets, "geoms") else [pockets]
    assert len(parts) == 1
    assert parts[0].centroid.distance(_Pt(10, 10)) < 0.5


# ===== D4 GPX: трек → буферизований полігон у локальних метрах =====

class _FakeGC:
    # API як у GlobalCenter: lon/lat → to_utm → to_local (НЕ to_local(lon,lat)!)
    def to_utm(self, lon, lat):
        return ((lon - 30.0) * 100000.0, (lat - 50.0) * 100000.0)

    def to_local(self, x_utm, y_utm):
        return (x_utm, y_utm)


def test_gpx_track_polygon_buffers_and_clips_to_zone():
    from services.gpx_track import build_gpx_track_polygon

    zone = _square(0, 0, 1000, 1000)
    # Горизонтальна лінія y=500, x 0..1200 (хвіст 200м за зоною — обрізається)
    track = [[30.0 + 0.012 * i / 39, 50.005] for i in range(40)]
    poly = build_gpx_track_polygon(
        gpx_track=track, global_center=_FakeGC(), zone_polygon_local=zone,
        scale_factor=0.1, width_mm=1.2,
    )
    assert poly is not None
    # Ширина 1.2мм при 0.1мм/м = 12м; довжина в зоні 1000м → ~12000м²
    assert 10000 < poly.area < 14000
    minx, miny, maxx, maxy = poly.bounds
    assert maxx <= 1000.0 + 1e-6 and minx >= -1e-6
    assert abs((miny + maxy) / 2 - 500.0) < 1.0


def test_gpx_track_gps_gap_splits_segments_but_still_builds():
    from services.gpx_track import build_gpx_track_polygon

    track = [[30.000, 50.001], [30.001, 50.001], [30.010, 50.009], [30.011, 50.009]]
    # стрибок (30.001,50.001)→(30.010,50.009) ≈ 1200м > 500м → 2 сегменти
    poly = build_gpx_track_polygon(
        gpx_track=track, global_center=_FakeGC(),
        zone_polygon_local=_square(-100, -100, 1500, 1500),
        scale_factor=0.1, width_mm=1.2,
    )
    assert poly is not None
    parts = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
    assert len(parts) == 2
