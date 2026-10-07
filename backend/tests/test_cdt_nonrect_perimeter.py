"""07.10.2026: CDT-рельєф (CDT_GROOVE=1) для НЕпрямокутних зон (коло, шестикутник, серце).

Баг: `edge_is_perim` вважав краєм зони лише ребра на сторонах ОБМЕЖУВАЛЬНОГО ПРЯМОКУТНИКА.
Для кола/серця стінка периметра будувалась лише до дна пазів (slab_z), а не до низу плити
(floor_z), і з орієнтацією стінки паза (всередину). Дно плити відривалось окремим «листом»
нульової товщини, тіло моделі запечатувалось на slab_z → у слайсері вивернуті грані знизу
(прод 6336d340: серце 200 мм, Біла Церква, «З рельєфом»).
"""
import math

import numpy as np
import pytest
from shapely.geometry import Polygon, box

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("triangle")

from services.cdt_groove_terrain import build_cdt_grooved_terrain  # noqa: E402


def _heart(w=400.0, h=360.0, n=160):
    raw = []
    for i in range(n):
        t = 2 * math.pi * i / n
        raw.append((16 * math.sin(t) ** 3, 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)))
    xs, ys = [p[0] for p in raw], [p[1] for p in raw]
    s = min(w / (max(xs) - min(xs)), h / (max(ys) - min(ys)))
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    return Polygon([((x - cx) * s, (y - cy) * s) for x, y in raw])


def _circle(r=180.0, n=48):
    return Polygon([(math.cos(2 * math.pi * i / n) * r, math.sin(2 * math.pi * i / n) * r) for i in range(n)])


def _height(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return 30.0 + 2.0 * np.sin(x / 40.0) + 1.5 * np.cos(y / 55.0)


def _road_mask(zone):
    # хрест доріг через центр — доходить до краю зони (як справжня дорожня сітка)
    return (box(-200, -6, 200, 6).union(box(-6, -200, 6, 200))).intersection(zone)


@pytest.mark.parametrize("zone_fn", [_heart, _circle], ids=["heart", "circle"])
def test_nonrect_zone_keeps_plate_down_to_floor(zone_fn):
    zone = zone_fn()
    floor_z, slab_z = 10.0, 24.0
    mesh = build_cdt_grooved_terrain(zone, _road_mask(zone), _height,
                                     slab_z=slab_z, floor_z=floor_z, seg_len=4.0, tri_area=40.0)
    assert mesh is not None and len(mesh.faces) > 0
    # тіло тягнеться до низу плити, а не «запечатане» на дні пазів
    assert float(mesh.bounds[0][2]) == pytest.approx(floor_z, abs=1e-3)
    comps = mesh.split(only_watertight=False)
    # жодних плоских «листів» нульової товщини
    sheets = [c for c in comps if float(c.bounds[1][2] - c.bounds[0][2]) < 1e-6]
    assert sheets == []
    main = max(comps, key=lambda c: len(c.faces))
    assert float(main.bounds[0][2]) == pytest.approx(floor_z, abs=1e-3)
    assert bool(main.is_volume)
    # стінки периметра дивляться НАЗОВНІ: вертикальні грані на краю зони
    n = main.face_normals
    c = main.triangles_center
    vertical = np.abs(n[:, 2]) < 0.2
    from shapely import distance as _dist, points as _pts
    on_edge = _dist(zone.exterior, _pts(c[:, :2])) < 0.05
    walls = vertical & on_edge
    assert walls.sum() > 20
    # нормаль назовні: крок уздовж нормалі виводить ЗА межі зони
    probe = c[walls, :2] + n[walls, :2] * 0.5
    from shapely import contains_xy
    outside = ~contains_xy(zone, probe[:, 0], probe[:, 1])
    assert outside.mean() > 0.95


def test_rect_zone_unchanged_behaviour():
    zone = box(-200, -150, 200, 150)
    floor_z, slab_z = 10.0, 24.0
    mesh = build_cdt_grooved_terrain(zone, _road_mask(zone), _height,
                                     slab_z=slab_z, floor_z=floor_z, seg_len=4.0, tri_area=40.0)
    assert float(mesh.bounds[0][2]) == pytest.approx(floor_z, abs=1e-3)
    assert bool(mesh.is_volume)
