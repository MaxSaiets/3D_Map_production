"""Регресія 07.10.2026: мапа-серце (неопукла зона) втрачала ~98 % рельєфу.

clip_mesh_to_polygon_planes різав рельєф півплощинами вздовж кожного ребра → лишалось
опукле ядро форми, а будинки/дороги (обрізані точно по полігону) висіли за основою.
"""
import numpy as np
import pytest
import trimesh
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from services.mesh_clipper import clip_mesh_to_polygon_planes
from services.solidifier_robust import create_solid_terrain_robust
from services.terrain_generator import _snap_and_extract_boundary_from_clipped


def _heart(scale=17.0, n=160):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return Polygon(np.c_[x, y] * scale)


def _puzzle():
    sq = Polygon([(-200, -200), (200, -200), (200, 200), (-200, 200)])
    return unary_union([sq, Point(250, 0).buffer(60)]).difference(Point(-200, 0).buffer(50))


def _grid(step=8.0, half=330.0):
    xs = np.arange(-half, half + 1e-9, step)
    X, Y = np.meshgrid(xs, xs)
    Z = 10 + 5 * np.sin(X / 40) * np.cos(Y / 55)
    n = len(xs)
    j, i = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    a = (j * n + i).ravel()
    F = np.r_[np.c_[a, a + 1, a + n + 1], np.c_[a, a + n + 1, a + n]]
    return trimesh.Trimesh(np.c_[X.ravel(), Y.ravel(), Z.ravel()], F, process=False)


def _area2d(m):
    p = m.vertices[m.faces][:, :, :2]
    return float(np.abs(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])).sum() / 2)


@pytest.mark.parametrize("shape", ["heart", "puzzle", "circle", "rect"])
def test_terrain_clip_covers_whole_shape(shape):
    poly = {
        "heart": _heart(),
        "puzzle": _puzzle(),
        "circle": Point(0, 0).buffer(250, 64),
        "rect": Polygon([(-250, -200), (250, -200), (250, 200), (-250, 200)]),
    }[shape]
    clipped = clip_mesh_to_polygon_planes(_grid(), poly)
    assert clipped is not None
    assert _area2d(clipped) == pytest.approx(poly.area, rel=1e-6)
    assert clipped.is_winding_consistent

    clipped.remove_unreferenced_vertices()
    top, boundary = _snap_and_extract_boundary_from_clipped(clipped, poly, tolerance=0.5)
    solid = create_solid_terrain_robust(top, poly, base_thickness=5.0, floor_z=-5.0, boundary_verts_3d=boundary)
    assert solid is not None and solid.is_watertight and solid.is_volume
    # дно рівно під формою: обʼєм ≈ площа × середня висота над підлогою
    assert solid.volume == pytest.approx(poly.area * 15.0, rel=0.01)
