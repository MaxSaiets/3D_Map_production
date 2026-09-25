# -*- coding: utf-8 -*-
"""Режим «Гори»: агент розуміє опис без помилок, spec завжди безпечний, геометрія герметична.
Без мережі: DEM підміняється синтетичною пірамідою."""
import numpy as np
import pytest

from services.mountains import agent as A
from services.mountains.mesh import apply_slope_band, frame_solid, tile_cuts


@pytest.mark.parametrize("text,size,height,fw,style,sides,figs", [
    ("Матергорн 40 см висотою 25 см, ободок 10 мм заокруглений, скельні стінки", 400, 250, 10, "rounded", "rock", []),
    ("Говерла 20x20 см висотою 8 см з хатиною", 200, 80, 10, "rounded", "slope", ["alpine_hut"]),
    ("Монблан 25 см без ободка, 2 людей по 20 мм", 250, None, 0, "none", "slope", ["hiker_wave", "hiker_wave"]),
    ("Everest 30cm, no frame, 3 people, rocky sides", 300, None, 0, "none", "rock", ["hiker_wave"] * 3),
    ("гора висотою 10 см розміром 15 см Фудзі, скелелаз лізе по канату", 150, 100, 10, "rounded", "slope", ["climber_rope"]),
])
def test_rule_parser(text, size, height, fw, style, sides, figs):
    r = A.understand(text)
    s = r["spec"]
    assert s["place"] is not None and s["place"]["source"] == "preset"
    assert s["size_mm"] == size and s["height_mm"] == height
    assert s["frame"]["width_mm"] == fw and s["frame"]["style"] == style and s["sides"] == sides
    assert [f["id"] for f in s["figures"]] == figs
    assert r["understood"] and r["questions"] == []


def test_no_place_asks_question():
    r = A.understand("хочу гору 20 см")
    assert r["spec"]["place"] is None and r["questions"]


def test_normalize_clamps_everything():
    s, warn = A.normalize_spec({"place": {"lat": 48.16, "lon": 24.5}, "size_mm": 9999, "height_mm": 5000, "frame": {"style": "weird", "width_mm": 300, "height_mm": -5},
                                "sides": "??", "figures": [{"id": "nope"}, {"id": "hiker_wave", "where": "moon", "height_mm": 999, "fx": 7}]})
    assert s["size_mm"] == 400 and s["height_mm"] == 300 and s["frame"]["style"] == "rounded"
    assert s["frame"]["width_mm"] <= 400 * 0.12 and s["frame"]["height_mm"] >= 3 and s["sides"] == "slope"
    assert [f["id"] for f in s["figures"]] == ["hiker_wave"] and s["figures"][0]["where"] == "summit" and s["figures"][0]["height_mm"] <= 40
    assert warn


def _pyramid(G=101, cell=1.0, h=60.0):
    xs = np.arange(G) * cell; X, Y = np.meshgrid(xs, xs)
    return 3.0 + h * np.clip(1 - np.maximum(np.abs(X - xs[-1] / 2), np.abs(Y - xs[-1] / 2)) / (xs[-1] / 2), 0, 1) + 0.3 * np.sin(X) * np.cos(Y)


@pytest.mark.parametrize("fw,fh,fillet,rock", [(10, 25, 5, False), (0, 0, None, False), (12, 40, None, True), (8, 20, 4, True)])
def test_frame_solid_watertight(fw, fh, fillet, rock):
    import trimesh
    G = 81; cell = 1.5; Z = _pyramid(G, cell)
    xs = np.arange(G) * cell
    V, F, info = frame_solid(xs, xs, Z, fw=fw, fh=fh, fillet=fillet, wall_rock=rock, wall_rows=40)
    m = trimesh.Trimesh(V, F, process=False)
    if m.volume < 0:
        m.invert()
    assert m.is_watertight and m.is_winding_consistent
    assert abs(m.extents[0] - ((G - 1) * cell + 2 * fw)) < 1e-6 and m.bounds[0][2] == 0.0


def test_slope_band_no_overhang_and_bounds():
    G = 121; cell = 1.0; Z = _pyramid(G, cell, 80)
    Zs = apply_slope_band(Z, cell, 12.0, 3.0)
    assert Zs.shape == Z.shape and Zs.min() >= 3.0 - 1e-6
    # усередині ядра рельєф не змінено (крім 4-мм плеча)
    assert np.allclose(Zs[30:-30, 30:-30], Z[30:-30, 30:-30])


def test_tile_cuts():
    assert tile_cuts(200, 256) == []
    c = tile_cuts(400, 256); assert len(c) == 2 and c[1] - c[0] <= 256 * 0.95 + 0.2
    assert len(tile_cuts(600, 256)) >= 2


def test_search_places_presets_first_and_dedup(monkeypatch):
    """«Знайти гору»: пресет першим, та сама вершина з OSM не дублюється, мережа не падає пошук."""
    from services.mountains import agent as A

    class R:
        status_code = 200
        def json(self):
            return [{"lat": "48.1601", "lon": "24.5004", "name": "Говерла", "category": "natural", "type": "peak", "display_name": "Говерла, Україна"},
                    {"lat": "48.3", "lon": "24.1", "name": "Ворохта", "category": "place", "type": "village", "display_name": "Ворохта, Надвірнянський район, Україна"}]
    monkeypatch.setattr(A.requests, "get", lambda *a, **k: R())
    monkeypatch.setattr(A, "_nominatim_last", [0.0])
    A._search_cache.clear()
    res = A.search_places("говерла", "uk")
    assert res[0]["source"] == "preset" and res[0]["preset_id"] == "hoverla"
    assert [r["name"] for r in res].count("Говерла") == 1
    assert any(r["name"] == "Ворохта" and r["area_km"] == 8.0 for r in res)
    assert A.search_places("г", "uk") == []
