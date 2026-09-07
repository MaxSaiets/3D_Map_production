"""Режим «опиши світ»: парсер промту + процедурний генератор.

Гарантії, які ці тести тримають (усе — реальні дефекти, знайдені 08.09.2026):
  * кожна форма будує ГЕРМЕТИЧНИЙ solid із КОНСИСТЕНТНОЮ намоткою і додатним
    об'ємом (раніше X-стіни були вивернуті, а mesh.volume цього не показував);
  * генерація швидка (без trimesh.fix_normals, який їв 13 с із 13.5 с);
  * різні описи дають РІЗНІ форми (раніше все падало в «купу в центрі»);
  * spec завжди у друкованих межах, seed відтворюваний, variant дає інший світ.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from services.llm_orchestrator import prompt_to_spec, SHAPES as PARSER_SHAPES
from services.procedural_generator import (
    SHAPES,
    generate_world_mesh,
    normalize_spec,
    _falloff,
)


def _spec(shape: str, **kw):
    base = {"shape": shape, "width_mm": 120.0, "max_height_mm": 22.0,
            "roughness": 0.55, "erosion": 0.6, "seed": 7}
    base.update(kw)
    return base


@pytest.mark.parametrize("shape", SHAPES)
def test_every_shape_is_printable_solid(shape):
    m = generate_world_mesh(_spec(shape))
    assert m.is_watertight, f"{shape}: не герметичний"
    assert m.is_winding_consistent, f"{shape}: неконсистентна намотка граней"
    assert m.volume > 0, f"{shape}: від'ємний об'єм (вивернуті нормалі)"
    assert len(m.faces) > 1000


@pytest.mark.parametrize("shape", SHAPES)
def test_dimensions_match_spec(shape):
    m = generate_world_mesh(_spec(shape, width_mm=100.0, max_height_mm=20.0, base_thickness_mm=3.0))
    lo, hi = m.bounds
    assert abs((hi[0] - lo[0]) - 100.0) < 0.01
    assert abs((hi[1] - lo[1]) - 100.0) < 0.01
    assert lo[2] == pytest.approx(0.0, abs=1e-6), "дно моделі має лежати на z=0"
    # верх не перевищує база+висота (+запас на згладження)
    assert hi[2] <= 3.0 + 20.0 + 0.5


def test_generation_is_fast():
    """Регрес-щит на trimesh.fix_normals (13 с) і Python-цикли по гранях (7-13 с)."""
    t = time.time()
    generate_world_mesh(_spec("mountain"))
    assert time.time() - t < 3.0


def test_seed_is_reproducible_and_variant_changes_world():
    a = generate_world_mesh(_spec("mountain", seed=42))
    b = generate_world_mesh(_spec("mountain", seed=42))
    c = generate_world_mesh(_spec("mountain", seed=43))
    assert np.allclose(a.vertices, b.vertices), "один seed має давати той самий світ"
    assert not np.allclose(a.vertices, c.vertices), "інший seed має давати інший світ"


def test_shapes_differ_from_each_other():
    """Головна претензія власника: «усе однакове». Порівнюємо профілі висот."""
    tops = {}
    for sh in SHAPES:
        m = generate_world_mesh(_spec(sh))
        z = m.vertices[:, 2]
        tops[sh] = (float(z.mean()), float(z.std()))
    pairs = [(a, b) for i, a in enumerate(SHAPES) for b in SHAPES[i + 1:]]
    same = [(a, b) for a, b in pairs
            if abs(tops[a][0] - tops[b][0]) < 0.05 and abs(tops[a][1] - tops[b][1]) < 0.05]
    assert not same, f"форми не відрізняються: {same}"


def test_slope_clamp_keeps_model_printable():
    """Без обмеження нахилу виходили звисання >70° (друкується як «спагеті»)."""
    spec = _spec("mountain", width_mm=120.0, max_height_mm=38.0, roughness=1.0, erosion=0.0)
    m = generate_world_mesh(spec)
    top = m.vertices[m.vertices[:, 2] > 0.01]
    assert len(top) > 0
    # Максимальний перепад між сусідніми клітинками сітки не перевищує ~2.6 клітинки.
    n = int(np.sqrt(len(m.vertices) / 2))
    cell = 120.0 / (n - 1)
    grid = m.vertices[: n * n, 2].reshape(n, n)
    dz = max(np.abs(np.diff(grid, axis=0)).max(), np.abs(np.diff(grid, axis=1)).max())
    assert dz <= cell * 2.6 + 0.35, f"нахил {dz:.2f} мм/клітинку перевищує друкований"


def test_falloff_has_no_diagonal_seam():
    """max(|x|,|y|) давав рівний діагональний шов через усю модель (видно на
    hillshade і в друці). Сепарабельна маска (добуток двох 1D) такого зламу мати
    НЕ може за побудовою — це властивість і перевіряємо, плюс гладкість профілю."""
    f = _falloff(64)
    assert f[0, 0] == pytest.approx(0.0) and f[32, 32] == pytest.approx(1.0)
    for i, j, k, l in [(5, 9, 40, 33), (2, 60, 31, 17), (12, 12, 50, 3)]:
        assert f[i, j] * f[k, l] == pytest.approx(f[i, l] * f[k, j], abs=1e-9), "маска не сепарабельна → можливий діагональний злам"
    # профіль — монотонний підйом до плато, без «зубців» (треті різниці малі
    # відносно перших: у зламу вони стрибають).
    prof = f[:32, 32]
    assert np.all(np.diff(prof) >= -1e-12), "маска має монотонно зростати до центру"
    assert np.abs(np.diff(prof, 3)).max() < np.abs(np.diff(prof)).max() * 0.5


def test_normalize_spec_clamps_to_printable_range():
    s = normalize_spec({"shape": "неіснуюча", "width_mm": 9999, "max_height_mm": -5,
                        "base_thickness_mm": 99, "roughness": 7, "erosion": -2})
    assert s["shape"] == "mountain"
    assert s["width_mm"] == 220.0
    assert 2.0 <= s["max_height_mm"] <= 40.0
    assert 1.0 <= s["base_thickness_mm"] <= 8.0
    assert 0.0 <= s["roughness"] <= 1.0
    assert 0.0 <= s["erosion"] <= 1.0


@pytest.mark.parametrize("prompt,expected", [
    ("Епічні засніжені гори", "mountain"),
    ("Острів-вулкан у морі", "volcano"),
    ("Глибокий каньйон з рікою", "valley"),
    ("Плавні зелені пагорби", "rolling"),
    ("Інопланетний кратер", "crater"),
    ("Піщані дюни пустелі", "ridges"),
    ("архіпелаг тропічних островів", "archipelago"),
    ("столова гора меза", "plateau"),
    ("тропічний острів з пляжем", "island"),
])
def test_prompt_maps_to_expected_shape(prompt, expected):
    """Усі 6 прикладів зі сторінки + нові форми мають розпізнаватись."""
    spec, src = prompt_to_spec(prompt, 120.0)
    assert spec["shape"] == expected, f"{prompt!r} → {spec['shape']}, очікували {expected}"
    assert src == "rules"


def test_prompt_modifiers_change_height_and_roughness():
    tall, _ = prompt_to_spec("високі гострі гори", 120.0)
    low, _ = prompt_to_spec("низькі пологі гори", 120.0)
    assert tall["max_height_mm"] > low["max_height_mm"] * 1.5
    rough, _ = prompt_to_spec("скелясті хаотичні гори", 120.0)
    smooth, _ = prompt_to_spec("гладкі округлі гори", 120.0)
    assert rough["roughness"] > smooth["roughness"]


def test_shape_override_wins_over_parser():
    spec, src = prompt_to_spec("Епічні засніжені гори", 120.0, shape_override="crater")
    assert spec["shape"] == "crater" and src == "user"
    # невідома форма ігнорується — вертаємось до парсера
    spec2, src2 = prompt_to_spec("Епічні засніжені гори", 120.0, shape_override="banana")
    assert spec2["shape"] == "mountain" and src2 == "rules"


def test_variant_changes_seed_but_not_shape():
    a, _ = prompt_to_spec("гори", 120.0, seed_extra=0)
    b, _ = prompt_to_spec("гори", 120.0, seed_extra=7919)
    assert a["shape"] == b["shape"]
    assert a["seed"] != b["seed"]


def test_parser_shapes_match_generator_shapes():
    assert set(PARSER_SHAPES) == set(SHAPES)
