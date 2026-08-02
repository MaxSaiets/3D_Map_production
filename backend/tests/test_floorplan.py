"""Тести сервісу «макет квартири».

Фікстури синтетичні й детерміновані (seed), тож тести не залежать від жодного
зовнішнього датасету й від наявності ваг нейромережі. Ключові інваріанти, які
вони стережуть, — саме ті, що вже ламались:
  • габарит виробу дорівнює замовленому (подвійне масштабування = брак),
  • меш герметичний і без відірваних шматків,
  • стіни не тонші за сопло після масштабування,
  • AGPL-залежність (PyMuPDF) не повертається в дерево.
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.floorplan import synth                                    # noqa: E402
from ml.floorplan.synth import RenderStyle                        # noqa: E402
from services.floorplan.builder import (BuildOptions,             # noqa: E402
                                        build_plan_mesh)
from services.floorplan.plan_model import PlanVector, Wall        # noqa: E402
from services.floorplan.vectorize import masks_to_plan            # noqa: E402

FIXTURE_SEED = 1002
FIXTURE_PX_PER_M = 70.0


def _fixture(seed: int = FIXTURE_SEED, px_per_m: float = FIXTURE_PX_PER_M):
    rng = np.random.default_rng(seed)
    spec = synth.generate_layout(rng)
    img, mask = synth.render_layout(spec, rng, RenderStyle(px_per_m=px_per_m, margin_px=110))
    return spec, img, mask, synth.layout_to_plan(spec)


def _png_bytes(img: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.dstack([img] * 3) if img.ndim == 2 else img).save(buffer, format="PNG")
    return buffer.getvalue()


# ── Геометрія ────────────────────────────────────────────────────────────────
def test_build_is_watertight_single_solid():
    _spec, _img, _mask, gt = _fixture()
    result = build_plan_mesh(gt, BuildOptions(model_size_mm=150.0))
    mesh = result.mesh
    assert mesh.is_watertight, "меш не герметичний — слайсер загубить стіни"
    assert mesh.is_volume
    assert float(mesh.volume) > 0
    bodies = mesh.split(only_watertight=False)
    assert len(bodies) == 1, f"модель розпалась на {len(bodies)} частин — вони відваляться"


def test_exported_size_matches_request():
    """Габарит МУСИТЬ дорівнювати замовленому: розмір — це і є товар."""
    _spec, _img, _mask, gt = _fixture()
    for requested in (100.0, 150.0, 250.0):
        result = build_plan_mesh(gt, BuildOptions(model_size_mm=requested))
        longest = max(result.mesh.extents[0], result.mesh.extents[1])
        assert abs(longest - requested) < 0.1, (
            f"замовили {requested} мм, отримали {longest:.2f} мм"
        )


def test_exported_3mf_keeps_size(tmp_path):
    """Файл на диску теж має бути потрібного розміру (без пере-масштабування)."""
    import trimesh

    _spec, _img, _mask, gt = _fixture()
    result = build_plan_mesh(gt, BuildOptions(model_size_mm=150.0))
    path = tmp_path / "plan.3mf"
    result.mesh.export(str(path))
    reloaded = trimesh.load(str(path), force="mesh")
    longest = max(reloaded.extents[0], reloaded.extents[1])
    assert abs(longest - 150.0) < 0.1, f"3MF на диску має {longest:.2f} мм замість 150"


def test_min_printable_wall_enforced():
    """Стіни тонші за сопло слайсер мовчки викидає — не даємо їм зʼявитись."""
    _spec, _img, _mask, gt = _fixture()
    options = BuildOptions(model_size_mm=60.0, min_wall_mm=1.2)   # навмисно дрібно
    result = build_plan_mesh(gt, options)
    assert result.stats["min_wall_mm"] == pytest.approx(1.2)
    assert any("потовщено" in w for w in result.warnings), (
        "потоншення стін мусить бути повідомлене користувачу"
    )


def test_maquette_height_is_capped():
    """Висота стін — головний важіль часу друку, тому дефолт низький."""
    _spec, _img, _mask, gt = _fixture()
    maquette = build_plan_mesh(gt, BuildOptions(model_size_mm=150.0))
    true_scale = build_plan_mesh(
        gt, BuildOptions(model_size_mm=150.0, wall_height_mode="true_scale"))
    assert maquette.stats["wall_height_mm"] < true_scale.stats["wall_height_mm"]
    assert maquette.stats["wall_height_mm"] == pytest.approx(22.0, abs=0.01)


def test_openings_are_cut():
    """Вирізані отвори мусять зменшувати обʼєм — інакше вони «намальовані»."""
    _spec, _img, _mask, gt = _fixture()
    with_holes = build_plan_mesh(gt, BuildOptions(model_size_mm=150.0))
    solid = build_plan_mesh(
        gt, BuildOptions(model_size_mm=150.0, cut_doors=False, cut_windows=False))
    assert float(with_holes.mesh.volume) < float(solid.mesh.volume) * 0.995


# ── Векторизація ─────────────────────────────────────────────────────────────
def test_vectorize_recovers_plan_dimensions():
    """Головний регресійний тест на РОЗМІР: з ідеальної маски план має
    відновлюватись у межах 1%. Саме тут ловиться забруднення розмірними лініями."""
    _spec, _img, mask, gt = _fixture()
    plan_px = masks_to_plan((mask == 1), (mask == 2), (mask == 3))
    assert plan_px.walls, "жодної стіни не відновлено"
    plan_m = PlanVector.from_pixel_dict(plan_px.to_dict(), 1.0 / FIXTURE_PX_PER_M)
    gt_w, gt_h = gt.size_m()
    rec_w, rec_h = plan_m.size_m()
    assert abs(rec_w - gt_w) / gt_w < 0.01, f"ширина {rec_w:.2f} проти {gt_w:.2f} м"
    assert abs(rec_h - gt_h) / gt_h < 0.01, f"висота {rec_h:.2f} проти {gt_h:.2f} м"


def test_vectorize_recovers_openings():
    _spec, _img, mask, gt = _fixture()
    plan_px = masks_to_plan((mask == 1), (mask == 2), (mask == 3))
    gt_doors = sum(1 for o in gt.openings if o.kind == "door")
    gt_windows = sum(1 for o in gt.openings if o.kind == "window")
    doors = sum(1 for o in plan_px.openings if o.kind == "door")
    windows = sum(1 for o in plan_px.openings if o.kind == "window")
    assert abs(doors - gt_doors) <= 1, f"дверей {doors}, мало бути {gt_doors}"
    assert abs(windows - gt_windows) <= 1, f"вікон {windows}, мало бути {gt_windows}"


def test_thin_lines_are_not_walls():
    """Розмірний ланцюжок не повинен ставати стіною (це давало +7% до габариту)."""
    from services.floorplan.vectorize import (VectorizeConfig,
                                              thin_component_cut_px)

    cfg = VectorizeConfig()
    shape = (900, 900)
    cut = thin_component_cut_px(shape, cfg)
    hairline_half_thickness = 1.0        # розмірна лінія в 1-2 px
    wall_half_thickness = 0.08 * 70.0 / 2.0   # перегородка 8 см при 70 px/м
    assert hairline_half_thickness < cut < wall_half_thickness


# ── Масштаб ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,expected_m", [
    ("2 800", 2.8), ("2800", 2.8), ("2,80", 2.8), ("2.80", 2.8),
    ("10430", 10.43), ("850", 0.85),
    ("12", None), ("абв", None), ("", None), ("999999", None),
])
def test_dimension_parsing(text, expected_m):
    from services.floorplan.scale import _parse_dimension

    got = _parse_dimension(text)
    if expected_m is None:
        assert got is None, f"'{text}' не мало розпарситись, а дало {got}"
    else:
        assert got == pytest.approx(expected_m, abs=0.005), f"'{text}' → {got}"


def test_scale_voting_matches_truth():
    """Голосування за масштабом по розмірних числах має влучати в межах 2%."""
    from services.floorplan.scale import from_ocr

    spec, _img, mask, _gt = _fixture()
    plan_px = masks_to_plan((mask == 1), (mask == 2), (mask == 3))
    xs = sorted({0.0, spec.width_m} | {round(s.const, 3) for s in spec.segs if not s.horizontal})
    ys = sorted({0.0, spec.height_m} | {round(s.const, 3) for s in spec.segs if s.horizontal})
    values = ([round(xs[i + 1] - xs[i], 3) for i in range(len(xs) - 1)]
              + [round(ys[i + 1] - ys[i], 3) for i in range(len(ys) - 1)])
    items = [{"text": str(v), "value_m": v, "score": 1.0} for v in values if v > 0.25]
    candidate = from_ocr(items, plan_px)
    assert candidate is not None, "масштаб не визначився з розмірних чисел"
    truth = 1.0 / FIXTURE_PX_PER_M
    assert abs(candidate.m_per_px - truth) / truth < 0.02


def test_area_scale_matches_truth():
    """Масштаб із загальної площі (та, що в договорі) — другий за точністю
    спосіб після лінійки. Заміряно 2.2% середньої похибки на синтетиці."""
    from services.floorplan.scale import from_area, interior_area_px2

    _spec, _img, mask, gt = _fixture()
    plan_px = masks_to_plan((mask == 1), (mask == 2), (mask == 3))
    gt_px = PlanVector.from_pixel_dict(gt.to_dict(), FIXTURE_PX_PER_M)
    true_area_m2 = interior_area_px2(gt_px) / (FIXTURE_PX_PER_M ** 2)
    assert true_area_m2 > 10, "еталонна площа не порахувалась"

    candidate = from_area(plan_px, true_area_m2)
    assert candidate is not None
    truth = 1.0 / FIXTURE_PX_PER_M
    assert abs(candidate.m_per_px - truth) / truth < 0.05


def test_area_survives_broken_outer_wall():
    """Розрив у ЗОВНІШНІЙ стіні не має вбивати масштаб «за площею».

    Балконні двері, вхід чи панорамне вікно розривають зовнішній контур, і тоді
    кільцевий метод не знаходить ЖОДНОГО замкненого кільця — площа приміщень
    виходить нульова, кнопка «Застосувати» мовчки нічого не робить, а в рядку
    підтвердження показується площа ГАБАРИТУ. На реальних планах Floor Plan CIS
    так було у 36% випадків. Запасний шлях — оболонка мережі."""
    from services.floorplan.scale import from_area, interior_area_px2

    # Прямокутна квартира 600×400 px; у верхній стіні проріз 160 px.
    t = 12.0
    walls = [
        Wall(x1=0.0, y1=0.0, x2=220.0, y2=0.0, thickness_m=t),
        Wall(x1=380.0, y1=0.0, x2=600.0, y2=0.0, thickness_m=t),   # ← розрив 160 px
        Wall(x1=600.0, y1=0.0, x2=600.0, y2=400.0, thickness_m=t),
        Wall(x1=600.0, y1=400.0, x2=0.0, y2=400.0, thickness_m=t),
        Wall(x1=0.0, y1=400.0, x2=0.0, y2=0.0, thickness_m=t),
    ]
    plan = PlanVector(walls=walls, image_size_px=(640, 440))

    area = interior_area_px2(plan)
    bbox = 600.0 * 400.0
    assert area > 0.55 * bbox, (
        f"площа приміщень {area:.0f} px² з габариту {bbox:.0f} px² — "
        "розрив у зовнішній стіні знову вбив оцінку"
    )
    # 60 м² на такій квартирі → масштаб має бути ≈ √(60 / площа_px)
    candidate = from_area(plan, 60.0)
    assert candidate is not None
    span_m = 600.0 * candidate.m_per_px
    assert 8.0 < span_m < 16.0, f"квартира вийшла {span_m:.1f} м завширшки"


def test_base_follows_apartment_outline_despite_gaps():
    """Проріз у зовнішній стіні не має перетворювати основу на прямокутну плиту.

    Г-подібна квартира з розривами на вході й на балконі: основа мусить лишитись
    Г-подібною (помітно менша за габаритний прямокутник), а не стати підносом.
    На реальних планах Floor Plan CIS плита спрацьовувала у 40 випадках із 87 —
    саме через ці розриви."""
    t = 0.30                                    # МЕТРИ: PlanVector завжди в метрах
    # Г-подібна квартира 12×10 м із вирізаним кутом 5.2×4.4 м і двома прорізами.
    walls = [
        Wall(x1=0.0, y1=0.0, x2=5.0, y2=0.0, thickness_m=t),
        Wall(x1=7.0, y1=0.0, x2=12.0, y2=0.0, thickness_m=t),        # вхід 2.0 м
        Wall(x1=12.0, y1=0.0, x2=12.0, y2=10.0, thickness_m=t),
        Wall(x1=12.0, y1=10.0, x2=6.8, y2=10.0, thickness_m=t),
        Wall(x1=6.8, y1=10.0, x2=6.8, y2=5.6, thickness_m=t),        # виїмка
        Wall(x1=6.8, y1=5.6, x2=0.0, y2=5.6, thickness_m=t),
        Wall(x1=0.0, y1=5.6, x2=0.0, y2=3.4, thickness_m=t),
        Wall(x1=0.0, y1=1.8, x2=0.0, y2=0.0, thickness_m=t),         # балкон 1.6 м
        Wall(x1=5.0, y1=2.8, x2=12.0, y2=2.8, thickness_m=t),        # перегородка
    ]
    plan = PlanVector(walls=walls, image_size_px=(1200, 1000))
    result = build_plan_mesh(plan, BuildOptions(model_size_mm=150.0))

    assert result.mesh.is_watertight, "меш негерметичний"
    assert result.mesh.body_count == 1, f"деталей {result.mesh.body_count}, має бути одна"
    assert not any("не зчеплен" in w for w in result.warnings), (
        f"впала у прямокутну плиту: {result.warnings}"
    )

    base = result.parts.get("base")
    assert base is not None, "основи немає"
    lo, hi = base.bounds
    bbox = (hi[0] - lo[0]) * (hi[1] - lo[1])
    # Площа підошви = площа нижньої грані.
    verts = base.vertices
    z_min = verts[:, 2].min()
    bottom = base.faces[
        np.all(np.isclose(verts[base.faces][:, :, 2], z_min, atol=1e-6), axis=1)
    ]
    area = 0.0
    for face in bottom:
        a, b, c = verts[face][:, :2]
        area += abs(np.cross(b - a, c - a)) / 2.0
    ratio = area / bbox
    # Г-подібна квартира займає ~0.8 габариту (виріз 260×220 з 600×500).
    assert 0.55 < ratio < 0.95, f"підошва {ratio:.2f} габариту — форму квартири втрачено"


def test_area_labels_are_not_read_as_lengths():
    """«12,1» на плані — це ПЛОЩА кімнати в м², а не 12.1 метра.

    Читання площ як довжин роздувало масштаб: квартира виходила 25×25 м, і
    фізичний запобіжник її відхиляв. На 99 реальних планах наскрізний успіх
    падав із 87 до 60."""
    from services.floorplan.scale import _parse_dimension

    for text in ("3500", "3 500", "4 680", "2 750", "1 720"):
        assert _parse_dimension(text) is not None, f"розмір «{text}» не прочитано"
    assert _parse_dimension("3,50") == pytest.approx(3.5)      # метри, два знаки
    for text in ("12,1", "16,4", "5,6", "45,9"):
        assert _parse_dimension(text) is None, f"площу «{text}» прийнято за довжину"


def test_ocr_scale_uses_label_position():
    """Розмір зіставляється з тим, ПІД ЧИМ він підписаний.

    Без цього голосування збиралось навколо хибного кластера: стін мало,
    відстаней між ними багато, і числа охоче лягали на повні прольоти —
    квартира 7.5 м виходила 3.7 м, причому «7 з 11 розмірів збіглись»."""
    from services.floorplan.scale import from_ocr

    # Квартира 800×400 px: перегородка посередині ділить її на 2 кімнати по 4 м.
    t = 20.0
    walls = [
        Wall(x1=0.0, y1=0.0, x2=800.0, y2=0.0, thickness_m=t),
        Wall(x1=800.0, y1=0.0, x2=800.0, y2=400.0, thickness_m=t),
        Wall(x1=800.0, y1=400.0, x2=0.0, y2=400.0, thickness_m=t),
        Wall(x1=0.0, y1=400.0, x2=0.0, y2=0.0, thickness_m=t),
        Wall(x1=400.0, y1=0.0, x2=400.0, y2=400.0, thickness_m=t),
    ]
    plan = PlanVector(walls=walls, image_size_px=(800, 400))

    def label(text, value, cx, cy, horizontal=True):
        half_w, half_h = (30.0, 8.0) if horizontal else (8.0, 30.0)
        box = [[cx - half_w, cy - half_h], [cx + half_w, cy - half_h],
               [cx + half_w, cy + half_h], [cx - half_w, cy + half_h]]
        return {"text": text, "box": box, "score": 0.99, "value_m": value}

    # Обидві кімнати підписані «3 800» (у світлі: 400 px − 20 px = 380 px).
    items = [
        label("3 800", 3.8, 200.0, -20.0),
        label("3 800", 3.8, 600.0, -20.0),
        label("3 800", 3.8, -20.0, 200.0, horizontal=False),
        label("7 800", 7.8, 400.0, 430.0),          # габарит по низу
    ]
    candidate = from_ocr(items, plan)
    assert candidate is not None, "масштаб за розмірами не визначився"
    # істина: 3.8 м на 380 px = 0.01 м/px
    assert candidate.m_per_px == pytest.approx(0.01, rel=0.05), (
        f"масштаб {candidate.m_per_px:.5f} замість 0.01 — "
        "розмір зіставлено не з тим прольотом"
    )


def test_parallel_duplicates_are_merged():
    """Роздвоєний скелет широкої стіни не має давати ДВІ стіни в редакторі."""
    from services.floorplan.vectorize import _merge_parallel_duplicates

    duplicates = [
        ((10.0, 100.0), (400.0, 100.0), 20.0),
        ((15.0, 112.0), (395.0, 112.0), 18.0),      # той самий контур, зсув 12 px
        ((10.0, 400.0), (400.0, 400.0), 20.0),      # окрема стіна за 300 px
    ]
    merged = _merge_parallel_duplicates(duplicates, snap=7.0)
    assert len(merged) == 2, f"мало лишитись 2 стіни, лишилось {len(merged)}"


def test_extend_closes_room_corner():
    """Стіни, що не дотягуються в куті, мають замикати кімнату після продовження."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    from services.floorplan.vectorize import _extend_dangling_ends

    # прямокутник, у якого кожна стіна не доходить до сусідньої на 12 px
    gap = 12.0
    raw = [
        ((0.0, 0.0), (300.0 - gap, 0.0), 20.0),
        ((300.0, 0.0), (300.0, 200.0 - gap), 20.0),
        ((300.0, 200.0), (gap, 200.0), 20.0),
        ((0.0, 200.0), (0.0, gap), 20.0),
    ]
    fixed = _extend_dangling_ends(raw, touch_tol=7.0)
    union = unary_union([
        LineString([p1, p2]).buffer(t / 2, cap_style=3, join_style=2) for p1, p2, t in fixed
    ])
    polygons = [union] if union.geom_type == "Polygon" else list(union.geoms)
    holes = sum(len(p.interiors) for p in polygons)
    assert holes == 1, f"кімната не замкнулась: дірок {holes}"


def test_scale_sanity_rejects_absurd():
    from services.floorplan.pipeline import _scale_sanity

    _spec, _img, mask, _gt = _fixture()
    plan_px = masks_to_plan((mask == 1), (mask == 2), (mask == 3))
    assert not _scale_sanity(plan_px, 1.0 / FIXTURE_PX_PER_M)
    assert _scale_sanity(plan_px, 1.0), "масштаб 1 м/px мав бути відхилений"


# ── Стійкість / ліцензії ─────────────────────────────────────────────────────
def test_plan_aligns_with_preview_on_large_sheet():
    """План МУСИТЬ лягати на превʼю, яке бачить користувач.

    Сторож конкретного багу: маски рахуються у зменшеній робочій роздільності,
    і коефіцієнт переводу в превʼю рахувався від ширини ОРИГІНАЛУ — тобто те
    саме зменшення застосовувалось удвічі. Кожен аркуш понад 1400 px показував
    би стіни вдвічі меншими за креслення під ними, і редагувати їх було б
    неможливо. На малих аркушах баг не проявлявся взагалі."""
    import base64

    from PIL import Image

    from services.floorplan.pipeline import analyze

    rng = np.random.default_rng(FIXTURE_SEED)
    spec = synth.generate_layout(rng)
    # аркуш ~2400 px — саме на такому баг і жив
    img, _mask = synth.render_layout(spec, rng, RenderStyle(px_per_m=180.0, margin_px=260))
    assert max(img.shape[:2]) > 1600, "фікстура має бути великою, інакше тест нічого не перевіряє"

    result = analyze(_png_bytes(img), use_ocr=False)
    assert result.plan.walls

    preview = Image.open(
        io.BytesIO(base64.b64decode(result.preview_data_url.split(",")[1]))
    ).convert("L")
    ink = np.array(preview) < 128
    ys, xs = np.nonzero(ink)
    assert xs.size > 100, "у превʼю немає креслення"
    ink_box = (xs.min(), ys.min(), xs.max(), ys.max())

    plan_xs = [c for w in result.plan.walls for c in (w.x1, w.x2)]
    plan_ys = [c for w in result.plan.walls for c in (w.y1, w.y2)]
    plan_box = (min(plan_xs), min(plan_ys), max(plan_xs), max(plan_ys))

    ink_width = ink_box[2] - ink_box[0]
    ink_height = ink_box[3] - ink_box[1]
    # План займає лише частину креслення (поза ним ще розмірні лінії), але
    # мусить бути ТОГО САМОГО порядку. При подвійному масштабуванні відношення
    # падає приблизно вдвічі.
    ratio_w = (plan_box[2] - plan_box[0]) / ink_width
    ratio_h = (plan_box[3] - plan_box[1]) / ink_height
    assert 0.55 < ratio_w < 1.05, f"ширина плану/креслення = {ratio_w:.2f}"
    assert 0.55 < ratio_h < 1.05, f"висота плану/креслення = {ratio_h:.2f}"


def test_pipeline_works_without_neural_net():
    """Ваг може не бути на сервері — продукт мусить працювати без них."""
    from services.floorplan.pipeline import analyze

    _spec, img, _mask, _gt = _fixture()
    result = analyze(_png_bytes(img), use_ocr=False, use_nn=False)
    assert result.plan.walls
    assert result.detector == "cv"
    assert result.preview_data_url.startswith("data:image/jpeg;base64,")


def test_no_agpl_pdf_dependency():
    """PyMuPDF (fitz) — AGPL-3.0: у платному сервісі його бути не може."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    pattern = re.compile(r"^\s*(import\s+fitz|from\s+fitz\b|import\s+pymupdf)", re.I | re.M)
    offenders = []
    for path in root.rglob("*.py"):
        parts = set(path.parts)
        if parts & {"venv", "venv_train", "__pycache__", "node_modules"}:
            continue
        try:
            if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                offenders.append(str(path.relative_to(root)))
        except OSError:
            continue
    assert not offenders, f"AGPL-залежність PyMuPDF повернулась у: {offenders}"


def test_nn_mask_quality_if_weights_present():
    """Якість маски нейромережі проти еталона — сторож препроцесингу.

    Цей тест існує через конкретний баг: експорт в ONNX загортав модель у
    ImageNet-нормалізацію, якої НЕ БУЛО при тренуванні. Помилки не виникало,
    валідаційна IoU лишалась 0.92, а на реальному вході маска ставала «все
    чорнило креслення підряд» — precision падала з 0.93 до 0.53. Будь-яка
    розбіжність препроцесингу тренування й інференсу валить саме precision,
    тому вона тут і перевіряється. Без файлу ваг тест пропускається."""
    from services.floorplan import detect_nn

    if not detect_nn.is_available():
        pytest.skip("ваг ONNX немає — сервіс працює класичним CV")

    precisions, recalls = [], []
    for seed in (1000, 1002, 1005, 1007):
        _spec, img, mask, _gt = _fixture(seed)
        result = detect_nn.detect(np.dstack([img] * 3))
        assert result is not None
        predicted = (result.wall_mask | result.door_mask | result.window_mask) > 0
        truth = mask > 0
        intersection = float((predicted & truth).sum())
        precisions.append(intersection / max(1.0, float(predicted.sum())))
        recalls.append(intersection / max(1.0, float(truth.sum())))

    precision = float(np.mean(precisions))
    recall = float(np.mean(recalls))
    assert precision > 0.75, (
        f"precision={precision:.3f} — маска роздута. Найімовірніша причина: "
        f"препроцесинг інференсу не збігається з тренуванням."
    )
    assert recall > 0.75, f"recall={recall:.3f} — мережа губить стіни"


def test_build_rejects_absurd_area():
    from services.floorplan.pipeline import FloorplanError, build

    _spec, _img, mask, _gt = _fixture()
    plan_px = masks_to_plan((mask == 1), (mask == 2), (mask == 3))
    with pytest.raises(FloorplanError):
        build(plan_px.to_dict(), 1.0)      # 1 м на піксель — квартира на гектар
