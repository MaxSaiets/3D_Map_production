"""Класична детекція стін без нейромережі.

Ключова ідея: на технічному плані СТІНА — це найтовстіший штрих. Текст,
розмірні лінії, штриховка, меблі, дуги дверей малюються тонко (1-3 px), стіни —
товсто (від 5 px і більше). Тому фільтруємо не за формою, а за ТОВЩИНОЮ:
distance transform → ядра стін → геодезична реконструкція назад до повної
ширини. Це на порядок стійкіше за Hough чи пошук контурів.

Модуль самодостатній: він і резервний шлях, коли ONNX-ваг немає, і «друга
думка» для перевірки нейромережі.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .preprocess import binarize


@dataclass
class CvDetectConfig:
    core_ratio: float = 0.34        # частка максимального радіуса → поріг ядра стіни
    min_core_radius_px: float = 2.2
    min_component_area_px: int = 90
    max_wall_fraction: float = 0.45  # більше — значить залили пів-аркуша
    min_wall_fraction: float = 0.006


@dataclass
class CvDetectResult:
    wall_mask: np.ndarray
    ink_mask: np.ndarray
    strategy: str                    # thick-stroke | closed-hollow
    confidence: float
    notes: List[str]


def _largest_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    import cv2

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num <= 1:
        return mask.astype(np.uint8)
    keep = np.zeros(num, dtype=bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[i] = True
    if not keep.any():
        keep[int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1] = True
    return keep[labels].astype(np.uint8)


def _geodesic_dilate(seed: np.ndarray, ink: np.ndarray, iterations: int = 60) -> np.ndarray:
    """Нарощує ядра стін всередині чорнила — так стіна відновлюється до повної
    ширини, але не «перетікає» на текст, бо той не має зв'язку з ядром."""
    import cv2

    kernel = np.ones((3, 3), np.uint8)
    cur = (seed > 0).astype(np.uint8)
    ink = (ink > 0).astype(np.uint8)
    for _ in range(iterations):
        grown = cv2.dilate(cur, kernel, iterations=1) & ink
        if np.array_equal(grown, cur):
            break
        cur = grown
    return cur


def _thick_stroke_walls(ink: np.ndarray, cfg: CvDetectConfig
                        ) -> Tuple[np.ndarray, float, List[str]]:
    import cv2

    notes: List[str] = []
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    values = dist[dist > 0]
    if values.size < 50:
        return np.zeros_like(ink), 0.0, ["Замало чорнила на зображенні."]

    top = float(np.percentile(values, 99.5))
    # Поріг НЕ може бути відсотком від найтовщої стіни: у житловому плані несуча
    # 0.4 м і перегородка 0.08 м відрізняються в 5 разів, і 34% від несучої —
    # це вже товще за перегородку, тобто ми стирали половину квартири. Прив'язка
    # до роздільної здатності відділяє стіну від тексту й розмірних ліній,
    # незалежно від того, наскільки товсті стіни на конкретному кресленні.
    from .vectorize import expected_min_wall_px

    res_floor = expected_min_wall_px(ink.shape) / 2.0 * 0.8
    threshold = max(cfg.min_core_radius_px, min(res_floor, top * cfg.core_ratio))
    core = (dist >= threshold).astype(np.uint8)
    if core.sum() < 30:
        threshold = max(1.6, threshold * 0.6)
        core = (dist >= threshold).astype(np.uint8)
    core = _largest_components(core, max(12, cfg.min_component_area_px // 6))
    if core.sum() < 20:
        return np.zeros_like(ink), 0.0, ["Товстих штрихів (стін) не знайдено."]

    walls = _geodesic_dilate(core, ink, iterations=int(top * 2.5) + 12)
    walls = _largest_components(walls, cfg.min_component_area_px)
    notes.append(f"Поріг товщини стіни ≈ {threshold * 2:.1f} px.")
    # впевненість: наскільки чітко стіни відділились від решти чорнила
    ratio = float(walls.sum()) / max(1.0, float(ink.sum()))
    confidence = float(np.clip(1.2 - abs(ratio - 0.45) * 1.6, 0.15, 0.9))
    return walls, confidence, notes


def _elongation_filter(mask: np.ndarray, min_elongation: float = 2.5) -> np.ndarray:
    """Лишає лише ВИДОВЖЕНІ компоненти — стіни, а не меблі.

    Після морфологічного замикання суцільними стають і стіни, і намальовані
    тонкими лініями меблі. Розрізняє їх видовженість: мережа стін має площу в
    багато разів більшу за квадрат власної півширини, а тумбочка — приблизно
    рівну їй."""
    import cv2

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num <= 1:
        return mask.astype(np.uint8)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    max_dist = np.zeros(num, dtype=np.float32)
    np.maximum.at(max_dist, labels.ravel(), dist.ravel())
    keep = np.zeros(num, dtype=bool)
    for i in range(1, num):
        half = max(1.0, float(max_dist[i]))
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area / (4.0 * half * half) >= min_elongation:
            keep[i] = True
    if not keep.any():
        keep[int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1] = True
    return keep[labels].astype(np.uint8)


def keep_main_structure(mask: np.ndarray) -> np.ndarray:
    """Лишає лише каркас квартири: головний замкнений контур і все ВСЕРЕДИНІ нього.

    Розмірні ланцюжки, штамп, підписи й рамка аркуша лежать ЗЗОВНІ контуру
    квартири і жодного разу його не торкаються — тому їх можна відрізати
    геометрично, без жодних порогів товщини. Саме вони давали 150 «стін»
    замість 12 і роздували габарит моделі."""
    import cv2
    from scipy.ndimage import binary_fill_holes

    m = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 2:
        return m
    filled_area = np.zeros(num, dtype=np.int64)
    for i in range(1, num):
        comp = (labels == i)
        try:
            filled_area[i] = int(binary_fill_holes(comp).sum())
        except Exception:
            filled_area[i] = int(comp.sum())
    main = int(np.argmax(filled_area[1:])) + 1
    main_filled = binary_fill_holes(labels == main)

    keep = np.zeros(num, dtype=bool)
    keep[main] = True
    for i in range(1, num):
        if i == main:
            continue
        comp = labels == i
        inside = float((comp & main_filled).sum()) / max(1.0, float(comp.sum()))
        if inside > 0.9:                 # перегородка всередині квартири
            keep[i] = True
    return keep[labels].astype(np.uint8)


def _enclosed_area(mask: np.ndarray) -> int:
    """Площа приміщень, замкнених цією маскою. Головна метрика якості детекції:
    правильна маска стін оточує кімнати, неправильна — ні."""
    from scipy.ndimage import binary_fill_holes

    if mask.sum() == 0:
        return 0
    try:
        return int(binary_fill_holes(mask.astype(bool)).sum())
    except Exception:
        return int(mask.sum())


def _closed_hollow_walls(ink: np.ndarray, cfg: CvDetectConfig
                         ) -> Tuple[np.ndarray, float, List[str]]:
    """План намальовано ПОРОЖНІМИ або ШТРИХОВАНИМИ стінами.

    Це не екзотика, а норма для БТІ й забудовницьких креслень: несучу стіну
    малюють двома лініями зі штриховкою між ними, і фільтр «товстого штриха»
    бачить там саму лише порожнечу. Морфологічне замикання зліплює контур і
    штриховку в суцільну смугу; далі відкидаємо все невидовжене (меблі)."""
    import cv2

    h, w = ink.shape[:2]
    best: Optional[Tuple[np.ndarray, int, int]] = None
    for k in (5, 7, 9, 11, 13, 17, 21):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
        candidate = _elongation_filter(_largest_components(closed, cfg.min_component_area_px))
        frac = float(candidate.sum()) / float(h * w)
        if not (cfg.min_wall_fraction <= frac <= cfg.max_wall_fraction):
            continue
        enclosed = _enclosed_area(candidate)
        # Беремо НАЙМЕНШЕ ядро, яке вже замикає кімнати: більші лише роздувають
        # стіни й зліплюють меблі зі стінами.
        if best is None or enclosed > best[1] * 1.05:
            best = (candidate, enclosed, k)
    if best is None:
        return np.zeros_like(ink), 0.0, []
    candidate, _enclosed, k = best
    return candidate, 0.45, [f"Стіни зібрано замиканням {k}×{k} px (штрихований/порожній стиль)."]


def detect_walls(rgb: np.ndarray, cfg: Optional[CvDetectConfig] = None) -> CvDetectResult:
    """RGB план → маска стін класичним CV."""
    cfg = cfg or CvDetectConfig()
    ink = binarize(rgb)
    h, w = ink.shape[:2]
    area = float(h * w)

    # Рахуємо ОБИДВІ стратегії й обираємо за площею замкнених кімнат: план, у
    # якого стіни справді оточують приміщення, майже завжди правильний. Раніше
    # ми вмикали запасну стратегію лише коли основна давала абсурдну частку
    # заливки — і мовчки приймали «товстий штрих», який на штрихованому плані
    # знаходив саму лише внутрішню перегородку (габарит виходив на 40% менший).
    thick, thick_conf, thick_notes = _thick_stroke_walls(ink, cfg)
    hollow, hollow_conf, hollow_notes = _closed_hollow_walls(ink, cfg)

    def _valid(mask: np.ndarray) -> bool:
        frac = float(mask.sum()) / area
        return mask.sum() > 0 and cfg.min_wall_fraction <= frac <= cfg.max_wall_fraction

    def _score(mask: np.ndarray) -> float:
        """Скільки площі КІМНАТ припадає на одиницю стіни.

        Порівнювати самі лише замкнені площі не можна: агресивне замикання
        «замикає» ще й розмірні ланцюжки та штамп і завжди перемагає. Відношення
        кімнати/стіни карає таке роздування — правильна маска стін тонка й
        оточує багато простору."""
        wall_px = float(mask.sum())
        if wall_px <= 0:
            return 0.0
        rooms = float(_enclosed_area(mask)) - wall_px
        return max(0.0, rooms) / wall_px

    thick = keep_main_structure(thick) if thick.sum() else thick
    hollow = keep_main_structure(hollow) if hollow.sum() else hollow

    options = []
    if _valid(thick):
        options.append(("thick-stroke", thick, thick_conf, thick_notes, _score(thick)))
    if _valid(hollow):
        options.append(("closed-hollow", hollow, hollow_conf, hollow_notes, _score(hollow)))

    if not options:
        frac = float(thick.sum()) / area
        return CvDetectResult(
            wall_mask=thick.astype(np.uint8), ink_mask=ink, strategy="thick-stroke",
            confidence=0.15,
            notes=thick_notes + [
                f"Стіни зайняли {frac * 100:.1f}% аркуша — це підозріло; перевірте вручну."
            ],
        )

    strategy, walls, confidence, notes, score = max(options, key=lambda o: o[4])
    if len(options) == 2:
        other = min(options, key=lambda o: o[4])
        if score > other[4] * 1.35:
            notes = list(notes) + [
                f"Обрано «{strategy}»: у {score / max(0.01, other[4]):.1f}× більше "
                f"площі кімнат на одиницю стіни."
            ]
        else:
            confidence = min(confidence, 0.4)     # стратегії не узгоджуються
    if strategy == "closed-hollow":
        notes = list(notes) + ["Стіни намальовані тонко/штриховкою — перевірте їх у редакторі."]

    return CvDetectResult(wall_mask=walls.astype(np.uint8), ink_mask=ink,
                          strategy=strategy, confidence=confidence, notes=notes)
