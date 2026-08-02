"""Визначення масштабу: скільки метрів в одному пікселі плану.

Це найважливіше число в усьому сервісі. Модель може бути ідеально герметичною
і все одно нікому не потрібною, якщо квартира вийшла 1:80 замість 1:60.

Стратегії, від найнадійнішої до найслабшої:
  1. reference — користувач провів лінію по відомому розміру і вписав довжину.
     Так роблять ВСІ професійні інструменти обміру (Bluebeam, PlanSwift,
     Sweet Home 3D), і так само робимо ми: це основний шлях.
  2. ocr       — читаємо розмірні числа й голосуванням підбираємо масштаб, за
     якого найбільше чисел збігається з реальними відстанями між стінами.
     Українські плани за ДСТУ ГОСТ 2.307:2013 пишуть МІЛІМЕТРИ без одиниць —
     це прибирає половину неоднозначності.
  3. pdf       — розмір сторінки в пунктах + напис «М 1:100».
  4. door      — типові двері 0.85 м; похибка ~15%, але краще, ніж нічого.
  5. assumed   — остання лінія оборони: припускаємо типову квартиру, щоб
     користувачу було що рухати в редакторі.

OCR ніколи не застосовується мовчки: результат показуємо як підказку, яку
користувач підтверджує. Автоматичний масштаб без підтвердження — це рекламації.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .plan_model import PlanVector

TYPICAL_DOOR_M = 0.85
TYPICAL_APARTMENT_SPAN_M = 11.0

# Правдоподібні межі: житловий план рідко буває вужчим за 2 м і ширшим за 60 м.
MIN_PLAN_SPAN_M = 2.0
MAX_PLAN_SPAN_M = 60.0


@dataclass
class ScaleCandidate:
    m_per_px: float
    source: str
    confidence: float
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "m_per_px": self.m_per_px,
            "source": self.source,
            "confidence": round(self.confidence, 3),
            "detail": self.detail,
        }


@dataclass
class ScaleResult:
    chosen: ScaleCandidate
    candidates: List[ScaleCandidate] = field(default_factory=list)
    ocr_texts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "m_per_px": self.chosen.m_per_px,
            "source": self.chosen.source,
            "confidence": round(self.chosen.confidence, 3),
            "detail": self.chosen.detail,
            "candidates": [
                {"m_per_px": c.m_per_px, "source": c.source,
                 "confidence": round(c.confidence, 3), "detail": c.detail}
                for c in self.candidates
            ],
            "ocr": self.ocr_texts[:60],
        }


# ═════════════════════════════════════════════════════════════════════════════
#  Прості стратегії
# ═════════════════════════════════════════════════════════════════════════════
def from_reference(pixel_length: float, real_length_m: float) -> Optional[ScaleCandidate]:
    """Користувач: «оця лінія — 5.2 м». Найточніше, що взагалі можливо."""
    if pixel_length <= 1e-6 or real_length_m <= 1e-6:
        return None
    return ScaleCandidate(
        m_per_px=real_length_m / pixel_length,
        source="reference",
        confidence=0.99,
        detail=f"{real_length_m:.2f} м на {pixel_length:.0f} px",
    )


def _wall_bbox_area_px2(plan_px: PlanVector) -> float:
    """Площа габаритного прямокутника мережі стін — верхня межа для площі
    приміщень і знаменник у перевірці правдоподібності."""
    xs = [c for w in plan_px.walls for c in (w.x1, w.x2)]
    ys = [c for w in plan_px.walls for c in (w.y1, w.y2)]
    if len(xs) < 2:
        return 0.0
    return float(max(1e-9, (max(xs) - min(xs)) * (max(ys) - min(ys))))


def _hull_area_px2(union: Any) -> float:
    """Запасний спосіб: ОПУКЛА ОБОЛОНКА мережі стін мінус самі стіни.

    ЧОМУ ОБОЛОНКА, А НЕ ЗАЛИВКА ДІРОК. Один-єдиний розрив у ЗОВНІШНІЙ стіні —
    балконні двері, вхід, панорамне вікно — і заливка дірок перестає працювати
    взагалі: контур просочується крізь щілину, обтягує мережу зсередини, і
    «дірки» не лишається. Те саме з зовнішнім контуром компоненти. Заміряно на
    реальних планах: обидва давали площу 0.08 габариту замість ≈0.8, тобто
    масштаб помилявся втричі. Оболонка розриву не помічає в принципі.

    Плата — Г-подібні квартири трохи завищуються. На реальній геометрії Swiss
    Dwellings із відомим істинним масштабом це все одно ТОЧНІШЕ за кільцевий
    метод: медіана похибки 7.2% проти 8.4%, 90-й перцентиль 27.6% проти 39.5%."""
    try:
        hull = union.convex_hull
    except Exception:
        return 0.0
    if hull.is_empty or hull.area <= 0.0:
        return 0.0
    return float(max(0.0, hull.area - union.area))


def interior_area_px2(plan_px: PlanVector) -> float:
    """Площа ПРИМІЩЕНЬ (без стін) у пікселях².

    Потрібна для найзручнішого для українця способу задати масштаб: він не
    завжди розбере розміри на скані, але площу своєї квартири знає напам'ять —
    вона стоїть у договорі й у документах БТІ."""
    try:
        from shapely.geometry import LineString
        from shapely.ops import unary_union
    except ImportError:
        return 0.0
    strips = []
    for wall in plan_px.walls:
        if wall.length_m < 1e-6:
            continue
        strips.append(
            LineString([(wall.x1, wall.y1), (wall.x2, wall.y2)])
            .buffer(wall.thickness_m / 2.0, cap_style=3, join_style=2)
        )
    if not strips:
        return 0.0
    ring_area = 0.0
    union = None
    try:
        from shapely.geometry import Polygon as _Poly

        union = unary_union(strips)
        polygons = [union] if union.geom_type == "Polygon" else list(union.geoms)
        # НЕ сума внутрішніх кілець, а «весь контур мінус стіни». Сума кілець
        # рахує лише ЗАМКНЕНІ кімнати, і кожна незамкнена (а таких близько 2%)
        # просто випадала з площі, завищуючи масштаб на десятки відсотків.
        # Різниця площ від замикання кімнат не залежить взагалі.
        outer = sum(_Poly(p.exterior).area for p in polygons)
        walls = sum(p.area for p in polygons)
        ring_area = float(max(0.0, outer - walls))
    except Exception:
        ring_area = 0.0

    # Кільцевий метод точніший на Г-подібних планах, але він мовчки
    # вироджується, щойно зовнішній контур квартири десь розірвано: кільця
    # просто немає. На реальних планах (Floor Plan CIS) він давав НУЛЬ у 36%
    # випадків і мізерні 0.08 габариту в медіані — тобто найзручніший спосіб
    # задати масштаб не працював, причому мовчки. Правдоподібна квартира займає
    # щонайменше чверть свого габариту; усе нижче вважаємо виродженим і беремо
    # оболонку.
    bbox = _wall_bbox_area_px2(plan_px)
    if bbox > 1.0 and ring_area >= 0.25 * bbox:
        return ring_area
    if union is None or union.is_empty:
        return ring_area
    hull_area = _hull_area_px2(union)
    return hull_area if hull_area > ring_area else ring_area


def from_area(plan_px: PlanVector, area_m2: float) -> Optional[ScaleCandidate]:
    """Користувач вписав загальну площу → масштаб = √(площа / площа в пікселях).

    Точність залежить від того, чи всі кімнати замкнулись у детекції, тому
    впевненість середня: це підказка для підтвердження, а не істина."""
    if area_m2 <= 1.0:
        return None
    px2 = interior_area_px2(plan_px)
    if px2 <= 1.0:
        return None
    return ScaleCandidate(
        m_per_px=math.sqrt(area_m2 / px2),
        source="area",
        confidence=0.6,
        detail=f"з площі {area_m2:.0f} м²",
    )


def from_doors(plan_px: PlanVector, typical_m: float = TYPICAL_DOOR_M) -> Optional[ScaleCandidate]:
    """Медіанна ширина знайдених дверей ≈ 0.85 м."""
    widths = [o.width_m for o in plan_px.openings if o.kind in ("door", "arch") and o.width_m > 2]
    if len(widths) < 2:
        return None
    median_px = float(np.median(widths))
    if median_px <= 1e-6:
        return None
    return ScaleCandidate(
        m_per_px=typical_m / median_px,
        source="door",
        confidence=0.35 if len(widths) >= 4 else 0.25,
        detail=f"{len(widths)} дверей, медіана {median_px:.0f} px ≈ {typical_m} м",
    )


def from_assumed(plan_px: PlanVector) -> ScaleCandidate:
    """Останній варіант: типова квартира. Показуємо явно, що це припущення."""
    width_px, height_px = plan_px.size_m()   # у пікселях, попри назву
    span = max(width_px, height_px, 1.0)
    return ScaleCandidate(
        m_per_px=TYPICAL_APARTMENT_SPAN_M / span,
        source="assumed",
        confidence=0.08,
        detail=f"Припущено типовий габарит {TYPICAL_APARTMENT_SPAN_M:.0f} м — уточніть лінійкою",
    )


def from_pdf_note(page_pts: Sequence[float], image_px: Sequence[int],
                  denominator: float) -> Optional[ScaleCandidate]:
    """PDF: сторінка в пунктах + «М 1:100» → точний масштаб без жодного CV.

    1 пункт = 1/72 дюйма = 25.4/72 мм на аркуші. При масштабі 1:N цей самий
    пункт відповідає N·25.4/72 мм у реальності."""
    if not page_pts or not image_px or denominator <= 1:
        return None
    try:
        px_per_pt = float(image_px[0]) / float(page_pts[0])
    except (ZeroDivisionError, TypeError, ValueError):
        return None
    if px_per_pt <= 1e-6:
        return None
    mm_per_pt_real = denominator * 25.4 / 72.0
    m_per_px = (mm_per_pt_real / 1000.0) / px_per_pt
    return ScaleCandidate(
        m_per_px=m_per_px, source="pdf", confidence=0.75,
        detail=f"PDF, масштаб 1:{denominator:.0f}",
    )


# ═════════════════════════════════════════════════════════════════════════════
#  OCR
# ═════════════════════════════════════════════════════════════════════════════
_SCALE_NOTE_RE = re.compile(r"1\s*[:：]\s*(\d{1,4})")
_NUM_RE = re.compile(r"^\d[\d\s.,]*$")


def _parse_dimension(text: str) -> Optional[float]:
    """Рядок із плану → довжина в МЕТРАХ, або None.

    Конвенції, які реально трапляються:
        «2 800», «2800»  → міліметри (ДСТУ; пробіл як роздільник тисяч)
        «2,80», «2.80»   → метри (ДВІ цифри після коми)
        «12,1»           → ПЛОЩА КІМНАТИ в м², а НЕ довжина
        «280»            → сантиметри (рідко) — трактуємо як мм, це безпечніше

    ⚠️ ОДНА ЦИФРА ПІСЛЯ КОМИ = ПЛОЩА. На планах з оголошень нерухомості й у БТІ
    підписують площі кімнат («12,1», «16,4»). Раніше вони читались як 12.1 метра
    завдовжки: масштаб роздувався, квартира виходила 25×25 м = 600 м², і її
    відхиляв фізичний запобіжник. Заміряно на 99 реальних планах — наскрізний
    успіх падав із 87 до 60. Довжини в метрах креслярі пишуть із ДВОМА знаками
    («3,50»), тож розрізнити можна надійно.
    """
    raw = text.strip().replace(" ", " ")
    if not _NUM_RE.match(raw):
        return None
    compact = raw.replace(" ", "")
    if "," in compact or "." in compact:
        try:
            value = float(compact.replace(",", "."))
        except ValueError:
            return None
        tail = compact.split(",")[-1] if "," in compact else compact.split(".")[-1]
        if len(tail) == 1:
            return None                    # площа кімнати, а не довжина
        # 2.80 → метри; 2800.0 → міліметри
        return value if 0.4 <= value <= 40.0 else (value / 1000.0 if 400 <= value <= 40000 else None)
    if not compact.isdigit():
        return None
    number = int(compact)
    if 300 <= number <= 40000:
        return number / 1000.0        # міліметри
    return None


def run_ocr(rgb: np.ndarray) -> List[Dict[str, Any]]:
    """RapidOCR (ONNXRuntime, ~46 МБ ваг, CPU) → [{text, box, score, value_m}].

    Обрано за замірами: читає й ПОВЕРНУТІ вертикальні розміри, які Tesseract
    стабільно губить, і влазить у 4 ГБ VPS (EasyOCR/PaddleOCR — ні).
    Відсутність пакета не є помилкою: масштаб просто визначиться інакше."""
    try:
        from rapidocr import RapidOCR  # type: ignore
    except ImportError:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except ImportError:
            return []
    try:
        engine = _ocr_engine(RapidOCR)
        result = engine(rgb)
    except Exception:
        return []

    raw = getattr(result, "boxes", None)
    out: List[Dict[str, Any]] = []
    if raw is not None:                      # rapidocr >= 2.x повертає обʼєкт
        texts = list(getattr(result, "txts", []) or [])
        scores = list(getattr(result, "scores", []) or [])
        # `raw or []` тут ЛАМАЄТЬСЯ: у rapidocr 3.x boxes — numpy-масив, і `or`
        # питає його істинність → ValueError «truth value is ambiguous». Через
        # це OCR мовчки падав у except і сервіс завжди вважав, що його немає.
        for i, box in enumerate(list(raw)):
            text = str(texts[i]) if i < len(texts) else ""
            score = float(scores[i]) if i < len(scores) else 0.0
            out.append({"text": text, "box": np.asarray(box).tolist(), "score": score})
    else:                                    # старий формат: [[box, text, score], ...]
        for item in (result[0] if isinstance(result, tuple) else result) or []:
            try:
                box, text, score = item[0], item[1], item[2]
            except Exception:
                continue
            out.append({"text": str(text), "box": np.asarray(box).tolist(), "score": float(score)})

    for item in out:
        item["value_m"] = _parse_dimension(item["text"])
    return out


_OCR_SINGLETON: Dict[str, Any] = {}


def _ocr_engine(cls):
    """Один інстанс на процес: ініціалізація тягне ваги з диска ~1 с."""
    engine = _OCR_SINGLETON.get("engine")
    if engine is None:
        engine = cls()
        _OCR_SINGLETON["engine"] = engine
    return engine


def scale_note_from_ocr(ocr_items: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Шукає напис «М 1:100» / «1:50» у розпізнаному тексті."""
    for item in ocr_items:
        match = _SCALE_NOTE_RE.search(str(item.get("text", "")))
        if match:
            try:
                denominator = float(match.group(1))
            except ValueError:
                continue
            if 10 <= denominator <= 2000:
                return denominator
    return None


def _candidate_spans(plan_px: PlanVector
                     ) -> Dict[str, List[Tuple[float, float, float]]]:
    """Пари стін по кожній осі: {"x": [(ліва, права, У_СВІТЛІ)], "y": [...]}.

    На відміну від `_candidate_distances_px` тут зберігаються САМІ КООРДИНАТИ —
    вони потрібні, щоб зіставити розмір із тим місцем, де він підписаний.

    Третє число — відстань У СВІТЛІ, тобто між ГРАНЯМИ стін, а не між осями.
    Саме її пишуть у розмірних ланцюжках житлових планів. Без цієї поправки
    масштаб виходив систематично на ~8% замалим: на кімнаті 3.5 м дві половинки
    стін по 0.15 м — це рівно ті 8%."""
    xs: List[Tuple[float, float]] = []
    ys: List[Tuple[float, float]] = []
    for wall in plan_px.walls:
        dx, dy = wall.x2 - wall.x1, wall.y2 - wall.y1
        if abs(dx) >= abs(dy):
            ys.append(((wall.y1 + wall.y2) / 2.0, wall.thickness_m))
        else:
            xs.append(((wall.x1 + wall.x2) / 2.0, wall.thickness_m))
    # Габарит сюди НЕ додаємо: bounds() дає межі СМУГ стін (вісь ± пів-товщини),
    # тобто ті самі зовнішні стіни, але зсунуті й із нульовою товщиною. Пари з
    # ними давали «у світлі», що насправді дорівнює відстані між ОСЯМИ, — і
    # масштаб виходив на 5% меншим. Осі зовнішніх стін уже є у списку.
    out: Dict[str, List[Tuple[float, float, float]]] = {"x": [], "y": []}
    for axis, raw in (("x", xs), ("y", ys)):
        # одна координата може прийти від кількох стін — беремо найтовщу
        merged: Dict[float, float] = {}
        for coord, thickness in raw:
            key = round(coord, 1)
            merged[key] = max(merged.get(key, 0.0), float(thickness))
        values = sorted(merged.items())
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                (a, ta), (b, tb) = values[i], values[j]
                clear = (b - a) - (ta + tb) / 2.0
                if clear > 4.0:
                    out[axis].append((a, b, clear))
    return out


def _text_axis_center(item: Dict[str, Any], to_plan: float
                      ) -> Optional[Tuple[str, float]]:
    """Напис розміру → (вісь, координата центру) у пікселях ПЛАНУ.

    Горизонтальний напис стоїть на горизонтальному розмірному ланцюжку й міряє
    відстань по X; повернутий на 90° — по Y. Це та сама конвенція ДСТУ ГОСТ
    2.307, за якою креслять усі плани, і вона надійніша за будь-яку евристику."""
    box = item.get("box")
    if box is None:
        return None
    arr = np.asarray(box, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return None
    w = float(arr[:, 0].max() - arr[:, 0].min())
    h = float(arr[:, 1].max() - arr[:, 1].min())
    if max(w, h) < 4.0:
        return None
    if w >= h:
        return "x", float(arr[:, 0].mean()) * to_plan
    return "y", float(arr[:, 1].mean()) * to_plan


def _candidate_distances_px(plan_px: PlanVector) -> List[float]:
    """Реальні відстані на плані, до яких можуть належати розмірні числа:
    між паралельними стінами + загальні габарити."""
    out: List[float] = []
    horizontals: List[float] = []
    verticals: List[float] = []
    for wall in plan_px.walls:
        dx, dy = wall.x2 - wall.x1, wall.y2 - wall.y1
        if abs(dx) >= abs(dy):
            horizontals.append((wall.y1 + wall.y2) / 2.0)
        else:
            verticals.append((wall.x1 + wall.x2) / 2.0)
    for values in (horizontals, verticals):
        values.sort()
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                d = values[j] - values[i]
                if d > 4.0:
                    out.append(d)
    minx, miny, maxx, maxy = plan_px.bounds()
    out.extend([maxx - minx, maxy - miny])
    return [d for d in out if d > 4.0]


def from_ocr(ocr_items: Sequence[Dict[str, Any]], plan_px: PlanVector,
             ocr_to_plan: float = 1.0) -> Optional[ScaleCandidate]:
    """Голосування: який масштаб робить найбільше розмірних чисел правдою?

    Ми НЕ намагаємось розібрати розмірні лінії зі стрілками й виносками — це
    крихко. Замість цього кожна пара (число V, відстань між стінами D) дає
    гіпотезу масштабу V/D; правильний масштаб набирає найбільше голосів. Це,
    по суті, RANSAC по одному параметру.

    ⚠️ ПОЗИЦІЯ НАПИСУ ОБОВ'ЯЗКОВА. Спершу зіставлялися ВСІ числа з УСІМА
    відстанями. На планах власника це давало стабільний і стабільно ХИБНИЙ
    консенсус: стін мало, пар відстаней багато, і великі числа охоче лягали на
    повні прольоти — квартира 7.5×7.5 м виходила 3.7×3.7 при «7 з 11 розмірів
    збіглись». Розмір підписують ПОСЕРЕДИНІ того, що він міряє, тож число
    зіставляємо лише з тими парами стін, між якими воно стоїть.

    `ocr_to_plan` — перевідник координат: OCR працює на повному аркуші, а план
    живе у зменшеній робочій копії."""
    values = [it["value_m"] for it in ocr_items if it.get("value_m")]
    if len(values) < 3 or not plan_px.walls:
        return None
    distances = _candidate_distances_px(plan_px)
    if not distances:
        return None

    minx, miny, maxx, maxy = plan_px.bounds()
    span_px = max(maxx - minx, maxy - miny, 1.0)
    lo = MIN_PLAN_SPAN_M / span_px
    hi = MAX_PLAN_SPAN_M / span_px

    # ФІЗИЧНЕ ОБМЕЖЕННЯ, без якого голосування збиралось навколо ПОМИЛКОВОГО
    # кластера. Розмір, написаний на кресленні, міряє щось УСЕРЕДИНІ цього
    # креслення — отже план не може бути меншим за найбільше прочитане число.
    # Заміряно на планах власника: без цієї перевірки квартира 7.5×7.5 м
    # виходила 3.7×3.7 м, і при цьому «7 з 11 розмірів збіглись» — консенсус
    # був стабільний і стабільно хибний, бо стін мало, а відстаней між ними
    # багато, і великі числа охоче лягали на повні прольоти.
    #   • нижня межа: найбільший розмір ≤ проліт плану × 1.25 (запас на те, що
    #     габаритний розмір міряє по ЗОВНІШНІХ гранях, а bounds() — по осях
    #     стін, плюс частина зовнішньої стіни могла не розпізнатись);
    #   • верхня: план не буває вчетверо більшим за свій найбільший розмір —
    #     хоч одна кімната чи габарит завжди підписані.
    max_value = max(values)
    lo = max(lo, max_value / (1.25 * span_px))
    hi = min(hi, 4.0 * max_value / span_px)
    if lo >= hi:
        return None

    # ── 1. Позиційне зіставлення: число ↔ те, що стоїть під ним ──────────────
    # Гіпотези зберігаємо РАЗОМ із номером числа, яке їх породило: голосувати
    # треба різними числами, а не кількістю гіпотез (див. нижче).
    spans = _candidate_spans(plan_px)
    hypotheses: List[Tuple[float, int]] = []
    positional = 0
    for idx, item in enumerate(ocr_items):
        v = item.get("value_m")
        if not v:
            continue
        geom = _text_axis_center(item, ocr_to_plan)
        if geom is None:
            continue
        axis, center = geom
        for a, b, clear in spans[axis]:
            mid = (a + b) / 2.0
            # напис мусить стояти приблизно посередині виміряного відрізка
            if abs(center - mid) > 0.35 * (b - a):
                continue
            s = v / clear                      # розмір у світлі, не між осями
            if lo <= s <= hi:
                hypotheses.append((s, idx))
                positional += 1

    # ── 2. Запасний шлях: якщо написи не лягли (повернутий аркуш, хитрий
    #      макет) — старе зіставлення «всі з усіма» під фізичним обмеженням.
    # Поріг рахує РІЗНІ числа, а не гіпотези: три незалежні розміри, що зійшлись
    # на одному масштабі, — це вже консенсус. За порогом «менше 5 гіпотез»
    # запасний шлях вмикався навіть тоді, коли позиційне зіставлення дало
    # ідеальну відповідь, і додавав відстані між ОСЯМИ — масштаб зсувався на 5%.
    positional_values = len({i for _, i in hypotheses})
    if positional_values < 3:
        for vi, v in enumerate(values):
            for d in distances:
                s = v / d
                if lo <= s <= hi:
                    hypotheses.append((s, 10_000 + vi))
    # Мінімум — три гіпотези, бо нижче й так вимагається підтримка від трьох
    # РІЗНИХ чисел. Поріг «5 гіпотез» відкидав ідеальні позиційні збіги, де
    # кожне число дало рівно одну гіпотезу.
    if len(hypotheses) < 3:
        return None

    # ГОЛОСУЮТЬ РІЗНІ ЧИСЛА, А НЕ ГІПОТЕЗИ. Одне число дає стільки гіпотез,
    # скільки пар стін під нього підійшло, тож підрахунок «скільки гіпотез у
    # кластері» вигравав тим, у кого пар більше, а не тим, кого підтвердило
    # більше незалежних розмірів. На плані власника через це перемагав кластер
    # від ОДНОГО числа: квартира 7.6 м виходила 4.5 м, хоча 7 із 8 розмірів
    # чесно вказували на правильний масштаб.
    logs = np.log(np.array([s for s, _ in hypotheses]))
    owners = np.array([i for _, i in hypotheses])
    tolerance = math.log(1.025)
    best_center, best_support, best_count = None, 0, 0
    for k, center in enumerate(logs):
        inlier = np.abs(logs - center) <= tolerance
        support = int(np.unique(owners[inlier]).size)
        count = int(inlier.sum())
        if support > best_support or (support == best_support and count > best_count):
            best_support, best_count, best_center = support, count, center
    if best_center is None or best_support < 3:
        return None
    inliers = logs[np.abs(logs - best_center) <= tolerance]
    m_per_px = float(np.exp(np.median(inliers)))

    supporting = best_support
    unique_values = max(1, len(set(round(x, 4) for x in values)))
    ratio = supporting / unique_values
    confidence = float(np.clip(0.25 + 0.6 * ratio, 0.25, 0.85))
    return ScaleCandidate(
        m_per_px=m_per_px, source="ocr", confidence=confidence,
        detail=f"{supporting}/{unique_values} розмірів збіглись (±2.5%)",
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Вибір
# ═════════════════════════════════════════════════════════════════════════════
def resolve_scale(plan_px: PlanVector, *, rgb: Optional[np.ndarray] = None,
                  reference_px: Optional[float] = None,
                  reference_m: Optional[float] = None,
                  pdf_page_pts: Optional[Sequence[float]] = None,
                  image_px: Optional[Sequence[int]] = None,
                  use_ocr: bool = True) -> ScaleResult:
    """Збирає всі гіпотези масштабу й вибирає найвпевненішу."""
    candidates: List[ScaleCandidate] = []
    ocr_items: List[Dict[str, Any]] = []

    if reference_px and reference_m:
        ref = from_reference(float(reference_px), float(reference_m))
        if ref:
            candidates.append(ref)

    if use_ocr and rgb is not None and not candidates:
        ocr_items = run_ocr(rgb)
        note = scale_note_from_ocr(ocr_items)
        if note and pdf_page_pts and image_px:
            pdf_candidate = from_pdf_note(pdf_page_pts, image_px, note)
            if pdf_candidate:
                candidates.append(pdf_candidate)
        # OCR читає ПОВНИЙ аркуш, а план живе у зменшеній робочій копії —
        # без цього перевідника позиції написів не збігаються з геометрією.
        ocr_to_plan = 1.0
        try:
            plan_w = float(plan_px.image_size_px[0] or 0.0)
            rgb_w = float(rgb.shape[1] or 0.0)
            if plan_w > 0 and rgb_w > 0:
                ocr_to_plan = plan_w / rgb_w
        except Exception:  # noqa: BLE001
            ocr_to_plan = 1.0
        ocr_candidate = from_ocr(ocr_items, plan_px, ocr_to_plan)
        if ocr_candidate:
            candidates.append(ocr_candidate)

    door_candidate = from_doors(plan_px)
    if door_candidate:
        candidates.append(door_candidate)

    # Перехресна перевірка: якщо двері й OCR розходяться менш ніж на 12% —
    # обидва майже напевно праві, піднімаємо довіру до OCR.
    ocr_c = next((c for c in candidates if c.source == "ocr"), None)
    if ocr_c and door_candidate:
        delta = abs(ocr_c.m_per_px - door_candidate.m_per_px) / max(1e-9, ocr_c.m_per_px)
        if delta < 0.12:
            ocr_c.confidence = min(0.92, ocr_c.confidence + 0.15)
            ocr_c.detail += f"; підтверджено шириною дверей (Δ{delta * 100:.0f}%)"

    candidates.append(from_assumed(plan_px))
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return ScaleResult(chosen=candidates[0], candidates=candidates, ocr_texts=ocr_items)
