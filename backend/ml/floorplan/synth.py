"""Процедурний генератор технічних планів квартир + піксель-точні маски.

НАВІЩО СИНТЕТИКА, А НЕ ГОТОВИЙ ДАТАСЕТ
--------------------------------------
1. ЛІЦЕНЗІЯ. CubiCasa5K (найпопулярніший датасет) — CC BY-NC-SA 4.0, тобто
   НЕкомерційна. monadruk.com продає друк → навчені на ньому ваги в проді це
   юридичний ризик. Синтетика знімає питання повністю: дані наші.
2. ДОМЕН. CubiCasa — фінські плани. Наші користувачі вантажать пострадянські
   БТІ/забудовницькі креслення: розмірні ланцюжки «2 800», штриховка несучих,
   підписи «Кімната 18,5 м²». Синтетику ми малюємо саме в цьому стилі.
3. ІДЕАЛЬНА РОЗМІТКА. Маска стіни рівно там, де стіна — без похибки анотатора.
4. ДИСК. 5.5 ГБ архіву CubiCasa нікуди не влазить (на всіх дисках ~10 ГБ).

Класи маски (softmax, 4 канали):
    0 background, 1 wall, 2 door, 3 window
«wall» — це БЕЗПЕРЕРВНА смуга стіни (включно з ділянками під отворами), а door/
window перекривають її зверху. На інференсі wall_full = wall ∪ door ∪ window —
так векторизатор отримує суцільні смуги без розривів, а отвори знає окремо.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ── Класи ────────────────────────────────────────────────────────────────────
CLASS_BG = 0
CLASS_WALL = 1
CLASS_DOOR = 2
CLASS_WINDOW = 3
NUM_CLASSES = 4
CLASS_NAMES = ("background", "wall", "door", "window")

Rect = Tuple[float, float, float, float]  # (x0, y0, x1, y1) у метрах


# ── Шрифти ───────────────────────────────────────────────────────────────────
def _font_candidates() -> List[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    mpl = os.path.join(
        here, "..", "..", "venv", "Lib", "site-packages", "matplotlib",
        "mpl-data", "fonts", "ttf",
    )
    out = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\consola.ttf",
        os.path.normpath(os.path.join(mpl, "DejaVuSans.ttf")),
        os.path.normpath(os.path.join(mpl, "DejaVuSansCondensed.ttf")),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    return [p for p in out if os.path.exists(p)]


_FONT_PATHS = _font_candidates()
_FONT_CACHE: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}


def _font(size: int, rng: Optional[np.random.Generator] = None) -> ImageFont.ImageFont:
    size = max(6, int(size))
    if not _FONT_PATHS:
        return ImageFont.load_default()
    path = _FONT_PATHS[0] if rng is None else _FONT_PATHS[int(rng.integers(len(_FONT_PATHS)))]
    key = (path, size)
    cached = _FONT_CACHE.get(key)
    if cached is None:
        try:
            cached = ImageFont.truetype(path, size)
        except OSError:
            cached = ImageFont.load_default()
        _FONT_CACHE[key] = cached
    return cached


# ── Осе-паралельні відрізки (внутрішнє представлення розкладки) ───────────────
@dataclass
class Seg:
    """Осе-паралельний відрізок центральної лінії стіни.

    horizontal=True → const це y, [a, b] це діапазон x. Інакше навпаки.
    Таке представлення дає дешеве віднімання інтервалів (виріз кутової кімнати)
    і злиття колінеарних шматків — з довільними відрізками це було б морокою.
    """

    horizontal: bool
    const: float
    a: float
    b: float
    thickness: float = 0.12
    bearing: bool = False
    exterior: bool = False

    def __post_init__(self) -> None:
        if self.a > self.b:
            self.a, self.b = self.b, self.a

    @property
    def length(self) -> float:
        return self.b - self.a

    def endpoints(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        if self.horizontal:
            return (self.a, self.const), (self.b, self.const)
        return (self.const, self.a), (self.const, self.b)

    def subtract(self, lo: float, hi: float) -> List["Seg"]:
        """Відняти інтервал [lo, hi] — використовується при вирізанні кута."""
        if hi <= self.a + 1e-6 or lo >= self.b - 1e-6:
            return [self]
        out: List[Seg] = []
        if lo > self.a + 1e-6:
            out.append(Seg(self.horizontal, self.const, self.a, lo, self.thickness,
                           self.bearing, self.exterior))
        if hi < self.b - 1e-6:
            out.append(Seg(self.horizontal, self.const, hi, self.b, self.thickness,
                           self.bearing, self.exterior))
        return out


@dataclass
class OpeningSpec:
    """Отвір у координатах метрів (осе-паралельний, як і стіни)."""

    seg_index: int
    center: float          # позиція вздовж відрізка (у тій самій осі, що a..b)
    width: float
    kind: str              # door | window
    swing_dir: int = 1     # 1/-1 — у який бік малювати дугу дверей


@dataclass
class LayoutSpec:
    """Готова розкладка: кімнати, стіни, отвори — все в метрах."""

    width_m: float
    height_m: float
    rooms: List[Rect] = field(default_factory=list)
    room_names: List[str] = field(default_factory=list)
    segs: List[Seg] = field(default_factory=list)
    openings: List[OpeningSpec] = field(default_factory=list)


# ── Назви кімнат (стиль наших ринків) ────────────────────────────────────────
_ROOM_NAMES_UK = ["Кімната", "Спальня", "Вітальня", "Кухня", "Санвузол", "Ванна",
                  "Коридор", "Передпокій", "Балкон", "Гардероб", "Кабінет", "Дитяча"]
_ROOM_NAMES_RU = ["Комната", "Спальня", "Гостиная", "Кухня", "Санузел", "Ванная",
                  "Коридор", "Прихожая", "Балкон", "Кладовая", "Кабинет", "Детская"]
_ROOM_NAMES_EN = ["Room", "Bedroom", "Living", "Kitchen", "Bath", "WC",
                  "Hall", "Corridor", "Balcony", "Closet", "Office", "Study"]
_ROOM_NAMES_PL = ["Pokój", "Sypialnia", "Salon", "Kuchnia", "Łazienka", "WC",
                  "Przedpokój", "Korytarz", "Balkon", "Garderoba", "Gabinet"]
_ROOM_SETS = [_ROOM_NAMES_UK, _ROOM_NAMES_RU, _ROOM_NAMES_EN, _ROOM_NAMES_PL]


# ═════════════════════════════════════════════════════════════════════════════
#  1. РОЗКЛАДКА
# ═════════════════════════════════════════════════════════════════════════════
def _bsp(rng: np.random.Generator, rect: Rect, depth: int, min_side: float,
         min_area: float, rooms: List[Rect], cuts: List[Tuple[int, Seg]]) -> None:
    """Рекурсивне бінарне розбиття прямокутника на кімнати.

    cuts збирає (глибина, відрізок) — глибина потім керує товщиною: перші розрізи
    роблять несучими, глибокі — тонкими перегородками, як у реальних будинках."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    can_v = w >= 2 * min_side
    can_h = h >= 2 * min_side
    too_small = (w * h) < 2 * min_area
    if depth <= 0 or too_small or not (can_v or can_h):
        rooms.append(rect)
        return
    # Ранній стоп → різноманіття розмірів кімнат (не всі однакові).
    if depth <= 2 and rng.random() < 0.30:
        rooms.append(rect)
        return

    if can_v and can_h:
        vertical = w >= h if rng.random() < 0.75 else rng.random() < 0.5
    else:
        vertical = can_v

    ratio = float(rng.uniform(0.35, 0.65))
    if vertical:
        cut = x0 + w * ratio
        cut = min(max(cut, x0 + min_side), x1 - min_side)
        cuts.append((depth, Seg(False, cut, y0, y1)))
        _bsp(rng, (x0, y0, cut, y1), depth - 1, min_side, min_area, rooms, cuts)
        _bsp(rng, (cut, y0, x1, y1), depth - 1, min_side, min_area, rooms, cuts)
    else:
        cut = y0 + h * ratio
        cut = min(max(cut, y0 + min_side), y1 - min_side)
        cuts.append((depth, Seg(True, cut, x0, x1)))
        _bsp(rng, (x0, y0, x1, cut), depth - 1, min_side, min_area, rooms, cuts)
        _bsp(rng, (x0, cut, x1, y1), depth - 1, min_side, min_area, rooms, cuts)


def _shared_edge(a: Rect, b: Rect, tol: float = 1e-6) -> Optional[Tuple[bool, float, float, float]]:
    """Спільна межа двох кімнат → (horizontal, const, lo, hi) або None."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    # вертикальна межа: права стіна a збігається з лівою b (або навпаки)
    for ca, cb in ((ax1, bx0), (bx1, ax0)):
        if abs(ca - cb) < tol:
            lo, hi = max(ay0, by0), min(ay1, by1)
            if hi - lo > 0.6:  # надто короткий контакт — двері не влізуть
                return (False, ca, lo, hi)
    for ca, cb in ((ay1, by0), (by1, ay0)):
        if abs(ca - cb) < tol:
            lo, hi = max(ax0, bx0), min(ax1, bx1)
            if hi - lo > 0.6:
                return (True, ca, lo, hi)
    return None


def _merge_collinear(segs: List[Seg]) -> List[Seg]:
    """Зливає колінеарні відрізки, що перекриваються/торкаються (одна товщина)."""
    out: List[Seg] = []
    buckets: Dict[Tuple[bool, float, float, bool], List[Seg]] = {}
    for s in segs:
        key = (s.horizontal, round(s.const, 4), round(s.thickness, 4), s.bearing)
        buckets.setdefault(key, []).append(s)
    for (horizontal, _const_key, _th_key, bearing), group in buckets.items():
        group.sort(key=lambda s: s.a)
        cur = None
        for s in group:
            # ВАЖЛИВО: беремо ТОЧНІ const/thickness з відрізка, а не округлені
            # значення ключа. Округлення тут колись зсувало стіну на ~4e-5 м, і
            # пошук стіни за спільною межею кімнат (_find_seg) переставав її
            # знаходити — усі МІЖКІМНАТНІ ДВЕРІ мовчки зникали з плану.
            if cur is None:
                cur = Seg(horizontal, s.const, s.a, s.b, s.thickness, bearing, s.exterior)
                continue
            if s.a <= cur.b + 1e-6:
                cur.b = max(cur.b, s.b)
                cur.exterior = cur.exterior or s.exterior
            else:
                out.append(cur)
                cur = Seg(horizontal, s.const, s.a, s.b, s.thickness, bearing, s.exterior)
        if cur is not None:
            out.append(cur)
    return out


def generate_layout(rng: np.random.Generator) -> LayoutSpec:
    """Створює правдоподібну розкладку квартири/будинку."""
    width = float(rng.uniform(7.0, 16.0))
    height = float(rng.uniform(6.0, 13.0))
    depth = int(rng.integers(2, 5))
    min_side = float(rng.uniform(1.9, 2.7))
    min_area = float(rng.uniform(5.0, 11.0))

    rooms: List[Rect] = []
    cuts: List[Tuple[int, Seg]] = []
    _bsp(rng, (0.0, 0.0, width, height), depth, min_side, min_area, rooms, cuts)

    # Товщини: несучі — зовнішні + перші (мілкі за глибиною = ранні) розрізи.
    ext_th = float(rng.choice([0.30, 0.38, 0.40, 0.51]))
    bear_th = float(rng.choice([0.25, 0.30, 0.38]))
    part_th = float(rng.choice([0.08, 0.10, 0.12, 0.15]))
    max_depth = max((d for d, _ in cuts), default=0)

    segs: List[Seg] = []
    for d, seg in cuts:
        is_bearing = d >= max_depth and rng.random() < 0.7
        seg.thickness = bear_th if is_bearing else part_th
        seg.bearing = is_bearing
        segs.append(seg)

    # Зовнішній контур
    outer = [
        Seg(True, 0.0, 0.0, width, ext_th, True, True),
        Seg(True, height, 0.0, width, ext_th, True, True),
        Seg(False, 0.0, 0.0, height, ext_th, True, True),
        Seg(False, width, 0.0, height, ext_th, True, True),
    ]

    # Г-подібний контур: викидаємо одну кутову кімнату.
    if len(rooms) >= 4 and rng.random() < 0.35:
        corners = [(0.0, 0.0), (width, 0.0), (0.0, height), (width, height)]
        cx, cy = corners[int(rng.integers(4))]
        best_i, best_d = -1, 1e9
        for i, (x0, y0, x1, y1) in enumerate(rooms):
            touches_x = abs(x0 - cx) < 1e-6 or abs(x1 - cx) < 1e-6
            touches_y = abs(y0 - cy) < 1e-6 or abs(y1 - cy) < 1e-6
            if not (touches_x and touches_y):
                continue
            area = (x1 - x0) * (y1 - y0)
            if area < 0.30 * width * height and area < best_d:
                best_i, best_d = i, area
        if best_i >= 0:
            rx0, ry0, rx1, ry1 = rooms.pop(best_i)
            trimmed: List[Seg] = []
            for s in outer:
                if s.horizontal and (abs(s.const - ry0) < 1e-6 or abs(s.const - ry1) < 1e-6):
                    trimmed.extend(s.subtract(rx0, rx1))
                elif (not s.horizontal) and (abs(s.const - rx0) < 1e-6 or abs(s.const - rx1) < 1e-6):
                    trimmed.extend(s.subtract(ry0, ry1))
                else:
                    trimmed.append(s)
            for s in trimmed:
                s.thickness, s.bearing, s.exterior = ext_th, True, True
            outer = [s for s in trimmed if s.length > 0.2]
            # внутрішні межі вирізаної кімнати стають зовнішніми стінами
            for horizontal, const, lo, hi in (
                (True, ry0, rx0, rx1), (True, ry1, rx0, rx1),
                (False, rx0, ry0, ry1), (False, rx1, ry0, ry1),
            ):
                on_border = (
                    (horizontal and (abs(const) < 1e-6 or abs(const - height) < 1e-6))
                    or ((not horizontal) and (abs(const) < 1e-6 or abs(const - width) < 1e-6))
                )
                if not on_border:
                    outer.append(Seg(horizontal, const, lo, hi, ext_th, True, True))

    segs = _merge_collinear(segs + outer)

    # ── Двері: кістяк зв'язності по графу суміжності кімнат ───────────────────
    openings: List[OpeningSpec] = []
    n = len(rooms)
    edges: List[Tuple[int, int, Tuple[bool, float, float, float]]] = []
    for i in range(n):
        for j in range(i + 1, n):
            shared = _shared_edge(rooms[i], rooms[j])
            if shared:
                edges.append((i, j, shared))
    rng.shuffle(edges)  # type: ignore[arg-type]

    parent = list(range(n))

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    chosen: List[Tuple[bool, float, float, float]] = []
    for i, j, shared in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            chosen.append(shared)
        elif rng.random() < 0.18:      # зайві прорізи (арки/другі двері)
            chosen.append(shared)

    def _find_seg(horizontal: bool, const: float, lo: float, hi: float) -> int:
        # 1 мм допуску: стіни ніколи не стоять ближче, тож переплутати не можемо.
        for idx, s in enumerate(segs):
            if s.horizontal != horizontal or abs(s.const - const) > 1e-3:
                continue
            if s.a <= lo + 1e-3 and s.b >= hi - 1e-3:
                return idx
        return -1

    used: List[Tuple[int, float, float]] = []
    for horizontal, const, lo, hi in chosen:
        idx = _find_seg(horizontal, const, lo, hi)
        if idx < 0:
            continue
        width_m = float(rng.uniform(0.70, 0.95))
        span = hi - lo
        if span < width_m + 0.30:
            continue
        center = float(rng.uniform(lo + width_m / 2 + 0.12, hi - width_m / 2 - 0.12))
        if any(k == idx and abs(center - c) < (width_m + w) / 2 + 0.15 for k, c, w in used):
            continue
        used.append((idx, center, width_m))
        openings.append(OpeningSpec(idx, center, width_m, "door",
                                    1 if rng.random() < 0.5 else -1))

    # ── Вікна: на зовнішніх стінах ───────────────────────────────────────────
    for idx, s in enumerate(segs):
        if not s.exterior or s.length < 1.6:
            continue
        count = 1 if s.length < 4.5 else int(rng.integers(1, 3))
        for k in range(count):
            width_m = float(rng.uniform(0.90, 1.80))
            slot = s.length / count
            lo = s.a + slot * k + 0.45
            hi = s.a + slot * (k + 1) - 0.45
            if hi - lo < width_m:
                continue
            center = float(rng.uniform(lo + width_m / 2, hi - width_m / 2))
            if any(o.seg_index == idx and abs(o.center - center) < (width_m + o.width) / 2 + 0.25
                   for o in openings):
                continue
            if rng.random() < 0.82:
                openings.append(OpeningSpec(idx, center, width_m, "window"))

    # Вхідні двері у зовнішній стіні
    ext_idxs = [i for i, s in enumerate(segs) if s.exterior and s.length > 1.6]
    if ext_idxs:
        idx = int(rng.choice(ext_idxs))
        s = segs[idx]
        width_m = float(rng.uniform(0.90, 1.10))
        if s.length > width_m + 1.0:
            center = float(rng.uniform(s.a + width_m / 2 + 0.4, s.b - width_m / 2 - 0.4))
            if not any(o.seg_index == idx and abs(o.center - center) < (width_m + o.width) / 2 + 0.2
                       for o in openings):
                openings.append(OpeningSpec(idx, center, width_m, "door",
                                            1 if rng.random() < 0.5 else -1))

    names_pool = _ROOM_SETS[int(rng.integers(len(_ROOM_SETS)))]
    room_names = [str(rng.choice(names_pool)) for _ in rooms]

    return LayoutSpec(width, height, rooms, room_names, segs, openings)


# ═════════════════════════════════════════════════════════════════════════════
#  2. РЕНДЕР
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class RenderStyle:
    px_per_m: float = 40.0
    margin_px: int = 90
    wall_fill: int = 25
    line_width: int = 2
    draw_dims: bool = True
    draw_labels: bool = True
    draw_areas: bool = True
    draw_hatch: bool = True
    draw_furniture: bool = True
    draw_frame: bool = True
    hollow_walls: bool = False   # стіни двома лініями замість заливки
    dim_units: str = "mm"        # mm | m | cm
    # Елементи, які найчастіше плутають детектор із стінами. Вони МУСЯТЬ бути в
    # навчальній вибірці, інакше модель уперше побачить їх у реального клієнта.
    draw_axes: bool = False      # осі з кружечками (класика будівельних креслень)
    draw_title_block: bool = False
    draw_room_badges: bool = False
    hatch_angle_alt: bool = False
    wall_gray: int = 25          # стіни не завжди чорні
    # МАРКЕТИНГОВИЙ РЕНДЕР ЗАБУДОВНИКА — окремий світ, а не варіація техплану:
    # бежеве тло, стіни сірою заливкою з тонким контуром, щільні меблі, замість
    # підписів самі цифри площ, жодних розмірних ланцюжків і штампів. Саме на
    # таких планах модель, навчена лише на кресленнях, приймала меблі за стіни
    # (198 тис. пікселів проти 30 тис. у розмітці). А в оголошеннях нерухомості
    # це найпоширеніший вигляд плану.
    marketing: bool = False
    paper_tone: int = 255        # тон тла аркуша


def _rand_style(rng: np.random.Generator) -> RenderStyle:
    style = RenderStyle(
        px_per_m=float(rng.uniform(26.0, 62.0)),
        margin_px=int(rng.integers(55, 150)),
        wall_fill=int(rng.integers(0, 70)),
        line_width=int(rng.integers(1, 4)),
        draw_dims=bool(rng.random() < 0.80),
        draw_labels=bool(rng.random() < 0.75),
        draw_areas=bool(rng.random() < 0.65),
        draw_hatch=bool(rng.random() < 0.35),
        draw_furniture=bool(rng.random() < 0.45),
        draw_frame=bool(rng.random() < 0.40),
        hollow_walls=bool(rng.random() < 0.22),
        dim_units=str(rng.choice(["mm", "mm", "mm", "m", "cm"])),
        draw_axes=bool(rng.random() < 0.30),
        draw_title_block=bool(rng.random() < 0.25),
        draw_room_badges=bool(rng.random() < 0.35),
        hatch_angle_alt=bool(rng.random() < 0.5),
        wall_gray=int(rng.integers(0, 110)),
    )
    # Кожен третій план — маркетинговий рендер забудовника. Це не «ще один
    # відтінок», а інший спосіб малювати план цілком, і в оголошеннях він
    # трапляється частіше за технічне креслення.
    if rng.random() < 0.33:
        style.marketing = True
        style.paper_tone = int(rng.integers(238, 253))
        style.wall_fill = int(rng.integers(90, 150))
        style.wall_gray = style.wall_fill
        style.draw_dims = False
        style.draw_axes = False
        style.draw_title_block = False
        style.draw_hatch = False
        style.hollow_walls = False
        style.draw_frame = False
        style.draw_furniture = True
        style.draw_labels = False        # замість підписів — самі цифри площ
        style.draw_areas = True
        style.draw_room_badges = bool(rng.random() < 0.5)
        style.margin_px = int(rng.integers(20, 60))
    return style


class _Canvas:
    """Малює одночасно у видиме зображення і в маску класів (без згладжування)."""

    def __init__(self, w: int, h: int, style: RenderStyle, ox: float, oy: float):
        self.style = style
        self.ox, self.oy = ox, oy
        self.img = Image.new("L", (w, h), 255)
        self.mask = Image.new("L", (w, h), CLASS_BG)
        self.d = ImageDraw.Draw(self.img)
        self.dm = ImageDraw.Draw(self.mask)

    def px(self, x: float, y: float) -> Tuple[float, float]:
        return (self.ox + x * self.style.px_per_m, self.oy + y * self.style.px_per_m)

    def wall_rect(self, seg: Seg) -> Tuple[float, float, float, float]:
        half = seg.thickness / 2.0
        if seg.horizontal:
            x0, y0 = self.px(seg.a, seg.const - half)
            x1, y1 = self.px(seg.b, seg.const + half)
        else:
            x0, y0 = self.px(seg.const - half, seg.a)
            x1, y1 = self.px(seg.const + half, seg.b)
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def opening_rect(self, seg: Seg, op: OpeningSpec) -> Tuple[float, float, float, float]:
        half_t = seg.thickness / 2.0
        half_w = op.width / 2.0
        if seg.horizontal:
            x0, y0 = self.px(op.center - half_w, seg.const - half_t)
            x1, y1 = self.px(op.center + half_w, seg.const + half_t)
        else:
            x0, y0 = self.px(seg.const - half_t, op.center - half_w)
            x1, y1 = self.px(seg.const + half_t, op.center + half_w)
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _draw_hatch(canvas: _Canvas, box: Tuple[float, float, float, float], step: int,
                shade: int, flip: bool = False) -> None:
    x0, y0, x1, y1 = box
    clip = Image.new("L", (max(1, int(x1 - x0)) + 1, max(1, int(y1 - y0)) + 1), 255)
    cd = ImageDraw.Draw(clip)
    w, h = clip.size
    for k in range(-h, w + h, max(3, step)):
        if flip:
            cd.line([(k, h), (k + h, 0)], fill=shade, width=1)
        else:
            cd.line([(k, 0), (k + h, h)], fill=shade, width=1)
    canvas.img.paste(clip, (int(x0), int(y0)))


def _draw_axes(canvas: _Canvas, rng: np.random.Generator, spec: "LayoutSpec",
               margin: int) -> None:
    """Розбивочні осі з кружечками — обов'язковий елемент будівельних креслень.

    Для детектора це найгірший ворог: довгі прямі лінії через увесь аркуш плюс
    кола, які легко приймаються за конструкцію. Модель мусить побачити їх у
    навчанні, інакше вперше зустріне у клієнта."""
    d = canvas.d
    font = _font(int(max(9, canvas.style.px_per_m * 0.26)), rng)
    radius = max(7, int(canvas.style.px_per_m * 0.20))
    xs = sorted({0.0, spec.width_m} | {round(s.const, 3) for s in spec.segs if not s.horizontal})
    ys = sorted({0.0, spec.height_m} | {round(s.const, 3) for s in spec.segs if s.horizontal})
    for i, x in enumerate(xs[:9]):
        px, _ = canvas.px(x, 0.0)
        top = margin * 0.22
        d.line([(px, top + radius), (px, canvas.img.height - margin * 0.25)], fill=150, width=1)
        d.ellipse([px - radius, top - radius, px + radius, top + radius], outline=90, width=1)
        label = str(i + 1)
        d.text((px - d.textlength(label, font=font) / 2, top - font.size / 2), label,
               fill=70, font=font)
    for j, y in enumerate(ys[:9]):
        _, py = canvas.px(0.0, y)
        left = margin * 0.22
        d.line([(left + radius, py), (canvas.img.width - margin * 0.25, py)], fill=150, width=1)
        d.ellipse([left - radius, py - radius, left + radius, py + radius], outline=90, width=1)
        label = chr(ord("А") + j)
        d.text((left - d.textlength(label, font=font) / 2, py - font.size / 2), label,
               fill=70, font=font)


def _draw_title_block(canvas: _Canvas, rng: np.random.Generator) -> None:
    """Штамп у куті: рамка з рядками тексту. Товсті лінії рамки — ще одне
    джерело хибних «стін», якщо модель такого не бачила."""
    d = canvas.d
    w, h = canvas.img.size
    bw = int(w * float(rng.uniform(0.24, 0.38)))
    bh = int(h * float(rng.uniform(0.08, 0.15)))
    x0, y0 = w - bw - 8, h - bh - 8
    d.rectangle([x0, y0, w - 8, h - 8], outline=40, width=2)
    rows = int(rng.integers(2, 5))
    font = _font(int(max(8, bh / (rows + 1) * 0.6)), rng)
    for i in range(1, rows):
        yy = y0 + bh * i / rows
        d.line([(x0, yy), (w - 8, yy)], fill=60, width=1)
    labels = ["Аркуш 1", "План 1-го поверху", "М 1:100", "Розробив", "ЖК «Сонячний»"]
    for i in range(rows):
        text = str(rng.choice(labels))
        d.text((x0 + 6, y0 + bh * i / rows + 3), text, fill=45, font=font)


def _draw_room_badges(canvas: _Canvas, rng: np.random.Generator, spec: "LayoutSpec") -> None:
    """Номер кімнати в кружечку — типово для БТІ."""
    d = canvas.d
    radius = max(8, int(canvas.style.px_per_m * 0.22))
    font = _font(int(radius * 1.1), rng)
    for index, room in enumerate(spec.rooms, start=1):
        cx = (room[0] + room[2]) / 2 + float(rng.uniform(-0.5, 0.5))
        cy = room[1] + (room[3] - room[1]) * 0.22
        px, py = canvas.px(cx, cy)
        d.ellipse([px - radius, py - radius, px + radius, py + radius], outline=50, width=1)
        label = str(index)
        d.text((px - d.textlength(label, font=font) / 2, py - font.size * 0.6), label,
               fill=50, font=font)


def _fmt_dim(value_m: float, units: str) -> str:
    if units == "mm":
        mm = int(round(value_m * 1000.0 / 10.0) * 10)
        s = f"{mm}"
        return f"{s[:-3]} {s[-3:]}" if len(s) > 3 else s
    if units == "cm":
        return f"{int(round(value_m * 100))}"
    return f"{value_m:.2f}".replace(".", ",")


def _draw_dim_chain(canvas: _Canvas, rng: np.random.Generator, positions: Sequence[float],
                    const_m: float, horizontal: bool, offset_px: float, units: str) -> None:
    """Розмірний ланцюжок з засічками й числами — головна ознака техплану."""
    if len(positions) < 2:
        return
    d = canvas.d
    font = _font(int(max(9, canvas.style.px_per_m * 0.30)), rng)
    tick = 5
    for i in range(len(positions) - 1):
        a, b = positions[i], positions[i + 1]
        if b - a < 0.25:
            continue
        if horizontal:
            (xa, y0) = canvas.px(a, const_m)
            (xb, _) = canvas.px(b, const_m)
            y = y0 + offset_px
            d.line([(xa, y), (xb, y)], fill=60, width=1)
            d.line([(xa, y - tick), (xa, y + tick)], fill=60, width=1)
            d.line([(xb, y - tick), (xb, y + tick)], fill=60, width=1)
            text = _fmt_dim(b - a, units)
            tw = d.textlength(text, font=font)
            d.text(((xa + xb) / 2 - tw / 2, y - tick - 3 - font.size), text, fill=40, font=font)
        else:
            (x0, ya) = canvas.px(const_m, a)
            (_, yb) = canvas.px(const_m, b)
            x = x0 + offset_px
            d.line([(x, ya), (x, yb)], fill=60, width=1)
            d.line([(x - tick, ya), (x + tick, ya)], fill=60, width=1)
            d.line([(x - tick, yb), (x + tick, yb)], fill=60, width=1)
            text = _fmt_dim(b - a, units)
            # вертикальний текст — рендеримо окремо й повертаємо
            tmp = Image.new("L", (int(d.textlength(text, font=font)) + 4, font.size + 4), 255)
            ImageDraw.Draw(tmp).text((2, 2), text, fill=40, font=font)
            tmp = tmp.rotate(90, expand=True)
            canvas.img.paste(tmp, (int(x - tick - 2 - tmp.width), int((ya + yb) / 2 - tmp.height / 2)))


def _draw_furniture(canvas: _Canvas, rng: np.random.Generator, room: Rect) -> None:
    """Тонкі меблі/сантехніка — головний джерело хибних спрацювань детектора,
    тому модель мусить бачити їх під час навчання і вчитися ігнорувати."""
    x0, y0, x1, y1 = room
    d = canvas.d
    count = int(rng.integers(3, 8)) if canvas.style.marketing else int(rng.integers(1, 4))
    for _ in range(count):
        w = float(rng.uniform(0.4, min(1.8, max(0.5, (x1 - x0) * 0.5))))
        h = float(rng.uniform(0.4, min(1.8, max(0.5, (y1 - y0) * 0.5))))
        px_ = float(rng.uniform(x0 + 0.25, max(x0 + 0.26, x1 - w - 0.25)))
        py_ = float(rng.uniform(y0 + 0.25, max(y0 + 0.26, y1 - h - 0.25)))
        a = canvas.px(px_, py_)
        b = canvas.px(px_ + w, py_ + h)
        shade = int(rng.integers(120, 190))
        if rng.random() < 0.3:
            d.ellipse([a, b], outline=shade, width=1)
        else:
            d.rectangle([a, b], outline=shade, width=1)
            if rng.random() < 0.4:
                d.line([a, b], fill=shade, width=1)


def render_layout(spec: LayoutSpec, rng: np.random.Generator,
                  style: Optional[RenderStyle] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Малює план → (grayscale uint8 HxW, mask uint8 HxW з класами)."""
    st = style or _rand_style(rng)
    margin = st.margin_px
    w_px = int(spec.width_m * st.px_per_m) + 2 * margin
    h_px = int(spec.height_m * st.px_per_m) + 2 * margin
    canvas = _Canvas(w_px, h_px, st, margin, margin)
    d, dm = canvas.d, canvas.dm
    if st.paper_tone < 255:
        d.rectangle([0, 0, w_px, h_px], fill=st.paper_tone)

    # 1. Підлога кімнат — легкий фон, іноді
    if rng.random() < 0.25:
        shade = int(rng.integers(238, 252))
        for room in spec.rooms:
            a = canvas.px(room[0], room[1])
            b = canvas.px(room[2], room[3])
            d.rectangle([a, b], fill=shade)

    # 2. Осі й меблі — ПІД стінами
    if st.draw_axes:
        _draw_axes(canvas, rng, spec, margin)
    if st.draw_furniture:
        for room in spec.rooms:
            if rng.random() < (0.95 if st.marketing else 0.6):
                _draw_furniture(canvas, rng, room)

    # 3. Стіни — заливка + БЕЗПЕРЕРВНА маска
    for seg in spec.segs:
        box = canvas.wall_rect(seg)
        dm.rectangle(box, fill=CLASS_WALL)
        # «Порожній» стиль (контур без заливки) застосовуємо ЛИШЕ до достатньо
        # товстих стін. Тонку перегородку 8 см на кресленні завжди заливають:
        # намальована контуром, вона вироджується у волосяну лінію завширшки в
        # 1 px, яку не відрізнити від розмірної. Раніше синтетика такі й
        # генерувала — і мережа вчилась на тому, чого в реальності не буває.
        hollow_ok = st.hollow_walls and not seg.bearing and seg.thickness >= 0.18
        if hollow_ok:
            d.rectangle(box, fill=255, outline=st.wall_fill, width=max(1, st.line_width - 1))
        else:
            d.rectangle(box, fill=max(st.wall_fill, st.wall_gray) if seg.bearing else st.wall_fill)
        if st.draw_hatch and seg.bearing and not st.hollow_walls and rng.random() < 0.7:
            _draw_hatch(canvas, box, int(rng.integers(4, 9)), int(rng.integers(90, 160)),
                        flip=st.hatch_angle_alt)
            d.rectangle(box, outline=st.wall_fill, width=1)

    # 4. Отвори — вирізаємо у зображенні, позначаємо у масці
    for op in spec.openings:
        seg = spec.segs[op.seg_index]
        box = canvas.opening_rect(seg, op)
        d.rectangle(box, fill=255)
        dm.rectangle(box, fill=CLASS_DOOR if op.kind == "door" else CLASS_WINDOW)
        x0, y0, x1, y1 = box
        if op.kind == "window":
            # 2-3 тонкі паралельні лінії поперек прорізу
            lines = int(rng.integers(2, 4))
            if seg.horizontal:
                for k in range(lines):
                    yy = y0 + (y1 - y0) * (k + 1) / (lines + 1)
                    d.line([(x0, yy), (x1, yy)], fill=st.wall_fill, width=1)
                d.line([(x0, y0), (x0, y1)], fill=st.wall_fill, width=1)
                d.line([(x1, y0), (x1, y1)], fill=st.wall_fill, width=1)
            else:
                for k in range(lines):
                    xx = x0 + (x1 - x0) * (k + 1) / (lines + 1)
                    d.line([(xx, y0), (xx, y1)], fill=st.wall_fill, width=1)
                d.line([(x0, y0), (x1, y0)], fill=st.wall_fill, width=1)
                d.line([(x0, y1), (x1, y1)], fill=st.wall_fill, width=1)
        else:
            # дверне полотно + дуга відкривання
            span = (x1 - x0) if seg.horizontal else (y1 - y0)
            if seg.horizontal:
                hx = x0 if op.swing_dir > 0 else x1
                cy = (y0 + y1) / 2
                sgn = 1 if rng.random() < 0.5 else -1
                d.line([(hx, cy), (hx, cy + sgn * span)], fill=st.wall_fill, width=1)
                bbox = [hx - span, cy - span, hx + span, cy + span]
                start, end = (0, 90) if sgn > 0 else (270, 360)
                if op.swing_dir < 0:
                    start, end = (90, 180) if sgn > 0 else (180, 270)
                d.arc(bbox, start=start, end=end, fill=int(min(255, st.wall_fill + 90)), width=1)
            else:
                hy = y0 if op.swing_dir > 0 else y1
                cx = (x0 + x1) / 2
                sgn = 1 if rng.random() < 0.5 else -1
                d.line([(cx, hy), (cx + sgn * span, hy)], fill=st.wall_fill, width=1)
                bbox = [cx - span, hy - span, cx + span, hy + span]
                start, end = (0, 90) if sgn > 0 else (90, 180)
                if op.swing_dir < 0:
                    start, end = (270, 360) if sgn > 0 else (180, 270)
                d.arc(bbox, start=start, end=end, fill=int(min(255, st.wall_fill + 90)), width=1)

    # 5. Підписи кімнат (у маркетинговому стилі — самі цифри площ)
    if st.draw_labels or st.marketing:
        font = _font(int(max(9, st.px_per_m * 0.34)), rng)
        for room, name in zip(spec.rooms, spec.room_names):
            cx = (room[0] + room[2]) / 2
            cy = (room[1] + room[3]) / 2
            area = (room[2] - room[0]) * (room[3] - room[1])
            px_, py_ = canvas.px(cx, cy)
            text = f"{area:.1f}".replace(".", ",") if st.marketing else name
            tw = d.textlength(text, font=font)
            if tw < (room[2] - room[0]) * st.px_per_m * 0.95:
                d.text((px_ - tw / 2, py_ - font.size), text, fill=int(rng.integers(30, 90)), font=font)
                if st.draw_areas and not st.marketing:
                    sub = f"{area:.1f}".replace(".", ",") + " м²"
                    sw = d.textlength(sub, font=font)
                    d.text((px_ - sw / 2, py_ + 2), sub, fill=int(rng.integers(50, 110)), font=font)

    # 6. Розмірні ланцюжки
    if st.draw_dims:
        xs = sorted({0.0, spec.width_m} | {round(s.const, 3) for s in spec.segs if not s.horizontal})
        ys = sorted({0.0, spec.height_m} | {round(s.const, 3) for s in spec.segs if s.horizontal})
        xs = [v for v in xs if -0.01 <= v <= spec.width_m + 0.01]
        ys = [v for v in ys if -0.01 <= v <= spec.height_m + 0.01]
        off = margin * 0.45
        _draw_dim_chain(canvas, rng, xs, 0.0, True, -off, st.dim_units)
        _draw_dim_chain(canvas, rng, ys, 0.0, False, -off, st.dim_units)
        if rng.random() < 0.5:
            _draw_dim_chain(canvas, rng, [0.0, spec.width_m], spec.height_m, True, off * 0.6, st.dim_units)

    if st.draw_room_badges:
        _draw_room_badges(canvas, rng, spec)
    if st.draw_title_block:
        _draw_title_block(canvas, rng)

    # 7. Рамка / штамп / позначка масштабу
    if st.draw_frame:
        d.rectangle([6, 6, w_px - 7, h_px - 7], outline=90, width=1)
    if rng.random() < 0.4:
        font = _font(int(max(10, st.px_per_m * 0.30)), rng)
        d.text((margin * 0.4, h_px - margin * 0.55),
               str(rng.choice(["М 1:100", "M 1:50", "Масштаб 1:100", "1:100"])),
               fill=60, font=font)

    return np.array(canvas.img, dtype=np.uint8), np.array(canvas.mask, dtype=np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
#  3. ДЕГРАДАЦІЇ (симуляція фото / сканів)
# ═════════════════════════════════════════════════════════════════════════════
def degrade(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
            out_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Чистий рендер → правдоподібне фото сторінки. Маска трансформується так само.

    Без цього кроку модель, натренована на ідеальних рендерах, розсипається на
    першому ж фото з телефона (перспектива + тінь + JPEG)."""
    import cv2  # локальний імпорт: тренувальна залежність, у прод не їде

    h, w = img.shape[:2]
    rgb = np.dstack([img, img, img]).astype(np.uint8)

    # 1. Перспектива + невеликий поворот
    if rng.random() < 0.85:
        jitter = float(rng.uniform(0.005, 0.055)) * max(w, h)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + rng.normal(0, jitter, src.shape).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        border = int(rng.integers(215, 255))
        rgb = cv2.warpPerspective(rgb, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(border, border, border))
        mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 2. Колір паперу + нерівномірне освітлення
    if rng.random() < 0.75:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        gx, gy = float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))
        grad = (xx / w) * gx + (yy / h) * gy
        grad = (grad - grad.min()) / max(1e-6, grad.max() - grad.min())
        strength = float(rng.uniform(0.05, 0.30))
        shade = (1.0 - strength * grad)[..., None]
        rgb = np.clip(rgb.astype(np.float32) * shade, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        tint = np.array([rng.uniform(0.93, 1.0), rng.uniform(0.95, 1.0),
                         rng.uniform(0.86, 1.0)], dtype=np.float32)
        rgb = np.clip(rgb.astype(np.float32) * tint, 0, 255).astype(np.uint8)

    # 3. Зернистість паперу
    if rng.random() < 0.7:
        noise = rng.normal(0, float(rng.uniform(2, 11)), rgb.shape).astype(np.float32)
        rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Розмиття / різкість
    if rng.random() < 0.55:
        k = int(rng.choice([3, 3, 5]))
        rgb = cv2.GaussianBlur(rgb, (k, k), 0)

    # 5. Контраст/яскравість
    alpha = float(rng.uniform(0.75, 1.35))
    beta = float(rng.uniform(-25, 25))
    rgb = np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 6. JPEG-артефакти
    if rng.random() < 0.6:
        q = int(rng.integers(35, 92))
        ok, enc = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            rgb = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # 7. Приведення до квадрата out_size×out_size з паддінгом (без спотворення пропорцій)
    scale = out_size / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    pad_v = out_size - nh
    pad_h = out_size - nw
    top, left = pad_v // 2, pad_h // 2
    pad_val = int(np.median(rgb[[0, -1], :, :])) if rgb.size else 245
    rgb = cv2.copyMakeBorder(rgb, top, pad_v - top, left, pad_h - left,
                             cv2.BORDER_CONSTANT, value=(pad_val, pad_val, pad_val))
    mask = cv2.copyMakeBorder(mask, top, pad_v - top, left, pad_h - left,
                              cv2.BORDER_CONSTANT, value=0)
    return rgb, mask


def finalize(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
             out_size: int = 512, clean: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Чистий рендер → навчальний приклад фіксованого розміру.

    Спільна для BSP-синтетики і для рендеру реальних планувань, щоб обидва
    джерела проходили ІДЕНТИЧНУ обробку — інакше модель вивчить не планування,
    а те, з якого генератора прийшов приклад."""
    if not clean:
        return degrade(img, mask, rng, out_size=out_size)

    import cv2

    rgb = np.dstack([img] * 3) if img.ndim == 2 else img
    scale = out_size / max(img.shape[:2])
    nw, nh = max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    pad_v, pad_h = out_size - nh, out_size - nw
    rgb = cv2.copyMakeBorder(rgb, pad_v // 2, pad_v - pad_v // 2, pad_h // 2,
                             pad_h - pad_h // 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    mask = cv2.copyMakeBorder(mask, pad_v // 2, pad_v - pad_v // 2, pad_h // 2,
                              pad_h - pad_h // 2, cv2.BORDER_CONSTANT, value=0)
    return rgb, mask


def make_sample(seed: int, out_size: int = 512, clean: bool = False
                ) -> Tuple[np.ndarray, np.ndarray, LayoutSpec]:
    """Один навчальний приклад: (RGB out_size², маска out_size², розкладка)."""
    rng = np.random.default_rng(seed)
    spec = generate_layout(rng)
    img, mask = render_layout(spec, rng)
    out_img, out_mask = finalize(img, mask, rng, out_size=out_size, clean=clean)
    return out_img, out_mask, spec


def layout_to_plan(spec: LayoutSpec):
    """LayoutSpec → PlanVector: ідеальна «правильна відповідь» у метрах.

    Використовується двічі: як еталон для метрик векторизатора і як фікстура
    геометричних тестів (будуємо меш із плану, який точно коректний)."""
    from services.floorplan.plan_model import (
        DEFAULT_DOOR_HEIGHT_M, DEFAULT_WALL_HEIGHT_M, DEFAULT_WINDOW_HEIGHT_M,
        DEFAULT_WINDOW_SILL_M, Opening, PlanVector, Room, Wall,
    )

    walls: List[Wall] = []
    for s in spec.segs:
        (x1, y1), (x2, y2) = s.endpoints()
        walls.append(Wall(x1=x1, y1=y1, x2=x2, y2=y2,
                          thickness_m=s.thickness, bearing=s.bearing))
    openings: List[Opening] = []
    for op in spec.openings:
        seg = spec.segs[op.seg_index]
        if seg.length <= 1e-6:
            continue
        t = (op.center - seg.a) / seg.length
        if op.kind == "window":
            openings.append(Opening(op.seg_index, t, op.width, "window",
                                    DEFAULT_WINDOW_SILL_M, DEFAULT_WINDOW_HEIGHT_M))
        else:
            openings.append(Opening(op.seg_index, t, op.width, "door",
                                    0.0, DEFAULT_DOOR_HEIGHT_M))
    rooms = [
        Room(polygon=[(r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])],
             name=name, area_m2=(r[2] - r[0]) * (r[3] - r[1]))
        for r, name in zip(spec.rooms, spec.room_names)
    ]
    return PlanVector(walls=walls, openings=openings, rooms=rooms,
                      wall_height_m=DEFAULT_WALL_HEIGHT_M, scale_source="synthetic",
                      confidence=1.0)


if __name__ == "__main__":  # ручний перегляд: python -m ml.floorplan.synth
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_preview")
    os.makedirs(out_dir, exist_ok=True)
    palette = np.array([[255, 255, 255], [40, 40, 40], [220, 60, 60], [60, 130, 220]], np.uint8)
    for i in range(n):
        rgb, mask, spec = make_sample(1000 + i)
        Image.fromarray(rgb).save(os.path.join(out_dir, f"s{i:02d}_img.png"))
        Image.fromarray(palette[mask]).save(os.path.join(out_dir, f"s{i:02d}_mask.png"))
        print(f"[{i}] {spec.width_m:.1f}x{spec.height_m:.1f} м, "
              f"{len(spec.rooms)} кімнат, {len(spec.segs)} стін, {len(spec.openings)} отворів")
    print("saved to:", out_dir)
