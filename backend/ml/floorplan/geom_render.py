"""Рендер РЕАЛЬНОЇ геометрії планів у наших кресленнєвих стилях.

Навіщо окремо від synth.py: там розкладка своя, BSP, і стіни строго осе-
паралельні прямокутники. Реальні квартири так не виглядають — там косі стіни,
еркери, ніші, кімнати неправильної форми. Модель, натренована лише на BSP,
вчиться на топології, якої в житті не буває.

Ідея: беремо СПРАВЖНЮ геометрію з відкритих датасетів (Swiss Dwellings — метричні
полігони стін і прорізів, CC BY 4.0, комерційне використання дозволене), а
малюємо її НАШИМИ конвенціями — штриховка несучих, розмірні ланцюжки в
міліметрах, кирилиця, осі з кружечками, штамп. Так модель бачить справжні
планування в тому оформленні, у якому їх принесуть наші користувачі.

Класи маски ті самі, що в synth: 0 фон, 1 стіна, 2 двері, 3 вікно.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ml.floorplan.synth import (
    CLASS_DOOR, CLASS_WALL, CLASS_WINDOW, RenderStyle, _Canvas, _draw_hatch,
    _draw_room_badges, _draw_title_block, _font, _rand_style, _ROOM_SETS,
)

Point = Tuple[float, float]


@dataclass
class RealLayout:
    """Одна квартира/поверх у МЕТРАХ. Полігони — списки точок (x, y)."""

    walls: List[List[Point]] = field(default_factory=list)
    doors: List[List[Point]] = field(default_factory=list)
    windows: List[List[Point]] = field(default_factory=list)
    rooms: List[List[Point]] = field(default_factory=list)
    room_names: List[str] = field(default_factory=list)
    source: str = ""

    def bounds(self) -> Tuple[float, float, float, float]:
        xs = [p[0] for poly in (self.walls or self.rooms) for p in poly]
        ys = [p[1] for poly in (self.walls or self.rooms) for p in poly]
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))

    def size_m(self) -> Tuple[float, float]:
        minx, miny, maxx, maxy = self.bounds()
        return (maxx - minx, maxy - miny)

    def normalized(self) -> "RealLayout":
        """Зсуває план у початок координат — далі рендер рахує від (0, 0)."""
        minx, miny, _, _ = self.bounds()
        shift = lambda polys: [[(x - minx, y - miny) for x, y in poly] for poly in polys]
        return RealLayout(
            walls=shift(self.walls), doors=shift(self.doors), windows=shift(self.windows),
            rooms=shift(self.rooms), room_names=list(self.room_names), source=self.source,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  Полігон → центральна лінія + товщина (для еталонного PlanVector)
# ═════════════════════════════════════════════════════════════════════════════
def polygon_to_wall(polygon: Sequence[Point]) -> Optional[Tuple[Point, Point, float]]:
    """Полігон стіни → (кінець, кінець, товщина) у метрах.

    Реальна стіна — це витягнутий чотирикутник; беремо мінімальний охопний
    прямокутник, довша сторона дає напрям і довжину, коротша — товщину.
    Так само працює для косих стін, на відміну від осе-паралельної логіки."""
    from shapely.geometry import Polygon

    try:
        poly = Polygon(polygon)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 1e-6:
            return None
        rect = poly.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)[:4]
        if len(coords) < 4:
            return None
    except Exception:
        return None

    edges = [(coords[i], coords[(i + 1) % 4]) for i in range(4)]
    lengths = [math.dist(a, b) for a, b in edges]
    long_i = int(np.argmax(lengths))
    thickness = min(lengths[(long_i + 1) % 4], lengths[(long_i + 3) % 4])
    if thickness < 1e-4 or lengths[long_i] < 1e-4:
        return None
    # середня лінія = середина довгої сторони та протилежної їй
    a1, a2 = edges[long_i]
    b1, b2 = edges[(long_i + 2) % 4]
    start = ((a1[0] + b2[0]) / 2.0, (a1[1] + b2[1]) / 2.0)
    end = ((a2[0] + b1[0]) / 2.0, (a2[1] + b1[1]) / 2.0)
    return (start, end, float(thickness))


def layout_to_plan(layout: RealLayout):
    """RealLayout → еталонний PlanVector у метрах (для метрик і тестів)."""
    from services.floorplan.plan_model import (
        DEFAULT_DOOR_HEIGHT_M, DEFAULT_WALL_HEIGHT_M, DEFAULT_WINDOW_HEIGHT_M,
        DEFAULT_WINDOW_SILL_M, Opening, PlanVector, Room, Wall,
    )

    walls: List[Wall] = []
    for polygon in layout.walls:
        converted = polygon_to_wall(polygon)
        if converted is None:
            continue
        (x1, y1), (x2, y2), thickness = converted
        walls.append(Wall(x1=x1, y1=y1, x2=x2, y2=y2, thickness_m=thickness,
                          bearing=thickness > 0.2))

    openings: List[Opening] = []
    for kind, polygons in (("door", layout.doors), ("window", layout.windows)):
        for polygon in polygons:
            converted = polygon_to_wall(polygon)
            if converted is None:
                continue
            (ax, ay), (bx, by), _t = converted
            cx, cy = (ax + bx) / 2.0, (ay + by) / 2.0
            width = math.dist((ax, ay), (bx, by))
            best_i, best_d, best_t = -1, 1e18, 0.5
            for i, wall in enumerate(walls):
                vx, vy = wall.x2 - wall.x1, wall.y2 - wall.y1
                length_sq = vx * vx + vy * vy
                if length_sq < 1e-9:
                    continue
                t = ((cx - wall.x1) * vx + (cy - wall.y1) * vy) / length_sq
                t = min(max(t, 0.0), 1.0)
                dist = math.dist((cx, cy), (wall.x1 + vx * t, wall.y1 + vy * t))
                if dist < best_d:
                    best_d, best_i, best_t = dist, i, t
            if best_i < 0 or best_d > 1.0:
                continue
            sill = DEFAULT_WINDOW_SILL_M if kind == "window" else 0.0
            height = DEFAULT_WINDOW_HEIGHT_M if kind == "window" else DEFAULT_DOOR_HEIGHT_M
            openings.append(
                Opening(best_i, best_t, width, kind, sill, height)
                .clamp(walls[best_i].length_m)
            )

    rooms = [
        Room(polygon=list(poly), name=name, area_m2=_polygon_area(poly))
        for poly, name in zip(layout.rooms, layout.room_names or [""] * len(layout.rooms))
    ]
    return PlanVector(walls=walls, openings=openings, rooms=rooms,
                      wall_height_m=DEFAULT_WALL_HEIGHT_M, scale_source="dataset",
                      confidence=1.0, notes=[layout.source])


def _polygon_area(polygon: Sequence[Point]) -> float:
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


# ═════════════════════════════════════════════════════════════════════════════
#  Рендер
# ═════════════════════════════════════════════════════════════════════════════
def _poly_px(canvas: _Canvas, polygon: Sequence[Point]) -> List[Tuple[float, float]]:
    return [canvas.px(x, y) for x, y in polygon]


def _bbox_px(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _draw_dim_chain_free(canvas: _Canvas, rng: np.random.Generator, values: Sequence[float],
                         horizontal: bool, offset_px: float, units: str) -> None:
    """Розмірний ланцюжок уздовж габариту (для довільної геометрії)."""
    from ml.floorplan.synth import _fmt_dim

    if len(values) < 2:
        return
    d = canvas.d
    font = _font(int(max(9, canvas.style.px_per_m * 0.28)), rng)
    tick = 5
    for i in range(len(values) - 1):
        a, b = values[i], values[i + 1]
        if b - a < 0.35:
            continue
        text = _fmt_dim(b - a, units)
        if horizontal:
            (xa, y0) = canvas.px(a, 0.0)
            (xb, _) = canvas.px(b, 0.0)
            y = y0 + offset_px
            d.line([(xa, y), (xb, y)], fill=60, width=1)
            d.line([(xa, y - tick), (xa, y + tick)], fill=60, width=1)
            d.line([(xb, y - tick), (xb, y + tick)], fill=60, width=1)
            width_text = d.textlength(text, font=font)
            d.text(((xa + xb) / 2 - width_text / 2, y - tick - 3 - font.size), text,
                   fill=40, font=font)
        else:
            (x0, ya) = canvas.px(0.0, a)
            (_, yb) = canvas.px(0.0, b)
            x = x0 + offset_px
            d.line([(x, ya), (x, yb)], fill=60, width=1)
            d.line([(x - tick, ya), (x + tick, ya)], fill=60, width=1)
            d.line([(x - tick, yb), (x + tick, yb)], fill=60, width=1)
            tmp = Image.new("L", (int(d.textlength(text, font=font)) + 4, font.size + 4), 255)
            ImageDraw.Draw(tmp).text((2, 2), text, fill=40, font=font)
            tmp = tmp.rotate(90, expand=True)
            canvas.img.paste(tmp, (int(x - tick - 2 - tmp.width),
                                   int((ya + yb) / 2 - tmp.height / 2)))


def render_layout(layout: RealLayout, rng: np.random.Generator,
                  style: Optional[RenderStyle] = None) -> Tuple[np.ndarray, np.ndarray]:
    """RealLayout → (grayscale uint8, маска класів uint8) у наших стилях."""
    layout = layout.normalized()
    st = style or _rand_style(rng)
    width_m, height_m = layout.size_m()
    if width_m <= 0.5 or height_m <= 0.5:
        raise ValueError("порожня геометрія")

    margin = st.margin_px
    w_px = int(width_m * st.px_per_m) + 2 * margin
    h_px = int(height_m * st.px_per_m) + 2 * margin
    if max(w_px, h_px) > 4000:                       # захист від велетенських поверхів
        st.px_per_m *= 3500.0 / max(w_px, h_px)
        w_px = int(width_m * st.px_per_m) + 2 * margin
        h_px = int(height_m * st.px_per_m) + 2 * margin
    canvas = _Canvas(w_px, h_px, st, margin, margin)
    d, dm = canvas.d, canvas.dm

    # 1. Підлога кімнат — легкий фон
    if st.draw_furniture and rng.random() < 0.35:
        shade = int(rng.integers(238, 252))
        for room in layout.rooms:
            if len(room) >= 3:
                d.polygon(_poly_px(canvas, room), fill=shade)

    # 2. Стіни: заливка + БЕЗПЕРЕРВНА маска
    for polygon in layout.walls:
        if len(polygon) < 3:
            continue
        pts = _poly_px(canvas, polygon)
        dm.polygon(pts, fill=CLASS_WALL)
        box = _bbox_px(pts)
        thin = (box[2] - box[0]) < 12 and (box[3] - box[1]) < 12
        # «Порожній» стиль лише для товстих стін — тонка перегородка, намальована
        # контуром, вироджується у волосяну лінію (перевірено, див. synth).
        thickness_px = min(box[2] - box[0], box[3] - box[1])
        if st.hollow_walls and thickness_px >= 12 and not thin:
            d.polygon(pts, fill=255, outline=st.wall_fill)
            d.line(pts + [pts[0]], fill=st.wall_fill, width=max(1, st.line_width - 1))
        else:
            d.polygon(pts, fill=max(st.wall_fill, st.wall_gray)
                      if thickness_px >= 14 else st.wall_fill)
        if st.draw_hatch and thickness_px >= 14 and not st.hollow_walls and rng.random() < 0.7:
            mask_layer = Image.new("L", canvas.img.size, 0)
            ImageDraw.Draw(mask_layer).polygon(pts, fill=255)
            hatch = Image.new("L", canvas.img.size, 255)
            hd = ImageDraw.Draw(hatch)
            step = int(rng.integers(4, 9))
            shade = int(rng.integers(90, 160))
            for k in range(-canvas.img.height, canvas.img.width + canvas.img.height, step):
                if st.hatch_angle_alt:
                    hd.line([(k, canvas.img.height), (k + canvas.img.height, 0)],
                            fill=shade, width=1)
                else:
                    hd.line([(k, 0), (k + canvas.img.height, canvas.img.height)],
                            fill=shade, width=1)
            canvas.img.paste(hatch, (0, 0), mask_layer)
            d.line(pts + [pts[0]], fill=st.wall_fill, width=1)

    # 3. Отвори
    for kind, polygons, cls in (("door", layout.doors, CLASS_DOOR),
                                ("window", layout.windows, CLASS_WINDOW)):
        for polygon in polygons:
            if len(polygon) < 3:
                continue
            pts = _poly_px(canvas, polygon)
            d.polygon(pts, fill=255)
            dm.polygon(pts, fill=cls)
            x0, y0, x1, y1 = _bbox_px(pts)
            if kind == "window":
                lines = int(rng.integers(2, 4))
                if (x1 - x0) >= (y1 - y0):
                    for k in range(lines):
                        yy = y0 + (y1 - y0) * (k + 1) / (lines + 1)
                        d.line([(x0, yy), (x1, yy)], fill=st.wall_fill, width=1)
                else:
                    for k in range(lines):
                        xx = x0 + (x1 - x0) * (k + 1) / (lines + 1)
                        d.line([(xx, y0), (xx, y1)], fill=st.wall_fill, width=1)
                d.line(pts + [pts[0]], fill=st.wall_fill, width=1)
            else:
                span = max(x1 - x0, y1 - y0)
                hx, hy = (x0, (y0 + y1) / 2) if (x1 - x0) >= (y1 - y0) else ((x0 + x1) / 2, y0)
                sgn = 1 if rng.random() < 0.5 else -1
                d.arc([hx - span, hy - span, hx + span, hy + span],
                      start=0 if sgn > 0 else 180, end=90 if sgn > 0 else 270,
                      fill=int(min(255, st.wall_fill + 90)), width=1)

    # 4. Підписи кімнат
    if st.draw_labels and layout.rooms:
        names_pool = _ROOM_SETS[int(rng.integers(len(_ROOM_SETS)))]
        font = _font(int(max(9, st.px_per_m * 0.32)), rng)
        for i, room in enumerate(layout.rooms):
            if len(room) < 3:
                continue
            pts = _poly_px(canvas, room)
            x0, y0, x1, y1 = _bbox_px(pts)
            area = _polygon_area(room)
            if area < 2.0:
                continue
            name = (layout.room_names[i] if i < len(layout.room_names) and layout.room_names[i]
                    else str(rng.choice(names_pool)))
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            text_width = d.textlength(name, font=font)
            if text_width < (x1 - x0) * 0.95:
                d.text((cx - text_width / 2, cy - font.size), name,
                       fill=int(rng.integers(30, 90)), font=font)
                if st.draw_areas:
                    sub = f"{area:.1f}".replace(".", ",") + " м²"
                    sub_width = d.textlength(sub, font=font)
                    d.text((cx - sub_width / 2, cy + 2), sub,
                           fill=int(rng.integers(50, 110)), font=font)

    # 5. Розмірні ланцюжки — по проєкціях стін на осі
    if st.draw_dims:
        xs = sorted({round(p[0], 2) for poly in layout.walls for p in poly} | {0.0, width_m})
        ys = sorted({round(p[1], 2) for poly in layout.walls for p in poly} | {0.0, height_m})
        # Прорідження щільніше: у реальної геометрії координат сотні, і без
        # цього розмірний ланцюжок перетворюється на суцільну кашу з цифр.
        xs = _thin_out(xs, max(0.9, width_m / 8))[:8]
        ys = _thin_out(ys, max(0.9, height_m / 8))[:8]
        offset = margin * 0.45
        _draw_dim_chain_free(canvas, rng, xs, True, -offset, st.dim_units)
        _draw_dim_chain_free(canvas, rng, ys, False, -offset, st.dim_units)

    if st.draw_title_block:
        _draw_title_block(canvas, rng)
    if st.draw_frame:
        d.rectangle([6, 6, w_px - 7, h_px - 7], outline=90, width=1)
    if rng.random() < 0.4:
        font = _font(int(max(10, st.px_per_m * 0.30)), rng)
        d.text((margin * 0.4, h_px - margin * 0.55),
               str(rng.choice(["М 1:100", "M 1:50", "Масштаб 1:100", "1:100"])),
               fill=60, font=font)

    return np.array(canvas.img, dtype=np.uint8), np.array(canvas.mask, dtype=np.uint8)


def _thin_out(values: Sequence[float], min_gap: float) -> List[float]:
    """Прорідження координат — інакше розмірний ланцюжок перетворюється на кашу."""
    out: List[float] = []
    for v in values:
        if not out or v - out[-1] >= min_gap:
            out.append(v)
    return out
