"""Спільна модель векторного плану приміщення.

Одна структура даних курсує через увесь сервіс «макет квартири»:

    зображення → детекція → PlanVector → (редактор користувача) → меш → 3MF

Тому вона навмисно проста й серіалізовна в JSON: фронтенд-редактор віддає
рівно те саме, що згенерував бекенд, лише з правками користувача.

Одиниці: **метри** (реальні розміри приміщення). Піксельні координати живуть
лише всередині детектора; на межі PlanVector вони вже помножені на m_per_px.
Осі: X — вправо, Y — ВНИЗ (як у зображенні). Це навмисно: так фронтенд-канвас
і бекенд говорять однією системою координат і ніхто нічого не дзеркалить.
При побудові мешу Y інвертується один раз (див. builder.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Point = Tuple[float, float]

# ── Типові значення для житлової забудови (метри) ────────────────────────────
DEFAULT_WALL_THICKNESS_M = 0.12      # міжкімнатна перегородка
DEFAULT_BEARING_THICKNESS_M = 0.38   # несуча / зовнішня стіна
DEFAULT_WALL_HEIGHT_M = 2.7          # висота стелі
DEFAULT_DOOR_WIDTH_M = 0.85
DEFAULT_DOOR_HEIGHT_M = 2.10
DEFAULT_WINDOW_WIDTH_M = 1.40
DEFAULT_WINDOW_SILL_M = 0.85
DEFAULT_WINDOW_HEIGHT_M = 1.45


def _f(value: Any, fallback: float = 0.0) -> float:
    """float() який не падає на None/''/NaN — вхід приходить із JSON фронтенда."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(out):
        return fallback
    return out


@dataclass
class Wall:
    """Відрізок стіни, заданий ЦЕНТРАЛЬНОЮ лінією та товщиною.

    Центральна лінія (а не два контури) — бо саме її редагує користувач, і саме
    з неї shapely.buffer() дає коректні стики на перехрестях.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    thickness_m: float = DEFAULT_WALL_THICKNESS_M
    bearing: bool = False
    height_m: Optional[float] = None  # None → загальна висота плану

    # ── геометрія ────────────────────────────────────────────────────────────
    @property
    def p1(self) -> Point:
        return (self.x1, self.y1)

    @property
    def p2(self) -> Point:
        return (self.x2, self.y2)

    @property
    def length_m(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def angle_rad(self) -> float:
        return math.atan2(self.y2 - self.y1, self.x2 - self.x1)

    def unit(self) -> Point:
        length = self.length_m
        if length < 1e-9:
            return (1.0, 0.0)
        return ((self.x2 - self.x1) / length, (self.y2 - self.y1) / length)

    def point_at(self, t: float) -> Point:
        """Точка на центральній лінії, t ∈ [0, 1]."""
        return (self.x1 + (self.x2 - self.x1) * t, self.y1 + (self.y2 - self.y1) * t)

    def is_degenerate(self, min_len_m: float = 0.02) -> bool:
        return self.length_m < min_len_m or self.thickness_m <= 1e-4

    # ── серіалізація ─────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "x1": round(self.x1, 5),
            "y1": round(self.y1, 5),
            "x2": round(self.x2, 5),
            "y2": round(self.y2, 5),
            "thickness_m": round(self.thickness_m, 5),
            "bearing": bool(self.bearing),
        }
        if self.height_m is not None:
            out["height_m"] = round(self.height_m, 5)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Wall":
        height = data.get("height_m")
        return cls(
            x1=_f(data.get("x1")),
            y1=_f(data.get("y1")),
            x2=_f(data.get("x2")),
            y2=_f(data.get("y2")),
            thickness_m=max(0.01, _f(data.get("thickness_m"), DEFAULT_WALL_THICKNESS_M)),
            bearing=bool(data.get("bearing", False)),
            height_m=None if height in (None, "") else max(0.05, _f(height, DEFAULT_WALL_HEIGHT_M)),
        )


@dataclass
class Opening:
    """Отвір у стіні: двері, вікно або арка.

    Прив'язаний до індексу стіни + позиції вздовж неї (а не до абсолютних
    координат), щоб перетягування стіни в редакторі тягнуло отвір за собою.
    """

    wall: int
    center_t: float                 # 0..1 вздовж центральної лінії стіни
    width_m: float = DEFAULT_DOOR_WIDTH_M
    kind: str = "door"              # door | window | arch
    sill_m: float = 0.0             # низ отвору від підлоги
    height_m: float = DEFAULT_DOOR_HEIGHT_M

    def clamp(self, wall_len_m: float) -> "Opening":
        """Не даємо отвору вилізти за краї стіни (після правок користувача)."""
        width = max(0.05, min(self.width_m, max(0.05, wall_len_m - 0.02)))
        half = (width / 2.0) / max(wall_len_m, 1e-6)
        self.width_m = width
        self.center_t = min(max(self.center_t, half), 1.0 - half)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wall": int(self.wall),
            "center_t": round(self.center_t, 5),
            "width_m": round(self.width_m, 5),
            "kind": self.kind,
            "sill_m": round(self.sill_m, 5),
            "height_m": round(self.height_m, 5),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Opening":
        kind = str(data.get("kind", "door")).lower()
        if kind not in ("door", "window", "arch"):
            kind = "door"
        default_sill = DEFAULT_WINDOW_SILL_M if kind == "window" else 0.0
        default_h = DEFAULT_WINDOW_HEIGHT_M if kind == "window" else DEFAULT_DOOR_HEIGHT_M
        default_w = DEFAULT_WINDOW_WIDTH_M if kind == "window" else DEFAULT_DOOR_WIDTH_M
        return cls(
            wall=int(_f(data.get("wall"), 0)),
            center_t=min(max(_f(data.get("center_t"), 0.5), 0.0), 1.0),
            width_m=max(0.05, _f(data.get("width_m"), default_w)),
            kind=kind,
            sill_m=max(0.0, _f(data.get("sill_m"), default_sill)),
            height_m=max(0.05, _f(data.get("height_m"), default_h)),
        )


@dataclass
class Room:
    """Кімната — лише для підписів/підлоги; на друк впливає опційно."""

    polygon: List[Point] = field(default_factory=list)
    name: str = ""
    area_m2: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "polygon": [[round(x, 4), round(y, 4)] for x, y in self.polygon],
            "name": self.name,
            "area_m2": round(self.area_m2, 3),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Room":
        poly = [(_f(p[0]), _f(p[1])) for p in (data.get("polygon") or []) if len(p) >= 2]
        return cls(polygon=poly, name=str(data.get("name", "")), area_m2=_f(data.get("area_m2")))


@dataclass
class PlanVector:
    """Повний векторний план у метрах — контракт між детектором, редактором і мешем."""

    walls: List[Wall] = field(default_factory=list)
    openings: List[Opening] = field(default_factory=list)
    rooms: List[Room] = field(default_factory=list)
    wall_height_m: float = DEFAULT_WALL_HEIGHT_M
    # Звідки взявся масштаб — показуємо користувачу, щоб він міг не довіряти
    # автоматиці: "ocr" | "reference" | "door" | "assumed" | "manual"
    scale_source: str = "assumed"
    m_per_px: float = 0.0            # масштаб вихідного зображення (діагностика)
    image_size_px: Optional[Tuple[int, int]] = None
    confidence: float = 0.0          # 0..1 впевненість автодетекції
    notes: List[str] = field(default_factory=list)

    # ── похідні величини ─────────────────────────────────────────────────────
    def bounds(self) -> Tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) з урахуванням товщини стін."""
        if not self.walls:
            return (0.0, 0.0, 0.0, 0.0)
        xs: List[float] = []
        ys: List[float] = []
        for w in self.walls:
            half = w.thickness_m / 2.0
            xs.extend([w.x1 - half, w.x1 + half, w.x2 - half, w.x2 + half])
            ys.extend([w.y1 - half, w.y1 + half, w.y2 - half, w.y2 + half])
        return (min(xs), min(ys), max(xs), max(ys))

    def size_m(self) -> Tuple[float, float]:
        minx, miny, maxx, maxy = self.bounds()
        return (maxx - minx, maxy - miny)

    def total_wall_length_m(self) -> float:
        return sum(w.length_m for w in self.walls)

    def sanitize(self) -> "PlanVector":
        """Прибирає вироджені стіни й отвори-сироти. Викликати ПІСЛЯ from_dict:
        редактор може віддати стіну нульової довжини або отвір на видаленій стіні."""
        keep: List[Wall] = []
        remap: Dict[int, int] = {}
        for idx, wall in enumerate(self.walls):
            if wall.is_degenerate():
                continue
            remap[idx] = len(keep)
            keep.append(wall)
        self.walls = keep

        fixed: List[Opening] = []
        for op in self.openings:
            new_idx = remap.get(op.wall)
            if new_idx is None:
                continue
            op.wall = new_idx
            fixed.append(op.clamp(self.walls[new_idx].length_m))
        self.openings = fixed
        return self

    # ── серіалізація ─────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "walls": [w.to_dict() for w in self.walls],
            "openings": [o.to_dict() for o in self.openings],
            "rooms": [r.to_dict() for r in self.rooms],
            "wall_height_m": round(self.wall_height_m, 4),
            "scale_source": self.scale_source,
            "m_per_px": self.m_per_px,
            "image_size_px": list(self.image_size_px) if self.image_size_px else None,
            "confidence": round(self.confidence, 3),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanVector":
        size = data.get("image_size_px")
        return cls(
            walls=[Wall.from_dict(w) for w in (data.get("walls") or [])],
            openings=[Opening.from_dict(o) for o in (data.get("openings") or [])],
            rooms=[Room.from_dict(r) for r in (data.get("rooms") or [])],
            wall_height_m=max(0.3, _f(data.get("wall_height_m"), DEFAULT_WALL_HEIGHT_M)),
            scale_source=str(data.get("scale_source", "manual")),
            m_per_px=_f(data.get("m_per_px")),
            image_size_px=(int(size[0]), int(size[1])) if size and len(size) >= 2 else None,
            confidence=_f(data.get("confidence")),
            notes=[str(n) for n in (data.get("notes") or [])],
        )

    @classmethod
    def from_pixel_dict(cls, data: Dict[str, Any], m_per_px: float) -> "PlanVector":
        """Варіант from_dict для редактора, який рахує в ПІКСЕЛЯХ зображення.

        Фронтенд малює поверх картинки, тож йому природно віддавати піксельні
        координати + один масштаб. Ми множимо тут — щоб конвертація жила в
        рівно одному місці й не роз'їхалась між клієнтом і сервером."""
        scale = max(1e-9, float(m_per_px))
        plan = cls.from_dict(data)
        for wall in plan.walls:
            wall.x1 *= scale
            wall.y1 *= scale
            wall.x2 *= scale
            wall.y2 *= scale
            wall.thickness_m *= scale
        for op in plan.openings:
            op.width_m *= scale
        for room in plan.rooms:
            room.polygon = [(x * scale, y * scale) for x, y in room.polygon]
            room.area_m2 *= scale * scale
        plan.m_per_px = scale
        return plan.sanitize()
