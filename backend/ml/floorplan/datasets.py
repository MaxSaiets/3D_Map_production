"""Реальні планування з відкритих датасетів → RealLayout.

ЛІЦЕНЗІЇ (сервіс комерційний, це не формальність):
  • Swiss Dwellings v3.0.0 — CC BY 4.0, комерційне використання ДОЗВОЛЕНЕ.
    Zenodo 10.5281/zenodo.7788422. ~45k швейцарських квартир, метричні полігони.
  • CubiCasa5K, RPLAN, Structured3D, ZInD — CC BY-NC / research-only.
    НЕ використовуємо для навчання. Навіть як «просто подивитись» — ваги, що
    з них виросли, вже похідна робота.

Навіщо реальні дані, якщо є свій генератор: BSP-розкладка дає лише прямокутні
кімнати й строго осе-паралельні стіни. У житті є еркери, скоси, ніші, коридори
складної форми — і модель, яка їх не бачила, на них губиться. Беремо справжню
ТОПОЛОГІЮ, а малюємо її своїми конвенціями (див. geom_render).

Використання:
    python -m ml.floorplan.datasets build --limit 6000
    python -m ml.floorplan.datasets stats
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import sys
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "_datasets", "swiss_layouts.pkl")
SWISS_CSV_CANDIDATES = [
    r"D:\floorplan_datasets\swiss-dwellings-v3.0.0\geometries.csv",
    os.path.join(HERE, "_datasets", "geometries.csv"),
]

WALL_SUBTYPES = {"WALL"}                     # RAILING/COLUMN — не стіни макета
DOOR_SUBTYPES = {"DOOR", "ENTRANCE_DOOR"}
WINDOW_SUBTYPES = {"WINDOW"}
# Технічні шахти й лоджії лишаємо як кімнати: на плані вони теж намальовані.
SKIP_AREAS = {"VOID"}

# Англійські підтипи датасету → підписи, які реально пишуть на наших планах.
ROOM_NAME_UK = {
    "ROOM": "Кімната", "BEDROOM": "Спальня", "LIVING_ROOM": "Вітальня",
    "LIVING_DINING": "Вітальня-їдальня", "KITCHEN": "Кухня", "KITCHEN_DINING": "Кухня-їдальня",
    "BATHROOM": "Санвузол", "CORRIDOR": "Коридор", "BALCONY": "Балкон", "LOGGIA": "Лоджія",
    "STOREROOM": "Комора", "STAIRCASE": "Сходи", "SHAFT": "Шахта", "ELEVATOR": "Ліфт",
    "DINING": "Їдальня", "OFFICE": "Кабінет", "WINTERGARTEN": "Зимовий сад",
    "TERRACE": "Тераса", "GARAGE": "Гараж", "BASEMENT": "Підвал",
}

MIN_AREA_M2 = 18.0
MAX_AREA_M2 = 260.0
MIN_WALLS = 4
MIN_ROOMS = 2


def _swiss_csv_path(explicit: Optional[str] = None) -> Optional[str]:
    for path in ([explicit] if explicit else []) + SWISS_CSV_CANDIDATES:
        if path and os.path.exists(path):
            return path
    return None


def _parse_polygon(wkt: str) -> Optional[List[Tuple[float, float]]]:
    """Мінімальний парсер WKT POLYGON — shapely тут зайвий і повільніший.

    Беремо ЛИШЕ зовнішнє кільце: дірки в полігоні стіни для нас нецікаві."""
    if not wkt.startswith("POLYGON"):
        return None
    start = wkt.find("((")
    end = wkt.find(")", start + 2)
    if start < 0 or end < 0:
        return None
    points: List[Tuple[float, float]] = []
    for chunk in wkt[start + 2:end].split(","):
        parts = chunk.split()
        if len(parts) < 2:
            continue
        try:
            points.append((float(parts[0]), float(parts[1])))
        except ValueError:
            return None
    if len(points) > 2 and points[0] == points[-1]:
        points.pop()
    return points if len(points) >= 3 else None


def _polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _dominant_angle(walls: Sequence[Sequence[Tuple[float, float]]]) -> float:
    """Головний напрям стін (rad). Реальні координати — у СК будівлі, тобто
    план може стояти під довільним кутом; креслення ж завжди рівне."""
    acc_sin = acc_cos = 0.0
    for polygon in walls:
        for i in range(len(polygon)):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % len(polygon)]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 0.3:
                continue
            angle = math.atan2(y2 - y1, x2 - x1) % (math.pi / 2)
            acc_sin += length * math.sin(4 * angle)
            acc_cos += length * math.cos(4 * angle)
    if abs(acc_sin) < 1e-9 and abs(acc_cos) < 1e-9:
        return 0.0
    return math.atan2(acc_sin, acc_cos) / 4.0


def _rotate(points: Sequence[Tuple[float, float]], angle: float
            ) -> List[Tuple[float, float]]:
    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    return [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in points]


def _bbox(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_overlap(a: Tuple[float, float, float, float],
                  b: Tuple[float, float, float, float], pad: float = 0.0) -> bool:
    return not (a[2] + pad < b[0] or b[2] < a[0] - pad
                or a[3] + pad < b[1] or b[3] < a[1] - pad)


def iter_swiss_apartments(csv_path: str, limit: int = 6000,
                          progress_every: int = 500) -> Iterator[Dict[str, Any]]:
    """Стрімить geometries.csv і віддає квартири по одній.

    ГРУПУЄМО ЗА ПОВЕРХОМ (plan_id), а не за квартирою. У цьому датасеті стіна
    належить поверху: вона розділяє дві квартири й записана один раз. Якщо
    брати лише те, що позначене цією квартирою, зовнішній контур зникає — план
    виходить дірявим, з висячими шматками стін (перевірено на рендері).
    Тому: збираємо ВСІ стіни поверху, а потім для кожної квартири лишаємо ті,
    що потрапляють у її габарит із запасом.

    Файл 1.1 ГБ — читаємо потоково і зупиняємось, щойно набрали ліміт."""
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    current_plan: Optional[str] = None
    plan_walls: List[List[Tuple[float, float]]] = []
    plan_doors: List[List[Tuple[float, float]]] = []
    plan_windows: List[List[Tuple[float, float]]] = []
    plan_features: List[List[Tuple[float, float]]] = []
    apartments: Dict[str, Dict[str, List]] = {}
    produced = 0

    def _emit_plan() -> Iterator[Dict[str, Any]]:
        nonlocal produced
        wall_boxes = [_bbox(p) for p in plan_walls]
        door_boxes = [_bbox(p) for p in plan_doors]
        window_boxes = [_bbox(p) for p in plan_windows]
        feature_boxes = [_bbox(p) for p in plan_features]
        for apartment_id, data in apartments.items():
            if len(data["rooms"]) < MIN_ROOMS:
                continue
            points = [pt for room in data["rooms"] for pt in room]
            if not points:
                continue
            box = _bbox(points)
            pick = lambda polys, boxes, pad: [
                poly for poly, poly_box in zip(polys, boxes)
                if _bbox_overlap(poly_box, box, pad)
            ]
            yield {
                "apartment_id": apartment_id,
                "walls": pick(plan_walls, wall_boxes, 0.6),
                "doors": pick(plan_doors, door_boxes, 0.6),
                "windows": pick(plan_windows, window_boxes, 0.6),
                "features": pick(plan_features, feature_boxes, 0.0),
                "rooms": data["rooms"],
                "room_names": data["room_names"],
            }
            produced += 1
            if progress_every and produced % progress_every == 0:
                print(f"  ...{produced} квартир", flush=True)
            if produced >= limit:
                return

    with open(csv_path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            plan_id = row["plan_id"]
            if plan_id != current_plan:
                if current_plan is not None:
                    for item in _emit_plan():
                        yield item
                    if produced >= limit:
                        return
                current_plan = plan_id
                plan_walls, plan_doors = [], []
                plan_windows, plan_features = [], []
                apartments = {}

            kind, subtype = row["entity_type"], row["entity_subtype"]
            polygon = _parse_polygon(row["geometry"])
            if polygon is None:
                continue
            if kind == "separator" and subtype in WALL_SUBTYPES:
                plan_walls.append(polygon)
            elif kind == "opening" and subtype in DOOR_SUBTYPES:
                plan_doors.append(polygon)
            elif kind == "opening" and subtype in WINDOW_SUBTYPES:
                plan_windows.append(polygon)
            elif kind == "feature":
                plan_features.append(polygon)
            elif kind == "area" and subtype not in SKIP_AREAS:
                if row.get("unit_usage") != "RESIDENTIAL":
                    continue
                apartment_id = row["apartment_id"]
                bucket = apartments.setdefault(apartment_id, {"rooms": [], "room_names": []})
                bucket["rooms"].append(polygon)
                bucket["room_names"].append(subtype)

        if current_plan is not None:
            for item in _emit_plan():
                yield item


def to_real_layout(raw: Dict[str, Any]):
    """Сира квартира → RealLayout: поворот у рівне положення + фільтри якості."""
    from ml.floorplan.geom_render import RealLayout

    walls = raw["walls"]
    if len(walls) < MIN_WALLS or len(raw["rooms"]) < MIN_ROOMS:
        return None
    angle = _dominant_angle(walls)
    rot = lambda polys: [_rotate(p, angle) for p in polys]
    layout = RealLayout(
        walls=rot(walls), doors=rot(raw["doors"]), windows=rot(raw["windows"]),
        rooms=rot(raw["rooms"]),
        room_names=[ROOM_NAME_UK.get(n, "Кімната") for n in raw["room_names"]],
        source=f"swiss:{raw.get('apartment_id', '')[:8]}",
    ).normalized()

    width, height = layout.size_m()
    if not (3.0 <= width <= 40.0 and 3.0 <= height <= 40.0):
        return None
    room_area = sum(_polygon_area(r) for r in layout.rooms)
    if not (MIN_AREA_M2 <= room_area <= MAX_AREA_M2):
        return None
    # Дуже витягнуті шматки — це зазвичай коридор поверху, а не квартира.
    if max(width, height) / max(1e-6, min(width, height)) > 4.0:
        return None
    return layout


def build_cache(limit: int = 6000, csv_path: Optional[str] = None,
                out_path: str = CACHE_PATH) -> str:
    """Один прохід по датасету → компактний кеш RealLayout для тренування."""
    path = _swiss_csv_path(csv_path)
    if path is None:
        raise SystemExit(
            "geometries.csv не знайдено. Завантажте Swiss Dwellings v3 "
            "(https://zenodo.org/records/7788422) і розпакуйте geometries.csv."
        )
    print(f"[datasets] читаю {path}", flush=True)
    layouts = []
    seen = 0
    for raw in iter_swiss_apartments(path, limit=limit * 3):
        seen += 1
        layout = to_real_layout(raw)
        if layout is None:
            continue
        layouts.append({
            "walls": layout.walls, "doors": layout.doors, "windows": layout.windows,
            "rooms": layout.rooms, "room_names": layout.room_names, "source": layout.source,
        })
        if len(layouts) >= limit:
            break
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as handle:
        pickle.dump(layouts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[datasets] відібрано {len(layouts)} квартир із {seen} переглянутих "
          f"→ {out_path} ({size_mb:.1f} МБ)", flush=True)
    return out_path


_CACHE: Optional[List] = None


def load_layouts(path: str = CACHE_PATH) -> List:
    """Кеш → список RealLayout. Порожній список, якщо кешу немає (це не помилка:
    тренування просто піде на самій синтетиці)."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not os.path.exists(path):
        _CACHE = []
        return _CACHE
    from ml.floorplan.geom_render import RealLayout

    with open(path, "rb") as handle:
        raw = pickle.load(handle)
    _CACHE = [RealLayout(**item) for item in raw]
    return _CACHE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "stats"])
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    if args.command == "build":
        build_cache(limit=args.limit, csv_path=args.csv)
        return

    layouts = load_layouts()
    if not layouts:
        print("кешу немає — спершу `build`")
        return
    areas = [sum(_polygon_area(r) for r in l.rooms) for l in layouts]
    walls = [len(l.walls) for l in layouts]
    rooms = [len(l.rooms) for l in layouts]
    print(f"квартир: {len(layouts)}")
    print(f"площа м²: медіана {np.median(areas):.0f}  [{np.min(areas):.0f}..{np.max(areas):.0f}]")
    print(f"стін:     медіана {np.median(walls):.0f}  [{np.min(walls)}..{np.max(walls)}]")
    print(f"кімнат:   медіана {np.median(rooms):.0f}  [{np.min(rooms)}..{np.max(rooms)}]")


if __name__ == "__main__":
    main()
