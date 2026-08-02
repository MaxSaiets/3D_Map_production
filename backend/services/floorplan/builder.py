"""PlanVector → друкований 3D-макет приміщення.

Геометрія навмисно тримається у 2D якнайдовше: стіни — це буфер центральних
ліній, двері — 2D-віднімання, і лише вікна (які не йдуть від підлоги) вимагають
одного 3D-boolean. Так ми уникаємо крихких булевих операцій, через які в цьому
проєкті вже горіли пази й зубці на щільних мешах.

Порядок:
    1. масштаб             м → мм  (модель заданого габариту)
    2. стіни               центральні лінії → buffer(товщина/2) → unary_union
    3. двері               2D-віднімання з підошви (наскрізь по висоті)
    4. екструзія           2D → призма стін
    5. вікна               3D-віднімання коробок (manifold3d)
    6. основа              заповнений контур + відступ, приварена до стін
    7. валідація           watertight / is_volume / товщина під сопло
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import unary_union

from .plan_model import Opening, PlanVector, Wall

# ── Правила друку (сопло 0.4 мм) ─────────────────────────────────────────────
# 1.2 мм = рівно 3 периметри 0.4 → стіна друкується суцільно, без заповнення:
# міцно і швидко. Тонше 0.8 слайсер починає викидати периметри → дірки.
NOZZLE_MM = 0.4
MIN_WALL_MM = 1.2
ABS_MIN_WALL_MM = 0.8
MIN_BASE_MM = 1.0
DEFAULT_BASE_MM = 2.0
DEFAULT_MODEL_MM = 150.0        # «золота середина» за замірами слайсера: ~4-5 год, ~45 г
MAQUETTE_WALL_HEIGHT_MM = 22.0
# Стіни втоплюємо в основу на цю величину: коплонарний стик — найгірший вхід
# для булевих операцій, перекриття робить об'єднання надійним.
WELD_MM = 0.25


@dataclass
class BuildOptions:
    """Усе, що впливає на друковану геометрію."""

    model_size_mm: float = DEFAULT_MODEL_MM   # найдовша сторона макета
    scale_denominator: Optional[float] = None  # 1:N; якщо задано — має пріоритет
    wall_height_mm: Optional[float] = None     # явна висота; має пріоритет над режимом
    # ВИСОТА СТІН — головний важіль часу друку. Заміри слайсером: 8 мм → 4 год,
    # 45 мм → 8.4 год на тій самій моделі, тоді як товщина стіни 1.2→2.0 мм
    # додає лише 3%. Тому дефолт — «макетний» низький борт (у нього ще й краще
    # видно планування зверху), а справжня висота стелі — свідомий вибір.
    wall_height_mode: str = "maquette"          # maquette | true_scale
    maquette_wall_height_mm: float = MAQUETTE_WALL_HEIGHT_MM
    min_wall_mm: float = MIN_WALL_MM
    base_plate: bool = True
    base_thickness_mm: float = DEFAULT_BASE_MM
    base_margin_mm: float = 2.0
    cut_doors: bool = True
    cut_windows: bool = True
    # Пороги дверей: лишити тонку перемичку знизу (як у реальних макетах),
    # щоб стіна не «розривалась» і виріб був жорсткішим.
    door_threshold_mm: float = 0.0
    max_model_mm: float = 250.0                # межа стола принтера
    min_model_mm: float = 40.0

    def normalized(self) -> "BuildOptions":
        self.model_size_mm = float(min(max(self.model_size_mm, self.min_model_mm), self.max_model_mm))
        self.min_wall_mm = float(max(self.min_wall_mm, ABS_MIN_WALL_MM))
        self.base_thickness_mm = float(max(self.base_thickness_mm, MIN_BASE_MM))
        self.base_margin_mm = float(max(self.base_margin_mm, 0.0))
        return self


@dataclass
class BuildResult:
    mesh: trimesh.Trimesh
    parts: Dict[str, trimesh.Trimesh] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class PlanBuildError(RuntimeError):
    """Геометрію побудувати неможливо (порожній/зіпсований план)."""


# ═════════════════════════════════════════════════════════════════════════════
#  Допоміжне
# ═════════════════════════════════════════════════════════════════════════════
def _as_polygons(geom) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if not g.is_empty]
    if hasattr(geom, "geoms"):
        out: List[Polygon] = []
        for g in geom.geoms:
            out.extend(_as_polygons(g))
        return out
    return []


def _clean(geom, grid: float = 1e-3):
    """Нормалізує геометрію: валідність + прив'язка координат до сітки.

    buffer(0) лікує самоперетини, але не прибирає мікро-слівери, які лишає
    об'єднання десятків прямокутників стін із перекриттям у кутах. Ці слівери
    (частки мікрона завширшки) потім валять булеве віднімання вікон — симптом
    був саме такий: «вікна не вдалось прорізати» і негерметичні стіни.
    set_precision прив'язує координати до сітки 1 мкм і зшиває їх."""
    if geom is None or geom.is_empty:
        return geom
    if not geom.is_valid:
        geom = geom.buffer(0)
    try:
        from shapely import set_precision

        snapped = set_precision(geom, grid)
        if snapped is not None and not snapped.is_empty and snapped.area > geom.area * 0.98:
            geom = snapped
    except Exception:
        pass
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def _fill_holes(geom) -> List[Polygon]:
    """Полігони без внутрішніх дірок — підошва будівлі (стіни + кімнати всередині)."""
    return [Polygon(p.exterior) for p in _as_polygons(geom) if p.exterior is not None]


def _close_footprint(wall_area, scale: float, bbox_area: float):
    """Зімкнути підошву квартири через ПРОРІЗИ у зовнішній стіні.

    ЧОМУ ЦЕ ПОТРІБНО. Вхідні двері, балконний блок і панорамне вікно розривають
    зовнішній контур квартири. Тоді об'єднання стін розпадається на кілька
    полігонів, «залити дірки» вже нічого не дає, і основа вироджується у
    прямокутну плиту: замість макета квартири виходить квартира на підносі.
    Заміряно на реальних планах Floor Plan CIS — так було у 40 випадках зі 87.

    Морфологічне замикання (розширити на d, стиснути назад на d) зшиває розриви
    завширшки до 2d і НЕ рухає самих стін: змінюється лише основа. Мітрові стики
    (join_style=2) лишають кути гострими.

    РАДІУС БЕРЕТЬСЯ ВІД ШИРИНИ ПРОРІЗУ В МЕТРАХ, А НЕ ВІД ТОВЩИНИ СТІНИ. Перша
    версія рахувала його від товщини (≈0.15 м) — розриви лишались незшитими, і
    підошва виходила у формі самих стін: одна деталь, але кімнати без підлоги.
    Це гірше за плиту. Тому мало «одного полігона» — потрібно, щоб підошва
    накрила правдоподібну частку габариту (Г-подібна квартира ≈ 0.6-0.8).

    Повертає підошву одним полігоном або None, якщо зімкнути не вдалось."""
    if bbox_area <= 1e-9:
        return None
    best = None
    # Двері 0.9 м → балконний блок → панорамне скління на всю кімнату.
    for gap_m in (1.0, 1.6, 2.2, 3.0):
        d = 0.5 * gap_m * scale                      # мм у координатах моделі
        if d <= 1e-6:
            break
        try:
            closed = wall_area.buffer(d, join_style=2, resolution=4)
            closed = closed.buffer(-d, join_style=2, resolution=4)
        except Exception:
            break
        candidate = _clean(unary_union(_fill_holes(closed)))
        if candidate is None or candidate.is_empty:
            continue
        polygons = _as_polygons(candidate)
        if len(polygons) != 1:
            continue
        poly = polygons[0]
        ratio = poly.area / bbox_area
        best = poly
        if ratio >= 0.60:
            # Підошва накрила квартиру — більший радіус лише з'їв би виїмки.
            break
    return best


def _extrude_one(poly: Polygon, height: float) -> Optional[trimesh.Trimesh]:
    try:
        mesh = trimesh.creation.extrude_polygon(poly, height=height, engine="earcut")
    except TypeError:          # старіші trimesh без параметра engine
        try:
            mesh = trimesh.creation.extrude_polygon(poly, height=height)
        except Exception:
            return None
    except Exception:
        return None
    if mesh is None or len(mesh.faces) == 0:
        return None
    return mesh


def _extrude(geom, height: float, z0: float = 0.0) -> Optional[trimesh.Trimesh]:
    """Екструзія shapely-геометрії у меш. Кожен полігон окремо → потім union."""
    polys = [p for p in _as_polygons(geom) if p.area > 1e-9]
    if not polys or height <= 1e-6:
        return None
    meshes: List[trimesh.Trimesh] = []
    for poly in polys:
        m = _extrude_one(poly, float(height))
        # ВАЖЛИВО перевіряти РЕЗУЛЬТАТ, а не лише виняток. extrude_polygon на
        # полігоні з кількома дірками (а підошва квартири — це саме такий:
        # контур + кімнати) часто НЕ падає, а тихо повертає негерметичний меш:
        # earcut додає точки Штейнера, які зварюються у защіп. Далі на такому
        # меші валиться булеве віднімання вікон — і причину шукаєш не там.
        if not _usable_solid(m):
            repaired = None
            for shrink in (1e-3, 5e-3, 2e-2):
                candidate = _clean(poly.buffer(-shrink).buffer(shrink))
                parts = [_extrude_one(p, float(height)) for p in _as_polygons(candidate)]
                parts = [p for p in parts if p is not None]
                if parts and all(_usable_solid(p) for p in parts):
                    repaired = parts
                    break
            if repaired:
                meshes.extend(repaired)
                continue
        if m is not None:
            meshes.append(m)
    if not meshes:
        return None
    mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    if abs(z0) > 1e-9:
        mesh.apply_translation((0.0, 0.0, float(z0)))
    return mesh


def _boolean(op: str, meshes: Sequence[trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
    """union/difference через manifold3d з м'яким fallback на конкатенацію.

    Ніколи не кидає: булеві операції — найкрихкіше місце пайплайну, і краще
    віддати трохи гірший меш, ніж повалити генерацію (перевірено на пазах мап)."""
    live = [m for m in meshes if m is not None and len(m.faces) > 0]
    if not live:
        return None
    if len(live) == 1:
        return live[0]
    for engine in ("manifold", None):
        try:
            kwargs = {"engine": engine} if engine else {}
            if op == "union":
                out = trimesh.boolean.union(live, **kwargs)
            else:
                out = trimesh.boolean.difference(live, **kwargs)
            if out is not None and len(out.faces) > 0:
                return out
        except Exception:
            continue
    return trimesh.util.concatenate(live) if op == "union" else live[0]


def _usable_solid(mesh: Optional[trimesh.Trimesh]) -> bool:
    """Чи можна віддавати цей меш далі: замкнений, з додатним обʼємом."""
    if mesh is None or len(mesh.faces) == 0:
        return False
    try:
        return bool(mesh.is_watertight) and float(mesh.volume) > 1e-3
    except Exception:
        return False


def _wall_strip(wall: Wall, thickness_mm: float, scale: float) -> Optional[Polygon]:
    """Смуга стіни у мм: buffer центральної лінії з КВАДРАТНИМИ торцями.

    Квадратний торець (cap_style=3) видовжує стіну на пів-товщини за її кінець —
    саме так стіни й креслять, і саме це змикає кути без жодного «підтягування»
    координат. З плоским торцем два кінці в куті мусили б збігтися ТОЧНО: розрив
    у 1.5 px лишав щілину, кімната не замикалась, і в макеті замість квартири
    виходив відкритий лабіринт. Спроба лікувати це притягуванням кінців
    спрацьовувала, але зсувала стіни (IoU підошви 0.76 → 0.54) — тобто ламала
    саме те, заради чого все робиться: розміри."""
    p1 = (wall.x1 * scale, wall.y1 * scale)
    p2 = (wall.x2 * scale, wall.y2 * scale)
    if math.dist(p1, p2) < 1e-6:
        return None
    strip = LineString([p1, p2]).buffer(
        thickness_mm / 2.0, cap_style=3, join_style=2, resolution=2
    )
    return strip if isinstance(strip, Polygon) and strip.area > 1e-9 else None


def _opening_box(wall: Wall, op: Opening, thickness_mm: float, scale: float,
                 overcut_mm: float) -> Optional[Polygon]:
    """Прямокутник прорізу в плані (мм), трохи довший за товщину стіни.

    Перебіг (overcut) обов'язковий: різати рівно по товщині — це коплонарні
    грані, від яких булеві операції лишають плівки й «зубці»."""
    length_m = wall.length_m
    if length_m < 1e-6:
        return None
    ux, uy = wall.unit()
    cx_m, cy_m = wall.point_at(op.center_t)
    half_w = (op.width_m * scale) / 2.0
    half_t = thickness_mm / 2.0 + overcut_mm
    cx, cy = cx_m * scale, cy_m * scale
    # локальні осі: вздовж стіни (ux,uy), поперек (-uy,ux)
    ax, ay = ux * half_w, uy * half_w
    bx, by = -uy * half_t, ux * half_t
    ring = [
        (cx - ax - bx, cy - ay - by),
        (cx + ax - bx, cy + ay - by),
        (cx + ax + bx, cy + ay + by),
        (cx - ax + bx, cy - ay + by),
    ]
    poly = Polygon(ring)
    return poly if poly.area > 1e-9 else None


# ═════════════════════════════════════════════════════════════════════════════
#  Головна функція
# ═════════════════════════════════════════════════════════════════════════════
def build_plan_mesh(plan: PlanVector, options: Optional[BuildOptions] = None,
                    progress=None) -> BuildResult:
    """Векторний план (метри) → друкований меш (міліметри, Z вгору)."""
    opts = (options or BuildOptions()).normalized()
    warnings: List[str] = []

    def _tick(pct: int, msg: str) -> None:
        if progress:
            try:
                progress(pct, msg)
            except Exception:
                pass

    plan.sanitize()
    if not plan.walls:
        raise PlanBuildError("У плані немає жодної стіни — нема з чого будувати макет.")

    # ── 1. Масштаб ───────────────────────────────────────────────────────────
    minx, miny, maxx, maxy = plan.bounds()
    span_m = max(maxx - minx, maxy - miny)
    if span_m <= 1e-6:
        raise PlanBuildError("Розміри плану нульові — перевірте масштаб.")
    if opts.scale_denominator and opts.scale_denominator > 1:
        scale = 1000.0 / float(opts.scale_denominator)   # мм на метр
        model_span = span_m * scale
        if model_span > opts.max_model_mm:
            scale = opts.max_model_mm / span_m
            warnings.append(
                f"Масштаб 1:{opts.scale_denominator:.0f} дав би {model_span:.0f} мм — "
                f"обмежено до {opts.max_model_mm:.0f} мм (стіл принтера)."
            )
    else:
        # Відступ основи додається З ОБОХ боків — віднімаємо його з цілі, інакше
        # «180 мм» на виході дає 184 мм і не влазить у заявлений габарит стола.
        usable = max(10.0, opts.model_size_mm - 2.0 * (opts.base_margin_mm if opts.base_plate else 0.0))
        scale = usable / span_m
    denom = 1000.0 / scale if scale > 1e-9 else 0.0

    # Y плану дивиться ВНИЗ (як у зображенні) — інвертуємо рівно тут, один раз.
    def _sx(x: float) -> float:
        return (x - minx) * scale

    walls_shifted: List[Wall] = []
    for w in plan.walls:
        walls_shifted.append(Wall(
            x1=w.x1 - minx, y1=maxy - w.y1, x2=w.x2 - minx, y2=maxy - w.y2,
            thickness_m=w.thickness_m, bearing=w.bearing, height_m=w.height_m,
        ))

    # ── 2. Смуги стін ────────────────────────────────────────────────────────
    _tick(15, "Будую стіни...")
    thin_count = 0
    strips: List[Polygon] = []
    thickness_mm_by_index: List[float] = []
    for wall in walls_shifted:
        raw_mm = wall.thickness_m * scale
        t_mm = raw_mm
        if t_mm < opts.min_wall_mm:
            t_mm = opts.min_wall_mm
            thin_count += 1
        thickness_mm_by_index.append(t_mm)
        strip = _wall_strip(wall, t_mm, scale)
        if strip is not None:
            strips.append(strip)
    if not strips:
        raise PlanBuildError("Стіни виродились після масштабування.")
    if thin_count:
        warnings.append(
            f"{thin_count} стін тонші за {opts.min_wall_mm:.1f} мм у цьому масштабі — "
            f"потовщено до {opts.min_wall_mm:.1f} мм, щоб надрукувались."
        )

    wall_area = _clean(unary_union(strips))
    if wall_area is None or wall_area.is_empty:
        raise PlanBuildError("Не вдалось об'єднати стіни.")

    # ── 3. Двері — 2D-віднімання ─────────────────────────────────────────────
    _tick(30, "Прорізаю двері...")
    if opts.wall_height_mm:
        wall_h_mm = float(opts.wall_height_mm)
    elif opts.wall_height_mode == "true_scale":
        wall_h_mm = plan.wall_height_m * scale
    else:
        wall_h_mm = float(opts.maquette_wall_height_mm)
    wall_h_mm = max(wall_h_mm, 1.0)
    true_h_mm = max(1e-6, plan.wall_height_m * scale)
    vz = wall_h_mm / true_h_mm          # вертикальний коефіцієнт стиснення
    overcut = max(0.35, opts.min_wall_mm * 0.4)

    door_cuts: List[Polygon] = []
    window_specs: List[Tuple[Polygon, float, float]] = []  # (полігон, z_низ, z_верх)
    for op in plan.openings:
        if op.wall < 0 or op.wall >= len(walls_shifted):
            continue
        wall = walls_shifted[op.wall]
        t_mm = thickness_mm_by_index[op.wall]
        poly = _opening_box(wall, op, t_mm, scale, overcut)
        if poly is None:
            continue
        # Стискаємо отвори по вертикалі разом зі стіною: інакше при макетній
        # висоті 22 мм вікно з підвіконням на 14 мм і верхом на 38 мм
        # перетворилось би на проріз до самого верху стіни.
        sill_mm = op.sill_m * scale * vz
        top_mm = min(wall_h_mm, (op.sill_m + op.height_m) * scale * vz)
        if op.kind in ("door", "arch"):
            if not opts.cut_doors:
                continue
            # Двері йдуть від підлоги — якщо ще й до верху стіни, це просто 2D-виріз.
            if opts.door_threshold_mm > 0.05:
                window_specs.append((poly, opts.door_threshold_mm, max(top_mm, opts.door_threshold_mm + 0.5)))
            elif top_mm >= wall_h_mm - 1e-6:
                door_cuts.append(poly)
            else:
                window_specs.append((poly, 0.0, top_mm))
        else:
            if not opts.cut_windows:
                continue
            if sill_mm <= 0.05 and top_mm >= wall_h_mm - 1e-6:
                door_cuts.append(poly)
            else:
                window_specs.append((poly, max(0.0, sill_mm), max(sill_mm + 0.5, top_mm)))

    footprint = wall_area
    if door_cuts:
        footprint = _clean(footprint.difference(unary_union(door_cuts)))
    if footprint is None or footprint.is_empty:
        raise PlanBuildError("Прорізи з'їли всі стіни — перевірте ширину дверей.")

    # ── 4. Екструзія стін ────────────────────────────────────────────────────
    _tick(45, "Піднімаю стіни...")
    base_mm = opts.base_thickness_mm if opts.base_plate else 0.0
    wall_z0 = max(0.0, base_mm - WELD_MM) if opts.base_plate else 0.0
    walls_mesh = _extrude(footprint, wall_h_mm + (base_mm - wall_z0 if opts.base_plate else 0.0), z0=wall_z0)
    if walls_mesh is None:
        raise PlanBuildError("Не вдалось екструдувати стіни.")

    # ── 5. Вікна — єдиний 3D-boolean ─────────────────────────────────────────
    if window_specs:
        _tick(60, "Прорізаю вікна...")
        cutters: List[trimesh.Trimesh] = []
        for poly, z0, z1 in window_specs:
            height = max(0.4, z1 - z0)
            cutter = _extrude(poly, height, z0=base_mm + z0)
            if cutter is not None:
                cutters.append(cutter)
        if cutters:
            merged_cutter = cutters[0] if len(cutters) == 1 else trimesh.util.concatenate(cutters)
            cut = _boolean("difference", [walls_mesh, merged_cutter])
            if not _usable_solid(cut):
                # Різати всі вікна однією склеєною формою швидше, але коли ці
                # форми торкаються одна одної (сусідні вікна в одній стіні),
                # результат виходить негерметичним. Послідовний виріз повільніший
                # на частки секунди, зате кожен крок — коректне тіло.
                sequential = walls_mesh
                for cutter in cutters:
                    step = _boolean("difference", [sequential, cutter])
                    if _usable_solid(step):
                        sequential = step
                cut = sequential if _usable_solid(sequential) else None
            if _usable_solid(cut):
                walls_mesh = cut
            else:
                warnings.append("Вікна не вдалось прорізати — залишені суцільними.")

    parts: Dict[str, trimesh.Trimesh] = {}
    combined: List[trimesh.Trimesh] = [walls_mesh]

    # ── 6. Основа ────────────────────────────────────────────────────────────
    base_mesh: Optional[trimesh.Trimesh] = None
    if opts.base_plate:
        _tick(72, "Роблю основу...")
        filled = _clean(unary_union(_fill_holes(wall_area)))
        # Розрив у зовнішній стіні (вхід, балкон, панорамне вікно) розсипає
        # підошву на острівці. Спершу пробуємо зшити самі прорізи — це лишає
        # основі форму квартири; прямокутна плита нижче лише як остання лінія.
        minx_w, miny_w, maxx_w, maxy_w = wall_area.bounds
        bbox_area = max(1e-9, (maxx_w - minx_w) * (maxy_w - miny_w))
        if filled is None or filled.is_empty or filled.area < 0.55 * bbox_area:
            closed = _close_footprint(wall_area, scale, bbox_area)
            if closed is not None and (filled is None or closed.area > filled.area):
                filled = closed
        if opts.base_margin_mm > 0:
            filled = _clean(filled.buffer(opts.base_margin_mm, join_style=2, resolution=4))
        # Якщо підошва розпалась на кілька острівців (буває, коли детектор не
        # зчепив частину стін), органічна основа дасть кілька окремих деталей —
        # на столі принтера вони роз'їдуться. Тоді кладемо суцільну прямокутну
        # плиту: виглядає навмисно і гарантує один виріб.
        pieces = _as_polygons(filled)
        if len(pieces) > 1:
            minx_b, miny_b, maxx_b, maxy_b = filled.bounds
            filled = box(minx_b, miny_b, maxx_b, maxy_b)
            warnings.append("Стіни виявились не зчеплені — основу зроблено суцільною плитою.")
        base_mesh = _extrude(filled, base_mm, z0=0.0)
        if base_mesh is not None:
            combined.append(base_mesh)
            parts["base"] = base_mesh
        else:
            warnings.append("Основу побудувати не вдалось — макет буде без підкладки.")

    parts["walls"] = walls_mesh

    _tick(84, "Зварюю в одну деталь...")
    mesh = _boolean("union", combined) if len(combined) > 1 else combined[0]
    if mesh is None or len(mesh.faces) == 0:
        raise PlanBuildError("Порожній меш після об'єднання.")

    # ── 7. Валідація / ремонт ────────────────────────────────────────────────
    _tick(92, "Перевіряю модель...")
    mesh = _repair(mesh)
    if not mesh.is_watertight:
        warnings.append("Меш не повністю герметичний — слайсер, найімовірніше, полагодить сам.")
    # Топологія. Спокуса перевіряти euler_number == 2−2·(кількість дірок)
    # ВІДКИНУТА свідомо: у макета «лоток» — основа закриває кімнати знизу, а
    # зверху вони відкриті, тож кожен проріз додає ручку, і очікуване значення
    # euler не виводиться з кількості кімнат. Така перевірка давала 100% хибних
    # тривог. Перевіряємо те, що справді ламає друк.
    try:
        euler = int(mesh.euler_number)
    except Exception:
        euler = 0
    if not mesh.is_winding_consistent:
        warnings.append("Непослідовна орієнтація граней — нормалі виправлено автоматично.")
    try:
        bbox_volume = float(np.prod(mesh.extents))
        if not (0.0 < float(mesh.volume) < bbox_volume * 1.001):
            warnings.append("Обʼєм моделі виглядає некоректним — перевірте у слайсері.")
    except Exception:
        pass
    # Відірвані шматки — це не косметика: на столі принтера вони відклеюються,
    # падають і псують решту друку. Дрібні прибираємо самі, про великі кажемо.
    try:
        bodies = [b for b in mesh.split(only_watertight=False) if len(b.faces) > 0]
        if len(bodies) > 1:
            volumes = [abs(float(b.volume)) for b in bodies]
            total = sum(volumes) or 1.0
            keep = [b for b, v in zip(bodies, volumes) if v / total >= 0.02]
            dropped = len(bodies) - len(keep)
            if keep and dropped:
                mesh = keep[0] if len(keep) == 1 else trimesh.util.concatenate(keep)
                warnings.append(
                    f"Прибрано {dropped} відірваних дрібних фрагментів — вони б "
                    f"відпали під час друку."
                )
            if len(keep) > 1:
                warnings.append(
                    f"Модель складається з {len(keep)} окремих частин — "
                    f"перевірте, чи всі стіни стоять на основі."
                )
    except Exception:
        pass

    # ТОЧНИЙ ГАБАРИТ. Масштаб рахувався з plan.bounds(), який додає півтовщини
    # стіни по ОБИДВОХ осях — для косих стін це трохи більше за реальну підошву,
    # і виріб виходив на ±0.6 мм не тим, що замовили. Розмір — це і є товар, тож
    # доводимо його по факту. Правка на частки відсотка, мінімальну товщину
    # стіни вона зачепити не може.
    if not opts.scale_denominator:
        extents = mesh.extents
        longest = float(max(extents[0], extents[1]))
        if longest > 1e-6:
            correction = opts.model_size_mm / longest
            if abs(correction - 1.0) > 1e-4:
                mesh.apply_scale((correction, correction, 1.0))
                scale *= correction
                denom = 1000.0 / scale if scale > 1e-9 else 0.0

    # На стіл: мінімальний кут у нуль по X/Y, низ на Z=0
    bounds = mesh.bounds
    mesh.apply_translation((-bounds[0][0], -bounds[0][1], -bounds[0][2]))

    size = mesh.extents
    stats = {
        "scale_denominator": round(denom, 1),
        "mm_per_m": round(scale, 4),
        "model_size_mm": [round(float(v), 2) for v in size],
        "wall_height_mm": round(wall_h_mm, 2),
        "base_thickness_mm": round(base_mm, 2),
        "min_wall_mm": round(opts.min_wall_mm, 2),
        "thinnest_wall_mm": round(min(thickness_mm_by_index), 2) if thickness_mm_by_index else 0.0,
        "plan_size_m": [round(maxx - minx, 3), round(maxy - miny, 3)],
        "walls": len(plan.walls),
        "doors": sum(1 for o in plan.openings if o.kind in ("door", "arch")),
        "windows": sum(1 for o in plan.openings if o.kind == "window"),
        "rooms": len(plan.rooms),
        "volume_cm3": round(float(mesh.volume) / 1000.0, 1) if mesh.is_volume else None,
        "filament_g": _filament_estimate_g(mesh),
        "triangles": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "euler_number": euler,
    }
    _tick(96, "Готово")
    return BuildResult(mesh=mesh, parts=parts, stats=stats, warnings=warnings)


def _filament_estimate_g(mesh: trimesh.Trimesh, shell_mm: float = 1.0,
                         infill: float = 0.15, density: float = 1.24) -> Optional[float]:
    """Оцінка ваги PLA з урахуванням того, що слайсер друкує оболонку + заповнення.

    Голий об'єм бреше втричі: зовнішні стіни макета можуть бути 6 мм завтовшки,
    але друкуються 3 периметрами + 15% infill. Модель: оболонка ~1 мм по всій
    площі (площа рахує обидві сторони → ділимо навпіл), решта — infill."""
    try:
        if not mesh.is_volume:
            return None
        volume = float(mesh.volume)
        shell = min(volume, float(mesh.area) * shell_mm * 0.5)
        return round((shell + max(0.0, volume - shell) * infill) / 1000.0 * density, 1)
    except Exception:
        return None


def _repair(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Ремонт ЛИШЕ якщо меш справді зламаний.

    Два уроки цього проєкту, зашиті сюди:
      • merge_vertices() на вже герметичному виході manifold3d СКЛЕЮЄ коінцидентні
        вершини різних поверхонь → ребро з 4 гранями → is_watertight стає False.
        Тобто «ремонт» ламав ідеальний меш. Тому спершу перевіряємо.
      • nondegenerate_faces() з порогом висоти вже нищив валідні тонкі грані
        (гребінець у пазах з'єднувачів) — не використовуємо взагалі."""
    try:
        if mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0:
            return mesh
    except Exception:
        pass
    try:
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(mesh)
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
    except Exception:
        pass
    return mesh
