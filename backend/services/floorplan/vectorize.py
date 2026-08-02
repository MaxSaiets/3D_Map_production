"""Маска стін → векторний план (відрізки центральних ліній + товщини + отвори).

Підхід: медіальна вісь (скелет) + локальна товщина з distance transform.
Розглядались альтернативи:
  • Hough-лінії — тонуть у розмірних лініях, штриховці й тексті плану;
  • морфологічне відкриття смугами (окремо H і V) — просте, але вбиває короткі
    стіни й будь-що не під 90°;
скелет працює однаково для будь-якої орієнтації й одразу дає товщину, а
ортогоналізація застосовується ПІСЛЯ, як м'яке підправлення, а не як припущення.

Координати на виході — ПІКСЕЛІ вхідного зображення (thickness теж у пікселях).
У метри це переводить PlanVector.from_pixel_dict() у момент побудови мешу, коли
масштаб уже підтверджений користувачем.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .plan_model import (
    DEFAULT_DOOR_HEIGHT_M, DEFAULT_WINDOW_HEIGHT_M, DEFAULT_WINDOW_SILL_M,
    Opening, PlanVector, Wall,
)

Pt = Tuple[float, float]


@dataclass
class VectorizeConfig:
    min_wall_len_px: float = 14.0
    min_thickness_px: float = 2.0
    max_thickness_px: float = 90.0
    rdp_tolerance_px: float = 2.5
    ortho_tolerance_deg: float = 13.0     # у межах — доводимо до 0/90°
    snap_px: float = 7.0
    spur_factor: float = 1.4              # відросток коротший за товщину×цей коеф. — шум
    min_component_px: int = 120           # дрібні плями = текст/меблі
    close_gap_px: int = 3
    # Фільтр «волосяних» компонент. Поріг прив'язаний до РОЗДІЛЬНОЇ ЗДАТНОСТІ, а
    # не до найтовщої стіни: несуча (0.4 м) товща за перегородку (0.08 м) у 5
    # разів, тож «відсоток від найтовщої» стирав саме перегородки — план
    # «худнув» на 30-75%. Спираємось на те, що найтонша РЕАЛЬНА стіна ≈ 0.6%
    # ширини аркуша, а розмірна лінія лишається 1-3 px незалежно від нього.
    min_wall_fraction_of_image: float = 0.006
    abs_min_half_thickness_px: float = 1.6
    thin_cut_safety: float = 0.72          # запас, щоб не зрізати саму перегородку
    smooth_factor: float = 0.45            # ядро згладжування контуру маски
    # Пороги для «прорізу як розриву» (у частках типової товщини стіни)
    gap_min_ratio: float = 1.4
    gap_max_ratio: float = 14.0


# ═════════════════════════════════════════════════════════════════════════════
#  Скелет → полілінії
# ═════════════════════════════════════════════════════════════════════════════
_NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def expected_min_wall_px(shape: Sequence[int], cfg: Optional[VectorizeConfig] = None) -> float:
    """Очікувана ПОВНА ширина найтоншої справжньої стіни, у пікселях аркуша."""
    cfg = cfg or VectorizeConfig()
    longest = float(max(shape[0], shape[1]))
    return max(3.0, longest * cfg.min_wall_fraction_of_image)


def thin_component_cut_px(shape: Sequence[int], cfg: Optional[VectorizeConfig] = None) -> float:
    """Поріг півтовщини, нижче якого компонента вважається волосяною лінією."""
    cfg = cfg or VectorizeConfig()
    return max(cfg.abs_min_half_thickness_px,
               expected_min_wall_px(shape, cfg) / 2.0 * cfg.thin_cut_safety)


def _clean_mask(mask: np.ndarray, cfg: VectorizeConfig) -> np.ndarray:
    """Прибирає з маски все, що не є стіною: волосяні лінії й дрібні плями.

    ГОЛОВНИЙ ФІЛЬТР — за ТОВЩИНОЮ, а не за площею. Розмірний ланцюжок може мати
    величезну площу (він тягнеться через увесь аркуш), але лишається волосяним:
    його max(distanceTransform) ≈ 1.4 px проти ≈ 23 px у справжньої стіни. Без
    цього фільтра засічки розмірних ліній ставали «стінами», габарит плану ріс
    на ~7%, і виріб друкувався фізично неправильного розміру — при тому що на
    екрані все виглядало пристойно. Плюс вони давали окрему відірвану стінку,
    яка на друку просто падає."""
    import cv2

    m = (mask > 0).astype(np.uint8)

    # Поріг рахуємо ДО замикання: воно зліплює сусідні волосинки у товщу пляму.
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num > 1:
        max_dist = np.zeros(num, dtype=np.float32)
        np.maximum.at(max_dist, labels.ravel(), dist.ravel())
        thin_cut = thin_component_cut_px(m.shape, cfg)
        keep = max_dist >= thin_cut
        keep[0] = False
        if not keep[1:].any():                    # нічого не лишилось — беремо найтовстішу
            keep[int(np.argmax(max_dist[1:])) + 1] = True
        m = keep[labels].astype(np.uint8)

    # ЗГЛАДЖУВАННЯ КОНТУРУ. Кожна зазубрина на межі маски породжує відросток
    # скелета, відросток — зайвий відрізок, і на зашумленому фото з 12 реальних
    # стін виходило 100+ «стін»: редактор ставав непридатним, а O(n²)-етапи
    # злипання розганялись до хвилини. Розмір ядра беремо від очікуваної
    # найтоншої стіни, щоб не з'їсти саму перегородку.
    smooth = max(3, int(round(expected_min_wall_px(m.shape, cfg) * cfg.smooth_factor)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth, smooth))
    smoothed = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
    # Запобіжник: якщо згладжування з'їло більш ніж третину стін (буває на
    # шумній масці класичного CV, де перегородки й так на межі), відкочуємось —
    # краще трохи зайвих відрізків, ніж половина квартири без стін.
    if smoothed.sum() >= m.sum() * 0.65:
        m = smoothed
    if cfg.close_gap_px > 0:
        k = np.ones((cfg.close_gap_px, cfg.close_gap_px), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m
    keep = np.zeros(num, dtype=bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= cfg.min_component_px:
            keep[i] = True
    if not keep.any():
        keep[int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1] = True
    return keep[labels].astype(np.uint8)


def _trace_chains(skel: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Скелет (0/1) → список ланцюжків пікселів (y, x).

    Прийом: видаляємо вузли (степінь ≠ 2), решта розпадається на прості ланцюги —
    їх легко впорядкувати. Потім кінці ланцюга дотягуємо назад до сусіднього
    вузла, щоб перехрестя не «розчепились»."""
    import cv2

    skel = (skel > 0).astype(np.uint8)
    if skel.sum() == 0:
        return []
    kernel = np.ones((3, 3), np.uint8)
    deg = cv2.filter2D(skel, cv2.CV_16U, kernel, borderType=cv2.BORDER_CONSTANT) - skel
    deg = deg * skel

    node_mask = ((deg != 2) & (skel > 0)).astype(np.uint8)
    chain_mask = ((skel > 0) & (node_mask == 0)).astype(np.uint8)

    height, width = skel.shape
    node_set = set(map(tuple, np.argwhere(node_mask > 0)))

    num, labels = cv2.connectedComponents(chain_mask, connectivity=8)
    chains: List[List[Tuple[int, int]]] = []
    for lab in range(1, num):
        pixels = [tuple(p) for p in np.argwhere(labels == lab)]
        if not pixels:
            continue
        pixel_set = set(pixels)

        def local_neighbors(p):
            y, x = p
            return [(y + dy, x + dx) for dy, dx in _NB8 if (y + dy, x + dx) in pixel_set]

        ends = [p for p in pixels if len(local_neighbors(p)) <= 1]
        start = ends[0] if ends else pixels[0]
        ordered: List[Tuple[int, int]] = [start]
        visited = {start}
        cur = start
        while True:
            nxt = None
            for cand in local_neighbors(cur):
                if cand not in visited:
                    nxt = cand
                    break
            if nxt is None:
                break
            ordered.append(nxt)
            visited.add(nxt)
            cur = nxt
        if len(ordered) < len(pixels) * 0.55:   # розгалужений залишок — пропускаємо
            continue

        # дотягуємо до вузлів на обох кінцях
        for idx, end in ((0, ordered[0]), (-1, ordered[-1])):
            y, x = end
            for dy, dx in _NB8:
                cand = (y + dy, x + dx)
                if cand in node_set:
                    if idx == 0:
                        ordered.insert(0, cand)
                    else:
                        ordered.append(cand)
                    break
        chains.append(ordered)

    # ізольовані вузли-пари (перехрестя без ланцюга між ними) ігноруємо навмисно:
    # це відрізки в 1-2 px, вони нижче за min_wall_len_px у будь-якому разі.
    return chains


def _simplify(chain: Sequence[Tuple[int, int]], tolerance: float) -> np.ndarray:
    """Дуглас-Пекер по ланцюжку пікселів → (N,2) масив (x, y)."""
    from skimage.measure import approximate_polygon

    arr = np.array([[p[1], p[0]] for p in chain], dtype=np.float64)  # (x, y)
    if len(arr) <= 2:
        return arr
    simplified = approximate_polygon(arr, tolerance=tolerance)
    return simplified if len(simplified) >= 2 else arr


# ═════════════════════════════════════════════════════════════════════════════
#  Ортогоналізація та зшивання
# ═════════════════════════════════════════════════════════════════════════════
def _apartment_bbox(solid: np.ndarray, min_room_px: int = 400
                    ) -> Optional[Tuple[float, float, float, float]]:
    """Габарит квартири = габарит усіх ЗАМКНЕНИХ приміщень у масці.

    Розмірні ланцюжки, штамп і рамка аркуша нічого не замикають, тому лежать
    поза цим прямокутником — і відрізаються геометрично, без порогів товщини.
    Це працює навіть тоді, коли морфологічне замикання вже зліпило ланцюжок зі
    стіною в одну компоненту (саме тому фільтр по компонентах тут не рятує)."""
    import cv2
    from scipy.ndimage import binary_fill_holes

    m = (solid > 0)
    try:
        rooms = binary_fill_holes(m) & ~m
    except Exception:
        return None
    rooms_u8 = rooms.astype(np.uint8)
    num, _labels, stats, _ = cv2.connectedComponentsWithStats(rooms_u8, connectivity=8)
    boxes = [
        (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
         stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH],
         stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT])
        for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_room_px
    ]
    if not boxes:
        return None
    return (float(min(b[0] for b in boxes)), float(min(b[1] for b in boxes)),
            float(max(b[2] for b in boxes)), float(max(b[3] for b in boxes)))


def _drop_outside(segments: List[Tuple[Pt, Pt, float]],
                  bbox: Optional[Tuple[float, float, float, float]]
                  ) -> List[Tuple[Pt, Pt, float]]:
    """Викидає відрізки, середина яких лежить поза габаритом квартири."""
    if bbox is None or not segments:
        return segments
    margin = max(s[2] for s in segments) * 1.2 + 4.0
    x0, y0, x1, y1 = bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin
    kept = [
        s for s in segments
        if x0 <= (s[0][0] + s[1][0]) / 2.0 <= x1 and y0 <= (s[0][1] + s[1][1]) / 2.0 <= y1
    ]
    return kept or segments


def _dominant_angle(segments: List[Tuple[Pt, Pt, float]]) -> float:
    """Головний напрям плану (rad, у діапазоні [-45°, 45°)), зважений довжиною."""
    if not segments:
        return 0.0
    acc_sin, acc_cos = 0.0, 0.0
    for (x1, y1), (x2, y2), _ in segments:
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1e-6:
            continue
        ang = math.atan2(y2 - y1, x2 - x1) % (math.pi / 2)  # 90°-періодичність
        acc_sin += length * math.sin(4 * ang)
        acc_cos += length * math.cos(4 * ang)
    if abs(acc_sin) < 1e-9 and abs(acc_cos) < 1e-9:
        return 0.0
    return math.atan2(acc_sin, acc_cos) / 4.0


def _snap_ortho(segments: List[Tuple[Pt, Pt, float]], base_angle: float,
                tol_deg: float) -> List[Tuple[Pt, Pt, float]]:
    """Доводить майже-осьові відрізки до точної осі, зберігаючи центр і довжину.

    Косі стіни (еркери, скоси) НЕ чіпаємо — вони бувають справжніми, і випрямити
    їх силоміць гірше, ніж лишити як є."""
    tol = math.radians(tol_deg)
    out: List[Tuple[Pt, Pt, float]] = []
    for (x1, y1), (x2, y2), thickness in segments:
        ang = math.atan2(y2 - y1, x2 - x1)
        length = math.hypot(x2 - x1, y2 - y1)
        best = None
        for k in range(-2, 3):
            target = base_angle + k * math.pi / 2
            diff = (ang - target + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) < tol and (best is None or abs(diff) < abs(best[1])):
                best = (target, diff)
        if best is None:
            out.append(((x1, y1), (x2, y2), thickness))
            continue
        target = best[0]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dx, dy = math.cos(target) * length / 2.0, math.sin(target) * length / 2.0
        out.append(((cx - dx, cy - dy), (cx + dx, cy + dy), thickness))
    return out


def _snap_axis_offsets(segments: List[Tuple[Pt, Pt, float]], base_angle: float,
                       tol_deg: float, cluster_px: float) -> List[Tuple[Pt, Pt, float]]:
    """Зводить паралельні стіни на СПІЛЬНУ пряму.

    Це те, чого бракувало найбільше. `_snap_ortho` вирівнює КУТИ, зберігаючи
    центр кожного відрізка, тож два шматки однієї стіни лишались зміщеними на
    кілька пікселів — і `_merge_collinear` їх не зливав. На ідеальній масці цього
    не видно (краї рівні), а на виході нейромережі чи класичного CV край
    «дихає», і одна стіна розсипалась на 5-8 відрізків: 12 справжніх стін
    перетворювались на 40 у редакторі.

    Групуємо відрізки кожного напряму за перпендикулярним зміщенням і садимо на
    зважену за довжиною медіану групи."""
    if len(segments) < 2:
        return segments
    tol = math.radians(tol_deg)
    families: Dict[int, List[int]] = {}
    offsets: Dict[int, float] = {}
    for index, ((x1, y1), (x2, y2), _t) in enumerate(segments):
        angle = math.atan2(y2 - y1, x2 - x1)
        for k in range(-2, 3):
            target = base_angle + k * math.pi / 2
            diff = (angle - target + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) < tol:
                # перпендикулярне зміщення середини відрізка від осі напряму
                nx, ny = -math.sin(target), math.cos(target)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                families.setdefault(k % 2, []).append(index)
                offsets[index] = cx * nx + cy * ny
                break

    result = list(segments)
    for indices in families.values():
        indices.sort(key=lambda i: offsets[i])
        group: List[int] = []
        for index in indices:
            if group and offsets[index] - offsets[group[-1]] > cluster_px:
                _apply_offset_group(result, group, offsets, base_angle)
                group = []
            group.append(index)
        if group:
            _apply_offset_group(result, group, offsets, base_angle)
    return result


def _apply_offset_group(segments: List[Tuple[Pt, Pt, float]], group: List[int],
                        offsets: Dict[int, float], base_angle: float) -> None:
    if len(group) < 2:
        return
    weights = [(offsets[i], math.dist(segments[i][0], segments[i][1])) for i in group]
    total = sum(w for _o, w in weights)
    if total <= 0:
        return
    weights.sort(key=lambda ow: ow[0])
    acc, target_offset = 0.0, weights[len(weights) // 2][0]
    for offset, weight in weights:
        acc += weight
        if acc >= total / 2.0:
            target_offset = offset
            break
    for index in group:
        shift = target_offset - offsets[index]
        if abs(shift) < 1e-9:
            continue
        (x1, y1), (x2, y2), thickness = segments[index]
        angle = math.atan2(y2 - y1, x2 - x1)
        best_target, best_diff = angle, math.pi
        for k in range(-2, 3):
            candidate = base_angle + k * math.pi / 2
            diff = abs((angle - candidate + math.pi) % (2 * math.pi) - math.pi)
            if diff < best_diff:
                best_diff, best_target = diff, candidate
        nx, ny = -math.sin(best_target), math.cos(best_target)
        segments[index] = ((x1 + nx * shift, y1 + ny * shift),
                           (x2 + nx * shift, y2 + ny * shift), thickness)


def _merge_parallel_duplicates(segments: List[Tuple[Pt, Pt, float]], snap: float
                               ) -> List[Tuple[Pt, Pt, float]]:
    """Зливає ПАРАЛЕЛЬНІ ДУБЛІКАТИ однієї стіни.

    На широкій смузі (несуча стіна + прилегла штриховка) скелет інколи
    роздвоюється, і одна стіна приходить двома майже паралельними відрізками за
    10-15 px один від одного. На накладенні це виглядало як «хибні знахідки»
    поруч зі справжніми — 13% зайвих ліній, які користувач мусив би стирати
    руками.

    Дві РІЗНІ стіни ніколи не стоять ближче за власну товщину (інакше вони б
    злилися в одну), тому поріг у 0.9 товщини безпечний: коридор шириною 1 м —
    це 70 px, на порядок більше."""
    result = list(segments)
    changed = True
    while changed:
        changed = False
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                (a1, a2, ta), (b1, b2, tb) = result[i], result[j]
                la, lb = math.dist(a1, a2), math.dist(b1, b2)
                if la < 1e-6 or lb < 1e-6:
                    continue
                aa = math.atan2(a2[1] - a1[1], a2[0] - a1[0])
                ba = math.atan2(b2[1] - b1[1], b2[0] - b1[0])
                if abs((aa - ba + math.pi / 2) % math.pi - math.pi / 2) > math.radians(6.0):
                    continue
                ux, uy = (a2[0] - a1[0]) / la, (a2[1] - a1[1]) / la
                nx, ny = -uy, ux
                # перпендикулярна відстань між прямими
                offset = abs((b1[0] - a1[0]) * nx + (b1[1] - a1[1]) * ny)
                if offset > max(snap, max(ta, tb) * 0.9):
                    continue
                # перекриття вздовж напряму
                pa = sorted([(a1[0] - a1[0]) * ux + (a1[1] - a1[1]) * uy,
                             (a2[0] - a1[0]) * ux + (a2[1] - a1[1]) * uy])
                pb = sorted([(b1[0] - a1[0]) * ux + (b1[1] - a1[1]) * uy,
                             (b2[0] - a1[0]) * ux + (b2[1] - a1[1]) * uy])
                overlap = min(pa[1], pb[1]) - max(pa[0], pb[0])
                if overlap < min(la, lb) * 0.4:
                    continue

                lo, hi = min(pa[0], pb[0]), max(pa[1], pb[1])
                weight = la + lb
                mid_offset = ((b1[0] - a1[0]) * nx + (b1[1] - a1[1]) * ny) * lb / weight
                base = (a1[0] + nx * mid_offset, a1[1] + ny * mid_offset)
                merged = (
                    (base[0] + ux * lo, base[1] + uy * lo),
                    (base[0] + ux * hi, base[1] + uy * hi),
                    max(ta, tb),
                )
                result = [s for k, s in enumerate(result) if k not in (i, j)] + [merged]
                changed = True
                break
            if changed:
                break
    return result


def _merge_collinear(segments: List[Tuple[Pt, Pt, float]], snap: float
                     ) -> List[Tuple[Pt, Pt, float]]:
    """Зливає відрізки, що лежать на одній лінії й майже торкаються."""
    result = list(segments)
    changed = True
    while changed:
        changed = False
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                a1, a2, ta = result[i]
                b1, b2, tb = result[j]
                if abs(ta - tb) > max(1.5, 0.35 * max(ta, tb)):
                    continue
                aa = math.atan2(a2[1] - a1[1], a2[0] - a1[0])
                ba = math.atan2(b2[1] - b1[1], b2[0] - b1[0])
                diff = abs((aa - ba + math.pi / 2) % math.pi - math.pi / 2)
                if diff > math.radians(4.0):
                    continue
                pairs = [(a1, b1), (a1, b2), (a2, b1), (a2, b2)]
                close = [(p, q) for p, q in pairs if math.dist(p, q) <= snap]
                if not close:
                    continue
                pts = [a1, a2, b1, b2]
                far = max(((p, q) for p in pts for q in pts),
                          key=lambda pq: math.dist(pq[0], pq[1]))
                # перпендикулярне відхилення від нової лінії має лишатись малим
                (nx1, ny1), (nx2, ny2) = far
                nlen = math.hypot(nx2 - nx1, ny2 - ny1)
                if nlen < 1e-6:
                    continue
                ok = True
                for px, py in pts:
                    dev = abs((nx2 - nx1) * (ny1 - py) - (nx1 - px) * (ny2 - ny1)) / nlen
                    if dev > max(1.5, snap * 0.6):
                        ok = False
                        break
                if not ok:
                    continue
                merged = (far[0], far[1], (ta + tb) / 2.0)
                result = [s for k, s in enumerate(result) if k not in (i, j)] + [merged]
                changed = True
                break
            if changed:
                break
    return result


def _bridge_openings(segments: List[Tuple[Pt, Pt, float]], cfg: VectorizeConfig
                     ) -> Tuple[List[Tuple[Pt, Pt, float]], List[Tuple[Pt, float]]]:
    """Зшиває колінеарні «пеньки» стіни через проріз і повертає самі прорізи.

    ЧОМУ ЦЕ ПОТРІБНО. Двері на плані малюють як РОЗРИВ у стіні. Розрив розриває
    й скелет, тож стіна приходить сюди двома окремими відрізками — і шукати
    «дірку всередині стіни» вже немає де. Через це класичний шлях не знаходив
    ЖОДНИХ дверей. Тут ми робимо зворотну операцію: бачимо два шматки на одній
    прямій із однаковою товщиною і правдоподібною щілиною між ними → це одна
    стіна з прорізом. Стіну зшиваємо, щілину запам'ятовуємо як отвір.

    Повертає (відрізки, [(середина_прорізу, ширина_px)])."""
    segs = list(segments)
    gaps: List[Tuple[Pt, float]] = []
    changed = True
    while changed:
        changed = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                a1, a2, ta = segs[i]
                b1, b2, tb = segs[j]
                if abs(ta - tb) > 0.35 * max(ta, tb):
                    continue
                aa = math.atan2(a2[1] - a1[1], a2[0] - a1[0])
                ba = math.atan2(b2[1] - b1[1], b2[0] - b1[0])
                if abs((aa - ba + math.pi / 2) % math.pi - math.pi / 2) > math.radians(4.0):
                    continue

                # найближча пара кінців = краї прорізу
                pairs = [(a1, b1), (a1, b2), (a2, b1), (a2, b2)]
                p, q = min(pairs, key=lambda pq: math.dist(pq[0], pq[1]))
                gap = math.dist(p, q)
                gap_min = max(cfg.gap_min_ratio * ta, 6.0)
                gap_max = cfg.gap_max_ratio * ta
                if not (gap_min <= gap <= gap_max):
                    continue

                pts = [a1, a2, b1, b2]
                far = max(((u, v) for u in pts for v in pts), key=lambda uv: math.dist(uv[0], uv[1]))
                span = math.dist(far[0], far[1])
                if span < 1e-6 or gap > span * 0.6:
                    continue
                # обидва шматки мусять лежати на одній прямій
                (nx1, ny1), (nx2, ny2) = far
                deviated = False
                for px, py in pts:
                    dev = abs((nx2 - nx1) * (ny1 - py) - (nx1 - px) * (ny2 - ny1)) / span
                    if dev > max(1.5, cfg.snap_px * 0.6):
                        deviated = True
                        break
                if deviated:
                    continue
                # щілина має бути МІЖ шматками, а не їх перекриттям
                if abs(math.dist(far[0], p) + gap + math.dist(q, far[1]) - span) > max(2.0, gap * 0.25):
                    continue

                gaps.append((((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0), gap))
                merged = (far[0], far[1], (ta + tb) / 2.0)
                segs = [s for k, s in enumerate(segs) if k not in (i, j)] + [merged]
                changed = True
                break
            if changed:
                break
    return segs, gaps


def _openings_from_points(segments: List[Tuple[Pt, Pt, float]],
                          records: List[Tuple[Pt, float]], kind: str) -> List[Opening]:
    """Прив'язує знайдені прорізи (точка + ширина) до фінальних відрізків стін."""
    out: List[Opening] = []
    for (cx, cy), width in records:
        best_j, best_d, best_t = -1, 1e18, 0.5
        for j, (p1, p2, _t) in enumerate(segments):
            vx, vy = p2[0] - p1[0], p2[1] - p1[1]
            seg_len2 = vx * vx + vy * vy
            if seg_len2 < 1e-9:
                continue
            t = ((cx - p1[0]) * vx + (cy - p1[1]) * vy) / seg_len2
            t_clamped = min(max(t, 0.0), 1.0)
            proj = (p1[0] + vx * t_clamped, p1[1] + vy * t_clamped)
            d = math.hypot(cx - proj[0], cy - proj[1])
            if d < best_d:
                best_d, best_j, best_t = d, j, t_clamped
        if best_j < 0 or best_d > max(segments[best_j][2] * 1.5, 10.0):
            continue
        length = math.dist(segments[best_j][0], segments[best_j][1])
        if length < 1e-6:
            continue
        sill = DEFAULT_WINDOW_SILL_M if kind == "window" else 0.0
        height = DEFAULT_WINDOW_HEIGHT_M if kind == "window" else DEFAULT_DOOR_HEIGHT_M
        out.append(Opening(best_j, best_t, width, kind, sill, height).clamp(length))
    return out


def _segments_touch(a: Tuple[Pt, Pt, float], b: Tuple[Pt, Pt, float], tol: float) -> bool:
    """Чи стикаються два відрізки (кінцями або перетином).

    Допуск підіймається до товщини стіни: два кути товстих зовнішніх стін
    сходяться в одній точці лише на кресленні, а на медіальній осі між ними
    завжди лишається зазор порядку пів-товщини."""
    (a1, a2, ta), (b1, b2, tb) = a, b
    tol = max(tol, (ta + tb) * 0.45)
    for p in (a1, a2):
        for q in (b1, b2):
            if math.dist(p, q) <= tol:
                return True
    # T-стик: кінець одного лежить на тілі іншого
    for p, (q1, q2) in ((a1, (b1, b2)), (a2, (b1, b2)), (b1, (a1, a2)), (b2, (a1, a2))):
        vx, vy = q2[0] - q1[0], q2[1] - q1[1]
        seg_len2 = vx * vx + vy * vy
        if seg_len2 < 1e-9:
            continue
        t = ((p[0] - q1[0]) * vx + (p[1] - q1[1]) * vy) / seg_len2
        if 0.0 <= t <= 1.0:
            proj = (q1[0] + vx * t, q1[1] + vy * t)
            if math.dist(p, proj) <= tol:
                return True
    return False


def _ray_hit(origin: Pt, direction: Pt, target: Tuple[Pt, Pt, float],
             max_distance: float) -> Optional[float]:
    """Відстань від origin уздовж direction до перетину з відрізком target."""
    (qx1, qy1), (qx2, qy2), thickness = target
    rx, ry = direction
    sx, sy = qx2 - qx1, qy2 - qy1
    denominator = rx * sy - ry * sx
    if abs(denominator) < 1e-9:
        return None                       # паралельні
    dx, dy = qx1 - origin[0], qy1 - origin[1]
    t = (dx * sy - dy * sx) / denominator          # уздовж променя
    u = (dx * ry - dy * rx) / denominator          # уздовж цілі
    if t <= 1e-6 or t > max_distance:
        return None
    # Невеликий виліт за край цілі дозволений: у кутку дві стіни сходяться саме
    # своїми кінцями, і вимога влучити строго всередину відрізка їх би розчепила.
    slack = max(2.0, thickness * 0.6) / max(1e-6, math.hypot(sx, sy))
    if not (-slack <= u <= 1.0 + slack):
        return None
    return t


def _extend_dangling_ends(segments: List[Tuple[Pt, Pt, float]], touch_tol: float
                          ) -> List[Tuple[Pt, Pt, float]]:
    """Продовжує «висячі» кінці стін до перетину з іншою стіною.

    БЕЗ ЦЬОГО КРОКУ КІМНАТИ НЕ ЗАМИКАЮТЬСЯ. Медіальна вісь тонкої перегородки
    обривається приблизно за пів-товщини ДО несучої стіни, у яку впирається:
    для зовнішньої стіни 0.4 м це ~14 px розриву. Око цього не бачить, стіни на
    вигляд стоять правильно — але контур кімнати розімкнений, і в макеті
    замість кімнат виходить відкритий лабіринт. Заміряно: 47 висячих кінців на
    8 планах, медіана розриву 19 px.

    Це звичайна CAD-операція «extend to intersection», з двома запобіжниками:
    тягнемо лише вздовж власного напряму відрізка й не далі, ніж на розумну
    частку його довжини — щоб випадковий уламок не простягнувся через пів-плану."""
    if len(segments) < 2:
        return segments
    result = [list(s) for s in segments]
    max_thickness = max(s[2] for s in segments)

    def dangling(point: Pt, skip: int) -> bool:
        for j, (q1, q2, _t) in enumerate(segments):
            if j == skip:
                continue
            vx, vy = q2[0] - q1[0], q2[1] - q1[1]
            length_sq = vx * vx + vy * vy
            if length_sq < 1e-9:
                continue
            t = ((point[0] - q1[0]) * vx + (point[1] - q1[1]) * vy) / length_sq
            t = min(max(t, 0.0), 1.0)
            if math.dist(point, (q1[0] + vx * t, q1[1] + vy * t)) <= touch_tol:
                return False
        return True

    for i, (p1, p2, thickness) in enumerate(segments):
        length = math.dist(p1, p2)
        if length < 1e-6:
            continue
        # 2.2× найтовщої стіни: зміряний розрив у куті доходив до 1.6 товщини
        # (медіальна вісь обривається за пів-товщини від КОЖНОЇ з двох стін, що
        # сходяться, плюс згладжування маски). Ліміт у 1.5 не дотягувався, і кут
        # лишався відкритим — кімната не замикалась. Захист від фантазії —
        # друге обмеження: не більше половини власної довжини відрізка.
        limit = min(max(2.2 * max_thickness, touch_tol * 3.0), length * 0.5)
        for which, point in ((0, p1), (1, p2)):
            if not dangling(point, i):
                continue
            other = p2 if which == 0 else p1
            ux = (point[0] - other[0]) / length
            uy = (point[1] - other[1]) / length
            best = None
            for j, target in enumerate(segments):
                if j == i:
                    continue
                hit = _ray_hit(point, (ux, uy), target, limit)
                if hit is not None and (best is None or hit < best):
                    best = hit
            if best is not None:
                result[i][which] = (point[0] + ux * best, point[1] + uy * best)
    return [((tuple(s[0])), (tuple(s[1])), float(s[2])) for s in result]  # type: ignore[misc]


def _keep_largest_network(segments: List[Tuple[Pt, Pt, float]], tol: float
                          ) -> List[Tuple[Pt, Pt, float]]:
    """Лишає найбільшу ЗВ'ЯЗНУ мережу стін.

    Стіни квартири завжди зчеплені між собою; контури меблів, дуги дверей і
    залишки штриховки — окремі острівці. Без цього фільтра редактор показував у
    ~9 разів більше «стін», ніж є насправді, і користувач мусив чистити їх
    руками замість того, щоб просто підтвердити результат."""
    n = len(segments)
    if n < 3:
        return segments
    parent = list(range(n))

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            if _segments_touch(segments[i], segments[j], tol):
                parent[find(i)] = find(j)

    totals: Dict[int, float] = {}
    for i, (p1, p2, _t) in enumerate(segments):
        root = find(i)
        totals[root] = totals.get(root, 0.0) + math.dist(p1, p2)
    best_total = max(totals.values())
    # Не «лише найбільша група», а «всі помітні»: зовнішня стіна цілком може
    # виявитись відчепленою від внутрішніх через проріз біля кута, і викидати
    # її разом із її вікнами — гірше, ніж лишити зайвий контур меблів.
    keep_roots = {root for root, total in totals.items() if total >= best_total * 0.15}
    kept = [s for i, s in enumerate(segments) if find(i) in keep_roots]
    return kept or segments


def _quantize_thickness(segments: List[Tuple[Pt, Pt, float]], rel_gap: float = 1.25
                        ) -> List[Tuple[Pt, Pt, float]]:
    """Зводить товщини до кількох типів стін (несуча / перегородка / ...).

    У реальному будинку товщин 2-3, а не 15. Медіана по скелету гуляє на ±1 px,
    і на тонкій перегородці в 3 px це ±30%. Групуємо близькі значення і беремо
    зважену за довжиною медіану — шум зникає, а типи стін лишаються різними."""
    if len(segments) < 2:
        return segments
    items = sorted(
        ((t, math.dist(p1, p2), i) for i, (p1, p2, t) in enumerate(segments)),
        key=lambda x: x[0],
    )
    groups: List[List[Tuple[float, float, int]]] = [[items[0]]]
    for item in items[1:]:
        if item[0] <= groups[-1][-1][0] * rel_gap + 0.5:
            groups[-1].append(item)
        else:
            groups.append([item])

    resolved: Dict[int, float] = {}
    for group in groups:
        total = sum(length for _t, length, _i in group)
        if total <= 0:
            value = float(np.median([t for t, _l, _i in group]))
        else:
            # зважена за довжиною медіана: довгі стіни задають тон, короткі не тягнуть
            acc, value = 0.0, group[len(group) // 2][0]
            for t, length, _i in group:
                acc += length
                if acc >= total / 2.0:
                    value = t
                    break
        for _t, _l, idx in group:
            resolved[idx] = float(value)
    return [(p1, p2, resolved.get(i, t)) for i, (p1, p2, t) in enumerate(segments)]


def _heal_junctions(segments: List[Tuple[Pt, Pt, float]], snap: float
                    ) -> List[Tuple[Pt, Pt, float]]:
    """Кластеризує близькі кінці й дотягує «висячі» кінці до сусідніх стін.

    Без цього кроку в макеті лишаються щілини в кутах, і кімнати перетікають
    одна в одну — на друку це виглядає як брак."""
    segs = [list(s) for s in segments]
    endpoints: List[Tuple[int, int]] = []
    for i, (p1, p2, _) in enumerate(segments):
        endpoints.append((i, 0))
        endpoints.append((i, 1))

    def get(idx: int, which: int) -> Pt:
        return tuple(segs[idx][which])  # type: ignore[return-value]

    def put(idx: int, which: int, value: Pt) -> None:
        segs[idx][which] = value

    # 1. кластеризація кінців
    used = [False] * len(endpoints)
    for a in range(len(endpoints)):
        if used[a]:
            continue
        ia, wa = endpoints[a]
        group = [(ia, wa)]
        used[a] = True
        for b in range(a + 1, len(endpoints)):
            if used[b]:
                continue
            ib, wb = endpoints[b]
            if ib == ia:
                continue
            # Допуск залежить від ТОВЩИНИ обох стін. Торці стін плоскі
            # (cap_style=flat), тому в куті їхні кінці мусять збігтися ТОЧНО:
            # розрив навіть у 1.5 px лишає щілину, і кімната не замикається.
            # А обриваються вони приблизно за пів-товщини від справжнього кута,
            # тож при 21 px стінах розбіжність у 12 px — норма, і фіксований
            # допуск у 7 px її не ловив.
            # Беремо ТОНШУ зі стін: інакше товста несуча притягувала б до себе
            # кінці тонких перегородок за 30 px і план «пливе» (IoU підошви
            # падав із 0.76 до 0.54). Змикання кутів забезпечує не це, а
            # квадратні торці у builder._wall_strip.
            corner_tol = max(snap, min(segs[ia][2], segs[ib][2]) * 0.6)
            if math.dist(get(ia, wa), get(ib, wb)) <= corner_tol:
                group.append((ib, wb))
                used[b] = True
        if len(group) > 1:
            cx = sum(get(i, w)[0] for i, w in group) / len(group)
            cy = sum(get(i, w)[1] for i, w in group) / len(group)
            for i, w in group:
                put(i, w, (cx, cy))

    # 2. висячі кінці → проєкція на найближчу стіну (T-з'єднання)
    for i in range(len(segs)):
        for w in (0, 1):
            p = get(i, w)
            best: Optional[Tuple[float, Pt]] = None
            for j in range(len(segs)):
                if j == i:
                    continue
                q1, q2 = get(j, 0), get(j, 1)
                vx, vy = q2[0] - q1[0], q2[1] - q1[1]
                seg_len2 = vx * vx + vy * vy
                if seg_len2 < 1e-9:
                    continue
                t = ((p[0] - q1[0]) * vx + (p[1] - q1[1]) * vy) / seg_len2
                # Зазор пропорційний ТОВЩИНІ: медіальна вісь товстої стіни
                # обривається приблизно за пів-товщини до кута, тож жорсткий
                # допуск у 2% довжини не дотягував зовнішні стіни до кута — і
                # цілий шматок квартири разом із вікнами випадав із мережі.
                slack = max(snap, segs[j][2] * 0.9) / math.sqrt(seg_len2)
                if not (-slack <= t <= 1.0 + slack):
                    continue
                t = min(max(t, 0.0), 1.0)
                proj = (q1[0] + vx * t, q1[1] + vy * t)
                dist = math.dist(p, proj)
                tol = max(snap, segs[j][2] * 0.7)
                if dist <= tol and (best is None or dist < best[0]):
                    best = (dist, proj)
            if best is not None and best[0] > 1e-6:
                put(i, w, best[1])

    return [((tuple(s[0])), (tuple(s[1])), float(s[2])) for s in segs]  # type: ignore[misc]


# ═════════════════════════════════════════════════════════════════════════════
#  Отвори
# ═════════════════════════════════════════════════════════════════════════════
def _assign_openings(segments: List[Tuple[Pt, Pt, float]], mask: Optional[np.ndarray],
                     kind: str, cfg: VectorizeConfig) -> List[Opening]:
    """Компоненти маски дверей/вікон → Opening, прив'язані до найближчої стіни."""
    if mask is None or mask.size == 0 or not segments:
        return []
    import cv2

    m = (mask > 0).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    out: List[Opening] = []
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < 20:
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        pixels = np.argwhere(labels == i)          # (y, x)
        best_j, best_d = -1, 1e18
        for j, (p1, p2, _t) in enumerate(segments):
            vx, vy = p2[0] - p1[0], p2[1] - p1[1]
            seg_len2 = vx * vx + vy * vy
            if seg_len2 < 1e-9:
                continue
            t = ((cx - p1[0]) * vx + (cy - p1[1]) * vy) / seg_len2
            t_clamped = min(max(t, 0.0), 1.0)
            proj = (p1[0] + vx * t_clamped, p1[1] + vy * t_clamped)
            d = math.dist((cx, cy), proj)
            if d < best_d:
                best_d, best_j = d, j
        if best_j < 0:
            continue
        p1, p2, thickness = segments[best_j]
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if length < 1e-6 or best_d > max(thickness * 1.6, 12.0):
            continue
        ux, uy = (p2[0] - p1[0]) / length, (p2[1] - p1[1]) / length
        # ширина отвору = розкид пікселів УЗДОВЖ стіни
        proj_t = (pixels[:, 1] - p1[0]) * ux + (pixels[:, 0] - p1[1]) * uy
        width_px = float(proj_t.max() - proj_t.min())
        center_px = float((proj_t.max() + proj_t.min()) / 2.0)
        if width_px < 4.0:
            continue
        center_t = center_px / length
        if not (-0.05 <= center_t <= 1.05):
            continue
        if kind == "window":
            op = Opening(best_j, min(max(center_t, 0.0), 1.0), width_px, "window",
                         DEFAULT_WINDOW_SILL_M, DEFAULT_WINDOW_HEIGHT_M)
        else:
            op = Opening(best_j, min(max(center_t, 0.0), 1.0), width_px, "door",
                         0.0, DEFAULT_DOOR_HEIGHT_M)
        out.append(op.clamp(length))
    return out


def _openings_from_gaps(segments: List[Tuple[Pt, Pt, float]], wall_mask: np.ndarray,
                        cfg: VectorizeConfig) -> List[Opening]:
    """Резервний спосіб без нейромережі: розрив у суцільній стіні = проріз.

    Працює лише коли маска стін НЕ безперервна (класичний CV якраз дає розриви
    на дверях — там просто немає чорнила)."""
    out: List[Opening] = []
    height, width = wall_mask.shape[:2]
    for j, (p1, p2, thickness) in enumerate(segments):
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if length < cfg.min_wall_len_px * 1.5:
            continue
        ux, uy = (p2[0] - p1[0]) / length, (p2[1] - p1[1]) / length
        steps = max(8, int(length))
        filled = np.zeros(steps, dtype=bool)
        for s in range(steps):
            t = (s + 0.5) / steps
            px, py = p1[0] + ux * length * t, p1[1] + uy * length * t
            ix, iy = int(round(px)), int(round(py))
            if 0 <= ix < width and 0 <= iy < height:
                filled[s] = wall_mask[iy, ix] > 0
        # шукаємо провали
        gap_min = max(cfg.gap_min_ratio * thickness, 6.0)
        gap_max = cfg.gap_max_ratio * thickness
        s = 0
        while s < steps:
            if filled[s]:
                s += 1
                continue
            e = s
            while e < steps and not filled[e]:
                e += 1
            gap_px = (e - s) / steps * length
            if gap_min <= gap_px <= gap_max and s > 0 and e < steps:
                center_t = ((s + e) / 2.0) / steps
                out.append(Opening(j, center_t, gap_px, "door", 0.0,
                                   DEFAULT_DOOR_HEIGHT_M).clamp(length))
            s = e + 1
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Головна функція
# ═════════════════════════════════════════════════════════════════════════════
def masks_to_plan(wall_mask: np.ndarray, door_mask: Optional[np.ndarray] = None,
                  window_mask: Optional[np.ndarray] = None, *,
                  cfg: Optional[VectorizeConfig] = None,
                  m_per_px: float = 0.0,
                  confidence: float = 0.0) -> PlanVector:
    """Маски → PlanVector у ПІКСЕЛЯХ вхідного зображення."""
    import cv2
    from skimage.morphology import skeletonize

    cfg = cfg or VectorizeConfig()
    notes: List[str] = []

    # Пороги в пікселях мусять масштабуватись із роздільною здатністю: 14 px —
    # це справжня стіна на аркуші в 600 px і шумовий вусик на аркуші в 2400 px.
    longest = float(max(wall_mask.shape[0], wall_mask.shape[1]))
    cfg = replace(
        cfg,
        min_wall_len_px=max(cfg.min_wall_len_px, longest * 0.025),
        rdp_tolerance_px=max(cfg.rdp_tolerance_px, longest * 0.003),
        snap_px=max(cfg.snap_px, longest * 0.008),
        min_component_px=max(cfg.min_component_px, int(longest * longest * 0.00015)),
    )

    wall_full = (wall_mask > 0)
    if door_mask is not None:
        wall_full |= (door_mask > 0)
    if window_mask is not None:
        wall_full |= (window_mask > 0)
    solid = _clean_mask(wall_full.astype(np.uint8), cfg)
    if solid.sum() < 50:
        return PlanVector(notes=["Стін не знайдено."], m_per_px=m_per_px)

    dist = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    skeleton = skeletonize(solid.astype(bool)).astype(np.uint8)
    chains = _trace_chains(skeleton)

    raw: List[Tuple[Pt, Pt, float]] = []
    for chain in chains:
        poly = _simplify(chain, cfg.rdp_tolerance_px)
        if len(poly) < 2:
            continue
        chain_arr = np.array(chain, dtype=np.int32)          # (y, x)
        for k in range(len(poly) - 1):
            (x1, y1), (x2, y2) = poly[k], poly[k + 1]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 3.0:
                continue
            # товщина = медіана 2·dist по пікселях скелета в межах цього шматка
            lo = np.array([min(x1, x2) - 2, min(y1, y2) - 2])
            hi = np.array([max(x1, x2) + 2, max(y1, y2) + 2])
            sel = ((chain_arr[:, 1] >= lo[0]) & (chain_arr[:, 1] <= hi[0]) &
                   (chain_arr[:, 0] >= lo[1]) & (chain_arr[:, 0] <= hi[1]))
            # Товщина смуги завширшки w px має max(distanceTransform) = (w+1)/2,
            # тож w = 2·d − 1. Без «−1» тонкі перегородки систематично товщали на
            # чверть, і масштаб виробу «пливе».
            if sel.sum() >= 2:
                vals = dist[chain_arr[sel][:, 0], chain_arr[sel][:, 1]] * 2.0 - 1.0
                thickness = float(np.median(vals))
            else:
                thickness = float(
                    dist[int(round((y1 + y2) / 2)), int(round((x1 + x2) / 2))] * 2.0 - 1.0
                )
            thickness = min(max(thickness, cfg.min_thickness_px), cfg.max_thickness_px)
            raw.append(((float(x1), float(y1)), (float(x2), float(y2)), thickness))

    if not raw:
        return PlanVector(notes=["Не вдалось виділити лінії стін."], m_per_px=m_per_px)

    # Відростки скелета біля перехресть — коротші за власну товщину
    raw = [s for s in raw
           if math.dist(s[0], s[1]) >= max(cfg.min_wall_len_px * 0.5, s[2] * cfg.spur_factor)]

    raw = _drop_outside(raw, _apartment_bbox(solid))
    base_angle = _dominant_angle(raw)
    raw = _snap_ortho(raw, base_angle, cfg.ortho_tolerance_deg)
    # Кластер зміщень беремо від типової товщини стіни: дві РІЗНІ стіни ніколи
    # не стоять ближче, ніж на власну товщину, тож переплутати їх не можна.
    thick_median = sorted(s[2] for s in raw)[len(raw) // 2] if raw else 6.0
    raw = _snap_axis_offsets(raw, base_angle, cfg.ortho_tolerance_deg,
                             max(cfg.snap_px, thick_median * 0.9))
    raw = _merge_parallel_duplicates(raw, cfg.snap_px)
    raw = _merge_collinear(raw, cfg.snap_px)
    raw = [s for s in raw if math.dist(s[0], s[1]) >= cfg.min_wall_len_px]
    raw = _quantize_thickness(raw)
    if not raw:
        return PlanVector(notes=["Усі знайдені стіни надто короткі."], m_per_px=m_per_px)
    raw, gap_records = _bridge_openings(raw, cfg)
    raw = _heal_junctions(raw, cfg.snap_px)
    raw = _extend_dangling_ends(raw, cfg.snap_px)
    raw = _keep_largest_network(raw, cfg.snap_px * 1.6)

    thicknesses = sorted(s[2] for s in raw)
    median_t = thicknesses[len(thicknesses) // 2]
    walls = [
        Wall(x1=p1[0], y1=p1[1], x2=p2[0], y2=p2[1], thickness_m=t, bearing=t > median_t * 1.5)
        for p1, p2, t in raw
    ]

    openings: List[Opening] = []
    openings += _assign_openings(raw, door_mask, "door", cfg)
    openings += _assign_openings(raw, window_mask, "window", cfg)
    if not openings:
        # Класичний шлях: спершу зшиті прорізи (двері як розрив стіни), потім
        # провали всередині суцільної стіни (буває на «порожніх» планах).
        openings = _openings_from_points(raw, gap_records, "door")
        if not openings:
            # Сканування провалів усередині суцільної стіни — лише коли зшивання
            # пеньків нічого не дало: на шумній масці воно знаходить десятки
            # неіснуючих «дверей».
            openings = _openings_from_gaps(raw, solid, cfg)
        if openings:
            notes.append("Отвори визначено за розривами стін (без нейромережі) — перевірте їх.")

    if abs(math.degrees(base_angle)) > 0.4:
        notes.append(f"Головний напрям стін {math.degrees(base_angle):+.1f}°.")

    plan = PlanVector(
        walls=walls, openings=openings, rooms=[],
        scale_source="pixels", m_per_px=m_per_px,
        image_size_px=(int(wall_mask.shape[1]), int(wall_mask.shape[0])),
        confidence=float(confidence), notes=notes,
    )
    return plan.sanitize()
