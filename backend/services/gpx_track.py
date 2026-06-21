"""D4 GPX-ТРЕК: маршрут користувача (біг/похід/вело) як ВРІЗАНИЙ (recessed) інлей
у поверхню мапи (раніше був підвищений над дорогами — змінено). Фронт парсить .gpx
сам і шле gpx_track=[[lon,lat],...]; тут трек перетворюється на буферизований
полігон у локальних метрах і:
  • повний пайплайн — врізана червона вставка по рельєфу (build_gpx_track_inlay_on_terrain),
    верх flush з поверхнею, тіло втоплене у базу на ~recess;
  • flat-пайплайн — плаский врізаний інлей (верх flush з base_top).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import trimesh
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry

TRACK_COLOR = [222, 28, 28, 255]  # ЧЕРВОНИЙ — маршрут має чітко виділятись (AMS-шар)
MAX_GPX_POINTS = 8000

# Phase 2 toggle: road-snap і де-хвилястість умикаються КАЛЛЕРОМ через параметри
# (snap_threshold_m>0 / vw_area_threshold>0). Цей прапор — глобальний kill-switch,
# щоб у разі регресії можна було вимкнути ВСЮ Phase-2 поведінку, не чіпаючи каллери,
# і дефолтний шлях (без road_lines_local) лишився ІДЕНТИЧНИМ старому (нуль регресій).
GPX_PHASE2_ENABLED = True


def _flatten_road_lines(road_lines_local: Any) -> list:
    """Нормалізує road_lines_local у плаский список shapely LineString.
    Каллер може передати list[LineString], list[MultiLineString] або суміш —
    розкладаємо MultiLineString на компоненти, відкидаємо порожнє/не-лінії,
    щоб _snap_coords_to_roads працював з однорідними геометріями."""
    out: list = []
    if not road_lines_local:
        return out
    try:
        items = list(road_lines_local)
    except TypeError:
        items = [road_lines_local]
    for rl in items:
        if rl is None or getattr(rl, "is_empty", False):
            continue
        geoms = getattr(rl, "geoms", None)
        if geoms is not None and not isinstance(rl, LineString):
            for g in geoms:
                if g is not None and not getattr(g, "is_empty", True) and getattr(g, "length", 0) > 0:
                    out.append(g)
        elif getattr(rl, "length", 0) > 0:
            out.append(rl)
    return out


def _chaikin_smooth(coords: list, passes: int) -> list:
    """Згладження Чайкіна (corner-cutting): кожна ланка ділиться 25/75 → плавна
    крива без гострих GPS-зламів. Кінці фіксовані. Безпечно для друку."""
    if passes <= 0 or len(coords) < 3:
        return coords
    out = list(coords)
    for _ in range(int(passes)):
        if len(out) < 3:
            break
        new = [out[0]]
        for i in range(len(out) - 1):
            p, q = out[i], out[i + 1]
            new.append((0.75 * p[0] + 0.25 * q[0], 0.75 * p[1] + 0.25 * q[1]))
            new.append((0.25 * p[0] + 0.75 * q[0], 0.25 * p[1] + 0.75 * q[1]))
        new.append(out[-1])
        out = new
    return out


def _visvalingam_simplify(coords: list, area_threshold: float) -> list:
    """Visvalingam-Whyatt спрощення: ітеративно прибирає внутрішню точку з НАЙМЕНШОЮ
    площею трикутника (сусід-точка-сусід), поки min-площа < area_threshold (м²).
    Краще за Douglas-Peucker прибирає дрібні «зубці»/хвилястість сирого GPS, зберігаючи
    загальну форму → друкований трек, що НЕ виляє і не самоперетинається."""
    if area_threshold <= 0 or len(coords) <= 3:
        return coords
    pts = [(float(c[0]), float(c[1])) for c in coords]

    def _tri_area(a, b, c) -> float:
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) * 0.5

    guard = len(pts) + 4
    while len(pts) > 3 and guard > 0:
        guard -= 1
        min_i, min_a = -1, float("inf")
        for i in range(1, len(pts) - 1):
            a = _tri_area(pts[i - 1], pts[i], pts[i + 1])
            if a < min_a:
                min_a, min_i = a, i
        if min_i < 0 or min_a >= area_threshold:
            break
        pts.pop(min_i)
    return pts


def _snap_coords_to_roads(coords: list, road_lines_local: Any, snap_threshold_m: float) -> list:
    """Притягує точки треку до найближчої осі дороги (місто): для кожної точки
    шукаємо найближчу дорогу; якщо ближче за поріг — проєктуємо на неї (nearest
    point на лінії через project+interpolate), інакше лишаємо як є (бездоріжжя/
    стежка/парк). Так маршрут іде ПО ДОРОГАХ де можливо, але далекі від доріг
    точки (природа) НЕ зміщуються.

    Phase 2: (1) MultiLineString розкладається на компоненти; (2) для швидкого
    пошуку найближчої дороги використовуємо STRtree (просторовий індекс) з
    fallback на лінійний скан; (3) точку, що НЕ ближче за поріг, лишаємо без змін."""
    if not road_lines_local or snap_threshold_m <= 0 or not GPX_PHASE2_ENABLED:
        return coords
    lines = _flatten_road_lines(road_lines_local)
    if not lines:
        return coords
    try:
        from shapely.geometry import Point

        # Просторовий індекс прискорює пошук на великих містах (сотні вулиць).
        # query-радіус = поріг; STRtree API різниться між shapely 1.x/2.x →
        # graceful fallback на лінійний скан, якщо індекс недоступний.
        tree = None
        try:
            from shapely.strtree import STRtree
            tree = STRtree(lines)
        except Exception:
            tree = None

        def _candidates(p):
            if tree is None:
                return lines
            try:
                # shapely 2.x: query() повертає індекси; 1.x — геометрії.
                res = tree.query(p.buffer(snap_threshold_m))
                cand = []
                for r in res:
                    if isinstance(r, (int, np.integer)):
                        cand.append(lines[int(r)])
                    else:
                        cand.append(r)
                return cand if cand else lines
            except Exception:
                return lines

        out = []
        for c in coords:
            p = Point(c[0], c[1])
            best = None
            best_d = snap_threshold_m
            for rl in _candidates(p):
                d = rl.distance(p)
                if d < best_d:
                    best_d = d
                    best = rl
            if best is not None:
                # Проєкція на найближчу точку осі дороги (range-clamped до кінців).
                proj = best.interpolate(best.project(p))
                out.append((proj.x, proj.y))
            else:
                out.append((c[0], c[1]))
        return out
    except Exception as exc:
        print(f"[GPX] snap-to-roads failed (non-fatal): {exc}")
        return coords


def gpx_track_to_local_geometry(
    gpx_track: Any,
    global_center: Any,
    *,
    simplify_m: float = 0.0,
    smooth_passes: int = 0,
    road_lines_local: Optional[list] = None,
    snap_threshold_m: float = 0.0,
    vw_area_threshold: float = 0.0,
) -> Optional[BaseGeometry]:
    """[[lon,lat],...] → LineString у локальних метрах (frame global_center).
    Розриви GPS (стрибок > 500м між сусідніми точками) ріжуть трек на сегменти.
    Для 3D-друку сирий GPS згладжується: (1) опційний snap до доріг міста,
    (2) Douglas-Peucker спрощення (прибирає тремтіння GPS), (3) Чайкін-згладження
    (прибирає хвилястість/зламки). Без цього буфер сирого треку — хвилястий і
    самоперетинається, погано друкується."""
    if not gpx_track:
        return None
    try:
        pts = []
        for item in list(gpx_track)[:MAX_GPX_POINTS]:
            lon, lat = float(item[0]), float(item[1])
            if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                continue
            # WGS84 → UTM → локальні метри (to_local приймає UTM, НЕ lon/lat!)
            x_utm, y_utm = global_center.to_utm(lon, lat)
            pts.append(global_center.to_local(x_utm, y_utm))
        if len(pts) < 2:
            return None
        segments: list[list[tuple[float, float]]] = [[pts[0]]]
        for prev, cur in zip(pts, pts[1:]):
            dx = cur[0] - prev[0]
            dy = cur[1] - prev[1]
            if (dx * dx + dy * dy) > 500.0 * 500.0:
                segments.append([cur])
            else:
                segments[-1].append(cur)

        lines = []
        for seg in segments:
            if len(seg) < 2:
                continue
            coords = seg
            # 1) Притягання до доріг (місто) — ДО спрощення, щоб ловити осі
            if road_lines_local and snap_threshold_m > 0:
                coords = _snap_coords_to_roads(coords, road_lines_local, snap_threshold_m)
            # 2) Douglas-Peucker — прибрати GPS-тремтіння дрібніше за tolerance
            if simplify_m and simplify_m > 0 and len(coords) >= 3:
                try:
                    simp = LineString(coords).simplify(float(simplify_m), preserve_topology=False)
                    sc = list(simp.coords)
                    if len(sc) >= 2:
                        coords = sc
                except Exception:
                    pass
            # 2b) Visvalingam — прибрати дрібні «зубці»/звивистість, що DP лишає на
            # зиґзаґах (трек перестає виляти й налазити на сусідні фічі/рельєф).
            if vw_area_threshold > 0 and len(coords) >= 4:
                coords = _visvalingam_simplify(coords, vw_area_threshold)
            # 3) Чайкін — згладити злами у плавну криву
            if smooth_passes and smooth_passes > 0:
                coords = _chaikin_smooth(coords, smooth_passes)
            if len(coords) >= 2:
                lines.append(LineString(coords))
        if not lines:
            return None
        return lines[0] if len(lines) == 1 else MultiLineString(lines)
    except Exception as exc:
        print(f"[GPX] track→local failed: {exc}")
        return None


def _ensure_valid_polygon(poly: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    """Гарантує валідний (is_valid) і непорожній полігон: спершу buffer(0), якщо
    лишився невалідним — make_valid (shapely 2.x) → відфільтрувати лише полігональні
    частини. Захищає від самоперетинів/виродження після snap+buffer+intersection."""
    if poly is None or getattr(poly, "is_empty", True):
        return None
    try:
        if not poly.is_valid:
            poly = poly.buffer(0)
    except Exception:
        pass
    try:
        if poly is not None and not poly.is_empty and not poly.is_valid:
            from shapely.validation import make_valid
            fixed = make_valid(poly)
            # make_valid може повернути GeometryCollection (лінії+полігони) —
            # лишаємо лише полігональну складову (площа треку).
            gtype = getattr(fixed, "geom_type", "")
            if gtype == "GeometryCollection":
                from shapely.geometry import MultiPolygon, Polygon as _Pg
                parts = [g for g in fixed.geoms if isinstance(g, _Pg) and not g.is_empty]
                fixed = (parts[0] if len(parts) == 1 else MultiPolygon(parts)) if parts else None
            poly = fixed
    except Exception:
        pass
    if poly is None or getattr(poly, "is_empty", True) or float(getattr(poly, "area", 0.0)) <= 0:
        return None
    return poly


def build_gpx_track_polygon(
    *,
    gpx_track: Any,
    global_center: Any,
    zone_polygon_local: Optional[BaseGeometry],
    scale_factor: float,
    width_mm: float = 1.2,
    road_lines_local: Optional[list] = None,
) -> Optional[BaseGeometry]:
    """Буферизований полігон треку, обрізаний по зоні. width_mm — у model-мм.
    Трек спершу спрощується+згладжується (друкований, не хвилястий); якщо передано
    road_lines_local (місто) — притягується до доріг."""
    sf = max(float(scale_factor), 1e-9)
    # Друкована мінімальна ширина: щонайменше 1.0мм моделі (>2× сопло 0.4) — тонше
    # друкується погано/рветься. half-ширина у світових метрах.
    eff_width_mm = max(float(width_mm), 1.0)
    half_w_m = eff_width_mm / 2.0 / sf
    # Спрощення/згладження масштабовані: прибираємо тремтіння дрібніше за ~0.6мм
    # моделі (world = 0.6/sf), але не грубіше за пів-ширини треку.
    simplify_m = min(max(0.6 / sf, 0.5), half_w_m * 1.2)
    # Поріг snap: ~18м у місті, але не менше за пів-ширину треку (щоб трек, що йде
    # майже по дорозі, гарантовано підхопився) і не агресивніше за ~25м.
    snap_threshold_m = (max(18.0, half_w_m) if road_lines_local else 0.0)
    snap_threshold_m = min(snap_threshold_m, 25.0) if snap_threshold_m > 0 else 0.0
    # Visvalingam де-хвилястість: прибираємо зиґзаґи дрібніші за ~пів-ширину треку.
    # МІСТО (є дороги) — легше (трек уздовж осей доріг); ПРИРОДА (без доріг) — агресивніше
    # + більше Чайкіна, бо лісові/гірські треки звивистіші й гірше друкуються.
    _is_city = bool(road_lines_local)
    vw_area_threshold = (half_w_m * (0.7 if _is_city else 1.3)) ** 2 * 0.5
    line = gpx_track_to_local_geometry(
        gpx_track, global_center,
        simplify_m=simplify_m, smooth_passes=(2 if _is_city else 3),
        road_lines_local=road_lines_local, snap_threshold_m=snap_threshold_m,
        vw_area_threshold=vw_area_threshold,
    )
    if line is None:
        return None
    try:
        # ROUND caps+joins (cap_style=1, join_style=1 у shapely = круглі) + buffer(0)
        # ОДРАЗУ прибирає самоперетини згладженого треку до кліпу.
        poly = line.buffer(half_w_m, cap_style=1, join_style=1).buffer(0)
        # ВАЛІДНІСТЬ: snap до доріг може породити самоперетини/«вісімки» — make_valid.
        poly = _ensure_valid_polygon(poly)
        if poly is None:
            print("[GPX] track buffer collapsed/empty — layer skipped")
            return None
        if zone_polygon_local is not None and not getattr(zone_polygon_local, "is_empty", True):
            poly = poly.intersection(zone_polygon_local)
            poly = _ensure_valid_polygon(poly)
        if poly is None:
            print("[GPX] track outside the selected zone — layer skipped")
            return None
        # Захист від виродження у нитку: після кліпу площа має покривати ≥ мінімальну
        # друковану ширину хоча б на короткому відрізку (half_w_m² як грубий поріг).
        if float(poly.area) <= (half_w_m * half_w_m) * 0.25:
            print("[GPX] track polygon too thin after clip — layer skipped")
            return None
        return poly
    except Exception as exc:
        print(f"[GPX] buffer failed: {exc}")
        return None


def build_gpx_track_inlay_on_terrain(
    *,
    gpx_track: Any,
    global_center: Any,
    zone_polygon_local: Optional[BaseGeometry],
    terrain_provider: Any,
    scale_factor: float,
    width_mm: float = 1.2,
    recess_mm: float = 0.6,
    clearance_mm: float = 0.2,
    road_lines_local: Optional[list] = None,
) -> tuple:
    """ВРІЗАНИЙ маршрут (інлей): повертає (insert_mesh, groove_cutter).
      • insert_mesh — червона вставка, ВЕРХ якої співпадає з поверхнею (flush,
        НЕ виступає), низ втоплений у рельєф; це окрема деталь «Track».
      • groove_cutter — трохи ширший суцільний інструмент (по track+clearance,
        від трохи над поверхнею до глибини recess) для boolean-вирізу жолоба в
        рельєфі, щоб вставка сиділа В каналі (а не зверху). Каллер ріже рельєф
        цим cutter-ом з graceful fallback (якщо boolean впав — лишається flush).
    road_lines_local (місто) — притягнути трек до доріг де можливо."""
    poly = build_gpx_track_polygon(
        gpx_track=gpx_track,
        global_center=global_center,
        zone_polygon_local=zone_polygon_local,
        scale_factor=scale_factor,
        width_mm=width_mm,
        road_lines_local=road_lines_local,
    )
    if poly is None or terrain_provider is None:
        return None, None
    try:
        from services.road_processor import create_road_surface_cap

        sf = max(float(scale_factor), 1e-9)
        recess_m = max(float(recess_mm), 0.3) / sf
        clearance_m = max(float(clearance_mm), 0.0) / sf
        cutter_top_m = 0.6 / sf       # cutter трохи НАД поверхнею (чистий зріз)
        cutter_bottom_extra = 0.4 / sf
        insert_embed = 0.4 / sf        # вставка трохи глибша за recess для зварки

        polys = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
        ins_meshes, cut_meshes = [], []
        for part in polys:
            if part.is_empty or float(part.area) <= 0:
                continue
            # ВСТАВКА: верх = поверхня (flush, top_z_offset=0), низ = -recess-embed
            ins = create_road_surface_cap(
                part, terrain_provider, scale_factor=sf,
                top_z_offset=0.0, cap_thickness_m=recess_m + insert_embed,
            )
            if ins is not None and ins.faces is not None and len(ins.faces) > 0:
                ins_meshes.append(ins)
            # CUTTER: ширший на clearance, від cutter_top над поверхнею до recess нижче
            cut_part = part.buffer(clearance_m, join_style=1) if clearance_m > 0 else part
            cut = create_road_surface_cap(
                cut_part, terrain_provider, scale_factor=sf,
                top_z_offset=cutter_top_m,
                cap_thickness_m=cutter_top_m + recess_m + cutter_bottom_extra,
            )
            if cut is not None and cut.faces is not None and len(cut.faces) > 0:
                cut_meshes.append(cut)
        if not ins_meshes:
            return None, None
        insert = ins_meshes[0] if len(ins_meshes) == 1 else trimesh.util.concatenate(ins_meshes)
        # INSERT — це ДРУКОВАНА червона вставка. create_road_surface_cap робить лише
        # fix_normals, тож тут даємо ту саму послідовність ремонту, що й cutter нижче
        # (інакше негерметична/вироджена вставка ламає слайс). Ремонтуємо ДО фарбування,
        # бо merge/update_faces змінюють кількість фейсів → tile фарб має йти на новий лік.
        try:
            insert.merge_vertices()
            insert.update_faces(insert.nondegenerate_faces())
            insert.remove_unreferenced_vertices()
            trimesh.repair.fix_winding(insert)
            trimesh.repair.fill_holes(insert)
            insert.fix_normals()
            if not bool(insert.is_volume):
                print("[GPX] track insert not watertight after repair — slicer may auto-fix")
        except Exception:
            pass
        insert.visual = trimesh.visual.ColorVisuals(
            face_colors=np.tile(TRACK_COLOR, (len(insert.faces), 1))
        )
        cutter = None
        if cut_meshes:
            cutter = cut_meshes[0] if len(cut_meshes) == 1 else trimesh.util.concatenate(cut_meshes)
            # Cutter МУСИТЬ бути замкненим обʼємом для manifold-difference ("not a
            # volume" інакше). Ремонтуємо: merge+winding+fill_holes.
            try:
                cutter.merge_vertices()
                cutter.update_faces(cutter.nondegenerate_faces())
                cutter.remove_unreferenced_vertices()
                trimesh.repair.fix_winding(cutter)
                trimesh.repair.fill_holes(cutter)
                cutter.fix_normals()
                if not bool(cutter.is_volume):
                    print("[GPX] track cutter not watertight after repair — groove may be skipped")
            except Exception:
                pass
        print(f"[GPX] Track INLAY on terrain: insert {len(insert.faces)} faces, recess={recess_mm}mm (flush top)")
        return insert, cutter
    except Exception as exc:
        print(f"[GPX] terrain track inlay failed (non-fatal): {exc}")
        return None, None


def build_gpx_track_mesh_on_terrain(
    *,
    gpx_track: Any,
    global_center: Any,
    zone_polygon_local: Optional[BaseGeometry],
    terrain_provider: Any,
    scale_factor: float,
    width_mm: float = 1.2,
    raise_mm: float = 0.6,
    road_lines_local: Optional[list] = None,
) -> Optional[trimesh.Trimesh]:
    """Сумісність: піднята «шапка» (raised). Новий код — build_gpx_track_inlay_on_terrain."""
    insert, _ = build_gpx_track_inlay_on_terrain(
        gpx_track=gpx_track, global_center=global_center,
        zone_polygon_local=zone_polygon_local, terrain_provider=terrain_provider,
        scale_factor=scale_factor, width_mm=width_mm,
        recess_mm=raise_mm, road_lines_local=road_lines_local,
    )
    return insert
