"""D4 GPX-ТРЕК: маршрут користувача (біг/похід/вело) як підвищена «дорога»
поверх мапи. Фронт парсить .gpx сам і шле gpx_track=[[lon,lat],...]; тут трек
перетворюється на буферизований полігон у локальних метрах і:
  • повний пайплайн — друкована «шапка» по рельєфу (create_road_surface_cap),
    шар ПОВЕРХ терейну без булевих врізань (перекриття зварює слайсер);
  • flat-пайплайн — звичайний плаский шар над дорогами.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import trimesh
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry

TRACK_COLOR = [222, 28, 28, 255]  # ЧЕРВОНИЙ — маршрут має чітко виділятись (AMS-шар)
MAX_GPX_POINTS = 8000


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


def _snap_coords_to_roads(coords: list, road_lines_local: list, snap_threshold_m: float) -> list:
    """Притягує точки треку до найближчої осі дороги (місто): для кожної точки
    шукаємо найближчу дорогу; якщо ближче за поріг — проєктуємо на неї, інакше
    лишаємо як є (бездоріжжя/стежка). Так маршрут іде ПО ДОРОГАХ де можливо."""
    if not road_lines_local or snap_threshold_m <= 0:
        return coords
    try:
        from shapely.geometry import Point
        out = []
        for c in coords:
            p = Point(c[0], c[1])
            best = None
            best_d = snap_threshold_m
            for rl in road_lines_local:
                d = rl.distance(p)
                if d < best_d:
                    best_d = d
                    best = rl
            if best is not None:
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
    snap_threshold_m = (18.0 if road_lines_local else 0.0)  # місто: притягувати до доріг у радіусі ~18м
    line = gpx_track_to_local_geometry(
        gpx_track, global_center,
        simplify_m=simplify_m, smooth_passes=2,
        road_lines_local=road_lines_local, snap_threshold_m=snap_threshold_m,
    )
    if line is None:
        return None
    try:
        # ROUND caps+joins (cap_style=1, join_style=1 у shapely = круглі) + buffer(0)
        # ОДРАЗУ прибирає самоперетини згладженого треку до кліпу.
        poly = line.buffer(half_w_m, cap_style=1, join_style=1).buffer(0)
        if zone_polygon_local is not None and not getattr(zone_polygon_local, "is_empty", True):
            poly = poly.intersection(zone_polygon_local).buffer(0)
        if poly is None or poly.is_empty or float(poly.area) <= 0:
            print("[GPX] track outside the selected zone — layer skipped")
            return None
        return poly
    except Exception as exc:
        print(f"[GPX] buffer failed: {exc}")
        return None


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
    """Друкована «шапка» треку по рельєфу: верх = terrain + raise_mm, низ
    втоплений у терейн (~0.8мм) — нічого не плаває і не потребує булевих.
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
        return None
    try:
        from services.road_processor import create_road_surface_cap

        raise_m = max(float(raise_mm), 0.2) / max(float(scale_factor), 1e-9)
        embed_m = 0.8 / max(float(scale_factor), 1e-9)
        meshes = []
        polys = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
        for part in polys:
            if part.is_empty or float(part.area) <= 0:
                continue
            cap = create_road_surface_cap(
                part,
                terrain_provider,
                scale_factor=float(scale_factor),
                top_z_offset=raise_m,
                cap_thickness_m=raise_m + embed_m,
            )
            if cap is not None and cap.faces is not None and len(cap.faces) > 0:
                meshes.append(cap)
        if not meshes:
            return None
        mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
        mesh.visual = trimesh.visual.ColorVisuals(
            face_colors=np.tile(TRACK_COLOR, (len(mesh.faces), 1))
        )
        print(f"[GPX] Track mesh on terrain: {len(mesh.faces)} faces, raise={raise_mm}mm")
        return mesh
    except Exception as exc:
        print(f"[GPX] terrain track mesh failed (non-fatal): {exc}")
        return None
