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


def gpx_track_to_local_geometry(
    gpx_track: Any,
    global_center: Any,
) -> Optional[BaseGeometry]:
    """[[lon,lat],...] → LineString у локальних метрах (frame global_center).
    Розриви GPS (стрибок > 500м між сусідніми точками) ріжуть трек на сегменти."""
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
        lines = [LineString(seg) for seg in segments if len(seg) >= 2]
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
) -> Optional[BaseGeometry]:
    """Буферизований полігон треку, обрізаний по зоні. width_mm — у model-мм."""
    line = gpx_track_to_local_geometry(gpx_track, global_center)
    if line is None:
        return None
    half_w_m = max(float(width_mm), 0.4) / 2.0 / max(float(scale_factor), 1e-9)
    try:
        poly = line.buffer(half_w_m, cap_style=1, join_style=1)
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
) -> Optional[trimesh.Trimesh]:
    """Друкована «шапка» треку по рельєфу: верх = terrain + raise_mm, низ
    втоплений у терейн (~0.8мм) — нічого не плаває і не потребує булевих."""
    poly = build_gpx_track_polygon(
        gpx_track=gpx_track,
        global_center=global_center,
        zone_polygon_local=zone_polygon_local,
        scale_factor=scale_factor,
        width_mm=width_mm,
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
