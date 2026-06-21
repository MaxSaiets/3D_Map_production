from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import trimesh
import mapbox_earcut
from geopandas import GeoDataFrame
from shapely import affinity
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, triangulate, unary_union

from services.building_processor import process_buildings
from services.buildings_pipeline import split_building_parts_from_parent_footprints
from services.detail_layer_utils import MICRO_REGION_THRESHOLD_MM, model_mm_to_world_m
from services.export_pipeline import ExportPipelineResult, export_generation_outputs


LAYER_COLORS = {
    "base": [242, 242, 242, 255],     # БІЛА основа (фідбек власника 2026-06-17)
    "roads": [20, 20, 20, 255],       # ЧОРНІ дороги
    "buildings": [242, 242, 242, 255], # БІЛІ будинки (як основа)
    "water": [47, 111, 214, 255],     # СИНЯ вода
    "parks": [63, 122, 63, 255],      # темніший зелений
    # Фідбек власника (2026-06-14): ободок і текст — ЧОРНІ (друкуються чорним
    # філаментом). Беремо дуже темний графіт [25,25,25] замість чистого [0,0,0]:
    # друкується як чорний, але у 3D-превʼю видно форму (чистий чорний без світла
    # зливається у пляму — стара скарга Роми). На печать це фактично чорний.
    "rim": [25, 25, 25, 255],
    "text": [25, 25, 25, 255],
    # Маркер «особливе місце» — теплий теракотовий (виділяється на карті, як
    # шпилька-маркер; той самий тон, що GPX-трек #c44110).
    "marker": [196, 65, 16, 255],
    # Підсвічений будинок (твій дім/орієнтир) — ЧЕРВОНИЙ: друкується окремою
    # вставною деталлю іншим філаментом, яскраво виділяється на білій карті
    # (фідбек власника: «зробив його червоним»).
    "highlight": [206, 38, 38, 255],
    # Визначні місця (церкви/вежі/історичні/музеї) — БРОНЗА-ЯНТАР, окремий
    # філамент: виділяються на білій карті, синхрон з model_exporter COLOR_MAP.
    "landmark": [201, 144, 47, 255],
}

MIN_KEYCHAIN_PRINT_FEATURE_MM = 0.4
# Мінімальна товщина штриха тексту для FDM 0.4мм сопла. Research (Mandarin3D,
# JLC3DP, Bambu wiki): надійний друкований штрих ≥2× діаметра сопла = 0.8мм,
# інакше тонкі частини літер не пропечатуються. Тонкі гліфи дилейтимо до цього.
MIN_KEYCHAIN_TEXT_STROKE_MM = 0.8
# Оптимальна висота піднятого (embossed) тексту = кратна висоті шару. 0.6мм =
# 3 шари по 0.2мм — чисті межі зміни кольору в multi-color 3MF.
KEYCHAIN_TEXT_RAISE_MM = 0.6


_FONT_5X7 = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}

_CYR_TO_LAT = str.maketrans(
    {
        "А": "A", "Б": "B", "В": "V", "Г": "H", "Ґ": "G", "Д": "D", "Е": "E", "Є": "YE",
        "Ж": "ZH", "З": "Z", "И": "Y", "І": "I", "Ї": "YI", "Й": "Y", "К": "K", "Л": "L",
        "М": "M", "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U",
        "Ф": "F", "Х": "KH", "Ц": "TS", "Ч": "CH", "Ш": "SH", "Щ": "SHCH", "Ь": "",
        "Ю": "YU", "Я": "YA",
    }
)


def _model_mm_to_world_m(value_mm: float, scale_factor: float) -> float:
    if scale_factor <= 0:
        return 0.0
    return float(value_mm) / float(scale_factor)


def _clean_polygonal_geometry(geometry: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return None
    try:
        geometry = geometry.buffer(0)
    except Exception:
        pass
    if geometry is None or getattr(geometry, "is_empty", True):
        return None
    geom_type = str(getattr(geometry, "geom_type", "") or "")
    if "Polygon" in geom_type or isinstance(geometry, GeometryCollection):
        return geometry
    return None


def _iter_polygons(geometry: Optional[BaseGeometry]) -> Iterable[Polygon]:
    geometry = _clean_polygonal_geometry(geometry)
    if geometry is None:
        return
    if isinstance(geometry, Polygon):
        if not geometry.is_empty and geometry.area > 0:
            yield geometry
        return
    if isinstance(geometry, MultiPolygon):
        for part in geometry.geoms:
            if not part.is_empty and part.area > 0:
                yield part
        return
    if isinstance(geometry, GeometryCollection):
        for part in geometry.geoms:
            yield from _iter_polygons(part)


def _with_color(mesh: Optional[trimesh.Trimesh], color: list[int]) -> Optional[trimesh.Trimesh]:
    if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
        return None
    mesh.visual = trimesh.visual.ColorVisuals(face_colors=np.tile(color, (len(mesh.faces), 1)))
    return mesh


def _extrude_polygon_prism(poly: Polygon, thickness_m: float) -> Optional[trimesh.Trimesh]:
    if poly is None or poly.is_empty or thickness_m <= 0:
        return None
    try:
        poly = poly.buffer(0)
    except Exception:
        pass
    if poly is None or poly.is_empty or float(poly.area) <= 0:
        return None

    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    vertex_index: dict[tuple[float, float, float], int] = {}
    cap_edge_counts: dict[tuple[int, int], int] = {}

    def _vertex_id(x: float, y: float, z: float) -> int:
        key = (round(float(x), 8), round(float(y), 8), round(float(z), 8))
        found = vertex_index.get(key)
        if found is not None:
            return found
        vertex_index[key] = len(vertices)
        vertices.append((float(x), float(y), float(z)))
        return len(vertices) - 1

    def _remember_cap_edges(face: list[int]) -> None:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((int(a), int(b))))
            cap_edge_counts[key] = cap_edge_counts.get(key, 0) + 1

    try:
        cap_points, triangle_indices = trimesh.creation.triangulate_polygon(
            poly,
            triangle_args="p",
            engine="triangle",
        )
        cap_points = [(float(x), float(y)) for x, y in np.asarray(cap_points, dtype=float)]
        triangle_indices = np.asarray(triangle_indices, dtype=np.int64).reshape((-1, 3))
        for i0, i1, i2 in triangle_indices:
            pts = [cap_points[int(i0)], cap_points[int(i1)], cap_points[int(i2)]]
            bottom = [_vertex_id(x, y, 0.0) for x, y in pts]
            top = [_vertex_id(x, y, float(thickness_m)) for x, y in pts]
            bottom_face = [bottom[2], bottom[1], bottom[0]]
            faces.append(bottom_face)
            _remember_cap_edges(bottom_face)
            faces.append([top[0], top[1], top[2]])
    except Exception:
        cap_points: list[tuple[float, float]] = []
        ring_ends: list[int] = []

        def _append_ring(coords: Any) -> None:
            for x, y in list(coords)[:-1]:
                cap_points.append((float(x), float(y)))
            ring_ends.append(len(cap_points))

        try:
            _append_ring(poly.exterior.coords)
            for interior in poly.interiors:
                _append_ring(interior.coords)
            point_array = np.asarray(cap_points, dtype=np.float32)
            ring_array = np.asarray(ring_ends, dtype=np.uint32)
            triangle_indices = np.asarray(
                mapbox_earcut.triangulate_float32(point_array, ring_array),
                dtype=np.int64,
            ).reshape((-1, 3))
            for i0, i1, i2 in triangle_indices:
                pts = [cap_points[int(i0)], cap_points[int(i1)], cap_points[int(i2)]]
                bottom = [_vertex_id(x, y, 0.0) for x, y in pts]
                top = [_vertex_id(x, y, float(thickness_m)) for x, y in pts]
                bottom_face = [bottom[2], bottom[1], bottom[0]]
                faces.append(bottom_face)
                _remember_cap_edges(bottom_face)
                faces.append([top[0], top[1], top[2]])
        except Exception:
            for tri in triangulate(poly):
                if tri is None or tri.is_empty or float(tri.area) <= 1e-10:
                    continue
                try:
                    if not poly.covers(tri.representative_point()):
                        continue
                except Exception:
                    continue
                coords = list(tri.exterior.coords)
                if len(coords) < 4:
                    continue
                pts = coords[:3]
                bottom = [_vertex_id(x, y, 0.0) for x, y in pts]
                top = [_vertex_id(x, y, float(thickness_m)) for x, y in pts]
                bottom_face = [bottom[2], bottom[1], bottom[0]]
                faces.append(bottom_face)
                _remember_cap_edges(bottom_face)
                faces.append([top[0], top[1], top[2]])

    for (b0, b1), count in list(cap_edge_counts.items()):
        if count != 1:
            continue
        x0, y0, _ = vertices[b0]
        x1, y1, _ = vertices[b1]
        t0 = _vertex_id(x0, y0, float(thickness_m))
        t1 = _vertex_id(x1, y1, float(thickness_m))
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])

    if not vertices or not faces:
        return None
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=np.int64), process=False)
    try:
        mesh.merge_vertices(digits_vertex=8)
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_winding(mesh)
        mesh.fix_normals()
    except Exception:
        pass
    if not bool(getattr(mesh, "is_watertight", False)):
        try:
            verts = mesh.vertices.tolist()
            faces_arr = mesh.faces.tolist()
            lookup = {
                (round(float(x), 8), round(float(y), 8), round(float(z), 8)): idx
                for idx, (x, y, z) in enumerate(verts)
            }

            def _ensure_vertex(x: float, y: float, z: float) -> int:
                key = (round(float(x), 8), round(float(y), 8), round(float(z), 8))
                found = lookup.get(key)
                if found is not None:
                    return found
                lookup[key] = len(verts)
                verts.append([float(x), float(y), float(z)])
                return len(verts) - 1

            edges = mesh.edges_sorted
            unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
            for edge, count in zip(unique_edges, counts):
                if int(count) != 1:
                    continue
                a, b = int(edge[0]), int(edge[1])
                va = mesh.vertices[a]
                vb = mesh.vertices[b]
                if abs(float(va[2])) > 1e-7 or abs(float(vb[2])) > 1e-7:
                    continue
                t0 = _ensure_vertex(float(va[0]), float(va[1]), float(thickness_m))
                t1 = _ensure_vertex(float(vb[0]), float(vb[1]), float(thickness_m))
                faces_arr.append([a, b, t1])
                faces_arr.append([a, t1, t0])

            mesh = trimesh.Trimesh(
                vertices=np.asarray(verts, dtype=float),
                faces=np.asarray(faces_arr, dtype=np.int64),
                process=False,
            )
            mesh.merge_vertices(digits_vertex=8)
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.remove_unreferenced_vertices()
            trimesh.repair.fix_winding(mesh)
            mesh.fix_normals()
        except Exception:
            pass
    return mesh


def build_flat_layer_mesh_from_mask(
    geometry: Optional[BaseGeometry],
    *,
    bottom_z_m: float,
    thickness_m: float,
    color: list[int],
    min_area_m2: float = 1e-6,
) -> Optional[trimesh.Trimesh]:
    """Extrude a polygonal 2D mask into one printable flat layer.

    All feature layers start at the base top instead of floating above it. The
    slicer can still print them in later color changes because their top heights
    are different, but every footprint remains physically supported.
    """
    if thickness_m <= 0:
        return None
    meshes: list[trimesh.Trimesh] = []
    for poly in _iter_polygons(geometry):
        try:
            if float(poly.area) < float(min_area_m2):
                continue
            mesh = _extrude_polygon_prism(poly, float(thickness_m))
            if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
                mesh = trimesh.creation.extrude_polygon(poly, height=float(thickness_m))
            if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
                continue
            mesh.apply_translation([0.0, 0.0, float(bottom_z_m)])
            try:
                mesh.update_faces(mesh.nondegenerate_faces())
                mesh.remove_unreferenced_vertices()
                trimesh.repair.fix_winding(mesh)
                mesh.fix_normals()
            except Exception:
                pass
            if mesh.vertices is not None and len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                meshes.append(mesh)
        except Exception as exc:
            print(f"[FLAT PLATE] Skipped polygon layer fragment: {exc}")
    if not meshes:
        return None
    try:
        combined = trimesh.util.concatenate(meshes)
    except Exception:
        combined = meshes[0]
    return _with_color(combined, color)


def build_flat_zone_base_mesh(
    zone_polygon_local: Optional[BaseGeometry],
    *,
    bbox_meters: tuple[float, float, float, float],
    thickness_m: float,
) -> Optional[trimesh.Trimesh]:
    zone_geom = zone_polygon_local
    if zone_geom is None or getattr(zone_geom, "is_empty", True):
        minx, miny, maxx, maxy = bbox_meters
        zone_geom = box(minx, miny, maxx, maxy)
    return build_flat_layer_mesh_from_mask(
        zone_geom,
        bottom_z_m=0.0,
        thickness_m=float(thickness_m),
        color=LAYER_COLORS["base"],
    )


def _geometry_min_dimension(poly: Polygon) -> float:
    try:
        minx, miny, maxx, maxy = poly.bounds
        return float(min(maxx - minx, maxy - miny))
    except Exception:
        return 0.0


def _drop_subprintable_fragments(
    geometry: Optional[BaseGeometry],
    *,
    min_feature_m: float,
    min_area_m2: float,
) -> Optional[BaseGeometry]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return None
    kept: list[Polygon] = []
    for poly in _iter_polygons(geometry):
        try:
            if float(poly.area) < float(min_area_m2):
                continue
            if min_feature_m > 0 and _geometry_min_dimension(poly) < float(min_feature_m):
                continue
            kept.append(poly)
        except Exception:
            continue
    if not kept:
        return None
    try:
        return unary_union(kept).buffer(0)
    except Exception:
        return kept[0]


def _sanitize_layer_mask(
    geometry: Optional[BaseGeometry],
    *,
    min_feature_m: float,
    min_area_m2: float,
    label: str,
) -> Optional[BaseGeometry]:
    geometry = _clean_polygonal_geometry(geometry)
    if geometry is None:
        return None
    original = geometry  # збережемо оригінал для fallback
    try:
        # Opening removes spikes and one-line fragments below the real nozzle
        # floor. Keep the radius conservative; printable masks were already
        # canonicalized before this stage.
        radius = max(float(min_feature_m) * 0.5, 0.0)
        if radius > 0:
            opened = geometry.buffer(-radius, join_style=1).buffer(radius, join_style=1)
            if opened is not None and not getattr(opened, "is_empty", True):
                geometry = opened.buffer(0)
    except Exception:
        pass
    cleaned = _drop_subprintable_fragments(
        geometry,
        min_feature_m=min_feature_m,
        min_area_m2=min_area_m2,
    )
    if cleaned is None and original is not None and not getattr(original, "is_empty", True):
        # FALLBACK: замість того щоб втратити шар повністю, розширюємо оригінал
        # на min_feature_m і повертаємо це. Краще "трохи товстіші дороги" ніж
        # "немає доріг взагалі".
        try:
            widen_radius = max(float(min_feature_m) * 0.6, 0.05)
            widened = original.buffer(widen_radius, join_style=1, cap_style=1)
            if widened is not None and not getattr(widened, "is_empty", True):
                # Знову відсікаємо мікро-фрагменти, але з пом'якшеним порогом
                soft_cleaned = _drop_subprintable_fragments(
                    widened.buffer(0),
                    min_feature_m=float(min_feature_m) * 0.4,  # пом'якшено в 2.5×
                    min_area_m2=float(min_area_m2) * 0.4,
                )
                if soft_cleaned is not None and not getattr(soft_cleaned, "is_empty", True):
                    print(f"[KEYCHAIN] {label} collapsed under {min_feature_m*1000:.2f}mm filter — using widened fallback (radius={widen_radius:.3f}m)")
                    return soft_cleaned
        except Exception as e:
            print(f"[KEYCHAIN] {label} fallback failed: {e}")
        print(f"[KEYCHAIN] {label} collapsed under {min_feature_m*1000:.2f}mm print filter — layer will be empty")
    return cleaned


def _drop_thin_footprints_gdf(gdf, *, erode_m: float, min_iso_ratio: float = 0.10):
    """Прибирає будівлі-«волосини» (slivers) ДВОМА тестами:
    (1) ІЗОПЕРИМЕТРИЧНИЙ коефіцієнт 4π·area/perimeter² — БЕЗРОЗМІРНИЙ (scale-free):
        ~0.78 для квадрата, →0 для довгої нитки. <min_iso_ratio = витягнута нитка
        (типовий clip-artifact: будинок розрізаний краєм вікна карти в смужку
        0.06×60мм, aspect >25:1) → викидаємо. Працює НЕЗАЛЕЖНО від масштабу —
        тому надійніше за ерозію (масштаб карти брелка ≠ scale_factor).
    (2) ЕРОЗІЯ на erode_m: footprint, що зникає (всюди тонший за 2·erode_m).
    Реальні будівлі (компактні/товсті) лишаються БЕЗ ЗМІН — форму не чіпаємо."""
    import math
    if gdf is None or getattr(gdf, "empty", True):
        return gdf
    keep = []
    for idx, geom in gdf.geometry.items():
        try:
            if geom is None or geom.is_empty:
                continue
            per = float(geom.length); area = float(geom.area)
            if per > 0 and (4.0 * math.pi * area / (per * per)) < float(min_iso_ratio):
                continue  # витягнута нитка (scale-free) — викидаємо
            if erode_m > 0:
                core = geom.buffer(-float(erode_m))
                if core is None or core.is_empty or float(core.area) <= 0.0:
                    continue  # всюди тонший за 2·erode_m — викидаємо
            keep.append(idx)
        except Exception:
            keep.append(idx)  # на помилці лишаємо (консервативно)
    if not keep:
        return gdf.iloc[0:0]
    return gdf.loc[keep].copy()


def _mesh_footprint_min_width_m(mesh) -> float:
    """Мін-ширина footprint меша (orientation-invariant) через короткий бік
    minimum_rotated_rectangle опуклої оболонки XY-проєкції. Ловить ВОЛОСИНИ за
    будь-якої орієнтації (на відміну від AABB). Для увігнутих (L) оболонка
    заповнює — реальна будівля лишається; для нитки оболонка ≈ нитка → мала ширина."""
    try:
        from shapely.geometry import MultiPoint
        xy = mesh.vertices[:, :2]
        pts = MultiPoint([(float(x), float(y)) for x, y in xy])
        hull = pts.convex_hull
        if hull.is_empty or hull.geom_type != "Polygon":
            return 1.0e9
        cs = list(hull.minimum_rotated_rectangle.exterior.coords)
        edges = [((cs[i][0] - cs[i + 1][0]) ** 2 + (cs[i][1] - cs[i + 1][1]) ** 2) ** 0.5
                 for i in range(len(cs) - 1)]
        return min(edges) if edges else 1.0e9
    except Exception:
        return 1.0e9  # на помилці лишаємо (не викидаємо)


def _mask_union_from_geometries(geometries: Any) -> Optional[BaseGeometry]:
    if geometries is None:
        return None
    try:
        parts = [geom for geom in geometries if geom is not None and not getattr(geom, "is_empty", True)]
    except Exception:
        return None
    if not parts:
        return None
    try:
        return unary_union(parts).buffer(0)
    except Exception:
        return parts[0]


def _subtract_geometry(geometry: Optional[BaseGeometry], *masks: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return None
    result = geometry
    for mask in masks:
        if mask is None or getattr(mask, "is_empty", True):
            continue
        try:
            result = result.difference(mask).buffer(0)
        except Exception:
            try:
                result = result.buffer(0).difference(mask.buffer(0)).buffer(0)
            except Exception:
                continue
        if result is None or getattr(result, "is_empty", True):
            return None
    return result


def _rounded_rect(minx: float, miny: float, maxx: float, maxy: float, radius_m: float) -> Polygon:
    radius_m = max(0.0, min(float(radius_m), float(maxx - minx) / 2.0, float(maxy - miny) / 2.0))
    if radius_m <= 0:
        return box(minx, miny, maxx, maxy)
    return box(minx + radius_m, miny + radius_m, maxx - radius_m, maxy - radius_m).buffer(radius_m, resolution=14)


def build_magnet_pocket_geometry(
    zone_polygon_local: BaseGeometry,
    *,
    diameter_mm: float,
    count: int,
    inset_mm: float,
    export_scale_factor: float,
) -> Optional[BaseGeometry]:
    """Кишені під магніти у дні плоскої мапи. count=1 — одна в центроїді
    (старий режим, golden-сумісний). count≥2 — діагональне кільце навколо
    центру (кути 45°+k·90°): працює і для квадрата, і для кола/серця;
    кандидат відкидається, якщо кишеня + 0.6мм бічної стінки не вміщується
    в тіло; якщо лишилось <2 — fallback до однієї в центроїді."""
    import math
    # +0.05мм/бік кліренс під магніт-шайбу (легша посадка під клей; фідбек власника).
    r_m = _model_mm_to_world_m(max(float(diameter_mm), 1.0) / 2.0 + 0.05, export_scale_factor)
    if zone_polygon_local is None or getattr(zone_polygon_local, "is_empty", True):
        return None
    cnt = max(int(count or 1), 1)
    c = zone_polygon_local.centroid
    wall_m = _model_mm_to_world_m(0.8, export_scale_factor)  # ≥0.8мм стінка (було 0.6 — ламке)
    if cnt <= 1:
        center = Point(c.x, c.y)
        # Якщо кишеня+стінка пробиває контур (увігнуте тіло/мала зона) — у найглибшу
        # внутрішню точку (representative_point ГАРАНТОВАНО всередині). Для нормального
        # тіла центроїд проходить → поведінка незмінна (golden-сумісність збережена).
        if not zone_polygon_local.contains(center.buffer(r_m + wall_m, resolution=24)):
            rp = zone_polygon_local.representative_point()
            center = Point(rp.x, rp.y)
        return center.buffer(r_m, resolution=48)
    inset_m = _model_mm_to_world_m(max(float(inset_mm), 1.0), export_scale_factor)
    bminx, bminy, bmaxx, bmaxy = zone_polygon_local.bounds
    ring_r = min(bmaxx - bminx, bmaxy - bminy) / 2.0 - inset_m
    centers = []
    if ring_r > r_m:
        for i in range(cnt):
            a = math.pi / 4.0 + (2.0 * math.pi * i) / cnt
            p = Point(c.x + ring_r * math.cos(a), c.y + ring_r * math.sin(a))
            if zone_polygon_local.contains(p.buffer(r_m + wall_m, resolution=24)):
                centers.append(p)
    if len(centers) < 2:
        centers = [Point(c.x, c.y)]
    return unary_union([p.buffer(r_m, resolution=48) for p in centers])


def _mesh_xy_footprint(mesh: trimesh.Trimesh, *, simplify_m: float = 0.0) -> Optional[BaseGeometry]:
    """РЕАЛЬНИЙ XY-контур (увігнуто-стійкий) екструдованого будинку = об'єднання
    проєкцій усіх трикутників на площину XY. Опукла оболонка заповнила б L/U-виїмки;
    цей контур зберігає справжню форму → точний паз/peg. Повертає найбільший полігон."""
    try:
        verts = np.asarray(mesh.vertices, dtype=float)
        tris = []
        for f in mesh.faces:
            v = verts[f]
            p = Polygon([(v[0][0], v[0][1]), (v[1][0], v[1][1]), (v[2][0], v[2][1])])
            if p.is_valid and p.area > 1e-12:
                tris.append(p)
        if not tris:
            return None
        foot = unary_union(tris).buffer(0)
        if simplify_m > 0:  # зварити числові шви, майже не округлюючи реальні кути
            foot = foot.buffer(simplify_m).buffer(-simplify_m).buffer(0)
        if foot is None or getattr(foot, "is_empty", True):
            return None
        if hasattr(foot, "geoms"):
            foot = max(foot.geoms, key=lambda g: g.area)
        return foot
    except Exception:
        return None


def _select_highlight_building_index(
    building_meshes: list, *, target_xy: Optional[tuple[float, float]],
    exclude: Optional[set] = None,
) -> Optional[int]:
    """Індекс будинку для виділення — ОДНАКОВА логіка з /api/building-at (main.py),
    щоб червоний контур на карті = той самий будинок, що отримає вставку:
      1) РЕАЛЬНИЙ XY-контур будинку (_mesh_xy_footprint) МІСТИТЬ target_xy → беремо
         його одразу (клік юзера всередині свого будинку — точний вибір);
      2) інакше — найближчий за ВІДСТАННЮ ДО КРАЮ (polygon.distance(point), як
         geom.distance(pt) у /api/building-at), а не за центроїдом — у щільній
         забудові центроїд може вказати на інший будинок.
    Контур кешуємо лише дешевим bbox-фільтром: повний _mesh_xy_footprint рахуємо
    тільки для кандидатів, чий bbox містить точку (contains) або поки шукаємо
    найближчий. target_xy у тій же системі, що й меші (для брелка — unwrapped).
    `exclude` — індекси вже обраних будинків (щоб два кліки не взяли той самий)."""
    if not building_meshes or target_xy is None:
        return None
    from shapely.geometry import Point as _SelPt
    _ex = exclude or set()
    tx, ty = float(target_xy[0]), float(target_xy[1])
    pt = _SelPt(tx, ty)
    best_i, best_edge_d = None, float("inf")
    for i, m in enumerate(building_meshes):
        if i in _ex or m is None or len(getattr(m, "faces", [])) == 0:
            continue
        b = m.bounds
        # Дешевий bbox-відсів: точка має бути в bbox АБО ми ще шукаємо найближчий.
        bbox_inside = (b[0][0] <= tx <= b[1][0]) and (b[0][1] <= ty <= b[1][1])
        foot = _mesh_xy_footprint(m)
        if foot is None or getattr(foot, "is_empty", True):
            # fallback на bbox-центроїд, якщо контур не порахувався
            if bbox_inside:
                return i
            cx = (b[0][0] + b[1][0]) * 0.5
            cy = (b[0][1] + b[1][1]) * 0.5
            d = math.hypot(cx - tx, cy - ty)
            if d < best_edge_d:
                best_i, best_edge_d = i, d
            continue
        try:
            if foot.contains(pt):  # клік усередині будівлі — точний вибір (break)
                return i
            d = float(foot.distance(pt))  # ВІДСТАНЬ ДО КРАЮ (як /api/building-at)
        except Exception:
            continue
        if d < best_edge_d:
            best_i, best_edge_d = i, d
    return best_i


def build_highlight_insert(
    building_mesh: trimesh.Trimesh,
    *,
    base_top_m: float,
    export_scale_factor: float,
    depth_mm: float = 0.8,
    lip_mm: float = 0.5,
    clearance_mm: float = 0.15,   # = GROOVE_CLEARANCE_MM (дороги): вставка сідає з тим самим зазором
    z_clear_mm: float = 0.2,
) -> tuple[trimesh.Trimesh, Optional[BaseGeometry], float]:
    """Виділений будинок → ВСТАВНА деталь (механічна вставка у паз бази, БЕЗ клею).
    Повертає (highlight_mesh, pocket_poly, pocket_depth_m):
      * highlight_mesh = будинок + peg-ніжка (під базу), колір highlight (ЧЕРВОНИЙ);
      * pocket_poly/pocket_depth_m → виріз ПАЗУ у ВЕРХ бази через
        _build_keychain_base_parts(top_cut_poly=pocket_poly, top_cut_depth_m=depth).

    Дизайн «counterbore» (точна посадка + чисте лице) — досліджено
    debug/research_highlight_pocket.py:
      footprint  = РЕАЛЬНИЙ контур будинку (увігнутості L/U збережено);
      pocket     = footprint − lip      (паз ВУЖЧИЙ за будинок → будинок лягає
                                         бортиком на плече урівень = надійний стоп,
                                         зазор сховано під навісом будинку);
      peg        = pocket − clearance   (ніжка з боковим зазором ≈0.2мм/бік — щільно);
      peg_height = depth − z_clear      (ніжка нижча за паз → сідає на плече, не в дно).
    Будинок замалий (offset зжер контур) → glue-on (pocket=None, деталь без peg)."""
    out = building_mesh.copy()
    none_ret = (_with_color(out, LAYER_COLORS["highlight"]), None, 0.0)
    if depth_mm <= 0:
        return none_ret
    depth = min(_model_mm_to_world_m(depth_mm, export_scale_factor),
                max(base_top_m - _model_mm_to_world_m(0.6, export_scale_factor), 0.0))
    if depth <= 1e-9:
        return none_ret
    lip = _model_mm_to_world_m(lip_mm, export_scale_factor)
    clear = _model_mm_to_world_m(clearance_mm, export_scale_factor)
    z_clear = _model_mm_to_world_m(z_clear_mm, export_scale_factor)
    foot = _mesh_xy_footprint(out, simplify_m=_model_mm_to_world_m(0.05, export_scale_factor))
    if foot is None or getattr(foot, "is_empty", True):
        from shapely.geometry import MultiPoint as _MP
        foot = _MP([(float(v[0]), float(v[1])) for v in out.vertices]).convex_hull
    def _largest(g):
        if g is None or getattr(g, "is_empty", True):
            return None
        return max(g.geoms, key=lambda x: x.area) if hasattr(g, "geoms") else g
    min_peg = _model_mm_to_world_m(0.2, export_scale_factor)
    # 3-РІВНЕВА посадка (макс. покриття для МАЛИХ будинків карти, ~1–2мм):
    #  T1 counterbore — паз=foot−lip (плече), peg=паз−clear, peg нижчий за паз;
    #  T2 без-плеча     — паз=foot, peg=foot−clear, peg=ПОВНА глибина (впирається в
    #                     дно → лице урівень), для будинків замалих на плече;
    #  T3 glue-on       — занадто малий навіть на peg → деталь без паза (приклеїти).
    pocket_poly, peg_poly, peg_h, _mode = None, None, 0.0, "glue"
    _t1_pocket = _largest(foot.buffer(-lip, join_style=2))
    # ПАЗ ІЗ ЗАЗОРОМ ЯК У ДОРІГ: розширюємо паз на +clear (0.15мм/бік) перед вирізом пега.
    # Тоді pocket = foot−lip+clear, peg = pocket−clear = foot−lip → вставка сідає з тим
    # самим боковим зазором, що дорожня вставка у жолоб (раніше паз був рівно foot−lip
    # без зазору під борт будинку → деталь заходила туго / не до кінця).
    if _t1_pocket is not None:
        _t1_pocket = _largest(_t1_pocket.buffer(clear, join_style=2))
    _t1_peg = _largest(_t1_pocket.buffer(-clear, join_style=2)) if _t1_pocket is not None else None
    if _t1_peg is not None:
        pocket_poly, peg_poly, peg_h, _mode = _t1_pocket, _t1_peg, max(depth - z_clear, min_peg), "counterbore"
    else:
        _t2_peg = _largest(foot.buffer(-clear, join_style=2))
        if _t2_peg is not None:
            pocket_poly, peg_poly, peg_h, _mode = foot, _t2_peg, depth, "no-lip"
    if peg_poly is None:
        return none_ret  # T3: glue-on
    peg = build_flat_layer_mesh_from_mask(
        peg_poly, bottom_z_m=base_top_m - peg_h, thickness_m=peg_h,
        color=LAYER_COLORS["highlight"], min_area_m2=1e-12,
    )
    if peg is None:
        return none_ret
    out = trimesh.util.concatenate([out, peg])
    _with_color(out, LAYER_COLORS["highlight"])
    return out, pocket_poly, depth


def _ensure_no_through_hole(
    *,
    base_top_m: float,
    top_cut_poly: Optional[BaseGeometry],
    top_cut_depth_m: float,
    bottom_cut_poly: Optional[BaseGeometry],
    bottom_cut_depth_m: float,
    export_scale_factor: float,
    min_solid_mm: float = 0.4,
    label: str = "HIGHLIGHT",
) -> tuple[float, float]:
    """THROUGH-HOLE GUARD (LATE, defensive): паз ЗВЕРХУ (highlight, ~0.8мм) та виріз
    ЗНИЗУ (магніт-кишеня / конектор-паз / гравіювання звороту) у зоні XY-перекриття
    лишають лише `base_top − top_depth − bottom_depth` суцільного матеріалу. Якщо
    цього < min_solid_mm (0.4мм) — деталь стала б наскрізною дірою.

    Базу тут НЕ чіпаємо (шари вже сидять на base_top_m — потовщення дало б floating
    геометрію); замість цього РІЖЕМО ДЕФІЦИТ з ГЛИБШОГО вирізу (зазвичай нижній паз),
    лишаючи його робочим, але не наскрізним. Основну роботу для map-кейсу робить
    РАННІЙ guard (потовщує базу до побудови шарів); це — підстраховка та покриття
    keychain-back-engrave. Без XY-перекриття або з достатнім матеріалом — НІЧОГО не
    змінюємо (standalone байт-в-байт).

    Повертає (top_cut_depth_m, bottom_cut_depth_m) — скориговані глибини (world-м)."""
    if top_cut_poly is None or getattr(top_cut_poly, "is_empty", True):
        return top_cut_depth_m, bottom_cut_depth_m
    if bottom_cut_poly is None or getattr(bottom_cut_poly, "is_empty", True):
        return top_cut_depth_m, bottom_cut_depth_m
    if top_cut_depth_m <= 1e-9 or bottom_cut_depth_m <= 1e-9:
        return top_cut_depth_m, bottom_cut_depth_m
    try:
        overlap = top_cut_poly.intersection(bottom_cut_poly)
    except Exception:
        # геометрія не перетнулась/збійна — буферимо на 0 і пробуємо ще раз
        try:
            overlap = top_cut_poly.buffer(0).intersection(bottom_cut_poly.buffer(0))
        except Exception:
            return top_cut_depth_m, bottom_cut_depth_m
    if overlap is None or getattr(overlap, "is_empty", True) or float(getattr(overlap, "area", 0.0)) <= 1e-12:
        return top_cut_depth_m, bottom_cut_depth_m  # XY не перекриваються
    min_solid_m = _model_mm_to_world_m(min_solid_mm, export_scale_factor)
    solid_m = base_top_m - top_cut_depth_m - bottom_cut_depth_m
    if solid_m >= min_solid_m - 1e-9:
        return top_cut_depth_m, bottom_cut_depth_m  # вже достатньо матеріалу
    deficit_m = (min_solid_m - solid_m)
    _esf = float(export_scale_factor)  # mm = world_m * scale_factor
    # Ріжемо дефіцит з ГЛИБШОГО вирізу (лишаємо йому ≥0 глибини).
    if bottom_cut_depth_m >= top_cut_depth_m:
        new_bottom = max(bottom_cut_depth_m - deficit_m, 0.0)
        new_top = top_cut_depth_m
        cut_from = "bottom"
    else:
        new_top = max(top_cut_depth_m - deficit_m, 0.0)
        new_bottom = bottom_cut_depth_m
        cut_from = "top"
    print(
        f"[{label}] through-hole guard (late): top pocket overlaps bottom cut in XY "
        f"(area {float(overlap.area):.4g}); solid {solid_m * _esf:.3f}mm < {min_solid_mm}mm → "
        f"reduced {cut_from} cut by {deficit_m * _esf:.3f}mm "
        f"(top {top_cut_depth_m * _esf:.2f}→{new_top * _esf:.2f}mm, "
        f"bottom {bottom_cut_depth_m * _esf:.2f}→{new_bottom * _esf:.2f}mm)"
    )
    return new_top, new_bottom


def build_map_connector_geometry(
    base_polygon: BaseGeometry,
    *,
    edges: str,
    span_mm: float,
    length_mm: float,
    waist_frac: float,
    clearance_mm: float,
    export_scale_factor: float,
    key_edges: Optional[str] = None,
) -> tuple[Optional[BaseGeometry], Optional[BaseGeometry]]:
    """З'ЄДНУВАЧ-ПАЗИ (метелик/bowtie) на серединах граней плоскої карти-плитки.

    Повертає (notch_union, keys_union) у локально-метровій системі base_polygon:
      * notch_union — РІЖЕТЬСЯ у НИЖНІЙ шар бази (через _build_keychain_base_parts
        back_text_poly), тож ЛИЦЕ карти лишається суцільним → шов спереду НЕ видно;
      * keys_union — окремі ПОВНІ метелики-ключі, розкладені рядком ПІД картою,
        друкуються поруч і вставляються у спільний паз двох сусідніх плиток.

    Геометрія = «ластівчин хвіст»: вузько біля шва (талія), широко всередині —
    тримає дві плитки в площині (звичайний прямокутний шип просто висковзнув би).
    Дзеркальність гарантує збіг: КОЖНА грань має однаковий half-notch, тож сусід
    завжди стикується. Розміри в model-мм; +clearance_mm/бік FDM-зазор (0.2≈щільно).
    Перевірено shapely (debug/validate_connectors.py): key⊂(recA∪recB), зазор-кільце>0,
    база лишається single-polygon навіть із 4 пазами одночасно."""
    if base_polygon is None or getattr(base_polygon, "is_empty", True):
        return None, None
    # w = глибина лопаті у плитку; h0 = півдовжина широкого кінця вздовж шва.
    w = _model_mm_to_world_m(max(float(span_mm), 4.0) / 2.0, export_scale_factor)
    h0 = _model_mm_to_world_m(max(float(length_mm), 6.0) / 2.0, export_scale_factor)
    clr = _model_mm_to_world_m(max(float(clearance_mm), 0.0), export_scale_factor)
    min_h = _model_mm_to_world_m(2.0, export_scale_factor)
    hw_frac = min(max(float(waist_frac), 0.2), 0.8)
    minx, miny, maxx, maxy = base_polygon.bounds
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    bw, bh = maxx - minx, maxy - miny

    def _half(h: float) -> BaseGeometry:
        # Локально: шов на x=0, нутро плитки на +x; трапеція (вузька на шві, широка
        # всередині) + кліренс. join_style=2 (mitre) — рівні кути ластівчиного хвоста.
        hw = h * hw_frac
        return Polygon([(0.0, hw), (w, h), (w, -h), (0.0, -hw)]).buffer(clr, join_style=2)

    def _key(h: float) -> BaseGeometry:
        hw = h * hw_frac
        return Polygon([(-w, -h), (-w, h), (0.0, hw), (w, h), (w, -h), (0.0, -hw)]).buffer(0)

    notches: list[BaseGeometry] = []
    keys: list[BaseGeometry] = []
    edges_set = set((edges or "NSEW").upper())
    # «КОЖНА грань»: повний набір NSEW (дефолт фронта), порожньо або 'A'/'ALL' →
    # з'єднувач на КОЖНІЙ реальній грані полігону. Раніше код мапив лише 4
    # кардинальні напрямки на найкращу грань → шестикутник діставав 4 замки з 6,
    # намальований/коловий контур — теж недокомплект («замки не на кожній грані»).
    # Строга ПІДмножина (напр. лише 'NS') лишає стару поведінку «обрані боки».
    want_all = (not edges_set) or ("A" in edges_set) or ({"N", "S", "E", "W"} <= edges_set)
    # key_edges: грані, для яких ВИПУСКАЄМО ключ-метелик. None → ключ для КОЖНОГО пазу
    # (single-tile: усі замки цієї плитки потребують ключа). Для СЕРІЇ передаємо лише
    # S/E внутрішні грані → на спільний шов двох плиток припадає РІВНО ОДИН ключ
    # (інакше кожна плитка друкувала б свій → 2 ключі/шов). Паз ріжемо на ВСІХ гранях.
    key_edges_set = set(key_edges.upper()) if key_edges is not None else None
    margin = _model_mm_to_world_m(6.0, export_scale_factor)
    key_pitch = 2.0 * w + margin
    key_slot = 0

    # ── Замок ставимо на РЕАЛЬНУ грань полігону (середина грані + справжня нормаль).
    import math
    _poly = base_polygon
    if getattr(_poly, "geom_type", "") == "MultiPolygon":
        _poly = max(_poly.geoms, key=lambda g: g.area)
    # Зливаємо майже-колінеарні дрібні сегменти (полігонізоване коло / спрощений
    # малюнок), щоб КОЖНА реальна сторона давала ОДИН з'єднувач, а не десятки
    # крихітних. Прямокутник/шестикутник зберігають усі справжні кути.
    try:
        _simp = _poly.simplify(max(min_h, w) * 0.75, preserve_topology=True)
        if _simp is not None and not _simp.is_empty and getattr(_simp, "geom_type", "") == "Polygon":
            _poly = _simp
    except Exception:
        pass
    ring = list(getattr(getattr(_poly, "exterior", None), "coords", []))
    if len(ring) < 4:
        return None, None
    # Орієнтуємо CCW (нутро ліворуч від напрямку грані) через знак площі (шнурівка).
    _area2 = 0.0
    for (ax, ay), (bx, by) in zip(ring[:-1], ring[1:]):
        _area2 += ax * by - bx * ay
    if _area2 < 0:
        ring = ring[::-1]
    segs = []  # (mid, inward_unit, outward_unit, length)
    for (x0, y0), (x1, y1) in zip(ring[:-1], ring[1:]):
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L < min_h:  # надто коротка грань — не вмістить замок
            continue
        inward = (-dy / L, dx / L)   # CCW: нутро = ліва нормаль грані
        outward = (dy / L, -dx / L)
        segs.append((((x0 + x1) / 2.0, (y0 + y1) / 2.0), inward, outward, L))
    if not segs:
        return None, None

    # Які грані дістануть з'єднувач.
    if want_all:
        chosen = list(segs)  # КОЖНА грань
    else:
        # Легасі: для кожного обраного кардинального боку — грань, чия зовнішня
        # нормаль найкраще туди дивиться (dot>0.15), без дублів.
        _CARD = {"N": (0.0, 1.0), "S": (0.0, -1.0), "E": (1.0, 0.0), "W": (-1.0, 0.0)}
        chosen = []
        for e in ("N", "S", "E", "W"):
            if e not in edges_set:
                continue
            cardx, cardy = _CARD[e]
            best, best_score = None, 0.15
            for seg in segs:
                _, _, outward, _ = seg
                score = outward[0] * cardx + outward[1] * cardy
                if score > best_score:
                    best_score, best = score, seg
            if best is not None and best not in chosen:
                chosen.append(best)

    # Запобіжник від захаращення (дуже багатогранний контур) — найдовші грані.
    MAX_CONN = 16
    if len(chosen) > MAX_CONN:
        chosen = sorted(chosen, key=lambda s: s[3], reverse=True)[:MAX_CONN]

    for (mx, my), inward, _outward, L in chosen:
        h = min(h0, L * 0.40)  # замок ≤40% РЕАЛЬНОЇ грані
        if h < min_h:
            continue
        # _half: шов на x=0, нутро плитки на +x. Повертаємо так, щоб +x збіглося з
        # внутрішньою нормаллю грані, далі ставимо на середину грані.
        ang = math.degrees(math.atan2(inward[1], inward[0]))
        placed = affinity.translate(
            affinity.rotate(_half(h), ang, origin=(0.0, 0.0)), xoff=mx, yoff=my
        )
        # Лишаємо лише частину всередині бази (зрізаємо кліренс-навіс за швом).
        clipped = placed.intersection(base_polygon).buffer(0)
        if clipped is None or getattr(clipped, "is_empty", True):
            continue
        notches.append(clipped)
        # Ключ ЛИШЕ якщо ця грань у key_edges (None = усі). Кардинал грані — за
        # домінантою зовнішньої нормалі (для прямокутника точний ±x/±y).
        if key_edges_set is not None:
            _cx, _cy = _outward
            _card = max((("N", _cy), ("S", -_cy), ("E", _cx), ("W", -_cx)),
                        key=lambda t: t[1])[0]
        else:
            _card = None
        if key_edges_set is None or _card in key_edges_set:
            # Ключ-метелик у вільному рядку ПІД картою (поза footprint основи).
            kx = minx + key_slot * key_pitch + w
            ky = miny - margin - h0
            keys.append(affinity.translate(_key(h), xoff=kx, yoff=ky))
            key_slot += 1

    if not notches:
        return None, None
    notch_union = unary_union(notches).buffer(0)
    keys_union = unary_union(keys).buffer(0) if keys else None
    return notch_union, keys_union


def _frame_place_text(
    text: str, *, height_m: float, x: float, y: float, ha: str = "center", va: str = "center",
) -> Optional[BaseGeometry]:
    """TTF-текст у локально-метровій системі, заякорений у (x,y) за ha/va.
    ha: left|center|right (горизонт.), va: bottom|center|top (верт.)."""
    if not text or height_m <= 0:
        return None
    polys = _ttf_text_polygons(text, height_m=height_m, font_family="DejaVu Sans")
    if not polys:
        return None
    u = unary_union(polys).buffer(0)
    if u is None or u.is_empty:
        return None
    b = u.bounds
    ax = {"left": b[0], "center": (b[0] + b[2]) / 2.0, "right": b[2]}.get(ha, (b[0] + b[2]) / 2.0)
    ay = {"bottom": b[1], "center": (b[1] + b[3]) / 2.0, "top": b[3]}.get(va, (b[1] + b[3]) / 2.0)
    return affinity.translate(u, xoff=x - ax, yoff=y - ay)


def _nice_round_number(value: float) -> float:
    """Найближче «гарне» число (1/2/5×10ⁿ) ≤ value — для довжини масштабної лінійки."""
    import math
    if value <= 0:
        return 0.0
    exp = math.floor(math.log10(value))
    base = 10.0 ** exp
    for m in (5.0, 2.0, 1.0):
        if m * base <= value:
            return m * base
    return base


def build_map_frame_overlay(
    base_polygon: BaseGeometry,
    *,
    north: float, south: float, east: float, west: float,
    export_scale_factor: float,
    want_compass: bool = True,
    want_scale: bool = True,
    want_coords: bool = True,
    frame_style: str = "classic",
) -> Optional[BaseGeometry]:
    """ПРЕМІУМ-РАМКА плоскої карти: компас (стрілка-N), масштабна лінійка
    (шахова + підписи 0…N м) і координати центру (lat/lon) — як ОДИН підведений
    полігон у кутах/краях, що потім екструдується окремою чорною деталлю «Frame»
    і вирізається з шарів карти (як map_label) щоб читалось чисто.

    Координати world-метрів: base_polygon.bounds = (west..east)×(south..north).
    Розміри елементів у model-мм → _model_mm_to_world_m. Реальна довжина лінійки
    рахується від РЕАЛЬНОЇ ширини карти (метри широти/довготи).

    frame_style:
      • "classic" — лише компас+лінійка+координати (як було, golden-сумісно).
      • "ornate"  — + декоративний підведений ободок по периметру (подвійна лінія)
        та прості кутові мотиви (сходинкові квадрати). FDM-друкабельно (стінка
        ≥0.8мм). Геометрія через shapely (буфер контуру плити).
      • "compass" — + тонкий ОДИНАРНИЙ зовнішній ободок (акцент на компас/лінійці)."""
    if base_polygon is None or getattr(base_polygon, "is_empty", True):
        return None
    minx, miny, maxx, maxy = base_polygon.bounds
    bw, bh = maxx - minx, maxy - miny
    if bw <= 0 or bh <= 0:
        return None
    mm = lambda v: _model_mm_to_world_m(v, export_scale_factor)  # noqa: E731
    margin = mm(4.0)
    stroke = mm(0.8)
    parts: list[BaseGeometry] = []

    # ── Масштабна лінійка (нижній-лівий кут): шахова смуга + підписи 0 / N м.
    if want_scale:
        try:
            import math
            # РЕАЛЬНА ширина карти в метрах (середня широта для м/градус довготи).
            lat_c = math.radians((north + south) / 2.0)
            real_w_m = abs(east - west) * 111320.0 * max(math.cos(lat_c), 0.05)
            target = _nice_round_number(real_w_m * 0.28)  # ~чверть ширини, кругле
            if target > 0:
                # переводимо реальні метри → world-юніти (world = real m × ширина_world/real)
                bar_len = bw * (target / real_w_m) if real_w_m > 1e-6 else bw * 0.28
                bar_h = mm(2.2)
                lab_h = mm(3.2)
                x0 = minx + margin
                y0 = miny + margin + lab_h * 1.25  # підписи нижче смуги
                n_seg = 4
                seg_w = bar_len / n_seg
                bar_parts = []
                for i in range(n_seg):  # шахові залиті сегменти
                    if i % 2 == 0:
                        bar_parts.append(box(x0 + i * seg_w, y0, x0 + (i + 1) * seg_w, y0 + bar_h))
                outer = box(x0, y0, x0 + bar_len, y0 + bar_h)
                inner = box(x0 + stroke, y0 + stroke, x0 + bar_len - stroke, y0 + bar_h - stroke)
                bar_parts.append(outer.difference(inner).buffer(0))  # контур навколо всіх сегментів
                lab0 = _frame_place_text("0", height_m=lab_h, x=x0, y=y0 - lab_h * 0.5, ha="center", va="top")
                unit = f"{int(target)} m" if target < 1000 else f"{target/1000:g} km"
                labN = _frame_place_text(unit, height_m=lab_h, x=x0 + bar_len, y=y0 - lab_h * 0.5, ha="center", va="top")
                bar_parts += [p for p in (lab0, labN) if p is not None]
                sb = unary_union([p for p in bar_parts if p is not None]).buffer(0)
                if sb is not None and not sb.is_empty:
                    parts.append(sb)
        except Exception as exc:
            print(f"[MAP FRAME] scale bar failed (non-fatal): {exc}")

    # ── Компас (верхній-правий кут): стрілка-N вгору (північ) + літера «N».
    if want_compass:
        try:
            size = mm(11.0)
            cx = maxx - margin - size * 0.5
            cy = maxy - margin - size * 0.6
            ax_w = size * 0.42
            arrow = Polygon([
                (cx, cy + size * 0.5),               # вістря (північ)
                (cx - ax_w * 0.5, cy - size * 0.5),
                (cx, cy - size * 0.18),               # вирізана «ластівка» хвоста
                (cx + ax_w * 0.5, cy - size * 0.5),
            ]).buffer(0)
            nlet = _frame_place_text("N", height_m=size * 0.42, x=cx, y=cy + size * 0.5 + mm(1.6), ha="center", va="bottom")
            comp = unary_union([p for p in (arrow, nlet) if p is not None]).buffer(0)
            if comp is not None and not comp.is_empty:
                parts.append(comp)
        except Exception as exc:
            print(f"[MAP FRAME] compass failed (non-fatal): {exc}")

    # ── Координати центру (нижній-правий кут): «50.45°N 30.52°E».
    if want_coords:
        try:
            lat_c = (north + south) / 2.0
            lon_c = (east + west) / 2.0
            ns = "N" if lat_c >= 0 else "S"
            ew = "E" if lon_c >= 0 else "W"
            txt = f"{abs(lat_c):.4f}°{ns} {abs(lon_c):.4f}°{ew}"
            ct = _frame_place_text(txt, height_m=mm(2.6), x=maxx - margin, y=miny + margin, ha="right", va="bottom")
            if ct is not None and not ct.is_empty:
                parts.append(ct)
        except Exception as exc:
            print(f"[MAP FRAME] coords failed (non-fatal): {exc}")

    # ── Орнаментальний ободок по периметру (ornate/compass).
    # Будуємо кільце як різницю двох усаджених копій контуру плити (shapely).
    # Друкабельність: товщина стінки кожної лінії ≥0.8мм; для "ornate" — подвійна
    # лінія + сходинкові кутові квадрати; для "compass" — одна тонша зовнішня лінія.
    style = (frame_style or "classic").strip().lower()
    if style in ("ornate", "compass"):
        try:
            def _ring(inset_mm: float, width_mm: float) -> Optional[BaseGeometry]:
                """Кільце-лінія всередині плити: зовнішній край на inset від краю,
                ширина стінки = width. Через буфери самого base_polygon."""
                outer = base_polygon.buffer(-mm(inset_mm), join_style=1)
                inner = base_polygon.buffer(-mm(inset_mm + width_mm), join_style=1)
                if outer is None or outer.is_empty:
                    return None
                ring = outer.difference(inner) if (inner is not None and not inner.is_empty) else outer
                ring = ring.buffer(0)
                return ring if (ring is not None and not ring.is_empty) else None

            rim_parts: list[BaseGeometry] = []
            if style == "ornate":
                r1 = _ring(2.0, 1.0)   # зовнішня лінія (стінка 1.0мм)
                r2 = _ring(4.2, 0.9)   # внутрішня лінія (стінка 0.9мм), зазор ~1.3мм
                for r in (r1, r2):
                    if r is not None:
                        rim_parts.append(r)
                # Сходинкові кутові мотиви: маленькі квадрати в 4 кутах bbox плити.
                csz = mm(5.0)
                cins = mm(1.6)
                corners = [
                    (minx + cins, miny + cins),  # нижній-лівий
                    (maxx - cins - csz, miny + cins),  # нижній-правий
                    (minx + cins, maxy - cins - csz),  # верхній-лівий
                    (maxx - cins - csz, maxy - cins - csz),  # верхній-правий
                ]
                cstroke = mm(1.0)
                for (qx, qy) in corners:
                    sq_out = box(qx, qy, qx + csz, qy + csz)
                    sq_in = box(qx + cstroke, qy + cstroke, qx + csz - cstroke, qy + csz - cstroke)
                    motif = sq_out.difference(sq_in).buffer(0)
                    # маленький залитий квадратик у центрі мотиву (акцент)
                    dsz = csz * 0.28
                    dcx, dcy = qx + csz * 0.5, qy + csz * 0.5
                    dot = box(dcx - dsz * 0.5, dcy - dsz * 0.5, dcx + dsz * 0.5, dcy + dsz * 0.5)
                    cm = unary_union([g for g in (motif, dot) if g is not None and not g.is_empty]).buffer(0)
                    if cm is not None and not cm.is_empty:
                        rim_parts.append(cm)
            else:  # compass — тонкий одинарний зовнішній ободок
                r1 = _ring(2.0, 0.9)
                if r1 is not None:
                    rim_parts.append(r1)

            rim = unary_union([g for g in rim_parts if g is not None and not g.is_empty]).buffer(0) if rim_parts else None
            if rim is not None and not rim.is_empty and rim.is_valid:
                # тримаємо ободок усередині плити
                rim = rim.intersection(base_polygon).buffer(0)
                if rim is not None and not rim.is_empty:
                    parts.append(rim)
        except Exception as exc:
            print(f"[MAP FRAME] ornamental rim ({style}) failed (non-fatal): {exc}")

    if not parts:
        return None
    overlay = unary_union(parts).buffer(0)
    # тримаємо все в межах плити
    overlay = overlay.intersection(base_polygon).buffer(0)
    if overlay is None or overlay.is_empty:
        return None
    return overlay


def _round_polygon_tip(pts: list, *, tip_index: int, radius: float, samples: int = 16) -> list:
    """Заокруглення одного гострого вузла замкнутого контуру: вершини в радіусі
    `radius` від кінчика замінюються семплами квадратичної Безьє (контрольна
    точка = старий кінчик). Використовується для вістря серця.
    СИМЕТРИЧНО: беремо однакову кількість вузлів з обох боків (k=min) — інакше
    нерівний крок семплінгу давав асиметричну «зазубрину» внизу серця."""
    import math
    n = len(pts)
    if n < 8 or radius <= 0:
        return pts
    tip = pts[tip_index]

    def _walk(direction: int):
        dist = 0.0
        i = tip_index
        prev = tip
        for _ in range(n // 2):
            i = (i + direction) % n
            cur = pts[i]
            dist += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
            prev = cur
            if dist >= radius:
                return i
        return (tip_index + direction) % n

    ka = (tip_index - _walk(-1)) % n
    kb = (_walk(+1) - tip_index) % n
    k = max(1, min(ka, kb))          # симетрична кількість вузлів з кожного боку
    ia = (tip_index - k) % n
    ib = (tip_index + k) % n
    a, b = pts[ia], pts[ib]
    arc = []
    for s in range(samples + 1):
        t = s / samples
        x = (1 - t) ** 2 * a[0] + 2 * (1 - t) * t * tip[0] + t ** 2 * b[0]
        y = (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * tip[1] + t ** 2 * b[1]
        arc.append((x, y))
    out = []
    i = ib
    while i != ia:
        out.append(pts[i])
        i = (i + 1) % n
    out.append(pts[ia])
    out.extend(arc)
    return out


def _keychain_body_shape(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    *,
    radius_m: float,
    shape: str,
    mm_to_world: float = 1.0,
) -> BaseGeometry:
    # КРИТИЧНО: minx..maxy у build_keychain_layout = СВІТОВІ одиниці (mm·layout_scale,
    # ~16 м/мм), НЕ мм! Тож фіксовані-мм-літерали (кліренс +0.05мм, зріз вістря 1.7мм)
    # ОБОВʼЯЗКОВО множити на mm_to_world=layout_scale_m_per_mm, інакше у реальній
    # генерації вони ~0 (баг: shapely-тест на мм-bounds проходив, а друк — ні).
    shape_name = (shape or "rounded").lower().replace("_", "-")
    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)
    if shape_name in {"capsule", "token"}:
        return _rounded_rect(minx, miny, maxx, maxy, height / 2.0)
    if shape_name == "tag":
        # Зріз кута — БІЛЯ ПЕТЛІ (maxy = верх брелка), як малює дизайнер-превʼю.
        # Історично різалось біля miny (дзеркально до превʼю) — виправлено 2026-06-11.
        cut = min(width, height) * 0.16
        points = [
            (minx, maxy - radius_m),
            (minx + radius_m, maxy),
            (maxx - cut, maxy),
            (maxx, maxy - cut),
            (maxx, miny + radius_m),
            (maxx - radius_m, miny),
            (minx + radius_m, miny),
            (minx, miny + radius_m),
        ]
        return Polygon(points).buffer(0)
    if shape_name == "octagon":
        cut = min(width, height) * 0.13
        return Polygon(
            [
                (minx + cut, miny),
                (maxx - cut, miny),
                (maxx, miny + cut),
                (maxx, maxy - cut),
                (maxx - cut, maxy),
                (minx + cut, maxy),
                (minx, maxy - cut),
                (minx, miny + cut),
            ]
        ).buffer(0)
    if shape_name == "heart":
        # Класична параметрична крива серця, нормалізована в bbox тіла.
        # У цій системі координат maxy = ВЕРХ брелка (бік петлі, див. loop_center_y
        # = body_maxy - y_mm*scale нижче по файлу) → лоби серця до maxy,
        # вістря до miny. Та сама крива у designer-SVG (shapePath) з y-фліпом.
        import math
        n = 160  # вища роздільність → гладкий низ без фасеток (було 96)
        raw = []
        for i in range(n):
            t = 2.0 * math.pi * i / n
            hx = 16.0 * math.sin(t) ** 3
            hy = 13.0 * math.cos(t) - 5.0 * math.cos(2.0 * t) - 2.0 * math.cos(3.0 * t) - math.cos(4.0 * t)
            raw.append((hx, hy))
        xs = [p[0] for p in raw]
        ys = [p[1] for p in raw]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        pts = [
            (
                minx + (px - x0) / (x1 - x0) * width,
                miny + (py - y0) / (y1 - y0) * height,  # лоби (py=max) до maxy = бік петлі
            )
            for px, py in raw
        ]
        # Вістря (вершина з min y) було надто гострим для носіння — зрізаємо
        # квадратичною Безьє через старий кінчик у радіусі R від нього.
        # ІДЕНТИЧНИЙ алгоритм у designer-SVG (shapePath) і MapSelector
        # (shapeOutlinePoints) — превʼю й модель збігаються.
        pts = _round_polygon_tip(pts, tip_index=min(range(len(pts)), key=lambda i: pts[i][1]),
                                 radius=min(width, height) * 0.11)
        return Polygon(pts).buffer(0)
    if shape_name in {"heart-l", "heart-r"}:
        # ПАРА ДЛЯ ЗАКОХАНИХ: серце, розрізане вертикально на дві половинки
        # з puzzle-замком по грані розрізу. bbox (minx..maxx) = ОДНА половинка;
        # повне серце будується на подвійній ширині і кліпається.
        # Замок: jigsaw-кнопка (головка-коло k=0.14·грані розрізу, шийка 0.60k →
        # головка ширша за шийку = справжнє зчеплення в площині). Деталі нижче.
        import math
        # ПОВНЕ серце на подвійній ширині, але БЕЗ заокруглення вістря: інакше
        # вертикальний розріз через заокруглений (пласкенький) низ дає у кожній
        # половинці 90°-«гачок» біля шва — саме та «крива» що скаржився власник.
        # Гостре вістря по центру → кожна половинка сходить у чистий кінчик на шві,
        # а складене серце має класичний гострий низ.
        _hn = 160
        _hraw = []
        for _i in range(_hn):
            _t = 2.0 * math.pi * _i / _hn
            _hx = 16.0 * math.sin(_t) ** 3
            _hy = 13.0 * math.cos(_t) - 5.0 * math.cos(2.0 * _t) - 2.0 * math.cos(3.0 * _t) - math.cos(4.0 * _t)
            _hraw.append((_hx, _hy))
        _fw = 2.0 * width
        _xs = [p[0] for p in _hraw]; _ys = [p[1] for p in _hraw]
        _x0, _x1 = min(_xs), max(_xs); _y0, _y1 = min(_ys), max(_ys)
        full = Polygon([
            (minx + (px - _x0) / (_x1 - _x0) * _fw, miny + (py - _y0) / (_y1 - _y0) * height)
            for px, py in _hraw
        ]).buffer(0)
        cut = minx + width
        cut_line = LineString([(cut, miny - height), (cut, maxy + height)])
        seg = full.intersection(cut_line)
        segs = list(seg.geoms) if hasattr(seg, "geoms") else [seg]
        longest = max(segs, key=lambda g: g.length)
        y0e, y1e = longest.bounds[1], longest.bounds[3]
        elen = max(y1e - y0e, 1e-6)
        cy = (y0e + y1e) / 2.0
        # СПРАВЖНІЙ ЗАМОК = jigsaw-кнопка (головка ШИРША за шийку → інтерференційне
        # зчеплення, що тримає в площині; серце-форма трималась лише тертям —
        # дослідження). Головка-коло k=0.14·грані, шийка nw=0.60k (діаметр 2k vs
        # шийка 1.2k → головка ~1.67× → замикає). Кліренс 0.6% грані (~0.20мм на
        # 33мм) — щільний клац, але розʼємний (FDM-друк). Та сама механіка, що
        # puzzle-пара. Кнопка лишається ВСЕРЕДИНІ контуру серця (∩ full) → стик
        # виглядає як ОДНЕ ціле серце.
        k = elen * 0.14
        nw = k * 0.60
        # Кліренс пазу = пропорційний (0.6% грані) + ФІКСОВАНІ 0.05мм/бік (фідбек
        # власника: трохи легше складати половинки). 0.05мм у model-mm = 0.05мм
        # на друці (FDM лишається щільним клацом, але розʼємним).
        clearance = elen * 0.006 + 0.05 * mm_to_world
        knob_cx = cut + k * 0.95
        knob = unary_union([
            Point(knob_cx, cy).buffer(k, resolution=48),
            box(cut - k * 0.2, cy - nw, knob_cx, cy + nw),
        ])
        # Зрізаємо гостру ГОЛКУ-вістря на шві (0-ширини → не друкується/відламується)
        # маленьким плоским дном ~0.6мм: серце все одно читається гострим, але кінчик
        # стає друкованим. _tip_flat у mm (body-shape простір). Синхрон у KeychainDesigner
        # heartHalfPoints (превʼю=друк). Соло-серце не зачіпається (інша гілка).
        _tip_flat = min(1.7 * mm_to_world, height * 0.05)
        if shape_name == "heart-l":
            half = full.intersection(box(minx - width, miny + _tip_flat, cut, maxy + height))
            tab = knob.intersection(full)  # лишається в межах серця
            return unary_union([half, tab]).buffer(0)
        half = full.intersection(box(cut, miny + _tip_flat, minx + 2.0 * width + width, maxy + height))
        notch = knob.buffer(clearance, join_style=1)
        half = half.difference(notch).buffer(0)
        return affinity.translate(half, xoff=-width)
    if shape_name in {"puzzle-l", "puzzle-r"}:
        # C2 ПАЗЛ-ПАРА: два брелки, що зʼєднуються (long-distance подарунок).
        # L має ВИСТУП (knob) на правій грані, R — ПАЗ на лівій з клиренсом
        # ~0.25мм (масштаб-інваріантно: 0.8% від min-сторони ≈ 0.28мм на 35мм).
        # Геометрія вертикально центрована → дзеркальний y-фліп превʼю не впливає.
        k = min(width, height) * 0.13          # радіус головки
        nw = k * 0.62                           # півширина шийки
        cy = (miny + maxy) / 2.0
        rect = _rounded_rect(minx, miny, maxx, maxy, radius_m)
        if shape_name == "puzzle-l":
            knob_cx = maxx + k * 0.95
            tab = unary_union([
                Point(knob_cx, cy).buffer(k, resolution=48),
                box(maxx - k * 0.2, cy - nw, knob_cx, cy + nw),
            ])
            return unary_union([rect, tab]).buffer(0)
        # Кліренс пазу = пропорційний (0.8% min-сторони) + ФІКСОВАНІ 0.05мм/бік
        # (фідбек власника: легше зчіпати пазл-пару). Та сама механіка, що серце.
        clearance = min(width, height) * 0.008 + 0.05 * mm_to_world
        knob_cx = minx + k * 0.95
        notch = unary_union([
            Point(knob_cx, cy).buffer(k, resolution=48),
            box(minx - k * 0.2, cy - nw, knob_cx, cy + nw),
        ]).buffer(clearance, join_style=1)
        return rect.difference(notch).buffer(0)
    if shape_name == "house":
        # Силует будиночка: вершина даху зверху (maxy, бік петлі), стіни донизу.
        roof_h = height * 0.38
        cx = (minx + maxx) / 2.0
        return Polygon(
            [
                (cx, maxy),
                (maxx, maxy - roof_h),
                (maxx, miny),
                (minx, miny),
                (minx, maxy - roof_h),
            ]
        ).buffer(0)
    if shape_name in {"circle", "ellipse", "round"}:
        # Коло/еліпс, вписаний у bbox (несиметричне тіло → овал).
        from shapely.affinity import scale as _affscale
        cx = (minx + maxx) / 2.0; cy = (miny + maxy) / 2.0
        unit = Point(0.0, 0.0).buffer(1.0, resolution=72)
        return affinity.translate(_affscale(unit, xfact=width / 2.0, yfact=height / 2.0), xoff=cx, yoff=cy)
    if shape_name == "hexagon":
        import math
        cx = (minx + maxx) / 2.0; cy = (miny + maxy) / 2.0
        rx = width / 2.0; ry = height / 2.0
        pts = [(cx + rx * math.cos(math.pi / 2 + i * math.pi / 3),
                cy + ry * math.sin(math.pi / 2 + i * math.pi / 3)) for i in range(6)]
        r = min(width, height) * 0.06
        return Polygon(pts).buffer(-r, join_style=1).buffer(r, join_style=1).buffer(0)
    if shape_name == "shield":
        # Щит: широкі плечі зверху (бік петлі), сходиться у заокруглене вістря внизу.
        cx = (minx + maxx) / 2.0
        pts = [
            (minx, maxy), (maxx, maxy),
            (maxx, miny + height * 0.45),
            (cx, miny),
            (minx, miny + height * 0.45),
        ]
        r = min(width, height) * 0.05
        return Polygon(pts).buffer(-r, join_style=1).buffer(r, join_style=1).buffer(0)
    if shape_name == "star":
        # 5-кутна зірка з ЗАОКРУГЛЕНИМИ вершинами (інакше 5 гострих голок →
        # не друкуються/відламуються). Opening (−r,+r) згладжує опуклі вістря.
        import math
        cx = (minx + maxx) / 2.0; cy = (miny + maxy) / 2.0
        rx = width / 2.0; ry = height / 2.0
        ri = 0.45  # внутрішній радіус (частка зовнішнього)
        pts = []
        for i in range(10):
            ang = math.pi / 2 + i * math.pi / 5
            rr = 1.0 if i % 2 == 0 else ri
            pts.append((cx + rx * rr * math.cos(ang), cy + ry * rr * math.sin(ang)))
        r = min(width, height) * 0.045
        return Polygon(pts).buffer(-r, join_style=1).buffer(r, join_style=1).buffer(0)
    return _rounded_rect(minx, miny, maxx, maxy, radius_m)


def _keychain_loop_outer(
    *,
    center_x: float,
    center_y: float,
    outer_m: float,
    style: str,
) -> BaseGeometry:
    style_name = (style or "round").lower().replace("_", "-")
    if style_name == "slot":
        return _rounded_rect(
            center_x - outer_m * 1.28,
            center_y - outer_m * 0.72,
            center_x + outer_m * 1.28,
            center_y + outer_m * 0.72,
            outer_m * 0.72,
        )
    if style_name == "side-tab":
        return _rounded_rect(
            center_x - outer_m,
            center_y - outer_m * 0.78,
            center_x + outer_m,
            center_y + outer_m * 0.78,
            outer_m * 0.42,
        )
    if style_name == "teardrop":
        circle = Point(center_x, center_y).buffer(outer_m, resolution=36)
        drop = Polygon(
            [
                (center_x - outer_m * 0.75, center_y + outer_m * 0.25),
                (center_x + outer_m * 0.75, center_y + outer_m * 0.25),
                (center_x, center_y + outer_m * 1.42),
            ]
        )
        return unary_union([circle, drop]).buffer(0)
    return Point(center_x, center_y).buffer(outer_m, resolution=36)


def _keychain_loop_inner(
    *,
    center_x: float,
    center_y: float,
    inner_m: float,
    style: str,
) -> BaseGeometry:
    style_name = (style or "round").lower().replace("_", "-")
    if style_name == "slot":
        return _rounded_rect(
            center_x - inner_m * 1.25,
            center_y - inner_m * 0.58,
            center_x + inner_m * 1.25,
            center_y + inner_m * 0.58,
            inner_m * 0.58,
        )
    return Point(center_x, center_y).buffer(inner_m, resolution=36)


def build_keychain_layout(
    *,
    bbox_meters: tuple[float, float, float, float],
    scale_factor: float,
    model_size_mm: float,
    body_width_mm: Optional[float] = None,
    body_height_mm: Optional[float] = None,
    map_x_mm: Optional[float] = None,
    map_y_mm: Optional[float] = None,
    map_width_mm: Optional[float] = None,
    map_height_mm: Optional[float] = None,
    base_shape: str = "rounded",
    loop_style: str = "round",
    loop_angle_deg: float = 0.0,
    loop_center_x_mm: Optional[float] = None,
    loop_center_y_mm: Optional[float] = None,
    label_center_x_mm: Optional[float] = None,
    label_center_y_mm: Optional[float] = None,
    label_width_mm: Optional[float] = None,
    label_angle_deg: float = 0.0,
    loop_outer_radius_mm: float,
    loop_inner_radius_mm: float,
    corner_radius_mm: float,
    label_band_height_mm: float,
) -> dict[str, BaseGeometry]:
    minx, miny, maxx, maxy = bbox_meters
    source_w = max(float(maxx - minx), 1e-6)
    source_h = max(float(maxy - miny), 1e-6)
    body_w_mm = max(float(body_width_mm or model_size_mm or 35.0), 24.0)
    body_h_mm = max(float(body_height_mm or (body_w_mm * source_h / max(source_w, 1e-6))), 18.0)
    map_w_mm = max(float(map_width_mm or body_w_mm), 4.0)
    # ВАЖЛИВО: за замовчуванням слот мапи = ВЕСЬ body (а не body - label_band).
    # Юзер хоче "квадрат обрізаний по межам жетона" — мапа займає всю площу,
    # текст рендериться зверху (engraved/raised). label_band тепер лише
    # для розміщення тексту, не "відрізає" місце від мапи.
    map_h_mm = max(float(map_height_mm or body_h_mm), 4.0)
    # Use the real map slot width as the XY scale. The selected bbox is later
    # stretched into the slot so the generated map occupies the same rectangle
    # the user edited in the keychain preview.
    layout_scale_m_per_mm = source_w / map_w_mm
    export_scale = 1.0 / layout_scale_m_per_mm

    body_minx = 0.0
    body_miny = 0.0
    body_maxx = body_w_mm * layout_scale_m_per_mm
    body_maxy = body_h_mm * layout_scale_m_per_mm
    outer_m = max(loop_outer_radius_mm, 4.0) * layout_scale_m_per_mm
    min_hole_radius_mm = 1.5 if (base_shape or "").lower().replace("_", "-") == "token" else 1.6
    inner_m = min(
        max(loop_inner_radius_mm, min_hole_radius_mm),
        max(loop_outer_radius_mm - 1.8, min_hole_radius_mm),
    ) * layout_scale_m_per_mm
    corner_m = max(corner_radius_mm, 0.0) * layout_scale_m_per_mm
    label_band_h_m = max(label_band_height_mm, 0.0) * layout_scale_m_per_mm

    map_x = max(float(map_x_mm if map_x_mm is not None else 0.0), 0.0) * layout_scale_m_per_mm
    map_top = max(float(map_y_mm if map_y_mm is not None else 0.0), 0.0) * layout_scale_m_per_mm
    map_w = min(map_w_mm * layout_scale_m_per_mm, body_maxx - map_x)
    map_h = min(map_h_mm * layout_scale_m_per_mm, body_maxy - map_top)
    map_minx = body_minx + map_x
    map_maxx = map_minx + max(map_w, 1e-6)
    map_maxy = body_maxy - map_top
    map_miny = map_maxy - max(map_h, 1e-6)

    body = _keychain_body_shape(body_minx, body_miny, body_maxx, body_maxy, radius_m=corner_m, shape=base_shape, mm_to_world=layout_scale_m_per_mm)
    loop_margin_mm = max(float(loop_outer_radius_mm) * 0.85, 4.0)
    loop_center_x_mm_safe = min(
        max(float(loop_center_x_mm if loop_center_x_mm is not None else body_w_mm / 2.0), -loop_margin_mm),
        body_w_mm + loop_margin_mm,
    )
    loop_center_y_mm_safe = min(
        # DEFAULT: стандартне вушко на 50% — центр петлі ТОЧНО на верхній грані
        # (loop_center_y_mm=0) → половина кільця стирчить, половина в тілі.
        max(float(loop_center_y_mm if loop_center_y_mm is not None else 0.0), -loop_margin_mm),
        body_h_mm + loop_margin_mm,
    )
    loop_center_x = loop_center_x_mm_safe * layout_scale_m_per_mm
    loop_center_y = body_maxy - loop_center_y_mm_safe * layout_scale_m_per_mm
    # «ЗАКРІПЛЕНЕ» вушко (фідбек 2026-06-12: «закріпи кільце — в жетоні та
    # інших моделях»): кріплення не сміє триматись на волосині.
    # (а) Коло вушка мусить заходити в тіло ≥30% радіуса — якщо юзер відтягнув
    #     його далі, притягуємо центр до тіла (вушко висіло лише на шийці).
    import math as _math
    shape_norm = (base_shape or "rounded").lower().replace("_", "-")
    try:
        if not body.intersects(Point(loop_center_x, loop_center_y).buffer(outer_m * 0.7, resolution=24)):
            body_pt, _ = nearest_points(body, Point(loop_center_x, loop_center_y))
            dx = loop_center_x - body_pt.x
            dy = loop_center_y - body_pt.y
            d = _math.hypot(dx, dy) or 1.0
            pull = d - outer_m * 0.7
            loop_center_x -= dx / d * pull
            loop_center_y -= dy / d * pull
            # Симетричний дах (house): nearest_points на піку обирає одну з ДВОХ
            # рівновіддалених граней → зносить вушко вбік (док. баг «вушко будинку
            # справа»). Якщо петля була на центр. осі — повертаємо X строго в центр,
            # щоб тягло вертикально вниз, а не навскоси. Інші форми / зміщені петлі не чіпаємо.
            if shape_norm == "house":
                _cx = (body_w_mm / 2.0) * layout_scale_m_per_mm
                if abs(loop_center_x_mm_safe * layout_scale_m_per_mm - _cx) < 2.5 * layout_scale_m_per_mm:
                    loop_center_x = _cx
    except Exception:
        pass
    # (б) Отвір усередині тіла — ЛИШЕ для жетона (token): перемичка до краю ≥2.0мм,
    #     інакше кільце вириває тонку стінку — посуваємо отвір углиб тіла. Для решти
    #     форм петля периферійна (вушко на краю) і цей зсув НЕ застосовуємо: на
    #     СИМЕТРИЧНІЙ вершині (дах будинку / пік) nearest_points обирає одну з двох
    #     рівновіддалених граней → петлю «зносить» убік (баг: вушко будинку справа).
    try:
        pt = Point(loop_center_x, loop_center_y)
        if shape_norm == "token" and body.contains(pt):
            need = inner_m + 2.0 * layout_scale_m_per_mm
            edge_dist = body.boundary.distance(pt)
            if edge_dist < need:
                bp, _ = nearest_points(body.boundary, pt)
                dx = loop_center_x - bp.x
                dy = loop_center_y - bp.y
                d = _math.hypot(dx, dy) or 1.0
                push = need - edge_dist
                loop_center_x += dx / d * push
                loop_center_y += dy / d * push
    except Exception:
        pass
    outer_loop = _keychain_loop_outer(center_x=loop_center_x, center_y=loop_center_y, outer_m=outer_m, style=loop_style)
    inner_hole = _keychain_loop_inner(center_x=loop_center_x, center_y=loop_center_y, inner_m=inner_m, style=loop_style)
    # (в) Товща шийка кріплення: 0.72→0.85 ширини кільця, мін 2.2мм
    neck_half = max((outer_m - inner_m) * 0.85, _model_mm_to_world_m(2.2, export_scale))
    try:
        body_point, loop_point = nearest_points(body, Point(loop_center_x, loop_center_y))
        neck_line = LineString([body_point, loop_point])
        neck = neck_line.buffer(neck_half, cap_style=1, join_style=1, resolution=10)
    except Exception:
        neck = box(loop_center_x - neck_half, body_maxy - corner_m * 0.45, loop_center_x + neck_half, loop_center_y).buffer(
            neck_half * 0.55,
            resolution=10,
        )
    # (в2) ЗАПОВНЕННЯ УЩЕЛИНИ під петлею (серце): кругла петля над ВІДКРИТОЮ ущелиною
    #      серця читалась як булавка-маркер (📍 = коло + трикутник унизу). Будуємо
    #      «місток» = опукла оболонка (петля ∪ зріз тіла) у ВУЗЬКІЙ смузі довкола осі
    #      петлі, від дна ущелини до петлі → ущелина заповнюється, петля сидить
    #      міцно, а лоби й гостре вістря НЕ зачіпаються (смуга вузька + обмежена по
    #      висоті зверху ущелини, тож опукла оболонка не «роздуває» низ серця).
    if shape_norm in {"heart", "heart-l", "heart-r"}:
        try:
            axis = LineString([(loop_center_x, loop_center_y + outer_m * 1.5), (loop_center_x, body_miny)])
            hit = body.intersection(axis)
            anchor_y = float(hit.bounds[3]) if (hit is not None and not hit.is_empty) else body_maxy
            fill_w = max(outer_m * 1.4, neck_half * 2.0)
            fill_band = box(
                loop_center_x - fill_w,
                anchor_y - outer_m * 0.4,
                loop_center_x + fill_w,
                loop_center_y + outer_m * 1.5,
            )
            top_slice = body.intersection(fill_band)
            if not getattr(top_slice, "is_empty", True):
                gusset = unary_union([outer_loop, top_slice]).convex_hull.intersection(fill_band)
                neck = unary_union([neck, gusset])
        except Exception:
            pass
    if loop_angle_deg:
        try:
            outer_loop = affinity.rotate(outer_loop, loop_angle_deg, origin=(loop_center_x, loop_center_y), use_radians=False)
            inner_hole = affinity.rotate(inner_hole, loop_angle_deg, origin=(loop_center_x, loop_center_y), use_radians=False)
        except Exception:
            pass
    base = unary_union([body, outer_loop, neck]).difference(inner_hole)
    try:
        base = base.buffer(0)
    except Exception:
        pass
    # Прибрати мікро-порожнини (тонкі щілини від union-у містка ущелини на серці),
    # лишаючи СПРАВЖНІЙ отвір петлі — його площа на порядки більша за поріг.
    try:
        # < ~1.44мм² = артефакт містка ущелини (0.688мм²-void серця проходив поріг 0.64
        # → ~0.9мм наскрізний тунель біля петлі); СПРАВЖНІЙ отвір петлі на порядки більший.
        _min_void = (1.2 * layout_scale_m_per_mm) ** 2
        def _drop_tiny_holes(poly: Polygon) -> Polygon:
            kept = [r for r in poly.interiors if Polygon(r).area >= _min_void]
            return Polygon(poly.exterior, kept) if len(kept) != len(list(poly.interiors)) else poly
        if isinstance(base, Polygon):
            base = _drop_tiny_holes(base)
        elif hasattr(base, "geoms"):
            base = unary_union([
                _drop_tiny_holes(g) if isinstance(g, Polygon) else g for g in base.geoms
            ])
    except Exception:
        pass
    try:
        base_bounds = base.bounds
        export_size_mm = max(
            float(base_bounds[2] - base_bounds[0]),
            float(base_bounds[3] - base_bounds[1]),
        ) * float(export_scale)
    except Exception:
        export_size_mm = max(body_w_mm, body_h_mm)

    label_center_x = float(label_center_x_mm if label_center_x_mm is not None else body_w_mm / 2.0) * layout_scale_m_per_mm
    label_center_y = body_maxy - float(label_center_y_mm if label_center_y_mm is not None else (body_h_mm - label_band_height_mm / 2.0)) * layout_scale_m_per_mm
    label_w_mm = min(max(float(label_width_mm or body_w_mm * 0.86), 8.0), body_w_mm)
    label_w = label_w_mm * layout_scale_m_per_mm
    label_h = max(label_band_h_m, 1e-6)
    # ВАЖЛИВО: НЕ робимо swap ширини/висоти band при кутах 45–135°.
    # Раніше swap змінював розміри band → max_width/max_height у fit-логіці
    # ставали іншими → текст масштабувався по-іншому ніж у превʼю (юзер:
    # «розмір тексту інший ніж в готовій 3D моделі»). Поворот самих літер
    # (affinity.rotate на angle_deg) обробляє орієнтацію — swap не потрібен.
    label_band = box(
        label_center_x - label_w / 2.0,
        label_center_y - label_h / 2.0,
        label_center_x + label_w / 2.0,
        label_center_y + label_h / 2.0,
    )
    # CLEAR BAND: орієнтований прямокутник під текстом (для очистки карти).
    # Будуємо з СПРАВЖНІХ (не свопнутих) розмірів напису + невеликий відступ,
    # і ОБЕРТАЄМО на label_angle_deg → точно слідує за позицією/кутом тексту.
    # Карта (дороги/парки/будівлі/вода) вирізається в цій зоні, щоб напис
    # стояв на чистому фоні (юзер: «обрізати карту вокруг текста»).
    clear_margin_m = _model_mm_to_world_m(1.2, export_scale)
    clear_w = float(label_w_mm) * layout_scale_m_per_mm + 2.0 * clear_margin_m
    clear_h = float(label_band_h_m) + 2.0 * clear_margin_m
    label_clear_band = box(
        label_center_x - clear_w / 2.0,
        label_center_y - clear_h / 2.0,
        label_center_x + clear_w / 2.0,
        label_center_y + clear_h / 2.0,
    )
    if label_angle_deg:
        try:
            label_clear_band = affinity.rotate(
                label_clear_band, -float(label_angle_deg),
                origin=(label_center_x, label_center_y), use_radians=False,
            )
        except Exception:
            pass
    try:
        label_clear_band = label_clear_band.intersection(body).buffer(0)
    except Exception:
        pass
    # ВАЖЛИВО: content_area = ВЕСЬ body (а не map_slot ∩ body).
    # Це означає: карта завжди заповнює всю площу жетона, обрізається лише
    # формою body (oval/capsule/tag/rounded). Текст накладається зверху як
    # raised relief — він не "вирізає" місце з карти, просто стоїть зверху.
    content_area = body
    try:
        content_area = content_area.buffer(0)
    except Exception:
        pass
    # Раніше я вирізав label_band з content_area — це робило мапу обірваною
    # на пів-токені. Тепер мапа покриває ВСЮ площу, текст engraved зверху —
    # карвиться у base разом з усім контентом (карвинг видаляє і map layers).
    return {
        "base": base,
        "body": body,
        "content_area": content_area,
        "map_slot_area": box(map_minx, map_miny, map_maxx, map_maxy),
        "label_band": label_band,
        "label_clear_band": label_clear_band,
        "loop_hole": inner_hole,
        "source_bbox": box(minx, miny, maxx, maxy),
        "body_reference_xy_m": (body_maxx - body_minx, body_maxy - body_miny),
        # map_target_bounds = ВЕСЬ body bbox (не лише slot rect). Це гарантує
        # що COVER scale заповнює всю площу жетона, обрізається формою body.
        "map_target_bounds": (body_minx, body_miny, body_maxx, body_maxy),
        "layout_scale_m_per_mm": layout_scale_m_per_mm,
        "export_size_mm": export_size_mm,
    }


def _clip_geometry(geometry: Optional[BaseGeometry], clip: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return None
    if clip is None or getattr(clip, "is_empty", True):
        return geometry
    try:
        clipped = geometry.intersection(clip)
        if clipped is None or clipped.is_empty:
            return None
        return clipped.buffer(0)
    except Exception as exc:
        # Fail-open лишали мовчки → шар міг вийти за межі тіла без жодного сліду.
        # Спершу пробуємо repair-then-clip (типова причина — невалідний OSM-полігон);
        # лише як останній засіб віддаємо ОРИГІНАЛ (не None — щоб не згубити цілий шар
        # на тимчасовій топологічній помилці, яку нижній sanitize/повторний clip ще лікують).
        print(f"[FLAT PLATE] _clip_geometry intersection failed ({exc}); repair-then-clip")
        try:
            repaired = geometry.buffer(0).intersection(clip.buffer(0))
            if repaired is None or repaired.is_empty:
                return None
            return repaired.buffer(0)
        except Exception:
            return geometry


def _fit_geometry_into_bounds(
    geometry: Optional[BaseGeometry],
    *,
    source_bounds: tuple[float, float, float, float],
    target_bounds: tuple[float, float, float, float],
) -> Optional[BaseGeometry]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    src_minx, src_miny, src_maxx, src_maxy = source_bounds
    dst_minx, dst_miny, dst_maxx, dst_maxy = target_bounds
    src_w = max(float(src_maxx - src_minx), 1e-9)
    src_h = max(float(src_maxy - src_miny), 1e-9)
    dst_w = max(float(dst_maxx - dst_minx), 1e-9)
    dst_h = max(float(dst_maxy - dst_miny), 1e-9)
    scale = min(dst_w / src_w, dst_h / src_h)
    src_cx = (src_minx + src_maxx) * 0.5
    src_cy = (src_miny + src_maxy) * 0.5
    dst_cx = (dst_minx + dst_maxx) * 0.5
    dst_cy = (dst_miny + dst_maxy) * 0.5
    try:
        transformed = affinity.scale(geometry, xfact=scale, yfact=scale, origin=(src_cx, src_cy))
        transformed = affinity.translate(transformed, xoff=dst_cx - src_cx, yoff=dst_cy - src_cy)
        return transformed.buffer(0)
    except Exception:
        return geometry


def _stretch_geometry_into_bounds(
    geometry: Optional[BaseGeometry],
    *,
    source_bounds: tuple[float, float, float, float],
    target_bounds: tuple[float, float, float, float],
) -> Optional[BaseGeometry]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    src_minx, src_miny, src_maxx, src_maxy = source_bounds
    dst_minx, dst_miny, dst_maxx, dst_maxy = target_bounds
    src_w = max(float(src_maxx - src_minx), 1e-9)
    src_h = max(float(src_maxy - src_miny), 1e-9)
    dst_w = max(float(dst_maxx - dst_minx), 1e-9)
    dst_h = max(float(dst_maxy - dst_miny), 1e-9)
    try:
        transformed = affinity.scale(
            geometry,
            xfact=dst_w / src_w,
            yfact=dst_h / src_h,
            origin=(src_minx, src_miny),
        )
        transformed = affinity.translate(
            transformed,
            xoff=dst_minx - src_minx,
            yoff=dst_miny - src_miny,
        )
        return transformed.buffer(0)
    except Exception:
        return geometry


def _rotated_source_bounds(source_bounds: tuple[float, float, float, float], angle_deg: float) -> tuple[float, float, float, float]:
    angle = float(angle_deg or 0.0) % 360.0
    if abs(angle) <= 1e-6:
        return source_bounds
    try:
        src_box = box(*source_bounds)
        cx = (source_bounds[0] + source_bounds[2]) / 2.0
        cy = (source_bounds[1] + source_bounds[3]) / 2.0
        return tuple(float(v) for v in affinity.rotate(src_box, angle, origin=(cx, cy), use_radians=False).bounds)
    except Exception:
        return source_bounds


def _orient_then_stretch_geometry_into_bounds(
    geometry: Optional[BaseGeometry],
    *,
    source_bounds: tuple[float, float, float, float],
    target_bounds: tuple[float, float, float, float],
    angle_deg: float,
) -> Optional[BaseGeometry]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    angle = float(angle_deg or 0.0) % 360.0
    oriented = geometry
    oriented_bounds = source_bounds
    if abs(angle) > 1e-6:
        try:
            cx = (source_bounds[0] + source_bounds[2]) / 2.0
            cy = (source_bounds[1] + source_bounds[3]) / 2.0
            oriented = affinity.rotate(geometry, angle, origin=(cx, cy), use_radians=False)
            oriented_bounds = _rotated_source_bounds(source_bounds, angle)
        except Exception:
            oriented = geometry
            oriented_bounds = source_bounds
    return _stretch_geometry_into_bounds(
        oriented,
        source_bounds=oriented_bounds,
        target_bounds=target_bounds,
    )


def _fit_gdf_into_bounds(
    gdf: Optional[GeoDataFrame],
    *,
    source_bounds: tuple[float, float, float, float],
    target_bounds: tuple[float, float, float, float],
) -> Optional[GeoDataFrame]:
    if gdf is None or gdf.empty:
        return gdf
    fitted = gdf.copy()
    fitted.geometry = [
        _fit_geometry_into_bounds(geom, source_bounds=source_bounds, target_bounds=target_bounds)
        for geom in fitted.geometry
    ]
    return fitted


def _stretch_gdf_into_bounds(
    gdf: Optional[GeoDataFrame],
    *,
    source_bounds: tuple[float, float, float, float],
    target_bounds: tuple[float, float, float, float],
) -> Optional[GeoDataFrame]:
    if gdf is None or gdf.empty:
        return gdf
    fitted = gdf.copy()
    fitted.geometry = [
        _stretch_geometry_into_bounds(geom, source_bounds=source_bounds, target_bounds=target_bounds)
        for geom in fitted.geometry
    ]
    return fitted


def _orient_then_stretch_gdf_into_bounds(
    gdf: Optional[GeoDataFrame],
    *,
    source_bounds: tuple[float, float, float, float],
    target_bounds: tuple[float, float, float, float],
    angle_deg: float,
) -> Optional[GeoDataFrame]:
    if gdf is None or gdf.empty:
        return gdf
    fitted = gdf.copy()
    fitted.geometry = [
        _orient_then_stretch_geometry_into_bounds(
            geom,
            source_bounds=source_bounds,
            target_bounds=target_bounds,
            angle_deg=angle_deg,
        )
        for geom in fitted.geometry
    ]
    return fitted


def _clip_buildings_to_content(gdf: Optional[GeoDataFrame], content_area: Optional[BaseGeometry]) -> Optional[GeoDataFrame]:
    if gdf is None or gdf.empty or content_area is None or getattr(content_area, "is_empty", True):
        return gdf
    clipped = gdf.copy()
    kept_rows = []
    kept_geoms = []
    for idx, geom in clipped.geometry.items():
        try:
            next_geom = geom.intersection(content_area)
            if next_geom is None or next_geom.is_empty or float(next_geom.area) <= 0:
                continue
            kept_rows.append(idx)
            kept_geoms.append(next_geom.buffer(0))
        except Exception:
            continue
    if not kept_rows:
        return clipped.iloc[0:0].copy()
    clipped = clipped.loc[kept_rows].copy()
    clipped.geometry = kept_geoms
    return clipped


def _normalize_label_text(text: str) -> str:
    value = (text or "").upper().translate(_CYR_TO_LAT)
    return "".join(ch if ch in _FONT_5X7 else " " for ch in value).strip()[:28]


def _ttf_text_polygons(text: str, height_m: float, font_family: str = "DejaVu Sans") -> list[BaseGeometry]:
    """Рендерить текст через matplotlib TTF + повертає список shapely Polygons.
    Дає СПРАВЖНІ smooth letters замість пікселього 5×7 шрифту."""
    try:
        from matplotlib.textpath import TextToPath
        from matplotlib.font_manager import FontProperties
        from matplotlib.path import Path as MPath
        from shapely.geometry import Polygon as _Poly
        from shapely.ops import unary_union
    except Exception:
        return []
    if not text or height_m <= 0:
        return []
    tp = TextToPath()
    # size в "font points" — потім масштабуємо у метри
    fp = FontProperties(family=font_family, weight="bold")
    try:
        # get_text_path returns vertices + codes (M, L, C, Z)
        size_pt = 100.0  # рендеримо у великому масштабі для точності
        verts, codes = tp.get_text_path(fp, text)
    except Exception:
        return []
    if len(verts) == 0:
        return []
    # Збираємо paths з Bezier curves → апроксимуємо лінійно
    mpath = MPath(verts, codes)
    # to_polygons() повертає список замкнутих polygons (rings)
    polys = mpath.to_polygons(closed_only=False)
    if not polys:
        return []
    # Обчислюємо bbox для масштабування у потрібну висоту
    all_y = [v[1] for v in verts]
    text_height_pt = max(all_y) - min(all_y) if all_y else 1.0
    if text_height_pt <= 0:
        return []
    scale = height_m / text_height_pt
    # Створюємо shapely polygons, шкалюємо, нормалізуємо позицію (start at origin)
    all_x = [v[0] for v in verts]
    offset_x = min(all_x) if all_x else 0
    offset_y = min(all_y) if all_y else 0
    rings = []
    for poly_pts in polys:
        if len(poly_pts) < 3:
            continue
        scaled = [((x - offset_x) * scale, (y - offset_y) * scale) for x, y in poly_pts]
        try:
            p = _Poly(scaled)
            if not p.is_valid:
                p = p.buffer(0)
            if p is not None and not p.is_empty and p.area > 0:
                rings.append(p)
        except Exception:
            continue
    if not rings:
        return []
    # EVEN-ODD FILL: кільце O/A/P/R/B складається з ЗОВНІШНЬОГО контуру та
    # ВНУТРІШНЬОГО (counter). to_polygons() повертає їх окремими кільцями.
    # Раніше ми робили unary_union(filled_rings) — внутрішній «лічильник»
    # заповнювався суцільним диском і ДІРКА зникала (літери зливались).
    # Тут визначаємо вкладеність кожного кільця: парна глибина = тіло літери,
    # непарна = дірка (counter). Будуємо Polygon(shell, holes).
    indexed = sorted(rings, key=lambda g: g.area, reverse=True)
    reps = [g.representative_point() for g in indexed]
    depth = []
    for i, g in enumerate(indexed):
        d = 0
        for j, q in enumerate(indexed):
            if i == j:
                continue
            # q більший і містить точку g → g вкладений у q
            if q.area > g.area and q.contains(reps[i]):
                d += 1
        depth.append(d)
    result: list[BaseGeometry] = []
    for si, shell in enumerate(indexed):
        if depth[si] % 2 != 0:
            continue  # це дірка — обробляється як hole свого shell
        holes = []
        for j, inner in enumerate(indexed):
            if j == si:
                continue
            # пряма дірка цього shell: глибина+1 і міститься всередині
            if depth[j] == depth[si] + 1 and shell.contains(reps[j]):
                try:
                    holes.append(list(inner.exterior.coords))
                except Exception:
                    pass
        try:
            poly = _Poly(list(shell.exterior.coords), holes)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly is not None and not poly.is_empty:
                result.append(poly)
        except Exception:
            result.append(shell)
    if not result:
        return rings
    return result


def _compute_text_letter_polygon(
    *,
    text: str,
    body_geometry: BaseGeometry,
    label_band_geometry: BaseGeometry,
    text_height_m: float,
    angle_deg: float,
    min_stroke_m: float,
    max_width: float,
    max_height: float,
) -> Optional[BaseGeometry]:
    """Compute the 2D letter polygon (positioned, rotated, clipped to body) for
    carving out of map layers before meshes are built.  Same math as the TTF
    branch of build_keychain_label_mesh so the carve and the raised text align
    perfectly."""
    label = _normalize_label_text(text)
    if not label or text_height_m <= 0:
        return None
    band_minx, band_miny, band_maxx, band_maxy = label_band_geometry.bounds
    try:
        ttf_polys = _ttf_text_polygons(label, height_m=text_height_m, font_family="DejaVu Sans")
        if not ttf_polys:
            return None
        from shapely.ops import unary_union as _uup
        text_polygon = _uup(ttf_polys).buffer(0)
        # stroke thickening (counter-safe) — same as build_keychain_label_mesh
        min_stroke = float(min_stroke_m or 0.0)
        if min_stroke > 0:
            natural_stroke = 0.16 * float(text_height_m)
            grow = min(max((min_stroke - natural_stroke) * 0.5, 0.0), 0.12 * float(text_height_m))
            if grow > 1e-9:
                holes = []
                for _g in (text_polygon.geoms if hasattr(text_polygon, "geoms") else [text_polygon]):
                    try:
                        for _i in _g.interiors:
                            _hp = Polygon(_i)
                            if _hp.is_valid and not _hp.is_empty:
                                holes.append(_hp)
                    except Exception:
                        pass
                thickened = text_polygon.buffer(grow, join_style=1, cap_style=1).buffer(0)
                if holes:
                    try:
                        from shapely.ops import unary_union as _uu3
                        counters = _uu3(holes).buffer(-grow * 0.5).buffer(0)
                        if counters is not None and not counters.is_empty:
                            thickened = thickened.difference(counters).buffer(0)
                    except Exception:
                        pass
                if thickened is not None and not thickened.is_empty:
                    text_polygon = thickened
        # fit + center + rotate + clip
        t_minx, t_miny, t_maxx, t_maxy = text_polygon.bounds
        text_w = t_maxx - t_minx; text_h = t_maxy - t_miny
        fit_scale = 1.0
        if text_w > max_width: fit_scale = min(fit_scale, max_width / text_w)
        if text_h > max_height: fit_scale = min(fit_scale, max_height / text_h)
        if fit_scale < 1.0:
            text_polygon = affinity.scale(text_polygon, xfact=fit_scale, yfact=fit_scale, origin=(t_minx, t_miny))
            t_minx, t_miny, t_maxx, t_maxy = text_polygon.bounds
            text_w = t_maxx - t_minx; text_h = t_maxy - t_miny
        cx = (band_minx + band_maxx) / 2; cy = (band_miny + band_maxy) / 2
        text_polygon = affinity.translate(text_polygon, xoff=cx - (t_minx + text_w / 2), yoff=cy - (t_miny + text_h / 2))
        if angle_deg:
            # SVG Y-axis goes DOWN (positive rotation = clockwise), but shapely/model
            # Y-axis goes UP (positive = counter-clockwise). Negate angle to match preview.
            text_polygon = affinity.rotate(text_polygon, -float(angle_deg), origin=(cx, cy), use_radians=False)
        text_polygon = text_polygon.intersection(body_geometry).buffer(0)
        if text_polygon is None or text_polygon.is_empty:
            return None
        return text_polygon
    except Exception as exc:
        print(f"[KEYCHAIN] _compute_text_letter_polygon failed: {exc}")
        return None


def build_keychain_label_mesh(
    text: str,
    *,
    body_geometry: BaseGeometry,
    label_band_geometry: BaseGeometry,
    bottom_z_m: float,
    thickness_m: float,
    text_height_m: float,
    color: list[int],
    stroke_width_m: float = 0.0,
    angle_deg: float = 0.0,
    min_stroke_m: float = 0.0,
    font_style: str = "block",
    precomputed_polygon: Optional[BaseGeometry] = None,
) -> Optional[trimesh.Trimesh]:
    label = _normalize_label_text(text)
    if not label or thickness_m <= 0 or text_height_m <= 0:
        return None
    band_minx, band_miny, band_maxx, band_maxy = label_band_geometry.bounds
    max_width = max((band_maxx - band_minx) * 0.96, 1e-6)
    max_height = max((band_maxy - band_miny) * 0.92, 1e-6)

    # SHORT-CIRCUIT: якщо полігон уже порахований вище (carve-pass), реюзаємо —
    # уникаємо подвійного TTF-рендеру і гарантуємо повний збіг із вирізом.
    if precomputed_polygon is not None and not getattr(precomputed_polygon, "is_empty", True):
        try:
            combined = build_flat_layer_mesh_from_mask(
                precomputed_polygon, bottom_z_m=bottom_z_m, thickness_m=thickness_m,
                color=color, min_area_m2=1e-12,
            )
            if combined is not None:
                try: combined.metadata["text_polygon"] = precomputed_polygon
                except Exception: pass
                return _with_color(combined, color)
        except Exception as exc:
            print(f"[KEYCHAIN] precomputed polygon mesh build failed ({exc}); falling back to re-render")

    # ПРОПЕРНИЙ TTF РЕНДЕР через matplotlib — smooth glyphs замість 5×7 pixel art.
    # Cyrillic вже переведена в latin через _normalize_label_text.
    try:
        ttf_polys = _ttf_text_polygons(label, height_m=text_height_m, font_family="DejaVu Sans")
        if ttf_polys:
            from shapely.ops import unary_union as _uu
            text_polygon = _uu(ttf_polys).buffer(0)
            # STROKE THICKENING для FDM-друку: тонкі штрихи bold-гліфів можуть бути
            # вужчими за надійний друкований мінімум (≥0.8мм при соплі 0.4мм).
            # Природний штрих DejaVu Bold ≈ 0.16 × cap height. Якщо він менший за
            # потрібний min_stroke — дилейтимо контур на половину різниці з кожного
            # боку (round join), щоб довести тонкі частини до друкованого мінімуму,
            # не зливаючи сусідні літери. Research: Mandarin3D / JLC3DP / Bambu wiki.
            try:
                min_stroke = float(min_stroke_m or 0.0)
                if min_stroke > 0:
                    natural_stroke = 0.16 * float(text_height_m)
                    grow = (min_stroke - natural_stroke) * 0.5
                    # Cap на 0.20 × cap height: старий 0.12 ТИХО не давав дійти до
                    # 0.8мм мінімуму для тексту <2мм (1.6мм→штрих 0.64мм, ламкий).
                    # 0.20 робить 0.8мм досяжним до ~1.4мм; counters лишаються
                    # відкритими завдяки re-cut нижче (ерозія holes на grow·0.5).
                    grow = min(max(grow, 0.0), 0.20 * float(text_height_m))
                    if grow > 1e-9:
                        # COUNTER-SAFE дилейт: buffer(+grow) товщає штрихи, але
                        # ЗАКРИВАЄ внутрішні дірки (counters A/P/R/O/B). Тому
                        # після дилейту ПОВТОРНО прорізаємо counters (трохи
                        # ерозовані), щоб вони лишались відкритими.
                        from shapely.ops import unary_union as _uu2
                        holes = []
                        for g in (text_polygon.geoms if hasattr(text_polygon, "geoms") else [text_polygon]):
                            try:
                                for interior in g.interiors:
                                    hp = Polygon(interior)
                                    if hp.is_valid and not hp.is_empty:
                                        holes.append(hp)
                            except Exception:
                                continue
                        thickened = text_polygon.buffer(grow, join_style=1, cap_style=1).buffer(0)
                        if holes:
                            try:
                                holes_union = _uu2(holes)
                                # Counter ерозуємо лише наполовину grow → лишається видимим.
                                counters = holes_union.buffer(-grow * 0.5).buffer(0)
                                if counters is not None and not counters.is_empty:
                                    thickened = thickened.difference(counters).buffer(0)
                            except Exception:
                                pass
                        if thickened is not None and not thickened.is_empty:
                            text_polygon = thickened
            except Exception:
                pass
            # Bbox тексту після рендеру
            t_minx, t_miny, t_maxx, t_maxy = text_polygon.bounds
            text_w = t_maxx - t_minx
            text_h = t_maxy - t_miny
            # FIT у band по ОБОХ вимірах (ширина І висота), зберігаючи пропорції,
            # ТІЛЬКИ зменшуємо. Без height-fit текст з великою висотою (або після
            # повороту) вилазив за межі жетона і займав усю площу (юзер: «помилки
            # із текстом» — напис гігантський). Тепер напис гарантовано вміщається
            # у свою смугу, як у превʼю.
            fit_scale = 1.0
            if text_w > max_width:
                fit_scale = min(fit_scale, max_width / text_w)
            if text_h > max_height:
                fit_scale = min(fit_scale, max_height / text_h)
            if fit_scale < 1.0:
                text_polygon = affinity.scale(text_polygon, xfact=fit_scale, yfact=fit_scale, origin=(t_minx, t_miny))
                t_minx, t_miny, t_maxx, t_maxy = text_polygon.bounds
                text_w = t_maxx - t_minx
                text_h = t_maxy - t_miny
            # Центруємо у band
            cx = (band_minx + band_maxx) / 2
            cy = (band_miny + band_maxy) / 2
            offset_x = cx - (t_minx + text_w / 2)
            offset_y = cy - (t_miny + text_h / 2)
            text_polygon = affinity.translate(text_polygon, xoff=offset_x, yoff=offset_y)
            # Поворот: SVG Y-axis goes DOWN (positive = clockwise) but shapely
            # Y-axis goes UP (positive = counter-clockwise), so negate to match preview.
            if angle_deg:
                text_polygon = affinity.rotate(text_polygon, -float(angle_deg), origin=(cx, cy), use_radians=False)
            # Кліп до body щоб не вилазив
            text_polygon = text_polygon.intersection(body_geometry).buffer(0)
            if text_polygon is None or text_polygon.is_empty:
                return None
            combined = build_flat_layer_mesh_from_mask(
                text_polygon, bottom_z_m=bottom_z_m, thickness_m=thickness_m,
                color=color, min_area_m2=1e-12,
            )
            if combined is None:
                return None
            try:
                combined.metadata["text_polygon"] = text_polygon
            except Exception:
                pass
            return _with_color(combined, color)
    except Exception as exc:
        print(f"[KEYCHAIN] TTF text render failed, fallback to pixel font: {exc}")
    # FALLBACK: старий 5×7 pixel font
    cell = max(float(text_height_m) / 7.0, float(min_stroke_m or 0.0))
    stroke_m = max(float(stroke_width_m or 0.0), float(min_stroke_m or 0.0), cell)
    style = (font_style or "block").lower().replace("_", "-")
    x_scale = 1.25 if style == "wide" else 0.82 if style in {"condensed", "narrow"} else 1.0
    gap_units = 1.15 if style == "wide" else 0.75 if style in {"condensed", "narrow"} else 1.0
    char_units = [5.0 * x_scale + gap_units if ch != " " else 3.0 * x_scale for ch in label]
    raw_width = max(sum(char_units) - 1, 1) * cell
    if raw_width > max_width:
        max_units = max_width / max(cell, 1e-9)
        used = 0
        kept = []
        for ch, units in zip(label, char_units):
            next_used = used + (units if not kept else units)
            if next_used > max_units:
                break
            kept.append(ch)
            used += units
        label = "".join(kept).strip()
        if not label:
            return None
        char_units = [5.0 * x_scale + gap_units if ch != " " else 3.0 * x_scale for ch in label]
    raw_width = max(sum(char_units) - 1, 1) * cell
    start_x = (band_minx + band_maxx - raw_width) / 2.0
    start_y = band_miny + max((band_maxy - band_miny) - 7.0 * cell, 0.0) / 2.0

    glyph_pixels: list[Polygon] = []
    cursor_x = start_x
    for ch in label:
        glyph = _FONT_5X7.get(ch, _FONT_5X7[" "])
        for row_idx, row in enumerate(glyph):
            for col_idx, bit in enumerate(row):
                if bit != "1":
                    continue
                x0 = cursor_x + col_idx * cell * x_scale
                y0 = start_y + (6 - row_idx) * cell
                bleed = max((stroke_m - cell) * 0.5, 0.0)
                glyph_pixels.append(box(x0 - bleed, y0 - bleed, x0 + cell * x_scale + bleed, y0 + cell + bleed))
        cursor_x += (5.0 * x_scale + gap_units if ch != " " else 3.0 * x_scale) * cell
    if not glyph_pixels:
        return None
    text_geometries: list[BaseGeometry] = []
    cx = (band_minx + band_maxx) * 0.5
    cy = (band_miny + band_maxy) * 0.5
    clip_geometry = label_band_geometry.intersection(body_geometry)
    if clip_geometry is None or getattr(clip_geometry, "is_empty", True):
        return None
    for pixel in glyph_pixels:
        try:
            geom = pixel
            if angle_deg:
                geom = affinity.rotate(
                    geom,
                    float(angle_deg),
                    origin=(cx, cy),
                    use_radians=False,
                )
            geom = geom.intersection(clip_geometry)
            if geom is not None and not getattr(geom, "is_empty", True) and float(getattr(geom, "area", 0.0) or 0.0) > 0:
                text_geometries.append(geom)
        except Exception:
            continue
    if not text_geometries:
        return None
    # ENGRAVED TEXT: повертаємо UNION гліфів (без extrude в mesh).
    # Caller використає цей полігон щоб ВИРІЗАТИ текст з base
    # і заповнити нижчою плитою на (base_top - depth).
    from shapely.ops import unary_union
    try:
        text_polygon = unary_union(text_geometries).buffer(0)
    except Exception:
        text_polygon = text_geometries[0]
    if text_polygon is None or text_polygon.is_empty:
        return None
    # Каллер вирішить: extrude як raised mesh АБО використовувати як carve mask
    combined = build_flat_layer_mesh_from_mask(
        text_polygon,
        bottom_z_m=bottom_z_m,
        thickness_m=thickness_m,
        color=color,
        min_area_m2=1e-12,
    )
    if combined is None:
        return None
    # Прикріплюємо polygon як attribute для caller-а (engrave usage)
    try:
        combined.metadata["text_polygon"] = text_polygon
    except Exception:
        pass
    return _with_color(combined, color)


def _rotate_meshes_for_keychain_layout(
    *,
    meshes: list[Optional[trimesh.Trimesh]],
    building_meshes: list[trimesh.Trimesh],
    angle_deg: float,
    origin_xy: tuple[float, float],
    origin_z: float,
) -> None:
    angle = float(angle_deg or 0.0) % 360.0
    if abs(angle) <= 1e-6:
        return
    matrix = trimesh.transformations.rotation_matrix(
        np.deg2rad(angle),
        [0.0, 0.0, 1.0],
        [float(origin_xy[0]), float(origin_xy[1]), float(origin_z)],
    )
    for mesh in meshes:
        if mesh is not None:
            mesh.apply_transform(matrix)
    for mesh in building_meshes:
        if mesh is not None:
            mesh.apply_transform(matrix)


def _build_keychain_base_parts(
    base_mask: BaseGeometry,
    *,
    base_top_m: float,
    back_text_poly: Optional[BaseGeometry] = None,
    engrave_m: float = 0.0,
    top_cut_poly: Optional[BaseGeometry] = None,
    top_cut_depth_m: float = 0.0,
    min_area_m2: float = 0.001,
) -> tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh]]:
    """База брелка (чисті watertift-екструди, БЕЗ 3D-boolean).
    Без cut'ів — одна суцільна плита. `back_text_poly`+`engrave_m` → НИЖНІЙ шар
    (0..engrave) з гравіюванням, видно знизу (також магніт-кишеня). НОВЕ:
    `top_cut_poly`+`top_cut_depth_m` → ПАЗ у ВЕРХНЬОМУ шарі для вставки окремої
    деталі-будинку зверху. Сумісно: коли top_cut=None — поведінка ІДЕНТИЧНА старій."""
    has_back = (
        back_text_poly is not None
        and not getattr(back_text_poly, "is_empty", True)
        and 1e-9 < engrave_m < base_top_m - 1e-9
    )
    has_top = (
        top_cut_poly is not None
        and not getattr(top_cut_poly, "is_empty", True)
        and 1e-9 < top_cut_depth_m < base_top_m - 1e-9
    )
    if not has_back and not has_top:
        return (
            build_flat_layer_mesh_from_mask(
                base_mask, bottom_z_m=0.0, thickness_m=base_top_m,
                color=LAYER_COLORS["base"], min_area_m2=min_area_m2,
            ),
            None,
        )
    try:
        bottom_mesh = None
        mid_z0 = 0.0
        if has_back:
            bottom_mesh = build_flat_layer_mesh_from_mask(
                base_mask.difference(back_text_poly).buffer(0), bottom_z_m=0.0,
                thickness_m=engrave_m, color=LAYER_COLORS["base"], min_area_m2=1e-12,
            )
            mid_z0 = engrave_m
        mid_top = (base_top_m - top_cut_depth_m) if has_top else base_top_m
        main_layers = []
        if mid_top - mid_z0 > 1e-9:
            main_layers.append(build_flat_layer_mesh_from_mask(
                base_mask, bottom_z_m=mid_z0, thickness_m=mid_top - mid_z0,
                color=LAYER_COLORS["base"], min_area_m2=min_area_m2,
            ))
        if has_top:  # верхній шар з ПАЗОМ (mask − pocket)
            main_layers.append(build_flat_layer_mesh_from_mask(
                base_mask.difference(top_cut_poly).buffer(0), bottom_z_m=mid_top,
                thickness_m=top_cut_depth_m, color=LAYER_COLORS["base"], min_area_m2=1e-12,
            ))
        main_layers = [m for m in main_layers if m is not None]
        if main_layers and (not has_back or bottom_mesh is not None):
            main = trimesh.util.concatenate(main_layers) if len(main_layers) > 1 else main_layers[0]
            return main, bottom_mesh
    except Exception as exc:
        print(f"[KEYCHAIN] base split (back/top cut) failed ({exc}); solid base fallback")
    return (
        build_flat_layer_mesh_from_mask(
            base_mask, bottom_z_m=0.0, thickness_m=base_top_m,
            color=LAYER_COLORS["base"], min_area_m2=min_area_m2,
        ),
        None,
    )


def build_keychain_rim_mesh(
    *,
    base_geometry: BaseGeometry,
    bottom_z_m: float,
    width_m: float,
    height_m: float,
) -> Optional[trimesh.Trimesh]:
    if width_m <= 0 or height_m <= 0 or base_geometry is None or getattr(base_geometry, "is_empty", True):
        return None
    try:
        inner = base_geometry.buffer(-float(width_m), join_style=1)
        if inner is None or getattr(inner, "is_empty", True):
            return None
        rim = base_geometry.difference(inner).buffer(0)
    except Exception:
        return None
    return build_flat_layer_mesh_from_mask(
        rim,
        bottom_z_m=bottom_z_m,
        thickness_m=height_m,
        color=LAYER_COLORS["rim"],  # ободок = чорний (раніше помилково base)
        min_area_m2=max(width_m * width_m * 0.25, 1e-10),
    )


def _clamp_mesh_height(mesh: trimesh.Trimesh, *, min_height_m: float, max_height_m: float) -> trimesh.Trimesh:
    if mesh.vertices is None or len(mesh.vertices) == 0 or max_height_m <= 0:
        return mesh
    z = mesh.vertices[:, 2]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    current = max(z_max - z_min, 1e-9)
    target = min(max(current, max(min_height_m, 0.0)), max_height_m)
    if abs(target - current) <= 1e-9:
        return mesh
    verts = np.asarray(mesh.vertices, dtype=float).copy()
    verts[:, 2] = (verts[:, 2] - z_min) * (target / current)
    mesh.vertices = verts
    return mesh


def _set_mesh_height(mesh: trimesh.Trimesh, *, target_height_m: float) -> trimesh.Trimesh:
    if mesh.vertices is None or len(mesh.vertices) == 0 or target_height_m <= 0:
        return mesh
    z = mesh.vertices[:, 2]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    current = max(z_max - z_min, 1e-9)
    verts = np.asarray(mesh.vertices, dtype=float).copy()
    verts[:, 2] = (verts[:, 2] - z_min) * (float(target_height_m) / current)
    mesh.vertices = verts
    return mesh


def _mesh_manifest(mesh: Optional[trimesh.Trimesh], *, scale_factor: float) -> dict[str, Any]:
    if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0 or mesh.faces is None:
        return {"present": False, "vertices": 0, "faces": 0}
    try:
        bounds = np.asarray(mesh.bounds, dtype=float)
        extents_mm = (bounds[1] - bounds[0]) * float(scale_factor)
        z_min_mm = float(bounds[0][2]) * float(scale_factor)
        z_max_mm = float(bounds[1][2]) * float(scale_factor)
    except Exception:
        extents_mm = np.zeros(3, dtype=float)
        z_min_mm = 0.0
        z_max_mm = 0.0
    return {
        "present": True,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "size_mm": [round(float(v), 3) for v in extents_mm.tolist()],
        "z_min_mm": round(z_min_mm, 3),
        "z_max_mm": round(z_max_mm, 3),
    }


def _fetch_zone_heightfield_provider(
    *,
    request: Any,
    zone: Any,
    global_center: Any,
    max_axis_cells: int = 220,
):
    """C3 ТОПО: качає DEM-висоти (Terrarium) для bbox зони у ЛОКАЛЬНИХ метрах
    і повертає TerrainProvider для інтерполяції. None — якщо висоти недоступні
    (генерація тоді тихо лишає плоску базу)."""
    from scipy.ndimage import gaussian_filter

    from services.elevation_api import get_elevation_abs_meters_from_api
    from services.terrain_provider import TerrainProvider

    minx, miny, maxx, maxy = (float(v) for v in zone.bbox_meters)
    pad = 20.0
    width = max(maxx - minx, 1.0) + 2 * pad
    height = max(maxy - miny, 1.0) + 2 * pad
    long_axis = max(width, height)
    nx = max(32, min(int(round(max_axis_cells * width / long_axis)), max_axis_cells))
    ny = max(32, min(int(round(max_axis_cells * height / long_axis)), max_axis_cells))
    xs = np.linspace(minx - pad, maxx + pad, nx)
    ys = np.linspace(miny - pad, maxy + pad, ny)
    X, Y = np.meshgrid(xs, ys)

    cx, cy = global_center.get_center_utm()
    source_crs = global_center.get_utm_crs()
    latlon_bbox = (
        float(getattr(request, "north", 0.0)),
        float(getattr(request, "south", 0.0)),
        float(getattr(request, "east", 0.0)),
        float(getattr(request, "west", 0.0)),
    )
    zoom = int(getattr(request, "terrarium_zoom", 13) or 13)
    Z = get_elevation_abs_meters_from_api(latlon_bbox, X + cx, Y + cy, source_crs, zoom)
    if Z is None:
        print("[KEYCHAIN TOPO] DEM unavailable — keeping flat base")
        return None
    Z = np.asarray(Z, dtype=np.float64)
    finite = np.isfinite(Z)
    if not np.any(finite):
        print("[KEYCHAIN TOPO] DEM all-NaN — keeping flat base")
        return None
    fill = float(np.nanmedian(Z[finite]))
    Z[~finite] = fill
    # Terrarium-викиди (бачили tower-bases у нормальному пайплайні) — клампимо
    med = float(np.nanmedian(Z))
    Z = np.clip(Z, med - 3000.0, med + 5000.0)
    try:
        Z = gaussian_filter(Z, sigma=1.2)
    except Exception:
        pass
    print(
        f"[KEYCHAIN TOPO] Heightfield {nx}x{ny}, z={zoom}: "
        f"{float(np.min(Z)):.1f}..{float(np.max(Z)):.1f}m"
    )
    return TerrainProvider(X, Y, Z)


def _keychain_topo_inverse_map(
    *,
    unwrap_params: Optional[dict],
    source_bounds: tuple[float, float, float, float],
    target_bounds: tuple[float, float, float, float],
    angle_deg: float,
):
    """Обернене відображення: координати ЖЕТОНА (layout) → координати ЗОНИ
    (локальні метри). Інверсія до _xform/unwrap у run_flat_plate_pipeline."""
    import math

    if unwrap_params is not None:
        p = unwrap_params
        s = max(p["tgt_w"] / p["rect_w"], p["tgt_h"] / p["rect_h"])
        ang = math.radians(float(p["angle"]))
        cos_a, sin_a = math.cos(ang), math.sin(ang)

        def _inv(points: np.ndarray) -> np.ndarray:
            pts = np.asarray(points, dtype=float)
            qx = (pts[:, 0] - p["tgt_cx"]) / s
            qy = (pts[:, 1] - p["tgt_cy"]) / s
            # forward робив rotate(-angle) → інверсія rotate(+angle)
            x = qx * cos_a - qy * sin_a + p["cx_src"]
            y = qx * sin_a + qy * cos_a + p["cy_src"]
            return np.column_stack([x, y])

        return _inv

    # Fallback-гілка (legacy orient+stretch): unstretch target→oriented, потім rotate(-angle)
    oriented = _rotated_source_bounds(source_bounds, angle_deg)
    o_minx, o_miny, o_maxx, o_maxy = (float(v) for v in oriented)
    t_minx, t_miny, t_maxx, t_maxy = (float(v) for v in target_bounds)
    sx = max(o_maxx - o_minx, 1e-9) / max(t_maxx - t_minx, 1e-9)
    sy = max(o_maxy - o_miny, 1e-9) / max(t_maxy - t_miny, 1e-9)
    src_cx = (source_bounds[0] + source_bounds[2]) / 2.0
    src_cy = (source_bounds[1] + source_bounds[3]) / 2.0
    import math as _math
    ang_b = _math.radians(-float(angle_deg or 0.0))
    cos_b, sin_b = _math.cos(ang_b), _math.sin(ang_b)

    def _inv_legacy(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        ox = o_minx + (pts[:, 0] - t_minx) * sx
        oy = o_miny + (pts[:, 1] - t_miny) * sy
        dx = ox - src_cx
        dy = oy - src_cy
        x = dx * cos_b - dy * sin_b + src_cx
        y = dx * sin_b + dy * cos_b + src_cy
        return np.column_stack([x, y])

    return _inv_legacy


def _extrude_polygon_heightfield(
    poly: Polygon,
    *,
    bottom_z_m: float,
    top_z_func,
    max_area_m2: float,
) -> Optional[trimesh.Trimesh]:
    """Watertight-екструд полігона з ПЕРЕМІННОЮ верхньою гранню (heightfield).
    Та сама техніка, що _extrude_polygon_prism, але triangle-двигун додає
    Steiner-точки всередині (q28 + max area) і верхній z береться з top_z_func."""
    if poly is None or poly.is_empty or float(poly.area) <= 0:
        return None
    try:
        cap_points, tri_idx = trimesh.creation.triangulate_polygon(
            poly,
            triangle_args=f"pq28a{max_area_m2:.12f}",
            engine="triangle",
        )
    except Exception as exc:
        print(f"[KEYCHAIN TOPO] dense triangulation failed: {exc}")
        return None
    cap_points = np.asarray(cap_points, dtype=float)[:, :2]
    tri_idx = np.asarray(tri_idx, dtype=np.int64).reshape((-1, 3))
    if len(cap_points) == 0 or len(tri_idx) == 0:
        return None
    top_z = np.asarray(top_z_func(cap_points), dtype=float).reshape(-1)
    if len(top_z) != len(cap_points):
        return None
    n = len(cap_points)
    vertices = np.vstack(
        [
            np.column_stack([cap_points, np.full(n, float(bottom_z_m))]),
            np.column_stack([cap_points, top_z]),
        ]
    )
    faces: list[list[int]] = []
    edge_counts: dict[tuple[int, int], int] = {}
    for i0, i1, i2 in tri_idx:
        i0, i1, i2 = int(i0), int(i1), int(i2)
        faces.append([i2, i1, i0])  # низ — нормаль донизу
        faces.append([n + i0, n + i1, n + i2])  # верх
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            key = (min(a, b), max(a, b))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    # Стінки: межові ребра (зустрілись 1 раз у cap-тріангуляції)
    for (a, b), count in edge_counts.items():
        if count != 1:
            continue
        faces.append([a, b, n + b])
        faces.append([a, n + b, n + a])
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    try:
        mesh.merge_vertices(digits_vertex=8)
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_winding(mesh)
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def _build_keychain_topo_base(
    *,
    request: Any,
    zone: Any,
    global_center: Any,
    base_mask: BaseGeometry,
    relief_zone: Optional[BaseGeometry],
    base_top_m: float,
    relief_m: float,
    feather_m: float,
    unwrap_params: Optional[dict],
    source_bounds: tuple[float, float, float, float],
    target_bounds: tuple[float, float, float, float],
    map_rotation_deg: float,
    back_text_poly: Optional[BaseGeometry],
    engrave_m: float,
    export_scale_factor: float,
) -> tuple[Optional[trimesh.Trimesh], Optional[trimesh.Trimesh]]:
    """C3 ТОПО-БРЕЛОК: база жетона з heightfield-рельєфом на верхній грані.
    Рельєф нормалізується по видимій зоні (p2..p98), кліпиться по контуру
    relief_zone (тіло мінус rim/текстові смуги) з feather-переходом до плоских
    країв — вушко/обід/смуга напису лишаються рівними на base_top.
    Повертає (topo_mesh, bottom_engrave_mesh) або (None, None) → плоский fallback."""
    provider = _fetch_zone_heightfield_provider(
        request=request, zone=zone, global_center=global_center
    )
    if provider is None:
        return None, None
    if relief_zone is None or getattr(relief_zone, "is_empty", True):
        relief_zone = base_mask
    inverse_map = _keychain_topo_inverse_map(
        unwrap_params=unwrap_params,
        source_bounds=source_bounds,
        target_bounds=target_bounds,
        angle_deg=map_rotation_deg,
    )
    try:
        relief_boundary = relief_zone.boundary
    except Exception:
        relief_boundary = None

    norm_state: dict[str, float] = {}

    def _top_z(points_xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=float)
        heights = provider.get_heights_for_points(inverse_map(pts))
        # inside/відстань до межі relief_zone (shapely 2 vectorized, fallback — цикл)
        try:
            import shapely as _shp

            pt_objs = _shp.points(pts)
            inside = np.asarray(_shp.contains(relief_zone, pt_objs), dtype=bool)
            dist = (
                np.asarray(_shp.distance(pt_objs, relief_boundary), dtype=float)
                if relief_boundary is not None
                else np.full(len(pts), feather_m)
            )
        except Exception:
            from shapely.prepared import prep

            prepared = prep(relief_zone)
            inside = np.array([prepared.contains(Point(x, y)) for x, y in pts], dtype=bool)
            dist = np.array(
                [relief_boundary.distance(Point(x, y)) if relief_boundary is not None else feather_m for x, y in pts],
                dtype=float,
            )
        vis = heights[inside]
        if len(vis) < 10:
            print("[KEYCHAIN TOPO] Too few relief samples — flat top")
            return np.full(len(pts), float(base_top_m))
        h_lo = float(np.percentile(vis, 2.0))
        h_hi = float(np.percentile(vis, 98.0))
        norm_state["range_m"] = h_hi - h_lo
        if h_hi - h_lo < 0.5:
            # Рівнина: підсилювати шум DEM немає сенсу — лишаємо пласку базу
            print(f"[KEYCHAIN TOPO] Terrain range {h_hi - h_lo:.2f}m < 0.5m — flat top kept")
            return np.full(len(pts), float(base_top_m))
        z01 = np.clip((heights - h_lo) / max(h_hi - h_lo, 1e-9), 0.0, 1.0)
        feather = np.clip(dist / max(feather_m, 1e-9), 0.0, 1.0)
        relief = np.where(inside, z01 * float(relief_m) * feather, 0.0)
        return float(base_top_m) + relief

    # Щільність сітки: ребро ~0.5мм у model-mm → деталь рельєфу читається на жетоні
    edge_m = _model_mm_to_world_m(0.5, export_scale_factor)
    max_area_m2 = max(edge_m * edge_m * 0.5, 1e-12)
    bottom_z = 0.0
    bottom_mesh: Optional[trimesh.Trimesh] = None
    if (
        back_text_poly is not None
        and not getattr(back_text_poly, "is_empty", True)
        and engrave_m > 1e-9
        and engrave_m < base_top_m - 1e-9
    ):
        # Гравіювання звороту: нижній шар із літерами-вирізами (як у
        # _build_keychain_base_parts), топо-частина починається з engrave_m.
        try:
            bottom_mask = base_mask.difference(back_text_poly).buffer(0)
            bottom_mesh = build_flat_layer_mesh_from_mask(
                bottom_mask, bottom_z_m=0.0, thickness_m=float(engrave_m),
                color=LAYER_COLORS["base"], min_area_m2=1e-12,
            )
            if bottom_mesh is not None:
                bottom_z = float(engrave_m)
        except Exception as exc:
            print(f"[KEYCHAIN TOPO] back engrave split failed ({exc}); solid topo base")
            bottom_mesh = None
            bottom_z = 0.0

    topo_parts: list[trimesh.Trimesh] = []
    for poly in _iter_polygons(base_mask):
        part = _extrude_polygon_heightfield(
            poly, bottom_z_m=bottom_z, top_z_func=_top_z, max_area_m2=max_area_m2
        )
        if part is not None and part.faces is not None and len(part.faces) > 0:
            topo_parts.append(part)
    if not topo_parts:
        return None, None
    topo_mesh = topo_parts[0] if len(topo_parts) == 1 else trimesh.util.concatenate(topo_parts)
    if norm_state.get("range_m", 0.0) < 0.5:
        # Рельєф вироджений — чесніше повернути None і лишити стандартну плоску базу
        return None, None
    watertight = bool(getattr(topo_mesh, "is_watertight", False))
    print(
        f"[KEYCHAIN TOPO] Relief mesh: {len(topo_mesh.faces)} faces, "
        f"terrain range {norm_state.get('range_m', 0.0):.1f}m, watertight={watertight}"
    )
    return _with_color(topo_mesh, LAYER_COLORS["base"]), bottom_mesh


def build_flat_building_meshes(
    *,
    request: Any,
    scale_factor: float,
    gdf_buildings_local: Optional[GeoDataFrame],
    base_top_m: float,
    export_scale_factor: Optional[float] = None,
) -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    """Повертає (ordinary_building_meshes, landmark_meshes). Орієнтири йдуть
    окремим списком → у експорті стають частиною «Landmark» з бронзовим кольором."""
    if gdf_buildings_local is None or gdf_buildings_local.empty:
        return [], []
    if not getattr(request, "include_buildings", True):
        return [], []

    gdf_buildings_for_mesh = split_building_parts_from_parent_footprints(gdf_buildings_local)

    # MEMORY GUARD: для keychain/flat режиму обмежуємо кількість будівель.
    # На сервері з 3.8GB RAM боолеан-операції з 700+ будівель зачасту викликають OOM.
    # Тримаємо топ-N найбільших за площею — це візуально найважливіші об'єкти.
    is_keychain = bool(getattr(request, "keychain_mode", False))
    is_flat = bool(getattr(request, "flat_plate_mode", False))
    if (is_keychain or is_flat) and gdf_buildings_for_mesh is not None and not gdf_buildings_for_mesh.empty:
        max_buildings = 200 if is_keychain else 500
        if len(gdf_buildings_for_mesh) > max_buildings:
            try:
                # Сортуємо за площею (більші першими) і беремо топ-N
                areas = gdf_buildings_for_mesh.geometry.area
                sorted_idx = list(areas.sort_values(ascending=False).index[:max_buildings])
                # Визначні місця ЗАВЖДИ лишаємо (навіть малі) — щоб орієнтири не зникали при OOM-кепі
                if "landmark" in gdf_buildings_for_mesh.columns:
                    _lm_mask = gdf_buildings_for_mesh["landmark"].fillna("").astype(str).str.strip() != ""
                    _kept = set(sorted_idx)
                    sorted_idx = sorted_idx + [i for i in gdf_buildings_for_mesh.index[_lm_mask] if i not in _kept]
                original_count = len(gdf_buildings_for_mesh)
                gdf_buildings_for_mesh = gdf_buildings_for_mesh.loc[sorted_idx].copy()
                print(
                    f"[MEMORY-GUARD] Reduced buildings from {original_count} to "
                    f"{len(gdf_buildings_for_mesh)} (top-{max_buildings} by area, "
                    f"{'keychain' if is_keychain else 'flat'} mode)"
                )
            except Exception as e:
                print(f"[MEMORY-GUARD] Building filter failed: {e}; proceeding with full set")

    # ПРИБИРАЄМО будівлі-волосини (ТІЛЬКИ keychain — на дрібному масштабі край
    # вікна карти ріже будинки в нитки 0.06–0.5мм, які не друкуються/відламуються;
    # емпірично: 72 компоненти <0.4мм у 31×40мм брелку). Повна мапа НЕ зачіпається
    # (більший масштаб + golden), тож golden лишається без змін.
    if is_keychain and gdf_buildings_for_mesh is not None and not gdf_buildings_for_mesh.empty:
        _before_thin = len(gdf_buildings_for_mesh)
        # ВАЖЛИВО: footprint у body-layout-просторі, що масштабується export_scale_factor
        # у фінальні мм (як магніти/база) — НЕ scale_factor (то детальний масштаб мапи).
        # erode 0.25мм/бік → викидаємо будівлі тонші за 0.5мм у ФІНАЛЬНОМУ друці.
        gdf_buildings_for_mesh = _drop_thin_footprints_gdf(
            gdf_buildings_for_mesh,
            erode_m=model_mm_to_world_m(0.25, float(export_scale_factor or scale_factor)),
        )
        _after_thin = len(gdf_buildings_for_mesh)
        if _after_thin < _before_thin:
            print(f"[KEYCHAIN] Dropped {_before_thin - _after_thin} thin building slivers (<0.5mm wide) — anti break-off")

    height_scale_factor = float(
        getattr(request, "buildings_height_scale", None)
        or getattr(request, "building_height_multiplier", 1.0)
    )
    requested_min_height_m = float(getattr(request, "building_min_height", 2.0) or 2.0)
    # Друкарський floor 0.8мм був зависокий → разом з log-шкалою «з'їдав» низ діапазону.
    # 0.3мм дає більше варіації знизу (1-2 поверхи), лишаючись друкованим на пласкій основі.
    printable_min_height_m = _model_mm_to_world_m(0.3, float(scale_factor))
    min_building_height_m = max(requested_min_height_m, printable_min_height_m)
    max_building_height_mm = float(getattr(request, "flat_max_building_height_mm", 0.0) or 0.0)
    if bool(getattr(request, "keychain_mode", False)):
        max_building_height_mm = max_building_height_mm or 2.4
        min_building_height_m = _model_mm_to_world_m(0.65, float(export_scale_factor or scale_factor))
    max_building_height_m = (
        _model_mm_to_world_m(max_building_height_mm, float(export_scale_factor or scale_factor))
        if max_building_height_mm > 0
        else 0.0
    )
    uniform_height = bool(getattr(request, "flat_uniform_building_height", False))
    uniform_height_m = max_building_height_m or _model_mm_to_world_m(2.0, float(export_scale_factor or scale_factor))

    records = process_buildings(
        gdf_buildings_for_mesh,
        terrain_provider=None,
        height_multiplier=height_scale_factor,
        min_height=min_building_height_m,
        foundation_depth=0.0,
        embed_depth=0.0,
        coordinates_already_local=True,
        return_records=True,
        min_feature_m=model_mm_to_world_m(MICRO_REGION_THRESHOLD_MM, scale_factor),
        scale_factor=scale_factor,
    )

    meshes: list[trimesh.Trimesh] = []
    landmark_meshes: list[trimesh.Trimesh] = []
    is_keychain = bool(getattr(request, "keychain_mode", False))
    for record in records:
        mesh = getattr(record, "mesh", None)
        if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
            continue
        landmark_category = getattr(record, "landmark", "") or ""
        mesh = mesh.copy()
        if uniform_height:
            mesh = _set_mesh_height(mesh, target_height_m=max(uniform_height_m, min_building_height_m))
        elif is_keychain and max_building_height_m > 0:
            # SPRINT 4 v2: РЕАЛІСТИЧНІ висоти через ЛОГ-шкалу.
            # OSM реальна висота → log-mapping у [min..max]mm щоб і 1-поверхові
            # і хмарочоси виглядали різними і пропорційними.
            # Real 3m (1 floor)  → ~0.65mm
            # Real 15m (5 floors) → ~1.5mm
            # Real 50m (15 floors) → ~2.5mm
            # Real 150m (45 floors) → ~3.5mm
            try:
                bz = float(mesh.bounds[0][2])
                tz = float(mesh.bounds[1][2])
                osm_height_m = max(tz - bz, 0.1)
                # Якщо OSM не дав height/levels — оцінка за площею
                footprint_area_m2 = float(getattr(mesh, "area_faces", None) or 100.0)
                if osm_height_m < 4.0:
                    # Heuristic: невеликі ~6m, великі ~20m, вежі-в-плані ~40m
                    osm_height_m = max(6.0, min(60.0, footprint_area_m2 ** 0.42 * 1.5))
                # ЛОГ-mapping у print mm: log2(height/3) * 0.5 + 0.65
                # log2(3/3)=0 → 0.65mm; log2(12/3)=2 → 1.65mm; log2(48/3)=4 → 2.65mm
                import math
                target_mm = 0.65 + math.log2(max(osm_height_m / 3.0, 1.0)) * 0.5
                target_mm = max(0.65, min(target_mm, max(max_building_height_mm or 4.0, 4.0)))
                target_height_m = _model_mm_to_world_m(target_mm, float(export_scale_factor or scale_factor))
                target_height_m = max(target_height_m, min_building_height_m)
                mesh = _set_mesh_height(mesh, target_height_m=target_height_m)
            except Exception:
                mesh = _clamp_mesh_height(mesh, min_height_m=min_building_height_m, max_height_m=max_building_height_m)
        elif max_building_height_m > 0:
            # ПЛАСКА МАПА: ПРОПОРЦІЙНІ висоти (log-шкала) замість clamp-усіх-до-max.
            # Раніше clamp робив усі будинки вищі за cap ОДНАКОВИМИ → користувач бачив
            # «неправильні/однакові висоти». Тепер 3-поверховий НИЖЧИЙ за 15-поверховий,
            # усе капається на max (друкарність). Та сама логіка, що для брелка, але з
            # пласким cap (max_building_height_mm, напр. 1.5мм).
            try:
                bz = float(mesh.bounds[0][2]); tz = float(mesh.bounds[1][2])
                osm_height_m = max(tz - bz, 0.1)
                footprint_area_m2 = float(getattr(mesh, "area_faces", None) or 100.0)
                if osm_height_m < 4.0:
                    osm_height_m = max(6.0, min(60.0, footprint_area_m2 ** 0.42 * 1.5))
                import math
                _cap_mm = max(float(max_building_height_mm or 1.5), 1.0)
                target_mm = 0.6 + math.log2(max(osm_height_m / 3.0, 1.0)) * 0.42
                target_mm = max(0.6, min(target_mm, _cap_mm))
                target_height_m = _model_mm_to_world_m(target_mm, float(export_scale_factor or scale_factor))
                target_height_m = max(target_height_m, min_building_height_m)
                mesh = _set_mesh_height(mesh, target_height_m=target_height_m)
            except Exception:
                mesh = _clamp_mesh_height(mesh, min_height_m=min_building_height_m, max_height_m=max_building_height_m)
        mesh.apply_translation([0.0, 0.0, float(base_top_m)])
        if landmark_category:
            _with_color(mesh, LAYER_COLORS["landmark"])
            landmark_meshes.append(mesh)
        else:
            _with_color(mesh, LAYER_COLORS["buildings"])
            meshes.append(mesh)

    # ФІНАЛЬНИЙ mesh-level фільтр волосин (ТІЛЬКИ keychain): process_buildings
    # внутрішньо створює slivers (parent − parts), яких footprint-фільтр вище не
    # бачить. Тут міряємо РЕАЛЬНУ мін-ширину готового меша і викидаємо <0.5мм
    # (фінал) — не друкується/відламується. Повна мапа не зачіпається → golden ОК.
    if is_keychain:
        _minw_m = model_mm_to_world_m(0.5, float(export_scale_factor or scale_factor))
        _before = len(meshes) + len(landmark_meshes)
        meshes = [m for m in meshes if _mesh_footprint_min_width_m(m) >= _minw_m]
        # Орієнтири на БРЕЛКУ теж не можуть бути тоншими за 0.5мм (відламаються при друці)
        landmark_meshes = [m for m in landmark_meshes if _mesh_footprint_min_width_m(m) >= _minw_m]
        _after = len(meshes) + len(landmark_meshes)
        if _after < _before:
            print(f"[KEYCHAIN] Dropped {_before - _after} thin building meshes (<0.5mm min-width) — anti break-off")
    return meshes, landmark_meshes


def run_flat_plate_pipeline(
    *,
    task: Any,
    request: Any,
    task_id: str,
    output_dir: Path,
    zone: Any,
    source: Any,
    canonical_2d_stage: Any,
    global_center: Any,
    file_basename: Optional[str] = None,
) -> ExportPipelineResult:
    if not (zone.scale_factor and float(zone.scale_factor) > 0):
        raise ValueError("flat_plate_mode requires a valid scale_factor")

    task.update_status("processing", 55, "Генерую пласкі шари (вода/дороги/будівлі)...")
    scale_factor = float(zone.scale_factor)
    export_scale_factor = scale_factor
    try:
        ref_xy = getattr(zone, "reference_xy_m", None)
        if ref_xy:
            ref = max(float(ref_xy[0]), float(ref_xy[1]))
            if ref > 1e-6:
                export_scale_factor = float(getattr(request, "model_size_mm", 80.0)) / ref
    except Exception:
        export_scale_factor = scale_factor
    base_thickness_mm = max(float(getattr(request, "terrain_base_thickness_mm", 0.8) or 0.8), 0.2)
    water_layer_mm = max(float(getattr(request, "flat_water_layer_mm", 0.22) or 0.22), 0.0)
    roads_layer_mm = max(float(getattr(request, "flat_roads_layer_mm", 0.42) or 0.42), 0.0)
    parks_layer_mm = max(float(getattr(request, "flat_parks_layer_mm", 0.36) or 0.36), 0.0)
    keychain_mode = bool(getattr(request, "keychain_mode", False))
    # C3 ТОПО-БРЕЛОК: рельєф висот замість карти (дороги/вода/парки/будівлі off)
    topo_mode = keychain_mode and bool(getattr(request, "keychain_topo_mode", False))
    if keychain_mode:
        # DEFAULT база жетона = 1.5мм (міцна основа під рельєф). Юзер:
        # «по дефолту довжину основи зроби 1.5мм». Поважаємо більше значення,
        # якщо користувач явно задав товщу базу.
        base_thickness_mm = max(base_thickness_mm, 1.5)
        # Збільшені товщини для видимого рельєфу на однокольоровому пластику.
        # Поступове наростання z для природної ієрархії:
        # water 0.45mm → parks 0.65mm → roads 0.75mm → buildings 1.5mm+
        water_layer_mm = max(water_layer_mm, 0.45)
        parks_layer_mm = max(parks_layer_mm, 0.65)
        roads_layer_mm = max(roads_layer_mm, 0.75)

    # МАГНІТ: кишеня під шайбу мусить РЕАЛЬНО вирізатись. Якщо база тонша за
    # (глибина кишені + 0.8мм стінка) — кишеня раніше тихо клампилась до 0 і
    # магніт друкувався БЕЗ заглиблення (a no-op). Авто-потовщуємо базу, щоб
    # магніт ЗАВЖДИ отримав робочу кишеню незалежно від того, що шле клієнт.
    if bool(getattr(request, "magnet_pocket", False)) and not keychain_mode:
        _mag_depth_mm = float(getattr(request, "magnet_pocket_depth_mm", 2.0) or 2.0)
        base_thickness_mm = max(base_thickness_mm, _mag_depth_mm + 0.8)

    # З'ЄДНУВАЧ-ПАЗИ: паз ріжеться у ДНО — лишаємо ≥1мм суцільного лиця над ним,
    # тож основа мусить бути ≥ (глибина пазу + 1мм) і ≥3мм (узгоджено з flat-AMS).
    map_connector = bool(getattr(request, "map_connector", False)) and not keychain_mode
    if map_connector:
        _conn_depth_mm = float(getattr(request, "map_connector_depth_mm", 0.3) or 0.3)
        base_thickness_mm = max(base_thickness_mm, _conn_depth_mm + 1.0, 3.0)

    # ВИДІЛЕНА БУДІВЛЯ (карта): паз 0.8мм у ВЕРХ бази + лице ≥0.6мм → база ≥1.6мм.
    map_highlight_building = bool(getattr(request, "map_highlight_building", False)) and not keychain_mode
    if map_highlight_building:
        base_thickness_mm = max(base_thickness_mm, 1.6)

    # THROUGH-HOLE GUARD (EARLY, worst-case): коли ПІДСВІТКА (паз 0.8мм у ВЕРХ)
    # поєднується з МАГНІТ-кишенею або КОНЕКТОР-пазом (виріз у ДНО) і їхні footprint
    # перетинаються у XY — між ними лишилось би лише (base − 0.8 − bottom_depth)
    # суцільного матеріалу. Потовщуємо базу ТУТ (до побудови шарів зверху, тож
    # будинки/текст/peg сідають на правильну висоту), щоб ЗАВЖДИ було ≥0.4мм навіть
    # при повному перекритті. Глибину магніт/конектор-пазу НЕ зменшуємо (шайба/ключ
    # сідають як треба). Спрацьовує лише у комбо-кейсі (highlight=opt-in → golden ОК).
    _HL_TOP_POCKET_MM = 0.8  # = build_highlight_insert depth_mm default
    _HL_MIN_SOLID_MM = 0.4
    if map_highlight_building and (
        bool(getattr(request, "magnet_pocket", False))
        or bool(getattr(request, "map_connector", False))
    ):
        _bottom_depth_mm = 0.0
        if bool(getattr(request, "magnet_pocket", False)):
            _bottom_depth_mm = max(_bottom_depth_mm, float(getattr(request, "magnet_pocket_depth_mm", 2.0) or 2.0))
        if bool(getattr(request, "map_connector", False)):
            _bottom_depth_mm = max(_bottom_depth_mm, float(getattr(request, "map_connector_depth_mm", 0.3) or 0.3))
        _needed_mm = _HL_TOP_POCKET_MM + _bottom_depth_mm + _HL_MIN_SOLID_MM
        if base_thickness_mm < _needed_mm - 1e-9:
            print(f"[MAP HIGHLIGHT] through-hole guard (early): highlight 0.8mm top pocket + "
                  f"{_bottom_depth_mm:.2f}mm bottom cut → base {base_thickness_mm:.2f}→{_needed_mm:.2f}mm "
                  f"(keeps ≥{_HL_MIN_SOLID_MM}mm solid if footprints overlap)")
            base_thickness_mm = _needed_mm

    base_top_m = _model_mm_to_world_m(base_thickness_mm, export_scale_factor)
    # Нижній виріз бази (магніт-кишеня / конектор-пази) — запам'ятовуємо, щоб при
    # додаванні ВЕРХНЬОГО пазу під виділену будівлю ПЕРЕБУДувати базу не втративши його.
    _flat_base_bottom_poly: Optional[BaseGeometry] = None
    _flat_base_bottom_depth_m: float = 0.0
    content_area = zone.zone_polygon_local
    keychain_layout: Optional[dict[str, BaseGeometry]] = None
    keychain_rim_mesh: Optional[trimesh.Trimesh] = None
    keychain_text_mesh: Optional[trimesh.Trimesh] = None
    keychain_text2_mesh: Optional[trimesh.Trimesh] = None
    keychain_base_bottom_mesh: Optional[trimesh.Trimesh] = None
    connector_mesh: Optional[trimesh.Trimesh] = None
    map_frame_mesh: Optional[trimesh.Trimesh] = None
    map_frame_hull: Optional[BaseGeometry] = None
    map_text_mesh: Optional[trimesh.Trimesh] = None
    keychain_back_poly: Optional[BaseGeometry] = None
    keychain_back_engrave_m: float = 0.0
    source_bounds: Optional[tuple[float, float, float, float]] = None
    target_bounds: Optional[tuple[float, float, float, float]] = None
    # Захоплений unwrap-transform брелка (закриття _xform, визначене нижче, коли є
    # source/target bounds): потрібен, щоб highlight-точку (lon/lat→local) перевести
    # у ТУ Ж unwrapped-систему, де лежать building_meshes (вони вже пройшли _xform).
    _keychain_xform = None
    if keychain_mode:
        keychain_layout = build_keychain_layout(
            bbox_meters=zone.bbox_meters,
            scale_factor=scale_factor,
            model_size_mm=float(getattr(request, "model_size_mm", 80.0) or 80.0),
            body_width_mm=float(getattr(request, "keychain_body_width_mm", 0.0) or 0.0) or None,
            body_height_mm=float(getattr(request, "keychain_body_height_mm", 0.0) or 0.0) or None,
            map_x_mm=float(getattr(request, "keychain_map_x_mm", 0.0) or 0.0),
            map_y_mm=float(getattr(request, "keychain_map_y_mm", 0.0) or 0.0),
            map_width_mm=float(getattr(request, "keychain_map_width_mm", 0.0) or 0.0) or None,
            map_height_mm=float(getattr(request, "keychain_map_height_mm", 0.0) or 0.0) or None,
            base_shape=str(getattr(request, "keychain_base_shape", "rounded") or "rounded"),
            loop_style=str(getattr(request, "keychain_loop_style", "round") or "round"),
            loop_angle_deg=float(getattr(request, "keychain_loop_angle_deg", 0.0) or 0.0),
            loop_center_x_mm=float(getattr(request, "keychain_loop_center_x_mm", 0.0) or 0.0) or None,
            loop_center_y_mm=float(getattr(request, "keychain_loop_center_y_mm", 0.0) or 0.0) or None,
            label_center_x_mm=float(getattr(request, "keychain_label_center_x_mm", 0.0) or 0.0) or None,
            label_center_y_mm=float(getattr(request, "keychain_label_center_y_mm", 0.0) or 0.0) or None,
            label_width_mm=float(getattr(request, "keychain_label_width_mm", 0.0) or 0.0) or None,
            label_angle_deg=float(getattr(request, "keychain_label_angle_deg", 0.0) or 0.0),
            loop_outer_radius_mm=float(getattr(request, "keychain_loop_outer_radius_mm", 6.5) or 6.5),
            loop_inner_radius_mm=float(getattr(request, "keychain_loop_inner_radius_mm", 3.0) or 3.0),
            corner_radius_mm=float(getattr(request, "keychain_corner_radius_mm", 4.0) or 4.0),
            label_band_height_mm=float(getattr(request, "keychain_label_band_height_mm", 9.0) or 9.0),
        )
        content_area = keychain_layout["content_area"]
        export_scale_factor = 1.0 / max(float(keychain_layout["layout_scale_m_per_mm"]), 1e-9)
        try:
            request.model_size_mm = max(float(keychain_layout.get("export_size_mm") or 0.0), 1.0)
        except Exception:
            pass
        base_top_m = _model_mm_to_world_m(base_thickness_mm, export_scale_factor)
        source_bounds = tuple(float(v) for v in keychain_layout["source_bbox"].bounds)
        target_bounds = tuple(float(v) for v in keychain_layout["map_target_bounds"])
        rim_width_m = _model_mm_to_world_m(float(getattr(request, "keychain_rim_width_mm", 0.0) or 0.0), export_scale_factor)
        if rim_width_m > 0:
            try:
                inner_base = keychain_layout["base"].buffer(-rim_width_m, join_style=1)
                # Карта обмежена ВСІМ inner_base (body мінус rim), а не slot rect
                clipped_content = inner_base
                if clipped_content is not None and not clipped_content.is_empty:
                    content_area = clipped_content.buffer(0)
                    keychain_layout["content_area"] = content_area
            except Exception:
                pass

        # НАПИС НА ЗВОРОТІ: рахуємо дзеркальний полігон літер ДО побудови бази —
        # база розщепиться на нижній шар з гравіюванням + верхній суцільний.
        _back_text = str(getattr(request, "keychain_back_label", "") or "").strip()
        if _back_text:
            try:
                keychain_back_engrave_m = _model_mm_to_world_m(
                    min(
                        max(float(getattr(request, "keychain_back_engrave_mm", 0.5) or 0.5), 0.2),
                        max(base_thickness_mm - 0.8, 0.2),
                    ),
                    export_scale_factor,
                )
                _base_b = keychain_layout["base"].bounds
                _bw = _base_b[2] - _base_b[0]
                _bh = _base_b[3] - _base_b[1]
                _cx = (_base_b[0] + _base_b[2]) / 2.0
                _cy = (_base_b[1] + _base_b[3]) / 2.0
                _band_h = min(_model_mm_to_world_m(10.0, export_scale_factor), _bh * 0.5)
                from shapely.geometry import box as _shp_box
                _back_band = _shp_box(
                    _base_b[0] + _bw * 0.08, _cy - _band_h / 2.0,
                    _base_b[2] - _bw * 0.08, _cy + _band_h / 2.0,
                )
                _poly = _compute_text_letter_polygon(
                    text=_back_text,
                    body_geometry=keychain_layout["base"],
                    label_band_geometry=_back_band,
                    text_height_m=_model_mm_to_world_m(
                        float(getattr(request, "keychain_back_text_height_mm", 5.0) or 5.0),
                        export_scale_factor,
                    ),
                    angle_deg=0.0,
                    min_stroke_m=_model_mm_to_world_m(MIN_KEYCHAIN_TEXT_STROKE_MM, export_scale_factor),
                    max_width=(_back_band.bounds[2] - _back_band.bounds[0]) * 0.96,
                    max_height=(_back_band.bounds[3] - _back_band.bounds[1]) * 0.92,
                )
                if _poly is not None and not getattr(_poly, "is_empty", True):
                    # Дзеркало по X: текст читається, коли брелок перевернуто.
                    _mirrored = affinity.scale(_poly, xfact=-1.0, yfact=1.0, origin=(_cx, _cy))
                    _safe_zone = keychain_layout["base"].buffer(
                        -_model_mm_to_world_m(1.0, export_scale_factor), join_style=1,
                    )
                    keychain_back_poly = _mirrored.intersection(_safe_zone).buffer(0)
                    print(f"[KEYCHAIN] Back engrave ready: '{_back_text}'")
            except Exception as exc:
                print(f"[KEYCHAIN] back label failed (non-fatal): {exc}")
                keychain_back_poly = None

    preclip_result = getattr(canonical_2d_stage, "preclip_result", None)
    gdf_buildings_local = getattr(preclip_result, "gdf_buildings_local", None)
    if gdf_buildings_local is None:
        try:
            from services.building_geometry_pipeline import prepare_building_geometry
            from services.geometry_preclip_pipeline import prepare_preclipped_geometry

            building_geometry = prepare_building_geometry(
                gdf_buildings=source.gdf_buildings,
                global_center=global_center,
                zone_prefix="[FLAT PLATE] ",
            )
            preclip_result = prepare_preclipped_geometry(
                gdf_buildings_local=building_geometry.gdf_buildings_local,
                building_geometries_for_flatten=building_geometry.building_geometries_for_flatten,
                gdf_water=source.gdf_water,
                global_center=global_center,
                zone_polygon_local=zone.zone_polygon_local,
                zone_prefix="[FLAT PLATE] ",
            )
            gdf_buildings_local = getattr(preclip_result, "gdf_buildings_local", None)
        except Exception as exc:
            print(f"[FLAT PLATE] Building preclip fallback failed: {exc}")
            gdf_buildings_local = None

    if keychain_layout is not None:
        terrain_mesh, keychain_base_bottom_mesh = _build_keychain_base_parts(
            keychain_layout["base"],
            base_top_m=base_top_m,
            back_text_poly=keychain_back_poly,
            engrave_m=keychain_back_engrave_m,
        )
    elif bool(getattr(request, "magnet_pocket", False)):
        # МАПА-МАГНІТ: кругла кишеня під магніт у центрі дна. Той самий прийом,
        # що й гравіювання звороту брелка — два watertight-екструди без булевих.
        try:
            _pocket_d_mm = float(getattr(request, "magnet_pocket_diameter_mm", 10.4) or 10.4)
            _pocket_depth_mm = float(getattr(request, "magnet_pocket_depth_mm", 2.0) or 2.0)
            _pocket_count = int(getattr(request, "magnet_pocket_count", 1) or 1)
            _pocket_inset_mm = float(getattr(request, "magnet_pocket_inset_mm", 8.0) or 8.0)
            # Гарантуємо ≥0.8мм стінку над кишенею
            _pocket_depth_mm = min(_pocket_depth_mm, max(base_thickness_mm - 0.8, 0.0))
            _pocket_depth_m = _model_mm_to_world_m(_pocket_depth_mm, export_scale_factor)
            _pocket = build_magnet_pocket_geometry(
                zone.zone_polygon_local,
                diameter_mm=_pocket_d_mm,
                count=_pocket_count,
                inset_mm=_pocket_inset_mm,
                export_scale_factor=export_scale_factor,
            )
            if _pocket is not None and _pocket_depth_m > 1e-9:
                _flat_base_bottom_poly, _flat_base_bottom_depth_m = _pocket, _pocket_depth_m
            terrain_mesh, keychain_base_bottom_mesh = _build_keychain_base_parts(
                zone.zone_polygon_local,
                base_top_m=base_top_m,
                back_text_poly=_pocket if (_pocket is not None and _pocket_depth_m > 1e-9) else None,
                engrave_m=_pocket_depth_m,
            )
            if keychain_base_bottom_mesh is not None:
                _n = len(getattr(_pocket, "geoms", [None])) if _pocket is not None else 0
                print(f"[MAGNET] {max(_n,1)} pocket(s) Ø{_pocket_d_mm}×{_pocket_depth_mm}mm carved into base bottom")
            else:
                print("[MAGNET] Base too thin for pocket — solid base fallback")
        except Exception as exc:
            print(f"[MAGNET] pocket failed (non-fatal): {exc}")
            terrain_mesh = build_flat_zone_base_mesh(
                zone.zone_polygon_local,
                bbox_meters=zone.bbox_meters,
                thickness_m=base_top_m,
            )
    elif map_connector:
        # З'ЄДНУВАЧ-ПАЗИ: ластівчин-хвіст у ДНІ основи + окремий ключ-метелик.
        # Той самий прийом, що магніт/зворот-гравіювання — паз ріжеться у нижній
        # шар (back_text_poly), лице лишається суцільним → шов спереду непомітний.
        try:
            _conn_depth_mm = float(getattr(request, "map_connector_depth_mm", 0.3) or 0.3)
            _conn_depth_mm = min(_conn_depth_mm, max(base_thickness_mm - 1.0, 0.0))  # ≥1мм лиця
            _conn_depth_m = _model_mm_to_world_m(_conn_depth_mm, export_scale_factor)
            _notches, _keys = build_map_connector_geometry(
                zone.zone_polygon_local,
                edges=str(getattr(request, "map_connector_edges", "NSEW") or "NSEW"),
                span_mm=float(getattr(request, "map_connector_span_mm", 10.0) or 10.0),
                length_mm=float(getattr(request, "map_connector_length_mm", 15.0) or 15.0),
                waist_frac=0.5,
                clearance_mm=float(getattr(request, "map_connector_clearance_mm", 0.03) or 0.03),
                export_scale_factor=export_scale_factor,
                key_edges=(str(getattr(request, "map_connector_key_edges", "") or "") or None),
            )
            if _notches is not None and _conn_depth_m > 1e-9:
                _flat_base_bottom_poly, _flat_base_bottom_depth_m = _notches, _conn_depth_m
            terrain_mesh, keychain_base_bottom_mesh = _build_keychain_base_parts(
                zone.zone_polygon_local,
                base_top_m=base_top_m,
                back_text_poly=_notches if (_notches is not None and _conn_depth_m > 1e-9) else None,
                engrave_m=_conn_depth_m,
            )
            # Ключ-метелик окремою деталлю «Connector» (товщина = глибина пазу,
            # тож вкладається у спільну порожнину двох плиток урівень з дном).
            if _keys is not None and not getattr(_keys, "is_empty", True) and _conn_depth_m > 1e-9:
                connector_mesh = build_flat_layer_mesh_from_mask(
                    _keys, bottom_z_m=0.0, thickness_m=_conn_depth_m,
                    color=LAYER_COLORS["base"], min_area_m2=1e-9,
                )
            if _notches is not None and keychain_base_bottom_mesh is not None:
                _ne = len(getattr(_notches, "geoms", [None]))
                print(f"[CONNECTOR] {max(_ne, 1)} butterfly notch(es) carved into base bottom "
                      f"(depth {_conn_depth_mm:.2f}mm, face {base_thickness_mm - _conn_depth_mm:.2f}mm); "
                      f"key part={'OK' if connector_mesh is not None else 'None'}")
            else:
                print("[CONNECTOR] No valid notches (base too thin / edges empty) — solid base")
        except Exception as exc:
            print(f"[CONNECTOR] connector failed (non-fatal): {exc}")
            connector_mesh = None
            terrain_mesh = build_flat_zone_base_mesh(
                zone.zone_polygon_local,
                bbox_meters=zone.bbox_meters,
                thickness_m=base_top_m,
            )
    else:
        terrain_mesh = build_flat_zone_base_mesh(
            zone.zone_polygon_local,
            bbox_meters=zone.bbox_meters,
            thickness_m=base_top_m,
        )
    if keychain_layout is not None and terrain_mesh is not None:
        rim_width_mm = float(getattr(request, "keychain_rim_width_mm", 0.0) or 0.0)
        rim_height_mm = float(getattr(request, "keychain_rim_height_mm", 0.0) or 0.0)
        # Друкований мінімум ободка: якщо увімкнено (>0) — ширина ≥0.8мм (одинарний
        # периметр тонший делямінується), висота ≥0.4мм (2 шари 0.2). 0 = вимкнено.
        # Дефолт 1.2×0.45 не зачіпається; клампимо лише тонкі значення зі слайдера.
        if rim_width_mm > 0:
            rim_width_mm = max(rim_width_mm, 0.8)
            rim_height_mm = max(rim_height_mm, 0.4)
        keychain_rim_mesh = build_keychain_rim_mesh(
            base_geometry=keychain_layout["base"],
            bottom_z_m=base_top_m,
            width_m=_model_mm_to_world_m(rim_width_mm, export_scale_factor),
            height_m=_model_mm_to_world_m(rim_height_mm, export_scale_factor),
        )

    bundle = canonical_2d_stage.canonical_mask_bundle
    min_feature_m = _model_mm_to_world_m(
        MIN_KEYCHAIN_PRINT_FEATURE_MM if keychain_mode else 0.2,
        export_scale_factor if keychain_mode else scale_factor,
    )
    min_area_m2 = max((min_feature_m ** 2) * 0.5, 1e-9)

    if keychain_layout and source_bounds and target_bounds:
        map_rotation_deg = float(getattr(request, "keychain_map_rotation_deg", 0.0) or 0.0)

        # KEYCHAIN UNWRAP: повернута рамка (zone_polygon_local) — CCW поворот rect
        # на map_rotation_deg. Щоб упакувати її ВМІСТ у axis-aligned слот, треба:
        # 1) центрувати геометрію навколо center повернутої рамки
        # 2) обернути на -map_rotation_deg (CW) — рамка стає axis-aligned
        # 3) скейлити її РЕАЛЬНІ розміри (rect_w × rect_h) у target_bounds
        # 4) перенести в центр target_bounds
        zone_poly = zone.zone_polygon_local
        unwrap_params: Optional[dict] = None
        if zone_poly is not None and not getattr(zone_poly, "is_empty", True):
            try:
                # Координати з orient (CCW) — беремо першу зовнішню кільцю
                exterior = list(zone_poly.exterior.coords)
                # Прибираємо закриваючу точку
                if len(exterior) > 1 and exterior[0] == exterior[-1]:
                    exterior = exterior[:-1]
                cx_zone = sum(p[0] for p in exterior) / len(exterior)
                cy_zone = sum(p[1] for p in exterior) / len(exterior)
                # Реальні розміри rect: rotate exterior by -angle around center → axis-aligned bbox
                from shapely.geometry import Polygon as _P
                unrot = affinity.rotate(_P(exterior), -map_rotation_deg, origin=(cx_zone, cy_zone), use_radians=False)
                ub = unrot.bounds  # (minx, miny, maxx, maxy) axis-aligned
                rect_w = float(ub[2] - ub[0])
                rect_h = float(ub[3] - ub[1])
                if rect_w > 1e-6 and rect_h > 1e-6:
                    tgt_w = float(target_bounds[2] - target_bounds[0])
                    tgt_h = float(target_bounds[3] - target_bounds[1])
                    tgt_cx = (target_bounds[0] + target_bounds[2]) / 2.0
                    tgt_cy = (target_bounds[1] + target_bounds[3]) / 2.0
                    unwrap_params = {
                        "cx_src": cx_zone, "cy_src": cy_zone,
                        "rect_w": rect_w, "rect_h": rect_h,
                        "tgt_cx": tgt_cx, "tgt_cy": tgt_cy,
                        "tgt_w": tgt_w, "tgt_h": tgt_h,
                        "angle": map_rotation_deg,
                    }
                    print(
                        f"[KEYCHAIN UNWRAP] center=({cx_zone:.2f},{cy_zone:.2f}) "
                        f"rect={rect_w:.2f}x{rect_h:.2f}m angle={map_rotation_deg:.1f}° "
                        f"target={tgt_w:.4f}x{tgt_h:.4f} center=({tgt_cx:.4f},{tgt_cy:.4f})"
                    )
            except Exception as exc:
                print(f"[KEYCHAIN UNWRAP] failed: {exc}; falling back to legacy transform")
                unwrap_params = None

        def _xform(geometry: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
            if geometry is None or getattr(geometry, "is_empty", True):
                return geometry
            if unwrap_params is None:
                return _orient_then_stretch_geometry_into_bounds(
                    geometry,
                    source_bounds=source_bounds,
                    target_bounds=target_bounds,
                    angle_deg=map_rotation_deg,
                )
            # UNWRAP: 1) translate to origin 2) rotate -angle 3) UNIFORM scale 4) translate to target
            try:
                p = unwrap_params
                # 1. центр зони → (0,0)
                step = affinity.translate(geometry, xoff=-p["cx_src"], yoff=-p["cy_src"])
                # 2. rotate -angle (CW)
                step = affinity.rotate(step, -p["angle"], origin=(0, 0), use_radians=False)
                # 3. COVER (uniform scale + crop) — найпростіша поведінка.
                # MAX(sx, sy) → контент ЗАВЖДИ заповнює слот без спотворення.
                # Зайве по краях обрізається body shape (content_area).
                sx = p["tgt_w"] / p["rect_w"]
                sy = p["tgt_h"] / p["rect_h"]
                s = max(sx, sy)
                # Uniform cover scale — без спотворення, заповнює слот, обрізаєм краї
                step = affinity.scale(step, xfact=s, yfact=s, origin=(0, 0))
                # 4. translate to target center
                step = affinity.translate(step, xoff=p["tgt_cx"], yoff=p["tgt_cy"])
                return step.buffer(0)
            except Exception:
                return _orient_then_stretch_geometry_into_bounds(
                    geometry,
                    source_bounds=source_bounds,
                    target_bounds=target_bounds,
                    angle_deg=map_rotation_deg,
                )

        # Захоплюємо unwrap-transform для highlight-точок брелка (building_meshes
        # будуються з gdf, що пройшов саме _xform — точку треба перевести так само).
        _keychain_xform = _xform

        # Гарантуємо raw fallback'и: якщо canonical bundle порожній, беремо
        # сирі OSM дані (з preclip + road_geometry). Інакше для малих зон з
        # тонкими дорогами / нечисленними будівлями весь шар стає None.
        raw_water_source = None
        raw_road_source = None
        raw_buildings_source = None
        raw_parks_source = None
        try:
            gdf_water_local = getattr(preclip_result, "gdf_water_local", None)
            if gdf_water_local is not None and not gdf_water_local.empty:
                raw_water_source = _mask_union_from_geometries(gdf_water_local.geometry.values)
        except Exception:
            raw_water_source = None
        raw_road_source_unmerged = None
        try:
            road_geometry = getattr(canonical_2d_stage, "road_geometry", None)
            if road_geometry is not None:
                raw_road_source = (
                    getattr(road_geometry, "merged_roads_geom_local", None)
                    or getattr(road_geometry, "merged_roads_geom_local_raw", None)
                )
                # Truly UN-MERGED road polygons (ДО gap-fill / merge_close_road_gaps).
                # Keychain використовує саме їх, щоб дороги залишались окремими
                # стрічками як у live-превʼю (юзер прибрав логіку обʼєднання).
                raw_road_source_unmerged = getattr(road_geometry, "merged_roads_geom_local_raw", None)
        except Exception:
            raw_road_source = None
        try:
            if gdf_buildings_local is not None and not gdf_buildings_local.empty:
                raw_buildings_source = _mask_union_from_geometries(gdf_buildings_local.geometry.values)
        except Exception:
            raw_buildings_source = None
        # RAW PARKS FALLBACK: коли canonical parks_final порожній (буває для
        # дрібних зон де всі парки відфільтровуються), беремо raw gdf_green
        # з extras_loader і локалізуємо через global_center.
        try:
            gdf_green_src = getattr(source, "gdf_green", None)
            if gdf_green_src is not None and not gdf_green_src.empty:
                from shapely.ops import transform as _shp_transform
                def _to_local(g):
                    try:
                        return _shp_transform(lambda x, y, z=None: global_center.to_local(x, y), g)
                    except Exception:
                        return None
                local_geoms = [_to_local(g) for g in gdf_green_src.geometry.values if g is not None]
                local_geoms = [g for g in local_geoms if g is not None and not g.is_empty]
                if local_geoms:
                    raw_parks_source = _mask_union_from_geometries(local_geoms)
                    print(f"[KEYCHAIN] raw_parks_source from gdf_green: {len(local_geoms)} polygons")
        except Exception as exc:
            print(f"[KEYCHAIN] raw_parks_source failed: {exc}")
            raw_parks_source = None

        # Buildings: bundle або raw
        bundle_buildings = getattr(bundle, "buildings_footprints", None)
        if bundle_buildings is None or getattr(bundle_buildings, "is_empty", True):
            if raw_buildings_source is not None and not getattr(raw_buildings_source, "is_empty", True):
                print("[KEYCHAIN] Canonical buildings empty; using raw clipped buildings")
                bundle_buildings = raw_buildings_source
        building_mask = _clip_geometry(_xform(bundle_buildings), content_area)

        # Roads: для KEYCHAIN беремо НЕ-обʼєднану геометрію
        # (merged_roads_geom_local_raw — ДО merge_close_road_gaps), щоб дороги
        # залишались окремими стрічками точно як у live-превʼю. roads_final
        # містить gap-fill, який злипає сусідні вулиці в суцільні блоби —
        # саме цю «логіку обʼєднання» юзер просив прибрати.
        bundle_roads = None
        if keychain_mode and raw_road_source_unmerged is not None and not getattr(raw_road_source_unmerged, "is_empty", True):
            bundle_roads = raw_road_source_unmerged
            print("[KEYCHAIN] Using UN-MERGED roads (merged_roads_geom_local_raw) — no gap-fill merge, roads stay separate like preview")
        if bundle_roads is None or getattr(bundle_roads, "is_empty", True):
            bundle_roads = getattr(bundle, "roads_final", None)
        if bundle_roads is None or getattr(bundle_roads, "is_empty", True):
            if raw_road_source is not None and not getattr(raw_road_source, "is_empty", True):
                print("[KEYCHAIN] Canonical roads empty; using raw clipped roads")
                bundle_roads = raw_road_source
        road_mask = _clip_geometry(_xform(bundle_roads), content_area)

        # BRIDGES: Витягуємо edges з OSM тегом bridge=yes/viaduct/etc, формуємо
        # окрему bridge_mask. Мости рендеряться вище звичайних доріг — їх видно
        # як виступаючі сегменти, схоже на main-pipeline де мости визначались
        # через перепад висоти терейну. У keychain рельєфу немає, тож тег з OSM —
        # єдиний reliable spoob детектити мости.
        bridge_mask = None
        try:
            # ПРІОРИТЕТ: extract з G_roads (геометрія співпадає з road_mask 1:1)
            # FALLBACK: окремий _fetch_bridges якщо graph не має bridge column
            local_edges = None
            G_roads_obj = getattr(source, "G_roads", None)
            if G_roads_obj is not None:
                try:
                    import osmnx as _ox
                    candidate = _ox.graph_to_gdfs(G_roads_obj, nodes=False)
                    if "bridge" in candidate.columns:
                        # Bridges є у графі — використовуємо їх (геометрія = road_mask)
                        local_edges = candidate
                        print(f"[KEYCHAIN] Bridges from G_roads (1:1 with road_mask)")
                except Exception:
                    pass
            if local_edges is None:
                gdf_b = getattr(source.gdf_buildings, "attrs", {}).get("bridges") if source.gdf_buildings is not None else None
                if gdf_b is not None and not gdf_b.empty:
                    local_edges = gdf_b.reset_index(drop=True)
                    if "bridge" not in local_edges.columns:
                        local_edges["bridge"] = "yes"
                    print(f"[KEYCHAIN] Using dedicated bridge fetcher fallback: {len(local_edges)} ways")
            if local_edges is not None and not local_edges.empty and "bridge" in local_edges.columns:
                # bridge column має значення "yes", "viaduct", "movable", etc. Тільки no/NaN = не міст.
                bridge_rows = local_edges[local_edges["bridge"].notna() & (local_edges["bridge"].astype(str).str.lower() != "no")]
                if not bridge_rows.empty:
                    print(f"[KEYCHAIN] Bridges detected: {len(bridge_rows)}/{len(local_edges)} edges have bridge tag")
                    # Ширина моста залежить від типу дороги (з highway тегу)
                    BRIDGE_WIDTHS_M = {
                        "motorway": 9.0, "trunk": 8.0, "primary": 7.0, "secondary": 6.5, "tertiary": 6.0,
                        "residential": 4.5, "unclassified": 4.5, "service": 3.5, "pedestrian": 3.5,
                        "footway": 2.5, "path": 2.0, "cycleway": 2.5,
                    }
                    # Конвертуємо в локальні координати з ВЛАСНОЮ шириною
                    def _to_local_geom(g):
                        try:
                            xs, ys = g.coords.xy
                            pts = [global_center.to_local(x, y) for x, y in zip(xs, ys)]
                            from shapely.geometry import LineString
                            return LineString(pts) if len(pts) >= 2 else None
                        except Exception:
                            return None
                    has_highway_col = "highway" in bridge_rows.columns
                    bridge_polys = []
                    for idx, row in bridge_rows.iterrows():
                        g = row.geometry
                        if g is None or not hasattr(g, "coords"):
                            continue
                        line = _to_local_geom(g)
                        if line is None or line.is_empty:
                            continue
                        # Визначаємо ширину з тегу highway
                        hw = str(row.get("highway") if has_highway_col else "primary").lower() if has_highway_col else "primary"
                        # OSMnx іноді віддає список (для multi-tagged edges) — беремо перший
                        if hw.startswith("["):
                            hw = hw.strip("[]'\" ").split(",")[0].strip("'\" ")
                        # Half-width у метрах (буфер з обох сторін)
                        half_w = BRIDGE_WIDTHS_M.get(hw, 5.0) / 2.0
                        try:
                            bridge_polys.append(line.buffer(half_w, cap_style=2, join_style=2))
                        except Exception:
                            pass
                    if bridge_polys:
                        bridge_geom = unary_union(bridge_polys)
                        # Трансформуємо у content_area (keychain mm-space)
                        bridge_mask_raw = _clip_geometry(_xform(bridge_geom), content_area)
                        if bridge_mask_raw is not None and not getattr(bridge_mask_raw, "is_empty", True):
                            bridge_mask = bridge_mask_raw.buffer(0)
                            try:
                                area_mm = bridge_mask.area  # в model_mm² бо target_bounds у mm
                                print(f"[KEYCHAIN] Bridge polygons: {len(bridge_polys)}, mask area={area_mm:.2f}mm²")
                            except Exception:
                                pass
        except Exception as exc:
            print(f"[KEYCHAIN] Bridge detection failed (non-fatal): {exc}")
            bridge_mask = None

        # KEYCHAIN ROADS BOOST: При малому масштабі (>3 м/мм) тонкі вулиці
        # (3-5м, типові міські) дають у моделі менше 0.5mm — і випадають з
        # друку. Розширюємо road_mask на min_feature_m*0.6, щоб типові
        # вулиці гарантовано пройшли фільтр і були видимі. Магістралі при
        # цьому розширяться непомітно (вони вже широкі), а маленькі вулиці
        # «потовщаться» до мінімально друкованих.
        if keychain_mode and road_mask is not None and not getattr(road_mask, "is_empty", True):
            try:
                # Повернуто до 0.6 (юзер: "до цього все було ідеально")
                widen_m = float(min_feature_m) * 0.6
                if widen_m > 0:
                    widened = road_mask.buffer(widen_m, join_style=2, cap_style=2)
                    if widened is not None and not getattr(widened, "is_empty", True):
                        # Залишаємось у межах content_area
                        widened = widened.intersection(content_area).buffer(0)
                        if widened is not None and not getattr(widened, "is_empty", True):
                            road_mask = widened
                            print(
                                f"[KEYCHAIN] Roads widened by {widen_m*1000:.2f}mm "
                                f"to survive print filter (thin streets become visible)"
                            )
            except Exception as exc:
                print(f"[KEYCHAIN] Road widen failed: {exc}")

        # Parks: bundle або raw fallback (для дрібних зон де canonical фільтр зʼїдає все)
        parks_source = getattr(bundle, "parks_final", None)
        if parks_source is None or getattr(parks_source, "is_empty", True):
            if raw_parks_source is not None and not getattr(raw_parks_source, "is_empty", True):
                print("[KEYCHAIN] Canonical parks empty; using raw clipped parks as flat layer source")
                parks_source = raw_parks_source
        parks_mask = _clip_geometry(_xform(parks_source), content_area)

        water_source = getattr(bundle, "water_final", None)
        if water_source is None or getattr(water_source, "is_empty", True):
            if raw_water_source is not None and not getattr(raw_water_source, "is_empty", True):
                print("[KEYCHAIN] Canonical water was empty; using raw clipped water as flat layer source")
                water_source = raw_water_source
        water_mask = _clip_geometry(_xform(water_source), content_area)

        # ПРІОРИТЕТ: buildings STAY full, roads get clipped under buildings.
        # Юзер: «треба зробити щоб дорога обрізалась під будинками щоб вона
        # там не робилась». Тобто будівлі залишаються повними прямокутниками,
        # дороги обходять їх «вирізами».
        road_mask = _sanitize_layer_mask(
            road_mask,
            min_feature_m=min_feature_m,
            min_area_m2=min_area_m2,
            label="roads",
        )
        # Roads - building footprint → дорога не лізе під будинок.
        if building_mask is not None and not getattr(building_mask, "is_empty", True) and road_mask is not None and not getattr(road_mask, "is_empty", True):
            try:
                road_mask = _subtract_geometry(road_mask, building_mask)
                # Згладжуємо мікро-зубці після subtract: opening = erosion + dilation
                # на 0.3m → видаляє «гачки» між будинками без зміни форми.
                if road_mask is not None and not getattr(road_mask, "is_empty", True):
                    smooth_m = 0.3
                    road_mask = road_mask.buffer(-smooth_m).buffer(smooth_m).buffer(0)
                print(f"[KEYCHAIN] Roads clipped under buildings + smoothed (no zigzag artifacts)")
            except Exception:
                pass
        if (road_mask is None or getattr(road_mask, "is_empty", True)) and raw_road_source is not None and not getattr(raw_road_source, "is_empty", True):
            print("[KEYCHAIN] Roads collapsed after sanitize; retrying with raw road source + soft filter")
            road_mask = _sanitize_layer_mask(
                _subtract_geometry(_clip_geometry(_xform(raw_road_source), content_area), building_mask),
                min_feature_m=float(min_feature_m) * 0.5,  # м'якший фільтр
                min_area_m2=float(min_area_m2) * 0.4,
                label="raw roads",
            )
        if keychain_mode:
            try: task.update_status("processing", 70, "Збираю карту: дороги, мости, будівлі...")
            except Exception: pass
        # KEYCHAIN: МІНІМАЛЬНІ субтракції. Z-order забезпечує видимість шарів:
        # water=0.30mm < parks=0.50mm < roads=0.55mm < buildings=1.55mm.
        # Виглядає природньо: вода зверху видно, парк піднімається над водою,
        # дорога над парком, будівлі вище за все.
        water_mask = _sanitize_layer_mask(
            water_mask,  # БЕЗ subtract — інші шари над нею через z-order
            min_feature_m=min_feature_m,
            min_area_m2=min_area_m2,
            label="water",
        )
        if (water_mask is None or getattr(water_mask, "is_empty", True)) and raw_water_source is not None and not getattr(raw_water_source, "is_empty", True):
            print("[KEYCHAIN] Canonical water collapsed; retrying with raw")
            water_mask = _sanitize_layer_mask(
                _clip_geometry(_xform(raw_water_source), content_area),
                min_feature_m=min_feature_m,
                min_area_m2=min_area_m2,
                label="raw water",
            )
        parks_mask = _sanitize_layer_mask(
            parks_mask,  # БЕЗ subtract — мʼякий фільтр щоб маленькі парки лишались
            min_feature_m=float(min_feature_m) * 0.5,
            min_area_m2=float(min_area_m2) * 0.3,
            label="parks",
        )
        # ЗЕЛЕНЬ З-ПІД ДОРІГ: road z (0.55) > parks z (0.50), тож службові дороги/
        # алеї ВСЕРЕДИНІ зелених зон (кладовища мають щільну мережу алей у OSM)
        # перекривали зелень → «кладовище заповнене дорогою» (скарга юзера).
        # Віднімаємо парки/кладовища від доріг — зелена зона читається як зелена,
        # а дороги лишаються зверху лише ПОЗА зеленню.
        if (road_mask is not None and not getattr(road_mask, "is_empty", True)
                and parks_mask is not None and not getattr(parks_mask, "is_empty", True)):
            try:
                road_mask = _subtract_geometry(road_mask, parks_mask)
                print("[KEYCHAIN] Roads clipped out of green/cemetery areas (зелень не залита дорогою)")
            except Exception:
                pass
        # BRIDGE = підмножина road_mask по bridge centerlines. Гарантує
        # 1:1 співпадіння з road network → жодних розривів.
        # Алгоритм: bridge centerlines buffered щедро (15m radius у source meters)
        # → перетин з road_mask = саме ті сегменти road що належать мостам.
        if bridge_mask is not None and not getattr(bridge_mask, "is_empty", True) and road_mask is not None and not getattr(road_mask, "is_empty", True):
            try:
                # Buffer широко щоб точно покрити road_mask навколо мосту
                bridge_buffer_m = max(float(min_feature_m) * 2.0, 8.0)  # 8m+ buffer
                bridge_buffered = bridge_mask.buffer(bridge_buffer_m, join_style=2, cap_style=2).buffer(0)
                # ПЕРЕТИН з road_mask: бридж = ті сегменти road які належать мостам
                bridge_aligned = road_mask.intersection(bridge_buffered).buffer(0)
                if not bridge_aligned.is_empty:
                    bridge_mask = bridge_aligned
                    print(f"[KEYCHAIN] Bridge = road_mask ∩ bridge_centerlines.buffer({bridge_buffer_m:.1f}m) — 1:1 alignment")
                else:
                    print(f"[KEYCHAIN] Bridge intersection empty, keeping original bridge_mask")
            except Exception as exc:
                print(f"[KEYCHAIN] Bridge alignment failed: {exc}")
        if (parks_mask is None or getattr(parks_mask, "is_empty", True)) and raw_parks_source is not None and not getattr(raw_parks_source, "is_empty", True):
            print("[KEYCHAIN] Canonical parks collapsed; retrying with raw + ultra-soft filter")
            parks_mask = _sanitize_layer_mask(
                _clip_geometry(_xform(raw_parks_source), content_area),
                min_feature_m=float(min_feature_m) * 0.3,
                min_area_m2=float(min_area_m2) * 0.15,
                label="raw parks",
            )
    else:
        road_mask = getattr(bundle, "roads_final", None)
        water_mask = getattr(bundle, "water_final", None)
        parks_mask = getattr(bundle, "parks_final", None)

    # C3 ТОПО-РЕЖИМ: terrain-only — шари карти не друкуються, їх замінює
    # heightfield-рельєф на базі (будується нижче, після text-блоків).
    # Текст/вушко/форма/зворот працюють як завжди.
    if topo_mode:
        road_mask = None
        water_mask = None
        parks_mask = None
        bridge_mask = None
        gdf_buildings_local = None
        print("[KEYCHAIN TOPO] Terrain-only: map layers skipped (relief replaces the map)")
        try:
            task.update_status("processing", 72, "Будую рельєф висот на жетоні...")
        except Exception:
            pass

    # ТЕКСТ: ЗА ЗАМОВЧУВАННЯМ вирізаються/підіймаються ТІЛЬКИ літери — карта
    # навколо напису НЕ очищається (юзер: «по дефолту вирізались тільки букви і
    # щоб область вокруг їх не вирізалась»). Прямокутна зона-підкладка під текст
    # вмикається лише явним прапором keychain_label_clear_band=true.
    # ВИРІЗАННЯ ФОРМИ ЛІТЕР з усіх шарів карти + будівель, щоб текст не «накладався»
    # поверх, а сидів у чистому слід-і власної форми (юзер: «вирізались тільки букви»,
    # «будівлі вирізались під текст»). Полігон рахуємо ОДИН раз тут і передаємо
    # у build_keychain_label_mesh → ідеальний збіг вирізу й напису.
    text_letter_poly = None
    if keychain_mode and keychain_layout is not None:
        try:
            _band = keychain_layout["label_band"]
            _bminx, _bminy, _bmaxx, _bmaxy = _band.bounds
            _max_w = max((_bmaxx - _bminx) * 0.96, 1e-6)
            _max_h = max((_bmaxy - _bminy) * 0.92, 1e-6)
            _label_text = str(getattr(request, "keychain_label", "") or "")
            _label_angle = float(getattr(request, "keychain_label_angle_deg", 0.0) or 0.0)
            _label_h_m = _model_mm_to_world_m(
                float(getattr(request, "keychain_label_text_height_mm", 3.8) or 3.8),
                export_scale_factor,
            )
            _min_stroke = _model_mm_to_world_m(MIN_KEYCHAIN_TEXT_STROKE_MM, export_scale_factor)
            text_letter_poly = _compute_text_letter_polygon(
                text=_label_text,
                body_geometry=keychain_layout["body"],
                label_band_geometry=_band,
                text_height_m=_label_h_m,
                angle_deg=_label_angle,
                min_stroke_m=_min_stroke,
                max_width=_max_w,
                max_height=_max_h,
            )
            if text_letter_poly is not None and not getattr(text_letter_poly, "is_empty", True):
                # ДВОШАРОВЕ вирізання:
                # 1) Точний виріз по ФОРМІ ЛІТЕР з 0.25mm зазором → текст не плаває
                # 2) CONVEX HULL + 0.3mm → «мінімальна віртуальна зона» навколо
                #    усього напису, щоб не було мікро-залишків між буквами/словами
                #    (юзер: «зона мінімальна та віртуальна щоб вирізалось вокруг
                #    текста, щоб не було залишків всередині»). НЕ додає меш.
                _carve_letters = text_letter_poly.buffer(
                    _model_mm_to_world_m(0.25, export_scale_factor)
                ).buffer(0)
                try:
                    _carve_hull = text_letter_poly.convex_hull.buffer(
                        _model_mm_to_world_m(0.3, export_scale_factor), join_style=1
                    ).buffer(0)
                except Exception:
                    _carve_hull = _carve_letters
                for _mname, _mref in [("road_mask", road_mask), ("parks_mask", parks_mask), ("water_mask", water_mask)]:
                    try:
                        if _mref is not None and not getattr(_mref, "is_empty", True):
                            _subtracted = _subtract_geometry(_mref, _carve_hull)
                            if _mname == "road_mask":   road_mask   = _subtracted
                            elif _mname == "parks_mask": parks_mask = _subtracted
                            elif _mname == "water_mask": water_mask = _subtracted
                    except Exception: pass
                print(f"[KEYCHAIN] Letters+hull carved from road/parks/water (angle={_label_angle}°, label='{_label_text}')")
        except Exception as exc:
            print(f"[KEYCHAIN] letter polygon compute failed: {exc}")

    # ПІДПИС НА ПЛОСКІЙ МАПІ/МАГНІТІ (не-keychain): смуга внизу плити,
    # той самий механізм літер + виріз hull із шарів карти.
    map_label_letter_poly = None
    map_label_band = None
    if not keychain_mode:
        _map_label = str(getattr(request, "map_label", "") or "").strip()
        if _map_label:
            try:
                _zone_poly = zone.zone_polygon_local
                _zb = _zone_poly.bounds
                _zw = _zb[2] - _zb[0]
                _zh = _zb[3] - _zb[1]
                _txt_h_m = _model_mm_to_world_m(
                    float(getattr(request, "map_label_text_height_mm", 5.0) or 5.0),
                    export_scale_factor,
                )
                _band_h = min(_txt_h_m * 1.7, _zh * 0.25)
                from shapely.geometry import box as _map_box
                map_label_band = _map_box(
                    _zb[0] + _zw * 0.05, _zb[1] + _zh * 0.03,
                    _zb[2] - _zw * 0.05, _zb[1] + _zh * 0.03 + _band_h,
                ).intersection(_zone_poly).buffer(0)
                if map_label_band is not None and not map_label_band.is_empty:
                    _mb = map_label_band.bounds
                    map_label_letter_poly = _compute_text_letter_polygon(
                        text=_map_label,
                        body_geometry=_zone_poly,
                        label_band_geometry=map_label_band,
                        text_height_m=_txt_h_m,
                        angle_deg=0.0,
                        min_stroke_m=_model_mm_to_world_m(MIN_KEYCHAIN_TEXT_STROKE_MM, export_scale_factor),
                        max_width=max((_mb[2] - _mb[0]) * 0.96, 1e-6),
                        max_height=max((_mb[3] - _mb[1]) * 0.92, 1e-6),
                    )
                if map_label_letter_poly is not None and not getattr(map_label_letter_poly, "is_empty", True):
                    _hull_ml = map_label_letter_poly.convex_hull.buffer(
                        _model_mm_to_world_m(0.3, export_scale_factor), join_style=1
                    ).buffer(0)
                    for _nm in ("road_mask", "parks_mask", "water_mask"):
                        try:
                            _ref = {"road_mask": road_mask, "parks_mask": parks_mask, "water_mask": water_mask}[_nm]
                            if _ref is not None and not getattr(_ref, "is_empty", True):
                                _s = _subtract_geometry(_ref, _hull_ml)
                                if _nm == "road_mask":
                                    road_mask = _s
                                elif _nm == "parks_mask":
                                    parks_mask = _s
                                else:
                                    water_mask = _s
                        except Exception:
                            pass
                    print(f"[MAP LABEL] Carved + ready: '{_map_label}'")
            except Exception as exc:
                print(f"[MAP LABEL] failed (non-fatal): {exc}")
                map_label_letter_poly = None

    # ДРУГИЙ РЯДОК (дата/координати): власна смуга одразу ПІД основною,
    # менший кегль; той самий двошаровий виріз із шарів карти.
    text2_letter_poly = None
    if keychain_mode and keychain_layout is not None:
        _label2_text = str(getattr(request, "keychain_label2", "") or "").strip()
        if _label2_text:
            try:
                _band = keychain_layout["label_band"]
                _b = _band.bounds
                _band_h2 = (_b[3] - _b[1])
                # maxy = верх брелка → «нижче» = менший y. Зсуваємо band вниз.
                _band2 = affinity.translate(_band, yoff=-_band_h2 * 0.95)
                _band2 = _band2.intersection(keychain_layout["body"]).buffer(0)
                if _band2 is not None and not getattr(_band2, "is_empty", True):
                    keychain_layout["label_band2"] = _band2
                    _b2 = _band2.bounds
                    text2_letter_poly = _compute_text_letter_polygon(
                        text=_label2_text,
                        body_geometry=keychain_layout["body"],
                        label_band_geometry=_band2,
                        text_height_m=_model_mm_to_world_m(
                            float(getattr(request, "keychain_label2_text_height_mm", 2.4) or 2.4),
                            export_scale_factor,
                        ),
                        angle_deg=float(getattr(request, "keychain_label_angle_deg", 0.0) or 0.0),
                        min_stroke_m=_model_mm_to_world_m(MIN_KEYCHAIN_TEXT_STROKE_MM, export_scale_factor),
                        max_width=max((_b2[2] - _b2[0]) * 0.96, 1e-6),
                        max_height=max((_b2[3] - _b2[1]) * 0.92, 1e-6),
                    )
                if text2_letter_poly is not None and not getattr(text2_letter_poly, "is_empty", True):
                    _carve_hull2 = text2_letter_poly.convex_hull.buffer(
                        _model_mm_to_world_m(0.3, export_scale_factor), join_style=1
                    ).buffer(0)
                    for _mname2 in ("road_mask", "parks_mask", "water_mask"):
                        try:
                            _mref2 = {"road_mask": road_mask, "parks_mask": parks_mask, "water_mask": water_mask}[_mname2]
                            if _mref2 is not None and not getattr(_mref2, "is_empty", True):
                                _sub2 = _subtract_geometry(_mref2, _carve_hull2)
                                if _mname2 == "road_mask":
                                    road_mask = _sub2
                                elif _mname2 == "parks_mask":
                                    parks_mask = _sub2
                                else:
                                    water_mask = _sub2
                        except Exception:
                            pass
                    print(f"[KEYCHAIN] Label2 carved: '{_label2_text}'")
            except Exception as exc:
                print(f"[KEYCHAIN] label2 polygon failed (non-fatal): {exc}")
                text2_letter_poly = None
            text_letter_poly = None

    label_clear_band = None
    if keychain_mode and keychain_layout is not None and bool(getattr(request, "keychain_label_clear_band", False)):
        label_clear_band = keychain_layout.get("label_clear_band")
        if label_clear_band is not None and not getattr(label_clear_band, "is_empty", True):
            try:
                if road_mask is not None and not getattr(road_mask, "is_empty", True):
                    road_mask = _subtract_geometry(road_mask, label_clear_band)
                if parks_mask is not None and not getattr(parks_mask, "is_empty", True):
                    parks_mask = _subtract_geometry(parks_mask, label_clear_band)
                if water_mask is not None and not getattr(water_mask, "is_empty", True):
                    water_mask = _subtract_geometry(water_mask, label_clear_band)
                print("[KEYCHAIN] Map layers cleared under label band (opt-in clean text background)")
            except Exception as exc:
                print(f"[KEYCHAIN] Label band map-clear failed: {exc}")
        else:
            label_clear_band = None

    # ПРЕМІУМ-РАМКА (компас + масштабна лінійка + координати): рахуємо ДО побудови
    # шарів, щоб вирізати її silhouette з road/parks/water (як map_label) — текст/
    # стрілка читаються чисто на тлі бази. Окрема чорна деталь «Frame» зверху.
    if not keychain_mode and bool(getattr(request, "map_frame", False)):
        try:
            map_frame_overlay = build_map_frame_overlay(
                zone.zone_polygon_local,
                north=float(getattr(request, "north", 0.0) or 0.0),
                south=float(getattr(request, "south", 0.0) or 0.0),
                east=float(getattr(request, "east", 0.0) or 0.0),
                west=float(getattr(request, "west", 0.0) or 0.0),
                export_scale_factor=export_scale_factor,
                want_compass=bool(getattr(request, "map_frame_compass", True)),
                want_scale=bool(getattr(request, "map_frame_scale", True)),
                want_coords=bool(getattr(request, "map_frame_coords", True)),
                frame_style=str(getattr(request, "frame_style", "classic") or "classic"),
            )
            if map_frame_overlay is not None and not map_frame_overlay.is_empty:
                map_frame_hull = map_frame_overlay.convex_hull.buffer(
                    _model_mm_to_world_m(0.6, export_scale_factor), join_style=1
                ).buffer(0)
                # Розбиваємо на per-елемент опуклі оболонки (компас/лінійка/координати
                # у різних кутах) — спільна опукла оболонка з'їла б половину карти.
                # ПАСТКА: ободок (ornate/compass) — кільцеподібний, його convex_hull =
                # вся плита. Для таких частин (hull-площа ≫ власної площі) ріжемо лише
                # буфер самої геометрії, а не опуклу оболонку.
                _pad = _model_mm_to_world_m(0.6, export_scale_factor)
                _clears = []
                for _g in (map_frame_overlay.geoms if hasattr(map_frame_overlay, "geoms") else [map_frame_overlay]):
                    try:
                        _hull = _g.convex_hull
                        _g_area = getattr(_g, "area", 0.0) or 0.0
                        if getattr(_hull, "area", 0.0) > _g_area * 4.0 + 1e-9:
                            _clears.append(_g.buffer(_pad, join_style=1))  # кільце → лише сам контур
                        else:
                            _clears.append(_hull.buffer(_pad, join_style=1))
                    except Exception:
                        pass
                map_frame_hull = unary_union(_clears).buffer(0) if _clears else map_frame_hull
                if road_mask is not None and not getattr(road_mask, "is_empty", True):
                    road_mask = _subtract_geometry(road_mask, map_frame_hull)
                if parks_mask is not None and not getattr(parks_mask, "is_empty", True):
                    parks_mask = _subtract_geometry(parks_mask, map_frame_hull)
                if water_mask is not None and not getattr(water_mask, "is_empty", True):
                    water_mask = _subtract_geometry(water_mask, map_frame_hull)
                map_frame_mesh = build_flat_layer_mesh_from_mask(
                    map_frame_overlay, bottom_z_m=base_top_m,
                    thickness_m=_model_mm_to_world_m(0.6, export_scale_factor),
                    color=LAYER_COLORS.get("rim", [25, 25, 25, 255]), min_area_m2=1e-12,
                )
                print(f"[MAP FRAME] style={str(getattr(request, 'frame_style', 'classic'))} compass/scale/coords ready ({len(getattr(map_frame_overlay,'geoms',[1]))} elems), carved from map layers")
        except Exception as exc:
            print(f"[MAP FRAME] failed (non-fatal): {exc}")
            map_frame_mesh = None

    # ВОДА = ОДИН ВЕРХНІЙ ШАР, врівень із землею (НЕ на всю глибину).
    # Юзер: "вода важається до кінця моделі з іншої сторони — треба тільки один
    # шар". Раніше water заповнювала 0 → base_top (вся товщина) і її було видно
    # знизу/збоку наскрізь. Тепер:
    #   • water_mesh (синій): тонкий ВЕРХНІЙ зріз (base_top - water_depth → base_top),
    #     верх співпадає з землею → flush.
    #   • water_plug_mesh (колір бази): заповнює НИЖНЮ частину вирізаної дірки
    #     (0 → base_top - water_depth), щоб знизу був колір бази, а не вода.
    # Результат: один шар води зверху, решта глибини — база.
    water_plug_mesh: Optional[trimesh.Trimesh] = None
    water_depth_m = _model_mm_to_world_m(water_layer_mm, export_scale_factor)
    water_clipped = _clip_geometry(water_mask, content_area)
    # КРИТИЧНО: water_mask може мати ВЕЛИКІ полігони (Дніпро, озера) що
    # тягнуться поза content_area. Robust intersection з content_area щоб
    # не було гігантських bbox у Bambu (іноді трапляється коли relation outer
    # не правильно сток'нувся).
    if keychain_mode and water_clipped is not None and not getattr(water_clipped, "is_empty", True) and keychain_layout is not None:
        try:
            # STRICT intersection — обмежуємо water до площі content_area
            water_clipped = water_clipped.intersection(keychain_layout["content_area"]).buffer(0)
            if water_clipped.is_empty:
                raise ValueError("water empty after strict clip")
            water_bottom_m = max(base_top_m - water_depth_m, 0.0)
            # Тонкий верхній шар води (flush з землею)
            water_mesh = build_flat_layer_mesh_from_mask(
                water_clipped,
                bottom_z_m=water_bottom_m,
                thickness_m=base_top_m - water_bottom_m,
                color=LAYER_COLORS["water"],
                min_area_m2=min_area_m2,
            )
            # Заглушка з кольором бази під водою (щоб не було наскрізної води)
            if water_bottom_m > 1e-6:
                water_plug_mesh = build_flat_layer_mesh_from_mask(
                    water_clipped,
                    bottom_z_m=0.0,
                    thickness_m=water_bottom_m,
                    color=LAYER_COLORS["base"],
                    min_area_m2=min_area_m2,
                )
        except Exception as exc:
            print(f"[KEYCHAIN] Water single-layer build failed, fallback: {exc}")
            water_mesh = build_flat_layer_mesh_from_mask(
                water_clipped, bottom_z_m=max(base_top_m - water_depth_m, 0.0),
                thickness_m=min(water_depth_m, base_top_m),
                color=LAYER_COLORS["water"], min_area_m2=min_area_m2,
            )
    else:
        # Non-keychain або no water: класичний шар поверх
        water_mesh = build_flat_layer_mesh_from_mask(
            water_clipped,
            bottom_z_m=base_top_m,
            thickness_m=water_depth_m,
            color=LAYER_COLORS["water"],
            min_area_m2=min_area_m2,
        )
    road_mesh = build_flat_layer_mesh_from_mask(
        _clip_geometry(road_mask, content_area),
        bottom_z_m=base_top_m,
        thickness_m=_model_mm_to_world_m(roads_layer_mm, export_scale_factor),
        color=LAYER_COLORS["roads"],
        min_area_m2=min_area_m2,
    )
    # Bridge mesh — НА РІВНІ ЗВИЧАЙНИХ ДОРІГ (не вище), просто інший колір.
    # Раніше робилось +0.2mm вище, але візуально мости "плавали в повітрі"
    # — особливо коли під ними немає води (тінь з боків). Тепер міст лежить
    # на тій же висоті що road, відрізняється тільки кольором.
    bridge_mesh = None
    if keychain_mode and 'bridge_mask' in dir() and bridge_mask is not None and not getattr(bridge_mask, "is_empty", True):
        try:
            bridge_top_m = base_top_m  # ТАКА Ж висота як roads
            bridge_thickness_m = _model_mm_to_world_m(roads_layer_mm, export_scale_factor)  # = roads
            # КРИТИЧНО: bridge — вузька лінія, агресивний min_area_m2 (як для парків)
            # її повністю зʼїдає. Використовуємо МІНІМАЛЬНИЙ поріг — площа полігона
            # моста 5×0.5mm = 2.5mm² → потрібен поріг ~0.5mm² у model space.
            bridge_min_area_m2 = _model_mm_to_world_m(0.5, export_scale_factor) ** 2
            bridge_mesh = build_flat_layer_mesh_from_mask(
                _clip_geometry(bridge_mask, content_area),
                bottom_z_m=bridge_top_m,
                thickness_m=bridge_thickness_m,
                color=LAYER_COLORS.get("rim", [92, 80, 58, 255]),  # темно-бежевий
                min_area_m2=bridge_min_area_m2,
            )
            if bridge_mesh is not None:
                print(f"[KEYCHAIN] Bridges rendered AT road level (dark-beige color, min_area={bridge_min_area_m2*1e6:.2f}mm²)")
            else:
                print(f"[KEYCHAIN] Bridge mesh build returned None (mask area may be below threshold)")
        except Exception as exc:
            print(f"[KEYCHAIN] Bridge mesh build failed: {exc}")
            bridge_mesh = None
    parks_mesh = build_flat_layer_mesh_from_mask(
        _clip_geometry(parks_mask, content_area),
        bottom_z_m=base_top_m,
        thickness_m=_model_mm_to_world_m(parks_layer_mm, export_scale_factor),
        color=LAYER_COLORS["parks"],
        min_area_m2=min_area_m2,
    )

    # D4 GPX-ТРЕК: маршрут користувача — підвищений шар ПОВЕРХ карти
    # (на 0.2мм вище доріг, теракотовий). Для брелка проходить той самий
    # unwrap-трансформ, що й дороги.
    gpx_track_mesh: Optional[trimesh.Trimesh] = None
    if getattr(request, "gpx_track", None) and not topo_mode:
        try:
            from services.gpx_track import TRACK_COLOR, build_gpx_track_polygon

            _gpx_poly = build_gpx_track_polygon(
                gpx_track=request.gpx_track,
                global_center=global_center,
                zone_polygon_local=zone.zone_polygon_local,
                scale_factor=scale_factor,
                width_mm=float(getattr(request, "gpx_width_mm", 1.2) or 1.2),
            )
            if _gpx_poly is not None and keychain_layout is not None and source_bounds and target_bounds:
                _gpx_poly = _xform(_gpx_poly)
            if _gpx_poly is not None:
                # ВРІЗАНИЙ маршрут: верх вставки flush з поверхнею (base_top), тіло
                # втоплене у базу на ~recess. Раніше був ПІДВИЩЕНИЙ над дорогами.
                # Глибина врізу = request.gpx_raise_mm (0.2–1.5мм), а НЕ хардкод —
                # full_generation_pipeline теж читає цей параметр (паритет рельєф/флет).
                _gpx_recess_mm = float(getattr(request, "gpx_raise_mm", 0.6) or 0.6)
                _gpx_recess_m = _model_mm_to_world_m(_gpx_recess_mm, export_scale_factor)
                gpx_track_mesh = build_flat_layer_mesh_from_mask(
                    _clip_geometry(_gpx_poly, content_area),
                    bottom_z_m=max(base_top_m - _gpx_recess_m, 0.0),
                    thickness_m=min(_gpx_recess_m, base_top_m),
                    color=TRACK_COLOR,
                    min_area_m2=max(_model_mm_to_world_m(0.3, export_scale_factor) ** 2, 1e-12),
                )
                if gpx_track_mesh is not None:
                    print(f"[GPX] Flat track INLAY built (flush, recess {_gpx_recess_mm:.2f}mm)")
        except Exception as exc:
            print(f"[GPX] flat track failed (non-fatal): {exc}")

    if keychain_mode:
        # КРИТИЧНО: тільки ОДИН transform — _xform (новий unwrap). Старий
        # _orient_then_stretch_gdf_into_bounds ВИДАЛЕНО, бо він робив подвійну
        # трансформацію поверх _xform → координати спотворювались, будівлі
        # летіли за слот і фільтрувались до 0.
        if gdf_buildings_local is not None and not gdf_buildings_local.empty:
            try:
                xformed_geoms = [_xform(g) for g in gdf_buildings_local.geometry.values]
                xformed_geoms = [g if g is not None and not getattr(g, "is_empty", True) else None for g in xformed_geoms]
                gdf_xformed = gdf_buildings_local.copy()
                gdf_xformed.geometry = xformed_geoms
                gdf_xformed = gdf_xformed[gdf_xformed.geometry.notna()]
                gdf_buildings_local = gdf_xformed
                print(f"[KEYCHAIN] Buildings transformed through unwrap: {len(gdf_buildings_local)} remaining")
            except Exception as exc:
                print(f"[KEYCHAIN] Building unwrap failed (using source coords): {exc}")
        # Будівлі також прибираємо з-під напису (clean text background).
        building_clip_area = content_area
        if keychain_mode and label_clear_band is not None and not getattr(label_clear_band, "is_empty", True):
            try:
                reduced = content_area.difference(label_clear_band).buffer(0)
                if reduced is not None and not reduced.is_empty:
                    building_clip_area = reduced
            except Exception:
                building_clip_area = content_area
        gdf_buildings_local = _clip_buildings_to_content(gdf_buildings_local, building_clip_area)
        # ВИРІЗАЄМО ФОРМУ ЛІТЕР з footprint-ів будівель → текст не лежить ЗВЕРХУ
        # будівлі, а сидить у вирізаному слід-у (юзер: «будівлі вирізались під
        # текст, бо зараз текст просто наклада[ється]»).
        if text_letter_poly is not None and not getattr(text_letter_poly, "is_empty", True) and gdf_buildings_local is not None:
            try:
                # Вирізаємо convex hull (щоб не було залишків між буквами)
                try:
                    carve_b = text_letter_poly.convex_hull.buffer(
                        _model_mm_to_world_m(0.3, export_scale_factor), join_style=1
                    ).buffer(0)
                except Exception:
                    carve_b = text_letter_poly.buffer(_model_mm_to_world_m(0.25, export_scale_factor)).buffer(0)
                kept_geoms = []
                for _, _row in gdf_buildings_local.iterrows():
                    _g = _row.geometry
                    if _g is None or getattr(_g, "is_empty", True):
                        continue
                    try:
                        _g2 = _g.difference(carve_b)
                        if _g2 is not None and not _g2.is_empty:
                            _nr = _row.copy(); _nr.geometry = _g2
                            kept_geoms.append(_nr)
                    except Exception:
                        kept_geoms.append(_row)
                if kept_geoms:
                    gdf_buildings_local = GeoDataFrame(kept_geoms, columns=gdf_buildings_local.columns, crs=getattr(gdf_buildings_local, "crs", None))
                print("[KEYCHAIN] Letters carved out of building footprints (text recessed, not floating)")
            except Exception as exc:
                print(f"[KEYCHAIN] letter carve from buildings failed: {exc}")

    if keychain_mode:
        try: task.update_status("processing", 80, "Будую 3D будівлі з висотами OSM...")
        except Exception: pass
    building_meshes, landmark_meshes = build_flat_building_meshes(
        request=request,
        scale_factor=scale_factor,
        export_scale_factor=export_scale_factor,
        gdf_buildings_local=gdf_buildings_local,
        base_top_m=base_top_m,
    )
    # ПРЕМІУМ-РАМКА: прибираємо будівлі під компасом/лінійкою/координатами, щоб
    # вони не стирчали крізь чорну деталь (маски road/water/parks вже вирізано).
    if map_frame_hull is not None and not getattr(map_frame_hull, "is_empty", True) and building_meshes:
        try:
            _before_bf = len(building_meshes)
            _kept = []
            for _bm in building_meshes:
                try:
                    _bb = _bm.bounds
                    _foot = box(float(_bb[0][0]), float(_bb[0][1]), float(_bb[1][0]), float(_bb[1][1]))
                    if map_frame_hull.intersects(_foot):
                        continue  # будівля під рамкою → прибрати
                except Exception:
                    pass
                _kept.append(_bm)
            building_meshes = _kept
            if len(building_meshes) < _before_bf:
                print(f"[MAP FRAME] Dropped {_before_bf - len(building_meshes)} building(s) under frame")
        except Exception as exc:
            print(f"[MAP FRAME] building drop failed (non-fatal): {exc}")
    if keychain_layout is not None:
        # RAISED TEXT — текст підіймається над базою як рельєф (попередня логіка).
        text_raise_mm = float(getattr(request, "keychain_label_raise_mm", KEYCHAIN_TEXT_RAISE_MM) or KEYCHAIN_TEXT_RAISE_MM)
        text_raise_m = _model_mm_to_world_m(text_raise_mm, export_scale_factor)
        _dbg_label = str(getattr(request, "keychain_label", "") or "")
        _dbg_angle = float(getattr(request, "keychain_label_angle_deg", 0.0) or 0.0)
        print(f"[KEYCHAIN TEXT] label='{_dbg_label}' angle_deg={_dbg_angle} "
              f"text_height_mm={getattr(request, 'keychain_label_text_height_mm', None)}")
        keychain_text_mesh = build_keychain_label_mesh(
            str(getattr(request, "keychain_label", "") or ""),
            body_geometry=keychain_layout["body"],
            label_band_geometry=keychain_layout["label_band"],
            bottom_z_m=base_top_m,                # на верхній грані бази
            thickness_m=text_raise_m,             # піднятий рельєф
            text_height_m=_model_mm_to_world_m(float(getattr(request, "keychain_label_text_height_mm", 3.8) or 3.8), export_scale_factor),
            color=LAYER_COLORS["text"],
            stroke_width_m=_model_mm_to_world_m(float(getattr(request, "keychain_label_stroke_mm", MIN_KEYCHAIN_TEXT_STROKE_MM) or MIN_KEYCHAIN_TEXT_STROKE_MM), export_scale_factor),
            angle_deg=float(getattr(request, "keychain_label_angle_deg", 0.0) or 0.0),
            min_stroke_m=_model_mm_to_world_m(MIN_KEYCHAIN_TEXT_STROKE_MM, export_scale_factor),
            font_style=str(getattr(request, "keychain_label_font_style", "block") or "block"),
            precomputed_polygon=text_letter_poly,
        )
        # Другий рядок — той самий механізм, полігон уже порахований у carve-блоці.
        if text2_letter_poly is not None and keychain_layout.get("label_band2") is not None:
            keychain_text2_mesh = build_keychain_label_mesh(
                str(getattr(request, "keychain_label2", "") or ""),
                body_geometry=keychain_layout["body"],
                label_band_geometry=keychain_layout["label_band2"],
                bottom_z_m=base_top_m,
                thickness_m=text_raise_m,
                text_height_m=_model_mm_to_world_m(
                    float(getattr(request, "keychain_label2_text_height_mm", 2.4) or 2.4),
                    export_scale_factor,
                ),
                color=LAYER_COLORS["text"],
                angle_deg=float(getattr(request, "keychain_label_angle_deg", 0.0) or 0.0),
                min_stroke_m=_model_mm_to_world_m(MIN_KEYCHAIN_TEXT_STROKE_MM, export_scale_factor),
                font_style=str(getattr(request, "keychain_label_font_style", "block") or "block"),
                precomputed_polygon=text2_letter_poly,
            )
        # VIRTUAL CLEAR ZONE навколо тексту: НЕ додаємо фізичну підкладку (щоб
        # не було сірої платформи під буквами — юзер: «зона створення що є
        # проблемою»). Натомість вирізаємо CONVEX HULL + 0.3мм padding з усіх
        # шарів карти через збережену маску carve_zone. Це прибирає мікро-залишки
        # доріг/парків між літерами — «зона мінімальна та віртуальна».
        if text_letter_poly is not None and not getattr(text_letter_poly, "is_empty", True):
            try:
                _hull_pad = _model_mm_to_world_m(0.3, export_scale_factor)
                _hull_zone = text_letter_poly.convex_hull.buffer(_hull_pad, join_style=1).buffer(0)
                _hull_zone = _hull_zone.intersection(keychain_layout["body"]).buffer(0)
                if _hull_zone is not None and not getattr(_hull_zone, "is_empty", True):
                    # Вирізаємо hull-зону з road/parks/water щоб не було залишків
                    if road_mesh is not None:
                        try:
                            _hull_geom = _clip_geometry(_xform(_hull_zone) if False else _hull_zone, content_area)
                        except Exception:
                            _hull_geom = _hull_zone
                    # Вирізаємо з вже побудованих масок через subtract (дорого але правильно)
                    # NOTE: road/parks/water mesh вже побудовані, тому вирізаємо з кожного mesh
                    print(f"[KEYCHAIN] Convex hull carve zone ready, area={_hull_zone.area*1e6:.1f}mm²")
            except Exception as exc:
                print(f"[KEYCHAIN] hull zone compute failed (non-fatal): {exc}")
                _hull_zone = None
        else:
            _hull_zone = None
        # Залишаємо тільки water depression carve (без text engrave)
        try:
            if water_clipped is not None and not getattr(water_clipped, "is_empty", True):
                base_poly_combined = keychain_layout["base"].difference(water_clipped).buffer(0)
                if base_poly_combined is not None and not base_poly_combined.is_empty:
                    if hasattr(base_poly_combined, "geoms"):
                        largest = max(base_poly_combined.geoms, key=lambda g: g.area)
                        n_dropped = sum(1 for g in base_poly_combined.geoms if g is not largest)
                        if n_dropped > 0:
                            print(f"[KEYCHAIN] Base fragments cleanup: dropped {n_dropped} tiny pieces")
                        base_poly_combined = largest
                if base_poly_combined is not None and not base_poly_combined.is_empty:
                    new_terrain, new_bottom = _build_keychain_base_parts(
                        base_poly_combined,
                        base_top_m=base_top_m,
                        back_text_poly=keychain_back_poly,
                        engrave_m=keychain_back_engrave_m,
                    )
                    if new_terrain is not None:
                        terrain_mesh = new_terrain
                        keychain_base_bottom_mesh = new_bottom
                        print(f"[KEYCHAIN] Base rebuilt with water depression ({water_layer_mm:.2f}mm)")
        except Exception as exc:
            print(f"[KEYCHAIN] Water carve failed: {exc}")

        # C3 ТОПО-БРЕЛОК: верхня грань бази стає heightfield-рельєфом.
        # Плоскими лишаються: вушко (поза content_area), rim, смуги напису.
        if topo_mode:
            try:
                # Кап 6мм (було 4мм) — гори на топо-брелку виразніші, лишаючись
                # друкабельними на жетоні (стінка ще достатня над рельєфом).
                relief_mm = min(max(float(getattr(request, "keychain_relief_mm", 2.2) or 2.2), 0.6), 6.0)
                # Під текстом рельєф флетимо до base_top (текст має стояти на рівному)
                _flatten_zones = []
                for _band_key, _txt in (
                    ("label_band", str(getattr(request, "keychain_label", "") or "").strip()),
                    ("label_band2", str(getattr(request, "keychain_label2", "") or "").strip()),
                ):
                    _bnd = keychain_layout.get(_band_key)
                    if _txt and _bnd is not None and not getattr(_bnd, "is_empty", True):
                        try:
                            _flatten_zones.append(
                                _bnd.buffer(_model_mm_to_world_m(1.2, export_scale_factor), join_style=1)
                            )
                        except Exception:
                            _flatten_zones.append(_bnd)
                _relief_zone = _subtract_geometry(content_area, *_flatten_zones) if _flatten_zones else content_area
                if _relief_zone is None or getattr(_relief_zone, "is_empty", True):
                    _relief_zone = content_area
                topo_mesh, topo_bottom = _build_keychain_topo_base(
                    request=request,
                    zone=zone,
                    global_center=global_center,
                    base_mask=keychain_layout["base"],
                    relief_zone=_relief_zone,
                    base_top_m=base_top_m,
                    relief_m=_model_mm_to_world_m(relief_mm, export_scale_factor),
                    feather_m=_model_mm_to_world_m(1.5, export_scale_factor),
                    unwrap_params=locals().get("unwrap_params"),
                    source_bounds=source_bounds,
                    target_bounds=target_bounds,
                    map_rotation_deg=float(getattr(request, "keychain_map_rotation_deg", 0.0) or 0.0),
                    back_text_poly=keychain_back_poly,
                    engrave_m=keychain_back_engrave_m,
                    export_scale_factor=export_scale_factor,
                )
                if topo_mesh is not None:
                    terrain_mesh = topo_mesh
                    keychain_base_bottom_mesh = topo_bottom
                    print(f"[KEYCHAIN TOPO] Relief base built: +{relief_mm:.2f}mm heightfield on token")
                else:
                    print("[KEYCHAIN TOPO] Relief unavailable — flat base kept (non-fatal)")
            except Exception as exc:
                print(f"[KEYCHAIN TOPO] relief failed (non-fatal, flat base kept): {exc}")

    # Підпис на плоскій мапі/магніті (не-keychain): рельєф 0.8мм на верхній грані бази.
    if not keychain_mode and map_label_letter_poly is not None and map_label_band is not None:
        map_text_mesh = build_keychain_label_mesh(
            str(getattr(request, "map_label", "") or ""),
            body_geometry=zone.zone_polygon_local,
            label_band_geometry=map_label_band,
            bottom_z_m=base_top_m,
            thickness_m=_model_mm_to_world_m(0.8, export_scale_factor),
            text_height_m=_model_mm_to_world_m(
                float(getattr(request, "map_label_text_height_mm", 5.0) or 5.0),
                export_scale_factor,
            ),
            color=LAYER_COLORS["text"],
            min_stroke_m=_model_mm_to_world_m(MIN_KEYCHAIN_TEXT_STROKE_MM, export_scale_factor),
            precomputed_polygon=map_label_letter_poly,
        )

    # ПІДСВІТКА БУДИНКУ (брелок): обрані будинки → ОКРЕМА ЧЕРВОНА вставна деталь
    # (паз+peg, механічна вставка БЕЗ клею) — друкується окремим філаментом,
    # вставляється у паз бази; економить час/філамент проти AMS.
    #  • highlight_points [[lon,lat],...] (клік юзера по своєму будинку) →
    #    КОЖЕН будинок окремо, ТОЧНО як у map-гілці (lon/lat→to_utm→to_local, далі
    #    через ЗАХОПЛЕНИЙ unwrap _keychain_xform у систему building_meshes);
    #  • БЕЗ точок → старий авто-пік будинку в ЦЕНТРІ тіла (golden-safe, байт-в-байт).
    highlight_building_mesh = None
    if (
        keychain_mode
        and bool(getattr(request, "keychain_highlight_building", False))
        and building_meshes
        and keychain_layout is not None
    ):
        try:
            # Збираємо цілі. Точки кліку (lon/lat) переводимо у unwrapped-систему
            # building_meshes тим самим transform, яким пройшли самі будівлі.
            _kc_raw_pts = list(getattr(request, "highlight_points", None) or [])
            _kc_single = getattr(request, "highlight_point", None)
            if not _kc_raw_pts and _kc_single and len(_kc_single) >= 2:
                _kc_raw_pts = [_kc_single]
            _kc_targets: list[tuple[float, float]] = []
            for _kc_hp in _kc_raw_pts[:12]:  # кап 12 будівель
                if not _kc_hp or len(_kc_hp) < 2:
                    continue
                try:
                    _kc_ux, _kc_uy = global_center.to_utm(float(_kc_hp[0]), float(_kc_hp[1]))
                    _kc_lx, _kc_ly = global_center.to_local(_kc_ux, _kc_uy)
                    _kc_cx, _kc_cy = float(_kc_lx), float(_kc_ly)
                    if _keychain_xform is not None:
                        # ВАЖЛИВО: _xform закінчується .buffer(0), що знищив би 0-площинну
                        # точку → переводимо МАЛЕНЬКИЙ квадрат і беремо його центроїд
                        # (той самий unwrap, що пройшли building_meshes).
                        _kc_probe = _keychain_xform(Point(_kc_cx, _kc_cy).buffer(0.5))
                        if _kc_probe is not None and not getattr(_kc_probe, "is_empty", True):
                            _kc_pc = _kc_probe.centroid
                            _kc_cx, _kc_cy = float(_kc_pc.x), float(_kc_pc.y)
                    _kc_targets.append((_kc_cx, _kc_cy))
                except Exception as _kc_hpx:
                    print(f"[KEYCHAIN HIGHLIGHT] point→local failed ({_kc_hpx}); skip point")
            if not _kc_targets:  # NO-POINTS FALLBACK — будинок у ЦЕНТРІ тіла (golden-safe)
                _hb_ctr = keychain_layout["body"].centroid
                _kc_targets = [(float(_hb_ctr.x), float(_hb_ctr.y))]

            _kc_chosen: list[int] = []
            for _kc_t in _kc_targets:
                _kc_i = _select_highlight_building_index(
                    building_meshes, target_xy=_kc_t, exclude=set(_kc_chosen)
                )
                if _kc_i is not None and _kc_i not in _kc_chosen:
                    _kc_chosen.append(_kc_i)
            if _kc_chosen:
                # Глибина пазу І peg-ніжки = ОДНЕ безпечне значення наперед: лишаємо
                # ≥0.4мм суцільного над зворот-гравіюванням. Раніше peg будувався на
                # повну 0.8мм, а through-hole-guard зрізав ЛИШЕ паз (0.8→0.6) → ніжка
                # на 0.2мм довша за паз = вставка не сідала. Тепер обидва з depth_mm.
                _kc_base_mm = base_top_m * export_scale_factor
                _kc_eng_mm = ((keychain_back_engrave_m or 0.0) * export_scale_factor) if keychain_back_poly is not None else 0.0
                _kc_hl_depth_mm = 0.8 if _kc_eng_mm <= 0 else max(min(0.8, _kc_base_mm - _kc_eng_mm - 0.4), 0.2)
                _kc_hl_meshes, _kc_pockets, _kc_pocket_depth = [], [], 0.0
                for _kc_i in _kc_chosen:  # індекси стабільні — НЕ видаляємо в циклі
                    _kc_m, _kc_pk, _kc_d = build_highlight_insert(
                        building_meshes[_kc_i], base_top_m=base_top_m,
                        export_scale_factor=export_scale_factor, depth_mm=_kc_hl_depth_mm,
                    )
                    if _kc_m is not None:
                        _kc_hl_meshes.append(_kc_m)
                    if _kc_pk is not None and _kc_d > 1e-9:
                        _kc_pockets.append(_kc_pk)
                        _kc_pocket_depth = _kc_d
                # прибираємо обрані будинки з шару Buildings (стають вставкою)
                _kc_chosen_set = set(_kc_chosen)
                building_meshes[:] = [b for j, b in enumerate(building_meshes) if j not in _kc_chosen_set]
                if _kc_pockets:
                    _kc_pocket_union = unary_union(_kc_pockets).buffer(0)
                    # THROUGH-HOLE GUARD: паз зверху + гравіювання звороту знизу не
                    # повинні з'їсти базу наскрізь (≥0.4мм суцільного матеріалу). База
                    # жетона фіксована (шари вже сидять зверху) → ріжемо глибший виріз.
                    _kc_safe_top, _kc_safe_bot = _ensure_no_through_hole(
                        base_top_m=base_top_m,
                        top_cut_poly=_kc_pocket_union, top_cut_depth_m=_kc_pocket_depth,
                        bottom_cut_poly=keychain_back_poly, bottom_cut_depth_m=keychain_back_engrave_m,
                        export_scale_factor=export_scale_factor, label="KEYCHAIN HIGHLIGHT",
                    )
                    _new_t, _new_b = _build_keychain_base_parts(
                        keychain_layout["base"], base_top_m=base_top_m,
                        back_text_poly=keychain_back_poly, engrave_m=_kc_safe_bot,
                        top_cut_poly=_kc_pocket_union, top_cut_depth_m=_kc_safe_top,
                    )
                    if _new_t is not None:
                        terrain_mesh, keychain_base_bottom_mesh = _new_t, _new_b
                if _kc_hl_meshes:
                    highlight_building_mesh = (
                        trimesh.util.concatenate(_kc_hl_meshes) if len(_kc_hl_meshes) > 1 else _kc_hl_meshes[0]
                    )
                    _with_color(highlight_building_mesh, LAYER_COLORS["highlight"])
                print(f"[KEYCHAIN HIGHLIGHT] {len(_kc_chosen)} building(s) → red 'Highlight' part"
                      + (f" + {len(_kc_pockets)} pocket(s) (механічна вставка)" if _kc_pockets else " (glue-on, замалий для peg)"))
        except Exception as _hbexc:
            print(f"[KEYCHAIN] highlight building failed (non-fatal): {_hbexc}")

    # ВИДІЛЕНІ БУДІВЛІ (КАРТА): користувач КЛІКАЄ свої будинки на карті
    # (highlight_points [[lon,lat],...] — дім, робота, орієнтири) → КОЖЕН стає ОКРЕМОЮ
    # ЧЕРВОНОЮ вставною деталлю (паз+peg). Усі — в ОДИН part «Highlight», усі пази —
    # одним вирізом у базі (зберігаючи нижній виріз магніт/конектор). Друкуються
    # окремим філаментом, вставляються → економія часу/філаменту проти AMS.
    if map_highlight_building and building_meshes and not keychain_mode:
        try:
            # Збираємо точки-цілі (lon,lat → to_utm → to_local). Підтримуємо і список
            # highlight_points, і одиночний highlight_point (бекв-сумісність).
            _raw_pts = list(getattr(request, "highlight_points", None) or [])
            _single = getattr(request, "highlight_point", None)
            if not _raw_pts and _single and len(_single) >= 2:
                _raw_pts = [_single]
            _targets: list[tuple[float, float]] = []
            for _hp in _raw_pts[:12]:  # кап 12 будівель
                if not _hp or len(_hp) < 2:
                    continue
                try:
                    _ux, _uy = global_center.to_utm(float(_hp[0]), float(_hp[1]))
                    _lx, _ly = global_center.to_local(_ux, _uy)
                    _targets.append((float(_lx), float(_ly)))
                except Exception as _hpx:
                    print(f"[MAP HIGHLIGHT] point→local failed ({_hpx}); skip point")
            if not _targets:  # без кліків → будинок у центрі зони
                _zc = zone.zone_polygon_local.centroid
                _targets = [(float(_zc.x), float(_zc.y))]

            _chosen: list[int] = []
            for _t in _targets:
                _i = _select_highlight_building_index(building_meshes, target_xy=_t, exclude=set(_chosen))
                if _i is not None and _i not in _chosen:
                    _chosen.append(_i)
            if _chosen:
                _hl_meshes, _pockets = [], []
                for _i in _chosen:  # будуємо вставки (НЕ видаляємо поки — індекси стабільні)
                    _m, _pk, _d = build_highlight_insert(
                        building_meshes[_i], base_top_m=base_top_m, export_scale_factor=export_scale_factor,
                    )
                    if _m is not None:
                        _hl_meshes.append(_m)
                    if _pk is not None and _d > 1e-9:
                        _pockets.append(_pk)
                        _mh_depth = _d
                # прибираємо обрані будинки з шару Buildings
                _chosen_set = set(_chosen)
                building_meshes[:] = [b for j, b in enumerate(building_meshes) if j not in _chosen_set]
                if _pockets:  # ОДИН виріз усіх пазів (+ збережений нижній виріз)
                    _pocket_union = unary_union(_pockets).buffer(0)
                    # THROUGH-HOLE GUARD (late): паз highlight (верх) ⨯ магніт/конектор
                    # (низ) — гарантуємо ≥0.4мм суцільного матеріалу. Основну роботу
                    # зробив РАННІЙ guard (потовщив базу), це — підстраховка: база вже
                    # фіксована (шари сидять зверху), тож тут лише ріжемо глибший виріз.
                    _mh_safe_top, _mh_safe_bot = _ensure_no_through_hole(
                        base_top_m=base_top_m,
                        top_cut_poly=_pocket_union, top_cut_depth_m=_mh_depth,
                        bottom_cut_poly=_flat_base_bottom_poly, bottom_cut_depth_m=_flat_base_bottom_depth_m,
                        export_scale_factor=export_scale_factor, label="MAP HIGHLIGHT",
                    )
                    _new_t, _new_b = _build_keychain_base_parts(
                        zone.zone_polygon_local, base_top_m=base_top_m,
                        back_text_poly=_flat_base_bottom_poly, engrave_m=_mh_safe_bot,
                        top_cut_poly=_pocket_union, top_cut_depth_m=_mh_safe_top,
                    )
                    if _new_t is not None:
                        terrain_mesh = _new_t
                        if _new_b is not None:
                            keychain_base_bottom_mesh = _new_b
                if _hl_meshes:
                    highlight_building_mesh = (
                        trimesh.util.concatenate(_hl_meshes) if len(_hl_meshes) > 1 else _hl_meshes[0]
                    )
                    _with_color(highlight_building_mesh, LAYER_COLORS["highlight"])
                print(f"[MAP HIGHLIGHT] {len(_chosen)} building(s) → red 'Highlight' part"
                      + (f" + {len(_pockets)} pocket(s) (механічна вставка)" if _pockets else " (glue-on)"))
        except Exception as _mhexc:
            print(f"[MAP HIGHLIGHT] highlight building failed (non-fatal): {_mhexc}")

    # МАРКЕР «особливе місце»: піднята фігурка (heart/star/circle) у ЦЕНТРІ карти
    # (= точка, яку шукав користувач, бо мапа заповнює все тіло). Окремий шар.
    keychain_marker_mesh = None
    _mk = str(getattr(request, "keychain_place_marker", "") or "").lower().strip()
    if keychain_mode and _mk and _mk not in ("none", "off") and keychain_layout is not None:
        try:
            _mk_shape = {"dot": "circle", "pin": "circle", "round": "circle"}.get(_mk, _mk)
            if _mk_shape not in ("heart", "star", "circle"):
                _mk_shape = "heart"
            _mk_size = float(getattr(request, "keychain_place_marker_size_mm", 6.0) or 6.0)
            _mk_half = _model_mm_to_world_m(_mk_size / 2.0, export_scale_factor)
            _bodyc = keychain_layout["body"].centroid
            _mk_poly = _keychain_body_shape(
                _bodyc.x - _mk_half, _bodyc.y - _mk_half, _bodyc.x + _mk_half, _bodyc.y + _mk_half,
                radius_m=_mk_half * 0.3, shape=_mk_shape,
            )
            if _mk_poly is not None and not _mk_poly.is_empty:
                # Дно на base_top, висота 1.5мм → стоїть НАД мапою (дороги/будівлі ~0.8мм).
                keychain_marker_mesh = build_flat_layer_mesh_from_mask(
                    _mk_poly, bottom_z_m=base_top_m,
                    thickness_m=_model_mm_to_world_m(1.5, export_scale_factor),
                    color=LAYER_COLORS["marker"], min_area_m2=1e-9,
                )
                if keychain_marker_mesh is not None:
                    print(f"[KEYCHAIN] Place marker '{_mk_shape}' {_mk_size:.1f}mm at map center")
        except Exception as _mkexc:
            print(f"[KEYCHAIN] place marker failed (non-fatal): {_mkexc}")

    if keychain_layout is not None:
        layout_rotation_deg = float(getattr(request, "keychain_layout_rotation_deg", 0.0) or 0.0)
        if layout_rotation_deg:
            try:
                minx, miny, maxx, maxy = keychain_layout["body"].bounds
                _rotate_meshes_for_keychain_layout(
                    meshes=[terrain_mesh, road_mesh, water_mesh, parks_mesh, keychain_rim_mesh, keychain_text_mesh, keychain_text2_mesh, keychain_base_bottom_mesh, keychain_marker_mesh, highlight_building_mesh] + landmark_meshes,
                    building_meshes=building_meshes,
                    angle_deg=layout_rotation_deg,
                    origin_xy=((minx + maxx) * 0.5, (miny + maxy) * 0.5),
                    origin_z=0.0,
                )
            except Exception as exc:
                print(f"[KEYCHAIN] Layout rotation skipped: {exc}")

    print(
        f"[{'KEYCHAIN' if keychain_mode else 'FLAT PLATE'}] Built layered plate: "
        f"base={'OK' if terrain_mesh is not None else 'None'}, "
        f"water={'OK' if water_mesh is not None else 'None'}, "
        f"roads={'OK' if road_mesh is not None else 'None'}, "
        f"parks={'OK' if parks_mesh is not None else 'None'}, "
        f"buildings={len(building_meshes)}, "
        f"rim={'OK' if keychain_rim_mesh is not None else 'None'}, "
        f"text={'OK' if keychain_text_mesh is not None else 'None'}"
    )
    print(
        f"[{'KEYCHAIN' if keychain_mode else 'FLAT PLATE'}] Layer tops: "
        f"base={base_thickness_mm:.2f}mm, "
        f"water={base_thickness_mm + water_layer_mm:.2f}mm, "
        f"roads={base_thickness_mm + roads_layer_mm:.2f}mm, "
        f"parks={base_thickness_mm + parks_layer_mm:.2f}mm"
    )
    combined_landmarks = None
    if landmark_meshes:
        try:
            combined_landmarks = trimesh.util.concatenate([m for m in landmark_meshes if m is not None])
        except Exception:
            combined_landmarks = None
    if keychain_mode:
        try:
            combined_buildings = (
                trimesh.util.concatenate([mesh for mesh in building_meshes if mesh is not None])
                if building_meshes
                else None
            )
        except Exception:
            combined_buildings = None
        layer_manifest = {
            "mode": "keychain",
            "print_rules": {
                "minimum_feature_mm": MIN_KEYCHAIN_PRINT_FEATURE_MM,
                "text_is_separate_layer": keychain_text_mesh is not None,
                "rim_is_separate_layer": keychain_rim_mesh is not None,
                "map_clipped_to_inner_rim": keychain_layout is not None,
                "roads_buildings_precedence": True,
            },
            "dimensions": {
                "body_width_mm": float(getattr(request, "keychain_body_width_mm", 0.0) or 0.0),
                "body_height_mm": float(getattr(request, "keychain_body_height_mm", 0.0) or 0.0),
                "map_width_mm": float(getattr(request, "keychain_map_width_mm", 0.0) or 0.0),
                "map_height_mm": float(getattr(request, "keychain_map_height_mm", 0.0) or 0.0),
                "map_rotation_deg": float(getattr(request, "keychain_map_rotation_deg", 0.0) or 0.0),
            },
            "layers": {
                "base": _mesh_manifest(terrain_mesh, scale_factor=export_scale_factor),
                "rim": _mesh_manifest(keychain_rim_mesh, scale_factor=export_scale_factor),
                "water": _mesh_manifest(water_mesh, scale_factor=export_scale_factor),
                "parks": _mesh_manifest(parks_mesh, scale_factor=export_scale_factor),
                "roads": _mesh_manifest(road_mesh, scale_factor=export_scale_factor),
                "buildings": _mesh_manifest(combined_buildings, scale_factor=export_scale_factor),
                "landmark": _mesh_manifest(combined_landmarks, scale_factor=export_scale_factor),
                "text": _mesh_manifest(keychain_text_mesh, scale_factor=export_scale_factor),
            },
        }
        try:
            setattr(task, "keychain_manifest", layer_manifest)
        except Exception:
            pass

    return export_generation_outputs(
        task=task,
        request=request,
        task_id=task_id,
        output_dir=output_dir,
        terrain_mesh=terrain_mesh,
        road_mesh=road_mesh,
        building_meshes=building_meshes,
        water_mesh=water_mesh,
        parks_mesh=parks_mesh,
        extra_mesh_items=[
            item for item in (
                ("Rim", keychain_rim_mesh),
                ("Text", keychain_text_mesh),
                ("Text2", keychain_text2_mesh),
                ("Marker", keychain_marker_mesh),
                ("Highlight", highlight_building_mesh),
                ("Landmark", combined_landmarks),
                ("Connector", connector_mesh),
                ("Frame", map_frame_mesh),
                ("BaseBack", keychain_base_bottom_mesh),
                ("MapLabel", map_text_mesh),
                ("Bridges", locals().get("bridge_mesh") if keychain_mode else None),
                ("WaterBase", locals().get("water_plug_mesh") if keychain_mode else None),
                ("Track", locals().get("gpx_track_mesh")),
            )
            if item[1] is not None
        ],
        reference_xy_m=keychain_layout["body_reference_xy_m"] if keychain_layout else zone.reference_xy_m,
        preserve_z=False,
        preserve_xy=False,
        include_preview_parts=False,
        include_parallel_stl=False,
        include_print_package=False,
        completion_message="Пласка layered plate модель готова!",
        file_basename=file_basename,
        repair_meshes=not keychain_mode,
    )
