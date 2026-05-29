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
    "base": [200, 180, 140, 255],
    "roads": [60, 60, 60, 255],
    "buildings": [225, 225, 225, 255],
    "water": [100, 150, 200, 255],
    "parks": [100, 150, 100, 255],
    "rim": [92, 80, 58, 255],
    "text": [245, 245, 238, 255],
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


def _keychain_body_shape(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    *,
    radius_m: float,
    shape: str,
) -> BaseGeometry:
    shape_name = (shape or "rounded").lower().replace("_", "-")
    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)
    if shape_name in {"capsule", "token"}:
        return _rounded_rect(minx, miny, maxx, maxy, height / 2.0)
    if shape_name == "tag":
        cut = min(width, height) * 0.16
        points = [
            (minx, miny + radius_m),
            (minx + radius_m, miny),
            (maxx - cut, miny),
            (maxx, miny + cut),
            (maxx, maxy - radius_m),
            (maxx - radius_m, maxy),
            (minx + radius_m, maxy),
            (minx, maxy - radius_m),
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

    body = _keychain_body_shape(body_minx, body_miny, body_maxx, body_maxy, radius_m=corner_m, shape=base_shape)
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
    outer_loop = _keychain_loop_outer(center_x=loop_center_x, center_y=loop_center_y, outer_m=outer_m, style=loop_style)
    inner_hole = _keychain_loop_inner(center_x=loop_center_x, center_y=loop_center_y, inner_m=inner_m, style=loop_style)
    neck_half = max((outer_m - inner_m) * 0.72, _model_mm_to_world_m(1.8, export_scale))
    try:
        body_point, loop_point = nearest_points(body, Point(loop_center_x, loop_center_y))
        neck_line = LineString([body_point, loop_point])
        neck = neck_line.buffer(neck_half, cap_style=1, join_style=1, resolution=10)
    except Exception:
        neck = box(loop_center_x - neck_half, body_maxy - corner_m * 0.45, loop_center_x + neck_half, loop_center_y).buffer(
            neck_half * 0.55,
            resolution=10,
        )
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
    # ВРАХОВУЄМО ОБЕРТАННЯ: angle 90°/270° → band стає VERTICAL (swap dims).
    # Інакше для token mode з rotated текстом band був горизонтальний 20×5,
    # а текст потребував вертикальний 5×20 — gli'fs не помістились.
    angle_normalized = abs(float(label_angle_deg or 0.0) % 180.0)
    is_perpendicular = 45.0 < angle_normalized < 135.0
    if is_perpendicular:
        # Swap: текст вертикальний → band вертикальний (висота=label_w, ширина=label_h)
        label_w, label_h = label_h, label_w
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
                label_clear_band, float(label_angle_deg),
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
) -> Optional[trimesh.Trimesh]:
    label = _normalize_label_text(text)
    if not label or thickness_m <= 0 or text_height_m <= 0:
        return None
    band_minx, band_miny, band_maxx, band_maxy = label_band_geometry.bounds
    max_width = max((band_maxx - band_minx) * 0.96, 1e-6)
    max_height = max((band_maxy - band_miny) * 0.92, 1e-6)

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
                    # Не дилейтимо надміру: cap на 0.12 × cap height щоб counters не закрились.
                    grow = min(max(grow, 0.0), 0.12 * float(text_height_m))
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
            # Поворот
            if angle_deg:
                text_polygon = affinity.rotate(text_polygon, float(angle_deg), origin=(cx, cy), use_radians=False)
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
        color=LAYER_COLORS["base"],
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


def build_flat_building_meshes(
    *,
    request: Any,
    scale_factor: float,
    gdf_buildings_local: Optional[GeoDataFrame],
    base_top_m: float,
    export_scale_factor: Optional[float] = None,
) -> list[trimesh.Trimesh]:
    if gdf_buildings_local is None or gdf_buildings_local.empty:
        return []
    if not getattr(request, "include_buildings", True):
        return []

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
                sorted_idx = areas.sort_values(ascending=False).index[:max_buildings]
                original_count = len(gdf_buildings_for_mesh)
                gdf_buildings_for_mesh = gdf_buildings_for_mesh.loc[sorted_idx].copy()
                print(
                    f"[MEMORY-GUARD] Reduced buildings from {original_count} to "
                    f"{len(gdf_buildings_for_mesh)} (top-{max_buildings} by area, "
                    f"{'keychain' if is_keychain else 'flat'} mode)"
                )
            except Exception as e:
                print(f"[MEMORY-GUARD] Building filter failed: {e}; proceeding with full set")
    height_scale_factor = float(
        getattr(request, "buildings_height_scale", None)
        or getattr(request, "building_height_multiplier", 1.0)
    )
    requested_min_height_m = float(getattr(request, "building_min_height", 2.0) or 2.0)
    printable_min_height_m = _model_mm_to_world_m(0.8, float(scale_factor))
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
    is_keychain = bool(getattr(request, "keychain_mode", False))
    for record in records:
        mesh = getattr(record, "mesh", None)
        if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
            continue
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
            mesh = _clamp_mesh_height(mesh, min_height_m=min_building_height_m, max_height_m=max_building_height_m)
        mesh.apply_translation([0.0, 0.0, float(base_top_m)])
        _with_color(mesh, LAYER_COLORS["buildings"])
        meshes.append(mesh)
    return meshes


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

    base_top_m = _model_mm_to_world_m(base_thickness_mm, export_scale_factor)
    content_area = zone.zone_polygon_local
    keychain_layout: Optional[dict[str, BaseGeometry]] = None
    keychain_rim_mesh: Optional[trimesh.Trimesh] = None
    keychain_text_mesh: Optional[trimesh.Trimesh] = None
    source_bounds: Optional[tuple[float, float, float, float]] = None
    target_bounds: Optional[tuple[float, float, float, float]] = None
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

    terrain_mesh = build_flat_zone_base_mesh(
        keychain_layout["base"] if keychain_layout else zone.zone_polygon_local,
        bbox_meters=zone.bbox_meters,
        thickness_m=base_top_m,
    )
    if keychain_layout is not None and terrain_mesh is not None:
        rim_width_mm = float(getattr(request, "keychain_rim_width_mm", 0.0) or 0.0)
        rim_height_mm = float(getattr(request, "keychain_rim_height_mm", 0.0) or 0.0)
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

    # ОЧИЩЕННЯ КАРТИ ПІД ТЕКСТОМ: вирізаємо орієнтований label_clear_band з усіх
    # map-шарів, щоб напис стояв на ЧИСТОМУ фоні (юзер: «обрізається карта вокруг
    # текста»). Band слідує за позицією/кутом напису → коректно при move/rotate.
    label_clear_band = None
    if keychain_mode and keychain_layout is not None:
        label_clear_band = keychain_layout.get("label_clear_band")
        if label_clear_band is not None and not getattr(label_clear_band, "is_empty", True):
            try:
                if road_mask is not None and not getattr(road_mask, "is_empty", True):
                    road_mask = _subtract_geometry(road_mask, label_clear_band)
                if parks_mask is not None and not getattr(parks_mask, "is_empty", True):
                    parks_mask = _subtract_geometry(parks_mask, label_clear_band)
                if water_mask is not None and not getattr(water_mask, "is_empty", True):
                    water_mask = _subtract_geometry(water_mask, label_clear_band)
                print("[KEYCHAIN] Map layers cleared under label band (clean text background)")
            except Exception as exc:
                print(f"[KEYCHAIN] Label band map-clear failed: {exc}")

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

    if keychain_mode:
        try: task.update_status("processing", 80, "Будую 3D будівлі з висотами OSM...")
        except Exception: pass
    building_meshes = build_flat_building_meshes(
        request=request,
        scale_factor=scale_factor,
        export_scale_factor=export_scale_factor,
        gdf_buildings_local=gdf_buildings_local,
        base_top_m=base_top_m,
    )
    if keychain_layout is not None:
        # RAISED TEXT — текст підіймається над базою як рельєф (попередня логіка).
        text_raise_mm = float(getattr(request, "keychain_label_raise_mm", KEYCHAIN_TEXT_RAISE_MM) or KEYCHAIN_TEXT_RAISE_MM)
        text_raise_m = _model_mm_to_world_m(text_raise_mm, export_scale_factor)
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
        )
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
                    new_terrain = build_flat_layer_mesh_from_mask(
                        base_poly_combined,
                        bottom_z_m=0.0,
                        thickness_m=base_top_m,
                        color=LAYER_COLORS["base"],
                        min_area_m2=0.001,
                    )
                    if new_terrain is not None:
                        terrain_mesh = new_terrain
                        print(f"[KEYCHAIN] Base rebuilt with water depression ({water_layer_mm:.2f}mm)")
        except Exception as exc:
            print(f"[KEYCHAIN] Water carve failed: {exc}")

    if keychain_layout is not None:
        layout_rotation_deg = float(getattr(request, "keychain_layout_rotation_deg", 0.0) or 0.0)
        if layout_rotation_deg:
            try:
                minx, miny, maxx, maxy = keychain_layout["body"].bounds
                _rotate_meshes_for_keychain_layout(
                    meshes=[terrain_mesh, road_mesh, water_mesh, parks_mesh, keychain_rim_mesh, keychain_text_mesh],
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
                ("Bridges", locals().get("bridge_mesh") if keychain_mode else None),
                ("WaterBase", locals().get("water_plug_mesh") if keychain_mode else None),
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
