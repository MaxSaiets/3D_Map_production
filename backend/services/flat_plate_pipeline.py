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
}


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
    if shape_name == "capsule":
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
    loop_outer_radius_mm: float,
    loop_inner_radius_mm: float,
    corner_radius_mm: float,
    label_band_height_mm: float,
) -> dict[str, BaseGeometry]:
    minx, miny, maxx, maxy = bbox_meters
    source_w = max(float(maxx - minx), 1e-6)
    source_h = max(float(maxy - miny), 1e-6)
    body_w_mm = max(float(body_width_mm or model_size_mm or 78.0), 24.0)
    body_h_mm = max(float(body_height_mm or (body_w_mm * source_h / max(source_w, 1e-6))), 18.0)
    map_w_mm = max(float(map_width_mm or body_w_mm), 4.0)
    map_h_mm = max(float(map_height_mm or max(body_h_mm - label_band_height_mm, 4.0)), 4.0)
    layout_scale_m_per_mm = max(source_w / map_w_mm, source_h / map_h_mm)
    export_scale = 1.0 / layout_scale_m_per_mm

    body_minx = 0.0
    body_miny = 0.0
    body_maxx = body_w_mm * layout_scale_m_per_mm
    body_maxy = body_h_mm * layout_scale_m_per_mm
    outer_m = max(loop_outer_radius_mm, 4.0) * layout_scale_m_per_mm
    inner_m = min(max(loop_inner_radius_mm, 1.6), max(loop_outer_radius_mm - 1.8, 1.6)) * layout_scale_m_per_mm
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
    loop_center_x = float(loop_center_x_mm if loop_center_x_mm is not None else (loop_outer_radius_mm + 3.0)) * layout_scale_m_per_mm
    loop_center_y = body_maxy - float(loop_center_y_mm if loop_center_y_mm is not None else -loop_outer_radius_mm * 0.45) * layout_scale_m_per_mm
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

    label_center_x = float(label_center_x_mm if label_center_x_mm is not None else body_w_mm / 2.0) * layout_scale_m_per_mm
    label_center_y = body_maxy - float(label_center_y_mm if label_center_y_mm is not None else (body_h_mm - label_band_height_mm / 2.0)) * layout_scale_m_per_mm
    label_w = body_w_mm * 0.86 * layout_scale_m_per_mm
    label_h = max(label_band_h_m, 1e-6)
    label_band = box(
        max(body_minx, label_center_x - label_w / 2.0),
        max(body_miny, label_center_y - label_h / 2.0),
        min(body_maxx, label_center_x + label_w / 2.0),
        min(body_maxy, label_center_y + label_h / 2.0),
    )
    content_area = box(map_minx, map_miny, map_maxx, map_maxy).intersection(body)
    try:
        content_area = content_area.buffer(0)
    except Exception:
        pass
    return {
        "base": base,
        "body": body,
        "content_area": content_area,
        "label_band": label_band,
        "loop_hole": inner_hole,
        "source_bbox": box(minx, miny, maxx, maxy),
        "body_reference_xy_m": (body_maxx - body_minx, body_maxy - body_miny),
        "map_target_bounds": (map_minx, map_miny, map_maxx, map_maxy),
        "layout_scale_m_per_mm": layout_scale_m_per_mm,
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


def build_keychain_label_mesh(
    text: str,
    *,
    body_geometry: BaseGeometry,
    label_band_geometry: BaseGeometry,
    bottom_z_m: float,
    thickness_m: float,
    text_height_m: float,
    color: list[int],
    angle_deg: float = 0.0,
) -> Optional[trimesh.Trimesh]:
    label = _normalize_label_text(text)
    if not label or thickness_m <= 0 or text_height_m <= 0:
        return None
    band_minx, band_miny, band_maxx, band_maxy = label_band_geometry.bounds
    max_width = max((band_maxx - band_minx) * 0.86, 1e-6)
    cell = text_height_m / 7.0
    char_units = [6 if ch != " " else 3 for ch in label]
    raw_width = max(sum(char_units) - 1, 1) * cell
    if raw_width > max_width:
        cell *= max_width / raw_width
    raw_width = max(sum(char_units) - 1, 1) * cell
    start_x = (band_minx + band_maxx - raw_width) / 2.0
    start_y = band_miny + max((band_maxy - band_miny) - 7.0 * cell, 0.0) / 2.0

    glyph_meshes: list[trimesh.Trimesh] = []
    cursor_x = start_x
    for ch in label:
        glyph = _FONT_5X7.get(ch, _FONT_5X7[" "])
        for row_idx, row in enumerate(glyph):
            for col_idx, bit in enumerate(row):
                if bit != "1":
                    continue
                x0 = cursor_x + col_idx * cell
                y0 = start_y + (6 - row_idx) * cell
                pixel = box(x0, y0, x0 + cell * 0.82, y0 + cell * 0.82)
                mesh = build_flat_layer_mesh_from_mask(
                    pixel,
                    bottom_z_m=bottom_z_m,
                    thickness_m=thickness_m,
                    color=color,
                    min_area_m2=1e-12,
                )
                if mesh is not None:
                    glyph_meshes.append(mesh)
        cursor_x += (6 if ch != " " else 3) * cell
    if not glyph_meshes:
        return None
    try:
        combined = trimesh.util.concatenate(glyph_meshes)
    except Exception:
        combined = glyph_meshes[0]
    if angle_deg:
        try:
            cx = (band_minx + band_maxx) * 0.5
            cy = (band_miny + band_maxy) * 0.5
            matrix = trimesh.transformations.rotation_matrix(
                np.deg2rad(float(angle_deg)),
                [0.0, 0.0, 1.0],
                [cx, cy, bottom_z_m],
            )
            combined.apply_transform(matrix)
        except Exception:
            pass
    return _with_color(combined, color)


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
    for record in records:
        mesh = getattr(record, "mesh", None)
        if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
            continue
        mesh = mesh.copy()
        if max_building_height_m > 0:
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

    task.update_status("processing", 55, "Генерація пласких шарів...")
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

    base_top_m = _model_mm_to_world_m(base_thickness_mm, export_scale_factor)
    content_area = zone.zone_polygon_local
    keychain_layout: Optional[dict[str, BaseGeometry]] = None
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
            loop_outer_radius_mm=float(getattr(request, "keychain_loop_outer_radius_mm", 6.5) or 6.5),
            loop_inner_radius_mm=float(getattr(request, "keychain_loop_inner_radius_mm", 3.0) or 3.0),
            corner_radius_mm=float(getattr(request, "keychain_corner_radius_mm", 4.0) or 4.0),
            label_band_height_mm=float(getattr(request, "keychain_label_band_height_mm", 9.0) or 9.0),
        )
        content_area = keychain_layout["content_area"]
        export_scale_factor = 1.0 / max(float(keychain_layout["layout_scale_m_per_mm"]), 1e-9)
        base_top_m = _model_mm_to_world_m(base_thickness_mm, export_scale_factor)
        source_bounds = tuple(float(v) for v in keychain_layout["source_bbox"].bounds)
        target_bounds = tuple(float(v) for v in keychain_layout["map_target_bounds"])

    terrain_mesh = build_flat_zone_base_mesh(
        keychain_layout["base"] if keychain_layout else zone.zone_polygon_local,
        bbox_meters=zone.bbox_meters,
        thickness_m=base_top_m,
    )

    bundle = canonical_2d_stage.canonical_mask_bundle
    min_area_m2 = max((_model_mm_to_world_m(0.15, scale_factor) ** 2), 1e-6)
    water_mesh = build_flat_layer_mesh_from_mask(
        _clip_geometry(
            _fit_geometry_into_bounds(getattr(bundle, "water_final", None), source_bounds=source_bounds, target_bounds=target_bounds)
            if keychain_layout else getattr(bundle, "water_final", None),
            content_area,
        ),
        bottom_z_m=base_top_m,
        thickness_m=_model_mm_to_world_m(water_layer_mm, export_scale_factor),
        color=LAYER_COLORS["water"],
        min_area_m2=min_area_m2,
    )
    road_mesh = build_flat_layer_mesh_from_mask(
        _clip_geometry(
            _fit_geometry_into_bounds(getattr(bundle, "roads_final", None), source_bounds=source_bounds, target_bounds=target_bounds)
            if keychain_layout else getattr(bundle, "roads_final", None),
            content_area,
        ),
        bottom_z_m=base_top_m,
        thickness_m=_model_mm_to_world_m(roads_layer_mm, export_scale_factor),
        color=LAYER_COLORS["roads"],
        min_area_m2=min_area_m2,
    )
    parks_mesh = build_flat_layer_mesh_from_mask(
        _clip_geometry(
            _fit_geometry_into_bounds(getattr(bundle, "parks_final", None), source_bounds=source_bounds, target_bounds=target_bounds)
            if keychain_layout else getattr(bundle, "parks_final", None),
            content_area,
        ),
        bottom_z_m=base_top_m,
        thickness_m=_model_mm_to_world_m(parks_layer_mm, export_scale_factor),
        color=LAYER_COLORS["parks"],
        min_area_m2=min_area_m2,
    )

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

    if keychain_mode:
        gdf_buildings_local = _fit_gdf_into_bounds(
            gdf_buildings_local,
            source_bounds=source_bounds,
            target_bounds=target_bounds,
        )
        gdf_buildings_local = _clip_buildings_to_content(gdf_buildings_local, content_area)

    building_meshes = build_flat_building_meshes(
        request=request,
        scale_factor=scale_factor,
        export_scale_factor=export_scale_factor,
        gdf_buildings_local=gdf_buildings_local,
        base_top_m=base_top_m,
    )
    if keychain_layout is not None:
        label_mesh = build_keychain_label_mesh(
            str(getattr(request, "keychain_label", "") or ""),
            body_geometry=keychain_layout["body"],
            label_band_geometry=keychain_layout["label_band"],
            bottom_z_m=base_top_m,
            thickness_m=_model_mm_to_world_m(float(getattr(request, "keychain_label_raise_mm", 0.45) or 0.45), export_scale_factor),
            text_height_m=_model_mm_to_world_m(float(getattr(request, "keychain_label_text_height_mm", 3.8) or 3.8), export_scale_factor),
            color=LAYER_COLORS["buildings"],
            angle_deg=float(getattr(request, "keychain_label_angle_deg", 0.0) or 0.0),
        )
        if label_mesh is not None:
            building_meshes.append(label_mesh)

    print(
        f"[{'KEYCHAIN' if keychain_mode else 'FLAT PLATE'}] Built layered plate: "
        f"base={'OK' if terrain_mesh is not None else 'None'}, "
        f"water={'OK' if water_mesh is not None else 'None'}, "
        f"roads={'OK' if road_mesh is not None else 'None'}, "
        f"parks={'OK' if parks_mesh is not None else 'None'}, "
        f"buildings={len(building_meshes)}"
    )
    print(
        f"[{'KEYCHAIN' if keychain_mode else 'FLAT PLATE'}] Layer tops: "
        f"base={base_thickness_mm:.2f}mm, "
        f"water={base_thickness_mm + water_layer_mm:.2f}mm, "
        f"roads={base_thickness_mm + roads_layer_mm:.2f}mm, "
        f"parks={base_thickness_mm + parks_layer_mm:.2f}mm"
    )

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
        reference_xy_m=keychain_layout["body_reference_xy_m"] if keychain_layout else zone.reference_xy_m,
        preserve_z=False,
        preserve_xy=False,
        include_preview_parts=False,
        include_parallel_stl=False,
        include_print_package=False,
        completion_message="Пласка layered plate модель готова!",
        file_basename=file_basename,
    )
