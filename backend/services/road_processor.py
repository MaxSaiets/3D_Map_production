"""
РЎРµСЂРІС–СЃ РґР»СЏ РѕР±СЂРѕР±РєРё РґРѕСЂС–Рі Р· Р±СѓС„РµСЂРёР·Р°С†С–С”СЋ С‚Р° РѕР±'С”РґРЅР°РЅРЅСЏРј
РџРѕРєСЂР°С‰РµРЅР° РІРµСЂСЃС–СЏ Р· С„С–Р·РёС‡РЅРѕСЋ С€РёСЂРёРЅРѕСЋ РґРѕСЂС–Рі С‚Р° РїС–РґС‚СЂРёРјРєРѕСЋ РјРѕСЃС‚С–РІ
Р’РёРєРѕСЂРёСЃС‚РѕРІСѓС” trimesh.creation.extrude_polygon РґР»СЏ РЅР°РґС–Р№РЅРѕС— С‚СЂС–Р°РЅРіСѓР»СЏС†С–С—
"""
import ast
import os
import osmnx as ox
import json
import trimesh
import numpy as np
import warnings
from shapely.ops import unary_union, transform, snap
from shapely.geometry import Polygon, MultiPolygon, box, LineString, Point
from shapely.geometry import mapping, shape
from typing import Optional, List, Tuple
import geopandas as gpd
from services.terrain_provider import TerrainProvider
from services.global_center import GlobalCenter
from services.building_supports import union_mesh_collection
from services.detail_layer_utils import MICRO_REGION_THRESHOLD_MM, MIN_ROAD_WIDTH_MODEL_MM, model_mm_to_world_m
from services.geometry_context import clean_geometry, looks_like_projected_meters, make_to_local_transformer, to_local_geometry_if_needed
from services.mesh_triangulation import extrude_polygon_grid, extrude_polygon_uniform, refine_mesh_long_edges
from services.mesh_quality import improve_mesh_for_3d_printing, validate_mesh_for_3d_printing
from services.processing_results import RoadProcessingResult
from scipy.spatial import cKDTree

# РџСЂРёРґСѓС€РµРЅРЅСЏ deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pandas')

NON_PRINTABLE_HIGHWAY_TYPES = {
    "steps",
    "corridor",
    "bridleway",
}

PRINTABLE_HIGHWAY_PRIORITY = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "residential",
    "living_street",
    "service",
    "unclassified",
    "pedestrian",
    "cycleway",
    "footway",
    "path",
    "steps",
]

# Canonical road masks should be driven only by vehicular / street network,
# not by sidewalks, footpaths, cycle-only tracks, or tiny park circulation.
# Keep all street-like road classes that can appear as solid roads on the
# source map, including local access / service streets.
DRIVABLE_HIGHWAY_PRIORITY = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "residential",
    "unclassified",
    "living_street",
    "service",
    "pedestrian",
]


def _extract_numeric_token(raw_value) -> float | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, (int, float)):
        try:
            value = float(raw_value)
        except Exception:
            return None
        return value if value > 0 else None
    text = str(raw_value).strip().lower()
    if not text:
        return None
    text = text.replace(",", ".")
    for chunk in text.replace(";", "|").split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        number = []
        dot_seen = False
        for ch in chunk:
            if ch.isdigit():
                number.append(ch)
            elif ch == "." and not dot_seen:
                number.append(ch)
                dot_seen = True
            elif number:
                break
        if not number:
            continue
        try:
            value = float("".join(number))
        except Exception:
            continue
        if value > 0:
            return value
    return None


def _resolve_osm_road_width_m(row, fallback_width_m: float) -> float:
    # Map-faithful width policy:
    # 1) take explicit OSM width when present,
    # 2) but clamp it close to class fallback to prevent oversized road blobs.
    # We intentionally avoid lanes/parking/estimation heuristics.
    width_m = _extract_numeric_token(row.get("width"))
    if width_m is not None and 0.8 <= float(width_m) <= 80.0:
        upper = max(float(fallback_width_m) * 1.15, float(fallback_width_m) + 0.35)
        return float(min(max(float(width_m), 0.8), upper))
    return float(fallback_width_m)


def normalize_highway_tag(raw_highway):
    values = []
    if isinstance(raw_highway, str):
        raw_text = raw_highway.strip()
        if raw_text.startswith("[") or raw_text.startswith("(") or raw_text.startswith("{"):
            try:
                parsed = ast.literal_eval(raw_text)
            except Exception:
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                raw_highway = parsed
            elif raw_text:
                raw_highway = [raw_text]
        elif ";" in raw_text:
            raw_highway = [part.strip() for part in raw_text.split(";") if part and part.strip()]
        else:
            raw_highway = [raw_text]
    if isinstance(raw_highway, (list, tuple, set)):
        values = [str(v).lower() for v in raw_highway if v]
    elif raw_highway:
        values = [str(raw_highway).lower()]
    if not values:
        return None
    unique_values = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
    printable_values = [value for value in unique_values if value not in NON_PRINTABLE_HIGHWAY_TYPES]
    if not printable_values:
        return None
    for highway_type in PRINTABLE_HIGHWAY_PRIORITY:
        if highway_type in printable_values:
            return highway_type
    return printable_values[0]


def normalize_drivable_highway_tag(raw_highway):
    normalized = normalize_highway_tag(raw_highway)
    if normalized in DRIVABLE_HIGHWAY_PRIORITY:
        return normalized
    return None


def _write_debug_geometry_geojson(name: str, geometry) -> None:
    if geometry is None or getattr(geometry, "is_empty", True):
        return
    try:
        backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        debug_dir = os.path.join(backend_dir, "debug", "generated", "road_masks")
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, f"{name}.geojson")
        feature = {
            "type": "Feature",
            "properties": {"name": name},
            "geometry": mapping(geometry),
        }
        payload = {"type": "FeatureCollection", "features": [feature]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass


def _rebuild_road_geometry(geometry):
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    try:
        polygons = []
        if isinstance(geometry, Polygon):
            polygons = [geometry]
        elif isinstance(geometry, MultiPolygon):
            polygons = [poly for poly in geometry.geoms if poly is not None and not poly.is_empty]
        elif hasattr(geometry, "geoms"):
            polygons = [poly for poly in geometry.geoms if poly is not None and not poly.is_empty]
        rebuilt = [shape(mapping(poly)) for poly in polygons if poly is not None and not poly.is_empty]
        if not rebuilt:
            return geometry
        return unary_union(rebuilt).buffer(0)
    except Exception:
        return geometry


def _clip_mesh_to_road_footprint(
    mesh: Optional[trimesh.Trimesh],
    road_polygon,
) -> Optional[trimesh.Trimesh]:
    if mesh is None or len(mesh.vertices) == 0 or road_polygon is None or getattr(road_polygon, "is_empty", True):
        return mesh
    try:
        road_polygon = _rebuild_road_geometry(road_polygon)
        if road_polygon is None or getattr(road_polygon, "is_empty", True):
            return mesh

        min_z = float(mesh.bounds[0][2]) - 2.0
        max_z = float(mesh.bounds[1][2]) + 2.0
        height = max(max_z - min_z, 1.0)
        road_polys = []
        if isinstance(road_polygon, Polygon):
            road_polys = [road_polygon]
        elif isinstance(road_polygon, MultiPolygon):
            road_polys = [poly for poly in road_polygon.geoms if poly is not None and not poly.is_empty]
        elif hasattr(road_polygon, "geoms"):
            road_polys = [poly for poly in road_polygon.geoms if poly is not None and not poly.is_empty]

        clip_parts = []
        for poly in road_polys:
            try:
                part = trimesh.creation.extrude_polygon(poly, height=height)
            except Exception:
                part = None
            if part is None or len(part.vertices) == 0:
                continue
            part.apply_translation([0.0, 0.0, min_z])
            clip_parts.append(part)
        if not clip_parts:
            return mesh
        clipper = trimesh.util.concatenate(clip_parts) if len(clip_parts) > 1 else clip_parts[0]

        original_mesh = mesh.copy()
        clipped = None
        try:
            import manifold3d

            def _to_manifold(input_mesh: trimesh.Trimesh) -> "manifold3d.Manifold":
                verts = np.asarray(input_mesh.vertices, dtype=np.float32)
                faces = np.asarray(input_mesh.faces, dtype=np.uint32)
                return manifold3d.Manifold(
                    manifold3d.Mesh(vert_properties=verts, tri_verts=faces)
                )

            clipped_manifold = manifold3d.Manifold.batch_boolean(
                [_to_manifold(mesh), _to_manifold(clipper)],
                manifold3d.OpType.Intersect,
            )
            if not clipped_manifold.is_empty():
                clipped_mesh = clipped_manifold.to_mesh()
                clipped = trimesh.Trimesh(
                    vertices=np.array(clipped_mesh.vert_properties, dtype=np.float64)[:, :3],
                    faces=np.array(clipped_mesh.tri_verts, dtype=np.int64),
                    process=False,
                )
        except Exception:
            clipped = trimesh.boolean.intersection([mesh, clipper], engine="manifold")

        if isinstance(clipped, list):
            clipped = trimesh.util.concatenate([m for m in clipped if m is not None and len(m.vertices) > 0])
        if clipped is not None and len(clipped.vertices) > 0 and len(clipped.faces) > 0:
            clipped.remove_unreferenced_vertices()
            clipped.fix_normals()
            try:
                clipped = _cleanup_road_mesh(clipped)
            except Exception:
                pass
            if _road_mesh_candidate_score(clipped, road_polygon) >= _road_mesh_candidate_score(original_mesh, road_polygon):
                return clipped
            return original_mesh
    except Exception as exc:
        print(f"[WARN] Road mesh footprint clipping failed: {exc}")
    return mesh


def _mesh_boundary_edge_count(mesh: Optional[trimesh.Trimesh]) -> int:
    if mesh is None or len(mesh.faces) == 0:
        return 0
    try:
        edges = mesh.edges_sorted
        if len(edges) == 0:
            return 0
        unique = trimesh.grouping.group_rows(edges, require_count=1)
        return int(len(unique))
    except Exception:
        return 0


def _cleanup_road_mesh(mesh: Optional[trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return mesh
    try:
        components = list(mesh.split(only_watertight=False))
    except Exception:
        components = [mesh]

    cleaned_components = []
    for component in components:
        if component is None or len(component.vertices) == 0 or len(component.faces) == 0:
            continue
        try:
            component = component.copy()
            if not component.is_volume:
                component.fill_holes()
            component.update_faces(component.unique_faces())
            component.update_faces(component.nondegenerate_faces())
            component.remove_unreferenced_vertices()
            component.merge_vertices(merge_tex=True, merge_norm=True)
            component.remove_unreferenced_vertices()
            component.fix_normals()
        except Exception:
            pass
        try:
            component_area = float(getattr(component, "area", 0.0) or 0.0)
            bounds = component.bounds
            extents = np.asarray(bounds[1] - bounds[0], dtype=float)
            max_xy = float(np.max(extents[:2])) if len(extents) >= 2 else 0.0
            if len(component.faces) <= 2 or component_area <= 1e-6 or max_xy <= 0.05:
                continue
        except Exception:
            pass
        if len(component.vertices) > 0 and len(component.faces) > 0:
            cleaned_components.append(component)

    if not cleaned_components:
        return mesh
    try:
        mesh = trimesh.util.concatenate(cleaned_components)
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def _mesh_projection_geometry(mesh: Optional[trimesh.Trimesh], *, top_only: bool) -> Optional[Polygon]:
    if mesh is None or len(mesh.faces) == 0:
        return None
    try:
        z_vals = mesh.vertices[:, 2]
        zmid = (float(np.min(z_vals)) + float(np.max(z_vals))) / 2.0
        polys = []
        for face in mesh.faces:
            tri3 = mesh.vertices[face]
            if top_only and float(np.mean(tri3[:, 2])) < zmid:
                continue
            poly = Polygon([(float(x), float(y)) for x, y, _ in tri3])
            if poly.is_empty or poly.area <= 1e-9:
                continue
            polys.append(poly)
        if not polys:
            return None
        geom = unary_union(polys)
        try:
            geom = geom.buffer(0)
        except Exception:
            pass
        return geom
    except Exception:
        return None


def _road_mesh_candidate_score(mesh: Optional[trimesh.Trimesh], road_polygon=None) -> tuple:
    if mesh is None or len(mesh.faces) == 0:
        return (-10**18, -10**18, -1, -1, -10**9, -1)
    top_gap_area = 0.0
    outside_area = 0.0
    if road_polygon is not None and not getattr(road_polygon, "is_empty", True):
        try:
            top_proj = _mesh_projection_geometry(mesh, top_only=True)
            if top_proj is not None and not getattr(top_proj, "is_empty", True):
                top_gap_area = float(road_polygon.difference(top_proj).area)
                outside_area = float(top_proj.difference(road_polygon).area)
            else:
                top_gap_area = float(getattr(road_polygon, "area", 0.0) or 0.0)
                outside_area = float(getattr(road_polygon, "area", 0.0) or 0.0)
        except Exception:
            top_gap_area = float(getattr(road_polygon, "area", 0.0) or 0.0)
            outside_area = float(getattr(road_polygon, "area", 0.0) or 0.0)
    return (
        -top_gap_area,
        -outside_area,
        1 if mesh.is_watertight else 0,
        1 if mesh.is_volume else 0,
        -_mesh_boundary_edge_count(mesh),
        int(len(mesh.faces)),
    )


def _road_mesh_topology_only_score(mesh: Optional[trimesh.Trimesh]) -> tuple:
    if mesh is None or len(mesh.faces) == 0:
        return (-1, -1, -10**9, -1)
    return (
        1 if mesh.is_watertight else 0,
        1 if mesh.is_volume else 0,
        -_mesh_boundary_edge_count(mesh),
        int(len(mesh.faces)),
    )


def create_deep_road_prism(
    polygon: Polygon,
    terrain_provider: TerrainProvider,
    scale_factor: float = 1.0,
    top_z_offset: float = 0.0,
    min_height: float = 1.0
) -> Optional[trimesh.Trimesh]:
    """
    РЎС‚РІРѕСЂСЋС” Р±Р»РѕРє РґРѕСЂРѕРіРё РґР»СЏ РІСЃС‚Р°РІРєРё (Inlay).
    Р”РЅРѕ СЂРѕР·СЂР°С…РѕРІСѓС”С‚СЊСЃСЏ Р»РѕРєР°Р»СЊРЅРѕ: min(surface_z) - 1РјРј.
    """
    if polygon is None or polygon.is_empty:
        return None

    try:
        # 1. Р•РєСЃС‚СЂСѓР·С–СЏ РєР°РЅРѕРЅС–С‡РЅРѕРіРѕ 2D РїРѕР»С–РіРѕРЅСѓ.
        # РќРµ РІС‚СЂР°С‡Р°С”РјРѕ РєСѓС‚Рё/Р·Р»Р°РјРё С‡РµСЂРµР· simplify, Р°Р»Рµ СЂРѕР±РёРјРѕ
        # С‚РѕРїРѕР»РѕРіС–С‡РЅРёР№ rebuild, С‰РѕР± extrude РЅРµ РїР°РґР°РІ РЅР° Р±СЂСѓРґРЅРёС… РєРѕРЅС‚СѓСЂР°С….
        polygon = _rebuild_road_geometry(polygon)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty: return None

        minx, miny, maxx, maxy = polygon.bounds
        max_dim = max(float(maxx - minx), float(maxy - miny))
        mesh = None
        candidates = []
        try:
            exact_mesh = trimesh.creation.extrude_polygon(polygon, height=1.0)
        except Exception:
            exact_mesh = None
        if exact_mesh is not None and len(exact_mesh.vertices) > 0 and len(exact_mesh.faces) > 0:
            candidates.append(exact_mesh)

        uniform_mesh = extrude_polygon_uniform(polygon, height=1.0, densify_max_m=2.0)
        if uniform_mesh is not None and len(uniform_mesh.vertices) > 0 and len(uniform_mesh.faces) > 0:
            candidates.append(uniform_mesh)

        grid_mesh = extrude_polygon_grid(
            polygon,
            height=1.0,
            target_edge_len_m=min(max(max_dim / 40.0, 3.0), 6.0),
            max_grid_points=12000,
        )
        if grid_mesh is not None and len(grid_mesh.vertices) > 0 and len(grid_mesh.faces) > 0:
            candidates.append(grid_mesh)

        if candidates:
            scored_candidates = []
            for candidate in candidates:
                candidate_mesh = candidate
                try:
                    candidate_mesh = _clip_mesh_to_road_footprint(candidate_mesh, polygon)
                except Exception:
                    pass
                try:
                    candidate_mesh = _cleanup_road_mesh(candidate_mesh)
                except Exception:
                    pass
                if candidate_mesh is None or len(candidate_mesh.vertices) == 0 or len(candidate_mesh.faces) == 0:
                    continue
                scored_candidates.append(candidate_mesh)
            if scored_candidates:
                mesh = max(scored_candidates, key=lambda m: _road_mesh_candidate_score(m, polygon))

        if mesh is None or len(mesh.vertices) == 0:
            return None

        vertices = mesh.vertices.copy()
        
        # 2. Р’РёР·РЅР°С‡Р°С”РјРѕ РІРµСЂС…РЅС– С‚Р° РЅРёР¶РЅС– РІРµСЂС€РёРЅРё
        z_vals = vertices[:, 2]
        z_center = (np.min(z_vals) + np.max(z_vals)) / 2.0
        
        top_mask = z_vals > z_center
        bottom_mask = ~top_mask
        
        # 3. РћР±С‡РёСЃР»СЋС”РјРѕ РІРёСЃРѕС‚Рё СЂРµР»СЊС”С„Сѓ
        xy_coords = vertices[:, :2]
        surface_z = terrain_provider.get_surface_heights_for_points(xy_coords)
        safe_z = np.nan_to_num(surface_z, nan=0.0)
        
        # 4. Р РѕР·СЂР°С…РѕРІСѓС”РјРѕ Р›РћРљРђР›Р¬РќР• РґРЅРѕ (С‚РѕРІС‰РёРЅР° 1РјРј Р°Р±Рѕ min_height)
        # 1РјРј Сѓ СЃРІС–С‚С– = 1.0 / scale_factor
        thickness_m = max(1.0 / scale_factor, 0.5 / scale_factor) 
        
        local_min_z = np.min(safe_z) if len(safe_z) > 0 else 0.0
        local_floor_z = local_min_z - thickness_m
        
        # 5. РњРѕРґРёС„С–РєСѓС”РјРѕ Z
        vertices[top_mask, 2] = safe_z[top_mask] + top_z_offset
        vertices[bottom_mask, 2] = local_floor_z
        
        # 6. РћРЅРѕРІР»СЋС”РјРѕ РјРµС€
        mesh.vertices = vertices
        mesh.fix_normals()
        
        return mesh

    except Exception as e:
        print(f"[WARN] Road prism generation failed: {e}")
        return None


def create_road_surface_cap(
    polygon: Polygon,
    terrain_provider: TerrainProvider,
    *,
    scale_factor: float = 1.0,
    top_z_offset: float = 0.0,
    cap_thickness_m: Optional[float] = None,
) -> Optional[trimesh.Trimesh]:
    if polygon is None or polygon.is_empty:
        return None

    try:
        polygon = _rebuild_road_geometry(polygon)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            return None

        mesh = extrude_polygon_uniform(polygon, height=1.0, densify_max_m=2.0)
        if mesh is None:
            mesh = trimesh.creation.extrude_polygon(polygon, height=1.0)
        if mesh is None or len(mesh.vertices) == 0:
            return None

        vertices = mesh.vertices.copy()
        z_vals = vertices[:, 2]
        z_center = (np.min(z_vals) + np.max(z_vals)) / 2.0
        top_mask = z_vals > z_center
        bottom_mask = ~top_mask

        xy_coords = vertices[:, :2]
        surface_z = terrain_provider.get_surface_heights_for_points(xy_coords)
        safe_z = np.nan_to_num(surface_z, nan=0.0)
        thickness_m = (
            float(cap_thickness_m)
            if cap_thickness_m is not None and cap_thickness_m > 0
            else max(0.25 / scale_factor, 0.12 / scale_factor)
        )

        vertices[top_mask, 2] = safe_z[top_mask] + top_z_offset
        vertices[bottom_mask, 2] = safe_z[bottom_mask] + top_z_offset - thickness_m
        mesh.vertices = vertices
        mesh.fix_normals()
        return mesh
    except Exception as exc:
        print(f"[WARN] Road cap generation failed: {exc}")
        return None


def create_bridge_supports(
    bridge_polygon: Polygon,
    bridge_height: float,
    terrain_provider: Optional[TerrainProvider],
    water_level: Optional[float],
    support_spacing: float = 20.0,  # Р’С–РґСЃС‚Р°РЅСЊ РјС–Р¶ РѕРїРѕСЂР°РјРё (РјРµС‚СЂРё)
    support_width: float = 2.0,  # РЁРёСЂРёРЅР° РѕРїРѕСЂРё (РјРµС‚СЂРё)
    min_support_height: float = 1.0,  # РњС–РЅС–РјР°Р»СЊРЅР° РІРёСЃРѕС‚Р° РѕРїРѕСЂРё (РјРµС‚СЂРё)
) -> List[trimesh.Trimesh]:
    """
    РЎС‚РІРѕСЂСЋС” РѕРїРѕСЂРё РґР»СЏ РјРѕСЃС‚Р°, СЏРєС– Р№РґСѓС‚СЊ РІС–Рґ РјРѕСЃС‚Р° РґРѕ Р·РµРјР»С–/РІРѕРґРё.
    Р¦Рµ РЅРµРѕР±С…С–РґРЅРѕ РґР»СЏ СЃС‚Р°Р±С–Р»СЊРЅРѕСЃС‚С– РїСЂРё 3D РґСЂСѓРєСѓ.
    
    Args:
        bridge_polygon: РџРѕР»С–РіРѕРЅ РјРѕСЃС‚Р°
        bridge_height: Р’РёСЃРѕС‚Р° РјРѕСЃС‚Р° (Z РєРѕРѕСЂРґРёРЅР°С‚Р°)
        terrain_provider: TerrainProvider РґР»СЏ РѕС‚СЂРёРјР°РЅРЅСЏ РІРёСЃРѕС‚ Р·РµРјР»С–
        water_level: Р С–РІРµРЅСЊ РІРѕРґРё РїС–Рґ РјРѕСЃС‚РѕРј (РѕРїС†С–РѕРЅР°Р»СЊРЅРѕ)
        support_spacing: Р’С–РґСЃС‚Р°РЅСЊ РјС–Р¶ РѕРїРѕСЂР°РјРё (РјРµС‚СЂРё)
        support_width: РЁРёСЂРёРЅР° РѕРїРѕСЂРё (РјРµС‚СЂРё)
        min_support_height: РњС–РЅС–РјР°Р»СЊРЅР° РІРёСЃРѕС‚Р° РѕРїРѕСЂРё (РјРµС‚СЂРё)
    
    Returns:
        РЎРїРёСЃРѕРє Trimesh РѕР±'С”РєС‚С–РІ РѕРїРѕСЂ
    """
    supports = []
    
    if bridge_polygon is None or terrain_provider is None:
        return supports
    
    try:
        # РћС‚СЂРёРјСѓС”РјРѕ С†РµРЅС‚СЂР°Р»СЊРЅСѓ Р»С–РЅС–СЋ РјРѕСЃС‚Р° (РґР»СЏ СЂРѕР·РјС–С‰РµРЅРЅСЏ РѕРїРѕСЂ)
        # Р’РёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ centroid С‚Р° bounds РґР»СЏ РІРёР·РЅР°С‡РµРЅРЅСЏ РЅР°РїСЂСЏРјРєСѓ РјРѕСЃС‚Р°
        bounds = bridge_polygon.bounds
        minx, miny, maxx, maxy = bounds
        center_x = (minx + maxx) / 2.0
        center_y = (miny + maxy) / 2.0
        
        # Р’РёР·РЅР°С‡Р°С”РјРѕ РЅР°РїСЂСЏРјРѕРє РјРѕСЃС‚Р° (РґРѕРІС€Р° СЃС‚РѕСЂРѕРЅР°)
        width = maxx - minx
        height = maxy - miny
        
        # РџРћРљР РђР©Р•РќРќРЇ: Р РѕР·РјС–С‰СѓС”РјРѕ РѕРїРѕСЂРё РїРѕ РєСЂР°СЏС… РјРѕСЃС‚Р° (РґР»СЏ СЃС‚Р°Р±С–Р»СЊРЅРѕСЃС‚С–) + С†РµРЅС‚СЂР°Р»СЊРЅС– РґР»СЏ РґРѕРІРіРёС… РјРѕСЃС‚С–РІ
        support_positions = []
        
        if width > height:
            # РњС–СЃС‚ Р№РґРµ РІР·РґРѕРІР¶ X
            # РћРїРѕСЂРё РїРѕ РєСЂР°СЏС… (Р»С–РІРёР№ С– РїСЂР°РІРёР№) - РґР»СЏ СЃС‚Р°Р±С–Р»СЊРЅРѕСЃС‚С–
            edge_y_positions = [miny + support_width, maxy - support_width]
            
            # Р¦РµРЅС‚СЂР°Р»СЊРЅС– РѕРїРѕСЂРё РІР·РґРѕРІР¶ X РґР»СЏ РґРѕРІРіРёС… РјРѕСЃС‚С–РІ
            num_center_supports = max(0, int((width - 40) / support_spacing))  # РЇРєС‰Рѕ РјС–СЃС‚ РґРѕРІС€РёР№ Р·Р° 40Рј
            if num_center_supports > 0:
                center_x_positions = np.linspace(minx + 20, maxx - 20, num_center_supports)
                # Р”РѕРґР°С”РјРѕ РѕРїРѕСЂРё РЅР° РѕР±РѕС… РєСЂР°СЏС… РґР»СЏ РєРѕР¶РЅРѕС— С†РµРЅС‚СЂР°Р»СЊРЅРѕС— РїРѕР·РёС†С–С—
                for cx in center_x_positions:
                    for ey in edge_y_positions:
                        support_positions.append((cx, ey))
            
            # Р”РѕРґР°С”РјРѕ РѕРїРѕСЂРё РЅР° РїРѕС‡Р°С‚РєСѓ С‚Р° РєС–РЅС†С– РјРѕСЃС‚Р° (РїРѕ РєСЂР°СЏС…)
            for ey in edge_y_positions:
                support_positions.append((minx + support_width, ey))
                support_positions.append((maxx - support_width, ey))
            
            # РЇРєС‰Рѕ РјС–СЃС‚ РєРѕСЂРѕС‚РєРёР№ - РґРѕРґР°С”РјРѕ РѕРїРѕСЂРё РІР·РґРѕРІР¶ С†РµРЅС‚СЂР°Р»СЊРЅРѕС— Р»С–РЅС–С—
            if width <= 40:
                num_supports = max(2, int(width / support_spacing) + 1)
                support_x_positions = np.linspace(minx + support_width, maxx - support_width, num_supports)
                for sx in support_x_positions:
                    support_positions.append((sx, center_y))
        else:
            # РњС–СЃС‚ Р№РґРµ РІР·РґРѕРІР¶ Y
            # РћРїРѕСЂРё РїРѕ РєСЂР°СЏС… (РІРµСЂС…РЅС–Р№ С– РЅРёР¶РЅС–Р№) - РґР»СЏ СЃС‚Р°Р±С–Р»СЊРЅРѕСЃС‚С–
            edge_x_positions = [minx + support_width, maxx - support_width]
            
            # Р¦РµРЅС‚СЂР°Р»СЊРЅС– РѕРїРѕСЂРё РІР·РґРѕРІР¶ Y РґР»СЏ РґРѕРІРіРёС… РјРѕСЃС‚С–РІ
            num_center_supports = max(0, int((height - 40) / support_spacing))
            if num_center_supports > 0:
                center_y_positions = np.linspace(miny + 20, maxy - 20, num_center_supports)
                for cy in center_y_positions:
                    for ex in edge_x_positions:
                        support_positions.append((ex, cy))
            
            # Р”РѕРґР°С”РјРѕ РѕРїРѕСЂРё РЅР° РїРѕС‡Р°С‚РєСѓ С‚Р° РєС–РЅС†С– РјРѕСЃС‚Р° (РїРѕ РєСЂР°СЏС…)
            for ex in edge_x_positions:
                support_positions.append((ex, miny + support_width))
                support_positions.append((ex, maxy - support_width))
            
            # РЇРєС‰Рѕ РјС–СЃС‚ РєРѕСЂРѕС‚РєРёР№ - РґРѕРґР°С”РјРѕ РѕРїРѕСЂРё РІР·РґРѕРІР¶ С†РµРЅС‚СЂР°Р»СЊРЅРѕС— Р»С–РЅС–С—
            if height <= 40:
                num_supports = max(2, int(height / support_spacing) + 1)
                support_y_positions = np.linspace(miny + support_width, maxy - support_width, num_supports)
                for sy in support_y_positions:
                    support_positions.append((center_x, sy))
        
        # Р’РёРґР°Р»СЏС”РјРѕ РґСѓР±Р»С–РєР°С‚Рё (СЏРєС‰Рѕ С”)
        support_positions = list(set(support_positions))
        
        print(f"  [BRIDGE SUPPORTS] РЎС‚РІРѕСЂРµРЅРѕ {len(support_positions)} РїРѕР·РёС†С–Р№ РѕРїРѕСЂ (РїРѕ РєСЂР°СЏС… + С†РµРЅС‚СЂР°Р»СЊРЅС–)")
        
        # РЎС‚РІРѕСЂСЋС”РјРѕ РѕРїРѕСЂРё
        for i, (x, y) in enumerate(support_positions):
            try:
                # РџРµСЂРµРІС–СЂСЏС”РјРѕ, С‡Рё С‚РѕС‡РєР° РІСЃРµСЂРµРґРёРЅС– РїРѕР»С–РіРѕРЅСѓ РјРѕСЃС‚Р°
                pt = Point(x, y)
                if not bridge_polygon.contains(pt) and not bridge_polygon.touches(pt):
                    continue
                
                # РџРћРљР РђР©Р•РќРќРЇ: РЎРµРјРїР»С–РЅРі РІРёСЃРѕС‚Рё РґР»СЏ РїР»РѕС‰С– РѕРїРѕСЂРё (РєС–Р»СЊРєР° С‚РѕС‡РѕРє Р·Р°РјС–СЃС‚СЊ РѕРґРЅС–С”С—)
                # Р¦Рµ Р·Р°Р±РµР·РїРµС‡СѓС” Р±С–Р»СЊС€ С‚РѕС‡РЅСѓ РІРёСЃРѕС‚Сѓ РґР»СЏ РІРµР»РёРєРёС… РѕРїРѕСЂ (2Рј x 2Рј)
                support_half = support_width / 2.0
                sample_points = np.array([
                    [x - support_half, y - support_half],  # Р›С–РІРёР№ РЅРёР¶РЅС–Р№ РєСѓС‚
                    [x + support_half, y - support_half],  # РџСЂР°РІРёР№ РЅРёР¶РЅС–Р№ РєСѓС‚
                    [x - support_half, y + support_half],  # Р›С–РІРёР№ РІРµСЂС…РЅС–Р№ РєСѓС‚
                    [x + support_half, y + support_half],  # РџСЂР°РІРёР№ РІРµСЂС…РЅС–Р№ РєСѓС‚
                    [x, y]  # Р¦РµРЅС‚СЂ
                ])
                
                # РћС‚СЂРёРјСѓС”РјРѕ РІРёСЃРѕС‚Рё РґР»СЏ РІСЃС–С… С‚РѕС‡РѕРє СЃРµРјРїР»С–РЅРіСѓ (РїРѕ СЂРµР°Р»СЊРЅС–Р№ РїРѕРІРµСЂС…РЅС– terrain mesh, СЏРєС‰Рѕ РґРѕСЃС‚СѓРїРЅРѕ)
                ground_zs = terrain_provider.get_surface_heights_for_points(sample_points)
                ground_z = float(np.mean(ground_zs))  # РЎРµСЂРµРґРЅС” Р·РЅР°С‡РµРЅРЅСЏ РґР»СЏ СЃС‚Р°Р±С–Р»СЊРЅРѕСЃС‚С–
                min_ground_z_sample = float(np.min(ground_zs))  # РњС–РЅС–РјР°Р»СЊРЅРµ РґР»СЏ РїРµСЂРµРІС–СЂРєРё РІРѕРґРё
                
                # Р’РёР·РЅР°С‡Р°С”РјРѕ РІРёСЃРѕС‚Сѓ РѕРїРѕСЂРё
                # РЇРєС‰Рѕ С” РІРѕРґР° - РѕРїРѕСЂР° Р№РґРµ РґРѕ СЂС–РІРЅСЏ РІРѕРґРё, С–РЅР°РєС€Рµ РґРѕ Р·РµРјР»С–
                # Р’РёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ min_ground_z_sample РґР»СЏ РїРµСЂРµРІС–СЂРєРё С‡Рё РѕРїРѕСЂР° РІ РІРѕРґС–
                if water_level is not None and min_ground_z_sample < water_level:
                    # РћРїРѕСЂР° РІ РІРѕРґС– - Р№РґРµ РґРѕ СЂС–РІРЅСЏ РІРѕРґРё
                    support_base_z = water_level
                else:
                    # РћРїРѕСЂР° РЅР° Р·РµРјР»С– - РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ СЃРµСЂРµРґРЅС” Р·РЅР°С‡РµРЅРЅСЏ
                    support_base_z = ground_z
                
                support_height = bridge_height - support_base_z
                
                # --- Р’РРџР РђР’Р›Р•РќРќРЇ: Р†РіРЅРѕСЂСѓС”РјРѕ РЅР°РґС‚Рѕ РєРѕСЂРѕС‚РєС– РѕРїРѕСЂРё (СЃРјС–С‚С‚СЏ РЅР° РґРѕСЂРѕРіР°С…) ---
                # Р—Р±С–Р»СЊС€РёРјРѕ РїРѕСЂС–Рі РґРѕ 4 РјРµС‚СЂС–РІ РґР»СЏ РЅР°РґС–Р№РЅРѕСЃС‚С– (РїСЂРёР±РёСЂР°С” С‡РѕСЂРЅС– Р±Р»РѕРєРё РЅР° РїРµСЂРµС…СЂРµСЃС‚СЏС…)
                if support_height < 4.0:
                    continue
                
                # РџРµСЂРµРІС–СЂСЏС”РјРѕ РјС–РЅС–РјР°Р»СЊРЅСѓ РІРёСЃРѕС‚Сѓ
                if support_height < min_support_height:
                    # РЇРєС‰Рѕ РѕРїРѕСЂР° Р·Р°РЅР°РґС‚Рѕ РЅРёР·СЊРєР°, РІСЃРµ РѕРґРЅРѕ СЃС‚РІРѕСЂСЋС”РјРѕ С—С— (РґР»СЏ СЃС‚Р°Р±С–Р»СЊРЅРѕСЃС‚С–)
                    # РђР»Рµ Р· РјС–РЅС–РјР°Р»СЊРЅРѕСЋ РІРёСЃРѕС‚РѕСЋ
                    support_height = max(min_support_height, 0.5)  # РњС–РЅС–РјСѓРј 0.5Рј РґР»СЏ РІРёРґРёРјРѕСЃС‚С–
                    print(f"  [BRIDGE SUPPORT] РћРїРѕСЂР° {i}: РІРёСЃРѕС‚Р° Р·Р±С–Р»СЊС€РµРЅР° РґРѕ РјС–РЅС–РјСѓРјСѓ {support_height:.2f}Рј")
                
                # РЎС‚РІРѕСЂСЋС”РјРѕ С†РёР»С–РЅРґСЂРёС‡РЅСѓ РѕРїРѕСЂСѓ
                # Р’РёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ box Р·Р°РјС–СЃС‚СЊ cylinder РґР»СЏ РїСЂРѕСЃС‚С–С€РѕС— РіРµРѕРјРµС‚СЂС–С— (РєСЂР°С‰Рµ РґР»СЏ 3D РґСЂСѓРєСѓ)
                support_mesh = trimesh.creation.box(
                    extents=[support_width, support_width, support_height],
                    transform=trimesh.transformations.translation_matrix([x, y, support_base_z + support_height / 2.0])
                )
                
                if support_mesh is not None and len(support_mesh.vertices) > 0:
                    # Р—Р°СЃС‚РѕСЃРѕРІСѓС”РјРѕ СЃС–СЂРёР№ РєРѕР»С–СЂ РґРѕ РѕРїРѕСЂ (Р±РµС‚РѕРЅ/РјРµС‚Р°Р»)
                    support_color = np.array([120, 120, 120, 255], dtype=np.uint8)  # РЎС–СЂРёР№ РєРѕР»С–СЂ
                    if len(support_mesh.faces) > 0:
                        face_colors = np.tile(support_color, (len(support_mesh.faces), 1))
                        support_mesh.visual = trimesh.visual.ColorVisuals(face_colors=face_colors)
                    supports.append(support_mesh)
                    
            except Exception as e:
                print(f"  [WARN] РџРѕРјРёР»РєР° СЃС‚РІРѕСЂРµРЅРЅСЏ РѕРїРѕСЂРё {i}: {e}")
                continue
        
    except Exception as e:
        print(f"  [WARN] РџРѕРјРёР»РєР° СЃС‚РІРѕСЂРµРЅРЅСЏ РѕРїРѕСЂ РґР»СЏ РјРѕСЃС‚Р°: {e}")
        import traceback
        traceback.print_exc()
    
    return supports





def detect_bridges(
    G_roads,
    water_geometries: Optional[List] = None,
    bridge_tag: str = 'bridge',
    bridge_buffer_m: float = 12.0,  # buffer around bridge centerline to mark only the bridge area
    clip_polygon: Optional[object] = None,  # Zone polygon for cross-zone bridge detection
) -> List[Tuple[object, object, float, bool, int]]:
    """
    Р’РёР·РЅР°С‡Р°С” РјРѕСЃС‚Рё: РґРѕСЂРѕРіРё, СЏРєС– РїРµСЂРµС‚РёРЅР°СЋС‚СЊ РІРѕРґСѓ Р°Р±Рѕ РјР°СЋС‚СЊ С‚РµРі bridge=yes
    
    Args:
        G_roads: OSMnx РіСЂР°С„ РґРѕСЂС–Рі Р°Р±Рѕ GeoDataFrame
        water_geometries: РЎРїРёСЃРѕРє РіРµРѕРјРµС‚СЂС–Р№ РІРѕРґРЅРёС… РѕР±'С”РєС‚С–РІ (Polygon/MultiPolygon)
        bridge_tag: РўРµРі РґР»СЏ РІРёР·РЅР°С‡РµРЅРЅСЏ РјРѕСЃС‚С–РІ РІ OSM
        
    Returns:
        List of tuples (bridge_line_geometry, bridge_area_geometry, bridge_height_offset, is_over_water, layer).
        - bridge_line_geometry: line-like geometry used for ramping (ideally LineString)
        - bridge_area_geometry: buffered polygon area used to intersect road polygons
        - bridge_height_offset: suggested lift/clearance (world meters)
        - is_over_water: True if bridge crosses water
        - layer: OSM layer value (0 = ground, 1 = first level, 2+ = higher levels)
        - start_connected_to_bridge: True if start of line connects to another bridge
        - end_connected_to_bridge: True if end of line connects to another bridge
    """
    bridges = []
    
    if G_roads is None:
        return bridges
    
    # Analyze connectivity if G_roads is a Graph
    elevated_nodes = set()
    node_bridge_count = {}
    
    is_graph = hasattr(G_roads, "nodes") and hasattr(G_roads, "edges") and not isinstance(G_roads, gpd.GeoDataFrame)
    
    if is_graph:
        try:
            # First pass: identify all bridge edges
            for u, v, k, data in G_roads.edges(keys=True, data=True):
                is_b = False
                
                # Helper to check bridge tag
                def check_tag(d):
                    val = d.get(bridge_tag)
                    if val and str(val).lower() in {"yes", "true", "1", "viaduct", "aqueduct"}:
                        return True
                    if str(d.get("bridge:structure", "")).lower() != "" or str(d.get("man_made", "")).lower() == "bridge":
                        return True
                    try:
                        if float(d.get("layer", 0)) >= 1:
                            return True
                    except:
                        pass
                    return False

                if check_tag(data):
                    node_bridge_count[u] = node_bridge_count.get(u, 0) + 1
                    node_bridge_count[v] = node_bridge_count.get(v, 0) + 1
            
            # Identify nodes where ALL incident edges (degree > 0) are bridges?
            # Or at least where there is MORE than 1 bridge connected?
            # If a node connects 2 bridges, it's an elevated joint -> no ramp.
            # If a node connects 1 bridge and 1 ground road -> it's a ramp.
            for n, count in node_bridge_count.items():
                if count >= 2:
                    elevated_nodes.add(n)
                    
        except Exception as e:
            print(f"[WARN] Connectivity analysis failed: {e}")

    # РџС–РґС‚СЂРёРјРєР° 2 СЂРµР¶РёРјС–РІ
    gdf_edges = None
    if isinstance(G_roads, gpd.GeoDataFrame):
        gdf_edges = G_roads
    else:
        if not hasattr(G_roads, "edges") or len(G_roads.edges) == 0:
            return bridges
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            gdf_edges = ox.graph_to_gdfs(G_roads, nodes=False)
    
    if gdf_edges is None or gdf_edges.empty:
        return bridges
    
    # РћР±'С”РґРЅСѓС”РјРѕ РІСЃС– РІРѕРґРЅС– РѕР±'С”РєС‚Рё РґР»СЏ РїРµСЂРµРІС–СЂРєРё РїРµСЂРµС‚РёРЅСѓ
    water_union = None
    if water_geometries:
        try:
            water_polys = []
            for wg in water_geometries:
                if wg is not None:
                    if isinstance(wg, Polygon):
                        water_polys.append(wg)
                    elif hasattr(wg, 'geoms'):  # MultiPolygon
                        water_polys.extend(wg.geoms)
            if water_polys:
                water_union = unary_union(water_polys)
        except Exception as e:
            print(f"[WARN] РџРѕРјРёР»РєР° РѕР±'С”РґРЅР°РЅРЅСЏ РІРѕРґРЅРёС… РѕР±'С”РєС‚С–РІ РґР»СЏ РІРёР·РЅР°С‡РµРЅРЅСЏ РјРѕСЃС‚С–РІ: {e}")
    
    # РџРµСЂРµРІС–СЂСЏС”РјРѕ РєРѕР¶РЅСѓ РґРѕСЂРѕРіСѓ
    for idx, row in gdf_edges.iterrows():
        try:
            geom = row.geometry
            if geom is None:
                continue
            
            is_bridge = False
            bridge_height = 2.0  # Р‘Р°Р·РѕРІР° РІРёСЃРѕС‚Р° РјРѕСЃС‚Р° (РјРµС‚СЂРё)
            is_over_water = False
            layer_val = 0  # Default layer (ground level)
            
            def _is_bridge_value(v) -> bool:
                # OSMnx/GeoDataFrame can store tag values as str/bool/list.
                if v is None:
                    return False
                if isinstance(v, (list, tuple, set)):
                    return any(_is_bridge_value(x) for x in v)
                if isinstance(v, bool):
                    return bool(v)
                try:
                    s = str(v).strip().lower()
                except Exception:
                    return False
                # OSM sometimes uses "viaduct" etc as bridge values
                return s in {"yes", "true", "1", "viaduct", "aqueduct"} or s.startswith("viaduct")

            # Helper: parse numeric layer
            def _layer_value(v) -> Optional[float]:
                if v is None:
                    return None
                if isinstance(v, (list, tuple, set)):
                    for x in v:
                        lv = _layer_value(x)
                        if lv is not None:
                            return lv
                    return None
                try:
                    return float(str(v).strip())
                except Exception:
                    return None

            # 1. РџРµСЂРµРІС–СЂРєР° С‚РµРіСѓ bridge РІ OSM
            if bridge_tag in row and _is_bridge_value(row.get(bridge_tag)):
                is_bridge = True
                # Р’РёР·РЅР°С‡Р°С”РјРѕ РІРёСЃРѕС‚Сѓ РјРѕСЃС‚Р° Р·Р° С‚РёРїРѕРј
                bridge_type = row.get('bridge:type', '')
                if 'suspension' in str(bridge_type).lower():
                    bridge_height = 5.0
                elif 'arch' in str(bridge_type).lower():
                    bridge_height = 4.0
                elif 'beam' in str(bridge_type).lower():
                    bridge_height = 3.0
                else:
                    bridge_height = 2.5

            # 1.1 Р”РѕРґР°С‚РєРѕРІС– С‚РµРіРё (С–РЅРєРѕР»Рё bridge:structure Р°Р±Рѕ man_made=bridge)
            if not is_bridge:
                try:
                    if _is_bridge_value(row.get("bridge:structure")) or _is_bridge_value(row.get("man_made")):
                        is_bridge = True
                        bridge_height = max(bridge_height, 2.5)
                except Exception:
                    pass

            # 1.2 Layer-based elevation (for viaducts/overpasses even without water)
            # Р’РРџР РђР’Р›Р•РќРќРЇ РґР»СЏ Р±Р°РіР°С‚РѕСЂС–РІРЅРµРІРёС… СЂРѕР·РІ'СЏР·РѕРє: layer=1 -> 6Рј, layer=2 -> 12Рј
            # Р’РёР·РЅР°С‡Р°С”РјРѕ layer Р·РЅР°С‡РµРЅРЅСЏ РґР»СЏ РІСЃС–С… РґРѕСЂС–Рі (РЅР°РІС–С‚СЊ СЏРєС‰Рѕ С†Рµ РЅРµ РјС–СЃС‚)
            try:
                layer_raw = _layer_value(row.get("layer"))
                if layer_raw is not None:
                    layer_val = int(layer_raw)
            except Exception:
                layer_val = 0
            
            if not is_bridge:
                try:
                    layer = _layer_value(row.get("layer"))
                    if layer is not None and layer >= 1.0:
                        is_bridge = True
                        # 6Рј РЅР° layer РґР»СЏ РїСЂР°РІРёР»СЊРЅРёС… С€Р°СЂС–РІ (layer=1 -> 6Рј, layer=2 -> 12Рј)
                        bridge_height = max(bridge_height, float(layer) * 6.0)
                except Exception:
                    pass
            
            # 1.4 CROSS-ZONE BRIDGE DETECTION
            # If road exits zone boundary towards water, classify as bridge
            # BUT only if it's a major road or has "bridge" in name
            if not is_bridge and clip_polygon is not None and water_union is not None:
                try:
                    from shapely.geometry import Point
                    # Check if road crosses zone boundary
                    if not clip_polygon.contains(geom):
                        # Road exits zone - check if it goes towards water
                        if geom.intersects(clip_polygon.boundary):
                            # Get end points
                            coords = list(geom.coords)
                            if len(coords) >= 2:
                                start_point = Point(coords[0])
                                end_point = Point(coords[-1])
                                
                                # Check if either end is outside zone and near water
                                for point in [start_point, end_point]:
                                    if not clip_polygon.contains(point):
                                        # Point outside zone - check distance to water
                                        distance_to_water = point.distance(water_union)
                                        if distance_to_water < 200.0:  # Within 200m of water
                                            # Additional check: only major roads or roads with "bridge" in name
                                            road_name = str(row.get('name', '')).lower()
                                            highway_type = str(row.get('highway', '')).lower()
                                            
                                            is_major_road = highway_type in ['motorway', 'trunk', 'primary']
                                            has_bridge_in_name = 'РјС–СЃС‚' in road_name or 'bridge' in road_name or 'РїР°С‚РѕРЅ' in road_name
                                            
                                            if is_major_road or has_bridge_in_name:
                                                is_bridge = True
                                                bridge_height = max(bridge_height, 10.0)  # Default bridge height
                                                break
                except Exception as e:
                    pass

            
            # 2. РџРµСЂРµРІС–СЂРєР° РїРµСЂРµС‚РёРЅСѓ Р· РІРѕРґРѕСЋ
            if not is_bridge and water_union is not None:
                try:
                    # РџРµСЂРµРІС–СЂСЏС”РјРѕ С‡Рё РґРѕСЂРѕРіР° РїРµСЂРµС‚РёРЅР°С” РІРѕРґСѓ
                    if geom.intersects(water_union):
                        intersection_length = geom.intersection(water_union).length
                        # RELAXED: Accept 1m+ intersection (for partial bridges at zone edges)
                        if intersection_length >= 1.0:
                            is_bridge = True
                            is_over_water = True
                        else:

                            # Р’РёСЃРѕС‚Р° РјРѕСЃС‚Р° Р·Р°Р»РµР¶РёС‚СЊ РІС–Рґ С€РёСЂРёРЅРё РІРѕРґРё
                            if hasattr(water_union, 'area'):
                                # Р—РЅР°С…РѕРґРёРјРѕ РЅР°Р№Р±Р»РёР¶С‡РёР№ РІРѕРґРЅРёР№ РѕР±'С”РєС‚ РґР»СЏ РѕС†С–РЅРєРё С€РёСЂРёРЅРё
                                min_dist = float('inf')
                                for wg in water_geometries:
                                    if wg is not None:
                                        try:
                                            dist = geom.distance(wg)
                                            if dist < min_dist:
                                                min_dist = dist
                                                if hasattr(wg, 'bounds'):
                                                    # РћС†С–РЅСЋС”РјРѕ СЂРѕР·РјС–СЂ РІРѕРґРЅРѕРіРѕ РѕР±'С”РєС‚Р°
                                                    bounds = wg.bounds
                                                    width = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
                                                    if width > 50:  # Р’РµР»РёРєР° СЂС–С‡РєР°
                                                        bridge_height = 4.0
                                                    elif width > 20:  # РЎРµСЂРµРґРЅСЏ СЂС–С‡РєР°
                                                        bridge_height = 3.0
                                                    else:  # РњР°Р»Р° СЂС–С‡РєР°
                                                        bridge_height = 2.0
                                        except:
                                            pass
                except Exception as e:
                    print(f"[WARN] РџРѕРјРёР»РєР° РїРµСЂРµРІС–СЂРєРё РїРµСЂРµС‚РёРЅСѓ РґРѕСЂРѕРіРё Р· РІРѕРґРѕСЋ: {e}")
            
            if is_bridge:
                # IMPORTANT: return an AREA geometry for bridge marking.
                # Using raw edge LineString causes almost all buffered road polygons to "intersect a bridge".
                try:
                    bridge_line = geom
                    # If bridge is detected by water intersection, constrain to the water-crossing portion.
                    if water_union is not None:
                        try:
                            inter = geom.intersection(water_union)
                            if inter is not None and not inter.is_empty:
                                bridge_line = inter
                        except Exception:
                            pass
                    # If MultiLineString -> take the longest part for ramp projection
                    try:
                        if getattr(bridge_line, "geom_type", "") == "MultiLineString":
                            bridge_line = max(list(bridge_line.geoms), key=lambda g: getattr(g, "length", 0.0), default=geom)
                    except Exception:
                        bridge_line = geom
                    
                    # DENSIFY bridge line before buffering to allow curvature/elevation tracking
                    bridge_line = densify_geometry(bridge_line, max_segment_length=10.0)
                    
                    # Buffer into an area (polygon) so later intersection is spatially tight.
                    try:
                        bridge_area = bridge_line.buffer(float(bridge_buffer_m), cap_style=2, join_style=2, resolution=4)
                    except Exception:
                        bridge_area = bridge_line.buffer(float(bridge_buffer_m))
                    if bridge_area is not None and not bridge_area.is_empty:
                        # РљРѕСЂРµРєС†С–СЏ РІРёСЃРѕС‚Рё РЅР° РѕСЃРЅРѕРІС– Layer
                        # Layer 1 = 6m, Layer 2 = 12m
                        final_height = max(bridge_height, float(layer_val) * 6.0)
                        if final_height < 4.0 and layer_val >= 1:
                            final_height = 4.0
                        
                        # Determine connectivity flags
                        start_elevated = False
                        end_elevated = False
                        if is_graph:
                            try:
                                # Try to match geometry endpoints to graph nodes
                                # This is tricky with simplified geometries. 
                                # We'll assume the rows came from graph_to_gdfs and index is (u, v, k)
                                if isinstance(idx, tuple) and len(idx) >= 2:
                                    u, v = idx[0], idx[1]
                                    if u in elevated_nodes:
                                        start_elevated = True
                                    if v in elevated_nodes:
                                        end_elevated = True
                            except Exception:
                                pass
                                
                        bridges.append((bridge_line, bridge_area, final_height, is_over_water, layer_val, start_elevated, end_elevated))
                except Exception:
                    # Fallback to raw geometry if buffering fails
                    try:
                        final_height = max(bridge_height, float(layer_val) * 6.0)
                        if final_height < 4.0 and layer_val >= 1:
                            final_height = 4.0
                        
                        # Densify fallback too
                        geom_d = densify_geometry(geom, max_segment_length=10.0)
                        bridges.append((geom_d, geom_d.buffer(float(bridge_buffer_m), resolution=4), final_height, is_over_water, layer_val, False, False))
                    except Exception:
                        pass
                
        except Exception as e:
            print(f"[WARN] РџРѕРјРёР»РєР° РѕР±СЂРѕР±РєРё РґРѕСЂРѕРіРё РґР»СЏ РІРёР·РЅР°С‡РµРЅРЅСЏ РјРѕСЃС‚Р°: {e}")
            continue
    
    print(f"[INFO] Р’РёР·РЅР°С‡РµРЅРѕ {len(bridges)} РјРѕСЃС‚С–РІ")
    return bridges


def densify_geometry(geom, max_segment_length=10.0):
    if geom is None or geom.is_empty:
        return geom
    
    gt = getattr(geom, "geom_type", "")
    if gt in ["LineString", "LinearRing"]:
        if geom.length <= max_segment_length:
            return geom
        
        import numpy as np
        # Calculate number of segments needed
        num_segments = int(np.ceil(geom.length / max_segment_length))
        if num_segments <= 1:
            return geom
            
        points = []
        # Interpolate points
        # For LinearRing, we want valid closed ring.
        # LineString logic works for LinearRing in shapely generally, but we must return LinearRing if input was one?
        # Actually Polygon constructor expects LinearRing or list of points.
        
        for i in range(num_segments + 1):
            fraction = float(i) / num_segments
            pt = geom.interpolate(fraction, normalized=True)
            points.append((pt.x, pt.y))
        
        from shapely.geometry import LineString, LinearRing
        if gt == "LinearRing":
            return LinearRing(points)
        return LineString(points)
        
    elif gt == "MultiLineString":
        parts = []
        for part in geom.geoms:
            parts.append(densify_geometry(part, max_segment_length))
        from shapely.geometry import MultiLineString
        return MultiLineString(parts)

    elif gt == "Polygon":
        # Densify exterior
        new_ext = densify_geometry(geom.exterior, max_segment_length)
        # Densify interiors
        new_ints = []
        for interior in geom.interiors:
            new_ints.append(densify_geometry(interior, max_segment_length))
        from shapely.geometry import Polygon
        return Polygon(new_ext, new_ints)

    elif gt == "MultiPolygon":
        parts = []
        for part in geom.geoms:
            parts.append(densify_geometry(part, max_segment_length))
        from shapely.geometry import MultiPolygon
        return MultiPolygon(parts)
        
    return geom

def build_road_polygons(
    G_roads,
    width_multiplier: float = 1.0,
    min_width_m: Optional[float] = None,
    extra_buffer_m: float = 0.0,  # Р”РѕРґР°С‚РєРѕРІРёР№ Р±СѓС„РµСЂ Р· РєРѕР¶РЅРѕРіРѕ Р±РѕРєСѓ РґРѕСЂРѕРіРё (РґР»СЏ СЃС‚РІРѕСЂРµРЅРЅСЏ "СѓР·Р±С–С‡С‡СЏ" РїСЂРё РІРёСЂС–Р·Р°РЅРЅС– Р· РїР°СЂРєС–РІ)
    scale_factor: Optional[float] = None,
) -> Optional[object]:
    """
    Builds merged road polygons (2D) from a roads graph/edges gdf.
    This is useful for terrain-first operations (flattening terrain under roads) and
    also allows reusing the merged geometry for mesh generation.
    """
    if G_roads is None:
        return None

    # Support graph or edges GeoDataFrame
    gdf_edges = None
    if isinstance(G_roads, gpd.GeoDataFrame):
        gdf_edges = G_roads
    else:
        if not hasattr(G_roads, "edges") or len(G_roads.edges) == 0:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            gdf_edges = ox.graph_to_gdfs(G_roads, nodes=False)

    # Canonical road masks are restricted to the drivable street network.
    # Sidewalks, footways, cycleways, pedestrian paths, and service spurs
    # should not materialize as printable road inserts.
    if gdf_edges is not None and "highway" in gdf_edges.columns:
        def _is_printable(hwy) -> bool:
            normalized = normalize_drivable_highway_tag(hwy)
            return normalized is not None

        mask = gdf_edges["highway"].apply(_is_printable)
        gdf_edges = gdf_edges[mask].copy()
        if gdf_edges.empty:
            return None
        gdf_edges["_normalized_highway"] = gdf_edges["highway"].apply(normalize_drivable_highway_tag)

    width_map = {
        "motorway": 4.8,
        "motorway_link": 4.2,
        "trunk": 4.2,
        "trunk_link": 3.7,
        "primary": 3.7,
        "primary_link": 3.2,
        "secondary": 3.2,
        "secondary_link": 2.8,
        "tertiary": 2.6,
        "tertiary_link": 2.3,
        "residential": 2.1,
        "living_street": 1.8,
        "service": 1.6,
        "unclassified": 1.9,
        "footway": 1.0,
        "path": 0.85,
        "cycleway": 1.0,
        "pedestrian": 0.95,
        "steps": 0.8,
    }

    def get_width(row):
        highway = row.get("_normalized_highway") or normalize_drivable_highway_tag(row.get('highway'))
        if not highway:
            return 3.0
        width = _resolve_osm_road_width_m(row, width_map.get(highway, 3.0))
        width = width * width_multiplier
        # Ensure minimum printable width (in world meters)
        try:
            if min_width_m is not None:
                width = max(float(width), float(min_width_m))
        except Exception:
            pass
        return (width / 2.0) + float(extra_buffer_m)

    if 'highway' in gdf_edges.columns:
        gdf_edges = gdf_edges.copy()
        
        # DENSIFY GEOMETRY BEFORE BUFFERING
        # This allows extrusion to follow terrain curvature
        gdf_edges["geometry"] = gdf_edges["geometry"].apply(lambda g: densify_geometry(g, max_segment_length=15.0))
        
        # Calculate buffer widths
        widths = gdf_edges.apply(get_width, axis=1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            gdf_edges["geometry"] = gdf_edges.geometry.buffer(widths, cap_style=2, join_style=1, resolution=4)
    else:
        # Fallback if no highway tag
        gdf_edges = gdf_edges.copy()
        gdf_edges["geometry"] = gdf_edges["geometry"].apply(lambda g: densify_geometry(g, max_segment_length=15.0))
        width = 3.0 * width_multiplier
        if min_width_m:
            width = max(width, float(min_width_m))
        rad = (width / 2.0) + float(extra_buffer_m)
        gdf_edges["geometry"] = gdf_edges.geometry.buffer(rad, cap_style=2, join_style=1, resolution=4)

    # Merge all polygons
    try:
        merged = unary_union(gdf_edges.geometry.values)
        return merged
    except Exception as e:
        print(f"[WARN] Failed to merge road polygons: {e}")
        return None


def merge_close_road_gaps(
    road_geom,
    min_gap_m: float,
) -> Optional[object]:
    """
    РћР±'С”РґРЅСѓС” РІСѓР·СЊРєС– РїСЂРѕРјС–Р¶РєРё РјС–Р¶ РґРѕСЂРѕРіР°РјРё (РЅР°РїСЂРёРєР»Р°Рґ, РјС–Р¶ Р·СѓСЃС‚СЂС–С‡РЅРёРјРё СЃРјСѓРіР°РјРё) Сѓ С”РґРёРЅСѓ РґРѕСЂРѕРіСѓ.
    РџСЂРѕРјС–Р¶РєРё РјРµРЅС€С– Р·Р° min_gap_m Р·Р°РјС–РЅСЋСЋС‚СЊСЃСЏ РґРѕСЂРѕРіРѕСЋ вЂ” С‰РѕР± РЅРµ СЃС‚РІРѕСЂСЋРІР°С‚Рё РЅРµРїСЂС–РЅС‚Р°Р±РµР»СЊРЅС– С€РјР°С‚РєРё СЂРµР»СЊС”С„Сѓ.
    
    Args:
        road_geom: Р“РµРѕРјРµС‚СЂС–СЏ РґРѕСЂС–Рі (Polygon Р°Р±Рѕ MultiPolygon)
        min_gap_m: РњС–РЅС–РјР°Р»СЊРЅР° С€РёСЂРёРЅР° РїСЂРѕРјС–Р¶РєСѓ РІ РјРµС‚СЂР°С… вЂ” СЏРєС‰Рѕ РјРµРЅС€Рµ, РѕР±'С”РґРЅСѓС”РјРѕ Р· РґРѕСЂРѕРіРѕСЋ
    
    Returns:
        РћР±'С”РґРЅР°РЅР° РіРµРѕРјРµС‚СЂС–СЏ
    """
    if road_geom is None or road_geom.is_empty:
        return road_geom
    if min_gap_m <= 0:
        return road_geom
    try:
        half_gap = min_gap_m / 2.0
        merged = road_geom.buffer(half_gap, join_style=2)
        if merged is None or merged.is_empty:
            return road_geom
        closed = merged.buffer(-half_gap, join_style=2)
        if closed is None or closed.is_empty:
            return road_geom
        if not closed.is_valid:
            closed = closed.buffer(0)
        added = closed.difference(road_geom)
        if added is None or getattr(added, "is_empty", True):
            return closed if closed is not None and not closed.is_empty else road_geom

        def _iter_polygons(geom):
            if geom is None or getattr(geom, "is_empty", True):
                return []
            if isinstance(geom, Polygon):
                return [geom]
            if isinstance(geom, MultiPolygon):
                return [poly for poly in geom.geoms if poly is not None and not poly.is_empty]
            if hasattr(geom, "geoms"):
                return [poly for poly in geom.geoms if isinstance(poly, Polygon) and not poly.is_empty]
            return []

        keep_additions = []
        for poly in _iter_polygons(added):
            try:
                minx, miny, maxx, maxy = poly.bounds
                width = float(maxx - minx)
                height = float(maxy - miny)
                min_dim = min(width, height)
                perimeter = float(getattr(poly, "length", 0.0) or 0.0)
                area = float(getattr(poly, "area", 0.0) or 0.0)
                equiv_width = (2.0 * area / perimeter) if perimeter > 0 else max(width, height)
                if min_dim > (min_gap_m * 1.25):
                    continue
                if equiv_width > (min_gap_m * 1.1):
                    continue
                keep_additions.append(poly)
            except Exception:
                continue

        if not keep_additions:
            return road_geom
        refined = unary_union([road_geom, *keep_additions]).buffer(0)
        return refined if refined is not None and not refined.is_empty else road_geom
    except Exception as e:
        print(f"[WARN] merge_close_road_gaps failed: {e}")
        return road_geom


def filter_non_printable_road_polygons(
    road_geom,
    *,
    min_feature_m: float,
    min_component_width_m: float = 0.0,
    min_area_factor: float = 0.35,
):
    if road_geom is None or getattr(road_geom, "is_empty", True) or min_feature_m <= 0:
        return road_geom

    min_area_m2 = max((min_feature_m ** 2) * float(min_area_factor), 1e-6)

    def _keep(poly) -> bool:
        if poly is None or poly.is_empty:
            return False
        try:
            poly = poly.buffer(0)
        except Exception:
            pass
        if poly is None or poly.is_empty:
            return False
        try:
            if float(getattr(poly, "area", 0.0) or 0.0) < min_area_m2:
                return False
            minx, miny, maxx, maxy = poly.bounds
            width = float(maxx - minx)
            height = float(maxy - miny)
            min_dim = min(width, height)
            max_dim = max(width, height)
            aspect_ratio = max_dim / max(min_dim, 1e-9)
            area = float(getattr(poly, "area", 0.0) or 0.0)
            if min_component_width_m > 0:
                component_min_area_m2 = max((float(min_component_width_m) ** 2) * 18.0, min_area_m2)
                if min_dim < float(min_component_width_m) and area < component_min_area_m2:
                    return False
            if min_dim < (min_feature_m * 0.75) and area < (min_area_m2 * 2.0):
                return False
            # Remove isolated road shards which are still too narrow to print as
            # standalone road fragments.
            if min_dim < (min_feature_m * 2.6) and aspect_ratio > 8.0 and area < (min_area_m2 * 40.0):
                return False
        except Exception:
            return False
        return True

    if isinstance(road_geom, Polygon):
        return road_geom if _keep(road_geom) else None
    if isinstance(road_geom, MultiPolygon):
        kept = [poly for poly in road_geom.geoms if _keep(poly)]
        if not kept:
            return None
        return unary_union(kept).buffer(0)
    if hasattr(road_geom, "geoms"):
        kept = [poly for poly in road_geom.geoms if isinstance(poly, Polygon) and _keep(poly)]
        if not kept:
            return None
        return unary_union(kept).buffer(0)
    return road_geom


def _fill_small_holes_in_polygon(
    poly: Polygon,
    *,
    min_feature_m: float,
    min_area_factor: float = 0.35,
    preserve_geom=None,
) -> Optional[Polygon]:
    if poly is None or poly.is_empty:
        return None

    min_area_m2 = max((min_feature_m ** 2) * float(min_area_factor), 1e-6)
    max_narrow_hole_min_dim_m = float(min_feature_m) * 2.5
    max_narrow_hole_area_m2 = max((float(min_feature_m) ** 2) * 10.0, min_area_m2)
    max_internal_void_min_dim_m = float(min_feature_m) * 7.0
    max_internal_void_area_m2 = max((float(min_feature_m) ** 2) * 320.0, min_area_m2)
    kept_interiors = []
    preserve_buffer = None
    if preserve_geom is not None and not getattr(preserve_geom, "is_empty", True):
        try:
            preserve_buffer = preserve_geom.buffer(0)
        except Exception:
            preserve_buffer = preserve_geom
    try:
        for interior in poly.interiors:
            try:
                hole_poly = Polygon(interior)
            except Exception:
                continue
            if hole_poly.is_empty:
                continue
            if preserve_buffer is not None:
                try:
                    overlap = hole_poly.intersection(preserve_buffer)
                    overlap_area = float(getattr(overlap, "area", 0.0) or 0.0)
                    hole_area = float(getattr(hole_poly, "area", 0.0) or 0.0)
                    if overlap_area > 0.0 and (
                        overlap_area >= (hole_area * 0.05) or hole_poly.centroid.within(preserve_buffer)
                    ):
                        kept_interiors.append(interior)
                        continue
                except Exception:
                    pass
            try:
                minx, miny, maxx, maxy = hole_poly.bounds
                width = float(maxx - minx)
                height = float(maxy - miny)
                area = float(getattr(hole_poly, "area", 0.0) or 0.0)
            except Exception:
                continue

            # Tiny or narrow voids between roads should be filled as road,
            # not preserved as printable terrain slivers.
            min_dim = min(width, height)
            max_dim = max(width, height)
            aspect_ratio = max_dim / max(min_dim, 1e-9)
            if area < min_area_m2:
                continue
            # Any interior hole thinner than the printable feature threshold
            # should not survive, even if it is long.
            if min_dim < (float(min_feature_m) * 1.1):
                continue
            # Long but very narrow wedges are also not printable and should be filled.
            if min_dim < max_narrow_hole_min_dim_m and area <= (max_narrow_hole_area_m2 * 6.0):
                continue
            # Very elongated slivers are also not printable, even when their
            # total area is a bit larger than a tiny-hole threshold.
            if min_dim < (max_narrow_hole_min_dim_m * 1.15) and aspect_ratio >= 3.5:
                continue
            # Remaining beige wedges in the debug mask are mostly long narrow
            # interior holes. Fill them based on width/aspect ratio instead of
            # relying only on tiny-area thresholds.
            if min_dim < (float(min_feature_m) * 3.6) and aspect_ratio >= 1.8:
                continue
            if min_dim < (float(min_feature_m) * 4.2) and area < (min_area_m2 * 140.0):
                continue
            if min_dim < max_internal_void_min_dim_m and aspect_ratio >= 1.2 and area < max_internal_void_area_m2:
                continue
            if preserve_buffer is not None:
                if min_dim < (float(min_feature_m) * 15.0) and area < max((float(min_feature_m) ** 2) * 2000.0, 1e-8):
                    continue
            if min_dim < min_feature_m:
                continue
            kept_interiors.append(interior)
    except Exception:
        return poly

    try:
        return Polygon(poly.exterior.coords, holes=kept_interiors).buffer(0)
    except Exception:
        return poly


def fill_small_road_voids(
    road_geom,
    *,
    min_feature_m: float,
    min_area_factor: float = 0.35,
    preserve_geom=None,
):
    if road_geom is None or getattr(road_geom, "is_empty", True) or min_feature_m <= 0:
        return road_geom

    if isinstance(road_geom, Polygon):
        return _fill_small_holes_in_polygon(
            road_geom,
            min_feature_m=min_feature_m,
            min_area_factor=min_area_factor,
            preserve_geom=preserve_geom,
        )
    if isinstance(road_geom, MultiPolygon):
        polys = [
            _fill_small_holes_in_polygon(
                poly,
                min_feature_m=min_feature_m,
                min_area_factor=min_area_factor,
                preserve_geom=preserve_geom,
            )
            for poly in road_geom.geoms
            if isinstance(poly, Polygon) and not poly.is_empty
        ]
        polys = [poly for poly in polys if poly is not None and not poly.is_empty]
        if not polys:
            return None
        return unary_union(polys).buffer(0)
    if hasattr(road_geom, "geoms"):
        polys = [
            _fill_small_holes_in_polygon(
                poly,
                min_feature_m=min_feature_m,
                min_area_factor=min_area_factor,
                preserve_geom=preserve_geom,
            )
            for poly in road_geom.geoms
            if isinstance(poly, Polygon) and not poly.is_empty
        ]
        polys = [poly for poly in polys if poly is not None and not poly.is_empty]
        if not polys:
            return None
        return unary_union(polys).buffer(0)
    return road_geom


def fill_narrow_orphan_road_holes(
    road_geom,
    *,
    max_hole_width_m: float,
    preserve_geom=None,
):
    if road_geom is None or getattr(road_geom, "is_empty", True) or max_hole_width_m <= 0:
        return road_geom

    preserve_buffer = None
    if preserve_geom is not None and not getattr(preserve_geom, "is_empty", True):
        try:
            preserve_buffer = preserve_geom.buffer(0)
        except Exception:
            preserve_buffer = preserve_geom

    def _fill_for_polygon(poly: Polygon) -> Optional[Polygon]:
        if poly is None or poly.is_empty:
            return None
        kept_interiors = []
        for interior in poly.interiors:
            try:
                hole_poly = Polygon(interior)
            except Exception:
                kept_interiors.append(interior)
                continue
            if hole_poly.is_empty:
                continue

            if preserve_buffer is not None:
                try:
                    overlap = hole_poly.intersection(preserve_buffer)
                    overlap_area = float(getattr(overlap, "area", 0.0) or 0.0)
                    hole_area = float(getattr(hole_poly, "area", 0.0) or 0.0)
                    if overlap_area > 0.0 and (
                        overlap_area >= (hole_area * 0.05) or hole_poly.centroid.within(preserve_buffer)
                    ):
                        kept_interiors.append(interior)
                        continue
                except Exception:
                    pass

            try:
                minx, miny, maxx, maxy = hole_poly.bounds
                width = float(maxx - minx)
                height = float(maxy - miny)
                min_dim = min(width, height)
                max_dim = max(width, height)
                area = float(getattr(hole_poly, "area", 0.0) or 0.0)
                aspect_ratio = max_dim / max(min_dim, 1e-9)
            except Exception:
                kept_interiors.append(interior)
                continue

            hole_width_limit = float(max_hole_width_m)
            relaxed_width_limit = hole_width_limit * 1.6
            max_area_m2 = max((hole_width_limit ** 2) * 24.0, 1e-8)
            relaxed_area_m2 = max((hole_width_limit ** 2) * 8.0, 1e-8)

            # Fill obviously narrow orphan slits first.
            if min_dim <= hole_width_limit and (area <= max_area_m2 or aspect_ratio >= 3.0):
                continue

            # Some residual white wedges are slightly wider than the nominal
            # threshold but still tiny compared to a real city-block void.
            # Allow a conservative relaxed pass for holes that remain both
            # narrow-ish and small-area or strongly elongated.
            if min_dim <= relaxed_width_limit and (
                area <= relaxed_area_m2 or aspect_ratio >= 4.5
            ):
                continue

            kept_interiors.append(interior)

        try:
            return Polygon(poly.exterior.coords, holes=kept_interiors).buffer(0)
        except Exception:
            return poly

    if isinstance(road_geom, Polygon):
        return _fill_for_polygon(road_geom)
    if isinstance(road_geom, MultiPolygon):
        polys = [_fill_for_polygon(poly) for poly in road_geom.geoms if isinstance(poly, Polygon) and not poly.is_empty]
        polys = [poly for poly in polys if poly is not None and not poly.is_empty]
        if not polys:
            return None
        return unary_union(polys).buffer(0)
    if hasattr(road_geom, "geoms"):
        polys = [_fill_for_polygon(poly) for poly in road_geom.geoms if isinstance(poly, Polygon) and not poly.is_empty]
        polys = [poly for poly in polys if poly is not None and not poly.is_empty]
        if not polys:
            return None
        return unary_union(polys).buffer(0)
    return road_geom


def fill_narrow_boundary_road_gaps(
    road_geom,
    *,
    zone_polygon,
    max_gap_width_m: float,
    preserve_geom=None,
):
    """Fill narrow terrain slivers between roads and the model boundary."""
    if (
        road_geom is None
        or getattr(road_geom, "is_empty", True)
        or zone_polygon is None
        or getattr(zone_polygon, "is_empty", True)
        or max_gap_width_m <= 0
    ):
        return road_geom

    preserve_buffer = None
    if preserve_geom is not None and not getattr(preserve_geom, "is_empty", True):
        try:
            preserve_buffer = preserve_geom.buffer(0)
        except Exception:
            preserve_buffer = preserve_geom

    try:
        terrain = zone_polygon.difference(road_geom).buffer(0)
    except Exception:
        return road_geom

    if terrain is None or getattr(terrain, "is_empty", True):
        return road_geom

    boundary_geom = zone_polygon.boundary
    additions = []
    terrain_parts = terrain.geoms if hasattr(terrain, "geoms") else [terrain]
    for part in terrain_parts:
        if part is None or getattr(part, "is_empty", True):
            continue
        try:
            if not part.intersects(boundary_geom):
                continue
        except Exception:
            continue
        if preserve_buffer is not None:
            try:
                overlap = part.intersection(preserve_buffer)
                if float(getattr(overlap, "area", 0.0) or 0.0) > 0.0:
                    continue
            except Exception:
                pass
        try:
            minx, miny, maxx, maxy = part.bounds
            width = float(maxx - minx)
            height = float(maxy - miny)
            min_dim = min(width, height)
            max_dim = max(width, height)
            area = float(getattr(part, "area", 0.0) or 0.0)
            avg_width = area / max(max_dim, 1e-9)
            aspect_ratio = max_dim / max(min_dim, 1e-9)
            road_shared = float(part.boundary.intersection(road_geom.boundary).length)
            border_shared = float(part.boundary.intersection(boundary_geom).length)
        except Exception:
            continue
        if road_shared <= 0.0 or border_shared <= 0.0:
            continue
        try:
            eroded = part.buffer(-(float(max_gap_width_m) / 2.0), join_style=2)
            too_thin = eroded is None or getattr(eroded, "is_empty", True)
        except Exception:
            too_thin = False
        if min_dim > (float(max_gap_width_m) * 1.25) and avg_width > (float(max_gap_width_m) * 0.95):
            continue
        if not too_thin and avg_width > float(max_gap_width_m) and aspect_ratio < 3.0:
            continue
        additions.append(part)

    if not additions:
        return road_geom

    try:
        return unary_union([road_geom, *additions]).buffer(0)
    except Exception:
        return road_geom


def merge_boundary_road_gaps_with_model_edge(
    road_geom,
    *,
    zone_polygon,
    max_gap_width_m: float,
    preserve_geom=None,
):
    """Treat model edge as a virtual road side and close narrow open boundary gaps."""
    if (
        road_geom is None
        or getattr(road_geom, "is_empty", True)
        or zone_polygon is None
        or getattr(zone_polygon, "is_empty", True)
        or max_gap_width_m <= 0
    ):
        return road_geom

    preserve_buffer = None
    if preserve_geom is not None and not getattr(preserve_geom, "is_empty", True):
        try:
            preserve_buffer = preserve_geom.buffer(0)
        except Exception:
            preserve_buffer = preserve_geom

    try:
        zone_clean = zone_polygon.buffer(0)
    except Exception:
        zone_clean = zone_polygon
    if zone_clean is None or getattr(zone_clean, "is_empty", True):
        return road_geom

    strip_radius = float(max_gap_width_m) / 2.0
    if strip_radius <= 0:
        return road_geom

    try:
        boundary_strip = zone_clean.boundary.buffer(strip_radius, join_style=2).intersection(zone_clean)
    except Exception:
        return road_geom
    if boundary_strip is None or getattr(boundary_strip, "is_empty", True):
        return road_geom

    if preserve_buffer is not None:
        try:
            boundary_strip = boundary_strip.difference(preserve_buffer).buffer(0)
        except Exception:
            pass
        if boundary_strip is None or getattr(boundary_strip, "is_empty", True):
            return road_geom

    try:
        virtual_roads = unary_union([road_geom, boundary_strip]).buffer(0)
    except Exception:
        return road_geom
    if virtual_roads is None or getattr(virtual_roads, "is_empty", True):
        return road_geom

    merged_virtual = merge_close_road_gaps(virtual_roads, float(max_gap_width_m))
    if merged_virtual is None or getattr(merged_virtual, "is_empty", True):
        return road_geom

    try:
        merged_without_strip = merged_virtual.difference(boundary_strip)
        merged_without_strip = merged_without_strip.intersection(zone_clean).buffer(0)
    except Exception:
        return road_geom

    if merged_without_strip is None or getattr(merged_without_strip, "is_empty", True):
        return road_geom

    try:
        additions = merged_without_strip.difference(road_geom)
    except Exception:
        return road_geom
    if additions is None or getattr(additions, "is_empty", True):
        return road_geom

    zone_boundary = None
    try:
        zone_boundary = zone_clean.boundary
    except Exception:
        zone_boundary = None

    keep_additions = []
    parts = additions.geoms if hasattr(additions, "geoms") else [additions]
    max_area_m2 = max((float(max_gap_width_m) ** 2) * 36.0, 1e-8)
    for part in parts:
        if part is None or getattr(part, "is_empty", True):
            continue
        try:
            minx, miny, maxx, maxy = part.bounds
            width = float(maxx - minx)
            height = float(maxy - miny)
            min_dim = min(width, height)
            max_dim = max(width, height)
            area = float(getattr(part, "area", 0.0) or 0.0)
            avg_width = area / max(max_dim, 1e-9)
            perimeter = float(getattr(part, "length", 0.0) or 0.0)
            equiv_width = (2.0 * area / perimeter) if perimeter > 0 else max_dim
            aspect_ratio = max_dim / max(min_dim, 1e-9)
            boundary_distance = (
                float(part.distance(zone_boundary)) if zone_boundary is not None else 0.0
            )
        except Exception:
            continue
        if boundary_distance > (float(max_gap_width_m) * 0.9):
            continue
        if min_dim > (float(max_gap_width_m) * 1.25) and avg_width > (float(max_gap_width_m) * 0.92):
            continue
        if equiv_width > (float(max_gap_width_m) * 1.1):
            continue
        if area > max_area_m2 and aspect_ratio < 4.0:
            continue
        keep_additions.append(part)

    if not keep_additions:
        return road_geom

    try:
        return unary_union([road_geom, *keep_additions]).buffer(0)
    except Exception:
        return road_geom


def fill_narrow_terrain_slivers_between_roads(
    road_geom,
    *,
    zone_polygon,
    max_gap_width_m: float,
    preserve_geom=None,
):
    """Absorb sub-printable terrain slivers that sit between nearby roads.

    This targets narrow non-road leftovers that are not polygon holes yet still
    behave like voids inside the road network. Large terrain islands should stay.
    """
    if (
        road_geom is None
        or getattr(road_geom, "is_empty", True)
        or zone_polygon is None
        or getattr(zone_polygon, "is_empty", True)
        or max_gap_width_m <= 0
    ):
        return road_geom

    preserve_buffer = None
    if preserve_geom is not None and not getattr(preserve_geom, "is_empty", True):
        try:
            preserve_buffer = preserve_geom.buffer(0)
        except Exception:
            preserve_buffer = preserve_geom

    try:
        terrain = zone_polygon.difference(road_geom).buffer(0)
    except Exception:
        return road_geom

    if terrain is None or getattr(terrain, "is_empty", True):
        return road_geom

    boundary_geom = zone_polygon.boundary
    additions = []
    terrain_parts = terrain.geoms if hasattr(terrain, "geoms") else [terrain]
    half_width = float(max_gap_width_m) / 2.0
    max_area_m2 = max((float(max_gap_width_m) ** 2) * 18.0, 1e-8)
    relaxed_area_m2 = max((float(max_gap_width_m) ** 2) * 42.0, 1e-8)
    for part in terrain_parts:
        if part is None or getattr(part, "is_empty", True):
            continue
        if preserve_buffer is not None:
            try:
                overlap = part.intersection(preserve_buffer)
                if float(getattr(overlap, "area", 0.0) or 0.0) > 0.0:
                    continue
            except Exception:
                pass
        try:
            minx, miny, maxx, maxy = part.bounds
            width = float(maxx - minx)
            height = float(maxy - miny)
            min_dim = min(width, height)
            max_dim = max(width, height)
            area = float(getattr(part, "area", 0.0) or 0.0)
            avg_width = area / max(max_dim, 1e-9)
            aspect_ratio = max_dim / max(min_dim, 1e-9)
            road_shared = float(part.boundary.intersection(road_geom.boundary).length)
            border_shared = float(part.boundary.intersection(boundary_geom).length)
        except Exception:
            continue
        if road_shared <= 0.0:
            continue
        # Boundary slivers are handled by the dedicated boundary pass.
        if border_shared > 0.0:
            continue
        try:
            eroded = part.buffer(-half_width, join_style=2)
            too_thin = eroded is None or getattr(eroded, "is_empty", True)
        except Exception:
            too_thin = False
        if min_dim > (float(max_gap_width_m) * 1.20) and avg_width > (float(max_gap_width_m) * 0.92):
            continue
        if area > relaxed_area_m2 and aspect_ratio < 3.0:
            continue
        if not too_thin and area > max_area_m2 and aspect_ratio < 4.0:
            continue
        additions.append(part)

    if not additions:
        return road_geom

    try:
        return unary_union([road_geom, *additions]).buffer(0)
    except Exception:
        return road_geom


def fill_narrow_terrain_channels_between_roads(
    road_geom,
    *,
    zone_polygon,
    max_gap_width_m: float,
    preserve_geom=None,
):
    """Fill narrow terrain channels even when they are part of a larger open terrain polygon."""
    if (
        road_geom is None
        or getattr(road_geom, "is_empty", True)
        or zone_polygon is None
        or getattr(zone_polygon, "is_empty", True)
        or max_gap_width_m <= 0
    ):
        return road_geom

    preserve_buffer = None
    if preserve_geom is not None and not getattr(preserve_geom, "is_empty", True):
        try:
            preserve_buffer = preserve_geom.buffer(0)
        except Exception:
            preserve_buffer = preserve_geom

    try:
        terrain = zone_polygon.difference(road_geom).buffer(0)
    except Exception:
        return road_geom

    if terrain is None or getattr(terrain, "is_empty", True):
        return road_geom

    half_width = float(max_gap_width_m) / 2.0
    if half_width <= 0:
        return road_geom

    try:
        opened = terrain.buffer(-half_width, join_style=2)
        opened = opened.buffer(half_width, join_style=2) if opened is not None and not getattr(opened, "is_empty", True) else None
        narrow_channels = terrain if opened is None or getattr(opened, "is_empty", True) else terrain.difference(opened)
    except Exception:
        return road_geom

    if narrow_channels is None or getattr(narrow_channels, "is_empty", True):
        return road_geom

    zone_boundary = zone_polygon.boundary
    road_boundary = road_geom.boundary
    max_area_m2 = max((float(max_gap_width_m) ** 2) * 42.0, 1e-8)
    additions = []
    channel_parts = narrow_channels.geoms if hasattr(narrow_channels, "geoms") else [narrow_channels]
    for part in channel_parts:
        if part is None or getattr(part, "is_empty", True):
            continue
        if preserve_buffer is not None:
            try:
                overlap = part.intersection(preserve_buffer)
                if float(getattr(overlap, "area", 0.0) or 0.0) > 0.0:
                    continue
            except Exception:
                pass
        try:
            minx, miny, maxx, maxy = part.bounds
            width = float(maxx - minx)
            height = float(maxy - miny)
            min_dim = min(width, height)
            max_dim = max(width, height)
            area = float(getattr(part, "area", 0.0) or 0.0)
            avg_width = area / max(max_dim, 1e-9)
            aspect_ratio = max_dim / max(min_dim, 1e-9)
            road_shared = float(part.boundary.intersection(road_boundary).length)
            border_shared = float(part.boundary.intersection(zone_boundary).length)
        except Exception:
            continue
        if road_shared <= 0.0:
            continue
        part_boundary_len = float(getattr(part, "length", 0.0) or 0.0)
        road_shared_ratio = road_shared / max(part_boundary_len, 1e-9)
        if min_dim > (float(max_gap_width_m) * 1.35) and avg_width > (float(max_gap_width_m) * 0.95):
            is_small_road_enclosed_wedge = (
                area <= max((float(max_gap_width_m) ** 2) * 7.2, 1e-8)
                and road_shared_ratio >= 0.55
                and aspect_ratio <= 1.35
            )
            if not is_small_road_enclosed_wedge:
                continue
        if area > max_area_m2 and aspect_ratio < 3.2 and border_shared <= 0.0:
            continue
        additions.append(part)

    if not additions:
        return road_geom

    try:
        return unary_union([road_geom, *additions]).buffer(0)
    except Exception:
        return road_geom


def fill_small_road_enclosed_terrain_islands(
    road_geom,
    *,
    zone_polygon,
    max_island_area_m2: float,
    preserve_geom=None,
):
    """Fill small terrain islands that are fully enclosed by road boundaries."""
    if (
        road_geom is None
        or getattr(road_geom, "is_empty", True)
        or zone_polygon is None
        or getattr(zone_polygon, "is_empty", True)
        or max_island_area_m2 <= 0
    ):
        return road_geom

    preserve_buffer = None
    if preserve_geom is not None and not getattr(preserve_geom, "is_empty", True):
        try:
            preserve_buffer = preserve_geom.buffer(0)
        except Exception:
            preserve_buffer = preserve_geom

    try:
        terrain = zone_polygon.difference(road_geom).buffer(0)
    except Exception:
        return road_geom
    if terrain is None or getattr(terrain, "is_empty", True):
        return road_geom

    zone_boundary = zone_polygon.boundary
    road_boundary = road_geom.boundary
    additions = []
    terrain_parts = terrain.geoms if hasattr(terrain, "geoms") else [terrain]
    for part in terrain_parts:
        if part is None or getattr(part, "is_empty", True):
            continue
        if preserve_buffer is not None:
            try:
                overlap = part.intersection(preserve_buffer)
                if float(getattr(overlap, "area", 0.0) or 0.0) > 0.0:
                    continue
            except Exception:
                pass
        try:
            area = float(getattr(part, "area", 0.0) or 0.0)
            if area <= 0.0 or area > float(max_island_area_m2):
                continue
            minx, miny, maxx, maxy = part.bounds
            width = float(maxx - minx)
            height = float(maxy - miny)
            min_dim = min(width, height)
            max_dim = max(width, height)
            avg_width = area / max(max_dim, 1e-9)
            aspect_ratio = max_dim / max(min_dim, 1e-9)
            road_shared = float(part.boundary.intersection(road_boundary).length)
            border_shared = float(part.boundary.intersection(zone_boundary).length)
            perimeter = float(getattr(part, "length", 0.0) or 0.0)
            road_shared_ratio = road_shared / max(perimeter, 1e-9)
        except Exception:
            continue
        if border_shared > 0.0:
            continue
        if road_shared_ratio < 0.985:
            continue
        additions.append(part)

    if not additions:
        return road_geom

    try:
        return unary_union([road_geom, *additions]).buffer(0)
    except Exception:
        return road_geom


def fill_compact_road_medians(
    road_geom,
    *,
    zone_polygon,
    channel_gap_width_m: float,
):
    """Fill compact median-like wedges left by near-parallel roads around junctions."""
    if (
        road_geom is None
        or getattr(road_geom, "is_empty", True)
        or zone_polygon is None
        or getattr(zone_polygon, "is_empty", True)
        or channel_gap_width_m <= 0
    ):
        return road_geom

    close_radius_m = float(max(min(channel_gap_width_m * 0.45, 5.0), 0.5))
    try:
        closed = road_geom.buffer(close_radius_m, join_style=2).buffer(-close_radius_m, join_style=2)
    except Exception:
        return road_geom
    if closed is None or getattr(closed, "is_empty", True):
        return road_geom

    try:
        additions = closed.difference(road_geom)
    except Exception:
        return road_geom
    if additions is None or getattr(additions, "is_empty", True):
        return road_geom

    zone_boundary = zone_polygon.boundary
    keep = []
    parts = additions.geoms if hasattr(additions, "geoms") else [additions]
    for part in parts:
        if part is None or getattr(part, "is_empty", True):
            continue
        try:
            minx, miny, maxx, maxy = part.bounds
            width = float(maxx - minx)
            height = float(maxy - miny)
            min_dim = min(width, height)
            max_dim = max(width, height)
            area = float(getattr(part, "area", 0.0) or 0.0)
            avg_width = area / max(max_dim, 1e-9)
            aspect_ratio = max_dim / max(min_dim, 1e-9)
            road_shared = float(part.boundary.intersection(road_geom.boundary).length)
            perimeter = float(getattr(part, "length", 0.0) or 0.0)
            road_shared_ratio = road_shared / max(perimeter, 1e-9)
            boundary_distance = float(part.distance(zone_boundary))
        except Exception:
            continue

        if area > (float(channel_gap_width_m) ** 2) * 20.0:
            continue
        if avg_width > (float(channel_gap_width_m) * 0.95):
            continue
        if aspect_ratio > 1.35:
            continue
        if road_shared_ratio < 0.88 or road_shared_ratio >= 0.985:
            continue
        if boundary_distance > (float(channel_gap_width_m) * 8.0):
            continue
        keep.append(part)

    if not keep:
        return road_geom

    try:
        return unary_union([road_geom, *keep]).buffer(0)
    except Exception:
        return road_geom


def fill_tiny_road_wedges(
    road_geom,
    *,
    gap_fill_m: float,
):
    """
    Conservatively fill only very small wedge-shaped additions that appear between
    neighboring road branches. This is intentionally stricter than a full closing
    delta, to avoid flooding intersections or widening roads too much.
    """
    if road_geom is None or getattr(road_geom, "is_empty", True) or gap_fill_m <= 0:
        return road_geom

    try:
        closed = merge_close_road_gaps(road_geom, float(gap_fill_m))
    except Exception:
        return road_geom

    if closed is None or getattr(closed, "is_empty", True):
        return road_geom

    try:
        additions = closed.difference(road_geom)
    except Exception:
        return road_geom

    if additions is None or getattr(additions, "is_empty", True):
        return road_geom

    max_min_dim_m = float(gap_fill_m) * 1.35
    max_long_dim_m = float(gap_fill_m) * 6.0
    max_area_m2 = max((float(gap_fill_m) ** 2) * 3.0, 1e-8)

    def _keep(poly) -> bool:
        if poly is None or poly.is_empty:
            return False
        try:
            poly = poly.buffer(0)
        except Exception:
            pass
        if poly is None or poly.is_empty:
            return False
        try:
            minx, miny, maxx, maxy = poly.bounds
            width = float(maxx - minx)
            height = float(maxy - miny)
            min_dim = min(width, height)
            max_dim = max(width, height)
            area = float(getattr(poly, "area", 0.0) or 0.0)
        except Exception:
            return False
        if min_dim > max_min_dim_m:
            return False
        if max_dim > max_long_dim_m:
            return False
        if area > max_area_m2:
            return False
        return True

    if isinstance(additions, Polygon):
        kept = [additions] if _keep(additions) else []
    elif isinstance(additions, MultiPolygon):
        kept = [poly for poly in additions.geoms if _keep(poly)]
    elif hasattr(additions, "geoms"):
        kept = [poly for poly in additions.geoms if isinstance(poly, Polygon) and _keep(poly)]
    else:
        kept = []

    if not kept:
        return road_geom

    try:
        return unary_union([road_geom, *kept]).buffer(0)
    except Exception:
        return road_geom


def trim_narrow_attached_road_branches(
    road_geom,
    *,
    min_width_m: float,
    zone_polygon=None,
):
    """Trim dead-end road branches that remain thinner than the printable width.

    The removal is intentionally conservative: only narrow elongated pieces with a
    single connection back into the reopened road body, or boundary slivers hugging
    the model edge, are removed. Thin connectors between two valid road areas must
    survive even if they temporarily disappear during the opening pass.
    """
    if road_geom is None or getattr(road_geom, "is_empty", True) or min_width_m <= 0:
        return road_geom

    radius = float(min_width_m) / 2.0
    if radius <= 0:
        return road_geom

    try:
        # This pass is only a printability cleanup, not a visual smoothing step.
        # Using a very low arc resolution keeps the morphology fast on large road
        # masks while preserving the dead-end trimming intent.
        opened = road_geom.buffer(-radius, join_style=1, resolution=1)
        reopened = opened.buffer(radius, join_style=1, resolution=1) if opened is not None and not getattr(opened, "is_empty", True) else None
    except Exception:
        return road_geom

    if reopened is None or getattr(reopened, "is_empty", True):
        return road_geom

    try:
        removals = road_geom.difference(reopened)
    except Exception:
        return road_geom
    if removals is None or getattr(removals, "is_empty", True):
        return road_geom

    boundary_geom = None
    if zone_polygon is not None and not getattr(zone_polygon, "is_empty", True):
        try:
            boundary_geom = zone_polygon.boundary
        except Exception:
            boundary_geom = None

    max_area_m2 = max((float(min_width_m) ** 2) * 12.0, 1e-8)
    max_min_dim_m = float(min_width_m) * 1.06
    max_avg_width_m = float(min_width_m) * 0.90
    reopened_boundary = None
    try:
        reopened_boundary = reopened.boundary
    except Exception:
        reopened_boundary = None

    contact_len_threshold = max(float(radius) * 0.22, 1e-6)

    keep_removals = []
    parts = removals.geoms if hasattr(removals, "geoms") else [removals]
    for part in parts:
        if part is None or getattr(part, "is_empty", True):
            continue
        try:
            minx, miny, maxx, maxy = part.bounds
            width = float(maxx - minx)
            height = float(maxy - miny)
            min_dim = min(width, height)
            max_dim = max(width, height)
            area = float(getattr(part, "area", 0.0) or 0.0)
            avg_width = area / max(max_dim, 1e-9)
            aspect_ratio = max_dim / max(min_dim, 1e-9)
        except Exception:
            continue
        if min_dim > max_min_dim_m and avg_width > max_avg_width_m:
            continue
        touches_boundary = False
        if boundary_geom is not None:
            try:
                touches_boundary = part.intersects(boundary_geom)
            except Exception:
                touches_boundary = False
        try:
            contact_len = (
                float(part.boundary.intersection(reopened_boundary).length)
                if reopened_boundary is not None
                else 0.0
            )
        except Exception:
            contact_len = 0.0
        has_reopened_contact = contact_len >= contact_len_threshold
        if not touches_boundary and contact_len >= (contact_len_threshold * 2.2):
            continue
        if area > max_area_m2 and aspect_ratio < 4.0 and not touches_boundary:
            continue
        if not touches_boundary and not has_reopened_contact:
            continue
        if not touches_boundary and aspect_ratio < 2.8 and avg_width > (float(min_width_m) * 0.82):
            continue
        keep_removals.append(part)

    if not keep_removals:
        return road_geom

    try:
        trimmed = road_geom.difference(unary_union(keep_removals)).buffer(0)
        return trimmed if trimmed is not None and not getattr(trimmed, "is_empty", True) else road_geom
    except Exception:
        return road_geom


def normalize_road_mask_for_print(
    road_geom,
    *,
    gap_fill_m: float = 0.0,
    min_feature_m: float = 0.0,
    trim_width_m: float = 0.0,
    preserve_geom=None,
    zone_polygon=None,
    orphan_hole_width_m: float = 0.0,
):
    """
    Normalize a 2D road mask for printing.
    Narrow voids between adjacent roads are filled into the road mask,
    and tiny leftover slivers/islands are removed before 3D extrusion.
    """
    if road_geom is None or getattr(road_geom, "is_empty", True):
        return road_geom

    normalized = road_geom
    effective_gap_fill_m = float(gap_fill_m or 0.0)
    pass_count = 3 if (effective_gap_fill_m > 0 and min_feature_m and min_feature_m > 0) else (2 if (min_feature_m and min_feature_m > 0) else 1)
    effective_orphan_hole_width_m = float(orphan_hole_width_m or 0.0)
    boundary_gap_width_m = float(
        effective_orphan_hole_width_m if effective_orphan_hole_width_m > 0 else effective_gap_fill_m
    )
    channel_gap_width_m = float(boundary_gap_width_m)
    if effective_gap_fill_m > 0:
        channel_gap_width_m = float(
            max(
                channel_gap_width_m,
                min(
                    effective_gap_fill_m * 1.6,
                    max(channel_gap_width_m * 1.35, effective_gap_fill_m),
                ),
            )
        )
    enclosed_island_area_m2 = float(
        max(
            (channel_gap_width_m ** 2) * 8.0 if channel_gap_width_m > 0 else 0.0,
            (boundary_gap_width_m ** 2) * 7.0 if boundary_gap_width_m > 0 else 0.0,
        )
    )

    for _ in range(pass_count):
        try:
            if hasattr(normalized, "geoms"):
                normalized = unary_union(
                    [geom for geom in normalized.geoms if geom is not None and not geom.is_empty]
                )
        except Exception:
            pass
        try:
            normalized = normalized.buffer(0)
        except Exception:
            pass

        if effective_gap_fill_m > 0:
            normalized = merge_close_road_gaps(normalized, effective_gap_fill_m)

        if min_feature_m and min_feature_m > 0:
            if effective_gap_fill_m > 0:
                normalized = fill_tiny_road_wedges(
                    normalized,
                    gap_fill_m=float(effective_gap_fill_m),
                )
            normalized = fill_small_road_voids(
                normalized,
                min_feature_m=float(max(effective_gap_fill_m, min_feature_m)),
                preserve_geom=preserve_geom,
            )
        if zone_polygon is not None and not getattr(zone_polygon, "is_empty", True) and boundary_gap_width_m > 0:
            normalized = merge_boundary_road_gaps_with_model_edge(
                normalized,
                zone_polygon=zone_polygon,
                max_gap_width_m=float(boundary_gap_width_m),
                preserve_geom=preserve_geom,
            )
        if effective_orphan_hole_width_m > 0:
            normalized = fill_narrow_orphan_road_holes(
                normalized,
                max_hole_width_m=float(effective_orphan_hole_width_m),
                preserve_geom=preserve_geom,
            )
            if zone_polygon is not None and not getattr(zone_polygon, "is_empty", True):
                normalized = fill_narrow_terrain_slivers_between_roads(
                    normalized,
                    zone_polygon=zone_polygon,
                    max_gap_width_m=float(effective_orphan_hole_width_m),
                    preserve_geom=preserve_geom,
                )
                normalized = fill_narrow_boundary_road_gaps(
                    normalized,
                    zone_polygon=zone_polygon,
                    max_gap_width_m=float(boundary_gap_width_m),
                    preserve_geom=preserve_geom,
                )
                normalized = fill_narrow_terrain_channels_between_roads(
                    normalized,
                    zone_polygon=zone_polygon,
                    max_gap_width_m=float(channel_gap_width_m),
                    preserve_geom=preserve_geom,
                )
                normalized = fill_small_road_enclosed_terrain_islands(
                    normalized,
                    zone_polygon=zone_polygon,
                    max_island_area_m2=float(enclosed_island_area_m2),
                    preserve_geom=preserve_geom,
                )
                normalized = fill_compact_road_medians(
                    normalized,
                    zone_polygon=zone_polygon,
                    channel_gap_width_m=float(channel_gap_width_m),
                )
        if trim_width_m and trim_width_m > 0:
            normalized = trim_narrow_attached_road_branches(
                normalized,
                min_width_m=float(trim_width_m),
                zone_polygon=zone_polygon,
            )
        if min_feature_m and min_feature_m > 0:
            normalized = filter_non_printable_road_polygons(
                normalized,
                min_feature_m=float(min_feature_m),
                min_component_width_m=float(trim_width_m or 0.0),
            )
        if normalized is None or getattr(normalized, "is_empty", True):
            return None

    if normalized is None or getattr(normalized, "is_empty", True):
        return None

    if effective_gap_fill_m > 0:
        try:
            normalized = merge_close_road_gaps(normalized, effective_gap_fill_m)
        except Exception:
            pass

    try:
        normalized = normalized.buffer(0)
    except Exception:
        pass
    return normalized


def process_roads(
    G_roads,
    width_multiplier: float = 1.0,
    terrain_provider: Optional[TerrainProvider] = None,
    floor_z: Optional[float] = None,  
    clearance_mm: float = 0.0,        
    scale_factor: float = 1.0,        
    road_height: float = 1.0,  # Р’РёСЃРѕС‚Р° РґРѕСЂРѕРіРё Сѓ "СЃРІС–С‚РѕРІРёС…" РѕРґРёРЅРёС†СЏС… (Р·РІРёС‡Р°Р№РЅРѕ РјРµС‚СЂРё РІ UTM-РїСЂРѕС”РєС†С–С—)
    road_embed: float = 0.0,   # РќР°СЃРєС–Р»СЊРєРё "РІС‚РёСЃРЅСѓС‚Рё" РІ СЂРµР»СЊС”С„ (Рј), С‰РѕР± РіР°СЂР°РЅС‚РѕРІР°РЅРѕ РЅРµ РІРёСЃС–Р»Р°
    road_elevation_mm: float = 0.0,
    merged_roads: Optional[object] = None,  # Optional precomputed merged road polygons
    water_geometries: Optional[List] = None,  # Р“РµРѕРјРµС‚СЂС–С— РІРѕРґРЅРёС… РѕР±'С”РєС‚С–РІ РґР»СЏ РІРёР·РЅР°С‡РµРЅРЅСЏ РјРѕСЃС‚С–РІ
    bridge_height_multiplier: float = 1.0,  # РњРЅРѕР¶РЅРёРє РґР»СЏ РІРёСЃРѕС‚Рё РјРѕСЃС‚С–РІ
    global_center: Optional[GlobalCenter] = None,  # Р“Р»РѕР±Р°Р»СЊРЅРёР№ С†РµРЅС‚СЂ РґР»СЏ РїРµСЂРµС‚РІРѕСЂРµРЅРЅСЏ РєРѕРѕСЂРґРёРЅР°С‚
    min_width_m: Optional[float] = None,  # РњС–РЅС–РјР°Р»СЊРЅР° С€РёСЂРёРЅР° РґРѕСЂРѕРіРё (РІ РјРµС‚СЂР°С… Сѓ world units)
    tiny_feature_threshold_mm: float = MICRO_REGION_THRESHOLD_MM,
    merge_gap_mm: float = 0.6,
    building_clearance_mm: float = 0.2,
    clip_polygon: Optional[object] = None,  # Zone polygon in LOCAL coords (for pre-clipping)
    city_cache_key: Optional[str] = None,  # City cache key for cross-zone bridge detection
    building_polygons: Optional[object] = None,  # Building footprints (LOCAL) to subtract from roads
    return_result: bool = False,
) -> Optional[trimesh.Trimesh | RoadProcessingResult]:
    """
    РћР±СЂРѕР±Р»СЏС” РґРѕСЂРѕР¶РЅСЋ РјРµСЂРµР¶Сѓ, СЃС‚РІРѕСЂСЋСЋС‡Рё 3D РјРµС€С– Р· РїСЂР°РІРёР»СЊРЅРѕСЋ С€РёСЂРёРЅРѕСЋ
    
    Args:
        G_roads: OSMnx РіСЂР°С„ РґРѕСЂС–Рі
        width_multiplier: РњРЅРѕР¶РЅРёРє РґР»СЏ С€РёСЂРёРЅРё РґРѕСЂС–Рі
    
    Returns:
        Trimesh РѕР±'С”РєС‚ Р· РѕР±'С”РґРЅР°РЅРёРјРё РґРѕСЂРѕРіР°РјРё
    """
    if G_roads is None:
        return RoadProcessingResult(mesh=None, source_polygons=None, cutting_polygons=None) if return_result else None

    # РџС–РґС‚СЂРёРјРєР° 2 СЂРµР¶РёРјС–РІ:
    # - OSMnx graph (СЏРє Р±СѓР»Рѕ)
    # - GeoDataFrame СЂРµР±РµСЂ (pyrosm network edges)
    gdf_edges = None
    if isinstance(G_roads, gpd.GeoDataFrame):
        gdf_edges = G_roads
    else:
        if not hasattr(G_roads, "edges") or len(G_roads.edges) == 0:
            return RoadProcessingResult(mesh=None, source_polygons=None, cutting_polygons=None) if return_result else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            gdf_edges = ox.graph_to_gdfs(G_roads, nodes=False)
    
    # Helper: decide if geometry looks like UTM (huge coordinates) and convert to local if needed
    def _looks_like_utm(g) -> bool:
        return looks_like_projected_meters(g)

    def _to_local_geom(g):
        if g is None or getattr(g, "is_empty", True) or global_center is None:
            return g
        try:
            if _looks_like_utm(g):
                return to_local_geometry_if_needed(g, global_center, force=True)

            # WGS84 lon/lat fallback -> UTM -> local meters
            bounds = g.bounds
            if (
                abs(float(bounds[0])) <= 180.0
                and abs(float(bounds[2])) <= 180.0
                and abs(float(bounds[1])) <= 90.0
                and abs(float(bounds[3])) <= 90.0
            ):
                def _wgs84_to_local(x, y, z=None):
                    x_utm, y_utm = global_center.to_utm(float(x), float(y))
                    x_local, y_local = global_center.to_local(x_utm, y_utm)
                    return (x_local, y_local) if z is None else (x_local, y_local, z)

                return transform(_wgs84_to_local, g)
        except Exception:
            pass
        return g

    # Build or reuse merged road geometry
    if merged_roads is None:
        print("РЎС‚РІРѕСЂРµРЅРЅСЏ Р±СѓС„РµСЂС–РІ РґРѕСЂС–Рі...")
        merged_roads = build_road_polygons(
            G_roads,
            width_multiplier=width_multiplier,
            min_width_m=min_width_m,
            scale_factor=scale_factor,
        )
    if merged_roads is None:
        return RoadProcessingResult(mesh=None, source_polygons=None, cutting_polygons=None) if return_result else None
    
    # Ensure merged_roads are in LOCAL coords if we have global_center
    merged_roads = _to_local_geom(merged_roads)
    _write_debug_geometry_geojson("road_mask_initial_local", merged_roads)
    
    # Pre-clip to zone polygon (LOCAL coords) to prevent roads outside the zone.
    if clip_polygon is not None:
        try:
            clip_poly_local = clip_polygon
            # If clip_polygon came in UTM, convert too
            clip_poly_local = _to_local_geom(clip_poly_local)
            merged_roads = merged_roads.intersection(clip_poly_local)
            # Snap cut edges to the zone boundary to keep border coordinates stable across adjacent tiles.
            # This reduces tiny per-tile numerical differences that show up as Z seams after draping.
            try:
                merged_roads = snap(merged_roads, clip_poly_local.boundary, 1e-6)
            except Exception:
                pass
            if merged_roads is None or merged_roads.is_empty:
                return RoadProcessingResult(mesh=None, source_polygons=None, cutting_polygons=None) if return_result else None
        except Exception:
            pass

    gap_merge_m = (float(merge_gap_mm) / 1000.0) / scale_factor if scale_factor and scale_factor > 0 else None
    tiny_feature_m = model_mm_to_world_m(tiny_feature_threshold_mm, scale_factor)
    cleanup_threshold_m = float(tiny_feature_m or 0.0)
    building_clearance_m = model_mm_to_world_m(building_clearance_mm, scale_factor)
    if min_width_m is None and scale_factor and scale_factor > 0:
        min_width_m = model_mm_to_world_m(MIN_ROAD_WIDTH_MODEL_MM, scale_factor)
    trim_width_m = float(min_width_m or 0.0)
    orphan_hole_width_m = float(
        max(
            tiny_feature_m or 0.0,
            (gap_merge_m or 0.0) * 1.2,
            trim_width_m,
        )
    )
    building_mask = None
    if (gap_merge_m and gap_merge_m > 0) or cleanup_threshold_m > 0:
        merged_roads = normalize_road_mask_for_print(
            merged_roads,
            gap_fill_m=float(gap_merge_m or 0.0),
            min_feature_m=float(cleanup_threshold_m),
            trim_width_m=float(trim_width_m),
            orphan_hole_width_m=float(orphan_hole_width_m),
        )
        if merged_roads is None or getattr(merged_roads, "is_empty", True):
            return RoadProcessingResult(mesh=None, source_polygons=None, cutting_polygons=None) if return_result else None

    # Preserve canonical road geometry. Buildings are clipped against roads
    # later, so the road layer itself should not be carved by building masks.
    # `building_mask` below is ONLY used as `preserve_geom` for hole-fill — it
    # prevents the normalizer from bridging across a footprint, without
    # fragmenting the road network itself.
    if building_polygons is not None and merged_roads is not None:
        try:
            building_mask = _to_local_geom(building_polygons)
            if building_mask is not None and not getattr(building_mask, "is_empty", False):
                if building_clearance_m > 0:
                    try:
                        building_mask = building_mask.buffer(building_clearance_m, join_style=2)
                    except Exception:
                        pass
        except Exception:
            pass

    # РЇРєС‰Рѕ С” СЂРµР»СЊС”С„ вЂ” РєР»С–РїРёРјРѕ РґРѕСЂРѕРіРё РІ РјРµР¶С– СЂРµР»СЊС”С„Сѓ (Р±СѓС„РµСЂРёР·Р°С†С–СЏ РјРѕР¶Рµ РІРёС…РѕРґРёС‚Рё Р·Р° bbox С– РґР°РІР°С‚Рё "РїСЂРѕРІР°Р»Рё")
    if terrain_provider is not None:
        try:
            min_x, max_x, min_y, max_y = terrain_provider.get_bounds()
            # Р’РРџР РђР’Р›Р•РќРќРЇ: Р РѕР·С€РёСЂСЋС”РјРѕ clip_box РЅР° 100Рј, С‰РѕР± РЅРµ РІС‚СЂР°С‚РёС‚Рё РґРѕСЂРѕРіРё РЅР° СЃР°РјРѕРјСѓ РєСЂР°СЋ
            clip = box(min_x - 100.0, min_y - 100.0, max_x + 100.0, max_y + 100.0)
            merged_roads = merged_roads.intersection(clip)
            # Р’РРџР РђР’Р›Р•РќРќРЇ: РџРµСЂРµРІС–СЂСЏС”РјРѕ С‡Рё СЂРµР·СѓР»СЊС‚Р°С‚ РЅРµ РїРѕСЂРѕР¶РЅС–Р№ С‚Р° РІР°Р»С–РґРЅРёР№
            if merged_roads is None or merged_roads.is_empty:
                print("[WARN] Р”РѕСЂРѕРіРё СЃС‚Р°Р»Рё РїРѕСЂРѕР¶РЅС–РјРё РїС–СЃР»СЏ РѕР±СЂС–Р·Р°РЅРЅСЏ РїРѕ СЂРµР»СЊС”С„Сѓ")
                return RoadProcessingResult(mesh=None, source_polygons=None, cutting_polygons=None) if return_result else None
            # Р’РёРїСЂР°РІР»СЏС”РјРѕ РіРµРѕРјРµС‚СЂС–СЋ СЏРєС‰Рѕ РїРѕС‚СЂС–Р±РЅРѕ
            if not merged_roads.is_valid:
                merged_roads = clean_geometry(merged_roads)
                if merged_roads.is_empty:
                    print("[WARN] Р”РѕСЂРѕРіРё СЃС‚Р°Р»Рё РїРѕСЂРѕР¶РЅС–РјРё РїС–СЃР»СЏ РІРёРїСЂР°РІР»РµРЅРЅСЏ")
                    return RoadProcessingResult(mesh=None, source_polygons=None, cutting_polygons=None) if return_result else None
            if ((gap_merge_m and gap_merge_m > 0) or cleanup_threshold_m > 0) and merged_roads is not None and not merged_roads.is_empty:
                merged_roads = normalize_road_mask_for_print(
                    merged_roads,
                    gap_fill_m=float(gap_merge_m or 0.0),
                    min_feature_m=float(cleanup_threshold_m),
                    trim_width_m=float(trim_width_m),
                    preserve_geom=building_mask,
                    zone_polygon=clip,
                    orphan_hole_width_m=float(orphan_hole_width_m),
                )
        except Exception as e:
            print(f"[WARN] РџРѕРјРёР»РєР° РѕР±СЂС–Р·Р°РЅРЅСЏ РґРѕСЂС–Рі РїРѕ СЂРµР»СЊС”С„Сѓ: {e}")
            pass

    # РљРѕРЅРІРµСЂС‚Р°С†С–СЏ РІ СЃРїРёСЃРѕРє РїРѕР»С–РіРѕРЅС–РІ РґР»СЏ РѕР±СЂРѕР±РєРё
    if merged_roads is None or merged_roads.is_empty:
        print("[WARN] merged_roads РїРѕСЂРѕР¶РЅС–Р№ Р°Р±Рѕ None")
        return RoadProcessingResult(mesh=None, source_polygons=None, cutting_polygons=None) if return_result else None
    merged_roads = _rebuild_road_geometry(merged_roads)
    if (gap_merge_m and gap_merge_m > 0) or cleanup_threshold_m > 0:
        merged_roads = normalize_road_mask_for_print(
            merged_roads,
            gap_fill_m=float(gap_merge_m or 0.0),
            min_feature_m=float(cleanup_threshold_m),
            trim_width_m=float(trim_width_m),
            preserve_geom=building_mask,
            zone_polygon=clip_polygon,
            orphan_hole_width_m=float(orphan_hole_width_m),
        )
        if merged_roads is None or getattr(merged_roads, "is_empty", True):
            print("[WARN] merged_roads became empty after non-printable fragment filtering")
            return RoadProcessingResult(mesh=None, source_polygons=None, cutting_polygons=None) if return_result else None
    _write_debug_geometry_geojson("road_mask_final_2d", merged_roads)
    
    if isinstance(merged_roads, Polygon):
        # РџРµСЂРµРІС–СЂСЏС”РјРѕ С‡Рё РїРѕР»С–РіРѕРЅ РјР°С” РґРѕСЃС‚Р°С‚РЅСЊРѕ С‚РѕС‡РѕРє
        if hasattr(merged_roads, 'exterior') and len(merged_roads.exterior.coords) < 3:
            print(f"[WARN] РџРѕР»С–РіРѕРЅ РґРѕСЂС–Рі РјР°С” РјРµРЅС€Рµ 3 С‚РѕС‡РѕРє ({len(merged_roads.exterior.coords)}), РїСЂРѕРїСѓСЃРєР°С”РјРѕ")
            return RoadProcessingResult(mesh=None, source_polygons=merged_roads, cutting_polygons=merged_roads) if return_result else None
        road_geoms = [merged_roads]
    elif isinstance(merged_roads, MultiPolygon):
        # Р¤С–Р»СЊС‚СЂСѓС”РјРѕ РїРѕР»С–РіРѕРЅРё Р· РґРѕСЃС‚Р°С‚РЅСЊРѕСЋ РєС–Р»СЊРєС–СЃС‚СЋ С‚РѕС‡РѕРє
        road_geoms = []
        for geom in merged_roads.geoms:
            if hasattr(geom, 'exterior') and len(geom.exterior.coords) >= 3:
                road_geoms.append(geom)
            else:
                print(f"[WARN] РџРѕР»С–РіРѕРЅ РґРѕСЂС–Рі РјР°С” РјРµРЅС€Рµ 3 С‚РѕС‡РѕРє, РїСЂРѕРїСѓСЃРєР°С”РјРѕ")
        if len(road_geoms) == 0:
            print("[WARN] Р’СЃС– РїРѕР»С–РіРѕРЅРё РґРѕСЂС–Рі РјР°СЋС‚СЊ РјРµРЅС€Рµ 3 С‚РѕС‡РѕРє")
            return RoadProcessingResult(mesh=None, source_polygons=merged_roads, cutting_polygons=merged_roads) if return_result else None
    else:
        print(f"[WARN] РќРµРІС–РґРѕРјРёР№ С‚РёРї РіРµРѕРјРµС‚СЂС–С— РїС–СЃР»СЏ РѕР±'С”РґРЅР°РЅРЅСЏ: {type(merged_roads)}")
        return RoadProcessingResult(mesh=None, source_polygons=merged_roads, cutting_polygons=merged_roads) if return_result else None
    
    # Р’РђР–Р›РР’Рћ: РџРµСЂРµС‚РІРѕСЂСЋС”РјРѕ water_geometries РІ Р»РѕРєР°Р»СЊРЅС– РєРѕРѕСЂРґРёРЅР°С‚Рё РґР»СЏ РІРёР·РЅР°С‡РµРЅРЅСЏ РјРѕСЃС‚С–РІ
    water_geoms_local = None
    if water_geometries is not None:
        try:
            water_geoms_local = [_to_local_geom(g) for g in water_geometries if g is not None and not getattr(g, "is_empty", False)]
        except Exception:
            water_geoms_local = water_geometries

    bridge_processing_enabled = False

    # Ensure edges are in local coords for local node sampling / optional bridge
    # detection. When bridge processing is disabled, we still keep only a small
    # local subset so the node elevation map does not scan the whole city graph.
    try:
        if global_center is not None and gdf_edges is not None and not gdf_edges.empty:
            # Normalize any incoming CRS (UTM/WGS84) to LOCAL for robust clipping.
            sample_geom = gdf_edges.iloc[0].geometry if len(gdf_edges) else None
            if sample_geom is not None:
                def _edge_to_local(g):
                    return _to_local_geom(g)

                gdf_edges = gdf_edges.copy()
                gdf_edges["geometry"] = gdf_edges["geometry"].apply(_edge_to_local)
                G_roads = gdf_edges
                print("[DEBUG] process_roads: Normalized road edges to LOCAL coordinates.")
            if clip_polygon is not None and not getattr(clip_polygon, "is_empty", True):
                try:
                    clip_bounds = clip_polygon.bounds
                    clip_window = box(
                        float(clip_bounds[0]) - 40.0,
                        float(clip_bounds[1]) - 40.0,
                        float(clip_bounds[2]) + 40.0,
                        float(clip_bounds[3]) + 40.0,
                    )
                    filtered_edges = gdf_edges[gdf_edges.geometry.intersects(clip_window)].copy()
                    if not filtered_edges.empty and len(filtered_edges) < len(gdf_edges):
                        print(
                            f"[DEBUG] process_roads: Prefiltered local road edges for 3D roads: "
                            f"{len(filtered_edges)}/{len(gdf_edges)} kept"
                        )
                        gdf_edges = filtered_edges
                        G_roads = gdf_edges
                except Exception:
                    pass
    except Exception:
        pass
    
    # Р’РёР·РЅР°С‡Р°С”РјРѕ РјРѕСЃС‚Рё РїРµСЂРµРґ РѕР±СЂРѕР±РєРѕСЋ (РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ Р»РѕРєР°Р»СЊРЅС– РєРѕРѕСЂРґРёРЅР°С‚Рё)
    # NOTE: detect_bridges returns bridge AREAS (buffered), not raw edge lines.
    if bridge_processing_enabled:
        bridges = detect_bridges(G_roads, water_geometries=water_geoms_local, clip_polygon=clip_polygon)
    else:
        bridges = []
    
    # --- INLAY MODE DETECTION ---
    use_inlay_mode = (clearance_mm > 0 or scale_factor > 0) and terrain_provider is not None
    if use_inlay_mode:
        print(f"[ROADS] Processing roads (Smart Inlay Mode). Clearance: {clearance_mm}mm, Scale: {scale_factor}")
    
    # Р РћР—Р”Р†Р›РЇР„РњРћ РњРћРЎРўР РќРђ РљРђРўР•Р“РћР Р†Р‡
    # Layer 1 (Low): Р¦Рµ С‡Р°СЃС‚РёРЅР° СЂРѕР·РІ'СЏР·РєРё, СЏРєР° Р·Р°РјС–РЅСЋС” РґРѕСЂРѕРіСѓ РЅР° Р·РµРјР»С–. Р‡С… С‚СЂРµР±Р° "РІРёСЂС–Р·Р°С‚Рё".
    # Layer 2+ (High): Р¦Рµ РµСЃС‚Р°РєР°РґРё, С‰Рѕ Р»РµС‚СЏС‚СЊ Р·РІРµСЂС…Сѓ. Р’РѕРЅРё РќР• РїРѕРІРёРЅРЅС– РІРёСЂС–Р·Р°С‚Рё РґРѕСЂРѕРіРё РїС–Рґ СЃРѕР±РѕСЋ.
    bridges_low = [b for b in bridges if len(b) >= 5 and b[4] <= 1]  # Layer <= 1
    bridges_high = [b for b in bridges if len(b) >= 5 and b[4] > 1]   # Layer > 1
    
    # РњР°СЃРєР° РґР»СЏ РІРёСЂС–Р·Р°РЅРЅСЏ (С‚С–Р»СЊРєРё РЅРёР·СЊРєС– РјРѕСЃС‚Рё + РјРѕСЃС‚Рё РЅР°Рґ РІРѕРґРѕСЋ)
    cut_mask_polys = [b[1] for b in bridges_low if b[1] is not None and not getattr(b[1], "is_empty", False)]
    # Р”РѕРґР°С”РјРѕ РјРѕСЃС‚Рё РЅР°Рґ РІРѕРґРѕСЋ РІ РјР°СЃРєСѓ РІРёСЂС–Р·Р°РЅРЅСЏ (С‰РѕР± РЅРµ Р±СѓР»Рѕ РґРѕСЂС–Рі РЅР° РґРЅС– СЂС–С‡РєРё)
    cut_mask_polys.extend([b[1] for b in bridges if len(b) >= 4 and b[3] and b[1] is not None and not getattr(b[1], "is_empty", False)])
    
    bridge_cut_union = None
    if cut_mask_polys:
        try:
            bridge_cut_union = unary_union(cut_mask_polys)
            if bridge_cut_union is not None and getattr(bridge_cut_union, "is_empty", False):
                bridge_cut_union = None
        except Exception:
            bridge_cut_union = None

    print(f"РЎС‚РІРѕСЂРµРЅРЅСЏ 3D РјРµС€Сѓ РґРѕСЂС–Рі Р· {len(road_geoms)} РїРѕР»С–РіРѕРЅС–РІ (РјРѕСЃС‚С–РІ: {len(bridges) if bridges else 0})...")
    # --- PRE-CALCULATE NODE ELEVATIONS ---
    # Create a KDTree of all graph nodes to force alignment at joints.
    node_tree = None
    node_elevations = None
    if G_roads is not None and terrain_provider is not None:
        try:
            # Extract nodes
            # If G_roads is GDF, we might not have nodes directly.
            # But we can extract endpoints of lines.
            points = []
            
            # Use raw G_roads if it is a graph
            if hasattr(G_roads, "nodes") and hasattr(G_roads, "graph"):
                 for n, d in G_roads.nodes(data=True):
                     if "x" in d and "y" in d:
                         points.append([d["x"], d["y"]])
            # Else if GDF, extract from geometry (less reliable for connectivity but helps geometry alignment)
            elif gdf_edges is not None:
                for idx, row in gdf_edges.iterrows():
                    if row.geometry:
                        if row.geometry.geom_type == 'LineString':
                            points.append(row.geometry.coords[0])
                            points.append(row.geometry.coords[-1])
                        elif row.geometry.geom_type == 'MultiLineString':
                             for g in row.geometry.geoms:
                                points.append(g.coords[0])
                                points.append(g.coords[-1])
            
            if points:
                points_arr = np.array(points)
                # Remove duplicates
                points_arr = np.unique(points_arr, axis=0)
                
                if len(points_arr) > 0:
                    node_tree = cKDTree(points_arr)
                    # Get terrain height for all nodes
                    node_zs = terrain_provider.get_surface_heights_for_points(points_arr)
                    node_elevations = node_zs # This is the RAW ground height at the node
                    print(f"[ROAD] Built Node Elevation Map for {len(points_arr)} nodes.")
        except Exception as e:
            print(f"[WARN] Failed to build node elevation map: {e}")

    road_meshes = []
    
    # Statistics
    stats = {'bridge': 0, 'ground': 0, 'anti_drown': 0}
    
    # --- DENSIFICATION STEP ---
    # Ensure all polygons are densified before extrusion to allow terrain draping
    # This recovers vertices lost during unary_union
    print(f"Densifying {len(road_geoms)} road polygons...")
    road_geoms_densified = []
    for g in road_geoms:
        try:
            densified = densify_geometry(g, max_segment_length=10.0)
            # CRITICAL: Validate geometry after densification to prevent TopologyException
            # buffer(0) fixes self-intersections and invalid geometries
            if not densified.is_valid:
                densified = densified.buffer(0)
            road_geoms_densified.append(densified)
        except Exception as e:
            print(f"[WARN] Failed to densify polygon: {e}. Using original.")
            road_geoms_densified.append(g)
    
    road_geoms = road_geoms_densified

    for poly in road_geoms:
        try:
            # Р’РёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ trimesh.creation.extrude_polygon РґР»СЏ РЅР°РґС–Р№РЅРѕС— РµРєСЃС‚СЂСѓР·С–С—
            # Р¦Рµ Р°РІС‚РѕРјР°С‚РёС‡РЅРѕ РѕР±СЂРѕР±Р»СЏС” РґС–СЂРєРё (holes) С‚Р° РїСЂР°РІРёР»СЊРЅРѕ С‚СЂС–Р°РЅРіСѓР»СЋС”
            try:
                def _iter_polys(g):
                    if g is None or getattr(g, "is_empty", False):
                        return []
                    gt = getattr(g, "geom_type", "")
                    if gt == "Polygon":
                        return [g]
                    if gt == "MultiPolygon":
                        return list(g.geoms)
                    if gt == "GeometryCollection":
                        return [gg for gg in g.geoms if getattr(gg, "geom_type", "") == "Polygon"]
                    return []

                def _process_one(poly_part: Polygon, is_bridge: bool, bridge_height_offset: float, bridge_line=None, start_elevated=False, end_elevated=False, layer=0):
                    # Р”Р»СЏ inlay: Р·Р°РіР°Р»СЊРЅР° РІРёСЃРѕС‚Р° = РІРёРґРёРјР° С‡Р°СЃС‚РёРЅР° + Р·Р°РіР»РёР±Р»РµРЅРЅСЏ
                    re = float(road_embed) if road_embed is not None else 0.0
                    rh = max(float(road_height) + re, 0.0001)
                    re = max(0.0, min(re, rh * 0.8))

                    if poly_part is None or poly_part.is_empty:
                        return

                    # Clean polygon if needed
                    try:
                        if not poly_part.is_valid:
                            poly_part = poly_part.buffer(0)
                        if poly_part.is_empty:
                            return
                        if hasattr(poly_part, "exterior") and len(poly_part.exterior.coords) < 3:
                            return
                    except Exception:
                        return

                    mesh = None
                    try:
                        minx, miny, maxx, maxy = poly_part.bounds
                        max_dim = max(float(maxx - minx), float(maxy - miny))
                        hole_count = len(getattr(poly_part, "interiors", []))
                        needs_uniform_topology = float(getattr(poly_part, "area", 0.0) or 0.0) > 800.0 or hole_count > 0
                        if needs_uniform_topology:
                            target_edge = min(max(max_dim / 20.0, 2.5), 6.0)
                            mesh = extrude_polygon_grid(
                                poly_part,
                                height=rh,
                                target_edge_len_m=target_edge,
                                boundary_segment_m=2.0,
                            )
                            if mesh is not None and not mesh.is_watertight:
                                mesh = refine_mesh_long_edges(mesh, max_edge_m=max(target_edge * 1.15, 3.0))
                    except Exception:
                        mesh = None

                    if mesh is None:
                        mesh = extrude_polygon_uniform(poly_part, height=rh, densify_max_m=2.0)
                    if mesh is None:
                        mesh = trimesh.creation.extrude_polygon(poly_part, height=rh)
                    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                        print(f"  [WARN] Extrude failed: mesh={'None' if mesh is None else f'{len(mesh.vertices)}v, {len(mesh.faces)}f'}, poly_area={poly_part.area:.2f}, is_bridge={is_bridge}")
                        return
                    
                    
                    print(f"  [DEBUG] Extruded: {len(mesh.vertices)}v, {len(mesh.faces)}f, is_bridge={is_bridge}, bridge_offset={bridge_height_offset:.2f}m")
                    
                    # CRITICAL: Fix normals immediately for bridges to ensure proper rendering
                    if is_bridge:
                        try:
                            mesh.fix_normals()
                            # Ensure consistent winding order
                            if not mesh.is_winding_consistent:
                                mesh.fix_normals()
                            print(f"  [DEBUG] Bridge normals fixed, is_watertight={mesh.is_watertight}, is_winding_consistent={mesh.is_winding_consistent}")
                        except Exception as e:
                            print(f"  [WARN] Failed to fix bridge normals: {e}")

                    # Project onto terrain
                    if terrain_provider is not None:
                        vertices = mesh.vertices.copy()
                        
                        # Р’РђР›Р†Р”РђР¦Р†РЇ: РџРµСЂРµРІС–СЂСЏС”РјРѕ РЅР° NaN/Inf РїРµСЂРµРґ РѕР±СЂРѕР±РєРѕСЋ (СЏРє Сѓ H:\3dMap)
                        if np.any(~np.isfinite(vertices)):
                            print(f"  [WARN] Р”РѕСЂРѕРіР°: Р·РЅР°Р№РґРµРЅРѕ NaN/Inf РІ РІРµСЂС€РёРЅР°С… РїС–СЃР»СЏ РµРєСЃС‚СЂСѓР·С–С—")
                            vertices = np.nan_to_num(vertices, nan=0.0, posinf=1e6, neginf=-1e6)
                        
                        old_z = vertices[:, 2].copy()
                        # Get terrain height at each vertex for this polygon
                        ground_z_values = terrain_provider.get_surface_heights_for_points(vertices[:, :2])
                        
                        # Р’РђР›Р†Р”РђР¦Р†РЇ: РџРµСЂРµРІС–СЂСЏС”РјРѕ РЅР° NaN/Inf Сѓ РІРёСЃРѕС‚Р°С… СЂРµР»СЊС”С„Сѓ (СЏРє Сѓ H:\3dMap)
                        if np.any(np.isnan(ground_z_values)) or np.any(np.isinf(ground_z_values)):
                            nan_count = np.sum(np.isnan(ground_z_values))
                            inf_count = np.sum(np.isinf(ground_z_values))
                            print(f"  [WARN] Р”РѕСЂРѕРіР°: Terrain РїРѕРІРµСЂРЅСѓРІ NaN/Inf РІРёСЃРѕС‚ (NaN: {nan_count}, Inf: {inf_count}), РІРёРєРѕСЂРёСЃС‚РѕРІСѓСЋ ground level 0.0")
                            ground_z_values = np.zeros_like(ground_z_values)
                        
                        # --- MULTI-LEVEL ROAD ELEVATION LOGIC ---
                        # Layer 0 (РЅРµ bridge) > Ground road РЅР° terrain
                        # Layer 1 (bridge) > РќРёР·СЊРєР° СЂРѕР·РІ'СЏР·РєР°/РјС–СЃС‚ (+6m)
                        # Layer 2+ (bridge) > Р’РёСЃРѕРєР° РµСЃС‚Р°РєР°РґР° (+12m, +18m, etc)
                        
                        # Get OSM layer value
                        osm_layer = int(layer) if is_bridge else 0
                        
                        if not is_bridge or osm_layer == 0:
                            # GROUND ROADS - road_elevation_mm (РјРј РјРѕРґРµР»С–) РЅР°Рґ terrain
                            # rh = road_height + road_embed (РїРѕРІРЅР° РІРёСЃРѕС‚Р° С–РЅР»РµСЏ)
                            # old_z: 0 (РґРЅРѕ) .. rh (РІРµСЂС…)
                            # road_top = ground_z + rh + offset = ground_z + road_elevation_mm/sf
                            # > offset = road_elevation_mm/sf - rh
                            road_z = ground_z_values + old_z + (road_elevation_mm / scale_factor) - rh
                            
                            # Р’РђР›Р†Р”РђР¦Р†РЇ: РџРµСЂРµРІС–СЂСЏС”РјРѕ РЅР° NaN/Inf РїРµСЂРµРґ Р·Р°СЃС‚РѕСЃСѓРІР°РЅРЅСЏРј (СЏРє Сѓ H:\3dMap)
                            if np.any(np.isnan(road_z)) or np.any(np.isinf(road_z)):
                                print(f"  [WARN] Р”РѕСЂРѕРіР°: РѕР±С‡РёСЃР»РµРЅРЅСЏ РІРёСЃРѕС‚ РґР°Р»Рѕ NaN/Inf, РІРёРїСЂР°РІР»СЏСЋ")
                                road_z = np.nan_to_num(road_z, nan=0.0, posinf=1e6, neginf=-1e6)
                            
                            vertices[:, 2] = road_z
                            
                        else:
                            # BRIDGES - РїС–РґРЅСЏС‚С– Р·Р°Р»РµР¶РЅРѕ РІС–Рґ layer
                            
                            # Calculate base elevation for this bridge
                            if hasattr(terrain_provider, 'original_heights_provider') and terrain_provider.original_heights_provider is not None:
                                # Р„ water data - СЂРѕР·СЂР°С…РѕРІСѓС”РјРѕ clearance
                                try:
                                    original_ground_z = terrain_provider.original_heights_provider.get_heights_for_points(vertices[:, :2])
                                    water_depth = original_ground_z - ground_z_values
                                    max_depth = np.max(water_depth)
                                    
                                    # Р’РёР·РЅР°С‡Р°С”РјРѕ Р±Р°Р·РѕРІСѓ РІРёСЃРѕС‚Сѓ РјРѕСЃС‚Р°
                                    if max_depth > 2.0:
                                        # РњС–СЃС‚ РЅР°Рґ РІРѕРґРѕСЋ - РІРёРєРѕСЂРёСЃС‚Р°С‚Рё clearance
                                        water_level = np.median(original_ground_z - 0.2)
                                        min_clearance = max(8.0, float(bridge_height_offset))
                                        bridge_base_z = water_level + min_clearance
                                        print(f"  [BRIDGE Layer {osm_layer}] Over water: base={bridge_base_z:.2f}m (clearance={min_clearance:.2f}m)")
                                    else:
                                        # Р•СЃС‚Р°РєР°РґР° РЅР°Рґ Р·РµРјР»РµСЋ - РІРёСЃРѕС‚Р° Р±Р°Р·СѓС”С‚СЊСЃСЏ РЅР° layer
                                        layer_height = osm_layer * 6.0  # Layer 1=6m, 2=12m, 3=18m
                                        bridge_base_z = np.median(ground_z_values) + layer_height
                                        print(f"  [BRIDGE Layer {osm_layer}] Overpass: base={bridge_base_z:.2f}m (layer_height={layer_height:.2f}m)")
                                    
                                except Exception as e:
                                    # Fallback
                                    layer_height = max(osm_layer * 6.0, 6.0)
                                    bridge_base_z = np.median(ground_z_values) + layer_height
                                    print(f"  [BRIDGE Layer {osm_layer}] Fallback: base={bridge_base_z:.2f}m")
                            else:
                                # РќРµРјР°С” water data - РІРёРєРѕСЂРёСЃС‚Р°С‚Рё layer height
                                layer_height = max(osm_layer * 6.0, 6.0)
                                bridge_base_z = np.median(ground_z_values) + layer_height
                                print(f"  [BRIDGE Layer {osm_layer}] No water data: base={bridge_base_z:.2f}m")
                            
                            # Apply bridge elevation with smooth ramps
                            # Calculate ramp transitions at start/end
                            ramp_t = None
                            if bridge_line is not None and hasattr(bridge_line, 'length') and bridge_line.length > 10.0:
                                try:
                                    from shapely.geometry import Point as _Pt
                                    line_len = float(bridge_line.length)
                                    ramp_len = min(line_len * 0.3, 30.0)  # 30% of bridge or 30m max
                                    
                                    # Calculate distance along bridge for each vertex
                                    xy = vertices[:, :2]
                                    distances = np.array([bridge_line.project(_Pt(x, y)) for x, y in xy])
                                    
                                    # Sigmoid ramp: 0 at ends, 1 in middle
                                    t_start = np.clip(distances / ramp_len, 0.0, 1.0)
                                    t_end = np.clip((line_len - distances) / ramp_len, 0.0, 1.0)
                                    ramp_t = np.minimum(t_start, t_end)
                                    ramp_t = ramp_t * ramp_t * (3.0 - 2.0 * ramp_t)  # Smoothstep
                                except:
                                    pass
                            
                            if ramp_t is not None:
                                # Blend: ground at ends, bridge in middle
                                base_ground = ground_z_values
                                base_bridge = np.full_like(ground_z_values, bridge_base_z)
                                base = base_ground * (1.0 - ramp_t) + base_bridge * ramp_t
                                vertices[:, 2] = base + old_z
                            else:
                                # No ramp - full bridge height
                                vertices[:, 2] = bridge_base_z + old_z
                            
                            # Do not bake bridge supports into the detachable road insert.
                            # They expand the XY footprint beyond the canonical road mask
                            # and can collide with buildings and grooves.
                            # If a road is NOT detected as a bridge but runs over deep water (depression),
                            # force it to sit on the Water Surface (Original Z) instead of the Riverbed (Ground Z).
                            # This saves "undetected" bridges from spawning underwater.
                            
                            is_drowning = False
                            if hasattr(terrain_provider, 'original_heights_provider') and terrain_provider.original_heights_provider is not None:
                                try:
                                    original_ground_z = terrain_provider.original_heights_provider.get_heights_for_points(vertices[:, :2])
                                    # Calculate depression depth at each vertex
                                    water_depth_approx = original_ground_z - ground_z_values
                                    max_depth = np.max(water_depth_approx) if len(water_depth_approx) > 0 else 0.0
                                    
                                    # CRITICAL UPGRADE: If road is even SLIGHTLY underwater (> 0.1m), promote it to a BRIDGE automatically.
                                    # This ensures "floating" roads become proper thick bridge meshes.
                                    if max_depth > 0.1:
                                        print(f"  [INFO] Anti-Drown: Road is {max_depth:.1f}m underwater. PROMOTING TO BRIDGE!")
                                        # Recursively process as bridge
                                        # Uses 5.0m height offset for reasonable bridge elevation
                                        safe_bridge_height = max(5.0, max_depth + 2.0)
                                        _process_one(poly_part, is_bridge=True, bridge_height_offset=safe_bridge_height, 
                                                    bridge_line=None, start_elevated=True, end_elevated=True)
                                        return # Stop processing as ground

                                    # Standard Anti-Drown (Pontoon mode) for shallow water or noise
                                    water_mask = water_depth_approx > 0.5
                                    if np.any(water_mask):
                                        # Lift to Water Surface + Bias
                                        road_z_water = original_ground_z + old_z + 0.2  # 0.2m bias to prevent drowning
                                        road_z[water_mask] = road_z_water[water_mask]
                                        
                                        is_drowning = True
                                        stats['anti_drown'] += 1
                                except Exception as e:
                                    print(f"  [WARN] Anti-drown check failed: {e}")

                            vertices[:, 2] = road_z
                            if not is_bridge: stats['ground'] += 1
                        
                        # Count bridge statistics (for both water-based and fallback bridges)
                        if is_bridge:
                            stats['bridge'] += 1
                        
                        # Р¤Р†РќРђР›Р¬РќРђ Р’РђР›Р†Р”РђР¦Р†РЇ: РџРµСЂРµРІС–СЂСЏС”РјРѕ РЅР° NaN/Inf РїС–СЃР»СЏ РѕР±С‡РёСЃР»РµРЅСЊ (СЏРє Сѓ H:\3dMap)
                        if np.any(np.isnan(vertices)) or np.any(np.isinf(vertices)):
                            print(f"  [ERROR] Р”РѕСЂРѕРіР°: РІРµСЂС€РёРЅРё РјС–СЃС‚СЏС‚СЊ NaN/Inf РїС–СЃР»СЏ РѕР±С‡РёСЃР»РµРЅСЊ, РІРёРїСЂР°РІР»СЏСЋ")
                            vertices = np.nan_to_num(vertices, nan=0.0, posinf=1e6, neginf=-1e6)
                        
                        mesh.vertices = vertices
                    else:
                        if float(re) > 0:
                            vertices = mesh.vertices.copy()
                            vertices[:, 2] = vertices[:, 2] - float(re)
                            mesh.vertices = vertices
                        if is_bridge: stats['bridge'] += 1

                    # Cleanup + color
                    try:
                        try:
                            trimesh.repair.fix_winding(mesh)
                        except Exception:
                            pass
                        mesh.fix_normals()
                        mesh.update_faces(mesh.unique_faces())
                        mesh.remove_unreferenced_vertices()
                        if not mesh.is_volume:
                            mesh.fill_holes()
                        mesh.merge_vertices(merge_tex=True, merge_norm=True)
                        print(f"  [DEBUG] After cleanup: {len(mesh.vertices)}v, {len(mesh.faces)}f")
                        
                        # Validate coordinates
                        if len(mesh.vertices) > 0:
                            has_nan = np.any(np.isnan(mesh.vertices))
                            has_inf = np.any(np.isinf(mesh.vertices))
                            if has_nan or has_inf:
                                print(f"  [ERROR] Invalid coordinates detected! NaN={has_nan}, Inf={has_inf}")
                                print(f"  [ERROR] Vertex sample: {mesh.vertices[:3]}")
                                return  # Skip this mesh
                    except Exception:
                        pass

                    if len(mesh.faces) > 0:
                        road_color = np.array([60, 60, 60, 255], dtype=np.uint8) if is_bridge else np.array([40, 40, 40, 255], dtype=np.uint8)
                        face_colors = np.tile(road_color, (len(mesh.faces), 1))
                        mesh.visual = trimesh.visual.ColorVisuals(face_colors=face_colors)

                    # Log bounds for bridges
                    if is_bridge and len(mesh.vertices) > 0:
                        bounds = mesh.bounds
                        print(f"  [DEBUG] Bridge bounds: Z from {bounds[0][2]:.2f} to {bounds[1][2]:.2f}")
                    
                    print(f"  [DEBUG] Adding mesh: {len(mesh.vertices)}v, {len(mesh.faces)}f, is_bridge={is_bridge}")
                    road_meshes.append(mesh)

                # Build parts: bridge pieces + remainder
                # --- REFACTORED ROBUST LOGIC ---
                # Instead of separating High/Low loops and trying to match intersections back to bridges,
                # we iterate bridges directly for positive parts, and use a cut-mask for ground parts.
                
                parts_to_process: List[Tuple[Polygon, bool, float, object, bool, bool, int]] = []
                
                # 1. GENERATE BRIDGE PARTS
                # Find all bridges that intersect this road polygon
                relevant_bridges = [b for b in bridges if b[1] is not None and b[1].intersects(poly)]
                
                for b in relevant_bridges:
                    try:
                        # Intersect road poly with SPECIFIC bridge area
                        # This guarantees we know exactly which bridge parameters to use
                        b_inter = poly.intersection(b[1])
                        
                        # Handle MultiPolygon result
                        for p in _iter_polys(b_inter):
                            if p.area < 0.0001: continue  # Relaxed filter from 0.01 to 0.0001
                            
                            b_line = b[0]
                            b_height = float(b[2]) * float(bridge_height_multiplier)
                            b_layer = b[4]
                            b_start_elev = b[5] if len(b) > 5 else False
                            b_end_elev = b[6] if len(b) > 6 else False
                            
                            parts_to_process.append((p, True, b_height, b_line, b_start_elev, b_end_elev, b_layer))
                    except Exception as e:
                        print(f"[WARN] Error processing specific bridge intersection: {e}")

                # 2. GENERATE GROUND PARTS
                # Ground = Poly MINUS (Low Bridges + Bridges Over Water)
                # High bridges (over land) do NOT cut the ground (road runs underneath)
                
                # Identify mask for cutting
                cut_mask_polys = []
                for b in bridges:
                    # Layer <= 1 OR Over Water = Cut Ground
                    if b[4] <= 1 or b[3]:
                        if b[1] is not None:
                            cut_mask_polys.append(b[1])
                
                if cut_mask_polys:
                    try:
                        bridge_cut_union_local = unary_union(cut_mask_polys)
                        ground_parts = poly.difference(bridge_cut_union_local)
                    except Exception as e:
                        print(f"[WARN] Error calculating ground difference: {e}")
                        ground_parts = poly # Fallback
                else:
                    ground_parts = poly
                
                # --- INLAY MODE: Process ground roads with create_deep_road_prism ---
                # РЇРєС‰Рѕ РІСЃС‚Р°РЅРѕРІР»РµРЅРѕ clearance_mm Р°Р±Рѕ scale_factor, РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ Inlay СЂРµР¶РёРј
                use_inlay_mode = (clearance_mm > 0 or scale_factor > 0) and terrain_provider is not None
                
                if use_inlay_mode:
                    # РћР±СЂРѕР±Р»СЏС”РјРѕ ground roads С‡РµСЂРµР· Inlay (create_deep_road_prism)
                    ground_roads_poly = ground_parts
                    
                    # Apply clearance only when resulting road keeps printable thickness.
                    if clearance_mm > 0 and ground_roads_poly and not ground_roads_poly.is_empty:
                        clearance_m = (clearance_mm / 1000.0) / scale_factor
                        try:
                            shrunk = ground_roads_poly.buffer(-clearance_m, join_style=2)
                            if shrunk is not None and not shrunk.is_empty:
                                min_half_m = 0.25 / (scale_factor * 1000.0) if scale_factor > 0 else 0.0
                                erosion_check = shrunk.buffer(-min_half_m, join_style=2) if min_half_m > 0 else shrunk
                                if erosion_check is not None and not erosion_check.is_empty:
                                    ground_roads_poly = shrunk
                                else:
                                    print("[ROAD] Skipping clearance: road < 0.5mm after shrink, keeping original")
                        except Exception as e:
                            print(f"[WARN] Clearance buffer failed: {e}")
                    
                    # Explode to list and process with create_deep_road_prism
                    if ground_roads_poly and not ground_roads_poly.is_empty:
                        polys = []
                        if hasattr(ground_roads_poly, "geoms"):
                            polys = list(ground_roads_poly.geoms)
                        else:
                            polys = [ground_roads_poly]
                            
                        for p in polys:
                            if p.area < 1e-4: continue
                            
                            # Keep ground-road inserts flush with terrain surface.
                            inlay_mesh = create_deep_road_prism(
                                p, terrain_provider, 
                                scale_factor=scale_factor, 
                                top_z_offset=0.0,
                                min_height=1.0
                            )

                            if inlay_mesh:
                                base_ground_mesh = _cleanup_road_mesh(inlay_mesh.copy())
                                final_ground_mesh = base_ground_mesh if base_ground_mesh is not None else inlay_mesh
                                try:
                                    final_ground_mesh = _clip_mesh_to_road_footprint(final_ground_mesh, p)
                                    final_ground_mesh = _cleanup_road_mesh(final_ground_mesh)
                                except Exception:
                                    pass

                                final_ground_mesh.visual.face_colors = [60, 60, 60, 255]
                                road_meshes.append(final_ground_mesh)
                                stats['ground'] += 1
                else:
                    # РЎС‚Р°РЅРґР°СЂС‚РЅР° РѕР±СЂРѕР±РєР° ground roads (С‡РµСЂРµР· _process_one)
                    for p in _iter_polys(ground_parts):
                        if p.area < 0.01: continue
                        parts_to_process.append((p, False, 0.0, None, False, False, 0))

                    # Handle case where everything was cut or empty
                    if not parts_to_process and not relevant_bridges:
                         parts_to_process.append((poly, False, 0.0, None, False, False, 0))
                    
                    # Process parts (bridges + ground roads in standard mode)
                    for part_poly, is_bridge, bridge_height_offset, bridge_line, start_elev, end_elev, layer_val in parts_to_process:
                        _process_one(part_poly, is_bridge, bridge_height_offset, bridge_line, start_elev, end_elev, layer=layer_val)
                
            except Exception as extrude_error:
                print(f"  РџРѕРјРёР»РєР° РµРєСЃС‚СЂСѓР·С–С— РїРѕР»С–РіРѕРЅСѓ: {extrude_error}")
                # Fallback: СЃРїСЂРѕР±СѓС”РјРѕ СЃС‚РІРѕСЂРёС‚Рё РїСЂРѕСЃС‚РёР№ РјРµС€
                continue
                
        except Exception as e:
            print(f"РџРѕРјРёР»РєР° РѕР±СЂРѕР±РєРё РїРѕР»С–РіРѕРЅСѓ РґРѕСЂРѕРіРё: {e}")
            continue
    
    if not road_meshes:
        print("РџРѕРїРµСЂРµРґР¶РµРЅРЅСЏ: РќРµ РІРґР°Р»РѕСЃСЏ СЃС‚РІРѕСЂРёС‚Рё Р¶РѕРґРЅРѕРіРѕ РјРµС€Сѓ РґРѕСЂС–Рі")
        return RoadProcessingResult(mesh=None, source_polygons=merged_roads, cutting_polygons=merged_roads) if return_result else None
    
    print(f"РЎС‚РІРѕСЂРµРЅРѕ {len(road_meshes)} РјРµС€С–РІ РґРѕСЂС–Рі")
    print(f"[STATS] Generated: Bridges={stats['bridge']}, Ground={stats['ground']}, Anti-Drown Activations={stats['anti_drown']}")
    
    # Roads already come from canonical 2D polygons, so preserve their footprint by
    # concatenating the generated meshes instead of running a 3D boolean union that
    # can bloat the XY outline near joints and building edges.
    print("РћР±'С”РґРЅР°РЅРЅСЏ РјРµС€С–РІ РґРѕСЂС–Рі...")
    try:
        mesh_candidates = []
        concat_candidate = trimesh.util.concatenate(road_meshes)
        if concat_candidate is not None and len(concat_candidate.vertices) > 0 and len(concat_candidate.faces) > 0:
            mesh_candidates.append(("concat", concat_candidate))

        if len(road_meshes) > 1:
            union_candidate = union_mesh_collection(road_meshes, label="roads")
            if union_candidate is not None and len(union_candidate.vertices) > 0 and len(union_candidate.faces) > 0:
                mesh_candidates.append(("union", union_candidate))

        if not mesh_candidates:
            combined_roads = None
        else:
            best_name = None
            best_score = None
            combined_roads = None
            for candidate_name, candidate_mesh in mesh_candidates:
                candidate = candidate_mesh
                if merged_roads is not None and not getattr(merged_roads, "is_empty", True):
                    candidate = _clip_mesh_to_road_footprint(candidate, merged_roads)
                try:
                    candidate = _cleanup_road_mesh(candidate)
                except Exception:
                    pass
                if candidate is None or len(candidate.vertices) == 0 or len(candidate.faces) == 0:
                    continue
                score = _road_mesh_candidate_score(candidate, merged_roads)
                if best_score is None or score > best_score:
                    best_score = score
                    best_name = candidate_name
                    combined_roads = candidate
            if best_name is not None:
                print(f"[ROAD] Selected {best_name} combined mesh candidate")

        if combined_roads is None:
            combined_roads = union_mesh_collection(road_meshes, label="roads")

        try:
            combined_roads = _cleanup_road_mesh(combined_roads)
            if len(combined_roads.faces) > 0:
                road_color = np.array([40, 40, 40, 255], dtype=np.uint8)
                combined_roads.visual = trimesh.visual.ColorVisuals(
                    face_colors=np.tile(road_color, (len(combined_roads.faces), 1))
                )
        except Exception:
            pass

        print(f"Р”РѕСЂРѕРіРё РѕР±'С”РґРЅР°РЅРѕ: {len(combined_roads.vertices)} РІРµСЂС€РёРЅ, {len(combined_roads.faces)} РіСЂР°РЅРµР№")
        
        # РџРѕРєСЂР°С‰РµРЅРЅСЏ mesh РґР»СЏ 3D РїСЂРёРЅС‚РµСЂР°
        # DISABLED: This removes small faces after scaling, which deletes bridge geometry!
        # print("РџРѕРєСЂР°С‰РµРЅРЅСЏ СЏРєРѕСЃС‚С– mesh РґР»СЏ 3D РїСЂРёРЅС‚РµСЂР° (Standard Mode)...")
        # combined_roads = improve_mesh_for_3d_printing(combined_roads, aggressive=False)
        
        
        # РџРµСЂРµРІС–СЂРєР° СЏРєРѕСЃС‚С–
        # DISABLED: This reports warnings about small faces that are expected after scaling
        # is_valid, mesh_warnings = validate_mesh_for_3d_printing(combined_roads)
        # if mesh_warnings:
        #     print(f"[INFO] РџРѕРїРµСЂРµРґР¶РµРЅРЅСЏ С‰РѕРґРѕ СЏРєРѕСЃС‚С– mesh РґРѕСЂС–Рі:")
        #     for w in mesh_warnings:
        #         print(f"  - {w}")
        
        
        if return_result:
            return RoadProcessingResult(
                mesh=combined_roads,
                source_polygons=merged_roads,
                cutting_polygons=merged_roads,
            )
        return combined_roads
    except Exception as e:
        print(f"РџРѕРјРёР»РєР° РѕР±'С”РґРЅР°РЅРЅСЏ РґРѕСЂС–Рі: {e}")
        # РџРѕРІРµСЂС‚Р°С”РјРѕ РїРµСЂС€РёР№ РјРµС€ СЏРєС‰Рѕ РЅРµ РІРґР°Р»РѕСЃСЏ РѕР±'С”РґРЅР°С‚Рё
        if road_meshes:
            if return_result:
                return RoadProcessingResult(
                    mesh=road_meshes[0],
                    source_polygons=merged_roads,
                    cutting_polygons=merged_roads,
                )
            return road_meshes[0]
        return RoadProcessingResult(mesh=None, source_polygons=merged_roads, cutting_polygons=merged_roads) if return_result else None


