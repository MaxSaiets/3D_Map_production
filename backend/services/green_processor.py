"""
Green areas (parks/forests/grass) processor.

Creates a thin embossed mesh that is draped onto terrain:
new_z = ground_z + old_z - embed

This makes parks/green areas stand out visually and be printable (has thickness).
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon, box, Point
from shapely.ops import transform, unary_union

from services.terrain_provider import TerrainProvider
from services.global_center import GlobalCenter
from services.geometry_context import clean_geometry, clip_geometry, to_local_geodataframe_if_needed, to_local_geometry_if_needed
from services.detail_layer_utils import MICRO_REGION_THRESHOLD_MM, model_mm_to_world_m
from services.mesh_triangulation import extrude_polygon_uniform
from services.processing_results import GreenAreaProcessingResult


def _polygon_min_dimension(poly: Polygon) -> float:
    try:
        minx, miny, maxx, maxy = poly.bounds
        return float(min(maxx - minx, maxy - miny))
    except Exception:
        return 0.0


def _polygon_meets_micro_threshold(poly: Polygon, min_feature_m: float) -> bool:
    if poly is None or poly.is_empty:
        return False
    if min_feature_m <= 0:
        return True
    try:
        min_dim = _polygon_min_dimension(poly)
        area = float(getattr(poly, "area", 0.0) or 0.0)
        min_area_m2 = max((float(min_feature_m) ** 2) * 0.5, 1e-8)
        return min_dim >= float(min_feature_m) and area >= min_area_m2
    except Exception:
        return True


def _recover_park_geometry(
    *,
    original_geom,
    current_geom,
    blocker_geom,
    min_feature_m: float,
):
    if original_geom is None or getattr(original_geom, "is_empty", True):
        return current_geom
    try:
        legal_geom = original_geom.difference(blocker_geom) if blocker_geom is not None and not getattr(blocker_geom, "is_empty", True) else original_geom
    except Exception:
        legal_geom = original_geom
    try:
        legal_geom = clean_geometry(legal_geom)
    except Exception:
        pass
    if legal_geom is None or getattr(legal_geom, "is_empty", True):
        return current_geom
    if current_geom is None or getattr(current_geom, "is_empty", True):
        return legal_geom
    try:
        legal_area = float(getattr(legal_geom, "area", 0.0) or 0.0)
        current_area = float(getattr(current_geom, "area", 0.0) or 0.0)
        if legal_area > 0 and current_area < legal_area * 0.985:
            return legal_geom
    except Exception:
        pass

    try:
        missing_geom = legal_geom.difference(current_geom)
    except Exception:
        return current_geom
    if missing_geom is None or getattr(missing_geom, "is_empty", True):
        return current_geom

    near_parts = []
    proximity = max(float(min_feature_m), 0.0)
    candidates = []
    if isinstance(missing_geom, Polygon):
        candidates = [missing_geom]
    elif isinstance(missing_geom, MultiPolygon):
        candidates = list(missing_geom.geoms)
    elif hasattr(missing_geom, "geoms"):
        try:
            candidates = [geom for geom in missing_geom.geoms if isinstance(geom, Polygon)]
        except Exception:
            candidates = []
    for part in candidates:
        if part is None or getattr(part, "is_empty", True):
            continue
        if _polygon_meets_micro_threshold(part, min_feature_m):
            continue
        try:
            if current_geom.intersects(part) or (proximity > 0 and current_geom.buffer(proximity).intersects(part)):
                near_parts.append(part)
        except Exception:
            continue
    if not near_parts:
        return current_geom
    try:
        return current_geom.union(unary_union(near_parts)).buffer(0)
    except Exception:
        return current_geom


def _iter_polygon_parts(geom) -> list[Polygon]:
    if geom is None or getattr(geom, "is_empty", False):
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [poly for poly in geom.geoms if isinstance(poly, Polygon) and not poly.is_empty]
    if hasattr(geom, "geoms"):
        try:
            return [poly for poly in geom.geoms if isinstance(poly, Polygon) and not poly.is_empty]
        except Exception:
            return []
    return []


def _safe_distance(lhs, rhs) -> float:
    if lhs is None or getattr(lhs, "is_empty", False) or rhs is None or getattr(rhs, "is_empty", False):
        return float("inf")
    try:
        return float(lhs.distance(rhs))
    except Exception:
        return float("inf")


def _absorb_tiny_zone_remainder_into_target(
    *,
    target_geom,
    zone_polygon_local,
    road_polygons=None,
    water_polygons=None,
    building_polygons=None,
    min_feature_m: float,
):
    if (
        target_geom is None
        or getattr(target_geom, "is_empty", True)
        or zone_polygon_local is None
        or getattr(zone_polygon_local, "is_empty", True)
        or min_feature_m <= 0
    ):
        return target_geom

    blockers = [target_geom]
    for geom in (road_polygons, water_polygons, building_polygons):
        if geom is not None and not getattr(geom, "is_empty", True):
            blockers.append(geom)

    try:
        occupied = unary_union(blockers).buffer(0)
        remainder = zone_polygon_local.difference(occupied)
    except Exception:
        return target_geom

    if remainder is None or getattr(remainder, "is_empty", True):
        return target_geom

    absorb_parts: list[Polygon] = []
    proximity_limit = max(float(min_feature_m) * 1.6, 0.05)
    for part in _iter_polygon_parts(remainder):
        if part is None or part.is_empty:
            continue
        if _polygon_meets_micro_threshold(part, min_feature_m):
            continue

        target_distance = _safe_distance(part, target_geom)
        if not np.isfinite(target_distance) or target_distance > proximity_limit:
            continue

        if _safe_distance(part, road_polygons) <= target_distance + 1e-6:
            continue
        if _safe_distance(part, water_polygons) <= target_distance + 1e-6:
            continue
        if _safe_distance(part, building_polygons) <= target_distance + 1e-6:
            continue

        absorb_parts.append(part)

    if not absorb_parts:
        return target_geom

    try:
        return unary_union([target_geom, *absorb_parts]).buffer(0)
    except Exception:
        return target_geom


def _polygon_equivalent_width(poly: Polygon) -> float:
    if poly is None or poly.is_empty:
        return 0.0
    try:
        area = float(getattr(poly, "area", 0.0) or 0.0)
        perimeter = float(getattr(poly, "length", 0.0) or 0.0)
        if perimeter <= 0.0:
            return 0.0
        return float((2.0 * area) / perimeter)
    except Exception:
        return 0.0


def _polygon_meets_land_threshold(poly: Polygon, min_feature_m: float) -> bool:
    if not _polygon_meets_micro_threshold(poly, min_feature_m):
        return False
    if min_feature_m <= 0:
        return True
    equiv_width = _polygon_equivalent_width(poly)
    return equiv_width >= float(min_feature_m) * 0.75


def _fill_tiny_park_voids(
    park_geom,
    *,
    min_feature_m: float,
    preserve_geom=None,
):
    if park_geom is None or getattr(park_geom, "is_empty", True) or min_feature_m <= 0:
        return park_geom

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
                    if hole_poly.intersects(preserve_buffer):
                        kept_interiors.append(interior)
                        continue
                except Exception:
                    pass
            try:
                minx, miny, maxx, maxy = hole_poly.bounds
                min_dim = min(float(maxx - minx), float(maxy - miny))
                area = float(getattr(hole_poly, "area", 0.0) or 0.0)
                equiv_width = _polygon_equivalent_width(hole_poly)
            except Exception:
                kept_interiors.append(interior)
                continue
            max_area_m2 = max((float(min_feature_m) ** 2) * 18.0, 1e-8)
            if (
                min_dim < (float(min_feature_m) * 1.05)
                or equiv_width < float(min_feature_m) * 0.80
            ) and area <= max_area_m2:
                continue
            kept_interiors.append(interior)
        try:
            return Polygon(poly.exterior.coords, holes=kept_interiors).buffer(0)
        except Exception:
            return poly

    polys = [_fill_for_polygon(poly) for poly in _iter_polygon_parts(park_geom)]
    polys = [poly for poly in polys if poly is not None and not poly.is_empty]
    if not polys:
        return None
    try:
        return unary_union(polys).buffer(0)
    except Exception:
        return polys[0]


def _opened_geometry(geom, radius_m: float):
    if geom is None or getattr(geom, "is_empty", True) or radius_m <= 0:
        return geom
    try:
        opened = geom.buffer(-float(radius_m), join_style=1)
        if opened is None or getattr(opened, "is_empty", True):
            return None
        opened = opened.buffer(float(radius_m), join_style=1)
        return clean_geometry(opened)
    except Exception:
        return clean_geometry(geom)


def _normalize_land_partition(
    *,
    parks_geom,
    legal_land,
    zone_polygon_local,
    park_intent_geom=None,
    min_feature_m: float,
):
    if (
        parks_geom is None
        or getattr(parks_geom, "is_empty", True)
        or legal_land is None
        or getattr(legal_land, "is_empty", True)
        or min_feature_m <= 0
    ):
        return parks_geom

    radius_m = float(min_feature_m) * 0.5
    if radius_m <= 0:
        return parks_geom

    try:
        parks_geom = clean_geometry(parks_geom.intersection(legal_land))
    except Exception:
        parks_geom = clean_geometry(parks_geom)
    if parks_geom is None or getattr(parks_geom, "is_empty", True):
        return parks_geom

    try:
        terrain_geom = clean_geometry(legal_land.difference(parks_geom))
    except Exception:
        terrain_geom = None

    park_core = _opened_geometry(parks_geom, radius_m)
    terrain_core = _opened_geometry(terrain_geom, radius_m)

    assigned_parts = [
        geom for geom in (park_core, terrain_core)
        if geom is not None and not getattr(geom, "is_empty", True)
    ]
    try:
        assigned_union = unary_union(assigned_parts).buffer(0) if assigned_parts else None
    except Exception:
        assigned_union = assigned_parts[0] if assigned_parts else None

    try:
        remainder_geom = legal_land.difference(assigned_union) if assigned_union is not None else legal_land
        remainder_geom = clean_geometry(remainder_geom)
    except Exception:
        remainder_geom = None

    park_parts: list[Polygon] = _iter_polygon_parts(park_core)
    terrain_core_geom = terrain_core

    intent_buffer = None
    if park_intent_geom is not None and not getattr(park_intent_geom, "is_empty", True):
        try:
            intent_buffer = park_intent_geom.buffer(radius_m, join_style=1).buffer(0)
        except Exception:
            intent_buffer = park_intent_geom

    boundary_geom = None
    try:
        boundary_geom = zone_polygon_local.boundary if zone_polygon_local is not None else None
    except Exception:
        boundary_geom = None

    for part in _iter_polygon_parts(remainder_geom):
        if part is None or part.is_empty:
            continue
        park_score = 0.0
        terrain_score = 0.0

        if park_core is not None and not getattr(park_core, "is_empty", True):
            try:
                if part.intersects(park_core):
                    park_score += 2.0
                park_score += max(0.0, radius_m - float(part.distance(park_core)))
            except Exception:
                pass
        if terrain_core_geom is not None and not getattr(terrain_core_geom, "is_empty", True):
            try:
                if part.intersects(terrain_core_geom):
                    terrain_score += 2.0
                terrain_score += max(0.0, radius_m - float(part.distance(terrain_core_geom)))
            except Exception:
                pass

        try:
            overlap_with_parks = float(getattr(part.intersection(parks_geom), "area", 0.0) or 0.0)
            if float(getattr(part, "area", 0.0) or 0.0) > 0:
                park_score += overlap_with_parks / float(part.area)
        except Exception:
            pass

        if intent_buffer is not None:
            try:
                overlap = part.intersection(intent_buffer)
                overlap_area = float(getattr(overlap, "area", 0.0) or 0.0)
                part_area = float(getattr(part, "area", 0.0) or 0.0)
                overlap_ratio = (overlap_area / part_area) if part_area > 0.0 else 0.0
                if part.centroid.within(intent_buffer):
                    park_score += 2.5
                park_score += overlap_ratio * 2.0
            except Exception:
                pass

        if boundary_geom is not None:
            try:
                if part.intersects(boundary_geom):
                    terrain_score += 2.5
            except Exception:
                pass

        if park_score > terrain_score:
            park_parts.append(part)

    if not park_parts:
        return None
    try:
        result = clean_geometry(unary_union(park_parts).buffer(0))
    except Exception:
        result = clean_geometry(park_parts[0])
    try:
        result = clean_geometry(result.intersection(legal_land))
    except Exception:
        pass
    return result


def _apply_fit_inset_to_green_mask(
    geometry,
    *,
    fit_clearance_m: float,
    min_feature_m: float,
):
    if (
        geometry is None
        or getattr(geometry, "is_empty", True)
        or fit_clearance_m <= 0
    ):
        return geometry
    try:
        inset = geometry.buffer(-float(fit_clearance_m), join_style=1)
        inset = clean_geometry(inset)
    except Exception:
        inset = geometry
    if inset is None or getattr(inset, "is_empty", True):
        return geometry

    kept: list[Polygon] = []
    for poly in _iter_polygon_parts(inset):
        if poly is None or poly.is_empty:
            continue
        if not _polygon_meets_land_threshold(poly, min_feature_m):
            continue
        kept.append(poly)
    if not kept:
        return geometry
    try:
        return clean_geometry(unary_union(kept).buffer(0))
    except Exception:
        return clean_geometry(kept[0])


def _finalize_land_masks(
    *,
    parks_geom,
    park_intent_geom=None,
    zone_polygon_local,
    road_polygons=None,
    water_polygons=None,
    building_polygons=None,
    min_feature_m: float,
):
    if (
        parks_geom is None
        or getattr(parks_geom, "is_empty", True)
        or zone_polygon_local is None
        or getattr(zone_polygon_local, "is_empty", True)
        or min_feature_m <= 0
    ):
        return parks_geom

    blockers = [
        geom for geom in (road_polygons, water_polygons, building_polygons)
        if geom is not None and not getattr(geom, "is_empty", True)
    ]
    blocker_union = None
    if blockers:
        try:
            blocker_union = unary_union(blockers).buffer(0)
        except Exception:
            blocker_union = None

    try:
        legal_land = zone_polygon_local.difference(blocker_union) if blocker_union is not None else zone_polygon_local
        legal_land = clean_geometry(legal_land)
    except Exception:
        legal_land = zone_polygon_local
    if legal_land is None or getattr(legal_land, "is_empty", True):
        return parks_geom

    try:
        parks_geom = clean_geometry(parks_geom.intersection(legal_land))
    except Exception:
        parks_geom = clean_geometry(parks_geom)
    if parks_geom is None or getattr(parks_geom, "is_empty", True):
        return parks_geom

    parks_geom = _fill_tiny_park_voids(
        parks_geom,
        min_feature_m=float(min_feature_m),
        preserve_geom=blocker_union,
    )
    if parks_geom is None or getattr(parks_geom, "is_empty", True):
        return parks_geom

    try:
        terrain_geom = clean_geometry(legal_land.difference(parks_geom))
    except Exception:
        terrain_geom = None

    park_parts = _iter_polygon_parts(parks_geom)
    terrain_parts = _iter_polygon_parts(terrain_geom)

    large_park_parts = [part for part in park_parts if _polygon_meets_land_threshold(part, min_feature_m)]
    small_park_parts = [part for part in park_parts if not _polygon_meets_land_threshold(part, min_feature_m)]
    large_terrain_parts = [part for part in terrain_parts if _polygon_meets_land_threshold(part, min_feature_m)]
    small_terrain_parts = [part for part in terrain_parts if not _polygon_meets_land_threshold(part, min_feature_m)]

    try:
        large_park_union = unary_union(large_park_parts).buffer(0) if large_park_parts else None
    except Exception:
        large_park_union = None
    try:
        large_terrain_union = unary_union(large_terrain_parts).buffer(0) if large_terrain_parts else None
    except Exception:
        large_terrain_union = None

    proximity_limit = max(float(min_feature_m) * 1.35, 0.05)
    reassigned_to_parks: list[Polygon] = []
    retained_parks: list[Polygon] = list(large_park_parts)

    intent_buffer = None
    if park_intent_geom is not None and not getattr(park_intent_geom, "is_empty", True):
        try:
            intent_buffer = park_intent_geom.buffer(proximity_limit, join_style=1).buffer(0)
        except Exception:
            intent_buffer = park_intent_geom

    for part in small_park_parts:
        if part is None or part.is_empty:
            continue
        keep = False
        if large_park_union is not None:
            try:
                keep = part.intersects(large_park_union) or part.distance(large_park_union) <= proximity_limit
            except Exception:
                keep = False
        if keep:
            retained_parks.append(part)

    boundary_geom = None
    try:
        boundary_geom = zone_polygon_local.boundary
    except Exception:
        boundary_geom = None

    for part in small_terrain_parts:
        if part is None or part.is_empty:
            continue
        touches_boundary = False
        if boundary_geom is not None:
            try:
                touches_boundary = part.intersects(boundary_geom)
            except Exception:
                touches_boundary = False
        if touches_boundary:
            continue
        near_park = False
        near_terrain = False
        if large_park_union is not None:
            try:
                near_park = part.intersects(large_park_union) or part.distance(large_park_union) <= proximity_limit
            except Exception:
                near_park = False
        if large_terrain_union is not None:
            try:
                near_terrain = part.intersects(large_terrain_union) or part.distance(large_terrain_union) <= proximity_limit
            except Exception:
                near_terrain = False
        intent_support = False
        overlap_ratio = 0.0
        centroid_inside = False
        if intent_buffer is not None:
            try:
                overlap = part.intersection(intent_buffer)
                overlap_area = float(getattr(overlap, "area", 0.0) or 0.0)
                part_area = float(getattr(part, "area", 0.0) or 0.0)
                overlap_ratio = (overlap_area / part_area) if part_area > 0.0 else 0.0
                centroid_inside = part.centroid.within(intent_buffer)
                intent_support = (
                    centroid_inside
                    or overlap_ratio >= 0.35
                    or part.intersects(intent_buffer)
                    or part.distance(intent_buffer) <= proximity_limit
                )
            except Exception:
                intent_support = False
        should_merge_to_park = (
            (near_park and not near_terrain and intent_support)
            or centroid_inside
            or overlap_ratio >= 0.60
        )
        if should_merge_to_park:
            reassigned_to_parks.append(part)

    final_parts = [part for part in [*retained_parks, *reassigned_to_parks] if part is not None and not part.is_empty]
    if not final_parts:
        return None
    try:
        final_parks = unary_union(final_parts).buffer(0)
    except Exception:
        final_parks = final_parts[0]
    return _normalize_land_partition(
        parks_geom=final_parks,
        legal_land=legal_land,
        zone_polygon_local=zone_polygon_local,
        park_intent_geom=park_intent_geom,
        min_feature_m=float(min_feature_m),
    )


def _create_high_res_mesh(poly: Polygon, height_m: float, target_edge_len_m: float) -> Optional[trimesh.Trimesh]:
    """
    Створює меш з UNIFORM тріангуляцією (remeshed) використовуючи Steiner points.
    Виправляє проблему діагональних смуг, створюючи рівномірні трикутники.
    
    Підхід:
    1. Resample Boundary - додає точки на контур для точного накладання на рельєф
    2. Internal Grid - генерує рівномірну сітку всередині полігону
    3. Delaunay Triangulation - створює рівномірні трикутники
    4. Extrude - витягує в 3D з боковими стінками
    
    Args:
        poly: Вхідний полігон
        height_m: Висота екструзії
        target_edge_len_m: Цільова довжина ребра в метрах (максимальна)
    
    Returns:
        Trimesh об'єкт з високою деталізацією та рівномірною топологією
    """
    try:
        if poly is None or poly.is_empty:
            return None

        if target_edge_len_m <= 0:
            target_edge_len_m = 3.0

        # Use extrude_polygon_uniform for all polygons.
        # It handles both simple polygons (densified boundary for uniform triangulation)
        # and holed polygons (no densification to avoid non-manifold edges), and
        # cleans near-duplicate vertices before extruding.
        mesh = extrude_polygon_uniform(poly, height=float(height_m), densify_max_m=target_edge_len_m)
        if mesh is not None and len(mesh.vertices) > 0:
            return mesh
        return trimesh.creation.extrude_polygon(poly, height=float(height_m))

    except Exception as e:
        print(f"[WARN] _create_high_res_mesh failed: {e}")
        try:
            return trimesh.creation.extrude_polygon(poly, height=float(height_m))
        except Exception:
            return None


def process_green_areas(
    gdf_green: gpd.GeoDataFrame,
    height_m: float,
    embed_m: float,
    terrain_provider: Optional[TerrainProvider] = None,
    global_center: Optional[GlobalCenter] = None,  # UTM -> local
    scale_factor: Optional[float] = None,  # model_mm / world_m
    zone_polygon_local: Optional[object] = None,
    min_feature_mm: float = MICRO_REGION_THRESHOLD_MM,
    simplify_mm: float = 0.4,
    fit_clearance_mm: float = 0.0,
    # --- НОВИЙ АРГУМЕНТ: Полігони доріг для вирізання ---
    road_polygons: Optional[object] = None,  # Shapely Polygon/MultiPolygon об'єднаних доріг (в локальних координатах)
    water_polygons: Optional[object] = None,  # Водні полігони (в локальних координатах) мають пріоритет над парками
    # --- НОВИЙ АРГУМЕНТ: Полігони будівель для вирізання ---
    building_polygons: Optional[object] = None,  # Shapely Polygon/MultiPolygon об'єднаних будівель (в локальних координатах)
    return_result: bool = False,
) -> Optional[trimesh.Trimesh | GreenAreaProcessingResult]:
    if gdf_green is None or gdf_green.empty:
        return GreenAreaProcessingResult(mesh=None, processed_polygons=None) if return_result else None

    # --- Coordinate Transform Block ---
    # Перетворюємо UTM -> local тільки якщо координати виглядають як UTM (великі числа)
    if global_center is not None:
        try:
            gdf_green = to_local_geodataframe_if_needed(gdf_green, global_center)
        except Exception:
            pass

    # --- Road Mask Preparation (Підготовка маски доріг для вирізання) ---
    # Переконуємось, що полігон доріг теж в локальних координатах
    road_mask = road_polygons
    if road_mask is not None:
        # Перевіряємо, чи маска не порожня
        try:
            if getattr(road_mask, "is_empty", False):
                road_mask = None
        except Exception:
            pass
        
        # Перетворення координат (якщо потрібно)
        if road_mask is not None and global_center is not None:
            # Перевірка: якщо координати доріг виглядають як UTM (великі числа), а ми вже в local
            try:
                road_mask = to_local_geometry_if_needed(road_mask, global_center)
                if road_mask is None or getattr(road_mask, "is_empty", False):
                    road_mask = None
            except Exception as e:
                print(f"[WARN] Помилка перетворення road_polygons в локальні координати: {e}")
                # Якщо не вдалося перетворити, не використовуємо маску
                road_mask = None

    # --- Clipping Block ---
    clip_box = None
    if terrain_provider is not None:
        try:
            min_x, max_x, min_y, max_y = terrain_provider.get_bounds()
            clip_box = box(min_x, min_y, max_x, max_y)
        except Exception:
            clip_box = None

    # --- Parameters Calculation ---
    simplify_tol_m = 0.5
    min_width_m = None
    micro_feature_m = 0.0
    fit_clearance_m = 0.0
    target_edge_len_m = 4.0  # Базове значення (крупніші трикутники для Low-Poly фактури)
    
    if scale_factor is not None and float(scale_factor) > 0:
        try:
            simplify_tol_m = max(0.05, float(simplify_mm) / float(scale_factor))
        except Exception:
            pass
        try:
            min_width_m = model_mm_to_world_m(min_feature_mm, scale_factor)
            micro_feature_m = float(min_width_m)
        except Exception:
            pass
        try:
            fit_clearance_m = model_mm_to_world_m(fit_clearance_mm, scale_factor)
        except Exception:
            fit_clearance_m = 0.0
        
        # Для виразнішої Low-Poly текстури нам потрібні БІЛЬШІ трикутники (~3.5мм на моделі)
        try:
            target_edge_len_m = 3.5 / float(scale_factor)  # 3.5mm на моделі
            # Обмежуємо, щоб не повісити систему на великих картах
            target_edge_len_m = max(2.0, min(target_edge_len_m, 12.0))
        except Exception:
            pass

    # --- Polygon Cleaning & Collection ---
    polys: list[Polygon] = []
    intent_polys: list[Polygon] = []

    def _iter_polys(g):
        if g is None or getattr(g, "is_empty", False):
            return []
        if isinstance(g, Polygon):
            return [g]
        if isinstance(g, MultiPolygon):
            return list(g.geoms)
        if hasattr(g, "geoms"):
            return [gg for gg in g.geoms if isinstance(gg, Polygon)]
        return []

    for _, row in gdf_green.iterrows():
        geom = getattr(row, "geometry", None)
        if geom is None or getattr(geom, "is_empty", False):
            continue
        try:
            geom = clean_geometry(geom)
        except Exception:
            pass
        if geom is None or getattr(geom, "is_empty", False):
            continue

        if clip_box is not None:
            try:
                geom = clip_geometry(geom, clip_box)
            except Exception:
                continue
            if geom is None or getattr(geom, "is_empty", False):
                continue

        original_geom = geom
        for intent_poly in _iter_polys(original_geom):
            if intent_poly is None or getattr(intent_poly, "is_empty", False):
                continue
            try:
                intent_poly = clean_geometry(intent_poly)
            except Exception:
                pass
            if intent_poly is None or getattr(intent_poly, "is_empty", False):
                continue
            intent_polys.append(intent_poly)
        local_blockers = []

        # --- ROAD CLIPPING (ВИРІЗАННЯ ДОРІГ) ---
        # Це найважливіший момент: віднімаємо дороги від зеленої зони
        # Це запобігає z-fighting та перетину між дорогами та парками
        if road_mask is not None:
            try:
                # ОПТИМІЗАЦІЯ: Обрізаємо road_mask по bounds поточного парку перед вирізанням
                # Це значно прискорює Boolean операцію для великих MultiPolygon доріг
                geom_bounds = geom.bounds
                road_mask_clipped = road_mask
                
                # Створюємо bounding box для парку з невеликим padding (для безпеки)
                padding = 10.0  # 10 метрів padding для уникнення проблем на краях
                clip_box_geom = box(
                    geom_bounds[0] - padding,
                    geom_bounds[1] - padding,
                    geom_bounds[2] + padding,
                    geom_bounds[3] + padding
                )
                
                # Обрізаємо road_mask по bounds парку
                try:
                    road_mask_clipped = road_mask.intersection(clip_box_geom)
                    if road_mask_clipped is None or getattr(road_mask_clipped, "is_empty", False):
                        # Якщо дороги не перетинаються з цим парком, пропускаємо вирізання
                        pass
                    else:
                        # Вирізаємо дороги з парку (Boolean Difference)
                        local_blockers.append(road_mask_clipped)
                        geom = geom.difference(road_mask_clipped)
                except Exception:
                    # Якщо intersection не вдався, пробуємо без обрізання (повільніше, але надійніше)
                    local_blockers.append(road_mask)
                    geom = geom.difference(road_mask)
                    
            except Exception as e:
                print(f"[WARN] Помилка вирізання доріг із парку: {e}")
                # Якщо помилка, залишаємо як є (краще мати парк, ніж нічого)
                pass

            # Перевіряємо, чи геометрія не стала порожньою після вирізання
            if geom is None or getattr(geom, "is_empty", False):
                continue

        # --- WATER CLIPPING (ВИРІЗАННЯ ВОДИ) ---
        # Вода має вищий пріоритет за парки, інакше park insert лягає поверх водойм
        if water_polygons is not None:
            try:
                geom_bounds = geom.bounds
                water_mask_clipped = water_polygons
                padding = 5.0
                clip_box_geom = box(
                    geom_bounds[0] - padding,
                    geom_bounds[1] - padding,
                    geom_bounds[2] + padding,
                    geom_bounds[3] + padding
                )
                try:
                    water_mask_clipped = water_polygons.intersection(clip_box_geom)
                    if water_mask_clipped is not None and not getattr(water_mask_clipped, "is_empty", False):
                        local_blockers.append(water_mask_clipped)
                        geom = geom.difference(water_mask_clipped)
                except Exception:
                    local_blockers.append(water_polygons)
                    geom = geom.difference(water_polygons)
            except Exception as e:
                print(f"[WARN] Помилка вирізання води із парку: {e}")
                pass

            if geom is None or getattr(geom, "is_empty", False):
                continue

        # --- BUILDING CLIPPING (ВИРІЗАННЯ БУДІВЕЛЬ) ---
        # Це запобігає накладанню парків на будівлі
        if building_polygons is not None:
            try:
                # ОПТИМІЗАЦІЯ: Обрізаємо building_mask по bounds поточного парку
                geom_bounds = geom.bounds
                building_mask_clipped = building_polygons
                
                # Створюємо bounding box для парку з padding
                padding = 5.0  # 5 метрів padding
                clip_box_geom = box(
                    geom_bounds[0] - padding,
                    geom_bounds[1] - padding,
                    geom_bounds[2] + padding,
                    geom_bounds[3] + padding
                )
                
                # Обрізаємо building_mask по bounds парку
                try:
                    building_mask_clipped = building_polygons.intersection(clip_box_geom)
                    if building_mask_clipped is None or getattr(building_mask_clipped, "is_empty", False):
                        # Якщо будівлі не перетинаються з цим парком, пропускаємо
                        pass
                    else:
                        local_blockers.append(building_mask_clipped)
                        geom = geom.difference(building_mask_clipped)
                except Exception:
                    # Якщо intersection не вдався, пробуємо без обрізання
                    local_blockers.append(building_polygons)
                    geom = geom.difference(building_polygons)
                    
            except Exception as e:
                print(f"[WARN] Помилка вирізання будівель із парку: {e}")
                # Якщо помилка, залишаємо як є
                pass
            
            # Перевіряємо, чи геометрія не стала порожньою після вирізання будівель
            if geom is None or getattr(geom, "is_empty", False):
                continue

        blocker_union = None
        if local_blockers:
            try:
                blocker_union = unary_union(
                    [b for b in local_blockers if b is not None and not getattr(b, "is_empty", False)]
                ).buffer(0)
            except Exception:
                blocker_union = None
        geom = _recover_park_geometry(
            original_geom=original_geom,
            current_geom=geom,
            blocker_geom=blocker_union,
            min_feature_m=float(micro_feature_m),
        )

        for poly in _iter_polys(geom):
            if poly is None or getattr(poly, "is_empty", False):
                continue
            try:
                poly = clean_geometry(poly)
            except Exception:
                pass
            if poly is None or getattr(poly, "is_empty", False):
                continue

            if not _polygon_meets_micro_threshold(poly, micro_feature_m):
                continue
            polys.append(poly)

    if not polys:
        return GreenAreaProcessingResult(mesh=None, processed_polygons=None) if return_result else None

    # Union overlapping parks (після вирізання доріг)
    # ВАЖЛИВО: Union робимо ПІСЛЯ вирізання доріг, щоб уникнути проблем з об'єднанням
    try:
        merged = unary_union(polys)
        polys = []
        for p in _iter_polys(merged):
            if p is not None and not p.is_empty:
                if not _polygon_meets_micro_threshold(p, micro_feature_m):
                    continue
                polys.append(p)
    except Exception:
        pass

    # Filter & Simplify
    filtered: list[Polygon] = []
    for poly in polys:
        if poly is None or poly.is_empty:
            continue
        try:
            poly = poly.simplify(float(simplify_tol_m), preserve_topology=True)
        except Exception:
            pass
        if poly is None or poly.is_empty:
            continue

        if min_width_m is not None and float(min_width_m) > 0:
            # Simple check via area/perimeter ratio
            try:
                per = float(getattr(poly, "length", 0.0) or 0.0)
                area = float(getattr(poly, "area", 0.0) or 0.0)
                if per > 0 and area > 0:
                    equiv_width = float((2.0 * area) / per)
                    if equiv_width < float(min_width_m):
                        continue
            except Exception:
                pass
        # Після simplify park footprint може знову підрости у road/water/building voids.
        # Повторно застосовуємо ті ж пріоритети, щоб фінальний insert/groove брався вже з коректної маски.
        try:
            if road_mask is not None and poly is not None and not poly.is_empty:
                poly = poly.difference(road_mask)
        except Exception:
            pass
        try:
            if water_polygons is not None and poly is not None and not poly.is_empty:
                poly = poly.difference(water_polygons)
        except Exception:
            pass
        try:
            if building_polygons is not None and poly is not None and not poly.is_empty:
                poly = poly.difference(building_polygons)
        except Exception:
            pass

        if poly is None or poly.is_empty:
            continue
        for sub_poly in _iter_polys(poly):
            if sub_poly is None or sub_poly.is_empty:
                continue
            if not _polygon_meets_micro_threshold(sub_poly, micro_feature_m):
                continue
            filtered.append(sub_poly)

    if not filtered:
        return GreenAreaProcessingResult(mesh=None, processed_polygons=None) if return_result else None

    processed_polygons = None
    park_intent_geom = None
    try:
        processed_polygons = unary_union(filtered)
        if processed_polygons is not None and getattr(processed_polygons, "is_empty", False):
            processed_polygons = None
    except Exception:
        processed_polygons = filtered[0] if filtered else None

    try:
        if intent_polys:
            park_intent_geom = unary_union(intent_polys).buffer(0)
            if zone_polygon_local is not None and not getattr(zone_polygon_local, "is_empty", True):
                park_intent_geom = clean_geometry(park_intent_geom.intersection(zone_polygon_local))
    except Exception:
        park_intent_geom = None

    processed_polygons = _absorb_tiny_zone_remainder_into_target(
        target_geom=processed_polygons,
        zone_polygon_local=zone_polygon_local,
        road_polygons=road_mask,
        water_polygons=water_polygons,
        building_polygons=building_polygons,
        min_feature_m=float(micro_feature_m),
    )
    processed_polygons = _finalize_land_masks(
        parks_geom=processed_polygons,
        park_intent_geom=park_intent_geom,
        zone_polygon_local=zone_polygon_local,
        road_polygons=road_mask,
        water_polygons=water_polygons,
        building_polygons=building_polygons,
        min_feature_m=float(micro_feature_m),
    )
    if processed_polygons is not None and not getattr(processed_polygons, "is_empty", False):
        reassigned_filtered: list[Polygon] = []
        for sub_poly in _iter_polygon_parts(processed_polygons):
            if sub_poly is None or sub_poly.is_empty:
                continue
            if not _polygon_meets_land_threshold(sub_poly, micro_feature_m):
                continue
            reassigned_filtered.append(sub_poly)
        if reassigned_filtered:
            filtered = reassigned_filtered
            processed_polygons = clean_geometry(unary_union(reassigned_filtered).buffer(0))

    processed_polygons = _apply_fit_inset_to_green_mask(
        processed_polygons,
        fit_clearance_m=float(fit_clearance_m),
        min_feature_m=float(micro_feature_m),
    )
    if processed_polygons is not None and not getattr(processed_polygons, "is_empty", False):
        inset_filtered: list[Polygon] = []
        for sub_poly in _iter_polygon_parts(processed_polygons):
            if sub_poly is None or sub_poly.is_empty:
                continue
            if not _polygon_meets_land_threshold(sub_poly, micro_feature_m):
                continue
            inset_filtered.append(sub_poly)
        if inset_filtered:
            filtered = inset_filtered

    # --- MESH GENERATION ---
    meshes: list[trimesh.Trimesh] = []
    for poly in filtered:
        try:
            # Перевіряємо розмір зони перед створенням мешу
            area = float(poly.area) if hasattr(poly, 'area') else 0.0
            is_small_poly = False
            relative_height = None
            if area > 0:
                equivalent_radius = np.sqrt(area / np.pi)
                if equivalent_radius * 2 < 3.0:  # Діаметр менше 3м
                    is_small_poly = True
                    print(f"[DEBUG] Створення мешу для маленької зони: площа={area:.2f}m2, діаметр={equivalent_radius*2:.2f}м")
            
            # 1. Створення високодеталізованого мешу
            mesh = _create_high_res_mesh(poly, float(height_m), target_edge_len_m)

            # ВИПРАВЛЕННЯ: Для дуже маленьких зон використовуємо extrude з денсифікацією (рівномірна тріангуляція)
            if (mesh is None or len(mesh.vertices) == 0) and is_small_poly:
                print(f"[WARN] _create_high_res_mesh не створив меш для маленької зони, використовуємо extrude_polygon_uniform")
                try:
                    mesh = extrude_polygon_uniform(poly, height=float(height_m), densify_max_m=1.0)
                    if mesh is None:
                        mesh = trimesh.creation.extrude_polygon(poly, height=float(height_m))
                    if mesh is not None and len(mesh.vertices) > 0:
                        print(f"[DEBUG] Простий extrude створив меш: {len(mesh.vertices)} вершин")
                except Exception as e:
                    print(f"[WARN] Простий extrude також не вдався: {e}")
                    continue
            
            if mesh is None or len(mesh.vertices) == 0:
                if is_small_poly:
                    print(f"[WARN] Меш не створено для маленької зони (площа={area:.2f}m2)")
                continue

            if is_small_poly:
                print(f"[DEBUG] Меш створено для маленької зони: {len(mesh.vertices)} вершин, {len(mesh.faces)} граней")

            # 2. Накладання на рельєф (Draping)
            if terrain_provider is not None:
                v = mesh.vertices.copy()
                old_z = v[:, 2].copy()
                ground_heights = terrain_provider.get_surface_heights_for_points(v[:, :2])
                
                z_min = float(np.min(old_z))
                z_max = float(np.max(old_z))
                z_range = z_max - z_min
                
                relative_height = np.zeros_like(old_z)
                if z_range > 1e-6:
                    relative_height = (old_z - z_min) / z_range
                
                # Занурюємо парки в рельєф на embed_m, щоб вони могли вирізати паз
                new_z = ground_heights - float(embed_m) + relative_height * float(height_m + embed_m)
                
                v[:, 2] = new_z
                mesh.vertices = v
                
                # ВИПРАВЛЕННЯ: Зберігаємо маску дна для цього конкретного мешу, 
                # поки ми точно знаємо, де знаходяться нижні вершини (relative_height ? 0).
                # Це безпечно і точно розділяє дно від верху і боків незалежно від схилу.
                mesh._bottom_mask = (relative_height <= 0.1)
        
            # 3. Додавання текстури з маскою країв
            if is_small_poly:
                print(f"[DEBUG] Застосування текстури для маленької зони...")
            mesh = _add_strong_faceted_texture(mesh, height_m, scale_factor, original_polygon=poly, global_center=global_center, relative_heights=relative_height)
            if is_small_poly:
                print(f"[DEBUG] Текстура застосована для маленької зони")

            if len(mesh.faces) > 0:
                meshes.append(mesh)

        except Exception as e:
            print(f"[WARN] Помилка обробки полігону: {e}")
            continue

    if not meshes:
        return GreenAreaProcessingResult(mesh=None, processed_polygons=processed_polygons) if return_result else None

    # За запитом: абсолютне плоске дно для всіх зелених зон
    # Знаходимо найнижчу точку (global_min_z) серед УСІХ парків
    global_min_z = min(float(np.min(m.vertices[:, 2])) for m in meshes)
    flattened_count = 0
    
    for m in meshes:
        if hasattr(m, '_bottom_mask'):
            # Опускаємо всі нижні вершини до цієї єдиної найнижчої точки
            m.vertices[m._bottom_mask, 2] = global_min_z
            flattened_count += np.sum(m._bottom_mask)
            
    print(f"[INFO] Parks absolute global bottom flattened: {flattened_count} vertices > Z={global_min_z:.4f}")

    try:
        combined_mesh = trimesh.util.concatenate(meshes)
        if return_result:
            return GreenAreaProcessingResult(mesh=combined_mesh, processed_polygons=processed_polygons)
        return combined_mesh
    except Exception:
        if return_result:
            return GreenAreaProcessingResult(mesh=meshes[0], processed_polygons=processed_polygons)
        return meshes[0]


def _add_strong_faceted_texture(
    mesh: trimesh.Trimesh, 
    height_m: float, 
    scale_factor: Optional[float] = None,
    original_polygon: Optional[Polygon] = None,
    global_center: Optional[GlobalCenter] = None,
    relative_heights: Optional[np.ndarray] = None,
) -> trimesh.Trimesh:
    """
    Adds a seam-stable faceted park texture using world-anchored coordinates.
    Only top vertices are modified and boundary fade keeps insert fit stable.
    """
    if mesh is None or len(mesh.vertices) == 0 or scale_factor is None or scale_factor <= 0:
        return mesh
    if original_polygon is None or original_polygon.is_empty:
        return mesh

    try:
        verts = mesh.vertices.copy()
        old_z = verts[:, 2].copy()
        if relative_heights is None or len(relative_heights) != len(verts):
            z_min = float(np.min(old_z))
            z_max = float(np.max(old_z))
            z_range = max(z_max - z_min, 1e-9)
            relative_heights = np.clip((old_z - z_min) / z_range, 0.0, 1.0)

        top_mask = relative_heights >= 0.45
        if not np.any(top_mask):
            return mesh

        amp_m = max(model_mm_to_world_m(0.46, scale_factor), float(height_m) * 0.10)
        amp_m = min(amp_m, max(float(height_m) * 0.35, model_mm_to_world_m(0.62, scale_factor)))
        boundary_fade_m = max(model_mm_to_world_m(0.8, scale_factor), amp_m * 3.2)
        frequency_m = max(model_mm_to_world_m(2.4, scale_factor), 1.6)

        verts_top = verts[top_mask]
        x_local = verts_top[:, 0]
        y_local = verts_top[:, 1]
        if global_center is not None:
            cx, cy = global_center.get_center_utm()
            x_world = x_local + float(cx)
            y_world = y_local + float(cy)
        else:
            x_world = x_local
            y_world = y_local

        noise = (
            np.sin((x_world / frequency_m) * 1.87 + 0.31)
            + np.cos((y_world / frequency_m) * 2.41 - 0.57)
            + np.sin(((x_world + y_world) / frequency_m) * 1.19 + 1.11)
        ) / 3.0

        fade = np.ones(len(verts_top), dtype=float)
        try:
            boundary = original_polygon.boundary
            distances = np.asarray(
                [float(boundary.distance(Point(x, y))) for x, y in zip(x_local, y_local)],
                dtype=float,
            )
            fade = np.clip(distances / max(boundary_fade_m, 1e-9), 0.0, 1.0)
        except Exception:
            pass

        height_bias = np.clip((relative_heights[top_mask] - 0.45) / 0.55, 0.0, 1.0)
        offsets = noise * fade * height_bias * amp_m
        verts[top_mask, 2] = verts[top_mask, 2] + offsets
        mesh.vertices = verts
    except Exception as exc:
        print(f"[WARN] Parks texture application failed: {exc}")
    return mesh
