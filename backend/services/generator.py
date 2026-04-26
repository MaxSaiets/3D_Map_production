"""
3D Map Generation Pipeline — чистий, лінійний pipeline.

Кроки:
  1.  Система координат (UTM → local)
  2.  Завантаження OSM даних
  3.  Конвертація геометрій у локальні координати
  4.  Полігони доріг
  5.  Генерація рельєфу (DEM)
  6.  Меш доріг + вирівнювання дна
  7.  Вирізання пазів для доріг (Blender boolean)
  8.  Меш парків
  9.  Вирізання пазів для парків (Blender boolean)
  10. Будівлі
  11. Вода
  12. Експорт
"""

from __future__ import annotations

import gc
import concurrent.futures
import traceback
from dataclasses import dataclass, field
from typing import Optional, Callable, List

import numpy as np
import trimesh
import geopandas as gpd
from shapely.ops import unary_union, transform as shapely_transform
from shapely.geometry import box as shapely_box

from services.global_center import GlobalCenter, set_global_center
from services.crs_utils import bbox_latlon_to_utm
from services.data_loader import fetch_city_data
from services.extras_loader import fetch_extras
from services.terrain_generator import create_terrain_mesh
from services.road_processor import process_roads, build_road_polygons, merge_close_road_gaps
from services.building_processor import process_buildings
from services.water_processor import process_water_surface
from services.green_processor import process_green_areas
from services.terrain_cutter import (
    cut_roads_from_solid_terrain,
    extend_road_mesh_to_uniform_bottom,
    cut_parks_from_solid_terrain,
    extend_parks_mesh_to_uniform_bottom,
)
from services.model_exporter import export_scene

# Зазор пазу по боках (XY): 0.15 мм з кожного боку
GROOVE_CLEARANCE_MM = 0.15
# Мінімальна ширина друкованого проміжку між дорогами (мм)
MIN_PRINTABLE_GAP_MM = 1.5


# ---------------------------------------------------------------------------
# Параметри генерації
# ---------------------------------------------------------------------------

@dataclass
class GenerationParams:
    # Bounding box
    north: float
    south: float
    east: float
    west: float

    # Модель
    model_size_mm: float = 80.0
    export_format: str = "3mf"  # "stl" або "3mf"

    # Рельєф
    terrain_resolution: int = 300
    terrain_z_scale: float = 3.0
    terrain_base_thickness_mm: float = 0.3
    terrain_smoothing_sigma: float = 2.0
    terrarium_zoom: int = 15
    terrain_subdivide: bool = True
    terrain_subdivide_levels: int = 1

    # Дороги
    road_width_multiplier: float = 1.0
    road_height_mm: float = 0.5   # висота дороги над рельєфом
    road_embed_mm: float = 0.3    # глибина врізання дороги в рельєф

    # Будівлі
    building_min_height: float = 2.0
    building_height_multiplier: float = 1.0
    building_foundation_mm: float = 0.6
    building_embed_mm: float = 0.2
    building_max_foundation_mm: float = 2.5

    # Парки
    include_parks: bool = True
    parks_height_mm: float = 0.6
    parks_embed_mm: float = 1.0

    # Вода
    water_depth_mm: float = 1.2


# ---------------------------------------------------------------------------
# Допоміжні функції
# ---------------------------------------------------------------------------

def _to_local_transformer(global_center: GlobalCenter):
    """Повертає функцію для shapely_transform: UTM coords → local coords."""
    cx = global_center.center_x_utm
    cy = global_center.center_y_utm

    def _transform(x, y, z=None):
        x_local = np.asarray(x) - cx
        y_local = np.asarray(y) - cy
        if z is not None:
            return x_local, y_local, z
        return x_local, y_local

    return _transform


def _convert_gdf_to_local(
    gdf: gpd.GeoDataFrame,
    to_local,
) -> gpd.GeoDataFrame:
    """Конвертує всі геометрії GDF у локальні координати."""
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(
        lambda g: shapely_transform(to_local, g)
        if g is not None and not g.is_empty
        else g
    )
    return gdf


def _safe_unary_union(geoms) -> Optional[object]:
    """Безпечний unary_union, повертає None при помилці."""
    valid = [g for g in geoms if g is not None and not getattr(g, "is_empty", True)]
    if not valid:
        return None
    try:
        return unary_union(valid)
    except Exception as e:
        print(f"[WARN] unary_union failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Головна функція генерації
# ---------------------------------------------------------------------------

def generate_3d_map(
    params: GenerationParams,
    output_path: str,
    status_fn: Optional[Callable[[int, str], None]] = None,
) -> str:
    """
    Генерує 3D карту з пазами для доріг та парків.

    Args:
        params: Параметри генерації.
        output_path: Шлях до вихідного файлу (*.3mf або *.stl).
        status_fn: Колбек (percent, message) для відслідковування прогресу.

    Returns:
        Абсолютний шлях до створеного файлу.
    """

    def status(pct: int, msg: str):
        print(f"[{pct:3d}%] {msg}")
        if status_fn:
            try:
                status_fn(pct, msg)
            except Exception:
                pass

    # ─── КРОК 1: Система координат ──────────────────────────────────────────
    status(5, "Ініціалізація системи координат...")

    latlon_bbox = (params.north, params.south, params.east, params.west)

    global_center = set_global_center(
        (params.north + params.south) / 2.0,
        (params.east + params.west) / 2.0,
    )

    # Конвертуємо bbox у локальні метри
    bbox_utm = bbox_latlon_to_utm(params.north, params.south, params.east, params.west)
    minx_utm, miny_utm, maxx_utm, maxy_utm = bbox_utm[:4]
    minx_local, miny_local = global_center.to_local(minx_utm, miny_utm)
    maxx_local, maxy_local = global_center.to_local(maxx_utm, maxy_utm)
    bbox_meters = (float(minx_local), float(miny_local), float(maxx_local), float(maxy_local))

    # Scale factor: мм/м (скільки міліметрів на моделі = 1 метр у реальності)
    zone_w = maxx_local - minx_local
    zone_h = maxy_local - miny_local
    zone_size_m = max(zone_w, zone_h, 1.0)
    scale_factor = params.model_size_mm / zone_size_m

    to_local = _to_local_transformer(global_center)

    print(f"[INFO] Зона: {zone_w:.1f} × {zone_h:.1f} м, scale={scale_factor:.5f} мм/м")

    # ─── КРОК 2: Завантаження даних OSM ─────────────────────────────────────
    status(10, "Завантаження даних OpenStreetMap...")

    gdf_buildings: gpd.GeoDataFrame = gpd.GeoDataFrame()
    gdf_water: gpd.GeoDataFrame = gpd.GeoDataFrame()
    gdf_green: gpd.GeoDataFrame = gpd.GeoDataFrame()
    G_roads = None
    source_crs = global_center.utm_crs

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_city = pool.submit(
                fetch_city_data,
                params.north, params.south, params.east, params.west,
                padding=0.005,
                target_crs=global_center.utm_crs,
            )
            f_green = pool.submit(
                fetch_extras,
                params.north, params.south, params.east, params.west,
                target_crs=global_center.utm_crs,
            )
            try:
                gdf_buildings, gdf_water, G_roads = f_city.result(timeout=120)
                if not gdf_buildings.empty:
                    source_crs = gdf_buildings.crs
            except Exception as e:
                print(f"[WARN] City data failed: {e}")
            try:
                gdf_green = f_green.result(timeout=120)
            except Exception as e:
                print(f"[WARN] Green areas data failed: {e}")
    except Exception as e:
        print(f"[ERROR] OSM loading error: {e}")

    n_roads = len(G_roads.edges) if G_roads is not None else 0
    print(f"[INFO] OSM: будівлі={len(gdf_buildings)}, вода={len(gdf_water)}, "
          f"дороги={n_roads}, зелені={len(gdf_green)}")

    # ─── КРОК 3: Конвертація у локальні координати ──────────────────────────
    status(15, "Конвертація координат у локальний простір...")

    gdf_buildings_local = _convert_gdf_to_local(gdf_buildings, to_local)
    gdf_water_local = _convert_gdf_to_local(gdf_water, to_local)
    gdf_green_local = _convert_gdf_to_local(gdf_green, to_local)

    building_geoms = [
        g for g in (gdf_buildings_local.geometry if not gdf_buildings_local.empty else [])
        if g is not None and not g.is_empty
    ]
    building_union = _safe_unary_union(building_geoms)

    water_geoms_local = [
        g for g in (gdf_water_local.geometry if not gdf_water_local.empty else [])
        if g is not None and not g.is_empty
    ]

    # ─── КРОК 4: Полігони доріг ──────────────────────────────────────────────
    status(20, "Побудова полігонів доріг...")

    # Ширина доріг × 3 для кращої видимості на моделі
    road_width_mult = params.road_width_multiplier * 3.0
    min_road_width_m = min(max(0.5 / scale_factor, 0.1), 14.0)

    merged_roads_utm = None   # UTM-координати (для process_roads)
    merged_roads_local = None  # локальні координати

    if G_roads is not None and n_roads > 0:
        try:
            merged_roads_utm = build_road_polygons(
                G_roads,
                width_multiplier=road_width_mult,
                min_width_m=min_road_width_m,
            )
            if merged_roads_utm is not None:
                # Об'єднуємо непринтабельно вузькі проміжки
                min_gap_m = (MIN_PRINTABLE_GAP_MM / 1000.0) / scale_factor
                merged_roads_utm = merge_close_road_gaps(merged_roads_utm, min_gap_m)

            if merged_roads_utm is not None:
                merged_roads_local = shapely_transform(to_local, merged_roads_utm)
                print(f"[INFO] Дороги: площа ~{merged_roads_local.area:.1f} m2")
        except Exception as e:
            print(f"[WARN] Road polygon build failed: {e}")
            traceback.print_exc()

    # ─── КРОК 5: Генерація рельєфу (DEM) ────────────────────────────────────
    status(25, "Генерація рельєфу з DEM даних...")

    base_thickness_m = params.terrain_base_thickness_mm / scale_factor
    water_depth_m = params.water_depth_mm / scale_factor

    terrain_mesh, terrain_provider = create_terrain_mesh(
        bbox_meters,
        z_scale=params.terrain_z_scale,
        resolution=params.terrain_resolution,
        latlon_bbox=latlon_bbox,
        source_crs=source_crs,
        terrarium_zoom=params.terrarium_zoom,
        base_thickness=base_thickness_m,
        flatten_buildings=bool(getattr(params, 'flatten_buildings_on_terrain', True)),
        building_geometries=building_geoms or None,
        flatten_roads=False,
        road_geometries=merged_roads_local,
        smoothing_sigma=params.terrain_smoothing_sigma,
        water_geometries=water_geoms_local or None,
        water_depth_m=water_depth_m,
        global_center=global_center,
        bbox_is_local=True,
        subdivide=params.terrain_subdivide,
        subdivide_levels=params.terrain_subdivide_levels,
    )

    if terrain_mesh is None:
        raise RuntimeError("Не вдалося згенерувати рельєф. Перевірте bbox та DEM-з'єднання.")

    terrain_z_min = float(terrain_mesh.bounds[0][2])
    terrain_z_max = float(terrain_mesh.bounds[1][2])
    print(f"[INFO] Рельєф: {len(terrain_mesh.vertices)} вершин, "
          f"Z=[{terrain_z_min:.3f}, {terrain_z_max:.3f}]")

    # ─── КРОК 6: Меш доріг ──────────────────────────────────────────────────
    status(40, "Генерація мешу доріг...")

    road_height_m = params.road_height_mm / scale_factor
    road_mesh: Optional[trimesh.Trimesh] = None

    if G_roads is not None:
        try:
            road_mesh = process_roads(
                G_roads,
                road_width_mult,
                terrain_provider=terrain_provider,
                floor_z=terrain_z_min,
                clearance_mm=GROOVE_CLEARANCE_MM,
                scale_factor=float(scale_factor),
                road_height=road_height_m,
                road_embed=0.0,
                merged_roads=merged_roads_utm,
                water_geometries=(
                    list(gdf_water.geometry.values)
                    if gdf_water is not None and not gdf_water.empty
                    else None
                ),
                bridge_height_multiplier=1.5,
                global_center=global_center,
                min_width_m=min_road_width_m,
                building_polygons=building_union,
            )
        except Exception as e:
            print(f"[WARN] Road mesh generation failed: {e}")
            traceback.print_exc()

    if road_mesh is not None:
        # Піднімаємо дороги на 0.5 мм над поверхнею рельєфу
        lift_m = 0.5 / scale_factor
        road_mesh.apply_translation([0, 0, lift_m])
        # Вирівнюємо дно доріг — однакова глибина пазу по всій моделі
        extend_road_mesh_to_uniform_bottom(road_mesh)
        print(f"[INFO] Дороги: {len(road_mesh.vertices)} вершин, "
              f"Z=[{road_mesh.bounds[0][2]:.3f}, {road_mesh.bounds[1][2]:.3f}]")
    else:
        print("[WARN] Меш доріг не створено")

    # ─── КРОК 7: Вирізання пазів для доріг ──────────────────────────────────
    status(50, "Вирізання пазів для доріг (Blender boolean)...")

    if road_mesh is not None:
        # Маска вирізання = полігон доріг (без ділянок під будівлями) + clearance
        road_cut_mask = None
        if merged_roads_local is not None:
            roads_no_buildings = merged_roads_local
            if building_union is not None:
                try:
                    roads_no_buildings = merged_roads_local.difference(building_union)
                    if roads_no_buildings.is_empty:
                        roads_no_buildings = merged_roads_local
                except Exception:
                    pass
            try:
                clearance_m = GROOVE_CLEARANCE_MM / scale_factor
                road_cut_mask = roads_no_buildings.buffer(clearance_m, join_style=2)
            except Exception:
                road_cut_mask = roads_no_buildings

        terrain_mesh = cut_roads_from_solid_terrain(
            terrain_mesh=terrain_mesh,
            road_polygons=road_cut_mask,
            clearance_m=GROOVE_CLEARANCE_MM / scale_factor,
            scale_factor=float(scale_factor),
            road_height_m=road_height_m,
            road_mesh=road_mesh,
        )

        if terrain_mesh is None:
            raise RuntimeError("Вирізання пазів доріг повернуло None — Blender відмовив.")

        print(f"[INFO] Рельєф після пазів доріг: {len(terrain_mesh.vertices)} вершин")

    # ─── КРОК 8: Меш парків ──────────────────────────────────────────────────
    status(60, "Генерація мешу зелених зон...")

    parks_mesh: Optional[trimesh.Trimesh] = None
    parks_cut_polygons = None

    if params.include_parks and not gdf_green_local.empty:
        # Маска доріг для вирізання з парків (ширша, з узбіччям)
        road_mask_for_parks = merged_roads_local
        if G_roads is not None and n_roads > 0:
            try:
                wide_roads = build_road_polygons(
                    G_roads,
                    width_multiplier=road_width_mult,
                    extra_buffer_m=3.0,
                )
                if wide_roads is not None:
                    road_mask_for_parks = shapely_transform(to_local, wide_roads)
            except Exception as e:
                print(f"[WARN] Wide road mask for parks failed: {e}")

        try:
            parks_mesh, parks_cut_polygons = process_green_areas(
                gdf_green_local,
                height_m=(params.parks_height_mm / scale_factor) / 2.0,
                embed_m=params.parks_embed_mm / scale_factor,
                terrain_provider=terrain_provider,
                global_center=global_center,
                scale_factor=float(scale_factor),
                road_polygons=road_mask_for_parks,
                building_polygons=building_union,
            )
        except Exception as e:
            print(f"[WARN] Park mesh generation failed: {e}")
            traceback.print_exc()

        if parks_mesh is not None:
            # Опускаємо дно мешу парків до найнижчої точки самого мешу
            extend_parks_mesh_to_uniform_bottom(parks_mesh)  # target_bottom_z=None → парк's own min Z
            parks_bottom = float(parks_mesh.vertices[:, 2].min())
            print(f"[INFO] Парки: {len(parks_mesh.vertices)} вершин, дно={parks_bottom:.4f}")
        else:
            print("[WARN] Меш парків не створено")

    # ─── КРОК 9: Вирізання пазів для парків ─────────────────────────────────
    status(65, "Вирізання пазів для зелених зон (Blender boolean)...")

    if parks_mesh is not None and params.include_parks:
        terrain_mesh = cut_parks_from_solid_terrain(
            terrain_mesh=terrain_mesh,
            parks_polygons=parks_cut_polygons,
            clearance_m=GROOVE_CLEARANCE_MM / scale_factor,
            scale_factor=float(scale_factor),
            parks_mesh=parks_mesh,
        )

        if terrain_mesh is None:
            raise RuntimeError("Вирізання пазів парків повернуло None — Blender відмовив.")

        print(f"[INFO] Рельєф після пазів парків: {len(terrain_mesh.vertices)} вершин")

    # ─── КРОК 10: Будівлі ───────────────────────────────────────────────────
    status(70, "Генерація будівель...")

    foundation_m = params.building_foundation_mm / scale_factor
    embed_m = params.building_embed_mm / scale_factor
    max_foundation_m = params.building_max_foundation_mm / scale_factor

    building_meshes: Optional[List[trimesh.Trimesh]] = None
    if not gdf_buildings_local.empty:
        try:
            building_meshes = process_buildings(
                gdf_buildings_local,
                min_height=params.building_min_height,
                height_multiplier=params.building_height_multiplier,
                terrain_provider=terrain_provider,
                foundation_depth=foundation_m,
                embed_depth=embed_m,
                max_foundation_depth=max_foundation_m,
                global_center=None,         # вже в локальних координатах
                coordinates_already_local=True,
            )
        except Exception as e:
            print(f"[WARN] Building generation failed: {e}")
            traceback.print_exc()

    n_buildings = len(building_meshes) if building_meshes else 0
    print(f"[INFO] Будівлі: {n_buildings} мешів")

    # ─── КРОК 11: Вода ──────────────────────────────────────────────────────
    status(80, "Генерація водних поверхонь...")

    water_mesh: Optional[trimesh.Trimesh] = None
    if not gdf_water_local.empty and terrain_provider is not None:
        # Товщина поверхні води: 0.2–0.6 мм на моделі
        surface_mm = float(min(max(params.water_depth_mm, 0.2), 0.6))
        thickness_m = surface_mm / scale_factor
        try:
            water_mesh = process_water_surface(
                gdf_water_local,
                thickness_m=thickness_m,
                depth_meters=water_depth_m,
                terrain_provider=terrain_provider,
                global_center=None,  # вже в локальних координатах
            )
        except Exception as e:
            print(f"[WARN] Water mesh generation failed: {e}")
            traceback.print_exc()

        if water_mesh is not None:
            print(f"[INFO] Вода: {len(water_mesh.vertices)} вершин")

    # ─── КРОК 12: Експорт ───────────────────────────────────────────────────
    status(90, f"Експорт у {params.export_format.upper()}...")

    gc.collect()  # звільняємо пам'ять перед важким export_scene

    export_scene(
        terrain_mesh=terrain_mesh,
        road_mesh=road_mesh,
        building_meshes=building_meshes,
        water_mesh=water_mesh,
        parks_mesh=parks_mesh,
        filename=output_path,
        format=params.export_format,
        model_size_mm=params.model_size_mm,
        add_flat_base=False,
        base_thickness_mm=params.terrain_base_thickness_mm,
    )

    status(100, "Готово!")
    print(f"[OK] Файл: {output_path}")
    return output_path
