"""
Сервіс для обробки будівель з екструзією та покращеними дахами
Покращено: додано посадку будівель на рельєф через TerrainProvider
"""
import geopandas as gpd
import trimesh
import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from typing import List, Optional
from dataclasses import dataclass
from services.terrain_provider import TerrainProvider
from services.global_center import GlobalCenter
from services.geometry_context import to_local_geodataframe_if_needed
import mapbox_earcut  # Для fallback методу extrude_building
import re
import gc  # For memory cleanup
import os


# Рендер визначних місць окремим БРОНЗОВИМ «Landmark»-шаром ВИМКНЕНО за рішенням
# власника (2026-06-21): «не треба виділяти окремим кольором визначні місця».
# Точні висоти будинків лишаються. Щоб повернути landmark-рендер → True.
LANDMARK_RENDERING_ENABLED = False


@dataclass
class BuildingMeshRecord:
    mesh: trimesh.Trimesh
    footprint: Optional[BaseGeometry]
    base_z: float
    landmark: str = ""   # OSM-категорія орієнтиру: "historic"/"tower"/"worship"/"attraction"; "" = звичайний будинок


def process_buildings(
    gdf_buildings: gpd.GeoDataFrame,
    min_height: float = 2.0,
    height_multiplier: float = 1.0,
    terrain_provider: Optional[TerrainProvider] = None,
    foundation_depth: float = 1.0,  # Глибина фундаменту в метрах (до масштабування)
    embed_depth: float = 0.0,       # Наскільки "втиснути" будівлю в землю (м), щоб не було щілин
    max_foundation_depth: Optional[float] = None,  # Запобіжник: максимальна глибина фундаменту (м)
    global_center: Optional[GlobalCenter] = None,  # Глобальний центр для перетворення координат (застарілий, використовується coordinates_already_local)
    coordinates_already_local: bool = False,  # ВИПРАВЛЕННЯ: якщо True, координати вже в локальних, не потрібно перетворювати
    return_records: bool = False,
    exclusion_polygons: Optional[BaseGeometry] = None,
    min_feature_m: float = 0.0,
    scale_factor: Optional[float] = None,
    max_height: Optional[float] = None,  # КАП висоти будівлі (світові метри); None=без капу
) -> List[trimesh.Trimesh] | List[BuildingMeshRecord]:
    """
    Обробляє будівлі, створюючи 3D меші з екструзією
    
    Args:
        gdf_buildings: GeoDataFrame з будівлями
        min_height: Мінімальна висота будівлі (метри)
        height_multiplier: Множник для висоти
    
    Returns:
        Список Trimesh об'єктів будівель
    """
    if gdf_buildings.empty:
        return []
    
    # ВИПРАВЛЕННЯ: Перетворюємо координати тільки якщо вони ще не в локальних
    # Якщо coordinates_already_local=True, координати вже перетворені в main.py
    if not coordinates_already_local and global_center is not None:
        try:
            print(f"[DEBUG] Перетворюємо gdf_buildings з UTM в локальні координати (fallback)")
            gdf_buildings = to_local_geodataframe_if_needed(gdf_buildings, global_center, force=True)
            print(f"[DEBUG] Перетворено {len(gdf_buildings)} геометрій будівель в локальні координати (fallback)")
        except Exception as e:
            print(f"[WARN] Не вдалося перетворити gdf_buildings в локальні координати: {e}")
            import traceback
            traceback.print_exc()
    elif coordinates_already_local:
        print(f"[DEBUG] Координати будівель вже в локальних, пропускаємо перетворення")
    
    building_meshes = []
    building_records: List[BuildingMeshRecord] = []

    # Мін. ДРУКОВНА ширина будинку (СПРАВЖНЯ, не bbox): тонкі «голки»/стовбці —
    # footprint вужчий за поріг УСЮДИ — ламаються на друку (і стають окремими
    # крихкими вежами коли union падає в concatenate на 4ГБ) → відкидаємо.
    # Тест = ерозія на півширини: якщо зникає весь полігон, він тонкий СКРІЗЬ
    # (rotation-independent, на відміну від bbox min_dim, що роздуває діагональні).
    # Орієнтири (is_landmark) НЕ чіпаємо. ENV BUILDING_MIN_WIDTH_MM (дефолт 0.6мм = поріг ДРУКОВНОСТІ ~1.5×сопло).
    # ВАЖЛИВО: це поріг ДРУКОВНОСТІ, а не «виглядає тонко». 1.75мм на масштабі цілого міста (~0.13мм/м)
    # = ~13м у світі → видаляло 73% НОРМАЛЬНИХ будинків (<13м) → «немає будинків».
    # Кіоски/ятки прибирає окремий фільтр ПЛОЩІ (BUILDING_MIN_AREA_M2) у full_generation_pipeline, не ширина.
    try:
        _bmw_mm = float(os.environ.get("BUILDING_MIN_WIDTH_MM", "0.6"))
    except Exception:
        _bmw_mm = 0.6
    building_min_width_m = (
        float(_bmw_mm) / float(scale_factor)
        if (scale_factor is not None and float(scale_factor) > 0.0 and _bmw_mm > 0)
        else 0.0
    )

    def _geometry_meets_threshold(poly: Optional[Polygon], is_landmark: bool = False) -> bool:
        if poly is None or poly.is_empty:
            return False
        # Орієнтири (церкви/вежі/історичні) НЕ відкидаємо за розміром — навіть малі мусять лишитись
        if is_landmark:
            return True
        # Тонкі будинки-голки: справжня ширина < поріг СКРІЗЬ → відкинути (стовбці)
        if building_min_width_m > 0:
            try:
                if poly.buffer(-building_min_width_m / 2.0).is_empty:
                    return False
            except Exception:
                pass
        if min_feature_m <= 0:
            return True
        try:
            minx, miny, maxx, maxy = poly.bounds
            min_dim = float(min(maxx - minx, maxy - miny))
            area = float(getattr(poly, "area", 0.0) or 0.0)
            min_area_m2 = max((float(min_feature_m) ** 2) * 0.5, 1e-8)
            return min_dim >= float(min_feature_m) and area >= min_area_m2
        except Exception:
            return True

    def _filter_to_threshold(geometry: Optional[BaseGeometry], is_landmark: bool) -> Optional[BaseGeometry]:
        """Keep only footprint parts that pass the printability threshold (drops thin
        tall "стовби"/needles — e.g. a building sliced by the zone window into a
        sub-nozzle strip). Returns None when nothing survives. Landmarks are exempt."""
        if geometry is None or getattr(geometry, "is_empty", True):
            return None
        if isinstance(geometry, Polygon):
            return geometry if _geometry_meets_threshold(geometry, is_landmark=is_landmark) else None
        if isinstance(geometry, MultiPolygon) or hasattr(geometry, "geoms"):
            try:
                parts = [
                    p for p in geometry.geoms
                    if isinstance(p, Polygon) and _geometry_meets_threshold(p, is_landmark=is_landmark)
                ]
            except Exception:
                return geometry
            if not parts:
                return None
            return parts[0] if len(parts) == 1 else MultiPolygon(parts)
        return geometry

    def _clip_building_geometry(geometry: Optional[BaseGeometry], landmark_category: str = "") -> Optional[BaseGeometry]:
        if geometry is None or getattr(geometry, "is_empty", True):
            return None
        if exclusion_polygons is None or getattr(exclusion_polygons, "is_empty", True):
            # No road/exclusion to clip against, but still drop needle footprints.
            return _filter_to_threshold(geometry, bool(landmark_category))
        lhs = geometry
        try:
            minx, miny, maxx, maxy = geometry.bounds
            pad = max(float(min_feature_m) * 2.0, 1.0)
            local_exclusion = exclusion_polygons.intersection(
                box(minx - pad, miny - pad, maxx + pad, maxy + pad)
            )
        except Exception:
            local_exclusion = exclusion_polygons
        try:
            lhs = geometry.buffer(0)
        except Exception:
            lhs = geometry
        try:
            local_exclusion = local_exclusion.buffer(0)
        except Exception:
            pass
        if local_exclusion is None or getattr(local_exclusion, "is_empty", True):
            # Building is not near any road exclusion — still drop needle footprints
            # (zone-window slivers) so thin tall "стовби" never reach the mesh.
            return _filter_to_threshold(lhs, bool(landmark_category))
        try:
            clipped = lhs.difference(local_exclusion)
        except Exception:
            try:
                clipped = unary_union([lhs]).buffer(0).difference(unary_union([local_exclusion]).buffer(0)).buffer(0)
            except Exception:
                # If the exclusion actually intersects the building but the boolean
                # still fails, drop the building instead of letting an unclipped
                # wall pass through the road groove.
                try:
                    if not getattr(lhs.intersection(local_exclusion), "is_empty", True):
                        return None
                except Exception:
                    pass
                return lhs
        try:
            clipped = clipped.buffer(0)
        except Exception:
            pass
        if clipped is None or getattr(clipped, "is_empty", True):
            return None
        kept: List[Polygon] = []
        if isinstance(clipped, Polygon):
            if _geometry_meets_threshold(clipped, is_landmark=bool(landmark_category)):
                kept.append(clipped)
        elif isinstance(clipped, MultiPolygon) or hasattr(clipped, "geoms"):
            try:
                for part in clipped.geoms:
                    if isinstance(part, Polygon) and _geometry_meets_threshold(part, is_landmark=bool(landmark_category)):
                        kept.append(part)
            except Exception:
                pass
        if not kept:
            return None
        if len(kept) == 1:
            return kept[0]
        return MultiPolygon(kept)

    def _append_building_mesh(
        mesh: trimesh.Trimesh,
        footprint: Optional[BaseGeometry],
        base_z_override: Optional[float] = None,
        landmark_category: str = "",
    ) -> None:
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return
        building_meshes.append(mesh)
        if not return_records:
            return
        try:
            clean_footprint = footprint.buffer(0) if footprint is not None else None
        except Exception:
            clean_footprint = footprint
        if base_z_override is not None:
            base_z = float(base_z_override)
        else:
            try:
                base_z = float(mesh.bounds[0][2])
            except Exception:
                base_z = 0.0
        building_records.append(
            BuildingMeshRecord(
                mesh=mesh,
                footprint=clean_footprint,
                base_z=base_z,
                landmark=landmark_category,
            )
        )

    def _model_mm_to_world_m(model_mm: float) -> float:
        try:
            if scale_factor is not None and float(scale_factor) > 0.0:
                return float(model_mm) / float(scale_factor)
        except Exception:
            pass
        return 0.0

    def ground_heights_for_geom(g) -> np.ndarray:
        """
        ПОКРАЩЕНА ВЕРСІЯ: Адаптивний семплінг рельєфу для будівель.
        Використовує більше точок для великих будівель та складного рельєфу.
        ВАЖЛИВО: Рельєф під будівлями вже вирівняний через flatten_heightfield_under_buildings,
        але для точності використовуємо адаптивний семплінг.
        """
        if terrain_provider is None:
            return np.array([0.0], dtype=float)
        try:
            pts = []
            # Polygon або MultiPolygon
            polys = []
            if isinstance(g, Polygon):
                polys = [g]
            elif isinstance(g, MultiPolygon) or hasattr(g, "geoms"):
                try:
                    polys = [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]
                except Exception:
                    polys = []

            if not polys:
                # fallback: хоча б центроїд
                c = g.centroid
                pts.append([c.x, c.y])
            else:
                for poly in polys:
                    if poly.exterior is None:
                        continue
                    
                    try:
                        minx, miny, maxx, maxy = poly.bounds
                        dx = float(maxx - minx)
                        dy = float(maxy - miny)
                        area = float(poly.area)
                        
                        # АДАПТИВНИЙ СЕМПЛІНГ: більше точок для великих будівель
                        # Для малих будівель (< 100 м?): мінімальний семплінг
                        # Для середніх (100-1000 м?): середній семплінг
                        # Для великих (> 1000 м?): щільний семплінг
                        
                        if area < 100.0:
                            # Малий: контур + кути + центр
                            coords = np.array(poly.exterior.coords)
                            if len(coords) > 0:
                                step = max(1, len(coords) // 8)
                                pts.extend(coords[::step, :2].tolist())
                        elif area < 1000.0:
                            # Середній: контур + регулярна сітка 3x3
                            coords = np.array(poly.exterior.coords)
                            if len(coords) > 0:
                                step = max(1, len(coords) // 16)
                                pts.extend(coords[::step, :2].tolist())
                            
                            # Регулярна сітка 3x3 всередині
                            for i in range(1, 4):
                                for j in range(1, 4):
                                    x = minx + (dx * i / 4.0)
                                    y = miny + (dy * j / 4.0)
                                    if poly.contains(Point(x, y)):
                                        pts.append([x, y])
                        else:
                            # Великий: контур + щільна сітка 5x5
                            coords = np.array(poly.exterior.coords)
                            if len(coords) > 0:
                                step = max(1, len(coords) // 32)
                                pts.extend(coords[::step, :2].tolist())
                            
                            # Щільна сітка 5x5 всередині
                            for i in range(1, 6):
                                for j in range(1, 6):
                                    x = minx + (dx * i / 6.0)
                                    y = miny + (dy * j / 6.0)
                                    if poly.contains(Point(x, y)):
                                        pts.append([x, y])
                        
                        # Завжди додаємо центроїд та кутові точки
                        c = poly.centroid
                        pts.append([c.x, c.y])
                        
                        corners = [
                            (minx, miny),  # Лівий нижній
                            (maxx, miny),  # Правий нижній
                            (maxx, maxy),  # Правий верхній
                            (minx, maxy),  # Лівий верхній
                        ]
                        for x, y in corners:
                            if poly.contains(Point(x, y)) or poly.touches(Point(x, y)):
                                pts.append([x, y])
                    except Exception:
                        # Fallback: хоча б центроїд
                        pass
                    try:
                        c = poly.centroid
                        pts.append([c.x, c.y])
                    except Exception:
                        pass
                    pass

            if len(pts) == 0:
                return np.array([0.0], dtype=float)

            pts_arr = np.array(pts, dtype=float)
            heights = terrain_provider.get_heights_for_points(pts_arr)
            if heights.size == 0:
                return np.array([0.0], dtype=float)
            heights = np.asarray(heights, dtype=float)
            
            # ВАЛІДАЦІЯ: Перевіряємо на NaN/Inf та виправляємо (як у H:\3dMap)
            if np.any(~np.isfinite(heights)):
                valid_heights = heights[np.isfinite(heights)]
                if valid_heights.size > 0:
                    # Якщо є валідні значення, використовуємо їх
                    return valid_heights
                else:
                    # Якщо всі NaN/Inf, повертаємо fallback
                    return np.array([0.0], dtype=float)
            
            return heights
        except Exception as e:
            mz = float(getattr(terrain_provider, "min_z", 0.0))
            return np.array([mz], dtype=float)

    def original_grade_max(g) -> Optional[float]:
        """Highest ORIGINAL (pre-flatten) terrain at the footprint boundary AND just
        outside it — the real grade the roof must clear so the building is not
        half-buried in a slope. We sample a ring a couple of metres OUTSIDE the
        footprint (not only the boundary): on a hillside the terrain that buries the
        uphill wall is the neighbour grade just beyond the footprint, which sits above
        the boundary itself. We use the *original* surface (before
        flatten_heightfield_under_buildings dug a min-pad and before gaussian
        smoothing); the smoothed final surface the building merges into is always
        <= original, so clearing the original guarantees no burial. Returns None when
        no original surface is available (then seating is unchanged)."""
        op = getattr(terrain_provider, "original_heights_provider", None) if terrain_provider is not None else None
        if op is None:
            return None
        try:
            if isinstance(g, Polygon):
                polys = [g]
            else:
                polys = [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]
            coords = []
            for poly in polys:
                if poly.exterior is not None:
                    coords.append(np.asarray(poly.exterior.coords)[:, :2])
                # ring just outside this part captures the uphill neighbour grade
                try:
                    outer = poly.buffer(_grade_ring_m).exterior
                    if outer is not None and outer.length > 0:
                        n = int(min(96, max(16, outer.length / max(_grade_ring_m, 0.5))))
                        ts = np.linspace(0.0, 1.0, n)
                        coords.append(np.array([outer.interpolate(t, normalized=True).coords[0] for t in ts]))
                except Exception:
                    pass
            if not coords:
                return None
            pts = np.vstack(coords)
            h = np.asarray(op.get_heights_for_points(pts), dtype=float)
            h = h[np.isfinite(h)]
            return float(h.max()) if h.size else None
        except Exception:
            return None

    # Minimum visible building height above the LOCAL grade on a slope (model mm).
    # Used to lift the roof so the uphill side is not swallowed by the terrain.
    try:
        _min_visible_slope_mm = float(os.environ.get("BUILDING_MIN_VISIBLE_ON_SLOPE_MM", "1.5"))
    except (TypeError, ValueError):
        _min_visible_slope_mm = 1.5
    _min_visible_slope_m = _model_mm_to_world_m(_min_visible_slope_mm) if _min_visible_slope_mm > 0 else 0.0
    # How far outside the footprint to sample the uphill neighbour grade (world m).
    try:
        _grade_ring_mm = float(os.environ.get("BUILDING_GRADE_RING_MM", "1.0"))
    except (TypeError, ValueError):
        _grade_ring_mm = 1.0
    _grade_ring_m = max(2.0, _model_mm_to_world_m(_grade_ring_mm))

    # MEMORY OPTIMIZATION: Process buildings in batches to reduce RAM usage
    # iterrows() is very slow and memory-intensive, so we process by indexing instead
    # Adaptive batch size: smaller batches for large datasets to avoid memory exhaustion
    total_buildings = len(gdf_buildings)
    if total_buildings > 5000:
        batch_size = 50  # Smaller batches for very large datasets
    elif total_buildings > 2000:
        batch_size = 75
    else:
        batch_size = 100  # Default batch size
    
    print(f"[INFO] Processing {total_buildings} buildings in batches of {batch_size}...")
    
    for batch_start in range(0, total_buildings, batch_size):
        batch_end = min(batch_start + batch_size, total_buildings)
        batch_indices = gdf_buildings.index[batch_start:batch_end]
        batch_gdf = gdf_buildings.loc[batch_indices]
        
        print(f"[INFO] Processing buildings batch {batch_start//batch_size + 1}/{(total_buildings + batch_size - 1)//batch_size} ({batch_start+1}-{batch_end}/{total_buildings})...")
        
        # Process this batch
        for idx in batch_indices:
            try:
                row = gdf_buildings.loc[idx]
                geom = row.geometry
                # Орієнтир (визначне місце) з OSM-тегу landmark; "" = звичайний будинок.
                # Вимкнено власником → завжди "" (жоден будинок не виділяється кольором).
                landmark_category = ""
                if LANDMARK_RENDERING_ENABLED:
                    try:
                        _lm = row.get("landmark", "") if hasattr(row, "get") else getattr(row, "landmark", "")
                        landmark_category = str(_lm).strip() if _lm is not None else ""
                    except Exception:
                        landmark_category = ""

                # Пропускаємо невалідні геометрії
                if geom is None:
                    continue
                
                # Перевіряємо валідність геометрії
                try:
                    if geom.is_empty:
                        continue
                    if not geom.is_valid:
                        # Спробуємо виправити геометрію
                        geom = geom.buffer(0)
                        if geom.is_empty:
                            continue
                        # Перевіряємо чи після виправлення геометрія має достатньо точок
                        if hasattr(geom, 'exterior') and len(geom.exterior.coords) < 3:
                            continue
                except Exception as e:
                    print(f"  [WARN] Помилка перевірки геометрії будівлі {idx}: {e}")
                    continue

                geom = _clip_building_geometry(geom, landmark_category=landmark_category)
                if geom is None or getattr(geom, "is_empty", True):
                    continue
                
                # Отримуємо висоту будівлі
                height = get_building_height(row, min_height) * height_multiplier
                # КАП висоти: без капу хмарочоси × множник давали 30мм-вежі на 90мм-
                # плитці — і друк-крихкі, і висоти «дико перепадають». Капуємо до max_height
                # (зберігаючи відносний порядок: нижчі лишаються нижчими).
                if max_height is not None and max_height > 0 and height > max_height:
                    height = max_height

                # Розраховуємо foundation_depth_eff заздалегідь (для використання в обох гілках)
                foundation_depth_eff = max(float(foundation_depth), float(embed_depth), 0.1)
                if max_foundation_depth is not None:
                    try:
                        foundation_depth_eff = min(float(foundation_depth_eff), float(max_foundation_depth))
                    except Exception:
                        pass
                foundation_depth_eff = max(float(foundation_depth_eff), 0.05)
                foundation_depth_m = float(foundation_depth_eff)
                terrain_sink_m = _model_mm_to_world_m(0.1)

                # Якщо рельєфу нема — не "топимо" будівлі фундаментом у нуль,
                # достатньо мінімального embed (щоб не було щілини з плоскою базою).
                if terrain_provider is None:
                    translate_z = -float(embed_depth) if float(embed_depth) > 0 else 0.0
                    flat_base_z = float(translate_z)
                else:
                    # СПРОЩЕНА ЛОГІКА: Беремо висоти безпосередньо з terrain mesh для координат будівлі
                    # ВАЖЛИВО: geom вже в локальних координатах (після перетворення через global_center)
                    # Рельєф під будівлями вже вирівняний через flatten_heightfield_under_buildings
                    heights = ground_heights_for_geom(geom)
                    
                    # ВАЛІДАЦІЯ: Перевіряємо на NaN/Inf та виправляємо (як у H:\3dMap)
                    if heights.size == 0 or np.any(~np.isfinite(heights)):
                        # Якщо не вдалося отримати висоти або є NaN/Inf - використовуємо fallback
                        if heights.size > 0:
                            valid_heights = heights[np.isfinite(heights)]
                            if len(valid_heights) > 0:
                                ground_min = float(np.min(valid_heights))
                                ground_max = float(np.max(valid_heights))
                            else:
                                ground_min = 0.0
                                ground_max = 0.0
                        else:
                            ground_min = 0.0
                            ground_max = 0.0
                        print(f"  [WARN] Будівля {idx}: невалідні висоти рельєфу, використовую fallback: {ground_min:.2f}m")
                    else:
                        ground_min = float(np.min(heights))
                        ground_max = float(np.max(heights))

                    # Visible building base follows the LOWEST terrain point under
                    # the footprint — matches flatten_heightfield_under_buildings,
                    # which flattens the pad to ref=min(h). Using ground_max made
                    # buildings FLOAT above relief maps: on a coarse DEM grid the
                    # flatten only sets whole cells to min, but the footprint's
                    # sampled points interpolate across neighbouring un-flattened
                    # (higher) cells, so ground_max > pad → the base hovered up to
                    # several metres above the surface. ground_min seats the base on
                    # the pad; on the high side terrain rises into the wall (natural
                    # cut-in), never a gap. Sinks 0.1 model-mm to kill the seam.
                    base_z = float(ground_min) - float(terrain_sink_m)
                    required_foundation_m = max(float(ground_max) - float(ground_min), 0.0) + float(terrain_sink_m)
                    foundation_depth_m = max(float(foundation_depth_eff), float(required_foundation_m), 0.05)

                    # SLOPE FIX (half-buried buildings): keep base_z at ground_min so the
                    # downhill side and foundation stay covered, but RAISE the roof so it
                    # clears the local uphill grade by _min_visible_slope_m. Without this,
                    # the (unflattened) terrain just outside the footprint rises ABOVE
                    # roof = base_z + height for short buildings on a slope, so the uphill
                    # side is swallowed -> the building reads as half-buried. We bump the
                    # effective height once here; it propagates to the extrude + the wall
                    # remap (and, for a MultiPolygon, every part inherits the same height,
                    # which is >= each part's local grade clearance since base_z is global
                    # ground_min). Relief-only (inside the terrain_provider branch).
                    _grade_max = original_grade_max(geom)
                    if _grade_max is not None and _min_visible_slope_m > 0.0:
                        _needed_h = (float(_grade_max) + float(_min_visible_slope_m)) - float(base_z)
                        if _needed_h > float(height):
                            height = float(_needed_h)

                    # translate_z - Z координата нижньої точки будівлі (base_z мінус фундамент)
                    translate_z = float(base_z) - float(foundation_depth_m)
                    flat_base_z = float(base_z)
                    
                    # Перевірка на валідність
                    if not np.isfinite(translate_z) or not np.isfinite(base_z):
                        print(f"  [WARN] Будівля {idx}: невалідні координати після обчислення, використовую fallback")
                        base_z = float(ground_min)
                        translate_z = float(base_z) - float(foundation_depth_m)
                        flat_base_z = float(base_z)
                
                # Simplify geometry to speed up triangulation and reduce vertex count
                try:
                    # Simplify with 0.1m tolerance (preserves shape but removes redundant points)
                    geom = geom.simplify(0.1, preserve_topology=True)
                except Exception:
                    pass

                # Екструзія полігону (використовуємо trimesh.creation.extrude_polygon)
                if isinstance(geom, Polygon):
                    # Перевіряємо чи полігон має достатньо точок
                    if hasattr(geom, 'exterior') and len(geom.exterior.coords) < 3:
                        # print(f"  [SKIP] Будівля {idx}: полігон має менше 3 точок")
                        continue
                    
                    try:
                        # Використовуємо вбудовану функцію trimesh для екструзії
                        mesh = trimesh.creation.extrude_polygon(geom, height=height)
                        
                        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                            print(f"  [WARN] Будівля {idx}: extrude_polygon повернув порожній mesh")
                            # Fallback на старий метод
                            mesh = extrude_building(geom, height)
                            if mesh is None or len(mesh.faces) == 0:
                                continue
                        
                        # FIX: Align to base (apply translation)
                        mesh.apply_translation([0, 0, translate_z])
                        if terrain_provider is not None and len(mesh.vertices) > 0:
                            vertices = mesh.vertices.copy()
                            
                            # ВАЛІДАЦІЯ: Перевіряємо на NaN/Inf перед обробкою (як у H:\3dMap)
                            if np.any(~np.isfinite(vertices)):
                                print(f"  [WARN] Будівля {idx}: знайдено NaN/Inf в вершинах після екструзії")
                                # Виправляємо NaN/Inf
                                vertices = np.nan_to_num(vertices, nan=0.0, posinf=1e6, neginf=-1e6)
                            
                            # Отримуємо висоти рельєфу для всіх вершин (як у H:\3dMap)
                            ground = terrain_provider.get_surface_heights_for_points(vertices[:, :2])
                            
                            # ВАЛІДАЦІЯ: Перевіряємо на NaN/Inf у висотах рельєфу (як у H:\3dMap)
                            if np.any(np.isnan(ground)) or np.any(np.isinf(ground)):
                                nan_count = np.sum(np.isnan(ground))
                                inf_count = np.sum(np.isinf(ground))
                                print(f"  [WARN] Будівля {idx}: Terrain повернув NaN/Inf висот (NaN: {nan_count}, Inf: {inf_count}), використовую ground level 0.0")
                                ground = np.zeros_like(ground)
                            
                            # ВИПРАВЛЕННЯ: Використовуємо нижні 20% висоти будівлі замість фіксованого порогу
                            building_height = float(height)
                            old_z = vertices[:, 2] - translate_z  # Відносна висота від translate_z
                            is_bottom = old_z < (building_height * 0.1)
                            is_top = old_z > (building_height * 0.9)
                            
                            # Straight walls start from the highest sampled
                            # terrain point (slightly sunk by 0.1 model-mm).
                            roof_z = flat_base_z + building_height

                            # ВАЛІДАЦІЯ
                            if not np.isfinite(flat_base_z) or not np.isfinite(roof_z):
                                print(f"  [WARN] Будівля {idx}: обчислення висот дало NaN, пропускаю")
                                continue

                            # Straight walls: linear interpolation from flat_base_z to roof_z
                            ratio = np.clip(old_z / building_height, 0.0, 1.0)
                            vertices[:, 2] = flat_base_z + ratio * building_height
                            
                            # ФІНАЛЬНА ВАЛІДАЦІЯ: Перевіряємо на NaN/Inf після обчислень (як у H:\3dMap)
                            if np.any(np.isnan(vertices)) or np.any(np.isinf(vertices)):
                                print(f"  [ERROR] Будівля {idx}: вершини містять NaN/Inf після обчислень, пропускаю")
                                continue
                            
                            mesh.vertices = vertices
                        
                        # Перевірка на валідність mesh
                        try:
                            if not mesh.is_volume:
                                mesh.fill_holes()
                            mesh.update_faces(mesh.unique_faces())
                            mesh.remove_unreferenced_vertices()
                            # ВИПРАВЛЕНО: Виправляємо нормалі для правильного відображення з усіх сторін
                            try:
                                trimesh.repair.fix_winding(mesh)
                            except Exception:
                                pass
                            mesh.fix_normals()
                        except Exception as fix_error:
                            print(f"  [WARN] Будівля {idx}: помилка виправлення mesh: {fix_error}")
                        
                        if mesh and len(mesh.faces) > 0 and len(mesh.vertices) > 0:
                            _append_building_mesh(mesh, geom, base_z_override=flat_base_z, landmark_category=landmark_category)
                        else:
                            print(f"  [SKIP] Будівля {idx}: mesh невалідний після обробки")
                    except Exception as e:
                        print(f"  [WARN] Помилка екструзії будівлі {idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fallback на старий метод
                        try:
                            mesh = extrude_building(geom, height)
                            if mesh:
                                mesh.apply_translation([0, 0, translate_z])
                                if len(mesh.faces) > 0:
                                    _append_building_mesh(mesh, geom, landmark_category=landmark_category)
                        except Exception:
                            pass
                # ВИПРАВЛЕННЯ: Якщо MultiPolygon, обробляємо кожен полігон окремо з ОКРЕМИМ translate_z
                elif hasattr(geom, 'geoms') or isinstance(geom, MultiPolygon):
                    geoms_list = geom.geoms if hasattr(geom, 'geoms') else [geom]
                    for poly_idx, poly in enumerate(geoms_list):
                        if not isinstance(poly, Polygon):
                            continue
                        
                        # Перевіряємо валідність полігону
                        try:
                            if poly.is_empty or not poly.is_valid:
                                poly = poly.buffer(0)
                                if poly.is_empty:
                                    continue
                            if hasattr(poly, 'exterior') and len(poly.exterior.coords) < 3:
                                continue
                        except Exception:
                            continue
                        
                        # ВИПРАВЛЕННЯ: Розраховуємо translate_z окремо для кожного полігону
                        poly_translate_z = translate_z  # Початкове значення
                        poly_flat_base_z = flat_base_z if terrain_provider is None else None
                        poly_foundation_depth_m = float(foundation_depth_m)
                        if terrain_provider is not None:
                            poly_heights = ground_heights_for_geom(poly)
                            if poly_heights.size > 0:
                                valid_ph = poly_heights[np.isfinite(poly_heights)]
                                if valid_ph.size > 0:
                                    poly_ground_min = float(np.min(valid_ph))
                                    poly_ground_max = float(np.max(valid_ph))
                                else:
                                    poly_ground_min = ground_min
                                    poly_ground_max = ground_max

                                # ground_min (not max) — seats base on the flattened
                                # pad; ground_max floated the building (see single-poly
                                # branch above for the full explanation).
                                poly_base_z = float(poly_ground_min) - float(terrain_sink_m)
                                poly_required_foundation_m = max(
                                    float(poly_ground_max) - float(poly_ground_min),
                                    0.0,
                                ) + float(terrain_sink_m)
                                poly_foundation_depth_m = max(
                                    float(foundation_depth_eff),
                                    float(poly_required_foundation_m),
                                    0.05,
                                )
                                poly_translate_z = float(poly_base_z) - float(poly_foundation_depth_m)
                                poly_flat_base_z = poly_base_z
                        
                        try:
                            mesh = trimesh.creation.extrude_polygon(poly, height=height)
                            
                            if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                                # Fallback
                                mesh = extrude_building(poly, height)
                                if mesh is None or len(mesh.faces) == 0:
                                    continue
                            
                            mesh.apply_translation([0, 0, poly_translate_z])
                            
                            # Така сама покращена агресивна перевірка як для одиночних полігонів
                            if terrain_provider is not None and len(mesh.vertices) > 0:
                                vertices = mesh.vertices.copy()
                                
                                # ВАЛІДАЦІЯ: Перевіряємо на NaN/Inf перед обробкою (як у H:\3dMap)
                                if np.any(~np.isfinite(vertices)):
                                    print(f"  [WARN] Будівля {idx} (MultiPolygon): знайдено NaN/Inf в вершинах після екструзії")
                                    vertices = np.nan_to_num(vertices, nan=0.0, posinf=1e6, neginf=-1e6)
                                
                                # Отримуємо висоти рельєфу для всіх вершин (як у H:\3dMap)
                                ground = terrain_provider.get_surface_heights_for_points(vertices[:, :2])
                                
                                # ВАЛІДАЦІЯ: Перевіряємо на NaN/Inf у висотах рельєфу
                                if np.any(np.isnan(ground)) or np.any(np.isinf(ground)):
                                    nan_count = np.sum(np.isnan(ground))
                                    inf_count = np.sum(np.isinf(ground))
                                    print(f"  [WARN] Будівля {idx} (MultiPolygon): Terrain повернув NaN/Inf висот (NaN: {nan_count}, Inf: {inf_count}), використовую ground level 0.0")
                                    ground = np.zeros_like(ground)
                                
                                building_height = float(height)
                                old_z = vertices[:, 2] - poly_translate_z  # Відносна висота від translate_z

                                _poly_flat_base = (
                                    float(poly_flat_base_z)
                                    if poly_flat_base_z is not None
                                    else float(poly_translate_z)
                                )
                                roof_z = _poly_flat_base + building_height

                                if not np.isfinite(_poly_flat_base) or not np.isfinite(roof_z):
                                    print(f"  [WARN] Будівля {idx} (MultiPolygon): обчислення висот дало NaN, пропускаю")
                                    continue

                                ratio = np.clip(old_z / building_height, 0.0, 1.0)
                                vertices[:, 2] = _poly_flat_base + ratio * building_height
                                    
                                
                                # ФІНАЛЬНА ВАЛІДАЦІЯ: Перевіряємо на NaN/Inf після обчислень (як у H:\3dMap)
                                if np.any(np.isnan(vertices)) or np.any(np.isinf(vertices)):
                                    print(f"  [ERROR] Будівля {idx} (MultiPolygon): вершини містять NaN/Inf після обчислень, пропускаю")
                                    continue
                                
                                mesh.vertices = vertices
                                
                                try:
                                    trimesh.repair.fix_winding(mesh)
                                except Exception:
                                    pass
                                try:
                                    mesh.fix_normals()
                                except Exception:
                                    pass
                            
                            # Перевірка на валідність mesh
                            try:
                                if not mesh.is_volume:
                                    mesh.fill_holes()
                                mesh.update_faces(mesh.unique_faces())
                                mesh.remove_unreferenced_vertices()
                            except Exception:
                                pass

                            if mesh and len(mesh.faces) > 0 and len(mesh.vertices) > 0:
                                _append_building_mesh(mesh, poly, base_z_override=poly_flat_base_z, landmark_category=landmark_category)
                        except Exception as e:
                            # Fallback
                            try:
                                mesh = extrude_building(poly, height)
                                if mesh:
                                    mesh.apply_translation([0, 0, poly_translate_z])
                                    if mesh and len(mesh.faces) > 0:
                                        _append_building_mesh(mesh, poly, base_z_override=poly_flat_base_z, landmark_category=landmark_category)
                            except Exception:
                                continue
            except Exception as e:
                print(f"Помилка обробки будівлі {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # MEMORY OPTIMIZATION: Explicitly free batch data after processing
        del batch_gdf
        gc.collect()
        
        print(f"[INFO] Batch {batch_start//batch_size + 1} completed. Total buildings so far: {len(building_meshes)}")
    
    print(f"Створено {len(building_meshes)} будівель")
    if return_records:
        return building_records
    return building_meshes


def get_building_height(row, min_height: float) -> float:
    """
    Визначає висоту будівлі з OSM тегів
    """
    # Спробуємо отримати висоту з тегів
    height = None

    def _parse_number(val) -> Optional[float]:
        if val is None:
            return None
        if isinstance(val, (int, float, np.integer, np.floating)):
            num = float(val)
            if np.isfinite(num):
                return num
            return None
        if isinstance(val, str):
            s = val.strip().replace(",", ".")
            m = re.search(r"[-+]?\d+(\.\d+)?", s)
            if not m:
                return None
            try:
                return float(m.group(0))
            except Exception:
                return None
        return None
    
    def _parse_height_m(val) -> Optional[float]:
        """
        Повертає висоту в метрах.
        Підтримка: "20", "20m", "20 m", "65 ft", "65feet".
        """
        if val is None:
            return None
        if isinstance(val, (int, float, np.integer, np.floating)):
            num = float(val)
            if np.isfinite(num):
                return num
            return None
        if isinstance(val, str):
            s = val.strip().lower().replace(",", ".")
            num = _parse_number(s)
            if num is None:
                return None
            # feet -> meters
            if "ft" in s or "feet" in s or "foot" in s:
                return float(num) * 0.3048
            return float(num)
        return None

    def _parse_levels(val) -> Optional[float]:
        """
        Рівні можуть бути "5", "5;6", "5-6". Беремо перше число.
        """
        return _parse_number(val)

    # 1) Явні висоти (height / building:height)
    for key in ["height", "building:height"]:
        if key in row:
            h = _parse_height_m(row.get(key))
            if h is not None and h > 0:
                height = max(float(height or 0.0), float(h))

    # 2) Рівні (levels) -> метри
    levels_m = None
    for key in ["building:levels", "building:levels:aboveground", "levels"]:
        if key in row:
            lv = _parse_levels(row.get(key))
            if lv is not None and lv > 0:
                # Класичне припущення: ~3м на поверх (стабільно і прогнозовано)
                levels_m = float(lv) * 3.0
                break
    if levels_m is not None:
        height = max(float(height or 0.0), float(levels_m))

    # 3) Roof додаємо, якщо є (в OSM часто окремо)
    roof_h = None
    for key in ["roof:height"]:
        if key in row:
            roof_h = _parse_height_m(row.get(key))
            break
    if roof_h is None and "roof:levels" in row:
        rv = _parse_levels(row.get("roof:levels"))
        if rv is not None and rv > 0:
            roof_h = float(rv) * 1.5
    if roof_h is not None and roof_h > 0:
        height = float(height or 0.0) + float(roof_h)

    # Якщо тегів нема — раніше ВСІ такі будинки ставали ОДНАКОВО плоскими (min_height)
    # → рідко-тегований район виглядав як однакові плити. Тепер даємо ДЕТЕРМІНОВАНУ
    # варіацію (+0..6м, ~0..2 поверхи) за id будівлі: цифри стабільні між генераціями,
    # район виглядає живим, але БЕЗ вигаданих «хмарочосів». Будинки з реальними
    # тегами (height/levels) цю гілку не зачіпає.
    if height is None or height < min_height:
        try:
            seed = abs(int(float(row.get("id", 0) or 0)))
        except Exception:
            seed = 0
        height = float(min_height) + float(seed % 4) * 2.0  # 0,2,4,6 м понад базу

    return height


def extrude_building(polygon: Polygon, height: float) -> Optional[trimesh.Trimesh]:
    """
    Екструдує полігон будівлі на вказану висоту
    
    Args:
        polygon: Полігон будівлі
        height: Висота екструзії (метри)
    
    Returns:
        Trimesh об'єкт будівлі
    """
    try:
        # Отримуємо координати зовнішнього контуру
        exterior_coords = np.array(polygon.exterior.coords[:-1])  # Видаляємо дублікат
        
        # Створюємо верхню та нижню поверхні
        vertices_bottom = np.column_stack([
            exterior_coords[:, 0],
            exterior_coords[:, 1],
            np.zeros(len(exterior_coords))
        ])
        
        vertices_top = np.column_stack([
            exterior_coords[:, 0],
            exterior_coords[:, 1],
            np.full(len(exterior_coords), height)
        ])
        
        # Тріангуляція для верхньої та нижньої поверхонь
        try:
            coords_flat = exterior_coords.flatten().tolist()
            triangles_flat = mapbox_earcut.triangulate_float32(coords_flat, [])
            triangles = np.array(triangles_flat).reshape(-1, 3)
        except Exception as e:
            # Fallback: проста тріангуляція через трикутники від першої вершини
            n = len(exterior_coords)
            triangles = np.array([[0, i, (i+1)%n] for i in range(1, n-1)])
        
        # Всі вершини
        all_vertices = np.vstack([vertices_bottom, vertices_top])
        
        # Індекси для нижньої поверхні (обернені для правильного напрямку нормалі)
        bottom_faces = triangles[:, ::-1]
        
        # Індекси для верхньої поверхні (з зсувом)
        # Оригінальна тріангуляція дає нормалі вверх, що правильно для даху
        top_faces = triangles + len(vertices_bottom)
        
        # Бічні стіни (квадри з двох трикутників)
        n = len(exterior_coords)
        side_faces = []
        for i in range(n):
            next_i = (i + 1) % n
            # ВИПРАВЛЕНО: Правильний winding order для нормалей НАЗОВНІ
            # При обході полігону проти годинникової, нормалі мають дивитись назовні
            side_faces.append([i, next_i, i + n])
            side_faces.append([next_i, next_i + n, i + n])
        
        # Об'єднуємо всі грані
        all_faces = np.vstack([
            bottom_faces,
            top_faces,
            np.array(side_faces)
        ])
        
        # Створюємо меш
        mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
        
        # Перевірка на валідність
        try:
            if not mesh.is_volume:
                # Спроба виправити
                mesh.fill_holes()
                mesh.update_faces(mesh.unique_faces())
                mesh.remove_unreferenced_vertices()
            # ВИПРАВЛЕНО: Виправляємо нормалі для правильного відображення з усіх сторін
            mesh.fix_normals()
        except Exception as fix_error:
            # Якщо не вдалося виправити, все одно повертаємо меш
            print(f"Попередження при виправленні мешу: {fix_error}")
        
        return mesh
        
    except Exception as e:
        print(f"Помилка екструзії будівлі: {e}")
        import traceback
        traceback.print_exc()
        return None
