from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import trimesh
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from services.groove_pipeline import prepare_road_cut_mask
from services.terrain_generator import create_terrain_mesh


def _iter_extrudable_polygons(geom: Any, min_area_m2: float = 1e-6) -> list:
    """Витягує список ВАЛІДНИХ, не-вироджених Polygon-ів з будь-якої геометрії
    (Polygon / MultiPolygon / GeometryCollection) для безпечної екструзії.

    Захист для AMS-бази: water-split через `.difference()` може дати MultiPolygon,
    GeometryCollection (з лініями/точками), невалідні self-intersecting полігони
    або вироджені «волоски» нульової площі. trimesh.extrude_polygon на такому
    кидає виняток → база зникає. Тут чистимо (make_valid → buffer(0)) і фільтруємо
    вироджене, повертаючи лише те, що реально екструдується.
    """
    if geom is None or getattr(geom, "is_empty", True):
        return []

    gt = getattr(geom, "geom_type", "")
    out: list = []
    if gt in ("MultiPolygon", "GeometryCollection"):
        for g in getattr(geom, "geoms", []):
            out.extend(_iter_extrudable_polygons(g, min_area_m2))
        return out
    if gt != "Polygon":
        # LineString / Point / тощо — не екструдується.
        return []

    poly = geom
    # Чистимо невалідну геометрію (self-intersection після difference).
    if not poly.is_valid:
        cleaned = None
        try:
            from shapely.validation import make_valid  # type: ignore

            cleaned = make_valid(poly)
        except Exception:  # noqa: BLE001
            try:
                cleaned = poly.buffer(0)
            except Exception:  # noqa: BLE001
                cleaned = None
        if cleaned is None or cleaned.is_empty:
            return []
        if getattr(cleaned, "geom_type", "") != "Polygon":
            # make_valid міг розпасти на MultiPolygon/Collection — рекурсія.
            return _iter_extrudable_polygons(cleaned, min_area_m2)
        poly = cleaned

    # Відкидаємо вироджене (нульова площа / занадто мало точок у кільці).
    try:
        if poly.area <= min_area_m2:
            return []
        if len(poly.exterior.coords) < 4:
            return []
    except Exception:  # noqa: BLE001
        return []
    return [poly]


@dataclass
class GenerationPipelineResult:
    terrain_mesh: Optional[trimesh.Trimesh]
    terrain_provider: Any
    road_cut_mask: Optional[BaseGeometry]
    road_height_m: Optional[float]
    road_embed_m: Optional[float]


def _compute_road_dimensions(
    *,
    request: Any,
    scale_factor: Optional[float],
    stl_extra_embed_m: float,
    road_height_m: Optional[float],
    road_embed_m: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    if scale_factor and scale_factor > 0:
        if road_height_m is None:
            road_height_m = float(request.road_height_mm) / float(scale_factor)
        if road_embed_m is None:
            road_embed_m = float(request.road_embed_mm) / float(scale_factor)
            if stl_extra_embed_m > 0:
                road_embed_m += float(stl_extra_embed_m)
    return road_height_m, road_embed_m


def process_generation_stage(
    *,
    task: Any,
    request: Any,
    scale_factor: Optional[float],
    bbox_meters: Any,
    latlon_bbox: Any,
    source_crs: Any,
    elevation_ref_m: float,
    baseline_offset_m: float,
    building_geometries_for_flatten: Any,
    merged_roads_geom_local: Optional[BaseGeometry],
    merged_roads_geom: Optional[BaseGeometry],
    building_union_local: Optional[BaseGeometry],
    gdf_water_local: Any,
    global_center: Any,
    zone_polygon_local: Optional[BaseGeometry],
    groove_clearance_mm: float,
    stl_extra_embed_m: float,
    zone_prefix: str = "",
    road_height_m: Optional[float] = None,
    road_embed_m: Optional[float] = None,
    road_cut_mask_override: Optional[BaseGeometry] = None,
) -> GenerationPipelineResult:
    terrain_mesh = None
    terrain_provider = None
    road_cut_mask = None

    if request.is_ams_mode:
        task.update_status("processing", 20, "Generating AMS flat terrain...")
        print(f"[INFO] {zone_prefix} AMS Mode: Generating flat layers. Scale factor: {scale_factor}")

        land_height_m = (1.0 / scale_factor) if scale_factor else 0.001

        try:
            poly_to_extrude = zone_polygon_local
            if poly_to_extrude is None:
                poly_to_extrude = box(*bbox_meters)

            if gdf_water_local is not None and not gdf_water_local.empty:
                try:
                    from shapely.ops import unary_union

                    water_union = unary_union(list(gdf_water_local.geometry.values))
                    if water_union and not water_union.is_empty:
                        diffed = poly_to_extrude.difference(water_union)
                        # Якщо вода ПОВНІСТЮ зʼїла зону (плитка цілком у воді) —
                        # difference дає порожнечу → бази не буде. Лишаємо суходіл як є
                        # (краще плоска база, ніж відсутня модель).
                        if diffed is not None and not diffed.is_empty:
                            poly_to_extrude = diffed
                            print(f"[INFO] {zone_prefix} AMS Mode: Subtracted water from flat land terrain.")
                        else:
                            print(f"[WARN] {zone_prefix} AMS Mode: water covers whole zone — keeping land base un-subtracted.")
                except Exception as exc:
                    print(f"[WARN] {zone_prefix} AMS Mode: Failed to subtract water from land: {exc}")

            # Вода могла РОЗБИТИ зону (річка через всю плитку) → difference дає
            # MultiPolygon/GeometryCollection, а extrude_polygon чекає ОДИН валідний
            # Polygon → виняток → terrain_mesh=None → AMS-мапа БЕЗ бази (непридатна).
            # Витягуємо КОЖЕН валідний, не-вироджений полігон, чистимо й екструдуємо
            # окремо, потім зшиваємо.
            _polys = _iter_extrudable_polygons(poly_to_extrude)
            if not _polys:
                print(f"[WARN] {zone_prefix} AMS Mode: no extrudable polygon after water-split — falling back to bbox plate.")
                _polys = _iter_extrudable_polygons(box(*bbox_meters))
            _meshes = []
            for _g in _polys:
                try:
                    _meshes.append(trimesh.creation.extrude_polygon(_g, height=land_height_m))
                except Exception as _ex:
                    print(f"[WARN] {zone_prefix} AMS Mode: extrude_polygon skipped a part: {_ex}")
            terrain_mesh = trimesh.util.concatenate(_meshes) if _meshes else None
            if terrain_mesh is None:
                print(f"[ERROR] {zone_prefix} AMS Mode: all parts failed to extrude — no base mesh.")
            terrain_provider = None
        except Exception as exc:
            print(f"[ERROR] {zone_prefix} AMS Terrain creation failed: {exc}")
            terrain_mesh = None
            terrain_provider = None
    else:
        print(f"[INFO] {zone_prefix} Realistic Mode: Calling create_terrain_mesh")

        if road_cut_mask_override is not None and not getattr(road_cut_mask_override, "is_empty", True):
            road_cut_mask = road_cut_mask_override
            print(f"[INFO] {zone_prefix} Using canonical bundle road_groove_mask for terrain cutting (inlay-aligned).")
        else:
            road_cut_mask = prepare_road_cut_mask(
                merged_roads_geom_local=merged_roads_geom_local,
                building_union_local=building_union_local,
                scale_factor=scale_factor,
                groove_clearance_mm=groove_clearance_mm,
                building_clearance_mm=0.2,
                zone_polygon_local=zone_polygon_local,
                min_printable_mm=0.4,
                zone_prefix=zone_prefix,
            )

        # ── Flat-map fast path ──────────────────────────────────────────
        # When terrain is OFF the base is a flat plate with no elevation detail
        # to represent. The default fine grid (~2m step → ~700k vertices) cost
        # ~230s here in per-vertex building/road flattening for a surface that is
        # visually flat — the groove booleans add the crisp cut edges regardless.
        # Use a coarse plate; massive speedup, no visible quality change.
        _terr_res = max(float(request.terrain_resolution), 1.0) if request.terrain_resolution is not None else 1.0
        _grid_step = getattr(request, "grid_step_m", None)
        _terrain_on = bool(getattr(request, "terrain_enabled", True))
        _z_scale = request.terrain_z_scale
        if not _terrain_on:
            _flat_step = float(os.getenv("FLAT_BASE_STEP_M", "8"))
            if _grid_step is None or _grid_step < _flat_step:
                _grid_step = _flat_step
            _terr_res = min(_terr_res, 60.0)
            _z_scale = 0.0  # guarantee a flat base (no elevation displacement)
        else:
            # Terrain ON: cap the grid to PRINTABLE resolution. Detail finer than
            # the ~0.4mm nozzle can't print, so tessellating finer only bloats the
            # mesh (was ~640k verts) and slows every stage. Relief is large-scale,
            # so a printable-resolution grid keeps it smooth. world_step = nozzle/scale.
            try:
                _nozzle_mm = float(os.getenv("PRINT_NOZZLE_MM", "0.4"))
                if scale_factor and float(scale_factor) > 0:
                    _print_step_m = _nozzle_mm / float(scale_factor)
                    if _grid_step is None or _grid_step < _print_step_m:
                        _grid_step = _print_step_m
                        print(f"[PERF] {zone_prefix}Terrain grid capped to printable step={_print_step_m:.2f}m (nozzle {_nozzle_mm}mm)")
            except Exception as _exc:
                print(f"[WARN] {zone_prefix}terrain grid cap skipped: {_exc}")
            print(f"[PERF] {zone_prefix}Flat map: coarse base grid step={_grid_step}m (skip dense terrain build)")
        # Друкований максимум рельєфу: гори (Карпати/Альпи) мають перепад у сотні
        # метрів → mesh надто крутий, boolean-груви ламаються, та й друкувати гостро.
        # Стискаємо так, щоб рельєф у МОДЕЛІ не перевищував ~TERRAIN_MAX_RELIEF_MM.
        # world_cap = target_mm / scale_factor. Звичайні міста (нижчий перепад)
        # не зачіпаються — компресія спрацьовує лише коли реально високі гори.
        # Кап ПРОПОРЦІЙНИЙ розміру моделі (раніше було жорстко 28мм для ВСІХ
        # розмірів → на XL 240мм це лише ~12% висоти (пласко), на S 55мм ~50%
        # (крихко/гостро)). Якоримо 80мм→28мм і масштабуємо: рельєф ≈ 35% висоти
        # моделі, клемп [14..55]мм. Так гори виглядають однаково виразно на S/M/L/XL.
        _max_relief_m = None
        # Цільовий ВИДИМИЙ рельєф (рівнинні міста підсилюються до нього). ~10% ширини
        # моделі, але не вище 0.7×кап (щоб ніколи не злитись із компресією гір).
        _target_relief_m = None
        try:
            _base_relief_mm = float(os.getenv("TERRAIN_MAX_RELIEF_MM", "28"))
            _model_mm = float(getattr(request, "model_size_mm", 80.0) or 80.0)
            _max_relief_mm = max(14.0, min(_base_relief_mm * (_model_mm / 80.0), 55.0))
            _target_pct = float(os.getenv("TERRAIN_TARGET_RELIEF_PCT", "0.10"))
            _target_relief_mm = max(6.0, min(_target_pct * _model_mm, 0.7 * _max_relief_mm))
            if scale_factor and float(scale_factor) > 0 and _max_relief_mm > 0:
                _max_relief_m = _max_relief_mm / float(scale_factor)
                _target_relief_m = _target_relief_mm / float(scale_factor)
        except Exception:
            _max_relief_m = None
            _target_relief_m = None
        terrain_mesh, terrain_provider = create_terrain_mesh(
            bbox_meters,
            z_scale=_z_scale,
            resolution=_terr_res,
            latlon_bbox=latlon_bbox,
            source_crs=source_crs,
            terrarium_zoom=request.terrarium_zoom,
            elevation_ref_m=elevation_ref_m,
            baseline_offset_m=baseline_offset_m,
            base_thickness=(float(request.terrain_base_thickness_mm) / float(scale_factor)) if scale_factor else 5.0,
            max_relief_m=_max_relief_m,
            target_relief_m=_target_relief_m,
            flatten_buildings=bool(getattr(request, 'flatten_buildings_on_terrain', True)),
            building_geometries=building_geometries_for_flatten,
            flatten_roads=False,
            road_geometries=merged_roads_geom_local or merged_roads_geom,
            smoothing_sigma=float(request.terrain_smoothing_sigma) if request.terrain_smoothing_sigma is not None else 0.0,
            water_geometries=None,
            water_depth_m=0.0,
            global_center=global_center,
            bbox_is_local=True,
            subdivide=bool(request.terrain_subdivide),
            subdivide_levels=int(request.terrain_subdivide_levels),
            zone_polygon=zone_polygon_local,
            grid_step_m=_grid_step,
            road_polygons_for_cutting=None,
        )

    road_height_m, road_embed_m = _compute_road_dimensions(
        request=request,
        scale_factor=scale_factor,
        stl_extra_embed_m=stl_extra_embed_m,
        road_height_m=road_height_m,
        road_embed_m=road_embed_m,
    )

    return GenerationPipelineResult(
        terrain_mesh=terrain_mesh,
        terrain_provider=terrain_provider,
        road_cut_mask=road_cut_mask,
        road_height_m=road_height_m,
        road_embed_m=road_embed_m,
    )
