from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Optional

import trimesh
from geopandas import GeoDataFrame
from shapely.geometry.base import BaseGeometry

from services.boolean_backends import BooleanBackend
from services.buildings_pipeline import process_building_layer
from services.canonical_mask_bundle import CanonicalMaskBundle
from services.detail_layer_utils import MIN_LAND_WIDTH_MODEL_MM, MICRO_REGION_THRESHOLD_MM, model_mm_to_world_m
from services.groove_pipeline import GrooveCutResult, cut_inlay_grooves, prepare_road_cut_mask
from services.inlay_fit import InlayFitConfig
from services.parks_pipeline import process_park_layer
from services.processing_results import GreenAreaProcessingResult, RoadLayerResult, RoadProcessingResult
from services.roads_pipeline import process_road_layer
from services.water_layer_pipeline import process_water_layer


@dataclass
class DetailLayerPipelineResult:
    terrain_mesh: Optional[trimesh.Trimesh]
    road_mesh: Optional[trimesh.Trimesh]
    road_result: Optional[RoadProcessingResult]
    road_cut_source: Any
    road_groove_mask: Optional[BaseGeometry]
    building_meshes: Any
    water_mesh: Optional[trimesh.Trimesh]
    water_cut_polygons: Optional[BaseGeometry]
    parks_mesh: Optional[trimesh.Trimesh]
    parks_result: Optional[GreenAreaProcessingResult]
    groove_result: Optional[GrooveCutResult]
    merged_building_mesh: Optional[trimesh.Trimesh]
    support_meshes: Any
    building_footprints: Optional[BaseGeometry]
    landmark_centroids: Optional[list] = None  # центроїди визначних місць (бронзова Landmark-деталь)


@dataclass
class CanonicalRoadMasks:
    road_insert_mask: Optional[BaseGeometry]
    road_groove_mask: Optional[BaseGeometry]
    building_exclusion_mask: Optional[BaseGeometry]
    support_exclusion_mask: Optional[BaseGeometry]


def _build_canonical_road_masks(
    *,
    road_insert_source: Optional[BaseGeometry],
    building_union_local: Optional[BaseGeometry],
    scale_factor: Optional[float],
    groove_clearance_mm: float,
    tiny_feature_threshold_mm: float,
    road_gap_fill_threshold_mm: float,
    zone_polygon_local: Optional[BaseGeometry],
    zone_prefix: str,
    road_groove_mask_override: Optional[BaseGeometry] = None,
    preserve_exact_masks: bool = False,
) -> CanonicalRoadMasks:
    # ── groove clearance in world metres ─────────────────────────────────────
    # This is used to compute building_exclusion_mask as road_insert.buffer(clearance).
    # Both paths (preserve-exact and normal) use the same formula so buildings are
    # always clipped at exactly groove_clearance_m from the road surface edge.
    groove_clearance_m = (
        float(groove_clearance_mm) / float(scale_factor)
        if scale_factor and float(scale_factor) > 0
        else 0.0
    )

    road_insert_mask = road_insert_source
    building_exclusion_mask = road_insert_mask
    support_exclusion_mask = road_insert_mask

    def _make_building_exclusion(road_mask, groove_mask):
        """
        building_exclusion = groove_mask

        groove_mask = road_insert.buffer(groove_clearance) extends
        groove_clearance INTO building footprints.  building.difference(groove_mask)
        clips building by groove_clearance from road-adjacent edges.
        """
        base = groove_mask if groove_mask is not None else road_mask
        if base is None or getattr(base, "is_empty", True):
            return road_mask
        return base

    # ── preserve-exact path ───────────────────────────────────────────────────
    if (
        preserve_exact_masks
        and road_insert_mask is not None
        and not getattr(road_insert_mask, "is_empty", True)
        and road_groove_mask_override is not None
        and not getattr(road_groove_mask_override, "is_empty", True)
    ):
        road_groove_mask = road_groove_mask_override
        try:
            support_exclusion_mask = road_groove_mask.union(road_insert_mask).buffer(0)
        except Exception:
            support_exclusion_mask = road_groove_mask
        building_exclusion_mask = _make_building_exclusion(road_insert_mask, road_groove_mask)
        return CanonicalRoadMasks(
            road_insert_mask=road_insert_mask,
            road_groove_mask=road_groove_mask,
            building_exclusion_mask=building_exclusion_mask,
            support_exclusion_mask=support_exclusion_mask,
        )

    # ── normal path ───────────────────────────────────────────────────────────
    if road_insert_mask is not None and not getattr(road_insert_mask, "is_empty", True):
        try:
            road_insert_mask = road_insert_mask.buffer(0)
        except Exception:
            pass

    # Road keeps its full shape — buildings yield to roads (buildings are clipped
    # by building_exclusion_mask = road_groove_mask, which is computed below).
    # DO NOT subtract buildings from road here.
    road_for_groove = road_insert_mask

    if road_groove_mask_override is not None and not getattr(road_groove_mask_override, "is_empty", True):
        road_groove_mask = road_groove_mask_override
    else:
        road_groove_mask = prepare_road_cut_mask(
            merged_roads_geom_local=road_for_groove,
            building_union_local=building_union_local,
            scale_factor=scale_factor,
            groove_clearance_mm=float(groove_clearance_mm),
            building_clearance_mm=0.0,
            zone_polygon_local=zone_polygon_local,
            min_printable_mm=float(tiny_feature_threshold_mm),
            road_gap_fill_threshold_mm=float(road_gap_fill_threshold_mm),
            zone_prefix=zone_prefix,
        )

    road_mask_for_support = road_insert_mask or road_for_groove
    if road_groove_mask is not None and not getattr(road_groove_mask, "is_empty", True):
        try:
            support_exclusion_mask = road_groove_mask.union(road_mask_for_support).buffer(0) if road_mask_for_support is not None else road_groove_mask
        except Exception:
            support_exclusion_mask = road_groove_mask
    else:
        support_exclusion_mask = road_mask_for_support

    building_exclusion_mask = _make_building_exclusion(road_insert_mask, road_groove_mask)

    return CanonicalRoadMasks(
        road_insert_mask=road_insert_mask,
        road_groove_mask=road_groove_mask,
        building_exclusion_mask=building_exclusion_mask,
        support_exclusion_mask=support_exclusion_mask,
    )


def process_detail_layers(
    *,
    task: Any,
    request: Any,
    scale_factor: Optional[float],
    terrain_provider: Any,
    terrain_mesh: Optional[trimesh.Trimesh],
    global_center: Any,
    G_roads: Any,
    water_geoms_for_bridges: Any,
    road_width_multiplier_effective: float,
    zone_polygon_local: Any,
    building_union_local: Any,
    merged_roads_geom_local: Any,
    road_cut_mask: Any,
    road_height_m: float,
    road_embed_m: float,
    stl_extra_embed_m: float,
    gdf_buildings_local: Optional[GeoDataFrame],
    gdf_water: Optional[GeoDataFrame],
    water_depth_m: float,
    gdf_green: Optional[GeoDataFrame],
    groove_clearance_mm: float,
    boolean_backend: Optional[BooleanBackend] = None,
    zone_prefix: str = "",
    canonical_mask_bundle: Optional[CanonicalMaskBundle] = None,
) -> DetailLayerPipelineResult:
    pipeline_start = time.perf_counter()

    def _log_stage(name: str, started_at: float) -> None:
        elapsed = time.perf_counter() - started_at
        total = time.perf_counter() - pipeline_start
        print(f"[TIMING] {zone_prefix}detail.{name}: {elapsed:.2f}s (detail total {total:.2f}s)")

    micro_region_threshold_mm = float(MICRO_REGION_THRESHOLD_MM)
    tiny_feature_threshold_mm = max(float(getattr(request, "tiny_feature_threshold_mm", 0.2)), 0.2)
    road_gap_fill_threshold_mm = max(
        float(getattr(request, "road_gap_fill_threshold_mm", 0.45)),
        float(tiny_feature_threshold_mm),
    )
    fit_config = InlayFitConfig(
        insert_side_clearance_mm=0.0,
        groove_side_clearance_mm=float(groove_clearance_mm),
    )
    # Roads keep the full groove clearance, but neighbouring inlays should not
    # inset by the same full amount or the shared white gap becomes visually
    # doubled. Use a split-fit for parks/water inserts.
    shared_inlay_fit_clearance_mm = float(fit_config.groove_side_clearance_mm) * 0.5

    # ① Replace small COMPACT terrain islands enclosed by road WITH road. This is the
    # one stage that has roads + parks (gdf_green) + buildings together AND owns the
    # source (merged_roads_geom_local) that process_road_layer turns into BOTH the
    # road inlay mesh AND road_cut_source — so the island is cut out of the terrain
    # AND filled with road, consistently, in the printed model. STRONG guards keep it
    # safe (this is the owner-sensitive road-merge area): only SMALL, COMPACT (low
    # aspect), >=70%-road-surrounded pockets that do NOT touch parks/buildings/zone-
    # edge are filled. Long strips between parallel roads (the courtyard-plate /
    # whack-a-mole regression) and any green/building-adjacent terrain are LEFT alone.
    # GATED OFF by default (ROAD_ISLAND_FILL_MM=0): proven no-op here because the
    # islands are NOT present in this source geometry — merged_roads_geom_local is the
    # FULL-WIDTH road, and the islands only appear DOWNSTREAM after the clearance/groove
    # narrowing exposes terrain between the narrowed roads. By then the geometry is
    # split between the canonical cut-mask and the centerline-built inlay mesh, so no
    # single injection reaches both. Kept (env-enablable) for the rare zone whose raw
    # road geometry genuinely encloses a compact terrain pocket; a real fix needs a
    # road-pipeline refactor that gives one island-aware source to cut + inlay.
    try:
        _rif_mm = float(os.environ.get("ROAD_ISLAND_FILL_MM", "0"))
    except (TypeError, ValueError):
        _rif_mm = 0.0
    if (
        _rif_mm > 0 and merged_roads_geom_local is not None
        and not getattr(merged_roads_geom_local, "is_empty", True)
        and scale_factor and float(scale_factor) > 0
        and zone_polygon_local is not None and not getattr(zone_polygon_local, "is_empty", True)
    ):
        try:
            from shapely.ops import unary_union as _uu
            _sf = float(scale_factor)
            _cap = float(model_mm_to_world_m(_rif_mm, _sf)) ** 2
            _touch = float(model_mm_to_world_m(0.4, _sf))
            _roads_g = merged_roads_geom_local
            _parks_g = getattr(canonical_mask_bundle, "parks_final", None) if canonical_mask_bundle is not None else None
            if (_parks_g is None or getattr(_parks_g, "is_empty", True)) and gdf_green is not None and not getattr(gdf_green, "empty", True):
                try:
                    _parks_g = _uu([g for g in gdf_green.geometry.values if g is not None and not getattr(g, "is_empty", True)]).buffer(0)
                except Exception:
                    _parks_g = None
            _occ = [g for g in (_roads_g, _parks_g, building_union_local) if g is not None and not getattr(g, "is_empty", True)]
            _terr = zone_polygon_local.difference(_uu(_occ)).buffer(0) if _occ else zone_polygon_local
            _pieces = [_terr] if getattr(_terr, "geom_type", "") == "Polygon" else list(getattr(_terr, "geoms", []))
            _zedge = zone_polygon_local.boundary
            _fill = []
            for _p in _pieces:
                if getattr(_p, "geom_type", "") != "Polygon" or _p.is_empty or _p.area > _cap:
                    continue
                _b = _p.bounds
                _dx, _dy = (_b[2] - _b[0]), (_b[3] - _b[1])
                if max(_dx, _dy) / max(min(_dx, _dy), 1e-6) > 3.0:
                    continue  # long strip -> would weld parallel roads
                _bnd = _p.boundary
                if getattr(_bnd, "length", 0.0) <= 0:
                    continue
                try:
                    if _bnd.intersection(_roads_g.buffer(_touch)).length / _bnd.length < 0.70:
                        continue  # not mostly road-surrounded -> leave
                except Exception:
                    continue
                _grow = _p.buffer(_touch)
                if _parks_g is not None and not getattr(_parks_g, "is_empty", True) and _grow.intersects(_parks_g):
                    continue
                if building_union_local is not None and not getattr(building_union_local, "is_empty", True) and _grow.intersects(building_union_local):
                    continue
                if _grow.intersects(_zedge):
                    continue  # at the tile border -> leave
                _fill.append(_p)
            if _fill:
                _fu = _uu(_fill)
                merged_roads_geom_local = _uu([merged_roads_geom_local, _fu]).buffer(0)
                print(
                    f"[INFO] {zone_prefix}road-island fill: replaced {len(_fill)} compact "
                    f"road-enclosed terrain islands ({_fu.area:.1f} m2) with road"
                )
        except Exception as _exc:
            print(f"[WARN] {zone_prefix}road-island fill failed: {_exc}")

    stage_start = time.perf_counter()
    preview_mode = os.environ.get("PREVIEW_MODE", "").lower() in ("1", "true", "yes")
    preview_roads_mask = (
        getattr(canonical_mask_bundle, "roads_final", None) if canonical_mask_bundle is not None else None
    )
    if preview_mode and preview_roads_mask is not None and not getattr(preview_roads_mask, "is_empty", True):
        road_layer = RoadLayerResult(
            mesh=None,
            road_result=RoadProcessingResult(
                mesh=None,
                source_polygons=preview_roads_mask,
                cutting_polygons=preview_roads_mask,
            ),
            road_cut_source=preview_roads_mask,
        )
        print(f"[INFO] {zone_prefix}PREVIEW_MODE: skipped expensive 3D road mesh; using canonical 2D road mask")
    else:
        road_layer = process_road_layer(
            task=task,
            request=request,
            scale_factor=scale_factor,
            terrain_provider=terrain_provider,
            terrain_mesh=terrain_mesh,
            global_center=global_center,
            G_roads=G_roads,
            water_geoms_for_bridges=water_geoms_for_bridges,
            road_width_multiplier_effective=road_width_multiplier_effective,
            zone_polygon_local=zone_polygon_local,
            building_union_local=building_union_local,
            merged_roads_geom_local=merged_roads_geom_local,
            road_height_m=road_height_m,
            road_embed_m=road_embed_m,
            stl_extra_embed_m=stl_extra_embed_m,
            fit_config=fit_config,
            road_polygons_override=preview_roads_mask,
        )
    _log_stage("roads", stage_start)
    road_mesh = road_layer.mesh
    road_result = road_layer.road_result
    road_cut_source = road_layer.road_cut_source
    road_insert_source = (
        getattr(canonical_mask_bundle, "roads_final", None)
        if canonical_mask_bundle is not None and getattr(canonical_mask_bundle, "roads_final", None) is not None
        else (
            road_result.source_polygons
            if road_result is not None and getattr(road_result, "source_polygons", None) is not None
            else road_cut_source
        )
    )
    canonical_masks = _build_canonical_road_masks(
        road_insert_source=road_insert_source,
        building_union_local=building_union_local,
        scale_factor=scale_factor,
        groove_clearance_mm=float(fit_config.groove_side_clearance_mm),
        tiny_feature_threshold_mm=float(tiny_feature_threshold_mm),
        road_gap_fill_threshold_mm=float(road_gap_fill_threshold_mm),
        zone_polygon_local=zone_polygon_local,
        zone_prefix=zone_prefix,
        road_groove_mask_override=(
            getattr(canonical_mask_bundle, "road_groove_mask", None) if canonical_mask_bundle is not None else None
        ),
        preserve_exact_masks=canonical_mask_bundle is not None,
    )
    road_insert_exclusion_polygons = canonical_masks.road_insert_mask
    canonical_road_groove_mask = canonical_masks.road_groove_mask
    support_exclusion_mask = canonical_masks.support_exclusion_mask
    building_exclusion_mask = canonical_masks.building_exclusion_mask

    stage_start = time.perf_counter()
    building_layer = process_building_layer(
        task=task,
        request=request,
        scale_factor=scale_factor,
        terrain_provider=terrain_provider,
        global_center=global_center,
        stl_extra_embed_m=stl_extra_embed_m,
        gdf_buildings_local=gdf_buildings_local,
        support_exclusion_polygons=support_exclusion_mask,
        road_insert_mask=road_insert_exclusion_polygons,
        road_groove_mask=canonical_road_groove_mask,
        building_exclusion_polygons=building_exclusion_mask,  # buildings clipped by road+groove; road mesh stays whole
    )
    _log_stage("buildings", stage_start)
    building_meshes = building_layer.meshes
    merged_building_mesh = building_layer.merged_mesh
    support_meshes = building_layer.support_meshes

    stage_start = time.perf_counter()
    water_layer = process_water_layer(
        task=task,
        request=request,
        scale_factor=scale_factor,
        terrain_provider=terrain_provider,
        global_center=global_center,
        gdf_water=gdf_water,
        water_depth_m=water_depth_m,
        road_polygons=road_insert_exclusion_polygons or canonical_road_groove_mask,
        building_polygons=building_union_local,
        coordinates_already_local=True,
        zone_prefix=zone_prefix,
        water_polygons_override=(
            getattr(canonical_mask_bundle, "water_final", None) if canonical_mask_bundle is not None else None
        ),
        fit_clearance_mm=float(shared_inlay_fit_clearance_mm),
    )
    _log_stage("water", stage_start)
    water_mesh = water_layer.mesh
    water_cut_polygons = water_layer.cutting_polygons

    stage_start = time.perf_counter()
    park_layer = process_park_layer(
        task=task,
        request=request,
        scale_factor=scale_factor,
        terrain_provider=terrain_provider,
        terrain_mesh=terrain_mesh,
        global_center=global_center,
        zone_polygon_local=zone_polygon_local,
        # Park inserts must respect the actual road groove footprint; otherwise the
        # final park inlay can compete with the road groove at shared boundaries.
        road_cut_source=canonical_road_groove_mask or road_insert_exclusion_polygons,
        building_union_local=building_union_local,
        water_polygons=water_cut_polygons,
        road_exclusion_clearance_mm=0.0,
        fit_clearance_mm=float(shared_inlay_fit_clearance_mm),
        gdf_green=gdf_green,
        zone_prefix=zone_prefix,
        park_polygons_override=(
            getattr(canonical_mask_bundle, "parks_final", None) if canonical_mask_bundle is not None else None
        ),
    )
    _log_stage("parks", stage_start)
    parks_mesh = park_layer.mesh
    parks_result = park_layer.parks_result

    has_road_grooves = (
        not request.is_ams_mode and terrain_mesh is not None and road_mesh is not None and scale_factor and scale_factor > 0
    )
    has_park_grooves = (
        not request.is_ams_mode and terrain_mesh is not None and parks_mesh is not None and scale_factor and scale_factor > 0
    )
    has_water_grooves = (
        not request.is_ams_mode and terrain_mesh is not None and water_mesh is not None and scale_factor and scale_factor > 0
    )

    groove_result = None
    # Різак паза зʼєднувача — щоб CDT-гілка вирізала паз ПОКИ меш герметичний
    # (до boolean-грувів парків/води, які відкривають меш і ламають manifold).
    _notch_cutter = None
    if (getattr(request, "map_connector", False) and terrain_mesh is not None
            and scale_factor and float(scale_factor) > 0 and not preview_mode):
        try:
            from services.flat_plate_pipeline import (
                build_map_connector_geometry as _bmcg,
                build_flat_layer_mesh_from_mask as _bflm,
                parse_connector_azimuths as _pca,
            )
            _sfn = float(scale_factor)
            _flz = float(terrain_mesh.bounds[0][2])
            _mh = float(terrain_mesh.bounds[1][2]) - _flz
            _dmm = float(getattr(request, "map_connector_depth_mm", 2.0) or 2.0)
            _dm = min(_dmm / _sfn, max(_mh * 0.6, 0.0))
            _ntc0, _ = _bmcg(
                zone_polygon_local,
                edges=str(getattr(request, "map_connector_edges", "NSEW") or "NSEW"),
                span_mm=float(getattr(request, "map_connector_span_mm", 10.0) or 10.0),
                length_mm=float(getattr(request, "map_connector_length_mm", 15.0) or 15.0),
                waist_frac=0.5,
                clearance_mm=float(getattr(request, "map_connector_clearance_mm", 0.03) or 0.03),
                export_scale_factor=_sfn,
                key_edges=(str(getattr(request, "map_connector_key_edges", "") or "") or None),
                edge_dirs=_pca(getattr(request, "map_connector_edge_az", "")),
                key_dirs=_pca(getattr(request, "map_connector_key_az", "")),
            )
            if _ntc0 is not None and _dm > 1e-6:
                _epsn = max(_mh * 0.01, 0.5 / _sfn)
                # ГЛИБИНА ПАЗА ≤ ЛОКАЛЬНОЇ ТОВЩИНИ МАТЕРІАЛУ: фронт шле плиту 0.3мм,
                # і на низькому рельєфі (край біля ріки) матеріалу над пазом < 2мм →
                # паз ПРОБИВАВ поверхню наскрізь (у дірі видно дороги — «чорні
                # штрихи в пазі»). Per-полігон: семплимо верх терену над кожним
                # пазом і ріжемо не глибше (мін.дах 0.5мм; мін.глибина паза 0.6мм).
                try:
                    import numpy as _npn2
                    import trimesh as _tmn3
                    _tN = terrain_mesh.face_normals
                    _tc = terrain_mesh.triangles_center
                    _topc = _tc[_tN[:, 2] > 0.3]
                    _geoms_n = list(getattr(_ntc0, "geoms", [_ntc0]))
                    _cutters = []
                    _roof_m = 0.5 / _sfn      # мін. 0.5мм даху над пазом
                    _mind_m = 0.6 / _sfn      # мін. глибина паза 0.6мм (інакше не тримає)
                    _wmask = getattr(canonical_mask_bundle, "water_final", None) \
                        if canonical_mask_bundle is not None else None
                    for _gp in _geoms_n:
                        if getattr(_gp, "is_empty", True):
                            continue
                        # ВОДА над пазом: водна ванна (2мм) глибша за дах паза →
                        # паз може ВІДКРИТИСЬ у ванну («пустота»). БУЛО: пропускали паз
                        # на БУДЬ-ЯКОМУ дотику води → у серії/панно на водних краях
                        # ЗНИКАВ конектор («з однієї сторони немає пазу»; напр. r5_c3
                        # North=0). СТАЛО: пропускаємо ЛИШЕ якщо паз майже ПОВНІСТЮ під
                        # водою (>85% площі) — там сенсу немає. Інакше пускаємо в per-edge
                        # depth-reduction нижче: воно семплить МІН верху терену над пазом
                        # (над водою = дно ванни) і ріже не глибше, лишаючи 0.5мм дах →
                        # паз у СУЦІЛЬНОМУ матеріалі, БЕЗ прориву у ванну. На геть тонкому
                        # краю depth-reduction сам пропустить (матеріал < паз+дах).
                        try:
                            if _wmask is not None and not getattr(_wmask, "is_empty", True):
                                _inter = _gp.intersection(_wmask)
                                _wfrac = (float(_inter.area) / float(_gp.area)) if (_gp.area > 0 and not getattr(_inter, "is_empty", True)) else 0.0
                                if _wfrac > 0.85:
                                    print(f"[GROOVE] {zone_prefix}notch skipped on one edge: "
                                          f"паз під водою на {_wfrac*100:.0f}% (немає суходолу)")
                                    continue
                        except Exception:  # noqa: BLE001
                            pass
                        _gb = _gp.bounds
                        _sel = ((_topc[:, 0] > _gb[0] - 1) & (_topc[:, 0] < _gb[2] + 1)
                                & (_topc[:, 1] > _gb[1] - 1) & (_topc[:, 1] < _gb[3] + 1))
                        _avail = (float(_topc[_sel][:, 2].min()) - _flz) if bool(_sel.any()) else _dm + _roof_m
                        _dp = min(_dm, max(_avail - _roof_m, 0.0))
                        if _dp < _mind_m:
                            print(f"[GROOVE] {zone_prefix}notch skipped on one edge: "
                                  f"матеріалу лише {_avail * _sfn:.2f}мм (< паз+дах)")
                            continue
                        _cp = _bflm(_gp, bottom_z_m=_flz - _epsn, thickness_m=_dp + _epsn,
                                    color=[128, 128, 128], min_area_m2=1e-12)
                        if _cp is not None and bool(getattr(_cp, "is_volume", False)):
                            _cutters.append(_cp)
                            if _dp < _dm - 1e-9:
                                print(f"[GROOVE] {zone_prefix}notch depth reduced to "
                                      f"{_dp * _sfn:.2f}мм on one edge (тонкий рельєф)")
                    _notch_cutter = _tmn3.util.concatenate(_cutters) if _cutters else None
                except Exception as _pdx:  # noqa: BLE001
                    print(f"[GROOVE] {zone_prefix}per-edge notch depth failed ({_pdx}); uniform")
                    _notch_cutter = _bflm(_ntc0, bottom_z_m=_flz - _epsn, thickness_m=_dm + _epsn,
                                          color=[128, 128, 128], min_area_m2=1e-12)
                if _notch_cutter is not None and not bool(getattr(_notch_cutter, "is_volume", False)):
                    _notch_cutter = None
        except Exception as _ncx:  # noqa: BLE001
            print(f"[GROOVE] {zone_prefix}notch cutter build failed (non-fatal): {_ncx}")
            _notch_cutter = None
    if preview_mode:
        print(f"[INFO] {zone_prefix}PREVIEW_MODE: skipped groove cutting; local preview uses surface decals")
    elif has_road_grooves or has_park_grooves or has_water_grooves:
        try:
            stage_start = time.perf_counter()
            groove_result = cut_inlay_grooves(
                notch_cutter_mesh=_notch_cutter,
                terrain_mesh=terrain_mesh,
                road_mesh=road_mesh,
                parks_mesh=parks_mesh,
                water_mesh=water_mesh,
                road_cut_mask=canonical_road_groove_mask or road_cut_mask,
                merged_roads_geom_local=road_insert_exclusion_polygons or road_cut_source,
                parks_polygons=(
                    getattr(canonical_mask_bundle, "parks_final", None)
                    if canonical_mask_bundle is not None and getattr(canonical_mask_bundle, "parks_final", None) is not None
                    else (parks_result.processed_polygons if parks_result is not None else None)
                ),
                water_polygons=(
                    getattr(canonical_mask_bundle, "water_final", None)
                    if canonical_mask_bundle is not None and getattr(canonical_mask_bundle, "water_final", None) is not None
                    else water_cut_polygons
                ),
                building_polygons=building_union_local,
                scale_factor=float(scale_factor),
                groove_clearance_mm=float(fit_config.groove_side_clearance_mm),
                road_embed_m=road_embed_m,
                parks_embed_mm=float(request.parks_embed_mm),
                water_depth_m=water_depth_m,
                boolean_backend=boolean_backend,
                zone_prefix=zone_prefix,
                zone_polygon_local=zone_polygon_local,
                min_printable_mm=max(float(MIN_LAND_WIDTH_MODEL_MM), tiny_feature_threshold_mm),
                parks_groove_override=(
                    getattr(canonical_mask_bundle, "parks_groove_mask", None) if canonical_mask_bundle is not None else None
                ),
                water_groove_override=(
                    (
                        getattr(canonical_mask_bundle, "water_groove_mask", None)
                        or getattr(canonical_mask_bundle, "water_final", None)
                    )
                    if canonical_mask_bundle is not None
                    else None
                ),
                use_exact_masks=canonical_mask_bundle is not None,
                # CDT clean-wall груви ПРОПУСКАЄМО (→ boolean) на SERIES-тайлах
                # (elevation_ref_m: CDT перебудовує межу з interior-only griddata → рве
                # шов-Z сусідів). CONNECTOR тепер ДОЗВОЛЕНО: pre-groove паз пропускається
                # коли CDT активний (у full_generation_pipeline), а ПІЗНІЙ блок ріже паз у
                # ЧИСТИЙ герметичний CDT-терен (manifold, drift≈0, стінки паза чисті) —
                # замість boolean-комба на ВСІХ стінках грувів (те, що бачив користувач).
                cdt_allowed=(
                    getattr(request, "elevation_ref_m", None) is None
                ),
            )
            terrain_mesh = groove_result.terrain_mesh
            _log_stage("grooves", stage_start)
        except Exception as exc:
            print(f"[WARN] {zone_prefix} Failed to cut grooves: {exc}")
            import traceback

            traceback.print_exc()

    return DetailLayerPipelineResult(
        terrain_mesh=terrain_mesh,
        road_mesh=road_mesh,
        road_result=road_result,
        road_cut_source=road_cut_source,
        road_groove_mask=canonical_road_groove_mask or road_cut_mask,
        building_meshes=building_meshes,
        water_mesh=water_mesh,
        water_cut_polygons=water_cut_polygons,
        parks_mesh=parks_mesh,
        parks_result=parks_result,
        groove_result=groove_result,
        merged_building_mesh=merged_building_mesh,
        support_meshes=support_meshes,
        building_footprints=(
            getattr(canonical_mask_bundle, "buildings_footprints", None)
            if canonical_mask_bundle is not None and getattr(canonical_mask_bundle, "buildings_footprints", None) is not None
            else getattr(building_layer, "footprints", None)
        ),
        landmark_centroids=getattr(building_layer, "landmark_centroids", None),
    )
