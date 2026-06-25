from __future__ import annotations

from typing import Any, Optional

import trimesh
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from services.detail_layer_utils import clamp_mesh_to_terrain_floor
from services.inlay_fit import InlayFitConfig
from services.printable_3d_validator import validate_road_mesh
from services.processing_results import RoadLayerResult, RoadProcessingResult
from services.road_processor import create_road_surface_cap, process_roads


def _iter_polygons(geometry: BaseGeometry | None) -> list[Any]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return []
    if getattr(geometry, "geom_type", "") == "Polygon":
        return [geometry]
    return [geom for geom in getattr(geometry, "geoms", []) if getattr(geom, "geom_type", "") == "Polygon"]


def _rebuild_road_mesh_from_mask(
    *,
    road_polygons: BaseGeometry | None,
    terrain_provider: Any,
    scale_factor: float,
    road_height_m: float,
    road_embed_m: float,
) -> trimesh.Trimesh | None:
    if road_polygons is None or getattr(road_polygons, "is_empty", True) or terrain_provider is None:
        return None
    meshes: list[trimesh.Trimesh] = []
    cap_thickness_m = max(float(road_height_m) + float(road_embed_m), 0.001)
    for poly in _iter_polygons(road_polygons):
        try:
            mesh = create_road_surface_cap(
                poly,
                terrain_provider,
                scale_factor=float(scale_factor),
                # Keep the road insert flush with the terrain surface.
                # Physical road thickness is still preserved by cap_thickness_m,
                # but the visible top should not float above the relief.
                top_z_offset=0.0,
                cap_thickness_m=float(cap_thickness_m),
            )
        except Exception:
            mesh = None
        if mesh is not None and len(mesh.vertices) > 0 and len(mesh.faces) > 0:
            meshes.append(mesh)
    if not meshes:
        return None
    try:
        return trimesh.util.concatenate(meshes)
    except Exception:
        return meshes[0]


def process_road_layer(
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
    road_height_m: float,
    road_embed_m: float,
    stl_extra_embed_m: float,
    fit_config: Optional[InlayFitConfig] = None,
    road_polygons_override: Any = None,
) -> RoadLayerResult:
    road_mesh = None
    road_result = None
    road_cut_source = merged_roads_geom_local
    road_clip_polygon = zone_polygon_local

    # In single-zone/bbox mode we often do not have an explicit polygon clip.
    # Derive one from the actual terrain mesh bounds so road inserts end with
    # exact axis-aligned 90-degree cuts on the model border.
    if road_clip_polygon is None and terrain_mesh is not None and len(terrain_mesh.vertices) > 0:
        try:
            bounds = terrain_mesh.bounds
            road_clip_polygon = box(
                float(bounds[0][0]),
                float(bounds[0][1]),
                float(bounds[1][0]),
                float(bounds[1][1]),
            )
        except Exception:
            road_clip_polygon = zone_polygon_local

    if not (scale_factor and scale_factor > 0 and (terrain_provider is not None or request.is_ams_mode)):
        return RoadLayerResult(mesh=None, road_result=None, road_cut_source=road_cut_source)

    if not getattr(request, "include_roads", True) or G_roads is None or len(G_roads) <= 0:
        return RoadLayerResult(mesh=None, road_result=None, road_cut_source=road_cut_source)

    task.update_status("processing", 40, "Generating roads (bridges + draping)...")
    total_road_embed_m = road_embed_m

    print("[ROAD DEBUG] === Road mesh creation ===")
    print(f"[ROAD DEBUG] road_height_m={road_height_m}, road_embed_m={road_embed_m}")
    print(f"[ROAD DEBUG] total_road_embed_m={total_road_embed_m}, stl_extra_embed_m={stl_extra_embed_m}")
    print(
        f"[ROAD DEBUG] scale_factor={scale_factor}, "
        f"road_height_mm={request.road_height_mm}, road_embed_mm={request.road_embed_mm}"
    )

    fit = fit_config or InlayFitConfig()
    effective_gap_fill_mm = float(getattr(request, "road_gap_fill_threshold_mm", 0.0))
    min_road_width_m = None
    road_mask_cleanup_mm = max(float(getattr(request, "tiny_feature_threshold_mm", 0.0) or 0.0), 0.0)

    road_result = process_roads(
        G_roads=G_roads,
        merged_roads=merged_roads_geom_local,
        road_height=road_height_m,
        road_embed=total_road_embed_m,
        terrain_provider=terrain_provider,
        global_center=global_center,
        scale_factor=float(scale_factor),
        water_geometries=water_geoms_for_bridges,
        width_multiplier=float(road_width_multiplier_effective),
        clip_polygon=road_clip_polygon,
        building_polygons=building_union_local,
        tiny_feature_threshold_mm=float(road_mask_cleanup_mm),
        merge_gap_mm=float(effective_gap_fill_mm),
        building_clearance_mm=0.0,
        clearance_mm=float(fit.insert_side_clearance_mm),
        min_width_m=min_road_width_m,
        return_result=True,
    )
    road_mesh = road_result.mesh if road_result is not None else None
    road_cut_source = road_result.cutting_polygons if road_result is not None else merged_roads_geom_local

    source_polygons_for_mesh = road_polygons_override
    if source_polygons_for_mesh is None and road_result is not None:
        source_polygons_for_mesh = getattr(road_result, "source_polygons", None)

    if source_polygons_for_mesh is not None and not getattr(source_polygons_for_mesh, "is_empty", True):
        rebuilt_road_mesh = _rebuild_road_mesh_from_mask(
            road_polygons=source_polygons_for_mesh,
            terrain_provider=terrain_provider,
            scale_factor=float(scale_factor),
            road_height_m=float(road_height_m),
            road_embed_m=float(total_road_embed_m),
        )
        if rebuilt_road_mesh is not None and len(rebuilt_road_mesh.vertices) > 0:
            road_mesh = rebuilt_road_mesh
            road_cut_source = source_polygons_for_mesh
            if road_result is None:
                road_result = RoadProcessingResult(
                    mesh=road_mesh,
                    source_polygons=source_polygons_for_mesh,
                    cutting_polygons=source_polygons_for_mesh,
                )
            else:
                road_result.source_polygons = source_polygons_for_mesh
                road_result.cutting_polygons = source_polygons_for_mesh

    if road_mesh is None:
        print("[ROAD DEBUG] Road mesh is None after process_roads()")
        return RoadLayerResult(mesh=None, road_result=road_result, road_cut_source=road_cut_source)

    print(f"[ROAD DEBUG] Road mesh created: {len(road_mesh.vertices)} verts, {len(road_mesh.faces)} faces")
    print(f"[ROAD DEBUG] Road Z range: [{road_mesh.bounds[0][2]:.4f}, {road_mesh.bounds[1][2]:.4f}]")
    if terrain_mesh is not None:
        print(f"[ROAD DEBUG] Terrain Z range: [{terrain_mesh.bounds[0][2]:.4f}, {terrain_mesh.bounds[1][2]:.4f}]")

    road_mesh = clamp_mesh_to_terrain_floor(road_mesh, terrain_mesh, label="ROAD DEBUG")
    if road_mesh is not None:
        print(f"[ROAD DEBUG] Road Z range after clamp: [{road_mesh.bounds[0][2]:.4f}, {road_mesh.bounds[1][2]:.4f}]")

    if road_mesh is not None and scale_factor and float(scale_factor) > 0:
        road_mesh = validate_road_mesh(road_mesh, scale_factor=float(scale_factor))

    return RoadLayerResult(
        mesh=road_mesh,
        road_result=road_result,
        road_cut_source=road_cut_source,
    )
