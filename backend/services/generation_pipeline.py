from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import trimesh
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from services.groove_pipeline import prepare_road_cut_mask
from services.terrain_generator import create_terrain_mesh


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
                        poly_to_extrude = poly_to_extrude.difference(water_union)
                        print(f"[INFO] {zone_prefix} AMS Mode: Subtracted water from flat land terrain.")
                except Exception as exc:
                    print(f"[WARN] {zone_prefix} AMS Mode: Failed to subtract water from land: {exc}")

            terrain_mesh = trimesh.creation.extrude_polygon(poly_to_extrude, height=land_height_m)
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

        terrain_mesh, terrain_provider = create_terrain_mesh(
            bbox_meters,
            z_scale=request.terrain_z_scale,
            resolution=max(float(request.terrain_resolution), 1.0) if request.terrain_resolution is not None else 1.0,
            latlon_bbox=latlon_bbox,
            source_crs=source_crs,
            terrarium_zoom=request.terrarium_zoom,
            elevation_ref_m=elevation_ref_m,
            baseline_offset_m=baseline_offset_m,
            base_thickness=(float(request.terrain_base_thickness_mm) / float(scale_factor)) if scale_factor else 5.0,
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
            grid_step_m=getattr(request, "grid_step_m", None),
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
