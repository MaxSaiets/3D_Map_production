from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import trimesh

from services.export_pipeline import ExportPipelineResult, export_generation_outputs
from services.terrain_pipeline_utils import (
    compute_water_depth_m,
    compute_water_surface_thickness_m,
    resolve_generation_source_crs,
)
from services.terrain_generator import create_terrain_mesh
from services.water_processor import process_water_surface


@dataclass
class TerrainOnlyPipelineResult:
    terrain_mesh: Optional[trimesh.Trimesh]
    terrain_provider: Any
    water_mesh: Optional[trimesh.Trimesh]
    export_result: ExportPipelineResult


def run_terrain_only_pipeline(
    *,
    task: Any,
    request: Any,
    task_id: str,
    output_dir: Path,
    bbox_meters: Any,
    latlon_bbox: Any,
    scale_factor: Optional[float],
    gdf_buildings: Any,
    G_roads: Any,
    gdf_water: Any,
    global_center: Any,
    reference_xy_m: Optional[float],
) -> TerrainOnlyPipelineResult:
    task.update_status("processing", 25, "Створення рельєфу для тестування (з водою, без будівель, доріг)...")

    source_crs = resolve_generation_source_crs(
        gdf_buildings=gdf_buildings,
        G_roads=G_roads,
        global_center=global_center,
        allow_global_center_fallback=True,
    )

    has_water = gdf_water is not None and not gdf_water.empty
    water_depth_m = compute_water_depth_m(
        water_depth_mm=float(request.water_depth),
        scale_factor=scale_factor,
    ) if has_water else None

    water_geoms_for_terrain = None
    water_depth_for_terrain = 0.0
    if has_water and water_depth_m is not None and water_depth_m > 0:
        water_geoms_for_terrain = list(gdf_water.geometry.values)
        water_depth_for_terrain = float(water_depth_m)

    elevation_ref_m = getattr(request, "elevation_ref_m", None)
    baseline_offset_m = getattr(request, "baseline_offset_m", 0.0)

    terrain_mesh, terrain_provider = create_terrain_mesh(
        bbox_meters,
        z_scale=request.terrain_z_scale,
        resolution=request.terrain_resolution,
        latlon_bbox=latlon_bbox,
        source_crs=source_crs,
        terrarium_zoom=request.terrarium_zoom,
        elevation_ref_m=elevation_ref_m,
        baseline_offset_m=baseline_offset_m,
        base_thickness=(float(request.terrain_base_thickness_mm) / float(scale_factor)) if scale_factor else 5.0,
        flatten_buildings=False,
        building_geometries=None,
        flatten_roads=False,
        road_geometries=None,
        smoothing_sigma=float(request.terrain_smoothing_sigma) if request.terrain_smoothing_sigma is not None else 0.0,
        water_geometries=water_geoms_for_terrain,
        water_depth_m=water_depth_for_terrain,
        subdivide=bool(request.terrain_subdivide),
        subdivide_levels=int(request.terrain_subdivide_levels),
        global_center=global_center,
    )
    if terrain_mesh is None:
        raise ValueError("Terrain mesh не створено, але terrain_only=True. Переконайтеся, що terrain_enabled=True або вказано валідні координати.")

    water_mesh = None
    print(f"[DEBUG] Water check: has_water={has_water}, terrain_provider={'OK' if terrain_provider else 'None'}, water_depth_m={water_depth_m}")
    if has_water:
        print(f"[DEBUG] gdf_water: {len(gdf_water)} об'єктів")

    if has_water and terrain_provider is not None and water_depth_m is not None and water_depth_m > 0:
        task.update_status("processing", 30, "Створення води для тестування...")
        thickness_m = compute_water_surface_thickness_m(
            water_depth_mm=float(request.water_depth),
            water_depth_m=water_depth_m,
            scale_factor=scale_factor,
        )
        water_mesh = process_water_surface(
            gdf_water,
            thickness_m=float(thickness_m) if thickness_m is not None else 0.0,
            depth_meters=float(water_depth_m),
            terrain_provider=terrain_provider,
            global_center=global_center,
        )
        if water_mesh:
            print(f"Вода: {len(water_mesh.vertices)} вершин, {len(water_mesh.faces)} граней")
        else:
            print("[WARN] Water mesh не створено! Перевірте gdf_water та параметри")
    else:
        print(f"[WARN] Water не створюється: has_water={has_water}, terrain_provider={'OK' if terrain_provider else 'None'}, water_depth_m={water_depth_m}")

    task.update_status("processing", 90, "Експорт рельєфу та води (тестовий режим)...")
    if terrain_mesh is not None:
        print("[DEBUG] Terrain Mesh Stats BEFORE Export:")
        print(f"  Vertices: {len(terrain_mesh.vertices)}")
        print(f"  Faces: {len(terrain_mesh.faces)}")
        print(f"  Bounds: {terrain_mesh.bounds}")
        size = terrain_mesh.bounds[1] - terrain_mesh.bounds[0]
        print(f"  Size: {size}")
    else:
        print("[DEBUG] Terrain Mesh is NONE")

    if reference_xy_m:
        print(f"[DEBUG] Reference XY (Meters): {reference_xy_m}")

    export_result = export_generation_outputs(
        task=task,
        request=request,
        task_id=task_id,
        output_dir=output_dir,
        terrain_mesh=terrain_mesh,
        road_mesh=None,
        building_meshes=None,
        water_mesh=water_mesh,
        parks_mesh=None,
        reference_xy_m=reference_xy_m,
        preserve_z=bool(getattr(request, "elevation_ref_m", None) is not None),
        preserve_xy=bool(getattr(request, "preserve_global_xy", False)),
        include_preview_parts=False,
        completion_message="Рельєф та вода готові!",
    )

    return TerrainOnlyPipelineResult(
        terrain_mesh=terrain_mesh,
        terrain_provider=terrain_provider,
        water_mesh=water_mesh,
        export_result=export_result,
    )
