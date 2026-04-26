from __future__ import annotations

from typing import Any, Optional

import numpy as np
import trimesh
from geopandas import GeoDataFrame
from shapely.geometry.base import BaseGeometry

from services.detail_layer_utils import (
    MIN_LAND_WIDTH_MODEL_MM,
    clamp_mesh_to_terrain_floor,
    model_mm_to_world_m,
    prepare_green_areas_for_processing,
)
from services.green_processor import _add_strong_faceted_texture, _create_high_res_mesh, process_green_areas
from services.processing_results import GreenAreaProcessingResult, ParkLayerResult


def _iter_polygons(geometry: BaseGeometry | None) -> list[Any]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return []
    if getattr(geometry, "geom_type", "") == "Polygon":
        return [geometry]
    return [geom for geom in getattr(geometry, "geoms", []) if getattr(geom, "geom_type", "") == "Polygon"]


def _apply_fit_inset_to_override(
    geometry: BaseGeometry | None,
    *,
    scale_factor: float,
    fit_clearance_mm: float,
) -> BaseGeometry | None:
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    if scale_factor <= 0 or fit_clearance_mm <= 0:
        return geometry
    try:
        inset_m = model_mm_to_world_m(float(fit_clearance_mm), float(scale_factor))
    except Exception:
        inset_m = 0.0
    if inset_m <= 0:
        return geometry
    try:
        inset = geometry.buffer(-float(inset_m), join_style=1).buffer(0)
    except Exception:
        inset = geometry
    if inset is None or getattr(inset, "is_empty", True):
        return geometry
    return inset


def _rebuild_park_mesh_from_polygons(
    *,
    park_polygons: BaseGeometry | None,
    terrain_provider: Any,
    scale_factor: float,
    park_height_m: float,
    park_embed_m: float,
    global_center: Any,
) -> Optional[trimesh.Trimesh]:
    if park_polygons is None or getattr(park_polygons, "is_empty", True) or terrain_provider is None:
        return None

    try:
        target_edge_len_m = 3.5 / float(scale_factor)
        target_edge_len_m = max(2.0, min(float(target_edge_len_m), 12.0))
    except Exception:
        target_edge_len_m = 4.0

    meshes: list[trimesh.Trimesh] = []
    for poly in _iter_polygons(park_polygons):
        try:
            mesh = _create_high_res_mesh(poly, float(park_height_m), float(target_edge_len_m))
        except Exception:
            mesh = None
        if mesh is None or len(mesh.vertices) == 0:
            continue

        relative_height = None
        try:
            verts = mesh.vertices.copy()
            old_z = verts[:, 2].copy()
            ground_heights = terrain_provider.get_surface_heights_for_points(verts[:, :2])
            z_min = float(np.min(old_z))
            z_max = float(np.max(old_z))
            z_range = z_max - z_min
            relative_height = np.zeros_like(old_z)
            if z_range > 1e-6:
                relative_height = (old_z - z_min) / z_range
            verts[:, 2] = ground_heights - float(park_embed_m) + relative_height * float(park_height_m + park_embed_m)
            mesh.vertices = verts
            mesh._bottom_mask = relative_height <= 0.1
        except Exception:
            pass

        try:
            mesh = _add_strong_faceted_texture(
                mesh,
                float(park_height_m),
                float(scale_factor),
                original_polygon=poly,
                global_center=global_center,
                relative_heights=relative_height,
            )
        except Exception:
            pass

        if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
            meshes.append(mesh)

    if not meshes:
        return None

    try:
        global_min_z = min(float(np.min(mesh.vertices[:, 2])) for mesh in meshes if len(mesh.vertices) > 0)
        for mesh in meshes:
            if hasattr(mesh, "_bottom_mask"):
                mesh.vertices[mesh._bottom_mask, 2] = global_min_z
    except Exception:
        pass

    try:
        return trimesh.util.concatenate(meshes)
    except Exception:
        return meshes[0]


def process_park_layer(
    *,
    task: Any,
    request: Any,
    scale_factor: Optional[float],
    terrain_provider: Any,
    terrain_mesh: Any,
    global_center: Any,
    zone_polygon_local: Any,
    road_cut_source: Any,
    building_union_local: Any,
    water_polygons: Any,
    road_exclusion_clearance_mm: float = 0.0,
    fit_clearance_mm: float = 0.0,
    gdf_green: Optional[GeoDataFrame],
    zone_prefix: str = "",
    park_polygons_override: Any = None,
) -> ParkLayerResult:
    if not (scale_factor and scale_factor > 0 and (terrain_provider is not None or request.is_ams_mode)):
        return ParkLayerResult(mesh=None, parks_result=None)

    if not getattr(request, "include_parks", False):
        print("[INFO] Parks layer skipped (include_parks=False)")
        return ParkLayerResult(mesh=None, parks_result=None)

    if park_polygons_override is None and (gdf_green is None or gdf_green.empty):
        print("[INFO] Parks layer skipped (gdf_green is empty)")
        return ParkLayerResult(mesh=None, parks_result=None)

    try:
        park_height_m = (float(request.parks_height_mm) / float(scale_factor)) / 4.0
        park_embed_m = float(request.parks_embed_mm) / float(scale_factor)
        park_min_feature_mm = max(
            float(getattr(request, "tiny_feature_threshold_mm", 0.2)),
            float(MIN_LAND_WIDTH_MODEL_MM),
        )
        if park_polygons_override is not None and not getattr(park_polygons_override, "is_empty", True):
            # Canonical override polygons are already fit-processed in the
            # canonical 2D stage. Applying inset again here causes drift between
            # canonical masks and detail-layer inputs (double clearance).
            try:
                park_polygons_override = park_polygons_override.buffer(0)
            except Exception:
                pass
            parks_mesh = _rebuild_park_mesh_from_polygons(
                park_polygons=park_polygons_override,
                terrain_provider=terrain_provider,
                scale_factor=float(scale_factor),
                park_height_m=float(park_height_m),
                park_embed_m=float(park_embed_m),
                global_center=global_center,
            )
            parks_mesh = clamp_mesh_to_terrain_floor(parks_mesh, terrain_mesh, label="PARK")
            return ParkLayerResult(
                mesh=parks_mesh,
                parks_result=GreenAreaProcessingResult(
                    mesh=parks_mesh,
                    processed_polygons=park_polygons_override,
                ),
            )

        road_exclusion_polygons: Optional[BaseGeometry] = road_cut_source
        if (
            road_exclusion_polygons is not None
            and scale_factor
            and scale_factor > 0
            and float(road_exclusion_clearance_mm) > 0.0
        ):
            try:
                road_gap_m = (float(road_exclusion_clearance_mm) / 1000.0) / float(scale_factor)
                road_exclusion_polygons = road_exclusion_polygons.buffer(road_gap_m, join_style=1)
                print(
                    f"[INFO] {zone_prefix} Parks road exclusion buffered by "
                    f"{road_gap_m:.4f}m ({road_exclusion_clearance_mm}mm)"
                )
            except Exception as exc:
                print(f"[WARN] {zone_prefix} Failed to buffer road exclusion for parks: {exc}")

        prepared_green = prepare_green_areas_for_processing(
            gdf_green,
            global_center=global_center,
            zone_polygon_local=zone_polygon_local,
        )
        parks_result = process_green_areas(
            prepared_green,
            # Parks were already being lowered to half of request height here.
            # The user asked for parks to sit another 2x lower than the current state,
            # so the effective relief height becomes one quarter of parks_height_mm.
            height_m=float(park_height_m),
            embed_m=float(park_embed_m),
            terrain_provider=terrain_provider,
            global_center=global_center,
            scale_factor=float(scale_factor),
            zone_polygon_local=zone_polygon_local,
            min_feature_mm=float(park_min_feature_mm),
            fit_clearance_mm=float(fit_clearance_mm),
            road_polygons=road_exclusion_polygons,
            water_polygons=water_polygons,
            building_polygons=building_union_local,
            return_result=True,
        )
        parks_mesh = parks_result.mesh if parks_result is not None else None
        if parks_result is not None and getattr(parks_result, "processed_polygons", None) is not None:
            rebuilt_parks_mesh = _rebuild_park_mesh_from_polygons(
                park_polygons=parks_result.processed_polygons,
                terrain_provider=terrain_provider,
                scale_factor=float(scale_factor),
                park_height_m=float(park_height_m),
                park_embed_m=float(park_embed_m),
                global_center=global_center,
            )
            if rebuilt_parks_mesh is not None and len(rebuilt_parks_mesh.vertices) > 0:
                parks_mesh = rebuilt_parks_mesh
        if parks_mesh is None:
            print(f"[WARN] process_green_areas returned None for {len(prepared_green)} parks")

        parks_mesh = clamp_mesh_to_terrain_floor(parks_mesh, terrain_mesh, label="PARK")

        if request.is_ams_mode and parks_mesh is not None:
            try:
                lift_m = (1.4 / scale_factor) if scale_factor else 0.0
                if lift_m > 0:
                    parks_mesh.apply_translation([0, 0, lift_m])
                    print(f"[INFO] {zone_prefix} AMS Mode: Parks lifted by {lift_m:.4f}m (Target: 1.4mm level)")
            except Exception as exc:
                print(f"[WARN] AMS Parks lifting failed: {exc}")

        return ParkLayerResult(mesh=parks_mesh, parks_result=parks_result)
    except Exception as exc:
        print(f"[WARN] extras layers failed: {exc}")
        import traceback

        traceback.print_exc()
        return ParkLayerResult(mesh=None, parks_result=None)
