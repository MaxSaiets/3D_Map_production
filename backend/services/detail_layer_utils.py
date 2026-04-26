from __future__ import annotations

from typing import Any, Optional

import trimesh
import numpy as np
from geopandas import GeoDataFrame
from shapely.ops import transform as transform_geometry

MICRO_REGION_THRESHOLD_MM = 0.7
MIN_ROAD_WIDTH_MODEL_MM = 0.55
MIN_LAND_WIDTH_MODEL_MM = 0.6


def model_mm_to_world_m(model_mm: float, scale_factor: Optional[float]) -> float:
    if scale_factor is None or scale_factor <= 0:
        return 0.0
    try:
        return max(float(model_mm) / float(scale_factor), 0.0)
    except Exception:
        return 0.0


def clamp_mesh_to_terrain_floor(
    mesh: Optional[trimesh.Trimesh],
    terrain_mesh: Optional[trimesh.Trimesh],
    *,
    label: str,
) -> Optional[trimesh.Trimesh]:
    if mesh is None or terrain_mesh is None or len(mesh.vertices) == 0:
        return mesh

    terrain_floor_z = float(terrain_mesh.bounds[0][2])
    mesh_min_z = float(mesh.vertices[:, 2].min())
    if mesh_min_z >= terrain_floor_z:
        return mesh

    vertices = mesh.vertices.copy()
    below_mask = vertices[:, 2] < terrain_floor_z
    clamped_count = int(below_mask.sum())
    vertices[below_mask, 2] = terrain_floor_z
    mesh.vertices = vertices
    print(
        f"[{label}] Clamped {clamped_count} vertices below terrain floor: "
        f"{mesh_min_z:.4f} -> {terrain_floor_z:.4f}"
    )
    return mesh


def prepare_green_areas_for_processing(
    gdf_green: Optional[GeoDataFrame],
    *,
    global_center: Any,
    zone_polygon_local: Any,
) -> Optional[GeoDataFrame]:
    if gdf_green is None or gdf_green.empty:
        return gdf_green

    prepared_green = gdf_green
    try:
        def to_local_transform(x, y, z=None):
            x_local, y_local = global_center.to_local(x, y)
            if z is not None:
                return (x_local, y_local, z)
            return (x_local, y_local)

        prepared_green = prepared_green.copy()
        prepared_green["geometry"] = prepared_green["geometry"].apply(
            lambda geom: transform_geometry(to_local_transform, geom)
            if geom is not None and not geom.is_empty
            else geom
        )
    except Exception as exc:
        print(f"[WARN] Failed to transform gdf_green to local coordinates: {exc}")

    if zone_polygon_local is not None and not zone_polygon_local.is_empty:
        try:
            def clip_to_zone(geom):
                if geom is None or geom.is_empty:
                    return None
                try:
                    out = geom.intersection(zone_polygon_local)
                except Exception:
                    return geom
                if out is None or out.is_empty:
                    return None
                try:
                    if hasattr(out, "area") and float(out.area) < 10.0:
                        return None
                except Exception:
                    pass
                return out

            prepared_green = prepared_green.copy()
            prepared_green["geometry"] = prepared_green["geometry"].apply(clip_to_zone)
            prepared_green = prepared_green[prepared_green.geometry.notna()]
            prepared_green = prepared_green[~prepared_green.geometry.is_empty]
        except Exception:
            pass

    return prepared_green


def filter_mesh_components_by_printability(
    mesh: Optional[trimesh.Trimesh],
    *,
    scale_factor: Optional[float],
    min_feature_mm: float = MICRO_REGION_THRESHOLD_MM,
    label: str = "MESH",
) -> Optional[trimesh.Trimesh]:
    if mesh is None or len(mesh.vertices) == 0 or scale_factor is None or scale_factor <= 0:
        return mesh

    min_feature_m = model_mm_to_world_m(min_feature_mm, scale_factor)
    if min_feature_m <= 0:
        return mesh

    try:
        components = mesh.split(only_watertight=False)
    except Exception:
        return mesh

    components = list(components)
    if len(components) <= 1:
        return mesh

    kept = []
    removed = 0
    min_area_m2 = max(min_feature_m * min_feature_m, 1e-8)

    for component in components:
        if component is None or len(component.vertices) == 0:
            removed += 1
            continue
        try:
            mins = component.vertices[:, :2].min(axis=0)
            maxs = component.vertices[:, :2].max(axis=0)
            extents_xy = np.asarray(maxs - mins, dtype=float)
            min_xy = float(np.min(extents_xy))
            max_xy = float(np.max(extents_xy))
            footprint_area = float(extents_xy[0] * extents_xy[1])
        except Exception:
            kept.append(component)
            continue

        # Remove isolated needle/sliver components that are too thin to print.
        if min_xy < min_feature_m and footprint_area < max(min_area_m2 * 2.0, min_feature_m * max_xy):
            removed += 1
            continue
        kept.append(component)

    if removed <= 0:
        return mesh

    if not kept:
        print(f"[{label}] Removed all {removed} thin mesh components below {min_feature_mm}mm")
        return None

    print(f"[{label}] Removed {removed} thin mesh components below {min_feature_mm}mm")
    try:
        return trimesh.util.concatenate(kept)
    except Exception:
        return kept[0] if kept else None
