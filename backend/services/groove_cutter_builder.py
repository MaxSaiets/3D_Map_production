from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


@dataclass
class UnifiedGrooveCutterBuildResult:
    cutter_mesh: Optional[trimesh.Trimesh]
    cut_bottom_z: Optional[float]
    cut_top_z: Optional[float]


def build_unified_groove_cutter(
    *,
    terrain_mesh: trimesh.Trimesh,
    road_polygons: Optional[BaseGeometry] = None,
    road_clearance_m: float = 0.0,
    parks_polygons: Optional[BaseGeometry] = None,
    parks_clearance_m: float = 0.0,
    parks_mesh: Optional[trimesh.Trimesh] = None,
    water_polygons: Optional[BaseGeometry] = None,
    water_clearance_m: float = 0.0,
    water_mesh: Optional[trimesh.Trimesh] = None,
    road_mesh: Optional[trimesh.Trimesh] = None,
    groove_depth_m: Optional[float] = None,
) -> UnifiedGrooveCutterBuildResult:
    if terrain_mesh is None:
        return UnifiedGrooveCutterBuildResult(cutter_mesh=None, cut_bottom_z=None, cut_top_z=None)

    has_roads = road_polygons is not None and not road_polygons.is_empty
    has_parks = parks_polygons is not None and not parks_polygons.is_empty
    has_water = water_polygons is not None and not water_polygons.is_empty
    if not has_roads and not has_parks and not has_water:
        print("[TERRAIN CUT] No road, park or water polygons provided, skipping cutter build")
        return UnifiedGrooveCutterBuildResult(cutter_mesh=None, cut_bottom_z=None, cut_top_z=None)

    bounds = terrain_mesh.bounds
    terrain_max_z = bounds[1][2]
    terrain_floor_z = bounds[0][2]
    terrain_height = terrain_max_z - terrain_floor_z

    mesh_min_z = float("inf")
    mesh_bottoms: dict[str, float] = {}
    if road_mesh is not None:
        road_bot_z = float(road_mesh.bounds[0][2])
        mesh_min_z = min(mesh_min_z, road_bot_z)
        mesh_bottoms["road"] = road_bot_z
        print(f"[TERRAIN CUT] Road mesh bottom Z: {road_bot_z:.4f}")
    if parks_mesh is not None:
        parks_bot_z = float(parks_mesh.bounds[0][2])
        mesh_min_z = min(mesh_min_z, parks_bot_z)
        mesh_bottoms["parks"] = parks_bot_z
        print(f"[TERRAIN CUT] Parks mesh bottom Z: {parks_bot_z:.4f}")
    if water_mesh is not None:
        water_bot_z = float(water_mesh.bounds[0][2])
        mesh_min_z = min(mesh_min_z, water_bot_z)
        mesh_bottoms["water"] = water_bot_z
        print(f"[TERRAIN CUT] Water mesh bottom Z: {water_bot_z:.4f}")

    if mesh_min_z < float("inf"):
        for label, mesh_obj in [("road", road_mesh), ("parks", parks_mesh), ("water", water_mesh)]:
            if mesh_obj is None:
                continue
            bot_z = mesh_bottoms.get(label)
            if bot_z is None:
                continue

            verts = mesh_obj.vertices.copy()
            face_normals = mesh_obj.face_normals
            faces = mesh_obj.faces
            bottom_faces = face_normals[:, 2] < -0.5
            bottom_vert_ids = np.unique(faces[bottom_faces].ravel())
            count = len(bottom_vert_ids)

            if count > 0:
                old_min = float(verts[bottom_vert_ids, 2].min())
                old_max = float(verts[bottom_vert_ids, 2].max())
                verts[bottom_vert_ids, 2] = mesh_min_z
                mesh_obj.vertices = verts
                print(
                    f"[TERRAIN CUT] {label} bottom aligned: "
                    f"Z[{old_min:.4f}..{old_max:.4f}] -> {mesh_min_z:.4f} ({count} verts)"
                )
            else:
                print(f"[TERRAIN CUT] {label}: no bottom faces found")

    if mesh_min_z < float("inf"):
        cut_bottom_z = mesh_min_z - 0.1
    elif groove_depth_m is not None and groove_depth_m > 0:
        z_values = terrain_mesh.vertices[:, 2]
        surface_min_z = float(np.percentile(z_values, 25))
        cut_bottom_z = surface_min_z - groove_depth_m
    else:
        cut_bottom_z = terrain_floor_z + terrain_height * 0.3

    min_floor = terrain_floor_z + 0.5
    cut_bottom_z = max(cut_bottom_z, min_floor)
    cut_top_z = terrain_max_z + 5.0
    cutter_height = cut_top_z - cut_bottom_z

    print(f"[TERRAIN CUT] Cutter Z=[{cut_bottom_z:.4f}, {cut_top_z:.4f}], h={cutter_height:.4f}")

    all_2d_polygons = []

    def _collect_polygons(geom, clearance_m, label):
        if geom is None or geom.is_empty:
            return []

        expanded = geom
        if clearance_m > 0:
            try:
                expanded = geom.buffer(clearance_m, join_style=2)
            except Exception:
                pass

        try:
            expanded = expanded.buffer(0)
        except Exception:
            pass

        polygons = []
        raw_geoms = list(expanded.geoms) if hasattr(expanded, "geoms") else [expanded]
        for item in raw_geoms:
            if item is None or item.is_empty:
                continue
            if item.geom_type == "Polygon" and item.area > 0.001:
                polygons.append(item)
            elif item.geom_type == "MultiPolygon":
                for poly in item.geoms:
                    if not poly.is_empty and poly.area > 0.001:
                        polygons.append(poly)

        print(f"[TERRAIN CUT] {label}: collected {len(polygons)} polygons")
        return polygons

    if has_roads:
        all_2d_polygons.extend(_collect_polygons(road_polygons, road_clearance_m, "Roads"))
    if has_parks:
        all_2d_polygons.extend(_collect_polygons(parks_polygons, parks_clearance_m, "Parks"))
    if has_water:
        all_2d_polygons.extend(_collect_polygons(water_polygons, water_clearance_m, "Water"))

    if not all_2d_polygons:
        print("[TERRAIN CUT] WARNING: No valid polygons collected!")
        return UnifiedGrooveCutterBuildResult(cutter_mesh=None, cut_bottom_z=cut_bottom_z, cut_top_z=cut_top_z)

    try:
        combined_2d = unary_union(all_2d_polygons)
        combined_2d = combined_2d.buffer(0)
        print(f"[TERRAIN CUT] Combined 2D: type={combined_2d.geom_type}, area={combined_2d.area:.2f} m2")
    except Exception as exc:
        print(f"[TERRAIN CUT] WARNING: unary_union failed: {exc}, falling back to individual polygons")
        combined_2d = None

    cutter_parts = []
    if combined_2d is not None and not combined_2d.is_empty:
        if combined_2d.geom_type == "Polygon":
            final_polys = [combined_2d]
        elif combined_2d.geom_type == "MultiPolygon":
            final_polys = list(combined_2d.geoms)
        else:
            final_polys = [g for g in combined_2d.geoms if g.geom_type == "Polygon" and g.area > 0.001]

        print(f"[TERRAIN CUT] Extruding {len(final_polys)} unified polygons...")
        for i, poly in enumerate(final_polys):
            if poly.is_empty or poly.area < 0.001:
                continue
            try:
                exact_poly = poly.buffer(0)
                if exact_poly.is_empty or exact_poly.area < 0.001 or exact_poly.geom_type != "Polygon":
                    exact_poly = poly

                part = trimesh.creation.extrude_polygon(exact_poly, height=cutter_height)
                curr_min = part.bounds[0][2]
                part.apply_translation([0, 0, cut_bottom_z - curr_min])
                cutter_parts.append(part)
            except Exception as exc:
                print(f"[TERRAIN CUT] WARN: Failed to extrude polygon {i} (area={poly.area:.2f}): {exc}")
                try:
                    hull = poly.convex_hull
                    if hull.geom_type == "Polygon" and hull.area > 0.001:
                        part = trimesh.creation.extrude_polygon(hull, height=cutter_height)
                        curr_min = part.bounds[0][2]
                        part.apply_translation([0, 0, cut_bottom_z - curr_min])
                        cutter_parts.append(part)
                        print(f"[TERRAIN CUT] Used convex hull fallback for polygon {i}")
                except Exception:
                    pass
    else:
        print("[TERRAIN CUT] Falling back to individual polygon extrusion...")
        for poly in all_2d_polygons:
            try:
                exact_poly = poly.buffer(0)
                if exact_poly.is_empty or exact_poly.geom_type != "Polygon":
                    continue
                part = trimesh.creation.extrude_polygon(exact_poly, height=cutter_height)
                curr_min = part.bounds[0][2]
                part.apply_translation([0, 0, cut_bottom_z - curr_min])
                cutter_parts.append(part)
            except Exception:
                continue

    if not cutter_parts:
        print("[TERRAIN CUT] WARNING: No cutter parts created!")
        return UnifiedGrooveCutterBuildResult(cutter_mesh=None, cut_bottom_z=cut_bottom_z, cut_top_z=cut_top_z)

    cutter_mesh = trimesh.util.concatenate(cutter_parts)
    print(f"[TERRAIN CUT] Final cutter: {len(cutter_mesh.vertices)} verts, {len(cutter_mesh.faces)} faces")
    print(
        f"[TERRAIN CUT] Cutter bounds: X=[{cutter_mesh.bounds[0][0]:.2f}, {cutter_mesh.bounds[1][0]:.2f}], "
        f"Y=[{cutter_mesh.bounds[0][1]:.2f}, {cutter_mesh.bounds[1][1]:.2f}], "
        f"Z=[{cutter_mesh.bounds[0][2]:.4f}, {cutter_mesh.bounds[1][2]:.4f}]"
    )
    return UnifiedGrooveCutterBuildResult(
        cutter_mesh=cutter_mesh,
        cut_bottom_z=cut_bottom_z,
        cut_top_z=cut_top_z,
    )
