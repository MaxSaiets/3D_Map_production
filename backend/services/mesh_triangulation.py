"""
Utilities for polygon extrusion with more uniform triangulation.
"""

from typing import Optional

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon


def densify_polygon_boundary(poly: Polygon, max_segment_m: float = 2.0) -> Polygon:
    if poly is None or poly.is_empty:
        return poly
    try:
        def _densify_ring(coords):
            pts = list(coords)
            if len(pts) < 3:
                return pts
            out = []
            for i in range(len(pts) - 1):
                p1, p2 = np.array(pts[i]), np.array(pts[i + 1])
                out.append(tuple(p1))
                dist = np.linalg.norm(p2 - p1)
                if dist > max_segment_m:
                    n = int(np.ceil(dist / max_segment_m))
                    for j in range(1, n):
                        t = j / n
                        out.append(tuple(p1 + (p2 - p1) * t))
            out.append(tuple(pts[-1]))
            return out

        exterior = list(poly.exterior.coords)
        new_exterior = _densify_ring(exterior)

        interiors = []
        for interior in poly.interiors:
            interior_pts = list(interior.coords)
            new_interior = _densify_ring(interior_pts)
            if len(new_interior) >= 3:
                interiors.append(new_interior)

        if len(new_exterior) < 3:
            return poly
        new_poly = Polygon(new_exterior, interiors if interiors else None)
        if new_poly.is_empty or not new_poly.is_valid:
            return poly
        return new_poly
    except Exception:
        return poly


def extrude_polygon_uniform(
    polygon: Polygon,
    height: float,
    densify_max_m: float = 1.0,
) -> Optional[trimesh.Trimesh]:
    if polygon is None or polygon.is_empty:
        return None
    try:
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            return None
        # Remove near-duplicate consecutive vertices (differ by < 1e-6 m).
        # These cause non-manifold edges in trimesh's extrude_polygon triangulator.
        try:
            cleaned = polygon.simplify(1e-6, preserve_topology=True)
            if cleaned is not None and not cleaned.is_empty and cleaned.is_valid:
                polygon = cleaned
        except Exception:
            pass

        # Polygons with holes: densifying boundary causes trimesh to produce
        # non-manifold edges (edge shared by 4 faces instead of 2) → not watertight.
        # Skip densification for holed polygons; extrude_polygon handles them correctly.
        has_holes = len(list(polygon.interiors)) > 0
        if has_holes:
            for engine in ("manifold", "triangle", None):
                try:
                    kwargs = {"engine": engine} if engine is not None else {}
                    mesh = trimesh.creation.extrude_polygon(
                        polygon,
                        height=float(height),
                        **kwargs,
                    )
                    if mesh is not None and len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                        return mesh
                except Exception:
                    continue
            return None
        densified = densify_polygon_boundary(polygon, max_segment_m=densify_max_m)
        return trimesh.creation.extrude_polygon(densified, height=float(height))
    except Exception:
        try:
            return trimesh.creation.extrude_polygon(polygon, height=float(height))
        except Exception:
            return None


def extrude_polygon_grid(
    polygon: Polygon,
    height: float,
    target_edge_len_m: float = 6.0,
    max_grid_points: int = 12000,
) -> Optional[trimesh.Trimesh]:
    if polygon is None or polygon.is_empty:
        return None
    try:
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            return None

        step_m = max(float(target_edge_len_m or 0.0), 1.0)
        minx, miny, maxx, maxy = polygon.bounds
        span_x = max(float(maxx - minx), 0.0)
        span_y = max(float(maxy - miny), 0.0)
        estimated_points = ((span_x / step_m) + 1.0) * ((span_y / step_m) + 1.0)
        if max_grid_points > 0 and estimated_points > max_grid_points:
            step_m *= float(np.sqrt(estimated_points / float(max_grid_points)))

        def _sample_ring(coords):
            pts = list(coords)
            if len(pts) < 3:
                return []
            out = []
            for i in range(len(pts) - 1):
                p1 = np.array(pts[i], dtype=float)
                p2 = np.array(pts[i + 1], dtype=float)
                out.append(p1)
                dist = float(np.linalg.norm(p2 - p1))
                if dist > step_m:
                    segs = int(np.ceil(dist / step_m))
                    for j in range(1, segs):
                        t = j / segs
                        out.append(p1 + (p2 - p1) * t)
            return out

        boundary_points = _sample_ring(polygon.exterior.coords)
        for interior in polygon.interiors:
            boundary_points.extend(_sample_ring(interior.coords))
        if not boundary_points:
            return extrude_polygon_uniform(polygon, height=float(height), densify_max_m=min(step_m, 4.0))

        x_range = np.arange(minx, maxx + step_m, step_m)
        y_range = np.arange(miny, maxy + step_m, step_m)
        if len(x_range) == 0:
            x_range = np.array([minx, maxx], dtype=float)
        if len(y_range) == 0:
            y_range = np.array([miny, maxy], dtype=float)
        xx, yy = np.meshgrid(x_range, y_range)
        grid_points = np.column_stack((xx.ravel(), yy.ravel()))

        try:
            from shapely.prepared import prep

            prepared = prep(polygon)
            inside_points = [
                pt for pt in grid_points if prepared.contains(Point(float(pt[0]), float(pt[1])))
            ]
        except Exception:
            inside_points = [
                pt for pt in grid_points if polygon.contains(Point(float(pt[0]), float(pt[1])))
            ]

        all_points = np.array(boundary_points + inside_points, dtype=float)
        if len(all_points) < 3:
            return extrude_polygon_uniform(polygon, height=float(height), densify_max_m=min(step_m, 4.0))

        tolerance = max(step_m * 0.15, 1e-6)
        unique_points = []
        seen = set()
        for pt in all_points:
            key = (round(float(pt[0]) / tolerance), round(float(pt[1]) / tolerance))
            if key in seen:
                continue
            seen.add(key)
            unique_points.append(pt)
        if len(unique_points) < 3:
            return extrude_polygon_uniform(polygon, height=float(height), densify_max_m=min(step_m, 4.0))

        vertices_2d = np.array(unique_points, dtype=float)
        try:
            from scipy.spatial import Delaunay

            tri = Delaunay(vertices_2d)
            final_faces = []
            for face in tri.simplices:
                tri_poly = Polygon(vertices_2d[face])
                if tri_poly.is_empty or tri_poly.area <= 1e-9:
                    continue
                try:
                    outside_area = float(tri_poly.difference(polygon).area)
                    inside_area = float(tri_poly.intersection(polygon).area)
                except Exception:
                    continue
                tol_area = max(float(tri_poly.area) * 1e-6, 1e-8)
                if inside_area > 0.0 and outside_area <= tol_area:
                    final_faces.append(face)
            if not final_faces:
                return extrude_polygon_uniform(polygon, height=float(height), densify_max_m=min(step_m, 4.0))
            faces_2d = np.array(final_faces, dtype=int)
        except Exception:
            return extrude_polygon_uniform(polygon, height=float(height), densify_max_m=min(step_m, 4.0))

        n_verts = len(vertices_2d)
        v_bottom = np.column_stack((vertices_2d, np.zeros(n_verts)))
        v_top = np.column_stack((vertices_2d, np.full(n_verts, float(height))))
        vertices_3d = np.vstack((v_bottom, v_top))

        f_bottom = np.fliplr(faces_2d)
        f_top = faces_2d + n_verts

        edge_count = {}
        for face in faces_2d:
            for i in range(3):
                edge = tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        side_faces = []
        for face in faces_2d:
            for i in range(3):
                v1 = int(face[i])
                v2 = int(face[(i + 1) % 3])
                edge = tuple(sorted((v1, v2)))
                if edge_count.get(edge, 0) != 1:
                    continue
                side_faces.append([v1, v2, v1 + n_verts])
                side_faces.append([v2, v2 + n_verts, v1 + n_verts])

        all_faces = np.vstack(
            [
                f_bottom,
                f_top,
                np.array(side_faces, dtype=int) if side_faces else np.empty((0, 3), dtype=int),
            ]
        )
        mesh = trimesh.Trimesh(vertices=vertices_3d, faces=all_faces, process=False)
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        return mesh
    except Exception:
        fallback_step = min(max(float(target_edge_len_m or 1.0), 1.0), 4.0)
        return extrude_polygon_uniform(polygon, height=float(height), densify_max_m=fallback_step)


def refine_mesh_long_edges(
    mesh: Optional[trimesh.Trimesh],
    max_edge_m: float,
    max_vertices: int = 40000,
) -> Optional[trimesh.Trimesh]:
    if mesh is None or len(mesh.vertices) == 0:
        return mesh
    try:
        max_edge_m = max(float(max_edge_m or 0.0), 0.5)
        vertices, faces = trimesh.remesh.subdivide_to_size(
            vertices=mesh.vertices,
            faces=mesh.faces,
            max_edge=max_edge_m,
            max_iter=2,
        )
        refined = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if len(refined.vertices) > max_vertices:
            return mesh
        refined.remove_unreferenced_vertices()
        refined.fix_normals()
        return refined
    except Exception:
        return mesh
