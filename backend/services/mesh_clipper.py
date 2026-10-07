"""
Utilities for clipping meshes using geometric shapes.
"""
import numpy as np
import trimesh
from typing import Optional, List, Tuple
from shapely.geometry import Polygon

def clip_mesh_to_bbox(
    mesh: trimesh.Trimesh,
    bbox: Tuple[float, float, float, float],
    tolerance: float = 0.001
) -> Optional[trimesh.Trimesh]:
    """Clips mesh to bbox (minx, miny, maxx, maxy) using plane slicing for exact edges."""
    if mesh is None or len(mesh.vertices) == 0: return mesh
    minx, miny, maxx, maxy = bbox
    
    # Define 4 planes (Normals pointing INWARD)
    planes = [
        # Left wall (x=minx), normal +X
        ([1, 0, 0], [minx, 0, 0]),
        # Right wall (x=maxx), normal -X
        ([-1, 0, 0], [maxx, 0, 0]),
        # Bottom wall (y=miny), normal +Y
        ([0, 1, 0], [0, miny, 0]),
        # Top wall (y=maxy), normal -Y
        ([0, -1, 0], [0, maxy, 0])
    ]
    
    current = mesh
    try:
        for normal, origin in planes:
            # slice_mesh_plane keeps the "positive" side of the normal
            current = trimesh.intersections.slice_mesh_plane(
                mesh=current, plane_normal=normal, plane_origin=origin, cap=False
            )
            if current is None or len(current.vertices) == 0: return None
        return current
    except Exception as e:
        print(f"[WARN] Bbox slice failed: {e}")
        return mesh # Fallback to original

def clip_mesh_to_polygon(
    mesh: trimesh.Trimesh,
    polygon_coords: List[Tuple[float, float]],
    global_center=None,
    tolerance: float = 0.001
) -> Optional[trimesh.Trimesh]:
    """
    ROBUST clipper:
    Keeps faces whose centroids are inside polygon.
    Edge precision limited by mesh resolution (lossy).
    Recommended for visual clipping, flattening logic, etc.
    NOT recommended for exact CAD operations unless mesh is dense.
    """
    if mesh is None or not len(mesh.vertices): return mesh
    
    # Handle shapely Polygon input
    if hasattr(polygon_coords, "exterior"):
        # It's a shapely Polygon
        polygon_coords = list(polygon_coords.exterior.coords)
    elif hasattr(polygon_coords, "coords"):
        # It's likely a LinearRing or similar
        polygon_coords = list(polygon_coords.coords)

    if not polygon_coords or len(polygon_coords) < 3: return mesh

    # 1. Prepare local polygon coords
    local_coords = []
    if global_center:
        lons = [c[0] for c in polygon_coords]
        lats = [c[1] for c in polygon_coords]
        for lon, lat in zip(lons, lats):
             try:
                 gx, gy = global_center.to_utm(lon, lat)
                 lx, ly = global_center.to_local(gx, gy)
                 local_coords.append((lx, ly))
             except: pass
    else:
        local_coords = polygon_coords

    if len(local_coords) < 3: return mesh
    
    # Ensure CCW orientation (Shapely default)
    poly = Polygon(local_coords)
    if not poly.is_valid: poly = poly.buffer(0)
    if poly.is_empty: return mesh
    
    # Extract shell coordinates
    if hasattr(poly, 'exterior'): ring = list(poly.exterior.coords)
    else: ring = list(poly.coords)
    
    # Remove collinear/close points to avoid degenerate slices
    clean_ring = []
    for p in ring:
         if not clean_ring or np.hypot(p[0]-clean_ring[-1][0], p[1]-clean_ring[-1][1]) > 0.01:
              clean_ring.append(p)
    if len(clean_ring) > 2 and clean_ring[0] == clean_ring[-1]:
         clean_ring.pop() # Remove duplicate end
         
    if len(clean_ring) < 3: return mesh

    # Check orientation (we need CCW for Left-Inside rule)
    # Signed area method
    area = 0.0
    for i in range(len(clean_ring)):
         j = (i + 1) % len(clean_ring)
         area += clean_ring[i][0] * clean_ring[j][1]
         area -= clean_ring[j][0] * clean_ring[i][1]
    
    is_ccw = area > 0
    if not is_ccw:
         clean_ring.reverse()

    # 2. Slice against every edge
    # 2. Robust 2D Clipping (Vertex Inclusion)
    # Iterative plane slicing is brittle for complex polygons (floating point errors, open meshes).
    # Instead, we mark vertices inside the polygon and keep faces where ALL vertices are inside.
    # This prevents "slicing through" triangles (leaving jagged edges), BUT since we have high-res terrain,
    # the jagged edge matches the grid resolution, which is usually acceptable and much more robust.
    # For "perfect" edges, we would need constrained Delaunay (which terrain_generator does in stitching mode).
    # This clipper is a fallback/utility.
    
    try:
        # Check vertex inclusion
        # Use ray tracing or matplotlib path or shapely (slow but reliable)
        # For speed with mpltPath:
        from matplotlib.path import Path
        path = Path(clean_ring)
        
        # Project mesh vertices to 2D
        # mesh.vertices is (N, 3)
        # pts = mesh.vertices[:, :2] # Unused, using centroids
        
        # Check inclusion
        # radius=0 mostly, or small tolerance
        # mask = path.contains_points(pts, radius=tolerance) # (unused, we use face centroids)
        
        # Filter faces: Keep face if ALL vertices are inside? 
        # Or if ANY? "All" creates a shrunk mesh (gap at border). 
        # "Any" creates an expanded mesh (sticks out).
        # Usually "All" is safer for "clipping to inside". 
        # But we want to fill the polygon.
        # Let's use "Centroid" check for faces?
        
        # Better: Centroid check.
        # Calculate face centroids
        if len(mesh.faces) > 0:
            face_centroids = mesh.vertices[mesh.faces].mean(axis=1)[:, :2]
            face_mask = path.contains_points(face_centroids, radius=tolerance)
            
            if np.any(face_mask):
                mesh.update_faces(face_mask)
                mesh.remove_unreferenced_vertices()
                return mesh
            else:
                return None # No faces inside
        else:
            return mesh

    except ImportError:
        # Fallback to shapely (slower)
        try:
             from shapely.geometry import Point
             from shapely.prepared import prep
             
             prep_poly = prep(poly)
             
             # Face centroids
             if len(mesh.faces) > 0:
                 face_centroids = mesh.vertices[mesh.faces].mean(axis=1)[:, :2]
                 face_mask = np.array([prep_poly.contains(Point(x,y)) for x,y in face_centroids], dtype=bool)
                 
                 if np.any(face_mask):
                     mesh.update_faces(face_mask)
                     mesh.remove_unreferenced_vertices()
                     return mesh
                 else:
                     return None
             return mesh
        except Exception as e:
             print(f"[WARN] Shapely clip fallback failed: {e}")
             return mesh

    except Exception as e:
        print(f"[WARN] Robust clip failed: {e}")
        return mesh


def _is_convex_polygon(poly) -> bool:
    """True лише для простого опуклого полігона без дірок (прямокутник, коло, шестикутник)."""
    if poly is None or poly.is_empty or poly.geom_type != "Polygon" or len(poly.interiors) > 0:
        return False
    hull_area = float(poly.convex_hull.area)
    if hull_area <= 0.0:
        return False
    return (hull_area - float(poly.area)) <= 1e-6 * hull_area


def clip_mesh_to_polygon_exact(mesh: trimesh.Trimesh, polygon) -> Optional[trimesh.Trimesh]:
    """Точне обрізання heightfield-поверхні по БУДЬ-ЯКОМУ полігону (неопуклий, з дірками).

    Грані далеко від межі: всередині — лишаються як є, зовні — відкидаються. Грані біля
    межі перетинаються з полігоном у 2D (shapely), шматок тріангулюється (earcut), Z
    береться з площини вихідної грані — як і при різі площинами, нові вершини лежать
    точно на межі полігона. Потім спільні вершини сусідніх шматків зливаються.
    """
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return mesh
    import shapely
    import mapbox_earcut as earcut

    poly = polygon
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    tri = V[F]
    xy = tri[:, :, :2]
    max_edge = np.max(np.linalg.norm(xy - np.roll(xy, 1, axis=1), axis=2), axis=1)
    cent = xy.mean(axis=1)

    boundary = poly.boundary
    shapely.prepare(boundary)
    near = shapely.dwithin(boundary, shapely.points(cent), max_edge + 1e-6)
    shapely.prepare(poly)
    inside_v = shapely.contains_xy(poly, V[:, 0], V[:, 1])
    keep_whole = inside_v[F].all(axis=1) & ~near

    # 2D-орієнтація вихідних граней (щоб шматки мали той самий winding)
    def _signed2(p):
        return ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))

    orient = _signed2(xy)
    new_verts, new_faces = [], []
    n_next = len(V)
    cand = np.nonzero(near & (np.abs(orient) > 1e-12))[0]
    if len(cand):
        rings = np.concatenate([xy[cand], xy[cand][:, :1]], axis=1)
        pieces = shapely.intersection(shapely.polygons(rings), poly)
        for k, piece in zip(cand, pieces):
            if piece is None or piece.is_empty:
                continue
            (xa, ya, za), (xb, yb, zb), (xc, yc, zc) = tri[k]
            d = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
            parts = piece.geoms if hasattr(piece, "geoms") else [piece]
            for part in parts:
                if part.geom_type != "Polygon" or part.area < 1e-10:
                    continue
                rings_xy = [np.asarray(part.exterior.coords)[:-1]] + \
                           [np.asarray(r.coords)[:-1] for r in part.interiors]
                pts = np.vstack(rings_xy)
                ends = np.cumsum([len(r) for r in rings_xy]).astype(np.uint32)
                idx = np.asarray(earcut.triangulate_float64(pts, ends), dtype=np.int64).reshape(-1, 3)
                if len(idx) == 0:
                    continue
                l1 = ((yb - yc) * (pts[:, 0] - xc) + (xc - xb) * (pts[:, 1] - yc)) / d
                l2 = ((yc - ya) * (pts[:, 0] - xc) + (xa - xc) * (pts[:, 1] - yc)) / d
                z = l1 * za + l2 * zb + (1.0 - l1 - l2) * zc
                flip = (_signed2(pts[idx]) * orient[k]) < 0
                idx[flip] = idx[flip][:, ::-1]
                new_verts.append(np.column_stack([pts, z]))
                new_faces.append(idx + n_next)
                n_next += len(pts)

    faces_out = [F[keep_whole]] + new_faces
    verts_out = np.vstack([V] + new_verts) if new_verts else V
    faces_out = np.vstack(faces_out) if faces_out else np.zeros((0, 3), dtype=np.int64)
    if len(faces_out) == 0:
        return None

    # Злиття спільних вершин: ключ — округлення до 1 мкм, координати лишаються ОРИГІНАЛЬНІ
    # (першої появи; вихідні вершини сітки стоять першими).
    key = np.round(verts_out / 1e-6).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    faces_out = first[inverse[faces_out]]
    out = trimesh.Trimesh(vertices=verts_out, faces=faces_out, process=False)
    out.update_faces(out.nondegenerate_faces())
    out.update_faces(out.unique_faces())
    out.remove_unreferenced_vertices()
    print(f"[CLIP] non-convex exact clip: kept {int(keep_whole.sum())} faces whole, "
          f"{len(cand)} boundary faces re-triangulated -> {len(out.faces)} faces "
          f"(area {out.area:.0f} vs polygon {poly.area:.0f} m2)")
    return out


def clip_mesh_to_polygon_planes(
    mesh: trimesh.Trimesh,
    polygon_coords: List[Tuple[float, float]],
    global_center=None,
) -> Optional[trimesh.Trimesh]:
    """
    EXACT-ish clipper for terrain: clips mesh by slicing with vertical planes along polygon edges.
    This creates new vertices along the cut and removes the "sawtooth" boundary you get from
    centroid/vertex-inclusion clipping.

    Notes:
    - Keeps the inside of the polygon (assumes CCW ring).
    - cap=False because we solidify later (walls+base).
    """
    if mesh is None or len(mesh.vertices) == 0:
        return mesh

    # Handle shapely Polygon input
    if hasattr(polygon_coords, "exterior"):
        polygon_coords = list(polygon_coords.exterior.coords)
    elif hasattr(polygon_coords, "coords"):
        polygon_coords = list(polygon_coords.coords)

    if not polygon_coords or len(polygon_coords) < 3:
        return mesh

    # Prepare local coords
    if global_center:
        local_coords = []
        lons = [c[0] for c in polygon_coords]
        lats = [c[1] for c in polygon_coords]
        for lon, lat in zip(lons, lats):
            try:
                gx, gy = global_center.to_utm(lon, lat)
                lx, ly = global_center.to_local(gx, gy)
                local_coords.append((lx, ly))
            except Exception:
                pass
    else:
        local_coords = polygon_coords

    if len(local_coords) < 3:
        return mesh

    # Ensure valid CCW ring via shapely
    poly = Polygon(local_coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return mesh

    # 07.10.2026: різ площинами вздовж КОЖНОГО ребра = перетин півплощин = опукле ядро
    # полігона. Для прямокутника/кола/шестикутника це вся форма, а для серця/пазла/
    # половинки серця лишалось ~1.5 % рельєфу (будинки й дороги висіли за основою).
    # Неопуклі форми ріжемо точно, потрикутно; опуклі — як раніше (байт-ідентично).
    if not _is_convex_polygon(poly):
        return clip_mesh_to_polygon_exact(mesh, poly)

    ring = list(poly.exterior.coords)
    if len(ring) > 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) < 3:
        return mesh

    # Ensure CCW orientation (signed area)
    area = 0.0
    for i in range(len(ring)):
        j = (i + 1) % len(ring)
        area += ring[i][0] * ring[j][1] - ring[j][0] * ring[i][1]
    if area < 0:
        ring = list(reversed(ring))

    current = mesh
    try:
        for i in range(len(ring)):
            x1, y1 = ring[i]
            x2, y2 = ring[(i + 1) % len(ring)]
            ex = x2 - x1
            ey = y2 - y1
            n = np.array([-ey, ex, 0.0], dtype=np.float64)  # left normal => inside for CCW
            ln = float(np.linalg.norm(n[:2]))
            if ln < 1e-9:
                continue
            n /= ln
            current = trimesh.intersections.slice_mesh_plane(
                mesh=current,
                plane_normal=n,
                plane_origin=[x1, y1, 0.0],
                cap=False,
            )
            if current is None or len(current.vertices) == 0:
                return None
        # light cleanup — merge_vertices is critical: slice_mesh_plane leaves duplicate
        # vertices at each cut plane, causing 1000+ disconnected components in the terrain.
        current.merge_vertices()
        current.update_faces(current.unique_faces())
        current.update_faces(current.nondegenerate_faces())
        current.remove_unreferenced_vertices()
        return current
    except Exception as e:
        print(f"[WARN] Polygon plane-slice clip failed: {e}")
        return mesh

def clip_all_meshes_to_bbox(mesh_items, bbox, tolerance=0.001):
    out = []
    for name, m in mesh_items:
         if m:
              c = clip_mesh_to_bbox(m, bbox, tolerance)
              if c and len(c.vertices): out.append((name, c))
    return out
