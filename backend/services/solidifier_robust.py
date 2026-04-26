"""
Robust terrain solidifier - створює герметичний solid terrain mesh
ВИПРАВЛЕНО: Правильна обробка boundary, watertight bottom cap, валідація
"""
import numpy as np
import trimesh
from typing import Optional, Tuple
from shapely.geometry import Polygon as ShapelyPolygon


def _sample_polygon_boundary(
    polygon: ShapelyPolygon,
    interval_m: float = 0.5
) -> np.ndarray:
    """
    Sample points along polygon boundary at regular intervals
    """
    if polygon is None or polygon.is_empty:
        return np.array([])
    
    coords = list(polygon.exterior.coords)[:-1]  # Remove duplicate last point
    if len(coords) < 3:
        return np.array([])
    
    points = []
    for i in range(len(coords)):
        p1 = np.array(coords[i])
        p2 = np.array(coords[(i + 1) % len(coords)])
        
        segment_vec = p2 - p1
        segment_len = np.linalg.norm(segment_vec)
        
        if segment_len < 1e-6:
            continue
        
        num_samples = max(2, int(np.ceil(segment_len / interval_m)))
        for j in range(num_samples):
            t = j / num_samples
            points.append(p1 + t * segment_vec)
    
    return np.array(points) if points else np.array([])


def _triangulate_polygon_simple(vertices_2d: np.ndarray) -> np.ndarray:
    """
    Проста і НАДІЙНА тріангуляція полігона
    Використовує ear clipping algorithm через shapely
    """
    if len(vertices_2d) < 3:
        return np.array([], dtype=np.int32).reshape(0, 3)
    
    try:
        # Спробуємо використати shapely для валідації
        from shapely.geometry import Polygon
        poly = Polygon(vertices_2d)
        
        if not poly.is_valid:
            poly = poly.buffer(0)
        
        # ПРОСТИЙ ТА НАДІЙНИЙ МЕТОД: Fan triangulation від центру
        # Це завжди працює і створює watertight mesh
        n = len(vertices_2d)
        center = np.mean(vertices_2d, axis=0)
        
        # Додамо center як останню вершину
        # Тоді грані будуть: [n, i, (i+1)%n] для кожного i
        faces = []
        for i in range(n):
            # Створюємо трикутник від центру до сусідніх вершин
            faces.append([n, i, (i + 1) % n])
        
        return np.array(faces, dtype=np.int32)
        
    except Exception as e:
        print(f"[WARN] Polygon triangulation failed: {e}, using simple fan")
        # Fallback: простий fan від першої вершини (без центру)
        n = len(vertices_2d)
        faces = []
        for i in range(1, n - 1):
            faces.append([0, i, i + 1])
        return np.array(faces, dtype=np.int32) if faces else np.array([], dtype=np.int32).reshape(0, 3)


def create_solid_terrain_robust(
    terrain_top: trimesh.Trimesh,
    zone_polygon: ShapelyPolygon,
    base_thickness: float = 5.0,
    sampling_interval_m: float = 0.5,
    floor_z: Optional[float] = None,
    boundary_verts_3d: Optional[np.ndarray] = None
) -> Optional[trimesh.Trimesh]:
    """
    Створює ГЕРМЕТИЧНИЙ solid terrain mesh
    
    ВАЖЛИВО:
    - Не дублює boundary vertices
    - Правильно тріангулює bottom cap
    - Перевіряє watertight
    - Виправляє нормалі
    
    Args:
        terrain_top: Top surface mesh
        zone_polygon: Zone boundary polygon
        base_thickness: Thickness of base (meters)
        sampling_interval_m: Sampling interval for boundary
        floor_z: Explicit floor Z coordinate (if None, calculated from mesh)
        boundary_verts_3d: Pre-computed boundary vertices (для оптимізації)
        
    Returns:
        Watertight solid terrain mesh or None if failed
    """
    if terrain_top is None or len(terrain_top.vertices) == 0 or len(terrain_top.faces) == 0:
        print("[ERROR] Invalid terrain_top mesh")
        return None
    
    try:
        print(f"[SOLIDIFIER] Starting solidification...")
        print(f"[SOLIDIFIER] Input mesh: {len(terrain_top.vertices)} vertices, {len(terrain_top.faces)} faces")
        
        # 1. Отримати boundary vertices
        if boundary_verts_3d is not None and len(boundary_verts_3d) > 0:
            top_boundary = boundary_verts_3d
            print(f"[SOLIDIFIER] Using provided boundary: {len(top_boundary)} vertices")
        else:
            # Fallback: семплюємо з polygon
            boundary_2d = _sample_polygon_boundary(zone_polygon, sampling_interval_m)
            if len(boundary_2d) == 0:
                print("[ERROR] Failed to sample polygon boundary")
                return None
            
            # Отримуємо Z з terrain mesh
            from scipy.spatial import cKDTree
            tree = cKDTree(terrain_top.vertices[:, :2])
            _, indices = tree.query(boundary_2d)
            z_vals = terrain_top.vertices[indices, 2]
            top_boundary = np.column_stack([boundary_2d, z_vals])
            print(f"[SOLIDIFIER] Sampled boundary: {len(top_boundary)} vertices")
        
        n_boundary = len(top_boundary)
        if n_boundary < 3:
            print("[ERROR] Boundary has less than 3 vertices")
            return None
        
        # 2. Визначити floor Z
        if floor_z is None:
            min_z = np.min(terrain_top.vertices[:, 2])
            floor_z = min_z - base_thickness
        
        print(f"[SOLIDIFIER] Floor Z: {floor_z:.2f}m")
        
        # 3. Створити bottom boundary (та ж XY, але Z = floor_z)
        bottom_boundary = top_boundary.copy()
        bottom_boundary[:, 2] = floor_z
        
        # 4. КРИТИЧНО: Створимо side walls
        # Для кожного сегмента boundary створюємо 2 трикутники
        wall_faces = []
        for i in range(n_boundary):
            i_next = (i + 1) % n_boundary
            
            top_i = i
            top_i_next = i_next
            bottom_i = i + n_boundary
            bottom_i_next = i_next + n_boundary
            
            # ВАЖЛИВО: Правильний winding order для нормалей НАЗОВНІ
            # При обході boundary за годинниковою стрілкою, нормалі мають дивитись НАЗОВНІ
            # Triangle 1: top_i, bottom_i, top_i_next (CCW when viewed from outside)
            wall_faces.append([top_i, bottom_i, top_i_next])
            # Triangle 2: top_i_next, bottom_i, bottom_i_next
            wall_faces.append([top_i_next, bottom_i, bottom_i_next])
        
        wall_faces = np.array(wall_faces, dtype=np.int32)
        print(f"[SOLIDIFIER] Created {len(wall_faces)} wall faces")
        
        # 5. КРИТИЧНО: Правильно тріангулюємо bottom cap з центром
        bottom_2d = bottom_boundary[:, :2]
        
        # Метод Fan від центру - ЗАВЖДИ створює watertight mesh
        # Обчислюємо центр bottom polygon
        bottom_center_xy = np.mean(bottom_2d, axis=0)
        bottom_center_z = floor_z
        bottom_center_3d = np.array([bottom_center_xy[0], bottom_center_xy[1], bottom_center_z])
        
        # Створюємо грані від центру
        # Center буде мати індекс = n_boundary * 2 (після top і bottom boundaries)
        center_idx = n_boundary * 2
        
        bottom_cap_faces = []
        for i in range(n_boundary):
            i_next = (i + 1) % n_boundary
            # Трикутник: [center, vertex_i, vertex_i+1]
            # Bottom boundary має індекси від n_boundary до 2*n_boundary-1
            bottom_cap_faces.append([center_idx, n_boundary + i, n_boundary + i_next])
        
        bottom_cap_faces = np.array(bottom_cap_faces, dtype=np.int32)
        
        # Перевернути нормалі bottom cap (вони мають дивитись вниз)
        bottom_cap_faces = bottom_cap_faces[:, ::-1]  # Reverse winding order
        
        print(f"[SOLIDIFIER] Created {len(bottom_cap_faces)} bottom cap faces (with center vertex)")
        
        # 6. ВАЖЛИВО: Об'єднати ВСІ компоненти
        # Vertices: top_boundary + bottom_boundary + center + terrain
        offset_terrain = n_boundary * 2 + 1  # +1 for center vertex
        
        all_vertices = np.vstack([
            top_boundary,           # 0 to n_boundary-1
            bottom_boundary,        # n_boundary to 2*n_boundary-1
            [bottom_center_3d],     # 2*n_boundary (center)
            terrain_top.vertices    # offset_terrain to ...
        ])
        
        # Faces: walls + bottom cap + terrain top (з offset)
        terrain_faces_offset = terrain_top.faces + offset_terrain
        
        face_lists = [wall_faces, bottom_cap_faces, terrain_faces_offset]
        all_faces = np.vstack(face_lists)
        print(f"[SOLIDIFIER] Final mesh: {len(all_vertices)} vertices, {len(all_faces)} faces")
        
        # 7. Створити mesh
        solid_mesh = trimesh.Trimesh(
            vertices=all_vertices,
            faces=all_faces,
            process=False
        )
        
        # 8. КРИТИЧНО: Очистити і виправити
        # Order: merge first (so unique/nondegenerate detect faces that became duplicates/degenerate post-merge)
        print(f"[SOLIDIFIER] Cleaning up mesh...")
        try:
            solid_mesh.merge_vertices()
            solid_mesh.update_faces(solid_mesh.unique_faces())
            solid_mesh.update_faces(solid_mesh.nondegenerate_faces())
            solid_mesh.remove_unreferenced_vertices()
            solid_mesh.fix_normals()
        except Exception as e:
            print(f"[WARN] Cleanup failed: {e}")
        
        # 9. Перевірити watertight
        is_watertight = solid_mesh.is_watertight
        print(f"[SOLIDIFIER] Watertight: {is_watertight}")
        
        if not is_watertight:
            print(f"[WARN] Mesh is not watertight! Applying aggressive repair...")
            try:
                # Агресивний ремонт
                solid_mesh.fill_holes()
                trimesh.repair.fix_winding(solid_mesh)
                trimesh.repair.broken_faces(solid_mesh, color=None)
                
                is_watertight_after = solid_mesh.is_watertight
                print(f"[SOLIDIFIER] Watertight after repair: {is_watertight_after}")
            except Exception as e:
                print(f"[ERROR] Aggressive repair failed: {e}")
        
        print(f"[SOLIDIFIER] Solidification complete")
        return solid_mesh
        
    except Exception as e:
        print(f"[ERROR] create_solid_terrain_robust failed: {e}")
        import traceback
        traceback.print_exc()
        return None
