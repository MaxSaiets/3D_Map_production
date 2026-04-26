"""
Сервіс для перевірки та покращення якості mesh для 3D принтера
Перевіряє мінімальні розміри, товщини, watertight та інші параметри
Включає детекцію та ремонт non-manifold edges
"""
import trimesh
import numpy as np
from typing import Optional, Tuple, List, Dict
import trimesh.repair


# Мінімальні розміри для 3D принтера (в міліметрах після масштабування)
MIN_WALL_THICKNESS_MM = 0.3  # Мінімальна товщина стінки
MIN_FEATURE_SIZE_MM = 0.2    # Мінімальний розмір деталі
MIN_OVERHANG_ANGLE = 45.0     # Мінімальний кут для overhang (градуси)


def detect_nonmanifold_edges(mesh: trimesh.Trimesh) -> Tuple[int, List[int]]:
    """
    Детектує non-manifold edges в mesh.
    
    Non-manifold edge = edge що належить != 2 граням:
    - 1 грань = boundary edge (дірка)
    - >2 граней = non-manifold edge (перетин)
    
    Returns:
        (кількість_non_manifold_edges, список_індексів)
    """
    try:
        if mesh is None or len(mesh.faces) == 0:
            return 0, []

        # Vectorized: replaced O(n) Python loop with numpy batch ops (~100x faster
        # for large meshes; edges_face shape is (n_edges, 2) with -1 for missing face).
        edges_face = mesh.edges_face
        if edges_face.ndim < 2:
            return 0, []
        face_counts = np.sum(edges_face >= 0, axis=1)
        nonmanifold_indices = np.where(face_counts != 2)[0]
        return int(len(nonmanifold_indices)), nonmanifold_indices.tolist()

    except Exception as e:
        print(f"[WARN] Помилка детекції non-manifold edges: {e}")
        return 0, []



def repair_nonmanifold(
    mesh: trimesh.Trimesh,
    aggressive: bool = True,
    verbose: bool = True,
    skip_fix_normals: bool = False
) -> Tuple[trimesh.Trimesh, Dict[str, any]]:
    """
    Виправляє non-manifold edges та інші проблеми mesh.
    
    Args:
        mesh: Trimesh об'єкт для ремонту
        aggressive: Якщо True, застосовує більш агресивні методи ремонту
        verbose: Якщо True, виводить детальну інформацію
        
    Returns:
        Tuple[Trimesh, Dict]: (відремонтований mesh, статистика ремонту)
    """
    if mesh is None:
        return None, {"error": "Mesh is None"}
    
    stats = {
        "initial_vertices": len(mesh.vertices),
        "initial_faces": len(mesh.faces),
        "initial_nonmanifold_edges": 0,
        "initial_watertight": mesh.is_watertight,
        "repairs_applied": [],
    }
    
    # Перевіряємо початковий стан
    initial_nm_count, initial_nm_edges = detect_nonmanifold_edges(mesh)
    stats["initial_nonmanifold_edges"] = initial_nm_count
    
    if verbose:
        print(f"[MESH REPAIR] Початковий стан: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, non-manifold edges={initial_nm_count}, watertight={mesh.is_watertight}")
    
    repaired = mesh.copy()
    
    try:
        # Крок 1: Видалення дублікатів та дегенерованих граней
        initial_face_count = len(repaired.faces)
        repaired.update_faces(repaired.unique_faces())
        repaired.update_faces(repaired.nondegenerate_faces())
        if len(repaired.faces) < initial_face_count:
            removed = initial_face_count - len(repaired.faces)
            stats["repairs_applied"].append(f"Removed {removed} duplicate/degenerate faces")
            if verbose:
                print(f"[MESH REPAIR] Видалено {removed} дублікатів/дегенерованих граней")
        
        # Крок 2: Об'єднання близьких вершин
        initial_vertex_count = len(repaired.vertices)
        repaired.merge_vertices()
        if len(repaired.vertices) < initial_vertex_count:
            merged = initial_vertex_count - len(repaired.vertices)
            stats["repairs_applied"].append(f"Merged {merged} vertices")
            if verbose:
                print(f"[MESH REPAIR] Об'єднано {merged} вершин")
        # After merge, faces that shared duplicate vertices may now be identical or degenerate —
        # remove them (same order fix as solidifier_robust.py).
        repaired.update_faces(repaired.unique_faces())
        repaired.update_faces(repaired.nondegenerate_faces())

        # Крок 3: Видалення невикористовуваних вершин
        repaired.remove_unreferenced_vertices()
        
        # Крок 4: Виправлення порядку вершин (winding) ПЕРЕД нормалями
        try:
            if not repaired.is_winding_consistent:
                trimesh.repair.fix_winding(repaired)
                stats["repairs_applied"].append("Fixed winding consistency")
                if verbose:
                    print(f"[MESH REPAIR] Виправлено порядок вершин")
        except Exception as e:
            if verbose:
                print(f"[MESH REPAIR] Не вдалось виправити winding: {e}")

        # Крок 5: Виправлення нормалей
        # КРИТИЧНО: НЕ викликаємо fix_normals() на рельєфі з пазами (grooves) —
        # ray casting heuristic інвертує нормалі на нековпуклих мешах.
        if not skip_fix_normals:
            try:
                repaired.fix_normals()
                stats["repairs_applied"].append("Fixed normals")
                if verbose:
                    print(f"[MESH REPAIR] Виправлено нормалі")
            except Exception as e:
                if verbose:
                    print(f"[MESH REPAIR] Не вдалось виправити нормалі: {e}")
        else:
            if verbose:
                print(f"[MESH REPAIR] Пропущено fix_normals (skip_fix_normals=True)")
        
        # Крок 6: Заповнення дірок
        # Пропускаємо для мешів з пазами — fill_holes може закрити пази
        if not repaired.is_watertight and not skip_fix_normals:
            try:
                repaired.fill_holes()
                if repaired.is_watertight:
                    stats["repairs_applied"].append("Filled holes - mesh is now watertight")
                    if verbose:
                        print(f"[MESH REPAIR] Заповнено дірки, mesh тепер watertight")
                else:
                    stats["repairs_applied"].append("Attempted to fill holes - some remain")
                    if verbose:
                        print(f"[MESH REPAIR] Спроба заповнити дірки - деякі залишились")
            except Exception as e:
                if verbose:
                    print(f"[MESH REPAIR] Помилка заповнення дірок: {e}")
        
        # Крок 7: Агресивні методи ремонту (якщо потрібно)
        if aggressive and not repaired.is_watertight and not skip_fix_normals:
            try:
                # Спробуємо використати trimesh.repair.broken_faces
                trimesh.repair.broken_faces(repaired)
                stats["repairs_applied"].append("Applied broken_faces repair")
                if verbose:
                    print(f"[MESH REPAIR] Застосовано ремонт зламаних граней")
            except Exception as e:
                if verbose:
                    print(f"[MESH REPAIR] Не вдалось застосувати broken_faces: {e}")
            
            # Якщо все ще не watertight, спробуємо convex hull як останній варіант
            # НЕ РЕКОМЕНДУЄТЬСЯ для складних моделей, бо змінює геометрію
            # if not repaired.is_watertight:
            #     try:
            #         repaired = repaired.convex_hull
            #         stats["repairs_applied"].append("WARNING: Applied convex hull (geometry changed)")
            #         if verbose:
            #             print(f"[MESH REPAIR] УВАГА: Застосовано convex hull (геометрія змінена)")
            #     except Exception as e:
            #         if verbose:
            #             print(f"[MESH REPAIR] Не вдалось застосувати convex hull: {e}")
        
        # Фінальна очистка
        repaired.update_faces(repaired.nondegenerate_faces())
        repaired.remove_unreferenced_vertices()
        
        # Перевірка фінального стану
        if len(repaired.vertices) == 0 or len(repaired.faces) == 0:
            if verbose:
                print("[MESH REPAIR] ПОМИЛКА: Mesh став порожнім після ремонту, повертаємо оригінал")
            return mesh, {**stats, "error": "Mesh became empty after repair"}
        
        final_nm_count, final_nm_edges = detect_nonmanifold_edges(repaired)
        
        stats["final_vertices"] = len(repaired.vertices)
        stats["final_faces"] = len(repaired.faces)
        stats["final_nonmanifold_edges"] = final_nm_count
        stats["final_watertight"] = repaired.is_watertight
        stats["improvement"] = initial_nm_count - final_nm_count
        
        if verbose:
            print(f"[MESH REPAIR] Фінальний стан: vertices={len(repaired.vertices)}, faces={len(repaired.faces)}, non-manifold edges={final_nm_count}, watertight={repaired.is_watertight}")
            print(f"[MESH REPAIR] Покращення: {stats['improvement']} non-manifold edges виправлено")
        
        return repaired, stats
        
    except Exception as e:
        if verbose:
            print(f"[MESH REPAIR] Критична помилка ремонту: {e}")
        return mesh, {**stats, "error": str(e)}


def validate_mesh_for_3d_printing(
    mesh: trimesh.Trimesh,
    scale_factor: Optional[float] = None,
    model_size_mm: float = 100.0,
) -> Tuple[bool, List[str]]:
    """
    Перевіряє mesh на придатність для 3D принтера
    
    Args:
        mesh: Trimesh об'єкт для перевірки
        scale_factor: Фактор масштабування (якщо відомий)
        model_size_mm: Розмір моделі в мм (для оцінки масштабу)
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_warnings)
    """
    warnings = []
    
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return False, ["Mesh порожній або невалідний"]
    
    # 1. Перевірка watertight
    if not mesh.is_watertight:
        warnings.append("Mesh не є watertight (можуть бути дірки)")
        try:
            mesh.fill_holes()
            if not mesh.is_watertight:
                warnings.append("Не вдалося заповнити всі дірки")
        except Exception as e:
            warnings.append(f"Помилка заповнення дірок: {e}")
    
    # 2. Перевірка мінімальних розмірів
    try:
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        min_dim = float(np.min(size))
        
        # Оцінюємо масштаб
        if scale_factor is None:
            # Оцінюємо за model_size_mm
            avg_xy = (size[0] + size[1]) / 2.0
            if avg_xy > 0:
                estimated_scale = model_size_mm / avg_xy
            else:
                estimated_scale = 1.0
        else:
            estimated_scale = scale_factor
        
        # Конвертуємо в мм
        min_dim_mm = min_dim * estimated_scale
        
        if min_dim_mm < MIN_FEATURE_SIZE_MM:
            warnings.append(f"Мінімальний розмір занадто малий: {min_dim_mm:.2f}мм (мінімум: {MIN_FEATURE_SIZE_MM}мм)")
    except Exception as e:
        warnings.append(f"Помилка перевірки розмірів: {e}")
    
    # 3+5. Перевірка товщини стінок і дегенерованих граней — vectorized.
    # Replaced two O(n) Python face loops with one numpy batch operation (~100x faster
    # for large meshes like terrain with ~100k faces).
    try:
        if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
            v = mesh.vertices[mesh.faces]          # (N, 3, 3)
            e1 = v[:, 1] - v[:, 0]                # (N, 3)
            e2 = v[:, 2] - v[:, 0]                # (N, 3)
            face_areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)  # (N,)

            min_area_mm2 = float(np.min(face_areas)) * (estimated_scale ** 2)
            if min_area_mm2 < 0.01:
                warnings.append(f"Знайдено дуже малі грані: {min_area_mm2:.4f}мм^2")

            degenerate_count = int(np.sum(face_areas < 1e-10))
            if degenerate_count > 0:
                warnings.append(f"Знайдено {degenerate_count} дегенерованих граней")
    except Exception as e:
        warnings.append(f"Помилка перевірки граней: {e}")

    # 4. Перевірка нормалей
    try:
        if not mesh.is_winding_consistent:
            warnings.append("Порядок вершин граней неконсистентний")
    except:
        pass
    
    is_valid = len(warnings) == 0 or all("можуть бути" in w or "дуже малі" in w for w in warnings)
    return is_valid, warnings


def improve_mesh_for_3d_printing(
    mesh: trimesh.Trimesh,
    aggressive: bool = True,
    verbose: bool = False,
    skip_fix_normals: bool = False,
) -> trimesh.Trimesh:
    """
    Покращує mesh для 3D принтера, включаючи ремонт non-manifold edges
    
    Args:
        mesh: Trimesh об'єкт для покращення
        aggressive: Якщо True, застосовує більш агресивні виправлення
        verbose: Якщо True, виводить детальну інформацію про ремонт
        
    Returns:
        Покращений Trimesh об'єкт
    """
    if mesh is None:
        return None
    
    # Використовуємо нову функцію repair_nonmanifold для комплексного ремонту
    improved, stats = repair_nonmanifold(mesh, aggressive=aggressive, verbose=verbose, skip_fix_normals=skip_fix_normals)
    
    # Якщо ремонт не вдався, повертаємо оригінал
    if improved is None or len(improved.vertices) == 0 or len(improved.faces) == 0:
        if verbose:
            print("[WARN] Mesh став порожнім після покращень, повертаємо оригінал")
        return mesh
    
    try:
        if mesh.is_watertight and not improved.is_watertight:
            if verbose:
                print("[WARN] Mesh repair regressed watertight topology; keeping original mesh")
            return mesh
    except Exception:
        pass

    return improved


def check_minimum_thickness(
    mesh: trimesh.Trimesh,
    scale_factor: Optional[float] = None,
    model_size_mm: float = 100.0,
) -> Tuple[bool, float]:
    """
    Перевіряє мінімальну товщину mesh
    
    Returns:
        Tuple[bool, float]: (is_valid, min_thickness_mm)
    """
    try:
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        min_dim = float(np.min(size))
        
        # Оцінюємо масштаб
        if scale_factor is None:
            avg_xy = (size[0] + size[1]) / 2.0
            if avg_xy > 0:
                estimated_scale = model_size_mm / avg_xy
            else:
                estimated_scale = 1.0
        else:
            estimated_scale = scale_factor
        
        min_thickness_mm = min_dim * estimated_scale
        is_valid = min_thickness_mm >= MIN_WALL_THICKNESS_MM
        
        return is_valid, min_thickness_mm
    except Exception:
        return False, 0.0

