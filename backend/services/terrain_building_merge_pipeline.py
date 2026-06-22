from __future__ import annotations

from typing import Any, Optional

import trimesh

from services.building_supports import union_mesh_collection
from services.processing_results import TerrainBuildingMergeResult
from services.terrain_cutter import extend_buildings_mesh_to_uniform_bottom


def _component_count(mesh: Optional[trimesh.Trimesh]) -> int:
    if mesh is None:
        return 0
    try:
        return len(mesh.split(only_watertight=False))
    except Exception:
        return 0


def merge_terrain_and_buildings(
    *,
    terrain_mesh: Optional[trimesh.Trimesh],
    building_meshes: Any,
    merged_building_mesh: Optional[trimesh.Trimesh] = None,
    support_meshes: Any = None,
    bottom_clearance_m: float = 0.0,
) -> TerrainBuildingMergeResult:
    # bottom_clearance_m: будинки опускаємо лише до (дно + clearance), НЕ до самого
    # дна. Для ЗʼЄДНУВАЧІВ це лишає нижню смугу (де ріжеться паз) ЧИСТОЮ підложкою
    # без будинків (інакше паз лишав «частинки будинків»). 0 = старе «до дна».
    _clr = max(float(bottom_clearance_m or 0.0), 0.0)
    if terrain_mesh is None:
        return TerrainBuildingMergeResult(
            terrain_mesh=terrain_mesh,
            building_meshes=building_meshes,
            merged_building_mesh=merged_building_mesh,
            support_meshes=support_meshes,
        )

    if merged_building_mesh is None:
        # Немає агрегованого building-меша. Якщо все ж є окремі будівлі —
        # ЗЛИВАЄМО їх у базу одним мешем (юзер: один шар), інакше просто база.
        has_buildings = (
            isinstance(building_meshes, (list, tuple)) and any(m is not None for m in building_meshes)
        ) or (building_meshes is not None and not isinstance(building_meshes, (list, tuple)))
        if has_buildings:
            try:
                target_z = float(terrain_mesh.bounds[0][2]) + _clr
                extend_buildings_mesh_to_uniform_bottom(building_meshes, target_z=target_z)
            except Exception as exc:
                print(f"[WARN] extend_buildings (no merged mesh) failed: {exc}")
            try:
                parts = [terrain_mesh]
                if isinstance(building_meshes, (list, tuple)):
                    parts.extend([m for m in building_meshes if m is not None])
                else:
                    parts.append(building_meshes)
                combined = trimesh.util.concatenate(parts) if len(parts) > 1 else terrain_mesh
                print(f"[INFO] base+buildings concatenated into one layer (no merged mesh path, {len(parts)} parts)")
            except Exception as exc:
                print(f"[WARN] concatenate (no merged mesh) failed: {exc}")
                combined = terrain_mesh
            return TerrainBuildingMergeResult(
                terrain_mesh=combined,
                building_meshes=None,
                merged_building_mesh=merged_building_mesh,
                support_meshes=support_meshes,
            )
        return TerrainBuildingMergeResult(
            terrain_mesh=terrain_mesh,
            building_meshes=None,
            merged_building_mesh=merged_building_mesh,
            support_meshes=support_meshes,
        )

    terrain_components = _component_count(terrain_mesh)
    building_components = _component_count(merged_building_mesh)

    # FIX (2026-05-15) — крок 1/2 для "один шар":
    # ДО boolean union опускаємо нижню грань кожної будівлі до самого дна
    # підложки terrain. Це гарантує що building solid ФІЗИЧНО перетинає
    # terrain solid (overlap не зникне на нерівному рельєфі), і
    # `union_mesh_collection` об'єднає їх в один merged solid замість
    # лишити плаваючі окремі компоненти.
    try:
        target_z_for_extend = float(terrain_mesh.bounds[0][2]) + _clr
        extend_buildings_mesh_to_uniform_bottom(
            building_meshes, target_z=target_z_for_extend
        )
        # Той самий extend на агрегаті, щоб не довелось перебудовувати union.
        extend_buildings_mesh_to_uniform_bottom(
            [merged_building_mesh], target_z=target_z_for_extend
        )
        print(
            f"[INFO] pre-merge: building bottoms extended to Z={target_z_for_extend:.4f} "
            f"(ensures overlap with terrain base for boolean union)"
        )
    except Exception as exc:
        print(f"[WARN] pre-merge extend_buildings failed: {exc}")

    base_mesh = None
    try:
        base_mesh = union_mesh_collection(
            [terrain_mesh, merged_building_mesh],
            label="terrain_buildings_base",
        )
    except Exception as exc:
        print(f"[WARN] terrain/building boolean merge failed: {exc}")
        base_mesh = None

    if base_mesh is not None:
        merged_components = _component_count(base_mesh)
        # NOTE: фрагментація більше НЕ скидає union у None. Юзер вимагає, щоб
        # будівлі та база були ОДНИМ шаром (без окремого building-меша). Навіть
        # якщо boolean union дав кілька disjoint-компонентів — це все одно один
        # combined mesh-об'єкт у 3MF, що й потрібно. Лишаємо лише діагностику.
        if merged_components > max(terrain_components + 8, terrain_components * 4):
            print(
                "[INFO] terrain/building union has multiple components "
                f"({merged_components}; terrain={terrain_components}, buildings={building_components}); "
                "still exported as a SINGLE base+buildings layer (user requested one layer)"
            )

    # ОДИН ШАР: будівлі ЗАВЖДИ зливаються з базою. Якщо boolean union не вдався,
    # робимо концат (геометричне об'єднання в один mesh-об'єкт) — для
    # одноматеріального друку це коректно нарізається слайсером. Окремий
    # building-шар більше НЕ експортується (building_meshes=None).
    if base_mesh is None:
        try:
            target_z = float(terrain_mesh.bounds[0][2]) + _clr
            extend_buildings_mesh_to_uniform_bottom(
                building_meshes, target_z=target_z
            )
            print(
                f"[INFO] merge fallback: building bottoms extended to Z={target_z:.4f} "
                f"(boolean union failed; concatenating into single base mesh)"
            )
        except Exception as exc:
            print(f"[WARN] extend_buildings_mesh_to_uniform_bottom failed: {exc}")
        try:
            parts = [terrain_mesh]
            if isinstance(building_meshes, (list, tuple)):
                parts.extend([m for m in building_meshes if m is not None])
            elif building_meshes is not None:
                parts.append(building_meshes)
            base_mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else terrain_mesh
            print(
                f"[INFO] base+buildings concatenated into one layer "
                f"({len(parts)} parts -> single mesh)"
            )
        except Exception as exc:
            print(f"[WARN] concatenate base+buildings failed ({exc}); keeping terrain only")
            base_mesh = terrain_mesh

    return TerrainBuildingMergeResult(
        terrain_mesh=base_mesh,
        building_meshes=None,
        merged_building_mesh=merged_building_mesh,
        support_meshes=support_meshes,
    )
