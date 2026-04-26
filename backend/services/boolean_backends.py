from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol

import trimesh
from shapely.geometry.base import BaseGeometry

from services.groove_cutter_builder import build_unified_groove_cutter
from services.terrain_cutter import cut_all_grooves, cut_grooves_sequentially


@dataclass
class GrooveBooleanRequest:
    terrain_mesh: trimesh.Trimesh
    road_polygons: Optional[BaseGeometry]
    road_clearance_m: float
    parks_polygons: Optional[BaseGeometry]
    parks_clearance_m: float
    parks_mesh: Optional[trimesh.Trimesh]
    water_polygons: Optional[BaseGeometry]
    water_clearance_m: float
    water_mesh: Optional[trimesh.Trimesh]
    scale_factor: float
    road_mesh: Optional[trimesh.Trimesh]
    groove_depth_m: float


class BooleanBackend(Protocol):
    def cut_grooves(self, request: GrooveBooleanRequest) -> Optional[trimesh.Trimesh]:
        ...


class BlenderBooleanBackend:
    name = "blender"

    def cut_grooves(self, request: GrooveBooleanRequest) -> Optional[trimesh.Trimesh]:
        return cut_grooves_sequentially(
            terrain_mesh=request.terrain_mesh,
            road_polygons=request.road_polygons,
            road_clearance_m=request.road_clearance_m,
            parks_polygons=request.parks_polygons,
            parks_clearance_m=request.parks_clearance_m,
            parks_mesh=request.parks_mesh,
            water_polygons=request.water_polygons,
            water_clearance_m=request.water_clearance_m,
            water_mesh=request.water_mesh,
            scale_factor=request.scale_factor,
            road_mesh=request.road_mesh,
            groove_depth_m=request.groove_depth_m,
        )


class NoOpBooleanBackend:
    name = "noop"

    def cut_grooves(self, request: GrooveBooleanRequest) -> Optional[trimesh.Trimesh]:
        return request.terrain_mesh


class ManifoldBooleanBackend:
    name = "manifold"

    def __init__(self, *, fallback_backend: Optional[BooleanBackend] = None) -> None:
        self.fallback_backend = fallback_backend or BlenderBooleanBackend()

    def cut_grooves(self, request: GrooveBooleanRequest) -> Optional[trimesh.Trimesh]:
        cutter_result = build_unified_groove_cutter(
            terrain_mesh=request.terrain_mesh,
            road_polygons=request.road_polygons,
            road_clearance_m=request.road_clearance_m,
            parks_polygons=request.parks_polygons,
            parks_clearance_m=request.parks_clearance_m,
            parks_mesh=request.parks_mesh,
            water_polygons=request.water_polygons,
            water_clearance_m=request.water_clearance_m,
            water_mesh=request.water_mesh,
            road_mesh=request.road_mesh,
            groove_depth_m=request.groove_depth_m,
        )
        cutter_mesh = cutter_result.cutter_mesh
        if cutter_mesh is None:
            return request.terrain_mesh

        try:
            import manifold3d
        except Exception as exc:
            print(f"[GROOVE] Manifold backend unavailable, falling back to {getattr(self.fallback_backend, 'name', 'fallback')}: {exc}")
            return self.fallback_backend.cut_grooves(request)

        try:
            # Call manifold3d directly instead of through trimesh.boolean.difference —
            # trimesh calls Manifold.from_mesh() which does not exist in the installed
            # manifold3d version. The installed version uses Manifold(Mesh(...)) directly.
            import numpy as np

            def _to_manifold(mesh: trimesh.Trimesh) -> "manifold3d.Manifold":
                verts = np.asarray(mesh.vertices, dtype=np.float32)
                tris = np.asarray(mesh.faces, dtype=np.uint32)
                m = manifold3d.Manifold(
                    manifold3d.Mesh(vert_properties=verts, tri_verts=tris)
                )
                if m.is_empty():
                    raise RuntimeError(
                        f"manifold3d conversion produced empty Manifold (status={m.status()})"
                    )
                return m

            a = _to_manifold(request.terrain_mesh)
            b = _to_manifold(cutter_mesh)
            diff = a - b  # Manifold.__sub__ = difference
            if diff.is_empty():
                raise RuntimeError("manifold3d difference produced empty result")

            md = diff.to_mesh()
            verts_out = np.array(md.vert_properties, dtype=np.float64)[:, :3]
            faces_out = np.array(md.tri_verts, dtype=np.int64)
            result = trimesh.Trimesh(vertices=verts_out, faces=faces_out, process=False)

            # Manifold leaves duplicate vertices at boolean boundary edges (groove walls/floors),
            # causing thousands of disconnected components. Merge them immediately — same fix
            # as mesh_clipper.py after slice_mesh_plane.
            result.merge_vertices()
            result.update_faces(result.unique_faces())
            result.update_faces(result.nondegenerate_faces())
            result.remove_unreferenced_vertices()
            if not bool(getattr(result, "is_watertight", False)):
                # Keep manifold output even when not perfectly watertight.
                # Blender fallback can introduce coordinate-frame drift on some
                # tiles; downstream groove validation will still reject bad
                # topology/space shifts.
                print("[GROOVE] Manifold result is non-watertight; keeping manifold output for stabilization checks")
            print(f"[GROOVE] Manifold direct: {len(result.vertices)} verts, "
                  f"{len(result.faces)} faces, {len(result.split())} components")
            return result
        except Exception as exc:
            print(f"[GROOVE] Manifold boolean failed, falling back to {getattr(self.fallback_backend, 'name', 'fallback')}: {exc}")
            return self.fallback_backend.cut_grooves(request)


def get_available_boolean_backends() -> list[str]:
    names = ["blender", "noop"]
    try:
        import manifold3d  # noqa: F401

        names.insert(0, "manifold")
    except Exception:
        pass
    names.insert(0, "auto")
    return names


def resolve_boolean_backend(
    backend: Optional[BooleanBackend] = None,
    *,
    backend_name: Optional[str] = None,
) -> BooleanBackend:
    if backend is not None:
        return backend

    selected_name = (backend_name or os.environ.get("BOOLEAN_BACKEND") or "auto").strip().lower()
    if selected_name in {"noop", "none", "disabled"}:
        return NoOpBooleanBackend()
    if selected_name == "manifold":
        return ManifoldBooleanBackend()
    if selected_name == "auto":
        try:
            import manifold3d  # noqa: F401

            return ManifoldBooleanBackend()
        except Exception:
            return BlenderBooleanBackend()

    return BlenderBooleanBackend()
