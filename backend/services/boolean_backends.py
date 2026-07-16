from __future__ import annotations

import os
import time
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


def _weld_repair_subtract(terrain_mesh, cutter_mesh):
    """Self-contained robust cut: pass a NON-watertight terrain through the manifold3d
    constructor (topological repair into a valid solid — which fill_holes/Blender cannot
    do), then boolean-difference IN-PROCESS. Proven on a dense 290k-face non-watertight
    mesh for the connector notch. This is what keeps large-city grooves OFF the Blender
    subprocess (whose BMesh load OOM-kills a 4GB server). Inlined here (not imported) so
    the fix is self-contained and can hotfix a prod tree that predates the shared helper.
    Returns trimesh.Trimesh or None."""
    try:
        import numpy as _np
        from manifold3d import Manifold as _Manifold, Mesh as _MMesh
        from trimesh.grouping import group_rows as _grp

        def _open_edges(m):
            try:
                return len(_grp(m.edges_sorted, require_count=1))
            except Exception:
                return -1

        def _to_man(m):
            _mm = _MMesh(vert_properties=_np.asarray(m.vertices, dtype=_np.float32),
                         tri_verts=_np.asarray(m.faces, dtype=_np.uint32))
            return _Manifold(_mm)

        # Make manifold3d accept the mesh. Coincident-vertex gaps are only ONE cause of
        # NotManifold; DUPLICATE faces (edge shared by >2 faces) and DEGENERATE (zero-area)
        # faces are just as common on the parks/water-grooved terrain and merge_vertices
        # alone does NOT remove them. So each attempt: merge coincident verts at a given
        # precision + drop duplicate + degenerate faces + fix winding, then check topology.
        import trimesh.repair as _rr

        def _clean(m):
            try:
                m.update_faces(m.unique_faces())
                m.update_faces(m.nondegenerate_faces())
                m.remove_unreferenced_vertices()
                _rr.fix_winding(m)
            except Exception:
                pass
            return m

        _terr = terrain_mesh
        if _open_edges(_terr) != 0:
            for _dg in (6, 5, 4, 7):
                _w = terrain_mesh.copy()
                try:
                    _w.merge_vertices(digits_vertex=_dg)
                except Exception:
                    continue
                _clean(_w)
                if _open_edges(_w) == 0:
                    _terr = _w
                    break
                # keep the best-cleaned candidate even if not perfectly sealed —
                # manifold3d can still repair winding on a duplicate/degenerate-free mesh
                _terr = _w
        _t = _to_man(_terr)
        if _t.is_empty() or _t.num_tri() == 0:
            return None
        _c = _to_man(cutter_mesh)
        _res = _t - _c
        if _res.is_empty() or _res.num_tri() == 0:
            return None
        _mo = _res.to_mesh()
        _out = trimesh.Trimesh(
            vertices=_np.asarray(_mo.vert_properties)[:, :3].astype(_np.float64),
            faces=_np.asarray(_mo.tri_verts), process=False)
        return _out if len(_out.faces) > 0 else None
    except Exception as _e:  # noqa: BLE001
        print(f"[GROOVE] weld-repair subtract error: {_e}")
        return None


class ManifoldBooleanBackend:
    name = "manifold"

    def __init__(self, *, fallback_backend: Optional[BooleanBackend] = None) -> None:
        self.fallback_backend = fallback_backend or BlenderBooleanBackend()

    def cut_grooves(self, request: GrooveBooleanRequest) -> Optional[trimesh.Trimesh]:
        # PERF: [TIMING][GROOVE] breakdown of the per-generation groove-cutting stage —
        # nothing this fine-grained existed before (only whole-stage totals in
        # full_generation_pipeline._log_stage). Pure logging, no behavior change.
        _t_cutter_start = time.perf_counter()
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
        print(f"[TIMING][GROOVE] build_unified_cutter: {time.perf_counter() - _t_cutter_start:.2f}s")
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

            _t_conv_start = time.perf_counter()
            a = _to_manifold(request.terrain_mesh)
            b = _to_manifold(cutter_mesh)
            _t_bool_start = time.perf_counter()
            diff = a - b  # Manifold.__sub__ = difference
            if diff.is_empty():
                raise RuntimeError("manifold3d difference produced empty result")
            _t_bool_end = time.perf_counter()

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
            _t_cleanup_end = time.perf_counter()
            print(f"[TIMING][GROOVE] to_manifold_convert: {_t_bool_start - _t_conv_start:.2f}s, "
                  f"manifold_difference: {_t_bool_end - _t_bool_start:.2f}s, "
                  f"cleanup: {_t_cleanup_end - _t_bool_end:.2f}s")
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
            # OOM ROOT CAUSE: on large cities the CDT terrain is not perfectly
            # manifold (open edges) → manifold3d rejects it (NotManifold) → we USED
            # to fall straight to Blender, whose subprocess loads the huge mesh into
            # BMesh (~10× RAM) and OOM-kills the 4GB server mid-groove. Before that,
            # try the WELD-repair subtract (merge_vertices until open_edges=0 →
            # manifold3d accepts → in-process difference). Proven for the connector
            # notch; keeps us off Blender in the exact case that crashes. Blender
            # stays as the last resort only if the weld also fails.
            print(f"[GROOVE] Manifold boolean failed ({exc}); trying weld-repair before Blender fallback")
            try:
                _rep = _weld_repair_subtract(request.terrain_mesh, cutter_mesh)
                if _rep is not None and len(getattr(_rep, "faces", [])) > 0:
                    _rep.merge_vertices()
                    _rep.update_faces(_rep.unique_faces())
                    _rep.update_faces(_rep.nondegenerate_faces())
                    _rep.remove_unreferenced_vertices()
                    print(f"[GROOVE] weld-repair manifold subtract succeeded "
                          f"({len(_rep.faces)} faces) — Blender fallback (OOM risk) avoided")
                    return _rep
                print("[GROOVE] weld-repair subtract produced nothing")
            except Exception as _rex:  # noqa: BLE001
                print(f"[GROOVE] weld-repair subtract failed: {_rex}")
            # Dump the failing terrain+cutter so a proper repair can be built offline
            # against the REAL mesh (local gens don't reproduce this failure). Cheap STL,
            # overwritten each time; disable with GROOVE_FAILDUMP=0.
            if os.environ.get("GROOVE_FAILDUMP", "1") == "1":
                try:
                    import os as _os2
                    _dir = os.environ.get("GROOVE_FAILDUMP_DIR") or _os2.path.join(
                        _os2.path.dirname(_os2.path.dirname(_os2.path.abspath(__file__))), "output")
                    _os2.makedirs(_dir, exist_ok=True)
                    request.terrain_mesh.export(_os2.path.join(_dir, "groove_failmesh_terrain.stl"))
                    cutter_mesh.export(_os2.path.join(_dir, "groove_failmesh_cutter.stl"))
                    print(f"[GROOVE] dumped failing terrain+cutter → {_dir}/groove_failmesh_*.stl")
                except Exception as _dx:  # noqa: BLE001
                    print(f"[GROOVE] faildump skipped: {_dx}")
            # OOM-GUARD: Blender on a huge non-manifold mesh loads it into BMesh (~10× RAM)
            # and OOM-kills a low-RAM server (this is the 'task lost / server updating'
            # crash). On large terrain, DO NOT run Blender — skip the parks/water groove and
            # keep the (already road-grooved) terrain so the map STILL generates and the
            # server stays alive. Small meshes: Blender is cheap and safe.
            _faces = getattr(request.terrain_mesh, "faces", None)
            _nf = len(_faces) if _faces is not None else 0
            try:
                _guard = int(os.environ.get("GROOVE_BLENDER_MAX_FACES", "120000"))
            except Exception:  # noqa: BLE001
                _guard = 120000
            if _nf > _guard:
                print(f"[GROOVE] OOM-GUARD: terrain {_nf} faces > {_guard} → SKIP Blender "
                      f"fallback; parks/water grooves omitted (map still generates, server safe)")
                return request.terrain_mesh
            print(f"[GROOVE] falling back to {getattr(self.fallback_backend, 'name', 'fallback')} "
                  f"(small mesh {_nf}f — Blender safe)")
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
