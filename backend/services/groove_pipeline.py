from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh
from shapely.geometry.base import BaseGeometry

from services.boolean_backends import (
    BooleanBackend,
    GrooveBooleanRequest,
    resolve_boolean_backend,
)
from services.detail_layer_utils import (
    MICRO_REGION_THRESHOLD_MM,
    model_mm_to_world_m,
)
from services.road_processor import merge_close_road_gaps
from services.terrain_cutter import cut_grooves_sequentially
from shapely.ops import unary_union


def _apply_exclusion_mask(geometry, exclusion_mask):
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    if exclusion_mask is None or getattr(exclusion_mask, "is_empty", True):
        return geometry
    try:
        geometry = geometry.difference(exclusion_mask)
        if geometry is not None and not getattr(geometry, "is_empty", True):
            geometry = geometry.buffer(0)
    except Exception:
        pass
    return geometry


def _drop_tiny_material_fragments(geometry, *, min_feature_m: float):
    if geometry is None or getattr(geometry, "is_empty", True) or min_feature_m <= 0:
        return geometry

    min_area_m2 = max((float(min_feature_m) ** 2) * 0.5, 1e-8)
    kept = []
    geoms = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
    for geom in geoms:
        if geom is None or getattr(geom, "is_empty", True):
            continue
        try:
            minx, miny, maxx, maxy = geom.bounds
            min_dim = min(float(maxx - minx), float(maxy - miny))
            area = float(getattr(geom, "area", 0.0) or 0.0)
            if min_dim < float(min_feature_m) or area < min_area_m2:
                continue
        except Exception:
            pass
        kept.append(geom)

    if not kept:
        return None
    try:
        return unary_union(kept).buffer(0)
    except Exception:
        return kept[0]


def prepare_building_groove_mask(
    building_polygons,
    *,
    groove_clearance_m: float,
    zone_polygon=None,
):
    """Thin ring around buildings where roads/parks/water must NOT reach.

    Why: buildings printed flush with roads merge into one mass on the final
    layer. A small clearance ring around each footprint gives the slicer a
    visible separator so walls and curbs don't fuse (user report 2026-04).

    How to apply: call once per zone, then subtract the returned geometry from
    roads, road_groove, parks, parks_groove, water, water_groove (but NOT from
    the building footprint itself).
    """
    if building_polygons is None or getattr(building_polygons, "is_empty", True):
        return None
    if groove_clearance_m <= 0.0:
        return None
    try:
        ring = building_polygons.buffer(float(groove_clearance_m), join_style=2)
        ring = ring.difference(building_polygons).buffer(0)
    except Exception:
        return None
    if ring is None or getattr(ring, "is_empty", True):
        return None
    if zone_polygon is not None and not getattr(zone_polygon, "is_empty", True):
        try:
            ring = ring.intersection(zone_polygon).buffer(0)
        except Exception:
            pass
    if ring is None or getattr(ring, "is_empty", True):
        return None
    return ring


def _prepare_parks_groove_mask(
    parks_polygons,
    *,
    road_groove_mask=None,
    water_polygons=None,
    building_polygons=None,
    groove_clearance_m: float,
    boundary_snap_m: float,
    zone_prefix: str = "",
):
    if parks_polygons is None or getattr(parks_polygons, "is_empty", True):
        return None

    # parks_polygons is already the canonical park insert mask. Re-subtracting
    # neighbouring road/water grooves here erases the park-side fit at shared
    # boundaries, so only keep hard building exclusions.
    parks_insert_mask = _apply_exclusion_mask(parks_polygons, building_polygons)
    if parks_insert_mask is None or getattr(parks_insert_mask, "is_empty", True):
        return None

    try:
        park_groove_mask = (
            parks_insert_mask.buffer(float(groove_clearance_m), join_style=2)
            if groove_clearance_m > 0.0
            else parks_insert_mask
        )
        park_groove_mask = park_groove_mask.buffer(0)
    except Exception:
        park_groove_mask = parks_insert_mask

    # Buildings keep hard precedence, but neighbouring inserts/grooves should
    # retain their own park-side fit instead of being suppressed here.
    for exclusion_mask, label in (
        (building_polygons, "buildings"),
    ):
        if exclusion_mask is None or getattr(exclusion_mask, "is_empty", True):
            continue
        try:
            park_groove_mask = park_groove_mask.difference(exclusion_mask).buffer(0)
        except Exception as exc:
            print(f"[DEBUG] {zone_prefix} Failed to subtract {label} from park groove: {exc}")

    return park_groove_mask


def _prepare_water_groove_mask(
    water_polygons,
    *,
    road_groove_mask=None,
    parks_groove_mask=None,
    building_polygons=None,
    groove_clearance_m: float,
    boundary_snap_m: float,
    zone_prefix: str = "",
):
    if water_polygons is None or getattr(water_polygons, "is_empty", True):
        return None

    # water_polygons is already the canonical insert. Suppressing it against
    # neighbouring grooves removes the water-side fit near roads/parks, so
    # only apply hard building exclusions here.
    water_insert_mask = _apply_exclusion_mask(water_polygons, building_polygons)
    if water_insert_mask is None or getattr(water_insert_mask, "is_empty", True):
        return None

    try:
        water_groove_mask = (
            water_insert_mask.buffer(float(groove_clearance_m), join_style=2)
            if groove_clearance_m > 0.0
            else water_insert_mask
        )
        water_groove_mask = water_groove_mask.buffer(0)
    except Exception:
        water_groove_mask = water_insert_mask

    for exclusion_mask, label in (
        (building_polygons, "buildings"),
    ):
        if exclusion_mask is None or getattr(exclusion_mask, "is_empty", True):
            continue
        try:
            water_groove_mask = water_groove_mask.difference(exclusion_mask).buffer(0)
        except Exception as exc:
            print(f"[DEBUG] {zone_prefix} Failed to subtract {label} from water groove: {exc}")

    return water_groove_mask


def _fill_narrow_road_holes(
    road_cut_mask,
    min_printable_m,
    preserve_geom=None,
    zone_prefix="",
):
    """Fill interior voids in road polygon that are too narrow to print (-> treated as road).

    When two parallel road grooves nearly touch, a thin enclosed terrain island forms
    between them. It won't print. This finds such holes and fills them so the whole
    enclosed area becomes road groove.
    """
    from shapely.geometry import Polygon

    holes_to_fill = []
    geoms = road_cut_mask.geoms if hasattr(road_cut_mask, "geoms") else [road_cut_mask]
    for geom in geoms:
        if not isinstance(geom, Polygon):
            continue
        for interior in geom.interiors:
            hole = Polygon(interior)
            # Hole is unprintable if it doesn't survive erosion by min_printable_m
            try:
                if preserve_geom is not None and not getattr(preserve_geom, "is_empty", True):
                    if hole.intersects(preserve_geom):
                        continue
                minx, miny, maxx, maxy = hole.bounds
                min_dim = min(float(maxx - minx), float(maxy - miny))
                area = float(getattr(hole, "area", 0.0) or 0.0)
                max_hole_area_m2 = max((float(min_printable_m) ** 2) * 10.0, 1e-8)
                max_hole_min_dim_m = float(min_printable_m) * 1.15
                # Preserve meaningful terrain islands. Only fill truly tiny holes
                # that are effectively narrower than the printable threshold.
                if min_dim > max_hole_min_dim_m or area > max_hole_area_m2:
                    continue
                if hole.buffer(-min_printable_m).is_empty:
                    holes_to_fill.append(hole)
            except Exception:
                pass
    if holes_to_fill:
        try:
            filled = unary_union(holes_to_fill)
            road_cut_mask = road_cut_mask.union(filled).buffer(0)
            print(
                f"[INFO] {zone_prefix} Filled {len(holes_to_fill)} narrow road interior"
                f" voids (< {min_printable_m:.4f}m wide) -> road"
            )
        except Exception as exc:
            print(f"[DEBUG] {zone_prefix} road hole-fill failed: {exc}")
    return road_cut_mask


def _restore_inner_hole_clearance(
    road_source_mask,
    *,
    clearance_m: float,
    preserve_geom=None,
    zone_prefix="",
):
    """Restore groove clearance on the inner side of road holes.

    Some cleanup passes are intentionally aggressive and can collapse the
    clearance ring around meaningful terrain islands inside road polygons.
    This helper rebuilds the expected inner ring directly from the canonical
    road-hole geometry so roundabouts and similar loops keep a visible groove
    on both sides of the road.
    """
    from shapely.geometry import Polygon

    if (
        road_source_mask is None
        or getattr(road_source_mask, "is_empty", True)
        or clearance_m <= 0
    ):
        return None

    preserve_buffer = None
    if preserve_geom is not None and not getattr(preserve_geom, "is_empty", True):
        try:
            preserve_buffer = preserve_geom.buffer(0)
        except Exception:
            preserve_buffer = preserve_geom

    restored_rings = []
    geoms = road_source_mask.geoms if hasattr(road_source_mask, "geoms") else [road_source_mask]
    for geom in geoms:
        if not isinstance(geom, Polygon):
            continue
        for interior in geom.interiors:
            try:
                hole = Polygon(interior)
            except Exception:
                continue
            if hole.is_empty:
                continue
            if preserve_buffer is not None:
                try:
                    if hole.intersects(preserve_buffer):
                        continue
                except Exception:
                    pass
            try:
                shrunk = hole.buffer(-float(clearance_m), join_style=2)
            except Exception:
                shrunk = None
            try:
                ring = hole if shrunk is None or getattr(shrunk, "is_empty", True) else hole.difference(shrunk)
                if ring is None or getattr(ring, "is_empty", True):
                    continue
                ring = ring.buffer(0)
                if ring is None or getattr(ring, "is_empty", True):
                    continue
                restored_rings.append(ring)
            except Exception:
                continue

    if not restored_rings:
        return None

    try:
        restored = unary_union(restored_rings).buffer(0)
        if restored is not None and not getattr(restored, "is_empty", True):
            print(f"[INFO] {zone_prefix} Restored inner road groove clearance around {len(restored_rings)} holes")
        return restored
    except Exception:
        return None


def _absorb_terrain_slivers(
    zone_polygon,
    road_cut_mask,
    parks_polys,
    water_polys,
    min_printable_m,
    protected_voids=None,
    zone_prefix="",
):
    """Find thin terrain strips that won't print and absorb each into the nearest groove.

    Priority: road > park > water.
    Returns updated (road_cut_mask, parks_polys, water_polys).
    """
    all_grooves = [
        g for g in (road_cut_mask, parks_polys, water_polys)
        if g is not None and not getattr(g, "is_empty", True)
    ]
    if not all_grooves:
        return road_cut_mask, parks_polys, water_polys

    try:
        combined = unary_union(all_grooves).buffer(0)
        terrain_2d = zone_polygon.difference(combined)
    except Exception as exc:
        print(f"[DEBUG] {zone_prefix} sliver terrain diff failed: {exc}")
        return road_cut_mask, parks_polys, water_polys

    if terrain_2d.is_empty:
        return road_cut_mask, parks_polys, water_polys

    half_m = min_printable_m / 2.0
    try:
        eroded = terrain_2d.buffer(-half_m)
        if eroded.is_empty:
            # Entire terrain zone is thin — something is wrong, skip
            return road_cut_mask, parks_polys, water_polys
        printable = eroded.buffer(half_m)
        slivers = terrain_2d.difference(printable)
    except Exception as exc:
        print(f"[DEBUG] {zone_prefix} sliver erosion failed: {exc}")
        return road_cut_mask, parks_polys, water_polys

    if slivers.is_empty:
        return road_cut_mask, parks_polys, water_polys

    proximity = min_printable_m * 2
    road_eat, park_eat, water_eat = [], [], []
    parts = list(slivers.geoms) if hasattr(slivers, "geoms") else [slivers]

    # Pre-build road influence zone once (expanding road mask by proximity).
    # A sliver is "between roads" only if it is FULLY enclosed in this zone —
    # meaning roads surround it from all sides (not just touch one edge).
    road_influence = None
    if road_cut_mask is not None and not road_cut_mask.is_empty:
        try:
            road_influence = road_cut_mask.buffer(proximity)
        except Exception:
            pass

    def _candidate_score(sliver_geom, target_geom):
        if target_geom is None or getattr(target_geom, "is_empty", True):
            return 0.0
        try:
            expanded = target_geom.buffer(max(min_printable_m * 0.2, 0.05))
            shared = sliver_geom.boundary.intersection(expanded).length
            if shared > 0:
                return float(shared)
            if sliver_geom.intersects(expanded):
                return float(sliver_geom.intersection(expanded).area)
        except Exception:
            return 0.0
        return 0.0

    def _candidate_distance(sliver_geom, target_geom):
        if target_geom is None or getattr(target_geom, "is_empty", True):
            return float("inf")
        try:
            return float(sliver_geom.distance(target_geom))
        except Exception:
            return float("inf")

    for sliver in parts:
        if getattr(sliver, "is_empty", True):
            continue
        try:
            minx, miny, maxx, maxy = sliver.bounds
            min_dim = min(float(maxx - minx), float(maxy - miny))
            max_dim = max(float(maxx - minx), float(maxy - miny))
            area = float(getattr(sliver, "area", 0.0) or 0.0)
            max_sliver_area_m2 = max((float(min_printable_m) ** 2) * 18.0, 1e-8)
            # Only absorb genuinely tiny printable failures. If a terrain patch
            # has meaningful width/area, preserve it as relief.
            if min_dim > (float(min_printable_m) * 1.1):
                continue
            if area > max_sliver_area_m2 and max_dim > (float(min_printable_m) * 4.0):
                continue
            if protected_voids is not None and not getattr(protected_voids, "is_empty", True):
                if sliver.overlaps(protected_voids) or sliver.within(protected_voids):
                    continue
            park_score = _candidate_score(sliver, parks_polys)
            water_score = _candidate_score(sliver, water_polys)
            road_score = _candidate_score(sliver, road_cut_mask)
            park_distance = _candidate_distance(sliver, parks_polys)
            water_distance = _candidate_distance(sliver, water_polys)
            road_distance = _candidate_distance(sliver, road_cut_mask)
            road_enclosed = (
                road_influence is not None
                and sliver.difference(road_influence).is_empty
            )
            if park_score <= 0.0 and parks_polys is not None and not parks_polys.is_empty:
                try:
                    if sliver.distance(parks_polys) <= min_printable_m:
                        park_score = min_printable_m
                except Exception:
                    pass
            if water_score <= 0.0 and water_polys is not None and not water_polys.is_empty:
                try:
                    if sliver.distance(water_polys) <= min_printable_m:
                        water_score = min_printable_m * 0.9
                except Exception:
                    pass
            if road_score <= 0.0 and road_cut_mask is not None and not getattr(road_cut_mask, "is_empty", True):
                try:
                    if road_distance <= min_printable_m:
                        road_score = min_printable_m * 1.05
                except Exception:
                    pass
            if road_enclosed and park_score <= 0.0 and water_score <= 0.0:
                road_score = max(road_score, min_printable_m * 1.5)

            if park_score >= water_score and park_score >= road_score and park_score > 0:
                park_eat.append(sliver)
            elif water_score >= park_score and water_score >= road_score and water_score > 0:
                water_eat.append(sliver)
            elif road_score > 0 and road_cut_mask is not None:
                road_eat.append(sliver)
            else:
                # Keep undecided terrain by default. A meaningful relief patch
                # should not be reassigned to road groove just because it is nearby.
                nearest_candidates = []
                if road_cut_mask is not None and not getattr(road_cut_mask, "is_empty", True):
                    nearest_candidates.append(("road", road_distance))
                if parks_polys is not None and not getattr(parks_polys, "is_empty", True):
                    nearest_candidates.append(("park", park_distance))
                if water_polys is not None and not getattr(water_polys, "is_empty", True):
                    nearest_candidates.append(("water", water_distance))
                nearest_candidates = [
                    (label, distance)
                    for label, distance in nearest_candidates
                    if np.isfinite(distance)
                ]
                if nearest_candidates:
                    nearest_label, nearest_distance = min(nearest_candidates, key=lambda item: (item[1], item[0] != "road"))
                    fallback_limit = float(min_printable_m) * 0.75
                    if nearest_distance <= fallback_limit:
                        if nearest_label == "road" and road_cut_mask is not None:
                            road_eat.append(sliver)
                        elif nearest_label == "park" and parks_polys is not None:
                            park_eat.append(sliver)
                        elif nearest_label == "water" and water_polys is not None:
                            water_eat.append(sliver)
        except Exception:
            pass

    try:
        if road_eat and road_cut_mask is not None:
            road_cut_mask = road_cut_mask.union(unary_union(road_eat)).buffer(0)
            print(f"[INFO] {zone_prefix} Absorbed {len(road_eat)} terrain slivers -> road groove")
        if park_eat and parks_polys is not None:
            parks_polys = parks_polys.union(unary_union(park_eat)).buffer(0)
            print(f"[INFO] {zone_prefix} Absorbed {len(park_eat)} terrain slivers -> park groove")
        if water_eat and water_polys is not None:
            water_polys = water_polys.union(unary_union(water_eat)).buffer(0)
            print(f"[INFO] {zone_prefix} Absorbed {len(water_eat)} terrain slivers -> water groove")
    except Exception as exc:
        print(f"[WARN] {zone_prefix} sliver absorption apply failed: {exc}")

    return road_cut_mask, parks_polys, water_polys


@dataclass
class GrooveCutResult:
    terrain_mesh: Optional[trimesh.Trimesh]
    road_polygons_used: Optional[BaseGeometry]
    parks_polygons_used: Optional[BaseGeometry]
    water_polygons_used: Optional[BaseGeometry]
    boolean_backend_name: str
    grooves_expected: bool = False
    change_applied: bool = False
    rejected: bool = False
    rejection_reason: Optional[str] = None
    failure_reason: Optional[str] = None
    changed_vertices: bool = False
    volume_removed_m3: Optional[float] = None
    volume_removed_ratio: Optional[float] = None


def _mesh_component_count(mesh: Optional[trimesh.Trimesh]) -> int:
    if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
        return 0
    try:
        return len(mesh.split(only_watertight=False))
    except Exception:
        return 0


def _geometry_has_area(geometry: Optional[BaseGeometry]) -> bool:
    if geometry is None or getattr(geometry, "is_empty", True):
        return False
    try:
        return float(getattr(geometry, "area", 0.0) or 0.0) > 1e-9
    except Exception:
        return True


def _mesh_changed(
    before_mesh: Optional[trimesh.Trimesh],
    after_mesh: Optional[trimesh.Trimesh],
    *,
    atol: float = 1e-7,
) -> bool:
    if before_mesh is None or after_mesh is None:
        return before_mesh is not after_mesh

    try:
        if len(before_mesh.vertices) != len(after_mesh.vertices):
            return True
        if len(before_mesh.faces) != len(after_mesh.faces):
            return True
        if not np.array_equal(np.asarray(before_mesh.faces), np.asarray(after_mesh.faces)):
            return True
        if not np.allclose(
            np.asarray(before_mesh.vertices, dtype=float),
            np.asarray(after_mesh.vertices, dtype=float),
            atol=atol,
            rtol=0.0,
        ):
            return True
    except Exception:
        return before_mesh is not after_mesh
    return False


def _xy_bounds_delta(
    lhs: Optional[trimesh.Trimesh],
    rhs: Optional[trimesh.Trimesh],
) -> float:
    if lhs is None or rhs is None:
        return float("inf")
    try:
        lhs_bounds = np.asarray(lhs.bounds, dtype=float)[:, :2]
        rhs_bounds = np.asarray(rhs.bounds, dtype=float)[:, :2]
        return float(np.max(np.abs(lhs_bounds - rhs_bounds)))
    except Exception:
        return float("inf")


def _roundtrip_mesh_via_stl(
    mesh: Optional[trimesh.Trimesh],
) -> Optional[trimesh.Trimesh]:
    if mesh is None:
        return None

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as temp_file:
            temp_path = temp_file.name
        mesh.export(temp_path)
        roundtrip_mesh = trimesh.load(temp_path, file_type="stl", force="mesh")
        if roundtrip_mesh is None or len(roundtrip_mesh.vertices) == 0:
            return None
        try:
            roundtrip_mesh.merge_vertices()
            roundtrip_mesh.update_faces(roundtrip_mesh.unique_faces())
            roundtrip_mesh.update_faces(roundtrip_mesh.nondegenerate_faces())
            roundtrip_mesh.remove_unreferenced_vertices()
        except Exception:
            pass
        return roundtrip_mesh
    except Exception:
        return None
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except Exception:
                pass


def _dominant_watertight_component(
    mesh: Optional[trimesh.Trimesh],
    *,
    original_mesh: Optional[trimesh.Trimesh],
    scale_factor: Optional[float],
    label: str,
    source_label: str,
) -> Optional[trimesh.Trimesh]:
    if mesh is None or len(mesh.vertices) == 0:
        return None
    try:
        components = list(mesh.split(only_watertight=False))
    except Exception:
        return None
    if not components:
        return None

    largest = max(components, key=lambda item: len(item.faces) if item is not None and item.faces is not None else 0)
    total_faces = sum(len(item.faces) for item in components if item is not None and item.faces is not None)
    largest_faces = len(largest.faces) if largest.faces is not None else 0
    extra_ratio = float(total_faces - largest_faces) / float(total_faces) if total_faces > 0 else 1.0
    xy_drift = _xy_bounds_delta(original_mesh, largest)
    xy_drift_limit = max(model_mm_to_world_m(0.10, scale_factor), 1e-3) if scale_factor and scale_factor > 0 else 1e-3

    if (
        largest is not None
        and largest_faces > 0
        and bool(getattr(largest, "is_watertight", False))
        and extra_ratio <= 0.02
        and xy_drift <= xy_drift_limit
    ):
        print(
            f"[GROOVE] {label}Stabilized groove result from {source_label} dominant component "
            f"(extra_ratio={extra_ratio:.4f}, xy_drift={xy_drift:.6f}m)"
        )
        return largest
    return None


def _stabilize_groove_result_mesh(
    candidate_mesh: Optional[trimesh.Trimesh],
    *,
    original_mesh: Optional[trimesh.Trimesh],
    scale_factor: Optional[float],
    label: str,
    allow_non_watertight_fallback: bool = False,
) -> Optional[trimesh.Trimesh]:
    if candidate_mesh is None:
        return None

    if bool(getattr(candidate_mesh, "is_watertight", False)):
        return candidate_mesh

    dominant_raw = _dominant_watertight_component(
        candidate_mesh,
        original_mesh=original_mesh,
        scale_factor=scale_factor,
        label=label,
        source_label="raw",
    )
    if dominant_raw is not None:
        return dominant_raw

    cleaned_mesh = candidate_mesh
    try:
        cleaned_mesh = candidate_mesh.copy()
        cleaned_mesh.merge_vertices()
        cleaned_mesh.update_faces(cleaned_mesh.unique_faces())
        cleaned_mesh.update_faces(cleaned_mesh.nondegenerate_faces())
        cleaned_mesh.remove_unreferenced_vertices()
    except Exception:
        cleaned_mesh = candidate_mesh

    if bool(getattr(cleaned_mesh, "is_watertight", False)):
        return cleaned_mesh

    dominant_cleaned = _dominant_watertight_component(
        cleaned_mesh,
        original_mesh=original_mesh,
        scale_factor=scale_factor,
        label=label,
        source_label="cleaned",
    )
    if dominant_cleaned is not None:
        return dominant_cleaned

    roundtrip_mesh = _roundtrip_mesh_via_stl(cleaned_mesh)
    if roundtrip_mesh is not None and bool(getattr(roundtrip_mesh, "is_watertight", False)):
        return roundtrip_mesh
    dominant_roundtrip = _dominant_watertight_component(
        roundtrip_mesh,
        original_mesh=original_mesh,
        scale_factor=scale_factor,
        label=label,
        source_label="roundtrip",
    )
    if dominant_roundtrip is not None:
        return dominant_roundtrip

    # Final fallback candidate: dominant component even when non-watertight.
    # By default we reject this if reference mesh exists (to avoid open-boundary
    # base artifacts). For sequential fallback path we can opt in and accept it
    # when geometry drift is tiny and terrain actually changed.
    try:
        components = list(cleaned_mesh.split(only_watertight=False))
    except Exception:
        components = []
    if components:
        dominant = max(
            [c for c in components if c is not None and c.faces is not None and len(c.faces) > 0],
            key=lambda item: len(item.faces),
            default=None,
        )
        if dominant is not None:
            total_faces = sum(len(c.faces) for c in components if c is not None and c.faces is not None)
            dominant_faces = len(dominant.faces) if dominant.faces is not None else 0
            extra_ratio = float(total_faces - dominant_faces) / float(total_faces) if total_faces > 0 else 1.0
            xy_drift = _xy_bounds_delta(original_mesh, dominant)
            xy_drift_limit = max(model_mm_to_world_m(0.10, scale_factor), 1e-3) if scale_factor and scale_factor > 0 else 1e-3
            if dominant_faces > 0 and extra_ratio <= 0.01 and xy_drift <= xy_drift_limit:
                if original_mesh is None:
                    print(
                        f"[GROOVE] {label}Stabilized groove result from dominant component "
                        f"(non-watertight fallback, no reference mesh; extra_ratio={extra_ratio:.4f}, "
                        f"xy_drift={xy_drift:.6f}m)"
                    )
                    return dominant
                if allow_non_watertight_fallback and _mesh_changed(original_mesh, dominant):
                    print(
                        f"[WARN] {label}Accepted non-watertight dominant groove fallback "
                        f"(sequential mode, extra_ratio={extra_ratio:.4f}, xy_drift={xy_drift:.6f}m)"
                    )
                    return dominant
                print(
                    f"[WARN] {label}Rejected non-watertight dominant groove fallback; "
                    f"preserving pre-groove terrain for deterministic sequential fallback"
                )
                return original_mesh

    if original_mesh is not None:
        if allow_non_watertight_fallback and _mesh_changed(original_mesh, cleaned_mesh):
            try:
                xy_drift = _xy_bounds_delta(original_mesh, cleaned_mesh)
            except Exception:
                xy_drift = float("inf")
            xy_drift_limit = max(model_mm_to_world_m(0.10, scale_factor), 1e-3) if scale_factor and scale_factor > 0 else 1e-3
            if xy_drift <= xy_drift_limit:
                print(
                    f"[WARN] {label}Accepted non-watertight groove fallback mesh "
                    f"(sequential mode, xy_drift={xy_drift:.6f}m)"
                )
                return cleaned_mesh
        print(
            f"[WARN] {label}Groove result remained non-watertight after stabilization; "
            f"using pre-groove terrain to force deterministic fallback"
        )
        return original_mesh
    return cleaned_mesh


def _bounds_delta_xyz(lhs: Optional[trimesh.Trimesh], rhs: Optional[trimesh.Trimesh]) -> tuple[float, float]:
    if lhs is None or rhs is None:
        return float("inf"), float("inf")
    try:
        lhs_bounds = np.asarray(lhs.bounds, dtype=float)
        rhs_bounds = np.asarray(rhs.bounds, dtype=float)
        xy_delta = float(np.max(np.abs(lhs_bounds[:, :2] - rhs_bounds[:, :2])))
        z_delta = float(np.max(np.abs(lhs_bounds[:, 2] - rhs_bounds[:, 2])))
        if not np.isfinite(xy_delta) or not np.isfinite(z_delta):
            return float("inf"), float("inf")
        return xy_delta, z_delta
    except Exception:
        return float("inf"), float("inf")


def prepare_road_cut_mask(
    merged_roads_geom_local: Optional[BaseGeometry],
    building_union_local: Optional[BaseGeometry],
    scale_factor: Optional[float],
    groove_clearance_mm: float,
    building_clearance_mm: float = 0.2,
    zone_polygon_local: Optional[BaseGeometry] = None,
    min_printable_mm: float = MICRO_REGION_THRESHOLD_MM,
    road_gap_fill_threshold_mm: float = 0.45,
    zone_prefix: str = "",
) -> Optional[BaseGeometry]:
    road_cut_mask = None
    merged_roads_for_cutting = merged_roads_geom_local
    building_mask = None

    if (
        merged_roads_for_cutting is not None
        and building_union_local is not None
        and not getattr(building_union_local, "is_empty", True)
    ):
        try:
            # IMPORTANT: subtract the BARE building footprint only (no clearance
            # buffer). The road MESH, built from canonical_mask_bundle.roads_final,
            # also reaches the bare wall in faithful mode. If the terrain cut is
            # pulled back by `building_clearance_mm` while the road mesh extends
            # to the wall, the strip between (≈1 m world at scale=0.2) prints as a
            # raised tan "wing" next to every building (user report 2026-04-19).
            # `building_clearance_mm` is preserved as a parameter for callers that
            # still need the old behaviour, but the default subtraction uses only
            # the bare footprint so terrain and road edges line up at the wall.
            building_mask = building_union_local
            try:
                building_mask = building_mask.buffer(0)
            except Exception:
                pass
            try:
                merged_roads_for_cutting = merged_roads_for_cutting.buffer(0)
            except Exception:
                pass
            try:
                merged_roads_for_cutting = merged_roads_for_cutting.difference(building_mask)
            except Exception:
                merged_roads_for_cutting = merged_roads_for_cutting.buffer(0).difference(
                    building_mask.buffer(0)
                )
            if merged_roads_for_cutting.is_empty:
                merged_roads_for_cutting = None
        except Exception as exc:
            print(f"[WARN] {zone_prefix} Failed to subtract buildings from road_cut_mask: {exc}")

    if merged_roads_for_cutting is None:
        print(f"[DEBUG] {zone_prefix} merged_roads_geom_local is None, cannot create road_cut_mask")
        return None
    if not hasattr(merged_roads_for_cutting, "is_empty"):
        print(
            f"[DEBUG] {zone_prefix} merged_roads_geom_local has no is_empty attribute, "
            f"type: {type(merged_roads_for_cutting)}"
        )
        return None
    if merged_roads_for_cutting.is_empty:
        print(f"[DEBUG] {zone_prefix} merged_roads_geom_local is empty, cannot create road_cut_mask")
        return None

    try:
        road_clearance_m = groove_clearance_mm / float(scale_factor) if scale_factor and scale_factor > 0 else 0.0
        print(
            f"[DEBUG] {zone_prefix} Creating road_cut_mask: clearance_m={road_clearance_m:.4f}, "
            f"scale_factor={scale_factor}, groove_clearance_mm={groove_clearance_mm}"
        )

        if road_clearance_m > 0:
            road_cut_mask = merged_roads_for_cutting.buffer(road_clearance_m, join_style=2)
            print(
                f"[INFO] {zone_prefix} Road cut mask created with clearance: "
                f"{road_clearance_m:.4f}m ({groove_clearance_mm}mm), area: {road_cut_mask.area:.2f} m2"
            )
        else:
            road_cut_mask = merged_roads_for_cutting
            print(
                f"[INFO] {zone_prefix} Road cut mask created without clearance "
                f"(exact road boundaries), area: {road_cut_mask.area:.2f} m2"
            )
        # DO NOT subtract buildings from road_cut_mask (groove mask).
        # The groove must extend groove_clearance INTO building footprints so a
        # visible channel exists between the road insert edge and the building wall.
        # If we subtract buildings here, the clearance buffer at the road–building
        # boundary is fully erased and no groove gap is printed.
        # The building mesh is clipped by building_exclusion_mask (= this groove mask)
        # in building_processor, so buildings remain correctly inset from the road.
        if road_cut_mask is not None and not getattr(road_cut_mask, "is_empty", True):
            try:
                road_cut_mask = road_cut_mask.buffer(0)
            except Exception:
                pass
    except Exception as exc:
        print(f"[WARN] {zone_prefix} Failed to create road_cut_mask: {exc}")
        import traceback

        traceback.print_exc()
        return None

    if road_cut_mask is None or getattr(road_cut_mask, "is_empty", True):
        return road_cut_mask

    # --- Thin terrain sliver removal ---
    # After adding groove clearance, narrow terrain strips (<min_printable_mm) can remain
    # between adjacent road grooves. They won't print and appear as hanging wedges.
    # Fix in two steps:
    # 1. Close sub-min_printable gaps directly in the groove mask (morphological closing).
    # 2. If zone polygon is available: erode the remaining terrain and absorb slivers
    #    that are adjacent to roads (nearest-material rule: road wins).
    if scale_factor and scale_factor > 0 and min_printable_mm > 0:
        real_min_m = model_mm_to_world_m(min_printable_mm, scale_factor)
        groove_gap_close_m = float(real_min_m)

        # Groove cleanup must be much stricter than canonical road-mask cleanup.
        # The road gap-fill threshold is appropriate for road topology, but it is
        # too aggressive for terrain grooves and can swallow legitimate relief
        # islands. Here we only close gaps below the actual printable threshold.
        try:
            road_cut_mask = merge_close_road_gaps(
                road_cut_mask,
                min_gap_m=float(groove_gap_close_m),
            )
            print(
                f"[INFO] {zone_prefix} Road groove: closed terrain gaps < "
                f"{float(min_printable_mm):.3f}mm"
            )
        except Exception as exc:
            print(f"[WARN] {zone_prefix} groove gap-close failed: {exc}")

        # Step 2: fill interior road polygon holes that are too narrow to print (-> road)
        # Handles the case of two parallel roads with a thin enclosed terrain island between them.
        try:
            road_cut_mask = _fill_narrow_road_holes(
                road_cut_mask,
                real_min_m,
                preserve_geom=building_mask,
                zone_prefix=zone_prefix,
            )
        except Exception as exc:
            print(f"[DEBUG] {zone_prefix} road hole-fill skipped: {exc}")

        # Cleanup passes can erase the groove ring on the inner side of larger
        # road loops. Restore that clearance explicitly from the canonical road
        # holes so ring roads keep a groove on both sides.
        try:
            inner_hole_clearance = _restore_inner_hole_clearance(
                merged_roads_for_cutting,
                clearance_m=float(road_clearance_m),
                preserve_geom=building_mask,
                zone_prefix=zone_prefix,
            )
            if inner_hole_clearance is not None and not getattr(inner_hole_clearance, "is_empty", True):
                road_cut_mask = road_cut_mask.union(inner_hole_clearance).buffer(0)
        except Exception as exc:
            print(f"[DEBUG] {zone_prefix} inner hole groove restore skipped: {exc}")

    # DO NOT subtract buildings from groove mask here.
    # The groove must extend groove_clearance INTO building footprints so a
    # visible channel exists between road insert edge and building wall.
    # Building meshes are clipped by building_exclusion_mask (= this groove_mask)
    # in building_processor, so they remain correctly inset from the road.

    return road_cut_mask


def cut_inlay_grooves(
    terrain_mesh: Optional[trimesh.Trimesh],
    road_mesh: Optional[trimesh.Trimesh],
    parks_mesh: Optional[trimesh.Trimesh],
    water_mesh: Optional[trimesh.Trimesh],
    road_cut_mask: Optional[BaseGeometry],
    merged_roads_geom_local: Optional[BaseGeometry],
    parks_polygons: Optional[BaseGeometry],
    water_polygons: Optional[BaseGeometry],
    building_polygons: Optional[BaseGeometry],
    scale_factor: Optional[float],
    groove_clearance_mm: float,
    road_embed_m: float,
    parks_embed_mm: float,
    water_depth_m: float,
    boolean_backend: Optional[BooleanBackend] = None,
    zone_prefix: str = "",
    zone_polygon_local: Optional[BaseGeometry] = None,
    min_printable_mm: float = MICRO_REGION_THRESHOLD_MM,
    parks_groove_override: Optional[BaseGeometry] = None,
    water_groove_override: Optional[BaseGeometry] = None,
    use_exact_masks: bool = False,
) -> GrooveCutResult:
    has_road_grooves = terrain_mesh is not None and road_mesh is not None and scale_factor and scale_factor > 0
    has_park_grooves = terrain_mesh is not None and parks_mesh is not None and scale_factor and scale_factor > 0
    has_water_grooves = terrain_mesh is not None and water_mesh is not None and scale_factor and scale_factor > 0

    if not has_road_grooves and not has_park_grooves and not has_water_grooves:
        return GrooveCutResult(
            terrain_mesh=terrain_mesh,
            road_polygons_used=road_cut_mask,
            parks_polygons_used=None,
            water_polygons_used=None,
            boolean_backend_name="skipped",
            grooves_expected=False,
            change_applied=False,
        )

    groove_clearance_m = groove_clearance_mm / float(scale_factor)
    boundary_snap_m = 0.0
    try:
        boundary_snap_m = min(
            model_mm_to_world_m(0.04, scale_factor),
            float(groove_clearance_m) * 0.15,
        )
    except Exception:
        boundary_snap_m = float(groove_clearance_m) * 0.15
    boundary_snap_m = max(float(boundary_snap_m), 0.0)

    road_polys_for_groove = None
    if has_road_grooves:
        road_polys_for_groove = road_cut_mask if road_cut_mask is not None else merged_roads_geom_local
        if road_polys_for_groove is None or (
            hasattr(road_polys_for_groove, "is_empty") and road_polys_for_groove.is_empty
        ):
            print(
                f"[GROOVE] WARNING: road_cut_mask={road_cut_mask is not None}, "
                f"merged_roads_geom_local={merged_roads_geom_local is not None} - no road polygons for groove!"
            )
            road_polys_for_groove = None

    parks_polys_for_cutting = None
    if has_park_grooves and parks_polygons is not None:
        if use_exact_masks and parks_groove_override is not None and not getattr(parks_groove_override, "is_empty", True):
            try:
                road_mask_for_park_cut = road_cut_mask if road_cut_mask is not None else merged_roads_geom_local
                parks_polys_for_cutting = _prepare_parks_groove_mask(
                    parks_polygons,
                    road_groove_mask=road_mask_for_park_cut,
                    water_polygons=water_polygons,
                    building_polygons=building_polygons,
                    groove_clearance_m=float(groove_clearance_m),
                    boundary_snap_m=float(boundary_snap_m),
                    zone_prefix=zone_prefix,
                )
            except Exception as exc:
                print(f"[DEBUG] {zone_prefix} exact park groove rebuild skipped: {exc}")
                parks_polys_for_cutting = parks_groove_override
        else:
            road_mask_for_park_cut = road_cut_mask if road_cut_mask is not None else merged_roads_geom_local
            parks_polys_for_cutting = _prepare_parks_groove_mask(
                parks_polygons,
                road_groove_mask=road_mask_for_park_cut,
                water_polygons=water_polygons,
                building_polygons=building_polygons,
                groove_clearance_m=float(groove_clearance_m),
                boundary_snap_m=float(boundary_snap_m),
                zone_prefix=zone_prefix,
            )

    water_polys_for_cutting = None
    if has_water_grooves and water_polygons is not None:
        if use_exact_masks and water_groove_override is not None and not getattr(water_groove_override, "is_empty", True):
            try:
                water_polys_for_cutting = _prepare_water_groove_mask(
                    water_polygons,
                    road_groove_mask=road_polys_for_groove,
                    parks_groove_mask=parks_polys_for_cutting,
                    building_polygons=building_polygons,
                    groove_clearance_m=float(groove_clearance_m),
                    boundary_snap_m=float(boundary_snap_m),
                    zone_prefix=zone_prefix,
                )
            except Exception as exc:
                print(f"[DEBUG] {zone_prefix} exact water groove rebuild skipped: {exc}")
                water_polys_for_cutting = water_groove_override
        else:
            water_polys_for_cutting = _prepare_water_groove_mask(
                water_polygons,
                road_groove_mask=road_polys_for_groove,
                parks_groove_mask=parks_polys_for_cutting,
                building_polygons=building_polygons,
                groove_clearance_m=float(groove_clearance_m),
                boundary_snap_m=float(boundary_snap_m),
                zone_prefix=zone_prefix,
            )

    # Absorb thin terrain slivers (that won't print) into the nearest groove.
    # Must happen after all groove polygons are finalised and before mesh cutting.
    if (
        not use_exact_masks
        and
        zone_polygon_local is not None
        and not getattr(zone_polygon_local, "is_empty", True)
        and scale_factor
        and scale_factor > 0
        and min_printable_mm > 0
    ):
        try:
            real_min_m = model_mm_to_world_m(min_printable_mm, scale_factor)
            road_polys_for_groove, parks_polys_for_cutting, water_polys_for_cutting = (
                _absorb_terrain_slivers(
                    zone_polygon=zone_polygon_local,
                    road_cut_mask=road_polys_for_groove,
                    parks_polys=parks_polys_for_cutting,
                    water_polys=water_polys_for_cutting,
                    min_printable_m=real_min_m,
                    protected_voids=building_polygons,
                    zone_prefix=zone_prefix,
                )
            )
        except Exception as exc:
            print(f"[DEBUG] {zone_prefix} terrain sliver absorption skipped: {exc}")

    if (
        not use_exact_masks
        and road_polys_for_groove is not None
        and parks_polys_for_cutting is not None
        and not getattr(parks_polys_for_cutting, "is_empty", True)
    ):
        try:
            road_polys_for_groove = road_polys_for_groove.difference(parks_polys_for_cutting)
            road_polys_for_groove = road_polys_for_groove.buffer(0)
        except Exception:
            pass
    if (
        not use_exact_masks
        and parks_polys_for_cutting is not None
        and road_polys_for_groove is not None
        and not getattr(road_polys_for_groove, "is_empty", True)
    ):
        try:
            parks_polys_for_cutting = parks_polys_for_cutting.difference(road_polys_for_groove)
            parks_polys_for_cutting = parks_polys_for_cutting.buffer(0)
        except Exception:
            pass
    if not use_exact_masks:
        parks_polys_for_cutting = _apply_exclusion_mask(parks_polys_for_cutting, building_polygons)
    if (
        not use_exact_masks
        and water_polys_for_cutting is not None
        and road_polys_for_groove is not None
        and not getattr(road_polys_for_groove, "is_empty", True)
    ):
        try:
            water_polys_for_cutting = water_polys_for_cutting.difference(road_polys_for_groove)
            water_polys_for_cutting = water_polys_for_cutting.buffer(0)
        except Exception:
            pass

    if not use_exact_masks:
        micro_region_m = model_mm_to_world_m(min_printable_mm, scale_factor)
        road_polys_for_groove = _drop_tiny_material_fragments(
            road_polys_for_groove,
            min_feature_m=float(micro_region_m),
        )
        parks_polys_for_cutting = _drop_tiny_material_fragments(
            parks_polys_for_cutting,
            min_feature_m=float(micro_region_m),
        )
    grooves_expected = any(
        _geometry_has_area(geometry)
        for geometry in (road_polys_for_groove, parks_polys_for_cutting, water_polys_for_cutting)
    )
    if not grooves_expected:
        print(f"[GROOVE] {zone_prefix}No canonical groove masks survived cleanup; skipping boolean cut")
        return GrooveCutResult(
            terrain_mesh=terrain_mesh,
            road_polygons_used=road_polys_for_groove,
            parks_polygons_used=parks_polys_for_cutting,
            water_polygons_used=water_polys_for_cutting,
            boolean_backend_name="skipped:no_masks",
            grooves_expected=False,
            change_applied=False,
        )

    print("[GROOVE] === Unified groove cutting ===")
    print(
        f"[GROOVE] Terrain: {len(terrain_mesh.vertices)} verts, "
        f"Z=[{terrain_mesh.bounds[0][2]:.4f}, {terrain_mesh.bounds[1][2]:.4f}]"
    )
    if road_polys_for_groove is not None:
        print(f"[GROOVE] Road polygons: area={road_polys_for_groove.area:.2f} m2")
    if parks_polys_for_cutting is not None:
        print(f"[GROOVE] Park polygons: area={parks_polys_for_cutting.area:.2f} m2")
    if water_polys_for_cutting is not None:
        print(f"[GROOVE] Water polygons: area={water_polys_for_cutting.area:.2f} m2")
    print(f"[GROOVE] Clearance: {groove_clearance_m:.4f}m ({groove_clearance_mm}mm)")

    terrain_before_vol = None
    terrain_before_mesh = terrain_mesh.copy() if terrain_mesh is not None else None
    terrain_before_components = _mesh_component_count(terrain_mesh)
    terrain_before_watertight = bool(getattr(terrain_mesh, "is_watertight", False)) if terrain_mesh is not None else False
    try:
        terrain_before_vol = float(terrain_mesh.volume)
    except Exception:
        pass

    max_embed_m = max(
        road_embed_m if road_embed_m else 0,
        (float(parks_embed_mm) / float(scale_factor)) if scale_factor and scale_factor > 0 else 0,
        float(water_depth_m) if water_depth_m else 0,
    )
    groove_depth_m = max_embed_m * 1.5 if max_embed_m > 0 else 1.0

    backend = resolve_boolean_backend(boolean_backend)
    backend_name = getattr(backend, "name", backend.__class__.__name__)
    print(f"[GROOVE] Boolean backend: {backend_name}")
    terrain_mesh = backend.cut_grooves(
        GrooveBooleanRequest(
            terrain_mesh=terrain_mesh,
            road_polygons=road_polys_for_groove,
            road_clearance_m=0.0,
            parks_polygons=parks_polys_for_cutting,
            parks_clearance_m=0.0,
            parks_mesh=parks_mesh,
            water_polygons=water_polys_for_cutting,
            water_clearance_m=0.0,
            water_mesh=water_mesh,
            scale_factor=float(scale_factor),
            road_mesh=road_mesh,
            groove_depth_m=groove_depth_m,
        )
    )
    if terrain_mesh is not None:
        try:
            terrain_mesh = _stabilize_groove_result_mesh(
                terrain_mesh,
                original_mesh=terrain_before_mesh,
                scale_factor=float(scale_factor),
                label=zone_prefix,
                # Primary manifold pass can return a topologically non-watertight
                # but spatially stable dominant component. Accepting it here
                # avoids dropping into sequential Blender fallback, which may
                # occasionally produce axis-drifted no-op outcomes.
                allow_non_watertight_fallback=True,
            )
        except Exception as exc:
            print(f"[DEBUG] {zone_prefix} terrain groove stabilization skipped: {exc}")

    # Hard spatial invariant: groove boolean must not teleport terrain bounds.
    # If a backend returns a shifted mesh (axis/units mismatch), keep previous.
    if terrain_mesh is not None and terrain_before_mesh is not None:
        try:
            xy_delta_m, z_delta_m = _bounds_delta_xyz(terrain_before_mesh, terrain_mesh)
            xy_drift_limit = max(model_mm_to_world_m(0.5, scale_factor), 0.75) if scale_factor and scale_factor > 0 else 0.75
            z_drift_limit = max(model_mm_to_world_m(1.0, scale_factor), 2.0) if scale_factor and scale_factor > 0 else 2.0
            if xy_delta_m > xy_drift_limit or z_delta_m > z_drift_limit:
                print(
                    f"[WARN] {zone_prefix} Rejecting groove result: bounds drift too large "
                    f"(xy_delta={xy_delta_m:.3f}m > {xy_drift_limit:.3f}m, "
                    f"z_delta={z_delta_m:.3f}m > {z_drift_limit:.3f}m)"
                )
                terrain_mesh = terrain_before_mesh
                backend_name = f"{backend_name}:rejected_drift"
        except Exception as exc:
            print(f"[DEBUG] {zone_prefix} groove drift check skipped: {exc}")

    reject_fragmented_result = False
    rejection_reason = None
    if terrain_mesh is not None:
        try:
            terrain_after_components = _mesh_component_count(terrain_mesh)
            terrain_after_watertight = bool(getattr(terrain_mesh, "is_watertight", False))
            component_limit = max(50, terrain_before_components * 25 if terrain_before_components > 0 else 50)
            if terrain_before_watertight and (not terrain_after_watertight) and terrain_after_components > 1:
                try:
                    rescue_components = list(terrain_mesh.split(only_watertight=False))
                except Exception:
                    rescue_components = []
                if rescue_components:
                    rescue_main = max(rescue_components, key=lambda mesh: len(mesh.faces))
                    rescue_total_faces = sum(len(mesh.faces) for mesh in rescue_components)
                    rescue_extra_ratio = (
                        float(rescue_total_faces - len(rescue_main.faces)) / float(rescue_total_faces)
                        if rescue_total_faces > 0
                        else 1.0
                    )
                    rescue_xy_drift = _xy_bounds_delta(terrain_before_mesh, rescue_main)
                    rescue_xy_drift_limit = max(model_mm_to_world_m(0.10, scale_factor), 1e-3)
                    if (
                        bool(getattr(rescue_main, "is_watertight", False))
                        and rescue_extra_ratio <= 0.02
                        and rescue_xy_drift <= rescue_xy_drift_limit
                    ):
                        terrain_mesh = rescue_main
                        terrain_after_components = 1
                        terrain_after_watertight = True
                        print(
                            f"[GROOVE] {zone_prefix}Recovered groove result from dominant watertight component "
                            f"(extra_ratio={rescue_extra_ratio:.4f}, xy_drift={rescue_xy_drift:.6f}m)"
                        )
            if terrain_before_watertight and (not terrain_after_watertight) and terrain_after_components > component_limit:
                reject_fragmented_result = True
                rejection_reason = "rejected_fragmented"
                print(
                    f"[WARN] {zone_prefix} Rejecting groove result: terrain fragmented from "
                    f"{terrain_before_components} to {terrain_after_components} components "
                    f"and lost watertightness"
                )
        except Exception as exc:
            print(f"[DEBUG] {zone_prefix} groove sanity check skipped: {exc}")

    if reject_fragmented_result:
        terrain_mesh = terrain_before_mesh
        backend_name = f"{backend_name}:{rejection_reason}"

    # If canonical groove masks exist but backend produced no terrain change,
    # retry once with sequential cutter path (road -> parks -> water). This is
    # slower but much more stable for pathological booleans.
    if grooves_expected and terrain_before_mesh is not None and not _mesh_changed(terrain_before_mesh, terrain_mesh):
        try:
            fallback_mesh = cut_grooves_sequentially(
                terrain_mesh=terrain_before_mesh.copy(),
                road_polygons=road_polys_for_groove,
                road_clearance_m=0.0,
                parks_polygons=parks_polys_for_cutting,
                parks_clearance_m=0.0,
                parks_mesh=parks_mesh,
                water_polygons=water_polys_for_cutting,
                water_clearance_m=0.0,
                water_mesh=water_mesh,
                scale_factor=float(scale_factor),
                road_mesh=road_mesh,
                groove_depth_m=groove_depth_m,
            )
            if fallback_mesh is not None:
                fallback_mesh = _stabilize_groove_result_mesh(
                    fallback_mesh,
                    original_mesh=terrain_before_mesh,
                    scale_factor=float(scale_factor),
                    label=zone_prefix,
                    allow_non_watertight_fallback=True,
                )
                xy_delta_m, z_delta_m = _bounds_delta_xyz(terrain_before_mesh, fallback_mesh)
                xy_drift_limit = max(model_mm_to_world_m(0.5, scale_factor), 0.75) if scale_factor and scale_factor > 0 else 0.75
                z_drift_limit = max(model_mm_to_world_m(1.0, scale_factor), 2.0) if scale_factor and scale_factor > 0 else 2.0
                if xy_delta_m <= xy_drift_limit and z_delta_m <= z_drift_limit and _mesh_changed(terrain_before_mesh, fallback_mesh):
                    terrain_mesh = fallback_mesh
                    backend_name = f"{backend_name}:fallback_seq"
                    print(
                        f"[GROOVE] {zone_prefix}Applied sequential groove fallback "
                        f"(xy_delta={xy_delta_m:.3f}m, z_delta={z_delta_m:.3f}m)"
                    )
                else:
                    print(
                        f"[WARN] {zone_prefix}Sequential groove fallback rejected "
                        f"(changed={_mesh_changed(terrain_before_mesh, fallback_mesh)}, "
                        f"xy_delta={xy_delta_m:.3f}m, z_delta={z_delta_m:.3f}m)"
                    )
        except Exception as exc:
            print(f"[DEBUG] {zone_prefix} sequential groove fallback failed: {exc}")

    changed_vertices = _mesh_changed(terrain_before_mesh, terrain_mesh)
    volume_removed_m3 = None
    volume_removed_ratio = None
    failure_reason = None
    if terrain_mesh is not None:
        print(
            f"[GROOVE] Terrain AFTER grooves: {len(terrain_mesh.vertices)} verts, "
            f"Z=[{terrain_mesh.bounds[0][2]:.4f}, {terrain_mesh.bounds[1][2]:.4f}]"
        )
        try:
            terrain_after_vol = float(terrain_mesh.volume)
            if terrain_before_vol is not None:
                volume_removed_m3 = float(terrain_before_vol - terrain_after_vol)
            if terrain_before_vol and terrain_before_vol > 0:
                volume_removed_ratio = float((terrain_before_vol - terrain_after_vol) / terrain_before_vol)
                removed_pct = volume_removed_ratio * 100
                print(f"[GROOVE] Volume removed: {removed_pct:.1f}%")
        except Exception:
            pass
    else:
        print(f"[WARN] {zone_prefix} Groove cutting returned None")
        failure_reason = "boolean_returned_none"

    change_applied = bool(terrain_mesh is not None and changed_vertices and not reject_fragmented_result)
    if grooves_expected and not change_applied and failure_reason is None:
        if reject_fragmented_result:
            failure_reason = rejection_reason or "rejected_fragmented"
        elif terrain_mesh is None:
            failure_reason = "boolean_returned_none"
        else:
            failure_reason = "boolean_noop"
            print(f"[WARN] {zone_prefix} Groove masks existed but terrain did not change after boolean cut")

    return GrooveCutResult(
        terrain_mesh=terrain_mesh,
        road_polygons_used=road_polys_for_groove,
        parks_polygons_used=parks_polys_for_cutting,
        water_polygons_used=water_polys_for_cutting,
        boolean_backend_name=backend_name,
        grooves_expected=grooves_expected,
        change_applied=change_applied,
        rejected=reject_fragmented_result,
        rejection_reason=rejection_reason,
        failure_reason=failure_reason,
        changed_vertices=changed_vertices,
        volume_removed_m3=volume_removed_m3,
        volume_removed_ratio=volume_removed_ratio,
    )
