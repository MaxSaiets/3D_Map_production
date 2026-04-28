from __future__ import annotations

from typing import Any, Optional

import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from services.detail_layer_utils import MICRO_REGION_THRESHOLD_MM, model_mm_to_world_m
from services.processing_results import WaterLayerResult
from services.water_processor import process_water_surface


def _is_printable_water_mesh(mesh: Any, *, min_face_count: int = 12) -> bool:
    """Return True if mesh has enough substance to slice cleanly.

    Why: a collapsed/degenerate water polygon yields a mesh with zero volume
    and non-manifold edges — PrusaSlicer rejects it (`slicer:water:slice_failed`).
    Catch it before export so the pipeline can drop water gracefully.
    """
    if mesh is None:
        return False
    faces = getattr(mesh, "faces", None)
    if faces is None or len(faces) < int(min_face_count):
        return False
    try:
        volume = float(getattr(mesh, "volume", 0.0) or 0.0)
    except Exception:
        volume = 0.0
    if not (volume == volume) or volume <= 1e-9:  # NaN-safe + tiny volume guard
        return False
    try:
        bounds = mesh.bounds
        extents = [float(bounds[1][i] - bounds[0][i]) for i in range(3)]
    except Exception:
        return False
    if any(e != e for e in extents) or min(extents) <= 1e-6:
        return False
    return True


def _apply_water_fit_inset(
    geometry: Any,
    *,
    scale_factor: Optional[float],
    fit_clearance_mm: float,
) -> Any:
    if geometry is None or getattr(geometry, "is_empty", True):
        return geometry
    if not (scale_factor and scale_factor > 0) or fit_clearance_mm <= 0:
        return geometry
    try:
        inset_m = model_mm_to_world_m(float(fit_clearance_mm), float(scale_factor))
    except Exception:
        inset_m = 0.0
    if inset_m <= 0:
        return geometry
    try:
        inset = geometry.buffer(-float(inset_m), join_style=1).buffer(0)
    except Exception:
        return None
    if inset is None or getattr(inset, "is_empty", True):
        # Polygon collapsed under inset — water is too narrow to sit inside a
        # groove with clearance. Signal "no water" rather than returning the
        # un-inset original, which would overlap groove walls.
        return None
    return inset


def _filter_water_fragments(
    geometry: Any,
    *,
    scale_factor: Optional[float],
) -> Any:
    if geometry is None or getattr(geometry, "is_empty", True):
        return None

    min_dim_m = 0.0
    if scale_factor and scale_factor > 0:
        min_dim_m = model_mm_to_world_m(MICRO_REGION_THRESHOLD_MM, scale_factor)
    min_dim_m = max(min_dim_m, 0.05)

    min_area_m2 = max(min_dim_m * min_dim_m, 0.01)

    polys: list[Polygon] = []
    raw_geoms = []
    if isinstance(geometry, Polygon):
        raw_geoms = [geometry]
    elif isinstance(geometry, MultiPolygon):
        raw_geoms = list(geometry.geoms)
    elif hasattr(geometry, "geoms"):
        raw_geoms = [g for g in geometry.geoms if isinstance(g, Polygon)]

    for poly in raw_geoms:
        if poly is None or poly.is_empty:
            continue
        try:
            poly = poly.buffer(0)
        except Exception:
            pass
        if poly is None or poly.is_empty or getattr(poly, "area", 0.0) < min_area_m2:
            continue
        minx, miny, maxx, maxy = poly.bounds
        if min(maxx - minx, maxy - miny) < min_dim_m:
            continue
        try:
            poly = poly.simplify(min_dim_m * 0.25, preserve_topology=True)
            poly = poly.buffer(0)
        except Exception:
            pass
        if poly is not None and not poly.is_empty and getattr(poly, "area", 0.0) >= min_area_m2:
            polys.append(poly)

    if not polys:
        return None
    try:
        merged = unary_union(polys).buffer(0)
    except Exception:
        merged = polys[0] if len(polys) == 1 else MultiPolygon(polys)
    return merged if merged is not None and not getattr(merged, "is_empty", True) else None


def _prepare_water_polygons(
    gdf_water: Optional[GeoDataFrame],
    *,
    road_polygons: Any = None,
    building_polygons: Any = None,
    scale_factor: Optional[float] = None,
    fit_clearance_mm: float = 0.0,
) -> Any:
    if gdf_water is None or gdf_water.empty:
        return None

    try:
        polygons = [geom for geom in gdf_water.geometry.values if geom is not None and not geom.is_empty]
        if not polygons:
            return None
        water_union = unary_union(polygons).buffer(0)
    except Exception:
        return None

    if water_union is None or getattr(water_union, "is_empty", True):
        return None

    for mask in (road_polygons, building_polygons):
        if mask is None or getattr(mask, "is_empty", True):
            continue
        try:
            water_union = water_union.difference(mask)
            if water_union is None or getattr(water_union, "is_empty", True):
                return None
            water_union = water_union.buffer(0)
        except Exception:
            pass

    water_union = _apply_water_fit_inset(
        water_union,
        scale_factor=scale_factor,
        fit_clearance_mm=float(fit_clearance_mm),
    )
    return _filter_water_fragments(water_union, scale_factor=scale_factor)


def _build_water_mesh_input(
    gdf_water: Optional[GeoDataFrame],
    *,
    polygons: Any,
) -> Optional[GeoDataFrame]:
    if polygons is None or getattr(polygons, "is_empty", True):
        return None
    try:
        return gpd.GeoDataFrame({"geometry": [polygons]}, geometry="geometry", crs=getattr(gdf_water, "crs", None))
    except Exception:
        return None


def process_water_layer(
    *,
    task: Any,
    request: Any,
    scale_factor: Optional[float],
    terrain_provider: Any,
    global_center: Any,
    gdf_water: Optional[GeoDataFrame],
    water_depth_m: float,
    road_polygons: Any = None,
    building_polygons: Any = None,
    coordinates_already_local: bool = False,
    zone_prefix: str = "",
    water_polygons_override: Any = None,
    fit_clearance_mm: float = 0.0,
) -> WaterLayerResult:
    if not (scale_factor and scale_factor > 0 and terrain_provider is not None):
        return WaterLayerResult(mesh=None, cutting_polygons=None)

    if not getattr(request, "include_water", True):
        return WaterLayerResult(mesh=None, cutting_polygons=None)

    if water_polygons_override is not None:
        if getattr(water_polygons_override, "is_empty", True):
            return WaterLayerResult(mesh=None, cutting_polygons=None)
        # Canonical override polygons already include fit + fragment filtering
        # in the canonical 2D stage. Running _filter_water_fragments here again
        # drops small pieces that canonical chose to keep and breaks the
        # canonical 2D -> 3D handoff parity check (symdiff drift).
        task.update_status("processing", 60, "Generating water surface...")
        try:
            water_cut_polygons = water_polygons_override.buffer(0)
        except Exception:
            water_cut_polygons = water_polygons_override
    else:
        if gdf_water is None or gdf_water.empty:
            return WaterLayerResult(mesh=None, cutting_polygons=None)
        task.update_status("processing", 60, "Generating water surface...")
        water_cut_polygons = _prepare_water_polygons(
            gdf_water,
            road_polygons=road_polygons,
            building_polygons=building_polygons,
            scale_factor=scale_factor,
            fit_clearance_mm=float(fit_clearance_mm),
        )
    water_mesh_input = _build_water_mesh_input(gdf_water, polygons=water_cut_polygons)
    if water_mesh_input is None or water_mesh_input.empty:
        return WaterLayerResult(mesh=None, cutting_polygons=None)

    water_mesh = None

    if request.is_ams_mode:
        water_thickness_m = getattr(request, "water_thickness", 0.4) / scale_factor if scale_factor else 0.4
        water_mesh = process_water_surface(
            water_mesh_input,
            thickness_m=water_thickness_m,
            depth_meters=0.0,
            terrain_provider=terrain_provider,
            global_center=global_center,
            coordinates_already_local=coordinates_already_local,
            scale_factor=scale_factor,
        )
        if water_mesh:
            lift_m = (0.6 / scale_factor) if scale_factor else 0.0
            water_mesh.apply_translation([0, 0, lift_m])
            print(f"[INFO] {zone_prefix} AMS Mode: Water lifted by {lift_m:.4f}m (Target: 0.6mm level)")
    else:
        thickness_m = float(getattr(request, "water_thickness", 1.0)) / scale_factor
        water_mesh = process_water_surface(
            water_mesh_input,
            thickness_m=float(thickness_m),
            depth_meters=water_depth_m,
            terrain_provider=terrain_provider,
            global_center=global_center,
            coordinates_already_local=coordinates_already_local,
            scale_factor=scale_factor,
        )

    if water_mesh is not None and not _is_printable_water_mesh(water_mesh):
        print(
            f"[WARN] {zone_prefix} water mesh degenerate after processing — "
            f"dropping water layer to keep slicer happy"
        )
        water_mesh = None
        water_cut_polygons = None

    return WaterLayerResult(mesh=water_mesh, cutting_polygons=water_cut_polygons)
