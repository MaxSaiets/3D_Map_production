from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import osmnx as ox
from geopandas import GeoDataFrame
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry, unary_union

from services.detail_layer_utils import MICRO_REGION_THRESHOLD_MM, model_mm_to_world_m
from services.road_processor import (
    build_road_polygons,
    normalize_drivable_highway_tag,
    normalize_road_mask_for_print,
)
from services.geometry_context import looks_like_projected_meters


@dataclass
class RoadGeometryPreparationResult:
    merged_roads_geom: Optional[BaseGeometry]
    merged_roads_geom_local: Optional[BaseGeometry]
    merged_roads_geom_local_raw: Optional[BaseGeometry]
    semantic_centerlines_local: Optional[BaseGeometry] = None


def _build_local_road_edges_subset(
    *,
    G_roads: Any,
    global_center: Any,
    zone_polygon_local: Optional[BaseGeometry],
    zone_prefix: str,
) -> Optional[GeoDataFrame]:
    if G_roads is None or global_center is None or zone_polygon_local is None or getattr(zone_polygon_local, "is_empty", True):
        return None

    try:
        if isinstance(G_roads, GeoDataFrame):
            gdf_edges = G_roads.copy()
        else:
            if not hasattr(G_roads, "edges") or len(G_roads.edges) == 0:
                return None
            gdf_edges = ox.graph_to_gdfs(G_roads, nodes=False)
        if gdf_edges is None or gdf_edges.empty:
            return None

        def to_local(x, y, z=None):
            # Robust CRS normalization:
            # - projected meters (UTM-like): subtract global center directly
            # - lon/lat (WGS84): convert to UTM first, then to local
            if abs(float(x)) <= 180.0 and abs(float(y)) <= 90.0:
                x_utm, y_utm = global_center.to_utm(float(x), float(y))
                x_local, y_local = global_center.to_local(x_utm, y_utm)
            else:
                x_local, y_local = global_center.to_local(float(x), float(y))
            return (x_local, y_local) if z is None else (x_local, y_local, z)

        gdf_local = gdf_edges.copy()
        sample_geom = gdf_local.geometry.iloc[0] if len(gdf_local) else None
        if sample_geom is not None and not getattr(sample_geom, "is_empty", True):
            if looks_like_projected_meters(sample_geom):
                gdf_local["geometry"] = gdf_local.geometry.apply(lambda geom: transform_geometry(to_local, geom))
            else:
                # If data appears unprojected (lon/lat), still normalize to local.
                gdf_local["geometry"] = gdf_local.geometry.apply(lambda geom: transform_geometry(to_local, geom))
        else:
            gdf_local["geometry"] = gdf_local.geometry.apply(lambda geom: transform_geometry(to_local, geom))

        minx, miny, maxx, maxy = zone_polygon_local.bounds
        # Keep a generous apron around the target zone so buffered roads that land
        # on the border still have their correct shape before the final polygon clip.
        clip_window = box(minx - 40.0, miny - 40.0, maxx + 40.0, maxy + 40.0)
        gdf_local = gdf_local[gdf_local.geometry.intersects(clip_window)].copy()
        if gdf_local.empty:
            return None

        print(
            f"[DEBUG] {zone_prefix} prefiltered road edges for local mask: "
            f"{len(gdf_local)}/{len(gdf_edges)} kept"
        )
        return gdf_local
    except Exception as exc:
        print(f"[WARN] {zone_prefix} Failed to prefilter road edges locally: {exc}")
        return None


def prepare_road_geometry(
    *,
    G_roads: Any,
    scale_factor: Optional[float],
    road_width_multiplier_effective: float,
    min_printable_gap_mm: float,
    tiny_feature_threshold_mm: float = MICRO_REGION_THRESHOLD_MM,
    road_gap_fill_threshold_mm: float = 0.0,
    enforce_printable_min_width: bool = False,
    min_gap_fill_floor_mm: float = 0.0,
    global_center: Any,
    zone_polygon_local: Optional[BaseGeometry],
    zone_prefix: str = "",
) -> RoadGeometryPreparationResult:
    merged_roads_geom = None
    merged_roads_geom_local = None
    merged_roads_geom_local_raw = None
    semantic_centerlines_local = None
    min_road_width_for_build = None
    effective_min_width_mm = max(
        float(min_printable_gap_mm or 0.0),
        float(min_gap_fill_floor_mm or 0.0),
        0.5,
    )
    effective_gap_fill_mm = max(
        float(road_gap_fill_threshold_mm or 0.0),
        0.5,  # floor: always fill at least 0.5mm-model gaps (nozzle tolerance)
    )
    effective_tiny_feature_mm = max(float(tiny_feature_threshold_mm or 0.0), effective_min_width_mm)
    if scale_factor and float(scale_factor) > 0:
        try:
            min_road_width_for_build = model_mm_to_world_m(effective_min_width_mm, float(scale_factor))
        except Exception:
            min_road_width_for_build = None
    local_edges_subset = _build_local_road_edges_subset(
        G_roads=G_roads,
        global_center=global_center,
        zone_polygon_local=zone_polygon_local,
        zone_prefix=zone_prefix,
    )

    try:
        if local_edges_subset is not None and not local_edges_subset.empty:
            local_edges_subset = local_edges_subset.copy()
            if "highway" in local_edges_subset.columns:
                local_edges_subset["_normalized_highway"] = local_edges_subset["highway"].apply(
                    normalize_drivable_highway_tag
                )
                local_edges_subset = local_edges_subset[local_edges_subset["_normalized_highway"].notna()].copy()
            try:
                semantic_parts = [
                    geom
                    for geom in local_edges_subset.geometry.values
                    if geom is not None and not getattr(geom, "is_empty", True)
                ]
                if semantic_parts:
                    semantic_centerlines_local = unary_union(semantic_parts)
                    if zone_polygon_local is not None and not getattr(zone_polygon_local, "is_empty", True):
                        semantic_centerlines_local = semantic_centerlines_local.intersection(zone_polygon_local)
            except Exception:
                semantic_centerlines_local = None

            merged_roads_geom_local = build_road_polygons(
                local_edges_subset,
                width_multiplier=float(road_width_multiplier_effective),
                min_width_m=min_road_width_for_build,
                scale_factor=scale_factor,
            )
            if merged_roads_geom_local is not None and not getattr(merged_roads_geom_local, "is_empty", True):
                if zone_polygon_local is not None:
                    merged_roads_geom_local = merged_roads_geom_local.intersection(zone_polygon_local)
                try:
                    merged_roads_geom_local = merged_roads_geom_local.buffer(0)
                except Exception:
                    pass

            merged_roads_geom_local_raw = merged_roads_geom_local

            # ── Printable gap-fill ─────────────────────────────────────────
            # Close sub-printable gaps between road polygons NOW, on the raw
            # buffered road mask (before layer-precedence clipping, before the
            # canonical bundle). This is the only safe location:
            #   • Later stages (runtime_canonical_masks, detail_layer_pipeline)
            #     operate on already-processed masks where a CLOSE fills entire
            #     city blocks, not just road endpoint gaps.
            #   • merge_close_road_gaps adds only narrow wedges/strips (guarded
            #     by per-polygon min_dim < 1.25×gap and equiv_width < 1.1×gap),
            #     so legitimate terrain patches between distinct roads are left
            #     intact.
            # Only gap_fill_m is passed (min_feature_m=trim_width_m=0) so
            # normalize_road_mask_for_print only runs merge_close_road_gaps and
            # never deletes road polygons.
            if merged_roads_geom_local is not None and scale_factor and float(scale_factor) > 0:
                gap_fill_m = model_mm_to_world_m(float(effective_gap_fill_mm), float(scale_factor))
                if gap_fill_m and gap_fill_m > 0:
                    try:
                        min_road_feature_m = model_mm_to_world_m(0.5, float(scale_factor))
                        # orphan_hole fills interior junction holes (triangular gaps at
                        # intersections). Keep it SMALL (0.5mm = 2.5m world) so only
                        # tight junction wedges are filled — not courtyards or medians.
                        orphan_hole_m = model_mm_to_world_m(0.5, float(scale_factor))
                        filled = normalize_road_mask_for_print(
                            merged_roads_geom_local,
                            gap_fill_m=float(gap_fill_m),
                            min_feature_m=float(min_road_feature_m),
                            trim_width_m=0.0,
                            orphan_hole_width_m=float(orphan_hole_m),
                            zone_polygon=zone_polygon_local,
                        )
                        if filled is not None and not getattr(filled, "is_empty", True):
                            merged_roads_geom_local = filled
                            print(
                                f"[INFO] {zone_prefix} merged_roads: gap-fill "
                                f"{effective_gap_fill_mm:.2f}mm model "
                                f"({gap_fill_m:.2f}m world) applied"
                            )
                    except Exception as exc:
                        print(f"[WARN] {zone_prefix} road gap-fill failed: {exc}")
        elif (
            G_roads is not None
            and not isinstance(G_roads, GeoDataFrame)
            and hasattr(G_roads, "edges")
            and len(G_roads.edges) > 0
        ):
            merged_roads_geom = build_road_polygons(
                G_roads,
                width_multiplier=float(road_width_multiplier_effective),
                min_width_m=min_road_width_for_build,
                scale_factor=scale_factor,
            )
            print(
                f"[DEBUG] {zone_prefix} merged_roads_geom created: area={merged_roads_geom.area:.2f} m2"
                if merged_roads_geom is not None and hasattr(merged_roads_geom, "area")
                else f"[DEBUG] {zone_prefix} merged_roads_geom created"
            )
        else:
            print(f"[DEBUG] {zone_prefix} G_roads is None or empty, cannot create merged_roads_geom")
    except Exception as exc:
        print(f"[WARN] {zone_prefix} Failed to create merged_roads_geom: {exc}")
        merged_roads_geom = None

    if merged_roads_geom_local is None and merged_roads_geom is not None and global_center is not None:
        try:
            def to_local(x, y, z=None):
                x_local, y_local = global_center.to_local(x, y)
                return (x_local, y_local) if z is None else (x_local, y_local, z)

            merged_roads_geom_local_raw = transform_geometry(to_local, merged_roads_geom)
            if zone_polygon_local is not None:
                merged_roads_geom_local = merged_roads_geom_local_raw.intersection(zone_polygon_local)
                print(
                    f"[DEBUG] {zone_prefix} merged_roads_geom_local created: area={merged_roads_geom_local.area:.2f} m2, empty={merged_roads_geom_local.is_empty}"
                    if merged_roads_geom_local is not None and hasattr(merged_roads_geom_local, "area")
                    else f"[DEBUG] {zone_prefix} merged_roads_geom_local created"
                )
            else:
                merged_roads_geom_local = merged_roads_geom_local_raw
                print(
                    f"[DEBUG] {zone_prefix} merged_roads_geom_local created (no zone clipping): area={merged_roads_geom_local.area:.2f} m2"
                    if merged_roads_geom_local is not None and hasattr(merged_roads_geom_local, "area")
                    else f"[DEBUG] {zone_prefix} merged_roads_geom_local created (no zone clipping)"
                )
        except Exception as exc:
            print(f"[WARN] {zone_prefix} Failed to convert merged_roads_geom to local: {exc}")
            import traceback

            traceback.print_exc()
            merged_roads_geom_local = None
            merged_roads_geom_local_raw = None
    else:
        if merged_roads_geom_local is None:
            if merged_roads_geom is None:
                print(f"[DEBUG] {zone_prefix} merged_roads_geom is None, cannot create merged_roads_geom_local")
            if global_center is None:
                print(f"[DEBUG] {zone_prefix} global_center is None, cannot create merged_roads_geom_local")

    if (
        (merged_roads_geom_local is None or getattr(merged_roads_geom_local, "is_empty", True))
        and merged_roads_geom_local_raw is not None
        and not getattr(merged_roads_geom_local_raw, "is_empty", True)
    ):
        try:
            merged_roads_geom_local = (
                merged_roads_geom_local_raw.intersection(zone_polygon_local).buffer(0)
                if zone_polygon_local is not None
                else merged_roads_geom_local_raw.buffer(0)
            )
            if merged_roads_geom_local is not None and not getattr(merged_roads_geom_local, "is_empty", True):
                print(f"[WARN] {zone_prefix} restored merged_roads_geom_local from raw road mask fallback")
        except Exception:
            merged_roads_geom_local = None

    return RoadGeometryPreparationResult(
        merged_roads_geom=merged_roads_geom,
        merged_roads_geom_local=merged_roads_geom_local,
        merged_roads_geom_local_raw=merged_roads_geom_local_raw,
        semantic_centerlines_local=semantic_centerlines_local,
    )
