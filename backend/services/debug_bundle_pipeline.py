from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from shapely.geometry import mapping
from shapely.ops import unary_union

from services.debug_renderers import render_geometry_png, render_mesh_top_png, render_overlay_png
from services.canonical_mask_bundle import CanonicalMaskBundle
from services.detail_layer_utils import prepare_green_areas_for_processing
from services.geometry_diagnostics import concatenate_meshes, ensure_valid_geometry, geometry_stats, mesh_stats, overlap_report


def _backend_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _debug_run_root(task_id: str) -> Path:
    return _backend_root() / "debug" / "runs" / task_id


def _collect_gdf_geometry(gdf: Any) -> Any:
    if gdf is None:
        return None
    try:
        geoms = [geom for geom in gdf.geometry.values if geom is not None and not geom.is_empty]
    except Exception:
        return None
    if not geoms:
        return None
    try:
        return unary_union(geoms).buffer(0)
    except Exception:
        try:
            return unary_union(geoms)
        except Exception:
            return None


def _write_geojson(path: Path, geometry: Any) -> Optional[Path]:
    geometry = ensure_valid_geometry(geometry)
    if geometry is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": mapping(geometry),
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _export_mesh(path: Path, mesh: Any) -> Optional[Path]:
    mesh = concatenate_meshes(mesh)
    if mesh is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path


def _safe_rel(path: Optional[Path], root: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())


def _safe_render(fn, *args, **kwargs) -> Optional[Path]:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"[WARN] Debug render skipped: {exc}")
        return None


def create_debug_bundle(
    *,
    task_id: str,
    request: Any,
    output_dir: Path,
    zone: Any,
    source: Any,
    terrain_stage: Any,
    detail_layers: Any,
    postprocess_result: Any,
    clip_result: Any,
    merge_result: Any,
    export_result: Any,
    global_center: Any,
    canonical_mask_bundle: Optional[CanonicalMaskBundle] = None,
) -> Optional[Path]:
    run_dir = _debug_run_root(task_id)
    masks_dir = run_dir / "masks"
    layers_dir = run_dir / "layers"
    overlays_dir = run_dir / "overlays"
    reports_dir = run_dir / "reports"
    for path in (masks_dir, layers_dir, overlays_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    roads_raw = ensure_valid_geometry(getattr(terrain_stage, "merged_roads_geom_local", None))
    detail_roads_final = ensure_valid_geometry(
        getattr(getattr(detail_layers, "road_result", None), "source_polygons", None)
        or getattr(detail_layers, "road_cut_source", None)
    )
    detail_road_groove = ensure_valid_geometry(
        getattr(getattr(detail_layers, "groove_result", None), "road_polygons_used", None)
        or getattr(detail_layers, "road_groove_mask", None)
        or getattr(terrain_stage, "road_cut_mask", None)
    )
    detail_buildings_footprints = ensure_valid_geometry(
        getattr(detail_layers, "building_footprints", None)
        or getattr(terrain_stage, "building_union_local", None)
    )
    water_raw = ensure_valid_geometry(_collect_gdf_geometry(getattr(terrain_stage, "gdf_water_local", None)))
    detail_water_final = ensure_valid_geometry(getattr(detail_layers, "water_cut_polygons", None))
    detail_water_groove = ensure_valid_geometry(getattr(getattr(detail_layers, "groove_result", None), "water_polygons_used", None))
    prepared_green = prepare_green_areas_for_processing(
        getattr(source, "gdf_green", None),
        global_center=global_center,
        zone_polygon_local=getattr(zone, "zone_polygon_local", None),
    )
    parks_raw = ensure_valid_geometry(_collect_gdf_geometry(prepared_green))
    detail_parks_final = ensure_valid_geometry(getattr(getattr(detail_layers, "parks_result", None), "processed_polygons", None))
    detail_parks_groove = ensure_valid_geometry(getattr(getattr(detail_layers, "groove_result", None), "parks_polygons_used", None))
    zone_polygon = ensure_valid_geometry(getattr(zone, "zone_polygon_local", None))

    # Canonical masks are the source of truth for stage hand-off checks.
    roads_final = ensure_valid_geometry(
        getattr(canonical_mask_bundle, "roads_final", None) if canonical_mask_bundle is not None else None
    ) or detail_roads_final
    roads_semantic = ensure_valid_geometry(
        getattr(canonical_mask_bundle, "roads_semantic_preview", None) if canonical_mask_bundle is not None else None
    )
    road_groove = ensure_valid_geometry(
        getattr(canonical_mask_bundle, "road_groove_mask", None) if canonical_mask_bundle is not None else None
    ) or detail_road_groove
    buildings_footprints = ensure_valid_geometry(
        getattr(canonical_mask_bundle, "buildings_footprints", None) if canonical_mask_bundle is not None else None
    ) or detail_buildings_footprints
    parks_final = ensure_valid_geometry(
        getattr(canonical_mask_bundle, "parks_final", None) if canonical_mask_bundle is not None else None
    ) or detail_parks_final
    parks_groove = ensure_valid_geometry(
        getattr(canonical_mask_bundle, "parks_groove_mask", None) if canonical_mask_bundle is not None else None
    ) or detail_parks_groove
    water_final = ensure_valid_geometry(
        getattr(canonical_mask_bundle, "water_final", None) if canonical_mask_bundle is not None else None
    ) or detail_water_final
    water_groove = ensure_valid_geometry(
        getattr(canonical_mask_bundle, "water_groove_mask", None) if canonical_mask_bundle is not None else None
    ) or detail_water_groove

    mask_paths = {
        "zone_polygon": _write_geojson(masks_dir / "zone_polygon.geojson", zone_polygon),
        "roads_raw": _write_geojson(masks_dir / "roads_raw.geojson", roads_raw),
        "roads_final": _write_geojson(masks_dir / "roads_final.geojson", roads_final),
        "roads_semantic_preview": _write_geojson(masks_dir / "roads_semantic_preview.geojson", roads_semantic),
        "road_groove_mask": _write_geojson(masks_dir / "road_groove_mask.geojson", road_groove),
        "buildings_footprints": _write_geojson(masks_dir / "buildings_footprints.geojson", buildings_footprints),
        "water_raw": _write_geojson(masks_dir / "water_raw.geojson", water_raw),
        "water_final": _write_geojson(masks_dir / "water_final.geojson", water_final),
        "water_groove_mask": _write_geojson(masks_dir / "water_groove_mask.geojson", water_groove),
        "parks_raw": _write_geojson(masks_dir / "parks_raw.geojson", parks_raw),
        "parks_final": _write_geojson(masks_dir / "parks_final.geojson", parks_final),
        "parks_groove_mask": _write_geojson(masks_dir / "parks_groove_mask.geojson", parks_groove),
    }

    supports_mesh = concatenate_meshes(getattr(detail_layers, "support_meshes", None))
    merged_building_mesh = concatenate_meshes(getattr(detail_layers, "merged_building_mesh", None))
    building_meshes_final = concatenate_meshes(getattr(merge_result, "building_meshes", None))

    layer_paths = {
        "terrain_before_grooves": _export_mesh(layers_dir / "terrain_before_grooves.stl", getattr(terrain_stage, "terrain_mesh", None)),
        "terrain_after_grooves": _export_mesh(layers_dir / "terrain_after_grooves.stl", getattr(detail_layers, "terrain_mesh", None)),
        "terrain_after_postprocess": _export_mesh(layers_dir / "terrain_after_postprocess.stl", getattr(postprocess_result, "terrain_mesh", None)),
        "terrain_after_clip": _export_mesh(layers_dir / "terrain_after_clip.stl", getattr(clip_result, "terrain_mesh", None)),
        "base_final": _export_mesh(layers_dir / "base_final.stl", getattr(merge_result, "terrain_mesh", None)),
        "roads_mesh": _export_mesh(layers_dir / "roads_mesh.stl", getattr(clip_result, "road_mesh", None)),
        "buildings_merged": _export_mesh(layers_dir / "buildings_merged.stl", merged_building_mesh or building_meshes_final),
        "supports_mesh": _export_mesh(layers_dir / "supports_mesh.stl", supports_mesh),
        "water_mesh": _export_mesh(layers_dir / "water_mesh.stl", getattr(clip_result, "water_mesh", None)),
        "parks_mesh": _export_mesh(layers_dir / "parks_mesh.stl", getattr(clip_result, "parks_mesh", None)),
    }

    overlay_paths = {
        "roads_buildings": _safe_render(
            render_overlay_png,
            [
                {"geometry": roads_final, "facecolor": "black", "alpha": 0.9},
                {"geometry": buildings_footprints, "facecolor": "#c8b48a", "alpha": 0.7},
            ],
            overlays_dir / "roads_buildings.png",
        ),
        "roads_parks": _safe_render(
            render_overlay_png,
            [
                {"geometry": parks_final, "facecolor": "#5a9d60", "alpha": 0.7},
                {"geometry": roads_final, "facecolor": "black", "alpha": 0.9},
            ],
            overlays_dir / "roads_parks.png",
        ),
        "roads_water": _safe_render(
            render_overlay_png,
            [
                {"geometry": water_final, "facecolor": "#6ea8d7", "alpha": 0.8},
                {"geometry": roads_final, "facecolor": "black", "alpha": 0.9},
            ],
            overlays_dir / "roads_water.png",
        ),
        "semantic_composite": _safe_render(
            render_overlay_png,
            [
                {"geometry": parks_final, "facecolor": "#5a9d60", "alpha": 0.8},
                {"geometry": water_final, "facecolor": "#6ea8d7", "alpha": 0.8},
                {"geometry": roads_semantic or roads_final, "facecolor": "#111111", "edgecolor": "#111111", "linewidth": 1.0, "alpha": 0.95},
                {"geometry": buildings_footprints, "facecolor": "#c8b48a", "alpha": 0.7},
            ],
            overlays_dir / "semantic_composite.png",
        ),
        "grooves_vs_inserts": _safe_render(
            render_overlay_png,
            [
                {"geometry": road_groove, "facecolor": "white", "alpha": 0.9},
                {"geometry": parks_groove, "facecolor": "white", "alpha": 0.9},
                {"geometry": water_groove, "facecolor": "white", "alpha": 0.9},
                {"geometry": roads_final, "facecolor": "black", "alpha": 0.9},
                {"geometry": parks_final, "facecolor": "#5a9d60", "alpha": 0.8},
                {"geometry": water_final, "facecolor": "#6ea8d7", "alpha": 0.8},
            ],
            overlays_dir / "grooves_vs_inserts.png",
        ),
        "base_top": _safe_render(render_mesh_top_png, getattr(merge_result, "terrain_mesh", None), overlays_dir / "base_top.png"),
        "roads_top": _safe_render(render_mesh_top_png, getattr(clip_result, "road_mesh", None), overlays_dir / "roads_top.png"),
        "parks_top": _safe_render(render_mesh_top_png, getattr(clip_result, "parks_mesh", None), overlays_dir / "parks_top.png"),
        "water_top": _safe_render(render_mesh_top_png, getattr(clip_result, "water_mesh", None), overlays_dir / "water_top.png"),
    }

    metrics = {
        "task_id": task_id,
        "request": {
            "export_format": getattr(request, "export_format", None),
            "model_size_mm": getattr(request, "model_size_mm", None),
            "groove_clearance_mm_per_side": 0.2,
            "tiny_feature_threshold_mm": 0.2,
            "road_gap_fill_threshold_mm": 0.6,
        },
        "groove_backend": getattr(getattr(detail_layers, "groove_result", None), "boolean_backend_name", None),
        "groove_result": {
            "grooves_expected": getattr(getattr(detail_layers, "groove_result", None), "grooves_expected", False),
            "change_applied": getattr(getattr(detail_layers, "groove_result", None), "change_applied", False),
            "rejected": getattr(getattr(detail_layers, "groove_result", None), "rejected", False),
            "rejection_reason": getattr(getattr(detail_layers, "groove_result", None), "rejection_reason", None),
            "failure_reason": getattr(getattr(detail_layers, "groove_result", None), "failure_reason", None),
            "changed_vertices": getattr(getattr(detail_layers, "groove_result", None), "changed_vertices", False),
            "volume_removed_m3": getattr(getattr(detail_layers, "groove_result", None), "volume_removed_m3", None),
            "volume_removed_ratio": getattr(getattr(detail_layers, "groove_result", None), "volume_removed_ratio", None),
        },
        "masks": {
            "zone_polygon": geometry_stats(zone_polygon),
            "roads_raw": geometry_stats(roads_raw),
            "roads_final": geometry_stats(roads_final),
            "road_groove_mask": geometry_stats(road_groove),
            "buildings_footprints": geometry_stats(buildings_footprints),
            "parks_raw": geometry_stats(parks_raw),
            "parks_final": geometry_stats(parks_final),
            "parks_groove_mask": geometry_stats(parks_groove),
            "water_raw": geometry_stats(water_raw),
            "water_final": geometry_stats(water_final),
            "water_groove_mask": geometry_stats(water_groove),
        },
        "overlaps": {
            "roads_vs_buildings": overlap_report(roads_final, buildings_footprints, lhs_name="roads_final", rhs_name="buildings_footprints"),
            "parks_vs_roads": overlap_report(parks_final, roads_final, lhs_name="parks_final", rhs_name="roads_final"),
            "water_vs_roads": overlap_report(water_final, roads_final, lhs_name="water_final", rhs_name="roads_final"),
            "parks_vs_water": overlap_report(parks_final, water_final, lhs_name="parks_final", rhs_name="water_final"),
        },
        "stage_handoff": {
            "canonical_vs_detail": {
                "roads_overlap_delta_m2": float(
                    abs(
                        overlap_report(roads_final, buildings_footprints, lhs_name="canonical_roads", rhs_name="canonical_buildings").get("overlap_area", 0.0)
                        - overlap_report(detail_roads_final, detail_buildings_footprints, lhs_name="detail_roads", rhs_name="detail_buildings").get("overlap_area", 0.0)
                    )
                ),
                "road_groove_overlap_delta_m2": float(
                    abs(
                        overlap_report(road_groove, buildings_footprints, lhs_name="canonical_road_groove", rhs_name="canonical_buildings").get("overlap_area", 0.0)
                        - overlap_report(detail_road_groove, detail_buildings_footprints, lhs_name="detail_road_groove", rhs_name="detail_buildings").get("overlap_area", 0.0)
                    )
                ),
                "parks_vs_roads_detail_overlap_m2": float(
                    overlap_report(detail_parks_final, detail_roads_final, lhs_name="detail_parks", rhs_name="detail_roads").get("overlap_area", 0.0)
                ),
                "water_vs_roads_detail_overlap_m2": float(
                    overlap_report(detail_water_final, detail_roads_final, lhs_name="detail_water", rhs_name="detail_roads").get("overlap_area", 0.0)
                ),
            }
        },
        "meshes": {
            "terrain_before_grooves": mesh_stats(getattr(terrain_stage, "terrain_mesh", None), label="terrain_before_grooves"),
            "terrain_after_grooves": mesh_stats(getattr(detail_layers, "terrain_mesh", None), label="terrain_after_grooves"),
            "terrain_after_postprocess": mesh_stats(getattr(postprocess_result, "terrain_mesh", None), label="terrain_after_postprocess"),
            "terrain_after_clip": mesh_stats(getattr(clip_result, "terrain_mesh", None), label="terrain_after_clip"),
            "base_final": mesh_stats(getattr(merge_result, "terrain_mesh", None), label="base_final"),
            "roads_mesh": mesh_stats(getattr(clip_result, "road_mesh", None), label="roads_mesh"),
            "buildings_merged": mesh_stats(merged_building_mesh or building_meshes_final, label="buildings_merged"),
            "supports_mesh": mesh_stats(supports_mesh, label="supports_mesh"),
            "water_mesh": mesh_stats(getattr(clip_result, "water_mesh", None), label="water_mesh"),
            "parks_mesh": mesh_stats(getattr(clip_result, "parks_mesh", None), label="parks_mesh"),
        },
    }

    manifest = {
        "task_id": task_id,
        "run_dir": str(run_dir.resolve()),
        "exports": {
            "primary": str(getattr(export_result, "output_file_abs", "")),
            "primary_format": getattr(export_result, "primary_format", None),
            "stl_preview": str(getattr(export_result, "stl_preview_abs", "")) if getattr(export_result, "stl_preview_abs", None) else None,
            "output_dir": str(output_dir.resolve()),
        },
        "masks": {name: _safe_rel(path, run_dir) for name, path in mask_paths.items()},
        "layers": {name: _safe_rel(path, run_dir) for name, path in layer_paths.items()},
        "overlays": {name: _safe_rel(path, run_dir) for name, path in overlay_paths.items()},
        "reports": {
            "metrics": "reports/metrics.json",
            "report": "reports/report.json",
            "manifest": "reports/manifest.json",
        },
    }

    (reports_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / "report.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir
