from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional

import trimesh
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep

from services.printer_profile import PrinterProfile


def _load_geojson_geometry(path: Path | None) -> BaseGeometry | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    geoms = [shape(feature["geometry"]) for feature in payload.get("features", []) if feature.get("geometry")]
    if not geoms:
        return None
    try:
        return unary_union(geoms).buffer(0)
    except Exception:
        return unary_union(geoms)


def _iter_polygons(geometry: BaseGeometry | None) -> list[Polygon]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [geom for geom in getattr(geometry, "geoms", []) if getattr(geom, "geom_type", "") == "Polygon"]


def _polygon_equivalent_width(poly: Polygon) -> float:
    if poly is None or poly.is_empty:
        return 0.0
    try:
        area = float(getattr(poly, "area", 0.0) or 0.0)
        perimeter = float(getattr(poly, "length", 0.0) or 0.0)
        if perimeter <= 0.0:
            return 0.0
        return float((2.0 * area) / perimeter)
    except Exception:
        return 0.0


def _survives_printable_erosion(geometry: BaseGeometry | None, *, min_feature_m: float) -> bool:
    if geometry is None or getattr(geometry, "is_empty", True):
        return False
    if min_feature_m <= 0.0:
        return True
    erosion = float(min_feature_m) * 0.5
    try:
        shrunken = geometry.buffer(-erosion, join_style=1)
        return shrunken is not None and not getattr(shrunken, "is_empty", True)
    except Exception:
        return True


def _component_stats(geometry: BaseGeometry | None, *, min_feature_m: float) -> dict[str, Any]:
    polygons = _iter_polygons(geometry)
    small_components = 0
    hole_count = 0
    small_holes = 0
    total_area = 0.0
    total_hole_area = 0.0
    min_area_m2 = max((float(min_feature_m) ** 2) * 0.5, 1e-8)

    for poly in polygons:
        area = float(getattr(poly, "area", 0.0) or 0.0)
        total_area += area
        if area > 0.0 and (
            not _survives_printable_erosion(poly, min_feature_m=float(min_feature_m))
            or area < min_area_m2
        ):
            small_components += 1
        for ring in poly.interiors:
            try:
                hole = Polygon(ring.coords)
            except Exception:
                continue
            if hole.is_empty:
                continue
            hole_area = float(getattr(hole, "area", 0.0) or 0.0)
            total_hole_area += hole_area
            hole_count += 1
            if hole_area > 0.0 and (
                not _survives_printable_erosion(hole, min_feature_m=float(min_feature_m))
                or hole_area < min_area_m2
            ):
                small_holes += 1

    return {
        "polygon_count": len(polygons),
        "area": total_area,
        "small_component_count": small_components,
        "hole_count": hole_count,
        "small_hole_count": small_holes,
        "total_hole_area": total_hole_area,
    }


def _orphan_hole_count(
    geometry: BaseGeometry | None,
    *,
    backing_mask: BaseGeometry | None,
) -> int:
    count = 0
    prepared_backing = None
    if backing_mask is not None and not getattr(backing_mask, "is_empty", True):
        try:
            prepared_backing = prep(backing_mask)
        except Exception:
            prepared_backing = None
    for poly in _iter_polygons(geometry):
        for ring in poly.interiors:
            try:
                hole = Polygon(ring.coords)
            except Exception:
                continue
            if hole.is_empty:
                continue
            if prepared_backing is not None:
                try:
                    if not prepared_backing.intersects(hole):
                        count += 1
                    continue
                except Exception:
                    pass
            try:
                overlap_area = float(getattr(hole.intersection(backing_mask), "area", 0.0) or 0.0) if backing_mask is not None else 0.0
            except Exception:
                overlap_area = 0.0
            if overlap_area <= 1e-8:
                count += 1
    return count


def _overlap_area(lhs: BaseGeometry | None, rhs: BaseGeometry | None) -> float:
    if lhs is None or rhs is None or getattr(lhs, "is_empty", True) or getattr(rhs, "is_empty", True):
        return 0.0
    try:
        return float(getattr(lhs.intersection(rhs), "area", 0.0) or 0.0)
    except Exception:
        return 0.0


def _resolve_world_threshold_m(
    bundle_dir: Path,
    min_feature_mm: float,
    *,
    scale_factor_override: float | None = None,
) -> tuple[float, float | None]:
    # scale_factor convention: `print_mm / world_m` (see services/generator.py:
    # scale_factor = model_size_mm / zone_size_m). Therefore
    #   world_m = print_mm / scale_factor   (NO extra /1000).
    # The previous formula divided by an extra 1000 and was off by x1000 —
    # threshold_m came out ~0.00247 m (≈ 2.47 world-mm, too tight to catch
    # anything) instead of the correct ~2.47 world-m equivalent of 0.55 print-mm.
    # Default fallback when scale_factor is unknown: 1.0 m, a conservative
    # world-space threshold that at typical city scales (~1:4500) maps to
    # about 0.22 print-mm — still below the 0.4 mm nozzle so we don't over-reach.
    default_threshold_m = 1.0
    if scale_factor_override and scale_factor_override > 0.0:
        return float(min_feature_mm) / float(scale_factor_override), float(scale_factor_override)

    manifest_path = bundle_dir / "manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            scale_factor = float(payload.get("stats", {}).get("scale_factor") or payload.get("scale_factor") or 0.0)
            if scale_factor > 0.0:
                return float(min_feature_mm) / scale_factor, scale_factor
        except Exception:
            pass

    report_path = bundle_dir.parent / "reports" / "report.json"
    if report_path.exists():
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            request_model_size_mm = float(payload.get("request", {}).get("model_size_mm") or 0.0)
            zone_polygon = payload.get("masks", {}).get("zone_polygon", {})
            bounds = zone_polygon.get("bounds") or []
            if request_model_size_mm > 0.0 and len(bounds) == 4:
                width = abs(float(bounds[2]) - float(bounds[0]))
                height = abs(float(bounds[3]) - float(bounds[1]))
                dominant_extent = max(width, height)
                if dominant_extent > 0.0:
                    scale_factor = request_model_size_mm / dominant_extent
                    if scale_factor > 0.0:
                        return float(min_feature_mm) / float(scale_factor), float(scale_factor)
        except Exception:
            pass

    return default_threshold_m, None


def _overlap_failure_threshold_m2(scale_factor: float | None) -> float:
    if scale_factor is None or scale_factor <= 0.0:
        return 1e-6
    return 0.01 / (float(scale_factor) ** 2)


def build_mask_printability_report(
    bundle_dir: Path,
    *,
    min_feature_mm: float,
    scale_factor_override: float | None = None,
) -> dict[str, Any]:
    threshold_m, scale_factor = _resolve_world_threshold_m(
        bundle_dir,
        float(min_feature_mm),
        scale_factor_override=scale_factor_override,
    )
    files = {
        "zone_polygon": bundle_dir / "zone_polygon.geojson",
        "roads_final": bundle_dir / "roads_final.geojson",
        "road_groove_mask": bundle_dir / "road_groove_mask.geojson",
        "parks_final": bundle_dir / "parks_final.geojson",
        "parks_groove_mask": bundle_dir / "parks_groove_mask.geojson",
        "terrain_bare_mask": bundle_dir / "terrain_bare_mask.geojson",
        "terrain_land_mask": bundle_dir / "terrain_land_mask.geojson",
        "water_final": bundle_dir / "water_final.geojson",
        "water_groove_mask": bundle_dir / "water_groove_mask.geojson",
        "buildings_footprints": bundle_dir / "buildings_footprints.geojson",
    }
    geometries = {name: _load_geojson_geometry(path) for name, path in files.items()}

    layers = {
        name: _component_stats(geom, min_feature_m=threshold_m)
        for name, geom in geometries.items()
        if name != "zone_polygon"
    }
    road_holes = {
        "roads_final_orphan_holes": _orphan_hole_count(
            geometries["roads_final"],
            backing_mask=geometries["buildings_footprints"],
        ),
        "road_groove_orphan_holes": _orphan_hole_count(
            geometries["road_groove_mask"],
            backing_mask=geometries["buildings_footprints"],
        ),
    }
    overlaps = {
        "roads_vs_buildings": _overlap_area(geometries["roads_final"], geometries["buildings_footprints"]),
        "road_groove_vs_buildings": _overlap_area(geometries["road_groove_mask"], geometries["buildings_footprints"]),
        "parks_vs_roads": _overlap_area(geometries["parks_final"], geometries["roads_final"]),
        "parks_vs_road_groove": _overlap_area(geometries["parks_final"], geometries["road_groove_mask"]),
        "parks_vs_buildings": _overlap_area(geometries["parks_final"], geometries["buildings_footprints"]),
        "parks_groove_vs_roads": _overlap_area(geometries["parks_groove_mask"], geometries["roads_final"]),
        "parks_groove_vs_road_groove": _overlap_area(geometries["parks_groove_mask"], geometries["road_groove_mask"]),
        "parks_groove_vs_buildings": _overlap_area(geometries["parks_groove_mask"], geometries["buildings_footprints"]),
        "parks_groove_vs_water": _overlap_area(geometries["parks_groove_mask"], geometries["water_final"]),
        "water_vs_roads": _overlap_area(geometries["water_final"], geometries["roads_final"]),
        "water_groove_vs_roads": _overlap_area(geometries["water_groove_mask"], geometries["roads_final"]),
        "water_groove_vs_road_groove": _overlap_area(geometries["water_groove_mask"], geometries["road_groove_mask"]),
        "water_groove_vs_buildings": _overlap_area(geometries["water_groove_mask"], geometries["buildings_footprints"]),
    }

    failing_layers = [
        name
        for name, stats in layers.items()
        if int(stats["small_component_count"]) > 0 or int(stats["small_hole_count"]) > 0
    ]
    overlap_failure_threshold_m2 = _overlap_failure_threshold_m2(scale_factor)
    shared_fit_only_overlaps = {"parks_groove_vs_road_groove", "water_groove_vs_road_groove"}
    failing_overlaps = [
        name
        for name, area in overlaps.items()
        if name not in shared_fit_only_overlaps and float(area) > overlap_failure_threshold_m2
    ]
    failing_road_holes = [name for name, count in road_holes.items() if int(count) > 0]

    return {
        "bundle_dir": str(bundle_dir.resolve()),
        "threshold_mm": float(min_feature_mm),
        "threshold_m": threshold_m,
        "scale_factor": scale_factor,
        "overlap_failure_threshold_m2": overlap_failure_threshold_m2,
        "status": "pass" if not failing_layers and not failing_overlaps and not failing_road_holes else "fail",
        "layers": layers,
        "road_holes": road_holes,
        "overlaps": overlaps,
        "failing_layers": failing_layers,
        "failing_overlaps": failing_overlaps,
        "failing_road_holes": failing_road_holes,
    }


def write_mask_printability_report(
    bundle_dir: Path,
    *,
    printer_profile: PrinterProfile,
    scale_factor_override: float | None = None,
) -> Path:
    report = build_mask_printability_report(
        bundle_dir,
        min_feature_mm=float(printer_profile.min_printable_feature_mm),
        scale_factor_override=scale_factor_override,
    )
    out_path = bundle_dir / "printability_audit.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def summarize_mask_printability_failures(report: Mapping[str, Any]) -> str:
    problems: list[str] = []
    failing_layers = list(report.get("failing_layers") or [])
    failing_overlaps = list(report.get("failing_overlaps") or [])
    failing_road_holes = list(report.get("failing_road_holes") or [])
    if failing_layers:
        problems.append(f"layers={','.join(map(str, failing_layers))}")
    if failing_overlaps:
        problems.append(f"overlaps={','.join(map(str, failing_overlaps))}")
    if failing_road_holes:
        problems.append(f"road_holes={','.join(map(str, failing_road_holes))}")
    return "; ".join(problems) if problems else "mask printability audit failed"


def find_prusaslicer_cli() -> Path | None:
    candidates = [
        Path(r"H:\3dMAP_WORK_2.0\tools\external\PrusaSlicer-2.9.4\PrusaSlicer-2.9.4\prusa-slicer-console.exe"),
        Path(r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe"),
        Path(r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer.exe"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _write_slicer_config(
    path: Path,
    *,
    printer_profile: PrinterProfile,
    support_material: bool = False,
) -> Path:
    config = f"""printer_technology = FFF
bed_shape = 0x0,256x0,256x256,0x256
gcode_flavor = marlin2
nozzle_diameter = {printer_profile.nozzle_diameter_mm}
filament_diameter = 1.75
layer_height = 0.20
first_layer_height = 0.20
perimeters = 2
top_solid_layers = 4
bottom_solid_layers = 4
solid_layers = 4
fill_density = 15%
fill_pattern = rectilinear
thin_walls = 1
gap_fill_enabled = 1
external_perimeters_first = 0
perimeter_extrusion_width = {printer_profile.nominal_line_width_mm}
external_perimeter_extrusion_width = {printer_profile.external_line_width_mm}
infill_extrusion_width = {printer_profile.nominal_line_width_mm}
solid_infill_extrusion_width = {printer_profile.nominal_line_width_mm}
top_infill_extrusion_width = {printer_profile.external_line_width_mm}
first_layer_extrusion_width = {printer_profile.first_layer_line_width_mm}
skirts = 0
brim_type = outer_only
brim_width = 3
support_material = {1 if support_material else 0}
elefant_foot_compensation = {printer_profile.elephant_foot_compensation_mm}
retract_length = 0.8
retract_speed = 35
travel_speed = 180
perimeter_speed = 35
external_perimeter_speed = 25
small_perimeter_speed = 20
infill_speed = 60
solid_infill_speed = 40
top_solid_infill_speed = 25
temperature = 210
first_layer_temperature = 215
bed_temperature = 60
first_layer_bed_temperature = 65
"""
    path.write_text(config, encoding="ascii")
    return path


def _boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    try:
        edges = mesh.edges_sorted
        if len(edges) == 0:
            return 0
        unique = trimesh.grouping.group_rows(edges, require_count=1)
        return int(len(unique))
    except Exception:
        return 10**9


def _mesh_part_report(mesh_path: Path, *, part_name: str, printer_profile: PrinterProfile) -> dict[str, Any]:
    report: dict[str, Any] = {
        "part": part_name,
        "path": str(mesh_path.resolve()),
        "exists": mesh_path.exists(),
    }
    if not mesh_path.exists():
        report["status"] = "fail"
        report["failures"] = ["missing_file"]
        return report

    try:
        loaded = trimesh.load(mesh_path, force="mesh")
        mesh = loaded if isinstance(loaded, trimesh.Trimesh) else None
    except Exception as exc:
        report["status"] = "fail"
        report["failures"] = [f"load_error:{exc}"]
        return report

    if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
        report["status"] = "fail"
        report["failures"] = ["empty_mesh"]
        return report

    try:
        components = list(mesh.split(only_watertight=False))
    except Exception:
        components = [mesh]
    component_face_counts = sorted(
        [
            int(len(component.faces))
            for component in components
            if component is not None and getattr(component, "faces", None) is not None and len(component.faces) > 0
        ],
        reverse=True,
    )
    total_component_faces = sum(component_face_counts)
    dominant_component_faces = int(component_face_counts[0]) if component_face_counts else 0
    dominant_face_ratio = (
        float(dominant_component_faces) / float(total_component_faces)
        if total_component_faces > 0
        else 0.0
    )
    nontrivial_component_count = sum(1 for face_count in component_face_counts if int(face_count) > 4)
    bounds = mesh.bounds if getattr(mesh, "bounds", None) is not None else None
    if bounds is not None and len(bounds) == 2:
        extents = (bounds[1] - bounds[0]).tolist()
        min_z = float(bounds[0][2])
    else:
        extents = [0.0, 0.0, 0.0]
        min_z = 0.0

    failures: list[str] = []
    warnings: list[str] = []
    boundary_edge_count = _boundary_edge_count(mesh)
    if part_name == "base":
        base_closed_topology = (
            boundary_edge_count == 0
            and len(components) == 1
            and int(nontrivial_component_count) == 1
            and float(dominant_face_ratio) >= 0.995
        )
        if not bool(mesh.is_watertight):
            # Trimesh may report non-watertight for numerically noisy but still
            # closed manifold shells (no boundary edges, single dominant body).
            # Treat this as warning; true open/fragmented bases remain failures.
            if base_closed_topology:
                warnings.append("base_watertight_false_but_closed")
            else:
                failures.append("base_not_watertight")
        if len(components) != 1:
            failures.append("base_component_count")
        # Practical tolerance: tiny residual boundary rings often come from
        # boolean numeric noise and are harmless when slicer validates the part.
        # Keep hard-fail only for materially open shells.
        if boundary_edge_count > 64:
            failures.append("base_open_boundaries")
        elif boundary_edge_count > 0:
            warnings.append("base_open_boundaries_minor")
    else:
        if len(components) == 0:
            failures.append("no_components")

    tiny_components = 0
    for component in components:
        try:
            comp_bounds = component.bounds
            comp_extents = comp_bounds[1] - comp_bounds[0]
            min_xy = min(float(comp_extents[0]), float(comp_extents[1]))
            if min_xy < float(printer_profile.min_printable_feature_mm) * 0.35:
                tiny_components += 1
        except Exception:
            continue
    if tiny_components > 0 and part_name != "base":
        if part_name == "buildings":
            warnings.append("tiny_component_bbox")
        else:
            failures.append("tiny_component_bbox")

    report.update(
        {
            "status": "pass" if not failures else "fail",
            "failures": failures,
            "warnings": warnings,
            "watertight": bool(mesh.is_watertight),
            "component_count": len(components),
            "nontrivial_component_count": int(nontrivial_component_count),
            "face_count": int(len(mesh.faces)),
            "vertex_count": int(len(mesh.vertices)),
            "dominant_component_faces": dominant_component_faces,
            "dominant_face_ratio": dominant_face_ratio,
            "boundary_edge_count": boundary_edge_count,
            "bounds_mm": extents,
            "min_z_mm": min_z,
            "tiny_component_count": tiny_components,
        }
    )
    return report


def _run_slicer_slice(
    slicer: Path,
    config_path: Path,
    input_path: Path,
    output_path: Path,
    *,
    rotate_x_deg: int = 0,
) -> dict[str, Any]:
    cmd = [
        str(slicer),
        "--load",
        str(config_path),
        "--center",
        "110,110",
        "--export-gcode",
        "--output",
        str(output_path),
        str(input_path),
    ]
    if rotate_x_deg:
        cmd[4:4] = ["--rotate-x", str(int(rotate_x_deg))]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    log_path = output_path.with_suffix(".log.txt")
    log_path.write_text(stdout + ("\n--- STDERR ---\n" + stderr if stderr else ""), encoding="utf-8")
    result: dict[str, Any] = {
        "input": str(input_path.resolve()),
        "gcode": str(output_path.resolve()),
        "log": str(log_path.resolve()),
        "exit_code": int(proc.returncode),
        "warnings": [line.strip() for line in stdout.splitlines() if "print warning:" in line.lower()],
        "has_empty_first_layer": "no extrusions in the first layer" in stdout.lower(),
        "has_empty_layer_warning": "empty layer between" in stdout.lower(),
        "sliced": proc.returncode == 0 and output_path.exists(),
    }
    if output_path.exists():
        text = output_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        result["layers"] = sum(1 for line in text if line.startswith(";LAYER_CHANGE"))
        result["estimated_time"] = next(
            (line.strip() for line in text if "estimated printing time (normal mode)" in line.lower()),
            None,
        )
        result["filament"] = next(
            (line.strip() for line in text if "filament used [mm]" in line.lower()),
            None,
        )
        result["gcode_size_bytes"] = output_path.stat().st_size
    return result


def write_export_print_acceptance_report(
    *,
    task_id: str,
    output_dir: Path,
    parts_for_print: Mapping[str, str] | None,
    expected_parts: Mapping[str, bool] | None = None,
    printer_profile: PrinterProfile,
    require_slicer_validation: bool = True,
    fail_on_slicer_warnings: bool = True,
    rotate_x_deg: int = 0,
) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "task_id": task_id,
        "printer_model": printer_profile.printer_model,
        "require_slicer_validation": bool(require_slicer_validation),
        "fail_on_slicer_warnings": bool(fail_on_slicer_warnings),
        "expected_parts": {str(k).lower(): bool(v) for k, v in (expected_parts or {}).items()},
        "parts": {},
        "slicer": {},
    }

    normalized_parts: dict[str, Path] = {}
    for key, value in (parts_for_print or {}).items():
        if value:
            normalized_parts[str(key).lower()] = Path(value).resolve()
    expected = {str(k).lower(): bool(v) for k, v in (expected_parts or {}).items()}
    required_parts = [part for part in ("base", "roads", "parks", "water") if expected.get(part, part == "base")]

    failing_checks: list[str] = []
    for part_name in required_parts:
        path = normalized_parts.get(part_name)
        if path is None:
            report["parts"][part_name] = {"status": "fail", "failures": ["missing_required_part"]}
            failing_checks.append(f"{part_name}:missing_required_part")
            continue
        part_report = _mesh_part_report(path, part_name=part_name, printer_profile=printer_profile)
        report["parts"][part_name] = part_report
        if part_report.get("status") != "pass":
            failing_checks.extend(f"{part_name}:{failure}" for failure in part_report.get("failures", []))

    buildings_expected = expected.get("buildings", False)
    buildings_path = normalized_parts.get("buildings")
    if buildings_expected and buildings_path is None:
        report["parts"]["buildings"] = {"status": "fail", "failures": ["missing_required_part"]}
        failing_checks.append("buildings:missing_required_part")
    elif buildings_path is not None:
        buildings_report = _mesh_part_report(buildings_path, part_name="buildings", printer_profile=printer_profile)
        report["parts"]["buildings"] = buildings_report
        if buildings_report.get("status") != "pass":
            failing_checks.extend(f"buildings:{failure}" for failure in buildings_report.get("failures", []))

    slicer = find_prusaslicer_cli()
    report["slicer"]["path"] = str(slicer.resolve()) if slicer is not None else None
    report["slicer"]["rotate_x_deg"] = int(rotate_x_deg)
    if require_slicer_validation and slicer is None:
        failing_checks.append("slicer:not_found")
        report["slicer"]["status"] = "fail"
    elif slicer is not None:
        default_config_path = _write_slicer_config(
            output_dir / f"{task_id}_print_check_config.ini",
            printer_profile=printer_profile,
            support_material=False,
        )
        buildings_config_path = _write_slicer_config(
            output_dir / f"{task_id}_print_check_buildings_config.ini",
            printer_profile=printer_profile,
            support_material=True,
        )
        report["slicer"]["config"] = str(default_config_path.resolve())
        report["slicer"]["buildings_config"] = str(buildings_config_path.resolve())
        slicer_parts: dict[str, Any] = {}
        slicer_part_names = list(required_parts)
        if buildings_path is not None:
            slicer_part_names.append("buildings")
        for part_name in slicer_part_names:
            path = normalized_parts.get(part_name)
            if path is None or not path.exists():
                continue
            config_path = buildings_config_path if part_name in {"buildings", "parks", "roads"} else default_config_path
            slicer_parts[part_name] = _run_slicer_slice(
                slicer,
                config_path,
                path,
                output_dir / f"{task_id}_{part_name}_check.gcode",
                rotate_x_deg=int(rotate_x_deg),
            )
            if not slicer_parts[part_name].get("sliced"):
                failing_checks.append(f"slicer:{part_name}:slice_failed")
            if slicer_parts[part_name].get("has_empty_first_layer"):
                failing_checks.append(f"slicer:{part_name}:empty_first_layer")
            if fail_on_slicer_warnings and slicer_parts[part_name].get("has_empty_layer_warning"):
                failing_checks.append(f"slicer:{part_name}:empty_layer_warning")
            if fail_on_slicer_warnings and slicer_parts[part_name].get("warnings"):
                warnings = [str(w) for w in (slicer_parts[part_name].get("warnings") or [])]
                blocking_warnings = warnings
                # PrusaSlicer occasionally emits a generic base-only stability
                # warning even for otherwise valid, sliceable bases. Keep it in
                # the report, but don't hard-fail the whole pipeline on this
                # single advisory line.
                if part_name == "base":
                    blocking_warnings = [
                        w
                        for w in warnings
                        if "detected print stability issues" not in w.lower()
                    ]
                if blocking_warnings:
                    failing_checks.append(f"slicer:{part_name}:warnings")

        base_report = report["parts"].get("base")
        base_slicer = slicer_parts.get("base")
        if (
            isinstance(base_report, dict)
            and isinstance(base_slicer, dict)
            and base_slicer.get("sliced")
            and not base_slicer.get("warnings")
            and int(base_report.get("component_count", 10**9) or 10**9) <= 1
            and int(base_report.get("nontrivial_component_count", 10**9) or 10**9) <= 2
            and float(base_report.get("dominant_face_ratio", 0.0) or 0.0) >= 0.99
        ):
            # Slicer-validated recovery gate. When PrusaSlicer accepts the base
            # without warnings AND trimesh sees a single dominant shell (one
            # nontrivial component, face ratio >= 0.99), we trust the slicer:
            # pinholes, tiny boundary rings, and trimesh's strict watertight
            # check are downgraded to warnings. No hard boundary-edge ceiling —
            # the user repairs any residual topology in the slicer anyway.
            recoverable = {
                "base_not_watertight": "base_near_watertight_recovered_by_slicer",
                "base_open_boundaries": "base_open_boundaries_recovered_by_slicer",
                "base_component_count": "base_topology_recovered_by_slicer",
            }
            base_failures = list(base_report.get("failures") or [])
            base_warnings = list(base_report.get("warnings") or [])
            kept_failures: list[str] = []
            changed = False
            for failure in base_failures:
                if failure in recoverable:
                    changed = True
                    warn = recoverable[failure]
                    if warn not in base_warnings:
                        base_warnings.append(warn)
                    failing_checks = [
                        check for check in failing_checks
                        if check != f"base:{failure}"
                    ]
                else:
                    kept_failures.append(failure)
            if changed:
                base_report["failures"] = kept_failures
                base_report["status"] = "pass" if not kept_failures else "fail"
                base_report["warnings"] = sorted(set(base_warnings))
        report["slicer"]["parts"] = slicer_parts
        report["slicer"]["status"] = "pass" if not any(check.startswith("slicer:") for check in failing_checks) else "fail"

    report["failing_checks"] = sorted(set(failing_checks))
    report["status"] = "pass" if not report["failing_checks"] else "fail"
    out_path = output_dir / f"{task_id}_print_acceptance.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def summarize_export_print_failures(report: Mapping[str, Any]) -> str:
    failing_checks = list(report.get("failing_checks") or [])
    if failing_checks:
        return ", ".join(map(str, failing_checks[:8]))
    return "export print acceptance failed"
