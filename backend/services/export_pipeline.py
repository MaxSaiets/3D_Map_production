from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, Optional, Tuple, List
import zipfile

import numpy as np
import trimesh

from services.model_exporter import export_preview_parts_3mf, export_scene


@dataclass
class ExportPipelineResult:
    output_file_abs: Path
    primary_format: str
    stl_preview_abs: Optional[Path]
    parts_from_main: Optional[Dict[str, str]]


def _find_prusaslicer_console() -> Optional[Path]:
    candidates = [
        Path(r"H:\3dMAP_WORK_2.0\tools\external\PrusaSlicer-2.9.4\PrusaSlicer-2.9.4\prusa-slicer-console.exe"),
        Path(r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe"),
        Path(r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer.exe"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _prepare_print_layout_inputs(
    *,
    output_dir: Path,
    task_id: str,
    parts_from_main: Optional[Dict[str, str]],
) -> List[str]:
    if not parts_from_main:
        return []

    layout_dir = (output_dir / f"{task_id}_print_layout_parts").resolve()
    layout_dir.mkdir(parents=True, exist_ok=True)

    ordered_paths: list[str] = []
    ordered_keys = ("base", "roads", "parks", "water", "buildings")
    for key in ordered_keys:
        raw_path = parts_from_main.get(key)
        if not raw_path:
            continue
        candidate = Path(raw_path)
        if not candidate.exists():
            continue

        if key != "buildings":
            ordered_paths.append(str(candidate))
            continue

        try:
            building_mesh = trimesh.load_mesh(candidate, force="mesh")
        except Exception:
            ordered_paths.append(str(candidate))
            continue

        if building_mesh is None or building_mesh.faces is None or len(building_mesh.faces) == 0:
            continue

        try:
            components = list(building_mesh.split(only_watertight=False))
        except Exception:
            components = [building_mesh]

        exported_count = 0
        for idx, component in enumerate(components, start=1):
            if component is None or component.faces is None or len(component.faces) == 0:
                continue
            part = component.copy()
            try:
                if len(part.faces) < 8:
                    continue
                bounds = part.bounds
                extents = np.asarray(bounds[1] - bounds[0], dtype=float)
                if extents.size < 3:
                    continue
                if float(np.max(extents)) <= 0.05:
                    continue
            except Exception:
                pass
            try:
                part.apply_translation([0.0, 0.0, -float(bounds[0][2])])
            except Exception:
                pass
            part_path = layout_dir / f"{task_id}_buildings_part_{idx:03d}.stl"
            try:
                part.export(part_path)
            except Exception:
                continue
            try:
                if not part_path.exists() or part_path.stat().st_size < 200:
                    continue
            except Exception:
                continue
            ordered_paths.append(str(part_path))
            exported_count += 1

        if exported_count == 0:
            ordered_paths.append(str(candidate))

    return ordered_paths


def _create_print_layout_3mf_with_prusaslicer(
    *,
    output_dir: Path,
    task_id: str,
    parts_from_main: Optional[Dict[str, str]],
) -> Optional[Path]:
    if not parts_from_main:
        return None

    slicer = _find_prusaslicer_console()
    if slicer is None:
        return None

    ordered_paths = _prepare_print_layout_inputs(
        output_dir=output_dir,
        task_id=task_id,
        parts_from_main=parts_from_main,
    )
    if not ordered_paths:
        return None

    out_path = (output_dir / f"{task_id}_print_layout.3mf").resolve()
    cmd = [
        str(slicer),
        "--center",
        "110,110",
        "--export-3mf",
        "--output",
        str(out_path),
        *ordered_paths,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    except Exception as exc:
        print(f"[WARN] Failed to launch PrusaSlicer for print-layout 3MF: {exc}")
        return None

    if proc.returncode != 0 or not out_path.exists():
        print("[WARN] PrusaSlicer 3MF export failed, falling back to trimesh layout export")
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        return None

    return out_path


def _create_print_package(
    *,
    output_dir: Path,
    task_id: str,
    primary_format: str,
    output_file_abs: Path,
    stl_preview_abs: Optional[Path],
    parts_from_main: Optional[Dict[str, str]],
) -> Optional[Path]:
    package_entries: Dict[str, str] = {}
    if output_file_abs.exists():
        package_entries[output_file_abs.name] = str(output_file_abs)
    if stl_preview_abs and stl_preview_abs.exists():
        package_entries[stl_preview_abs.name] = str(stl_preview_abs)
    if parts_from_main:
        for _, path in parts_from_main.items():
            candidate = Path(path)
            if candidate.exists():
                package_entries[candidate.name] = str(candidate)
    print_layout_candidate = (output_dir / f"{task_id}_print_layout.3mf").resolve()
    if print_layout_candidate.exists():
        package_entries[print_layout_candidate.name] = str(print_layout_candidate)

    if not package_entries:
        return None

    manifest = {
        "task_id": task_id,
        "primary_format": primary_format,
        "recommended_for_slicer": "separate STL parts",
        "recommended_files": {
            "base": f"{task_id}_base.stl",
            "roads": f"{task_id}_roads.stl",
            "parks": f"{task_id}_parks.stl",
            "water": f"{task_id}_water.stl",
        },
        "files": sorted(package_entries.keys()),
        "notes": [
            "Use separate STL parts for print-fit validation and printing.",
            "3MF may render inconsistently across slicers; STL parts are the canonical print path.",
        ],
    }

    package_path = (output_dir / f"{task_id}_print_package.zip").resolve()
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for arcname, src in sorted(package_entries.items()):
            archive.write(src, arcname)
        archive.writestr("print_package_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    return package_path


def _create_print_layout_3mf(
    *,
    output_dir: Path,
    task_id: str,
    parts_from_main: Optional[Dict[str, str]],
) -> Optional[Path]:
    if not parts_from_main:
        return None

    slicer_layout = _create_print_layout_3mf_with_prusaslicer(
        output_dir=output_dir,
        task_id=task_id,
        parts_from_main=parts_from_main,
    )
    if slicer_layout is not None and slicer_layout.exists():
        return slicer_layout

    color_map = {
        "base": [200, 180, 140, 255],
        "terrain": [200, 180, 140, 255],
        "roads": [60, 60, 60, 255],
        "buildings": [160, 160, 160, 255],
        "water": [100, 150, 200, 255],
        "parks": [100, 150, 100, 255],
    }
    ordered_keys = ["base", "roads", "parks", "water", "buildings"]
    spacing_mm = 8.0
    cursor_x = 0.0
    plate_y_center = 110.0
    scene = trimesh.Scene()

    for key in ordered_keys:
        raw_path = parts_from_main.get(key)
        if not raw_path:
            continue
        part_path = Path(raw_path)
        if not part_path.exists():
            continue
        try:
            mesh = trimesh.load_mesh(part_path, force="mesh")
        except Exception:
            continue
        if mesh is None or mesh.faces is None or len(mesh.faces) == 0:
            continue

        part = mesh.copy()
        bounds = part.bounds
        extents = bounds[1] - bounds[0]
        translate = np.array([
            cursor_x - bounds[0][0],
            plate_y_center - ((bounds[0][1] + bounds[1][1]) / 2.0),
            -bounds[0][2],
        ], dtype=float)
        part.apply_translation(translate)

        color = color_map.get(key, [150, 150, 150, 255])
        try:
            part.visual = trimesh.visual.ColorVisuals(face_colors=np.tile(color, (len(part.faces), 1)))
        except Exception:
            pass

        scene.add_geometry(part, node_name=key, geom_name=key)
        cursor_x += float(extents[0]) + spacing_mm

    if len(scene.geometry) == 0:
        return None

    out_path = (output_dir / f"{task_id}_print_layout.3mf").resolve()
    scene.export(str(out_path))
    return out_path


def export_generation_outputs(
    *,
    task: Any,
    request: Any,
    task_id: str,
    output_dir: Path,
    terrain_mesh: Optional[trimesh.Trimesh],
    road_mesh: Optional[trimesh.Trimesh],
    building_meshes: Any,
    water_mesh: Optional[trimesh.Trimesh],
    parks_mesh: Optional[trimesh.Trimesh],
    reference_xy_m: Optional[float],
    preserve_z: bool = False,
    preserve_xy: bool = False,
    include_preview_parts: bool = True,
    completion_message: str = "РњРѕРґРµР»СЊ РіРѕС‚РѕРІР°!",
) -> ExportPipelineResult:
    primary_format = request.export_format.lower()
    output_file = output_dir / f"{task_id}.{primary_format}"
    output_file_abs = output_file.resolve()
    assembly_3mf_abs: Optional[Path] = None
    stl_parts_from_preview: Optional[Dict[str, str]] = None

    terrain_mesh_for_export = terrain_mesh
    building_meshes_for_export = building_meshes
    merge_buildings_into_base = False
    if merge_buildings_into_base:
        try:
            valid_buildings = [mesh for mesh in building_meshes if mesh is not None and len(mesh.vertices) > 0]
            if valid_buildings:
                terrain_mesh_for_export = trimesh.util.concatenate([terrain_mesh, *valid_buildings])
                building_meshes_for_export = None
                print("[EXPORT] Buildings will be exported as part of Base")
        except Exception as exc:
            print(f"[WARN] Failed to merge buildings into export base: {exc}")
            terrain_mesh_for_export = terrain_mesh
            building_meshes_for_export = building_meshes

    print(
        f"РњРµС€С–: terrain={'OK' if terrain_mesh_for_export else 'None'}, roads={'OK' if road_mesh else 'None'}, "
        f"buildings={len(building_meshes_for_export) if building_meshes_for_export else 0}, water={'OK' if water_mesh else 'None'}, "
        f"parks={'OK' if parks_mesh else 'None'}"
    )

    parts_from_main = export_scene(
        terrain_mesh=terrain_mesh_for_export,
        road_mesh=road_mesh,
        building_meshes=building_meshes_for_export,
        water_mesh=water_mesh,
        parks_mesh=parks_mesh,
        filename=str(output_file_abs),
        format=request.export_format,
        model_size_mm=request.model_size_mm,
        add_flat_base=(terrain_mesh is None),
        base_thickness_mm=float(request.terrain_base_thickness_mm),
        reference_xy_m=reference_xy_m,
        preserve_z=preserve_z,
        preserve_xy=preserve_xy,
        rotate_to_ground=False,
    )

    if parts_from_main and isinstance(parts_from_main, dict) and primary_format == "stl":
        for part_name, path in parts_from_main.items():
            if str(part_name).lower() == "stl":
                continue
            task.set_output(f"{part_name}_stl", str(Path(path).resolve()))

    stl_preview_abs: Optional[Path] = None
    if primary_format == "3mf":
        stl_preview_abs = (output_dir / f"{task_id}.stl").resolve()
        stl_parts = export_scene(
            terrain_mesh=terrain_mesh_for_export,
            road_mesh=road_mesh,
            building_meshes=building_meshes_for_export,
            water_mesh=water_mesh,
            parks_mesh=parks_mesh,
            filename=str(stl_preview_abs),
            format="stl",
            model_size_mm=request.model_size_mm,
            add_flat_base=(terrain_mesh is None),
            base_thickness_mm=float(request.terrain_base_thickness_mm),
            reference_xy_m=reference_xy_m,
            preserve_z=preserve_z,
            preserve_xy=preserve_xy,
            rotate_to_ground=False,
        )
        stl_parts_from_preview = stl_parts if isinstance(stl_parts, dict) else None
        if stl_parts and isinstance(stl_parts, dict):
            for part_name, path in stl_parts.items():
                if str(part_name).lower() == "stl":
                    continue
                task.set_output(f"{part_name}_stl", str(Path(path).resolve()))

    if include_preview_parts:
        try:
            preview_items: list[Tuple[str, trimesh.Trimesh]] = []
            if terrain_mesh_for_export is not None:
                preview_items.append(("Base", terrain_mesh_for_export))
            if road_mesh is not None:
                preview_items.append(("Roads", road_mesh))
            if building_meshes_for_export:
                try:
                    combined_buildings = trimesh.util.concatenate([b for b in building_meshes_for_export if b is not None])
                    if combined_buildings is not None and len(combined_buildings.vertices) > 0:
                        preview_items.append(("Buildings", combined_buildings))
                except Exception:
                    pass
            if water_mesh is not None:
                preview_items.append(("Water", water_mesh))
            if parks_mesh is not None:
                preview_items.append(("Parks", parks_mesh))

            if preview_items:
                prefix = str((output_dir / task_id).resolve())
                include_components = {
                    "base": getattr(request, "preview_include_base", True),
                    "terrain": getattr(request, "preview_include_base", True),
                    "roads": getattr(request, "preview_include_roads", True),
                    "buildings": getattr(request, "preview_include_buildings", True) and not merge_buildings_into_base,
                    "water": getattr(request, "preview_include_water", True),
                    "parks": getattr(request, "preview_include_parks", True),
                    "green": getattr(request, "preview_include_parks", True),
                }
                parts = export_preview_parts_3mf(
                    output_prefix=prefix,
                    mesh_items=preview_items,
                    model_size_mm=request.model_size_mm,
                    add_flat_base=(terrain_mesh is None),
                    base_thickness_mm=float(request.terrain_base_thickness_mm),
                    rotate_to_ground=False,
                    reference_xy_m=reference_xy_m,
                    preserve_z=preserve_z,
                    preserve_xy=preserve_xy,
                    include_components=include_components,
                )
                for part_name, path in parts.items():
                    task.set_output(part_name, str(Path(path).resolve()))
        except Exception as exc:
            print(f"[WARN] Preview parts export failed: {exc}")

    if not output_file_abs.exists():
        if primary_format == "3mf":
            stl_fallback = (output_dir / f"{task_id}.stl").resolve()
            if stl_fallback.exists():
                task.set_output("stl", str(stl_fallback))
                task.complete(str(stl_fallback))
                task.update_status("completed", 100, "3MF РЅРµ Р·РіРµРЅРµСЂРѕРІР°РЅРѕ, Р°Р»Рµ STL СЃС‚РІРѕСЂРµРЅРѕ (fallback).")
                print(f"[WARN] 3MF РЅРµ СЃС‚РІРѕСЂРµРЅРѕ РґР»СЏ {task_id}, РІРёРєРѕСЂРёСЃС‚Р°РЅРѕ STL fallback: {stl_fallback}")
                return ExportPipelineResult(
                    output_file_abs=stl_fallback,
                    primary_format="stl",
                    stl_preview_abs=stl_preview_abs,
                    parts_from_main=parts_from_main,
                )
        raise FileNotFoundError(f"Р¤Р°Р№Р» РЅРµ Р±СѓР»Рѕ СЃС‚РІРѕСЂРµРЅРѕ: {output_file_abs}")

    task.set_output(primary_format, str(output_file_abs))
    if stl_preview_abs and stl_preview_abs.exists():
        task.set_output("stl", str(stl_preview_abs))
    parts_for_layout = stl_parts_from_preview or parts_from_main
    print_layout_3mf_abs = _create_print_layout_3mf(
        output_dir=output_dir,
        task_id=task_id,
        parts_from_main=parts_for_layout,
    )
    if print_layout_3mf_abs and print_layout_3mf_abs.exists():
        task.set_output("print_layout_3mf", str(print_layout_3mf_abs))
        if primary_format == "3mf":
            task.set_output("assembly_3mf", str(output_file_abs))

    package_abs = _create_print_package(
        output_dir=output_dir,
        task_id=task_id,
        primary_format=primary_format,
        output_file_abs=output_file_abs,
        stl_preview_abs=stl_preview_abs,
        parts_from_main=parts_from_main,
    )
    if package_abs and package_abs.exists():
        task.set_output("print_package", str(package_abs))

    task.complete(str(output_file_abs))
    task.update_status("completed", 100, completion_message)

    return ExportPipelineResult(
        output_file_abs=output_file_abs,
        primary_format=primary_format,
        stl_preview_abs=stl_preview_abs,
        parts_from_main=parts_from_main,
    )
