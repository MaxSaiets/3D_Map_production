from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from services.debug_renderers import render_mesh_top_png, render_overlay_png
from services.geometry_diagnostics import ensure_valid_geometry


@dataclass
class StageSnapshotCollector:
    task_id: str
    run_dir: Path
    zone_prefix: str = ""
    entries: list[dict[str, Any]] = field(default_factory=list)

    def _safe_rel(self, path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        try:
            return str(path.resolve().relative_to(self.run_dir.resolve()))
        except Exception:
            return str(path.resolve())

    def _register(self, stage: str, name: str, path: Optional[Path]) -> None:
        if path is None:
            return
        self.entries.append(
            {
                "stage": str(stage),
                "name": str(name),
                "path": self._safe_rel(path),
                "abs_path": str(path.resolve()),
            }
        )

    def _safe_overlay(self, stage: str, name: str, layers: list[dict[str, Any]]) -> Optional[Path]:
        out = (self.run_dir / stage / f"{name}.png").resolve()
        try:
            rendered = render_overlay_png(layers, out, background="white")
        except Exception as exc:
            print(f"[WARN] {self.zone_prefix}stage snapshot overlay skipped ({stage}/{name}): {exc}")
            return None
        if rendered is None:
            return None
        self._register(stage, name, rendered)
        return rendered

    def _safe_mesh(self, stage: str, name: str, mesh: Any) -> Optional[Path]:
        out = (self.run_dir / stage / f"{name}.png").resolve()
        try:
            rendered = render_mesh_top_png(mesh, out)
        except Exception as exc:
            print(f"[WARN] {self.zone_prefix}stage snapshot mesh skipped ({stage}/{name}): {exc}")
            return None
        if rendered is None:
            return None
        self._register(stage, name, rendered)
        return rendered

    def capture_canonical(self, canonical_mask_bundle: Any) -> None:
        if canonical_mask_bundle is None:
            return
        roads = ensure_valid_geometry(getattr(canonical_mask_bundle, "roads_final", None))
        roads_semantic = ensure_valid_geometry(getattr(canonical_mask_bundle, "roads_semantic_preview", None))
        buildings = ensure_valid_geometry(getattr(canonical_mask_bundle, "buildings_footprints", None))
        parks = ensure_valid_geometry(getattr(canonical_mask_bundle, "parks_final", None))
        water = ensure_valid_geometry(getattr(canonical_mask_bundle, "water_final", None))
        road_groove = ensure_valid_geometry(getattr(canonical_mask_bundle, "road_groove_mask", None))
        parks_groove = ensure_valid_geometry(getattr(canonical_mask_bundle, "parks_groove_mask", None))
        water_groove = ensure_valid_geometry(getattr(canonical_mask_bundle, "water_groove_mask", None))

        self._safe_overlay(
            "02_canonical_2d",
            "canonical_layers",
            [
                {"geometry": parks, "facecolor": "#7CB07A", "alpha": 0.8},
                {"geometry": water, "facecolor": "#6EA8D7", "alpha": 0.9},
                {"geometry": roads_semantic or roads, "facecolor": "#111111", "edgecolor": "#111111", "linewidth": 1.0, "alpha": 0.95},
                {"geometry": buildings, "facecolor": "#C8B48A", "alpha": 0.8},
            ],
        )
        self._safe_overlay(
            "02_canonical_2d",
            "canonical_grooves_vs_inserts",
            [
                {"geometry": road_groove, "facecolor": "#FFFFFF", "alpha": 0.95},
                {"geometry": parks_groove, "facecolor": "#FFFFFF", "alpha": 0.95},
                {"geometry": water_groove, "facecolor": "#FFFFFF", "alpha": 0.95},
                {"geometry": roads, "facecolor": "#111111", "alpha": 0.95},
                {"geometry": parks, "facecolor": "#7CB07A", "alpha": 0.75},
                {"geometry": water, "facecolor": "#6EA8D7", "alpha": 0.75},
            ],
        )

    def capture_terrain_stage(self, terrain_stage: Any) -> None:
        self._safe_mesh("03_terrain_stage", "terrain_before_detail", getattr(terrain_stage, "terrain_mesh", None))

    def capture_detail_stage(self, detail_layers: Any) -> None:
        roads = ensure_valid_geometry(
            getattr(getattr(detail_layers, "road_result", None), "source_polygons", None)
            or getattr(detail_layers, "road_cut_source", None)
        )
        parks = ensure_valid_geometry(getattr(getattr(detail_layers, "parks_result", None), "processed_polygons", None))
        water = ensure_valid_geometry(getattr(detail_layers, "water_cut_polygons", None))
        buildings = ensure_valid_geometry(getattr(detail_layers, "building_footprints", None))

        self._safe_mesh("04_detail_layers", "terrain_after_detail", getattr(detail_layers, "terrain_mesh", None))
        self._safe_mesh("04_detail_layers", "roads_mesh", getattr(detail_layers, "road_mesh", None))
        self._safe_mesh("04_detail_layers", "parks_mesh", getattr(detail_layers, "parks_mesh", None))
        self._safe_mesh("04_detail_layers", "water_mesh", getattr(detail_layers, "water_mesh", None))
        self._safe_overlay(
            "04_detail_layers",
            "detail_masks_overlay",
            [
                {"geometry": parks, "facecolor": "#7CB07A", "alpha": 0.75},
                {"geometry": water, "facecolor": "#6EA8D7", "alpha": 0.9},
                {"geometry": roads, "facecolor": "#111111", "alpha": 0.9},
                {"geometry": buildings, "facecolor": "#C8B48A", "alpha": 0.8},
            ],
        )

    def capture_postprocess_stage(self, postprocess_result: Any) -> None:
        self._safe_mesh("05_postprocess", "terrain_after_postprocess", getattr(postprocess_result, "terrain_mesh", None))
        self._safe_mesh("05_postprocess", "roads_after_postprocess", getattr(postprocess_result, "road_mesh", None))
        self._safe_mesh("05_postprocess", "parks_after_postprocess", getattr(postprocess_result, "parks_mesh", None))
        self._safe_mesh("05_postprocess", "water_after_postprocess", getattr(postprocess_result, "water_mesh", None))

    def capture_clip_stage(self, clip_result: Any) -> None:
        self._safe_mesh("06_clip", "terrain_after_clip", getattr(clip_result, "terrain_mesh", None))
        self._safe_mesh("06_clip", "roads_after_clip", getattr(clip_result, "road_mesh", None))
        self._safe_mesh("06_clip", "parks_after_clip", getattr(clip_result, "parks_mesh", None))
        self._safe_mesh("06_clip", "water_after_clip", getattr(clip_result, "water_mesh", None))

    def capture_merge_stage(self, merge_result: Any) -> None:
        self._safe_mesh("07_merge", "base_final", getattr(merge_result, "terrain_mesh", None))

    def capture_export_stage(self, export_result: Any) -> None:
        out = getattr(export_result, "output_file_abs", None)
        if out is None:
            return
        self.entries.append(
            {
                "stage": "08_export",
                "name": "primary_output",
                "path": str(out),
                "abs_path": str(Path(out).resolve()),
            }
        )

    def finalize(self) -> Path:
        manifest = {
            "task_id": self.task_id,
            "run_dir": str(self.run_dir.resolve()),
            "snapshots": self.entries,
        }
        manifest_path = (self.run_dir / "manifest.json").resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest_path


def create_stage_snapshot_collector(*, task_id: str, debug_root: Path, zone_prefix: str = "") -> StageSnapshotCollector:
    run_dir = (debug_root / "stage_snapshots" / task_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return StageSnapshotCollector(task_id=task_id, run_dir=run_dir, zone_prefix=zone_prefix)
