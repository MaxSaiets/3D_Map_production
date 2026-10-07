"""Локальний прогін мапи НЕпрямокутної форми (серце / коло / пазл) — як /create на проді.

Полігон будується тим самим рівнянням, що й frontend/components/MapSelector.tsx
(shapeOutlinePoints). За замовчуванням — серце біля Палацу спорту (prod task 75fb4e9f,
07.10.2026), плоска основа, 80 мм.

  venv/Scripts/python.exe ../tools/run_shape_zone.py --shape heart [--preview]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

# bbox prod-задачі 75fb4e9f (Київ, Палац спорту)
DEFAULT_BBOX = dict(north=50.4331098599337, south=50.428079317353756,
                    east=30.519122721515238, west=30.511225634925076)


def shape_outline_m(shape: str, w: float, h: float) -> list[tuple[float, float]]:
    if shape == "circle":
        r = min(w, h) / 2
        return [(math.cos(2 * math.pi * i / 40) * r, math.sin(2 * math.pi * i / 40) * r) for i in range(40)]
    if shape == "heart":
        raw = []
        for i in range(160):
            t = 2 * math.pi * i / 160
            raw.append((16 * math.sin(t) ** 3,
                        13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)))
        xs, ys = [p[0] for p in raw], [p[1] for p in raw]
        s = min(w / (max(xs) - min(xs)), h / (max(ys) - min(ys)))
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        return [((x - cx) * s, (y - cy) * s) for x, y in raw]
    if shape == "rect":
        return [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    raise SystemExit(f"unknown shape {shape}")


def shape_polygon_lonlat(shape: str, bbox: dict) -> list[list[float]]:
    lat0 = (bbox["north"] + bbox["south"]) / 2
    lon0 = (bbox["east"] + bbox["west"]) / 2
    m_lon = 111_320 * math.cos(math.radians(lat0))
    w = (bbox["east"] - bbox["west"]) * m_lon
    h = (bbox["north"] - bbox["south"]) * 111_320
    pts = [[lon0 + x / m_lon, lat0 + y / 111_320] for x, y in shape_outline_m(shape, w, h)]
    return pts + [pts[0]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="heart", choices=["heart", "circle", "rect"])
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--terrain", action="store_true", help="рельєф увімкнено (на проді для цієї мапи — вимкнено)")
    ap.add_argument("--size", type=float, default=80.0)
    args = ap.parse_args()

    from main import GenerationRequest, generate_model_task, tasks  # noqa: E402
    from services.generation_task import GenerationTask  # noqa: E402

    poly = shape_polygon_lonlat(args.shape, DEFAULT_BBOX)
    lons, lats = [p[0] for p in poly], [p[1] for p in poly]
    request = GenerationRequest(
        north=max(lats), south=min(lats), east=max(lons), west=min(lons),
        zone_polygon_coords=poly,
        terrain_enabled=bool(args.terrain),
        model_size_mm=args.size,
        preview_mode=bool(args.preview),
        export_format="3mf",
    )
    task_id = f"shape_{args.shape}_{uuid.uuid4().hex[:8]}"
    tasks[task_id] = GenerationTask(task_id=task_id, request=request)
    generate_model_task(task_id=task_id, request=request, zone_polygon_coords=poly)
    task = tasks[task_id]
    print(json.dumps({"task_id": task_id, "status": task.status, "output_file": task.output_file,
                      "output_files": task.output_files, "error": task.error}, indent=2, ensure_ascii=False))
    return 0 if task.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
