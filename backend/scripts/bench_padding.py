# -*- coding: utf-8 -*-
"""Чи впливає розмір буфера OSM-фетчу на геометрію? Два прогони тієї самої зони
через ЖИВИЙ бекенд (uvicorn у підпроцесі) з різними OSM_ROAD_PADDING_DEG /
OSM_LOADER_PADDING_DEG → sha256 усіх вихідних файлів задачі + час fetch_source.

    venv/Scripts/python scripts/bench_padding.py            # Відень, превʼю 80 мм
    venv/Scripts/python scripts/bench_padding.py --print    # друкарський 3MF

Кеші вимкнені (RESULT_CACHE=0, OSM_DATA_CACHE_ENABLED=0), тож обидва прогони
реально ходять в Overpass.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8123
# Відень, ~800 м навколо Штефансдому (80 мм × 1:10000)
BODY = {
    "north": 48.2122, "south": 48.2050, "east": 16.3790, "west": 16.3682,
    "road_width_multiplier": 0.8, "road_height_mm": 0.5, "road_embed_mm": 0.3,
    "building_min_height": 5.0, "building_height_multiplier": 1.8,
    "building_foundation_mm": 0.6, "building_embed_mm": 0.2, "water_depth": 2.0,
    "terrain_enabled": True, "terrain_z_scale": 1.0, "terrain_base_thickness_mm": 1.3,
    "color_palette": "classic", "terrain_resolution": 180, "terrarium_zoom": 15,
    "flatten_buildings_on_terrain": False, "flatten_roads_on_terrain": False,
    "export_format": "3mf", "model_size_mm": 80, "context_padding_m": 400.0,
    "is_ams_mode": False, "flat_plate_mode": False, "preview_mode": True,
    "preview_include_base": True, "preview_include_roads": True,
    "preview_include_buildings": True, "preview_include_water": True, "preview_include_parks": True,
}


def _post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=60) as r:
        return json.loads(r.read())


def run(road_pad: float, loader_pad: float, log_path: Path, print_mode: bool) -> dict:
    env = dict(os.environ)
    env.update({
        "OSM_ROAD_PADDING_DEG": str(road_pad), "OSM_LOADER_PADDING_DEG": str(loader_pad),
        "RESULT_CACHE": "0", "OSM_DATA_CACHE_ENABLED": "0", "PYTHONIOENCODING": "utf-8",
        # без буферизації (інакше [TIMING] губиться при terminate) і без meshopt
        # (trimesh не читає стиснутий GLB — а нам треба хеш геометрії)
        "PYTHONUNBUFFERED": "1", "PREVIEW_MESHOPT": "0",
    })
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", str(PORT), "--log-level", "warning"],
        cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(120):
            try:
                _get("/api/health")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("бекенд не піднявся")
        body = dict(BODY)
        if print_mode:
            body["preview_mode"] = False
        t0 = time.time()
        r = _post("/api/generate", body)
        tid = r["task_id"]
        while True:
            st = _get(f"/api/status/{tid}")
            if st["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(3)
        total = time.time() - t0
        if st["status"] != "completed":
            raise RuntimeError(f"задача {st['status']}: {st.get('message')}")
        files = {}
        seen = set()
        for key in ("download_url", "download_url_3mf", "download_url_glb", "download_url_stl"):
            u = st.get(key)
            if u:
                p = ROOT / "output" / Path(u).name
                if p.exists() and p.name not in seen:
                    seen.add(p.name)
                    files[p.name] = geom_hash(p)
        # per-part файли поруч (model_<size>_<task8>_<hash8>*.ext)
        stem = Path(st.get("download_url") or "").stem
        if stem:
            for p in sorted((ROOT / "output").glob(stem.rsplit("_", 1)[0] + "*")):
                if p.name not in seen and p.suffix.lower() in (".glb", ".3mf", ".stl"):
                    seen.add(p.name)
                    files[p.name] = geom_hash(p)
        return {"task": tid, "total_s": round(total, 1), "files": files}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        log.close()


def geom_hash(path: Path) -> str:
    """sha256 вершин+граней кожної частини (байти файлу містять метадані/час)."""
    import numpy as np
    import trimesh

    try:
        obj = trimesh.load(str(path), force="scene")
    except Exception as exc:  # noqa: BLE001
        return f"load-error:{exc}"[:40]
    h = hashlib.sha256()
    geoms = obj.geometry.values() if hasattr(obj, "geometry") else [obj]
    parts = []
    for g in geoms:
        if not hasattr(g, "vertices"):
            continue
        v = np.round(np.asarray(g.vertices, dtype=np.float64), 4)
        f = np.asarray(g.faces, dtype=np.int64)
        parts.append((len(v), len(f), hashlib.sha256(v.tobytes() + f.tobytes()).hexdigest()[:10]))
    for p in sorted(parts):
        h.update(repr(p).encode())
    return f"{len(parts)}p:" + h.hexdigest()[:12]


def fetch_time(log_path: Path):
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.findall(r"\[TIMING\] fetch_source: ([\d.]+)s", txt)
    els = re.findall(r"\[BUNDLE\].*?(\d+) елементів за ([\d.]+) с", txt)
    return (float(m[-1]) if m else None, els[-1] if els else None)


def main():
    print_mode = "--print" in sys.argv
    out = ROOT / "scratch_bench"
    out.mkdir(exist_ok=True)
    # --tight=0.004,0.002 — інші «вузькі» значення
    tight = (0.002, 0.001)
    for a in sys.argv:
        if a.startswith("--tight="):
            rp, lp = a.split("=", 1)[1].split(",")
            tight = (float(rp), float(lp))
    configs = [("default", 0.01, 0.005), ("tight", tight[0], tight[1])]
    res = {}
    for name, rp, lp in configs:
        log = out / f"padding_{name}.log"
        print(f"=== {name}: road={rp} loader={lp} ===", flush=True)
        res[name] = run(rp, lp, log, print_mode)
        res[name]["fetch"] = fetch_time(log)
        print(json.dumps(res[name], ensure_ascii=False), flush=True)
    a, b = res["default"], res["tight"]
    print("\n=== ПІДСУМОК ===")
    print(f"час: default {a['total_s']} с (fetch {a['fetch']}) vs tight {b['total_s']} с (fetch {b['fetch']})")
    fa = {re.sub(r"_[0-9a-f]{8}_[0-9a-f]{8}", "_X", k): v for k, v in a["files"].items()}
    fb = {re.sub(r"_[0-9a-f]{8}_[0-9a-f]{8}", "_X", k): v for k, v in b["files"].items()}
    same = True
    for k in sorted(set(fa) | set(fb)):
        ok = fa.get(k) == fb.get(k)
        same &= ok
        print(f"{'OK ' if ok else 'DIFF'} {k}: {fa.get(k)} vs {fb.get(k)}")
    print("РЕЗУЛЬТАТ:", "ІДЕНТИЧНО" if same else "Є РОЗБІЖНОСТІ")


if __name__ == "__main__":
    main()
