# -*- coding: utf-8 -*-
"""Нічний пакет: 3D-модель + студійний рендер для КОЖНОЇ сторінки міста/району.

Черга: 23 обласні центри → 55 адмін-районів → 65 міст другого кола
(джерела: frontend/lib/templates.ts, lib/cityRaions.ts, lib/uaCities2.ts).
Для кожної точки: POST /api/generate на ЛОКАЛЬНИЙ бекенд (мапа 80 мм, 1:10 000 =
ділянка 800×800 м навколо центру) → 3MF → Blender EEVEE (-b, 2 потоки, IDLE-пріоритет)
→ webp 640×480 і 400×300 у frontend/public/maps-renders/{id}.webp.

Спокійно й поетапно: строго по одній задачі, пауза між задачами (ввічливо до Overpass),
готові рендери пропускаються (можна перезапускати скільки завгодно), збій однієї
точки не зупиняє чергу. Наприкінці пише lib/mapRenders.ts — список готових id.

Запуск:  python tools/night_city_renders.py [--api http://127.0.0.1:8001] [--limit N] [--only kyiv,lviv]
"""
import argparse, json, math, re, shutil, subprocess, sys, time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FE = ROOT / "frontend"
OUT = FE / "public" / "maps-renders"
WORK = Path(r"D:\3dmap_tmp\city_renders")
BLENDER = Path(r"D:\Soft\Blender\blender.exe")
RENDER_PY = Path(r"D:\3dmap_tmp\mnt_src\render_product.py")
BACKEND_OUTPUT = ROOT / "backend" / "output"
LOG = WORK / "night.log"
IDLE = 0x00000040  # IDLE_PRIORITY_CLASS


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def targets():
    t = []
    src = (FE / "lib" / "templates.ts").read_text(encoding="utf-8")
    for key, lat, lon in re.findall(r'\{ key: "(\w+)",\s*label: "[^"]+",\s*center: \[([\d.]+), ([\d.]+)\]', src):
        slug = "ivano-frankivsk" if key == "IvanoFrankivsk" else key.lower().replace("_", "-")
        t.append((slug, float(lat), float(lon)))
    src = (FE / "lib" / "cityRaions.ts").read_text(encoding="utf-8")
    for city, slug, lat, lon in re.findall(r'citySlug: "([\w-]+)", slug: "([\w-]+)",.*?center: \[([\d.]+), ([\d.]+)\]', src):
        t.append((f"{city}--{slug}", float(lat), float(lon)))
    src = (FE / "lib" / "uaCities2.ts").read_text(encoding="utf-8")
    for slug, lat, lon in re.findall(r'slug: "([\w-]+)", center: \[([\d.]+), ([\d.]+)\]', src):
        t.append((slug, float(lat), float(lon)))
    # 27.09.2026: відомі вулиці (lib/cityStreets.ts) — наступний прохід після міст/районів.
    streets = FE / "lib" / "cityStreets.ts"
    if streets.exists():
        src = streets.read_text(encoding="utf-8")
        for city, slug, lat, lon in re.findall(r'citySlug: "([\w-]+)", slug: "([\w-]+)",.*?center: \[([\d.]+), ([\d.]+)\]', src):
            t.append((f"{city}--{slug}", float(lat), float(lon)))
    return t


def map_req(lat, lon, size=80):
    half = size * 10 / 2
    dlat = half / 111320
    dlon = half / (111320 * math.cos(math.radians(lat)))
    return {"north": lat + dlat, "south": lat - dlat, "east": lon + dlon, "west": lon - dlon,
            "road_width_multiplier": 0.8, "road_height_mm": 0.5, "road_embed_mm": 0.3,
            "building_min_height": 5.0, "building_height_multiplier": 1.8, "building_foundation_mm": 0.6,
            "building_embed_mm": 0.2, "water_depth": 2.0, "terrain_enabled": True, "terrain_z_scale": 1.0,
            "terrain_base_thickness_mm": 1.3, "flat_uniform_building_height": False, "color_palette": "classic",
            "terrain_resolution": 180, "terrarium_zoom": 15, "flatten_buildings_on_terrain": False,
            "flatten_roads_on_terrain": False, "export_format": "3mf", "model_size_mm": size,
            "context_padding_m": 400.0, "is_ams_mode": False, "flat_plate_mode": False, "preview_mode": False,
            "preview_include_base": True, "preview_include_roads": True, "preview_include_buildings": True,
            "preview_include_water": True, "preview_include_parks": True}


def generate(api, rid, lat, lon):
    r = requests.post(f"{api}/api/generate", json=map_req(lat, lon), timeout=60)
    r.raise_for_status()
    tid = r.json()["task_id"]
    t0 = time.time()
    while True:
        time.sleep(6)
        s = requests.get(f"{api}/api/status/{tid}", timeout=30).json()
        if s.get("status") in ("completed", "failed", "error", "cancelled"):
            break
        if time.time() - t0 > 20 * 60:
            raise RuntimeError("timeout 20 min")
    if s["status"] != "completed":
        raise RuntimeError(f"status={s['status']} {s.get('message')}")
    name = Path(s.get("download_url_3mf") or s["download_url"]).name
    dst = WORK / f"{rid}.3mf"
    shutil.copy(BACKEND_OUTPUT / name, dst)
    return dst, round(time.time() - t0)


def render(src3mf, rid):
    png = WORK / f"{rid}.png"
    cmd = [str(BLENDER), "-b", "--threads", "2", "-P", str(RENDER_PY), "--", str(src3mf), str(png), "48"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=15 * 60, creationflags=IDLE)
    if not png.exists():
        raise RuntimeError("blender: " + (p.stderr or p.stdout)[-400:])
    im = Image.open(png).convert("RGB")
    w, h = im.size
    im = im.resize((640, 480), Image.LANCZOS)
    im.save(OUT / f"{rid}.webp", "WEBP", quality=84, method=6)
    im.resize((400, 300), Image.LANCZOS).save(OUT / f"{rid}-400.webp", "WEBP", quality=82, method=6)


def write_manifest():
    ids = sorted(p.stem for p in OUT.glob("*.webp") if not p.stem.endswith("-400"))
    body = ",\n".join(f'  "{i}"' for i in ids)
    (FE / "lib" / "mapRenders.ts").write_text(
        "/**\n * Готові студійні рендери 3D-моделей для сторінок міст/районів\n"
        " * (/maps-renders/{id}.webp, id = slug міста або «{місто}--{район}»).\n"
        " * ФАЙЛ ГЕНЕРУЄ tools/night_city_renders.py — не правити вручну.\n */\n"
        f"export const MAP_RENDERS: ReadonlySet<string> = new Set([\n{body}{',' if ids else ''}\n]);\n",
        encoding="utf-8")
    return len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8001")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--pause", type=int, default=20)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    todo = targets()
    if a.only:
        keep = set(a.only.split(","))
        todo = [x for x in todo if x[0] in keep]
    log(f"START {len(todo)} точок")
    done = fail = 0
    for rid, lat, lon in todo:
        if (OUT / f"{rid}.webp").exists():
            continue
        if a.limit and done + fail >= a.limit:
            break
        try:
            src, sec = generate(a.api, rid, lat, lon)
            render(src, rid)
            done += 1
            log(f"OK   {rid} (генерація {sec} с)")
        except Exception as e:  # одна точка не зупиняє ніч
            fail += 1
            log(f"FAIL {rid}: {e}")
        write_manifest()
        time.sleep(a.pause)
    n = write_manifest()
    log(f"END ok={done} fail={fail} усього рендерів={n}")


if __name__ == "__main__":
    main()
