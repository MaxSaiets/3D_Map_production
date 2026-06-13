#!/usr/bin/env python3
"""Нічні GOLDEN-тести геометрії monadruk.

Генерує фіксовані еталонні моделі через локальний бекенд, рахує метрики
(кількість частин, обʼєми, watertight, bbox) і порівнює з еталоном
deploy/golden_baseline.json. Розбіжність або падіння генерації → Telegram-алерт.

Перший запуск (еталона нема) — створює еталон і повідомляє про це.
Оновити еталон свідомо: видалити golden_baseline.json і запустити вручну.

Запуск: /opt/3dmap/backend/venv/bin/python /opt/3dmap/deploy/golden_check.py
Env: TG_BOT_TOKEN, TG_CHAT_ID (з deploy/.health.env), GOLDEN_API (default http://127.0.0.1:8000)
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request

API = os.environ.get("GOLDEN_API", "http://127.0.0.1:8000")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_PATH = os.path.join(BASE_DIR, "golden_baseline.json")
OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "backend", "output"))
VOL_TOL = 0.05          # ±5% на обʼєми (дрібні правки OSM допустимі)
POLL_TIMEOUT_S = 1200   # 20 хв на кейс
POLL_STEP_S = 15

# Фіксовані кейси: київський центр (локальна DuckDB → стабільні дані).
CASES = [
    {
        "id": "keychain_heart_labels",
        "body": {
            "north": 50.4510, "south": 50.4492, "east": 30.5251, "west": 30.5220,
            "terrain_enabled": False, "terrain_base_thickness_mm": 1.5,
            "terrain_resolution": 120, "terrarium_zoom": 13,
            "export_format": "3mf", "model_size_mm": 46, "context_padding_m": 35,
            "flat_plate_mode": True, "keychain_mode": True,
            "keychain_label": "KYIV", "keychain_label2": "50.4501, 30.5236",
            "keychain_back_label": "GOLDEN",
            "keychain_base_shape": "heart",
            "keychain_body_width_mm": 46, "keychain_body_height_mm": 42,
            "keychain_map_width_mm": 46, "keychain_map_height_mm": 42,
            "keychain_loop_center_x_mm": 23, "keychain_loop_center_y_mm": 1.5,
            "keychain_label_center_x_mm": 23, "keychain_label_center_y_mm": 25,
            "keychain_loop_outer_radius_mm": 4, "keychain_loop_inner_radius_mm": 2,
            "keychain_corner_radius_mm": 0, "keychain_label_band_height_mm": 5,
            "keychain_label_text_height_mm": 3.0, "keychain_label_width_mm": 22,
            "keychain_rim_width_mm": 1.2, "keychain_rim_height_mm": 0.45,
            "flat_water_layer_mm": 0.22, "flat_roads_layer_mm": 0.42, "flat_parks_layer_mm": 0.36,
        },
    },
    {
        "id": "magnet_map_label",
        "body": {
            "north": 50.4512, "south": 50.4494, "east": 30.5250, "west": 30.5222,
            "terrain_enabled": False, "terrain_base_thickness_mm": 3.0,
            "terrain_resolution": 120, "terrarium_zoom": 13,
            "export_format": "3mf", "model_size_mm": 60, "context_padding_m": 100,
            "flat_plate_mode": True, "magnet_pocket": True, "map_label": "KYIV",
            "flat_water_layer_mm": 0.22, "flat_roads_layer_mm": 0.42, "flat_parks_layer_mm": 0.36,
        },
    },
    # 2026-06-13: захист нової геометрії цієї серії (раніше не покрита).
    {
        # Серце-пара для закоханих: half-heart + замок (SH-кліп + knob).
        "id": "heart_pair_left",
        "body": {
            "north": 50.4510, "south": 50.4492, "east": 30.5246, "west": 30.5224,
            "terrain_enabled": False, "terrain_base_thickness_mm": 1.5,
            "terrain_resolution": 120, "terrarium_zoom": 13,
            "export_format": "3mf", "model_size_mm": 44, "context_padding_m": 35,
            "flat_plate_mode": True, "keychain_mode": True, "keychain_label": "L",
            "keychain_base_shape": "heart-l",
            "keychain_body_width_mm": 30, "keychain_body_height_mm": 44,
            "keychain_map_width_mm": 30, "keychain_map_height_mm": 44,
            "keychain_loop_center_x_mm": 15, "keychain_loop_center_y_mm": 0,
            "keychain_label_center_x_mm": 15, "keychain_label_center_y_mm": 33,
            "keychain_loop_outer_radius_mm": 5.5, "keychain_loop_inner_radius_mm": 2.6,
            "keychain_corner_radius_mm": 0, "keychain_label_band_height_mm": 5,
            "keychain_label_text_height_mm": 2.8, "keychain_label_width_mm": 16,
            "keychain_rim_width_mm": 1.6, "keychain_rim_height_mm": 0.6,
            "flat_water_layer_mm": 0.22, "flat_roads_layer_mm": 0.42, "flat_parks_layer_mm": 0.36,
        },
    },
    {
        # Пазл-L: виступ (knob) на правій грані — long-distance пара.
        "id": "puzzle_left",
        "body": {
            "north": 50.4513, "south": 50.4491, "east": 30.5248, "west": 30.5222,
            "terrain_enabled": False, "terrain_base_thickness_mm": 1.5,
            "terrain_resolution": 120, "terrarium_zoom": 13,
            "export_format": "3mf", "model_size_mm": 42, "context_padding_m": 35,
            "flat_plate_mode": True, "keychain_mode": True, "keychain_label": "L",
            "keychain_base_shape": "puzzle-l",
            "keychain_body_width_mm": 40, "keychain_body_height_mm": 42,
            "keychain_map_width_mm": 40, "keychain_map_height_mm": 42,
            "keychain_loop_center_x_mm": 20, "keychain_loop_center_y_mm": 0,
            "keychain_label_center_x_mm": 20, "keychain_label_center_y_mm": 37.5,
            "keychain_loop_outer_radius_mm": 5.5, "keychain_loop_inner_radius_mm": 2.6,
            "keychain_corner_radius_mm": 5, "keychain_label_band_height_mm": 6,
            "keychain_label_text_height_mm": 3.2, "keychain_label_width_mm": 30,
            "keychain_rim_width_mm": 1.6, "keychain_rim_height_mm": 0.6,
            "flat_water_layer_mm": 0.22, "flat_roads_layer_mm": 0.42, "flat_parks_layer_mm": 0.36,
        },
    },
    {
        # Магніт із 4 кишенями під шайби Ø4×2мм (новий multi-pocket режим).
        "id": "magnet_four_pockets",
        "body": {
            "north": 50.4512, "south": 50.4494, "east": 30.5250, "west": 30.5222,
            "terrain_enabled": False, "terrain_base_thickness_mm": 3.0,
            "terrain_resolution": 120, "terrarium_zoom": 13,
            "export_format": "3mf", "model_size_mm": 60, "context_padding_m": 100,
            "flat_plate_mode": True, "magnet_pocket": True, "map_label": "KYIV",
            "magnet_pocket_diameter_mm": 4.4, "magnet_pocket_depth_mm": 2.1,
            "magnet_pocket_count": 4, "magnet_pocket_inset_mm": 8,
            "flat_water_layer_mm": 0.22, "flat_roads_layer_mm": 0.42, "flat_parks_layer_mm": 0.36,
        },
    },
    {
        # GPX-трек на брелку → шар Track (підвищена лінія поверх карти).
        "id": "keychain_gpx_track",
        "body": {
            "north": 50.4512, "south": 50.4494, "east": 30.5248, "west": 30.5222,
            "terrain_enabled": False, "terrain_base_thickness_mm": 1.5,
            "terrain_resolution": 120, "terrarium_zoom": 13,
            "export_format": "3mf", "model_size_mm": 42, "context_padding_m": 35,
            "flat_plate_mode": True, "keychain_mode": True, "keychain_label": "RUN",
            "keychain_base_shape": "rounded",
            "keychain_body_width_mm": 42, "keychain_body_height_mm": 42,
            "keychain_map_width_mm": 42, "keychain_map_height_mm": 42,
            "keychain_loop_center_x_mm": 21, "keychain_loop_center_y_mm": 0,
            "keychain_label_center_x_mm": 21, "keychain_label_center_y_mm": 35,
            "keychain_loop_outer_radius_mm": 5.5, "keychain_loop_inner_radius_mm": 2.6,
            "keychain_corner_radius_mm": 5, "keychain_label_band_height_mm": 6,
            "keychain_label_text_height_mm": 3.0, "keychain_label_width_mm": 24,
            "keychain_rim_width_mm": 1.6, "keychain_rim_height_mm": 0.6,
            "flat_water_layer_mm": 0.22, "flat_roads_layer_mm": 0.42, "flat_parks_layer_mm": 0.36,
            "gpx_track": [
                [30.5226, 50.4496], [30.5230, 50.4499], [30.5234, 50.4502],
                [30.5238, 50.4505], [30.5242, 50.4508], [30.5245, 50.4510],
            ],
        },
    },
]


def tg_alert(text: str) -> None:
    token = os.environ.get("TG_BOT_TOKEN", "")
    chat = os.environ.get("TG_CHAT_ID", "")
    if not token or not chat:
        print(f"[GOLDEN] TG not configured; alert text:\n{text}")
        return
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
        urllib.request.urlopen(
            f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=20
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[GOLDEN] TG send failed: {exc}")


def api_json(path: str, payload: dict | None = None) -> dict:
    url = f"{API}{path}"
    if payload is None:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def run_case(case: dict) -> dict:
    """Генерує кейс і повертає метрики. Кидає виняток при падінні."""
    resp = api_json("/api/generate", case["body"])
    task_id = resp["task_id"]
    deadline = time.time() + POLL_TIMEOUT_S
    status = {}
    while time.time() < deadline:
        time.sleep(POLL_STEP_S)
        status = api_json(f"/api/status/{task_id}")
        if status.get("status") in ("completed", "failed"):
            break
    if status.get("status") != "completed":
        raise RuntimeError(f"generation not completed: {status.get('status')} ({status.get('message')})")

    url = status.get("download_url") or ""
    fname = os.path.basename(url)
    path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(path):
        raise RuntimeError(f"output file not found: {path}")

    import trimesh  # з venv бекенда
    scene = trimesh.load(path)
    parts = {}
    for name, mesh in scene.geometry.items():
        parts[name] = {
            "volume": round(float(mesh.volume), 1),
            "watertight": bool(mesh.is_watertight),
        }
    return {"parts": parts, "file": fname}


def compare(case_id: str, got: dict, want: dict) -> list[str]:
    problems: list[str] = []
    got_parts, want_parts = got["parts"], want["parts"]
    missing = set(want_parts) - set(got_parts)
    extra = set(got_parts) - set(want_parts)
    if missing:
        problems.append(f"{case_id}: зникли частини {sorted(missing)}")
    if extra:
        problems.append(f"{case_id}: зайві частини {sorted(extra)}")
    for name in set(want_parts) & set(got_parts):
        w, g = want_parts[name], got_parts[name]
        if w["watertight"] and not g["watertight"]:
            problems.append(f"{case_id}/{name}: втрачено watertight")
        wv, gv = float(w["volume"]), float(g["volume"])
        if wv > 0 and abs(gv - wv) / wv > VOL_TOL:
            problems.append(f"{case_id}/{name}: обʼєм {gv:.0f} проти еталона {wv:.0f} (>±{VOL_TOL*100:.0f}%)")
    return problems


def main() -> int:
    results: dict = {}
    failures: list[str] = []
    for case in CASES:
        try:
            print(f"[GOLDEN] running {case['id']}...")
            results[case["id"]] = run_case(case)
            print(f"[GOLDEN] {case['id']} OK: {json.dumps(results[case['id']]['parts'])}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{case['id']}: ГЕНЕРАЦІЯ ВПАЛА — {exc}")

    if not os.path.exists(BASELINE_PATH):
        if failures:
            tg_alert("🟠 GOLDEN: перший запуск, але є падіння:\n" + "\n".join(failures))
            return 1
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        tg_alert("🟢 GOLDEN: еталон геометрії створено (perші нічні метрики зафіксовано).")
        return 0

    with open(BASELINE_PATH, encoding="utf-8") as f:
        baseline = json.load(f)

    problems = list(failures)
    for case_id, got in results.items():
        want = baseline.get(case_id)
        if want is None:
            problems.append(f"{case_id}: нема в еталоні (онови baseline)")
            continue
        problems.extend(compare(case_id, got, want))

    if problems:
        tg_alert(
            "🔴 GOLDEN-тести геометрії: відхилення!\n"
            + "\n".join(problems[:12])
            + "\n\nЯкщо це свідома зміна пайплайна або правки OSM — видали "
              "/opt/3dmap/deploy/golden_baseline.json і запусти скрипт вручну для нового еталона."
        )
        return 1
    print("[GOLDEN] all cases match baseline ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
