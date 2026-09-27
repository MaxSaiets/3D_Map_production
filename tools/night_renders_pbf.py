# -*- coding: utf-8 -*-
"""Нічні рендери БЕЗ Overpass (28.09+): піднімає локальний бекенд у режимі OSM_SOURCE=pbf
(OSM_PBF_PATH = D:\3dmap_tmp\osm\current.osm.pbf, куди night_city_renders --pbf-cuts
перед кожною точкою копіює вирізку міста), проганяє чергу двічі й гасить бекенд.
Не дає Windows заснути, поки працює. Лог: D:\3dmap_tmp\night_pbf.log
Запуск: python tools/night_renders_pbf.py [--delay 0]"""
import argparse, os, subprocess, sys, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
LOG = Path(r"D:\3dmap_tmp\night_pbf.log")


def log(m):
    line = f"{time.strftime('%d.%m %H:%M:%S')} {m}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f: f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--delay", type=int, default=0); a = ap.parse_args()
    try:
        import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception: pass
    time.sleep(a.delay)
    # 27.09: режим OSM_SOURCE=pbf застарілий (порожні моделі) → та сама локальна ukraine.duckdb, що й на проді.
    env = dict(os.environ, RESULT_CACHE="0", OSM_DUCKDB_PATH=r"D:\3dmap_tmp\osm\ukraine.duckdb",
               PYTHONIOENCODING="utf-8")
    be = subprocess.Popen([str(ROOT / "backend" / "venv" / "Scripts" / "python.exe"), "-m", "uvicorn", "main:app",
                           "--host", "127.0.0.1", "--port", "8001"], cwd=ROOT / "backend", env=env,
                          stdout=open(r"D:\3dmap_tmp\backend_pbf.log", "a", encoding="utf-8"), stderr=subprocess.STDOUT)
    log(f"бекенд pbf pid {be.pid}")
    for _ in range(60):
        try:
            if requests.get("http://127.0.0.1:8001/api/health", timeout=3).ok: break
        except Exception: time.sleep(3)
    try:
        for i in (1, 2):
            log(f"прохід {i}")
            subprocess.run([sys.executable, "tools/night_city_renders.py", "--pause", "5"], cwd=ROOT,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    finally:
        be.terminate()
        log("готово, бекенд зупинено")


if __name__ == "__main__":
    main()
