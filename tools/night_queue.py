# -*- coding: utf-8 -*-
"""Нічний оркестратор (27.09.2026): усе СТРОГО ПОСЛІДОВНО щодо Overpass.

Урок першої ночі: збір OSM і генерація моделей паралельно → Overpass відмовив
(«джерело карт тимчасово недоступне») і 130 з 143 точок упали. Тому:
  0) затримка старту (--delay, типово 30 хв);
  1) дані OSM для сторінок (osm_page_stats.py), два проходи (другий добирає збої);
  2) ШІ-тексти (ai_page_texts.py) — Gemini не чіпає Overpass, тож стартують ПАРАЛЕЛЬНО
     з кроком 3, щойно є дані OSM; самі чекають на квоту;
  3) моделі й рендери (night_city_renders.py): міста → райони → міста 2-го кола → вулиці,
     два проходи; кожна точка з повторами при перевантаженні.
Лог: D:\\3dmap_tmp\\night_queue.log
"""
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = Path(r"D:\3dmap_tmp\night_queue.log")
PY = sys.executable


def log(m):
    line = f"{time.strftime('%d.%m %H:%M:%S')} {m}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args, out_name):
    out = open(Path(r"D:\3dmap_tmp") / out_name, "a", encoding="utf-8")
    log("RUN " + " ".join(args))
    p = subprocess.run([PY, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    log(f"DONE rc={p.returncode} {args[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=int, default=1800)
    a = ap.parse_args()
    # Не давати Windows заснути, поки черга працює (ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
    # налаштування живлення не змінюються, запит знімається разом із процесом.
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        log("режим «не засинати» увімкнено на час черги")
    except Exception as e:  # noqa: BLE001
        log(f"не вдалося увімкнути «не засинати»: {e}")
    log(f"START, затримка {a.delay // 60} хв")
    time.sleep(a.delay)
    for i in (1, 2):
        run(["tools/osm_page_stats.py", "--pause", "15"], "q_osm.log")
    ai = subprocess.Popen([PY, "tools/ai_page_texts.py", "--pause", "12"], cwd=ROOT,
                          stdout=open(r"D:\3dmap_tmp\q_ai.log", "a", encoding="utf-8"), stderr=subprocess.STDOUT)
    log("AI тексти стартували паралельно (pid %s)" % ai.pid)
    for i in (1, 2):
        run(["tools/night_city_renders.py", "--pause", "25"], "q_render.log")
    log("рендери завершено; чекаю ШІ-тексти")
    ai.wait()
    log("ВСЕ ГОТОВО")


if __name__ == "__main__":
    main()
