# -*- coding: utf-8 -*-
"""Імпорт текстів сторінок, написаних вручну/іншою моделлю, через ТОЙ САМИЙ валідатор,
що й ai_page_texts.py. Пройшло — у кеш D:\3dmap_tmp\ai_texts (ok=True); ні — друкує помилки.
Запуск: python tools/import_page_texts.py file.json [--model name]   (file: {id: {uk: [...], en: [...]}})"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ai_page_texts as a

src = Path(sys.argv[1]); model = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--model" else "claude-opus"
data = json.loads(src.read_text(encoding="utf-8"))
P = a.pages(); ok = bad = 0
for pid, res in data.items():
    if pid not in P:
        print("НЕВІДОМИЙ id", pid); bad += 1; continue
    kind, facts, name = P[pid]
    facts = a.add_osm(pid, dict(facts))
    errs = a.validate(res, facts)
    if errs:
        print("FAIL", pid, errs); bad += 1; continue
    rec = {"ok": True, "model": model, "facts": facts}; rec.update(a.clean(res))
    (a.CACHE / f"{pid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    ok += 1
print("OK", ok, "FAIL", bad, "усього в pageAiText.ts:", a.write_ts())
