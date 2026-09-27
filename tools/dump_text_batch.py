# -*- coding: utf-8 -*-
"""Вивантажує факти наступної партії сторінок без тексту: python tools/dump_text_batch.py N out.json [kind]"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ai_page_texts as a
n, out = int(sys.argv[1]), sys.argv[2]
kind_only = sys.argv[3] if len(sys.argv) > 3 else None
P = a.pages()
done = {f[:-5] for f in os.listdir(a.CACHE)}
todo = [pid for pid in P if pid not in done and (a.OSM / f"{pid}.json").exists()
        and (kind_only is None or P[pid][0] == kind_only)]
batch = [{"id": pid, "kind": P[pid][0], "name": P[pid][2], "facts": a.add_osm(pid, dict(P[pid][1]))} for pid in todo[:n]]
Path(out).write_text(json.dumps(batch, ensure_ascii=False, indent=0), encoding="utf-8")
print(len(todo), "без тексту; у партії", len(batch))
