# -*- coding: utf-8 -*-
"""Знаходить кращий центр для міст/районів, чия точка (Wikidata) впала на ліс,
воду чи промзону (у ділянці 800×800 м замало будинків).

Один прохід pyosmium по D:\\3dmap_tmp\\osm\\ukraine-latest.osm.pbf: рахує будівлі
в клітинках ~100 м навколо кожної точки (радіус ~3 км), потім ковзним вікном
800×800 м шукає найщільнішу ділянку з легким штрафом за відстань від вихідної точки.

Запуск: py311 tools/osm_recenter.py --in low.json --out recenter.json
low.json: [[id, lat, lon], ...]
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path

import osmium

PBF = r"D:\3dmap_tmp\osm\ukraine-latest.osm.pbf"
R_DEG = 0.03   # ~3 км
CELL = 0.0009  # ~100 м


class H(osmium.SimpleHandler):
    def __init__(self, pts):
        super().__init__()
        self.pts = pts
        self.cnt = {rid: defaultdict(int) for rid, _, _ in pts}

    def way(self, w):
        if "building" not in w.tags:
            return
        try:
            locs = [(n.location.lat, n.location.lon) for n in w.nodes if n.location.valid()]
        except osmium.InvalidLocationError:
            return
        if not locs:
            return
        la = sum(p[0] for p in locs) / len(locs)
        lo = sum(p[1] for p in locs) / len(locs)
        for rid, plat, plon in self.pts:
            if abs(la - plat) < R_DEG and abs(lo - plon) < R_DEG * 1.5:
                self.cnt[rid][(int((la - plat) // CELL), int((lo - plon) // CELL))] += 1


def best(counts, plat):
    win_lat = int(round(0.0072 / CELL))            # 800 м по широті
    win_lon = int(round(0.0072 / math.cos(math.radians(plat)) / CELL))
    keys = list(counts)
    if not keys:
        return None
    lo_i = min(k[0] for k in keys); hi_i = max(k[0] for k in keys)
    lo_j = min(k[1] for k in keys); hi_j = max(k[1] for k in keys)
    top = (-1, 0, 0)
    for i in range(lo_i, hi_i + 1, 2):
        for j in range(lo_j, hi_j + 1, 2):
            s = sum(counts.get((a, b), 0) for a in range(i, i + win_lat) for b in range(j, j + win_lon))
            ci, cj = i + win_lat / 2, j + win_lon / 2
            dist = math.hypot(ci, cj)                  # у клітинках від вихідної точки
            score = s * (1 - min(0.5, dist / 60))      # штраф за віддаленість (до −50 %)
            if score > top[0]:
                top = (score, ci, cj, s)
    return top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pts = json.loads(Path(a.inp).read_text(encoding="utf-8"))
    h = H(pts)
    h.apply_file(PBF, locations=True, idx="flex_mem")
    res = {}
    for rid, plat, plon in pts:
        t = best(h.cnt[rid], plat)
        if not t or t[0] <= 0:
            continue
        nlat = plat + t[1] * CELL
        nlon = plon + t[2] * CELL
        res[rid] = {"old": [plat, plon], "new": [round(nlat, 4), round(nlon, 4)], "buildings": t[3]}
        print(rid, res[rid], flush=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
