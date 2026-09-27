# -*- coding: utf-8 -*-
"""Нарізає з ukraine-latest.osm.pbf РЕГІОНАЛЬНІ PBF (по одному на місто: саме місто,
його райони й вулиці), щоб локальний бекенд (OSM_SOURCE=pbf) читав кілька десятків МБ,
а не 1 ГБ на кожну модель. Тест 27.09: на повному файлі бекенд за 3 хв не дочитав
одну ділянку й зʼїв 5,5 ГБ памʼяті.

Один прохід pyosmium з локаціями, без великих множин у памʼяті:
  * вузол у рамці регіону → у файл регіону;
  * лінія, в якої хоч одна вершина в рамці → у файл (запамʼятовуємо лише id ліній);
  * відношення з хоч одним членом-лінією регіону → у файл.
Рамка = межі точок регіону + запас MARGIN_KM (модель бере лише ~1 км від центру ділянки,
тож обрізані на краю регіону лінії її не зачіпають).

Запуск (py311 з pyosmium): tools/osm_cut_extracts.py --targets targets.json
Вихід: D:\\3dmap_tmp\\osm\\cuts\\{місто}.osm.pbf
"""
import argparse, json, math, time
from collections import defaultdict
from pathlib import Path

import osmium

PBF = r"D:\3dmap_tmp\osm\ukraine-latest.osm.pbf"
OUT = Path(r"D:\3dmap_tmp\osm\cuts")
MARGIN_KM = 2.5
CELL = 0.05


def region_of(rid: str) -> str:
    return rid.split("--")[0]


class H(osmium.SimpleHandler):
    def __init__(self, boxes):
        super().__init__()
        self.boxes = boxes
        self.grid = defaultdict(list)
        for reg, b in boxes.items():
            for i in range(int(b[0] // CELL), int(b[2] // CELL) + 1):
                for j in range(int(b[1] // CELL), int(b[3] // CELL) + 1):
                    self.grid[(i, j)].append(reg)
        OUT.mkdir(parents=True, exist_ok=True)
        self.w = {reg: osmium.SimpleWriter(str(OUT / f"{reg}.osm.pbf")) for reg in boxes}
        self.ways = defaultdict(set)
        self.n = self.nw = self.nr = 0

    def hit(self, lat, lon):
        out = []
        for reg in self.grid.get((int(lat // CELL), int(lon // CELL)), ()):
            b = self.boxes[reg]
            if b[0] <= lat <= b[2] and b[1] <= lon <= b[3]:
                out.append(reg)
        return out

    def node(self, n):
        for reg in self.hit(n.location.lat, n.location.lon):
            self.w[reg].add_node(n); self.n += 1

    def way(self, w):
        regs = set()
        for nd in w.nodes:
            if nd.location.valid():
                regs.update(self.hit(nd.location.lat, nd.location.lon))
        for reg in regs:
            self.w[reg].add_way(w); self.ways[reg].add(w.id); self.nw += 1

    def relation(self, r):
        regs = set()
        for m in r.members:
            if m.type == "w":
                for reg, s in self.ways.items():
                    if m.ref in s: regs.add(reg)
        for reg in regs:
            self.w[reg].add_relation(r); self.nr += 1

    def close(self):
        for x in self.w.values(): x.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    a = ap.parse_args()
    rows = json.loads(Path(a.targets).read_text(encoding="utf-8"))
    pts = defaultdict(list)
    for rid, lat, lon in rows:
        pts[region_of(rid)].append((lat, lon))
    boxes = {}
    for reg, ps in pts.items():
        if (OUT / f"{reg}.osm.pbf").exists():
            continue
        la = [p[0] for p in ps]; lo = [p[1] for p in ps]
        dlat = MARGIN_KM * 1000 / 111320
        dlon = MARGIN_KM * 1000 / (111320 * math.cos(math.radians(sum(la) / len(la))))
        boxes[reg] = (min(la) - dlat, min(lo) - dlon, max(la) + dlat, max(lo) + dlon)
    if not boxes:
        print("усі вирізки вже є"); return
    t0 = time.time()
    h = H(boxes)
    h.apply_file(PBF, locations=True, idx="flex_mem")
    h.close()
    print(f"регіонів {len(boxes)}, вузлів {h.n}, ліній {h.nw}, відношень {h.nr}, {round(time.time() - t0)} с", flush=True)


if __name__ == "__main__":
    main()
