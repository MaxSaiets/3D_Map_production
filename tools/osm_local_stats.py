# -*- coding: utf-8 -*-
"""Дані OSM для SEO-сторінок ЛОКАЛЬНО, з витягу Geofabrik (без Overpass).

27.09.2026: публічні Overpass лежали дві ночі поспіль → рахуємо з
D:\\3dmap_tmp\\osm\\ukraine-latest.osm.pbf одним проходом pyosmium.

Для кожної точки (ділянка 800×800 м навколо центру — та сама, що в рендерах)
рахує те саме, що tools/osm_page_stats.py: будівлі, поверховість, назви парків,
водойм і пам'яток. Результат — у той самий кеш D:\\3dmap_tmp\\osm_stats\\{id}.json,
тож AiPageText/OsmModelContents і ai_page_texts.py підхоплюють його без змін.

Запуск (потрібен pyosmium; є в Python 3.11):
  py311 tools/osm_local_stats.py --targets D:\\3dmap_tmp\\osm\\targets.json
targets.json: [[id, lat, lon], ...] — дамп з night_city_renders.targets().
"""
import argparse, json, math, time
from collections import defaultdict
from pathlib import Path

import osmium

PBF = Path(r"D:\3dmap_tmp\osm\ukraine-latest.osm.pbf")
CACHE = Path(r"D:\3dmap_tmp\osm_stats")
HALF_M = 400
CELL = 0.01  # градуси; ділянка ≈ 0.0072° × 0.011° — кожна точка потрапляє в ≤ 4 клітинки

AM = ("place_of_worship", "theatre", "university")
TU = ("museum", "attraction")
HI = ("monument", "memorial", "castle", "building", "church", "fort")
CH = ("cathedral", "church")
GENERIC = {"сквер", "парк", "сад", "park", "garden"}


class Targets:
    def __init__(self, rows):
        self.box = {}
        self.grid = defaultdict(list)
        for rid, lat, lon in rows:
            dlat = HALF_M / 111320
            dlon = HALF_M / (111320 * math.cos(math.radians(lat)))
            b = (lat - dlat, lon - dlon, lat + dlat, lon + dlon)
            self.box[rid] = b
            for i in range(int(math.floor(b[0] / CELL)), int(math.floor(b[2] / CELL)) + 1):
                for j in range(int(math.floor(b[1] / CELL)), int(math.floor(b[3] / CELL)) + 1):
                    self.grid[(i, j)].append(rid)

    def hit(self, lat, lon):
        out = []
        for rid in self.grid.get((int(math.floor(lat / CELL)), int(math.floor(lon / CELL))), ()):
            b = self.box[rid]
            if b[0] <= lat <= b[2] and b[1] <= lon <= b[3]:
                out.append(rid)
        return out


def poi_kind(t):
    if t.get("amenity") == "place_of_worship" or t.get("building") in CH or t.get("historic") == "church":
        return "church"
    if t.get("amenity") == "theatre": return "theatre"
    if t.get("amenity") == "university": return "university"
    if t.get("tourism") == "museum": return "museum"
    if t.get("historic") in ("castle", "fort"): return "castle"
    if t.get("historic") in ("monument", "memorial"): return "monument"
    return "sight"


def is_poi(t):
    return t.get("amenity") in AM or t.get("tourism") in TU or t.get("historic") in HI or t.get("building") in CH


class H(osmium.SimpleHandler):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.S = defaultdict(lambda: {"b": 0, "lv": [], "parks": [], "water": [], "pois": []})
        self.n = 0

    def _name(self, t):
        return t.get("name:uk") or t.get("name")

    def _add(self, rids, key, val):
        for r in rids:
            lst = self.S[r][key]
            names = [x[0] if isinstance(x, list) else x for x in lst]
            if val and (val[0] if isinstance(val, list) else val) not in names:
                lst.append(val)

    def node(self, n):
        t = n.tags
        if not (is_poi(t) and self._name(t)):
            return
        rids = self.T.hit(n.location.lat, n.location.lon)
        if rids:
            self._add(rids, "pois", [self._name(t), poi_kind(t)])

    def way(self, w):
        t = w.tags
        bld = "building" in t
        park = t.get("leisure") in ("park", "garden")
        water = t.get("natural") == "water" or t.get("waterway") in ("river", "canal")
        poi = is_poi(t)
        if not (bld or park or water or poi):
            return
        self.n += 1
        try:
            locs = [(nd.location.lat, nd.location.lon) for nd in w.nodes if nd.location.valid()]
        except osmium.InvalidLocationError:
            return
        if not locs:
            return
        name = self._name(t)
        if water or park:
            # довгі лінії/великі полігони: достатньо, щоб хоч одна вершина була в ділянці
            rids = set()
            for la, lo in locs[:: max(1, len(locs) // 60)]:
                rids.update(self.T.hit(la, lo))
            if name and name.strip().lower() not in GENERIC:
                self._add(rids, "water" if water else "parks", name)
            if not bld:
                return
        cla = sum(p[0] for p in locs) / len(locs)
        clo = sum(p[1] for p in locs) / len(locs)
        rids = self.T.hit(cla, clo)
        if not rids:
            return
        if bld:
            lv = None
            try:
                v = float(str(t.get("building:levels", "")).split(";")[0].replace(",", "."))
                if 0 < v < 120: lv = v
            except ValueError:
                pass
            for r in rids:
                self.S[r]["b"] += 1
                if lv: self.S[r]["lv"].append(lv)
        if poi and name:
            self._add(rids, "pois", [name, poi_kind(t)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    a = ap.parse_args()
    rows = json.loads(Path(a.targets).read_text(encoding="utf-8"))
    T = Targets(rows)
    h = H(T)
    t0 = time.time()
    h.apply_file(str(PBF), locations=True, idx="flex_mem")
    print(f"прохід {round(time.time() - t0)} с, objects {h.n}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    n = 0
    for rid, _, _ in rows:
        s = h.S.get(rid)
        if not s or not s["b"]:
            continue
        out = {"b": s["b"]}
        if s["lv"]:
            out["maxLv"] = int(max(s["lv"]))
            out["avgLv"] = round(sum(s["lv"]) / len(s["lv"]), 1)
            out["lvShare"] = round(len(s["lv"]) / max(1, s["b"]), 2)
        if s["parks"]: out["parks"] = s["parks"][:4]
        if s["water"]: out["water"] = s["water"][:3]
        if s["pois"]: out["pois"] = s["pois"][:8]
        (CACHE / f"{rid}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        n += 1
    print(f"записано {n} з {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
