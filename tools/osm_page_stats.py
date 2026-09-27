# -*- coding: utf-8 -*-
"""Реальні дані OpenStreetMap для кожної SEO-сторінки міста/району/вулиці:
що саме потрапить у 3D-модель (ділянка 800×800 м навколо центру сторінки —
та сама, що рендериться tools/night_city_renders.py).

Для кожної точки: кількість будівель, поверховість (макс/середня з тих, де вказано),
назви парків і водойм, названі пам'ятки (храми, музеї, театри, університети,
історичні об'єкти, пам'ятники) — лише з тегом name.

Спокійно: один запит Overpass на точку, пауза між запитами, повтор на дзеркалі,
готові точки пропускаються (кеш у D:\\3dmap_tmp\\osm_stats\\*.json).
Наприкінці пише frontend/lib/pageOsmStats.ts.

Запуск: python tools/osm_page_stats.py [--pause 8] [--limit N]
"""
import argparse, json, math, sys, time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from night_city_renders import targets, FE  # той самий список точок і id

CACHE = Path(r"D:\3dmap_tmp\osm_stats")
# 27.09: вночі kumi/private.coffee висіли тайм-аутами 60–120 с на кожну точку → лише живе дзеркало,
# коротший тайм-аут; kumi — запасне.
MIRRORS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
UA = {"User-Agent": "monadruk-seo-stats/1.0"}


def bbox(lat, lon, size_m=800):
    half = size_m / 2
    dlat = half / 111320
    dlon = half / (111320 * math.cos(math.radians(lat)))
    return f"{lat - dlat:.6f},{lon - dlon:.6f},{lat + dlat:.6f},{lon + dlon:.6f}"


def query(lat, lon):
    b = bbox(lat, lon)
    q = f"""[out:json][timeout:50];
(way["building"]({b});relation["building"]({b}););out count;
(way["building"]["building:levels"]({b}););out tags;
(nwr["leisure"~"^(park|garden)$"]["name"]({b});nwr["natural"="water"]["name"]({b});nwr["waterway"~"^(river|canal)$"]["name"]({b});
 nwr["amenity"~"^(place_of_worship|theatre|university)$"]["name"]({b});nwr["tourism"~"^(museum|attraction)$"]["name"]({b});
 nwr["historic"~"^(monument|memorial|castle|building|church|fort)$"]["name"]({b});nwr["building"~"^(cathedral|church)$"]["name"]({b}););out tags;"""
    last = None
    for m in MIRRORS:
        try:
            r = requests.post(m, data={"data": q}, headers=UA, timeout=60)
            if r.status_code == 200:
                return r.json()["elements"]
            last = f"{m} {r.status_code}"
        except Exception as e:
            last = f"{m} {e}"
        time.sleep(5)
    raise RuntimeError(last)


def summarise(els):
    lv = []
    nb = 0
    parks, water, pois = [], [], []
    seen_el, seen_name = set(), set()
    for e in els:
        key = (e.get("type"), e.get("id"))
        if key in seen_el:
            continue
        seen_el.add(key)
        t = e.get("tags", {})
        name = t.get("name:uk") or t.get("name")
        is_bld = "building" in t
        if is_bld:
            nb += 1
            try:
                v = float(str(t.get("building:levels", "")).split(";")[0].replace(",", "."))
                if 0 < v < 120:
                    lv.append(v)
            except ValueError:
                pass
        if not name or name in seen_name:
            continue
        am_ok = t.get("amenity") in ("place_of_worship", "theatre", "university")
        tu_ok = t.get("tourism") in ("museum", "attraction")
        hi_ok = t.get("historic") in ("monument", "memorial", "castle", "building", "church", "fort")
        ch_ok = t.get("building") in ("cathedral", "church")
        if is_bld and not (am_ok or tu_ok or hi_ok or ch_ok):
            continue
        if name.strip().lower() in ("сквер", "парк", "сад", "park", "garden"):
            continue
        seen_name.add(name)
        if t.get("leisure") in ("park", "garden"):
            parks.append(name)
        elif t.get("natural") == "water" or t.get("waterway"):
            water.append(name)
        elif am_ok or tu_ok or hi_ok or ch_ok:
            kind = ("church" if t.get("amenity") == "place_of_worship" or t.get("building") in ("cathedral", "church") or t.get("historic") == "church"
                    else "theatre" if t.get("amenity") == "theatre"
                    else "university" if t.get("amenity") == "university"
                    else "museum" if t.get("tourism") == "museum"
                    else "castle" if t.get("historic") in ("castle", "fort")
                    else "monument" if t.get("historic") in ("monument", "memorial")
                    else "sight")
            pois.append([name, kind])
    # легкий запит: будівлі приходять як «out count» + лише ті, що мають поверховість
    cnt = next((e for e in els if e.get("type") == "count"), None)
    if cnt:
        tg = cnt.get("tags", {})
        nb = int(tg.get("total") or (int(tg.get("ways", 0)) + int(tg.get("relations", 0))))
    out = {"b": nb}
    if lv:
        out["maxLv"] = int(max(lv))
        out["avgLv"] = round(sum(lv) / len(lv), 1)
        out["lvShare"] = round(len(lv) / max(1, nb), 2)
    if parks: out["parks"] = parks[:4]
    if water: out["water"] = water[:3]
    if pois: out["pois"] = pois[:8]
    return out


def write_ts():
    data = {}
    for p in sorted(CACHE.glob("*.json")):
        data[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (FE / "lib" / "pageOsmStats.ts").write_text(
        "/**\n * Реальні дані OSM для ділянки 800×800 м кожної SEO-сторінки міста/району/вулиці\n"
        " * (b — будівель, maxLv/avgLv — поверхи, parks/water/pois — назви).\n"
        " * ФАЙЛ ГЕНЕРУЄ tools/osm_page_stats.py — не правити вручну.\n */\n"
        "export interface OsmStats { b: number; maxLv?: number; avgLv?: number; lvShare?: number; parks?: string[]; water?: string[]; pois?: [string, string][] }\n"
        f"export const PAGE_OSM_STATS: Record<string, OsmStats> = {body};\n", encoding="utf-8")
    return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    n = ok = 0
    for rid, lat, lon in targets():
        f = CACHE / f"{rid}.json"
        if f.exists():
            continue
        if a.limit and n >= a.limit:
            break
        n += 1
        try:
            s = summarise(query(lat, lon))
            f.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
            ok += 1
            print(time.strftime("%H:%M:%S"), "OK", rid, s.get("b"), flush=True)
        except Exception as e:
            print(time.strftime("%H:%M:%S"), "FAIL", rid, e, flush=True)
        time.sleep(a.pause)
    print("END", ok, "/", n, "всього", write_ts(), flush=True)


if __name__ == "__main__":
    main()
