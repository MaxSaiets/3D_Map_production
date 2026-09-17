# -*- coding: utf-8 -*-
"""Живе порівняння: шари з ОДНОГО Overpass-запиту (services/overpass_bundle) проти
старого шляху по шарах. Друкує час обох і чи збігаються будівлі/вода/мости/дороги/зелень.

    venv/Scripts/python scripts/bench_bundle.py                 # Відень, ~600 м
    venv/Scripts/python scripts/bench_bundle.py 48.2100 48.2050 16.3750 16.3680

Потрібен доступ до overpass-api.de. Кеш parquet вимикається, щоб обидва шляхи
справді ходили в мережу.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

os.environ["OSM_DATA_CACHE_ENABLED"] = "0"
os.environ.setdefault("OSM_EXTRAS_CACHE_ENABLED", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd  # noqa: E402

from services import overpass_bundle  # noqa: E402
from services.data_loader import fetch_city_data  # noqa: E402
from services.extras_loader import fetch_extras  # noqa: E402


def gdf_sig(gdf) -> tuple[int, str]:
    if gdf is None or getattr(gdf, "empty", True):
        return 0, "-"
    h = hashlib.sha256()
    rows = []
    for idx, row in gdf.iterrows():
        tags = tuple(sorted((k, str(row[k])) for k in gdf.columns if k != "geometry" and row[k] == row[k]))
        rows.append((str(idx), row.geometry.wkb, tags))
    for r in sorted(rows, key=lambda r: r[0]):
        h.update(r[0].encode())
        h.update(r[1])
        h.update(repr(r[2]).encode())
    return len(gdf), h.hexdigest()[:12]


def graph_sig(G) -> tuple[int, str]:
    if G is None or not hasattr(G, "edges"):
        return 0, "-"
    h = hashlib.sha256()
    for n, d in sorted(G.nodes(data=True)):
        h.update(f"{n}:{round(d['x'], 6)}:{round(d['y'], 6)}".encode())
    edges = []
    for u, v, k, d in G.edges(keys=True, data=True):
        g = d.get("geometry")
        edges.append((u, v, k, str(d.get("osmid")), str(d.get("highway")), g.wkb if g is not None else b""))
    for e in sorted(edges, key=str):
        h.update(repr(e[:5]).encode())
        h.update(e[5])
    return G.number_of_edges(), h.hexdigest()[:12]


def run(north, south, east, west, use_bundle: bool):
    road_padding, loader_padding = 0.01, 0.005
    bundle = None
    if use_bundle:
        n, s, e, w = north + road_padding, south - road_padding, east + road_padding, west - road_padding
        bundle = overpass_bundle.LazyBundle(
            city_bbox=(w - loader_padding, s - loader_padding, e + loader_padding, n + loader_padding),
            extras_bbox=(west, south, east, north),
            feature_tags=[
                {"building": True}, {"building:part": True},
                {"natural": "water", "water": True, "waterway": ["riverbank", "dock", "canal"],
                 "landuse": ["reservoir", "basin"], "man_made": ["water_well", "reservoir_covered"]},
                {"bridge": True},
                {"railway": ["rail", "light_rail", "narrow_gauge", "tram", "subway", "funicular"]},
            ],
            extras_tags=[{
                "leisure": ["park", "garden", "playground", "recreation_ground", "pitch", "nature_reserve", "golf_course"],
                "landuse": ["grass", "meadow", "forest", "village_green", "cemetery", "allotments", "orchard", "recreation_ground"],
                "natural": ["wood", "grassland", "scrub", "heath"],
            }],
            label="bench",
        )
    t0 = time.time()
    b, w, g = fetch_city_data(
        north + road_padding, south - road_padding, east + road_padding, west - road_padding,
        padding=loader_padding, target_crs="EPSG:32633", include_building_parts=True, bundle=bundle,
    )
    t1 = time.time()
    green = fetch_extras(north, south, east, west, target_crs="EPSG:32633", bundle=bundle)
    t2 = time.time()
    bridges = getattr(b, "attrs", {}).get("bridges") if b is not None else None
    return {
        "city_s": round(t1 - t0, 1), "extras_s": round(t2 - t1, 1),
        "buildings": gdf_sig(b), "water": gdf_sig(w), "bridges": gdf_sig(bridges),
        "roads": graph_sig(g), "green": gdf_sig(green),
        "bundle_fetch_s": round(bundle.fetch_seconds or 0, 1) if bundle else None,
    }


def main():
    if len(sys.argv) >= 5:
        north, south, east, west = map(float, sys.argv[1:5])
    else:
        north, south, east, west = 48.2100, 48.2050, 16.3750, 16.3680  # Відень, Штефансдом
    print(f"bbox N={north} S={south} E={east} W={west}")
    order = sys.argv[5] if len(sys.argv) > 5 else "bundle-first"
    modes = [True, False] if order == "bundle-first" else [False, True]
    results = {}
    for use_bundle in modes:
        print(f"\n=== {'ПАКЕТ (1 запит)' if use_bundle else 'СТАРИЙ ШЛЯХ (по шарах)'} ===", flush=True)
        results[use_bundle] = run(north, south, east, west, use_bundle)
        print(results[use_bundle], flush=True)
    a, b = results[True], results[False]
    print("\n=== ПІДСУМОК ===")
    print(f"час: пакет {a['city_s'] + a['extras_s']} с (сам запит {a['bundle_fetch_s']} с) vs старий {b['city_s'] + b['extras_s']} с")
    same = True
    for k in ("buildings", "water", "bridges", "roads", "green"):
        ok = a[k] == b[k]
        same &= ok
        print(f"{k:10s} {'OK ' if ok else 'DIFF'} пакет={a[k]} старий={b[k]}")
    print("РЕЗУЛЬТАТ:", "ІДЕНТИЧНО" if same else "Є РОЗБІЖНОСТІ")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
