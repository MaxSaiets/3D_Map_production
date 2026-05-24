"""Benchmark різних стратегій парсингу OSM PBF.

Кожна стратегія обмежена N ways щоб оцінити перформанс без запуску на весь файл.
Запуск:
    python -u bench_osm.py /f/3dmap_data/osm/ukraine-latest.osm.pbf 100000
"""
import gc
import sys
import time

import osmium

PBF_PATH = sys.argv[1] if len(sys.argv) > 1 else r"F:/3dmap_data/osm/ukraine-latest.osm.pbf"
MAX_WAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000


class StopAfterN(Exception):
    pass


# ---------- STRATEGY 1: shapely WKT ----------
class HandlerShapely(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        from shapely.geometry import LineString
        self.LineString = LineString
        self.count = 0
        self.processed = 0
        self.start = time.time()

    def way(self, w):
        self.count += 1
        if self.count > MAX_WAYS:
            raise StopAfterN()
        if "building" not in w.tags and "highway" not in w.tags:
            return
        try:
            coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
        except Exception:
            return
        if len(coords) < 2:
            return
        _ = self.LineString(coords).wkt
        self.processed += 1


# ---------- STRATEGY 2: raw WKT string ----------
class HandlerRawWkt(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.processed = 0
        self.start = time.time()

    def way(self, w):
        self.count += 1
        if self.count > MAX_WAYS:
            raise StopAfterN()
        if "building" not in w.tags and "highway" not in w.tags:
            return
        try:
            coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
        except Exception:
            return
        if len(coords) < 2:
            return
        _ = "LINESTRING(" + ", ".join(f"{lon:.7f} {lat:.7f}" for lon, lat in coords) + ")"
        self.processed += 1


# ---------- STRATEGY 3: raw WKT + tight loop ----------
class HandlerOptimized(osmium.SimpleHandler):
    """Inline всі fast-path; уникаємо двічі ітеруватись."""

    BUILDING_KEYS = frozenset({"building"})
    HIGHWAY_KEYS = frozenset({"highway"})

    def __init__(self):
        super().__init__()
        self.count = 0
        self.processed = 0
        self.start = time.time()

    def way(self, w):
        self.count += 1
        if self.count > MAX_WAYS:
            raise StopAfterN()
        tags = w.tags
        # Дешева перевірка спершу: чи є потрібний тег взагалі
        if "building" not in tags and "highway" not in tags:
            return
        # Беремо ноди одним проходом + bbox + WKT inline
        parts = []
        first_lon = None
        first_lat = None
        minlon = maxlon = None
        minlat = maxlat = None
        n_count = 0
        for node in w.nodes:
            loc = node.location
            if not loc.valid():
                continue
            lon = loc.lon
            lat = loc.lat
            parts.append(f"{lon:.7f} {lat:.7f}")
            if n_count == 0:
                first_lon = lon
                first_lat = lat
                minlon = maxlon = lon
                minlat = maxlat = lat
            else:
                if lon < minlon: minlon = lon
                elif lon > maxlon: maxlon = lon
                if lat < minlat: minlat = lat
                elif lat > maxlat: maxlat = lat
            n_count += 1
        if n_count < 2:
            return
        wkt = "LINESTRING(" + ", ".join(parts) + ")"
        self.processed += 1


# ---------- STRATEGY 4: skip WKT, just count + bbox ----------
class HandlerBboxOnly(osmium.SimpleHandler):
    """Базова стеля швидкості — лише bbox без WKT."""

    def __init__(self):
        super().__init__()
        self.count = 0
        self.processed = 0
        self.start = time.time()

    def way(self, w):
        self.count += 1
        if self.count > MAX_WAYS:
            raise StopAfterN()
        if "building" not in w.tags and "highway" not in w.tags:
            return
        n_count = 0
        minlon = maxlon = minlat = maxlat = None
        for node in w.nodes:
            loc = node.location
            if not loc.valid():
                continue
            lon = loc.lon
            lat = loc.lat
            if n_count == 0:
                minlon = maxlon = lon
                minlat = maxlat = lat
            else:
                if lon < minlon: minlon = lon
                elif lon > maxlon: maxlon = lon
                if lat < minlat: minlat = lat
                elif lat > maxlat: maxlat = lat
            n_count += 1
        if n_count < 2:
            return
        self.processed += 1


def bench(name, HandlerCls):
    print(f"\n=== {name} ===", flush=True)
    h = HandlerCls()
    t0 = time.time()
    try:
        h.apply_file(PBF_PATH, locations=True)
    except StopAfterN:
        pass
    el = time.time() - t0
    rate = h.count / max(el, 0.001)
    proc_rate = h.processed / max(el, 0.001)
    print(f"  Time: {el:.2f}s | total ways: {h.count} | processed: {h.processed}", flush=True)
    print(f"  Rate: {rate:.0f} ways/sec total | {proc_rate:.0f} processed/sec", flush=True)
    return rate, proc_rate


def main():
    print(f"Benchmark PBF parsing strategies (limit: {MAX_WAYS} ways)", flush=True)
    print(f"PBF: {PBF_PATH}", flush=True)
    results = {}
    results["bbox_only (baseline)"] = bench("bbox_only", HandlerBboxOnly)
    gc.collect()
    results["raw_wkt"] = bench("raw_wkt", HandlerRawWkt)
    gc.collect()
    results["raw_wkt_optimized"] = bench("raw_wkt_optimized", HandlerOptimized)
    gc.collect()
    results["shapely_wkt"] = bench("shapely_wkt", HandlerShapely)
    print("\n=== SUMMARY ===", flush=True)
    print(f"{'Strategy':<30} {'ways/s':>10} {'proc/s':>10}", flush=True)
    for name, (rate, proc) in results.items():
        print(f"{name:<30} {rate:>10.0f} {proc:>10.0f}", flush=True)


if __name__ == "__main__":
    main()
