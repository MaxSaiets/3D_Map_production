"""Видобуває ТІЛЬКИ залізничні колії з PBF у маленьку duckdb.

Навіщо окремо від build_osm_db.py: повний ребілд ukraine.duckdb — це години і
3.6ГБ, а на проді ще й ризик OOM (4ГБ RAM). Колії — крихітний підмножинний шар,
його можна зібрати локально за хвилини і залити на сервер одним невеликим файлом.

Результат — таблиця з тією САМОЮ схемою, що `roads` у ukraine.duckdb, зі
значенням highway='railway'. На сервері вона просто вливається в `roads`, після
чого і генерація, і превʼю беруть колії з бази, а не з Overpass.

Usage:
    python build_rails_table.py <input.pbf> <output.duckdb>
"""

import sys
import time
from pathlib import Path

import duckdb
import osmium
import pandas as pd

# Синхрон з build_osm_db.RAILWAY_TAGS
RAILWAY_TAGS = frozenset({"rail", "light_rail", "narrow_gauge", "tram",
                          "subway", "funicular"})
RAILWAY_PSEUDO_HIGHWAY = "railway"
COLS = ["id", "highway", "bridge", "wkt", "minlon", "minlat", "maxlon", "maxlat"]
BATCH_SIZE = 100_000


class RailHandler(osmium.SimpleHandler):
    def __init__(self, conn):
        super().__init__()
        self.conn = conn
        self.rows = []
        self.total = 0
        self.skipped_underground = 0
        self.way_count = 0
        self.last_log = time.time()

    def way(self, w):
        self.way_count += 1
        if time.time() - self.last_log > 15:
            print(f"  ways={self.way_count:,} rails={self.total:,}", flush=True)
            self.last_log = time.time()

        tags = w.tags
        if not tags or len(w.nodes) < 2:
            return
        if tags.get("railway") not in RAILWAY_TAGS:
            return

        # Підземка на поверхні не існує — синхрон з build_osm_db/data_loader.
        tunnel = (tags.get("tunnel") or "").strip().lower()
        layer = tags.get("layer")
        try:
            underground = (tunnel not in ("", "no")) or (float(layer) < 0 if layer else False)
        except (TypeError, ValueError):
            underground = tunnel not in ("", "no")
        if underground:
            self.skipped_underground += 1
            return

        coords = []
        try:
            for n in w.nodes:
                if n.location.valid():
                    coords.append((n.lon, n.lat))
        except Exception:
            return
        if len(coords) < 2:
            return

        minlon = maxlon = coords[0][0]
        minlat = maxlat = coords[0][1]
        for lon, lat in coords[1:]:
            if lon < minlon: minlon = lon
            elif lon > maxlon: maxlon = lon
            if lat < minlat: minlat = lat
            elif lat > maxlat: maxlat = lat

        wkt = "LINESTRING(" + ", ".join(f"{lo:.7f} {la:.7f}" for lo, la in coords) + ")"
        bridge = tags.get("bridge") or "no"
        self.rows.append((w.id, RAILWAY_PSEUDO_HIGHWAY, bridge, wkt,
                          minlon, minlat, maxlon, maxlat))
        self.total += 1
        if len(self.rows) >= BATCH_SIZE:
            self._flush()

    def _flush(self):
        if not self.rows:
            return
        df = pd.DataFrame(self.rows, columns=COLS)  # noqa: F841 — duckdb бачить через scope
        self.conn.execute("INSERT INTO rails SELECT * FROM df")
        self.rows = []


def main():
    if len(sys.argv) < 3:
        print("Usage: python build_rails_table.py <input.pbf> <output.duckdb>")
        sys.exit(1)
    pbf_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    if not pbf_path.exists():
        print(f"PBF not found: {pbf_path}")
        sys.exit(1)
    if out_path.exists():
        out_path.unlink()
    wal = out_path.with_suffix(out_path.suffix + ".wal")
    if wal.exists():
        wal.unlink()

    conn = duckdb.connect(str(out_path))
    conn.execute("SET memory_limit='2GB'")
    conn.execute("""
        CREATE TABLE rails (
            id BIGINT, highway VARCHAR, bridge VARCHAR, wkt VARCHAR,
            minlon DOUBLE, minlat DOUBLE, maxlon DOUBLE, maxlat DOUBLE
        );
    """)

    handler = RailHandler(conn)
    print(f"Parsing {pbf_path} (rails only)...", flush=True)
    t0 = time.time()
    handler.apply_file(str(pbf_path), locations=True)
    handler._flush()
    print(f"Done in {(time.time() - t0)/60:.1f}m", flush=True)
    print(f"  rails kept: {handler.total:,}", flush=True)
    print(f"  underground skipped: {handler.skipped_underground:,}", flush=True)

    cnt = conn.execute("SELECT COUNT(*) FROM rails").fetchone()[0]
    bbox = conn.execute(
        "SELECT MIN(minlat), MAX(maxlat), MIN(minlon), MAX(maxlon) FROM rails"
    ).fetchone()
    print(f"  rows in db: {cnt:,}")
    print(f"  bbox: lat {bbox[0]:.3f}..{bbox[1]:.3f} lon {bbox[2]:.3f}..{bbox[3]:.3f}")
    conn.close()
    print(f"Written: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
