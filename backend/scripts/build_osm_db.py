"""
build_osm_db.py — конвертує OSM PBF (Geofabrik Ukraine) у локальну DuckDB-БД.

Створює таблиці:
  buildings(id INTEGER, levels INTEGER, wkt VARCHAR, minlon DOUBLE, minlat DOUBLE, maxlon DOUBLE, maxlat DOUBLE)
  roads(id, highway VARCHAR, bridge VARCHAR, wkt, bbox...)
  water(id, type VARCHAR, wkt, bbox...)
  parks(id, type VARCHAR, wkt, bbox...)
  bridges(id, highway, wkt, bbox...)

Кожна таблиця має індекс на (minlon, minlat) + (maxlon, maxlat) для bbox-queries.

Запуск:
    python backend/scripts/build_osm_db.py /f/3dmap_data/osm/ukraine-latest.osm.pbf /f/3dmap_data/osm/ukraine.duckdb

Очікуваний час: 20-60 хв на середньому ноуті. Output: ~2.5 GB DuckDB.
"""
import sys
import time
from pathlib import Path

import duckdb
import osmium
from shapely.geometry import LineString, Polygon


BUILDING_TAGS = {"yes", "house", "apartments", "residential", "commercial",
                 "industrial", "retail", "office", "church", "school",
                 "university", "hospital", "warehouse", "garage", "kindergarten"}

HIGHWAY_TAGS = {"motorway", "trunk", "primary", "secondary", "tertiary",
                "residential", "unclassified", "service", "pedestrian",
                "footway", "path", "cycleway"}

WATER_TAGS_NATURAL = {"water"}
WATER_TAGS_WATERWAY = {"riverbank", "dock", "canal"}
WATER_TAGS_LANDUSE = {"reservoir", "basin"}

PARK_TAGS_LEISURE = {"park", "garden", "nature_reserve", "recreation_ground", "pitch", "playground", "golf_course"}
PARK_TAGS_LANDUSE = {"grass", "meadow", "forest", "village_green", "cemetery", "allotments", "orchard"}
PARK_TAGS_NATURAL = {"wood", "grassland", "scrub", "heath"}


class Handler(osmium.SimpleHandler):
    """Парсить OSM way і node об'єкти."""

    def __init__(self, conn):
        super().__init__()
        self.conn = conn
        self.factory = osmium.geom.WKTFactory()
        self.buildings = []
        self.roads = []
        self.bridges = []
        self.water = []
        self.parks = []
        self.count_buildings = 0
        self.count_roads = 0
        self.count_bridges = 0
        self.count_water = 0
        self.count_parks = 0
        self.last_flush = time.time()

    def _bbox(self, coords):
        """Повертає (minlon, minlat, maxlon, maxlat) з списку (lon, lat)."""
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return (min(lons), min(lats), max(lons), max(lats))

    def way(self, w):
        tags = {t.k: t.v for t in w.tags}
        if not tags or len(w.nodes) < 2:
            return
        # Збираємо координати один раз
        try:
            coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
        except Exception:
            return
        if len(coords) < 2:
            return
        bbox = self._bbox(coords)
        wkt_line = LineString(coords).wkt if len(coords) >= 2 else None
        wkt_poly = Polygon(coords).wkt if len(coords) >= 3 and coords[0] == coords[-1] else wkt_line

        # Buildings (polygons only)
        if "building" in tags and len(coords) >= 4 and coords[0] == coords[-1]:
            try:
                levels = int(tags.get("building:levels") or 0)
            except Exception:
                levels = 0
            self.buildings.append((w.id, levels, wkt_poly, *bbox))
            self.count_buildings += 1

        # Bridges (any way with bridge=yes)
        if tags.get("bridge", "no") not in ("no", ""):
            hw = tags.get("highway", "primary")
            self.bridges.append((w.id, hw, wkt_line, *bbox))
            self.count_bridges += 1

        # Roads
        hw = tags.get("highway")
        if hw and hw in HIGHWAY_TAGS:
            self.roads.append((w.id, hw, tags.get("bridge") or "no", wkt_line, *bbox))
            self.count_roads += 1

        # Water (polygons)
        is_water = (
            tags.get("natural") in WATER_TAGS_NATURAL
            or tags.get("waterway") in WATER_TAGS_WATERWAY
            or tags.get("landuse") in WATER_TAGS_LANDUSE
        )
        if is_water and len(coords) >= 4 and coords[0] == coords[-1]:
            wtype = tags.get("natural") or tags.get("waterway") or tags.get("landuse") or "water"
            self.water.append((w.id, wtype, wkt_poly, *bbox))
            self.count_water += 1

        # Parks (polygons)
        is_park = (
            tags.get("leisure") in PARK_TAGS_LEISURE
            or tags.get("landuse") in PARK_TAGS_LANDUSE
            or tags.get("natural") in PARK_TAGS_NATURAL
        )
        if is_park and len(coords) >= 4 and coords[0] == coords[-1]:
            ptype = tags.get("leisure") or tags.get("landuse") or tags.get("natural") or "park"
            self.parks.append((w.id, ptype, wkt_poly, *bbox))
            self.count_parks += 1

        # Періодичний flush щоб не тримати все в RAM
        if time.time() - self.last_flush > 30:
            self._flush()

    def _flush(self):
        if self.buildings:
            self.conn.executemany(
                "INSERT INTO buildings VALUES (?, ?, ?, ?, ?, ?, ?)", self.buildings
            )
            self.buildings.clear()
        if self.roads:
            self.conn.executemany(
                "INSERT INTO roads VALUES (?, ?, ?, ?, ?, ?, ?, ?)", self.roads
            )
            self.roads.clear()
        if self.bridges:
            self.conn.executemany(
                "INSERT INTO bridges VALUES (?, ?, ?, ?, ?, ?, ?)", self.bridges
            )
            self.bridges.clear()
        if self.water:
            self.conn.executemany(
                "INSERT INTO water VALUES (?, ?, ?, ?, ?, ?, ?)", self.water
            )
            self.water.clear()
        if self.parks:
            self.conn.executemany(
                "INSERT INTO parks VALUES (?, ?, ?, ?, ?, ?, ?)", self.parks
            )
            self.parks.clear()
        self.last_flush = time.time()
        print(f"[BUILD] Flushed. Totals: b={self.count_buildings} r={self.count_roads} "
              f"br={self.count_bridges} w={self.count_water} p={self.count_parks}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python build_osm_db.py <input.pbf> <output.duckdb>")
        sys.exit(1)
    pbf_path = Path(sys.argv[1])
    db_path = Path(sys.argv[2])
    if not pbf_path.exists():
        print(f"PBF not found: {pbf_path}")
        sys.exit(1)
    if db_path.exists():
        print(f"Removing old DB: {db_path}")
        db_path.unlink()

    print(f"Building DuckDB from {pbf_path} -> {db_path}")
    conn = duckdb.connect(str(db_path))

    # Створюємо таблиці
    conn.execute("""
        CREATE TABLE buildings (
            id BIGINT, levels INTEGER, wkt VARCHAR,
            minlon DOUBLE, minlat DOUBLE, maxlon DOUBLE, maxlat DOUBLE
        );
        CREATE TABLE roads (
            id BIGINT, highway VARCHAR, bridge VARCHAR, wkt VARCHAR,
            minlon DOUBLE, minlat DOUBLE, maxlon DOUBLE, maxlat DOUBLE
        );
        CREATE TABLE bridges (
            id BIGINT, highway VARCHAR, wkt VARCHAR,
            minlon DOUBLE, minlat DOUBLE, maxlon DOUBLE, maxlat DOUBLE
        );
        CREATE TABLE water (
            id BIGINT, type VARCHAR, wkt VARCHAR,
            minlon DOUBLE, minlat DOUBLE, maxlon DOUBLE, maxlat DOUBLE
        );
        CREATE TABLE parks (
            id BIGINT, type VARCHAR, wkt VARCHAR,
            minlon DOUBLE, minlat DOUBLE, maxlon DOUBLE, maxlat DOUBLE
        );
    """)

    handler = Handler(conn)
    print(f"Parsing PBF... (may take 20-60 min for Ukraine)")
    t0 = time.time()
    handler.apply_file(str(pbf_path), locations=True)
    handler._flush()
    elapsed = time.time() - t0
    print(f"Parsing done in {elapsed:.1f}s")

    # Створюємо індекси для швидких bbox-queries
    print("Creating bbox indexes...")
    for table in ("buildings", "roads", "bridges", "water", "parks"):
        conn.execute(f"CREATE INDEX idx_{table}_lon ON {table}(minlon, maxlon)")
        conn.execute(f"CREATE INDEX idx_{table}_lat ON {table}(minlat, maxlat)")

    # Підсумок
    for table in ("buildings", "roads", "bridges", "water", "parks"):
        cnt = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {cnt} rows")

    conn.close()
    size_mb = db_path.stat().st_size / 1024 / 1024
    print(f"DB built: {db_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
