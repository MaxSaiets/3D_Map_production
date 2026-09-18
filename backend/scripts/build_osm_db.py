"""
build_osm_db.py — конвертує OSM PBF (Geofabrik Ukraine) у локальну DuckDB-БД.

Використовує osmium з DISK-BACKED node index — RAM ≤2-3 GB замість 15-30 GB.

Алгоритм:
1) Перший прохід: створюємо node index на диску (RAM ~500MB, диск ~3GB).
2) Другий прохід: парсимо way з координатами через цей index.
3) Пишемо в DuckDB батчами по 50K.

Запуск:
    python -u backend/scripts/build_osm_db.py /f/3dmap_data/osm/ukraine-latest.osm.pbf /f/3dmap_data/osm/ukraine.duckdb

Очікуваний час: 15-30 хв. Output: ~2-4 GB DuckDB.
"""
import gc
import sys
import time
from pathlib import Path

import duckdb
import osmium
import pandas as pd


HIGHWAY_TAGS = frozenset({"motorway", "trunk", "primary", "secondary", "tertiary",
                "residential", "unclassified", "service", "pedestrian",
                "footway", "path", "cycleway"})
# Залізниця — окремий OSM-клас (railway=*, не highway=*), тому донедавна не
# потрапляла в базу взагалі: вокзали й сортувальні станції виходили порожньою
# основою. Кладемо колії в ту саму таблицю roads під псевдо-класом
# highway='railway' — друкуються тим самим чорним філаментом, що й дороги.
RAILWAY_TAGS = frozenset({"rail", "light_rail", "narrow_gauge", "tram",
                          "subway", "funicular"})
RAILWAY_PSEUDO_HIGHWAY = "railway"
WATERWAY_TAGS = frozenset({"riverbank", "dock", "canal"})
LANDUSE_WATER = frozenset({"reservoir", "basin"})
LEISURE_PARK = frozenset({"park", "garden", "nature_reserve", "pitch", "playground", "golf_course"})
LANDUSE_PARK = frozenset({"grass", "meadow", "forest", "village_green", "cemetery", "allotments", "orchard"})
NATURAL_PARK = frozenset({"wood", "grassland", "scrub", "heath"})

# Великий batch — DataFrame bulk insert у 100× швидше за executemany
BATCH_SIZE = 200_000

MONUMENT_HISTORIC = {"monument", "memorial"}
MONUMENT_POINT_RADIUS_M = 7.0  # footprint для монумент-точок (Ø14 м ≈ 1.4 мм на мапі 150 мм; Ø10 давав голку)
MONUMENT_MAN_MADE = {"monument", "obelisk", "tower"}


def _parse_height(tags) -> float:
    """Явна висота у метрах з OSM height / building:height ("20", "20 m", "65 ft")."""
    import re as _re
    height = 0.0
    for hk in ("height", "building:height"):
        hv = tags.get(hk)
        if not hv:
            continue
        m = _re.search(r"[-+]?\d+(?:\.\d+)?", str(hv).replace(",", "."))
        if not m:
            continue
        try:
            h = float(m.group(0))
            sl = str(hv).lower()
            if "ft" in sl or "feet" in sl or "'" in sl:
                h *= 0.3048
            height = max(height, h)
        except Exception:
            pass
    return height


BUILDING_COLS = ["id", "levels", "height", "landmark", "wkt", "minlon", "minlat", "maxlon", "maxlat"]
ROAD_COLS = ["id", "highway", "bridge", "wkt", "minlon", "minlat", "maxlon", "maxlat"]
BRIDGE_COLS = ["id", "highway", "wkt", "minlon", "minlat", "maxlon", "maxlat"]
WATER_COLS = ["id", "type", "wkt", "minlon", "minlat", "maxlon", "maxlat"]
PARKS_COLS = ROAD_COLS  # type замість highway, але мейпинг той самий
PARKS_COLS = ["id", "type", "wkt", "minlon", "minlat", "maxlon", "maxlat"]


class FastHandler(osmium.SimpleHandler):
    """Streaming-handler з рукописною WKT-генерацією (10× швидше за shapely).
    Підтримує both ways AND areas (multipolygon relations) — критично для
    великих водойм типу Дніпра."""

    def __init__(self, conn):
        super().__init__()
        self.conn = conn
        self.wkt_factory = osmium.geom.WKTFactory()
        self.buildings = []
        self.roads = []
        self.bridges = []
        self.water = []
        self.parks = []
        # 18.09.2026: монументи-ТОЧКИ з висотою (Батьківщина-Мати = node
        # historic=monument height=102, а полігон постаменту поруч висоти не має).
        # Osmium віддає nodes ДО ways/areas, тож у area() ці точки вже зібрані і
        # монумент-полігон отримує висоту точки всередині свого bbox.
        self.monument_pts = []  # (lon, lat, height_m)
        self.tot = {"buildings": 0, "roads": 0, "bridges": 0, "water": 0, "parks": 0}
        self.last_log = time.time()
        self.start = time.time()
        self.way_count = 0
        self.area_count = 0

    def node(self, n):
        tags = n.tags
        if not tags or tags.get("historic") not in MONUMENT_HISTORIC:
            return
        h = _parse_height(tags)
        if h <= 0:
            return
        try:
            self.monument_pts.append((int(n.id), float(n.location.lon), float(n.location.lat), h))
        except Exception:
            pass

    def emit_monument_points(self):
        """Монументи-ТОЧКИ з висотою → синтетичний footprint-«будівля» (восьмикутник
        R=MONUMENT_POINT_RADIUS_M) з висотою точки і landmark='historic'. Так
        Батьківщина-Мати (node height=102 без полігона) стає стрункою вежею на
        мапі, а не витягує цілий музей під собою на 102 м (перевірено — силос).
        id = −node_id, щоб не перетинатись із way/relation-id."""
        import math
        n_added = 0
        for nid, lon, lat, h in self.monument_pts:
            r_lat = MONUMENT_POINT_RADIUS_M / 111320.0
            r_lon = r_lat / max(math.cos(math.radians(lat)), 1e-6)
            ring = []
            for k in range(8):
                a = 2.0 * math.pi * k / 8.0
                ring.append(f"{lon + r_lon * math.cos(a):.7f} {lat + r_lat * math.sin(a):.7f}")
            ring.append(ring[0])
            wkt = "MULTIPOLYGON(((" + ",".join(ring) + ")))"
            self.buildings.append((-int(nid), 0, float(h), "historic", wkt,
                                   lon - r_lon, lat - r_lat, lon + r_lon, lat + r_lat))
            n_added += 1
        if self.buildings:
            self._flush()
        print(f"  monument points → synthetic landmark footprints: {n_added}", flush=True)

    def area(self, a):
        """Зібрана area — закритий way АБО multipolygon relation.
        Критично для Дніпра, великих лісів і парків (relation outer).
        Обробляє: water, parks, buildings (все що є polygon)."""
        self.area_count += 1
        tags = a.tags
        if not tags:
            return
        # Монументи/меморіали/обеліски/вежі як «будівлі»-орієнтири (landmark=historic),
        # інакше Батьківщина-Мати, Мотрин-Мати тощо ніколи не потрапляють у модель.
        is_monument = (
            tags.get("historic") in MONUMENT_HISTORIC
            or tags.get("man_made") in MONUMENT_MAN_MADE
        )
        is_building = "building" in tags or is_monument
        nat = tags.get("natural")
        wway = tags.get("waterway")
        lu = tags.get("landuse")
        leis = tags.get("leisure")
        is_water = (
            nat == "water"
            or wway in WATERWAY_TAGS
            or lu in LANDUSE_WATER
        )
        is_park = (
            leis in LEISURE_PARK
            or lu in LANDUSE_PARK
            or nat in NATURAL_PARK
        )
        if not (is_water or is_park or is_building):
            return
        try:
            wkt = self.wkt_factory.create_multipolygon(a)
        except Exception:
            return
        # Bbox з WKT (швидше за shapely)
        import re
        nums = re.findall(r"-?\d+\.?\d*", wkt)
        if len(nums) < 4:
            return
        lons = [float(nums[i]) for i in range(0, len(nums), 2)]
        lats = [float(nums[i]) for i in range(1, len(nums), 2)]
        if not lons or not lats:
            return
        minlon, maxlon = min(lons), max(lons)
        minlat, maxlat = min(lats), max(lats)
        rid = a.orig_id() if hasattr(a, "orig_id") else (a.id if hasattr(a, "id") else 0)
        if is_building:
            try:
                levels = int(float(tags.get("building:levels") or 0))
            except Exception:
                levels = 0
            # Явна висота у метрах (OSM height / building:height; "20", "20 m", "65 ft").
            # Точніша за levels×3 — використовується першочергово у get_building_height.
            height = _parse_height(tags)
            # Орієнтир (визначне місце): церква/вежа/історична/пам'ятка → окрема
            # категорія для кольору + збереження навіть малих footprint у генерації.
            bt = tags.get("building") or ""
            if tags.get("historic") or is_monument:
                landmark = "historic"
            elif tags.get("man_made") == "tower" or bt == "tower":
                landmark = "tower"
            elif tags.get("amenity") == "place_of_worship" or bt in ("church", "cathedral", "chapel", "mosque", "temple", "synagogue"):
                landmark = "worship"
            elif tags.get("tourism") in ("attraction", "museum") or bt == "castle":
                landmark = "attraction"
            else:
                landmark = ""
            self.buildings.append((rid, levels, height, landmark, wkt, minlon, minlat, maxlon, maxlat))
        if is_water:
            self.water.append((rid, nat or wway or lu or "water", wkt, minlon, minlat, maxlon, maxlat))
        if is_park:
            ptype = leis or lu or nat or "park"
            self.parks.append((rid, ptype, wkt, minlon, minlat, maxlon, maxlat))
        total_batch = (len(self.buildings) + len(self.roads) + len(self.bridges)
                       + len(self.water) + len(self.parks))
        if total_batch >= BATCH_SIZE:
            self._flush()

    def way(self, w):
        """Only LINEAR features: highway, bridge. Polygons (building/water/park)
        обробляються в area() — він краще handles relations multipolygon."""
        self.way_count += 1
        tags = w.tags
        if not tags or len(w.nodes) < 2:
            return
        has_highway = "highway" in tags
        has_bridge = "bridge" in tags
        # Підземка (метро/тунелі) на поверхні не існує — у друковану мапу не йде.
        _tunnel = (tags.get("tunnel") or "").strip().lower()
        _layer = tags.get("layer")
        try:
            _underground = (_tunnel not in ("", "no")) or (float(_layer) < 0 if _layer else False)
        except (TypeError, ValueError):
            _underground = _tunnel not in ("", "no")
        has_railway = tags.get("railway") in RAILWAY_TAGS and not _underground
        # Building/water/park НЕ обробляємо тут — лише area()
        if not (has_highway or has_bridge or has_railway):
            return

        # Координати з locations index
        coords = []
        try:
            for n in w.nodes:
                if n.location.valid():
                    coords.append((n.lon, n.lat))
        except Exception:
            return
        if len(coords) < 2:
            return

        # Bbox + WKT — ОДНА ітерація, без зайвих shapely
        minlon = maxlon = coords[0][0]
        minlat = maxlat = coords[0][1]
        for lon, lat in coords[1:]:
            if lon < minlon: minlon = lon
            elif lon > maxlon: maxlon = lon
            if lat < minlat: minlat = lat
            elif lat > maxlat: maxlat = lat

        is_closed = len(coords) >= 4 and coords[0] == coords[-1]
        # Рукописний WKT — у 10-50× швидше за shapely.wkt
        coords_str = ", ".join(f"{lon:.7f} {lat:.7f}" for lon, lat in coords)
        wkt_line = "LINESTRING(" + coords_str + ")"
        wkt_poly = "POLYGON((" + coords_str + "))" if is_closed else wkt_line
        wid = w.id

        # Bridge — linear, тут
        if has_bridge:
            br = tags.get("bridge")
            if br and br not in ("no", ""):
                self.bridges.append((wid, tags.get("highway") or "primary", wkt_line, minlon, minlat, maxlon, maxlat))

        # Road — linear, тут
        if has_highway:
            hw = tags.get("highway")
            if hw in HIGHWAY_TAGS:
                br = tags.get("bridge") or "no"
                self.roads.append((wid, hw, br, wkt_line, minlon, minlat, maxlon, maxlat))

        # Залізниця — у ту саму таблицю roads під псевдо-класом 'railway'.
        # Гілка окрема від has_highway: у OSM колія рідко несе highway-тег, а
        # коли несе (переїзд) — обидві геометрії потрібні.
        if has_railway:
            br = tags.get("bridge") or "no"
            self.roads.append((wid, RAILWAY_PSEUDO_HIGHWAY, br, wkt_line,
                               minlon, minlat, maxlon, maxlat))

        # Building/water/parks обробляються в area() — підтримує multipolygon relations

        # Flush якщо батч переповнено
        total_batch = (len(self.buildings) + len(self.roads) + len(self.bridges)
                       + len(self.water) + len(self.parks))
        if total_batch >= BATCH_SIZE:
            self._flush()
            now = time.time()
            if now - self.last_log >= 15:
                self.last_log = now
                el = now - self.start
                rate = self.way_count / max(el, 1)
                print(f"  [{el/60:.1f}m | {self.way_count} ways, {rate:.0f}/s] "
                      f"b={self.tot['buildings']} r={self.tot['roads']} "
                      f"br={self.tot['bridges']} w={self.tot['water']} p={self.tot['parks']}", flush=True)

    def _flush(self):
        """DataFrame bulk insert — у 100× швидше за executemany."""
        if self.buildings:
            df = pd.DataFrame(self.buildings, columns=BUILDING_COLS)
            self.conn.execute("INSERT INTO buildings SELECT * FROM df")
            self.tot["buildings"] += len(self.buildings); self.buildings.clear()
        if self.roads:
            df = pd.DataFrame(self.roads, columns=ROAD_COLS)
            self.conn.execute("INSERT INTO roads SELECT * FROM df")
            self.tot["roads"] += len(self.roads); self.roads.clear()
        if self.bridges:
            df = pd.DataFrame(self.bridges, columns=BRIDGE_COLS)
            self.conn.execute("INSERT INTO bridges SELECT * FROM df")
            self.tot["bridges"] += len(self.bridges); self.bridges.clear()
        if self.water:
            df = pd.DataFrame(self.water, columns=WATER_COLS)
            self.conn.execute("INSERT INTO water SELECT * FROM df")
            self.tot["water"] += len(self.water); self.water.clear()
        if self.parks:
            df = pd.DataFrame(self.parks, columns=PARKS_COLS)
            self.conn.execute("INSERT INTO parks SELECT * FROM df")
            self.tot["parks"] += len(self.parks); self.parks.clear()
        gc.collect()


def main():
    if len(sys.argv) < 3:
        print("Usage: python build_osm_db.py <input.pbf> <output.duckdb>", flush=True)
        sys.exit(1)
    pbf_path = Path(sys.argv[1])
    db_path = Path(sys.argv[2])
    if not pbf_path.exists():
        print(f"PBF not found: {pbf_path}", flush=True)
        sys.exit(1)
    if db_path.exists():
        db_path.unlink()
    wal = db_path.with_suffix(db_path.suffix + ".wal")
    if wal.exists():
        wal.unlink()

    # In-RAM compact node index: ~1.5 GB RAM для України (вкладається у 8GB).
    # Disk-backed dense_file_array нестерпно повільний на Windows (random IO).
    # sparse_mem_array — стандартний швидкий варіант для країнових PBF.
    print(f"Building DuckDB from {pbf_path} -> {db_path}", flush=True)
    print(f"Node index: in-RAM sparse_mem_array (~1.5GB RAM, fast)", flush=True)

    conn = duckdb.connect(str(db_path))
    # Limit DuckDB memory to 2GB (we have 8GB total budget, leave RAM for osmium)
    conn.execute("SET memory_limit='2GB'")
    conn.execute("""
        CREATE TABLE buildings (
            id BIGINT, levels INTEGER, height DOUBLE, landmark VARCHAR, wkt VARCHAR,
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

    handler = FastHandler(conn)
    print(f"Parsing PBF with in-RAM sparse_mem_array (fast)...", flush=True)
    t0 = time.time()
    # locations=True використовує дефолтний sparse_mem_array — швидкий і ОК для України
    handler.apply_file(str(pbf_path), locations=True)
    handler._flush()
    handler.emit_monument_points()
    elapsed = time.time() - t0
    print(f"Parsing done in {elapsed/60:.1f}m", flush=True)
    print(f"Final totals: {handler.tot}", flush=True)

    print("Creating bbox indexes...", flush=True)
    for table in ("buildings", "roads", "bridges", "water", "parks"):
        conn.execute(f"CREATE INDEX idx_{table}_lon ON {table}(minlon, maxlon)")
        conn.execute(f"CREATE INDEX idx_{table}_lat ON {table}(minlat, maxlat)")

    for table in ("buildings", "roads", "bridges", "water", "parks"):
        cnt = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {cnt} rows", flush=True)

    conn.close()
    size_mb = db_path.stat().st_size / 1024 / 1024
    print(f"DB built: {db_path} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
