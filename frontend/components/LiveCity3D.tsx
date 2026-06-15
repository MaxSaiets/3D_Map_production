"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslations } from "next-intl";

type Bounds = { north: number; south: number; east: number; west: number };

type DesignShape = {
  bodyWidthMm: number;
  bodyHeightMm: number;
  cornerRadiusMm: number;
  mapXMm: number;
  mapYMm: number;
  mapWidthMm: number;
  mapHeightMm: number;
  loopXMm: number;
  loopYMm: number;
  loopOuterMm: number;
  loopInnerMm: number;
  rimWidthMm: number;
  baseShape: "rounded" | "capsule" | "tag" | "octagon" | "token";
};

type Pts = Array<[number, number]>;
type BuildingRec = { points: Pts; levels: number };
type RoadRec = { points: Pts; widthM: number; kind: "major" | "minor" | "service" };
type FountainRec = { lon: number; lat: number; radiusM: number };
type CityData = {
  buildings: BuildingRec[];
  roads: RoadRec[];
  water: Pts[];
  parks: Pts[];
  plazas: Pts[];
  fountains: FountainRec[];
  trees: FountainRec[];
  bridges: Array<{ points: Pts; widthM: number }>;
};

const OVERPASS_URLS = [
  "https://overpass-api.de/api/interpreter",
  "https://overpass.kumi.systems/api/interpreter",
];

// Мінімальна друкована деталь — все що менше, не друкується якісно у 0.4mm соплі
const MIN_PRINT_MM = 0.6;

// LRU-кеш OSM-відповідей по bbox-ключу. Друге заходження в ту саму зону —
// миттєве (без 3-10 сек запиту до Overpass).
const OSM_CACHE = new Map<string, any>();
const OSM_CACHE_MAX = 20;
const bboxKey = (b: Bounds) =>
  `${b.south.toFixed(5)},${b.west.toFixed(5)},${b.north.toFixed(5)},${b.east.toFixed(5)}`;

async function fetchFromLocalDB(b: Bounds, abortSignal?: AbortSignal): Promise<CityData | null> {
  // SPRINT 1: спершу пробуємо локальну DuckDB (50-200ms). Fallback на Overpass.
  const apiUrl = (process.env.NEXT_PUBLIC_API_URL || "").replace(/\/$/, "") + "/api/osm/extract";
  try {
    const res = await fetch(
      `${apiUrl}?north=${b.north}&south=${b.south}&east=${b.east}&west=${b.west}`,
      { signal: abortSignal },
    );
    if (!res.ok) return null;
    const data = await res.json();
    if (data.source !== "local") return null;
    // Парсимо WKT → точки (тільки LINESTRING і POLYGON для нашого випадку)
    const parseWkt = (wkt: string): Pts | null => {
      if (!wkt) return null;
      const m = wkt.match(/\(\(([^()]+)\)\)|\(([^()]+)\)/);
      if (!m) return null;
      const coordsStr = m[1] || m[2];
      return coordsStr.split(",").map((p) => {
        const [lon, lat] = p.trim().split(/\s+/).map(Number);
        return [lon, lat] as [number, number];
      }).filter((c) => isFinite(c[0]) && isFinite(c[1]));
    };
    const buildings: BuildingRec[] = (data.buildings || []).map((b: any) => {
      const pts = parseWkt(b.wkt);
      return pts ? { points: pts, levels: Math.max(1, Math.min(40, Number(b.levels) || 3)) } : null;
    }).filter(Boolean);
    const roadWidths: Record<string, number> = {
      motorway: 14, trunk: 12, primary: 10, secondary: 8, tertiary: 7,
      residential: 5, unclassified: 5, service: 3.5, pedestrian: 4,
    };
    const roads: RoadRec[] = (data.roads || []).map((r: any) => {
      if (!roadWidths[r.highway]) return null;
      const pts = parseWkt(r.wkt);
      if (!pts) return null;
      const kind: RoadRec["kind"] =
        ["motorway","trunk","primary","secondary"].includes(r.highway) ? "major"
        : ["residential","tertiary","unclassified"].includes(r.highway) ? "minor" : "service";
      return { points: pts, widthM: roadWidths[r.highway], kind };
    }).filter(Boolean);
    const water: Pts[] = (data.water || []).map((w: any) => parseWkt(w.wkt)).filter(Boolean) as Pts[];
    const parks: Pts[] = (data.parks || []).map((p: any) => parseWkt(p.wkt)).filter(Boolean) as Pts[];
    // The local DuckDB only covers Ukraine. For a bbox it doesn't cover (a
    // foreign city) it returns source="local" but with empty arrays — which
    // would render a blank preview. Treat "no buildings AND no roads" as a
    // miss so the caller falls back to the worldwide Overpass query.
    if (buildings.length === 0 && roads.length === 0) {
      return null;
    }
    // CityData type вимагає plazas, fountains, trees, bridges — заповнюємо порожніми
    // (вони не друкуються у моделі і прибрані з рендеру у Sprint 3.5)
    return {
      buildings, roads, water, parks,
      plazas: [] as Pts[],
      fountains: [] as any[],
      trees: [] as any[],
      bridges: [] as any[],
    } as any;
  } catch (e: any) {
    if (e.name === "AbortError") throw e;
    return null;
  }
}

async function fetchOSMForBounds(b: Bounds, abortSignal?: AbortSignal): Promise<CityData> {
  // Спершу — локальна DuckDB. Якщо немає або помилка — Overpass.
  const local = await fetchFromLocalDB(b, abortSignal);
  if (local !== null) {
    return local;
  }
  const bbox = `${b.south},${b.west},${b.north},${b.east}`;
  // Додаємо: area:highway=pedestrian (площі типу Майдан), place=square, landuse=pedestrian
  // ЛИШЕ ті теги що реально друкуються у 3MF backend:
  // - buildings, water polygons, parks polygons
  // - roads ТІЛЬКИ значимої ширини (без footway/path/cycleway — вони відсіються
  //   фільтром min_feature 0.5mm у backend, тож показувати їх у превʼю — обман)
  // - bridges (з road network)
  const q = `[out:json][timeout:12];(way["building"](${bbox});way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|service|pedestrian)$"](${bbox});way["natural"~"^(water|wood)$"](${bbox});way["waterway"~"^(riverbank|dock)$"](${bbox});way["leisure"~"^(park|garden|nature_reserve)$"](${bbox});way["landuse"~"^(forest|grass|cemetery)$"](${bbox});way["bridge"="yes"](${bbox});relation["natural"="water"](${bbox});relation["leisure"="park"](${bbox});relation["landuse"="forest"](${bbox}););out geom;`;
  let lastErr: any = null;
  for (const url of OVERPASS_URLS) {
    try {
      const res = await fetch(url, {
        method: "POST",
        body: q,
        headers: { "Content-Type": "text/plain" },
        signal: abortSignal,
      });
      if (!res.ok) { lastErr = new Error("Overpass " + res.status); continue; }
      const data = await res.json();
      const buildings: BuildingRec[] = [];
      const roads: RoadRec[] = [];
      const water: Pts[] = [];
      const parks: Pts[] = [];
      const plazas: Pts[] = [];
      const fountains: FountainRec[] = [];
      const trees: FountainRec[] = [];   // дерева як крапки-кружки
      const bridges: Array<{ points: Pts; widthM: number }> = [];

      // Замкнутий полігон = перша точка == остання (з допуском floating point)
      const isClosedRing = (pts: Pts) =>
        pts.length >= 4 &&
        Math.abs(pts[0][0] - pts[pts.length - 1][0]) < 1e-9 &&
        Math.abs(pts[0][1] - pts[pts.length - 1][1]) < 1e-9;

      function classifyAndPush(tags: any, points: Pts, fromRelation = false) {
        if (!points || points.length < 2) return;
        // Для members з relation — closing робиться на рівні relation
        // (стичка кількох ways у замкнуте кільце). Окремі way-members ЗДЕБІЛЬШОГО
        // не closed, тож вимога isClosedRing для них некоректна.
        const polygonOk = (pts: Pts) => fromRelation ? pts.length >= 3 : isClosedRing(pts);
        const isPlaza =
          tags.place === "square" || tags.place === "plaza" ||
          tags.landuse === "pedestrian" ||
          (tags.highway === "pedestrian" && (tags.area === "yes" || isClosedRing(points)));
        const isBridge = tags.bridge === "yes" || tags.man_made === "bridge";
        if (tags.building) {
          buildings.push({ points, levels: Math.max(1, Math.min(40, Number(tags["building:levels"]) || 3)) });
        } else if (isPlaza && polygonOk(points)) {
          // Площа (Майдан, променад) — окремий шар, не дорога і не вода
          plazas.push(points);
        } else if (isBridge && !tags.building) {
          // Міст — рендериться поверх води, ширший за звичайну дорогу
          const widths: Record<string, number> = {
            motorway: 16, trunk: 14, primary: 12, secondary: 10, tertiary: 9,
            residential: 7, unclassified: 7, service: 5, pedestrian: 5,
            footway: 3, path: 3, cycleway: 3,
          };
          const w = widths[String(tags.highway)] || 8;
          bridges.push({ points, widthM: w });
        } else if (tags.highway) {
          // ТІЛЬКИ дороги що реально друкуються — без footway/path/cycleway.
          // Ці тонкі стежки в 0.4mm соплі неможливо надрукувати, у backend їх
          // дропає фільтр min_feature_m, тож і в превʼю не показуємо (інакше
          // обман — у моделі їх не буде).
          const widths: Record<string, number> = {
            motorway: 14, trunk: 12, primary: 10, secondary: 8, tertiary: 7,
            residential: 5, unclassified: 5, service: 3.5, pedestrian: 4,
          };
          if (!widths[String(tags.highway)]) return;  // пропускаємо footway/path/etc
          const kind: RoadRec["kind"] =
            ["motorway", "trunk", "primary", "secondary"].includes(String(tags.highway)) ? "major"
            : ["residential", "tertiary", "unclassified"].includes(String(tags.highway)) ? "minor" : "service";
          roads.push({ points, widthM: widths[String(tags.highway)] || 4, kind });
        } else if (tags.natural === "water" || tags.waterway === "riverbank" || tags.waterway === "dock") {
          // ТІЛЬКИ полігони. waterway=river/stream/ditch — це лінії (way без area).
          // Для members з relation — кільце замикається на рівні relation,
          // тож не вимагаємо closed ring окремо.
          if (polygonOk(points)) water.push(points);
        } else if (tags.leisure || tags.landuse || tags.natural) {
          const isGreen =
            ["park", "garden", "pitch", "playground", "nature_reserve", "golf_course"].includes(tags.leisure) ||
            ["grass", "forest", "recreation_ground", "meadow", "village_green", "cemetery", "allotments", "orchard"].includes(tags.landuse) ||
            ["wood", "grassland", "scrub", "heath"].includes(tags.natural);
          // Зелень теж: standalone way → closed ring; relation member → ok як ≥3 точок.
          if (isGreen && polygonOk(points)) parks.push(points);
        }
      }

      // Перевірка чи лінія/полігон ПЕРЕТИНАЄ bbox (не тільки має точки всередині).
      // Для великої води (Дніпро) — її outer way може не мати ВЛАСНИХ точок
      // у малому bbox, але berm тягнеться через нього. Тоді перевіряємо чи
      // bbox center попадає в полігон.
      const inBbox = (pts: Pts) =>
        pts.some(([lon, lat]) => lon >= b.west && lon <= b.east && lat >= b.south && lat <= b.north);
      const lineIntersectsBbox = (pts: Pts) => {
        if (inBbox(pts)) return true;
        // Bbox полігона перетинає селекцію?
        let n = -Infinity, s = Infinity, e = -Infinity, w = Infinity;
        for (const [lon, lat] of pts) {
          if (lat > n) n = lat; if (lat < s) s = lat;
          if (lon > e) e = lon; if (lon < w) w = lon;
        }
        return !(e < b.west || w > b.east || n < b.south || s > b.north);
      };

      for (const el of data.elements || []) {
        const tags = el.tags || {};
        if (el.type === "node" && tags.amenity === "fountain") {
          if (el.lon >= b.west && el.lon <= b.east && el.lat >= b.south && el.lat <= b.north) {
            fountains.push({ lon: el.lon, lat: el.lat, radiusM: 3.5 });
          }
        } else if (el.type === "node" && tags.natural === "tree") {
          if (el.lon >= b.west && el.lon <= b.east && el.lat >= b.south && el.lat <= b.north) {
            trees.push({ lon: el.lon, lat: el.lat, radiusM: 2.0 });
          }
        } else if (el.type === "way" && el.geometry) {
          const points: Pts = el.geometry.map((g: any) => [g.lon, g.lat]);
          if (lineIntersectsBbox(points)) classifyAndPush(tags, points);
        } else if (el.type === "relation" && el.members) {
          // Multipolygon: outers — це АРКИ (шматки), які формують кільце.
          // Дніпро: outer = довжелезна лінія по західному березі, інша по
          // східному, плюс торці. Окремий arc НЕ замкнутий і fill дає рандом.
          // Тож стичкуємо outers по spільних endpoints у замкнуті кільця.
          const outerArcs: Pts[] = [];
          for (const m of el.members) {
            if (m.type === "way" && m.geometry && m.geometry.length >= 2 && m.role !== "inner") {
              outerArcs.push(m.geometry.map((g: any) => [g.lon, g.lat]));
            }
          }
          const stitched = stitchArcs(outerArcs);
          for (const ring of stitched) {
            if (lineIntersectsBbox(ring)) classifyAndPush(tags, ring, true);
          }
        }
      }
      return { buildings, roads, water, parks, plazas, fountains, trees, bridges } as any;
    } catch (e: any) {
      if (e.name === "AbortError") throw e;
      lastErr = e;
    }
  }
  throw lastErr || new Error("Overpass unreachable");
}

/** Стичкує OSM relation outer members (окремі arcs) у замкнуті кільця.
 *  Дніпро/великі парки — outer = N окремих ways по endpoint-match. Якщо
 *  залишився незамкнутий хвіст — замикаємо штучно (краще зайвий fill, ніж нічого). */
function stitchArcs(arcs: Pts[]): Pts[] {
  if (arcs.length === 0) return [];
  if (arcs.length === 1) {
    const a = arcs[0];
    if (a.length < 2) return [];
    // Якщо вже closed — повертаємо як є
    if (Math.abs(a[0][0] - a[a.length - 1][0]) < 1e-9 && Math.abs(a[0][1] - a[a.length - 1][1]) < 1e-9) return [a];
    return [[...a, a[0]]]; // штучно замикаємо
  }
  const rings: Pts[] = [];
  const pool = arcs.map((a) => [...a]);  // копія, щоб мутувати
  const eq = (p: [number, number], q: [number, number]) =>
    Math.abs(p[0] - q[0]) < 1e-7 && Math.abs(p[1] - q[1]) < 1e-7;

  while (pool.length > 0) {
    const ring: Pts = pool.shift()!;
    let progress = true;
    while (progress && !(ring.length > 1 && eq(ring[0], ring[ring.length - 1]))) {
      progress = false;
      for (let i = 0; i < pool.length; i++) {
        const arc = pool[i];
        if (eq(ring[ring.length - 1], arc[0])) {
          ring.push(...arc.slice(1));
          pool.splice(i, 1); progress = true; break;
        } else if (eq(ring[ring.length - 1], arc[arc.length - 1])) {
          ring.push(...[...arc].reverse().slice(1));
          pool.splice(i, 1); progress = true; break;
        } else if (eq(ring[0], arc[arc.length - 1])) {
          ring.unshift(...arc.slice(0, -1));
          pool.splice(i, 1); progress = true; break;
        } else if (eq(ring[0], arc[0])) {
          ring.unshift(...[...arc].reverse().slice(0, -1));
          pool.splice(i, 1); progress = true; break;
        }
      }
    }
    // Якщо все одно не замкнулось (broken multipolygon) — закриваємо силою
    if (ring.length >= 3 && !eq(ring[0], ring[ring.length - 1])) {
      ring.push(ring[0]);
    }
    if (ring.length >= 4) rings.push(ring);
  }
  return rings;
}

function polygonArea(pts: Pts): number {
  let a = 0;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    a += (pts[j][0] + pts[i][0]) * (pts[j][1] - pts[i][1]);
  }
  return Math.abs(a / 2);
}

function pointsToPath(pts: Pts): string {
  if (pts.length === 0) return "";
  return "M" + pts.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" L") + "Z";
}

function polylineToPath(pts: Pts): string {
  if (pts.length === 0) return "";
  return "M" + pts.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" L");
}

type LiveCityProps = {
  bounds: Bounds;
  design: DesignShape;
  /** Поворот рамки на карті — preview обертає контент на -кут щоб ділянка
   *  виглядала axis-aligned у map slot. */
  cropRotationDeg?: number;
  /** 4 кути обернутого rect [[lon, lat], ...]. Preview clipує OSM по цьому
   *  полігону — показує ТІЛЬКИ те що буде на готовому 3MF, нічого зайвого. */
  cropPolygon?: Array<[number, number]> | null;
};

/** Спільна логіка: fetch OSM → проєкція lon/lat у мм слота → фільтр друкованих
 *  фіч. Повертає готові до рендеру шляхи у мм-координатах батьківського SVG. */
function useCityPrintable({ bounds, design, cropRotationDeg = 0, cropPolygon = null }: LiveCityProps) {
  const [data, setData] = useState<CityData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const reqRef = useRef<AbortController | null>(null);

  // Bbox для OSM-fetch — розширений до bbox ПОВЕРНУТОЇ зони, інакше при
  // куті ≠ 0 кути зони вилазять за axis-aligned bounds і дані не вантажаться.
  const fetchBounds = useMemo<Bounds>(() => {
    if (!cropPolygon || cropPolygon.length < 3) return bounds;
    let n = -Infinity, s = Infinity, e = -Infinity, w = Infinity;
    for (const [lon, lat] of cropPolygon) {
      if (lat > n) n = lat; if (lat < s) s = lat;
      if (lon > e) e = lon; if (lon < w) w = lon;
    }
    return {
      north: Math.max(n, bounds.north),
      south: Math.min(s, bounds.south),
      east: Math.max(e, bounds.east),
      west: Math.min(w, bounds.west),
    };
  }, [bounds, cropPolygon]);

  useEffect(() => {
    if (!fetchBounds) return;
    // Cache hit → миттєвий результат, без сетіменту loading
    const key = bboxKey(fetchBounds);
    const cached = OSM_CACHE.get(key);
    if (cached) {
      setData(cached);
      setError(null);
      return;
    }
    reqRef.current?.abort();
    const ctrl = new AbortController();
    reqRef.current = ctrl;
    const timer = setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const d = await fetchOSMForBounds(fetchBounds, ctrl.signal);
        if (!ctrl.signal.aborted) {
          // LRU: видаляємо найстаріший якщо переповнено
          if (OSM_CACHE.size >= OSM_CACHE_MAX) {
            const firstKey = OSM_CACHE.keys().next().value;
            if (firstKey) OSM_CACHE.delete(firstKey);
          }
          OSM_CACHE.set(key, d);
          setData(d);
        }
      } catch (e: any) {
        if (e.name !== "AbortError") setError(e.message || "Overpass");
      } finally {
        if (!ctrl.signal.aborted) setLoading(false);
      }
    }, 250);  // зменшено з 400 → швидший відгук при дрібних рухах
    return () => { clearTimeout(timer); ctrl.abort(); };
  }, [fetchBounds.north, fetchBounds.south, fetchBounds.east, fetchBounds.west]);

  // ПРАВИЛЬНА проекція: беремо ПОВЕРНУТУ зону як viewport.
  // 1) Центр зони (lon, lat) — або з cropPolygon, або з bounds.
  // 2) Розміри зони widthM × heightM — з bounds (bounds = bbox unrotated rect
  //    того ж widthM × heightM при тому ж центрі).
  // 3) Для кожної точки:
  //    a) у локальні метри від центру (east, north)
  //    b) повертаємо на -θ (інверсія повороту рамки) → координати у власних
  //       осях рамки
  //    c) скейлимо у слот мм
  // Результат: повернута зона ЗАВЖДИ ідеально лягає на слот, content обертається
  // разом з нею. Зона «розгортається», вміст «крутиться у середині», слот завжди
  // прямокутний і повністю заповнений.
  // Стабільний ключ полігона: bounds/cropPolygon приходять новими обʼєктами на
  // кожен рендер батька — без цього project (і вся проєкція printable) перераховувалися
  // б на КОЖЕН ре-рендер, навіть коли геометрія не змінилась (drag тексту, введення напису).
  const cropKey = cropPolygon && cropPolygon.length >= 3
    ? cropPolygon.map((p) => `${p[0].toFixed(7)},${p[1].toFixed(7)}`).join(";")
    : "";

  /* eslint-disable react-hooks/exhaustive-deps -- bounds/cropPolygon навмисно
     представлені примітивними deps (числа bbox + cropKey), щоб уникнути
     перерахунку через нову ідентичність обʼєктів. */
  const project = useMemo(() => {
    // Центр зони — від cropPolygon (точно) або з bounds (fallback)
    let cLon: number, cLat: number;
    if (cropPolygon && cropPolygon.length >= 3) {
      cLon = cropPolygon.reduce((a, p) => a + p[0], 0) / cropPolygon.length;
      cLat = cropPolygon.reduce((a, p) => a + p[1], 0) / cropPolygon.length;
    } else {
      cLon = (bounds.east + bounds.west) / 2;
      cLat = (bounds.north + bounds.south) / 2;
    }
    const mPerDegLng = 111_320 * Math.max(Math.cos((cLat * Math.PI) / 180), 0.18);
    const widthM = (bounds.east - bounds.west) * mPerDegLng || 1;
    const heightM = (bounds.north - bounds.south) * 111_320 || 1;
    const theta = (cropRotationDeg * Math.PI) / 180;
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);
    const mmPerM = Math.min(design.mapWidthMm / widthM, design.mapHeightMm / heightM);
    // COVER scale — uniform max, заповнює слот, обрізається при різній aspect
    const sx = design.mapWidthMm / widthM;
    const sy = design.mapHeightMm / heightM;
    const s = Math.max(sx, sy);
    const cxSlot = design.mapXMm + design.mapWidthMm / 2;
    const cySlot = design.mapYMm + design.mapHeightMm / 2;
    return {
      lonLatToMm: (lon: number, lat: number): [number, number] => {
        const dx = (lon - cLon) * mPerDegLng;
        const dy = (lat - cLat) * 111_320;
        const rx = dx * cosT + dy * sinT;
        const ry = -dx * sinT + dy * cosT;
        const mx = cxSlot + rx * s;
        const my = cySlot - ry * s;
        return [mx, my];
      },
      mmPerM,
    };
  }, [bounds.north, bounds.south, bounds.east, bounds.west, cropKey, cropRotationDeg, design.mapXMm, design.mapYMm, design.mapWidthMm, design.mapHeightMm]);
  /* eslint-enable react-hooks/exhaustive-deps */

  // Фільтр + конвертація — тільки те що буде друкуватися
  const printable = useMemo(() => {
    if (!data) return { buildings: [], roads: [], water: [], parks: [], plazas: [], fountains: [], trees: [], bridges: [] };
    const { lonLatToMm, mmPerM } = project;
    // Buildings: пропускаємо ті де min-розмір < MIN_PRINT_MM
    const buildings = data.buildings
      .map((b) => b.points.map(([lon, lat]) => lonLatToMm(lon, lat)))
      .filter((pts) => {
        if (pts.length < 3) return false;
        const area = polygonArea(pts);
        // М'якший поріг — показуємо все що ≥ 0.5×0.5mm (видимий блок).
        // Зайве "видалення" робилось бекендом, тут preview має бути informative.
        return area >= 0.25;
      });
    // Roads: переводимо реальну ширину м → mm і пропускаємо тонкі
    const roads = data.roads
      .map((r) => ({
        path: polylineToPath(r.points.map(([lon, lat]) => lonLatToMm(lon, lat))),
        widthMm: Math.max(MIN_PRINT_MM, r.widthM * mmPerM),
        kind: r.kind,
      }))
      .filter((r) => r.widthMm >= MIN_PRINT_MM * 0.8);  // дороги тонші 0.5mm дропаємо
    const water = data.water
      .map((pts) => pts.map(([lon, lat]) => lonLatToMm(lon, lat)))
      .filter((pts) => pts.length >= 3 && polygonArea(pts) >= 4)  // ≥ 2×2 mm
      .sort((a, b) => polygonArea(b) - polygonArea(a));            // великі знизу
    const parks = data.parks
      .map((pts) => pts.map(([lon, lat]) => lonLatToMm(lon, lat)))
      .filter((pts) => pts.length >= 3 && polygonArea(pts) >= 4)
      .sort((a, b) => polygonArea(b) - polygonArea(a));
    const plazas = data.plazas
      .map((pts) => pts.map(([lon, lat]) => lonLatToMm(lon, lat)))
      .filter((pts) => pts.length >= 3 && polygonArea(pts) >= 4)
      .sort((a, b) => polygonArea(b) - polygonArea(a));
    const bridges = data.bridges.map((br) => ({
      path: polylineToPath(br.points.map(([lon, lat]) => lonLatToMm(lon, lat))),
      widthMm: Math.max(MIN_PRINT_MM * 1.5, br.widthM * mmPerM),
    }));
    const fountains = data.fountains
      .map((f) => ({
        center: lonLatToMm(f.lon, f.lat),
        rMm: Math.max(0.8, f.radiusM * mmPerM),
      }));
    const trees = data.trees
      .map((t) => ({
        center: lonLatToMm(t.lon, t.lat),
        rMm: Math.max(0.6, t.radiusM * mmPerM),
      }));
    return { buildings, roads, water, parks, plazas, bridges, fountains, trees };
  }, [data, project]);

  return { printable, loading, error, hasData: !!data };
}

/** Феатур-шляхи міста як набір SVG-елементів (parks/water/roads/bridges/buildings)
 *  у мм-координатах слота. Вже спроектовані — рендеряться прямо у батьківському
 *  SVG. Окрема функція, щоб однаково використати і в standalone, і в layer. */
function CityFeaturePaths({ printable }: { printable: ReturnType<typeof useCityPrintable>["printable"] }) {
  return (
    <>
      {printable.parks.map((pts, i) => (
        <path key={`p-${i}`} d={pointsToPath(pts)} fill="#88b06e" />
      ))}
      {printable.water.map((pts, i) => (
        <path key={`w-${i}`} d={pointsToPath(pts)} fill="#5a91c4" />
      ))}
      {printable.roads.map((r, i) => (
        <path
          key={`r-${i}`}
          d={r.path}
          stroke={r.kind === "major" ? "#1a1a1a" : r.kind === "minor" ? "#3a3a3a" : "#5a5a5a"}
          strokeWidth={r.widthMm}
          fill="none"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      ))}
      {printable.bridges.map((br, i) => (
        <path
          key={`br-${i}`}
          d={br.path}
          stroke="#2a2a2a"
          strokeWidth={br.widthMm}
          fill="none"
          strokeLinecap="butt"
          strokeLinejoin="round"
        />
      ))}
      {printable.buildings.map((pts, i) => (
        <path key={`b-${i}`} d={pointsToPath(pts)} fill="#cfc1a3" stroke="#a89a7d" strokeWidth={0.15} />
      ))}
    </>
  );
}

/**
 * SVG-LAYER варіант: повертає лише <g> з нативними SVG-шляхами для ВБУДОВУВАННЯ
 * у батьківський SVG (KeychainDesigner). Це КЛЮЧОВО для iOS/Safari: попередній
 * варіант через <foreignObject> на Safari ігнорував x/y/transform і малював
 * прев'ю у лівому верхньому куті (WebKit bug #23113). Нативні шляхи у спільній
 * мм-системі координат позиціонуються правильно скрізь і успадковують поворот/кліп.
 */
export function LiveCitySvgPaths(props: LiveCityProps) {
  const t = useTranslations("live");
  const { printable, loading, error, hasData } = useCityPrintable(props);
  const { design } = props;
  const cx = design.mapXMm + design.mapWidthMm / 2;
  const cy = design.mapYMm + design.mapHeightMm / 2;
  return (
    <g>
      <CityFeaturePaths printable={printable} />
      {loading && (
        <g pointerEvents="none">
          <circle cx={cx} cy={cy} r={Math.min(design.mapWidthMm, design.mapHeightMm) * 0.09} fill="none" stroke="#0f766e" strokeWidth={0.6} strokeDasharray="2 1.4">
            <animateTransform attributeName="transform" type="rotate" from={`0 ${cx} ${cy}`} to={`360 ${cx} ${cy}`} dur="0.9s" repeatCount="indefinite" />
          </circle>
        </g>
      )}
      {error && (
        <text x={cx} y={design.mapYMm + design.mapHeightMm - 1.5} textAnchor="middle" fontSize={1.8} fill="#b91c1c">
          {error.slice(0, 28)}
        </text>
      )}
      {!hasData && !loading && !error && (
        <text x={cx} y={cy} textAnchor="middle" dominantBaseline="middle" fontSize={2.2} fontWeight={700} fill="#6a5d44">
          {t("pickAreaOnMap")}
        </text>
      )}
    </g>
  );
}

/** Чистий 2D preview ділянки (standalone div+svg). Зберігається для можливого
 *  окремого використання; на брелку застосовується LiveCitySvgPaths. */
export function LiveCity3D(props: LiveCityProps) {
  const t = useTranslations("live");
  const { printable, loading, error, hasData } = useCityPrintable(props);
  const { design } = props;
  const vb = `${design.mapXMm} ${design.mapYMm} ${design.mapWidthMm} ${design.mapHeightMm}`;
  const clipId = useMemo(() => `liveCityClip-${Math.random().toString(36).slice(2, 8)}`, []);
  return (
    <div className="relative h-full w-full overflow-hidden bg-[#e0d4b5]">
      <svg viewBox={vb} preserveAspectRatio="xMidYMid meet" style={{ width: "100%", height: "100%", display: "block" }}>
        <defs>
          <clipPath id={clipId}>
            <rect x={design.mapXMm} y={design.mapYMm} width={design.mapWidthMm} height={design.mapHeightMm} />
          </clipPath>
        </defs>
        <rect x={design.mapXMm} y={design.mapYMm} width={design.mapWidthMm} height={design.mapHeightMm} fill="#e0d4b5" />
        <g clipPath={`url(#${clipId})`}>
          <CityFeaturePaths printable={printable} />
        </g>
      </svg>
      {loading && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-white/40 border-t-[#0f766e]" aria-label={t("loading")} />
        </div>
      )}
      {error && (
        <div className="pointer-events-none absolute inset-x-1 bottom-1 rounded bg-red-500/90 px-1.5 py-0.5 text-center text-[8px] text-white">
          {error.slice(0, 40)}
        </div>
      )}
      {!hasData && !loading && !error && (
        <div className="absolute inset-0 flex items-center justify-center text-[10px] text-[#6a5d44]">
          {t("pickArea")}
        </div>
      )}
    </div>
  );
}
