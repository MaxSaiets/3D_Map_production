"use client";

import { useEffect, useMemo, useRef, useState } from "react";

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

async function fetchOSMForBounds(b: Bounds, abortSignal?: AbortSignal): Promise<CityData> {
  const bbox = `${b.south},${b.west},${b.north},${b.east}`;
  // Додаємо: area:highway=pedestrian (площі типу Майдан), place=square, landuse=pedestrian
  const q = `[out:json][timeout:12];(way["building"](${bbox});way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|service|pedestrian|footway|path|cycleway)$"](${bbox});way["highway"="pedestrian"]["area"="yes"](${bbox});way["place"~"^(square|plaza)$"](${bbox});way["natural"~"^(water|wood|grassland|scrub|heath)$"](${bbox});way["waterway"](${bbox});way["leisure"~"^(park|garden|pitch|playground|nature_reserve)$"](${bbox});way["landuse"~"^(grass|forest|recreation_ground|meadow|village_green|cemetery|allotments|orchard|pedestrian)$"](${bbox});way["man_made"="bridge"](${bbox});way["bridge"="yes"](${bbox});relation["natural"="water"](${bbox});relation["leisure"="park"](${bbox});relation["highway"="pedestrian"](${bbox});relation["landuse"="forest"](${bbox});node["amenity"="fountain"](${bbox});node["natural"="tree"](${bbox}););out geom;`;
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

      function classifyAndPush(tags: any, points: Pts) {
        if (!points || points.length < 2) return;
        const isPlaza =
          tags.place === "square" || tags.place === "plaza" ||
          tags.landuse === "pedestrian" ||
          (tags.highway === "pedestrian" && (tags.area === "yes" || isClosedRing(points)));
        const isBridge = tags.bridge === "yes" || tags.man_made === "bridge";
        if (tags.building) {
          buildings.push({ points, levels: Math.max(1, Math.min(40, Number(tags["building:levels"]) || 3)) });
        } else if (isPlaza && isClosedRing(points)) {
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
          const widths: Record<string, number> = {
            motorway: 14, trunk: 12, primary: 10, secondary: 8, tertiary: 7,
            residential: 5, unclassified: 5, service: 3.5, pedestrian: 4,
            footway: 2, path: 2, cycleway: 2.5,
          };
          const kind: RoadRec["kind"] =
            ["motorway", "trunk", "primary", "secondary"].includes(String(tags.highway)) ? "major"
            : ["residential", "tertiary", "unclassified"].includes(String(tags.highway)) ? "minor" : "service";
          roads.push({ points, widthM: widths[String(tags.highway)] || 4, kind });
        } else if (tags.natural === "water" || tags.waterway === "riverbank" || tags.waterway === "dock") {
          // ТІЛЬКИ полігони. waterway=river/stream/ditch — це лінії (way без area),
          // якщо їх запхати в polygon-fill, отримуємо рандомні сині трикутники.
          if (isClosedRing(points)) water.push(points);
        } else if (tags.leisure || tags.landuse || tags.natural) {
          const isGreen =
            ["park", "garden", "pitch", "playground", "nature_reserve", "golf_course"].includes(tags.leisure) ||
            ["grass", "forest", "recreation_ground", "meadow", "village_green", "cemetery", "allotments", "orchard"].includes(tags.landuse) ||
            ["wood", "grassland", "scrub", "heath"].includes(tags.natural);
          // Зелень теж тільки якщо замкнутий полігон. Лінії (наприклад tree_row)
          // не повинні створювати fill.
          if (isGreen && isClosedRing(points)) parks.push(points);
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
          // Multipolygon: для воді/парків обробляємо outer members, ALWAYS
          // (навіть якщо точки за bbox — це шматок великої водойми/лісу що
          // ПЕРЕКРИВАЄ нашу зону). Backend на стороні preview обрізає по
          // slot-clipPath, тож лишній рендер не страшний.
          for (const m of el.members) {
            if (m.type === "way" && m.geometry && m.geometry.length >= 2 && m.role !== "inner") {
              const points: Pts = m.geometry.map((g: any) => [g.lon, g.lat]);
              if (lineIntersectsBbox(points)) classifyAndPush(tags, points);
            }
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

/** Чистий 2D preview ділянки — рендериться тими ж SVG-координатами як map area.
 *  Показує ТІЛЬКИ те, що буде роздруковано: будівлі, дороги, воду, парки.
 *  Без OSM-міток, іконок, POI — чисто схематично, як у фінальному 3MF. */
export function LiveCity3D({
  bounds,
  design,
  cropRotationDeg = 0,
  cropPolygon = null,
}: {
  bounds: Bounds;
  design: DesignShape;
  /** Поворот рамки на карті — preview обертає контент на -кут щоб ділянка
   *  виглядала axis-aligned у map slot. */
  cropRotationDeg?: number;
  /** 4 кути обернутого rect [[lon, lat], ...]. Preview clipує OSM по цьому
   *  полігону — показує ТІЛЬКИ те що буде на готовому 3MF, нічого зайвого. */
  cropPolygon?: Array<[number, number]> | null;
}) {
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
    return {
      lonLatToMm: (lon: number, lat: number): [number, number] => {
        // Локальні метри відносно центру зони
        const dx = (lon - cLon) * mPerDegLng;
        const dy = (lat - cLat) * 111_320;
        // Інверсія повороту рамки: CW rotation на θ (= CCW на -θ)
        const rx = dx * cosT + dy * sinT;
        const ry = -dx * sinT + dy * cosT;
        // rx ∈ [-widthM/2, widthM/2] для точок усередині зони → слот mm
        const mx = design.mapXMm + design.mapWidthMm * (rx / widthM + 0.5);
        const my = design.mapYMm + design.mapHeightMm * (0.5 - ry / heightM);
        return [mx, my];
      },
      mmPerM,
    };
  }, [bounds, cropPolygon, cropRotationDeg, design.mapXMm, design.mapYMm, design.mapWidthMm, design.mapHeightMm]);

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

  // Розміри viewBox матчать map area (мм)
  const vb = `${design.mapXMm} ${design.mapYMm} ${design.mapWidthMm} ${design.mapHeightMm}`;

  // Slot-shape clip — щоб контент який виходить за межі слота не вилазив.
  const clipId = useMemo(() => `liveCityClip-${Math.random().toString(36).slice(2, 8)}`, []);

  return (
    <div className="relative h-full w-full overflow-hidden bg-[#e0d4b5]">
      <svg
        viewBox={vb}
        preserveAspectRatio="xMidYMid meet"
        style={{ width: "100%", height: "100%", display: "block" }}
      >
        <defs>
          <clipPath id={clipId}>
            <rect
              x={design.mapXMm}
              y={design.mapYMm}
              width={design.mapWidthMm}
              height={design.mapHeightMm}
            />
          </clipPath>
        </defs>
        {/* Базова бежева плита на весь слот */}
        <rect
          x={design.mapXMm}
          y={design.mapYMm}
          width={design.mapWidthMm}
          height={design.mapHeightMm}
          fill="#e0d4b5"
        />
        {/* Контент уже спроектований так що повернута зона = слот.
            Slot-clip обрізає випадковий overshoot за межами слота. */}
        <g clipPath={`url(#${clipId})`}>
          {printable.parks.map((pts, i) => (
            <path key={`p-${i}`} d={pointsToPath(pts)} fill="#88b06e" />
          ))}
          {printable.water.map((pts, i) => (
            <path key={`w-${i}`} d={pointsToPath(pts)} fill="#5a91c4" />
          ))}
          {/* Пішохідні площі — світло-лавандовий як на OSM-тайлах */}
          {printable.plazas.map((pts, i) => (
            <path key={`pl-${i}`} d={pointsToPath(pts)} fill="#d4cce0" />
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
          {/* Мости — поверх води/доріг, темно-сірі */}
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
          {/* Дерева — маленькі зелені крапки */}
          {printable.trees.map((t, i) => (
            <circle key={`t-${i}`} cx={t.center[0]} cy={t.center[1]} r={t.rMm} fill="#6b9c52" />
          ))}
          {/* Фонтани — сині кружки з білим обведенням */}
          {printable.fountains.map((f, i) => (
            <circle
              key={`f-${i}`}
              cx={f.center[0]}
              cy={f.center[1]}
              r={f.rMm}
              fill="#5a91c4"
              stroke="#ffffff"
              strokeWidth={0.3}
            />
          ))}
        </g>
      </svg>
      {loading && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div
            className="h-8 w-8 animate-spin rounded-full border-2 border-white/40 border-t-[#0f766e]"
            aria-label="Завантаження"
          />
        </div>
      )}
      {error && (
        <div className="pointer-events-none absolute inset-x-1 bottom-1 rounded bg-red-500/90 px-1.5 py-0.5 text-center text-[8px] text-white">
          {error.slice(0, 40)}
        </div>
      )}
      {!data && !loading && !error && (
        <div className="absolute inset-0 flex items-center justify-center text-[10px] text-[#6a5d44]">
          Оберіть ділянку
        </div>
      )}
    </div>
  );
}
