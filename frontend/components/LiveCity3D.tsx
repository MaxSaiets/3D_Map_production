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
type CityData = { buildings: BuildingRec[]; roads: RoadRec[]; water: Pts[]; parks: Pts[] };

const OVERPASS_URLS = [
  "https://overpass-api.de/api/interpreter",
  "https://overpass.kumi.systems/api/interpreter",
];

// Мінімальна друкована деталь — все що менше, не друкується якісно у 0.4mm соплі
const MIN_PRINT_MM = 0.6;

async function fetchOSMForBounds(b: Bounds, abortSignal?: AbortSignal): Promise<CityData> {
  const bbox = `${b.south},${b.west},${b.north},${b.east}`;
  const q = `[out:json][timeout:15];(way["building"](${bbox});way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|service|pedestrian)$"](${bbox});way["natural"~"^(water|wood|grassland|scrub|heath)$"](${bbox});way["waterway"](${bbox});way["leisure"~"^(park|garden|pitch|playground|nature_reserve)$"](${bbox});way["landuse"~"^(grass|forest|recreation_ground|meadow|village_green|cemetery|allotments|orchard|farmland)$"](${bbox});relation["natural"="water"](${bbox});relation["leisure"="park"](${bbox}););out geom;`;
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

      function classifyAndPush(tags: any, points: Pts) {
        if (!points || points.length < 2) return;
        if (tags.building) {
          buildings.push({ points, levels: Math.max(1, Math.min(40, Number(tags["building:levels"]) || 3)) });
        } else if (tags.highway) {
          const widths: Record<string, number> = {
            motorway: 14, trunk: 12, primary: 10, secondary: 8, tertiary: 7,
            residential: 5, unclassified: 5, service: 3.5, pedestrian: 4,
          };
          const kind: RoadRec["kind"] =
            ["motorway", "trunk", "primary", "secondary"].includes(String(tags.highway)) ? "major"
            : ["residential", "tertiary", "unclassified"].includes(String(tags.highway)) ? "minor" : "service";
          roads.push({ points, widthM: widths[String(tags.highway)] || 4, kind });
        } else if (tags.natural === "water" || tags.waterway) {
          water.push(points);
        } else if (tags.leisure || tags.landuse || tags.natural) {
          const isGreen =
            ["park", "garden", "pitch", "playground", "nature_reserve"].includes(tags.leisure) ||
            ["grass", "forest", "recreation_ground", "meadow", "village_green", "cemetery", "allotments", "orchard", "farmland"].includes(tags.landuse) ||
            ["wood", "grassland", "scrub", "heath"].includes(tags.natural);
          if (isGreen) parks.push(points);
        }
      }

      for (const el of data.elements || []) {
        const tags = el.tags || {};
        if (el.type === "way" && el.geometry) {
          const points: Pts = el.geometry.map((g: any) => [g.lon, g.lat]);
          classifyAndPush(tags, points);
        } else if (el.type === "relation" && el.members) {
          // Multipolygon (наприклад, Дніпро): обробляємо кожен member як окремий полігон.
          // Це КРИТИЧНО для великих водойм/парків — вони майже завжди relations в OSM.
          for (const m of el.members) {
            if (m.type === "way" && m.geometry && m.geometry.length >= 2) {
              const points: Pts = m.geometry.map((g: any) => [g.lon, g.lat]);
              // role "outer" — основний контур, "inner" — отвір. Беремо outer.
              if (m.role !== "inner") {
                classifyAndPush(tags, points);
              }
            }
          }
        }
      }
      return { buildings, roads, water, parks };
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
}: {
  bounds: Bounds;
  design: DesignShape;
  /** Поворот рамки на карті — для preview повертаємо OSM-дані на -кут,
   *  щоб користувач бачив свою rotated-зону в "розпрямленому" вигляді
   *  саме так як буде на брелку. */
  cropRotationDeg?: number;
}) {
  const [data, setData] = useState<CityData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const reqRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!bounds) return;
    reqRef.current?.abort();
    const ctrl = new AbortController();
    reqRef.current = ctrl;
    const timer = setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const d = await fetchOSMForBounds(bounds, ctrl.signal);
        if (!ctrl.signal.aborted) setData(d);
      } catch (e: any) {
        if (e.name !== "AbortError") setError(e.message || "Overpass");
      } finally {
        if (!ctrl.signal.aborted) setLoading(false);
      }
    }, 400);
    return () => { clearTimeout(timer); ctrl.abort(); };
  }, [bounds.north, bounds.south, bounds.east, bounds.west]);

  // lat/lon → координати всередині map area (у тих же мм як SVG-дизайнер).
  // Frame (рамка SVG) лишається прямокутним та axis-aligned, але ВМІСТ
  // повертається — щоб коли користувач крутить рамку на мапі, preview
  // показував іншу геометричну ділянку (rotated rect intersects different
  // streets/buildings). Це matches що backend зробить при генерації моделі.
  const project = useMemo(() => {
    const cLat = (bounds.north + bounds.south) / 2;
    const mPerDegLng = 111_320 * Math.max(Math.cos((cLat * Math.PI) / 180), 0.18);
    const wM = (bounds.east - bounds.west) * mPerDegLng || 1;
    const hM = (bounds.north - bounds.south) * 111_320 || 1;
    const mmPerM = Math.min(design.mapWidthMm / wM, design.mapHeightMm / hM);
    const fitWmm = wM * mmPerM;
    const fitHmm = hM * mmPerM;
    const ox = design.mapXMm + (design.mapWidthMm - fitWmm) / 2;
    const oy = design.mapYMm + (design.mapHeightMm - fitHmm) / 2;
    // -cropRotationDeg: коли user-rect повертається CW на мапі, ми повертаємо
    // ВМІСТ CCW так, щоб user-rect-aligned content виглядав axis-aligned
    // у map slot брелка.
    const angleRad = (-Number(cropRotationDeg || 0) * Math.PI) / 180;
    const cosA = Math.cos(angleRad);
    const sinA = Math.sin(angleRad);
    return {
      lonLatToMm: (lon: number, lat: number): [number, number] => {
        const u = (lon - bounds.west) * mPerDegLng / wM;
        const v = 1 - (lat - bounds.south) * 111_320 / hM;
        // Поворот навколо центру bbox (0.5, 0.5)
        const du = u - 0.5, dv = v - 0.5;
        const u2 = 0.5 + du * cosA - dv * sinA;
        const v2 = 0.5 + du * sinA + dv * cosA;
        return [ox + u2 * fitWmm, oy + v2 * fitHmm];
      },
      mmPerM,
    };
  }, [bounds, design.mapXMm, design.mapYMm, design.mapWidthMm, design.mapHeightMm, cropRotationDeg]);

  // Фільтр + конвертація — тільки те що буде друкуватися
  const printable = useMemo(() => {
    if (!data) return { buildings: [], roads: [], water: [], parks: [] };
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
      .filter((pts) => pts.length >= 3 && polygonArea(pts) >= MIN_PRINT_MM * MIN_PRINT_MM * 4);
    const parks = data.parks
      .map((pts) => pts.map(([lon, lat]) => lonLatToMm(lon, lat)))
      .filter((pts) => pts.length >= 3 && polygonArea(pts) >= MIN_PRINT_MM * MIN_PRINT_MM * 4);
    return { buildings, roads, water, parks };
  }, [data, project]);

  // Розміри viewBox матчать map area (мм)
  const vb = `${design.mapXMm} ${design.mapYMm} ${design.mapWidthMm} ${design.mapHeightMm}`;

  return (
    <div className="relative h-full w-full overflow-hidden bg-[#e0d4b5]">
      <svg
        viewBox={vb}
        preserveAspectRatio="xMidYMid meet"
        style={{ width: "100%", height: "100%", display: "block" }}
      >
        {/* Базова бежева плита */}
        <rect
          x={design.mapXMm}
          y={design.mapYMm}
          width={design.mapWidthMm}
          height={design.mapHeightMm}
          fill="#e0d4b5"
        />
        {/* Парки (зелені) */}
        {printable.parks.map((pts, i) => (
          <path key={`p-${i}`} d={pointsToPath(pts)} fill="#88b06e" />
        ))}
        {/* Вода (синя) */}
        {printable.water.map((pts, i) => (
          <path key={`w-${i}`} d={pointsToPath(pts)} fill="#5a91c4" />
        ))}
        {/* Дороги (товсті лінії за класом) */}
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
        {/* Будівлі (бежеві блоки) */}
        {printable.buildings.map((pts, i) => (
          <path key={`b-${i}`} d={pointsToPath(pts)} fill="#cfc1a3" stroke="#a89a7d" strokeWidth={0.15} />
        ))}
      </svg>
      {loading && (
        <div className="pointer-events-none absolute right-1 top-1 rounded-full bg-emerald-500/90 px-2 py-0.5 text-[9px] font-semibold text-white">
          Завантаження…
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
