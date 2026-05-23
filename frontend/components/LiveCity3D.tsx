"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

type Bounds = { north: number; south: number; east: number; west: number };

type BuildingRec = { points: Array<[number, number]>; levels: number };
type RoadRec = { points: Array<[number, number]>; width: number; kind: "major" | "minor" | "service" };
type WaterRec = { points: Array<[number, number]> };
type ParkRec = { points: Array<[number, number]> };
type CityData = { buildings: BuildingRec[]; roads: RoadRec[]; water: WaterRec[]; parks: ParkRec[] };

const OVERPASS_URLS = [
  "https://overpass-api.de/api/interpreter",
  "https://overpass.kumi.systems/api/interpreter",
];

async function fetchOSMForBounds(b: Bounds, abortSignal?: AbortSignal): Promise<CityData> {
  const bbox = `${b.south},${b.west},${b.north},${b.east}`;
  const q = `[out:json][timeout:15];(way["building"](${bbox});way["highway"](${bbox});way["natural"="water"](${bbox});way["waterway"](${bbox});relation["natural"="water"](${bbox});way["leisure"="park"](${bbox});way["landuse"="grass"](${bbox}););out geom;`;
  let lastErr: any = null;
  for (const url of OVERPASS_URLS) {
    try {
      const res = await fetch(url, {
        method: "POST",
        body: q,
        headers: { "Content-Type": "text/plain" },
        signal: abortSignal,
      });
      if (!res.ok) {
        lastErr = new Error("Overpass " + res.status);
        continue;
      }
      const data = await res.json();
      const buildings: BuildingRec[] = [];
      const roads: RoadRec[] = [];
      const water: WaterRec[] = [];
      const parks: ParkRec[] = [];
      for (const el of data.elements || []) {
        if (!el.geometry || el.geometry.length < 2) continue;
        const points: Array<[number, number]> = el.geometry.map((g: any) => [g.lon, g.lat]);
        const tags = el.tags || {};
        if (tags.building) {
          const lvl = Number(tags["building:levels"]) || 3;
          buildings.push({ points, levels: Math.max(1, Math.min(40, lvl)) });
        } else if (tags.highway) {
          const widths: Record<string, number> = {
            motorway: 14, trunk: 12, primary: 10, secondary: 8, tertiary: 7,
            residential: 5, unclassified: 5, service: 3.5, footway: 1.6, path: 1.2, cycleway: 2, pedestrian: 4,
          };
          const w = widths[String(tags.highway)] || 4;
          const kind: RoadRec["kind"] =
            ["motorway", "trunk", "primary", "secondary"].includes(String(tags.highway)) ? "major"
            : ["residential", "tertiary", "unclassified"].includes(String(tags.highway)) ? "minor"
            : "service";
          roads.push({ points, width: w, kind });
        } else if (tags.natural === "water" || tags.waterway) {
          water.push({ points });
        } else if (tags.leisure === "park" || tags.landuse === "grass") {
          parks.push({ points });
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

function lonLatToLocal(lon: number, lat: number, b: Bounds, sceneSize: number = 2): [number, number] {
  // Normalize lon/lat into a square scene [-sceneSize/2..sceneSize/2]
  const cLat = (b.north + b.south) / 2;
  const mPerDegLng = 111_320 * Math.max(Math.cos((cLat * Math.PI) / 180), 0.18);
  const wM = (b.east - b.west) * mPerDegLng;
  const hM = (b.north - b.south) * 111_320;
  const maxDim = Math.max(wM, hM);
  // Keep aspect ratio: bigger of width/height fills sceneSize, smaller dim shorter
  const xM = (lon - b.west) * mPerDegLng;
  const yM = (lat - b.south) * 111_320;
  return [
    (xM / maxDim - (wM / maxDim) / 2) * sceneSize,
    (yM / maxDim - (hM / maxDim) / 2) * sceneSize,
  ];
}

function makeShape(pts: Array<[number, number]>): THREE.Shape | null {
  if (pts.length < 3) return null;
  const shape = new THREE.Shape();
  shape.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) shape.lineTo(pts[i][0], pts[i][1]);
  return shape;
}

function ExtrudedPolygon({ pts, height, color, zOffset = 0, roughness = 0.7 }: {
  pts: Array<[number, number]>;
  height: number;
  color: string;
  zOffset?: number;
  roughness?: number;
}) {
  const geom = useMemo(() => {
    const shape = makeShape(pts);
    if (!shape) return null;
    return new THREE.ExtrudeGeometry(shape, { depth: height, bevelEnabled: false });
  }, [pts, height]);
  if (!geom) return null;
  return (
    <mesh geometry={geom} rotation={[-Math.PI / 2, 0, 0]} position={[0, zOffset, 0]} castShadow receiveShadow>
      <meshStandardMaterial color={color} roughness={roughness} metalness={0.05} />
    </mesh>
  );
}

function bufferLineToPoly(points: Array<[number, number]>, width: number): Array<[number, number]> | null {
  if (points.length < 2) return null;
  const out: Array<[number, number]> = [];
  const halfW = width / 2;
  for (let i = 0; i < points.length; i++) {
    const prev = points[Math.max(0, i - 1)];
    const next = points[Math.min(points.length - 1, i + 1)];
    const dx = next[0] - prev[0];
    const dy = next[1] - prev[1];
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len, ny = dx / len;
    out.push([points[i][0] + nx * halfW, points[i][1] + ny * halfW]);
  }
  for (let i = points.length - 1; i >= 0; i--) {
    const prev = points[Math.max(0, i - 1)];
    const next = points[Math.min(points.length - 1, i + 1)];
    const dx = next[0] - prev[0];
    const dy = next[1] - prev[1];
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len, ny = dx / len;
    out.push([points[i][0] - nx * halfW, points[i][1] - ny * halfW]);
  }
  return out;
}

/** Real-time 3D printable preview — looks like the final keychain that would
 *  come out of the printer: base plate + extruded buildings + raised roads,
 *  same visual style as Bambu Studio slicer view. */
export function LiveCity3D({ bounds }: { bounds: Bounds }) {
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
        if (e.name !== "AbortError") setError(e.message || "Overpass недоступний");
      } finally {
        if (!ctrl.signal.aborted) setLoading(false);
      }
    }, 600);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [bounds.north, bounds.south, bounds.east, bounds.west]);

  // Compute aspect for scene
  const cLat = (bounds.north + bounds.south) / 2;
  const mPerDegLng = 111_320 * Math.max(Math.cos((cLat * Math.PI) / 180), 0.18);
  const wM = Math.max(1, (bounds.east - bounds.west) * mPerDegLng);
  const hM = Math.max(1, (bounds.north - bounds.south) * 111_320);
  const maxDim = Math.max(wM, hM);
  const sceneW = (wM / maxDim) * 2;
  const sceneH = (hM / maxDim) * 2;

  const localBuildings = useMemo(() => {
    if (!data) return [];
    return data.buildings.map((b) => ({
      pts: b.points.map(([lon, lat]) => lonLatToLocal(lon, lat, bounds)),
      // Visible heights at scene scale ~2: 1 floor ≈ 0.04
      h: Math.max(0.04, Math.min(0.6, b.levels * 0.04)),
    }));
  }, [data, bounds]);

  const localRoads = useMemo(() => {
    if (!data) return [];
    // Scale road width to scene units (2 = full scene). Min visible 0.012.
    return data.roads.map((r) => {
      const localPts = r.points.map(([lon, lat]) => lonLatToLocal(lon, lat, bounds));
      const w = Math.max(0.012, (r.width / maxDim) * 2 * 1.5); // slightly thicker for visibility
      return { polyPts: bufferLineToPoly(localPts, w), kind: r.kind };
    }).filter((r) => r.polyPts !== null) as Array<{ polyPts: Array<[number, number]>; kind: RoadRec["kind"] }>;
  }, [data, bounds, maxDim]);

  const localWater = useMemo(() => {
    if (!data) return [];
    return data.water.map((w) => w.points.map(([lon, lat]) => lonLatToLocal(lon, lat, bounds)));
  }, [data, bounds]);

  const localParks = useMemo(() => {
    if (!data) return [];
    return data.parks.map((p) => p.points.map(([lon, lat]) => lonLatToLocal(lon, lat, bounds)));
  }, [data, bounds]);

  return (
    <div className="relative h-full w-full overflow-hidden rounded-[10px] bg-[#0f172a]">
      {loading && (
        <div className="pointer-events-none absolute right-1 top-1 z-10 rounded-full bg-emerald-500/90 px-2 py-0.5 text-[9px] font-semibold text-white">
          Завантаження…
        </div>
      )}
      {error && (
        <div className="pointer-events-none absolute inset-x-1 bottom-1 z-10 rounded bg-red-500/90 px-1.5 py-0.5 text-center text-[8px] text-white">
          {error.slice(0, 40)}
        </div>
      )}
      {!data && !loading && !error && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/40 text-[10px] text-white/70">
          Оберіть ділянку
        </div>
      )}
      <Canvas
        shadows
        camera={{ position: [1.4, 1.7, 1.4], fov: 32, near: 0.01, far: 20 }}
        dpr={[1, 2]}
        gl={{ antialias: true }}
        style={{ background: "linear-gradient(to bottom, #a8d0e6 0%, #f5ecd0 100%)" }}
      >
        <ambientLight intensity={0.7} />
        <directionalLight position={[2, 3, 1]} intensity={1.4} castShadow shadow-mapSize={[2048, 2048]}>
          <orthographicCamera attach="shadow-camera" args={[-2, 2, 2, -2, 0.1, 8]} />
        </directionalLight>
        <directionalLight position={[-1.5, 1, -1]} intensity={0.4} />

        {/* Base plate — like keychain body */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.005, 0]} receiveShadow>
          <planeGeometry args={[sceneW * 1.08, sceneH * 1.08]} />
          <meshStandardMaterial color="#c8a96a" roughness={0.85} />
        </mesh>
        {/* Rim border */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]} receiveShadow>
          <ringGeometry args={[Math.min(sceneW, sceneH) * 0.42, Math.min(sceneW, sceneH) * 0.5, 64]} />
          <meshStandardMaterial color="#7a6440" roughness={0.7} />
        </mesh>

        {/* Parks (green) */}
        {localParks.map((pts, i) => (
          <ExtrudedPolygon key={`p-${i}`} pts={pts} height={0.012} color="#88b06e" zOffset={0.002} />
        ))}
        {/* Water (blue) */}
        {localWater.map((pts, i) => (
          <ExtrudedPolygon key={`w-${i}`} pts={pts} height={0.008} color="#5a91c4" zOffset={0.003} roughness={0.35} />
        ))}
        {/* Roads — major roads thicker/darker */}
        {localRoads.map((r, i) => (
          <ExtrudedPolygon
            key={`r-${i}`}
            pts={r.polyPts}
            height={r.kind === "major" ? 0.022 : 0.018}
            color={r.kind === "major" ? "#1a1a1a" : r.kind === "minor" ? "#3a3a3a" : "#6a6a6a"}
            zOffset={0.005}
          />
        ))}
        {/* Buildings — extruded prisms */}
        {localBuildings.map((b, i) => (
          <ExtrudedPolygon key={`b-${i}`} pts={b.pts} height={b.h} color="#e8dfca" zOffset={0.005} roughness={0.8} />
        ))}

        <OrbitControls
          enablePan={false}
          enableZoom={false}
          maxPolarAngle={Math.PI / 2.2}
          minPolarAngle={Math.PI / 5}
          autoRotate
          autoRotateSpeed={0.6}
        />
      </Canvas>
    </div>
  );
}
