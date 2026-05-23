"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

type Bounds = { north: number; south: number; east: number; west: number };

type BuildingRec = { points: Array<[number, number]>; levels: number };
type RoadRec = { points: Array<[number, number]>; width: number };
type CityData = { buildings: BuildingRec[]; roads: RoadRec[] };

const OVERPASS_URL = "https://overpass-api.de/api/interpreter";

async function fetchOSMForBounds(b: Bounds, abortSignal?: AbortSignal): Promise<CityData> {
  const bbox = `${b.south},${b.west},${b.north},${b.east}`;
  const q = `[out:json][timeout:15];(way["building"](${bbox});way["highway"](${bbox}););out geom;`;
  const res = await fetch(OVERPASS_URL, {
    method: "POST",
    body: q,
    headers: { "Content-Type": "text/plain" },
    signal: abortSignal,
  });
  if (!res.ok) throw new Error("Overpass " + res.status);
  const data = await res.json();
  const buildings: BuildingRec[] = [];
  const roads: RoadRec[] = [];
  for (const el of data.elements || []) {
    if (!el.geometry || el.geometry.length < 2) continue;
    const points: Array<[number, number]> = el.geometry.map((g: any) => [g.lon, g.lat]);
    if (el.tags?.building) {
      const lvl = Number(el.tags["building:levels"]) || 3;
      buildings.push({ points, levels: Math.max(1, Math.min(40, lvl)) });
    } else if (el.tags?.highway) {
      const widths: Record<string, number> = {
        motorway: 12, trunk: 10, primary: 8, secondary: 7, tertiary: 6,
        residential: 5, service: 3.5, footway: 1.5, path: 1, cycleway: 2, pedestrian: 4,
      };
      const w = widths[String(el.tags.highway)] || 4;
      roads.push({ points, width: w });
    }
  }
  return { buildings, roads };
}

function lonLatToLocal(lon: number, lat: number, b: Bounds): [number, number] {
  // Normalize lon/lat into [-0.5, 0.5] range, aspect-aware
  const cLat = (b.north + b.south) / 2;
  const mPerDegLng = 111_320 * Math.max(Math.cos((cLat * Math.PI) / 180), 0.18);
  const wM = (b.east - b.west) * mPerDegLng;
  const hM = (b.north - b.south) * 111_320;
  const xM = (lon - b.west) * mPerDegLng;
  const yM = (lat - b.south) * 111_320;
  // Y from south→up. Normalize.
  return [xM / wM - 0.5, yM / hM - 0.5];
}

function ExtrudedPolygon({ pts, height, color }: { pts: Array<[number, number]>; height: number; color: string }) {
  const geom = useMemo(() => {
    if (pts.length < 3) return null;
    const shape = new THREE.Shape();
    shape.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) shape.lineTo(pts[i][0], pts[i][1]);
    return new THREE.ExtrudeGeometry(shape, { depth: height, bevelEnabled: false });
  }, [pts, height]);
  if (!geom) return null;
  return (
    <mesh geometry={geom} rotation={[-Math.PI / 2, 0, 0]} castShadow receiveShadow>
      <meshStandardMaterial color={color} roughness={0.7} metalness={0.05} />
    </mesh>
  );
}

function RoadStrip({ points, width, color }: { points: Array<[number, number]>; width: number; color: string }) {
  // Buffer the line into a thick polygon manually via offset normals
  const polyPts = useMemo(() => {
    if (points.length < 2) return null;
    const out: Array<[number, number]> = [];
    const halfW = width / 2;
    // Forward pass
    for (let i = 0; i < points.length; i++) {
      const prev = points[Math.max(0, i - 1)];
      const next = points[Math.min(points.length - 1, i + 1)];
      const dx = next[0] - prev[0];
      const dy = next[1] - prev[1];
      const len = Math.hypot(dx, dy) || 1;
      const nx = -dy / len;
      const ny = dx / len;
      out.push([points[i][0] + nx * halfW, points[i][1] + ny * halfW]);
    }
    // Backward pass
    for (let i = points.length - 1; i >= 0; i--) {
      const prev = points[Math.max(0, i - 1)];
      const next = points[Math.min(points.length - 1, i + 1)];
      const dx = next[0] - prev[0];
      const dy = next[1] - prev[1];
      const len = Math.hypot(dx, dy) || 1;
      const nx = -dy / len;
      const ny = dx / len;
      out.push([points[i][0] - nx * halfW, points[i][1] - ny * halfW]);
    }
    return out;
  }, [points, width]);
  if (!polyPts) return null;
  return <ExtrudedPolygon pts={polyPts} height={0.4} color={color} />;
}

/** Real-time 3D preview of the city in the user's selected bbox.
 *  Fetches building/road footprints from Overpass API and renders as
 *  Three.js extrusions — same visual style as the final 3MF print. */
export function LiveCity3D({ bounds }: { bounds: Bounds }) {
  const [data, setData] = useState<CityData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const reqRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!bounds) return;
    // Cancel pending request
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
        if (e.name !== "AbortError") setError(e.message || "Overpass error");
      } finally {
        if (!ctrl.signal.aborted) setLoading(false);
      }
    }, 600);  // debounce: wait 600ms after user stops dragging
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [bounds.north, bounds.south, bounds.east, bounds.west]);

  // Compute aspect for scaling
  const cLat = (bounds.north + bounds.south) / 2;
  const mPerDegLng = 111_320 * Math.max(Math.cos((cLat * Math.PI) / 180), 0.18);
  const wM = Math.max(1, (bounds.east - bounds.west) * mPerDegLng);
  const hM = Math.max(1, (bounds.north - bounds.south) * 111_320);
  const sceneWidth = 1; // normalized
  const sceneDepth = hM / wM;
  const buildingHScale = 0.0025; // 1 floor ~= 3m, scale to model units

  const localBuildings = useMemo(() => {
    if (!data) return [];
    return data.buildings.map((b) => ({
      pts: b.points.map(([lon, lat]) => lonLatToLocal(lon, lat, bounds)),
      h: Math.max(0.005, b.levels * buildingHScale),
    }));
  }, [data, bounds]);

  const localRoads = useMemo(() => {
    if (!data) return [];
    return data.roads.map((r) => ({
      pts: r.points.map(([lon, lat]) => lonLatToLocal(lon, lat, bounds)),
      w: r.width / wM,
    }));
  }, [data, bounds, wM]);

  return (
    <div className="relative h-full w-full overflow-hidden rounded-[14px] bg-[#e8e1cc]">
      {(loading || !data) && (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center bg-black/35 backdrop-blur-sm">
          <div className="rounded-full bg-white/90 px-3 py-1.5 text-xs font-semibold text-[#1f2420]">
            {loading ? "Завантаження 3D…" : "Готую сцену…"}
          </div>
        </div>
      )}
      {error && (
        <div className="pointer-events-none absolute inset-x-0 bottom-2 z-10 mx-2 rounded-md bg-red-500/90 px-2 py-1 text-center text-[10px] text-white">
          {error}
        </div>
      )}
      <Canvas
        shadows
        camera={{ position: [0.6, 0.7, 0.6], fov: 35, near: 0.01, far: 10 }}
        dpr={[1, 2]}
        style={{ background: "linear-gradient(to bottom, #c4d8e8 0%, #e8e1cc 100%)" }}
      >
        <ambientLight intensity={0.55} />
        <directionalLight position={[1, 2, 0.5]} intensity={1.2} castShadow shadow-mapSize={[1024, 1024]}>
          <orthographicCamera attach="shadow-camera" args={[-1, 1, 1, -1, 0.1, 5]} />
        </directionalLight>
        {/* Base plate */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
          <planeGeometry args={[sceneWidth, sceneDepth]} />
          <meshStandardMaterial color="#d4cdb6" roughness={0.95} />
        </mesh>
        {/* Roads — slightly raised, dark */}
        {localRoads.map((r, i) => (
          <RoadStrip key={`r-${i}`} points={r.pts} width={r.w} color="#3a3a3a" />
        ))}
        {/* Buildings — extruded */}
        {localBuildings.map((b, i) => (
          <ExtrudedPolygon key={`b-${i}`} pts={b.pts} height={b.h} color="#cfc7b3" />
        ))}
        <OrbitControls
          enablePan={false}
          enableZoom={false}
          maxPolarAngle={Math.PI / 2.5}
          minPolarAngle={Math.PI / 6}
          autoRotate
          autoRotateSpeed={0.5}
        />
      </Canvas>
    </div>
  );
}
