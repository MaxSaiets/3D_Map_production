"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

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

type BuildingRec = { points: Array<[number, number]>; levels: number };
type RoadRec = { points: Array<[number, number]>; width: number; kind: "major" | "minor" | "service" };
type CityData = { buildings: BuildingRec[]; roads: RoadRec[]; water: Array<{ points: Array<[number, number]> }>; parks: Array<{ points: Array<[number, number]> }> };

const OVERPASS_URLS = [
  "https://overpass-api.de/api/interpreter",
  "https://overpass.kumi.systems/api/interpreter",
];

async function fetchOSMForBounds(b: Bounds, abortSignal?: AbortSignal): Promise<CityData> {
  const bbox = `${b.south},${b.west},${b.north},${b.east}`;
  const q = `[out:json][timeout:15];(way["building"](${bbox});way["highway"](${bbox});way["natural"="water"](${bbox});way["waterway"](${bbox});way["leisure"="park"](${bbox}););out geom;`;
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
      const water: CityData["water"] = [];
      const parks: CityData["parks"] = [];
      for (const el of data.elements || []) {
        if (!el.geometry || el.geometry.length < 2) continue;
        const points: Array<[number, number]> = el.geometry.map((g: any) => [g.lon, g.lat]);
        const tags = el.tags || {};
        if (tags.building) {
          buildings.push({ points, levels: Math.max(1, Math.min(40, Number(tags["building:levels"]) || 3)) });
        } else if (tags.highway) {
          const w: Record<string, number> = {
            motorway: 14, trunk: 12, primary: 10, secondary: 8, tertiary: 7,
            residential: 5, unclassified: 5, service: 3.5, footway: 1.5, path: 1.2, pedestrian: 4,
          };
          const kind: RoadRec["kind"] =
            ["motorway", "trunk", "primary", "secondary"].includes(String(tags.highway)) ? "major"
            : ["residential", "tertiary", "unclassified"].includes(String(tags.highway)) ? "minor" : "service";
          roads.push({ points, width: w[String(tags.highway)] || 4, kind });
        } else if (tags.natural === "water" || tags.waterway) {
          water.push({ points });
        } else if (tags.leisure === "park") {
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

function makeShape(pts: Array<[number, number]>): THREE.Shape | null {
  if (pts.length < 3) return null;
  const shape = new THREE.Shape();
  shape.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) shape.lineTo(pts[i][0], pts[i][1]);
  shape.closePath();
  return shape;
}

function bufferLineToPoly(points: Array<[number, number]>, width: number): Array<[number, number]> | null {
  if (points.length < 2) return null;
  const out: Array<[number, number]> = [];
  const halfW = width / 2;
  for (let i = 0; i < points.length; i++) {
    const prev = points[Math.max(0, i - 1)];
    const next = points[Math.min(points.length - 1, i + 1)];
    const dx = next[0] - prev[0], dy = next[1] - prev[1];
    const len = Math.hypot(dx, dy) || 1;
    out.push([points[i][0] + (-dy / len) * halfW, points[i][1] + (dx / len) * halfW]);
  }
  for (let i = points.length - 1; i >= 0; i--) {
    const prev = points[Math.max(0, i - 1)];
    const next = points[Math.min(points.length - 1, i + 1)];
    const dx = next[0] - prev[0], dy = next[1] - prev[1];
    const len = Math.hypot(dx, dy) || 1;
    out.push([points[i][0] - (-dy / len) * halfW, points[i][1] - (dx / len) * halfW]);
  }
  return out;
}

/** Keychain body shape — rounded rect with loop hole on top */
function bodyShape(d: DesignShape): THREE.Shape {
  const w = d.bodyWidthMm;
  const h = d.bodyHeightMm;
  const r = Math.min(d.cornerRadiusMm, Math.min(w, h) / 2);
  const shape = new THREE.Shape();
  shape.moveTo(r, 0);
  shape.lineTo(w - r, 0);
  shape.quadraticCurveTo(w, 0, w, r);
  shape.lineTo(w, h - r);
  shape.quadraticCurveTo(w, h, w - r, h);
  shape.lineTo(r, h);
  shape.quadraticCurveTo(0, h, 0, h - r);
  shape.lineTo(0, r);
  shape.quadraticCurveTo(0, 0, r, 0);
  return shape;
}

/** Loop ring on top of body */
function loopShape(d: DesignShape): { ring: THREE.Shape; pos: [number, number] } | null {
  if (d.loopOuterMm <= 0 || d.baseShape === "token") return null;
  const ring = new THREE.Shape();
  ring.absarc(d.loopXMm, d.loopYMm, d.loopOuterMm, 0, Math.PI * 2, false);
  const hole = new THREE.Path();
  hole.absarc(d.loopXMm, d.loopYMm, d.loopInnerMm, 0, Math.PI * 2, true);
  ring.holes.push(hole);
  return { ring, pos: [d.loopXMm, d.loopYMm] };
}

export function LiveCity3D({ bounds, design }: { bounds: Bounds; design: DesignShape }) {
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
    }, 600);
    return () => { clearTimeout(timer); ctrl.abort(); };
  }, [bounds.north, bounds.south, bounds.east, bounds.west]);

  // Lat/lon → coords inside the map area (in mm of the keychain layout)
  const lonLatToMap = useMemo(() => {
    const cLat = (bounds.north + bounds.south) / 2;
    const mPerDegLng = 111_320 * Math.max(Math.cos((cLat * Math.PI) / 180), 0.18);
    const wM = (bounds.east - bounds.west) * mPerDegLng || 1;
    const hM = (bounds.north - bounds.south) * 111_320 || 1;
    return (lon: number, lat: number): [number, number] => {
      const u = (lon - bounds.west) * mPerDegLng / wM;       // 0..1
      const v = (lat - bounds.south) * 111_320 / hM;          // 0..1
      return [design.mapXMm + u * design.mapWidthMm, design.mapYMm + v * design.mapHeightMm];
    };
  }, [bounds.north, bounds.south, bounds.east, bounds.west, design.mapXMm, design.mapYMm, design.mapWidthMm, design.mapHeightMm]);

  const buildingMeshes = useMemo(() => {
    if (!data) return [];
    return data.buildings.map((b) => ({
      shape: makeShape(b.points.map(([lon, lat]) => lonLatToMap(lon, lat))),
      h: Math.max(0.6, Math.min(6, b.levels * 0.6)),  // mm in keychain space
    })).filter((b) => b.shape !== null) as Array<{ shape: THREE.Shape; h: number }>;
  }, [data, lonLatToMap]);

  const roadMeshes = useMemo(() => {
    if (!data) return [];
    return data.roads.map((r) => {
      const pts = r.points.map(([lon, lat]) => lonLatToMap(lon, lat));
      // road width in mm — depends on real m width vs map slot scale
      const cLat = (bounds.north + bounds.south) / 2;
      const mPerDegLng = 111_320 * Math.max(Math.cos((cLat * Math.PI) / 180), 0.18);
      const wM = (bounds.east - bounds.west) * mPerDegLng || 1;
      const mmPerM = design.mapWidthMm / wM;
      const wMm = Math.max(0.4, r.width * mmPerM);
      const poly = bufferLineToPoly(pts, wMm);
      return { shape: poly ? makeShape(poly) : null, kind: r.kind };
    }).filter((r) => r.shape !== null) as Array<{ shape: THREE.Shape; kind: RoadRec["kind"] }>;
  }, [data, lonLatToMap, design.mapWidthMm, bounds]);

  const waterMeshes = useMemo(() => {
    if (!data) return [];
    return data.water.map((w) => makeShape(w.points.map(([lon, lat]) => lonLatToMap(lon, lat)))).filter(Boolean) as THREE.Shape[];
  }, [data, lonLatToMap]);

  const parkMeshes = useMemo(() => {
    if (!data) return [];
    return data.parks.map((p) => makeShape(p.points.map(([lon, lat]) => lonLatToMap(lon, lat)))).filter(Boolean) as THREE.Shape[];
  }, [data, lonLatToMap]);

  const body = useMemo(() => bodyShape(design), [design]);
  const loop = useMemo(() => loopShape(design), [design]);

  // Scene: center the keychain at origin so OrbitControls work nicely
  const offsetX = -design.bodyWidthMm / 2;
  const offsetY = -design.bodyHeightMm / 2;

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
        camera={{ position: [design.bodyWidthMm * 0.7, design.bodyHeightMm * 0.9, design.bodyWidthMm * 0.7], fov: 32, near: 0.1, far: 500 }}
        dpr={[1, 2]}
        gl={{ antialias: true }}
        style={{ background: "linear-gradient(to bottom, #1e293b 0%, #475569 60%, #f5ecd0 100%)" }}
      >
        <ambientLight intensity={0.65} />
        <directionalLight position={[design.bodyWidthMm * 1.5, design.bodyHeightMm * 2, design.bodyWidthMm]} intensity={1.4} castShadow shadow-mapSize={[2048, 2048]}>
          <orthographicCamera attach="shadow-camera" args={[-design.bodyWidthMm, design.bodyWidthMm, design.bodyHeightMm, -design.bodyHeightMm, 0.1, 200]} />
        </directionalLight>
        <directionalLight position={[-design.bodyWidthMm, design.bodyHeightMm, -design.bodyWidthMm]} intensity={0.35} />

        <group position={[offsetX, 0, offsetY]} rotation={[-Math.PI / 2, 0, 0]}>
          {/* ── KEYCHAIN BODY ─────────────────────────────── */}
          <mesh castShadow receiveShadow>
            <extrudeGeometry args={[body, { depth: 2.0, bevelEnabled: false }]} />
            <meshStandardMaterial color="#c8a96a" roughness={0.75} metalness={0.05} />
          </mesh>

          {/* ── LOOP ring on top ──────────────────────────── */}
          {loop && (
            <mesh castShadow receiveShadow position={[0, 0, 0]}>
              <extrudeGeometry args={[loop.ring, { depth: 2.0, bevelEnabled: false }]} />
              <meshStandardMaterial color="#c8a96a" roughness={0.75} metalness={0.05} />
            </mesh>
          )}

          {/* ── RIM border (raised outline) ───────────────── */}
          {design.rimWidthMm > 0 && (() => {
            const rim = bodyShape(design);
            const inner = bodyShape({
              ...design,
              bodyWidthMm: design.bodyWidthMm - 2 * design.rimWidthMm,
              bodyHeightMm: design.bodyHeightMm - 2 * design.rimWidthMm,
              cornerRadiusMm: Math.max(design.cornerRadiusMm - design.rimWidthMm, 0),
            });
            // Translate inner so it's centered in rim
            const path = new THREE.Path();
            const innerPts = inner.getPoints(32).map((p) => [p.x + design.rimWidthMm, p.y + design.rimWidthMm] as [number, number]);
            path.moveTo(innerPts[0][0], innerPts[0][1]);
            for (let i = 1; i < innerPts.length; i++) path.lineTo(innerPts[i][0], innerPts[i][1]);
            rim.holes.push(path);
            return (
              <mesh castShadow receiveShadow position={[0, 0, 2.0]}>
                <extrudeGeometry args={[rim, { depth: 0.45, bevelEnabled: false }]} />
                <meshStandardMaterial color="#7a6440" roughness={0.7} />
              </mesh>
            );
          })()}

          {/* ── MAP CONTENT (extruded city) ────────────────── */}
          <group position={[0, 0, 2.0]}>
            {/* Parks first (lowest) */}
            {parkMeshes.map((s, i) => (
              <mesh key={`p-${i}`} castShadow receiveShadow>
                <extrudeGeometry args={[s, { depth: 0.32, bevelEnabled: false }]} />
                <meshStandardMaterial color="#88b06e" roughness={0.85} />
              </mesh>
            ))}
            {/* Water */}
            {waterMeshes.map((s, i) => (
              <mesh key={`w-${i}`} castShadow receiveShadow>
                <extrudeGeometry args={[s, { depth: 0.28, bevelEnabled: false }]} />
                <meshStandardMaterial color="#5a91c4" roughness={0.4} />
              </mesh>
            ))}
            {/* Roads */}
            {roadMeshes.map((r, i) => (
              <mesh key={`r-${i}`} castShadow receiveShadow>
                <extrudeGeometry args={[r.shape, { depth: r.kind === "major" ? 0.5 : 0.44, bevelEnabled: false }]} />
                <meshStandardMaterial color={r.kind === "major" ? "#1a1a1a" : r.kind === "minor" ? "#3a3a3a" : "#5a5a5a"} roughness={0.65} />
              </mesh>
            ))}
            {/* Buildings — extruded prisms */}
            {buildingMeshes.map((b, i) => (
              <mesh key={`b-${i}`} castShadow receiveShadow>
                <extrudeGeometry args={[b.shape, { depth: b.h, bevelEnabled: false }]} />
                <meshStandardMaterial color="#e8dfca" roughness={0.8} />
              </mesh>
            ))}
          </group>
        </group>

        <OrbitControls
          enablePan={false}
          enableZoom={false}
          maxPolarAngle={Math.PI / 2.1}
          minPolarAngle={Math.PI / 6}
          autoRotate
          autoRotateSpeed={0.5}
        />
      </Canvas>
    </div>
  );
}
