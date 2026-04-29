"use client";

import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useMemo } from "react";
import * as THREE from "three";
import { useGenerationStore } from "@/store/generation-store";

const COLORS = {
  base: "#d8d0bd",
  road: "#30343b",
  building: "#cfd4dc",
  water: "#4e94c8",
  park: "#5f9f68",
};

function project(lon: number, lat: number, center: { lat: number; lon: number }) {
  const latScale = 111_320;
  const lonScale = 111_320 * Math.cos((center.lat * Math.PI) / 180);
  return {
    x: (lon - center.lon) * lonScale,
    z: -(lat - center.lat) * latScale,
  };
}

function polygonRings(geometry: any): number[][][][] {
  if (!geometry) return [];
  if (geometry.type === "Polygon") return [geometry.coordinates];
  if (geometry.type === "MultiPolygon") return geometry.coordinates;
  return [];
}

function lineStrings(geometry: any): number[][][] {
  if (!geometry) return [];
  if (geometry.type === "LineString") return [geometry.coordinates];
  if (geometry.type === "MultiLineString") return geometry.coordinates;
  return [];
}

function shapeFromRing(ring: number[][], center: { lat: number; lon: number }, scale: number) {
  const pts = ring.map(([lon, lat]) => {
    const p = project(lon, lat, center);
    return new THREE.Vector2(p.x * scale, p.z * scale);
  });
  return new THREE.Shape(pts);
}

function FeatureLayer({ preview }: { preview: NonNullable<ReturnType<typeof useGenerationStore.getState>["fastPreview"]> }) {
  const group = useMemo(() => {
    const root = new THREE.Group();
    const center = preview.center;
    const widthM = Math.max(
      80,
      Math.abs(project(preview.bounds.east, preview.bounds.north, center).x - project(preview.bounds.west, preview.bounds.south, center).x),
    );
    const scale = 92 / widthM;

    const baseShape = new THREE.Shape([
      new THREE.Vector2(project(preview.bounds.west, preview.bounds.north, center).x * scale, project(preview.bounds.west, preview.bounds.north, center).z * scale),
      new THREE.Vector2(project(preview.bounds.east, preview.bounds.north, center).x * scale, project(preview.bounds.east, preview.bounds.north, center).z * scale),
      new THREE.Vector2(project(preview.bounds.east, preview.bounds.south, center).x * scale, project(preview.bounds.east, preview.bounds.south, center).z * scale),
      new THREE.Vector2(project(preview.bounds.west, preview.bounds.south, center).x * scale, project(preview.bounds.west, preview.bounds.south, center).z * scale),
    ]);
    const base = new THREE.Mesh(
      new THREE.ShapeGeometry(baseShape),
      new THREE.MeshStandardMaterial({ color: COLORS.base, roughness: 0.9, side: THREE.DoubleSide }),
    );
    base.rotation.x = -Math.PI / 2;
    base.position.y = -0.08;
    root.add(base);

    for (const feature of preview.layers.parks?.features ?? []) {
      for (const rings of polygonRings(feature.geometry)) {
        if (!rings[0]) continue;
        const mesh = new THREE.Mesh(
          new THREE.ShapeGeometry(shapeFromRing(rings[0], center, scale)),
          new THREE.MeshStandardMaterial({ color: COLORS.park, roughness: 0.95, side: THREE.DoubleSide }),
        );
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.y = 0.025;
        root.add(mesh);
      }
    }

    for (const feature of preview.layers.water?.features ?? []) {
      for (const rings of polygonRings(feature.geometry)) {
        if (!rings[0]) continue;
        const mesh = new THREE.Mesh(
          new THREE.ShapeGeometry(shapeFromRing(rings[0], center, scale)),
          new THREE.MeshStandardMaterial({ color: COLORS.water, roughness: 0.4, side: THREE.DoubleSide }),
        );
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.y = 0.035;
        root.add(mesh);
      }
    }

    const roadMaterial = new THREE.LineBasicMaterial({ color: COLORS.road });
    for (const feature of preview.layers.roads?.features ?? []) {
      for (const line of lineStrings(feature.geometry)) {
        const points = line.map(([lon, lat]) => {
          const p = project(lon, lat, center);
          return new THREE.Vector3(p.x * scale, 0.09, p.z * scale);
        });
        if (points.length > 1) root.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), roadMaterial));
      }
    }

    for (const feature of preview.layers.buildings?.features ?? []) {
      const height = Math.max(0.9, Math.min(10, Number(feature.properties?.height_m ?? 9) * scale * 0.9));
      for (const rings of polygonRings(feature.geometry)) {
        if (!rings[0]) continue;
        const geometry = new THREE.ExtrudeGeometry(shapeFromRing(rings[0], center, scale), {
          depth: height,
          bevelEnabled: false,
        });
        geometry.rotateX(-Math.PI / 2);
        const mesh = new THREE.Mesh(
          geometry,
          new THREE.MeshStandardMaterial({ color: COLORS.building, roughness: 0.85 }),
        );
        mesh.position.y = 0.08;
        root.add(mesh);
      }
    }

    root.rotation.y = Math.PI;
    return root;
  }, [preview]);

  return <primitive object={group} />;
}

export function FastPreview3D() {
  const preview = useGenerationStore((state) => state.fastPreview);

  return (
    <div className="h-full min-h-[360px] w-full bg-[#f5f1e8]">
      {!preview ? (
        <div className="flex h-full min-h-[360px] items-center justify-center px-6 text-center text-sm text-slate-500">
          Оберіть ділянку на мапі й натисніть “Створити прев'ю”.
        </div>
      ) : (
        <Canvas dpr={[1, 1.5]} gl={{ antialias: true }}>
          <PerspectiveCamera makeDefault position={[0, 80, 105]} fov={42} />
          <ambientLight intensity={0.9} />
          <directionalLight position={[40, 70, 40]} intensity={1.4} />
          <FeatureLayer preview={preview} />
          <gridHelper args={[120, 12, "#d7cbb5", "#e7decc"]} position={[0, -0.1, 0]} />
          <OrbitControls enableDamping minDistance={45} maxDistance={180} />
        </Canvas>
      )}
    </div>
  );
}
