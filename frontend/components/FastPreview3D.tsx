"use client";

import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { Canvas, useLoader } from "@react-three/fiber";
import { RotateCcw } from "lucide-react";
import { Component, ReactNode, Suspense, useMemo } from "react";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import type { FastPreviewResponse } from "@/lib/api";

type LayerKey = "terrain" | "roads" | "buildings" | "water" | "parks";

const MATERIALS: Record<string, { plate: string; building: string; road: string; water: string; park: string }> = {
  white: { plate: "#2d6f3f", building: "#46494a", road: "#090b0b", water: "#1f5f73", park: "#c4b287" },
  concrete: { plate: "#6f776c", building: "#3f4242", road: "#101111", water: "#486b73", park: "#b9ab88" },
  graphite: { plate: "#26382c", building: "#55595b", road: "#050606", water: "#163d52", park: "#7c765e" },
  green: { plate: "#2d6f3f", building: "#474a4b", road: "#080909", water: "#1e6179", park: "#326c3b" },
  terracotta: { plate: "#7a6f47", building: "#46494a", road: "#0a0908", water: "#315f66", park: "#c8ad82" },
};

class PreviewCanvasErrorBoundary extends Component<
  { resetKey: string; children: ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidUpdate(prevProps: { resetKey: string }) {
    if (prevProps.resetKey !== this.props.resetKey && this.state.hasError) {
      this.setState({ hasError: false });
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-[#fbf8ef]">
          <div className="max-w-[280px] text-center">
            <div className="font-serif text-xl text-[#1f2420]">Preview файл ще недоступний</div>
            <p className="mt-2 text-xs leading-5 text-[#777064]">
              Модель згенерована, але браузер не зміг завантажити STL. Натисніть “Камера” або оновіть preview за кілька секунд.
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

function projectPoint(lng: number, lat: number, center: { lat: number; lng: number }) {
  const metersPerDegLat = 111_320;
  const metersPerDegLng = 111_320 * Math.cos((center.lat * Math.PI) / 180);
  return {
    x: (lng - center.lng) * metersPerDegLng,
    z: -(lat - center.lat) * metersPerDegLat,
  };
}

function shapeFromPolygon(coords: any, center: { lat: number; lng: number }, scale: number) {
  if (!Array.isArray(coords) || coords.length === 0) return null;
  const rings = coords
    .map((ring: any[]) =>
      ring
        .filter((pt) => Array.isArray(pt) && pt.length >= 2)
        .map((pt) => {
          const p = projectPoint(Number(pt[0]), Number(pt[1]), center);
          return new THREE.Vector2(p.x * scale, p.z * scale);
        }),
    )
    .filter((ring: THREE.Vector2[]) => ring.length >= 3);
  if (!rings.length) return null;
  const shape = new THREE.Shape(rings[0]);
  rings.slice(1).forEach((ring: THREE.Vector2[]) => {
    shape.holes.push(new THREE.Path(ring));
  });
  return shape;
}

function polygonsFromGeometry(geometry: any): any[] {
  if (!geometry) return [];
  if (geometry.type === "Polygon") return [geometry.coordinates];
  if (geometry.type === "MultiPolygon") return geometry.coordinates;
  return [];
}

function PlanarPolygon({
  geometry,
  center,
  scale,
  y,
  color,
  opacity = 1,
}: {
  geometry: any;
  center: { lat: number; lng: number };
  scale: number;
  y: number;
  color: string;
  opacity?: number;
}) {
  const geometries = useMemo(() => {
    return polygonsFromGeometry(geometry)
      .map((coords: any) => {
        const shape = shapeFromPolygon(coords, center, scale);
        if (!shape) return null;
        const geom = new THREE.ShapeGeometry(shape);
        geom.rotateX(-Math.PI / 2);
        geom.computeVertexNormals();
        return geom;
      })
      .filter(Boolean) as THREE.BufferGeometry[];
  }, [geometry, center, scale]);

  return (
    <group position={[0, y, 0]}>
      {geometries.map((geom, index) => (
        <mesh key={index} geometry={geom} receiveShadow>
          <meshStandardMaterial color={color} roughness={0.86} transparent={opacity < 1} opacity={opacity} side={THREE.DoubleSide} />
        </mesh>
      ))}
    </group>
  );
}

function TerrainSurface({
  width,
  height,
  color,
  relief,
  scale,
  heightfield,
}: {
  width: number;
  height: number;
  color: string;
  relief: number;
  scale: number;
  heightfield?: FastPreviewResponse["layers"]["terrain"]["heightfield"];
}) {
  const geometry = useMemo(() => {
    if (heightfield?.x?.length && heightfield?.y?.length && Array.isArray(heightfield.z)) {
      const xs = heightfield.x.map(Number);
      const ys = heightfield.y.map(Number);
      const rows = ys.length;
      const cols = xs.length;
      if (rows >= 2 && cols >= 2 && heightfield.z.length >= rows) {
        const positions: number[] = [];
        const indices: number[] = [];
        for (let row = 0; row < rows; row += 1) {
          const zRow = heightfield.z[row] ?? [];
          for (let col = 0; col < cols; col += 1) {
            const x = xs[col] * scale;
            const y = Math.max(0, Number(zRow[col] ?? 0)) * scale * 1.15;
            const z = -ys[row] * scale;
            positions.push(x, y, z);
          }
        }
        for (let row = 0; row < rows - 1; row += 1) {
          for (let col = 0; col < cols - 1; col += 1) {
            const a = row * cols + col;
            const b = (row + 1) * cols + col;
            const c = row * cols + col + 1;
            const d = (row + 1) * cols + col + 1;
            indices.push(a, b, c, c, b, d);
          }
        }
        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
        geom.setIndex(indices);
        geom.computeVertexNormals();
        return geom;
      }
    }

    const segments = 26;
    const geom = new THREE.PlaneGeometry(width, height, segments, segments);
    geom.rotateX(-Math.PI / 2);
    const pos = geom.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i += 1) {
      const x = pos.getX(i);
      const z = pos.getZ(i);
      const nx = width > 0 ? x / width : 0;
      const nz = height > 0 ? z / height : 0;
      const ridge =
        Math.sin(nx * Math.PI * 5.2 + 0.7) * 0.45 +
        Math.cos(nz * Math.PI * 4.4 - 0.35) * 0.34 +
        Math.sin((nx + nz) * Math.PI * 6.1) * 0.21;
      const edgeFalloff = Math.max(0.35, 1 - Math.max(Math.abs(nx), Math.abs(nz)) * 0.72);
      pos.setY(i, Math.max(0, ridge * relief * edgeFalloff));
    }
    geom.computeVertexNormals();
    return geom;
  }, [width, height, relief, scale, heightfield]);

  return (
    <mesh geometry={geometry} receiveShadow castShadow>
      <meshStandardMaterial color={color} roughness={0.9} flatShading />
    </mesh>
  );
}

function BuildingBlock({
  geometry,
  center,
  scale,
  height,
  color,
}: {
  geometry: any;
  center: { lat: number; lng: number };
  scale: number;
  height: number;
  color: string;
}) {
  const geometries = useMemo(() => {
    return polygonsFromGeometry(geometry)
      .map((coords: any) => {
        const shape = shapeFromPolygon(coords, center, scale);
        if (!shape) return null;
        const geom = new THREE.ExtrudeGeometry(shape, {
          depth: Math.max(0.08, height),
          bevelEnabled: false,
          curveSegments: 1,
        });
        geom.rotateX(-Math.PI / 2);
        geom.computeVertexNormals();
        return geom;
      })
      .filter(Boolean) as THREE.BufferGeometry[];
  }, [geometry, center, scale, height]);

  return (
    <group position={[0, 0.18, 0]}>
      {geometries.map((geom, index) => (
        <mesh key={index} geometry={geom} castShadow receiveShadow>
          <meshStandardMaterial color={color} roughness={0.72} />
        </mesh>
      ))}
    </group>
  );
}

function PreviewScene({
  preview,
  visibleLayers,
  material,
}: {
  preview: FastPreviewResponse;
  visibleLayers: Record<LayerKey, boolean>;
  material: string;
}) {
  const palette = MATERIALS[material] ?? MATERIALS.white;
  const widthM = Math.max(
    80,
    Math.abs(projectPoint(preview.bounds.east, preview.center.lat, preview.center).x - projectPoint(preview.bounds.west, preview.center.lat, preview.center).x),
  );
  const heightM = Math.max(
    80,
    Math.abs(projectPoint(preview.center.lng, preview.bounds.north, preview.center).z - projectPoint(preview.center.lng, preview.bounds.south, preview.center).z),
  );
  const modelLogic = preview.model_logic ?? {};
  const modelSizeMm = Number(modelLogic.model_size_mm || 180);
  const visualScale = 16 / Math.max(modelSizeMm, 1);
  const scaleFactorMmPerM = Number(modelLogic.scale_factor_mm_per_m || modelSizeMm / Math.max(widthM, heightM));
  const scale = scaleFactorMmPerM * visualScale;
  const plateW = widthM * scale;
  const plateH = heightM * scale;
  const mmToScene = (value: number, fallback: number) => Math.max(0.01, Number.isFinite(value) ? value * visualScale : fallback);
  const baseThickness = mmToScene(Number(modelLogic.terrain_base_thickness_mm), 0.5);
  const relief = Math.max(0.14, Math.min(0.9, Number(modelLogic.terrain_z_scale || 0.5) * 0.55));
  const terrainHeightfield = preview.layers.terrain?.heightfield ?? null;
  const terrainTopY = Math.max(
    relief,
    Number(terrainHeightfield?.z_max_m ?? 0) * scale * 1.15,
  );
  const roadY = terrainTopY + mmToScene(Number(modelLogic.road_height_mm), 0.08);
  const parkY = terrainTopY + mmToScene(Number(modelLogic.parks_height_mm), 0.06);
  const waterY = relief * 0.35;

  const buildingFeatures = preview.layers.buildings?.features ?? [];
  const roadFeatures = preview.layers.roads?.features ?? [];
  const waterFeatures = preview.layers.water?.features ?? [];
  const parkFeatures = preview.layers.parks?.features ?? [];

  return (
    <>
      <color attach="background" args={["#050918"]} />
      <ambientLight intensity={0.5} />
      <hemisphereLight args={["#f7f3e7", "#0b1420", 0.5]} />
      <directionalLight position={[16, 22, 14]} intensity={1.25} castShadow />
      <directionalLight position={[-14, 10, -10]} intensity={0.16} />

      <group rotation={[0, -0.15, 0]}>
        {visibleLayers.terrain && (
          <>
            <mesh position={[0, -baseThickness / 2, 0]} receiveShadow castShadow>
              <boxGeometry args={[plateW, baseThickness, plateH]} />
              <meshStandardMaterial color="#8d7b55" roughness={0.9} />
            </mesh>
            <TerrainSurface
              width={plateW}
              height={plateH}
              color={palette.plate}
              relief={relief}
              scale={scale}
              heightfield={terrainHeightfield}
            />
          </>
        )}

        {visibleLayers.water &&
          waterFeatures.map((feature: any, index: number) => (
            <PlanarPolygon key={`water-${index}`} geometry={feature.geometry} center={preview.center} scale={scale} y={waterY} color={palette.water} />
          ))}

        {visibleLayers.parks &&
          parkFeatures.map((feature: any, index: number) => (
            <PlanarPolygon key={`park-${index}`} geometry={feature.geometry} center={preview.center} scale={scale} y={parkY} color={palette.park} />
          ))}

        {visibleLayers.roads &&
          roadFeatures.map((feature: any, index: number) => (
            <PlanarPolygon key={`road-${index}`} geometry={feature.geometry} center={preview.center} scale={scale} y={roadY} color={palette.road} />
          ))}

        {visibleLayers.buildings &&
          buildingFeatures.map((feature: any, index: number) => (
            <BuildingBlock
              key={`building-${index}`}
              geometry={feature.geometry}
              center={preview.center}
              scale={scale}
              height={mmToScene(Number(feature.properties?.height_mm), 0.8) + terrainTopY * 0.5}
              color={palette.building}
            />
          ))}
      </group>
    </>
  );
}

function FullPipelineStlModel({ url, material }: { url: string; material: string }) {
  const sourceGeometry = useLoader(STLLoader, url);
  const palette = MATERIALS[material] ?? MATERIALS.white;
  const geometry = useMemo(() => {
    const geom = sourceGeometry.clone();
    geom.rotateX(-Math.PI / 2);
    geom.computeBoundingBox();
    const box = geom.boundingBox;
    if (box) {
      const center = new THREE.Vector3();
      const size = new THREE.Vector3();
      box.getCenter(center);
      box.getSize(size);
      geom.translate(-center.x, -box.min.y, -center.z);
      const maxSide = Math.max(size.x, size.z, 1);
      const scale = 16 / maxSide;
      geom.scale(scale, scale, scale);
    }
    geom.computeVertexNormals();
    geom.computeBoundingSphere();
    return geom;
  }, [sourceGeometry]);

  return (
    <mesh geometry={geometry} castShadow receiveShadow>
      <meshStandardMaterial color={palette.building} roughness={0.74} metalness={0.02} />
    </mesh>
  );
}

function FullPipelinePreviewScene({ url, material }: { url: string; material: string }) {
  return (
    <>
      <ambientLight intensity={0.72} />
      <hemisphereLight args={["#ffffff", "#c6bfae", 0.58]} />
      <directionalLight position={[16, 22, 14]} intensity={1.08} castShadow />
      <directionalLight position={[-14, 10, -10]} intensity={0.28} />
      <group rotation={[0, -0.2, 0]}>
        <FullPipelineStlModel url={url} material={material} />
      </group>
    </>
  );
}

export function FastPreview3D({
  preview,
  loading,
  error,
  visibleLayers,
  material,
  onReset,
}: {
  preview: FastPreviewResponse | null;
  loading: boolean;
  error?: string | null;
  visibleLayers: Record<LayerKey, boolean>;
  material: string;
  onReset?: () => void;
}) {
  const processing = preview?.preview_status === "processing";
  const failed = preview?.preview_status === "failed";
  const ready = Boolean(preview && !processing && !failed && !loading && !error);
  const previewMessage = String(preview?.model_logic?.preview_message || "");
  const fullPipelineModelUrl = preview?.model_file_url || preview?.preview_stl || null;

  return (
    <div className="relative h-full min-h-[300px] w-full overflow-hidden bg-[#050918]">
      {!ready && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-[#fbf8ef]">
          <div className="max-w-[260px] text-center">
            <div className="mx-auto mb-4 h-20 w-28 rounded-[8px] border border-[#ddd4c4] bg-[#f0eadf] shadow-inner" />
            <div className="font-serif text-xl text-[#1f2420]">
              {loading || processing ? "Готуємо точне preview" : error || failed ? "Preview не вдалося створити" : "Виділіть ділянку на мапі"}
            </div>
            <p className="mt-2 text-xs leading-5 text-[#777064]">
              {loading || processing
                ? previewMessage || "Генеруємо повну 3D-модель тим самим pipeline, що й основна генерація: рельєф, шари, пази та export. Це може зайняти кілька хвилин."
                : error || previewMessage || "Після вибору району тут зʼявиться повна 3D-модель з рельєфом, дорогами, будівлями, водою, парками і пазами."}
            </p>
          </div>
        </div>
      )}

      {ready && preview && fullPipelineModelUrl && (
        <PreviewCanvasErrorBoundary resetKey={`${fullPipelineModelUrl}:${material}`}>
          <Canvas shadows dpr={[1, 2]} camera={{ position: [20, 17, 20], fov: 30, near: 0.1, far: 180 }}>
            <Suspense fallback={null}>
              <PerspectiveCamera makeDefault position={[20, 17, 20]} fov={30} />
              <OrbitControls
                makeDefault
                enableDamping
                dampingFactor={0.09}
                minDistance={8}
                maxDistance={60}
                maxPolarAngle={Math.PI * 0.5}
                minPolarAngle={Math.PI * 0.12}
                enablePan={false}
                target={[0, 0.6, 0]}
              />
              <FullPipelinePreviewScene url={fullPipelineModelUrl} material={material} />
            </Suspense>
          </Canvas>
        </PreviewCanvasErrorBoundary>
      )}

      {ready && preview && !fullPipelineModelUrl && (
        <PreviewCanvasErrorBoundary resetKey={`geojson:${preview.preview_id}:${material}`}>
          <Canvas shadows dpr={[1, 2]} camera={{ position: [18, 10, 18], fov: 34, near: 0.1, far: 180 }}>
            <Suspense fallback={null}>
              <PerspectiveCamera makeDefault position={[18, 10, 18]} fov={34} />
              <OrbitControls
                makeDefault
                enableDamping
                dampingFactor={0.09}
                minDistance={7}
                maxDistance={46}
                maxPolarAngle={Math.PI * 0.56}
                minPolarAngle={Math.PI * 0.08}
                enablePan={false}
                target={[0, 0.35, 0]}
              />
              <PreviewScene preview={preview} visibleLayers={visibleLayers} material={material} />
            </Suspense>
          </Canvas>
        </PreviewCanvasErrorBoundary>
      )}

      <button
        type="button"
        onClick={onReset}
        className="absolute bottom-3 right-3 z-20 inline-flex items-center gap-2 rounded-[6px] border border-[#ddd4c4] bg-[#fffdf7] px-3 py-2 text-xs font-medium text-[#3a3f3a] shadow-sm"
      >
        <RotateCcw size={14} />
        Камера
      </button>
    </div>
  );
}
