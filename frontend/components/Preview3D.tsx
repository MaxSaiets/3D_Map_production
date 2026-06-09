"use client";

import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { Suspense, useEffect, useMemo, useState, useRef } from "react";
import { useGenerationStore } from "@/store/generation-store";
import { api } from "@/lib/api";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { ThreeMFLoader } from "three/examples/jsm/loaders/3MFLoader.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { useFrame } from "@react-three/fiber";

function bakeStlZUpToThreeYUp(object: THREE.Object3D) {
  // STL/3MF зазвичай Z-up, а Three.js сцена Y-up.
  // Перетворюємо модель так, щоб вона “лежала” на grid (XZ), і не ставала “стіною”.
  const rot = -Math.PI / 2;

  const bakeMesh = (mesh: THREE.Mesh) => {
    const geom = mesh.geometry as THREE.BufferGeometry | undefined;
    if (!geom) return;
    try {
      geom.rotateX(rot);
      geom.computeBoundingBox();
      geom.computeBoundingSphere();
      // Нормалі після повороту (щоб освітлення було коректним)
      // computeVertexNormals може бути важким на великих мешах, але для STL превʼю це ок.
      geom.computeVertexNormals();
    } catch {
      // ignore
    }
  };

  if (object instanceof THREE.Mesh) {
    bakeMesh(object);
  } else {
    object.traverse((child) => {
      if (child instanceof THREE.Mesh) bakeMesh(child);
    });
  }
}

async function loadStlAsMesh(blob: Blob, color: number): Promise<THREE.Mesh> {
  const url = URL.createObjectURL(blob);
  const loader = new STLLoader();
  return await new Promise<THREE.Mesh>((resolve, reject) => {
    loader.load(
      url,
      (geometry) => {
        URL.revokeObjectURL(url);
        // DoubleSide: preview meshes from backend can be top-only (no bottom
        // cap in preview mode). Without DoubleSide the underside renders as
        // transparent and you see the 3D viewport's floor grid through it.
        const material = new THREE.MeshStandardMaterial({ color, flatShading: true, side: THREE.DoubleSide });
        const mesh = new THREE.Mesh(geometry, material);
        bakeStlZUpToThreeYUp(mesh);
        resolve(mesh);
      },
      undefined,
      (err) => {
        URL.revokeObjectURL(url);
        reject(err);
      }
    );
  });
}

async function loadColoredPartsFromBlobs(blobs: Partial<Record<"base" | "roads" | "buildings" | "water", Blob>>): Promise<THREE.Group> {
  const group = new THREE.Group();
  const colors: Record<string, number> = {
    base: 0xc8b48e,
    terrain: 0xc8b48e,
    roads: 0x3c3c3c,
    buildings: 0xe3e3e3,
    water: 0x6496c8,
    parks: 0x649664,
    green: 0x649664,
  };

  const entries = Object.entries(blobs) as Array<[keyof typeof blobs, Blob]>;
  for (const [part, blob] of entries) {
    if (!blob) continue;
    const mesh = await loadStlAsMesh(blob, colors[part as string] ?? 0x888888);
// removed-debug-log
    // mark part type for later preview toggles (e.g. shading)
    (mesh as any).userData = { ...(mesh as any).userData, part };
    const mat = mesh.material as THREE.MeshStandardMaterial;
    // Налаштування матеріалів для кращої читабельності
    if (part === "base") {
      mat.flatShading = true; // Use flat shading to show sharp wall edges
      mat.transparent = false;
      mat.opacity = 1.0;
      mat.roughness = 1.0;
      mat.metalness = 0.0;
      // ВАЖЛИВО: якщо нормалі бази інколи "перевернуті", з FrontSide вона здається прозорою зверху.
      // Для превʼю робимо DoubleSide.
      mat.side = THREE.DoubleSide;
      mat.needsUpdate = true;
    } else if (part === "buildings") {
      mat.flatShading = false;
      mat.roughness = 0.9;
      mat.metalness = 0.0;
      // Прибирає мерехтіння на стику з землею в превʼю
      mat.polygonOffset = true;
      // Робимо дуже мʼяко, щоб не створювало ілюзію “будівлі висять”.
      mat.polygonOffsetFactor = -0.1;
      mat.polygonOffsetUnits = -1;
      mat.needsUpdate = true;
    } else if (part === "roads") {
      mat.flatShading = true;
      mat.roughness = 0.95;
      mat.metalness = 0.0;
      // Для превʼю: легкий polygonOffset, щоб дороги не "зливалися" з землею (z-fighting),
      // але без агресивних значень (які давали ефект “висять”).
      mat.polygonOffset = true;
      mat.polygonOffsetFactor = -0.1;
      mat.polygonOffsetUnits = -1;
      mat.needsUpdate = true;
    } else if (part === "water") {
      // Вода як "видимий шар": без прозорості, щоб не було “шипів”/стіночок, видимих крізь воду.
      // Для друку вода все одно керується геометрією на бекенді.
      mat.transparent = false;
      mat.opacity = 1.0;
      mat.roughness = 0.3;
      mat.metalness = 0.0;
      mat.needsUpdate = true;
    } else if (part === "parks") {
      mat.flatShading = false;
      mat.roughness = 1.0;
      mat.metalness = 0.0;
      // Prevent z-fighting “thin green lines” on top of terrain in preview
      mat.polygonOffset = true;
      mat.polygonOffsetFactor = -0.2;
      mat.polygonOffsetUnits = -2;
      mat.needsUpdate = true;
    }
    group.add(mesh);
  }
  return group;
}

// Функція для завантаження локального 3MF fallback.
async function load3MF(blob: Blob): Promise<THREE.Group> {
  const zipUrl = URL.createObjectURL(blob);
  return await new Promise<THREE.Group>((resolve, reject) => {
    const loader = new ThreeMFLoader();
    loader.load(
      zipUrl,
      (object) => {
        URL.revokeObjectURL(zipUrl);
        const group = new THREE.Group();
        group.add(object);

        let totalVertices = 0;
        let totalMeshes = 0;
        const colorMap: Record<string, number> = {
          base: 0xc8b48e,
          terrain: 0xc8b48e,
          roads: 0x3c3c3c,
          buildings: 0xe3e3e3,
          water: 0x6496c8,
          parks: 0x649664,
          green: 0x649664,
        };

        group.traverse((child) => {
          if (!(child instanceof THREE.Mesh)) return;

          totalMeshes++;
          const geometry = child.geometry;
          if (geometry.attributes.position) {
            totalVertices += geometry.attributes.position.count;
          }

          const materials = Array.isArray(child.material) ? child.material : [child.material];

          const name = child.name.toLowerCase();
          let partKey: string | null = null;
          let partColor: number | null = null;
          for (const [key, color] of Object.entries(colorMap)) {
            if (name.includes(key)) {
              partKey = key;
              partColor = color;
              break;
            }
          }
          if (partKey) {
            (child as any).userData = { ...(child as any).userData, part: partKey };
          }

          for (const material of materials) {
            if (!material) continue;
            const maybeColored = material as THREE.Material & { color?: THREE.Color };
            if (partColor !== null && maybeColored.color?.getHex() === 0xffffff) {
              maybeColored.color.setHex(partColor);
            }
            material.side = THREE.DoubleSide;
            material.needsUpdate = true;
          }
        });

        if (totalVertices === 0) {
          reject(new Error("Модель не містить вершин"));
          return;
        }

// removed-debug-log
        resolve(group);
      },
      undefined,
      (error) => {
        URL.revokeObjectURL(zipUrl);
        reject(error);
      }
    );
  });
}

async function loadGLB(blob: Blob): Promise<THREE.Group> {
  const url = URL.createObjectURL(blob);
  return await new Promise<THREE.Group>((resolve, reject) => {
    const loader = new GLTFLoader();
    loader.load(
      url,
      (gltf) => {
        URL.revokeObjectURL(url);
        const group = gltf.scene || new THREE.Group();
        // Trimesh exports GLB in the same Z-up map space as STL/3MF. Rotate
        // the scene root cheaply; do not bake every geometry or recompute
        // normals here, because preview GLBs can contain hundreds of thousands
        // of faces and that kept the viewer stuck in the loading placeholder.
        group.rotation.x = -Math.PI / 2;
        group.updateMatrixWorld(true);
        let totalVertices = 0;
        let totalMeshes = 0;
        const colorMap: Record<string, { color: number; part: string }> = {
          base: { color: 0xc8b48e, part: "base" },
          terrain: { color: 0xc8b48e, part: "terrain" },
          roads: { color: 0x3c3c3c, part: "roads" },
          buildings: { color: 0xe3e3e3, part: "buildings" },
          water: { color: 0x6496c8, part: "water" },
          parks: { color: 0x649664, part: "parks" },
          green: { color: 0x649664, part: "parks" },
        };
        group.traverse((child) => {
          if (!(child instanceof THREE.Mesh)) return;
          totalMeshes++;
          const geometry = child.geometry;
          if (geometry.attributes.position) {
            totalVertices += geometry.attributes.position.count;
          }
          const materialNames = (Array.isArray(child.material) ? child.material : [child.material])
            .map((material) => material?.name || "")
            .join(" ");
          const name = `${child.name || ""} ${child.parent?.name || ""} ${materialNames}`.toLowerCase();
          const entry = Object.entries(colorMap).find(([key]) => name.includes(key))?.[1];
          if (entry) {
            child.userData = { ...(child.userData || {}), part: entry.part };
          }
          const isSurfaceDecal =
            entry?.part === "roads" || entry?.part === "parks" || entry?.part === "water";
          child.material = new THREE.MeshBasicMaterial({
            color: entry?.color ?? 0x9a9a9a,
            side: THREE.DoubleSide,
            polygonOffset: isSurfaceDecal,
            polygonOffsetFactor: -2,
            polygonOffsetUnits: -2,
          });
        });
        if (totalVertices === 0) {
          reject(new Error("GLB preview не містить вершин"));
          return;
        }
// removed-debug-log
        resolve(group);
      },
      undefined,
      (error) => {
        URL.revokeObjectURL(url);
        reject(error);
      }
    );
  });
}

async function loadPreviewModelForTask(taskId: string): Promise<THREE.Group> {
  let glbError: unknown = null;
  try {
    const blobGlb = await api.downloadModel(taskId, "glb");
    if (!blobGlb || blobGlb.size <= 100) {
      throw new Error("Локальне GLB preview порожнє або не створене");
    }
    try {
      return await loadGLB(blobGlb);
    } catch (error) {
      glbError = error;
      const type = String(blobGlb.type || "").toLowerCase();
      if (type.includes("3mf")) {
        return await load3MF(blobGlb);
      }
      throw error;
    }
  } catch (error) {
    glbError = error;
  }

  try {
    const blob3mf = await api.downloadModel(taskId, "3mf");
    if (!blob3mf || blob3mf.size <= 100) {
      throw new Error("Локальне 3MF preview порожнє або не створене");
    }
    return await load3MF(blob3mf);
  } catch (error3mf: any) {
    const glbMessage = glbError instanceof Error ? glbError.message : String(glbError || "");
    const mfMessage = error3mf instanceof Error ? error3mf.message : String(error3mf || "");
    throw new Error(`Не вдалося завантажити preview: GLB (${glbMessage}); 3MF (${mfMessage})`);
  }
}

// Компонент для автоматичного позиціювання камери
function CameraController() {
  const { downloadUrl, showAllZones, taskIds } = useGenerationStore();
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);

  useEffect(() => {
    // Налаштовуємо камеру для кращого перегляду
    const timer = setTimeout(() => {
      if (cameraRef.current) {
        // Для batch preview (всі зони) - більша відстань для кращого огляду
        if (showAllZones && taskIds && taskIds.length > 1) {
          const zoneCount = taskIds.length;
          // Відстань залежить від кількості зон (більше зон = більша відстань)
          const baseDistance = 300;
          const distanceMultiplier = Math.max(1, Math.sqrt(zoneCount) * 0.5);
          const distance = baseDistance * distanceMultiplier;
          cameraRef.current.position.set(distance, distance * 0.8, distance);
          cameraRef.current.lookAt(0, 0, 0);
// removed-debug-log
        } else {
          // Для однієї зони - стандартна відстань
          const distance = 300;
          cameraRef.current.position.set(distance, distance, distance);
          cameraRef.current.lookAt(0, 0, 0);
// removed-debug-log
        }
        cameraRef.current.updateProjectionMatrix();
      }
    }, 100);

    return () => clearTimeout(timer);
  }, [downloadUrl, showAllZones, taskIds]);

  return (
    <PerspectiveCamera
      ref={cameraRef}
      makeDefault
      position={[300, 300, 300]}
      fov={50}
      near={0.1}
      far={2000}
    />
  );
}

type RotateMode = "camera" | "model";
type CameraMode = "orbit" | "fly";

function FreeFlyControls({
  enabled,
  speed,
  onSpeedChange,
}: {
  enabled: boolean;
  speed: number;
  onSpeedChange: (v: number) => void;
}) {
  const { camera, gl } = useThree();
  const stateRef = useRef({
    keys: new Set<string>(),
    mouseDown: false,
    yaw: 0,
    pitch: 0,
    speed: 120, // units/sec (preview space)
    boost: 3.0,
    sensitivity: 0.0025,
  });

  const tmpForward = useMemo(() => new THREE.Vector3(), []);
  const tmpRight = useMemo(() => new THREE.Vector3(), []);
  const tmpUp = useMemo(() => new THREE.Vector3(0, 1, 0), []);
  const tmpMove = useMemo(() => new THREE.Vector3(), []);
  const euler = useMemo(() => new THREE.Euler(0, 0, 0, "YXZ"), []);

  // Initialize yaw/pitch from camera when enabling
  useEffect(() => {
    if (!enabled) return;
    // Sync external speed setting into control state
    stateRef.current.speed = Math.max(10, Math.min(800, Number(speed) || 120));
    const q = camera.quaternion.clone();
    euler.setFromQuaternion(q, "YXZ");
    stateRef.current.yaw = euler.y;
    stateRef.current.pitch = euler.x;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, speed]);

  useEffect(() => {
    if (!enabled) return;

    const el = gl.domElement;

    const onKeyDown = (e: KeyboardEvent) => {
      stateRef.current.keys.add(e.code);
      // Prevent page scroll when using arrows/space
      if (["Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.code)) {
        e.preventDefault();
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      stateRef.current.keys.delete(e.code);
    };
    const onMouseDown = (e: MouseEvent) => {
      // Right click or middle click enables look-around while held
      if (e.button === 2 || e.button === 1) {
        stateRef.current.mouseDown = true;
        e.preventDefault();
      }
    };
    const onMouseUp = (e: MouseEvent) => {
      if (e.button === 2 || e.button === 1) {
        stateRef.current.mouseDown = false;
        e.preventDefault();
      }
    };
    const onMouseMove = (e: MouseEvent) => {
      if (!stateRef.current.mouseDown) return;
      const s = stateRef.current;
      s.yaw -= e.movementX * s.sensitivity;
      s.pitch -= e.movementY * s.sensitivity;
      // Clamp pitch to avoid flipping
      const lim = Math.PI / 2 - 0.01;
      s.pitch = Math.max(-lim, Math.min(lim, s.pitch));
    };
    const onWheel = (e: WheelEvent) => {
      // Adjust speed with wheel (no page scroll when hovering canvas)
      const s = stateRef.current;
      const delta = Math.sign(e.deltaY);
      s.speed = Math.max(10, Math.min(800, s.speed * (delta > 0 ? 0.9 : 1.1)));
      onSpeedChange(s.speed);
      e.preventDefault();
    };
    const onContextMenu = (e: MouseEvent) => {
      // Disable context menu on canvas so RMB is usable
      e.preventDefault();
    };

    window.addEventListener("keydown", onKeyDown, { passive: false });
    window.addEventListener("keyup", onKeyUp);
    el.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mouseup", onMouseUp);
    window.addEventListener("mousemove", onMouseMove);
    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("contextmenu", onContextMenu);

    return () => {
      window.removeEventListener("keydown", onKeyDown as any);
      window.removeEventListener("keyup", onKeyUp as any);
      el.removeEventListener("mousedown", onMouseDown as any);
      window.removeEventListener("mouseup", onMouseUp as any);
      el.removeEventListener("mousemove", onMouseMove as any);
      el.removeEventListener("wheel", onWheel as any);
      el.removeEventListener("contextmenu", onContextMenu as any);
      stateRef.current.keys.clear();
      stateRef.current.mouseDown = false;
    };
  }, [enabled, gl.domElement]);

  useFrame((_, delta) => {
    if (!enabled) return;

    const s = stateRef.current;

    // Update camera rotation
    euler.set(s.pitch, s.yaw, 0);
    camera.quaternion.setFromEuler(euler);

    // Movement
    const keys = s.keys;
    const boost = keys.has("ShiftLeft") || keys.has("ShiftRight") ? s.boost : 1.0;
    const v = s.speed * boost * Math.min(delta, 0.05);

    tmpMove.set(0, 0, 0);

    // Forward is -Z in camera space
    tmpForward.set(0, 0, -1).applyQuaternion(camera.quaternion);
    tmpRight.set(1, 0, 0).applyQuaternion(camera.quaternion);

    // Optional: keep forward movement mostly horizontal for easier navigation
    tmpForward.y = 0;
    tmpRight.y = 0;
    tmpForward.normalize();
    tmpRight.normalize();

    if (keys.has("KeyW")) tmpMove.addScaledVector(tmpForward, v);
    if (keys.has("KeyS")) tmpMove.addScaledVector(tmpForward, -v);
    if (keys.has("KeyA")) tmpMove.addScaledVector(tmpRight, -v);
    if (keys.has("KeyD")) tmpMove.addScaledVector(tmpRight, v);
    if (keys.has("KeyE")) tmpMove.addScaledVector(tmpUp, v);
    if (keys.has("KeyQ")) tmpMove.addScaledVector(tmpUp, -v);

    if (tmpMove.lengthSq() > 0) {
      camera.position.add(tmpMove);
      camera.updateMatrixWorld();
    }
  });

  return null;
}

function ModelLoader({ rotateMode }: { rotateMode: RotateMode }) {
  const three = useThree();
  const camera = three.camera;
  const controls = (three as any).controls as { target?: THREE.Vector3; update?: () => void } | undefined;
  const { 
    downloadUrl, 
    activeTaskId, 
    exportFormat, 
    showAllZones, 
    taskIds, 
    taskStatuses, 
    batchZoneMetaByTaskId, 
    terrainSmoothShading,
    previewIncludeBase,
    previewIncludeRoads,
    previewIncludeBuildings,
    previewIncludeWater,
    previewIncludeParks,
  } = useGenerationStore();
  const [model, setModel] = useState<THREE.Group | THREE.Mesh | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasLoadedTestModel, setHasLoadedTestModel] = useState(false);

  // Керування поворотом моделі (а не камери)
  const modelGroupRef = useRef<THREE.Group | null>(null);
  const dragRef = useRef<{ dragging: boolean; x: number; y: number }>({ dragging: false, x: 0, y: 0 });

  const resetModelRotation = () => {
    if (modelGroupRef.current) {
      modelGroupRef.current.rotation.set(0, 0, 0);
    }
  };

  const fitCameraToObject = (object: THREE.Object3D) => {
    object.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty()) return;

    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    if (!Number.isFinite(maxDim) || maxDim <= 0) return;

    const perspective = camera as THREE.PerspectiveCamera;
    const fov = THREE.MathUtils.degToRad(perspective.fov || 50);
    const distance = (maxDim / (2 * Math.tan(fov / 2))) * 1.35;
    const horizontal = Math.max(distance, maxDim * 0.85);
    const vertical = Math.max(distance * 0.72, size.y * 1.35, 70);

    camera.position.set(center.x + horizontal, center.y + vertical, center.z + horizontal);
    if (controls?.target) {
      controls.target.copy(center);
      controls.update?.();
    } else {
      camera.lookAt(center.x, center.y, center.z);
    }
    perspective.near = Math.max(0.01, distance / 1000);
    perspective.far = Math.max(2000, distance * 20, maxDim * 20);
    perspective.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
  };

  // Немає готової задачі -> показуємо локальний placeholder нижче.
  // Не ходимо в /api/test-model: це старий demo fallback, який тільки дає 404 у preview.
  useEffect(() => {
    if (downloadUrl) return;
    setHasLoadedTestModel(false);
    setLoading(false);
  }, [downloadUrl]);

  // Batch preview:
  // - If tiles are exported in global XY (stitching mode), we should NOT normalize each tile individually,
  //   and we should NOT do artificial grid layout. Just load as-is and normalize the whole group once.
  // - If tiles are still centered (legacy), we fallback to the old grid layout so user can still see all zones.
  useEffect(() => {
    if (!showAllZones) return;
    if (!taskIds || taskIds.length < 2) return;

    const completedIds = taskIds.filter((id) => (taskStatuses as any)?.[id]?.status === "completed");
    const idsToLoad = completedIds.length ? completedIds : [];
    if (idsToLoad.length === 0) {
      // ще нічого не готово
      setModel(null);
      return;
    }

    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const loadZoneModelRaw = async (id: string) => {
          const loadedModel = await loadPreviewModelForTask(id);
          loadedModel.updateMatrixWorld(true);
          return { id, obj: loadedModel };
        };

        const models = (await Promise.all(idsToLoad.map((id) => loadZoneModelRaw(id)))).filter(Boolean) as any[];
        if (!models.length) {
          setModel(null);
          setLoading(false);
          return;
        }

        const group = new THREE.Group();
        for (const m of models) {
          m.obj.updateMatrixWorld(true);
          group.add(m.obj);
        }

        // Decide whether tiles already have meaningful relative positions.
        // If all centers are ~the same => legacy centered exports => use grid layout.
        const centers = models.map((m) => new THREE.Box3().setFromObject(m.obj).getCenter(new THREE.Vector3()));
        const mean = centers.reduce((acc, c) => acc.add(c), new THREE.Vector3()).multiplyScalar(1 / Math.max(1, centers.length));
        const spread = centers.reduce((acc, c) => acc + c.clone().sub(mean).length(), 0) / Math.max(1, centers.length);
        // Treat tiles as already globally-positioned ONLY if their centres are
        // spread far apart relative to a tile's own size. Otherwise (tiles all
        // near origin) they would overlap into one stacked blob — so fall through
        // to an explicit non-overlapping grid layout.
        const maxTileDim = Math.max(
          1,
          ...models.map((m) => {
            const s = new THREE.Box3().setFromObject(m.obj).getSize(new THREE.Vector3());
            return Math.max(s.x, s.z);
          }),
        );
        const looksGlobal = spread > maxTileDim * 0.6;

        if (!looksGlobal) {
          // Legacy layout fallback (keep previous behavior)
          const zoneInfo = models.map((m) => {
            const box = new THREE.Box3().setFromObject(m.obj);
            return {
              size: box.getSize(new THREE.Vector3()),
              center: box.getCenter(new THREE.Vector3()),
              min: box.min.clone(),
              model: m
            };
          });

          const metaByTaskId = batchZoneMetaByTaskId || {};
          const canUseMapLayout = models.every((m) => {
            const meta = (metaByTaskId as any)[m.id];
            return meta && (meta.row != null || meta.col != null);
          });

          if (canUseMapLayout) {
            const rows = models.map((m) => Number((metaByTaskId as any)[m.id].row ?? 0));
            const cols = models.map((m) => Number((metaByTaskId as any)[m.id].col ?? 0));
            const minRow = Math.min(...rows);
            const minCol = Math.min(...cols);

            const maxW = Math.max(...zoneInfo.map((z) => z.size.x));
            const maxD = Math.max(...zoneInfo.map((z) => z.size.z));
            const stepX = maxW * 1.0;
            const stepZ = maxD * 1.0;

            zoneInfo.forEach((item) => {
              const meta = (metaByTaskId as any)[item.model.id] || {};
              const r = Number(meta.row ?? 0) - minRow;
              const c = Number(meta.col ?? 0) - minCol;
              const xShift = (r % 2) ? stepX * 0.5 : 0.0;

              item.model.obj.position.x = c * stepX + xShift - item.center.x;
              item.model.obj.position.z = r * stepZ - item.center.z;
              item.model.obj.position.y = -item.min.y;
              item.model.obj.updateMatrixWorld(true);
            });
          } else {
            // Fallback: Simple grid layout based on index if no row/col meta
            console.warn("Batch preview: No row/col metadata found, using fallback grid layout");
            const count = zoneInfo.length;
            const cols = Math.ceil(Math.sqrt(count));
            const maxW = Math.max(...zoneInfo.map((z) => z.size.x));
            const maxD = Math.max(...zoneInfo.map((z) => z.size.z));
            const padding = 20; // mm

            zoneInfo.forEach((item, index) => {
              const r = Math.floor(index / cols);
              const c = index % cols;

              item.model.obj.position.x = c * (maxW + padding) - item.center.x;
              item.model.obj.position.z = r * (maxD + padding) - item.center.z;
              item.model.obj.position.y = -item.min.y;
              item.model.obj.updateMatrixWorld(true);
            });
          }
        }

        // 5. Центруємо всю групу
        const groupBox = new THREE.Box3().setFromObject(group);
        const gCenter = groupBox.getCenter(new THREE.Vector3());
        const gMin = groupBox.min.clone();
        group.position.x -= gCenter.x;
        group.position.z -= gCenter.z;
        group.position.y -= gMin.y;
        group.updateMatrixWorld(true);

        // Додаємо легкі візуальні індикатори для кожної зони (опціонально)
        // Для продуктивності не додаємо складні об'єкти, але зберігаємо інформацію

        (group as any).userData = { batch: true, ids: idsToLoad, zoneCount: idsToLoad.length };
        setModel(group);
      } catch (e: any) {
        setError(e?.message || "Помилка завантаження batch превʼю");
      } finally {
        setLoading(false);
      }
    };

    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showAllZones, taskIds.join(","), JSON.stringify(taskStatuses)]);

  useEffect(() => {
    if (showAllZones) return;
    // ВАЖЛИВО: Не скидаємо модель, якщо вона вже завантажена
    // Модель може зникнути, якщо downloadUrl тимчасово стає null під час оновлення стану
    if (!downloadUrl || !activeTaskId) {
      // Не скидаємо модель, якщо вже завантажена тестова або інша модель
      // Це запобігає зникненню моделі під час оновлення стану
      if (!hasLoadedTestModel && !model) {
        // Тільки якщо немає ні тестової моделі, ні завантаженої моделі
        setModel(null);
        setError(null);
      }
      return;
    }

    // Якщо модель вже завантажена для цього taskId, не перезавантажуємо
    const currentTaskId = (model as any)?.userData?.taskId;
    if (model && currentTaskId === activeTaskId) {
// removed-debug-log
      return;
    }

    // Якщо завантажуємо нову модель, скидаємо попередню (якщо вона не тестова)
    if (model && !hasLoadedTestModel) {
// removed-debug-log
      setModel(null);
    }

    const loadModel = async () => {
      setLoading(true);
      setError(null);
      try {
// removed-debug-log
        const loadedModel = await loadPreviewModelForTask(activeTaskId);

        // Стабільні трансформації для превʼю:
        // - масштабуємо під камеру
        // - центруємо лише X/Z
        // - ставимо на "підлогу" (minY=0)
        loadedModel.position.set(0, 0, 0);
        loadedModel.scale.set(1, 1, 1);
        loadedModel.updateMatrixWorld(true);

        const box = new THREE.Box3().setFromObject(loadedModel);
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);

// removed-debug-log
// removed-debug-log

        if (maxDim === 0) {
          throw new Error("Модель має нульовий розмір");
        }

        const targetSize = maxDim < 0.1 ? 300 : 220;
        const viewScale = targetSize / maxDim;
        loadedModel.scale.set(viewScale, viewScale, viewScale);
        loadedModel.updateMatrixWorld(true);

        const boxAfter = new THREE.Box3().setFromObject(loadedModel);
        const center = boxAfter.getCenter(new THREE.Vector3());
        const min = boxAfter.min.clone();

        loadedModel.position.x -= center.x;
        loadedModel.position.z -= center.z;
        loadedModel.position.y -= min.y;
        loadedModel.updateMatrixWorld(true);

        // Перевіряємо розміри після обробки
        const boxFinal = new THREE.Box3().setFromObject(loadedModel);
        const sizeAfter = boxFinal.getSize(new THREE.Vector3());
        const centerAfter = boxFinal.getCenter(new THREE.Vector3());
// removed-debug-log
// removed-debug-log
// removed-debug-log

        // Зберігаємо інформацію про модель для налаштування камери
        (loadedModel as any).userData = {
          size: sizeAfter,
          center: centerAfter,
          maxDim: Math.max(sizeAfter.x, sizeAfter.y, sizeAfter.z),
          taskId: activeTaskId,  // Зберігаємо taskId, щоб не перезавантажувати
          exportFormat: exportFormat
        };

// removed-debug-log
        fitCameraToObject(loadedModel);
        setModel(loadedModel);
        setLoading(false);
// removed-debug-log
      } catch (error: any) {
        console.error("Помилка завантаження моделі:", error);
        setError(error.message || "Помилка завантаження моделі");
        setLoading(false);
      }
    };

    loadModel();
  }, [downloadUrl, activeTaskId, exportFormat, showAllZones]);

  // Terrain shading toggle: seam lines on slopes are often just normal discontinuity between separate tiles.
  useEffect(() => {
    if (!model) return;
    (model as any).traverse?.((child: any) => {
      if (!(child instanceof THREE.Mesh)) return;
      if (child.userData?.part !== "base") return;
      const mat = child.material as THREE.MeshStandardMaterial | undefined;
      if (!mat) return;
      // smooth shading = vertex normals; can show a seam line between separate meshes
      mat.flatShading = !terrainSmoothShading;
      mat.needsUpdate = true;
      try {
        (child.geometry as THREE.BufferGeometry | undefined)?.computeVertexNormals();
      } catch {
        // ignore
      }
    });
  }, [model, terrainSmoothShading]);

  // Component visibility toggle: показуємо/приховуємо компоненти в реальному часі
  useEffect(() => {
    if (!model) return;
    const visibilityMap: Record<string, boolean> = {
      base: previewIncludeBase,
      terrain: previewIncludeBase,
      roads: previewIncludeRoads,
      buildings: previewIncludeBuildings,
      water: previewIncludeWater,
      parks: previewIncludeParks,
      green: previewIncludeParks,
    };
    
    (model as any).traverse?.((child: any) => {
      if (!(child instanceof THREE.Mesh)) return;
      const part = child.userData?.part;
      if (part) {
        const shouldBeVisible = visibilityMap[part] ?? true;
        child.visible = shouldBeVisible;
      }
    });
  }, [model, previewIncludeBase, previewIncludeRoads, previewIncludeBuildings, previewIncludeWater, previewIncludeParks]);

  // Component visibility toggle: показуємо/приховуємо компоненти в реальному часі
  useEffect(() => {
    if (!model) return;
    const visibilityMap: Record<string, boolean> = {
      base: previewIncludeBase,
      terrain: previewIncludeBase,
      roads: previewIncludeRoads,
      buildings: previewIncludeBuildings,
      water: previewIncludeWater,
      parks: previewIncludeParks,
      green: previewIncludeParks,
    };
    
    (model as any).traverse?.((child: any) => {
      if (!(child instanceof THREE.Mesh)) return;
      const part = child.userData?.part;
      if (part) {
        const shouldBeVisible = visibilityMap[part] ?? true;
        child.visible = shouldBeVisible;
      }
    });
  }, [model, previewIncludeBase, previewIncludeRoads, previewIncludeBuildings, previewIncludeWater, previewIncludeParks]);

  if (loading) {
    return (
      <>
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} />
        <gridHelper args={[100, 100]} />
        <axesHelper args={[50]} />
        <mesh>
          <boxGeometry args={[10, 10, 10]} />
          <meshStandardMaterial color="orange" />
        </mesh>
      </>
    );
  }

  if (error) {
    console.error("Помилка в ModelLoader:", error);
    return (
      <>
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} />
        <gridHelper args={[100, 100]} />
        <axesHelper args={[50]} />
        <mesh>
          <boxGeometry args={[10, 10, 10]} />
          <meshStandardMaterial color="red" />
        </mesh>
      </>
    );
  }

  if (!model) {
// removed-debug-log
    return (
      <>
        <ambientLight intensity={0.8} />
        <directionalLight position={[100, 100, 100]} intensity={1.0} />
        <directionalLight position={[-100, -100, -100]} intensity={0.5} />
        <gridHelper args={[200, 20]} />
        <axesHelper args={[100]} />
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[20, 20, 20]} />
          <meshStandardMaterial color="orange" />
        </mesh>
      </>
    );
  }

// removed-debug-log
// removed-debug-log

  // Перевіряємо, чи модель має геометрію
  let hasGeometry = false;
  let vertexCount = 0;
  if (model instanceof THREE.Group) {
    model.traverse((child) => {
      if (child instanceof THREE.Mesh && child.geometry) {
        hasGeometry = true;
        if (child.geometry.attributes.position) {
          vertexCount += child.geometry.attributes.position.count;
        }
      }
    });
  } else if (model instanceof THREE.Mesh && model.geometry) {
    hasGeometry = true;
    if (model.geometry.attributes.position) {
      vertexCount = model.geometry.attributes.position.count;
    }
  }

// removed-debug-log

  if (!hasGeometry || vertexCount === 0) {
    console.warn("⚠️ Модель не містить геометрії або має 0 вершин!");
    // Не повертаємо null, щоб не зникнути - показуємо placeholder
    return (
      <>
        <ambientLight intensity={0.8} />
        <directionalLight position={[100, 100, 100]} intensity={1.0} />
        <directionalLight position={[-100, -100, -100]} intensity={0.5} />
        <gridHelper args={[200, 20]} />
        <axesHelper args={[100]} />
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[20, 20, 20]} />
          <meshStandardMaterial color="red" />
        </mesh>
      </>
    );
  }

  // В режимі rotateMode="model" — drag миші крутить модель (а не камеру)
  const onPointerDown = (e: any) => {
    if (rotateMode !== "model") return;
    e.stopPropagation();
    dragRef.current.dragging = true;
    dragRef.current.x = e.clientX;
    dragRef.current.y = e.clientY;
  };

  const onPointerUp = (e: any) => {
    if (rotateMode !== "model") return;
    e.stopPropagation();
    dragRef.current.dragging = false;
  };

  const onPointerMove = (e: any) => {
    if (rotateMode !== "model") return;
    if (!dragRef.current.dragging) return;
    e.stopPropagation();
    const dx = e.clientX - dragRef.current.x;
    const dy = e.clientY - dragRef.current.y;
    dragRef.current.x = e.clientX;
    dragRef.current.y = e.clientY;

    const group = modelGroupRef.current;
    if (!group) return;

    // Чутливість
    const speed = 0.01;
    group.rotation.y += dx * speed;
    group.rotation.x += dy * speed;
  };

  return (
    <>
      <ambientLight intensity={0.55} />
      <hemisphereLight args={[0xffffff, 0x2b2b2b, 0.65]} />
      <directionalLight position={[200, 250, 150]} intensity={1.0} />
      <directionalLight position={[-200, -150, -100]} intensity={0.35} />
      {/* Обгортаємо модель у Group, щоб можна було крутити саме модель */}
      <group
        ref={modelGroupRef}
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
        onPointerMove={onPointerMove}
        onDoubleClick={(e) => {
          if (rotateMode !== "model") return;
          e.stopPropagation();
          resetModelRotation();
        }}
      >
        <primitive object={model} />
      </group>
    </>
  );
}

export function Preview3D({ capture = false }: { capture?: boolean } = {}) {
  const {
    downloadUrl,
    isGenerating,
    progress,
    terrainSmoothShading,
    setTerrainSmoothShading,
    taskStatuses,
    activeTaskId,
    previewIncludeBase,
    previewIncludeRoads,
    previewIncludeBuildings,
    previewIncludeWater,
    previewIncludeParks,
    setPreviewIncludeBase,
    setPreviewIncludeRoads,
    setPreviewIncludeBuildings,
    setPreviewIncludeWater,
    setPreviewIncludeParks,
  } = useGenerationStore();
  const [gridVisible, setGridVisible] = useState(false);
  const [axesVisible, setAxesVisible] = useState(false);
  const [rotateMode, setRotateMode] = useState<RotateMode>("camera");
  const [cameraMode, setCameraMode] = useState<CameraMode>("orbit");
  const [flySpeed, setFlySpeed] = useState<number>(120);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [isFs, setIsFs] = useState(false);
  const [toolsOpen, setToolsOpen] = useState(false);

  useEffect(() => {
    const onFs = () => setIsFs(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onFs);
    return () => document.removeEventListener("fullscreenchange", onFs);
  }, []);
  const toggleFullscreen = () => {
    const el = containerRef.current;
    if (!el) return;
    if (document.fullscreenElement) {
      document.exitFullscreen?.();
    } else {
      (el.requestFullscreen?.() as Promise<void> | undefined)?.catch(() => {});
    }
  };

  return (
    <div ref={containerRef} className="relative h-full w-full bg-slate-950" style={{ minHeight: "100%" }}>
      {/* Компактна панель: на весь екран + (опційно) інструменти. За замовчуванням
          інструменти приховані — щоб було видно саму модель. */}
      {!capture && (
        <div className="absolute right-3 top-3 z-30 flex items-center gap-2">
          <button
            type="button"
            onClick={toggleFullscreen}
            className="flex h-10 items-center gap-1.5 rounded-full border border-white/15 bg-[rgba(2,6,23,0.7)] px-3 text-[12px] font-semibold text-white backdrop-blur transition hover:bg-[rgba(2,6,23,0.9)]"
            title="На весь екран"
          >
            {isFs ? "✕ Згорнути" : "⤢ На весь екран"}
          </button>
          <button
            type="button"
            onClick={() => setToolsOpen((v) => !v)}
            className={`flex h-10 w-10 items-center justify-center rounded-full border border-white/15 backdrop-blur transition ${toolsOpen ? "bg-white text-[#0b1020]" : "bg-[rgba(2,6,23,0.7)] text-white hover:bg-[rgba(2,6,23,0.9)]"}`}
            title="Інструменти перегляду"
          >
            ⚙
          </button>
        </div>
      )}
      {!capture && toolsOpen && <div className="pointer-events-none absolute inset-x-3 top-16 z-20 flex justify-end">
        <div className="pointer-events-auto w-full max-w-[320px] rounded-[24px] border border-white/10 bg-[rgba(2,6,23,0.74)] px-3 py-3 text-white shadow-[0_20px_55px_rgba(2,6,23,0.45)] backdrop-blur">
          <div className="mb-3 flex items-start justify-between gap-3">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-white/55">
                Preview Tools
              </div>
              <div className="mt-1 text-sm font-semibold">Швидке керування сценою</div>
            </div>
            <div className="rounded-full bg-white/10 px-3 py-1 text-[11px] font-semibold text-white/70">
              {cameraMode === "fly" ? "Fly" : "Orbit"}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <button
              className="rounded-2xl bg-white/10 px-3 py-2 text-left text-xs font-medium transition hover:bg-white/20"
              onClick={() => setGridVisible((v) => !v)}
            >
              <span className="block text-[10px] uppercase tracking-[0.18em] text-white/45">Сітка</span>
              <span className="mt-1 block">{gridVisible ? "Увімкнена" : "Прихована"}</span>
            </button>
            <button
              className="rounded-2xl bg-white/10 px-3 py-2 text-left text-xs font-medium transition hover:bg-white/20"
              onClick={() => setAxesVisible((v) => !v)}
            >
              <span className="block text-[10px] uppercase tracking-[0.18em] text-white/45">Осі</span>
              <span className="mt-1 block">{axesVisible ? "Увімкнені" : "Приховані"}</span>
            </button>
            <button
              className="rounded-2xl bg-white/10 px-3 py-2 text-left text-xs font-medium transition hover:bg-white/20"
              onClick={() => setRotateMode((m) => (m === "camera" ? "model" : "camera"))}
              title="Camera: крутиться камера. Model: drag обертає саму модель."
            >
              <span className="block text-[10px] uppercase tracking-[0.18em] text-white/45">Обертання</span>
              <span className="mt-1 block">{rotateMode === "camera" ? "Камера" : "Модель"}</span>
            </button>
            <button
              className="rounded-2xl bg-white/10 px-3 py-2 text-left text-xs font-medium transition hover:bg-white/20"
              onClick={() => setCameraMode((m) => (m === "orbit" ? "fly" : "orbit"))}
              title="Orbit: стандартний огляд. Fly: вільний рух WASD + mouse."
            >
              <span className="block text-[10px] uppercase tracking-[0.18em] text-white/45">Камера</span>
              <span className="mt-1 block">{cameraMode === "orbit" ? "Orbit" : "Fly"}</span>
            </button>
          </div>

          {cameraMode === "fly" && (
            <div className="mt-3 rounded-[18px] bg-white/8 px-3 py-3">
              <div className="flex items-center justify-between gap-3 text-xs">
                <span className="font-medium">Швидкість польоту</span>
                <span className="tabular-nums text-white/70">{Math.round(flySpeed)}</span>
              </div>
              <input
                className="mt-2 w-full"
                type="range"
                min={10}
                max={800}
                step={5}
                value={flySpeed}
                onChange={(e) => setFlySpeed(Number(e.target.value))}
              />
            </div>
          )}

          <div className="mt-3 flex items-center justify-between gap-3 rounded-[18px] bg-white/8 px-3 py-3 text-xs">
            <div>
              <div className="font-medium">Terrain shading</div>
              <div className="mt-1 text-white/60">
                {terrainSmoothShading ? "Плавне затінення схилів" : "Більш чіткі грані рельєфу"}
              </div>
            </div>
            <button
              className="rounded-full bg-white/10 px-3 py-2 font-semibold transition hover:bg-white/20"
              onClick={() => setTerrainSmoothShading(!terrainSmoothShading)}
              title="Smooth = плавні нормалі. Flat = чіткіший рельєф без помітних швів."
            >
              {terrainSmoothShading ? "Smooth" : "Flat"}
            </button>
          </div>

          <div className="mt-3 text-[10px] leading-4 text-white/55">
            {cameraMode === "fly"
              ? "Fly: WASD рух, Q/E вгору-вниз, Shift прискорення, права кнопка миші для огляду."
              : rotateMode === "model"
                ? "Drag по моделі обертає її, подвійний клік скидає орієнтацію."
                : "Drag керує камерою, wheel змінює дистанцію."}
          </div>

          <div className="hidden">
            <div className="flex items-center justify-between gap-3">
            <span>Grid</span>
            <button
              className="px-2 py-1 rounded bg-white/10 hover:bg-white/20"
              onClick={() => setGridVisible((v) => !v)}
            >
              {gridVisible ? "Hide" : "Show"}
            </button>
          </div>
          <div className="flex items-center justify-between gap-3">
            <span>Axes</span>
            <button
              className="px-2 py-1 rounded bg-white/10 hover:bg-white/20"
              onClick={() => setAxesVisible((v) => !v)}
            >
              {axesVisible ? "Hide" : "Show"}
            </button>
          </div>
          <div className="flex items-center justify-between gap-3">
            <span>Rotate</span>
            <button
              className="px-2 py-1 rounded bg-white/10 hover:bg-white/20"
              onClick={() => setRotateMode((m) => (m === "camera" ? "model" : "camera"))}
              title="Camera: крутиться камера (OrbitControls). Model: крутиться сама модель (drag по моделі)."
            >
              {rotateMode === "camera" ? "Camera" : "Model"}
            </button>
          </div>
          <div className="flex items-center justify-between gap-3">
            <span>Camera</span>
            <button
              className="px-2 py-1 rounded bg-white/10 hover:bg-white/20"
              onClick={() => setCameraMode((m) => (m === "orbit" ? "fly" : "orbit"))}
              title="Orbit: стандартний огляд. Fly: вільний політ (WASD, Q/E, Shift, RMB+mouse)."
            >
              {cameraMode === "orbit" ? "Orbit" : "Fly"}
            </button>
          </div>
          <div className="text-[10px] text-white/70">
            {cameraMode === "fly"
              ? "Fly: WASD рух, Q/E вгору/вниз, Shift швидше, RMB+mouse дивитись, wheel = speed."
              : (rotateMode === "model" ? "Drag по моделі = rotate. Double-click = reset." : "Drag = rotate camera.")}
          </div>
          {cameraMode === "fly" && (
            <div className="pt-1">
              <div className="flex items-center justify-between gap-3">
                <span>Fly speed</span>
                <span className="text-[10px] text-white/70 tabular-nums">{Math.round(flySpeed)}</span>
              </div>
              <input
                className="w-full"
                type="range"
                min={10}
                max={800}
                step={5}
                value={flySpeed}
                onChange={(e) => setFlySpeed(Number(e.target.value))}
              />
            </div>
          )}
          <div className="flex items-center justify-between gap-3 pt-1">
            <span>Terrain</span>
            <button
              className="px-2 py-1 rounded bg-white/10 hover:bg-white/20"
              onClick={() => setTerrainSmoothShading(!terrainSmoothShading)}
              title="Smooth = плавні нормалі (може бути видимий шов між тайлами). Flat = шов майже не видно (але видно грані)."
            >
              {terrainSmoothShading ? "Smooth" : "Flat"}
            </button>
          </div>
          <div className="border-t border-white/20 pt-2 mt-2 space-y-1">
            <div className="text-xs font-semibold text-white/90 mb-1">Видимість компонентів:</div>
            <label className="flex items-center justify-between gap-3 cursor-pointer">
              <span className="text-xs">Рельєф</span>
              <input
                type="checkbox"
                checked={previewIncludeBase}
                onChange={(e) => setPreviewIncludeBase(e.target.checked)}
                className="w-4 h-4"
              />
            </label>
            <label className="flex items-center justify-between gap-3 cursor-pointer">
              <span className="text-xs">Дороги</span>
              <input
                type="checkbox"
                checked={previewIncludeRoads}
                onChange={(e) => setPreviewIncludeRoads(e.target.checked)}
                className="w-4 h-4"
              />
            </label>
            <label className="flex items-center justify-between gap-3 cursor-pointer">
              <span className="text-xs">Будівлі</span>
              <input
                type="checkbox"
                checked={previewIncludeBuildings}
                onChange={(e) => setPreviewIncludeBuildings(e.target.checked)}
                className="w-4 h-4"
              />
            </label>
            <label className="flex items-center justify-between gap-3 cursor-pointer">
              <span className="text-xs">Вода</span>
              <input
                type="checkbox"
                checked={previewIncludeWater}
                onChange={(e) => setPreviewIncludeWater(e.target.checked)}
                className="w-4 h-4"
              />
            </label>
            <label className="flex items-center justify-between gap-3 cursor-pointer">
              <span className="text-xs">Парки</span>
              <input
                type="checkbox"
                checked={previewIncludeParks}
                onChange={(e) => setPreviewIncludeParks(e.target.checked)}
                className="w-4 h-4"
              />
            </label>
          </div>
        </div>
      </div>
      </div>}
      {isGenerating && !capture && (
        <div className="absolute inset-0 flex items-center justify-center text-white z-10 pointer-events-none">
          <div className="text-center">
            <p className="text-lg mb-2">Генерація моделі...</p>
            <p className="text-sm text-gray-400">{progress}%</p>
          </div>
        </div>
      )}
      <Canvas style={{ width: '100%', height: '100%', display: 'block' }}>
        <Suspense fallback={null}>
          <CameraController />
          <FreeFlyControls enabled={cameraMode === "fly"} speed={flySpeed} onSpeedChange={setFlySpeed} />
          <OrbitControls
            makeDefault
            enabled={cameraMode === "orbit"}
            enableDamping
            dampingFactor={0.05}
            minDistance={10}
            maxDistance={2000}
            target={[0, 0, 0]}
            autoRotate={false}
            enableRotate={rotateMode === "camera"}
          />
          {gridVisible && <gridHelper args={[200, 20]} />}
          {axesVisible && <axesHelper args={[100]} />}
          <ModelLoader rotateMode={rotateMode} />
        </Suspense>
      </Canvas>
    </div>
  );
}
