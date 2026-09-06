"use client";

import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera, Html } from "@react-three/drei";
import { Component, Suspense, useEffect, useMemo, useState, useRef } from "react";
import type { ErrorInfo, ReactNode } from "react";
import { useTranslations } from "next-intl";
import { useGenerationStore } from "@/store/generation-store";
import { useShallow } from "zustand/react/shallow";
import { api } from "@/lib/api";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { ThreeMFLoader } from "three/examples/jsm/loaders/3MFLoader.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";
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
    base: 0xf2f2f2,
    terrain: 0xf2f2f2,
    roads: 0x141414,
    buildings: 0xc4c4c4,
    water: 0x2f6fd6,
    parks: 0x5c965c,
    green: 0x5c965c,
    highlight: 0xce2626, // виділений будинок — ЧЕРВОНИЙ (як друк)
    frame: 0x191919,     // преміум-рамка — ЧОРНА
    connector: 0xf2f2f2, // з'єднувач — колір основи
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
        // Превʼю-палітра ≈ ДРУК (backend COLOR_MAP), щоб «що бачиш = що друкується»
        // і кожен шар ЧІТКО відрізнявся (скарга: на білій основі будинки зливались).
        // Палітра за фідбеком власника (2026-06-17): основа+рельєф = БІЛІ, вода СИНЯ,
        // зелені зони ТЕМНІШИЙ зелений, дороги ЧОРНІ. ЄДИНЕ відхилення превʼю↔друк:
        // будинки у ПРЕВʼЮ світло-СІРІ 0xc4c4c4 (щоб ВИДНО на білій основі — інакше
        // білий-на-білому невидимий), а в ДРУЦІ лишаються білі (COLOR_MAP).
        const colorMap: Record<string, number> = {
          baseback: 0xf2f2f2,
          base: 0xf2f2f2,    // основа/рельєф — БІЛА
          terrain: 0xf2f2f2,
          buildings: 0xc4c4c4, // будинки — світло-СІРІ у ПРЕВʼЮ (видно на білій основі; друк = білі)
          roads: 0x141414,   // дороги — ЧОРНІ
          water: 0x2f6fd6,   // вода — СИНЯ
          parks: 0x5c965c,   // парки/зелень — темніший зелений
          green: 0x5c965c,
          poi: 0xf0a030,
          track: 0xdc2626, // GPX-маршрут — ЧЕРВОНИЙ, чітко виділяється на превʼю
          marker: 0xc44110, // маркер «особливе місце» — теракотовий
          highlight: 0xce2626, // виділений будинок — ЧЕРВОНИЙ (як друк)
          landmark: 0xd69e42, // визначні місця (церкви/вежі/історичні) — БРОНЗА (друк #C9902F)
          frame: 0x191919,  // преміум-рамка — ЧОРНА
          connector: 0xf2f2f2, // з'єднувач-метелик — колір основи
          maplabel: 0x191919,
          rim: 0x191919,   // ободок брелка — ЧОРНИЙ (друкується чорним)
          text: 0x191919,  // текст/назва — ЧОРНИЙ
          text2: 0x191919,
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
            const maybeColored = material as THREE.Material & { color?: THREE.Color; map?: unknown; metalness?: number; roughness?: number };
            // ФОРСУЄМО кольори ВСІХ розпізнаних шарів (не лише track/rim/text), щоб
            // превʼю було ЧІТКИМ — «що бачиш = що друкується». Раніше решта шарів
            // фарбувалась лише коли матеріал білий → на рельєфі будинки приходили з
            // земляним матеріалом (не білим) і зливались із бежевою основою
            // («не зрозуміло що і як»). Тепер кожен шар має свій чіткий колір.
            if (partColor !== null && maybeColored.color) {
              maybeColored.color.setHex(partColor);
              // Прибираємо текстуру/металік, що могли б перебити суцільний колір.
              if ("map" in maybeColored) (maybeColored as any).map = null;
              if (typeof maybeColored.metalness === "number") maybeColored.metalness = 0.0;
              if (typeof maybeColored.roughness === "number") maybeColored.roughness = 0.85;
            }
            // FLAT SHADING на ВСІХ шарах: ThreeMFLoader дефолтно дає smooth-нормалі,
            // які на плоскій верхній грані ЗМІШУЮТЬ нормаль кепа зі стінкою на
            // спільних граничних вершинах → «віяло»-градієнт на парках/дорогах, ніби
            // меш кривий і йде з однієї точки (скарга власника). Per-face нормалі =
            // кожен трикутник рівний → плоский верх однотонний, як друк.
            if ("flatShading" in maybeColored) (maybeColored as any).flatShading = true;
            material.side = THREE.DoubleSide;
            material.needsUpdate = true;
          }
        });

        if (totalVertices === 0) {
          reject(new Error("Модель не містить вершин"));
          return;
        }

        // КЛАСти ПЛОСКО, як у житті: 3mf (як STL/GLB) у Z-up просторі trimesh —
        // треба повернути на -90° по X у Y-up three.js, інакше брелок/мапа СТОЇТЬ
        // вертикально (скарга «мапа стоїть, а має лежати»). loadGLB і STL це вже
        // роблять; load3MF цей поворот пропускав.
        group.rotation.x = -Math.PI / 2;
        group.updateMatrixWorld(true);
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

// Дзеркало backend COLOR_PALETTES (model_exporter.py) — щоб ПРЕВʼЮ = ДРУК для
// кольорових тем (#2). classic = поточна превʼю-палітра (нижче), решта = оверрайди
// за ключем шару. Часткові: відсутні ключі лишають дефолтний превʼю-колір.
const PREVIEW_PALETTES: Record<string, Record<string, number>> = {
  sepia: {
    base: 0xe0cea9, terrain: 0xe0cea9, baseback: 0xe0cea9, connector: 0xe0cea9,
    buildings: 0xc4a978, roads: 0x5c3e20, water: 0x96a596, parks: 0x8e8c5c, green: 0x8e8c5c,
    text: 0x462d14, text2: 0x462d14, rim: 0x462d14, maplabel: 0x462d14, frame: 0x462d14,
  },
  noir: {
    base: 0xececec, terrain: 0xececec, baseback: 0xececec, connector: 0xececec,
    buildings: 0xb2b2b2, roads: 0x0f0f0f, water: 0x787878, parks: 0x969696, green: 0x969696,
  },
  ocean: {
    base: 0xecf4fb, terrain: 0xecf4fb, baseback: 0xecf4fb, connector: 0xecf4fb,
    buildings: 0x96b6d4, roads: 0x1c325c, water: 0x145cac, parks: 0x5c987a, green: 0x5c987a,
    text: 0x162a52, text2: 0x162a52, rim: 0x162a52, maplabel: 0x162a52,
  },
  neon: {
    base: 0x1a1a2a, terrain: 0x1a1a2a, baseback: 0x1a1a2a, connector: 0x1a1a2a,
    buildings: 0x3e3e5c, roads: 0xe83ca2, water: 0x28c8de, parks: 0x78e85c, green: 0x78e85c,
    text: 0xf0f0ff, text2: 0xf0f0ff, rim: 0xf0f0ff, maplabel: 0xf0f0ff,
  },
};

async function loadGLB(blob: Blob): Promise<THREE.Group> {
  const url = URL.createObjectURL(blob);
  return await new Promise<THREE.Group>((resolve, reject) => {
    const loader = new GLTFLoader();
    // Preview GLBs from the backend may be meshopt-compressed (gltfpack -cc);
    // decode both compressed and uncompressed GLBs with the same loader.
    if (typeof loader.setMeshoptDecoder === "function") {
      loader.setMeshoptDecoder(MeshoptDecoder);
    }
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
        // ЧІТКА палітра превʼю (flat MeshBasic) — кожен шар УНІКАЛЬНИЙ за яскравістю,
        // щоб «видно що і як». БУЛО: будинки 0xe3e3e3 (світло-сірі) ≈ бежева земля →
        // зливались. ТЕПЕР: земля бежева (світла), будинки СІРІ (темніші, = друк),
        // дороги майже чорні, вода блакитна, парки зелені. baseback/maplabel/poi теж.
        // Палітра за фідбеком власника (2026-06-17): основа+рельєф+будинки = БІЛІ
        // (один колір), вода СИНЯ, зелень ТЕМНІШЕ, дороги ЧОРНІ.
        const colorMap: Record<string, { color: number; part: string }> = {
          baseback: { color: 0xf2f2f2, part: "base" },
          base: { color: 0xf2f2f2, part: "base" },       // основа — БІЛА
          terrain: { color: 0xf2f2f2, part: "terrain" },
          buildings: { color: 0xc4c4c4, part: "buildings" }, // будинки — світло-СІРІ у ПРЕВʼЮ (видно на білій основі; друк = білі)
          roads: { color: 0x141414, part: "roads" },     // дороги — ЧОРНІ
          water: { color: 0x2f6fd6, part: "water" },     // вода — СИНЯ
          parks: { color: 0x5c965c, part: "parks" },     // парки — темніший зелений
          green: { color: 0x5c965c, part: "parks" },
          poi: { color: 0xf0a030, part: "poi" },
          track: { color: 0xdc2626, part: "track" },     // GPX — червоний
          marker: { color: 0xc44110, part: "marker" },   // маркер «особливе місце» — теракотовий
          highlight: { color: 0xce2626, part: "highlight" }, // виділений будинок — ЧЕРВОНИЙ
          landmark: { color: 0xd69e42, part: "landmark" }, // визначні місця — БРОНЗА (друк #C9902F)
          frame: { color: 0x191919, part: "frame" },     // преміум-рамка — ЧОРНА
          connector: { color: 0xf2f2f2, part: "connector" }, // з'єднувач — колір основи
          maplabel: { color: 0x191919, part: "maplabel" },
          rim: { color: 0x191919, part: "rim" },         // ободок — чорний
          text: { color: 0x191919, part: "text" },       // текст — чорний
          text2: { color: 0x191919, part: "text2" },
        };
        // Кольорова тема (#2): якщо обрано не-classic палітру — перефарбовуємо превʼю
        // у ті самі кольори, що бек запікає у 3MF (PREVIEW_PALETTES = дзеркало backend).
        try {
          const pal = useGenerationStore.getState().simpleColorPalette;
          const overrides = pal && pal !== "classic" ? PREVIEW_PALETTES[pal] : null;
          if (overrides) {
            for (const [key, hex] of Object.entries(overrides)) {
              if (colorMap[key]) colorMap[key] = { color: hex, part: colorMap[key].part };
            }
          }
        } catch {
          /* store недоступний — лишаємо classic-превʼю */
        }
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
          // LIT матеріал (було MeshBasicMaterial-unlit) — інакше будинки малювались
          // ПЛОСКИМ кольором без тіней і ЗЛИВАЛИСЬ з білою основою. MeshStandard +
          // flatShading (гострі грані, без smooth-градієнта) → directional-світло
          // затінює БОКИ будинків → видно обʼєм, будинки виділяються на основі.
          child.material = new THREE.MeshStandardMaterial({
            color: entry?.color ?? 0x9a9a9a,
            side: THREE.DoubleSide,
            flatShading: true,
            roughness: 1.0,
            metalness: 0.0,
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

// Чи є меш частиною ЗʼЄДНУВАЧА (ключа-метелика). Перевіряємо НАДІЙНО: userData.part
// (ставить GLB-лоадер), УВЕСЬ ланцюг предків за назвою, і назви матеріалів — бо у
// композиті ключ може бути на рівні node-обгортки (GLTFLoader кладе імʼя на node,
// а не на сам Mesh) → перевірки лише c.name/c.parent.name його пропускали і ключі
// (що лежать ДАЛЕКО від карт) роздували габарит → серія рендерилась дрібною.
function isConnectorMesh(c: any): boolean {
  if (!c || !c.isMesh) return false;
  if (c.userData?.part === "connector") return true;
  let o: any = c;
  while (o) { if (/connector/i.test(String(o.name || ""))) return true; o = o.parent; }
  const mats = Array.isArray(c.material) ? c.material : [c.material];
  if (mats.some((m: any) => /connector/i.test(String(m?.name || "")))) return true;
  return false;
}

// Download a preview blob with retry+backoff. The preview file (GLB/3MF) may still be
// generating when the viewer first asks for it — a non-ready file comes back empty
// (<=100 bytes) or 404s. Retrying a few times fixes the "Помилка завантаження моделі"
// race for both single and multi-model (series) creation.
async function downloadPreviewBlobWithRetry(
  taskId: string,
  format: "glb" | "3mf",
  attempts = 5,
): Promise<Blob> {
  let lastErr: unknown = null;
  for (let i = 0; i < attempts; i++) {
    try {
      const blob = await api.downloadModel(taskId, format);
      if (blob && blob.size > 100) return blob;
      lastErr = new Error(`${format.toUpperCase()} preview порожнє (спроба ${i + 1}/${attempts})`);
    } catch (e) {
      lastErr = e;
    }
    if (i < attempts - 1) {
      // 0.8s, 1.6s, 2.4s, 3.2s — total ~8s of grace while the file finishes.
      await new Promise((r) => setTimeout(r, 800 * (i + 1)));
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr || "download failed"));
}

async function loadPreviewModelForTask(taskId: string): Promise<THREE.Group> {
  // ЧЕРГУЄМО формати glb→3mf у КОЖНОМУ раунді замість «5 GLB-ретраїв, потім
  // 3MF»: flat-пайплайн (брелки/магніти/плоскі мапи) GLB не генерує ВЗАГАЛІ,
  // тож стара схема тримала користувача ~10–15с на чорному екрані, поки
  // марні GLB-спроби вигорали. Тепер flat-превʼю вантажиться з другої спроби
  // (~1–2с), а для обʼємних мап GLB так само перший — нічого не змінилось.
  // Раунди з бек-офом лишаються: файл може ще дописуватись у мить success.
  const errors: string[] = [];
  for (let round = 0; round < 3; round++) {
    if (round > 0) {
      // 0.8s, 1.6s — грейс, поки файл закінчує писатись (та сама сума, що була).
      await new Promise((r) => setTimeout(r, 800 * round));
    }
    for (const format of ["glb", "3mf"] as const) {
      try {
        const blob = await downloadPreviewBlobWithRetry(taskId, format, 1);
        try {
          return format === "glb" ? await loadGLB(blob) : await load3MF(blob);
        } catch (parseError) {
          // Бек інколи віддає 3MF-контент на glb-запит — пробуємо як 3MF.
          const type = String(blob.type || "").toLowerCase();
          if (format === "glb" && type.includes("3mf")) return await load3MF(blob);
          throw parseError;
        }
      } catch (e) {
        errors.push(`${format} (раунд ${round + 1}): ${e instanceof Error ? e.message : String(e)}`);
      }
    }
  }
  throw new Error(`Не вдалося завантажити preview після кількох спроб: ${errors.slice(-2).join("; ")}`);
}

// Компонент для автоматичного позиціювання камери
function CameraController() {
  const { downloadUrl, showAllZones, taskIds } = useGenerationStore(useShallow((st) => ({
    downloadUrl: st.downloadUrl, showAllZones: st.showAllZones, taskIds: st.taskIds,
  })));
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);

  useEffect(() => {
    // Налаштовуємо камеру для кращого перегляду
    const timer = setTimeout(() => {
      if (cameraRef.current) {
        // СЕРІЯ (showAllZones): НЕ виставляємо фіксовану відстань — нею раніше камера
        // зависала далеко (хардкод 300×) і композит виглядав дрібним, бо це
        // перебивало точний fitCameraToObject(group) (який кадрує за РЕАЛЬНИМ
        // габаритом карт). Лишаємо камеру на відкуп batch-fit (фолбек — стартова
        // позиція PerspectiveCamera). Для ОДНІЄЇ зони лишаємо стандартну відстань.
        if (!(showAllZones && taskIds && taskIds.length > 1)) {
          const distance = 300;
          cameraRef.current.position.set(distance, distance, distance);
          cameraRef.current.lookAt(0, 0, 0);
          cameraRef.current.updateProjectionMatrix();
        }
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

function ModelLoader({ rotateMode, onError }: { rotateMode: RotateMode; onError?: (msg: string | null) => void }) {
  const t = useTranslations("preview");
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
    gridType,
    terrainSmoothShading,
    previewIncludeBase,
    previewIncludeRoads,
    previewIncludeBuildings,
    previewIncludeWater,
    previewIncludeParks,
  } = useGenerationStore(useShallow((st) => ({
    downloadUrl: st.downloadUrl,
    activeTaskId: st.activeTaskId,
    exportFormat: st.exportFormat,
    showAllZones: st.showAllZones,
    taskIds: st.taskIds,
    taskStatuses: st.taskStatuses,
    batchZoneMetaByTaskId: st.batchZoneMetaByTaskId,
    gridType: st.gridType,
    terrainSmoothShading: st.terrainSmoothShading,
    previewIncludeBase: st.previewIncludeBase,
    previewIncludeRoads: st.previewIncludeRoads,
    previewIncludeBuildings: st.previewIncludeBuildings,
    previewIncludeWater: st.previewIncludeWater,
    previewIncludeParks: st.previewIncludeParks,
  })));
  const [model, setModel] = useState<THREE.Group | THREE.Mesh | null>(null);
  // Звільняємо GPU-пам'ять попередньої моделі при заміні/розмонтуванні: кожне прев'ю
  // вантажить нову сцену, а стара лишала geometry/material/texture у VRAM → WebGL-витік
  // і падіння канви («context lost») на слабких пристроях після багатьох ітерацій.
  useEffect(() => {
    return () => {
      const old = model as any;
      old?.traverse?.((o: any) => {
        if (o.isMesh) {
          o.geometry?.dispose?.();
          (Array.isArray(o.material) ? o.material : [o.material]).forEach((mat: any) => {
            mat?.map?.dispose?.();
            mat?.dispose?.();
          });
        }
      });
    };
  }, [model]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasLoadedTestModel, setHasLoadedTestModel] = useState(false);

  // PERF: батч-прев'ю має перезавантажувати тайли ЛИШЕ коли змінився набір
  // ЗАВЕРШЕНИХ зон. Раніше ефект залежав від JSON.stringify(taskStatuses), тож
  // кожен progress-тік під час полінгу повторно завантажував усі GLB/3MF блоби.
  // Стабільний підпис (відсортовані completed-id) усуває ці зайві ре-фетчі.
  const completedSignature = useMemo(() => {
    if (!showAllZones || !taskIds || taskIds.length < 2) return "";
    return taskIds
      .filter((id) => (taskStatuses as any)?.[id]?.status === "completed")
      .sort()
      .join(",");
  }, [showAllZones, taskIds, taskStatuses]);

  // Surface the load error to the HTML layer (Preview3D) — components inside the
  // R3F <Canvas> can only render three.js objects, so a visible message has to
  // live outside the canvas. Previously a failed load showed only empty lights.
  useEffect(() => {
    onError?.(error);
  }, [error, onError]);

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
    // Кадруємо за КАРТАМИ (Base/Roads/…), БЕЗ шару Connector: ключі-метелики
    // лежать ОКРЕМО під/збоку плиток і роздували габарит → модель виглядала
    // дрібною. Фокус на самих плитках = модель велика й чітка.
    const box = new THREE.Box3();
    let anyMap = false;
    object.traverse((c: any) => {
      if (c.isMesh && !isConnectorMesh(c)) {
        box.expandByObject(c);
        anyMap = true;
      }
    });
    if (!anyMap || box.isEmpty()) box.setFromObject(object);
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
          // ЗʼЄДНУВАЧ-КЛЮЧ — окрема «метелик»-пластина, що ЛЕЖИТЬ ПОЗА контуром плитки
          // (бек кладе її збоку/знизу для друку). У СКЛАДЕНОМУ превʼю ці пластини
          // висять біля швів як «обривчасті стінки/зуби». Ховаємо їх. НАДІЙНО,
          // НЕЗАЛЕЖНО ВІД ІМЕНІ: 3MF-лоадер (превʼю серії вантажить .3mf, бо GLB для
          // плитки нема) втрачає теги шарів → isConnectorMesh за назвою НЕ ловить.
          // Тому ловимо ГЕОМЕТРИЧНО: ключ лежить ПОЗА footprint основної (найбільшої
          // за XZ-площею) деталі плитки. Ховаємо + тегаємо part="connector", щоб
          // mapBoxOf/масштаб/камера теж його виключали.
          try {
            // 1) Знаходимо ОСНОВУ = меш з найбільшим bbox-обсягом (база/рельєф плитки).
            const meshList: any[] = [];
            m.obj.traverse((c: any) => { if (c.isMesh) meshList.push(c); });
            let baseBox: THREE.Box3 | null = null;
            let baseVol = -1;
            for (const c of meshList) {
              const b = new THREE.Box3().setFromObject(c);
              const s = b.getSize(new THREE.Vector3());
              const vol = Math.max(s.x, 0.001) * Math.max(s.y, 0.001) * Math.max(s.z, 0.001);
              if (vol > baseVol) { baseVol = vol; baseBox = b; }
            }
            if (baseBox) {
              const bb: THREE.Box3 = baseBox;
              // 2) ПЛОЩИНА-FOOTPRINT = дві НАЙБІЛЬШІ осі основи (третя = висота). Так
              // детект НЕ залежить від орієнтації (Y-up чи Z-up після лоадера/повороту).
              const bs = bb.getSize(new THREE.Vector3());
              const dims: Array<[string, number]> = [["x", bs.x], ["y", bs.y], ["z", bs.z]];
              dims.sort((a, b) => b[1] - a[1]);
              const a1 = dims[0][0] as "x" | "y" | "z";
              const a2 = dims[1][0] as "x" | "y" | "z";
              const ov1 = (b: THREE.Box3) => Math.max(0, Math.min((bb.max as any)[a1], (b.max as any)[a1]) - Math.max((bb.min as any)[a1], (b.min as any)[a1]));
              const ov2 = (b: THREE.Box3) => Math.max(0, Math.min((bb.max as any)[a2], (b.max as any)[a2]) - Math.max((bb.min as any)[a2], (b.min as any)[a2]));
              for (const c of meshList) {
                const b = new THREE.Box3().setFromObject(c);
                const s = b.getSize(new THREE.Vector3());
                const area = Math.max(1e-6, ((s as any)[a1]) * ((s as any)[a2]));
                const overlap = (ov1(b) * ov2(b)) / area;
                const verts = c.geometry?.attributes?.position?.count ?? 9999;
                // Ключ лежить ПОЗА footprint основи (overlap≈0) І/АБО це крихітний меш
                // (метелик ~12-24 вершин проти тисяч у карти). Будь-який сигнал → ховаємо.
                if (overlap < 0.5 || verts < 64) {
                  c.visible = false;
                  c.userData = { ...(c.userData || {}), part: "connector" };
                }
              }
            }
          } catch { /* геометричний детект не критичний */ }
          // Додатково — за назвою/userData (коли теги збереглися, напр. GLB).
          m.obj.traverse((c: any) => { if (isConnectorMesh(c)) c.visible = false; });
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

        // Габарит БЕЗ шару Connector (ключі друкуються ПІД мапою окремо → роздували
        // габарит плитки і ламали розкладку/стик). Беремо лише КАРТУ (Base/Roads/…).
        const mapBoxOf = (obj: THREE.Object3D): THREE.Box3 => {
          const b = new THREE.Box3();
          let any = false;
          obj.traverse((c: any) => {
            if (c.isMesh && !isConnectorMesh(c)) {
              b.expandByObject(c);
              any = true;
            }
          });
          return any && !b.isEmpty() ? b : new THREE.Box3().setFromObject(obj);
        };
        const zoneInfo = models.map((m) => {
          const box = mapBoxOf(m.obj);
          return {
            size: box.getSize(new THREE.Vector3()),
            center: box.getCenter(new THREE.Vector3()),
            min: box.min.clone(),
            model: m,
          };
        });
        const metaByTaskId = batchZoneMetaByTaskId || {};

        // СПІЛЬНА ПІДЛОГА для всієї серії: коли бек зберігає абсолютну Z (preserve_z
        // для серії з elevation_ref), плитки мають РІЗНИЙ нижній край відповідно до
        // спільного baseline рельєфу. Якщо обнуляти КОЖНУ по власному min.y — це знову
        // розвалює baseline (сходинка/злам на шві). Тому опускаємо ВСЮ групу на ОДНЕ
        // спільне глобальне min.y → відносні висоти збережено → рельєф НЕПЕРЕРВНИЙ
        // через шов. (Якщо плитки й так на одній підлозі — це тотожно старій поведінці.)
        const sharedFloorY = Math.min(...zoneInfo.map((z) => z.min.y));

        // ГЕОГРАФІЧНА РОЗКЛАДКА (найточніша): кожна плитка має центроїд cx/cy (lng,lat)
        // → ставимо за РЕАЛЬНИМИ позиціями → точна тесселяція гекса/квадрата, ключі
        // сідають під своїми плитками (а не плавають). Раніше row/col-сітка ставила
        // гекси «криво». Падіння на стару логіку лише якщо центроїдів немає.
        const canUseGeo = models.length > 0 && models.every((m) => {
          const mt = (metaByTaskId as any)[m.id];
          return mt && Number.isFinite(mt.cx) && Number.isFinite(mt.cy);
        });

        if (canUseGeo) {
          const cyArr = models.map((m) => Number((metaByTaskId as any)[m.id].cy));
          const lat0 = cyArr.reduce((a, b) => a + b, 0) / cyArr.length;
          const cosLat = Math.max(Math.cos((lat0 * Math.PI) / 180), 0.05);
          const proj = models.map((m) => {
            const mt = (metaByTaskId as any)[m.id];
            return { x: Number(mt.cx) * 111320 * cosLat, y: Number(mt.cy) * 110540 };
          });
          const meanX = proj.reduce((a, p) => a + p.x, 0) / proj.length;
          const meanY = proj.reduce((a, p) => a + p.y, 0) / proj.length;
          let nn = Infinity;
          for (let a = 0; a < proj.length; a += 1) {
            for (let b = a + 1; b < proj.length; b += 1) {
              const d = Math.hypot(proj[a].x - proj[b].x, proj[a].y - proj[b].y);
              if (d > 1 && d < nn) nn = d;
            }
          }
          const maxW = Math.max(...zoneInfo.map((z) => z.size.x));
          // ТОЧНИЙ масштаб = scale_factor (мм/м), ЄДИНИЙ для конгруентних плиток. Гео-
          // крок між сусідами × sf = реальна ширина плитки → стикуються ВПРИТУЛ (шов
          // зникає, бо стінки сусідів збігаються). Раніше scale=maxW/nn брав rendered-
          // ширину НАЙШИРШОЇ плитки → вужча сусідка лишала зазор («бока»/лінії на шві).
          // Фолбек на maxW/nn лише якщо sf нема (старі дані).
          const sfArr = models
            .map((m) => Number((metaByTaskId as any)[m.id]?.sf))
            .filter((v) => Number.isFinite(v) && v > 0)
            .sort((a, b) => a - b);
          const sfMedian = sfArr.length ? sfArr[Math.floor(sfArr.length / 2)] : 0;
          const scale = sfMedian > 0
            ? sfMedian
            : ((Number.isFinite(nn) && nn > 1) ? maxW / nn : 1);
          zoneInfo.forEach((item, i) => {
            const ox = (proj[i].x - meanX) * scale;
            const oz = -(proj[i].y - meanY) * scale; // північ угору → -z
            item.model.obj.position.x = ox - item.center.x;
            item.model.obj.position.z = oz - item.center.z;
            item.model.obj.position.y = -sharedFloorY;
            item.model.obj.updateMatrixWorld(true);
          });
        } else if (!looksGlobal) {
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
            const stepZ = (gridType === "hexagonal") ? maxD * 0.75 : maxD * 1.0;

            zoneInfo.forEach((item) => {
              const meta = (metaByTaskId as any)[item.model.id] || {};
              const r = Number(meta.row ?? 0) - minRow;
              const c = Number(meta.col ?? 0) - minCol;
              const xShift = (gridType === "hexagonal" && (r % 2)) ? stepX * 0.5 : 0.0;

              item.model.obj.position.x = c * stepX + xShift - item.center.x;
              item.model.obj.position.z = r * stepZ - item.center.z;
              item.model.obj.position.y = -sharedFloorY;
              item.model.obj.updateMatrixWorld(true);
            });
          } else {
            console.warn("Batch preview: no geo/row-col metadata, using fallback grid layout");
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
              item.model.obj.position.y = -sharedFloorY;
              item.model.obj.updateMatrixWorld(true);
            });
          }
        }

        // 5. Масштабуємо ВСЮ групу під камеру за габаритом КАРТ (без ключів) — ТОЧНО
        // як одиночна модель (targetSize≈220). Без цього композит лишався у сирих
        // мм (~160) і рендерився ДРІБНИМ. Масштаб групи рівномірний → тесселяція
        // зберігається; ключі-метелики не враховуємо, щоб не роздути габарит.
        const mapBox0 = new THREE.Box3();
        let anyMapMesh = false;
        group.traverse((c: any) => {
          if (c.isMesh && !isConnectorMesh(c)) {
            mapBox0.expandByObject(c); anyMapMesh = true;
          }
        });
        if (!anyMapMesh || mapBox0.isEmpty()) mapBox0.setFromObject(group);
        const mapDim = Math.max(...mapBox0.getSize(new THREE.Vector3()).toArray());
        if (Number.isFinite(mapDim) && mapDim > 0.0001) {
          const viewScale = 220 / mapDim;
          group.scale.setScalar(viewScale);
          group.updateMatrixWorld(true);
        }

        // 6. Центруємо всю групу (після масштабу)
        const groupBox = new THREE.Box3().setFromObject(group);
        const gCenter = groupBox.getCenter(new THREE.Vector3());
        const gMin = groupBox.min.clone();
        group.position.x -= gCenter.x;
        group.position.z -= gCenter.z;
        group.position.y -= gMin.y;
        group.updateMatrixWorld(true);

        // 7. Кадруємо камеру на ВСЮ серію (карти, без шару Connector). Без цього
        // композит-група не мала підгону камери (fitCameraToObject звався лише для
        // одиночної моделі) → серія рендерилась ДРІБНОЮ в центрі канви.
        try { fitCameraToObject(group); } catch { /* камера-фіт не критичний */ }

        // Додаємо легкі візуальні індикатори для кожної зони (опціонально)
        // Для продуктивності не додаємо складні об'єкти, але зберігаємо інформацію

        (group as any).userData = { batch: true, ids: idsToLoad, zoneCount: idsToLoad.length };
        setModel(group);
      } catch (e: any) {
        setError(e?.message || t("batchLoadError"));
      } finally {
        setLoading(false);
      }
    };

    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showAllZones, taskIds.join(","), completedSignature]);

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
        setError(error.message || t("modelLoadError"));
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

  // Поки модель вантажиться / помилка / ще нема моделі — НЕ показуємо
  // дебаг-кубик/сітку/осі (це читалось як «зламаний рендер» на першому вході).
  // Сцена лишається чистою; підказку-empty-state показує оверлей у Preview3D,
  // а статус генерації — окремий оверлей «Генерація…».
  if (loading || error || !model) {
    if (error) console.error("Помилка в ModelLoader:", error);
    return (
      <>
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 10, 5]} intensity={0.8} />
        {/* Видимий стан завантаження превʼю: раніше тут була просто чорна
            сцена на час фетчу glb/3mf — читалось як «превʼю зламане». */}
        {loading && !error && (
          <Html center zIndexRange={[5, 0]}>
            <div className="pointer-events-none flex items-center gap-2 whitespace-nowrap rounded-full border border-white/15 bg-black/45 px-4 py-2 text-[13px] font-semibold text-white/85 backdrop-blur">
              <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/30 border-t-white/90" />
              {t("previewLoading")}
            </div>
          </Html>
        )}
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

// Catches WebGL context-creation/loss failures (and any throw inside the R3F
// tree) so a broken GPU/driver shows a friendly localized message instead of a
// blank/crashed canvas. The model stays orderable — only the on-screen preview
// is affected. React error boundaries must be class components (no hook API),
// so the localized fallback UI is injected via the `fallback` prop from the
// parent functional component that has access to next-intl.
class CanvasErrorBoundary extends Component<
  { children: ReactNode; fallback: ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };

  static getDerivedStateFromError(): { hasError: boolean } {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Keep a console trail for debugging GPU/driver issues without crashing the page.
    console.error("Preview3D Canvas error:", error, info?.componentStack);
  }

  render() {
    if (this.state.hasError) return this.props.fallback;
    return this.props.children;
  }
}

export function Preview3D({ capture = false }: { capture?: boolean } = {}) {
  const t = useTranslations("preview");
  const {
    downloadUrl,
    isGenerating,
    guidedMode,
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
  } = useGenerationStore(useShallow((st) => ({
    downloadUrl: st.downloadUrl,
    isGenerating: st.isGenerating,
    guidedMode: st.guidedMode,
    progress: st.progress,
    terrainSmoothShading: st.terrainSmoothShading,
    setTerrainSmoothShading: st.setTerrainSmoothShading,
    taskStatuses: st.taskStatuses,
    activeTaskId: st.activeTaskId,
    previewIncludeBase: st.previewIncludeBase,
    previewIncludeRoads: st.previewIncludeRoads,
    previewIncludeBuildings: st.previewIncludeBuildings,
    previewIncludeWater: st.previewIncludeWater,
    previewIncludeParks: st.previewIncludeParks,
    setPreviewIncludeBase: st.setPreviewIncludeBase,
    setPreviewIncludeRoads: st.setPreviewIncludeRoads,
    setPreviewIncludeBuildings: st.setPreviewIncludeBuildings,
    setPreviewIncludeWater: st.setPreviewIncludeWater,
    setPreviewIncludeParks: st.setPreviewIncludeParks,
  })));
  const [gridVisible, setGridVisible] = useState(false);
  const [axesVisible, setAxesVisible] = useState(false);
  const [rotateMode, setRotateMode] = useState<RotateMode>("camera");
  const [cameraMode, setCameraMode] = useState<CameraMode>("orbit");
  const [flySpeed, setFlySpeed] = useState<number>(120);
  const [isFs, setIsFs] = useState(false);
  const [toolsOpen, setToolsOpen] = useState(false);
  // Set when the WebGL context is lost after creation (GPU reset / driver crash)
  // — shows the same friendly fallback as the render-time error boundary.
  const [canvasFailed, setCanvasFailed] = useState(false);
  // Bubbled up from ModelLoader (inside the Canvas) so a failed model load shows
  // a visible localized message instead of an empty scene.
  const [loadError, setLoadError] = useState<string | null>(null);
  // F-02 (скрол-пастка на телефоні): three.OrbitControls ставить canvas
  // `touch-action:none`, і сторінка не прокручується свайпом над сценою. На
  // coarse-pointer пристроях керування вмикається лише після дотику
  // («Торкніться, щоб покрутити»), інакше canvas пропускає вертикальний скрол.
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [coarse, setCoarse] = useState(false);
  const [touchActive, setTouchActive] = useState(false);
  useEffect(() => {
    try { setCoarse(window.matchMedia("(pointer: coarse)").matches); } catch { /* ignore */ }
  }, []);
  const controlsEnabled = !coarse || touchActive || isFs;
  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const apply = () => {
      const c = root.querySelector("canvas");
      // !important — бо OrbitControls пише touchAction='none' у конструкторі (без important).
      if (c) c.style.setProperty("touch-action", controlsEnabled ? "none" : "pan-y", "important");
    };
    apply();
    // Canvas зʼявляється асинхронно (Suspense) — дочекатись і застосувати ще раз.
    const mo = new MutationObserver(apply);
    mo.observe(root, { childList: true, subtree: true });
    return () => mo.disconnect();
  }, [controlsEnabled]);
  // Вихід із режиму обертання, коли сцена майже зникла з екрана (скрол далі).
  useEffect(() => {
    if (!touchActive || !rootRef.current || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(
      (entries) => { if (entries.some((e) => !e.isIntersecting)) setTouchActive(false); },
      { threshold: 0.25 },
    );
    io.observe(rootRef.current);
    return () => io.disconnect();
  }, [touchActive]);
  // CSS-розгортання (працює на iPhone, на відміну від Fullscreen API).
  useEffect(() => {
    if (!isFs) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = prev; };
  }, [isFs]);
  const toggleFullscreen = () => setIsFs((v) => !v);

  return (
    <div
      ref={rootRef}
      className={isFs ? "fixed inset-0 z-[9999] bg-slate-950" : "relative h-full w-full bg-slate-950"}
      style={isFs ? undefined : { minHeight: "100%" }}
    >
      {/* Компактна панель: на весь екран + (опційно) інструменти. За замовчуванням
          інструменти приховані — щоб було видно саму модель. */}
      {!capture && (
        <div className="absolute right-3 top-3 z-30 flex items-center gap-2">
          <button
            type="button"
            onClick={toggleFullscreen}
            className="flex h-10 items-center gap-1.5 rounded-full border border-white/15 bg-[rgba(2,6,23,0.7)] px-3 text-[12px] font-semibold text-white backdrop-blur transition hover:bg-[rgba(2,6,23,0.9)]"
            title={t("tools.fullscreen")}
          >
            {isFs ? `✕ ${t("tools.collapse")}` : `⤢ ${t("tools.fullscreen")}`}
          </button>
          {/* ⚙ dev-tools панель ПРИБРАНА (власник: «максимально просто») — Orbit/Fly/
              WASD/Grid/Axes/Smooth-Flat/шари були інженерні. Лишається перетягування
              для обертання + повноекранний режим. Панель нижче лишена в коді під
              toolsOpen (завжди false → не рендериться); тут НЕ показуємо кнопку. */}
        </div>
      )}
      {!capture && toolsOpen && <div className="pointer-events-none absolute inset-x-3 top-16 z-20 flex justify-end">
        <div className="pointer-events-auto w-full max-w-[320px] max-h-[calc(100dvh-6rem)] overflow-y-auto rounded-[24px] border border-white/10 bg-[rgba(2,6,23,0.74)] px-3 py-3 text-white shadow-[0_20px_55px_rgba(2,6,23,0.45)] backdrop-blur">
          <div className="mb-3 flex items-start justify-between gap-3">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-white/55">
                Preview Tools
              </div>
              <div className="mt-1 text-sm font-semibold">{t("tools.sceneControl")}</div>
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
              <span className="block text-[10px] uppercase tracking-[0.18em] text-white/45">{t("tools.grid")}</span>
              <span className="mt-1 block">{gridVisible ? t("tools.onF") : t("tools.hiddenF")}</span>
            </button>
            <button
              className="rounded-2xl bg-white/10 px-3 py-2 text-left text-xs font-medium transition hover:bg-white/20"
              onClick={() => setAxesVisible((v) => !v)}
            >
              <span className="block text-[10px] uppercase tracking-[0.18em] text-white/45">{t("tools.axes")}</span>
              <span className="mt-1 block">{axesVisible ? t("tools.onPl") : t("tools.hiddenPl")}</span>
            </button>
            <button
              className="rounded-2xl bg-white/10 px-3 py-2 text-left text-xs font-medium transition hover:bg-white/20"
              onClick={() => setRotateMode((m) => (m === "camera" ? "model" : "camera"))}
              title={t("tools.rotateHint")}
            >
              <span className="block text-[10px] uppercase tracking-[0.18em] text-white/45">{t("tools.rotation")}</span>
              <span className="mt-1 block">{rotateMode === "camera" ? t("tools.camera") : t("tools.model")}</span>
            </button>
            <button
              className="rounded-2xl bg-white/10 px-3 py-2 text-left text-xs font-medium transition hover:bg-white/20"
              onClick={() => setCameraMode((m) => (m === "orbit" ? "fly" : "orbit"))}
              title={t("tools.cameraHint")}
            >
              <span className="block text-[10px] uppercase tracking-[0.18em] text-white/45">{t("tools.camera")}</span>
              <span className="mt-1 block">{cameraMode === "orbit" ? "Orbit" : "Fly"}</span>
            </button>
          </div>

          {cameraMode === "fly" && (
            <div className="mt-3 rounded-[18px] bg-white/8 px-3 py-3">
              <div className="flex items-center justify-between gap-3 text-xs">
                <span className="font-medium">{t("tools.flySpeed")}</span>
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
              <div className="font-medium">{t("tools.terrainShading")}</div>
              <div className="mt-1 text-white/60">
                {terrainSmoothShading ? t("tools.shadingSmooth") : t("tools.shadingFlat")}
              </div>
            </div>
            <button
              className="rounded-full bg-white/10 px-3 py-2 font-semibold transition hover:bg-white/20"
              onClick={() => setTerrainSmoothShading(!terrainSmoothShading)}
              title={t("tools.shadingToggleHint")}
            >
              {terrainSmoothShading ? "Smooth" : "Flat"}
            </button>
          </div>

          <div className="mt-3 text-[10px] leading-4 text-white/55">
            {cameraMode === "fly"
              ? t("tools.hintFly")
              : rotateMode === "model"
                ? t("tools.hintModel")
                : t("tools.hintCamera")}
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
            <div className="text-xs font-semibold text-white/90 mb-1">{t("tools.visibility")}</div>
            <label className="flex items-center justify-between gap-3 cursor-pointer">
              <span className="text-xs">{t("tools.terrain")}</span>
              <input
                type="checkbox"
                checked={previewIncludeBase}
                onChange={(e) => setPreviewIncludeBase(e.target.checked)}
                className="w-4 h-4"
              />
            </label>
            <label className="flex items-center justify-between gap-3 cursor-pointer">
              <span className="text-xs">{t("tools.roads")}</span>
              <input
                type="checkbox"
                checked={previewIncludeRoads}
                onChange={(e) => setPreviewIncludeRoads(e.target.checked)}
                className="w-4 h-4"
              />
            </label>
            <label className="flex items-center justify-between gap-3 cursor-pointer">
              <span className="text-xs">{t("tools.buildings")}</span>
              <input
                type="checkbox"
                checked={previewIncludeBuildings}
                onChange={(e) => setPreviewIncludeBuildings(e.target.checked)}
                className="w-4 h-4"
              />
            </label>
            <label className="flex items-center justify-between gap-3 cursor-pointer">
              <span className="text-xs">{t("tools.water")}</span>
              <input
                type="checkbox"
                checked={previewIncludeWater}
                onChange={(e) => setPreviewIncludeWater(e.target.checked)}
                className="w-4 h-4"
              />
            </label>
            <label className="flex items-center justify-between gap-3 cursor-pointer">
              <span className="text-xs">{t("tools.parks")}</span>
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
      {/* D-3 (2026-09-03): у guided прогрес живе ОДИН — у панелі з етапами.
          Сцена дублювала його своїм оверлеєм (третій індикатор в аудиті). */}
      {isGenerating && !capture && !guidedMode && (
        <div className="absolute inset-0 flex items-center justify-center text-white z-10 pointer-events-none">
          <div className="text-center">
            <p className="text-lg mb-2">{t("generating")}</p>
            <p className="text-sm text-gray-400">{progress}%</p>
          </div>
        </div>
      )}
      {!downloadUrl && !isGenerating && !capture && (
        // Перший вхід: дружня підказка замість сирого three.js-кубика.
        <div className="absolute inset-0 z-10 flex items-center justify-center px-6 pointer-events-none">
          <div className="max-w-[300px] text-center">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="mx-auto mb-3 h-10 w-10 text-white/60"
              aria-hidden="true"
            >
              <path d="M21 10c0 7-9 12-9 12s-9-5-9-12a9 9 0 0 1 18 0Z" />
              <circle cx="12" cy="10" r="3" />
            </svg>
            <p className="text-[15px] font-medium text-white/90">{t("emptyTitle")}</p>
            <p className="mt-1.5 text-[13px] leading-relaxed text-white/55">{t("emptyBody")}</p>
          </div>
        </div>
      )}
      {/* F-02: на телефоні сцена спершу «прозора» для скролу; тап вмикає обертання. */}
      {coarse && !isFs && !capture && !!downloadUrl && !isGenerating && (
        touchActive ? (
          <button
            type="button"
            onClick={() => setTouchActive(false)}
            className="absolute bottom-3 left-1/2 z-30 -translate-x-1/2 rounded-full border border-white/20 bg-[rgba(2,6,23,0.75)] px-4 py-2 text-[12px] font-semibold text-white backdrop-blur"
          >
            ✓ {t("tapDone")}
          </button>
        ) : (
          <button
            type="button"
            data-testid="preview-tap-to-rotate"
            onClick={() => setTouchActive(true)}
            aria-label={t("tapRotate")}
            className="absolute inset-0 z-[15] flex items-end justify-center bg-transparent pb-3"
          >
            <span className="rounded-full border border-white/20 bg-[rgba(2,6,23,0.75)] px-4 py-2 text-[12px] font-semibold text-white backdrop-blur">
              {t("tapRotate")}
            </span>
          </button>
        )
      )}
      <CanvasErrorBoundary
        fallback={
          <div className="absolute inset-0 z-20 flex items-center justify-center px-6 text-center">
            <div className="max-w-[320px]">
              <div aria-hidden className="mx-auto mb-3 text-3xl">🖥️</div>
              <p className="text-[15px] font-semibold text-white/90">{t("webglUnavailable")}</p>
              <p className="mt-1.5 text-[13px] leading-relaxed text-white/55">{t("webglUnavailableBody")}</p>
            </div>
          </div>
        }
      >
        <Canvas
          style={{ width: '100%', height: '100%', display: 'block' }}
          // preserveDrawingBuffer: без нього WebGL очищає буфер після компонування,
          // і toDataURL (знімок прев'ю для замовлення / мініатюра в кабінеті) виходить
          // ПОРОЖНІМ — «прев'ю 2» приходило оператору білим кадром. З ним знімок читає
          // фактично відрендерену сцену.
          gl={{ preserveDrawingBuffer: true }}
          // Surface a hard WebGL context-creation failure (no GPU / driver) to the
          // error boundary instead of leaving an invisible/broken canvas.
          onCreated={({ gl }) => {
            const canvas = gl.domElement;
            const onLost = (e: Event) => {
              e.preventDefault();
              setCanvasFailed(true);
            };
            canvas.addEventListener("webglcontextlost", onLost, false);
          }}
        >
          <Suspense fallback={null}>
            <CameraController />
            <FreeFlyControls enabled={cameraMode === "fly"} speed={flySpeed} onSpeedChange={setFlySpeed} />
            <OrbitControls
              makeDefault
              enabled={cameraMode === "orbit" && controlsEnabled}
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
            <ModelLoader rotateMode={rotateMode} onError={setLoadError} />
          </Suspense>
        </Canvas>
      </CanvasErrorBoundary>
      {/* Visible localized message when a model fails to load (404/empty/parse). */}
      {loadError && !isGenerating && !capture && (
        <div className="absolute inset-x-0 bottom-[92px] sm:bottom-3 z-20 flex justify-center px-4">
          <div className="max-w-[340px] rounded-[16px] border border-red-400/30 bg-red-950/70 px-4 py-2.5 text-center text-[12px] leading-4 text-red-100 backdrop-blur">
            {t("modelLoadError")}
          </div>
        </div>
      )}
      {canvasFailed && (
        <div className="absolute inset-0 z-20 flex items-center justify-center px-6 text-center">
          <div className="max-w-[320px]">
            <div aria-hidden className="mx-auto mb-3 text-3xl">🖥️</div>
            <p className="text-[15px] font-semibold text-white/90">{t("webglUnavailable")}</p>
            <p className="mt-1.5 text-[13px] leading-relaxed text-white/55">{t("webglUnavailableBody")}</p>
          </div>
        </div>
      )}
    </div>
  );
}
