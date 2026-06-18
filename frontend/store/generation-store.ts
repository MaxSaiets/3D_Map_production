import { create } from "zustand";
import { LatLngBounds } from "leaflet";

import type { TaskStatus } from "@/lib/api";

interface GenerationState {
  selectedArea: LatLngBounds | null;
  isGenerating: boolean;
  // Для single: taskGroupId === activeTaskId
  // Для batch: taskGroupId === "batch_<uuid>", activeTaskId === один з taskIds
  taskGroupId: string | null;
  taskIds: string[];
  activeTaskId: string | null;
  progress: number;
  status: string;
  downloadUrl: string | null;
  printQuality: TaskStatus["print_quality"] | null;
  taskStatuses: Record<string, TaskStatus>;
  showAllZones: boolean;
  // Batch preview: mapping taskId -> zone meta (so we can place tiles like on the map)
  batchZoneMetaByTaskId: Record<string, { zoneId: string; row?: number; col?: number }>;

  // Параметри генерації
  roadWidthMultiplier: number;
  roadHeightMm: number;
  roadEmbedMm: number;
  buildingMinHeight: number;
  buildingHeightMultiplier: number;
  buildingFoundationMm: number;
  buildingEmbedMm: number;
  waterDepth: number;
  terrainEnabled: boolean;
  terrainZScale: number;
  terrainBaseThicknessMm: number;
  terrainResolution: number;
  terrariumZoom: number;
  exportFormat: "stl" | "3mf";
  modelSizeMm: number; // Розмір моделі в міліметрах

  // Single-figure rotation (NOT grid mode): rotated rectangle corners [lon,lat]
  // sent to the backend as zone_polygon_coords so OSM is cropped to the figure.
  zonePolygonCoords: Array<[number, number]> | null;
  cropRotationDeg: number;

  // D4 GPX: bbox+точки завантаженого треку. Overlay карти центрує зону на
  // треку і малює полілінію; зберігається в store (не event), бо overlay
  // перебудовується при зміні розміру моделі і має знов застосувати фокус.
  gpxFocus: { west: number; south: number; east: number; north: number; points: Array<[number, number]> } | null;

  // Стан простої панелі /create — У STORE, НЕ в useState! Панель змонтована
  // ДВІЧІ (desktop sidebar + mobile таб): локальний стан розсинхронізовувався
  // між копіями → вибране «Панно 3×3»/GPX/магніт губилось при генерації з
  // іншої копії (юзер: «панно згенерувало не те»).
  simplePanelMode: 0 | 2 | 3;
  simpleMagnetMode: boolean;
  simpleMapLabel: string;
  // styleId + активний шаблон теж спільні (панель монтується двічі) — інакше
  // вибраний стиль/шаблон розсинхронізується між desktop/mobile копіями і при
  // ресайзі або генерації з іншої копії застосовувався б старий стиль.
  simpleStyleId: string;
  simpleTemplate: string | null;
  // «Плоска кольорова (AMS)» — пласка багатокольорова плитка-карта (кожен шар
  // окремий колір-філамент), міцна основа 3мм. Стан спільний (панель ×2).
  simpleFlatAms: boolean;
  // З'ЄДНУВАЧ-ПАЗИ (метелик): «ластівчин-хвіст» пази на гранях + деталь-ключ,
  // щоб стикувати дві плоскі карти. Стан спільний (панель монтується двічі).
  simpleConnector: boolean;
  // ПРЕМІУМ-РАМКА: компас + масштабна лінійка + координати поверх плоскої карти.
  simpleFrame: boolean;
  // РЕЛЬЄФ (висоти землі): окремий перемикач для УСІХ режимів карт. Джерело правди
  // для terrain_enabled у Просто-флоу (стиль лише задає дефолт). Вмикається лише на
  // 3D-карті (повний пайплайн); плоскі режими (AMS/магніт/конектор/рамка) — без рельєфу.
  simpleRelief: boolean;
  // ПЛАСКІ БУДИНКИ: у плоских режимах будинки = тонкі footprint-плити одної висоти
  // (чистіший AMS-друк) замість лог-масштабованих блоків. Opt-in.
  simpleFlatBuildings: boolean;
  // ВИДІЛЕНІ БУДІВЛІ: користувач клікає свої будинки (дім/робота/орієнтири) на карті
  // → highlightPoints [[lon,lat],...] → КОЖЕН окрема ЧЕРВОНА вставна деталь. Плоский режим.
  mapHighlightBuilding: boolean;
  highlightPoints: Array<[number, number]>;
  // Контури обраних будівель (паралельно highlightPoints, той самий індекс) — щоб
  // підсвітити на карті РЕАЛЬНИЙ контур будинку, а не лише крапку. null = ще не завантажено.
  highlightFootprints: Array<Array<[number, number]> | null>;
  gpxName: string | null;
  gpxNote: string | null;

  // Preview only
  // Preview only
  terrainSmoothShading: boolean;

  // AMS Mode
  isAmsMode: boolean;
  flatPlateMode: boolean;

  // Fast Preview Mode (~30s vs full 5-15min)
  previewMode: boolean;

  // Preview visibility controls
  previewIncludeBase: boolean;
  previewIncludeRoads: boolean;
  previewIncludeBuildings: boolean;
  previewIncludeWater: boolean;
  previewIncludeParks: boolean;

  // Actions
  setSelectedArea: (area: LatLngBounds | null) => void;
  setZonePolygonCoords: (coords: Array<[number, number]> | null) => void;
  setGpxFocus: (focus: GenerationState["gpxFocus"]) => void;
  setSimplePanelMode: (mode: 0 | 2 | 3) => void;
  setSimpleMagnetMode: (on: boolean) => void;
  setSimpleMapLabel: (label: string) => void;
  setSimpleStyleId: (id: string) => void;
  setSimpleTemplate: (id: string | null) => void;
  setSimpleFlatAms: (on: boolean) => void;
  setSimpleConnector: (on: boolean) => void;
  setSimpleFrame: (on: boolean) => void;
  setSimpleRelief: (on: boolean) => void;
  setSimpleFlatBuildings: (on: boolean) => void;
  setMapHighlightBuilding: (on: boolean) => void;
  clearHighlights: () => void;
  addHighlightPoint: (pt: [number, number]) => void;
  removeHighlightAt: (index: number) => void;
  setHighlightFootprint: (pt: [number, number], poly: Array<[number, number]>) => void;
  setGpxName: (name: string | null) => void;
  setGpxNote: (note: string | null) => void;
  setCropRotationDeg: (deg: number) => void;
  setGenerating: (isGenerating: boolean) => void;
  setTaskGroup: (groupId: string | null, taskIds?: string[], productType?: "map" | "keychain") => void;
  setActiveTaskId: (taskId: string | null) => void;
  setTaskStatuses: (statuses: Record<string, TaskStatus>) => void;
  setShowAllZones: (value: boolean) => void;
  setBatchZoneMetaByTaskId: (value: Record<string, { zoneId: string; row?: number; col?: number }>) => void;
  updateProgress: (progress: number, status: string) => void;
  setDownloadUrl: (url: string | null) => void;
  setPrintQuality: (pq: TaskStatus["print_quality"] | null) => void;

  // Параметри
  setRoadWidthMultiplier: (value: number) => void;
  setRoadHeightMm: (value: number) => void;
  setRoadEmbedMm: (value: number) => void;
  setBuildingMinHeight: (value: number) => void;
  setBuildingHeightMultiplier: (value: number) => void;
  setBuildingFoundationMm: (value: number) => void;
  setBuildingEmbedMm: (value: number) => void;
  setWaterDepth: (value: number) => void;
  setTerrainEnabled: (value: boolean) => void;
  setTerrainZScale: (value: number) => void;
  setTerrainBaseThicknessMm: (value: number) => void;
  setTerrainResolution: (value: number) => void;
  setTerrariumZoom: (value: number) => void;
  setExportFormat: (format: "stl" | "3mf") => void;
  setModelSizeMm: (value: number) => void;

  setTerrainSmoothShading: (value: boolean) => void;
  setAmsMode: (value: boolean) => void;
  setFlatPlateMode: (value: boolean) => void;
  setPreviewMode: (value: boolean) => void;

  setPreviewIncludeBase: (value: boolean) => void;
  setPreviewIncludeRoads: (value: boolean) => void;
  setPreviewIncludeBuildings: (value: boolean) => void;
  setPreviewIncludeWater: (value: boolean) => void;
  setPreviewIncludeParks: (value: boolean) => void;

  reset: () => void;
}

const initialState = {
  selectedArea: null,
  isGenerating: false,
  taskGroupId: null,
  taskIds: [] as string[],
  activeTaskId: null,
  progress: 0,
  status: "",
  downloadUrl: null,
  printQuality: null as TaskStatus["print_quality"] | null,
  taskStatuses: {} as Record<string, TaskStatus>,
  showAllZones: false,
  batchZoneMetaByTaskId: {} as Record<string, { zoneId: string; row?: number; col?: number }>,
  // На 10×10см “реальні” ширини доріг часто виглядають надто товстими — ставимо мʼякший дефолт.
  roadWidthMultiplier: 0.8,
  // Дороги: менша висота + трохи більше втиснення дають кращий вигляд і менше z-fighting
  roadHeightMm: 0.5,
  roadEmbedMm: 0.3,
  // Реальні OSM висоти на масштабі 10x10см часто виглядають занадто низько,
  // тому робимо трохи вищі дефолти (користувач може змінити слайдерами).
  buildingMinHeight: 5.0,
  buildingHeightMultiplier: 1.8,
  buildingFoundationMm: 0.6,
  buildingEmbedMm: 0.2,
  waterDepth: 2.0,
  terrainEnabled: true,
  terrainZScale: 1.0,
  // Тонка “підложка” під рельєф (мм на фінальній моделі)
  terrainBaseThicknessMm: 0.3,
  // Вища деталізація рельєфу -> менші трикутники, більше “реальності”
  terrainResolution: 180,
  terrariumZoom: 15,
  exportFormat: "3mf" as const,
  modelSizeMm: 80.0, // 80мм = 8см за замовчуванням
  zonePolygonCoords: null,
  cropRotationDeg: 0,
  gpxFocus: null,
  simplePanelMode: 0 as const,
  simpleMagnetMode: false,
  simpleMapLabel: "",
  simpleStyleId: "full",
  simpleTemplate: null,
  simpleFlatAms: false,
  simpleConnector: false,
  simpleFrame: false,
  simpleRelief: false,
  simpleFlatBuildings: false,
  mapHighlightBuilding: false,
  highlightPoints: [],
  highlightFootprints: [],
  gpxName: null,
  gpxNote: null,

  // Preview: smooth shading can show a visible seam between separate tiles on slopes
  terrainSmoothShading: false,
  isAmsMode: false,
  flatPlateMode: false,
  previewMode: true,  // default to fast preview for buyers
  // Preview visibility defaults - all enabled
  previewIncludeBase: true,
  previewIncludeRoads: true,
  previewIncludeBuildings: true,
  previewIncludeWater: true,
  previewIncludeParks: true,
};

export const useGenerationStore = create<GenerationState>((set) => ({
  ...initialState,

  setSelectedArea: (area) => set({ selectedArea: area }),
  setZonePolygonCoords: (coords) => set({ zonePolygonCoords: coords }),
  setGpxFocus: (focus) => set({ gpxFocus: focus }),
  setSimplePanelMode: (mode) => set({ simplePanelMode: mode }),
  setSimpleMagnetMode: (on) => set({ simpleMagnetMode: on }),
  setSimpleMapLabel: (label) => set({ simpleMapLabel: label }),
  setSimpleStyleId: (id) => set({ simpleStyleId: id }),
  setSimpleTemplate: (id) => set({ simpleTemplate: id }),
  setSimpleFlatAms: (on) => set({ simpleFlatAms: on }),
  setSimpleConnector: (on) => set({ simpleConnector: on }),
  setSimpleFrame: (on) => set({ simpleFrame: on }),
  setSimpleRelief: (on) => set({ simpleRelief: on }),
  setSimpleFlatBuildings: (on) => set({ simpleFlatBuildings: on }),
  setMapHighlightBuilding: (on) => set({ mapHighlightBuilding: on }),
  clearHighlights: () => set({ highlightPoints: [], highlightFootprints: [] }),
  addHighlightPoint: (pt) => set((st) => (st.highlightPoints.length >= 12 ? st : ({
    highlightPoints: [...st.highlightPoints, pt],
    highlightFootprints: [...st.highlightFootprints, null],
  }))),
  removeHighlightAt: (index) => set((st) => (index < 0 || index >= st.highlightPoints.length ? st : ({
    highlightPoints: st.highlightPoints.filter((_, i) => i !== index),
    highlightFootprints: st.highlightFootprints.filter((_, i) => i !== index),
  }))),
  setHighlightFootprint: (pt, poly) => set((st) => {
    const i = st.highlightPoints.findIndex((p) => p[0] === pt[0] && p[1] === pt[1]);
    if (i < 0) return st;
    const f = [...st.highlightFootprints];
    f[i] = poly;
    return { highlightFootprints: f };
  }),
  setGpxName: (name) => set({ gpxName: name }),
  setGpxNote: (note) => set({ gpxNote: note }),
  setCropRotationDeg: (deg) => set({ cropRotationDeg: deg }),
  setGenerating: (isGenerating) => set({ isGenerating }),
  setTaskGroup: (taskGroupId, taskIds, productType = "map") =>
    set((s) => {
      const nextTaskIds = taskIds ?? (taskGroupId ? [taskGroupId] : []);
      const nextActive = s.activeTaskId && nextTaskIds.includes(s.activeTaskId)
        ? s.activeTaskId
        : (nextTaskIds[0] ?? null);
      // Зберігаємо в localStorage щоб відновити після refresh. ПРОДУКТ-ТИП (map|keychain)
      // потрібен щоб /create і /keychains відновлювали ЛИШЕ свої задачі (ключ той самий).
      if (typeof window !== "undefined") {
        if (taskGroupId) {
          localStorage.setItem("3dmap_task_group_id", taskGroupId);
          localStorage.setItem("3dmap_task_ids", JSON.stringify(nextTaskIds));
          localStorage.setItem("3dmap_task_product", productType);
        } else {
          localStorage.removeItem("3dmap_task_group_id");
          localStorage.removeItem("3dmap_task_ids");
          localStorage.removeItem("3dmap_task_product");
        }
      }
      return {
        taskGroupId,
        taskIds: nextTaskIds,
        activeTaskId: nextActive,
        // при новій задачі скидаємо статуси і URL
        taskStatuses: {},
        downloadUrl: null,
        printQuality: null,
        progress: 0,
        status: "waiting",
      };
    }),
  setActiveTaskId: (activeTaskId) => set({ activeTaskId }),
  setTaskStatuses: (taskStatuses) => set({ taskStatuses }),
  setShowAllZones: (showAllZones) => set({ showAllZones }),
  setBatchZoneMetaByTaskId: (batchZoneMetaByTaskId) => set({ batchZoneMetaByTaskId }),
  updateProgress: (progress, status) => set({ progress, status }),
  setDownloadUrl: (url) => set({ downloadUrl: url }),
  setPrintQuality: (pq) => set({ printQuality: pq }),

  setRoadWidthMultiplier: (value) => set({ roadWidthMultiplier: value }),
  setRoadHeightMm: (value) => set({ roadHeightMm: value }),
  setRoadEmbedMm: (value) => set({ roadEmbedMm: value }),
  setBuildingMinHeight: (value) => set({ buildingMinHeight: value }),
  setBuildingHeightMultiplier: (value) => set({ buildingHeightMultiplier: value }),
  setBuildingFoundationMm: (value) => set({ buildingFoundationMm: value }),
  setBuildingEmbedMm: (value) => set({ buildingEmbedMm: value }),
  setWaterDepth: (value) => set({ waterDepth: value }),
  setTerrainEnabled: (value) => set({ terrainEnabled: value }),
  setTerrainZScale: (value) => set({ terrainZScale: value }),
  setTerrainBaseThicknessMm: (value) => set({ terrainBaseThicknessMm: value }),
  setTerrainResolution: (value) => set({ terrainResolution: value }),
  setTerrariumZoom: (value) => set({ terrariumZoom: value }),
  setExportFormat: (format) => set({ exportFormat: format }),
  setModelSizeMm: (value) => set({ modelSizeMm: value }),

  setTerrainSmoothShading: (value) => set({ terrainSmoothShading: value }),
  setAmsMode: (value) => set({ isAmsMode: value }),
  setFlatPlateMode: (value) => set({ flatPlateMode: value }),
  setPreviewMode: (value) => set({ previewMode: value }),

  setPreviewIncludeBase: (value) => set({ previewIncludeBase: value }),
  setPreviewIncludeRoads: (value) => set({ previewIncludeRoads: value }),
  setPreviewIncludeBuildings: (value) => set({ previewIncludeBuildings: value }),
  setPreviewIncludeWater: (value) => set({ previewIncludeWater: value }),
  setPreviewIncludeParks: (value) => set({ previewIncludeParks: value }),

  reset: () => set(initialState),
}));
