"use client";

import dynamic from "next/dynamic";
// ЛОКАЛІЗОВАНИЙ Link (@/i18n/navigation), НЕ next/link: інакше внутрішні
// посилання з /en/create губили префікс локалі (/keychains замість /en/keychains).
import { Link } from "@/i18n/navigation";
import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { Download, KeyRound, User, X, Home as HomeIcon, ShoppingBag } from "lucide-react";
import { ControlPanel } from "@/components/ControlPanel";
import { useGenerationStore } from "@/store/generation-store";
import { useShallow } from "zustand/react/shallow";
import { GPX_MAX_M_PER_MM } from "@/lib/generation";
import { OnboardingTour } from "@/components/OnboardingTour";
import { ConstructorIntro, useIntroGate } from "@/components/ConstructorIntro";
import { SimpleControlPanel } from "@/components/SimpleControlPanel";
import { ScenarioFlow } from "@/components/ScenarioFlow";
import { MAP_TEMPLATES, cityKeychainText } from "@/lib/templates";
import { useAuth } from "@/components/AuthProvider";
import { saveGrid, getGrid } from "@/lib/grids";
import { useTranslations, useLocale } from "next-intl";
import { SIMPLE_SIZES } from "@/lib/generation";
import { mapPriceEur, MAP_SIZE_PRICES_UAH } from "@/lib/mapPrices";

function MapLoading() {
  const tc = useTranslations("create");
  return (
    <div className="flex h-full min-h-[320px] items-center justify-center rounded-[24px] bg-[rgba(255,255,255,0.65)] text-sm text-[var(--text-secondary)]">
      {tc("loadingMap")}
    </div>
  );
}

function GridLoading() {
  const tc = useTranslations("create");
  return (
    <div className="flex h-full min-h-[320px] items-center justify-center rounded-[24px] bg-[rgba(255,255,255,0.65)] text-sm text-[var(--text-secondary)]">
      {tc("loadingGrid")}
    </div>
  );
}

function Preview3DLoading() {
  const tc = useTranslations("create");
  return (
    <div className="flex h-full min-h-[320px] items-center justify-center rounded-[20px] bg-[#0f172a] text-sm text-white/70">
      {tc("loading3d")}
    </div>
  );
}

const MapSelector = dynamic(
  () => import("@/components/MapSelector").then((mod) => ({ default: mod.MapSelector })),
  {
    ssr: false,
    loading: () => <MapLoading />,
  },
);

const HexagonalGrid = dynamic(() => import("@/components/HexagonalGrid"), {
  ssr: false,
  loading: () => <GridLoading />,
});

// Реальний 3D перегляд згенерованої моделі (Three.js, важкий — поза критичним шляхом, ssr:false)
const Preview3D = dynamic(
  () => import("@/components/Preview3D").then((mod) => ({ default: mod.Preview3D })),
  {
    ssr: false,
    loading: () => <Preview3DLoading />,
  },
);

const CITIES: Record<
  string,
  { bounds: { north: number; south: number; east: number; west: number }; center: [number, number] }
> = {
  Kyiv:           { bounds: { north: 50.60, south: 50.20, east: 30.80, west: 30.20 }, center: [50.4501, 30.5234] },
  Khmelnytskyi:   { bounds: { north: 49.48, south: 49.36, east: 27.08, west: 26.88 }, center: [49.42, 26.98] },
  Lviv:           { bounds: { north: 49.90, south: 49.78, east: 24.11, west: 23.95 }, center: [49.8397, 24.0297] },
  Odesa:          { bounds: { north: 46.56, south: 46.39, east: 30.83, west: 30.61 }, center: [46.4825, 30.7233] },
  Dnipro:         { bounds: { north: 48.55, south: 48.37, east: 35.14, west: 34.95 }, center: [48.4647, 35.0462] },
  Kharkiv:        { bounds: { north: 50.07, south: 49.92, east: 36.34, west: 36.12 }, center: [49.9935, 36.2304] },
  Vinnytsia:      { bounds: { north: 49.28, south: 49.18, east: 28.53, west: 28.40 }, center: [49.2331, 28.4682] },
  Zaporizhzhia:   { bounds: { north: 47.90, south: 47.78, east: 35.22, west: 35.07 }, center: [47.8388, 35.1396] },
  Kryvyi_Rih:     { bounds: { north: 47.98, south: 47.85, east: 33.44, west: 33.28 }, center: [47.9105, 33.3918] },
  Mykolaiv:       { bounds: { north: 46.99, south: 46.92, east: 32.08, west: 31.97 }, center: [46.9750, 32.0000] },
  Poltava:        { bounds: { north: 49.64, south: 49.54, east: 34.61, west: 34.48 }, center: [49.5883, 34.5514] },
  Cherkasy:       { bounds: { north: 49.47, south: 49.40, east: 32.11, west: 31.99 }, center: [49.4444, 32.0598] },
  Chernihiv:      { bounds: { north: 51.54, south: 51.44, east: 31.32, west: 31.22 }, center: [51.4982, 31.2893] },
  Ternopil:       { bounds: { north: 49.59, south: 49.52, east: 25.65, west: 25.53 }, center: [49.5535, 25.5948] },
  IvanoFrankivsk: { bounds: { north: 48.96, south: 48.88, east: 24.76, west: 24.65 }, center: [48.9226, 24.7111] },
  Zhytomyr:       { bounds: { north: 50.30, south: 50.23, east: 28.72, west: 28.61 }, center: [50.2547, 28.6587] },
  Sumy:           { bounds: { north: 50.95, south: 50.88, east: 34.84, west: 34.74 }, center: [50.9077, 34.7981] },
  Rivne:          { bounds: { north: 50.65, south: 50.57, east: 26.31, west: 26.18 }, center: [50.6199, 26.2516] },
  Lutsk:          { bounds: { north: 50.80, south: 50.70, east: 25.38, west: 25.27 }, center: [50.7472, 25.3254] },
  Uzhhorod:       { bounds: { north: 48.65, south: 48.60, east: 22.33, west: 22.26 }, center: [48.6238, 22.2947] },
  Chernivtsi:     { bounds: { north: 48.33, south: 48.26, east: 25.99, west: 25.90 }, center: [48.2921, 25.9310] },
  Kherson:        { bounds: { north: 46.67, south: 46.61, east: 32.67, west: 32.57 }, center: [46.6354, 32.6169] },
  Kropyvnytskyi:  { bounds: { north: 48.54, south: 48.47, east: 32.30, west: 32.20 }, center: [48.5132, 32.2597] },
};


export default function Home() {
  const tc = useTranslations("create");
  const tCity = useTranslations("cities");
  const locale = useLocale();
  // Вступний блок «що вийде». Стан тут (не в компоненті) — його читає й тур.
  const { introVisible, dismissIntro } = useIntroGate("intro_create_v1");
  // СІТКА СЕРІЇ — у store (НЕ page-level useState): панель монтується ДВІЧІ
  // (desktop aside + mobile section), локальний стан розсинхронізовувався б між
  // копіями (той самий клас багу, що й simplePanelMode). Сітка тепер валідна і в
  // «Просто», і в «Профі».
  const showHexGrid = useGenerationStore((st) => st.showHexGrid);
  const setShowHexGrid = useGenerationStore((st) => st.setShowHexGrid);
  const selectedZones = useGenerationStore((st) => st.selectedZones);
  const setSelectedZones = useGenerationStore((st) => st.setSelectedZones);
  const gridType = useGenerationStore((st) => st.gridType);
  const setGridType = useGenerationStore((st) => st.setGridType);
  const hexSizeM = useGenerationStore((st) => st.hexSizeM);
  const setHexSizeM = useGenerationStore((st) => st.setHexSizeM);
  const gridRotationDeg = useGenerationStore((st) => st.gridRotationDeg);
  const setGridRotationDeg = useGenerationStore((st) => st.setGridRotationDeg);
  // Поворот сітки: повзунок оновлює ЛОКАЛЬНИЙ pendingRot (дешево — лише підпис),
  // а важку сітку (тисячі клітин) обертаємо лише на ВІДПУСКАННЯ повзунка (commit).
  const [pendingRot, setPendingRot] = useState(gridRotationDeg);
  useEffect(() => { setPendingRot(gridRotationDeg); }, [gridRotationDeg]);
  const [currentCityKey, setCurrentCityKey] = useState("Kyiv");
  const [proMode, setProMode] = useState(false);
  // GUIDED-РЕЖИМ (сценарний вхід): ДЕФОЛТ УВІМКНЕНО для всіх — новий користувач
  // бачить 3 кроки замість ~43 контролів. Вимикається «Повний конструктор» /
  // «Підлаштувати деталі» (localStorage), вмикається кнопкою «← Простий режим».
  // Deep-link параметри (?template/?city/?grid/?capture) = намір у повний
  // конструктор → guided авто-вимикається (БЕЗ запису в localStorage).
  const tSc = useTranslations("scenario");
  const [guided, setGuidedState] = useState(true);
  useEffect(() => {
    try {
      const p = new URLSearchParams(window.location.search);
      if (p.get("template") || p.get("city") || p.get("grid") || p.get("capture")) {
        setGuidedState(false);
        return;
      }
      setGuidedState(localStorage.getItem("3dmap_guided_v1") !== "0");
    } catch { /* приватний режим → лишаємо guided on */ }
  }, []);
  const setGuided = (v: boolean) => {
    setGuidedState(v);
    try { localStorage.setItem("3dmap_guided_v1", v ? "1" : "0"); } catch { /* ignore */ }
  };
  useEffect(() => {
    try {
      const pro = localStorage.getItem("3dmap_pro_mode") === "1";
      setProMode(pro);
      // Серія-сітка тепер доступна в ОБОХ режимах — відновлюємо її НЕЗАЛЕЖНО від
      // proMode (раніше була лише у «Профі»; тепер «Серія зон» — це тумблер, що
      // лишає користувача у простій панелі).
      if (localStorage.getItem("3dmap_hex_grid") === "1") setShowHexGrid(true);
    } catch {/* ignore */}
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const toggleProMode = (v: boolean) => {
    setProMode(v);
    try { localStorage.setItem("3dmap_pro_mode", v ? "1" : "0"); } catch {/* ignore */}
    // Сітку БІЛЬШЕ НЕ гасимо при виході у «Просто» — вона валідна в обох режимах.
  };
  // showHexGrid зберігаємо у localStorage, щоб режим сітки переживав reload.
  const setShowHexGridPersist = (v: boolean) => {
    setShowHexGrid(v);
    try { localStorage.setItem("3dmap_hex_grid", v ? "1" : "0"); } catch {/* ignore */}
  };
  // map2model-стиль: ОДНА велика сцена — карта АБО 3D-рендер (не тісно поруч).
  // Перемикач зверху; авто-перехід на рендер, коли стартує генерація.
  const [stageView, setStageView] = useState<"map" | "render">("map");

  // ЗМІНА МІСТА: скидаємо зону ПЕРЕД зміною center — інакше overlay після
  // ремаунта карти відновлює рамку зі СТАРОГО міста і fitBounds повертає
  // карту назад (Рома: «коли обираєш місто, карта не переходить»).
  // /keychains уже робить так само.
  const handleCityChange = useCallback((key: string) => {
    useGenerationStore.getState().setSelectedArea(null);
    setCurrentCityKey(key);
  }, []);

  // Магнітний підпис (гравіювання) слідує за обраним містом через БУДЬ-ЯКИЙ шлях
  // зміни currentCityKey — дропдаун, ?city=, ?template=, відновлення сітки (Рома:
  // «щоб назви правильно брались для міст»). Перезаписуємо лише АВТО-значення:
  // порожнє поле або назву якогось міста зі списку. Власний підпис користувача
  // (напр. «ROMA ❤️») не збігається з жодною назвою → лишається недоторканим.
  useEffect(() => {
    const store = useGenerationStore.getState();
    const cur = (store.simpleMapLabel || "").trim();
    const isAuto = !cur || Object.keys(CITIES).some((k) => cityKeychainText(k) === cur);
    if (isAuto) store.setSimpleMapLabel(cityKeychainText(currentCityKey));
  }, [currentCityKey]);

  // useShallow: без селектора ця top-level сторінка ре-рендерилась на КОЖЕН store.set()
  // (кожен тік Pro-слайдера) і каскадно ре-рендерила важке дерево (leaflet-мапа +
  // three.js-превʼю + обидві панелі, жодна не memoized) — це зводило нанівець
  // per-field селектори самих панелей. Тепер сторінка ре-рендериться лише коли реально
  // змінюється одне з ЦИХ полів. Output-identical (ті самі значення).
  const { isGenerating, progress, status, downloadUrl, selectedArea, taskGroupId, taskIds, setTaskGroup, setGenerating, setActiveTaskId, setSelectedArea,
    modelSizeMm, cropRotationDeg, setCropRotationDeg, setZonePolygonCoords } = useGenerationStore(useShallow((st) => ({
      isGenerating: st.isGenerating, progress: st.progress, status: st.status, downloadUrl: st.downloadUrl,
      selectedArea: st.selectedArea, taskGroupId: st.taskGroupId, taskIds: st.taskIds,
      setTaskGroup: st.setTaskGroup, setGenerating: st.setGenerating, setActiveTaskId: st.setActiveTaskId,
      setSelectedArea: st.setSelectedArea, modelSizeMm: st.modelSizeMm, cropRotationDeg: st.cropRotationDeg,
      setCropRotationDeg: st.setCropRotationDeg, setZonePolygonCoords: st.setZonePolygonCoords,
    })));

  // Rotatable single-figure selector (only when NOT in grid mode). Reuses the
  // proven keychain crop overlay as a plain rotatable rectangle; its rotated
  // corners flow to the store as zone_polygon_coords so the backend crops OSM to
  // the figure. Sized by the 1:10000 model-size rule (mapWidthMm * 10 m).
  const handleMapRotation = useCallback((deg: number) => setCropRotationDeg(((deg % 360) + 360) % 360), [setCropRotationDeg]);
  const FIGURE_SHAPES = [
    { id: "rounded", label: tc("shapeRect") },
    { id: "circle", label: tc("shapeCircle") },
    { id: "hexagon", label: tc("shapeHexagon") },
    { id: "octagon", label: tc("shapeOctagon") },
    { id: "capsule", label: tc("shapeCapsule") },
    { id: "heart", label: tc("shapeHeart") },
  ] as const;
  const [figureShape, setFigureShape] = useState<string>("rounded");
  // Прямокутник за замовчуванням — ГОСТРІ кути 90° (фідбек власника). Тумблер
  // вмикає заокруглення (лише для прямокутника; решта форм заокруглені своєю суттю).
  const [roundCorners, setRoundCorners] = useState(false);
  // GPX: коли трек завантажено, дозволяємо зоні розширюватись понад 1:10000
  // (до GPX_MAX_M_PER_MM) — інакше довгий маршрут фізично не влазив і юзер
  // не міг збільшити зону.
  const gpxLoaded = Boolean(useGenerationStore((st) => st.gpxFocus));
  const mapCrop = useMemo(() => (showHexGrid ? undefined : {
    aspectRatio: 1,
    maxMetersPerMm: gpxLoaded ? GPX_MAX_M_PER_MM : 10,
    targetMetersPerMm: 6,
    mapWidthMm: modelSizeMm || 80,
    mapHeightMm: modelSizeMm || 80,
    baseShape: figureShape as any,
    cornerRadiusMm: figureShape === "rounded" && roundCorners ? 6 : 0,
    cropToShape: true,
    followGpxFocus: true,
    rotationDeg: cropRotationDeg,
    onRotationChange: handleMapRotation,
    onPolygonChange: (poly: Array<[number, number]>) => setZonePolygonCoords(poly),
  }), [showHexGrid, modelSizeMm, cropRotationDeg, handleMapRotation, setZonePolygonCoords, figureShape, roundCorners, gpxLoaded]);

  // Clear the rotated polygon when switching INTO grid mode (grid has its own
  // zone logic) so a stale figure crop can't leak into grid generation.
  useEffect(() => {
    if (showHexGrid) { setZonePolygonCoords(null); setCropRotationDeg(0); }
  }, [showHexGrid, setZonePolygonCoords, setCropRotationDeg]);

  // ── Personal city grids ─────────────────────────────────────────────
  const { getIdToken, openLogin } = useAuth();
  const [gridId, setGridId] = useState<string | null>(null);
  const [gridNotice, setGridNotice] = useState<string | null>(null);
  const [gridArea, setGridArea] = useState<{ north: number; south: number; east: number; west: number } | null>(null);
  // Клітини, вже куплені у попередньому замовленні цієї сітки (id `hex_2_3`),
  // щоб показати їх золотим і дати докупити лише НОВІ сусідні (продовження панно).
  const [boughtCells, setBoughtCells] = useState<Set<string>>(new Set());

  // Префікс id клітини за типом сітки (бек: hex_/square_/circle_{row}_{col}).
  const cellIdPrefix = (gt?: string) => (gt === "square" ? "square" : gt === "circle" ? "circle" : "hex");

  // Локалізована назва типу сітки для назви збереженої сітки.
  const gridTypeLabel = (gt: string) =>
    gt === "square" ? tc("gridTypeSquares") : gt === "circle" ? tc("gridTypeCircles") : tc("gridTypeHexagons");

  // Load a saved grid from history (?grid=<id>): reproduces the same tiling so
  // the user can pick neighbouring cells and generate them.
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("grid");
    if (!id) return;
    (async () => {
      const token = await getIdToken();
      const g = await getGrid(token, id);
      if (!g) return;
      if (g.city && CITIES[g.city]) setCurrentCityKey(g.city);
      if (g.grid_type) setGridType(g.grid_type);
      if (g.hex_size_m) setHexSizeM(g.hex_size_m);
      if (g.bounds) setGridArea(g.bounds);
      if (typeof g.rotation_deg === "number") setGridRotationDeg(g.rotation_deg);
      setGridId(g.id || id);
      // Куплені клітини → золоті, не обираються повторно.
      const px = cellIdPrefix(g.grid_type);
      // Будь-яка збережена клітина = частина попереднього панно → золота на
      // повторному відкритті (продовження = додати сусідні, замовити лише нові).
      const bought = new Set<string>(
        (g.cells || [])
          .filter((c: any) => c && c.row != null && c.col != null)
          .map((c: any) => `${px}_${c.row}_${c.col}`),
      );
      setBoughtCells(bought);
      // Завантажена сітка → повний режим сітки (Профі-панель має генерацію серії).
      setShowHexGridPersist(true);
      toggleProMode(true);
      const gridName = g.name || g.city || tc("gridFallbackName");
      setGridNotice(
        bought.size > 0
          ? tc("gridBoughtNotice", { name: gridName, count: bought.size })
          : tc("gridLoadedNotice", { name: gridName }),
      );
    })();
  }, [getIdToken]);

  const handleSaveGrid = useCallback(async () => {
    const token = await getIdToken();
    if (!token) { setGridNotice(tc("gridLoginToSave")); return; }
    const city = CITIES[currentCityKey];
    const grid = await saveGrid(token, {
      id: gridId || undefined,
      name: `${tCity(currentCityKey)} · ${gridTypeLabel(gridType)}`,
      city: currentCityKey,
      center: city?.center,
      grid_type: gridType,
      hex_size_m: hexSizeM,
      bounds: gridArea || city?.bounds,
      rotation_deg: gridRotationDeg,
      // row/col живуть у feature.properties (GeoJSON) — раніше читалось z.row
      // → падало на індекс i, тож збережені клітини не збігались зі справжніми
      // координатами сітки (продовження відкривало не ті зони).
      cells: (selectedZones || []).map((z: any, i: number) => {
        const p = z?.properties ?? z ?? {};
        return {
          row: p.row ?? p.gridRow ?? i, col: p.col ?? p.gridCol ?? 0,
          task_id: z?.task_id ?? p.task_id, ...(z?.id ? { zone_id: z.id } : {}),
        };
      }),
    });
    if (grid?.id) { setGridId(grid.id); setGridNotice(tc("gridSaved")); }
    else setGridNotice(tc("gridSaveFailed"));
  }, [getIdToken, gridId, currentCityKey, gridType, hexSizeM, selectedZones, gridArea, gridRotationDeg]);

  // ПРОДОВЖЕННЯ: авто-зберігаємо сітку ОДРАЗУ після генерації серії з task_id
  // кожної клітини. Так куплені зони лишаються «золотими» при наступному відкритті
  // (кабінет → Мої сітки → Відкрити) — людина додає сусідні й замовляє лише нові.
  const handleSeriesGenerated = useCallback(
    async (cells: Array<{ row: number; col: number; task_id?: string; zone_id?: string }>) => {
      const px = cellIdPrefix(gridType);
      // ОБ'ЄДНУЄМО з раніше купленими (save_grid замінює масив cells цілком → без
      // мерджу другий заказ стер би перший). Реконструюємо старі з boughtCells.
      const byKey = new Map<string, { row: number; col: number; task_id?: string }>();
      for (const id of boughtCells) {
        const parts = id.split("_");
        const row = Number(parts[parts.length - 2]);
        const col = Number(parts[parts.length - 1]);
        if (Number.isFinite(row) && Number.isFinite(col)) byKey.set(`${row}_${col}`, { row, col });
      }
      for (const c of cells) byKey.set(`${c.row}_${c.col}`, c); // нові (з task_id) перекривають
      const merged = [...byKey.values()];
      // Одразу підсвічуємо щойно згенеровані клітини золотим (не чекаючи reload).
      setBoughtCells(new Set(merged.map((c) => `${px}_${c.row}_${c.col}`)));
      try {
        const token = await getIdToken();
        if (!token) return; // не залогінений — золоті лишаться лише на сесію
        const city = CITIES[currentCityKey];
        const grid = await saveGrid(token, {
          id: gridId || undefined,
          name: `${tCity(currentCityKey)} · ${gridTypeLabel(gridType)}`,
          city: currentCityKey,
          center: city?.center,
          grid_type: gridType,
          hex_size_m: hexSizeM,
          bounds: gridArea || city?.bounds,
          rotation_deg: gridRotationDeg,
          cells: merged,
        });
        if (grid?.id) setGridId(grid.id);
      } catch { /* збереження не критичне */ }
    },
    [getIdToken, gridId, currentCityKey, gridType, hexSizeM, gridArea, boughtCells, gridRotationDeg],
  );

  // ── Capture mode (?capture=<templateId>): auto-select the district area and
  // run a real preview generation through the site's own pipeline, so an
  // automated screenshot of the 3D preview produces an authentic gallery image.
  // Harmless for normal users (only triggers when the param is present).
  useEffect(() => {
    try {
      const params = new URLSearchParams(window.location.search);
      const cap = params.get("capture");
      if (!cap) return;
      const tpl = MAP_TEMPLATES.find((t) => t.id === cap);
      if (!tpl) return;
      const [lat, lon] = tpl.center;
      const s = tpl.span;
      const lonPad = s / Math.max(Math.cos((lat * Math.PI) / 180), 0.2);
      const north = lat + s, south = lat - s, east = lon + lonPad, west = lon - lonPad;
      (async () => {
        const L = await import("leaflet");
        const bounds = new L.LatLngBounds([south, west], [north, east]);
        setSelectedArea(bounds as any);
        const { api } = await import("@/lib/api");
        const req: any = {
          north, south, east, west,
          road_width_multiplier: 0.8, road_height_mm: 0.5, road_embed_mm: 0.3,
          building_min_height: 5.0, building_height_multiplier: 1.8,
          building_foundation_mm: 0.6, building_embed_mm: 0.2,
          water_depth: 2.0, terrain_enabled: false, terrain_z_scale: 1.0,
          terrain_base_thickness_mm: 1.3, terrain_resolution: 180, terrarium_zoom: 15,
          flatten_buildings_on_terrain: false, flatten_roads_on_terrain: false,
          export_format: "3mf", model_size_mm: 80, context_padding_m: 400.0,
          is_ams_mode: false, flat_plate_mode: false, preview_mode: true,
          preview_include_base: true, preview_include_roads: true,
          preview_include_buildings: true, preview_include_water: true, preview_include_parks: true,
        };
        setGenerating(true);
        const r = await api.generateModel(req);
        setTaskGroup(r.task_id, [r.task_id]);
        setActiveTaskId(r.task_id);
      })();
    } catch {/* ignore */}
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Expose a readiness flag for automated capture once the model is ready.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    if (!params.get("capture")) return;
    (window as any).__captureReady = Boolean(downloadUrl);
    if (downloadUrl) document.body.setAttribute("data-capture-ready", "1");
  }, [downloadUrl]);

  // Preselect city from ?template= (links from landing-page template gallery)
  useEffect(() => {
    try {
      const params = new URLSearchParams(window.location.search);
      const tpl = params.get("template");
      const cityParam = params.get("city");
      if (cityParam && CITIES[cityParam]) { setCurrentCityKey(cityParam); return; }
      if (!tpl) return;
      // template ids look like "kyiv-podil" → map prefix to city key
      const prefix = tpl.split("-")[0].toLowerCase();
      const map: Record<string, string> = {
        kyiv: "Kyiv", lviv: "Lviv", odesa: "Odesa", kharkiv: "Kharkiv", dnipro: "Dnipro",
        chernivtsi: "Chernivtsi", ivano: "IvanoFrankivsk", uzhhorod: "Uzhhorod",
      };
      if (map[prefix]) setCurrentCityKey(map[prefix]);
    } catch {/* ignore */}
  }, []);

  // Відновлюємо task_id з localStorage після refresh — ЛИШЕ задачі мап (не брелків,
  // бо /create і /keychains ділять той самий ключ; інакше відновили б чужу задачу).
  useEffect(() => {
    // try/catch: битий/підмінений 3dmap_task_ids (частковий запис при краші браузера,
    // storage-eviction) кинув би JSON.parse синхронно у useEffect → error boundary /
    // розмонтування дерева = порожній /create при завантаженні. /keychains уже
    // обгорнутий так само; Array.isArray-guard проти valid-non-array (напр. "5").
    try {
      const savedGroupId = localStorage.getItem("3dmap_task_group_id");
      const savedTaskIds = localStorage.getItem("3dmap_task_ids");
      const savedProduct = localStorage.getItem("3dmap_task_product");
      if (savedGroupId && !taskGroupId && savedProduct !== "keychain") {
        const parsed = savedTaskIds ? JSON.parse(savedTaskIds) : [savedGroupId];
        const ids = Array.isArray(parsed) ? parsed : [savedGroupId];
        setTaskGroup(savedGroupId, ids, "map");
        setGenerating(true);
      }
    } catch { /* ignore corrupt saved task ids */ }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ГЕО за IP: коли НЕМАЄ збереженого чернетки і не обрано зону/сітку — центруємо
  // карту на країну користувача (Cloudflare CF-IPCountry → /api/geo). Для UA (або
  // помилки fetch) нічого не робимо — Київ лишається дефолтом. Локаль/мова не
  // чіпаємо (next-intl сам визначає). Подію шлемо З ЗАТРИМКОЮ (~900мс), бо
  // MapSelector імпортується динамічно й слухач реєструється після маунта.
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      if (localStorage.getItem("monadruk:draft:create")) return; // є чернетка → не чіпаємо
      const st = useGenerationStore.getState();
      if (st.selectedArea || st.showHexGrid || (st.selectedZones || []).length > 0) return; // вже щось обрано
      // grid=/template=/city= з URL теж означають намір — не перебиваємо центр
      const params = new URLSearchParams(window.location.search);
      if (params.get("grid") || params.get("template") || params.get("city") || params.get("capture")) return;
      (async () => {
        try {
          const { API_BASE_URL } = await import("@/lib/api");
          const r = await fetch(`${API_BASE_URL}/api/geo`).then((res) => (res.ok ? res.json() : null));
          if (!r || r.country === "UA") return; // UA → Київ за дефолтом
          const lat = Number(r.lat), lon = Number(r.lng);
          if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
          // ЗАТРИМКА: даємо динамічному MapSelector змонтуватись і повісити слухач.
          timer = setTimeout(() => {
            try {
              // centerOnly: лише центруємо карту на країні відвідувача (zoom рівня
              // країни), БЕЗ авто-розміщення зони виділення в столиці.
              window.dispatchEvent(new CustomEvent("monadruk:map-goto", { detail: { lat, lon, label: r.label, centerOnly: true, zoom: 6 } }));
            } catch { /* no-op */ }
          }, 900);
        } catch { /* graceful no-op */ }
      })();
    } catch { /* graceful no-op */ }
    return () => { if (timer) clearTimeout(timer); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleCancelTask = async () => {
    if (!taskGroupId) return;
    try {
      const { api } = await import("@/lib/api");
      await api.cancelTask(taskGroupId);
    } catch {/* ignore */}
    setTaskGroup(null);
    setGenerating(false);
    localStorage.removeItem("3dmap_task_group_id");
    localStorage.removeItem("3dmap_task_ids");
  };

  // UX: момент успіху — щойно модель готова, на мобілці плавно прокручуємо до 3D
  // (одноекранно превʼю вже на сторінці нижче — просто наводимо на нього).
  const prevDownloadRef = useRef<string | null>(null);
  useEffect(() => {
    if (downloadUrl && !prevDownloadRef.current && typeof window !== "undefined" && window.innerWidth < 1024) {
      document.getElementById("panel-preview")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    prevDownloadRef.current = downloadUrl ?? null;
  }, [downloadUrl]);

  const currentCity = CITIES[currentCityKey];
  const hasMapSelection = Boolean(selectedArea);
  const zoneCount = selectedZones.length;
  // Живий підсумок ціни серії (UX-аудит): раніше сума за клітини зʼявлялась лише
  // на фінальній CTA-кнопці — пізній sticker-shock при 3×3+ сітці. Показуємо
  // «N зон × ціна = сума» одразу в степ-банері, щойно обрано ≥1 клітину.
  const isEuLocale = locale !== "uk";
  const dispZonePrice = (uah: number) => (isEuLocale ? `€${mapPriceEur(uah)}` : `${uah} ₴`);
  const nearestSimpleSize = SIMPLE_SIZES.reduce(
    (best, s) => (Math.abs(s.mm - (modelSizeMm || 80)) < Math.abs(best.mm - (modelSizeMm || 80)) ? s : best),
    SIMPLE_SIZES[0],
  );
  const zonesPriceSummary = zoneCount > 0
    ? tc("zonesPriceSummary", {
        count: zoneCount,
        perZone: dispZonePrice(nearestSimpleSize.price),
        total: dispZonePrice(nearestSimpleSize.price * zoneCount),
      })
    : null;
  // Авто-перехід на 3D-рендер у мить старту генерації (rising edge). Назад на
  // карту — лише ручним перемикачем (deps не міняються → ефект не вертає).
  const prevGenStageRef = useRef(false);
  useEffect(() => {
    if (isGenerating && !prevGenStageRef.current) setStageView("render");
    prevGenStageRef.current = isGenerating;
  }, [isGenerating]);
  const canShowRender = isGenerating || Boolean(downloadUrl);
  const switchStage = (v: "map" | "render") => {
    if (v === "render" && !canShowRender) return; // нема що показувати ще
    setStageView(v);
    // leaflet/three перерахують розмір, коли контейнер знову став видимим
    if (typeof window !== "undefined") {
      setTimeout(() => { try { window.dispatchEvent(new Event("resize")); } catch { /* no-op */ } }, 80);
    }
  };
  const selectionLabel = showHexGrid
    ? zoneCount > 0
      ? tc("zonesReady", { count: zoneCount })
      : tc("pickZonesOnMap")
    : hasMapSelection
      ? tc("areaReady")
      : tc("markOneArea");
  const statusLabel = isGenerating
    ? `${progress}% • ${progress >= 90 ? tc("almostDone") : (status || tc("generationInProgress"))}`
    : downloadUrl
      ? tc("fileReady")
      : tc("readyToConfigure");

  // ОДНОЕКРАННО (як /keychains): панелі НЕ перемикаються табами, а стоять усі
  // разом у скрол-колонці на мобільному. Порядок: карта → налаштування →
  // превʼю (логічний потік). На десктопі налаштування у власному aside (тут
  // lg:hidden), а карта+превʼю стоять у правій колонці. Степер лише прокручує.
  // Карта і рендер тепер ОДНА сцена (взаємовиключні) → обидва order-1, налаштування
  // (мобільні) йдуть ПІСЛЯ сцени. Перемикач — order-0 (над сценою).
  const mapPanelClasses = "order-1 flex";
  const previewPanelClasses = "order-1 flex";
  const settingsPanelClasses = "order-2 flex lg:hidden";

  return (
    <div className="min-h-[100dvh] bg-transparent">
      {/* UX: тур лише ДО першої генерації — інакше перекривав 3D-результат
          і прогрес. Текст без «панелі зліва» (на мобілці її нема — там таби). */}
      {/* Тур — лише ПІСЛЯ закриття вступного блоку: два онбординги нараз
          (блок «що вийде» + плаваюча підказка) захаращували перший екран. */}
      {/* GUIDED: тур не потрібен — сценарний флоу сам веде по кроках. */}
      {!guided && !isGenerating && !downloadUrl && !introVisible && (
        <OnboardingTour
          storageKey="onb_create_v1"
          steps={[
            { title: tc("tourCityTitle"), body: tc("tourCityBody") },
            { title: tc("tourFrameTitle"), body: tc("tourFrameBody") },
            { title: tc("tourGenerateTitle"), body: tc("tourGenerateBody") },
          ]}
        />
      )}
      <div id="main-content" tabIndex={-1} className="mx-auto flex min-h-[100dvh] max-w-[1760px] flex-col px-3 pb-24 pt-3 sm:px-4 lg:px-6 lg:pb-6">
        <header className="sticky top-0 z-30 rounded-[18px] border border-[var(--surface-border)] bg-[rgba(252,249,243,0.92)] px-3 py-2.5 shadow-[0_10px_30px_rgba(31,41,55,0.07)] backdrop-blur lg:static lg:px-4">
          <div className="flex flex-wrap items-center gap-2.5">
            {/* Back to home (prominent, always visible). Суцільний білий +
                сильніша рамка + легка тінь → контрастніше на кремовій шапці;
                min-h-[40px] = зручний тап-таргет. */}
            <Link
              href="/"
              title={tc("backHomeTitle")}
              className="inline-flex min-h-[40px] items-center gap-1.5 rounded-full border border-[var(--surface-border)] bg-white px-3.5 py-2 text-[13px] font-semibold text-[var(--text-secondary)] shadow-[0_2px_8px_rgba(15,23,42,0.05)] transition hover:border-[rgba(11,92,87,0.4)] hover:text-[var(--text-primary)]"
            >
              <HomeIcon size={15} /> <span className="hidden sm:inline">{tc("backHome")}</span>
            </Link>
            <span className="hidden h-5 w-px bg-[var(--surface-border)] sm:block" />
            <h1 className="font-title text-base font-semibold tracking-tight text-[var(--text-primary)] sm:text-lg">
              {tc("title")}
            </h1>

            {/* Controls (compact toolbar, right-aligned). Вибір міста + статус
                «Оберіть зони» прибрано зі шапки (власник: захаращено) — місто тепер
                у лівій панелі (там, де райони), статус і так видно в панелі/на барі. */}
            <div className="ml-auto flex flex-wrap items-center gap-2">
              {/* БРЕЛОК: помітна заливна акцент-кнопка; підпис ВИДНО і на мобільному
                  (раніше hidden sm:inline → на телефоні лишалась лише іконка й
                  користувач не знав про брелки). */}
              <Link
                href="/keychains"
                className="inline-flex min-h-[40px] items-center gap-1.5 rounded-full border border-[var(--accent-strong)] bg-[var(--accent-strong)] px-3.5 py-2 text-[13px] font-semibold text-white shadow-[0_2px_8px_rgba(11,92,87,0.25)] transition hover:bg-[rgba(11,92,87,0.92)]"
              >
                <KeyRound size={15} /> <span>{tc("keychain")}</span>
              </Link>
              {/* Кабінет лишаємо компактним (іконка-онлі на мобільному). */}
              <Link
                href="/account"
                title={tc("account")}
                className="inline-flex min-h-[40px] items-center gap-1.5 rounded-full border border-[var(--surface-border)] bg-white px-3.5 py-2 text-[13px] font-semibold text-[var(--text-secondary)] shadow-[0_2px_8px_rgba(15,23,42,0.05)] transition hover:border-[rgba(11,92,87,0.4)] hover:text-[var(--text-primary)]"
              >
                <User size={15} /> <span className="hidden sm:inline">{tc("account")}</span>
              </Link>
            </div>
          </div>

          {/* Навігацію спрощено: степер КРОК 1/2/3 прибрано (зайвий chrome —
              сцена карта⇄рендер уже самопояснювана). Раніше тут був ще й
              дубль-ряд табів — теж прибрано. Нижній бар = StickyActionBar. */}
        </header>

        {/* Вступ «ось що вийде» — ДО інструмента. Воронка показувала обвал
            view→area (71%): людина бачила чужу карту Києва й не розуміла,
            що отримає. Зникає назавжди після першої взаємодії. */}
        {/* GUIDED: вступний блок теж зайвий — крок 1 сам показує «що вийде». */}
        {!guided && !isGenerating && !downloadUrl && (
          <ConstructorIntro
            visible={introVisible}
            onDismiss={dismissIntro}
            variant="map"
            photos={["map-1", "panno-1", "map-3", "map-2"]}
            scrollToId="panel-map"
            priceFrom={MAP_SIZE_PRICES_UAH[55]}
          />
        )}

        <div className="mt-3 flex flex-1 flex-col gap-3 lg:min-h-0 lg:grid lg:grid-cols-[380px,minmax(0,1fr)]">
          {/* GUIDED: aside стає видимим і на мобільному (order-2 = ПІД картою),
              всередині — ScenarioFlow (монтується РІВНО РАЗ — локальний step-стан
              не розсинхронізовується, на відміну від панелей ×2) + прихована
              «машинна» копія SimpleControlPanel: тримає полінг статусу задачі,
              OrderDialog (слухач monadruk:open-order → портал у body) і запуск
              генерації по події monadruk:guided-generate. Візуально — нуль
              контролів (display:none). */}
          <aside id="panel-settings" className={guided ? "order-2 min-h-0 lg:order-none lg:block" : "hidden min-h-0 lg:block"}>
            {guided ? (
              <>
                <ScenarioFlow onExitGuided={() => setGuided(false)} />
                <div className="hidden" aria-hidden="true">
                  <SimpleControlPanel
                    availableCities={CITIES}
                    selectedCityKey={currentCityKey}
                    onCityChange={handleCityChange}
                    cityLabel={tCity}
                    showStickyBar={false}
                    listenGuidedGenerate
                    onSeriesGenerated={handleSeriesGenerated}
                  />
                </div>
              </>
            ) : (
            <div className="flex h-full flex-col overflow-hidden rounded-[30px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_22px_70px_rgba(15,23,42,0.08)] backdrop-blur">
              <div className="flex shrink-0 items-center justify-between gap-2 border-b border-[var(--surface-border)] px-4 py-3">
                <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
                  {proMode ? tc("expertMode") : tc("quickCreate")}
                </span>
                <div className="flex items-center gap-1.5">
                  {/* Назад у сценарний вхід (guided) — маленька кнопка в шапці панелі. */}
                  <button
                    type="button"
                    onClick={() => setGuided(true)}
                    className="rounded-full border border-[var(--surface-border)] bg-white/80 px-2.5 py-1 text-[11px] font-semibold text-[var(--text-secondary)] transition hover:border-[rgba(11,92,87,0.4)] hover:text-[var(--text-primary)]"
                  >
                    {tSc("backToGuided")}
                  </button>
                <div className="flex items-center gap-1 rounded-full border border-[var(--surface-border)] bg-white/80 p-0.5 text-xs">
                  <button type="button" onClick={() => toggleProMode(false)}
                    className={`rounded-full px-3 py-1 font-semibold transition ${!proMode ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)]"}`}>{tc("modeSimple")}</button>
                  <button type="button" onClick={() => toggleProMode(true)}
                    className={`rounded-full px-3 py-1 font-semibold transition ${proMode ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)]"}`}>{tc("modePro")}</button>
                </div>
                </div>
              </div>
              <div className="min-h-0 flex-1 overflow-hidden">
                {proMode ? (
                  <ControlPanel
                    showHexGrid={showHexGrid}
                    setShowHexGrid={setShowHexGridPersist}
                    selectedZones={selectedZones}
                    setSelectedZones={setSelectedZones}
                    gridType={gridType}
                    setGridType={setGridType}
                    hexSizeM={hexSizeM}
                    setHexSizeM={setHexSizeM}
                    availableCities={CITIES}
                    selectedCityKey={currentCityKey}
                    onCityChange={handleCityChange}
                    onSeriesGenerated={handleSeriesGenerated}
                  />
                ) : (
                  // Місто вибирається у шапці (завжди видно) — у панелі дубль
                  // прибрано, щоб не плодити два однакові селектори.
                  <SimpleControlPanel
                    availableCities={CITIES}
                    selectedCityKey={currentCityKey}
                    onCityChange={handleCityChange}
                    cityLabel={tCity}
                    onAdvanced={() => toggleProMode(true)}
                    showStickyBar={false}
                    onSeriesGenerated={handleSeriesGenerated}
                  />
                )}
              </div>
            </div>
            )}
          </aside>

          {/* map2model-стиль: ОДНА велика сцена на всю ширину — карта АБО 3D-рендер,
              перемикач зверху. Раніше карта+превʼю тіснились поруч (≈половина кожне). */}
          <section className="flex flex-1 flex-col gap-3 lg:min-h-0">
            {/* На МОБІЛЬНОМУ карта у звичайному потоці (НЕ sticky) — раніше
                `sticky top-[60px]` «приліплював» високу карту, і вона перекривала
                контент + заважала прокрутити/обрати (скарга власника). Прилипання
                лишаємо ЛИШЕ на десктопі (lg:sticky), де є окремий aside-сайдбар. */}
            <div className="order-1 flex min-h-0 flex-col gap-3 lg:sticky lg:top-4 lg:z-auto">
              {/* ДЕСКТОП: sticky top-4 → карта «їде» за користувачем поки гортає aside. */}
            {/* Перемикач сцени: Карта ⇄ 3D-модель (рендер доступний після генерації) */}
            <div className="order-0 flex shrink-0 items-center gap-1 rounded-full border border-[var(--surface-border)] bg-[var(--surface-panel)] p-1 shadow-[0_8px_24px_rgba(15,23,42,0.06)] backdrop-blur">
              <button
                type="button"
                onClick={() => switchStage("map")}
                aria-pressed={stageView === "map"}
                data-testid="stage-map"
                className={`flex flex-1 items-center justify-center gap-1.5 rounded-full px-4 py-2 text-sm font-semibold transition ${stageView === "map" ? "bg-[var(--accent-strong)] text-white shadow" : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"}`}
              >
                🗺 {tc("stageMap")}
              </button>
              <button
                type="button"
                onClick={() => switchStage("render")}
                aria-pressed={stageView === "render"}
                disabled={!canShowRender}
                data-testid="stage-render"
                title={canShowRender ? undefined : tc("stageRenderLocked")}
                className={`flex flex-1 items-center justify-center gap-1.5 rounded-full px-4 py-2 text-sm font-semibold transition ${
                  stageView === "render"
                    ? "bg-[var(--accent-strong)] text-white shadow"
                    : canShowRender
                      ? "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                      : "cursor-not-allowed text-[var(--text-secondary)] opacity-45"
                }`}
              >
                🧊 {tc("stageRender")}{isGenerating ? ` · ${progress}%` : ""}
              </button>
            </div>
            {/* ПОСТІЙНИЙ КРОК-БАНЕР (UX-аудит P0): selectionLabel раніше рахувався,
                але НІДЕ не рендерився — новий користувач не бачив «що робити далі».
                Тонкий рядок над сценою: поточний крок + наступна дія. Видно на
                мобільному І десктопі; ховається лише під час генерації (там прогрес). */}
            {!isGenerating && !guided && (
              <div className="order-0 flex shrink-0 items-center gap-2 rounded-full border border-[rgba(11,92,87,0.25)] bg-[rgba(15,118,110,0.07)] px-3.5 py-1.5" aria-live="polite">
                <span className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[var(--accent-strong)] text-[11px] font-bold text-white">
                  {downloadUrl ? "3" : (hasMapSelection || zoneCount > 0) ? "2" : "1"}
                </span>
                <p className="min-w-0 truncate text-[12.5px] font-semibold text-[var(--text-primary)]">
                  {downloadUrl
                    ? tc("stepReadyOrder")
                    : (showHexGrid && zonesPriceSummary)
                      ? zonesPriceSummary
                      : (hasMapSelection || zoneCount > 0)
                        ? tc("stepNextSize")
                        : selectionLabel}
                </p>
              </div>
            )}
            <div id="panel-map" className={`${mapPanelClasses} ${stageView === "map" ? "" : "hidden"}`}>
              {/* Карта — головна взаємодія: на десктопі вся сцена (перемикач +
                  картка) ВЛІЗАЄ в один екран (calc під шапку) → без скролу
                  сторінки. На мобільному лишаємо min-h і дозволяємо скрол. */}
              <div className="flex min-h-[360px] flex-1 flex-col overflow-hidden rounded-[30px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_22px_70px_rgba(15,23,42,0.08)] backdrop-blur lg:h-[calc(100dvh-140px)] lg:min-h-0 lg:max-h-[calc(100dvh-110px)]">
                {/* РЕДИЗАЙН: керування сіткою більше НЕ стоїть стосом над картою
                    (з'їдало вертикаль) — карта займає всю картку, а контроли
                    плавають компактною карткою у ПРАВОМУ ВЕРХНЬОМУ куті прямо НА
                    мапі. Сегмент «Одна ділянка / Серія» доступний В ОБОХ режимах;
                    форма клітинок + збереження — лише у режимі сітки. */}

                {/* ВИБІР ФОРМИ ФІГУРИ — лишається У ПОТОЦІ над картою ЛИШЕ для
                    одиночної ділянки (single-режим не чіпаємо). GUIDED ховає
                    (сценарний вхід = мінімум контролів; форма = у повному UI). */}
                {!showHexGrid && !guided && (
                  <div role="radiogroup" aria-label={tc("shapeFieldLabel")} className="mx-4 mt-3 flex items-center gap-1.5 overflow-x-auto pb-1 [-ms-overflow-style:none] [scrollbar-width:none] sm:flex-wrap sm:overflow-visible sm:pb-0">
                    <span className="shrink-0 text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{tc("shapeFieldLabel")}</span>
                    {FIGURE_SHAPES.map((sh) => (
                      <button
                        key={sh.id}
                        type="button"
                        role="radio"
                        aria-checked={figureShape === sh.id}
                        onClick={() => setFigureShape(sh.id)}
                        className={`flex min-h-[40px] shrink-0 items-center whitespace-nowrap rounded-full border px-3.5 py-2 text-xs font-semibold transition ${
                          figureShape === sh.id
                            ? "border-[rgba(11,92,87,0.45)] bg-[rgba(15,118,110,0.14)] text-[var(--text-primary)]"
                            : "border-[var(--surface-border)] bg-white/80 text-[var(--text-secondary)] hover:border-[rgba(11,92,87,0.3)]"
                        }`}
                      >
                        {sh.label}
                      </button>
                    ))}
                    {/* Прямокутник: за замовчуванням гострі 90° кути; тумблер вмикає заокруглення. */}
                    {figureShape === "rounded" && (
                      <button
                        type="button"
                        aria-pressed={roundCorners}
                        data-testid="round-corners-toggle"
                        onClick={() => setRoundCorners((v) => !v)}
                        title={tc("roundCornersHint")}
                        className={`flex min-h-[40px] shrink-0 items-center whitespace-nowrap rounded-full border px-3.5 py-2 text-xs font-semibold transition ${
                          roundCorners
                            ? "border-[rgba(11,92,87,0.45)] bg-[rgba(15,118,110,0.14)] text-[var(--text-primary)]"
                            : "border-[var(--surface-border)] bg-white/80 text-[var(--text-secondary)] hover:border-[rgba(11,92,87,0.3)]"
                        }`}
                      >
                        {roundCorners ? "✓ " : ""}{tc("roundCorners")}
                      </button>
                    )}
                  </div>
                )}

                {/* МАПА + плаваючі оверлеї. relative — щоб абсолютно-позиційовані
                    контроли лягали поверх карти. МОБ: 42dvh (було 60dvh — карта
                    з'їдала весь перший екран і ховала розмір/ціну/CTA нижче згину;
                    UX-аудит P0 «користувачам нічого не ясно»). */}
                <div className="relative flex min-h-[42dvh] flex-1 flex-col bg-[rgba(255,255,255,0.55)] p-2 sm:min-h-[460px] sm:p-3 lg:min-h-0">
                  {showHexGrid ? (
                    <HexagonalGrid
                      // boughtCells.size у ключі: коли куплені клітини
                      // підвантажились, грід перемальовується з золотими.
                      key={`hex-grid-${currentCityKey}-${boughtCells.size}`}
                      bounds={currentCity.bounds}
                      onZonesSelected={setSelectedZones}
                      gridType={gridType}
                      hexSizeM={hexSizeM}
                      rotationDeg={gridRotationDeg}
                      onAreaChange={setGridArea}
                      initialArea={gridArea}
                      boughtCells={boughtCells}
                    />
                  ) : (
                    <div className="h-full overflow-hidden rounded-[24px]">
                      <MapSelector center={currentCity.center} keychainCrop={mapCrop} />
                    </div>
                  )}

                  {/* ПІДКАЗКА «як обрати зону» — компактна плаваюча картка у правому
                      верху НА мапі. Лише моб/планшет (lg:hidden): на десктопі картка
                      керування й так стоїть у правому верху і містить ту саму підказку. */}
                  {showHexGrid && (
                    <div className="pointer-events-none absolute right-2 top-2 z-[500] w-[160px] max-w-[58%] rounded-[12px] border border-[var(--surface-border)] bg-[var(--surface-panel)]/95 px-2.5 py-1.5 shadow-[0_8px_24px_rgba(15,23,42,0.12)] backdrop-blur lg:hidden">
                      <p className="text-[11px] font-semibold text-[var(--text-primary)]">{tc("gridSelectHintTitle")}</p>
                      <p className="mt-0.5 text-[10px] leading-tight text-[var(--text-secondary)]">{tc("pickAdjacentZones")}</p>
                    </div>
                  )}

                  {/* ПЛАВАЮЧА КАРТКА КЕРУВАННЯ (правий верх НА мапі): сегмент
                      ділянка/серія завжди; форма клітинок + збереження + підказка
                      лише у режимі сітки. z-[500] — над leaflet-панелями. */}
                  {/* МОБ: оверлей у потоці ПІД картою (не плаває), щоб НЕ накладатись
                      на тулбар сітки (лівий-верх). ДЕСКТОП: плаває у правому верху НА мапі. */}
                  {/* ДЕСКТОП top-right: у режимі «Одна ділянка» MapSelector малює
                      кластер повороту (↺ 0° ↻) теж у right-2 top-2 → накладались
                      (скарга «елементи перекривають»). У single-zone опускаємо панель
                      нижче кластера повороту (lg:top-[80px], бо кластер ~46px + різні
                      positioned-ancestors); у серії повороту немає → лишаємо top-2. */}
                  {/* GUIDED ховає плаваючу картку керування (Ділянка/Серія + сітка) —
                      сценарний вхід працює лише з одиночною зоною. */}
                  {!guided && (
                  <div className={`relative order-2 mt-2 w-full z-[500] space-y-1.5 rounded-[16px] border border-[var(--surface-border)] bg-[var(--surface-panel)]/95 px-2.5 py-2 shadow-[0_8px_24px_rgba(15,23,42,0.12)] backdrop-blur lg:absolute lg:order-none lg:right-2 lg:mt-0 lg:w-[220px] lg:max-w-[calc(100%-1rem)] ${showHexGrid ? "lg:top-2" : "lg:top-[80px]"}`}>
                    {/* Сегмент-контрол: дві пігулки в один ряд (доступно в ОБОХ режимах). */}
                    <div className="flex items-center gap-1" role="tablist" aria-label={tc("selectionModeAria")}>
                      <button
                        type="button"
                        role="tab"
                        aria-selected={!showHexGrid}
                        onClick={() => { setShowHexGridPersist(false); }}
                        className={`flex min-h-[44px] flex-1 items-center justify-center rounded-full border px-2 py-2 text-center text-[13px] font-semibold transition ${
                          !showHexGrid
                            ? "border-[rgba(11,92,87,0.5)] bg-[rgba(15,118,110,0.12)] text-[var(--text-primary)]"
                            : "border-[var(--surface-border)] bg-white/80 text-[var(--text-secondary)] hover:border-[rgba(11,92,87,0.3)]"
                        }`}
                      >
                        {tc("singleAreaTab")}
                      </button>
                      <button
                        type="button"
                        role="tab"
                        aria-selected={showHexGrid}
                        onClick={() => { setShowHexGridPersist(true); }}
                        className={`flex min-h-[44px] flex-1 items-center justify-center rounded-full border px-2 py-2 text-center text-[13px] font-semibold transition ${
                          showHexGrid
                            ? "border-[rgba(11,92,87,0.5)] bg-[rgba(15,118,110,0.12)] text-[var(--text-primary)]"
                            : "border-[var(--surface-border)] bg-white/80 text-[var(--text-secondary)] hover:border-[rgba(11,92,87,0.3)]"
                        }`}
                      >
                        {tc("seriesTab")}
                      </button>
                    </div>

                    {/* Лише у режимі сітки: форма клітинок (без «Кола») + збереження + підказка. */}
                    {showHexGrid && (
                      <>
                        <div className="grid grid-cols-2 gap-1" role="radiogroup" aria-label={tc("cellShapeAria")}>
                          {([
                            ["hexagonal", tc("gridHexLabel"), tc("gridHexHint")],
                            ["square", tc("gridSquareLabel"), tc("gridSquareHint")],
                          ] as Array<["hexagonal" | "square", string, string]>).map(([gt, label, hint]) => (
                            <button
                              key={gt}
                              type="button"
                              role="radio"
                              aria-checked={gridType === gt}
                              onClick={() => setGridType(gt)}
                              title={hint}
                              className={`flex min-h-[44px] items-center justify-center rounded-[12px] border px-2 py-2 text-center text-[13px] font-semibold transition ${
                                gridType === gt
                                  ? "border-[rgba(11,92,87,0.5)] bg-[rgba(15,118,110,0.12)] text-[var(--text-primary)]"
                                  : "border-[var(--surface-border)] bg-white/80 text-[var(--text-secondary)] hover:border-[rgba(11,92,87,0.3)]"
                              }`}
                            >
                              {label}
                            </button>
                          ))}
                        </div>
                        {/* Поворот сітки: повзунок 0–90°. Обертає клітини довкола
                            центру → можна вирівняти серію під діагональну річку/район.
                            Коміт на відпускання (важка сітка не лагає при тягненні). */}
                        <div className="flex items-center gap-2">
                          <span className="shrink-0 text-[11px] font-medium text-[var(--text-secondary)]">{tc("gridRotateLabel")}</span>
                          <input
                            type="range"
                            min={0}
                            max={90}
                            step={1}
                            value={pendingRot}
                            onChange={(e) => setPendingRot(Number(e.target.value))}
                            onPointerUp={() => setGridRotationDeg(pendingRot)}
                            onTouchEnd={() => setGridRotationDeg(pendingRot)}
                            onKeyUp={() => setGridRotationDeg(pendingRot)}
                            aria-label={tc("gridRotateLabel")}
                            className="h-2 flex-1 cursor-pointer accent-[var(--accent-strong,#0f766e)]"
                          />
                          <span className="w-8 shrink-0 text-right text-[11px] font-semibold tabular-nums text-[var(--text-primary)]">{pendingRot}°</span>
                          {pendingRot !== 0 && (
                            <button
                              type="button"
                              onClick={() => { setPendingRot(0); setGridRotationDeg(0); }}
                              title={tc("gridRotateReset")}
                              aria-label={tc("gridRotateReset")}
                              className="shrink-0 rounded-full border border-[var(--surface-border)] px-1.5 py-0.5 text-[10px] text-[var(--text-secondary)] transition hover:border-[rgba(11,92,87,0.3)]"
                            >
                              ↺0
                            </button>
                          )}
                        </div>
                        <button
                          type="button"
                          onClick={handleSaveGrid}
                          className="flex min-h-[44px] w-full items-center justify-center rounded-full border border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.1)] px-2 py-2 text-[13px] font-semibold text-[var(--text-primary)] transition hover:bg-[rgba(15,118,110,0.18)]"
                        >
                          {tc("saveGridButton")}
                        </button>
                        <p className="text-[10px] leading-tight text-[var(--text-secondary)]">{tc("saveGridHint")}</p>
                        <p className="text-[11px] font-medium leading-tight text-[var(--text-primary)]">{tc("pickAdjacentZones")}</p>
                      </>
                    )}
                  </div>
                  )}

                  {/* gridNotice — плаваючий тост ВНИЗУ по центру карти (не штовхає мапу). */}
                  {gridNotice && (
                    <div className="absolute bottom-2 left-1/2 z-[500] -translate-x-1/2 max-w-[calc(100%-1rem)] rounded-[14px] border border-[rgba(11,92,87,0.3)] bg-[var(--surface-panel)]/95 px-3 py-2 text-center text-[12px] text-[var(--text-primary)] shadow-[0_8px_24px_rgba(15,23,42,0.12)] backdrop-blur">
                      {gridNotice}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {stageView === "render" && (
            <div id="panel-preview" className={previewPanelClasses}>
              <div className="flex min-h-[320px] flex-1 flex-col overflow-hidden rounded-[30px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_22px_70px_rgba(15,23,42,0.08)] backdrop-blur lg:h-[calc(100dvh-140px)] lg:min-h-0 lg:max-h-[calc(100dvh-110px)]">
                <div className="flex items-start justify-between gap-4 border-b border-[var(--surface-border)] px-4 py-4 sm:px-5">
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[var(--text-secondary)]">
                      {tc("preview3d")}
                    </p>
                    <h2 className="mt-1 font-title text-xl font-semibold text-[var(--text-primary)]">
                      {tc("previewSubtitle")}
                    </h2>
                  </div>

                  <div className="rounded-[18px] border border-[var(--surface-border)] bg-white/80 px-3 py-2 text-right">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
                      {tc("statusHeading")}
                    </div>
                    <div className="mt-1 flex items-center justify-end gap-2">
                      <span className="text-sm font-semibold text-[var(--text-primary)]">{statusLabel}</span>
                      {isGenerating && taskGroupId && (
                        <button
                          onClick={handleCancelTask}
                          className="inline-flex items-center gap-1 rounded-full bg-red-100 px-2 py-0.5 text-[10px] font-semibold text-red-700 hover:bg-red-200 transition-colors"
                          title={tc("cancelTitle")}
                        >
                          <X size={10} /> {tc("cancel")}
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                <div className="min-h-[420px] flex-1 p-2 sm:p-3 lg:min-h-0">
                  <div className="h-full overflow-hidden rounded-[24px] border border-[rgba(15,23,42,0.12)]">
                    <Preview3D />
                  </div>
                </div>

                {/* UX-FIX (воронка generate→order): екран успіху раніше НЕ мав
                    жодної кнопки замовлення — людина генерувала, качала й ішла
                    (2/3 не відкривали форму). CTA прямо в картці превʼю, коли
                    модель готова; відкриває OrderDialog через глобальну подію
                    (той самий канал, що «Швидкий статус»). */}
                {/* У guided НЕ показуємо (v2): панель ScenarioFlow уже має бронзову
                    «Замовити друк · ціна» + рекап — два однакові CTA на екрані
                    виглядали як баг. У повному режимі CTA лишається (воронка). */}
                {downloadUrl && !isGenerating && !guided && (
                  <div className="border-t border-[var(--surface-border)] px-4 py-3 sm:px-5">
                    <button
                      type="button"
                      onClick={() => window.dispatchEvent(new Event("monadruk:open-order"))}
                      className="inline-flex w-full items-center justify-center gap-2 rounded-full bg-[var(--bronze,#8E6B3D)] px-6 py-3.5 text-[15px] font-semibold text-white shadow-[0_8px_24px_rgba(142,107,61,0.35)] transition hover:brightness-110"
                    >
                      <ShoppingBag className="h-5 w-5" /> {tc("previewOrderCta")}
                    </button>
                    <p className="mt-1.5 text-center text-[12px] text-[var(--text-secondary)]">
                      {tc("previewOrderHint")}
                    </p>
                  </div>
                )}
              </div>
            </div>
            )}
            </div>{/* /sticky stage wrapper */}

            {/* GUIDED: мобільна копія панелей НЕ рендериться (ScenarioFlow уже
                стоїть під картою через aside order-2; «машинна» копія — в aside). */}
            {!guided && (
            <div id="panel-settings-mobile" className={settingsPanelClasses}>
              <div className="flex min-h-[420px] flex-1 flex-col overflow-hidden rounded-[30px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_22px_70px_rgba(15,23,42,0.08)] backdrop-blur lg:hidden">
                <div className="flex shrink-0 items-center justify-between gap-2 border-b border-[var(--surface-border)] px-4 py-3">
                  <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
                    {proMode ? tc("expertMode") : tc("quickCreate")}
                  </span>
                  <div className="flex items-center gap-1.5">
                    {/* Назад у сценарний вхід (guided). */}
                    <button
                      type="button"
                      onClick={() => setGuided(true)}
                      className="rounded-full border border-[var(--surface-border)] bg-white/80 px-2.5 py-1 text-[11px] font-semibold text-[var(--text-secondary)] transition hover:border-[rgba(11,92,87,0.4)] hover:text-[var(--text-primary)]"
                    >
                      {tSc("backToGuided")}
                    </button>
                  <div className="flex items-center gap-1 rounded-full border border-[var(--surface-border)] bg-white/80 p-0.5 text-xs">
                    <button type="button" onClick={() => toggleProMode(false)}
                      className={`rounded-full px-3 py-1 font-semibold transition ${!proMode ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)]"}`}>{tc("modeSimple")}</button>
                    <button type="button" onClick={() => toggleProMode(true)}
                      className={`rounded-full px-3 py-1 font-semibold transition ${proMode ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)]"}`}>{tc("modePro")}</button>
                  </div>
                  </div>
                </div>
                {proMode ? (
                  <ControlPanel
                    showHexGrid={showHexGrid}
                    setShowHexGrid={setShowHexGridPersist}
                    selectedZones={selectedZones}
                    setSelectedZones={setSelectedZones}
                    gridType={gridType}
                    setGridType={setGridType}
                    hexSizeM={hexSizeM}
                    setHexSizeM={setHexSizeM}
                    availableCities={CITIES}
                    selectedCityKey={currentCityKey}
                    onCityChange={handleCityChange}
                    onSeriesGenerated={handleSeriesGenerated}
                  />
                ) : (
                  <SimpleControlPanel
                    availableCities={CITIES}
                    selectedCityKey={currentCityKey}
                    onCityChange={handleCityChange}
                    cityLabel={tCity}
                    onAdvanced={() => toggleProMode(true)}
                    onSeriesGenerated={handleSeriesGenerated}
                  />
                )}
              </div>
            </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
