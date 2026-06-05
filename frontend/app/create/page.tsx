"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useState, useEffect } from "react";
import { Download, KeyRound, Layers3, Loader2, Map as MapIcon, Settings2, X } from "lucide-react";
import { Preview3D } from "@/components/Preview3D";
import { ControlPanel } from "@/components/ControlPanel";
import { useGenerationStore } from "@/store/generation-store";
import { OnboardingTour } from "@/components/OnboardingTour";
import { WizardSteps } from "@/components/WizardSteps";
import { SimpleControlPanel } from "@/components/SimpleControlPanel";
import { MAP_TEMPLATES } from "@/lib/templates";

type WorkspaceView = "map" | "preview" | "settings";

const MapSelector = dynamic(
  () => import("@/components/MapSelector").then((mod) => ({ default: mod.MapSelector })),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full min-h-[320px] items-center justify-center rounded-[24px] bg-[rgba(255,255,255,0.65)] text-sm text-[var(--text-secondary)]">
        Завантаження карти...
      </div>
    ),
  },
);

const HexagonalGrid = dynamic(() => import("@/components/HexagonalGrid"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full min-h-[320px] items-center justify-center rounded-[24px] bg-[rgba(255,255,255,0.65)] text-sm text-[var(--text-secondary)]">
      Завантаження сітки...
    </div>
  ),
});

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


const CITY_LABELS: Record<string, string> = {
  Kyiv: "Київ", Khmelnytskyi: "Хмельницький", Lviv: "Львів", Odesa: "Одеса",
  Dnipro: "Дніпро", Kharkiv: "Харків", Vinnytsia: "Вінниця", Zaporizhzhia: "Запоріжжя",
  Kryvyi_Rih: "Кривий Ріг", Mykolaiv: "Миколаїв", Poltava: "Полтава", Cherkasy: "Черкаси",
  Chernihiv: "Чернігів", Ternopil: "Тернопіль", IvanoFrankivsk: "Івано-Франківськ",
  Zhytomyr: "Житомир", Sumy: "Суми", Rivne: "Рівне", Lutsk: "Луцьк",
  Uzhhorod: "Ужгород", Chernivtsi: "Чернівці", Kherson: "Херсон", Kropyvnytskyi: "Кропивницький",
};

const WORKSPACE_TABS: Array<{ id: WorkspaceView; label: string; icon: typeof MapIcon }> = [
  { id: "map", label: "Мапа", icon: MapIcon },
  { id: "preview", label: "Прев'ю", icon: Layers3 },
  { id: "settings", label: "Налаштування", icon: Settings2 },
];

export default function Home() {
  const [showHexGrid, setShowHexGrid] = useState(false);
  const [selectedZones, setSelectedZones] = useState<any[]>([]);
  const [gridType, setGridType] = useState<"hexagonal" | "square" | "circle">("hexagonal");
  const [hexSizeM, setHexSizeM] = useState(300.0);
  const [currentCityKey, setCurrentCityKey] = useState("Kyiv");
  const [workspaceView, setWorkspaceView] = useState<WorkspaceView>("map");
  const [proMode, setProMode] = useState(false);
  useEffect(() => {
    try { setProMode(localStorage.getItem("3dmap_pro_mode") === "1"); } catch {/* ignore */}
  }, []);
  const toggleProMode = (v: boolean) => {
    setProMode(v);
    try { localStorage.setItem("3dmap_pro_mode", v ? "1" : "0"); } catch {/* ignore */}
  };

  const { isGenerating, progress, status, downloadUrl, selectedArea, taskGroupId, taskIds, setTaskGroup, setGenerating, setActiveTaskId, setSelectedArea } = useGenerationStore();

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
      setWorkspaceView("preview");
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
          terrain_base_thickness_mm: 0.3, terrain_resolution: 180, terrarium_zoom: 15,
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

  // Відновлюємо task_id з localStorage після refresh
  useEffect(() => {
    const savedGroupId = localStorage.getItem("3dmap_task_group_id");
    const savedTaskIds = localStorage.getItem("3dmap_task_ids");
    if (savedGroupId && !taskGroupId) {
      const ids = savedTaskIds ? JSON.parse(savedTaskIds) : [savedGroupId];
      setTaskGroup(savedGroupId, ids);
      setGenerating(true);
    }
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

  const currentCity = CITIES[currentCityKey];
  const selectedCityLabel = CITY_LABELS[currentCityKey] ?? currentCityKey;
  const hasMapSelection = Boolean(selectedArea);
  const zoneCount = selectedZones.length;
  const selectionLabel = showHexGrid
    ? zoneCount > 0
      ? `${zoneCount} зон готово`
      : "Оберіть зони на мапі"
    : hasMapSelection
      ? "Ділянка готова до генерації"
      : "Позначте одну ділянку";
  const statusLabel = isGenerating
    ? `${progress}% • ${status || "Генерація триває"}`
    : downloadUrl
      ? "Файл готовий до завантаження"
      : "Готово до налаштування";

  const mapPanelClasses = workspaceView === "map" ? "flex" : "hidden lg:flex";
  const previewPanelClasses = workspaceView === "preview" ? "flex" : "hidden lg:flex";
  const settingsPanelClasses = workspaceView === "settings" ? "flex" : "hidden";

  return (
    <div className="min-h-[100dvh] bg-transparent">
      <OnboardingTour
        storageKey="onb_create_v1"
        steps={[
          { title: "Оберіть місто", body: "Виберіть місто зі списку у панелі зліва — карта одразу перенесеться туди." },
          { title: "Виділіть ділянку", body: "Намалюйте прямокутник на карті — це і буде ваша 3D-мапа. Оптимально 0.5–4 км²." },
          { title: "Налаштуйте та згенеруйте", body: "Оберіть розмір і шари, тоді натисніть «Згенерувати». За кілька хвилин завантажте готовий 3MF." },
        ]}
      />
      <div className="mx-auto flex min-h-[100dvh] max-w-[1760px] flex-col px-3 pb-24 pt-3 sm:px-4 lg:px-6 lg:pb-6">
        <header className="sticky top-0 z-30 rounded-[28px] border border-[var(--surface-border)] bg-[rgba(252,249,243,0.86)] px-4 py-4 shadow-[0_18px_60px_rgba(31,41,55,0.08)] backdrop-blur lg:static lg:px-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="space-y-2">
              <Link href="/" className="inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.22em] text-[var(--text-secondary)] transition hover:text-[var(--text-primary)]">
                ← monadruk
              </Link>
              <div>
                <h1 className="font-title text-2xl font-semibold tracking-tight text-[var(--text-primary)] sm:text-3xl">
                  Конструктор 3D-мапи
                </h1>
                <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--text-secondary)] sm:text-[15px]">
                  Обери ділянку міста, налаштуй модель та завантаж готовий 3MF для Bambu Studio / PrusaSlicer.
                </p>
              </div>
            </div>

            <div className="grid gap-2 sm:grid-cols-2 lg:min-w-[360px]">
              <Link
                href="/keychains"
                className="flex items-center gap-3 rounded-[22px] border border-[rgba(11,92,87,0.22)] bg-[rgba(15,118,110,0.08)] px-4 py-3 text-[var(--accent-strong)] transition hover:border-[rgba(11,92,87,0.36)] hover:bg-[rgba(15,118,110,0.12)] sm:col-span-2"
              >
                <span className="rounded-2xl bg-white/80 p-2">
                  <KeyRound size={18} />
                </span>
                <span>
                  <span className="block text-[11px] font-semibold uppercase tracking-[0.18em] opacity-70">
                    Новий інструмент
                  </span>
                  <span className="mt-1 block text-sm font-semibold">Створити брелок з мапою</span>
                </span>
              </Link>
              <div className="rounded-[22px] border border-[var(--surface-border)] bg-[rgba(255,255,255,0.8)] px-4 py-3">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
                  Місто
                </div>
                <select
                  value={currentCityKey}
                  onChange={(e) => setCurrentCityKey(e.target.value)}
                  className="mt-1 w-full bg-transparent text-sm font-semibold text-[var(--text-primary)] outline-none cursor-pointer"
                >
                  {Object.keys(CITIES).map((key) => (
                    <option key={key} value={key}>{CITY_LABELS[key] ?? key}</option>
                  ))}
                </select>
              </div>
              <div className="rounded-[22px] border border-[var(--surface-border)] bg-[rgba(255,255,255,0.8)] px-4 py-3">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
                  Статус
                </div>
                <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">{selectionLabel}</div>
              </div>
            </div>
          </div>

          <div className="mt-4 flex gap-2 overflow-x-auto pb-1 lg:hidden">
            {WORKSPACE_TABS.map(({ id, label, icon: Icon }) => {
              const isActive = workspaceView === id;
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => setWorkspaceView(id)}
                  className={`flex min-w-fit items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition ${
                    isActive
                      ? "bg-[var(--accent-strong)] text-white shadow-[0_14px_30px_rgba(11,92,87,0.24)]"
                      : "bg-white/75 text-[var(--text-secondary)]"
                  }`}
                >
                  <Icon size={16} />
                  {label}
                </button>
              );
            })}
          </div>
        </header>

        <div className="mt-3">
          <WizardSteps
            state={{
              cityLabel: selectedCityLabel,
              hasSelection: hasMapSelection || zoneCount > 0,
              isGenerating,
              hasDownload: Boolean(downloadUrl),
              progress,
            }}
          />
        </div>

        <div className="mt-3 flex min-h-0 flex-1 flex-col gap-3 lg:grid lg:grid-cols-[380px,minmax(0,1fr)]">
          <aside className="hidden min-h-0 lg:block">
            <div className="flex h-full flex-col overflow-hidden rounded-[30px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_22px_70px_rgba(15,23,42,0.08)] backdrop-blur">
              <div className="flex shrink-0 items-center justify-between gap-2 border-b border-[var(--surface-border)] px-4 py-3">
                <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
                  {proMode ? "Експертний режим" : "Швидке створення"}
                </span>
                <div className="flex items-center gap-1 rounded-full border border-[var(--surface-border)] bg-white/80 p-0.5 text-xs">
                  <button type="button" onClick={() => toggleProMode(false)}
                    className={`rounded-full px-3 py-1 font-semibold transition ${!proMode ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)]"}`}>Просто</button>
                  <button type="button" onClick={() => toggleProMode(true)}
                    className={`rounded-full px-3 py-1 font-semibold transition ${proMode ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)]"}`}>Про</button>
                </div>
              </div>
              <div className="min-h-0 flex-1 overflow-hidden">
                {proMode ? (
                  <ControlPanel
                    showHexGrid={showHexGrid}
                    setShowHexGrid={setShowHexGrid}
                    selectedZones={selectedZones}
                    setSelectedZones={setSelectedZones}
                    gridType={gridType}
                    setGridType={setGridType}
                    hexSizeM={hexSizeM}
                    setHexSizeM={setHexSizeM}
                    availableCities={CITIES}
                    selectedCityKey={currentCityKey}
                    onCityChange={setCurrentCityKey}
                  />
                ) : (
                  <SimpleControlPanel
                    availableCities={CITIES}
                    selectedCityKey={currentCityKey}
                    onCityChange={setCurrentCityKey}
                  onAdvanced={() => toggleProMode(true)}
                  />
                )}
              </div>
            </div>
          </aside>

          <section className="flex min-h-0 flex-1 flex-col gap-3">
            <div className={mapPanelClasses}>
              <div className="flex min-h-[360px] flex-1 flex-col overflow-hidden rounded-[30px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_22px_70px_rgba(15,23,42,0.08)] backdrop-blur lg:min-h-[440px]">
                <div className="flex items-start justify-between gap-4 border-b border-[var(--surface-border)] px-4 py-4 sm:px-5">
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[var(--text-secondary)]">
                      {showHexGrid ? "Grid Selection" : "Single Selection"}
                    </p>
                    <h2 className="mt-1 font-title text-xl font-semibold text-[var(--text-primary)]">
                      {showHexGrid ? "Оберіть зони для серії" : "Позначте ділянку на мапі"}
                    </h2>
                    <p className="mt-1 text-sm text-[var(--text-secondary)]">
                      {showHexGrid
                        ? "Працюйте з кількома зонами та швидко готуйте пакетний рендер."
                        : "Виділіть одну ділянку, щоб швидко згенерувати модель і перейти до прев'ю."}
                    </p>
                  </div>

                  <div className="rounded-[18px] border border-[var(--surface-border)] bg-white/80 px-3 py-2 text-right">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
                      Режим
                    </div>
                    <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
                      {showHexGrid ? "Сітка зон" : "Одна ділянка"}
                    </div>
                  </div>
                </div>

                <div className="min-h-0 flex-1 bg-[rgba(255,255,255,0.55)] p-2 sm:p-3">
                  {showHexGrid ? (
                    <HexagonalGrid
                      key={`hex-grid-${currentCityKey}`}
                      bounds={currentCity.bounds}
                      onZonesSelected={setSelectedZones}
                      gridType={gridType}
                      hexSizeM={hexSizeM}
                    />
                  ) : (
                    <div className="h-full overflow-hidden rounded-[24px]">
                      <MapSelector center={currentCity.center} />
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className={previewPanelClasses}>
              <div className="flex min-h-[320px] flex-1 flex-col overflow-hidden rounded-[30px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_22px_70px_rgba(15,23,42,0.08)] backdrop-blur lg:min-h-[360px]">
                <div className="flex items-start justify-between gap-4 border-b border-[var(--surface-border)] px-4 py-4 sm:px-5">
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[var(--text-secondary)]">
                      3D Preview
                    </p>
                    <h2 className="mt-1 font-title text-xl font-semibold text-[var(--text-primary)]">
                      Перевіряйте форму моделі ще до завантаження
                    </h2>
                    <p className="mt-1 text-sm text-[var(--text-secondary)]">
                      На телефоні прев'ю винесене в окремий екран, щоб не конфліктувати з картою та налаштуваннями.
                    </p>
                  </div>

                  <div className="rounded-[18px] border border-[var(--surface-border)] bg-white/80 px-3 py-2 text-right">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
                      Стан
                    </div>
                    <div className="mt-1 flex items-center justify-end gap-2">
                      <span className="text-sm font-semibold text-[var(--text-primary)]">{statusLabel}</span>
                      {isGenerating && taskGroupId && (
                        <button
                          onClick={handleCancelTask}
                          className="inline-flex items-center gap-1 rounded-full bg-red-100 px-2 py-0.5 text-[10px] font-semibold text-red-700 hover:bg-red-200 transition-colors"
                          title="Скасувати генерацію"
                        >
                          <X size={10} /> Скасувати
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                <div className="min-h-0 flex-1 p-2 sm:p-3">
                  <div className="h-full overflow-hidden rounded-[24px] border border-[rgba(15,23,42,0.12)]">
                    <Preview3D />
                  </div>
                </div>
              </div>
            </div>

            <div className={settingsPanelClasses}>
              <div className="flex min-h-[420px] flex-1 flex-col overflow-hidden rounded-[30px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_22px_70px_rgba(15,23,42,0.08)] backdrop-blur lg:hidden">
                <div className="flex shrink-0 items-center justify-between gap-2 border-b border-[var(--surface-border)] px-4 py-3">
                  <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
                    {proMode ? "Експертний режим" : "Швидке створення"}
                  </span>
                  <div className="flex items-center gap-1 rounded-full border border-[var(--surface-border)] bg-white/80 p-0.5 text-xs">
                    <button type="button" onClick={() => toggleProMode(false)}
                      className={`rounded-full px-3 py-1 font-semibold transition ${!proMode ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)]"}`}>Просто</button>
                    <button type="button" onClick={() => toggleProMode(true)}
                      className={`rounded-full px-3 py-1 font-semibold transition ${proMode ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)]"}`}>Про</button>
                  </div>
                </div>
                {proMode ? (
                  <ControlPanel
                    showHexGrid={showHexGrid}
                    setShowHexGrid={setShowHexGrid}
                    selectedZones={selectedZones}
                    setSelectedZones={setSelectedZones}
                    gridType={gridType}
                    setGridType={setGridType}
                    hexSizeM={hexSizeM}
                    setHexSizeM={setHexSizeM}
                    availableCities={CITIES}
                    selectedCityKey={currentCityKey}
                    onCityChange={setCurrentCityKey}
                  />
                ) : (
                  <SimpleControlPanel
                    availableCities={CITIES}
                    selectedCityKey={currentCityKey}
                    onCityChange={setCurrentCityKey}
                  onAdvanced={() => toggleProMode(true)}
                  />
                )}
              </div>
            </div>
          </section>
        </div>

        <div className="pointer-events-none fixed inset-x-0 bottom-0 z-40 px-3 pb-3 lg:hidden">
          <div className="pointer-events-auto rounded-[26px] border border-[rgba(255,255,255,0.55)] bg-[rgba(15,23,42,0.9)] px-4 py-3 text-white shadow-[0_22px_60px_rgba(15,23,42,0.3)] backdrop-blur">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-white/55">
                  Швидкий статус
                </p>
                <div className="mt-1 text-sm font-semibold">
                  {isGenerating ? (
                    <span className="inline-flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      {status || "Генерація триває"}
                    </span>
                  ) : downloadUrl ? (
                    <span className="inline-flex items-center gap-2">
                      <Download className="h-4 w-4" />
                      Модель готова
                    </span>
                  ) : (
                    selectionLabel
                  )}
                </div>
                {isGenerating && <p className="mt-1 text-xs text-white/65">{progress}% виконано</p>}
                {!isGenerating && downloadUrl && (
                  <a
                    href={downloadUrl}
                    download
                    className="mt-2 inline-flex items-center gap-2 rounded-full bg-teal-600 px-4 py-2 text-xs font-bold text-white"
                  >
                    <Download size={14} />
                    Завантажити 3MF
                  </a>
                )}
              </div>

              <div className="flex flex-wrap justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setWorkspaceView("map")}
                  className={`rounded-full px-3 py-2 text-xs font-semibold ${
                    workspaceView === "map" ? "bg-white text-slate-900" : "bg-white/10 text-white"
                  }`}
                >
                  Мапа
                </button>
                <button
                  type="button"
                  onClick={() => setWorkspaceView("preview")}
                  className={`rounded-full px-3 py-2 text-xs font-semibold ${
                    workspaceView === "preview" ? "bg-white text-slate-900" : "bg-white/10 text-white"
                  }`}
                >
                  Прев'ю
                </button>
                <button
                  type="button"
                  onClick={() => setWorkspaceView("settings")}
                  className={`rounded-full px-3 py-2 text-xs font-semibold ${
                    workspaceView === "settings" ? "bg-white text-slate-900" : "bg-white/10 text-white"
                  }`}
                >
                  Дії
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
