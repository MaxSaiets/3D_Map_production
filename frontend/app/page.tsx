"use client";

import dynamic from "next/dynamic";
import { useState } from "react";
import { Download, Layers3, Loader2, Map as MapIcon, Settings2 } from "lucide-react";
import { Preview3D } from "@/components/Preview3D";
import { ControlPanel } from "@/components/ControlPanel";
import { useGenerationStore } from "@/store/generation-store";

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
  Kyiv: {
    bounds: { north: 50.6, south: 50.2, east: 30.8, west: 30.2 },
    center: [50.4501, 30.5234],
  },
  Khmelnytskyi: {
    bounds: { north: 49.48, south: 49.36, east: 27.08, west: 26.88 },
    center: [49.42, 26.98],
  },
};

const CITY_LABELS: Record<string, string> = {
  Kyiv: "Київ",
  Khmelnytskyi: "Хмельницький",
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

  const { isGenerating, progress, status, downloadUrl, selectedArea } = useGenerationStore();

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
      <div className="mx-auto flex min-h-[100dvh] max-w-[1760px] flex-col px-3 pb-24 pt-3 sm:px-4 lg:px-6 lg:pb-6">
        <header className="sticky top-0 z-30 rounded-[28px] border border-[var(--surface-border)] bg-[rgba(252,249,243,0.86)] px-4 py-4 shadow-[0_18px_60px_rgba(31,41,55,0.08)] backdrop-blur lg:static lg:px-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="space-y-2">
              <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-[var(--text-secondary)]">
                3D Map Workspace
              </p>
              <div>
                <h1 className="font-title text-2xl font-semibold tracking-tight text-[var(--text-primary)] sm:text-3xl">
                  Мобільний простір для генерації 3D-мап
                </h1>
                <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--text-secondary)] sm:text-[15px]">
                  Оберіть місто та ділянку, налаштуйте модель і переходьте між мапою,
                  прев'ю та параметрами без перевантаженого інтерфейсу.
                </p>
              </div>
            </div>

            <div className="grid gap-2 sm:grid-cols-2 lg:min-w-[360px]">
              <div className="rounded-[22px] border border-[var(--surface-border)] bg-[rgba(255,255,255,0.8)] px-4 py-3">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
                  Місто
                </div>
                <div className="mt-1 text-base font-semibold text-[var(--text-primary)]">{selectedCityLabel}</div>
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

        <div className="mt-3 flex min-h-0 flex-1 flex-col gap-3 lg:grid lg:grid-cols-[380px,minmax(0,1fr)]">
          <aside className="hidden min-h-0 lg:block">
            <div className="h-full overflow-hidden rounded-[30px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_22px_70px_rgba(15,23,42,0.08)] backdrop-blur">
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
                      key={`hex-grid-${gridType}-${hexSizeM}-${currentCityKey}`}
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
                    <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">{statusLabel}</div>
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
                  <p className="mt-1 text-xs text-white/65">Відкрийте вкладку “Дії”, щоб завантажити файл.</p>
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
