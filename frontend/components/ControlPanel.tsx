"use client";

import { useEffect, useRef, useState } from "react";
import {
  Building2,
  CheckCircle2,
  ChevronDown,
  Download,
  Grid,
  Layers3,
  Loader2,
  MapPinned,
  Mountain,
  Play,
  Route,
  Sparkles,
} from "lucide-react";
import { api } from "@/lib/api";
import { useGenerationStore } from "@/store/generation-store";

interface ControlPanelProps {
  showHexGrid?: boolean;
  setShowHexGrid?: (show: boolean) => void;
  selectedZones?: any[];
  setSelectedZones?: (zones: any[]) => void;
  gridType?: "hexagonal" | "square" | "circle";
  setGridType?: (type: "hexagonal" | "square" | "circle") => void;
  hexSizeM?: number;
  setHexSizeM?: (size: number) => void;
  availableCities?: Record<string, any>;
  selectedCityKey?: string;
  onCityChange?: (cityKey: string) => void;
}

type AdvancedPanel = "roads" | "buildings" | "terrain" | "preview";

function SectionFrame({
  eyebrow,
  title,
  description,
  children,
}: {
  eyebrow: string;
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-[28px] border border-[var(--surface-border)] bg-[var(--surface-panel-strong)] p-4 shadow-[0_12px_36px_rgba(15,23,42,0.06)] sm:p-5">
      <div className="mb-4">
        <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[var(--text-secondary)]">
          {eyebrow}
        </p>
        <h2 className="mt-1 font-title text-xl font-semibold text-[var(--text-primary)]">{title}</h2>
        <p className="mt-1 text-sm leading-6 text-[var(--text-secondary)]">{description}</p>
      </div>
      <div className="space-y-4">{children}</div>
    </section>
  );
}

function InfoPill({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: string;
  tone?: "default" | "accent" | "success";
}) {
  const toneClasses =
    tone === "accent"
      ? "border-teal-200 bg-teal-50 text-teal-800"
      : tone === "success"
        ? "border-emerald-200 bg-emerald-50 text-emerald-800"
        : "border-[var(--surface-border)] bg-white/80 text-[var(--text-primary)]";

  return (
    <div className={`rounded-[20px] border px-3 py-2 ${toneClasses}`}>
      <div className="text-[11px] font-semibold uppercase tracking-[0.16em] opacity-70">{label}</div>
      <div className="mt-1 text-sm font-semibold">{value}</div>
    </div>
  );
}

function SliderField({
  label,
  valueLabel,
  min,
  max,
  step,
  value,
  onChange,
  hint,
}: {
  label: string;
  valueLabel: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
  hint?: string;
}) {
  return (
    <div className="rounded-[22px] border border-[var(--surface-border)] bg-white/80 p-3">
      <div className="flex items-start justify-between gap-3">
        <label className="text-sm font-medium text-[var(--text-primary)]">{label}</label>
        <span className="text-sm font-semibold text-[var(--accent-strong)]">{valueLabel}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="mt-3 w-full"
      />
      {hint && <p className="mt-2 text-xs leading-5 text-[var(--text-secondary)]">{hint}</p>}
    </div>
  );
}

function AccordionButton({
  title,
  description,
  icon: Icon,
  isOpen,
  onClick,
}: {
  title: string;
  description: string;
  icon: typeof Route;
  isOpen: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex w-full items-center justify-between gap-3 rounded-[22px] border border-[var(--surface-border)] bg-white/80 px-4 py-3 text-left transition hover:border-[rgba(11,92,87,0.25)] hover:bg-white"
    >
      <div className="flex items-start gap-3">
        <div className="mt-0.5 rounded-2xl bg-[rgba(11,92,87,0.08)] p-2 text-[var(--accent-strong)]">
          <Icon size={18} />
        </div>
        <div>
          <div className="text-sm font-semibold text-[var(--text-primary)]">{title}</div>
          <div className="text-xs leading-5 text-[var(--text-secondary)]">{description}</div>
        </div>
      </div>
      <ChevronDown
        size={18}
        className={`shrink-0 text-[var(--text-secondary)] transition-transform ${isOpen ? "rotate-180" : ""}`}
      />
    </button>
  );
}

function CheckboxRow({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex items-start justify-between gap-4 rounded-[20px] border border-[var(--surface-border)] bg-white/80 px-4 py-3">
      <div>
        <div className="text-sm font-medium text-[var(--text-primary)]">{label}</div>
        {description && <div className="mt-1 text-xs leading-5 text-[var(--text-secondary)]">{description}</div>}
      </div>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-1 h-4 w-4 shrink-0"
      />
    </label>
  );
}

export function ControlPanel({
  showHexGrid: externalShowHexGrid,
  setShowHexGrid: externalSetShowHexGrid,
  selectedZones: externalSelectedZones,
  setSelectedZones: externalSetSelectedZones,
  gridType: externalGridType,
  setGridType: externalSetGridType,
  hexSizeM: externalHexSizeM,
  setHexSizeM: externalSetHexSizeM,
  availableCities,
  selectedCityKey,
  onCityChange,
}: ControlPanelProps = {}) {
  const {
    selectedArea,
    isGenerating,
    taskGroupId,
    taskIds = [],
    activeTaskId,
    taskStatuses = {},
    showAllZones,
    progress,
    status,
    downloadUrl,
    roadWidthMultiplier,
    roadHeightMm,
    roadEmbedMm,
    buildingMinHeight,
    buildingHeightMultiplier,
    buildingFoundationMm,
    buildingEmbedMm,
    waterDepth,
    terrainEnabled,
    terrainZScale,
    terrainBaseThicknessMm,
    terrainResolution,
    terrariumZoom,
    exportFormat,
    modelSizeMm,
    isAmsMode,
    previewIncludeBase,
    previewIncludeRoads,
    previewIncludeBuildings,
    previewIncludeWater,
    previewIncludeParks,
    setRoadWidthMultiplier,
    setRoadHeightMm,
    setRoadEmbedMm,
    setBuildingMinHeight,
    setBuildingHeightMultiplier,
    setBuildingFoundationMm,
    setBuildingEmbedMm,
    setWaterDepth,
    setTerrainEnabled,
    setTerrainZScale,
    setTerrainBaseThicknessMm,
    setTerrainResolution,
    setTerrariumZoom,
    setExportFormat,
    setModelSizeMm,
    setAmsMode,
    setPreviewIncludeBase,
    setPreviewIncludeRoads,
    setPreviewIncludeBuildings,
    setPreviewIncludeWater,
    setPreviewIncludeParks,
    setGenerating,
    setTaskGroup,
    setActiveTaskId,
    setTaskStatuses,
    setBatchZoneMetaByTaskId,
    setShowAllZones,
    updateProgress,
    setDownloadUrl,
  } = useGenerationStore();

  const [error, setError] = useState<string | null>(null);
  const [internalShowHexGrid, setInternalShowHexGrid] = useState(false);
  const [internalSelectedZones, setInternalSelectedZones] = useState<any[]>([]);
  const [internalGridType, setInternalGridType] = useState<"hexagonal" | "square" | "circle">("hexagonal");
  const [internalHexSizeM, setInternalHexSizeM] = useState(1000.0);
  const pollingInFlightRef = useRef(false);
  const [openPanels, setOpenPanels] = useState<Record<AdvancedPanel, boolean>>({
    roads: false,
    buildings: false,
    terrain: false,
    preview: false,
  });

  const showHexGrid = externalShowHexGrid !== undefined ? externalShowHexGrid : internalShowHexGrid;
  const setShowHexGrid = externalSetShowHexGrid || setInternalShowHexGrid;
  const selectedZones = externalSelectedZones !== undefined ? externalSelectedZones : internalSelectedZones;
  const setSelectedZones = externalSetSelectedZones || setInternalSelectedZones;
  const gridType = externalGridType !== undefined ? externalGridType : internalGridType;
  const setGridType = externalSetGridType || setInternalGridType;
  const hexSizeM = externalHexSizeM !== undefined ? externalHexSizeM : internalHexSizeM;
  const setHexSizeM = externalSetHexSizeM || setInternalHexSizeM;

  useEffect(() => {
    if (!taskGroupId || !isGenerating) return;

    const interval = setInterval(async () => {
      if (pollingInFlightRef.current) return;
      pollingInFlightRef.current = true;
      try {
        if (taskIds.length > 1) {
          const results = await Promise.all(
            taskIds.map(async (id) => {
              try {
                return await api.getStatus(id);
              } catch {
                return {
                  task_id: id,
                  status: "failed",
                  progress: 0,
                  message: "Status fetch failed",
                  download_url: null,
                } as any;
              }
            }),
          );

          const tasksList = results as any[];
          const total = tasksList.length;
          const completed = tasksList.filter((task) => task.status === "completed").length;
          const failed = tasksList.filter((task) => task.status === "failed").length;

          const nextStatuses: Record<string, any> = {};
          for (const task of tasksList) nextStatuses[task.task_id] = task;
          setTaskStatuses(nextStatuses);

          const avg = tasksList.length
            ? Math.round(tasksList.reduce((sum, task) => sum + (task.progress || 0), 0) / tasksList.length)
            : 0;
          updateProgress(avg, `Зони: ${completed}/${total} готово${failed ? `, помилок: ${failed}` : ""}`);

          const active =
            (activeTaskId ? nextStatuses[activeTaskId] : null) || (taskIds[0] ? nextStatuses[taskIds[0]] : null);
          if (active && active.status === "completed") {
            setDownloadUrl(active.download_url);
          }

          if (completed + failed >= total) {
            setGenerating(false);
            if (failed) {
              const firstFailed = tasksList.find((task) => task.status === "failed");
              if (firstFailed) setError(firstFailed.message || "Одна з зон не згенерувалася");
            }
          }
          return;
        }

        const resp = await api.getStatus(taskGroupId);
        const single = resp as any;
        setTaskStatuses({ [single.task_id]: single });
        updateProgress(single.progress, single.message);
        if (single.status === "completed") {
          setGenerating(false);
          setDownloadUrl(single.download_url);
        } else if (single.status === "failed") {
          setGenerating(false);
          setError(single.message);
        }
      } catch (pollError: any) {
        console.error("Помилка перевірки статусу:", pollError);
        if (pollError?.response?.status === 404) {
          setGenerating(false);
          setTaskGroup(null, []);
          setActiveTaskId(null);
          setTaskStatuses({});
          setDownloadUrl(null);
          updateProgress(0, "");
          setError("Попередню задачу не знайдено на сервері. Можна запускати нову генерацію.");
        }
      } finally {
        pollingInFlightRef.current = false;
      }
    }, 2000);

    return () => {
      clearInterval(interval);
      pollingInFlightRef.current = false;
    };
  }, [
    taskGroupId,
    isGenerating,
    updateProgress,
    setGenerating,
    setDownloadUrl,
    activeTaskId,
    taskIds,
    setTaskStatuses,
    setTaskGroup,
    setActiveTaskId,
  ]);

  const handleGenerate = async () => {
    if (!selectedArea) {
      setError("Виберіть область на карті");
      return;
    }

    setError(null);
    setGenerating(true);

    try {
      const request = {
        north: selectedArea.getNorth(),
        south: selectedArea.getSouth(),
        east: selectedArea.getEast(),
        west: selectedArea.getWest(),
        road_width_multiplier: roadWidthMultiplier,
        road_height_mm: roadHeightMm,
        road_embed_mm: roadEmbedMm,
        building_min_height: buildingMinHeight,
        building_height_multiplier: buildingHeightMultiplier,
        building_foundation_mm: buildingFoundationMm,
        building_embed_mm: buildingEmbedMm,
        water_depth: waterDepth,
        terrain_enabled: terrainEnabled,
        terrain_z_scale: terrainZScale,
        terrain_base_thickness_mm: terrainBaseThicknessMm,
        terrain_resolution: terrainResolution,
        terrarium_zoom: terrariumZoom,
        flatten_buildings_on_terrain: false,
        flatten_roads_on_terrain: false,
        export_format: exportFormat,
        model_size_mm: modelSizeMm,
        context_padding_m: 400.0,
        is_ams_mode: isAmsMode,
        preview_include_base: previewIncludeBase,
        preview_include_roads: previewIncludeRoads,
        preview_include_buildings: previewIncludeBuildings,
        preview_include_water: previewIncludeWater,
        preview_include_parks: previewIncludeParks,
      };

      const response = await api.generateModel(request);
      setTaskGroup(response.task_id, [response.task_id]);
      setActiveTaskId(response.task_id);
    } catch (generateError: any) {
      console.error("[ERROR] Помилка генерації моделі:", generateError);
      setError(generateError.message || "Помилка генерації моделі");
      setGenerating(false);
    }
  };

  const handleDownload = async () => {
    if (!activeTaskId || !downloadUrl) return;

    try {
      const fbUrl = taskStatuses[activeTaskId]?.firebase_url;
      const isFormatMatch = fbUrl && fbUrl.toLowerCase().split("?")[0].endsWith(`.${exportFormat.toLowerCase()}`);
      const blob =
        fbUrl && isFormatMatch ? await api.downloadFile(fbUrl) : await api.downloadModel(activeTaskId, exportFormat);

      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `model_${activeTaskId.slice(0, 8)}.${exportFormat}`;
      document.body.appendChild(link);
      link.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(link);
    } catch (downloadError) {
      console.error("[Download Error]", downloadError);
      setError("Помилка завантаження файлу");
    }
  };

  const kyivBounds = {
    north: 50.6,
    south: 50.2,
    east: 30.8,
    west: 30.2,
  };

  const getSelectedZonesBounds = (zones: any[]) => {
    const lats: number[] = [];
    const lons: number[] = [];

    zones.forEach((zone) => {
      const coordinates = zone?.geometry?.coordinates;
      if (!Array.isArray(coordinates)) return;
      coordinates.forEach((ring: any[]) => {
        if (!Array.isArray(ring)) return;
        ring.forEach((coord) => {
          if (!Array.isArray(coord) || coord.length < 2) return;
          const lon = Number(coord[0]);
          const lat = Number(coord[1]);
          if (Number.isFinite(lat) && Number.isFinite(lon)) {
            lats.push(lat);
            lons.push(lon);
          }
        });
      });
    });

    if (lats.length === 0 || lons.length === 0) return null;

    return {
      north: Math.max(...lats),
      south: Math.min(...lats),
      east: Math.max(...lons),
      west: Math.min(...lons),
    };
  };

  const handleGenerateZones = async () => {
    if (selectedZones.length === 0) {
      setError("Виберіть хоча б одну зону");
      return;
    }

    setError(null);
    setGenerating(true);
    setShowHexGrid(false);

    try {
      const zonesSorted = [...selectedZones].sort((a, b) => {
        const ar = Number(a?.properties?.row ?? 0);
        const br = Number(b?.properties?.row ?? 0);
        if (ar !== br) return ar - br;
        const ac = Number(a?.properties?.col ?? 0);
        const bc = Number(b?.properties?.col ?? 0);
        if (ac !== bc) return ac - bc;
        const aid = String(a?.id || a?.properties?.id || "");
        const bid = String(b?.id || b?.properties?.id || "");
        return aid.localeCompare(bid);
      });

      let requestBounds = getSelectedZonesBounds(zonesSorted) || kyivBounds;
      if (availableCities && selectedCityKey && availableCities[selectedCityKey]) {
        requestBounds = getSelectedZonesBounds(zonesSorted) || availableCities[selectedCityKey].bounds;
      }

      const request = {
        north: requestBounds.north,
        south: requestBounds.south,
        east: requestBounds.east,
        west: requestBounds.west,
        road_width_multiplier: roadWidthMultiplier,
        road_height_mm: roadHeightMm,
        road_embed_mm: roadEmbedMm,
        building_min_height: buildingMinHeight,
        building_height_multiplier: buildingHeightMultiplier,
        building_foundation_mm: buildingFoundationMm,
        building_embed_mm: buildingEmbedMm,
        water_depth: waterDepth,
        terrain_enabled: terrainEnabled,
        terrain_z_scale: terrainZScale,
        terrain_base_thickness_mm: terrainBaseThicknessMm,
        terrain_resolution: terrainResolution,
        terrarium_zoom: terrariumZoom,
        terrain_smoothing_sigma: 2.0,
        terrain_subdivide: false,
        terrain_subdivide_levels: 1,
        flatten_buildings_on_terrain: false,
        flatten_roads_on_terrain: false,
        export_format: exportFormat,
        model_size_mm: modelSizeMm,
        is_ams_mode: isAmsMode,
        preview_include_base: previewIncludeBase,
        preview_include_roads: previewIncludeRoads,
        preview_include_buildings: previewIncludeBuildings,
        preview_include_water: previewIncludeWater,
        preview_include_parks: previewIncludeParks,
      };

      const response = await api.generateZones(zonesSorted, request);
      const ids =
        (response as any).all_task_ids && (response as any).all_task_ids.length
          ? (response as any).all_task_ids
          : [response.task_id];
      setTaskGroup(response.task_id, ids);
      setActiveTaskId(ids[0] ?? null);
      if (ids.length > 1) setShowAllZones(true);

      try {
        const meta: Record<string, any> = {};
        for (let i = 0; i < ids.length; i += 1) {
          const zone = zonesSorted[i];
          const zoneId = String(zone?.id || zone?.properties?.id || `zone_${i}`);
          meta[String(ids[i])] = {
            zoneId,
            row: zone?.properties?.row,
            col: zone?.properties?.col,
          };
        }
        setBatchZoneMetaByTaskId(meta);
      } catch {
        // ignore metadata sync issues
      }
    } catch (generateError: any) {
      setError(generateError.message || "Помилка генерації моделей для зон");
      setGenerating(false);
    }
  };

  const togglePanel = (panel: AdvancedPanel) => {
    setOpenPanels((current) => ({ ...current, [panel]: !current[panel] }));
  };

  const cityOptions = availableCities ? Object.keys(availableCities) : [];
  const selectionReady = showHexGrid ? selectedZones.length > 0 : Boolean(selectedArea);
  const selectionMode = showHexGrid ? "Сітка зон" : "Одна ділянка";
  const selectionCopy = showHexGrid
    ? selectedZones.length > 0
      ? `${selectedZones.length} зон готово до пакетної генерації`
      : "Спочатку оберіть зони на мапі"
    : selectedArea
      ? "Ділянка вибрана, можна запускати генерацію"
      : "Намалюйте область на мапі, щоб перейти до генерації";
  const primaryActionLabel = showHexGrid
    ? `Згенерувати ${selectedZones.length > 0 ? `${selectedZones.length} зон` : "вибрані зони"}`
    : "Згенерувати 3D модель";
  const statusSummary = isGenerating
    ? status || "Генерація триває"
    : downloadUrl
      ? "Модель готова до завантаження"
      : taskGroupId
        ? "Останній рендер збережено в сесії"
        : "Немає активних задач";

  return (
    <div className="h-full overflow-y-auto px-4 py-4 sm:px-5">
      <div className="space-y-4 pb-8">
        <SectionFrame
          eyebrow="Essentials"
          title="Керуйте потоком без зайвого шуму"
          description="Спершу виберіть місто та спосіб роботи, потім задайте базові параметри й запустіть генерацію."
        >
          <div className="grid gap-3 sm:grid-cols-2">
            <InfoPill label="Режим" value={selectionMode} tone="accent" />
            <InfoPill
              label="Готовність"
              value={selectionCopy}
              tone={selectionReady ? "success" : "default"}
            />
          </div>

          {cityOptions.length > 0 && onCityChange && selectedCityKey && (
            <div className="rounded-[24px] border border-[var(--surface-border)] bg-white/80 p-4">
              <label className="mb-2 flex items-center gap-2 text-sm font-semibold text-[var(--text-primary)]">
                <MapPinned size={16} className="text-[var(--accent-strong)]" />
                Місто
              </label>
              <select
                value={selectedCityKey}
                onChange={(e) => {
                  onCityChange(e.target.value);
                  setError(null);
                  setSelectedZones([]);
                }}
                className="w-full rounded-2xl border border-[var(--surface-border)] bg-[rgba(255,255,255,0.95)] px-4 py-3 text-sm text-[var(--text-primary)] outline-none transition focus:border-[rgba(11,92,87,0.35)]"
              >
                {cityOptions.map((city) => (
                  <option key={city} value={city}>
                    {city === "Kyiv" ? "Київ" : city === "Khmelnytskyi" ? "Хмельницький" : city}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="grid gap-3 sm:grid-cols-2">
            <button
              type="button"
              onClick={() => {
                setShowHexGrid(false);
                setSelectedZones([]);
                setError(null);
              }}
              className={`rounded-[22px] border px-4 py-4 text-left transition ${
                !showHexGrid
                  ? "border-[rgba(11,92,87,0.28)] bg-[rgba(15,118,110,0.08)] shadow-[0_16px_32px_rgba(11,92,87,0.12)]"
                  : "border-[var(--surface-border)] bg-white/80"
              }`}
            >
              <div className="flex items-center gap-2 text-sm font-semibold text-[var(--text-primary)]">
                <MapPinned size={16} className="text-[var(--accent-strong)]" />
                Одна ділянка
              </div>
              <p className="mt-2 text-xs leading-5 text-[var(--text-secondary)]">
                Найшвидший шлях для одного рендеру з простою взаємодією на телефоні.
              </p>
            </button>

            <button
              type="button"
              onClick={() => {
                setShowHexGrid(true);
                setError(null);
              }}
              className={`rounded-[22px] border px-4 py-4 text-left transition ${
                showHexGrid
                  ? "border-[rgba(11,92,87,0.28)] bg-[rgba(15,118,110,0.08)] shadow-[0_16px_32px_rgba(11,92,87,0.12)]"
                  : "border-[var(--surface-border)] bg-white/80"
              }`}
            >
              <div className="flex items-center gap-2 text-sm font-semibold text-[var(--text-primary)]">
                <Grid size={16} className="text-[var(--accent-strong)]" />
                Серія зон
              </div>
              <p className="mt-2 text-xs leading-5 text-[var(--text-secondary)]">
                Оберіть кілька клітин і згенеруйте пакет моделей для прев'ю та друку.
              </p>
            </button>
          </div>

          {showHexGrid && (
            <div className="rounded-[24px] border border-[rgba(15,118,110,0.15)] bg-[rgba(15,118,110,0.05)] p-4">
              <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-[var(--text-primary)]">
                <Grid size={16} className="text-[var(--accent-strong)]" />
                Параметри сітки
              </div>

              <div className="grid gap-3">
                <div>
                  <label className="mb-2 block text-sm font-medium text-[var(--text-primary)]">Тип сітки</label>
                  <select
                    value={gridType}
                    onChange={(e) => {
                      const newType = e.target.value as "hexagonal" | "square" | "circle";
                      setGridType(newType);
                      setSelectedZones([]);
                    }}
                    className="w-full rounded-2xl border border-[var(--surface-border)] bg-white px-4 py-3 text-sm text-[var(--text-primary)] outline-none transition focus:border-[rgba(11,92,87,0.35)]"
                  >
                    <option value="hexagonal">Шестикутники</option>
                    <option value="square">Квадрати</option>
                    <option value="circle">Круги</option>
                  </select>
                </div>

                <SliderField
                  label={gridType === "circle" ? "Діаметр круга" : "Розмір клітинки"}
                  valueLabel={`${hexSizeM.toFixed(0)} м`}
                  min={200}
                  max={2000}
                  step={100}
                  value={hexSizeM}
                  onChange={(value) => {
                    setHexSizeM(value);
                    setSelectedZones([]);
                  }}
                  hint="Змінюйте масштаб клітин перед вибором зон. Для мобільного перегляду стартова клітинка менша, щоб простіше потрапляти по цілям."
                />

                <div className="grid gap-3 sm:grid-cols-2">
                  <InfoPill label="Вибрано зон" value={String(selectedZones.length)} tone={selectedZones.length > 0 ? "success" : "default"} />
                  <InfoPill
                    label="Стан вибору"
                    value={selectedZones.length > 0 ? "Готово до запуску" : "Оберіть зони на мапі"}
                    tone={selectedZones.length > 0 ? "success" : "default"}
                  />
                </div>
              </div>
            </div>
          )}

          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-[24px] border border-[var(--surface-border)] bg-white/80 p-4">
              <label className="mb-2 flex items-center gap-2 text-sm font-semibold text-[var(--text-primary)]">
                <Sparkles size={16} className="text-[var(--accent-strong)]" />
                Формат експорту
              </label>
              <select
                value={exportFormat}
                onChange={(e) => setExportFormat(e.target.value as "stl" | "3mf")}
                className="w-full rounded-2xl border border-[var(--surface-border)] bg-white px-4 py-3 text-sm text-[var(--text-primary)] outline-none transition focus:border-[rgba(11,92,87,0.35)]"
              >
                <option value="3mf">3MF (рекомендовано)</option>
                <option value="stl">STL</option>
              </select>
            </div>

            <SliderField
              label="Розмір моделі"
              valueLabel={`${modelSizeMm.toFixed(0)} мм`}
              min={50}
              max={500}
              step={10}
              value={modelSizeMm}
              onChange={setModelSizeMm}
              hint={`${(modelSizeMm / 10).toFixed(1)} см на фінальній моделі.`}
            />
          </div>

          <div className="rounded-[24px] border border-[rgba(11,92,87,0.18)] bg-[rgba(11,92,87,0.06)] p-4">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="text-sm font-semibold text-[var(--text-primary)]">Основна дія</div>
                <div className="mt-1 text-xs leading-5 text-[var(--text-secondary)]">
                  {showHexGrid
                    ? "Коли зони вже обрані, запускайте серію одним натисканням."
                    : "Після позначення ділянки генерація стартує одразу з поточними параметрами."}
                </div>
              </div>

              <div className="flex flex-col gap-2 sm:items-end">
                <button
                  type="button"
                  onClick={showHexGrid ? handleGenerateZones : handleGenerate}
                  disabled={!selectionReady || isGenerating}
                  className="inline-flex min-h-12 items-center justify-center gap-2 rounded-full bg-[var(--accent-strong)] px-5 py-3 text-sm font-semibold text-white shadow-[0_16px_32px_rgba(11,92,87,0.24)] transition hover:bg-[var(--accent)] disabled:cursor-not-allowed disabled:bg-slate-400"
                >
                  {isGenerating ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Генерація...
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4" />
                      {primaryActionLabel}
                    </>
                  )}
                </button>

                {showHexGrid && selectedZones.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setSelectedZones([])}
                    className="rounded-full px-4 py-2 text-xs font-semibold text-[var(--text-secondary)] transition hover:bg-black/5"
                  >
                    Очистити вибір
                  </button>
                )}
              </div>
            </div>
          </div>
        </SectionFrame>

        <SectionFrame
          eyebrow="Output"
          title="Статус, батч і завантаження"
          description="Тут зібрані прогрес задачі, пакетний режим, готові файли та активна зона для прев'ю."
        >
          <div className="grid gap-3 sm:grid-cols-2">
            <InfoPill label="Стан рендера" value={statusSummary} tone={downloadUrl ? "success" : "default"} />
            <InfoPill
              label="Активна задача"
              value={activeTaskId ? activeTaskId.slice(0, 12) : "ще не створена"}
              tone={activeTaskId ? "accent" : "default"}
            />
          </div>

          {isGenerating && (
            <div className="rounded-[24px] border border-[rgba(11,92,87,0.15)] bg-[rgba(11,92,87,0.06)] p-4">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-sm font-semibold text-[var(--text-primary)]">
                  <Loader2 className="h-4 w-4 animate-spin text-[var(--accent-strong)]" />
                  {status || "Обробка..."}
                </div>
                <span className="text-sm font-semibold text-[var(--accent-strong)]">{progress}%</span>
              </div>
              <div className="mt-3 h-2 overflow-hidden rounded-full bg-white/80">
                <div
                  className="h-full rounded-full bg-[var(--accent-strong)] transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {taskIds.length > 1 && (
            <div className="rounded-[24px] border border-[var(--surface-border)] bg-white/80 p-4">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <div className="text-sm font-semibold text-[var(--text-primary)]">Згенеровані зони</div>
                  <div className="mt-1 text-xs leading-5 text-[var(--text-secondary)]">
                    Оберіть активну зону для прев'ю й завантаження або увімкніть спільний перегляд усіх зон.
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => {
                    const next = !showAllZones;
                    setShowAllZones(next);
                    setError(null);
                    setDownloadUrl(null);
                  }}
                  className={`rounded-full px-4 py-2 text-xs font-semibold ${
                    showAllZones
                      ? "bg-[rgba(11,92,87,0.12)] text-[var(--accent-strong)]"
                      : "bg-slate-900 text-white"
                  }`}
                >
                  {showAllZones ? "Показувати одну зону" : "Показати всі зони разом"}
                </button>
              </div>

              <div className="mt-4 max-h-52 space-y-2 overflow-auto pr-1">
                {taskIds.map((id) => (
                  <button
                    key={id}
                    type="button"
                    onClick={async () => {
                      if (showAllZones) return;
                      setActiveTaskId(id);
                      setError(null);

                      const taskState = taskStatuses[id];
                      if (taskState && taskState.status === "completed" && taskState.download_url) {
                        setDownloadUrl(taskState.download_url);
                        return;
                      }

                      setDownloadUrl(null);
                      try {
                        const resp = await api.getStatus(id);
                        const single = resp as any;
                        if (single && single.status === "completed" && single.download_url) {
                          setDownloadUrl(single.download_url);
                        }
                      } catch {
                        // ignore single fetch issues
                      }
                    }}
                    className={`w-full rounded-[18px] border px-3 py-3 text-left transition ${
                      id === activeTaskId
                        ? "border-[rgba(11,92,87,0.22)] bg-[rgba(11,92,87,0.08)]"
                        : "border-[var(--surface-border)] bg-white hover:bg-[rgba(15,23,42,0.03)]"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-sm font-semibold text-[var(--text-primary)]">{id}</span>
                      {id === activeTaskId && <CheckCircle2 className="h-4 w-4 text-[var(--accent-strong)]" />}
                    </div>
                    <div className="mt-1 text-xs text-[var(--text-secondary)]">
                      {taskStatuses[id]?.status || "Очікує оновлення статусу"}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {downloadUrl && (
            <button
              type="button"
              onClick={handleDownload}
              className="inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-full bg-emerald-600 px-5 py-3 text-sm font-semibold text-white shadow-[0_16px_30px_rgba(5,150,105,0.22)] transition hover:bg-emerald-500"
            >
              <Download className="h-4 w-4" />
              Завантажити модель
            </button>
          )}

          {error && (
            <div className="rounded-[22px] border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}
        </SectionFrame>

        <SectionFrame
          eyebrow="Advanced"
          title="Тонке налаштування моделі"
          description="Детальні параметри сховані в акордеони, щоб мобільний сценарій залишався чистим і швидким."
        >
          <AccordionButton
            title="Дороги"
            description="Ширина, висота та посадка доріг у рельєф."
            icon={Route}
            isOpen={openPanels.roads}
            onClick={() => togglePanel("roads")}
          />
          {openPanels.roads && (
            <div className="space-y-3">
              <SliderField
                label="Ширина доріг"
                valueLabel={roadWidthMultiplier.toFixed(1)}
                min={0.3}
                max={2}
                step={0.1}
                value={roadWidthMultiplier}
                onChange={setRoadWidthMultiplier}
              />
              <SliderField
                label="Висота доріг"
                valueLabel={`${roadHeightMm.toFixed(1)} мм`}
                min={0.2}
                max={3}
                step={0.1}
                value={roadHeightMm}
                onChange={setRoadHeightMm}
              />
              <SliderField
                label="Втиснення в рельєф"
                valueLabel={`${roadEmbedMm.toFixed(1)} мм`}
                min={0}
                max={1}
                step={0.1}
                value={roadEmbedMm}
                onChange={setRoadEmbedMm}
                hint="Допомагає прибрати візуальне мерехтіння на стику дороги з рельєфом."
              />
            </div>
          )}

          <AccordionButton
            title="Будівлі"
            description="Контроль мінімальної висоти, масштабу та фундаменту."
            icon={Building2}
            isOpen={openPanels.buildings}
            onClick={() => togglePanel("buildings")}
          />
          {openPanels.buildings && (
            <div className="space-y-3">
              <SliderField
                label="Мінімальна висота"
                valueLabel={`${buildingMinHeight.toFixed(1)} м`}
                min={1}
                max={10}
                step={0.5}
                value={buildingMinHeight}
                onChange={setBuildingMinHeight}
              />
              <SliderField
                label="Множник висоти"
                valueLabel={buildingHeightMultiplier.toFixed(1)}
                min={0.5}
                max={3}
                step={0.1}
                value={buildingHeightMultiplier}
                onChange={setBuildingHeightMultiplier}
              />
              <SliderField
                label="Фундамент"
                valueLabel={`${buildingFoundationMm.toFixed(1)} мм`}
                min={0.1}
                max={3}
                step={0.1}
                value={buildingFoundationMm}
                onChange={setBuildingFoundationMm}
              />
              <SliderField
                label="Втиснення в основу"
                valueLabel={`${buildingEmbedMm.toFixed(1)} мм`}
                min={0}
                max={1}
                step={0.1}
                value={buildingEmbedMm}
                onChange={setBuildingEmbedMm}
                hint="Якщо будівлі ніби провалюються під землю — зменшуйте значення."
              />
            </div>
          )}

          <AccordionButton
            title="Рельєф, вода та AMS"
            description="Terrain, water depth і друкований flat mode в одному місці."
            icon={Mountain}
            isOpen={openPanels.terrain}
            onClick={() => togglePanel("terrain")}
          />
          {openPanels.terrain && (
            <div className="space-y-3">
              <SliderField
                label="Глибина води"
                valueLabel={`${waterDepth.toFixed(1)} мм`}
                min={0.5}
                max={5}
                step={0.5}
                value={waterDepth}
                onChange={setWaterDepth}
              />

              <CheckboxRow
                label="AMS / Flat Mode"
                description="Оптимізація під шаровий друк: рельєф стає пласкішим і більш передбачуваним."
                checked={isAmsMode}
                onChange={setAmsMode}
              />

              {!isAmsMode && (
                <>
                  <CheckboxRow
                    label="Увімкнути рельєф"
                    description="Керуйте terrain-параметрами лише коли рельєф справді потрібен у фінальній моделі."
                    checked={terrainEnabled}
                    onChange={setTerrainEnabled}
                  />

                  {terrainEnabled && (
                    <div className="space-y-3">
                      <SliderField
                        label="Множник висоти рельєфу"
                        valueLabel={terrainZScale.toFixed(1)}
                        min={0.5}
                        max={3}
                        step={0.1}
                        value={terrainZScale}
                        onChange={setTerrainZScale}
                      />
                      <SliderField
                        label="Деталізація mesh"
                        valueLabel={`${terrainResolution}×${terrainResolution}`}
                        min={120}
                        max={320}
                        step={20}
                        value={terrainResolution}
                        onChange={(value) => setTerrainResolution(parseInt(String(value), 10))}
                        hint="Більше значення дає дрібнішу сітку, але уповільнює генерацію."
                      />
                      <SliderField
                        label="Terrarium zoom"
                        valueLabel={String(terrariumZoom)}
                        min={11}
                        max={16}
                        step={1}
                        value={terrariumZoom}
                        onChange={(value) => setTerrariumZoom(parseInt(String(value), 10))}
                        hint="14–15 найчастіше оптимальні для балансу якості й швидкості."
                      />
                      <SliderField
                        label="Товщина основи рельєфу"
                        valueLabel={`${terrainBaseThicknessMm.toFixed(1)} мм`}
                        min={0.2}
                        max={20}
                        step={0.1}
                        value={terrainBaseThicknessMm}
                        onChange={setTerrainBaseThicknessMm}
                        hint="Тонка підложка підходить для друку, але не варто робити її надто крихкою."
                      />
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          <AccordionButton
            title="Компоненти в прев'ю"
            description="Окремо вмикайте рельєф, дороги, будівлі, воду та парки."
            icon={Layers3}
            isOpen={openPanels.preview}
            onClick={() => togglePanel("preview")}
          />
          {openPanels.preview && (
            <div className="space-y-3">
              <CheckboxRow
                label="Рельєф"
                description="Основа і terrain шар у 3D-прев'ю."
                checked={previewIncludeBase}
                onChange={setPreviewIncludeBase}
              />
              <CheckboxRow
                label="Дороги"
                description="Показувати дорожню сітку на моделі."
                checked={previewIncludeRoads}
                onChange={setPreviewIncludeRoads}
              />
              <CheckboxRow
                label="Будівлі"
                description="Видимість будівель у прев'ю."
                checked={previewIncludeBuildings}
                onChange={setPreviewIncludeBuildings}
              />
              <CheckboxRow
                label="Вода"
                description="Показувати водойми в прев'ю."
                checked={previewIncludeWater}
                onChange={setPreviewIncludeWater}
              />
              <CheckboxRow
                label="Парки"
                description="Зелений шар парків і зелених зон."
                checked={previewIncludeParks}
                onChange={setPreviewIncludeParks}
              />
              <div className="rounded-[20px] border border-[rgba(15,118,110,0.12)] bg-[rgba(15,118,110,0.05)] px-4 py-3 text-xs leading-5 text-[var(--text-secondary)]">
                Налаштування видимості зберігаються в сесії та впливають на наступний запуск генерації.
              </div>
            </div>
          )}
        </SectionFrame>
      </div>
    </div>
  );
}
