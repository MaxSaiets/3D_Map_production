"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Download, KeyRound, Loader2, Play, Type } from "lucide-react";
import { api } from "@/lib/api";
import { useGenerationStore } from "@/store/generation-store";
import {
  type KeychainBaseShape,
  type KeychainDesignerConfig,
  type KeychainLoopStyle,
} from "@/components/KeychainDesigner";

function SliderField({
  label,
  valueLabel,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string;
  valueLabel: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block rounded-[22px] border border-[var(--surface-border)] bg-white/80 p-3">
      <div className="flex items-start justify-between gap-3">
        <span className="text-sm font-medium text-[var(--text-primary)]">{label}</span>
        <span className="text-sm font-semibold text-[var(--accent-strong)]">{valueLabel}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-3 w-full"
      />
    </label>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[18px] border border-[var(--surface-border)] bg-white/80 px-3 py-2">
      <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">
        {label}
      </div>
      <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]" data-testid={`metric-${label.toLowerCase()}`}>
        {value}
      </div>
    </div>
  );
}

function ChoiceButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-[16px] border px-3 py-2 text-sm font-semibold transition ${
        active
          ? "border-[rgba(11,92,87,0.38)] bg-[rgba(15,118,110,0.12)] text-[var(--accent-strong)]"
          : "border-[var(--surface-border)] bg-white/80 text-[var(--text-primary)] hover:bg-white"
      }`}
    >
      {label}
    </button>
  );
}

function fitDesign(next: KeychainDesignerConfig): KeychainDesignerConfig {
  const bodyWidthMm = Math.min(140, Math.max(35, next.bodyWidthMm));
  const bodyHeightMm = Math.min(96, Math.max(26, next.bodyHeightMm));
  const minMapWidthMm = Math.min(28, bodyWidthMm);
  const minMapHeightMm = Math.min(18, bodyHeightMm);
  const mapXMm = Math.min(Math.max(next.mapXMm, 0), Math.max(bodyWidthMm - minMapWidthMm, 0));
  const mapYMm = Math.min(Math.max(next.mapYMm, 0), Math.max(bodyHeightMm - minMapHeightMm, 0));
  const loopMargin = Math.max(next.loopOuterMm * 0.85, 4);
  return {
    ...next,
    bodyWidthMm,
    bodyHeightMm,
    mapXMm,
    mapYMm,
    mapWidthMm: Math.min(Math.max(next.mapWidthMm, minMapWidthMm), Math.max(bodyWidthMm - mapXMm, minMapWidthMm)),
    mapHeightMm: Math.min(Math.max(next.mapHeightMm, minMapHeightMm), Math.max(bodyHeightMm - mapYMm, minMapHeightMm)),
    labelXMm: Math.min(Math.max(next.labelXMm, 4), Math.max(bodyWidthMm - 4, 4)),
    labelYMm: Math.min(Math.max(next.labelYMm, 4), Math.max(bodyHeightMm - 4, 4)),
    labelWidthMm: Math.min(Math.max(next.labelWidthMm, 8), bodyWidthMm),
    labelTextHeightMm: Math.min(Math.max(next.labelTextHeightMm, 2.4), 8.5),
    labelStrokeMm: Math.min(Math.max(next.labelStrokeMm, 0.4), 2.0),
    loopXMm: Math.min(Math.max(next.loopXMm, -loopMargin), bodyWidthMm + loopMargin),
    loopYMm: Math.min(Math.max(next.loopYMm, -loopMargin), bodyHeightMm + loopMargin),
    loopInnerMm: Math.min(Math.max(next.loopInnerMm, 1.6), Math.max(next.loopOuterMm - 1.4, 1.6)),
    rimWidthMm: Math.min(Math.max(next.rimWidthMm, 0), 6),
    rimHeightMm: Math.min(Math.max(next.rimHeightMm, 0), 3),
  };
}

function selectedAreaMeters(selectedArea: ReturnType<typeof useGenerationStore.getState>["selectedArea"]) {
  if (!selectedArea) return null;
  const north = selectedArea.getNorth();
  const south = selectedArea.getSouth();
  const east = selectedArea.getEast();
  const west = selectedArea.getWest();
  const latMid = ((north + south) / 2) * (Math.PI / 180);
  return {
    widthM: Math.abs(east - west) * 111_320 * Math.max(Math.cos(latMid), 0.2),
    heightM: Math.abs(north - south) * 111_320,
  };
}

export function KeychainControlPanel({
  label,
  onLabelChange,
  design,
  onDesignChange,
}: {
  label: string;
  onLabelChange: (value: string) => void;
  design: KeychainDesignerConfig;
  onDesignChange: (value: KeychainDesignerConfig) => void;
}) {
  const {
    selectedArea,
    isGenerating,
    taskGroupId,
    activeTaskId,
    progress,
    status,
    downloadUrl,
    taskStatuses,
    setGenerating,
    setTaskGroup,
    setActiveTaskId,
    setTaskStatuses,
    setShowAllZones,
    updateProgress,
    setDownloadUrl,
  } = useGenerationStore();

  const [error, setError] = useState<string | null>(null);
  const [baseThicknessMm, setBaseThicknessMm] = useState(2.0);
  const [roadLayerMm, setRoadLayerMm] = useState(0.44);
  const [parkLayerMm, setParkLayerMm] = useState(0.34);
  const [waterLayerMm, setWaterLayerMm] = useState(0.28);
  const [buildingMaxMm, setBuildingMaxMm] = useState(2.2);
  const pollingInFlightRef = useRef(false);
  const printScale = useMemo(() => {
    const size = selectedAreaMeters(selectedArea);
    if (!size) return null;
    const metersPerMm = Math.max(size.widthM / Math.max(design.mapWidthMm, 1), size.heightM / Math.max(design.mapHeightMm, 1));
    const minPrintableWorldM = metersPerMm * 0.4;
    return {
      ...size,
      metersPerMm,
      minPrintableWorldM,
      tooLarge: metersPerMm > 8,
    };
  }, [selectedArea, design.mapWidthMm, design.mapHeightMm]);

  const updateDesign = (patch: Partial<KeychainDesignerConfig>) => {
    onDesignChange(fitDesign({ ...design, ...patch }));
  };

  const placeLoop = (position: "top-left" | "top-right" | "right" | "bottom-left") => {
    const presets = {
      "top-left": { loopXMm: Math.min(8.5, design.bodyWidthMm / 2), loopYMm: -4, loopAngleDeg: 0 },
      "top-right": { loopXMm: Math.max(design.bodyWidthMm - 8.5, design.bodyWidthMm / 2), loopYMm: -4, loopAngleDeg: 0 },
      right: { loopXMm: design.bodyWidthMm + Math.max(design.loopOuterMm * 0.58, 3.2), loopYMm: design.bodyHeightMm / 2, loopAngleDeg: 270 },
      "bottom-left": { loopXMm: Math.min(8.5, design.bodyWidthMm / 2), loopYMm: design.bodyHeightMm + Math.max(design.loopOuterMm * 0.58, 3.2), loopAngleDeg: 180 },
    } satisfies Record<string, Partial<KeychainDesignerConfig>>;
    updateDesign(presets[position]);
  };

  useEffect(() => {
    if (!taskGroupId || !isGenerating) return;

    const interval = window.setInterval(async () => {
      if (pollingInFlightRef.current) return;
      pollingInFlightRef.current = true;
      try {
        const resp = await api.getStatus(taskGroupId);
        const task = resp as any;
        setTaskStatuses({ [task.task_id]: task });
        updateProgress(task.progress, task.message);
        if (task.status === "completed") {
          setGenerating(false);
          setDownloadUrl(task.download_url);
        } else if (task.status === "failed") {
          setGenerating(false);
          setError(task.message || "Брелок не згенерувався");
        }
      } catch (pollError) {
        console.error("[Keychain] status error", pollError);
      } finally {
        pollingInFlightRef.current = false;
      }
    }, 3500);

    return () => {
      window.clearInterval(interval);
      pollingInFlightRef.current = false;
    };
  }, [taskGroupId, isGenerating, setGenerating, setTaskStatuses, setDownloadUrl, updateProgress]);

  const handleGenerate = async () => {
    if (!selectedArea) {
      setError("Спочатку позначте ділянку на мапі");
      return;
    }
    if (printScale?.tooLarge) {
      setError("Зона завелика для брелка: зменшіть crop на мапі або збільшіть область карти на брелку. Мінімальна друкована деталь має бути від 0.4 мм.");
      return;
    }

    setError(null);
    setGenerating(true);
    setShowAllZones(false);

    try {
      const response = await api.generateModel({
        north: selectedArea.getNorth(),
        south: selectedArea.getSouth(),
        east: selectedArea.getEast(),
        west: selectedArea.getWest(),
        road_width_multiplier: 0.62,
        road_height_mm: roadLayerMm,
        road_embed_mm: 0,
        building_min_height: 1,
        building_height_multiplier: 1,
        building_foundation_mm: 0.2,
        building_embed_mm: 0,
        water_depth: 0.2,
        terrain_enabled: false,
        terrain_z_scale: 0,
        terrain_base_thickness_mm: baseThicknessMm,
        terrain_resolution: 120,
        terrarium_zoom: 13,
        flatten_buildings_on_terrain: false,
        export_format: "3mf",
        model_size_mm: Math.max(design.bodyWidthMm, design.bodyHeightMm),
        context_padding_m: 35,
        is_ams_mode: false,
        flat_plate_mode: true,
        keychain_mode: true,
        keychain_label: label,
        keychain_base_shape: design.baseShape,
        keychain_loop_style: design.loopStyle,
        keychain_loop_angle_deg: design.loopAngleDeg,
        keychain_body_width_mm: design.bodyWidthMm,
        keychain_body_height_mm: design.bodyHeightMm,
        keychain_map_x_mm: design.mapXMm,
        keychain_map_y_mm: design.mapYMm,
        keychain_map_width_mm: design.mapWidthMm,
        keychain_map_height_mm: design.mapHeightMm,
        keychain_loop_center_x_mm: design.loopXMm,
        keychain_loop_center_y_mm: design.loopYMm,
        keychain_label_center_x_mm: design.labelXMm,
        keychain_label_center_y_mm: design.labelYMm,
        keychain_label_angle_deg: design.labelAngleDeg,
        keychain_loop_outer_radius_mm: design.loopOuterMm,
        keychain_loop_inner_radius_mm: design.loopInnerMm,
        keychain_corner_radius_mm: design.cornerRadiusMm,
        keychain_label_band_height_mm: design.labelBandMm,
        keychain_label_raise_mm: 0.45,
        keychain_label_text_height_mm: design.labelTextHeightMm,
        keychain_label_width_mm: design.labelWidthMm,
        keychain_label_stroke_mm: design.labelStrokeMm,
        keychain_rim_width_mm: design.rimWidthMm,
        keychain_rim_height_mm: design.rimHeightMm,
        flat_water_layer_mm: waterLayerMm,
        flat_roads_layer_mm: roadLayerMm,
        flat_parks_layer_mm: parkLayerMm,
        flat_max_building_height_mm: buildingMaxMm,
        preview_mode: false,
        preview_include_base: true,
        preview_include_roads: true,
        preview_include_buildings: true,
        preview_include_water: true,
        preview_include_parks: true,
      });
      setTaskGroup(response.task_id, [response.task_id]);
      setActiveTaskId(response.task_id);
    } catch (generateError: any) {
      setError(generateError.message || "Помилка генерації брелка");
      setGenerating(false);
    }
  };

  const handleDownload = async () => {
    if (!activeTaskId || !downloadUrl) return;
    try {
      const blob = await api.downloadModel(activeTaskId, "3mf");
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      const filename = taskStatuses[activeTaskId]?.download_url_3mf?.split(/[\\/]/).pop() || "map_keychain.3mf";
      link.download = filename.endsWith(".3mf") ? filename : "map_keychain.3mf";
      document.body.appendChild(link);
      link.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(link);
    } catch (downloadError) {
      console.error("[Keychain] download error", downloadError);
      setError("Не вдалося завантажити 3MF");
    }
  };

  const canGenerate = Boolean(selectedArea) && !isGenerating && !printScale?.tooLarge;
  const currentStatus = isGenerating ? `${progress}% • ${status || "Генерація брелка"}` : downloadUrl ? "3MF готовий" : "Готово";

  return (
    <div className="h-full overflow-y-auto px-4 py-4 sm:px-5">
      <div className="space-y-4 pb-8">
        <section className="rounded-[28px] border border-[var(--surface-border)] bg-[var(--surface-panel-strong)] p-4 shadow-[0_12px_36px_rgba(15,23,42,0.06)] sm:p-5">
          <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[var(--text-secondary)]">
            Keychain Studio
          </p>
          <h2 className="mt-1 font-title text-xl font-semibold text-[var(--text-primary)]">
            Брелок як готовий продукт
          </h2>
          <p className="mt-1 text-sm leading-6 text-[var(--text-secondary)]">
            Основа, шари карти, невисокі будівлі, підпис і посилена петля збираються в одному 3MF.
          </p>

          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <Metric label="Стан" value={currentStatus} />
            <Metric label="Розмір" value={`${Math.round(design.bodyWidthMm)} x ${Math.round(design.bodyHeightMm)} мм`} />
            <Metric label="Карта" value={`${Math.round(design.mapWidthMm)} x ${Math.round(design.mapHeightMm)} мм`} />
            <Metric
              label="Масштаб"
              value={printScale ? `${printScale.metersPerMm.toFixed(1)} м/мм` : "немає crop"}
            />
            <Metric label="Вушко" value={`${design.loopStyle} • ${Math.round(design.loopAngleDeg)}°`} />
          </div>
          {printScale && (
            <div className={`mt-3 rounded-[18px] border px-3 py-2 text-xs leading-5 ${
              printScale.tooLarge
                ? "border-red-200 bg-red-50 text-red-700"
                : "border-[rgba(11,92,87,0.22)] bg-[rgba(15,118,110,0.08)] text-[var(--accent-strong)]"
            }`}>
              Мінімальна деталь 0.4 мм зараз дорівнює ~{printScale.minPrintableWorldM.toFixed(1)} м у реальності.
              {printScale.tooLarge ? " Crop завеликий: дрібні дороги, вода й текст у слайсері розсипляться." : " Масштаб придатний для FDM."}
            </div>
          )}
        </section>

        <section className="rounded-[28px] border border-[var(--surface-border)] bg-[var(--surface-panel-strong)] p-4 shadow-[0_12px_36px_rgba(15,23,42,0.06)] sm:p-5">
          <div className="flex items-start gap-3">
            <div className="rounded-2xl bg-[rgba(11,92,87,0.08)] p-2 text-[var(--accent-strong)]">
              <Type size={18} />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-[var(--text-primary)]">Підпис знизу</h3>
              <p className="mt-1 text-xs leading-5 text-[var(--text-secondary)]">
                Напис у прев'ю змінюється одразу. У фінальний 3MF потрапляє поточне значення на момент генерації.
              </p>
            </div>
          </div>
          <input
            value={label}
            onChange={(event) => onLabelChange(event.target.value.toUpperCase().slice(0, 28))}
            placeholder="KYIV MAP"
            className="mt-4 w-full rounded-[20px] border border-[var(--surface-border)] bg-white/90 px-4 py-3 text-sm font-semibold uppercase tracking-[0.08em] text-[var(--text-primary)] outline-none transition focus:border-[var(--accent)]"
          />
        </section>

        <section className="rounded-[28px] border border-[var(--surface-border)] bg-[var(--surface-panel-strong)] p-4 shadow-[0_12px_36px_rgba(15,23,42,0.06)] sm:p-5">
          <div className="flex items-start gap-3">
            <div className="rounded-2xl bg-[rgba(11,92,87,0.08)] p-2 text-[var(--accent-strong)]">
              <KeyRound size={18} />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-[var(--text-primary)]">Механіка брелка</h3>
              <p className="mt-1 text-xs leading-5 text-[var(--text-secondary)]">
                Петля друкується як частина основи, нижня смуга резервується під текст.
              </p>
            </div>
          </div>

          <div className="mt-4 space-y-3">
            <SliderField label="Ширина основи" valueLabel={`${design.bodyWidthMm.toFixed(0)} мм`} min={35} max={140} step={1} value={design.bodyWidthMm} onChange={(value) => updateDesign({ bodyWidthMm: value })} />
            <SliderField label="Висота основи" valueLabel={`${design.bodyHeightMm.toFixed(0)} мм`} min={26} max={96} step={1} value={design.bodyHeightMm} onChange={(value) => updateDesign({ bodyHeightMm: value })} />
            <SliderField label="Товщина основи" valueLabel={`${baseThicknessMm.toFixed(1)} мм`} min={1.6} max={4.0} step={0.1} value={baseThicknessMm} onChange={setBaseThicknessMm} />
            <SliderField label="Зовнішній радіус петлі" valueLabel={`${design.loopOuterMm.toFixed(1)} мм`} min={4.5} max={11} step={0.1} value={design.loopOuterMm} onChange={(value) => updateDesign({ loopOuterMm: value, loopInnerMm: Math.min(design.loopInnerMm, value - 1.4) })} />
            <SliderField label="Отвір під кільце" valueLabel={`${design.loopInnerMm.toFixed(1)} мм`} min={2.0} max={6.5} step={0.1} value={design.loopInnerMm} onChange={(value) => updateDesign({ loopInnerMm: Math.min(value, design.loopOuterMm - 1.4) })} />
            <SliderField label="Заокруглення кутів" valueLabel={`${design.cornerRadiusMm.toFixed(1)} мм`} min={0} max={9} step={0.1} value={design.cornerRadiusMm} onChange={(value) => updateDesign({ cornerRadiusMm: value })} />
            <SliderField label="Смуга під напис" valueLabel={`${design.labelBandMm.toFixed(1)} мм`} min={5} max={18} step={0.5} value={design.labelBandMm} onChange={(value) => updateDesign({ labelBandMm: value })} />
            <SliderField label="Ширина бокової грані" valueLabel={`${design.rimWidthMm.toFixed(1)} мм`} min={0} max={5} step={0.1} value={design.rimWidthMm} onChange={(value) => updateDesign({ rimWidthMm: value })} />
            <SliderField label="Висота бокової грані" valueLabel={`${design.rimHeightMm.toFixed(2)} мм`} min={0} max={1.6} step={0.05} value={design.rimHeightMm} onChange={(value) => updateDesign({ rimHeightMm: value })} />
          </div>
        </section>

        <section className="rounded-[28px] border border-[var(--surface-border)] bg-[var(--surface-panel-strong)] p-4 shadow-[0_12px_36px_rgba(15,23,42,0.06)] sm:p-5">
          <h3 className="text-sm font-semibold text-[var(--text-primary)]">Форма і вушко</h3>
          <div className="mt-4 space-y-4">
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">Форма підложки</div>
              <div className="grid grid-cols-2 gap-2">
                {([
                  ["rounded", "Прямокутник"],
                  ["capsule", "Капсула"],
                  ["tag", "Tag"],
                  ["octagon", "Октагон"],
                ] as Array<[KeychainBaseShape, string]>).map(([shape, text]) => (
                  <ChoiceButton key={shape} label={text} active={design.baseShape === shape} onClick={() => updateDesign({ baseShape: shape })} />
                ))}
              </div>
            </div>
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">Тип вушка</div>
              <div className="grid grid-cols-2 gap-2">
                {([
                  ["round", "Кругле"],
                  ["teardrop", "Крапля"],
                  ["slot", "Слот"],
                  ["side-tab", "Плашка"],
                ] as Array<[KeychainLoopStyle, string]>).map(([style, text]) => (
                  <ChoiceButton key={style} label={text} active={design.loopStyle === style} onClick={() => updateDesign({ loopStyle: style })} />
                ))}
              </div>
            </div>
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">Позиція вушка</div>
              <div className="grid grid-cols-2 gap-2">
                <ChoiceButton label="Зліва зверху" active={design.loopXMm < design.bodyWidthMm / 2 && design.loopYMm < 0} onClick={() => placeLoop("top-left")} />
                <ChoiceButton label="Справа зверху" active={design.loopXMm > design.bodyWidthMm / 2 && design.loopYMm < 0} onClick={() => placeLoop("top-right")} />
                <ChoiceButton label="Справа" active={design.loopXMm > design.bodyWidthMm} onClick={() => placeLoop("right")} />
                <ChoiceButton label="Знизу зліва" active={design.loopYMm > design.bodyHeightMm} onClick={() => placeLoop("bottom-left")} />
              </div>
            </div>
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">Поворот</div>
              <div className="grid grid-cols-4 gap-2">
                {[0, 90, 180, 270].map((angle) => (
                  <ChoiceButton key={`loop-${angle}`} label={`${angle}°`} active={design.loopAngleDeg === angle} onClick={() => updateDesign({ loopAngleDeg: angle })} />
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="rounded-[28px] border border-[var(--surface-border)] bg-[var(--surface-panel-strong)] p-4 shadow-[0_12px_36px_rgba(15,23,42,0.06)] sm:p-5">
          <h3 className="text-sm font-semibold text-[var(--text-primary)]">Зона карти і напис</h3>
          <div className="mt-4 space-y-3">
            <SliderField label="Ширина зони карти" valueLabel={`${design.mapWidthMm.toFixed(0)} мм`} min={Math.min(28, design.bodyWidthMm)} max={design.bodyWidthMm} step={1} value={design.mapWidthMm} onChange={(value) => updateDesign({ mapWidthMm: value })} />
            <SliderField label="Висота зони карти" valueLabel={`${design.mapHeightMm.toFixed(0)} мм`} min={Math.min(18, design.bodyHeightMm)} max={design.bodyHeightMm} step={1} value={design.mapHeightMm} onChange={(value) => updateDesign({ mapHeightMm: value })} />
            <SliderField label="Ширина напису" valueLabel={`${design.labelWidthMm.toFixed(0)} мм`} min={8} max={design.bodyWidthMm} step={1} value={design.labelWidthMm} onChange={(value) => updateDesign({ labelWidthMm: value })} />
            <SliderField label="Висота літер" valueLabel={`${design.labelTextHeightMm.toFixed(1)} мм`} min={2.4} max={8.5} step={0.1} value={design.labelTextHeightMm} onChange={(value) => updateDesign({ labelTextHeightMm: value })} />
            <SliderField label="Товщина штриха" valueLabel={`${design.labelStrokeMm.toFixed(2)} мм`} min={0.4} max={2.0} step={0.05} value={design.labelStrokeMm} onChange={(value) => updateDesign({ labelStrokeMm: value })} />
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">Поворот тексту</div>
              <div className="grid grid-cols-4 gap-2">
                {[0, 90, 180, 270].map((angle) => (
                  <ChoiceButton key={`label-${angle}`} label={`${angle}°`} active={design.labelAngleDeg === angle} onClick={() => updateDesign({ labelAngleDeg: angle })} />
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="rounded-[28px] border border-[var(--surface-border)] bg-[var(--surface-panel-strong)] p-4 shadow-[0_12px_36px_rgba(15,23,42,0.06)] sm:p-5">
          <h3 className="text-sm font-semibold text-[var(--text-primary)]">Шари карти</h3>
          <div className="mt-4 space-y-3">
            <SliderField label="Дороги" valueLabel={`${roadLayerMm.toFixed(2)} мм`} min={0.4} max={0.9} step={0.01} value={roadLayerMm} onChange={setRoadLayerMm} />
            <SliderField label="Парки" valueLabel={`${parkLayerMm.toFixed(2)} мм`} min={0.18} max={0.75} step={0.01} value={parkLayerMm} onChange={setParkLayerMm} />
            <SliderField label="Вода" valueLabel={`${waterLayerMm.toFixed(2)} мм`} min={0.24} max={0.55} step={0.01} value={waterLayerMm} onChange={setWaterLayerMm} />
            <SliderField label="Максимум будівель" valueLabel={`${buildingMaxMm.toFixed(1)} мм`} min={0.8} max={5.0} step={0.1} value={buildingMaxMm} onChange={setBuildingMaxMm} />
          </div>
        </section>

        {error && (
          <div className="rounded-[20px] border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>
        )}

        <div className="grid gap-3">
          <button
            type="button"
            onClick={handleGenerate}
            disabled={!canGenerate}
            className="inline-flex items-center justify-center gap-2 rounded-[22px] bg-[var(--accent-strong)] px-4 py-3 text-sm font-semibold text-white shadow-[0_14px_30px_rgba(11,92,87,0.24)] transition disabled:cursor-not-allowed disabled:opacity-45"
          >
            {isGenerating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {isGenerating ? "Генерація..." : "Створити брелок"}
          </button>
          <button
            type="button"
            onClick={handleDownload}
            disabled={!downloadUrl || !activeTaskId}
            className="inline-flex items-center justify-center gap-2 rounded-[22px] border border-[var(--surface-border)] bg-white/85 px-4 py-3 text-sm font-semibold text-[var(--text-primary)] transition disabled:cursor-not-allowed disabled:opacity-45"
          >
            <Download className="h-4 w-4" />
            Завантажити 3MF
          </button>
        </div>
      </div>
    </div>
  );
}
