"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useTranslations } from "next-intl";
import { AlignCenter, AlertTriangle, CheckCircle2, Download, KeyRound, Loader2, Map as MapIcon, Play, RotateCcw, ShoppingBag, SlidersHorizontal, Type } from "lucide-react";
import { OrderDialog } from "@/components/OrderDialog";
import { StickyActionBar } from "@/components/StickyActionBar";
import { useAuth } from "@/components/AuthProvider";
import { gatedDownload } from "@/lib/download";
import { fetchQuote, type Quote } from "@/lib/pricing";
import { getKeychainDesignerSvg, svgToPngDataUrl } from "@/lib/capturePreview";
import { api } from "@/lib/api";
import { useGenerationStore } from "@/store/generation-store";
import {
  DEFAULT_KEYCHAIN_DESIGN,
  type KeychainBaseShape,
  type KeychainDesignerConfig,
  type KeychainLabelFontStyle,
  type KeychainLoopStyle,
} from "@/components/KeychainDesigner";

type PanelSection = "product" | "map" | "label" | "review" | "advanced";
type PrintTone = "good" | "warn" | "bad";
type PrintCheck = {
  id: string;
  ok: boolean;
  tone: PrintTone;
  label: string;
  detail: string;
};

const PANEL_SECTIONS: Array<{ id: PanelSection; labelKey: string }> = [
  { id: "product", labelKey: "tabs.product" },
  { id: "map", labelKey: "tabs.map" },
  { id: "label", labelKey: "tabs.label" },
  { id: "review", labelKey: "tabs.review" },
];

const PANEL_CARD_CLASS =
  "rounded-[28px] border border-[var(--surface-border)] bg-[var(--surface-panel-strong)] p-4 shadow-[0_12px_36px_rgba(15,23,42,0.06)] sm:p-5";
// ── FDM 0.4mm nozzle print standards ──────────────────────────────────────────
// На основі дослідження reddit/r/3Dprinting + Bambu/Prusa/All3DP рекомендацій:
// - Nozzle 0.4mm → мінімальна XY-деталь = 0.4mm (1× nozzle)
// - Рекомендовано: 0.5-0.6mm (1.25-1.5× nozzle) — для надійного друку
// - Min wall thickness: 0.8mm (2× nozzle), оптимально 1.2mm (3×)
// - Min hole diameter: 2mm (5× nozzle)
// - Min text stroke: 0.6mm для читання
//
// Для МАСШТАБУ КАРТИ:
// - Звичайна вулиця у місті = 6m wide
// - Мінімальна вулиця/тротуар = 2-3m
// - При 5 м/мм: вулиця 3m = 0.6mm (на межі), 6m = 1.2mm (комфортно)
// - При 7 м/мм: вулиця 3m = 0.43mm (зливаються), 6m = 0.86mm (на межі)
const MIN_PRINT_FEATURE_MM = 0.4;  // 1× nozzle — абсолютний фізичний мінімум
const RECOMMENDED_FEATURE_MM = 0.6; // 1.5× nozzle — рекомендовано
// Пороги масштабу карти (м/мм):
// - ≤ 5.0 м/мм: ЗЕЛЕНИЙ (стандартні вулиці 3m = 0.6mm, чудова деталізація)
// - 5.0–7.0 м/мм: ЖОВТИЙ (вулиці 6m = OK, тонкі тротуари зникнуть)
// - > 7.0 м/мм: ЧЕРВОНИЙ (тільки магістралі читаються, дрібниці зливаються)
const GOOD_METERS_PER_MM = 5.0;
const HARD_METERS_PER_MM = 7.0;

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
      className={`min-h-[44px] rounded-[16px] border px-3 py-2 text-sm font-semibold transition ${
        active
          ? "border-[rgba(11,92,87,0.38)] bg-[rgba(15,118,110,0.12)] text-[var(--accent-strong)]"
          : "border-[var(--surface-border)] bg-white/80 text-[var(--text-primary)] hover:bg-white"
      }`}
    >
      {label}
    </button>
  );
}

function QuickActionButton({
  children,
  onClick,
}: {
  children: ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex min-h-[44px] items-center justify-center gap-2 rounded-[16px] border border-[var(--surface-border)] bg-white/85 px-3 py-2 text-xs font-semibold text-[var(--text-primary)] transition hover:bg-white"
    >
      {children}
    </button>
  );
}

function SectionHeader({
  icon,
  title,
  description,
}: {
  icon: ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="flex items-start gap-3">
      <div className="rounded-2xl bg-[rgba(11,92,87,0.08)] p-2 text-[var(--accent-strong)]">
        {icon}
      </div>
      <div>
        <h3 className="text-sm font-semibold text-[var(--text-primary)]">{title}</h3>
        <p className="mt-1 text-xs leading-5 text-[var(--text-secondary)]">{description}</p>
      </div>
    </div>
  );
}

function PrintabilityCard({
  tone,
  title,
  detail,
}: {
  tone: "idle" | "good" | "warn" | "bad";
  title: string;
  detail: string;
}) {
  const classes = {
    idle: "border-[var(--surface-border)] bg-white/80 text-[var(--text-primary)]",
    good: "border-[rgba(11,92,87,0.22)] bg-[rgba(15,118,110,0.08)] text-[var(--accent-strong)]",
    warn: "border-amber-200 bg-amber-50 text-amber-800",
    bad: "border-red-200 bg-red-50 text-red-700",
  }[tone];
  const Icon = tone === "bad" || tone === "warn" ? AlertTriangle : CheckCircle2;

  return (
    <div className={`rounded-[20px] border px-3 py-3 ${classes}`}>
      <div className="flex items-start gap-2">
        <Icon className="mt-0.5 h-4 w-4 flex-none" />
        <div>
          <div className="text-sm font-semibold">{title}</div>
          <div className="mt-1 text-xs leading-5 opacity-90">{detail}</div>
        </div>
      </div>
    </div>
  );
}

function PrintCheckRow({
  ok,
  label,
  detail,
  tone = ok ? "good" : "bad",
}: {
  ok: boolean;
  label: string;
  detail: string;
  tone?: PrintTone;
}) {
  return (
    <div className={`rounded-[16px] border px-3 py-2 text-xs ${
      tone === "good"
        ? "border-[rgba(11,92,87,0.18)] bg-[rgba(15,118,110,0.06)] text-[var(--accent-strong)]"
        : tone === "warn"
          ? "border-amber-200 bg-amber-50 text-amber-800"
          : "border-red-200 bg-red-50 text-red-700"
    }`}>
      <div className="flex items-start gap-2">
        {tone === "good" ? <CheckCircle2 className="mt-0.5 h-4 w-4 flex-none" /> : <AlertTriangle className="mt-0.5 h-4 w-4 flex-none" />}
        <div>
          <div className="font-semibold">{label}</div>
          <div className="mt-0.5 leading-4 opacity-90">{detail}</div>
        </div>
      </div>
    </div>
  );
}

function fitDesign(next: KeychainDesignerConfig): KeychainDesignerConfig {
  const bodyWidthMm = Math.min(140, Math.max(35, next.bodyWidthMm));
  const bodyHeightMm = Math.min(96, Math.max(26, next.bodyHeightMm));
  const tokenMode = next.baseShape === "token";
  const minMapWidthMm = Math.min(tokenMode ? 18 : 28, bodyWidthMm);
  const minMapHeightMm = Math.min(tokenMode ? 8 : 18, bodyHeightMm);
  const mapXMm = Math.min(Math.max(next.mapXMm, 0), Math.max(bodyWidthMm - minMapWidthMm, 0));
  const mapYMm = Math.min(Math.max(next.mapYMm, 0), Math.max(bodyHeightMm - minMapHeightMm, 0));
  const loopAttachOffset = Math.max(next.loopOuterMm * 0.58, tokenMode ? 2.6 : 3.2);
  const loopMargin = Math.max(loopAttachOffset, 3.8);
  const loopOuterMm = Math.min(Math.max(next.loopOuterMm, tokenMode ? 2.4 : 3.4), tokenMode ? 6 : 11);
  const loopInnerMm = tokenMode
    ? Math.min(Math.max(next.loopInnerMm, 1.5), Math.max(loopOuterMm - 0.8, 1.5))
    : Math.min(Math.max(next.loopInnerMm, 1.4), Math.max(loopOuterMm - 1.35, 1.4));
  const tokenHoleWallMm = loopInnerMm + 2.0;
  return {
    ...next,
    bodyWidthMm,
    bodyHeightMm,
    layoutRotationDeg: ((Math.round((next.layoutRotationDeg || 0) / 90) * 90) % 360 + 360) % 360,
    mapXMm,
    mapYMm,
    mapWidthMm: Math.min(Math.max(next.mapWidthMm, minMapWidthMm), Math.max(bodyWidthMm - mapXMm, minMapWidthMm)),
    mapHeightMm: Math.min(Math.max(next.mapHeightMm, minMapHeightMm), Math.max(bodyHeightMm - mapYMm, minMapHeightMm)),
    mapRotationDeg: ((Math.round((next.mapRotationDeg || 0) / 15) * 15) % 360 + 360) % 360,
    labelXMm: Math.min(Math.max(next.labelXMm, 4), Math.max(bodyWidthMm - 4, 4)),
    labelYMm: Math.min(Math.max(next.labelYMm, 4), Math.max(bodyHeightMm - 4, 4)),
    labelWidthMm: Math.min(Math.max(next.labelWidthMm, 6), bodyWidthMm),
    // Знижено пороги: дозволяємо дрібний текст (від 1.6мм height, 0.4мм stroke).
    // FDM 0.4мм друк може зробити текст 1.6мм висотою якщо stroke ≥ 0.4мм.
    // Користувач сам відповідальний за читабельність — попередження "warn",
    // але не блокуємо вибір.
    labelTextHeightMm: Math.min(Math.max(next.labelTextHeightMm, 2.0), 8.5),
    labelStrokeMm: Math.min(Math.max(next.labelStrokeMm, 0.8), 2.0),
    // Смуга напису ЗАВЖДИ ≥ висота літер + 1.4мм паддінг. Бекенд вписує текст у
    // смугу — якщо смуга менша за текст, експорт зменшував би напис і він НЕ
    // збігався б з превʼю. Тримаючи смугу ≥ тексту, WYSIWYG зберігається.
    labelBandMm: Math.min(Math.max(next.labelBandMm, 3, Math.min(Math.max(next.labelTextHeightMm, 2.0), 8.5) + 1.4), 18),
    loopOuterMm,
    loopInnerMm,
    loopXMm: tokenMode
      ? Math.min(Math.max(next.loopXMm, tokenHoleWallMm), bodyWidthMm - tokenHoleWallMm)
      : Math.min(Math.max(next.loopXMm, -loopMargin), bodyWidthMm + loopMargin),
    loopYMm: tokenMode
      ? Math.min(Math.max(next.loopYMm, tokenHoleWallMm), bodyHeightMm - tokenHoleWallMm)
      : Math.min(Math.max(next.loopYMm, -loopMargin), bodyHeightMm + loopMargin),
    rimWidthMm: Math.min(Math.max(next.rimWidthMm, 0), 6),
    rimHeightMm: Math.min(Math.max(next.rimHeightMm, 0), 3),
  };
}

function selectedAreaMeters(selectedArea: ReturnType<typeof useGenerationStore.getState>["selectedArea"]) {
  if (!selectedArea) return null;
  // Захист: у store може потрапити plain object (битий draft) — не валимо рендер
  if (typeof (selectedArea as any).getNorth !== "function") return null;
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

function shrinkBoundsToMeters(
  selectedArea: ReturnType<typeof useGenerationStore.getState>["selectedArea"],
  targetWidthM: number,
  targetHeightM: number,
) {
  if (!selectedArea) return null;
  if (typeof (selectedArea as any).getCenter !== "function") return null;
  const center = selectedArea.getCenter();
  const latMid = center.lat * (Math.PI / 180);
  const halfLatDeg = (targetHeightM / 2) / 111_320;
  const halfLngDeg = (targetWidthM / 2) / (111_320 * Math.max(Math.cos(latMid), 0.2));
  const BoundsCtor = selectedArea.constructor as new (
    southWest: [number, number],
    northEast: [number, number],
  ) => NonNullable<ReturnType<typeof useGenerationStore.getState>["selectedArea"]>;
  return new BoundsCtor(
    [center.lat - halfLatDeg, center.lng - halfLngDeg],
    [center.lat + halfLatDeg, center.lng + halfLngDeg],
  );
}

export function KeychainControlPanel({
  label,
  onLabelChange,
  design,
  onDesignChange,
  cropRotationDeg = 0,
  cropPolygon = null,
}: {
  label: string;
  onLabelChange: (value: string) => void;
  design: KeychainDesignerConfig;
  onDesignChange: (value: KeychainDesignerConfig) => void;
  /** Поворот рамки вибору ділянки на карті (з MapSelector). Додається до
   *  design.mapRotationDeg щоб згенерована модель показала саме той вміст,
   *  який користувач "обвів" нахиленою рамкою. */
  cropRotationDeg?: number;
  /** 4 кути обернутого rect'а [[lon, lat], ...]. Backend обрізає OSM точно
   *  по полігону а не bbox — гарантовано показує тільки те що обрав юзер. */
  cropPolygon?: Array<[number, number]> | null;
}) {
  const t = useTranslations("kc");
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
    setSelectedArea,
    gpxFocus, setGpxFocus, gpxName, setGpxName,
  } = useGenerationStore();
  // D4 GPX-трек на брелку — точки живуть у store.gpxFocus (їх же використовує
  // карта-оверлей для авто-фокусу зони на маршрут).
  const gpxTrack = gpxFocus?.points ?? null;

  const [error, setError] = useState<string | null>(null);
  const [quote, setQuote] = useState<Quote | null>(null);
  const [label2, setLabel2] = useState("");
  const [backLabel, setBackLabel] = useState("");
  // C3 ТОПО-БРЕЛОК: рельєф висот замість карти (Карпати/Альпи)
  const [topoMode, setTopoMode] = useState(false);
  const [reliefMm, setReliefMm] = useState(2.2);
  const [baseThicknessMm, setBaseThicknessMm] = useState(1.5);

  // Жива орієнтовна ціна брелка (одна базова — розмір на неї не впливає).
  useEffect(() => {
    let alive = true;
    fetchQuote("keychain").then((q) => { if (alive) setQuote(q); });
    return () => { alive = false; };
  }, []);
  const [roadLayerMm, setRoadLayerMm] = useState(0.44);
  const [parkLayerMm, setParkLayerMm] = useState(0.34);
  const [waterLayerMm, setWaterLayerMm] = useState(0.28);
  const [buildingMaxMm, setBuildingMaxMm] = useState(2.2);
  const [uniformBuildingHeight, setUniformBuildingHeight] = useState(false);
  const [activeSection, setActiveSection] = useState<PanelSection>("product");
  const [expertMode, setExpertMode] = useState(false);
  const [orderOpen, setOrderOpen] = useState(false);
  const { getIdToken, openLogin } = useAuth();
  const [quota, setQuota] = useState<{ remaining: number; limit: number; isAdmin?: boolean } | null>(null);
  const pollingInFlightRef = useRef(false);
  const pollFailRef = useRef(0);

  // Free-download counter (login + 5 free, then order). Refreshed after each download.
  const refreshQuota = useCallback(async () => {
    try {
      const token = await getIdToken();
      if (!token) { setQuota(null); return; }
      const API = process.env.NEXT_PUBLIC_API_URL || "";
      const r = await fetch(`${API}/api/account/quota`, { headers: { Authorization: `Bearer ${token}` } });
      if (!r.ok) return;
      const d = await r.json();
      const q = d?.quota || d;
      if (q) setQuota({ remaining: Number(q.remaining ?? 0), limit: Number(q.limit ?? 5), isAdmin: Boolean(q.is_admin) });
    } catch { /* ignore */ }
  }, [getIdToken]);
  useEffect(() => { refreshQuota(); }, [refreshQuota]);
  const printScale = useMemo(() => {
    const size = selectedAreaMeters(selectedArea);
    if (!size) return null;
    const metersPerMm = Math.max(size.widthM / Math.max(design.mapWidthMm, 1), size.heightM / Math.max(design.mapHeightMm, 1));
    const minPrintableWorldM = metersPerMm * MIN_PRINT_FEATURE_MM;
    return {
      ...size,
      metersPerMm,
      minPrintableWorldM,
      tooLarge: metersPerMm > HARD_METERS_PER_MM,
      onEdge: metersPerMm > GOOD_METERS_PER_MM,
    };
  }, [selectedArea, design.mapWidthMm, design.mapHeightMm]);

  const updateDesign = (patch: Partial<KeychainDesignerConfig>) => {
    onDesignChange(fitDesign({ ...design, ...patch }));
  };

  // AUTO-CLAMP після завантаження preset (наприклад Token 45×26 має 3.1mm
  // текст, що нижче print-safe 3.2). Прозоро для користувача — він просто
  // бачить вже виправлені значення замість червоного warning.
  useEffect(() => {
    const safe = fitDesign(design);
    if (
      safe.labelTextHeightMm !== design.labelTextHeightMm ||
      safe.labelStrokeMm !== design.labelStrokeMm
    ) {
      onDesignChange(safe);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [design.labelTextHeightMm, design.labelStrokeMm]);

  const sectionClass = (section: PanelSection) =>
    `${activeSection === section ? "block" : "hidden"} ${PANEL_CARD_CLASS}`;
  // a11y: секції — це tabpanel'и табів-навігації нижче (role=tablist/tab).
  const sectionProps = (section: PanelSection) => ({
    className: sectionClass(section),
    role: "tabpanel",
    id: `kc-tabpanel-${section}`,
    "aria-labelledby": `kc-tab-${section}`,
    hidden: activeSection !== section,
  });

  const visibleSections = expertMode
    ? [...PANEL_SECTIONS, { id: "advanced" as const, labelKey: "tabs.advanced" }]
    : PANEL_SECTIONS;

  const printability = (() => {
    if (!selectedArea || !printScale) {
      return {
        tone: "idle" as const,
        title: t("printability.idle.title"),
        detail: t("printability.idle.detail"),
      };
    }
    if (topoMode) {
      // Топо: масштаб вулиць не важливий — рельєф читається й на великих зонах
      return {
        tone: "good" as const,
        title: t("printability.topo.title"),
        detail: t("printability.topo.detail"),
      };
    }
    if (printScale.tooLarge) {
      return {
        tone: "bad" as const,
        title: t("printability.tooLarge.title"),
        detail: t("printability.tooLarge.detail", { m: printScale.minPrintableWorldM.toFixed(1) }),
      };
    }
    if (printScale.onEdge) {
      return {
        tone: "warn" as const,
        title: t("printability.onEdge.title"),
        detail: t("printability.onEdge.detail", { m: printScale.minPrintableWorldM.toFixed(1) }),
      };
    }
    return {
      tone: "good" as const,
      title: t("printability.good.title"),
      detail: t("printability.good.detail", { m: printScale.minPrintableWorldM.toFixed(1) }),
    };
  })();

  const printChecks = useMemo<PrintCheck[]>(() => {
    const labelTooLong = label.trim().length > Math.max(10, Math.floor(design.labelWidthMm / 1.9));
    const loopHoleDiameterMm = design.loopInnerMm * 2;
    const loopWallMm = design.loopOuterMm - design.loopInnerMm;
    // Топо-режим: масштаб вулиць не важливий — будь-яка зона з рамкою OK
    const cropTone: PrintTone = topoMode
      ? (selectedArea ? "good" : "bad")
      : !selectedArea || !printScale || printScale.tooLarge
        ? "bad"
        : printScale.onEdge
          ? "warn"
          : "good";
    // Текст auto-clamp у fitDesign до 0.6mm/3.2mm — RED warning неможливий.
    // Single shape: good якщо комфорт, warn якщо тільки на межі або довгий напис.
    const textTone: PrintTone = labelTooLong
      ? "warn"
      : design.labelStrokeMm >= 0.7 && design.labelTextHeightMm >= 3.6
        ? "good"
        : "warn";
    const layerTone: PrintTone = topoMode
      ? "good"
      : roadLayerMm >= 0.44 && waterLayerMm >= 0.28 && buildingMaxMm <= 3.0
        ? "good"
        : roadLayerMm >= 0.4 && waterLayerMm >= 0.24 && buildingMaxMm <= 3.2
          ? "warn"
          : "bad";
    return [
      {
        id: "crop",
        ok: cropTone !== "bad",
        tone: cropTone,
        label: t("check.crop.label"),
        detail: printScale
          ? t("check.crop.detail", { min: MIN_PRINT_FEATURE_MM.toFixed(1), m: printScale.minPrintableWorldM.toFixed(1) })
          : t("check.crop.empty"),
      },
      {
        id: "text-stroke",
        ok: true,  // text auto-clamped у fitDesign — завжди friendly
        tone: textTone,
        label: t("check.text.label"),
        detail: labelTooLong
          ? t("check.text.tooLong")
          : t("check.text.detail", { stroke: design.labelStrokeMm.toFixed(2), height: design.labelTextHeightMm.toFixed(1) }),
      },
      {
        id: "loop",
        ok: loopHoleDiameterMm >= 2.8 && loopWallMm >= 1.25,
        tone: loopHoleDiameterMm >= 3.0 && loopWallMm >= 1.45 ? "good" : loopHoleDiameterMm >= 2.8 && loopWallMm >= 1.25 ? "warn" : "bad",
        label: t("check.loop.label"),
        detail: t("check.loop.detail", { hole: loopHoleDiameterMm.toFixed(1), wall: loopWallMm.toFixed(1) }),
      },
      {
        id: "base",
        // OK = можна друкувати (matched backend floor 1.0mm).
        // Тонкі брелки можуть гнутись/зламатись, але це вибір користувача.
        ok: baseThicknessMm >= 1.0 && design.rimWidthMm >= 0.4,
        tone: baseThicknessMm >= 1.8 && design.rimWidthMm >= 0.8 ? "good" : baseThicknessMm >= 1.2 && design.rimWidthMm >= 0.6 ? "warn" : "warn",
        label: t("check.base.label"),
        detail: t("check.base.detail", { base: baseThicknessMm.toFixed(1), rim: design.rimWidthMm.toFixed(1) }),
      },
      {
        id: "layers",
        ok: layerTone !== "bad",
        tone: layerTone,
        label: t("check.layers.label"),
        detail: topoMode
          ? t("check.layers.topo", { relief: reliefMm.toFixed(1) })
          : t("check.layers.detail", { roads: roadLayerMm.toFixed(2), water: waterLayerMm.toFixed(2), buildings: buildingMaxMm.toFixed(1) }),
      },
    ];
  }, [
    t,
    topoMode,
    reliefMm,
    baseThicknessMm,
    buildingMaxMm,
    design.labelStrokeMm,
    design.labelTextHeightMm,
    design.labelWidthMm,
    design.loopInnerMm,
    design.loopOuterMm,
    design.rimWidthMm,
    label,
    printScale,
    roadLayerMm,
    selectedArea,
    waterLayerMm,
  ]);

  const blockingPrintIssues = printChecks.filter((check) => !check.ok);
  const readyTone = blockingPrintIssues.length === 0
    ? printability.tone === "warn"
      ? "warn"
      : "good"
    : "bad";
  const readyLabel = readyTone === "good" ? t("ready.good") : readyTone === "warn" ? t("ready.warn") : t("ready.bad");
  const nextAction = blockingPrintIssues[0]
    ? t("ready.fix", { issue: blockingPrintIssues[0].label.toLowerCase() })
    : printability.tone === "warn"
      ? t("ready.nextWarn")
      : t("ready.nextGood");

  const repairForPrint = () => {
    // Триетапна стратегія для покращення масштабу:
    // 1) Якщо є місце на брелку — розширюємо саму карту (мінімум жертви)
    // 2) Якщо все одно завелика — обрізаємо ділянку на мапі
    // 3) Додатково підвищуємо мінімуми текст/вушко/rim

    let nextMapWidth = design.mapWidthMm;
    let nextMapHeight = design.mapHeightMm;

    if (printScale?.tooLarge) {
      // Спершу максимізуємо саму карту в межах тіла (мінус rim + label band)
      const rim = Math.max(design.rimWidthMm, 1.0);
      const maxMapW = Math.max(design.bodyWidthMm - 2 * rim - 4, 14);
      const maxMapH = Math.max(design.bodyHeightMm - 2 * rim - design.labelBandMm - 4, 14);
      nextMapWidth = Math.max(design.mapWidthMm, maxMapW);
      nextMapHeight = Math.max(design.mapHeightMm, maxMapH);
    }

    // Перевіряємо, чи розширення карти достатньо. Якщо ні — обрізаємо ділянку.
    if (selectedArea && printScale?.tooLarge) {
      // Цільовий масштаб — за ширшу зі сторін
      const targetWidthM = Math.max(nextMapWidth * GOOD_METERS_PER_MM, 60);
      const targetHeightM = Math.max(nextMapHeight * GOOD_METERS_PER_MM, 60);
      const repairedBounds = shrinkBoundsToMeters(selectedArea, targetWidthM, targetHeightM);
      if (repairedBounds) {
        setSelectedArea(repairedBounds);
      }
    }
    const repaired = fitDesign({
      ...design,
      mapWidthMm: nextMapWidth,
      mapHeightMm: nextMapHeight,
      labelStrokeMm: Math.max(design.labelStrokeMm, 0.8),
      labelTextHeightMm: Math.max(design.labelTextHeightMm, 3.8),
      labelWidthMm: Math.max(design.labelWidthMm, Math.min(design.bodyWidthMm - 4, Math.max(22, label.trim().length * 2.1))),
      loopOuterMm: Math.max(design.loopOuterMm, 4.5),
      loopInnerMm: Math.max(design.loopInnerMm, 1.5),
      rimWidthMm: Math.max(design.rimWidthMm, 1.0),
      rimHeightMm: Math.max(design.rimHeightMm, 0.4),
      mapXMm: Math.max((design.bodyWidthMm - nextMapWidth) / 2, 0),
      mapYMm: Math.max((design.bodyHeightMm - design.labelBandMm - nextMapHeight) / 2, 0),
    });
    onDesignChange(repaired);
    setBaseThicknessMm((value) => Math.max(value, 2.0));
    setRoadLayerMm((value) => Math.max(value, 0.44));
    setWaterLayerMm((value) => Math.max(value, 0.28));
    setParkLayerMm((value) => Math.max(value, 0.32));
    setBuildingMaxMm((value) => Math.min(Math.max(value, 1.4), 3.0));
    setActiveSection(blockingPrintIssues[0]?.id === "crop" ? "map" : blockingPrintIssues[0]?.id === "text-stroke" ? "label" : "product");
  };

  const resetToStandard = () => {
    onDesignChange(DEFAULT_KEYCHAIN_DESIGN);
    setActiveSection("map");
  };

  const centerMap = () => {
    updateDesign({
      mapXMm: Math.max((design.bodyWidthMm - design.mapWidthMm) / 2, 0),
      mapYMm: Math.max((design.bodyHeightMm - design.labelBandMm - design.mapHeightMm) / 2, 0),
    });
    setActiveSection("map");
  };

  const maximizeMapArea = () => {
    updateDesign({
      mapXMm: 2,
      mapYMm: 2,
      mapWidthMm: Math.max(design.bodyWidthMm - 4, 28),
      mapHeightMm: Math.max(design.bodyHeightMm - design.labelBandMm - 5, 18),
    });
    setActiveSection("map");
  };

  const centerLabel = () => {
    updateDesign({
      labelXMm: design.bodyWidthMm / 2,
      labelYMm: design.bodyHeightMm - design.labelBandMm / 2,
      labelAngleDeg: 0,
    });
    setActiveSection("label");
  };

  const centerLoop = () => {
    updateDesign({
      loopXMm: design.bodyWidthMm / 2,
      // Стандартне вушко на 50%: центр на верхній грані → половина стирчить,
      // половина в тілі. Отвір видно як наскрізну дірку.
      loopYMm: 0,
      loopAngleDeg: 0,
    });
    setActiveSection("product");
  };

  const applyTextPreset = (preset: "s" | "m" | "l") => {
    // ЛИШЕ розмір+товщина штриха. Раніше пресет ще й перезаписував labelFontStyle
    // → користувач, що явно обрав шрифт, мовчки втрачав його при зміні розміру.
    // Шрифт тепер незалежний (окремий селектор «Шрифт для друку»).
    const presets = {
      s: { labelTextHeightMm: 3.4, labelStrokeMm: 0.65 },
      m: { labelTextHeightMm: 4.2, labelStrokeMm: 0.8 },
      l: { labelTextHeightMm: 5.2, labelStrokeMm: 1.0 },
    } satisfies Record<string, Partial<KeychainDesignerConfig>>;
    updateDesign(presets[preset]);
  };

  const placeLabel = (position: "bottom" | "top" | "left" | "right") => {
    const margin = Math.max(design.labelBandMm / 2 + 1.2, 4);
    const presets = {
      bottom: {
        labelXMm: design.bodyWidthMm / 2,
        labelYMm: Math.max(design.bodyHeightMm - design.labelBandMm / 2, margin),
        labelAngleDeg: 0,
      },
      top: {
        labelXMm: design.bodyWidthMm / 2,
        labelYMm: margin,
        labelAngleDeg: 180,
      },
      left: {
        labelXMm: margin,
        labelYMm: design.bodyHeightMm / 2,
        labelAngleDeg: 270,
      },
      right: {
        labelXMm: design.bodyWidthMm - margin,
        labelYMm: design.bodyHeightMm / 2,
        labelAngleDeg: 90,
      },
    } satisfies Record<string, Partial<KeychainDesignerConfig>>;
    updateDesign(presets[position]);
    setActiveSection("label");
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
    pollFailRef.current = 0;

    const interval = window.setInterval(async () => {
      if (pollingInFlightRef.current) return;
      pollingInFlightRef.current = true;
      try {
        const resp = await api.getStatus(taskGroupId);
        const task = resp as any;
        pollFailRef.current = 0;
        setTaskStatuses({ [task.task_id]: task });
        updateProgress(task.progress, task.message);
        if (task.status === "completed") {
          setGenerating(false);
          setDownloadUrl(task.download_url);
        } else if (task.status === "failed" || task.status === "cancelled") {
          // cancelled теж термінальний — інакше «Генерація N%» крутилась би вічно
          setGenerating(false);
          if (task.status === "failed") setError(task.message || t("error.generateFailed"));
        }
      } catch (pollError) {
        // Задача зникла (404 після рестарту бека) / мережа: після ~4 невдач підряд
        // розблоковуємо UI, щоб юзер міг повторити (як у SimpleControlPanel).
        pollFailRef.current += 1;
        console.error("[Keychain] status error", pollError);
        if (pollFailRef.current >= 4) {
          setGenerating(false);
          setError(t("error.generateFailed"));
        }
      } finally {
        pollingInFlightRef.current = false;
      }
    }, 3500);

    return () => {
      window.clearInterval(interval);
      pollingInFlightRef.current = false;
    };
  }, [t, taskGroupId, isGenerating, setGenerating, setTaskStatuses, setDownloadUrl, updateProgress]);

  const handleGenerate = async () => {
    if (!selectedArea) {
      setError(t("error.noArea"));
      return;
    }
    if (blockingPrintIssues.length > 0) {
      setError(t("error.notReady", { issues: blockingPrintIssues.map((issue) => issue.label.toLowerCase()).join(", ") }));
      return;
    }

    setError(null);
    setGenerating(true);
    setShowAllZones(false);
    // Ads/GA4: генерація = сильний сигнал наміру (ремаркетинг-аудиторія).
    import("@/lib/analytics").then((m) => m.trackConversion("generate", { props: { product: "keychain" } })).catch(() => {});

    try {
      // КРИТИЧНО: коли рамка повернута (cropPolygon, cropRotationDeg !== 0),
      // selectedArea — це bbox UNROTATED прямокутника. Реальна повернута зона
      // має кути ЗА межами цього малого bbox. Backend має качати OSM для
      // BBOX ПОВЕРНУТОЇ ЗОНИ (cropPolygon), інакше у мостах/будинках на
      // краях зони буде пустота.
      let fetchNorth = selectedArea.getNorth();
      let fetchSouth = selectedArea.getSouth();
      let fetchEast = selectedArea.getEast();
      let fetchWest = selectedArea.getWest();
      if (cropPolygon && cropPolygon.length >= 3) {
        let n = -Infinity, s = Infinity, e = -Infinity, w = Infinity;
        for (const [lon, lat] of cropPolygon) {
          if (lat > n) n = lat; if (lat < s) s = lat;
          if (lon > e) e = lon; if (lon < w) w = lon;
        }
        fetchNorth = Math.max(fetchNorth, n);
        fetchSouth = Math.min(fetchSouth, s);
        fetchEast = Math.max(fetchEast, e);
        fetchWest = Math.min(fetchWest, w);
      }
      // PADDING: розширюємо bbox на ~60м щоб дороги/будівлі на межі мали контекст
      // (інакше дороги «заокруглюються» на краях коли немає сусідніх сегментів).
      const cLatRad = ((fetchNorth + fetchSouth) / 2) * Math.PI / 180;
      const padLat = 60 / 111_320;
      const padLon = 60 / (111_320 * Math.max(Math.cos(cLatRad), 0.2));
      fetchNorth += padLat;
      fetchSouth -= padLat;
      fetchEast += padLon;
      fetchWest -= padLon;
      const response = await api.generateModel({
        north: fetchNorth,
        south: fetchSouth,
        east: fetchEast,
        west: fetchWest,
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
        keychain_label2: label2,
        keychain_back_label: backLabel,
        keychain_base_shape: design.baseShape,
        keychain_layout_rotation_deg: design.layoutRotationDeg,
        keychain_loop_style: design.loopStyle,
        keychain_loop_angle_deg: design.loopAngleDeg,
        keychain_body_width_mm: design.bodyWidthMm,
        keychain_body_height_mm: design.bodyHeightMm,
        keychain_map_x_mm: design.mapXMm,
        keychain_map_y_mm: design.mapYMm,
        keychain_map_width_mm: design.mapWidthMm,
        keychain_map_height_mm: design.mapHeightMm,
        // ОБИДВА кути сумуються: cropRotationDeg (поворот рамки на мапі) +
        // design.mapRotationDeg (поворот у дизайнері).
        // Backend застосує сумарний кут до полігона + ОБЕРТАЄ дані так само
        // як preview, щоб модель матчилась з тим що бачить юзер.
        keychain_map_rotation_deg: ((Number(cropRotationDeg || 0) + Number(design.mapRotationDeg || 0)) % 360 + 360) % 360,
        keychain_loop_center_x_mm: design.loopXMm,
        keychain_loop_center_y_mm: design.loopYMm,
        keychain_label_center_x_mm: design.labelXMm,
        keychain_label_center_y_mm: design.labelYMm,
        keychain_label_angle_deg: design.labelAngleDeg,
        keychain_loop_outer_radius_mm: design.loopOuterMm,
        keychain_loop_inner_radius_mm: design.loopInnerMm,
        keychain_corner_radius_mm: design.cornerRadiusMm,
        keychain_label_band_height_mm: design.labelBandMm,
        keychain_label_raise_mm: 0.8,
        keychain_label_text_height_mm: design.labelTextHeightMm,
        keychain_label_width_mm: design.labelWidthMm,
        keychain_label_stroke_mm: design.labelStrokeMm,
        keychain_label_font_style: design.labelFontStyle,
        keychain_rim_width_mm: design.rimWidthMm,
        keychain_rim_height_mm: design.rimHeightMm,
        // C3 ТОПО: рельєф висот замість карти (бек вимикає шари сам)
        keychain_topo_mode: topoMode,
        keychain_relief_mm: reliefMm,
        // D4 GPX-трек на брелку (бек малює підвищений шар поверх карти; з топо
        // несумісний — там рельєф). Точки беремо зі store.gpxFocus.
        ...(!topoMode && gpxTrack && gpxTrack.length >= 2 ? { gpx_track: gpxTrack } : {}),
        // Точний полігон обернутого rect — backend обрізає OSM по ньому,
        // а не по axis-aligned bbox. Так модель показує саме те що обрав юзер.
        ...(cropPolygon && cropPolygon.length >= 3 ? { zone_polygon_coords: cropPolygon } : {}),
        flat_water_layer_mm: waterLayerMm,
        flat_roads_layer_mm: roadLayerMm,
        flat_parks_layer_mm: parkLayerMm,
        flat_max_building_height_mm: buildingMaxMm,
        flat_uniform_building_height: uniformBuildingHeight,
        preview_mode: false,
        preview_include_base: true,
        preview_include_roads: true,
        preview_include_buildings: true,
        preview_include_water: true,
        preview_include_parks: true,
      });
      setTaskGroup(response.task_id, [response.task_id], "keychain");
      setActiveTaskId(response.task_id);
    } catch (generateError: any) {
      const apiDetail = generateError?.response?.data?.detail;
      const apiMessage = Array.isArray(apiDetail)
        ? apiDetail.map((item: any) => item?.msg || item?.message || JSON.stringify(item)).join("; ")
        : typeof apiDetail === "string"
          ? apiDetail
          : generateError.message;
      setError(apiMessage || t("error.generic"));
      setGenerating(false);
    }
  };

  const handleDownload = async () => {
    if (!downloadUrl) return;
    // Невелике превʼю того, що згенеровано — зберігається у кабінет (картка моделі).
    let preview = "";
    const svg = getKeychainDesignerSvg();
    if (svg) preview = (await svgToPngDataUrl(svg, { maxWidth: 360 })) || "";
    const res = await gatedDownload({
      taskId: taskGroupId || activeTaskId,
      downloadUrl: taskStatuses[activeTaskId || ""]?.download_url_3mf || downloadUrl,
      meta: { title: label, product_type: "keychain" },
      preview,
      getIdToken, openLogin,
      onLimit: () => window.dispatchEvent(new CustomEvent("monadruk:open-contact", {
        detail: { message: t("error.limitContact") },
      })),
    });
    if (res.status === "error") setError(t("error.downloadFailed"));
    if (res.status === "ok") {
      if (res.quota && typeof res.quota.remaining === "number") {
        setQuota((q) => ({ remaining: res.quota!.remaining as number, limit: q?.limit ?? 5, isAdmin: q?.isAdmin }));
      } else {
        refreshQuota();
      }
    }
  };

  const canGenerate = Boolean(selectedArea) && !isGenerating && blockingPrintIssues.length === 0;
  // ЗАМОВИТИ ОДРАЗУ: стартує генерацію у фоні + відкриває форму, щоб не чекати.
  const orderNow = () => {
    if (!downloadUrl && canGenerate) handleGenerate();
    setOrderOpen(true);
  };
  const currentStatus = isGenerating ? `${progress}% • ${status || t("status.generating")}` : downloadUrl ? t("status.ready3mf") : t("status.ready");
  const activeTaskStatus = activeTaskId ? taskStatuses[activeTaskId] : null;
  const generatedManifest = activeTaskStatus?.keychain_manifest;
  const generatedLayers = generatedManifest?.layers ?? null;
  const generatedLayerOrder = [
    ["base", t("layer.base")],
    ["rim", t("layer.rim")],
    ["water", t("layer.water")],
    ["parks", t("layer.parks")],
    ["roads", t("layer.roads")],
    ["buildings", t("layer.buildings")],
    ["text", t("layer.text")],
  ] as const;
  const generatedLayerChecks = generatedLayers
    ? generatedLayerOrder.map(([key, title]) => {
        const layer = generatedLayers[key];
        const present = Boolean(layer?.present);
        const zMax = typeof layer?.z_max_mm === "number" ? layer.z_max_mm : null;
        const size = Array.isArray(layer?.size_mm) ? layer.size_mm : [];
        const missingIsBad = key === "base" || key === "rim" || key === "roads" || (key === "text" && label.trim().length > 0);
        const tooThin = present && zMax !== null && key !== "base" && zMax < 0.24;
        const tone: PrintTone = !present
          ? missingIsBad
            ? "bad"
            : "warn"
          : tooThin
            ? "warn"
            : "good";
        return {
          key,
          title,
          tone,
          present,
          detail: present
            ? t("layer.top", { z: zMax !== null ? zMax.toFixed(2) : "-" }) + (size.length >= 2 ? t("layer.size", { w: size[0].toFixed(1), h: size[1].toFixed(1) }) : "")
            : key === "water"
              ? t("layer.noWater")
              : t("layer.missing"),
        };
      })
    : [];
  const generatedBad = generatedLayerChecks.filter((item) => item.tone === "bad");
  const generatedWarn = generatedLayerChecks.filter((item) => item.tone === "warn");
  const generatedVerdict = !generatedLayers
    ? null
    : generatedBad.length > 0
      ? {
          tone: "bad" as const,
          title: t("verdict.bad.title"),
          detail: t("verdict.bad.detail", { layers: generatedBad.map((item) => item.title.toLowerCase()).join(", ") }),
        }
      : generatedWarn.length > 0
        ? {
            tone: "warn" as const,
            title: t("verdict.warn.title"),
            detail: t("verdict.warn.detail", { layers: generatedWarn.map((item) => item.title.toLowerCase()).join(", ") }),
          }
        : {
            tone: "good" as const,
            title: t("verdict.good.title"),
            detail: t("verdict.good.detail"),
          };

  return (
    <div className="h-full overflow-y-auto px-3 py-3 sm:px-4">
      <div className="space-y-4 pb-4 lg:pb-8">
        <section className="rounded-[24px] border border-[var(--surface-border)] bg-[var(--surface-panel-strong)] p-4 shadow-[0_10px_28px_rgba(15,23,42,0.05)]">
          <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[var(--text-secondary)]">
            {t("header.eyebrow")}
          </p>
          <h2 className="mt-1 font-title text-lg font-semibold text-[var(--text-primary)]">
            {t("header.title")}
          </h2>
          <p className="mt-1 hidden text-sm leading-5 text-[var(--text-secondary)] sm:block">
            {t("header.subtitle")}
          </p>

          <div className={`mt-3 rounded-[22px] border px-4 py-3 ${
            readyTone === "good"
              ? "border-[rgba(11,92,87,0.22)] bg-[rgba(15,118,110,0.1)] text-[var(--accent-strong)]"
              : readyTone === "warn"
                ? "border-amber-200 bg-amber-50 text-amber-800"
                : "border-red-200 bg-red-50 text-red-700"
          }`}>
            <div className="flex items-start gap-3">
              {readyTone === "good" ? <CheckCircle2 className="mt-0.5 h-5 w-5 flex-none" /> : <AlertTriangle className="mt-0.5 h-5 w-5 flex-none" />}
              <div>
                <div className="text-sm font-bold">{readyLabel}</div>
                <div className="mt-1 text-xs leading-5 opacity-90">{nextAction}</div>
              </div>
            </div>
          </div>

          <details className="group mt-3 rounded-[18px] border border-[var(--surface-border)] bg-white/55">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-3 py-2.5 text-[13px] font-semibold text-[var(--text-primary)] [&::-webkit-details-marker]:hidden">
              {t("details.summary")}
              <span className="text-[var(--text-secondary)] transition group-open:rotate-180">▾</span>
            </summary>
            <div className="space-y-3 px-2 pb-3">
              <PrintabilityCard {...printability} />
              <div className="rounded-[18px] border border-[var(--surface-border)] bg-white/65 p-2">
                <div className="mb-2 px-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">
                  {t("details.printability")}
                </div>
                <div className="grid gap-2">
                  {printChecks.slice(0, 3).map((check) => (
                    <PrintCheckRow key={check.id} ok={check.ok} tone={check.tone} label={check.label} detail={check.detail} />
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <QuickActionButton onClick={resetToStandard}><RotateCcw size={15} />35 x 55</QuickActionButton>
                <QuickActionButton onClick={centerMap}><MapIcon size={15} />{t("quick.centerMap")}</QuickActionButton>
                <QuickActionButton onClick={centerLabel}><AlignCenter size={15} />{t("quick.centerLabel")}</QuickActionButton>
                <QuickActionButton onClick={centerLoop}><KeyRound size={15} />{t("quick.loopTop")}</QuickActionButton>
                <QuickActionButton onClick={repairForPrint}><CheckCircle2 size={15} />{t("quick.autoFix")}</QuickActionButton>
              </div>
            </div>
          </details>

          <button
            type="button"
            onClick={() => {
              setExpertMode((value) => !value);
              setActiveSection(expertMode ? "product" : "advanced");
            }}
            className="mt-3 inline-flex min-h-[44px] w-full items-center justify-center gap-2 rounded-[18px] border border-[var(--surface-border)] bg-white/85 px-3 py-2 text-sm font-semibold text-[var(--text-primary)] transition hover:bg-white"
          >
            <SlidersHorizontal size={16} />
            {expertMode ? t("advanced.hide") : t("advanced.show")}
          </button>
        </section>

        <nav className="-mx-1 rounded-[20px] border border-[var(--surface-border)] bg-[rgba(252,249,243,0.96)] p-1 shadow-[0_8px_22px_rgba(15,23,42,0.06)] backdrop-blur lg:sticky lg:top-0 lg:z-20">
          <div role="tablist" aria-label={t("tabsAria")} className="grid grid-cols-4 gap-1">
            {visibleSections.map((section, index) => (
              <button
                key={section.id}
                type="button"
                role="tab"
                id={`kc-tab-${section.id}`}
                aria-selected={activeSection === section.id}
                aria-controls={`kc-tabpanel-${section.id}`}
                onClick={() => setActiveSection(section.id)}
                className={`min-h-[42px] rounded-[16px] px-2 text-[11px] font-semibold transition sm:text-xs ${
                  activeSection === section.id
                    ? "bg-[var(--accent-strong)] text-white shadow-[0_10px_22px_rgba(11,92,87,0.22)]"
                    : "text-[var(--text-secondary)] hover:bg-white/80"
                }`}
              >
                <span className="hidden sm:inline">{section.id === "advanced" ? "" : `${index + 1}. `}</span>{t(section.labelKey)}
              </button>
            ))}
          </div>
        </nav>

        <section {...sectionProps("product")}>
          <SectionHeader
            icon={<KeyRound size={18} />}
            title={t("product.title")}
            description={t("product.desc")}
          />
          <div className="mt-4 grid grid-cols-2 gap-3">
            <Metric label={t("metric.base")} value={`${Math.round(design.bodyWidthMm)} x ${Math.round(design.bodyHeightMm)} ${t("unit.mm")}`} />
            <Metric label={t("metric.loop")} value={design.baseShape === "token" ? `${t("metric.hole")} Ø${(design.loopInnerMm * 2).toFixed(1)}` : design.loopStyle === "round" ? t("loop.round") : design.loopStyle === "slot" ? t("loop.slot") : design.loopStyle === "side-tab" ? t("loop.sideTab") : t("loop.teardrop")} />
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2">
            <ChoiceButton label="35 x 55" active={Math.round(design.bodyWidthMm) === 35 && Math.round(design.bodyHeightMm) === 55} onClick={resetToStandard} />
            <ChoiceButton label={t("product.maxMap")} active={design.mapWidthMm >= design.bodyWidthMm - 5} onClick={maximizeMapArea} />
            <ChoiceButton label={t("product.token")} active={design.baseShape === "token"} onClick={() => updateDesign({
              bodyWidthMm: 55,
              bodyHeightMm: 30,
              cornerRadiusMm: 15,
              baseShape: "token",
              loopStyle: "round",
              loopXMm: 27.5,
              loopYMm: 5.0,
              loopOuterMm: 2.8,
              loopInnerMm: 1.5,
              mapXMm: 0,
              mapYMm: 0,
              mapWidthMm: 55,
              mapHeightMm: 30,
              mapRotationDeg: 0,
              labelXMm: 32,
              labelYMm: 25.2,
              labelWidthMm: 34,
              labelBandMm: 6,
              labelTextHeightMm: 3.2,
              labelStrokeMm: 0.9,
              rimWidthMm: 0.9,
              rimHeightMm: 0.35,
            })} />
            <ChoiceButton label={t("product.centerLoop")} active={Math.abs(design.loopXMm - design.bodyWidthMm / 2) < 1 && Math.abs(design.loopYMm) < 2} onClick={centerLoop} />
            <ChoiceButton label={t("product.sideLoop")} active={design.loopXMm > design.bodyWidthMm} onClick={() => placeLoop("right")} />
          </div>
          <div className="mt-4">
            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("product.layoutRotation")}</div>
            <div className="grid grid-cols-4 gap-2">
              {[0, 90, 180, 270].map((angle) => (
                <ChoiceButton
                  key={`layout-${angle}`}
                  label={`${angle}°`}
                  active={(design.layoutRotationDeg || 0) === angle}
                  onClick={() => updateDesign({ layoutRotationDeg: angle })}
                />
              ))}
            </div>
          </div>
        </section>

        <section {...sectionProps("map")}>
          <SectionHeader
            icon={<MapIcon size={18} />}
            title={t("map.title")}
            description={t("map.desc")}
          />
          <div className="mt-4 space-y-3">
            <PrintabilityCard {...printability} />
            {/* C3 ТОПО-БРЕЛОК: рельєф висот замість карти */}
            <label className="flex min-h-[52px] items-center gap-3 rounded-[18px] border border-[var(--surface-border)] bg-white/80 px-3 py-2 text-sm font-semibold text-[var(--text-primary)]">
              <input
                type="checkbox"
                checked={topoMode}
                onChange={(event) => setTopoMode(event.target.checked)}
                className="h-5 w-5 accent-[var(--accent-strong)]"
              />
              <span>
                {t("topo.label")}
                <span className="block text-[11px] font-normal leading-4 text-[var(--text-secondary)]">
                  {t("topo.hint")}
                </span>
              </span>
            </label>
            {topoMode && (
              <SliderField
                label={t("topo.reliefHeight")}
                valueLabel={`${reliefMm.toFixed(1)} ${t("unit.mm")}`}
                min={0.6}
                max={4.0}
                step={0.1}
                value={reliefMm}
                onChange={setReliefMm}
              />
            )}
            {/* D4 GPX-трек на брелку (маршрут пробіжки/походу як рельєфна лінія).
                Несумісний з топо — там рельєф замість карти. */}
            {!topoMode && (
              <div className="rounded-[18px] border border-[var(--surface-border)] bg-white/80 px-3 py-2.5">
                <label className="flex cursor-pointer items-center justify-between gap-3 text-sm font-semibold text-[var(--text-primary)]">
                  <span>{t("gpx.title")}</span>
                  <input
                    type="file"
                    accept=".gpx,application/gpx+xml"
                    data-testid="kc-gpx-input"
                    className="hidden"
                    onChange={async (e) => {
                      const file = e.target.files?.[0];
                      e.target.value = "";
                      if (!file) return;
                      try {
                        const { parseGpx, gpxBounds } = await import("@/lib/gpx");
                        const parsed = parseGpx(await file.text());
                        if (!parsed) { setGpxName(null); setGpxFocus(null); setError(t("gpx.readError")); return; }
                        setError(null);
                        setGpxName(parsed.name || file.name.replace(/\.gpx$/i, ""));
                        const bb = gpxBounds(parsed.points);
                        if (bb) {
                          const [w, s_, e_, n] = bb;
                          setGpxFocus({ west: w, south: s_, east: e_, north: n, points: parsed.points });
                        }
                      } catch { setError(t("gpx.readError")); }
                    }}
                  />
                  <span className="rounded-full border border-[var(--surface-border)] bg-white px-3 py-1 text-[12px] font-semibold text-[var(--accent-strong)]">
                    {gpxTrack ? t("gpx.replace") : t("gpx.pickFile")}
                  </span>
                </label>
                {gpxTrack ? (
                  <div className="mt-2 space-y-1">
                    <div className="flex items-center justify-between gap-2 text-[12px] text-[var(--text-secondary)]">
                      <span className="truncate">{t("gpx.loaded", { name: gpxName ?? "", count: gpxTrack.length })}</span>
                      <button type="button" onClick={() => { setGpxName(null); setGpxFocus(null); }} className="shrink-0 font-semibold text-red-700 hover:underline">{t("gpx.remove")}</button>
                    </div>
                    <p className="text-[11px] leading-4 text-[var(--text-secondary)]">{t("gpx.privacy")}</p>
                  </div>
                ) : (
                  <p className="mt-1 text-[11px] leading-4 text-[var(--text-secondary)]">{t("gpx.hint")}</p>
                )}
              </div>
            )}
            <div className="grid grid-cols-2 gap-2">
              <QuickActionButton onClick={centerMap}>
                <MapIcon size={15} />
                {t("quick.centerMap")}
              </QuickActionButton>
              <QuickActionButton onClick={maximizeMapArea}>
                <AlignCenter size={15} />
                {t("quick.maxMap")}
              </QuickActionButton>
            </div>
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("map.orientation")}</div>
              <div className="grid grid-cols-3 gap-2">
                {[0, 90, 180].map((angle) => (
                  <ChoiceButton
                    key={`map-orientation-${angle}`}
                    label={angle === 0 ? "0°" : angle === 90 ? "90°" : "180°"}
                    active={(design.mapRotationDeg || 0) === angle}
                    onClick={() => updateDesign({ mapRotationDeg: angle })}
                  />
                ))}
              </div>
              <div className="mt-2 grid grid-cols-2 gap-2">
                <QuickActionButton onClick={() => updateDesign({ mapRotationDeg: ((design.mapRotationDeg || 0) - 15 + 360) % 360 })}>
                  ↺ 15°
                </QuickActionButton>
                <QuickActionButton onClick={() => updateDesign({ mapRotationDeg: ((design.mapRotationDeg || 0) + 15) % 360 })}>
                  ↻ 15°
                </QuickActionButton>
              </div>
            </div>
            <Metric
              label={t("metric.detail")}
              value={
                printScale
                  ? printScale.tooLarge
                    ? t("detail.bad")
                    : printScale.minPrintableWorldM > 2.8
                      ? t("detail.edge")
                      : t("detail.good")
                  : t("detail.pickCrop")
              }
            />
          </div>
        </section>

        <section {...sectionProps("label")}>
          <SectionHeader
            icon={<Type size={18} />}
            title={t("label.title")}
            description={t("label.desc")}
          />
          <input
            value={label}
            onChange={(event) => onLabelChange(event.target.value.toUpperCase().slice(0, 28))}
            placeholder="KYIV"
            className="mt-4 w-full rounded-[20px] border border-[var(--surface-border)] bg-white/90 px-4 py-3 text-sm font-semibold uppercase tracking-[0.08em] text-[var(--text-primary)] outline-none transition focus:border-[var(--accent)]"
          />
          <div className="mt-4 grid grid-cols-3 gap-2">
            <ChoiceButton label="S" active={design.labelTextHeightMm < 3.8} onClick={() => applyTextPreset("s")} />
            <ChoiceButton label="M" active={design.labelTextHeightMm >= 3.8 && design.labelTextHeightMm < 4.8} onClick={() => applyTextPreset("m")} />
            <ChoiceButton label="L" active={design.labelTextHeightMm >= 4.8} onClick={() => applyTextPreset("l")} />
          </div>
          <div className="mt-4">
            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("label.printFont")}</div>
            <div className="grid grid-cols-3 gap-2">
              {([
                ["block", t("font.block")],
                ["wide", t("font.wide")],
                ["condensed", t("font.narrow")],
              ] as Array<[KeychainLabelFontStyle, string]>).map(([font, text]) => (
                <ChoiceButton key={font} label={text} active={design.labelFontStyle === font} onClick={() => updateDesign({ labelFontStyle: font })} />
              ))}
            </div>
          </div>
          <div className="mt-4">
            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("label.placement")}</div>
            <div className="grid grid-cols-2 gap-2">
              <ChoiceButton label={t("place.bottom")} active={design.labelYMm > design.bodyHeightMm * 0.68 && design.labelAngleDeg === 0} onClick={() => placeLabel("bottom")} />
              <ChoiceButton label={t("place.top")} active={design.labelYMm < design.bodyHeightMm * 0.32} onClick={() => placeLabel("top")} />
              <ChoiceButton label={t("place.left")} active={design.labelXMm < design.bodyWidthMm * 0.32} onClick={() => placeLabel("left")} />
              <ChoiceButton label={t("place.right")} active={design.labelXMm > design.bodyWidthMm * 0.68} onClick={() => placeLabel("right")} />
            </div>
          </div>
          <div className="mt-3 rounded-[18px] border border-[var(--surface-border)] bg-white/80 px-3 py-2 text-xs leading-5 text-[var(--text-secondary)]">
            {t("label.current", { height: design.labelTextHeightMm.toFixed(1), stroke: design.labelStrokeMm.toFixed(2), style: design.labelFontStyle })}
          </div>
          <div className="mt-4 space-y-3">
            <SliderField label={t("slider.labelWidth")} valueLabel={`${design.labelWidthMm.toFixed(0)} ${t("unit.mm")}`} min={6} max={design.bodyWidthMm} step={1} value={design.labelWidthMm} onChange={(value) => updateDesign({ labelWidthMm: value })} />
            <SliderField label={t("slider.textHeight")} valueLabel={`${design.labelTextHeightMm.toFixed(1)} ${t("unit.mm")}`} min={1.6} max={8.5} step={0.1} value={design.labelTextHeightMm} onChange={(value) => updateDesign({ labelTextHeightMm: value })} />
            <SliderField label={t("slider.strokeWidth")} valueLabel={`${design.labelStrokeMm.toFixed(2)} ${t("unit.mm")}`} min={0.4} max={2.0} step={0.05} value={design.labelStrokeMm} onChange={(value) => updateDesign({ labelStrokeMm: value })} />
          </div>

          <div className="mt-5">
            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("label.secondLine")}</div>
            <div className="flex gap-2">
              <input
                value={label2}
                onChange={(event) => setLabel2(event.target.value.toUpperCase().slice(0, 28))}
                placeholder="12.06.2026"
                className="min-w-0 flex-1 rounded-[20px] border border-[var(--surface-border)] bg-white/90 px-4 py-3 text-sm font-semibold uppercase tracking-[0.08em] text-[var(--text-primary)] outline-none transition focus:border-[var(--accent)]"
              />
              <button
                type="button"
                title={t("label.insertCoords")}
                disabled={!selectedArea}
                onClick={() => {
                  if (!selectedArea) return;
                  const c = selectedArea.getCenter();
                  setLabel2(`${c.lat.toFixed(4)}, ${c.lng.toFixed(4)}`);
                }}
                className="shrink-0 rounded-[20px] border border-[var(--surface-border)] bg-white/90 px-3 text-lg transition hover:border-[var(--accent)] disabled:cursor-not-allowed disabled:opacity-40"
              >
                📍
              </button>
            </div>
            <p className="mt-1.5 text-[11px] leading-4 text-[var(--text-secondary)]">{t("label.secondHint")}</p>
          </div>

          <div className="mt-4">
            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("label.backTitle")}</div>
            <input
              value={backLabel}
              onChange={(event) => setBackLabel(event.target.value.toUpperCase().slice(0, 28))}
              placeholder={t("label.backPlaceholder")}
              className="w-full rounded-[20px] border border-[var(--surface-border)] bg-white/90 px-4 py-3 text-sm font-semibold uppercase tracking-[0.08em] text-[var(--text-primary)] outline-none transition focus:border-[var(--accent)]"
            />
            <p className="mt-1.5 text-[11px] leading-4 text-[var(--text-secondary)]">{t("label.backHint")}</p>
          </div>
        </section>

        <section {...sectionProps("review")}>
          <SectionHeader
            icon={<CheckCircle2 size={18} />}
            title={t("review.title")}
            description={t("review.desc")}
          />
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <Metric label={t("metric.status")} value={currentStatus} />
            <Metric label={t("metric.product")} value={`${Math.round(design.bodyWidthMm)} x ${Math.round(design.bodyHeightMm)} ${t("unit.mm")}`} />
            <Metric label={t("metric.map")} value={`${Math.round(design.mapWidthMm)} x ${Math.round(design.mapHeightMm)} ${t("unit.mm")}`} />
            <Metric label={t("metric.signature")} value={label || t("review.noText")} />
          </div>
          <div className="mt-4">
            <PrintabilityCard {...printability} />
          </div>
          <div className="mt-4 grid gap-2">
            {printChecks.map((check) => (
              <PrintCheckRow key={`review-${check.id}`} ok={check.ok} tone={check.tone} label={check.label} detail={check.detail} />
            ))}
          </div>
          {generatedManifest && generatedLayers ? (
            <div className="mt-4 rounded-[22px] border border-[var(--surface-border)] bg-white/82 p-3">
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">
                {t("review.generated3mf")}
              </div>
              {generatedVerdict ? (
                <div className="mt-2">
                  <PrintabilityCard {...generatedVerdict} />
                </div>
              ) : null}
              <div className="mt-2 grid gap-2">
                {generatedLayerChecks.map((layer) => {
                  return (
                    <div key={layer.key} className="flex items-center justify-between gap-3 rounded-[14px] border border-[var(--surface-border)] bg-white/75 px-3 py-2 text-xs">
                      <span className="font-semibold text-[var(--text-primary)]">{layer.title}</span>
                      <span className={layer.tone === "good" ? "text-[var(--accent-strong)]" : layer.tone === "warn" ? "text-amber-700" : "text-red-700"}>
                        {layer.tone === "good" ? t("layerTone.ok") : layer.tone === "warn" ? t("layerTone.warn") : t("layerTone.bad")} · {layer.detail}
                      </span>
                    </div>
                  );
                })}
              </div>
              <div className="mt-2 text-xs leading-5 text-[var(--text-secondary)]">
                {t("review.layersNote")}
              </div>
            </div>
          ) : null}
          <div className="mt-4 grid gap-3">
            <button
              type="button"
              onClick={handleGenerate}
              disabled={!canGenerate}
              className="inline-flex min-h-[48px] items-center justify-center gap-2 rounded-[22px] bg-[var(--accent-strong)] px-4 py-3 text-sm font-semibold text-white shadow-[0_14px_30px_rgba(11,92,87,0.24)] transition disabled:cursor-not-allowed disabled:opacity-45"
            >
              {isGenerating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {isGenerating ? t("btn.generating") : t("btn.create3mf")}
            </button>
            {/* «Замовити» — одразу після «Створити», на видному місці (бронзова) */}
            <button
              type="button"
              onClick={() => setOrderOpen(true)}
              className="inline-flex min-h-[52px] items-center justify-center gap-2 rounded-[22px] bg-[var(--bronze,#8E6B3D)] px-4 py-3.5 text-[15px] font-extrabold text-white shadow-[0_16px_34px_rgba(142,107,61,0.32)] transition hover:opacity-90"
            >
              <ShoppingBag className="h-5 w-5" />
              {downloadUrl ? t("btn.buyOrder") : t("btn.order")}{quote ? ` · ${quote.formatted}` : ""}
            </button>
            <button
              type="button"
              onClick={handleDownload}
              disabled={!downloadUrl || !activeTaskId}
              className="inline-flex min-h-[48px] items-center justify-center gap-2 rounded-[22px] border border-[var(--surface-border)] bg-white/85 px-4 py-3 text-sm font-semibold text-[var(--text-primary)] transition disabled:cursor-not-allowed disabled:opacity-45"
            >
              <Download className="h-4 w-4" />
              {t("btn.download3mf")}
            </button>
            {downloadUrl && quota && !quota.isAdmin && (
              <div className={`-mt-1 text-center text-[12px] font-medium ${quota.remaining > 0 ? "text-[var(--text-secondary)]" : "text-amber-700"}`}>
                {quota.remaining > 0
                  ? t("quota.remaining", { remaining: quota.remaining, limit: quota.limit })
                  : t("quota.exhausted")}
              </div>
            )}
          </div>
        </section>

        <OrderDialog
          open={orderOpen}
          onClose={() => setOrderOpen(false)}
          taskId={taskGroupId}
          productType="keychain"
          priceText={quote?.formatted}
          modelPending={!downloadUrl}
          summary={{
            label,
            size: `${Math.round(design.bodyWidthMm)}×${Math.round(design.bodyHeightMm)} ${t("unit.mm")}`,
          }}
        />

        {/* Мобільний sticky-бар: ціна завжди видима + головна дія стану */}
        <div className="h-20 lg:hidden" aria-hidden="true" />
        <StickyActionBar
          // Ціну показуємо ЛИШЕ на кроці оформлення (OrderDialog), не тут.
          priceLabel={t("sticky.priceLabel")}
          price={null}
          actionLabel={downloadUrl ? t("btn.order") : isGenerating ? t("sticky.generating", { progress }) : t("sticky.createKeychain")}
          busy={isGenerating}
          // НЕ блокуємо коли зона не вибрана / є print-issue — інакше на мобільному
          // єдина видима кнопка стає глухим disabled без пояснення (dead-end).
          // Замість цього тап дає тост із причиною + перемикає на потрібну секцію.
          disabled={isGenerating}
          onAction={() => {
            if (downloadUrl) { setOrderOpen(true); return; }
            if (!selectedArea) {
              setError(t("error.noArea"));
              window.dispatchEvent(new CustomEvent("monadruk:toast", { detail: { type: "warn", message: t("error.noArea") } }));
              setActiveSection("map");
              return;
            }
            if (blockingPrintIssues.length > 0) {
              const issue = blockingPrintIssues[0];
              const msg = t("error.notReady", { issues: blockingPrintIssues.map((i) => i.label.toLowerCase()).join(", ") });
              setError(msg);
              window.dispatchEvent(new CustomEvent("monadruk:toast", { detail: { type: "warn", message: msg } }));
              setActiveSection(issue.id === "crop" ? "map" : issue.id === "text-stroke" ? "label" : "product");
              return;
            }
            handleGenerate();
          }}
          // Готово → «Завантажити»; до/під час → «Замовити» (order-now)
          secondaryLabel={downloadUrl ? t("sticky.download") : (selectedArea ? t("btn.order") : undefined)}
          onSecondary={downloadUrl ? handleDownload : (selectedArea ? orderNow : undefined)}
        />

        <section {...sectionProps("advanced")}>
          <SectionHeader
            icon={<SlidersHorizontal size={18} />}
            title={t("advanced.title")}
            description={t("advanced.desc")}
          />

          <div className="mt-4 space-y-3">
            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("advanced.baseLoop")}</div>
            <SliderField label={t("slider.baseWidth")} valueLabel={`${design.bodyWidthMm.toFixed(0)} ${t("unit.mm")}`} min={35} max={140} step={1} value={design.bodyWidthMm} onChange={(value) => updateDesign({ bodyWidthMm: value })} />
            <SliderField label={t("slider.baseHeight")} valueLabel={`${design.bodyHeightMm.toFixed(0)} ${t("unit.mm")}`} min={26} max={96} step={1} value={design.bodyHeightMm} onChange={(value) => updateDesign({ bodyHeightMm: value })} />
            <SliderField label={t("product.layoutRotation")} valueLabel={`${design.layoutRotationDeg.toFixed(0)}°`} min={0} max={270} step={90} value={design.layoutRotationDeg} onChange={(value) => updateDesign({ layoutRotationDeg: value })} />
            <SliderField label={t("slider.baseThickness")} valueLabel={`${baseThicknessMm.toFixed(1)} ${t("unit.mm")}`} min={1.0} max={4.0} step={0.1} value={baseThicknessMm} onChange={setBaseThicknessMm} />
            <SliderField label={design.baseShape === "token" ? t("slider.holeGuideRadius") : t("slider.loopOuterRadius")} valueLabel={`${design.loopOuterMm.toFixed(1)} ${t("unit.mm")}`} min={design.baseShape === "token" ? 2.4 : 4.5} max={design.baseShape === "token" ? 6 : 11} step={0.1} value={design.loopOuterMm} onChange={(value) => updateDesign({ loopOuterMm: value, loopInnerMm: Math.min(design.loopInnerMm, value - 1.4) })} />
            <SliderField
              label={t("slider.ringHole")}
              valueLabel={design.baseShape === "token" ? `Ø${(design.loopInnerMm * 2).toFixed(1)} ${t("unit.mm")}` : `${design.loopInnerMm.toFixed(1)} ${t("unit.mm")}`}
              min={design.baseShape === "token" ? 1.5 : 2.0}
              max={design.baseShape === "token" ? 3.5 : 6.5}
              step={0.1}
              value={design.loopInnerMm}
              onChange={(value) => updateDesign({ loopInnerMm: design.baseShape === "token" ? Math.min(value, design.loopOuterMm - 0.8) : Math.min(value, design.loopOuterMm - 1.4) })}
            />
            <SliderField label={t("slider.cornerRadius")} valueLabel={`${design.cornerRadiusMm.toFixed(1)} ${t("unit.mm")}`} min={0} max={9} step={0.1} value={design.cornerRadiusMm} onChange={(value) => updateDesign({ cornerRadiusMm: value })} />
            <SliderField label={t("slider.labelBand")} valueLabel={`${design.labelBandMm.toFixed(1)} ${t("unit.mm")}`} min={3} max={18} step={0.5} value={design.labelBandMm} onChange={(value) => updateDesign({ labelBandMm: value })} />
            <SliderField label={t("slider.rimWidth")} valueLabel={`${design.rimWidthMm.toFixed(1)} ${t("unit.mm")}`} min={0} max={5} step={0.1} value={design.rimWidthMm} onChange={(value) => updateDesign({ rimWidthMm: value })} />
            <SliderField label={t("slider.rimHeight")} valueLabel={`${design.rimHeightMm.toFixed(2)} ${t("unit.mm")}`} min={0} max={1.6} step={0.05} value={design.rimHeightMm} onChange={(value) => updateDesign({ rimHeightMm: value })} />
          </div>

          <div className="mt-5 space-y-4">
            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("advanced.shapeLoop")}</div>
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("advanced.baseShape")}</div>
              <div className="grid grid-cols-2 gap-2">
                {([
                  ["rounded", t("shape.rounded")],
                  ["token", t("shape.token")],
                  ["capsule", t("shape.capsule")],
                  ["tag", t("shape.tag")],
                  ["octagon", t("shape.octagon")],
                  ["heart", t("shape.heart")],
                  ["house", t("shape.house")],
                  ["puzzle-l", t("shape.puzzleL")],
                  ["puzzle-r", t("shape.puzzleR")],
                  ["heart-l", t("shape.heartL")],
                  ["heart-r", t("shape.heartR")],
                ] as Array<[KeychainBaseShape, string]>).map(([shape, text]) => (
                  <ChoiceButton key={shape} label={text} active={design.baseShape === shape} onClick={() => updateDesign({ baseShape: shape })} />
                ))}
              </div>
            </div>
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("advanced.loopType")}</div>
              <div className="grid grid-cols-2 gap-2">
                {([
                  ["round", t("loop.round")],
                  ["teardrop", t("loop.teardrop")],
                  ["slot", t("loop.slot")],
                  ["side-tab", t("loop.sideTab")],
                ] as Array<[KeychainLoopStyle, string]>).map(([style, text]) => (
                  <ChoiceButton key={style} label={text} active={design.loopStyle === style} onClick={() => updateDesign({ loopStyle: style })} />
                ))}
              </div>
            </div>
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("advanced.loopPosition")}</div>
              <div className="grid grid-cols-2 gap-2">
                <ChoiceButton label={t("loopPos.topLeft")} active={design.loopXMm < design.bodyWidthMm / 2 && design.loopYMm < 0} onClick={() => placeLoop("top-left")} />
                <ChoiceButton label={t("loopPos.topRight")} active={design.loopXMm > design.bodyWidthMm / 2 && design.loopYMm < 0} onClick={() => placeLoop("top-right")} />
                <ChoiceButton label={t("loopPos.right")} active={design.loopXMm > design.bodyWidthMm} onClick={() => placeLoop("right")} />
                <ChoiceButton label={t("loopPos.bottomLeft")} active={design.loopYMm > design.bodyHeightMm} onClick={() => placeLoop("bottom-left")} />
              </div>
            </div>
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("advanced.rotation")}</div>
              <div className="grid grid-cols-4 gap-2">
                {[0, 90, 180, 270].map((angle) => (
                  <ChoiceButton key={`loop-${angle}`} label={`${angle}°`} active={design.loopAngleDeg === angle} onClick={() => updateDesign({ loopAngleDeg: angle })} />
                ))}
              </div>
            </div>
          </div>

          <div className="mt-5 space-y-3">
            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("advanced.mapText")}</div>
            <SliderField label={t("slider.mapWidth")} valueLabel={`${design.mapWidthMm.toFixed(0)} ${t("unit.mm")}`} min={Math.min(28, design.bodyWidthMm)} max={design.bodyWidthMm} step={1} value={design.mapWidthMm} onChange={(value) => updateDesign({ mapWidthMm: value })} />
            <SliderField label={t("slider.mapHeight")} valueLabel={`${design.mapHeightMm.toFixed(0)} ${t("unit.mm")}`} min={Math.min(18, design.bodyHeightMm)} max={design.bodyHeightMm} step={1} value={design.mapHeightMm} onChange={(value) => updateDesign({ mapHeightMm: value })} />
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("advanced.mapOrientation")}</div>
              <div className="grid grid-cols-4 gap-2">
                {[0, 90, 180, 270].map((angle) => (
                  <ChoiceButton key={`map-${angle}`} label={`${angle}°`} active={(design.mapRotationDeg || 0) === angle} onClick={() => updateDesign({ mapRotationDeg: angle })} />
                ))}
              </div>
            </div>
            <SliderField label={t("slider.labelWidth")} valueLabel={`${design.labelWidthMm.toFixed(0)} ${t("unit.mm")}`} min={6} max={design.bodyWidthMm} step={1} value={design.labelWidthMm} onChange={(value) => updateDesign({ labelWidthMm: value })} />
            <SliderField label={t("slider.textHeight")} valueLabel={`${design.labelTextHeightMm.toFixed(1)} ${t("unit.mm")}`} min={1.6} max={8.5} step={0.1} value={design.labelTextHeightMm} onChange={(value) => updateDesign({ labelTextHeightMm: value })} />
            <SliderField label={t("slider.strokeWidth")} valueLabel={`${design.labelStrokeMm.toFixed(2)} ${t("unit.mm")}`} min={0.4} max={2.0} step={0.05} value={design.labelStrokeMm} onChange={(value) => updateDesign({ labelStrokeMm: value })} />
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("advanced.printFont")}</div>
              <div className="grid grid-cols-3 gap-2">
                {([
                  ["block", t("font.block")],
                  ["wide", t("font.wide")],
                  ["condensed", t("font.narrow")],
                ] as Array<[KeychainLabelFontStyle, string]>).map(([font, text]) => (
                  <ChoiceButton key={`advanced-font-${font}`} label={text} active={design.labelFontStyle === font} onClick={() => updateDesign({ labelFontStyle: font })} />
                ))}
              </div>
            </div>
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("advanced.textRotation")}</div>
              <div className="grid grid-cols-4 gap-2">
                {[0, 90, 180, 270].map((angle) => (
                  <ChoiceButton key={`label-${angle}`} label={`${angle}°`} active={design.labelAngleDeg === angle} onClick={() => updateDesign({ labelAngleDeg: angle })} />
                ))}
              </div>
            </div>
          </div>

          <div className="mt-5 space-y-3">
            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("advanced.mapLayers")}</div>
            <SliderField label={t("layer.roads")} valueLabel={`${roadLayerMm.toFixed(2)} ${t("unit.mm")}`} min={0.4} max={0.9} step={0.01} value={roadLayerMm} onChange={setRoadLayerMm} />
            <SliderField label={t("layer.parks")} valueLabel={`${parkLayerMm.toFixed(2)} ${t("unit.mm")}`} min={0.18} max={0.75} step={0.01} value={parkLayerMm} onChange={setParkLayerMm} />
            <SliderField label={t("layer.water")} valueLabel={`${waterLayerMm.toFixed(2)} ${t("unit.mm")}`} min={0.24} max={0.55} step={0.01} value={waterLayerMm} onChange={setWaterLayerMm} />
            <SliderField label={t("slider.maxBuildings")} valueLabel={`${buildingMaxMm.toFixed(1)} ${t("unit.mm")}`} min={0.8} max={5.0} step={0.1} value={buildingMaxMm} onChange={setBuildingMaxMm} />
            <label className="flex min-h-[52px] items-center gap-3 rounded-[18px] border border-[var(--surface-border)] bg-white/80 px-3 py-2 text-sm font-semibold text-[var(--text-primary)]">
              <input
                type="checkbox"
                checked={uniformBuildingHeight}
                onChange={(event) => setUniformBuildingHeight(event.target.checked)}
                className="h-5 w-5 accent-[var(--accent-strong)]"
              />
              {t("advanced.uniformBuildings")}
            </label>
          </div>
        </section>

        {error && (
          <div className="rounded-[20px] border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            <p>{error}</p>
            {selectedArea && !isGenerating && (
              <button
                type="button"
                onClick={() => handleGenerate()}
                className="mt-2 inline-flex items-center gap-1 text-[13px] font-semibold text-red-800 underline-offset-2 hover:underline"
              >
                ↻ {t("error.retry")}
              </button>
            )}
          </div>
        )}

        <div className="hidden gap-3 lg:grid">
          <button
            type="button"
            onClick={handleGenerate}
            disabled={!canGenerate}
            className="inline-flex min-h-[48px] items-center justify-center gap-2 rounded-[22px] bg-[var(--accent-strong)] px-4 py-3 text-sm font-semibold text-white shadow-[0_14px_30px_rgba(11,92,87,0.24)] transition disabled:cursor-not-allowed disabled:opacity-45"
          >
            {isGenerating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {isGenerating ? t("btn.generating") : t("sticky.createKeychain")}
          </button>
          <button
            type="button"
            onClick={handleDownload}
            disabled={!downloadUrl || !activeTaskId}
            className="inline-flex min-h-[48px] items-center justify-center gap-2 rounded-[22px] border border-[var(--surface-border)] bg-white/85 px-4 py-3 text-sm font-semibold text-[var(--text-primary)] transition disabled:cursor-not-allowed disabled:opacity-45"
          >
            <Download className="h-4 w-4" />
            {t("btn.download3mf")}
          </button>
          <button
            type="button"
            onClick={() => setOrderOpen(true)}
            className="inline-flex min-h-[48px] items-center justify-center gap-2 rounded-[22px] bg-[var(--bronze,#8E6B3D)] px-4 py-3 text-sm font-bold text-white shadow-[0_14px_30px_rgba(142,107,61,0.28)] transition hover:opacity-90"
          >
            <ShoppingBag className="h-4 w-4" />
            {downloadUrl ? t("btn.buyOrder") : t("btn.order")}
          </button>
        </div>

        {/* Мобільний другий нижній бар (sticky bottom-3) ПРИБРАНО: він дублював
            StickyActionBar (портал, fixed bottom-0) — обидва внизу на мобільному
            давали два конкурентні бари. StickyActionBar уже має Створити/
            Завантажити/Замовити; статус+розмір видно у превʼю та степері. */}
      </div>
    </div>
  );
}
