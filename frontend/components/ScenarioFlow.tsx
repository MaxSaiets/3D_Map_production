"use client";

import React, { useEffect, useRef, useState } from "react";
import { getVariant } from "@/lib/ab";
import { GuidedStickyBar } from "@/components/GuidedStickyBar";
import { useDownloadQuota } from "@/lib/useDownloadQuota";
import { useTranslations, useLocale } from "next-intl";
import { ArrowLeft, Check, Download, Home, Loader2, MapPin, PenLine, Share2, ShoppingBag, X, ShieldCheck } from "lucide-react";
import { MapSearchBox } from "@/components/MapSearchBox";
// ЛОКАЛІЗОВАНИЙ Link (@/i18n/navigation), НЕ next/link — інакше лінк на
// /keychains з /en/create губив би префікс локалі.
import { Link } from "@/i18n/navigation";
import { Button } from "@/components/ui/Button";
import { ShareQr } from "@/components/ShareQr";
import { useShallow } from "zustand/react/shallow";
import { useGenerationStore } from "@/store/generation-store";
import { SIMPLE_SIZES } from "@/lib/generation";
import { fetchQuote, type Quote } from "@/lib/pricing";
import { CITIES, MAP_TEMPLATES } from "@/lib/templates";
import { WORLD_CITIES } from "@/lib/worldCities";
import { CITY_PAGES } from "@/lib/cityPages";
import {
  KEYCHAIN_PRICE_UAH,
  MAP_MAGNET_PRICE_UAH,
  MAP_RELIEF_ADDON_UAH,
  mapPriceEur,
  mapPriceForSizeUah,
} from "@/lib/mapPrices";
import { GenerationStages } from "@/components/GenerationStages";
import { SalesAlternatives } from "@/components/SalesAlternatives";

/** Зона ПІД РОЗМІР плитки: ~7.5 м/мм — «добра деталізація» і гарантовано в
 *  безпечних межах (isSafe = ≤10 м/мм у MapSelector). Фіксована 800×800
 *  для S (55 мм) давала 14.5 м/мм → червоне «Ділянка завелика. Зменши рамку»
 *  прямо в guided-автозоні. Тепер зона їде за розміром: S≈410, M≈600, L≈825. */
const zoneForSizeM = (sizeMm: number) => Math.round(sizeMm * 7);

/** Швидкі міста: 1 тап замість друкування адреси. Та сама подія, що й пошук. */
const QUICK_CITIES: Array<{ uk: string; en: string; lat: number; lon: number }> = [
  { uk: "Київ", en: "Kyiv", lat: 50.4501, lon: 30.5234 },
  { uk: "Львів", en: "Lviv", lat: 49.8419, lon: 24.0315 },
  { uk: "Одеса", en: "Odesa", lat: 46.4825, lon: 30.7233 },
  { uk: "Харків", en: "Kharkiv", lat: 49.9935, lon: 36.2304 },
];

/** ФОРМА мапи в guided (24.09.2026, власник: «не повністю все можна вибрати»).
 *  id = baseShape рамки MapSelector (та сама, що в повному конструкторі) — рамка
 *  малює контур і шле його полігоном у zone_polygon_coords; бекенд ріже по ньому
 *  (перевірено локальною генерацією: коло/шестикутник/серце). Магніт — лише квадрат. */
const GUIDED_SHAPES = [
  { id: "rounded", key: "shapeSquare", path: "M3 3h18v18H3z" },
  { id: "circle", key: "shapeCircle", path: "M12 2.5a9.5 9.5 0 1 0 0 19a9.5 9.5 0 1 0 0-19z" },
  { id: "hexagon", key: "shapeHexagon", path: "M12 2l8.66 5v10L12 22l-8.66-5V7z" },
  { id: "heart", key: "shapeHeart", path: "M12 21s-8.5-5.3-8.5-11.2A4.8 4.8 0 0 1 12 6.6a4.8 4.8 0 0 1 8.5 3.2C20.5 15.7 12 21 12 21z" },
] as const;

/** Guided-розмір: будь-яке ребро 50–200 мм (стіл P1 = 256 мм; більше — у розширеному режимі). */
const SIZE_MIN_MM = 50;
const SIZE_MAX_MM = 200;
/** Висота рельєфу (terrain_z_scale) і будинків (building_height_multiplier) — три зрозумілі кроки. */
const RELIEF_Z_OPTIONS = [{ v: 1, key: "reliefNatural" }, { v: 1.6, key: "reliefBold" }, { v: 2.5, key: "reliefDramatic" }] as const;
const BUILDING_H_OPTIONS = [{ v: 1.2, key: "bhLow" }, { v: 1.8, key: "bhNormal" }, { v: 2.6, key: "bhHigh" }] as const;

/** Пронумерована секція кроку 2 — власник (19.09.2026): «не зрозуміло, що де». */
function Section({ n, title, hint, children }: { n: number; title: string; hint?: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-2 rounded-[18px] border border-[var(--surface-border)] bg-white/55 p-3" data-testid={`guided-sec-${n}`}>
      <div className="flex items-center gap-2">
        <span className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[var(--accent-strong)] text-[12px] font-bold text-white">{n}</span>
        <h3 className="font-title text-[16px] font-semibold leading-tight text-[var(--text-primary)]">{title}</h3>
      </div>
      {hint && <p className="text-[11.5px] leading-snug text-[var(--text-secondary)]">{hint}</p>}
      {children}
    </section>
  );
}

/** Ряд пігулок-радіо (одне значення з кількох). */
function ChipGroup({ label, value, options, onPick, testId }: {
  label: string; value: number; options: Array<{ v: number; label: string; sub?: string }>; onPick: (v: number) => void; testId: string;
}) {
  return (
    <div data-testid={testId}>
      <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{label}</p>
      <div className="mt-1.5 flex flex-wrap gap-1.5" role="radiogroup" aria-label={label}>
        {options.map((o) => {
          const active = Math.abs(o.v - value) < 1e-6;
          return (
            <button
              key={o.v}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => onPick(o.v)}
              className={`inline-flex min-h-10 items-center gap-1.5 rounded-full border px-3.5 py-2 text-[12.5px] font-semibold transition ${
                active
                  ? "border-[rgba(11,92,87,0.5)] bg-[rgba(15,118,110,0.12)] text-[var(--text-primary)]"
                  : "border-[var(--surface-border)] bg-white/80 text-[var(--text-secondary)] hover:border-[rgba(11,92,87,0.35)] hover:text-[var(--text-primary)]"
              }`}
            >
              {o.label}
              {o.sub && <span className={`text-[11px] ${active ? "text-[var(--accent-strong)]" : "opacity-70"}`}>{o.sub}</span>}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** Пігулка-перемикач (увімк/вимк) з підказкою під нею. */
function ToggleChip({ on, label, hint, onToggle, testId }: { on: boolean; label: string; hint?: string; onToggle: () => void; testId: string }) {
  return (
    <div>
      <button
        type="button"
        aria-pressed={on}
        onClick={onToggle}
        data-testid={testId}
        className={`inline-flex min-h-10 items-center justify-center gap-2 rounded-full border px-3.5 py-2 text-[13px] font-semibold transition ${
          on
            ? "border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.1)] text-[var(--text-primary)]"
            : "border-[var(--surface-border)] bg-white/80 text-[var(--text-primary)] hover:border-[rgba(11,92,87,0.35)]"
        }`}
      >
        <span aria-hidden className="w-3 text-[var(--accent-strong)]">{on ? "✓" : ""}</span> {label}
      </button>
      {on && hint && <p className="mt-1.5 text-[12px] leading-snug text-[var(--text-secondary)]">{hint}</p>}
    </div>
  );
}

/** Сценарії, що лишаються всередині guided-флоу (брелок = лінк, повний = вихід). */
type ScenarioId = "map3d" | "relief" | "flat" | "magnet";
const SCENARIO_IDS: ScenarioId[] = ["map3d", "relief", "flat", "magnet"];

/**
 * СЦЕНАРНИЙ ВХІД /create (guided-режим). UX-аудит: новий користувач бачив ~43
 * інтерактивні контроли до першої генерації. Тут — ДВА кроки: ЩО створюємо →
 * ДЕ місце (пошук на карті, зона стає сама) + розмір і ОДНА кнопка на тому ж
 * екрані (нуль зайвих кліків, без окремого кроку).
 *
 * A-2/A-3/A-4 (2026-09-03): `?product=` відкриває одразу крок 2; крок 1 = 4
 * картки + один рядок «Ще:»; CTA активна ЗАВЖДИ (бейдж каже, яке місце буде
 * надруковано); екран «готово» = 2 дії (замовити / завантажити), кнопка
 * «Оновити превʼю» зʼявляється лише коли щось змінили після генерації.
 *
 * Компонент НЕ дублює логіку генерації: кнопка шле window-подію
 * `monadruk:guided-generate`, яку слухає прихована «машинна» копія
 * SimpleControlPanel (проп listenGuidedGenerate) і викликає той САМИЙ
 * handleGenerate, що й кнопка «Створити прев'ю». Прогрес/готовність читаємо зі
 * спільного generation-store (панель-поллер оновлює його як завжди). Ціна на
 * CTA — той самий quote-механізм (fetchQuote), що живить прайс у панелі.
 *
 * МОНТУЄТЬСЯ РІВНО РАЗ (aside із responsive-order): локальний step-стан не
 * розсинхронізується — на відміну від панелей, які монтуються двічі.
 */
export function ScenarioFlow({ onExitGuided }: { onExitGuided: () => void }) {
  const t = useTranslations("scenario");
  const dlQuota = useDownloadQuota();
  const locale = useLocale();
  // Діаспора (не-uk) бачить € за тим самим позиційним курсом, що й решта сайту.
  const isEu = locale !== "uk";
  const disp = (uah: number) => (isEu ? `€${mapPriceEur(uah)}` : `${uah} ₴`);

  const s = useGenerationStore(useShallow((st) => ({
    selectedArea: st.selectedArea,
    isGenerating: st.isGenerating,
    progress: st.progress,
    status: st.status,
    etaS: st.etaS,
    elapsedS: st.elapsedS,
    queued: st.queued,
    sourceWaitS: st.sourceWaitS,
    queueEta: st.queueEta,
    reconnecting: st.reconnecting,
    genError: st.genError,
    printPrep: st.printPrep,
    taskRestored: st.taskRestored,
    pendingGenerate: st.pendingGenerate,
    setPendingGenerate: st.setPendingGenerate,
    templateId: st.templateId,
    setTemplateId: st.setTemplateId,
    downloadUrl: st.downloadUrl,
    taskGroupId: st.taskGroupId,
    modelSizeMm: st.modelSizeMm,
    setModelSizeMm: st.setModelSizeMm,
    setSimpleFormat: st.setSimpleFormat,
    setSimpleFlatAms: st.setSimpleFlatAms,
    setSimpleRelief: st.setSimpleRelief,
    setPreviewMode: st.setPreviewMode,
    setShowHexGrid: st.setShowHexGrid,
    // Персоналізація (юзер: «немає легких доступів до вказати будинок, текст»):
    mapHighlightBuilding: st.mapHighlightBuilding,
    highlightPoints: st.highlightPoints,
    setMapHighlightBuilding: st.setMapHighlightBuilding,
    clearHighlights: st.clearHighlights,
    simpleMapLabel: st.simpleMapLabel,
    setSimpleMapLabel: st.setSimpleMapLabel,
    suggestedMapLabel: st.suggestedMapLabel,
    simpleConnector: st.simpleConnector,
    setSimpleConnector: st.setSimpleConnector,
    // Вигляд (19.09.2026): лише те, що реально міняє модель у guided-сценаріях.
    terrainZScale: st.terrainZScale,
    setTerrainZScale: st.setTerrainZScale,
    buildingHeightMultiplier: st.buildingHeightMultiplier,
    setBuildingHeightMultiplier: st.setBuildingHeightMultiplier,
    simpleFlatBuildings: st.simpleFlatBuildings,
    setSimpleFlatBuildings: st.setSimpleFlatBuildings,
    simpleFrame: st.simpleFrame,
    setSimpleFrame: st.setSimpleFrame,
    figureShape: st.figureShape,
    setFigureShape: st.setFigureShape,
    // Панно частинами (24.09.2026): N×N плиток з однієї рамки + пазл-замки на стиках.
    simplePanelMode: st.simplePanelMode,
    setSimplePanelMode: st.setSimplePanelMode,
    simpleSeriesConnectors: st.simpleSeriesConnectors,
    setSimpleSeriesConnectors: st.setSimpleSeriesConnectors,
  })));

  // T-2.1: текстовий лінк «Поділитись» у ГОТОВО-банері — та сама /share/{taskId}
  // сторінка, що й повна панель (SimpleControlPanel.doShare), без картинки-прев'ю
  // (це best-effort деталь повної панелі, тут досить самого лінку).
  const [shareCopied, setShareCopied] = useState(false);
  // Прапорець «іде завантаження друк-файлу» зі стора: кнопка живе тут, а саме
  // завантаження виконує SimpleControlPanel (через window-подію).
  const dlBusy = useGenerationStore((st) => st.downloadBusy);
  const doShareGuided = async () => {
    if (!s.taskGroupId) return;
    import("@/lib/analytics").then((m) => m.track("guided_share", { product: "map" })).catch(() => {});
    const url = `${window.location.origin}/share/${s.taskGroupId}`;
    try {
      if (typeof navigator.share === "function") {
        await navigator.share({ url, title: "Monadruk" }).catch(() => {});
      } else {
        await navigator.clipboard.writeText(url);
        setShareCopied(true);
        setTimeout(() => setShareCopied(false), 2500);
      }
    } catch { /* ignore */ }
  };

  // A/B "cta": рендеримо "A" на сервері (SSR-безпечно), варіант призначаємо
  // ЛИШЕ ПІСЛЯ монтування — інакше клієнтський рендер розійшовся б із SSR
  // (hydration mismatch), бо getVariant читає localStorage.
  const [ctaVariant, setCtaVariant] = useState<"A" | "B">("A");
  useEffect(() => { setCtaVariant(getVariant("cta")); }, []);

  const [scenario, setScenario] = useState<ScenarioId | null>(null);
  // started: генерацію запущено САМЕ з guided-флоу (відрізняємо від відновленої
  // з localStorage задачі минулої сесії — для неї success-екран не форсуємо).
  const [started, setStarted] = useState(false);
  // Напис на мапі — ОПЦІЙНИЙ (v3, юзер: «текст не по дефолту, а коли включаю»):
  // поле зʼявляється лише після кліку «Додати напис»; вибір сценарію чистить його.
  const [labelOn, setLabelOn] = useState(false);

  // A-6: єдиний вихід у розширений режим + подія для воронки (раніше 5 назв
  // і жодної події — не було видно, скільки людей тікає з простого режиму).
  const exitGuided = (from: string) => {
    import("@/lib/analytics").then((m) => m.track("mode_switch", { product: "map", to: "advanced", from })).catch(() => {});
    onExitGuided();
  };

  // v3.1 (юзер: «не можна пересувати рамку, коли обрання будинку увімкнене»):
  // режим кліку АВТО-ВИМИКАЄТЬСЯ одразу після вибору будинку — вибір лишається
  // (друк дивиться на highlightPoints, не на прапор), а рамка знову рухома.
  // Кнопка «Мій дім» повторним кліком повертає режим (додати ще/змінити).
  const hlCountRef = useRef(0);
  useEffect(() => {
    if (s.mapHighlightBuilding && s.highlightPoints.length > hlCountRef.current) {
      s.setMapHighlightBuilding(false);
      // Guided-воронка: позначили «мій дім» на мапі.
      import("@/lib/analytics").then((m) => m.track("guided_home", { product: "map", action: "mark" })).catch(() => {});
    }
    hlCountRef.current = s.highlightPoints.length;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.highlightPoints.length, s.mapHighlightBuilding]);

  // Guided-воронка: очистили позначку «мій дім» (обидва місця виклику clearHighlights).
  const clearHomeGuided = () => {
    import("@/lib/analytics").then((m) => m.track("guided_home", { product: "map", action: "clear" })).catch(() => {});
    s.clearHighlights();
  };

  // ЖИВА ЦІНА на CTA: ТОЙ САМИЙ quote-механізм, що в SimpleControlPanel /
  // StickyActionBar (fetchQuote з бекенд-прайсу; fallback нижче — mapPrices.ts).
  const [quote, setQuote] = useState<Quote | null>(null);
  useEffect(() => {
    if (!scenario) return;
    let alive = true;
    const magnet = scenario === "magnet";
    // Повзунок розміру шле десятки значень за секунду — питаємо ціну з паузою.
    const h = setTimeout(() => {
      fetchQuote("map", magnet ? 60 : s.modelSizeMm, magnet ? false : scenario === "relief")
        .then((q) => { if (alive) setQuote(q); });
    }, 200);
    return () => { alive = false; clearTimeout(h); };
  }, [scenario, s.modelSizeMm]);

  // АВТО-ЗОНА після пошуку. MapSearchBox шле `monadruk:map-goto` {lat,lon,label};
  // KeychainCropOverlay у MapSelector на це ВЖЕ переносить зону, але зберігає її
  // поточний розмір. Робимо так само, як pickTemplate у SimpleControlPanel
  // (клік по готовому району): повторний диспатч тієї ж події з ЯВНИМ widthM —
  // handler ставить зону ПІД ОБРАНИЙ РОЗМІР навколо точки (zoneForSizeM).
  // Guard: події, що вже несуть widthM або centerOnly, ігноруємо — інакше цикл.
  const sizeMmRef = useRef(80);
  useEffect(() => {
    sizeMmRef.current = scenario === "magnet" ? 60 : s.modelSizeMm;
  }, [scenario, s.modelSizeMm]);

  // A-4: CTA активна завжди, а «яке місце буде надруковано» каже БЕЙДЖ:
  // дефолтна київська рамка → «Центр Києва (за замовчуванням) — знайдіть свою
  // адресу», після пошуку/чіпа → «✓ Місце обрано: Львів», після ручного зсуву
  // рамки → «✓ Місце обрано: обрана ділянка на карті». Раніше сіра кнопка без
  // пояснення біля неї була головною «прихованою обовʼязковою дією» (F-11).
  const [placePicked, setPlacePicked] = useState(false);
  const createdAtRef = useRef(0);
  const touchedRef = useRef(false);
  const [placeLabel, setPlaceLabel] = useState<string>("");
  const prevAreaRef = useRef<typeof s.selectedArea>(null);
  // Перші ~2.5 с після монтування рамку ставить/перемасштабовує сам код (дефолт
  // Києва, пресет розміру з ?product=) — це НЕ вибір користувача (інакше магніт
  // одразу показував «Місце обрано: обрана ділянка на карті»).
  const mountedAtRef = useRef(Date.now());
  useEffect(() => {
    const prev = prevAreaRef.current;
    prevAreaRef.current = s.selectedArea;
    if (Date.now() - mountedAtRef.current < 2500 && Date.now() - lastGotoRef.current > 1500) return;
    if (prev && s.selectedArea && s.selectedArea !== prev) {
      // Зсув рамки пізніше ніж 2.5 с після старту генерації = дія користувача
      // (раніше — доліт карти після пошуку, він не має вмикати «Оновити превʼю»).
      // Nightly cache warming: реальна користувацька зміна після старту знімає
      // тег шаблону — прогрів більше не має сенсу кешувати саме цей deep-link.
      if (Date.now() - createdAtRef.current > 2500) { touchedRef.current = true; s.setTemplateId(null); }
      // Guided-воронка: ручний зсув/ресайз рамки — трек лише на ПЕРШИЙ перехід
      // «місце ще не обране» → «обране» (щоб не спамити подіями на кожен рух карти).
      if (!placePicked && !customPlaceTrackedRef.current) {
        customPlaceTrackedRef.current = true;
        import("@/lib/analytics").then((m) => m.track("guided_place", { product: "map", place: "custom" })).catch(() => {});
      }
      setPlacePicked(true);
      // Ручний зсув/ресайз рамки після пошуку — назву місця вже не гарантуємо.
      setPlaceLabel((cur) => (cur && lastGotoRef.current && Date.now() - lastGotoRef.current < 1500 ? cur : ""));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.selectedArea]);
  const customPlaceTrackedRef = useRef(false);
  const lastGotoRef = useRef(0);
  useEffect(() => {
    const onPick = (e: Event) => {
      const d = (e as CustomEvent).detail as
        | { lat?: number; lon?: number; widthM?: number; centerOnly?: boolean; label?: string }
        | undefined;
      // Лише користувацькі події (пошук/чіп): наші власні ре-диспатчі несуть widthM.
      if (!d || d.centerOnly || typeof d.widthM === "number") return;
      if (!Number.isFinite(d.lat) || !Number.isFinite(d.lon)) return;
      lastGotoRef.current = Date.now();
      // Nightly cache warming: реальна користувацька зміна після старту знімає
      // тег шаблону — прогрів більше не має сенсу кешувати саме цей deep-link.
      if (Date.now() - createdAtRef.current > 2500) { touchedRef.current = true; s.setTemplateId(null); }
      setPlacePicked(true);
      const label = typeof d.label === "string" ? d.label.trim() : "";
      if (label) setPlaceLabel(label);
      // Guided-воронка: дискретна подія (чіп/пошук), а не рух карти — трекаємо завжди.
      import("@/lib/analytics").then((m) => m.track("guided_place", { product: "map", place: label || "search" })).catch(() => {});
    };
    window.addEventListener("monadruk:map-goto", onPick as EventListener);
    return () => window.removeEventListener("monadruk:map-goto", onPick as EventListener);
  }, []);
  useEffect(() => {
    const onGoto = (e: Event) => {
      const d = (e as CustomEvent).detail as
        | { lat: number; lon: number; widthM?: number; centerOnly?: boolean }
        | undefined;
      if (!d || d.centerOnly) return;
      if (typeof d.widthM === "number" && d.widthM > 0) return;
      if (!Number.isFinite(d.lat) || !Number.isFinite(d.lon)) return;
      // Затримка — даємо overlay спершу відпрацювати оригінальну подію (переліт).
      window.setTimeout(() => {
        lastGotoRef.current = Date.now();
        window.dispatchEvent(new CustomEvent("monadruk:map-goto", {
          detail: { lat: d.lat, lon: d.lon, widthM: zoneForSizeM(sizeMmRef.current) },
        }));
      }, 120);
    };
    window.addEventListener("monadruk:map-goto", onGoto as EventListener);
    return () => window.removeEventListener("monadruk:map-goto", onGoto as EventListener);
  }, []);

  // Вибір сценарію = пресет формату у store (той самий шлях, що сегмент-контрол
  // «Формат» у SimpleControlPanel) + одразу крок 2. M (80 мм) — передвибраний.
  const pick = (id: ScenarioId, source: "card" | "url" = "card") => {
    // Guided-воронка: яку картку обирають (adмінка порівнює зі звичайним funnel).
    import("@/lib/analytics").then((m) => {
      m.track("guided_pick", { product: "map", scenario: id, source });
      m.track("guided_step", { product: "map", step: 2 });
    }).catch(() => {});
    s.setShowHexGrid(false);
    try { localStorage.setItem("3dmap_hex_grid", "0"); } catch { /* ignore */ }
    s.setPreviewMode(true);
    s.setSimpleFormat(id === "magnet" ? "magnet" : id === "flat" ? "flat" : "relief3d");
    if (id === "flat") s.setSimpleFlatAms(true);
    // ПІСЛЯ setSimpleFormat: relief3d зберігає попередній simpleRelief — явно
    // ставимо потрібне значення (map3d = без рельєфу, relief = з рельєфом).
    s.setSimpleRelief(id === "relief");
    if (id !== "magnet") s.setModelSizeMm(80);
    if (id === "magnet") s.setFigureShape("rounded");
    s.setSimplePanelMode(0);
    s.setSimpleMapLabel("");
    setLabelOn(false);
    setScenario(id);
    // Nightly cache warming: ручний вибір іншої картки (не той самий deep-link
    // ?template=) — це вже не «шаблонний» перегляд, знімаємо тег.
    if (source === "card") s.setTemplateId(null);
  };

  // T-3.1 (F-07) + A-2: deep-links. `?product=map3d|relief|flat|magnet` (головна,
  // сторінки нагод) відкриває одразу крок 2 з обраним товаром; `?template=<id>` /
  // `?city=<key>` (SEO-сторінки /maps, галерея шаблонів) — ще й ставить рамку на
  // район/центр міста. Подія map-goto БЕЗ widthM = «користувацька»: слухач вище
  // позначає «Місце обрано» і ре-диспатчить зону під обраний розмір.
  // ?lat=&lon= (кабінет: «Створити знову» зі збереженої моделі) — точні
  // координати замість готового шаблону/міста; ?size= (мм) обирає розмір,
  // якщо він збігається з одним із SIMPLE_SIZES.
  useEffect(() => {
    try {
      const p = new URLSearchParams(window.location.search);
      const prod = p.get("product");
      const tplId = p.get("template");
      const cityKey = p.get("city");
      // ПАСТКА: Number(null) === 0 → без перевірки p.has() кожен вхід стрибав на крок 2.
      const latParam = p.has("lat") ? Number(p.get("lat")) : NaN;
      const lonParam = p.has("lon") ? Number(p.get("lon")) : NaN;
      const hasLatLon = Number.isFinite(latParam) && Number.isFinite(lonParam);
      const prodId = prod && (SCENARIO_IDS as string[]).includes(prod) ? (prod as ScenarioId) : null;
      if (!prodId && !tplId && !cityKey && !hasLatLon) return;
      let center: [number, number] | undefined;
      let label = "";
      const tpl = tplId ? MAP_TEMPLATES.find((x) => x.id === tplId) : undefined;
      if (tpl) { center = tpl.center; label = tpl.district; }
      else if (cityKey) {
        // Назва — мовою інтерфейсу: 11.09.2026 віденець із /de/maps/vienna бачив
        // «Ort ausgewählt: Відень». Для міст України беремо назви зі сторінок /maps.
        const wc = WORLD_CITIES.find((x) => x.key === cityKey);
        const uc = CITIES.find((x) => x.key === cityKey);
        const loc = locale as keyof (typeof WORLD_CITIES)[number]["names"];
        if (wc) { center = wc.center; label = wc.names?.[loc] || wc.names?.en || wc.names?.uk || ""; }
        else if (uc) {
          center = uc.center;
          const page = CITY_PAGES.find((x) => x.key === cityKey);
          label = page?.names?.[loc] || uc.label || "";
        }
      } else if (hasLatLon) {
        center = [latParam, lonParam];
      }
      pick(prodId ?? (tpl && tpl.style === "relief" ? "relief" : "map3d"), "url");
      if (tpl?.sizeMm) s.setModelSizeMm(tpl.sizeMm);
      const sizeParam = Number(p.get("size"));
      if (Number.isFinite(sizeParam) && sizeParam >= SIZE_MIN_MM && sizeParam <= SIZE_MAX_MM) {
        const mm = Math.round(sizeParam / 5) * 5;
        s.setModelSizeMm(mm);
        // Чернетка конструктора (SimpleControlPanel, monadruk:draft:create) відновлює
        // свій modelSizeMm на маунті й перебивала явний ?size= з посилання —
        // повторюємо після її ефекту.
        window.setTimeout(() => useGenerationStore.getState().setModelSizeMm(mm), 150);
      }
      // Nightly cache warming: тегуємо задачу id шаблону з ?template=<id>, щоб
      // /api/generate передав template_id — нічний прогрів кешує саме ці прев'ю.
      // Скидається нижче (touchedRef) щойно користувач щось торкне.
      if (tplId && tpl) s.setTemplateId(tplId);
      if (!center) return;
      const [lat, lon] = center;
      window.setTimeout(() => {
        window.dispatchEvent(new CustomEvent("monadruk:map-goto", { detail: { lat, lon, label } }));
      }, 400);
    } catch { /* ignore */ }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── РОЗМІР (19.09.2026): будь-яке значення 50–200 мм, не лише S/M/L/XL ──
  const track = (ev: string, props: Record<string, unknown>) =>
    import("@/lib/analytics").then((m) => m.track(ev, { product: "map", ...props })).catch(() => {});
  /** «8 см» / «9.5 см» — без toLocaleString (локаль браузера ≠ локаль сайту в тестах). */
  const sizeCm = (mm: number) => `${Math.round(mm) / 10} см`;
  /** «M · 8 см» для пресету, «9.5 см» для довільного розміру — рекап, sticky-бар, месенджер. */
  const shapeDef = GUIDED_SHAPES.find((x) => x.id === s.figureShape) ?? GUIDED_SHAPES[0];
  /** Частини (панно): 1 = одна модель, 2/3 = сітка N×N плиток. Лише для обʼємних мап. */
  const parts: 1 | 2 | 3 = (scenario === "map3d" || scenario === "relief") && s.simplePanelMode > 0 ? (s.simplePanelMode as 2 | 3) : 1;
  const tiles = parts * parts;
  const setParts = (g: 1 | 2 | 3) => {
    s.setSimplePanelMode(g === 1 ? 0 : g);
    // Плитки панно — лише квадратні (сітка ріже прямокутну рамку).
    if (g > 1) s.setFigureShape("rounded");
    track("guided_look", { setting: "parts", value: g });
    syncZoneToSize(s.modelSizeMm, g);
  };
  const sizeLabel = (mm: number) => {
    const z = SIMPLE_SIZES.find((x) => x.mm === Math.round(mm));
    const size = z ? `${z.label} · ${z.cm}` : sizeCm(mm);
    // Форма в рекапі — лише коли не квадрат (квадрат = типовий, не шумимо).
    if (parts > 1) return `${t("partsGrid", { g: parts })} · ${size}`;
    return shapeDef.id === "rounded" || scenario === "magnet" ? size : `${t(shapeDef.key)} · ${size}`;
  };
  const isCustomSize = !SIMPLE_SIZES.some((z) => z.mm === s.modelSizeMm);
  // Текст числового поля живе окремо: інакше набір «1» → «12» → «120» клампився б
  // на кожній літері. Синхронізуємо зі стором, коли розмір змінили не з поля.
  const [sizeText, setSizeText] = useState(String(s.modelSizeMm / 10));
  useEffect(() => { setSizeText(String(s.modelSizeMm / 10)); }, [s.modelSizeMm]);
  const zoneSyncRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** Зона ЇДЕ ЗА РОЗМІРОМ: перецентровуємо навколо поточного центру з масштабом
   *  під нову плитку — інакше S зі старою 800-м зоною ловила червоне «завелика»,
   *  а XL марнувала деталізацію. Лише коли місце вже ОБРАНЕ (інакше ресайзили б
   *  дефолтну київську рамку, якої юзер не торкався). */
  const syncZoneToSize = (mm: number, g: number = parts) => {
    if (zoneSyncRef.current) { clearTimeout(zoneSyncRef.current); zoneSyncRef.current = null; }
    // 24.09.2026: рамку підганяємо ЗАВЖДИ (і дефолтну київську) — інакше після зміни
    // розміру карта писала одну ділянку, а підказка розміру іншу. Подія з widthM
    // не вважається «вибором місця» (слухач вище її ігнорує).
    const c = s.selectedArea?.getCenter?.();
    if (!c) return;
    window.dispatchEvent(new CustomEvent("monadruk:map-goto", { detail: { lat: c.lat, lon: c.lng, widthM: zoneForSizeM(mm * g) } }));
  };
  /** Єдиний вхід зміни розміру (пресет / повзунок / поле). `live` = під час
   *  тягнення повзунка: стор оновлюємо одразу (ціна, підпис), зону — з паузою. */
  const applySize = (raw: number, source: string, live = false) => {
    const mm = Math.round(Math.min(SIZE_MAX_MM, Math.max(SIZE_MIN_MM, raw)) / 5) * 5;
    if (mm !== s.modelSizeMm) s.setModelSizeMm(mm);
    if (!live) {
      track("guided_size", { sizeMm: mm, label: SIMPLE_SIZES.find((z) => z.mm === mm)?.label ?? "custom", source });
      syncZoneToSize(mm);
    } else {
      if (zoneSyncRef.current) clearTimeout(zoneSyncRef.current);
      zoneSyncRef.current = setTimeout(() => syncZoneToSize(mm), 350);
    }
  };

  // A-3: «Оновити превʼю» лише коли щось РЕАЛЬНО змінилось після генерації —
  // знімок параметрів у момент старту; поки він збігається, на екрані «готово»
  // рівно дві дії (замовити / завантажити).
  const areaKey = (() => {
    try {
      const b = s.selectedArea as unknown as { toBBoxString?: () => string } | null;
      return b?.toBBoxString ? b.toBBoxString() : "";
    } catch { return ""; }
  })();
  const paramsKey = JSON.stringify({
    scenario, size: s.modelSizeMm, area: areaKey, label: s.simpleMapLabel,
    hl: s.highlightPoints.length, conn: s.simpleConnector,
    zs: s.terrainZScale, bh: s.buildingHeightMultiplier, fb: s.simpleFlatBuildings, fr: s.simpleFrame,
  });
  const [snapshotKey, setSnapshotKey] = useState<string | null>(null);

  // H-4 (2026-09-05): на повільному звʼязку Leaflet вантажиться ~6 с, і доти
  // `selectedArea` порожній. Раніше кнопка була disabled — людина тапала в
  // мертву кнопку. Тепер намір ЗАПАМʼЯТОВУЄМО і запускаємо, щойно рамка є.
  const waitingForMap = s.pendingGenerate;
  const setWaitingForMap = s.setPendingGenerate;
  const create = () => {
    if (s.isGenerating) return;
    if (!s.selectedArea) { setWaitingForMap(true); return; }
    setWaitingForMap(false);
    import("@/lib/analytics")
      .then((m) => m.track("guided_generate", { product: "map", scenario, sizeMm: s.modelSizeMm, placePicked, place: placeLabel || "custom" }))
      .catch(() => {});
    setRan(false);
    setStarted(true);
    setSnapshotKey(paramsKey);
    createdAtRef.current = Date.now();
    touchedRef.current = false;
    window.dispatchEvent(new Event("monadruk:guided-generate"));
    // O-1 (реальний прогін на iPhone): після тапу по sticky-CTA сторінка стоїть на
    // картках розмірів, а смуга стадій/ETA лишається вище за екраном — юзер бачить
    // лише тісний бокс «СТАН 10%» у панелі превʼю. На мобільному ведемо до стадій.
    if (typeof window !== "undefined" && window.innerWidth < 1024) {
      window.setTimeout(() => {
        document.querySelector('[data-testid="generation-stages"]')?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 250);
    }
  };

  const basePrice = SIMPLE_SIZES[0].price;
  const reliefAddon = scenario === "relief" ? MAP_RELIEF_ADDON_UAH : 0;
  // Ціна на CTA: живий quote; fallback — канонічна таблиця mapPrices (без хардкоду).
  // Проміжні розміри (повзунок) — лінійно між тарифами, як і на бекенді.
  const tilePriceUah = scenario === "magnet"
    ? (quote?.price ?? MAP_MAGNET_PRICE_UAH)
    : (quote?.price ?? mapPriceForSizeUah(s.modelSizeMm) + reliefAddon);
  // Панно N×N = N² окремих плиток (quote — за одну).
  const ctaPriceUah = tilePriceUah * tiles;

  // 18.09.2026 (власник: «мало карток і реальні фото — бред»): картки = РЕНДЕРИ моделей
  // (public/showcase/card-r-*, render_product.py з реальних 3MF Львова), а не фото; крок 1 показує
  // ВСІ продукти сайту: 4 мапи (вибір сценарію тут) + гора/панно/світ/брелок (перехід на свій режим).
  const cards: Array<{
    id: ScenarioId | "mountain" | "panno" | "world" | "keychain";
    img: string;
    title: string;
    desc: string;
    price: string;
    href?: string;
  }> = [
    { id: "map3d", img: "card-r-map3d", title: t("map3dTitle"), desc: t("map3dDesc"), price: t("from", { price: disp(basePrice) }) },
    { id: "relief", img: "card-r-relief", title: t("reliefTitle"), desc: t("reliefDesc"), price: t("from", { price: disp(basePrice + MAP_RELIEF_ADDON_UAH) }) },
    { id: "flat", img: "card-r-flat", title: t("flatTitle"), desc: t("flatDesc"), price: t("from", { price: disp(basePrice) }) },
    { id: "magnet", img: "card-r-magnet", title: t("magnetTitle"), desc: t("magnetDesc"), price: disp(MAP_MAGNET_PRICE_UAH) },
    { id: "mountain", img: "card-r-mountain", title: t("mountainsTitle"), desc: t("mountainsDesc"), price: t("priceOnRequest"), href: "/mountains" },
    { id: "panno", img: "card-r-panno", title: t("pannoTitle"), desc: t("pannoDesc"), price: t("from", { price: disp(basePrice * 4) }), href: "" },
    { id: "keychain", img: "card-r-keychain", title: t("keychainTitle"), desc: t("keychainDesc"), price: t("from", { price: disp(KEYCHAIN_PRICE_UAH) }), href: "/keychains" },
    { id: "world", img: "card-r-world", title: t("worldsTitle"), desc: t("worldsDesc"), price: t("priceOnRequest"), href: "/worlds" },
  ];

  // F-10: «не вдалося» показуємо лише якщо генерація СПРАВДІ стартувала (isGenerating
  // побував true) і завершилась без файлу. Без цього в асинхронному проміжку між кліком
  // і isGenerating=true червона помилка блимала при кожному успішному кліку.
  const [ran, setRan] = useState(false);
  useEffect(() => { if (s.isGenerating) setRan(true); }, [s.isGenerating]);
  // perf-2026-09-03: чесний ETA — залишок від медіани реальних прогонів (бекенд).
  const etaText = (() => {
    if (typeof s.etaS !== "number" || s.etaS <= 0) return null;
    const elapsed = typeof s.elapsedS === "number" ? s.elapsedS : 0;
    // Перевищили прогноз на 20 % — чесно кажемо «довше, ніж зазвичай», а не «менше хвилини».
    if (elapsed > s.etaS * 1.2 + 15) return t("etaOver");
    const left = Math.max(0, s.etaS - elapsed);
    if (left < 45) return t("etaSoon");
    return t("etaLeft", { min: Math.max(1, Math.ceil(left / 60)) }); // P-1: округлення ВГОРУ — краще недообіцяти
  })();
  // Карта віддала рамку — виконуємо відкладений намір користувача.
  useEffect(() => {
    if (!waitingForMap || !s.selectedArea || s.isGenerating) return;
    // ⚠️НЕ запускати синхронно в цьому ефекті: панель-слухач перепідписується
    // на `monadruk:guided-generate` КОЖЕН рендер (ефект без deps), тож у мить
    // нашого коміту слухача може не бути — подія летіла в порожнечу (відтворено).
    const id = window.setTimeout(() => { setWaitingForMap(false); create(); }, 60);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [waitingForMap, s.selectedArea, s.isGenerating]);
  const generatingView = s.isGenerating;
  // C-1: після F5 задача відновлюється зі сторіджу — показуємо готову модель.
  const successView = (started || s.taskRestored) && !s.isGenerating && !!s.downloadUrl;
  // C-3: помилка = реальний fail з бекенду (з причиною), а не «немає файлу».
  const failedNote = !!s.genError && !s.isGenerating && (started || s.taskRestored);
  // Guided-воронка: результат генерації — ОДИН раз на прогін (ключ = момент
  // старту create(), а не taskGroupId, бо в помилки його може не бути).
  // ⭐ПАСТКА 08.09: було useRef(0). Для ВІДНОВЛЕНОЇ з localStorage задачі
  // createdAtRef теж лишається 0, тож умова `!==` ніколи не спрацьовувала —
  // і `guided_result` не надсилався ЖОДНОГО разу (за 7 днів проду: 31
  // guided_download і 0 guided_result). Через це в адмінці й тижневому
  // дайджесті «Генерацій ✓/✗» завжди показувало 0/0. -1 не дорівнює жодному
  // Date.now() і жодному 0 → перший успіх рахується завжди.
  const resultTrackedAtRef = useRef(-1);
  useEffect(() => {
    if (successView && resultTrackedAtRef.current !== createdAtRef.current) {
      resultTrackedAtRef.current = createdAtRef.current;
      import("@/lib/analytics").then((m) => m.track("guided_result", { product: "map", ok: true, elapsedS: s.elapsedS })).catch(() => {});
    } else if (failedNote && resultTrackedAtRef.current !== createdAtRef.current) {
      resultTrackedAtRef.current = createdAtRef.current;
      import("@/lib/analytics").then((m) => m.track("guided_result", { product: "map", ok: false, reason: s.genError || undefined })).catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [successView, failedNote]);
  // Готово: якщо користувач нічого не чіпав під час генерації, знімок = поточні
  // параметри (доліт карти/авто-зона після пошуку не мають давати «Оновити превʼю»).
  const prevSuccessRef = useRef(false);
  useEffect(() => {
    if (successView && !prevSuccessRef.current && !touchedRef.current) setSnapshotKey(paramsKey);
    prevSuccessRef.current = successView;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [successView]);
  const dirty = successView && snapshotKey !== null && snapshotKey !== paramsKey;
  const displayStep = generatingView || successView || scenario !== null ? 2 : 1;

  const cardBtnCls = "group flex flex-col overflow-hidden rounded-[18px] border border-[var(--surface-border)] bg-white/80 text-left shadow-[0_4px_14px_rgba(15,23,42,0.05)] transition hover:border-[rgba(11,92,87,0.45)] hover:shadow-[0_8px_24px_rgba(15,23,42,0.1)]";
  // "bronze"/"sm" — сирий слот примітива (Button.tsx): повністю бесспокий вигляд,
  // жоден із 4 варіантів його не описує без спотворення, тож весь клас іде через className.
  const moreLinkCls = `rounded-full border border-[var(--surface-border)] bg-white/70 px-2.5 py-1 text-[11.5px] font-semibold text-[var(--text-secondary)] transition hover:border-[rgba(11,92,87,0.4)] hover:text-[var(--text-primary)]`;

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-[30px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_22px_70px_rgba(15,23,42,0.08)] backdrop-blur" data-testid="scenario-flow">
      {/* Шапка: степ-індикатор + назад до сценаріїв */}
      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-[var(--surface-border)] px-4 py-3">
        <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
          {successView ? t("readyBadge") : t("stepOf", { step: displayStep })}
        </span>
        {scenario !== null && !generatingView && (
          <Button
            variant="secondary"
            size="sm"
            onClick={() => { setScenario(null); setStarted(false); }}
          >
            <ArrowLeft size={12} /> {t("back")}
          </Button>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {scenario === null ? (
          /* ── КРОК 1: ЩО СТВОРЮЄМО? ── */
          <div>
            <h2 className="font-title text-lg font-semibold text-[var(--text-primary)]">{t("step1Title")}</h2>
            <div className="mt-3 grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-2">
              {cards.map((c, i) => {
                const inner = (<>
                  {/* N-3 (перф): картка 640×480 показується ~150–300 px → 400w за 1×, 640 за 2×
                      (489→189 КБ на 12 файлів). Перші дві картки — над згином і є LCP-елементом
                      на /create → eager, решта lazy. eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`/showcase/${c.img}-400.webp`}
                    srcSet={`/showcase/${c.img}-400.webp 1x, /showcase/${c.img}.webp 2x`}
                    alt={c.title}
                    loading={i < 2 ? "eager" : "lazy"}
                    className="aspect-[4/3] w-full object-cover transition duration-500 group-hover:scale-[1.04]"
                  />
                  <span className="flex flex-1 flex-col gap-0.5 px-2.5 py-2">
                    <span className="text-[13px] font-semibold leading-tight text-[var(--text-primary)]">{c.title}</span>
                    <span className="text-[12px] font-semibold text-[var(--accent-strong)]">{c.price}</span>
                    <span className="text-[11px] leading-snug text-[var(--text-secondary)]">{c.desc}</span>
                  </span>
                </>);
                if (c.id === "panno") {
                  return <button key={c.id} type="button" onClick={() => { pick("map3d"); s.setSimplePanelMode(2); s.setSimpleSeriesConnectors(true); }} className={cardBtnCls} data-testid="scenario-card-panno">{inner}</button>;
                }
                return c.href
                  ? <Link key={c.id} href={c.href} className={cardBtnCls} data-testid={`scenario-card-${c.id}`}>{inner}</Link>
                  : <button key={c.id} type="button" onClick={() => pick(c.id as ScenarioId)} className={cardBtnCls} data-testid={`scenario-card-${c.id}`}>{inner}</button>;
              })}
            </div>
            {/* A-2: крок 1 = вибір ТОВАРУ. Решта можливостей сайту — один компактний
                рядок лінків (повний блок з описами живе на головній, T-D.6), щоб
                перший екран конструктора не був мапою сайту з 15 цілей. */}
            <div className="mt-4">
              <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("moreTitle")}</p>
              <div className="mt-2 flex flex-wrap gap-1.5" data-testid="scenario-more">
                <Link href="/maket" className={moreLinkCls}>{t("maketTitle")}</Link>
                <Link href="/showcase" className={moreLinkCls}>{t("showcaseTitle")}</Link>
                {/* ПАСТКА: variant="bronze" давав білий текст на білому тлі через className-override →
                    пігулка виглядала ПОРОЖНЬОЮ (скрін власника 18.09). Тепер явні кольори. */}
                <button type="button" onClick={() => exitGuided("step1")} data-testid="scenario-full"
                  className={`${moreLinkCls} !bg-[var(--bronze,#8E6B3D)] !text-white !border-transparent`}>
                  {t("fullTitle")} · {t("fullDescLong")}
                </button>
              </div>
            </div>
          </div>
        ) : (
          /* ── КРОК 2: ДЕ ВАШЕ МІСЦЕ? + розмір і CTA на тому ж екрані ──
              (карта лишається видимою поруч/вище; рамка зони — інтерактивна) */
          <div className="flex flex-col gap-3">
            {/* ГОТОВО-банер (A-3): рівно дві дії — замовити або завантажити. Усі
                контролі НИЖЧЕ лишаються живими; «Оновити превʼю» зʼявляється
                тільки коли щось змінили. Превʼю крутиться на сцені поруч. */}
            {successView && (
              <div className="flex flex-col gap-2.5" data-testid="guided-success">
                <div className="flex items-center gap-2 text-[16px] font-semibold text-[var(--text-primary)]">
                  <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-[var(--accent-strong)] text-white"><Check size={15} /></span>
                  {t("readyTitle")}
                </div>
                {/* Рекап: що саме готове (сценарій · розмір · місце) + підказка. */}
                <p className="text-[12.5px] leading-snug text-[var(--text-secondary)]">
                  <b className="text-[var(--text-primary)]">
                    {scenario === "magnet" ? t("magnetTitle") : `${cards.find((c) => c.id === scenario)?.title ?? ""} · ${sizeLabel(s.modelSizeMm)}`}
                    {placeLabel ? ` · ${placeLabel}` : ""}
                  </b>
                  {" — "}{t("readyHint")}
                </p>
                {/* R-04 (Cities3ds): одна чесна фраза-обіцянка проти головного сумніву
                    «а надрукують те, що бачу?». Превʼю і 3MF — з одного canonical_2d. */}
                <p className="flex items-start gap-1.5 text-[11.5px] leading-snug text-[var(--text-secondary)]" data-testid="guided-promise">
                  <ShieldCheck size={13} className="mt-[1px] shrink-0 text-[var(--accent-strong)]" /> {t("previewPromise")}
                </p>
                <Button
                  variant="bronze"
                  size="md"
                  onClick={() => {
                    import("@/lib/analytics").then((m) => m.track("guided_order_click", { product: "map", priceUah: ctaPriceUah })).catch(() => {});
                    window.dispatchEvent(new Event("monadruk:open-order"));
                  }}
                  data-testid="guided-order"
                  className="w-full"
                >
                  <ShoppingBag size={18} /> {t("orderPrint")} · {disp(ctaPriceUah)}
                </Button>
                <p className="text-center text-[11.5px] leading-snug text-[var(--text-secondary)]">{t("readyDelivery")}</p>
                {/* «Не зрозуміло, як качати» (власник): завантаження — рівноправна
                    кнопка з чесним підписом (вхід через Google, файл готується ≈2 хв). */}
                <div className="flex items-center gap-2 pt-0.5">
                  <span className="h-px flex-1 bg-[var(--surface-border)]" />
                  <span className="text-[10.5px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">{t("waySelf")}</span>
                  <span className="h-px flex-1 bg-[var(--surface-border)]" />
                </div>
                <Button
                  variant="secondary"
                  size="lg"
                  data-testid="guided-download"
                  // ⭐08.09.2026: кнопка не блокувалась і кожен клік запускав НОВУ
                  // повну генерацію друку (прод: 31 клік за 5 хв → 35 генерацій).
                  disabled={dlBusy}
                  busy={dlBusy}
                  onClick={() => {
                    import("@/lib/analytics").then((m) => m.track("guided_download", { product: "map" })).catch(() => {});
                    window.dispatchEvent(new Event("monadruk:guided-download"));
                  }}
                  className="w-full"
                >
                  <Download size={17} /> {t("downloadCta")}
                </Button>
                <p className="text-center text-[11px] leading-snug text-[var(--text-secondary)]">{t("downloadSub")}</p>
                {/* T-D.5: залогінений бачить залишок безкоштовних файлів прямо тут. */}
                {dlQuota && !dlQuota.isAdmin && dlQuota.limit > 0 && (
                  <p className="text-center text-[11px] font-semibold text-[var(--accent-strong)]">{t("quotaLeft", { n: dlQuota.remaining, limit: dlQuota.limit })}</p>
                )}
                {/* S-1/S-2: месенджер-замовлення + «що заважає» (див. SalesAlternatives). */}
                <SalesAlternatives
                  product="map"
                  taskId={s.taskGroupId}
                  summary={`${scenario === "magnet" ? t("magnetTitle") : `${cards.find((c) => c.id === scenario)?.title ?? ""} · ${sizeLabel(s.modelSizeMm)}`}${placeLabel ? ` · ${placeLabel}` : ""}`}
                  priceUah={ctaPriceUah}
                />
                {/* T-2.1: текстовий лінк, НЕ третя кнопка — банер лишається з рівно
                    двома заповненими кнопками (замовити/завантажити). */}
                {!!s.taskGroupId && (
                  <Button
                    variant="ghost"
                    size="sm"
                    data-testid="guided-share"
                    onClick={doShareGuided}
                    className="mx-auto"
                  >
                    <Share2 size={13} /> {shareCopied ? t("shareCopied") : t("shareLink")}
                  </Button>
                )}
                {/* I-2 (R-04/TerraPrinter): на десктопі — QR, щоб відкрити цю ж сцену на
                    телефоні й показати рідним; на мобільному QR безглуздий (є «Поділитись»). */}
                {!!s.taskGroupId && (
                  <ShareQr
                    url={`${window.location.origin}/share/${s.taskGroupId}`}
                    size={88}
                    label={t("qrHint")}
                    className="mx-auto hidden lg:flex"
                  />
                )}
                <div className="my-0.5 flex items-center gap-2">
                  <span className="h-px flex-1 bg-[var(--surface-border)]" />
                  <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("changeSomething")}</span>
                  <span className="h-px flex-1 bg-[var(--surface-border)]" />
                </div>
              </div>
            )}
            {/* Генерація (A-5): ОДНА смуга прогресу з названими етапами замість
                сирого рядка статусу бекенду; панель під нею жива. */}
            {generatingView && (
              <GenerationStages
                progress={s.progress || 0}
                kind={scenario === "flat" || scenario === "magnet" ? "flat" : "map"}
                title={t("generating")}
                note={
                  // ⭐09.09: під час рестарту бекенду опитувач раніше мовчав
                  // ~10 с і оголошував «модель застаріла». Тепер чекаємо довше
                  // і чесно кажемо, що відбувається.
                  s.reconnecting
                    ? t("reconnecting")
                    : s.sourceWaitS != null
                      ? t("sourceWait", { s: s.sourceWaitS })
                      : t("etaNote")
                }
                eta={etaText}
                queued={s.queued}
                queuedTitle={t("queuedTitle")}
                queuedNote={
                  // ⭐09.09.2026: «за кілька хвилин» було неправдою — прод 08.09
                  // показав очікування до 43 хв під цим самим написом. Коли
                  // сервер знає оцінку, показуємо число, яке справді зменшується.
                  s.queueEta == null
                    ? t("queuedNote")
                    : s.queueEta < 60
                      ? t("queuedEtaSoon")
                      : t("queuedEta", { left: t("etaLeft", { min: Math.round(s.queueEta / 60) }) })
                }
                printPrep={s.printPrep}
                printPrepLabel={t("printPrepLine")}
                printPrepOffer={
                  <div className="mt-1 rounded-xl border border-[rgba(142,107,61,0.35)] bg-[rgba(255,247,230,0.75)] p-2.5" data-testid="wait-offer">
                    <p className="text-[12.5px] font-semibold text-[var(--text-primary)]">{t("waitOfferTitle")}</p>
                    <p className="mt-0.5 text-[11.5px] leading-snug text-[var(--text-secondary)]">{t("waitOfferSub")}</p>
                    <Button
                      variant="bronze"
                      size="sm"
                      data-testid="wait-offer-cta"
                      className="mt-2 w-full"
                      onClick={() => {
                        import("@/lib/analytics").then((m) => m.track("wait_offer_click", { product: "map" })).catch(() => {});
                        window.dispatchEvent(new Event("monadruk:open-order"));
                      }}
                    >
                      {t("orderPrint")} · {disp(ctaPriceUah)}
                    </Button>
                  </div>
                }
                cancelLabel={t("cancelGen")}
                onCancel={() => window.dispatchEvent(new Event("monadruk:guided-cancel"))}
                stages={{ data: t("stageData"), terrain: t("stageTerrain"), detail: t("stageDetail"), file: t("stageFile") }}
              />
            )}
            <Section n={1} title={t("step2Title")}>
            {/* ПОШУК ПРЯМО В ПАНЕЛІ (v2): раніше поле жило лише на карті, а панель
                давала довгу інструкцію «йдіть шукайте там» — погляд стрибав. Тепер
                друкуєш адресу тут; та сама подія monadruk:map-goto → автозона. */}
            <div className="rounded-full border border-[var(--surface-border)] bg-white/80 px-1.5 py-0.5 focus-within:border-[rgba(11,92,87,0.45)]">
              <MapSearchBox variant="panel" />
            </div>
            <div className="flex flex-wrap gap-1.5">
              {QUICK_CITIES.map((c) => (
                <button
                  key={c.en}
                  type="button"
                  onClick={() => window.dispatchEvent(new CustomEvent("monadruk:map-goto", {
                    detail: { lat: c.lat, lon: c.lon, label: locale === "uk" ? c.uk : c.en },
                  }))}
                  className="min-h-10 rounded-full border border-[var(--surface-border)] bg-white/70 px-3.5 py-2 text-[12px] font-semibold text-[var(--text-secondary)] transition hover:border-[rgba(11,92,87,0.4)] hover:text-[var(--text-primary)]"
                >
                  {locale === "uk" ? c.uk : c.en}
                </button>
              ))}
            </div>
            {/* A-4: бейдж завжди каже, ЯКЕ місце піде в друк. */}
            {!placePicked ? (
              <div className="flex flex-col gap-1" data-testid="place-default">
                <div className="inline-flex items-center gap-2 self-start rounded-full border border-[var(--surface-border)] bg-white/80 px-3.5 py-2 text-[13px] font-semibold text-[var(--text-primary)]">
                  <MapPin size={15} className="text-[var(--accent-strong)]" /> {t("defaultPlace")}
                </div>
                <p className="text-[11.5px] leading-snug text-[var(--text-secondary)]">{t("defaultPlaceHint")}</p>
              </div>
            ) : (
              <div className="inline-flex max-w-full items-center gap-2 self-start rounded-full border border-[rgba(11,92,87,0.35)] bg-[rgba(15,118,110,0.1)] px-3.5 py-2 text-[13px] font-semibold text-[var(--text-primary)]" data-testid="place-picked">
                <Check size={15} className="shrink-0 text-[var(--accent-strong)]" />
                <span className="truncate">{t("placeChosen")}{placeLabel ? `: ${placeLabel}` : `: ${t("customPlace")}`}</span>
              </div>
            )}
            </Section>
            {/* ── 2. РОЗМІР (19.09.2026, власник: «розмірів повноцінно не можна
                вибрати») — 4 звичні пресети + БУДЬ-ЯКИЙ розмір 50–200 мм повзунком/
                полем. Ціна жива з /api/quote (бекенд рахує проміжні розміри лінійно
                між тарифами; fallback — та сама формула mapPriceForSizeUah). Магніт
                фіксований 60 мм. */}
            {scenario === "magnet" ? (
              <Section n={2} title={t("sizeTitle")}>
                <p className="text-[13px] leading-relaxed text-[var(--text-secondary)]">
                  {t("magnetFixedNote", { price: disp(quote?.price ?? MAP_MAGNET_PRICE_UAH) })}
                </p>
              </Section>
            ) : (
              <Section n={2} title={t("sizeTitle")} hint={t("sizeHint")}>
                {/* 24.09.2026 (власник: «щоб одразу кастомний розмір, форма, зʼєднання /
                    панно частинами — і все зрозуміло»): три компактні рядки замість
                    високих карток + окремого повзунка. */}
                <div data-testid="guided-shape">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("shapeLabel")}</p>
                  <div className="mt-1.5 grid grid-cols-4 gap-1.5" role="radiogroup" aria-label={t("shapeLabel")}>
                    {GUIDED_SHAPES.map((sh) => {
                      const active = s.figureShape === sh.id;
                      const disabled = parts > 1 && sh.id !== "rounded";
                      return (
                        <button
                          key={sh.id}
                          type="button"
                          role="radio"
                          aria-checked={active}
                          disabled={disabled}
                          data-testid={`shape-${sh.id}`}
                          onClick={() => { s.setFigureShape(sh.id); track("guided_look", { setting: "shape", value: sh.id }); }}
                          className={`flex min-h-[52px] flex-col items-center justify-center gap-0.5 rounded-[14px] border px-1 py-1.5 transition disabled:cursor-not-allowed disabled:opacity-40 ${
                            active
                              ? "border-[rgba(11,92,87,0.5)] bg-[rgba(15,118,110,0.12)] text-[var(--accent-strong)]"
                              : "border-[var(--surface-border)] bg-white/80 text-[var(--text-secondary)] hover:border-[rgba(11,92,87,0.3)]"
                          }`}
                        >
                          <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d={sh.path} fill="currentColor" fillOpacity={active ? 0.25 : 0.12} stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" /></svg>
                          <span className="text-[11px] font-semibold leading-tight text-[var(--text-primary)]">{t(sh.key)}</span>
                        </button>
                      );
                    })}
                  </div>
                  <p className="mt-1 text-[11.5px] leading-snug text-[var(--text-secondary)]">{parts > 1 ? t("shapeTilesNote") : t("shapeHint")}</p>
                </div>

                <div data-testid="guided-size">
                  <div className="flex items-baseline justify-between gap-2">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{parts > 1 ? t("tileSizeLabel") : t("sizeLabel")}</p>
                    <span className="text-[13px] font-bold text-[var(--accent-strong)]" data-testid="custom-size-price">{disp(ctaPriceUah)}</span>
                  </div>
                  <div className="mt-1.5 flex flex-wrap items-center gap-1.5" role="radiogroup" aria-label={t("sizeLabel")}>
                    {SIMPLE_SIZES.map((z) => {
                      const active = s.modelSizeMm === z.mm;
                      return (
                        <button
                          key={z.key}
                          type="button"
                          role="radio"
                          aria-checked={active}
                          aria-label={`${z.label} · ${z.cm} · ${disp(z.price + reliefAddon)}`}
                          onClick={() => applySize(z.mm, z.label)}
                          className={`inline-flex min-h-10 flex-col items-center justify-center rounded-[14px] border px-2.5 py-1 leading-tight transition ${
                            active
                              ? "border-[rgba(11,92,87,0.5)] bg-[rgba(15,118,110,0.12)]"
                              : "border-[var(--surface-border)] bg-white/80 hover:border-[rgba(11,92,87,0.3)]"
                          }`}
                        >
                          <span className="text-[12.5px] font-bold text-[var(--text-primary)]">{z.cm}</span>
                          <span className="text-[10.5px] font-semibold text-[var(--accent-strong)]">{disp(z.price + reliefAddon)}</span>
                        </button>
                      );
                    })}
                    {/* Свій розмір — ОДРАЗУ поруч з пресетами, у сантиметрах (крок 0,5). */}
                    <label
                      className={`inline-flex min-h-10 items-center gap-1 rounded-[14px] border px-2.5 py-1 transition focus-within:border-[rgba(11,92,87,0.5)] ${
                        isCustomSize ? "border-[rgba(11,92,87,0.5)] bg-[rgba(15,118,110,0.12)]" : "border-dashed border-[var(--surface-border)] bg-white/80"
                      }`}
                      data-testid="custom-size"
                    >
                      <span className="text-[11.5px] font-semibold text-[var(--text-secondary)]">{t("customSizeShort")}</span>
                      <input
                        type="number"
                        inputMode="decimal"
                        min={SIZE_MIN_MM / 10}
                        max={SIZE_MAX_MM / 10}
                        step={0.5}
                        value={sizeText}
                        onChange={(e) => {
                          setSizeText(e.target.value);
                          const v = Number(e.target.value.replace(",", "."));
                          if (Number.isFinite(v) && v >= SIZE_MIN_MM / 10 && v <= SIZE_MAX_MM / 10) applySize(v * 10, "input", true);
                        }}
                        onBlur={() => { const v = Number(sizeText.replace(",", ".")); applySize(Number.isFinite(v) && v > 0 ? v * 10 : s.modelSizeMm, "input"); }}
                        onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
                        aria-label={t("customSizeMm")}
                        className="w-11 bg-transparent text-right text-[13px] font-bold text-[var(--text-primary)] focus:outline-none"
                        data-testid="size-input"
                      />
                      <span className="text-[12px] text-[var(--text-secondary)]">{t("cm")}</span>
                    </label>
                  </div>
                  <p className="mt-1 text-[11.5px] leading-snug text-[var(--text-secondary)]" data-testid="size-summary">
                    {(() => {
                      const z = SIMPLE_SIZES.find((x) => x.mm === s.modelSizeMm);
                      const cmp = z ? `${t(`sizeCmp${z.label}` as "sizeCmpS" | "sizeCmpM" | "sizeCmpL" | "sizeCmpXL")} · ` : "";
                      return `${cmp}${t("areaApprox", { zone: zoneForSizeM(s.modelSizeMm * parts) })} · ${t("sizeRange", { min: SIZE_MIN_MM / 10, max: SIZE_MAX_MM / 10 })}`;
                    })()}
                  </p>
                </div>

                {(scenario === "map3d" || scenario === "relief") && (
                  <div data-testid="guided-parts">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("partsLabel")}</p>
                    <div className="mt-1.5 grid grid-cols-3 gap-1.5" role="radiogroup" aria-label={t("partsLabel")}>
                      {([1, 2, 3] as const).map((g) => {
                        const active = parts === g;
                        return (
                          <button
                            key={g}
                            type="button"
                            role="radio"
                            aria-checked={active}
                            data-testid={`parts-${g}`}
                            onClick={() => setParts(g)}
                            className={`flex min-h-[52px] flex-col items-center justify-center gap-0.5 rounded-[14px] border px-1 py-1.5 transition ${
                              active
                                ? "border-[rgba(11,92,87,0.5)] bg-[rgba(15,118,110,0.12)]"
                                : "border-[var(--surface-border)] bg-white/80 hover:border-[rgba(11,92,87,0.3)]"
                            }`}
                          >
                            <span className="grid gap-[2px]" style={{ gridTemplateColumns: `repeat(${g}, 7px)` }} aria-hidden="true">
                              {Array.from({ length: g * g }, (_, i) => <span key={i} className={`h-[7px] w-[7px] rounded-[1.5px] ${active ? "bg-[var(--accent-strong)]" : "bg-[var(--text-secondary)] opacity-50"}`} />)}
                            </span>
                            <span className="text-[11.5px] font-semibold leading-tight text-[var(--text-primary)]">{g === 1 ? t("partsOne") : t("partsGrid", { g })}</span>
                            <span className="text-[10.5px] leading-tight text-[var(--text-secondary)]">{g === 1 ? t("partsOneSub") : t("partsTiles", { n: g * g })}</span>
                          </button>
                        );
                      })}
                    </div>
                    {parts === 1 && (
                      <div className="mt-1.5">
                        {/* Зʼєднувачі (юзер: «немає щоб включити зʼєднувачі») — тепер поруч
                            із «Частинами», де їх шукають: пази по краях одиночної плитки,
                            щоб пізніше доставити сусідню. Той самий store-прапор, що в повній панелі. */}
                        <ToggleChip
                          on={s.simpleConnector}
                          label={t("connectors")}
                          hint={t("connectorsHint")}
                          onToggle={() => s.setSimpleConnector(!s.simpleConnector)}
                          testId="single-connectors"
                        />
                      </div>
                    )}
                    {parts > 1 && (
                      <div className="mt-1.5 flex flex-col gap-1.5">
                        <ToggleChip
                          on={s.simpleSeriesConnectors}
                          label={t("partsConnect")}
                          hint={t("partsConnectHint")}
                          onToggle={() => { s.setSimpleSeriesConnectors(!s.simpleSeriesConnectors); track("guided_look", { setting: "parts_connectors", value: !s.simpleSeriesConnectors }); }}
                          testId="parts-connectors"
                        />
                        <p className="text-[11.5px] leading-snug text-[var(--text-secondary)]" data-testid="parts-total">
                          {t("partsTotal", { cm: Math.round(s.modelSizeMm * parts) / 10, n: tiles, tile: disp(tilePriceUah), total: disp(ctaPriceUah) })}
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </Section>
            )}
            {/* ── 3. ВИГЛЯД — лише ті налаштування, що РЕАЛЬНО міняють модель у цьому
                сценарії (перевірено по buildSingleMapReq/бекенду): висота рельєфу
                (terrain_z_scale), висота будинків (building_height_multiplier), пласкі
                будинки й рамка з компасом для плоских. Шари (дороги/вода/парки)
                СВІДОМО не тут — на бекенді вони керують лише роздільними
                превʼю-частинами, не самою моделлю (був би «мовчазний» перемикач). */}
            <Section n={3} title={t("lookTitle")} hint={t("lookHint")}>
              {scenario === "relief" && (
                <ChipGroup
                  label={t("reliefHeight")}
                  value={s.terrainZScale}
                  options={RELIEF_Z_OPTIONS.map((o) => ({ v: o.v, label: t(o.key as "reliefNatural" | "reliefBold" | "reliefDramatic"), sub: `×${o.v}` }))}
                  onPick={(v) => { s.setTerrainZScale(v); track("guided_look", { setting: "relief_z", value: v }); }}
                  testId="relief-z"
                />
              )}
              {(scenario === "map3d" || scenario === "relief") && (
                <ChipGroup
                  label={t("buildingHeight")}
                  value={s.buildingHeightMultiplier}
                  options={BUILDING_H_OPTIONS.map((o) => ({ v: o.v, label: t(o.key as "bhLow" | "bhNormal" | "bhHigh"), sub: `×${o.v}` }))}
                  onPick={(v) => { s.setBuildingHeightMultiplier(v); track("guided_look", { setting: "building_h", value: v }); }}
                  testId="building-h"
                />
              )}
              {(scenario === "flat" || scenario === "magnet") && (
                <ChipGroup
                  label={t("flatBuildings")}
                  value={s.simpleFlatBuildings ? 1 : 0}
                  options={[{ v: 0, label: t("fbVolumetric") }, { v: 1, label: t("fbFlat") }]}
                  onPick={(v) => { s.setSimpleFlatBuildings(v === 1); track("guided_look", { setting: "flat_buildings", value: v }); }}
                  testId="flat-buildings"
                />
              )}
              {scenario === "flat" && (
                <ToggleChip
                  on={s.simpleFrame}
                  label={t("frameCompass")}
                  hint={t("frameCompassHint")}
                  onToggle={() => { s.setSimpleFrame(!s.simpleFrame); track("guided_look", { setting: "frame", value: !s.simpleFrame }); }}
                  testId="frame-toggle"
                />
              )}
            </Section>
            {/* ПЕРСОНАЛІЗАЦІЯ (v2, юзер: «немає легких доступів»): мій дім +
                напис — емоційне ядро продукту, тепер на видноті. Обидва
                контроли пишуть у ті САМІ поля стору, що й повний конструктор. */}
            <Section n={4} title={t("personalizeTitle")} hint={t("personalizeHint")}>
              {parts > 1 ? (
                <p className="text-[12.5px] leading-snug text-[var(--text-secondary)]" data-testid="personalize-panno-note">{t("personalizePannoNote")}</p>
              ) : (<>
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    aria-pressed={s.mapHighlightBuilding}
                    onClick={() => s.setMapHighlightBuilding(!s.mapHighlightBuilding)}
                    className={`inline-flex flex-1 items-center justify-center gap-2 rounded-full border px-3 py-2.5 text-[13px] font-semibold transition ${
                      s.mapHighlightBuilding
                        ? "border-[rgba(192,57,43,0.45)] bg-[rgba(192,57,43,0.1)] text-[#8f2a20]"
                        : s.highlightPoints.length > 0
                          ? "border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.1)] text-[var(--text-primary)]"
                          : "border-[var(--surface-border)] bg-white/80 text-[var(--text-primary)] hover:border-[rgba(11,92,87,0.35)]"
                    }`}
                  >
                    <Home size={15} className={s.mapHighlightBuilding ? "text-[#c0392b]" : "text-[var(--accent-strong)]"} />
                    {s.highlightPoints.length > 0
                      ? t("myHomeCount", { n: s.highlightPoints.length })
                      : t("myHome")}
                  </button>
                  {s.highlightPoints.length > 0 && (
                    <button
                      type="button"
                      onClick={clearHomeGuided}
                      aria-label={t("myHomeClear")}
                      className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-[var(--surface-border)] bg-white/80 text-[var(--text-secondary)] transition hover:text-[#8f2a20]"
                    >
                      <X size={14} />
                    </button>
                  )}
                </div>
                {/* T-3.8: підтвердження ЩО саме позначено — раніше кнопка міняла
                    напис на лічильник без пояснення, що це буде окрема деталь. */}
                {s.highlightPoints.length > 0 && (
                  <p className="text-[12px] leading-snug text-[var(--text-secondary)]">
                    {t("myHomeMarked")}{" "}
                    <button
                      type="button"
                      onClick={clearHomeGuided}
                      className="font-semibold text-[var(--text-primary)] underline underline-offset-2 hover:text-[#8f2a20]"
                    >
                      {t("myHomeClear")}
                    </button>
                  </p>
                )}
                {s.mapHighlightBuilding && s.highlightPoints.length === 0 && (
                  <p className="text-[12px] leading-snug text-[#8f2a20]">{t("myHomeHintClick")}</p>
                )}
                {/* Напис тепер підтримують ОБИДВА пайплайни (бекенд 2026-07-23:
                    піднятий напис на передній смузі обʼємної/рельєфної мапи),
                    тож чіп доступний для всіх сценаріїв. */}
                {(!labelOn ? (
                  <button
                    type="button"
                    onClick={() => { if (!s.simpleMapLabel.trim() && s.suggestedMapLabel) s.setSimpleMapLabel(s.suggestedMapLabel); setLabelOn(true); }}
                    className="inline-flex items-center justify-center gap-2 rounded-full border border-[var(--surface-border)] bg-white/80 px-3 py-2.5 text-[13px] font-semibold text-[var(--text-primary)] transition hover:border-[rgba(11,92,87,0.35)]"
                  >
                    <PenLine size={15} className="text-[var(--accent-strong)]" /> {t("addLabel")}
                  </button>
                ) : (
                  <div className="flex items-center gap-2">
                    <label className="flex min-w-0 flex-1 items-center gap-2 rounded-full border border-[var(--surface-border)] bg-white/80 px-3.5 py-2 focus-within:border-[rgba(11,92,87,0.45)]">
                      <PenLine size={14} className="shrink-0 text-[var(--accent-strong)]" />
                      <input
                        autoFocus
                        value={s.simpleMapLabel}
                        onChange={(e) => s.setSimpleMapLabel(e.target.value.slice(0, 24))}
                        maxLength={24}
                        placeholder={t("mapLabelPlaceholder")}
                        aria-label={t("mapLabelPlaceholder")}
                        className="w-full bg-transparent text-[13px] font-medium text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:outline-none"
                      />
                    </label>
                    <button
                      type="button"
                      onClick={() => { s.setSimpleMapLabel(""); setLabelOn(false); }}
                      aria-label={t("myHomeClear")}
                      className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-[var(--surface-border)] bg-white/80 text-[var(--text-secondary)] transition hover:text-[var(--text-primary)]"
                    >
                      <X size={14} />
                    </button>
                  </div>
                ))}
              </div>
                          </>)}
            </Section>
            {/* C-3: помилка з ПРИЧИНОЮ і діями. Раніше — один загальний рядок
                «Не вдалося згенерувати», хоча бекенд віддає зрозумілий текст
                (замало даних / зона завелика / сервер зайнятий). */}
            {failedNote && (
              <div className="flex flex-col gap-2 rounded-[12px] border border-red-200 bg-red-50 px-3 py-2.5" data-testid="guided-error">
                <p className="text-[12.5px] leading-snug text-red-800">{s.genError || t("genFailed")}</p>
                <div className="flex flex-wrap gap-2">
                  <Button
                    variant="bronze"
                    size="sm"
                    onClick={create}
                    data-testid="guided-retry"
                    className="rounded-full border !border-red-300 !bg-white px-3 py-1.5 text-[12px] font-semibold !text-red-800 transition hover:!bg-red-100"
                  >
                    {t("tryAgain")}
                  </Button>
                  {/zона|зона|завелик|too large|large/i.test(s.genError || "") && (
                    <Button
                      variant="bronze"
                      size="sm"
                      onClick={() => {
                        // Зменшуємо рамку навколо поточного центру до 70 % — типова
                        // причина відмови бекенду «Зона завелика для моделі N см».
                        const c = s.selectedArea?.getCenter?.();
                        if (!c) return;
                        window.dispatchEvent(new CustomEvent("monadruk:map-goto", {
                          detail: { lat: c.lat, lon: c.lng, widthM: Math.round(zoneForSizeM(s.modelSizeMm) * 0.7) },
                        }));
                      }}
                      className="rounded-full border !border-red-300 !bg-white px-3 py-1.5 text-[12px] font-semibold !text-red-800 transition hover:!bg-red-100"
                    >
                      {t("shrinkZone")}
                    </Button>
                  )}
                  <Button
                    variant="bronze"
                    size="sm"
                    onClick={() => window.dispatchEvent(new CustomEvent("monadruk:open-contact", { detail: { message: `${t("genFailed")} ${s.genError || ""}`.trim() } }))}
                    className="rounded-full !bg-transparent px-3 py-1.5 text-[12px] font-semibold !text-red-800 underline underline-offset-2"
                  >
                    {t("contactUs")}
                  </Button>
                </div>
              </div>
            )}
            {/* F-08: превʼю безкоштовне — ціна не на кнопці дії, а рядком під нею.
                A-3/A-4: кнопка активна завжди; після успіху зʼявляється лише як
                «Оновити превʼю», коли параметри змінились. */}
            {!s.isGenerating && (!successView || dirty) && (
              <>
                <Button
                  variant={successView ? "primary" : "bronze"}
                  size="lg"
                  onClick={create}
                  disabled={waitingForMap}
                  data-testid="scenario-create"
                  className="w-full"
                >
                  {successView ? t("updateModel") : ctaVariant === "B" ? t("previewCtaB") : t("previewCta")}
                </Button>
                {waitingForMap && (
                  <p className="mt-1.5 text-center text-[12px] font-semibold text-[var(--accent-strong)]" aria-live="polite" data-testid="map-loading-wait">
                    {t("mapLoadingWait")}
                  </p>
                )}
                {!successView && (
                  <p className="mt-1.5 text-center text-[12px] font-semibold text-[var(--text-secondary)]">
                    {t("printFromLine", { price: disp(ctaPriceUah) })}
                  </p>
                )}
                {/* ⭐09.09.2026, заміряно на 30 днях: з 23 людей, які СТВОРИЛИ
                    модель, 15 — не з України (FR/ES/DE/AT/IT/MX/CH), і десятеро
                    користувались англійським інтерфейсом. Друк і доставка — лише
                    по Україні, але дізнавались вони про це аж на екрані «готово»,
                    змарнувавши 1–4 хвилини генерації (закордонні — найповільніші,
                    бо йдуть через Overpass) і місце в черзі. Кажемо правду ДО
                    кнопки: що вони отримають напевно — файл. */}
                {!successView && locale !== "uk" && (
                  <p
                    className="mt-1.5 text-center text-[11.5px] leading-snug text-[var(--text-secondary)]"
                    data-testid="ua-only-early"
                  >
                    {t("uaOnly")}
                  </p>
                )}
              </>
            )}
            {/* A-6: єдиний вихід у розширений режим (стан зони/формату/розміру
                зберігається — юзер продовжує там же). */}
            <Button
              variant="ghost"
              size="md"
              onClick={() => exitGuided("step2")}
              className="mt-2 w-full text-center"
            >
              {t("advancedSettings")}
            </Button>
          </div>
        )}
      </div>
      {/* F-04: на мобільному ціна + головна дія стану завжди внизу екрана (портал). */}
      <GuidedStickyBar
        visible={displayStep === 2}
        label={scenario === "magnet" ? t("magnetTitle") : sizeLabel(s.modelSizeMm)}
        price={disp(ctaPriceUah)}
        busy={generatingView}
        tone={successView && !dirty ? "bronze" : "primary"}
        disabled={!generatingView && waitingForMap}
        cta={generatingView
          ? `${Math.max(0, Math.min(100, s.progress || 0))}%`
          : successView ? (dirty ? t("updateModel") : t("orderPrint")) : (ctaVariant === "B" ? t("previewCtaShortB") : t("previewCtaShort"))}
        onCta={() => {
          if (successView && !dirty) window.dispatchEvent(new Event("monadruk:open-order"));
          else create();
        }}
      />
    </div>
  );
}
