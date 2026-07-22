"use client";

import { useEffect, useRef, useState } from "react";
import { useTranslations, useLocale } from "next-intl";
import { ArrowLeft, Check, Home, KeyRound, Loader2, MapPin, PenLine, ShoppingBag, Sliders, X } from "lucide-react";
import { MapSearchBox } from "@/components/MapSearchBox";
// ЛОКАЛІЗОВАНИЙ Link (@/i18n/navigation), НЕ next/link — інакше лінк на
// /keychains з /en/create губив би префікс локалі.
import { Link } from "@/i18n/navigation";
import { useShallow } from "zustand/react/shallow";
import { useGenerationStore } from "@/store/generation-store";
import { SIMPLE_SIZES } from "@/lib/generation";
import { fetchQuote, type Quote } from "@/lib/pricing";
import {
  KEYCHAIN_PRICE_UAH,
  MAP_MAGNET_PRICE_UAH,
  MAP_RELIEF_ADDON_UAH,
  mapPriceEur,
} from "@/lib/mapPrices";

/** Авто-зона навколо точки пошуку: ~800×800 м — «good detail» для мапи 8 см. */
const GUIDED_ZONE_M = 800;

/** Сценарії, що лишаються всередині guided-флоу (брелок = лінк, повний = вихід). */
type ScenarioId = "map3d" | "relief" | "flat" | "magnet";

/**
 * СЦЕНАРНИЙ ВХІД /create (guided-режим). UX-аудит: новий користувач бачив ~43
 * інтерактивні контроли до першої генерації. Тут — ДВА кроки: ЩО створюємо →
 * ДЕ місце (пошук на карті, зона стає сама) + розмір і ОДНА кнопка «Створити
 * модель · ціна» на тому ж екрані (нуль зайвих кліків, без окремого кроку).
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
  const locale = useLocale();
  // Діаспора (не-uk) бачить € за тим самим позиційним курсом, що й решта сайту.
  const isEu = locale !== "uk";
  const disp = (uah: number) => (isEu ? `€${mapPriceEur(uah)}` : `${uah} ₴`);

  const s = useGenerationStore(useShallow((st) => ({
    selectedArea: st.selectedArea,
    isGenerating: st.isGenerating,
    progress: st.progress,
    status: st.status,
    downloadUrl: st.downloadUrl,
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
    simpleConnector: st.simpleConnector,
    setSimpleConnector: st.setSimpleConnector,
  })));

  const [scenario, setScenario] = useState<ScenarioId | null>(null);
  // started: генерацію запущено САМЕ з guided-флоу (відрізняємо від відновленої
  // з localStorage задачі минулої сесії — для неї success-екран не форсуємо).
  const [started, setStarted] = useState(false);
  // Напис на мапі — ОПЦІЙНИЙ (v3, юзер: «текст не по дефолту, а коли включаю»):
  // поле зʼявляється лише після кліку «Додати напис»; вибір сценарію чистить його.
  const [labelOn, setLabelOn] = useState(false);

  // v3.1 (юзер: «не можна пересувати рамку, коли обрання будинку увімкнене»):
  // режим кліку АВТО-ВИМИКАЄТЬСЯ одразу після вибору будинку — вибір лишається
  // (друк дивиться на highlightPoints, не на прапор), а рамка знову рухома.
  // Кнопка «Мій дім» повторним кліком повертає режим (додати ще/змінити).
  const hlCountRef = useRef(0);
  useEffect(() => {
    if (s.mapHighlightBuilding && s.highlightPoints.length > hlCountRef.current) {
      s.setMapHighlightBuilding(false);
    }
    hlCountRef.current = s.highlightPoints.length;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.highlightPoints.length, s.mapHighlightBuilding]);

  // ЖИВА ЦІНА на CTA: ТОЙ САМИЙ quote-механізм, що в SimpleControlPanel /
  // StickyActionBar (fetchQuote з бекенд-прайсу; fallback нижче — mapPrices.ts).
  const [quote, setQuote] = useState<Quote | null>(null);
  useEffect(() => {
    if (!scenario) return;
    let alive = true;
    const magnet = scenario === "magnet";
    fetchQuote("map", magnet ? 60 : s.modelSizeMm, magnet ? false : scenario === "relief")
      .then((q) => { if (alive) setQuote(q); });
    return () => { alive = false; };
  }, [scenario, s.modelSizeMm]);

  // АВТО-ЗОНА після пошуку. MapSearchBox шле `monadruk:map-goto` {lat,lon,label};
  // KeychainCropOverlay у MapSelector на це ВЖЕ переносить зону, але зберігає її
  // поточний розмір. Робимо так само, як pickTemplate у SimpleControlPanel
  // (клік по готовому району): повторний диспатч тієї ж події з ЯВНИМ widthM —
  // handler ставить зону рівно ~800×800 м навколо точки. Guard: події, що вже
  // несуть widthM або centerOnly (гео-при-вході), ігноруємо — інакше цикл.
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
        window.dispatchEvent(new CustomEvent("monadruk:map-goto", {
          detail: { lat: d.lat, lon: d.lon, widthM: GUIDED_ZONE_M },
        }));
      }, 120);
    };
    window.addEventListener("monadruk:map-goto", onGoto as EventListener);
    return () => window.removeEventListener("monadruk:map-goto", onGoto as EventListener);
  }, []);

  // Вибір сценарію = пресет формату у store (той самий шлях, що сегмент-контрол
  // «Формат» у SimpleControlPanel) + одразу крок 2. M (80 мм) — передвибраний.
  const pick = (id: ScenarioId) => {
    s.setShowHexGrid(false);
    try { localStorage.setItem("3dmap_hex_grid", "0"); } catch { /* ignore */ }
    s.setPreviewMode(true);
    s.setSimpleFormat(id === "magnet" ? "magnet" : id === "flat" ? "flat" : "relief3d");
    if (id === "flat") s.setSimpleFlatAms(true);
    // ПІСЛЯ setSimpleFormat: relief3d зберігає попередній simpleRelief — явно
    // ставимо потрібне значення (map3d = без рельєфу, relief = з рельєфом).
    s.setSimpleRelief(id === "relief");
    if (id !== "magnet") s.setModelSizeMm(80);
    s.setSimpleMapLabel("");
    setLabelOn(false);
    setScenario(id);
  };

  const create = () => {
    if (!s.selectedArea || s.isGenerating) return;
    setStarted(true);
    window.dispatchEvent(new Event("monadruk:guided-generate"));
  };

  const basePrice = SIMPLE_SIZES[0].price;
  const reliefAddon = scenario === "relief" ? MAP_RELIEF_ADDON_UAH : 0;
  // Ціна на CTA: живий quote; fallback — канонічна таблиця mapPrices (без хардкоду).
  const fallbackSize = SIMPLE_SIZES.reduce(
    (best, z) => (Math.abs(z.mm - s.modelSizeMm) < Math.abs(best.mm - s.modelSizeMm) ? z : best),
    SIMPLE_SIZES[0],
  );
  const ctaPriceUah = scenario === "magnet"
    ? (quote?.price ?? MAP_MAGNET_PRICE_UAH)
    : (quote?.price ?? fallbackSize.price + reliefAddon);

  const cards: Array<{
    id: ScenarioId;
    img: string;
    title: string;
    desc: string;
    price: string;
  }> = [
    { id: "map3d", img: "real-1", title: t("map3dTitle"), desc: t("map3dDesc"), price: t("from", { price: disp(basePrice) }) },
    { id: "relief", img: "real-2", title: t("reliefTitle"), desc: t("reliefDesc"), price: t("from", { price: disp(basePrice + MAP_RELIEF_ADDON_UAH) }) },
    { id: "flat", img: "real-8", title: t("flatTitle"), desc: t("flatDesc"), price: t("from", { price: disp(basePrice) }) },
    { id: "magnet", img: "real-7", title: t("magnetTitle"), desc: t("magnetDesc"), price: disp(MAP_MAGNET_PRICE_UAH) },
  ];

  const generatingView = s.isGenerating;
  const successView = started && !s.isGenerating && !!s.downloadUrl;
  const failedNote = started && !s.isGenerating && !s.downloadUrl;
  const displayStep = generatingView || successView || scenario !== null ? 2 : 1;

  const cardBtnCls = "group flex flex-col overflow-hidden rounded-[18px] border border-[var(--surface-border)] bg-white/80 text-left shadow-[0_4px_14px_rgba(15,23,42,0.05)] transition hover:border-[rgba(11,92,87,0.45)] hover:shadow-[0_8px_24px_rgba(15,23,42,0.1)]";

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-[30px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_22px_70px_rgba(15,23,42,0.08)] backdrop-blur" data-testid="scenario-flow">
      {/* Шапка: степ-індикатор + назад до сценаріїв */}
      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-[var(--surface-border)] px-4 py-3">
        <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
          {successView ? t("readyBadge") : t("stepOf", { step: displayStep })}
        </span>
        {scenario !== null && !generatingView && (
          <button
            type="button"
            onClick={() => { setScenario(null); setStarted(false); }}
            className="inline-flex items-center gap-1 rounded-full border border-[var(--surface-border)] bg-white/80 px-2.5 py-1 text-[11px] font-semibold text-[var(--text-secondary)] transition hover:border-[rgba(11,92,87,0.4)] hover:text-[var(--text-primary)]"
          >
            <ArrowLeft size={12} /> {t("back")}
          </button>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {scenario === null ? (
          /* ── КРОК 1: ЩО СТВОРЮЄМО? ── */
          <div>
            <h2 className="font-title text-lg font-semibold text-[var(--text-primary)]">{t("step1Title")}</h2>
            <div className="mt-3 grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-2">
              {cards.map((c) => (
                <button key={c.id} type="button" onClick={() => pick(c.id)} className={cardBtnCls}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`/showcase/${c.img}.webp`}
                    alt={c.title}
                    loading="lazy"
                    className="aspect-[4/3] w-full object-cover transition duration-500 group-hover:scale-[1.04]"
                  />
                  <span className="flex flex-1 flex-col gap-0.5 px-2.5 py-2">
                    <span className="text-[13px] font-semibold leading-tight text-[var(--text-primary)]">{c.title}</span>
                    <span className="text-[12px] font-semibold text-[var(--accent-strong)]">{c.price}</span>
                    <span className="text-[11px] leading-snug text-[var(--text-secondary)]">{c.desc}</span>
                  </span>
                </button>
              ))}
              {/* Брелок — ОКРЕМИЙ конструктор: просто лінк, нічого не пресетимо. */}
              <Link href="/keychains" className={cardBtnCls}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src="/showcase/real-6.webp"
                  alt={t("keychainTitle")}
                  loading="lazy"
                  className="aspect-[4/3] w-full object-cover transition duration-500 group-hover:scale-[1.04]"
                />
                <span className="flex flex-1 flex-col gap-0.5 px-2.5 py-2">
                  <span className="inline-flex items-center gap-1 text-[13px] font-semibold leading-tight text-[var(--text-primary)]"><KeyRound size={12} /> {t("keychainTitle")}</span>
                  <span className="text-[12px] font-semibold text-[var(--accent-strong)]">{t("from", { price: disp(KEYCHAIN_PRICE_UAH) })}</span>
                  <span className="text-[11px] leading-snug text-[var(--text-secondary)]">{t("keychainDesc")}</span>
                </span>
              </Link>
              {/* Повний конструктор — вихід із guided (усе як раніше). */}
              <button type="button" onClick={onExitGuided} className={cardBtnCls} data-testid="scenario-full">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src="/showcase/real-9.webp"
                  alt={t("fullTitle")}
                  loading="lazy"
                  className="aspect-[4/3] w-full object-cover transition duration-500 group-hover:scale-[1.04]"
                />
                <span className="flex flex-1 flex-col gap-0.5 px-2.5 py-2">
                  <span className="inline-flex items-center gap-1 text-[13px] font-semibold leading-tight text-[var(--text-primary)]"><Sliders size={12} /> {t("fullTitle")}</span>
                  <span className="text-[11px] leading-snug text-[var(--text-secondary)]">{t("fullDesc")}</span>
                </span>
              </button>
            </div>
          </div>
        ) : (
          /* ── КРОК 2: ДЕ ВАШЕ МІСЦЕ? + розмір і CTA на тому ж екрані ──
              (карта лишається видимою поруч/вище; рамка зони — інтерактивна) */
          <div className="flex flex-col gap-3">
            {/* ГОТОВО-банер (v3): модель є — головна дія зверху, а ВСІ контролі
                НИЖЧЕ лишаються живими: міняй місце/дім/напис/розмір і одразу
                «Оновити модель». Превʼю крутиться на сцені поруч — один екран. */}
            {successView && (
              <div className="flex flex-col gap-2.5">
                <div className="flex items-center gap-2 text-[16px] font-semibold text-[var(--text-primary)]">
                  <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-[var(--accent-strong)] text-white"><Check size={15} /></span>
                  {t("readyTitle")}
                </div>
                <p className="text-[12.5px] leading-snug text-[var(--text-secondary)]">{t("readyHint")}</p>
                <button
                  type="button"
                  onClick={() => window.dispatchEvent(new Event("monadruk:open-order"))}
                  className="inline-flex w-full items-center justify-center gap-2 rounded-full bg-[var(--bronze,#8E6B3D)] px-6 py-3.5 text-[15px] font-semibold text-white shadow-[0_8px_24px_rgba(142,107,61,0.35)] transition hover:brightness-110"
                >
                  <ShoppingBag size={18} /> {t("orderPrint")} · {disp(ctaPriceUah)}
                </button>
                <div className="flex items-center justify-center gap-4">
                  <button type="button" onClick={onExitGuided} className="text-[12px] font-semibold text-[var(--text-secondary)] underline-offset-2 transition hover:text-[var(--text-primary)] hover:underline">
                    {t("tuneDetails")}
                  </button>
                  <button type="button" onClick={() => { setStarted(false); setScenario(null); }} className="text-[12px] font-semibold text-[var(--text-secondary)] underline-offset-2 transition hover:text-[var(--text-primary)] hover:underline">
                    {t("createAnother")}
                  </button>
                </div>
                <div className="my-0.5 flex items-center gap-2">
                  <span className="h-px flex-1 bg-[var(--surface-border)]" />
                  <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">{t("changeSomething")}</span>
                  <span className="h-px flex-1 bg-[var(--surface-border)]" />
                </div>
              </div>
            )}
            {/* Генерація (v3): компактний прогрес над контролями — панель жива. */}
            {generatingView && (
              <div className="flex flex-col gap-2 rounded-[16px] border border-[var(--surface-border)] bg-white/70 px-3.5 py-3">
                <div className="flex items-center gap-2 text-[14px] font-semibold text-[var(--text-primary)]">
                  <Loader2 size={16} className="animate-spin text-[var(--accent-strong)]" /> {t("generating")}
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-[rgba(15,23,42,0.08)]">
                  <div className="h-full rounded-full bg-[var(--accent-strong)] transition-all duration-700" style={{ width: `${Math.max(4, Math.min(100, s.progress || 0))}%` }} />
                </div>
                <p className="text-[12px] text-[var(--text-secondary)]" aria-live="polite">
                  {s.progress}%{s.status ? ` · ${s.status}` : ""} · {t("etaNote")}
                </p>
              </div>
            )}
            <h2 className="font-title text-lg font-semibold text-[var(--text-primary)]">{t("step2Title")}</h2>
            {/* ПОШУК ПРЯМО В ПАНЕЛІ (v2): раніше поле жило лише на карті, а панель
                давала довгу інструкцію «йдіть шукайте там» — погляд стрибав. Тепер
                друкуєш адресу тут; та сама подія monadruk:map-goto → автозона. */}
            <div className="rounded-full border border-[var(--surface-border)] bg-white/80 px-1.5 py-0.5 focus-within:border-[rgba(11,92,87,0.45)]">
              <MapSearchBox variant="panel" />
            </div>
            {!s.selectedArea ? (
              <div className="inline-flex items-center gap-2 rounded-full border border-[var(--surface-border)] bg-white/80 px-3.5 py-2 text-[13px] font-medium text-[var(--text-secondary)]">
                <MapPin size={15} className="animate-pulse text-[var(--accent-strong)]" /> {t("waitingPlace")}
              </div>
            ) : (
              <>
                <div className="inline-flex items-center gap-2 self-start rounded-full border border-[rgba(11,92,87,0.35)] bg-[rgba(15,118,110,0.1)] px-3.5 py-2 text-[13px] font-semibold text-[var(--text-primary)]">
                  <Check size={15} className="text-[var(--accent-strong)]" /> {t("placeChosen")}
                </div>
                {/* ПЕРСОНАЛІЗАЦІЯ (v2, юзер: «немає легких доступів»): мій дім +
                    напис — емоційне ядро продукту, тепер на видноті. Обидва
                    контроли пишуть у ті САМІ поля стору, що й повний конструктор. */}
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">{t("personalizeTitle")}</p>
                  <div className="mt-2 flex flex-col gap-2">
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
                        <Home size={15} className={s.mapHighlightBuilding ? "text-[#c0392b]" : s.highlightPoints.length > 0 ? "text-[var(--accent-strong)]" : "text-[var(--accent-strong)]"} />
                        {s.highlightPoints.length > 0
                          ? t("myHomeCount", { n: s.highlightPoints.length })
                          : t("myHome")}
                      </button>
                      {s.highlightPoints.length > 0 && (
                        <button
                          type="button"
                          onClick={() => s.clearHighlights()}
                          aria-label={t("myHomeClear")}
                          className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-[var(--surface-border)] bg-white/80 text-[var(--text-secondary)] transition hover:text-[#8f2a20]"
                        >
                          <X size={14} />
                        </button>
                      )}
                    </div>
                    {s.mapHighlightBuilding && s.highlightPoints.length === 0 && (
                      <p className="text-[12px] leading-snug text-[#8f2a20]">{t("myHomeHintClick")}</p>
                    )}
                    {/* Напис підтримує flat_plate-пайплайн: плоска/магніт завжди,
                        а обʼємна — коли є дім-вставка (вона сама форсує flat). */}
                    {(scenario === "flat" || scenario === "magnet" || s.highlightPoints.length > 0) && (!labelOn ? (
                      <button
                        type="button"
                        onClick={() => setLabelOn(true)}
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
                    {/* Зʼєднувачі (юзер: «немає щоб включити зʼєднувачі»): пази по
                        краях плитки — потім можна доклеїти сусідні. Магніт малий —
                        не показуємо. Той самий store-прапор, що в повній панелі. */}
                    {scenario !== "magnet" && (
                      <button
                        type="button"
                        aria-pressed={s.simpleConnector}
                        onClick={() => s.setSimpleConnector(!s.simpleConnector)}
                        className={`inline-flex items-center justify-center gap-2 rounded-full border px-3 py-2.5 text-[13px] font-semibold transition ${
                          s.simpleConnector
                            ? "border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.1)] text-[var(--text-primary)]"
                            : "border-[var(--surface-border)] bg-white/80 text-[var(--text-primary)] hover:border-[rgba(11,92,87,0.35)]"
                        }`}
                      >
                        <span aria-hidden>{s.simpleConnector ? "✓" : ""}</span> {t("connectors")}
                      </button>
                    )}
                    {scenario !== "magnet" && s.simpleConnector && (
                      <p className="text-[12px] leading-snug text-[var(--text-secondary)]">{t("connectorsHint")}</p>
                    )}
                  </div>
                </div>
                {/* Розмір: одразу тут (без окремого кроку). Магніт — фіксований. */}
                {scenario === "magnet" ? (
                  <p className="text-[13px] leading-relaxed text-[var(--text-secondary)]">
                    {t("magnetFixedNote", { price: disp(quote?.price ?? MAP_MAGNET_PRICE_UAH) })}
                  </p>
                ) : (
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">{t("sizeTitle")}</p>
                    <div className="mt-2 grid grid-cols-2 gap-2" role="radiogroup" aria-label={t("sizeTitle")}>
                      {SIMPLE_SIZES.map((z) => (
                        <button
                          key={z.key}
                          type="button"
                          role="radio"
                          aria-checked={s.modelSizeMm === z.mm}
                          onClick={() => s.setModelSizeMm(z.mm)}
                          className={`flex min-h-[64px] flex-col items-center justify-center gap-0.5 rounded-[16px] border px-2 py-2 transition ${
                            s.modelSizeMm === z.mm
                              ? "border-[rgba(11,92,87,0.5)] bg-[rgba(15,118,110,0.12)]"
                              : "border-[var(--surface-border)] bg-white/80 hover:border-[rgba(11,92,87,0.3)]"
                          }`}
                        >
                          <span className="text-[15px] font-bold text-[var(--text-primary)]">{z.label} · {z.cm}</span>
                          <span className="text-[13px] font-semibold text-[var(--accent-strong)]">{disp(z.price + reliefAddon)}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                {failedNote && (
                  <p className="rounded-[12px] border border-red-200 bg-red-50 px-3 py-2 text-[12.5px] text-red-700">
                    {t("genFailed")}
                  </p>
                )}
                {!s.isGenerating && (
                <button
                  type="button"
                  onClick={create}
                  data-testid="scenario-create"
                  className="inline-flex w-full items-center justify-center gap-2 rounded-full bg-[var(--bronze,#8E6B3D)] px-6 py-4 text-[16px] font-semibold text-white shadow-[0_8px_24px_rgba(142,107,61,0.35)] transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {successView ? t("updateModel") : t("createModel")} · {disp(ctaPriceUah)}
                </button>
                )}
                {/* Вихід у повний конструктор ПРЯМО з кроку 2 (обіцяно в макеті):
                    стан (зона/формат/розмір) зберігається — юзер продовжує там же. */}
                <button
                  type="button"
                  onClick={onExitGuided}
                  className="mt-2 w-full text-center text-[12px] text-[var(--text-secondary)] underline underline-offset-2 hover:text-[var(--text-primary)]"
                >
                  {t("advancedSettings")}
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
