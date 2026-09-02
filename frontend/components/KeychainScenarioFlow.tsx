"use client";

import { useEffect, useRef, useState } from "react";
import { GuidedStickyBar } from "@/components/GuidedStickyBar";
import { useTranslations, useLocale } from "next-intl";
import { ArrowLeft, Check, Home, Loader2, MapPin, PenLine, ShoppingBag, Sliders, X } from "lucide-react";
import { MapSearchBox } from "@/components/MapSearchBox";
import { useShallow } from "zustand/react/shallow";
import { useGenerationStore } from "@/store/generation-store";
import { fetchQuote, type Quote } from "@/lib/pricing";
import { KEYCHAIN_PRICE_UAH, mapPriceEur } from "@/lib/mapPrices";
import {
  KEYCHAIN_TEMPLATES,
  type KeychainDesignerConfig,
} from "@/components/KeychainDesigner";

/** Швидкі міста: 1 тап замість друкування адреси (та сама подія, що й пошук).
 *  text — авто-напис для гравіювання (як defaultText у повній панелі). */
const QUICK_CITIES: Array<{ uk: string; en: string; text: string; lat: number; lon: number }> = [
  { uk: "Київ", en: "Kyiv", text: "KYIV", lat: 50.4501, lon: 30.5234 },
  { uk: "Львів", en: "Lviv", text: "LVIV", lat: 49.8419, lon: 24.0315 },
  { uk: "Одеса", en: "Odesa", text: "ODESA", lat: 46.4825, lon: 30.7233 },
  { uk: "Харків", en: "Kharkiv", text: "KHARKIV", lat: 49.9935, lon: 36.2304 },
];

/** Картки кроку 1: найпопулярніші шаблони з KEYCHAIN_TEMPLATES + фото друків.
 *  autoLabel:false — шаблони, де смуга напису у вузькій частині форми
 *  (половинка серця): авто-LVIV там зрізається до крихти (перевірено на
 *  реальній генерації — Text 0.8мм³ проти 17мм³ у серця). Для них напис
 *  лишаємо порожнім/ручним — користувач бачить розміщення в живому макеті. */
const CARD_DEFS: Array<{ tplId: string; img: string; titleKey: string; descKey: string; autoLabel?: boolean }> = [
  { tplId: "heart-46", img: "card-kc-heart", titleKey: "cardHeartTitle", descKey: "cardHeartDesc" },
  { tplId: "heart-pair-left", img: "card-kc-heartpair", titleKey: "cardHeartPairTitle", descKey: "cardHeartPairDesc", autoLabel: false },
  { tplId: "token-55", img: "card-kc-token", titleKey: "cardTokenTitle", descKey: "cardTokenDesc" },
  { tplId: "classic-wide", img: "card-kc-rect", titleKey: "cardRectTitle", descKey: "cardRectDesc" },
];

/**
 * СЦЕНАРНИЙ ВХІД /keychains (guided-режим) — той самий патерн, що ScenarioFlow
 * на /create: ДВА кроки (який брелок → де місце + напис) і ОДНА CTA з ціною
 * замість важкої панелі з десятками контролів. SVG-макет (KeychainDesigner),
 * карта і 3D-превʼю лишаються видимими поруч — guided це ШАР ПОВЕРХ.
 *
 * Компонент НЕ дублює логіку генерації: кнопка шле window-подію
 * `monadruk:kc-guided-generate`, яку слухає прихована «машинна» копія
 * KeychainControlPanel (проп listenGuidedGenerate) і викликає той САМИЙ
 * handleGenerate, що й кнопка «Створити 3MF». Замовлення/завантаження — теж
 * події (`monadruk:kc-guided-order` / `monadruk:kc-guided-download`), бо
 * OrderDialog і квота-гейт живуть у панелі (портал у body — видимі попри
 * display:none обгортку). Прогрес/готовність читаємо зі спільного стора.
 *
 * Шаблон картки застосовується ТИМ САМИМ кодом, що й повна панель: сторінка
 * передає свій applyTemplate (скидання повороту + setDesign), а значення
 * беремо З KEYCHAIN_TEMPLATES — нуль хардкоду розмірів.
 */
export function KeychainScenarioFlow({
  onExitGuided,
  onApplyTemplate,
  label,
  onLabelChange,
  backLabel,
  onBackLabelChange,
  placeMarker,
  onPlaceMarkerChange,
}: {
  onExitGuided: () => void;
  /** applyTemplate зі сторінки — той самий шлях, що клік по шаблону в повному UI. */
  onApplyTemplate: (design: KeychainDesignerConfig) => void;
  label: string;
  onLabelChange: (value: string) => void;
  backLabel: string;
  onBackLabelChange: (value: string) => void;
  placeMarker: "" | "heart" | "star" | "circle";
  onPlaceMarkerChange: (value: "" | "heart" | "star" | "circle") => void;
}) {
  const t = useTranslations("kcScenario");
  const locale = useLocale();
  // Діаспора (не-uk) бачить € за тим самим позиційним курсом, що й решта сайту.
  const isEu = locale !== "uk";
  const disp = (uah: number) => (isEu ? `€${mapPriceEur(uah)}` : `${uah} ₴`);
  const tOrder = useTranslations("order");

  const s = useGenerationStore(useShallow((st) => ({
    selectedArea: st.selectedArea,
    isGenerating: st.isGenerating,
    progress: st.progress,
    status: st.status,
    downloadUrl: st.downloadUrl,
    // Виділення будинку «мій дім» — той самий store, що на /create; машинна
    // копія KeychainControlPanel читає його і шле keychain_highlight_building.
    mapHighlightBuilding: st.mapHighlightBuilding,
    highlightPoints: st.highlightPoints,
    setMapHighlightBuilding: st.setMapHighlightBuilding,
    clearHighlights: st.clearHighlights,
  })));

  // Режим кліку АВТО-ВИМИКАЄТЬСЯ одразу після вибору будинку (як на /create):
  // вибір лишається (друк дивиться на highlightPoints), а рамка знову рухома.
  const hlCountRef = useRef(0);
  useEffect(() => {
    if (s.mapHighlightBuilding && s.highlightPoints.length > hlCountRef.current) {
      s.setMapHighlightBuilding(false);
    }
    hlCountRef.current = s.highlightPoints.length;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.highlightPoints.length, s.mapHighlightBuilding]);

  const [tplId, setTplId] = useState<string | null>(null);
  // Чи редагував користувач напис ВЛАСНОРУЧ: якщо ні — чіп міста оновлює його
  // авто-текстом (LVIV тощо), як робить селектор міста в повній панелі; якщо
  // так — ніколи не затираємо користувацький текст.
  const labelManualRef = useRef(false);
  // started: генерацію запущено САМЕ з guided-флоу (відрізняємо від відновленої
  // з localStorage задачі минулої сесії — для неї success-екран не форсуємо).
  const [started, setStarted] = useState(false);
  // Зворотний напис — ОПЦІЙНИЙ чіп: поле зʼявляється лише після кліку.
  const [backOn, setBackOn] = useState(false);

  // ЖИВА ЦІНА на CTA: той самий quote-механізм, що в KeychainControlPanel
  // (fetchQuote("keychain") з бекенд-прайсу; fallback — KEYCHAIN_PRICE_UAH).
  const [quote, setQuote] = useState<Quote | null>(null);
  useEffect(() => {
    let alive = true;
    fetchQuote("keychain").then((q) => { if (alive) setQuote(q); });
    return () => { alive = false; };
  }, []);
  const priceUah = quote?.price ?? KEYCHAIN_PRICE_UAH;

  // Чи дозволений авто-напис для поточного шаблону (див. CARD_DEFS.autoLabel).
  const autoLabelRef = useRef(true);

  // ЧЕСНИЙ «Місце обрано»: дефолтна рамка (Київ) ставиться сама, тож
  // selectedArea != null ≠ вибір користувача. Без гейта: набрав адресу, не
  // клікнув підказку → зелений бейдж + активна CTA → мовчки брелок Києва.
  // Вибір = подія пошуку/чіпа (map-goto) АБО ручний зсув рамки (зміна
  // selectedArea після вже наявної; перший авто-сет не рахується).
  const [placePicked, setPlacePicked] = useState(false);
  const prevAreaRef = useRef<typeof s.selectedArea>(null);
  useEffect(() => {
    const prev = prevAreaRef.current;
    prevAreaRef.current = s.selectedArea;
    if (prev && s.selectedArea && s.selectedArea !== prev) setPlacePicked(true);
  }, [s.selectedArea]);
  useEffect(() => {
    const onPick = (e: Event) => {
      const d = (e as CustomEvent).detail as
        | { lat?: number; lon?: number; centerOnly?: boolean }
        | undefined;
      if (!d || d.centerOnly) return;
      if (!Number.isFinite(d.lat) || !Number.isFinite(d.lon)) return;
      setPlacePicked(true);
      // ПОШУК нового місця робить старий авто-напис (KYIV…) брехнею —
      // чистимо його, якщо користувач не вводив свій. Чіпи міст не постраждають:
      // їх onClick ставить правильний текст СИНХРОННО після цього ж dispatch
      // (Париж більше не гравіюється як «KYIV» — відтворено наскрізним тестом).
      if (!labelManualRef.current) onLabelChange("");
    };
    window.addEventListener("monadruk:map-goto", onPick as EventListener);
    return () => window.removeEventListener("monadruk:map-goto", onPick as EventListener);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // Вибір картки = застосувати шаблон тим САМИМ кодом, що повна панель
  // (page.applyTemplate), значення — з KEYCHAIN_TEMPLATES.
  const pick = (id: string) => {
    const tpl = KEYCHAIN_TEMPLATES.find((k) => k.id === id);
    if (!tpl) return;
    // Guided-воронка: яку картку-шаблон обирають.
    import("@/lib/analytics").then((m) => m.track("guided_pick", { product: "keychain", scenario: id })).catch(() => {});
    onApplyTemplate(tpl.design);
    const def = CARD_DEFS.find((c) => c.tplId === id);
    autoLabelRef.current = def?.autoLabel !== false;
    // Для шаблонів без авто-напису чистимо дефолтний KYIV — інакше він
    // мовчки гравіюється у вузькій частині форми (зрізаний до уламка).
    if (!autoLabelRef.current && !labelManualRef.current) onLabelChange("");
    setTplId(id);
  };

  const create = () => {
    if (!s.selectedArea || !placePicked || s.isGenerating) return;
    import("@/lib/analytics")
      .then((m) => m.track("guided_generate", { product: "keychain", scenario: tplId }))
      .catch(() => {});
    setRan(false);
    setStarted(true);
    window.dispatchEvent(new Event("monadruk:kc-guided-generate"));
  };

  // F-10: помилку показуємо лише після РЕАЛЬНОГО запуску генерації (див. ScenarioFlow).
  const [ran, setRan] = useState(false);
  useEffect(() => { if (s.isGenerating) setRan(true); }, [s.isGenerating]);
  const generatingView = s.isGenerating;
  const successView = started && !s.isGenerating && !!s.downloadUrl;
  const failedNote = started && ran && !s.isGenerating && !s.downloadUrl;
  const displayStep = generatingView || successView || tplId !== null ? 2 : 1;

  const cardBtnCls = "group flex flex-col overflow-hidden rounded-[18px] border border-[var(--surface-border)] bg-white/80 text-left shadow-[0_4px_14px_rgba(15,23,42,0.05)] transition hover:border-[rgba(11,92,87,0.45)] hover:shadow-[0_8px_24px_rgba(15,23,42,0.1)]";
  const chipOnCls = "border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.1)] text-[var(--text-primary)]";
  const chipOffCls = "border-[var(--surface-border)] bg-white/80 text-[var(--text-primary)] hover:border-[rgba(11,92,87,0.35)]";

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-[30px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_22px_70px_rgba(15,23,42,0.08)] backdrop-blur" data-testid="kc-scenario-flow">
      {/* Шапка: степ-індикатор + назад до карток */}
      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-[var(--surface-border)] px-4 py-3">
        <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
          {successView ? t("readyBadge") : t("stepOf", { step: displayStep })}
        </span>
        {tplId !== null && !generatingView && (
          <button
            type="button"
            onClick={() => { setTplId(null); setStarted(false); }}
            className="inline-flex items-center gap-1 rounded-full border border-[var(--surface-border)] bg-white/80 px-2.5 py-1 text-[11px] font-semibold text-[var(--text-secondary)] transition hover:border-[rgba(11,92,87,0.4)] hover:text-[var(--text-primary)]"
          >
            <ArrowLeft size={12} /> {t("back")}
          </button>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {tplId === null && !generatingView && !successView ? (
          /* ── КРОК 1: ЯКИЙ БРЕЛОК? ── */
          <div>
            <h2 className="font-title text-lg font-semibold text-[var(--text-primary)]">{t("step1Title")}</h2>
            <div className="mt-3 grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-2">
              {CARD_DEFS.map((c) => (
                <button key={c.tplId} type="button" onClick={() => pick(c.tplId)} className={cardBtnCls} data-testid={`kc-scenario-${c.tplId}`}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`/showcase/${c.img}.webp`}
                    alt={t(c.titleKey)}
                    loading="lazy"
                    className="aspect-[4/3] w-full object-cover transition duration-500 group-hover:scale-[1.04]"
                  />
                  <span className="flex flex-1 flex-col gap-0.5 px-2.5 py-2">
                    <span className="text-[13px] font-semibold leading-tight text-[var(--text-primary)]">{t(c.titleKey)}</span>
                    <span className="text-[12px] font-semibold text-[var(--accent-strong)]">{t("from", { price: disp(priceUah) })}</span>
                    <span className="text-[11px] leading-snug text-[var(--text-secondary)]">{t(c.descKey)}</span>
                  </span>
                </button>
              ))}
              {/* Повний дизайнер — вихід із guided (усі форми, повзунки, GPX, топо). */}
              <button type="button" onClick={onExitGuided} className={cardBtnCls} data-testid="kc-scenario-full">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src="/showcase/card-kc-full.webp"
                  alt={t("cardFullTitle")}
                  loading="lazy"
                  className="aspect-[4/3] w-full object-cover transition duration-500 group-hover:scale-[1.04]"
                />
                <span className="flex flex-1 flex-col gap-0.5 px-2.5 py-2">
                  <span className="inline-flex items-center gap-1 text-[13px] font-semibold leading-tight text-[var(--text-primary)]"><Sliders size={12} /> {t("cardFullTitle")}</span>
                  <span className="text-[11px] leading-snug text-[var(--text-secondary)]">{t("cardFullDesc")}</span>
                </span>
              </button>
            </div>
          </div>
        ) : (
          /* ── КРОК 2: ДЕ ВАШЕ МІСЦЕ? + напис і CTA на тому ж екрані ──
              (карта і SVG-макет лишаються видимими поруч — усе живе) */
          <div className="flex flex-col gap-3">
            {/* ГОТОВО-банер: модель є — головна дія зверху, а ВСІ контролі нижче
                лишаються живими: міняй місце/напис і одразу «Оновити брелок». */}
            {successView && (
              <div className="flex flex-col gap-2.5">
                <div className="flex items-center gap-2 text-[16px] font-semibold text-[var(--text-primary)]">
                  <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-[var(--accent-strong)] text-white"><Check size={15} /></span>
                  {t("readyTitle")}
                </div>
                <p className="text-[12.5px] leading-snug text-[var(--text-secondary)]">{t("readyHint")}</p>
                <button
                  type="button"
                  onClick={() => window.dispatchEvent(new Event("monadruk:kc-guided-order"))}
                  className="inline-flex w-full items-center justify-center gap-2 rounded-full bg-[var(--bronze,#8E6B3D)] px-6 py-3.5 text-[15px] font-semibold text-white shadow-[0_8px_24px_rgba(142,107,61,0.35)] transition hover:brightness-110"
                >
                  <ShoppingBag size={18} /> {t("orderPrint")} · {disp(priceUah)}
                </button>
                <p className="text-center text-[11.5px] leading-snug text-[var(--text-secondary)]">{t("readyDelivery")}</p>
                <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1">
                  <button
                    type="button"
                    onClick={() => window.dispatchEvent(new Event("monadruk:kc-guided-download"))}
                    className="text-[12px] font-semibold text-[var(--text-secondary)] underline-offset-2 transition hover:text-[var(--text-primary)] hover:underline"
                  >
                    {t("downloadFree")}
                  </button>
                  <button type="button" onClick={onExitGuided} className="text-[12px] font-semibold text-[var(--text-secondary)] underline-offset-2 transition hover:text-[var(--text-primary)] hover:underline">
                    {t("tuneDetails")}
                  </button>
                  <button type="button" onClick={() => { setStarted(false); setTplId(null); }} className="text-[12px] font-semibold text-[var(--text-secondary)] underline-offset-2 transition hover:text-[var(--text-primary)] hover:underline">
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
            {/* Генерація: компактний прогрес над контролями — панель лишається живою. */}
            {generatingView && (
              <div className="flex flex-col gap-2 rounded-[16px] border border-[var(--surface-border)] bg-white/70 px-3.5 py-3">
                <div className="flex items-center gap-2 text-[14px] font-semibold text-[var(--text-primary)]">
                  <Loader2 size={16} className="animate-spin text-[var(--accent-strong)]" /> {t("generating")}
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-[rgba(15,23,42,0.08)]" role="progressbar" aria-label={t("generating")} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.max(0, Math.min(100, s.progress || 0))}>
                  <div className="h-full rounded-full bg-[var(--accent-strong)] transition-all duration-700" style={{ width: `${Math.max(4, Math.min(100, s.progress || 0))}%` }} />
                </div>
                <p className="text-[12px] text-[var(--text-secondary)]" aria-live="polite">
                  {s.progress}%{s.status ? ` · ${s.status}` : ""} · {t("etaNote")}
                </p>
              </div>
            )}
            <h2 className="font-title text-lg font-semibold text-[var(--text-primary)]">{t("step2Title")}</h2>
            {/* Пошук ПРЯМО в панелі: та сама подія monadruk:map-goto, яку слухає
                KeychainCropOverlay (зона переноситься, розмір лишається за шаблоном). */}
            <div className="rounded-full border border-[var(--surface-border)] bg-white/80 px-1.5 py-0.5 focus-within:border-[rgba(11,92,87,0.45)]">
              <MapSearchBox variant="panel" />
            </div>
            <div className="flex flex-wrap gap-1.5">
              {QUICK_CITIES.map((c) => (
                <button
                  key={c.en}
                  type="button"
                  onClick={() => {
                    window.dispatchEvent(new CustomEvent("monadruk:map-goto", {
                      detail: { lat: c.lat, lon: c.lon, label: locale === "uk" ? c.uk : c.en },
                    }));
                    // Напис слідує за містом, поки користувач не ввів свій
                    // (і шаблон дозволяє авто-напис — не половинка серця).
                    if (!labelManualRef.current && autoLabelRef.current) onLabelChange(c.text);
                  }}
                  className="rounded-full border border-[var(--surface-border)] bg-white/70 px-3 py-1.5 text-[12px] font-semibold text-[var(--text-secondary)] transition hover:border-[rgba(11,92,87,0.4)] hover:text-[var(--text-primary)]"
                >
                  {locale === "uk" ? c.uk : c.en}
                </button>
              ))}
            </div>
            {!placePicked ? (
              <div className="inline-flex items-center gap-2 rounded-full border border-[var(--surface-border)] bg-white/80 px-3.5 py-2 text-[13px] font-medium text-[var(--text-secondary)]">
                <MapPin size={15} className="animate-pulse text-[var(--accent-strong)]" /> {t("waitingPlace")}
              </div>
            ) : (
              <>
                <div className="inline-flex items-center gap-2 self-start rounded-full border border-[rgba(11,92,87,0.35)] bg-[rgba(15,118,110,0.1)] px-3.5 py-2 text-[13px] font-semibold text-[var(--text-primary)]">
                  <Check size={15} className="text-[var(--accent-strong)]" /> {t("placeChosen")}
                </div>
                {/* ПЕРСОНАЛІЗАЦІЯ: напис (той САМИЙ page-стан, що живить SVG-макет
                    і друк), маркер місця ♥/★/●, опційний напис на звороті. */}
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">{t("personalizeTitle")}</p>
                  <div className="mt-2 flex flex-col gap-2">
                    <label className="flex min-w-0 items-center gap-2 rounded-full border border-[var(--surface-border)] bg-white/80 px-3.5 py-2 focus-within:border-[rgba(11,92,87,0.45)]">
                      <PenLine size={14} className="shrink-0 text-[var(--accent-strong)]" />
                      <input
                        value={label}
                        onChange={(e) => {
                          labelManualRef.current = true;
                          onLabelChange(e.target.value.toUpperCase().slice(0, 28));
                        }}
                        maxLength={28}
                        placeholder={t("labelPlaceholder")}
                        aria-label={t("labelPlaceholder")}
                        className="w-full bg-transparent text-[13px] font-medium text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:outline-none"
                      />
                    </label>
                    {/* Маркер особливого місця — той самий page-стан placeMarker. */}
                    <div className="flex items-center gap-1.5" role="radiogroup" aria-label={t("markerTitle")}>
                      <span className="text-[12px] font-semibold text-[var(--text-secondary)]">{t("markerTitle")}</span>
                      {(["heart", "star", "circle"] as const).map((m) => (
                        <button
                          key={m}
                          type="button"
                          role="radio"
                          aria-checked={placeMarker === m}
                          onClick={() => onPlaceMarkerChange(placeMarker === m ? "" : m)}
                          className={`inline-flex h-9 w-9 items-center justify-center rounded-full border text-[15px] transition ${placeMarker === m ? chipOnCls : chipOffCls}`}
                        >
                          {m === "heart" ? "♥" : m === "star" ? "★" : "●"}
                        </button>
                      ))}
                    </div>
                    {/* «МІЙ ДІМ» — виділити свій будинок ОКРЕМОЮ деталлю (юзер:
                        «де вибір будинку»). Той самий store-механізм, що на
                        /create: тумблер вмикає режим кліку по карті, після кліку
                        режим сам вимикається, вибір лишається. Машинна копія
                        KeychainControlPanel читає store → keychain_highlight_building. */}
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
                        {s.highlightPoints.length > 0 ? t("myHomeCount", { n: s.highlightPoints.length }) : t("myHome")}
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
                    {/* Напис на звороті — опційний чіп (гравіювання на дні). */}
                    {!backOn && !backLabel ? (
                      <button
                        type="button"
                        onClick={() => setBackOn(true)}
                        className={`inline-flex items-center justify-center gap-2 rounded-full border px-3 py-2.5 text-[13px] font-semibold transition ${chipOffCls}`}
                      >
                        <PenLine size={15} className="text-[var(--accent-strong)]" /> {t("addBackLabel")}
                      </button>
                    ) : (
                      <div className="flex items-center gap-2">
                        <label className="flex min-w-0 flex-1 items-center gap-2 rounded-full border border-[var(--surface-border)] bg-white/80 px-3.5 py-2 focus-within:border-[rgba(11,92,87,0.45)]">
                          <PenLine size={14} className="shrink-0 text-[var(--accent-strong)]" />
                          <input
                            autoFocus={backOn}
                            value={backLabel}
                            onChange={(e) => onBackLabelChange(e.target.value.toUpperCase().slice(0, 28))}
                            maxLength={28}
                            placeholder={t("backLabelPlaceholder")}
                            aria-label={t("backLabelPlaceholder")}
                            className="w-full bg-transparent text-[13px] font-medium text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:outline-none"
                          />
                        </label>
                        <button
                          type="button"
                          onClick={() => { onBackLabelChange(""); setBackOn(false); }}
                          aria-label={t("clearField")}
                          className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-[var(--surface-border)] bg-white/80 text-[var(--text-secondary)] transition hover:text-[var(--text-primary)]"
                        >
                          <X size={14} />
                        </button>
                      </div>
                    )}
                  </div>
                </div>
                {failedNote && (
                  <p className="rounded-[12px] border border-red-200 bg-red-50 px-3 py-2 text-[12.5px] text-red-700">
                    {t("genFailed")}
                  </p>
                )}
                {!s.isGenerating && (
                  <>
                  {/* F-08: превʼю безкоштовне — ціна рядком під кнопкою, не на ній. */}
                  <button
                    type="button"
                    onClick={create}
                    disabled={!placePicked}
                    title={!placePicked ? t("waitingPlace") : undefined}
                    data-testid="kc-scenario-create"
                    className={`inline-flex w-full items-center justify-center gap-2 rounded-full px-6 py-4 text-[16px] font-semibold text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50 ${successView ? "bg-[var(--accent-strong)] shadow-[0_8px_24px_rgba(11,92,87,0.3)]" : "bg-[var(--bronze,#8E6B3D)] shadow-[0_8px_24px_rgba(142,107,61,0.35)]"}`}
                  >
                    {successView ? t("updateKeychain") : t("previewCta")}
                  </button>
                  {!successView && (
                    <p className="mt-1.5 text-center text-[12px] font-semibold text-[var(--text-secondary)]">
                      {t("printFromLine", { price: disp(priceUah) })}
                    </p>
                  )}
                  </>
                )}
                {/* Вихід у повний дизайнер прямо з кроку 2 — стан зберігається. */}
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
      {/* F-04: sticky ціна + дія стану на мобільному (портал). */}
      <GuidedStickyBar
        visible={displayStep === 2}
        testId="kc-guided-sticky-bar"
        label={tOrder("prodKeychain")}
        price={disp(priceUah)}
        busy={generatingView}
        tone={successView ? "bronze" : "primary"}
        disabled={!generatingView && !successView && !placePicked}
        cta={generatingView
          ? `${Math.max(0, Math.min(100, s.progress || 0))}%`
          : successView ? t("orderPrint") : t("previewCtaShort")}
        onCta={() => {
          if (successView) window.dispatchEvent(new Event("monadruk:kc-guided-order"));
          else create();
        }}
      />
    </div>
  );
}
