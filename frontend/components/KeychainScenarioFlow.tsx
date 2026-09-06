"use client";

import { useEffect, useRef, useState } from "react";
import { GuidedStickyBar } from "@/components/GuidedStickyBar";
import { useDownloadQuota } from "@/lib/useDownloadQuota";
import { useTranslations, useLocale } from "next-intl";
import { ArrowLeft, Check, Download, Home, MapPin, PenLine, Share2, ShoppingBag, Sliders, X, ShieldCheck } from "lucide-react";
import { MapSearchBox } from "@/components/MapSearchBox";
import { useShallow } from "zustand/react/shallow";
import { useGenerationStore } from "@/store/generation-store";
import { fetchQuote, type Quote } from "@/lib/pricing";
import { KEYCHAIN_PRICE_UAH, mapPriceEur } from "@/lib/mapPrices";
import {
  KEYCHAIN_TEMPLATES,
  type KeychainDesignerConfig,
} from "@/components/KeychainDesigner";
import { GenerationStages } from "@/components/GenerationStages";
import { Button } from "@/components/ui/Button";
import { ShareQr } from "@/components/ShareQr";

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
 * A-3/A-4/A-5 (2026-09-03): CTA активна завжди (бейдж каже, яке місце буде на
 * брелку), «готово» = 2 дії, «Оновити превʼю» лише коли щось змінили, одна
 * смуга прогресу з етапами.
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
  const dlQuota = useDownloadQuota();
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
    etaS: st.etaS,
    elapsedS: st.elapsedS,
    queued: st.queued,
    genError: st.genError,
    printPrep: st.printPrep,
    taskRestored: st.taskRestored,
    pendingGenerate: st.pendingGenerate,
    setPendingGenerate: st.setPendingGenerate,
    downloadUrl: st.downloadUrl,
    taskGroupId: st.taskGroupId,
    // Виділення будинку «мій дім» — той самий store, що на /create; машинна
    // копія KeychainControlPanel читає його і шле keychain_highlight_building.
    mapHighlightBuilding: st.mapHighlightBuilding,
    highlightPoints: st.highlightPoints,
    setMapHighlightBuilding: st.setMapHighlightBuilding,
    clearHighlights: st.clearHighlights,
  })));

  // T-2.1: текстовий лінк «Поділитись» у ГОТОВО-банері — та сама /share/{taskId}
  // сторінка, що й повна панель (SimpleControlPanel.doShare).
  const [shareCopied, setShareCopied] = useState(false);
  const doShareGuided = async () => {
    if (!s.taskGroupId) return;
    import("@/lib/analytics").then((m) => m.track("guided_share", { product: "keychain" })).catch(() => {});
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

  // A-6: єдиний вихід у розширений режим + подія для воронки.
  const exitGuided = (from: string) => {
    import("@/lib/analytics").then((m) => m.track("mode_switch", { product: "keychain", to: "advanced", from })).catch(() => {});
    onExitGuided();
  };

  // Режим кліку АВТО-ВИМИКАЄТЬСЯ одразу після вибору будинку (як на /create):
  // вибір лишається (друк дивиться на highlightPoints), а рамка знову рухома.
  const hlCountRef = useRef(0);
  useEffect(() => {
    if (s.mapHighlightBuilding && s.highlightPoints.length > hlCountRef.current) {
      s.setMapHighlightBuilding(false);
      // Guided-воронка: позначили «мій дім» на мапі.
      import("@/lib/analytics").then((m) => m.track("guided_home", { product: "keychain", action: "mark" })).catch(() => {});
    }
    hlCountRef.current = s.highlightPoints.length;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.highlightPoints.length, s.mapHighlightBuilding]);

  // Guided-воронка: очистили позначку «мій дім» (обидва місця виклику clearHighlights).
  const clearHomeGuided = () => {
    import("@/lib/analytics").then((m) => m.track("guided_home", { product: "keychain", action: "clear" })).catch(() => {});
    s.clearHighlights();
  };

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

  // A-4: CTA активна завжди; бейдж каже, ЯКЕ місце буде на брелку (дефолтна
  // рамка Києва → «Центр Києва (за замовчуванням)»; пошук/чіп → назва; ручний
  // зсув рамки → «обрана ділянка на карті»).
  const [placePicked, setPlacePicked] = useState(false);
  const createdAtRef = useRef(0);
  const touchedRef = useRef(false);
  const [placeLabel, setPlaceLabel] = useState<string>("");
  const lastGotoRef = useRef(0);
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
      if (Date.now() - createdAtRef.current > 2500) touchedRef.current = true;
      // Guided-воронка: ручний зсув/ресайз рамки — трек лише на ПЕРШИЙ перехід
      // «місце ще не обране» → «обране» (щоб не спамити подіями на кожен рух карти).
      if (!placePicked && !customPlaceTrackedRef.current) {
        customPlaceTrackedRef.current = true;
        import("@/lib/analytics").then((m) => m.track("guided_place", { product: "keychain", place: "custom" })).catch(() => {});
      }
      setPlacePicked(true);
      setPlaceLabel((cur) => (cur && Date.now() - lastGotoRef.current < 1500 ? cur : ""));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.selectedArea]);
  const customPlaceTrackedRef = useRef(false);
  useEffect(() => {
    const onPick = (e: Event) => {
      const d = (e as CustomEvent).detail as
        | { lat?: number; lon?: number; centerOnly?: boolean; label?: string }
        | undefined;
      if (!d || d.centerOnly) return;
      if (!Number.isFinite(d.lat) || !Number.isFinite(d.lon)) return;
      lastGotoRef.current = Date.now();
      if (Date.now() - createdAtRef.current > 2500) touchedRef.current = true;
      setPlacePicked(true);
      const pickedLabel = typeof d.label === "string" ? d.label.trim() : "";
      if (pickedLabel) setPlaceLabel(pickedLabel);
      // Guided-воронка: дискретна подія (чіп/пошук), а не рух карти — трекаємо завжди.
      import("@/lib/analytics").then((m) => m.track("guided_place", { product: "keychain", place: pickedLabel || "search" })).catch(() => {});
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
    import("@/lib/analytics").then((m) => {
      m.track("guided_pick", { product: "keychain", scenario: id });
      m.track("guided_step", { product: "keychain", step: 2 });
    }).catch(() => {});
    onApplyTemplate(tpl.design);
    const def = CARD_DEFS.find((c) => c.tplId === id);
    autoLabelRef.current = def?.autoLabel !== false;
    // Для шаблонів без авто-напису чистимо дефолтний KYIV — інакше він
    // мовчки гравіюється у вузькій частині форми (зрізаний до уламка).
    if (!autoLabelRef.current && !labelManualRef.current) onLabelChange("");
    setTplId(id);
  };

  // Deep-link ?lat=&lon= (кабінет: «Створити знову» зі збереженої моделі) —
  // переносить рамку карти на точні координати. Той самий event, що й пошук/
  // чіп міста; слухач вище позначає «Місце обрано». Шаблон карткою користувач
  // обирає сам на кроці 1 (тут навмисно НЕ форсуємо крок 2).
  useEffect(() => {
    try {
      const p = new URLSearchParams(window.location.search);
      const latParam = Number(p.get("lat"));
      const lonParam = Number(p.get("lon"));
      if (!Number.isFinite(latParam) || !Number.isFinite(lonParam)) return;
      window.setTimeout(() => {
        window.dispatchEvent(new CustomEvent("monadruk:map-goto", { detail: { lat: latParam, lon: lonParam, label: "" } }));
      }, 400);
    } catch { /* ignore */ }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // A-3: знімок параметрів у момент старту → «Оновити превʼю» лише при змінах.
  const areaKey = (() => {
    try {
      const b = s.selectedArea as unknown as { toBBoxString?: () => string } | null;
      return b?.toBBoxString ? b.toBBoxString() : "";
    } catch { return ""; }
  })();
  const paramsKey = JSON.stringify({ tplId, area: areaKey, label, backLabel, placeMarker, hl: s.highlightPoints.length });
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
      .then((m) => m.track("guided_generate", { product: "keychain", scenario: tplId, placePicked, place: placeLabel || "custom" }))
      .catch(() => {});
    setRan(false);
    setStarted(true);
    setSnapshotKey(paramsKey);
    createdAtRef.current = Date.now();
    touchedRef.current = false;
    window.dispatchEvent(new Event("monadruk:kc-guided-generate"));
    // O-1 (реальний прогін на iPhone): після тапу по sticky-CTA сторінка стоїть на
    // картках розмірів, а смуга стадій/ETA лишається вище за екраном — юзер бачить
    // лише тісний бокс «СТАН 10%» у панелі превʼю. На мобільному ведемо до стадій.
    if (typeof window !== "undefined" && window.innerWidth < 1024) {
      window.setTimeout(() => {
        document.querySelector('[data-testid="generation-stages"]')?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 250);
    }
  };

  // F-10: помилку показуємо лише після РЕАЛЬНОГО запуску генерації (див. ScenarioFlow).
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
  const resultTrackedAtRef = useRef(0);
  useEffect(() => {
    if (successView && resultTrackedAtRef.current !== createdAtRef.current) {
      resultTrackedAtRef.current = createdAtRef.current;
      import("@/lib/analytics").then((m) => m.track("guided_result", { product: "keychain", ok: true, elapsedS: s.elapsedS })).catch(() => {});
    } else if (failedNote && resultTrackedAtRef.current !== createdAtRef.current) {
      resultTrackedAtRef.current = createdAtRef.current;
      import("@/lib/analytics").then((m) => m.track("guided_result", { product: "keychain", ok: false, reason: s.genError || undefined })).catch(() => {});
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
          <Button
            variant="secondary"
            size="sm"
            onClick={() => { setTplId(null); setStarted(false); }}
          >
            <ArrowLeft size={12} /> {t("back")}
          </Button>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {tplId === null && !generatingView && !successView ? (
          /* ── КРОК 1: ЯКИЙ БРЕЛОК? ── */
          <div>
            <h2 className="font-title text-lg font-semibold text-[var(--text-primary)]">{t("step1Title")}</h2>
            <div className="mt-3 grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-2">
              {CARD_DEFS.map((c, i) => (
                <button key={c.tplId} type="button" onClick={() => pick(c.tplId)} className={cardBtnCls} data-testid={`kc-scenario-${c.tplId}`}>
                  {/* N-3 (перф): 400w за 1×, 640 за 2×; перші дві картки над згином → eager.
                      eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`/showcase/${c.img}-400.webp`}
                    srcSet={`/showcase/${c.img}-400.webp 1x, /showcase/${c.img}.webp 2x`}
                    alt={t(c.titleKey)}
                    loading={i < 2 ? "eager" : "lazy"}
                    className="aspect-[4/3] w-full object-cover transition duration-500 group-hover:scale-[1.04]"
                  />
                  <span className="flex flex-1 flex-col gap-0.5 px-2.5 py-2">
                    <span className="text-[13px] font-semibold leading-tight text-[var(--text-primary)]">{t(c.titleKey)}</span>
                    <span className="text-[12px] font-semibold text-[var(--accent-strong)]">{t("from", { price: disp(priceUah) })}</span>
                    <span className="text-[11px] leading-snug text-[var(--text-secondary)]">{t(c.descKey)}</span>
                  </span>
                </button>
              ))}
            </div>
            {/* A-6: розширений режим — один текстовий лінк під картками, не пʼята картка. */}
            <Button
              variant="bronze"
              size="sm"
              onClick={() => exitGuided("step1")}
              data-testid="kc-scenario-full"
              className="mt-3 inline-flex items-center gap-1.5 text-[12px] font-semibold text-[var(--text-secondary)] underline-offset-2 hover:text-[var(--text-primary)] hover:underline"
            >
              <Sliders size={12} /> {t("cardFullTitle")} — {t("cardFullDesc")}
            </Button>
          </div>
        ) : (
          /* ── КРОК 2: ДЕ ВАШЕ МІСЦЕ? + напис і CTA на тому ж екрані ──
              (карта і SVG-макет лишаються видимими поруч — усе живе) */
          <div className="flex flex-col gap-3">
            {/* ГОТОВО-банер (A-3): рівно дві дії — замовити або завантажити. */}
            {successView && (
              <div className="flex flex-col gap-2.5" data-testid="kc-guided-success">
                <div className="flex items-center gap-2 text-[16px] font-semibold text-[var(--text-primary)]">
                  <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-[var(--accent-strong)] text-white"><Check size={15} /></span>
                  {t("readyTitle")}
                </div>
                <p className="text-[12.5px] leading-snug text-[var(--text-secondary)]">
                  {placeLabel ? <b className="text-[var(--text-primary)]">{placeLabel} — </b> : null}{t("readyHint")}
                </p>
                {/* R-04 (Cities3ds): одна чесна фраза-обіцянка проти головного сумніву
                    «а надрукують те, що бачу?». Превʼю і 3MF — з одного canonical_2d. */}
                <p className="flex items-start gap-1.5 text-[11.5px] leading-snug text-[var(--text-secondary)]" data-testid="kc-guided-promise">
                  <ShieldCheck size={13} className="mt-[1px] shrink-0 text-[var(--accent-strong)]" /> {t("previewPromise")}
                </p>
                <Button
                  variant="bronze"
                  size="md"
                  onClick={() => {
                    import("@/lib/analytics").then((m) => m.track("guided_order_click", { product: "keychain", priceUah })).catch(() => {});
                    window.dispatchEvent(new Event("monadruk:kc-guided-order"));
                  }}
                  data-testid="kc-guided-order"
                  className="w-full"
                >
                  <ShoppingBag size={18} /> {t("orderPrint")} · {disp(priceUah)}
                </Button>
                <p className="text-center text-[11.5px] leading-snug text-[var(--text-secondary)]">{t("readyDelivery")}</p>
                {/* Завантаження — рівноправна кнопка, а не дрібний лінк (див. ScenarioFlow). */}
                <div className="flex items-center gap-2 pt-0.5">
                  <span className="h-px flex-1 bg-[var(--surface-border)]" />
                  <span className="text-[10.5px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">{t("waySelf")}</span>
                  <span className="h-px flex-1 bg-[var(--surface-border)]" />
                </div>
                <Button
                  variant="secondary"
                  size="lg"
                  data-testid="kc-guided-download"
                  onClick={() => {
                    import("@/lib/analytics").then((m) => m.track("guided_download", { product: "keychain" })).catch(() => {});
                    window.dispatchEvent(new Event("monadruk:kc-guided-download"));
                  }}
                  className="w-full"
                >
                  <Download size={17} /> {t("downloadCta")}
                </Button>
                <p className="text-center text-[11px] leading-snug text-[var(--text-secondary)]">{t("downloadSub")}</p>
                {/* T-D.5: залогінений бачить залишок безкоштовних файлів прямо тут. */}
                {dlQuota && !dlQuota.isAdmin && (
                  <p className="text-center text-[11px] font-semibold text-[var(--accent-strong)]">{t("quotaLeft", { n: dlQuota.remaining, limit: dlQuota.limit })}</p>
                )}
                {/* T-2.1: текстовий лінк, НЕ третя кнопка — рівно дві заповнені
                    кнопки лишаються (замовити/завантажити). */}
                {!!s.taskGroupId && (
                  <Button
                    variant="ghost"
                    size="sm"
                    data-testid="kc-guided-share"
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
            {/* Генерація (A-5): одна смуга прогресу з етапами — панель лишається живою. */}
            {generatingView && (
              <GenerationStages
                progress={s.progress || 0}
                kind="flat"
                title={t("generating")}
                note={t("etaNote")}
                eta={etaText}
                queued={s.queued}
                queuedTitle={t("queuedTitle")}
                queuedNote={t("queuedNote")}
                printPrep={s.printPrep}
                printPrepLabel={t("printPrepLine")}
                cancelLabel={t("cancelGen")}
                onCancel={() => window.dispatchEvent(new Event("monadruk:guided-cancel"))}
                stages={{ data: t("stageData"), detail: t("stageDetail"), file: t("stageFile") }}
              />
            )}
            <h2 className="font-title text-lg font-semibold text-[var(--text-primary)]">{t("step2Title")}</h2>
            {/* Орієнтир «як це працює» — власник: «не зрозуміло, як усе створювати». */}
            {!successView && !generatingView && (
              <p className="mt-1 text-[11.5px] leading-snug text-[var(--text-secondary)]">{t("howItWorks")}</p>
            )}
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
            {/* A-4: бейдж завжди каже, ЯКЕ місце буде на брелку. */}
            {!placePicked ? (
              <div className="flex flex-col gap-1" data-testid="kc-place-default">
                <div className="inline-flex items-center gap-2 self-start rounded-full border border-[var(--surface-border)] bg-white/80 px-3.5 py-2 text-[13px] font-semibold text-[var(--text-primary)]">
                  <MapPin size={15} className="text-[var(--accent-strong)]" /> {t("defaultPlace")}
                </div>
                <p className="text-[11.5px] leading-snug text-[var(--text-secondary)]">{t("defaultPlaceHint")}</p>
              </div>
            ) : (
              <div className="inline-flex max-w-full items-center gap-2 self-start rounded-full border border-[rgba(11,92,87,0.35)] bg-[rgba(15,118,110,0.1)] px-3.5 py-2 text-[13px] font-semibold text-[var(--text-primary)]" data-testid="kc-place-picked">
                <Check size={15} className="shrink-0 text-[var(--accent-strong)]" />
                <span className="truncate">{t("placeChosen")}{placeLabel ? `: ${placeLabel}` : `: ${t("customPlace")}`}</span>
              </div>
            )}
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
            {/* C-3: причина з бекенду + дії (повтор, написати нам). */}
            {failedNote && (
              <div className="flex flex-col gap-2 rounded-[12px] border border-red-200 bg-red-50 px-3 py-2.5" data-testid="kc-guided-error">
                <p className="text-[12.5px] leading-snug text-red-800">{s.genError || t("genFailed")}</p>
                <div className="flex flex-wrap gap-2">
                  <Button
                    variant="bronze"
                    size="sm"
                    onClick={create}
                    data-testid="kc-guided-retry"
                    className="rounded-full border border-red-300 bg-white px-3 py-1.5 text-[12px] font-semibold text-red-800 transition hover:bg-red-100"
                  >
                    {t("tryAgain")}
                  </Button>
                  <Button
                    variant="bronze"
                    size="sm"
                    onClick={() => window.dispatchEvent(new CustomEvent("monadruk:open-contact", { detail: { message: `${t("genFailed")} ${s.genError || ""}`.trim() } }))}
                    className="rounded-full px-3 py-1.5 text-[12px] font-semibold text-red-800 underline underline-offset-2"
                  >
                    {t("contactUs")}
                  </Button>
                </div>
              </div>
            )}
            {!s.isGenerating && (!successView || dirty) && (
              <>
              {/* F-08: превʼю безкоштовне — ціна рядком під кнопкою, не на ній. */}
              <Button
                variant={successView ? "primary" : "bronze"}
                size="lg"
                onClick={create}
                disabled={waitingForMap}
                data-testid="kc-scenario-create"
                className="w-full"
              >
                {successView ? t("updateKeychain") : t("previewCta")}
              </Button>
              {waitingForMap && (
                <p className="mt-1.5 text-center text-[12px] font-semibold text-[var(--accent-strong)]" aria-live="polite" data-testid="kc-map-loading-wait">
                  {t("mapLoadingWait")}
                </p>
              )}
              {!successView && (
                <p className="mt-1.5 text-center text-[12px] font-semibold text-[var(--text-secondary)]">
                  {t("printFromLine", { price: disp(priceUah) })}
                </p>
              )}
              </>
            )}
            {/* A-6: єдиний вихід у розширений режим — стан зберігається. */}
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
      {/* F-04: sticky ціна + дія стану на мобільному (портал). */}
      <GuidedStickyBar
        visible={displayStep === 2}
        testId="kc-guided-sticky-bar"
        label={tOrder("prodKeychain")}
        price={disp(priceUah)}
        busy={generatingView}
        tone={successView && !dirty ? "bronze" : "primary"}
        disabled={!generatingView && waitingForMap}
        cta={generatingView
          ? `${Math.max(0, Math.min(100, s.progress || 0))}%`
          : successView ? (dirty ? t("updateKeychain") : t("orderPrint")) : t("previewCtaShort")}
        onCta={() => {
          if (successView && !dirty) window.dispatchEvent(new Event("monadruk:kc-guided-order"));
          else create();
        }}
      />
    </div>
  );
}
