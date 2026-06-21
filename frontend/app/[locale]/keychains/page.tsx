"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowLeft, KeyRound, Layers3, Map as MapIcon, User } from "lucide-react";
import { KeychainControlPanel } from "@/components/KeychainControlPanel";
import { KeychainSlicerPreview } from "@/components/KeychainLifePreview";
import {
  DEFAULT_KEYCHAIN_DESIGN,
  KeychainDesigner,
  TemplateMiniature,
  KEYCHAIN_TEMPLATES,
  type KeychainDesignerConfig,
} from "@/components/KeychainDesigner";
import { useGenerationStore } from "@/store/generation-store";
import { GPX_MAX_M_PER_MM } from "@/lib/generation";
import { useTranslations } from "next-intl";
import { OnboardingTour } from "@/components/OnboardingTour";

function MapLoading() {
  const t = useTranslations("kcp");
  return (
    <div className="flex h-full min-h-[320px] items-center justify-center rounded-[24px] bg-[rgba(255,255,255,0.65)] text-sm text-[var(--text-secondary)]">
      {t("loadingMap")}
    </div>
  );
}

function Preview3DLoading() {
  const t = useTranslations("kcp");
  return (
    <div className="flex h-full min-h-[320px] items-center justify-center rounded-[20px] bg-[#0f172a] text-sm text-white/70">
      {t("loading3d")}
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

// Реальний 3D перегляд згенерованого брелка (Three.js, важкий, потрібен ssr:false)
const Preview3D = dynamic(
  () => import("@/components/Preview3D").then((mod) => ({ default: mod.Preview3D })),
  {
    ssr: false,
    loading: () => <Preview3DLoading />,
  },
);

const CITIES: Record<string, { center: [number, number]; label: string; defaultText: string }> = {
  Kyiv:           { center: [50.4501, 30.5234], label: "Київ",           defaultText: "KYIV" },
  Khmelnytskyi:   { center: [49.42, 26.98],     label: "Хмельницький",   defaultText: "KHMELNYTSKYI" },
  Lviv:           { center: [49.8397, 24.0297], label: "Львів",           defaultText: "LVIV" },
  Odesa:          { center: [46.4825, 30.7233], label: "Одеса",           defaultText: "ODESA" },
  Dnipro:         { center: [48.4647, 35.0462], label: "Дніпро",          defaultText: "DNIPRO" },
  Kharkiv:        { center: [49.9935, 36.2304], label: "Харків",          defaultText: "KHARKIV" },
  Vinnytsia:      { center: [49.2331, 28.4682], label: "Вінниця",         defaultText: "VINNYTSIA" },
  Ternopil:       { center: [49.5535, 25.5948], label: "Тернопіль",       defaultText: "TERNOPIL" },
  IvanoFrankivsk: { center: [48.9226, 24.7111], label: "Івано-Франківськ",defaultText: "IVANO-FRANKIVSK" },
  Chernihiv:      { center: [51.4982, 31.2893], label: "Чернігів",        defaultText: "CHERNIHIV" },
  Zaporizhzhia:   { center: [47.8388, 35.1396], label: "Запоріжжя",       defaultText: "ZAPORIZHZHIA" },
  Kryvyi_Rih:     { center: [47.9105, 33.3918], label: "Кривий Ріг",      defaultText: "KRYVYI RIH" },
  Mykolaiv:       { center: [46.9750, 32.0000], label: "Миколаїв",        defaultText: "MYKOLAIV" },
  Poltava:        { center: [49.5883, 34.5514], label: "Полтава",         defaultText: "POLTAVA" },
  Cherkasy:       { center: [49.4444, 32.0598], label: "Черкаси",         defaultText: "CHERKASY" },
  Zhytomyr:       { center: [50.2547, 28.6587], label: "Житомир",         defaultText: "ZHYTOMYR" },
  Sumy:           { center: [50.9077, 34.7981], label: "Суми",            defaultText: "SUMY" },
  Rivne:          { center: [50.6199, 26.2516], label: "Рівне",           defaultText: "RIVNE" },
  Lutsk:          { center: [50.7472, 25.3254], label: "Луцьк",           defaultText: "LUTSK" },
  Uzhhorod:       { center: [48.6238, 22.2947], label: "Ужгород",         defaultText: "UZHHOROD" },
  Chernivtsi:     { center: [48.2921, 25.9310], label: "Чернівці",        defaultText: "CHERNIVTSI" },
  Kherson:        { center: [46.6354, 32.6169], label: "Херсон",          defaultText: "KHERSON" },
  Kropyvnytskyi:  { center: [48.5132, 32.2597], label: "Кропивницький",   defaultText: "KROPYVNYTSKYI" },
  Manual:         { center: [49.0, 31.0],        label: "Інше / вручну",  defaultText: "CITY" },
};

type MobileTab = "map" | "settings" | "design";

export default function KeychainsPage() {
  const [currentCityKey, setCurrentCityKey] = useState("Kyiv");
  const [label, setLabel] = useState("KYIV");
  // backLabel піднято сюди (а не у KeychainControlPanel), щоб back-превʼю дизайнера
  // показував реальний напис звороту, а не плейсхолдер.
  const [backLabel, setBackLabel] = useState("");
  // label2 (другий рядок — дата/координати) теж піднято сюди, щоб дизайнер показав
  // його у превʼю (інакше гравіювалось без перегляду — WYSIWYG-розрив).
  const [label2, setLabel2] = useState("");
  const [design, setDesign] = useState<KeychainDesignerConfig>(DEFAULT_KEYCHAIN_DESIGN);
  const [sidePreview, setSidePreview] = useState<"slicer" | "model3d">("model3d");
  const [cropRotationDeg, setCropRotationDeg] = useState(0);
  const [mobileTab, setMobileTab] = useState<MobileTab>("map");
  const [cropPolygon, setCropPolygon] = useState<Array<[number, number]> | null>(null);
  const { selectedArea, downloadUrl, isGenerating, progress, status, setSelectedArea, setTaskGroup, setGenerating, taskGroupId } = useGenerationStore();
  // Завантажений GPX-трек (store.gpxFocus) → зона має розширюватись як на /create,
  // інакше довгий маршрут обрізало (maxMetersPerMm був жорстко 7).
  const gpxFocus = useGenerationStore((s) => s.gpxFocus);
  const tKc = useTranslations("kc"); // локалізовані назви шаблонів брелків
  const t = useTranslations("kcp");
  const tCity = useTranslations("cities");

  const currentCity = CITIES[currentCityKey] ?? CITIES.Manual;
  const mapAspectRatio = design.mapWidthMm / Math.max(design.mapHeightMm, 1);

  // Очищаємо попередню зону при відкритті /keychains щоб MapSelector стартував
  // з targetMetersPerMm (зелена зона), а не з пам'яті з main мапи.
  useEffect(() => {
    setSelectedArea(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Відновлюємо задачу БРЕЛКА після refresh (ЛИШЕ keychain-задачі — не мап, бо ключ
  // localStorage спільний). Інакше генерація брелка зависала «осиротілою» при перезавантаженні.
  useEffect(() => {
    try {
      const savedGroupId = localStorage.getItem("3dmap_task_group_id");
      const savedProduct = localStorage.getItem("3dmap_task_product");
      if (savedGroupId && !taskGroupId && savedProduct === "keychain") {
        const savedTaskIds = localStorage.getItem("3dmap_task_ids");
        const ids = savedTaskIds ? JSON.parse(savedTaskIds) : [savedGroupId];
        setTaskGroup(savedGroupId, ids, "keychain");
        setGenerating(true);
      }
    } catch { /* ignore */ }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Чернетка дизайну: форма/напис/місто переживають перезавантаження сторінки.
  // Зону карти навмисно НЕ відновлюємо (див. ефект вище — стартуємо з чистої).
  useEffect(() => {
    try {
      const raw = localStorage.getItem("monadruk:draft:keychain");
      if (!raw) return;
      const d = JSON.parse(raw);
      if (d.design) setDesign({ ...DEFAULT_KEYCHAIN_DESIGN, ...d.design });
      if (typeof d.label === "string" && d.label) setLabel(d.label);
      if (d.cityKey && CITIES[d.cityKey]) setCurrentCityKey(d.cityKey);
    } catch { /* ignore */ }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    const timer = setTimeout(() => {
      try {
        localStorage.setItem("monadruk:draft:keychain", JSON.stringify({ design, label, cityKey: currentCityKey }));
      } catch { /* ignore */ }
    }, 800);
    return () => clearTimeout(timer);
  }, [design, label, currentCityKey]);

  // Deep-link city: /keychains?city=<key> вибирає місто (паритет з /create).
  // Виконуємо ПІСЛЯ відновлення чернетки — query-param має пріоритет над localStorage.
  useEffect(() => {
    try {
      const cityParam = new URLSearchParams(window.location.search).get("city");
      if (cityParam && CITIES[cityParam]) {
        setCurrentCityKey(cityParam);
        setLabel(CITIES[cityParam].defaultText);
        setSelectedArea(null);
      }
    } catch { /* ignore */ }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Якщо користувач змінює шаблон (Token 45×26, 35×55, тощо) — мапа має інший
  // aspect ratio для карти. Скидаємо crop щоб MapSelector перерахував його під
  // новий aspect (вертикальний vs горизонтальний). Інакше залишається стара форма.
  useEffect(() => {
    setSelectedArea(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mapAspectRatio]);
  const handleCropRotationChange = useCallback((rotationDeg: number) => {
    setCropRotationDeg(rotationDeg);
  }, []);
  // Застосувати готовий шаблон: міняємо весь дизайн І скидаємо поворот карти,
  // щоб новий шаблон ставав чисто (інакше лишався старий кут від попередньої форми).
  const applyTemplate = useCallback((next: KeychainDesignerConfig) => {
    setCropRotationDeg(0);
    setDesign(next);
  }, []);
  const keychainCrop = useMemo(
    () => ({
      aspectRatio: mapAspectRatio,
      // GPX: коли є трек — даємо зоні розтягнутись (як на мапах), щоб маршрут влазив.
      maxMetersPerMm: gpxFocus ? GPX_MAX_M_PER_MM : 7.0,
      targetMetersPerMm: 3.5,
      mapWidthMm: design.mapWidthMm,
      mapHeightMm: design.mapHeightMm,
      baseShape: design.baseShape,
      cornerRadiusMm: design.cornerRadiusMm,
      rotationDeg: cropRotationDeg,
      onRotationChange: handleCropRotationChange,
      onPolygonChange: setCropPolygon,
      // D4 GPX: зона авто-наводиться на завантажений трек (як на /create)
      followGpxFocus: true,
    }),
    [cropRotationDeg, design.mapHeightMm, design.mapWidthMm, design.baseShape, design.cornerRadiusMm, handleCropRotationChange, mapAspectRatio, gpxFocus],
  );
  // Mobile = single scroll: every panel is visible and stacked (no tab juggling).
  // The bottom bar just smooth-scrolls to a section. Desktop keeps the 3-col grid.
  const mapPanelClasses = "flex";
  const settingsPanelClasses = "block";
  const designPanelClasses = "flex";
  const scrollTo = (id: string) => {
    setMobileTab(id === "kc-map" ? "map" : id === "kc-settings" ? "settings" : "design");
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="min-h-[100dvh] bg-transparent">
      {/* UX: тур лише до першої генерації — не перекриває прогрес/3D-результат */}
      {!isGenerating && !downloadUrl && (
        <OnboardingTour
          storageKey="onb_keychain_v1"
          steps={[
            { title: t("tour.step1Title"), body: t("tour.step1Body") },
            { title: t("tour.step2Title"), body: t("tour.step2Body") },
            { title: t("tour.step3Title"), body: t("tour.step3Body") },
          ]}
        />
      )}
      <div className="mx-auto flex min-h-[100dvh] max-w-[1800px] flex-col px-2 pb-4 pt-2 sm:px-4 lg:px-5">
        <header className="rounded-[24px] border border-[var(--surface-border)] bg-[rgba(252,249,243,0.92)] px-4 py-3 shadow-[0_12px_40px_rgba(31,41,55,0.07)] backdrop-blur lg:px-5">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div className="space-y-2">
              <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-[var(--text-secondary)]">
                {t("eyebrow")}
              </p>
              <div>
                <h1 className="font-title text-xl font-semibold tracking-tight text-[var(--text-primary)] sm:text-2xl">
                  {t("title")}
                </h1>
                <p className="mt-2 hidden max-w-3xl text-sm leading-6 text-[var(--text-secondary)] xl:block xl:text-[15px]">
                  {t("subtitle")}
                </p>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              {/* Паритет із шапкою /create: [На головну] + [Карти→/create] +
                  [Кабінет]. Раніше «Карти» вело на «/» (домашня), тож з
                  конструктора брелка не можна було стрибнути одразу в
                  конструктор МАП. Тепер «Карти» = помітна акцент-кнопка → /create
                  (дзеркало create→«Брелок»), а домашню дає окреме «На головну». */}
              <div className="flex gap-2">
                <Link
                  href="/"
                  title={t("navHome")}
                  className="flex min-h-[44px] items-center justify-center gap-1.5 rounded-[22px] border border-[var(--surface-border)] bg-white/80 px-3 py-3 text-sm font-semibold text-[var(--text-primary)] transition hover:bg-white"
                >
                  <ArrowLeft size={16} /> <span className="hidden sm:inline">{t("navHome")}</span>
                </Link>
                <Link
                  href="/create"
                  className="flex min-h-[44px] flex-1 items-center justify-center gap-2 rounded-[22px] border border-[var(--accent-strong)] bg-[var(--accent-strong)] px-3 py-3 text-sm font-semibold text-white shadow-[0_2px_8px_rgba(11,92,87,0.25)] transition hover:bg-[rgba(11,92,87,0.92)]"
                >
                  <MapIcon size={16} /> {t("navMaps")}
                </Link>
                <Link
                  href="/account"
                  title={t("navAccount")}
                  className="flex min-h-[44px] items-center justify-center gap-1.5 rounded-[22px] border border-[rgba(11,92,87,0.25)] bg-[rgba(15,118,110,0.08)] px-3 py-3 text-sm font-semibold text-[var(--accent-strong)] transition hover:bg-[rgba(15,118,110,0.14)]"
                >
                  <User size={16} /> <span className="hidden sm:inline">{t("navAccount")}</span>
                </Link>
              </div>
            </div>
          </div>
        </header>

        {/* Степер «Крок 1/2/3» прибрано (власник: зайвий chrome). Натомість —
            компактний вибір міста (перенесено зі шапки). */}
        <div className="mt-3 flex items-center gap-2 rounded-[18px] border border-[var(--surface-border)] bg-white/70 px-3 py-2">
          <span className="shrink-0 text-[12px] font-semibold text-[var(--text-secondary)]">{t("city")}</span>
          <select
            value={currentCityKey}
            aria-label={t("city")}
            onChange={(event) => {
              const nextKey = event.target.value;
              // Скидаємо рамку СТАРОГО міста, інакше crop-overlay фітить назад на стару зону.
              setSelectedArea(null);
              setCurrentCityKey(nextKey);
              setLabel(CITIES[nextKey]?.defaultText ?? "CITY");
            }}
            className="min-h-[40px] flex-1 rounded-full border border-[var(--surface-border)] bg-white px-3 py-2 text-sm font-semibold text-[var(--text-primary)] outline-none transition focus:border-[rgba(11,92,87,0.4)]"
          >
            {Object.keys(CITIES).map((cityKey) => (
              <option key={cityKey} value={cityKey}>{tCity(cityKey)}</option>
            ))}
          </select>
        </div>

        {/* Перший крок — вибір форми брелка. */}
        <div className="mt-3 rounded-[24px] border border-[var(--surface-border)] bg-[var(--surface-panel)] p-3 shadow-[0_18px_54px_rgba(15,23,42,0.06)] backdrop-blur sm:p-4">
          <div className="mb-2 flex flex-wrap items-center gap-x-2 gap-y-1 px-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
            {t("pickShape")}
            <span className="ml-auto rounded-full bg-[rgba(46,74,58,0.07)] px-2 py-0.5 text-[10px] normal-case tracking-normal text-[var(--accent-strong)]">
              {t("pickShapeHint")}
            </span>
          </div>
          <div className="flex gap-2.5 overflow-x-auto pb-1">
            {KEYCHAIN_TEMPLATES.map((t) => {
              const active =
                t.design.baseShape === design.baseShape &&
                Math.round(t.design.bodyWidthMm) === Math.round(design.bodyWidthMm) &&
                Math.round(t.design.bodyHeightMm) === Math.round(design.bodyHeightMm);
              return (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => applyTemplate(t.design)}
                  aria-pressed={active}
                  className={`flex min-w-[150px] max-w-[170px] shrink-0 flex-col gap-2 rounded-[18px] border p-2.5 text-left transition ${
                    active
                      ? "border-[rgba(11,92,87,0.5)] bg-[rgba(15,118,110,0.1)] shadow-[0_10px_24px_rgba(11,92,87,0.14)]"
                      : "border-[var(--surface-border)] bg-white/80 hover:border-[rgba(11,92,87,0.25)]"
                  }`}
                >
                  <span className="block w-full overflow-hidden rounded-[12px]">
                    <TemplateMiniature design={t.design} label={label} active={active} />
                  </span>
                  <span className="flex items-center justify-between gap-2">
                    <span className="truncate text-sm font-semibold text-[var(--text-primary)]">{t.nameKey ? tKc(t.nameKey) : t.name}</span>
                    <span className="shrink-0 rounded-md bg-[rgba(46,74,58,0.08)] px-1.5 py-0.5 text-[10px] font-semibold text-[var(--accent-strong)]">
                      {Math.round(t.design.bodyWidthMm)}×{Math.round(t.design.bodyHeightMm)}
                    </span>
                  </span>
                  <span className="line-clamp-1 block text-[11px] leading-4 text-[var(--text-secondary)] lg:hidden">{t.descKey ? tKc(t.descKey) : t.description}</span>
                </button>
              );
            })}
          </div>
          <p className="mt-2 px-1 text-[11px] leading-4 text-[var(--text-secondary)] lg:hidden">
            {t.rich("dragHint", {
              rotate: (chunks) => <span className="font-semibold text-[var(--accent-strong)]">{chunks}</span>,
              b: (chunks) => <span className="font-semibold">{chunks}</span>,
            })}
          </p>
        </div>

        {/* Десктоп: ФІКСОВАНА висота сітки + 2 рівні рядки (1fr/1fr). Карта спанить
            обидва (повна висота), а design (рядок1) і 3D-превʼю (рядок2) ділять висоту
            навпіл — контент скролиться всередині (overflow-hidden). Раніше превʼю
            виштовхувалось на цілий екран нижче (повновисотна карта розтягувала рядок1). */}
        {/* ЛЕПТОП-ФІКС: рядки за КОНТЕНТОМ (auto), а не 1fr/1fr від висоти екрана —
            інакше на низькому ноуті grid ділив ~600px навпіл (~300px/панель) і
            overflow-hidden РІЗАВ налаштування/превʼю (скарга «все закріплено, нічого
            не видно»). Тепер кожна панель = своя висота, сторінка СКРОЛИТЬСЯ. */}
        <div className="mt-3 grid min-h-0 flex-1 gap-3 pb-24 lg:grid-cols-[340px_minmax(0,1.08fr)_minmax(380px,0.92fr)] lg:grid-rows-[auto_auto] lg:pb-10">
          <div id="kc-map" className={`${mapPanelClasses} order-2 min-h-[460px] scroll-mt-3 flex-col overflow-hidden rounded-[24px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_22px_70px_rgba(15,23,42,0.08)] backdrop-blur lg:order-2 lg:col-start-2 lg:row-start-1 lg:h-[calc(100dvh-150px)] lg:min-h-[440px]`}>
            <div className="flex items-center justify-between gap-3 border-b border-[var(--surface-border)] px-4 py-2.5 sm:px-5 sm:py-3">
              <div>
                <h2 className="flex items-center gap-2 font-title text-base font-semibold text-[var(--text-primary)] sm:text-lg">
                  <MapIcon size={16} /> {t("mapTitle")}
                </h2>
                <p className="mt-0.5 hidden text-xs leading-5 text-[var(--text-secondary)] sm:block">
                  {t("mapSubtitle")}
                </p>
              </div>
            </div>
            <div className="min-h-[56dvh] flex-1 bg-[rgba(255,255,255,0.55)] p-2 sm:min-h-[400px] sm:p-3 lg:min-h-0">
              <div className="h-full overflow-hidden rounded-[24px]">
                <MapSelector center={currentCity.center} keychainCrop={keychainCrop} />
              </div>
            </div>
          </div>

          <aside id="kc-settings" className={`${settingsPanelClasses} order-3 scroll-mt-3 overflow-hidden rounded-[24px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_18px_54px_rgba(15,23,42,0.08)] lg:order-1 lg:col-start-1 lg:row-start-1 lg:max-h-[calc(100dvh-150px)] lg:overflow-y-auto lg:backdrop-blur`}>
            <KeychainControlPanel
              label={label}
              onLabelChange={setLabel}
              label2={label2}
              onLabel2Change={setLabel2}
              backLabel={backLabel}
              onBackLabelChange={setBackLabel}
              design={design}
              onDesignChange={setDesign}
              cropRotationDeg={cropRotationDeg}
              cropPolygon={cropPolygon}
            />
          </aside>

          {/* PRODUCT LAYOUT — редактор форми. Перед картою (order-1). */}
          <section id="kc-design" className={`${designPanelClasses} order-1 scroll-mt-3 flex-col overflow-hidden rounded-[24px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_18px_54px_rgba(15,23,42,0.08)] backdrop-blur lg:order-3 lg:col-start-3 lg:row-start-1 lg:h-[calc(100dvh-150px)]`}>
              <div className="flex items-start justify-between gap-3 border-b border-[var(--surface-border)] px-4 py-3 sm:px-5">
                <div>
                  <p className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.22em] text-[var(--text-secondary)]">
                    <Layers3 size={14} />
                    {t("layoutEyebrow")}
                  </p>
                  <h2 className="mt-1 font-title text-lg font-semibold text-[var(--text-primary)]">
                    {t("layoutTitle")}
                  </h2>
                  <p className="mt-1 hidden text-xs leading-5 text-[var(--text-secondary)] sm:block sm:text-sm">
                    {t("layoutSubtitle")}
                  </p>
                </div>
                <div className="rounded-[18px] border border-[rgba(11,92,87,0.22)] bg-[rgba(15,118,110,0.08)] px-3 py-2 text-[var(--accent-strong)]">
                  <div className="flex items-center gap-2 text-sm font-semibold">
                    <KeyRound size={16} />
                    3MF
                  </div>
                </div>
              </div>
              <div className="grid min-h-0 flex-1 gap-3 p-2 sm:p-3">
                <div className="flex min-h-[380px] flex-col overflow-hidden rounded-[22px] border border-[rgba(15,23,42,0.12)] sm:min-h-[460px] lg:min-h-0">
                  <div className="min-h-[280px] flex-1 sm:min-h-[340px]">
                    <KeychainDesigner
                      value={design}
                      label={label}
                      label2={label2}
                      backLabel={backLabel}
                      onChange={setDesign}
                      cropRotationDeg={cropRotationDeg}
                      cropPolygon={cropPolygon}
                      mapBounds={selectedArea ? {
                        north: selectedArea.getNorth(),
                        south: selectedArea.getSouth(),
                        east: selectedArea.getEast(),
                        west: selectedArea.getWest(),
                      } : null}
                    />
                  </div>
                </div>
              </div>
          </section>

          {/* 3D-ВІДОБРАЖЕННЯ — окремо, ПІСЛЯ кнопки «Створити» (order-4). */}
          <section id="kc-preview3d" className={`${designPanelClasses} order-4 scroll-mt-3 flex-col overflow-hidden rounded-[24px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_18px_54px_rgba(15,23,42,0.08)] backdrop-blur lg:order-4 lg:col-start-1 lg:col-span-3 lg:row-start-2 lg:min-h-[420px]`}>
              <div className="flex items-center justify-between gap-3 border-b border-[var(--surface-border)] px-4 py-3 sm:px-5">
                <h2 className="flex items-center gap-2 font-title text-base font-semibold text-[var(--text-primary)] sm:text-lg">
                  <Layers3 size={16} /> {t("preview3dTitle")}
                </h2>
                <div className="flex overflow-hidden rounded-full border border-[var(--surface-border)] bg-white/70 p-0.5">
                  <button
                    type="button"
                    onClick={() => setSidePreview("model3d")}
                    aria-pressed={sidePreview === "model3d"}
                    className={`min-h-[40px] rounded-full px-3 text-xs font-semibold ${sidePreview === "model3d" ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)]"}`}
                  >
                    3D{downloadUrl ? " ●" : ""}
                  </button>
                  <button
                    type="button"
                    onClick={() => setSidePreview("slicer")}
                    aria-pressed={sidePreview === "slicer"}
                    className={`min-h-[40px] rounded-full px-3 text-xs font-semibold ${sidePreview === "slicer" ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)]"}`}
                  >
                    {t("layers")}
                  </button>
                </div>
              </div>
              <div className="min-h-[360px] flex-1 p-2 sm:p-3">
                <div className="relative h-full min-h-[340px] overflow-hidden rounded-[22px] border border-[rgba(15,23,42,0.12)]">
                  {sidePreview === "slicer" && <KeychainSlicerPreview design={design} label={label} />}
                  {sidePreview === "model3d" && (
                    downloadUrl ? (
                      <Preview3D />
                    ) : (
                      <div className="flex h-full min-h-[340px] flex-col items-center justify-center gap-2 rounded-[22px] bg-[#0f172a] p-6 text-center text-white/85">
                        <KeyRound size={32} className="text-[#5eead4]" />
                        <div className="font-title text-lg">{t("emptyTitle")}</div>
                        <div className="text-sm leading-6 text-white/55">
                          {t("emptyBody")}
                        </div>
                        {isGenerating && (
                          <div className="mt-3 inline-flex items-center gap-2 rounded-full bg-white/10 px-3 py-1.5 text-xs font-semibold">
                            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
                            {t("generatingPct", { progress })}
                          </div>
                        )}
                      </div>
                    )
                  )}
                </div>
              </div>
          </section>
        </div>
      </div>

      {/* Мобільна навігація = StickyActionBar (ціна+«Створити») з
          KeychainControlPanel. Степер «Крок 1/2/3» прибрано (власник). */}
    </div>
  );
}
