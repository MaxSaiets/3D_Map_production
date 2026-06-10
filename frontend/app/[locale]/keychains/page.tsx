"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowLeft, KeyRound, Layers3, Map as MapIcon, Settings2, User } from "lucide-react";
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
import { OnboardingTour } from "@/components/OnboardingTour";
import { WizardSteps } from "@/components/WizardSteps";

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

// Реальний 3D перегляд згенерованого брелка (Three.js, важкий, потрібен ssr:false)
const Preview3D = dynamic(
  () => import("@/components/Preview3D").then((mod) => ({ default: mod.Preview3D })),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full min-h-[320px] items-center justify-center rounded-[20px] bg-[#0f172a] text-sm text-white/70">
        Завантаження 3D перегляду…
      </div>
    ),
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
  const [design, setDesign] = useState<KeychainDesignerConfig>(DEFAULT_KEYCHAIN_DESIGN);
  const [sidePreview, setSidePreview] = useState<"slicer" | "model3d">("model3d");
  const [cropRotationDeg, setCropRotationDeg] = useState(0);
  const [mobileTab, setMobileTab] = useState<MobileTab>("map");
  const [cropPolygon, setCropPolygon] = useState<Array<[number, number]> | null>(null);
  const { selectedArea, downloadUrl, isGenerating, progress, status, setSelectedArea } = useGenerationStore();

  const currentCity = CITIES[currentCityKey] ?? CITIES.Manual;
  const mapAspectRatio = design.mapWidthMm / Math.max(design.mapHeightMm, 1);

  // Очищаємо попередню зону при відкритті /keychains щоб MapSelector стартував
  // з targetMetersPerMm (зелена зона), а не з пам'яті з main мапи.
  useEffect(() => {
    setSelectedArea(null);
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
      maxMetersPerMm: 7.0,
      targetMetersPerMm: 3.5,
      mapWidthMm: design.mapWidthMm,
      mapHeightMm: design.mapHeightMm,
      baseShape: design.baseShape,
      cornerRadiusMm: design.cornerRadiusMm,
      rotationDeg: cropRotationDeg,
      onRotationChange: handleCropRotationChange,
      onPolygonChange: setCropPolygon,
    }),
    [cropRotationDeg, design.mapHeightMm, design.mapWidthMm, design.baseShape, design.cornerRadiusMm, handleCropRotationChange, mapAspectRatio],
  );
  const statusLabel = isGenerating
    ? `${progress}% • ${status || "Генерація"}`
    : downloadUrl
      ? "Брелок готовий"
      : selectedArea
        ? "Ділянка вибрана"
        : "Оберіть ділянку";

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
      <OnboardingTour
        storageKey="onb_keychain_v1"
        steps={[
          { title: "Оберіть місто та район", body: "Виберіть місто й точку на карті — це буде мапа на вашому брелку." },
          { title: "Оберіть шаблон", body: "Жетон 55×30, класичний чи квадратний — натисніть, і розміри виставляться автоматично." },
          { title: "Додайте напис", body: "Введіть текст (напр. назву міста), посуньте чи поверніть його. Тоді натисніть «Згенерувати»." },
        ]}
      />
      <div className="mx-auto flex min-h-[100dvh] max-w-[1800px] flex-col px-2 pb-4 pt-2 sm:px-4 lg:px-5">
        <header className="rounded-[24px] border border-[var(--surface-border)] bg-[rgba(252,249,243,0.92)] px-4 py-3 shadow-[0_12px_40px_rgba(31,41,55,0.07)] backdrop-blur lg:px-5">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div className="space-y-2">
              <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-[var(--text-secondary)]">
                Конструктор брелків
              </p>
              <div>
                <h1 className="font-title text-xl font-semibold tracking-tight text-[var(--text-primary)] sm:text-2xl">
                  Майстерня брелків з мапою
                </h1>
                <p className="mt-2 hidden max-w-3xl text-sm leading-6 text-[var(--text-secondary)] sm:block sm:text-[15px]">
                  Пласка багатоколірна пластина з посиленою петлею, чистою смугою під напис і контрольованою висотою будинків.
                </p>
              </div>
            </div>

            <div className="grid gap-2 sm:grid-cols-3 lg:min-w-[500px]">
              <div className="flex gap-2">
                <Link
                  href="/"
                  className="flex min-h-[48px] flex-1 items-center justify-center gap-2 rounded-[22px] border border-[var(--surface-border)] bg-white/80 px-3 py-3 text-sm font-semibold text-[var(--text-primary)] transition hover:bg-white"
                >
                  <ArrowLeft size={16} /> Мапи
                </Link>
                <Link
                  href="/account"
                  className="flex min-h-[48px] flex-1 items-center justify-center gap-2 rounded-[22px] border border-[rgba(11,92,87,0.25)] bg-[rgba(15,118,110,0.08)] px-3 py-3 text-sm font-semibold text-[var(--accent-strong)] transition hover:bg-[rgba(15,118,110,0.14)]"
                >
                  <User size={16} /> Кабінет
                </Link>
              </div>
              <div className="rounded-[22px] border border-[var(--surface-border)] bg-white/80 px-4 py-3">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
                  Місто
                </div>
                <select
                  value={currentCityKey}
                  onChange={(event) => {
                    const nextKey = event.target.value;
                    setCurrentCityKey(nextKey);
                    setLabel(CITIES[nextKey]?.defaultText ?? "CITY");
                  }}
                  className="mt-1 min-h-[32px] w-full bg-transparent text-sm font-semibold text-[var(--text-primary)] outline-none"
                >
                  {Object.keys(CITIES).map((cityKey) => (
                    <option key={cityKey} value={cityKey}>
                      {CITIES[cityKey].label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="rounded-[22px] border border-[var(--surface-border)] bg-white/80 px-4 py-3">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
                  Стан
                </div>
                <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">{statusLabel}</div>
              </div>
            </div>
          </div>
        </header>

        <div className="mt-3">
          <WizardSteps
            variant="keychain"
            state={{
              cityLabel: CITIES[currentCityKey]?.label ?? currentCityKey,
              hasSelection: Boolean(selectedArea),
              isGenerating,
              hasDownload: Boolean(downloadUrl),
              progress,
            }}
            onStepClick={(key) => {
              const id = key === "place" ? "kc-map" : key === "settings" ? "kc-design" : "kc-preview3d";
              document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
            }}
          />
        </div>

        {/* Step 1: pick a keychain form template — the prominent first decision */}
        <div className="mt-3 rounded-[24px] border border-[var(--surface-border)] bg-[var(--surface-panel)] p-3 shadow-[0_18px_54px_rgba(15,23,42,0.06)] backdrop-blur sm:p-4">
          <div className="mb-2 flex flex-wrap items-center gap-x-2 gap-y-1 px-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-[var(--accent-strong)] text-[10px] font-bold text-white">1</span>
            Оберіть форму брелка
            <span className="ml-auto rounded-full bg-[rgba(46,74,58,0.07)] px-2 py-0.5 text-[10px] normal-case tracking-normal text-[var(--accent-strong)]">
              Натисніть приклад — форма застосується
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
                    <span className="truncate text-sm font-semibold text-[var(--text-primary)]">{t.name}</span>
                    <span className="shrink-0 rounded-md bg-[rgba(46,74,58,0.08)] px-1.5 py-0.5 text-[10px] font-semibold text-[var(--accent-strong)]">
                      {Math.round(t.design.bodyWidthMm)}×{Math.round(t.design.bodyHeightMm)}
                    </span>
                  </span>
                  <span className="line-clamp-2 block text-[11px] leading-4 text-[var(--text-secondary)]">{t.description}</span>
                </button>
              );
            })}
          </div>
          <p className="mt-2 px-1 text-[11px] leading-4 text-[var(--text-secondary)]">
            Далі: перетягуйте карту, напис і вушко прямо в прев'ю. Карту й напис можна <span className="font-semibold text-[var(--accent-strong)]">обертати</span> — тягніть кутову ручку <span className="font-semibold">⟳</span> на карті або зелену ручку <span className="font-semibold">↻</span> над написом.
          </p>
        </div>

        <div className="mt-3 grid min-h-0 flex-1 gap-3 pb-20 lg:grid-cols-[340px_minmax(0,1.08fr)_minmax(360px,0.92fr)] lg:pb-0">
          <div id="kc-map" className={`${mapPanelClasses} order-2 min-h-[460px] scroll-mt-3 flex-col overflow-hidden rounded-[24px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_22px_70px_rgba(15,23,42,0.08)] backdrop-blur lg:order-2 lg:col-start-2 lg:row-start-1 lg:min-h-[calc(100dvh-150px)]`}>
            <div className="flex items-center justify-between gap-3 border-b border-[var(--surface-border)] px-4 py-2.5 sm:px-5 sm:py-3">
              <div>
                <h2 className="flex items-center gap-2 font-title text-base font-semibold text-[var(--text-primary)] sm:text-lg">
                  <MapIcon size={16} /> Постав форму на карту
                </h2>
                <p className="mt-0.5 hidden text-xs leading-5 text-[var(--text-secondary)] sm:block">
                  Перетягни рамку; ручка ⟳ на карті — обертання. Бірюзова рамка тримає пропорції з превʼю.
                </p>
              </div>
            </div>
            <div className="min-h-[400px] flex-1 bg-[rgba(255,255,255,0.55)] p-2 sm:p-3 lg:min-h-0">
              <div className="h-full overflow-hidden rounded-[24px]">
                <MapSelector center={currentCity.center} keychainCrop={keychainCrop} />
              </div>
            </div>
          </div>

          <aside id="kc-settings" className={`${settingsPanelClasses} order-3 scroll-mt-3 overflow-hidden rounded-[24px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_18px_54px_rgba(15,23,42,0.08)] lg:order-1 lg:col-start-1 lg:row-start-1 lg:max-h-[calc(100dvh-150px)] lg:backdrop-blur`}>
            <KeychainControlPanel
              label={label}
              onLabelChange={setLabel}
              design={design}
              onDesignChange={setDesign}
              cropRotationDeg={cropRotationDeg}
              cropPolygon={cropPolygon}
            />
          </aside>

          {/* PRODUCT LAYOUT — редактор форми. Перед картою (order-1). */}
          <section id="kc-design" className={`${designPanelClasses} order-1 scroll-mt-3 flex-col overflow-hidden rounded-[24px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_18px_54px_rgba(15,23,42,0.08)] backdrop-blur lg:order-3 lg:col-start-3 lg:row-start-1 lg:min-h-[calc(100dvh-150px)]`}>
              <div className="flex items-start justify-between gap-3 border-b border-[var(--surface-border)] px-4 py-3 sm:px-5">
                <div>
                  <p className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.22em] text-[var(--text-secondary)]">
                    <Layers3 size={14} />
                    Product Layout
                  </p>
                  <h2 className="mt-1 font-title text-lg font-semibold text-[var(--text-primary)]">
                    Розмір, зона карти, вушко і підпис
                  </h2>
                  <p className="mt-1 hidden text-xs leading-5 text-[var(--text-secondary)] sm:block sm:text-sm">
                    Підбери форму брелка локально, потім встав обрану ділянку карти в пунктирну область.
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
          <section id="kc-preview3d" className={`${designPanelClasses} order-4 scroll-mt-3 flex-col overflow-hidden rounded-[24px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_18px_54px_rgba(15,23,42,0.08)] backdrop-blur lg:order-4 lg:col-start-3 lg:row-start-2`}>
              <div className="flex items-center justify-between gap-3 border-b border-[var(--surface-border)] px-4 py-3 sm:px-5">
                <h2 className="flex items-center gap-2 font-title text-base font-semibold text-[var(--text-primary)] sm:text-lg">
                  <Layers3 size={16} /> 3D-перегляд готового брелка
                </h2>
                <div className="flex overflow-hidden rounded-full border border-[var(--surface-border)] bg-white/70 p-0.5">
                  <button
                    type="button"
                    onClick={() => setSidePreview("model3d")}
                    className={`min-h-[32px] rounded-full px-3 text-[11px] font-semibold ${sidePreview === "model3d" ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)]"}`}
                  >
                    3D{downloadUrl ? " ●" : ""}
                  </button>
                  <button
                    type="button"
                    onClick={() => setSidePreview("slicer")}
                    className={`min-h-[32px] rounded-full px-3 text-[11px] font-semibold ${sidePreview === "slicer" ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)]"}`}
                  >
                    Шари
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
                        <div className="font-title text-lg">3D модель з'явиться після створення</div>
                        <div className="text-sm leading-6 text-white/55">
                          Натисніть «Створити 3MF» — і тут зʼявиться реальний 3D-перегляд з усіма шарами.
                        </div>
                        {isGenerating && (
                          <div className="mt-3 inline-flex items-center gap-2 rounded-full bg-white/10 px-3 py-1.5 text-xs font-semibold">
                            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
                            Генерація: {progress}%
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

      {/* Mobile bottom tab bar — sticky, прихований на десктопі */}
      <nav className="fixed inset-x-0 bottom-0 z-40 border-t border-[var(--surface-border)] bg-[rgba(252,249,243,0.96)] px-2 py-2 shadow-[0_-8px_24px_rgba(15,23,42,0.08)] backdrop-blur lg:hidden">
        <div className="mx-auto grid max-w-md grid-cols-3 gap-1.5">
          <button
            type="button"
            onClick={() => scrollTo("kc-design")}
            className={`flex min-h-[52px] flex-col items-center justify-center gap-0.5 rounded-[16px] px-2 py-1 text-[11px] font-semibold transition ${
              mobileTab === "design" ? "bg-[var(--accent-strong)] text-white shadow-[0_8px_18px_rgba(11,92,87,0.22)]" : "text-[var(--text-secondary)]"
            }`}
          >
            <Layers3 size={18} />
            Превʼю
          </button>
          <button
            type="button"
            onClick={() => scrollTo("kc-map")}
            className={`flex min-h-[52px] flex-col items-center justify-center gap-0.5 rounded-[16px] px-2 py-1 text-[11px] font-semibold transition ${
              mobileTab === "map" ? "bg-[var(--accent-strong)] text-white shadow-[0_8px_18px_rgba(11,92,87,0.22)]" : "text-[var(--text-secondary)]"
            }`}
          >
            <MapIcon size={18} />
            Карта
          </button>
          <button
            type="button"
            onClick={() => scrollTo("kc-settings")}
            className={`flex min-h-[52px] flex-col items-center justify-center gap-0.5 rounded-[16px] px-2 py-1 text-[11px] font-semibold transition ${
              mobileTab === "settings" ? "bg-[var(--accent-strong)] text-white shadow-[0_8px_18px_rgba(11,92,87,0.22)]" : "text-[var(--text-secondary)]"
            }`}
          >
            <Settings2 size={18} />
            Створити
          </button>
        </div>
      </nav>
    </div>
  );
}
