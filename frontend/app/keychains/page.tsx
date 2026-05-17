"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useMemo, useState } from "react";
import { ArrowLeft, KeyRound, Layers3, Map as MapIcon } from "lucide-react";
import { KeychainControlPanel } from "@/components/KeychainControlPanel";
import {
  DEFAULT_KEYCHAIN_DESIGN,
  KeychainDesigner,
  KeychainTemplateStrip,
  type KeychainDesignerConfig,
} from "@/components/KeychainDesigner";
import { Preview3D } from "@/components/Preview3D";
import { useGenerationStore } from "@/store/generation-store";

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

const CITIES: Record<string, { center: [number, number] }> = {
  Kyiv: { center: [50.4501, 30.5234] },
  Khmelnytskyi: { center: [49.42, 26.98] },
};

const CITY_LABELS: Record<string, string> = {
  Kyiv: "Київ",
  Khmelnytskyi: "Хмельницький",
};

export default function KeychainsPage() {
  const [currentCityKey, setCurrentCityKey] = useState("Kyiv");
  const [label, setLabel] = useState("KYIV MAP");
  const [design, setDesign] = useState<KeychainDesignerConfig>(DEFAULT_KEYCHAIN_DESIGN);
  const { selectedArea, downloadUrl, isGenerating, progress, status } = useGenerationStore();
  const currentCity = CITIES[currentCityKey];
  const keychainCrop = useMemo(
    () => ({
      aspectRatio: design.mapWidthMm / Math.max(design.mapHeightMm, 1),
      maxMetersPerMm: 7.5,
      mapWidthMm: design.mapWidthMm,
      mapHeightMm: design.mapHeightMm,
    }),
    [design.mapHeightMm, design.mapWidthMm],
  );
  const statusLabel = isGenerating
    ? `${progress}% • ${status || "Генерація"}`
    : downloadUrl
      ? "Брелок готовий"
      : selectedArea
        ? "Ділянка вибрана"
        : "Оберіть ділянку";

  return (
    <div className="min-h-[100dvh] bg-transparent">
      <div className="mx-auto flex min-h-[100dvh] max-w-[1760px] flex-col px-3 pb-6 pt-3 sm:px-4 lg:px-6">
        <header className="rounded-[28px] border border-[var(--surface-border)] bg-[rgba(252,249,243,0.9)] px-4 py-4 shadow-[0_18px_60px_rgba(31,41,55,0.08)] backdrop-blur lg:px-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="space-y-2">
              <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-[var(--text-secondary)]">
                Keychain Studio
              </p>
              <div>
                <h1 className="font-title text-2xl font-semibold tracking-tight text-[var(--text-primary)] sm:text-3xl">
                  Майстерня брелків з мапою
                </h1>
                <p className="mt-2 hidden max-w-3xl text-sm leading-6 text-[var(--text-secondary)] sm:block sm:text-[15px]">
                  Пласка багатоколірна пластина з посиленою петлею, чистою смугою під напис і контрольованою висотою будинків.
                </p>
              </div>
            </div>

            <div className="grid gap-2 sm:grid-cols-3 lg:min-w-[520px]">
              <Link
                href="/"
                className="flex items-center gap-2 rounded-[22px] border border-[var(--surface-border)] bg-white/80 px-4 py-3 text-sm font-semibold text-[var(--text-primary)] transition hover:bg-white"
              >
                <ArrowLeft size={17} />
                До мап
              </Link>
              <div className="rounded-[22px] border border-[var(--surface-border)] bg-white/80 px-4 py-3">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
                  Місто
                </div>
                <select
                  value={currentCityKey}
                  onChange={(event) => setCurrentCityKey(event.target.value)}
                  className="mt-1 w-full bg-transparent text-sm font-semibold text-[var(--text-primary)] outline-none"
                >
                  {Object.keys(CITIES).map((cityKey) => (
                    <option key={cityKey} value={cityKey}>
                      {CITY_LABELS[cityKey] ?? cityKey}
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

        <div className="mt-3 grid min-h-0 flex-1 gap-3 lg:h-[calc(100dvh-150px)] lg:min-h-[720px] lg:grid-cols-[390px,minmax(0,1fr)]">
          <aside className="order-2 min-h-0 overflow-hidden rounded-[30px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_22px_70px_rgba(15,23,42,0.08)] backdrop-blur lg:order-1">
            <KeychainControlPanel
              label={label}
              onLabelChange={setLabel}
              design={design}
              onDesignChange={setDesign}
            />
          </aside>

          <section className="order-1 grid min-h-0 gap-3 lg:order-2 lg:grid-rows-[minmax(430px,1fr),minmax(280px,0.72fr)]">
            <div className="order-2 flex min-h-[520px] flex-col overflow-hidden rounded-[30px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_22px_70px_rgba(15,23,42,0.08)] backdrop-blur lg:order-1 lg:min-h-0">
              <div className="flex items-start justify-between gap-4 border-b border-[var(--surface-border)] px-4 py-4 sm:px-5">
                <div>
                  <p className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.22em] text-[var(--text-secondary)]">
                    <Layers3 size={14} />
                    Product Layout
                  </p>
                  <h2 className="mt-1 font-title text-xl font-semibold text-[var(--text-primary)]">
                    Розмір, зона карти, вушко і підпис
                  </h2>
                  <p className="mt-1 text-sm text-[var(--text-secondary)]">
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
              <div className="grid min-h-0 flex-1 gap-3 p-2 sm:p-3 xl:grid-cols-[minmax(0,1.08fr),minmax(320px,0.92fr)]">
                <div className="flex min-h-[390px] flex-col overflow-hidden rounded-[24px] border border-[rgba(15,23,42,0.12)] lg:min-h-0">
                  <div className="min-h-[320px] flex-1 lg:min-h-0">
                    <KeychainDesigner value={design} label={label} onChange={setDesign} />
                  </div>
                  <KeychainTemplateStrip value={design} label={label} onSelect={setDesign} />
                </div>
                <div className="min-h-[390px] overflow-hidden rounded-[24px] border border-[rgba(15,23,42,0.12)] lg:min-h-0">
                  <Preview3D />
                </div>
              </div>
            </div>

            <div className="order-1 flex min-h-[430px] flex-col overflow-hidden rounded-[30px] border border-[var(--surface-border)] bg-[var(--surface-panel)] shadow-[0_22px_70px_rgba(15,23,42,0.08)] backdrop-blur lg:order-2 lg:min-h-0">
              <div className="flex items-start justify-between gap-4 border-b border-[var(--surface-border)] px-4 py-4 sm:px-5">
                <div>
                  <p className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.22em] text-[var(--text-secondary)]">
                    <MapIcon size={14} />
                    Map Crop
                  </p>
                  <h2 className="mt-1 font-title text-xl font-semibold text-[var(--text-primary)]">
                    Поставте форму брелка на карту
                  </h2>
                  <p className="mt-1 text-sm text-[var(--text-secondary)]">
                    Бірюзова рамка повторює пропорції області карти з превю і не дає вибрати crop, який дрібніший за 0.4 мм у друці.
                  </p>
                </div>
              </div>
              <div className="min-h-0 flex-1 bg-[rgba(255,255,255,0.55)] p-2 sm:p-3">
                <div className="h-full overflow-hidden rounded-[24px]">
                  <MapSelector
                    key={`${currentCityKey}-${Math.round(design.mapWidthMm)}-${Math.round(design.mapHeightMm)}`}
                    center={currentCity.center}
                    keychainCrop={keychainCrop}
                  />
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
