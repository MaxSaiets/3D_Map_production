"use client";

import { Check, MapPin, Crop, SlidersHorizontal, Download } from "lucide-react";

export interface WizardState {
  cityLabel: string;
  hasSelection: boolean;
  isGenerating: boolean;
  hasDownload: boolean;
  progress: number;
}

type StepStatus = "done" | "current" | "todo";

/**
 * Lightweight guided stepper that sits above the /create workspace.
 * It does NOT rebuild the workspace — it reflects the user's real progress
 * through the natural flow and highlights the next actionable step, giving
 * a wizard-like guidance while keeping the proven map+controls layout.
 */
export function WizardSteps({ state }: { state: WizardState }) {
  const { cityLabel, hasSelection, isGenerating, hasDownload, progress } = state;

  // Derive per-step status from the real workspace state.
  const cityDone = Boolean(cityLabel);
  const steps: Array<{
    key: string;
    label: string;
    hint: string;
    icon: typeof MapPin;
    status: StepStatus;
  }> = [
    {
      key: "city",
      label: "Місто",
      hint: cityLabel ? cityLabel : "Оберіть місто",
      icon: MapPin,
      status: cityDone ? "done" : "current",
    },
    {
      key: "area",
      label: "Ділянка",
      hint: hasSelection ? "Виділено" : "Намалюйте прямокутник на мапі",
      icon: Crop,
      status: hasSelection ? "done" : cityDone ? "current" : "todo",
    },
    {
      key: "settings",
      label: "Параметри",
      hint: "Розмір і шари (за бажанням)",
      icon: SlidersHorizontal,
      status: hasDownload ? "done" : hasSelection ? "current" : "todo",
    },
    {
      key: "result",
      label: "Готово",
      hint: isGenerating
        ? `Генерація ${progress}%`
        : hasDownload
          ? "Завантажте 3MF"
          : "Згенеруйте модель",
      icon: Download,
      status: hasDownload ? "current" : "todo",
    },
  ];

  return (
    <nav
      aria-label="Кроки створення"
      className="flex items-stretch gap-1.5 overflow-x-auto rounded-[22px] border border-[var(--surface-border)] bg-[rgba(255,255,255,0.7)] p-1.5 backdrop-blur sm:gap-2"
    >
      {steps.map((step, i) => {
        const Icon = step.icon;
        const isDone = step.status === "done";
        const isCurrent = step.status === "current";
        return (
          <div
            key={step.key}
            className={`flex min-w-fit flex-1 items-center gap-2.5 rounded-[16px] px-3 py-2 transition ${
              isCurrent
                ? "bg-[var(--accent-strong,#2E4A3A)] text-white shadow-[0_10px_24px_rgba(11,92,87,0.22)]"
                : isDone
                  ? "bg-[rgba(15,118,110,0.08)] text-[var(--accent-strong,#2E4A3A)]"
                  : "text-[var(--text-secondary)]"
            }`}
          >
            <span
              className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[12px] font-bold ${
                isCurrent
                  ? "bg-white/20 text-white"
                  : isDone
                    ? "bg-[var(--accent-strong,#2E4A3A)] text-white"
                    : "bg-black/5 text-[var(--text-secondary)]"
              }`}
            >
              {isDone ? <Check size={14} /> : <Icon size={14} />}
            </span>
            <span className="min-w-0">
              <span className="block text-[10px] font-semibold uppercase tracking-[0.16em] opacity-75">
                Крок {i + 1}
              </span>
              <span className="block truncate text-[13px] font-semibold leading-tight">{step.label}</span>
              <span
                className={`block truncate text-[11px] leading-tight ${
                  isCurrent ? "text-white/80" : "opacity-70"
                }`}
              >
                {step.hint}
              </span>
            </span>
          </div>
        );
      })}
    </nav>
  );
}
