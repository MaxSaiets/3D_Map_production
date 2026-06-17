"use client";

import { Check, MapPin, SlidersHorizontal, Download } from "lucide-react";
import { useTranslations } from "next-intl";

export interface WizardState {
  cityLabel: string;
  hasSelection: boolean;
  isGenerating: boolean;
  hasDownload: boolean;
  progress: number;
}

export type WizardStepKey = "place" | "settings" | "result";

type StepStatus = "done" | "current" | "todo";

/**
 * Lightweight guided stepper that sits above the /create workspace.
 * It does NOT rebuild the workspace — it reflects the user's real progress
 * through the natural flow and highlights the next actionable step, giving
 * a wizard-like guidance while keeping the proven map+controls layout.
 * 3 кроки = 3 рішення: Місце → Вигляд → Результат. Клік по кроку (якщо
 * передано onStepClick) веде до відповідної зони конструктора.
 */
export function WizardSteps({
  state,
  variant = "map",
  onStepClick,
}: {
  state: WizardState;
  variant?: "map" | "keychain";
  onStepClick?: (key: WizardStepKey) => void;
}) {
  const t = useTranslations("wizard");
  const { cityLabel, hasSelection, isGenerating, hasDownload, progress } = state;
  const settingsLabel = variant === "keychain" ? t("settingsKeychain") : t("settingsMap");
  const settingsHint = variant === "keychain" ? t("hintKeychain") : t("hintMap");

  // Derive per-step status from the real workspace state.
  const cityDone = Boolean(cityLabel);
  const steps: Array<{
    key: WizardStepKey;
    label: string;
    hint: string;
    icon: typeof MapPin;
    status: StepStatus;
  }> = [
    {
      key: "place",
      label: t("place"),
      // UX: рамка стоїть ЗА ЗАМОВЧУВАННЯМ (і на мапі, і на брелку) — «Виділено»
      // брехало юзеру, що він уже щось зробив. Чесний текст: рамка готова.
      hint: !cityDone
        ? t("cityPrompt")
        : hasSelection
          ? `${cityLabel} · ${t("areaReady")}`
          : t("areaPrompt"),
      icon: MapPin,
      status: hasSelection ? "done" : "current",
    },
    {
      key: "settings",
      label: settingsLabel,
      hint: settingsHint,
      icon: SlidersHorizontal,
      status: hasDownload ? "done" : hasSelection ? "current" : "todo",
    },
    {
      key: "result",
      label: t("done"),
      hint: isGenerating
        ? t("generating", { progress })
        : hasDownload
          ? t("downloadReady")
          : t("generatePrompt"),
      icon: Download,
      status: hasDownload ? "current" : isGenerating ? "current" : "todo",
    },
  ];

  return (
    <nav
      aria-label={t("aria")}
      className="flex items-stretch gap-1.5 flex-wrap rounded-[22px] border border-[var(--surface-border)] bg-[rgba(255,255,255,0.7)] p-1.5 backdrop-blur sm:gap-2"
    >
      {steps.map((step, i) => {
        const Icon = step.icon;
        const isDone = step.status === "done";
        const isCurrent = step.status === "current";
        const Tag: any = onStepClick ? "button" : "div";
        return (
          <Tag
            key={step.key}
            {...(onStepClick ? { type: "button", onClick: () => onStepClick(step.key) } : {})}
            className={`flex min-w-fit flex-1 items-center gap-1.5 rounded-[16px] px-2 py-2 text-left transition sm:gap-2.5 sm:px-3 ${
              onStepClick ? "cursor-pointer hover:opacity-90" : ""
            } ${
              isCurrent
                ? "bg-[var(--accent-strong,#2E4A3A)] text-white shadow-[0_10px_24px_rgba(11,92,87,0.22)]"
                : isDone
                  ? "bg-[rgba(15,118,110,0.08)] text-[var(--accent-strong,#2E4A3A)]"
                  : "text-[var(--text-secondary)]"
            }`}
          >
            <span
              className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[12px] font-bold sm:h-7 sm:w-7 ${
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
              <span className="hidden sm:block text-[10px] font-semibold uppercase tracking-[0.16em]">
                {t("step", { n: i + 1 })}
              </span>
              <span className="block truncate text-[13px] font-semibold leading-tight">{step.label}</span>
              <span
                className={`hidden sm:block truncate text-[11px] leading-tight ${
                  isCurrent ? "text-white/80" : ""
                }`}
              >
                {step.hint}
              </span>
            </span>
          </Tag>
        );
      })}
    </nav>
  );
}
