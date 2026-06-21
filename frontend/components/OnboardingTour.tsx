"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { X, ArrowRight, Sparkles } from "lucide-react";

export interface TourStep {
  title: string;
  body: string;
}

/**
 * Lightweight, non-blocking onboarding.
 * Shows a small floating card sequence on first visit (per storageKey).
 * Dismissable; "не показувати знову" persists in localStorage.
 * Not a modal that blocks the UI — it floats bottom-right.
 */
export function OnboardingTour({
  storageKey,
  steps,
}: {
  storageKey: string;
  steps: TourStep[];
}) {
  const t = useTranslations("tour");
  const [idx, setIdx] = useState(0);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    try {
      if (!localStorage.getItem(storageKey)) {
        // small delay so the page renders first
        const t = setTimeout(() => setVisible(true), 900);
        return () => clearTimeout(t);
      }
    } catch {/* ignore */}
  }, [storageKey]);

  if (!visible || steps.length === 0) return null;

  const step = steps[idx];
  const last = idx === steps.length - 1;

  const close = () => {
    setVisible(false);
    try { localStorage.setItem(storageKey, "1"); } catch {/* ignore */}
  };

  return (
    // Мобільний: картка ВНИЗУ (над нижнім баром), щоб НЕ перекривати карту/шапку
    // (раніше була зверху й закривала мапу — скарга власника). Десктоп: внизу праворуч.
    <div className="pointer-events-none fixed inset-0 z-[70] flex items-end justify-center px-3 pb-[92px] sm:justify-end sm:p-6">
      <div
        className="pointer-events-auto w-full max-w-[360px] rounded-[18px] border border-line bg-paper-2 p-5 shadow-lift fade-up"
        role="region"
        aria-label={t("hintAria")}
      >
        <div className="mb-3 flex items-start justify-between">
          <div className="flex items-center gap-2">
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-forest text-[#F4EFE4]">
              <Sparkles size={14} />
            </span>
            <span className="eyebrow">{t("hintCounter", { current: idx + 1, total: steps.length })}</span>
          </div>
          <button onClick={close} aria-label={t("closeAria")} className="grid h-9 w-9 place-items-center rounded-md text-ink-3 transition hover:bg-bg-2 hover:text-ink">
            <X size={16} />
          </button>
        </div>
        <h4 className="mb-1.5 font-serif text-[20px] text-ink">{step.title}</h4>
        <p className="mb-4 text-[14px] leading-relaxed text-ink-2">{step.body}</p>
        <div className="flex items-center justify-between">
          <button onClick={close} className="text-[12px] text-ink-3 underline-offset-2 hover:underline">
            {t("dontShowAgain")}
          </button>
          {last ? (
            <button onClick={close} className="btn btn-primary btn-sm">{t("done")}</button>
          ) : (
            <button onClick={() => setIdx((i) => i + 1)} className="btn btn-primary btn-sm">
              {t("next")} <ArrowRight size={14} />
            </button>
          )}
        </div>
        {/* progress dots — декоративні; крок озвучується текстовим лічильником (hintCounter) вище */}
        <div className="mt-4 flex justify-center gap-1.5" aria-hidden="true">
          {steps.map((_, i) => (
            <span key={i} className="h-1.5 rounded-full transition-all"
              style={{ width: i === idx ? 18 : 6, background: i === idx ? "var(--forest)" : "var(--line-2)" }} />
          ))}
        </div>
      </div>
    </div>
  );
}
