"use client";

import { useEffect, useState } from "react";
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
    <div className="pointer-events-none fixed inset-0 z-[60] flex items-end justify-end p-4 sm:p-6">
      <div
        className="pointer-events-auto w-full max-w-[360px] rounded-[18px] border border-line bg-paper-2 p-5 shadow-lift fade-up"
        role="dialog"
        aria-label="Підказка"
      >
        <div className="mb-3 flex items-start justify-between">
          <div className="flex items-center gap-2">
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-forest text-[#F4EFE4]">
              <Sparkles size={14} />
            </span>
            <span className="eyebrow">Підказка · {idx + 1}/{steps.length}</span>
          </div>
          <button onClick={close} aria-label="Закрити" className="rounded-md p-1 text-ink-3 transition hover:bg-bg-2 hover:text-ink">
            <X size={16} />
          </button>
        </div>
        <h4 className="mb-1.5 font-serif text-[20px] text-ink">{step.title}</h4>
        <p className="mb-4 text-[14px] leading-relaxed text-ink-2">{step.body}</p>
        <div className="flex items-center justify-between">
          <button onClick={close} className="text-[12px] text-ink-3 underline-offset-2 hover:underline">
            Не показувати знову
          </button>
          {last ? (
            <button onClick={close} className="btn btn-primary btn-sm">Зрозуміло</button>
          ) : (
            <button onClick={() => setIdx((i) => i + 1)} className="btn btn-primary btn-sm">
              Далі <ArrowRight size={14} />
            </button>
          )}
        </div>
        {/* progress dots */}
        <div className="mt-4 flex justify-center gap-1.5">
          {steps.map((_, i) => (
            <span key={i} className="h-1.5 rounded-full transition-all"
              style={{ width: i === idx ? 18 : 6, background: i === idx ? "var(--forest)" : "var(--line-2)" }} />
          ))}
        </div>
      </div>
    </div>
  );
}
