"use client";

import { createPortal } from "react-dom";
import { Loader2 } from "lucide-react";

/**
 * Мобільна (lg:hidden) закріплена панель знизу: завжди видима орієнтовна ціна +
 * головна дія поточного стану конструктора. Десктоп має ці ж кнопки в сайдбарі.
 * ВАЖЛИВО: позиціювання px/fixed (НЕ dvh) — інакше стрибає на iOS Safari.
 */
export function StickyActionBar({
  priceLabel,
  price,
  actionLabel,
  onAction,
  disabled = false,
  busy = false,
}: {
  priceLabel: string;
  price: string | null;
  actionLabel: string;
  onAction: () => void;
  disabled?: boolean;
  busy?: boolean;
}) {
  // Портал у <body>: предки з backdrop-filter/transform стають containing block
  // для position:fixed і «приклеюють» бар до панелі замість вьюпорта.
  const bar = (
    <div
      className="fixed inset-x-0 bottom-0 z-[60] border-t border-[var(--surface-border)] bg-[rgba(252,249,243,0.97)] px-4 pt-2.5 shadow-[0_-8px_30px_rgba(15,23,42,0.10)] backdrop-blur lg:hidden"
      style={{ paddingBottom: "calc(env(safe-area-inset-bottom, 0px) + 10px)" }}
    >
      <div className="mx-auto flex max-w-[640px] items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">{priceLabel}</div>
          <div className="truncate text-[17px] font-bold leading-tight text-[var(--text-primary)]">{price ?? "—"}</div>
        </div>
        <button
          type="button"
          onClick={onAction}
          disabled={disabled}
          className="inline-flex min-h-12 shrink-0 items-center justify-center gap-2 rounded-full bg-[var(--accent-strong)] px-6 py-3 text-sm font-bold text-white shadow-[0_12px_24px_rgba(11,92,87,0.28)] transition hover:bg-[var(--accent)] disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          {busy && <Loader2 className="h-4 w-4 animate-spin" />}
          {actionLabel}
        </button>
      </div>
    </div>
  );

  if (typeof document === "undefined") return null;
  return createPortal(bar, document.body);
}
