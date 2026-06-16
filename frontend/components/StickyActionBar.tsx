"use client";

import { useEffect, useState } from "react";
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
  secondaryLabel,
  onSecondary,
}: {
  priceLabel: string;
  price: string | null;
  actionLabel: string;
  onAction: () => void;
  disabled?: boolean;
  busy?: boolean;
  // Друга дія (напр. «Завантажити» поряд із «Замовити друк» на екрані готового).
  secondaryLabel?: string;
  onSecondary?: () => void;
}) {
  // HYDRATION-FIX: на сервері порталу нема (document undefined), а клієнт
  // рендерив його одразу при гідрації → React-warning «server HTML mismatch»
  // на КОЖНОМУ завантаженні /create і /keychains. Монтуємо портал лише після
  // mount-ефекту — перший клієнтський рендер збігається з серверним (null).
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  // Портал у <body>: предки з backdrop-filter/transform стають containing block
  // для position:fixed і «приклеюють» бар до панелі замість вьюпорта.
  const bar = (
    <div
      className="fixed inset-x-0 bottom-0 z-[60] border-t border-[var(--surface-border)] bg-[rgba(252,249,243,0.97)] px-4 pt-2.5 shadow-[0_-8px_30px_rgba(15,23,42,0.10)] backdrop-blur lg:hidden"
      style={{ paddingBottom: "calc(env(safe-area-inset-bottom, 0px) + 10px)" }}
    >
      <div className="mx-auto flex max-w-[640px] items-center gap-2.5">
        {/* Ціну показуємо ЛИШЕ на кроці оформлення (price != null) — тоді ліворуч
            лейбл+ціна, праворуч кнопки. Під час створення (price=null) лейбл
            «3D-мапа» лише крав місце й обрізався до «3…» — тож прибираємо його, а
            дві кнопки розтягуємо на всю ширину (flex-1): великі й без обрізання. */}
        {/* Ціна — НАТУРАЛЬНА ширина (shrink-0, nowrap): ніколи не обрізається. Кнопки
            ділять решту (flex-1, truncate, менший padding), тож «≈390₴» завжди видно. */}
        {price != null && (
          <div className="shrink-0">
            <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--text-secondary)]">{priceLabel}</div>
            <div className="whitespace-nowrap text-[16px] font-bold leading-tight text-[var(--text-primary)]">{price}</div>
          </div>
        )}
        <div className={`flex min-w-0 items-center gap-2 ${price != null ? "flex-1" : "w-full"}`}>
          {secondaryLabel && onSecondary && (
            <button
              type="button"
              onClick={onSecondary}
              className="inline-flex min-h-12 min-w-0 flex-1 items-center justify-center truncate rounded-full border border-[var(--surface-border)] bg-white px-3 py-3 text-sm font-bold text-[var(--text-primary)] transition hover:bg-white/70"
            >
              {secondaryLabel}
            </button>
          )}
          <button
            type="button"
            onClick={onAction}
            disabled={disabled}
            className="inline-flex min-h-12 min-w-0 flex-1 items-center justify-center gap-1.5 truncate rounded-full bg-[var(--accent-strong)] px-3 py-3 text-sm font-bold text-white shadow-[0_12px_24px_rgba(11,92,87,0.28)] transition hover:bg-[var(--accent)] disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {busy && <Loader2 className="h-4 w-4 shrink-0 animate-spin" />}
            <span className="truncate">{actionLabel}</span>
          </button>
        </div>
      </div>
    </div>
  );

  if (!mounted || typeof document === "undefined") return null;
  return createPortal(bar, document.body);
}
