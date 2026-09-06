"use client";

import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";

/**
 * Sticky-бар для guided-флоу на мобільному (F-04): ціна + головна дія стану завжди
 * на видноті. Портал у body (панель має overflow-hidden), лише < lg. Виставляє
 * `--sticky-h` на <html>, щоб cookie-банер, контакт-FAB і нижній padding сторінки
 * піднялись над баром (T-1.8) — і чистить змінну при демонтажі.
 */
export function GuidedStickyBar({
  visible,
  label,
  price,
  cta,
  onCta,
  disabled = false,
  tone = "primary",
  busy = false,
  testId = "guided-sticky-bar",
}: {
  visible: boolean;
  /** Лівий підпис, напр. «M · 8 см» або назва шаблону. */
  label: string;
  /** Ціна поруч із підписом, напр. «350 ₴». */
  price: string;
  /** Текст кнопки стану (Показати превʼю / 45 % / Замовити друк). */
  cta: string;
  onCta: () => void;
  disabled?: boolean;
  /** primary = бірюза (створити), bronze = замовлення (як у панелі). */
  tone?: "primary" | "bronze";
  /** Генерація триває — кнопка неактивна, показує прогрес. */
  busy?: boolean;
  testId?: string;
}) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!visible) {
      document.documentElement.style.removeProperty("--sticky-h");
      return;
    }
    const el = ref.current;
    if (!el) return;
    const mq = window.matchMedia("(min-width: 1024px)");
    const apply = () => {
      const h = mq.matches ? 0 : el.getBoundingClientRect().height;
      document.documentElement.style.setProperty("--sticky-h", `${Math.round(h)}px`);
    };
    apply();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(apply) : null;
    ro?.observe(el);
    mq.addEventListener("change", apply);
    return () => {
      ro?.disconnect();
      mq.removeEventListener("change", apply);
      document.documentElement.style.removeProperty("--sticky-h");
    };
  }, [visible]);

  if (!visible || typeof document === "undefined") return null;

  const btnCls = tone === "bronze"
    ? "bg-[var(--bronze,#8E6B3D)] shadow-[0_8px_20px_rgba(142,107,61,0.35)]"
    : "bg-[var(--accent-strong)] shadow-[0_8px_20px_rgba(11,92,87,0.3)]";

  return createPortal(
    <div
      ref={ref}
      data-testid={testId}
      className="fixed inset-x-0 bottom-0 z-[60] border-t border-[var(--surface-border)] bg-[rgba(251,248,240,0.97)] px-3 pt-2 backdrop-blur lg:hidden"
      style={{ paddingBottom: "calc(env(safe-area-inset-bottom, 0px) + 8px)" }}
    >
      <div className="mx-auto flex max-w-[680px] items-center gap-3">
        <div className="min-w-0 flex-1 leading-tight">
          <div className="truncate text-[12px] font-semibold text-[var(--text-secondary)]">{label}</div>
          <div className="text-[17px] font-extrabold text-[var(--text-primary)]">{price}</div>
        </div>
        <button
          type="button"
          onClick={onCta}
          disabled={disabled || busy}
          aria-busy={busy || undefined}
          className={`inline-flex min-h-[48px] shrink-0 items-center justify-center rounded-full px-5 text-[14px] font-bold text-white transition disabled:cursor-not-allowed disabled:opacity-60 ${btnCls}`}
        >
          {cta}
        </button>
      </div>
    </div>,
    document.body,
  );
}
