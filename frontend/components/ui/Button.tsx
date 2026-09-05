"use client";

import * as React from "react";
import { Loader2 } from "lucide-react";

/**
 * T-6.3: єдиний CTA-примітив. Раніше ~145 інлайн `className="… rounded-full …"`
 * дублювали кольори з різними (і невідповідними одне одному) фолбеками
 * (`bg-[var(--forest,#2E4A3A)]` vs `#2f6b46` vs `#2F4A3C`) — CSS-змінні завжди
 * визначені в app/globals.css, тож фолбек ніколи не рендериться, але дублікати
 * робили зміни крихкими. Тут — таблиця variant×size з ГОТОВИМИ рядками класів,
 * скопійованими 1:1 із реальних кнопок (жодних змін вигляду). Кожен запис —
 * самодостатній літерал: без спільної "бази", яку компонує JSX-порядок класів
 * (Tailwind не гарантує перемогу пізнішого класу над раннім при конфлікті
 * властивості), тож немає ризику, що клас з боку виклику "поб'є" клас варіанта
 * чи навпаки. Layout-утиліти (`w-full`, `flex-1`, `truncate`, `mx-auto`, …) НЕ
 * входять до таблиці — їх додає викликач через `className` (він завжди йде
 * останнім, тому такі суто адитивні класи безпечно домішуються).
 */
export type ButtonVariant = "primary" | "bronze" | "secondary" | "ghost";
export type ButtonSize = "sm" | "md" | "lg";

const VARIANT_SIZE_CLASSES: Record<ButtonVariant, Record<ButtonSize, string>> = {
  primary: {
    // Головна CTA гайдед-флоу (успіх): ScenarioFlow/KeychainScenarioFlow "scenario-create".
    lg: "inline-flex items-center justify-center gap-2 rounded-full bg-[var(--accent-strong)] px-6 py-4 text-[16px] font-semibold text-white shadow-[0_8px_24px_rgba(11,92,87,0.3)] transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50",
    // StickyActionBar: головна дія.
    md: "inline-flex items-center justify-center gap-1.5 rounded-full bg-[var(--accent-strong)] px-3 py-3 text-sm font-bold text-white shadow-[0_12px_24px_rgba(11,92,87,0.28)] transition hover:bg-[var(--accent)] disabled:cursor-not-allowed disabled:bg-slate-400",
    // order-success: "toAccount" (forest, НЕ accent-strong — інший відтінок).
    sm: "inline-flex items-center gap-2 rounded-full bg-[var(--forest,#2F4A3C)] px-6 py-3 text-sm font-bold text-white hover:opacity-90",
  },
  bronze: {
    // Головна CTA гайдед-флоу (до успіху): ScenarioFlow/KeychainScenarioFlow "scenario-create".
    lg: "inline-flex items-center justify-center gap-2 rounded-full bg-[var(--bronze,#8E6B3D)] px-6 py-4 text-[16px] font-semibold text-white shadow-[0_8px_24px_rgba(142,107,61,0.35)] transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50",
    // "Замовити друк" на екрані готово: guided-order / kc-guided-order.
    md: "inline-flex items-center justify-center gap-2 rounded-full bg-[var(--bronze,#8E6B3D)] px-6 py-3.5 text-[15px] font-semibold text-white shadow-[0_8px_24px_rgba(142,107,61,0.35)] transition hover:brightness-110",
    // Малий bronze (чіпи/вторинні CTA).
    sm: "inline-flex items-center justify-center gap-1.5 rounded-full bg-[var(--bronze,#8E6B3D)] px-4 py-2 text-[13px] font-semibold text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50",
  },
  secondary: {
    // Кнопка "Завантажити": guided-download / kc-guided-download.
    lg: "inline-flex items-center justify-center gap-2 rounded-full border border-[rgba(11,92,87,0.45)] bg-white px-6 py-3 text-[14.5px] font-semibold text-[var(--text-primary)] transition hover:border-[var(--accent-strong)] hover:bg-[rgba(15,118,110,0.06)]",
    // StickyActionBar: другорядна дія.
    md: "inline-flex items-center justify-center rounded-full border border-[var(--surface-border)] bg-white px-3 py-3 text-sm font-bold text-[var(--text-primary)] transition hover:bg-white/70",
    // Кнопка "Назад" у шапці кроку 2: ScenarioFlow/KeychainScenarioFlow.
    sm: "inline-flex items-center gap-1 rounded-full border border-[var(--surface-border)] bg-white/80 px-2.5 py-1 text-[11px] font-semibold text-[var(--text-secondary)] transition hover:border-[rgba(11,92,87,0.4)] hover:text-[var(--text-primary)]",
  },
  ghost: {
    // order-success: "makeAnother"/"toHome" (border-line пілюля).
    lg: "inline-flex items-center gap-2 rounded-full border border-line px-5 py-3 text-sm font-semibold text-ink-2 hover:bg-bg-2",
    // Текстовий лінк "Розширені налаштування": ScenarioFlow/KeychainScenarioFlow.
    md: "text-[12px] text-[var(--text-secondary)] underline underline-offset-2 hover:text-[var(--text-primary)]",
    // Текстовий лінк "Поділитись" у банері готово.
    sm: "inline-flex items-center gap-1.5 text-[12px] font-semibold text-[var(--accent-strong)] underline underline-offset-2 hover:text-[var(--text-primary)]",
  },
};

/** Той самий рядок класів, що й `<Button variant size>`, — для `<Link>`/`<a>`. */
export function buttonClasses(variant: ButtonVariant, size: ButtonSize): string {
  return VARIANT_SIZE_CLASSES[variant][size];
}

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant: ButtonVariant;
  size: ButtonSize;
  /** Показує спінер (lucide Loader2) перед children і виставляє aria-busy.
   *  НЕ вимикає кнопку сам по собі — керуй `disabled` окремо (як у StickyActionBar). */
  busy?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant, size, busy, className, children, type = "button", ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      aria-busy={busy || undefined}
      className={[buttonClasses(variant, size), className].filter(Boolean).join(" ")}
      {...rest}
    >
      {busy && <Loader2 className="h-4 w-4 shrink-0 animate-spin" aria-hidden="true" />}
      {children}
    </button>
  );
});
