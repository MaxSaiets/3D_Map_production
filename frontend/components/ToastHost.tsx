"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";

/**
 * Глобальний хост тостів. Слухає window-подію `monadruk:toast` і показує
 * транзитне повідомлення зверху по центру. Раніше MapSelector диспатчив цю
 * подію (напр. «зону обмежено»), але ЖОДЕН компонент її не слухав → користувач
 * не бачив пояснення. Деталі події:
 *   { type?: "warn"|"info"|"error", message?: string, ns?: string, key?: string, params?: object }
 * Якщо є key — перекладаємо через next-intl (root-translator + повний шлях
 * `ns.key`), інакше показуємо готовий message.
 */
type ToastDetail = {
  type?: "warn" | "info" | "error";
  message?: string;
  ns?: string;
  key?: string;
  params?: Record<string, string | number>;
};
type Toast = { id: number; text: string; type: NonNullable<ToastDetail["type"]> };

export function ToastHost() {
  const t = useTranslations();
  const [toasts, setToasts] = useState<Toast[]>([]);

  useEffect(() => {
    let seq = 0;
    const handler = (e: Event) => {
      const d = (e as CustomEvent<ToastDetail>).detail || {};
      let text = d.message;
      if (!text && d.key) {
        const full = d.ns ? `${d.ns}.${d.key}` : d.key;
        try {
          text = t(full as never, d.params as never);
        } catch {
          text = full;
        }
      }
      if (!text) return;
      const id = ++seq;
      setToasts((ts) => [...ts.slice(-2), { id, text, type: d.type || "info" }]);
      window.setTimeout(() => setToasts((ts) => ts.filter((x) => x.id !== id)), 5200);
    };
    window.addEventListener("monadruk:toast", handler as EventListener);
    return () => window.removeEventListener("monadruk:toast", handler as EventListener);
  }, [t]);

  if (!toasts.length) return null;
  return (
    <div className="pointer-events-none fixed left-1/2 top-4 z-[90] flex w-[min(92vw,460px)] -translate-x-1/2 flex-col gap-2">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          role="status"
          className={`pointer-events-auto rounded-xl border px-4 py-2.5 text-[13px] font-medium shadow-lg backdrop-blur ${
            toast.type === "error"
              ? "border-red-300 bg-red-50/95 text-red-800"
              : toast.type === "warn"
                ? "border-amber-300 bg-amber-50/95 text-amber-900"
                : "border-line bg-white/95 text-ink"
          }`}
        >
          {toast.text}
        </div>
      ))}
    </div>
  );
}
