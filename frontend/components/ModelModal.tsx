"use client";

import { useEffect, useId, useRef } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import { X } from "lucide-react";
import { useTranslations } from "next-intl";

// Localized loading fallback for the dynamically-imported viewer. Defined as a
// component (not an inline arrow) so it can call the next-intl hook — the raw
// string used to be hardcoded Ukrainian («Завантаження 3D…»).
function ViewerLoading() {
  const t = useTranslations("modal");
  return <div className="flex h-full items-center justify-center text-white/70">{t("loading")}</div>;
}

const Model3DViewer = dynamic(() => import("@/components/Model3DViewer"), {
  ssr: false,
  loading: () => <ViewerLoading />,
});

export type ModalModel = { url: string; label: string; kind: "key" | "map"; price?: string };

/** Fullscreen, draggable + zoomable 3D viewer for one model. */
export default function ModelModal({ model, onClose }: { model: ModalModel | null; onClose: () => void }) {
  const t = useTranslations("modal");
  const titleId = useId();
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const closeBtnRef = useRef<HTMLButtonElement | null>(null);
  // Whatever held focus before the modal opened, so we can restore it on close.
  const restoreFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!model) return;
    restoreFocusRef.current = (document.activeElement as HTMLElement) ?? null;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { onClose(); return; }
      // Trap focus inside the dialog so Tab can't escape to the page behind it.
      if (e.key === "Tab" && dialogRef.current) {
        const focusable = dialogRef.current.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), input, [tabindex]:not([tabindex="-1"])',
        );
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        const active = document.activeElement as HTMLElement | null;
        if (e.shiftKey && (active === first || !dialogRef.current.contains(active))) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && active === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    // Move focus into the dialog (close button) so screen readers + keyboard land here.
    const focusTimer = window.setTimeout(() => closeBtnRef.current?.focus(), 0);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
      window.clearTimeout(focusTimer);
      restoreFocusRef.current?.focus?.();
    };
  }, [model, onClose]);

  if (!model) return null;
  return (
    <div
      className="fixed inset-0 z-[10000] flex items-center justify-center bg-ink/85 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="relative flex w-full max-w-[920px] flex-col overflow-hidden rounded-[24px] border border-white/15 bg-gradient-to-b from-[#f4efe3] to-[#e7ddc9] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          ref={closeBtnRef}
          onClick={onClose}
          aria-label={t("close")}
          className="absolute right-3 top-3 z-10 grid h-10 w-10 place-items-center rounded-full bg-ink/80 text-white transition hover:bg-ink"
        >
          <X size={18} />
        </button>
        <Model3DViewer url={model.url} height={520} allowZoom autoRotate />
        <div className="flex items-center justify-between gap-3 border-t border-black/5 bg-white/55 px-5 py-4">
          <div>
            <div id={titleId} className="font-serif text-lg text-ink">{model.label}</div>
            <div className="text-[12px] text-ink-3">
              {model.kind === "key" ? t("descKey") : t("descMap")} · {t("hint")}
            </div>
          </div>
          <Link
            href={model.kind === "key" ? "/keychains" : "/create"}
            className="shrink-0 rounded-full bg-forest px-5 py-2.5 text-sm font-bold text-white hover:brightness-110"
          >
            {model.price || (model.kind === "key" ? t("ctaKey") : t("ctaMap")) } →
          </Link>
        </div>
      </div>
    </div>
  );
}
