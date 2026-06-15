"use client";

import { useEffect } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import { X } from "lucide-react";
import { useTranslations } from "next-intl";

const Model3DViewer = dynamic(() => import("@/components/Model3DViewer"), {
  ssr: false,
  loading: () => <div className="flex h-full items-center justify-center text-white/70">Завантаження 3D…</div>,
});

export type ModalModel = { url: string; label: string; kind: "key" | "map"; price?: string };

/** Fullscreen, draggable + zoomable 3D viewer for one model. */
export default function ModelModal({ model, onClose }: { model: ModalModel | null; onClose: () => void }) {
  const t = useTranslations("modal");
  useEffect(() => {
    if (!model) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => { document.body.style.overflow = prev; window.removeEventListener("keydown", onKey); };
  }, [model, onClose]);

  if (!model) return null;
  return (
    <div
      className="fixed inset-0 z-[10000] flex items-center justify-center bg-ink/85 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="relative flex w-full max-w-[920px] flex-col overflow-hidden rounded-[24px] border border-white/15 bg-gradient-to-b from-[#f4efe3] to-[#e7ddc9] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          aria-label={t("close")}
          className="absolute right-3 top-3 z-10 grid h-10 w-10 place-items-center rounded-full bg-ink/80 text-white transition hover:bg-ink"
        >
          <X size={18} />
        </button>
        <Model3DViewer url={model.url} height={520} allowZoom autoRotate />
        <div className="flex items-center justify-between gap-3 border-t border-black/5 bg-white/55 px-5 py-4">
          <div>
            <div className="font-serif text-lg text-ink">{model.label}</div>
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
