"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { API_BASE_URL } from "@/lib/api";

// three.js/drei не потрібні, доки не прийшла відповідь /api/share — тягнемо
// чанк лише коли реально є glb_url (той самий трюк, що й на лендінгу: T-6.5).
const Model3DViewerLazy = dynamic(() => import("@/components/Model3DViewerLazy"), { ssr: false });

interface ShareInfo {
  task_id: string;
  glb_url: string | null;
  png_url: string | null;
  product: "map" | "keychain" | null;
}

/**
 * Клієнтський віджет share-сторінки: підвантажує `/api/share/{taskId}` і,
 * якщо модель ще жива (glb_url є), показує інтерактивний 3D-вʼювер замість
 * статичного OG-скріншота. 90 днів — термін життя файлів на бекенді; після
 * цього glb_url приходить null і лишається лише картинка з поясненням.
 */
export default function ShareViewer({ taskId, ogImage }: { taskId: string; ogImage: string }) {
  const t = useTranslations("share");
  const [info, setInfo] = useState<ShareInfo | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE_URL}/api/share/${taskId}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data: ShareInfo | null) => {
        if (!cancelled) setInfo(data);
      })
      .catch(() => {
        if (!cancelled) setInfo(null);
      })
      .finally(() => {
        if (!cancelled) setLoaded(true);
      });
    return () => {
      cancelled = true;
    };
  }, [taskId]);

  const glbUrl = info?.glb_url
    ? info.glb_url.startsWith("http")
      ? info.glb_url
      : `${API_BASE_URL}${info.glb_url}`
    : null;

  return (
    <div>
      {glbUrl ? (
        <div data-testid="share-viewer">
          <Model3DViewerLazy
            url={glbUrl}
            height={420}
            allowZoom
            autoRotate
            poster={ogImage}
            label={info?.product === "keychain" ? t("ctaKeychain") : t("ctaMap")}
          />
        </div>
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={ogImage}
          alt={t("imageAlt")}
          className="mx-auto mt-8 w-full max-w-[560px] rounded-[24px] border border-line-soft bg-white/70 shadow-[0_18px_50px_rgba(46,74,58,0.12)]"
        />
      )}
      {loaded && !glbUrl && (
        <p data-testid="share-unavailable" className="mt-4 text-sm text-[var(--text-secondary)]">{t("unavailable")}</p>
      )}
      <p data-testid="share-expires" className="mt-3 text-xs text-[var(--text-secondary)]">{t("expires")}</p>
      {glbUrl && <p data-testid="share-viewer-hint" className="mt-1 text-xs text-[var(--text-secondary)]">{t("viewerHint")}</p>}
    </div>
  );
}
