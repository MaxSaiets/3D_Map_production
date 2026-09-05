"use client";

import { useEffect, useState } from "react";

interface ShareQrProps {
  url: string;
  /** px, default 96 */
  size?: number;
  label?: string;
  className?: string;
}

export function ShareQr({ url, size = 96, label, className }: ShareQrProps) {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setSrc(null);

    (async () => {
      try {
        const { toDataURL } = await import("qrcode");
        const dataUrl = await toDataURL(url, {
          margin: 1,
          width: size * 2,
          color: { dark: "#1f2a24", light: "#ffffff" },
        });
        if (!cancelled) {
          setSrc(dataUrl);
        }
      } catch {
        if (!cancelled) {
          setSrc(null);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [url, size]);

  return (
    <figure
      data-testid="share-qr"
      className={`flex flex-col items-center gap-1 ${className ?? ""}`}
    >
      <div
        className="rounded-[12px] border border-[var(--surface-border)] bg-white p-1.5"
        style={{ width: size, height: size }}
      >
        {src ? (
          <img
            src={src}
            alt={label ?? url}
            width={size}
            height={size}
            style={{ width: "100%", height: "100%" }}
          />
        ) : null}
      </div>
      {label ? (
        <span className="text-[10.5px] leading-tight text-[var(--text-secondary)]">
          {label}
        </span>
      ) : null}
    </figure>
  );
}

export default ShareQr;
