"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { Camera, ShieldCheck } from "lucide-react";

// Реальні фото надрукованих виробів. Власник кладе print-1.jpg ... print-6.jpg у
// public/prints/ — вони зʼявляються автоматично. Доки фото нема — акуратний
// плейсхолдер (не битий <img>). Закриває «нуль доказу друку» з UX-аудиту.
const PRINTS = Array.from({ length: 6 }, (_, i) => `/prints/print-${i + 1}.jpg`);

export default function RealPrints() {
  const t = useTranslations("home.prints");
  return (
    <section className="bg-paper py-20 lg:py-28">
      <div className="mx-auto max-w-[1360px] px-5 lg:px-8">
        <h2 className="mb-3 max-w-[620px] text-[clamp(28px,3.2vw,46px)]">{t("title")}</h2>
        <p className="mb-5 max-w-[560px] text-[15px] text-ink-2">{t("sub")}</p>
        <div className="mb-10 inline-flex items-center gap-2 rounded-full border border-line-soft bg-bg px-4 py-2 text-[13px] font-semibold text-ink">
          <ShieldCheck size={16} className="text-bronze" /> {t("guarantee")}
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 sm:gap-4 lg:grid-cols-6">
          {PRINTS.map((src, i) => (
            <PrintTile key={src} src={src} alt={t("alt", { n: i + 1 })} placeholder={t("soon")} />
          ))}
        </div>
      </div>
    </section>
  );
}

function PrintTile({ src, alt, placeholder }: { src: string; alt: string; placeholder: string }) {
  const [failed, setFailed] = useState(false);
  return (
    <div className="aspect-square overflow-hidden rounded-[20px] border border-line-soft bg-bg-2">
      {failed ? (
        <div className="flex h-full flex-col items-center justify-center gap-2 text-ink-3">
          <Camera size={22} />
          <span className="px-2 text-center text-[11px] leading-4">{placeholder}</span>
        </div>
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={src}
          alt={alt}
          loading="lazy"
          className="h-full w-full object-cover transition-transform duration-500 hover:scale-105"
          onError={() => setFailed(true)}
        />
      )}
    </div>
  );
}
