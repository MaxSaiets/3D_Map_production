"use client";

import { useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import { ArrowRight } from "lucide-react";

export default function NotFound() {
  const t = useTranslations("notFound");
  return (
    <div className="mx-auto flex min-h-[70dvh] max-w-[640px] flex-col items-center justify-center px-5 text-center">
      <div className="font-serif text-[clamp(72px,16vw,140px)] leading-none text-forest/25">404</div>
      <h1 className="mt-2 font-serif text-[clamp(24px,4vw,38px)] text-ink">{t("title")}</h1>
      <p className="mt-3 max-w-[440px] text-[15px] leading-relaxed text-ink-2">{t("text")}</p>
      <Link
        href="/"
        className="mt-7 inline-flex min-h-12 items-center justify-center gap-2 rounded-full bg-forest px-7 text-[15px] font-semibold text-[#F4EFE4] transition hover:brightness-110"
      >
        {t("home")} <ArrowRight size={16} />
      </Link>
    </div>
  );
}
