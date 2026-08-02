"use client";

import dynamic from "next/dynamic";
import { Link } from "@/i18n/navigation";
import { ArrowLeft, Home } from "lucide-react";
import { useTranslations } from "next-intl";

// Студія тягне canvas-редактор і (лениво) three.js — на сервері їй нічого робити.
const FloorplanStudio = dynamic(() => import("@/components/FloorplanStudio"), {
  ssr: false,
  loading: () => (
    <div className="mx-auto mt-10 h-[420px] w-full max-w-[1180px] animate-pulse rounded-[28px] bg-[rgba(255,255,255,0.6)]" />
  ),
});

export default function MaketPage() {
  const t = useTranslations("maket");

  return (
    <main className="min-h-screen bg-[var(--ivory,#f7f5ee)]">
      <div className="mx-auto w-full max-w-[1180px] px-4 pt-6">
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-[13px] text-[var(--text-secondary,#5a655a)] transition hover:text-[var(--text-primary,#1c2320)]"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          {t("backHome")}
        </Link>
        <header className="mt-4 text-center">
          <p className="inline-flex items-center gap-1.5 rounded-full border border-[var(--line-soft,#e3e0d5)] bg-white/60 px-3 py-1 text-[12px] text-[var(--text-secondary,#5a655a)]">
            <Home className="h-3.5 w-3.5" />
            {t("badge")}
          </p>
          <h1 className="mt-3 font-[var(--font-display,Cormorant)] text-[30px] leading-tight text-[var(--text-primary,#1c2320)] sm:text-[38px]">
            {t("h1")}
          </h1>
          <p className="mx-auto mt-3 max-w-[620px] text-[15px] leading-relaxed text-[var(--text-secondary,#5a655a)]">
            {t("subtitle")}
          </p>
        </header>
      </div>
      <FloorplanStudio />
    </main>
  );
}
