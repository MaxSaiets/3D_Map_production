"use client";

import { useEffect } from "react";
import { useTranslations } from "next-intl";
import { ArrowRight, RotateCcw, Home, KeyRound } from "lucide-react";
import { Link } from "@/i18n/navigation";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  const t = useTranslations("error");

  useEffect(() => {
    // Log for diagnostics only — never surface raw error.message to users.
    // eslint-disable-next-line no-console
    console.error("App error boundary:", error);
  }, [error]);

  return (
    <div className="flex min-h-[100dvh] items-center justify-center bg-paper px-5 py-16 text-ink">
      <div className="w-full max-w-[520px] text-center">
        <div className="eyebrow eyebrow-dot mx-auto inline-flex justify-center">monadruk</div>
        <h1 className="mt-5 font-serif text-[clamp(30px,5vw,48px)] leading-tight text-forest">
          {t("title")}
        </h1>
        <p className="mx-auto mt-5 max-w-[420px] text-[15px] leading-relaxed text-ink-2">
          {t("help")}
        </p>

        <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
          <button
            type="button"
            onClick={reset}
            className="inline-flex min-h-[48px] items-center gap-2 rounded-full bg-forest px-6 py-3 text-sm font-bold text-[#F4EFE4] shadow-[0_10px_24px_rgba(46,74,58,0.28)] transition hover:opacity-90"
            style={{ background: "var(--forest, #2E4A3A)" }}
          >
            <RotateCcw size={16} /> {t("retry")}
          </button>
          <Link
            href="/"
            className="inline-flex min-h-[48px] items-center gap-2 rounded-full border border-line px-5 py-3 text-sm font-semibold text-ink-2 transition hover:border-forest/40 hover:text-ink"
          >
            <Home size={16} /> {t("home")}
          </Link>
        </div>

        <div className="mt-5 flex flex-wrap items-center justify-center gap-x-5 gap-y-2 text-[13px] font-semibold text-ink-3">
          <Link href="/create" className="inline-flex items-center gap-1 hover:text-ink">
            {t("createMap")} <ArrowRight size={13} />
          </Link>
          <Link href="/contacts" className="inline-flex items-center gap-1 hover:text-ink">
            <KeyRound size={13} /> {t("contacts")}
          </Link>
        </div>

        {error?.digest && (
          <p className="mt-8 text-[11px] text-ink-3/70">
            <span className="font-mono">{error.digest}</span>
          </p>
        )}
      </div>
    </div>
  );
}
