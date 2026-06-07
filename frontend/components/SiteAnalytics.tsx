"use client";

import Script from "next/script";
import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import { GA_ID, getConsent, setConsent } from "@/lib/analytics";

/**
 * GDPR-friendly analytics: shows a localized consent banner; Google Analytics
 * is loaded ONLY after the visitor accepts (privacy-first — decline = no GA).
 * Set NEXT_PUBLIC_GA_ID to enable GA; the banner still works without it.
 */
export default function SiteAnalytics() {
  const t = useTranslations("consent");
  const [consent, setConsentState] = useState<"granted" | "denied" | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setConsentState(getConsent());
    setReady(true);
    const onChange = (e: Event) => setConsentState((e as CustomEvent).detail);
    window.addEventListener("mnd:consent", onChange);
    return () => window.removeEventListener("mnd:consent", onChange);
  }, []);

  const decide = (v: "granted" | "denied") => { setConsent(v); setConsentState(v); };

  return (
    <>
      {GA_ID && consent === "granted" && (
        <>
          <Script src={`https://www.googletagmanager.com/gtag/js?id=${GA_ID}`} strategy="afterInteractive" />
          <Script id="ga-init" strategy="afterInteractive">
            {`window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());gtag('config','${GA_ID}',{anonymize_ip:true});`}
          </Script>
        </>
      )}

      {ready && consent === null && (
        <div className="fixed inset-x-3 bottom-3 z-[60] mx-auto max-w-[680px] rounded-2xl border border-line bg-paper/95 p-4 shadow-lift backdrop-blur sm:flex sm:items-center sm:gap-4">
          <p className="text-[13px] leading-relaxed text-ink-2">
            {t("text")}{" "}
            <Link href="/privacy" className="font-semibold text-forest underline">{t("more")}</Link>
          </p>
          <div className="mt-3 flex shrink-0 gap-2 sm:mt-0">
            <button
              onClick={() => decide("denied")}
              className="min-h-[40px] flex-1 rounded-full border border-line px-4 text-[13px] font-semibold text-ink-2 transition hover:border-forest/40 sm:flex-none"
            >
              {t("decline")}
            </button>
            <button
              onClick={() => decide("granted")}
              className="min-h-[40px] flex-1 rounded-full bg-forest px-5 text-[13px] font-bold text-[#F4EFE4] transition hover:brightness-110 sm:flex-none"
            >
              {t("accept")}
            </button>
          </div>
        </div>
      )}
    </>
  );
}
