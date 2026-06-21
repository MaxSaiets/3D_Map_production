"use client";

import { Box } from "lucide-react";
import { useTranslations, useLocale } from "next-intl";
import { Link, usePathname } from "@/i18n/navigation";
import { BUSINESS } from "@/lib/legal";

// Локалізований лейбл сторінки «Ціни» (без правок 6 messages-файлів).
const PRICES_LABEL: Record<string, string> = { uk: "Ціни", en: "Prices", de: "Preise", es: "Precios", fr: "Tarifs", pl: "Cennik" };

/* ---------- Footer ----------
   Shared global footer: legal links + ФОП requisites + contacts.
   Extracted from the landing page so every content page (legal, maps, share)
   gets a consistent footer + a way home. */
export function SiteFooter() {
  const t = useTranslations("home.footer");
  const pricesLabel = PRICES_LABEL[useLocale()] ?? PRICES_LABEL.uk;
  return (
    <footer className="border-t border-line-soft py-12">
      <div className="mx-auto flex max-w-[1360px] flex-col gap-6 px-5 text-sm text-ink-3 lg:px-8">
        <div className="flex flex-col items-center justify-between gap-6 md:flex-row">
          <div className="flex items-center gap-2 font-serif text-lg text-ink">
            <Box size={18} className="text-forest" /> monadruk
          </div>
          <div className="flex flex-wrap justify-center gap-x-2 gap-y-2">
            {/* min-h 44px — комфортний touch-target на мобільних (WCAG) */}
            <Link href="/create" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("create")}</Link>
            <Link href="/keychains" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("keychains")}</Link>
            <Link href="/prices" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{pricesLabel}</Link>
            <Link href="/maps" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("maps")}</Link>
            <Link href="/podarunok" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("gift")}</Link>
            <Link href="/account" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("account")}</Link>
            <Link href="/delivery" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("delivery")}</Link>
            <Link href="/refund" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("refund")}</Link>
            <Link href="/offer" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("offer")}</Link>
            <Link href="/contacts" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("contacts")}</Link>
            <Link href="/privacy" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("privacy")}</Link>
            <Link href="/terms" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("terms")}</Link>
          </div>
        </div>
        {/* Контакти + реквізити продавця — вимога платіжних систем (LiqPay). */}
        <div className="flex flex-col items-center gap-1 border-t border-line-soft pt-5 text-center text-[13px] text-ink-3 md:flex-row md:flex-wrap md:justify-center md:gap-x-4">
          <span>{t("entity")}</span>
          <span className="hidden md:inline">·</span>
          <a className="hover:text-ink" href={`mailto:${BUSINESS.email}`}>{BUSINESS.email}</a>
          <span className="hidden md:inline">·</span>
          <a className="hover:text-ink" href={`tel:${BUSINESS.phone}`}>{BUSINESS.phoneDisplay}</a>
          <span className="hidden md:inline">·</span>
          <span>{t("addressLine")}</span>
        </div>
        <div className="text-center">© {new Date().getFullYear()} monadruk.com</div>
      </div>
    </footer>
  );
}

/* Routes that render their own chrome (full-screen builders) or already include
   <SiteFooter /> themselves (the landing "/"). The global footer is suppressed
   on these so we never double up or break the builder layout. */
const NO_GLOBAL_FOOTER = new Set<string>(["/", "/create", "/keychains"]);
function isBuilderPath(pathname: string): boolean {
  // /capture and /capture/[id] are full-screen capture flows.
  return pathname === "/capture" || pathname.startsWith("/capture/");
}

/**
 * Mounted globally in the locale layout. Renders <SiteFooter /> on every
 * content page (legal, maps, share, account, admin, showcase…) so they all get
 * the legal links, ФОП requisites and a way home — but NOT on the landing page
 * (which renders its own footer) nor the full-screen builders.
 */
export function GlobalFooter() {
  const pathname = usePathname();
  if (NO_GLOBAL_FOOTER.has(pathname) || isBuilderPath(pathname)) return null;
  return <SiteFooter />;
}

export default SiteFooter;
