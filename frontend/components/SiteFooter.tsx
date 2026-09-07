"use client";

import { Box, Instagram, Youtube, Send, Facebook } from "lucide-react";
import { useTranslations } from "next-intl";
import { Link, usePathname } from "@/i18n/navigation";
import { BUSINESS } from "@/lib/legal";

/* ---------- Footer ----------
   Shared global footer: legal links + ФОП requisites + contacts.
   Extracted from the landing page so every content page (legal, maps, share)
   gets a consistent footer + a way home. */
export function SiteFooter() {
  const t = useTranslations("home.footer");
  const tNav = useTranslations("nav");
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
            <Link href="/prices" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{tNav("prices")}</Link>
            <Link href="/maps" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("maps")}</Link>
            <Link href="/brelok" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("brelok")}</Link>
            <Link href="/podarunok" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("gift")}</Link>
            <Link href="/panno" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("panno")}</Link>
            <Link href="/karpaty" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("karpaty")}</Link>
            <Link href="/corporate" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("corporate")}</Link>
            <Link href="/maket" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("maket")}</Link>
            <Link href="/blog" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("blog")}</Link>
            <Link href="/showcase" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{tNav("gallery")}</Link>
            <Link href="/worlds" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("worlds3d")}</Link>
            <Link href="/account" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("account")}</Link>
            <Link href="/delivery" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("delivery")}</Link>
            <Link href="/refund" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("refund")}</Link>
            <Link href="/offer" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("offer")}</Link>
            <Link href="/contacts" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("contacts")}</Link>
            <Link href="/privacy" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("privacy")}</Link>
            <Link href="/terms" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("terms")}</Link>
          </div>
        </div>
        {/* Соцмережі — видимі лінки для людей + бренд-сигнал для пошуковиків
            (дзеркалять Organization.sameAs у layout). UTM — щоб бачити зворотний
            трафік сайт→соцмережа в аналітиці власника платформ. */}
        <div className="flex items-center justify-center gap-2">
          <a href="https://www.instagram.com/monadruk/" target="_blank" rel="noopener me" aria-label="Instagram Monadruk"
             className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-line-soft text-ink-3 transition hover:border-[var(--accent)] hover:text-ink">
            <Instagram size={18} />
          </a>
          <a href="https://www.youtube.com/@monadruk" target="_blank" rel="noopener me" aria-label="YouTube Monadruk"
             className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-line-soft text-ink-3 transition hover:border-[var(--accent)] hover:text-ink">
            <Youtube size={18} />
          </a>
          <a href="https://t.me/monadruk" target="_blank" rel="noopener me" aria-label="Telegram Monadruk"
             className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-line-soft text-ink-3 transition hover:border-[var(--accent)] hover:text-ink">
            <Send size={18} />
          </a>
          <a href="https://www.tiktok.com/@monadruk" target="_blank" rel="noopener me" aria-label="TikTok Monadruk"
             className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-line-soft text-ink-3 transition hover:border-[var(--accent)] hover:text-ink">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden><path d="M16.5 3c.3 2.3 1.7 3.9 4 4.1v3.2c-1.5 0-2.9-.5-4-1.3v6.4c0 3.2-2.6 5.6-5.8 5.6S5 18.6 5 15.4s2.6-5.6 5.7-5.6c.3 0 .6 0 .9.1v3.3a2.5 2.5 0 0 0-.9-.2 2.4 2.4 0 1 0 2.4 2.4V3h3.4z"/></svg>
          </a>
          <a href="https://www.facebook.com/1080540055153795" target="_blank" rel="noopener me" aria-label="Facebook Monadruk"
             className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-line-soft text-ink-3 transition hover:border-[var(--accent)] hover:text-ink">
            <Facebook size={18} />
          </a>
          <a href="https://www.pinterest.com/monadruk/" target="_blank" rel="noopener me" aria-label="Pinterest Monadruk"
             className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-line-soft text-ink-3 transition hover:border-[var(--accent)] hover:text-ink">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden><path d="M12 2a10 10 0 0 0-3.6 19.3c-.1-.8-.2-2 0-2.9l1.2-5s-.3-.6-.3-1.5c0-1.4.8-2.4 1.8-2.4.8 0 1.3.6 1.3 1.4 0 .9-.6 2.2-.8 3.4-.2 1 .5 1.8 1.5 1.8 1.8 0 3.2-1.9 3.2-4.6 0-2.4-1.7-4.1-4.2-4.1-2.9 0-4.5 2.1-4.5 4.4 0 .9.3 1.8.8 2.3.1.1.1.2.1.3l-.3 1.2c0 .2-.2.2-.4.1-1.2-.6-2-2.4-2-3.9 0-3.2 2.3-6.1 6.6-6.1 3.5 0 6.2 2.5 6.2 5.8 0 3.4-2.2 6.2-5.2 6.2-1 0-2-.5-2.3-1.1l-.6 2.4c-.2.9-.8 2-1.2 2.6A10 10 0 1 0 12 2z"/></svg>
          </a>
        </div>
        {/* Контакти + реквізити продавця — вимога платіжних систем (LiqPay). */}
        <div className="flex flex-col items-center gap-1 border-t border-line-soft pt-5 text-center text-[13px] text-ink-3 md:flex-row md:flex-wrap md:justify-center md:gap-x-4">
          <span>{t("entity")}</span>
          <span className="hidden md:inline">·</span>
          <a className="inline-flex min-h-[44px] items-center hover:text-ink" href={`mailto:${BUSINESS.email}`}>{BUSINESS.email}</a>
          <span className="hidden md:inline">·</span>
          <a className="inline-flex min-h-[44px] items-center hover:text-ink" href={`tel:${BUSINESS.phone}`}>{BUSINESS.phoneDisplay}</a>
          <span className="hidden md:inline">·</span>
          <span>{t("addressLine")}</span>
        </div>
        <div className="flex flex-col items-center gap-1 text-center">
          <span>© {new Date().getFullYear()} monadruk.com</span>
          {/* Право змінити рішення про cookie в будь-який момент (GDPR): скидає
              cookie згоди і показує банер (SiteAnalytics) знову. */}
          <button
            type="button"
            onClick={() => {
              document.cookie = "mnd_consent=; path=/; max-age=0";
              window.dispatchEvent(new CustomEvent("mnd:consent", { detail: null }));
            }}
            className="inline-flex min-h-[32px] items-center px-2 text-[12px] text-ink-3 underline underline-offset-2 hover:text-ink"
          >
            {t("cookieSettings")}
          </button>
        </div>
      </div>
    </footer>
  );
}

/* Routes that render their own chrome (full-screen builders) or already include
   <SiteFooter /> themselves (the landing "/"). The global footer is suppressed
   on these so we never double up or break the builder layout. */
// /start — посадкова link-in-bio: миттєва і без 19-лінкового футера (3 дії + фото).
// 08.09.2026 (власник): футер ПОТРІБЕН і на /create та /keychains — контакти, оферта,
// спосіб написати в Telegram. Білдер рендерить свій контент, футер іде після SEO-прози.
const NO_GLOBAL_FOOTER = new Set<string>(["/", "/start"]);
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
