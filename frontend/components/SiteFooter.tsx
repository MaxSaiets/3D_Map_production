"use client";

import { Box, Instagram, Youtube, Send } from "lucide-react";
import { useTranslations, useLocale } from "next-intl";
import { Link, usePathname } from "@/i18n/navigation";
import { BUSINESS } from "@/lib/legal";

// Локалізований лейбл сторінки «Ціни» (без правок 6 messages-файлів).
const PRICES_LABEL: Record<string, string> = { uk: "Ціни", en: "Prices", de: "Preise", es: "Precios", fr: "Tarifs", pl: "Cennik" };
// Лейбли нових SEO-сторінок (blog/showcase/worlds) — теж без правок messages.
const BLOG_LABEL: Record<string, string> = { uk: "Блог", en: "Blog", de: "Blog", es: "Blog", fr: "Blog", pl: "Blog" };
const SHOWCASE_LABEL: Record<string, string> = { uk: "Галерея", en: "Gallery", de: "Galerie", es: "Galería", fr: "Galerie", pl: "Galeria" };
const WORLDS_LABEL: Record<string, string> = { uk: "3D-світи", en: "3D Worlds", de: "3D-Welten", es: "Mundos 3D", fr: "Mondes 3D", pl: "Światy 3D" };
const PANNO_LABEL: Record<string, string> = { uk: "Карта на стіну", en: "Wall map", de: "Wandkarte", es: "Mapa de pared", fr: "Carte murale", pl: "Mapa na ścianę" };
const KARPATY_LABEL: Record<string, string> = { uk: "Мапа Карпат", en: "Carpathians map", de: "Karpaten-Karte", es: "Mapa de los Cárpatos", fr: "Carte des Carpates", pl: "Mapa Karpat" };
// T-6.8 (F-26): /corporate був сиротою (лише sitemap) — даємо вхід із футера.
const CORPORATE_LABEL: Record<string, string> = { uk: "Для бізнесу", en: "For business", de: "Für Unternehmen", es: "Para empresas", fr: "Pour les entreprises", pl: "Dla firm" };
// SEO-FIX: хаб /brelok був СИРОТОЮ — на нього не вело жодне посилання з
// головної чи /maps, тому 24 сторінки /brelok/{місто} висіли в GSC як
// «Виявлено — наразі не проіндексовано» (Google не витрачав краул-бюджет).
const BRELOK_LABEL: Record<string, string> = { uk: "Брелки по містах", en: "City keychains", de: "Stadt-Anhänger", es: "Llaveros por ciudad", fr: "Porte-clés par ville", pl: "Breloki miast" };

/* ---------- Footer ----------
   Shared global footer: legal links + ФОП requisites + contacts.
   Extracted from the landing page so every content page (legal, maps, share)
   gets a consistent footer + a way home. */
export function SiteFooter() {
  const t = useTranslations("home.footer");
  const locale = useLocale();
  const pricesLabel = PRICES_LABEL[locale] ?? PRICES_LABEL.uk;
  const blogLabel = BLOG_LABEL[locale] ?? BLOG_LABEL.uk;
  const showcaseLabel = SHOWCASE_LABEL[locale] ?? SHOWCASE_LABEL.uk;
  const worldsLabel = WORLDS_LABEL[locale] ?? WORLDS_LABEL.uk;
  const pannoLabel = PANNO_LABEL[locale] ?? PANNO_LABEL.uk;
  const karpatyLabel = KARPATY_LABEL[locale] ?? KARPATY_LABEL.uk;
  const brelokLabel = BRELOK_LABEL[locale] ?? BRELOK_LABEL.uk;
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
            <Link href="/brelok" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{brelokLabel}</Link>
            <Link href="/podarunok" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("gift")}</Link>
            <Link href="/panno" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{pannoLabel}</Link>
            <Link href="/karpaty" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{karpatyLabel}</Link>
            <Link href="/corporate" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{CORPORATE_LABEL[locale] ?? CORPORATE_LABEL.uk}</Link>
            <Link href="/maket" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{t("maket")}</Link>
            <Link href="/blog" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{blogLabel}</Link>
            <Link href="/showcase" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{showcaseLabel}</Link>
            <Link href="/worlds" className="inline-flex min-h-[44px] items-center px-2.5 hover:text-ink">{worldsLabel}</Link>
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
        <div className="text-center">© {new Date().getFullYear()} monadruk.com</div>
      </div>
    </footer>
  );
}

/* Routes that render their own chrome (full-screen builders) or already include
   <SiteFooter /> themselves (the landing "/"). The global footer is suppressed
   on these so we never double up or break the builder layout. */
// /start — посадкова link-in-bio: миттєва і без 19-лінкового футера (3 дії + фото).
const NO_GLOBAL_FOOTER = new Set<string>(["/", "/create", "/keychains", "/start"]);
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
