import { defineRouting } from "next-intl/routing";

// uk = default, served at the root (no /uk prefix) so existing Ukrainian URLs
// and their SEO are preserved. Other locales are prefixed: /en, /de, /pl, /fr, /es, /ro.
// ro (07.10.2026) — для Молдови (державна мова — румунська): повний переклад
// інтерфейсу (messages/ro.json); SEO-контент сторінок міст/блогу — en-фолбек,
// тож у sitemap/індексі лише ядро (див. RO_INDEXED_PATHS).
export const locales = ["uk", "en", "de", "pl", "fr", "es", "ro"] as const;
export type AppLocale = (typeof locales)[number];
export const defaultLocale: AppLocale = "uk";

// Human labels + BCP-47 tags for hreflang / og:locale.
export const localeMeta: Record<AppLocale, { label: string; htmlLang: string; ogLocale: string }> = {
  uk: { label: "Українська", htmlLang: "uk", ogLocale: "uk_UA" },
  en: { label: "English", htmlLang: "en", ogLocale: "en_US" },
  de: { label: "Deutsch", htmlLang: "de", ogLocale: "de_DE" },
  pl: { label: "Polski", htmlLang: "pl", ogLocale: "pl_PL" },
  fr: { label: "Français", htmlLang: "fr", ogLocale: "fr_FR" },
  es: { label: "Español", htmlLang: "es", ogLocale: "es_ES" },
  ro: { label: "Română", htmlLang: "ro", ogLocale: "ro_RO" },
};

/** Сторінки, які у ro мають повний румунський текст (інтерфейс із messages/ro.json
 *  або перекладений COPY) — лише вони в sitemap і в індексі. Решта /ro/* —
 *  англійський фолбек контенту → `X-Robots-Tag: noindex, follow` (middleware.ts). */
export const RO_INDEXED_PATHS: readonly string[] = [
  "", "/create", "/keychains", "/mountains", "/prices", "/brelok", "/maket", "/worlds", "/pro",
];

export const routing = defineRouting({
  locales,
  defaultLocale,
  localePrefix: "as-needed",
  localeDetection: true,
});
