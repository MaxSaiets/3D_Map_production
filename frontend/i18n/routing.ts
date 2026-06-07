import { defineRouting } from "next-intl/routing";

// uk = default, served at the root (no /uk prefix) so existing Ukrainian URLs
// and their SEO are preserved. Other locales are prefixed: /en, /de, /pl, /fr, /es.
export const locales = ["uk", "en", "de", "pl", "fr", "es"] as const;
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
};

export const routing = defineRouting({
  locales,
  defaultLocale,
  localePrefix: "as-needed",
  localeDetection: true,
});
