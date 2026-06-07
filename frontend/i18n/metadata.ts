import type { Metadata } from "next";
import { getTranslations } from "next-intl/server";
import { locales, localeMeta, defaultLocale, routing, type AppLocale } from "./routing";

export const BASE = "https://monadruk.com";

/** Absolute URL for a path in a given locale (default locale = no prefix). */
export function localeUrl(locale: AppLocale, path = "") {
  const clean = path === "/" ? "" : path;
  return locale === defaultLocale ? `${BASE}${clean || "/"}` : `${BASE}/${locale}${clean}`;
}

function isLocale(x: string): x is AppLocale {
  return (routing.locales as readonly string[]).includes(x);
}

/**
 * Build localized SEO metadata for a page in one call.
 *
 *   export async function generateMetadata({ params }) {
 *     return pageMetadata({ locale: params.locale, path: "/keychains", ns: "keychainsMeta" });
 *   }
 *
 * The namespace `ns` must expose: title, description, and optionally keywords + og.
 * hreflang alternates for every locale + x-default are added automatically.
 */
export async function pageMetadata({
  locale: rawLocale,
  path,
  ns,
}: {
  locale: string;
  path: string;
  ns: string;
}): Promise<Metadata> {
  const locale: AppLocale = isLocale(rawLocale) ? rawLocale : defaultLocale;
  const t = await getTranslations({ locale, namespace: ns });

  const languages: Record<string, string> = {};
  for (const l of locales) languages[localeMeta[l].htmlLang] = localeUrl(l, path);
  languages["x-default"] = localeUrl(defaultLocale, path);

  const title = t("title");
  const description = t("description");
  let keywords: string[] | undefined;
  try { keywords = t("keywords").split(",").map((s) => s.trim()); } catch { keywords = undefined; }

  return {
    title,
    description,
    keywords,
    alternates: { canonical: localeUrl(locale, path), languages },
    openGraph: {
      title,
      description,
      url: localeUrl(locale, path),
      siteName: "Monadruk",
      type: "website",
      locale: localeMeta[locale].ogLocale,
      alternateLocale: locales.filter((l) => l !== locale).map((l) => localeMeta[l].ogLocale),
    },
    twitter: { card: "summary_large_image", title, description },
  };
}
