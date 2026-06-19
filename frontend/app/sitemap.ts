import type { MetadataRoute } from "next";
import { locales, localeMeta, defaultLocale } from "@/i18n/routing";
import { CITY_PAGES } from "@/lib/cityPages";

const BASE = "https://monadruk.com";
const PATHS: { path: string; changeFrequency: MetadataRoute.Sitemap[number]["changeFrequency"]; priority: number }[] = [
  { path: "", changeFrequency: "weekly", priority: 1.0 },
  { path: "/create", changeFrequency: "monthly", priority: 0.9 },
  { path: "/keychains", changeFrequency: "monthly", priority: 0.9 },
  { path: "/showcase", changeFrequency: "weekly", priority: 0.8 },
  { path: "/prices", changeFrequency: "monthly", priority: 0.7 },
  { path: "/maps", changeFrequency: "monthly", priority: 0.8 },
  // Programmatic SEO: сторінка під кожне місто (23 × 6 локалей)
  ...CITY_PAGES.map((c) => ({
    path: `/maps/${c.slug}`,
    changeFrequency: "monthly" as const,
    priority: 0.7,
  })),
  { path: "/delivery", changeFrequency: "monthly", priority: 0.4 },
  { path: "/contacts", changeFrequency: "yearly", priority: 0.3 },
  { path: "/offer", changeFrequency: "yearly", priority: 0.2 },
  { path: "/refund", changeFrequency: "yearly", priority: 0.2 },
  { path: "/privacy", changeFrequency: "yearly", priority: 0.2 },
  { path: "/terms", changeFrequency: "yearly", priority: 0.2 },
];

function url(locale: string, path: string) {
  return locale === defaultLocale ? `${BASE}${path || "/"}` : `${BASE}/${locale}${path}`;
}

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();
  const entries: MetadataRoute.Sitemap = [];
  for (const { path, changeFrequency, priority } of PATHS) {
    const languages: Record<string, string> = {};
    for (const l of locales) languages[localeMeta[l].htmlLang] = url(l, path);
    for (const l of locales) {
      entries.push({
        url: url(l, path),
        lastModified: now,
        changeFrequency,
        priority: l === defaultLocale ? priority : Math.max(0.1, priority - 0.1),
        alternates: { languages },
      });
    }
  }
  return entries;
}
