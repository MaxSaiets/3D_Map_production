import type { MetadataRoute } from "next";
import { BUSINESS } from "@/lib/legal";
import { locales, localeMeta, defaultLocale } from "@/i18n/routing";
import { CITY_PAGES, WORLD_CITY_PAGES } from "@/lib/cityPages";
import { BLOG_ARTICLES, blogLocales } from "@/lib/blog";
import { OCCASION_PAGES, DISTRICT_PAGES, districtLocales } from "@/lib/cityLanding";
import { GALLERY_ITEMS, GALLERY_LOCALES } from "@/lib/gallery";
import { PEAKS, PEAK_LOCALES } from "@/lib/mountainPages";
const PEAKS_LASTMOD = new Date("2026-09-25");

const BASE = "https://monadruk.com";
// Дата контенту хвилі city×product/occasion сторінок (2026-07-13) — окремо від
// STATIC_LASTMOD, щоб не сигналити «змінилось усе» для старих сторінок.
const WAVE2_LASTMOD = new Date("2026-07-13");
// Хвиля 4: міста Європи (нові сторінки під de/pl/fr/es).
const WAVE4_LASTMOD = new Date("2026-07-29");
// Хвиля 5 (2026-09-24): окрема сторінка на кожне фото галереї (/foto/[slug]), лише uk+en.
const GALLERY_LASTMOD = new Date("2026-09-24");
const LEGAL_LASTMOD = new Date(BUSINESS.updated); // до PATHS — інакше TDZ
const PATHS: { path: string; changeFrequency: MetadataRoute.Sitemap[number]["changeFrequency"]; priority: number; lastmod?: Date; only?: readonly string[] }[] = [
  { path: "", changeFrequency: "weekly", priority: 1.0 },
  { path: "/create", changeFrequency: "monthly", priority: 0.9 },
  { path: "/keychains", changeFrequency: "monthly", priority: 0.9 },
  { path: "/showcase", changeFrequency: "weekly", priority: 0.8 },
  { path: "/mountains", changeFrequency: "monthly", priority: 0.7 },
  { path: "/worlds", changeFrequency: "monthly", priority: 0.6 },
  { path: "/prices", changeFrequency: "monthly", priority: 0.7 },
  { path: "/maps", changeFrequency: "monthly", priority: 0.8 },
  { path: "/brelok", changeFrequency: "monthly", priority: 0.8 }, // індекс-хаб брелоків (keychain money-path)
  { path: "/panno", changeFrequency: "monthly", priority: 0.8 }, // лендінг «карта на стіну/панно» (найбільший кластер попиту, аудит 16.07)
  { path: "/3d-model-mista", changeFrequency: "monthly", priority: 0.8, lastmod: new Date("2026-09-25"), only: ["uk", "en"] }, // «купити 3D-модель міста / макет міста» (25.09.2026)
  { path: "/karpaty", changeFrequency: "monthly", priority: 0.8 }, // лендінг «рельєфна/топографічна мапа Карпат» (транзакційний кластер, аудит 16.07)
  { path: "/maket", changeFrequency: "monthly", priority: 0.8 }, // «макет квартири з плану» — новий сервіс, окремий пошуковий кластер
  { path: "/corporate", changeFrequency: "monthly", priority: 0.7 }, // B2B-лендінг «корпоративні подарунки/мерч» (аудит 16.07)
  { path: "/podarunok", changeFrequency: "monthly", priority: 0.7 },
  // Блог: індекс + статті (контент-глибина під інформаційні запити)
  { path: "/blog", changeFrequency: "weekly", priority: 0.6 },
  ...BLOG_ARTICLES.map((a) => ({
    path: `/blog/${a.slug}`,
    changeFrequency: "monthly" as const,
    priority: 0.6,
    lastmod: new Date(a.date), // дата публікації статті — точніша за глобальний STATIC_LASTMOD
    only: blogLocales(a), // лише локалі зі справжнім перекладом (решта noindex)
  })),
  // Programmatic SEO: сторінка під кожне місто (23 × 6 локалей). lastmod=WAVE2 —
  // сторінки допрацьовано 2026-07-13 (FAQ+факти+блог-лінки), не чіпаний June STATIC_LASTMOD.
  ...CITY_PAGES.map((c) => ({
    path: `/maps/${c.slug}`,
    changeFrequency: "monthly" as const,
    priority: 0.7,
    lastmod: WAVE2_LASTMOD,
  })),
  // Хвиля 4 (2026-07-29): міста ЄВРОПИ — контент під de/pl/fr/es-аудиторію.
  // Свіжий lastmod = сигнал «нове», priority 0.75 > українських, бо ці сторінки
  // ще не в індексі й ми хочемо, щоб краулер узяв їх першими.
  ...WORLD_CITY_PAGES.map((c) => ({
    path: `/maps/${c.slug}`,
    changeFrequency: "monthly" as const,
    priority: 0.75,
    lastmod: WAVE4_LASTMOD,
  })),
  // Хвиля 3 (2026-07-13): райони міст — найточніший рівень запиту
  ...DISTRICT_PAGES.map((d) => ({
    path: `/maps/${d.citySlug}/${d.slug}`,
    changeFrequency: "monthly" as const,
    priority: 0.65,
    lastmod: WAVE2_LASTMOD,
    only: districtLocales(d), // лише справжні переклади (решта noindex)
  })),
  // Хвиля 2 (2026-07-13): місто × продукт + лендінги під нагоду
  ...CITY_PAGES.map((c) => ({
    path: `/brelok/${c.slug}`,
    changeFrequency: "monthly" as const,
    priority: 0.6,
    lastmod: WAVE2_LASTMOD,
  })),
  ...CITY_PAGES.map((c) => ({
    path: `/podarunok/${c.slug}`,
    changeFrequency: "monthly" as const,
    priority: 0.6,
    lastmod: WAVE2_LASTMOD,
  })),
  ...OCCASION_PAGES.map((o) => ({
    path: `/podarunok/${o.slug}`,
    changeFrequency: "monthly" as const,
    priority: 0.7,
    lastmod: WAVE2_LASTMOD,
    only: districtLocales(o), // лише справжні переклади нагоди
  })),
  { path: "/delivery", changeFrequency: "monthly", priority: 0.4 },
  { path: "/contacts", changeFrequency: "yearly", priority: 0.3 },
  { path: "/offer", changeFrequency: "yearly", priority: 0.2 },
  { path: "/refund", changeFrequency: "yearly", priority: 0.2 },
  // J-1: юр-тексти переписані 2026-09-05 → lastmod з BUSINESS.updated (єдине джерело дати).
  { path: "/privacy", changeFrequency: "yearly", priority: 0.2, lastmod: LEGAL_LASTMOD },
  { path: "/terms", changeFrequency: "yearly", priority: 0.2, lastmod: LEGAL_LASTMOD },
];

function url(locale: string, path: string) {
  return locale === defaultLocale ? `${BASE}${path || "/"}` : `${BASE}/${locale}${path}`;
}

// Статичні сторінки (міста, юр-доки) НЕ оновлюються щодеплою — даємо фіксовану дату
// контенту, щоб не слати Google хибний сигнал «змінилось усе» на кожен білд. Лише
// справді динамічні сторінки отримують now. Оновлювати STATIC_LASTMOD при зміні контенту.
const STATIC_LASTMOD = new Date("2026-06-21");
const DYNAMIC_PATHS = new Set(["", "/create", "/keychains", "/showcase"]);

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();
  const entries: MetadataRoute.Sitemap = [];
  for (const { path, changeFrequency, priority, lastmod, only } of PATHS) {
    const ls = only ? locales.filter((l) => only.includes(l)) : locales;
    const languages: Record<string, string> = {};
    for (const l of ls) languages[localeMeta[l].htmlLang] = url(l, path);
    languages["x-default"] = url(defaultLocale, path); // консистентно з per-page hreflang
    for (const l of ls) {
      entries.push({
        url: url(l, path),
        lastModified: DYNAMIC_PATHS.has(path) ? now : lastmod ?? STATIC_LASTMOD,
        changeFrequency,
        priority: l === defaultLocale ? priority : Math.max(0.1, priority - 0.1),
        alternates: { languages },
      });
    }
  }
  // /foto/[slug]: лише uk+en (інші локалі noindex → не в сайтмапі).
  for (const g of GALLERY_ITEMS) {
    const path = `/foto/${g.slug}`;
    const languages: Record<string, string> = { "x-default": url(defaultLocale, path) };
    for (const l of GALLERY_LOCALES) languages[localeMeta[l].htmlLang] = url(l, path);
    for (const l of GALLERY_LOCALES) {
      entries.push({
        url: url(l, path),
        lastModified: GALLERY_LASTMOD,
        changeFrequency: "monthly",
        priority: l === defaultLocale ? 0.55 : 0.45,
        alternates: { languages },
      });
    }
  }
  // /gory і /gory/[slug] (25.09.2026): усі 6 мов — справжні тексти.
  for (const path of ["/gory", ...PEAKS.map((p) => `/gory/${p.slug}`)]) {
    const languages: Record<string, string> = { "x-default": url(defaultLocale, path) };
    for (const l of PEAK_LOCALES) languages[localeMeta[l].htmlLang] = url(l, path);
    for (const l of PEAK_LOCALES) {
      entries.push({
        url: url(l, path),
        lastModified: PEAKS_LASTMOD,
        changeFrequency: "monthly",
        priority: path === "/gory" ? (l === defaultLocale ? 0.7 : 0.6) : l === defaultLocale ? 0.6 : 0.5,
        alternates: { languages },
      });
    }
  }
  return entries;
}
