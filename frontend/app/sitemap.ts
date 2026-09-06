import type { MetadataRoute } from "next";
import { BUSINESS } from "@/lib/legal";
import { locales, localeMeta, defaultLocale } from "@/i18n/routing";
import { CITY_PAGES, WORLD_CITY_PAGES } from "@/lib/cityPages";
import { BLOG_ARTICLES } from "@/lib/blog";
import { OCCASION_PAGES, DISTRICT_PAGES } from "@/lib/cityLanding";

const BASE = "https://monadruk.com";
// Дата контенту хвилі city×product/occasion сторінок (2026-07-13) — окремо від
// STATIC_LASTMOD, щоб не сигналити «змінилось усе» для старих сторінок.
const WAVE2_LASTMOD = new Date("2026-07-13");
// Хвиля 4: міста Європи (нові сторінки під de/pl/fr/es).
const WAVE4_LASTMOD = new Date("2026-07-29");
const LEGAL_LASTMOD = new Date(BUSINESS.updated); // до PATHS — інакше TDZ
const PATHS: { path: string; changeFrequency: MetadataRoute.Sitemap[number]["changeFrequency"]; priority: number; lastmod?: Date }[] = [
  { path: "", changeFrequency: "weekly", priority: 1.0 },
  { path: "/create", changeFrequency: "monthly", priority: 0.9 },
  { path: "/keychains", changeFrequency: "monthly", priority: 0.9 },
  { path: "/showcase", changeFrequency: "weekly", priority: 0.8 },
  { path: "/worlds", changeFrequency: "monthly", priority: 0.6 },
  { path: "/prices", changeFrequency: "monthly", priority: 0.7 },
  { path: "/maps", changeFrequency: "monthly", priority: 0.8 },
  { path: "/brelok", changeFrequency: "monthly", priority: 0.8 }, // індекс-хаб брелоків (keychain money-path)
  { path: "/panno", changeFrequency: "monthly", priority: 0.8 }, // лендінг «карта на стіну/панно» (найбільший кластер попиту, аудит 16.07)
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
  for (const { path, changeFrequency, priority, lastmod } of PATHS) {
    const languages: Record<string, string> = {};
    for (const l of locales) languages[localeMeta[l].htmlLang] = url(l, path);
    languages["x-default"] = url(defaultLocale, path); // консистентно з per-page hreflang
    for (const l of locales) {
      entries.push({
        url: url(l, path),
        lastModified: DYNAMIC_PATHS.has(path) ? now : lastmod ?? STATIC_LASTMOD,
        changeFrequency,
        priority: l === defaultLocale ? priority : Math.max(0.1, priority - 0.1),
        alternates: { languages },
      });
    }
  }
  return entries;
}
