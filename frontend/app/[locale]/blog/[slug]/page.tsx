import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { setRequestLocale } from "next-intl/server";
import { BASE, localeUrl } from "@/i18n/metadata";
import { routing, locales, localeMeta, defaultLocale, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { BLOG_ARTICLES, BLOG_BY_SLUG, blogContent, blogIndexMeta } from "@/lib/blog";
import { MAP_CITY_PAGES, MAP_CITY_PAGE_BY_SLUG } from "@/lib/cityPages";

// Тематичні міста для гео-таргетованих статей (ЄС-хвиля): чипи ведуть на
// релевантні /maps/{місто}, а не на ротаційний зріз. Решта статей — ротація.
const PREFERRED_CITY_LINKS: Record<string, string[]> = {
  "3d-karte-deutsche-staedte": ["berlin", "munich", "hamburg", "cologne", "vienna", "zurich"],
  "mapa-3d-polskich-miast": ["warsaw", "krakow", "wroclaw", "gdansk", "poznan", "lodz"],
};

/**
 * Сторінка статті блогу: Article JSON-LD + breadcrumbs + внутрішні лінки
 * (CTA у конструктор/каталог). Статика (SSG) — контент з lib/blog.ts.
 */

export function generateStaticParams() {
  return BLOG_ARTICLES.map((a) => ({ slug: a.slug }));
}

export const dynamicParams = false;

export async function generateMetadata({
  params,
}: {
  params: { locale: string; slug: string };
}): Promise<Metadata> {
  const article = BLOG_BY_SLUG[params.slug];
  if (!article) return {};
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  const c = blogContent(article, locale);
  const path = `/blog/${article.slug}`;
  const languages: Record<string, string> = {};
  for (const l of locales) languages[localeMeta[l].htmlLang] = localeUrl(l, path);
  languages["x-default"] = localeUrl(defaultLocale, path);
  return {
    title: c.title,
    description: c.description,
    alternates: { canonical: localeUrl(locale, path), languages },
    openGraph: {
      title: c.title,
      description: c.description,
      url: localeUrl(locale, path),
      siteName: "Monadruk",
      type: "article",
      images: [`${BASE}/opengraph-image`],
    },
  };
}

export default async function BlogArticlePage({
  params,
}: {
  params: { locale: string; slug: string };
}) {
  const article = BLOG_BY_SLUG[params.slug];
  if (!article) notFound();
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  setRequestLocale(locale);
  const c = blogContent(article, locale);
  const m = blogIndexMeta(locale);
  const path = `/blog/${article.slug}`;

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "BlogPosting",
        headline: c.h1,
        description: c.description,
        datePublished: article.date,
        dateModified: article.date,
        inLanguage: localeMeta[locale].htmlLang,
        author: { "@type": "Organization", name: "Monadruk", url: BASE },
        publisher: { "@id": `${BASE}/#org` },
        mainEntityOfPage: localeUrl(locale, path),
        image: `${BASE}/real/map-1.webp`,
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: m.h1, item: localeUrl(locale, "/blog") },
          { "@type": "ListItem", position: 3, name: c.h1, item: localeUrl(locale, path) },
        ],
      },
    ],
  };

  return (
    <main id="main-content" tabIndex={-1} className="mx-auto max-w-[760px] px-5 py-14 lg:py-20">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      <nav className="text-[13px] text-ink-3" aria-label="breadcrumb">
        <Link href="/" className="hover:underline">Monadruk</Link>
        {" / "}
        <Link href="/blog" className="hover:underline">{m.h1}</Link>
      </nav>
      <h1 className="mt-5 text-[clamp(26px,3.6vw,40px)] leading-tight">{c.h1}</h1>
      <p className="mt-5 text-[15.5px] leading-relaxed text-ink-2">{c.intro}</p>

      {c.sections.map((s, i) => (
        <section key={i} className="mt-8">
          {s.h2 ? <h2 className="text-[20px] font-semibold text-ink">{s.h2}</h2> : null}
          {s.p.map((para, j) => (
            <p key={j} className="mt-3 text-[15px] leading-relaxed text-ink-2">{para}</p>
          ))}
        </section>
      ))}

      <div className="mt-10">
        <Link
          href={c.ctaHref}
          className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-6 py-3 text-sm font-semibold text-white transition hover:opacity-90"
        >
          {c.ctaLabel}
        </Link>
      </div>

      {c.outro ? (
        <section className="mt-10 rounded-[18px] border border-line-soft bg-white/60 px-5 py-5">
          <p className="text-[14.5px] leading-relaxed text-ink-2">{c.outro}</p>
        </section>
      ) : null}

      {/* Перелінковка блог → міста (SEO_PLAN етап 1): кожна стаття лінкує СВОЇ
          6 міст — ротація за індексом статті по ВСІХ /maps (23 УА + 64 ЄС), щоб
          лінки рівномірно розійшлись і Google діставав сторінки міст з контенту,
          а не лише з sitemap. Детермінований вибір → стабільний SSG-вивід. */}
      <section className="mt-12">
        <h2 className="text-[18px] font-semibold text-ink">
          {locale === "uk" ? "3D-мапи міст" : "3D city maps"}
        </h2>
        <ul className="mt-4 flex flex-wrap gap-2">
          {(() => {
            const preferred = PREFERRED_CITY_LINKS[article.slug]
              ?.map((s) => MAP_CITY_PAGE_BY_SLUG[s])
              .filter(Boolean);
            if (preferred?.length) return preferred;
            const idx = Math.max(0, BLOG_ARTICLES.findIndex((a) => a.slug === article.slug));
            const start = (idx * 6) % MAP_CITY_PAGES.length;
            return [0, 1, 2, 3, 4, 5].map((k) => MAP_CITY_PAGES[(start + k) % MAP_CITY_PAGES.length]);
          })().map((c) => (
            <li key={c.slug}>
              <Link
                href={`/maps/${c.slug}`}
                className="inline-block rounded-full border border-line-soft bg-white/70 px-4 py-2 text-[13.5px] font-medium text-ink-2 transition hover:border-[var(--accent)] hover:text-ink"
              >
                {c.names[locale]}
              </Link>
            </li>
          ))}
        </ul>
      </section>

      {/* Інші статті — внутрішня перелінковка */}
      <section className="mt-12">
        <h2 className="text-[18px] font-semibold text-ink">{m.h1}</h2>
        <ul className="mt-4 grid gap-2">
          {BLOG_ARTICLES.filter((a) => a.slug !== article.slug).map((a) => (
            <li key={a.slug}>
              <Link href={`/blog/${a.slug}`} className="text-[14.5px] font-medium text-[var(--accent-strong)] hover:underline">
                {blogContent(a, locale).h1} →
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
