import type { Metadata } from "next";
import { setRequestLocale } from "next-intl/server";
import { localeUrl } from "@/i18n/metadata";
import { routing, locales, localeMeta, defaultLocale, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { BLOG_ARTICLES, blogContent, blogIndexMeta } from "@/lib/blog";

/**
 * Блог-індекс: контент-глибина під інформаційні запити (top-of-funnel SEO).
 * Контент у lib/blog.ts (uk+en; інші локалі → en). Той самий верстальний
 * патерн, що /podarunok та /maps/[city].
 */

export async function generateMetadata({
  params,
}: {
  params: { locale: string };
}): Promise<Metadata> {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  const m = blogIndexMeta(locale);
  const path = "/blog";
  const languages: Record<string, string> = {};
  for (const l of locales) languages[localeMeta[l].htmlLang] = localeUrl(l, path);
  languages["x-default"] = localeUrl(defaultLocale, path);
  return {
    title: m.title,
    description: m.description,
    alternates: { canonical: localeUrl(locale, path), languages },
    openGraph: { title: m.title, description: m.description, url: localeUrl(locale, path), siteName: "Monadruk", type: "website" },
  };
}

export default async function BlogIndexPage({
  params,
}: {
  params: { locale: string };
}) {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  setRequestLocale(locale);
  const m = blogIndexMeta(locale);

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Blog",
        name: m.h1,
        description: m.description,
        url: localeUrl(locale, "/blog"),
        blogPost: BLOG_ARTICLES.map((a) => ({
          "@type": "BlogPosting",
          headline: blogContent(a, locale).h1,
          datePublished: a.date,
          url: localeUrl(locale, `/blog/${a.slug}`),
        })),
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: m.h1, item: localeUrl(locale, "/blog") },
        ],
      },
    ],
  };

  return (
    <main id="main-content" tabIndex={-1} className="mx-auto max-w-[820px] px-5 py-14 lg:py-20">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      <nav className="text-[13px] text-ink-3" aria-label="breadcrumb">
        <Link href="/" className="hover:underline">Monadruk</Link>
        {" / "}
        <span className="text-ink">{m.h1}</span>
      </nav>
      <h1 className="mt-5 text-[clamp(28px,4vw,46px)] leading-tight">{m.h1}</h1>
      <p className="mt-5 text-[15px] leading-relaxed text-ink-2">{m.intro}</p>

      <ul className="mt-10 grid gap-5">
        {BLOG_ARTICLES.map((a) => {
          const c = blogContent(a, locale);
          return (
            <li key={a.slug}>
              <Link
                href={`/blog/${a.slug}`}
                className="block rounded-[18px] border border-line-soft bg-white/60 px-6 py-6 transition hover:border-[var(--accent)]"
              >
                <h2 className="text-[19px] font-semibold leading-snug text-ink">{c.h1}</h2>
                <p className="mt-2 text-[14px] leading-relaxed text-ink-2">{c.description}</p>
                <span className="mt-3 inline-block text-[13.5px] font-semibold text-[var(--accent-strong)]">
                  {m.readLabel} →
                </span>
              </Link>
            </li>
          );
        })}
      </ul>
    </main>
  );
}
