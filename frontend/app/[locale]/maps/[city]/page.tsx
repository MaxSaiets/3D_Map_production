import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { BASE, localeUrl } from "@/i18n/metadata";
import { routing, locales, localeMeta, defaultLocale, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { CITY_PAGES, CITY_PAGE_BY_SLUG } from "@/lib/cityPages";

/**
 * Programmatic SEO: статична сторінка під кожне місто (23 × 6 локалей).
 * Запити виду «3д мапа Львова», «brelok z mapą Lwowa», «3D-Karte von Kiew».
 */
export function generateStaticParams() {
  return CITY_PAGES.map((c) => ({ city: c.slug }));
}

export const dynamicParams = false;

export async function generateMetadata({
  params,
}: {
  params: { locale: string; city: string };
}): Promise<Metadata> {
  const city = CITY_PAGE_BY_SLUG[params.city];
  if (!city) return {};
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  const t = await getTranslations({ locale, namespace: "cityPages" });
  const name = city.names[locale];
  const path = `/maps/${city.slug}`;

  const languages: Record<string, string> = {};
  for (const l of locales) languages[localeMeta[l].htmlLang] = localeUrl(l, path);
  languages["x-default"] = localeUrl(defaultLocale, path);

  const title = t("title", { city: name });
  const description = t("description", { city: name });
  return {
    title,
    description,
    alternates: { canonical: localeUrl(locale, path), languages },
    openGraph: {
      title,
      description,
      url: localeUrl(locale, path),
      siteName: "Monadruk",
      type: "website",
      locale: localeMeta[locale].ogLocale,
    },
    twitter: { card: "summary_large_image", title, description },
  };
}

export default async function CityPage({
  params,
}: {
  params: { locale: string; city: string };
}) {
  const city = CITY_PAGE_BY_SLUG[params.city];
  if (!city) notFound();
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  setRequestLocale(locale);
  const t = await getTranslations({ locale, namespace: "cityPages" });
  const name = city.names[locale];

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Product",
        name: t("title", { city: name }),
        description: t("description", { city: name }),
        image: `${BASE}/showcase/map-1.png`,
        brand: { "@type": "Brand", name: "Monadruk" },
        offers: {
          "@type": "Offer",
          priceCurrency: locale === "uk" ? "UAH" : "EUR",
          price: locale === "uk" ? "290" : "7",
          availability: "https://schema.org/InStock",
          url: localeUrl(locale, `/maps/${city.slug}`),
        },
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: t("breadcrumb"), item: localeUrl(locale, "/maps") },
          { "@type": "ListItem", position: 3, name, item: localeUrl(locale, `/maps/${city.slug}`) },
        ],
      },
    ],
  };

  const others = CITY_PAGES.filter((c) => c.slug !== city.slug).slice(0, 12);

  return (
    <main className="mx-auto max-w-[820px] px-5 py-14 lg:py-20">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      <nav className="text-[13px] text-ink-3" aria-label="breadcrumb">
        <Link href="/" className="hover:underline">Monadruk</Link>
        {" / "}
        <Link href="/maps" className="hover:underline">{t("breadcrumb")}</Link>
        {" / "}
        <span className="text-ink">{name}</span>
      </nav>
      <h1 className="mt-5 text-[clamp(28px,4vw,46px)] leading-tight">{t("h1", { city: name })}</h1>
      <p className="mt-5 text-[15px] leading-relaxed text-ink-2">{t("p1", { city: name })}</p>
      <p className="mt-3 text-[15px] leading-relaxed text-ink-2">{t("p2", { city: name })}</p>
      <div className="mt-8 flex flex-wrap gap-3">
        <Link
          href={`/create?city=${city.key}`}
          className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-6 py-3 text-sm font-semibold text-white transition hover:opacity-90"
        >
          {t("ctaMap", { city: name })}
        </Link>
        <Link
          href={`/keychains?city=${city.key}`}
          className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] border border-line-soft bg-white/80 px-6 py-3 text-sm font-semibold text-ink transition hover:border-[var(--accent)]"
        >
          {t("ctaKeychain", { city: name })}
        </Link>
      </div>
      <h2 className="mt-14 text-[20px] font-semibold">{t("others")}</h2>
      <ul className="mt-4 flex flex-wrap gap-2">
        {others.map((c) => (
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
    </main>
  );
}
