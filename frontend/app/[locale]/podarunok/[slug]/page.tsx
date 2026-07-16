import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { setRequestLocale, getTranslations } from "next-intl/server";
import { BASE, localeUrl, priceValidUntil } from "@/i18n/metadata";
import { routing, locales, localeMeta, defaultLocale, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { CITY_PAGES, CITY_PAGE_BY_SLUG } from "@/lib/cityPages";
import { cityFacts, CITY_FACTS } from "@/lib/cityFacts";
import { mapPriceRange, KEYCHAIN_PRICE_UAH, mapPriceEur } from "@/lib/mapPrices";
import {
  giftCityCopy,
  contentLocale,
  cityFaq,
  occasionFaq,
  OCCASION_PAGES,
  OCCASION_BY_SLUG,
  landingCopy,
  type CityLandingCopy,
} from "@/lib/cityLanding";
import { BLOG_ARTICLES, blogLocale } from "@/lib/blog";

/**
 * Programmatic SEO рівень 2: подарунок × місто (23) + лендінги під нагоду (5).
 * Один динамічний роут обробляє обидва типи слагів: спершу нагоди
 * (OCCASION_BY_SLUG), інакше місто (CITY_PAGE_BY_SLUG). Запити виду
 * «подарунок київ», «подарунок на річницю», «подарунок на новосілля».
 */
export function generateStaticParams() {
  return [
    ...OCCASION_PAGES.map((o) => ({ slug: o.slug })),
    ...CITY_PAGES.map((c) => ({ slug: c.slug })),
  ];
}

export const dynamicParams = false;

function resolveCopy(slug: string, locale: AppLocale): { c: CityLandingCopy; isCity: boolean; cityName?: string } | null {
  const cl = contentLocale(locale);
  const occ = OCCASION_BY_SLUG[slug];
  if (occ) return { c: landingCopy(occ.content, cl), isCity: false };
  const city = CITY_PAGE_BY_SLUG[slug];
  if (city) {
    const name = city.names[locale];
    return { c: giftCityCopy(cl, name, slug, cityFacts(slug)), isCity: true, cityName: name };
  }
  return null;
}

export async function generateMetadata({
  params,
}: {
  params: { locale: string; slug: string };
}): Promise<Metadata> {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  const r = resolveCopy(params.slug, locale);
  if (!r) return {};
  const path = `/podarunok/${params.slug}`;
  const languages: Record<string, string> = {};
  for (const l of locales) languages[localeMeta[l].htmlLang] = localeUrl(l, path);
  languages["x-default"] = localeUrl(defaultLocale, path);
  return {
    title: r.c.title,
    description: r.c.description,
    alternates: { canonical: localeUrl(locale, path), languages },
    openGraph: {
      title: r.c.title,
      description: r.c.description,
      url: localeUrl(locale, path),
      siteName: "Monadruk",
      type: "website",
      locale: localeMeta[locale].ogLocale,
      images: [`${BASE}/opengraph-image`],
    },
    twitter: { card: "summary_large_image", title: r.c.title, description: r.c.description, images: [`${BASE}/opengraph-image`] },
  };
}

export default async function GiftSlugPage({
  params,
}: {
  params: { locale: string; slug: string };
}) {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  const r = resolveCopy(params.slug, locale);
  if (!r) notFound();
  setRequestLocale(locale);
  const t = await getTranslations({ locale, namespace: "cityPages" });
  const isUA = locale === "uk";
  const { c, isCity } = r;
  const path = `/podarunok/${params.slug}`;
  const city = isCity ? CITY_PAGE_BY_SLUG[params.slug] : undefined;
  const occ = !isCity ? OCCASION_BY_SLUG[params.slug] : undefined;
  const facts = city ? cityFacts(city.slug) : undefined;
  const faq = city ? cityFaq(contentLocale(locale), r.cityName ?? "", "podarunok") : occasionFaq(contentLocale(locale));
  const nf = new Intl.NumberFormat(locale === "uk" ? "uk-UA" : locale);
  const pn = (o: { uk: string; latin: string }) => (locale === "uk" ? o.uk : o.latin);

  const range = mapPriceRange(locale);
  // Сторінка подарунка рекламує «від 120 ₴» (брелок — найдешевший SKU), тож
  // AggregateOffer.lowPrice МУСИТЬ = ціні брелка, а не мапи (250) — інакше
  // structured-data суперечить видимій ціні → Google Merchant price-mismatch.
  const giftLowPrice = locale === "uk" ? String(KEYCHAIN_PRICE_UAH) : String(mapPriceEur(KEYCHAIN_PRICE_UAH));
  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Product",
        name: c.h1,
        description: c.description,
        image: `${BASE}/showcase/map-1.webp`,
        brand: { "@type": "Brand", name: "Monadruk" },
        sku: `MND-GIFT-${params.slug}`,
        offers: {
          "@type": "AggregateOffer",
          priceCurrency: range.currency,
          lowPrice: giftLowPrice,
          highPrice: range.high,
          offerCount: range.offerCount,
          priceValidUntil: priceValidUntil(),
          availability: "https://schema.org/InStock",
          url: localeUrl(locale, path),
        },
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: isUA ? "Подарунки" : "Gifts", item: localeUrl(locale, "/podarunok") },
          { "@type": "ListItem", position: 3, name: c.h1, item: localeUrl(locale, path) },
        ],
      },
      {
        "@type": "FAQPage",
        mainEntity: faq.map((f) => ({
          "@type": "Question",
          name: f.q,
          acceptedAnswer: { "@type": "Answer", text: f.a },
        })),
      },
    ],
  };

  // Перелінковка: для міст — сусідні gift-сторінки; для нагод — усі інші нагоди.
  const hav = (a: readonly [number, number], b: readonly [number, number]) => {
    const R = 6371, toRad = (d: number) => (d * Math.PI) / 180;
    const dLat = toRad(b[0] - a[0]), dLng = toRad(b[1] - a[1]);
    const s = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(a[0])) * Math.cos(toRad(b[0])) * Math.sin(dLng / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(s));
  };
  const otherCities = city
    ? [...CITY_PAGES.filter((x) => x.slug !== city.slug)]
        .sort((a, b) => hav(city.center, a.center) - hav(city.center, b.center))
        .slice(0, 8)
        .concat(
          [...CITY_PAGES.filter((x) => x.slug !== city.slug)]
            .sort((a, b) => (CITY_FACTS[b.slug]?.population ?? 0) - (CITY_FACTS[a.slug]?.population ?? 0))
            .slice(0, 3),
        )
        .filter((x, i, arr) => arr.findIndex((y) => y.slug === x.slug) === i)
    : [];
  const otherOccasions = OCCASION_PAGES.filter((o) => o.slug !== params.slug);

  const ui = isUA
    ? {
        breadcrumb: "Подарунки",
        ctaMap: "Створити 3D-мапу",
        ctaKeychain: "Створити брелок",
        othersCity: "Подарунки з інших міст",
        occasions: "Подарунки під нагоду",
        cityLinks: city ? { map: `3D-мапа міста — ${r.cityName}`, brelok: `Брелок з картою — ${r.cityName}` } : null,
      }
    : {
        breadcrumb: "Gifts",
        ctaMap: "Create a 3D map",
        ctaKeychain: "Create a keychain",
        othersCity: "Gifts from other cities",
        occasions: "Gifts by occasion",
        cityLinks: city ? { map: `3D city map — ${r.cityName}`, brelok: `Map keychain — ${r.cityName}` } : null,
      };

  const ctaHref = occ?.ctaHref ?? (city ? `/create?city=${city.key}` : "/create");

  return (
    <main id="main-content" tabIndex={-1} className="mx-auto max-w-[820px] px-5 py-14 lg:py-20">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      <nav className="text-[13px] text-ink-3" aria-label="breadcrumb">
        <Link href="/" className="hover:underline">Monadruk</Link>
        {" / "}
        <Link href="/podarunok" className="hover:underline">{ui.breadcrumb}</Link>
        {" / "}
        <span className="text-ink">{r.cityName ?? c.h1}</span>
      </nav>
      <h1 className="mt-5 text-[clamp(28px,4vw,46px)] leading-tight">{c.h1}</h1>
      {c.intro.map((p, i) => (
        <p key={i} className={`${i === 0 ? "mt-5" : "mt-3"} text-[15px] leading-relaxed text-ink-2`}>{p}</p>
      ))}

      <div className="mt-8 flex flex-wrap gap-3">
        <Link
          href={ctaHref}
          className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-6 py-3 text-sm font-semibold text-white transition hover:opacity-90"
        >
          {ui.ctaMap}
        </Link>
        <Link
          href="/keychains"
          className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] border border-line-soft bg-white/80 px-6 py-3 text-sm font-semibold text-ink transition hover:border-[var(--accent)]"
        >
          {ui.ctaKeychain}
        </Link>
      </div>

      {c.sections.map((s, i) => (
        <section key={i} className="mt-10">
          <h2 className="text-[20px] font-semibold">{s.h2}</h2>
          {s.p.map((para, j) => (
            <p key={j} className="mt-3 text-[15px] leading-relaxed text-ink-2">{para}</p>
          ))}
        </section>
      ))}

      {/* Факти про місто (лише для city-варіанту) — анти-doorway. */}
      {facts && city && (
        <section className="mt-10 rounded-[18px] border border-line-soft bg-white/60 px-5 py-5">
          <h2 className="text-[16px] font-semibold text-ink">
            {t("factsTitle", { city: r.cityName })}
          </h2>
          <dl className="mt-3 grid gap-x-7 gap-y-1.5 text-[14px] sm:grid-cols-2">
            {[
              [t("fPopulation"), `${nf.format(facts.population)} (${facts.populationYear})`],
              [facts.firstMention ? t("fFirstMention") : t("fFounded"), String(facts.founded)],
              [t("fRiver"), pn(facts.river)],
              [t("fLandmark"), pn(facts.landmark)],
            ].map(([label, value]) => (
              <div key={label} className="flex items-baseline justify-between gap-3 border-b border-line-soft/50 py-1">
                <dt className="text-ink-3">{label}</dt>
                <dd className="text-right font-semibold text-ink">{value}</dd>
              </div>
            ))}
          </dl>
        </section>
      )}

      {/* FAQ (видимий, +FAQPage JSON-LD вище). */}
      <section className="mt-10">
        <h2 className="text-[20px] font-semibold">{isUA ? "Часті запитання" : "FAQ"}</h2>
        <dl className="mt-4 flex flex-col gap-4">
          {faq.map((f) => (
            <div key={f.q}>
              <dt className="text-[15px] font-semibold text-ink">{f.q}</dt>
              <dd className="mt-1.5 text-[14.5px] leading-relaxed text-ink-2">{f.a}</dd>
            </div>
          ))}
        </dl>
      </section>

      {ui.cityLinks && city && (
        <section className="mt-12 flex flex-col gap-2 rounded-[18px] border border-line-soft bg-white/60 px-5 py-5">
          <Link href={`/maps/${city.slug}`} className="text-[14.5px] font-semibold text-[var(--accent-strong)] hover:underline">
            {ui.cityLinks.map} →
          </Link>
          <Link href={`/brelok/${city.slug}`} className="text-[14.5px] font-semibold text-[var(--accent-strong)] hover:underline">
            {ui.cityLinks.brelok} →
          </Link>
        </section>
      )}

      {/* Читати далі: 2 статті блогу — контент↔продукт перелінковка. */}
      <section className="mt-10">
        <h2 className="text-[18px] font-semibold text-ink">{t("readMore")}</h2>
        <ul className="mt-3 flex flex-col gap-2">
          {BLOG_ARTICLES.slice(2, 4).map((a) => (
            <li key={a.slug}>
              <Link href={`/blog/${a.slug}`} className="text-[14.5px] font-medium text-[var(--accent-strong)] hover:underline">
                {a.content[blogLocale(locale)].h1} →
              </Link>
            </li>
          ))}
        </ul>
      </section>

      {/* Нагоди — з міської сторінки і з іншої нагоди (перелінковка кластера). */}
      <section className="mt-12">
        <h2 className="text-[20px] font-semibold">{ui.occasions}</h2>
        <ul className="mt-4 flex flex-wrap gap-2">
          {otherOccasions.map((o) => (
            <li key={o.slug}>
              <Link
                href={`/podarunok/${o.slug}`}
                className="inline-block rounded-full border border-line-soft bg-white/70 px-4 py-2 text-[13.5px] font-medium text-ink-2 transition hover:border-[var(--accent)] hover:text-ink"
              >
                {landingCopy(o.content, contentLocale(locale)).h1}
              </Link>
            </li>
          ))}
        </ul>
      </section>

      {otherCities.length > 0 && (
        <section className="mt-10">
          <h2 className="text-[20px] font-semibold">{ui.othersCity}</h2>
          <ul className="mt-4 flex flex-wrap gap-2">
            {otherCities.map((x) => (
              <li key={x.slug}>
                <Link
                  href={`/podarunok/${x.slug}`}
                  className="inline-block rounded-full border border-line-soft bg-white/70 px-4 py-2 text-[13.5px] font-medium text-ink-2 transition hover:border-[var(--accent)] hover:text-ink"
                >
                  {x.names[locale]}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}
