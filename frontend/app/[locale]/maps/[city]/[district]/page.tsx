import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { setRequestLocale } from "next-intl/server";
import { BASE, localeUrl, priceValidUntil } from "@/i18n/metadata";
import { routing, locales, localeMeta, defaultLocale, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { CITY_PAGE_BY_SLUG } from "@/lib/cityPages";
import { MAP_TEMPLATES } from "@/lib/templates";
import { mapPriceRange } from "@/lib/mapPrices";
import { DISTRICT_PAGES, DISTRICT_BY_CITY_SLUG, contentLocale, cityFaq, landingCopy } from "@/lib/cityLanding";

/**
 * Programmatic SEO рівень 3: район міста (/maps/[city]/[district], 12 × 6 локалей).
 * Найточніший запит («3d мапа поділу», «мапа площі ринок львів») — найвища
 * конверсія. Контент — ручний, з lib/cityLanding.ts DISTRICT_PAGES (12 записів,
 * дзеркалять MAP_TEMPLATES із lib/templates.ts). Deep-link на готову сцену
 * конструктора: /create?template={templateId}.
 */
export function generateStaticParams() {
  return DISTRICT_PAGES.map((d) => ({ city: d.citySlug, district: d.slug }));
}

export const dynamicParams = false;

function resolve(citySlug: string, districtSlug: string) {
  const district = DISTRICT_PAGES.find((d) => d.citySlug === citySlug && d.slug === districtSlug);
  if (!district) return null;
  const city = CITY_PAGE_BY_SLUG[citySlug];
  if (!city) return null;
  const tpl = MAP_TEMPLATES.find((t) => t.id === district.templateId);
  if (!tpl) return null;
  return { district, city, tpl };
}

export async function generateMetadata({
  params,
}: {
  params: { locale: string; city: string; district: string };
}): Promise<Metadata> {
  const r = resolve(params.city, params.district);
  if (!r) return {};
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  const c = landingCopy(r.district.content, contentLocale(locale));
  const path = `/maps/${r.city.slug}/${r.district.slug}`;
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
      type: "website",
      locale: localeMeta[locale].ogLocale,
      images: [`${BASE}/opengraph-image`],
    },
    twitter: { card: "summary_large_image", title: c.title, description: c.description, images: [`${BASE}/opengraph-image`] },
  };
}

export default async function DistrictPage({
  params,
}: {
  params: { locale: string; city: string; district: string };
}) {
  const r = resolve(params.city, params.district);
  if (!r) notFound();
  const { district, city, tpl } = r;
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  setRequestLocale(locale);
  const isUA = locale === "uk";
  const cl = contentLocale(locale);
  const c = landingCopy(district.content, cl);
  const cityName = city.names[locale];
  const districtName = isUA ? tpl.district : district.enName;
  const path = `/maps/${city.slug}/${district.slug}`;
  const faq = cityFaq(cl, cityName, "podarunok");

  const range = mapPriceRange(locale);
  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Product",
        name: c.h1,
        description: c.description,
        image: `${BASE}/real/map-1.webp`,
        brand: { "@type": "Brand", name: "Monadruk" },
        sku: `MND-DISTRICT-${district.templateId}`,
        offers: {
          "@type": "AggregateOffer",
          priceCurrency: range.currency,
          lowPrice: range.low,
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
          { "@type": "ListItem", position: 2, name: isUA ? "Мапи" : "Maps", item: localeUrl(locale, "/maps") },
          { "@type": "ListItem", position: 3, name: cityName, item: localeUrl(locale, `/maps/${city.slug}`) },
          { "@type": "ListItem", position: 4, name: districtName, item: localeUrl(locale, path) },
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

  const siblings = (DISTRICT_BY_CITY_SLUG[city.slug] ?? []).filter((d) => d.slug !== district.slug);

  const ui = isUA
    ? {
        maps: "Мапи",
        cta: "Створити цю мапу",
        recommend: `Рекомендований розмір — ${tpl.sizeMm ?? 80} мм${tpl.style === "relief" ? ", з увімкненим рельєфом" : ""}.`,
        cityLink: `Усі райони міста — ${cityName}`,
        brelok: `Брелок з районом — ${districtName}`,
        siblings: "Інші райони цього міста",
      }
    : {
        maps: "Maps",
        cta: "Create this map",
        recommend: `Recommended size — ${tpl.sizeMm ?? 80} mm${tpl.style === "relief" ? ", with relief enabled" : ""}.`,
        cityLink: `All districts of ${cityName}`,
        brelok: `Keychain with this district`,
        siblings: "Other districts of this city",
      };

  return (
    <main id="main-content" tabIndex={-1} className="mx-auto max-w-[820px] px-5 py-14 lg:py-20">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      <nav className="text-[13px] text-ink-3" aria-label="breadcrumb">
        <Link href="/" className="hover:underline">Monadruk</Link>
        {" / "}
        <Link href="/maps" className="hover:underline">{ui.maps}</Link>
        {" / "}
        <Link href={`/maps/${city.slug}`} className="hover:underline">{cityName}</Link>
        {" / "}
        <span className="text-ink">{districtName}</span>
      </nav>
      <h1 className="mt-5 text-[clamp(28px,4vw,46px)] leading-tight">{c.h1}</h1>
      {c.intro.map((p, i) => (
        <p key={i} className={`${i === 0 ? "mt-5" : "mt-3"} text-[15px] leading-relaxed text-ink-2`}>{p}</p>
      ))}
      <p className="mt-3 text-[14px] font-semibold text-[var(--accent-strong)]">{ui.recommend}</p>

      <div className="mt-8 flex flex-wrap gap-3">
        <Link
          href={`/create?template=${district.templateId}`}
          className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-6 py-3 text-sm font-semibold text-white transition hover:opacity-90"
        >
          {ui.cta}
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

      {/* FAQ (спільний з іншими сторінками міста, +FAQPage JSON-LD вище). */}
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

      {/* Крос-лінки на кластер міста. */}
      <section className="mt-10 flex flex-col gap-2 rounded-[18px] border border-line-soft bg-white/60 px-5 py-5">
        <Link href={`/maps/${city.slug}`} className="text-[14.5px] font-semibold text-[var(--accent-strong)] hover:underline">
          {ui.cityLink} →
        </Link>
        <Link href={`/brelok/${city.slug}`} className="text-[14.5px] font-semibold text-[var(--accent-strong)] hover:underline">
          {ui.brelok} →
        </Link>
      </section>

      {siblings.length > 0 && (
        <section className="mt-10">
          <h2 className="text-[18px] font-semibold text-ink">{ui.siblings}</h2>
          <ul className="mt-3 flex flex-wrap gap-2">
            {siblings.map((d) => (
              <li key={d.slug}>
                <Link
                  href={`/maps/${city.slug}/${d.slug}`}
                  className="inline-block rounded-full border border-line-soft bg-white/70 px-4 py-2 text-[13.5px] font-medium text-ink-2 transition hover:border-[var(--accent)] hover:text-ink"
                >
                  {isUA ? MAP_TEMPLATES.find((t) => t.id === d.templateId)?.district : d.enName}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}
