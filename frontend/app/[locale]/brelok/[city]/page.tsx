import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { setRequestLocale } from "next-intl/server";
import { BASE, localeUrl, priceValidUntil } from "@/i18n/metadata";
import { routing, locales, localeMeta, defaultLocale, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { CITY_PAGES, CITY_PAGE_BY_SLUG } from "@/lib/cityPages";
import { cityFacts, CITY_FACTS } from "@/lib/cityFacts";
import { MAP_TEMPLATES } from "@/lib/templates";
import { KEYCHAIN_PRICE_UAH, mapPriceEur } from "@/lib/mapPrices";
import { brelokCityCopy, contentLocale } from "@/lib/cityLanding";

/**
 * Programmatic SEO рівень 2: брелок × місто (23 × 6 локалей).
 * Запити виду «брелок київ», «брелок з картою львова», «map keychain kyiv».
 * Контент з lib/cityLanding.ts (uk/en, факти міста вплетені у прозу) —
 * НЕ клоакінг: сторінка видима, злінкована з /maps/[city] і сайтмапи.
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
  const c = brelokCityCopy(contentLocale(locale), city.names[locale], city.slug, cityFacts(city.slug));
  const path = `/brelok/${city.slug}`;
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

export default async function BrelokCityPage({
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
  const name = city.names[locale];
  const isUA = locale === "uk";
  const c = brelokCityCopy(contentLocale(locale), name, city.slug, cityFacts(city.slug));
  const path = `/brelok/${city.slug}`;

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Product",
        name: c.h1,
        description: c.description,
        image: `${BASE}/showcase/keychain-5.webp`,
        brand: { "@type": "Brand", name: "Monadruk" },
        sku: `MND-KEYCHAIN-${city.slug}`,
        offers: {
          "@type": "Offer",
          priceCurrency: isUA ? "UAH" : "EUR",
          price: isUA ? String(KEYCHAIN_PRICE_UAH) : String(mapPriceEur(KEYCHAIN_PRICE_UAH)),
          priceValidUntil: priceValidUntil(),
          availability: "https://schema.org/InStock",
          url: localeUrl(locale, path),
        },
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: isUA ? "Брелоки" : "Keychains", item: localeUrl(locale, "/keychains") },
          { "@type": "ListItem", position: 3, name, item: localeUrl(locale, path) },
        ],
      },
    ],
  };

  // Райони міста → глибокі лінки на готову сцену конструктора.
  const districts = MAP_TEMPLATES.filter((tpl) => tpl.cityKey === city.key);

  // Споріднені міста (найближчі + найбільші) → перелінковка МІЖ брелок-сторінками.
  const hav = (a: readonly [number, number], b: readonly [number, number]) => {
    const R = 6371, toRad = (d: number) => (d * Math.PI) / 180;
    const dLat = toRad(b[0] - a[0]), dLng = toRad(b[1] - a[1]);
    const s = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(a[0])) * Math.cos(toRad(b[0])) * Math.sin(dLng / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(s));
  };
  const rest = CITY_PAGES.filter((x) => x.slug !== city.slug);
  const nearest = [...rest].sort((a, b) => hav(city.center, a.center) - hav(city.center, b.center)).slice(0, 8);
  const biggest = [...rest]
    .sort((a, b) => (CITY_FACTS[b.slug]?.population ?? 0) - (CITY_FACTS[a.slug]?.population ?? 0))
    .filter((x) => !nearest.some((n) => n.slug === x.slug))
    .slice(0, 3);
  const others = [...nearest, ...biggest];

  const ui = isUA
    ? {
        breadcrumb: "Брелоки",
        cta: "Створити брелок у конструкторі",
        ctaGift: "Подарунок з цього міста",
        districts: "Популярні райони на брелок",
        price: `Ціна: від ${KEYCHAIN_PRICE_UAH} ₴ за брелок · виготовлення 1–3 робочі дні`,
        others: "Брелоки з інших міст",
        map: `3D-мапа міста — ${name}`,
      }
    : {
        breadcrumb: "Keychains",
        cta: "Create a keychain in the builder",
        ctaGift: "A gift from this city",
        districts: "Popular districts for a keychain",
        price: `Price: from ${KEYCHAIN_PRICE_UAH} ₴ (≈€${mapPriceEur(KEYCHAIN_PRICE_UAH)}) · made in 1–3 business days`,
        others: "Keychains from other cities",
        map: `3D city map — ${name}`,
      };

  return (
    <main id="main-content" tabIndex={-1} className="mx-auto max-w-[820px] px-5 py-14 lg:py-20">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      <nav className="text-[13px] text-ink-3" aria-label="breadcrumb">
        <Link href="/" className="hover:underline">Monadruk</Link>
        {" / "}
        <Link href="/keychains" className="hover:underline">{ui.breadcrumb}</Link>
        {" / "}
        <span className="text-ink">{name}</span>
      </nav>
      <h1 className="mt-5 text-[clamp(28px,4vw,46px)] leading-tight">{c.h1}</h1>
      {c.intro.map((p, i) => (
        <p key={i} className={`${i === 0 ? "mt-5" : "mt-3"} text-[15px] leading-relaxed text-ink-2`}>{p}</p>
      ))}
      <p className="mt-3 text-[14px] font-semibold text-[var(--accent-strong)]">{ui.price}</p>

      <div className="mt-8 flex flex-wrap gap-3">
        <Link
          href={`/keychains?city=${city.key}`}
          className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-6 py-3 text-sm font-semibold text-white transition hover:opacity-90"
        >
          {ui.cta}
        </Link>
        <Link
          href={`/podarunok/${city.slug}`}
          className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] border border-line-soft bg-white/80 px-6 py-3 text-sm font-semibold text-ink transition hover:border-[var(--accent)]"
        >
          {ui.ctaGift}
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

      {districts.length > 0 && (
        <section className="mt-12">
          <h2 className="text-[20px] font-semibold">{ui.districts}</h2>
          <ul className="mt-4 grid gap-3 sm:grid-cols-2">
            {districts.map((d) => (
              <li key={d.id}>
                <Link
                  href={`/create?template=${d.id}`}
                  className="block h-full rounded-[16px] border border-line-soft bg-white/70 px-4 py-3 transition hover:border-[var(--accent)]"
                >
                  <span className="text-[15px] font-semibold text-ink">{d.district}</span>
                  {locale === "uk" && d.blurb ? (
                    <span className="mt-1 block text-[13px] leading-snug text-ink-2">{d.blurb}</span>
                  ) : null}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* Крос-лінк на сторінку мапи цього ж міста (siloing місто-кластера). */}
      <section className="mt-12 rounded-[18px] border border-line-soft bg-white/60 px-5 py-5">
        <Link href={`/maps/${city.slug}`} className="text-[14.5px] font-semibold text-[var(--accent-strong)] hover:underline">
          {ui.map} →
        </Link>
      </section>

      <h2 className="mt-12 text-[20px] font-semibold">{ui.others}</h2>
      <ul className="mt-4 flex flex-wrap gap-2">
        {others.map((x) => (
          <li key={x.slug}>
            <Link
              href={`/brelok/${x.slug}`}
              className="inline-block rounded-full border border-line-soft bg-white/70 px-4 py-2 text-[13.5px] font-medium text-ink-2 transition hover:border-[var(--accent)] hover:text-ink"
            >
              {x.names[locale]}
            </Link>
          </li>
        ))}
      </ul>
    </main>
  );
}
