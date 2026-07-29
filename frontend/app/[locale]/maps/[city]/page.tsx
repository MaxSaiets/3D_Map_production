import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { BASE, localeUrl, priceValidUntil } from "@/i18n/metadata";
import { routing, locales, localeMeta, defaultLocale, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { MAP_CITY_PAGES as CITY_PAGES, MAP_CITY_PAGE_BY_SLUG as CITY_PAGE_BY_SLUG, isWorldCity } from "@/lib/cityPages";
import { cityFacts, CITY_FACTS } from "@/lib/cityFacts";
import { WORLD_CITY_BY_SLUG } from "@/lib/worldCities";
import { cityProse, cityDerivedFacts } from "@/lib/cityProse";
import { MAP_TEMPLATES } from "@/lib/templates";
import { mapPriceRange } from "@/lib/mapPrices";
import { getCatalog, formatCatalogPrice } from "@/lib/catalog";
import { cityFaq, contentLocale, DISTRICT_PAGES } from "@/lib/cityLanding";
import { BLOG_ARTICLES, blogContent } from "@/lib/blog";

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
      // Робочий рут-OG (colocated [locale]-OG дають 307→404 через next-intl as-needed).
      images: [`${BASE}/opengraph-image`],
    },
    twitter: { card: "summary_large_image", title, description, images: [`${BASE}/opengraph-image`] },
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

  // Унікальні факти + локалізація: числа через Intl, власні назви uk/latin.
  // Для міст Європи факти живуть у WORLD_CITIES (той самий контракт CityFacts).
  const facts = cityFacts(city.slug) ?? WORLD_CITY_BY_SLUG[city.slug]?.facts ?? null;
  const isWorld = isWorldCity(city.slug);
  const nf = new Intl.NumberFormat(locale === "uk" ? "uk-UA" : locale);
  const pn = (o: { uk: string; latin: string }) => (locale === "uk" ? o.uk : o.latin);
  const faq = cityFaq(contentLocale(locale), name, "podarunok");

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Product",
        name: t("title", { city: name }),
        description: t("description", { city: name }),
        image: `${BASE}/real/map-1.webp`,
        brand: { "@type": "Brand", name: "Monadruk" },
        sku: `MND-MAP-${city.slug}`,
        offers: {
          // Сторінка про 3D-МАПУ міста → діапазон цін мапи (S–XL), а не брелка.
          // Ціни з єдиного джерела lib/mapPrices.ts (синхрон з pricing.json) — без дрейфу.
          "@type": "AggregateOffer",
          priceCurrency: mapPriceRange(locale).currency,
          lowPrice: mapPriceRange(locale).low,
          highPrice: mapPriceRange(locale).high,
          offerCount: mapPriceRange(locale).offerCount,
          priceValidUntil: priceValidUntil(),
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

  // Видимі ціни (раніше лише у JSON-LD AggregateOffer) — з єдиного каталогу
  // (синхрон з /prices). Показуємо мапи S–XL + рельєф + магніт + брелок.
  const cat = getCatalog(locale);
  const priceItems = [
    ...cat.categories[0].items, // 3D-мапи (S/M/L/XL + рельєф)
    ...(cat.categories[1]?.items ?? []), // магніт
    ...(cat.categories[2]?.items ?? []), // брелок
  ];

  // Споріднені міста: найближчі за відстанню (локальний кластер) + кілька
  // найбільших за населенням — щоб блок був УНІКАЛЬНИМ на кожній сторінці
  // (не той самий фіксований список) і давав осмислену внутрішню перелінковку.
  const hav = (a: readonly [number, number], b: readonly [number, number]) => {
    const R = 6371, toRad = (d: number) => (d * Math.PI) / 180;
    const dLat = toRad(b[0] - a[0]), dLng = toRad(b[1] - a[1]);
    const s = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(a[0])) * Math.cos(toRad(b[0])) * Math.sin(dLng / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(s));
  };
  const rest = CITY_PAGES.filter((c) => c.slug !== city.slug);
  const nearest = [...rest].sort((a, b) => hav(city.center, a.center) - hav(city.center, b.center)).slice(0, 9);
  const biggest = [...rest]
    .sort((a, b) => (CITY_FACTS[b.slug]?.population ?? 0) - (CITY_FACTS[a.slug]?.population ?? 0))
    .filter((c) => !nearest.some((n) => n.slug === c.slug))
    .slice(0, 3);
  const others = [...nearest, ...biggest];

  // Райони міста (rank 3): унікальні prose-блерби + глибокі лінки на /create?template=
  const districts = MAP_TEMPLATES.filter((tpl) => tpl.cityKey === city.key);

  return (
    <main id="main-content" tabIndex={-1} className="mx-auto max-w-[820px] px-5 py-14 lg:py-20">
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
      {/* Унікальна prose-фраза на основі фактів (анти-doorway): дужкова конструкція
          уникає відмінкових/родових узгоджень з власними назвами у 6 мовах. */}
      {/* УНІКАЛЬНИЙ ОПИС: замість однієї спільної фрази (яка й давала 78%
          дублювання між містами) — 6 структур × обчислювані інсайти
          (щільність, вік, площа) з реальних даних. Див. lib/cityProse.ts. */}
      {facts && (
        <div className="mt-3 flex flex-col gap-3">
          {cityProse({ slug: city.slug, name, facts, locale }).map((para, idx) => (
            <p key={idx} className="text-[15px] leading-relaxed text-ink-2">{para}</p>
          ))}
        </div>
      )}

      {/* Унікальні факти про місто (анти-doorway): кожна сторінка отримує
          відмінні числа/назви, тож контент не byte-identical. Гард: якщо для
          slug немає даних — секція не рендериться (поведінка як раніше). */}
      {facts && (
        <section className="mt-9 rounded-[18px] border border-line-soft bg-white/60 px-5 py-5">
          <h2 className="text-[16px] font-semibold text-ink">{t("factsTitle", { city: name })}</h2>
          <dl className="mt-3 grid gap-x-7 gap-y-1.5 text-[14px] sm:grid-cols-2">
            {[
              [t("fPopulation"), `${nf.format(facts.population)} (${facts.populationYear})`],
              [facts.firstMention ? t("fFirstMention") : t("fFounded"), String(facts.founded)],
              [t("fArea"), `${nf.format(Math.round(facts.area_km2))} ${locale === "uk" ? "км²" : "km²"}`],
              [t("fRiver"), pn(facts.river)],
              [t("fOblast"), pn(facts.oblast)],
              [t("fLandmark"), pn(facts.landmark)],
              // Обчислювані показники: унікальні для кожного міста числа,
              // яких немає в жодного іншого (анти-дедуплікація).
              [
                locale === "uk" ? "Щільність" : "Density",
                `${nf.format(cityDerivedFacts(facts).density)} ${locale === "uk" ? "осіб/км²" : "people/km²"}`,
              ],
              [
                locale === "uk" ? "Вік міста" : "City age",
                `${nf.format(cityDerivedFacts(facts).age)} ${locale === "uk" ? "років" : "years"}`,
              ],
            ].map(([label, value]) => (
              <div key={label} className="flex items-baseline justify-between gap-3 border-b border-line-soft/50 py-1">
                <dt className="text-ink-3">{label}</dt>
                <dd className="text-right font-semibold text-ink">{value}</dd>
              </div>
            ))}
          </dl>
        </section>
      )}

      <div className="mt-8 flex flex-wrap gap-3">
        <Link
          href={`/create?city=${city.key}`}
          className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-6 py-3 text-sm font-semibold text-white transition hover:opacity-90"
        >
          {t("ctaMap", { city: name })}
        </Link>
        <Link
          href={isWorld ? "/keychains" : `/keychains?city=${city.key}`}
          className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] border border-line-soft bg-white/80 px-6 py-3 text-sm font-semibold text-ink transition hover:border-[var(--accent)]"
        >
          {t("ctaKeychain", { city: name })}
        </Link>
      </div>

      {/* Райони міста (rank 3): унікальний контент + глибокі лінки на готову сцену
          конструктора (/create?template=). Блерби uk-only → решта локалей лише назва. */}
      {districts.length > 0 && (
        <section className="mt-12">
          <h2 className="text-[20px] font-semibold">{t("districtsTitle")}</h2>
          <ul className="mt-4 grid gap-3 sm:grid-cols-2">
            {districts.map((d) => {
              const dp = DISTRICT_PAGES.find((x) => x.templateId === d.id);
              const href = dp ? `/maps/${city.slug}/${dp.slug}` : `/create?template=${d.id}`;
              return (
                <li key={d.id}>
                  <Link
                    href={href}
                    className="block h-full rounded-[16px] border border-line-soft bg-white/70 px-4 py-3 transition hover:border-[var(--accent)]"
                  >
                    <span className="text-[15px] font-semibold text-ink">{d.district}</span>
                    {locale === "uk" && d.blurb ? (
                      <span className="mt-1 block text-[13px] leading-snug text-ink-2">{d.blurb}</span>
                    ) : null}
                  </Link>
                </li>
              );
            })}
          </ul>
        </section>
      )}

      {/* Видимі ЦІНИ у гривнях — для покупця, SEO і перевірки LiqPay (115 сторінок). */}
      <section className="mt-12 rounded-[18px] border border-line-soft bg-white/60 px-5 py-5">
        <h2 className="text-[17px] font-semibold text-ink">{cat.h1} — {name}</h2>
        <ul className="mt-3 grid gap-x-7 gap-y-1.5 text-[14px] text-ink-2 sm:grid-cols-2">
          {priceItems.map((it) => (
            <li key={it.name} className="flex items-baseline justify-between gap-3 border-b border-line-soft/50 py-1">
              <span>{it.name}</span>
              <span className="whitespace-nowrap font-semibold text-[var(--accent-strong)]">
                {formatCatalogPrice(it.uah, it.kind, locale)}
              </span>
            </li>
          ))}
        </ul>
        <Link href="/prices" className="mt-3 inline-block text-[13.5px] font-semibold text-[var(--accent-strong)] hover:underline">
          {cat.h1} →
        </Link>
      </section>

      {/* FAQ (видимий, +FAQPage JSON-LD вище). */}
      <section className="mt-12">
        <h2 className="text-[20px] font-semibold">{locale === "uk" ? "Часті запитання" : "FAQ"}</h2>
        <dl className="mt-4 flex flex-col gap-4">
          {faq.map((f) => (
            <div key={f.q}>
              <dt className="text-[15px] font-semibold text-ink">{f.q}</dt>
              <dd className="mt-1.5 text-[14.5px] leading-relaxed text-ink-2">{f.a}</dd>
            </div>
          ))}
        </dl>
      </section>

      {/* Читати далі: 2 статті блогу. */}
      <section className="mt-10">
        <h2 className="text-[18px] font-semibold text-ink">{locale === "uk" ? "Читати далі" : "Read more"}</h2>
        <ul className="mt-3 flex flex-col gap-2">
          {BLOG_ARTICLES.slice(4, 6).map((a) => (
            <li key={a.slug}>
              <Link href={`/blog/${a.slug}`} className="text-[14.5px] font-medium text-[var(--accent-strong)] hover:underline">
                {blogContent(a, locale).h1} →
              </Link>
            </li>
          ))}
        </ul>
      </section>

      {/* Крос-лінки на brelok/podarunok сторінки цього ж міста (кластер міста,
          хвиля 2 programmatic SEO). Тексти bilingual-inline (контент цих сторінок
          uk/en з lib/cityLanding — той самий принцип, без нових i18n-ключів). */}
      {/* Гард: для міст Європи сторінок /brelok і /podarunok НЕМА (свідомо, щоб
          не роздувати краул-бюджет) → ведемо на хаби, а не в 404. */}
      <section className="mt-12 flex flex-col gap-2 rounded-[18px] border border-line-soft bg-white/60 px-5 py-5">
        <Link href={isWorld ? "/brelok" : `/brelok/${city.slug}`} className="text-[14.5px] font-semibold text-[var(--accent-strong)] hover:underline">
          {locale === "uk" ? `Брелок з картою міста — ${name}` : `City map keychain — ${name}`} →
        </Link>
        <Link href={isWorld ? "/podarunok" : `/podarunok/${city.slug}`} className="text-[14.5px] font-semibold text-[var(--accent-strong)] hover:underline">
          {locale === "uk" ? `Подарунок з міста — ${name}` : `A gift from ${name}`} →
        </Link>
      </section>

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
