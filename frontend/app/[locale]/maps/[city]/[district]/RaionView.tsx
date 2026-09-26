import { BASE, localeUrl, priceValidUntil } from "@/i18n/metadata";
import type { AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import type { CityPage } from "@/lib/cityPages";
import { RAIONS_BY_CITY, bearingFrom, type CityRaion } from "@/lib/cityRaions";
import { STREET_PAGES_BY_CITY } from "@/lib/streetPages";
import { DISTRICT_BY_CITY_SLUG } from "@/lib/cityLanding";
import { MAP_TEMPLATES } from "@/lib/templates";
import { mapPriceRange } from "@/lib/mapPrices";
import { cityFaq, contentLocale } from "@/lib/cityLanding";
import MapRenderFigure, { mapRenderUrl } from "@/components/MapRenderFigure";

/** Назва району в родовому для uk-речень: «Оболонський район» → «Оболонського району». */
function ukGen(n: string): string {
  return n
    .replace(/ький район$/, "ького району")
    .replace(/ний район$/, "ного району")
    .replace(/ій район$/, "ього району")
    .replace(/ий район$/, "ого району");
}

/**
 * 26.09.2026: сторінка адміністративного району (lib/cityRaions.ts), uk/en.
 * Унікальність: напрям і відстань від центру міста, сусідні райони з відстанями,
 * орієнтири (якщо відомі) + deep-link у конструктор на центр району.
 */
export default function RaionView({ raion, city, locale }: { raion: CityRaion; city: CityPage; locale: AppLocale }) {
  const isUA = locale === "uk";
  const cityName = isUA ? city.names.uk : city.names.en;
  const rName = isUA ? raion.uk : raion.en;
  const path = `/maps/${city.slug}/${raion.slug}`;
  const nf = new Intl.NumberFormat(isUA ? "uk-UA" : "en-US", { maximumFractionDigits: 1 });
  const range = mapPriceRange(locale);
  const faq = cityFaq(contentLocale(locale), cityName, "podarunok");
  const fromCentre = bearingFrom(city.center, raion.center);
  const siblings = (RAIONS_BY_CITY[city.slug] ?? [])
    .filter((r) => r.slug !== raion.slug)
    .map((r) => ({ r, b: bearingFrom(raion.center, r.center) }))
    .sort((a, b) => a.b.km - b.b.km);
  const neighbourhoods = (DISTRICT_BY_CITY_SLUG[city.slug] ?? []).map((dp) => ({
    dp,
    tpl: MAP_TEMPLATES.find((t) => t.id === dp.templateId),
  })).filter((x) => x.tpl);
  const lm = raion.landmarks ?? [];
  // Відомі вулиці, для яких цей район — найближчий центр (та сама логіка, що у StreetView).
  const streets = (STREET_PAGES_BY_CITY[city.slug] ?? []).filter((s) => {
    const all = RAIONS_BY_CITY[city.slug] ?? [];
    const best = all.map((r) => ({ r, km: bearingFrom(s.center, r.center).km })).sort((a, b) => a.km - b.km)[0];
    return best?.r.slug === raion.slug;
  });
  const createHref = `/create?lat=${raion.center[0]}&lon=${raion.center[1]}`;
  const central = fromCentre.km < 2.5;

  const t = isUA
    ? {
        h1: `3D-модель: ${rName} (${cityName}) — купити макет району`,
        intro: [
          `Купити 3D-мапу ${ukGen(raion.uk)} чи замовити макет своєї вулиці — просто: конструктор відкривається одразу на центрі району, ви пересуваєте рамку на свій квартал і бачите модель за кілька хвилин. Будинки з реальними висотами, дороги, парки й водойми — за даними OpenStreetMap.`,
          central
            ? `${raion.uk} охоплює саму серцевину міста ${cityName}, тож на моделі буде найщільніша забудова й головні вулиці.`
            : `Центр району лежить приблизно за ${nf.format(fromCentre.km)} км ${fromCentre.uk} від центру міста ${cityName} — на мапі добре видно, як район переходить у сусідні квартали.`,
          lm.length
            ? `Що найчастіше беруть у модель: ${lm.map((x) => x.uk).join(", ")}. Саме ці місця роблять мапу впізнаваною для тих, хто тут живе.`
            : `Найкраще працює мапа конкретного житлового кварталу чи вулиці, де ви живете або виросли, — рамку можна поставити точно на свій будинок.`,
        ],
        tip: "Рекомендований розмір для району — 8–11 см; для кількох кварталів з довкіллям — 15 см або панно з плиток.",
        cta: "Створити 3D-мапу району",
        ctaK: "Брелок з мапою району",
        buyH2: "Скільки коштує і як купити",
        buy: `3D-мапа району — від ${range.low} ₴ до ${range.high} ₴ залежно від розміру, брелок з картою — від 170 ₴, файл 3MF для самодруку — 149 ₴. Друк 2–4 робочі дні, доставка Новою Поштою по Україні.`,
        sib: `Інші райони міста ${cityName}`,
        nb: "Готові сцени кварталів",
        cityLink: `3D-модель міста ${cityName}`,
        faqT: "Часті запитання",
        km: "км",
      }
    : {
        h1: `3D model of ${rName} (${cityName}) — buy a district map`,
        intro: [
          `Buying a 3D map of ${rName} or a model of your own street is simple: the builder opens right at the centre of the district, you move the frame to your block and see the model in a few minutes. Buildings with real heights, roads, parks and water — from OpenStreetMap data.`,
          central
            ? `${rName} covers the very heart of ${cityName}, so the model shows the densest blocks and main streets.`
            : `The district centre lies about ${nf.format(fromCentre.km)} km ${fromCentre.en} of central ${cityName} — the map shows clearly how it blends into neighbouring blocks.`,
          lm.length
            ? `What people usually include: ${lm.map((x) => x.en).join(", ")} — the places that make the map recognisable to locals.`
            : `A map of the exact block or street where you live or grew up works best — the frame can sit right on your building.`,
        ],
        tip: "Recommended size for a district: 8–11 cm; for several blocks with surroundings — 15 cm or a multi-tile panel.",
        cta: "Create a 3D map of this district",
        ctaK: "District map keychain",
        buyH2: "Price and how to buy",
        buy: `A district 3D map costs ${range.low}–${range.high} ${range.currency} depending on size; the 3MF file for self-printing is also available. Production takes 2–4 working days.`,
        sib: `Other districts of ${cityName}`,
        nb: "Ready-made neighbourhood scenes",
        cityLink: `3D model of ${cityName}`,
        faqT: "FAQ",
        km: "km",
      };

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Product",
        name: t.h1,
        description: t.intro[0],
        image: mapRenderUrl(`${city.slug}--${raion.slug}`, BASE) ?? `${BASE}/real/map-1.webp`,
        brand: { "@type": "Brand", name: "Monadruk" },
        sku: `MND-RAION-${city.slug}-${raion.slug}`,
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
          { "@type": "ListItem", position: 2, name: isUA ? "Мапи міст" : "City maps", item: localeUrl(locale, "/maps") },
          { "@type": "ListItem", position: 3, name: cityName, item: localeUrl(locale, `/maps/${city.slug}`) },
          { "@type": "ListItem", position: 4, name: rName, item: localeUrl(locale, path) },
        ],
      },
      {
        "@type": "FAQPage",
        mainEntity: faq.map((f) => ({ "@type": "Question", name: f.q, acceptedAnswer: { "@type": "Answer", text: f.a } })),
      },
    ],
  };

  return (
    <main id="main-content" tabIndex={-1} className="mx-auto max-w-[820px] px-5 py-14 lg:py-20">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      <nav className="text-[13px] text-ink-3" aria-label="breadcrumb">
        <Link href="/" className="hover:underline">Monadruk</Link>
        {" / "}
        <Link href="/maps" className="hover:underline">{isUA ? "Мапи міст" : "City maps"}</Link>
        {" / "}
        <Link href={`/maps/${city.slug}`} className="hover:underline">{cityName}</Link>
        {" / "}
        <span className="text-ink">{rName}</span>
      </nav>
      <h1 className="mt-5 text-[clamp(28px,4vw,46px)] leading-tight">{t.h1}</h1>
      {t.intro.map((p, i) => (
        <p key={i} className={`${i === 0 ? "mt-5" : "mt-3"} text-[15px] leading-relaxed text-ink-2`}>{p}</p>
      ))}
      <p className="mt-3 text-[14px] font-semibold text-[var(--accent-strong)]">{t.tip}</p>

      <div className="mt-8 flex flex-wrap gap-3">
        <Link href={createHref} className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-6 py-3 text-sm font-semibold text-white transition hover:opacity-90">
          {t.cta}
        </Link>
        <Link href="/keychains" className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] border border-line-soft bg-white/80 px-6 py-3 text-sm font-semibold text-ink transition hover:border-[var(--accent)]">
          {t.ctaK}
        </Link>
      </div>

      <MapRenderFigure id={`${city.slug}--${raion.slug}`} name={`${rName}, ${cityName}`} isUA={isUA} />

      <section className="mt-10">
        <h2 className="text-[20px] font-semibold">{t.buyH2}</h2>
        <p className="mt-3 text-[15px] leading-relaxed text-ink-2">{t.buy}</p>
      </section>

      {siblings.length > 0 && (
        <section className="mt-10">
          <h2 className="text-[18px] font-semibold text-ink">{t.sib}</h2>
          <ul className="mt-3 flex flex-wrap gap-2">
            {siblings.map(({ r, b }) => (
              <li key={r.slug}>
                <Link href={`/maps/${city.slug}/${r.slug}`} className="inline-block rounded-full border border-line-soft bg-white/70 px-4 py-2 text-[13.5px] font-medium text-ink-2 transition hover:border-[var(--accent)] hover:text-ink">
                  {isUA ? r.uk : r.en} · {nf.format(b.km)} {t.km}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}

      {streets.length > 0 && (
        <section className="mt-10">
          <h2 className="text-[18px] font-semibold text-ink">{isUA ? "3D-модель вулиці в цьому районі" : "Street 3D maps in this district"}</h2>
          <ul className="mt-3 flex flex-wrap gap-2">
            {streets.map((s) => (
              <li key={s.slug}>
                <Link href={`/maps/${city.slug}/${s.slug}`} className="inline-block rounded-full border border-line-soft bg-white/70 px-4 py-2 text-[13.5px] font-medium text-ink-2 transition hover:border-[var(--accent)] hover:text-ink">
                  {isUA ? s.uk : s.en || s.uk}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}

      {neighbourhoods.length > 0 && (
        <section className="mt-10">
          <h2 className="text-[18px] font-semibold text-ink">{t.nb}</h2>
          <ul className="mt-3 flex flex-wrap gap-2">
            {neighbourhoods.map(({ dp, tpl }) => (
              <li key={dp.slug}>
                <Link href={`/maps/${city.slug}/${dp.slug}`} className="inline-block rounded-full border border-line-soft bg-white/70 px-4 py-2 text-[13.5px] font-medium text-ink-2 transition hover:border-[var(--accent)] hover:text-ink">
                  {isUA ? tpl!.district : dp.enName}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="mt-10">
        <h2 className="text-[20px] font-semibold">{t.faqT}</h2>
        <dl className="mt-4 flex flex-col gap-4">
          {faq.map((f) => (
            <div key={f.q}>
              <dt className="text-[15px] font-semibold text-ink">{f.q}</dt>
              <dd className="mt-1.5 text-[14.5px] leading-relaxed text-ink-2">{f.a}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="mt-10 flex flex-col gap-2 rounded-[18px] border border-line-soft bg-white/60 px-5 py-5">
        <Link href={`/maps/${city.slug}`} className="text-[14.5px] font-semibold text-[var(--accent-strong)] hover:underline">{t.cityLink} →</Link>
        <Link href="/3d-model-mista" className="text-[14.5px] font-semibold text-[var(--accent-strong)] hover:underline">
          {isUA ? "Купити 3D-модель міста: розміри, ціни, доставка" : "Buy a 3D city model: sizes, prices, delivery"} →
        </Link>
      </section>
    </main>
  );
}
