import { BASE, localeUrl, priceValidUntil } from "@/i18n/metadata";
import type { AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { MAP_CITY_PAGES, isWorldCity, type CityPage } from "@/lib/cityPages";
import { UA_CITY2_BY_SLUG } from "@/lib/uaCities2";
import { CITY_FACTS } from "@/lib/cityFacts";
import { mapPriceRange } from "@/lib/mapPrices";
import { getCatalog, formatCatalogPrice } from "@/lib/catalog";
import { cityFaq, contentLocale } from "@/lib/cityLanding";
import { bearingFrom } from "@/lib/cityRaions";

/**
 * 26.09.2026: сторінка міста ДРУГОГО КОЛА (lib/uaCities2.ts). Текст uk/en
 * будується з реальних даних міста (населення, площа, щільність, найближчі
 * міста й відстані до них, річка/візитівка — якщо відомі). Відсутні поля
 * просто не показуються. Ключові форми запиту: «купити», «замовити», «зробити».
 */
export default function UaCity2View({ city, locale }: { city: CityPage; locale: AppLocale }) {
  const d = UA_CITY2_BY_SLUG[city.slug]!;
  const isUA = locale === "uk";
  const name = isUA ? d.names.uk : d.names.en;
  const nf = new Intl.NumberFormat(isUA ? "uk-UA" : "en-US");
  const path = `/maps/${city.slug}`;
  const faq = cityFaq(contentLocale(locale), name, "podarunok");
  const range = mapPriceRange(locale);
  const density = d.area_km2 ? Math.round(d.population / d.area_km2) : null;

  // Найближчі міста України (обидва кола) з відстанню — унікальний блок для кожної сторінки.
  const ua = MAP_CITY_PAGES.filter((c) => !isWorldCity(c.slug) && c.slug !== city.slug);
  const near = ua
    .map((c) => ({ c, b: bearingFrom(d.center, c.center) }))
    .sort((a, b) => a.b.km - b.b.km)
    .slice(0, 10);
  const nearestBig = ua
    .filter((c) => (CITY_FACTS[c.slug]?.population ?? 0) > 0)
    .map((c) => ({ c, b: bearingFrom(d.center, c.center) }))
    .sort((a, b) => a.b.km - b.b.km)[0];

  const sizeTip = (() => {
    const a = d.area_km2 ?? 0;
    if (isUA) {
      if (a && a < 40) return `Площа міста — близько ${nf.format(Math.round(a))} км², тож центр із головними вулицями вміщується в одну мапу 11–15 см без втрати деталей.`;
      if (a && a < 120) return `На ${nf.format(Math.round(a))} км² місто цілком в одну модель не лягає — зазвичай беруть центр або свій район і друкують мапу 8–11 см.`;
      return `Місто велике${a ? ` (${nf.format(Math.round(a))} км²)` : ""}, тому найкраще працює мапа конкретного району чи вулиці, а для всього міста — панно з кількох плиток.`;
    }
    if (a && a < 40) return `The city covers about ${nf.format(Math.round(a))} km², so the centre with its main streets fits one 11–15 cm map with full detail.`;
    if (a && a < 120) return `At ${nf.format(Math.round(a))} km² the whole city will not fit one model — people usually pick the centre or their own neighbourhood at 8–11 cm.`;
    return `The city is large${a ? ` (${nf.format(Math.round(a))} km²)` : ""}, so a single district or street works best; for the whole city, a multi-tile panel.`;
  })();

  const t = isUA
    ? {
        h1: `3D-модель міста ${name} — купити макет чи 3D-мапу`,
        p1: `Хочете купити 3D-модель міста ${name} або замовити макет свого району? Ми друкуємо обʼємні мапи з реальних даних OpenStreetMap: вулиці, будинки з висотами, парки й водойми. Обираєте ділянку в конструкторі, дивитеся превʼю безкоштовно — і замовляєте друк з доставкою Новою Поштою.`,
        p2: `Мапа від ${range.low} ₴, брелок з картою — від 170 ₴. Виготовлення 2–4 робочі дні. Можна також купити лише файл 3MF для самостійного друку.`,
        facts: `Коротко про місто ${name}`,
        fPop: "Населення", fArea: "Площа", fDens: "Щільність", fObl: "Область", fRiver: "Водойма", fLm: "Візитівка",
        fFounded: d.firstMention ? "Перша згадка" : "Засноване",
        cta: `Створити 3D-мапу: ${name}`, ctaK: "Брелок з картою міста",
        prices: "Ціни на 3D-мапи та брелоки",
        near: "Найближчі міста", km: "км",
        faqT: "Часті запитання",
        more: "Купити 3D-модель міста: розміри, ціни, доставка",
        all: "Усі міста України",
        buyH2: `Як замовити 3D-мапу міста ${name}`,
        buy: [
          `Відкрийте конструктор — він стартує одразу на центрі міста ${name}. Посуньте рамку на свій район, вулицю чи двір.`,
          "Оберіть розмір і стиль (з рельєфом чи без) — превʼю моделі зʼявиться за кілька хвилин.",
          "Замовте друк і вкажіть відділення Нової Пошти — модель надрукуємо за 2–4 робочі дні.",
        ],
      }
    : {
        h1: `3D model of ${name} — buy a printed city map`,
        p1: `Want to buy a 3D model of ${name} or order a map of your own neighbourhood? We print tactile maps from real OpenStreetMap data: streets, buildings with real heights, parks and water. Pick the area in the builder, preview it for free and order the print.`,
        p2: `Maps from ≈€8, map keychains from ≈€4. Production takes 2–4 working days. You can also buy just the 3MF file to print yourself.`,
        facts: `${name} at a glance`,
        fPop: "Population", fArea: "Area", fDens: "Density", fObl: "Region", fRiver: "Water", fLm: "Landmark",
        fFounded: d.firstMention ? "First mentioned" : "Founded",
        cta: `Create a 3D map of ${name}`, ctaK: "City map keychain",
        prices: "Prices",
        near: "Nearby cities", km: "km",
        faqT: "FAQ",
        more: "Buy a 3D city model: sizes, prices, delivery",
        all: "All cities",
        buyH2: `How to order a 3D map of ${name}`,
        buy: [
          `Open the builder — it starts right at the centre of ${name}. Move the frame to your district, street or yard.`,
          "Choose size and style (with or without relief) — the preview is ready in a few minutes.",
          "Order the print and enter the delivery details — we print it in 2–4 working days.",
        ],
      };

  const prose: string[] = [];
  prose.push(
    isUA
      ? `${name} — місто ${d.oblast.uk.replace(/а область$/, "ої області")}${density ? ` з населенням ${nf.format(d.population)} (${d.populationYear}) і щільністю близько ${nf.format(density)} осіб/км²` : ` з населенням ${nf.format(d.population)} (${d.populationYear})`}.${nearestBig ? ` До міста ${nearestBig.c.names.uk} — ${nf.format(Math.round(nearestBig.b.km))} км ${nearestBig.b.uk}.` : ""}`
      : `${name} is a city in ${d.oblast.latin}${density ? ` with ${nf.format(d.population)} residents (${d.populationYear}) and about ${nf.format(density)} people/km²` : ` with ${nf.format(d.population)} residents (${d.populationYear})`}.${nearestBig ? ` ${nearestBig.c.names.en} lies ${nf.format(Math.round(nearestBig.b.km))} km to the ${nearestBig.b.en}.` : ""}`,
  );
  if (d.river || d.landmark) {
    prose.push(
      isUA
        ? `На 3D-мапі місто впізнають насамперед за ${[d.river ? `водоймою (${d.river.uk})` : "", d.landmark ? `візитівкою — ${d.landmark.uk}` : ""].filter(Boolean).join(" і ")}: саме ці орієнтири люди шукають на моделі першими.`
        : `On a 3D map the city is recognised first by ${[d.river ? `its water (${d.river.latin})` : "", d.landmark ? `its landmark, ${d.landmark.latin}` : ""].filter(Boolean).join(" and ")} — the anchors people look for first.`,
    );
  }
  prose.push(sizeTip);

  const factsRows: [string, string][] = [
    [t.fPop, `${nf.format(d.population)} (${d.populationYear})`],
    ...(d.area_km2 ? ([[t.fArea, `${nf.format(Math.round(d.area_km2))} ${isUA ? "км²" : "km²"}`]] as [string, string][]) : []),
    ...(density ? ([[t.fDens, `${nf.format(density)} ${isUA ? "осіб/км²" : "people/km²"}`]] as [string, string][]) : []),
    [t.fObl, isUA ? d.oblast.uk : d.oblast.latin],
    ...(d.river ? ([[t.fRiver, isUA ? d.river.uk : d.river.latin]] as [string, string][]) : []),
    ...(d.landmark ? ([[t.fLm, isUA ? d.landmark.uk : d.landmark.latin]] as [string, string][]) : []),
    ...(d.founded ? ([[t.fFounded, String(d.founded)]] as [string, string][]) : []),
  ];

  const cat = getCatalog(locale);
  const priceItems = [...cat.categories[0].items, ...(cat.categories[1]?.items ?? []), ...(cat.categories[2]?.items ?? [])];
  const createHref = `/create?lat=${d.center[0]}&lon=${d.center[1]}`;

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Product",
        name: t.h1,
        description: t.p1,
        image: `${BASE}/real/map-1.webp`,
        brand: { "@type": "Brand", name: "Monadruk" },
        sku: `MND-MAP-${city.slug}`,
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
          { "@type": "ListItem", position: 3, name, item: localeUrl(locale, path) },
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
        <span className="text-ink">{name}</span>
      </nav>
      <h1 className="mt-5 text-[clamp(28px,4vw,46px)] leading-tight">{t.h1}</h1>
      <p className="mt-5 text-[15px] leading-relaxed text-ink-2">{t.p1}</p>
      <p className="mt-3 text-[15px] leading-relaxed text-ink-2">{t.p2}</p>
      {prose.map((p, i) => (
        <p key={i} className="mt-3 text-[15px] leading-relaxed text-ink-2">{p}</p>
      ))}

      <div className="mt-8 flex flex-wrap gap-3">
        <Link href={createHref} className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-6 py-3 text-sm font-semibold text-white transition hover:opacity-90">
          {t.cta}
        </Link>
        <Link href="/keychains" className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] border border-line-soft bg-white/80 px-6 py-3 text-sm font-semibold text-ink transition hover:border-[var(--accent)]">
          {t.ctaK}
        </Link>
      </div>

      <section className="mt-9 rounded-[18px] border border-line-soft bg-white/60 px-5 py-5">
        <h2 className="text-[16px] font-semibold text-ink">{t.facts}</h2>
        <dl className="mt-3 grid gap-x-7 gap-y-1.5 text-[14px] sm:grid-cols-2">
          {factsRows.map(([label, value]) => (
            <div key={label} className="flex items-baseline justify-between gap-3 border-b border-line-soft/50 py-1">
              <dt className="text-ink-3">{label}</dt>
              <dd className="text-right font-semibold text-ink">{value}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="mt-12">
        <h2 className="text-[20px] font-semibold">{t.buyH2}</h2>
        <ol className="mt-3 list-decimal pl-5 text-[15px] leading-relaxed text-ink-2">
          {t.buy.map((s) => <li key={s} className="mt-1.5">{s}</li>)}
        </ol>
      </section>

      <section className="mt-12 rounded-[18px] border border-line-soft bg-white/60 px-5 py-5">
        <h2 className="text-[17px] font-semibold text-ink">{t.prices} — {name}</h2>
        <ul className="mt-3 grid gap-x-7 gap-y-1.5 text-[14px] text-ink-2 sm:grid-cols-2">
          {priceItems.map((it) => (
            <li key={it.name} className="flex items-baseline justify-between gap-3 border-b border-line-soft/50 py-1">
              <span>{it.name}</span>
              <span className="whitespace-nowrap font-semibold text-[var(--accent-strong)]">{formatCatalogPrice(it.uah, it.kind, locale)}</span>
            </li>
          ))}
        </ul>
        <Link href="/3d-model-mista" className="mt-3 inline-block text-[13.5px] font-semibold text-[var(--accent-strong)] hover:underline">{t.more} →</Link>
      </section>

      <section className="mt-12">
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

      <h2 className="mt-14 text-[20px] font-semibold">{t.near}</h2>
      <ul className="mt-4 flex flex-wrap gap-2">
        {near.map(({ c, b }) => (
          <li key={c.slug}>
            <Link href={`/maps/${c.slug}`} className="inline-block rounded-full border border-line-soft bg-white/70 px-4 py-2 text-[13.5px] font-medium text-ink-2 transition hover:border-[var(--accent)] hover:text-ink">
              {isUA ? c.names.uk : c.names.en} · {nf.format(Math.round(b.km))} {t.km}
            </Link>
          </li>
        ))}
      </ul>
      <Link href="/maps" className="mt-6 inline-block text-[14px] font-semibold text-[var(--accent-strong)] hover:underline">{t.all} →</Link>
    </main>
  );
}
