import { BASE, localeUrl, priceValidUntil } from "@/i18n/metadata";
import type { AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import type { CityPage } from "@/lib/cityPages";
import { RAIONS_BY_CITY, bearingFrom } from "@/lib/cityRaions";
import { STREET_PAGES_BY_CITY, type CityStreet } from "@/lib/streetPages";
import { mapPriceRange } from "@/lib/mapPrices";
import { cityFaq, contentLocale } from "@/lib/cityLanding";
import MapRenderFigure, { mapRenderUrl } from "@/components/MapRenderFigure";

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return Math.abs(h);
}

/**
 * 27.09.2026: сторінка відомої вулиці (lib/cityStreets.ts, факти з Wikidata),
 * uk/en. Унікальність — факти (довжина, на честь кого, рік), обчислення (район,
 * відстань від центру, сусідні вулиці з метрами), 3 варіанти структури абзацу
 * за хешем slug. Джерело фактів — посилання на статтю Вікіпедії.
 */
export default function StreetView({ street, city, locale }: { street: CityStreet; city: CityPage; locale: AppLocale }) {
  const isUA = locale === "uk";
  const cityName = isUA ? city.names.uk : city.names.en;
  const sName = isUA ? street.uk : street.en || street.uk;
  const path = `/maps/${city.slug}/${street.slug}`;
  const nf = new Intl.NumberFormat(isUA ? "uk-UA" : "en-US", { maximumFractionDigits: 1 });
  const range = mapPriceRange(locale);
  const faq = cityFaq(contentLocale(locale), cityName, "podarunok");
  const fromCentre = bearingFrom(city.center, street.center);
  const raion = (RAIONS_BY_CITY[city.slug] ?? [])
    .map((r) => ({ r, km: bearingFrom(street.center, r.center).km }))
    .sort((a, b) => a.km - b.km)[0]?.r;
  const near = (STREET_PAGES_BY_CITY[city.slug] ?? [])
    .filter((s) => s.slug !== street.slug)
    .map((s) => ({ s, m: Math.round(bearingFrom(street.center, s.center).km * 1000) }))
    .sort((a, b) => a.m - b.m)
    .slice(0, 8);
  const v = hash(street.slug) % 3;
  const len = street.lengthM;
  const createHref = `/create?lat=${street.center[0]}&lon=${street.center[1]}`;
  const wikiUrl = `https://uk.wikipedia.org/wiki/${encodeURIComponent(street.wiki.replace(/ /g, "_"))}`;
  const renderId = `${city.slug}--${street.slug}`;
  // Назва всередині речення: «Вулиця Івана Франка» → «вулиця Івана Франка» (родові терміни з малої).
  const nameMid = isUA
    ? sName.replace(/^(Вулиця|Площа|Алея|Бульвар|Проспект|Узвіз|Провулок|Майдан|Набережна|Шосе)/, (w) => w.toLowerCase())
    : sName;

  const sizeUk = !len
    ? "Для вулиці найкраще працює мапа 8–11 см: рамку ставимо так, щоб вулиця йшла через усю модель."
    : len <= 600
      ? `Вулиця коротка (близько ${nf.format(len)} м), тож уся вона разом із сусідніми кварталами вміщується в мапу 8 см.`
      : len <= 1500
        ? `При довжині близько ${nf.format(len)} м вулиця повністю лягає на мапу 11–15 см — видно її від початку до кінця.`
        : `Вулиця довга (близько ${nf.format(len / 1000)} км), тому зазвичай беруть найвідоміший відрізок або друкують панно з кількох плиток.`;
  const sizeEn = !len
    ? "An 8–11 cm map works best for a street: we place the frame so the street runs across the whole model."
    : len <= 600
      ? `The street is short (about ${nf.format(len)} m), so all of it plus the neighbouring blocks fits an 8 cm map.`
      : len <= 1500
        ? `At about ${nf.format(len)} m the whole street fits an 11–15 cm map, end to end.`
        : `The street is long (about ${nf.format(len / 1000)} km), so people usually pick its best-known stretch or print a multi-tile panel.`;

  const whereUk = fromCentre.km < 1
    ? `Вулиця в самому центрі міста ${cityName}`
    : `Вулиця розташована приблизно за ${nf.format(fromCentre.km)} км ${fromCentre.uk} від центру міста ${cityName}`;
  const whereEn = fromCentre.km < 1
    ? `The street is right in the centre of ${cityName}`
    : `The street lies about ${nf.format(fromCentre.km)} km ${fromCentre.en} of central ${cityName}`;

  const namedUk = street.namedAfter ? `Назва — на честь: ${street.namedAfter}.` : "";
  const namedEn = street.namedAfter ? `The name honours ${street.namedAfter}.` : "";
  const sinceUk = street.since ? ` Відома з ${street.since} року.` : "";
  const sinceEn = street.since ? ` Known since ${street.since}.` : "";

  const t = isUA
    ? {
        h1: `3D-модель: ${sName}, ${cityName} — купити макет вулиці`,
        paras: [
          [
            `Купити 3D-мапу, на якій ${nameMid}, — це спосіб зберегти місце, яке багато важить: де ви жили, познайомились чи відкрили свою справу. Модель друкуємо з реальних даних OpenStreetMap: будинки з висотами, тротуари, дерева в парках і водойми.`,
            `${whereUk}${raion ? ` (${raion.uk})` : ""}. ${namedUk}${sinceUk}`,
            sizeUk,
          ],
          [
            `${whereUk}${raion ? `, у межах району: ${raion.uk}` : ""}. ${namedUk}${sinceUk}`,
            `Замовити макет, на якому ${nameMid}, можна за кілька хвилин: конструктор відкривається одразу на ній, ви бачите превʼю безкоштовно й оформлюєте друк з доставкою Новою Поштою.`,
            sizeUk,
          ],
          [
            sizeUk,
            `${whereUk}. ${namedUk}${sinceUk}`,
            `3D-модель, на якій ${nameMid}, — частий подарунок на річницю чи новосілля: людям важливо впізнати свій будинок, і на моделі його справді видно.`,
          ],
        ][v],
        cta: "Створити 3D-мапу вулиці",
        ctaK: "Брелок з мапою вулиці",
        facts: "Факти про вулицю",
        fCity: "Місто", fRaion: "Район", fLen: "Довжина", fNamed: "Названа на честь", fSince: "Відома з", fCentre: "Від центру",
        buyH2: "Скільки коштує 3D-мапа вулиці",
        buy: `Мапа — від ${range.low} ₴ до ${range.high} ₴ залежно від розміру, брелок з мапою — від 170 ₴, файл 3MF для самодруку — 149 ₴. Друк 2–4 робочі дні.`,
        near: "Відомі вулиці поруч", m: "м",
        src: "Джерело фактів: Вікіпедія",
        raionLink: raion ? `3D-модель району: ${raion.uk}` : "",
        cityLink: `3D-модель міста ${cityName}`,
        faqT: "Часті запитання",
      }
    : {
        h1: `3D model of ${sName}, ${cityName} — buy a street map`,
        paras: [
          [
            `A 3D map of ${sName} is a way to keep a place that matters — where you lived, met or started your business. We print it from real OpenStreetMap data: buildings with real heights, pavements, park trees and water.`,
            `${whereEn}${raion ? ` (${raion.en})` : ""}. ${namedEn}${sinceEn}`,
            sizeEn,
          ],
          [
            `${whereEn}${raion ? `, within ${raion.en}` : ""}. ${namedEn}${sinceEn}`,
            `Ordering a model of ${sName} takes minutes: the builder opens right on the street, the preview is free and the print ships to you.`,
            sizeEn,
          ],
          [
            sizeEn,
            `${whereEn}. ${namedEn}${sinceEn}`,
            `A 3D model of ${sName} is a popular anniversary or housewarming gift: people love recognising their own building, and on the model it really shows.`,
          ],
        ][v],
        cta: "Create a 3D map of this street",
        ctaK: "Street map keychain",
        facts: "Street facts",
        fCity: "City", fRaion: "District", fLen: "Length", fNamed: "Named after", fSince: "Known since", fCentre: "From the centre",
        buyH2: "How much a street 3D map costs",
        buy: `Maps cost ${range.low}–${range.high} ${range.currency} depending on size; a 3MF file for self-printing is also available. Production takes 2–4 working days.`,
        near: "Well-known streets nearby", m: "m",
        src: "Source of facts: Wikipedia (Ukrainian)",
        raionLink: raion ? `3D model of ${raion.en}` : "",
        cityLink: `3D model of ${cityName}`,
        faqT: "FAQ",
      };

  const facts: [string, string][] = [
    [t.fCity, cityName],
    ...(raion ? ([[t.fRaion, isUA ? raion.uk : raion.en]] as [string, string][]) : []),
    [t.fCentre, `${nf.format(fromCentre.km)} ${isUA ? "км" : "km"}`],
    ...(len ? ([[t.fLen, `${nf.format(len)} ${t.m}`]] as [string, string][]) : []),
    ...(street.namedAfter ? ([[t.fNamed, street.namedAfter]] as [string, string][]) : []),
    ...(street.since ? ([[t.fSince, String(street.since)]] as [string, string][]) : []),
  ];

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Product",
        name: t.h1,
        description: t.paras[0],
        image: mapRenderUrl(renderId, BASE) ?? `${BASE}/real/map-1.webp`,
        brand: { "@type": "Brand", name: "Monadruk" },
        sku: `MND-STREET-${city.slug}-${street.slug}`,
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
          { "@type": "ListItem", position: 4, name: sName, item: localeUrl(locale, path) },
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
        <span className="text-ink">{sName}</span>
      </nav>
      <h1 className="mt-5 text-[clamp(28px,4vw,46px)] leading-tight">{t.h1}</h1>
      {t.paras.map((p, i) => (
        <p key={i} className={`${i === 0 ? "mt-5" : "mt-3"} text-[15px] leading-relaxed text-ink-2`}>{p}</p>
      ))}

      <div className="mt-8 flex flex-wrap gap-3">
        <Link href={createHref} className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-6 py-3 text-sm font-semibold text-white transition hover:opacity-90">
          {t.cta}
        </Link>
        <Link href="/keychains" className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] border border-line-soft bg-white/80 px-6 py-3 text-sm font-semibold text-ink transition hover:border-[var(--accent)]">
          {t.ctaK}
        </Link>
      </div>

      <MapRenderFigure id={renderId} name={`${sName}, ${cityName}`} isUA={isUA} />

      <section className="mt-9 rounded-[18px] border border-line-soft bg-white/60 px-5 py-5">
        <h2 className="text-[16px] font-semibold text-ink">{t.facts}</h2>
        <dl className="mt-3 grid gap-x-7 gap-y-1.5 text-[14px] sm:grid-cols-2">
          {facts.map(([label, value]) => (
            <div key={label} className="flex items-baseline justify-between gap-3 border-b border-line-soft/50 py-1">
              <dt className="text-ink-3">{label}</dt>
              <dd className="text-right font-semibold text-ink">{value}</dd>
            </div>
          ))}
        </dl>
        <a href={wikiUrl} rel="noopener nofollow" target="_blank" className="mt-3 inline-block text-[12.5px] text-ink-3 underline underline-offset-2">
          {t.src}
        </a>
      </section>

      <section className="mt-10">
        <h2 className="text-[20px] font-semibold">{t.buyH2}</h2>
        <p className="mt-3 text-[15px] leading-relaxed text-ink-2">{t.buy}</p>
      </section>

      {near.length > 0 && (
        <section className="mt-10">
          <h2 className="text-[18px] font-semibold text-ink">{t.near}</h2>
          <ul className="mt-3 flex flex-wrap gap-2">
            {near.map(({ s, m }) => (
              <li key={s.slug}>
                <Link href={`/maps/${city.slug}/${s.slug}`} className="inline-block rounded-full border border-line-soft bg-white/70 px-4 py-2 text-[13.5px] font-medium text-ink-2 transition hover:border-[var(--accent)] hover:text-ink">
                  {isUA ? s.uk : s.en || s.uk} · {nf.format(m)} {t.m}
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
        {raion && (
          <Link href={`/maps/${city.slug}/${raion.slug}`} className="text-[14.5px] font-semibold text-[var(--accent-strong)] hover:underline">{t.raionLink} →</Link>
        )}
        <Link href={`/maps/${city.slug}`} className="text-[14.5px] font-semibold text-[var(--accent-strong)] hover:underline">{t.cityLink} →</Link>
      </section>
    </main>
  );
}
