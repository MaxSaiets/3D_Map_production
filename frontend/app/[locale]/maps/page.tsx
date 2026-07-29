import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { pageMetadata, localeUrl } from "@/i18n/metadata";
import { routing, defaultLocale, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { CITY_PAGES, WORLD_CITY_PAGES } from "@/lib/cityPages";

const MAPS_FAQ: Record<"uk" | "en", { q: string; a: string }[]> = {
  uk: [
    { q: "Чи є моє місто у списку?", a: "Список постійно розширюється — уже понад 20 найбільших міст України. Якщо вашого міста немає, оберіть будь-яку точку прямо в конструкторі: працює для всього світу, не лише зі списку." },
    { q: "Чим сторінка міста відрізняється від конструктора?", a: "Сторінка міста показує факти (населення, річку, візитівку) і популярні райони з готовими налаштуваннями. Конструктор — де фактично обираєте ділянку і генеруєте модель." },
    { q: "Чи можна замовити мапу району, якого немає серед прикладів?", a: "Так — приклади районів лише прискорюють старт. У конструкторі можна обрати будь-яку ділянку в межах міста чи за його межами." },
  ],
  en: [
    { q: "Is my city on the list?", a: "The list keeps growing — over 20 of Ukraine's largest cities already. If your city isn't listed, pick any point directly in the builder: it works worldwide, not just from the list." },
    { q: "How is a city page different from the builder?", a: "A city page shows facts (population, river, landmark) and popular districts with ready settings. The builder is where you actually pick the area and generate the model." },
    { q: "Can I order a district that isn't in the examples?", a: "Yes — the example districts just speed up the start. In the builder you can pick any area within or beyond the city." },
  ],
};

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  return pageMetadata({ locale: params.locale, path: "/maps", ns: "mapsMeta" });
}

export default async function MapsIndexPage({ params }: { params: { locale: string } }) {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  setRequestLocale(locale);
  const t = await getTranslations({ locale, namespace: "mapsMeta" });
  const tc = await getTranslations({ locale, namespace: "cityPages" });
  const isUA = locale === "uk";
  const faq = MAPS_FAQ[isUA ? "uk" : "en"];

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "CollectionPage",
        name: t("title"),
        description: t("description"),
        url: localeUrl(locale, "/maps"),
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: tc("breadcrumb"), item: localeUrl(locale, "/maps") },
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

  return (
    <main id="main-content" tabIndex={-1} className="mx-auto max-w-[920px] px-5 py-14 lg:py-20">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      <h1 className="text-[clamp(28px,4vw,46px)] leading-tight">{t("title")}</h1>
      <p className="mt-4 max-w-[640px] text-[15px] leading-relaxed text-ink-2">{t("description")}</p>

      <section className="mt-10 max-w-[680px]">
        <h2 className="text-[20px] font-semibold">{t("h2how")}</h2>
        <p className="mt-3 text-[15px] leading-relaxed text-ink-2">{t("pHow")}</p>
      </section>

      <h2 className="mt-12 text-[20px] font-semibold">{t("h2cities")}</h2>
      <ul className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {CITY_PAGES.map((c) => (
          <li key={c.slug}>
            <Link
              href={`/maps/${c.slug}`}
              className="block rounded-[18px] border border-line-soft bg-white/70 px-4 py-3.5 text-[15px] font-semibold text-ink transition hover:border-[var(--accent)]"
            >
              {c.names[locale]}
            </Link>
          </li>
        ))}
      </ul>

      {/* SEO-РОЗШИРЕННЯ НА ЄС: міста Європи. Для de/pl/fr/es-аудиторії це
          єдиний контент про ЇХНІ міста — раніше всі локалі бачили лише
          українські. Конструктор і так працює для всього світу. */}
      <h2 className="mt-12 text-[20px] font-semibold">
        {isUA ? "Міста Європи" : "European cities"}
      </h2>
      <p className="mt-2 max-w-[640px] text-[14.5px] leading-relaxed text-ink-2">
        {isUA
          ? "Конструктор працює для будь-якої точки планети. Ось популярні європейські міста з готовими налаштуваннями."
          : "The builder works for any point on the planet. Here are popular European cities with ready settings."}
      </p>
      <ul className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {WORLD_CITY_PAGES.map((c) => (
          <li key={c.slug}>
            <Link
              href={`/maps/${c.slug}`}
              className="block rounded-[18px] border border-line-soft bg-white/70 px-4 py-3.5 text-[15px] font-semibold text-ink transition hover:border-[var(--accent)]"
            >
              {c.names[locale]}
            </Link>
          </li>
        ))}
      </ul>

      {/* SEO-FIX: перехресний лінк на кластер брелків. Раніше хаб /brelok не мав
          жодного входу з головних сторінок → 24 сторінки /brelok/{місто} висіли
          в GSC як «Виявлено — наразі не проіндексовано». */}
      <p className="mt-6 text-[15px] leading-relaxed text-ink-2">
        {isUA ? "Потрібен не панно на стіну, а щось кишенькове? " : "Looking for something pocket-sized instead of a wall piece? "}
        <Link href="/brelok" className="font-semibold text-ink underline underline-offset-2 hover:text-[var(--accent-strong)]">
          {isUA ? "Брелки з мапою по містах" : "City map keychains"}
        </Link>
        {isUA ? " — той самий район, але на ключах." : " — the same district, but on your keys."}
      </p>

      <section className="mt-14 max-w-[680px]">
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

      <section className="mt-10 max-w-[680px] rounded-[18px] border border-line-soft bg-white/60 px-5 py-6">
        <h2 className="text-[20px] font-semibold">{t("h2gift")}</h2>
        <p className="mt-3 text-[15px] leading-relaxed text-ink-2">{t("pGift")}</p>
        <div className="mt-5 flex flex-wrap gap-3">
          <Link href="/create" className="inline-flex min-h-[44px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-5 py-2.5 text-sm font-semibold text-white transition hover:opacity-90">
            {t("ctaCreate")}
          </Link>
          <Link href="/keychains" className="inline-flex min-h-[44px] items-center justify-center rounded-[22px] border border-line-soft bg-white/80 px-5 py-2.5 text-sm font-semibold text-ink transition hover:border-[var(--accent)]">
            {t("ctaKeychain")}
          </Link>
        </div>
      </section>
    </main>
  );
}
