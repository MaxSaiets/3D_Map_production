import type { Metadata } from "next";
import { setRequestLocale } from "next-intl/server";
import { BASE, localeUrl } from "@/i18n/metadata";
import { routing, defaultLocale, localeMeta, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { CITY_PAGES } from "@/lib/cityPages";
import { KEYCHAIN_PRICE_UAH, MAP_SIZE_PRICES_UAH } from "@/lib/mapPrices";

/**
 * «Купити 3D-модель міста» (25.09.2026). Власник: на запити «купити модель міста»,
 * «купити 3d модель міста», «макет міста купити» monadruk у Google немає — сайт скрізь
 * казав «3D-мапа», а люди шукають «модель» і «макет». Комерційна посадкова під ці
 * формулювання: чесні ціни, що саме отримує покупець, міста, FAQ + JSON-LD.
 * Лише uk і en (інші мови — noindex, canonical на en), бо продаж — Україна.
 */
const PATH = "/3d-model-mista";
const INDEXED: AppLocale[] = ["uk", "en"];
const M = MAP_SIZE_PRICES_UAH;

type Copy = {
  title: string; description: string; h1: string; intro: string; cta: string; ctaKey: string;
  getTitle: string; get: string[]; pricesTitle: string; prices: [string, string][];
  howTitle: string; how: string[]; citiesTitle: string; cityLabel: (n: string) => string;
  faqTitle: string; faq: { q: string; a: string }[]; crumb: string;
};

const COPY: Record<"uk" | "en", Copy> = {
  uk: {
    title: "Купити 3D-модель міста — макет вашого району на замовлення від 350 ₴",
    description: "Купити 3D-модель міста чи макет району: справжні вулиці, будинки з реальною висотою, парки й річки. Будь-яке місто України чи світу, превʼю онлайн безкоштовно, друк від 350 ₴, доставка Новою Поштою.",
    h1: "Купити 3D-модель міста — макет будь-якого району",
    intro: "Обʼємна модель міста, яку можна поставити на стіл чи повісити на стіну: вулиці, будинки з реальною висотою, парки, річки й мости. Ви самі обираєте ділянку — центр Києва, свій двір у Львові, рідне село чи квартал у Варшаві — і за 1–2 хвилини бачите готову 3D-модель онлайн. Далі ми друкуємо її й надсилаємо Новою Поштою.",
    cta: "Створити 3D-модель міста",
    ctaKey: "Брелок з моделлю міста",
    getTitle: "Що ви отримуєте",
    get: [
      "Надрукований макет міста з екологічного пластику PLA: вулиці темною лінією, парки зеленим, будинки — обʼємні білі блоки.",
      "Форма моделі на вибір: квадрат, коло, шестикутник або серце; розмір від 5,5 до 20 см.",
      "Великі панно на стіну з 4 або 9 плиток із замками — для цілого міста.",
      "Напис на моделі: назва міста, дата, координати чи кілька слів.",
      "Якщо у вас є 3D-принтер — можна купити лише файл 3D-моделі (3MF) за 149 ₴.",
    ],
    pricesTitle: "Ціни",
    prices: [
      ["3D-модель S · 5,5 см", `${M[55]} ₴`],
      ["3D-модель M · 8 см", `${M[80]} ₴`],
      ["3D-модель L · 11 см", `${M[110]} ₴`],
      ["3D-модель XL · 15 см", `${M[150]} ₴`],
      ["Рельєф (пагорби, схили)", "+85 ₴"],
      ["Панно 2×2 (4 плитки по 8 см)", `${M[80] * 4} ₴`],
      ["Брелок з моделлю міста", `${KEYCHAIN_PRICE_UAH} ₴`],
      ["Файл 3D-моделі (3MF) для самодруку", "149 ₴"],
    ],
    howTitle: "Як замовити",
    how: [
      "Відкрийте конструктор і введіть адресу або оберіть місто — рамка стане на потрібну ділянку.",
      "Оберіть форму, розмір і напис; безкоштовне 3D-превʼю буде готове за 1–2 хвилини.",
      "Замовте друк: 2–4 робочі дні й доставка Новою Поштою по Україні.",
    ],
    citiesTitle: "3D-моделі міст України",
    cityLabel: (n) => `3D-модель: ${n}`,
    faqTitle: "Часті запитання",
    faq: [
      { q: "Чи можна купити 3D-модель саме мого міста чи села?", a: "Так. Модель будується з даних OpenStreetMap для будь-якої точки світу — від центру Києва до невеликого села. Якщо будинків на карті мало, модель буде простішою, це видно в безкоштовному превʼю ще до оплати." },
      { q: "Чим 3D-модель міста відрізняється від 3D-мапи?", a: "Нічим — це одне й те саме: обʼємний макет району з вулицями й будинками. Ми називаємо його і 3D-мапою, і 3D-моделлю, і макетом міста." },
      { q: "Скільки коштує макет міста?", a: `Від ${M[55]} ₴ за модель 5,5 см до ${M[150]} ₴ за 15 см; можна задати будь-який розмір від 5 до 20 см — ціна рахується автоматично. Панно з плиток — ціна плитки × кількість.` },
      { q: "Чи можна купити лише файл 3D-моделі міста?", a: "Так, файл 3MF для власного 3D-принтера коштує 149 ₴. Його можна відкрити в Bambu Studio чи PrusaSlicer." },
      { q: "Куди доставляєте?", a: "Надруковані моделі — лише по Україні, Новою Поштою. Друк займає 2–4 робочі дні." },
    ],
    crumb: "3D-модель міста",
  },
  en: {
    title: "Buy a 3D city model — a printed model of your neighbourhood",
    description: "Buy a 3D city model: real streets, buildings at real height, parks and rivers. Any city in the world, free online preview, printed from 350 ₴ (≈8 €), shipped within Ukraine; 3MF file for self-printing ≈3 €.",
    h1: "Buy a 3D city model of any neighbourhood",
    intro: "A 3D model of a city for your desk or wall: streets, buildings at real height, parks, rivers and bridges. You pick the area — Kyiv city centre, your street in Lviv or a block in Warsaw — and see the finished 3D model online in 1–2 minutes. Then we print it and ship it within Ukraine.",
    cta: "Create a 3D city model",
    ctaKey: "City model keychain",
    getTitle: "What you get",
    get: [
      "A printed city model in eco PLA: dark roads, green parks, raised white buildings.",
      "Your choice of shape — square, circle, hexagon or heart — from 5.5 to 20 cm.",
      "Large wall panels of 4 or 9 interlocking tiles for a whole city.",
      "Text on the model: city name, a date, coordinates or a few words.",
      "Have a 3D printer? Buy just the 3D model file (3MF) for ≈3 €.",
    ],
    pricesTitle: "Prices",
    prices: [
      ["3D model S · 5.5 cm", `${M[55]} ₴`],
      ["3D model M · 8 cm", `${M[80]} ₴`],
      ["3D model L · 11 cm", `${M[110]} ₴`],
      ["3D model XL · 15 cm", `${M[150]} ₴`],
      ["Terrain relief", "+85 ₴"],
      ["2×2 panel (4 tiles of 8 cm)", `${M[80] * 4} ₴`],
      ["City model keychain", `${KEYCHAIN_PRICE_UAH} ₴`],
      ["3D model file (3MF) for self-printing", "149 ₴ (≈3 €)"],
    ],
    howTitle: "How to order",
    how: [
      "Open the builder and type an address or pick a city — the frame snaps to the area.",
      "Choose shape, size and text; the free 3D preview is ready in 1–2 minutes.",
      "Order the print: 2–4 business days, Nova Poshta delivery within Ukraine.",
    ],
    citiesTitle: "3D models of Ukrainian cities",
    cityLabel: (n) => `3D model: ${n}`,
    faqTitle: "FAQ",
    faq: [
      { q: "Can I buy a 3D model of my own town?", a: "Yes. The model is built from OpenStreetMap data for any point in the world. If a place has few mapped buildings the model is simpler — you see that in the free preview before paying." },
      { q: "Is a 3D city model the same as a 3D map?", a: "Yes — it is the same thing: a raised model of a neighbourhood with streets and buildings." },
      { q: "Can I buy just the 3D model file?", a: "Yes, a 3MF file for your own 3D printer costs 149 ₴ (≈3 €) and opens in Bambu Studio or PrusaSlicer." },
      { q: "Where do you ship?", a: "Printed models ship within Ukraine only, via Nova Poshta, 2–4 business days after printing." },
    ],
    crumb: "3D city model",
  },
};

const resolve = (raw: string): AppLocale =>
  ((routing.locales as readonly string[]).includes(raw) ? raw : defaultLocale) as AppLocale;

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  const locale = resolve(params.locale);
  const indexed = INDEXED.includes(locale);
  const c = COPY[locale === "uk" ? "uk" : "en"];
  return {
    title: { absolute: `${c.title} | Monadruk` },
    description: c.description,
    robots: indexed ? undefined : { index: false, follow: true },
    alternates: {
      canonical: localeUrl(indexed ? locale : "en", PATH),
      languages: {
        ...Object.fromEntries(INDEXED.map((x) => [localeMeta[x].htmlLang, localeUrl(x, PATH)])),
        "x-default": localeUrl("uk", PATH),
      },
    },
    openGraph: { title: c.title, description: c.description, url: localeUrl(locale, PATH), siteName: "Monadruk", images: [{ url: `${BASE}/showcase/real-1.webp` }] },
  };
}

export default function CityModelPage({ params }: { params: { locale: string } }) {
  const locale = resolve(params.locale);
  setRequestLocale(locale);
  const c = COPY[locale === "uk" ? "uk" : "en"];

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Product",
        name: c.h1,
        description: c.description,
        image: [`${BASE}/showcase/real-1.webp`, `${BASE}/showcase/real-7.webp`],
        brand: { "@type": "Brand", name: "Monadruk" },
        offers: {
          "@type": "AggregateOffer",
          priceCurrency: "UAH",
          lowPrice: M[55],
          highPrice: M[150],
          offerCount: 4,
          availability: "https://schema.org/InStock",
          url: localeUrl(locale, PATH),
        },
      },
      {
        "@type": "FAQPage",
        mainEntity: c.faq.map((f) => ({ "@type": "Question", name: f.q, acceptedAnswer: { "@type": "Answer", text: f.a } })),
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: c.crumb, item: localeUrl(locale, PATH) },
        ],
      },
    ],
  };

  return (
    <main id="main-content" tabIndex={-1} className="mx-auto max-w-[980px] px-5 py-12 lg:py-16">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      <nav className="text-[13px] text-ink-3" aria-label="breadcrumb">
        <Link href="/" className="hover:underline">Monadruk</Link>
        {" / "}
        <span className="text-ink">{c.crumb}</span>
      </nav>

      <h1 className="mt-4 text-[clamp(28px,4vw,46px)] leading-tight">{c.h1}</h1>

      <div className="mt-6 grid gap-8 lg:grid-cols-[1fr_1fr] lg:items-start">
        <div>
          <p className="text-[15.5px] leading-relaxed text-ink-2">{c.intro}</p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link href="/create" className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-6 py-3 text-sm font-semibold text-white transition hover:opacity-90">
              {c.cta} →
            </Link>
            <Link href="/keychains" className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] border border-line-soft bg-white/80 px-6 py-3 text-sm font-semibold text-ink transition hover:border-[var(--accent)]">
              {c.ctaKey}
            </Link>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3">
          {["real-1", "real-7", "real-3", "real-8"].map((n) => (
            <figure key={n} className="overflow-hidden rounded-[18px] border border-line bg-paper">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={`/showcase/${n}.webp`} alt={c.h1} loading={n === "real-1" ? "eager" : "lazy"} className="aspect-square w-full object-cover" />
            </figure>
          ))}
        </div>
      </div>

      <section className="mt-12 grid gap-6 lg:grid-cols-2">
        <div className="rounded-[18px] border border-line-soft bg-white/60 px-5 py-5">
          <h2 className="text-[18px] font-semibold">{c.getTitle}</h2>
          <ul className="mt-3 list-disc space-y-1.5 pl-5 text-[14.5px] leading-relaxed text-ink-2">
            {c.get.map((s) => <li key={s}>{s}</li>)}
          </ul>
        </div>
        <div className="rounded-[18px] border border-line-soft bg-white/60 px-5 py-5">
          <h2 className="text-[18px] font-semibold">{c.pricesTitle}</h2>
          <dl className="mt-2 text-[14px]">
            {c.prices.map(([k, v]) => (
              <div key={k} className="flex items-baseline justify-between gap-3 border-b border-line-soft/50 py-1.5 last:border-0">
                <dt className="text-ink-2">{k}</dt>
                <dd className="font-semibold text-ink">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      <section className="mt-10">
        <h2 className="text-[20px] font-semibold">{c.howTitle}</h2>
        <ol className="mt-3 list-decimal space-y-1.5 pl-5 text-[15px] leading-relaxed text-ink-2">
          {c.how.map((s) => <li key={s}>{s}</li>)}
        </ol>
      </section>

      <section className="mt-10">
        <h2 className="text-[20px] font-semibold">{c.citiesTitle}</h2>
        <ul className="mt-4 flex flex-wrap gap-2">
          {CITY_PAGES.map((x) => (
            <li key={x.slug}>
              <Link href={`/maps/${x.slug}`} className="inline-block rounded-full border border-line-soft bg-white/70 px-4 py-2 text-[13.5px] font-medium text-ink-2 transition hover:border-[var(--accent)] hover:text-ink">
                {c.cityLabel(x.names[locale])}
              </Link>
            </li>
          ))}
        </ul>
      </section>

      <section className="mt-12">
        <h2 className="text-[20px] font-semibold">{c.faqTitle}</h2>
        <dl className="mt-4 flex flex-col gap-4">
          {c.faq.map((f) => (
            <div key={f.q}>
              <dt className="text-[15px] font-semibold text-ink">{f.q}</dt>
              <dd className="mt-1.5 text-[14.5px] leading-relaxed text-ink-2">{f.a}</dd>
            </div>
          ))}
        </dl>
      </section>
    </main>
  );
}
