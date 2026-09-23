import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { setRequestLocale } from "next-intl/server";
import { BASE, localeUrl } from "@/i18n/metadata";
import { routing, defaultLocale, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { GALLERY_ITEMS, GALLERY_BY_SLUG, relatedGallery, type GalleryItem, type GalleryKind } from "@/lib/gallery";
import { CITY_PAGES } from "@/lib/cityPages";
import { KEYCHAIN_PRICE_UAH, MAP_SIZE_PRICES_UAH, MAP_RELIEF_ADDON_UAH, mapPriceEur } from "@/lib/mapPrices";

/**
 * Окрема сторінка на кожне фото галереї (/foto/[slug]) — Google Картинки +
 * довгий хвіст. Контент — lib/gallery.ts (uk/en). de/pl/fr/es рендеряться
 * англійською з noindex і canonical на en (не плодимо дублі, фокус — UA).
 */
export function generateStaticParams() {
  return GALLERY_ITEMS.map((g) => ({ slug: g.slug }));
}

export const dynamicParams = false;

function resolveLocale(raw: string): AppLocale {
  return ((routing.locales as readonly string[]).includes(raw) ? raw : defaultLocale) as AppLocale;
}

const indexable = (l: AppLocale) => l === "uk" || l === "en";

type Copy = {
  gallery: string; real: string; render: string; buy: string; landing: string; price: string;
  what: string; related: string; cities: string; allPhotos: string; facts: [string, string][];
  kindName: string; cityHref: (slug: string) => string; cityLabel: (name: string) => string;
};

function copy(kind: GalleryKind, en: boolean): Copy {
  const M = MAP_SIZE_PRICES_UAH;
  const key = kind === "keychain" || kind === "heart" || kind === "group";
  const eur = (uah: number) => `≈€${mapPriceEur(uah)}`;
  const base = en
    ? { gallery: "Gallery", real: "Photo of a real print", render: "Builder render (before printing)", related: "Similar photos", cities: "Pick your city", allPhotos: "All gallery photos" }
    : { gallery: "Галерея", real: "Фото реального друку", render: "Рендер моделі з конструктора (до друку)", related: "Схожі фото", cities: "Оберіть своє місто", allPhotos: "Усі фото галереї" };

  if (key) {
    return {
      ...base,
      kindName: en ? "Map keychain" : "Брелок з картою",
      buy: en ? "Create a keychain like this" : "Створити такий брелок",
      landing: "/brelok",
      price: en ? `from ${eur(KEYCHAIN_PRICE_UAH)}` : `від ${KEYCHAIN_PRICE_UAH} ₴`,
      what: en ? "About this keychain" : "Про такий брелок",
      facts: en
        ? [["Product", "3D-printed keychain with a map of your place"], ["Price", `from ${eur(KEYCHAIN_PRICE_UAH)}`], ["Shapes", "rectangle, oval, capsule, tag, heart and more"], ["Text", "HOME, LOVE, a name or a date"], ["Material", "Eco PLA plastic"], ["Lead time", "2–4 business days + Nova Poshta delivery"]]
        : [["Виріб", "3D-друкований брелок з картою вашого місця"], ["Ціна", `від ${KEYCHAIN_PRICE_UAH} ₴`], ["Форми", "прямокутник, овал, капсула, жетон, серце та інші"], ["Напис", "HOME, LOVE, ім'я чи дата"], ["Матеріал", "пластик Eco PLA"], ["Терміни", "2–4 робочі дні + доставка Новою Поштою"]],
      cityHref: (s) => `/brelok/${s}`,
      cityLabel: (n) => (en ? `Keychain — ${n}` : `Брелок — ${n}`),
    };
  }
  if (kind === "panno") {
    return {
      ...base,
      kindName: en ? "Map tile panel" : "Панно з плиток",
      buy: en ? "Design a panel" : "Зібрати своє панно",
      landing: "/panno",
      price: en ? `from ${eur(M[80] * 4)} for 2×2` : `від ${M[80] * 4} ₴ за 2×2`,
      what: en ? "About the tile panel" : "Про панно з плиток",
      facts: en
        ? [["Product", "wall panel of 3D-printed hexagonal map tiles"], ["Price", `from ${eur(M[80] * 4)} (2×2 of 8 cm tiles)`], ["Tiles", "8 or 11 cm, streets match across seams"], ["Relief", `optional terrain relief +${MAP_RELIEF_ADDON_UAH} UAH`], ["Material", "Eco PLA plastic"], ["Lead time", "2–4 business days + Nova Poshta delivery"]]
        : [["Виріб", "панно на стіну з 3D-друкованих шестикутних плиток"], ["Ціна", `від ${M[80] * 4} ₴ (2×2 плитки по 8 см)`], ["Плитки", "8 або 11 см, вулиці збігаються на стиках"], ["Рельєф", `за бажанням, +${MAP_RELIEF_ADDON_UAH} ₴`], ["Матеріал", "пластик Eco PLA"], ["Терміни", "2–4 робочі дні + доставка Новою Поштою"]],
      cityHref: (s) => `/maps/${s}`,
      cityLabel: (n) => (en ? `3D map — ${n}` : `3D-мапа — ${n}`),
    };
  }
  return {
    ...base,
    kindName: en ? "3D city map" : "3D-мапа міста",
    buy: en ? "Create a map like this" : "Створити таку мапу",
    landing: "/maps",
    price: en ? `from ${eur(M[55])}` : `від ${M[55]} ₴`,
    what: en ? "About the 3D map" : "Про 3D-мапу",
    facts: en
      ? [["Product", "3D-printed map of any area on OpenStreetMap"], ["Sizes and prices", `5.5 cm ${eur(M[55])} · 8 cm ${eur(M[80])} · 11 cm ${eur(M[110])} · 15 cm ${eur(M[150])}`], ["My home", "a red marker on your building"], ["Relief", `optional terrain relief +${MAP_RELIEF_ADDON_UAH} UAH`], ["Material", "Eco PLA plastic"], ["Lead time", "2–4 business days + Nova Poshta delivery"]]
      : [["Виріб", "3D-друкована мапа будь-якої ділянки з OpenStreetMap"], ["Розміри і ціни", `5,5 см ${M[55]} ₴ · 8 см ${M[80]} ₴ · 11 см ${M[110]} ₴ · 15 см ${M[150]} ₴`], ["Мій дім", "червона позначка на вашому будинку"], ["Рельєф", `за бажанням, +${MAP_RELIEF_ADDON_UAH} ₴`], ["Матеріал", "пластик Eco PLA"], ["Терміни", "2–4 робочі дні + доставка Новою Поштою"]],
    cityHref: (s) => `/maps/${s}`,
    cityLabel: (n) => (en ? `3D map — ${n}` : `3D-мапа — ${n}`),
  };
}

function ctaHref(kind: GalleryKind) {
  if (kind === "keychain" || kind === "heart" || kind === "group") return "/keychains";
  if (kind === "panno") return "/panno";
  return "/create";
}

function pageTitle(g: GalleryItem, en: boolean) {
  const t = en ? g.title.en : g.title.uk;
  return `${t} | ${en ? "Monadruk gallery" : "Галерея Monadruk"}`;
}

export async function generateMetadata({ params }: { params: { locale: string; slug: string } }): Promise<Metadata> {
  const g = GALLERY_BY_SLUG[params.slug];
  if (!g) return {};
  const locale = resolveLocale(params.locale);
  const en = locale !== "uk";
  const path = `/foto/${g.slug}`;
  const title = pageTitle(g, en);
  const full = en ? g.desc.en : g.desc.uk;
  const description = full.length <= 158 ? full : `${full.slice(0, 155).replace(/\s+\S*$/, "")}…`;
  const img = `${BASE}${g.src}`;
  return {
    title: { absolute: title },
    description,
    alternates: {
      canonical: indexable(locale) ? localeUrl(locale, path) : localeUrl("en", path),
      languages: { uk: localeUrl("uk", path), en: localeUrl("en", path), "x-default": localeUrl("uk", path) },
    },
    robots: indexable(locale) ? undefined : { index: false, follow: true },
    openGraph: {
      title,
      description,
      url: localeUrl(locale, path),
      siteName: "Monadruk",
      type: "article",
      images: [{ url: img, width: g.w, height: g.h, alt: en ? g.alt.en : g.alt.uk }],
    },
    twitter: { card: "summary_large_image", title, description, images: [img] },
  };
}

export default function FotoPage({ params }: { params: { locale: string; slug: string } }) {
  const g = GALLERY_BY_SLUG[params.slug];
  if (!g) notFound();
  const locale = resolveLocale(params.locale);
  setRequestLocale(locale);
  const en = locale !== "uk";
  const L = en ? "en" : "uk";
  const c = copy(g.kind, en);
  const path = `/foto/${g.slug}`;
  const related = relatedGallery(g.slug, 6);

  // 6 міст ротацією за індексом фото — кожна фото-сторінка веде на різні міста.
  const gi = GALLERY_ITEMS.indexOf(g);
  const cities = Array.from({ length: 6 }, (_, k) => CITY_PAGES[(gi * 3 + k) % CITY_PAGES.length]);

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "ImageObject",
        contentUrl: `${BASE}${g.src}`,
        url: localeUrl(locale, path),
        name: g.title[L],
        caption: g.alt[L],
        description: g.desc[L],
        width: g.w,
        height: g.h,
        encodingFormat: "image/webp",
        creator: { "@type": "Organization", name: "Monadruk", url: BASE },
        creditText: "Monadruk",
        copyrightNotice: "© Monadruk",
        inLanguage: L,
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: c.gallery, item: localeUrl(locale, "/showcase") },
          { "@type": "ListItem", position: 3, name: g.title[L], item: localeUrl(locale, path) },
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
        <Link href="/showcase" className="hover:underline">{c.gallery}</Link>
        {" / "}
        <span className="text-ink">{c.kindName}</span>
      </nav>

      <h1 className="mt-4 text-[clamp(26px,3.6vw,40px)] leading-tight">{g.title[L]}</h1>

      <div className="mt-6 grid gap-8 lg:grid-cols-[1.15fr_1fr] lg:items-start">
        <figure className="overflow-hidden rounded-[22px] border border-line bg-paper">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={g.src} alt={g.alt[L]} width={g.w} height={g.h} className="h-auto w-full" fetchPriority="high" />
          <figcaption className="px-4 py-3 text-[13px] text-ink-3">
            {g.source === "real" ? c.real : c.render} · {g.alt[L]}
          </figcaption>
        </figure>

        <div>
          <p className="text-[15.5px] leading-relaxed text-ink-2">{g.desc[L]}</p>
          <p className="mt-4 text-[15px] font-semibold text-[var(--accent-strong)]">{c.kindName}: {c.price}</p>
          <div className="mt-5 flex flex-wrap gap-3">
            <Link
              href={ctaHref(g.kind)}
              className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-6 py-3 text-sm font-semibold text-white transition hover:opacity-90"
            >
              {c.buy} →
            </Link>
            <Link
              href={c.landing}
              className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] border border-line-soft bg-white/80 px-6 py-3 text-sm font-semibold text-ink transition hover:border-[var(--accent)]"
            >
              {c.kindName}
            </Link>
          </div>

          <section className="mt-8 rounded-[18px] border border-line-soft bg-white/60 px-5 py-4">
            <h2 className="text-[16px] font-semibold text-ink">{c.what}</h2>
            <dl className="mt-2 text-[14px]">
              {c.facts.map(([k, v]) => (
                <div key={k} className="flex items-baseline justify-between gap-3 border-b border-line-soft/50 py-1.5 last:border-0">
                  <dt className="text-ink-3">{k}</dt>
                  <dd className="text-right font-semibold text-ink">{v}</dd>
                </div>
              ))}
            </dl>
          </section>
        </div>
      </div>

      <section className="mt-12">
        <h2 className="text-[20px] font-semibold">{c.related}</h2>
        <ul className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3">
          {related.map((r) => (
            <li key={r.slug}>
              <Link href={`/foto/${r.slug}`} className="group block overflow-hidden rounded-[18px] border border-line bg-paper">
                <div className="aspect-square overflow-hidden">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={r.src} alt={r.alt[L]} loading="lazy" className="h-full w-full object-cover transition duration-500 group-hover:scale-[1.05]" />
                </div>
                <span className="block px-3 py-2.5 text-[13px] font-medium leading-snug text-ink">{r.title[L]}</span>
              </Link>
            </li>
          ))}
        </ul>
        <Link href="/showcase" className="mt-5 inline-block text-[14.5px] font-semibold text-[var(--accent-strong)] hover:underline">
          {c.allPhotos} →
        </Link>
      </section>

      <section className="mt-10">
        <h2 className="text-[18px] font-semibold">{c.cities}</h2>
        <ul className="mt-3 flex flex-wrap gap-2">
          {cities.map((x) => (
            <li key={x.slug}>
              <Link
                href={c.cityHref(x.slug)}
                className="inline-block rounded-full border border-line-soft bg-white/70 px-4 py-2 text-[13.5px] font-medium text-ink-2 transition hover:border-[var(--accent)] hover:text-ink"
              >
                {c.cityLabel(x.names[locale])}
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
