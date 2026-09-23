import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { setRequestLocale } from "next-intl/server";
import { BASE, localeUrl } from "@/i18n/metadata";
import { routing, defaultLocale, localeMeta, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { GALLERY_ITEMS, GALLERY_BY_SLUG, GALLERY_LOCALES, relatedGallery, type GalleryItem, type GalleryKind, type GalleryLocale } from "@/lib/gallery";
import { CITY_PAGES } from "@/lib/cityPages";
import { KEYCHAIN_PRICE_UAH, MAP_SIZE_PRICES_UAH, mapPriceEur } from "@/lib/mapPrices";

/**
 * Окрема сторінка на кожне фото галереї (/foto/[slug]) — Google Картинки +
 * довгий хвіст. Контент — lib/gallery.ts (uk/en) + lib/galleryI18n.ts (de/pl/fr/es);
 * усі 6 мов — справжні переклади, індексуються.
 */
export function generateStaticParams() {
  return GALLERY_ITEMS.map((g) => ({ slug: g.slug }));
}

export const dynamicParams = false;

function resolveLocale(raw: string): AppLocale {
  return ((routing.locales as readonly string[]).includes(raw) ? raw : defaultLocale) as AppLocale;
}

type L = GalleryLocale;
type KindKey = "key" | "panno" | "map";
const kindKey = (k: GalleryKind): KindKey => (k === "panno" ? "panno" : k === "map" ? "map" : "key");

// Інтерфейс сторінки шістьма мовами. Ціни: ₴ для uk, ≈€ для решти (друк — доставка лише по Україні).
const UI: Record<L, {
  gallery: string; real: string; render: string; related: string; cities: string; allPhotos: string; brand: string;
  kind: Record<KindKey, string>; buy: Record<KindKey, string>; what: Record<KindKey, string>;
  city: Record<KindKey, string>; lbl: string[]; from: string; lead: string; material: string;
}> = {
  uk: { gallery: "Галерея", real: "Фото реального друку", render: "Рендер моделі з конструктора (до друку)", related: "Схожі фото", cities: "Оберіть своє місто", allPhotos: "Усі фото галереї", brand: "Галерея Monadruk",
    kind: { key: "Брелок з картою", panno: "Панно з плиток", map: "3D-мапа міста" }, buy: { key: "Створити такий брелок", panno: "Зібрати своє панно", map: "Створити таку мапу" },
    what: { key: "Про такий брелок", panno: "Про панно з плиток", map: "Про 3D-мапу" }, city: { key: "Брелок", panno: "3D-мапа", map: "3D-мапа" },
    lbl: ["Виріб", "Ціна", "Файл для самодруку", "Матеріал", "Терміни"], from: "від", lead: "2–4 робочі дні + доставка Новою Поштою", material: "пластик Eco PLA" },
  en: { gallery: "Gallery", real: "Photo of a real print", render: "Builder render (before printing)", related: "Similar photos", cities: "Pick your city", allPhotos: "All gallery photos", brand: "Monadruk gallery",
    kind: { key: "Map keychain", panno: "Map tile panel", map: "3D city map" }, buy: { key: "Create a keychain like this", panno: "Design a panel", map: "Create a map like this" },
    what: { key: "About this keychain", panno: "About the tile panel", map: "About the 3D map" }, city: { key: "Keychain", panno: "3D map", map: "3D map" },
    lbl: ["Product", "Price", "File to print yourself", "Material", "Lead time"], from: "from", lead: "2–4 business days, delivery within Ukraine", material: "Eco PLA plastic" },
  de: { gallery: "Galerie", real: "Foto eines echten Drucks", render: "Rendering aus dem Konfigurator (vor dem Druck)", related: "Ähnliche Fotos", cities: "Wähle deine Stadt", allPhotos: "Alle Fotos der Galerie", brand: "Monadruk Galerie",
    kind: { key: "Karten-Anhänger", panno: "Kachelbild mit Karte", map: "3D-Stadtkarte" }, buy: { key: "So einen Anhänger erstellen", panno: "Eigenes Kachelbild gestalten", map: "So eine Karte erstellen" },
    what: { key: "Über diesen Anhänger", panno: "Über das Kachelbild", map: "Über die 3D-Karte" }, city: { key: "Anhänger", panno: "3D-Karte", map: "3D-Karte" },
    lbl: ["Produkt", "Preis", "Datei zum Selbstdrucken", "Material", "Lieferzeit"], from: "ab", lead: "2–4 Werktage, Versand innerhalb der Ukraine", material: "Eco-PLA-Kunststoff" },
  pl: { gallery: "Galeria", real: "Zdjęcie prawdziwego wydruku", render: "Render z kreatora (przed drukiem)", related: "Podobne zdjęcia", cities: "Wybierz swoje miasto", allPhotos: "Wszystkie zdjęcia galerii", brand: "Galeria Monadruk",
    kind: { key: "Brelok z mapą", panno: "Panel z płytek z mapą", map: "Mapa 3D miasta" }, buy: { key: "Stwórz taki brelok", panno: "Zaprojektuj swój panel", map: "Stwórz taką mapę" },
    what: { key: "O takim breloku", panno: "O panelu z płytek", map: "O mapie 3D" }, city: { key: "Brelok", panno: "Mapa 3D", map: "Mapa 3D" },
    lbl: ["Produkt", "Cena", "Plik do własnego druku", "Materiał", "Czas realizacji"], from: "od", lead: "2–4 dni robocze, wysyłka na terenie Ukrainy", material: "tworzywo Eco PLA" },
  fr: { gallery: "Galerie", real: "Photo d'une impression réelle", render: "Rendu du configurateur (avant impression)", related: "Photos similaires", cities: "Choisis ta ville", allPhotos: "Toutes les photos de la galerie", brand: "Galerie Monadruk",
    kind: { key: "Porte-clés carte", panno: "Tableau de tuiles carte", map: "Carte 3D de ville" }, buy: { key: "Créer un porte-clés comme celui-ci", panno: "Composer ton tableau", map: "Créer une carte comme celle-ci" },
    what: { key: "À propos de ce porte-clés", panno: "À propos du tableau de tuiles", map: "À propos de la carte 3D" }, city: { key: "Porte-clés", panno: "Carte 3D", map: "Carte 3D" },
    lbl: ["Produit", "Prix", "Fichier à imprimer soi-même", "Matière", "Délai"], from: "dès", lead: "2–4 jours ouvrés, livraison en Ukraine", material: "plastique Eco PLA" },
  es: { gallery: "Galería", real: "Foto de una impresión real", render: "Render del configurador (antes de imprimir)", related: "Fotos similares", cities: "Elige tu ciudad", allPhotos: "Todas las fotos de la galería", brand: "Galería Monadruk",
    kind: { key: "Llavero con mapa", panno: "Panel de baldosas con mapa", map: "Mapa 3D de ciudad" }, buy: { key: "Crear un llavero así", panno: "Diseñar tu panel", map: "Crear un mapa así" },
    what: { key: "Sobre este llavero", panno: "Sobre el panel de baldosas", map: "Sobre el mapa 3D" }, city: { key: "Llavero", panno: "Mapa 3D", map: "Mapa 3D" },
    lbl: ["Producto", "Precio", "Archivo para imprimir tú", "Material", "Plazo"], from: "desde", lead: "2–4 días hábiles, envío dentro de Ucrania", material: "plástico Eco PLA" },
};

type Copy = {
  gallery: string; real: string; render: string; buy: string; landing: string; price: string;
  what: string; related: string; cities: string; allPhotos: string; facts: [string, string][];
  kindName: string; cityHref: (slug: string) => string; cityLabel: (name: string) => string;
};

const FILE_UAH = 149; // друк-файл 3MF (памʼять file-sale-149)

function copy(kind: GalleryKind, l: L): Copy {
  const u = UI[l];
  const k = kindKey(kind);
  const M = MAP_SIZE_PRICES_UAH;
  const money = (uah: number) => (l === "uk" ? `${uah} ₴` : `≈${mapPriceEur(uah)} €`);
  const priceUah = k === "key" ? KEYCHAIN_PRICE_UAH : k === "panno" ? M[80] * 4 : M[55];
  const price = `${u.from} ${money(priceUah)}${k === "panno" ? " (2×2)" : ""}`;
  return {
    gallery: u.gallery, real: u.real, render: u.render, related: u.related, cities: u.cities, allPhotos: u.allPhotos,
    kindName: u.kind[k], buy: u.buy[k], what: u.what[k],
    landing: k === "key" ? "/brelok" : k === "panno" ? "/panno" : "/maps",
    price,
    facts: [
      [u.lbl[0], u.kind[k]],
      [u.lbl[1], price],
      [u.lbl[2], `3MF, ${FILE_UAH} ₴${l === "uk" ? "" : ` (≈${mapPriceEur(FILE_UAH)} €)`}`],
      [u.lbl[3], u.material],
      [u.lbl[4], u.lead],
    ],
    cityHref: (s) => (k === "key" ? `/brelok/${s}` : `/maps/${s}`),
    cityLabel: (n) => `${u.city[k]} — ${n}`,
  };
}

function ctaHref(kind: GalleryKind) {
  if (kind === "keychain" || kind === "heart" || kind === "group") return "/keychains";
  if (kind === "panno") return "/panno";
  return "/create";
}

function pageTitle(g: GalleryItem, l: L) {
  return `${g.title[l]} | ${UI[l].brand}`;
}

export async function generateMetadata({ params }: { params: { locale: string; slug: string } }): Promise<Metadata> {
  const g = GALLERY_BY_SLUG[params.slug];
  if (!g) return {};
  const locale = resolveLocale(params.locale);
  const l = locale as L;
  const path = `/foto/${g.slug}`;
  const title = pageTitle(g, l);
  const full = g.desc[l];
  const description = full.length <= 158 ? full : `${full.slice(0, 155).replace(/\s+\S*$/, "")}…`;
  const img = `${BASE}${g.src}`;
  return {
    title: { absolute: title },
    description,
    alternates: {
      canonical: localeUrl(locale, path),
      languages: {
        ...Object.fromEntries(GALLERY_LOCALES.map((x) => [localeMeta[x].htmlLang, localeUrl(x, path)])),
        "x-default": localeUrl("uk", path),
      },
    },
    openGraph: {
      title,
      description,
      url: localeUrl(locale, path),
      siteName: "Monadruk",
      type: "article",
      images: [{ url: img, width: g.w, height: g.h, alt: g.alt[l] }],
    },
    twitter: { card: "summary_large_image", title, description, images: [img] },
  };
}

export default function FotoPage({ params }: { params: { locale: string; slug: string } }) {
  const g = GALLERY_BY_SLUG[params.slug];
  if (!g) notFound();
  const locale = resolveLocale(params.locale);
  setRequestLocale(locale);
  const L = locale as GalleryLocale;
  const c = copy(g.kind, L);
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
