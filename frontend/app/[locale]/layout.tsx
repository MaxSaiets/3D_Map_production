import type { Metadata } from "next";
import { Cormorant_Garamond, Manrope, JetBrains_Mono } from "next/font/google";
import { notFound } from "next/navigation";
import { NextIntlClientProvider } from "next-intl";
import { setRequestLocale, getMessages, getTranslations } from "next-intl/server";
import "../globals.css";
import { ContactWidget } from "@/components/ContactWidget";
import { AuthProvider } from "@/components/AuthProvider";
import SiteAnalytics from "@/components/SiteAnalytics";
import { ToastHost } from "@/components/ToastHost";
import { GlobalFooter } from "@/components/SiteFooter";
import { routing, locales, localeMeta, type AppLocale } from "@/i18n/routing";
import { BUSINESS } from "@/lib/legal";
import { mapPriceRange } from "@/lib/mapPrices";
import { priceValidUntil } from "@/i18n/metadata";

const BASE = "https://monadruk.com";

// Self-hosted via next/font: no render-blocking external stylesheet, no CLS.
const serif = Cormorant_Garamond({
  subsets: ["latin"], weight: ["400", "500", "600"], style: ["normal", "italic"],
  variable: "--font-serif", display: "swap",
});
const sans = Manrope({
  subsets: ["latin", "cyrillic"], weight: ["300", "400", "500", "600", "700"],
  variable: "--font-sans", display: "swap",
});
const mono = JetBrains_Mono({
  subsets: ["latin"], weight: ["400", "500"], variable: "--font-mono", display: "swap",
});

export function generateStaticParams() {
  return routing.locales.map((locale) => ({ locale }));
}

export const viewport = { themeColor: "#2E4A3A" };

function localePath(locale: AppLocale, path = "") {
  return locale === routing.defaultLocale ? `${BASE}${path || "/"}` : `${BASE}/${locale}${path}`;
}

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  const locale = ((routing.locales as readonly string[]).includes(params.locale) ? params.locale : routing.defaultLocale) as AppLocale;
  const t = await getTranslations({ locale, namespace: "meta" });

  // hreflang map: every locale → its homepage URL, plus x-default.
  const languages: Record<string, string> = {};
  for (const l of locales) languages[localeMeta[l].htmlLang] = localePath(l);
  languages["x-default"] = `${BASE}/`;

  return {
    metadataBase: new URL(BASE),
    title: { default: t("homeTitle"), template: "%s · Monadruk" },
    description: t("homeDescription"),
    applicationName: "Monadruk",
    authors: [{ name: "Monadruk" }],
    creator: "Monadruk",
    keywords: t("keywords").split(",").map((s) => s.trim()),
    alternates: { canonical: localePath(locale), languages },
    verification: {
      ...(process.env.NEXT_PUBLIC_GOOGLE_SITE_VERIFICATION
        ? { google: process.env.NEXT_PUBLIC_GOOGLE_SITE_VERIFICATION }
        : {}),
      other: { "p:domain_verify": "d57db6841e30b47e8c24e654e0b0e049" },
    },
    robots: {
      index: true, follow: true,
      googleBot: { index: true, follow: true, "max-image-preview": "large" },
    },
    openGraph: {
      title: t("ogTitle"),
      description: t("ogDescription"),
      url: localePath(locale),
      siteName: "Monadruk",
      type: "website",
      locale: localeMeta[locale].ogLocale,
      alternateLocale: locales.filter((l) => l !== locale).map((l) => localeMeta[l].ogLocale),
      // Робочий КОРЕНЕВИЙ OG (app/opengraph-image.tsx, 200). Colocated [locale]-OG
      // дають 307→404 через next-intl as-needed → явно вказуємо рут, щоб соцкартка
      // мала зображення. (Per-city custom OG — окрема задача.)
      images: [`${BASE}/opengraph-image`],
    },
    twitter: {
      card: "summary_large_image",
      title: t("ogTitle"),
      description: t("ogDescription"),
      images: [`${BASE}/opengraph-image`],
    },
  };
}

export default async function LocaleLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: { locale: string };
}) {
  const { locale } = params;
  if (!(routing.locales as readonly string[]).includes(locale)) notFound();
  setRequestLocale(locale);
  const messages = await getMessages();
  const t = await getTranslations({ locale: locale as AppLocale, namespace: "meta" });
  // Валюта/ціна офера = позиційні (UAH для uk, EUR для решти) — синхрон з
  // create/keychains layout (раніше тут був хардкод UAH для всіх локалей).
  const isUA = locale === "uk";
  const al = locale as AppLocale;

  const jsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        // Org + LocalBusiness/Store у ОДНОМУ вузлі (multi-type) → краще для
        // локальної комерції: contacts + offers + geo + ціновий діапазон.
        "@type": ["Organization", "Store", "OnlineStore"],
        "@id": `${BASE}/#org`,
        name: "Monadruk",
        legalName: BUSINESS.ownerFull,
        url: BASE,
        logo: `${BASE}/icon`,
        image: `${BASE}/opengraph-image`,
        description: t("orgDescription"),
        email: BUSINESS.email,
        telephone: BUSINESS.phone,
        vatID: BUSINESS.taxId,
        currenciesAccepted: "UAH, EUR",
        paymentAccepted: "Visa, Mastercard, Apple Pay, Google Pay",
        priceRange: "₴₴",
        address: {
          "@type": "PostalAddress",
          addressCountry: "UA",
          addressRegion: "Хмельницька область",
          addressLocality: "Хмельницький",
          streetAddress: "вул. Завадського, 38",
        },
        // Координати магазину (Хмельницький, вул. Завадського, 38) — для
        // локального пошуку/Maps rich-results.
        geo: { "@type": "GeoCoordinates", latitude: 49.4229, longitude: 26.9871 },
        areaServed: [
          { "@type": "Country", name: "Ukraine" },
          { "@type": "AdministrativeArea", name: "European Union" },
        ],
        contactPoint: {
          "@type": "ContactPoint",
          contactType: "customer support",
          email: BUSINESS.email,
          telephone: BUSINESS.phone,
          areaServed: ["UA", "EU"],
          availableLanguage: ["uk", "en", "pl", "de"],
        },
        // Каталог пропозицій магазину — окремий продукт-офер на кожен товар.
        makesOffer: [
          {
            "@type": "Offer",
            name: t("offerKeychain"),
            priceCurrency: isUA ? "UAH" : "EUR",
            price: isUA ? "120" : "3",
            priceValidUntil: priceValidUntil(),
            url: `${BASE}/keychains`,
            availability: "https://schema.org/InStock",
          },
          {
            "@type": "Offer",
            name: t("offerMap"),
            priceCurrency: mapPriceRange(al).currency,
            price: mapPriceRange(al).low,
            priceValidUntil: priceValidUntil(),
            url: `${BASE}/create`,
            availability: "https://schema.org/InStock",
          },
        ],
        // sameAs = «це справжній бізнес із живими профілями» (E-E-A-T сигнал).
        // Підтверджені живі профілі; TikTok додати, щойно власник підтвердить хендл.
        sameAs: [
          "https://t.me/monadruk",
          "https://www.instagram.com/monadruk/",
          "https://www.youtube.com/@monadruk",
        ],
      },
      {
        "@type": "WebSite",
        "@id": `${BASE}/#website`,
        url: BASE,
        name: "Monadruk",
        inLanguage: localeMeta[locale as AppLocale].htmlLang,
        publisher: { "@id": `${BASE}/#org` },
      },
      {
        "@type": "Service",
        "@id": `${BASE}/#service`,
        name: t("serviceName"),
        serviceType: t("serviceType"),
        provider: { "@id": `${BASE}/#org` },
        description: t("serviceDescription"),
        // Офери НЕ дублюємо тут — вони живуть на вузлі Organization.makesOffer
        // (Service.provider → #org). Дубль давав краулеру 2 конфліктні джерела.
      },
    ],
  };

  return (
    <html lang={localeMeta[locale as AppLocale].htmlLang} className={`${serif.variable} ${sans.variable} ${mono.variable}`}>
      <head>
        {/* Раннє резолвлення DNS для зовнішніх origin-ів карти (OSM-тайли +
            cdnjs leaflet-маркери) — прискорює перший рендер карти на /create та
            /keychains. Дешеві hint-и, без відкриття зайвих зʼєднань. */}
        <link rel="dns-prefetch" href="https://tile.openstreetmap.org" />
        <link rel="dns-prefetch" href="https://cdnjs.cloudflare.com" />
        <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      </head>
      <body className="antialiased">
        {/* Skip-to-content (WCAG 2.4.1): перший фокусований елемент, невидимий
            доки не сфокусований; веде на <main id="main-content"> сторінки. */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[100] focus:rounded-lg focus:bg-forest focus:px-4 focus:py-2 focus:text-white focus:shadow-lift"
        >
          {t("skipToContent")}
        </a>
        <NextIntlClientProvider messages={messages}>
          <AuthProvider>
            {children}
            {/* Global footer (legal links + ФОП requisites + contacts + way home)
                on every content page. Suppressed on "/" (renders its own) and on
                the full-screen builders /create, /keychains, /capture. */}
            <GlobalFooter />
            <ToastHost />
            <ContactWidget />
            <SiteAnalytics />
          </AuthProvider>
        </NextIntlClientProvider>
      </body>
    </html>
  );
}
