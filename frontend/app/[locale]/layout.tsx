import type { Metadata } from "next";
import { Cormorant_Garamond, Manrope, JetBrains_Mono } from "next/font/google";
import { notFound } from "next/navigation";
import { NextIntlClientProvider } from "next-intl";
import { setRequestLocale, getMessages, getTranslations } from "next-intl/server";
import "../globals.css";
import { ContactWidget } from "@/components/ContactWidget";
import { AuthProvider } from "@/components/AuthProvider";
import SiteAnalytics from "@/components/SiteAnalytics";
import { routing, locales, localeMeta, type AppLocale } from "@/i18n/routing";
import { BUSINESS } from "@/lib/legal";

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
    verification: process.env.NEXT_PUBLIC_GOOGLE_SITE_VERIFICATION
      ? { google: process.env.NEXT_PUBLIC_GOOGLE_SITE_VERIFICATION }
      : undefined,
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
    },
    twitter: {
      card: "summary_large_image",
      title: t("ogTitle"),
      description: t("ogDescription"),
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

  const jsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Organization",
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
        address: {
          "@type": "PostalAddress",
          addressCountry: "UA",
          addressRegion: "Хмельницька область",
          addressLocality: "Хмельницький",
          streetAddress: "вул. Завадського, 38",
        },
        contactPoint: {
          "@type": "ContactPoint",
          contactType: "customer support",
          email: BUSINESS.email,
          telephone: BUSINESS.phone,
          areaServed: ["UA", "EU"],
          availableLanguage: ["uk", "en", "pl", "de"],
        },
        sameAs: ["https://t.me/monadruk"],
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
        offers: [
          { "@type": "Offer", name: t("offerKeychain"), priceCurrency: "UAH", price: "120", url: `${BASE}/keychains`, availability: "https://schema.org/InStock" },
          { "@type": "Offer", name: t("offerMap"), priceCurrency: "UAH", price: "250", url: `${BASE}/create`, availability: "https://schema.org/InStock" },
        ],
      },
    ],
  };

  return (
    <html lang={localeMeta[locale as AppLocale].htmlLang} className={`${serif.variable} ${sans.variable} ${mono.variable}`}>
      <head>
        <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />
      </head>
      <body className="antialiased">
        <NextIntlClientProvider messages={messages}>
          <AuthProvider>
            {children}
            <ContactWidget />
            <SiteAnalytics />
          </AuthProvider>
        </NextIntlClientProvider>
      </body>
    </html>
  );
}
