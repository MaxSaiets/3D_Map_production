import type { Metadata } from "next";
import { getTranslations } from "next-intl/server";
import { pageMetadata, BASE, localeUrl, priceValidUntil } from "@/i18n/metadata";
import { routing, defaultLocale, type AppLocale } from "@/i18n/routing";

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  // ogImage: false — маршрут має власний colocated opengraph-image.tsx.
  return pageMetadata({ locale: params.locale, path: "/keychains", ns: "keychainsMeta", ogImage: false });
}

export default async function KeychainsLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: { locale: string };
}) {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  const t = await getTranslations({ locale, namespace: "keychainsMeta" });
  const tm = await getTranslations({ locale, namespace: "meta" });
  const nav = await getTranslations({ locale, namespace: "nav" });
  const isUA = locale === "uk";

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Product",
        name: tm("offerKeychain"),
        description: t("description"),
        image: `${BASE}/showcase/keychain-5.webp`,
        brand: { "@type": "Brand", name: "Monadruk" },
        sku: "MND-KEYCHAIN",
        offers: {
          "@type": "Offer",
          priceCurrency: isUA ? "UAH" : "EUR",
          price: isUA ? "120" : "3",
          priceValidUntil: priceValidUntil(),
          availability: "https://schema.org/InStock",
          url: localeUrl(locale, "/keychains"),
        },
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: nav("keychains"), item: localeUrl(locale, "/keychains") },
        ],
      },
    ],
  };

  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      {children}
    </>
  );
}
