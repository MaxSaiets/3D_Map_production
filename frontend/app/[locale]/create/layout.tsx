import type { Metadata } from "next";
import { getTranslations } from "next-intl/server";
import { pageMetadata, BASE, localeUrl, priceValidUntil } from "@/i18n/metadata";
import { routing, defaultLocale, type AppLocale } from "@/i18n/routing";
import { mapPriceRange } from "@/lib/mapPrices";

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  // ogImage: false — маршрут має власний colocated opengraph-image.tsx.
  return pageMetadata({ locale: params.locale, path: "/create", ns: "createMeta", ogImage: false });
}

export default async function CreateLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: { locale: string };
}) {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  const t = await getTranslations({ locale, namespace: "createMeta" });
  const tm = await getTranslations({ locale, namespace: "meta" });
  const nav = await getTranslations({ locale, namespace: "nav" });

  // Product (3D-мапа міста) + BreadcrumbList — rich results у пошуку,
  // дзеркально до /keychains layout.
  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Product",
        name: tm("offerMap"),
        description: t("description"),
        image: `${BASE}/showcase/map-1.webp`,
        brand: { "@type": "Brand", name: "Monadruk" },
        sku: "MND-MAP",
        offers: {
          // Ціна-floor з ЄДИНОГО джерела (lib/mapPrices, синхрон з pricing.json +
          // city-сторінками) — раніше хардкод "250"/"6" не збігався з реальними 150/≈4.
          "@type": "Offer",
          priceCurrency: mapPriceRange(locale).currency,
          price: mapPriceRange(locale).low,
          priceValidUntil: priceValidUntil(),
          availability: "https://schema.org/InStock",
          url: localeUrl(locale, "/create"),
        },
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: nav("createMap"), item: localeUrl(locale, "/create") },
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
