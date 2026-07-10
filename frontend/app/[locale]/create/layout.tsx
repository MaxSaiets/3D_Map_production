import type { Metadata } from "next";
import { getTranslations } from "next-intl/server";
import { pageMetadata, BASE, localeUrl, priceValidUntil } from "@/i18n/metadata";
import { routing, defaultLocale, type AppLocale } from "@/i18n/routing";
import { mapPriceRange } from "@/lib/mapPrices";
import { seoProse } from "@/lib/seoProse";

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  // Colocated [locale]-OG дають 307→404 (next-intl as-needed) → беремо робочий рут-OG.
  return pageMetadata({ locale: params.locale, path: "/create", ns: "createMeta" });
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

  const prose = seoProse("create", locale);
  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      {children}
      {/* SEO-проза ПІД конструктором: client-builder майже не має індексованого
          тексту, хоча таргетить грошові запити. Серверний блок = краулер бачить
          контент; користувачу нижче згину не заважає. */}
      <section className="mx-auto max-w-[820px] px-5 py-10">
        <h2 className="text-[18px] font-semibold text-[var(--text-primary,#1c2320)]">{prose.h2}</h2>
        <p className="mt-3 text-[14px] leading-relaxed text-[var(--text-secondary,#5a655a)]">{prose.p1}</p>
        <p className="mt-2 text-[14px] leading-relaxed text-[var(--text-secondary,#5a655a)]">{prose.p2}</p>
      </section>
    </>
  );
}
