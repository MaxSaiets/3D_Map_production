import type { Metadata } from "next";
import { getTranslations } from "next-intl/server";
import {
  BASE, localeUrl, MERCHANT_RETURN_POLICY_LD, pageMetadata, priceValidFrom, priceValidUntil,
} from "@/i18n/metadata";
import { defaultLocale, routing, type AppLocale } from "@/i18n/routing";

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  return pageMetadata({ locale: params.locale, path: "/maket", ns: "maketMeta" });
}

export default async function MaketLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: { locale: string };
}) {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  const t = await getTranslations({ locale, namespace: "maketMeta" });
  const isUA = locale === "uk";

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Product",
        name: t("productName"),
        description: t("description"),
        image: `${BASE}/real/real-1.webp`,
        brand: { "@type": "Brand", name: "Monadruk" },
        sku: "MND-FLOORPLAN",
        offers: {
          "@type": "Offer",
          priceCurrency: isUA ? "UAH" : "EUR",
          price: isUA ? "890" : "22",
          priceValidUntil: priceValidUntil(),
          validFrom: priceValidFrom(),
          availability: "https://schema.org/InStock",
          url: localeUrl(locale, "/maket"),
          hasMerchantReturnPolicy: MERCHANT_RETURN_POLICY_LD,
        },
      },
      {
        "@type": "HowTo",
        name: t("howToName"),
        step: [1, 2, 3].map((index) => ({
          "@type": "HowToStep",
          position: index,
          name: t(`step${index}`),
        })),
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: t("productName"), item: localeUrl(locale, "/maket") },
        ],
      },
    ],
  };

  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      {children}
      <section className="mx-auto max-w-[820px] px-5 py-10">
        <h2 className="text-[18px] font-semibold text-[var(--text-primary,#1c2320)]">{t("seoH2")}</h2>
        <p className="mt-3 text-[14px] leading-relaxed text-[var(--text-secondary,#5a655a)]">{t("seoP1")}</p>
        <p className="mt-2 text-[14px] leading-relaxed text-[var(--text-secondary,#5a655a)]">{t("seoP2")}</p>
      </section>
    </>
  );
}
