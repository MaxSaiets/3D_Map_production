import type { Metadata } from "next";
import { getTranslations } from "next-intl/server";
import { pageMetadata, localeUrl } from "@/i18n/metadata";
import { ProSubscription } from "@/components/ProSubscription";
import { SUB_PRICE } from "@/lib/legal/subscription";
import { FILE_PRICE_UAH, breakEvenFiles } from "@/lib/mapPrices";
import { BUSINESS } from "@/lib/legal";

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  return pageMetadata({ locale: params.locale, path: "/pro", ns: "proMeta" });
}

/** 07.10.2026: структуровані дані для пошуку — підписка як Product з двома Offer (UAH/USD,
 *  щомісячно) + FAQPage з тих самих текстів, що й видимий FAQ на сторінці. */
export default async function ProPage({ params }: { params: { locale: string } }) {
  const locale = params.locale;
  const t = await getTranslations({ locale, namespace: "proSub" });
  const tm = await getTranslations({ locale, namespace: "proMeta" });
  const price = `${new Intl.NumberFormat(locale === "uk" ? "uk-UA" : locale).format(SUB_PRICE.UAH)} ₴`;
  const vals = { breakEven: breakEvenFiles(SUB_PRICE.UAH, FILE_PRICE_UAH), price, filePrice: `${FILE_PRICE_UAH} ₴` };
  const url = localeUrl(locale as never, "/pro");
  const offer = (amount: number, currency: string) => ({
    "@type": "Offer",
    price: String(amount),
    priceCurrency: currency,
    availability: "https://schema.org/InStock",
    url,
    seller: { "@type": "Organization", name: BUSINESS.storeName },
    priceSpecification: {
      "@type": "UnitPriceSpecification",
      price: String(amount),
      priceCurrency: currency,
      billingDuration: "P1M",
      unitCode: "MON",
    },
  });
  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Product",
        name: "Monadruk Pro",
        description: tm("description"),
        url,
        brand: { "@type": "Brand", name: "Monadruk" },
        offers: [offer(SUB_PRICE.UAH, "UAH"), offer(SUB_PRICE.USD, "USD")],
      },
      {
        "@type": "FAQPage",
        mainEntity: [1, 2, 3, 4, 5, 6].map((i) => ({
          "@type": "Question",
          name: t(`faq${i}q`),
          acceptedAnswer: { "@type": "Answer", text: t(`faq${i}a`, vals) },
        })),
      },
    ],
  };
  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      <ProSubscription />
    </>
  );
}
