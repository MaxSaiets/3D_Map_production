import type { Metadata } from "next";
import { getTranslations } from "next-intl/server";
import { pageMetadata, BASE, localeUrl, priceValidUntil } from "@/i18n/metadata";
import { routing, defaultLocale, type AppLocale } from "@/i18n/routing";
import { seoProse } from "@/lib/seoProse";

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  // Colocated [locale]-OG дають 307→404 (next-intl as-needed) → беремо робочий рут-OG.
  return pageMetadata({ locale: params.locale, path: "/keychains", ns: "keychainsMeta" });
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

  const prose = seoProse("keychains", locale);
  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      {children}
      {/* SEO-проза ПІД конструктором (див. create/layout — той самий патерн). */}
      <section className="mx-auto max-w-[820px] px-5 py-10">
        <h2 className="text-[18px] font-semibold text-[var(--text-primary,#1c2320)]">{prose.h2}</h2>
        <p className="mt-3 text-[14px] leading-relaxed text-[var(--text-secondary,#5a655a)]">{prose.p1}</p>
        <p className="mt-2 text-[14px] leading-relaxed text-[var(--text-secondary,#5a655a)]">{prose.p2}</p>
      </section>
    </>
  );
}
