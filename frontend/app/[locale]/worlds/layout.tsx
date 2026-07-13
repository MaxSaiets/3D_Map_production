import type { Metadata } from "next";
import { getTranslations } from "next-intl/server";
import { pageMetadata, BASE, localeUrl } from "@/i18n/metadata";
import { routing, defaultLocale, type AppLocale } from "@/i18n/routing";
import { seoProse, proseFaq } from "@/lib/seoProse";

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  return pageMetadata({ locale: params.locale, path: "/worlds", ns: "worldsMeta" });
}

export default async function WorldsLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: { locale: string };
}) {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  const t = await getTranslations({ locale, namespace: "worldsMeta" });
  const nav = await getTranslations({ locale, namespace: "nav" });
  const prose = seoProse("worlds", locale);
  const faq = proseFaq("worlds", locale);
  const isUA = locale === "uk";

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebApplication",
        name: t("title"),
        description: t("description"),
        applicationCategory: "DesignApplication",
        operatingSystem: "Web",
        url: localeUrl(locale, "/worlds"),
        image: `${BASE}/showcase/map-1.webp`,
        offers: { "@type": "Offer", price: "0", priceCurrency: "UAH" },
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: nav("worlds"), item: localeUrl(locale, "/worlds") },
        ],
      },
      {
        "@type": "FAQPage",
        mainEntity: faq.map((f) => ({
          "@type": "Question",
          name: f.q,
          acceptedAnswer: { "@type": "Answer", text: f.a },
        })),
      },
    ],
  };

  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      {children}
      {/* SEO-проза + FAQ ПІД інструментом (client-компонент майже без тексту). */}
      <section className="mx-auto max-w-[820px] px-5 py-10">
        <h2 className="text-[18px] font-semibold text-[var(--text-primary,#1c2320)]">{prose.h2}</h2>
        <p className="mt-3 text-[14px] leading-relaxed text-[var(--text-secondary,#5a655a)]">{prose.p1}</p>
        <p className="mt-2 text-[14px] leading-relaxed text-[var(--text-secondary,#5a655a)]">{prose.p2}</p>
        <h2 className="mt-8 text-[18px] font-semibold text-[var(--text-primary,#1c2320)]">
          {isUA ? "Часті запитання" : "FAQ"}
        </h2>
        <dl className="mt-3 flex flex-col gap-3">
          {faq.map((f) => (
            <div key={f.q}>
              <dt className="text-[14.5px] font-semibold text-[var(--text-primary,#1c2320)]">{f.q}</dt>
              <dd className="mt-1 text-[14px] leading-relaxed text-[var(--text-secondary,#5a655a)]">{f.a}</dd>
            </div>
          ))}
        </dl>
      </section>
    </>
  );
}
