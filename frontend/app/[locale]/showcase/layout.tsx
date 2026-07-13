import type { Metadata } from "next";
import { getTranslations } from "next-intl/server";
import { pageMetadata, localeUrl } from "@/i18n/metadata";
import { routing, defaultLocale, type AppLocale } from "@/i18n/routing";
import { proseFaq } from "@/lib/seoProse";

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  return pageMetadata({ locale: params.locale, path: "/showcase", ns: "showcaseMeta" });
}

export default async function ShowcaseLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: { locale: string };
}) {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  const t = await getTranslations({ locale, namespace: "showcaseMeta" });
  const nav = await getTranslations({ locale, namespace: "nav" });
  const faq = proseFaq("showcase", locale);
  const isUA = locale === "uk";

  // CollectionPage (галерея) + BreadcrumbList для rich results.
  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "CollectionPage",
        name: t("title"),
        description: t("description"),
        url: localeUrl(locale, "/showcase"),
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: nav("gallery"), item: localeUrl(locale, "/showcase") },
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
      {/* SEO FAQ ПІД галереєю (client-компонент, майже без індексованого тексту). */}
      <section className="mx-auto max-w-[820px] px-5 py-10">
        <h2 className="text-[18px] font-semibold text-[var(--text-primary,#1c2320)]">
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
