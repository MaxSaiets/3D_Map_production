import type { Metadata } from "next";
import { getTranslations } from "next-intl/server";
import { pageMetadata, localeUrl } from "@/i18n/metadata";
import { routing, defaultLocale, type AppLocale } from "@/i18n/routing";
import { proseFaq } from "@/lib/seoProse";
import { Link } from "@/i18n/navigation";
import { GALLERY_ITEMS } from "@/lib/gallery";

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
        hasPart: GALLERY_ITEMS.map((g) => ({
          "@type": "ImageObject",
          contentUrl: `https://monadruk.com${g.src}`,
          name: isUA ? g.title.uk : g.title.en,
          url: localeUrl(locale === "uk" ? "uk" : "en", `/foto/${g.slug}`),
        })),
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
      {/* Серверна сітка: кожне фото → власна сторінка /foto/[slug] (краулабельні
          лінки + унікальні alt; client-галерея вище відкриває лише модалку). */}
      <section className="mx-auto max-w-[1280px] px-5 pb-4 pt-10 lg:px-8">
        <h2 className="text-center font-serif text-[clamp(22px,2.6vw,32px)] text-ink">
          {isUA ? "Усі фото й моделі — з описом" : "All photos and models — with descriptions"}
        </h2>
        <ul className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
          {GALLERY_ITEMS.map((g) => (
            <li key={g.slug}>
              <Link href={`/foto/${g.slug}`} className="group block overflow-hidden rounded-[16px] border border-line bg-paper">
                <div className="aspect-square overflow-hidden">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={g.src} alt={isUA ? g.alt.uk : g.alt.en} loading="lazy" className="h-full w-full object-cover transition duration-500 group-hover:scale-[1.05]" />
                </div>
                <span className="block px-2.5 py-2 text-[12px] font-medium leading-snug text-ink-2">{isUA ? g.title.uk : g.title.en}</span>
              </Link>
            </li>
          ))}
        </ul>
      </section>
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
