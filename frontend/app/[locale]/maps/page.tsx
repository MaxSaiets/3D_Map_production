import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { pageMetadata, localeUrl } from "@/i18n/metadata";
import { routing, defaultLocale, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { CITY_PAGES } from "@/lib/cityPages";

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  return pageMetadata({ locale: params.locale, path: "/maps", ns: "mapsMeta" });
}

export default async function MapsIndexPage({ params }: { params: { locale: string } }) {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  setRequestLocale(locale);
  const t = await getTranslations({ locale, namespace: "mapsMeta" });
  const tc = await getTranslations({ locale, namespace: "cityPages" });

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "CollectionPage",
        name: t("title"),
        description: t("description"),
        url: localeUrl(locale, "/maps"),
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: tc("breadcrumb"), item: localeUrl(locale, "/maps") },
        ],
      },
    ],
  };

  return (
    <main className="mx-auto max-w-[920px] px-5 py-14 lg:py-20">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      <h1 className="text-[clamp(28px,4vw,46px)] leading-tight">{t("title")}</h1>
      <p className="mt-4 max-w-[640px] text-[15px] leading-relaxed text-ink-2">{t("description")}</p>
      <ul className="mt-10 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {CITY_PAGES.map((c) => (
          <li key={c.slug}>
            <Link
              href={`/maps/${c.slug}`}
              className="block rounded-[18px] border border-line-soft bg-white/70 px-4 py-3.5 text-[15px] font-semibold text-ink transition hover:border-[var(--accent)]"
            >
              {c.names[locale]}
            </Link>
          </li>
        ))}
      </ul>
    </main>
  );
}
