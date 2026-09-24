import type { Metadata } from "next";
import { getTranslations } from "next-intl/server";
import { pageMetadata, BASE, localeUrl } from "@/i18n/metadata";
import { routing, defaultLocale, type AppLocale } from "@/i18n/routing";

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  return pageMetadata({ locale: params.locale, path: "/mountains", ns: "mountainsMeta" });
}

/** /mountains — реальні гори світу (swissALTI3D / Copernicus) як друкована модель з ободком, боками, фігурками. */
export default async function MountainsLayout({ children, params }: { children: React.ReactNode; params: { locale: string } }) {
  const locale = ((routing.locales as readonly string[]).includes(params.locale) ? params.locale : defaultLocale) as AppLocale;
  const t = await getTranslations({ locale, namespace: "mountainsMeta" });
  const nav = await getTranslations({ locale, namespace: "nav" });
  const tm = await getTranslations({ locale, namespace: "mountains" });
  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      { "@type": "WebApplication", name: t("title"), description: t("description"), applicationCategory: "DesignApplication", operatingSystem: "Web",
        url: localeUrl(locale, "/mountains"), image: `${BASE}/mountains/presets/matterhorn.jpg`, offers: { "@type": "Offer", price: "0", priceCurrency: "UAH" } },
      { "@type": "BreadcrumbList", itemListElement: [
        { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
        { "@type": "ListItem", position: 2, name: nav("mountains"), item: localeUrl(locale, "/mountains") } ] },
    ],
  };
  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      {/* H1 у серверному HTML: сама студія (MountainStudio) — ssr:false. */}
      <header className="mx-auto max-w-[1180px] px-4 pt-8 text-center sm:pt-12">
        <span className="inline-block rounded-full border border-[var(--surface-border)] bg-[var(--surface-panel)] px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--text-secondary)]">{tm("badge")}</span>
        <h1 className="mt-4 text-3xl font-semibold text-[var(--text-primary)] sm:text-4xl">{tm("title")}</h1>
        <p className="mx-auto mt-3 max-w-2xl text-[var(--text-secondary)]">{tm("subtitle")}</p>
      </header>
      {children}
      <section className="mx-auto max-w-[820px] px-5 py-10">
        <h2 className="text-[18px] font-semibold text-[var(--text-primary,#1c2320)]">{t("proseH2")}</h2>
        <p className="mt-3 text-[14px] leading-relaxed text-[var(--text-secondary,#5a655a)]">{t("proseP1")}</p>
        <p className="mt-2 text-[14px] leading-relaxed text-[var(--text-secondary,#5a655a)]">{t("proseP2")}</p>
      </section>
    </>
  );
}
