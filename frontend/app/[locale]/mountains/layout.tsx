import type { Metadata } from "next";
import { getTranslations } from "next-intl/server";
import { pageMetadata, BASE, localeUrl } from "@/i18n/metadata";
import { routing, defaultLocale, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { PEAKS, type PeakLocale } from "@/lib/mountainPages";
import { PEAK_UI } from "@/lib/mountainPagesUi";
import { MOUNTAINS_EXTRA } from "@/lib/mountainsExtra";

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  return pageMetadata({ locale: params.locale, path: "/mountains", ns: "mountainsMeta" });
}

/** /mountains — реальні гори світу (swissALTI3D / Copernicus) як друкована модель з ободком, боками, фігурками. */
export default async function MountainsLayout({ children, params }: { children: React.ReactNode; params: { locale: string } }) {
  const locale = ((routing.locales as readonly string[]).includes(params.locale) ? params.locale : defaultLocale) as AppLocale;
  const t = await getTranslations({ locale, namespace: "mountainsMeta" });
  const nav = await getTranslations({ locale, namespace: "nav" });
  const tm = await getTranslations({ locale, namespace: "mountains" });
  const ex = MOUNTAINS_EXTRA[locale];
  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      { "@type": "WebApplication", name: t("title"), description: t("description"), applicationCategory: "DesignApplication", operatingSystem: "Web",
        url: localeUrl(locale, "/mountains"), image: `${BASE}/mountains/presets/matterhorn.jpg`, offers: { "@type": "Offer", price: "0", priceCurrency: "UAH" } },
      { "@type": "BreadcrumbList", itemListElement: [
        { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
        { "@type": "ListItem", position: 2, name: nav("mountains"), item: localeUrl(locale, "/mountains") } ] },
      { "@type": "FAQPage", mainEntity: ex.faq.map((f) => ({ "@type": "Question", name: f.q, acceptedAnswer: { "@type": "Answer", text: f.a } })) },
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
        <h2 className="mt-8 text-[18px] font-semibold text-[var(--text-primary,#1c2320)]">{ex.h2who}</h2>
        <div className="mt-3 grid gap-3 sm:grid-cols-3">
          {ex.who.map((w) => (
            <div key={w.h3} className="rounded-[16px] border border-[var(--surface-border)] bg-white/70 px-4 py-3">
              <h3 className="text-[14.5px] font-semibold text-[var(--text-primary,#1c2320)]">{w.h3}</h3>
              <p className="mt-1 text-[13.5px] leading-relaxed text-[var(--text-secondary,#5a655a)]">{w.p}</p>
            </div>
          ))}
        </div>
        <h2 className="mt-8 text-[18px] font-semibold text-[var(--text-primary,#1c2320)]">{ex.h2how}</h2>
        <ol className="mt-3 flex list-decimal flex-col gap-1.5 pl-5 text-[14px] leading-relaxed text-[var(--text-secondary,#5a655a)]">
          {ex.how.map((h) => <li key={h}>{h}</li>)}
        </ol>
        <h2 className="mt-8 text-[18px] font-semibold text-[var(--text-primary,#1c2320)]">{ex.h2faq}</h2>
        <dl className="mt-3 flex flex-col gap-3">
          {ex.faq.map((f) => (
            <div key={f.q}>
              <dt className="text-[14.5px] font-semibold text-[var(--text-primary,#1c2320)]">{f.q}</dt>
              <dd className="mt-1 text-[14px] leading-relaxed text-[var(--text-secondary,#5a655a)]">{f.a}</dd>
            </div>
          ))}
        </dl>
        {/* Сторінки вершин (/gory/[slug]) — перелінковка для пошуку. */}
        <h2 className="mt-8 text-[18px] font-semibold text-[var(--text-primary,#1c2320)]">
          <Link href="/gory" className="hover:underline">{PEAK_UI[locale as PeakLocale].crumb}</Link>
        </h2>
        <ul className="mt-3 flex flex-wrap gap-2">
          {PEAKS.map((p) => (
            <li key={p.slug}>
              <Link href={`/gory/${p.slug}`} className="inline-block rounded-full border border-[var(--surface-border)] bg-white/70 px-3.5 py-1.5 text-[13px] font-medium text-[var(--text-secondary)] transition hover:text-[var(--text-primary)]">
                {p.t[locale as PeakLocale].name}
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </>
  );
}
