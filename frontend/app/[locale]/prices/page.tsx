import type { Metadata } from "next";
import { Link } from "@/i18n/navigation";
import { localeUrl, BASE } from "@/i18n/metadata";
import { locales, localeMeta, defaultLocale } from "@/i18n/routing";
import { ShieldCheck, ArrowRight } from "lucide-react";
import { getCatalog, formatCatalogPrice } from "@/lib/catalog";
import { FILE_PRICE_UAH, breakEvenFiles } from "@/lib/mapPrices";
import { SUB_PRICE } from "@/lib/legal/subscription";
import { BUSINESS, IBAN_DISPLAY } from "@/lib/legal";

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  const c = getCatalog(params.locale);
  const languages: Record<string, string> = {};
  for (const l of locales) languages[localeMeta[l].htmlLang] = localeUrl(l, "/prices");
  languages["x-default"] = localeUrl(defaultLocale, "/prices");
  const url = localeUrl(params.locale as never, "/prices");
  return {
    title: c.metaTitle,
    description: c.metaDescription,
    alternates: { canonical: url, languages },
    openGraph: { title: c.metaTitle, description: c.metaDescription, url, siteName: "Monadruk", type: "website", images: [`${BASE}/opengraph-image`] },
    twitter: { card: "summary_large_image", title: c.metaTitle, description: c.metaDescription, images: [`${BASE}/opengraph-image`] },
  };
}

export default function PricesPage({ params }: { params: { locale: string } }) {
  const locale = params.locale;
  const c = getCatalog(locale);
  // З якого файлу на місяць підписка дешевша за поштучну купівлю (2100/149 → 15).
  const proBreakEven = breakEvenFiles(SUB_PRICE.UAH, FILE_PRICE_UAH);

  // Structured data: каталог пропозицій з реальними цінами (UAH) для пошуковиків
  // + BreadcrumbList (консистентність з рештою контент-сторінок).
  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "OfferCatalog",
        name: c.metaTitle,
        url: localeUrl(locale as never, "/prices"),
        itemListElement: c.categories.flatMap((cat) =>
          cat.items
            .filter((it) => it.uah > 0)
            .map((it) => ({
              "@type": "Offer",
              name: it.name,
              description: it.desc,
              price: String(it.uah),
              priceCurrency: "UAH",
              availability: "https://schema.org/InStock",
              seller: { "@type": "Organization", name: BUSINESS.storeName },
            })),
        ),
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale as never, "/") },
          { "@type": "ListItem", position: 2, name: c.h1, item: localeUrl(locale as never, "/prices") },
        ],
      },
      {
        "@type": "FAQPage",
        mainEntity: c.faq.map((f) => ({
          "@type": "Question",
          name: f.q,
          acceptedAnswer: { "@type": "Answer", text: f.a },
        })),
      },
    ],
  };

  return (
    <div id="main-content" tabIndex={-1} className="mx-auto max-w-[860px] px-5 py-12 lg:px-8">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      <Link href="/" className="text-[13px] font-semibold text-ink-2 hover:text-ink">← monadruk</Link>

      <h1 className="mt-4 font-serif text-3xl font-semibold text-ink lg:text-4xl">{c.metaTitle}</h1>
      <p className="mt-3 max-w-[620px] text-[15px] leading-relaxed text-ink-2">{c.intro}</p>

      {c.categories.map((cat) => (
        <section key={cat.title} className="mt-9">
          <h2 className="mb-3 text-xs font-bold uppercase tracking-[0.16em] text-ink-3">{cat.title}</h2>
          <ul className="overflow-hidden rounded-2xl border border-line">
            {cat.items.map((it) => (
              <li
                key={it.name}
                className="flex items-start justify-between gap-4 border-b border-line-soft/70 px-4 py-3.5 last:border-b-0"
              >
                <div className="min-w-0">
                  <div className="text-[15px] font-semibold text-ink">
                    {it.href ? (
                      <Link href={it.href} className="underline-offset-2 hover:underline">{it.name}</Link>
                    ) : it.name}
                  </div>
                  <div className="mt-0.5 text-[13px] leading-snug text-ink-3">{it.desc}</div>
                </div>
                <div className="whitespace-nowrap pt-0.5 text-[15px] font-bold text-forest" style={{ color: "var(--forest, #2E4A3A)" }}>
                  {formatCatalogPrice(it.uah, it.kind, locale, it.usd)}
                </div>
              </li>
            ))}
          </ul>
        </section>
      ))}

      {/* 07.10.2026: Pro — для тих, хто друкує багато або на продаж. */}
      <section className="mt-9 rounded-[22px] p-6 text-[#F4EFE4] sm:p-7" style={{ background: "var(--forest, #2E4A3A)" }} data-testid="prices-pro">
        <div className="text-[11px] font-bold uppercase tracking-[0.16em] opacity-80">{c.pro.eyebrow}</div>
        <h2 className="mt-2 flex items-center gap-2 font-serif text-2xl">
          <ShieldCheck size={22} className="shrink-0" /> {c.pro.title}
        </h2>
        <p className="mt-2 max-w-[600px] text-[14.5px] leading-relaxed opacity-90">{c.pro.text.replace("{n}", String(proBreakEven))}</p>
        <Link
          href="/pro"
          className="mt-4 inline-flex min-h-[44px] items-center gap-2 rounded-full bg-[#F4EFE4] px-5 text-sm font-bold text-[#1E2A22] transition hover:opacity-90"
        >
          {c.pro.cta} <ArrowRight size={15} />
        </Link>
      </section>

      {/* Умови (валюта, доставка, оплата) — вимога активації LiqPay */}
      <section className="mt-10 rounded-2xl border border-line bg-bg-2/40 px-5 py-5">
        <h2 className="mb-2 text-sm font-bold text-ink">{c.notesTitle}</h2>
        <ul className="list-disc space-y-1.5 pl-5 text-[13.5px] leading-relaxed text-ink-2">
          {c.notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      </section>

      {/* FAQ (видимий, +FAQPage JSON-LD вище). */}
      <section className="mt-10">
        <h2 className="mb-3 text-xs font-bold uppercase tracking-[0.16em] text-ink-3">{c.faqTitle}</h2>
        <dl className="flex flex-col gap-4">
          {c.faq.map((f) => (
            <div key={f.q}>
              <dt className="text-[14.5px] font-semibold text-ink">{f.q}</dt>
              <dd className="mt-1 text-[13.5px] leading-relaxed text-ink-2">{f.a}</dd>
            </div>
          ))}
        </dl>
      </section>

      {/* Реквізити продавця (ФОП) — вимога платіжних систем */}
      <section className="mt-6 text-[13px] leading-relaxed text-ink-3">
        <h2 className="mb-1.5 text-sm font-bold text-ink">{c.sellerTitle}</h2>
        <p>{BUSINESS.ownerFull}</p>
        <p>РНОКПП {BUSINESS.taxId} · IBAN {IBAN_DISPLAY}</p>
        <p>{BUSINESS.storeAddress}</p>
        <p>
          <a className="hover:text-ink" href={`mailto:${BUSINESS.email}`}>{BUSINESS.email}</a>
          {" · "}
          <a className="hover:text-ink" href={`tel:${BUSINESS.phone}`}>{BUSINESS.phoneDisplay}</a>
        </p>
      </section>

      {/* Документи */}
      <section className="mt-6 text-[13.5px] text-ink-2">
        <p className="mb-2">{c.docsIntro}</p>
        <div className="flex flex-wrap gap-x-4 gap-y-1.5 font-semibold text-forest" style={{ color: "var(--forest, #2E4A3A)" }}>
          <Link href="/offer" className="hover:underline">{c.docs.offer}</Link>
          <Link href="/delivery" className="hover:underline">{c.docs.delivery}</Link>
          <Link href="/refund" className="hover:underline">{c.docs.refund}</Link>
          <Link href="/contacts" className="hover:underline">{c.docs.contacts}</Link>
        </div>
      </section>

      <div className="mt-9">
        <Link
          href="/create"
          className="inline-flex min-h-[48px] items-center gap-2 rounded-full bg-forest px-6 py-3 text-sm font-bold text-[#F4EFE4] shadow-[0_10px_24px_rgba(46,74,58,0.28)] transition hover:opacity-90"
          style={{ background: "var(--forest, #2E4A3A)" }}
        >
          {c.ctaLabel}
        </Link>
      </div>
    </div>
  );
}
