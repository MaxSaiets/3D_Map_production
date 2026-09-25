import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { setRequestLocale } from "next-intl/server";
import { BASE, localeUrl } from "@/i18n/metadata";
import { routing, defaultLocale, localeMeta, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { PEAKS, PEAK_BY_SLUG, PEAK_LOCALES, relatedPeaks, type PeakLocale } from "@/lib/mountainPages";
import { PEAK_UI } from "@/lib/mountainPagesUi";

/**
 * Окрема сторінка на кожну вершину режиму «Гори» (/gory/[slug]) — 25.09.2026.
 * Унікальний текст шістьма мовами (lib/mountainPages.ts), фото пресету з бекенду,
 * CTA → /mountains?peak=<presetId> (MountainStudio одразу обирає вершину).
 */
export function generateStaticParams() {
  return PEAKS.map((p) => ({ slug: p.slug }));
}

export const dynamicParams = false;

const resolve = (raw: string): AppLocale =>
  ((routing.locales as readonly string[]).includes(raw) ? raw : defaultLocale) as AppLocale;

const fmt = (n: number) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " ");

export async function generateMetadata({ params }: { params: { locale: string; slug: string } }): Promise<Metadata> {
  const p = PEAK_BY_SLUG[params.slug];
  if (!p) return {};
  const locale = resolve(params.locale);
  const l = locale as PeakLocale;
  const tx = p.t[l];
  const path = `/gory/${p.slug}`;
  const full = `${tx.p[0]} ${tx.p[1]}`;
  const description = full.length <= 158 ? full : `${full.slice(0, 155).replace(/\s+\S*$/, "")}…`;
  const img = `${BASE}${p.photo}`;
  return {
    title: { absolute: `${tx.title} | Monadruk` },
    description,
    alternates: {
      canonical: localeUrl(locale, path),
      languages: {
        ...Object.fromEntries(PEAK_LOCALES.map((x) => [localeMeta[x].htmlLang, localeUrl(x, path)])),
        "x-default": localeUrl("uk", path),
      },
    },
    openGraph: { title: tx.title, description, url: localeUrl(locale, path), siteName: "Monadruk", type: "article", images: [{ url: img, alt: tx.name }] },
    twitter: { card: "summary_large_image", title: tx.title, description, images: [img] },
  };
}

export default function PeakPage({ params }: { params: { locale: string; slug: string } }) {
  const p = PEAK_BY_SLUG[params.slug];
  if (!p) notFound();
  const locale = resolve(params.locale);
  setRequestLocale(locale);
  const l = locale as PeakLocale;
  const tx = p.t[l];
  const u = PEAK_UI[l];
  const path = `/gory/${p.slug}`;
  const studio = `/mountains?peak=${p.presetId}`;
  const related = relatedPeaks(p.slug, 6);

  const facts: [string, string][] = [
    [u.facts.elev, `${fmt(p.elev)} ${u.m}`],
    [u.facts.where, tx.country],
    [u.facts.area, `≈${String(p.areaKm).replace(".", l === "en" ? "." : ",")}×${String(p.areaKm).replace(".", l === "en" ? "." : ",")} ${u.km}`],
    [u.facts.size, u.sizeVal],
    [u.facts.file, u.fileVal],
    [u.facts.print, u.printVal],
  ];

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebPage",
        name: tx.title,
        url: localeUrl(locale, path),
        inLanguage: l,
        about: {
          "@type": "Mountain",
          name: tx.name,
          geo: { "@type": "GeoCoordinates", latitude: p.lat, longitude: p.lon, elevation: p.elev },
        },
        primaryImageOfPage: { "@type": "ImageObject", contentUrl: `${BASE}${p.photo}`, caption: tx.name },
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: u.crumb, item: localeUrl(locale, "/gory") },
          { "@type": "ListItem", position: 3, name: tx.name, item: localeUrl(locale, path) },
        ],
      },
    ],
  };

  return (
    <main id="main-content" tabIndex={-1} className="mx-auto max-w-[980px] px-5 py-12 lg:py-16">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      <nav className="text-[13px] text-ink-3" aria-label="breadcrumb">
        <Link href="/" className="hover:underline">Monadruk</Link>
        {" / "}
        <Link href="/gory" className="hover:underline">{u.crumb}</Link>
        {" / "}
        <span className="text-ink">{tx.name}</span>
      </nav>

      <h1 className="mt-4 text-[clamp(26px,3.6vw,40px)] leading-tight">{tx.title}</h1>

      <div className="mt-6 grid gap-8 lg:grid-cols-[1.1fr_1fr] lg:items-start">
        <figure className="overflow-hidden rounded-[22px] border border-line bg-paper">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={p.photo} alt={`${tx.name} — ${tx.country}`} className="aspect-[4/3] h-auto w-full object-cover" fetchPriority="high" />
          <figcaption className="px-4 py-3 text-[13px] text-ink-3">{u.photoCaption}: {tx.name}, {fmt(p.elev)} {u.m}</figcaption>
        </figure>

        <div>
          <p className="text-[15.5px] leading-relaxed text-ink-2">{tx.p[0]}</p>
          <div className="mt-5 flex flex-wrap gap-3">
            <Link
              href={studio}
              className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-6 py-3 text-sm font-semibold text-white transition hover:opacity-90"
            >
              {u.cta(tx.name)} →
            </Link>
          </div>
          <section className="mt-7 rounded-[18px] border border-line-soft bg-white/60 px-5 py-4">
            <dl className="text-[14px]">
              {facts.map(([k, v]) => (
                <div key={k} className="flex items-baseline justify-between gap-3 border-b border-line-soft/50 py-1.5 last:border-0">
                  <dt className="text-ink-3">{k}</dt>
                  <dd className="text-right font-semibold text-ink">{v}</dd>
                </div>
              ))}
            </dl>
          </section>
        </div>
      </div>

      <section className="mt-10 grid gap-6 lg:grid-cols-2">
        <p className="text-[15px] leading-relaxed text-ink-2">{tx.p[1]}</p>
        <p className="text-[15px] leading-relaxed text-ink-2">{tx.p[2]}</p>
      </section>

      <section className="mt-10 grid gap-6 lg:grid-cols-2">
        <div className="rounded-[18px] border border-line-soft bg-white/60 px-5 py-5">
          <h2 className="text-[18px] font-semibold">{u.howTitle}</h2>
          <ol className="mt-3 list-decimal space-y-1.5 pl-5 text-[14.5px] leading-relaxed text-ink-2">
            {u.how.map((s) => <li key={s}>{s}</li>)}
          </ol>
        </div>
        <div className="rounded-[18px] border border-line-soft bg-white/60 px-5 py-5">
          <h2 className="text-[18px] font-semibold">{u.optsTitle}</h2>
          <ul className="mt-3 list-disc space-y-1.5 pl-5 text-[14.5px] leading-relaxed text-ink-2">
            {u.opts.map((s) => <li key={s}>{s}</li>)}
          </ul>
        </div>
      </section>
      <p className="mt-5 text-[13px] leading-relaxed text-ink-3">{u.dataNote}</p>

      <section className="mt-12">
        <h2 className="text-[20px] font-semibold">{u.others}</h2>
        <ul className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3">
          {related.map((r) => (
            <li key={r.slug}>
              <Link href={`/gory/${r.slug}`} className="group block overflow-hidden rounded-[18px] border border-line bg-paper">
                <div className="aspect-[4/3] overflow-hidden">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={r.photo} alt={r.t[l].name} loading="lazy" className="h-full w-full object-cover transition duration-500 group-hover:scale-[1.05]" />
                </div>
                <span className="block px-3 py-2.5 text-[13.5px] font-medium leading-snug text-ink">
                  {r.t[l].name} · {fmt(r.elev)} {u.m}
                </span>
              </Link>
            </li>
          ))}
        </ul>
        <div className="mt-5 flex flex-wrap gap-4 text-[14.5px] font-semibold">
          <Link href="/gory" className="text-[var(--accent-strong)] hover:underline">{u.crumb} →</Link>
          <Link href="/mountains" className="text-[var(--accent-strong)] hover:underline">{u.ctaAny} →</Link>
        </div>
      </section>
    </main>
  );
}
