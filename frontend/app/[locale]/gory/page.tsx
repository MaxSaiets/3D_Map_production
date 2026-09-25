import type { Metadata } from "next";
import { setRequestLocale } from "next-intl/server";
import { BASE, localeUrl } from "@/i18n/metadata";
import { routing, defaultLocale, localeMeta, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { PEAKS, PEAK_LOCALES, type PeakLocale } from "@/lib/mountainPages";
import { PEAK_UI } from "@/lib/mountainPagesUi";

/** /gory — каталог сторінок вершин режиму «Гори» (25.09.2026). */
const resolve = (raw: string): AppLocale =>
  ((routing.locales as readonly string[]).includes(raw) ? raw : defaultLocale) as AppLocale;

const fmt = (n: number) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " ");

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  const locale = resolve(params.locale);
  const u = PEAK_UI[locale as PeakLocale];
  return {
    title: { absolute: `${u.indexTitle} | Monadruk` },
    description: u.indexDesc,
    alternates: {
      canonical: localeUrl(locale, "/gory"),
      languages: {
        ...Object.fromEntries(PEAK_LOCALES.map((x) => [localeMeta[x].htmlLang, localeUrl(x, "/gory")])),
        "x-default": localeUrl("uk", "/gory"),
      },
    },
    openGraph: { title: u.indexTitle, description: u.indexDesc, url: localeUrl(locale, "/gory"), siteName: "Monadruk", images: [{ url: `${BASE}/mountains/presets/matterhorn.jpg` }] },
  };
}

export default function GoryIndex({ params }: { params: { locale: string } }) {
  const locale = resolve(params.locale);
  setRequestLocale(locale);
  const l = locale as PeakLocale;
  const u = PEAK_UI[l];
  // Українські вершини першими для uk, решта — за висотою.
  const ua = new Set(["hoverla", "petros", "pip-ivan", "ai-petri"]);
  const list = [...PEAKS].sort((a, b) => {
    if (l === "uk" && ua.has(a.slug) !== ua.has(b.slug)) return ua.has(a.slug) ? -1 : 1;
    return b.elev - a.elev;
  });

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "ItemList",
        name: u.indexH1,
        itemListElement: list.map((p, i) => ({ "@type": "ListItem", position: i + 1, url: localeUrl(locale, `/gory/${p.slug}`), name: p.t[l].name })),
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: u.crumb, item: localeUrl(locale, "/gory") },
        ],
      },
    ],
  };

  return (
    <main id="main-content" tabIndex={-1} className="mx-auto max-w-[1100px] px-5 py-12 lg:py-16">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      <nav className="text-[13px] text-ink-3" aria-label="breadcrumb">
        <Link href="/" className="hover:underline">Monadruk</Link>
        {" / "}
        <span className="text-ink">{u.crumb}</span>
      </nav>
      <h1 className="mt-4 text-[clamp(28px,4vw,46px)] leading-tight">{u.indexH1}</h1>
      <p className="mt-4 max-w-[760px] text-[15.5px] leading-relaxed text-ink-2">{u.indexIntro}</p>
      <div className="mt-6">
        <Link
          href="/mountains"
          className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-6 py-3 text-sm font-semibold text-white transition hover:opacity-90"
        >
          {u.ctaAny} →
        </Link>
      </div>

      <ul className="mt-10 grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
        {list.map((p, i) => (
          <li key={p.slug}>
            <Link href={`/gory/${p.slug}`} className="group block h-full overflow-hidden rounded-[18px] border border-line bg-paper">
              <div className="aspect-[4/3] overflow-hidden">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={p.photo} alt={p.t[l].name} loading={i < 4 ? "eager" : "lazy"} className="h-full w-full object-cover transition duration-500 group-hover:scale-[1.05]" />
              </div>
              <span className="block px-3 pt-2.5 text-[14px] font-semibold leading-snug text-ink">{p.t[l].name}</span>
              <span className="block px-3 pb-3 text-[12.5px] text-ink-3">{fmt(p.elev)} {u.m} · {p.t[l].country}</span>
            </Link>
          </li>
        ))}
      </ul>
      <p className="mt-8 text-[13px] leading-relaxed text-ink-3">{u.dataNote}</p>
    </main>
  );
}
