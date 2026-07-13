import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { localeUrl, pageMetadata } from "@/i18n/metadata";
import { routing, defaultLocale, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { CITY_PAGES } from "@/lib/cityPages";

/**
 * Подарункова/під-нагоду посадкова сторінка («3D-мапа та брелок на подарунок»).
 * Програмне SEO під запити «подарунок на річницю / весілля / новосілля / для пари».
 * Кожна нагода → коротке prose + CTA у конструктор (/create або /keychains).
 * JSON-LD: ItemList нагод + BreadcrumbList. Той самий патерн, що /maps/[city].
 */

export const dynamicParams = false;

// Нагоди (data-driven): id → i18n-ключ (gift.<id>Title/<id>Desc/<id>Cta) + ціль CTA.
// Хвиля 2: картки ведуть на повні лендінги /podarunok/[slug] (глибший контент),
// а не одразу в конструктор.
const OCCASIONS = [
  { id: "anniversary", href: "/podarunok/na-richnytsyu" },
  { id: "birthday", href: "/podarunok/na-den-narodzhennya" },
  { id: "housewarming", href: "/podarunok/na-novosillya" },
  { id: "couple", href: "/podarunok/dlya-pary" },
  { id: "corporate", href: "/podarunok/korporatyvnyi-podarunok" },
] as const;

export async function generateMetadata({
  params,
}: {
  params: { locale: string };
}): Promise<Metadata> {
  return pageMetadata({ locale: params.locale, path: "/podarunok", ns: "giftMeta" });
}

export default async function GiftPage({
  params,
}: {
  params: { locale: string };
}) {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  setRequestLocale(locale);
  const t = await getTranslations({ locale, namespace: "gift" });

  const path = "/podarunok";
  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "ItemList",
        name: t("h1"),
        description: t("intro"),
        itemListElement: OCCASIONS.map((o, i) => ({
          "@type": "ListItem",
          position: i + 1,
          name: t(`${o.id}Title`),
          description: t(`${o.id}Desc`),
        })),
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: t("breadcrumb"), item: localeUrl(locale, path) },
        ],
      },
    ],
  };

  return (
    <main id="main-content" tabIndex={-1} className="mx-auto max-w-[820px] px-5 py-14 lg:py-20">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      <nav className="text-[13px] text-ink-3" aria-label="breadcrumb">
        <Link href="/" className="hover:underline">Monadruk</Link>
        {" / "}
        <span className="text-ink">{t("breadcrumb")}</span>
      </nav>
      <h1 className="mt-5 text-[clamp(28px,4vw,46px)] leading-tight">{t("h1")}</h1>
      <p className="mt-5 text-[15px] leading-relaxed text-ink-2">{t("intro")}</p>
      <p className="mt-3 text-[15px] leading-relaxed text-ink-2">{t("intro2")}</p>

      <div className="mt-8 flex flex-wrap gap-3">
        <Link
          href="/create"
          className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-6 py-3 text-sm font-semibold text-white transition hover:opacity-90"
        >
          {t("ctaMap")}
        </Link>
        <Link
          href="/keychains"
          className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] border border-line-soft bg-white/80 px-6 py-3 text-sm font-semibold text-ink transition hover:border-[var(--accent)]"
        >
          {t("ctaKeychain")}
        </Link>
      </div>

      <section className="mt-12">
        <h2 className="text-[20px] font-semibold">{t("occasionsTitle")}</h2>
        <ul className="mt-5 grid gap-4 sm:grid-cols-2">
          {OCCASIONS.map((o) => (
            <li key={o.id}>
              <div className="flex h-full flex-col rounded-[18px] border border-line-soft bg-white/60 px-5 py-5">
                <h3 className="text-[16px] font-semibold text-ink">{t(`${o.id}Title`)}</h3>
                <p className="mt-2 flex-1 text-[14px] leading-relaxed text-ink-2">{t(`${o.id}Desc`)}</p>
                <Link
                  href={o.href}
                  className="mt-4 inline-flex min-h-[44px] items-center text-[14px] font-semibold text-[var(--accent-strong)] hover:underline"
                >
                  {t(`${o.id}Cta`)} →
                </Link>
              </div>
            </li>
          ))}
        </ul>
      </section>

      {/* Подарунок × місто (хвиля 2 programmatic SEO): чіпи на /podarunok/[city].
          Заголовок bilingual-inline — контент цільових сторінок uk/en з lib. */}
      <section className="mt-12">
        <h2 className="text-[20px] font-semibold">
          {locale === "uk" ? "Подарунок з вашого міста" : "A gift from your city"}
        </h2>
        <ul className="mt-4 flex flex-wrap gap-2">
          {CITY_PAGES.map((c) => (
            <li key={c.slug}>
              <Link
                href={`/podarunok/${c.slug}`}
                className="inline-block rounded-full border border-line-soft bg-white/70 px-4 py-2 text-[13.5px] font-medium text-ink-2 transition hover:border-[var(--accent)] hover:text-ink"
              >
                {c.names[locale]}
              </Link>
            </li>
          ))}
        </ul>
      </section>

      {/* Закривальний абзац для SEO + м'яка повторна CTA. */}
      <section className="mt-12 rounded-[18px] border border-line-soft bg-white/60 px-5 py-5">
        <p className="text-[14.5px] leading-relaxed text-ink-2">{t("outro")}</p>
        <Link
          href="/prices"
          className="mt-3 inline-block text-[13.5px] font-semibold text-[var(--accent-strong)] hover:underline"
        >
          {t("pricesLink")} →
        </Link>
      </section>
    </main>
  );
}
