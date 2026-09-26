import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { localeUrl, pageMetadata } from "@/i18n/metadata";
import { routing, defaultLocale, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { CITY_PAGES } from "@/lib/cityPages";
import { occasionFaq, contentLocale } from "@/lib/cityLanding";
import { BLOG_ARTICLES, blogContent, blogLocales } from "@/lib/blog";

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
  const isUA = locale === "uk";
  const faq = occasionFaq(contentLocale(locale));
  // Подарункові статті блогу (нові зверху), лише ті, що мають справжній переклад цією мовою.
  const giftArticles = BLOG_ARTICLES
    .filter((a) => /podarun|podaruvaty/.test(a.slug) && blogLocales(a).includes(locale))
    .sort((a, b) => b.date.localeCompare(a.date))
    .slice(0, 12);

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

      {/* 25.09.2026: «Кому подарувати» — ті самі формулювання, що в пошуку й у рекламній
          групі «Подарунки» (Планувальник: 1–10 тис./міс кожне). Відповідність запит →
          оголошення → сторінка піднімає Quality Score і знижує ціну кліку. Лише uk. */}
      {isUA && (
        <section className="mt-12">
          <h2 className="text-[20px] font-semibold">Кому подарувати 3D-мапу чи брелок з картою</h2>
          <ul className="mt-4 grid gap-2 sm:grid-cols-2">
            {[
              ["Подарунок хлопцю", "shcho-podaruvaty-khloptsevi-na-den-narodzhennya", "Мапа району, де він виріс, або місця вашого знайомства."],
              ["Подарунок дівчині", "podarunok-divchyni", "Мапа-серце місця першого побачення з датою."],
              ["Подарунок чоловіку", "podarunok-choloviku", "Для того, у кого вже все є: його місто в 3D."],
              ["Подарунок на річницю весілля", "podarunok-cholovikovi-na-richnytsyu-vesillya", "Місце весілля чи першої спільної квартири."],
              ["Подарунок батькам", "podarunok-batkam-na-richnytsyu-vesillya", "Рідний двір чи вулиця, де ви росли."],
              ["Подарунок тату", "shcho-podaruvaty-tatovi-na-den-narodzhennya", "Його рідне місто чи село обʼємною мапою."],
              ["Подарунок військовому", "podarunok-viyskovomu-3d-mapa-ridnoho-mista", "Брелок з мапою дому — легкий, завжди з собою."],
              ["Подарунок вчителю", "podarunok-vchytelyu-na-den-vchytelya", "Мапа району школи з написом класу."],
              ["Подарунок на новосілля", "podarunok-na-novosillya-druzyam", "3D-мапа нового району на поличку."],
              ["Подарунок колезі", "podarunok-kolezi-na-zvilnennya-abo-pereyizd", "Мапа офісу чи міста, куди колега переїжджає."],
            ].map(([title, slug, desc]) => (
              <li key={slug}>
                <Link href={`/blog/${slug}`} className="block rounded-[14px] border border-line-soft bg-white/60 px-4 py-3 transition hover:border-[var(--accent)]">
                  <span className="block text-[14.5px] font-semibold text-ink">{title}</span>
                  <span className="mt-0.5 block text-[13px] leading-snug text-ink-2">{desc}</span>
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* 26.09.2026: три найчастіші рекламні запити з низьким Quality Score
          («подарунок на день народження», «подарок парню», «подарунок на день захисника»)
          — окремі H2 з тими самими словами + посилання на глибші сторінки. Лише uk. */}
      {isUA && (
        <section className="mt-12 flex flex-col gap-8">
          {[
            {
              h: "Подарунок на день народження, який не загубиться серед інших",
              p: "Оригінальний подарунок на день народження — це 3D-мапа місця, важливого саме для іменинника: двір дитинства, район першої квартири, місто, де він народився. Модель друкуємо з реальних даних карт: будинки, вулиці, річки й парки в обʼємі. 3D-мапа на полицю коштує від 350 ₴ (5,5 см) до 770 ₴ (15 см), брелок з картою — від 170 ₴. Виготовлення 2–4 дні, доставка Новою поштою по всій Україні.",
              links: [
                ["Подарунок на день народження — ідеї та розміри", "/podarunok/na-den-narodzhennya"],
                ["Що подарувати хлопцю на день народження", "/blog/shcho-podaruvaty-khloptsevi-na-den-narodzhennya"],
              ],
            },
            {
              h: "Подарунок хлопцю (подарок парню) з сенсом",
              p: "Коли не хочеться дарувати ще один гаджет чи парфуми, подаруйте хлопцю мапу місця, де ви познайомились, або його рідного міста. Брелок з картою завжди на ключах, а мапа-панно стане на полицю чи стіну. На звороті можна додати дату чи короткий напис — подарунок стане особистим.",
              links: [
                ["Брелок з мапою на ключі", "/keychains"],
                ["Мапа для пари", "/podarunok/dlya-pary"],
              ],
            },
            {
              h: "Подарунок на День захисників і захисниць",
              p: "Військовому найцінніше те, що нагадує про дім. Брелок з мапою рідного міста чи вулиці легкий, міцний і завжди поруч. Для побратимів чи підрозділу робимо серію з однаковою мапою і різними написами.",
              links: [
                ["Ідеї подарунків на День захисників", "/blog/podarunok-na-den-zakhysnykiv-i-zakhysnyts"],
                ["Подарунок військовому", "/blog/podarunok-viyskovomu-3d-mapa-ridnoho-mista"],
              ],
            },
          ].map((b) => (
            <div key={b.h}>
              <h2 className="text-[20px] font-semibold">{b.h}</h2>
              <p className="mt-3 text-[15px] leading-relaxed text-ink-2">{b.p}</p>
              <p className="mt-2 flex flex-wrap gap-x-5 gap-y-1">
                {b.links.map(([label, href]) => (
                  <Link key={href} href={href} className="text-[14px] font-semibold text-[var(--accent-strong)] hover:underline">
                    {label} →
                  </Link>
                ))}
              </p>
            </div>
          ))}
        </section>
      )}

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

      {/* FAQ (видимий, +FAQPage JSON-LD вище). */}
      <section className="mt-12">
        <h2 className="text-[20px] font-semibold">{isUA ? "Часті запитання" : "FAQ"}</h2>
        <dl className="mt-4 flex flex-col gap-4">
          {faq.map((f) => (
            <div key={f.q}>
              <dt className="text-[15px] font-semibold text-ink">{f.q}</dt>
              <dd className="mt-1.5 text-[14.5px] leading-relaxed text-ink-2">{f.a}</dd>
            </div>
          ))}
        </dl>
      </section>

      {/* 24.09.2026: внутрішні посилання на подарункові статті блогу (сезонні — День
          захисників, День вчителя, 14 лютого — і нагоди). Лише мови з реальним перекладом. */}
      {giftArticles.length > 0 && (
        <section className="mt-12">
          <h2 className="text-[20px] font-semibold">{locale === "uk" ? "Ідеї подарунків" : "Gift ideas"}</h2>
          <ul className="mt-4 grid gap-2 sm:grid-cols-2">
            {giftArticles.map((a) => (
              <li key={a.slug}>
                <Link
                  href={`/blog/${a.slug}`}
                  className="block rounded-[14px] border border-line-soft bg-white/60 px-4 py-3 text-[14px] font-medium leading-snug text-ink-2 transition hover:border-[var(--accent)] hover:text-ink"
                >
                  {blogContent(a, locale).h1}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}

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
