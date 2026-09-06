"use client";

import { ArrowRight, ArrowUpRight, Boxes, Download, KeyRound, Layers3, LayoutGrid, Leaf, MapPin, Ruler, Sparkles, Star, Truck } from "lucide-react";
import dynamic from "next/dynamic";
import { useTranslations, useLocale } from "next-intl";

import { Link } from "@/i18n/navigation";
import { MAP_TEMPLATES, MAP_STYLE_PRESETS } from "@/lib/templates";
import { CITY_PAGES } from "@/lib/cityPages";
import { MAP_PRICE_RANGE, MAP_SIZE_PRICES_UAH, mapPriceEur } from "@/lib/mapPrices";
import { SiteHeader } from "@/components/SiteHeader";
import { SiteFooter } from "@/components/SiteFooter";

const ShowcaseSection = dynamic(() => import("@/components/ShowcaseSection"), { ssr: false });
const RealPrints = dynamic(() => import("@/components/RealPrints"), { ssr: false });
// T-6.5: three.js (≈1.5 МБ у dev) вантажиться, лише коли демо доїхало до екрана —
// на телефоні герой-демо нижче згину (заміряно: канвасів 0, а чанк уже завантажено).
const Model3DViewer = dynamic(() => import("@/components/Model3DViewerLazy"), { ssr: false });

/* ---------- decorative isometric map tile (pure SVG, fast) ---------- */
function MapTile({ accent = "#2E4A3A", paper = "#EFE6D2" }: { accent?: string; paper?: string }) {
  return (
    <svg viewBox="0 0 200 160" className="h-full w-full" preserveAspectRatio="xMidYMid slice">
      <rect width="200" height="160" fill={paper} />
      <g opacity="0.9">
        <path d="M0 90 L80 50 L130 75 L70 110 Z" fill="#C9D9D4" opacity="0.5" />
        <path d="M30 30 L60 15 L90 30 L60 45 Z" fill="#D6DDB5" opacity="0.7" />
        {[...Array(7)].map((_, i) => (
          <rect key={i} x={20 + i * 22} y={60 + (i % 3) * 8} width="14" height={18 + (i % 4) * 10}
            fill={accent} opacity={0.18 + (i % 3) * 0.12} transform={`skewY(-12)`} />
        ))}
        <path d="M0 70 L200 40" stroke={accent} strokeWidth="2" opacity="0.25" />
        <path d="M10 110 L180 80" stroke={accent} strokeWidth="3" opacity="0.2" />
      </g>
    </svg>
  );
}

function Eyebrow({ children, dot, light }: { children: React.ReactNode; dot?: boolean; light?: boolean }) {
  return (
    <div className={`eyebrow ${dot ? "eyebrow-dot" : ""}`} style={light ? { color: "rgba(244,239,228,.7)" } : undefined}>
      {children}
    </div>
  );
}

/* ---------- FAQ (visible + FAQPage structured data for rich results) ---------- */
function Faq() {
  const t = useTranslations("home.faq");
  const items = [1, 2, 3, 4, 5, 6, 7, 8, 9].map((i) => ({ q: t(`q${i}`), a: t(`a${i}`) }));
  const faqLd = {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    mainEntity: items.map((it) => ({
      "@type": "Question",
      name: it.q,
      acceptedAnswer: { "@type": "Answer", text: it.a },
    })),
  };
  return (
    <section className="border-t border-line-soft" aria-labelledby="faq-title">
      <div className="mx-auto max-w-[820px] px-5 py-16 lg:py-24">
        <div className="text-center">
          <Eyebrow dot>{t("eyebrow")}</Eyebrow>
          <h2 id="faq-title" className="mt-4 text-[clamp(28px,4vw,46px)] leading-tight">
            {t("title")}
          </h2>
        </div>
        <div className="mt-9 divide-y divide-line-soft">
          {items.map((it, i) => (
            <details key={i} className="group py-4">
              <summary className="flex cursor-pointer list-none items-center justify-between gap-4 text-[16px] font-semibold text-ink [&::-webkit-details-marker]:hidden">
                {it.q}
                <ArrowRight size={16} className="shrink-0 text-ink-3 transition group-open:rotate-90" />
              </summary>
              <p className="mt-3 text-[15px] leading-relaxed text-ink-2">{it.a}</p>
            </details>
          ))}
        </div>
      </div>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(faqLd) }} />
    </section>
  );
}

/* ---------- SEO text block: видимий локалізований опис із пошуковими
   фразами (3D-мапа міста, брелок з картою, топо-рельєф, магніт) ---------- */
function SeoTextBlock() {
  const t = useTranslations("home.seo");
  return (
    <section className="border-t border-line-soft" aria-labelledby="seo-title">
      <div className="mx-auto max-w-[820px] px-5 py-12 lg:py-16">
        <h2 id="seo-title" className="text-[clamp(20px,2.6vw,28px)] leading-snug">
          {t("title")}
        </h2>
        <p className="mt-4 text-[14.5px] leading-relaxed text-ink-2">{t("p1")}</p>
        <p className="mt-3 text-[14.5px] leading-relaxed text-ink-2">{t("p2")}</p>
      </div>
    </section>
  );
}

/* ---------- Popular cities (internal links → de-orphan city pages) ---------- */
function PopularCities() {
  const t = useTranslations("home.popularCities");
  const locale = useLocale();
  const cities = CITY_PAGES.slice(0, 16);
  return (
    <section className="border-t border-line-soft bg-bg-2/40">
      <div className="mx-auto max-w-[1360px] px-5 py-14 lg:px-8 lg:py-20">
        <h2 className="text-[clamp(22px,2.8vw,34px)]">{t("title")}</h2>
        <p className="mt-3 text-[14.5px] text-ink-2">{t("subtitle")}</p>
        <ul className="mt-6 flex flex-wrap gap-2.5">
          {cities.map((c) => (
            <li key={c.slug}>
              <Link
                href={`/maps/${c.slug}`}
                className="inline-flex min-h-[44px] items-center rounded-full border border-line-soft bg-paper/70 px-4 py-2 text-[14px] font-semibold text-ink-2 transition hover:border-forest/40 hover:text-ink"
              >
                {(c.names as Record<string, string>)[locale] ?? c.names.uk}
              </Link>
            </li>
          ))}
          <li>
            <Link
              href="/maps"
              className="inline-flex min-h-[44px] items-center gap-1.5 rounded-full bg-forest/10 px-4 py-2 text-[14px] font-bold text-forest transition hover:bg-forest/15"
              style={{ color: "var(--forest, #2E4A3A)" }}
            >
              {t("allCities")} <ArrowRight size={14} />
            </Link>
          </li>
        </ul>
      </div>
    </section>
  );
}

export default function HomePage() {
  return (
    <div className="min-h-[100dvh]">
      <SiteHeader />
      {/* Ціль skip-to-content (після хедера) — клавіатура/скрінрідер стрибає сюди,
          оминаючи навігацію. tabIndex=-1 робить <main> програмно фокусованим. */}
      <main id="main-content" tabIndex={-1}>
        <Hero />
        <PathSelector />
        {/* T-4.8: реальні фото друків одразу після вибору шляху — довіра до
            того, як прокрутити 3 екрани (особливо на телефоні). */}
        <RealPrints />
        <ShowcaseSection />
        <HowItWorks />
        <TemplatesGallery />
        <Craft />
        <Testimonials />
        <Faq />
        <SeoTextBlock />
        <PopularCities />
        <FinalCTA />
      </main>
      <SiteFooter />
    </div>
  );
}

/* ---------- Hero ---------- */
function Hero() {
  const t = useTranslations("home.hero");
  const tAlt = useTranslations("home.alt");
  return (
    <section className="border-b border-line-soft">
      <div className="mx-auto grid max-w-[1360px] items-center gap-12 px-5 py-16 lg:grid-cols-[1fr_1.05fr] lg:px-8 lg:py-24">
        <div className="fade-up">
          <Eyebrow dot>{t("eyebrow")}</Eyebrow>
          <h1 className="mt-6 text-[clamp(44px,6vw,84px)] leading-[1.04]">
            {t("title1")}<br />
            <span className="italic text-forest">{t("titleItalic")}</span> {t("title2")}
          </h1>
          <p className="mt-7 max-w-[520px] text-[17px] leading-relaxed text-ink-2">
            {t("desc")}
          </p>
          <div className="mt-9 flex flex-wrap gap-3">
            <Link href="/create" className="btn btn-primary btn-lg">
              {t("ctaCreate")} <ArrowRight size={16} />
            </Link>
            <Link href="/keychains" className="btn btn-ghost btn-lg">
              <KeyRound size={16} /> {t("ctaKeychain")}
            </Link>
          </div>
          <Link href="/prices" className="mt-5 inline-flex items-center gap-2 rounded-full border border-line-soft bg-paper/70 px-4 py-2 text-[13px] font-semibold text-ink-2 transition hover:border-forest/40 hover:text-ink">
            <Truck size={15} className="text-forest" /> {t("shipPill", { p: MAP_PRICE_RANGE.uk.low })}
          </Link>
          <div className="mt-12 flex flex-wrap gap-x-9 gap-y-5 border-t border-line-soft pt-8">
            <Stat n={t("stat1n")} l={t("stat1l")} />
            <Stat n={t("stat2n")} l={t("stat2l")} />
            <Stat n={t("stat3n")} l={t("stat3l")} />
          </div>
        </div>
        <div className="relative">
          <div className="card card-paper overflow-hidden rounded-[20px] p-4 shadow-lift">
            <div className="flex items-center justify-between px-2 pb-3 pt-1">
              <div className="flex items-center gap-2">
                <span className="pulse h-2 w-2 rounded-full bg-forest" />
                <span className="eyebrow">{t("demoLabel")}</span>
              </div>
              <span className="font-mono text-[11px] text-ink-3">{tAlt("dims")}</span>
            </div>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              <div>
                <div className="overflow-hidden rounded-[14px] border border-line-soft bg-gradient-to-b from-[#f6f1e6] to-[#ece4d3]">
                  <Model3DViewer url="/models/keychain-fea.glb" height={260} label={t("viewerKeychain")} poster="/showcase/card-keychain-400.webp" />
                </div>
                <p className="mt-1.5 px-1 text-center text-[11px] font-medium uppercase tracking-[0.08em] text-ink-3">{t("capKeychain")}</p>
              </div>
              <div>
                <div className="overflow-hidden rounded-[14px] border border-line-soft bg-gradient-to-b from-[#f6f1e6] to-[#ece4d3]">
                  <Model3DViewer url="/models/map-dense.glb" height={260} label={t("viewerMap")} poster="/showcase/card-map3d-400.webp" />
                </div>
                <p className="mt-1.5 px-1 text-center text-[11px] font-medium uppercase tracking-[0.08em] text-ink-3">{t("capMap")}</p>
              </div>
            </div>
            <div className="flex items-center justify-between px-1 pt-4">
              <span className="text-[13px] text-ink-2">{t("dragHint")}</span>
              <Link href="/create" className="btn btn-primary btn-sm">
                {t("try")} <ArrowRight size={14} />
              </Link>
            </div>
          </div>
          <FloatBadge cls="-right-3 -top-3"><Leaf size={14} className="text-forest" /> {t("badgeEco")}</FloatBadge>
          <FloatBadge cls="-bottom-4 left-8"><Download size={14} className="text-forest" /> {t("badgeReady")}</FloatBadge>
        </div>
      </div>
    </section>
  );
}

function Stat({ n, l, star }: { n: string; l: string; star?: boolean }) {
  return (
    <div>
      <div className="flex items-center gap-1.5 font-serif text-[22px] leading-tight">
        {n} {star && <Star size={17} className="text-bronze" fill="currentColor" />}
      </div>
      <div className="mt-1.5 text-[11px] uppercase tracking-[0.08em] text-ink-3">{l}</div>
    </div>
  );
}

function FloatBadge({ children, cls }: { children: React.ReactNode; cls: string }) {
  return (
    <div className={`absolute ${cls} inline-flex items-center gap-2 rounded-full border border-line bg-paper-2 px-4 py-2.5 text-[13px] font-medium shadow-soft`}>
      {children}
    </div>
  );
}

/* ---------- Two paths ---------- */
function PathSelector() {
  const t = useTranslations("home.path");
  return (
    <section className="mx-auto max-w-[1360px] px-5 py-16 lg:px-8 lg:py-20">
      <div className="grid gap-5 md:grid-cols-2">
        <PathCard
          href="/create"
          primary
          eyebrow={t("eyebrow1")}
          title={t("title1")}
          desc={t("desc1")}
          cta={t("cta1")}
          icon={<Sparkles size={22} />}
        />
        <PathCard
          href="/keychains"
          eyebrow={t("eyebrow2")}
          title={t("title2")}
          desc={t("desc2")}
          cta={t("cta2")}
          icon={<KeyRound size={22} />}
        />
      </div>
      {/* Власник: «не зрозуміло, які взагалі можливості». Два великі шляхи лишаються
          головними, а решта продуктів — видимим рядком одразу під ними (раніше про
          них можна було дізнатись лише випадково з футера). */}
      <MoreCapabilities />
      <GiftOccasions />
    </section>
  );
}

/* Власник: «дуже мало людей знаходять сайт». Люди НЕ шукають «купити 3D-мапу»
   (памʼять keyword-research), вони шукають «подарунок на річницю / новосілля /
   для пари». Сторінки під ці наміри вже є (/podarunok/<привід>), але з головної
   на них не вело жодне посилання — тепер ведуть чіпи одразу під вибором шляху. */
function GiftOccasions() {
  const t = useTranslations("home.occasions");
  const items = [
    { slug: "na-richnytsyu", label: t("richnytsia") },
    { slug: "na-den-narodzhennya", label: t("birthday") },
    { slug: "na-novosillya", label: t("housewarming") },
    { slug: "dlya-pary", label: t("couple") },
    { slug: "korporatyvnyi-podarunok", label: t("corporate") },
  ];
  return (
    <div className="mt-8" data-testid="home-gift-occasions">
      <div className="eyebrow mb-1">{t("title")}</div>
      <p className="mb-3 text-[14px] text-ink-2">{t("sub")}</p>
      <div className="flex flex-wrap gap-2.5">
        {items.map((it) => (
          <Link
            key={it.slug}
            href={`/podarunok/${it.slug}`}
            className="inline-flex min-h-[44px] items-center rounded-full border border-line-soft bg-paper/70 px-4 py-2 text-[14px] font-semibold text-ink-2 transition hover:border-forest/40 hover:text-ink"
          >
            🎁 {it.label}
          </Link>
        ))}
        <Link
          href="/podarunok"
          className="inline-flex min-h-[44px] items-center gap-1 rounded-full px-3 py-2 text-[14px] font-semibold text-forest underline-offset-2 hover:underline"
        >
          {t("all")} <ArrowRight size={14} />
        </Link>
      </div>
    </div>
  );
}

function MoreCapabilities() {
  const t = useTranslations("scenario");
  const items = [
    { href: "/panno", icon: <LayoutGrid size={15} />, title: t("pannoTitle"), desc: t("pannoDesc") },
    { href: "/maket", icon: <Ruler size={15} />, title: t("maketTitle"), desc: t("maketDesc") },
    { href: "/worlds", icon: <Sparkles size={15} />, title: t("worldsTitle"), desc: t("worldsDesc") },
    { href: "/showcase", icon: <Boxes size={15} />, title: t("showcaseTitle"), desc: t("showcaseDesc") },
  ];
  return (
    <div className="mt-8" data-testid="home-more-capabilities">
      <div className="eyebrow mb-3">{t("moreTitle")}</div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {items.map((it) => (
          <Link
            key={it.href}
            href={it.href}
            className="flex items-start gap-2.5 rounded-[18px] border border-line bg-paper px-4 py-3 transition hover:-translate-y-0.5 hover:border-forest/40"
          >
            <span className="mt-0.5 shrink-0 text-forest">{it.icon}</span>
            <span className="min-w-0">
              <span className="block text-[14px] font-semibold leading-tight text-ink">{it.title}</span>
              <span className="block text-[12px] leading-snug text-ink-3">{it.desc}</span>
            </span>
          </Link>
        ))}
      </div>
    </div>
  );
}

function PathCard({ href, eyebrow, title, desc, cta, icon, primary }: {
  href: string; eyebrow: string; title: string; desc: string; cta: string; icon: React.ReactNode; primary?: boolean;
}) {
  return (
    <Link
      href={href}
      className="group relative flex min-h-[300px] flex-col gap-6 overflow-hidden rounded-[24px] border border-line p-9 transition-transform hover:-translate-y-1"
      style={{ background: primary ? "var(--forest)" : "var(--paper)", color: primary ? "#F4EFE4" : "var(--ink)" }}
    >
      <div className="flex items-start justify-between">
        <div className="eyebrow" style={{ color: primary ? "rgba(244,239,228,.7)" : "var(--ink-3)" }}>{eyebrow}</div>
        <div className="flex h-12 w-12 items-center justify-center rounded-full"
          style={{ background: primary ? "rgba(244,239,228,.12)" : "var(--bg-2)", color: primary ? "#F4EFE4" : "var(--forest)" }}>
          {icon}
        </div>
      </div>
      <div className="mt-auto">
        <h3 className="mb-3 text-[34px]" style={{ color: primary ? "#F4EFE4" : "var(--ink)" }}>{title}</h3>
        <p className="max-w-[440px] text-[15px] leading-relaxed" style={{ color: primary ? "rgba(244,239,228,.78)" : "var(--ink-2)" }}>{desc}</p>
      </div>
      <div className="inline-flex items-center gap-2 text-sm font-semibold">
        {cta} <ArrowRight size={16} className="transition-transform group-hover:translate-x-1" />
      </div>
    </Link>
  );
}

/* ---------- How it works ---------- */
function HowItWorks() {
  const t = useTranslations("home.how");
  const steps = [
    { n: "01", t: t("s1t"), d: t("s1d") },
    { n: "02", t: t("s2t"), d: t("s2d") },
    { n: "03", t: t("s3t"), d: t("s3d") },
    { n: "04", t: t("s4t"), d: t("s4d") },
  ];
  // HowTo-розмітка (локалізована) — rich-result «як це працює» у пошуку.
  const howLd = {
    "@context": "https://schema.org",
    "@type": "HowTo",
    name: t("title"),
    description: t("sub"),
    step: steps.map((s, i) => ({ "@type": "HowToStep", position: i + 1, name: s.t, text: s.d })),
  };
  return (
    <section id="how" className="bg-ink py-20 text-[#E8E1CC] lg:py-28">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(howLd) }} />
      <div className="mx-auto max-w-[1360px] px-5 lg:px-8">
        <div className="mb-12 flex flex-col justify-between gap-6 md:flex-row md:items-end">
          <div>
            <Eyebrow dot light>{t("eyebrow")}</Eyebrow>
            <h2 className="mt-4 max-w-[600px] text-[clamp(32px,4vw,56px)] text-[#F4EFE4]">
              {t("title")}
            </h2>
          </div>
          <p className="max-w-[340px] text-[15px] leading-relaxed text-[#A8AC9F]">
            {t("sub")}
          </p>
        </div>
        <div className="grid gap-px border-t border-[#2A3830] md:grid-cols-4">
          {steps.map((s, i) => (
            <div
              key={i}
              className={`pt-10 ${i > 0 ? "md:pl-8" : ""} ${i < 3 ? "md:border-r md:border-[#2A3830] md:pr-8" : ""}`}
            >
              <div className="mb-8 font-mono text-[13px] text-bronze-2">{s.n}</div>
              <h3 className="mb-3.5 text-[26px] text-[#F4EFE4]">{s.t}</h3>
              <p className="text-[14px] leading-relaxed text-[#A8AC9F]">{s.d}</p>
            </div>
          ))}
        </div>
        <div className="mt-14">
          <Link href="/create" className="btn btn-bronze btn-lg">{t("cta")} <ArrowRight size={16} /></Link>
        </div>
      </div>
    </section>
  );
}

// Плитки шаблонів району: замість циклічного stock-фото (T-4.7, false visual
// claim) — стилізований градієнт у палітрі ivory/forest/bronze, що чесно
// виглядає навмисно, а не як «зламана картинка».
const TILE_GRADIENTS = [
  "linear-gradient(135deg, var(--forest) 0%, var(--forest-2) 100%)",
  "linear-gradient(135deg, var(--forest-2) 0%, var(--ink) 100%)",
  "linear-gradient(135deg, var(--bronze) 0%, var(--forest-2) 100%)",
  "linear-gradient(135deg, var(--forest-3) 0%, var(--forest-2) 100%)",
];

/* ---------- Templates gallery ---------- */
function TemplatesGallery() {
  const t = useTranslations("home.templates");
  // Валюта як на решті сайту: uk — гривні, інші локалі — євро за тим самим
  // позиційним курсом (mapPriceEur). Раніше плитка писала «₴» усім мовам.
  const tileLocale = useLocale();
  const tilePrice = (uah: number) => (tileLocale === "uk" ? `${uah} ₴` : `€${mapPriceEur(uah)}`);
  const tAlt = useTranslations("home.alt");
  const tg = useTranslations("gallery");
  const tCity = useTranslations("cities");
  // Map a Ukrainian DATA tag to its gallery.badge.* key (data fields stay UA).
  const badgeKey = (tag?: string) =>
    tag === "Бестселер" ? "bestseller" : tag === "Новинка" ? "new" : tag === "Популярне" ? "popular" : null;
  return (
    <section id="templates" className="mx-auto max-w-[1360px] px-5 py-20 lg:px-8 lg:py-24">
      <div className="mb-10 flex flex-col justify-between gap-4 md:flex-row md:items-end">
        <div>
          <Eyebrow dot>{t("eyebrow")}</Eyebrow>
          <h2 className="mt-4 text-[clamp(30px,3.4vw,52px)]">{t("title")}</h2>
          <p className="mt-3 max-w-[520px] text-[15px] text-ink-2">
            {t("desc")}
          </p>
        </div>
        <Link href="/create" className="btn btn-ghost hidden sm:inline-flex">
          {t("all")} <ArrowRight size={14} />
        </Link>
      </div>
      <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {MAP_TEMPLATES.slice(0, 9).map((t, i) => (
          <Link
            key={t.id}
            href={{ pathname: "/create", query: { template: t.id } }}
            className="group overflow-hidden rounded-[18px] border border-line-soft bg-paper transition-transform hover:-translate-y-1 hover:shadow-soft"
          >
            <div
              className="relative aspect-[16/10] overflow-hidden transition-transform duration-500 group-hover:scale-[1.03]"
              style={{ background: TILE_GRADIENTS[i % TILE_GRADIENTS.length] }}
              role="img"
              aria-label={tAlt("districtMap", { district: tg(`district.${t.id}`), city: tCity(t.cityKey) })}
            >
              {/* Стилізована плитка замість фото — реального рендеру району ще нема,
                  а циклічне stock-фото (map-N.webp) видавало себе за фото району
                  (T-4.7). Контурні кола + пін чесно кажуть «це шаблон», не фото. */}
              <div
                className="absolute inset-0 opacity-[0.16]"
                style={{
                  backgroundImage:
                    "repeating-radial-gradient(circle at 82% 78%, transparent 0, transparent 14px, rgba(244,239,228,.9) 15px, rgba(244,239,228,.9) 16px)",
                }}
              />
              <div className="absolute left-3 top-3 flex h-9 w-9 items-center justify-center rounded-full bg-[rgba(244,239,228,.14)] text-[#F4EFE4]">
                <MapPin size={18} />
              </div>
              {badgeKey(t.tag) && (
                <span className="absolute right-3 top-3 rounded-full bg-paper-2/90 px-3 py-1 text-[11px] font-semibold text-forest">
                  {tg(`badge.${badgeKey(t.tag)}`)}
                </span>
              )}
              <div className="absolute bottom-3 left-3 right-3 flex items-end justify-between text-[#F4EFE4]">
                <div className="font-serif text-[15px] leading-tight opacity-90">{tCity(t.cityKey)}</div>
                <div className="text-right">
                  <div className="text-[13px] font-semibold">{t.sizeMm ?? 80} мм</div>
                  <div className="text-[12px] opacity-80">{tilePrice(MAP_SIZE_PRICES_UAH[(t.sizeMm ?? 80) as keyof typeof MAP_SIZE_PRICES_UAH] ?? Number(MAP_PRICE_RANGE.uk.low))}</div>
                </div>
              </div>
            </div>
            <div className="flex items-center justify-between px-4 py-4">
              <div>
                <div className="font-serif text-[19px] leading-tight">{tg(`district.${t.id}`)}</div>
                <div className="mt-0.5 text-[12px] text-ink-3">{tCity(t.cityKey)}</div>
              </div>
              <ArrowUpRight size={18} className="text-ink-3 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
            </div>
          </Link>
        ))}
      </div>

      {/* Style presets */}
      <div className="mt-14">
        <Eyebrow>{t("stylesEyebrow")}</Eyebrow>
        <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {MAP_STYLE_PRESETS.map((p) => (
            <div key={p.id} className="rounded-[14px] border border-line bg-paper p-5">
              <Layers3 size={20} className="text-forest" />
              <div className="mt-3 font-semibold">{tg(`style.${p.id}.label`)}</div>
              <div className="mt-1 text-[13px] text-ink-3">{tg(`style.${p.id}.blurb`)}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ---------- Craft / specs ---------- */
function Craft() {
  const t = useTranslations("home.craft");
  const tAlt = useTranslations("home.alt");
  const specs = [
    { icon: <Ruler size={18} />, t: t("spec1t"), d: t("spec1d") },
    { icon: <Leaf size={18} />, t: t("spec2t"), d: t("spec2d") },
    { icon: <Layers3 size={18} />, t: t("spec3t"), d: t("spec3d") },
    { icon: <Download size={18} />, t: t("spec4t"), d: t("spec4d") },
  ];
  return (
    <section className="mx-auto max-w-[1360px] px-5 py-20 lg:px-8">
      <div className="grid items-center gap-16 lg:grid-cols-2">
        <div>
          <Eyebrow dot>{t("eyebrow")}</Eyebrow>
          <h2 className="mt-4 mb-6 text-[clamp(30px,3.4vw,52px)]">{t("title")} <span className="italic">{t("titleItalic")}</span></h2>
          <p className="mb-9 max-w-[520px] text-[16px] leading-relaxed text-ink-2">
            {t("desc")}
          </p>
          <div className="grid grid-cols-2 gap-7">
            {specs.map((s) => (
              <div key={s.t} className="flex items-start gap-3.5">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-[10px] bg-bg-2 text-forest">{s.icon}</div>
                <div>
                  <div className="text-[15px] font-semibold">{s.t}</div>
                  <div className="text-[13px] text-ink-3">{s.d}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4">
          {/* eslint-disable @next/next/no-img-element */}
          {/* Реальні фото друків (не рендери) — «справжній бізнес»-сигнал + чесна
              картинка того, що приїде клієнту. Джерело: public/real/. */}
          <div className="aspect-[1/1.7] overflow-hidden rounded-[18px] border border-line-soft">
            <img src="/real/heart-1.webp" alt={tAlt("keychainMap")} loading="lazy" className="h-full w-full object-cover" />
          </div>
          <div className="flex flex-col gap-4">
            <div className="aspect-[1.4/1] overflow-hidden rounded-[18px] border border-line-soft">
              <img src="/real/map-1.webp" alt={tAlt("district")} loading="lazy" className="h-full w-full object-cover" />
            </div>
            <div className="aspect-[1.4/1] overflow-hidden rounded-[18px] border border-line-soft">
              <img src="/real/panno-1.webp" alt={tAlt("district")} loading="lazy" className="h-full w-full object-cover" />
            </div>
          </div>
          {/* eslint-enable @next/next/no-img-element */}
        </div>
      </div>
    </section>
  );
}

/* ---------- Trust facts (T-4.4) ---------- */
/* Раніше тут були 6 цитат із вигаданими іменами (Anna/Taras/…) без джерела й дати —
   ризик довіри і GSC merchant-listing spam. Реальних відгуків у репозиторії немає,
   тож секція показує лише ПЕРЕВІРЮВАНІ факти: усе нижче можна побачити на сайті
   (галерея реальних друків, дані OSM, способи доставки й оплати, ФОП у футері). */
function Testimonials() {
  const t = useTranslations("home.testimonials");
  const facts = [
    { t: t("f1t"), d: t("f1d") },
    { t: t("f2t"), d: t("f2d") },
    { t: t("f3t"), d: t("f3d") },
    { t: t("f4t"), d: t("f4d") },
  ];
  return (
    <section className="bg-bg-2 py-20 lg:py-28" data-testid="home-trust">
      <div className="mx-auto max-w-[1360px] px-5 lg:px-8">
        <h2 className="mb-3 max-w-[560px] text-[clamp(28px,3.2vw,46px)]">{t("title")}</h2>
        <p className="mb-10 max-w-[620px] text-[15px] text-ink-2">{t("sub")}</p>
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {facts.map((f, i) => (
            <article key={i} className="card card-paper flex flex-col p-7">
              <h3 className="font-serif text-[20px] leading-snug">{f.t}</h3>
              <p className="mt-3 text-[14px] leading-relaxed text-ink-2">{f.d}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ---------- Final CTA ---------- */
function FinalCTA() {
  const t = useTranslations("home.cta");
  const tAlt = useTranslations("home.alt");
  const tGift = useTranslations("gift");
  return (
    <section className="mx-auto max-w-[1360px] px-5 py-20 lg:px-8 lg:py-24">
      <div className="grid items-center gap-12 overflow-hidden rounded-[32px] bg-forest px-8 py-16 text-[#F4EFE4] lg:grid-cols-[1.4fr_1fr] lg:px-16">
        <div>
          <Eyebrow dot light>{t("eyebrow")}</Eyebrow>
          <h2 className="mb-6 mt-4 max-w-[560px] text-[clamp(30px,3.4vw,52px)] text-[#F4EFE4]">
            {t("title")}
          </h2>
          <p className="mb-9 max-w-[480px] text-[16px] leading-relaxed text-[rgba(244,239,228,0.78)]">
            {t("desc")}
          </p>
          <div className="flex flex-wrap gap-3">
            <Link href="/create" className="btn btn-bronze btn-lg">{t("create")} <ArrowRight size={16} /></Link>
            <Link href="/keychains" className="btn btn-ghost btn-lg" style={{ color: "#F4EFE4", borderColor: "rgba(244,239,228,0.4)" }}>
              {t("keychain")}
            </Link>
          </div>
          {/* Внутрішнє перелінкування на подарункову сторінку (SEO + сезонний попит). */}
          <Link
            href="/podarunok"
            className="mt-5 inline-flex items-center gap-1.5 text-[14px] font-semibold underline-offset-4 hover:underline"
            style={{ color: "rgba(244,239,228,0.85)" }}
          >
            {tGift("homeCtaLink")} <ArrowRight size={14} />
          </Link>
        </div>
        <div className="aspect-square overflow-hidden rounded-[24px] border border-[rgba(244,239,228,0.15)] bg-[rgba(244,239,228,0.06)]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/real/map-2.webp" alt={tAlt("cityMap")} loading="lazy" className="h-full w-full object-cover" />
        </div>
      </div>
    </section>
  );
}

