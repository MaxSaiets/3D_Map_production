"use client";

import {
  ArrowRight, ArrowUpRight, Layers3, Leaf, Ruler,
  Sparkles, KeyRound, Download, Star,
} from "lucide-react";
import dynamic from "next/dynamic";
import { useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import { MAP_TEMPLATES, MAP_STYLE_PRESETS } from "@/lib/templates";
import { SiteHeader } from "@/components/SiteHeader";
import { SiteFooter } from "@/components/SiteFooter";

const ShowcaseSection = dynamic(() => import("@/components/ShowcaseSection"), { ssr: false });
const Model3DViewer = dynamic(() => import("@/components/Model3DViewer"), { ssr: false });

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

export default function HomePage() {
  return (
    <div className="min-h-[100dvh]">
      <SiteHeader />
      <Hero />
      <PathSelector />
      <ShowcaseSection />
      <HowItWorks />
      <TemplatesGallery />
      <Craft />
      <Testimonials />
      <Faq />
      <SeoTextBlock />
      <FinalCTA />
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
                  <Model3DViewer url="/models/keychain-fea.glb" height={260} label={t("viewerKeychain")} />
                </div>
                <p className="mt-1.5 px-1 text-center text-[11px] font-medium uppercase tracking-[0.08em] text-ink-3">{t("capKeychain")}</p>
              </div>
              <div>
                <div className="overflow-hidden rounded-[14px] border border-line-soft bg-gradient-to-b from-[#f6f1e6] to-[#ece4d3]">
                  <Model3DViewer url="/models/map-dense.glb" height={260} label={t("viewerMap")} />
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
    </section>
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

/* ---------- Templates gallery ---------- */
function TemplatesGallery() {
  const t = useTranslations("home.templates");
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
            <div className="relative aspect-[16/10] overflow-hidden">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`/showcase/map-${(i % 11) + 1}.png`}
                alt={tAlt("districtMap", { district: tg(`district.${t.id}`), city: tCity(t.cityKey) })}
                loading="lazy"
                className="absolute inset-0 h-full w-full object-cover transition-transform duration-500 group-hover:scale-[1.06]"
              />
              {badgeKey(t.tag) && (
                <span className="absolute left-3 top-3 rounded-full bg-paper-2/90 px-3 py-1 text-[11px] font-semibold text-forest">
                  {tg(`badge.${badgeKey(t.tag)}`)}
                </span>
              )}
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
          <div className="aspect-[1/1.7] overflow-hidden rounded-[18px] border border-line-soft">
            <img src="/showcase/keychain-1.png" alt={tAlt("keychainMap")} loading="lazy" className="h-full w-full object-cover" />
          </div>
          <div className="flex flex-col gap-4">
            <div className="aspect-[1.4/1] overflow-hidden rounded-[18px] border border-line-soft">
              <img src="/showcase/map-2.png" alt={tAlt("district")} loading="lazy" className="h-full w-full object-cover" />
            </div>
            <div className="aspect-[1.4/1] overflow-hidden rounded-[18px] border border-line-soft">
              <img src="/showcase/keychain-5.png" alt={tAlt("keychainMap")} loading="lazy" className="h-full w-full object-cover" />
            </div>
          </div>
          {/* eslint-enable @next/next/no-img-element */}
        </div>
      </div>
    </section>
  );
}

/* ---------- Testimonials ---------- */
function Testimonials() {
  const t = useTranslations("home.testimonials");
  const items = [
    { q: t("q1"), a: "Anna" },
    { q: t("q2"), a: "Taras" },
    { q: t("q3"), a: "Olena" },
    { q: t("q4"), a: "Dmytro" },
    { q: t("q5"), a: "Iryna" },
    { q: t("q6"), a: "Maksym" },
  ];
  const tBadge = t("badge");
  return (
    <section className="bg-bg-2 py-20 lg:py-28">
      <div className="mx-auto max-w-[1360px] px-5 lg:px-8">
        <h2 className="mb-3 max-w-[560px] text-[clamp(28px,3.2vw,46px)]">
          {t("title")}
        </h2>
        <p className="mb-10 text-[15px] text-ink-2">{t("sub")}</p>
        <div className="-mx-5 flex snap-x snap-mandatory gap-5 overflow-x-auto px-5 pb-4 lg:-mx-8 lg:px-8 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {items.map((t, i) => (
            <article key={i} className="card card-paper flex w-[300px] shrink-0 snap-start flex-col p-7">
              <div className="mb-4 flex gap-1">
                {[...Array(5)].map((_, k) => <Star key={k} size={14} className="text-bronze" fill="currentColor" />)}
              </div>
              <p className="mb-6 flex-1 font-serif text-[18px] leading-snug">«{t.q}»</p>
              <div className="flex items-center justify-between border-t border-line-soft pt-5">
                <div className="text-[14px] font-semibold">{t.a}</div>
                <span className="text-[11px] uppercase tracking-[0.1em] text-ink-3">{tBadge}</span>
              </div>
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
        </div>
        <div className="aspect-square overflow-hidden rounded-[24px] border border-[rgba(244,239,228,0.15)] bg-[rgba(244,239,228,0.06)]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/showcase/map-1.png" alt={tAlt("cityMap")} loading="lazy" className="h-full w-full object-cover" />
        </div>
      </div>
    </section>
  );
}

