"use client";

import Link from "next/link";
import { useState } from "react";
import {
  ArrowRight, ArrowUpRight, Layers3, Leaf, Ruler, ShieldCheck,
  Sparkles, KeyRound, MapPin, Download, Star, Search, Box, Truck,
} from "lucide-react";
import { MAP_TEMPLATES, MAP_STYLE_PRESETS } from "@/lib/templates";

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

export default function HomePage() {
  return (
    <div className="min-h-[100dvh]">
      <SiteHeader />
      <Hero />
      <PathSelector />
      <HowItWorks />
      <TemplatesGallery />
      <Craft />
      <Testimonials />
      <FinalCTA />
      <SiteFooter />
    </div>
  );
}

/* ---------- Header ---------- */
function SiteHeader() {
  const [open, setOpen] = useState(false);
  return (
    <header className="sticky top-0 z-50 border-b border-line-soft bg-[rgba(244,239,228,0.85)] backdrop-blur">
      <div className="mx-auto flex max-w-[1360px] items-center justify-between px-5 py-4 lg:px-8">
        <Link href="/" className="flex items-center gap-2 font-serif text-xl font-semibold tracking-tight text-ink">
          <Box size={22} className="text-forest" />
          monadruk
        </Link>
        <nav className="hidden items-center gap-8 text-sm text-ink-2 md:flex">
          <a href="#how" className="hover:text-ink">Як це працює</a>
          <a href="#templates" className="hover:text-ink">Шаблони</a>
          <Link href="/keychains" className="hover:text-ink">Брелки</Link>
        </nav>
        <div className="flex items-center gap-2.5">
          <Link
            href="/keychains"
            className="hidden items-center gap-1.5 rounded-full border border-bronze/40 bg-bronze/10 px-4 py-2 text-sm font-semibold text-bronze transition hover:bg-bronze/20 sm:inline-flex"
            style={{ borderColor: "rgba(142,107,61,0.4)", color: "var(--bronze, #8E6B3D)", background: "rgba(142,107,61,0.08)" }}
          >
            <KeyRound size={15} /> Брелок
          </Link>
          <Link
            href="/create"
            className="inline-flex items-center gap-1.5 rounded-full bg-forest px-5 py-2.5 text-sm font-bold text-[#F4EFE4] shadow-[0_10px_24px_rgba(46,74,58,0.28)] transition hover:opacity-90"
            style={{ background: "var(--forest, #2E4A3A)" }}
          >
            Створити мапу <ArrowRight size={15} />
          </Link>
        </div>
      </div>
    </header>
  );
}

/* ---------- Hero ---------- */
function Hero() {
  return (
    <section className="border-b border-line-soft">
      <div className="mx-auto grid max-w-[1360px] items-center gap-12 px-5 py-16 lg:grid-cols-[1fr_1.05fr] lg:px-8 lg:py-24">
        <div className="fade-up">
          <Eyebrow dot>Преміум 3D-мапи · Друк удома або на замовлення</Eyebrow>
          <h1 className="mt-6 text-[clamp(44px,6vw,84px)] leading-[1.04]">
            Твоє місто.<br />
            <span className="italic text-forest">Виміряне</span> в 3D.
          </h1>
          <p className="mt-7 max-w-[520px] text-[17px] leading-relaxed text-ink-2">
            Обери район, що для тебе щось значить — або будь-яку точку на Землі.
            Ми перетворимо її на тактильну архітектурну мапу з висотами будинків,
            парків і річок. Завантаж готовий 3D-файл і друкуй.
          </p>
          <div className="mt-9 flex flex-wrap gap-3">
            <Link href="/create" className="btn btn-primary btn-lg">
              Створити свою мапу <ArrowRight size={16} />
            </Link>
            <Link href="/keychains" className="btn btn-ghost btn-lg">
              <KeyRound size={16} /> Брелок з мапою
            </Link>
          </div>
          <div className="mt-12 flex flex-wrap gap-x-9 gap-y-5 border-t border-line-soft pt-8">
            <Stat n="Будь-яке місто" l="по всьому світу" />
            <Stat n="3MF · STL" l="готово до друку" />
            <Stat n="Eco PLA" l="біопластик" />
          </div>
        </div>
        <div className="relative">
          <div className="card card-paper overflow-hidden rounded-[20px] p-4 shadow-lift">
            <div className="flex items-center justify-between px-2 pb-3 pt-1">
              <div className="flex items-center gap-2">
                <span className="pulse h-2 w-2 rounded-full bg-forest" />
                <span className="eyebrow">Жива демонстрація · Київ, Поділ</span>
              </div>
              <span className="font-mono text-[11px] text-ink-3">2.4 × 1.8 км</span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="aspect-[5/4] overflow-hidden rounded-[14px] border border-line-soft">
                <MapTile />
              </div>
              <div className="aspect-[5/4] overflow-hidden rounded-[14px] border border-line-soft bg-bg-2">
                <MapTile accent="#1F3328" paper="#E2D9C2" />
              </div>
            </div>
            <div className="flex items-center justify-between px-1 pt-4">
              <span className="text-[13px] text-ink-2">Зліва — карта, справа — 3D-результат</span>
              <Link href="/create" className="btn btn-primary btn-sm">
                Спробувати <ArrowRight size={14} />
              </Link>
            </div>
          </div>
          <FloatBadge cls="-right-3 -top-3"><Leaf size={14} className="text-forest" /> Eco PLA</FloatBadge>
          <FloatBadge cls="-bottom-4 left-8"><Download size={14} className="text-forest" /> Готовий 3MF</FloatBadge>
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
  return (
    <section className="mx-auto max-w-[1360px] px-5 py-16 lg:px-8 lg:py-20">
      <div className="grid gap-5 md:grid-cols-2">
        <PathCard
          href="/create"
          primary
          eyebrow="Шлях 1 · Власна геометрія"
          title="Створити свою мапу"
          desc="Знайди будь-яке місце на Землі, окресли зону, обери стиль і розмір. 5 кроків — близько 3 хвилин."
          cta="Запустити конструктор"
          icon={<Sparkles size={22} />}
        />
        <PathCard
          href="/keychains"
          eyebrow="Шлях 2 · Аксесуар"
          title="Брелок з мапою"
          desc="Мініатюра твого району на ключах. Жетон 55×30, класичний або квадратний — з твоїм написом."
          cta="Відкрити майстерню брелків"
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
  const steps = [
    { n: "01", t: "Виділяєте зону", d: "Шукайте місто та перетягуйте рамку. Розміри в км — без координат." },
    { n: "02", t: "Налаштовуєте", d: "Стиль, шари, розмір. Превʼю оновлюється в реальному часі." },
    { n: "03", t: "Генеруємо", d: "Сервер будує точну 3D-модель з даних OpenStreetMap і висот." },
    { n: "04", t: "Завантажуєте", d: "Готовий 3MF для Bambu Studio / PrusaSlicer. Друкуйте вдома." },
  ];
  return (
    <section id="how" className="bg-ink py-20 text-[#E8E1CC] lg:py-28">
      <div className="mx-auto max-w-[1360px] px-5 lg:px-8">
        <div className="mb-12 flex flex-col justify-between gap-6 md:flex-row md:items-end">
          <div>
            <Eyebrow dot light>Процес</Eyebrow>
            <h2 className="mt-4 max-w-[600px] text-[clamp(32px,4vw,56px)] text-[#F4EFE4]">
              Від точки на карті — до моделі у твоєму принтері
            </h2>
          </div>
          <p className="max-w-[340px] text-[15px] leading-relaxed text-[#A8AC9F]">
            Прозоро. Без технічного жаргону. Готовий файл для друку — за кілька хвилин.
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
          <Link href="/create" className="btn btn-bronze btn-lg">Почати <ArrowRight size={16} /></Link>
        </div>
      </div>
    </section>
  );
}

/* ---------- Templates gallery ---------- */
function TemplatesGallery() {
  const accents = ["#2E4A3A", "#3F5B45", "#9A7242", "#5B5E5A"];
  return (
    <section id="templates" className="mx-auto max-w-[1360px] px-5 py-20 lg:px-8 lg:py-24">
      <div className="mb-10 flex flex-col justify-between gap-4 md:flex-row md:items-end">
        <div>
          <Eyebrow dot>Готові шаблони</Eyebrow>
          <h2 className="mt-4 text-[clamp(30px,3.4vw,52px)]">Почни з відомого району</h2>
          <p className="mt-3 max-w-[520px] text-[15px] text-ink-2">
            Обери готовий пресет — він одразу відкриє конструктор з виставленою зоною. Або створи з нуля.
          </p>
        </div>
        <Link href="/create" className="btn btn-ghost hidden sm:inline-flex">
          Усі міста <ArrowRight size={14} />
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
              <MapTile accent={accents[i % accents.length]} />
              {t.tag && (
                <span className="absolute left-3 top-3 rounded-full bg-paper-2/90 px-3 py-1 text-[11px] font-semibold text-forest">
                  {t.tag}
                </span>
              )}
            </div>
            <div className="flex items-center justify-between px-4 py-4">
              <div>
                <div className="font-serif text-[19px] leading-tight">{t.district}</div>
                <div className="mt-0.5 text-[12px] text-ink-3">{t.city}</div>
              </div>
              <ArrowUpRight size={18} className="text-ink-3 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
            </div>
          </Link>
        ))}
      </div>

      {/* Style presets */}
      <div className="mt-14">
        <Eyebrow>Стилі готової мапи</Eyebrow>
        <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {MAP_STYLE_PRESETS.map((p) => (
            <div key={p.id} className="rounded-[14px] border border-line bg-paper p-5">
              <Layers3 size={20} className="text-forest" />
              <div className="mt-3 font-semibold">{p.label}</div>
              <div className="mt-1 text-[13px] text-ink-3">{p.blurb}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ---------- Craft / specs ---------- */
function Craft() {
  const specs = [
    { icon: <Ruler size={18} />, t: "Точний друк", d: "Оптимізовано під FDM" },
    { icon: <Leaf size={18} />, t: "PLA-біопластик", d: "Кукурудзяний крохмаль" },
    { icon: <Layers3 size={18} />, t: "Реальні дані", d: "OpenStreetMap + висоти" },
    { icon: <Download size={18} />, t: "Формат 3MF", d: "Bambu / Prusa готово" },
  ];
  return (
    <section className="mx-auto max-w-[1360px] px-5 py-20 lg:px-8">
      <div className="grid items-center gap-16 lg:grid-cols-2">
        <div>
          <Eyebrow dot>Якість</Eyebrow>
          <h2 className="mt-4 mb-6 text-[clamp(30px,3.4vw,52px)]">Не сувенір. <span className="italic">Документ.</span></h2>
          <p className="mb-9 max-w-[520px] text-[16px] leading-relaxed text-ink-2">
            Кожна мапа — це геодезичні дані OpenStreetMap і реальні висоти. Модель
            автоматично спрощується для чистого FDM-друку: мінімальні товщини,
            кольорові шари для багатоколірних принтерів (Bambu AMS).
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
          <div className="aspect-[1/1.7] overflow-hidden rounded-[18px] border border-line-soft"><MapTile accent="#2E4A3A" paper="#EFE6D2" /></div>
          <div className="flex flex-col gap-4">
            <div className="aspect-[1.4/1] overflow-hidden rounded-[18px] border border-line-soft"><MapTile accent="#3F5B45" paper="#E2D9C2" /></div>
            <div className="aspect-[1.4/1] overflow-hidden rounded-[18px] border border-line-soft"><MapTile accent="#9A7242" paper="#F4EFE4" /></div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ---------- Testimonials ---------- */
function Testimonials() {
  const items = [
    { q: "Все дуже сподобалось, дякую! Результат перевершив очікування.", a: "Анна" },
    { q: "Швидко й зручно. Все вийшло чудово, рекомендую.", a: "Тарас" },
    { q: "Дуже якісно, все чітко. Залишилась задоволена.", a: "Олена" },
    { q: "Простий і приємний сервіс. Усе спрацювало з першого разу.", a: "Дмитро" },
    { q: "Гарний результат, акуратно зроблено. Дякую за роботу!", a: "Ірина" },
    { q: "Зробив за кілька хвилин, усе зрозуміло. Класно!", a: "Максим" },
    { q: "Дуже задоволена, вийшло саме так, як хотіла.", a: "Софія" },
    { q: "Все на висоті, користуватися легко. Дякую!", a: "Андрій" },
  ];
  return (
    <section className="bg-bg-2 py-20 lg:py-28">
      <div className="mx-auto max-w-[1360px] px-5 lg:px-8">
        <h2 className="mb-3 max-w-[560px] text-[clamp(28px,3.2vw,46px)]">
          Що кажуть клієнти
        </h2>
        <p className="mb-10 text-[15px] text-ink-2">Гортайте, щоб побачити більше відгуків →</p>
        <div className="-mx-5 flex snap-x snap-mandatory gap-5 overflow-x-auto px-5 pb-4 lg:-mx-8 lg:px-8 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {items.map((t, i) => (
            <article key={i} className="card card-paper flex w-[300px] shrink-0 snap-start flex-col p-7">
              <div className="mb-4 flex gap-1">
                {[...Array(5)].map((_, k) => <Star key={k} size={14} className="text-bronze" fill="currentColor" />)}
              </div>
              <p className="mb-6 flex-1 font-serif text-[18px] leading-snug">«{t.q}»</p>
              <div className="flex items-center justify-between border-t border-line-soft pt-5">
                <div className="text-[14px] font-semibold">{t.a}</div>
                <span className="text-[11px] uppercase tracking-[0.1em] text-ink-3">Відгук</span>
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
  return (
    <section className="mx-auto max-w-[1360px] px-5 py-20 lg:px-8 lg:py-24">
      <div className="grid items-center gap-12 overflow-hidden rounded-[32px] bg-forest px-8 py-16 text-[#F4EFE4] lg:grid-cols-[1.4fr_1fr] lg:px-16">
        <div>
          <Eyebrow dot light>Готові почати?</Eyebrow>
          <h2 className="mb-6 mt-4 max-w-[560px] text-[clamp(30px,3.4vw,52px)] text-[#F4EFE4]">
            Створіть мапу місця, що значить більше за крапку на карті.
          </h2>
          <p className="mb-9 max-w-[480px] text-[16px] leading-relaxed text-[rgba(244,239,228,0.78)]">
            5 кроків, близько 3 хвилин — і ви завантажуєте готовий 3D-файл для друку.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link href="/create" className="btn btn-bronze btn-lg">Створити мапу <ArrowRight size={16} /></Link>
            <Link href="/keychains" className="btn btn-ghost btn-lg" style={{ color: "#F4EFE4", borderColor: "rgba(244,239,228,0.4)" }}>
              Брелок
            </Link>
          </div>
        </div>
        <div className="aspect-square overflow-hidden rounded-[24px] border border-[rgba(244,239,228,0.15)]">
          <MapTile accent="#1F3328" paper="#3A5446" />
        </div>
      </div>
    </section>
  );
}

/* ---------- Footer ---------- */
function SiteFooter() {
  return (
    <footer className="border-t border-line-soft py-12">
      <div className="mx-auto flex max-w-[1360px] flex-col items-center justify-between gap-6 px-5 text-sm text-ink-3 md:flex-row lg:px-8">
        <div className="flex items-center gap-2 font-serif text-lg text-ink">
          <Box size={18} className="text-forest" /> monadruk
        </div>
        <div className="flex gap-6">
          <Link href="/create" className="hover:text-ink">Створити мапу</Link>
          <Link href="/keychains" className="hover:text-ink">Брелки</Link>
          <a href="#how" className="hover:text-ink">Як це працює</a>
        </div>
        <div>© {new Date().getFullYear()} monadruk.com</div>
      </div>
    </footer>
  );
}
