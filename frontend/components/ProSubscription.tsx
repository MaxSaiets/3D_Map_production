"use client";

/**
 * Підписка Monadruk Pro: продаюча сторінка + оформлення (LiqPay regular payment) і
 * керування (статус, скасування в один клік). Юридично важливе:
 *  • три ОКРЕМІ непоставлені за замовчуванням позначки (умови, автопродовження,
 *    негайний доступ = втрата 14-денної відмови) — без усіх трьох кнопка неактивна,
 *    бекенд теж відхиляє (400); тексти позначок ідуть у запис згоди;
 *  • ціна, періодичність і дата наступного списання видно ДО оплати;
 *  • скасування — кнопка тут же, без дзвінків/листів.
 *
 * 07.10.2026: сторінка була голим списком із 4 пунктів — людина не бачила, чим Pro
 * кращий за поштучний файл (149 ₴) і що саме лише Pro дає право ПРОДАВАТИ вироби
 * (поштучний файл — особисте використання, оферта п. «Ліцензія»). Додано:
 * калькулятор «з якого файлу Pro вигідніший» (ціна файлу — з /api/subscription/plans,
 * та сама, що в чеку файлу), порівняння, «для кого», FAQ, липку кнопку на мобільному.
 * Цифри тільки чесні: ніяких «тисяч клієнтів» і вигаданих відгуків.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocale, useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import {
  CheckCircle2, ChevronDown, Infinity as InfinityIcon, Loader2, Minus, MousePointerClick,
  Printer, ShieldCheck, Store, Sparkles, Building2, Send, Mail,
} from "lucide-react";
import { useAuth } from "@/components/AuthProvider";
import { SUB_PRICE } from "@/lib/legal/subscription";
import { BUSINESS } from "@/lib/legal";
import { FILE_PRICE_UAH, breakEvenFiles } from "@/lib/mapPrices";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";
type Ccy = "UAH" | "USD";

interface SubView {
  status: string; active?: boolean; amount?: number; currency?: Ccy;
  paid_until?: string | null; renews_at?: string | null; cancelled_at?: string | null;
  payments?: { amount: number; currency: string; ts: string }[];
}

const LOCALE_TAG: Record<string, string> = { uk: "uk-UA", en: "en-US", de: "de-DE", es: "es-ES", fr: "fr-FR", pl: "pl-PL", ro: "ro-RO" };
/** Приблизний курс для ціни файлу в доларах (дзеркало pricing.json fx.uah_per_usd), якщо бекенд не відповів. */
const TG_URL = "https://t.me/monadruk";
const UAH_PER_USD_FALLBACK = 41.5;

function fmtPrice(ccy: Ccy, amount: number, locale: string, digits = 0) {
  return ccy === "UAH"
    ? `${new Intl.NumberFormat(LOCALE_TAG[locale] ?? "uk-UA", { maximumFractionDigits: digits }).format(amount)} ₴`
    : new Intl.NumberFormat(LOCALE_TAG[locale] ?? "en-US", { style: "currency", currency: "USD", maximumFractionDigits: digits, minimumFractionDigits: 0 }).format(amount);
}

function addMonth(d: Date) {
  const r = new Date(d);
  const day = r.getDate();
  r.setDate(1); r.setMonth(r.getMonth() + 1);
  r.setDate(Math.min(day, new Date(r.getFullYear(), r.getMonth() + 1, 0).getDate()));
  return r;
}

export function ProSubscription() {
  const t = useTranslations("proSub");
  const locale = useLocale();
  const dateFmt = (iso?: string | null | Date) => {
    if (!iso) return "—";
    const d = iso instanceof Date ? iso : new Date(iso);
    return new Intl.DateTimeFormat(LOCALE_TAG[locale] ?? "uk-UA", { year: "numeric", month: "long", day: "numeric" }).format(d);
  };
  const { user, loading, signIn, getIdToken } = useAuth();
  const [ccy, setCcy] = useState<Ccy>(locale === "uk" ? "UAH" : "USD");
  const [plans, setPlans] = useState<Record<Ccy, number>>({ ...SUB_PRICE });
  const [filePrice, setFilePrice] = useState<Record<Ccy, number>>({
    UAH: FILE_PRICE_UAH, USD: Math.round((FILE_PRICE_UAH / UAH_PER_USD_FALLBACK) * 10) / 10,
  });
  const [sub, setSub] = useState<SubView | null>(null);
  const [checks, setChecks] = useState({ terms: false, autorenew: false, digital: false });
  const [busy, setBusy] = useState<"pay" | "cancel" | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [pay, setPay] = useState<{ action_url: string; data: string; signature: string } | null>(null);
  const [filesPerMonth, setFilesPerMonth] = useState(20);
  const formRef = useRef<HTMLFormElement>(null);
  const checkoutRef = useRef<HTMLDivElement>(null);
  const [checkoutVisible, setCheckoutVisible] = useState(true);
  const [scrolled, setScrolled] = useState(false);
  // 07.10.2026: продаж підписки вмикає бекенд (SUB_SALES_OPEN, /api/subscription/plans).
  // Доки прапорця нема — замість оплати картка «скоро» (документи ще не готові).
  const [salesOpen, setSalesOpen] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 700);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  useEffect(() => { if (pay) formRef.current?.submit(); }, [pay]);

  // Валюта за країною (Cloudflare): Україна → гривня, решта світу → долар.
  useEffect(() => {
    fetch(`${API_BASE}/api/subscription/plans`).then((r) => (r.ok ? r.json() : null)).then((j) => {
      if (!j) return;
      if (j.plans) setPlans({ UAH: Number(j.plans.UAH) || SUB_PRICE.UAH, USD: Number(j.plans.USD) || SUB_PRICE.USD });
      if (j.file && Number(j.file.UAH) > 0) {
        const uah = Number(j.file.UAH);
        setFilePrice({ UAH: uah, USD: Number(j.file.USD) > 0 ? Number(j.file.USD) : Math.round((uah / UAH_PER_USD_FALLBACK) * 10) / 10 });
      }
      if (j.suggested === "UAH" || j.suggested === "USD") setCcy(j.suggested);
      setSalesOpen(j.sales_open === true);
    }).catch(() => {});
  }, []);

  // Липка кнопка на мобільному — лише коли картку оформлення не видно.
  useEffect(() => {
    const el = checkoutRef.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([e]) => setCheckoutVisible(e.isIntersecting), { threshold: 0.15 });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const loadSub = useCallback(async () => {
    const token = await getIdToken();
    if (!token) return;
    try {
      const r = await fetch(`${API_BASE}/api/subscription`, { headers: { Authorization: `Bearer ${token}` }, cache: "no-store" });
      if (r.ok) setSub((await r.json()).subscription);
    } catch { /* мережа — покажемо форму */ }
  }, [getIdToken]);

  useEffect(() => { if (user) loadSub(); }, [user, loadSub]);

  // Повернення з LiqPay (?paid=1): callback може дійти із запізненням — опитуємо кілька разів.
  useEffect(() => {
    if (!user || typeof window === "undefined") return;
    if (!new URLSearchParams(location.search).get("paid")) return;
    let n = 0;
    const id = window.setInterval(() => { n++; loadSub(); if (n >= 10) window.clearInterval(id); }, 3000);
    return () => window.clearInterval(id);
  }, [user, loadSub]);

  const price = fmtPrice(ccy, plans[ccy], locale);
  const fileDigits = ccy === "USD" ? 1 : 0;
  const filePriceLabel = `${ccy === "USD" ? "≈ " : ""}${fmtPrice(ccy, filePrice[ccy], locale, fileDigits)}`;
  const breakEven = breakEvenFiles(plans[ccy], filePrice[ccy]);
  const perFileTotal = filesPerMonth * filePrice[ccy];
  const saving = perFileTotal - plans[ccy];
  const nextCharge = useMemo(() => dateFmt(addMonth(new Date())), [locale]); // eslint-disable-line react-hooks/exhaustive-deps
  const allChecked = checks.terms && checks.autorenew && checks.digital;

  // Плоскі тексти позначок (без посилань) — те, що бачила людина; зберігаються як доказ згоди.
  const consentTexts = () => [
    t("cbTermsPlain"),
    t("cbAutorenew", { price }),
    t("cbDigital"),
  ];

  const subscribe = async () => {
    if (!allChecked) return;
    setMsg(null); setBusy("pay");
    try {
      const token = await getIdToken();
      if (!token) { signIn(); return; }
      import("@/lib/analytics").then((m) => m.track("pro_checkout", { currency: ccy })).catch(() => {});
      const r = await fetch(`${API_BASE}/api/subscription/checkout`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ currency: ccy, locale, accept_terms: true, accept_autorenew: true, accept_digital: true, consent_texts: consentTexts() }),
      });
      if (r.status === 503) { setMsg(t("payUnavailable")); return; }
      if (r.status === 409) { await loadSub(); return; }
      if (!r.ok) throw new Error(String(r.status));
      const j = await r.json();
      setPay(j.payment);
    } catch {
      setMsg(t("error"));
    } finally {
      setBusy(null);
    }
  };

  const cancel = async () => {
    if (!sub) return;
    if (!window.confirm(t("cancelConfirm", { date: dateFmt(sub.paid_until) }))) return;
    setMsg(null); setBusy("cancel");
    try {
      const token = await getIdToken();
      const r = await fetch(`${API_BASE}/api/subscription/cancel`, { method: "POST", headers: { Authorization: `Bearer ${token}` } });
      if (!r.ok) throw new Error(String(r.status));
      setSub((await r.json()).subscription);
      setMsg(t("cancelled"));
    } catch {
      setMsg(t("cancelFailed"));
    } finally {
      setBusy(null);
    }
  };

  const goCheckout = () => {
    import("@/lib/analytics").then((m) => m.track("pro_cta", { place: "hero" })).catch(() => {});
    checkoutRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const features = [t("f1"), t("f2"), t("f3"), t("f4")];
  const showStatus = sub && (sub.active || sub.status === "pending");
  const isActive = !!sub && sub.status === "active";

  // Порівняння: три колонки (безкоштовно / один файл / Pro). true = ✓, false = —, рядок = текст.
  const cmpRows: { label: string; cells: (string | boolean)[] }[] = [
    { label: t("cmpPreview"), cells: [true, true, true] },
    { label: t("cmpGenerate"), cells: [t("cmpLimited"), t("cmpLimited"), t("cmpUnlimited")] },
    { label: t("cmpFiles"), cells: [false, t("cmpOneFile"), t("cmpUnlimited")] },
    { label: t("cmpPersonal"), cells: [false, true, true] },
    { label: t("cmpCommercial"), cells: [false, false, true] },
  ];

  const audience = [
    { icon: Printer, title: t("who1Title"), text: t("who1Text") },
    { icon: Store, title: t("who2Title"), text: t("who2Text") },
    { icon: Building2, title: t("who3Title"), text: t("who3Text") },
  ];

  const faq = [1, 2, 3, 4, 5, 6].map((i) => ({ q: t(`faq${i}q`), a: t(`faq${i}a`, { breakEven, price, filePrice: filePriceLabel }) }));

  const statusCard = showStatus && (
    <div data-testid="pro-status">
      <h2 className="font-serif text-2xl text-ink">
        {sub!.status === "pending" ? t("pending") : sub!.status === "cancelled" ? t("cancelledTitle") : t("activeTitle")}
      </h2>
      {sub!.status === "active" && (
        <p className="mt-2 text-sm text-ink-2">{t("renewsOn", { date: dateFmt(sub!.renews_at), price: fmtPrice((sub!.currency as Ccy) || "UAH", sub!.amount || 0, locale) })}</p>
      )}
      {sub!.status === "cancelled" && (
        <p className="mt-2 text-sm text-ink-2">{t("cancelledNote", { date: dateFmt(sub!.paid_until) })}</p>
      )}
      {sub!.status === "pending" && <Loader2 className="mt-3 h-5 w-5 animate-spin text-ink-3" />}
      {!!sub!.payments?.length && (
        <div className="mt-5">
          <div className="text-[11px] uppercase tracking-wide text-ink-3">{t("payments")}</div>
          <ul className="mt-1 space-y-1 text-sm text-ink-2">
            {sub!.payments!.slice().reverse().map((p) => (
              <li key={p.ts} className="tabular-nums">{dateFmt(p.ts)} — {fmtPrice(p.currency as Ccy, p.amount, locale)}</li>
            ))}
          </ul>
        </div>
      )}
      <div className="mt-6 flex flex-wrap gap-3">
        <Link href="/create" className="inline-flex min-h-[44px] items-center rounded-full bg-forest px-5 py-2.5 text-sm font-bold text-white hover:opacity-90" style={{ background: "var(--forest,#2E4A3A)" }}>
          {t("goCreate")}
        </Link>
        <Link href="/account" className="inline-flex min-h-[44px] items-center rounded-full border border-line bg-white px-5 py-2.5 text-sm font-semibold text-ink hover:bg-paper">
          {t("goAccount")}
        </Link>
        {sub!.status === "active" && (
          <button onClick={cancel} disabled={busy === "cancel"}
            className="inline-flex min-h-[44px] items-center gap-2 rounded-full px-3 py-2.5 text-sm font-semibold text-ink-3 underline-offset-2 hover:text-ink hover:underline disabled:opacity-60">
            {busy === "cancel" && <Loader2 className="h-4 w-4 animate-spin" />} {t("cancel")}
          </button>
        )}
      </div>
    </div>
  );

  const soonCard = (
    <div data-testid="pro-soon">
      <h2 className="font-serif text-2xl text-ink">{t("soonTitle")}</h2>
      <ul className="mt-4 space-y-2">
        {features.map((f) => (
          <li key={f} className="flex items-start gap-2 text-[14px] text-ink-2">
            <CheckCircle2 size={17} className="mt-0.5 shrink-0 text-forest" /><span>{f}</span>
          </li>
        ))}
      </ul>
      <p className="mt-5 border-t border-line pt-5 text-sm leading-relaxed text-ink-2">{t("soonText", { filePrice: filePriceLabel })}</p>
      <Link href="/create" className="mt-4 inline-flex min-h-[48px] w-full items-center justify-center rounded-full bg-forest px-5 py-3 text-[15px] font-bold text-white hover:opacity-90" style={{ background: "var(--forest,#2E4A3A)" }}>
        {t("soonCta")}
      </Link>
      {/* 07.10.2026: поки продаж закритий — збираємо зацікавлених напряму */}
      <div className="mt-5 rounded-2xl border border-line bg-white p-4" data-testid="pro-interest">
        <p className="text-[14px] font-semibold text-ink">{t("soonContactTitle")}</p>
        <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{t("soonContactText")}</p>
        <div className="mt-3 grid gap-2 sm:grid-cols-2">
          <a href={TG_URL} target="_blank" rel="noopener"
            onClick={() => { import("@/lib/analytics").then((m) => m.track("pro_interest", { channel: "tg" })).catch(() => {}); }}
            className="inline-flex min-h-[44px] items-center justify-center gap-2 rounded-full border border-[#1f8fcb]/40 bg-[#eef7fc] px-4 text-[14px] font-semibold text-[#1f8fcb] hover:bg-[#e1f1fa]">
            <Send size={16} /> Telegram
          </a>
          <a href={`mailto:${BUSINESS.email}?subject=${encodeURIComponent("Monadruk Pro")}`}
            onClick={() => { import("@/lib/analytics").then((m) => m.track("pro_interest", { channel: "email" })).catch(() => {}); }}
            className="inline-flex min-h-[44px] items-center justify-center gap-2 rounded-full border border-line bg-white px-4 text-[14px] font-semibold text-ink hover:bg-paper">
            <Mail size={16} /> {t("soonContactEmail")}
          </a>
        </div>
        <p className="mt-2 text-center text-[12px] text-ink-3">{BUSINESS.email}</p>
      </div>
    </div>
  );

  const checkoutCard = (
    <div>
      <div role="radiogroup" aria-label={t("currency")} className="inline-flex rounded-full border border-line bg-white p-1">
        {(["UAH", "USD"] as Ccy[]).map((c) => (
          <button key={c} role="radio" aria-checked={ccy === c} onClick={() => setCcy(c)}
            className={`min-h-[36px] rounded-full px-3 py-1.5 text-[13px] font-semibold sm:px-4 sm:text-sm ${ccy === c ? "bg-ink text-white" : "text-ink-2"}`}>
            {c === "UAH" ? t("currencyUah") : t("currencyUsd")}
          </button>
        ))}
      </div>
      <div className="mt-5 flex items-baseline gap-2">
        <span className="font-serif text-4xl text-ink" data-testid="pro-price">{price}</span>
        <span className="text-ink-2">{t("perMonth")}</span>
      </div>
      <p className="mt-1 text-sm text-ink-2">{t("priceNote", { date: nextCharge })}</p>
      <ul className="mt-4 space-y-2">
        {features.map((f) => (
          <li key={f} className="flex items-start gap-2 text-[14px] text-ink-2">
            <CheckCircle2 size={17} className="mt-0.5 shrink-0 text-forest" /><span>{f}</span>
          </li>
        ))}
      </ul>

      {!user ? (
        <div className="mt-6 border-t border-line pt-5">
          <p className="text-sm text-ink-2">{t("loginFirst")}</p>
          <button onClick={signIn} disabled={loading} data-testid="pro-login"
            className="mt-3 inline-flex min-h-[48px] w-full items-center justify-center rounded-full bg-forest px-5 py-3 text-[15px] font-bold text-white hover:opacity-90" style={{ background: "var(--forest,#2E4A3A)" }}>
            {t("login")}
          </button>
          <p className="mt-2 text-center text-[12px] text-ink-3">{t("loginHint")}</p>
        </div>
      ) : (
        <div className="mt-6 space-y-3 border-t border-line pt-5">
          <label className="flex cursor-pointer items-start gap-3 text-sm text-ink-2">
            <input type="checkbox" className="mt-0.5 h-5 w-5 shrink-0 accent-[#2E4A3A]" checked={checks.terms} onChange={(e) => setChecks((c) => ({ ...c, terms: e.target.checked }))} />
            <span>{t.rich("cbTerms", {
              terms: (ch) => <Link href="/pro-terms" className="text-forest underline" target="_blank">{ch}</Link>,
              offer: (ch) => <Link href="/offer" className="text-forest underline" target="_blank">{ch}</Link>,
              privacy: (ch) => <Link href="/privacy" className="text-forest underline" target="_blank">{ch}</Link>,
            })}</span>
          </label>
          <label className="flex cursor-pointer items-start gap-3 text-sm text-ink-2">
            <input type="checkbox" className="mt-0.5 h-5 w-5 shrink-0 accent-[#2E4A3A]" checked={checks.autorenew} onChange={(e) => setChecks((c) => ({ ...c, autorenew: e.target.checked }))} />
            <span>{t("cbAutorenew", { price })}</span>
          </label>
          <label className="flex cursor-pointer items-start gap-3 text-sm text-ink-2">
            <input type="checkbox" className="mt-0.5 h-5 w-5 shrink-0 accent-[#2E4A3A]" checked={checks.digital} onChange={(e) => setChecks((c) => ({ ...c, digital: e.target.checked }))} />
            <span>{t("cbDigital")}</span>
          </label>
          <button onClick={subscribe} disabled={!allChecked || busy === "pay"} data-testid="pro-subscribe"
            className="mt-2 inline-flex min-h-[48px] w-full items-center justify-center gap-2 rounded-full bg-forest px-5 py-3.5 text-[15px] font-bold text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
            style={{ background: "var(--forest,#2E4A3A)" }}>
            {busy === "pay" && <Loader2 className="h-4 w-4 animate-spin" />} {t("subscribe", { price })}
          </button>
          {!allChecked && <p className="text-center text-[12px] text-ink-3">{t("checkAll")}</p>}
        </div>
      )}
      <p className="mt-4 flex items-center justify-center gap-1.5 text-center text-[12px] text-ink-3">
        <ShieldCheck size={14} /> {t("sellerNote")}
      </p>
    </div>
  );

  return (
    <div className="mx-auto max-w-[1120px] px-5 pb-28 pt-10 lg:px-8 lg:pb-16">
      <Link href="/" className="text-[13px] font-semibold text-ink-2 hover:text-ink">← monadruk</Link>

      {/* ── Hero + оформлення ───────────────────────────────────────────── */}
      <div className="mt-6 grid gap-8 lg:grid-cols-[1fr_440px] lg:items-start">
        <div>
          <div className="inline-flex items-center gap-1.5 rounded-full border border-line bg-paper px-3 py-1 text-[12px] font-bold uppercase tracking-wide text-forest">
            <ShieldCheck size={14} /> Monadruk Pro
          </div>
          <h1 className="mt-3 font-serif text-[clamp(30px,4.4vw,46px)] leading-[1.08] text-ink">{t("title")}</h1>
          <p className="mt-4 max-w-[560px] text-[16px] leading-relaxed text-ink-2">{t("subtitle")}</p>

          <dl className="mt-7 grid max-w-[560px] grid-cols-1 gap-2 sm:grid-cols-3 sm:gap-3">
            {[
              { v: <InfinityIcon size={26} strokeWidth={2.2} aria-label="∞" />, l: t("stat1") },
              { v: t("stat2v", { n: breakEven }), l: t("stat2") },
              { v: <MousePointerClick size={24} />, l: t("stat3") },
            ].map((s, i) => (
              <div key={i} className="flex items-center gap-3 rounded-[18px] border border-line bg-paper px-3.5 py-2.5 sm:block sm:px-3 sm:py-3.5">
                <dt className="flex h-8 w-10 shrink-0 items-center font-serif text-[22px] text-forest sm:w-auto">{s.v}</dt>
                <dd className="text-[13px] leading-snug text-ink-2 sm:mt-1 sm:text-[12.5px]">{s.l}</dd>
              </div>
            ))}
          </dl>

          <div className="mt-7 flex flex-wrap items-center gap-3">
            {!isActive && salesOpen && (
              <button onClick={goCheckout} className="inline-flex min-h-[48px] items-center gap-2 rounded-full bg-forest px-6 text-[15px] font-bold text-white hover:opacity-90 lg:hidden" style={{ background: "var(--forest,#2E4A3A)" }}>
                <Sparkles size={16} /> {t("ctaPrimary")}
              </button>
            )}
            <Link href="/create" className="inline-flex min-h-[48px] items-center rounded-full border border-line bg-white px-6 text-[15px] font-semibold text-ink hover:bg-paper">
              {t("ctaTry")}
            </Link>
          </div>
          <p className="mt-2 text-[13px] text-ink-3">{t("tryNote")}</p>
        </div>

        <div id="checkout" ref={checkoutRef} className="scroll-mt-24 rounded-[24px] border border-line bg-paper p-6 shadow-[0_18px_40px_rgba(30,40,32,0.08)] sm:p-7 lg:sticky lg:top-24">
          {showStatus ? statusCard : salesOpen ? checkoutCard : soonCard}
          {msg && <div className="mt-4 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">{msg}</div>}
        </div>
      </div>

      {/* ── Калькулятор вигоди ──────────────────────────────────────────── */}
      <section className="mt-16 rounded-[24px] border border-line bg-paper p-6 sm:p-8" aria-labelledby="pro-calc">
        <h2 id="pro-calc" className="font-serif text-[28px] text-ink">{t("calcTitle")}</h2>
        <p className="mt-1 text-[14px] text-ink-2">{t("calcLead")}</p>
        <label className="mt-6 block">
          <span className="flex items-baseline justify-between gap-3 text-sm font-semibold text-ink">
            {t("calcLabel")}
            <span className="font-serif text-2xl text-forest tabular-nums" data-testid="pro-calc-n">{t("calcFiles", { n: filesPerMonth })}</span>
          </span>
          <input type="range" min={1} max={60} step={1} value={filesPerMonth}
            onChange={(e) => setFilesPerMonth(Number(e.target.value))}
            className="mt-3 w-full accent-[#2E4A3A]" aria-label={t("calcLabel")} data-testid="pro-calc-range" />
        </label>
        <div className="mt-5 grid gap-3 sm:grid-cols-2">
          <div className="rounded-[18px] border border-line bg-white p-4">
            <div className="text-[12px] font-semibold uppercase tracking-wide text-ink-3">{t("calcPerFile", { price: filePriceLabel })}</div>
            <div className="mt-1 font-serif text-3xl tabular-nums text-ink">{ccy === "USD" ? "≈ " : ""}{fmtPrice(ccy, perFileTotal, locale)}</div>
          </div>
          <div className={`rounded-[18px] border p-4 ${saving > 0 ? "border-forest bg-[rgba(46,74,58,0.06)]" : "border-line bg-white"}`}>
            <div className="text-[12px] font-semibold uppercase tracking-wide text-ink-3">{t("calcPro")}</div>
            <div className="mt-1 font-serif text-3xl tabular-nums text-ink">{price}</div>
          </div>
        </div>
        <p className="mt-4 text-[15px] font-semibold text-ink" data-testid="pro-calc-verdict">
          {saving > 0
            ? t("calcSave", { amount: `${ccy === "USD" ? "≈ " : ""}${fmtPrice(ccy, saving, locale)}` })
            : t("calcNotYet", { n: breakEven })}
        </p>
        <p className="mt-1 text-[13px] text-ink-3">{t("calcLicense")}</p>
        {ccy === "USD" && <p className="mt-1 text-[12px] text-ink-3">{t("calcApprox")}</p>}
      </section>

      {/* ── Порівняння ──────────────────────────────────────────────────── */}
      <section className="mt-12" aria-labelledby="pro-cmp">
        <h2 id="pro-cmp" className="font-serif text-[28px] text-ink">{t("cmpTitle")}</h2>
        <div className="mt-5 overflow-hidden rounded-[20px] border border-line">
          <table className="w-full table-fixed border-collapse text-left text-[12.5px] sm:text-[14px]">
            <colgroup><col className="w-[31%]" /><col /><col /><col /></colgroup>
            <thead>
              <tr className="bg-paper">
                <th scope="col" className="px-2.5 py-3 font-semibold text-ink-3 sm:px-4"><span className="sr-only">{t("cmpFeature")}</span></th>
                <th scope="col" className="px-2 py-3 align-bottom font-semibold text-ink sm:px-4">{t("cmpFree")}</th>
                <th scope="col" className="px-2 py-3 align-bottom font-semibold text-ink sm:px-4">{t("cmpFile", { price: fmtPrice("UAH", filePrice.UAH, locale) })}</th>
                <th scope="col" className="bg-[rgba(46,74,58,0.08)] px-2 py-3 align-bottom font-bold text-forest sm:px-4">Pro<span className="block text-[11.5px] font-semibold sm:inline sm:text-[14px]"><span className="hidden sm:inline"> · </span>{price}{t("perMonthShort")}</span></th>
              </tr>
            </thead>
            <tbody>
              {cmpRows.map((r) => (
                <tr key={r.label} className="border-t border-line">
                  <th scope="row" className="px-2.5 py-3 font-medium text-ink sm:px-4">{r.label}</th>
                  {r.cells.map((c, i) => (
                    <td key={i} className={`px-2 py-3 text-ink-2 sm:px-4 ${i === 2 ? "bg-[rgba(46,74,58,0.05)] font-semibold text-ink" : ""}`}>
                      {c === true ? <CheckCircle2 size={18} className="text-forest" aria-label="✓" />
                        : c === false ? <Minus size={18} className="text-ink-3" aria-label="—" />
                        : c}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── Для кого ────────────────────────────────────────────────────── */}
      <section className="mt-12" aria-labelledby="pro-who">
        <h2 id="pro-who" className="font-serif text-[28px] text-ink">{t("whoTitle")}</h2>
        <div className="mt-5 grid gap-3 md:grid-cols-3">
          {audience.map(({ icon: Icon, title, text }) => (
            <div key={title} className="rounded-[20px] border border-line bg-paper p-5">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[rgba(46,74,58,0.10)] text-forest"><Icon size={19} /></div>
              <h3 className="mt-3 text-[16px] font-semibold text-ink">{title}</h3>
              <p className="mt-1 text-[14px] leading-relaxed text-ink-2">{text}</p>
            </div>
          ))}
        </div>
        <div className="mt-5 flex flex-wrap gap-2">
          {[
            { href: "/create", label: t("prodMaps") },
            { href: "/keychains", label: t("prodKeychains") },
            { href: "/panno", label: t("prodPanno") },
            { href: "/mountains", label: t("prodMountains") },
          ].map((p) => (
            <Link key={p.href} href={p.href} className="inline-flex min-h-[40px] items-center rounded-full border border-line bg-white px-4 text-[13.5px] font-semibold text-ink-2 hover:text-ink">
              {p.label} →
            </Link>
          ))}
        </div>
      </section>

      {/* ── FAQ ─────────────────────────────────────────────────────────── */}
      <section className="mt-12" aria-labelledby="pro-faq">
        <h2 id="pro-faq" className="font-serif text-[28px] text-ink">{t("faqTitle")}</h2>
        <div className="mt-4 divide-y divide-line rounded-[20px] border border-line bg-paper">
          {faq.map((f) => (
            <details key={f.q} className="group px-5 py-1">
              <summary className="flex min-h-[52px] cursor-pointer list-none items-center justify-between gap-3 text-[15px] font-semibold text-ink">
                {f.q}
                <ChevronDown size={18} className="shrink-0 text-ink-3 transition group-open:rotate-180" />
              </summary>
              <p className="pb-4 text-[14px] leading-relaxed text-ink-2">{f.a}</p>
            </details>
          ))}
        </div>
      </section>

      <div className="mt-8 space-y-2 text-[13px] text-ink-3">
        <p>{t("osmNote")}</p>
        <p><Link href="/pro-terms" className="text-forest underline">{t("termsLink")}</Link></p>
      </div>

      {/* Липка кнопка на мобільному: веде до картки оформлення. */}
      {!isActive && salesOpen && !checkoutVisible && scrolled && (
        <div className="fixed inset-x-0 bottom-0 z-40 border-t border-line bg-[rgba(244,239,228,0.96)] px-4 pb-[calc(12px+env(safe-area-inset-bottom))] pt-3 backdrop-blur lg:hidden" data-testid="pro-sticky">
          <button onClick={goCheckout} className="inline-flex min-h-[48px] w-full items-center justify-center gap-2 rounded-full bg-forest text-[15px] font-bold text-white" style={{ background: "var(--forest,#2E4A3A)" }}>
            <Sparkles size={16} /> {t("ctaSticky", { price })}
          </button>
        </div>
      )}

      {pay && (
        <form ref={formRef} method="POST" action={pay.action_url} acceptCharset="utf-8" className="hidden">
          <input type="hidden" name="data" value={pay.data} />
          <input type="hidden" name="signature" value={pay.signature} />
        </form>
      )}
    </div>
  );
}
