"use client";

/**
 * Підписка Monadruk Pro: оформлення (LiqPay regular payment) і керування (статус,
 * скасування в один клік). Юридично важливе:
 *  • три ОКРЕМІ непоставлені за замовчуванням позначки (умови, автопродовження,
 *    негайний доступ = втрата 14-денної відмови) — без усіх трьох кнопка неактивна,
 *    бекенд теж відхиляє (400); тексти позначок ідуть у запис згоди;
 *  • ціна, періодичність і дата наступного списання видно ДО оплати;
 *  • скасування — кнопка тут же, без дзвінків/листів.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocale, useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import { CheckCircle2, Loader2, ShieldCheck } from "lucide-react";
import { useAuth } from "@/components/AuthProvider";
import { SUB_PRICE } from "@/lib/legal/subscription";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";
type Ccy = "UAH" | "USD";

interface SubView {
  status: string; active?: boolean; amount?: number; currency?: Ccy;
  paid_until?: string | null; renews_at?: string | null; cancelled_at?: string | null;
  payments?: { amount: number; currency: string; ts: string }[];
}

const LOCALE_TAG: Record<string, string> = { uk: "uk-UA", en: "en-US", de: "de-DE", es: "es-ES", fr: "fr-FR", pl: "pl-PL" };

function fmtPrice(ccy: Ccy, amount: number, locale: string) {
  return ccy === "UAH"
    ? `${new Intl.NumberFormat(LOCALE_TAG[locale] ?? "uk-UA").format(amount)} ₴`
    : new Intl.NumberFormat(LOCALE_TAG[locale] ?? "en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(amount);
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
  const [sub, setSub] = useState<SubView | null>(null);
  const [checks, setChecks] = useState({ terms: false, autorenew: false, digital: false });
  const [busy, setBusy] = useState<"pay" | "cancel" | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [pay, setPay] = useState<{ action_url: string; data: string; signature: string } | null>(null);
  const formRef = useRef<HTMLFormElement>(null);
  useEffect(() => { if (pay) formRef.current?.submit(); }, [pay]);

  // Валюта за країною (Cloudflare): Україна → гривня, решта світу → долар.
  useEffect(() => {
    fetch(`${API_BASE}/api/subscription/plans`).then((r) => (r.ok ? r.json() : null)).then((j) => {
      if (!j) return;
      if (j.plans) setPlans({ UAH: Number(j.plans.UAH) || SUB_PRICE.UAH, USD: Number(j.plans.USD) || SUB_PRICE.USD });
      if (j.suggested === "UAH" || j.suggested === "USD") setCcy(j.suggested);
    }).catch(() => {});
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

  const features = [t("f1"), t("f2"), t("f3"), t("f4")];
  const showStatus = sub && (sub.active || sub.status === "pending");

  return (
    <div className="mx-auto max-w-[760px] px-5 py-12 lg:px-8">
      <Link href="/" className="text-[13px] font-semibold text-ink-2 hover:text-ink">← monadruk</Link>
      <div className="mt-6 inline-flex items-center gap-1.5 rounded-full border border-line bg-paper px-3 py-1 text-[12px] font-bold uppercase tracking-wide text-forest">
        <ShieldCheck size={14} /> Monadruk Pro
      </div>
      <h1 className="mt-3 font-serif text-3xl text-ink sm:text-4xl">{t("title")}</h1>
      <p className="mt-3 text-[15px] text-ink-2">{t("subtitle")}</p>

      <ul className="mt-6 space-y-2.5">
        {features.map((f) => (
          <li key={f} className="flex items-start gap-2.5 text-[15px] text-ink-2">
            <CheckCircle2 size={18} className="mt-0.5 shrink-0 text-forest" /><span>{f}</span>
          </li>
        ))}
      </ul>

      <div className="mt-8 rounded-[24px] border border-line bg-paper p-6 sm:p-8">
        {showStatus ? (
          <div>
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
              <Link href="/create" className="inline-flex items-center rounded-full bg-forest px-5 py-2.5 text-sm font-bold text-white hover:opacity-90" style={{ background: "var(--forest,#2E4A3A)" }}>
                {t("goCreate")}
              </Link>
              {sub!.status === "active" && (
                <button onClick={cancel} disabled={busy === "cancel"}
                  className="inline-flex items-center gap-2 rounded-full border border-line bg-white px-5 py-2.5 text-sm font-semibold text-ink hover:bg-paper disabled:opacity-60">
                  {busy === "cancel" && <Loader2 className="h-4 w-4 animate-spin" />} {t("cancel")}
                </button>
              )}
            </div>
          </div>
        ) : (
          <div>
            <div role="radiogroup" aria-label={t("currency")} className="inline-flex rounded-full border border-line bg-white p-1">
              {(["UAH", "USD"] as Ccy[]).map((c) => (
                <button key={c} role="radio" aria-checked={ccy === c} onClick={() => setCcy(c)}
                  className={`rounded-full px-4 py-1.5 text-sm font-semibold ${ccy === c ? "bg-ink text-white" : "text-ink-2"}`}>
                  {c === "UAH" ? t("currencyUah") : t("currencyUsd")}
                </button>
              ))}
            </div>
            <div className="mt-5 flex items-baseline gap-2">
              <span className="font-serif text-4xl text-ink">{price}</span>
              <span className="text-ink-2">{t("perMonth")}</span>
            </div>
            <p className="mt-1 text-sm text-ink-2">{t("priceNote", { date: nextCharge })}</p>

            {!user ? (
              <div className="mt-6">
                <p className="text-sm text-ink-2">{t("loginFirst")}</p>
                <button onClick={signIn} disabled={loading}
                  className="mt-3 inline-flex items-center rounded-full bg-forest px-5 py-3 text-sm font-bold text-white hover:opacity-90" style={{ background: "var(--forest,#2E4A3A)" }}>
                  {t("login")}
                </button>
              </div>
            ) : (
              <div className="mt-6 space-y-3">
                <label className="flex items-start gap-3 text-sm text-ink-2">
                  <input type="checkbox" className="mt-1 h-4 w-4 shrink-0" checked={checks.terms} onChange={(e) => setChecks((c) => ({ ...c, terms: e.target.checked }))} />
                  <span>{t.rich("cbTerms", {
                    terms: (ch) => <Link href="/pro-terms" className="text-forest underline" target="_blank">{ch}</Link>,
                    offer: (ch) => <Link href="/offer" className="text-forest underline" target="_blank">{ch}</Link>,
                    privacy: (ch) => <Link href="/privacy" className="text-forest underline" target="_blank">{ch}</Link>,
                  })}</span>
                </label>
                <label className="flex items-start gap-3 text-sm text-ink-2">
                  <input type="checkbox" className="mt-1 h-4 w-4 shrink-0" checked={checks.autorenew} onChange={(e) => setChecks((c) => ({ ...c, autorenew: e.target.checked }))} />
                  <span>{t("cbAutorenew", { price })}</span>
                </label>
                <label className="flex items-start gap-3 text-sm text-ink-2">
                  <input type="checkbox" className="mt-1 h-4 w-4 shrink-0" checked={checks.digital} onChange={(e) => setChecks((c) => ({ ...c, digital: e.target.checked }))} />
                  <span>{t("cbDigital")}</span>
                </label>
                <button onClick={subscribe} disabled={!allChecked || busy === "pay"}
                  className="mt-2 inline-flex w-full items-center justify-center gap-2 rounded-full bg-forest px-5 py-3.5 text-[15px] font-bold text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40 sm:w-auto"
                  style={{ background: "var(--forest,#2E4A3A)" }}>
                  {busy === "pay" && <Loader2 className="h-4 w-4 animate-spin" />} {t("subscribe", { price })}
                </button>
              </div>
            )}
          </div>
        )}
        {msg && <div className="mt-4 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">{msg}</div>}
      </div>

      <div className="mt-6 space-y-2 text-[13px] text-ink-3">
        <p>{t("osmNote")}</p>
        <p>{t("sellerNote")} <Link href="/pro-terms" className="text-forest underline">{t("termsLink")}</Link></p>
      </div>

      {pay && (
        <form ref={formRef} method="POST" action={pay.action_url} acceptCharset="utf-8" className="hidden">
          <input type="hidden" name="data" value={pay.data} />
          <input type="hidden" name="signature" value={pay.signature} />
        </form>
      )}
    </div>
  );
}
