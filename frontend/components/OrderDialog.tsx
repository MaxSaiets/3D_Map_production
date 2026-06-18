"use client";

import { useState, useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { X, Loader2, CheckCircle2, Truck, Package, ShieldCheck, Lock, PhoneCall, Wallet } from "lucide-react";
import { useTranslations } from "next-intl";
import { capturePreviewImages } from "@/lib/capturePreview";
import { useAuth } from "@/components/AuthProvider";
import { NovaPoshtaPicker } from "@/components/NovaPoshtaPicker";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

type Delivery = "nova" | "ukr" | "pickup" | "novapost_eu" | "meest";
type Region = "ua" | "eu";

/** Рідні назви — впізнавані і для місцевих, і для українців за кордоном; не потребують перекладу. */
const EU_COUNTRIES = [
  "Polska", "Deutschland", "Česko", "Slovensko", "Österreich", "Italia", "España",
  "France", "Nederland", "België", "Lietuva", "Latvija", "Eesti", "Portugal", "România",
];

export interface OrderSummary {
  city?: string;
  district?: string;
  label?: string;   // keychain text
  size?: string;
}

/**
 * Customer order form. Captures screenshots of the live 3D preview and posts
 * the order to the backend, which forwards everything to the Telegram CRM.
 */
export function OrderDialog({
  open,
  onClose,
  taskId,
  productType,
  summary,
  priceText,
  modelPending = false,
}: {
  open: boolean;
  onClose: () => void;
  taskId: string | null;
  productType: "map" | "keychain";
  summary: OrderSummary;
  /** Жива орієнтовна ціна з /api/quote; без неї — статичний i18n-fallback. */
  priceText?: string;
  /** Модель ще генерується (order-now) — показуємо заспокійливу примітку. */
  modelPending?: boolean;
}) {
  const t = useTranslations("order");
  const [name, setName] = useState("");
  const [phone, setPhone] = useState("");
  const [email, setEmail] = useState("");
  const [region, setRegion] = useState<Region>("ua");
  const [euCountry, setEuCountry] = useState("");
  const [delivery, setDelivery] = useState<Delivery>("nova");
  const [city, setCity] = useState("");
  const [branch, setBranch] = useState("");
  const [address, setAddress] = useState("");
  const [comment, setComment] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [orderNumber, setOrderNumber] = useState<string | null>(null);
  // payment: { provider?: "liqpay", action_url?, data?, signature?, url?, label? }
  const [payment, setPayment] = useState<any>(null);
  const { getIdToken } = useAuth();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const firstInputRef = useRef<HTMLInputElement | null>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  // Скидаємо екран успіху при КОЖНОМУ відкритті — інакше після одного замовлення
  // повторне відкриття (інша модель) показувало СТАРИЙ «#123 прийнято» замість форми,
  // і друге замовлення неможливо було оформити без перезавантаження. (контакт лишаємо.)
  useEffect(() => {
    if (open) {
      setOrderNumber(null); setPayment(null); setError(null); setSending(false);
      // Воронка: користувач відкрив форму замовлення (передостанній крок).
      import("@/lib/analytics").then((m) => m.trackFunnel("order_open")).catch(() => {});
    }
  }, [open]);

  // Escape closes the dialog; only active while open.
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  // Focus management: move focus into the dialog on open, restore on close (best-effort).
  useEffect(() => {
    if (!open) return;
    previouslyFocusedRef.current = (document.activeElement as HTMLElement) || null;
    // Defer so the portal content is mounted before focusing.
    const id = window.setTimeout(() => {
      if (firstInputRef.current) firstInputRef.current.focus();
      else if (containerRef.current) containerRef.current.focus();
    }, 0);
    return () => {
      window.clearTimeout(id);
      const prev = previouslyFocusedRef.current;
      if (prev && typeof prev.focus === "function") {
        try { prev.focus(); } catch { /* ignore */ }
      }
    };
  }, [open]);

  // Focus trap: keep Tab focus within the dialog, wrapping at the edges.
  const handleTrapKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "Tab") return;
    const container = containerRef.current;
    if (!container) return;
    const focusable = Array.from(
      container.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
      )
    ).filter((el) => el.offsetParent !== null || el === document.activeElement);
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement as HTMLElement | null;
    if (e.shiftKey) {
      if (active === first || !container.contains(active)) {
        e.preventDefault();
        last.focus();
      }
    } else {
      if (active === last || !container.contains(active)) {
        e.preventDefault();
        first.focus();
      }
    }
  };

  if (!open) return null;

  const submit = async () => {
    if (!name.trim()) { setError(t("errName")); return; }
    if (!phone.trim()) { setError(t("errPhone")); return; }
    // Базова перевірка телефону: лишаємо тільки цифри, очікуємо ≥10 (UA +380 = 12).
    if (phone.replace(/\D/g, "").length < 10) { setError(t("errPhoneFormat")); return; }
    if (email.trim() && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) { setError(t("errEmail")); return; }
    if (region === "eu") {
      if (!euCountry) { setError(t("errEuCountry")); return; }
      if (!city.trim()) { setError(t("errCity")); return; }
      if (delivery === "novapost_eu" && !branch.trim()) { setError(t("errBranchEu")); return; }
      if (delivery === "meest" && !address.trim()) { setError(t("errAddressEu")); return; }
    } else if (delivery !== "pickup") {
      if (!city.trim()) { setError(t("errCity")); return; }
      if (!branch.trim()) { setError(delivery === "nova" ? t("errNova") : t("errUkr")); return; }
      // Укрпошта потребує і місто+індекс, і вулицю/будинок (інакше недоставне).
      if (delivery === "ukr" && !address.trim()) { setError(t("errUkrAddress")); return; }
    }
    setError(null);
    setSending(true);
    try {
      // Захоплюємо превʼю (SVG-дизайнер брелка + 3D-canvas) — оператор у Telegram
      // побачить ТОЧНО що замовив клієнт (текст, розташування) ще до друку.
      const screenshots = await capturePreviewImages();
      // Якщо клієнт залогінений — замовлення привʼяжеться до акаунта (видно в кабінеті).
      let token: string | null = null;
      try { token = await getIdToken(); } catch { /* ignore */ }
      const res = await fetch(`${API_BASE}/api/order`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
        body: JSON.stringify({
          name, phone, email, product_type: productType, task_id: taskId,
          delivery_method: delivery,
          delivery_country: region === "ua" ? "Україна" : euCountry,
          delivery_city: city,
          delivery_branch: branch, delivery_address: address, comment,
          est_price: priceText || (productType === "keychain" ? t("estPriceKeychain") : t("estPriceMap")),
          summary, screenshots,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setOrderNumber(String(data.order_number));
      setPayment(data.payment || null);
      // Google Ads / GA4 conversion — головна ціль реклами (надіслане замовлення = лід).
      try {
        const { trackConversion, trackFunnel } = await import("@/lib/analytics");
        trackFunnel("order_submit"); // останній крок воронки — замовлення надіслане
        // Валюту беремо з ТОГО САМОГО рядка, що й число (priceText), а не з region —
        // інакше EUR-замовлення слало б UAH-суму як EUR (×10 інфляція конверсії).
        const raw = String(priceText || "");
        const value = raw ? Number(raw.replace(/[^\d]/g, "")) || undefined : undefined;
        trackConversion("order", {
          value,
          currency: raw.includes("€") ? "EUR" : "UAH",
          transactionId: String(data.order_number),
          props: { product: productType, delivery },
        });
      } catch { /* ignore */ }
    } catch (e: any) {
      setError(t("sendFail"));
    } finally {
      setSending(false);
    }
  };

  const fieldCls = "w-full rounded-2xl border border-[var(--surface-border)] bg-white px-4 py-3 text-sm text-[var(--text-primary)] outline-none transition focus:border-[rgba(11,92,87,0.4)]";

  // Portal to <body>: ancestors with backdrop-filter/transform become the
  // containing block for position:fixed, clipping the dialog inside side panels.
  const dialog = (
    <div
      className="fixed inset-0 z-[80] flex items-end justify-center bg-black/40 p-0 backdrop-blur-sm sm:items-center sm:p-4"
      onClick={onClose}
    >
      <div
        ref={containerRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-labelledby="order-dialog-title"
        className="max-h-[92dvh] w-full max-w-[460px] overflow-y-auto rounded-t-[28px] border border-[var(--surface-border)] bg-[var(--surface-panel,#fff)] p-5 shadow-[0_30px_80px_rgba(15,23,42,0.35)] sm:rounded-[28px]"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleTrapKeyDown}
      >
        {orderNumber ? (
          <div className="py-6 text-center">
            <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-emerald-100 text-emerald-700">
              <CheckCircle2 size={30} />
            </div>
            <h3 id="order-dialog-title" className="font-serif text-2xl text-[var(--text-primary)]">{t("acceptedTitle")}</h3>
            <p className="mt-2 text-sm text-[var(--text-secondary)]">
              {t("orderNo")} <b className="text-[var(--text-primary)]">#{orderNumber}</b>.<br />
              {t("acceptedText")}
            </p>
            {payment && (() => {
              const payLabel = `${t("payNow")} · ${priceText || (productType === "keychain" ? t("estPriceKeychain") : t("estPriceMap"))}`;
              const btnCls = "mt-5 inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-full bg-[var(--bronze,#8E6B3D)] px-5 py-3 text-[15px] font-extrabold text-white shadow-[0_16px_34px_rgba(142,107,61,0.32)] transition hover:opacity-90";
              if (payment.provider === "liqpay" && payment.data && payment.signature) {
                // Форма-POST на LiqPay checkout (стандартна кнопка LiqPay), у новій вкладці.
                return (
                  <>
                    <form action={payment.action_url} method="POST" acceptCharset="utf-8" target="_blank" className="mt-0">
                      <input type="hidden" name="data" value={payment.data} />
                      <input type="hidden" name="signature" value={payment.signature} />
                      <button type="submit" className={btnCls}>{payLabel}</button>
                    </form>
                    <p className="mt-2 text-[11px] leading-4 text-[var(--text-secondary)]">{t("payLater")}</p>
                  </>
                );
              }
              if (payment.url) {
                return (
                  <>
                    <a href={payment.url} target="_blank" rel="noopener noreferrer" className={btnCls}>{payLabel}</a>
                    <p className="mt-2 text-[11px] leading-4 text-[var(--text-secondary)]">{t("payLater")}</p>
                  </>
                );
              }
              return null;
            })()}
            <button onClick={onClose} className="mt-5 inline-flex min-h-12 w-full items-center justify-center rounded-full bg-[var(--accent-strong)] px-5 py-3 text-sm font-semibold text-white">
              {t("doneBtn")}
            </button>
          </div>
        ) : (
          <>
            <div className="mb-4 flex items-start justify-between">
              <div className="flex items-center gap-2">
                <span className="flex h-9 w-9 items-center justify-center rounded-full bg-[var(--accent-strong)] text-white"><Package size={18} /></span>
                <div>
                  <h3 id="order-dialog-title" className="font-serif text-xl text-[var(--text-primary)]">{t("title")}</h3>
                  <p className="text-[11px] text-[var(--text-secondary)]">{productType === "keychain" ? t("prodKeychain") : t("prodMap")}{summary.size ? ` · ${summary.size}` : ""}</p>
                </div>
              </div>
              <button onClick={onClose} aria-label={t("aria.close")} className="rounded-lg p-1 text-[var(--text-secondary)] hover:bg-black/5"><X size={20} /></button>
            </div>

            <div className="space-y-3">
              {modelPending && (
                <div className="flex items-center gap-2 rounded-2xl border border-[rgba(11,92,87,0.2)] bg-[rgba(15,118,110,0.07)] px-3 py-2 text-[12px] leading-4 text-[var(--text-primary)]">
                  <span aria-hidden>🛠</span>
                  <span>{t("modelPending")}</span>
                </div>
              )}
              <input ref={firstInputRef} className={fieldCls} placeholder={t("phName")} aria-label={t("phName")} value={name} onChange={(e) => setName(e.target.value)} />
              <input className={fieldCls} placeholder={t("phPhone")} aria-label={t("phPhone")} value={phone} onChange={(e) => setPhone(e.target.value)} inputMode="tel" />
              {/* Email необовʼязковий — для підтвердження замовлення на пошту. */}
              <input className={fieldCls} placeholder={t("phEmail")} aria-label={t("phEmail")} value={email} onChange={(e) => setEmail(e.target.value)} inputMode="email" type="email" autoComplete="email" />

              <div className="space-y-1.5">
                <p className="px-1 text-[11px] font-semibold uppercase tracking-wide text-[var(--text-secondary)]">{t("deliveryHeading")}</p>
                <div role="radiogroup" aria-label={t("aria.region")} className="flex items-center gap-2 rounded-2xl border border-[var(--surface-border)] bg-white/70 p-1 text-xs">
                  {([["ua", t("regionUa")], ["eu", t("regionEu")]] as [Region, string][]).map(([k, lbl]) => (
                    <button key={k} type="button" role="radio" aria-checked={region === k}
                      onClick={() => { setRegion(k); setDelivery(k === "ua" ? "nova" : "novapost_eu"); }}
                      className={`flex-1 min-h-11 rounded-xl px-2 py-2 text-sm font-semibold transition ${region === k ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)]"}`}>
                      {lbl}
                    </button>
                  ))}
                </div>

                <div role="radiogroup" aria-label={t("aria.method")} className="flex items-center gap-2 rounded-2xl border border-[var(--surface-border)] bg-white/70 p-1 text-xs">
                  {(region === "ua"
                    ? ([["nova", t("nova")], ["ukr", t("ukr")]] as [Delivery, string][])
                    : ([["novapost_eu", "Nova Post (EU)"], ["meest", "Meest"]] as [Delivery, string][])
                  ).map(([k, lbl]) => (
                    <button key={k} type="button" role="radio" aria-checked={delivery === k} onClick={() => setDelivery(k)}
                      className={`flex-1 min-h-11 rounded-xl px-2 py-2 text-sm font-semibold transition ${delivery === k ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)]"}`}>
                      {lbl}
                    </button>
                  ))}
                </div>
                <p className="px-1 text-[11px] leading-4 text-[var(--text-secondary)]">{t("deliveryHint")}</p>
                {/* Орієнтовна вартість доставки за тарифом перевізника (2025-12):
                    НП ~70₴, Укрпошта ~45₴, ЄС ~€8 — знімає страх «приховані витрати». */}
                <p className="px-1 text-[11px] font-semibold leading-4 text-[var(--accent-strong)]">
                  {delivery === "pickup"
                    ? t("costPickup")
                    : delivery === "nova"
                      ? t("costNova")
                      : delivery === "ukr"
                        ? t("costUkr")
                        : t("costEu")}
                </p>
              </div>

              {region === "eu" && (
                <select className={fieldCls} aria-label={t("phCountry")} value={euCountry} onChange={(e) => setEuCountry(e.target.value)}>
                  <option value="">{t("phCountry")}</option>
                  {EU_COUNTRIES.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              )}

              {delivery !== "pickup" && (
                region === "ua" && delivery === "nova" ? (
                  // Нова Пошта: пошук міста + відділення через API (фолбек на ручне
                  // введення, якщо ключ NOVA_POSHTA_API_KEY не налаштовано на сервері).
                  <NovaPoshtaPicker city={city} branch={branch} setCity={setCity} setBranch={setBranch} inputCls={fieldCls} />
                ) : (
                <>
                  <input className={fieldCls} placeholder={t("phCity")} aria-label={t("phCity")} value={city} onChange={(e) => setCity(e.target.value)} />
                  {region === "ua" ? (
                    <>
                      <input className={fieldCls} placeholder={delivery === "nova" ? t("phNova") : t("phUkr")} aria-label={delivery === "nova" ? t("phNova") : t("phUkr")} value={branch} onChange={(e) => setBranch(e.target.value)} />
                      {delivery === "ukr" && (
                        <input className={fieldCls} placeholder={t("phAddress")} aria-label={t("phAddress")} value={address} onChange={(e) => setAddress(e.target.value)} />
                      )}
                    </>
                  ) : delivery === "novapost_eu" ? (
                    <input className={fieldCls} placeholder={t("phBranchEu")} aria-label={t("phBranchEu")} value={branch} onChange={(e) => setBranch(e.target.value)} />
                  ) : (
                    <input className={fieldCls} placeholder={t("phAddressEu")} aria-label={t("phAddressEu")} value={address} onChange={(e) => setAddress(e.target.value)} />
                  )}
                </>
                )
              )}

              <textarea className={`${fieldCls} min-h-[64px] resize-none`} placeholder={t("phComment")} aria-label={t("phComment")} value={comment} onChange={(e) => setComment(e.target.value)} />

              {/* Ціна переїхала у sticky-футер (завжди на видноті, без скролу до кінця);
                  тут лишилось ЩО входить у ціну — знімає страх «а доставка окремо?». */}
              <p className="rounded-2xl border border-[rgba(176,141,87,0.35)] bg-[rgba(176,141,87,0.16)] px-4 py-3 text-[12px] leading-5 text-[var(--text-secondary)]">{t("priceIncludes")}</p>

              <div className="flex items-start gap-2 rounded-2xl bg-[rgba(46,74,58,0.06)] px-3 py-2.5 text-[12px] leading-5 text-[var(--text-secondary)]">
                <Truck size={14} className="mt-0.5 shrink-0 text-[var(--accent-strong)]" />
                <span>
                  {/* UX: чіткі строки знімають головний страх «коли отримаю?» */}
                  <b className="text-[var(--text-primary)]">{t("leadTime")}</b>
                  {" — "}
                  {t("paymentNote")}
                </span>
              </div>

              {/* Сигнали довіри — знижують відмову на останньому кроці воронки:
                  замовлення без передоплати, оператор підтверджує, дані захищені. */}
              <ul className="grid gap-1.5 rounded-2xl border border-[rgba(11,92,87,0.16)] bg-[rgba(15,118,110,0.05)] px-3 py-2.5 text-[12px] leading-5 text-[var(--text-secondary)]">
                <li className="flex items-center gap-2"><ShieldCheck size={13} className="shrink-0 text-[var(--accent-strong)]" /><span><b className="text-[var(--text-primary)]">{t("trustGuarantee")}</b> — {t("trustGuaranteeDesc")}</span></li>
                <li className="flex items-center gap-2"><Wallet size={13} className="shrink-0 text-[var(--accent-strong)]" /><span><b className="text-[var(--text-primary)]">{t("trustNoPrepay")}</b> — {t("trustNoPrepayDesc")}</span></li>
                <li className="flex items-center gap-2"><PhoneCall size={13} className="shrink-0 text-[var(--accent-strong)]" /><span>{t("trustOperator")}</span></li>
                <li className="flex items-center gap-2"><Lock size={13} className="shrink-0 text-[var(--accent-strong)]" /><span>{t("trustSecure")}</span></li>
              </ul>

              <p className="flex items-center justify-center gap-1.5 text-center text-[12px] leading-5 text-[var(--text-secondary)]">
                <ShieldCheck size={12} className="shrink-0 text-[var(--accent-strong)]" />
                {t("submitReassure")}
              </p>

              {/* Sticky-футер: ЦІНА + CTA завжди на видноті — не треба скролити крізь
                  усю форму до кнопки. Лишається приклеєним до низу скрол-панелі. */}
              <div className="sticky bottom-0 -mx-5 -mb-5 border-t border-[var(--surface-border)] bg-[var(--surface-panel,#fff)] px-5 pb-4 pt-3">
                {error && <div role="alert" className="mb-2 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">{error}</div>}
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-[12px] font-semibold text-[var(--text-secondary)]">{t("estPriceLabel")}</span>
                  <b className="text-[17px] font-extrabold text-[var(--text-primary)]">{priceText || (productType === "keychain" ? t("estPriceKeychain") : t("estPriceMap"))}</b>
                </div>
                <button onClick={submit} disabled={sending}
                  className="inline-flex min-h-[52px] w-full items-center justify-center gap-2 rounded-full bg-[var(--accent-strong)] px-5 py-3.5 text-sm font-bold text-white shadow-[0_16px_32px_rgba(11,92,87,0.24)] transition hover:bg-[var(--accent)] disabled:opacity-60">
                  {sending ? (<><Loader2 className="h-4 w-4 animate-spin" /> {t("sending")}</>) : t("submit")}
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );

  if (typeof document === "undefined") return null;
  return createPortal(dialog, document.body);
}
