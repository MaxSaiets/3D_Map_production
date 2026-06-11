"use client";

import { useState } from "react";
import { createPortal } from "react-dom";
import { X, Loader2, CheckCircle2, Truck, Package } from "lucide-react";
import { useTranslations } from "next-intl";
import { capturePreviewImages } from "@/lib/capturePreview";
import { useAuth } from "@/components/AuthProvider";

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
}: {
  open: boolean;
  onClose: () => void;
  taskId: string | null;
  productType: "map" | "keychain";
  summary: OrderSummary;
  /** Жива орієнтовна ціна з /api/quote; без неї — статичний i18n-fallback. */
  priceText?: string;
}) {
  const t = useTranslations("order");
  const [name, setName] = useState("");
  const [phone, setPhone] = useState("");
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
  const [paymentUrl, setPaymentUrl] = useState<string | null>(null);
  const { getIdToken } = useAuth();

  if (!open) return null;

  const submit = async () => {
    if (!name.trim()) { setError(t("errName")); return; }
    if (!phone.trim()) { setError(t("errPhone")); return; }
    if (region === "eu") {
      if (!euCountry || !city.trim() || (!branch.trim() && !address.trim())) {
        setError(t("errEu"));
        return;
      }
    } else if (delivery !== "pickup" && (!city.trim() || !branch.trim())) {
      setError(delivery === "nova" ? t("errNova") : t("errUkr"));
      return;
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
          name, phone, product_type: productType, task_id: taskId,
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
      setPaymentUrl(data.payment?.url || null);
      try { const { track } = await import("@/lib/analytics"); track("order", { product: productType, delivery }); } catch { /* ignore */ }
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
    <div className="fixed inset-0 z-[80] flex items-end justify-center bg-black/40 p-0 backdrop-blur-sm sm:items-center sm:p-4" onClick={onClose}>
      <div
        className="max-h-[92dvh] w-full max-w-[460px] overflow-y-auto rounded-t-[28px] border border-[var(--surface-border)] bg-[var(--surface-panel,#fff)] p-5 shadow-[0_30px_80px_rgba(15,23,42,0.35)] sm:rounded-[28px]"
        onClick={(e) => e.stopPropagation()}
      >
        {orderNumber ? (
          <div className="py-6 text-center">
            <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-emerald-100 text-emerald-700">
              <CheckCircle2 size={30} />
            </div>
            <h3 className="font-serif text-2xl text-[var(--text-primary)]">{t("acceptedTitle")}</h3>
            <p className="mt-2 text-sm text-[var(--text-secondary)]">
              {t("orderNo")} <b className="text-[var(--text-primary)]">#{orderNumber}</b>.<br />
              {t("acceptedText")}
            </p>
            {paymentUrl && (
              <>
                <a
                  href={paymentUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-5 inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-full bg-[var(--bronze,#8E6B3D)] px-5 py-3 text-[15px] font-extrabold text-white shadow-[0_16px_34px_rgba(142,107,61,0.32)] transition hover:opacity-90"
                >
                  {t("payNow")} · {priceText || ""}
                </a>
                <p className="mt-2 text-[11px] leading-4 text-[var(--text-secondary)]">{t("payLater")}</p>
              </>
            )}
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
                  <h3 className="font-serif text-xl text-[var(--text-primary)]">{t("title")}</h3>
                  <p className="text-[11px] text-[var(--text-secondary)]">{productType === "keychain" ? t("prodKeychain") : t("prodMap")}{summary.size ? ` · ${summary.size}` : ""}</p>
                </div>
              </div>
              <button onClick={onClose} className="rounded-lg p-1 text-[var(--text-secondary)] hover:bg-black/5"><X size={20} /></button>
            </div>

            <div className="space-y-3">
              <input className={fieldCls} placeholder={t("phName")} value={name} onChange={(e) => setName(e.target.value)} />
              <input className={fieldCls} placeholder={t("phPhone")} value={phone} onChange={(e) => setPhone(e.target.value)} inputMode="tel" />

              <div className="flex items-center gap-2 rounded-2xl border border-[var(--surface-border)] bg-white/70 p-1 text-xs">
                {([["ua", t("regionUa")], ["eu", t("regionEu")]] as [Region, string][]).map(([k, lbl]) => (
                  <button key={k} type="button"
                    onClick={() => { setRegion(k); setDelivery(k === "ua" ? "nova" : "novapost_eu"); }}
                    className={`flex-1 rounded-xl px-2 py-2 font-semibold transition ${region === k ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)]"}`}>
                    {lbl}
                  </button>
                ))}
              </div>

              <div className="flex items-center gap-2 rounded-2xl border border-[var(--surface-border)] bg-white/70 p-1 text-xs">
                {(region === "ua"
                  ? ([["nova", t("nova")], ["ukr", t("ukr")]] as [Delivery, string][])
                  : ([["novapost_eu", "Nova Post (EU)"], ["meest", "Meest"]] as [Delivery, string][])
                ).map(([k, lbl]) => (
                  <button key={k} type="button" onClick={() => setDelivery(k)}
                    className={`flex-1 rounded-xl px-2 py-2 font-semibold transition ${delivery === k ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)]"}`}>
                    {lbl}
                  </button>
                ))}
              </div>

              {region === "eu" && (
                <select className={fieldCls} value={euCountry} onChange={(e) => setEuCountry(e.target.value)}>
                  <option value="">{t("phCountry")}</option>
                  {EU_COUNTRIES.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              )}

              {delivery !== "pickup" && (
                <>
                  <input className={fieldCls} placeholder={t("phCity")} value={city} onChange={(e) => setCity(e.target.value)} />
                  {region === "ua" ? (
                    <>
                      <input className={fieldCls} placeholder={delivery === "nova" ? t("phNova") : t("phUkr")} value={branch} onChange={(e) => setBranch(e.target.value)} />
                      {delivery === "ukr" && (
                        <input className={fieldCls} placeholder={t("phAddress")} value={address} onChange={(e) => setAddress(e.target.value)} />
                      )}
                    </>
                  ) : delivery === "novapost_eu" ? (
                    <input className={fieldCls} placeholder={t("phBranchEu")} value={branch} onChange={(e) => setBranch(e.target.value)} />
                  ) : (
                    <input className={fieldCls} placeholder={t("phAddressEu")} value={address} onChange={(e) => setAddress(e.target.value)} />
                  )}
                </>
              )}

              <textarea className={`${fieldCls} min-h-[64px] resize-none`} placeholder={t("phComment")} value={comment} onChange={(e) => setComment(e.target.value)} />

              <div className="flex items-center justify-between rounded-2xl bg-[rgba(176,141,87,0.12)] px-4 py-2.5 text-sm">
                <span className="text-[var(--text-secondary)]">{t("estPriceLabel")}</span>
                <b className="text-[var(--text-primary)]">{priceText || (productType === "keychain" ? t("estPriceKeychain") : t("estPriceMap"))}</b>
              </div>

              <div className="flex items-start gap-2 rounded-2xl bg-[rgba(46,74,58,0.06)] px-3 py-2.5 text-[11px] leading-4 text-[var(--text-secondary)]">
                <Truck size={14} className="mt-0.5 shrink-0 text-[var(--accent-strong)]" />
                <span>
                  {/* UX: чіткі строки знімають головний страх «коли отримаю?» */}
                  <b className="text-[var(--text-primary)]">{t("leadTime")}</b>
                  {" — "}
                  {t("paymentNote")}
                </span>
              </div>

              {error && <div className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">{error}</div>}

              <button onClick={submit} disabled={sending}
                className="inline-flex min-h-13 w-full items-center justify-center gap-2 rounded-full bg-[var(--accent-strong)] px-5 py-3.5 text-sm font-bold text-white shadow-[0_16px_32px_rgba(11,92,87,0.24)] transition hover:bg-[var(--accent)] disabled:opacity-60">
                {sending ? (<><Loader2 className="h-4 w-4 animate-spin" /> {t("sending")}</>) : t("submit")}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );

  if (typeof document === "undefined") return null;
  return createPortal(dialog, document.body);
}
