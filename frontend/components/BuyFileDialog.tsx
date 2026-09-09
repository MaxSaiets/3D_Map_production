"use client";

import { useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import { X, FileDown, Loader2 } from "lucide-react";

/**
 * ⭐09.09.2026, рішення власника: друк-файл коштує 149 ₴.
 *
 * Навіщо окремий діалог, а не наявна форма замовлення. Заміряно за 30 днів:
 * модель створили 23 людини, і лише 8 з України. Друк і доставка — тільки по
 * Україні, тож дві третини тих, хто зробив усю роботу, не мали що купити.
 * Форма замовлення просить телефон і відділення Нової Пошти, бо ВЕЗЕ виріб.
 * Файл нікуди не везуть — потрібна лише пошта. Саме ця різниця й робить його
 * продаваним за межі України.
 *
 * Ціну не хардкодимо: беремо з /api/file/access/<task>, щоб вона правилась у
 * pricing.json без релізу.
 */
const API = process.env.NEXT_PUBLIC_API_URL || "";

type Access = { paid: boolean; priceUah: number; currency: string };
type Checkout = {
  alreadyPaid: boolean;
  orderNumber?: string;
  priceUah?: number;
  payment?: { action_url: string; data: string; signature: string };
};

export function BuyFileDialog({
  taskId,
  open,
  onClose,
  defaultEmail = "",
  onAlreadyPaid,
}: {
  taskId: string | null | undefined;
  open: boolean;
  onClose: () => void;
  defaultEmail?: string;
  /** Файл уже оплачено — панель може одразу спробувати завантаження. */
  onAlreadyPaid?: () => void;
}) {
  const t = useTranslations("scenario");
  const [access, setAccess] = useState<Access | null>(null);
  const [email, setEmail] = useState(defaultEmail);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const formRef = useRef<HTMLFormElement | null>(null);
  const [pay, setPay] = useState<Checkout["payment"] | null>(null);

  useEffect(() => { setEmail((prev) => prev || defaultEmail); }, [defaultEmail]);

  useEffect(() => {
    if (!open || !taskId) return;
    let alive = true;
    setError(null);
    fetch(`${API}/api/file/access/${encodeURIComponent(taskId)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d: Access | null) => { if (alive && d) setAccess(d); })
      .catch(() => { /* ціну покажемо з відповіді чеку */ });
    return () => { alive = false; };
  }, [open, taskId]);

  // Форму LiqPay сабмітимо лише після того, як вона з'явилась у DOM.
  useEffect(() => { if (pay) formRef.current?.submit(); }, [pay]);

  if (!open) return null;

  const price = access?.priceUah ?? 0;
  const priceLabel = price > 0 ? String(price) : "…";

  const buy = async () => {
    if (!taskId || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${API}/api/file/checkout`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task_id: taskId, email, locale: "uk" }),
      });
      if (res.status === 422) { setError(t("buyFileBadEmail")); return; }
      if (!res.ok) { setError(t("buyFileError")); return; }
      const data: Checkout = await res.json();
      if (data.alreadyPaid) {
        setAccess((a) => ({ paid: true, priceUah: a?.priceUah ?? price, currency: "UAH" }));
        onAlreadyPaid?.();
        return;
      }
      if (data.payment?.data && data.payment?.signature) {
        import("@/lib/analytics").then((m) => m.track("file_checkout", { priceUah: data.priceUah })).catch(() => {});
        setPay(data.payment);          // ефект вище надішле форму
        return;
      }
      setError(t("buyFileError"));
    } catch {
      setError(t("buyFileError"));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-[120] flex items-end justify-center bg-black/45 p-0 sm:items-center sm:p-4"
      role="dialog"
      aria-modal="true"
      aria-label={t("buyFileTitle")}
      data-testid="buy-file-dialog"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="w-full max-w-[420px] rounded-t-3xl bg-[var(--surface-panel,#fff)] p-5 shadow-2xl sm:rounded-3xl">
        <div className="mb-3 flex items-start justify-between gap-3">
          <h2 className="flex items-center gap-2 text-[17px] font-semibold text-[var(--text-primary)]">
            <FileDown size={18} className="text-[var(--accent-strong)]" />
            {t("buyFileTitle")}
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label={t("buyFileClose")}
            className="rounded-lg p-1 text-[var(--text-secondary)] hover:bg-black/5"
          >
            <X size={20} />
          </button>
        </div>

        {access?.paid ? (
          <p className="text-[14px] leading-relaxed text-[var(--text-primary)]" data-testid="buy-file-already">
            {t("buyFileAlready")}
          </p>
        ) : (
          <>
            <p className="text-[13.5px] leading-relaxed text-[var(--text-secondary)]">{t("buyFileBody")}</p>
            <label className="mt-4 block">
              <span className="mb-1 block text-[12px] font-semibold text-[var(--text-primary)]">
                {t("buyFileEmailLabel")}
              </span>
              <input
                type="email"
                inputMode="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@example.com"
                data-testid="buy-file-email"
                className="w-full rounded-xl border border-[var(--surface-border)] bg-white px-3 py-2.5 text-[14px] text-[var(--text-primary)] outline-none focus:border-[var(--accent-strong)]"
              />
            </label>
            {error && (
              <p className="mt-2 text-[12.5px] font-semibold text-[#a4342c]" data-testid="buy-file-error">{error}</p>
            )}
            <button
              type="button"
              onClick={buy}
              disabled={busy}
              data-testid="buy-file-pay"
              className="mt-4 inline-flex h-12 w-full items-center justify-center gap-2 rounded-full bg-[var(--accent-strong)] px-4 text-[14.5px] font-bold text-white transition hover:opacity-90 disabled:opacity-60"
            >
              {busy && <Loader2 size={16} className="animate-spin" />}
              {t("buyFilePay", { price: priceLabel })}
            </button>
            <p className="mt-2 text-center text-[11.5px] leading-snug text-[var(--text-secondary)]">
              {t("buyFileNote")}
            </p>
          </>
        )}

        {/* Форма LiqPay: додається в DOM і одразу надсилається ефектом вище. */}
        {pay && (
          <form ref={formRef} method="POST" action={pay.action_url} acceptCharset="utf-8" className="hidden">
            <input type="hidden" name="data" value={pay.data} />
            <input type="hidden" name="signature" value={pay.signature} />
          </form>
        )}
      </div>
    </div>
  );
}

export default BuyFileDialog;
