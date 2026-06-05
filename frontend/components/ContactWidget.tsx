"use client";

import { useEffect, useState } from "react";
import { MessageCircle, X, Loader2, CheckCircle2 } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

/**
 * Floating round "contact us" button (bottom-right) + popup.
 * Collects phone + message and sends it to the Telegram CRM via /api/contact.
 * Mounted globally so it's available on every page.
 */
export function ContactWidget() {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [phone, setPhone] = useState("");
  const [message, setMessage] = useState("");
  const [sending, setSending] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Allow any part of the app to open this widget (e.g. download-limit reached)
  // via window.dispatchEvent(new CustomEvent("monadruk:open-contact", {detail:{message}})).
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail || {};
      if (detail.message) setMessage(String(detail.message));
      setDone(false);
      setOpen(true);
    };
    window.addEventListener("monadruk:open-contact", handler as EventListener);
    return () => window.removeEventListener("monadruk:open-contact", handler as EventListener);
  }, []);

  const submit = async () => {
    if (!phone.trim()) { setError("Вкажіть телефон"); return; }
    setError(null);
    setSending(true);
    try {
      const res = await fetch(`${API_BASE}/api/contact`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name, phone, message,
          source: typeof window !== "undefined" ? window.location.pathname : "",
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setDone(true);
    } catch {
      setError("Не вдалося відправити. Спробуйте ще раз.");
    } finally {
      setSending(false);
    }
  };

  const fieldCls = "w-full rounded-2xl border border-[var(--surface-border,#e3dccb)] bg-white px-4 py-3 text-sm text-[var(--ink,#1B2A22)] outline-none transition focus:border-[rgba(46,74,58,0.4)]";

  return (
    <>
      {/* Popup */}
      {open && (
        <div className="fixed bottom-24 right-4 z-[90] w-[calc(100vw-2rem)] max-w-[340px] rounded-[22px] border border-[var(--surface-border,#e3dccb)] bg-[var(--paper-2,#fff)] p-5 shadow-[0_24px_64px_rgba(15,23,42,0.28)] fade-up">
          {done ? (
            <div className="py-4 text-center">
              <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-emerald-100 text-emerald-700">
                <CheckCircle2 size={26} />
              </div>
              <h4 className="font-serif text-lg text-[var(--ink,#1B2A22)]">Дякуємо!</h4>
              <p className="mt-1 text-sm text-[var(--ink-2,#4b5a50)]">Ми зв'яжемося з вами найближчим часом.</p>
              <button onClick={() => { setOpen(false); setDone(false); setName(""); setPhone(""); setMessage(""); }}
                className="mt-4 w-full rounded-full bg-[var(--forest,#2E4A3A)] px-4 py-2.5 text-sm font-semibold text-white">Закрити</button>
            </div>
          ) : (
            <>
              <div className="mb-3 flex items-start justify-between">
                <div>
                  <h4 className="font-serif text-lg text-[var(--ink,#1B2A22)]">Звʼязатися з нами</h4>
                  <p className="text-[12px] text-[var(--ink-3,#7c887f)]">Залиште телефон — ми напишемо вам.</p>
                </div>
                <button onClick={() => setOpen(false)} className="rounded-lg p-1 text-[var(--ink-3,#7c887f)] hover:bg-black/5"><X size={18} /></button>
              </div>
              <div className="space-y-2.5">
                <input className={fieldCls} placeholder="Ім'я (необовʼязково)" value={name} onChange={(e) => setName(e.target.value)} />
                <input className={fieldCls} placeholder="Телефон (+380…)" value={phone} onChange={(e) => setPhone(e.target.value)} inputMode="tel" />
                <textarea className={`${fieldCls} min-h-[70px] resize-none`} placeholder="Повідомлення (необовʼязково)" value={message} onChange={(e) => setMessage(e.target.value)} />
                {error && <div className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">{error}</div>}
                <button onClick={submit} disabled={sending}
                  className="inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-full bg-[var(--forest,#2E4A3A)] px-4 py-3 text-sm font-bold text-white transition hover:opacity-90 disabled:opacity-60">
                  {sending ? (<><Loader2 className="h-4 w-4 animate-spin" /> Надсилаємо…</>) : "Надіслати"}
                </button>
              </div>
            </>
          )}
        </div>
      )}

      {/* Floating button */}
      <button
        type="button"
        aria-label="Звʼязатися з нами"
        onClick={() => setOpen((v) => !v)}
        className="fixed bottom-5 right-4 z-[90] flex h-14 w-14 items-center justify-center rounded-full text-white shadow-[0_14px_34px_rgba(46,74,58,0.4)] transition hover:scale-105"
        style={{ background: "var(--forest, #2E4A3A)" }}
      >
        {open ? <X size={24} /> : <MessageCircle size={24} />}
      </button>
    </>
  );
}
