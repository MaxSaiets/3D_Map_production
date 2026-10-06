"use client";

/**
 * Адмінка → «Підписки й безлім»: платні підписки Pro (LiqPay) і ручні гранти
 * безліміту (пошта/uid на N днів, до дати або безстроково) з відкликанням.
 * Лише для власника, тож тексти українською без i18n (як і решта адмін-заголовків).
 */
import { useCallback, useEffect, useState } from "react";
import { Loader2, ShieldCheck, Trash2 } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

interface Grant {
  id: string; who: string; until: string | null; note?: string; created_at?: string;
  created_by?: string; revoked_at?: string | null; active: boolean;
}
interface Sub {
  order_id?: string; email?: string; status: string; active?: boolean; amount?: number; currency?: string;
  paid_until?: string | null; renews_at?: string | null; created_at?: string; cancelled_at?: string | null;
  payments?: { amount: number; currency: string; ts: string }[]; country?: string; terms_version?: string;
}

const d = (iso?: string | null) => (iso ? new Date(iso.length === 10 ? `${iso}T12:00:00` : iso).toLocaleDateString("uk-UA") : "—");
const STATUS: Record<string, string> = {
  active: "активна", cancelled: "скасована (діє до кінця періоду)", pending: "очікує оплати",
  failed: "оплата не пройшла", refunded: "кошти повернено",
};

export function AdminProPanel({ getIdToken }: { getIdToken: () => Promise<string | null> }) {
  const [grants, setGrants] = useState<Grant[]>([]);
  const [envGrants, setEnvGrants] = useState<string[]>([]);
  const [subs, setSubs] = useState<Sub[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [who, setWho] = useState("");
  const [mode, setMode] = useState<"days" | "until" | "forever">("days");
  const [days, setDays] = useState(30);
  const [until, setUntil] = useState("");
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [showOld, setShowOld] = useState(false);

  const api = useCallback(async (path: string, init?: RequestInit) => {
    const token = await getIdToken();
    const r = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { ...(init?.headers || {}), Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j?.detail || `HTTP ${r.status}`);
    return j;
  }, [getIdToken]);

  const load = useCallback(async () => {
    setLoading(true); setErr(null);
    try {
      const [g, s] = await Promise.all([api("/api/admin/grants"), api("/api/admin/subscriptions")]);
      setGrants(g.grants || []); setEnvGrants(g.env || []); setSubs(s.subscriptions || []);
    } catch (e) {
      setErr(String((e as Error).message));
    } finally {
      setLoading(false);
    }
  }, [api]);

  useEffect(() => { load(); }, [load]);

  const add = async () => {
    if (!who.trim()) return;
    setSaving(true); setErr(null);
    try {
      const body: Record<string, unknown> = { who: who.trim(), note };
      if (mode === "days") body.days = days;
      else if (mode === "until") body.until = until;
      else body.forever = true;
      await api("/api/admin/grants", { method: "POST", body: JSON.stringify(body) });
      setWho(""); setNote("");
      await load();
    } catch (e) {
      setErr(String((e as Error).message));
    } finally {
      setSaving(false);
    }
  };

  const revoke = async (g: Grant) => {
    if (!window.confirm(`Забрати безлім у ${g.who}?`)) return;
    try {
      await api(`/api/admin/grants/${g.id}`, { method: "DELETE" });
      await load();
    } catch (e) {
      setErr(String((e as Error).message));
    }
  };

  const activeSubs = subs.filter((s) => s.active);
  const mrr = activeSubs.filter((s) => s.status === "active")
    .reduce<Record<string, number>>((acc, s) => { acc[s.currency || "?"] = (acc[s.currency || "?"] || 0) + (s.amount || 0); return acc; }, {});
  const shownGrants = showOld ? grants : grants.filter((g) => g.active);
  const inputCls = "rounded-xl border border-line bg-white px-3 py-2 text-sm text-ink";

  return (
    <div className="mt-5 space-y-6">
      {err && <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">{err}</div>}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          ["Активні підписки", activeSubs.length],
          ["Щомісячний дохід", Object.entries(mrr).map(([c, v]) => `${v} ${c}`).join(" + ") || "0"],
          ["Усього оформлень", subs.filter((s) => s.status !== "pending").length],
          ["Активні гранти", grants.filter((g) => g.active).length + envGrants.length],
        ].map(([label, val]) => (
          <div key={label as string} className="rounded-[14px] border border-line bg-paper p-4">
            <div className="font-serif text-[24px] text-ink">{val as any}</div>
            <div className="text-[12px] text-ink-3">{label as string}</div>
          </div>
        ))}
      </div>

      {/* ── Видати безлім вручну ── */}
      <div className="rounded-[14px] border border-line bg-paper p-4">
        <div className="mb-3 flex items-center gap-2 text-[15px] font-semibold text-ink"><ShieldCheck size={16} className="text-forest" /> Видати безлім вручну</div>
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex min-w-[240px] flex-1 flex-col gap-1 text-[12px] text-ink-3">
            Пошта або uid
            <input className={inputCls} value={who} onChange={(e) => setWho(e.target.value)} placeholder="name@gmail.com" />
          </label>
          <label className="flex flex-col gap-1 text-[12px] text-ink-3">
            Строк
            <select className={inputCls} value={mode} onChange={(e) => setMode(e.target.value as any)}>
              <option value="days">На кількість днів</option>
              <option value="until">До дати</option>
              <option value="forever">Безстроково</option>
            </select>
          </label>
          {mode === "days" && (
            <label className="flex flex-col gap-1 text-[12px] text-ink-3">
              Днів
              <div className="flex gap-1">
                <input type="number" min={1} max={3650} className={`${inputCls} w-24`} value={days} onChange={(e) => setDays(Math.max(1, Number(e.target.value) || 1))} />
                {[1, 7, 30, 365].map((n) => (
                  <button key={n} type="button" onClick={() => setDays(n)} className={`rounded-xl border px-2 text-[12px] ${days === n ? "border-forest text-forest" : "border-line text-ink-2"}`}>{n}</button>
                ))}
              </div>
            </label>
          )}
          {mode === "until" && (
            <label className="flex flex-col gap-1 text-[12px] text-ink-3">
              Останній день (включно)
              <input type="date" className={inputCls} value={until} onChange={(e) => setUntil(e.target.value)} />
            </label>
          )}
          <label className="flex min-w-[200px] flex-1 flex-col gap-1 text-[12px] text-ink-3">
            Нотатка (за що)
            <input className={inputCls} value={note} onChange={(e) => setNote(e.target.value)} placeholder="друкарня, бартер, тест…" />
          </label>
          <button onClick={add} disabled={saving || !who.trim() || (mode === "until" && !until)}
            className="inline-flex items-center gap-2 rounded-full bg-forest px-5 py-2.5 text-sm font-bold text-white disabled:opacity-40" style={{ background: "var(--forest,#2E4A3A)" }}>
            {saving && <Loader2 className="h-4 w-4 animate-spin" />} Видати
          </button>
        </div>
        <p className="mt-2 text-[12px] text-ink-3">Діє одразу, без рестарту. Людина бачить «Безліміт» у кабінеті після оновлення сторінки. Пошта має бути та, з якою вона входить.</p>
      </div>

      {/* ── Список грантів ── */}
      <div className="rounded-[14px] border border-line bg-paper p-4">
        <div className="mb-3 flex items-center justify-between">
          <div className="text-[15px] font-semibold text-ink">Гранти безліміту</div>
          <label className="flex items-center gap-2 text-[12px] text-ink-3">
            <input type="checkbox" checked={showOld} onChange={(e) => setShowOld(e.target.checked)} /> показати завершені й відкликані
          </label>
        </div>
        {loading && <Loader2 className="h-4 w-4 animate-spin text-ink-3" />}
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]">
            <thead><tr className="text-left text-ink-3">
              <th className="py-1.5 pr-2">Хто</th><th className="px-2">До</th><th className="px-2">Нотатка</th><th className="px-2">Видано</th><th className="px-2">Стан</th><th />
            </tr></thead>
            <tbody>
              {shownGrants.map((g) => (
                <tr key={g.id} className="border-t border-line">
                  <td className="py-1.5 pr-2 font-semibold text-ink">{g.who}</td>
                  <td className="px-2 text-ink-2">{g.until ? d(g.until) : "безстроково"}</td>
                  <td className="px-2 text-ink-2">{g.note || "—"}</td>
                  <td className="px-2 text-ink-3">{d(g.created_at)}</td>
                  <td className="px-2">{g.active ? <span className="text-forest">діє</span> : <span className="text-ink-3">{g.revoked_at ? `відкликано ${d(g.revoked_at)}` : "завершився"}</span>}</td>
                  <td className="px-2 text-right">{g.active && (
                    <button onClick={() => revoke(g)} title="Забрати безлім" className="inline-flex items-center gap-1 text-[12px] text-red-700 hover:underline"><Trash2 size={13} /> забрати</button>
                  )}</td>
                </tr>
              ))}
              {envGrants.map((e) => (
                <tr key={e} className="border-t border-line text-ink-3">
                  <td className="py-1.5 pr-2">{e.split(":")[0]}</td>
                  <td className="px-2">{e.includes(":") ? e.split(":")[1] : "безстроково"}</td>
                  <td className="px-2" colSpan={4}>із .env на сервері (міняється лише там)</td>
                </tr>
              ))}
              {!shownGrants.length && !envGrants.length && (
                <tr><td colSpan={6} className="py-2 text-ink-3">Немає грантів.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Платні підписки ── */}
      <div className="rounded-[14px] border border-line bg-paper p-4">
        <div className="mb-3 text-[15px] font-semibold text-ink">Платні підписки Pro</div>
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]">
            <thead><tr className="text-left text-ink-3">
              <th className="py-1.5 pr-2">Пошта</th><th className="px-2">Стан</th><th className="px-2">Ціна</th><th className="px-2">Оплачено до</th><th className="px-2">Платежів</th><th className="px-2">Країна</th><th className="px-2">Оформлено</th>
            </tr></thead>
            <tbody>
              {subs.map((s) => (
                <tr key={s.order_id} className="border-t border-line">
                  <td className="py-1.5 pr-2 font-semibold text-ink">{s.email || "—"}</td>
                  <td className={`px-2 ${s.active ? "text-forest" : "text-ink-3"}`}>{STATUS[s.status] || s.status}</td>
                  <td className="px-2 text-ink-2">{s.amount} {s.currency}/міс</td>
                  <td className="px-2 text-ink-2">{d(s.paid_until)}</td>
                  <td className="px-2 text-ink-2">{s.payments?.length ?? 0}</td>
                  <td className="px-2 text-ink-3">{s.country || "—"}</td>
                  <td className="px-2 text-ink-3">{d(s.created_at)}</td>
                </tr>
              ))}
              {!subs.length && <tr><td colSpan={7} className="py-2 text-ink-3">Поки жодної підписки.</td></tr>}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-[12px] text-ink-3">«Очікує оплати» — людина відкрила оплату LiqPay, але не завершила. Згода з умовами (час, редакція, тексти позначок) збережена на сервері в data/subscriptions.json.</p>
      </div>
    </div>
  );
}
