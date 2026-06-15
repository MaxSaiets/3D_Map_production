"use client";
export const dynamic = "force-dynamic";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { ArrowLeft, Loader2, Package, Users, RefreshCw, BarChart3, CheckCircle2 } from "lucide-react";
import { useAuth } from "@/components/AuthProvider";
import { setOwnerOptOut } from "@/lib/analytics";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

// Статуси замовлення (серверні значення) → ключі account-неймспейсу (вже перекладені).
const ORDER_STATUSES = ["new", "paid", "printed", "shipped", "done"] as const;
type OrderStatus = (typeof ORDER_STATUSES)[number];
const ORDER_STATUS_KEYS: Record<OrderStatus, string> = {
  new: "orderStatusNew", paid: "orderStatusPaid", printed: "orderStatusPrinted",
  shipped: "orderStatusShipped", done: "orderStatusDone",
};

export default function AdminPage() {
  const ts = useTranslations("account"); // повторно використовуємо вже перекладені статуси
  const ta = useTranslations("adminPanel"); // нові адмін-рядки
  const { user, loading, configured, signIn, getIdToken } = useAuth();
  const [tab, setTab] = useState<"stats" | "orders" | "users">("stats");
  const [orders, setOrders] = useState<any[]>([]);
  const [users, setUsers] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  // order_number, що зараз оновлюється (показуємо спінер на його select).
  const [savingStatus, setSavingStatus] = useState<string | null>(null);

  // Зміна статусу замовлення на сервері + оптимістичне оновлення картки.
  const setOrderStatus = useCallback(async (orderNumber: string | number, status: OrderStatus) => {
    const token = await getIdToken();
    if (!token) return;
    const num = String(orderNumber);
    setSavingStatus(num);
    try {
      const res = await fetch(`${API_BASE}/api/admin/orders/${encodeURIComponent(num)}/status`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ status }),
      });
      if (res.ok) {
        setOrders((os) => os.map((o) => (String(o.order_number) === num ? { ...o, status } : o)));
      }
    } catch {/* ignore */} finally {
      setSavingStatus(null);
    }
  }, [getIdToken]);

  const load = useCallback(async () => {
    const token = await getIdToken();
    if (!token) return;
    setBusy(true);
    try {
      const q = await fetch(`${API_BASE}/api/account/quota`, { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json());
      const admin = Boolean(q?.quota?.is_admin);
      setIsAdmin(admin);
      if (admin) {
        setOwnerOptOut(); // власні заходи більше не псують статистику відвідувачів
        const [o, us, st] = await Promise.all([
          fetch(`${API_BASE}/api/admin/orders`, { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json()),
          fetch(`${API_BASE}/api/admin/users`, { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json()),
          fetch(`${API_BASE}/api/admin/stats`, { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json()).catch(() => null),
        ]);
        setOrders(o.orders || []); setUsers(us.users || []); setStats(st || null);
      }
    } catch {/* ignore */} finally { setBusy(false); }
  }, [getIdToken]);

  useEffect(() => { if (user) load(); }, [user, load]);

  return (
    <div className="mx-auto min-h-[100dvh] max-w-[1100px] px-5 py-8 lg:px-8">
      <Link href="/" className="mb-6 inline-flex items-center gap-1.5 text-[13px] font-semibold text-ink-2 hover:text-ink"><ArrowLeft size={15} /> На сайт</Link>
      <div className="flex items-end justify-between">
        <h1 className="font-serif text-[clamp(26px,4vw,40px)] text-ink">Адмін-панель</h1>
        {user && isAdmin && (
          <button onClick={load} disabled={busy} className="inline-flex items-center gap-2 rounded-full border border-line px-4 py-2 text-sm font-semibold text-ink-2 hover:bg-bg-2">
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw size={15} />} Оновити
          </button>
        )}
      </div>

      {configured && !user && !loading && (
        <div className="mt-10 rounded-[20px] border border-line bg-paper p-10 text-center">
          <p className="text-ink-2">Потрібен вхід адміністратора.</p>
          <button onClick={signIn} className="mt-4 rounded-full bg-forest px-5 py-3 text-sm font-bold text-white">Увійти</button>
        </div>
      )}
      {loading && <div className="mt-10 flex justify-center"><Loader2 className="animate-spin text-forest" /></div>}
      {user && isAdmin === false && (
        <div className="mt-10 rounded-[20px] border border-amber-200 bg-amber-50 p-6 text-center text-amber-800">
          Цей акаунт не є адміністратором.
        </div>
      )}

      {user && isAdmin && (
        <>
          <div className="mt-5 flex flex-wrap gap-2">
            <button onClick={() => setTab("stats")} className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold ${tab === "stats" ? "bg-forest text-white" : "border border-line text-ink-2"}`}><BarChart3 size={15} /> Статистика</button>
            <button onClick={() => setTab("orders")} className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold ${tab === "orders" ? "bg-forest text-white" : "border border-line text-ink-2"}`}><Package size={15} /> Замовлення ({orders.length})</button>
            <button onClick={() => setTab("users")} className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold ${tab === "users" ? "bg-forest text-white" : "border border-line text-ink-2"}`}><Users size={15} /> Користувачі ({users.length})</button>
          </div>

          {tab === "stats" && (
            <div className="mt-5">
              {!stats ? <p className="text-ink-3">Поки немає даних аналітики.</p> : (
                <>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                    {[
                      ["Унікальні відвідувачі", stats.totals?.uniqueVisitors],
                      ["Перегляди сторінок", stats.totals?.pageviews],
                      ["Усього подій", stats.totals?.events],
                    ].map(([label, val]) => (
                      <div key={label as string} className="rounded-[14px] border border-line bg-paper p-4">
                        <div className="font-serif text-[28px] text-ink">{val ?? 0}</div>
                        <div className="text-[12px] text-ink-3">{label as string}</div>
                      </div>
                    ))}
                  </div>

                  {(stats.byDay?.length > 0) && (
                    <div className="mt-5 rounded-[14px] border border-line bg-paper p-4">
                      <div className="mb-3 text-[13px] font-semibold text-ink-2">Перегляди за днями</div>
                      <div className="flex items-stretch gap-1.5" style={{ height: 120 }}>
                        {stats.byDay.map((d: any) => {
                          const max = Math.max(...stats.byDay.map((x: any) => x.pageviews || 0), 1);
                          // h-full на колонці ОБОВ'ЯЗКОВЕ: стовпчик має % висоту,
                          // а % резолвиться лише проти батька з ВИЗНАЧЕНОЮ висотою.
                          // Без цього (старий items-end → колонка стискалась до
                          // контенту) усі бари виходили 0 → графік порожній.
                          return (
                            <div key={d.day} className="flex h-full flex-1 flex-col items-center justify-end" title={`${d.day}: ${d.pageviews} переглядів, ${d.visitors} відвідувачів`}>
                              <div className="w-full rounded-t bg-forest/80" style={{ height: `${Math.round(((d.pageviews || 0) / max) * 100)}%`, minHeight: d.pageviews ? 3 : 0 }} />
                            </div>
                          );
                        })}
                      </div>
                      <div className="mt-1.5 flex justify-between text-[10px] text-ink-3">
                        <span>{stats.byDay[0]?.day}</span><span>{stats.byDay[stats.byDay.length - 1]?.day}</span>
                      </div>
                    </div>
                  )}

                  {stats.funnel?.length > 0 && <Funnel funnel={stats.funnel} />}

                  <div className="mt-5 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                    <StatList title="Країни" rows={stats.byCountry} />
                    <StatList title="Топ сторінок" rows={stats.topPaths} />
                    <StatList title="Події" rows={stats.topEvents} />
                    <StatList title="Мови" rows={stats.byLocale} />
                  </div>

                  <div className="mt-4 grid gap-4 md:grid-cols-2">
                    <StatList title="Звідки прийшли (реферери)" rows={stats.topRefs} />
                    <StatList title="Топ кліків (елемент)" rows={stats.topClicks} />
                  </div>

                  {stats.clicksByPath && Object.keys(stats.clicksByPath).length > 0 && (
                    <ClickMaps clicksByPath={stats.clicksByPath} />
                  )}

                  <p className="mt-4 text-[12px] text-ink-3">Власна аналітика на сервері · без cookie-стеження · IP не зберігається (лише денний хеш + код країни Cloudflare) · твої власні (адмінські) заходи й заходи з localhost не рахуються.</p>
                </>
              )}
            </div>
          )}

          {tab === "orders" && (
            <div className="mt-5 space-y-3">
              {/* Платіжні події у журналі (type:"payment") не є замовленнями — ховаємо. */}
              {orders.filter((o) => o.type !== "payment").length === 0 ? <p className="text-ink-3">Поки немає замовлень.</p> : orders.filter((o) => o.type !== "payment").map((o, i) => {
                const status: OrderStatus = (ORDER_STATUSES as readonly string[]).includes(o.status) ? o.status : "new";
                const saving = savingStatus === String(o.order_number);
                const deliveryParts = [o.delivery_country, o.delivery_city, o.delivery_branch, o.delivery_address].filter(Boolean).join(", ");
                return (
                  <div key={i} className="rounded-[14px] border border-line bg-paper p-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="font-serif text-[17px] text-ink">#{o.order_number} · {o.name || "—"}</div>
                      <div className="text-[12px] text-ink-3">{o.created_at ? new Date(o.created_at).toLocaleString("uk") : ""}</div>
                    </div>
                    <div className="mt-1 text-[13px] text-ink-2">
                      📞 {o.phone || "—"} · {o.product_type === "keychain" ? "Брелок" : "Мапа"}
                      {o.summary?.city ? ` · ${o.summary.city}` : ""}
                      {o.summary?.district ? ` · ${o.summary.district}` : ""}
                      {o.summary?.label ? ` · «${o.summary.label}»` : ""}
                      {o.summary?.size ? ` · ${o.summary.size}` : ""}
                    </div>
                    {o.user_email && <div className="mt-1 text-[13px] text-ink-2">✉️ {o.user_email}</div>}
                    {o.est_price && <div className="mt-1 text-[13px] font-semibold text-ink">💰 {o.est_price} <span className="font-normal text-ink-3">{ta("withoutDelivery")}</span></div>}
                    {deliveryParts && (
                      <div className="mt-1 text-[13px] text-ink-2">
                        🚚 {o.delivery_method ? `${o.delivery_method} · ` : ""}{deliveryParts}
                      </div>
                    )}
                    {o.comment && <div className="mt-1 text-[13px] text-ink-3">💬 {o.comment}</div>}
                    <div className="mt-1 text-[12px] text-ink-3">
                      💳 {o.payment_url ? ta("payShown") : ta("payManual")}
                    </div>

                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      <label className="inline-flex items-center gap-2 text-[12px] text-ink-2">
                        {ta("statusLabel")}
                        <select
                          value={status}
                          disabled={saving}
                          onChange={(e) => setOrderStatus(o.order_number, e.target.value as OrderStatus)}
                          className="rounded-full border border-line bg-paper px-3 py-1.5 text-[13px] font-semibold text-ink disabled:opacity-60"
                        >
                          {ORDER_STATUSES.map((s) => (
                            <option key={s} value={s}>{ts(ORDER_STATUS_KEYS[s])}</option>
                          ))}
                        </select>
                      </label>
                      {status !== "paid" && (
                        <button
                          onClick={() => setOrderStatus(o.order_number, "paid")}
                          disabled={saving}
                          className="inline-flex items-center gap-1.5 rounded-full border border-forest/40 px-3 py-1.5 text-[12px] font-semibold text-forest hover:bg-forest/5 disabled:opacity-60"
                        >
                          <CheckCircle2 size={14} /> {ta("markPaid")}
                        </button>
                      )}
                      {saving && <Loader2 className="h-4 w-4 animate-spin text-ink-3" />}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {tab === "users" && (
            <div className="mt-5 overflow-x-auto">
              <table className="w-full text-left text-[14px]">
                <thead className="text-[12px] uppercase text-ink-3">
                  <tr><th className="py-2">Email</th><th className="py-2">Завантажень</th><th className="py-2">Моделей</th><th className="py-2">Дата</th></tr>
                </thead>
                <tbody>
                  {users.map((u) => (
                    <tr key={u.uid} className="border-t border-line">
                      <td className="py-2 text-ink">{u.email || u.uid.slice(0, 8)}</td>
                      <td className="py-2 text-ink-2">{u.downloads}</td>
                      <td className="py-2 text-ink-2">{u.models}</td>
                      <td className="py-2 text-ink-3">{u.created_at ? new Date(u.created_at * 1000).toLocaleDateString("uk") : ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function StatList({ title, rows }: { title: string; rows?: [string, number][] }) {
  const data = rows || [];
  const max = Math.max(...data.map((r) => r[1] || 0), 1);
  return (
    <div className="rounded-[14px] border border-line bg-paper p-4">
      <div className="mb-2 text-[13px] font-semibold text-ink-2">{title}</div>
      {data.length === 0 ? <p className="text-[12px] text-ink-3">—</p> : (
        <ul className="space-y-1.5">
          {data.map(([label, count]) => (
            <li key={label} className="text-[12px]">
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-ink-2" title={label || "/"}>{label || "/"}</span>
                <span className="shrink-0 font-semibold text-ink">{count}</span>
              </div>
              <div className="mt-0.5 h-1 rounded bg-bg-2"><div className="h-1 rounded bg-forest/70" style={{ width: `${Math.round((count / max) * 100)}%` }} /></div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

const FUNNEL_LABELS: Record<string, string> = {
  view: "Зайшли в конструктор",
  area: "Виділили зону на карті",
  generate: "Натиснули «Згенерувати»",
  order_open: "Відкрили форму замовлення",
  order_submit: "Оформили замовлення",
};

function Funnel({ funnel }: { funnel: { step: string; count: number; pct: number }[] }) {
  const max = Math.max(...funnel.map((f) => f.count), 1);
  return (
    <div className="mt-5 rounded-[14px] border border-line bg-paper p-4">
      <div className="mb-1 text-[13px] font-semibold text-ink-2">Воронка конверсії</div>
      <div className="mb-3 text-[11px] text-ink-3">Скільки відвідувачів доходить до кожного кроку (і де відвалюються).</div>
      <div className="space-y-2.5">
        {funnel.map((f, i) => {
          const prev = i > 0 ? funnel[i - 1].count : f.count;
          const drop = prev > 0 ? Math.round(((prev - f.count) / prev) * 100) : 0;
          return (
            <div key={f.step}>
              <div className="mb-1 flex items-center justify-between text-[12px]">
                <span className="text-ink-2">{i + 1}. {FUNNEL_LABELS[f.step] || f.step}</span>
                <span className="font-semibold text-ink">{f.count}<span className="ml-1 font-normal text-ink-3">· {f.pct}%</span></span>
              </div>
              <div className="h-3.5 rounded bg-bg-2">
                <div className="h-3.5 rounded bg-forest/80" style={{ width: `${Math.round((f.count / max) * 100)}%` }} />
              </div>
              {i > 0 && drop > 0 && <div className="mt-0.5 text-[10px] font-semibold text-red-600">↓ втрата −{drop}% на цьому кроці</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ClickMaps({ clicksByPath }: { clicksByPath: Record<string, [number, number][]> }) {
  return (
    <div className="mt-5">
      <div className="mb-1 text-[13px] font-semibold text-ink-2">Карта кліків — куди тикають користувачі</div>
      <div className="mb-2 text-[11px] text-ink-3">Точки = кліки у % екрана (x/y). Скупчення показують, що привертає увагу.</div>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {Object.entries(clicksByPath).map(([path, pts]) => (
          <div key={path} className="rounded-[14px] border border-line bg-paper p-3">
            <div className="mb-2 truncate text-[12px] font-semibold text-ink-2" title={path}>{path || "/"}<span className="ml-1 font-normal text-ink-3">· {pts.length} кліків</span></div>
            <div className="relative w-full overflow-hidden rounded-lg border border-line bg-bg-2" style={{ aspectRatio: "16 / 10" }}>
              {pts.map((p, i) => (
                <span key={i} className="absolute h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-forest" style={{ left: `${p[0]}%`, top: `${p[1]}%`, opacity: 0.28 }} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
