"use client";
export const dynamic = "force-dynamic";

import Link from "next/link";
import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import { useTranslations } from "next-intl";
import { ArrowLeft, Loader2, Package, Users, RefreshCw, BarChart3, CheckCircle2, Search, Download, ChevronDown, ChevronRight } from "lucide-react";
import { useAuth } from "@/components/AuthProvider";
import { setOwnerOptOut } from "@/lib/analytics";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

// Топ-кліки: бек дає "/шлях · <ярлик>". Коли клік був по елементу БЕЗ тексту
// (карта/контейнер/іконка), ярлик = голий тег (div/canvas/path) — незрозуміло
// «що тикали». Замінюємо голі теги на людські описи; справжні текст-ярлики
// (з пробілами/великими літерами) проходять без змін.
const CLICK_TAG_LABELS: Record<string, string> = {
  div: "область/контейнер", canvas: "🗺 карта або 3D-превʼю", section: "секція",
  path: "іконка (SVG)", svg: "іконка", a: "посилання", button: "кнопка",
  img: "зображення", span: "напис", li: "пункт списку", ul: "список",
};
function prettyClicks(rows?: [string, number][]): [string, number][] {
  if (!Array.isArray(rows)) return [];
  return rows.map(([label, count]) => {
    const m = /^(.*) · ([a-z]+)$/.exec(label || "");
    return m && CLICK_TAG_LABELS[m[2]] ? [`${m[1]} · ${CLICK_TAG_LABELS[m[2]]}`, count] : [label, count];
  });
}

// Статуси замовлення (серверні значення) → ключі account-неймспейсу (вже перекладені).
const ORDER_STATUSES = ["pending_payment", "new", "paid", "printed", "shipped", "done"] as const;
type OrderStatus = (typeof ORDER_STATUSES)[number];
const ORDER_STATUS_KEYS: Record<OrderStatus, string> = {
  pending_payment: "orderStatusPending", new: "orderStatusNew", paid: "orderStatusPaid",
  printed: "orderStatusPrinted", shipped: "orderStatusShipped", done: "orderStatusDone",
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
  // Пошук/фільтр замовлень: текст (телефон / № / ім'я) + статус.
  const [orderQuery, setOrderQuery] = useState("");
  const [orderStatusFilter, setOrderStatusFilter] = useState<"" | OrderStatus>("");
  // uid користувача, чий рядок розгорнуто (деталі: моделі/замовлення/активність).
  const [expandedUser, setExpandedUser] = useState<string | null>(null);

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

  // Лише справжні замовлення (платіжні події type:"payment" — не замовлення).
  const realOrders = useMemo(() => orders.filter((o) => o.type !== "payment"), [orders]);

  // Застосовуємо пошук (телефон / № / ім'я / email) + фільтр статусу.
  const filteredOrders = useMemo(() => {
    const q = orderQuery.trim().toLowerCase();
    return realOrders.filter((o) => {
      if (orderStatusFilter) {
        const st = (ORDER_STATUSES as readonly string[]).includes(o.status) ? o.status : "new";
        if (st !== orderStatusFilter) return false;
      }
      if (!q) return true;
      const hay = [o.order_number, o.phone, o.name, o.user_email]
        .filter(Boolean).join(" ").toLowerCase();
      return hay.includes(q);
    });
  }, [realOrders, orderQuery, orderStatusFilter]);

  // Кількість замовлень на користувача (за email) — для розгортання рядка.
  const ordersByEmail = useMemo(() => {
    const map: Record<string, any[]> = {};
    for (const o of realOrders) {
      const key = (o.user_email || "").toLowerCase();
      if (!key) continue;
      (map[key] ||= []).push(o);
    }
    return map;
  }, [realOrders]);

  // Експорт відфільтрованих замовлень у CSV (клієнтський, без бекенда).
  const exportOrdersCsv = useCallback(() => {
    const cols = [
      ["order_number", ta("csvOrder")],
      ["created_at", ta("csvDate")],
      ["status", ta("csvStatus")],
      ["name", ta("csvName")],
      ["phone", ta("csvPhone")],
      ["user_email", ta("csvEmail")],
      ["product_type", ta("csvProduct")],
      ["est_price", ta("csvPrice")],
      ["delivery_method", ta("csvDelivery")],
    ] as const;
    const cell = (v: unknown) => {
      const s = v == null ? "" : String(v);
      return /[",\n;]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const header = cols.map(([, label]) => cell(label)).join(",");
    const rows = filteredOrders.map((o) =>
      cols.map(([key]) => cell(o[key])).join(","),
    );
    // BOM, щоб Excel коректно читав кирилицю.
    const csv = "﻿" + [header, ...rows].join("\r\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `monadruk_orders_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  }, [filteredOrders, ta]);

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
                      {/* ПІКСЕЛЬНІ висоти (не %!) — % резолвиться проти батька з
                          визначеною висотою і капризно виходив 0 (порожній графік
                          повторювався). Px + число НАД баром = завжди читабельно. */}
                      <div className="flex items-end gap-1.5" style={{ minHeight: 132 }}>
                        {(() => {
                          const max = Math.max(...stats.byDay.map((x: any) => x.pageviews || 0), 1);
                          return stats.byDay.map((d: any) => {
                            const pv = d.pageviews || 0;
                            const px = pv ? Math.max(Math.round((pv / max) * 108), 4) : 0;
                            return (
                              <div key={d.day} className="flex flex-1 flex-col items-center justify-end" title={`${d.day}: ${pv} переглядів`}>
                                <span className="mb-0.5 text-[9px] font-bold text-ink-2">{pv || ""}</span>
                                <div className="w-full rounded-t bg-forest/80" style={{ height: px }} />
                              </div>
                            );
                          });
                        })()}
                      </div>
                      <div className="mt-1.5 flex justify-between text-[10px] text-ink-3">
                        <span>{stats.byDay[0]?.day}</span><span>{stats.byDay[stats.byDay.length - 1]?.day}</span>
                      </div>
                    </div>
                  )}

                  {stats.funnel?.length > 0 && <Funnel funnel={stats.funnel} />}

                  {stats.guided && <GuidedFunnel g={stats.guided} />}

                  <div className="mt-5 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                    <StatList title="Країни" rows={stats.byCountry} />
                    <StatList title="Топ сторінок" rows={stats.topPaths} />
                    <StatList title="Події" rows={stats.topEvents} />
                    <StatList title="Мови" rows={stats.byLocale} />
                  </div>

                  <div className="mt-4 grid gap-4 md:grid-cols-2">
                    <StatList title="Звідки прийшли (реферери)" rows={stats.topRefs} />
                    <StatList title="Топ кліків (де тикали)" rows={prettyClicks(stats.topClicks)} />
                  </div>

                  {Array.isArray(stats.recentVisitors) && stats.recentVisitors.length > 0 && (
                    <RecentVisits visitors={stats.recentVisitors} />
                  )}

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
              {/* Пошук / фільтр / експорт CSV */}
              <div className="flex flex-wrap items-center gap-2">
                <div className="relative min-w-[220px] flex-1">
                  <Search size={15} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink-3" />
                  <input
                    value={orderQuery}
                    onChange={(e) => setOrderQuery(e.target.value)}
                    placeholder={ta("searchPlaceholder")}
                    aria-label={ta("searchPlaceholder")}
                    className="w-full rounded-full border border-line bg-paper py-2 pl-9 pr-3 text-[13px] text-ink placeholder:text-ink-3 focus:outline-none focus:ring-2 focus:ring-forest/30"
                  />
                </div>
                <select
                  value={orderStatusFilter}
                  onChange={(e) => setOrderStatusFilter(e.target.value as "" | OrderStatus)}
                  aria-label={ta("filterStatus")}
                  className="rounded-full border border-line bg-paper px-3 py-2 text-[13px] font-semibold text-ink-2"
                >
                  <option value="">{ta("filterAllStatuses")}</option>
                  {ORDER_STATUSES.map((s) => (
                    <option key={s} value={s}>{ts(ORDER_STATUS_KEYS[s])}</option>
                  ))}
                </select>
                <button
                  onClick={exportOrdersCsv}
                  disabled={filteredOrders.length === 0}
                  className="inline-flex items-center gap-2 rounded-full border border-line px-4 py-2 text-[13px] font-semibold text-ink-2 hover:bg-bg-2 disabled:opacity-50"
                >
                  <Download size={15} /> {ta("exportCsv")}
                </button>
              </div>
              <div className="text-[12px] text-ink-3">{ta("ordersShown", { shown: filteredOrders.length, total: realOrders.length })}</div>

              {realOrders.length === 0 ? <p className="text-ink-3">Поки немає замовлень.</p>
                : filteredOrders.length === 0 ? <p className="text-ink-3">{ta("noMatchingOrders")}</p>
                : filteredOrders.map((o, i) => {
                const status: OrderStatus = (ORDER_STATUSES as readonly string[]).includes(o.status) ? o.status : "new";
                const saving = savingStatus === String(o.order_number);
                const deliveryParts = [o.delivery_country, o.delivery_city, o.delivery_branch, o.delivery_address].filter(Boolean).join(", ");
                return (
                  <div key={`${o.order_number}-${i}`} className="rounded-[14px] border border-line bg-paper p-4">
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
                  <tr>
                    <th className="py-2 pl-1 pr-2"><span className="sr-only">{ta("expand")}</span></th>
                    <th className="py-2">Email</th><th className="py-2">Завантажень</th><th className="py-2">Моделей</th><th className="py-2">Дата</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((u) => {
                    const open = expandedUser === u.uid;
                    const userOrders = ordersByEmail[(u.email || "").toLowerCase()] || [];
                    // Остання активність: новіше з реєстрації та останнього замовлення.
                    const lastOrderTs = userOrders.reduce((mx: number, o: any) => {
                      const t = o.created_at ? Date.parse(o.created_at) : NaN;
                      return Number.isFinite(t) ? Math.max(mx, t) : mx;
                    }, 0);
                    const regTs = u.created_at ? u.created_at * 1000 : 0;
                    const lastActivity = Math.max(lastOrderTs, regTs);
                    return (
                      <Fragment key={u.uid}>
                        <tr
                          className="cursor-pointer border-t border-line hover:bg-bg-2"
                          onClick={() => setExpandedUser(open ? null : u.uid)}
                        >
                          <td className="py-2 pl-1 pr-2 text-ink-3">
                            <button
                              type="button"
                              aria-label={open ? ta("collapse") : ta("expand")}
                              aria-expanded={open}
                              className="inline-flex items-center justify-center rounded p-0.5 hover:text-ink"
                              onClick={(e) => { e.stopPropagation(); setExpandedUser(open ? null : u.uid); }}
                            >
                              {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                            </button>
                          </td>
                          <td className="py-2 text-ink">{u.email || u.uid.slice(0, 8)}</td>
                          <td className="py-2 text-ink-2">{u.downloads}</td>
                          <td className="py-2 text-ink-2">{u.models}</td>
                          <td className="py-2 text-ink-3">{u.created_at ? new Date(u.created_at * 1000).toLocaleDateString("uk") : ""}</td>
                        </tr>
                        {open && (
                          <tr className="border-t border-line bg-bg-2/40">
                            <td />
                            <td colSpan={4} className="py-3 pr-2">
                              <div className="grid gap-3 sm:grid-cols-3">
                                <div className="rounded-[12px] border border-line bg-paper p-3">
                                  <div className="text-[11px] uppercase tracking-wide text-ink-3">{ta("detailModels")}</div>
                                  <div className="mt-0.5 font-serif text-[20px] text-ink">{u.models}</div>
                                </div>
                                <div className="rounded-[12px] border border-line bg-paper p-3">
                                  <div className="text-[11px] uppercase tracking-wide text-ink-3">{ta("detailOrders")}</div>
                                  <div className="mt-0.5 font-serif text-[20px] text-ink">{userOrders.length}</div>
                                </div>
                                <div className="rounded-[12px] border border-line bg-paper p-3">
                                  <div className="text-[11px] uppercase tracking-wide text-ink-3">{ta("detailLastActivity")}</div>
                                  <div className="mt-0.5 text-[14px] font-semibold text-ink">
                                    {lastActivity ? new Date(lastActivity).toLocaleDateString("uk") : "—"}
                                  </div>
                                </div>
                              </div>
                              <div className="mt-2 text-[12px] text-ink-3">
                                {ta("detailDownloads")}: <span className="font-semibold text-ink-2">{u.downloads}</span>
                                {" · "}UID: <span className="font-mono text-ink-2">{u.uid.slice(0, 12)}</span>
                              </div>
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    );
                  })}
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
  paid: "Оплатили (LiqPay)",
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

const GUIDED_LABELS: Record<string, string> = {
  pick: "Обрали сценарій (крок 1)",
  step2: "Дійшли до кроку 2 (місце/напис)",
  generate: "Натиснули «Готово» (генерація)",
  order_open: "Відкрили форму замовлення",
};

/** Guided-воронка: де саме люди відвалюються всередині 2-крокового флоу
 *  (/create і /keychains). Дані — за обраний період (periodDays), на відміну
 *  від класичної воронки вище (вона за весь час). */
function GuidedFunnel({ g }: { g: any }) {
  const steps: { step: string; count: number; pct: number | null }[] = Array.isArray(g.steps) ? g.steps : [];
  const rows = (r: any): [string, number][] => (Array.isArray(r) ? r : []).map((x: any) => [String(x[0]), Number(x[1]) || 0]);
  const picks = rows(g.picksByScenario);
  const modes = rows(g.modeSwitch);
  const quotaAt = rows(g.quotaBlock?.byAt);
  const dash = (n: number) => (n > 0 ? n : "—");
  // Що конкретно обирають у guided-флоу (старий бекенд без choices — просто ховаємо блок).
  const choices = g.choices as
    | { sizes?: [number, number][]; places?: [string, number][]; homeMarked?: number; shares?: number; downloads?: number; orderClicks?: number; results?: { ok?: number; fail?: number } }
    | undefined;
  const sizeRows: [string, number][] = (choices?.sizes || []).map(([mm, n]) => [`${mm} мм`, Number(n) || 0]);
  const placeRows = rows(choices?.places);
  return (
    <div className="mt-5 rounded-[14px] border border-line bg-paper p-4">
      <div className="mb-1 text-[13px] font-semibold text-ink-2">Guided-воронка (2 кроки)</div>
      <div className="mb-3 text-[11px] text-ink-3">
        Новий спрощений флоу конструкторів за останні {g.periodDays ?? 30} днів. % = конверсія з попереднього кроку.
      </div>
      <table className="w-full text-[12px]">
        <tbody>
          {steps.map((s, i) => (
            <tr key={s.step} className="border-b border-line/60 last:border-0">
              <td className="py-1.5 text-ink-2">{i + 1}. {GUIDED_LABELS[s.step] || s.step}</td>
              <td className="py-1.5 text-right font-semibold text-ink">{s.count ?? 0}</td>
              <td className="w-16 py-1.5 text-right text-ink-3">
                {i === 0 || s.pct === null || s.pct === undefined ? "—" : `${s.pct}%`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <div className="rounded-lg border border-line bg-bg-2 px-3 py-2 text-[12px]">
          <div className="mb-1 font-semibold text-ink-2">Генерації</div>
          <div className="text-ink-3">Обрали своє місце: <b className="text-ink">{dash(g.generate?.placePicked || 0)}</b></div>
          <div className="text-ink-3">На дефолтному місці: <b className="text-ink">{dash(g.generate?.placeDefault || 0)}</b></div>
        </div>
        <div className="rounded-lg border border-line bg-bg-2 px-3 py-2 text-[12px]">
          <div className="mb-1 font-semibold text-ink-2">Тертя</div>
          <div className="text-ink-3">Уперлись у ліміт (квота): <b className="text-ink">{dash(g.quotaBlock?.total || 0)}</b>
            {quotaAt.length > 0 && <span className="ml-1">({quotaAt.map(([k, n]) => `${k}: ${n}`).join(", ")})</span>}
          </div>
          <div className="text-ink-3">Довге очікування файлу: <b className="text-ink">{dash(g.downloadWait || 0)}</b></div>
          <div className="text-ink-3">Перейшли в розширений режим: <b className="text-ink">{dash(modes.reduce((a, m) => a + m[1], 0))}</b></div>
        </div>
      </div>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <StatList title="Сценарії (що обирають)" rows={picks} />
        <StatList title="Перемикання режиму" rows={modes} />
      </div>
      {choices && (
        <>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <StatList title="Розміри" rows={sizeRows} />
            <StatList title="Місця" rows={placeRows} />
          </div>
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-5">
            {[
              ["Мій дім позначено", choices.homeMarked],
              ["Поділились", choices.shares],
              ["Завантажили", choices.downloads],
              ["Клік Замовити", choices.orderClicks],
              ["Генерацій ✓ / ✗", `${choices.results?.ok ?? 0} / ${choices.results?.fail ?? 0}`],
            ].map(([label, val]) => (
              <div key={label as string} className="rounded-lg border border-line bg-bg-2 px-3 py-2 text-center">
                <div className="font-serif text-[18px] text-ink">{val ?? 0}</div>
                <div className="text-[10.5px] text-ink-3">{label as string}</div>
              </div>
            ))}
          </div>
        </>
      )}
      {g.byDevice == null && (
        <p className="mt-2 text-[10px] text-ink-3">Розбивки «мобільний / комп'ютер» немає: аналітика не зберігає User-Agent.</p>
      )}
    </div>
  );
}

// Мітки для рядків таймлайну одного відвідувача (timeline: [{t, e, p}]).
// Хардкод, як решта цього адмін-файлу (не проходить через next-intl).
function timelineEventLabel(e: string, p: Record<string, unknown> | undefined): string {
  const P = p || {};
  const s = (k: string) => (P[k] == null ? "" : String(P[k]));
  switch (e) {
    case "pageview": return s("path") ? `Сторінка ${s("path")}` : "Перегляд";
    case "guided_pick": return `Обрав сценарій: ${s("scenario")}`;
    case "guided_step": return `Крок ${s("step")}`;
    case "guided_size": return `Розмір: ${s("sizeMm")} мм`;
    case "guided_place": return `Місце: ${s("place")}`;
    case "guided_home": return P.action === "clear" ? "Мій дім: очистив" : "Мій дім: позначив";
    case "guided_generate": return `Запустив генерацію (${s("scenario")}, ${s("sizeMm")} мм, ${s("place")})`;
    // ПАСТКА: /api/track зберігає props як РЯДКИ → "False" truthy; порівнюємо текстом.
    case "guided_result": return ["True", "true", "1"].includes(s("ok")) ? `Готово ✓ ${s("elapsedS")} с` : `Помилка: ${s("reason")}`;
    case "guided_share": return "Поділився 3D";
    case "guided_download": return "Завантажив файл";
    case "guided_order_click": return `Клік «Замовити друк» (${s("priceUah")} ₴)`;
    case "funnel": return `Воронка: ${s("step")}`;
    case "mode_switch": return `Режим: ${s("mode")}`;
    case "quota_block": return `Ліміт завантажень (${s("at")})`;
    case "download_model": return "Завантаження моделі";
    case "download_fail": return `Збій завантаження файлу: ${s("msg")}`;
    case "order_paid_confirmed": return "Оплата підтверджена";
    case "click": return `Клік: ${s("el")}`;
    default:
      if (e.startsWith("maket_")) return `Макет: ${e}`;
      return e;
  }
}

function RecentVisits({ visitors }: { visitors: any[] }) {
  const [expanded, setExpanded] = useState<number | null>(null);
  const fmt = (iso: string) => {
    try { return new Date(iso).toLocaleString("uk-UA", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }); }
    catch { return (iso || "").slice(5, 16).replace("T", " "); }
  };
  const hm = (iso: string) => {
    try { return new Date(iso).toLocaleTimeString("uk-UA", { hour: "2-digit", minute: "2-digit" }); }
    catch { return (iso || "").slice(11, 16); }
  };
  // HH:MM:SS для рядків таймлайну (детальніше, ніж hm() у зведеному рядку).
  const hms = (iso: string) => {
    try { return new Date(iso).toLocaleTimeString("uk-UA", { hour: "2-digit", minute: "2-digit", second: "2-digit" }); }
    catch { return (iso || "").slice(11, 19); }
  };
  // Час на сайті у людському вигляді: «12 с», «3 хв 20 с», «1 год 05 хв».
  const dur = (s: number) => {
    s = Math.max(0, Math.round(s || 0));
    if (s < 60) return `${s} с`;
    const m = Math.floor(s / 60), ss = s % 60;
    if (m < 60) return ss ? `${m} хв ${ss} с` : `${m} хв`;
    const h = Math.floor(m / 60), mm = m % 60;
    return `${h} год ${String(mm).padStart(2, "0")} хв`;
  };
  // Українська множина: 1 захід / 2-4 заходи / 5+ заходів.
  const plural = (n: number, one: string, few: string, many: string) => {
    const m10 = n % 10, m100 = n % 100;
    if (m10 === 1 && m100 !== 11) return one;
    if (m10 >= 2 && m10 <= 4 && (m100 < 10 || m100 >= 20)) return few;
    return many;
  };
  const flag = (cc: string) => {
    if (!cc || cc.length !== 2) return "🌐";
    try { return String.fromCodePoint(...[...cc.toUpperCase()].map((c) => 0x1f1e6 + c.charCodeAt(0) - 65)); }
    catch { return cc; }
  };
  return (
    <div className="mt-5 rounded-[14px] border border-line bg-paper p-4">
      <div className="mb-1 text-[13px] font-semibold text-ink-2">Останні візити (анонімні)</div>
      <div className="mb-3 text-[11px] text-ink-3">Кожен рядок = ОДИН відвідувач (без cookie/IP). Видно: звідки прийшов, з якої країни, <b>скільки часу був на сайті</b>, які сторінки дивився та коли.</div>
      <div className="space-y-1.5">
        {visitors.map((v, i) => {
          const timeline: Array<{ t: string; e: string; p?: Record<string, unknown> }> = Array.isArray(v.timeline) ? v.timeline : [];
          const open = expanded === i;
          const canExpand = timeline.length > 0;
          return (
            <div key={(v.id || "") + i} className="rounded-lg border border-line bg-bg-2 px-3 py-2 text-[12px]">
              <div
                className={`flex flex-col gap-1 sm:flex-row sm:items-start sm:gap-3 ${canExpand ? "cursor-pointer" : ""}`}
                onClick={() => canExpand && setExpanded(open ? null : i)}
              >
                <div className="flex items-center gap-2 sm:w-36 sm:shrink-0">
                  {canExpand ? (
                    <button
                      type="button"
                      aria-label={open ? "Згорнути" : "Розгорнути"}
                      aria-expanded={open}
                      className="inline-flex items-center justify-center rounded p-0.5 text-ink-3 hover:text-ink"
                      onClick={(e) => { e.stopPropagation(); setExpanded(open ? null : i); }}
                    >
                      {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                    </button>
                  ) : null}
                  <span className="text-base leading-none">{flag(v.cc)}</span>
                  <span className="font-semibold text-ink-2">{v.cc}</span>
                  <span className="font-mono text-[10px] text-ink-3">#{v.id}</span>
                </div>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-ink-2"><span className="text-ink-3">звідки:</span> <b>{v.ref}</b></div>
                  <div className="truncate text-[11px] text-ink-3">{(v.paths || []).join("  ›  ") || "—"}</div>
                </div>
                <div className="shrink-0 text-left text-[11px] sm:w-44 sm:text-right">
                  <div className="font-semibold text-forest">⏱ був {dur(v.duration)}</div>
                  <div className="text-ink-3">
                    {v.events} {plural(v.events, "дія", "дії", "дій")}
                    {v.sessions > 1 ? ` · ${v.sessions} ${plural(v.sessions, "захід", "заходи", "заходів")}` : ""}
                  </div>
                  <div className="text-ink-3">{fmt(v.first)} → {hm(v.last)}</div>
                </div>
              </div>
              {open && canExpand && (
                <div className="mt-2 space-y-1 border-t border-line/60 pt-2">
                  {timeline.map((row, j) => (
                    <div key={j} className="flex items-baseline gap-2 text-[11px]">
                      <span className="shrink-0 font-mono text-ink-3">{hms(row.t)}</span>
                      <span className="text-ink-2">{timelineEventLabel(row.e, row.p)}</span>
                    </div>
                  ))}
                </div>
              )}
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
