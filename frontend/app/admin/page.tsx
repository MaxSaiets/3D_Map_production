"use client";
export const dynamic = "force-dynamic";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { ArrowLeft, Loader2, Package, Users, RefreshCw } from "lucide-react";
import { useAuth } from "@/components/AuthProvider";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export default function AdminPage() {
  const { user, loading, configured, signIn, getIdToken } = useAuth();
  const [tab, setTab] = useState<"orders" | "users">("orders");
  const [orders, setOrders] = useState<any[]>([]);
  const [users, setUsers] = useState<any[]>([]);
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    const token = await getIdToken();
    if (!token) return;
    setBusy(true);
    try {
      const q = await fetch(`${API_BASE}/api/account/quota`, { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json());
      const admin = Boolean(q?.quota?.is_admin);
      setIsAdmin(admin);
      if (admin) {
        const [o, us] = await Promise.all([
          fetch(`${API_BASE}/api/admin/orders`, { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json()),
          fetch(`${API_BASE}/api/admin/users`, { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json()),
        ]);
        setOrders(o.orders || []); setUsers(us.users || []);
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
          <div className="mt-5 flex gap-2">
            <button onClick={() => setTab("orders")} className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold ${tab === "orders" ? "bg-forest text-white" : "border border-line text-ink-2"}`}><Package size={15} /> Замовлення ({orders.length})</button>
            <button onClick={() => setTab("users")} className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold ${tab === "users" ? "bg-forest text-white" : "border border-line text-ink-2"}`}><Users size={15} /> Користувачі ({users.length})</button>
          </div>

          {tab === "orders" && (
            <div className="mt-5 space-y-3">
              {orders.length === 0 ? <p className="text-ink-3">Поки немає замовлень.</p> : orders.map((o, i) => (
                <div key={i} className="rounded-[14px] border border-line bg-paper p-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="font-serif text-[17px] text-ink">#{o.order_number} · {o.name || "—"}</div>
                    <div className="text-[12px] text-ink-3">{o.created_at ? new Date(o.created_at).toLocaleString("uk") : ""}</div>
                  </div>
                  <div className="mt-1 text-[13px] text-ink-2">
                    📞 {o.phone || "—"} · {o.product_type === "keychain" ? "Брелок" : "Мапа"}
                    {o.summary?.district ? ` · ${o.summary.district}` : ""}{o.summary?.size ? ` · ${o.summary.size}` : ""}
                    {o.delivery_method ? ` · ${o.delivery_method} ${o.delivery_city || ""} ${o.delivery_branch || ""}` : ""}
                  </div>
                  {o.comment && <div className="mt-1 text-[13px] text-ink-3">💬 {o.comment}</div>}
                </div>
              ))}
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
