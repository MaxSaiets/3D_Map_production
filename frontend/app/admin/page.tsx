"use client";

import { useEffect, useState } from "react";
import { ArrowLeft, Loader2, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";

export default function AdminPage() {
  const [token, setToken] = useState("");
  const [orders, setOrders] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadOrders = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.getAdminOrders(token || undefined);
      setOrders(response.orders);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || "Не вдалося завантажити заявки.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadOrders();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <main className="min-h-[100dvh] bg-[#f4efe4] px-3 py-3 text-slate-900 lg:px-5">
      <div className="mx-auto max-w-7xl">
        <header className="flex flex-col gap-3 border border-black/10 bg-white/85 px-4 py-4 shadow-sm sm:flex-row sm:items-end sm:justify-between">
          <div>
            <a href="/" className="inline-flex items-center gap-2 text-sm font-semibold text-teal-700">
              <ArrowLeft size={16} />
              Назад до сайту
            </a>
            <h1 className="mt-2 text-2xl font-semibold tracking-tight">Адмінка заявок 3dMAP</h1>
          </div>
          <div className="flex gap-2">
            <input
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="Admin token"
              className="min-h-10 border border-black/10 px-3 text-sm outline-none focus:border-teal-700"
            />
            <button
              type="button"
              onClick={loadOrders}
              className="inline-flex min-h-10 items-center gap-2 bg-teal-700 px-4 text-sm font-semibold text-white"
            >
              {isLoading ? <Loader2 size={16} className="animate-spin" /> : <RefreshCw size={16} />}
              Оновити
            </button>
          </div>
        </header>

        {error && <div className="mt-3 border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">{error}</div>}

        <section className="mt-3 grid gap-3">
          {orders.length === 0 && !isLoading ? (
            <div className="border border-black/10 bg-white px-4 py-8 text-center text-sm text-slate-500">
              Заявок поки немає.
            </div>
          ) : (
            orders.map((order) => (
              <article key={order.id} className="border border-black/10 bg-white p-4 shadow-sm">
                <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                  <div>
                    <div className="text-xs font-semibold uppercase tracking-[0.16em] text-teal-700">{order.status}</div>
                    <h2 className="mt-1 text-lg font-semibold">{order.customer_name}</h2>
                    <div className="text-sm text-slate-600">{order.contact}</div>
                  </div>
                  <div className="text-sm text-slate-500">{order.created_at}</div>
                </div>
                <div className="mt-3 grid gap-2 text-sm sm:grid-cols-3">
                  <div className="bg-slate-50 p-3">
                    <b>Місто</b>
                    <br />
                    {order.city || "-"}
                  </div>
                  <div className="bg-slate-50 p-3">
                    <b>Preview</b>
                    <br />
                    {order.preview_id || "-"}
                  </div>
                  <div className="bg-slate-50 p-3">
                    <b>Площа</b>
                    <br />
                    {order.options?.metrics?.area_m2 ? `${order.options.metrics.area_m2} м2` : "-"}
                  </div>
                </div>
                {order.message && <p className="mt-3 whitespace-pre-wrap text-sm text-slate-700">{order.message}</p>}
                <details className="mt-3 text-sm">
                  <summary className="cursor-pointer font-semibold text-teal-700">Технічні дані</summary>
                  <pre className="mt-2 max-h-80 overflow-auto bg-slate-950 p-3 text-xs text-slate-100">
                    {JSON.stringify(order, null, 2)}
                  </pre>
                </details>
              </article>
            ))
          )}
        </section>
      </div>
    </main>
  );
}
