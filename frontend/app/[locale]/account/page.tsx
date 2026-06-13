"use client";
export const dynamic = "force-dynamic";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { ArrowLeft, Box, Download, Loader2, LogOut, ShieldCheck, Map as MapIcon, KeyRound, CheckCircle2 } from "lucide-react";
import { useAuth } from "@/components/AuthProvider";
import { gatedDownload } from "@/lib/download";
import { listGrids, deleteGrid, type CityGrid } from "@/lib/grids";
import { OrderDialog } from "@/components/OrderDialog";
import { ShoppingBag } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

interface Quota { downloads: number; limit: number; remaining: number; is_admin: boolean; can_download: boolean }
interface AccModel { task_id: string; title?: string; city?: string; product_type?: string; download_url?: string; ts?: number; preview?: string }
interface AccOrder {
  order_number?: string | number; created_at?: string; status?: string; product_type?: string;
  est_price?: string; delivery_country?: string; delivery_city?: string;
  summary?: { city?: string; district?: string; label?: string; size?: string };
}

const ORDER_STATUS_LABELS: Record<string, string> = {
  new: "Прийнято", paid: "Оплачено", printed: "Надруковано", shipped: "Відправлено", done: "Виконано",
};

export default function AccountPage() {
  const { user, loading, configured, signIn, signOut, getIdToken } = useAuth();
  const [quota, setQuota] = useState<Quota | null>(null);
  const [models, setModels] = useState<AccModel[]>([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [grids, setGrids] = useState<CityGrid[]>([]);
  const [orders, setOrders] = useState<AccOrder[]>([]);
  // Замовлення друку з раніше згенерованої моделі (генеруй зараз — замов потім).
  const [orderModel, setOrderModel] = useState<AccModel | null>(null);
  // Safety: ніколи не лишаємо вічний спінер. Якщо Firebase не відповів за 3.5с
  // (повільний клієнт / не гідратувалось) — показуємо екран входу, а не крутилку.
  const [gracePassed, setGracePassed] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setGracePassed(true), 3500);
    return () => clearTimeout(t);
  }, []);

  const load = useCallback(async () => {
    const token = await getIdToken();
    if (!token) return;
    try {
      const [q, m, g, o] = await Promise.all([
        fetch(`${API_BASE}/api/account/quota`, { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json()),
        fetch(`${API_BASE}/api/account/models`, { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json()),
        listGrids(token),
        fetch(`${API_BASE}/api/account/orders`, { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json()).catch(() => ({ orders: [] })),
      ]);
      setQuota(q.quota); setModels(m.models || []); setGrids(g || []); setOrders(o.orders || []);
    } catch {/* ignore */}
  }, [getIdToken]);

  const removeGrid = useCallback(async (id: string) => {
    const token = await getIdToken();
    if (await deleteGrid(token, id)) setGrids((gs) => gs.filter((x) => x.id !== id));
  }, [getIdToken]);

  useEffect(() => { if (user) load(); }, [user, load]);

  const download = async (m: AccModel) => {
    setBusy(true); setNotice(null);
    const res = await gatedDownload({
      taskId: m.task_id, downloadUrl: m.download_url,
      meta: { title: m.title, city: m.city, product_type: (m.product_type as any) || "map" },
      getIdToken, openLogin: signIn,
      onLimit: () => setNotice("Вичерпано 5 безкоштовних завантажень. Натисніть «Звʼязатися» внизу — і ми домовимось про друк/оплату."),
    });
    if (res.status === "ok") setQuota(res.quota);
    setBusy(false);
  };

  return (
    <div className="mx-auto min-h-[100dvh] max-w-[1100px] px-5 py-8 lg:px-8">
      <Link href="/create" className="mb-6 inline-flex items-center gap-1.5 text-[13px] font-semibold text-ink-2 hover:text-ink">
        <ArrowLeft size={15} /> Конструктор
      </Link>

      <div className="mb-2 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-serif text-[clamp(28px,4vw,44px)] text-ink">Мій кабінет</h1>
          <p className="mt-1 text-[14px] text-ink-2">Історія моделей і завантаження.</p>
        </div>
        {user && (
          <button onClick={() => signOut()} className="inline-flex items-center gap-2 rounded-full border border-line px-4 py-2 text-sm font-semibold text-ink-2 hover:bg-bg-2">
            <LogOut size={15} /> Вийти
          </button>
        )}
      </div>

      {!configured && (
        <div className="mt-4 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          Firebase ще не налаштований.
        </div>
      )}

      {/* Not logged in — гарний екран входу + коротко про сайт + плюси (один екран) */}
      {configured && !user && (loading && !gracePassed ? (
        <div className="mt-16 flex justify-center"><Loader2 className="animate-spin text-forest" /></div>
      ) : (
        <div className="mt-6 grid items-stretch gap-4 lg:grid-cols-2">
          {/* Ліворуч: вхід */}
          <div className="flex flex-col justify-center rounded-[24px] border border-line bg-paper p-6 text-center sm:p-8">
            <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-forest/10 text-forest">
              <KeyRound size={22} />
            </div>
            <h2 className="font-serif text-2xl text-ink">Увійдіть у кабінет</h2>
            <p className="mx-auto mt-2 max-w-[360px] text-sm text-ink-2">
              Зберігаємо ваші моделі та сітки міста. <b>5 безкоштовних</b> завантажень повної 3MF-моделі.
            </p>
            <button onClick={signIn} className="mt-5 inline-flex items-center justify-center gap-2 rounded-full bg-forest px-5 py-3 text-sm font-bold text-white transition hover:opacity-90" style={{ background: "var(--forest,#2E4A3A)" }}>
              Увійти / Зареєструватися
            </button>
            <p className="mt-3 text-[12px] text-ink-3">Email, телефон або Google — за кілька секунд.</p>
            <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
              <Link href="/create" className="inline-flex min-h-[44px] items-center gap-1.5 rounded-full border border-line px-4 text-sm font-semibold text-ink-2 hover:text-ink">
                <MapIcon size={15} /> Створити мапу
              </Link>
              <Link href="/keychains" className="inline-flex min-h-[44px] items-center gap-1.5 rounded-full border px-4 text-sm font-semibold" style={{ borderColor: "rgba(142,107,61,0.4)", color: "var(--bronze,#8E6B3D)" }}>
                <KeyRound size={15} /> Брелок
              </Link>
            </div>
          </div>
          {/* Праворуч: коротко про сайт + плюси */}
          <div className="rounded-[24px] border border-line bg-paper p-6 sm:p-8">
            <h3 className="font-serif text-xl text-ink">monadruk — 3D-мапи й брелки</h3>
            <p className="mt-1 text-sm text-ink-2">Перетвори будь-яке місце Землі на 3D-сувенір. Усе у браузері, готове до друку.</p>
            <ul className="mt-4 space-y-2.5">
              {[
                "Будь-яке місто світу — 3D-мапа за ~3 хвилини",
                "Брелки-жетони з вашим районом і написом",
                "5 безкоштовних завантажень 3MF (повна модель)",
                "Готово до 3D-друку (FDM 0.4 мм) — або замовте друк у нас",
                "Історія моделей і збережені сітки міста в кабінеті",
              ].map((b) => (
                <li key={b} className="flex items-start gap-2.5 text-sm text-ink-2">
                  <CheckCircle2 size={18} className="mt-0.5 shrink-0 text-forest" />
                  <span>{b}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      ))}

      {/* Logged in */}
      {user && (
        <>
          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            <div className="rounded-[18px] border border-line bg-paper p-5">
              <div className="text-[11px] uppercase tracking-wide text-ink-3">Акаунт</div>
              <div className="mt-1 truncate text-[15px] font-semibold text-ink">{user.email || user.phoneNumber || "—"}</div>
            </div>
            <div className="rounded-[18px] border border-line bg-paper p-5">
              <div className="text-[11px] uppercase tracking-wide text-ink-3">Завантаження</div>
              <div className="mt-1 text-[15px] font-semibold text-ink">
                {quota ? (quota.is_admin ? "Безліміт" : `${quota.downloads} / ${quota.limit}`) : "…"}
              </div>
            </div>
            <div className="rounded-[18px] border border-line bg-paper p-5">
              <div className="text-[11px] uppercase tracking-wide text-ink-3">Статус</div>
              <div className="mt-1 inline-flex items-center gap-1.5 text-[15px] font-semibold text-ink">
                {quota?.is_admin ? <><ShieldCheck size={16} className="text-forest" /> Адмін</> : "Стандарт"}
              </div>
            </div>
          </div>

          {notice && <div className="mt-4 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">{notice}</div>}

          {orders.length > 0 && (
            <>
              <h3 className="mb-3 mt-8 font-serif text-xl text-ink">Мої замовлення</h3>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {orders.map((o, i) => (
                  <div key={`${o.order_number}-${i}`} className="rounded-[16px] border border-line bg-paper p-4">
                    <div className="flex items-center justify-between gap-2">
                      <div className="font-serif text-[17px] text-ink">#{o.order_number}</div>
                      <span className="rounded-full bg-[rgba(15,118,110,0.10)] px-2.5 py-1 text-[11px] font-semibold text-forest">
                        {ORDER_STATUS_LABELS[o.status || "new"] || o.status}
                      </span>
                    </div>
                    <div className="mt-1 text-[12px] text-ink-3">
                      {o.product_type === "keychain" ? "Брелок" : "3D-мапа"}
                      {o.summary?.size ? ` · ${o.summary.size}` : ""}
                      {o.created_at ? ` · ${new Date(o.created_at).toLocaleDateString("uk")}` : ""}
                    </div>
                    {(o.summary?.city || o.summary?.label) && (
                      <div className="mt-1 truncate text-[12px] text-ink-2">
                        {[o.summary?.city, o.summary?.district, o.summary?.label].filter(Boolean).join(" · ")}
                      </div>
                    )}
                    <div className="mt-2 text-[13px] font-semibold text-ink">{o.est_price || ""}</div>
                  </div>
                ))}
              </div>
            </>
          )}

          <h3 className="mb-3 mt-8 font-serif text-xl text-ink">Мої моделі</h3>
          {models.length === 0 ? (
            <div className="rounded-[18px] border border-dashed border-line bg-paper px-4 py-10 text-center text-sm text-ink-3">
              Поки немає моделей. <Link href="/create" className="font-semibold text-forest underline-offset-2 hover:underline">Створити першу →</Link>
            </div>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {models.map((m) => (
                <div key={m.task_id} className="flex flex-col overflow-hidden rounded-[16px] border border-line bg-paper">
                  {/* Превʼю того, що було згенеровано (зберігається при завантаженні) */}
                  <div className="flex aspect-[4/3] items-center justify-center overflow-hidden bg-[#0b1020]">
                    {m.preview ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={m.preview} alt={m.title || m.city || "превʼю"} className="h-full w-full object-contain" loading="lazy" />
                    ) : (
                      <Box size={26} className="text-white/30" />
                    )}
                  </div>
                  <div className="p-4 pt-3">
                  <div className="font-serif text-[17px] text-ink">{m.title || m.city || (m.product_type === "keychain" ? "Брелок" : "3D-мапа")}</div>
                  <div className="text-[12px] text-ink-3">{m.product_type === "keychain" ? "Брелок" : "Мапа"}{m.ts ? ` · ${new Date(m.ts * 1000).toLocaleDateString("uk")}` : ""}</div>
                  <div className="mt-3 flex gap-2">
                    {/* Замовити друк цієї моделі (управління+покупка: генеруй зараз, замов потім) */}
                    <button onClick={() => setOrderModel(m)}
                      className="inline-flex min-h-10 flex-1 items-center justify-center gap-2 rounded-full px-4 py-2 text-sm font-bold text-white"
                      style={{ background: "var(--bronze,#8E6B3D)" }}>
                      <ShoppingBag size={15} /> Замовити друк
                    </button>
                    <button onClick={() => download(m)} disabled={busy}
                      title="Завантажити 3MF"
                      className="inline-flex min-h-10 items-center justify-center gap-2 rounded-full border border-line px-3 py-2 text-sm font-semibold text-ink-2 hover:bg-bg-2 disabled:opacity-60">
                      {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download size={15} />}
                    </button>
                  </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          <h3 className="mb-3 mt-10 font-serif text-xl text-ink">Мої сітки міста</h3>
          {grids.length === 0 ? (
            <div className="rounded-[18px] border border-dashed border-line bg-paper px-4 py-10 text-center text-sm text-ink-3">
              Поки немає збережених сіток. У конструкторі ввімкніть «Сітка зон», створіть сітку й натисніть «Зберегти сітку» — потім зможете догенерувати сусідні комірки.
            </div>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {grids.map((g) => (
                <div key={g.id} className="flex flex-col rounded-[16px] border border-line bg-paper p-4">
                  <div className="font-serif text-[17px] text-ink">{g.name || g.city || "Сітка"}</div>
                  <div className="text-[12px] text-ink-3">
                    {g.grid_type === "square" ? "Квадрати" : g.grid_type === "circle" ? "Кола" : "Гексагони"}
                    {g.hex_size_m ? ` · ${Math.round(g.hex_size_m)} м` : ""}
                    {` · ${(g.cells || []).length} комірок`}
                    {g.updated_at ? ` · ${new Date(g.updated_at * 1000).toLocaleDateString("uk")}` : ""}
                  </div>
                  <div className="mt-3 flex gap-2">
                    <Link href={`/create?grid=${g.id}`}
                      className="inline-flex min-h-10 flex-1 items-center justify-center gap-2 rounded-full bg-forest px-4 py-2 text-sm font-semibold text-white">
                      Відкрити
                    </Link>
                    <button onClick={() => g.id && removeGrid(g.id)}
                      className="inline-flex min-h-10 items-center justify-center rounded-full border border-line px-3 text-sm font-semibold text-ink-2 hover:bg-bg-2">
                      Видалити
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Замовлення друку з картки збереженої моделі */}
      <OrderDialog
        open={!!orderModel}
        onClose={() => setOrderModel(null)}
        taskId={orderModel?.task_id ?? null}
        productType={(orderModel?.product_type as "map" | "keychain") || "map"}
        summary={{
          city: orderModel?.city,
          label: orderModel?.product_type === "keychain" ? orderModel?.title : undefined,
        }}
      />
    </div>
  );
}
