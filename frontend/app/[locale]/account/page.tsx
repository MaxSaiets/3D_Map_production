"use client";
export const dynamic = "force-dynamic";

import Link from "next/link";
import { Component, type ReactNode, useCallback, useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { ArrowLeft, Box, Download, Loader2, LogOut, ShieldCheck, Map as MapIcon, KeyRound, CheckCircle2, AlertTriangle, ChevronDown } from "lucide-react";
import { useAuth } from "@/components/AuthProvider";
import { gatedDownload } from "@/lib/download";
import { listGrids, deleteGrid, type CityGrid } from "@/lib/grids";
import { OrderDialog } from "@/components/OrderDialog";
import Model3DViewer from "@/components/Model3DViewer";
import { ShoppingBag } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

/**
 * Невеликий error boundary навколо живого 3D-вʼювера: якщо GLB не завантажився
 * (мережа / битий файл / WebGL недоступний) — показуємо плейсхолдер Box, а не
 * ламаємо весь кабінет. r3f кидає під час рендеру, тому потрібен саме boundary.
 */
class ModelErrorBoundary extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch() { /* проковтнуто — fallback уже показано */ }
  render() { return this.state.failed ? this.props.fallback : this.props.children; }
}

interface Quota { downloads: number; limit: number; remaining: number; is_admin: boolean; can_download: boolean }
interface AccModel { task_id: string; title?: string; city?: string; product_type?: string; download_url?: string; ts?: number; preview?: string }
interface AccOrder {
  order_number?: string | number; created_at?: string; status?: string; product_type?: string;
  est_price?: string; delivery_country?: string; delivery_city?: string;
  summary?: { city?: string; district?: string; label?: string; size?: string };
}

const ORDER_STATUS_KEYS: Record<string, string> = {
  new: "orderStatusNew", paid: "orderStatusPaid", printed: "orderStatusPrinted",
  shipped: "orderStatusShipped", done: "orderStatusDone",
};

export default function AccountPage() {
  const t = useTranslations("account");
  const { user, loading, configured, signIn, signOut, getIdToken } = useAuth();
  const [quota, setQuota] = useState<Quota | null>(null);
  const [models, setModels] = useState<AccModel[]>([]);
  const [busy, setBusy] = useState<string | null>(null); // task_id, що зараз качається
  const [notice, setNotice] = useState<string | null>(null);
  const [grids, setGrids] = useState<CityGrid[]>([]);
  const [orders, setOrders] = useState<AccOrder[]>([]);
  // Завантаження даних кабінету могло впасти (мережа / бекенд) — показуємо
  // дружній повтор, а не порожню сторінку.
  const [loadError, setLoadError] = useState(false);
  const [loadingData, setLoadingData] = useState(false);
  // task_id моделі, для якої відкрите меню вибору формату (3MF/STL).
  const [fmtMenu, setFmtMenu] = useState<string | null>(null);
  // Замовлення друку з раніше згенерованої моделі (генеруй зараз — замов потім).
  const [orderModel, setOrderModel] = useState<AccModel | null>(null);
  // Safety: ніколи не лишаємо вічний спінер. Якщо Firebase не відповів за 3.5с
  // (повільний клієнт / не гідратувалось) — показуємо екран входу, а не крутилку.
  const [gracePassed, setGracePassed] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setGracePassed(true), 3500);
    return () => clearTimeout(t);
  }, []);
  // Закриваємо меню вибору формату при кліку поза ним / Esc.
  useEffect(() => {
    if (!fmtMenu) return;
    const close = () => setFmtMenu(null);
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setFmtMenu(null); };
    window.addEventListener("click", close);
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("click", close); window.removeEventListener("keydown", onKey); };
  }, [fmtMenu]);

  const load = useCallback(async () => {
    const token = await getIdToken();
    if (!token) return;
    setLoadingData(true); setLoadError(false);
    // Квота й моделі — критичні (без них екран порожній): їх збій = показуємо
    // повтор. Сітки й замовлення другорядні — їх падіння не валить кабінет.
    const fetchJson = (path: string) =>
      fetch(`${API_BASE}/api/account/${path}`, { headers: { Authorization: `Bearer ${token}` } })
        .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); });
    try {
      const [q, m] = await Promise.all([fetchJson("quota"), fetchJson("models")]);
      setQuota(q.quota); setModels(m.models || []);
      // другорядні дані — м'який збій
      const [g, o] = await Promise.all([
        listGrids(token).catch(() => [] as CityGrid[]),
        fetchJson("orders").catch(() => ({ orders: [] })),
      ]);
      setGrids(g || []); setOrders(o.orders || []);
    } catch {
      setLoadError(true);
    } finally {
      setLoadingData(false);
    }
  }, [getIdToken]);

  const removeGrid = useCallback(async (id: string) => {
    const token = await getIdToken();
    if (await deleteGrid(token, id)) setGrids((gs) => gs.filter((x) => x.id !== id));
  }, [getIdToken]);

  useEffect(() => { if (user) load(); }, [user, load]);

  const download = async (m: AccModel, format: "3mf" | "stl" = "3mf") => {
    setFmtMenu(null); setBusy(m.task_id); setNotice(null);
    // Квота-гейт + реєстрація завантаження йдуть через спільний gatedDownload
    // (він віддає 3MF — головний формат). Для STL після проходження гейту
    // дотягуємо STL-файл з публічного format-ендпойнту (той самий task_id),
    // тож ліміт усе одно враховано рівно один раз.
    const res = await gatedDownload({
      taskId: m.task_id, downloadUrl: m.download_url,
      meta: { title: m.title, city: m.city, product_type: (m.product_type as any) || "map" },
      getIdToken, openLogin: signIn,
      onLimit: () => setNotice(t("limitNotice")),
    });
    if (res.status === "ok") {
      if (format === "stl") {
        try {
          const r = await fetch(`${API_BASE}/api/download/${m.task_id}?format=stl`);
          if (r.ok) {
            const blob = await r.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url; a.download = `${(m.title || m.city || "monadruk").replace(/[^\w.-]+/g, "_")}_${m.task_id.slice(0, 8)}.stl`;
            document.body.appendChild(a); a.click(); a.remove();
            URL.revokeObjectURL(url);
          } else {
            setNotice(t("stlUnavailable"));
          }
        } catch {
          setNotice(t("stlUnavailable"));
        }
      }
      // gatedDownload повертає лише {remaining} з X-Quota-Remaining header — НЕ перезаписуємо
      // повний quota-обʼєкт (інакше downloads/limit/is_admin стають undefined). Перечитуємо.
      load();
    }
    setBusy(null);
  };

  return (
    <div className="mx-auto min-h-[100dvh] max-w-[1100px] px-5 py-8 lg:px-8">
      <Link href="/create" className="mb-6 inline-flex items-center gap-1.5 text-[13px] font-semibold text-ink-2 hover:text-ink">
        <ArrowLeft size={15} /> {t("backToBuilder")}
      </Link>

      <div className="mb-2 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-serif text-[clamp(28px,4vw,44px)] text-ink">{t("title")}</h1>
          <p className="mt-1 text-[14px] text-ink-2">{t("subtitle")}</p>
        </div>
        {user && (
          <button onClick={() => signOut()} className="inline-flex items-center gap-2 rounded-full border border-line px-4 py-2 text-sm font-semibold text-ink-2 hover:bg-bg-2">
            <LogOut size={15} /> {t("signOut")}
          </button>
        )}
      </div>

      {!configured && (
        <div className="mt-4 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          {t("firebaseNotConfigured")}
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
            <h2 className="font-serif text-2xl text-ink">{t("loginHeading")}</h2>
            <p className="mx-auto mt-2 max-w-[360px] text-sm text-ink-2">
              {t.rich("loginPitch", { b: (chunks) => <b>{chunks}</b> })}
            </p>
            <button onClick={signIn} className="mt-5 inline-flex items-center justify-center gap-2 rounded-full bg-forest px-5 py-3 text-sm font-bold text-white transition hover:opacity-90" style={{ background: "var(--forest,#2E4A3A)" }}>
              {t("loginButton")}
            </button>
            <p className="mt-3 text-[12px] text-ink-3">{t("loginHint")}</p>
            <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
              <Link href="/create" className="inline-flex min-h-[44px] items-center gap-1.5 rounded-full border border-line px-4 text-sm font-semibold text-ink-2 hover:text-ink">
                <MapIcon size={15} /> {t("createMap")}
              </Link>
              <Link href="/keychains" className="inline-flex min-h-[44px] items-center gap-1.5 rounded-full border px-4 text-sm font-semibold" style={{ borderColor: "rgba(142,107,61,0.4)", color: "var(--bronze,#8E6B3D)" }}>
                <KeyRound size={15} /> {t("keychain")}
              </Link>
            </div>
          </div>
          {/* Праворуч: коротко про сайт + плюси */}
          <div className="rounded-[24px] border border-line bg-paper p-6 sm:p-8">
            <h3 className="font-serif text-xl text-ink">{t("aboutHeading")}</h3>
            <p className="mt-1 text-sm text-ink-2">{t("aboutText")}</p>
            <ul className="mt-4 space-y-2.5">
              {[
                t("bullet1"),
                t("bullet2"),
                t("bullet3"),
                t("bullet4"),
                t("bullet5"),
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
              <div className="text-[11px] uppercase tracking-wide text-ink-3">{t("accountLabel")}</div>
              <div className="mt-1 truncate text-[15px] font-semibold text-ink">{user.email || user.phoneNumber || "—"}</div>
            </div>
            <div className="rounded-[18px] border border-line bg-paper p-5">
              <div className="text-[11px] uppercase tracking-wide text-ink-3">{t("downloadsLabel")}</div>
              <div className="mt-1 text-[15px] font-semibold text-ink">
                {quota ? (quota.is_admin ? t("unlimited") : `${quota.downloads} / ${quota.limit}`) : "…"}
              </div>
            </div>
            <div className="rounded-[18px] border border-line bg-paper p-5">
              <div className="text-[11px] uppercase tracking-wide text-ink-3">{t("statusLabel")}</div>
              <div className="mt-1 inline-flex items-center gap-1.5 text-[15px] font-semibold text-ink">
                {quota?.is_admin ? <><ShieldCheck size={16} className="text-forest" /> {t("admin")}</> : t("standard")}
              </div>
            </div>
          </div>

          {notice && <div className="mt-4 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">{notice}</div>}

          {loadError && (
            <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
              <span className="inline-flex items-center gap-2"><AlertTriangle size={16} /> {t("loadError")}</span>
              <button onClick={load} disabled={loadingData}
                className="inline-flex items-center gap-2 rounded-full border border-amber-300 bg-white px-3 py-1.5 text-[13px] font-semibold text-amber-900 hover:bg-amber-100 disabled:opacity-60">
                {loadingData ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null} {t("retry")}
              </button>
            </div>
          )}

          {orders.length > 0 && (
            <>
              <h3 className="mb-3 mt-8 font-serif text-xl text-ink">{t("myOrders")}</h3>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {orders.map((o, i) => (
                  <div key={`${o.order_number}-${i}`} className="rounded-[16px] border border-line bg-paper p-4">
                    <div className="flex items-center justify-between gap-2">
                      <div className="font-serif text-[17px] text-ink">#{o.order_number}</div>
                      <span className="rounded-full bg-[rgba(15,118,110,0.10)] px-2.5 py-1 text-[11px] font-semibold text-forest">
                        {ORDER_STATUS_KEYS[o.status || "new"] ? t(ORDER_STATUS_KEYS[o.status || "new"]) : o.status}
                      </span>
                    </div>
                    <div className="mt-1 text-[12px] text-ink-3">
                      {o.product_type === "keychain" ? t("keychain") : t("map3d")}
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

          <h3 className="mb-3 mt-8 font-serif text-xl text-ink">{t("myModels")}</h3>
          {models.length === 0 ? (
            <div className="rounded-[18px] border border-dashed border-line bg-paper px-4 py-10 text-center text-sm text-ink-3">
              {t("noModels")} <Link href="/create" className="font-semibold text-forest underline-offset-2 hover:underline">{t("createFirst")}</Link>
            </div>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {models.map((m) => (
                <div key={m.task_id} className="flex flex-col overflow-hidden rounded-[16px] border border-line bg-paper">
                  {/* Превʼю того, що було згенеровано. Якщо знятого превʼю немає —
                      рендеримо ЖИВУ модель (GLB з бекенда) у мініатюрному вʼювері;
                      при збої завантаження error boundary показує плейсхолдер Box. */}
                  <div className="flex aspect-[4/3] items-center justify-center overflow-hidden bg-[#0b1020]">
                    {m.preview ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={m.preview} alt={m.title || m.city || t("previewAlt")} className="h-full w-full object-contain" loading="lazy" />
                    ) : (
                      <ModelErrorBoundary fallback={<Box size={26} className="text-white/30" />}>
                        <Model3DViewer
                          url={`${API_BASE}/api/download/${m.task_id}?format=glb`}
                          flat={m.product_type !== "keychain"}
                          height={150}
                          autoRotate
                        />
                      </ModelErrorBoundary>
                    )}
                  </div>
                  <div className="p-4 pt-3">
                  <div className="font-serif text-[17px] text-ink">{m.title || m.city || (m.product_type === "keychain" ? t("keychain") : t("map3d"))}</div>
                  {/* Деталі моделі: тип · дата · місто/локація */}
                  <dl className="mt-1 space-y-0.5 text-[12px] text-ink-3">
                    <div className="flex items-center gap-1.5">
                      <dt className="text-ink-3">{t("detailType")}:</dt>
                      <dd className="font-medium text-ink-2">{m.product_type === "keychain" ? t("keychain") : t("mapShort")}</dd>
                    </div>
                    {m.ts ? (
                      <div className="flex items-center gap-1.5">
                        <dt className="text-ink-3">{t("detailDate")}:</dt>
                        <dd className="font-medium text-ink-2">{new Date(m.ts * 1000).toLocaleDateString("uk")}</dd>
                      </div>
                    ) : null}
                    {m.city ? (
                      <div className="flex items-center gap-1.5">
                        <dt className="text-ink-3">{t("detailLocation")}:</dt>
                        <dd className="truncate font-medium text-ink-2">{m.city}</dd>
                      </div>
                    ) : null}
                  </dl>
                  <div className="mt-3 flex gap-2">
                    {/* Замовити друк цієї моделі (управління+покупка: генеруй зараз, замов потім) */}
                    <button onClick={() => setOrderModel(m)}
                      className="inline-flex min-h-10 flex-1 items-center justify-center gap-2 rounded-full px-4 py-2 text-sm font-bold text-white"
                      style={{ background: "var(--bronze,#8E6B3D)" }}>
                      <ShoppingBag size={15} /> {t("orderPrint")}
                    </button>
                    {/* Завантаження з вибором формату (3MF за замовч. або STL) */}
                    <div className="relative" onClick={(e) => e.stopPropagation()}>
                      <button
                        onClick={() => setFmtMenu((cur) => (cur === m.task_id ? null : m.task_id))}
                        disabled={busy === m.task_id}
                        title={t("downloadTitle")}
                        aria-label={t("downloadAria")}
                        aria-haspopup="menu"
                        aria-expanded={fmtMenu === m.task_id}
                        className="inline-flex min-h-10 items-center justify-center gap-1 rounded-full border border-line px-3 py-2 text-sm font-semibold text-ink-2 hover:bg-bg-2 disabled:opacity-60">
                        {busy === m.task_id ? <Loader2 className="h-4 w-4 animate-spin" /> : <><Download size={15} /><ChevronDown size={13} /></>}
                      </button>
                      {fmtMenu === m.task_id && busy !== m.task_id && (
                        <div role="menu" className="absolute right-0 z-10 mt-1 w-44 overflow-hidden rounded-xl border border-line bg-paper py-1 shadow-lg">
                          <button role="menuitem" onClick={() => download(m, "3mf")}
                            className="flex w-full items-center gap-2 px-3 py-2 text-left text-[13px] text-ink hover:bg-bg-2">
                            <Download size={14} className="text-ink-3" /> {t("downloadFormat3mf")}
                          </button>
                          <button role="menuitem" onClick={() => download(m, "stl")}
                            className="flex w-full items-center gap-2 px-3 py-2 text-left text-[13px] text-ink hover:bg-bg-2">
                            <Download size={14} className="text-ink-3" /> {t("downloadFormatStl")}
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          <h3 className="mb-3 mt-10 font-serif text-xl text-ink">{t("myGrids")}</h3>
          {grids.length === 0 ? (
            <div className="rounded-[18px] border border-dashed border-line bg-paper px-4 py-10 text-center text-sm text-ink-3">
              {t("noGrids")}
            </div>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {grids.map((g) => (
                <div key={g.id} className="flex flex-col rounded-[16px] border border-line bg-paper p-4">
                  <div className="font-serif text-[17px] text-ink">{g.name || g.city || t("gridFallback")}</div>
                  <div className="text-[12px] text-ink-3">
                    {g.grid_type === "square" ? t("gridSquare") : g.grid_type === "circle" ? t("gridCircle") : t("gridHex")}
                    {g.hex_size_m ? ` · ${t("gridMeters", { n: Math.round(g.hex_size_m) })}` : ""}
                    {` · ${t("gridCells", { n: (g.cells || []).length })}`}
                    {g.updated_at ? ` · ${new Date(g.updated_at * 1000).toLocaleDateString("uk")}` : ""}
                  </div>
                  <div className="mt-3 flex gap-2">
                    <Link href={`/create?grid=${g.id}`}
                      className="inline-flex min-h-10 flex-1 items-center justify-center gap-2 rounded-full bg-forest px-4 py-2 text-sm font-semibold text-white">
                      {t("open")}
                    </Link>
                    <button onClick={() => g.id && removeGrid(g.id)}
                      className="inline-flex min-h-10 items-center justify-center rounded-full border border-line px-3 text-sm font-semibold text-ink-2 hover:bg-bg-2">
                      {t("delete")}
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
