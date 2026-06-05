"use client";
export const dynamic = "force-dynamic";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { ArrowLeft, Box, Download, Loader2, LogOut, ShieldCheck, Sparkles } from "lucide-react";
import { useAuth } from "@/components/AuthProvider";
import { gatedDownload } from "@/lib/download";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

interface Quota { downloads: number; limit: number; remaining: number; is_admin: boolean; can_download: boolean }
interface AccModel { task_id: string; title?: string; city?: string; product_type?: string; download_url?: string; ts?: number }

export default function AccountPage() {
  const { user, loading, configured, signIn, signOut, getIdToken } = useAuth();
  const [quota, setQuota] = useState<Quota | null>(null);
  const [models, setModels] = useState<AccModel[]>([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    const token = await getIdToken();
    if (!token) return;
    try {
      const [q, m] = await Promise.all([
        fetch(`${API_BASE}/api/account/quota`, { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json()),
        fetch(`${API_BASE}/api/account/models`, { headers: { Authorization: `Bearer ${token}` } }).then((r) => r.json()),
      ]);
      setQuota(q.quota); setModels(m.models || []);
    } catch {/* ignore */}
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

      {/* Not logged in */}
      {configured && !user && !loading && (
        <div className="mt-10 flex flex-col items-center rounded-[24px] border border-line bg-paper py-16 text-center">
          <Sparkles className="mb-3 text-forest" />
          <h2 className="font-serif text-2xl text-ink">Увійдіть, щоб бачити моделі</h2>
          <p className="mt-2 max-w-[420px] text-sm text-ink-2">Кабінет зберігає історію генерацій. 5 безкоштовних завантажень повної моделі.</p>
          <button onClick={signIn} className="mt-5 inline-flex items-center gap-2 rounded-full bg-forest px-5 py-3 text-sm font-bold text-white">
            Увійти / Зареєструватися
          </button>
        </div>
      )}

      {loading && <div className="mt-10 flex justify-center"><Loader2 className="animate-spin text-forest" /></div>}

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

          <h3 className="mb-3 mt-8 font-serif text-xl text-ink">Мої моделі</h3>
          {models.length === 0 ? (
            <div className="rounded-[18px] border border-dashed border-line bg-paper px-4 py-10 text-center text-sm text-ink-3">
              Поки немає моделей. <Link href="/create" className="font-semibold text-forest underline-offset-2 hover:underline">Створити першу →</Link>
            </div>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {models.map((m) => (
                <div key={m.task_id} className="flex flex-col rounded-[16px] border border-line bg-paper p-4">
                  <div className="flex items-center gap-2 text-forest"><Box size={18} /></div>
                  <div className="mt-2 font-serif text-[17px] text-ink">{m.title || m.city || (m.product_type === "keychain" ? "Брелок" : "3D-мапа")}</div>
                  <div className="text-[12px] text-ink-3">{m.product_type === "keychain" ? "Брелок" : "Мапа"}{m.ts ? ` · ${new Date(m.ts * 1000).toLocaleDateString("uk")}` : ""}</div>
                  <button onClick={() => download(m)} disabled={busy}
                    className="mt-3 inline-flex min-h-10 items-center justify-center gap-2 rounded-full bg-forest px-4 py-2 text-sm font-semibold text-white disabled:opacity-60">
                    {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download size={15} />} Завантажити
                  </button>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
