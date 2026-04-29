"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, Box, CheckCircle2, Clock3, Download, Loader2, LogIn, LogOut, Sparkles } from "lucide-react";
import { FastPreview3D } from "@/components/FastPreview3D";
import { useAuth } from "@/components/AuthProvider";
import { api, type AccountModel, type AccountResponse } from "@/lib/api";

const STATUS_LABELS: Record<string, string> = {
  processing: "Створюється",
  pending: "У черзі",
  completed: "Готово",
  failed: "Помилка",
};

const STATUS_STYLES: Record<string, string> = {
  processing: "bg-[#e9e0cf] text-[#7a5b2d]",
  pending: "bg-[#e9e0cf] text-[#7a5b2d]",
  completed: "bg-[#dde9df] text-[#1f5b49]",
  failed: "bg-[#f3ded8] text-[#8a3b2f]",
};

function ModelCard({ model }: { model: AccountModel }) {
  const preview = model.preview_snapshot ?? null;
  const activeLayers = useMemo(() => ({
    terrain: Boolean(model.layers?.terrain ?? true),
    roads: Boolean(model.layers?.roads ?? true),
    buildings: Boolean(model.layers?.buildings ?? true),
    water: Boolean(model.layers?.water ?? true),
    parks: Boolean(model.layers?.parks ?? true),
  }), [model.layers]);

  return (
    <article className="grid overflow-hidden rounded-[10px] border border-[#dfd7c8] bg-[#fffaf1] lg:grid-cols-[300px,minmax(0,1fr)]">
      <div className="h-[260px] border-b border-[#dfd7c8] bg-[#fbf8ef] lg:border-b-0 lg:border-r">
        {preview ? (
          <FastPreview3D preview={preview} loading={false} visibleLayers={activeLayers} material={model.material || "white"} />
        ) : (
          <div className="grid h-full place-items-center text-center text-sm text-[#8a8173]">
            <div>
              <Box className="mx-auto mb-3 text-[#1f5b49]" />
              Preview буде тут після створення
            </div>
          </div>
        )}
      </div>
      <div className="p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-[#8a8173]">{model.city || "Місто"} · {model.id}</p>
            <h2 className="mt-1 font-serif text-3xl">{model.title || "3D-мапа"}</h2>
          </div>
          <span className={`rounded-full px-3 py-1 text-xs font-semibold ${STATUS_STYLES[model.status] || "bg-[#f1eadf] text-[#71695e]"}`}>
            {STATUS_LABELS[model.status] || model.status}
          </span>
        </div>

        <div className="mt-5 grid gap-3 text-sm sm:grid-cols-4">
          <div className="rounded-[8px] bg-[#f1eadf] p-3">
            <p className="text-xs text-[#8a8173]">Прогрес</p>
            <p className="mt-1 font-medium">{model.progress ?? 0}%</p>
          </div>
          <div className="rounded-[8px] bg-[#f1eadf] p-3">
            <p className="text-xs text-[#8a8173]">Розмір</p>
            <p className="mt-1 font-medium">{Number(model.model_size_mm || 0) / 10} см</p>
          </div>
          <div className="rounded-[8px] bg-[#f1eadf] p-3">
            <p className="text-xs text-[#8a8173]">Формат</p>
            <p className="mt-1 font-medium">3MF/STL</p>
          </div>
          <div className="rounded-[8px] bg-[#f1eadf] p-3">
            <p className="text-xs text-[#8a8173]">Дата</p>
            <p className="mt-1 font-medium">{model.created_at ? new Date(model.created_at).toLocaleDateString("uk-UA") : "—"}</p>
          </div>
        </div>

        <p className="mt-4 min-h-6 text-sm text-[#71695e]">{model.error || model.message || "Очікуємо оновлення статусу."}</p>

        <div className="mt-5 flex flex-wrap gap-2">
          {(model.download_url_3mf || model.download_url) && (
            <a className="inline-flex h-10 items-center gap-2 rounded-[6px] bg-[#1f5b49] px-4 text-sm font-semibold text-white" href={model.download_url_3mf || model.download_url || "#"}>
              <Download size={15} /> Завантажити 3MF
            </a>
          )}
          {model.download_url_stl && (
            <a className="inline-flex h-10 items-center gap-2 rounded-[6px] border border-[#dfd7c8] bg-[#fffaf1] px-4 text-sm font-semibold" href={model.download_url_stl}>
              <Download size={15} /> STL
            </a>
          )}
        </div>
      </div>
    </article>
  );
}

export default function AccountPage() {
  const { user, loading: authLoading, configured, signIn, signOut } = useAuth();
  const [account, setAccount] = useState<AccountResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadAccount = async () => {
    if (!user) return;
    setLoading(true);
    setError(null);
    try {
      setAccount(await api.getAccountModels());
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || "Не вдалося завантажити кабінет");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAccount();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  const models = account?.models ?? [];
  const usage = account?.usage;

  return (
    <main className="min-h-screen bg-[#f3efe7] p-4 text-[#1f2420] lg:p-6">
      <div className="mx-auto max-w-[1380px] border border-[#dfd7c8] bg-[#f7f2e8]">
        <header className="flex flex-col gap-4 border-b border-[#dfd7c8] bg-[#fffaf1] px-5 py-5 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <Link href="/" className="inline-flex items-center gap-2 text-sm text-[#71695e]"><ArrowLeft size={15} /> Конструктор</Link>
            <h1 className="mt-3 font-serif text-4xl">Мій кабінет</h1>
            <p className="mt-2 text-sm text-[#71695e]">Історія повністю згенерованих моделей, ліміт акаунта і файли для завантаження.</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {user && <span className="rounded-full border border-[#dfd7c8] bg-[#f1eadf] px-3 py-2 text-sm">{user.email}</span>}
            {user ? (
              <button className="inline-flex h-10 items-center gap-2 rounded-[6px] bg-[#1f2420] px-4 text-sm font-semibold text-white" onClick={signOut}>
                <LogOut size={15} /> Вийти
              </button>
            ) : (
              <button className="inline-flex h-10 items-center gap-2 rounded-[6px] bg-[#1f5b49] px-4 text-sm font-semibold text-white" onClick={signIn} disabled={!configured || authLoading}>
                <LogIn size={15} /> Увійти через Google
              </button>
            )}
          </div>
        </header>

        {!configured && (
          <div className="border-b border-[#dfd7c8] bg-[#f5ded8] px-5 py-3 text-sm text-[#8a3b2f]">
            Firebase client config ще не заповнений у env. Додайте NEXT_PUBLIC_FIREBASE_* змінні.
          </div>
        )}

        {user ? (
          <>
            <section className="grid gap-4 border-b border-[#dfd7c8] p-5 md:grid-cols-4">
              <div className="rounded-[10px] border border-[#dfd7c8] bg-[#fffaf1] p-5">
                <Sparkles className="text-[#1f5b49]" size={18} />
                <p className="mt-4 text-xs uppercase tracking-[0.16em] text-[#8a8173]">Безкоштовний ліміт</p>
                <div className="mt-1 font-serif text-4xl">{usage?.remaining ?? 0}/{usage?.free_limit ?? 10}</div>
              </div>
              <div className="rounded-[10px] border border-[#dfd7c8] bg-[#fffaf1] p-5">
                <CheckCircle2 className="text-[#1f5b49]" size={18} />
                <p className="mt-4 text-xs uppercase tracking-[0.16em] text-[#8a8173]">Готові</p>
                <div className="mt-1 font-serif text-4xl">{usage?.completed ?? 0}</div>
              </div>
              <div className="rounded-[10px] border border-[#dfd7c8] bg-[#fffaf1] p-5">
                <Clock3 className="text-[#1f5b49]" size={18} />
                <p className="mt-4 text-xs uppercase tracking-[0.16em] text-[#8a8173]">Використано/у роботі</p>
                <div className="mt-1 font-serif text-4xl">{usage?.used ?? 0}</div>
              </div>
              <button onClick={loadAccount} className="rounded-[10px] border border-[#dfd7c8] bg-[#1f2420] p-5 text-left text-white">
                {loading ? <Loader2 className="animate-spin" size={18} /> : <Clock3 size={18} />}
                <p className="mt-4 text-xs uppercase tracking-[0.16em] text-white/60">Оновлення</p>
                <div className="mt-1 font-serif text-3xl">Статуси</div>
              </button>
            </section>

            {error && <div className="border-b border-[#dfd7c8] bg-[#f5ded8] px-5 py-3 text-sm text-[#8a3b2f]">{error}</div>}

            <section className="space-y-4 p-5">
              {models.length ? models.map((model) => <ModelCard key={model.id} model={model} />) : (
                <div className="grid min-h-[360px] place-items-center rounded-[10px] border border-[#dfd7c8] bg-[#fffaf1] text-center">
                  <div>
                    <Box className="mx-auto mb-4 text-[#1f5b49]" size={34} />
                    <h2 className="font-serif text-3xl">Історія поки порожня</h2>
                    <p className="mt-2 text-sm text-[#71695e]">Створіть повну модель із конструктора, і вона зʼявиться тут.</p>
                  </div>
                </div>
              )}
            </section>
          </>
        ) : (
          <section className="grid min-h-[520px] place-items-center p-6 text-center">
            <div className="max-w-md">
              <h2 className="font-serif text-4xl">Увійдіть, щоб бачити моделі</h2>
              <p className="mt-3 text-sm leading-6 text-[#71695e]">Google-вхід створить кабінет, де буде історія генерацій і 10 безкоштовних запусків повної моделі.</p>
              <button className="mt-6 inline-flex h-11 items-center gap-2 rounded-[6px] bg-[#1f5b49] px-5 text-sm font-semibold text-white" onClick={signIn} disabled={!configured || authLoading}>
                <LogIn size={16} /> Увійти через Google
              </button>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}
