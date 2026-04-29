"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, Clipboard, Loader2, Play, Search } from "lucide-react";
import { api } from "@/lib/api";

const STATUSES: Record<string, string> = {
  new: "Нова",
  contacted: "На звʼязку",
  in_progress: "В роботі",
  done: "Готово",
  archived: "Архів",
};

export default function AdminPage() {
  const [token, setToken] = useState("");
  const [orders, setOrders] = useState<any[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [startingGeneration, setStartingGeneration] = useState(false);
  const [copyState, setCopyState] = useState<"idle" | "copied">("idle");
  const [error, setError] = useState<string | null>(null);

  const loadOrders = async (nextToken = token) => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getAdminOrders(nextToken || undefined);
      setOrders(data.orders ?? []);
      if (!selectedId && data.orders?.length) setSelectedId(data.orders[0].id);
      if (nextToken) window.localStorage.setItem("admin-token", nextToken);
    } catch (err: any) {
      setError(err?.response?.data?.detail || "Не вдалося завантажити заявки");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const saved = window.localStorage.getItem("admin-token") || "";
    setToken(saved);
    loadOrders(saved);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const filtered = orders.filter((order) => {
    const haystack = `${order.id} ${order.name} ${order.contact} ${order.city}`.toLowerCase();
    return haystack.includes(query.toLowerCase());
  });
  const selected = filtered.find((order) => order.id === selectedId) ?? filtered[0];
  const selectedRecipe = selected?.generation_request ?? {};
  const selectedLayers = selected?.layers ?? {};
  const selectedZones = selected?.selected_zones ?? [];
  const selectedBounds = selected?.bounds ?? {};

  const copyRecipe = async () => {
    if (!selected) return;
    await navigator.clipboard.writeText(JSON.stringify(selectedRecipe, null, 2));
    setCopyState("copied");
    window.setTimeout(() => setCopyState("idle"), 1400);
  };

  const startGeneration = async () => {
    if (!selected) return;
    setStartingGeneration(true);
    setError(null);
    try {
      const result = await api.startOrderGeneration(selected.id, token || undefined);
      setOrders((current) => current.map((order) => (
        order.id === selected.id
          ? { ...order, status: "in_progress", generation_task_id: result.task_id, generation_status: result.status, generation_all_task_ids: result.all_task_ids ?? [] }
          : order
      )));
    } catch (err: any) {
      setError(err?.response?.data?.detail || "Не вдалося запустити Blender");
    } finally {
      setStartingGeneration(false);
    }
  };

  return (
    <main className="min-h-screen bg-[#f3efe7] p-4 text-[#1f2420] lg:p-6">
      <div className="mx-auto max-w-[1500px] border border-[#dfd7c8] bg-[#fffaf1]">
        <header className="flex flex-col gap-4 border-b border-[#dfd7c8] px-5 py-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-[#8a8173]">3d-fish · адмін-панель</p>
            <h1 className="mt-1 font-serif text-3xl">Заявки</h1>
          </div>
          <div className="flex flex-col gap-2 sm:flex-row">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-[#8a8173]" size={15} />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Пошук заявки, імʼя, місто..."
                className="h-10 w-full rounded-[6px] border border-[#dfd7c8] bg-[#f7f2e8] pl-9 pr-3 text-sm outline-none sm:w-[320px]"
              />
            </div>
            <input
              value={token}
              onChange={(event) => setToken(event.target.value)}
              placeholder="ADMIN_API_TOKEN"
              className="h-10 rounded-[6px] border border-[#dfd7c8] bg-[#f7f2e8] px-3 text-sm outline-none"
            />
            <button
              type="button"
              onClick={() => loadOrders()}
              className="inline-flex h-10 items-center justify-center gap-2 rounded-[6px] bg-[#1f2420] px-4 text-sm font-semibold text-white"
            >
              {loading ? <Loader2 size={15} className="animate-spin" /> : <CheckCircle2 size={15} />}
              Оновити
            </button>
          </div>
        </header>

        {error && <div className="border-b border-[#dfd7c8] bg-[#f5ded8] px-5 py-3 text-sm text-[#8a3b2f]">{error}</div>}

        <section className="grid min-h-[720px] lg:grid-cols-[260px,minmax(0,1fr),420px]">
          <aside className="border-b border-[#dfd7c8] p-4 lg:border-b-0 lg:border-r">
            <div className="grid grid-cols-2 gap-2 lg:grid-cols-1">
              {Object.entries(STATUSES).map(([key, label]) => {
                const count = orders.filter((order) => (order.status || "new") === key).length;
                return (
                  <div key={key} className="flex justify-between rounded-[7px] bg-[#f1eadf] px-3 py-2 text-sm">
                    <span>{label}</span>
                    <span className="font-mono text-xs">{count}</span>
                  </div>
                );
              })}
            </div>
          </aside>

          <div className="overflow-x-auto border-b border-[#dfd7c8] lg:border-b-0 lg:border-r">
            <table className="w-full min-w-[720px] border-collapse text-sm">
              <thead className="bg-[#f7f2e8] font-mono text-[10px] uppercase tracking-[0.16em] text-[#8a8173]">
                <tr>
                  <th className="px-4 py-3 text-left">ID</th>
                  <th className="px-4 py-3 text-left">Клієнт</th>
                  <th className="px-4 py-3 text-left">Місто</th>
                  <th className="px-4 py-3 text-left">Розмір</th>
                  <th className="px-4 py-3 text-right">Ціна</th>
                  <th className="px-4 py-3 text-left">Статус</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((order) => (
                  <tr
                    key={order.id}
                    onClick={() => setSelectedId(order.id)}
                    className={`cursor-pointer border-t border-[#dfd7c8] ${selected?.id === order.id ? "bg-[#dde9df]" : "hover:bg-[#f7f2e8]"}`}
                  >
                    <td className="px-4 py-4 font-mono text-xs">{order.id}</td>
                    <td className="px-4 py-4">
                      <div className="font-medium">{order.name || "Без імені"}</div>
                      <div className="text-xs text-[#7a7466]">{order.contact}</div>
                    </td>
                    <td className="px-4 py-4">{order.city}</td>
                    <td className="px-4 py-4">{Number(order.model_size_mm || 0) / 10} см</td>
                    <td className="px-4 py-4 text-right font-medium">{order.price_uah ? `${order.price_uah} ₴` : "—"}</td>
                    <td className="px-4 py-4">
                      <span className="rounded-full bg-[#e8f0e9] px-2 py-1 text-xs text-[#1f5b49]">
                        {STATUSES[order.status || "new"] ?? order.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <aside className="p-4">
            {selected ? (
              <div className="space-y-4">
                <div className="rounded-[8px] border border-[#dfd7c8] bg-[#f7f2e8] p-4">
                  <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-[#8a8173]">Заявка</p>
                  <h2 className="mt-1 font-serif text-3xl">{selected.name || "Без імені"}</h2>
                  <p className="mt-2 text-sm text-[#71695e]">{selected.contact}</p>
                  <div className="mt-4 grid gap-2">
                    <button
                      type="button"
                      onClick={startGeneration}
                      disabled={startingGeneration}
                      className="inline-flex h-10 items-center justify-center gap-2 rounded-[6px] bg-[#1f5b49] px-4 text-sm font-semibold text-white disabled:opacity-60"
                    >
                      {startingGeneration ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
                      Запустити Blender
                    </button>
                    <button
                      type="button"
                      onClick={copyRecipe}
                      className="inline-flex h-10 items-center justify-center gap-2 rounded-[6px] border border-[#dfd7c8] bg-[#fffaf1] px-4 text-sm font-semibold"
                    >
                      <Clipboard size={15} />
                      {copyState === "copied" ? "JSON скопійовано" : "Скопіювати recipe JSON"}
                    </button>
                  </div>
                </div>
                <div className="rounded-[8px] border border-[#dfd7c8] p-4">
                  <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-[#8a8173]">Деталі</p>
                  <div className="mt-3 divide-y divide-[#e4dccd] text-sm">
                    {[
                      ["Місто", selected.city],
                      ["Preview", selected.preview_id || "—"],
                      ["Режим ділянки", selected.area_mode || "rect"],
                      ["Зон", selectedZones.length ? `${selectedZones.length}` : "—"],
                      ["Розмір", `${Number(selected.model_size_mm || 0) / 10} см`],
                      ["Матеріал", selected.material],
                      ["Шари", Object.entries(selectedLayers).filter(([, v]) => v).map(([k]) => k).join(", ")],
                      ["Ціна", selected.price_uah ? `${selected.price_uah} ₴` : "—"],
                      ["Blender task", selected.generation_task_id || "—"],
                    ].map(([label, value]) => (
                      <div key={label} className="flex justify-between gap-4 py-2">
                        <span className="text-[#8a8173]">{label}</span>
                        <span className="text-right font-medium">{String(value || "—")}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="rounded-[8px] border border-[#dfd7c8] p-4">
                  <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-[#8a8173]">Параметри для локальної генерації</p>
                  <div className="mt-3 divide-y divide-[#e4dccd] text-sm">
                    {[
                      ["Bounds", `${Number(selectedBounds.south).toFixed(5)}..${Number(selectedBounds.north).toFixed(5)}, ${Number(selectedBounds.west).toFixed(5)}..${Number(selectedBounds.east).toFixed(5)}`],
                      ["Road width", selectedRecipe.road_width_multiplier],
                      ["Building min / mult", `${selectedRecipe.building_min_height} м / x${selectedRecipe.building_height_multiplier}`],
                      ["Terrain", selectedRecipe.terrain_enabled ? `${selectedRecipe.terrain_resolution} · z${selectedRecipe.terrain_z_scale}` : "вимкнено"],
                      ["Export", selectedRecipe.export_format || "3mf"],
                    ].map(([label, value]) => (
                      <div key={label} className="flex justify-between gap-4 py-2">
                        <span className="text-[#8a8173]">{label}</span>
                        <span className="max-w-[230px] text-right font-medium">{String(value || "—")}</span>
                      </div>
                    ))}
                  </div>
                  <pre className="mt-3 max-h-[220px] overflow-auto rounded-[6px] bg-[#1f2420] p-3 text-[11px] leading-5 text-[#fffaf1]">
                    {JSON.stringify(selectedRecipe, null, 2)}
                  </pre>
                </div>
                <div className="rounded-[8px] border border-[#dfd7c8] bg-[#f7f2e8] p-4">
                  <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-[#8a8173]">Коментар</p>
                  <p className="mt-2 text-sm leading-6">{selected.comment || "Коментар не залишили."}</p>
                </div>
              </div>
            ) : (
              <div className="grid h-full place-items-center text-sm text-[#8a8173]">Заявок ще немає</div>
            )}
          </aside>
        </section>
      </div>
    </main>
  );
}
