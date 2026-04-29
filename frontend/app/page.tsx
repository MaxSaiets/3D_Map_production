"use client";

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import { CheckCircle2, Loader2, Mail, Map as MapIcon, Send, SlidersHorizontal } from "lucide-react";
import { FastPreview3D } from "@/components/FastPreview3D";
import { api } from "@/lib/api";
import { useGenerationStore } from "@/store/generation-store";

const MapSelector = dynamic(
  () => import("@/components/MapSelector").then((mod) => ({ default: mod.MapSelector })),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full min-h-[420px] items-center justify-center bg-white/70 text-sm text-slate-500">
        Завантаження карти...
      </div>
    ),
  },
);

const CITIES: Record<string, { label: string; center: [number, number] }> = {
  Kyiv: { label: "Київ", center: [50.4501, 30.5234] },
  Khmelnytskyi: { label: "Хмельницький", center: [49.42, 26.98] },
};

export default function Home() {
  const selectedArea = useGenerationStore((state) => state.selectedArea);
  const selectedShapeGeoJson = useGenerationStore((state) => state.selectedShapeGeoJson);
  const fastPreview = useGenerationStore((state) => state.fastPreview);
  const setFastPreview = useGenerationStore((state) => state.setFastPreview);

  const [cityKey, setCityKey] = useState("Kyiv");
  const [includeTerrain, setIncludeTerrain] = useState(true);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [customerName, setCustomerName] = useState("");
  const [contact, setContact] = useState("");
  const [message, setMessage] = useState("");

  const boundsPayload = useMemo(() => {
    if (!selectedArea) return null;
    return {
      north: selectedArea.getNorth(),
      south: selectedArea.getSouth(),
      east: selectedArea.getEast(),
      west: selectedArea.getWest(),
    };
  }, [selectedArea]);

  const createPreview = async () => {
    if (!boundsPayload) {
      setError("Спочатку намалюйте область на мапі.");
      return;
    }
    setError(null);
    setNotice(null);
    setIsPreviewLoading(true);
    try {
      const preview = await api.generatePreview({
        ...boundsPayload,
        polygon_geojson: selectedShapeGeoJson,
        include_terrain: includeTerrain,
      });
      setFastPreview(preview);
      setNotice("Прев'ю готове. Можна залишити заявку.");
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || "Не вдалося створити швидке прев'ю.");
    } finally {
      setIsPreviewLoading(false);
    }
  };

  const submitOrder = async () => {
    if (!boundsPayload || !fastPreview) {
      setError("Спочатку створіть прев'ю.");
      return;
    }
    if (!customerName.trim() || !contact.trim()) {
      setError("Вкажіть ім'я та контакт.");
      return;
    }
    setError(null);
    setNotice(null);
    setIsSubmitting(true);
    try {
      const order = await api.submitOrder({
        preview_id: fastPreview.preview_id,
        customer_name: customerName.trim(),
        contact: contact.trim(),
        message: message.trim() || undefined,
        city: CITIES[cityKey]?.label,
        bounds: boundsPayload,
        polygon_geojson: selectedShapeGeoJson,
        options: {
          includeTerrain,
          metrics: fastPreview.metrics,
        },
      });
      setNotice(`Заявку ${order.id} збережено. Адмін побачить її в панелі.`);
      setCustomerName("");
      setContact("");
      setMessage("");
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || "Не вдалося відправити заявку.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const metrics = fastPreview?.metrics;

  return (
    <main className="min-h-[100dvh] bg-[#f4efe4] text-slate-900">
      <div className="mx-auto grid min-h-[100dvh] max-w-[1760px] grid-rows-[auto,1fr] gap-3 px-3 py-3 lg:px-5">
        <header className="flex flex-col gap-3 border border-black/10 bg-white/80 px-4 py-4 shadow-sm backdrop-blur lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-teal-700">3dMAP preview studio</p>
            <h1 className="mt-1 text-2xl font-semibold tracking-tight sm:text-3xl">Швидке прев'ю 3D-мапи</h1>
          </div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(CITIES).map(([key, city]) => (
              <button
                key={key}
                type="button"
                onClick={() => setCityKey(key)}
                className={`min-h-10 px-4 text-sm font-semibold transition ${
                  key === cityKey ? "bg-teal-700 text-white" : "border border-black/10 bg-white text-slate-700"
                }`}
              >
                {city.label}
              </button>
            ))}
          </div>
        </header>

        <section className="grid min-h-0 gap-3 lg:grid-cols-[minmax(0,1.1fr),minmax(380px,0.9fr)]">
          <div className="grid min-h-[520px] grid-rows-[auto,1fr] overflow-hidden border border-black/10 bg-white shadow-sm">
            <div className="flex items-center justify-between gap-3 border-b border-black/10 px-4 py-3">
              <div className="flex items-center gap-2">
                <MapIcon size={18} className="text-teal-700" />
                <div>
                  <div className="text-sm font-semibold">Оберіть ділянку</div>
                  <div className="text-xs text-slate-500">Прямокутник, полігон або коло на мапі</div>
                </div>
              </div>
              <label className="flex items-center gap-2 text-sm text-slate-600">
                <input type="checkbox" checked={includeTerrain} onChange={(e) => setIncludeTerrain(e.target.checked)} />
                Рельєф
              </label>
            </div>
            <MapSelector center={CITIES[cityKey].center} />
          </div>

          <div className="grid min-h-0 gap-3 lg:grid-rows-[minmax(360px,1fr),auto]">
            <div className="overflow-hidden border border-black/10 bg-white shadow-sm">
              <div className="flex items-center justify-between gap-3 border-b border-black/10 px-4 py-3">
                <div className="flex items-center gap-2">
                  <SlidersHorizontal size={18} className="text-teal-700" />
                  <div>
                    <div className="text-sm font-semibold">Браузерне 3D-прев'ю</div>
                    <div className="text-xs text-slate-500">Без Blender, пазів і 3MF export</div>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={createPreview}
                  disabled={isPreviewLoading || !boundsPayload}
                  className="inline-flex min-h-10 items-center gap-2 bg-teal-700 px-4 text-sm font-semibold text-white disabled:bg-slate-400"
                >
                  {isPreviewLoading ? <Loader2 size={16} className="animate-spin" /> : <CheckCircle2 size={16} />}
                  Створити прев'ю
                </button>
              </div>
              <FastPreview3D />
            </div>

            <div className="border border-black/10 bg-white p-4 shadow-sm">
              {metrics && (
                <div className="mb-4 grid grid-cols-4 gap-2 text-center text-xs">
                  <div className="bg-slate-50 p-2"><b>{metrics.buildings}</b><br />будівель</div>
                  <div className="bg-slate-50 p-2"><b>{metrics.roads}</b><br />доріг</div>
                  <div className="bg-slate-50 p-2"><b>{metrics.water}</b><br />води</div>
                  <div className="bg-slate-50 p-2"><b>{metrics.parks}</b><br />парків</div>
                </div>
              )}

              <div className="grid gap-2 sm:grid-cols-2">
                <input
                  value={customerName}
                  onChange={(e) => setCustomerName(e.target.value)}
                  placeholder="Ім'я"
                  className="min-h-11 border border-black/10 px-3 text-sm outline-none focus:border-teal-700"
                />
                <input
                  value={contact}
                  onChange={(e) => setContact(e.target.value)}
                  placeholder="Телефон, Telegram або email"
                  className="min-h-11 border border-black/10 px-3 text-sm outline-none focus:border-teal-700"
                />
              </div>
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Коментар до замовлення"
                className="mt-2 min-h-20 w-full border border-black/10 px-3 py-2 text-sm outline-none focus:border-teal-700"
              />
              <button
                type="button"
                onClick={submitOrder}
                disabled={isSubmitting || !fastPreview}
                className="mt-3 inline-flex min-h-11 w-full items-center justify-center gap-2 bg-rose-600 px-4 text-sm font-semibold text-white disabled:bg-slate-400"
              >
                {isSubmitting ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
                Надіслати заявку
              </button>
              <a href="/admin" className="mt-3 inline-flex items-center gap-2 text-sm font-semibold text-teal-700">
                <Mail size={16} />
                Відкрити адмінку
              </a>
              {notice && <div className="mt-3 bg-emerald-50 px-3 py-2 text-sm text-emerald-800">{notice}</div>}
              {error && <div className="mt-3 bg-rose-50 px-3 py-2 text-sm text-rose-800">{error}</div>}
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
