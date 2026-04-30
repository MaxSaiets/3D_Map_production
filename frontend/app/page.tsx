"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  Check,
  Grid2X2,
  Hexagon,
  Layers3,
  LogIn,
  Loader2,
  MapPinned,
  PenLine,
  Phone,
  RotateCcw,
  Square,
} from "lucide-react";
import { FastPreview3D } from "@/components/FastPreview3D";
import type { MapSelection } from "@/components/MapSelector";
import { useAuth } from "@/components/AuthProvider";
import { api, type FastPreviewResponse, type GenerationRequest } from "@/lib/api";

const MapSelector = dynamic(
  () => import("@/components/MapSelector").then((mod) => ({ default: mod.MapSelector })),
  {
    ssr: false,
    loading: () => <div className="grid h-full min-h-[420px] place-items-center bg-[#eee8dd] text-sm text-[#7a7466]">Карта завантажується...</div>,
  },
);

const HexagonalGrid = dynamic(() => import("@/components/HexagonalGrid"), {
  ssr: false,
  loading: () => <div className="grid h-full min-h-[420px] place-items-center bg-[#eee8dd] text-sm text-[#7a7466]">Сітка завантажується...</div>,
});

type AreaMode = "rect" | "grid" | "hex" | "freehand";
type LayerKey = "terrain" | "roads" | "buildings" | "water" | "parks";
type MaterialKey = "white" | "concrete" | "graphite" | "green" | "terracotta";

const CITIES = {
  Kyiv: {
    label: "Київ",
    center: [50.4501, 30.5234] as [number, number],
    bounds: { north: 50.627232, south: 50.178116, east: 30.830663, west: 30.164347 },
  },
  Lviv: {
    label: "Львів",
    center: [49.8397, 24.0297] as [number, number],
    bounds: { north: 49.843, south: 49.837, east: 24.035, west: 24.025 },
  },
  Odesa: {
    label: "Одеса",
    center: [46.4825, 30.7233] as [number, number],
    bounds: { north: 46.486, south: 46.48, east: 30.729, west: 30.718 },
  },
  Kharkiv: {
    label: "Харків",
    center: [49.9935, 36.2304] as [number, number],
    bounds: { north: 49.997, south: 49.991, east: 36.236, west: 36.225 },
  },
  Dnipro: {
    label: "Дніпро",
    center: [48.4647, 35.0462] as [number, number],
    bounds: { north: 48.469, south: 48.463, east: 35.052, west: 35.041 },
  },
};

const AREA_MODES: Array<{ id: AreaMode; label: string; hint: string; icon: typeof Square }> = [
  { id: "rect", label: "Прямокутник", hint: "Найшвидше", icon: Square },
  { id: "grid", label: "Сітка зон", hint: "Серія", icon: Grid2X2 },
  { id: "hex", label: "Гексагони", hint: "Точніше", icon: Hexagon },
  { id: "freehand", label: "Намалювати", hint: "Контур", icon: PenLine },
];

const LAYER_META: Array<{ key: LayerKey; label: string; hint: string }> = [
  { key: "terrain", label: "Рельєф", hint: "Контур і основа" },
  { key: "roads", label: "Дороги", hint: "Широкі маски" },
  { key: "buildings", label: "Будівлі", hint: "Прості блоки" },
  { key: "water", label: "Вода", hint: "Річки й озера" },
  { key: "parks", label: "Парки", hint: "Зелені зони" },
];

const MATERIALS: Array<{ key: MaterialKey; label: string; color: string; hint: string }> = [
  { key: "white", label: "Білий", color: "#f4f0e7", hint: "Класика" },
  { key: "concrete", label: "Бетон", color: "#c9c3b8", hint: "Архітектурний" },
  { key: "graphite", label: "Графіт", color: "#303336", hint: "Темний" },
  { key: "green", label: "Зелений", color: "#cfdccb", hint: "Акцентний" },
  { key: "terracotta", label: "Теракот", color: "#d8ad89", hint: "Теплий" },
];

function zoneBounds(zones: any[]) {
  const coords: Array<[number, number]> = [];
  zones.forEach((zone) => {
    const geometry = zone?.geometry ?? zone?.feature?.geometry;
    const rings = geometry?.type === "Polygon" ? geometry.coordinates : geometry?.type === "MultiPolygon" ? geometry.coordinates.flat() : [];
    rings.forEach((ring: any[]) => ring.forEach((pt) => Array.isArray(pt) && coords.push([Number(pt[0]), Number(pt[1])])));
  });
  if (!coords.length) return null;
  const lngs = coords.map((pt) => pt[0]);
  const lats = coords.map((pt) => pt[1]);
  return { west: Math.min(...lngs), east: Math.max(...lngs), south: Math.min(...lats), north: Math.max(...lats) };
}

function areaKm2(bounds: { north: number; south: number; east: number; west: number }) {
  const latM = Math.abs(bounds.north - bounds.south) * 111.32;
  const lngM = Math.abs(bounds.east - bounds.west) * 111.32 * Math.cos((((bounds.north + bounds.south) / 2) * Math.PI) / 180);
  return Math.max(0.01, latM * lngM);
}

function money(value: number) {
  return new Intl.NumberFormat("uk-UA").format(value);
}

export default function Home() {
  const { user, loading: authLoading, configured: authConfigured, signIn } = useAuth();
  const [cityKey, setCityKey] = useState<keyof typeof CITIES>("Kyiv");
  const [areaMode, setAreaMode] = useState<AreaMode>("hex");
  const [selection, setSelection] = useState<MapSelection | null>({
    bounds: CITIES.Kyiv.bounds,
    polygonGeoJson: {
      type: "Polygon",
      coordinates: [[
        [CITIES.Kyiv.bounds.west, CITIES.Kyiv.bounds.south],
        [CITIES.Kyiv.bounds.east, CITIES.Kyiv.bounds.south],
        [CITIES.Kyiv.bounds.east, CITIES.Kyiv.bounds.north],
        [CITIES.Kyiv.bounds.west, CITIES.Kyiv.bounds.north],
        [CITIES.Kyiv.bounds.west, CITIES.Kyiv.bounds.south],
      ]],
    },
  });
  const [selectedZones, setSelectedZones] = useState<any[]>([]);
  const [modelSizeMm, setModelSizeMm] = useState(180);
  const [material, setMaterial] = useState<MaterialKey>("white");
  const [layers, setLayers] = useState<Record<LayerKey, boolean>>({
    terrain: true,
    roads: true,
    buildings: true,
    water: true,
    parks: true,
  });
  const [preview, setPreview] = useState<FastPreviewResponse | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [contact, setContact] = useState("");
  const [comment, setComment] = useState("");
  const [orderState, setOrderState] = useState<"idle" | "sending" | "sent" | "error">("idle");
  const [accountGenerationState, setAccountGenerationState] = useState<"idle" | "sending" | "sent" | "limit" | "error">("idle");
  const [confirmGenerationOpen, setConfirmGenerationOpen] = useState(false);

  const city = CITIES[cityKey];
  const activeBounds = useMemo(() => {
    if (areaMode === "grid" || areaMode === "hex") return zoneBounds(selectedZones) ?? selection?.bounds ?? city.bounds;
    return selection?.bounds ?? city.bounds;
  }, [areaMode, selectedZones, selection, city.bounds]);
  const activePolygon = useMemo(() => {
    if ((areaMode === "grid" || areaMode === "hex") && selectedZones.length === 1) {
      return selectedZones[0]?.geometry ?? selectedZones[0]?.feature?.geometry;
    }
    return selection?.polygonGeoJson;
  }, [areaMode, selectedZones, selection]);
  const activeLayers = Object.values(layers).filter(Boolean).length;
  const price = 690 + Math.round(modelSizeMm * 4.2) + activeLayers * 120 + Math.round(areaKm2(activeBounds) * 80);
  const buildGenerationRequest = useCallback((): GenerationRequest => ({
    ...activeBounds,
    road_width_multiplier: 0.8,
    road_height_mm: 0.5,
    road_embed_mm: 0.3,
    building_min_height: 5.0,
    building_height_multiplier: 1.8,
    building_foundation_mm: 0.6,
    building_embed_mm: 0.2,
    water_depth: 1.2,
    terrain_enabled: layers.terrain,
    terrain_z_scale: 0.5,
    terrain_base_thickness_mm: 0.3,
    terrain_resolution: 180,
    terrarium_zoom: 15,
    flatten_buildings_on_terrain: true,
    export_format: "3mf",
    model_size_mm: modelSizeMm,
    is_ams_mode: false,
    context_padding_m: 80,
    preview_include_base: layers.terrain,
    preview_include_roads: layers.roads,
    preview_include_buildings: layers.buildings,
    preview_include_water: layers.water,
    preview_include_parks: layers.parks,
  }), [activeBounds, layers, modelSizeMm]);

  useEffect(() => {
    setSelection({
      bounds: city.bounds,
      polygonGeoJson: {
        type: "Polygon",
        coordinates: [[
          [city.bounds.west, city.bounds.south],
          [city.bounds.east, city.bounds.south],
          [city.bounds.east, city.bounds.north],
          [city.bounds.west, city.bounds.north],
          [city.bounds.west, city.bounds.south],
        ]],
      },
    });
    setSelectedZones([]);
  }, [cityKey, city.bounds]);

  const reloadPreview = useCallback(async () => {
    if (!activeBounds) return;
    if ((areaMode === "grid" || areaMode === "hex") && selectedZones.length !== 1) {
      setPreview(null);
      setPreviewLoading(false);
      setPreviewError(null);
      return;
    }
    setPreviewLoading(true);
    setPreviewError(null);
    try {
      const generationRequest = buildGenerationRequest();
      const data = await api.createFastPreview({
        ...activeBounds,
        polygon_geojson: activePolygon,
        generation_request: generationRequest,
        include_terrain: generationRequest.preview_include_base,
        include_roads: generationRequest.preview_include_roads,
        include_buildings: generationRequest.preview_include_buildings,
        include_water: generationRequest.preview_include_water,
        include_parks: generationRequest.preview_include_parks,
      });
      setPreview(data);
      if (data.preview_status === "processing") {
        window.setTimeout(() => {
          reloadPreview();
        }, 5000);
      } else if (data.preview_status === "failed") {
        setPreviewError(String(data.model_logic?.preview_message || "Preview не вдалося створити"));
      }
    } catch (error: any) {
      setPreviewError(error?.response?.data?.detail || error?.message || "Помилка preview");
    } finally {
      setPreviewLoading(false);
    }
  }, [activeBounds, activePolygon, areaMode, buildGenerationRequest, selectedZones.length]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      reloadPreview();
    }, 450);
    return () => window.clearTimeout(timer);
  }, [reloadPreview]);

  const submitOrder = async () => {
    if (!contact.trim()) {
      setOrderState("error");
      return;
    }
    setOrderState("sending");
    try {
      const generationRequest = buildGenerationRequest();
      await api.createSiteOrder({
        name,
        contact,
        city: city.label,
        bounds: activeBounds,
        polygon_geojson: activePolygon,
        preview_id: preview?.preview_id,
        model_size_mm: modelSizeMm,
        material,
        layers,
        price_uah: price,
        comment,
        area_mode: areaMode,
        selected_zones: selectedZones,
        grid_type: areaMode === "hex" ? "hexagonal" : areaMode === "grid" ? "square" : areaMode,
        hex_size_m: areaMode === "hex" ? 1000 : 800,
        preview_metrics: preview?.metrics ?? {},
        model_logic: preview?.model_logic ?? {},
        generation_request: generationRequest,
      });
      setOrderState("sent");
    } catch {
      setOrderState("error");
    }
  };

  const confirmFullModelInAccount = async () => {
    if (!user) {
      await signIn();
      return;
    }
    if ((areaMode === "grid" || areaMode === "hex") && selectedZones.length !== 1) {
      alert("Оберіть одну гексагональну зону перед запуском повної генерації.");
      return;
    }
    setConfirmGenerationOpen(true);
  };

  const startFullModelInAccount = async () => {
    setConfirmGenerationOpen(false);
    setAccountGenerationState("sending");
    try {
      const generationRequest = buildGenerationRequest();
      await api.startAccountGeneration({
        title: `${city.label} · ${modelSizeMm / 10} см`,
        city: city.label,
        preview_id: preview?.preview_id,
        preview_snapshot: preview,
        bounds: activeBounds,
        polygon_geojson: activePolygon,
        model_size_mm: modelSizeMm,
        material,
        layers,
        generation_request: generationRequest,
      });
      setAccountGenerationState("sent");
    } catch (error: any) {
      setAccountGenerationState(error?.response?.status === 402 ? "limit" : "error");
    }
  };

  return (
    <main className="min-h-screen bg-[#f3efe7] text-[#1f2420]">
      <div className="mx-auto max-w-[1640px] border-x border-[#dfd7c8] bg-[#f7f2e8] shadow-[0_22px_80px_rgba(38,33,24,0.08)]">
        <header className="flex items-center justify-between border-b border-[#dfd7c8] bg-[#fffaf1] px-5 py-4 lg:px-9">
          <div className="flex items-center gap-8">
            <div className="flex items-center gap-3">
              <div className="grid h-8 w-8 place-items-center rounded-[6px] bg-[#1f5b49] text-white">
                <Layers3 size={17} />
              </div>
              <div className="font-serif text-2xl">3d-fish</div>
            </div>
            <nav className="hidden items-center gap-7 text-sm text-[#6f685d] lg:flex">
              <a className="border-b border-[#1f2420] pb-1 text-[#1f2420]" href="#constructor">Конструктор</a>
              <a href="#how">Як це працює</a>
              <a href="#order">Ціна</a>
              <a href="/account">Кабінет</a>
              <a href="/admin">Адмінка</a>
            </nav>
          </div>
          <div className="hidden items-center gap-4 text-[11px] uppercase tracking-[0.16em] text-[#746c60] sm:flex">
            <span className="h-1.5 w-1.5 rounded-full bg-[#1f5b49]" />
            Виготовлення 3 дні
            {user ? (
              <a href="/account" className="rounded-[6px] bg-[#1f2420] px-3 py-2 text-white normal-case tracking-normal">
                Мій кабінет
              </a>
            ) : (
              <button
                type="button"
                onClick={() => void signIn()}
                disabled={authLoading}
                className="rounded-[6px] bg-[#1f2420] px-3 py-2 text-white normal-case tracking-normal disabled:opacity-60"
              >
                Увійти
              </button>
            )}
          </div>
        </header>

        <section className="border-b border-[#dfd7c8] bg-[#fffaf1] px-5 py-8 lg:px-9 lg:py-10">
          <p className="font-mono text-[10px] uppercase tracking-[0.24em] text-[#8a8173]">Конструктор · швидке preview</p>
          <div className="mt-3 grid gap-5 lg:grid-cols-[1fr,220px] lg:items-end">
            <div>
              <h1 className="max-w-[920px] font-serif text-4xl leading-[0.98] tracking-[-0.02em] sm:text-5xl lg:text-6xl">
                Виберіть ділянку. Побачте preview. Замовте 3D-мапу.
              </h1>
              <p className="mt-4 max-w-2xl text-sm leading-6 text-[#71695e]">
                Preview створюється швидко: зона обрізає всі шари, дороги показуються як широкі маски, будівлі як прості блоки. Повна модель з пазами готується вже після заявки.
              </p>
            </div>
            <div className="text-left lg:text-right">
              <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-[#8a8173]">Орієнтовно</p>
              <div className="mt-1 font-serif text-4xl text-[#1f5b49]">3 дні</div>
              <p className="mt-1 text-xs text-[#71695e]">доставка по Україні</p>
            </div>
          </div>
        </section>

        <section id="constructor" className="grid items-start gap-4 p-4 lg:grid-cols-[330px,minmax(0,1fr),360px] lg:p-6">
          <aside className="space-y-4">
            <div className="rounded-[10px] border border-[#dfd7c8] bg-[#fffaf1] p-5">
              <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-[#8a8173]">5 кроків</p>
              <div className="mt-4 space-y-3">
                {["Місто", "Ділянка", "Розмір", "Шари", "Матеріал"].map((label, index) => (
                  <div key={label} className="flex items-center gap-3 text-sm">
                    <span className={`grid h-6 w-6 place-items-center rounded-full border text-[11px] ${index === 0 ? "border-[#1f5b49] bg-[#1f5b49] text-white" : index === 1 ? "border-[#1f5b49] text-[#1f5b49]" : "border-[#d8cfbd] text-[#8a8173]"}`}>
                      {index === 0 ? <Check size={13} /> : index + 1}
                    </span>
                    <span className={index <= 1 ? "font-medium text-[#1f2420]" : "text-[#8a8173]"}>{label}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-[10px] border border-[#dfd7c8] bg-[#fffaf1] p-5">
              <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-[#8a8173]">Крок 1</p>
              <h2 className="mt-1 font-serif text-2xl">Місто</h2>
              <div className="mt-4 flex flex-wrap gap-2">
                {(Object.keys(CITIES) as Array<keyof typeof CITIES>).map((key) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setCityKey(key)}
                    className={`rounded-full border px-3 py-2 text-xs font-medium ${cityKey === key ? "border-[#1f5b49] bg-[#dde9df] text-[#173d32]" : "border-[#dfd7c8] bg-[#f7f2e8] text-[#71695e]"}`}
                  >
                    {CITIES[key].label}
                  </button>
                ))}
              </div>
            </div>

            <div className="rounded-[10px] border border-[#dfd7c8] bg-[#fffaf1] p-5">
              <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-[#8a8173]">Крок 2</p>
              <h2 className="mt-1 font-serif text-2xl">Як виділити ділянку</h2>
              <div className="mt-4 grid grid-cols-2 gap-2">
                {AREA_MODES.map(({ id, label, hint, icon: Icon }) => (
                  <button
                    key={id}
                    type="button"
                    onClick={() => setAreaMode(id)}
                    className={`rounded-[8px] border p-3 text-left transition ${areaMode === id ? "border-[#1f5b49] bg-[#dde9df]" : "border-[#dfd7c8] bg-[#f1eadf] hover:bg-[#f7f2e8]"}`}
                  >
                    <Icon size={15} className="mb-2 text-[#1f5b49]" />
                    <div className="text-xs font-semibold">{label}</div>
                    <div className="mt-1 text-[11px] text-[#7a7466]">{hint}</div>
                  </button>
                ))}
              </div>
            </div>

            <div className="rounded-[10px] border border-[#dfd7c8] bg-[#fffaf1] p-5">
              <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-[#8a8173]">Крок 3</p>
              <h2 className="mt-1 font-serif text-2xl">Розмір моделі</h2>
              <div className="mt-4 grid grid-cols-3 gap-2">
                {[120, 180, 240].map((size) => (
                  <button
                    key={size}
                    type="button"
                    onClick={() => setModelSizeMm(size)}
                    className={`rounded-[7px] border px-3 py-3 text-center ${modelSizeMm === size ? "border-[#1f5b49] bg-[#dde9df]" : "border-[#dfd7c8] bg-[#f1eadf]"}`}
                  >
                    <div className="font-serif text-xl">{size / 10}см</div>
                    <div className="text-[10px] text-[#8a8173]">{size === 180 ? "подар." : size === 240 ? "прем." : "настіл."}</div>
                  </button>
                ))}
              </div>
            </div>

            <div className="rounded-[10px] border border-[#dfd7c8] bg-[#fffaf1] p-5">
              <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-[#8a8173]">Крок 4</p>
              <h2 className="mt-1 font-serif text-2xl">Шари моделі</h2>
              <div className="mt-4 space-y-2">
                {LAYER_META.map(({ key, label, hint }) => (
                  <label key={key} className="flex cursor-pointer items-center justify-between rounded-[7px] bg-[#f1eadf] px-3 py-2">
                    <span>
                      <span className="block text-sm font-medium">{label}</span>
                      <span className="text-[11px] text-[#7a7466]">{hint}</span>
                    </span>
                    <input
                      type="checkbox"
                      checked={layers[key]}
                      onChange={(event) => setLayers((prev) => ({ ...prev, [key]: event.target.checked }))}
                      className="h-4 w-4 accent-[#1f5b49]"
                    />
                  </label>
                ))}
              </div>
            </div>

            <div className="rounded-[10px] border border-[#dfd7c8] bg-[#fffaf1] p-5">
              <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-[#8a8173]">Крок 5</p>
              <h2 className="mt-1 font-serif text-2xl">Матеріал</h2>
              <div className="mt-4 grid grid-cols-5 gap-2">
                {MATERIALS.map((item) => (
                  <button
                    key={item.key}
                    type="button"
                    title={`${item.label} · ${item.hint}`}
                    onClick={() => setMaterial(item.key)}
                    className={`h-11 rounded-[7px] border ${material === item.key ? "border-[#1f5b49] ring-2 ring-[#cfe0d4]" : "border-[#dfd7c8]"}`}
                    style={{ background: item.color }}
                  />
                ))}
              </div>
              <div className="mt-2 text-xs font-medium">{MATERIALS.find((item) => item.key === material)?.label}</div>
            </div>
          </aside>

          <section className="min-w-0 rounded-[10px] border border-[#dfd7c8] bg-[#fffaf1]">
            <div className="flex items-center justify-between gap-4 border-b border-[#dfd7c8] px-5 py-4">
              <div>
                <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-[#8a8173]">Карта</p>
                <h2 className="font-serif text-2xl">Виділіть район</h2>
              </div>
              <button
                type="button"
                onClick={() => reloadPreview()}
                className="inline-flex items-center gap-2 rounded-[6px] border border-[#dfd7c8] bg-[#fffaf1] px-3 py-2 text-xs font-medium"
              >
                {previewLoading ? <Loader2 size={14} className="animate-spin" /> : <RotateCcw size={14} />}
                Оновити
              </button>
            </div>
            <div className="h-[430px] p-3 sm:h-[520px] lg:h-[640px]">
              <div className="h-full overflow-hidden rounded-[8px] border border-[#dfd7c8]">
                {areaMode === "grid" || areaMode === "hex" ? (
                  <HexagonalGrid
                    key={`${cityKey}-${areaMode}`}
                    bounds={city.bounds}
                    onZonesSelected={setSelectedZones}
                    gridType={areaMode === "hex" ? "hexagonal" : "square"}
                    hexSizeM={areaMode === "hex" ? 1000 : 800}
                  />
                ) : (
                  <MapSelector
                    key={cityKey}
                    center={city.center}
                    initialBounds={city.bounds}
                    onSelectionChange={(next) => next && setSelection(next)}
                  />
                )}
              </div>
            </div>
            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-[#dfd7c8] px-5 py-3 text-xs text-[#7a7466]">
              <span>Площа: {areaKm2(activeBounds).toFixed(2)} км²</span>
              <span>{areaMode === "grid" || areaMode === "hex" ? `Вибрано зон: ${selectedZones.length}` : "Можна намалювати власний контур"}</span>
            </div>
          </section>

          <aside className="space-y-4">
            <div className="overflow-hidden rounded-[10px] border border-[#dfd7c8] bg-[#fffaf1]">
              <div className="flex items-center justify-between border-b border-[#dfd7c8] px-4 py-3">
                <div>
                  <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-[#8a8173]">Preview</p>
                  <h2 className="font-serif text-xl">Як виглядатиме модель</h2>
                </div>
                <span className={`rounded-full px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] ${previewLoading || preview?.preview_status === "processing" ? "bg-[#efe6d8] text-[#8a5a40]" : previewError || preview?.preview_status === "failed" ? "bg-[#f3ded8] text-[#8a3b2f]" : "bg-[#dde9df] text-[#1f5b49]"}`}>
                  {previewLoading || preview?.preview_status === "processing" ? "рахується" : previewError || preview?.preview_status === "failed" ? "помилка" : "готово"}
                </span>
              </div>
              <div className="h-[360px]">
                <FastPreview3D preview={preview} loading={previewLoading} error={previewError} visibleLayers={layers} material={material} onReset={reloadPreview} />
              </div>
              <div className="flex flex-wrap gap-2 border-t border-[#dfd7c8] p-3">
                {LAYER_META.filter((item) => layers[item.key]).map((item) => (
                  <span key={item.key} className="rounded-full border border-[#1f5b49] bg-[#e8f0e9] px-3 py-1 text-xs text-[#1f5b49]">✓ {item.label}</span>
                ))}
              </div>
            </div>

            <div className="rounded-[10px] border border-[#dfd7c8] bg-[#fffaf1] p-5">
              <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-[#8a8173]">Підсумок</p>
              <div className="mt-4 divide-y divide-[#e4dccd] text-sm">
                {[
                  ["Місто", city.label],
                  ["Розмір", `${modelSizeMm / 10} см · ${modelSizeMm} мм`],
                  ["Площа", `${areaKm2(activeBounds).toFixed(2)} км²`],
                  ["Шари", `${activeLayers}/5`],
                  ["Матеріал", MATERIALS.find((item) => item.key === material)?.label ?? "Білий"],
                  ["Preview", preview ? `${preview.metrics.elapsed_ms} мс · ${preview.metrics.buildings} буд.` : "ще немає"],
                ].map(([label, value]) => (
                  <div key={label} className="flex justify-between gap-4 py-2">
                    <span className="text-[#8a8173]">{label}</span>
                    <span className="text-right font-medium">{value}</span>
                  </div>
                ))}
              </div>
              <div className="mt-4 rounded-[8px] bg-[#f1eadf] p-4">
                <p className="text-xs uppercase tracking-[0.14em] text-[#8a8173]">Орієнтовна вартість</p>
                <div className="mt-1 font-serif text-4xl">{money(price)} ₴</div>
                <p className="mt-1 text-xs text-[#7a7466]">Точно після перевірки ділянки</p>
              </div>
            </div>

            <div id="order" className="rounded-[10px] bg-[#1f2420] p-5 text-[#fffaf1]">
              <h2 className="font-serif text-2xl">Залиште заявку</h2>
              <p className="mt-2 text-sm leading-6 text-white/70">Без передоплати. Ми перевіримо ділянку і напишемо вам з точним preview та ціною.</p>
              <div className="mt-4 space-y-2">
                <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Імʼя" className="w-full rounded-[6px] border border-white/10 bg-white/8 px-3 py-3 text-sm outline-none placeholder:text-white/40" />
                <input value={contact} onChange={(e) => setContact(e.target.value)} placeholder="+380 / Telegram / email" className="w-full rounded-[6px] border border-white/10 bg-white/8 px-3 py-3 text-sm outline-none placeholder:text-white/40" />
                <textarea value={comment} onChange={(e) => setComment(e.target.value)} placeholder="Коментар, адреса, побажання" rows={3} className="w-full resize-none rounded-[6px] border border-white/10 bg-white/8 px-3 py-3 text-sm outline-none placeholder:text-white/40" />
              </div>
              <button
                type="button"
                onClick={submitOrder}
                disabled={orderState === "sending"}
                className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-[6px] bg-[#1f5b49] px-4 py-3 text-sm font-semibold text-white disabled:opacity-60"
              >
                {orderState === "sending" ? <Loader2 size={16} className="animate-spin" /> : orderState === "sent" ? <Check size={16} /> : <Phone size={16} />}
                {orderState === "sent" ? "Заявку надіслано" : "Отримати preview і ціну"}
                {orderState === "idle" && <ArrowRight size={16} />}
              </button>
              <button
                type="button"
                onClick={confirmFullModelInAccount}
                disabled={accountGenerationState === "sending" || (!authConfigured && !user) || authLoading}
                className="mt-2 inline-flex w-full items-center justify-center gap-2 rounded-[6px] border border-white/15 bg-white/8 px-4 py-3 text-sm font-semibold text-white disabled:opacity-60"
              >
                {accountGenerationState === "sending" ? <Loader2 size={16} className="animate-spin" /> : user ? <Layers3 size={16} /> : <LogIn size={16} />}
                {accountGenerationState === "sent"
                  ? "Повна модель додана в кабінет"
                  : user
                    ? "Створити повну модель у кабінеті"
                    : "Увійти й отримати 10 генерацій"}
              </button>
              {orderState === "error" && <p className="mt-2 text-xs text-[#f3b5a9]">Додайте контакт або спробуйте ще раз.</p>}
              {accountGenerationState === "limit" && <p className="mt-2 text-xs text-[#f3b5a9]">Ліміт 10 генерацій вичерпано. Напишіть нам для продовження.</p>}
              {accountGenerationState === "error" && <p className="mt-2 text-xs text-[#f3b5a9]">Не вдалося запустити повну генерацію. Перевірте кабінет або спробуйте ще раз.</p>}
            </div>
          </aside>
        </section>

        {confirmGenerationOpen && (
          <div className="fixed inset-0 z-[1000] grid place-items-center bg-black/35 px-4">
            <div className="w-full max-w-[460px] rounded-[10px] border border-[#dfd7c8] bg-[#fffaf1] p-6 shadow-[0_22px_80px_rgba(0,0,0,0.22)]">
              <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-[#8a8173]">Підтвердження</p>
              <h2 className="mt-2 font-serif text-3xl">Запустити повну генерацію?</h2>
              <p className="mt-3 text-sm leading-6 text-[#71695e]">
                Це створює справжню 3D-модель у вашому кабінеті і використовує одну з 10 безкоштовних генерацій. Процес може тривати довго.
              </p>
              <div className="mt-5 rounded-[8px] bg-[#f1eadf] p-4 text-sm">
                <div className="flex justify-between gap-4 py-1"><span className="text-[#8a8173]">Місто</span><b>{city.label}</b></div>
                <div className="flex justify-between gap-4 py-1"><span className="text-[#8a8173]">Режим</span><b>{areaMode === "hex" ? "Гексагони" : areaMode === "grid" ? "Сітка зон" : "Ділянка"}</b></div>
                <div className="flex justify-between gap-4 py-1"><span className="text-[#8a8173]">Зон</span><b>{selectedZones.length || 1}</b></div>
                <div className="flex justify-between gap-4 py-1"><span className="text-[#8a8173]">Розмір</span><b>{modelSizeMm / 10} см</b></div>
              </div>
              <div className="mt-5 grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => setConfirmGenerationOpen(false)}
                  className="rounded-[6px] border border-[#dfd7c8] bg-[#fffaf1] px-4 py-3 text-sm font-semibold text-[#1f2420]"
                >
                  Скасувати
                </button>
                <button
                  type="button"
                  onClick={startFullModelInAccount}
                  className="rounded-[6px] bg-[#1f5b49] px-4 py-3 text-sm font-semibold text-white"
                >
                  Так, створити
                </button>
              </div>
            </div>
          </div>
        )}

        <section id="how" className="border-t border-[#dfd7c8] bg-[#eee8dd] px-5 py-10 lg:px-9">
          <p className="font-mono text-[10px] uppercase tracking-[0.24em] text-[#8a8173]">Три кроки</p>
          <h2 className="mt-2 font-serif text-4xl">Виберіть. Налаштуйте. Тримайте у руках.</h2>
          <div className="mt-7 grid gap-4 md:grid-cols-3">
            {[
              ["01", "Оберіть район", "Виділіть на карті будь-яку частину міста: від кварталу до кількох зон."],
              ["02", "Увімкніть потрібні шари", "Preview оновлюється швидко і показує спрощену форму майбутньої моделі."],
              ["03", "Ми підготуємо модель", "Повну модель з пазами, експортом і перевіркою друку робимо після заявки."],
            ].map(([n, title, text]) => (
              <div key={n} className="rounded-[8px] border border-[#dfd7c8] bg-[#fffaf1] p-6">
                <p className="font-mono text-[10px] text-[#1f5b49]">{n}</p>
                <h3 className="mt-3 font-serif text-2xl">{title}</h3>
                <p className="mt-2 text-sm leading-6 text-[#71695e]">{text}</p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
