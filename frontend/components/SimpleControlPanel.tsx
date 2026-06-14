"use client";

import { useCallback, useEffect, useState } from "react";
import { Loader2, Play, Download, MapPin, Check, Sparkles, ShoppingBag } from "lucide-react";
import { useTranslations } from "next-intl";
import { useGenerationStore } from "@/store/generation-store";
import { MAP_TEMPLATES, MAP_STYLE_PRESETS } from "@/lib/templates";
import { buildMapRequest, SIMPLE_SIZES, GPX_MAX_M_PER_MM } from "@/lib/generation";
import { OrderDialog } from "@/components/OrderDialog";
import { StickyActionBar } from "@/components/StickyActionBar";
import { useAuth } from "@/components/AuthProvider";
import { gatedDownload } from "@/lib/download";
import { fetchQuote, type Quote } from "@/lib/pricing";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

/**
 * Simple, preset-first map builder shown by default.
 * Three decisions: location (featured district card or draw) → style → size → Generate.
 * All fine-grained sliders live in the full ControlPanel ("Про" mode).
 */
export function SimpleControlPanel({
  availableCities,
  selectedCityKey,
  onCityChange,
  onAdvanced,
  showStickyBar = true,
}: {
  availableCities?: Record<string, { center: [number, number]; bounds: any }>;
  selectedCityKey?: string;
  onCityChange?: (key: string) => void;
  onAdvanced?: () => void;
  // Панель монтується ДВІЧІ (desktop aside + mobile section). StickyActionBar
  // — портал у <body>, тож обидві копії малювали його → ДВА бари на мобільному
  // (+ inline-кнопка = «3 кнопки генерації»). Малюємо лише з мобільної копії.
  showStickyBar?: boolean;
}) {
  const t = useTranslations("simple");
  const tOrder = useTranslations("order");
  const s = useGenerationStore();
  const {
    selectedArea, setSelectedArea,
    isGenerating, downloadUrl, progress, status, printQuality,
    taskGroupId, setTaskGroup, setActiveTaskId, setGenerating,
    setDownloadUrl, setTaskStatuses, updateProgress,
    modelSizeMm, setModelSizeMm, previewMode, setPreviewMode, setGpxFocus,
    setTerrainEnabled,
    setPreviewIncludeBuildings, setPreviewIncludeRoads,
    setPreviewIncludeWater, setPreviewIncludeParks,
  } = s;

  const [styleId, setStyleId] = useState<string>("full");
  // МАГНІТ/ПАННО/GPX — у zustand store, НЕ useState: панель змонтована двічі
  // (desktop + mobile), локальний стан розсинхронізовувався між копіями і
  // вибір губився при генерації з іншої копії.
  const magnetMode = s.simpleMagnetMode;
  const setMagnetMode = s.setSimpleMagnetMode;
  const mapLabel = s.simpleMapLabel;
  const setMapLabel = s.setSimpleMapLabel;
  const panelMode = s.simplePanelMode;
  const setPanelMode = s.setSimplePanelMode;
  // D4 GPX-трек: точки живуть у gpxFocus (їх же використовує карта-оверлей)
  const gpxTrack = s.gpxFocus?.points ?? null;
  const gpxName = s.gpxName;
  const setGpxName = s.setGpxName;
  const gpxNote = s.gpxNote;
  const setGpxNote = s.setGpxNote;
  // E4 ШЕРИНГ: рендер моделі → /share/{task} з og:image
  const [shareBusy, setShareBusy] = useState(false);
  const [shareCopied, setShareCopied] = useState(false);

  const doShare = async () => {
    if (!taskGroupId) return;
    setShareBusy(true);
    try {
      const { capturePreviewImages } = await import("@/lib/capturePreview");
      const shots = await capturePreviewImages();
      const png = shots.find((s) => s.startsWith("data:image/png"));
      if (png) {
        await fetch(`${API_BASE}/api/share/preview`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task_id: taskGroupId, image: png }),
        });
      }
      const url = `${window.location.origin}/share/${taskGroupId}`;
      if (typeof navigator.share === "function") {
        await navigator.share({ url, title: "Monadruk" }).catch(() => {});
      } else {
        await navigator.clipboard.writeText(url);
        setShareCopied(true);
        setTimeout(() => setShareCopied(false), 2500);
      }
    } catch { /* ignore */ }
    setShareBusy(false);
  };
  const [quote, setQuote] = useState<Quote | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTemplate, setActiveTemplate] = useState<string | null>(null);
  const [orderOpen, setOrderOpen] = useState(false);
  const [dlBusy, setDlBusy] = useState(false);
  const { getIdToken, openLogin } = useAuth();
  const [quota, setQuota] = useState<{ remaining: number; limit: number; isAdmin?: boolean } | null>(null);

  // Лічильник безкоштовних завантажень (вхід + 5 безкоштовних, далі замовлення).
  const refreshQuota = useCallback(async () => {
    try {
      const token = await getIdToken();
      if (!token) { setQuota(null); return; }
      const r = await fetch(`${API_BASE}/api/account/quota`, { headers: { Authorization: `Bearer ${token}` } });
      if (!r.ok) return;
      const d = await r.json();
      const q = d?.quota || d;
      if (q) setQuota({ remaining: Number(q.remaining ?? 0), limit: Number(q.limit ?? 5), isAdmin: Boolean(q.is_admin) });
    } catch { /* ignore */ }
  }, [getIdToken]);
  useEffect(() => { refreshQuota(); }, [refreshQuota]);

  // Жива орієнтовна ціна — оновлюється при зміні розміру/стилю (relief = +надбавка).
  useEffect(() => {
    let alive = true;
    const relief = MAP_STYLE_PRESETS.find((p) => p.id === styleId)?.layers.terrain ?? false;
    fetchQuote("map", modelSizeMm, relief).then((q) => { if (alive) setQuote(q); });
    return () => { alive = false; };
  }, [modelSizeMm, styleId]);

  // Fallback-ціна (поки /api/quote вантажиться): з локальної таблиці розмірів,
  // щоб sticky-бар не показував порожнє «—» на першому екрані.
  const simpleFallbackPrice = (() => {
    const near = SIMPLE_SIZES.reduce((best, z) =>
      Math.abs(z.mm - modelSizeMm) < Math.abs(best.mm - modelSizeMm) ? z : best, SIMPLE_SIZES[0]);
    return `≈ ${near.price} ₴`;
  })();

  const doGatedDownload = async () => {
    setDlBusy(true);
    const res = await gatedDownload({
      taskId: taskGroupId, downloadUrl,
      meta: { title: selectedCityKey, city: selectedCityKey, product_type: "map" },
      getIdToken, openLogin,
      onLimit: () => window.dispatchEvent(new CustomEvent("monadruk:open-contact", {
        detail: { message: t("limitMsg") },
      })),
    });
    if (res.quota && typeof res.quota.remaining === "number") {
      setQuota((q) => ({ remaining: res.quota!.remaining as number, limit: q?.limit ?? 5, isAdmin: q?.isAdmin }));
    } else if (res.status === "ok") {
      refreshQuota();
    }
    setDlBusy(false);
  };

  const cityKeys = availableCities ? Object.keys(availableCities) : [];
  const featured = MAP_TEMPLATES.filter((t) => t.cityKey === selectedCityKey);

  // Poll task status (this panel can be the only one mounted).
  // UX-FIX: термінальні стани (cancelled/зниклий task після рестарту сервера)
  // ОБОВʼЯЗКОВО розблоковують UI — інакше «Генерація N%» висіла назавжди
  // і переживала перезавантаження сторінки (taskGroupId у localStorage).
  useEffect(() => {
    if (!taskGroupId) return;
    let stop = false;
    let pollFails = 0;
    const iv = setInterval(async () => {
      try {
        const { api } = await import("@/lib/api");
        const r: any = await api.getStatus(taskGroupId);
        if (stop) return;
        pollFails = 0;
        setTaskStatuses({ [r.task_id]: r });
        // D3 ПАННО: batch-статус — агрегуємо прогрес плиток; коли всі готові,
        // даємо посилання на zip-архів з усіма плитками + layout.png
        if (r.status === "multiple") {
          const total = Number(r.total || 0);
          const done = Number(r.completed || 0);
          const subTasks: any[] = r.tasks || [];
          const avg = subTasks.length
            ? Math.round(subTasks.reduce((acc, st) => acc + Number(st.progress || 0), 0) / subTasks.length)
            : 0;
          updateProgress(avg, `${t("panelTiles")}: ${done}/${total}`);
          const allTerminal = subTasks.length > 0 && subTasks.every(
            (st) => st.status === "completed" || st.status === "failed" || st.status === "cancelled",
          );
          if (subTasks.some((st) => st.status === "failed")) {
            setGenerating(false);
            setError(t("errGen"));
            clearInterval(iv);
          } else if (total > 0 && done === total) {
            setDownloadUrl(`/api/zones/${taskGroupId}/download_all`);
            setGenerating(false);
            clearInterval(iv);
          } else if (allTerminal) {
            // Скасований batch: тихо розблоковуємо — юзер може почати заново
            setGenerating(false);
            clearInterval(iv);
          }
          return;
        }
        updateProgress(r.progress, r.message);
        if (r.status === "completed") {
          setDownloadUrl(r.download_url);
          (s as any).setPrintQuality?.(r.print_quality ?? null);
          setGenerating(false);
          clearInterval(iv);
        } else if (r.status === "failed" || r.status === "cancelled") {
          setGenerating(false);
          if (r.status === "failed") setError(r.message || t("errGen"));
          clearInterval(iv);
        }
      } catch {
        // 404/мережа: 4 поспіль (~10с) = задача зникла (рестарт сервера) —
        // розблоковуємо UI замість вічного спінера
        pollFails += 1;
        if (pollFails >= 4 && !stop) {
          setGenerating(false);
          setError(t("errStale"));
          clearInterval(iv);
        }
      }
    }, 2500);
    return () => { stop = true; clearInterval(iv); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskGroupId]);

  // UX-FIX: «Замовити друк» зі сторонніх місць (Швидкий статус на мобілці)
  // відкриває OrderDialog цієї панелі через глобальну подію.
  useEffect(() => {
    const open = () => setOrderOpen(true);
    window.addEventListener("monadruk:open-order", open);
    return () => window.removeEventListener("monadruk:open-order", open);
  }, []);

  // UX-FIX: явне скасування генерації — DELETE на бек + миттєве розблокування
  const cancelGeneration = async () => {
    try {
      const { api } = await import("@/lib/api");
      if (taskGroupId) await api.cancelTask(taskGroupId).catch(() => {});
    } finally {
      setGenerating(false);
      updateProgress(0, "");
    }
  };

  const applyStyle = (id: string) => {
    const preset = MAP_STYLE_PRESETS.find((p) => p.id === id);
    if (!preset) return;
    setStyleId(id);
    setPreviewIncludeBuildings(preset.layers.buildings);
    setPreviewIncludeRoads(preset.layers.roads);
    setPreviewIncludeWater(preset.layers.water);
    setPreviewIncludeParks(preset.layers.parks);
    setTerrainEnabled(preset.layers.terrain);
  };

  // Чернетка конструктора: зона/стиль/розмір переживають перезавантаження.
  // Відновлюємо РАЗ при маунті; зберігаємо з debounce при змінах.
  useEffect(() => {
    try {
      const raw = localStorage.getItem("monadruk:draft:create");
      if (!raw) return;
      const d = JSON.parse(raw);
      if (d.styleId && MAP_STYLE_PRESETS.some((p) => p.id === d.styleId)) applyStyle(d.styleId);
      if (typeof d.modelSizeMm === "number" && d.modelSizeMm >= 40) setModelSizeMm(d.modelSizeMm);
      // КРИТИЧНО: JSON.parse повертає plain object, НЕ Leaflet LatLngBounds —
      // прямий setSelectedArea(d.selectedArea) валив /create і /keychains
      // («getNorth/getCenter is not a function», store спільний між роутами).
      // Реконструюємо справжній L.LatLngBounds з координат чернетки.
      if (!selectedArea && d.selectedArea && typeof d.selectedArea === "object") {
        const sw = d.selectedArea._southWest;
        const ne = d.selectedArea._northEast;
        if (
          sw && ne &&
          typeof sw.lat === "number" && typeof sw.lng === "number" &&
          typeof ne.lat === "number" && typeof ne.lng === "number"
        ) {
          import("leaflet")
            .then((L) => {
              setSelectedArea(new L.LatLngBounds([sw.lat, sw.lng], [ne.lat, ne.lng]) as any);
            })
            .catch(() => { /* ignore */ });
        }
      }
    } catch { /* ignore */ }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    const timer = setTimeout(() => {
      try {
        localStorage.setItem("monadruk:draft:create", JSON.stringify({ selectedArea, styleId, modelSizeMm }));
      } catch { /* ignore */ }
    }, 800);
    return () => clearTimeout(timer);
  }, [selectedArea, styleId, modelSizeMm]);

  const pickTemplate = async (id: string) => {
    const tpl = MAP_TEMPLATES.find((t) => t.id === id);
    if (!tpl) return;
    setActiveTemplate(id);
    setError(null);
    const [lat, lon] = tpl.center;
    const span = tpl.span;
    const lonPad = span / Math.max(Math.cos((lat * Math.PI) / 180), 0.2);
    const L = await import("leaflet");
    setSelectedArea(new L.LatLngBounds([lat - span, lon - lonPad], [lat + span, lon + lonPad]) as any);
  };

  const handleGenerate = async () => {
    if (!selectedArea) { setError(t("errSelectArea")); return; }
    setError(null);
    setGenerating(true);
    try {
      // Derive layer flags from the CURRENTLY SELECTED style preset, not from the
      // store. The store can lag the visible selection (it isn't synced on mount),
      // which previously sent terrain_enabled=false even when "З рельєфом" was
      // highlighted → flat model. Reading the preset guarantees payload == UI.
      const preset = MAP_STYLE_PRESETS.find((p) => p.id === styleId);
      const layerTerrain = preset ? preset.layers.terrain : s.terrainEnabled;
      const layerBuildings = preset ? preset.layers.buildings : s.previewIncludeBuildings;
      const layerRoads = preset ? preset.layers.roads : s.previewIncludeRoads;
      const layerWater = preset ? preset.layers.water : s.previewIncludeWater;
      const layerParks = preset ? preset.layers.parks : s.previewIncludeParks;
      const req = buildMapRequest({
        north: selectedArea.getNorth(), south: selectedArea.getSouth(),
        east: selectedArea.getEast(), west: selectedArea.getWest(),
        roadWidthMultiplier: s.roadWidthMultiplier, roadHeightMm: s.roadHeightMm, roadEmbedMm: s.roadEmbedMm,
        buildingMinHeight: s.buildingMinHeight, buildingHeightMultiplier: s.buildingHeightMultiplier,
        buildingFoundationMm: s.buildingFoundationMm, buildingEmbedMm: s.buildingEmbedMm,
        waterDepth: s.waterDepth, terrainEnabled: magnetMode ? false : layerTerrain, terrainZScale: s.terrainZScale,
        terrainBaseThicknessMm: magnetMode ? 3.0 : s.terrainBaseThicknessMm, terrainResolution: s.terrainResolution,
        terrariumZoom: s.terrariumZoom, exportFormat: s.exportFormat,
        modelSizeMm: magnetMode ? 60 : s.modelSizeMm,
        isAmsMode: s.isAmsMode,
        // Панно = повні 3D-плитки: магніт/превʼю вимикаються примусово
        flatPlateMode: panelMode > 0 ? false : magnetMode ? true : s.flatPlateMode,
        previewMode: panelMode > 0 || magnetMode ? false : s.previewMode,
        magnetPocket: panelMode > 0 ? false : magnetMode,
        mapLabel: magnetMode && panelMode === 0 ? mapLabel : "",
        gpxTrack,
        previewIncludeBase: s.previewIncludeBase, previewIncludeRoads: layerRoads,
        previewIncludeBuildings: layerBuildings, previewIncludeWater: layerWater,
        previewIncludeParks: layerParks,
        // Панно: плитки ріжуться строго по своїх bbox — фігурний полігон
        // (rounded-rect/коло/серце з /create) сюди потрапляти НЕ повинен.
        zonePolygonCoords: panelMode > 0 ? null : s.zonePolygonCoords,
      });
      const { api } = await import("@/lib/api");
      // D3 ПАННО: ділимо зону на R×C плиток (row 0 = ПІВНІЧ, col 0 = захід)
      // і шлемо batch у /api/generate-zones — бек гарантує ідеальні шви
      // (preserve_global_xy + спільний elevation_ref + grid_step).
      if (panelMode > 0) {
        const N = selectedArea.getNorth(), S = selectedArea.getSouth();
        const E = selectedArea.getEast(), W = selectedArea.getWest();
        const G = panelMode;
        const zones: any[] = [];
        for (let r = 0; r < G; r++) {
          for (let c = 0; c < G; c++) {
            const zn = N - (r * (N - S)) / G;
            const zs = N - ((r + 1) * (N - S)) / G;
            const zw = W + (c * (E - W)) / G;
            const ze = W + ((c + 1) * (E - W)) / G;
            zones.push({
              id: `tile_${r}_${c}`,
              geometry: { type: "Polygon", coordinates: [[[zw, zs], [ze, zs], [ze, zn], [zw, zn], [zw, zs]]] },
              properties: { row: r, col: c },
            });
          }
        }
        const batch = await api.generateZones(zones, req as any);
        const ids = batch.all_task_ids?.length ? batch.all_task_ids : [batch.task_id];
        setTaskGroup(batch.task_id, ids);
        setActiveTaskId(ids[0] ?? null);
        // СКЛАДЕНЕ ПРЕВʼЮ: показуємо всі плитки разом (та сама механіка, що
        // «Показати всі зони» у Профі) + row/col, щоб кожна стала на місце.
        s.setShowAllZones(true);
        const meta: Record<string, { zoneId: string; row?: number; col?: number }> = {};
        for (let i = 0; i < ids.length && i < zones.length; i += 1) {
          meta[String(ids[i])] = {
            zoneId: String(zones[i].id),
            row: zones[i].properties?.row,
            col: zones[i].properties?.col,
          };
        }
        s.setBatchZoneMetaByTaskId(meta);
        return;
      }
      const r = await api.generateModel(req as any);
      setTaskGroup(r.task_id, [r.task_id]);
      setActiveTaskId(r.task_id);
      s.setShowAllZones(false);
      s.setBatchZoneMetaByTaskId({});
    } catch (e: any) {
      setError(e?.message || t("errGen"));
      setGenerating(false);
    }
  };

  // ЗАМОВИТИ ОДРАЗУ: користувач не мусить чекати 1-3 хв перед замовленням.
  // Клік стартує генерацію у фоні (set taskGroupId) і ВІДРАЗУ відкриває форму.
  // Поки юзер заповнює контакти — модель готується; замовлення несе task_id,
  // тож оператор отримає і конфіг, і готову модель (бек приймає order і без
  // завершеної генерації).
  const orderNow = () => {
    if (!selectedArea) { setError(t("errSelectArea")); return; }
    if (!downloadUrl && !isGenerating) handleGenerate();
    setOrderOpen(true);
  };

  const downloadHref = downloadUrl
    ? (downloadUrl.startsWith("http") ? downloadUrl : `${API_BASE}${downloadUrl}`)
    : null;

  return (
    <div className="h-full overflow-y-auto px-4 py-4 sm:px-5">
      <div className="space-y-5 pb-8">
        {/* 1. City */}
        {cityKeys.length > 0 && onCityChange && (
          <div>
            <div className="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
              <MapPin size={14} /> {t("step1city")}
            </div>
            <select
              value={selectedCityKey}
              onChange={(e) => { onCityChange(e.target.value); setActiveTemplate(null); }}
              className="w-full rounded-2xl border border-[var(--surface-border)] bg-white/90 px-4 py-3 text-sm font-semibold text-[var(--text-primary)] outline-none transition focus:border-[rgba(11,92,87,0.35)]"
            >
              {cityKeys.map((k) => (
                <option key={k} value={k}>{k === "Kyiv" ? t("kyiv") : k === "Khmelnytskyi" ? t("khmel") : k}</option>
              ))}
            </select>
          </div>
        )}

        {/* 2. Featured districts */}
        <div>
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
              <Sparkles size={14} /> {t("step2districts")}
            </div>
            <span className="text-[11px] text-[var(--text-secondary)]">{t("orDraw")}</span>
          </div>
          {featured.length > 0 ? (
            <div className="grid grid-cols-1 gap-2">
              {featured.map((t) => {
                const active = activeTemplate === t.id;
                return (
                  <button
                    key={t.id}
                    type="button"
                    onClick={() => pickTemplate(t.id)}
                    className={`flex items-center gap-3 rounded-[20px] border px-3 py-3 text-left transition ${
                      active
                        ? "border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.1)] shadow-[0_10px_24px_rgba(11,92,87,0.14)]"
                        : "border-[var(--surface-border)] bg-white/80 hover:border-[rgba(11,92,87,0.25)]"
                    }`}
                  >
                    <span className="relative grid h-12 w-12 shrink-0 place-items-center overflow-hidden rounded-[14px] bg-[rgba(46,74,58,0.08)] text-[var(--accent-strong)]">
                      {/* Іконка-плейсхолдер (поки нема фото шаблону) — щоб не зяяв
                          порожній квадрат; реальне прев'ю накладається зверху. */}
                      <MapPin size={18} className="opacity-45" />
                      <img
                        src={`/templates/${t.id}.webp`}
                        alt={t.district}
                        className="absolute inset-0 h-full w-full object-cover"
                        onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
                      />
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="flex items-center gap-2">
                        <span className="truncate text-sm font-semibold text-[var(--text-primary)]">{t.district}</span>
                        {t.tag && <span className="rounded-full bg-[var(--accent-strong)] px-2 py-0.5 text-[9px] font-bold uppercase tracking-wide text-white">{t.tag}</span>}
                      </span>
                      <span className="mt-0.5 block truncate text-[11px] text-[var(--text-secondary)]">{t.blurb}</span>
                    </span>
                    {active && <Check size={16} className="shrink-0 text-[var(--accent-strong)]" />}
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="rounded-[18px] border border-dashed border-[var(--surface-border)] bg-white/60 px-4 py-4 text-center text-xs text-[var(--text-secondary)]">
              {t("noDistricts")}
            </div>
          )}
          <div className="mt-2 text-[11px] text-[var(--text-secondary)]">
            {selectedArea ? t("areaSelected") : t("areaNotSelected")}
          </div>
        </div>

        {/* 3. Style */}
        <div>
          <div className="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
            {t("step3style")}
          </div>
          <div className="grid grid-cols-2 gap-2">
            {MAP_STYLE_PRESETS.map((p) => {
              const active = styleId === p.id;
              return (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => applyStyle(p.id)}
                  className={`rounded-[18px] border px-3 py-3 text-left transition ${
                    active
                      ? "border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.1)]"
                      : "border-[var(--surface-border)] bg-white/80 hover:border-[rgba(11,92,87,0.25)]"
                  }`}
                >
                  <span className="block text-sm font-semibold text-[var(--text-primary)]">{p.label}</span>
                  <span className="mt-0.5 block text-[11px] leading-4 text-[var(--text-secondary)]">{p.blurb}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* 4. Size */}
        <div>
          <div className="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">
            {t("step4size")}
          </div>
          <div className="grid grid-cols-4 gap-2">
            {SIMPLE_SIZES.map((sz) => {
              const active = Math.abs(modelSizeMm - sz.mm) < 1;
              return (
                <button
                  key={sz.key}
                  type="button"
                  onClick={() => setModelSizeMm(sz.mm)}
                  className={`rounded-[16px] border px-2 py-3 text-center transition ${
                    active
                      ? "border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.1)]"
                      : "border-[var(--surface-border)] bg-white/80 hover:border-[rgba(11,92,87,0.25)]"
                  }`}
                >
                  <span className="block text-base font-bold text-[var(--text-primary)]">{sz.label}</span>
                  <span className="block text-[10px] text-[var(--text-secondary)]">{sz.cm}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Магніт: плаский формат 6 см з кишенею під магніт у дні */}
        <button
          type="button"
          onClick={() => setMagnetMode(!magnetMode)}
          className={`w-full rounded-[18px] border px-4 py-3 text-left transition ${
            magnetMode
              ? "border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.1)]"
              : "border-[var(--surface-border)] bg-white/80 hover:border-[rgba(11,92,87,0.25)]"
          }`}
        >
          <span className="flex items-center justify-between text-sm font-semibold text-[var(--text-primary)]">
            🧲 {t("magnetToggle")}
            {magnetMode && <Check size={16} className="text-[var(--accent-strong)]" />}
          </span>
          <span className="mt-0.5 block text-[11px] leading-4 text-[var(--text-secondary)]">{t("magnetHint")}</span>
        </button>
        {magnetMode && (
          <input
            value={mapLabel}
            onChange={(e) => setMapLabel(e.target.value.toUpperCase().slice(0, 24))}
            placeholder={t("mapLabelPh")}
            className="w-full rounded-[18px] border border-[var(--surface-border)] bg-white/90 px-4 py-3 text-sm font-semibold uppercase tracking-[0.08em] text-[var(--text-primary)] outline-none transition focus:border-[rgba(11,92,87,0.4)]"
          />
        )}

        {/* D4 GPX-трек: маршрут (біг/похід/вело) як підвищений шар поверх мапи */}
        <div className="rounded-[18px] border border-[var(--surface-border)] bg-white/80 px-4 py-3">
          <label className="flex cursor-pointer items-center justify-between gap-3 text-sm font-semibold text-[var(--text-primary)]">
            <span>🏃 {t("gpxUpload")}</span>
            <input
              type="file"
              accept=".gpx,application/gpx+xml"
              data-testid="gpx-input"
              className="hidden"
              onChange={async (e) => {
                const file = e.target.files?.[0];
                e.target.value = "";
                if (!file) return;
                try {
                  const { parseGpx, gpxBounds } = await import("@/lib/gpx");
                  const parsed = parseGpx(await file.text());
                  if (!parsed) { setGpxName(null); setGpxNote(null); setGpxFocus(null); setError(t("gpxErr")); return; }
                  setError(null);
                  setGpxName(parsed.name || file.name.replace(/\.gpx$/i, ""));
                  // Авто-фокус: зона і карта їдуть до треку (раніше трек з іншого
                  // міста мовчки обрізався по чужій зоні → у моделі його не було).
                  const bb = gpxBounds(parsed.points);
                  if (bb) {
                    const [w, s_, e_, n] = bb;
                    const latC = (s_ + n) / 2;
                    const wM = (e_ - w) * 111320 * Math.max(Math.cos((latC * Math.PI) / 180), 0.2);
                    const hM = (n - s_) * 111320;
                    const spanM = Math.max(wM, hM) * 1.1;
                    // Авто-розмір: найменший пресет, чия зона (мм × 10 м, 1:10000)
                    // покриває трек. Якщо трек більший — НЕ ріжемо його: зона
                    // розширюється до GPX_MAX_M_PER_MM (плоска мапа друкується
                    // шарами, точний масштаб не критичний — дрібніші деталі).
                    const fit = SIMPLE_SIZES.find((sz) => sz.mm * 10 >= spanM);
                    const target = fit ?? SIMPLE_SIZES[SIMPLE_SIZES.length - 1];
                    if (target.mm > (modelSizeMm || 0)) setModelSizeMm(target.mm);
                    if (fit) setGpxNote(t("gpxMoved"));
                    else if (spanM <= target.mm * GPX_MAX_M_PER_MM) {
                      const scale = Math.round((spanM / target.mm) * 1000);
                      setGpxNote(t("gpxScaled", { scale: `1:${scale}` }));
                    } else setGpxNote(t("gpxPartial"));
                    setGpxFocus({ west: w, south: s_, east: e_, north: n, points: parsed.points });
                  }
                } catch { setError(t("gpxErr")); }
              }}
            />
            <span className="rounded-full border border-[var(--surface-border)] bg-white px-3 py-1 text-[12px] font-semibold text-[var(--accent-strong)]">
              {gpxTrack ? t("gpxReplace") : t("gpxChoose")}
            </span>
          </label>
          {gpxTrack ? (
            <div className="mt-2 space-y-1">
              <div className="flex items-center justify-between gap-2 text-[12px] text-[var(--text-secondary)]">
                <span className="truncate">✓ {gpxName} · {gpxTrack.length} {t("gpxPoints")}</span>
                <button type="button" onClick={() => { setGpxName(null); setGpxNote(null); setGpxFocus(null); }} className="shrink-0 font-semibold text-red-700 hover:underline">
                  {t("gpxClear")}
                </button>
              </div>
              {gpxNote && <p data-testid="gpx-note" className="text-[11px] leading-4 font-semibold text-[var(--accent-strong)]">{gpxNote}</p>}
              <p className="text-[11px] leading-4 text-[var(--text-secondary)]">{t("gpxPrivacy")}</p>
            </div>
          ) : (
            <p className="mt-1 text-[11px] leading-4 text-[var(--text-secondary)]">{t("gpxHint")}</p>
          )}
        </div>

        {/* D3 ПАННО: серія зшитих плиток 2×2/3×3 + zip зі схемою розкладки */}
        <div className="rounded-[18px] border border-[var(--surface-border)] bg-white/80 px-4 py-3">
          <div className="flex items-center justify-between gap-2 text-sm font-semibold text-[var(--text-primary)]">
            <span>🖼 {t("panelToggle")}</span>
            <div className="flex gap-1.5" data-testid="panel-chips">
              {([[0, t("panelOff")], [2, "2×2"], [3, "3×3"]] as Array<[0 | 2 | 3, string]>).map(([mode, label]) => (
                <button
                  key={`panel-${mode}`}
                  type="button"
                  onClick={() => setPanelMode(mode)}
                  className={`rounded-full border px-3 py-1 text-[12px] font-semibold transition ${
                    panelMode === mode
                      ? "border-[rgba(11,92,87,0.4)] bg-[rgba(15,118,110,0.12)] text-[var(--accent-strong)]"
                      : "border-[var(--surface-border)] bg-white text-[var(--text-secondary)] hover:border-[rgba(11,92,87,0.25)]"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <p className="mt-1 text-[11px] leading-4 text-[var(--text-secondary)]">
            {panelMode > 0 ? t("panelHintOn", { tiles: panelMode * panelMode }) : t("panelHint")}
          </p>
        </div>

        {/* Generate */}
        <div className="space-y-3">
          <div className="flex items-center justify-center gap-1 rounded-full border border-[var(--surface-border)] bg-white/80 p-1 text-xs">
            <button
              type="button"
              onClick={() => setPreviewMode(true)}
              disabled={isGenerating}
              className={`flex-1 rounded-full px-3 py-1.5 font-semibold transition ${previewMode ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)]"}`}
            >
              {t("quickPreview")}
            </button>
            <button
              type="button"
              onClick={() => setPreviewMode(false)}
              disabled={isGenerating}
              className={`flex-1 rounded-full px-3 py-1.5 font-semibold transition ${!previewMode ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)]"}`}
            >
              {t("forPrint")}
            </button>
          </div>

          <button
            type="button"
            onClick={handleGenerate}
            disabled={!selectedArea || isGenerating}
            className="inline-flex min-h-13 w-full items-center justify-center gap-2 rounded-full bg-[var(--accent-strong)] px-5 py-3.5 text-sm font-bold text-white shadow-[0_16px_32px_rgba(11,92,87,0.24)] transition hover:bg-[var(--accent)] disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {isGenerating ? (
              <><Loader2 className="h-4 w-4 animate-spin" /> {t("generating")} {progress}%</>
            ) : panelMode > 0 ? (
              <><Play className="h-4 w-4" /> {t("generateTiles", { tiles: panelMode * panelMode })}</>
            ) : (
              <><Play className="h-4 w-4" /> {t("generate")}</>
            )}
          </button>
          {/* UX: чесне очікування — час генерації відомий заздалегідь */}
          {!isGenerating && (
            <p className="-mt-1 text-center text-[11px] text-[var(--text-secondary)]">
              {panelMode > 0 ? t("etaTiles", { tiles: panelMode * panelMode }) : t("etaSingle")}
            </p>
          )}
          {isGenerating && (
            <button
              type="button"
              onClick={cancelGeneration}
              className="-mt-1 inline-flex w-full items-center justify-center gap-1 text-[12px] font-semibold text-red-700 underline-offset-2 hover:underline"
            >
              {t("cancelGen")}
            </button>
          )}

          {/* «Замовити друк» — одразу після «Створити», завжди на видному місці */}
          <button
            type="button"
            onClick={() => setOrderOpen(true)}
            className="inline-flex min-h-13 w-full items-center justify-center gap-2 rounded-full bg-[var(--bronze,#8E6B3D)] px-5 py-3.5 text-[15px] font-extrabold text-white shadow-[0_16px_34px_rgba(142,107,61,0.32)] transition hover:opacity-90"
          >
            <ShoppingBag className="h-5 w-5" /> {t("orderPrint")}{quote ? ` · ${quote.formatted}` : ""}
          </button>

          {error && (
            <div className="rounded-[16px] border border-red-200 bg-red-50 px-4 py-2.5 text-xs text-red-700">{error}</div>
          )}

          {downloadUrl && printQuality && printQuality.status !== "ok" && (printQuality.warnings?.length ?? 0) > 0 && (
            <div className="rounded-[16px] border border-amber-200 bg-amber-50 px-4 py-2.5 text-xs text-amber-900">
              {t("qualityWarn")}
            </div>
          )}

          {downloadUrl && downloadUrl.includes("/download_all") && (
            <div data-testid="panel-howto" className="rounded-[16px] border border-[rgba(11,92,87,0.25)] bg-[rgba(15,118,110,0.07)] px-4 py-2.5 text-xs leading-5 text-[var(--text-primary)]">
              {t("panelHowto", { count: s.taskIds?.length ?? 0 })}
            </div>
          )}

          {downloadUrl && (
            <button
              type="button"
              onClick={doGatedDownload}
              disabled={dlBusy}
              className="inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-full border border-[var(--surface-border)] bg-white px-5 py-3 text-sm font-semibold text-[var(--text-primary)] transition hover:bg-white/70 disabled:opacity-60"
            >
              {dlBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}{" "}
              {downloadUrl?.includes("/download_all") ? t("panelZip") : t("downloadFile")}
            </button>
          )}

          {downloadUrl && !downloadUrl.includes("/download_all") && (
            <button
              type="button"
              onClick={doShare}
              disabled={shareBusy}
              className="inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-full border border-[var(--surface-border)] bg-white/70 px-5 py-3 text-sm font-semibold text-[var(--text-primary)] transition hover:bg-white disabled:opacity-60"
            >
              {shareBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <span aria-hidden>🔗</span>}{" "}
              {shareCopied ? t("shareCopied") : t("shareBtn")}
            </button>
          )}

          {downloadUrl && quota && !quota.isAdmin && (
            <div className={`-mt-1 text-center text-[12px] font-medium ${quota.remaining > 0 ? "text-[var(--text-secondary)]" : "text-amber-700"}`}>
              {quota.remaining > 0
                ? `Залишилось ${quota.remaining} з ${quota.limit} безкоштовних завантажень`
                : "Безкоштовні завантаження вичерпано — оформіть замовлення друку"}
            </div>
          )}

          {onAdvanced && (
            <button
              type="button"
              onClick={onAdvanced}
              className="w-full text-center text-[12px] text-[var(--text-secondary)] underline-offset-2 hover:underline"
            >
              {t("advancedHint")}
            </button>
          )}

        </div>
      </div>

      <OrderDialog
        open={orderOpen}
        onClose={() => setOrderOpen(false)}
        taskId={taskGroupId}
        productType="map"
        priceText={quote?.formatted ?? simpleFallbackPrice}
        modelPending={!downloadUrl}
        summary={{
          city: selectedCityKey,
          district: MAP_TEMPLATES.find((t) => t.id === activeTemplate)?.district,
          size: SIMPLE_SIZES.find((z) => Math.abs(modelSizeMm - z.mm) < 1)?.cm,
        }}
      />

      {/* Мобільний sticky-бар: ціна завжди на екрані + головна дія стану.
          Лише з ОДНІЄЇ копії панелі (showStickyBar) — інакше дубль порталів. */}
      {showStickyBar && (
        <>
          <div className="h-20 lg:hidden" aria-hidden="true" />
          <StickyActionBar
            // Ціну показуємо ЛИШЕ на кроці оформлення (OrderDialog), не під час
            // створення — ліворуч тихий продукт-лейбл, price=null.
            priceLabel={tOrder("prodMap")}
            price={null}
            actionLabel={
              downloadUrl
                ? t("orderPrint")
                : isGenerating
                  ? `${t("generating")} ${progress}%`
                  : t("generate")
            }
            busy={isGenerating}
            disabled={!downloadUrl && (!selectedArea || isGenerating)}
            onAction={() => { if (downloadUrl) setOrderOpen(true); else handleGenerate(); }}
            // Друга дія залежить від стану:
            //  • готово → «Завантажити» поряд із «Замовити»;
            //  • до/під час генерації → «Замовити» (order-now: фонова генерація
            //    + форма одразу, без очікування 1-3 хв).
            secondaryLabel={downloadUrl ? t("downloadShort") : (selectedArea ? t("orderPrint") : undefined)}
            onSecondary={downloadUrl ? doGatedDownload : (selectedArea ? orderNow : undefined)}
          />
        </>
      )}
    </div>
  );
}
