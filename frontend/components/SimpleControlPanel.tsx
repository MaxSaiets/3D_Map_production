"use client";

import { useCallback, useEffect, useState } from "react";
import { Loader2, Play, Download, MapPin, Check, Sparkles, ShoppingBag } from "lucide-react";
import { useTranslations } from "next-intl";
import { useGenerationStore } from "@/store/generation-store";
import { MAP_TEMPLATES, MAP_STYLE_PRESETS } from "@/lib/templates";
import { buildMapRequest, SIMPLE_SIZES } from "@/lib/generation";
import { OrderDialog } from "@/components/OrderDialog";
import { useAuth } from "@/components/AuthProvider";
import { gatedDownload } from "@/lib/download";

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
}: {
  availableCities?: Record<string, { center: [number, number]; bounds: any }>;
  selectedCityKey?: string;
  onCityChange?: (key: string) => void;
  onAdvanced?: () => void;
}) {
  const t = useTranslations("simple");
  const s = useGenerationStore();
  const {
    selectedArea, setSelectedArea,
    isGenerating, downloadUrl, progress, status, printQuality,
    taskGroupId, setTaskGroup, setActiveTaskId, setGenerating,
    setDownloadUrl, setTaskStatuses, updateProgress,
    modelSizeMm, setModelSizeMm, previewMode, setPreviewMode,
    setTerrainEnabled,
    setPreviewIncludeBuildings, setPreviewIncludeRoads,
    setPreviewIncludeWater, setPreviewIncludeParks,
  } = s;

  const [styleId, setStyleId] = useState<string>("full");
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

  const doGatedDownload = async () => {
    setDlBusy(true);
    const res = await gatedDownload({
      taskId: taskGroupId, downloadUrl,
      meta: { city: selectedCityKey, product_type: "map" },
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
  useEffect(() => {
    if (!taskGroupId) return;
    let stop = false;
    const iv = setInterval(async () => {
      try {
        const { api } = await import("@/lib/api");
        const r: any = await api.getStatus(taskGroupId);
        if (stop) return;
        setTaskStatuses({ [r.task_id]: r });
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
      } catch {/* ignore */}
    }, 2500);
    return () => { stop = true; clearInterval(iv); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskGroupId]);

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
        waterDepth: s.waterDepth, terrainEnabled: layerTerrain, terrainZScale: s.terrainZScale,
        terrainBaseThicknessMm: s.terrainBaseThicknessMm, terrainResolution: s.terrainResolution,
        terrariumZoom: s.terrariumZoom, exportFormat: s.exportFormat, modelSizeMm: s.modelSizeMm,
        isAmsMode: s.isAmsMode, flatPlateMode: s.flatPlateMode, previewMode: s.previewMode,
        previewIncludeBase: s.previewIncludeBase, previewIncludeRoads: layerRoads,
        previewIncludeBuildings: layerBuildings, previewIncludeWater: layerWater,
        previewIncludeParks: layerParks,
        zonePolygonCoords: s.zonePolygonCoords,
      });
      const { api } = await import("@/lib/api");
      const r = await api.generateModel(req as any);
      setTaskGroup(r.task_id, [r.task_id]);
      setActiveTaskId(r.task_id);
    } catch (e: any) {
      setError(e?.message || t("errGen"));
      setGenerating(false);
    }
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
                    <span className="h-12 w-12 shrink-0 overflow-hidden rounded-[14px] bg-[rgba(46,74,58,0.08)]">
                      {/* real preview if available; falls back to brand block */}
                      <img
                        src={`/templates/${t.id}.webp`}
                        alt={t.district}
                        className="h-full w-full object-cover"
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
            {isGenerating ? (<><Loader2 className="h-4 w-4 animate-spin" /> {t("generating")} {progress}%</>) : (<><Play className="h-4 w-4" /> {t("generate")}</>)}
          </button>

          {error && (
            <div className="rounded-[16px] border border-red-200 bg-red-50 px-4 py-2.5 text-xs text-red-700">{error}</div>
          )}

          {downloadUrl && printQuality && printQuality.status !== "ok" && (printQuality.warnings?.length ?? 0) > 0 && (
            <div className="rounded-[16px] border border-amber-200 bg-amber-50 px-4 py-2.5 text-xs text-amber-900">
              {t("qualityWarn")}
            </div>
          )}

          {downloadUrl && (
            <button
              type="button"
              onClick={doGatedDownload}
              disabled={dlBusy}
              className="inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-full border border-[var(--surface-border)] bg-white px-5 py-3 text-sm font-semibold text-[var(--text-primary)] transition hover:bg-white/70 disabled:opacity-60"
            >
              {dlBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />} {t("downloadFile")}
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

          {downloadUrl && (
            <button
              type="button"
              onClick={() => setOrderOpen(true)}
              className="inline-flex min-h-13 w-full items-center justify-center gap-2 rounded-full bg-[var(--bronze,#8E6B3D)] px-5 py-3.5 text-sm font-bold text-white shadow-[0_16px_32px_rgba(142,107,61,0.28)] transition hover:opacity-90"
            >
              <ShoppingBag className="h-4 w-4" /> {t("orderPrint")}
            </button>
          )}
        </div>
      </div>

      <OrderDialog
        open={orderOpen}
        onClose={() => setOrderOpen(false)}
        taskId={taskGroupId}
        productType="map"
        summary={{
          city: selectedCityKey,
          district: MAP_TEMPLATES.find((t) => t.id === activeTemplate)?.district,
          size: SIMPLE_SIZES.find((z) => Math.abs(modelSizeMm - z.mm) < 1)?.cm,
        }}
      />
    </div>
  );
}
