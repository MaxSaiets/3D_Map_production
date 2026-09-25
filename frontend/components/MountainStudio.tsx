"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { useLocale, useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import { Mountain, Map as MapIcon, Globe2, Send, Instagram, Share2, Download, Image as ImageIcon, Plus, X, Loader2, MapPin } from "lucide-react";
import { api, type MountainSpec, type MountainPreset, type MountainFigure, type MountainPreview, type MountainResultSpec, type MountainFigureSpec } from "@/lib/api";
import { AgentBox } from "@/components/AgentBox";

const Model3DViewer = dynamic(() => import("@/components/Model3DViewer"), { ssr: false });
const MountainMapPicker = dynamic(() => import("@/components/MountainMapPicker"), { ssr: false, loading: () => <div className="h-[360px] animate-pulse rounded-2xl bg-[rgba(15,23,42,0.05)]" /> });

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";
const SIZES = [120, 150, 200, 250, 300, 400];
const AREAS = [2, 3, 4, 6, 8, 12, 20, 30];
type Tab = "presets" | "map";

const DEFAULT_SPEC: MountainSpec = {
  place: null, area_km: null, size_mm: 200, height_mm: null,
  frame: { style: "rounded", width_mm: 10, height_mm: 25 }, sides: "slope", base_mm: 3, figures: [], texture: "satellite", bed_mm: 256,
};

const abs = (u: string | null | undefined) => (u ? (u.startsWith("http") ? u : `${API_BASE}${u}`) : null);
// 1:43 215 — з пробілом-роздільником незалежно від локалі браузера (toLocaleString давав «43,215» у тестах)
const fmt = (n: number) => String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, " ");

export default function MountainStudio() {
  const t = useTranslations("mountains");
  const locale = useLocale();
  const [tab, setTab] = useState<Tab>("presets");
  const [spec, setSpec] = useState<MountainSpec>(DEFAULT_SPEC);
  const [presets, setPresets] = useState<MountainPreset[]>([]);
  const [figures, setFigures] = useState<MountainFigure[]>([]);
  const [preview, setPreview] = useState<MountainPreview | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewErr, setPreviewErr] = useState<string | null>(null);
  const [pickingFig, setPickingFig] = useState<number | null>(null);
  const [customSize, setCustomSize] = useState("");

  const [taskId, setTaskId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ glb: string; print: string | null; spec: MountainResultSpec | null } | null>(null);
  const [copied, setCopied] = useState(false);
  const [shared, setShared] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.mountainsPresets(locale).then((r) => setPresets(r.presets)).catch(() => {});
    api.mountainsFigures(locale).then((r) => setFigures(r.figures)).catch(() => {});
  }, [locale]);

  // Deep-link ?peak=<presetId> (сторінки /gory/[slug]): вершина обрана одразу.
  const peakAppliedRef = useRef(false);
  useEffect(() => {
    if (peakAppliedRef.current || !presets.length) return;
    peakAppliedRef.current = true;
    try {
      const id = new URLSearchParams(window.location.search).get("peak");
      const p = id ? presets.find((x) => x.id === id) : undefined;
      if (p) setSpec((s) => ({ ...s, place: { name: p.name, lat: p.lat, lon: p.lon, source: "preset", preset_id: p.id, area_km: p.area_km, elev: p.elev }, area_km: p.area_km }));
    } catch { /* ignore */ }
  }, [presets]);

  const areaKm = spec.area_km ?? spec.place?.area_km ?? 4;

  // миттєве превʼю: hillshade + знімок, коли міняється місце/ділянка/розмір (з дебаунсом)
  useEffect(() => {
    if (!spec.place) { setPreview(null); return; }
    const h = setTimeout(async () => {
      setPreviewBusy(true); setPreviewErr(null);
      try {
        setPreview(await api.mountainsPreview({ lat: spec.place!.lat, lon: spec.place!.lon, area_km: areaKm, size_mm: spec.size_mm, height_mm: spec.height_mm ?? undefined, satellite: spec.texture !== "none" }));
      } catch (e: any) { setPreviewErr(e?.response?.data?.detail || t("previewFailed")); }
      finally { setPreviewBusy(false); }
    }, 500);
    return () => clearTimeout(h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec.place?.lat, spec.place?.lon, areaKm, spec.size_mm, spec.height_mm, spec.texture]);

  const pickPreset = (p: MountainPreset) => {
    setSpec((s) => ({ ...s, place: { name: p.name, lat: p.lat, lon: p.lon, source: "preset", preset_id: p.id, area_km: p.area_km, elev: p.elev }, area_km: p.area_km }));
  };
  const pickOnMap = useCallback((lat: number, lon: number) => {
    setSpec((s) => ({ ...s, place: { name: `${lat.toFixed(4)}, ${lon.toFixed(4)}`, lat, lon, source: "map" }, area_km: s.area_km ?? 4 }));
  }, []);

  const setFrame = (patch: Partial<MountainSpec["frame"]>) => setSpec((s) => ({ ...s, frame: { ...s.frame, ...patch } }));
  const addFigure = (f: MountainFigure) => setSpec((s) => ({
    ...s, figures: [...s.figures, { id: f.id, where: (f.kind === "climbing" ? "steepest" : f.kind === "building" ? "slope" : "summit") as MountainFigureSpec["where"], height_mm: null }].slice(0, 6),
  }));
  const updFigure = (i: number, patch: Partial<MountainFigureSpec>) => setSpec((s) => ({ ...s, figures: s.figures.map((f, k) => (k === i ? { ...f, ...patch } : f)) }));
  const delFigure = (i: number) => setSpec((s) => ({ ...s, figures: s.figures.filter((_, k) => k !== i) }));

  // клік по превʼю → точка для фігурки (превʼю «північ угорі»: fy = 1 − y)
  const onPreviewClick = (e: React.MouseEvent<HTMLImageElement>) => {
    if (pickingFig == null) return;
    const r = e.currentTarget.getBoundingClientRect();
    const fx = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); const fy = 1 - Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
    updFigure(pickingFig, { where: "point", fx: +fx.toFixed(3), fy: +fy.toFixed(3) }); setPickingFig(null);
  };

  const stop = () => { if (timer.current) { clearInterval(timer.current); timer.current = null; } };
  useEffect(() => () => stop(), []);

  const generate = async () => {
    if (!spec.place || busy) return;
    setError(null); setResult(null); setBusy(true); setProgress(3); setStatusMsg(t("starting"));
    try {
      const r = await api.mountainsGenerate(spec);
      setTaskId(r.task_id);
      import("@/lib/analytics").then((m) => m.track("mountain_generate", { place: spec.place?.name, size: spec.size_mm, figures: spec.figures.length })).catch(() => {});
    } catch (e: any) { setError(e?.response?.data?.detail || e?.message || t("genFailed")); setBusy(false); }
  };

  useEffect(() => {
    if (!taskId) return;
    stop();
    timer.current = setInterval(async () => {
      try {
        const s: any = await api.getStatus(taskId);
        setProgress(Number(s.progress) || 0); setStatusMsg(s.message || "");
        if (s.status === "completed") {
          stop(); setBusy(false);
          // /files/*.3mf на проді закритий SafeStatic (друк-файли не віддаються за іменем, task #70) —
          // друк-файл беремо через /api/download/<task>?format=3mf (стрімить локальний файл задачі).
          setResult({ glb: abs(s.download_url_glb) || "", print: `${API_BASE}/api/download/${taskId}?format=3mf`, spec: (s.world_spec as MountainResultSpec) || null });
          setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
        } else if (s.status === "failed" || s.status === "error") { stop(); setBusy(false); setError(s.message || t("genFailed")); }
      } catch { /* повторимо наступним тіком */ }
    }, 1500);
    return () => stop();
  }, [taskId]); // eslint-disable-line react-hooks/exhaustive-deps

  const shareUrl = () => (taskId && typeof window !== "undefined" ? `${window.location.origin}/share/${taskId}` : "");
  const openChat = async (channel: "tg" | "ig") => {
    import("@/lib/analytics").then((m) => m.track("messenger_order", { channel, product: "mountain" })).catch(() => {});
    const text = t("msgPrefill", { place: spec.place?.name || "", size: spec.size_mm, link: shareUrl() });
    try { await navigator.clipboard.writeText(text); setCopied(true); } catch { setCopied(false); }
    window.open(channel === "tg" ? "https://t.me/monadruk" : "https://ig.me/m/monadruk", "_blank", "noopener");
  };
  const doShare = async () => {
    const url = shareUrl(); if (!url) return;
    try { if (typeof navigator.share === "function") await navigator.share({ url, title: "Monadruk" }); else await navigator.clipboard.writeText(url); setShared(true); } catch { /* скасовано */ }
  };

  const figName = (id: string) => figures.find((f) => f.id === id)?.name || id;
  const figMeta = (id: string) => figures.find((f) => f.id === id);
  const chip = (active: boolean) => `min-h-10 rounded-full border px-3.5 py-2 text-[12.5px] font-semibold transition ${active ? "border-[var(--accent-strong)] bg-[rgba(15,118,110,0.1)] text-[var(--accent-strong)]" : "border-[var(--surface-border)] bg-white text-[var(--text-secondary)] hover:border-[rgba(11,92,87,0.3)]"}`;

  const agentExamples = useMemo(() => [t("agentEx1"), t("agentEx2"), t("agentEx3")], [t]);

  return (
    <div id="main-content" tabIndex={-1} className="mx-auto max-w-[1180px] px-4 pb-8 pt-6 sm:pb-12">
      {/* Заголовок (badge + H1 + підзаголовок) рендериться на сервері в mountains/layout.tsx —
          ця студія ssr:false, і H1 звідси Google у HTML не бачив (аудит 24.09.2026). */}

      {/* Вкладки режимів */}
      <div role="tablist" aria-label={t("tabsAria")} className="mx-auto mb-6 flex w-fit flex-wrap justify-center gap-1 rounded-full border border-[var(--surface-border)] bg-[var(--surface-panel)] p-1">
        {([["presets", Mountain, t("tabPresets")], ["map", MapIcon, t("tabMap")]] as const).map(([k, Icon, label]) => (
          <button key={k} role="tab" aria-selected={tab === k} data-testid={`mnt-tab-${k}`} onClick={() => setTab(k)}
            className={`inline-flex min-h-10 items-center gap-1.5 rounded-full px-4 text-[13px] font-semibold transition ${tab === k ? "bg-[var(--accent-strong)] text-white" : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"}`}>
            <Icon size={15} /> {label}
          </button>
        ))}
        <Link href="/worlds" role="tab" aria-selected={false} className="inline-flex min-h-10 items-center gap-1.5 rounded-full px-4 text-[13px] font-semibold text-[var(--text-secondary)] transition hover:text-[var(--text-primary)]">
          <Globe2 size={15} /> {t("tabWorlds")}
        </Link>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr),minmax(0,1.05fr)]">
        {/* ЛІВА: місце + агент + параметри */}
        <section className="rounded-[28px] border border-[var(--surface-border)] bg-[var(--surface-panel)] p-5 shadow-[0_18px_60px_rgba(15,23,42,0.07)]">
          <AgentBox<MountainSpec>
            testId="mnt-agent"
            placeholder={t("agentPlaceholder")}
            examples={agentExamples}
            ask={(text) => api.mountainsAgent(text, locale, spec)}
            onApply={(s) => { setSpec({ ...DEFAULT_SPEC, ...s }); if (s.place && s.place.source === "map") setTab("map"); }}
          />

          {/* Крок 1: місце */}
          <h2 className="mt-5 text-sm font-semibold text-[var(--text-primary)]">{t("step1")}</h2>
          {tab === "presets" ? (
            <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-3" data-testid="mnt-presets">
              {presets.map((p) => {
                const active = spec.place?.preset_id === p.id;
                return (
                  <button key={p.id} type="button" onClick={() => pickPreset(p)} data-testid={`mnt-preset-${p.id}`} aria-pressed={active}
                    className={`group overflow-hidden rounded-2xl border text-left transition ${active ? "border-[var(--accent-strong)] ring-2 ring-[rgba(15,118,110,0.25)]" : "border-[var(--surface-border)] hover:border-[rgba(11,92,87,0.4)]"}`}>
                    <div className="relative aspect-[4/3] bg-[rgba(15,23,42,0.06)]">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={p.photo} alt={p.name} loading="lazy" className="h-full w-full object-cover" onError={(e) => { (e.currentTarget as HTMLImageElement).style.visibility = "hidden"; }} />
                      <span className="absolute bottom-1 right-1.5 rounded-full bg-black/55 px-1.5 py-0.5 text-[10px] font-semibold text-white">{p.elev} м</span>
                    </div>
                    <div className="px-2 py-1.5">
                      <div className="truncate text-[12.5px] font-semibold text-[var(--text-primary)]">{p.name}</div>
                      <div className="truncate text-[11px] text-[var(--text-secondary)]">{p.country}</div>
                    </div>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="mt-2">
              <p className="mb-2 text-[12px] text-[var(--text-secondary)]">{t("mapHint")}</p>
              <MountainMapPicker lat={spec.place?.lat ?? null} lon={spec.place?.lon ?? null} areaKm={areaKm} onPick={pickOnMap} height={340} />
            </div>
          )}
          {spec.place && (
            <p className="mt-2 inline-flex items-center gap-1 text-[12.5px] text-[var(--text-primary)]" data-testid="mnt-place">
              <MapPin size={13} className="text-[var(--accent-strong)]" /> <b>{spec.place.name}</b>
              <span className="text-[var(--text-secondary)]">· {spec.place.lat.toFixed(4)}, {spec.place.lon.toFixed(4)}</span>
            </p>
          )}

          {/* Ділянка */}
          <div className="mt-4">
            <div className="text-sm font-semibold text-[var(--text-primary)]">{t("areaLabel")} <span className="font-normal text-[var(--text-secondary)]">· {areaKm} × {areaKm} км</span></div>
            <div className="mt-2 flex flex-wrap gap-1.5" role="radiogroup">
              {AREAS.map((a) => <button key={a} type="button" role="radio" aria-checked={areaKm === a} onClick={() => setSpec((s) => ({ ...s, area_km: a }))} className={chip(areaKm === a)}>{a} км</button>)}
            </div>
            <p className="mt-1 text-[11px] text-[var(--text-secondary)]">{t("areaHint")}</p>
          </div>

          {/* Розмір / висота */}
          <div className="mt-4">
            <div className="text-sm font-semibold text-[var(--text-primary)]">{t("sizeLabel")}</div>
            <div className="mt-2 flex flex-wrap gap-1.5" role="radiogroup">
              {SIZES.map((mm) => <button key={mm} type="button" role="radio" aria-checked={spec.size_mm === mm} data-testid={`mnt-size-${mm}`} onClick={() => setSpec((s) => ({ ...s, size_mm: mm }))} className={chip(spec.size_mm === mm)}>{mm} мм</button>)}
              <input value={customSize} onChange={(e) => { setCustomSize(e.target.value); const v = Number(e.target.value); if (v >= 60 && v <= 400) setSpec((s) => ({ ...s, size_mm: v })); }} inputMode="numeric" placeholder={t("sizeCustom")}
                className="min-h-10 w-24 rounded-full border border-[var(--surface-border)] bg-white px-3 text-[12.5px] outline-none focus:border-[var(--accent-strong)]" />
            </div>
            <p className="mt-1 text-[11px] text-[var(--text-secondary)]">{spec.size_mm > (spec.bed_mm ?? 256) ? t("sizeTiles", { bed: spec.bed_mm ?? 256 }) : t("sizeOne")}</p>
          </div>
          <div className="mt-4">
            <div className="text-sm font-semibold text-[var(--text-primary)]">{t("heightLabel")}</div>
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <button type="button" onClick={() => setSpec((s) => ({ ...s, height_mm: null }))} className={chip(spec.height_mm == null)}>{t("heightAuto")}</button>
              {[50, 80, 120, 180, 250].map((h) => <button key={h} type="button" onClick={() => setSpec((s) => ({ ...s, height_mm: h }))} className={chip(spec.height_mm === h)}>{h} мм</button>)}
            </div>
            {preview && (
              <p className="mt-1 text-[11px] text-[var(--text-secondary)]">
                {t("heightInfo", { relief: preview.relief_m, natural: preview.relief_mm_natural, scale: fmt(preview.scale) })}
                {spec.height_mm && preview.zexag_for_height ? ` · ${t("zexag", { k: preview.zexag_for_height })}` : ""}
              </p>
            )}
          </div>

          {/* Ободок */}
          <div className="mt-4">
            <div className="text-sm font-semibold text-[var(--text-primary)]">{t("frameLabel")}</div>
            <div className="mt-2 flex flex-wrap gap-1.5" role="radiogroup">
              {(["none", "flat", "rounded"] as const).map((k) => <button key={k} type="button" role="radio" aria-checked={spec.frame.style === k} data-testid={`mnt-frame-${k}`} onClick={() => setFrame({ style: k, width_mm: k === "none" ? 0 : (spec.frame.width_mm || 10) })} className={chip(spec.frame.style === k)}>{t(`frame_${k}`)}</button>)}
            </div>
            {spec.frame.style !== "none" && (
              <div className="mt-2 grid grid-cols-2 gap-2">
                <label className="text-[12px] text-[var(--text-secondary)]">{t("frameWidth")}
                  <input type="range" min={5} max={30} step={1} value={spec.frame.width_mm} onChange={(e) => setFrame({ width_mm: Number(e.target.value) })} className="mt-1 w-full accent-[var(--accent-strong)]" />
                  <span className="text-[var(--text-primary)]">{spec.frame.width_mm} мм</span>
                </label>
                <label className="text-[12px] text-[var(--text-secondary)]">{t("frameHeight")}
                  <input type="range" min={3} max={60} step={1} value={spec.frame.height_mm} onChange={(e) => setFrame({ height_mm: Number(e.target.value) })} className="mt-1 w-full accent-[var(--accent-strong)]" />
                  <span className="text-[var(--text-primary)]">{spec.frame.height_mm} мм</span>
                </label>
              </div>
            )}
          </div>

          {/* Боки */}
          <div className="mt-4">
            <div className="text-sm font-semibold text-[var(--text-primary)]">{t("sidesLabel")}</div>
            <div className="mt-2 flex flex-wrap gap-1.5" role="radiogroup">
              {(["slope", "rock", "vertical"] as const).map((k) => <button key={k} type="button" role="radio" aria-checked={spec.sides === k} data-testid={`mnt-sides-${k}`} onClick={() => setSpec((s) => ({ ...s, sides: k }))} className={chip(spec.sides === k)}>{t(`sides_${k}`)}</button>)}
            </div>
            <p className="mt-1 text-[11px] text-[var(--text-secondary)]">{t(`sidesHint_${spec.sides}`)}</p>
          </div>

          {/* Фігурки */}
          <div className="mt-4">
            <div className="text-sm font-semibold text-[var(--text-primary)]">{t("figuresLabel")}</div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {figures.map((f) => (
                <button key={f.id} type="button" onClick={() => addFigure(f)} disabled={spec.figures.length >= 6} data-testid={`mnt-fig-add-${f.id}`}
                  className="inline-flex min-h-10 items-center gap-1.5 rounded-full border border-[var(--surface-border)] bg-white px-3 text-[12.5px] font-semibold text-[var(--text-primary)] transition hover:border-[var(--accent-strong)] disabled:opacity-50">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={f.thumb} alt="" className="h-6 w-6 rounded-full object-cover" onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }} />
                  <Plus size={13} /> {f.name}
                </button>
              ))}
            </div>
            {spec.figures.length > 0 && (
              <ul className="mt-2 space-y-1.5" data-testid="mnt-fig-list">
                {spec.figures.map((f, i) => {
                  const m = figMeta(f.id);
                  return (
                    <li key={i} className="flex flex-wrap items-center gap-2 rounded-xl border border-[var(--surface-border)] bg-white px-2.5 py-1.5 text-[12px]">
                      <b className="text-[var(--text-primary)]">{figName(f.id)}</b>
                      <select value={f.where} onChange={(e) => updFigure(i, { where: e.target.value as MountainFigureSpec["where"] })} className="rounded-lg border border-[var(--surface-border)] bg-white px-1.5 py-1">
                        {(["summit", "steepest", "slope", "point"] as const).map((w) => <option key={w} value={w}>{t(`where_${w}`)}</option>)}
                      </select>
                      {f.where === "point" && (
                        <button type="button" onClick={() => setPickingFig(i)} className={`rounded-lg border px-2 py-1 ${pickingFig === i ? "border-[var(--accent-strong)] text-[var(--accent-strong)]" : "border-[var(--surface-border)]"}`}>
                          {f.fx != null ? `${Math.round(f.fx * 100)} %, ${Math.round((f.fy ?? 0) * 100)} %` : t("pickPoint")}
                        </button>
                      )}
                      <label className="inline-flex items-center gap-1 text-[var(--text-secondary)]">{t("figHeight")}
                        <input type="number" min={m?.min_height_mm ?? 6} max={m?.max_height_mm ?? 50} value={f.height_mm ?? ""} placeholder={String(Math.round((m?.default_height_mm ?? 15) * Math.min(2, Math.max(0.5, spec.size_mm / 200))))}
                          onChange={(e) => updFigure(i, { height_mm: e.target.value ? Number(e.target.value) : null })} className="w-14 rounded-lg border border-[var(--surface-border)] px-1.5 py-1" /> мм
                      </label>
                      <button type="button" aria-label={t("remove")} onClick={() => delFigure(i)} className="ml-auto rounded-full p-1 text-[var(--text-secondary)] hover:text-red-600"><X size={14} /></button>
                    </li>
                  );
                })}
              </ul>
            )}
            <p className="mt-1 text-[11px] text-[var(--text-secondary)]">{t("figuresHint")}</p>
          </div>

          <label className="mt-4 flex items-center gap-2 text-[12.5px] text-[var(--text-primary)]">
            <input type="checkbox" checked={spec.texture !== "none"} onChange={(e) => setSpec((s) => ({ ...s, texture: e.target.checked ? "satellite" : "none" }))} className="accent-[var(--accent-strong)]" /> {t("textureLabel")}
          </label>

          <button type="button" onClick={generate} disabled={busy || !spec.place} data-testid="mnt-generate"
            className="mt-5 w-full rounded-full bg-[var(--accent-strong)] px-5 py-3 text-sm font-semibold text-white transition disabled:opacity-50">
            {busy ? `${progress}% · ${statusMsg || t("generating")}` : t("generateButton")}
          </button>
          {!spec.place && <p className="mt-1.5 text-center text-[11.5px] text-[var(--text-secondary)]">{t("needPlace")}</p>}
          {error && <p role="alert" data-testid="mnt-error" className="mt-3 text-sm text-red-600">{error}</p>}
          <p className="mt-3 text-[11px] leading-4 text-[var(--text-secondary)]">{t("hint")}</p>
        </section>

        {/* ПРАВА: превʼю / результат */}
        <section className="flex min-h-[460px] flex-col gap-3 rounded-[28px] border border-[var(--surface-border)] bg-[var(--surface-panel)] p-3 shadow-[0_18px_60px_rgba(15,23,42,0.07)]">
          {!result && (
            <div className="flex flex-1 flex-col">
              <div className="mb-1.5 flex items-center justify-between px-1">
                <span className="inline-flex items-center gap-1.5 text-[13px] font-semibold text-[var(--text-primary)]"><ImageIcon size={14} /> {t("previewTitle")}</span>
                {previewBusy && <Loader2 size={14} className="animate-spin text-[var(--text-secondary)]" />}
              </div>
              {preview ? (
                <div className="relative overflow-hidden rounded-2xl bg-[rgba(15,23,42,0.03)]">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={preview.png} alt={t("previewAlt")} data-testid="mnt-preview" onClick={onPreviewClick}
                    className={`aspect-square w-full object-cover ${pickingFig != null ? "cursor-crosshair ring-4 ring-[var(--accent-strong)]" : ""}`} />
                  {spec.figures.filter((f) => f.where === "point" && f.fx != null).map((f, i) => (
                    <span key={i} className="pointer-events-none absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-[var(--accent-strong)]" style={{ left: `${(f.fx ?? 0) * 100}%`, top: `${(1 - (f.fy ?? 0)) * 100}%` }} />
                  ))}
                  {pickingFig != null && <div className="absolute inset-x-0 top-0 bg-[var(--accent-strong)]/90 px-3 py-1.5 text-center text-[12px] font-semibold text-white">{t("pickPointHint")}</div>}
                  <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/60 to-transparent px-3 pb-2 pt-6 text-[11.5px] text-white">
                    {t("previewStats", { min: preview.elev_min, max: preview.elev_max, scale: fmt(preview.scale) })} · {preview.sources.join(" + ")}
                  </div>
                </div>
              ) : (
                <div className="flex flex-1 items-center justify-center rounded-2xl bg-[rgba(15,23,42,0.03)] text-center text-[var(--text-secondary)]">
                  <div><div className="text-4xl">🏔️</div><p className="mt-3 max-w-xs text-sm">{previewErr || (busy ? statusMsg || t("generating") : t("previewEmpty"))}</p></div>
                </div>
              )}
              {busy && (
                <div className="mt-3 h-2 overflow-hidden rounded-full bg-[rgba(15,23,42,0.08)]"><div className="h-full bg-[var(--accent-strong)] transition-all" style={{ width: `${progress}%` }} /></div>
              )}
            </div>
          )}

          {result && (
            <div ref={resultRef} className="flex flex-1 flex-col" data-testid="mnt-result">
              <div className="flex-1 overflow-hidden rounded-2xl bg-[rgba(15,23,42,0.03)]">
                {result.glb && <Model3DViewer url={result.glb} height={440} flat allowZoom autoRotate label={t("title")} />}
              </div>
              {result.spec && (
                <p className="mt-2 text-center text-[12.5px] text-[var(--text-secondary)]" data-testid="mnt-built">
                  {t("builtAs", { place: result.spec.place, scale: fmt(result.spec.scale), height: Math.round(result.spec.height_mm), zexag: result.spec.zexag })}
                  {result.spec.tiles.length > 0 ? ` · ${t("tilesCount", { n: result.spec.tiles.length })}` : ""} · {result.spec.sources.join(" + ")}
                </p>
              )}
              <div className="mt-2 grid gap-2 sm:grid-cols-2">
                {result.print && <a href={result.print} download className="inline-flex min-h-11 items-center justify-center gap-1.5 rounded-full bg-[var(--accent-strong)] px-4 text-[13px] font-semibold text-white transition hover:brightness-110"><Download size={14} /> {t("downloadPrint")}</a>}
                {result.spec?.tiles_zip && <a href={abs(result.spec.tiles_zip)!} download className="inline-flex min-h-11 items-center justify-center gap-1.5 rounded-full border border-[var(--accent-strong)] px-4 text-[13px] font-semibold text-[var(--accent-strong)]"><Download size={14} /> {t("downloadTiles")}</a>}
                {result.spec?.preview_png && <a href={abs(result.spec.preview_png)!} target="_blank" rel="noopener" className="inline-flex min-h-11 items-center justify-center gap-1.5 rounded-full border border-[var(--surface-border)] bg-white px-4 text-[13px] font-semibold text-[var(--text-primary)]"><ImageIcon size={14} /> {t("photoTop")}</a>}
                {result.spec?.paint_jpg && <a href={abs(result.spec.paint_jpg)!} target="_blank" rel="noopener" className="inline-flex min-h-11 items-center justify-center gap-1.5 rounded-full border border-[var(--surface-border)] bg-white px-4 text-[13px] font-semibold text-[var(--text-primary)]"><ImageIcon size={14} /> {t("paintGuide")}</a>}
                <a href={result.glb} download className="inline-flex min-h-11 items-center justify-center gap-1.5 rounded-full border border-[var(--surface-border)] bg-white px-4 text-[13px] font-semibold text-[var(--text-primary)]"><Download size={14} /> {t("downloadGlb")}</a>
                <button type="button" onClick={() => setResult(null)} className="inline-flex min-h-11 items-center justify-center rounded-full border border-[var(--surface-border)] bg-white px-4 text-[13px] font-semibold text-[var(--text-primary)]">{t("editAgain")}</button>
              </div>
              <div className="mt-3 rounded-2xl border border-[var(--surface-border)] bg-white/70 p-3" data-testid="mnt-order">
                <p className="text-[13.5px] font-semibold text-[var(--text-primary)]">{t("orderTitle")}</p>
                <p className="mt-1 text-[12px] leading-snug text-[var(--text-secondary)]">{t("orderSub")}</p>
                <div className="mt-2.5 grid grid-cols-2 gap-2">
                  <button type="button" onClick={() => openChat("tg")} className="inline-flex min-h-11 items-center justify-center gap-1.5 rounded-full border border-[var(--surface-border)] bg-white px-3 text-[12.5px] font-semibold text-[var(--text-primary)] transition hover:border-[var(--accent-strong)]"><Send size={14} className="text-[#2AABEE]" /> Telegram</button>
                  <button type="button" onClick={() => openChat("ig")} className="inline-flex min-h-11 items-center justify-center gap-1.5 rounded-full border border-[var(--surface-border)] bg-white px-3 text-[12.5px] font-semibold text-[var(--text-primary)] transition hover:border-[var(--accent-strong)]"><Instagram size={14} className="text-[#E1306C]" /> Instagram</button>
                </div>
                <button type="button" onClick={doShare} className="mx-auto mt-2 flex min-h-10 items-center gap-1.5 text-[12px] font-semibold text-[var(--accent-strong)] underline underline-offset-2"><Share2 size={13} /> {shared ? t("shareCopied") : t("shareLink")}</button>
                <p className="mt-1 text-center text-[11.5px] text-[var(--text-secondary)]" aria-live="polite">{copied ? t("msgCopied") : ""}</p>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
