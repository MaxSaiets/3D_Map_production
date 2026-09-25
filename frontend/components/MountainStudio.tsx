"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { useLocale, useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import {
  Send, Instagram, Share2, Download, Image as ImageIcon, Plus, X, Loader2, MapPin, Search, Sparkles,
  ChevronDown, Map as MapIcon, SlidersHorizontal, Check, Globe2,
} from "lucide-react";
import {
  api, type MountainSpec, type MountainPreset, type MountainFigure, type MountainPreview, type MountainResultSpec,
  type MountainFigureSpec, type MountainSearchHit,
} from "@/lib/api";
import { AgentBox } from "@/components/AgentBox";

const Model3DViewer = dynamic(() => import("@/components/Model3DViewer"), { ssr: false });
const MountainMapPicker = dynamic(() => import("@/components/MountainMapPicker"), { ssr: false, loading: () => <div className="h-[300px] animate-pulse rounded-2xl bg-[rgba(15,23,42,0.05)]" /> });

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";
const TG_URL = "https://t.me/monadruk";
/** Розміри плити в см: людям звичніші сантиметри (як у /create), бекенд отримує мм. */
const SIZES_CM = [15, 20, 25, 30, 40];
const AREAS = [2, 3, 4, 6, 8, 12, 20, 30];
const POPULAR = 6;
const DRAFT_KEY = "mnt_draft_v1";

/** Три готові «вигляди» замість трьох окремих груп контролів (ободок × боки × розміри ободка).
 *  Точні значення лишаються в «Точних налаштуваннях» для тих, кому треба. */
const STYLES = {
  classic: { frame: { style: "rounded", width_mm: 10, height_mm: 25 }, sides: "slope" },
  rock: { frame: { style: "flat", width_mm: 10, height_mm: 25 }, sides: "rock" },
  bare: { frame: { style: "none", width_mm: 0, height_mm: 25 }, sides: "vertical" },
} as const;
type StyleId = keyof typeof STYLES;

const DEFAULT_SPEC: MountainSpec = {
  place: null, area_km: null, size_mm: 200, height_mm: null,
  frame: { style: "rounded", width_mm: 10, height_mm: 25 }, sides: "slope", base_mm: 3, figures: [], texture: "satellite", bed_mm: 256,
};

const abs = (u: string | null | undefined) => (u ? (u.startsWith("http") ? u : `${API_BASE}${u}`) : null);
// 1:43 215 — з пробілом-роздільником незалежно від локалі браузера (toLocaleString давав «43,215» у тестах)
const fmt = (n: number) => String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, " ");

function readDraft(): MountainSpec | null {
  try {
    const raw = window.localStorage.getItem(DRAFT_KEY);
    if (!raw) return null;
    const s = JSON.parse(raw) as MountainSpec;
    return s && typeof s.size_mm === "number" && s.frame ? { ...DEFAULT_SPEC, ...s } : null;
  } catch { return null; }
}

export default function MountainStudio() {
  const t = useTranslations("mountains");
  const locale = useLocale();
  const [spec, setSpec] = useState<MountainSpec>(DEFAULT_SPEC);
  const [presets, setPresets] = useState<MountainPreset[]>([]);
  const [figures, setFigures] = useState<MountainFigure[]>([]);
  const [preview, setPreview] = useState<MountainPreview | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewErr, setPreviewErr] = useState<string | null>(null);
  const [pickingFig, setPickingFig] = useState<number | null>(null);
  const [customCm, setCustomCm] = useState("");
  const [showAllPeaks, setShowAllPeaks] = useState(false);
  const [showMap, setShowMap] = useState(false);
  const [showAgent, setShowAgent] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // пошук гори за назвою
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<MountainSearchHit[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);

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
  const hydrated = useRef(false);

  useEffect(() => {
    api.mountainsPresets(locale).then((r) => setPresets(r.presets)).catch(() => {});
    api.mountainsFigures(locale).then((r) => setFigures(r.figures)).catch(() => {});
  }, [locale]);

  // Старт: deep-link ?peak=<presetId> (сторінки /gory/[slug], блог) → чернетка → Говерла за замовчуванням.
  // Порожній правий блок «оберіть вершину» був першим, що бачила людина; тепер одразу видно рельєф.
  useEffect(() => {
    if (hydrated.current || !presets.length) return;
    hydrated.current = true;
    let want: string | null = null;
    try { want = new URLSearchParams(window.location.search).get("peak"); } catch { /* ignore */ }
    const byUrl = want ? presets.find((p) => p.id === want) : undefined;
    const draft = byUrl ? null : readDraft();
    if (draft?.place) { setSpec(draft); if (draft.place.source !== "preset") setShowMap(true); return; }
    const p = byUrl || presets.find((x) => x.id === "hoverla") || presets[0];
    setSpec((s) => ({ ...s, place: { name: p.name, lat: p.lat, lon: p.lon, source: "preset", preset_id: p.id, area_km: p.area_km, elev: p.elev }, area_km: p.area_km }));
  }, [presets]);

  useEffect(() => {
    if (!hydrated.current) return;
    try { window.localStorage.setItem(DRAFT_KEY, JSON.stringify(spec)); } catch { /* приватний режим */ }
  }, [spec]);

  const areaKm = spec.area_km ?? spec.place?.area_km ?? 4;

  // миттєве превʼю: hillshade + знімок, коли міняється місце/ділянка/розмір (з дебаунсом)
  useEffect(() => {
    if (!spec.place) { setPreview(null); return; }
    const h = setTimeout(async () => {
      setPreviewBusy(true); setPreviewErr(null);
      try {
        setPreview(await api.mountainsPreview({ lat: spec.place!.lat, lon: spec.place!.lon, area_km: areaKm, size_mm: spec.size_mm, height_mm: spec.height_mm ?? undefined, satellite: spec.texture !== "none" }));
      } catch (e: any) { setPreviewErr(e?.response?.data?.detail || t("previewFailed")); setPreview(null); }
      finally { setPreviewBusy(false); }
    }, 450);
    return () => clearTimeout(h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec.place?.lat, spec.place?.lon, areaKm, spec.size_mm, spec.height_mm, spec.texture]);

  // пошук: пресети фільтруємо одразу, мережу питаємо з дебаунсом
  const localHits = useMemo<MountainSearchHit[]>(() => {
    const ql = q.trim().toLowerCase();
    if (ql.length < 2) return [];
    return presets.filter((p) => p.name.toLowerCase().includes(ql) || p.id.includes(ql))
      .map((p) => ({ name: p.name, lat: p.lat, lon: p.lon, source: "preset" as const, preset_id: p.id, area_km: p.area_km, elev: p.elev, display: p.country }));
  }, [q, presets]);
  useEffect(() => {
    const qq = q.trim();
    if (qq.length < 3) { setHits(null); setSearching(false); return; }
    setSearching(true);
    const h = setTimeout(async () => {
      try { setHits((await api.mountainsSearch(qq, locale)).results); } catch { setHits([]); }
      finally { setSearching(false); }
    }, 400);
    return () => clearTimeout(h);
  }, [q, locale]);
  const shownHits = hits ?? localHits;

  const resetResult = () => { setResult(null); setError(null); };

  const pickPreset = (p: MountainPreset) => {
    resetResult();
    setSpec((s) => ({ ...s, place: { name: p.name, lat: p.lat, lon: p.lon, source: "preset", preset_id: p.id, area_km: p.area_km, elev: p.elev }, area_km: p.area_km }));
  };
  const pickHit = (h: MountainSearchHit) => {
    resetResult(); setSearchOpen(false); setQ("");
    setSpec((s) => ({ ...s, place: { name: h.name, lat: h.lat, lon: h.lon, source: h.source, preset_id: h.preset_id, area_km: h.area_km, elev: h.elev }, area_km: h.area_km ?? 4 }));
    // знайдене не з каталогу — показуємо мапу, щоб людина бачила, яку саме ділянку взято
    if (h.source !== "preset") setShowMap(true);
    import("@/lib/analytics").then((m) => m.track("mountain_search_pick", { source: h.source, name: h.name })).catch(() => {});
  };
  const pickOnMap = useCallback((lat: number, lon: number) => {
    setResult(null);
    setSpec((s) => ({ ...s, place: { name: `${lat.toFixed(4)}, ${lon.toFixed(4)}`, lat, lon, source: "map" }, area_km: s.area_km ?? 4 }));
  }, []);

  const setFrame = (patch: Partial<MountainSpec["frame"]>) => setSpec((s) => ({ ...s, frame: { ...s.frame, ...patch } }));
  const setStyle = (id: StyleId) => { const st = STYLES[id]; setSpec((s) => ({ ...s, frame: { ...st.frame }, sides: st.sides })); };
  const styleOf = (s: MountainSpec): StyleId | null =>
    (Object.keys(STYLES) as StyleId[]).find((k) => STYLES[k].frame.style === s.frame.style && STYLES[k].sides === s.sides) ?? null;
  const activeStyle = styleOf(spec);

  const newFigure = (f: MountainFigure): MountainFigureSpec => ({ id: f.id, where: f.kind === "climbing" ? "steepest" : f.kind === "building" ? "slope" : "summit", height_mm: null });
  const addFigure = (f: MountainFigure) => setSpec((s) => ({ ...s, figures: [...s.figures, newFigure(f)].slice(0, 6) }));
  const toggleFigure = (f: MountainFigure) => setSpec((s) => (s.figures.some((x) => x.id === f.id)
    ? { ...s, figures: s.figures.filter((x) => x.id !== f.id) }
    : { ...s, figures: [...s.figures, newFigure(f)].slice(0, 6) }));
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
      import("@/lib/analytics").then((m) => m.track("mountain_generate", { place: spec.place?.name, size: spec.size_mm, figures: spec.figures.length, style: activeStyle || "custom" })).catch(() => {});
    } catch (e: any) { setError(e?.response?.data?.detail || e?.message || t("genFailed")); setBusy(false); }
  };

  useEffect(() => {
    if (!taskId) return;
    stop();
    let failures = 0;
    timer.current = setInterval(async () => {
      try {
        const s: any = await api.getStatus(taskId);
        failures = 0;
        setProgress(Number(s.progress) || 0); setStatusMsg(s.message || "");
        if (s.status === "completed") {
          stop(); setBusy(false);
          // /files/*.3mf на проді закритий SafeStatic (друк-файли не віддаються за іменем, task #70) —
          // друк-файл беремо через /api/download/<task>?format=3mf (стрімить локальний файл задачі).
          setResult({ glb: abs(s.download_url_glb) || "", print: `${API_BASE}/api/download/${taskId}?format=3mf`, spec: (s.world_spec as MountainResultSpec) || null });
          setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
        } else if (s.status === "failed" || s.status === "error") { stop(); setBusy(false); setError(s.message || t("genFailed")); }
      } catch {
        // 404 після рестарту бекенда — інакше кнопка вічно крутилась би «Генерую…»
        failures += 1;
        if (failures >= 8) { stop(); setBusy(false); setError(t("genFailed")); }
      }
    }, 1500);
    return () => stop();
  }, [taskId]); // eslint-disable-line react-hooks/exhaustive-deps

  const shareUrl = () => (taskId && typeof window !== "undefined" ? `${window.location.origin}/share/${taskId}` : "");
  const openChat = async (channel: "tg" | "ig") => {
    import("@/lib/analytics").then((m) => m.track("messenger_order", { channel, product: "mountain" })).catch(() => {});
    const text = t("msgPrefill", { place: spec.place?.name || "", size: spec.size_mm, link: shareUrl() });
    try { await navigator.clipboard.writeText(text); setCopied(true); } catch { setCopied(false); }
    window.open(channel === "tg" ? TG_URL : "https://ig.me/m/monadruk", "_blank", "noopener");
  };
  const doShare = async () => {
    const url = shareUrl(); if (!url) return;
    try { if (typeof navigator.share === "function") await navigator.share({ url, title: "Monadruk" }); else await navigator.clipboard.writeText(url); setShared(true); } catch { /* скасовано */ }
  };

  const figName = (id: string) => figures.find((f) => f.id === id)?.name || id;
  const figMeta = (id: string) => figures.find((f) => f.id === id);
  const chip = (active: boolean) => `min-h-10 rounded-full border px-3.5 py-2 text-[12.5px] font-semibold transition ${active ? "border-[var(--accent-strong)] bg-[rgba(15,118,110,0.1)] text-[var(--accent-strong)]" : "border-[var(--surface-border)] bg-white text-[var(--text-secondary)] hover:border-[rgba(11,92,87,0.3)]"}`;
  const card = (active: boolean) => `rounded-2xl border p-3 text-left transition ${active ? "border-[var(--accent-strong)] bg-[rgba(15,118,110,0.07)] ring-2 ring-[rgba(15,118,110,0.18)]" : "border-[var(--surface-border)] bg-white hover:border-[rgba(11,92,87,0.4)]"}`;
  const stepTitle = (n: number, label: string) => (
    <h2 className="flex items-center gap-2 text-[15px] font-semibold text-[var(--text-primary)]">
      <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-[var(--accent-strong)] text-[12px] font-bold text-white">{n}</span>
      {label}
    </h2>
  );

  const agentExamples = useMemo(() => [t("agentEx1"), t("agentEx2"), t("agentEx3")], [t]);
  const visiblePresets = showAllPeaks ? presets : presets.slice(0, POPULAR);
  const sizeCm = spec.size_mm / 10;
  const figureCount = spec.figures.length;
  const reliefMm = spec.height_mm ?? (preview ? Math.round(preview.relief_mm_natural) : null);

  // підсумок одним рядком — людина бачить, ЩО саме буде надруковано, поруч із кнопкою
  const summary = spec.place ? [
    spec.place.name,
    `${sizeCm}×${sizeCm} ${t("cm")}`,
    preview ? `1:${fmt(preview.scale)}` : null,
    reliefMm ? t("summaryRelief", { mm: reliefMm }) : null,
    activeStyle ? t(`style_${activeStyle}`) : t("style_custom"),
    figureCount ? t("summaryFigures", { n: figureCount }) : null,
  ].filter(Boolean).join(" · ") : "";

  const generateBtn = () => (
    <button type="button" onClick={generate} disabled={busy || !spec.place} data-testid="mnt-generate"
      className="inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-full bg-[var(--accent-strong)] px-5 py-3 text-[15px] font-semibold text-white shadow-[0_10px_24px_rgba(15,118,110,0.25)] transition hover:brightness-110 disabled:opacity-50">
      {busy ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
      {busy ? `${progress}% · ${statusMsg || t("generating")}` : t("generateButton")}
    </button>
  );

  return (
    <div id="main-content" tabIndex={-1} className="mx-auto max-w-[1180px] px-4 pb-32 pt-6 lg:pb-12">
      {/* Заголовок (badge + H1 + підзаголовок) рендериться на сервері в mountains/layout.tsx —
          ця студія ssr:false, і H1 звідси Google у HTML не бачив (аудит 24.09.2026). */}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr),minmax(0,1.05fr)]">
        {/* ЛІВА: три кроки */}
        <section className="space-y-6 rounded-[28px] border border-[var(--surface-border)] bg-[var(--surface-panel)] p-4 shadow-[0_18px_60px_rgba(15,23,42,0.07)] sm:p-5">
          {/* AI — згорнутий помічник, а не перше, що бачить людина */}
          <div>
            <button type="button" onClick={() => setShowAgent((v) => !v)} aria-expanded={showAgent} data-testid="mnt-agent-toggle"
              className="flex min-h-11 w-full items-center gap-2 rounded-2xl border border-dashed border-[rgba(15,118,110,0.35)] bg-[rgba(15,118,110,0.04)] px-3.5 py-2.5 text-left text-[13px] font-semibold text-[var(--accent-strong)] transition hover:bg-[rgba(15,118,110,0.08)]">
              <Sparkles size={15} /> <span className="flex-1">{t("agentToggle")}</span>
              <ChevronDown size={16} className={`transition ${showAgent ? "rotate-180" : ""}`} />
            </button>
            {showAgent && (
              <div className="mt-2">
                <AgentBox<MountainSpec>
                  testId="mnt-agent"
                  placeholder={t("agentPlaceholder")}
                  examples={agentExamples}
                  ask={(text) => api.mountainsAgent(text, locale, spec)}
                  onApply={(s) => { resetResult(); setSpec({ ...DEFAULT_SPEC, ...s }); if (s.place && s.place.source !== "preset") setShowMap(true); }}
                />
              </div>
            )}
          </div>

          {/* Крок 1: гора */}
          <div>
            {stepTitle(1, t("stepPlace"))}
            <div className="relative mt-3">
              <Search size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-secondary)]" />
              <input
                value={q} onChange={(e) => { setQ(e.target.value); setSearchOpen(true); }} onFocus={() => setSearchOpen(true)}
                onBlur={() => setTimeout(() => setSearchOpen(false), 150)}
                onKeyDown={(e) => { if (e.key === "Enter" && shownHits[0]) { e.preventDefault(); pickHit(shownHits[0]); } if (e.key === "Escape") setSearchOpen(false); }}
                placeholder={t("searchPlaceholder")} aria-label={t("searchLabel")} data-testid="mnt-search"
                className="min-h-12 w-full rounded-full border border-[var(--surface-border)] bg-white pl-10 pr-10 text-[14px] text-[var(--text-primary)] outline-none transition focus:border-[var(--accent-strong)] focus:ring-2 focus:ring-[rgba(15,118,110,0.15)]"
              />
              {searching && <Loader2 size={15} className="absolute right-3.5 top-1/2 -translate-y-1/2 animate-spin text-[var(--text-secondary)]" />}
              {searchOpen && q.trim().length >= 2 && (
                <ul role="listbox" data-testid="mnt-search-results" className="absolute inset-x-0 top-full z-30 mt-1.5 max-h-[320px] overflow-auto rounded-2xl border border-[var(--surface-border)] bg-white p-1 shadow-[0_18px_40px_rgba(15,23,42,0.14)]">
                  {shownHits.map((h, i) => (
                    <li key={`${h.lat},${h.lon},${i}`}>
                      <button type="button" role="option" aria-selected={false} onMouseDown={(e) => e.preventDefault()} onClick={() => pickHit(h)}
                        className="flex w-full items-start gap-2 rounded-xl px-3 py-2 text-left transition hover:bg-[rgba(15,118,110,0.07)]">
                        <MapPin size={14} className="mt-[3px] shrink-0 text-[var(--accent-strong)]" />
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-[13.5px] font-semibold text-[var(--text-primary)]">{h.name}{h.elev ? <span className="font-normal text-[var(--text-secondary)]"> · {h.elev} м</span> : null}</span>
                          {h.display && <span className="block truncate text-[11.5px] text-[var(--text-secondary)]">{h.display}</span>}
                        </span>
                      </button>
                    </li>
                  ))}
                  {!searching && shownHits.length === 0 && (hits !== null || q.trim().length < 3) && (
                    <li className="px-3 py-2.5 text-[12.5px] text-[var(--text-secondary)]">{t("searchNothing")}</li>
                  )}
                  {searching && shownHits.length === 0 && (
                    <li className="px-3 py-2.5 text-[12.5px] text-[var(--text-secondary)]">{t("searching")}</li>
                  )}
                </ul>
              )}
            </div>

            <div className="mt-3 grid grid-cols-3 gap-2" data-testid="mnt-presets">
              {visiblePresets.map((p) => {
                const active = spec.place?.preset_id === p.id;
                return (
                  <button key={p.id} type="button" onClick={() => pickPreset(p)} data-testid={`mnt-preset-${p.id}`} aria-pressed={active}
                    className={`group overflow-hidden rounded-2xl border text-left transition ${active ? "border-[var(--accent-strong)] ring-2 ring-[rgba(15,118,110,0.25)]" : "border-[var(--surface-border)] hover:border-[rgba(11,92,87,0.4)]"}`}>
                    <div className="relative aspect-[4/3] bg-[rgba(15,23,42,0.06)]">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={p.photo} alt={p.name} loading="lazy" className="h-full w-full object-cover" onError={(e) => { (e.currentTarget as HTMLImageElement).style.visibility = "hidden"; }} />
                      {active && <span className="absolute left-1.5 top-1.5 inline-flex h-5 w-5 items-center justify-center rounded-full bg-[var(--accent-strong)] text-white"><Check size={12} /></span>}
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
            <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1">
              {presets.length > POPULAR && (
                <button type="button" onClick={() => setShowAllPeaks((v) => !v)} className="min-h-10 text-[12.5px] font-semibold text-[var(--accent-strong)] underline-offset-2 hover:underline">
                  {showAllPeaks ? t("showLess") : t("showAll", { n: presets.length })}
                </button>
              )}
              <button type="button" onClick={() => setShowMap((v) => !v)} data-testid="mnt-tab-map" aria-expanded={showMap}
                className="inline-flex min-h-10 items-center gap-1.5 text-[12.5px] font-semibold text-[var(--accent-strong)] underline-offset-2 hover:underline">
                <MapIcon size={14} /> {showMap ? t("hideMap") : spec.place ? t("refineOnMap") : t("pickOnMap")}
              </button>
            </div>
            {showMap && (
              <div className="mt-2">
                <p className="mb-2 text-[12px] text-[var(--text-secondary)]">{t("mapHint")}</p>
                <MountainMapPicker lat={spec.place?.lat ?? null} lon={spec.place?.lon ?? null} areaKm={areaKm} onPick={pickOnMap} height={300} />
              </div>
            )}
            {spec.place && (
              <p className="mt-2 inline-flex flex-wrap items-center gap-1 text-[12.5px] text-[var(--text-primary)]" data-testid="mnt-place">
                <MapPin size={13} className="text-[var(--accent-strong)]" /> <b>{spec.place.name}</b>
                <span className="text-[var(--text-secondary)]">· {spec.place.lat.toFixed(4)}, {spec.place.lon.toFixed(4)}</span>
              </p>
            )}

            <div className="mt-3">
              <div className="text-[13px] font-semibold text-[var(--text-primary)]">{t("areaLabel")} <span className="font-normal text-[var(--text-secondary)]">· {areaKm} × {areaKm} км</span></div>
              <div className="mt-1.5 flex flex-wrap gap-1.5" role="radiogroup" aria-label={t("areaLabel")}>
                {AREAS.map((a) => <button key={a} type="button" role="radio" aria-checked={areaKm === a} onClick={() => setSpec((s) => ({ ...s, area_km: a }))} className={chip(areaKm === a)}>{a} км</button>)}
              </div>
              <p className="mt-1 text-[11px] text-[var(--text-secondary)]">{t("areaHint")}</p>
            </div>
          </div>

          {/* Крок 2: розмір */}
          <div>
            {stepTitle(2, t("stepSize"))}
            <div className="mt-3 flex flex-wrap gap-1.5" role="radiogroup" aria-label={t("sizeLabel")}>
              {SIZES_CM.map((cm) => (
                <button key={cm} type="button" role="radio" aria-checked={spec.size_mm === cm * 10} data-testid={`mnt-size-${cm * 10}`}
                  onClick={() => { setCustomCm(""); setSpec((s) => ({ ...s, size_mm: cm * 10 })); }} className={chip(spec.size_mm === cm * 10)}>
                  {cm} {t("cm")}
                </button>
              ))}
              <input value={customCm} inputMode="decimal" placeholder={t("sizeCustomCm")} aria-label={t("sizeCustomCm")}
                onChange={(e) => { setCustomCm(e.target.value); const v = Number(e.target.value.replace(",", ".")); if (v >= 6 && v <= 40) setSpec((s) => ({ ...s, size_mm: Math.round(v * 10) })); }}
                className={`min-h-10 w-28 rounded-full border bg-white px-3 text-[12.5px] outline-none focus:border-[var(--accent-strong)] ${customCm && !SIZES_CM.includes(sizeCm) ? "border-[var(--accent-strong)]" : "border-[var(--surface-border)]"}`} />
            </div>
            <p className="mt-1 text-[11px] text-[var(--text-secondary)]">{spec.size_mm > (spec.bed_mm ?? 256) ? t("sizeTiles", { bed: spec.bed_mm ?? 256 }) : t("sizeOne")}</p>
          </div>

          {/* Крок 3: вигляд */}
          <div>
            {stepTitle(3, t("stepLook"))}
            <div className="mt-3 grid grid-cols-3 gap-2" role="radiogroup" aria-label={t("stepLook")}>
              {(Object.keys(STYLES) as StyleId[]).map((id) => (
                <button key={id} type="button" role="radio" aria-checked={activeStyle === id} data-testid={`mnt-style-${id}`} onClick={() => setStyle(id)} className={card(activeStyle === id)}>
                  <div className="text-[13px] font-semibold text-[var(--text-primary)]">{t(`style_${id}`)}</div>
                  <div className="mt-0.5 text-[11px] leading-snug text-[var(--text-secondary)]">{t(`style_${id}_d`)}</div>
                </button>
              ))}
            </div>

            <div className="mt-4 text-[13px] font-semibold text-[var(--text-primary)]">{t("figuresShort")}</div>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {figures.map((f) => {
                const on = spec.figures.some((x) => x.id === f.id);
                return (
                  <button key={f.id} type="button" onClick={() => toggleFigure(f)} aria-pressed={on} data-testid={`mnt-fig-add-${f.id}`}
                    className={`inline-flex min-h-10 items-center gap-1.5 rounded-full border px-3 text-[12.5px] font-semibold transition ${on ? "border-[var(--accent-strong)] bg-[rgba(15,118,110,0.1)] text-[var(--accent-strong)]" : "border-[var(--surface-border)] bg-white text-[var(--text-primary)] hover:border-[var(--accent-strong)]"}`}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={f.thumb} alt="" className="h-6 w-6 rounded-full object-cover" onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }} />
                    {on ? <Check size={13} /> : <Plus size={13} />} {f.name}
                  </button>
                );
              })}
            </div>
            <p className="mt-1 text-[11px] text-[var(--text-secondary)]">{t("figuresHint")}</p>
          </div>

          {/* Точні налаштування — згорнуті: більшості людей вистачає трьох кроків вище */}
          <div className="rounded-2xl border border-[var(--surface-border)] bg-white/60">
            <button type="button" onClick={() => setShowAdvanced((v) => !v)} aria-expanded={showAdvanced} data-testid="mnt-advanced"
              className="flex min-h-12 w-full items-center gap-2 px-3.5 py-3 text-left">
              <SlidersHorizontal size={15} className="text-[var(--text-secondary)]" />
              <span className="flex-1">
                <span className="block text-[13px] font-semibold text-[var(--text-primary)]">{t("advanced")}</span>
                <span className="block text-[11px] text-[var(--text-secondary)]">{t("advancedHint")}</span>
              </span>
              <ChevronDown size={16} className={`text-[var(--text-secondary)] transition ${showAdvanced ? "rotate-180" : ""}`} />
            </button>
            {showAdvanced && (
              <div className="space-y-4 border-t border-[var(--surface-border)] px-3.5 pb-4 pt-3">
                <div>
                  <div className="text-[13px] font-semibold text-[var(--text-primary)]">{t("heightLabel")}</div>
                  <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
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

                <div>
                  <div className="text-[13px] font-semibold text-[var(--text-primary)]">{t("frameLabel")}</div>
                  <div className="mt-1.5 flex flex-wrap gap-1.5" role="radiogroup" aria-label={t("frameLabel")}>
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

                <div>
                  <div className="text-[13px] font-semibold text-[var(--text-primary)]">{t("sidesLabel")}</div>
                  <div className="mt-1.5 flex flex-wrap gap-1.5" role="radiogroup" aria-label={t("sidesLabel")}>
                    {(["slope", "rock", "vertical"] as const).map((k) => <button key={k} type="button" role="radio" aria-checked={spec.sides === k} data-testid={`mnt-sides-${k}`} onClick={() => setSpec((s) => ({ ...s, sides: k }))} className={chip(spec.sides === k)}>{t(`sides_${k}`)}</button>)}
                  </div>
                  <p className="mt-1 text-[11px] text-[var(--text-secondary)]">{t(`sidesHint_${spec.sides}`)}</p>
                </div>

                <div>
                  <div className="text-[13px] font-semibold text-[var(--text-primary)]">{t("figuresLabel")}</div>
                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                    {figures.map((f) => (
                      <button key={f.id} type="button" onClick={() => addFigure(f)} disabled={spec.figures.length >= 6}
                        className="inline-flex min-h-9 items-center gap-1 rounded-full border border-[var(--surface-border)] bg-white px-2.5 text-[12px] font-semibold text-[var(--text-primary)] transition hover:border-[var(--accent-strong)] disabled:opacity-50">
                        <Plus size={12} /> {f.name}
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
                </div>

                <label className="flex items-center gap-2 text-[12.5px] text-[var(--text-primary)]">
                  <input type="checkbox" checked={spec.texture !== "none"} onChange={(e) => setSpec((s) => ({ ...s, texture: e.target.checked ? "satellite" : "none" }))} className="accent-[var(--accent-strong)]" /> {t("textureLabel")}
                </label>
              </div>
            )}
          </div>

          <p className="text-[11px] leading-4 text-[var(--text-secondary)]">{t("hint")}</p>
          <Link href="/worlds" className="inline-flex min-h-10 items-center gap-1.5 text-[12.5px] font-semibold text-[var(--accent-strong)] underline-offset-2 hover:underline">
            <Globe2 size={14} /> {t("worldsLink")}
          </Link>
        </section>

        {/* ПРАВА: превʼю / результат — липка на десктопі, щоб кнопка й підсумок були завжди видно */}
        <section className="flex min-h-[460px] flex-col gap-3 self-start rounded-[28px] border border-[var(--surface-border)] bg-[var(--surface-panel)] p-3 shadow-[0_18px_60px_rgba(15,23,42,0.07)] lg:sticky lg:top-[136px]">
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
                    className={`aspect-square w-full object-cover transition ${previewBusy ? "opacity-60" : ""} ${pickingFig != null ? "cursor-crosshair ring-4 ring-[var(--accent-strong)]" : ""}`} />
                  {spec.figures.filter((f) => f.where === "point" && f.fx != null).map((f, i) => (
                    <span key={i} className="pointer-events-none absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-[var(--accent-strong)]" style={{ left: `${(f.fx ?? 0) * 100}%`, top: `${(1 - (f.fy ?? 0)) * 100}%` }} />
                  ))}
                  {pickingFig != null && <div className="absolute inset-x-0 top-0 bg-[var(--accent-strong)]/90 px-3 py-1.5 text-center text-[12px] font-semibold text-white">{t("pickPointHint")}</div>}
                  <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/60 to-transparent px-3 pb-2 pt-6 text-[11.5px] text-white">
                    {t("previewStats", { min: preview.elev_min, max: preview.elev_max, scale: fmt(preview.scale) })} · {preview.sources.join(" + ")}
                  </div>
                  {busy && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-white/70 backdrop-blur-[2px]">
                      <Loader2 size={28} className="animate-spin text-[var(--accent-strong)]" />
                      <p className="px-6 text-center text-[13px] font-semibold text-[var(--text-primary)]">{statusMsg || t("generating")}</p>
                      <div className="h-2 w-2/3 overflow-hidden rounded-full bg-[rgba(15,23,42,0.1)]"><div className="h-full bg-[var(--accent-strong)] transition-all" style={{ width: `${progress}%` }} /></div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex aspect-square items-center justify-center rounded-2xl bg-[rgba(15,23,42,0.03)] text-center text-[var(--text-secondary)]">
                  <div>
                    {previewBusy ? <Loader2 size={30} className="mx-auto animate-spin" /> : <div className="text-4xl">🏔️</div>}
                    <p className="mt-3 max-w-xs text-sm">{previewErr || (previewBusy ? t("previewLoading") : t("previewEmpty"))}</p>
                  </div>
                </div>
              )}

              {summary && <p className="mt-3 px-1 text-center text-[13px] font-semibold text-[var(--text-primary)]" data-testid="mnt-summary">{summary}</p>}
              <div className="mt-3 hidden lg:block">{generateBtn()}</div>
              <p className="mt-1.5 text-center text-[11.5px] text-[var(--text-secondary)]">{spec.place ? t("etaHint") : t("needPlace")}</p>
              {error && <p role="alert" data-testid="mnt-error" className="mt-2 rounded-xl bg-[#fdf3f3] px-3 py-2 text-center text-[13px] text-[#8a2b2b]">{error}</p>}
              <a href={TG_URL} target="_blank" rel="noopener" onClick={() => { import("@/lib/analytics").then((m) => m.track("messenger_open", { channel: "tg", from: "mountains" })).catch(() => {}); }}
                className="mx-auto mt-2 inline-flex min-h-10 items-center gap-1.5 text-[12.5px] font-semibold text-[var(--text-secondary)] transition hover:text-[var(--text-primary)]">
                <Send size={13} className="text-[#2AABEE]" /> {t("askUs")}
              </a>
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

              {/* Головна дія після генерації — замовити друк; файли — другим рядом */}
              <div className="mt-3 rounded-2xl border border-[rgba(15,118,110,0.3)] bg-[rgba(15,118,110,0.05)] p-3" data-testid="mnt-order">
                <p className="text-[14px] font-semibold text-[var(--text-primary)]">{t("orderTitle")}</p>
                <p className="mt-1 text-[12px] leading-snug text-[var(--text-secondary)]">{t("orderSub")}</p>
                <div className="mt-2.5 grid grid-cols-2 gap-2">
                  <button type="button" onClick={() => openChat("tg")} className="inline-flex min-h-11 items-center justify-center gap-1.5 rounded-full bg-[#2AABEE] px-3 text-[13px] font-semibold text-white transition hover:brightness-105"><Send size={14} /> Telegram</button>
                  <button type="button" onClick={() => openChat("ig")} className="inline-flex min-h-11 items-center justify-center gap-1.5 rounded-full border border-[var(--surface-border)] bg-white px-3 text-[13px] font-semibold text-[var(--text-primary)] transition hover:border-[var(--accent-strong)]"><Instagram size={14} className="text-[#E1306C]" /> Instagram</button>
                </div>
                <p className="mt-1.5 text-center text-[11.5px] text-[var(--text-secondary)]" aria-live="polite">{copied ? t("msgCopied") : ""}</p>
              </div>

              <div className="mt-2 grid gap-2 sm:grid-cols-2">
                {result.print && <a href={result.print} download className="inline-flex min-h-11 items-center justify-center gap-1.5 rounded-full border border-[var(--accent-strong)] px-4 text-[13px] font-semibold text-[var(--accent-strong)] transition hover:bg-[rgba(15,118,110,0.06)]"><Download size={14} /> {t("downloadPrint")}</a>}
                {result.spec?.tiles_zip && <a href={abs(result.spec.tiles_zip)!} download className="inline-flex min-h-11 items-center justify-center gap-1.5 rounded-full border border-[var(--accent-strong)] px-4 text-[13px] font-semibold text-[var(--accent-strong)]"><Download size={14} /> {t("downloadTiles")}</a>}
                {result.spec?.preview_png && <a href={abs(result.spec.preview_png)!} target="_blank" rel="noopener" className="inline-flex min-h-11 items-center justify-center gap-1.5 rounded-full border border-[var(--surface-border)] bg-white px-4 text-[13px] font-semibold text-[var(--text-primary)]"><ImageIcon size={14} /> {t("photoTop")}</a>}
                {result.spec?.paint_jpg && <a href={abs(result.spec.paint_jpg)!} target="_blank" rel="noopener" className="inline-flex min-h-11 items-center justify-center gap-1.5 rounded-full border border-[var(--surface-border)] bg-white px-4 text-[13px] font-semibold text-[var(--text-primary)]"><ImageIcon size={14} /> {t("paintGuide")}</a>}
                <a href={result.glb} download className="inline-flex min-h-11 items-center justify-center gap-1.5 rounded-full border border-[var(--surface-border)] bg-white px-4 text-[13px] font-semibold text-[var(--text-primary)]"><Download size={14} /> {t("downloadGlb")}</a>
                <button type="button" onClick={() => setResult(null)} className="inline-flex min-h-11 items-center justify-center rounded-full border border-[var(--surface-border)] bg-white px-4 text-[13px] font-semibold text-[var(--text-primary)]">{t("editAgain")}</button>
              </div>
              <button type="button" onClick={doShare} className="mx-auto mt-2 flex min-h-10 items-center gap-1.5 text-[12px] font-semibold text-[var(--accent-strong)] underline underline-offset-2"><Share2 size={13} /> {shared ? t("shareCopied") : t("shareLink")}</button>
            </div>
          )}
        </section>
      </div>

      {/* Мобільний: підсумок + кнопка завжди під пальцем (раніше кнопка була під 7 блоками налаштувань) */}
      {!result && (
        <div className="fixed inset-x-0 bottom-0 z-30 border-t border-[var(--surface-border)] bg-[rgba(250,248,242,0.97)] px-4 pb-[calc(env(safe-area-inset-bottom,0px)+10px)] pt-2.5 backdrop-blur lg:hidden" data-testid="mnt-mobile-bar">
          {summary && <p className="mb-1.5 truncate text-center text-[12px] font-semibold text-[var(--text-primary)]">{summary}</p>}
          {generateBtn()}
        </div>
      )}
    </div>
  );
}
