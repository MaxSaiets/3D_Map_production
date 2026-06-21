"use client";

import { useState, useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import { useTranslations } from "next-intl";
import { api } from "@/lib/api";

const Model3DViewer = dynamic(() => import("@/components/Model3DViewer"), { ssr: false });

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

// Приклади-промти (швидке заповнення). Ключі → i18n; значення тут лише як fallback.
const EXAMPLES = [
  "epicMountains", "volcanoIsland", "deepCanyon", "rollingHills", "alienCrater", "desertDunes",
] as const;

const SIZES: { key: string; mm: number }[] = [
  { key: "s", mm: 80 }, { key: "m", mm: 120 }, { key: "l", mm: 180 },
];

export default function WorldsPage() {
  const t = useTranslations("worlds");
  const [prompt, setPrompt] = useState("");
  const [sizeMm, setSizeMm] = useState(120);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [glbUrl, setGlbUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState("");
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = () => { if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; } };
  useEffect(() => () => stopPolling(), []);

  const handleGenerate = async () => {
    const p = prompt.trim();
    if (!p || busy) return;
    setError(null); setGlbUrl(null); setBusy(true); setProgress(5); setStatusMsg(t("starting"));
    try {
      const resp = await api.generateCustom(p, sizeMm);
      setTaskId(resp.task_id);
    } catch (e: any) {
      setError(e?.message || t("errGen")); setBusy(false);
    }
  };

  useEffect(() => {
    if (!taskId) return;
    stopPolling();
    timerRef.current = setInterval(async () => {
      try {
        const s: any = await api.getStatus(taskId);
        setProgress(Number(s.progress) || 0);
        setStatusMsg(s.message || "");
        if (s.status === "completed") {
          stopPolling();
          setGlbUrl(`${API_BASE}/api/files/custom_${taskId.slice(0, 8)}.glb`);
          setBusy(false);
        } else if (s.status === "failed" || s.status === "error") {
          stopPolling(); setError(s.message || t("errGen")); setBusy(false);
        }
      } catch {
        /* транзієнтна помилка полінгу — наступний тік повторить */
      }
    }, 1500);
    return () => stopPolling();
  }, [taskId]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="mx-auto max-w-[1100px] px-4 py-10 sm:py-14">
      <header className="mb-8 text-center">
        <span className="inline-block rounded-full border border-[var(--surface-border)] bg-[var(--surface-panel)] px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--text-secondary)]">
          {t("badge")}
        </span>
        <h1 className="mt-4 text-3xl font-semibold text-[var(--text-primary)] sm:text-4xl">{t("title")}</h1>
        <p className="mx-auto mt-3 max-w-2xl text-[var(--text-secondary)]">{t("subtitle")}</p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr),minmax(0,1.1fr)]">
        {/* Ввід */}
        <section className="rounded-[28px] border border-[var(--surface-border)] bg-[var(--surface-panel)] p-5 shadow-[0_18px_60px_rgba(15,23,42,0.07)]">
          <label className="block text-sm font-semibold text-[var(--text-primary)]">{t("inputLabel")}</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder={t("inputPlaceholder")}
            rows={4}
            maxLength={2000}
            className="mt-2 w-full resize-none rounded-2xl border border-[var(--surface-border)] bg-white/90 px-4 py-3 text-[var(--text-primary)] outline-none focus:border-[var(--accent-strong)]"
          />
          <div className="mt-3 flex flex-wrap gap-2">
            {EXAMPLES.map((k) => (
              <button key={k} type="button" onClick={() => setPrompt(t(`ex.${k}`))}
                className="rounded-full border border-[var(--surface-border)] bg-white/80 px-3 py-1.5 text-[12px] text-[var(--text-secondary)] transition hover:border-[var(--accent-strong)] hover:text-[var(--text-primary)]">
                {t(`ex.${k}`)}
              </button>
            ))}
          </div>

          <div className="mt-4">
            <div className="text-sm font-semibold text-[var(--text-primary)]">{t("sizeLabel")}</div>
            <div className="mt-2 grid grid-cols-3 gap-2" role="radiogroup" aria-label={t("sizeLabel")}>
              {SIZES.map(({ key, mm }) => (
                <button key={key} type="button" role="radio" aria-checked={sizeMm === mm} onClick={() => setSizeMm(mm)}
                  className={`min-h-[44px] rounded-2xl border px-3 py-2 text-sm font-semibold transition ${
                    sizeMm === mm ? "border-[var(--accent-strong)] bg-[rgba(15,118,110,0.1)] text-[var(--accent-strong)]"
                                  : "border-[var(--surface-border)] bg-white text-[var(--text-secondary)] hover:border-[rgba(11,92,87,0.3)]"}`}>
                  {t(`size.${key}`)}<span className="ml-1 text-[11px] opacity-70">{mm}мм</span>
                </button>
              ))}
            </div>
          </div>

          <button type="button" onClick={handleGenerate} disabled={busy || !prompt.trim()}
            className="mt-5 w-full rounded-full bg-[var(--accent-strong)] px-5 py-3 text-sm font-semibold text-white transition disabled:opacity-50">
            {busy ? `${progress}% · ${statusMsg || t("generating")}` : t("generateButton")}
          </button>
          {error && <p role="alert" className="mt-3 text-sm text-red-600">{error}</p>}
          <p className="mt-3 text-[11px] leading-4 text-[var(--text-secondary)]">{t("hint")}</p>
        </section>

        {/* Превʼю */}
        <section className="flex min-h-[420px] flex-col rounded-[28px] border border-[var(--surface-border)] bg-[var(--surface-panel)] p-3 shadow-[0_18px_60px_rgba(15,23,42,0.07)]">
          {glbUrl ? (
            <>
              <div className="flex-1 overflow-hidden rounded-2xl bg-[rgba(15,23,42,0.03)]">
                <Model3DViewer url={glbUrl} height={420} flat={false} allowZoom autoRotate label={t("title")} />
              </div>
              <a href={glbUrl} download
                className="mt-3 inline-flex items-center justify-center rounded-full border border-[var(--accent-strong)] px-5 py-2.5 text-sm font-semibold text-[var(--accent-strong)] transition hover:bg-[rgba(15,118,110,0.08)]">
                {t("downloadGlb")}
              </a>
            </>
          ) : (
            <div className="flex flex-1 items-center justify-center rounded-2xl bg-[rgba(15,23,42,0.03)] text-center text-[var(--text-secondary)]">
              <div>
                <div className="text-4xl">🌍</div>
                <p className="mt-3 max-w-xs text-sm">{busy ? (statusMsg || t("generating")) : t("previewEmpty")}</p>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
