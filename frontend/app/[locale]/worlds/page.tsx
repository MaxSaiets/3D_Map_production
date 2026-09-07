"use client";

import { useState, useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import { Send, Instagram, Share2 } from "lucide-react";
import { useTranslations } from "next-intl";
import { api } from "@/lib/api";
import { BetaBanner } from "@/components/BetaBanner";

const Model3DViewer = dynamic(() => import("@/components/Model3DViewer"), { ssr: false });

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

// Приклади-промти (швидке заповнення). Ключі → i18n.
const EXAMPLES = [
  "epicMountains", "volcanoIsland", "deepCanyon", "rollingHills", "alienCrater", "desertDunes",
] as const;

// Форми, які реально вміє бекенд (services/procedural_generator.SHAPES) + "auto".
// ТРИМАТИ В СИНХРОНІ з бекендом: невідому форму бекенд мовчки замінить на mountain.
const SHAPES = [
  "auto", "mountain", "volcano", "island", "archipelago",
  "valley", "plateau", "crater", "ridges", "rolling",
] as const;
type Shape = (typeof SHAPES)[number];

const SIZES: { key: string; mm: number }[] = [
  { key: "s", mm: 80 }, { key: "m", mm: 120 }, { key: "l", mm: 180 },
];

export default function WorldsPage() {
  const t = useTranslations("worlds");
  const [prompt, setPrompt] = useState("");
  const [sizeMm, setSizeMm] = useState(120);
  const [shape, setShape] = useState<Shape>("auto");
  const [variant, setVariant] = useState(0);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [glbUrl, setGlbUrl] = useState<string | null>(null);
  const [printUrl, setPrintUrl] = useState<string | null>(null);
  const [builtShape, setBuiltShape] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState("");
  const [slow, setSlow] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const ticksRef = useRef(0);

  const [copied, setCopied] = useState(false);
  const [shared, setShared] = useState(false);

  // Опис для чату + посилання на 3D-сцену: t.me/ig.me не вміють prefill для
  // звичайних акаунтів, тож кладемо текст у буфер і відкриваємо чат (той самий
  // прийом, що в SalesAlternatives для мап і брелоків).
  const shareUrl = () => (taskId && typeof window !== "undefined" ? `${window.location.origin}/share/${taskId}` : "");
  const openChat = async (channel: "tg" | "ig") => {
    import("@/lib/analytics").then((m) => m.track("messenger_order", { channel, product: "world" })).catch(() => {});
    const text = t("msgPrefill", { shape: builtShape || "", size: sizeMm, link: shareUrl() });
    try { await navigator.clipboard.writeText(text); setCopied(true); } catch { setCopied(false); }
    window.open(channel === "tg" ? "https://t.me/monadruk" : "https://ig.me/m/monadruk", "_blank", "noopener");
  };
  const doShare = async () => {
    const url = shareUrl();
    if (!url) return;
    import("@/lib/analytics").then((m) => m.track("guided_share", { product: "world" })).catch(() => {});
    try {
      if (typeof navigator.share === "function") { await navigator.share({ url, title: "Monadruk" }); }
      else { await navigator.clipboard.writeText(url); }
      setShared(true);
    } catch { /* користувач скасував — нічого не показуємо */ }
  };

  const stopPolling = () => { if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; } };
  useEffect(() => () => stopPolling(), []);

  const start = async (nextVariant: number) => {
    const p = prompt.trim();
    if (!p || busy) return;
    setError(null); setGlbUrl(null); setPrintUrl(null); setBuiltShape(null);
    setBusy(true); setProgress(5); setSlow(false); ticksRef.current = 0;
    setStatusMsg(t("starting")); setVariant(nextVariant);
    try {
      const resp = await api.generateCustom(p, sizeMm, {
        shape: shape === "auto" ? undefined : shape,
        variant: nextVariant,
      });
      setTaskId(resp.task_id);
    } catch (e: any) {
      setError(e?.message || t("genFailed")); setBusy(false);
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
        ticksRef.current += 1;
        if (ticksRef.current > 40) setSlow(true);   // ~60 c
        if (s.status === "completed") {
          stopPolling();
          // Беремо URL зі статусу (раніше шлях складався вручну з task_id — ламався
          // щоразу, коли бекенд міняв іменування файлів).
          const glb = s.download_url_glb || `/api/files/custom_${taskId.slice(0, 8)}.glb`;
          setGlbUrl(glb.startsWith("http") ? glb : `${API_BASE}${glb}`);
          const p3 = s.download_url_3mf || s.download_url;
          setPrintUrl(p3 ? (p3.startsWith("http") ? p3 : `${API_BASE}${p3}`) : null);
          setBuiltShape(s.world_spec?.shapeUk || s.world_spec?.shape || null);
          setBusy(false);
        } else if (s.status === "failed" || s.status === "error") {
          stopPolling(); setError(s.message || t("genFailed")); setBusy(false);
        }
      } catch {
        /* транзієнтна помилка полінгу — наступний тік повторить */
      }
    }, 1500);
    return () => stopPolling();
  }, [taskId]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <>
      <BetaBanner mode="worlds" />
      <div id="main-content" tabIndex={-1} className="mx-auto max-w-[1100px] px-4 py-10 sm:py-14">
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
            <label htmlFor="world-prompt" className="block text-sm font-semibold text-[var(--text-primary)]">{t("inputLabel")}</label>
            <textarea
              id="world-prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder={t("inputPlaceholder")}
              rows={4}
              maxLength={2000}
              data-testid="world-prompt"
              className="mt-2 w-full resize-none rounded-2xl border border-[var(--surface-border)] bg-white/90 px-4 py-3 text-[var(--text-primary)] outline-none focus:border-[var(--accent-strong)]"
            />
            <div className="mt-3 flex flex-wrap gap-2">
              {EXAMPLES.map((k) => (
                <button key={k} type="button" onClick={() => setPrompt(t(`ex.${k}`))}
                  className="min-h-10 rounded-full border border-[var(--surface-border)] bg-white/80 px-3 py-1.5 text-[12px] text-[var(--text-secondary)] transition hover:border-[var(--accent-strong)] hover:text-[var(--text-primary)]">
                  {t(`ex.${k}`)}
                </button>
              ))}
            </div>

            {/* Вибір форми: раніше форму вгадував лише парсер і користувач ніяк не
                міг це виправити — «отримую не те, що просив». */}
            <div className="mt-4">
              <div className="text-sm font-semibold text-[var(--text-primary)]">{t("shapeLabel")}</div>
              <div className="mt-2 flex flex-wrap gap-2" role="radiogroup" aria-label={t("shapeLabel")}>
                {SHAPES.map((s) => (
                  <button key={s} type="button" role="radio" aria-checked={shape === s}
                    data-testid={`world-shape-${s}`}
                    onClick={() => setShape(s)}
                    className={`min-h-10 rounded-full border px-3.5 py-2 text-[12.5px] font-semibold transition ${
                      shape === s ? "border-[var(--accent-strong)] bg-[rgba(15,118,110,0.1)] text-[var(--accent-strong)]"
                                  : "border-[var(--surface-border)] bg-white text-[var(--text-secondary)] hover:border-[rgba(11,92,87,0.3)]"}`}>
                    {t(`shape_${s}`)}
                  </button>
                ))}
              </div>
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

            <button type="button" onClick={() => start(0)} disabled={busy || !prompt.trim()}
              data-testid="world-generate"
              className="mt-5 w-full rounded-full bg-[var(--accent-strong)] px-5 py-3 text-sm font-semibold text-white transition disabled:opacity-50">
              {busy ? `${progress}% · ${statusMsg || t("generating")}` : t("generateButton")}
            </button>
            {busy && slow && <p className="mt-2 text-[12px] text-[var(--text-secondary)]">{t("tooLong")}</p>}
            {error && <p role="alert" data-testid="world-error" className="mt-3 text-sm text-red-600">{error}</p>}
            <p className="mt-3 text-[11px] leading-4 text-[var(--text-secondary)]">{t("hint")}</p>
          </section>

          {/* Превʼю */}
          <section className="flex min-h-[420px] flex-col rounded-[28px] border border-[var(--surface-border)] bg-[var(--surface-panel)] p-3 shadow-[0_18px_60px_rgba(15,23,42,0.07)]">
            {glbUrl ? (
              <>
                <div className="flex-1 overflow-hidden rounded-2xl bg-[rgba(15,23,42,0.03)]">
                  {/* ПАСТКА 08.09: тут стояло flat={false} → вʼюер вважав світ «брелоком»:
                      камера фронтальна, геометрія НЕ клалась горизонтально, і рельєф
                      виглядав завалено-обрізаним. Світ — така сама пласка плитка з
                      Z-вгору, як мапа, тож flat (isMap) = true. */}
                  <Model3DViewer url={glbUrl} height={420} flat allowZoom autoRotate label={t("title")} />
                </div>
                {builtShape && (
                  <p data-testid="world-built" className="mt-2 text-center text-[12.5px] text-[var(--text-secondary)]">
                    {t("builtAs", { shape: builtShape })}
                  </p>
                )}
                <div className="mt-2 grid gap-2 sm:grid-cols-3">
                  <button type="button" onClick={() => start(variant + 1)} disabled={busy}
                    data-testid="world-reroll"
                    className="min-h-11 rounded-full border border-[var(--surface-border)] bg-white px-4 text-[13px] font-semibold text-[var(--text-primary)] transition hover:border-[var(--accent-strong)] disabled:opacity-50">
                    {t("reroll")}
                  </button>
                  <a href={glbUrl} download
                    className="inline-flex min-h-11 items-center justify-center rounded-full border border-[var(--accent-strong)] px-4 text-[13px] font-semibold text-[var(--accent-strong)] transition hover:bg-[rgba(15,118,110,0.08)]">
                    {t("downloadGlb")}
                  </a>
                  {printUrl && (
                    <a href={printUrl} download
                      className="inline-flex min-h-11 items-center justify-center rounded-full bg-[var(--accent-strong)] px-4 text-[13px] font-semibold text-white transition hover:brightness-110">
                      {t("downloadPrint")}
                    </a>
                  )}
                </div>

                {/* Шлях до замовлення. До 08.09 його НЕ БУЛО ЗОВСІМ: смуга режиму
                    казала «напишіть нам», а писати не було куди — глухий кут
                    воронки (у /maket форма є, у /worlds не було). Ціни фіксованої
                    нема (світ друкується під розмір), тож ведемо в чат із готовим
                    описом і посиланням на 3D-сцену. */}
                <div className="mt-3 rounded-2xl border border-[var(--surface-border)] bg-white/70 p-3" data-testid="world-order">
                  <p className="text-[13.5px] font-semibold text-[var(--text-primary)]">{t("orderTitle")}</p>
                  <p className="mt-1 text-[12px] leading-snug text-[var(--text-secondary)]">{t("orderSub")}</p>
                  <div className="mt-2.5 grid grid-cols-2 gap-2">
                    <button type="button" data-testid="world-msg-tg" onClick={() => openChat("tg")}
                      className="inline-flex min-h-11 items-center justify-center gap-1.5 rounded-full border border-[var(--surface-border)] bg-white px-3 text-[12.5px] font-semibold text-[var(--text-primary)] transition hover:border-[var(--accent-strong)]">
                      <Send size={14} className="text-[#2AABEE]" /> Telegram
                    </button>
                    <button type="button" data-testid="world-msg-ig" onClick={() => openChat("ig")}
                      className="inline-flex min-h-11 items-center justify-center gap-1.5 rounded-full border border-[var(--surface-border)] bg-white px-3 text-[12.5px] font-semibold text-[var(--text-primary)] transition hover:border-[var(--accent-strong)]">
                      <Instagram size={14} className="text-[#E1306C]" /> Instagram
                    </button>
                  </div>
                  <button type="button" data-testid="world-share" onClick={doShare}
                    className="mx-auto mt-2 flex min-h-10 items-center gap-1.5 text-[12px] font-semibold text-[var(--accent-strong)] underline underline-offset-2">
                    <Share2 size={13} /> {shared ? t("shareCopied") : t("shareLink")}
                  </button>
                  <p className="mt-1 text-center text-[11.5px] leading-snug text-[var(--text-secondary)]" aria-live="polite">
                    {copied ? t("msgCopied") : ""}
                  </p>
                </div>
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
    </>
  );
}
