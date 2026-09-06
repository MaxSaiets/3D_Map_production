"use client";

import { useEffect, useRef, useState } from "react";
import { Check, Loader2 } from "lucide-react";

/**
 * A-5 (2026-09-03): ОДНА смуга прогресу з названими етапами замість трьох
 * індикаторів із сирим рядком статусу бекенду («Будую рельєф місцевості (~3-5
 * хв)…»). Етап визначаємо за progress-порогами, які бекенд ставить у
 * update_status: map — 15 дані OSM, 20 рельєф, 40 дороги/вода/будівлі, 75
 * збирання, 85 експорт; flat/брелок — 10 дані, 55 шари, 70/80 збирання, 96 файл.
 */
export function GenerationStages({
  progress,
  kind,
  title,
  note,
  stages,
  eta,
  queued,
  queuedTitle,
  queuedNote,
  printPrep,
  printPrepLabel,
  onCancel,
  cancelLabel,
}: {
  progress: number;
  kind: "map" | "flat";
  title: string;
  note: string;
  stages: { data: string; terrain?: string; detail: string; file: string };
  /** Текст ETA (вже сформатований), напр. «≈ 1 хв». Опційно. */
  eta?: string | null;
  /** C-4: сервер зайнятий іншою генерацією — це НЕ «0 %», а окремий стан. */
  queued?: boolean;
  queuedTitle?: string;
  queuedNote?: string;
  /** C-5: підготовка друкарського файлу після превʼю (0–100). */
  printPrep?: number | null;
  printPrepLabel?: string;
  /** C-2: скасувати генерацію. */
  onCancel?: () => void;
  cancelLabel?: string;
}) {
  // D-1 (2026-09-03): бекенд віддає прогрес СТРИБКАМИ (15 → 20 → 50 → 85), тож
  // смуга стояла по 30–60 с і виглядала як зависання. Показуємо плавне значення:
  // підтягуємось до серверного, а між оновленнями повільно повземо вперед — але
  // НЕ більше ніж на 7 п.п. попереду сервера і ніколи не до 100 % (100 = готово).
  const target = Math.max(0, Math.min(100, progress || 0));
  const [shown, setShown] = useState(target);
  const shownRef = useRef(target);
  useEffect(() => { if (target > shownRef.current) { shownRef.current = target; setShown(target); } }, [target]);
  useEffect(() => {
    if (queued) return; // у черзі нічого не «повземо» — робота ще не почалась
    const id = window.setInterval(() => {
      const cap = Math.min(97, target + 7);
      if (shownRef.current < cap) {
        shownRef.current = Math.min(cap, shownRef.current + 0.4);
        setShown(Math.round(shownRef.current * 10) / 10);
      }
    }, 400);
    return () => window.clearInterval(id);
  }, [target, queued]);
  const p = Math.round(Math.max(target, shown));
  const list = kind === "map"
    ? [
        { key: "data", label: stages.data, from: 0 },
        { key: "terrain", label: stages.terrain ?? stages.detail, from: 20 },
        { key: "detail", label: stages.detail, from: 40 },
        { key: "file", label: stages.file, from: 75 },
      ]
    : [
        { key: "data", label: stages.data, from: 0 },
        { key: "detail", label: stages.detail, from: 55 },
        { key: "file", label: stages.file, from: 80 },
      ];
  let active = 0;
  list.forEach((st, i) => { if (p >= st.from) active = i; });

  const q = !!queued;
  return (
    <div className="flex flex-col gap-2 scroll-mt-[72px] rounded-[16px] border border-[var(--surface-border)] bg-white/70 px-3.5 py-3" data-testid="generation-stages">
      <div className="flex items-center justify-between gap-2 text-[14px] font-semibold text-[var(--text-primary)]">
        <span className="inline-flex items-center gap-2"><Loader2 size={16} className="animate-spin text-[var(--accent-strong)]" /> {q ? (queuedTitle ?? title) : title}</span>
        <span className="text-[12px] font-semibold text-[var(--text-secondary)]" style={{ fontVariantNumeric: "tabular-nums" }}>{p}%{eta ? ` · ${eta}` : ""}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-[rgba(15,23,42,0.08)]" role="progressbar" aria-label={title} aria-valuemin={0} aria-valuemax={100} aria-valuenow={p}>
        <div className="h-full rounded-full bg-[var(--accent-strong)] transition-all duration-700" style={{ width: `${Math.max(4, p)}%` }} />
      </div>
      {q && queuedNote && (
        <p className="text-[12px] leading-snug text-[var(--text-secondary)]" data-testid="gen-queued">{queuedNote}</p>
      )}
      <ol className={`flex flex-col gap-1 ${q ? "opacity-50" : ""}`} aria-live="polite">
        {list.map((st, i) => {
          const done = i < active;
          const cur = i === active;
          return (
            <li key={st.key} className={`flex items-center gap-2 text-[12px] ${cur ? "font-semibold text-[var(--text-primary)]" : done ? "text-[var(--text-secondary)]" : "text-[var(--text-secondary)] opacity-60"}`}>
              <span className={`inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] ${done ? "bg-[var(--accent-strong)] text-white" : cur ? "border border-[var(--accent-strong)] text-[var(--accent-strong)]" : "border border-[var(--surface-border)]"}`}>
                {done ? <Check size={10} /> : i + 1}
              </span>
              <span>{st.label}</span>
            </li>
          );
        })}
      </ol>
      <p className="text-[11.5px] leading-snug text-[var(--text-secondary)]">{note}</p>
      {typeof printPrep === "number" && printPrepLabel && (
        <p className="text-[12px] font-semibold text-[var(--accent-strong)]" data-testid="gen-printprep">
          {printPrepLabel} {printPrep}%
        </p>
      )}
      {onCancel && cancelLabel && (
        <button
          type="button"
          onClick={onCancel}
          data-testid="gen-cancel"
          className="mt-0.5 self-start text-[12px] font-semibold text-[var(--text-secondary)] underline underline-offset-2 transition hover:text-[#8f2a20]"
        >
          {cancelLabel}
        </button>
      )}
    </div>
  );
}
