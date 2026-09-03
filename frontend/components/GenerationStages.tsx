"use client";

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
}: {
  progress: number;
  kind: "map" | "flat";
  title: string;
  note: string;
  stages: { data: string; terrain?: string; detail: string; file: string };
  /** Текст ETA (вже сформатований), напр. «≈ 1 хв». Опційно. */
  eta?: string | null;
}) {
  const p = Math.max(0, Math.min(100, progress || 0));
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

  return (
    <div className="flex flex-col gap-2 rounded-[16px] border border-[var(--surface-border)] bg-white/70 px-3.5 py-3" data-testid="generation-stages">
      <div className="flex items-center justify-between gap-2 text-[14px] font-semibold text-[var(--text-primary)]">
        <span className="inline-flex items-center gap-2"><Loader2 size={16} className="animate-spin text-[var(--accent-strong)]" /> {title}</span>
        <span className="text-[12px] font-semibold text-[var(--text-secondary)]" style={{ fontVariantNumeric: "tabular-nums" }}>{p}%{eta ? ` · ${eta}` : ""}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-[rgba(15,23,42,0.08)]" role="progressbar" aria-label={title} aria-valuemin={0} aria-valuemax={100} aria-valuenow={p}>
        <div className="h-full rounded-full bg-[var(--accent-strong)] transition-all duration-700" style={{ width: `${Math.max(4, p)}%` }} />
      </div>
      <ol className="flex flex-col gap-1" aria-live="polite">
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
    </div>
  );
}
