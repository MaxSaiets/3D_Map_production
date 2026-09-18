"use client";

import { useState } from "react";
import { Sparkles, Check, AlertTriangle, HelpCircle, Loader2 } from "lucide-react";
import { useTranslations } from "next-intl";
import type { AgentAnswer } from "@/lib/api";

/**
 * Агент режиму: вільний опис → «як я зрозумів» → людина підтверджує → форма заповнюється.
 *
 * Принцип «без помилок» (власник, 18.09.2026): агент ніколи не запускає генерацію сам і не
 * підміняє вибір мовчки. Він показує розібраний план рядками, попередження (що обмежив і чому)
 * та питання (чого бракує), і лише кнопка «Застосувати» переносить план у форму. Далі людина
 * бачить усі значення в контролах і сама тисне «Згенерувати».
 */
export function AgentBox<T>({
  ask, onApply, placeholder, examples, testId = "agent",
}: {
  ask: (text: string) => Promise<AgentAnswer<T>>;
  onApply: (spec: T) => void;
  placeholder: string;
  examples?: string[];
  testId?: string;
}) {
  const t = useTranslations("agent");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [answer, setAnswer] = useState<AgentAnswer<T> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [applied, setApplied] = useState(false);

  const run = async () => {
    if (!text.trim() || busy) return;
    setBusy(true); setError(null); setApplied(false);
    try { setAnswer(await ask(text.trim())); }
    catch (e: any) { setError(e?.response?.data?.detail || e?.message || t("failed")); }
    finally { setBusy(false); }
  };

  return (
    <div className="rounded-2xl border border-[rgba(15,118,110,0.25)] bg-[rgba(15,118,110,0.04)] p-3" data-testid={`${testId}-box`}>
      <div className="flex items-center gap-1.5 text-[13px] font-semibold text-[var(--text-primary)]">
        <Sparkles size={15} className="text-[var(--accent-strong)]" /> {t("title")}
      </div>
      <p className="mt-0.5 text-[11.5px] leading-snug text-[var(--text-secondary)]">{t("hint")}</p>
      <textarea
        value={text} onChange={(e) => setText(e.target.value)} rows={2} maxLength={2000} placeholder={placeholder}
        data-testid={`${testId}-input`}
        onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(); }}
        className="mt-2 w-full resize-none rounded-xl border border-[var(--surface-border)] bg-white/90 px-3 py-2 text-[13.5px] text-[var(--text-primary)] outline-none focus:border-[var(--accent-strong)]"
      />
      {examples && examples.length > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {examples.map((ex) => (
            <button key={ex} type="button" onClick={() => setText(ex)}
              className="rounded-full border border-[var(--surface-border)] bg-white/80 px-2.5 py-1 text-[11.5px] text-[var(--text-secondary)] transition hover:border-[var(--accent-strong)] hover:text-[var(--text-primary)]">
              {ex}
            </button>
          ))}
        </div>
      )}
      <button type="button" onClick={run} disabled={busy || !text.trim()} data-testid={`${testId}-run`}
        className="mt-2 inline-flex min-h-10 items-center gap-1.5 rounded-full bg-[var(--accent-strong)] px-4 text-[13px] font-semibold text-white transition disabled:opacity-50">
        {busy ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />} {busy ? t("thinking") : t("understand")}
      </button>
      {error && <p role="alert" className="mt-2 text-[12.5px] text-red-600">{error}</p>}

      {answer && (
        <div className="mt-3 rounded-xl border border-[var(--surface-border)] bg-white p-3" data-testid={`${testId}-answer`}>
          <div className="flex items-center justify-between">
            <span className="text-[12.5px] font-semibold text-[var(--text-primary)]">{t("understood")}</span>
            <span className="text-[11px] text-[var(--text-secondary)]">
              {t("confidence", { pct: Math.round(answer.confidence * 100) })} · {answer.source === "llm" ? "AI" : answer.source === "user" ? t("srcUser") : t("srcRules")}
            </span>
          </div>
          <ul className="mt-1.5 space-y-1">
            {answer.understood.map((line, i) => (
              <li key={i} className="flex gap-1.5 text-[12.5px] leading-snug text-[var(--text-primary)]">
                <Check size={13} className="mt-[3px] shrink-0 text-[var(--accent-strong)]" /> <span>{line}</span>
              </li>
            ))}
          </ul>
          {answer.warnings.length > 0 && (
            <ul className="mt-2 space-y-1">
              {answer.warnings.map((w, i) => (
                <li key={i} className="flex gap-1.5 text-[12px] leading-snug text-[#8a5a00]">
                  <AlertTriangle size={13} className="mt-[2px] shrink-0" /> <span>{w}</span>
                </li>
              ))}
            </ul>
          )}
          {answer.questions.length > 0 && (
            <ul className="mt-2 space-y-1">
              {answer.questions.map((q, i) => (
                <li key={i} className="flex gap-1.5 text-[12px] leading-snug text-[var(--text-primary)]">
                  <HelpCircle size={13} className="mt-[2px] shrink-0 text-[var(--accent-strong)]" /> <span>{q}</span>
                </li>
              ))}
            </ul>
          )}
          <div className="mt-2.5 flex flex-wrap items-center gap-2">
            <button type="button" data-testid={`${testId}-apply`} onClick={() => { onApply(answer.spec); setApplied(true); }}
              className="inline-flex min-h-9 items-center gap-1.5 rounded-full border border-[var(--accent-strong)] px-3.5 text-[12.5px] font-semibold text-[var(--accent-strong)] transition hover:bg-[rgba(15,118,110,0.08)]">
              <Check size={13} /> {applied ? t("applied") : t("apply")}
            </button>
            <span className="text-[11.5px] text-[var(--text-secondary)]">{t("applyHint")}</span>
          </div>
        </div>
      )}
    </div>
  );
}
