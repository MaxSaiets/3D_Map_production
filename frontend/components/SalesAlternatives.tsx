"use client";

import { useEffect, useRef, useState } from "react";
import { MessageCircle, Instagram, Send } from "lucide-react";
import { useTranslations, useLocale } from "next-intl";

// S-1/S-2 (2026-09-07, «немає продажів»): за 45 днів 28 людей згенерували модель,
// 4 відкрили форму, 0 надіслали. Форма — не єдиний шлях: український покупець часто
// пише в месенджер (власник у біо TikTok сам каже «пиши в Дірект»). Тут:
//   1) «Зручніше в месенджері?» — Telegram / Instagram із заздалегідь скопійованим
//      текстом (продукт · розмір · місце · лінк на 3D), бо t.me не вміє prefill для
//      звичайних акаунтів — копіюємо в буфер і відкриваємо чат;
//   2) мікро-опитування «що заважає замовити» — показується ПІСЛЯ кліку «Завантажити»
//      або через 30 с на екрані «Готово» без кліку «Замовити»; одна відповідь на задачу;
//      відповіді летять у /api/track → адмінка «Чому не замовляють» і тижневий дайджест;
//   3) для не-uk локалей — чесний рядок «друк і доставка лише по Україні».
// Жодного стану зверху: клік «Замовити»/«Завантажити» ловимо за window-подіями,
// які вже шлють обидва guided-флоу.

const TG_URL = "https://t.me/monadruk";
const IG_URL = "https://ig.me/m/monadruk";
const SURVEY_DELAY_MS = 30_000;
const REASONS = ["price", "self", "look", "abroad", "other"] as const;

export function SalesAlternatives({
  product,
  taskId,
  summary,
  priceUah,
}: {
  product: "map" | "keychain";
  taskId: string | null | undefined;
  /** Людський рекап для тексту в месенджер: «3D-мапа · M · Київ». */
  summary: string;
  priceUah: number;
}) {
  const t = useTranslations("scenario");
  const locale = useLocale();
  const [copied, setCopied] = useState<"tg" | "ig" | null>(null);
  const [surveyOpen, setSurveyOpen] = useState(false);
  const [answered, setAnswered] = useState<(typeof REASONS)[number] | true | null>(null);
  const orderClickedRef = useRef(false);
  const surveyKey = `mnd_why_${taskId || "na"}`;

  // Опитування: після «Завантажити» одразу, інакше через 30 с без «Замовити».
  useEffect(() => {
    if (!taskId) return;
    try {
      const saved = localStorage.getItem(surveyKey);
      if (saved) {
        // Зберігали саму причину — відновлюємо персональну відповідь, а не
        // загальне «дякуємо»: людина вже бачила її і має побачити ту саму.
        setAnswered((REASONS as readonly string[]).includes(saved) ? (saved as (typeof REASONS)[number]) : true);
        setSurveyOpen(true);
        return;
      }
    } catch { /* приватний режим */ }
    const onOrder = () => { orderClickedRef.current = true; };
    const onDownload = () => setSurveyOpen(true);
    window.addEventListener("monadruk:open-order", onOrder);
    window.addEventListener("monadruk:kc-guided-order", onOrder);
    window.addEventListener("monadruk:guided-download", onDownload);
    window.addEventListener("monadruk:kc-guided-download", onDownload);
    const timer = window.setTimeout(() => { if (!orderClickedRef.current) setSurveyOpen(true); }, SURVEY_DELAY_MS);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("monadruk:open-order", onOrder);
      window.removeEventListener("monadruk:kc-guided-order", onOrder);
      window.removeEventListener("monadruk:guided-download", onDownload);
      window.removeEventListener("monadruk:kc-guided-download", onDownload);
    };
  }, [taskId, surveyKey]);

  const prefill = () => {
    const link = taskId && typeof window !== "undefined" ? `${window.location.origin}/share/${taskId}` : "";
    return t("msgPrefill", { summary, price: priceUah, link });
  };

  const openMessenger = async (channel: "tg" | "ig") => {
    // taskId + рекап їдуть у подію навмисно: бекенд одразу шле власнику
    // сповіщення про гарячий лід, і без них воно було б непридатне до дії —
    // «хтось щось хотів» замість «мапа M, Київ, 770 ₴, ось модель».
    import("@/lib/analytics")
      .then((m) => m.track("messenger_order", { channel, product, priceUah, taskId: taskId || "", summary }))
      .catch(() => {});
    const text = prefill();
    try { await navigator.clipboard.writeText(text); setCopied(channel); } catch { setCopied(null); }
    // Відкриваємо ПІСЛЯ копіювання — інакше Safari губить дозвіл на буфер.
    window.open(channel === "tg" ? TG_URL : IG_URL, "_blank", "noopener");
  };

  const answer = (reason: (typeof REASONS)[number]) => {
    import("@/lib/analytics").then((m) => m.track("why_not_order", { reason, product, locale })).catch(() => {});
    try { localStorage.setItem(surveyKey, reason); } catch { /* ignore */ }
    setAnswered(reason);
  };

  return (
    <div className="flex flex-col gap-2" data-testid="sales-alternatives">
      {locale !== "uk" && (
        <p className="rounded-lg border border-[var(--surface-border)] bg-[var(--surface-muted,rgba(0,0,0,0.03))] px-2.5 py-1.5 text-[11px] leading-snug text-[var(--text-secondary)]" data-testid="ua-only-note">
          {t("uaOnly")}
        </p>
      )}
      <div className="flex items-center gap-2">
        <span className="h-px flex-1 bg-[var(--surface-border)]" />
        <span className="text-[10.5px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">{t("msgOr")}</span>
        <span className="h-px flex-1 bg-[var(--surface-border)]" />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <button
          type="button"
          onClick={() => openMessenger("tg")}
          data-testid="msg-telegram"
          className="inline-flex h-10 items-center justify-center gap-1.5 rounded-full border border-[var(--surface-border)] bg-[var(--surface-panel,#fff)] px-3 text-[12.5px] font-semibold text-[var(--text-primary)] transition hover:border-[var(--accent-strong)]"
        >
          <Send size={14} className="text-[#2AABEE]" /> {t("msgTelegram")}
        </button>
        <button
          type="button"
          onClick={() => openMessenger("ig")}
          data-testid="msg-instagram"
          className="inline-flex h-10 items-center justify-center gap-1.5 rounded-full border border-[var(--surface-border)] bg-[var(--surface-panel,#fff)] px-3 text-[12.5px] font-semibold text-[var(--text-primary)] transition hover:border-[var(--accent-strong)]"
        >
          <Instagram size={14} className="text-[#E1306C]" /> {t("msgInstagram")}
        </button>
      </div>
      <p className="text-center text-[11px] leading-snug text-[var(--text-secondary)]" aria-live="polite">
        {copied ? t("msgCopied") : t("msgHint")}
      </p>
      {surveyOpen && !answered && (
        <div className="mt-1 rounded-xl border border-[var(--surface-border)] bg-[var(--surface-panel,#fff)] p-2.5" data-testid="why-not-order">
          <p className="mb-1.5 flex items-center gap-1.5 text-[12px] font-semibold text-[var(--text-primary)]">
            <MessageCircle size={13} className="text-[var(--accent-strong)]" /> {t("whyTitle")}
          </p>
          <div className="flex flex-wrap gap-1.5">
            {REASONS.map((r) => (
              <button
                key={r}
                type="button"
                onClick={() => answer(r)}
                data-testid={`why-${r}`}
                className="rounded-full border border-[var(--surface-border)] px-2.5 py-1 text-[11.5px] text-[var(--text-primary)] transition hover:border-[var(--accent-strong)] hover:bg-[var(--surface-muted,rgba(0,0,0,0.03))]"
              >
                {t(`why_${r}`)}
              </button>
            ))}
          </div>
        </div>
      )}
      {surveyOpen && answered && (
        // ⭐09.09.2026: раніше тут був глухий кут — «Дякуємо» і все. Це єдиний
        // прямий зворотний звʼязок, який сайт збирає, і людина, яка щойно
        // пояснила, ЧОМУ не замовляє, заслуговує на відповідь по суті:
        // «надрукую сам» → файл уже готовий, «дорого» → що входить у ціну,
        // «не в Україні» → куди возимо. Без тиску: вона вже сказала «ні».
        <p
          className="text-center text-[11px] leading-snug text-[var(--text-secondary)]"
          data-testid={answered === true ? "why-thanks" : `why-reply-${answered}`}
        >
          {answered === true ? t("whyThanks") : t(`whyReply_${answered}`)}
        </p>
      )}
    </div>
  );
}
