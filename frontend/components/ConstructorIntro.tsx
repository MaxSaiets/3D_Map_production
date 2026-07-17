"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { ArrowDown, X } from "lucide-react";

/**
 * Вступний блок НАД конструктором: «ось що вийде» + 3 кроки + CTA.
 *
 * НАВІЩО: дані воронки (17.07) показали найбільший обвал на ПЕРШОМУ кроці —
 * view 7 → area 2 (71% йдуть, не виділивши зону). Людина потрапляла одразу на
 * карту Києва й не розуміла, що саме вона отримає. Блок показує РЕАЛЬНІ фото
 * друків (public/real/, не рендери) до того, як просити щось робити.
 *
 * Метрика успіху: співвідношення area/view в /admin має зрости від ~29%.
 *
 * ПОВЕДІНКА: показується лише до першої взаємодії (localStorage). Той, хто вже
 * тут був, одразу бачить інструмент — блок не має ставати перепоною для своїх.
 */
/**
 * Стан вступного блоку. Живе в СТОРІНЦІ, а не всередині компонента, бо його
 * має бачити й OnboardingTour: інакше новачок отримує ДВА онбординги нараз
 * (блок + плаваюча підказка) — саме той зайвий chrome, який ми прибирали.
 * Порядок: спершу «що вийде», і лише коли блок закрито — підказки по кроках.
 */
export function useIntroGate(storageKey: string) {
  // Показуємо за замовчуванням: ~весь трафік — нові відвідувачі, тож
  // оптимізуємо саме під них (без миготіння). Хто вже дивився — блок
  // ховається одразу після монтування.
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    try {
      if (localStorage.getItem(storageKey)) setVisible(false);
    } catch { /* приватний режим — лишаємо видимим */ }
  }, [storageKey]);

  const dismiss = () => {
    setVisible(false);
    try { localStorage.setItem(storageKey, "1"); } catch { /* ignore */ }
  };

  return { introVisible: visible, dismissIntro: dismiss };
}

export function ConstructorIntro({
  visible,
  onDismiss,
  variant,
  photos,
  scrollToId,
  priceFrom,
}: {
  visible: boolean;
  onDismiss: () => void;
  /** Ключ у namespace `intro` — визначає тексти (мапа чи брелок). */
  variant: "map" | "keychain";
  /** Імена файлів у /public/real/ без розширення. */
  photos: string[];
  /** id елемента, до якого скролимо по CTA (#panel-map / #kc-map). */
  scrollToId: string;
  priceFrom: number;
}) {
  const t = useTranslations(`intro.${variant}`);
  const tc = useTranslations("intro");

  if (!visible) return null;

  const dismiss = (reason: "cta" | "skip") => {
    onDismiss();
    import("@/lib/analytics")
      .then((m) => m.track("click", { id: `intro_${reason}`, variant }))
      .catch(() => {});
  };

  const start = () => {
    dismiss("cta");
    // Даємо React прибрати блок, потім скролимо до карти.
    requestAnimationFrame(() => {
      document.getElementById(scrollToId)?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  };

  const steps = [1, 2, 3] as const;

  return (
    <section
      className="mt-3 overflow-hidden rounded-[22px] border border-line bg-paper px-4 py-4 shadow-soft sm:px-6 sm:py-5"
      aria-labelledby="intro-title"
    >
      <div className="flex items-start justify-between gap-3">
        <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-ink-3">
          {t("eyebrow")}
        </span>
        <button
          onClick={() => dismiss("skip")}
          className="inline-flex min-h-[32px] shrink-0 items-center gap-1 rounded-full border border-line px-3 text-[12px] font-medium text-ink-3 transition hover:border-forest/40 hover:text-ink"
        >
          {tc("skip")} <X size={12} aria-hidden="true" />
        </button>
      </div>

      <h2 id="intro-title" className="mt-2 font-serif text-[clamp(22px,3vw,30px)] leading-tight text-ink">
        {t("title")}
      </h2>
      <p className="mt-1.5 max-w-[56ch] text-[14px] leading-relaxed text-ink-2">{t("lead")}</p>

      {/* Реальні фото друків — головний доказ.
          МОБІЛЬНИЙ: лише 2 фото — при 4 блок виростав до 944px (екран 812) і CTA
          опинявся ЗА межами першого екрана, тобто ламав те, що мав полагодити.
          `-sm` = 720px-варіанти: повні файли 1100px важили 616КБ на четвірку, а
          слот тут ~284px — учетверо зайве. Тепер 288КБ.
          eager для ВСІХ: фото над згином, і з lazy браузер їх не запитував
          узагалі — блок показував порожні рамки замість доказу. */}
      <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        {photos.map((f, i) => (
          <div
            key={f}
            className={`overflow-hidden rounded-[12px] border border-line-soft bg-bg-2 ${i > 1 ? "hidden sm:block" : ""}`}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`/real/${f}-sm.webp`}
              alt={t("alt")}
              width={720}
              height={540}
              loading="eager"
              fetchPriority={i === 0 ? "high" : "auto"}
              decoding="async"
              /* sm+: 4:3 замість квадрата — на ноуті 1280×720 квадратні тайли по
                 300px виштовхували CTA за межі першого екрана. */
              className="aspect-square h-full w-full object-cover sm:aspect-[4/3]"
            />
          </div>
        ))}
      </div>

      {/* Мобільний: крок = компактний РЯДОК (номер ліворуч). sm+: картки колонками. */}
      <ol className="mt-3 grid gap-2 sm:mt-4 sm:grid-cols-3">
        {steps.map((n) => (
          <li
            key={n}
            className="flex items-start gap-2.5 rounded-[12px] border border-line-soft bg-paper-2 p-2.5 sm:block sm:p-3"
          >
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-forest text-[12px] font-semibold text-[#F4EFE4] sm:mb-2">
              {n}
            </span>
            <div className="min-w-0">
              <div className="text-[13px] font-semibold text-ink">{t(`s${n}t`)}</div>
              <div className="mt-0.5 text-[12px] leading-relaxed text-ink-3">{t(`s${n}b`)}</div>
            </div>
          </li>
        ))}
      </ol>

      <div className="mt-5 flex flex-wrap items-center gap-3">
        <button
          onClick={start}
          className="inline-flex min-h-[48px] items-center gap-2 rounded-full bg-forest px-6 text-[15px] font-semibold text-white transition hover:brightness-110"
        >
          {t("cta")} <ArrowDown size={16} aria-hidden="true" />
        </button>
        <span className="text-[13px] text-ink-2">
          {tc("priceFrom", { price: priceFrom })}
        </span>
      </div>
    </section>
  );
}
