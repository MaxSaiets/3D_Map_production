"use client";

import { FlaskConical } from "lucide-react";
import { useTranslations } from "next-intl";

/**
 * Смуга «тестовий режим» для експериментальних сервісів (/worlds, /maket).
 *
 * Власник (08.09.2026): «зазнач зверху що це тестові режими тощо щоб зверху було
 * видно завжди». Тому: sticky під шапкою (top-[64px]), а не разовий банер, який
 * зникає після скролу — людина має бачити статус і тоді, коли вже отримала
 * модель і думає замовляти. Закрити не можна свідомо: це не реклама, а
 * попередження про якість результату.
 */
export function BetaBanner({ mode }: { mode: "worlds" | "maket" }) {
  const t = useTranslations("beta");
  return (
    <div
      role="status"
      data-testid="beta-banner"
      className="sticky top-[56px] z-40 border-b border-[rgba(142,107,61,0.28)] bg-[rgba(255,247,230,0.97)] backdrop-blur sm:top-[64px]"
    >
      <div className="mx-auto flex max-w-[1180px] items-start gap-2 px-4 py-2">
        <FlaskConical className="mt-[2px] h-4 w-4 shrink-0 text-[var(--bronze,#8E6B3D)]" />
        <p className="text-[12.5px] leading-snug text-[var(--text-primary,#1c2320)]">
          <b className="font-semibold">{t("title")}</b>{" "}
          <span className="text-[var(--text-secondary,#5a655a)]">{t(mode)}</span>
        </p>
      </div>
    </div>
  );
}
