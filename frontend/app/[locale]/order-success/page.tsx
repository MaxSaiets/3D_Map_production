"use client";
export const dynamic = "force-dynamic";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { CheckCircle2, Loader2, Clock, ArrowRight, Home, LayoutGrid } from "lucide-react";
import { track } from "@/lib/analytics";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

type PayState = "checking" | "paid" | "pending" | "unknown";

export default function OrderSuccessPage() {
  const t = useTranslations("orderSuccess");
  const [order, setOrder] = useState<string>("");
  const [state, setState] = useState<PayState>("checking");

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const ord = (params.get("order") || "").trim();
    setOrder(ord);
    track("view", { id: "order_success", order: ord });
    if (!ord) { setState("unknown"); return; }

    let cancelled = false;
    let tries = 0;
    // Опитуємо серверну перевірку статусу LiqPay кілька разів: платіж міг щойно
    // завершитись, а статус в API зʼявляється з невеликою затримкою.
    const poll = async () => {
      tries += 1;
      try {
        const r = await fetch(`${API_BASE}/api/liqpay/status/${encodeURIComponent(ord)}`, { cache: "no-store" });
        const d = await r.json();
        if (cancelled) return;
        if (d?.paid) { setState("paid"); track("order_paid_confirmed", { order: ord }); return; }
        if (tries < 5) { setTimeout(poll, 2000); return; }
        setState(d?.configured === false ? "unknown" : "pending");
      } catch {
        if (cancelled) return;
        if (tries < 5) { setTimeout(poll, 2000); return; }
        setState("pending");
      }
    };
    poll();
    return () => { cancelled = true; };
  }, []);

  const isPaid = state === "paid";
  const isChecking = state === "checking";

  return (
    <div className="mx-auto flex min-h-[100dvh] max-w-[720px] flex-col items-center px-5 py-16 text-center lg:py-24">
      <div className="flex h-20 w-20 items-center justify-center rounded-full" style={{
        background: isPaid ? "color-mix(in srgb, var(--forest,#2F4A3C) 12%, transparent)"
          : "color-mix(in srgb, var(--bronze,#8E6B3D) 12%, transparent)",
      }}>
        {isChecking ? <Loader2 className="h-9 w-9 animate-spin text-[var(--bronze,#8E6B3D)]" />
          : isPaid ? <CheckCircle2 className="h-10 w-10 text-[var(--forest,#2F4A3C)]" />
          : <Clock className="h-9 w-9 text-[var(--bronze,#8E6B3D)]" />}
      </div>

      <h1 className="mt-6 font-serif text-[clamp(28px,5vw,44px)] leading-tight text-ink">
        {isChecking ? t("checkingTitle") : isPaid ? t("paidTitle") : t("pendingTitle")}
      </h1>

      {order && (
        <p className="mt-2 text-[15px] text-ink-2">
          {t("orderLabel")} <span className="font-semibold text-ink">#{order}</span>
        </p>
      )}

      <p className="mt-4 max-w-[520px] text-[15px] leading-relaxed text-ink-2">
        {isChecking ? t("checkingBody") : isPaid ? t("paidBody") : t("pendingBody")}
      </p>

      {isPaid && (
        <ul className="mt-6 flex max-w-[460px] flex-col gap-2 text-left">
          {[t("step1"), t("step2"), t("step3")].map((s, i) => (
            <li key={i} className="flex items-start gap-2.5 text-[14px] text-ink-2">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-[var(--forest,#2F4A3C)]" />
              <span>{s}</span>
            </li>
          ))}
        </ul>
      )}

      <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
        <Link href="/account" className="inline-flex items-center gap-2 rounded-full bg-[var(--forest,#2F4A3C)] px-6 py-3 text-sm font-bold text-white hover:opacity-90">
          {t("toAccount")} <ArrowRight className="h-4 w-4" />
        </Link>
        <Link href="/create" className="inline-flex items-center gap-2 rounded-full border border-line px-5 py-3 text-sm font-semibold text-ink-2 hover:bg-bg-2">
          <LayoutGrid className="h-4 w-4" /> {t("makeAnother")}
        </Link>
        <Link href="/" className="inline-flex items-center gap-2 rounded-full border border-line px-5 py-3 text-sm font-semibold text-ink-2 hover:bg-bg-2">
          <Home className="h-4 w-4" /> {t("toHome")}
        </Link>
      </div>

      <p className="mt-8 text-[13px] text-ink-3">{t("support")}</p>
    </div>
  );
}
