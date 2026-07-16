"use client";

import Script from "next/script";
import { useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
import { useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import { GA_ID, GADS_ID, GTAG_ON, META_PIXEL_ID, getConsent, setConsent, track, trackPing, isOwnerOptOut, gtagConsentUpdate, clickLabel, campaignParams } from "@/lib/analytics";

// Відомі НЕшкідливі помилки браузера/Firebase — не засмічуємо ними /admin.
// «Connection to Indexed Database server lost» = Firebase Auth persistence у
// Safari/приватному режимі/кількох вкладках; на роботу сайту не впливає.
const BENIGN_ERR = /Indexed Database|IndexedDB|ResizeObserver loop|Script error\.?$|Load failed/i;

/**
 * GDPR-friendly analytics: shows a localized consent banner; Google Analytics
 * is loaded ONLY after the visitor accepts (privacy-first — decline = no GA).
 * Set NEXT_PUBLIC_GA_ID to enable GA; the banner still works without it.
 */
export default function SiteAnalytics() {
  const t = useTranslations("consent");
  const pathname = usePathname();
  const [consent, setConsentState] = useState<"granted" | "denied" | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setConsentState(getConsent());
    setReady(true);
    const onChange = (e: Event) => setConsentState((e as CustomEvent).detail);
    window.addEventListener("mnd:consent", onChange);
    return () => window.removeEventListener("mnd:consent", onChange);
  }, []);

  // Page-view tracking (only fires once consent is granted; track() self-guards).
  // Перший pageview несе кампанія-параметри (utm/gclid/fbclid) з URL приземлення —
  // атрибуція платного трафіку (раніше зливалось у голий google/facebook referrer).
  const firstPvRef = useRef(true);
  useEffect(() => {
    if (consent !== "granted") return;
    if (firstPvRef.current) {
      firstPvRef.current = false;
      const camp = campaignParams();
      track("pageview", Object.keys(camp).length ? camp : undefined);
    } else {
      track("pageview");
    }
  }, [consent, pathname]);

  // Meta Pixel page-view on route change (base pixel + first PageView fire on
  // script load below; this catches subsequent client-side navigations).
  useEffect(() => {
    if (consent !== "granted" || !META_PIXEL_ID) return;
    const fbq = (window as any).fbq;
    if (typeof fbq === "function") fbq("track", "PageView");
  }, [consent, pathname]);

  // Карта кліків: де користувачі тикають (нормовані % в'юпорта) + підпис елемента.
  // Допомагає зрозуміти, що приваблює увагу й де люди «застрягають». Капи на сесію,
  // щоб не роздути лог. track() сам гейтить за згодою/власником/dev.
  useEffect(() => {
    if (consent !== "granted") return;
    let n = 0;
    const onClick = (e: MouseEvent) => {
      if (n >= 50) return; // максимум 50 кліків на завантаження сторінки
      const vw = Math.max(window.innerWidth, 1), vh = Math.max(window.innerHeight, 1);
      const x = Math.round((e.clientX / vw) * 1000) / 10; // % з 0.1 точністю
      const y = Math.round((e.clientY / vh) * 1000) / 10;
      n++;
      track("click", { x, y, el: clickLabel(e.target).slice(0, 48) });
    };
    document.addEventListener("click", onClick, { passive: true, capture: true });
    return () => document.removeEventListener("click", onClick, { capture: true } as any);
  }, [consent, pathname]);

  // Серцебиття присутності: поки вкладка ВИДИМА, раз на 30с шлемо «ping», щоб
  // бекенд рахував РЕАЛЬНИЙ час на сайті (а не лише проміжок між кліками — інакше
  // хто прочитав одну сторінку без кліків показувався б як 0 хв). Пауза, коли
  // вкладку приховано (перемкнули таб) → рахуємо лише активний час. Фінальний ping
  // при відході ловить останній відрізок (тож короткі візити точні й без інтервалу).
  // 30с (а не 15с) удвічі менше роздуває лог; кап 240/завантаження (~2 год).
  // track()-гарди (dev/згода/власник) — всередині trackPing().
  useEffect(() => {
    if (consent !== "granted") return;
    let pings = 0;
    // Гейт БЕЗДІЯЛЬНОСТІ: раніше видима-але-покинута вкладка (юзер відійшов, таб
    // на передньому плані) продовжувала пінгувати ~2 год і роздувала «час на сайті».
    // Тепер пінгуємо лише якщо була реальна активність за останні IDLE_MS.
    const IDLE_MS = 5 * 60 * 1000;
    let lastActivity = Date.now();
    const bump = () => { lastActivity = Date.now(); };
    const acts = ["mousemove", "mousedown", "keydown", "scroll", "touchstart"];
    acts.forEach((e) => window.addEventListener(e, bump, { passive: true }));
    const beat = () => {
      if (document.visibilityState !== "visible") return;
      if (Date.now() - lastActivity > IDLE_MS) return; // видима, але покинута
      if (pings >= 240) return;
      pings++;
      trackPing();
    };
    const id = window.setInterval(beat, 30000);
    const onHidden = () => { if (document.visibilityState === "hidden") trackPing(); };
    const onLeave = () => trackPing();
    document.addEventListener("visibilitychange", onHidden);
    window.addEventListener("pagehide", onLeave);
    return () => {
      window.clearInterval(id);
      acts.forEach((e) => window.removeEventListener(e, bump));
      document.removeEventListener("visibilitychange", onHidden);
      window.removeEventListener("pagehide", onLeave);
    };
  }, [consent]);

  // Google Consent Mode v2: push the consent decision to gtag (Ads/GA4). Default
  // is denied (set in the init script) → conversions modelled cookielessly until accept.
  useEffect(() => {
    if (consent === "granted") gtagConsentUpdate(true);
    else if (consent === "denied") gtagConsentUpdate(false);
  }, [consent]);

  // Free self-hosted error monitoring (Sentry alternative): report uncaught JS
  // errors to /api/track. No PII, throttled. Visible in /admin → topEvents.
  useEffect(() => {
    const API = process.env.NEXT_PUBLIC_API_URL || "";
    let sent = 0;
    const report = (msg: string, src?: string) => {
      if (sent >= 10) return;
      if (isOwnerOptOut()) return;           // не логуємо помилки власника
      if (BENIGN_ERR.test(String(msg))) return; // відомий нешкідливий шум
      sent++;
      try {
        const body = JSON.stringify({ event: "js_error", path: location.pathname, locale: document.documentElement.lang || "", props: { msg: String(msg).slice(0, 200), src: String(src || "").slice(0, 120) } });
        if (navigator.sendBeacon) navigator.sendBeacon(`${API}/api/track`, new Blob([body], { type: "application/json" }));
        else fetch(`${API}/api/track`, { method: "POST", body, headers: { "Content-Type": "application/json" }, keepalive: true }).catch(() => {});
      } catch { /* ignore */ }
    };
    const onErr = (e: ErrorEvent) => report(e.message, `${e.filename}:${e.lineno}`);
    const onRej = (e: PromiseRejectionEvent) => report(`unhandledrejection: ${e.reason}`);
    window.addEventListener("error", onErr);
    window.addEventListener("unhandledrejection", onRej);
    return () => { window.removeEventListener("error", onErr); window.removeEventListener("unhandledrejection", onRej); };
  }, []);

  const decide = (v: "granted" | "denied") => { setConsent(v); setConsentState(v); };

  return (
    <>
      {/* gtag.js (GA4 + Google Ads). Loaded once any Google ID is set. Consent Mode v2:
          default DENIED → cookieless modelled conversions until the visitor accepts. */}
      {GTAG_ON && (
        <>
          <Script src={`https://www.googletagmanager.com/gtag/js?id=${GA_ID || GADS_ID}`} strategy="afterInteractive" />
          <Script id="ga-init" strategy="afterInteractive">
            {`window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());` +
              `gtag('consent','default',{ad_storage:'denied',ad_user_data:'denied',ad_personalization:'denied',analytics_storage:'denied',wait_for_update:500});` +
              // returning visitor, що вже погодився → застосувати granted ОДРАЗУ (синхронно,
              // до config) — інакше update з React-ефекту міг загубитись через гонку завантаження.
              `var _mc=document.cookie.match(/mnd_consent=(granted|denied)/);if(_mc&&_mc[1]==='granted'){gtag('consent','update',{ad_storage:'granted',ad_user_data:'granted',ad_personalization:'granted',analytics_storage:'granted'});}` +
              (GA_ID ? `gtag('config','${GA_ID}',{anonymize_ip:true});` : ``) +
              (GADS_ID ? `gtag('config','${GADS_ID}');` : ``)}
          </Script>
        </>
      )}

      {/* Meta Pixel — loaded ONLY after explicit consent (no cookieless modelling
          equivalent to Google Consent Mode exists for Meta, so this is the privacy-safe
          gate). Dormant until NEXT_PUBLIC_META_PIXEL_ID is set. */}
      {consent === "granted" && META_PIXEL_ID && (
        <Script id="meta-pixel-init" strategy="afterInteractive">
          {`!function(f,b,e,v,n,t,s){if(f.fbq)return;n=f.fbq=function(){n.callMethod?n.callMethod.apply(n,arguments):n.queue.push(arguments)};if(!f._fbq)f._fbq=n;n.push=n;n.loaded=!0;n.version='2.0';n.queue=[];t=b.createElement(e);t.async=!0;t.src=v;s=b.getElementsByTagName(e)[0];s.parentNode.insertBefore(t,s)}(window,document,'script','https://connect.facebook.net/en_US/fbevents.js');` +
            `fbq('init','${META_PIXEL_ID}');fbq('track','PageView');`}
        </Script>
      )}

      {ready && consent === null && (
        <div className="fixed inset-x-3 bottom-[96px] z-[50] mx-auto max-w-[680px] rounded-2xl border border-line bg-paper/95 p-4 shadow-lift backdrop-blur sm:bottom-3 sm:flex sm:items-center sm:gap-4">
          <p className="text-[13px] leading-relaxed text-ink-2">
            {t("text")}{" "}
            <Link href="/privacy" className="font-semibold text-forest underline">{t("more")}</Link>
          </p>
          <div className="mt-3 flex shrink-0 gap-2 sm:mt-0">
            <button
              onClick={() => decide("denied")}
              className="min-h-[44px] flex-1 rounded-full border border-line px-4 text-[13px] font-semibold text-ink-2 transition hover:border-forest/40 sm:flex-none"
            >
              {t("decline")}
            </button>
            <button
              onClick={() => decide("granted")}
              className="min-h-[44px] flex-1 rounded-full bg-forest px-5 text-[13px] font-bold text-[#F4EFE4] transition hover:brightness-110 sm:flex-none"
            >
              {t("accept")}
            </button>
          </div>
        </div>
      )}
    </>
  );
}
