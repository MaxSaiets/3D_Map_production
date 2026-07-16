// Free, self-hosted analytics — events go to our own backend (/api/track).
// No third party, no cost, data stays on our server. Consent-gated.
// Optional Google stack (off until env IDs are set):
//   NEXT_PUBLIC_GA_ID    = "G-XXXXXXX"  (Google Analytics 4 — measurement)
//   NEXT_PUBLIC_GADS_ID  = "AW-XXXXXXX" (Google Ads — conversion tracking)
//   NEXT_PUBLIC_GADS_LABEL_ORDER|CONTACT|GENERATE = "AW-XXXXXXX/aBcDeF…" (conversion action send_to)
export const GA_ID = process.env.NEXT_PUBLIC_GA_ID;
export const GADS_ID = process.env.NEXT_PUBLIC_GADS_ID;
export const GTAG_ON = Boolean(GA_ID || GADS_ID); // load gtag.js if any Google ID is set

// Optional Meta (Facebook/Instagram) Pixel — off until the owner creates a Meta
// Business/Ads account and sets the ID. NEXT_PUBLIC_META_PIXEL_ID = "1234567890123".
export const META_PIXEL_ID = process.env.NEXT_PUBLIC_META_PIXEL_ID;

export type ConversionAction = "order" | "contact" | "generate" | "lead";
const GADS_LABELS: Record<ConversionAction, string | undefined> = {
  order: process.env.NEXT_PUBLIC_GADS_LABEL_ORDER,
  contact: process.env.NEXT_PUBLIC_GADS_LABEL_CONTACT,
  generate: process.env.NEXT_PUBLIC_GADS_LABEL_GENERATE,
  lead: process.env.NEXT_PUBLIC_GADS_LABEL_CONTACT, // alias
};

const API = process.env.NEXT_PUBLIC_API_URL || "";

export const CONSENT_COOKIE = "mnd_consent";
export const OWNER_COOKIE = "mnd_owner";

/** Власник/адмін опт-аут: коли стоїть cookie mnd_owner=1, ВЛАСНІ заходи не
 *  рахуються — щоб твоє часте тестування /create та /keychains не псувало
 *  статистику реальних відвідувачів. Ставиться автоматично на /admin. */
export function isOwnerOptOut(): boolean {
  if (typeof document === "undefined") return false;
  return /(?:^|;\s*)mnd_owner=1/.test(document.cookie);
}

export function setOwnerOptOut() {
  if (typeof document === "undefined") return;
  document.cookie = `mnd_owner=1;path=/;max-age=${60 * 60 * 24 * 365};samesite=lax`;
}

export function getConsent(): "granted" | "denied" | null {
  if (typeof document === "undefined") return null;
  const m = document.cookie.match(/(?:^|;\s*)mnd_consent=(granted|denied)/);
  return (m?.[1] as "granted" | "denied") ?? null;
}

export function setConsent(value: "granted" | "denied") {
  if (typeof document === "undefined") return;
  document.cookie = `mnd_consent=${value};path=/;max-age=${60 * 60 * 24 * 365};samesite=lax`;
  window.dispatchEvent(new CustomEvent("mnd:consent", { detail: value }));
}

/** Track an event (pageview by default). No-ops without consent. */
/** Кампанія-параметри з URL приземлення (utm_*, gclid, fbclid) — для атрибуції
 *  платного трафіку. Раніше НЕ захоплювались, тож платні кліки Google/Facebook
 *  зливались у голий google.com/facebook.com referrer і кампанію було не розрізнити.
 *  Повертає {} якщо параметрів немає. */
export function campaignParams(): Record<string, string> {
  if (typeof window === "undefined") return {};
  try {
    const q = new URLSearchParams(location.search);
    const out: Record<string, string> = {};
    for (const k of ["utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"]) {
      const v = q.get(k);
      if (v) out[k] = v.slice(0, 120);
    }
    return out;
  } catch { return {}; }
}

export function track(event: string, props?: Record<string, unknown>) {
  if (typeof window === "undefined") return;
  // DEV/PREVIEW guard: не логуємо з localhost/прев'ю — інакше тестування
  // конструкторів (та e2e) роздуває статистику й js_error без коду країни
  // Cloudflare → у адмінці виглядає як «сміттєві» дані без гео.
  const host = location.hostname;
  if (host === "localhost" || host === "127.0.0.1" || host === "0.0.0.0" || host.endsWith(".local")) return;
  if (getConsent() !== "granted") return;
  if (isOwnerOptOut()) return; // не рахуємо власні (адмінські) заходи
  try {
    const body = JSON.stringify({
      event,
      path: location.pathname,
      locale: document.documentElement.lang || "",
      ref: document.referrer || "",
      props: props || undefined,
    });
    const url = `${API}/api/track`;
    if (navigator.sendBeacon) {
      navigator.sendBeacon(url, new Blob([body], { type: "application/json" }));
    } else {
      fetch(url, { method: "POST", body, headers: { "Content-Type": "application/json" }, keepalive: true }).catch(() => {});
    }
  } catch { /* ignore */ }
  const g = (window as any).gtag;
  if (typeof g === "function") g("event", event, props || {});
}

/** Серцебиття присутності: поки вкладка відкрита й видима, шлемо легкий «ping»
 *  у власну аналітику, щоб адмінка рахувала РЕАЛЬНИЙ час на сайті (а не лише
 *  проміжок між кліками). НЕ йде в Google Analytics — щоб не роздувати GA
 *  технічними подіями. Ті ж гарди, що й track(): dev/localhost, згода, власник. */
export function trackPing() {
  if (typeof window === "undefined") return;
  const host = location.hostname;
  if (host === "localhost" || host === "127.0.0.1" || host === "0.0.0.0" || host.endsWith(".local")) return;
  if (getConsent() !== "granted") return;
  if (isOwnerOptOut()) return;
  try {
    const body = JSON.stringify({
      event: "ping",
      path: location.pathname,
      locale: document.documentElement.lang || "",
      ref: document.referrer || "",
    });
    const url = `${API}/api/track`;
    if (navigator.sendBeacon) navigator.sendBeacon(url, new Blob([body], { type: "application/json" }));
    else fetch(url, { method: "POST", body, headers: { "Content-Type": "application/json" }, keepalive: true }).catch(() => {});
  } catch { /* ignore */ }
}

/** Воронка конверсії — ключові кроки шляху покупця. Адмінка показує, де
 *  користувачі «відвалюються» (view → area → generate → order_open → order_submit). */
export const FUNNEL_STEPS = ["view", "area", "generate", "order_open", "order_submit", "paid"] as const;
export type FunnelStep = (typeof FUNNEL_STEPS)[number];

/** Фіксує крок воронки ОДИН раз за сесію (щоб лічильник = к-сть сесій, що дійшли
 *  до кроку, а не к-сть кліків). Кроки 1-в-1 у sessionStorage. */
export function trackFunnel(step: FunnelStep, props?: Record<string, unknown>) {
  if (typeof window === "undefined") return;
  try {
    const k = `mnd_f_${step}`;
    if (sessionStorage.getItem(k)) return;
    sessionStorage.setItem(k, "1");
  } catch { /* sessionStorage може бути недоступний (приватний режим) */ }
  track("funnel", { step, ...(props || {}) });
}

/** Читабельний підпис елемента, по якому клікнули (для теплокарти/топ-кліків). */
export function clickLabel(target: EventTarget | null): string {
  let n = target as HTMLElement | null;
  for (let i = 0; i < 4 && n; i++) {
    const dt = (n as HTMLElement).dataset?.track;
    if (dt) return dt;
    const aria = n.getAttribute?.("aria-label");
    if (aria) return aria;
    const tag = n.tagName?.toLowerCase();
    if (tag === "button" || tag === "a") {
      const txt = (n.innerText || n.textContent || "").trim().replace(/\s+/g, " ");
      if (txt) return txt.slice(0, 40);
      return n.getAttribute("title") || tag;
    }
    n = n.parentElement;
  }
  const el = target as HTMLElement | null;
  return (el?.tagName || "?").toLowerCase();
}

// Meta standard-event mapping (Purchase/Lead have built-in Ads-optimisation value
// on Meta; "generate" has no standard equivalent → sent as a custom event).
const META_STANDARD_EVENT: Record<ConversionAction, string | null> = {
  order: "Purchase",
  contact: "Lead",
  lead: "Lead",
  generate: null,
};

/**
 * Conversion tracking for Google Ads + GA4 + Meta Pixel. Fire on revenue/lead
 * actions (order submitted, contact request, generation finished). Sends:
 *   1) self-hosted /api/track (consent-gated, for /admin stats)
 *   2) a GA4 event (mark as a key event/conversion in the GA4 UI)
 *   3) a Google Ads conversion (`send_to: AW-XXX/label`) so Smart Bidding can optimise.
 *   4) a Meta Pixel event (Purchase/Lead/custom), if NEXT_PUBLIC_META_PIXEL_ID is set.
 * Google Consent Mode v2 handles privacy: with ad_storage denied the conversion is
 * still MODELLED (cookieless), so ad campaigns keep learning even from decliners.
 * Meta has no consent-mode equivalent — the fbq script itself only loads post-consent
 * (see SiteAnalytics.tsx), so no separate gate is needed here.
 */
export function trackConversion(
  action: ConversionAction,
  opts?: { value?: number; currency?: string; transactionId?: string; props?: Record<string, unknown> },
) {
  if (typeof window === "undefined") return;
  const value = opts?.value;
  const currency = opts?.currency || "UAH";
  // (1) self-hosted (own banner consent gates this inside track())
  track(`conv_${action}`, { value, ...(opts?.props || {}) });
  const g = (window as any).gtag;
  if (typeof g === "function") {
    // (2) GA4 key event — name it generate_lead/purchase per GA conventions
    const ga4Event = action === "order" ? "generate_lead" : action === "contact" ? "contact" : action;
    try { g("event", ga4Event, { value, currency, ...(opts?.props || {}) }); } catch { /* noop */ }
    // (3) Google Ads conversion
    const sendTo = GADS_LABELS[action];
    if (sendTo) {
      const payload: Record<string, unknown> = { send_to: sendTo };
      if (value != null) { payload.value = value; payload.currency = currency; }
      if (opts?.transactionId) payload.transaction_id = opts.transactionId;
      try { g("event", "conversion", payload); } catch { /* noop */ }
    }
  }
  // (4) Meta Pixel
  const fbq = (window as any).fbq;
  if (typeof fbq === "function") {
    const std = META_STANDARD_EVENT[action];
    const params: Record<string, unknown> = {};
    if (value != null) { params.value = value; params.currency = currency; }
    try {
      if (std) fbq("track", std, params);
      else fbq("trackCustom", "Generate", params);
    } catch { /* noop */ }
  }
}

/**
 * Google Consent Mode v2 — call BEFORE gtag config (default = denied), then
 * update on banner accept. Keeps us GDPR-compliant while still feeding Ads
 * (denied = cookieless modelled conversions).
 */
export function gtagConsentDefault() {
  const g = (window as any).gtag;
  if (typeof g !== "function") return;
  g("consent", "default", {
    ad_storage: "denied",
    ad_user_data: "denied",
    ad_personalization: "denied",
    analytics_storage: "denied",
    wait_for_update: 500,
  });
}

export function gtagConsentUpdate(granted: boolean) {
  const g = (window as any).gtag;
  if (typeof g !== "function") return;
  const v = granted ? "granted" : "denied";
  g("consent", "update", {
    ad_storage: v,
    ad_user_data: v,
    ad_personalization: v,
    analytics_storage: v,
  });
}
