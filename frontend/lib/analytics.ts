// Free, self-hosted analytics — events go to our own backend (/api/track).
// No third party, no cost, data stays on our server. Consent-gated.
export const GA_ID = process.env.NEXT_PUBLIC_GA_ID; // optional extra; off by default
const API = process.env.NEXT_PUBLIC_API_URL || "";

export const CONSENT_COOKIE = "mnd_consent";

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
export function track(event: string, props?: Record<string, unknown>) {
  if (typeof window === "undefined") return;
  if (getConsent() !== "granted") return;
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
