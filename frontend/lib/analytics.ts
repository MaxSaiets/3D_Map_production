// Lightweight analytics helper. No-ops until GA is loaded (after cookie consent).
export const GA_ID = process.env.NEXT_PUBLIC_GA_ID;

/** Track a conversion / interaction event (e.g. track("generate_map")). */
export function track(name: string, params?: Record<string, unknown>) {
  if (typeof window === "undefined") return;
  const g = (window as any).gtag;
  if (typeof g === "function") g("event", name, params || {});
}

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
