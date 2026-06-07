import { getRequestConfig } from "next-intl/server";
import { routing } from "./routing";

// Deep-merge helper: values from `over` win, but any key missing in `over`
// falls back to `base`. Lets you add a key to the default locale (uk) only and
// have every other locale keep working (showing the uk text) until translated.
function deepMerge(base: any, over: any): any {
  if (Array.isArray(base) || typeof base !== "object" || base === null) return over ?? base;
  const out: any = { ...base };
  for (const k of Object.keys(over ?? {})) {
    out[k] = k in base ? deepMerge(base[k], over[k]) : over[k];
  }
  return out;
}

export default getRequestConfig(async ({ requestLocale }) => {
  const requested = await requestLocale;
  const locale =
    requested && (routing.locales as readonly string[]).includes(requested)
      ? requested
      : routing.defaultLocale;

  const base = (await import(`../messages/${routing.defaultLocale}.json`)).default;
  const messages =
    locale === routing.defaultLocale
      ? base
      : deepMerge(base, (await import(`../messages/${locale}.json`)).default);

  return {
    locale,
    messages,
    // Don't crash on a missing key — fall back to the key path (already covered
    // by the merge above, this is just a safety net for hard-coded namespaces).
    getMessageFallback: ({ key, namespace }) => (namespace ? `${namespace}.${key}` : key),
    onError() {},
  };
});
