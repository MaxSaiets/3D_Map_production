/**
 * Орієнтовна ціна з бекенда (/api/quote) для sticky-бара і форми замовлення.
 * Кеш у памʼяті на сесію; при недоступному API повертає null — UI показує
 * статичний fallback (i18n estPrice*), нічого не ламається.
 */
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export type Quote = { currency: string; price: number; formatted: string; approx: boolean };

const cache = new Map<string, Quote | null>();

export async function fetchQuote(
  product: "map" | "keychain",
  sizeMm?: number,
  relief?: boolean,
): Promise<Quote | null> {
  const key = `${product}|${sizeMm ?? ""}|${relief ? 1 : 0}`;
  if (cache.has(key)) return cache.get(key) ?? null;
  try {
    const params = new URLSearchParams({ product });
    if (sizeMm) params.set("size_mm", String(sizeMm));
    if (relief) params.set("relief", "1");
    const res = await fetch(`${API_BASE}/api/quote?${params}`, { cache: "no-store" });
    if (!res.ok) { cache.set(key, null); return null; }
    const q: Quote = await res.json();
    cache.set(key, q);
    return q;
  } catch {
    cache.set(key, null);
    return null;
  }
}
