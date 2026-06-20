// Fetch the building footprint at a clicked/hovered point, so the map can OUTLINE
// the exact building the user selects for highlighting (not just drop a dot).
// Uses the backend /api/building-at (same local OSM DuckDB the generator uses) →
// fast AND exactly matches what gets printed. Returns the outline as [lon,lat][] or null.

// SAME-ORIGIN base ("" → relative): on prod Caddy proxies /api to the backend, so
// the browser hits https://<site>/api/... . Раніше дефолт був "http://localhost:8000",
// тож якщо NEXT_PUBLIC_API_URL не задано на білді — браузер юзера стукав у СВІЙ
// localhost:8000 → fetch падав → будинок НЕ підсвічувався. Relative безпечніший.
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "";

type LonLat = [number, number];

// Кеш результатів (footprint або null) по ~1м-сітці — щоб hover не спамив бек і
// клік по вже-наведеному будинку був миттєвим.
const _cache = new Map<string, LonLat[] | null>();

/** Building outline (([lon,lat]) ring) at/nearest the point; null if none/failure. */
export async function fetchBuildingAt(lat: number, lon: number): Promise<LonLat[] | null> {
  const key = `${lat.toFixed(5)},${lon.toFixed(5)}`;
  if (_cache.has(key)) return _cache.get(key) ?? null;
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 8000);
    const res = await fetch(`${API_BASE_URL}/api/building-at?lat=${lat}&lon=${lon}`, { signal: ctrl.signal });
    clearTimeout(t);
    if (!res.ok) return null; // не кешуємо тимчасові помилки (5xx/мережа)
    const data = await res.json();
    const fp = data?.footprint;
    const result: LonLat[] | null =
      Array.isArray(fp) && fp.length >= 3
        ? fp.map((p: [number, number]) => [Number(p[0]), Number(p[1])] as LonLat)
        : null;
    // Кешуємо лише ОДНОЗНАЧНУ відповідь (200): footprint або «тут будівлі нема».
    if (_cache.size > 600) _cache.clear();
    _cache.set(key, result);
    return result;
  } catch {
    return null;
  }
}
