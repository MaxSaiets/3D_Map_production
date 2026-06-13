/**
 * Геокодер на Nominatim (OpenStreetMap, безкоштовно, без ключа). Дозволяє
 * знайти БУДЬ-ЯКЕ місто / село / адресу — не лише 23 преднабори. Закриває
 * сценарій «мого міста нема у списку» (найбільша функціональна діра конструктора).
 *
 * Політика Nominatim: ≤1 запит/сек з браузера — тому ввід дебаунситься у UI.
 */
export interface GeoResult {
  lat: number;
  lon: number;
  label: string;      // коротка назва для списку
  full: string;       // повна адреса (display_name)
}

export async function geocodeSearch(query: string, signal?: AbortSignal): Promise<GeoResult[]> {
  const q = query.trim();
  if (q.length < 3) return [];
  try {
    const url =
      "https://nominatim.openstreetmap.org/search?format=jsonv2&limit=6&accept-language=uk&q=" +
      encodeURIComponent(q);
    const r = await fetch(url, { signal, headers: { Accept: "application/json" } });
    if (!r.ok) return [];
    const data = (await r.json()) as Array<{
      lat: string; lon: string; name?: string; display_name: string;
    }>;
    return data
      .map((d) => {
        const lat = parseFloat(d.lat);
        const lon = parseFloat(d.lon);
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
        const full = d.display_name || d.name || "";
        const label = d.name || full.split(",").slice(0, 2).join(", ") || full;
        return { lat, lon, label, full } as GeoResult;
      })
      .filter(Boolean) as GeoResult[];
  } catch {
    return [];
  }
}

/** Зворотний геокод для кнопки «📍 Я тут» — назва місця за координатами. */
export async function reverseGeocode(lat: number, lon: number): Promise<string | null> {
  try {
    const url =
      "https://nominatim.openstreetmap.org/reverse?format=jsonv2&accept-language=uk&lat=" +
      lat + "&lon=" + lon;
    const r = await fetch(url, { headers: { Accept: "application/json" } });
    if (!r.ok) return null;
    const d = (await r.json()) as { display_name?: string; name?: string };
    return d.name || (d.display_name ? d.display_name.split(",").slice(0, 2).join(", ") : null);
  } catch {
    return null;
  }
}
