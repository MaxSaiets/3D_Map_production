/**
 * D4 GPX: парсер маршруту в браузері. Повертає [[lon,lat],...] для gpx_track.
 * Підтримує <trkpt> (треки), <rtept> (маршрути) і <wpt> як fallback.
 * Точки проріджуються до MAX_POINTS, щоб запит лишався легким.
 */
export const GPX_MAX_POINTS = 1500;

export interface ParsedGpx {
  points: Array<[number, number]>; // [lon, lat]
  name?: string;
  totalPoints: number; // до проріджування
}

export function parseGpx(xmlText: string): ParsedGpx | null {
  try {
    const doc = new DOMParser().parseFromString(xmlText, "application/xml");
    if (doc.querySelector("parsererror")) return null;
    let nodes = Array.from(doc.querySelectorAll("trkpt"));
    if (nodes.length < 2) nodes = Array.from(doc.querySelectorAll("rtept"));
    if (nodes.length < 2) nodes = Array.from(doc.querySelectorAll("wpt"));
    const raw: Array<[number, number]> = [];
    for (const node of nodes) {
      const lat = parseFloat(node.getAttribute("lat") || "");
      const lon = parseFloat(node.getAttribute("lon") || "");
      if (Number.isFinite(lat) && Number.isFinite(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180) {
        raw.push([lon, lat]);
      }
    }
    if (raw.length < 2) return null;
    // Рівномірне проріджування до ліміту (перша й остання точки зберігаються)
    let points = raw;
    if (raw.length > GPX_MAX_POINTS) {
      const step = (raw.length - 1) / (GPX_MAX_POINTS - 1);
      points = Array.from({ length: GPX_MAX_POINTS }, (_, i) => raw[Math.round(i * step)]);
    }
    const name = doc.querySelector("trk > name, rte > name, metadata > name")?.textContent?.trim() || undefined;
    return { points, name, totalPoints: raw.length };
  } catch {
    return null;
  }
}

/** Bbox треку [west, south, east, north] — для підказки «трек поза зоною». */
export function gpxBounds(points: Array<[number, number]>): [number, number, number, number] | null {
  if (!points.length) return null;
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  for (const [lon, lat] of points) {
    if (lon < w) w = lon;
    if (lon > e) e = lon;
    if (lat < s) s = lat;
    if (lat > n) n = lat;
  }
  return [w, s, e, n];
}
