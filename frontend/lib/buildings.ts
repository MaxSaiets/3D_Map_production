// Fetch the building footprint at a clicked point, so the map can OUTLINE the exact
// building the user selected for highlighting (not just drop a dot).
// Uses the backend /api/building-at (same local OSM DuckDB the generator uses) →
// fast AND exactly matches what gets printed. Returns the outline as [lon,lat][] or null.

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type LonLat = [number, number];

/** Building outline (([lon,lat]) ring) at/nearest the point; null if none/failure. */
export async function fetchBuildingAt(lat: number, lon: number): Promise<LonLat[] | null> {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 8000);
    const res = await fetch(`${API_BASE_URL}/api/building-at?lat=${lat}&lon=${lon}`, { signal: ctrl.signal });
    clearTimeout(t);
    if (!res.ok) return null;
    const data = await res.json();
    const fp = data?.footprint;
    if (Array.isArray(fp) && fp.length >= 3) {
      return fp.map((p: [number, number]) => [Number(p[0]), Number(p[1])] as LonLat);
    }
    return null;
  } catch {
    return null;
  }
}
