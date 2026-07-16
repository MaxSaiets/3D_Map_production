// Per-user city grid API (server-side, requires login). Lets a user save a tiled
// city grid to history and later reopen it to generate neighbouring cells.
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export interface GridCell {
  row: number;
  col: number;
  task_id?: string;
  download_url?: string;
  [k: string]: any;
}

export interface CityGrid {
  id?: string;
  name?: string;
  city?: string;
  center?: [number, number] | null;
  grid_type?: "hexagonal" | "square" | "circle";
  hex_size_m?: number;
  bounds?: { north: number; south: number; east: number; west: number } | null;
  rotation_deg?: number;
  cells?: GridCell[];
  created_at?: number;
  updated_at?: number;
}

function authHeaders(token: string | null): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}`, "Content-Type": "application/json" }
               : { "Content-Type": "application/json" };
}

export async function listGrids(token: string | null): Promise<CityGrid[]> {
  const r = await fetch(`${API_BASE}/api/account/grids`, { headers: authHeaders(token) });
  if (!r.ok) return [];
  const d = await r.json();
  return d.grids || [];
}

export async function saveGrid(token: string | null, grid: CityGrid): Promise<CityGrid | null> {
  const r = await fetch(`${API_BASE}/api/account/grids`, {
    method: "POST", headers: authHeaders(token), body: JSON.stringify(grid),
  });
  if (!r.ok) return null;
  const d = await r.json();
  return d.grid || null;
}

export async function getGrid(token: string | null, id: string): Promise<CityGrid | null> {
  const r = await fetch(`${API_BASE}/api/account/grids/${id}`, { headers: authHeaders(token) });
  if (!r.ok) return null;
  const d = await r.json();
  return d.grid || null;
}

export async function deleteGrid(token: string | null, id: string): Promise<boolean> {
  const r = await fetch(`${API_BASE}/api/account/grids/${id}`, {
    method: "DELETE", headers: authHeaders(token),
  });
  return r.ok;
}

