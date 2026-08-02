import axios from "axios";
import { API_BASE_URL } from "./api";

/**
 * Клієнт сервісу «макет квартири»: план приміщення → друкована 3D-модель.
 *
 * ВАЖЛИВО ПРО ОДИНИЦІ. У плані, який повертає /analyze і приймає /generate,
 * координати та товщини — у ПІКСЕЛЯХ прев'ю-зображення, попри суфікси `_m` у
 * назвах полів (бекенд свідомо перевикористовує одну структуру до і після
 * застосування масштабу). У метри це переводить сам бекенд, множачи на
 * `m_per_px` у момент побудови. Тому редактор рахує все в пікселях картинки —
 * і жодних перетворень у браузері не потрібно.
 */

export interface FpWall {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  /** у ПІКСЕЛЯХ прев'ю (див. коментар вище) */
  thickness_m: number;
  bearing?: boolean;
}

export type FpOpeningKind = "door" | "window" | "arch";

export interface FpOpening {
  /** індекс у масиві walls */
  wall: number;
  /** 0..1 уздовж центральної лінії стіни */
  center_t: number;
  /** у ПІКСЕЛЯХ прев'ю */
  width_m: number;
  kind: FpOpeningKind;
  sill_m: number;
  height_m: number;
}

export interface FpPlan {
  walls: FpWall[];
  openings: FpOpening[];
  rooms: Array<{ polygon: number[][]; name: string; area_m2: number }>;
  wall_height_m: number;
  scale_source: string;
  m_per_px: number;
  image_size_px: [number, number] | null;
  confidence: number;
  notes: string[];
}

export interface FpScaleCandidate {
  m_per_px: number;
  source: "reference" | "ocr" | "pdf" | "door" | "assumed" | string;
  confidence: number;
  detail: string;
}

export interface FpAnalyzeResponse {
  plan: FpPlan;
  scale: FpScaleCandidate & { candidates: FpScaleCandidate[]; ocr: Array<Record<string, unknown>> };
  preview: string;
  image_size_px: [number, number];
  detector: "nn" | "cv";
  confidence: number;
  notes: string[];
  warnings: string[];
  timings_ms: Record<string, number>;
  estimate: {
    plan_width_m: number;
    plan_height_m: number;
    area_m2: number;
    sheet_width_m: number;
    /** Площа приміщень у пікселях² — з неї рахується масштаб за площею з договору. */
    interior_px2: number;
  };
}

export interface FpGenerateRequest {
  plan: FpPlan;
  m_per_px: number;
  model_size_mm: number;
  wall_height_mode?: "maquette" | "true_scale";
  wall_height_mm?: number | null;
  wall_height_m?: number | null;
  base_plate?: boolean;
  base_thickness_mm?: number;
  min_wall_mm?: number;
  cut_doors?: boolean;
  cut_windows?: boolean;
}

export interface FpCapabilities {
  neural_detector: boolean;
  ocr_scale: boolean;
  pdf: boolean;
  max_upload_mb: number;
  sizes_mm: number[];
}

/** Читає файл як data-URL — бекенд приймає саме його (multipart тут не використовується). */
export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Не вдалось прочитати файл"));
    reader.readAsDataURL(file);
  });
}

function messageFrom(error: unknown, fallback: string): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string" && detail.trim()) return detail;
  }
  return fallback;
}

export const floorplanApi = {
  async capabilities(): Promise<FpCapabilities> {
    const { data } = await axios.get<FpCapabilities>(`${API_BASE_URL}/api/floorplan/capabilities`);
    return data;
  },

  async analyze(params: {
    image: string;
    filename?: string;
    referencePx?: number;
    referenceM?: number;
    useOcr?: boolean;
  }): Promise<FpAnalyzeResponse> {
    try {
      const { data } = await axios.post<FpAnalyzeResponse>(
        `${API_BASE_URL}/api/floorplan/analyze`,
        {
          image: params.image,
          filename: params.filename,
          reference_px: params.referencePx,
          reference_m: params.referenceM,
          use_ocr: params.useOcr !== false,
        },
        { timeout: 180000 },
      );
      return data;
    } catch (error) {
      throw new Error(messageFrom(error, "Не вдалось проаналізувати план."));
    }
  },

  async generate(request: FpGenerateRequest): Promise<{ task_id: string; status: string }> {
    try {
      const { data } = await axios.post<{ task_id: string; status: string }>(
        `${API_BASE_URL}/api/floorplan/generate`,
        {
          plan: request.plan,
          m_per_px: request.m_per_px,
          model_size_mm: request.model_size_mm,
          wall_height_mode: request.wall_height_mode ?? "maquette",
          wall_height_mm: request.wall_height_mm ?? null,
          wall_height_m: request.wall_height_m ?? null,
          base_plate: request.base_plate ?? true,
          base_thickness_mm: request.base_thickness_mm ?? 2.0,
          min_wall_mm: request.min_wall_mm ?? 1.2,
          cut_doors: request.cut_doors ?? true,
          cut_windows: request.cut_windows ?? true,
        },
        { timeout: 120000 },
      );
      return data;
    } catch (error) {
      throw new Error(messageFrom(error, "Не вдалось запустити генерацію."));
    }
  },
};

// ── Геометрія редактора (усе в пікселях прев'ю) ──────────────────────────────
export function wallLength(wall: FpWall): number {
  return Math.hypot(wall.x2 - wall.x1, wall.y2 - wall.y1);
}

export function pointAt(wall: FpWall, t: number): { x: number; y: number } {
  return { x: wall.x1 + (wall.x2 - wall.x1) * t, y: wall.y1 + (wall.y2 - wall.y1) * t };
}

/** Відстань від точки до відрізка + позиція проєкції (0..1). */
export function distanceToWall(
  wall: FpWall,
  px: number,
  py: number,
): { distance: number; t: number } {
  const vx = wall.x2 - wall.x1;
  const vy = wall.y2 - wall.y1;
  const lengthSq = vx * vx + vy * vy;
  if (lengthSq < 1e-9) return { distance: Math.hypot(px - wall.x1, py - wall.y1), t: 0 };
  let t = ((px - wall.x1) * vx + (py - wall.y1) * vy) / lengthSq;
  t = Math.min(Math.max(t, 0), 1);
  const proj = pointAt(wall, t);
  return { distance: Math.hypot(px - proj.x, py - proj.y), t };
}

/** Габарит плану в пікселях з урахуванням товщин. */
export function planBounds(plan: FpPlan) {
  if (!plan.walls.length) return { minX: 0, minY: 0, maxX: 0, maxY: 0 };
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const wall of plan.walls) {
    const half = wall.thickness_m / 2;
    minX = Math.min(minX, wall.x1 - half, wall.x2 - half);
    minY = Math.min(minY, wall.y1 - half, wall.y2 - half);
    maxX = Math.max(maxX, wall.x1 + half, wall.x2 + half);
    maxY = Math.max(maxY, wall.y1 + half, wall.y2 + half);
  }
  return { minX, minY, maxX, maxY };
}

/** Розміри та площа плану в МЕТРАХ для заданого масштабу. */
export function planMetrics(plan: FpPlan, mPerPx: number) {
  const { minX, minY, maxX, maxY } = planBounds(plan);
  const width = (maxX - minX) * mPerPx;
  const height = (maxY - minY) * mPerPx;
  return { width, height, area: width * height };
}

/** Медіанна товщина — нову стіну малюємо такою ж, як решта. */
export function medianThickness(plan: FpPlan): number {
  if (!plan.walls.length) return 6;
  const values = plan.walls.map((w) => w.thickness_m).sort((a, b) => a - b);
  return values[Math.floor(values.length / 2)];
}

/** Прилипання до 0/90° відносно домінантного напряму плану. */
export function snapToAxis(
  x1: number, y1: number, x2: number, y2: number, enabled: boolean,
): { x2: number; y2: number } {
  if (!enabled) return { x2, y2 };
  const dx = x2 - x1;
  const dy = y2 - y1;
  return Math.abs(dx) >= Math.abs(dy) ? { x2, y2: y1 } : { x2: x1, y2 };
}

export const MODEL_SIZES_MM = [100, 150, 200, 250] as const;
export const WALL_HEIGHTS_M = [2.5, 2.7, 3.0, 3.2] as const;
