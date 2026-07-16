// Shared builder for the map generation request, so the simple panel, the full
// ControlPanel and the capture route all produce identical payloads.

import { MAP_SIZE_PRICES_UAH, type MapSizeMm } from "@/lib/mapPrices";
import { api } from "@/lib/api";

export interface MapRequestParams {
  north: number; south: number; east: number; west: number;
  roadWidthMultiplier?: number;
  roadHeightMm?: number;
  roadEmbedMm?: number;
  buildingMinHeight?: number;
  buildingHeightMultiplier?: number;
  buildingFoundationMm?: number;
  buildingEmbedMm?: number;
  waterDepth?: number;
  terrainEnabled?: boolean;
  terrainZScale?: number;
  terrainBaseThicknessMm?: number;
  /** Пласкі будинки у плоских режимах: усі однакової низької висоти (footprint-плити). */
  flatUniformBuildingHeight?: boolean;
  flatMaxBuildingHeightMm?: number;
  /** Кольорова тема/палітра (#2): classic | sepia | noir | ocean | neon. */
  colorPalette?: string;
  terrainResolution?: number;
  terrariumZoom?: number;
  exportFormat?: "stl" | "3mf";
  modelSizeMm?: number;
  isAmsMode?: boolean;
  flatPlateMode?: boolean;
  previewMode?: boolean;
  previewIncludeBase?: boolean;
  previewIncludeRoads?: boolean;
  previewIncludeBuildings?: boolean;
  previewIncludeWater?: boolean;
  previewIncludeParks?: boolean;
  /** Rotated-rectangle corners [lon,lat] for a single figure (not grid). When
   *  set, the backend crops OSM to this polygon instead of the axis-aligned bbox. */
  zonePolygonCoords?: Array<[number, number]> | null;
  /** Мапа-магніт: плаский режим + 4 кишені під магніти-шайби Ø4×2мм у дні
   *  (діагональне кільце; кишеня Ø4.4×2.1 = кліренс 0.4/0.1 для вклеювання). */
  magnetPocket?: boolean;
  /** Підпис на плоскій мапі/магніті (рельєфний текст у смузі внизу). */
  mapLabel?: string;
  /** D4 GPX-трек [[lon,lat],...] — підвищений шар-маршрут поверх мапи. */
  gpxTrack?: Array<[number, number]> | null;
  /** З'єднувач-пази (метелик): «ластівчин-хвіст» пази на серединах граней +
   *  окрема деталь-ключ. Дві плоскі карти стикуються пазами (паз у дні 3мм
   *  основи → спереду непомітно). Лише плоский режим. */
  mapConnector?: boolean;
  /** Преміум-рамка: компас + масштабна лінійка + координати центру окремою
   *  чорною деталлю поверх плоскої карти. Лише плоский режим. */
  mapFrame?: boolean;
  /** Стиль рамки: classic | ornate | compass (мапиться у backend frame_style). */
  frameStyle?: string;
  /** Виділені будівлі: окремі ЧЕРВОНІ вставні деталі (паз+peg). Плоский режим. */
  mapHighlightBuilding?: boolean;
  /** [[lon,lat],...] обраних будівель (кліки по карті); інакше — будинок у центрі. */
  highlightPoints?: Array<[number, number]>;
}

export function buildMapRequest(p: MapRequestParams) {
  return {
    north: p.north, south: p.south, east: p.east, west: p.west,
    road_width_multiplier: p.roadWidthMultiplier ?? 0.8,
    road_height_mm: p.roadHeightMm ?? 0.5,
    road_embed_mm: p.roadEmbedMm ?? 0.3,
    building_min_height: p.buildingMinHeight ?? 5.0,
    building_height_multiplier: p.buildingHeightMultiplier ?? 1.8,
    building_foundation_mm: p.buildingFoundationMm ?? 0.6,
    building_embed_mm: p.buildingEmbedMm ?? 0.2,
    water_depth: p.waterDepth ?? 2.0,
    terrain_enabled: p.terrainEnabled ?? true,
    terrain_z_scale: p.terrainZScale ?? 1.0,
    terrain_base_thickness_mm: p.terrainBaseThicknessMm ?? 1.3,
    flat_uniform_building_height: Boolean(p.flatUniformBuildingHeight),
    ...(p.flatMaxBuildingHeightMm ? { flat_max_building_height_mm: p.flatMaxBuildingHeightMm } : {}),
    color_palette: p.colorPalette ?? "classic",
    terrain_resolution: p.terrainResolution ?? 180,
    terrarium_zoom: p.terrariumZoom ?? 15,
    flatten_buildings_on_terrain: false,
    flatten_roads_on_terrain: false,
    export_format: p.exportFormat ?? "3mf",
    model_size_mm: p.modelSizeMm ?? 80,
    context_padding_m: 400.0,
    is_ams_mode: Boolean(p.isAmsMode) && !p.flatPlateMode,
    flat_plate_mode: Boolean(p.flatPlateMode),
    preview_mode: Boolean(p.previewMode) && !p.flatPlateMode,
    preview_include_base: p.previewIncludeBase ?? true,
    preview_include_roads: p.previewIncludeRoads ?? true,
    preview_include_buildings: p.previewIncludeBuildings ?? true,
    preview_include_water: p.previewIncludeWater ?? true,
    preview_include_parks: p.previewIncludeParks ?? true,
    ...(p.zonePolygonCoords && p.zonePolygonCoords.length >= 3
      ? { zone_polygon_coords: p.zonePolygonCoords }
      : {}),
    ...(p.magnetPocket
      ? {
          magnet_pocket: true,
          // Шайби Ø4×2мм: кишеня з кліренсом 0.4мм по діаметру і 0.1мм по
          // глибині; 4 шт по кутах — тримає рівно і не обертається.
          magnet_pocket_diameter_mm: 4.4,
          magnet_pocket_depth_mm: 2.1,
          magnet_pocket_count: 4,
          magnet_pocket_inset_mm: 8,
        }
      : {}),
    ...(p.mapLabel && p.mapLabel.trim() ? { map_label: p.mapLabel.trim() } : {}),
    ...(p.gpxTrack && p.gpxTrack.length >= 2 ? { gpx_track: p.gpxTrack } : {}),
    // З'єднувач-пази: бек має дефолти (NSEW, 10×15×2мм, кліренс 0.2) — шлемо лише прапор.
    ...(p.mapConnector ? { map_connector: true } : {}),
    // Преміум-рамка: бек має дефолти (компас+лінійка+координати) — шлемо прапор + стиль.
    ...(p.mapFrame ? { map_frame: true, frame_style: p.frameStyle ?? "classic" } : {}),
    // Виділені будівлі: прапор + точки [[lon,lat],...] обраних будівель (якщо клікнули).
    ...(p.mapHighlightBuilding ? { map_highlight_building: true } : {}),
    ...(p.mapHighlightBuilding && p.highlightPoints && p.highlightPoints.length > 0
      ? { highlight_points: p.highlightPoints }
      : {}),
  };
}

// Гнучкий масштаб для GPX-треків: коли трек не влазить у стандартний 1:10000
// (10 м/мм), зона може розширюватись аж до 35 м/мм (~1:35000) — плоска мапа
// друкується шарами, тож точний масштаб не критичний, лише дрібніші деталі.
export const GPX_MAX_M_PER_MM = 35;

// Curated size options for the simple flow (mm + estimated price in ₴).
// `price` — fallback (живу ціну дає /api/quote). ЦІНА НЕ задається тут вручну:
// береться з канонічної таблиці MAP_SIZE_PRICES_UAH (lib/mapPrices.ts), яка
// дзеркалить backend/pricing.json. Так fallback фізично не може розійтися з
// прайсом і з SEO-схемами — одне джерело на всі три місця.
export const SIMPLE_SIZES = [
  { key: "s",  label: "S",  mm: 55,  cm: "5.5 см", price: MAP_SIZE_PRICES_UAH[55] },
  { key: "m",  label: "M",  mm: 80,  cm: "8 см",   price: MAP_SIZE_PRICES_UAH[80] },
  { key: "l",  label: "L",  mm: 110, cm: "11 см",  price: MAP_SIZE_PRICES_UAH[110] },
  { key: "xl", label: "XL", mm: 150, cm: "15 см",  price: MAP_SIZE_PRICES_UAH[150] },
] as const satisfies ReadonlyArray<{ key: string; label: string; mm: MapSizeMm; cm: string; price: number }>;

// СПІЛЬНА генерація СЕРІЇ зон (батч клітин сітки). Винесено сюди, щоб і
// «Профі» ControlPanel, і «Просто» SimpleControlPanel кликали ОДИН код —
// інакше дві копії логіки розходяться (anti-drift). Функція НЕ чіпає store:
// валідацію і всі store-побічні-ефекти (setTaskGroup/setShowAllZones/...) робить
// викликач; тут лише сортування зон, виклик API і деривація ids/meta.
export interface RunZoneGenArgs {
  selectedZones: any[];
  request: Record<string, any>;
  onSeriesGenerated?: (cells: Array<{ row: number; col: number; task_id?: string; zone_id?: string }>) => void;
}
export interface RunZoneGenResult {
  taskId: string;
  taskIds: string[];
  batchMeta: Record<string, { zoneId: string; row?: number; col?: number; cx?: number; cy?: number; sf?: number }>;
  zonesSorted: any[];
}
export async function runZoneGeneration(args: RunZoneGenArgs): Promise<RunZoneGenResult> {
  const { selectedZones, request, onSeriesGenerated } = args;
  // Сортуємо клітини row→col→id, щоб ids[i] стабільно відповідали zonesSorted[i]
  // (продовження панно: збереження сітки за row/col, складене превʼю за порядком).
  const zonesSorted = [...selectedZones].sort((a, b) => {
    const ar = Number(a?.properties?.row ?? 0);
    const br = Number(b?.properties?.row ?? 0);
    if (ar !== br) return ar - br;
    const ac = Number(a?.properties?.col ?? 0);
    const bc = Number(b?.properties?.col ?? 0);
    if (ac !== bc) return ac - bc;
    const aid = String(a?.id || a?.properties?.id || "");
    const bid = String(b?.id || b?.properties?.id || "");
    return aid.localeCompare(bid);
  });

  const response = await api.generateZones(zonesSorted, request as any);
  const ids: string[] =
    (response as any).all_task_ids && (response as any).all_task_ids.length
      ? (response as any).all_task_ids
      : [response.task_id];

  // Продовження панно: віддаємо викликачу клітини з task_id (zonesSorted[i]↔ids[i])
  // → авто-збереження сітки, щоб ці зони лишились «куплені» (золоті) надалі.
  try {
    onSeriesGenerated?.(
      zonesSorted.map((z: any, i: number) => ({
        row: Number(z?.properties?.row ?? 0),
        col: Number(z?.properties?.col ?? 0),
        task_id: ids[i],
        zone_id: String(z?.id ?? z?.properties?.id ?? ""),
      })),
    );
  } catch { /* збереження сітки не критичне для генерації */ }

  const batchMeta: Record<string, { zoneId: string; row?: number; col?: number; cx?: number; cy?: number; sf?: number }> = {};
  // Розмір моделі (мм) для обчислення scale_factor — точно як на беку:
  // scale_factor = model_size_mm / max(zone_w_m, zone_h_m).
  const _modelMm = Number(request?.model_size_mm) || 80;
  for (let i = 0; i < ids.length; i += 1) {
    const zone = zonesSorted[i];
    const zoneId = String(zone?.id || zone?.properties?.id || `zone_${i}`);
    // ГЕОГРАФІЧНИЙ ЦЕНТРОЇД клітини (lng,lat) → превʼю розкладає плитки за РЕАЛЬНИМИ
    // позиціями (точна тесселяція гекса/квадрата), а не за приблизною row/col-сіткою
    // (через яку плитки «стояли криво»).
    let cx: number | undefined, cy: number | undefined, sf: number | undefined;
    const ring: number[][] = zone?.geometry?.coordinates?.[0] || [];
    if (ring.length >= 3) {
      let sx = 0, sy = 0, n = 0;
      let minLng = Infinity, maxLng = -Infinity, minLat = Infinity, maxLat = -Infinity;
      for (const p of ring) {
        if (Number.isFinite(p?.[0]) && Number.isFinite(p?.[1])) {
          sx += p[0]; sy += p[1]; n++;
          if (p[0] < minLng) minLng = p[0]; if (p[0] > maxLng) maxLng = p[0];
          if (p[1] < minLat) minLat = p[1]; if (p[1] > maxLat) maxLat = p[1];
        }
      }
      if (n > 0) {
        cx = sx / n; cy = sy / n;
        // scale_factor (мм/м) — ЄДИНИЙ геометричний масштаб для конгруентних плиток.
        // Превʼю ставить сусідів на nn_світ × sf → точна тесселяція (гекс: усі 6
        // сусідів рівновіддалені = flat-to-flat = nn). Замість per-tile rendered-bbox
        // ширини (що різниться вмістом доріг → зазори/«бока»).
        const cosLat = Math.max(Math.cos((cy * Math.PI) / 180), 0.05);
        const wM = (maxLng - minLng) * 111320 * cosLat;
        const hM = (maxLat - minLat) * 110540;
        const zoneSizeM = Math.max(wM, hM, 1);
        sf = _modelMm / zoneSizeM;
      }
    }
    batchMeta[String(ids[i])] = {
      zoneId,
      row: zone?.properties?.row,
      col: zone?.properties?.col,
      cx,
      cy,
      sf,
    };
  }

  return { taskId: response.task_id, taskIds: ids, batchMeta, zonesSorted };
}
