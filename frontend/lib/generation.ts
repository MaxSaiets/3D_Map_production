// Shared builder for the map generation request, so the simple panel, the full
// ControlPanel and the capture route all produce identical payloads.

import { MAP_SIZE_PRICES_UAH, type MapSizeMm } from "@/lib/mapPrices";

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
    terrain_base_thickness_mm: p.terrainBaseThicknessMm ?? 0.3,
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
