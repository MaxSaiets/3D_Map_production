// Shared builder for the map generation request, so the simple panel, the full
// ControlPanel and the capture route all produce identical payloads.

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
  /** Мапа-магніт: плаский режим + кишеня під магніт Ø10×2мм у центрі дна. */
  magnetPocket?: boolean;
  /** Підпис на плоскій мапі/магніті (рельєфний текст у смузі внизу). */
  mapLabel?: string;
  /** D4 GPX-трек [[lon,lat],...] — підвищений шар-маршрут поверх мапи. */
  gpxTrack?: Array<[number, number]> | null;
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
    ...(p.magnetPocket ? { magnet_pocket: true } : {}),
    ...(p.mapLabel && p.mapLabel.trim() ? { map_label: p.mapLabel.trim() } : {}),
    ...(p.gpxTrack && p.gpxTrack.length >= 2 ? { gpx_track: p.gpxTrack } : {}),
  };
}

// Curated size options for the simple flow (mm + estimated price in ₴).
// price = fallback, узгоджено з backend/pricing.json (живу ціну дає /api/quote).
export const SIMPLE_SIZES = [
  { key: "s",  label: "S", mm: 55,  cm: "5.5 см", price: 690 },
  { key: "m",  label: "M", mm: 80,  cm: "8 см",   price: 890 },
  { key: "l",  label: "L", mm: 110, cm: "11 см",  price: 1290 },
  { key: "xl", label: "XL", mm: 150, cm: "15 см", price: 1790 },
] as const;
