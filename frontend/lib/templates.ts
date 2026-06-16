// ============================================================
// Shared template library — cities, map district presets,
// keychain style presets. Used by home, maps, keychains.
// ============================================================

export type CityKey = string;

export interface CityDef {
  key: CityKey;
  label: string;          // Ukrainian display name
  center: [number, number];
  bounds: { north: number; south: number; east: number; west: number };
}

// 23 Ukrainian regional centres (matches backend coverage)
export const CITIES: CityDef[] = [
  { key: "Kyiv",           label: "Київ",            center: [50.4501, 30.5234], bounds: { north: 50.60, south: 50.20, east: 30.80, west: 30.20 } },
  { key: "Lviv",           label: "Львів",           center: [49.8397, 24.0297], bounds: { north: 49.90, south: 49.78, east: 24.11, west: 23.95 } },
  { key: "Odesa",          label: "Одеса",           center: [46.4825, 30.7233], bounds: { north: 46.56, south: 46.39, east: 30.83, west: 30.61 } },
  { key: "Kharkiv",        label: "Харків",          center: [49.9935, 36.2304], bounds: { north: 50.07, south: 49.92, east: 36.34, west: 36.12 } },
  { key: "Dnipro",         label: "Дніпро",          center: [48.4647, 35.0462], bounds: { north: 48.55, south: 48.37, east: 35.14, west: 34.95 } },
  { key: "Vinnytsia",      label: "Вінниця",         center: [49.2331, 28.4682], bounds: { north: 49.28, south: 49.18, east: 28.53, west: 28.40 } },
  { key: "Khmelnytskyi",   label: "Хмельницький",    center: [49.42, 26.98],     bounds: { north: 49.48, south: 49.36, east: 27.08, west: 26.88 } },
  { key: "Zaporizhzhia",   label: "Запоріжжя",       center: [47.8388, 35.1396], bounds: { north: 47.90, south: 47.78, east: 35.22, west: 35.07 } },
  { key: "Kryvyi_Rih",     label: "Кривий Ріг",      center: [47.9105, 33.3918], bounds: { north: 47.98, south: 47.85, east: 33.44, west: 33.28 } },
  { key: "Mykolaiv",       label: "Миколаїв",        center: [46.9750, 32.0000], bounds: { north: 46.99, south: 46.92, east: 32.08, west: 31.97 } },
  { key: "Poltava",        label: "Полтава",         center: [49.5883, 34.5514], bounds: { north: 49.64, south: 49.54, east: 34.61, west: 34.48 } },
  { key: "Cherkasy",       label: "Черкаси",         center: [49.4444, 32.0598], bounds: { north: 49.47, south: 49.40, east: 32.11, west: 31.99 } },
  { key: "Chernihiv",      label: "Чернігів",        center: [51.4982, 31.2893], bounds: { north: 51.54, south: 51.44, east: 31.32, west: 31.22 } },
  { key: "Ternopil",       label: "Тернопіль",       center: [49.5535, 25.5948], bounds: { north: 49.59, south: 49.52, east: 25.65, west: 25.53 } },
  { key: "IvanoFrankivsk", label: "Івано-Франківськ",center: [48.9226, 24.7111], bounds: { north: 48.96, south: 48.88, east: 24.76, west: 24.65 } },
  { key: "Zhytomyr",       label: "Житомир",         center: [50.2547, 28.6587], bounds: { north: 50.30, south: 50.23, east: 28.72, west: 28.61 } },
  { key: "Sumy",           label: "Суми",            center: [50.9077, 34.7981], bounds: { north: 50.95, south: 50.88, east: 34.84, west: 34.74 } },
  { key: "Rivne",          label: "Рівне",           center: [50.6199, 26.2516], bounds: { north: 50.65, south: 50.57, east: 26.31, west: 26.18 } },
  { key: "Lutsk",          label: "Луцьк",           center: [50.7472, 25.3254], bounds: { north: 50.80, south: 50.70, east: 25.38, west: 25.27 } },
  { key: "Uzhhorod",       label: "Ужгород",         center: [48.6238, 22.2947], bounds: { north: 48.65, south: 48.60, east: 22.33, west: 22.26 } },
  { key: "Chernivtsi",     label: "Чернівці",        center: [48.2921, 25.9310], bounds: { north: 48.33, south: 48.26, east: 25.99, west: 25.90 } },
  { key: "Kherson",        label: "Херсон",          center: [46.6354, 32.6169], bounds: { north: 46.67, south: 46.61, east: 32.67, west: 32.57 } },
  { key: "Kropyvnytskyi",  label: "Кропивницький",   center: [48.5132, 32.2597], bounds: { north: 48.54, south: 48.47, east: 32.30, west: 32.20 } },
];

export const CITY_LABELS: Record<string, string> = Object.fromEntries(
  CITIES.map((c) => [c.key, c.label]),
);

// Cyrillic→Latin transliteration — MIRRORS backend _CYR_TO_LAT in
// services/flat_plate_pipeline.py. The keychain/magnet engraving font is latin
// only (DejaVu Sans), so the backend transliterates anyway; doing it here too
// means the text the user SEES in the label field equals what gets printed.
// Keep these two tables in sync.
const CYR_TO_LAT: Record<string, string> = {
  А: "A", Б: "B", В: "V", Г: "H", Ґ: "G", Д: "D", Е: "E", Є: "YE",
  Ж: "ZH", З: "Z", И: "Y", І: "I", Ї: "YI", Й: "Y", К: "K", Л: "L",
  М: "M", Н: "N", О: "O", П: "P", Р: "R", С: "S", Т: "T", У: "U",
  Ф: "F", Х: "KH", Ц: "TS", Ч: "CH", Ш: "SH", Щ: "SHCH", Ь: "",
  Ю: "YU", Я: "YA",
};

/** Uppercase latin transliteration of a (Ukrainian) string. Latin chars pass
 *  through unchanged, so it is safe to call on names already in latin. */
export function transliterateUA(text: string): string {
  return (text || "")
    .toUpperCase()
    .split("")
    .map((ch) => (ch in CYR_TO_LAT ? CYR_TO_LAT[ch] : ch))
    .join("");
}

/** Canonical latin engraving text for a city key (e.g. "Lviv" → "LVIV").
 *  Single source of truth for the keychain label and the magnet map label. */
export function cityKeychainText(key: CityKey): string {
  const c = CITIES.find((x) => x.key === key);
  return c ? transliterateUA(c.label) : "CITY";
}

// ---- Map district templates: famous places, ready to generate ----
export interface MapTemplate {
  id: string;
  cityKey: CityKey;
  city: string;           // display
  district: string;
  tag?: string;           // "Бестселер" | "Новинка" | ""
  blurb: string;
  center: [number, number];
  // ~radius in degrees around center for a ~1.5 km² district view
  span: number;
}

export const MAP_TEMPLATES: MapTemplate[] = [
  { id: "kyiv-podil",     cityKey: "Kyiv",   city: "Київ",   district: "Поділ",            tag: "Бестселер", blurb: "Старе серце Києва — звивисті вулиці та Андріївський узвіз.", center: [50.4660, 30.5170], span: 0.012 },
  { id: "kyiv-pechersk",  cityKey: "Kyiv",   city: "Київ",   district: "Печерськ",          tag: "Новинка",   blurb: "Лаврські пагорби, Маріїнський парк, парадні проспекти.",   center: [50.4280, 30.5430], span: 0.012 },
  { id: "kyiv-khreshchatyk", cityKey: "Kyiv", city: "Київ", district: "Хрещатик",          tag: "",          blurb: "Центральна вісь міста та Майдан Незалежності.",            center: [50.4490, 30.5230], span: 0.011 },
  { id: "lviv-rynok",     cityKey: "Lviv",   city: "Львів",  district: "Площа Ринок",       tag: "Бестселер", blurb: "Ратуша, бруківка та щільна сітка кварталів старого міста.", center: [49.8419, 24.0315], span: 0.009 },
  { id: "lviv-citadel",   cityKey: "Lviv",   city: "Львів",  district: "Цитадель",          tag: "",          blurb: "Пагорб з парком, обвитий серпантином історичних вулиць.",  center: [49.8330, 24.0240], span: 0.010 },
  { id: "odesa-deribasivska", cityKey: "Odesa", city: "Одеса", district: "Дерибасівська",   tag: "Бестселер", blurb: "Серце Одеси: Дерибасівська, Міський сад, бульвари.",        center: [46.4846, 30.7400], span: 0.011 },
  { id: "odesa-prymorsky",cityKey: "Odesa",  city: "Одеса",  district: "Приморський",       tag: "",          blurb: "Дюк, схил до Потьомкінських сходів і морський фасад.",      center: [46.4880, 30.7430], span: 0.011 },
  { id: "kharkiv-svobody",cityKey: "Kharkiv",city: "Харків", district: "Площа Свободи",     tag: "",          blurb: "Держпром і промениста сітка проспектів центру.",           center: [49.9988, 36.2300], span: 0.013 },
  { id: "dnipro-naberezhna", cityKey: "Dnipro", city: "Дніпро", district: "Набережна",      tag: "",          blurb: "Широка дуга Дніпра, мости й хвиля висоток.",                center: [48.4570, 35.0530], span: 0.013 },
  { id: "chernivtsi-rez", cityKey: "Chernivtsi", city: "Чернівці", district: "Резиденція",  tag: "",          blurb: "Резиденція митрополитів і кам'яні фасади навколо.",         center: [48.2960, 25.9240], span: 0.010 },
  { id: "ivano-center",   cityKey: "IvanoFrankivsk", city: "Івано-Франківськ", district: "Стометрівка", tag: "", blurb: "Пішохідний центр і ратуша на площі Ринок.",          center: [48.9226, 24.7111], span: 0.010 },
  { id: "uzhhorod-old",   cityKey: "Uzhhorod", city: "Ужгород", district: "Старе місто",    tag: "",          blurb: "Набережна Ужа, замок і найдовша липова алея Європи.",       center: [48.6210, 22.2980], span: 0.010 },
];

// ---- Map "style" presets (layers + look) ----
export interface MapStylePreset {
  id: string;
  label: string;
  blurb: string;
  layers: { buildings: boolean; roads: boolean; water: boolean; parks: boolean; terrain: boolean };
}

export const MAP_STYLE_PRESETS: MapStylePreset[] = [
  { id: "full",     label: "Повна деталізація", blurb: "Будівлі, дороги, вода, парки — усе.",        layers: { buildings: true, roads: true, water: true, parks: true, terrain: false } },
  { id: "relief",   label: "З рельєфом",        blurb: "Додає пагорби й перепади висот місцевості.", layers: { buildings: true, roads: true, water: true, parks: true, terrain: true } },
  { id: "minimal",  label: "Мінімалізм",        blurb: "Лише будівлі й дороги — чистий вигляд.",      layers: { buildings: true, roads: true, water: false, parks: false, terrain: false } },
  { id: "nature",   label: "Природа",           blurb: "Акцент на воді й парках.",                   layers: { buildings: true, roads: true, water: true, parks: true, terrain: true } },
];

// ---- Keychain style templates ----
export interface KeychainTemplateDef {
  id: string;
  label: string;
  blurb: string;
  tag?: string;
  // maps onto KeychainDesignerConfig partial
  config: Record<string, unknown>;
}

export const KEYCHAIN_TEMPLATES: KeychainTemplateDef[] = [
  {
    id: "heart-46",
    label: "Серце 46 × 42",
    blurb: "Мапа місця, що в серці — подарунок для двох.",
    tag: "Новинка",
    config: { baseShape: "heart", bodyWidthMm: 46, bodyHeightMm: 42, cornerRadiusMm: 0, mapWidthMm: 46, mapHeightMm: 42, loopXMm: 23, loopYMm: 1.5, loopOuterMm: 4, loopInnerMm: 2, labelXMm: 23, labelYMm: 27, labelWidthMm: 22, labelBandMm: 5, labelTextHeightMm: 3.0, labelStrokeMm: 0.9 },
  },
  {
    id: "token-55",
    label: "Жетон 55 × 30",
    blurb: "Класичний жетон з отвором — як монета на ключі.",
    tag: "Популярне",
    config: { baseShape: "token", bodyWidthMm: 55, bodyHeightMm: 30, cornerRadiusMm: 15, mapWidthMm: 55, mapHeightMm: 30, loopXMm: 27.5, loopYMm: 4, loopOuterMm: 2.8, loopInnerMm: 1.5, labelXMm: 32, labelYMm: 25.2, labelWidthMm: 34, labelBandMm: 6, labelTextHeightMm: 3.2, labelStrokeMm: 0.9, rimWidthMm: 0.9, rimHeightMm: 0.35 },
  },
  {
    id: "classic-35x55",
    label: "Класичний 35 × 55",
    blurb: "Вертикальний брелок з вушком зверху.",
    tag: "Бестселер",
    config: { baseShape: "rounded", bodyWidthMm: 35, bodyHeightMm: 55, cornerRadiusMm: 4.2, mapWidthMm: 35, mapHeightMm: 55, loopXMm: 17.5, loopYMm: 0, loopOuterMm: 4, loopInnerMm: 2, labelXMm: 17.5, labelYMm: 51, labelWidthMm: 30, labelBandMm: 5, labelTextHeightMm: 3.2, labelStrokeMm: 0.9 },
  },
  {
    id: "wide-55x35",
    label: "Широкий 55 × 35",
    blurb: "Горизонтальний формат для широкого району.",
    config: { baseShape: "rounded", bodyWidthMm: 55, bodyHeightMm: 35, cornerRadiusMm: 5, mapWidthMm: 55, mapHeightMm: 35, loopXMm: 27.5, loopYMm: 0, loopOuterMm: 4, loopInnerMm: 2, labelXMm: 27.5, labelYMm: 31, labelWidthMm: 46, labelBandMm: 5, labelTextHeightMm: 3.2, labelStrokeMm: 0.9 },
  },
  {
    id: "square-45",
    label: "Квадрат 45 × 45",
    blurb: "Симетрична площа з петлею у кутку.",
    config: { baseShape: "rounded", bodyWidthMm: 45, bodyHeightMm: 45, cornerRadiusMm: 5, mapWidthMm: 45, mapHeightMm: 45, loopXMm: 22.5, loopYMm: 0, loopOuterMm: 4, loopInnerMm: 2, labelXMm: 22.5, labelYMm: 41, labelWidthMm: 38, labelBandMm: 5, labelTextHeightMm: 3.2, labelStrokeMm: 0.9 },
  },
];

// (Видалено мертвий SIZE_PRICES зі застарілими цінами {s:1990…} — ніде не
//  імпортувався; єдине джерело цін = MAP_SIZE_PRICES_UAH у lib/mapPrices.ts.)
