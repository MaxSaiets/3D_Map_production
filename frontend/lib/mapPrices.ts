// ──────────────────────────────────────────────────────────────────────────
// ЄДИНЕ ДЖЕРЕЛО ПРАВДИ для цін МАПИ на фронті.
// ⚠ ТРИМАТИ В СИНХРОНІ з backend/pricing.json → map.sizes_mm (жива ціна йде з
// /api/quote; це fallback + structured-data). Усе нижче (SIMPLE_SIZES у
// generation.ts, AggregateOffer на 138 city-pages, create/keychains LD-схеми)
// читає ОДНУ цю таблицю, щоб ціни не «дрейфували» між місцями (Google Merchant
// «price mismatch» + розбіжність UI vs реальний прайс).
//
// Як змінити ціни:
//   1) backend/pricing.json → map.sizes_mm (runtime-джерело, оператор бачить)
//   2) MAP_SIZE_PRICES_UAH нижче (фронт-fallback + SEO) — мають збігатися
// ──────────────────────────────────────────────────────────────────────────

/** Канонічна таблиця ЦІН МАПИ за розміром: ребро моделі (мм) → ціна (UAH).
 *  Дзеркало backend/pricing.json → map.sizes_mm. 60мм = магніт (окремий SKU,
 *  не входить у звичайну S/M/L/XL лінійку, тому тут немає). */
export const MAP_SIZE_PRICES_UAH = {
  55: 350,  // S  (5.5 см)   2026-09-08: +40 % (було 250/350/450/550)
  80: 490,  // M  (8 см)
  110: 630, // L  (11 см)
  150: 770, // XL (15 см)
} as const;

export type MapSizeMm = keyof typeof MAP_SIZE_PRICES_UAH;

/** Магніт-мапа (60мм) — окремий SKU. UAH. Дзеркало pricing.json map.sizes_mm["60"]. */
export const MAP_MAGNET_PRICE_UAH = 210;

/** Брелок-мапа (3D-друк) — базова ціна. UAH. Дзеркало pricing.json keychain.base. */
export const KEYCHAIN_PRICE_UAH = 170;

/** Макет квартири з плану — ціна за ФІЗИЧНИМ розміром виробу (мм → UAH).
 *  Дзеркало backend/pricing.json → floorplan.sizes_mm. Покупець обирає
 *  сантиметри, а не архітектурний масштаб — так само роблять Etsy-продавці. */
export const FLOORPLAN_SIZE_PRICES_UAH = {
  100: 830,
  150: 1250,
  200: 1810,
  250: 2510,
} as const;

export type FloorplanSizeMm = keyof typeof FLOORPLAN_SIZE_PRICES_UAH;

export function floorplanPriceUah(sizeMm: number): number {
  const sizes = Object.keys(FLOORPLAN_SIZE_PRICES_UAH).map(Number);
  const nearest = sizes.reduce((a, b) => (Math.abs(b - sizeMm) < Math.abs(a - sizeMm) ? b : a));
  return FLOORPLAN_SIZE_PRICES_UAH[nearest as FloorplanSizeMm];
}

/** Надбавка за рельєф (terrain). UAH. Дзеркало pricing.json map.relief_addon. */
export const MAP_RELIEF_ADDON_UAH = 85;

/** Позиційний курс UAH→EUR (як на лендінгу, НЕ біржовий ФХ). Округлюємо до
 *  «гарних» євро (350₴≈8€, 770₴≈18€). */
export const EUR_PER_UAH = 0.024;

/** Ціна розміру у EUR (позиційний курс, округлення до цілого євро). */
export function mapPriceEur(uah: number): number {
  return Math.round(uah * EUR_PER_UAH);
}

const _uahValues = Object.values(MAP_SIZE_PRICES_UAH);
const _lowUah = Math.min(..._uahValues);
const _highUah = Math.max(..._uahValues);

// Діапазон цін для structured-data (AggregateOffer). Виводиться з таблиці вище —
// одна зміна ціни оновлює і UI, і SEO одночасно.
export const MAP_PRICE_RANGE = {
  offerCount: String(_uahValues.length),
  uk: { currency: "UAH", low: String(_lowUah), high: String(_highUah) },
  eu: { currency: "EUR", low: String(mapPriceEur(_lowUah)), high: String(mapPriceEur(_highUah)) },
} as const;

export function mapPriceRange(locale: string) {
  const r = locale === "uk" ? MAP_PRICE_RANGE.uk : MAP_PRICE_RANGE.eu;
  return { ...r, offerCount: MAP_PRICE_RANGE.offerCount };
}
