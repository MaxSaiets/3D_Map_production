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
  55: 250,  // S  (5.5 см)
  80: 350,  // M  (8 см)
  110: 450, // L  (11 см)
  150: 550, // XL (15 см)
} as const;

export type MapSizeMm = keyof typeof MAP_SIZE_PRICES_UAH;

/** Магніт-мапа (60мм) — окремий SKU. UAH. Дзеркало pricing.json map.sizes_mm["60"]. */
export const MAP_MAGNET_PRICE_UAH = 150;

/** Брелок-мапа (3D-друк) — базова ціна. UAH. Дзеркало pricing.json keychain.base. */
export const KEYCHAIN_PRICE_UAH = 120;

/** Надбавка за рельєф (terrain). UAH. Дзеркало pricing.json map.relief_addon. */
export const MAP_RELIEF_ADDON_UAH = 60;

/** Позиційний курс UAH→EUR (як на лендінгу, НЕ біржовий ФХ). Округлюємо до
 *  «гарних» євро (250₴≈6€, 550₴≈13€). */
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
