// Канонічний діапазон цін МАПИ для structured-data (AggregateOffer на 138 city-pages
// + create/keychains LD-схеми). ⚠ ТРИМАТИ В СИНХРОНІ з backend/pricing.json → map.sizes_mm
// (зараз S250 … XL890 UAH). EU = позиційний курс ~×0.024 (як на лендінгу), НЕ ФХ.
// Централізовано тут, щоб зміна цін не «дрейфувала» у 138 сторінках structured-data
// (Google Merchant «price mismatch»).
export const MAP_PRICE_RANGE = {
  offerCount: "4",
  uk: { currency: "UAH", low: "250", high: "890" },
  eu: { currency: "EUR", low: "6", high: "21" },
} as const;

export function mapPriceRange(locale: string) {
  const r = locale === "uk" ? MAP_PRICE_RANGE.uk : MAP_PRICE_RANGE.eu;
  return { ...r, offerCount: MAP_PRICE_RANGE.offerCount };
}
