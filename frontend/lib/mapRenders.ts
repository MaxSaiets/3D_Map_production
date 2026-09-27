/**
 * Готові студійні рендери 3D-моделей для сторінок міст/районів
 * (/maps-renders/{id}.webp, id = slug міста або «{місто}--{район}»).
 * ФАЙЛ ГЕНЕРУЄ tools/night_city_renders.py — не правити вручну.
 */
export const MAP_RENDERS: ReadonlySet<string> = new Set([
  "cherkasy",
  "chernihiv",
  "dnipro",
  "kharkiv",
  "khmelnytskyi",
  "kryvyi-rih",
  "kyiv",
  "lviv",
  "mykolaiv",
  "odesa",
  "poltava",
  "ternopil",
  "vinnytsia",
  "zaporizhzhia",
]);
