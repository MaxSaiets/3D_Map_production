/**
 * Готові студійні рендери 3D-моделей для сторінок міст/районів
 * (/maps-renders/{id}.webp, id = slug міста або «{місто}--{район}»).
 * ФАЙЛ ГЕНЕРУЄ tools/night_city_renders.py — не правити вручну.
 */
export const MAP_RENDERS: ReadonlySet<string> = new Set([
  "dnipro",
  "kharkiv",
  "kyiv",
  "lviv",
  "odesa",
]);
