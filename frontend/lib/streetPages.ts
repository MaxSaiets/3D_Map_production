import { CITY_STREETS, type CityStreet } from "@/lib/cityStreets";
import { DISTRICT_PAGES } from "@/lib/cityLanding";
import { CITY_RAIONS } from "@/lib/cityRaions";

export type { CityStreet };

/**
 * Вулиці, що отримують власну сторінку /maps/{city}/{slug}: без колізій зі
 * slug-ами шаблонних кварталів (напр. «khreshchatyk» уже має ручну сторінку)
 * і адмін-районів.
 */
export const STREET_PAGES: CityStreet[] = CITY_STREETS.filter(
  (s) =>
    !DISTRICT_PAGES.some((d) => d.citySlug === s.citySlug && d.slug === s.slug) &&
    !CITY_RAIONS.some((r) => r.citySlug === s.citySlug && r.slug === s.slug),
);

export const STREET_PAGES_BY_CITY: Record<string, CityStreet[]> = STREET_PAGES.reduce(
  (acc, s) => {
    (acc[s.citySlug] ??= []).push(s);
    return acc;
  },
  {} as Record<string, CityStreet[]>,
);
