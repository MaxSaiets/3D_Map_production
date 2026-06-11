import { CITIES } from "@/lib/templates";
import type { AppLocale } from "@/i18n/routing";

/**
 * Programmatic SEO: дані для сторінок міст /maps/[slug].
 * Слаг — латиницею в нижньому регістрі; назви локалізовані (uk з CITIES,
 * pl/de мають усталені екзоніми, en/fr/es — офіційна транслітерація).
 */
export interface CityPage {
  slug: string;
  key: string; // CITIES key
  names: Record<AppLocale, string>;
  center: [number, number];
}

const NAME_OVERRIDES: Record<string, Partial<Record<AppLocale, string>>> = {
  Kyiv: { pl: "Kijów", de: "Kiew" },
  Lviv: { pl: "Lwów", de: "Lemberg" },
  Odesa: { pl: "Odessa", de: "Odessa", fr: "Odessa", es: "Odesa" },
  Kharkiv: { pl: "Charków", de: "Charkiw" },
  Dnipro: { pl: "Dniepr" },
  Vinnytsia: { pl: "Winnica", de: "Winnyzja" },
  Khmelnytskyi: { pl: "Chmielnicki", de: "Chmelnyzkyj" },
  Zaporizhzhia: { pl: "Zaporoże", de: "Saporischschja" },
  Kryvyi_Rih: { pl: "Krzywy Róg", de: "Krywyj Rih" },
  Mykolaiv: { pl: "Mikołajów", de: "Mykolajiw" },
  Poltava: { pl: "Połtawa", de: "Poltawa" },
  Cherkasy: { pl: "Czerkasy", de: "Tscherkassy" },
  Chernihiv: { pl: "Czernihów", de: "Tschernihiw" },
  Ternopil: { pl: "Tarnopol" },
  IvanoFrankivsk: { pl: "Iwano-Frankiwsk", de: "Iwano-Frankiwsk" },
  Zhytomyr: { pl: "Żytomierz", de: "Schytomyr" },
  Rivne: { pl: "Równe", de: "Riwne" },
  Lutsk: { pl: "Łuck", de: "Luzk" },
  Uzhhorod: { pl: "Użhorod", de: "Uschhorod" },
  Chernivtsi: { pl: "Czerniowce", de: "Tscherniwzi" },
  Kherson: { pl: "Chersoń", de: "Cherson" },
  Kropyvnytskyi: { pl: "Kropywnycki", de: "Kropywnyzkyj" },
};

/** "Kryvyi_Rih" → "Kryvyi Rih", "IvanoFrankivsk" → "Ivano-Frankivsk" */
function latinName(key: string): string {
  if (key === "IvanoFrankivsk") return "Ivano-Frankivsk";
  return key.replace(/_/g, " ");
}

function slugify(key: string): string {
  if (key === "IvanoFrankivsk") return "ivano-frankivsk";
  return key.toLowerCase().replace(/_/g, "-");
}

export const CITY_PAGES: CityPage[] = CITIES.map((c) => {
  const latin = latinName(c.key);
  const o = NAME_OVERRIDES[c.key] || {};
  return {
    slug: slugify(c.key),
    key: c.key,
    center: c.center,
    names: {
      uk: c.label,
      en: o.en ?? latin,
      de: o.de ?? latin,
      pl: o.pl ?? latin,
      fr: o.fr ?? latin,
      es: o.es ?? latin,
    },
  };
});

export const CITY_PAGE_BY_SLUG: Record<string, CityPage> = Object.fromEntries(
  CITY_PAGES.map((c) => [c.slug, c]),
);
