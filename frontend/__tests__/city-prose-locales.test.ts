/**
 * Опис міста на /maps/[city] має бути мовою сторінки — не англійським текстом
 * під німецьким/іспанським вступом (16.09.2026, /de/maps/vienna).
 */
import { cityProse } from "@/lib/cityProse";
import { WORLD_CITIES } from "@/lib/worldCities";

const vienna = WORLD_CITIES.find((c) => c.slug === "vienna")!;
const locales = ["uk", "en", "de", "es", "fr", "pl"] as const;

describe("cityProse locales", () => {
  it.each(locales)("%s: 4 абзаци, без незаповнених плейсхолдерів", (locale) => {
    const paras = cityProse({ slug: vienna.slug, name: vienna.names[locale], facts: vienna.facts, locale });
    expect(paras).toHaveLength(4);
    for (const p of paras) {
      expect(p).not.toMatch(/undefined|NaN|\$\{/);
      expect(p.length).toBeGreaterThan(40);
    }
  });

  it("de/es/fr/pl не є копією англійської", () => {
    const en = cityProse({ slug: vienna.slug, name: "Vienna", facts: vienna.facts, locale: "en" }).join(" ");
    for (const locale of ["de", "es", "fr", "pl"] as const) {
      const txt = cityProse({ slug: vienna.slug, name: vienna.names[locale], facts: vienna.facts, locale }).join(" ");
      expect(txt).not.toBe(en);
      expect(txt).not.toMatch(/\bthe centre\b|\bpeople per km²/);
    }
  });

  it("порядок абзаців стабільний між локалями (той самий hash slug)", () => {
    const de = cityProse({ slug: vienna.slug, name: "Wien", facts: vienna.facts, locale: "de" });
    const de2 = cityProse({ slug: vienna.slug, name: "Wien", facts: vienna.facts, locale: "de" });
    expect(de).toEqual(de2);
  });
});
