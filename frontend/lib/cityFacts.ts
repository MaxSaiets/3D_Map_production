/**
 * Унікальні факти про місто для сторінок /maps/[slug] — щоб контент НЕ був
 * byte-identical шаблоном (анти-doorway, SEO content-quality). Числа
 * локале-незалежні; власні назви (річка/область/візитівка) мають uk + latin
 * (uk-локаль бере .uk, решта — .latin, як NAME_OVERRIDES у cityPages).
 *
 * Дані зібрано й перехресно верифіковано (населення — офіційна оцінка
 * Держстату на 01.01.2022, довоєнна; рік фіксується у populationYear).
 * Ключ = slug із cityPages.
 */
export interface CityFacts {
  population: number;       // осіб
  populationYear: number;   // рік оцінки населення
  founded: number;          // рік заснування АБО першої згадки (див. firstMention)
  firstMention: boolean;    // true → "Перша згадка", false → "Засноване"
  area_km2: number;
  river: { uk: string; latin: string };
  oblast: { uk: string; latin: string };
  landmark: { uk: string; latin: string };
}

export const CITY_FACTS: Record<string, CityFacts> = {
  kyiv: { population: 2952301, populationYear: 2022, founded: 482, firstMention: false, area_km2: 835.58,
    river: { uk: "Дніпро", latin: "Dnipro" }, oblast: { uk: "Місто зі спеціальним статусом", latin: "City with special status" },
    landmark: { uk: "Софійський собор", latin: "Saint Sophia Cathedral" } },
  lviv: { population: 717273, populationYear: 2022, founded: 1256, firstMention: true, area_km2: 171.71,
    river: { uk: "Полтва", latin: "Poltva" }, oblast: { uk: "Львівська область", latin: "Lviv Oblast" },
    landmark: { uk: "Львівська опера", latin: "Lviv Opera House" } },
  odesa: { population: 1010537, populationYear: 2022, founded: 1794, firstMention: false, area_km2: 162.42,
    river: { uk: "Чорне море", latin: "Black Sea" }, oblast: { uk: "Одеська область", latin: "Odesa Oblast" },
    landmark: { uk: "Потьомкінські сходи", latin: "Potemkin Stairs" } },
  kharkiv: { population: 1421125, populationYear: 2022, founded: 1654, firstMention: false, area_km2: 350,
    river: { uk: "Лопань", latin: "Lopan" }, oblast: { uk: "Харківська область", latin: "Kharkiv Oblast" },
    landmark: { uk: "Держпром", latin: "Derzhprom" } },
  dnipro: { population: 968502, populationYear: 2022, founded: 1776, firstMention: false, area_km2: 409.72,
    river: { uk: "Дніпро", latin: "Dnipro" }, oblast: { uk: "Дніпропетровська область", latin: "Dnipropetrovsk Oblast" },
    landmark: { uk: "Спасо-Преображенський собор", latin: "Transfiguration Cathedral" } },
  vinnytsia: { population: 369739, populationYear: 2022, founded: 1363, firstMention: true, area_km2: 140,
    river: { uk: "Південний Буг", latin: "Southern Bug" }, oblast: { uk: "Вінницька область", latin: "Vinnytsia Oblast" },
    landmark: { uk: "Фонтан «Рошен»", latin: "Roshen Fountain" } },
  khmelnytskyi: { population: 274452, populationYear: 2022, founded: 1431, firstMention: true, area_km2: 93.05,
    river: { uk: "Південний Буг", latin: "Southern Bug" }, oblast: { uk: "Хмельницька область", latin: "Khmelnytskyi Oblast" },
    landmark: { uk: "Собор Різдва Богородиці", latin: "Cathedral of the Nativity" } },
  zaporizhzhia: { population: 710052, populationYear: 2022, founded: 1770, firstMention: false, area_km2: 331,
    river: { uk: "Дніпро", latin: "Dnipro" }, oblast: { uk: "Запорізька область", latin: "Zaporizhzhia Oblast" },
    landmark: { uk: "Острів Хортиця", latin: "Khortytsia Island" } },
  "kryvyi-rih": { population: 603904, populationYear: 2022, founded: 1775, firstMention: false, area_km2: 430,
    river: { uk: "Інгулець", latin: "Inhulets" }, oblast: { uk: "Дніпропетровська область", latin: "Dnipropetrovsk Oblast" },
    landmark: { uk: "Скелі «Орлине гніздо»", latin: "Eagle's Nest Rocks" } },
  mykolaiv: { population: 470011, populationYear: 2022, founded: 1789, firstMention: false, area_km2: 252.83,
    river: { uk: "Південний Буг", latin: "Southern Bug" }, oblast: { uk: "Миколаївська область", latin: "Mykolaiv Oblast" },
    landmark: { uk: "Музей суднобудування", latin: "Museum of Shipbuilding" } },
  poltava: { population: 279593, populationYear: 2022, founded: 1174, firstMention: true, area_km2: 103,
    river: { uk: "Ворскла", latin: "Vorskla" }, oblast: { uk: "Полтавська область", latin: "Poltava Oblast" },
    landmark: { uk: "Монумент Слави", latin: "Column of Glory" } },
  cherkasy: { population: 269836, populationYear: 2022, founded: 1284, firstMention: false, area_km2: 75,
    river: { uk: "Дніпро", latin: "Dnipro" }, oblast: { uk: "Черкаська область", latin: "Cherkasy Oblast" },
    landmark: { uk: "Пагорб Слави", latin: "Hill of Glory" } },
  chernihiv: { population: 282747, populationYear: 2022, founded: 907, firstMention: true, area_km2: 79,
    river: { uk: "Десна", latin: "Desna" }, oblast: { uk: "Чернігівська область", latin: "Chernihiv Oblast" },
    landmark: { uk: "Спасо-Преображенський собор", latin: "Transfiguration Cathedral" } },
  ternopil: { population: 225004, populationYear: 2022, founded: 1540, firstMention: true, area_km2: 86,
    river: { uk: "Серет", latin: "Seret" }, oblast: { uk: "Тернопільська область", latin: "Ternopil Oblast" },
    landmark: { uk: "Тернопільський замок", latin: "Ternopil Castle" } },
  "ivano-frankivsk": { population: 238196, populationYear: 2022, founded: 1662, firstMention: false, area_km2: 83.7,
    river: { uk: "Бистриця", latin: "Bystrytsia" }, oblast: { uk: "Івано-Франківська область", latin: "Ivano-Frankivsk Oblast" },
    landmark: { uk: "Ратуша", latin: "Town Hall" } },
  zhytomyr: { population: 261624, populationYear: 2022, founded: 884, firstMention: false, area_km2: 61,
    river: { uk: "Тетерів", latin: "Teteriv" }, oblast: { uk: "Житомирська область", latin: "Zhytomyr Oblast" },
    landmark: { uk: "Міст у парку Шодуар", latin: "Shodouar Park Bridge" } },
  sumy: { population: 256474, populationYear: 2022, founded: 1655, firstMention: false, area_km2: 95.39,
    river: { uk: "Псел", latin: "Psel" }, oblast: { uk: "Сумська область", latin: "Sumy Oblast" },
    landmark: { uk: "Спасо-Преображенський собор", latin: "Transfiguration Cathedral" } },
  rivne: { population: 243873, populationYear: 2022, founded: 1283, firstMention: true, area_km2: 71,
    river: { uk: "Устя", latin: "Ustia" }, oblast: { uk: "Рівненська область", latin: "Rivne Oblast" },
    landmark: { uk: "Костел святого Антонія", latin: "St. Anthony's Church" } },
  lutsk: { population: 215986, populationYear: 2022, founded: 1085, firstMention: true, area_km2: 40.23,
    river: { uk: "Стир", latin: "Styr" }, oblast: { uk: "Волинська область", latin: "Volyn Oblast" },
    landmark: { uk: "Замок Любарта", latin: "Lubart's Castle" } },
  uzhhorod: { population: 115449, populationYear: 2022, founded: 1154, firstMention: true, area_km2: 65,
    river: { uk: "Уж", latin: "Uzh" }, oblast: { uk: "Закарпатська область", latin: "Zakarpattia Oblast" },
    landmark: { uk: "Ужгородський замок", latin: "Uzhhorod Castle" } },
  chernivtsi: { population: 264298, populationYear: 2022, founded: 1408, firstMention: true, area_km2: 153,
    river: { uk: "Прут", latin: "Prut" }, oblast: { uk: "Чернівецька область", latin: "Chernivtsi Oblast" },
    landmark: { uk: "Резиденція митрополитів", latin: "Metropolitans' Residence" } },
  kherson: { population: 279131, populationYear: 2022, founded: 1778, firstMention: false, area_km2: 135.7,
    river: { uk: "Дніпро", latin: "Dnipro" }, oblast: { uk: "Херсонська область", latin: "Kherson Oblast" },
    landmark: { uk: "Катерининський собор", latin: "St. Catherine's Cathedral" } },
  kropyvnytskyi: { population: 219676, populationYear: 2022, founded: 1754, firstMention: false, area_km2: 103,
    river: { uk: "Інгул", latin: "Inhul" }, oblast: { uk: "Кіровоградська область", latin: "Kirovohrad Oblast" },
    landmark: { uk: "Фортеця святої Єлисавети", latin: "Fortress of St. Elizabeth" } },
};

export function cityFacts(slug: string): CityFacts | undefined {
  return CITY_FACTS[slug];
}
