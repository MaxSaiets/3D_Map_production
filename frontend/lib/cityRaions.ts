/**
 * SEO 26.09.2026: сторінки АДМІНІСТРАТИВНИХ РАЙОНІВ великих міст —
 * /maps/{city}/{slug}, лише uk + en. Запити «3d модель оболонського району»,
 * «макет Салтівки купити», «мапа Сихова».
 *
 * Координати центрів — Wikidata (P625), де вони є; для решти — центр
 * житлового масиву району. Орієнтири вписано ЛИШЕ безсумнівні; якщо їх немає,
 * сторінка тримається на обчислюваних фактах (напрям і відстань від центру
 * міста, сусідні райони), без вигаданих деталей.
 * Посилання «Створити» веде в конструктор одразу на центр району (?lat&lon).
 */
export interface CityRaion {
  citySlug: string;
  slug: string;
  uk: string; // «Оболонський район»
  en: string; // «Obolonskyi District»
  center: [number, number];
  landmarks?: { uk: string; en: string }[];
}

const L = (uk: string, en: string) => ({ uk, en });

export const CITY_RAIONS: CityRaion[] = [
  // Київ
  { citySlug: "kyiv", slug: "holosiivskyi-rayon", uk: "Голосіївський район", en: "Holosiivskyi District", center: [50.3276, 30.5512],
    landmarks: [L("Голосіївський парк", "Holosiivskyi Park"), L("ВДНГ", "VDNH Expocenter"), L("музей просто неба «Пирогів»", "Pyrohiv open-air museum")] },
  { citySlug: "kyiv", slug: "darnytskyi-rayon", uk: "Дарницький район", en: "Darnytskyi District", center: [50.4040, 30.6330],
    landmarks: [L("Позняки", "Pozniaky"), L("Осокорки", "Osokorky"), L("Харківський масив", "Kharkivskyi masyv")] },
  { citySlug: "kyiv", slug: "desnianskyi-rayon", uk: "Деснянський район", en: "Desnianskyi District", center: [50.5191, 30.6744],
    landmarks: [L("Троєщина", "Troieshchyna"), L("Лісовий масив", "Lisovyi masyv")] },
  { citySlug: "kyiv", slug: "dniprovskyi-rayon", uk: "Дніпровський район", en: "Dniprovskyi District", center: [50.4564, 30.6444],
    landmarks: [L("Русанівка", "Rusanivka"), L("Гідропарк", "Hidropark"), L("Лівобережна", "Livoberezhna")] },
  { citySlug: "kyiv", slug: "obolonskyi-rayon", uk: "Оболонський район", en: "Obolonskyi District", center: [50.5318, 30.4210],
    landmarks: [L("Оболонська набережна", "Obolon embankment"), L("парк «Наталка»", "Natalka Park")] },
  { citySlug: "kyiv", slug: "pecherskyi-rayon", uk: "Печерський район", en: "Pecherskyi District", center: [50.4288, 30.5531],
    landmarks: [L("Києво-Печерська лавра", "Kyiv Pechersk Lavra"), L("Маріїнський палац", "Mariinskyi Palace"), L("монумент «Батьківщина-Мати»", "Motherland Monument")] },
  { citySlug: "kyiv", slug: "podilskyi-rayon", uk: "Подільський район", en: "Podilskyi District", center: [50.4895, 30.4522],
    landmarks: [L("Контрактова площа", "Kontraktova Square"), L("Андріївський узвіз", "Andriivskyi Descent"), L("Виноградар", "Vynohradar")] },
  { citySlug: "kyiv", slug: "sviatoshynskyi-rayon", uk: "Святошинський район", en: "Sviatoshynskyi District", center: [50.4689, 30.3350],
    landmarks: [L("Академмістечко", "Akademmistechko"), L("Борщагівка", "Borshchahivka")] },
  { citySlug: "kyiv", slug: "solomianskyi-rayon", uk: "Солом'янський район", en: "Solomianskyi District", center: [50.4206, 30.4578],
    landmarks: [L("аеропорт «Київ» (Жуляни)", "Kyiv Zhuliany Airport"), L("Солом'янка", "Solomianka")] },
  { citySlug: "kyiv", slug: "shevchenkivskyi-rayon", uk: "Шевченківський район", en: "Shevchenkivskyi District", center: [50.4520, 30.5010],
    landmarks: [L("Золоті ворота", "Golden Gate"), L("Софійський собор", "Saint Sophia Cathedral"), L("Лук'янівка", "Lukianivka")] },

  // Харків
  { citySlug: "kharkiv", slug: "shevchenkivskyi-rayon", uk: "Шевченківський район", en: "Shevchenkivskyi District", center: [50.0167, 36.2167],
    landmarks: [L("Держпром", "Derzhprom"), L("майдан Свободи", "Freedom Square")] },
  { citySlug: "kharkiv", slug: "kyivskyi-rayon", uk: "Київський район", en: "Kyivskyi District", center: [50.0413, 36.2995] },
  { citySlug: "kharkiv", slug: "saltivskyi-rayon", uk: "Салтівський район", en: "Saltivskyi District", center: [50.0038, 36.3337],
    landmarks: [L("Північна Салтівка", "Pivnichna Saltivka")] },
  { citySlug: "kharkiv", slug: "nemyshlianskyi-rayon", uk: "Немишлянський район", en: "Nemyshlianskyi District", center: [49.9667, 36.3357] },
  { citySlug: "kharkiv", slug: "industrialnyi-rayon", uk: "Індустріальний район", en: "Industrialnyi District", center: [49.9500, 36.3667],
    landmarks: [L("Харківський тракторний завод", "Kharkiv Tractor Plant")] },
  { citySlug: "kharkiv", slug: "slobidskyi-rayon", uk: "Слобідський район", en: "Slobidskyi District", center: [49.9333, 36.2667] },
  { citySlug: "kharkiv", slug: "osnovianskyi-rayon", uk: "Основ'янський район", en: "Osnovianskyi District", center: [49.9833, 36.2333] },
  { citySlug: "kharkiv", slug: "kholodnohirskyi-rayon", uk: "Холодногірський район", en: "Kholodnohirskyi District", center: [49.9954, 36.1774],
    landmarks: [L("Холодна гора", "Kholodna Hora")] },
  { citySlug: "kharkiv", slug: "novobavarskyi-rayon", uk: "Новобаварський район", en: "Novobavarskyi District", center: [49.9667, 36.2000],
    landmarks: [L("Нова Баварія", "Nova Bavariia")] },

  // Одеса
  { citySlug: "odesa", slug: "prymorskyi-rayon", uk: "Приморський район", en: "Prymorskyi District", center: [46.4775, 30.7326],
    landmarks: [L("Дерибасівська", "Derybasivska Street"), L("Оперний театр", "Opera House"), L("Потьомкінські сходи", "Potemkin Stairs")] },
  { citySlug: "odesa", slug: "kyivskyi-rayon", uk: "Київський район", en: "Kyivskyi District", center: [46.4000, 30.7200],
    landmarks: [L("житловий масив Таїрова", "Tairova"), L("Великий Фонтан", "Velykyi Fontan")] },
  { citySlug: "odesa", slug: "malynovskyi-rayon", uk: "Малиновський район", en: "Malynovskyi District", center: [46.4550, 30.6800],
    landmarks: [L("Черемушки", "Cheriomushky")] },
  { citySlug: "odesa", slug: "khadzhybeiskyi-rayon", uk: "Хаджибейський район", en: "Khadzhybeiskyi District", center: [46.5300, 30.7100],
    landmarks: [L("Лузанівка", "Luzanivka")] },

  // Дніпро
  { citySlug: "dnipro", slug: "tsentralnyi-rayon", uk: "Центральний район", en: "Tsentralnyi District", center: [48.4667, 35.0333],
    landmarks: [L("проспект Яворницького", "Yavornytskoho Avenue")] },
  { citySlug: "dnipro", slug: "sobornyi-rayon", uk: "Соборний район", en: "Sobornyi District", center: [48.4500, 35.0667] },
  { citySlug: "dnipro", slug: "shevchenkivskyi-rayon", uk: "Шевченківський район", en: "Shevchenkivskyi District", center: [48.4091, 35.0129] },
  { citySlug: "dnipro", slug: "chechelivskyi-rayon", uk: "Чечелівський район", en: "Chechelivskyi District", center: [48.4211, 34.9669] },
  { citySlug: "dnipro", slug: "novokodatskyi-rayon", uk: "Новокодацький район", en: "Novokodatskyi District", center: [48.4750, 34.9450] },
  { citySlug: "dnipro", slug: "amur-nyzhnodniprovskyi-rayon", uk: "Амур-Нижньодніпровський район", en: "Amur-Nyzhnodniprovskyi District", center: [48.5211, 34.9783] },
  { citySlug: "dnipro", slug: "industrialnyi-rayon", uk: "Індустріальний район", en: "Industrialnyi District", center: [48.5167, 35.0833] },
  { citySlug: "dnipro", slug: "samarskyi-rayon", uk: "Самарський район", en: "Samarskyi District", center: [48.4169, 35.1178] },

  // Львів
  { citySlug: "lviv", slug: "halytskyi-rayon", uk: "Галицький район", en: "Halytskyi District", center: [49.8410, 24.0310],
    landmarks: [L("площа Ринок", "Rynok Square"), L("Львівська опера", "Lviv Opera House")] },
  { citySlug: "lviv", slug: "lychakivskyi-rayon", uk: "Личаківський район", en: "Lychakivskyi District", center: [49.8331, 24.0833],
    landmarks: [L("Личаківський цвинтар", "Lychakiv Cemetery")] },
  { citySlug: "lviv", slug: "sykhivskyi-rayon", uk: "Сихівський район", en: "Sykhivskyi District", center: [49.8000, 24.0550],
    landmarks: [L("Сихів", "Sykhiv")] },
  { citySlug: "lviv", slug: "frankivskyi-rayon", uk: "Франківський район", en: "Frankivskyi District", center: [49.8111, 23.9969],
    landmarks: [L("Стрийський парк", "Stryiskyi Park")] },
  { citySlug: "lviv", slug: "zaliznychnyi-rayon", uk: "Залізничний район", en: "Zaliznychnyi District", center: [49.8314, 23.9578],
    landmarks: [L("головний залізничний вокзал", "Lviv main railway station")] },
  { citySlug: "lviv", slug: "shevchenkivskyi-rayon", uk: "Шевченківський район", en: "Shevchenkivskyi District", center: [49.8700, 23.9842],
    landmarks: [L("Замарстинів", "Zamarstyniv")] },

  // Запоріжжя
  { citySlug: "zaporizhzhia", slug: "khortytskyi-rayon", uk: "Хортицький район", en: "Khortytskyi District", center: [47.8164, 35.0547],
    landmarks: [L("острів Хортиця", "Khortytsia Island")] },
  { citySlug: "zaporizhzhia", slug: "oleksandrivskyi-rayon", uk: "Олександрівський район", en: "Oleksandrivskyi District", center: [47.8250, 35.1667] },
  { citySlug: "zaporizhzhia", slug: "voznesenivskyi-rayon", uk: "Вознесенівський район", en: "Voznesenivskyi District", center: [47.8167, 35.1833] },
  { citySlug: "zaporizhzhia", slug: "dniprovskyi-rayon", uk: "Дніпровський район", en: "Dniprovskyi District", center: [47.8792, 35.0688] },
  { citySlug: "zaporizhzhia", slug: "zavodskyi-rayon", uk: "Заводський район", en: "Zavodskyi District", center: [47.8961, 35.1536] },
  { citySlug: "zaporizhzhia", slug: "kosmichnyi-rayon", uk: "Космічний район", en: "Kosmichnyi District", center: [47.7796, 35.2178] },
  { citySlug: "zaporizhzhia", slug: "shevchenkivskyi-rayon", uk: "Шевченківський район", en: "Shevchenkivskyi District", center: [47.8489, 35.2536] },

  // Кривий Ріг
  { citySlug: "kryvyi-rih", slug: "tsentralno-miskyi-rayon", uk: "Центрально-Міський район", en: "Tsentralno-Miskyi District", center: [47.9025, 33.3367] },
  { citySlug: "kryvyi-rih", slug: "metalurhiinyi-rayon", uk: "Металургійний район", en: "Metalurhiinyi District", center: [47.9072, 33.3872] },
  { citySlug: "kryvyi-rih", slug: "dovhyntsivskyi-rayon", uk: "Довгинцівський район", en: "Dovhyntsivskyi District", center: [47.9117, 33.4231] },
  { citySlug: "kryvyi-rih", slug: "saksahanskyi-rayon", uk: "Саксаганський район", en: "Saksahanskyi District", center: [47.9408, 33.4181] },
  { citySlug: "kryvyi-rih", slug: "pokrovskyi-rayon", uk: "Покровський район", en: "Pokrovskyi District", center: [47.9989, 33.4492] },
  { citySlug: "kryvyi-rih", slug: "ternivskyi-rayon", uk: "Тернівський район", en: "Ternivskyi District", center: [48.1450, 33.5553] },
  { citySlug: "kryvyi-rih", slug: "inhuletskyi-rayon", uk: "Інгулецький район", en: "Inhuletskyi District", center: [47.8367, 33.3475] },

  // Миколаїв
  { citySlug: "mykolaiv", slug: "tsentralnyi-rayon", uk: "Центральний район", en: "Tsentralnyi District", center: [47.0011, 31.9520] },
  { citySlug: "mykolaiv", slug: "zavodskyi-rayon", uk: "Заводський район", en: "Zavodskyi District", center: [46.9418, 31.9498] },
  { citySlug: "mykolaiv", slug: "inhulskyi-rayon", uk: "Інгульський район", en: "Inhulskyi District", center: [46.9477, 32.0662] },
  { citySlug: "mykolaiv", slug: "korabelnyi-rayon", uk: "Корабельний район", en: "Korabelnyi District", center: [46.8733, 32.0210] },
];

export const RAIONS_BY_CITY: Record<string, CityRaion[]> = CITY_RAIONS.reduce(
  (acc, r) => {
    (acc[r.citySlug] ??= []).push(r);
    return acc;
  },
  {} as Record<string, CityRaion[]>,
);

export const RAION_LOCALES = ["uk", "en"] as const;

/** Відстань (км) і напрям (uk/en) від точки a до b — для унікального тексту сторінки. */
export function bearingFrom(a: readonly [number, number], b: readonly [number, number]) {
  const R = 6371, toRad = (d: number) => (d * Math.PI) / 180;
  const dLat = toRad(b[0] - a[0]), dLng = toRad(b[1] - a[1]);
  const s = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(a[0])) * Math.cos(toRad(b[0])) * Math.sin(dLng / 2) ** 2;
  const km = 2 * R * Math.asin(Math.sqrt(s));
  const y = Math.sin(dLng) * Math.cos(toRad(b[0]));
  const x = Math.cos(toRad(a[0])) * Math.sin(toRad(b[0])) - Math.sin(toRad(a[0])) * Math.cos(toRad(b[0])) * Math.cos(dLng);
  const deg = (Math.atan2(y, x) * 180) / Math.PI + 360;
  const i = Math.round(deg / 45) % 8;
  const uk = ["на північ", "на північний схід", "на схід", "на південний схід", "на південь", "на південний захід", "на захід", "на північний захід"][i];
  const en = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"][i];
  return { km, uk, en };
}
