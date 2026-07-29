import type { AppLocale } from "@/i18n/routing";
import type { CityFacts } from "@/lib/cityFacts";

/**
 * УНІКАЛЬНИЙ ТЕКСТ ДЛЯ КОЖНОЇ СТОРІНКИ МІСТА (29.07.2026).
 *
 * ПРОБЛЕМА, яку лікуємо: виміряно 78-79% збігу тексту між сторінками міст
 * (454 слова, з них ~32 унікальні) → Google дедуплікує, сторінки лишаються
 * «Crawled/Discovered — currently not indexed».
 *
 * ЧОМУ НЕ РАНТАЙМ-ВИКЛИК LLM: Google із березня 2024 має політику
 * «scaled content abuse» — масова автогенерація сторінок без доданої цінності
 * карається незалежно від того, чим згенеровано. Плюс 64 міста × 6 мов = 384
 * виклики на кожен білд (вартість, латентність, недетермінованість, потреба
 * в ключі). Тому текст будується ДЕТЕРМІНОВАНО з РЕАЛЬНИХ даних міста:
 *
 *   1) обчислювані інсайти, яких немає в жодного іншого міста:
 *      щільність населення, вік, співвідношення площі, «читабельність» рельєфу;
 *   2) 6 різних СТРУКТУР абзацу — вибір за хешем slug (стабільний між білдами);
 *   3) підстановка власних назв (річка/регіон/орієнтир) у різні синтаксичні
 *      позиції, щоб збігалися не речення, а лише термінологія.
 *
 * Результат: кожна сторінка має власний набір чисел, формулювань і акцентів —
 * контент унікальний за суттю, а не переспам синонімами.
 */

export interface CityProseInput {
  slug: string;
  name: string;
  facts: CityFacts;
  locale: AppLocale;
}

/** Стабільний хеш slug → індекс варіанта (однаковий між білдами і локалями). */
function pick(slug: string, variants: number, salt = 0): number {
  let h = 2166136261 ^ salt;
  for (let i = 0; i < slug.length; i++) {
    h ^= slug.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h) % variants;
}

const CURRENT_YEAR = 2026;

type Derived = {
  density: number;        // осіб/км²
  age: number;            // років від заснування/першої згадки
  densityBand: 0 | 1 | 2; // 0 просторе, 1 середнє, 2 щільне
  sizeBand: 0 | 1 | 2;    // 0 невелике, 1 середнє, 2 велике
};

function derive(f: CityFacts): Derived {
  const density = Math.round(f.population / Math.max(f.area_km2, 1));
  const age = CURRENT_YEAR - f.founded;
  const densityBand = density > 4000 ? 2 : density > 1800 ? 1 : 0;
  const sizeBand = f.population > 1000000 ? 2 : f.population > 400000 ? 1 : 0;
  return { density, age, densityBand, sizeBand };
}

/* ── Українська ──────────────────────────────────────────────────────────── */
function uk(i: CityProseInput, d: Derived, nf: Intl.NumberFormat): string[] {
  const { name, facts: f } = i;
  const river = f.river.uk;
  const lm = f.landmark.uk;
  const region = f.oblast.uk;
  const v = pick(i.slug, 6);

  const densityPhrase = [
    `забудова просторова — ${nf.format(d.density)} осіб на км², тож у моделі добре видно окремі квартали й розриви між ними`,
    `щільність ${nf.format(d.density)} осіб на км² дає рівний ритм кварталів — мапа читається як цілісний масив`,
    `щільні ${nf.format(d.density)} осіб на км² перетворюють центр на суцільний рельєф дахів, де вулиці читаються як прорізи`,
  ][d.densityBand];

  const scalePhrase = [
    `За площею ${nf.format(Math.round(f.area_km2))} км² місто вміщується в одну модель без втрати деталей.`,
    `Площа ${nf.format(Math.round(f.area_km2))} км² означає, що для однієї моделі зазвичай беруть район, а не місто цілком.`,
    `На ${nf.format(Math.round(f.area_km2))} км² повне місто в одну плитку не влізе — тому найкраще працює або центр, або панно з кількох частин.`,
  ][d.sizeBand];

  const agePhrase =
    d.age > 900
      ? `${name} має за плечима понад ${Math.floor(d.age / 100) * 100} років — історичне ядро й пізніші райони мають помітно різну геометрію вулиць, і це видно на моделі.`
      : d.age > 400
        ? `Місту близько ${Math.round(d.age / 50) * 50} років, тож регулярні квартали пізніших епох сусідять зі старим нерегулярним центром.`
        : `Порівняно молода забудова (${d.age} років) дає впорядковану сітку вулиць — модель виходить графічною й читабельною.`;

  const A = `Головна водойма — ${river}, а архітектурна візитівка — ${lm}. Саме ці два орієнтири роблять 3D-мапу впізнаваною з першого погляду: русло дає природну діагональ, а домінанта — точку, за яку чіпляється око.`;
  const B = `У тривимірі ${name} впізнають за двома речами: лінією, яку прокладає ${river}, і масою ${lm} у щільній забудові. Все інше — контекст, що тримає ці орієнтири.`;
  const C = `${lm} і ${river} — те, що люди шукають на моделі першим. Решта кварталів працює як фон, який задає масштаб цим орієнтирам.`;

  const orientation = [A, B, C][pick(i.slug, 3, 7)];

  const closing = [
    `Регіон — ${region}. Для друку зазвичай беруть ділянку 1–3 км² навколо центру: у такому масштабі будинки лишаються окремими об'ємами, а не зливаються в суцільну плиту.`,
    `Адміністративно це ${region}. Оптимальна ділянка для однієї моделі — 1–3 км²: далі дрібні вулиці перестають читатись на друку 0.4 мм соплом.`,
    `${region} — саме тут проходять межі, за якими будується модель. Ділянка 1–3 км² дає найкращий баланс між охопленням і деталізацією.`,
  ][pick(i.slug, 3, 13)];

  // 6 різних порядків подачі — структура сторінки теж відрізняється
  const order: string[][] = [
    [orientation, `${scalePhrase} Тут ${densityPhrase}.`, agePhrase, closing],
    [`${agePhrase}`, orientation, `${scalePhrase} При цьому ${densityPhrase}.`, closing],
    [`${scalePhrase}`, orientation, `Крім того, ${densityPhrase}.`, `${agePhrase} ${closing}`],
    [orientation, agePhrase, `${scalePhrase} Додатково: ${densityPhrase}.`, closing],
    [`Тут ${densityPhrase}. ${scalePhrase}`, orientation, agePhrase, closing],
    [`${agePhrase}`, `${scalePhrase} ${orientation}`, `Варто врахувати: ${densityPhrase}.`, closing],
  ];
  return order[v];
}

/* ── Англійська (база для en + фолбек інших локалей) ─────────────────────── */
function en(i: CityProseInput, d: Derived, nf: Intl.NumberFormat): string[] {
  const { name, facts: f } = i;
  const river = f.river.latin;
  const lm = f.landmark.latin;
  const region = f.oblast.latin;
  const v = pick(i.slug, 6);

  const densityPhrase = [
    `the layout is spacious — ${nf.format(d.density)} people per km², so individual blocks and the gaps between them stay legible in the model`,
    `a density of ${nf.format(d.density)} people per km² gives an even block rhythm, and the map reads as one continuous fabric`,
    `at ${nf.format(d.density)} people per km² the centre becomes a solid relief of rooftops, with streets reading as cuts through it`,
  ][d.densityBand];

  const scalePhrase = [
    `At ${nf.format(Math.round(f.area_km2))} km² the city fits into a single model without losing detail.`,
    `With ${nf.format(Math.round(f.area_km2))} km² of area, one model usually covers a district rather than the whole city.`,
    `Spanning ${nf.format(Math.round(f.area_km2))} km², the full city will not fit one tile — the centre alone, or a multi-tile panel, works best.`,
  ][d.sizeBand];

  const agePhrase =
    d.age > 900
      ? `${name} carries more than ${Math.floor(d.age / 100) * 100} years of history — the old core and later districts have visibly different street geometry, and the model shows it.`
      : d.age > 400
        ? `The city is roughly ${Math.round(d.age / 50) * 50} years old, so regular later blocks sit next to an irregular historic centre.`
        : `Relatively young fabric (${d.age} years) means an ordered street grid — the model comes out graphic and easy to read.`;

  const A = `The defining water feature is the ${river}, and the architectural landmark is ${lm}. These two anchors make the 3D map recognisable at a glance: the channel gives a natural diagonal, the landmark gives the eye something to hold.`;
  const B = `In three dimensions ${name} is recognised by two things: the line drawn by the ${river} and the mass of ${lm} within dense blocks. Everything else is context holding those anchors.`;
  const C = `${lm} and the ${river} are what people look for first on the model. The remaining blocks act as a backdrop that sets the scale.`;

  const orientation = [A, B, C][pick(i.slug, 3, 7)];

  const closing = [
    `The region is ${region}. A print usually covers 1–3 km² around the centre: at that scale buildings stay separate volumes instead of merging into a slab.`,
    `Administratively this is ${region}. The sweet spot for a single model is 1–3 km² — beyond that, narrow streets stop resolving at a 0.4 mm nozzle.`,
    `${region} sets the boundaries the model is built from. A 1–3 km² area balances coverage against detail best.`,
  ][pick(i.slug, 3, 13)];

  const order: string[][] = [
    [orientation, `${scalePhrase} Here ${densityPhrase}.`, agePhrase, closing],
    [`${agePhrase}`, orientation, `${scalePhrase} Meanwhile ${densityPhrase}.`, closing],
    [`${scalePhrase}`, orientation, `On top of that, ${densityPhrase}.`, `${agePhrase} ${closing}`],
    [orientation, agePhrase, `${scalePhrase} Also worth noting: ${densityPhrase}.`, closing],
    [`Here ${densityPhrase}. ${scalePhrase}`, orientation, agePhrase, closing],
    [`${agePhrase}`, `${scalePhrase} ${orientation}`, `Worth factoring in: ${densityPhrase}.`, closing],
  ];
  return order[v];
}

/**
 * Абзаци унікального опису міста. uk — власна версія, решта локалей — en
 * (той самий підхід, що й у блозі: краще якісний en, ніж машинний переклад).
 */
export function cityProse(input: CityProseInput): string[] {
  const d = derive(input.facts);
  const nf = new Intl.NumberFormat(input.locale === "uk" ? "uk-UA" : input.locale);
  return input.locale === "uk" ? uk(input, d, nf) : en(input, d, nf);
}

/** Обчислені показники для видимого блоку фактів (унікальні числа на сторінку). */
export function cityDerivedFacts(facts: CityFacts) {
  return derive(facts);
}
