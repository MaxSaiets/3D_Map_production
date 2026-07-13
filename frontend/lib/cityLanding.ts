// ──────────────────────────────────────────────────────────────────────────
// Programmatic SEO рівень 2: місто × продукт (/brelok/[city], /podarunok/[city])
// + лендінги під нагоду (/podarunok/na-richnytsyu тощо).
//
// Контент живе тут (патерн lib/blog.ts): uk + en повні, de/es/fr/pl → en-фолбек.
// Анти-doorway: кожна сторінка отримує (а) унікальні факти міста з cityFacts
// (річка/візитівка/населення) вплетені у прозу, (б) РІЗНІ варіанти вступу,
// що ротуються детерміновано за slug (не Math.random — SSG має бути стабільним),
// (в) списки районів з MAP_TEMPLATES, (г) власні ціни/CTA. Жодного клоакінгу:
// сторінки видимі користувачам, злінковані з /maps/[city] та /podarunok.
// ──────────────────────────────────────────────────────────────────────────
import type { CityFacts } from "@/lib/cityFacts";

export type ContentLocale = "uk" | "en";
export function contentLocale(locale: string): ContentLocale {
  return locale === "uk" ? "uk" : "en";
}

/** Детермінований вибір варіанта за slug — стабільний між білдами. */
export function pickVariant<T>(slug: string, variants: readonly T[]): T {
  let h = 0;
  for (let i = 0; i < slug.length; i++) h = (h * 31 + slug.charCodeAt(i)) >>> 0;
  return variants[h % variants.length];
}

export interface LandingSection { h2: string; p: string[] }
export interface CityLandingCopy {
  title: string;
  description: string;
  h1: string;
  intro: string[];
  sections: LandingSection[];
}
export interface FaqItem { q: string; a: string }

const nfUk = new Intl.NumberFormat("uk-UA");
const nfEn = new Intl.NumberFormat("en");

/** FAQ для сторінок брелок/подарунок × місто (5 питань, підстановка міста).
 *  Не doorway-вода: реальні відповіді покупцю (терміни/ділянка/напис/доставка/оплата),
 *  + FAQPage JSON-LD → шанс на розгорнутий сніпет у видачі. */
export function cityFaq(cl: ContentLocale, name: string, kind: "brelok" | "podarunok"): FaqItem[] {
  if (cl === "uk") {
    return [
      {
        q: "Скільки триває виготовлення?",
        a: `Виготовлення ${kind === "brelok" ? "брелока" : "мапи чи брелока"} з районом міста (${name}) займає 1–3 робочі дні, після чого модель передається у доставку.`,
      },
      {
        q: `Яку ділянку в місті (${name}) краще обрати?`,
        a: "Найкраще виглядають ділянки зі змішаною забудовою: трохи вулиць, парк або вода — так район впізнається з першого погляду. Занадто велика ділянка робить деталі дрібними.",
      },
      {
        q: "Чи можна додати напис на звороті?",
        a: "Так, на звороті брелока чи мапи можна додати власний текст — ім'я, дату, координати чи назву міста. Опція доступна в конструкторі перед оформленням замовлення.",
      },
      {
        q: `Як відбувається доставка у ${name}?`,
        a: "Доставка Новою Поштою по всій Україні (відділення чи поштомат) або Nova Post / Meest у 15 країн ЄС. Вартість доставки оплачується окремо за тарифом перевізника.",
      },
      {
        q: "Як оплатити замовлення?",
        a: "Оплата карткою Visa/Mastercard онлайн через LiqPay або накладеним платежем при отриманні — на вибір при оформленні.",
      },
    ];
  }
  return [
    {
      q: "How long does production take?",
      a: `Making a ${kind === "brelok" ? "keychain" : "map or keychain"} with a ${name} district takes 1–3 business days, then the model goes to shipping.`,
    },
    {
      q: `Which area of ${name} should I pick?`,
      a: "Mixed areas look best: some streets, a park or water — the district stays recognizable at first glance. A too-large area makes details too small.",
    },
    {
      q: "Can I add text on the back?",
      a: "Yes — the back of a keychain or map can carry your own text: a name, a date, coordinates or a city name. The option is available in the builder before checkout.",
    },
    {
      q: `How is delivery to ${name} handled?`,
      a: "Delivery by Nova Poshta across Ukraine, or Nova Post / Meest to 15 EU countries. Shipping cost is charged separately at the carrier's rate.",
    },
    {
      q: "How do I pay for an order?",
      a: "Card payment online (Visa/Mastercard via LiqPay) or cash on delivery — your choice at checkout.",
    },
  ];
}

/** FAQ для лендінгів під нагоду (без прив'язки до конкретного міста). */
export function occasionFaq(cl: ContentLocale): FaqItem[] {
  if (cl === "uk") {
    return [
      {
        q: "Скільки триває виготовлення подарунка?",
        a: "1–3 робочі дні на виготовлення плюс доставка Новою Поштою. Якщо подарунок терміновий — брелок друкується найшвидше.",
      },
      {
        q: "Не знаю, яку ділянку обрати — що робити?",
        a: "Напишіть нам у чат — підкажемо, як виглядатиме обраний район, і зберемо безкоштовне превʼю перед замовленням.",
      },
      {
        q: "Чи можна додати особистий напис?",
        a: "Так — ім'я, дату, координати чи коротку фразу можна додати на звороті брелока або на підставці мапи.",
      },
      {
        q: "Скільки це коштує?",
        a: "Брелок — від 120 ₴, магніт — 150 ₴, 3D-мапа на полицю — від 250 до 550 ₴ залежно від розміру.",
      },
    ];
  }
  return [
    {
      q: "How long does a gift take to make?",
      a: "1–3 business days plus delivery. If the gift is urgent, a keychain prints fastest.",
    },
    {
      q: "I'm not sure which area to pick — what do I do?",
      a: "Message us — we'll help and build a free preview before you order.",
    },
    {
      q: "Can I add a personal message?",
      a: "Yes — a name, date, coordinates or short phrase can go on the back of a keychain or the base of a map.",
    },
    {
      q: "How much does it cost?",
      a: "A keychain from ≈€3, a magnet, a shelf 3D map from ≈€6 to €13 depending on size.",
    },
  ];
}

// ── Брелок × місто ────────────────────────────────────────────────────────
export function brelokCityCopy(
  cl: ContentLocale,
  name: string,
  slug: string,
  facts: CityFacts | undefined,
): CityLandingCopy {
  if (cl === "uk") {
    const river = facts?.river.uk;
    const landmark = facts?.landmark.uk;
    const intro1 = pickVariant(slug, [
      `Брелок-мапа — це жетон 55×30 мм, на якому рельєфом надруковано обраний район міста (${name}): вулиці, парки, вода. На звороті — власний напис: ім'я, дата чи координати.`,
      `Маленька мапа, що завжди з ключами: брелок 55×30 мм з рельєфним відбитком району міста (${name}). Вулиці читаються пальцями, а на звороті можна додати власний напис.`,
      `Носити з собою рідний район (${name}) — реально: брелок-мапа 55×30 мм передає вулиці, парки та воду рельєфом 3D-друку, з персональним написом на звороті.`,
    ]);
    const intro2 =
      river && landmark
        ? `Для цього міста добре працюють ділянки біля води (${river}) та навколо візитівки (${landmark}) — такі райони впізнаються на брелоку з першого погляду.`
        : `Найкраще виглядають ділянки зі змішаною забудовою: трохи вулиць, парк або вода — так район впізнається з першого погляду.`;
    return {
      title: `Брелок з картою міста — ${name} | 3D-друк від 120 ₴`,
      description: `Брелок-мапа 55×30 мм з районом міста (${name}): вулиці й парки рельєфом, власний напис на звороті. 3D-друк з Eco PLA від 120 ₴, доставка по Україні та ЄС.`,
      h1: `Брелок з картою міста — ${name}`,
      intro: [intro1, intro2],
      sections: [
        {
          h2: "Як створити брелок зі своїм районом",
          p: [
            `Відкрийте конструктор, знайдіть місто (${name}) і пересуньте рамку на потрібний район: рідну вулицю, двір дитинства, місце знайомства. Сервіс сам збере модель з даних OpenStreetMap — з реальними вулицями, парками й водою.`,
            "Можна додати напис на звороті — ім'я, дату, координати чи назву міста. А якщо у вас є GPX-трек пробіжки або походу — завантажте його, і маршрут ляже рельєфною лінією поверх карти.",
          ],
        },
        {
          h2: "Матеріал, якість і терміни",
          p: [
            "Друкуємо з біопластику Eco PLA: легкий, приємний на дотик, витримує щоденне носіння з ключами. Лінії вулиць друкуються з роздільністю 0,4 мм — навіть провулки лишаються читабельними.",
            "Виготовлення 1–3 робочі дні. Доставка Новою Поштою по Україні або Nova Post / Meest у країни ЄС. Оплата карткою онлайн або при отриманні.",
          ],
        },
        {
          h2: "Кому дарують такий брелок",
          p: [
            `Тим, хто переїхав і сумує за домом. Парі — два брелоки-серця з районами двох людей, що з'єднуються як пазл. Бігунам — з треком улюбленого маршруту. Колезі, що змінює місто, — з районом офісу. Брелок з картою міста (${name}) — недорогий подарунок, якого точно ні в кого немає.`,
          ],
        },
      ],
    };
  }
  const river = facts?.river.latin;
  const landmark = facts?.landmark.latin;
  const intro1 = pickVariant(slug, [
    `A map keychain is a 55×30 mm tag with your chosen district of ${name} printed in relief: streets, parks, water. On the back — your own text: a name, a date or coordinates.`,
    `A little map that lives on your keys: a 55×30 mm keychain with a relief print of a ${name} district. Streets read under your fingers, and the back takes a custom engraving.`,
    `Carrying your home district of ${name} with you is real: the 55×30 mm map keychain renders streets, parks and water in 3D-printed relief, with a personal text on the back.`,
  ]);
  const intro2 =
    river && landmark
      ? `For this city, areas near the water (${river}) and around the landmark (${landmark}) work especially well — such districts are recognizable at first glance.`
      : `Mixed areas look best: some streets, a park or water — the district stays recognizable at first glance.`;
  return {
    title: `City map keychain — ${name} | 3D-printed from ≈€3`,
    description: `55×30 mm map keychain with a ${name} district: streets and parks in relief, custom text on the back. Eco PLA 3D print, shipping to Ukraine & EU.`,
    h1: `City map keychain — ${name}`,
    intro: [intro1, intro2],
    sections: [
      {
        h2: "How to create a keychain with your district",
        p: [
          `Open the builder, find ${name} and move the frame over the district you want: your home street, a childhood backyard, the place you met. The service assembles the model from OpenStreetMap data — real streets, parks and water.`,
          "Add text on the back — a name, a date, coordinates. If you have a GPX track of a run or a hike, upload it and the route becomes a relief line over the map.",
        ],
      },
      {
        h2: "Material, quality and lead time",
        p: [
          "Printed in Eco PLA bioplastic: light, pleasant to touch, fine for daily use with keys. Street lines print at 0.4 mm resolution, so even small lanes stay readable.",
          "Production 1–3 business days. Delivery by Nova Poshta across Ukraine or Nova Post / Meest to the EU. Card payment online or on delivery.",
        ],
      },
      {
        h2: "Who gets one as a gift",
        p: [
          `People who moved away and miss home. Couples — two heart keychains with each person's district that connect like a puzzle. Runners — with a favorite route's track. A colleague changing cities — with the office block. A ${name} map keychain is an inexpensive gift no one else has.`,
        ],
      },
    ],
  };
}

// ── Подарунок × місто ─────────────────────────────────────────────────────
export function giftCityCopy(
  cl: ContentLocale,
  name: string,
  slug: string,
  facts: CityFacts | undefined,
): CityLandingCopy {
  if (cl === "uk") {
    const pop = facts ? nfUk.format(facts.population) : null;
    const landmark = facts?.landmark.uk;
    const intro1 = pickVariant(slug, [
      `Шукаєте подарунок, пов'язаний з містом (${name})? Персональна 3D-мапа — це фізична модель обраного району: будинки з реальними висотами, вулиці, парки, вода. Річ, яка існує в одному екземплярі.`,
      `Подарунок з характером міста (${name}): 3D-мапа району, який щось означає — двір дитинства, вулиця першого побачення, дім, куди щойно переїхали. Друкуємо модель з реальними будинками й вулицями.`,
      `Найкращі подарунки — про пам'ять місця. 3D-мапа міста (${name}) перетворює улюблений район на фізичну модель: реальна забудова, парки й вода, обрані саме вами.`,
    ]);
    const intro2 = landmark
      ? `Люди впізнають свій район за секунду — «це ж наш будинок!». ${pop ? `У місті мешкає ${pop} людей, ` : ""}і в кожного — свій куточок: хтось обере ділянку біля візитівки міста (${landmark}), хтось — власний двір.`
      : "Люди впізнають свій район за секунду — «це ж наш будинок!» — і саме ця мить робить подарунок пам'ятним.";
    return {
      title: `Подарунок з міста ${name}: персональна 3D-мапа від 120 ₴`,
      description: `Ідея подарунка (${name}): 3D-мапа району від 250 ₴, брелок-мапа від 120 ₴, магніт 150 ₴. Річниця, новосілля, день народження. Виготовлення 1–3 дні.`,
      h1: `Подарунок з міста — ${name}`,
      intro: [intro1, intro2],
      sections: [
        {
          h2: "Під яку нагоду",
          p: [
            "Річниця — район, де ви познайомились. Новосілля — новий район на полицю нової квартири. День народження — рідне місто людини, що переїхала. Випуск — університетський квартал. Колезі на прощання — район офісу.",
            "Не знаєте, яку ділянку обрати? Напишіть нам — підкажемо, як виглядатиме район, і зберемо превʼю безкоштовно.",
          ],
        },
        {
          h2: "Формати під різний бюджет",
          p: [
            `Брелок-мапа (від 120 ₴) — недорогий знак уваги з написом на звороті. Магніт на холодильник (150 ₴) — щоденне нагадування про місто (${name}). 3D-мапа на полицю (від 250 ₴ за 5,5 см до 550 ₴ за 15 см) — повноцінний інтер'єрний подарунок; за бажанням — з рельєфом місцевості (+60 ₴).`,
          ],
        },
        {
          h2: "Як замовити за 5 хвилин",
          p: [
            `Оберіть ділянку на карті в конструкторі — сервіс збере модель автоматично за 2–4 хвилини. Друкуємо з біопластику Eco PLA і надсилаємо Новою Поштою; можна замовити й цифровий файл для власного друку. Виготовлення 1–3 робочі дні — встигнути до дати легко.`,
          ],
        },
      ],
    };
  }
  const pop = facts ? nfEn.format(facts.population) : null;
  const landmark = facts?.landmark.latin;
  const intro1 = pickVariant(slug, [
    `Looking for a gift tied to ${name}? A personal 3D map is a physical model of a chosen district: buildings with real heights, streets, parks, water. A one-of-a-kind object.`,
    `A gift with the character of ${name}: a 3D map of a district that means something — a childhood backyard, the street of a first date, a new home. We print the model with real buildings and streets.`,
    `The best gifts are about the memory of a place. A 3D map of ${name} turns a favorite district into a physical model: real buildings, parks and water, chosen by you.`,
  ]);
  const intro2 = landmark
    ? `People recognize their neighborhood in a second — "that's our house!". ${pop ? `${pop} people live in the city, ` : ""}and everyone has their own corner: some pick the area around the landmark (${landmark}), others — their own backyard.`
    : `People recognize their neighborhood in a second — "that's our house!" — and that moment makes the gift memorable.`;
  return {
    title: `A gift from ${name}: a personal 3D map from ≈€3`,
    description: `Gift ideas from ${name}: district 3D map from ≈€6, map keychain from ≈€3, fridge magnet. Anniversary, housewarming, birthday. Made in 1–3 days.`,
    h1: `A gift from the city — ${name}`,
    intro: [intro1, intro2],
    sections: [
      {
        h2: "For which occasion",
        p: [
          "Anniversary — the district where you met. Housewarming — the new neighborhood for the new flat's shelf. Birthday — the hometown of someone who moved away. Graduation — the campus quarter. A leaving colleague — the office block.",
          "Not sure which area to pick? Message us — we'll help and build a free preview.",
        ],
      },
      {
        h2: "Formats for any budget",
        p: [
          `Map keychain (from ≈€3) — a small token with text on the back. Fridge magnet — a daily reminder of ${name}. Shelf 3D map (≈€6–13 depending on size) — a real interior piece, with terrain relief if you like.`,
        ],
      },
      {
        h2: "Order in 5 minutes",
        p: [
          "Pick the area in the builder — the model is assembled automatically in 2–4 minutes. We print in Eco PLA and ship across Ukraine and to the EU; a digital file for self-printing is also available. Production takes 1–3 business days.",
        ],
      },
    ],
  };
}

// ── Лендінги під нагоду (/podarunok/[slug], slug ∉ міста) ────────────────
export interface OccasionPage {
  slug: string;
  ctaHref: string;
  content: Record<ContentLocale, CityLandingCopy>;
}

export const OCCASION_PAGES: OccasionPage[] = [
  {
    slug: "na-richnytsyu",
    ctaHref: "/create",
    content: {
      uk: {
        title: "Подарунок на річницю: 3D-мапа місця, де все почалось",
        description:
          "Ідея подарунка на річницю стосунків чи весілля: 3D-мапа району, де ви познайомились, або пара брелоків-сердець, що з'єднуються як пазл. Від 120 ₴.",
        h1: "Подарунок на річницю: мапа місця, де все почалось",
        intro: [
          "У кожної пари є свої координати: лавка в парку, кав'ярня на розі, зупинка, де вперше зустрілись. Подарунок на річницю, який працює найкраще, — не річ з полиці, а пам'ять місця: 3D-мапа району, де почалась ваша історія.",
          "Це модель з реальними будинками, вулицями й парками — саме тими, якими ви ходили. Така річ існує в одному екземплярі, бо ділянку обираєте ви.",
        ],
        sections: [
          {
            h2: "Два формати для пари",
            p: [
              "3D-мапа на полицю (від 250 ₴) — район першого побачення чи першої спільної квартири, за бажанням з рельєфом. Можна замовити з підписаною датою.",
              "Пара брелоків-«сердець» (від 120 ₴ за штуку) — два райони двох людей, які з'єднуються в одне серце, як пазл. Працює особливо зворушливо для пар з різних міст.",
            ],
          },
          {
            h2: "Як встигнути до дати",
            p: [
              "Виготовлення займає 1–3 робочі дні плюс доставка Новою Поштою. Оберіть ділянку в конструкторі — модель збереться автоматично, і ви одразу побачите превʼю в 3D.",
            ],
          },
        ],
      },
      en: {
        title: "Anniversary gift: a 3D map of where it all began",
        description:
          "An anniversary gift idea: a 3D map of the district where you met, or a pair of heart keychains that connect like a puzzle. From ≈€3.",
        h1: "Anniversary gift: a map of where it all began",
        intro: [
          "Every couple has its coordinates: a bench in the park, a corner café, the stop where you first met. The anniversary gift that works best isn't an off-the-shelf object — it's the memory of a place: a 3D map of the district where your story began.",
          "It's a model with real buildings, streets and parks — the very ones you walked. One of a kind, because you choose the exact area.",
        ],
        sections: [
          {
            h2: "Two formats for a couple",
            p: [
              "A shelf 3D map (from ≈€6) — the district of a first date or a first shared flat, with terrain relief if you like.",
              "A pair of heart keychains (from ≈€3 each) — two districts of two people that connect into one heart, like a puzzle. Especially touching for couples from different cities.",
            ],
          },
          {
            h2: "Making it in time",
            p: [
              "Production takes 1–3 business days plus delivery. Pick the area in the builder — the model assembles automatically and you see a 3D preview right away.",
            ],
          },
        ],
      },
    },
  },
  {
    slug: "na-den-narodzhennya",
    ctaHref: "/create",
    content: {
      uk: {
        title: "Подарунок на день народження: персональна 3D-мапа",
        description:
          "Що подарувати на день народження людині, в якої все є: 3D-мапа рідного району, брелок з маршрутом чи магніт з улюбленим містом. Від 120 ₴, 1–3 дні.",
        h1: "Подарунок на день народження, якого ні в кого немає",
        intro: [
          "Найскладніше — дарувати людям, у яких «все є». Чергова свічка забудеться за тиждень; працює те, що має особисте значення. Персональна 3D-мапа — це шматочок міста, з яким пов'язана історія іменинника: двір дитинства, район першої квартири, місто, куди мріє повернутись.",
        ],
        sections: [
          {
            h2: "Ідеї під різних людей",
            p: [
              "Тому, хто переїхав, — мапа рідного міста. Бігуну чи велосипедисту — брелок з GPX-треком улюбленого маршруту. Мандрівникові — мапа міста мрії. Батькам — двір, де виросли діти. Другу-новоселу — новий район.",
            ],
          },
          {
            h2: "Бюджет і терміни",
            p: [
              "Брелок-мапа — від 120 ₴, магніт — 150 ₴, 3D-мапа на полицю — від 250 до 550 ₴ залежно від розміру. Виготовлення 1–3 робочі дні, доставка Новою Поштою по Україні та у країни ЄС.",
            ],
          },
        ],
      },
      en: {
        title: "Birthday gift: a personal 3D city map",
        description:
          "What to give someone who has everything: a 3D map of their home district, a route keychain or a city magnet. From ≈€3, made in 1–3 days.",
        h1: "A birthday gift no one else has",
        intro: [
          "The hardest gifts are for people who «have everything». What works is personal meaning: a 3D map is a piece of the city tied to the birthday person's story — a childhood backyard, the first flat's district, the city they dream of returning to.",
        ],
        sections: [
          {
            h2: "Ideas for different people",
            p: [
              "For someone who moved away — a map of their hometown. For a runner or cyclist — a keychain with a GPX track. For a traveler — the dream city. For parents — the backyard where the kids grew up. For a friend in a new flat — the new district.",
            ],
          },
          {
            h2: "Budget and lead time",
            p: [
              "Map keychain from ≈€3, magnet, shelf 3D map ≈€6–13 depending on size. Production 1–3 business days, delivery to Ukraine and the EU.",
            ],
          },
        ],
      },
    },
  },
  {
    slug: "na-novosillya",
    ctaHref: "/create",
    content: {
      uk: {
        title: "Подарунок на новосілля: 3D-мапа нового району",
        description:
          "Оригінальний подарунок на новосілля: 3D-мапа району, куди щойно переїхали друзі — з реальними будинками й вулицями. Від 250 ₴, виготовлення 1–3 дні.",
        h1: "Подарунок на новосілля: новий район на полиці",
        intro: [
          "Новосілля — це початок нової глави, і найкращий подарунок — той, що цю главу відкриває. 3D-мапа нового району показує дім у контексті: вулиці, парки поруч, річку за квартал. Господарі щодня бачитимуть свій новий світ на полиці — і власний будинок на ньому.",
        ],
        sections: [
          {
            h2: "Чому це краще за класичні подарунки",
            p: [
              "Посуд і текстиль губляться серед десятків однакових коробок. Мапа нового району — єдина у своєму роді: така сама є хіба що в сусідів, і то якщо вони теж замовлять. Це подарунок і про дім, і про місце — обидва сенси новосілля одразу.",
              "Порада: оберіть ділянку так, щоб будинок новоселів був ближче до центру мапи — його одразу шукатимуть очима.",
            ],
          },
          {
            h2: "Формат і терміни",
            p: [
              "Найпопулярніший розмір для подарунка — M (8 см, 350 ₴) або L (11 см, 450 ₴). Виготовлення 1–3 робочі дні; якщо новосілля вже скоро — замовте брелок (від 120 ₴), він друкується найшвидше.",
            ],
          },
        ],
      },
      en: {
        title: "Housewarming gift: a 3D map of the new neighborhood",
        description:
          "An original housewarming gift: a 3D map of the district your friends just moved to — real buildings and streets. From ≈€6, made in 1–3 days.",
        h1: "Housewarming gift: the new neighborhood on a shelf",
        intro: [
          "A housewarming is the start of a new chapter, and the best gift opens that chapter. A 3D map of the new district shows the home in context: the streets, the parks nearby, the river a block away — with their own house on it.",
        ],
        sections: [
          {
            h2: "Why it beats classic gifts",
            p: [
              "Dishes and textiles get lost among a dozen identical boxes. A map of the new neighborhood is one of a kind — a gift about both the home and the place at once.",
              "Tip: choose the area so the new home sits near the center of the map — it's the first thing everyone looks for.",
            ],
          },
          {
            h2: "Format and lead time",
            p: [
              "The most popular gift sizes are M (8 cm) and L (11 cm). Production 1–3 business days; if the party is very soon, a keychain (from ≈€3) prints fastest.",
            ],
          },
        ],
      },
    },
  },
  {
    slug: "dlya-pary",
    ctaHref: "/keychains",
    content: {
      uk: {
        title: "Подарунок для пари: брелоки-серця з районами двох людей",
        description:
          "Пара брелоків-«сердець», що з'єднуються як пазл: район одного + район іншої. Подарунок для пари на річницю чи День закоханих. Від 120 ₴ за брелок.",
        h1: "Подарунок для пари: два райони — одне серце",
        intro: [
          "Пара брелоків-«сердець» — це дві половинки, що з'єднуються в одне серце, як пазл. На одній половинці — район одного, на другій — район іншої. Разом вони складаються у спільну історію: два міста, дві вулиці, одна пара.",
          "Особливо зворушливо працює для пар, що познайомились у різних містах або живуть на відстані: кожен носить свою половинку, і вони пасують тільки одна до одної.",
        ],
        sections: [
          {
            h2: "Як це замовити",
            p: [
              "У конструкторі брелоків оберіть шаблон «серце-пара», потім першу ділянку і другу ділянку. На звороті кожної половинки можна додати напис — ім'я, дату, координати.",
              "Виготовлення 1–3 робочі дні, разом — від 240 ₴ за пару. Доставка Новою Поштою або у країни ЄС.",
            ],
          },
        ],
      },
      en: {
        title: "Couple gift: heart keychains with two people's districts",
        description:
          "A pair of heart keychains that connect like a puzzle: one person's district + the other's. An anniversary or Valentine's gift. From ≈€3 per keychain.",
        h1: "A couple's gift: two districts — one heart",
        intro: [
          "A pair of heart keychains is two halves that connect into one heart, like a puzzle. One half carries one person's district, the other — the partner's. Together they tell a shared story: two cities, two streets, one couple.",
          "It works especially well for couples who met in different cities or live apart: each carries their half, and they only fit each other.",
        ],
        sections: [
          {
            h2: "How to order",
            p: [
              "In the keychain builder pick the «heart pair» template, then the first area and the second area. Each half takes a custom text on the back — a name, a date, coordinates.",
              "Production 1–3 business days, from ≈€6 per pair. Delivery to Ukraine and the EU.",
            ],
          },
        ],
      },
    },
  },
  {
    slug: "korporatyvnyi-podarunok",
    ctaHref: "/create",
    content: {
      uk: {
        title: "Корпоративні подарунки: 3D-мапи та брелоки з районом офісу",
        description:
          "Корпоративний подарунок зі змістом: брелоки з районом офісу для команди, 3D-мапа міста для партнерів. Тираж від 5 шт., персоналізація написом.",
        h1: "Корпоративні подарунки з мапою: офіс, місто, команда",
        intro: [
          "Корпоративні подарунки часто безликі — ще один блокнот, ще одна чашка. Мапа працює інакше: брелок з районом офісу або 3D-мапа міста, де виросла компанія, — це історія про «наше місце», яку приємно тримати в руках.",
        ],
        sections: [
          {
            h2: "Сценарії для бізнесу",
            p: [
              "Команді — брелоки з районом офісу (від 120 ₴/шт., однаковий макет — швидший друк). Колезі, що йде, — мапа району офісу з підписаною датою. Партнерам з інших міст — мапа вашого міста як фірмовий сувенір. Релокованій команді — мапи рідних міст кожного.",
            ],
          },
          {
            h2: "Тиражі та персоналізація",
            p: [
              "Друкуємо від 1 штуки, для тиражів від 5 шт. — узгодимо терміни та вартість окремо. На звороті брелоків можна додати назву компанії чи дату події. Напишіть нам — підкажемо формат під ваш привід і бюджет.",
            ],
          },
        ],
      },
      en: {
        title: "Corporate gifts: 3D maps and keychains with the office district",
        description:
          "A corporate gift with meaning: office-district keychains for the team, a city 3D map for partners. Runs from 5 pcs, personalized text.",
        h1: "Corporate gifts with a map: office, city, team",
        intro: [
          "Corporate gifts are often faceless — another notebook, another mug. A map works differently: a keychain with the office district or a 3D map of the company's home city is a story about «our place» that's pleasant to hold.",
        ],
        sections: [
          {
            h2: "Business scenarios",
            p: [
              "For the team — office-district keychains (from ≈€3 each, same layout prints faster). For a leaving colleague — the office district with a date. For out-of-town partners — your city as a branded souvenir. For a relocated team — each person's hometown.",
            ],
          },
          {
            h2: "Runs and personalization",
            p: [
              "We print from 1 piece; for runs of 5+ we agree timing and pricing individually. Keychain backs can carry the company name or an event date. Message us — we'll suggest a format for your occasion and budget.",
            ],
          },
        ],
      },
    },
  },
];

export const OCCASION_BY_SLUG: Record<string, OccasionPage> = Object.fromEntries(
  OCCASION_PAGES.map((o) => [o.slug, o]),
);

// ── Райони міст (/maps/[city]/[district]) ─────────────────────────────────
// Ручний контент під кожен MAP_TEMPLATES-запис (12 районів) — найточніший
// рівень запиту («3d мапа поділу», «мапа площі ринок львів»), найвища конверсія.
export interface DistrictPage {
  templateId: string;   // = MapTemplate.id з lib/templates.ts
  citySlug: string;      // = CityPage.slug
  slug: string;          // латиницею, унікальний у межах міста
  enName: string;        // назва району англійською (для non-uk locale H1)
  content: Record<ContentLocale, CityLandingCopy>;
}

export const DISTRICT_PAGES: DistrictPage[] = [
  {
    templateId: "kyiv-podil",
    citySlug: "kyiv",
    slug: "podil",
    enName: "Podil",
    content: {
      uk: {
        title: "3D-мапа Подолу (Київ): Андріївський узвіз рельєфом",
        description: "3D-мапа Подолу — старого серця Києва: звивисті вулиці, Андріївський узвіз, Контрактова площа. Готовий шаблон, від 250 ₴.",
        h1: "3D-мапа Подолу",
        intro: [
          "Поділ — найстаріший район Києва, де вулиці досі йдуть так, як пролягали century тому: криво, вузько, з підйомом до Андріївського узвозу. На 3D-мапі ця щільна історична забудова виглядає особливо ефектно — кожен будинок читається окремо.",
          "Готовий шаблон району вже налаштований: правильна ділянка, стиль «повна деталізація», розмір 8 см. Достатньо натиснути «Створити» — і за кілька хвилин модель готова до перегляду.",
        ],
        sections: [
          {
            h2: "Що видно на моделі",
            p: ["Контрактова площа, звивисті вулички до Андріївської церкви, щільні квартали старої забудови. Рельєф не увімкнено за замовчуванням — Поділ рівнинний, а ось сусідній Замковий пагорб уже вимагав би рельєфу."],
          },
        ],
      },
      en: {
        title: "3D map of Podil (Kyiv): the Andriivskyi Descent in relief",
        description: "A 3D map of Podil, Kyiv's oldest district: winding streets, the Andriivskyi Descent, Kontraktova Square. Ready template, from ≈€6.",
        h1: "3D map of Podil",
        intro: [
          "Podil is Kyiv's oldest district, where streets still run as they did centuries ago: crooked, narrow, climbing toward the Andriivskyi Descent. On a 3D map this dense historic fabric looks especially striking — every building reads on its own.",
          "The ready-made district template is already tuned: the right area, 'full detail' style, 8 cm size. Just press 'Create' and the model is ready to view in minutes.",
        ],
        sections: [
          { h2: "What the model shows", p: ["Kontraktova Square, winding lanes up to Andriivska Church, dense old-town blocks. Relief is off by default — Podil is flat; the neighboring Castle Hill would call for it."] },
        ],
      },
    },
  },
  {
    templateId: "kyiv-pechersk",
    citySlug: "kyiv",
    slug: "pechersk",
    enName: "Pechersk",
    content: {
      uk: {
        title: "3D-мапа Печерська (Київ) з рельєфом лаврських пагорбів",
        description: "3D-мапа Печерська: Лавра, Маріїнський парк, парадні проспекти — з реальним рельєфом дніпровських схилів. Розмір 11 см, від 450 ₴.",
        h1: "3D-мапа Печерська",
        intro: [
          "Печерськ — це пагорби. Лаврські схили, Маріїнський парк над кручею, урядові проспекти нагорі — перепад висот тут один з найпомітніших у Києві. Тому шаблон району одразу йде з увімкненим рельєфом і розміром 11 см, щоб перепади читались пальцями.",
          "Модель показує парадну частину Печерська: широкі проспекти, зелень парку, характерний силует на схилі.",
        ],
        sections: [
          { h2: "Чому саме з рельєфом", p: ["Без рельєфу Печерськ втрачає половину характеру — саме перепад від Дніпра до верхнього міста робить район впізнаваним. Розмір 11 см обраний свідомо: на 8 см дрібні деталі схилу губляться."] },
        ],
      },
      en: {
        title: "3D map of Pechersk (Kyiv) with real hill relief",
        description: "A 3D map of Pechersk: the Lavra, Mariinsky Park, government avenues — with real Dnipro-slope relief. 11 cm size, from ≈€10.",
        h1: "3D map of Pechersk",
        intro: [
          "Pechersk is hills. The Lavra slopes, Mariinsky Park over the cliff, government avenues above — one of Kyiv's most noticeable elevation drops. That's why this district template ships with relief on by default, sized at 11 cm so the drops read under your fingers.",
          "The model shows Pechersk's grand face: wide avenues, park greenery, a distinctive silhouette on the slope.",
        ],
        sections: [
          { h2: "Why relief matters here", p: ["Without relief, Pechersk loses half its character — the drop from the Dnipro to the upper city is what makes it recognizable. 11 cm is deliberate: at 8 cm the slope's fine detail gets lost."] },
        ],
      },
    },
  },
  {
    templateId: "kyiv-khreshchatyk",
    citySlug: "kyiv",
    slug: "khreshchatyk",
    enName: "Khreshchatyk",
    content: {
      uk: {
        title: "3D-мапа Хрещатика та Майдану Незалежності (Київ)",
        description: "3D-мапа центральної вулиці Києва — Хрещатика і Майдану Незалежності. Готовий шаблон, стиль «повна деталізація», від 250 ₴.",
        h1: "3D-мапа Хрещатика",
        intro: [
          "Хрещатик — головна вісь Києва: від Бессарабки до Європейської площі, з Майданом Незалежності посередині. На 3D-мапі впізнається одразу — характерна форма вулиці й площі не сплутати ні з чим.",
          "Готовий шаблон охоплює саме цю ділянку в масштабі, де і вулиця, і прилеглі квартали лишаються деталізованими.",
        ],
        sections: [
          { h2: "Кому підходить", p: ["Класичний вибір для тих, хто хоче впізнаване «обличчя» Києва, а не конкретний двір: Майдан на моделі читають усі, хто там бував."] },
        ],
      },
      en: {
        title: "3D map of Khreshchatyk and Independence Square (Kyiv)",
        description: "A 3D map of Kyiv's main street — Khreshchatyk and Independence Square. Ready template, 'full detail' style, from ≈€6.",
        h1: "3D map of Khreshchatyk",
        intro: [
          "Khreshchatyk is Kyiv's main axis, from Bessarabska to European Square, with Independence Square in the middle. On a 3D map it's instantly recognizable — the street's and square's shape can't be mistaken for anywhere else.",
          "The ready template covers exactly this area at a scale where both the street and the surrounding blocks stay detailed.",
        ],
        sections: [
          { h2: "Who it suits", p: ["A classic pick for anyone who wants Kyiv's recognizable 'face' rather than a specific backyard — everyone who's been there reads the Maidan on the model."] },
        ],
      },
    },
  },
  {
    templateId: "lviv-rynok",
    citySlug: "lviv",
    slug: "rynok",
    enName: "Rynok Square",
    content: {
      uk: {
        title: "3D-мапа площі Ринок (Львів): ратуша й бруківка",
        description: "3D-мапа площі Ринок у Львові — ратуша, бруківка, щільна сітка кварталів старого міста. Бестселер, від 250 ₴.",
        h1: "3D-мапа площі Ринок",
        intro: [
          "Площа Ринок — серце старого Львова: ратуша посередині, кам'яниці по периметру, вузькі вулички навсібіч. Це найпопулярніший шаблон району в конструкторі — щільна середньовічна забудова робить модель візуально насиченою навіть у компактному розмірі.",
          "Готовий шаблон одразу центрує ділянку на площі й підбирає стиль «повна деталізація».",
        ],
        sections: [
          { h2: "Що робить цю модель особливою", p: ["Радіальна сітка вуличок навколо площі — рідкісний для 3D-мап візерунок, який виглядає як справжня мініатюра середньовічного міста."] },
        ],
      },
      en: {
        title: "3D map of Rynok Square (Lviv): the town hall and cobblestones",
        description: "A 3D map of Lviv's Rynok Square — the town hall, cobblestones, dense old-town blocks. A bestseller, from ≈€6.",
        h1: "3D map of Rynok Square",
        intro: [
          "Rynok Square is the heart of old Lviv: the town hall in the middle, townhouses around the perimeter, narrow lanes radiating out. It's the most popular district template in the builder — the dense medieval fabric makes the model visually rich even at a compact size.",
          "The ready template centers the area on the square and picks 'full detail' style automatically.",
        ],
        sections: [
          { h2: "What makes this model special", p: ["The radial grid of lanes around the square is a pattern rare in 3D maps — it reads like a genuine medieval-town miniature."] },
        ],
      },
    },
  },
  {
    templateId: "lviv-citadel",
    citySlug: "lviv",
    slug: "tsytadel",
    enName: "the Citadel",
    content: {
      uk: {
        title: "3D-мапа Цитаделі (Львів) з рельєфом пагорба",
        description: "3D-мапа району Цитадель у Львові: пагорб з парком, серпантин історичних вулиць, рельєф увімкнено. Від 250 ₴.",
        h1: "3D-мапа Цитаделі",
        intro: [
          "Цитадель — пагорб над центром Львова, обвитий серпантином вулиць і накритий парком. Рельєф тут не опція, а суть району: без нього зникає сама причина, чому ця ділянка цікава.",
          "Шаблон одразу вмикає рельєф і центрує ділянку так, щоб і підйом, і забудова навколо нього лишились у кадрі.",
        ],
        sections: [
          { h2: "Для кого цей район", p: ["Для тих, хто любить Львів не за листівковий центр, а за приховані куточки — пагорб з парком, який мало хто фотографує, але який добре знають місцеві."] },
        ],
      },
      en: {
        title: "3D map of the Citadel (Lviv) with hill relief",
        description: "A 3D map of Lviv's Citadel district: a park-covered hill, winding historic streets, relief on. From ≈€6.",
        h1: "3D map of the Citadel",
        intro: [
          "The Citadel is a hill over central Lviv, wrapped in winding streets and topped with a park. Relief here isn't an option — it's the point of the district; without it, the reason this area is interesting disappears.",
          "The template turns relief on by default and centers the area so both the climb and the surrounding streets stay in frame.",
        ],
        sections: [
          { h2: "Who this district suits", p: ["For those who love Lviv not for its postcard center but for its hidden corners — a park-topped hill few photograph but locals know well."] },
        ],
      },
    },
  },
  {
    templateId: "odesa-deribasivska",
    citySlug: "odesa",
    slug: "derybasivska",
    enName: "Deribasivska Street",
    content: {
      uk: {
        title: "3D-мапа Дерибасівської (Одеса): серце міста",
        description: "3D-мапа Дерибасівської вулиці в Одесі — Міський сад, бульвари, серце міста. Бестселер, від 250 ₴.",
        h1: "3D-мапа Дерибасівської",
        intro: [
          "Дерибасівська — вулиця, з якою асоціюється вся Одеса: Міський сад, кав'ярні під платанами, бульварна забудова. Шаблон району охоплює саме цю пішохідну частину міста.",
          "Готова ділянка й стиль «повна деталізація» — модель одразу передає атмосферу одеського центру.",
        ],
        sections: [
          { h2: "Найпопулярніший вибір з Одеси", p: ["Саме ця ділянка — найчастіший запит на мапу Одеси: впізнавана без підпису, компактна, з мальовничою забудовою."] },
        ],
      },
      en: {
        title: "3D map of Deribasivska Street (Odesa): the city's heart",
        description: "A 3D map of Odesa's Deribasivska Street — the City Garden, boulevards, the heart of the city. A bestseller, from ≈€6.",
        h1: "3D map of Deribasivska Street",
        intro: [
          "Deribasivska is the street Odesa is known for: the City Garden, cafés under plane trees, boulevard architecture. The district template covers exactly this pedestrian part of the city.",
          "A ready area and 'full detail' style — the model captures the atmosphere of central Odesa instantly.",
        ],
        sections: [
          { h2: "Odesa's most popular pick", p: ["This exact area is the most requested map of Odesa: recognizable without a caption, compact, with picturesque architecture."] },
        ],
      },
    },
  },
  {
    templateId: "odesa-prymorsky",
    citySlug: "odesa",
    slug: "prymorskyi",
    enName: "Prymorskyi District",
    content: {
      uk: {
        title: "3D-мапа Приморського району (Одеса): сходи й порт",
        description: "3D-мапа Приморського району Одеси: Дюк, схил до Потьомкінських сходів, морський фасад. Рельєф увімкнено, від 250 ₴.",
        h1: "3D-мапа Приморського району",
        intro: [
          "Приморський бульвар з пам'ятником Дюку, схил до Потьомкінських сходів і порту внизу — ділянка, де рельєф справді щось показує: перепад від бульвару до моря.",
          "Шаблон одразу вмикає рельєф, щоб цей схил читався на моделі, а не губився на пласкій мапі.",
        ],
        sections: [
          { h2: "Чому саме тут рельєф", p: ["Одеса загалом рівнинна, але цей конкретний схил до порту — виняток, і саме він робить Приморський район візуально цікавим у 3D."] },
        ],
      },
      en: {
        title: "3D map of the Prymorskyi District (Odesa): the stairs and the port",
        description: "A 3D map of Odesa's Prymorskyi District: the Duke statue, the slope to the Potemkin Stairs, the sea facade. Relief on, from ≈€6.",
        h1: "3D map of the Prymorskyi District",
        intro: [
          "The Prymorskyi Boulevard with the Duke statue, the slope down to the Potemkin Stairs and the port below — an area where relief genuinely shows something: the drop from boulevard to sea.",
          "The template turns relief on so this slope reads on the model instead of getting lost on a flat map.",
        ],
        sections: [
          { h2: "Why relief matters here", p: ["Odesa is mostly flat, but this specific slope to the port is the exception — and it's exactly what makes the Prymorskyi District visually interesting in 3D."] },
        ],
      },
    },
  },
  {
    templateId: "kharkiv-svobody",
    citySlug: "kharkiv",
    slug: "svobody",
    enName: "Svobody Square",
    content: {
      uk: {
        title: "3D-мапа площі Свободи (Харків): Держпром і проспекти",
        description: "3D-мапа площі Свободи в Харкові: Держпром і промениста сітка проспектів центру. Розмір 11 см, від 450 ₴.",
        h1: "3D-мапа площі Свободи",
        intro: [
          "Площа Свободи — одна з найбільших міських площ Європи, а Держпром на ній — символ конструктивізму, який неможливо сплутати ні з чим іншим. Промениста сітка проспектів навколо робить модель геометрично видовищною.",
          "Через масштаб площі шаблон одразу йде з розміром 11 см — на менших розмірах геометрія проспектів втрачає чіткість.",
        ],
        sections: [
          { h2: "Що впізнається на моделі", p: ["Силует Держпрому — найвпізнаваніша будівля Харкова — і сама форма площі, яку видно тільки з висоти пташиного польоту чи на 3D-моделі."] },
        ],
      },
      en: {
        title: "3D map of Svobody Square (Kharkiv): Derzhprom and the avenues",
        description: "A 3D map of Kharkiv's Svobody Square: Derzhprom and the radiant grid of central avenues. 11 cm size, from ≈€10.",
        h1: "3D map of Svobody Square",
        intro: [
          "Svobody Square is one of Europe's largest city squares, and Derzhprom on it is a constructivist landmark unlike anything else. The radiating grid of avenues around it makes the model geometrically striking.",
          "Given the square's scale, the template ships at 11 cm — smaller sizes lose the avenue geometry's crispness.",
        ],
        sections: [
          { h2: "What's recognizable on the model", p: ["Derzhprom's silhouette — Kharkiv's most recognizable building — and the square's own shape, visible only from a bird's-eye view or on a 3D model."] },
        ],
      },
    },
  },
  {
    templateId: "dnipro-naberezhna",
    citySlug: "dnipro",
    slug: "naberezhna",
    enName: "the Embankment",
    content: {
      uk: {
        title: "3D-мапа набережної Дніпра (Дніпро): річка й мости",
        description: "3D-мапа набережної в Дніпрі: широка дуга річки, мости й хвиля висоток. Розмір 11 см, з акцентом на воду.",
        h1: "3D-мапа набережної Дніпра",
        intro: [
          "Широка дуга Дніпра, кілька мостів через неї і характерна хвиля сучасних висоток на березі — ця ділянка про воду й масштаб річки, не про щільну історичну забудову.",
          "Шаблон використовує стиль «природа» з акцентом на воду й рельєф — саме річка тут головна деталь моделі.",
        ],
        sections: [
          { h2: "Чому 11 см", p: ["Ширина Дніпра вимагає більшого розміру моделі, інакше річка займе лише вузьку смужку — на 11 см вода й береги лишаються пропорційними."] },
        ],
      },
      en: {
        title: "3D map of the Dnipro embankment (Dnipro city): the river and bridges",
        description: "A 3D map of the embankment in Dnipro city: the river's wide arc, bridges, a wave of high-rises. 11 cm size, water-focused.",
        h1: "3D map of the Dnipro embankment",
        intro: [
          "The Dnipro river's wide arc, several bridges across it, and a distinctive wave of modern high-rises on the bank — this area is about water and river scale, not dense historic fabric.",
          "The template uses the 'nature' style with an emphasis on water and relief — the river is the model's main feature here.",
        ],
        sections: [
          { h2: "Why 11 cm", p: ["The Dnipro's width needs a bigger model size, or the river becomes a narrow strip — at 11 cm the water and banks stay proportional."] },
        ],
      },
    },
  },
  {
    templateId: "chernivtsi-rez",
    citySlug: "chernivtsi",
    slug: "rezydentsiya",
    enName: "the Metropolitans' Residence",
    content: {
      uk: {
        title: "3D-мапа Резиденції митрополитів (Чернівці)",
        description: "3D-мапа району Резиденції в Чернівцях: кам'яні фасади навколо історичної будівлі, від 250 ₴.",
        h1: "3D-мапа Резиденції митрополитів",
        intro: [
          "Резиденція буковинських митрополитів — будівля зі списку ЮНЕСКО, а навколо неї — щільна кам'яна забудова центру Чернівців. Ділянка компактна й деталізована навіть у розмірі 8 см.",
          "Шаблон центрує ділянку саме на цьому кварталі.",
        ],
        sections: [
          { h2: "Для кого", p: ["Для тих, хто хоче показати архітектурну перлину Чернівців, яку менше знають за межами міста, ніж Львів чи Київ, але яка нічим їм не поступається."] },
        ],
      },
      en: {
        title: "3D map of the Metropolitans' Residence (Chernivtsi)",
        description: "A 3D map of the Residence district in Chernivtsi: stone facades around a historic UNESCO building, from ≈€6.",
        h1: "3D map of the Metropolitans' Residence",
        intro: [
          "The Residence of Bukovinian Metropolitans is a UNESCO-listed building, surrounded by dense stone architecture of central Chernivtsi. The area is compact and detailed even at 8 cm.",
          "The template centers the area right on this quarter.",
        ],
        sections: [
          { h2: "Who it's for", p: ["For those who want to showcase Chernivtsi's architectural gem — less known outside the city than Lviv or Kyiv, but no less striking."] },
        ],
      },
    },
  },
  {
    templateId: "ivano-center",
    citySlug: "ivano-frankivsk",
    slug: "stometrivka",
    enName: "Stometrivka",
    content: {
      uk: {
        title: "3D-мапа Стометрівки (Івано-Франківськ): пішохідний центр",
        description: "3D-мапа пішохідного центру Івано-Франківська — Стометрівка і ратуша на площі Ринок, від 250 ₴.",
        h1: "3D-мапа Стометрівки",
        intro: [
          "Стометрівка — головна пішохідна вулиця Івано-Франківська, що веде до площі Ринок з ратушею. Компактний, щільний і впізнаваний центр міста.",
          "Готовий шаблон охоплює саме цю прогулянкову ділянку.",
        ],
        sections: [
          { h2: "Що на моделі", p: ["Ратуша на площі Ринок і забудова вздовж пішохідної вулиці — символи міста, які одразу впізнають франківці."] },
        ],
      },
      en: {
        title: "3D map of Stometrivka (Ivano-Frankivsk): the pedestrian center",
        description: "A 3D map of Ivano-Frankivsk's pedestrian center — Stometrivka Street and the town hall on Rynok Square, from ≈€6.",
        h1: "3D map of Stometrivka",
        intro: [
          "Stometrivka is Ivano-Frankivsk's main pedestrian street, leading to Rynok Square and its town hall. A compact, dense, recognizable city center.",
          "The ready template covers exactly this walking area.",
        ],
        sections: [
          { h2: "What the model shows", p: ["The town hall on Rynok Square and the buildings along the pedestrian street — symbols locals recognize instantly."] },
        ],
      },
    },
  },
  {
    templateId: "uzhhorod-old",
    citySlug: "uzhhorod",
    slug: "stare-misto",
    enName: "the Old Town",
    content: {
      uk: {
        title: "3D-мапа Старого міста (Ужгород): замок і липова алея",
        description: "3D-мапа старого Ужгорода: набережна Ужа, замок і найдовша липова алея Європи. Стиль «природа», від 250 ₴.",
        h1: "3D-мапа Старого міста",
        intro: [
          "Набережна річки Уж, Ужгородський замок на пагорбі й найдовша безперервна липова алея Європи — ця ділянка про воду й зелень так само, як про історичну забудову.",
          "Шаблон використовує стиль «природа», щоб і річка, і алея дерев були помітні на моделі.",
        ],
        sections: [
          { h2: "Унікальність району", p: ["Липова алея — офіційний рекорд, і на 3D-мапі її довга пряма лінія добре читається поруч із замком на пагорбі."] },
        ],
      },
      en: {
        title: "3D map of the Old Town (Uzhhorod): the castle and the linden alley",
        description: "A 3D map of old Uzhhorod: the Uzh embankment, the castle, Europe's longest linden alley. 'Nature' style, from ≈€6.",
        h1: "3D map of the Old Town",
        intro: [
          "The Uzh river embankment, Uzhhorod Castle on a hill, and Europe's longest continuous linden alley — this area is as much about water and greenery as about historic architecture.",
          "The template uses the 'nature' style so both the river and the tree alley are visible on the model.",
        ],
        sections: [
          { h2: "What makes it unique", p: ["The linden alley is an official record, and on the 3D map its long straight line reads clearly next to the hilltop castle."] },
        ],
      },
    },
  },
];

export const DISTRICT_BY_CITY_SLUG: Record<string, DistrictPage[]> = DISTRICT_PAGES.reduce(
  (acc, d) => {
    (acc[d.citySlug] ??= []).push(d);
    return acc;
  },
  {} as Record<string, DistrictPage[]>,
);
