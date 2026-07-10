// ──────────────────────────────────────────────────────────────────────────
// БЛОГ (SEO-статті): контент живе тут (як lib/legal/content.ts), БЕЗ messages —
// щоб не роздувати i18n-словники. uk + en повні; de/es/fr/pl → en-фолбек.
// Кожна стаття: керований slug, дата, секції. Рендер: app/[locale]/blog/*.
// Мета: контент-глибина під довгі запити («як зробити 3d мапу міста»,
// «подарунок 3d карта», «брелок з маршрутом gpx», «3d мапа києва»...) —
// сторінки, що можуть ранжуватись самі і лінкують у конструктор/каталог.
// ──────────────────────────────────────────────────────────────────────────

export type BlogSection = { h2?: string; p: string[] };
export type BlogArticleContent = {
  title: string;        // <title> (SEO)
  description: string;  // meta description
  h1: string;
  intro: string;
  sections: BlogSection[];
  ctaLabel: string;
  ctaHref: string;      // internal link target
  outro?: string;
};
export type BlogArticle = {
  slug: string;
  date: string;         // ISO — published
  content: { uk: BlogArticleContent; en: BlogArticleContent };
};

export const BLOG_INDEX_META = {
  uk: {
    title: "Блог про 3D-мапи, брелоки та 3D-друк — Monadruk",
    description:
      "Корисні статті: як створити 3D-мапу міста, ідеї подарунків з картою, брелоки з GPX-маршрутом, поради з 3D-друку мап.",
    h1: "Блог Monadruk",
    intro: "Гайди та ідеї: персональні 3D-мапи міст, брелоки з маршрутами і подарунки, які щось означають.",
    readLabel: "Читати",
  },
  en: {
    title: "Blog: 3D city maps, keychains & 3D printing — Monadruk",
    description:
      "Guides and ideas: how to create a 3D city map, map gift ideas, GPX route keychains, 3D-printing tips.",
    h1: "Monadruk Blog",
    intro: "Guides and ideas: personal 3D city maps, route keychains and gifts that mean something.",
    readLabel: "Read",
  },
} as const;

export function blogLocale(locale: string): "uk" | "en" {
  return locale === "uk" ? "uk" : "en";
}

export const BLOG_ARTICLES: BlogArticle[] = [
  {
    slug: "yak-stvoryty-3d-mapu-mista",
    date: "2026-07-08",
    content: {
      uk: {
        title: "Як створити 3D-мапу міста для друку: покрокова інструкція",
        description:
          "Покроково: як за 5 хвилин зробити 3D-мапу свого району — вибір ділянки, розмір, рельєф, друк удома або на замовлення. Безкоштовний конструктор.",
        h1: "Як створити 3D-мапу міста: покрокова інструкція",
        intro:
          "3D-мапа міста — це тривимірна модель району з реальними будинками, вулицями, парками й річками. Її можна надрукувати на 3D-принтері та поставити на полицю, повісити на стіну або подарувати. У цій інструкції — як зробити таку мапу самостійно за кілька хвилин, без жодних навичок 3D-моделювання.",
        sections: [
          {
            h2: "Крок 1. Оберіть ділянку на карті",
            p: [
              "Відкрийте конструктор і знайдіть своє місто — Київ, Львів, Одесу чи будь-яку точку світу. Пересуньте рамку на район, який хочете увічнити: рідну вулицю, центр міста, місце першого побачення. Рамка показує саме ту ділянку, що потрапить у модель.",
              "Порада: найкраще виглядають ділянки з різноманітною забудовою — трохи вулиць, парк, річка. Занадто великий шматок міста робить деталі дрібними, тому для моделі 8–11 см оптимальна ділянка 400–800 метрів.",
            ],
          },
          {
            h2: "Крок 2. Розмір і рельєф",
            p: [
              "Оберіть розмір готової моделі: S (5,5 см) — компактний сувенір, M (8 см) — золота середина, L (11 см) та XL (15 см) — помітна річ на полиці. Ціна залежить від розміру — від 250 ₴.",
              "Якщо місцевість горбиста (Київ, Львів, Карпати) — увімкніть рельєф: модель отримає справжні перепади висот, і пагорби буде видно й на дотик. Для рівнинних міст рельєф можна не вмикати.",
            ],
          },
          {
            h2: "Крок 3. Генерація і перевірка",
            p: [
              "Натисніть «Створити» — за 2–4 хвилини сервіс збере модель з відкритих даних OpenStreetMap: будинки з реальними висотами, дороги, зелені зони, вода. Готову модель можна покрутити прямо в браузері та роздивитись з усіх боків.",
            ],
          },
          {
            h2: "Крок 4. Друк: удома або на замовлення",
            p: [
              "Є принтер? Завантажте готовий файл 3MF або STL — він одразу відкривається в Bambu Studio чи PrusaSlicer, без додаткової обробки. Кольори шарів уже розставлені.",
              "Немає принтера? Замовте друк: ми надрукуємо мапу з екологічного біопластику Eco PLA і надішлемо Новою Поштою по Україні або в країни ЄС. Оплата карткою онлайн або при отриманні.",
            ],
          },
        ],
        ctaLabel: "Створити свою 3D-мапу",
        ctaHref: "/create",
        outro:
          "Створення моделі безкоштовне — ви платите лише за друк і доставку, якщо замовляєте готовий виріб.",
      },
      en: {
        title: "How to create a 3D city map for printing: step-by-step",
        description:
          "Step by step: make a 3D map of your neighborhood in 5 minutes — pick an area, size, relief, print at home or order. Free builder.",
        h1: "How to create a 3D city map: step-by-step",
        intro:
          "A 3D city map is a physical model of a district with real buildings, streets, parks and rivers. You can print it on a 3D printer, put it on a shelf or give it as a gift. This guide shows how to make one in minutes — no 3D-modeling skills needed.",
        sections: [
          {
            h2: "Step 1. Pick an area on the map",
            p: [
              "Open the builder and find your city — Kyiv, Lviv, or any point on Earth. Move the frame over the district you want to keep: your home street, the city center, the place you first met. What's inside the frame becomes the model.",
              "Tip: areas with mixed content look best — some streets, a park, a river. For an 8–11 cm model, a 400–800 m area is the sweet spot.",
            ],
          },
          {
            h2: "Step 2. Size and relief",
            p: [
              "Choose the final size: S (5.5 cm) is a compact souvenir, M (8 cm) is the sweet spot, L (11 cm) and XL (15 cm) stand out on a shelf. Price depends on size — from 250 ₴ (≈€6).",
              "If the terrain is hilly (Kyiv, Lviv, the Carpathians) — enable relief: the model gets real elevation and the hills are visible and touchable.",
            ],
          },
          {
            h2: "Step 3. Generate and inspect",
            p: [
              "Click «Create» — in 2–4 minutes the service assembles the model from OpenStreetMap data: buildings with real heights, roads, greenery, water. Rotate the finished model right in the browser.",
            ],
          },
          {
            h2: "Step 4. Print at home or order",
            p: [
              "Have a printer? Download the ready 3MF or STL file — it opens directly in Bambu Studio or PrusaSlicer with layer colors pre-assigned.",
              "No printer? Order a print: we print in eco-friendly Eco PLA and ship across Ukraine and to 15 EU countries.",
            ],
          },
        ],
        ctaLabel: "Create your 3D map",
        ctaHref: "/create",
        outro: "Creating a model is free — you only pay for printing and delivery if you order the finished item.",
      },
    },
  },
  {
    slug: "podarunok-yakyi-shchos-oznachaye",
    date: "2026-07-08",
    content: {
      uk: {
        title: "Що подарувати людині, в якої все є: персональна 3D-мапа",
        description:
          "Ідея подарунка, якого точно ні в кого немає: 3D-мапа місця, що щось означає — рідний двір, місце знайомства, перше спільне житло. Від 250 ₴.",
        h1: "Що подарувати людині, в якої все є",
        intro:
          "Найскладніші подарунки — для тих, у кого «все є». Чергова свічка чи листівка забудуться за тиждень. Працює інше: подарунок, який щось означає саме для цієї людини. Персональна 3D-мапа — це шматочок міста, з яким пов'язана її історія: двір дитинства, вулиця першого побачення, дім, куди щойно переїхали.",
        sections: [
          {
            h2: "Чому мапа працює як подарунок",
            p: [
              "Це не річ «з полиці магазину» — така модель існує в одному екземплярі, бо ділянку обираєте ви. Люди впізнають свій район за секунду: «це ж наш будинок!» — і саме ця мить робить подарунок пам'ятним.",
              "3D-мапа доречна майже на будь-яку нагоду: річниця (місце знайомства), новосілля (новий район), день народження (рідне місто людини, що переїхала), випуск (університетський квартал), для колеги, що йде з команди (район офісу).",
            ],
          },
          {
            h2: "Формати під різний бюджет",
            p: [
              "Брелок-мапа (від 120 ₴) — недорогий знак уваги: карта району з власним написом на звороті. Магніт на холодильник (150 ₴) — щоденне нагадування про місце. 3D-мапа на полицю (від 250 ₴ за 5,5 см до 550 ₴ за 15 см) — повноцінний інтер'єрний подарунок, з рельєфом місцевості за бажанням.",
            ],
          },
          {
            h2: "Як замовити за 5 хвилин",
            p: [
              "Оберіть ділянку на карті у конструкторі, розмір — і сервіс збере модель автоматично. Друкуємо з біопластику Eco PLA та надсилаємо Новою Поштою; можна замовити і цифровий файл для власного друку. Встигнути до дати легко: виготовлення 1–3 робочі дні.",
            ],
          },
        ],
        ctaLabel: "Підібрати подарунок-мапу",
        ctaHref: "/podarunok",
        outro: "Не знаєте, яку ділянку обрати? Напишіть нам — підкажемо, як виглядатиме район, і зберемо превʼю.",
      },
      en: {
        title: "A gift for someone who has everything: a personal 3D map",
        description:
          "A gift no one else has: a 3D map of a place that matters — a childhood street, where you met, your first home. From ≈€6.",
        h1: "A gift for someone who has everything",
        intro:
          "The hardest gifts are for people who «have everything». What works is meaning: a personal 3D map is a piece of the city tied to their story — the childhood backyard, the street of a first date, the home they just moved into.",
        sections: [
          {
            h2: "Why a map works as a gift",
            p: [
              "It's one of a kind — you choose the exact area. People recognize their neighborhood in a second: «that's our house!» — and that moment makes the gift memorable.",
              "It fits almost any occasion: anniversary (where you met), housewarming (the new district), birthday (the hometown of someone who moved away), graduation (the campus quarter), a leaving colleague (the office block).",
            ],
          },
          {
            h2: "Formats for any budget",
            p: [
              "Map keychain (from 120 ₴ ≈ €3) — a small token with custom text on the back. Fridge magnet (150 ₴) — a daily reminder of a place. Shelf 3D map (250–550 ₴ ≈ €6–13 depending on size) — a real interior piece, with terrain relief if you like.",
            ],
          },
          {
            h2: "Order in 5 minutes",
            p: [
              "Pick the area in the builder, choose a size — the model is assembled automatically. We print in Eco PLA and ship to Ukraine and the EU; you can also get the digital file. Production takes 1–3 business days.",
            ],
          },
        ],
        ctaLabel: "Pick a map gift",
        ctaHref: "/podarunok",
        outro: "Not sure which area to pick? Message us — we'll help and build a preview.",
      },
    },
  },
  {
    slug: "brelok-z-kartoyu-mista",
    date: "2026-07-08",
    content: {
      uk: {
        title: "Брелок з картою міста: персональний аксесуар від 120 ₴",
        description:
          "Брелок-мапа 55×30 мм: вулиці й парки вашого району рельєфом, власний напис на звороті. 3D-друк на замовлення від 120 ₴, доставка по Україні та ЄС.",
        h1: "Брелок з картою міста — маленька мапа, що завжди з тобою",
        intro:
          "Брелок-мапа — це жетон 55×30 мм, на якому рельєфом надруковано карту обраного району: вулиці, парки, річки. На звороті — власний напис: назва міста, дата, ім'я чи координати. Це найдоступніший спосіб носити з собою місце, яке щось означає.",
        sections: [
          {
            h2: "Що можна зобразити",
            p: [
              "Будь-яку точку світу: рідний район, місто, де народилась дитина, вулицю, де жили студентами. Морське узбережжя з лінією пляжу, гірський хребет з рельєфом висот — теж працює: для гір є окремий топо-режим, де вершини й долини читаються пальцями.",
              "Популярні варіанти: пара брелоків-«сердець» з районами двох людей (з'єднуються як пазл), брелок бігуна з GPX-треком улюбленого маршруту, корпоративні брелоки з районом офісу для команди.",
            ],
          },
          {
            h2: "Матеріал і якість",
            p: [
              "Друкуємо з біопластику Eco PLA — легкий, приємний на дотик, не боїться щоденного носіння з ключами. Лінії вулиць надруковані з роздільністю 0,4 мм — навіть маленькі провулки залишаються читабельними.",
            ],
          },
          {
            h2: "Ціна і терміни",
            p: [
              "Брелок-мапа — від 120 ₴. Виготовлення 1–3 робочі дні, доставка Новою Пошта по Україні або Nova Post/Meest у 15 країн ЄС. Оплата карткою онлайн (LiqPay) або при отриманні.",
            ],
          },
        ],
        ctaLabel: "Створити брелок зі своїм районом",
        ctaHref: "/keychains",
      },
      en: {
        title: "City map keychain: a personal accessory from €3",
        description:
          "55×30 mm map keychain: your neighborhood's streets and parks in relief, custom text on the back. 3D-printed to order, shipping to Ukraine & EU.",
        h1: "A city map keychain — a little map that's always with you",
        intro:
          "The map keychain is a 55×30 mm tag with your chosen district printed in relief: streets, parks, rivers. On the back — your text: a city name, a date, a name or coordinates. The most affordable way to carry a place that matters.",
        sections: [
          {
            h2: "What you can put on it",
            p: [
              "Any point on Earth: your home district, the city where your child was born, the street where you lived as students. A seashore with the beach line or a mountain ridge with elevation relief also works — mountains have a dedicated topo mode where peaks and valleys read under your fingers.",
              "Popular picks: a pair of «heart» keychains with two people's districts (they connect like a puzzle), a runner's keychain with a GPX track, corporate keychains with the office block for a team.",
            ],
          },
          {
            h2: "Material and quality",
            p: [
              "Printed in Eco PLA bioplastic — light, pleasant to touch, fine for daily use with keys. Street lines are printed at 0.4 mm resolution, so even small lanes stay readable.",
            ],
          },
          {
            h2: "Price and lead time",
            p: [
              "Map keychain — from 120 ₴ (≈€3). Production 1–3 business days, delivery across Ukraine and to 15 EU countries. Card payment online or on delivery.",
            ],
          },
        ],
        ctaLabel: "Create a keychain with your district",
        ctaHref: "/keychains",
      },
    },
  },
  {
    slug: "brelok-gpx-marshrut",
    date: "2026-07-08",
    content: {
      uk: {
        title: "Брелок з GPX-маршрутом: увічни свій забіг, похід чи веломаршрут",
        description:
          "Завантаж GPX-трек зі Strava чи Garmin — і отримай брелок з рельєфною лінією свого маршруту поверх карти району. Пам'ять про марафон чи похід від 120 ₴.",
        h1: "Брелок з GPX-маршрутом: твій трек у пластику",
        intro:
          "Перший марафон, вело-сотка, похід на Говерлу — трек цих маршрутів лежить у Strava чи Garmin і його ніхто не бачить. Брелок з GPX-маршрутом перетворює трек на фізичну річ: рельєфна лінія маршруту проходить поверх карти району прямо на жетоні, який завжди з ключами.",
        sections: [
          {
            h2: "Як це працює",
            p: [
              "Експортуйте GPX-файл зі Strava, Garmin Connect, Komoot чи будь-якого трекера. Завантажте його в конструктор — сервіс сам знайде місце маршруту, підбере масштаб, щоб трек вліз повністю, і накладе лінію треку поверх вулиць. Лінія маршруту притягується до доріг, тож виглядає акуратно навіть із «шумного» GPS-запису.",
            ],
          },
          {
            h2: "Кому це дарують",
            p: [
              "Бігунам — фініш першого марафону чи улюблене коло парком. Велосипедистам — маршрут багатоденки. Туристам — трек походу в Карпати. Це подарунок, який неможливо купити готовим: маршрут у кожного свій.",
              "Можна додати напис на звороті: назву події, дату, час фінішу — «Kyiv Marathon 2026 · 3:58».",
            ],
          },
          {
            h2: "Ціна",
            p: [
              "Брелок з GPX-треком коштує як звичайний брелок-мапа — від 120 ₴. Виготовлення 1–3 робочі дні, доставка по Україні та ЄС.",
            ],
          },
        ],
        ctaLabel: "Завантажити свій GPX-трек",
        ctaHref: "/keychains",
      },
      en: {
        title: "GPX route keychain: your run, hike or ride in plastic",
        description:
          "Upload a GPX track from Strava or Garmin — get a keychain with your route as a relief line over the district map. A marathon or hike memory from ≈€3.",
        h1: "GPX route keychain: your track made physical",
        intro:
          "Your first marathon, a century ride, a Hoverla hike — those tracks sit invisible in Strava or Garmin. A GPX keychain turns the track into a physical object: the route runs in relief over the district map on a tag that lives on your keys.",
        sections: [
          {
            h2: "How it works",
            p: [
              "Export a GPX file from Strava, Garmin Connect, Komoot or any tracker. Upload it in the builder — the service locates the route, scales the map so the whole track fits and lays the route line over the streets, snapping it to roads so even noisy GPS looks clean.",
            ],
          },
          {
            h2: "Who gets these as gifts",
            p: [
              "Runners — the first-marathon finish or a favorite park loop. Cyclists — a multi-day route. Hikers — a Carpathian trail. It's a gift that can't be bought off the shelf: everyone's route is their own.",
              "Add text on the back: event name, date, finish time — «Kyiv Marathon 2026 · 3:58».",
            ],
          },
          {
            h2: "Price",
            p: [
              "A GPX keychain costs the same as a regular map keychain — from 120 ₴ (≈€3). Production 1–3 business days, shipping to Ukraine and the EU.",
            ],
          },
        ],
        ctaLabel: "Upload your GPX track",
        ctaHref: "/keychains",
      },
    },
  },
  {
    slug: "3d-mapa-kyeva",
    date: "2026-07-08",
    content: {
      uk: {
        title: "3D-мапа Києва: улюблений район на твоїй полиці",
        description:
          "Поділ, Печерськ, Оболонь чи Русанівка — 3D-мапа будь-якого району Києва з реальними будинками і рельєфом дніпровських схилів. Друк від 250 ₴.",
        h1: "3D-мапа Києва: місто, яке можна потримати в руках",
        intro:
          "Київ — місто з характером у кожному районі: андріївські схили, річкова сітка Русанівки, радянський модернізм Оболоні, старий Поділ. 3D-мапа перетворює улюблений район на фізичну модель: будинки з реальними висотами, вулиці, парки, Дніпро — і навіть рельєф київських пагорбів.",
        sections: [
          {
            h2: "Які райони виглядають найкраще",
            p: [
              "Поділ і центр — щільна історична забудова з виразними кварталами. Печерськ з Лаврою — рельєф схилів плюс упізнавані домінанти. Русанівка й Оболонь — унікальна водна сітка каналів і заток Дніпра, яка на 3D-моделі виглядає особливо ефектно. Виберіть ділянку 400–800 м — цього достатньо, щоб район було впізнати з першого погляду.",
              "Київ — горбисте місто, тож увімкнений рельєф дає моделі справжні перепади висот: від рівня Дніпра до верхнього міста.",
            ],
          },
          {
            h2: "Розміри та ціни",
            p: [
              "S 5,5 см — 250 ₴, M 8 см — 350 ₴, L 11 см — 450 ₴, XL 15 см — 550 ₴; рельєф +60 ₴. Також є брелок з районом Києва (від 120 ₴) і магніт на холодильник (150 ₴).",
            ],
          },
          {
            h2: "Кияни дарують це так",
            p: [
              "Мапа двору, де виріс, — батькам. Район першої спільної квартири — на річницю. Улюблений маршрут набережною — брелоком з GPX-треком. Модель району офісу — команді на пам'ять.",
            ],
          },
        ],
        ctaLabel: "Створити 3D-мапу Києва",
        ctaHref: "/maps/kyiv",
      },
      en: {
        title: "3D map of Kyiv: your favorite district on a shelf",
        description:
          "Podil, Pechersk, Obolon or Rusanivka — a 3D map of any Kyiv district with real buildings and the relief of the Dnipro hills. Printed from ≈€6.",
        h1: "3D map of Kyiv: a city you can hold",
        intro:
          "Kyiv has character in every district: the Andriivskyi slopes, Rusanivka's canal grid, Obolon's modernism, old Podil. A 3D map turns a favorite district into a physical model: buildings with real heights, streets, parks, the Dnipro — and the real relief of Kyiv's hills.",
        sections: [
          {
            h2: "Which districts look best",
            p: [
              "Podil and the center — dense historic quarters. Pechersk with the Lavra — hill relief plus recognizable landmarks. Rusanivka and Obolon — a unique water grid of canals and Dnipro bays that looks striking in 3D. A 400–800 m area is enough to recognize the district at first glance.",
              "Kyiv is hilly, so enabling relief gives the model true elevation — from the Dnipro up to the Upper City.",
            ],
          },
          {
            h2: "Sizes and prices",
            p: [
              "S 5.5 cm — 250 ₴, M 8 cm — 350 ₴, L 11 cm — 450 ₴, XL 15 cm — 550 ₴; relief +60 ₴. There's also a Kyiv district keychain (from 120 ₴) and a fridge magnet (150 ₴).",
            ],
          },
          {
            h2: "How Kyivans gift it",
            p: [
              "The childhood backyard — for parents. The district of a first shared flat — for an anniversary. A favorite riverside route — as a GPX keychain. The office block — for the team.",
            ],
          },
        ],
        ctaLabel: "Create a 3D map of Kyiv",
        ctaHref: "/maps/kyiv",
      },
    },
  },
];

export const BLOG_BY_SLUG: Record<string, BlogArticle> = Object.fromEntries(
  BLOG_ARTICLES.map((a) => [a.slug, a]),
);
