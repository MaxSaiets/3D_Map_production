// ──────────────────────────────────────────────────────────────────────────
// СТАТТІ ПІД ПОДАРУНКОВІ ЗАПИТИ (2026-09-03). Власник: «мало людей знаходять сайт»;
// памʼять keyword-research: «купити 3D-мапу» НЕ шукають, а шукають «що подарувати
// хлопцю / дівчині на річницю / батькам / на новосілля». Кожна стаття відповідає на
// РЕАЛЬНИЙ запит і веде в конструктор (guided) або на occasion-сторінку.
// uk + en обовʼязкові (blogContent() фолбечить de/es/fr/pl на en).
// ──────────────────────────────────────────────────────────────────────────
import type { BlogArticle } from "@/lib/blog";

export const GIFT_ARTICLES_2026: BlogArticle[] = [
  {
    slug: "shcho-podaruvaty-khloptsevi-na-den-narodzhennya",
    date: "2026-09-03",
    content: {
      uk: {
        title: "Що подарувати хлопцю на день народження: 3D-мапа його району",
        description:
          "Ідея подарунка хлопцю на день народження, якого ні в кого немає: обʼємна 3D-мапа району, де він виріс, або брелок з його вулицею. Від 120 ₴, друк за 2–4 дні.",
        h1: "Що подарувати хлопцю на день народження, якщо шкарпетки вже були",
        intro:
          "Гаджети він купує сам, парфуми — лотерея, а «сертифікат на емоції» забудеться за тиждень. Працює інше: річ, у якій є ВІН. Наприклад, обʼємна мапа району, де він виріс, або кварталу, де ви познайомились — з реальними будинками, вулицями й парком, який він упізнає з першого погляду.",
        sections: [
          {
            h2: "Чому саме мапа",
            p: [
              "Це не сувенір із полиці: модель існує в одному екземплярі, бо зроблена під конкретні координати. На столі чи полиці вона одразу стає предметом розмови — «а це наш двір, а ось школа».",
              "Мапа не має розміру одягу й не залежить від смаку: місце, яке щось означає, подобається завжди.",
            ],
          },
          {
            h2: "Три варіанти під різний бюджет",
            p: [
              "Брелок з мапою (від 120 ₴) — його вулиця завжди з ключами, можна додати напис або дату на звороті.",
              "Магніт на холодильник (150 ₴) — компактна мапа 6 см, якщо хочеться недорогого, але особистого.",
              "Обʼємна 3D-мапа району (від 250 ₴) — головний подарунок: будинки з реальними висотами, парки, річка. Розмір від 5,5 до 15 см.",
            ],
          },
          {
            h2: "Як зробити за 5 хвилин",
            p: [
              "Відкрийте конструктор, знайдіть адресу або натисніть місто, посуньте рамку на потрібний квартал — 3D-превʼю зʼявиться за 1–2 хвилини і безкоштовно.",
              "Далі два шляхи: замовити друк з доставкою Новою Поштою по Україні або завантажити файл і надрукувати самому.",
            ],
          },
          {
            h2: "Що додати, щоб було «його»",
            p: [
              "Позначте його будинок червоною вставкою, додайте напис — імʼя, рік або координати. Для пари є брелок-серце з двох половинок: його район і ваш.",
            ],
          },
        ],
        ctaLabel: "Створити мапу його району",
        ctaHref: "/create",
        outro: "Друк і відправка — 2–4 робочі дні. Якщо день народження вже завтра — замовте брелок: він друкується найшвидше.",
      },
      en: {
        title: "Birthday gift for your boyfriend: a 3D map of his neighbourhood",
        description:
          "A birthday gift idea nobody else has: a 3D map of the neighbourhood he grew up in, or a keychain with his street. From 120 ₴, printed in 2–4 days.",
        h1: "What to give your boyfriend for his birthday when socks are already taken",
        intro:
          "He buys gadgets himself, perfume is a lottery, and an “experience voucher” is forgotten in a week. What works is a thing that has HIM in it: a 3D map of the neighbourhood he grew up in, or the block where you met — with real buildings, streets and the park he recognises at a glance.",
        sections: [
          { h2: "Why a map", p: ["It is not a shelf souvenir: the model exists once, because it is made for specific coordinates. On a desk it instantly becomes a conversation piece.", "A map has no clothing size and no taste risk: a place that matters always lands."] },
          { h2: "Three options for any budget", p: ["Map keychain (from 120 ₴) — his street always on the keys, with an optional engraving on the back.", "Fridge magnet (150 ₴) — a compact 6 cm map when you want something small but personal.", "3D district map (from 250 ₴) — the main gift: buildings with real heights, parks, river. 5.5 to 15 cm."] },
          { h2: "How to make it in 5 minutes", p: ["Open the builder, find the address or tap a city, drag the frame over the block — a free 3D preview appears in 1–2 minutes.", "Then two paths: order a print shipped within Ukraine, or download the file and print it yourself."] },
          { h2: "Make it his", p: ["Mark his house with a red inlay, add a name, a year or coordinates. For couples there is a two-half heart keychain: his district and yours."] },
        ],
        ctaLabel: "Create a map of his neighbourhood",
        ctaHref: "/create",
        outro: "Print and dispatch take 2–4 business days. If the birthday is tomorrow — order a keychain, it prints fastest.",
      },
    },
  },
  {
    slug: "podarunok-divchyni-na-richnytsyu",
    date: "2026-09-03",
    content: {
      uk: {
        title: "Подарунок дівчині на річницю стосунків: мапа місця, де все почалось",
        description:
          "Що подарувати дівчині на річницю: 3D-мапа кварталу першого побачення або пара брелоків-сердець із вашими районами. Персональний напис, друк 2–4 дні, доставка по Україні.",
        h1: "Подарунок дівчині на річницю: місце, де все почалось",
        intro:
          "Квіти зівʼянуть, а координати лишаться. Мапа кварталу, де ви вперше зустрілись, кавʼярні на розі чи лавки в парку — це подарунок, який розповідає вашу історію без слів.",
        sections: [
          {
            h2: "Дві ідеї, які працюють найкраще",
            p: [
              "Обʼємна мапа першого побачення: район 400–800 м із реальними будинками, парком і вулицею, якою ви йшли. Поставте на полицю — і кожен погляд повертає в той день.",
              "Брелок-серце з двох половинок: одна половинка — її район, друга — ваш. Разом складаються в ціле серце, окремо — кожен носить своє.",
            ],
          },
          {
            h2: "Напис, який робить подарунок вашим",
            p: [
              "Дата першої зустрічі, її імʼя або координати місця — на плоскій мапі, магніті чи на звороті брелока. Напис додається одним кліком у конструкторі.",
            ],
          },
          {
            h2: "Скільки коштує і коли буде готово",
            p: [
              "Брелок — від 120 ₴, пара сердець — два брелоки. Обʼємна мапа — від 250 ₴ (5,5 см) до 550 ₴ (15 см). Друк і відправка Новою Поштою — 2–4 робочі дні.",
            ],
          },
        ],
        ctaLabel: "Створити мапу вашого місця",
        ctaHref: "/podarunok/na-richnytsyu",
        outro: "Не знаєте точну адресу? Досить назви вулиці або кавʼярні — пошук на карті підкаже, а рамку можна посунути рукою.",
      },
      en: {
        title: "Anniversary gift for your girlfriend: a map of where it all began",
        description:
          "What to give your girlfriend for an anniversary: a 3D map of the block of your first date, or a pair of heart keychains with both your neighbourhoods. Personal engraving, printed in 2–4 days.",
        h1: "Anniversary gift for her: the place where it all began",
        intro:
          "Flowers fade, coordinates stay. A map of the block where you first met, the corner café or the bench in the park — a gift that tells your story without words.",
        sections: [
          { h2: "Two ideas that work best", p: ["A 3D map of the first date: a 400–800 m district with real buildings, the park and the street you walked. Put it on a shelf and every glance brings that day back.", "A two-half heart keychain: one half is her neighbourhood, the other is yours. Together they make a whole heart; apart, each carries their own."] },
          { h2: "An engraving that makes it yours", p: ["The date you met, her name or the coordinates of the place — on a flat map, a magnet or the back of the keychain. Added with one click in the builder."] },
          { h2: "Price and timing", p: ["Keychain from 120 ₴; a heart pair is two keychains. 3D map from 250 ₴ (5.5 cm) to 550 ₴ (15 cm). Print and dispatch in 2–4 business days."] },
        ],
        ctaLabel: "Create a map of your place",
        ctaHref: "/podarunok/na-richnytsyu",
        outro: "Don’t know the exact address? A street or café name is enough — the map search suggests it, and the frame can be dragged by hand.",
      },
    },
  },
  {
    slug: "podarunok-batkam-na-richnytsyu-vesillya",
    date: "2026-09-03",
    content: {
      uk: {
        title: "Подарунок батькам на річницю весілля: 3D-мапа їхнього першого дому",
        description:
          "Ідея подарунка батькам на річницю весілля: обʼємна мапа району, де вони починали, або міста, де народились. Панно з плиток для великого ювілею. Друк 2–4 дні.",
        h1: "Подарунок батькам на річницю весілля, який не пилитиметься в шафі",
        intro:
          "У батьків «усе є», тому виграють не речі, а память. Мапа району, де вони знімали першу квартиру, вулиці, якою ходили до РАГСу, або рідного міста, з якого переїхали — це подарунок, над яким вони будуть довго схилятися й показувати гостям.",
        sections: [
          {
            h2: "Формат під ювілей",
            p: [
              "На «звичайну» річницю — обʼємна мапа M або L (8–11 см) на полицю. На 25 чи 30 років — панно з кількох плиток на стіну: район стає картиною.",
              "Якщо батьки в різних містах народились — дві мапи-«половинки» або два магніти на один холодильник.",
            ],
          },
          {
            h2: "Що написати",
            p: [
              "Дата весілля і рік ювілею — найпростіше й найточніше. Напис ставиться на плоскій мапі чи магніті; на обʼємній — окремою табличкою-вставкою.",
            ],
          },
          {
            h2: "Як замовити, якщо ви в іншому місті",
            p: [
              "Конструктор працює з будь-якою адресою в Україні: знайдіть вулицю батьків, поставте рамку, перевірте 3D-превʼю. Доставка Новою Поштою одразу на їхнє відділення — вкажіть його у формі замовлення.",
            ],
          },
        ],
        ctaLabel: "Створити мапу для батьків",
        ctaHref: "/create",
        outro: "Порада: додайте до замовлення коротку записку в коментарі — ми покладемо її в коробку.",
      },
      en: {
        title: "Wedding anniversary gift for parents: a 3D map of their first home",
        description:
          "A gift idea for parents’ wedding anniversary: a 3D map of the district where they started, or of the city they were born in. A tile panel for big jubilees. Printed in 2–4 days.",
        h1: "A parents’ anniversary gift that won’t gather dust in a cupboard",
        intro:
          "Parents “have everything”, so memories win over things. A map of the district of their first rented flat, the street they walked to the registry office, or the hometown they left — a gift they will lean over for a long time and show to guests.",
        sections: [
          { h2: "A format for the jubilee", p: ["For a regular anniversary — a 3D map M or L (8–11 cm) for the shelf. For 25 or 30 years — a multi-tile wall panel: the district becomes a picture.", "If they were born in different cities — two “half” maps, or two magnets on one fridge."] },
          { h2: "What to write", p: ["The wedding date and the jubilee year is the simplest and most precise. The text goes on a flat map or magnet; on a 3D map — as a separate inlay plate."] },
          { h2: "Ordering from another city", p: ["The builder works with any address in Ukraine: find their street, place the frame, check the 3D preview. Nova Poshta delivers straight to their branch — specify it in the order form."] },
        ],
        ctaLabel: "Create a map for your parents",
        ctaHref: "/create",
        outro: "Tip: add a short note in the order comment — we will put it in the box.",
      },
    },
  },
  {
    slug: "podarunok-na-novosillya-druzyam",
    date: "2026-09-03",
    content: {
      uk: {
        title: "Подарунок на новосілля друзям: мапа нового району замість чергового горщика",
        description:
          "Що подарувати на новосілля: 3D-мапа кварталу нової квартири або магніт з новим районом. Стає першим декором у порожній квартирі. Від 150 ₴, друк 2–4 дні.",
        h1: "Подарунок на новосілля, який одразу стане декором",
        intro:
          "У новій квартирі порожні полиці й стіни — і горщик із квіткою це не рятує. Мапа нового району з реальними будинками, парком і вулицею, на якій тепер живуть друзі, займає своє місце на полиці в перший же вечір.",
        sections: [
          {
            h2: "Що обрати",
            p: [
              "Магніт з мапою нового району (150 ₴) — недорогий, доречний і одразу на холодильник.",
              "Обʼємна мапа кварталу (від 250 ₴) — для полиці у вітальні; позначте їхній будинок червоною вставкою, щоб гості одразу бачили «ось ми».",
              "Панно з плиток на стіну — якщо новосілля велике й дарують кілька друзів разом.",
            ],
          },
          {
            h2: "Напис на мапі",
            p: [
              "Назва вулиці, дата переїзду або просто «Дім». Напис додається в конструкторі одним кліком і друкується на плоскій мапі чи магніті.",
            ],
          },
          {
            h2: "Не знаєте точної адреси?",
            p: [
              "Достатньо назви вулиці й міста — пошук на карті знайде, а рамку ви посунете на потрібний будинок. 3D-превʼю безкоштовне, замовлення — після того, як побачите результат.",
            ],
          },
        ],
        ctaLabel: "Створити мапу нового району",
        ctaHref: "/podarunok/na-novosillya",
      },
      en: {
        title: "Housewarming gift for friends: a map of the new district instead of another pot plant",
        description:
          "What to give for a housewarming: a 3D map of the block of the new flat, or a magnet with the new district. Becomes the first decor in an empty flat. From 150 ₴, printed in 2–4 days.",
        h1: "A housewarming gift that becomes decor on day one",
        intro:
          "A new flat has empty shelves and walls — and a pot plant doesn’t fix that. A map of the new district with real buildings, the park and the street your friends now live on takes its place on the shelf the very first evening.",
        sections: [
          { h2: "What to choose", p: ["A magnet with the new district (150 ₴) — inexpensive, fitting and straight onto the fridge.", "A 3D map of the block (from 250 ₴) — for the living-room shelf; mark their building with a red inlay so guests see “that’s us”.", "A tile wall panel — for a big housewarming when several friends chip in."] },
          { h2: "Text on the map", p: ["The street name, the moving date or simply “Home”. Added in the builder with one click, printed on a flat map or magnet."] },
          { h2: "Don’t know the exact address?", p: ["A street and city are enough — the map search finds it and you drag the frame onto the right building. The 3D preview is free; you order after you see the result."] },
        ],
        ctaLabel: "Create a map of the new district",
        ctaHref: "/podarunok/na-novosillya",
      },
    },
  },
  {
    slug: "podarunok-tomu-khto-lyubyt-svoye-misto",
    date: "2026-09-03",
    content: {
      uk: {
        title: "Подарунок тому, хто любить своє місто: 3D-мапа замість магнітика з ринку",
        description:
          "Оригінальний подарунок для людини, яка любить своє місто: обʼємна мапа улюбленого району, рельєфна мапа з пагорбами або брелок із рідною вулицею. Будь-яке місто України.",
        h1: "Подарунок людині, яка любить своє місто",
        intro:
          "Є люди, які знають кожен двір і кожну кавʼярню свого району — і гордяться цим. Магнітик із сувенірного кіоску їх не здивує. А от обʼємна мапа саме їхнього кварталу — з будинками справжньої висоти, парком і річкою — здивує точно.",
        sections: [
          {
            h2: "Яку мапу обрати",
            p: [
              "Обʼємна мапа міста — класика: будинки, вулиці, парки. Обирайте район, а не «все місто»: 400–800 м зі змішаною забудовою впізнаються з першого погляду.",
              "Рельєфна мапа — для горбистих міст: Києва, Львова, Ужгорода, Чернівців. Схили, пагорби й горизонталі стають частиною моделі.",
              "Брелок або магніт — коли хочеться малого, але особистого: вулиця, де людина виросла, завжди з ключами.",
            ],
          },
          {
            h2: "Не лише обласні центри",
            p: [
              "Конструктор працює з будь-якою точкою на карті — містечком, селом, дачним кооперативом. Якщо там є вулиці й будинки в OpenStreetMap, буде і модель.",
            ],
          },
          {
            h2: "Як це виглядає в руках",
            p: [
              "Друкуємо з екологічного біопластику Eco PLA, розміри від 5,5 до 15 см. Перед замовленням ви бачите безкоштовне 3D-превʼю — те, що приїде, тільки в пластику.",
            ],
          },
        ],
        ctaLabel: "Створити мапу улюбленого району",
        ctaHref: "/create",
        outro: "Подивіться, як виглядають надруковані мапи різних міст — у галереї реальних друків.",
      },
      en: {
        title: "A gift for someone who loves their city: a 3D map instead of a market magnet",
        description:
          "An original gift for a person who loves their city: a 3D map of their favourite district, a relief map with hills, or a keychain with their home street. Any city in Ukraine.",
        h1: "A gift for someone who loves their city",
        intro:
          "Some people know every courtyard and every café of their district — and are proud of it. A magnet from a souvenir kiosk won’t surprise them. A 3D map of exactly their block — with buildings of real height, the park and the river — definitely will.",
        sections: [
          { h2: "Which map to choose", p: ["A 3D city map is the classic: buildings, streets, parks. Choose a district, not “the whole city”: 400–800 m of mixed streets is recognisable at a glance.", "A relief map — for hilly cities: Kyiv, Lviv, Uzhhorod, Chernivtsi. Slopes and contour lines become part of the model.", "A keychain or magnet — when you want something small but personal: the street they grew up on, always on the keys."] },
          { h2: "Not only big cities", p: ["The builder works with any point on the map — a small town, a village, a summer-house cooperative. If it has streets and buildings in OpenStreetMap, it has a model."] },
          { h2: "What it looks like in hand", p: ["Printed in eco-friendly Eco PLA, 5.5 to 15 cm. Before ordering you see a free 3D preview — exactly what arrives, only in plastic."] },
        ],
        ctaLabel: "Create a map of their favourite district",
        ctaHref: "/create",
        outro: "See how printed maps of different cities look in the gallery of real prints.",
      },
    },
  },
];
