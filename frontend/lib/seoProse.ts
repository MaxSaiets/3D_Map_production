// SEO-проза для builder-сторінок (/create, /keychains): це client-side
// конструктори майже без індексованого тексту, хоча саме вони таргетять
// грошові запити («конструктор 3д мапи», «брелок з картою»). Серверний
// текст-блок рендериться ПІД конструктором (нижче згину) — краулер бачить
// контент, користувач UI не втрачає. uk/en повні; інші локалі → en.

export type SeoProse = { h2: string; p1: string; p2: string };

const CREATE: Record<"uk" | "en", SeoProse> = {
  uk: {
    h2: "Онлайн-конструктор 3D-мапи міста",
    p1: "Monadruk перетворює будь-яку точку світу на друковану 3D-модель: оберіть район на карті — і за кілька хвилин отримаєте тривимірну мапу з реальними висотами будинків, вулицями, парками й річками за даними OpenStreetMap. Для горбистих міст можна ввімкнути рельєф місцевості, а серію сусідніх плиток — з'єднати у настінне панно.",
    p2: "Готову модель друкуємо з екологічного біопластику Eco PLA у розмірах від 5,5 до 15 см (ціна від 250 ₴) і надсилаємо Новою Поштою по Україні або у 15 країн ЄС. Якщо у вас є власний 3D-принтер — завантажте готовий файл 3MF/STL і надрукуйте вдома.",
  },
  en: {
    h2: "Online 3D city map builder",
    p1: "Monadruk turns any point on Earth into a printable 3D model: pick a district on the map and in minutes get a three-dimensional map with real building heights, streets, parks and rivers from OpenStreetMap data. Hilly cities can be rendered with true terrain relief, and adjacent tiles can be joined into a wall panel.",
    p2: "We print the finished model in eco-friendly Eco PLA in sizes from 5.5 to 15 cm (from ≈€6) and ship across Ukraine and to 15 EU countries. Have your own 3D printer? Download the ready 3MF/STL file and print at home.",
  },
};

const KEYCHAINS: Record<"uk" | "en", SeoProse> = {
  uk: {
    h2: "Брелок з картою міста на замовлення",
    p1: "Брелок-мапа — це жетон 55×30 мм з рельєфною картою обраного району: вулиці, парки й річки, які можна відчути пальцями. Додайте власний напис на звороті — назву міста, дату чи координати. Є режим гірського рельєфу (топо-брелок) і брелок з вашим GPX-маршрутом зі Strava чи Garmin.",
    p2: "Друкуємо з Eco PLA за 1–3 робочі дні, ціна від 120 ₴. Доставка Новою Поштою по Україні та Nova Post/Meest у країни ЄС. Пара брелоків-«сердець» з районами двох людей з'єднується як пазл — популярний подарунок для пар.",
  },
  en: {
    h2: "Custom city map keychain",
    p1: "The map keychain is a 55×30 mm tag with a relief map of your chosen district: streets, parks and rivers you can feel with your fingers. Add custom text on the back — a city name, a date or coordinates. There's a mountain-relief topo mode and a keychain with your GPX route from Strava or Garmin.",
    p2: "Printed in Eco PLA within 1–3 business days, from ≈€3. Shipping across Ukraine and to EU countries. A pair of «heart» keychains with two people's districts joins like a puzzle — a popular couple's gift.",
  },
};

const WORLDS: Record<"uk" | "en", SeoProse> = {
  uk: {
    h2: "AI-генератор фантастичних 3D-світів",
    p1: "Worlds — експериментальний інструмент Monadruk: опишіть ландшафт словами («вулканічний острів», «глибокий каньйон», «пологі пагорби») — і AI згенерує унікальну 3D-модель рельєфу, якої не існує на реальній карті. На відміну від конструктора мап міст, тут немає прив'язки до OpenStreetMap — лише уява.",
    p2: "Готову модель можна одразу покрутити в браузері й завантажити файл GLB безкоштовно. Розміри від 8 до 18 см. Для друку такого світу на замовлення — напишіть нам у чат.",
  },
  en: {
    h2: "AI generator of fantasy 3D worlds",
    p1: "Worlds is an experimental Monadruk tool: describe a landscape in words (\"volcanic island\", \"deep canyon\", \"rolling hills\") and AI generates a unique 3D terrain model that doesn't exist on any real map. Unlike the city map builder, there's no tie to OpenStreetMap here — just imagination.",
    p2: "Rotate the finished model right in the browser and download the GLB file for free. Sizes from 8 to 18 cm. To order this world printed, message us in chat.",
  },
};

export function seoProse(page: "create" | "keychains" | "worlds", locale: string): SeoProse {
  const dict = page === "create" ? CREATE : page === "keychains" ? KEYCHAINS : WORLDS;
  return dict[locale === "uk" ? "uk" : "en"];
}

export type ProseFaqItem = { q: string; a: string };

const CREATE_FAQ: Record<"uk" | "en", ProseFaqItem[]> = {
  uk: [
    { q: "Скільки коштує 3D-мапа?", a: "Від 250 ₴ за розмір S (5,5 см) до 550 ₴ за XL (15 см). Рельєф місцевості — опція +60 ₴." },
    { q: "Яку ділянку краще обрати?", a: "Ділянку 400–800 метрів зі змішаною забудовою: трохи вулиць, парк або вода — так район впізнається з першого погляду." },
    { q: "Скільки триває виготовлення?", a: "1–3 робочі дні на друк, потім доставка Новою Поштою по Україні або Nova Post/Meest у країни ЄС." },
    { q: "Чи можна надрукувати самому?", a: "Так — завантажте готовий файл 3MF або STL, він одразу відкривається в Bambu Studio чи PrusaSlicer." },
  ],
  en: [
    { q: "How much does a 3D map cost?", a: "From ≈€6 for size S (5.5 cm) to ≈€13 for XL (15 cm). Terrain relief is an option, +≈€1.5." },
    { q: "Which area should I pick?", a: "A 400–800 m area with mixed content: some streets, a park or water — the district stays recognizable at first glance." },
    { q: "How long does production take?", a: "1–3 business days to print, then shipping across Ukraine or via Nova Post/Meest to the EU." },
    { q: "Can I print it myself?", a: "Yes — download the ready 3MF or STL file, it opens directly in Bambu Studio or PrusaSlicer." },
  ],
};

const WORLDS_FAQ: Record<"uk" | "en", ProseFaqItem[]> = {
  uk: [
    { q: "Це справжня карта чи вигадана?", a: "Вигадана — AI генерує рельєф із текстового опису, без прив'язки до реальної місцевості." },
    { q: "Чи можна завантажити файл безкоштовно?", a: "Так, файл GLB для перегляду — безкоштовно. Друк такого світу на замовлення обговорюється окремо в чаті." },
    { q: "Які розміри доступні?", a: "S (8 см), M (12 см) та L (18 см)." },
  ],
  en: [
    { q: "Is this a real map or a fictional one?", a: "Fictional — AI generates terrain from a text prompt, with no tie to a real location." },
    { q: "Can I download the file for free?", a: "Yes, the GLB preview file is free. Ordering a print of this world is discussed separately in chat." },
    { q: "What sizes are available?", a: "S (8 cm), M (12 cm) and L (18 cm)." },
  ],
};

const SHOWCASE_FAQ: Record<"uk" | "en", ProseFaqItem[]> = {
  uk: [
    { q: "Це реальні фото чи 3D-рендери?", a: "У галереї є і те, і те: розділ «Як це виглядає надрукованим» — реальні фото готових виробів, решта плиток — інтерактивні 3D-моделі (можна покрутити пальцем чи мишею)." },
    { q: "З якого матеріалу друкуються ці зразки?", a: "Усі зразки — з біопластику Eco PLA, того самого матеріалу, яким друкуються замовлення покупців." },
    { q: "Чи можна замовити точно такий самий розмір чи район?", a: "Так — у конструкторі можна обрати будь-яку ділянку та розмір, включно з тими, що показані в галереї." },
  ],
  en: [
    { q: "Are these real photos or 3D renders?", a: "Both: the 'Printed in real life' section shows real photos of finished items, while the rest of the grid is interactive 3D models you can rotate with a finger or mouse." },
    { q: "What material are these samples printed in?", a: "All samples are printed in Eco PLA bioplastic — the same material used for customer orders." },
    { q: "Can I order the exact same size or district?", a: "Yes — the builder lets you pick any area and size, including the ones shown in the gallery." },
  ],
};

export function proseFaq(page: "create" | "worlds" | "showcase", locale: string): ProseFaqItem[] {
  const dict = page === "create" ? CREATE_FAQ : page === "worlds" ? WORLDS_FAQ : SHOWCASE_FAQ;
  return dict[locale === "uk" ? "uk" : "en"];
}
