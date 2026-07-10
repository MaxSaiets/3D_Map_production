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

export function seoProse(page: "create" | "keychains", locale: string): SeoProse {
  const dict = page === "create" ? CREATE : KEYCHAINS;
  return dict[locale === "uk" ? "uk" : "en"];
}
