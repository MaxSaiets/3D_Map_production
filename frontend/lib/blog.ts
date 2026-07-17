// ──────────────────────────────────────────────────────────────────────────
// БЛОГ (SEO-статті): контент живе тут (як lib/legal/content.ts), БЕЗ messages —
// щоб не роздувати i18n-словники. uk + en повні; de/es/fr/pl → en-фолбек.
// Кожна стаття: керований slug, дата, секції. Рендер: app/[locale]/blog/*.
// Мета: контент-глибина під довгі запити («як зробити 3d мапу міста»,
// «подарунок 3d карта», «брелок з маршрутом gpx», «3d мапа києва»...) —
// сторінки, що можуть ранжуватись самі і лінкують у конструктор/каталог.
// ──────────────────────────────────────────────────────────────────────────
import type { AppLocale } from "@/i18n/routing";

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
  // uk + en завжди; de/es/fr/pl додаються поступово. Доступ через blogContent()
  // з м'яким en-фолбеком, щоб частково перекладені статті не падали.
  content: { uk: BlogArticleContent; en: BlogArticleContent } & Partial<Record<AppLocale, BlogArticleContent>>;
};

/** Контент статті для локалі з м'яким фолбеком на en (для ще-не-перекладених de/es/fr/pl). */
export function blogContent(article: BlogArticle, locale: string): BlogArticleContent {
  return article.content[locale as AppLocale] ?? article.content.en;
}

type BlogIndexMeta = { title: string; description: string; h1: string; intro: string; readLabel: string };

export const BLOG_INDEX_META: Partial<Record<AppLocale, BlogIndexMeta>> = {
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
  de: {
    title: "Blog: 3D-Stadtkarten, Anhänger & 3D-Druck — Monadruk",
    description:
      "Anleitungen und Ideen: wie man eine 3D-Stadtkarte erstellt, Geschenkideen mit Karte, Anhänger mit GPX-Route, Tipps zum 3D-Druck.",
    h1: "Monadruk-Blog",
    intro: "Anleitungen und Ideen: persönliche 3D-Stadtkarten, Routen-Anhänger und Geschenke mit Bedeutung.",
    readLabel: "Lesen",
  },
  pl: {
    title: "Blog: mapy miast 3D, breloki i druk 3D — Monadruk",
    description:
      "Poradniki i pomysły: jak stworzyć mapę miasta 3D, pomysły na prezent z mapą, breloki z trasą GPX, porady dotyczące druku 3D.",
    h1: "Blog Monadruk",
    intro: "Poradniki i pomysły: spersonalizowane mapy miast 3D, breloki z trasami i prezenty, które coś znaczą.",
    readLabel: "Czytaj",
  },
  fr: {
    title: "Blog : cartes de ville 3D, porte-clés et impression 3D — Monadruk",
    description:
      "Guides et idées : comment créer une carte de ville 3D, idées de cadeaux avec carte, porte-clés avec trace GPX, conseils d'impression 3D.",
    h1: "Blog Monadruk",
    intro: "Guides et idées : cartes de ville 3D personnalisées, porte-clés d'itinéraire et cadeaux qui ont du sens.",
    readLabel: "Lire",
  },
  es: {
    title: "Blog: mapas de ciudad 3D, llaveros e impresión 3D — Monadruk",
    description:
      "Guías e ideas: cómo crear un mapa de ciudad 3D, ideas de regalo con mapa, llaveros con ruta GPX, consejos de impresión 3D.",
    h1: "Blog Monadruk",
    intro: "Guías e ideas: mapas de ciudad 3D personalizados, llaveros de ruta y regalos con significado.",
    readLabel: "Leer",
  },
};

/** Мета блог-індексу для локалі з en-фолбеком. */
export function blogIndexMeta(locale: string): BlogIndexMeta {
  return BLOG_INDEX_META[locale as AppLocale] ?? BLOG_INDEX_META.en!;
}

/** Локаль для блогу: uk|en повні; de/es/fr/pl де вже перекладено, інакше en-фолбек у blogContent/blogIndexMeta. */
export function blogLocale(locale: string): AppLocale {
  return (["uk", "en", "de", "pl", "fr", "es"].includes(locale) ? locale : "en") as AppLocale;
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
      de: {
        title: "3D-Stadtkarte zum Drucken erstellen: Schritt für Schritt",
        description:
          "Schritt für Schritt: In 5 Minuten eine 3D-Karte deines Viertels erstellen — Bereich, Größe, Relief wählen, zu Hause drucken oder bestellen. Kostenloser Konfigurator.",
        h1: "3D-Stadtkarte erstellen: Schritt für Schritt",
        intro:
          "Eine 3D-Stadtkarte ist ein physisches Modell eines Viertels mit echten Gebäuden, Straßen, Parks und Flüssen. Du kannst sie auf einem 3D-Drucker drucken, ins Regal stellen oder verschenken. Diese Anleitung zeigt, wie du eine solche Karte in wenigen Minuten selbst erstellst — ganz ohne 3D-Modellierungskenntnisse.",
        sections: [
          {
            h2: "Schritt 1. Bereich auf der Karte wählen",
            p: [
              "Öffne den Konfigurator und finde deine Stadt — Kyiv, Lwiw, Odessa oder jeden Punkt der Welt. Schiebe den Rahmen über das Viertel, das du festhalten möchtest: deine Straße, das Stadtzentrum, den Ort des ersten Treffens. Was im Rahmen liegt, wird zum Modell.",
              "Tipp: Bereiche mit gemischtem Inhalt wirken am besten — ein paar Straßen, ein Park, ein Fluss. Für ein 8–11-cm-Modell ist ein Bereich von 400–800 Metern ideal.",
            ],
          },
          {
            h2: "Schritt 2. Größe und Relief",
            p: [
              "Wähle die Endgröße: S (5,5 cm) ist ein kompaktes Souvenir, M (8 cm) die goldene Mitte, L (11 cm) und XL (15 cm) fallen im Regal auf. Der Preis richtet sich nach der Größe — ab ≈6 €.",
              "Ist die Gegend hügelig (Kyiv, Lwiw, die Karpaten), aktiviere das Relief: Das Modell erhält echte Höhenunterschiede, und die Hügel sind sichtbar und fühlbar. Für flache Städte kannst du es weglassen.",
            ],
          },
          {
            h2: "Schritt 3. Generieren und prüfen",
            p: [
              "Klicke auf «Erstellen» — in 2–4 Minuten baut der Dienst das Modell aus OpenStreetMap-Daten: Gebäude mit echten Höhen, Straßen, Grünflächen, Wasser. Das fertige Modell kannst du direkt im Browser drehen.",
            ],
          },
          {
            h2: "Schritt 4. Zu Hause drucken oder bestellen",
            p: [
              "Hast du einen Drucker? Lade die fertige 3MF- oder STL-Datei herunter — sie öffnet sich direkt in Bambu Studio oder PrusaSlicer, mit vorbelegten Schichtfarben.",
              "Kein Drucker? Bestelle den Druck: Wir drucken aus umweltfreundlichem Eco PLA und versenden in die Ukraine und in 15 EU-Länder. Kartenzahlung online oder bei Lieferung.",
            ],
          },
        ],
        ctaLabel: "Deine 3D-Karte erstellen",
        ctaHref: "/create",
        outro: "Das Erstellen des Modells ist kostenlos — du zahlst nur für Druck und Versand, wenn du das fertige Stück bestellst.",
      },
      pl: {
        title: "Jak stworzyć mapę miasta 3D do druku: krok po kroku",
        description:
          "Krok po kroku: zrób mapę 3D swojej okolicy w 5 minut — wybierz obszar, rozmiar, rzeźbę, wydrukuj w domu lub zamów. Darmowy kreator.",
        h1: "Jak stworzyć mapę miasta 3D: krok po kroku",
        intro:
          "Mapa miasta 3D to fizyczny model dzielnicy z prawdziwymi budynkami, ulicami, parkami i rzekami. Możesz ją wydrukować na drukarce 3D, postawić na półce lub podarować. Ten poradnik pokazuje, jak zrobić taką mapę w kilka minut, bez umiejętności modelowania 3D.",
        sections: [
          {
            h2: "Krok 1. Wybierz obszar na mapie",
            p: [
              "Otwórz kreator i znajdź swoje miasto — Kijów, Lwów, Odessę lub dowolny punkt na świecie. Przesuń ramkę na dzielnicę, którą chcesz zachować: swoją ulicę, centrum miasta, miejsce pierwszego spotkania. To, co znajdzie się w ramce, staje się modelem.",
              "Wskazówka: obszary o zróżnicowanej zabudowie wyglądają najlepiej — kilka ulic, park, rzeka. Dla modelu 8–11 cm idealny jest obszar 400–800 metrów.",
            ],
          },
          {
            h2: "Krok 2. Rozmiar i rzeźba terenu",
            p: [
              "Wybierz rozmiar końcowy: S (5,5 cm) to kompaktowa pamiątka, M (8 cm) złoty środek, L (11 cm) i XL (15 cm) wyróżniają się na półce. Cena zależy od rozmiaru — od ≈6 €.",
              "Jeśli teren jest pagórkowaty (Kijów, Lwów, Karpaty), włącz rzeźbę: model otrzyma prawdziwe różnice wysokości, a wzgórza będzie widać i można je wyczuć. Dla płaskich miast można ją pominąć.",
            ],
          },
          {
            h2: "Krok 3. Wygeneruj i sprawdź",
            p: [
              "Kliknij «Utwórz» — w 2–4 minuty serwis złoży model z danych OpenStreetMap: budynki o prawdziwych wysokościach, drogi, zieleń, woda. Gotowy model możesz obrócić bezpośrednio w przeglądarce.",
            ],
          },
          {
            h2: "Krok 4. Wydrukuj w domu lub zamów",
            p: [
              "Masz drukarkę? Pobierz gotowy plik 3MF lub STL — otwiera się bezpośrednio w Bambu Studio lub PrusaSlicer, z przypisanymi kolorami warstw.",
              "Nie masz drukarki? Zamów druk: drukujemy z ekologicznego Eco PLA i wysyłamy na Ukrainę oraz do 15 krajów UE. Płatność kartą online lub przy odbiorze.",
            ],
          },
        ],
        ctaLabel: "Utwórz swoją mapę 3D",
        ctaHref: "/create",
        outro: "Tworzenie modelu jest darmowe — płacisz tylko za druk i wysyłkę, jeśli zamówisz gotowy przedmiot.",
      },
      fr: {
        title: "Comment créer une carte de ville 3D à imprimer : étape par étape",
        description:
          "Étape par étape : créez une carte 3D de votre quartier en 5 minutes — zone, taille, relief, impression à la maison ou commande. Configurateur gratuit.",
        h1: "Comment créer une carte de ville 3D : étape par étape",
        intro:
          "Une carte de ville 3D est un modèle physique d'un quartier avec de vrais bâtiments, rues, parcs et rivières. Vous pouvez l'imprimer sur une imprimante 3D, la poser sur une étagère ou l'offrir. Ce guide montre comment en réaliser une en quelques minutes, sans compétences en modélisation 3D.",
        sections: [
          {
            h2: "Étape 1. Choisissez une zone sur la carte",
            p: [
              "Ouvrez le configurateur et trouvez votre ville — Kyiv, Lviv, Odessa ou n'importe quel point du globe. Déplacez le cadre sur le quartier que vous voulez garder : votre rue, le centre-ville, le lieu de votre première rencontre. Ce qui est dans le cadre devient le modèle.",
              "Astuce : les zones au contenu varié rendent le mieux — quelques rues, un parc, une rivière. Pour un modèle de 8–11 cm, une zone de 400–800 mètres est idéale.",
            ],
          },
          {
            h2: "Étape 2. Taille et relief",
            p: [
              "Choisissez la taille finale : S (5,5 cm) est un souvenir compact, M (8 cm) le juste milieu, L (11 cm) et XL (15 cm) se remarquent sur une étagère. Le prix dépend de la taille — dès ≈6 €.",
              "Si le terrain est vallonné (Kyiv, Lviv, les Carpates), activez le relief : le modèle obtient de vrais dénivelés et les collines se voient et se touchent. Pour les villes plates, vous pouvez l'omettre.",
            ],
          },
          {
            h2: "Étape 3. Générez et vérifiez",
            p: [
              "Cliquez sur «Créer» — en 2–4 minutes le service assemble le modèle à partir des données OpenStreetMap : bâtiments aux hauteurs réelles, routes, espaces verts, eau. Vous pouvez faire pivoter le modèle terminé dans le navigateur.",
            ],
          },
          {
            h2: "Étape 4. Imprimez chez vous ou commandez",
            p: [
              "Vous avez une imprimante ? Téléchargez le fichier 3MF ou STL prêt — il s'ouvre directement dans Bambu Studio ou PrusaSlicer, avec les couleurs de couche déjà attribuées.",
              "Pas d'imprimante ? Commandez l'impression : nous imprimons en Eco PLA écologique et livrons en Ukraine et dans 15 pays de l'UE. Paiement par carte en ligne ou à la livraison.",
            ],
          },
        ],
        ctaLabel: "Créer votre carte 3D",
        ctaHref: "/create",
        outro: "Créer le modèle est gratuit — vous ne payez que l'impression et la livraison si vous commandez la pièce finie.",
      },
      es: {
        title: "Cómo crear un mapa de ciudad 3D para imprimir: paso a paso",
        description:
          "Paso a paso: haz un mapa 3D de tu barrio en 5 minutos — elige zona, tamaño, relieve, imprime en casa o pídelo. Configurador gratis.",
        h1: "Cómo crear un mapa de ciudad 3D: paso a paso",
        intro:
          "Un mapa de ciudad 3D es un modelo físico de un barrio con edificios, calles, parques y ríos reales. Puedes imprimirlo en una impresora 3D, ponerlo en una estantería o regalarlo. Esta guía muestra cómo hacer uno en minutos, sin conocimientos de modelado 3D.",
        sections: [
          {
            h2: "Paso 1. Elige una zona en el mapa",
            p: [
              "Abre el configurador y encuentra tu ciudad — Kyiv, Leópolis, Odesa o cualquier punto del mundo. Mueve el marco sobre el barrio que quieres conservar: tu calle, el centro de la ciudad, el lugar del primer encuentro. Lo que queda dentro del marco se convierte en el modelo.",
              "Consejo: las zonas con contenido variado quedan mejor — algunas calles, un parque, un río. Para un modelo de 8–11 cm, una zona de 400–800 metros es lo ideal.",
            ],
          },
          {
            h2: "Paso 2. Tamaño y relieve",
            p: [
              "Elige el tamaño final: S (5,5 cm) es un recuerdo compacto, M (8 cm) el punto justo, L (11 cm) y XL (15 cm) destacan en una estantería. El precio depende del tamaño — desde ≈6 €.",
              "Si el terreno es montañoso (Kyiv, Leópolis, los Cárpatos), activa el relieve: el modelo obtiene desniveles reales y las colinas se ven y se tocan. Para ciudades llanas puedes omitirlo.",
            ],
          },
          {
            h2: "Paso 3. Genera y revisa",
            p: [
              "Pulsa «Crear» — en 2–4 minutos el servicio ensambla el modelo con datos de OpenStreetMap: edificios con alturas reales, carreteras, zonas verdes, agua. Puedes girar el modelo terminado en el navegador.",
            ],
          },
          {
            h2: "Paso 4. Imprime en casa o pídelo",
            p: [
              "¿Tienes impresora? Descarga el archivo 3MF o STL listo — se abre directamente en Bambu Studio o PrusaSlicer, con los colores de capa ya asignados.",
              "¿Sin impresora? Pide la impresión: imprimimos en Eco PLA ecológico y enviamos a Ucrania y a 15 países de la UE. Pago con tarjeta online o contra entrega.",
            ],
          },
        ],
        ctaLabel: "Crear tu mapa 3D",
        ctaHref: "/create",
        outro: "Crear el modelo es gratis — solo pagas la impresión y el envío si pides la pieza terminada.",
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
      de: {
        title: "Was schenkt man jemandem, der alles hat: eine persönliche 3D-Karte",
        description:
          "Ein Geschenk, das niemand sonst hat: eine 3D-Karte eines Ortes mit Bedeutung — der Kindheitshof, der Ort des Kennenlernens, die erste gemeinsame Wohnung. Ab ≈6 €.",
        h1: "Was schenkt man jemandem, der alles hat",
        intro:
          "Die schwierigsten Geschenke sind für Menschen, die «alles haben». Was wirkt, ist Bedeutung: eine persönliche 3D-Karte ist ein Stück Stadt, das mit ihrer Geschichte verbunden ist — der Hof der Kindheit, die Straße des ersten Dates, die Wohnung, in die sie gerade eingezogen sind.",
        sections: [
          {
            h2: "Warum eine Karte als Geschenk wirkt",
            p: [
              "Sie ist ein Einzelstück — du wählst den genauen Bereich. Menschen erkennen ihr Viertel in einer Sekunde: «das ist ja unser Haus!» — und genau dieser Moment macht das Geschenk unvergesslich.",
              "Sie passt zu fast jedem Anlass: Jahrestag (der Ort des Kennenlernens), Einzug (das neue Viertel), Geburtstag (die Heimatstadt eines Weggezogenen), Abschluss (das Uni-Viertel), eine scheidende Kollegin (das Büro-Viertel).",
            ],
          },
          {
            h2: "Formate für jedes Budget",
            p: [
              "Karten-Anhänger (ab ≈3 €) — eine kleine Aufmerksamkeit mit eigenem Text auf der Rückseite. Kühlschrankmagnet (≈4 €) — eine tägliche Erinnerung an einen Ort. 3D-Karte fürs Regal (≈6–13 € je nach Größe) — ein echtes Interieurstück, auf Wunsch mit Geländerelief.",
            ],
          },
          {
            h2: "In 5 Minuten bestellen",
            p: [
              "Wähle den Bereich im Konfigurator und eine Größe — das Modell wird automatisch erstellt. Wir drucken aus Eco PLA und versenden in die Ukraine und die EU; auch die Digitaldatei ist erhältlich. Fertigung 1–3 Werktage.",
            ],
          },
        ],
        ctaLabel: "Ein Karten-Geschenk auswählen",
        ctaHref: "/podarunok",
        outro: "Unsicher, welchen Bereich wählen? Schreib uns — wir helfen und bauen eine Vorschau.",
      },
      pl: {
        title: "Co podarować komuś, kto ma wszystko: spersonalizowana mapa 3D",
        description:
          "Prezent, którego nie ma nikt inny: mapa 3D miejsca, które coś znaczy — rodzinne podwórko, miejsce poznania, pierwsze wspólne mieszkanie. Od ≈6 €.",
        h1: "Co podarować komuś, kto ma wszystko",
        intro:
          "Najtrudniejsze prezenty są dla osób, które «mają wszystko». Działa znaczenie: spersonalizowana mapa 3D to kawałek miasta związany z ich historią — podwórko z dzieciństwa, ulica pierwszej randki, mieszkanie, do którego właśnie się wprowadzili.",
        sections: [
          {
            h2: "Dlaczego mapa działa jako prezent",
            p: [
              "To egzemplarz jedyny w swoim rodzaju — wybierasz dokładny obszar. Ludzie rozpoznają swoją dzielnicę w sekundę: «to nasz dom!» — i właśnie ten moment czyni prezent niezapomnianym.",
              "Pasuje niemal na każdą okazję: rocznica (miejsce poznania), parapetówka (nowa dzielnica), urodziny (rodzinne miasto osoby, która wyjechała), ukończenie studiów (dzielnica uczelni), odchodzący współpracownik (dzielnica biura).",
            ],
          },
          {
            h2: "Formaty na każdy budżet",
            p: [
              "Brelok-mapa (od ≈3 €) — drobny gest z własnym napisem z tyłu. Magnes na lodówkę (≈4 €) — codzienne przypomnienie o miejscu. Mapa 3D na półkę (≈6–13 € zależnie od rozmiaru) — pełnoprawny element wnętrza, opcjonalnie z rzeźbą terenu.",
            ],
          },
          {
            h2: "Zamów w 5 minut",
            p: [
              "Wybierz obszar w kreatorze i rozmiar — model powstaje automatycznie. Drukujemy z Eco PLA i wysyłamy na Ukrainę oraz do UE; dostępny jest też plik cyfrowy. Wykonanie 1–3 dni robocze.",
            ],
          },
        ],
        ctaLabel: "Wybierz prezent-mapę",
        ctaHref: "/podarunok",
        outro: "Nie wiesz, jaki obszar wybrać? Napisz do nas — pomożemy i przygotujemy podgląd.",
      },
      fr: {
        title: "Que offrir à quelqu'un qui a tout : une carte 3D personnalisée",
        description:
          "Un cadeau que personne d'autre n'a : une carte 3D d'un lieu qui compte — la cour d'enfance, le lieu de la rencontre, le premier logement commun. Dès ≈6 €.",
        h1: "Que offrir à quelqu'un qui a tout",
        intro:
          "Les cadeaux les plus difficiles sont pour ceux qui «ont tout». Ce qui marche, c'est le sens : une carte 3D personnalisée est un morceau de ville lié à leur histoire — la cour d'enfance, la rue d'un premier rendez-vous, le logement où ils viennent d'emménager.",
        sections: [
          {
            h2: "Pourquoi une carte fonctionne en cadeau",
            p: [
              "C'est une pièce unique — vous choisissez la zone exacte. Les gens reconnaissent leur quartier en une seconde : «mais c'est notre maison !» — et c'est ce moment qui rend le cadeau mémorable.",
              "Elle convient à presque toute occasion : anniversaire de rencontre, pendaison de crémaillère (le nouveau quartier), anniversaire (la ville natale d'un proche parti), remise de diplôme (le quartier du campus), un collègue qui part (le quartier du bureau).",
            ],
          },
          {
            h2: "Des formats pour tous les budgets",
            p: [
              "Porte-clés carte (dès ≈3 €) — une petite attention avec un texte au dos. Magnet de frigo (≈4 €) — un rappel quotidien d'un lieu. Carte 3D pour l'étagère (≈6–13 € selon la taille) — une vraie pièce d'intérieur, avec relief du terrain si vous voulez.",
            ],
          },
          {
            h2: "Commander en 5 minutes",
            p: [
              "Choisissez la zone dans le configurateur et une taille — le modèle est assemblé automatiquement. Nous imprimons en Eco PLA et livrons en Ukraine et dans l'UE ; le fichier numérique est aussi disponible. Fabrication 1–3 jours ouvrés.",
            ],
          },
        ],
        ctaLabel: "Choisir un cadeau-carte",
        ctaHref: "/podarunok",
        outro: "Vous hésitez sur la zone ? Écrivez-nous — on vous aide et on prépare un aperçu.",
      },
      es: {
        title: "Qué regalar a quien lo tiene todo: un mapa 3D personalizado",
        description:
          "Un regalo que nadie más tiene: un mapa 3D de un lugar que significa algo — el patio de la infancia, el lugar donde os conocisteis, el primer hogar juntos. Desde ≈6 €.",
        h1: "Qué regalar a quien lo tiene todo",
        intro:
          "Los regalos más difíciles son para quienes «lo tienen todo». Lo que funciona es el significado: un mapa 3D personalizado es un trozo de ciudad ligado a su historia — el patio de la infancia, la calle de una primera cita, la casa a la que acaban de mudarse.",
        sections: [
          {
            h2: "Por qué un mapa funciona como regalo",
            p: [
              "Es una pieza única — tú eliges la zona exacta. La gente reconoce su barrio en un segundo: «¡esa es nuestra casa!» — y ese momento hace el regalo memorable.",
              "Encaja en casi cualquier ocasión: aniversario (dónde os conocisteis), inauguración de casa (el nuevo barrio), cumpleaños (la ciudad natal de alguien que se mudó), graduación (el barrio del campus), un compañero que se va (el barrio de la oficina).",
            ],
          },
          {
            h2: "Formatos para cada presupuesto",
            p: [
              "Llavero-mapa (desde ≈3 €) — un pequeño detalle con texto propio al dorso. Imán de nevera (≈4 €) — un recordatorio diario de un lugar. Mapa 3D para la estantería (≈6–13 € según el tamaño) — una auténtica pieza de interior, con relieve del terreno si quieres.",
            ],
          },
          {
            h2: "Pide en 5 minutos",
            p: [
              "Elige la zona en el configurador y un tamaño — el modelo se genera automáticamente. Imprimimos en Eco PLA y enviamos a Ucrania y la UE; también está disponible el archivo digital. Fabricación 1–3 días hábiles.",
            ],
          },
        ],
        ctaLabel: "Elegir un regalo-mapa",
        ctaHref: "/podarunok",
        outro: "¿No sabes qué zona elegir? Escríbenos — te ayudamos y preparamos una vista previa.",
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
      de: {
        title: "Schlüsselanhänger mit Stadtkarte: persönliches Accessoire ab ≈3 €",
        description:
          "Karten-Anhänger 55×30 mm: Straßen und Parks deines Viertels als Relief, eigener Text auf der Rückseite. 3D-Druck auf Bestellung ab ≈3 €, Versand Ukraine & EU.",
        h1: "Schlüsselanhänger mit Stadtkarte — eine kleine Karte, die immer dabei ist",
        intro:
          "Der Karten-Anhänger ist ein 55×30-mm-Täfelchen, auf dem das gewählte Viertel als Relief gedruckt ist: Straßen, Parks, Flüsse. Auf der Rückseite dein Text: Stadtname, Datum, Name oder Koordinaten. Die günstigste Art, einen Ort bei sich zu tragen, der etwas bedeutet.",
        sections: [
          {
            h2: "Was man darauf abbilden kann",
            p: [
              "Jeden Punkt der Erde: dein Heimatviertel, die Stadt, in der dein Kind geboren wurde, die Straße, in der ihr als Studenten gewohnt habt. Eine Meeresküste mit Strandlinie oder ein Bergkamm mit Höhenrelief geht auch — für Berge gibt es einen eigenen Topo-Modus, in dem Gipfel und Täler unter den Fingern lesbar werden.",
              "Beliebte Varianten: ein Paar «Herz»-Anhänger mit den Vierteln zweier Menschen (sie fügen sich wie ein Puzzle zusammen), ein Läufer-Anhänger mit GPX-Track der Lieblingsstrecke, Firmen-Anhänger mit dem Büro-Viertel fürs Team.",
            ],
          },
          {
            h2: "Material und Qualität",
            p: [
              "Gedruckt aus Eco-PLA-Bioplastik — leicht, angenehm griffig, hält den täglichen Gebrauch am Schlüssel aus. Straßenlinien werden mit 0,4 mm Auflösung gedruckt, sodass selbst kleine Gassen lesbar bleiben.",
            ],
          },
          {
            h2: "Preis und Lieferzeit",
            p: [
              "Karten-Anhänger — ab ≈3 €. Fertigung 1–3 Werktage, Versand mit Nova Poshta in der Ukraine oder Nova Post/Meest in 15 EU-Länder. Kartenzahlung online oder bei Lieferung.",
            ],
          },
        ],
        ctaLabel: "Anhänger mit deinem Viertel erstellen",
        ctaHref: "/keychains",
      },
      pl: {
        title: "Brelok z mapą miasta: osobiste akcesorium od ≈3 €",
        description:
          "Brelok-mapa 55×30 mm: ulice i parki Twojej dzielnicy w reliefie, własny napis z tyłu. Druk 3D na zamówienie od ≈3 €, wysyłka Ukraina i UE.",
        h1: "Brelok z mapą miasta — mała mapa, która zawsze jest przy Tobie",
        intro:
          "Brelok-mapa to zawieszka 55×30 mm, na której wybraną dzielnicę wydrukowano reliefem: ulice, parki, rzeki. Z tyłu Twój napis: nazwa miasta, data, imię lub współrzędne. Najtańszy sposób, by nosić przy sobie miejsce, które coś znaczy.",
        sections: [
          {
            h2: "Co można przedstawić",
            p: [
              "Dowolny punkt świata: rodzinną dzielnicę, miasto, w którym urodziło się dziecko, ulicę, przy której mieszkaliście jako studenci. Wybrzeże morza z linią plaży lub grzbiet górski z reliefem wysokości też działa — dla gór jest osobny tryb topo, w którym szczyty i doliny czyta się palcami.",
              "Popularne warianty: para breloków-«serc» z dzielnicami dwóch osób (łączą się jak puzzle), brelok biegacza ze śladem GPX ulubionej trasy, firmowe breloki z dzielnicą biura dla zespołu.",
            ],
          },
          {
            h2: "Materiał i jakość",
            p: [
              "Drukujemy z bioplastiku Eco PLA — lekki, przyjemny w dotyku, wytrzymuje codzienne noszenie przy kluczach. Linie ulic drukowane z rozdzielczością 0,4 mm — nawet małe uliczki pozostają czytelne.",
            ],
          },
          {
            h2: "Cena i termin",
            p: [
              "Brelok-mapa — od ≈3 €. Wykonanie 1–3 dni robocze, wysyłka Nova Poshta na Ukrainie lub Nova Post/Meest do 15 krajów UE. Płatność kartą online lub przy odbiorze.",
            ],
          },
        ],
        ctaLabel: "Stwórz brelok ze swoją dzielnicą",
        ctaHref: "/keychains",
      },
      fr: {
        title: "Porte-clés avec carte de ville : accessoire personnel dès ≈3 €",
        description:
          "Porte-clés carte 55×30 mm : les rues et parcs de votre quartier en relief, texte personnel au dos. Impression 3D sur mesure dès ≈3 €, livraison Ukraine et UE.",
        h1: "Porte-clés avec carte de ville — une petite carte toujours avec vous",
        intro:
          "Le porte-clés carte est une plaque de 55×30 mm sur laquelle le quartier choisi est imprimé en relief : rues, parcs, rivières. Au dos, votre texte : nom de ville, date, prénom ou coordonnées. La façon la plus abordable de porter sur soi un lieu qui compte.",
        sections: [
          {
            h2: "Ce qu'on peut y représenter",
            p: [
              "N'importe quel point du globe : votre quartier natal, la ville où votre enfant est né, la rue où vous viviez étudiants. Un littoral avec la ligne de plage ou une crête de montagne en relief fonctionne aussi — pour la montagne il y a un mode topo dédié où sommets et vallées se lisent du bout des doigts.",
              "Choix populaires : une paire de porte-clés «cœur» avec les quartiers de deux personnes (ils s'emboîtent comme un puzzle), un porte-clés de coureur avec une trace GPX, des porte-clés d'entreprise avec le quartier du bureau pour l'équipe.",
            ],
          },
          {
            h2: "Matériau et qualité",
            p: [
              "Imprimé en Eco PLA — léger, agréable au toucher, supporte l'usage quotidien sur les clés. Les lignes de rues sont imprimées à 0,4 mm de résolution, si bien que même les petites ruelles restent lisibles.",
            ],
          },
          {
            h2: "Prix et délai",
            p: [
              "Porte-clés carte — dès ≈3 €. Fabrication 1–3 jours ouvrés, livraison par Nova Poshta en Ukraine ou Nova Post/Meest dans 15 pays de l'UE. Paiement par carte en ligne ou à la livraison.",
            ],
          },
        ],
        ctaLabel: "Créer un porte-clés avec votre quartier",
        ctaHref: "/keychains",
      },
      es: {
        title: "Llavero con mapa de ciudad: accesorio personal desde ≈3 €",
        description:
          "Llavero-mapa 55×30 mm: las calles y parques de tu barrio en relieve, texto personal al dorso. Impresión 3D a medida desde ≈3 €, envío Ucrania y UE.",
        h1: "Llavero con mapa de ciudad — un pequeño mapa que siempre te acompaña",
        intro:
          "El llavero-mapa es una placa de 55×30 mm en la que el barrio elegido está impreso en relieve: calles, parques, ríos. Al dorso, tu texto: nombre de la ciudad, fecha, nombre o coordenadas. La forma más asequible de llevar contigo un lugar que significa algo.",
        sections: [
          {
            h2: "Qué se puede representar",
            p: [
              "Cualquier punto del mundo: tu barrio natal, la ciudad donde nació tu hijo, la calle donde vivíais de estudiantes. Una costa con la línea de playa o una cresta de montaña con relieve también funciona — para montañas hay un modo topo propio donde cumbres y valles se leen con los dedos.",
              "Opciones populares: un par de llaveros «corazón» con los barrios de dos personas (encajan como un puzle), un llavero de corredor con un track GPX, llaveros corporativos con el barrio de la oficina para el equipo.",
            ],
          },
          {
            h2: "Material y calidad",
            p: [
              "Impreso en bioplástico Eco PLA — ligero, agradable al tacto, aguanta el uso diario en las llaves. Las líneas de las calles se imprimen a 0,4 mm de resolución, de modo que hasta las callejuelas pequeñas siguen siendo legibles.",
            ],
          },
          {
            h2: "Precio y plazo",
            p: [
              "Llavero-mapa — desde ≈3 €. Fabricación 1–3 días hábiles, envío con Nova Poshta en Ucrania o Nova Post/Meest a 15 países de la UE. Pago con tarjeta online o contra entrega.",
            ],
          },
        ],
        ctaLabel: "Crear un llavero con tu barrio",
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
      de: {
        title: "Schlüsselanhänger mit GPX-Route: verewige deinen Lauf, deine Wanderung oder Radtour",
        description:
          "Lade einen GPX-Track aus Strava oder Garmin hoch — und erhalte einen Anhänger mit der Reliefline deiner Route über der Viertelkarte. Marathon- oder Wander-Andenken ab ≈3 €.",
        h1: "Schlüsselanhänger mit GPX-Route: dein Track in Plastik",
        intro:
          "Der erste Marathon, eine 100-km-Radtour, eine Hoverla-Wanderung — der Track dieser Routen liegt in Strava oder Garmin und niemand sieht ihn. Ein GPX-Anhänger macht den Track zu einem physischen Objekt: die Reliefline der Route verläuft über der Viertelkarte direkt auf dem Anhänger, der immer am Schlüssel ist.",
        sections: [
          {
            h2: "Wie es funktioniert",
            p: [
              "Exportiere eine GPX-Datei aus Strava, Garmin Connect, Komoot oder einem beliebigen Tracker. Lade sie in den Konfigurator — der Dienst findet den Ort der Route, wählt den Maßstab, damit der ganze Track passt, und legt die Routenlinie über die Straßen. Die Linie wird an die Straßen angezogen, sodass selbst eine «verrauschte» GPS-Aufzeichnung sauber aussieht.",
            ],
          },
          {
            h2: "Wem man das schenkt",
            p: [
              "Läufern — das Ziel des ersten Marathons oder die Lieblingsrunde durch den Park. Radfahrern — die Route einer Mehrtagestour. Wanderern — der Track einer Karpaten-Tour. Ein Geschenk, das man nicht fertig kaufen kann: jeder hat seine eigene Route.",
              "Auf der Rückseite lässt sich Text hinzufügen: Eventname, Datum, Zielzeit — «Kyiv Marathon 2026 · 3:58».",
            ],
          },
          {
            h2: "Preis",
            p: [
              "Ein GPX-Anhänger kostet so viel wie ein normaler Karten-Anhänger — ab ≈3 €. Fertigung 1–3 Werktage, Versand in die Ukraine und die EU.",
            ],
          },
        ],
        ctaLabel: "Deinen GPX-Track hochladen",
        ctaHref: "/keychains",
      },
      pl: {
        title: "Brelok z trasą GPX: uwiecznij swój bieg, wędrówkę lub trasę rowerową",
        description:
          "Wgraj ślad GPX ze Stravy lub Garmina — i otrzymaj brelok z reliefową linią swojej trasy na mapie dzielnicy. Pamiątka z maratonu lub wędrówki od ≈3 €.",
        h1: "Brelok z trasą GPX: Twój ślad w plastiku",
        intro:
          "Pierwszy maraton, setka na rowerze, wejście na Howerlę — ślad tych tras leży w Stravie lub Garminie i nikt go nie widzi. Brelok GPX zamienia ślad w fizyczną rzecz: reliefowa linia trasy biegnie po mapie dzielnicy prosto na zawieszce, która zawsze jest przy kluczach.",
        sections: [
          {
            h2: "Jak to działa",
            p: [
              "Wyeksportuj plik GPX ze Stravy, Garmin Connect, Komoot lub dowolnego trackera. Wgraj go do kreatora — serwis sam znajdzie miejsce trasy, dobierze skalę, by cały ślad się zmieścił, i nałoży linię trasy na ulice. Linia trasy przyciąga się do dróg, więc wygląda schludnie nawet przy «zaszumionym» zapisie GPS.",
            ],
          },
          {
            h2: "Komu się to daruje",
            p: [
              "Biegaczom — meta pierwszego maratonu lub ulubiona pętla po parku. Rowerzystom — trasa kilkudniowej wyprawy. Turystom — ślad wędrówki w Karpaty. To prezent, którego nie można kupić gotowego: każdy ma swoją trasę.",
              "Z tyłu można dodać napis: nazwę wydarzenia, datę, czas mety — «Kyiv Marathon 2026 · 3:58».",
            ],
          },
          {
            h2: "Cena",
            p: [
              "Brelok ze śladem GPX kosztuje tyle co zwykły brelok-mapa — od ≈3 €. Wykonanie 1–3 dni robocze, wysyłka na Ukrainę i do UE.",
            ],
          },
        ],
        ctaLabel: "Wgraj swój ślad GPX",
        ctaHref: "/keychains",
      },
      fr: {
        title: "Porte-clés avec trace GPX : immortalisez votre course, randonnée ou sortie vélo",
        description:
          "Téléversez une trace GPX depuis Strava ou Garmin — et obtenez un porte-clés avec la ligne en relief de votre parcours sur la carte du quartier. Souvenir de marathon ou de rando dès ≈3 €.",
        h1: "Porte-clés avec trace GPX : votre parcours en plastique",
        intro:
          "Le premier marathon, une sortie de 100 km à vélo, une montée à la Hoverla — la trace de ces parcours dort dans Strava ou Garmin et personne ne la voit. Un porte-clés GPX transforme la trace en objet physique : la ligne en relief du parcours passe sur la carte du quartier, à même la plaque toujours sur vos clés.",
        sections: [
          {
            h2: "Comment ça marche",
            p: [
              "Exportez un fichier GPX depuis Strava, Garmin Connect, Komoot ou n'importe quel traceur. Téléversez-le dans le configurateur — le service localise le parcours, choisit l'échelle pour que toute la trace tienne, et pose la ligne du parcours sur les rues. La ligne s'aligne sur les routes, si bien qu'un enregistrement GPS «bruité» reste net.",
            ],
          },
          {
            h2: "À qui l'offrir",
            p: [
              "Aux coureurs — l'arrivée du premier marathon ou la boucle préférée dans le parc. Aux cyclistes — l'itinéraire d'un raid de plusieurs jours. Aux randonneurs — la trace d'une sortie dans les Carpates. Un cadeau qu'on ne peut pas acheter tout fait : chacun a son propre parcours.",
              "On peut ajouter un texte au dos : nom de l'événement, date, temps d'arrivée — «Kyiv Marathon 2026 · 3:58».",
            ],
          },
          {
            h2: "Prix",
            p: [
              "Un porte-clés GPX coûte comme un porte-clés carte classique — dès ≈3 €. Fabrication 1–3 jours ouvrés, livraison en Ukraine et dans l'UE.",
            ],
          },
        ],
        ctaLabel: "Téléverser votre trace GPX",
        ctaHref: "/keychains",
      },
      es: {
        title: "Llavero con ruta GPX: inmortaliza tu carrera, senderismo o ruta en bici",
        description:
          "Sube un track GPX de Strava o Garmin — y obtén un llavero con la línea en relieve de tu ruta sobre el mapa del barrio. Recuerdo de maratón o excursión desde ≈3 €.",
        h1: "Llavero con ruta GPX: tu track en plástico",
        intro:
          "El primer maratón, una ruta de 100 km en bici, una subida a la Hoverla — el track de esas rutas está en Strava o Garmin y nadie lo ve. Un llavero GPX convierte el track en un objeto físico: la línea en relieve de la ruta recorre el mapa del barrio en la propia placa que siempre llevas en las llaves.",
        sections: [
          {
            h2: "Cómo funciona",
            p: [
              "Exporta un archivo GPX de Strava, Garmin Connect, Komoot o cualquier tracker. Súbelo al configurador — el servicio localiza la ruta, elige la escala para que quepa todo el track y coloca la línea de la ruta sobre las calles. La línea se ajusta a las carreteras, así que hasta un registro GPS «con ruido» queda limpio.",
            ],
          },
          {
            h2: "A quién regalarlo",
            p: [
              "A corredores — la meta del primer maratón o la vuelta favorita por el parque. A ciclistas — la ruta de una travesía de varios días. A excursionistas — el track de una salida a los Cárpatos. Un regalo que no se puede comprar hecho: cada uno tiene su propia ruta.",
              "Se puede añadir texto al dorso: nombre del evento, fecha, tiempo de meta — «Kyiv Marathon 2026 · 3:58».",
            ],
          },
          {
            h2: "Precio",
            p: [
              "Un llavero GPX cuesta lo mismo que un llavero-mapa normal — desde ≈3 €. Fabricación 1–3 días hábiles, envío a Ucrania y la UE.",
            ],
          },
        ],
        ctaLabel: "Subir tu track GPX",
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
      de: {
        title: "3D-Karte von Kiew: dein Lieblingsviertel im Regal",
        description:
          "Podil, Petschersk, Obolon oder Rusaniwka — eine 3D-Karte jedes Kiewer Viertels mit echten Gebäuden und dem Relief der Dnipro-Hänge. Druck ab ≈6 €.",
        h1: "3D-Karte von Kiew: eine Stadt, die man in der Hand halten kann",
        intro:
          "Kiew hat in jedem Viertel Charakter: die Andrijiwskyj-Hänge, das Kanalraster von Rusaniwka, der Modernismus von Obolon, das alte Podil. Eine 3D-Karte macht aus dem Lieblingsviertel ein physisches Modell: Gebäude mit echten Höhen, Straßen, Parks, der Dnipro — und das echte Relief der Kiewer Hügel.",
        sections: [
          {
            h2: "Welche Viertel am besten wirken",
            p: [
              "Podil und das Zentrum — dichte historische Quartiere. Petschersk mit der Lawra — Hangrelief plus wiedererkennbare Dominanten. Rusaniwka und Obolon — ein einzigartiges Wassergitter aus Kanälen und Dnipro-Buchten, das in 3D besonders eindrucksvoll aussieht. Ein Bereich von 400–800 m reicht, damit das Viertel auf den ersten Blick erkennbar ist.",
              "Kiew ist hügelig, deshalb gibt aktiviertes Relief dem Modell echte Höhenunterschiede — vom Dnipro-Niveau bis zur Oberstadt.",
            ],
          },
          {
            h2: "Größen und Preise",
            p: [
              "S 5,5 cm — ≈6 €, M 8 cm — ≈8 €, L 11 cm — ≈11 €, XL 15 cm — ≈13 €; Relief +≈1,5 €. Es gibt auch einen Anhänger mit einem Kiewer Viertel (ab ≈3 €) und einen Kühlschrankmagneten (≈4 €).",
            ],
          },
          {
            h2: "So verschenken es Kiewer",
            p: [
              "Die Karte des Hofs, in dem man aufwuchs — für die Eltern. Das Viertel der ersten gemeinsamen Wohnung — zum Jahrestag. Die Lieblingsstrecke am Ufer — als Anhänger mit GPX-Track. Das Modell des Büro-Viertels — fürs Team zur Erinnerung.",
            ],
          },
        ],
        ctaLabel: "3D-Karte von Kiew erstellen",
        ctaHref: "/maps/kyiv",
      },
      pl: {
        title: "Mapa 3D Kijowa: ulubiona dzielnica na Twojej półce",
        description:
          "Podół, Peczersk, Obołoń czy Rusaniwka — mapa 3D dowolnej dzielnicy Kijowa z prawdziwymi budynkami i rzeźbą dnieprzańskich zboczy. Druk od ≈6 €.",
        h1: "Mapa 3D Kijowa: miasto, które można wziąć do ręki",
        intro:
          "Kijów ma charakter w każdej dzielnicy: zbocza Andrijiwskiego, kanałowa siatka Rusaniwki, modernizm Obołonia, stary Podół. Mapa 3D zamienia ulubioną dzielnicę w fizyczny model: budynki o prawdziwych wysokościach, ulice, parki, Dniepr — i prawdziwa rzeźba kijowskich wzgórz.",
        sections: [
          {
            h2: "Które dzielnice wyglądają najlepiej",
            p: [
              "Podół i centrum — gęsta historyczna zabudowa z wyrazistymi kwartałami. Peczersk z Ławrą — rzeźba zboczy plus rozpoznawalne dominanty. Rusaniwka i Obołoń — unikalna wodna siatka kanałów i zatok Dniepru, która w 3D wygląda szczególnie efektownie. Obszar 400–800 m wystarczy, by rozpoznać dzielnicę na pierwszy rzut oka.",
              "Kijów jest pagórkowaty, więc włączona rzeźba daje modelowi prawdziwe różnice wysokości — od poziomu Dniepru po Górne Miasto.",
            ],
          },
          {
            h2: "Rozmiary i ceny",
            p: [
              "S 5,5 cm — ≈6 €, M 8 cm — ≈8 €, L 11 cm — ≈11 €, XL 15 cm — ≈13 €; rzeźba +≈1,5 €. Jest też brelok z dzielnicą Kijowa (od ≈3 €) i magnes na lodówkę (≈4 €).",
            ],
          },
          {
            h2: "Jak kijowianie to darują",
            p: [
              "Mapa podwórka, na którym się dorastało — rodzicom. Dzielnica pierwszego wspólnego mieszkania — na rocznicę. Ulubiona trasa nad brzegiem — brelokiem ze śladem GPX. Model dzielnicy biura — zespołowi na pamiątkę.",
            ],
          },
        ],
        ctaLabel: "Stwórz mapę 3D Kijowa",
        ctaHref: "/maps/kyiv",
      },
      fr: {
        title: "Carte 3D de Kyiv : votre quartier préféré sur l'étagère",
        description:
          "Podil, Petchersk, Obolon ou Rusanivka — une carte 3D de n'importe quel quartier de Kyiv avec de vrais bâtiments et le relief des coteaux du Dnipro. Impression dès ≈6 €.",
        h1: "Carte 3D de Kyiv : une ville qu'on peut tenir en main",
        intro:
          "Kyiv a du caractère dans chaque quartier : les coteaux d'Andriivskyi, la trame de canaux de Rusanivka, le modernisme d'Obolon, le vieux Podil. Une carte 3D transforme le quartier préféré en modèle physique : bâtiments aux hauteurs réelles, rues, parcs, le Dnipro — et le vrai relief des collines de Kyiv.",
        sections: [
          {
            h2: "Quels quartiers rendent le mieux",
            p: [
              "Podil et le centre — un tissu historique dense aux îlots marqués. Petchersk avec la Laure — relief des coteaux et repères reconnaissables. Rusanivka et Obolon — une trame d'eau unique de canaux et de baies du Dnipro, particulièrement spectaculaire en 3D. Une zone de 400–800 m suffit pour reconnaître le quartier au premier coup d'œil.",
              "Kyiv est vallonnée : activer le relief donne au modèle de vrais dénivelés — du niveau du Dnipro jusqu'à la ville haute.",
            ],
          },
          {
            h2: "Tailles et prix",
            p: [
              "S 5,5 cm — ≈6 €, M 8 cm — ≈8 €, L 11 cm — ≈11 €, XL 15 cm — ≈13 € ; relief +≈1,5 €. Il existe aussi un porte-clés avec un quartier de Kyiv (dès ≈3 €) et un magnet de frigo (≈4 €).",
            ],
          },
          {
            h2: "Comment les Kyiviens l'offrent",
            p: [
              "La carte de la cour où l'on a grandi — aux parents. Le quartier du premier appartement commun — pour un anniversaire de couple. Le parcours préféré sur les quais — en porte-clés avec trace GPX. Le modèle du quartier du bureau — à l'équipe, en souvenir.",
            ],
          },
        ],
        ctaLabel: "Créer une carte 3D de Kyiv",
        ctaHref: "/maps/kyiv",
      },
      es: {
        title: "Mapa 3D de Kyiv: tu barrio favorito en la estantería",
        description:
          "Podil, Pechersk, Obolon o Rusanivka — un mapa 3D de cualquier barrio de Kyiv con edificios reales y el relieve de las laderas del Dnipró. Impresión desde ≈6 €.",
        h1: "Mapa 3D de Kyiv: una ciudad que puedes sostener en la mano",
        intro:
          "Kyiv tiene carácter en cada barrio: las laderas de Andriivskyi, la retícula de canales de Rusanivka, el modernismo de Obolon, el viejo Podil. Un mapa 3D convierte el barrio favorito en un modelo físico: edificios con alturas reales, calles, parques, el Dnipró — y el relieve auténtico de las colinas de Kyiv.",
        sections: [
          {
            h2: "Qué barrios lucen mejor",
            p: [
              "Podil y el centro — trama histórica densa con manzanas marcadas. Pechersk con la Lavra — relieve de laderas más hitos reconocibles. Rusanivka y Obolon — una retícula de agua única de canales y bahías del Dnipró que en 3D resulta especialmente vistosa. Una zona de 400–800 m basta para reconocer el barrio a primera vista.",
              "Kyiv es una ciudad con colinas, así que activar el relieve da al modelo desniveles reales — desde el nivel del Dnipró hasta la ciudad alta.",
            ],
          },
          {
            h2: "Tamaños y precios",
            p: [
              "S 5,5 cm — ≈6 €, M 8 cm — ≈8 €, L 11 cm — ≈11 €, XL 15 cm — ≈13 €; relieve +≈1,5 €. También hay llavero con un barrio de Kyiv (desde ≈3 €) e imán de nevera (≈4 €).",
            ],
          },
          {
            h2: "Cómo lo regalan los kyivitas",
            p: [
              "El mapa del patio donde creciste — a los padres. El barrio del primer piso compartido — por un aniversario. La ruta favorita junto al río — como llavero con track GPX. El modelo del barrio de la oficina — al equipo, de recuerdo.",
            ],
          },
        ],
        ctaLabel: "Crear un mapa 3D de Kyiv",
        ctaHref: "/maps/kyiv",
      },
    },
  },
  {
    slug: "shcho-take-3d-druk",
    date: "2026-07-13",
    content: {
      uk: {
        title: "Що таке 3D-друк і як він працює: просте пояснення",
        description:
          "3D-друк простими словами: як принтер шар за шаром вирощує річ із пластику, що таке FDM і чому 3D-мапи міст друкують саме так.",
        h1: "Що таке 3D-друк: просте пояснення без жаргону",
        intro:
          "3D-друк — це спосіб виготовити фізичну річ прямо з цифрової моделі: принтер розплавляє пластикову нитку і викладає її тонкими шарами, один поверх одного, поки з шарів не «виросте» готовий предмет. Саме так друкуються наші 3D-мапи міст, брелоки й магніти.",
        sections: [
          {
            h2: "Як це працює: шар за шаром",
            p: [
              "Найпоширеніша технологія — FDM (моделювання наплавленням). Котушка пластикової нитки подається у гарячу голову принтера (~200 °C), плавиться і видавлюється крізь тонке сопло — зазвичай 0,4 мм. Голова рухається за траєкторією, яку розрахувала програма, і малює один горизонтальний зріз моделі. Потім платформа опускається на частку міліметра — і малюється наступний шар.",
              "Типова висота шару — 0,2 мм: у сантиметрі висоти моделі — п'ятдесят шарів. Тому на боках 3D-друкованих речей видно характерні тонкі смужки — це і є шари, «відбитки пальців» технології.",
            ],
          },
          {
            h2: "Чому це ідеально для мап міст",
            p: [
              "Місто — це і є набір висот: будинки різної поверховості, дороги внизу, пагорби рельєфу. 3D-принтер відтворює ці перепади буквально: кожен будинок отримує свою реальну висоту, річка лягає нижче набережної, а пагорби читаються пальцями.",
              "Кожна модель унікальна — принтеру байдуже, друкувати сотню однакових речей чи сто різних. Саме тому персональна мапа вашого району коштує як серійний сувенір, а не як індивідуальне литво.",
            ],
          },
          {
            h2: "Скільки це триває і скільки коштує",
            p: [
              "Брелок друкується близько години, мапа 8 см — кілька годин, велика панель — до доби. Звідси й терміни виготовлення 1–3 робочі дні. Ціни: брелок від 120 ₴, мапа від 250 ₴ — дешевше за більшість «іменних» подарунків.",
            ],
          },
        ],
        ctaLabel: "Спробувати конструктор мап",
        ctaHref: "/create",
        outro: "Створення 3D-моделі свого району в конструкторі — безкоштовне: платите лише якщо замовляєте друк.",
      },
      en: {
        title: "What is 3D printing and how does it work: a simple explanation",
        description:
          "3D printing in plain words: how a printer grows an object layer by layer from plastic, what FDM is and why 3D city maps are printed this way.",
        h1: "What is 3D printing: a jargon-free explanation",
        intro:
          "3D printing makes a physical object directly from a digital model: the printer melts plastic filament and lays it down in thin layers, one on top of another, until the object 'grows' out of them. That's exactly how our 3D city maps, keychains and magnets are made.",
        sections: [
          {
            h2: "How it works: layer by layer",
            p: [
              "The most common technology is FDM (fused deposition modeling). A spool of plastic filament feeds into a hot print head (~200 °C), melts, and extrudes through a thin nozzle — usually 0.4 mm. The head follows a computed path and draws one horizontal slice of the model; then the platform drops a fraction of a millimeter and the next layer is drawn.",
              "A typical layer is 0.2 mm high — fifty layers per centimeter. That's why 3D-printed objects show fine stripes on their sides: those are the layers, the technology's fingerprints.",
            ],
          },
          {
            h2: "Why it's perfect for city maps",
            p: [
              "A city is a set of heights: buildings of different storeys, roads below, terrain hills. A 3D printer reproduces these literally — every building gets its real height, the river sits below the embankment, hills read under your fingers.",
              "Every model is unique — the printer doesn't care whether it prints a hundred identical objects or a hundred different ones. That's why a personal map of your district costs like a mass souvenir, not like custom casting.",
            ],
          },
          {
            h2: "How long and how much",
            p: [
              "A keychain prints in about an hour, an 8 cm map in a few hours, a large panel up to a day. Hence the 1–3 business day lead time. Prices: keychain from ≈€3, map from ≈€6 — cheaper than most personalized gifts.",
            ],
          },
        ],
        ctaLabel: "Try the map builder",
        ctaHref: "/create",
        outro: "Building a 3D model of your district is free — you only pay if you order a print.",
      },
      de: {
        title: "Was ist 3D-Druck und wie funktioniert er: eine einfache Erklärung",
        description:
          "3D-Druck in einfachen Worten: wie ein Drucker Schicht für Schicht ein Objekt aus Kunststoff wachsen lässt, was FDM ist und warum 3D-Stadtkarten genau so gedruckt werden.",
        h1: "Was ist 3D-Druck: eine Erklärung ohne Fachjargon",
        intro:
          "3D-Druck ist eine Art, ein physisches Objekt direkt aus einem digitalen Modell herzustellen: Der Drucker schmilzt einen Kunststofffaden und legt ihn in dünnen Schichten übereinander, bis daraus das fertige Objekt «wächst». Genau so entstehen unsere 3D-Stadtkarten, Anhänger und Magnete.",
        sections: [
          {
            h2: "Wie es funktioniert: Schicht für Schicht",
            p: [
              "Die verbreitetste Technologie ist FDM (Schmelzschichtung). Eine Spule Kunststofffaden läuft in den heißen Druckkopf (~200 °C), schmilzt und wird durch eine dünne Düse gepresst — meist 0,4 mm. Der Kopf folgt der berechneten Bahn und zeichnet einen horizontalen Schnitt des Modells; dann senkt sich die Plattform um Bruchteile eines Millimeters und die nächste Schicht entsteht.",
              "Eine typische Schicht ist 0,2 mm hoch: fünfzig Schichten pro Zentimeter. Deshalb sieht man an den Seiten 3D-gedruckter Dinge feine Streifen — das sind die Schichten, die «Fingerabdrücke» der Technologie.",
            ],
          },
          {
            h2: "Warum das ideal für Stadtkarten ist",
            p: [
              "Eine Stadt ist genau das: eine Menge Höhen — Gebäude verschiedener Geschosszahl, Straßen darunter, Geländehügel. Der 3D-Drucker gibt diese Unterschiede buchstäblich wieder: jedes Gebäude bekommt seine echte Höhe, der Fluss liegt unter der Uferpromenade, Hügel werden mit den Fingern lesbar.",
              "Jedes Modell ist einzigartig — dem Drucker ist es egal, ob er hundert gleiche oder hundert verschiedene Dinge druckt. Deshalb kostet eine persönliche Karte deines Viertels wie ein Serien-Souvenir und nicht wie ein Einzelguss.",
            ],
          },
          {
            h2: "Wie lange es dauert und was es kostet",
            p: [
              "Ein Anhänger druckt etwa eine Stunde, eine 8-cm-Karte einige Stunden, ein großes Panel bis zu einem Tag. Daher die Fertigungszeit von 1–3 Werktagen. Preise: Anhänger ab ≈3 €, Karte ab ≈6 € — günstiger als die meisten personalisierten Geschenke.",
            ],
          },
        ],
        ctaLabel: "Den Karten-Konfigurator testen",
        ctaHref: "/create",
        outro: "Das Erstellen eines 3D-Modells deines Viertels im Konfigurator ist kostenlos — du zahlst nur, wenn du den Druck bestellst.",
      },
      pl: {
        title: "Czym jest druk 3D i jak działa: proste wyjaśnienie",
        description:
          "Druk 3D prostymi słowami: jak drukarka warstwa po warstwie wytwarza przedmiot z plastiku, czym jest FDM i dlaczego mapy miast 3D drukuje się właśnie tak.",
        h1: "Czym jest druk 3D: proste wyjaśnienie bez żargonu",
        intro:
          "Druk 3D to sposób wytworzenia fizycznej rzeczy wprost z modelu cyfrowego: drukarka topi plastikową żyłkę i układa ją cienkimi warstwami, jedna na drugiej, aż z warstw «wyrośnie» gotowy przedmiot. Właśnie tak drukowane są nasze mapy miast 3D, breloki i magnesy.",
        sections: [
          {
            h2: "Jak to działa: warstwa po warstwie",
            p: [
              "Najpowszechniejsza technologia to FDM (osadzanie topionego materiału). Szpula plastikowej żyłki wchodzi do gorącej głowicy drukarki (~200 °C), topi się i jest wytłaczana przez cienką dyszę — zwykle 0,4 mm. Głowica porusza się po torze wyliczonym przez program i rysuje jeden poziomy przekrój modelu. Potem platforma opada o ułamek milimetra — i rysowana jest kolejna warstwa.",
              "Typowa wysokość warstwy to 0,2 mm: w centymetrze wysokości modelu jest pięćdziesiąt warstw. Dlatego na bokach drukowanych rzeczy widać charakterystyczne cienkie prążki — to właśnie warstwy, «odciski palców» technologii.",
            ],
          },
          {
            h2: "Dlaczego to idealne dla map miast",
            p: [
              "Miasto to właśnie zbiór wysokości: budynki o różnej liczbie pięter, drogi na dole, wzgórza terenu. Drukarka 3D odtwarza te różnice dosłownie: każdy budynek dostaje swoją prawdziwą wysokość, rzeka układa się poniżej bulwaru, a wzgórza czyta się palcami.",
              "Każdy model jest unikalny — drukarce jest obojętne, czy drukuje sto takich samych rzeczy, czy sto różnych. Właśnie dlatego spersonalizowana mapa Twojej dzielnicy kosztuje jak seryjna pamiątka, a nie jak indywidualny odlew.",
            ],
          },
          {
            h2: "Ile to trwa i ile kosztuje",
            p: [
              "Brelok drukuje się około godziny, mapa 8 cm — kilka godzin, duży panel — nawet dobę. Stąd terminy realizacji 1–3 dni robocze. Ceny: brelok od ≈3 €, mapa od ≈6 € — taniej niż większość «imiennych» prezentów.",
            ],
          },
        ],
        ctaLabel: "Wypróbuj kreator map",
        ctaHref: "/create",
        outro: "Stworzenie modelu 3D swojej dzielnicy w kreatorze jest darmowe: płacisz tylko, jeśli zamawiasz druk.",
      },
      fr: {
        title: "Qu'est-ce que l'impression 3D et comment ça marche : une explication simple",
        description:
          "L'impression 3D en mots simples : comment une imprimante fait croître un objet couche par couche à partir de plastique, ce qu'est le FDM et pourquoi les cartes de ville 3D s'impriment ainsi.",
        h1: "Qu'est-ce que l'impression 3D : une explication sans jargon",
        intro:
          "L'impression 3D fabrique un objet physique directement à partir d'un modèle numérique : l'imprimante fait fondre un fil plastique et le dépose en couches fines, l'une sur l'autre, jusqu'à ce que l'objet «pousse». C'est exactement ainsi que naissent nos cartes de ville 3D, porte-clés et magnets.",
        sections: [
          {
            h2: "Comment ça marche : couche par couche",
            p: [
              "La technologie la plus répandue est le FDM (dépôt de fil fondu). Une bobine de fil plastique entre dans la tête chaude de l'imprimante (~200 °C), fond et est extrudée par une buse fine — généralement 0,4 mm. La tête suit la trajectoire calculée et dessine une tranche horizontale du modèle ; puis le plateau descend d'une fraction de millimètre et la couche suivante est tracée.",
              "Une couche typique fait 0,2 mm de haut : cinquante couches par centimètre. C'est pourquoi les objets imprimés en 3D montrent de fines stries sur les côtés — ce sont les couches, les «empreintes digitales» de la technologie.",
            ],
          },
          {
            h2: "Pourquoi c'est parfait pour les cartes de ville",
            p: [
              "Une ville, c'est précisément un ensemble de hauteurs : des bâtiments de différents étages, des routes en dessous, des collines de relief. L'imprimante 3D reproduit ces écarts littéralement : chaque bâtiment reçoit sa hauteur réelle, la rivière se place sous les quais, les collines se lisent du bout des doigts.",
              "Chaque modèle est unique — peu importe à l'imprimante d'imprimer cent objets identiques ou cent différents. C'est pourquoi une carte personnelle de votre quartier coûte comme un souvenir de série, et non comme un moulage sur mesure.",
            ],
          },
          {
            h2: "Combien de temps et combien ça coûte",
            p: [
              "Un porte-clés s'imprime en une heure environ, une carte de 8 cm en quelques heures, un grand panneau jusqu'à une journée. D'où le délai de fabrication de 1–3 jours ouvrés. Prix : porte-clés dès ≈3 €, carte dès ≈6 € — moins cher que la plupart des cadeaux personnalisés.",
            ],
          },
        ],
        ctaLabel: "Essayer le configurateur de cartes",
        ctaHref: "/create",
        outro: "Créer le modèle 3D de votre quartier dans le configurateur est gratuit : vous ne payez que si vous commandez l'impression.",
      },
      es: {
        title: "Qué es la impresión 3D y cómo funciona: una explicación sencilla",
        description:
          "La impresión 3D en palabras simples: cómo una impresora hace crecer un objeto capa a capa a partir de plástico, qué es FDM y por qué los mapas de ciudad 3D se imprimen así.",
        h1: "Qué es la impresión 3D: una explicación sin jerga",
        intro:
          "La impresión 3D fabrica un objeto físico directamente a partir de un modelo digital: la impresora funde un filamento de plástico y lo deposita en capas finas, una sobre otra, hasta que el objeto «crece». Así es exactamente como se hacen nuestros mapas de ciudad 3D, llaveros e imanes.",
        sections: [
          {
            h2: "Cómo funciona: capa a capa",
            p: [
              "La tecnología más extendida es FDM (modelado por deposición fundida). Una bobina de filamento plástico entra en el cabezal caliente de la impresora (~200 °C), se funde y se extruye por una boquilla fina — normalmente de 0,4 mm. El cabezal sigue la trayectoria calculada y dibuja un corte horizontal del modelo; luego la plataforma baja una fracción de milímetro y se traza la siguiente capa.",
              "Una capa típica mide 0,2 mm de alto: cincuenta capas por centímetro. Por eso en los laterales de las piezas impresas en 3D se ven finas franjas — son las capas, las «huellas dactilares» de la tecnología.",
            ],
          },
          {
            h2: "Por qué es perfecto para mapas de ciudad",
            p: [
              "Una ciudad es justamente un conjunto de alturas: edificios de distintas plantas, calles abajo, colinas del terreno. La impresora 3D reproduce esos desniveles literalmente: cada edificio recibe su altura real, el río queda por debajo del paseo, las colinas se leen con los dedos.",
              "Cada modelo es único — a la impresora le da igual imprimir cien piezas iguales o cien distintas. Por eso un mapa personal de tu barrio cuesta como un recuerdo de serie y no como una fundición a medida.",
            ],
          },
          {
            h2: "Cuánto tarda y cuánto cuesta",
            p: [
              "Un llavero se imprime en cerca de una hora, un mapa de 8 cm en unas horas, un panel grande hasta un día. De ahí el plazo de 1–3 días hábiles. Precios: llavero desde ≈3 €, mapa desde ≈6 € — más barato que la mayoría de regalos personalizados.",
            ],
          },
        ],
        ctaLabel: "Probar el configurador de mapas",
        ctaHref: "/create",
        outro: "Crear el modelo 3D de tu barrio en el configurador es gratis: solo pagas si pides la impresión.",
      },
    },
  },
  {
    slug: "eco-pla-shcho-tse",
    date: "2026-07-13",
    content: {
      uk: {
        title: "Eco PLA: що це за пластик і чому ним друкують мапи",
        description:
          "PLA — біопластик з кукурудзяного крохмалю: безпечний, без запаху, тримає деталь 0,4 мм. Чому ми друкуємо 3D-мапи саме з Eco PLA і як він поводиться вдома.",
        h1: "Eco PLA: матеріал, з якого надруковано вашу мапу",
        intro:
          "Усі наші вироби — мапи, брелоки, магніти — друкуються з PLA (полілактид): біопластику, який виробляють з рослинної сировини на кшталт кукурудзяного крохмалю. Це найпопулярніший матеріал якісного побутового 3D-друку, і ось чому ми обрали саме його.",
        sections: [
          {
            h2: "Безпечний і приємний",
            p: [
              "PLA не має запаху, не виділяє шкідливих речовин за кімнатної температури і безпечний для дому з дітьми й тваринами. На дотик — гладкий, теплий, «не дешевий»: смужки шарів дають характерну приємну фактуру.",
              "На відміну від багатьох нафтових пластиків, PLA — біорозкладний у промислових умовах компостування. Для довкілля це один з найм'якших пластиків узагалі.",
            ],
          },
          {
            h2: "Точність, яка тримає вулиці",
            p: [
              "Мапа міста вимагає дрібної деталі: лінія провулка — це доріжка пластику завширшки менше міліметра. PLA має мінімальну усадку під час охолодження, тому геометрія не «пливе»: вулиці лишаються рівними, стіни будинків — вертикальними, а пази серійних панелей сходяться між собою.",
            ],
          },
          {
            h2: "Як він живе вдома: що можна і чого уникати",
            p: [
              "На полиці, стіні чи холодильнику PLA живе роками без змін. Єдине справжнє обмеження — тепло: за +60 °C матеріал починає м'якшати, тож не лишайте виріб на торпеді авто влітку і не ставте біля духовки чи батареї впритул.",
              "Пил знімається сухим пензлем або вологою серветкою. Брелок спокійно витримує щоденне носіння з ключами — подряпини на матовому пластику майже непомітні.",
            ],
          },
        ],
        ctaLabel: "Замовити виріб з Eco PLA",
        ctaHref: "/create",
      },
      en: {
        title: "Eco PLA: what this plastic is and why we print maps with it",
        description:
          "PLA is a bioplastic made from corn starch: safe, odorless, holds 0.4 mm detail. Why our 3D maps are printed in Eco PLA and how it behaves at home.",
        h1: "Eco PLA: the material your map is printed in",
        intro:
          "All our items — maps, keychains, magnets — are printed in PLA (polylactide): a bioplastic produced from plant feedstock such as corn starch. It's the most popular material in quality desktop 3D printing, and here's why we chose it.",
        sections: [
          {
            h2: "Safe and pleasant",
            p: [
              "PLA is odorless, emits nothing harmful at room temperature and is safe for a home with kids and pets. To the touch it's smooth and warm; the layer lines give a distinctive pleasant texture.",
              "Unlike many oil-based plastics, PLA is biodegradable under industrial composting. Environmentally it's one of the mildest plastics there is.",
            ],
          },
          {
            h2: "Precision that holds the streets",
            p: [
              "A city map demands fine detail: a small lane is a plastic path under a millimeter wide. PLA has minimal shrinkage while cooling, so the geometry doesn't drift: streets stay straight, building walls vertical, and the slots of multi-tile panels fit together.",
            ],
          },
          {
            h2: "Life at home: what's fine and what to avoid",
            p: [
              "On a shelf, wall or fridge PLA lives for years unchanged. The one real limit is heat: above +60 °C the material starts to soften, so don't leave the item on a car dashboard in summer or right next to an oven or radiator.",
              "Dust comes off with a dry brush or a damp cloth. A keychain easily survives daily life with keys — scratches on matte plastic are barely visible.",
            ],
          },
        ],
        ctaLabel: "Order an Eco PLA piece",
        ctaHref: "/create",
      },
      de: {
        title: "Eco PLA: was das für ein Kunststoff ist und warum wir damit Karten drucken",
        description:
          "PLA ist ein Biokunststoff aus Maisstärke: sicher, geruchlos, hält 0,4-mm-Details. Warum wir 3D-Karten aus Eco PLA drucken und wie er sich zu Hause verhält.",
        h1: "Eco PLA: das Material, aus dem deine Karte gedruckt ist",
        intro:
          "Alle unsere Stücke — Karten, Anhänger, Magnete — werden aus PLA (Polylactid) gedruckt: einem Biokunststoff, der aus pflanzlichen Rohstoffen wie Maisstärke hergestellt wird. Es ist das beliebteste Material im hochwertigen Desktop-3D-Druck, und darum haben wir es gewählt.",
        sections: [
          {
            h2: "Sicher und angenehm",
            p: [
              "PLA ist geruchlos, gibt bei Raumtemperatur nichts Schädliches ab und ist sicher für Haushalte mit Kindern und Tieren. Es fühlt sich glatt und warm an, «nicht billig»: die Schichtlinien geben eine charakteristische, angenehme Textur.",
              "Anders als viele erdölbasierte Kunststoffe ist PLA unter industriellen Kompostierbedingungen biologisch abbaubar. Ökologisch ist es einer der sanftesten Kunststoffe überhaupt.",
            ],
          },
          {
            h2: "Präzision, die die Straßen hält",
            p: [
              "Eine Stadtkarte verlangt feine Details: die Linie einer Gasse ist eine Kunststoffbahn unter einem Millimeter Breite. PLA hat minimalen Schwund beim Abkühlen, deshalb «verläuft» die Geometrie nicht: Straßen bleiben gerade, Gebäudewände senkrecht, und die Steckverbindungen von Serienpanels passen zueinander.",
            ],
          },
          {
            h2: "Wie es zu Hause lebt: was geht und was zu vermeiden ist",
            p: [
              "Auf dem Regal, an der Wand oder am Kühlschrank hält PLA jahrelang unverändert. Die einzige echte Grenze ist Wärme: ab +60 °C beginnt das Material weich zu werden — lass das Stück also im Sommer nicht auf dem Armaturenbrett liegen und stelle es nicht direkt neben Ofen oder Heizkörper.",
              "Staub geht mit einem trockenen Pinsel oder einem feuchten Tuch ab. Ein Anhänger übersteht den täglichen Gebrauch am Schlüsselbund problemlos — Kratzer sind auf mattem Kunststoff kaum sichtbar.",
            ],
          },
        ],
        ctaLabel: "Ein Stück aus Eco PLA bestellen",
        ctaHref: "/create",
      },
      pl: {
        title: "Eco PLA: co to za plastik i dlaczego drukujemy nim mapy",
        description:
          "PLA to bioplastik ze skrobi kukurydzianej: bezpieczny, bezwonny, utrzymuje detal 0,4 mm. Dlaczego drukujemy mapy 3D właśnie z Eco PLA i jak zachowuje się w domu.",
        h1: "Eco PLA: materiał, z którego wydrukowano Twoją mapę",
        intro:
          "Wszystkie nasze wyroby — mapy, breloki, magnesy — drukowane są z PLA (polilaktydu): bioplastiku wytwarzanego z surowców roślinnych, takich jak skrobia kukurydziana. To najpopularniejszy materiał w dobrym druku 3D, i oto dlaczego wybraliśmy właśnie jego.",
        sections: [
          {
            h2: "Bezpieczny i przyjemny",
            p: [
              "PLA nie ma zapachu, nie wydziela szkodliwych substancji w temperaturze pokojowej i jest bezpieczny dla domu z dziećmi i zwierzętami. W dotyku jest gładki, ciepły, «nie tani»: prążki warstw dają charakterystyczną, przyjemną fakturę.",
              "W odróżnieniu od wielu plastików ropopochodnych PLA jest biodegradowalny w warunkach kompostowania przemysłowego. Dla środowiska to jeden z najłagodniejszych plastików w ogóle.",
            ],
          },
          {
            h2: "Precyzja, która utrzymuje ulice",
            p: [
              "Mapa miasta wymaga drobnego detalu: linia uliczki to ścieżka plastiku o szerokości poniżej milimetra. PLA ma minimalny skurcz podczas stygnięcia, więc geometria nie «pływa»: ulice pozostają proste, ściany budynków pionowe, a zaczepy paneli seryjnych pasują do siebie.",
            ],
          },
          {
            h2: "Jak żyje w domu: co można, a czego unikać",
            p: [
              "Na półce, ścianie czy lodówce PLA żyje latami bez zmian. Jedyne prawdziwe ograniczenie to ciepło: powyżej +60 °C materiał zaczyna mięknąć, więc nie zostawiaj wyrobu latem na desce rozdzielczej auta i nie stawiaj tuż przy piekarniku czy kaloryferze.",
              "Kurz usuwa się suchym pędzelkiem lub wilgotną ściereczką. Brelok spokojnie wytrzymuje codzienne noszenie z kluczami — zarysowania na matowym plastiku są prawie niewidoczne.",
            ],
          },
        ],
        ctaLabel: "Zamów wyrób z Eco PLA",
        ctaHref: "/create",
      },
      fr: {
        title: "Eco PLA : quel est ce plastique et pourquoi nous imprimons les cartes avec",
        description:
          "Le PLA est un bioplastique issu de l'amidon de maïs : sûr, sans odeur, tient le détail de 0,4 mm. Pourquoi nos cartes 3D sont imprimées en Eco PLA et comment il se comporte à la maison.",
        h1: "Eco PLA : la matière dont votre carte est imprimée",
        intro:
          "Toutes nos pièces — cartes, porte-clés, magnets — sont imprimées en PLA (polylactide) : un bioplastique produit à partir de matières végétales comme l'amidon de maïs. C'est le matériau le plus répandu en impression 3D de qualité, et voici pourquoi nous l'avons choisi.",
        sections: [
          {
            h2: "Sûr et agréable",
            p: [
              "Le PLA est sans odeur, n'émet rien de nocif à température ambiante et convient à un foyer avec enfants et animaux. Au toucher il est lisse, chaud, «pas cheap» : les stries de couches donnent une texture agréable caractéristique.",
              "Contrairement à beaucoup de plastiques pétroliers, le PLA est biodégradable en compostage industriel. Pour l'environnement, c'est l'un des plastiques les plus doux qui soient.",
            ],
          },
          {
            h2: "Une précision qui tient les rues",
            p: [
              "Une carte de ville exige du détail fin : la ligne d'une ruelle est un cordon de plastique de moins d'un millimètre de large. Le PLA a un retrait minimal au refroidissement, donc la géométrie ne «flotte» pas : les rues restent droites, les murs verticaux, et les emboîtements des panneaux en série s'ajustent entre eux.",
            ],
          },
          {
            h2: "Sa vie à la maison : ce qui va et ce qu'il faut éviter",
            p: [
              "Sur une étagère, un mur ou un frigo, le PLA vit des années sans changer. La seule vraie limite est la chaleur : au-delà de +60 °C le matériau commence à ramollir — ne laissez donc pas la pièce sur le tableau de bord en été, ni collée à un four ou un radiateur.",
              "La poussière s'enlève au pinceau sec ou avec un chiffon humide. Un porte-clés supporte sans souci l'usage quotidien avec les clés — les rayures sont à peine visibles sur du plastique mat.",
            ],
          },
        ],
        ctaLabel: "Commander une pièce en Eco PLA",
        ctaHref: "/create",
      },
      es: {
        title: "Eco PLA: qué plástico es y por qué imprimimos mapas con él",
        description:
          "El PLA es un bioplástico de almidón de maíz: seguro, sin olor, aguanta detalles de 0,4 mm. Por qué imprimimos los mapas 3D en Eco PLA y cómo se comporta en casa.",
        h1: "Eco PLA: el material con el que está impreso tu mapa",
        intro:
          "Todas nuestras piezas — mapas, llaveros, imanes — se imprimen en PLA (poliláctido): un bioplástico producido a partir de materias vegetales como el almidón de maíz. Es el material más popular en la impresión 3D de calidad, y por eso lo elegimos.",
        sections: [
          {
            h2: "Seguro y agradable",
            p: [
              "El PLA no tiene olor, no emite sustancias nocivas a temperatura ambiente y es seguro para una casa con niños y mascotas. Al tacto es suave y cálido, «no barato»: las franjas de las capas dan una textura agradable característica.",
              "A diferencia de muchos plásticos derivados del petróleo, el PLA es biodegradable en compostaje industrial. Para el medio ambiente es uno de los plásticos más benignos que existen.",
            ],
          },
          {
            h2: "Precisión que sostiene las calles",
            p: [
              "Un mapa de ciudad exige detalle fino: la línea de una callejuela es un cordón de plástico de menos de un milímetro de ancho. El PLA tiene una contracción mínima al enfriarse, así que la geometría no «se mueve»: las calles quedan rectas, las paredes verticales y las uniones de los paneles en serie encajan entre sí.",
            ],
          },
          {
            h2: "Su vida en casa: qué se puede y qué evitar",
            p: [
              "En una estantería, pared o nevera el PLA vive años sin cambios. El único límite real es el calor: por encima de +60 °C el material empieza a ablandarse, así que no dejes la pieza en el salpicadero del coche en verano ni pegada al horno o al radiador.",
              "El polvo se quita con un pincel seco o un paño húmedo. Un llavero aguanta sin problema el uso diario con las llaves — los arañazos apenas se ven en el plástico mate.",
            ],
          },
        ],
        ctaLabel: "Pedir una pieza en Eco PLA",
        ctaHref: "/create",
      },
    },
  },
  {
    slug: "yak-obraty-rozmir-3d-mapy",
    date: "2026-07-13",
    content: {
      uk: {
        title: "S, M, L чи XL: як обрати розмір 3D-мапи міста",
        description:
          "Порівняння розмірів 3D-мапи: S 5,5 см (250 ₴), M 8 см (350 ₴), L 11 см (450 ₴), XL 15 см (550 ₴). Який розмір під яку ділянку, полицю і бюджет.",
        h1: "Як обрати розмір 3D-мапи: чесне порівняння",
        intro:
          "Розмір — головне рішення при замовленні мапи: він визначає і ціну, і те, наскільки детально читатиметься район. Коротка версія: M (8 см) — найуніверсальніший; далі — нюанси, які варто знати до замовлення.",
        sections: [
          {
            h2: "Чотири розміри на прикладах",
            p: [
              "S (5,5 см, 250 ₴) — компактний сувенір: добре для щільного центру з виразними кварталами, стоїть на робочому столі чи поличці з дрібницями. Дрібні провулки на S уже зливаються, тож обирайте невелику ділянку 300–500 м.",
              "M (8 см, 350 ₴) — золота середина: ділянка 400–800 м читається повністю, модель помітна на полиці, але не претендує на пів кімнати. Найчастіший вибір для подарунка.",
              "L (11 см, 450 ₴) і XL (15 см, 550 ₴) — інтер'єрні речі: видно і двори, і окремі будинки, можна брати ширшу ділянку до кілометра-півтора. XL особливо виграє з рельєфом — перепади висот на великій площі виглядають драматично.",
            ],
          },
          {
            h2: "Правило ділянки: менше — детальніше",
            p: [
              "Фізичний розмір ділиться на розмір ділянки: що більший шматок міста ви охоплюєте, то дрібнішим стає кожен будинок. Ділянка 500 м на моделі 8 см дає масштаб, за якого видно кожен дім; ділянка 2 км на тій самій моделі перетворює квартали на текстуру. Хочете «весь центр» — беріть L/XL; хочете «свій двір» — вистачить S/M.",
            ],
          },
          {
            h2: "Рельєф і серії",
            p: [
              "Рельєф місцевості (+60 ₴) додає мапі справжні перепади висот — обов'язково для Києва, Львова чи Карпат, необов'язково для рівнинних міст. А якщо хочеться охопити велику територію без втрати деталей — замовте серію з кількох панелей, що з'єднуються пазами в одне полотно.",
            ],
          },
        ],
        ctaLabel: "Обрати ділянку і розмір",
        ctaHref: "/create",
      },
      en: {
        title: "S, M, L or XL: choosing your 3D city map size",
        description:
          "3D map size comparison: S 5.5 cm, M 8 cm, L 11 cm, XL 15 cm. Which size fits which area, shelf and budget.",
        h1: "How to choose a 3D map size: an honest comparison",
        intro:
          "Size is the main decision when ordering a map: it sets both the price and how much district detail survives. Short version: M (8 cm) is the most universal; below are the nuances worth knowing before you order.",
        sections: [
          {
            h2: "Four sizes with examples",
            p: [
              "S (5.5 cm) — a compact souvenir: great for a dense center with distinct blocks; small lanes start merging, so pick a small 300–500 m area.",
              "M (8 cm) — the sweet spot: a 400–800 m area reads fully, the model is noticeable on a shelf without claiming half the room. The most common gift choice.",
              "L (11 cm) and XL (15 cm) — interior pieces: you see courtyards and individual buildings, and can take a wider area up to 1–1.5 km. XL especially shines with terrain relief — elevation across a large area looks dramatic.",
            ],
          },
          {
            h2: "The area rule: smaller means more detailed",
            p: [
              "Physical size divides by the captured area: the bigger the piece of city, the smaller each building becomes. A 500 m area on an 8 cm model shows every house; a 2 km area on the same model turns blocks into texture. Want 'the whole center' — go L/XL; want 'my backyard' — S/M is enough.",
            ],
          },
          {
            h2: "Relief and series",
            p: [
              "Terrain relief adds real elevation — a must for Kyiv, Lviv or the Carpathians, optional for flat cities. And to cover a large territory without losing detail, order a series of tiles that connect into one panel.",
            ],
          },
        ],
        ctaLabel: "Pick an area and size",
        ctaHref: "/create",
      },
      de: {
        title: "S, M, L oder XL: die richtige Größe der 3D-Stadtkarte wählen",
        description:
          "Größenvergleich der 3D-Karte: S 5,5 cm (≈6 €), M 8 cm (≈8 €), L 11 cm (≈11 €), XL 15 cm (≈13 €). Welche Größe zu welchem Bereich, Regal und Budget passt.",
        h1: "Wie man die Größe der 3D-Karte wählt: ein ehrlicher Vergleich",
        intro:
          "Die Größe ist die wichtigste Entscheidung beim Bestellen: Sie bestimmt den Preis und wie detailliert das Viertel lesbar bleibt. Kurzfassung: M (8 cm) ist am universellsten; darunter die Nuancen, die man vor der Bestellung kennen sollte.",
        sections: [
          {
            h2: "Vier Größen an Beispielen",
            p: [
              "S (5,5 cm, ≈6 €) — ein kompaktes Souvenir: gut für ein dichtes Zentrum mit markanten Quartieren, passt auf den Schreibtisch. Kleine Gassen verschmelzen bei S bereits, also wähle einen kleinen Bereich von 300–500 m.",
              "M (8 cm, ≈8 €) — die goldene Mitte: ein Bereich von 400–800 m ist vollständig lesbar, das Modell fällt im Regal auf, ohne das halbe Zimmer zu beanspruchen. Die häufigste Wahl als Geschenk.",
              "L (11 cm, ≈11 €) und XL (15 cm, ≈13 €) — Interieurstücke: Höfe und einzelne Gebäude sind sichtbar, man kann einen breiteren Bereich bis 1–1,5 km nehmen. XL gewinnt besonders mit Relief — Höhenunterschiede auf großer Fläche wirken dramatisch.",
            ],
          },
          {
            h2: "Die Bereichsregel: kleiner heißt detaillierter",
            p: [
              "Die physische Größe teilt sich durch den erfassten Bereich: je größer das Stück Stadt, desto kleiner jedes Gebäude. Ein 500-m-Bereich auf einem 8-cm-Modell zeigt jedes Haus; ein 2-km-Bereich auf demselben Modell macht aus Quartieren Textur. Du willst «das ganze Zentrum» — nimm L/XL; du willst «meinen Hof» — S/M reicht.",
            ],
          },
          {
            h2: "Relief und Serien",
            p: [
              "Das Geländerelief (+≈1,5 €) gibt der Karte echte Höhenunterschiede — Pflicht für Kyiv, Lwiw oder die Karpaten, optional für flache Städte. Und wenn du ein großes Gebiet ohne Detailverlust abdecken willst — bestelle eine Serie aus mehreren Kacheln, die sich zu einem Panel verbinden.",
            ],
          },
        ],
        ctaLabel: "Bereich und Größe wählen",
        ctaHref: "/create",
      },
      pl: {
        title: "S, M, L czy XL: jak wybrać rozmiar mapy miasta 3D",
        description:
          "Porównanie rozmiarów mapy 3D: S 5,5 cm (≈6 €), M 8 cm (≈8 €), L 11 cm (≈11 €), XL 15 cm (≈13 €). Jaki rozmiar do jakiego obszaru, półki i budżetu.",
        h1: "Jak wybrać rozmiar mapy 3D: uczciwe porównanie",
        intro:
          "Rozmiar to główna decyzja przy zamawianiu mapy: określa i cenę, i to, jak szczegółowo będzie czytelna dzielnica. Krótko: M (8 cm) jest najbardziej uniwersalny; poniżej niuanse, które warto znać przed zamówieniem.",
        sections: [
          {
            h2: "Cztery rozmiary na przykładach",
            p: [
              "S (5,5 cm, ≈6 €) — kompaktowa pamiątka: dobra dla gęstego centrum z wyrazistymi kwartałami, stoi na biurku. Małe uliczki przy S już się zlewają, więc wybierz niewielki obszar 300–500 m.",
              "M (8 cm, ≈8 €) — złoty środek: obszar 400–800 m czyta się w całości, model jest widoczny na półce, ale nie zajmuje pół pokoju. Najczęstszy wybór na prezent.",
              "L (11 cm, ≈11 €) i XL (15 cm, ≈13 €) — elementy wnętrza: widać podwórka i pojedyncze budynki, można wziąć szerszy obszar do 1–1,5 km. XL szczególnie zyskuje z rzeźbą terenu — różnice wysokości na dużej powierzchni wyglądają dramatycznie.",
            ],
          },
          {
            h2: "Zasada obszaru: mniejszy znaczy bardziej szczegółowy",
            p: [
              "Rozmiar fizyczny dzieli się przez wielkość obszaru: im większy kawałek miasta, tym mniejszy każdy budynek. Obszar 500 m na modelu 8 cm pokazuje każdy dom; obszar 2 km na tym samym modelu zamienia kwartały w teksturę. Chcesz «całe centrum» — bierz L/XL; chcesz «swoje podwórko» — wystarczy S/M.",
            ],
          },
          {
            h2: "Rzeźba terenu i serie",
            p: [
              "Rzeźba terenu (+≈1,5 €) dodaje mapie prawdziwe różnice wysokości — obowiązkowa dla Kijowa, Lwowa czy Karpat, opcjonalna dla płaskich miast. A jeśli chcesz objąć duży teren bez utraty detali — zamów serię kilku kafelków łączących się w jeden panel.",
            ],
          },
        ],
        ctaLabel: "Wybierz obszar i rozmiar",
        ctaHref: "/create",
      },
      fr: {
        title: "S, M, L ou XL : comment choisir la taille de votre carte de ville 3D",
        description:
          "Comparatif des tailles de carte 3D : S 5,5 cm (≈6 €), M 8 cm (≈8 €), L 11 cm (≈11 €), XL 15 cm (≈13 €). Quelle taille pour quelle zone, quelle étagère et quel budget.",
        h1: "Comment choisir la taille d'une carte 3D : un comparatif honnête",
        intro:
          "La taille est la décision principale : elle fixe le prix et le niveau de détail qui survit. Version courte : M (8 cm) est la plus universelle ; ci-dessous les nuances à connaître avant de commander.",
        sections: [
          {
            h2: "Quatre tailles en exemples",
            p: [
              "S (5,5 cm, ≈6 €) — un souvenir compact : parfait pour un centre dense aux îlots marqués, tient sur un bureau. Les petites ruelles fusionnent déjà en S : choisissez une petite zone de 300–500 m.",
              "M (8 cm, ≈8 €) — le juste milieu : une zone de 400–800 m se lit entièrement, le modèle se remarque sur une étagère sans envahir la pièce. Le choix le plus fréquent pour un cadeau.",
              "L (11 cm, ≈11 €) et XL (15 cm, ≈13 €) — des pièces d'intérieur : on voit les cours et les bâtiments individuels, et on peut prendre une zone plus large jusqu'à 1–1,5 km. XL brille surtout avec le relief — les dénivelés sur une grande surface sont spectaculaires.",
            ],
          },
          {
            h2: "La règle de la zone : plus petit = plus détaillé",
            p: [
              "La taille physique se divise par la zone capturée : plus le morceau de ville est grand, plus chaque bâtiment rétrécit. Une zone de 500 m sur un modèle de 8 cm montre chaque maison ; une zone de 2 km sur le même modèle transforme les îlots en texture. Vous voulez «tout le centre» — prenez L/XL ; vous voulez «ma cour» — S/M suffit.",
            ],
          },
          {
            h2: "Relief et séries",
            p: [
              "Le relief du terrain (+≈1,5 €) donne à la carte de vrais dénivelés — indispensable pour Kyiv, Lviv ou les Carpates, facultatif pour les villes plates. Et pour couvrir un grand territoire sans perdre le détail — commandez une série de tuiles qui s'assemblent en un panneau.",
            ],
          },
        ],
        ctaLabel: "Choisir une zone et une taille",
        ctaHref: "/create",
      },
      es: {
        title: "S, M, L o XL: cómo elegir el tamaño de tu mapa de ciudad 3D",
        description:
          "Comparativa de tamaños del mapa 3D: S 5,5 cm (≈6 €), M 8 cm (≈8 €), L 11 cm (≈11 €), XL 15 cm (≈13 €). Qué tamaño para qué zona, estantería y presupuesto.",
        h1: "Cómo elegir el tamaño de un mapa 3D: una comparativa honesta",
        intro:
          "El tamaño es la decisión principal al pedir un mapa: determina el precio y cuánto detalle del barrio sobrevive. Versión corta: M (8 cm) es el más universal; abajo, los matices que conviene saber antes de pedir.",
        sections: [
          {
            h2: "Cuatro tamaños con ejemplos",
            p: [
              "S (5,5 cm, ≈6 €) — un recuerdo compacto: va bien para un centro denso con manzanas marcadas, cabe en el escritorio. Las callejuelas pequeñas ya se funden en S, así que elige una zona pequeña de 300–500 m.",
              "M (8 cm, ≈8 €) — el punto justo: una zona de 400–800 m se lee entera, el modelo se nota en la estantería sin ocupar media habitación. La elección más común para regalo.",
              "L (11 cm, ≈11 €) y XL (15 cm, ≈13 €) — piezas de interior: se ven patios y edificios individuales, y puedes tomar una zona más amplia de hasta 1–1,5 km. XL luce especialmente con relieve — los desniveles en gran superficie resultan espectaculares.",
            ],
          },
          {
            h2: "La regla de la zona: más pequeña, más detalle",
            p: [
              "El tamaño físico se divide entre la zona capturada: cuanto mayor el trozo de ciudad, más pequeño cada edificio. Una zona de 500 m en un modelo de 8 cm muestra cada casa; una de 2 km en el mismo modelo convierte las manzanas en textura. ¿Quieres «todo el centro»? — L/XL; ¿quieres «mi patio»? — basta S/M.",
            ],
          },
          {
            h2: "Relieve y series",
            p: [
              "El relieve del terreno (+≈1,5 €) da al mapa desniveles reales — imprescindible para Kyiv, Leópolis o los Cárpatos, opcional para ciudades llanas. Y si quieres abarcar un territorio grande sin perder detalle — pide una serie de azulejos que se unen en un solo panel.",
            ],
          },
        ],
        ctaLabel: "Elegir zona y tamaño",
        ctaHref: "/create",
      },
    },
  },
  {
    slug: "relief-na-3d-mapi",
    date: "2026-07-13",
    content: {
      uk: {
        title: "Рельєф на 3D-мапі: коли він потрібен, а коли ні",
        description:
          "Опція «рельєф місцевості» (+60 ₴) додає мапі реальні перепади висот. Для яких міст рельєф обов'язковий, для яких зайвий, і як він друкується.",
        h1: "Рельєф на 3D-мапі: вмикати чи ні",
        intro:
          "Рельєф — опція, що додає мапі третій вимір ландшафту: пагорби, схили й долини друкуються з реальних супутникових даних висот. Це +60 ₴ до ціни — і для одних міст це найкраща частина моделі, а для інших просто непомітна. Розберімося, коли воно того варте.",
        sections: [
          {
            h2: "Міста, де рельєф — обов'язковий",
            p: [
              "Київ: перепад між рівнем Дніпра і верхнім містом — близько ста метрів; з рельєфом видно і схили, і чому Андріївський узвіз — узвіз. Львів з його пагорбами, Кам'янець-Подільський з каньйоном, будь-які Карпати — тут рельєф робить половину враження.",
              "Окремий випадок — гори без міста: полонини, хребти, місце походу. Для них є топографічний режим брелока, де сам рельєф і є сюжетом.",
            ],
          },
          {
            h2: "Міста, де можна зекономити",
            p: [
              "Степові й приморські рівнини — Херсон, Миколаїв, більша частина Одеси — мають перепади в межах кількох метрів: на моделі 8 см це частки міліметра, які губляться на тлі будинків. Тут чесніше лишити рельєф вимкненим і витратити різницю на розмір більший.",
            ],
          },
          {
            h2: "Як рельєф друкується",
            p: [
              "Висоти беруться з глобальних даних супутникової зйомки, згладжуються від шуму і масштабуються так, щоб перепади читались, але будинки не «тонули». Дороги й річки акуратно лягають на схили — річка тече долиною, а не висить у повітрі.",
            ],
          },
        ],
        ctaLabel: "Створити мапу з рельєфом",
        ctaHref: "/create",
      },
      en: {
        title: "Terrain relief on a 3D map: when you need it and when you don't",
        description:
          "The terrain relief option adds real elevation to the map. For which cities relief is a must, where it's pointless, and how it prints.",
        h1: "Terrain relief on a 3D map: on or off",
        intro:
          "Relief adds the landscape's third dimension: hills, slopes and valleys print from real satellite elevation data. For some cities it's the best part of the model; for others it's simply invisible. Let's sort out when it's worth it.",
        sections: [
          {
            h2: "Cities where relief is a must",
            p: [
              "Kyiv: the drop between the Dnipro level and the Upper City is about a hundred meters — with relief you see the slopes and why the descents are descents. Lviv with its hills, any Carpathian town — relief does half the impression there.",
              "A separate case is mountains without a city: ridges, a hiking spot. For those there's the topographic keychain mode where the relief itself is the story.",
            ],
          },
          {
            h2: "Cities where you can save",
            p: [
              "Steppe and seaside plains — Kherson, Mykolaiv, most of Odesa — vary by a few meters: on an 8 cm model that's fractions of a millimeter lost behind the buildings. Here it's more honest to keep relief off and spend the difference on a bigger size.",
            ],
          },
          {
            h2: "How relief is printed",
            p: [
              "Elevations come from global satellite data, get denoised and scaled so drops stay readable without drowning the buildings. Roads and rivers drape neatly over slopes — the river flows in the valley instead of hanging mid-air.",
            ],
          },
        ],
        ctaLabel: "Create a map with relief",
        ctaHref: "/create",
      },
      de: {
        title: "Geländerelief auf der 3D-Karte: wann es sich lohnt und wann nicht",
        description:
          "Die Option «Geländerelief» (+≈1,5 €) gibt der Karte echte Höhenunterschiede. Für welche Städte Relief Pflicht ist, wo es überflüssig ist und wie es gedruckt wird.",
        h1: "Geländerelief auf der 3D-Karte: an oder aus",
        intro:
          "Relief fügt der Karte die dritte Dimension der Landschaft hinzu: Hügel, Hänge und Täler werden aus echten Satelliten-Höhendaten gedruckt. Für manche Städte ist es der beste Teil des Modells, für andere schlicht unsichtbar. Klären wir, wann es sich lohnt.",
        sections: [
          {
            h2: "Städte, in denen Relief Pflicht ist",
            p: [
              "Kyiv: der Unterschied zwischen Dnipro-Niveau und Oberstadt beträgt rund hundert Meter — mit Relief sieht man die Hänge und versteht, warum die Abstiege Abstiege sind. Lwiw mit seinen Hügeln, Kamjanez-Podilskyj mit der Schlucht, die ganzen Karpaten — hier macht das Relief den halben Eindruck.",
              "Ein Sonderfall sind Berge ohne Stadt: Almen, Kämme, der Ort einer Wanderung. Dafür gibt es den topografischen Anhänger-Modus, in dem das Relief selbst die Geschichte ist.",
            ],
          },
          {
            h2: "Städte, wo man sparen kann",
            p: [
              "Steppen- und Küstenebenen — Cherson, Mykolajiw, der größte Teil von Odessa — variieren um wenige Meter: auf einem 8-cm-Modell sind das Bruchteile eines Millimeters, die hinter den Gebäuden verschwinden. Hier ist es ehrlicher, das Relief auszulassen und die Differenz in eine größere Größe zu stecken.",
            ],
          },
          {
            h2: "Wie das Relief gedruckt wird",
            p: [
              "Die Höhen stammen aus globalen Satellitendaten, werden entrauscht und so skaliert, dass die Unterschiede lesbar bleiben, ohne die Gebäude zu «ertränken». Straßen und Flüsse legen sich sauber über die Hänge — der Fluss fließt im Tal statt in der Luft zu hängen.",
            ],
          },
        ],
        ctaLabel: "Karte mit Relief erstellen",
        ctaHref: "/create",
      },
      pl: {
        title: "Rzeźba terenu na mapie 3D: kiedy jest potrzebna, a kiedy nie",
        description:
          "Opcja «rzeźba terenu» (+≈1,5 €) dodaje mapie prawdziwe różnice wysokości. Dla których miast rzeźba jest obowiązkowa, gdzie zbędna i jak się drukuje.",
        h1: "Rzeźba terenu na mapie 3D: włączać czy nie",
        intro:
          "Rzeźba dodaje mapie trzeci wymiar krajobrazu: wzgórza, zbocza i doliny drukują się z prawdziwych satelitarnych danych wysokości. Dla jednych miast to najlepsza część modelu, dla innych po prostu niewidoczna. Rozłóżmy na czynniki, kiedy warto.",
        sections: [
          {
            h2: "Miasta, gdzie rzeźba jest obowiązkowa",
            p: [
              "Kijów: różnica między poziomem Dniepru a Górnym Miastem to około stu metrów; z rzeźbą widać zbocza i to, dlaczego zejścia są zejściami. Lwów ze swoimi wzgórzami, Kamieniec Podolski z kanionem, całe Karpaty — tu rzeźba robi połowę wrażenia.",
              "Osobny przypadek to góry bez miasta: połoniny, grzbiety, miejsce wędrówki. Dla nich jest tryb topograficzny breloka, gdzie sama rzeźba jest tematem.",
            ],
          },
          {
            h2: "Miasta, gdzie można zaoszczędzić",
            p: [
              "Stepowe i nadmorskie równiny — Chersoń, Mikołajów, większość Odessy — mają różnice rzędu kilku metrów: na modelu 8 cm to ułamki milimetra, które giną za budynkami. Tu uczciwiej zostawić rzeźbę wyłączoną i wydać różnicę na większy rozmiar.",
            ],
          },
          {
            h2: "Jak drukuje się rzeźba",
            p: [
              "Wysokości pochodzą z globalnych danych satelitarnych, są odszumiane i skalowane tak, by różnice były czytelne, a budynki nie «tonęły». Drogi i rzeki układają się starannie na zboczach — rzeka płynie doliną, a nie wisi w powietrzu.",
            ],
          },
        ],
        ctaLabel: "Stwórz mapę z rzeźbą terenu",
        ctaHref: "/create",
      },
      fr: {
        title: "Relief du terrain sur une carte 3D : quand il faut, quand il ne sert à rien",
        description:
          "L'option «relief du terrain» (+≈1,5 €) ajoute de vrais dénivelés à la carte. Pour quelles villes le relief est indispensable, où il est inutile, et comment il s'imprime.",
        h1: "Relief du terrain sur une carte 3D : activer ou non",
        intro:
          "Le relief ajoute la troisième dimension du paysage : collines, pentes et vallées s'impriment à partir de vraies données satellites d'altitude. Pour certaines villes c'est la meilleure partie du modèle ; pour d'autres, c'est tout simplement invisible. Voyons quand ça vaut le coup.",
        sections: [
          {
            h2: "Les villes où le relief est indispensable",
            p: [
              "Kyiv : la différence entre le niveau du Dnipro et la ville haute atteint une centaine de mètres ; avec le relief on voit les coteaux et on comprend pourquoi les descentes sont des descentes. Lviv et ses collines, Kamianets-Podilskyi et son canyon, toutes les Carpates — là, le relief fait la moitié de l'effet.",
              "Cas à part : la montagne sans ville — alpages, crêtes, le lieu d'une randonnée. Pour cela il y a le mode topographique du porte-clés, où le relief lui-même est le sujet.",
            ],
          },
          {
            h2: "Les villes où l'on peut économiser",
            p: [
              "Les plaines steppiques et littorales — Kherson, Mykolaïv, la majeure partie d'Odessa — varient de quelques mètres : sur un modèle de 8 cm, ce sont des fractions de millimètre qui se perdent derrière les bâtiments. Là, il est plus honnête de laisser le relief désactivé et de mettre la différence dans une taille supérieure.",
            ],
          },
          {
            h2: "Comment le relief s'imprime",
            p: [
              "Les altitudes proviennent de données satellites globales, sont débruitées et mises à l'échelle pour que les dénivelés restent lisibles sans «noyer» les bâtiments. Routes et rivières se drapent proprement sur les pentes — la rivière coule dans la vallée au lieu de flotter en l'air.",
            ],
          },
        ],
        ctaLabel: "Créer une carte avec relief",
        ctaHref: "/create",
      },
      es: {
        title: "Relieve del terreno en un mapa 3D: cuándo hace falta y cuándo no",
        description:
          "La opción «relieve del terreno» (+≈1,5 €) añade desniveles reales al mapa. Para qué ciudades el relieve es imprescindible, dónde sobra y cómo se imprime.",
        h1: "Relieve del terreno en un mapa 3D: activar o no",
        intro:
          "El relieve añade al mapa la tercera dimensión del paisaje: colinas, laderas y valles se imprimen a partir de datos satelitales de elevación reales. Para unas ciudades es la mejor parte del modelo; para otras, simplemente invisible. Veamos cuándo merece la pena.",
        sections: [
          {
            h2: "Ciudades donde el relieve es imprescindible",
            p: [
              "Kyiv: el desnivel entre el nivel del Dnipró y la ciudad alta ronda los cien metros; con relieve se ven las laderas y se entiende por qué las cuestas son cuestas. Leópolis con sus colinas, Kamianets-Podilskyi con su cañón, todos los Cárpatos — aquí el relieve hace la mitad de la impresión.",
              "Caso aparte: montaña sin ciudad — praderas de altura, crestas, el lugar de una excursión. Para eso está el modo topográfico del llavero, donde el propio relieve es el tema.",
            ],
          },
          {
            h2: "Ciudades donde se puede ahorrar",
            p: [
              "Las llanuras esteparias y costeras — Jersón, Mykolaiv, la mayor parte de Odesa — varían unos pocos metros: en un modelo de 8 cm son fracciones de milímetro que se pierden tras los edificios. Aquí es más honesto dejar el relieve apagado y gastar la diferencia en un tamaño mayor.",
            ],
          },
          {
            h2: "Cómo se imprime el relieve",
            p: [
              "Las alturas provienen de datos satelitales globales, se les quita el ruido y se escalan para que los desniveles sean legibles sin «ahogar» los edificios. Carreteras y ríos se posan con precisión sobre las laderas — el río corre por el valle en lugar de flotar en el aire.",
            ],
          },
        ],
        ctaLabel: "Crear un mapa con relieve",
        ctaHref: "/create",
      },
    },
  },
  {
    slug: "yak-doglyadaty-za-3d-drukom",
    date: "2026-07-13",
    content: {
      uk: {
        title: "Як доглядати за 3D-друкованим виробом: 5 простих правил",
        description:
          "Догляд за 3D-мапою, брелоком чи магнітом з PLA: як чистити від пилу, чого уникати (тепло, розчинники), як зберегти вигляд на роки.",
        h1: "Догляд за 3D-друкованим виробом",
        intro:
          "3D-друкована мапа не вимагає особливого догляду — PLA стабільний і живе на полиці роками. Але кілька простих правил допоможуть зберегти вигляд новим, а одне-єдине справжнє табу вбереже від зіпсованої моделі.",
        sections: [
          {
            h2: "Головне табу: тепло",
            p: [
              "PLA розм'якшується від ~60 °C. Це температура, якої легко досягти у двох місцях: на торпеді автомобіля влітку і біля кухонної духовки чи батареї впритул. Деформовану теплом модель не відновити. Звичайне сонячне підвіконня безпечне — пряме сонце кімнатної температури моделі не шкодить, хіба що за роки може ледь висвітлити яскраві кольори.",
            ],
          },
          {
            h2: "Чищення: сухо або ледь волого",
            p: [
              "Пил з мапи найкраще знімати м'яким сухим пензлем (личить косметичний) — він дістає між будинками. Раз на пів року можна пройтись вологою серветкою; уникайте лише розчинників, ацетону і спиртових засобів — вони роз'їдають поверхню пластику.",
              "Брелок можна просто мити теплою (не гарячою) водою з милом.",
            ],
          },
          {
            h2: "Дрібний ремонт",
            p: [
              "Якщо від падіння відколовся дрібний елемент — його чудово клеїть звичайний цианакрилатний суперклей: крапля, притиснути на 30 секунд, шов непомітний. Для панно на стіні використовуйте двосторонній скотч для картин або пласкі гачки — свердлити пластик не потрібно.",
            ],
          },
        ],
        ctaLabel: "Створити свою мапу",
        ctaHref: "/create",
      },
      en: {
        title: "Caring for a 3D-printed item: 5 simple rules",
        description:
          "Care for a PLA 3D map, keychain or magnet: how to dust it, what to avoid (heat, solvents), how to keep it looking new for years.",
        h1: "Caring for a 3D-printed item",
        intro:
          "A 3D-printed map needs no special care — PLA is stable and lives on a shelf for years. A few simple rules keep it looking new, and one single real taboo saves you from a ruined model.",
        sections: [
          {
            h2: "The one taboo: heat",
            p: [
              "PLA softens from ~60 °C — easily reached in two places: a car dashboard in summer and right next to an oven or radiator. A heat-warped model can't be restored. An ordinary sunny windowsill is safe; direct sun at room temperature only slightly fades bright colors over years.",
            ],
          },
          {
            h2: "Cleaning: dry or barely damp",
            p: [
              "Dust comes off best with a soft dry brush that reaches between the buildings. Twice a year wipe with a damp cloth; just avoid solvents, acetone and alcohol cleaners — they etch the plastic surface.",
              "A keychain can simply be washed with warm (not hot) soapy water.",
            ],
          },
          {
            h2: "Small repairs",
            p: [
              "If a tiny element chips off in a fall, ordinary cyanoacrylate super glue fixes it: a drop, press for 30 seconds, the seam is invisible. For a wall panel use picture-hanging tape or flat hooks — no drilling needed.",
            ],
          },
        ],
        ctaLabel: "Create your map",
        ctaHref: "/create",
      },
      de: {
        title: "Pflege eines 3D-gedruckten Stücks: 5 einfache Regeln",
        description:
          "Pflege für eine PLA-3D-Karte, einen Anhänger oder Magneten: wie man Staub entfernt, was zu vermeiden ist (Hitze, Lösungsmittel) und wie es jahrelang wie neu bleibt.",
        h1: "Pflege eines 3D-gedruckten Stücks",
        intro:
          "Eine 3D-gedruckte Karte braucht keine besondere Pflege — PLA ist stabil und lebt jahrelang im Regal. Ein paar einfache Regeln halten sie wie neu, und ein einziges echtes Tabu bewahrt dich vor einem ruinierten Modell.",
        sections: [
          {
            h2: "Das eine Tabu: Hitze",
            p: [
              "PLA erweicht ab ~60 °C — leicht erreicht an zwei Orten: dem Armaturenbrett im Sommer und direkt neben Ofen oder Heizkörper. Ein hitzeverformtes Modell lässt sich nicht wiederherstellen. Eine gewöhnliche sonnige Fensterbank ist sicher; direkte Sonne bei Raumtemperatur lässt kräftige Farben über Jahre nur leicht ausbleichen.",
            ],
          },
          {
            h2: "Reinigung: trocken oder leicht feucht",
            p: [
              "Staub geht am besten mit einem weichen trockenen Pinsel ab, der zwischen die Gebäude reicht. Zweimal im Jahr mit einem feuchten Tuch abwischen; vermeide nur Lösungsmittel, Aceton und alkoholhaltige Reiniger — sie greifen die Kunststoffoberfläche an.",
              "Einen Anhänger kann man einfach mit warmem (nicht heißem) Seifenwasser waschen.",
            ],
          },
          {
            h2: "Kleine Reparaturen",
            p: [
              "Bricht bei einem Sturz ein winziges Element ab, klebt gewöhnlicher Cyanacrylat-Sekundenkleber es perfekt: ein Tropfen, 30 Sekunden andrücken, die Naht ist unsichtbar. Für ein Wandpanel nimm Bilder-Klebeband oder flache Haken — Bohren ist nicht nötig.",
            ],
          },
        ],
        ctaLabel: "Deine Karte erstellen",
        ctaHref: "/create",
      },
      pl: {
        title: "Jak dbać o wyrób z druku 3D: 5 prostych zasad",
        description:
          "Pielęgnacja mapy 3D, breloka lub magnesu z PLA: jak usuwać kurz, czego unikać (ciepło, rozpuszczalniki), jak zachować wygląd jak nowy przez lata.",
        h1: "Jak dbać o wyrób z druku 3D",
        intro:
          "Wydrukowana mapa 3D nie wymaga specjalnej pielęgnacji — PLA jest stabilny i żyje na półce latami. Kilka prostych zasad utrzyma ją jak nową, a jedno jedyne prawdziwe tabu uchroni przed zniszczonym modelem.",
        sections: [
          {
            h2: "Jedyne tabu: ciepło",
            p: [
              "PLA mięknie od ~60 °C — łatwo o to w dwóch miejscach: na desce rozdzielczej auta latem i tuż przy piekarniku czy kaloryferze. Odkształconego przez ciepło modelu nie da się przywrócić. Zwykły słoneczny parapet jest bezpieczny; bezpośrednie słońce w temperaturze pokojowej tylko lekko wypłukuje jaskrawe kolory przez lata.",
            ],
          },
          {
            h2: "Czyszczenie: na sucho lub ledwo wilgotno",
            p: [
              "Kurz najlepiej schodzi miękkim, suchym pędzelkiem, który sięga między budynki. Dwa razy w roku przetrzyj wilgotną ściereczką; unikaj tylko rozpuszczalników, acetonu i środków na alkoholu — trawią powierzchnię plastiku.",
              "Brelok można po prostu umyć ciepłą (nie gorącą) wodą z mydłem.",
            ],
          },
          {
            h2: "Drobne naprawy",
            p: [
              "Jeśli po upadku odłamie się drobny element — świetnie skleja go zwykły cyjanoakrylowy super klej: kropla, docisnąć na 30 sekund, spoina niewidoczna. Do panelu na ścianie użyj taśmy do obrazów lub płaskich haczyków — wiercenie nie jest potrzebne.",
            ],
          },
        ],
        ctaLabel: "Stwórz swoją mapę",
        ctaHref: "/create",
      },
      fr: {
        title: "Entretenir une pièce imprimée en 3D : 5 règles simples",
        description:
          "Entretien d'une carte 3D, d'un porte-clés ou d'un magnet en PLA : comment dépoussiérer, quoi éviter (chaleur, solvants), comment garder l'aspect neuf pendant des années.",
        h1: "Entretenir une pièce imprimée en 3D",
        intro:
          "Une carte imprimée en 3D ne demande aucun entretien particulier — le PLA est stable et vit des années sur une étagère. Quelques règles simples la gardent comme neuve, et un seul vrai tabou vous évite un modèle fichu.",
        sections: [
          {
            h2: "Le seul tabou : la chaleur",
            p: [
              "Le PLA ramollit dès ~60 °C — facilement atteint à deux endroits : le tableau de bord en été et le voisinage immédiat d'un four ou d'un radiateur. Un modèle déformé par la chaleur ne se répare pas. Un rebord de fenêtre ensoleillé ordinaire est sans danger ; le soleil direct à température ambiante ne fait que légèrement pâlir les couleurs vives au fil des ans.",
            ],
          },
          {
            h2: "Nettoyage : à sec ou à peine humide",
            p: [
              "La poussière part le mieux avec un pinceau doux et sec qui passe entre les bâtiments. Deux fois par an, essuyez avec un chiffon humide ; évitez seulement les solvants, l'acétone et les nettoyants alcoolisés — ils attaquent la surface du plastique.",
              "Un porte-clés peut simplement se laver à l'eau tiède (pas chaude) savonneuse.",
            ],
          },
          {
            h2: "Petites réparations",
            p: [
              "Si un petit élément se casse lors d'une chute, une colle cyanoacrylate ordinaire le répare parfaitement : une goutte, presser 30 secondes, la jointure est invisible. Pour un panneau mural, utilisez de l'adhésif pour cadres ou des crochets plats — inutile de percer.",
            ],
          },
        ],
        ctaLabel: "Créer votre carte",
        ctaHref: "/create",
      },
      es: {
        title: "Cómo cuidar una pieza impresa en 3D: 5 reglas sencillas",
        description:
          "Cuidado de un mapa 3D, llavero o imán de PLA: cómo quitar el polvo, qué evitar (calor, disolventes), cómo mantener el aspecto de nuevo durante años.",
        h1: "Cómo cuidar una pieza impresa en 3D",
        intro:
          "Un mapa impreso en 3D no necesita cuidados especiales — el PLA es estable y vive años en una estantería. Unas pocas reglas sencillas lo mantienen como nuevo, y un único tabú real te salva de un modelo arruinado.",
        sections: [
          {
            h2: "El único tabú: el calor",
            p: [
              "El PLA se ablanda a partir de ~60 °C — algo fácil en dos sitios: el salpicadero del coche en verano y justo al lado de un horno o radiador. Un modelo deformado por el calor no se puede restaurar. Un alféizar soleado normal es seguro; el sol directo a temperatura ambiente solo desvanece ligeramente los colores vivos con los años.",
            ],
          },
          {
            h2: "Limpieza: en seco o apenas húmedo",
            p: [
              "El polvo sale mejor con un pincel suave y seco que llegue entre los edificios. Dos veces al año pasa un paño húmedo; evita solo disolventes, acetona y limpiadores con alcohol — atacan la superficie del plástico.",
              "Un llavero se puede lavar simplemente con agua tibia (no caliente) y jabón.",
            ],
          },
          {
            h2: "Reparaciones pequeñas",
            p: [
              "Si en una caída se desprende un elemento diminuto, un superpegamento de cianoacrilato corriente lo arregla perfectamente: una gota, presionar 30 segundos, la junta queda invisible. Para un panel de pared usa cinta de colgar cuadros o ganchos planos — no hace falta taladrar.",
            ],
          },
        ],
        ctaLabel: "Crear tu mapa",
        ctaHref: "/create",
      },
    },
  },
  {
    slug: "magnit-z-kartoyu-mista",
    date: "2026-07-13",
    content: {
      uk: {
        title: "Магніт з картою міста на холодильник: сувенір, якого немає в кіосках",
        description:
          "Магніт-мапа 6 см (150 ₴): рельєфна 3D-карта обраного району замість типового сувеніра. Своє місто, своя вулиця, свій двір — на холодильнику.",
        h1: "Магніт з картою міста: сувенір про ваше місце",
        intro:
          "Магніти з подорожей зазвичай однакові: Ейфелева вежа, герб міста, панорама з листівки. Магніт-мапа працює інакше — це рельєфна 3D-карта конкретного району, який обираєте ви: не «Київ узагалі», а саме ваш двір на Оболоні чи бабусина вулиця у Львові.",
        sections: [
          {
            h2: "Що це фізично",
            p: [
              "Плаский жетон близько 6 см з надрукованою рельєфом картою: вулиці, будинки, парки, вода. На звороті — магнітна основа, що впевнено тримається на холодильнику чи будь-якій сталевій поверхні. Друк з біопластику Eco PLA, ціна — 150 ₴.",
            ],
          },
          {
            h2: "Навіщо дарують магніти-мапи",
            p: [
              "Це наймасовіший формат «пам'яті місця»: недорогий, легкий (можна надіслати листом навіть за кордон), і його бачать щодня — холодильник відкривають частіше, ніж дивляться на полицю. Колекція з кількох магнітів різних районів складається у власну маленьку мапу життя: місто дитинства, місто навчання, місто, де народилися діти.",
              "Для бізнесу — сувенір з районом офісу чи міста компанії, який гості справді забирають і чіпляють, а не лишають у шухляді.",
            ],
          },
          {
            h2: "Як замовити",
            p: [
              "У конструкторі оберіть ділянку — так само, як для великої мапи, — і формат «магніт». Виготовлення 1–3 робочі дні, доставка Новою Поштою по Україні або у країни ЄС.",
            ],
          },
        ],
        ctaLabel: "Створити магніт зі своїм районом",
        ctaHref: "/create",
      },
      en: {
        title: "A city map fridge magnet: the souvenir kiosks don't sell",
        description:
          "A 6 cm map magnet: a relief 3D map of your chosen district instead of a generic souvenir. Your city, your street, your backyard — on the fridge.",
        h1: "A city map magnet: a souvenir about your place",
        intro:
          "Travel magnets are usually identical: a tower, a coat of arms, a postcard panorama. A map magnet works differently — it's a relief 3D map of the exact district you choose: not 'Kyiv in general' but your own backyard in Obolon or your grandmother's street in Lviv.",
        sections: [
          {
            h2: "What it physically is",
            p: [
              "A flat ~6 cm tag with a relief-printed map: streets, buildings, parks, water. On the back — a magnetic base that holds firmly on a fridge or any steel surface. Printed in Eco PLA bioplastic.",
            ],
          },
          {
            h2: "Why people gift map magnets",
            p: [
              "It's the most accessible 'memory of a place' format: inexpensive, light (ships abroad in an envelope), and seen daily — the fridge gets opened more often than the shelf gets looked at. A few magnets of different districts become a little map of one's life: the childhood city, the university city, the city where the kids were born.",
              "For business — a souvenir with the office district that guests actually take and hang, instead of leaving in a drawer.",
            ],
          },
          {
            h2: "How to order",
            p: [
              "Pick the area in the builder — same as for a big map — and choose the magnet format. Production 1–3 business days, delivery to Ukraine and the EU.",
            ],
          },
        ],
        ctaLabel: "Create a magnet with your district",
        ctaHref: "/create",
      },
      de: {
        title: "Kühlschrankmagnet mit Stadtkarte: das Souvenir, das es am Kiosk nicht gibt",
        description:
          "Karten-Magnet 6 cm (≈4 €): eine 3D-Reliefkarte deines Viertels statt eines Standard-Souvenirs. Deine Stadt, deine Straße, dein Hof — am Kühlschrank.",
        h1: "Magnet mit Stadtkarte: ein Souvenir über deinen Ort",
        intro:
          "Reisemagnete sehen meist gleich aus: ein Turm, ein Wappen, eine Postkarten-Panorama. Ein Karten-Magnet funktioniert anders — er ist eine 3D-Reliefkarte genau des Viertels, das du wählst: nicht «Kyiv allgemein», sondern dein Hof in Obolon oder die Straße deiner Großmutter in Lwiw.",
        sections: [
          {
            h2: "Was es physisch ist",
            p: [
              "Ein flaches Täfelchen von etwa 6 cm mit reliefgedruckter Karte: Straßen, Gebäude, Parks, Wasser. Auf der Rückseite eine magnetische Basis, die sicher am Kühlschrank oder jeder Stahlfläche hält. Gedruckt aus Eco-PLA-Bioplastik, Preis ≈4 €.",
            ],
          },
          {
            h2: "Warum man Karten-Magnete verschenkt",
            p: [
              "Es ist das zugänglichste Format der «Erinnerung an einen Ort»: günstig, leicht (lässt sich sogar als Brief ins Ausland schicken) und täglich sichtbar — den Kühlschrank öffnet man öfter, als man aufs Regal schaut. Eine Sammlung mehrerer Magnete verschiedener Viertel ergibt eine eigene kleine Lebenskarte: die Stadt der Kindheit, die Stadt des Studiums, die Stadt, in der die Kinder geboren wurden.",
              "Für Unternehmen — ein Souvenir mit dem Büro-Viertel, das Gäste wirklich mitnehmen und aufhängen, statt es in der Schublade zu lassen.",
            ],
          },
          {
            h2: "So bestellst du",
            p: [
              "Wähle im Konfigurator den Bereich — genau wie für eine große Karte — und das Format «Magnet». Fertigung 1–3 Werktage, Versand in die Ukraine und die EU.",
            ],
          },
        ],
        ctaLabel: "Magnet mit deinem Viertel erstellen",
        ctaHref: "/create",
      },
      pl: {
        title: "Magnes z mapą miasta na lodówkę: pamiątka, której nie ma w kioskach",
        description:
          "Magnes-mapa 6 cm (≈4 €): reliefowa mapa 3D wybranej dzielnicy zamiast typowej pamiątki. Twoje miasto, twoja ulica, twoje podwórko — na lodówce.",
        h1: "Magnes z mapą miasta: pamiątka o Twoim miejscu",
        intro:
          "Magnesy z podróży zwykle są takie same: wieża, herb miasta, panorama z pocztówki. Magnes-mapa działa inaczej — to reliefowa mapa 3D konkretnej dzielnicy, którą wybierasz Ty: nie «Kijów w ogóle», tylko Twoje podwórko na Obołoniu albo ulica babci we Lwowie.",
        sections: [
          {
            h2: "Co to jest fizycznie",
            p: [
              "Płaska zawieszka około 6 cm z mapą wydrukowaną reliefem: ulice, budynki, parki, woda. Z tyłu magnetyczna podstawa, która pewnie trzyma się lodówki lub dowolnej stalowej powierzchni. Druk z bioplastiku Eco PLA, cena ≈4 €.",
            ],
          },
          {
            h2: "Po co daruje się magnesy-mapy",
            p: [
              "To najbardziej dostępny format «pamięci o miejscu»: niedrogi, lekki (można wysłać listem nawet za granicę) i widoczny codziennie — lodówkę otwiera się częściej, niż patrzy na półkę. Kolekcja kilku magnesów różnych dzielnic składa się we własną małą mapę życia: miasto dzieciństwa, miasto studiów, miasto, w którym urodziły się dzieci.",
              "Dla biznesu — pamiątka z dzielnicą biura, którą goście naprawdę zabierają i wieszają, zamiast zostawiać w szufladzie.",
            ],
          },
          {
            h2: "Jak zamówić",
            p: [
              "W kreatorze wybierz obszar — tak samo jak dla dużej mapy — i format «magnes». Wykonanie 1–3 dni robocze, wysyłka na Ukrainę lub do krajów UE.",
            ],
          },
        ],
        ctaLabel: "Stwórz magnes ze swoją dzielnicą",
        ctaHref: "/create",
      },
      fr: {
        title: "Magnet de frigo avec carte de ville : le souvenir qu'aucun kiosque ne vend",
        description:
          "Magnet-carte 6 cm (≈4 €) : une carte 3D en relief du quartier de votre choix au lieu d'un souvenir standard. Votre ville, votre rue, votre cour — sur le frigo.",
        h1: "Magnet avec carte de ville : un souvenir de votre lieu",
        intro:
          "Les magnets de voyage se ressemblent tous : une tour, un blason, une panorama de carte postale. Un magnet-carte fonctionne autrement — c'est une carte 3D en relief du quartier précis que vous choisissez : pas «Kyiv en général», mais votre cour à Obolon ou la rue de votre grand-mère à Lviv.",
        sections: [
          {
            h2: "Ce que c'est physiquement",
            p: [
              "Une plaque plate d'environ 6 cm avec une carte imprimée en relief : rues, bâtiments, parcs, eau. Au dos, une base magnétique qui tient fermement sur un frigo ou toute surface en acier. Imprimé en Eco PLA, prix ≈4 €.",
            ],
          },
          {
            h2: "Pourquoi on offre des magnets-cartes",
            p: [
              "C'est le format le plus accessible de «mémoire d'un lieu» : peu cher, léger (il part même à l'étranger dans une enveloppe) et vu chaque jour — on ouvre le frigo plus souvent qu'on ne regarde l'étagère. Une collection de plusieurs magnets de quartiers différents compose une petite carte de sa vie : la ville de l'enfance, celle des études, celle où les enfants sont nés.",
              "Pour une entreprise — un souvenir avec le quartier du bureau que les invités emportent et accrochent vraiment, au lieu de le laisser dans un tiroir.",
            ],
          },
          {
            h2: "Comment commander",
            p: [
              "Dans le configurateur, choisissez la zone — comme pour une grande carte — et le format «magnet». Fabrication 1–3 jours ouvrés, livraison en Ukraine et dans l'UE.",
            ],
          },
        ],
        ctaLabel: "Créer un magnet avec votre quartier",
        ctaHref: "/create",
      },
      es: {
        title: "Imán de nevera con mapa de ciudad: el recuerdo que no venden en los quioscos",
        description:
          "Imán-mapa de 6 cm (≈4 €): un mapa 3D en relieve del barrio que elijas en lugar de un recuerdo genérico. Tu ciudad, tu calle, tu patio — en la nevera.",
        h1: "Imán con mapa de ciudad: un recuerdo sobre tu lugar",
        intro:
          "Los imanes de viaje suelen ser idénticos: una torre, un escudo, una panorámica de postal. Un imán-mapa funciona distinto — es un mapa 3D en relieve del barrio exacto que eliges tú: no «Kyiv en general», sino tu patio en Obolon o la calle de tu abuela en Leópolis.",
        sections: [
          {
            h2: "Qué es físicamente",
            p: [
              "Una placa plana de unos 6 cm con el mapa impreso en relieve: calles, edificios, parques, agua. Al dorso, una base magnética que se sujeta con firmeza a la nevera o a cualquier superficie de acero. Impreso en bioplástico Eco PLA, precio ≈4 €.",
            ],
          },
          {
            h2: "Por qué se regalan imanes-mapa",
            p: [
              "Es el formato más accesible de «memoria de un lugar»: barato, ligero (se envía al extranjero incluso en un sobre) y se ve a diario — la nevera se abre más veces de las que se mira la estantería. Una colección de varios imanes de distintos barrios compone un pequeño mapa de la propia vida: la ciudad de la infancia, la de los estudios, la ciudad donde nacieron los hijos.",
              "Para empresas — un recuerdo con el barrio de la oficina que los invitados sí se llevan y cuelgan, en vez de dejarlo en un cajón.",
            ],
          },
          {
            h2: "Cómo pedirlo",
            p: [
              "En el configurador elige la zona — igual que para un mapa grande — y el formato «imán». Fabricación 1–3 días hábiles, envío a Ucrania y a la UE.",
            ],
          },
        ],
        ctaLabel: "Crear un imán con tu barrio",
        ctaHref: "/create",
      },
    },
  },
  {
    slug: "3d-druk-na-zamovlennya",
    date: "2026-07-16",
    content: {
      uk: {
        title: "3D-друк на замовлення: Київ, Львів, Вінниця та вся Україна",
        description:
          "3D-друк мап, брелоків і панно на замовлення з доставкою в будь-яке місто України: Київ, Львів, Одесу, Дніпро, Вінницю. Eco PLA, 1–3 робочі дні, від 120 ₴.",
        h1: "3D-друк на замовлення з доставкою по всій Україні",
        intro:
          "Шукаєте 3D-друк на замовлення у своєму місті? Ми спеціалізуємось на одному типі виробів — персональних 3D-мапах: карти районів, брелоки з маршрутами, настінні панно й магніти. Друкуємо у власній майстерні та надсилаємо Новою Поштою в будь-яке місто України за 1–3 робочі дні — тож не важливо, чи ви в Києві, Львові, Вінниці чи маленькому селищі: доставка працює однаково швидко.",
        sections: [
          {
            h2: "Що ми друкуємо",
            p: [
              "3D-мапи міст (від 250 ₴): обираєте будь-який район — вулиці, будинки з реальними висотами, парки й річки друкуються об'ємною моделлю 5,5–15 см. Брелоки з картою чи GPX-маршрутом (від 120 ₴): жетон 55×30 мм з вашим районом і написом. Панно на стіну: великі мапи з кількох плиток. Магніти з районом міста (150 ₴).",
              "Ми не друкуємо чужі STL-файли, деталі чи фігурки — лише мапи. Зате мапи робимо краще за будь-кого: власний конструктор будує модель з даних OpenStreetMap за кілька хвилин, і ви бачите 3D-превʼю до оплати.",
            ],
          },
          {
            h2: "Як це працює для будь-якого міста",
            p: [
              "Київ, Харків, Одеса, Дніпро, Львів, Запоріжжя, Вінниця, Полтава, Луцьк, Хмельницький, Ужгород — конструктор працює з будь-якою точкою України та світу. Обираєте ділянку на карті, налаштовуєте розмір і стиль, тиснете «Створити» — за 2–4 хвилини модель готова.",
              "Далі два шляхи: замовляєте друк у нас (Eco PLA, виготовлення 1–3 робочі дні, доставка Новою Поштою) — або завантажуєте файл 3MF/STL і друкуєте на власному принтері. Файл відкривається у Bambu Studio та PrusaSlicer без жодної підготовки.",
            ],
          },
          {
            h2: "Ціни",
            p: [
              "Брелок з картою або GPX-треком — від 120 ₴. Магніт на холодильник — 150 ₴. 3D-мапа міста: S (5,5 см) 250 ₴, M (8 см) 350 ₴, L (11 см) 450 ₴, XL (15 см) 550 ₴. Рельєф місцевості +60 ₴. Панно з плиток — ціна за кількість плиток, рахується у конструкторі одразу.",
            ],
          },
          {
            h2: "Чому не звичайна друк-студія",
            p: [
              "Класичні 3D-друк-студії беруть будь-які замовлення й потребують готовий файл. У нас навпаки: файл не потрібен — конструктор сам будує модель вашого району. Це означає нуль передоплат за моделювання, передбачувану ціну і превʼю до замовлення.",
            ],
          },
        ],
        ctaLabel: "Створити свою модель",
        ctaHref: "/create",
        outro:
          "Не впевнені, який район обрати чи який розмір підійде? Напишіть нам — підкажемо й зберемо превʼю безкоштовно.",
      },
      en: {
        title: "Custom 3D printing of maps — shipping across Ukraine & EU",
        description:
          "Custom 3D-printed maps, keychains and wall panels shipped to any city: Kyiv, Lviv, Odesa, Dnipro, Vinnytsia and the EU. Eco PLA, 1–3 business days, from ≈€3.",
        h1: "Custom 3D printing of maps with delivery across Ukraine and the EU",
        intro:
          "Looking for custom 3D printing in your city? We specialise in one type of product — personal 3D maps: district maps, route keychains, wall panels and magnets. We print in our own workshop and ship to any city of Ukraine in 1–3 business days, plus 15 EU countries — so it doesn't matter whether you're in Kyiv, Lviv, Vinnytsia or a small town.",
        sections: [
          {
            h2: "What we print",
            p: [
              "3D city maps (from ≈€6): pick any district — streets, buildings with real heights, parks and rivers printed as a 5.5–15 cm model. Map or GPX-route keychains (from ≈€3): a 55×30 mm tag with your district and text. Wall panels assembled from tiles. Fridge magnets with a city district.",
              "We don't print third-party STL files, parts or figurines — only maps. But we do maps better than anyone: our builder assembles the model from OpenStreetMap data in minutes, and you see a 3D preview before paying.",
            ],
          },
          {
            h2: "How it works for any city",
            p: [
              "The builder works with any point in Ukraine and the world. Pick an area on the map, set the size and style, hit «Create» — the model is ready in 2–4 minutes.",
              "Then two paths: order the print from us (Eco PLA, made in 1–3 business days, shipped) — or download the 3MF/STL and print on your own machine. The file opens in Bambu Studio and PrusaSlicer with zero preparation.",
            ],
          },
          {
            h2: "Prices",
            p: [
              "Map or GPX keychain — from ≈€3. Fridge magnet — ≈€4. 3D city map: S (5.5 cm) ≈€6, M (8 cm) ≈€8, L (11 cm) ≈€11, XL (15 cm) ≈€13. Terrain relief +≈€1.5. Tile panels are priced per tile, calculated live in the builder.",
            ],
          },
          {
            h2: "Why not a regular print studio",
            p: [
              "Classic 3D-print studios take any job and need a ready file. With us it's the opposite: no file needed — the builder constructs the model of your district itself. That means no modelling fees, a predictable price and a preview before you order.",
            ],
          },
        ],
        ctaLabel: "Create your model",
        ctaHref: "/create",
        outro: "Not sure which district or size to pick? Message us — we'll help and build a free preview.",
      },
      de: {
        title: "3D-Druck auf Bestellung: Karten mit Versand in die Ukraine und die EU",
        description:
          "3D-gedruckte Karten, Anhänger und Wandpanels auf Bestellung, Versand in jede Stadt: Kyiv, Lwiw, Odessa, Winnyzja und 15 EU-Länder. Eco PLA, 1–3 Werktage, ab ≈3 €.",
        h1: "3D-Druck auf Bestellung mit Versand in die Ukraine und die EU",
        intro:
          "Du suchst 3D-Druck auf Bestellung in deiner Stadt? Wir sind auf einen Produkttyp spezialisiert — persönliche 3D-Karten: Viertelkarten, Anhänger mit Routen, Wandpanels und Magnete. Wir drucken in der eigenen Werkstatt und versenden in jede Stadt der Ukraine in 1–3 Werktagen sowie in 15 EU-Länder.",
        sections: [
          {
            h2: "Was wir drucken",
            p: [
              "3D-Stadtkarten (ab ≈6 €): du wählst ein beliebiges Viertel — Straßen, Gebäude mit echten Höhen, Parks und Flüsse werden als 5,5–15-cm-Modell gedruckt. Anhänger mit Karte oder GPX-Route (ab ≈3 €): ein 55×30-mm-Täfelchen mit deinem Viertel und Text. Wandpanels aus mehreren Kacheln. Magnete mit einem Stadtviertel.",
              "Wir drucken keine fremden STL-Dateien, Bauteile oder Figuren — nur Karten. Dafür machen wir Karten besser als alle anderen: unser Konfigurator baut das Modell in Minuten aus OpenStreetMap-Daten, und du siehst die 3D-Vorschau vor der Zahlung.",
            ],
          },
          {
            h2: "Wie es für jede Stadt funktioniert",
            p: [
              "Der Konfigurator arbeitet mit jedem Punkt der Ukraine und der Welt. Wähle einen Bereich auf der Karte, stelle Größe und Stil ein, klicke «Erstellen» — in 2–4 Minuten ist das Modell fertig.",
              "Dann zwei Wege: du bestellst den Druck bei uns (Eco PLA, Fertigung 1–3 Werktage, Versand) — oder lädst die 3MF/STL-Datei herunter und druckst auf dem eigenen Gerät. Die Datei öffnet sich ohne Vorbereitung in Bambu Studio und PrusaSlicer.",
            ],
          },
          {
            h2: "Preise",
            p: [
              "Anhänger mit Karte oder GPX — ab ≈3 €. Kühlschrankmagnet — ≈4 €. 3D-Stadtkarte: S (5,5 cm) ≈6 €, M (8 cm) ≈8 €, L (11 cm) ≈11 €, XL (15 cm) ≈13 €. Geländerelief +≈1,5 €. Kachel-Panels werden pro Kachel berechnet, live im Konfigurator.",
            ],
          },
          {
            h2: "Warum kein gewöhnliches Druck-Studio",
            p: [
              "Klassische 3D-Druck-Studios nehmen jeden Auftrag an und brauchen eine fertige Datei. Bei uns ist es umgekehrt: keine Datei nötig — der Konfigurator baut das Modell deines Viertels selbst. Das heißt: keine Modellierungsgebühren, ein vorhersehbarer Preis und eine Vorschau vor der Bestellung.",
            ],
          },
        ],
        ctaLabel: "Dein Modell erstellen",
        ctaHref: "/create",
        outro: "Unsicher, welches Viertel oder welche Größe? Schreib uns — wir helfen und bauen eine kostenlose Vorschau.",
      },
      pl: {
        title: "Druk 3D na zamówienie: mapy z wysyłką na Ukrainę i do UE",
        description:
          "Drukowane w 3D mapy, breloki i panele ścienne na zamówienie, z wysyłką do każdego miasta: Kijów, Lwów, Odessa, Winnica i 15 krajów UE. Eco PLA, 1–3 dni robocze, od ≈3 €.",
        h1: "Druk 3D na zamówienie z wysyłką na Ukrainę i do UE",
        intro:
          "Szukasz druku 3D na zamówienie w swoim mieście? Specjalizujemy się w jednym typie wyrobów — spersonalizowanych mapach 3D: mapy dzielnic, breloki z trasami, panele ścienne i magnesy. Drukujemy we własnej pracowni i wysyłamy do każdego miasta Ukrainy w 1–3 dni robocze oraz do 15 krajów UE.",
        sections: [
          {
            h2: "Co drukujemy",
            p: [
              "Mapy miast 3D (od ≈6 €): wybierasz dowolną dzielnicę — ulice, budynki o prawdziwych wysokościach, parki i rzeki drukują się jako model 5,5–15 cm. Breloki z mapą lub trasą GPX (od ≈3 €): zawieszka 55×30 mm z Twoją dzielnicą i napisem. Panele ścienne z kilku kafelków. Magnesy z dzielnicą miasta.",
              "Nie drukujemy cudzych plików STL, części ani figurek — tylko mapy. Za to mapy robimy lepiej niż ktokolwiek: własny kreator buduje model z danych OpenStreetMap w kilka minut, a Ty widzisz podgląd 3D przed płatnością.",
            ],
          },
          {
            h2: "Jak to działa dla dowolnego miasta",
            p: [
              "Kreator działa z dowolnym punktem Ukrainy i świata. Wybierasz obszar na mapie, ustawiasz rozmiar i styl, klikasz «Utwórz» — w 2–4 minuty model jest gotowy.",
              "Dalej dwie drogi: zamawiasz druk u nas (Eco PLA, wykonanie 1–3 dni robocze, wysyłka) — albo pobierasz plik 3MF/STL i drukujesz na własnej drukarce. Plik otwiera się w Bambu Studio i PrusaSlicer bez żadnego przygotowania.",
            ],
          },
          {
            h2: "Ceny",
            p: [
              "Brelok z mapą lub śladem GPX — od ≈3 €. Magnes na lodówkę — ≈4 €. Mapa miasta 3D: S (5,5 cm) ≈6 €, M (8 cm) ≈8 €, L (11 cm) ≈11 €, XL (15 cm) ≈13 €. Rzeźba terenu +≈1,5 €. Panele z kafelków wyceniane za kafelek, liczone na bieżąco w kreatorze.",
            ],
          },
          {
            h2: "Dlaczego nie zwykłe studio druku",
            p: [
              "Klasyczne studia druku 3D przyjmują każde zlecenie i wymagają gotowego pliku. U nas odwrotnie: plik nie jest potrzebny — kreator sam buduje model Twojej dzielnicy. To oznacza zero opłat za modelowanie, przewidywalną cenę i podgląd przed zamówieniem.",
            ],
          },
        ],
        ctaLabel: "Stwórz swój model",
        ctaHref: "/create",
        outro: "Nie wiesz, jaką dzielnicę czy rozmiar wybrać? Napisz do nas — pomożemy i przygotujemy darmowy podgląd.",
      },
      fr: {
        title: "Impression 3D sur mesure : cartes livrées en Ukraine et dans l'UE",
        description:
          "Cartes, porte-clés et panneaux muraux imprimés en 3D sur mesure, livrés dans toute ville : Kyiv, Lviv, Odessa, Vinnytsia et 15 pays de l'UE. Eco PLA, 1–3 jours ouvrés, dès ≈3 €.",
        h1: "Impression 3D sur mesure avec livraison en Ukraine et dans l'UE",
        intro:
          "Vous cherchez de l'impression 3D sur mesure dans votre ville ? Nous sommes spécialisés dans un seul type de produit — les cartes 3D personnelles : cartes de quartier, porte-clés d'itinéraire, panneaux muraux et magnets. Nous imprimons dans notre propre atelier et livrons dans toute ville d'Ukraine en 1–3 jours ouvrés, ainsi que dans 15 pays de l'UE.",
        sections: [
          {
            h2: "Ce que nous imprimons",
            p: [
              "Cartes de ville 3D (dès ≈6 €) : vous choisissez n'importe quel quartier — rues, bâtiments aux hauteurs réelles, parcs et rivières s'impriment en modèle de 5,5–15 cm. Porte-clés avec carte ou trace GPX (dès ≈3 €) : une plaque de 55×30 mm avec votre quartier et un texte. Panneaux muraux en plusieurs tuiles. Magnets avec un quartier de ville.",
              "Nous n'imprimons pas de fichiers STL tiers, de pièces ou de figurines — uniquement des cartes. En revanche, les cartes, nous les faisons mieux que quiconque : notre configurateur assemble le modèle à partir des données OpenStreetMap en quelques minutes, et vous voyez l'aperçu 3D avant de payer.",
            ],
          },
          {
            h2: "Comment ça marche pour n'importe quelle ville",
            p: [
              "Le configurateur fonctionne avec n'importe quel point d'Ukraine et du monde. Choisissez une zone sur la carte, réglez la taille et le style, cliquez sur «Créer» — le modèle est prêt en 2–4 minutes.",
              "Ensuite deux voies : vous commandez l'impression chez nous (Eco PLA, fabrication 1–3 jours ouvrés, livraison) — ou vous téléchargez le fichier 3MF/STL et imprimez sur votre propre machine. Le fichier s'ouvre dans Bambu Studio et PrusaSlicer sans aucune préparation.",
            ],
          },
          {
            h2: "Prix",
            p: [
              "Porte-clés avec carte ou GPX — dès ≈3 €. Magnet de frigo — ≈4 €. Carte de ville 3D : S (5,5 cm) ≈6 €, M (8 cm) ≈8 €, L (11 cm) ≈11 €, XL (15 cm) ≈13 €. Relief du terrain +≈1,5 €. Les panneaux en tuiles sont facturés à la tuile, calculés en direct dans le configurateur.",
            ],
          },
          {
            h2: "Pourquoi pas un studio d'impression classique",
            p: [
              "Les studios d'impression 3D classiques acceptent tous les travaux et exigent un fichier prêt. Chez nous, c'est l'inverse : pas besoin de fichier — le configurateur construit lui-même le modèle de votre quartier. Cela signifie zéro frais de modélisation, un prix prévisible et un aperçu avant commande.",
            ],
          },
        ],
        ctaLabel: "Créer votre modèle",
        ctaHref: "/create",
        outro: "Vous hésitez sur le quartier ou la taille ? Écrivez-nous — on vous aide et on prépare un aperçu gratuit.",
      },
      es: {
        title: "Impresión 3D a medida: mapas con envío a Ucrania y la UE",
        description:
          "Mapas, llaveros y paneles de pared impresos en 3D a medida, con envío a cualquier ciudad: Kyiv, Leópolis, Odesa, Vinnytsia y 15 países de la UE. Eco PLA, 1–3 días hábiles, desde ≈3 €.",
        h1: "Impresión 3D a medida con envío a Ucrania y la UE",
        intro:
          "¿Buscas impresión 3D a medida en tu ciudad? Nos especializamos en un solo tipo de producto — mapas 3D personales: mapas de barrio, llaveros con rutas, paneles de pared e imanes. Imprimimos en nuestro propio taller y enviamos a cualquier ciudad de Ucrania en 1–3 días hábiles, además de a 15 países de la UE.",
        sections: [
          {
            h2: "Qué imprimimos",
            p: [
              "Mapas de ciudad 3D (desde ≈6 €): eliges cualquier barrio — calles, edificios con alturas reales, parques y ríos se imprimen como modelo de 5,5–15 cm. Llaveros con mapa o ruta GPX (desde ≈3 €): una placa de 55×30 mm con tu barrio y un texto. Paneles de pared de varios azulejos. Imanes con un barrio de la ciudad.",
              "No imprimimos archivos STL ajenos, piezas ni figuras — solo mapas. Pero los mapas los hacemos mejor que nadie: nuestro configurador construye el modelo a partir de datos de OpenStreetMap en minutos, y ves la vista previa 3D antes de pagar.",
            ],
          },
          {
            h2: "Cómo funciona para cualquier ciudad",
            p: [
              "El configurador funciona con cualquier punto de Ucrania y del mundo. Eliges una zona en el mapa, ajustas tamaño y estilo, pulsas «Crear» — en 2–4 minutos el modelo está listo.",
              "Después, dos caminos: pides la impresión con nosotros (Eco PLA, fabricación 1–3 días hábiles, envío) — o descargas el archivo 3MF/STL e imprimes en tu propia máquina. El archivo se abre en Bambu Studio y PrusaSlicer sin ninguna preparación.",
            ],
          },
          {
            h2: "Precios",
            p: [
              "Llavero con mapa o GPX — desde ≈3 €. Imán de nevera — ≈4 €. Mapa de ciudad 3D: S (5,5 cm) ≈6 €, M (8 cm) ≈8 €, L (11 cm) ≈11 €, XL (15 cm) ≈13 €. Relieve del terreno +≈1,5 €. Los paneles de azulejos se cobran por azulejo, calculado al momento en el configurador.",
            ],
          },
          {
            h2: "Por qué no un estudio de impresión normal",
            p: [
              "Los estudios de impresión 3D clásicos aceptan cualquier encargo y necesitan un archivo listo. Con nosotros es al revés: no hace falta archivo — el configurador construye él mismo el modelo de tu barrio. Eso significa cero costes de modelado, precio predecible y vista previa antes de pedir.",
            ],
          },
        ],
        ctaLabel: "Crear tu modelo",
        ctaHref: "/create",
        outro: "¿No sabes qué barrio o tamaño elegir? Escríbenos — te ayudamos y preparamos una vista previa gratis.",
      },
    },
  },
  {
    slug: "podarunok-viyskovomu",
    date: "2026-07-16",
    content: {
      uk: {
        title: "Подарунок військовому: мапа місця, за яке він стоїть",
        description:
          "Ідея подарунка військовому — чоловіку, хлопцю, побратиму: 3D-мапа рідного міста чи брелок з домом, який завжди в кишені. Особисте, не банальне, від 120 ₴.",
        h1: "Подарунок військовому: шматочок дому, який можна тримати в руках",
        intro:
          "Браслети виживання й термокружки вже подаровані по три рази. Якщо шукаєте подарунок військовому — чоловіку, синові, побратиму — подумайте не про спорядження, а про те, чого на службі бракує найбільше: дім. 3D-мапа рідного району чи брелок з вулицею, де на нього чекають — маленька фізична річ, яка нагадує, за що все це.",
        sections: [
          {
            h2: "Чому мапа, а не ще один тактичний аксесуар",
            p: [
              "Спорядження купують за списком, а подарунок має говорити. Брелок 55×30 мм з рельєфом рідного двору поміщається в кишеню форми й важить кілька грамів. На звороті — напис: ім'я, дата, «чекаємо вдома», координати.",
              "Для дому теж працює: мапа міста, яке він захищає, на полиці в родини — з написом чи датою. Такий подарунок однаково сильний в обидва боки.",
            ],
          },
          {
            h2: "Що обирають найчастіше",
            p: [
              "Брелок з рідним районом (від 120 ₴) — найпрактичніше: легкий, міцний Eco PLA, витримує щоденне носіння. Мапа рідного міста 8–11 см (від 350 ₴) — на полицю чи в бліндаж. Брелоки-«серця» для пари — половинка з її районом, половинка з його: з'єднуються як пазл.",
            ],
          },
          {
            h2: "Практичні деталі",
            p: [
              "Виготовлення 1–3 робочі дні, Нова Пошта доставляє й на фронтові напрямки — вкажіть відділення, яке працює. Пластик легкий і не дзвенить, гострих країв немає. Якщо не знаєте точну адресу дитинства — досить назви району чи школи, допоможемо знайти ділянку.",
            ],
          },
        ],
        ctaLabel: "Створити мапу його дому",
        ctaHref: "/create",
        outro: "Не впевнені з районом? Напишіть нам — підберемо ділянку разом і покажемо превʼю до замовлення.",
      },
      en: {
        title: "A gift for a soldier: the map of the place they stand for",
        description:
          "A gift idea for a serviceman — husband, boyfriend, brother-in-arms: a 3D map of the home city or a keychain with the home street, always in a pocket. From ≈€3.",
        h1: "A gift for a soldier: a piece of home you can hold",
        intro:
          "Survival bracelets and thermal mugs have been gifted three times over. If you're looking for a gift for a serviceman — husband, son, brother-in-arms — think not about gear but about what's scarcest on duty: home. A 3D map of the home district or a keychain with the street where they're awaited — a small physical thing that reminds what it's all for.",
        sections: [
          {
            h2: "Why a map and not another tactical accessory",
            p: [
              "Gear is bought from a checklist; a gift should speak. A 55×30 mm keychain with the relief of the home yard fits a uniform pocket and weighs a few grams. On the back — an inscription: a name, a date, «waiting at home», coordinates.",
              "It works both ways: a map of the city they defend, on the family's shelf at home — with a date or inscription.",
            ],
          },
          {
            h2: "What people choose most",
            p: [
              "A keychain with the home district (from ≈€3) — the most practical: light, sturdy Eco PLA, survives daily carry. A home-city map 8–11 cm (from ≈€8) — for a shelf. Heart-pair keychains — one half with her district, one with his: they connect like a puzzle.",
            ],
          },
          {
            h2: "Practical details",
            p: [
              "Made in 1–3 business days; Nova Poshta delivers to most areas — specify a working branch. The plastic is light, doesn't jingle, no sharp edges. If you don't know the exact childhood address — a district or school name is enough, we'll help find the area.",
            ],
          },
        ],
        ctaLabel: "Create the map of their home",
        ctaHref: "/create",
        outro: "Not sure about the district? Message us — we'll pick the area together and show a preview before you order.",
      },
      de: {
        title: "Ein Geschenk für einen Soldaten: die Karte des Ortes, für den er steht",
        description:
          "Geschenkidee für einen Militärangehörigen — Mann, Freund, Kamerad: eine 3D-Karte der Heimatstadt oder ein Anhänger mit dem Zuhause, das immer in der Tasche ist. Persönlich, nicht banal, ab ≈3 €.",
        h1: "Ein Geschenk für einen Soldaten: ein Stück Zuhause zum Anfassen",
        intro:
          "Survival-Armbänder und Thermobecher sind schon dreimal verschenkt. Wenn du ein Geschenk für einen Militärangehörigen suchst — Mann, Sohn, Kameraden — denk nicht an Ausrüstung, sondern an das, was im Dienst am meisten fehlt: Zuhause. Eine 3D-Karte des Heimatviertels oder ein Anhänger mit der Straße, in der man auf ihn wartet — ein kleines physisches Ding, das daran erinnert, wofür das alles ist.",
        sections: [
          {
            h2: "Warum eine Karte und nicht noch ein taktisches Accessoire",
            p: [
              "Ausrüstung kauft man nach Liste, ein Geschenk soll sprechen. Ein Anhänger 55×30 mm mit dem Relief des heimischen Hofs passt in die Uniformtasche und wiegt ein paar Gramm. Auf der Rückseite eine Gravur: Name, Datum, «wir warten zu Hause», Koordinaten.",
              "Für zu Hause funktioniert es genauso: eine Karte der Stadt, die er verteidigt, im Regal der Familie — mit Text oder Datum. So ein Geschenk wirkt in beide Richtungen gleich stark.",
            ],
          },
          {
            h2: "Was am häufigsten gewählt wird",
            p: [
              "Ein Anhänger mit dem Heimatviertel (ab ≈3 €) — am praktischsten: leichtes, robustes Eco PLA, hält den täglichen Gebrauch aus. Eine Karte der Heimatstadt 8–11 cm (ab ≈8 €) — fürs Regal. «Herz»-Anhänger für ein Paar — eine Hälfte mit ihrem Viertel, eine mit seinem: sie fügen sich wie ein Puzzle.",
            ],
          },
          {
            h2: "Praktische Details",
            p: [
              "Fertigung 1–3 Werktage; Nova Poshta liefert in die meisten Gebiete — gib eine funktionierende Filiale an. Der Kunststoff ist leicht, klappert nicht, keine scharfen Kanten. Wenn du die genaue Adresse der Kindheit nicht kennst — ein Viertel- oder Schulname reicht, wir helfen beim Finden.",
            ],
          },
        ],
        ctaLabel: "Die Karte seines Zuhauses erstellen",
        ctaHref: "/create",
        outro: "Unsicher beim Viertel? Schreib uns — wir wählen den Bereich gemeinsam und zeigen eine Vorschau vor der Bestellung.",
      },
      pl: {
        title: "Prezent dla żołnierza: mapa miejsca, za które stoi",
        description:
          "Pomysł na prezent dla wojskowego — męża, chłopaka, pobratymca: mapa 3D rodzinnego miasta lub brelok z domem, który zawsze jest w kieszeni. Osobisty, nie banalny, od ≈3 €.",
        h1: "Prezent dla żołnierza: kawałek domu, który można trzymać w rękach",
        intro:
          "Bransoletki survivalowe i kubki termiczne zostały już podarowane po trzy razy. Jeśli szukasz prezentu dla wojskowego — męża, syna, pobratymca — pomyśl nie o wyposażeniu, ale o tym, czego na służbie brakuje najbardziej: o domu. Mapa 3D rodzinnej dzielnicy albo brelok z ulicą, przy której na niego czekają — mała fizyczna rzecz, która przypomina, po co to wszystko.",
        sections: [
          {
            h2: "Dlaczego mapa, a nie kolejny taktyczny gadżet",
            p: [
              "Wyposażenie kupuje się z listy, a prezent ma mówić. Brelok 55×30 mm z reliefem rodzinnego podwórka mieści się w kieszeni munduru i waży kilka gramów. Z tyłu napis: imię, data, «czekamy w domu», współrzędne.",
              "Dla domu działa tak samo: mapa miasta, którego broni, na półce rodziny — z napisem lub datą. Taki prezent jest równie mocny w obie strony.",
            ],
          },
          {
            h2: "Co wybierają najczęściej",
            p: [
              "Brelok z rodzinną dzielnicą (od ≈3 €) — najbardziej praktyczny: lekki, wytrzymały Eco PLA, znosi codzienne noszenie. Mapa rodzinnego miasta 8–11 cm (od ≈8 €) — na półkę. Breloki-«serca» dla pary — połówka z jej dzielnicą, połówka z jego: łączą się jak puzzle.",
            ],
          },
          {
            h2: "Praktyczne szczegóły",
            p: [
              "Wykonanie 1–3 dni robocze; Nova Poshta dowozi na większość kierunków — podaj działające oddziału. Plastik jest lekki i nie brzęczy, nie ma ostrych krawędzi. Jeśli nie znasz dokładnego adresu z dzieciństwa — wystarczy nazwa dzielnicy lub szkoły, pomożemy znaleźć obszar.",
            ],
          },
        ],
        ctaLabel: "Stwórz mapę jego domu",
        ctaHref: "/create",
        outro: "Nie masz pewności co do dzielnicy? Napisz do nas — dobierzemy obszar razem i pokażemy podgląd przed zamówieniem.",
      },
      fr: {
        title: "Un cadeau pour un soldat : la carte du lieu qu'il défend",
        description:
          "Idée de cadeau pour un militaire — mari, petit ami, frère d'armes : une carte 3D de sa ville natale ou un porte-clés avec sa maison, toujours dans la poche. Personnel, pas banal, dès ≈3 €.",
        h1: "Un cadeau pour un soldat : un morceau de chez soi qu'on peut tenir",
        intro:
          "Les bracelets de survie et les mugs thermos ont déjà été offerts trois fois. Si vous cherchez un cadeau pour un militaire — mari, fils, frère d'armes — pensez non pas à l'équipement, mais à ce qui manque le plus au service : la maison. Une carte 3D du quartier natal ou un porte-clés avec la rue où on l'attend — une petite chose physique qui rappelle pourquoi tout cela.",
        sections: [
          {
            h2: "Pourquoi une carte et pas un énième accessoire tactique",
            p: [
              "L'équipement s'achète sur liste ; un cadeau doit parler. Un porte-clés de 55×30 mm avec le relief de la cour natale tient dans une poche d'uniforme et pèse quelques grammes. Au dos, une gravure : un prénom, une date, «on t'attend à la maison», des coordonnées.",
              "Pour la maison, ça marche aussi : la carte de la ville qu'il défend, sur l'étagère de la famille — avec un texte ou une date. Un tel cadeau est aussi fort dans les deux sens.",
            ],
          },
          {
            h2: "Ce qu'on choisit le plus souvent",
            p: [
              "Un porte-clés avec le quartier natal (dès ≈3 €) — le plus pratique : Eco PLA léger et solide, supporte le port quotidien. Une carte de la ville natale 8–11 cm (dès ≈8 €) — pour l'étagère. Des porte-clés «cœur» pour un couple — une moitié avec son quartier à elle, l'autre avec le sien : ils s'emboîtent comme un puzzle.",
            ],
          },
          {
            h2: "Détails pratiques",
            p: [
              "Fabrication 1–3 jours ouvrés ; Nova Poshta livre la plupart des directions — indiquez une agence en service. Le plastique est léger et ne cliquette pas, sans arêtes vives. Si vous ne connaissez pas l'adresse exacte de l'enfance — un nom de quartier ou d'école suffit, nous vous aidons à trouver la zone.",
            ],
          },
        ],
        ctaLabel: "Créer la carte de sa maison",
        ctaHref: "/create",
        outro: "Vous hésitez sur le quartier ? Écrivez-nous — on choisit la zone ensemble et on montre un aperçu avant la commande.",
      },
      es: {
        title: "Un regalo para un soldado: el mapa del lugar por el que está de pie",
        description:
          "Idea de regalo para un militar — marido, novio, compañero de armas: un mapa 3D de su ciudad natal o un llavero con su casa, siempre en el bolsillo. Personal, no banal, desde ≈3 €.",
        h1: "Un regalo para un soldado: un trozo de casa que se puede sostener",
        intro:
          "Las pulseras de supervivencia y las tazas térmicas ya se han regalado tres veces. Si buscas un regalo para un militar — marido, hijo, compañero de armas — piensa no en el equipo, sino en lo que más falta en el servicio: la casa. Un mapa 3D del barrio natal o un llavero con la calle donde lo esperan — una pequeña cosa física que recuerda para qué es todo esto.",
        sections: [
          {
            h2: "Por qué un mapa y no otro accesorio táctico",
            p: [
              "El equipo se compra por lista; un regalo debe hablar. Un llavero de 55×30 mm con el relieve del patio natal cabe en el bolsillo del uniforme y pesa unos gramos. Al dorso, un grabado: un nombre, una fecha, «te esperamos en casa», coordenadas.",
              "Para la casa funciona igual: el mapa de la ciudad que defiende, en la estantería de la familia — con un texto o una fecha. Un regalo así es igual de fuerte en ambos sentidos.",
            ],
          },
          {
            h2: "Qué eligen más a menudo",
            p: [
              "Un llavero con el barrio natal (desde ≈3 €) — lo más práctico: Eco PLA ligero y resistente, aguanta el uso diario. Un mapa de la ciudad natal de 8–11 cm (desde ≈8 €) — para la estantería. Llaveros «corazón» para una pareja — una mitad con el barrio de ella, otra con el de él: encajan como un puzle.",
            ],
          },
          {
            h2: "Detalles prácticos",
            p: [
              "Fabricación 1–3 días hábiles; Nova Poshta llega a la mayoría de destinos — indica una sucursal operativa. El plástico es ligero y no tintinea, sin bordes afilados. Si no sabes la dirección exacta de la infancia — basta el nombre del barrio o de la escuela, te ayudamos a encontrar la zona.",
            ],
          },
        ],
        ctaLabel: "Crear el mapa de su casa",
        ctaHref: "/create",
        outro: "¿Dudas con el barrio? Escríbenos — elegimos la zona juntos y mostramos una vista previa antes de pedir.",
      },
    },
  },
  {
    slug: "podarunok-bihunu",
    date: "2026-07-16",
    content: {
      uk: {
        title: "Подарунок бігуну: його маршрут, надрукований у пластику",
        description:
          "Що подарувати бігуну, в якого вже все є: брелок з GPX-треком першого марафону чи улюбленого кола. Завантажуємо трек зі Strava — друкуємо рельєфом. Від 120 ₴.",
        h1: "Подарунок бігуну, який неможливо купити готовим",
        intro:
          "У бігуна вже є годинник, гелі, пояс і треті кросівки. Але є річ, якої немає в жодному магазині: його власний маршрут. Перший марафон, ранкове коло парком, стометрівка набережною — усе це лежить треками у Strava. Ми перетворюємо трек на фізичну річ: рельєфна лінія маршруту поверх карти району на брелку чи мапі.",
        sections: [
          {
            h2: "Як це виглядає",
            p: [
              "Жетон 55×30 мм: вулиці району друкуються тонким рельєфом, а лінія маршруту — виразніше, поверх. На звороті — напис: «Kyiv Half 2026 · 1:47» або просто дата й дистанція. Маршрут прив'язується до доріг, тож навіть «шумний» GPS-запис виглядає акуратно.",
            ],
          },
          {
            h2: "Приводи, які працюють",
            p: [
              "Фініш першого марафону чи півмарафону — класика: цифри часу на звороті перетворюють брелок на медаль, яка завжди з собою. Ювілейний забіг клубу, перша сотка велосипедом, навіть маршрут, яким пара бігала на побаченнях — усе, що записано треком, можна надрукувати.",
              "Для бігового клубу чи корпоративного забігу робимо серії: однаковий маршрут, різні імена й часи на звороті.",
            ],
          },
          {
            h2: "Як замовити",
            p: [
              "Експортуйте GPX зі Strava, Garmin Connect чи Komoot (у Strava: активність → три крапки → «Експорт GPX»). Завантажте файл у конструктор — сервіс сам знайде місце, підбере масштаб і покаже превʼю. Від 120 ₴, виготовлення 1–3 робочі дні.",
            ],
          },
        ],
        ctaLabel: "Завантажити GPX-трек",
        ctaHref: "/keychains",
        outro: "Даруєте сюрпризом і не маєте доступу до треку? Підійде скріншот маршруту — намалюємо ділянку за ним.",
      },
      en: {
        title: "A gift for a runner: their route, printed in plastic",
        description:
          "What to give a runner who has everything: a keychain with the GPX track of their first marathon or favourite loop. Upload from Strava — we print it in relief. From ≈€3.",
        h1: "A gift for a runner that can't be bought off the shelf",
        intro:
          "A runner already owns the watch, the gels, the belt and a third pair of shoes. But there's one thing no store has: their own route. The first marathon, the morning park loop — it all sits as tracks in Strava. We turn a track into a physical object: a relief route line over the district map on a keychain or a full map.",
        sections: [
          {
            h2: "What it looks like",
            p: [
              "A 55×30 mm tag: the district's streets print in fine relief, the route line stands out on top. On the back — «Kyiv Half 2026 · 1:47» or just the date and distance. The route snaps to roads, so even noisy GPS looks clean.",
            ],
          },
          {
            h2: "Occasions that work",
            p: [
              "A first marathon or half finish is the classic: the time on the back turns the keychain into a medal that's always with you. A club's anniversary run, a first century ride, even the route a couple used to run on dates — anything recorded as a track can be printed.",
              "For running clubs and corporate races we make series: same route, different names and times on the back.",
            ],
          },
          {
            h2: "How to order",
            p: [
              "Export a GPX from Strava, Garmin Connect or Komoot (in Strava: activity → three dots → «Export GPX»). Upload it in the builder — the service locates the route, scales the map and shows a preview. From ≈€3, made in 1–3 business days.",
            ],
          },
        ],
        ctaLabel: "Upload a GPX track",
        ctaHref: "/keychains",
        outro: "Gifting as a surprise without access to the track? A route screenshot works — we'll trace the area from it.",
      },
    },
  },
  {
    slug: "podarunok-pereselentsyu",
    date: "2026-07-16",
    content: {
      uk: {
        title: "Подарунок переселенцю: рідне місто, яке завжди поруч",
        description:
          "Що подарувати людині, яка виїхала з рідного міста: 3D-мапа рідного району чи брелок з домом. Подарунок переселенцю, друзям за кордоном, рідним. Від 120 ₴, доставка в ЄС.",
        h1: "Подарунок переселенцю: шматочок рідного міста",
        intro:
          "Мільйони людей зараз живуть не там, де виросли. Комусь довелось виїхати від війни, хтось переїхав за роботою чи навчанням — але рідний двір пам'ятають усі. 3D-мапа рідного району — подарунок, який працює сильніше за будь-які слова: ось твоя вулиця, твоя школа, твій парк. Вони на місці. Вони чекають.",
        sections: [
          {
            h2: "Чому це влучає",
            p: [
              "Людині далеко від дому не бракує речей — їй бракує місця. Фотографії лишаються в телефоні, а мапа стоїть на полиці нової квартири в Варшаві, Берліні чи Празі й щодня нагадує: дім існує. Це подарунок і на день народження, і «просто так», і на новосілля на новому місці.",
              "Особливо сильно працює для міст, куди зараз не поїхати: Маріуполь, Донецьк, Луганськ, Херсонщина. OpenStreetMap пам'ятає ці вулиці — і ми можемо їх надрукувати.",
            ],
          },
          {
            h2: "Формати",
            p: [
              "Мапа рідного району 8–15 см (від 350 ₴) — на полицю. Брелок з двором дитинства (від 120 ₴) — щоб дім був у кишені. Магніт (150 ₴) — на холодильник нової кухні. Панно з кількох плиток — коли хочеться повісити на стіну ціле місто.",
              "На звороті брелока — напис: назва міста, «додому повернемось», координати дому чи ім'я.",
            ],
          },
          {
            h2: "Доставка за кордон",
            p: [
              "Надсилаємо Новою Поштою по Україні та Nova Post / Meest у 15 країн ЄС — Польщу, Німеччину, Чехію та інші. Виготовлення 1–3 робочі дні. Замовити можна з будь-якої країни, оплата карткою онлайн.",
            ],
          },
        ],
        ctaLabel: "Створити мапу рідного міста",
        ctaHref: "/create",
        outro: "Якщо рідне місто зараз окуповане чи зруйноване — мапа друкується за довоєнними даними OpenStreetMap. Таким, яким його пам'ятають.",
      },
      en: {
        title: "A gift for someone far from home: their hometown, always near",
        description:
          "What to give a person who left their home city: a 3D map of the home district or a keychain with the home street. For refugees, friends abroad, family. From ≈€3, EU delivery.",
        h1: "A gift for someone far from home: a piece of their city",
        intro:
          "Millions of people now live away from where they grew up — displaced by war, moved for work or study. But everyone remembers their home yard. A 3D map of the home district is a gift that speaks louder than words: here is your street, your school, your park. They're still there. They're waiting.",
        sections: [
          {
            h2: "Why it lands",
            p: [
              "A person far from home doesn't lack things — they lack a place. Photos stay in the phone, but a map stands on the shelf of a new flat in Warsaw, Berlin or Prague and reminds daily: home exists. It works for birthdays, housewarmings in a new country, or no occasion at all.",
              "It's especially powerful for cities one can't visit now: Mariupol, Donetsk, Luhansk. OpenStreetMap remembers those streets — and we can print them.",
            ],
          },
          {
            h2: "Formats",
            p: [
              "A home-district map 8–15 cm (from ≈€8) for a shelf. A childhood-yard keychain (from ≈€3) so home fits in a pocket. A fridge magnet (≈€4) for the new kitchen. A multi-tile panel when you want a whole city on the wall.",
              "On the keychain's back — an inscription: the city name, coordinates of home, or a name.",
            ],
          },
          {
            h2: "Delivery abroad",
            p: [
              "We ship across Ukraine and via Nova Post / Meest to 15 EU countries — Poland, Germany, Czechia and more. Production 1–3 business days. Order from any country, card payment online.",
            ],
          },
        ],
        ctaLabel: "Create the map of a home city",
        ctaHref: "/create",
        outro: "If the home city is occupied or damaged — the map prints from pre-war OpenStreetMap data. The way it's remembered.",
      },
    },
  },
];

export const BLOG_BY_SLUG: Record<string, BlogArticle> = Object.fromEntries(
  BLOG_ARTICLES.map((a) => [a.slug, a]),
);
