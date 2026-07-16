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
    },
  },
];

export const BLOG_BY_SLUG: Record<string, BlogArticle> = Object.fromEntries(
  BLOG_ARTICLES.map((a) => [a.slug, a]),
);
