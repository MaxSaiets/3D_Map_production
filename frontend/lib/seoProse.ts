// SEO-проза для builder-сторінок (/create, /keychains, /worlds): це client-side
// конструктори майже без індексованого тексту, хоча саме вони таргетять грошові
// запити. Серверний текст-блок рендериться ПІД конструктором (нижче згину) —
// краулер бачить контент, користувач UI не втрачає.
//
// ТЕПЕР повні 6 локалей (uk/en/de/pl/fr/es). Селектор бере локаль, якщо є, інакше
// м'який фолбек на en — тож додавання/пропуск локалі ніколи не ламає рендер.

export type SeoProse = { h2: string; p1: string; p2: string };

const CREATE: Record<string, SeoProse> = {
  uk: {
    h2: "Онлайн-конструктор 3D-мапи міста",
    p1: "Monadruk перетворює будь-яку точку світу на друковану 3D-модель: оберіть район на карті — і за кілька хвилин отримаєте тривимірну мапу з реальними висотами будинків, вулицями, парками й річками за даними OpenStreetMap. Для горбистих міст можна ввімкнути рельєф місцевості, а серію сусідніх плиток — з'єднати у настінне панно.",
    p2: "Готову модель друкуємо з екологічного біопластику Eco PLA у розмірах від 5,5 до 15 см (ціна від 250 ₴) і надсилаємо Новою Поштою по Україні. Якщо у вас є власний 3D-принтер — завантажте готовий файл 3MF/STL і надрукуйте вдома.",
  },
  en: {
    h2: "Online 3D city map builder",
    p1: "Monadruk turns any point on Earth into a printable 3D model: pick a district on the map and in minutes get a three-dimensional map with real building heights, streets, parks and rivers from OpenStreetMap data. Hilly cities can be rendered with true terrain relief, and adjacent tiles can be joined into a wall panel.",
    p2: "We print the finished model in eco-friendly Eco PLA in sizes from 5.5 to 15 cm (from ≈€6) and ship across Ukraine. Have your own 3D printer? Download the ready 3MF/STL file and print at home.",
  },
  de: {
    h2: "Online-Konfigurator für 3D-Stadtkarten",
    p1: "Monadruk verwandelt jeden Punkt der Erde in ein druckbares 3D-Modell: Wähle ein Viertel auf der Karte und erhalte in wenigen Minuten eine dreidimensionale Karte mit echten Gebäudehöhen, Straßen, Parks und Flüssen aus OpenStreetMap-Daten. Für hügelige Städte lässt sich das Geländerelief aktivieren, und benachbarte Kacheln können zu einem Wandpanel zusammengefügt werden.",
    p2: "Das fertige Modell drucken wir aus umweltfreundlichem Eco PLA in Größen von 5,5 bis 15 cm (ab ≈6 €) und versenden es innerhalb der Ukraine. Du hast einen eigenen 3D-Drucker? Lade die fertige 3MF/STL-Datei herunter und drucke zu Hause.",
  },
  pl: {
    h2: "Kreator mapy miasta 3D online",
    p1: "Monadruk zamienia dowolny punkt na Ziemi w drukowalny model 3D: wybierz dzielnicę na mapie, a w kilka minut otrzymasz trójwymiarową mapę z rzeczywistymi wysokościami budynków, ulicami, parkami i rzekami na podstawie danych OpenStreetMap. Dla pagórkowatych miast można włączyć rzeźbę terenu, a sąsiednie kafle połączyć w panel ścienny.",
    p2: "Gotowy model drukujemy z ekologicznego bioplastiku Eco PLA w rozmiarach od 5,5 do 15 cm (od ≈6 €) i wysyłamy na Ukrainę. Masz własną drukarkę 3D? Pobierz gotowy plik 3MF/STL i wydrukuj w domu.",
  },
  fr: {
    h2: "Configurateur de carte de ville 3D en ligne",
    p1: "Monadruk transforme n'importe quel point du globe en un modèle 3D imprimable : choisissez un quartier sur la carte et obtenez en quelques minutes une carte tridimensionnelle avec les hauteurs réelles des bâtiments, les rues, les parcs et les rivières d'après les données OpenStreetMap. Pour les villes vallonnées, le relief du terrain peut être activé, et des tuiles voisines assemblées en un panneau mural.",
    p2: "Nous imprimons le modèle fini en Eco PLA écologique, en tailles de 5,5 à 15 cm (à partir de ≈6 €), et l'expédions en Ukraine. Vous avez votre propre imprimante 3D ? Téléchargez le fichier 3MF/STL prêt et imprimez chez vous.",
  },
  es: {
    h2: "Configurador de mapa de ciudad 3D en línea",
    p1: "Monadruk convierte cualquier punto de la Tierra en un modelo 3D imprimible: elige un distrito en el mapa y en minutos obtén un mapa tridimensional con alturas reales de edificios, calles, parques y ríos a partir de datos de OpenStreetMap. Para ciudades con colinas se puede activar el relieve del terreno, y unir baldosas vecinas en un panel de pared.",
    p2: "Imprimimos el modelo terminado en Eco PLA ecológico, en tamaños de 5,5 a 15 cm (desde ≈6 €), y lo enviamos dentro de Ucrania. ¿Tienes tu propia impresora 3D? Descarga el archivo 3MF/STL listo e imprime en casa.",
  },
};

const KEYCHAINS: Record<string, SeoProse> = {
  uk: {
    h2: "Брелок з картою міста на замовлення",
    p1: "Брелок-мапа — це жетон 55×30 мм з рельєфною картою обраного району: вулиці, парки й річки, які можна відчути пальцями. Додайте власний напис на звороті — назву міста, дату чи координати. Є режим гірського рельєфу (топо-брелок) і брелок з вашим GPX-маршрутом зі Strava чи Garmin.",
    p2: "Друкуємо з Eco PLA за 1–3 робочі дні, ціна від 120 ₴. Доставка Новою Поштою по Україні. Пара брелоків-«сердець» з районами двох людей з'єднується як пазл — популярний подарунок для пар.",
  },
  en: {
    h2: "Custom city map keychain",
    p1: "The map keychain is a 55×30 mm tag with a relief map of your chosen district: streets, parks and rivers you can feel with your fingers. Add custom text on the back — a city name, a date or coordinates. There's a mountain-relief topo mode and a keychain with your GPX route from Strava or Garmin.",
    p2: "Printed in Eco PLA within 1–3 business days, from ≈€3. Shipping across Ukraine and to EU countries. A pair of «heart» keychains with two people's districts joins like a puzzle — a popular couple's gift.",
  },
  de: {
    h2: "Individueller Schlüsselanhänger mit Stadtkarte",
    p1: "Der Karten-Anhänger ist eine 55×30-mm-Plakette mit einer Reliefkarte deines gewählten Viertels: Straßen, Parks und Flüsse, die man mit den Fingern erfühlen kann. Füge auf der Rückseite eigenen Text hinzu — einen Stadtnamen, ein Datum oder Koordinaten. Es gibt einen Gebirgsrelief-Modus (Topo-Anhänger) und einen Anhänger mit deiner GPX-Route aus Strava oder Garmin.",
    p2: "Gedruckt aus Eco PLA in 1–3 Werktagen, ab ≈3 €. Versand innerhalb der Ukraine. Ein Paar «Herz»-Anhänger mit den Vierteln zweier Menschen fügt sich wie ein Puzzle zusammen — ein beliebtes Geschenk für Paare.",
  },
  pl: {
    h2: "Brelok z mapą miasta na zamówienie",
    p1: "Brelok-mapa to zawieszka 55×30 mm z reliefową mapą wybranej dzielnicy: ulice, parki i rzeki, które można poczuć palcami. Dodaj własny napis z tyłu — nazwę miasta, datę lub współrzędne. Jest tryb reliefu górskiego (brelok topo) i brelok z Twoją trasą GPX ze Strava lub Garmin.",
    p2: "Drukujemy z Eco PLA w 1–3 dni robocze, od ≈3 €. Wysyłka na Ukrainę. Para breloków-«serc» z dzielnicami dwóch osób łączy się jak puzzle — popularny prezent dla par.",
  },
  fr: {
    h2: "Porte-clés personnalisé avec carte de ville",
    p1: "Le porte-clés carte est une plaque de 55×30 mm avec une carte en relief du quartier choisi : rues, parcs et rivières que l'on peut sentir du bout des doigts. Ajoutez votre texte au dos — un nom de ville, une date ou des coordonnées. Il existe un mode relief de montagne (porte-clés topo) et un porte-clés avec votre parcours GPX de Strava ou Garmin.",
    p2: "Imprimé en Eco PLA en 1 à 3 jours ouvrés, à partir de ≈3 €. Livraison en Ukraine. Une paire de porte-clés «cœur» avec les quartiers de deux personnes s'assemble comme un puzzle — un cadeau populaire pour les couples.",
  },
  es: {
    h2: "Llavero personalizado con mapa de ciudad",
    p1: "El llavero mapa es una placa de 55×30 mm con un mapa en relieve del distrito elegido: calles, parques y ríos que se pueden sentir con los dedos. Añade tu propio texto en el reverso — el nombre de una ciudad, una fecha o coordenadas. Hay un modo de relieve montañoso (llavero topo) y un llavero con tu ruta GPX de Strava o Garmin.",
    p2: "Impreso en Eco PLA en 1–3 días hábiles, desde ≈3 €. Envío dentro de Ucrania. Un par de llaveros «corazón» con los distritos de dos personas encaja como un rompecabezas — un regalo popular para parejas.",
  },
};

const WORLDS: Record<string, SeoProse> = {
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
  de: {
    h2: "KI-Generator für fantastische 3D-Welten",
    p1: "Worlds ist ein experimentelles Monadruk-Werkzeug: Beschreibe eine Landschaft in Worten („Vulkaninsel“, „tiefer Canyon“, „sanfte Hügel“) und die KI erzeugt ein einzigartiges 3D-Geländemodell, das auf keiner echten Karte existiert. Anders als beim Stadtkarten-Konfigurator gibt es hier keine Bindung an OpenStreetMap — nur Fantasie.",
    p2: "Drehe das fertige Modell direkt im Browser und lade die GLB-Datei kostenlos herunter. Größen von 8 bis 18 cm. Um diese Welt gedruckt zu bestellen, schreib uns im Chat.",
  },
  pl: {
    h2: "Generator AI fantastycznych światów 3D",
    p1: "Worlds to eksperymentalne narzędzie Monadruk: opisz krajobraz słowami („wulkaniczna wyspa”, „głęboki kanion”, „łagodne wzgórza”), a AI wygeneruje unikalny model terenu 3D, który nie istnieje na żadnej prawdziwej mapie. W przeciwieństwie do kreatora map miast nie ma tu powiązania z OpenStreetMap — tylko wyobraźnia.",
    p2: "Gotowy model obróć od razu w przeglądarce i pobierz plik GLB za darmo. Rozmiary od 8 do 18 cm. Aby zamówić wydruk takiego świata, napisz do nas na czacie.",
  },
  fr: {
    h2: "Générateur IA de mondes 3D fantastiques",
    p1: "Worlds est un outil expérimental de Monadruk : décrivez un paysage en mots (« île volcanique », « canyon profond », « collines douces ») et l'IA génère un modèle de terrain 3D unique qui n'existe sur aucune carte réelle. Contrairement au configurateur de cartes de villes, il n'y a ici aucun lien avec OpenStreetMap — juste l'imagination.",
    p2: "Faites tourner le modèle fini directement dans le navigateur et téléchargez le fichier GLB gratuitement. Tailles de 8 à 18 cm. Pour commander l'impression de ce monde, écrivez-nous dans le chat.",
  },
  es: {
    h2: "Generador de IA de mundos 3D fantásticos",
    p1: "Worlds es una herramienta experimental de Monadruk: describe un paisaje con palabras («isla volcánica», «cañón profundo», «colinas suaves») y la IA genera un modelo de terreno 3D único que no existe en ningún mapa real. A diferencia del configurador de mapas de ciudades, aquí no hay vínculo con OpenStreetMap — solo imaginación.",
    p2: "Gira el modelo terminado directamente en el navegador y descarga el archivo GLB gratis. Tamaños de 8 a 18 cm. Para pedir la impresión de este mundo, escríbenos en el chat.",
  },
};

export function seoProse(page: "create" | "keychains" | "worlds", locale: string): SeoProse {
  const dict = page === "create" ? CREATE : page === "keychains" ? KEYCHAINS : WORLDS;
  return dict[locale] ?? dict.en;
}

export type ProseFaqItem = { q: string; a: string };

const CREATE_FAQ: Record<string, ProseFaqItem[]> = {
  uk: [
    { q: "Скільки коштує 3D-мапа?", a: "Від 250 ₴ за розмір S (5,5 см) до 550 ₴ за XL (15 см). Рельєф місцевості — опція +60 ₴." },
    { q: "Яку ділянку краще обрати?", a: "Ділянку 400–800 метрів зі змішаною забудовою: трохи вулиць, парк або вода — так район впізнається з першого погляду." },
    { q: "Скільки триває виготовлення?", a: "1–3 робочі дні на друк, потім доставка Новою Поштою по Україні." },
    { q: "Чи можна надрукувати самому?", a: "Так — завантажте готовий файл 3MF або STL, він одразу відкривається в Bambu Studio чи PrusaSlicer." },
  ],
  en: [
    { q: "How much does a 3D map cost?", a: "From ≈€6 for size S (5.5 cm) to ≈€13 for XL (15 cm). Terrain relief is an option, +≈€1.5." },
    { q: "Which area should I pick?", a: "A 400–800 m area with mixed content: some streets, a park or water — the district stays recognizable at first glance." },
    { q: "How long does production take?", a: "1–3 business days to print, then shipping across Ukraine." },
    { q: "Can I print it myself?", a: "Yes — download the ready 3MF or STL file, it opens directly in Bambu Studio or PrusaSlicer." },
  ],
  de: [
    { q: "Was kostet eine 3D-Karte?", a: "Von ≈6 € für Größe S (5,5 cm) bis ≈13 € für XL (15 cm). Geländerelief ist eine Option, +≈1,5 €." },
    { q: "Welchen Bereich soll ich wählen?", a: "Einen Bereich von 400–800 m mit gemischtem Inhalt: ein paar Straßen, ein Park oder Wasser — so bleibt das Viertel auf den ersten Blick erkennbar." },
    { q: "Wie lange dauert die Herstellung?", a: "1–3 Werktage Druck, dann Versand in die Ukraine." },
    { q: "Kann ich es selbst drucken?", a: "Ja — lade die fertige 3MF- oder STL-Datei herunter, sie öffnet direkt in Bambu Studio oder PrusaSlicer." },
  ],
  pl: [
    { q: "Ile kosztuje mapa 3D?", a: "Od ≈6 € za rozmiar S (5,5 cm) do ≈13 € za XL (15 cm). Rzeźba terenu to opcja, +≈1,5 €." },
    { q: "Jaki obszar najlepiej wybrać?", a: "Obszar 400–800 m z mieszaną zabudową: trochę ulic, park lub woda — dzielnica pozostaje rozpoznawalna od pierwszego spojrzenia." },
    { q: "Ile trwa wykonanie?", a: "1–3 dni robocze na druk, potem wysyłka na Ukrainę." },
    { q: "Czy mogę wydrukować sam?", a: "Tak — pobierz gotowy plik 3MF lub STL, otwiera się od razu w Bambu Studio lub PrusaSlicer." },
  ],
  fr: [
    { q: "Combien coûte une carte 3D ?", a: "De ≈6 € pour la taille S (5,5 cm) à ≈13 € pour XL (15 cm). Le relief du terrain est une option, +≈1,5 €." },
    { q: "Quelle zone choisir ?", a: "Une zone de 400 à 800 m au contenu varié : quelques rues, un parc ou de l'eau — le quartier reste reconnaissable au premier coup d'œil." },
    { q: "Combien de temps prend la fabrication ?", a: "1 à 3 jours ouvrés pour l'impression, puis livraison en Ukraine." },
    { q: "Puis-je l'imprimer moi-même ?", a: "Oui — téléchargez le fichier 3MF ou STL prêt, il s'ouvre directement dans Bambu Studio ou PrusaSlicer." },
  ],
  es: [
    { q: "¿Cuánto cuesta un mapa 3D?", a: "Desde ≈6 € para el tamaño S (5,5 cm) hasta ≈13 € para XL (15 cm). El relieve del terreno es una opción, +≈1,5 €." },
    { q: "¿Qué zona conviene elegir?", a: "Una zona de 400–800 m con contenido variado: algunas calles, un parque o agua — el distrito sigue siendo reconocible a primera vista." },
    { q: "¿Cuánto tarda la fabricación?", a: "1–3 días hábiles de impresión, luego envío a Ucrania." },
    { q: "¿Puedo imprimirlo yo mismo?", a: "Sí — descarga el archivo 3MF o STL listo, se abre directamente en Bambu Studio o PrusaSlicer." },
  ],
};

const WORLDS_FAQ: Record<string, ProseFaqItem[]> = {
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
  de: [
    { q: "Ist das eine echte oder eine erfundene Karte?", a: "Erfunden — die KI erzeugt das Gelände aus einem Textprompt, ohne Bezug zu einem echten Ort." },
    { q: "Kann ich die Datei kostenlos herunterladen?", a: "Ja, die GLB-Vorschaudatei ist kostenlos. Der Druck dieser Welt auf Bestellung wird separat im Chat besprochen." },
    { q: "Welche Größen sind verfügbar?", a: "S (8 cm), M (12 cm) und L (18 cm)." },
  ],
  pl: [
    { q: "Czy to prawdziwa mapa, czy fikcyjna?", a: "Fikcyjna — AI generuje teren z opisu tekstowego, bez powiązania z prawdziwym miejscem." },
    { q: "Czy mogę pobrać plik za darmo?", a: "Tak, plik podglądu GLB jest bezpłatny. Druk takiego świata na zamówienie omawiamy osobno na czacie." },
    { q: "Jakie rozmiary są dostępne?", a: "S (8 cm), M (12 cm) i L (18 cm)." },
  ],
  fr: [
    { q: "Est-ce une vraie carte ou une carte fictive ?", a: "Fictive — l'IA génère le terrain à partir d'une description textuelle, sans lien avec un lieu réel." },
    { q: "Puis-je télécharger le fichier gratuitement ?", a: "Oui, le fichier d'aperçu GLB est gratuit. L'impression de ce monde sur commande se discute séparément dans le chat." },
    { q: "Quelles tailles sont disponibles ?", a: "S (8 cm), M (12 cm) et L (18 cm)." },
  ],
  es: [
    { q: "¿Es un mapa real o ficticio?", a: "Ficticio — la IA genera el terreno a partir de una descripción de texto, sin vínculo con un lugar real." },
    { q: "¿Puedo descargar el archivo gratis?", a: "Sí, el archivo de vista previa GLB es gratuito. La impresión de este mundo por encargo se trata por separado en el chat." },
    { q: "¿Qué tamaños hay disponibles?", a: "S (8 cm), M (12 cm) y L (18 cm)." },
  ],
};

const SHOWCASE_FAQ: Record<string, ProseFaqItem[]> = {
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
  de: [
    { q: "Sind das echte Fotos oder 3D-Renderings?", a: "Beides: Der Bereich „In echt gedruckt“ zeigt echte Fotos fertiger Artikel, der Rest des Rasters sind interaktive 3D-Modelle, die du mit Finger oder Maus drehen kannst." },
    { q: "Aus welchem Material werden diese Muster gedruckt?", a: "Alle Muster werden aus Eco-PLA-Bioplastik gedruckt — demselben Material wie Kundenbestellungen." },
    { q: "Kann ich genau dieselbe Größe oder dasselbe Viertel bestellen?", a: "Ja — im Konfigurator lässt sich jeder Bereich und jede Größe wählen, auch die in der Galerie gezeigten." },
  ],
  pl: [
    { q: "Czy to prawdziwe zdjęcia, czy rendery 3D?", a: "Jedno i drugie: sekcja „Jak wygląda wydrukowane” to prawdziwe zdjęcia gotowych wyrobów, a reszta to interaktywne modele 3D, które można obracać palcem lub myszą." },
    { q: "Z jakiego materiału są te próbki?", a: "Wszystkie próbki są z bioplastiku Eco PLA — tego samego materiału, z którego drukowane są zamówienia klientów." },
    { q: "Czy mogę zamówić dokładnie taki sam rozmiar lub dzielnicę?", a: "Tak — w kreatorze można wybrać dowolny obszar i rozmiar, w tym te pokazane w galerii." },
  ],
  fr: [
    { q: "Sont-ce de vraies photos ou des rendus 3D ?", a: "Les deux : la section « Imprimé en vrai » montre de vraies photos d'articles finis, le reste de la grille étant des modèles 3D interactifs que vous pouvez faire tourner au doigt ou à la souris." },
    { q: "En quel matériau ces échantillons sont-ils imprimés ?", a: "Tous les échantillons sont imprimés en bioplastique Eco PLA — le même matériau que les commandes des clients." },
    { q: "Puis-je commander exactement la même taille ou le même quartier ?", a: "Oui — le configurateur permet de choisir n'importe quelle zone et taille, y compris celles montrées dans la galerie." },
  ],
  es: [
    { q: "¿Son fotos reales o renders 3D?", a: "Ambos: la sección «Impreso en la vida real» muestra fotos reales de artículos terminados, y el resto son modelos 3D interactivos que puedes girar con el dedo o el ratón." },
    { q: "¿De qué material están impresas estas muestras?", a: "Todas las muestras están impresas en bioplástico Eco PLA — el mismo material que los pedidos de los clientes." },
    { q: "¿Puedo pedir exactamente el mismo tamaño o distrito?", a: "Sí — el configurador permite elegir cualquier zona y tamaño, incluidos los mostrados en la galería." },
  ],
};

export function proseFaq(page: "create" | "worlds" | "showcase", locale: string): ProseFaqItem[] {
  const dict = page === "create" ? CREATE_FAQ : page === "worlds" ? WORLDS_FAQ : SHOWCASE_FAQ;
  return dict[locale] ?? dict.en;
}
