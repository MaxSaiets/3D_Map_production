import type { Metadata } from "next";
import { setRequestLocale } from "next-intl/server";
import { BASE, localeUrl, priceValidUntil, priceValidFrom, MERCHANT_RETURN_POLICY_LD } from "@/i18n/metadata";
import { routing, locales, localeMeta, defaultLocale, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { MAP_SIZE_PRICES_UAH, MAP_RELIEF_ADDON_UAH, KEYCHAIN_PRICE_UAH, mapPriceEur } from "@/lib/mapPrices";

/**
 * Лендінг /karpaty — «рельєфна 3D-мапа Карпат / топографічна карта».
 * Кластер з аудиту запитів 16.07.2026: «топографічна карта купити (України/
 * Карпат/областей)», «рельєфна карта карпат/києва», «мапа карпат купити»,
 * «3д мапа карпат» — прямий транзакційний попит; рельєф-режим у конструкторі
 * існує давно, але сторінки під цей інтент не було. 6 локалей, Product LD.
 */
type KarpatyCopy = {
  title: string; description: string; h1: string; intro: string; introB: string;
  h2what: string; pWhat: string[]; h2formats: string;
  formats: { label: string; desc: string; price: string }[];
  h2gpx: string; pGpx: string[]; h2how: string; how: string[]; h2faq: string;
  faq: { q: string; a: string }[]; cta: string; ctaKeychain: string; photosAlt: string;
};

const M = MAP_SIZE_PRICES_UAH;
const R = MAP_RELIEF_ADDON_UAH;
const eur = (uah: number) => mapPriceEur(uah);

const COPY: Record<AppLocale, KarpatyCopy> = {
  uk: {
    title: "Рельєфна 3D-мапа Карпат — купити топографічну модель",
    description: `Об'ємна топографічна карта Карпат з реальними висотами: Говерла, Чорногора, Боржава — будь-який хребет чи маршрут. 3D-друк від ${M[80] + R} ₴, топо-брелок від ${KEYCHAIN_PRICE_UAH} ₴.`,
    h1: "Рельєфна 3D-мапа Карпат",
    intro: "Справжня об'ємна модель гір: кожен хребет, долина й вершина з реальними перепадами висот за супутниковими даними. Не малюнок горизонталей — фізичний рельєф, який читається пальцями.",
    introB: "Оберіть будь-яку ділянку Карпат — Говерлу з Чорногірським хребтом, Боржаву, Драгобрат чи стежку вашого походу — і ми надрукуємо її як настільну модель, панно на стіну або топо-брелок.",
    h2what: "Що таке рельєфна 3D-мапа",
    pWhat: [
      "Конструктор бере супутникові дані висот і будує точну тривимірну поверхню обраної ділянки. Гори здіймаються, долини западають — масштаб висот можна підсилити, щоб рельєф читався ще виразніше.",
      "Поверх рельєфу лягають стежки, річки, ліси й населені пункти з OpenStreetMap. Модель друкується з екологічного пластику Eco PLA й не потребує жодної обробки.",
    ],
    h2formats: "Формати та ціни",
    formats: [
      { label: `Настільна мапа з рельєфом (8–15 см)`, desc: "модель обраного хребта чи долини на полицю", price: `від ${M[80] + R} ₴` },
      { label: "Панно з плиток із рельєфом", desc: "великий шматок гір із кількох плиток на стіну", price: `від ${(M[80] + R) * 4} ₴` },
      { label: `Топо-брелок (55×30 мм)`, desc: "вершина чи маршрут у кишені — з написом на звороті", price: `від ${KEYCHAIN_PRICE_UAH} ₴` },
      { label: "Мапа з GPX-треком походу", desc: "ваш маршрут рельєфною лінією поверх гір", price: `від ${KEYCHAIN_PRICE_UAH} ₴` },
    ],
    h2gpx: "Ваш похід — на мапі",
    pGpx: [
      "Ходили на Говерлу чи пройшли Чорногірський хребет? Завантажте GPX-трек зі Strava, Garmin чи Komoot — лінія маршруту ляже рельєфом поверх гір саме там, де ви йшли. Такої мапи немає більше ні в кого: це ваш маршрут.",
    ],
    h2how: "Як замовити",
    how: [
      "Відкрийте конструктор і знайдіть потрібну ділянку Карпат (чи будь-яких гір світу).",
      "Увімкніть режим рельєфу — модель одразу стане об'ємною. За бажанням підсильте масштаб висот.",
      "Додайте GPX-трек, якщо хочете увічнити маршрут.",
      "Замовте друк — виготовимо за 1–3 робочі дні й надішлемо Новою Поштою або в ЄС.",
    ],
    h2faq: "Часті запитання",
    faq: [
      { q: "Чим це відрізняється від паперової топографічної карти?", a: "Паперова карта показує висоти горизонталями, які треба вміти читати. Рельєфна 3D-мапа — фізично об'ємна: Говерлу видно як гору, долину Прута — як западину. Це і навігаційний сувенір, і настільна модель місця, яке ви пройшли ногами." },
      { q: "Які саме Карпати можна замовити?", a: "Будь-яку ділянку: Чорногора з Говерлою і Петросом, Боржава, Свидовець із Драгобратом, Ґорґани, Мармароси — або польські Татри, Альпи, будь-які гори світу. Ви самі обираєте рамку в конструкторі." },
      { q: "Наскільки точний рельєф?", a: "Висоти беруться з супутникових даних (роздільність ~30 м) — для моделі 8–15 см цього достатньо, щоб кожен хребет і сідловина були на своїх місцях. Масштаб висот можна підсилити ×1.5–2, щоб гори виглядали драматичніше." },
      { q: "Можна замовити рельєфну мапу міста, не гір?", a: "Так — рельєф вмикається для будь-якої ділянки. Для горбистих міст (Київ, Львів) це особливо гарно: видно печерські пагорби чи Високий Замок. Для рівнинних міст рельєф майже непомітний — чесно попереджаємо." },
      { q: "Скільки коштує і як швидко?", a: `Настільна рельєфна мапа — від ${M[80] + R} ₴ (розмір M, 8 см). Топо-брелок — від ${KEYCHAIN_PRICE_UAH} ₴. Виготовлення 1–3 робочі дні, доставка Новою Поштою по Україні та в 15 країн ЄС.` },
    ],
    cta: "Створити мапу Карпат у конструкторі",
    ctaKeychain: "Топо-брелок з вершиною",
    photosAlt: "Фото реальної 3D-друкованої рельєфної мапи гір",
  },
  en: {
    title: "3D relief map of the Carpathians — buy a topographic model",
    description: `A physical topographic map of the Carpathians with real elevations: Hoverla, Chornohora, Borzhava — any ridge or trail. 3D-printed from ≈€${eur(M[80] + R)}, topo keychain from ≈€${eur(KEYCHAIN_PRICE_UAH)}.`,
    h1: "3D relief map of the Carpathians",
    intro: "A true three-dimensional model of the mountains: every ridge, valley and peak with real elevation from satellite data. Not contour lines on paper — physical relief you can read with your fingers.",
    introB: "Pick any part of the Carpathians — Hoverla with the Chornohora ridge, Borzhava, Drahobrat or your own trekking route — and we'll print it as a desk model, a wall panel or a topo keychain.",
    h2what: "What a relief 3D map is",
    pWhat: [
      "The builder takes satellite elevation data and constructs a precise 3D surface of the chosen area. Mountains rise, valleys sink — the height scale can be amplified so the relief reads even stronger.",
      "Trails, rivers, forests and villages from OpenStreetMap are draped over the terrain. The model is printed in eco-friendly Eco PLA and needs no finishing.",
    ],
    h2formats: "Formats and prices",
    formats: [
      { label: "Desk relief map (8–15 cm)", desc: "a model of the chosen ridge or valley for a shelf", price: `from ≈€${eur(M[80] + R)}` },
      { label: "Multi-tile relief panel", desc: "a large piece of the mountains for a wall", price: `from ≈€${eur((M[80] + R) * 4)}` },
      { label: "Topo keychain (55×30 mm)", desc: "a summit or route in your pocket, text on the back", price: `from ≈€${eur(KEYCHAIN_PRICE_UAH)}` },
      { label: "Map with your GPX track", desc: "your route as a relief line over the mountains", price: `from ≈€${eur(KEYCHAIN_PRICE_UAH)}` },
    ],
    h2gpx: "Your trek — on the map",
    pGpx: [
      "Climbed Hoverla or walked the Chornohora ridge? Upload a GPX track from Strava, Garmin or Komoot — the route line is laid in relief over the mountains exactly where you walked. Nobody else has this map: it's your route.",
    ],
    h2how: "How to order",
    how: [
      "Open the builder and find the part of the Carpathians (or any mountains on Earth) you want.",
      "Enable terrain mode — the model becomes three-dimensional instantly. Amplify the height scale if you like.",
      "Add a GPX track if you want to immortalise a route.",
      "Order the print — made in 1–3 business days, shipped to 15 EU countries.",
    ],
    h2faq: "FAQ",
    faq: [
      { q: "How is it different from a paper topographic map?", a: "A paper map shows elevation as contour lines you need to know how to read. A relief 3D map is physically volumetric: Hoverla is visibly a mountain, the Prut valley a depression. It's both a navigation souvenir and a desk model of a place you walked." },
      { q: "Which parts of the Carpathians can I order?", a: "Any: Chornohora with Hoverla and Petros, Borzhava, Svydovets with Drahobrat, Gorgany — or the Polish Tatras, the Alps, any mountains on Earth. You choose the frame in the builder yourself." },
      { q: "How accurate is the relief?", a: "Elevations come from satellite data (~30 m resolution) — plenty for an 8–15 cm model to put every ridge and saddle in its place. The height scale can be amplified ×1.5–2 for a more dramatic look." },
      { q: "Can I get a relief map of a city rather than mountains?", a: "Yes — terrain can be enabled for any area. Hilly cities (Kyiv, Lviv) look especially good. For flat cities the relief is barely visible — we're honest about that." },
      { q: "Price and lead time?", a: `A desk relief map from ≈€${eur(M[80] + R)} (size M, 8 cm). Topo keychain from ≈€${eur(KEYCHAIN_PRICE_UAH)}. Production 1–3 business days, EU shipping.` },
    ],
    cta: "Create a Carpathians map in the builder",
    ctaKeychain: "Topo keychain with a summit",
    photosAlt: "Photo of a real 3D-printed relief mountain map",
  },
  de: {
    title: "3D-Reliefkarte der Karpaten — topografisches Modell kaufen",
    description: `Physische topografische Karte der Karpaten mit echten Höhen: Hoverla, Tschornohora, Borschawa — jeder Kamm, jede Route. 3D-Druck ab ≈${eur(M[80] + R)} €, Topo-Anhänger ab ≈${eur(KEYCHAIN_PRICE_UAH)} €.`,
    h1: "3D-Reliefkarte der Karpaten",
    intro: "Ein echtes dreidimensionales Modell der Berge: jeder Kamm, jedes Tal und jeder Gipfel mit realen Höhen aus Satellitendaten. Keine Höhenlinien auf Papier — physisches Relief, das man mit den Fingern liest.",
    introB: "Wähle einen beliebigen Teil der Karpaten — die Hoverla mit dem Tschornohora-Kamm, Borschawa, Drahobrat oder deine eigene Trekkingroute — und wir drucken ihn als Tischmodell, Wandpanel oder Topo-Anhänger.",
    h2what: "Was eine Relief-3D-Karte ist",
    pWhat: [
      "Der Konfigurator nimmt Satelliten-Höhendaten und baut eine präzise 3D-Oberfläche des gewählten Bereichs. Berge erheben sich, Täler senken sich — die Höhenskala lässt sich verstärken.",
      "Wege, Flüsse, Wälder und Orte aus OpenStreetMap legen sich über das Gelände. Gedruckt aus Eco PLA, keine Nachbearbeitung nötig.",
    ],
    h2formats: "Formate und Preise",
    formats: [
      { label: "Tisch-Reliefkarte (8–15 cm)", desc: "Modell des gewählten Kamms oder Tals fürs Regal", price: `ab ≈${eur(M[80] + R)} €` },
      { label: "Relief-Panel aus Kacheln", desc: "ein großes Stück Gebirge für die Wand", price: `ab ≈${eur((M[80] + R) * 4)} €` },
      { label: "Topo-Anhänger (55×30 mm)", desc: "Gipfel oder Route in der Tasche, Text auf der Rückseite", price: `ab ≈${eur(KEYCHAIN_PRICE_UAH)} €` },
      { label: "Karte mit deinem GPX-Track", desc: "deine Route als Relieflinie über den Bergen", price: `ab ≈${eur(KEYCHAIN_PRICE_UAH)} €` },
    ],
    h2gpx: "Deine Tour — auf der Karte",
    pGpx: [
      "Auf der Hoverla gewesen oder den Tschornohora-Kamm gegangen? Lade einen GPX-Track aus Strava, Garmin oder Komoot hoch — die Routenlinie legt sich als Relief über die Berge, genau dort, wo du gegangen bist.",
    ],
    h2how: "So bestellst du",
    how: [
      "Öffne den Konfigurator und finde den gewünschten Teil der Karpaten (oder beliebiger Berge).",
      "Aktiviere den Relief-Modus — das Modell wird sofort dreidimensional.",
      "Füge einen GPX-Track hinzu, wenn du eine Route verewigen willst.",
      "Bestelle den Druck — Fertigung in 1–3 Werktagen, Versand in 15 EU-Länder.",
    ],
    h2faq: "Häufige Fragen",
    faq: [
      { q: "Was unterscheidet sie von einer Papier-Topokarte?", a: "Eine Papierkarte zeigt Höhen als Linien, die man lesen können muss. Eine Relief-3D-Karte ist physisch plastisch: die Hoverla ist sichtbar ein Berg. Zugleich Souvenir und Tischmodell eines Ortes, den du selbst gegangen bist." },
      { q: "Welche Teile der Karpaten sind möglich?", a: "Alle: Tschornohora mit Hoverla, Borschawa, Swydowez mit Drahobrat, Gorgany — oder die Tatra, die Alpen, jedes Gebirge der Erde. Den Rahmen wählst du selbst im Konfigurator." },
      { q: "Wie genau ist das Relief?", a: "Höhen stammen aus Satellitendaten (~30 m Auflösung) — für ein 8–15-cm-Modell mehr als genug. Die Höhenskala lässt sich ×1,5–2 verstärken." },
      { q: "Geht auch eine Relief-Stadtkarte?", a: "Ja — das Relief lässt sich für jeden Bereich aktivieren. Hügelige Städte (Kyiv, Lwiw) wirken besonders gut; bei flachen Städten ist es kaum sichtbar — das sagen wir ehrlich." },
      { q: "Preis und Lieferzeit?", a: `Tisch-Reliefkarte ab ≈${eur(M[80] + R)} € (Größe M, 8 cm). Topo-Anhänger ab ≈${eur(KEYCHAIN_PRICE_UAH)} €. Fertigung 1–3 Werktage, EU-Versand.` },
    ],
    cta: "Karpaten-Karte im Konfigurator erstellen",
    ctaKeychain: "Topo-Anhänger mit Gipfel",
    photosAlt: "Foto einer echten 3D-gedruckten Relief-Bergkarte",
  },
  pl: {
    title: "Mapa 3D Karpat z rzeźbą terenu — model topograficzny",
    description: `Fizyczna mapa topograficzna Karpat z prawdziwymi wysokościami: Howerla, Czarnohora, Borżawa — dowolny grzbiet lub szlak. Druk 3D od ≈${eur(M[80] + R)} €, brelok topo od ≈${eur(KEYCHAIN_PRICE_UAH)} €.`,
    h1: "Mapa 3D Karpat z rzeźbą terenu",
    intro: "Prawdziwy trójwymiarowy model gór: każdy grzbiet, dolina i szczyt z realnymi wysokościami z danych satelitarnych. Nie poziomice na papierze — fizyczna rzeźba, którą czyta się palcami.",
    introB: "Wybierz dowolny fragment Karpat — Howerlę z grzbietem Czarnohory, Borżawę, Drahobrat albo trasę własnej wędrówki — a wydrukujemy go jako model na biurko, panel na ścianę lub brelok topo.",
    h2what: "Czym jest mapa 3D z rzeźbą",
    pWhat: [
      "Kreator pobiera satelitarne dane wysokości i buduje precyzyjną powierzchnię 3D wybranego obszaru. Góry się wznoszą, doliny zapadają — skalę wysokości można wzmocnić.",
      "Szlaki, rzeki, lasy i miejscowości z OpenStreetMap układają się na terenie. Model drukowany z Eco PLA, bez obróbki.",
    ],
    h2formats: "Formaty i ceny",
    formats: [
      { label: "Mapa na biurko z rzeźbą (8–15 cm)", desc: "model wybranego grzbietu lub doliny na półkę", price: `od ≈${eur(M[80] + R)} €` },
      { label: "Panel z kafelków z rzeźbą", desc: "duży fragment gór na ścianę", price: `od ≈${eur((M[80] + R) * 4)} €` },
      { label: "Brelok topo (55×30 mm)", desc: "szczyt lub trasa w kieszeni, napis z tyłu", price: `od ≈${eur(KEYCHAIN_PRICE_UAH)} €` },
      { label: "Mapa z Twoim śladem GPX", desc: "Twoja trasa jako linia reliefowa nad górami", price: `od ≈${eur(KEYCHAIN_PRICE_UAH)} €` },
    ],
    h2gpx: "Twoja wędrówka — na mapie",
    pGpx: [
      "Byłeś na Howerli albo przeszedłeś Czarnohorę? Wgraj ślad GPX ze Stravy, Garmina lub Komoot — linia trasy ułoży się reliefem na górach dokładnie tam, gdzie szedłeś. Takiej mapy nie ma nikt inny.",
    ],
    h2how: "Jak zamówić",
    how: [
      "Otwórz kreator i znajdź wybrany fragment Karpat (lub dowolnych gór świata — także Tatr).",
      "Włącz tryb rzeźby terenu — model od razu stanie się trójwymiarowy.",
      "Dodaj ślad GPX, jeśli chcesz uwiecznić trasę.",
      "Zamów druk — wykonanie 1–3 dni robocze, wysyłka do 15 krajów UE.",
    ],
    h2faq: "Częste pytania",
    faq: [
      { q: "Czym różni się od papierowej mapy topograficznej?", a: "Papierowa mapa pokazuje wysokości poziomicami, które trzeba umieć czytać. Mapa 3D z rzeźbą jest fizycznie przestrzenna: Howerla to widoczna góra. To pamiątka i model miejsca, które przeszedłeś na własnych nogach." },
      { q: "Które fragmenty Karpat można zamówić?", a: "Dowolne: Czarnohora z Howerlą, Borżawa, Świdowiec z Drahobratem, Gorgany — albo polskie Tatry, Alpy, dowolne góry świata. Ramkę wybierasz sam w kreatorze." },
      { q: "Jak dokładna jest rzeźba?", a: "Wysokości pochodzą z danych satelitarnych (~30 m) — dla modelu 8–15 cm w zupełności wystarczy. Skalę wysokości można wzmocnić ×1,5–2." },
      { q: "Można zamówić mapę miasta z rzeźbą?", a: "Tak — rzeźbę można włączyć dla dowolnego obszaru. Pagórkowate miasta (Kijów, Lwów) wyglądają szczególnie dobrze; w płaskich miastach relief jest ledwo widoczny — mówimy o tym uczciwie." },
      { q: "Cena i termin?", a: `Mapa na biurko z rzeźbą od ≈${eur(M[80] + R)} € (rozmiar M, 8 cm). Brelok topo od ≈${eur(KEYCHAIN_PRICE_UAH)} €. Wykonanie 1–3 dni robocze, wysyłka do UE.` },
    ],
    cta: "Stwórz mapę Karpat w kreatorze",
    ctaKeychain: "Brelok topo ze szczytem",
    photosAlt: "Zdjęcie prawdziwej drukowanej mapy gór z rzeźbą terenu",
  },
  fr: {
    title: "Carte 3D en relief des Carpates — modèle topographique",
    description: `Carte topographique physique des Carpates avec les vraies altitudes : Hoverla, Tchornohora, Borjava — chaque crête ou sentier. Impression 3D dès ≈${eur(M[80] + R)} €, porte-clés topo dès ≈${eur(KEYCHAIN_PRICE_UAH)} €.`,
    h1: "Carte 3D en relief des Carpates",
    intro: "Un vrai modèle tridimensionnel des montagnes : chaque crête, vallée et sommet avec les altitudes réelles issues de données satellites. Pas des courbes de niveau sur papier — un relief physique qui se lit du bout des doigts.",
    introB: "Choisissez n'importe quelle partie des Carpates — la Hoverla et la crête de Tchornohora, Borjava, Drahobrat ou votre propre itinéraire de randonnée — et nous l'imprimons en modèle de bureau, panneau mural ou porte-clés topo.",
    h2what: "Qu'est-ce qu'une carte 3D en relief",
    pWhat: [
      "Le configurateur prend les données satellites d'altitude et construit une surface 3D précise de la zone choisie. Les montagnes s'élèvent, les vallées s'enfoncent — l'échelle des hauteurs peut être amplifiée.",
      "Sentiers, rivières, forêts et villages d'OpenStreetMap se drapent sur le terrain. Imprimé en Eco PLA, sans finition nécessaire.",
    ],
    h2formats: "Formats et prix",
    formats: [
      { label: "Carte relief de bureau (8–15 cm)", desc: "modèle de la crête ou vallée choisie pour une étagère", price: `dès ≈${eur(M[80] + R)} €` },
      { label: "Panneau en tuiles avec relief", desc: "un grand morceau de montagne pour le mur", price: `dès ≈${eur((M[80] + R) * 4)} €` },
      { label: "Porte-clés topo (55×30 mm)", desc: "un sommet ou un itinéraire en poche, texte au dos", price: `dès ≈${eur(KEYCHAIN_PRICE_UAH)} €` },
      { label: "Carte avec votre trace GPX", desc: "votre itinéraire en ligne de relief sur les montagnes", price: `dès ≈${eur(KEYCHAIN_PRICE_UAH)} €` },
    ],
    h2gpx: "Votre randonnée — sur la carte",
    pGpx: [
      "Monté à la Hoverla ou parcouru la crête de Tchornohora ? Téléversez une trace GPX depuis Strava, Garmin ou Komoot — la ligne de l'itinéraire se pose en relief sur les montagnes, exactement là où vous êtes passé.",
    ],
    h2how: "Comment commander",
    how: [
      "Ouvrez le configurateur et trouvez la partie des Carpates (ou de n'importe quelles montagnes) voulue.",
      "Activez le mode relief — le modèle devient tridimensionnel instantanément.",
      "Ajoutez une trace GPX pour immortaliser un itinéraire.",
      "Commandez l'impression — fabrication en 1–3 jours ouvrés, livraison dans 15 pays de l'UE.",
    ],
    h2faq: "Questions fréquentes",
    faq: [
      { q: "Quelle différence avec une carte topographique papier ?", a: "Une carte papier montre l'altitude par des courbes de niveau qu'il faut savoir lire. Une carte 3D en relief est physiquement volumique : la Hoverla est visiblement une montagne. À la fois souvenir et modèle d'un lieu parcouru à pied." },
      { q: "Quelles parties des Carpates ?", a: "Toutes : Tchornohora avec la Hoverla, Borjava, Svydovets avec Drahobrat, Gorgany — ou les Tatras, les Alpes, toutes les montagnes du monde. Vous choisissez le cadre vous-même." },
      { q: "Quelle précision du relief ?", a: "Les altitudes viennent de données satellites (~30 m de résolution) — largement assez pour un modèle de 8–15 cm. L'échelle des hauteurs peut être amplifiée ×1,5–2." },
      { q: "Une ville en relief plutôt que des montagnes ?", a: "Oui — le relief s'active pour n'importe quelle zone. Les villes vallonnées (Kyiv, Lviv) rendent très bien ; pour les villes plates, il est à peine visible — on vous le dit honnêtement." },
      { q: "Prix et délais ?", a: `Carte relief de bureau dès ≈${eur(M[80] + R)} € (taille M, 8 cm). Porte-clés topo dès ≈${eur(KEYCHAIN_PRICE_UAH)} €. Fabrication 1–3 jours ouvrés, livraison UE.` },
    ],
    cta: "Créer une carte des Carpates",
    ctaKeychain: "Porte-clés topo avec un sommet",
    photosAlt: "Photo d'une vraie carte de montagne en relief imprimée en 3D",
  },
  es: {
    title: "Mapa 3D en relieve de los Cárpatos — modelo topográfico",
    description: `Mapa topográfico físico de los Cárpatos con alturas reales: Hoverla, Chornohora, Borzhava — cualquier cresta o ruta. Impresión 3D desde ≈${eur(M[80] + R)} €, llavero topo desde ≈${eur(KEYCHAIN_PRICE_UAH)} €.`,
    h1: "Mapa 3D en relieve de los Cárpatos",
    intro: "Un verdadero modelo tridimensional de las montañas: cada cresta, valle y cumbre con alturas reales de datos satelitales. No son curvas de nivel en papel — es relieve físico que se lee con los dedos.",
    introB: "Elige cualquier parte de los Cárpatos — la Hoverla con la cresta de Chornohora, Borzhava, Drahobrat o tu propia ruta de trekking — y la imprimimos como modelo de escritorio, panel de pared o llavero topo.",
    h2what: "Qué es un mapa 3D en relieve",
    pWhat: [
      "El configurador toma datos satelitales de elevación y construye una superficie 3D precisa de la zona elegida. Las montañas se elevan, los valles se hunden — la escala de alturas se puede amplificar.",
      "Senderos, ríos, bosques y pueblos de OpenStreetMap se superponen al terreno. Impreso en Eco PLA, sin acabados necesarios.",
    ],
    h2formats: "Formatos y precios",
    formats: [
      { label: "Mapa de escritorio en relieve (8–15 cm)", desc: "modelo de la cresta o valle elegido para una estantería", price: `desde ≈${eur(M[80] + R)} €` },
      { label: "Panel de azulejos con relieve", desc: "un gran fragmento de montaña para la pared", price: `desde ≈${eur((M[80] + R) * 4)} €` },
      { label: "Llavero topo (55×30 mm)", desc: "una cumbre o ruta en el bolsillo, texto al dorso", price: `desde ≈${eur(KEYCHAIN_PRICE_UAH)} €` },
      { label: "Mapa con tu track GPX", desc: "tu ruta como línea en relieve sobre las montañas", price: `desde ≈${eur(KEYCHAIN_PRICE_UAH)} €` },
    ],
    h2gpx: "Tu ruta — en el mapa",
    pGpx: [
      "¿Subiste a la Hoverla o recorriste la cresta de Chornohora? Sube un track GPX de Strava, Garmin o Komoot — la línea de la ruta se coloca en relieve sobre las montañas exactamente por donde caminaste.",
    ],
    h2how: "Cómo pedir",
    how: [
      "Abre el configurador y encuentra la parte de los Cárpatos (o cualquier montaña del mundo).",
      "Activa el modo relieve — el modelo se vuelve tridimensional al instante.",
      "Añade un track GPX si quieres inmortalizar una ruta.",
      "Pide la impresión — fabricación en 1–3 días hábiles, envío a 15 países de la UE.",
    ],
    h2faq: "Preguntas frecuentes",
    faq: [
      { q: "¿En qué se diferencia de un mapa topográfico de papel?", a: "Un mapa de papel muestra la altura con curvas de nivel que hay que saber leer. Un mapa 3D en relieve es físicamente volumétrico: la Hoverla se ve como una montaña. Es recuerdo y modelo de un lugar que recorriste a pie." },
      { q: "¿Qué partes de los Cárpatos puedo pedir?", a: "Cualquiera: Chornohora con la Hoverla, Borzhava, Svydovets con Drahobrat, Gorgany — o los Tatras, los Alpes, cualquier montaña del mundo. Tú eliges el marco en el configurador." },
      { q: "¿Qué precisión tiene el relieve?", a: "Las alturas provienen de datos satelitales (~30 m de resolución) — de sobra para un modelo de 8–15 cm. La escala de alturas se puede amplificar ×1,5–2." },
      { q: "¿Un mapa de ciudad en relieve en vez de montañas?", a: "Sí — el relieve se activa para cualquier zona. Las ciudades con colinas (Kyiv, Leópolis) lucen especialmente bien; en ciudades llanas apenas se nota — lo decimos con honestidad." },
      { q: "¿Precio y plazos?", a: `Mapa de escritorio en relieve desde ≈${eur(M[80] + R)} € (talla M, 8 cm). Llavero topo desde ≈${eur(KEYCHAIN_PRICE_UAH)} €. Fabricación 1–3 días hábiles, envío a la UE.` },
    ],
    cta: "Crear un mapa de los Cárpatos",
    ctaKeychain: "Llavero topo con una cumbre",
    photosAlt: "Foto de un mapa de montaña en relieve impreso en 3D real",
  },
};

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  const c = COPY[locale];
  const path = "/karpaty";
  const languages: Record<string, string> = {};
  for (const l of locales) languages[localeMeta[l].htmlLang] = localeUrl(l, path);
  languages["x-default"] = localeUrl(defaultLocale, path);
  return {
    title: c.title,
    description: c.description,
    alternates: { canonical: localeUrl(locale, path), languages },
    openGraph: {
      title: c.title, description: c.description, url: localeUrl(locale, path),
      siteName: "Monadruk", type: "website", locale: localeMeta[locale].ogLocale,
      images: [`${BASE}/opengraph-image`],
    },
    twitter: { card: "summary_large_image", title: c.title, description: c.description, images: [`${BASE}/opengraph-image`] },
  };
}

export default async function KarpatyPage({ params }: { params: { locale: string } }) {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  setRequestLocale(locale);
  const c = COPY[locale];
  const isUA = locale === "uk";
  const path = "/karpaty";

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Product",
        name: c.h1,
        description: c.description,
        image: [`${BASE}/showcase/real-4.webp`, `${BASE}/showcase/real-8.webp`],
        brand: { "@type": "Brand", name: "Monadruk" },
        sku: "MND-RELIEF-KARPATY",
        offers: {
          "@type": "AggregateOffer",
          priceCurrency: isUA ? "UAH" : "EUR",
          lowPrice: isUA ? String(KEYCHAIN_PRICE_UAH) : String(eur(KEYCHAIN_PRICE_UAH)),
          highPrice: isUA ? String((M[110] + R) * 9) : String(eur((M[110] + R) * 9)),
          offerCount: "4",
          priceValidUntil: priceValidUntil(),
          validFrom: priceValidFrom(),
          availability: "https://schema.org/InStock",
          url: localeUrl(locale, path),
          hasMerchantReturnPolicy: MERCHANT_RETURN_POLICY_LD,
        },
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: c.h1, item: localeUrl(locale, path) },
        ],
      },
      {
        "@type": "FAQPage",
        mainEntity: c.faq.map((f) => ({ "@type": "Question", name: f.q, acceptedAnswer: { "@type": "Answer", text: f.a } })),
      },
    ],
  };

  return (
    <main id="main-content" tabIndex={-1} className="mx-auto max-w-[920px] px-5 py-14 lg:py-20">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      <h1 className="text-[clamp(28px,4vw,46px)] leading-tight">{c.h1}</h1>
      <p className="mt-4 max-w-[680px] text-[15px] leading-relaxed text-ink-2">{c.intro}</p>
      <p className="mt-3 max-w-[680px] text-[15px] leading-relaxed text-ink-2">{c.introB}</p>

      <section className="mt-8 flex flex-wrap gap-3">
        <Link href="/create" className="inline-flex min-h-[44px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-5 py-2.5 text-sm font-semibold text-white transition hover:opacity-90">
          {c.cta}
        </Link>
        <Link href="/keychains" className="inline-flex min-h-[44px] items-center justify-center rounded-[22px] border border-line-soft bg-white/80 px-5 py-2.5 text-sm font-semibold text-ink transition hover:border-[var(--accent)]">
          {c.ctaKeychain}
        </Link>
      </section>

      <section className="mt-10 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {["real-4", "real-8", "real-3", "real-9"].map((img) => (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            key={img}
            src={`/showcase/${img}.webp`}
            alt={c.photosAlt}
            loading="lazy"
            className="aspect-square w-full rounded-[14px] border border-line-soft object-cover"
          />
        ))}
      </section>

      <section className="mt-12 max-w-[680px]">
        <h2 className="text-[20px] font-semibold">{c.h2what}</h2>
        {c.pWhat.map((p, i) => (
          <p key={i} className="mt-3 text-[15px] leading-relaxed text-ink-2">{p}</p>
        ))}
      </section>

      <section className="mt-12">
        <h2 className="text-[20px] font-semibold">{c.h2formats}</h2>
        <ul className="mt-4 grid gap-3 sm:grid-cols-2">
          {c.formats.map((s) => (
            <li key={s.label} className="rounded-[18px] border border-line-soft bg-white/70 px-5 py-4">
              <p className="text-[15px] font-semibold text-ink">{s.label}</p>
              <p className="mt-1 text-[13.5px] leading-relaxed text-ink-2">{s.desc}</p>
              <p className="mt-2 text-[15px] font-semibold text-[var(--accent-strong)]">{s.price}</p>
            </li>
          ))}
        </ul>
      </section>

      <section className="mt-12 max-w-[680px]">
        <h2 className="text-[20px] font-semibold">{c.h2gpx}</h2>
        {c.pGpx.map((p, i) => (
          <p key={i} className="mt-3 text-[15px] leading-relaxed text-ink-2">{p}</p>
        ))}
      </section>

      <section className="mt-12 max-w-[680px]">
        <h2 className="text-[20px] font-semibold">{c.h2how}</h2>
        <ol className="mt-4 flex flex-col gap-2.5">
          {c.how.map((step, i) => (
            <li key={i} className="flex gap-3 text-[15px] leading-relaxed text-ink-2">
              <span className="mt-0.5 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[var(--accent-strong)]/10 text-[13px] font-bold text-[var(--accent-strong)]">{i + 1}</span>
              {step}
            </li>
          ))}
        </ol>
      </section>

      <section className="mt-12 max-w-[680px]">
        <h2 className="text-[20px] font-semibold">{c.h2faq}</h2>
        <dl className="mt-4 flex flex-col gap-4">
          {c.faq.map((f) => (
            <div key={f.q}>
              <dt className="text-[15px] font-semibold text-ink">{f.q}</dt>
              <dd className="mt-1.5 text-[14.5px] leading-relaxed text-ink-2">{f.a}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="mt-10">
        <Link href="/create" className="inline-flex min-h-[44px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-5 py-2.5 text-sm font-semibold text-white transition hover:opacity-90">
          {c.cta}
        </Link>
      </section>
    </main>
  );
}
