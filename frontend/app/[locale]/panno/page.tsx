import type { Metadata } from "next";
import { setRequestLocale } from "next-intl/server";
import { BASE, localeUrl, priceValidUntil, priceValidFrom, MERCHANT_RETURN_POLICY_LD } from "@/i18n/metadata";
import { routing, locales, localeMeta, defaultLocale, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { CITY_PAGES } from "@/lib/cityPages";
import { MAP_SIZE_PRICES_UAH, mapPriceEur } from "@/lib/mapPrices";

/**
 * Лендінг /panno — «карта міста на стіну / панно з плиток».
 * НАЙБІЛЬШИЙ незакритий кластер попиту з аудиту запитів 16.07.2026
 * («карта на стіну купити», «панно карта україни», «карта з підсвіткою»,
 * «декор на стіну карта») — продукт (серія плиток 2×2/3×3 у конструкторі)
 * існував, але не мав жодної сторінки під цей інтент. 6 локалей, FAQPage
 * + Product LD, фото реальних друків, лінки на /create (режим сітки).
 */
type PannoCopy = {
  title: string; description: string; h1: string; intro: string; introB: string;
  h2what: string; pWhat: string[]; h2sizes: string; sizes: { label: string; desc: string; price: string }[];
  h2how: string; how: string[]; h2cities: string; h2faq: string;
  faq: { q: string; a: string }[]; cta: string; ctaSecondary: string; photosAlt: string;
};

const M = MAP_SIZE_PRICES_UAH; // {55:250, 80:350, 110:450, 150:550}
const eur = (uah: number) => mapPriceEur(uah);

const COPY: Record<AppLocale, PannoCopy> = {
  uk: {
    title: "Панно-карта міста на стіну — купити 3D-мапу з плиток",
    description: `Карта міста на стіну з 3D-друкованих плиток: обери район — зберемо панно 2×2 чи 3×3 з вулицями, кварталами й річками. Від ${M[80] * 4} ₴ за панно 2×2, доставка по Україні та ЄС.`,
    h1: "Панно-карта міста на стіну",
    intro: "Настінна 3D-карта, зібрана з друкованих плиток: кожна плитка — фрагмент міста з реальними вулицями, будинками й водою, а разом вони складаються у велике панно вашого району.",
    introB: "Це не плакат і не дерев'яний контур країни — це точна тривимірна модель місця, яке щось означає: ваш квартал, набережна, центр рідного міста. Плитки з'єднуються пазами в суцільне полотно.",
    h2what: "Що це таке",
    pWhat: [
      "Ви обираєте ділянку міста в конструкторі й ділите її на сітку — 2×2, 3×3 або будь-яку іншу. Кожна клітинка друкується окремою плиткою з екологічного пластику Eco PLA, а зʼєднувальні пази тримають плитки разом на стіні чи полиці.",
      "Вулиці, будинки з реальними висотами, парки й річки беруться з відкритих даних OpenStreetMap. Для горбистих міст можна ввімкнути рельєф — перепади висот стануть відчутними на дотик.",
    ],
    h2sizes: "Розміри та ціни",
    sizes: [
      { label: "Панно 2×2 (4 плитки 8 см)", desc: "≈16×16 см — компактне панно на полицю чи невелику стіну", price: `від ${M[80] * 4} ₴` },
      { label: "Панно 3×3 (9 плиток 8 см)", desc: "≈24×24 см — повноцінний настінний акцент", price: `від ${M[80] * 9} ₴` },
      { label: "Панно з плиток 11 см", desc: "великі деталі, читається з відстані", price: `від ${M[110]} ₴ за плитку` },
      { label: "Одна велика мапа XL 15 см", desc: "якщо панно не потрібне — одна цільна модель", price: `від ${M[150]} ₴` },
    ],
    h2how: "Як замовити",
    how: [
      "Відкрийте конструктор і знайдіть своє місто чи будь-яку точку світу.",
      "Увімкніть режим сітки та виділіть клітинки, які мають увійти в панно — ціна рахується одразу.",
      "Натисніть «Створити» — за кілька хвилин побачите 3D-превʼю всіх плиток.",
      "Замовте друк — виготовимо за 1–3 робочі дні й надішлемо Новою Поштою по Україні або в ЄС.",
    ],
    h2cities: "Популярні міста для панно",
    h2faq: "Часті запитання",
    faq: [
      { q: "Чим панно-карта краща за дерев'яну карту на стіну?", a: "Дерев'яні карти — це контур країни чи світу, однаковий у всіх. Панно-карта Monadruk — це ВАШЕ місце: конкретний район з реальними вулицями й будинками у 3D. Такого панно немає більше ні в кого." },
      { q: "Чи є підсвітка?", a: "Вбудованої підсвітки немає — панно складається з друкованих плиток. Але воно чудово виглядає під спрямованим світлом: рельєф вулиць дає глибокі тіні. Багато клієнтів підсвічують панно звичайною LED-стрічкою за периметром." },
      { q: "Як плитки кріпляться до стіни?", a: "Плитки легкі (Eco PLA), тримаються на двосторонньому спінені скотчі або монтажних смужках без свердління. Між собою з'єднуються пазами, тож панно висить рівно." },
      { q: "Можна замовити панно не з міста, а з гір чи моря?", a: "Так — будь-яка точка світу: узбережжя з лінією пляжу, карпатський хребет з рельєфом, озеро. Для гір увімкніть режим рельєфу, і висоти стануть об'ємними." },
      { q: "Скільки коштує і як довго чекати?", a: `Панно 2×2 з плиток 8 см — від ${M[80] * 4} ₴, панно 3×3 — від ${M[80] * 9} ₴. Виготовлення 1–3 робочі дні + доставка Новою Поштою. Оплата карткою онлайн або при отриманні.` },
    ],
    cta: "Зібрати своє панно в конструкторі",
    ctaSecondary: "Подивитись живі фото",
    photosAlt: "Фото реального 3D-друкованого панно-карти міста",
  },
  en: {
    title: "City map wall panel — buy a 3D tile map",
    description: `A wall map built from 3D-printed tiles: pick a district and we assemble a 2×2 or 3×3 panel with streets, blocks and rivers. From ≈€${eur(M[80] * 4)} for a 2×2 panel, EU shipping.`,
    h1: "3D city map wall panel",
    intro: "A wall map assembled from printed tiles: each tile is a fragment of the city with real streets, buildings and water — together they form a large panel of your neighbourhood.",
    introB: "Not a poster and not a wooden country outline — a precise three-dimensional model of a place that means something: your block, the riverfront, your hometown centre. Tiles connect with joints into one seamless piece.",
    h2what: "What it is",
    pWhat: [
      "You pick an area in the builder and split it into a grid — 2×2, 3×3 or any other. Each cell is printed as a separate Eco PLA tile; connector joints hold the tiles together on a wall or shelf.",
      "Streets, buildings with real heights, parks and rivers come from OpenStreetMap data. For hilly cities you can enable terrain relief — the elevation becomes touchable.",
    ],
    h2sizes: "Sizes and prices",
    sizes: [
      { label: "2×2 panel (four 8 cm tiles)", desc: "≈16×16 cm — compact panel for a shelf or small wall", price: `from ≈€${eur(M[80] * 4)}` },
      { label: "3×3 panel (nine 8 cm tiles)", desc: "≈24×24 cm — a proper wall statement", price: `from ≈€${eur(M[80] * 9)}` },
      { label: "Panel of 11 cm tiles", desc: "bigger details, readable from a distance", price: `from ≈€${eur(M[110])} per tile` },
      { label: "One large XL map, 15 cm", desc: "a single solid model if you don't need a panel", price: `from ≈€${eur(M[150])}` },
    ],
    h2how: "How to order",
    how: [
      "Open the builder and find your city or any point on Earth.",
      "Enable grid mode and select the cells for your panel — the price updates live.",
      "Click «Create» — in minutes you'll see a 3D preview of all tiles.",
      "Order the print — made in 1–3 business days and shipped to 15 EU countries.",
    ],
    h2cities: "Popular cities for a panel",
    h2faq: "FAQ",
    faq: [
      { q: "How is it better than a wooden wall map?", a: "Wooden maps are country or world outlines — identical for everyone. A Monadruk panel is YOUR place: a specific district with real streets and buildings in 3D. Nobody else has this panel." },
      { q: "Does it light up?", a: "There's no built-in lighting — the panel is made of printed tiles. It looks great under directed light though: street relief casts deep shadows. Many customers add a simple LED strip behind the perimeter." },
      { q: "How do tiles attach to the wall?", a: "Tiles are light (Eco PLA) and hold on double-sided foam tape or adhesive strips, no drilling. They interlock with joints, so the panel hangs straight." },
      { q: "Can I order mountains or a seashore instead of a city?", a: "Yes — any point on Earth: a coastline with the beach, a Carpathian ridge with relief, a lake. Enable terrain mode for mountains and the elevation becomes three-dimensional." },
      { q: "Price and lead time?", a: `A 2×2 panel of 8 cm tiles from ≈€${eur(M[80] * 4)}, 3×3 from ≈€${eur(M[80] * 9)}. Production 1–3 business days plus shipping. Card payment online.` },
    ],
    cta: "Build your panel in the builder",
    ctaSecondary: "See real photos",
    photosAlt: "Photo of a real 3D-printed city map wall panel",
  },
  de: {
    title: "3D-Stadtkarte für die Wand — Kachel-Panel kaufen",
    description: `Wandkarte aus 3D-gedruckten Kacheln: Viertel wählen — wir bauen ein 2×2- oder 3×3-Panel mit Straßen, Blöcken und Flüssen. Ab ≈${eur(M[80] * 4)} € für 2×2, EU-Versand.`,
    h1: "3D-Stadtkarten-Panel für die Wand",
    intro: "Eine Wandkarte aus gedruckten Kacheln: Jede Kachel ist ein Stück Stadt mit echten Straßen, Gebäuden und Wasser — zusammen ergeben sie ein großes Panel deines Viertels.",
    introB: "Kein Poster und kein Holz-Umriss eines Landes — ein präzises dreidimensionales Modell eines Ortes mit Bedeutung: dein Block, das Flussufer, das Zentrum deiner Heimatstadt. Die Kacheln verbinden sich über Steckverbindungen zu einem Ganzen.",
    h2what: "Was das ist",
    pWhat: [
      "Du wählst im Konfigurator einen Bereich und teilst ihn in ein Raster — 2×2, 3×3 oder beliebig. Jede Zelle wird als separate Eco-PLA-Kachel gedruckt; Verbindungen halten die Kacheln an Wand oder Regal zusammen.",
      "Straßen, Gebäude mit echten Höhen, Parks und Flüsse stammen aus OpenStreetMap-Daten. Für hügelige Städte lässt sich das Geländerelief aktivieren — die Höhen werden fühlbar.",
    ],
    h2sizes: "Größen und Preise",
    sizes: [
      { label: "2×2-Panel (vier 8-cm-Kacheln)", desc: "≈16×16 cm — kompaktes Panel für Regal oder kleine Wand", price: `ab ≈${eur(M[80] * 4)} €` },
      { label: "3×3-Panel (neun 8-cm-Kacheln)", desc: "≈24×24 cm — echtes Wand-Statement", price: `ab ≈${eur(M[80] * 9)} €` },
      { label: "Panel aus 11-cm-Kacheln", desc: "größere Details, aus der Distanz lesbar", price: `ab ≈${eur(M[110])} € pro Kachel` },
      { label: "Eine große XL-Karte, 15 cm", desc: "ein massives Einzelmodell, wenn kein Panel nötig ist", price: `ab ≈${eur(M[150])} €` },
    ],
    h2how: "So bestellst du",
    how: [
      "Öffne den Konfigurator und finde deine Stadt oder jeden Punkt der Erde.",
      "Aktiviere den Raster-Modus und wähle die Zellen für dein Panel — der Preis aktualisiert sich live.",
      "Klicke auf «Erstellen» — in Minuten siehst du die 3D-Vorschau aller Kacheln.",
      "Bestelle den Druck — Fertigung in 1–3 Werktagen, Versand in 15 EU-Länder.",
    ],
    h2cities: "Beliebte Städte für ein Panel",
    h2faq: "Häufige Fragen",
    faq: [
      { q: "Was ist besser als eine Holz-Weltkarte?", a: "Holzkarten sind Länder- oder Welt-Umrisse — für alle identisch. Ein Monadruk-Panel ist DEIN Ort: ein konkretes Viertel mit echten Straßen und Gebäuden in 3D. Dieses Panel hat sonst niemand." },
      { q: "Gibt es Beleuchtung?", a: "Keine eingebaute — das Panel besteht aus gedruckten Kacheln. Unter gerichtetem Licht wirkt es aber stark: das Straßenrelief wirft tiefe Schatten. Viele Kunden setzen eine LED-Leiste hinter den Rand." },
      { q: "Wie halten die Kacheln an der Wand?", a: "Die Kacheln sind leicht (Eco PLA) und halten mit doppelseitigem Schaumklebeband oder Klebestreifen — ohne Bohren. Untereinander verbinden sie sich mit Steckverbindungen." },
      { q: "Geht auch Gebirge oder Meer statt Stadt?", a: "Ja — jeder Punkt der Erde: eine Küste mit Strandlinie, ein Karpaten-Kamm mit Relief, ein See. Für Berge das Relief aktivieren, und die Höhen werden plastisch." },
      { q: "Preis und Lieferzeit?", a: `2×2-Panel aus 8-cm-Kacheln ab ≈${eur(M[80] * 4)} €, 3×3 ab ≈${eur(M[80] * 9)} €. Fertigung 1–3 Werktage plus Versand. Kartenzahlung online.` },
    ],
    cta: "Panel im Konfigurator bauen",
    ctaSecondary: "Echte Fotos ansehen",
    photosAlt: "Foto eines echten 3D-gedruckten Stadtkarten-Panels",
  },
  pl: {
    title: "Mapa miasta 3D na ścianę — panel z kafelków",
    description: `Mapa na ścianę z drukowanych kafelków 3D: wybierz dzielnicę — złożymy panel 2×2 lub 3×3 z ulicami, kwartałami i rzekami. Od ≈${eur(M[80] * 4)} € za panel 2×2, wysyłka do UE.`,
    h1: "Panel-mapa miasta 3D na ścianę",
    intro: "Ścienna mapa złożona z drukowanych kafelków: każdy kafelek to fragment miasta z prawdziwymi ulicami, budynkami i wodą — razem tworzą duży panel Twojej dzielnicy.",
    introB: "To nie plakat ani drewniany kontur kraju — to precyzyjny trójwymiarowy model miejsca, które coś znaczy: Twój kwartał, nabrzeże, centrum rodzinnego miasta. Kafelki łączą się zaczepami w jedną całość.",
    h2what: "Co to jest",
    pWhat: [
      "Wybierasz obszar w kreatorze i dzielisz go na siatkę — 2×2, 3×3 lub dowolną. Każda komórka jest drukowana jako osobny kafelek z Eco PLA; złącza trzymają kafelki razem na ścianie lub półce.",
      "Ulice, budynki o prawdziwych wysokościach, parki i rzeki pochodzą z danych OpenStreetMap. Dla pagórkowatych miast można włączyć rzeźbę terenu — różnice wysokości staną się wyczuwalne.",
    ],
    h2sizes: "Rozmiary i ceny",
    sizes: [
      { label: "Panel 2×2 (cztery kafelki 8 cm)", desc: "≈16×16 cm — kompaktowy panel na półkę lub małą ścianę", price: `od ≈${eur(M[80] * 4)} €` },
      { label: "Panel 3×3 (dziewięć kafelków 8 cm)", desc: "≈24×24 cm — pełnoprawny akcent ścienny", price: `od ≈${eur(M[80] * 9)} €` },
      { label: "Panel z kafelków 11 cm", desc: "większe detale, czytelne z dystansu", price: `od ≈${eur(M[110])} € za kafelek` },
      { label: "Jedna duża mapa XL 15 cm", desc: "pojedynczy model, jeśli panel nie jest potrzebny", price: `od ≈${eur(M[150])} €` },
    ],
    h2how: "Jak zamówić",
    how: [
      "Otwórz kreator i znajdź swoje miasto lub dowolny punkt świata.",
      "Włącz tryb siatki i zaznacz komórki panelu — cena liczy się na bieżąco.",
      "Kliknij «Utwórz» — w kilka minut zobaczysz podgląd 3D wszystkich kafelków.",
      "Zamów druk — wykonanie 1–3 dni robocze, wysyłka do 15 krajów UE.",
    ],
    h2cities: "Popularne miasta na panel",
    h2faq: "Częste pytania",
    faq: [
      { q: "Czym to lepsze od drewnianej mapy na ścianę?", a: "Drewniane mapy to kontury kraju lub świata — identyczne u wszystkich. Panel Monadruk to TWOJE miejsce: konkretna dzielnica z prawdziwymi ulicami i budynkami w 3D. Takiego panelu nie ma nikt inny." },
      { q: "Czy jest podświetlenie?", a: "Wbudowanego nie ma — panel składa się z drukowanych kafelków. Świetnie wygląda jednak w skierowanym świetle: relief ulic daje głębokie cienie. Wielu klientów dodaje taśmę LED za obwodem." },
      { q: "Jak kafelki trzymają się ściany?", a: "Kafelki są lekkie (Eco PLA), trzymają się na dwustronnej taśmie piankowej lub paskach montażowych, bez wiercenia. Między sobą łączą się zaczepami." },
      { q: "Można zamówić góry lub morze zamiast miasta?", a: "Tak — dowolny punkt świata: wybrzeże z linią plaży, grzbiet Karpat z rzeźbą terenu, jezioro. Dla gór włącz tryb rzeźby, a wysokości staną się przestrzenne." },
      { q: "Cena i czas realizacji?", a: `Panel 2×2 z kafelków 8 cm od ≈${eur(M[80] * 4)} €, 3×3 od ≈${eur(M[80] * 9)} €. Wykonanie 1–3 dni robocze plus wysyłka. Płatność kartą online.` },
    ],
    cta: "Złóż swój panel w kreatorze",
    ctaSecondary: "Zobacz prawdziwe zdjęcia",
    photosAlt: "Zdjęcie prawdziwego drukowanego panelu-mapy miasta 3D",
  },
  fr: {
    title: "Carte de ville 3D murale — panneau en tuiles",
    description: `Carte murale en tuiles imprimées en 3D : choisissez un quartier — nous assemblons un panneau 2×2 ou 3×3 avec rues, îlots et rivières. Dès ≈${eur(M[80] * 4)} € le panneau 2×2, livraison UE.`,
    h1: "Panneau-carte de ville 3D pour le mur",
    intro: "Une carte murale assemblée de tuiles imprimées : chaque tuile est un fragment de la ville avec de vraies rues, bâtiments et eau — ensemble elles forment un grand panneau de votre quartier.",
    introB: "Ni un poster, ni un contour de pays en bois — un modèle tridimensionnel précis d'un lieu qui compte : votre pâté de maisons, les quais, le centre de votre ville natale. Les tuiles s'emboîtent en un ensemble continu.",
    h2what: "Qu'est-ce que c'est",
    pWhat: [
      "Vous choisissez une zone dans le configurateur et la divisez en grille — 2×2, 3×3 ou autre. Chaque cellule est imprimée en tuile Eco PLA séparée ; des emboîtements tiennent les tuiles ensemble au mur ou sur une étagère.",
      "Rues, bâtiments aux hauteurs réelles, parcs et rivières proviennent des données OpenStreetMap. Pour les villes vallonnées, activez le relief — les dénivelés deviennent tangibles.",
    ],
    h2sizes: "Tailles et prix",
    sizes: [
      { label: "Panneau 2×2 (quatre tuiles 8 cm)", desc: "≈16×16 cm — panneau compact pour étagère ou petit mur", price: `dès ≈${eur(M[80] * 4)} €` },
      { label: "Panneau 3×3 (neuf tuiles 8 cm)", desc: "≈24×24 cm — un vrai accent mural", price: `dès ≈${eur(M[80] * 9)} €` },
      { label: "Panneau de tuiles 11 cm", desc: "détails plus grands, lisibles à distance", price: `dès ≈${eur(M[110])} € la tuile` },
      { label: "Une grande carte XL 15 cm", desc: "un modèle unique si vous ne voulez pas de panneau", price: `dès ≈${eur(M[150])} €` },
    ],
    h2how: "Comment commander",
    how: [
      "Ouvrez le configurateur et trouvez votre ville ou n'importe quel point du globe.",
      "Activez le mode grille et sélectionnez les cellules du panneau — le prix se calcule en direct.",
      "Cliquez sur «Créer» — en quelques minutes, l'aperçu 3D de toutes les tuiles apparaît.",
      "Commandez l'impression — fabrication en 1–3 jours ouvrés, livraison dans 15 pays de l'UE.",
    ],
    h2cities: "Villes populaires pour un panneau",
    h2faq: "Questions fréquentes",
    faq: [
      { q: "En quoi est-ce mieux qu'une carte murale en bois ?", a: "Les cartes en bois sont des contours de pays ou du monde — identiques pour tous. Un panneau Monadruk, c'est VOTRE lieu : un quartier précis avec de vraies rues et bâtiments en 3D. Personne d'autre n'a ce panneau." },
      { q: "Y a-t-il un éclairage ?", a: "Pas d'éclairage intégré — le panneau est fait de tuiles imprimées. Il rend très bien sous une lumière dirigée : le relief des rues crée des ombres profondes. Beaucoup ajoutent un ruban LED derrière le périmètre." },
      { q: "Comment les tuiles tiennent-elles au mur ?", a: "Les tuiles sont légères (Eco PLA) et tiennent avec du ruban mousse double face ou des languettes adhésives, sans percer. Entre elles, elles s'emboîtent." },
      { q: "Montagne ou mer au lieu d'une ville ?", a: "Oui — n'importe quel point du globe : un littoral avec la plage, une crête des Carpates en relief, un lac. Activez le mode relief pour la montagne." },
      { q: "Prix et délais ?", a: `Panneau 2×2 en tuiles 8 cm dès ≈${eur(M[80] * 4)} €, 3×3 dès ≈${eur(M[80] * 9)} €. Fabrication 1–3 jours ouvrés plus livraison. Paiement par carte en ligne.` },
    ],
    cta: "Assembler votre panneau",
    ctaSecondary: "Voir les photos réelles",
    photosAlt: "Photo d'un vrai panneau-carte de ville imprimé en 3D",
  },
  es: {
    title: "Mapa de ciudad 3D para pared — panel de azulejos",
    description: `Mapa de pared con azulejos impresos en 3D: elige un barrio — montamos un panel 2×2 o 3×3 con calles, manzanas y ríos. Desde ≈${eur(M[80] * 4)} € el panel 2×2, envío a la UE.`,
    h1: "Panel-mapa de ciudad 3D para la pared",
    intro: "Un mapa de pared montado con azulejos impresos: cada azulejo es un fragmento de la ciudad con calles, edificios y agua reales — juntos forman un gran panel de tu barrio.",
    introB: "No es un póster ni un contorno de madera de un país — es un modelo tridimensional preciso de un lugar con significado: tu manzana, el paseo del río, el centro de tu ciudad natal. Los azulejos encajan formando una pieza continua.",
    h2what: "Qué es",
    pWhat: [
      "Eliges una zona en el configurador y la divides en una cuadrícula — 2×2, 3×3 o la que quieras. Cada celda se imprime como un azulejo Eco PLA independiente; las uniones mantienen los azulejos juntos en la pared o estantería.",
      "Calles, edificios con alturas reales, parques y ríos provienen de datos de OpenStreetMap. Para ciudades con colinas, activa el relieve — los desniveles se vuelven táctiles.",
    ],
    h2sizes: "Tamaños y precios",
    sizes: [
      { label: "Panel 2×2 (cuatro azulejos de 8 cm)", desc: "≈16×16 cm — panel compacto para estantería o pared pequeña", price: `desde ≈${eur(M[80] * 4)} €` },
      { label: "Panel 3×3 (nueve azulejos de 8 cm)", desc: "≈24×24 cm — un auténtico protagonista de pared", price: `desde ≈${eur(M[80] * 9)} €` },
      { label: "Panel de azulejos de 11 cm", desc: "detalles más grandes, legibles a distancia", price: `desde ≈${eur(M[110])} € por azulejo` },
      { label: "Un mapa grande XL de 15 cm", desc: "un modelo único si no necesitas panel", price: `desde ≈${eur(M[150])} €` },
    ],
    h2how: "Cómo pedir",
    how: [
      "Abre el configurador y encuentra tu ciudad o cualquier punto del mundo.",
      "Activa el modo cuadrícula y selecciona las celdas del panel — el precio se calcula al momento.",
      "Pulsa «Crear» — en minutos verás la vista previa 3D de todos los azulejos.",
      "Pide la impresión — fabricación en 1–3 días hábiles, envío a 15 países de la UE.",
    ],
    h2cities: "Ciudades populares para un panel",
    h2faq: "Preguntas frecuentes",
    faq: [
      { q: "¿En qué es mejor que un mapa de madera?", a: "Los mapas de madera son contornos de países o del mundo — idénticos para todos. Un panel Monadruk es TU lugar: un barrio concreto con calles y edificios reales en 3D. Nadie más tiene este panel." },
      { q: "¿Tiene iluminación?", a: "No lleva iluminación integrada — el panel se compone de azulejos impresos. Luce muy bien con luz dirigida: el relieve de las calles crea sombras profundas. Muchos clientes añaden una tira LED por el perímetro." },
      { q: "¿Cómo se fijan los azulejos a la pared?", a: "Los azulejos son ligeros (Eco PLA) y se sujetan con cinta de espuma de doble cara o tiras adhesivas, sin taladrar. Entre sí encajan con uniones." },
      { q: "¿Puedo pedir montañas o costa en vez de ciudad?", a: "Sí — cualquier punto del mundo: una costa con la línea de playa, una cresta de los Cárpatos con relieve, un lago. Activa el modo relieve para montañas." },
      { q: "¿Precio y plazos?", a: `Panel 2×2 de azulejos de 8 cm desde ≈${eur(M[80] * 4)} €, 3×3 desde ≈${eur(M[80] * 9)} €. Fabricación 1–3 días hábiles más envío. Pago con tarjeta online.` },
    ],
    cta: "Montar tu panel en el configurador",
    ctaSecondary: "Ver fotos reales",
    photosAlt: "Foto de un panel-mapa de ciudad real impreso en 3D",
  },
};

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  const c = COPY[locale];
  const path = "/panno";
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

export default async function PannoPage({ params }: { params: { locale: string } }) {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  setRequestLocale(locale);
  const c = COPY[locale];
  const isUA = locale === "uk";
  const path = "/panno";

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Product",
        name: c.h1,
        description: c.description,
        image: [`${BASE}/real/panno-1.webp`, `${BASE}/real/panno-2.webp`, `${BASE}/real/panno-3.webp`],
        brand: { "@type": "Brand", name: "Monadruk" },
        sku: "MND-PANNO",
        offers: {
          "@type": "AggregateOffer",
          priceCurrency: isUA ? "UAH" : "EUR",
          lowPrice: isUA ? String(M[80] * 4) : String(eur(M[80] * 4)),
          highPrice: isUA ? String(M[110] * 9) : String(eur(M[110] * 9)),
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
        <Link href="/showcase" className="inline-flex min-h-[44px] items-center justify-center rounded-[22px] border border-line-soft bg-white/80 px-5 py-2.5 text-sm font-semibold text-ink transition hover:border-[var(--accent)]">
          {c.ctaSecondary}
        </Link>
      </section>

      {/* Живі фото реальних друків — E-E-A-T «справжній бізнес», не рендери */}
      <section className="mt-10 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {["panno-1", "panno-2", "panno-3", "panno-4"].map((img) => (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            key={img}
            src={`/real/${img}.webp`}
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
        <h2 className="text-[20px] font-semibold">{c.h2sizes}</h2>
        <ul className="mt-4 grid gap-3 sm:grid-cols-2">
          {c.sizes.map((s) => (
            <li key={s.label} className="rounded-[18px] border border-line-soft bg-white/70 px-5 py-4">
              <p className="text-[15px] font-semibold text-ink">{s.label}</p>
              <p className="mt-1 text-[13.5px] leading-relaxed text-ink-2">{s.desc}</p>
              <p className="mt-2 text-[15px] font-semibold text-[var(--accent-strong)]">{s.price}</p>
            </li>
          ))}
        </ul>
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

      <section className="mt-12">
        <h2 className="text-[20px] font-semibold">{c.h2cities}</h2>
        <ul className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {CITY_PAGES.slice(0, 12).map((city) => (
            <li key={city.slug}>
              <Link
                href={`/maps/${city.slug}`}
                className="block rounded-[18px] border border-line-soft bg-white/70 px-4 py-3.5 text-[15px] font-semibold text-ink transition hover:border-[var(--accent)]"
              >
                {city.names[locale]}
              </Link>
            </li>
          ))}
        </ul>
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
