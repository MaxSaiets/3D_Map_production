// ──────────────────────────────────────────────────────────────────────────
// Переклади статей блогу на de/pl/fr/es (2026-09-24, «просування — всі мови»).
// Не дослівно з en, а локалізовано: ціни в ≈€ (EUR_PER_UAH), чесно — друк
// доставляємо лише по Україні, з-за кордону — файл 3MF (149 ₴ ≈ 3 €) і друк
// на своєму принтері / у локальному 3D-сервісі. Нових фактів не додаємо.
// Зливається в BLOG_ARTICLES у lib/blog.ts → статті з перекладом стають
// індексованими (blogLocales), без перекладу — noindex + canonical на en.
// ──────────────────────────────────────────────────────────────────────────
import type { BlogArticleContent } from "@/lib/blog";

export type BlogExtraLocale = "de" | "pl" | "fr" | "es";

export const BLOG_I18N_EXTRA: Record<string, Partial<Record<BlogExtraLocale, BlogArticleContent>>> = {
  "shcho-podaruvaty-khloptsevi-na-den-narodzhennya": {
    de: {
      title: "Geburtstagsgeschenk für den Freund: 3D-Karte seines Viertels",
      description: "Eine Geschenkidee, die sonst niemand hat: eine 3D-Karte des Viertels, in dem er aufgewachsen ist, oder ein Anhänger mit seiner Straße. Druckdatei ≈3 €.",
      h1: "Was schenkt man dem Freund zum Geburtstag, wenn Socken schon vergeben sind?",
      intro: "Gadgets kauft er selbst, Parfüm ist Glückssache, und ein Erlebnisgutschein ist nach einer Woche vergessen. Was wirkt, ist etwas, in dem ER steckt: eine 3D-Karte des Viertels, in dem er aufgewachsen ist, oder des Blocks, wo ihr euch kennengelernt habt — mit echten Gebäuden, Straßen und dem Park, den er sofort wiedererkennt.",
      sections: [
        { h2: "Warum eine Karte", p: [
          "Es ist kein Regal-Souvenir: Das Modell gibt es nur einmal, weil es für bestimmte Koordinaten entsteht. Auf dem Schreibtisch wird es sofort zum Gesprächsthema.",
          "Eine Karte hat keine Kleidergröße und kein Geschmacksrisiko: Ein Ort, der etwas bedeutet, trifft immer.",
        ] },
        { h2: "Drei Varianten für jedes Budget", p: [
          "Karten-Anhänger (ab ≈4 €) — seine Straße immer am Schlüsselbund, auf Wunsch mit Gravur auf der Rückseite.",
          "Kühlschrankmagnet (≈5 €) — eine kompakte 6-cm-Karte, wenn es klein, aber persönlich sein soll.",
          "3D-Viertelkarte (ab ≈8 €) — das Hauptgeschenk: Gebäude in echter Höhe, Parks, Fluss. 5,5 bis 15 cm.",
        ] },
        { h2: "In 5 Minuten erstellt", p: [
          "Konfigurator öffnen, Adresse suchen oder Stadt antippen, Rahmen über den Block ziehen — nach 1–2 Minuten erscheint eine kostenlose 3D-Vorschau.",
          "Dann zwei Wege: Druck bestellen (Versand innerhalb der Ukraine) oder die 3MF-Datei (149 ₴ ≈ 3 €) herunterladen und selbst oder in einem 3D-Druckservice in deiner Nähe drucken.",
        ] },
        { h2: "Persönlich machen", p: [
          "Markiere sein Haus mit einem roten Einsatz, füge einen Namen, ein Jahr oder Koordinaten hinzu. Für Paare gibt es einen zweiteiligen Herz-Anhänger: sein Viertel und deins.",
        ] },
      ],
      ctaLabel: "Karte seines Viertels erstellen",
      ctaHref: "/create",
      outro: "Druck und Versand dauern 2–4 Werktage. Ist der Geburtstag schon morgen — nimm die Datei und lass sie vor Ort drucken, oder bestelle einen Anhänger, er ist am schnellsten gedruckt.",
    },
    pl: {
      title: "Prezent urodzinowy dla chłopaka: mapa 3D jego dzielnicy",
      description: "Pomysł na prezent, którego nikt inny nie ma: mapa 3D dzielnicy, w której dorastał, albo brelok z jego ulicą. Plik do druku ≈3 €.",
      h1: "Co dać chłopakowi na urodziny, gdy skarpetki już były?",
      intro: "Gadżety kupuje sam, perfumy to loteria, a „voucher na przeżycie” znika z pamięci po tygodniu. Działa rzecz, w której jest ON: mapa 3D dzielnicy, w której dorastał, albo kwartału, gdzie się poznaliście — z prawdziwymi budynkami, ulicami i parkiem, który rozpozna od razu.",
      sections: [
        { h2: "Dlaczego mapa", p: [
          "To nie pamiątka na półkę: model istnieje w jednym egzemplarzu, bo powstaje dla konkretnych współrzędnych. Na biurku od razu staje się tematem rozmowy.",
          "Mapa nie ma rozmiaru ani ryzyka nietrafionego gustu: miejsce, które coś znaczy, zawsze trafia.",
        ] },
        { h2: "Trzy opcje na każdy budżet", p: [
          "Brelok z mapą (od ≈4 €) — jego ulica zawsze przy kluczach, z opcjonalnym grawerem na odwrocie.",
          "Magnes na lodówkę (≈5 €) — kompaktowa mapa 6 cm, gdy chcesz czegoś małego, ale osobistego.",
          "Mapa 3D dzielnicy (od ≈8 €) — główny prezent: budynki w prawdziwej wysokości, parki, rzeka. Od 5,5 do 15 cm.",
        ] },
        { h2: "Jak zrobić to w 5 minut", p: [
          "Otwórz kreator, znajdź adres albo kliknij miasto, przeciągnij ramkę na kwartał — darmowy podgląd 3D pojawi się po 1–2 minutach.",
          "Potem dwie drogi: zamów wydruk (wysyłka na terenie Ukrainy) albo pobierz plik 3MF (149 ₴ ≈ 3 €) i wydrukuj sam lub w lokalnym serwisie druku 3D.",
        ] },
        { h2: "Niech będzie jego", p: [
          "Zaznacz jego dom czerwoną wstawką, dodaj imię, rok lub współrzędne. Dla par jest brelok-serce z dwóch połówek: jego dzielnica i twoja.",
        ] },
      ],
      ctaLabel: "Stwórz mapę jego dzielnicy",
      ctaHref: "/create",
      outro: "Druk i wysyłka trwają 2–4 dni robocze. Jeśli urodziny są jutro — weź plik i wydrukuj go na miejscu albo zamów brelok, drukuje się najszybciej.",
    },
    fr: {
      title: "Cadeau d'anniversaire pour son copain : la carte 3D de son quartier",
      description: "Une idée de cadeau que personne d'autre n'a : la carte 3D du quartier où il a grandi, ou un porte-clés avec sa rue. Fichier à imprimer ≈3 €.",
      h1: "Quoi offrir à son copain pour son anniversaire quand les chaussettes, c'est déjà fait ?",
      intro: "Les gadgets, il les achète lui-même, le parfum est une loterie et un « bon pour une expérience » s'oublie en une semaine. Ce qui marche, c'est un objet qui le contient, LUI : la carte 3D du quartier où il a grandi, ou du pâté de maisons où vous vous êtes rencontrés — avec les vrais bâtiments, les rues et le parc qu'il reconnaît au premier coup d'œil.",
      sections: [
        { h2: "Pourquoi une carte", p: [
          "Ce n'est pas un souvenir d'étagère : le modèle n'existe qu'une fois, car il est fait pour des coordonnées précises. Sur un bureau, il devient tout de suite un sujet de conversation.",
          "Une carte n'a ni taille ni risque de faute de goût : un lieu qui compte fait toujours mouche.",
        ] },
        { h2: "Trois options pour tous les budgets", p: [
          "Porte-clés carte (dès ≈4 €) — sa rue toujours sur ses clés, avec une gravure au dos si tu veux.",
          "Aimant de frigo (≈5 €) — une carte compacte de 6 cm pour un petit geste personnel.",
          "Carte 3D du quartier (dès ≈8 €) — le cadeau principal : bâtiments à leur vraie hauteur, parcs, rivière. De 5,5 à 15 cm.",
        ] },
        { h2: "Le faire en 5 minutes", p: [
          "Ouvre le configurateur, cherche l'adresse ou touche une ville, fais glisser le cadre sur le pâté de maisons — un aperçu 3D gratuit apparaît en 1 à 2 minutes.",
          "Ensuite, deux options : commander l'impression (livraison en Ukraine) ou télécharger le fichier 3MF (149 ₴ ≈ 3 €) et l'imprimer toi-même ou dans un service d'impression 3D près de chez toi.",
        ] },
        { h2: "Le rendre unique", p: [
          "Marque sa maison avec un insert rouge, ajoute un prénom, une année ou des coordonnées. Pour les couples, il existe un porte-clés cœur en deux moitiés : son quartier et le tien.",
        ] },
      ],
      ctaLabel: "Créer la carte de son quartier",
      ctaHref: "/create",
      outro: "Impression et expédition : 2–4 jours ouvrés. Si l'anniversaire est demain, prends le fichier et fais-le imprimer sur place, ou commande un porte-clés : c'est le plus rapide à imprimer.",
    },
    es: {
      title: "Regalo de cumpleaños para tu novio: un mapa 3D de su barrio",
      description: "Una idea de regalo que nadie más tiene: un mapa 3D del barrio donde creció o un llavero con su calle. Archivo para imprimir ≈3 €.",
      h1: "Qué regalarle a tu novio por su cumpleaños cuando los calcetines ya están vistos",
      intro: "Los gadgets se los compra él, el perfume es una lotería y un «bono de experiencia» se olvida en una semana. Funciona algo que lo lleve a ÉL dentro: un mapa 3D del barrio donde creció o de la manzana donde os conocisteis, con los edificios reales, las calles y el parque que reconoce a primera vista.",
      sections: [
        { h2: "Por qué un mapa", p: [
          "No es un recuerdo de estantería: el modelo existe una sola vez porque se hace para unas coordenadas concretas. En el escritorio se convierte enseguida en tema de conversación.",
          "Un mapa no tiene talla ni riesgo de fallar con el gusto: un lugar importante siempre acierta.",
        ] },
        { h2: "Tres opciones para cualquier presupuesto", p: [
          "Llavero con mapa (desde ≈4 €): su calle siempre en las llaves, con grabado opcional en la parte de atrás.",
          "Imán de nevera (≈5 €): un mapa compacto de 6 cm cuando quieres algo pequeño pero personal.",
          "Mapa 3D del barrio (desde ≈8 €): el regalo principal, con edificios a su altura real, parques y río. De 5,5 a 15 cm.",
        ] },
        { h2: "Cómo hacerlo en 5 minutos", p: [
          "Abre el configurador, busca la dirección o toca una ciudad, arrastra el marco sobre la manzana y en 1–2 minutos aparece una vista previa 3D gratis.",
          "Después, dos caminos: pedir la impresión (envío dentro de Ucrania) o descargar el archivo 3MF (149 ₴ ≈ 3 €) e imprimirlo tú o en un servicio de impresión 3D cercano.",
        ] },
        { h2: "Hazlo suyo", p: [
          "Marca su casa con una pieza roja, añade un nombre, un año o unas coordenadas. Para parejas hay un llavero corazón de dos mitades: su barrio y el tuyo.",
        ] },
      ],
      ctaLabel: "Crear el mapa de su barrio",
      ctaHref: "/create",
      outro: "La impresión y el envío tardan 2–4 días hábiles. Si el cumpleaños es mañana, descarga el archivo e imprímelo cerca, o pide un llavero: es lo que se imprime más rápido.",
    },
  },

  "podarunok-divchyni-na-richnytsyu": {
    de: {
      title: "Jahrestagsgeschenk für die Freundin: Karte vom Ort, an dem alles begann",
      description: "Was schenkt man der Freundin zum Jahrestag? Eine 3D-Karte vom Block eures ersten Dates oder zwei Herz-Anhänger mit euren beiden Vierteln. Mit persönlicher Gravur.",
      h1: "Jahrestagsgeschenk für sie: der Ort, an dem alles begann",
      intro: "Blumen welken, Koordinaten bleiben. Eine Karte vom Block, wo ihr euch zum ersten Mal getroffen habt, vom Café an der Ecke oder von der Parkbank — ein Geschenk, das eure Geschichte ohne Worte erzählt.",
      sections: [
        { h2: "Zwei Ideen, die am besten funktionieren", p: [
          "Eine 3D-Karte vom ersten Date: ein Viertel von 400–800 m mit echten Gebäuden, dem Park und der Straße, durch die ihr gegangen seid. Aufs Regal gestellt, bringt jeder Blick diesen Tag zurück.",
          "Ein zweiteiliger Herz-Anhänger: Die eine Hälfte ist ihr Viertel, die andere deins. Zusammen ergeben sie ein ganzes Herz; getrennt trägt jede ihren Teil.",
        ] },
        { h2: "Eine Gravur, die es zu eurem macht", p: [
          "Das Kennenlerndatum, ihr Name oder die Koordinaten des Ortes — auf einer flachen Karte, einem Magneten oder der Rückseite des Anhängers. Mit einem Klick im Konfigurator hinzugefügt.",
        ] },
        { h2: "Preis und Dauer", p: [
          "Anhänger ab ≈4 €; ein Herz-Paar sind zwei Anhänger. 3D-Karte von ≈8 € (5,5 cm) bis ≈18 € (15 cm). Druck und Versand innerhalb der Ukraine in 2–4 Werktagen; außerhalb der Ukraine kaufst du die 3MF-Datei (≈3 €) und druckst sie vor Ort.",
        ] },
      ],
      ctaLabel: "Karte eures Ortes erstellen",
      ctaHref: "/podarunok/na-richnytsyu",
      outro: "Du kennst die genaue Adresse nicht? Ein Straßen- oder Café-Name reicht — die Kartensuche schlägt ihn vor, und den Rahmen kannst du von Hand verschieben.",
    },
    pl: {
      title: "Prezent na rocznicę dla dziewczyny: mapa miejsca, gdzie wszystko się zaczęło",
      description: "Co dać dziewczynie na rocznicę? Mapę 3D kwartału pierwszej randki albo parę breloków-serc z waszymi dzielnicami. Z osobistym grawerem.",
      h1: "Prezent na rocznicę dla niej: miejsce, gdzie wszystko się zaczęło",
      intro: "Kwiaty więdną, współrzędne zostają. Mapa kwartału, gdzie spotkaliście się pierwszy raz, kawiarni na rogu czy ławki w parku — prezent, który opowiada waszą historię bez słów.",
      sections: [
        { h2: "Dwa pomysły, które działają najlepiej", p: [
          "Mapa 3D pierwszej randki: dzielnica 400–800 m z prawdziwymi budynkami, parkiem i ulicą, którą szliście. Postawiona na półce przywołuje ten dzień przy każdym spojrzeniu.",
          "Brelok-serce z dwóch połówek: jedna to jej dzielnica, druga twoja. Razem tworzą całe serce, osobno każde nosi swoją.",
        ] },
        { h2: "Grawer, który czyni go waszym", p: [
          "Data poznania, jej imię albo współrzędne miejsca — na płaskiej mapie, magnesie lub odwrocie breloka. Dodajesz jednym kliknięciem w kreatorze.",
        ] },
        { h2: "Cena i czas", p: [
          "Brelok od ≈4 €; para serc to dwa breloki. Mapa 3D od ≈8 € (5,5 cm) do ≈18 € (15 cm). Druk i wysyłka na terenie Ukrainy w 2–4 dni robocze; poza Ukrainą kupujesz plik 3MF (≈3 €) i drukujesz na miejscu.",
        ] },
      ],
      ctaLabel: "Stwórz mapę waszego miejsca",
      ctaHref: "/podarunok/na-richnytsyu",
      outro: "Nie znasz dokładnego adresu? Wystarczy nazwa ulicy lub kawiarni — wyszukiwarka mapy ją podpowie, a ramkę przesuniesz ręcznie.",
    },
    fr: {
      title: "Cadeau d'anniversaire de couple pour elle : la carte du lieu où tout a commencé",
      description: "Quoi offrir à sa copine pour un anniversaire de couple ? La carte 3D du quartier du premier rendez-vous ou deux porte-clés cœur avec vos deux quartiers. Gravure personnalisée.",
      h1: "Cadeau d'anniversaire pour elle : le lieu où tout a commencé",
      intro: "Les fleurs fanent, les coordonnées restent. La carte du pâté de maisons de votre première rencontre, du café du coin ou du banc dans le parc — un cadeau qui raconte votre histoire sans un mot.",
      sections: [
        { h2: "Deux idées qui marchent le mieux", p: [
          "Une carte 3D du premier rendez-vous : un quartier de 400 à 800 m avec les vrais bâtiments, le parc et la rue où vous avez marché. Posée sur une étagère, chaque regard ramène ce jour-là.",
          "Un porte-clés cœur en deux moitiés : l'une est son quartier, l'autre le tien. Ensemble, elles forment un cœur entier ; séparées, chacun garde le sien.",
        ] },
        { h2: "Une gravure qui le rend unique", p: [
          "La date de votre rencontre, son prénom ou les coordonnées du lieu — sur une carte plate, un aimant ou au dos du porte-clés. Ajoutée en un clic dans le configurateur.",
        ] },
        { h2: "Prix et délais", p: [
          "Porte-clés dès ≈4 € ; une paire de cœurs, c'est deux porte-clés. Carte 3D de ≈8 € (5,5 cm) à ≈18 € (15 cm). Impression et livraison en Ukraine en 2–4 jours ouvrés ; hors d'Ukraine, tu achètes le fichier 3MF (≈3 €) et tu l'imprimes près de chez toi.",
        ] },
      ],
      ctaLabel: "Créer la carte de votre lieu",
      ctaHref: "/podarunok/na-richnytsyu",
      outro: "Tu ne connais pas l'adresse exacte ? Un nom de rue ou de café suffit — la recherche de la carte le propose, et le cadre se déplace à la main.",
    },
    es: {
      title: "Regalo de aniversario para tu novia: el mapa del lugar donde todo empezó",
      description: "Qué regalarle a tu novia por vuestro aniversario: un mapa 3D de la manzana de la primera cita o un par de llaveros corazón con vuestros dos barrios. Con grabado personal.",
      h1: "Regalo de aniversario para ella: el lugar donde todo empezó",
      intro: "Las flores se marchitan, las coordenadas quedan. El mapa de la manzana donde os visteis por primera vez, de la cafetería de la esquina o del banco del parque: un regalo que cuenta vuestra historia sin palabras.",
      sections: [
        { h2: "Dos ideas que funcionan mejor", p: [
          "Un mapa 3D de la primera cita: un barrio de 400–800 m con los edificios reales, el parque y la calle por la que paseasteis. En una estantería, cada mirada trae de vuelta ese día.",
          "Un llavero corazón de dos mitades: una es su barrio y la otra el tuyo. Juntas forman un corazón entero; por separado, cada uno lleva el suyo.",
        ] },
        { h2: "Un grabado que lo hace vuestro", p: [
          "La fecha en que os conocisteis, su nombre o las coordenadas del lugar, en un mapa plano, un imán o el reverso del llavero. Se añade con un clic en el configurador.",
        ] },
        { h2: "Precio y plazos", p: [
          "Llavero desde ≈4 €; un par de corazones son dos llaveros. Mapa 3D de ≈8 € (5,5 cm) a ≈18 € (15 cm). Impresión y envío dentro de Ucrania en 2–4 días hábiles; fuera de Ucrania compras el archivo 3MF (≈3 €) y lo imprimes cerca de ti.",
        ] },
      ],
      ctaLabel: "Crear el mapa de vuestro lugar",
      ctaHref: "/podarunok/na-richnytsyu",
      outro: "¿No sabes la dirección exacta? Basta con el nombre de la calle o de la cafetería: el buscador del mapa lo sugiere y el marco se mueve a mano.",
    },
  },

  "podarunok-batkam-na-richnytsyu-vesillya": {
    de: {
      title: "Hochzeitstagsgeschenk für die Eltern: 3D-Karte ihrer ersten Wohnung",
      description: "Geschenkidee zum Hochzeitstag der Eltern: eine 3D-Karte des Viertels, in dem alles anfing, oder ihrer Geburtsstadt. Ein Kachelbild für große Jubiläen.",
      h1: "Ein Geschenk zum Hochzeitstag der Eltern, das nicht im Schrank verstaubt",
      intro: "Eltern „haben schon alles“, deshalb gewinnen Erinnerungen gegen Dinge. Eine Karte des Viertels ihrer ersten Mietwohnung, der Straße zum Standesamt oder der Heimatstadt, die sie verlassen haben — ein Geschenk, über das sie sich lange beugen und das sie Gästen zeigen.",
      sections: [
        { h2: "Das passende Format zum Jubiläum", p: [
          "Für einen gewöhnlichen Hochzeitstag — eine 3D-Karte M oder L (8–11 cm) fürs Regal. Für 25 oder 30 Jahre — ein Wandbild aus mehreren Kacheln: Das Viertel wird zum Bild.",
          "Wurden sie in verschiedenen Städten geboren — zwei „halbe“ Karten oder zwei Magnete am selben Kühlschrank.",
        ] },
        { h2: "Was man schreibt", p: [
          "Hochzeitsdatum und Jubiläumsjahr sind am einfachsten und genauesten. Der Text kommt auf eine flache Karte oder einen Magneten; bei einer 3D-Karte als separate Einlegeplatte.",
        ] },
        { h2: "Bestellen aus einer anderen Stadt oder einem anderen Land", p: [
          "Der Konfigurator funktioniert mit jeder Adresse: Straße suchen, Rahmen setzen, 3D-Vorschau prüfen. Leben die Eltern in der Ukraine, liefert Nova Poshta direkt in ihre Filiale. Leben sie anderswo, kaufst du die 3MF-Datei (≈3 €) und lässt sie vor Ort drucken.",
        ] },
      ],
      ctaLabel: "Karte für die Eltern erstellen",
      ctaHref: "/create",
      outro: "Tipp: Schreib eine kurze Nachricht in den Bestellkommentar — wir legen sie in die Schachtel.",
    },
    pl: {
      title: "Prezent dla rodziców na rocznicę ślubu: mapa 3D ich pierwszego mieszkania",
      description: "Pomysł na prezent na rocznicę ślubu rodziców: mapa 3D dzielnicy, gdzie wszystko się zaczęło, albo miasta, w którym się urodzili. Panel z płytek na duże jubileusze.",
      h1: "Prezent na rocznicę ślubu rodziców, który nie będzie się kurzył w szafie",
      intro: "Rodzice „mają wszystko”, więc wspomnienia wygrywają z rzeczami. Mapa dzielnicy ich pierwszego wynajętego mieszkania, ulicy, którą szli do urzędu stanu cywilnego, albo rodzinnego miasta, które opuścili — prezent, nad którym długo się pochylą i który pokażą gościom.",
      sections: [
        { h2: "Format na jubileusz", p: [
          "Na zwykłą rocznicę — mapa 3D M lub L (8–11 cm) na półkę. Na 25 czy 30 lat — panel ścienny z kilku płytek: dzielnica staje się obrazem.",
          "Jeśli urodzili się w różnych miastach — dwie „połówkowe” mapy albo dwa magnesy na jednej lodówce.",
        ] },
        { h2: "Co napisać", p: [
          "Data ślubu i rok jubileuszu to najprostsze i najtrafniejsze rozwiązanie. Tekst trafia na płaską mapę lub magnes; na mapie 3D — jako osobna płytka.",
        ] },
        { h2: "Zamówienie z innego miasta lub kraju", p: [
          "Kreator działa z każdym adresem: znajdź ich ulicę, ustaw ramkę, sprawdź podgląd 3D. Jeśli rodzice mieszkają na Ukrainie, Nowa Poczta dostarczy prosto do ich oddziału. Jeśli gdzie indziej — kupujesz plik 3MF (≈3 €) i drukujesz go na miejscu.",
        ] },
      ],
      ctaLabel: "Stwórz mapę dla rodziców",
      ctaHref: "/create",
      outro: "Wskazówka: dopisz krótką wiadomość w komentarzu do zamówienia — włożymy ją do pudełka.",
    },
    fr: {
      title: "Cadeau d'anniversaire de mariage pour ses parents : la carte 3D de leur premier logement",
      description: "Idée cadeau pour l'anniversaire de mariage des parents : la carte 3D du quartier où tout a commencé, ou de leur ville natale. Un tableau en tuiles pour les grands jubilés.",
      h1: "Un cadeau d'anniversaire de mariage qui ne prendra pas la poussière",
      intro: "Les parents « ont déjà tout », alors les souvenirs l'emportent sur les objets. La carte du quartier de leur premier appartement, de la rue qui menait à la mairie ou de la ville natale qu'ils ont quittée — un cadeau sur lequel ils se pencheront longtemps et qu'ils montreront aux invités.",
      sections: [
        { h2: "Un format pour le jubilé", p: [
          "Pour un anniversaire ordinaire — une carte 3D M ou L (8–11 cm) pour l'étagère. Pour 25 ou 30 ans — un tableau mural en plusieurs tuiles : le quartier devient une œuvre.",
          "S'ils sont nés dans des villes différentes — deux cartes « moitiés » ou deux aimants sur le même frigo.",
        ] },
        { h2: "Quoi écrire", p: [
          "La date du mariage et l'année du jubilé, c'est le plus simple et le plus juste. Le texte va sur une carte plate ou un aimant ; sur une carte 3D, sur une plaque à part.",
        ] },
        { h2: "Commander depuis une autre ville ou un autre pays", p: [
          "Le configurateur accepte n'importe quelle adresse : trouve leur rue, place le cadre, vérifie l'aperçu 3D. S'ils vivent en Ukraine, Nova Poshta livre directement à leur point relais. Sinon, tu achètes le fichier 3MF (≈3 €) et tu le fais imprimer sur place.",
        ] },
      ],
      ctaLabel: "Créer une carte pour tes parents",
      ctaHref: "/create",
      outro: "Astuce : ajoute un petit mot dans le commentaire de commande — nous le glisserons dans la boîte.",
    },
    es: {
      title: "Regalo de aniversario de boda para tus padres: un mapa 3D de su primera casa",
      description: "Idea de regalo para el aniversario de boda de tus padres: un mapa 3D del barrio donde empezaron o de la ciudad donde nacieron. Un panel de baldosas para grandes aniversarios.",
      h1: "Un regalo de aniversario para tus padres que no acabará en un armario",
      intro: "Los padres «ya lo tienen todo», así que los recuerdos ganan a las cosas. El mapa del barrio de su primer piso de alquiler, de la calle por la que fueron al registro civil o de la ciudad natal que dejaron: un regalo sobre el que se inclinarán mucho rato y que enseñarán a las visitas.",
      sections: [
        { h2: "Un formato para el aniversario", p: [
          "Para un aniversario normal, un mapa 3D M o L (8–11 cm) para la estantería. Para 25 o 30 años, un panel de pared de varias baldosas: el barrio se convierte en un cuadro.",
          "Si nacieron en ciudades distintas, dos mapas «mitad» o dos imanes en la misma nevera.",
        ] },
        { h2: "Qué escribir", p: [
          "La fecha de la boda y el año del aniversario es lo más sencillo y preciso. El texto va en un mapa plano o un imán; en un mapa 3D, como una placa aparte.",
        ] },
        { h2: "Pedir desde otra ciudad u otro país", p: [
          "El configurador funciona con cualquier dirección: busca su calle, coloca el marco y revisa la vista previa 3D. Si tus padres viven en Ucrania, Nova Poshta lo entrega en su oficina. Si viven en otro sitio, compras el archivo 3MF (≈3 €) y lo imprimes allí.",
        ] },
      ],
      ctaLabel: "Crear un mapa para tus padres",
      ctaHref: "/create",
      outro: "Consejo: añade una nota breve en el comentario del pedido y la meteremos en la caja.",
    },
  },
};
