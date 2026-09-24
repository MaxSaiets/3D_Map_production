// Переклади статей блогу de/pl/fr/es — частина 4 (правила — див. lib/blogI18nExtra.ts).
import type { BlogArticleContent } from "@/lib/blog";
import type { BlogExtraLocale } from "@/lib/blogI18nExtra";

export const BLOG_I18N_EXTRA_4: Record<string, Partial<Record<BlogExtraLocale, BlogArticleContent>>> = {
  "podarunok-na-richnytsyu-stosunkiv": {
    de: {
      title: "Geschenk zum Jahrestag der Beziehung: eine Karte eures ersten Dates",
      description: "Geschenkidee zum Jahrestag für Freund oder Freundin: eine 3D-Karte oder ein Anhänger mit dem Ort eures ersten Dates und eingraviertem Datum. Anhänger ab ≈4 €.",
      h1: "Ein Jahrestagsgeschenk, das an einen einzigen Ort gebunden ist",
      intro: "Jahrestagsgeschenke tragen meist eine Aufschrift wie „unser Datum“ oder „unsere Namen“ — und lassen sich leicht mit einem Dutzend ähnlicher verwechseln. Eine Karte wirkt anders: Sie zeigt den genauen Ort, an dem alles begann — das Café, den Park, die Straße oder die U-Bahn-Station eures ersten Dates. Kein abstraktes Beziehungssymbol, sondern ein Stück Geografie, das nur ihr erkennt.",
      sections: [
        { h2: "Warum der Ort des ersten Dates und nicht nur eine Aufschrift", p: [
          "Ein „Ich liebe dich“ passt auf jedes Geschenk. Eine Karte des Blocks, in dem ihr euch zum ersten Mal getroffen habt, kann man nicht fertig kaufen — es gibt sie nur für diese eine Geschichte.",
          "Sie ist auch eine Gelegenheit, sich an halb vergessene Details zu erinnern: den Straßennamen, die Form des Café-Gebäudes, einen Park in der Nähe — alles als echter Grundriss auf der Karte.",
        ] },
        { h2: "Geschenkformate", p: [
          "Ein Anhänger mit der Karte des Treffpunkts (ab ≈4 €) — auch als Paar: einer für ihn, einer für sie, mit passender Gravur oder Datum.",
          "Ein Kühlschrankmagnet (≈5 €) — eine leichte Ergänzung zum Hauptgeschenk oder eine eigene Variante.",
          "Eine 3D-Tischkarte in Größe S oder M (≈8 oder ≈12 €) — fürs Regal oder den Nachttisch, mit mehr Details rund um den Treffpunkt.",
          "Soll auch das Gelände sichtbar sein (etwa ein Park auf einem Hügel oder eine Uferpromenade), gibt es die Relief-Option für ≈2 €.",
        ] },
        { h2: "In 5 Minuten erstellt", p: [
          "Produkt wählen — Anhänger oder Karte —, den Ort eures ersten Dates (Café, Park, Straße) auf der Karte finden, die kostenlose 3D-Vorschau bestätigen (1–2 Minuten) und eine Gravur mit Datum oder Namen hinzufügen.",
          "Dann den Druck mit Versand innerhalb der Ukraine bestellen — oder außerhalb der Ukraine die 3MF-Datei (≈3 €) kaufen und vor Ort drucken lassen.",
        ] },
        { h2: "Ein passendes Paar oder ein gemeinsames Geschenk", p: [
          "Zwei Anhänger mit demselben Ort und Datum: Einer bleibt zu Hause, der andere ist unterwegs dabei.",
          "Reicht das Budget, lässt sich beides kombinieren: ein Anhänger für den einen und eine größere Tischkarte desselben Ortes für die Wohnung.",
          "Das Datum muss nicht das erste Date sein — es kann der Tag sein, an dem ihr zusammengezogen seid oder euch verlobt habt; wichtig ist, dass es euch beiden etwas bedeutet.",
        ] },
        { h2: "Kartendaten und Material", p: [
          "Die Karte entsteht aus OpenStreetMap-Daten — echte Gebäude, Straßen, Wasser und Parks rund um den gewählten Punkt, keine gezeichnete Illustration.",
          "Das Material ist Eco PLA; Anhänger und Karte halten den Alltag und Jahre im Regal aus.",
          "Hat das Café inzwischen geschlossen oder heißt anders, ist das kein Problem: Die Karte zeigt die Straße und den Grundriss des Blocks, kein Ladenschild — der Ort bleibt erkennbar.",
        ] },
      ],
      ctaLabel: "Jahrestagskarte erstellen",
      ctaHref: "/podarunok/na-richnytsyu",
      outro: "Zahlung online; gedruckte Modelle versenden wir nur innerhalb der Ukraine, 2–4 Werktage nach dem Druck. Weißt du die genaue Adresse nicht mehr, reicht der Name des Cafés oder Parks — die Kartensuche findet den Ort.",
    },
    pl: {
      title: "Prezent na rocznicę związku: mapa waszej pierwszej randki",
      description: "Pomysł na prezent na rocznicę dla chłopaka lub dziewczyny: mapa 3D lub brelok z miejscem pierwszej randki i wygrawerowaną datą. Brelok od ≈4 €.",
      h1: "Prezent na rocznicę związany z jednym konkretnym miejscem",
      intro: "Prezenty na rocznicę zwykle mają napis w stylu „nasza data” czy „nasze imiona” — i łatwo je pomylić z tuzinem podobnych. Mapa działa inaczej: pokazuje dokładne miejsce, gdzie wszystko się zaczęło — kawiarnię, park, ulicę czy stację metra pierwszej randki. To nie abstrakcyjny symbol związku, ale kawałek geografii, który rozpoznajecie tylko wy.",
      sections: [
        { h2: "Dlaczego miejsce pierwszej randki, a nie tylko napis", p: [
          "Napis „kocham cię” pasuje do każdego prezentu. Mapy kwartału, gdzie spotkaliście się pierwszy raz, nie kupisz z półki — istnieje tylko dla tej jednej historii.",
          "To też okazja, by przypomnieć sobie na wpół zapomniane szczegóły: nazwę ulicy, kształt budynku kawiarni, pobliski park — wszystko widać na mapie jako prawdziwy układ.",
        ] },
        { h2: "Formaty prezentu", p: [
          "Brelok z mapą miejsca spotkania (od ≈4 €) — może być w parze: jeden dla niego, drugi dla niej, z pasującym grawerem lub datą.",
          "Magnes na lodówkę (≈5 €) — lekki dodatek do głównego prezentu albo osobna opcja.",
          "Mapa 3D na biurko w rozmiarze S lub M (≈8 lub ≈12 €) — na półkę lub szafkę nocną, z większą liczbą szczegółów wokół miejsca spotkania.",
          "Jeśli chcesz pokazać też ukształtowanie terenu (np. park na wzgórzu czy bulwar), możesz dodać opcję rzeźby terenu za ≈2 €.",
        ] },
        { h2: "Jak zrobić to w 5 minut", p: [
          "Wybierz produkt — brelok lub mapę — znajdź miejsce pierwszej randki (kawiarnię, park, ulicę), zatwierdź darmowy podgląd 3D (1–2 minuty) i dodaj grawer z datą lub imionami.",
          "Potem zamów druk z wysyłką na terenie Ukrainy — albo poza Ukrainą kup plik 3MF (≈3 €) i wydrukuj go na miejscu.",
        ] },
        { h2: "Para breloków czy jeden wspólny prezent", p: [
          "Dwa breloki z tym samym miejscem i datą: jeden zostaje w domu, drugi jest zawsze przy tobie.",
          "Jeśli budżet pozwala, możesz połączyć oba: brelok dla jednej osoby i większą mapę tego samego miejsca do domu.",
          "Data nie musi dotyczyć pierwszej randki — może to być dzień, w którym zamieszkaliście razem albo się zaręczyliście; ważne, żeby coś dla was znaczyła.",
        ] },
        { h2: "Dane mapy i materiał", p: [
          "Mapa powstaje z danych OpenStreetMap — prawdziwe budynki, drogi, woda i parki wokół wybranego punktu, a nie rysowana ilustracja.",
          "Materiał to Eco PLA, więc brelok i mapa wytrzymają codzienne noszenie i lata na półce.",
          "Jeśli kawiarnia została zamknięta lub zmieniła nazwę, to nie problem: mapa pokazuje ulicę i układ kwartału, a nie szyld — miejsce wciąż da się rozpoznać.",
        ] },
      ],
      ctaLabel: "Stwórz mapę na rocznicę",
      ctaHref: "/podarunok/na-richnytsyu",
      outro: "Płatność online; wydrukowane modele wysyłamy tylko na terenie Ukrainy, 2–4 dni robocze po druku. Jeśli nie pamiętasz dokładnego adresu, wystarczy nazwa kawiarni lub parku — wyszukiwarka znajdzie miejsce.",
    },
    fr: {
      title: "Cadeau pour un anniversaire de couple : la carte de votre premier rendez-vous",
      description: "Idée cadeau d'anniversaire de couple pour lui ou elle : une carte 3D ou un porte-clés avec le lieu du premier rendez-vous et la date gravée. Porte-clés dès ≈4 €.",
      h1: "Un cadeau d'anniversaire de couple lié à un lieu précis",
      intro: "Les cadeaux d'anniversaire portent souvent une inscription comme « notre date » ou « nos prénoms » — faciles à confondre avec une dizaine d'autres. Une carte fonctionne autrement : elle montre l'endroit exact où tout a commencé — le café, le parc, la rue ou la station de métro de votre premier rendez-vous. Pas un symbole abstrait, mais un morceau de géographie que vous seuls reconnaissez.",
      sections: [
        { h2: "Pourquoi le lieu du premier rendez-vous, et pas seulement une inscription", p: [
          "Un « je t'aime » va sur n'importe quel cadeau. La carte du quartier de votre première rencontre ne s'achète pas toute faite : elle n'existe que pour votre histoire.",
          "C'est aussi l'occasion de retrouver un détail à moitié oublié : le nom de la rue, la forme du bâtiment du café, un parc voisin — tout apparaît sur la carte avec le vrai tracé.",
        ] },
        { h2: "Les formats", p: [
          "Un porte-clés avec la carte du lieu de rencontre (dès ≈4 €) — possible en paire : un pour lui, un pour elle, avec une gravure ou une date assortie.",
          "Un aimant de frigo (≈5 €) — un petit complément au cadeau principal, ou une option à part entière.",
          "Une carte 3D de bureau en taille S ou M (≈8 ou ≈12 €) — pour une étagère ou une table de nuit, avec plus de détails autour du lieu.",
          "Pour montrer aussi le relief (un parc sur une colline ou un quai, par exemple), l'option relief coûte ≈2 €.",
        ] },
        { h2: "Le faire en 5 minutes", p: [
          "Choisis un produit — porte-clés ou carte —, trouve le lieu du premier rendez-vous (café, parc, rue), valide l'aperçu 3D gratuit (1 à 2 minutes) et ajoute une gravure avec la date ou les prénoms.",
          "Ensuite, commande l'impression livrée en Ukraine — ou, hors d'Ukraine, achète le fichier 3MF (≈3 €) et fais-le imprimer sur place.",
        ] },
        { h2: "Une paire assortie ou un cadeau commun", p: [
          "Deux porte-clés avec le même lieu et la même date : l'un reste à la maison, l'autre vous accompagne.",
          "Si le budget le permet, combine les deux : un porte-clés pour l'un, et une carte de bureau plus grande du même lieu pour la maison.",
          "La date n'est pas forcément celle du premier rendez-vous — ce peut être le jour où vous avez emménagé ensemble ou vous êtes fiancés ; l'important, c'est qu'elle compte pour vous deux.",
        ] },
        { h2: "Données et matière", p: [
          "La carte est construite à partir des données OpenStreetMap — vrais bâtiments, routes, eau et parcs autour du point choisi, pas une illustration dessinée.",
          "La matière est l'Eco PLA : porte-clés et carte supportent l'usage quotidien et des années sur une étagère.",
          "Si le café a fermé ou changé de nom depuis, pas de problème : la carte montre la rue et le plan du quartier, pas l'enseigne — le lieu reste reconnaissable.",
        ] },
      ],
      ctaLabel: "Créer une carte d'anniversaire",
      ctaHref: "/podarunok/na-richnytsyu",
      outro: "Paiement en ligne ; les modèles imprimés ne sont livrés qu'en Ukraine, 2–4 jours ouvrés après impression. Si tu ne te souviens plus de l'adresse exacte, le nom du café ou du parc suffit — la recherche trouvera le lieu.",
    },
    es: {
      title: "Regalo de aniversario de pareja: el mapa de vuestra primera cita",
      description: "Idea de regalo de aniversario para tu novio o novia: un mapa 3D o un llavero con el lugar de la primera cita y la fecha grabada. Llavero desde ≈4 €.",
      h1: "Un regalo de aniversario ligado a un lugar concreto",
      intro: "Los regalos de aniversario suelen llevar un texto como «nuestra fecha» o «nuestros nombres», y es fácil confundirlos con una docena de regalos parecidos. Un mapa funciona distinto: muestra el lugar exacto donde todo empezó, la cafetería, el parque, la calle o la estación de metro de vuestra primera cita. No es un símbolo abstracto, sino un trozo de geografía que solo reconocéis vosotros.",
      sections: [
        { h2: "Por qué el lugar de la primera cita y no solo un texto", p: [
          "Un «te quiero» cabe en cualquier regalo. El mapa de la manzana donde os visteis por primera vez no se compra hecho: solo existe para vuestra historia.",
          "También es una ocasión para recordar detalles medio olvidados: el nombre de la calle, la forma del edificio de la cafetería, un parque cercano, todo visible en el mapa con su trazado real.",
        ] },
        { h2: "Formatos de regalo", p: [
          "Un llavero con el mapa del lugar del encuentro (desde ≈4 €), que puede ir en pareja: uno para cada uno, con grabado o fecha a juego.",
          "Un imán de nevera (≈5 €): un complemento ligero al regalo principal o una opción por sí sola.",
          "Un mapa 3D de escritorio en tamaño S o M (≈8 o ≈12 €): para una estantería o la mesilla, con más detalle alrededor del lugar.",
          "Si quieres mostrar también el relieve (un parque en una colina o un paseo junto al río, por ejemplo), la opción de relieve cuesta ≈2 €.",
        ] },
        { h2: "Cómo hacerlo en 5 minutos", p: [
          "Elige un producto (llavero o mapa), busca el lugar de la primera cita (cafetería, parque, calle), confirma la vista previa 3D gratis (1–2 minutos) y añade un grabado con la fecha o los nombres.",
          "Después pide la impresión con envío dentro de Ucrania o, fuera de Ucrania, compra el archivo 3MF (≈3 €) e imprímelo cerca de ti.",
        ] },
        { h2: "Una pareja a juego o un regalo compartido", p: [
          "Dos llaveros con el mismo lugar y la misma fecha: uno se queda en casa y el otro va contigo.",
          "Si el presupuesto lo permite, puedes combinar ambos: un llavero para uno y un mapa de escritorio más grande del mismo lugar para casa.",
          "La fecha no tiene que ser la de la primera cita: puede ser el día en que os fuisteis a vivir juntos o os prometisteis; lo importante es que signifique algo para los dos.",
        ] },
        { h2: "Datos del mapa y material", p: [
          "El mapa se construye con datos de OpenStreetMap: edificios, carreteras, agua y parques reales alrededor del punto elegido, no una ilustración dibujada.",
          "El material es Eco PLA, así que llavero y mapa aguantan el uso diario y años en una estantería.",
          "Si la cafetería ha cerrado o cambiado de nombre, no pasa nada: el mapa muestra la calle y la forma de la manzana, no un letrero, así que el lugar se sigue reconociendo.",
        ] },
      ],
      ctaLabel: "Crear un mapa de aniversario",
      ctaHref: "/podarunok/na-richnytsyu",
      outro: "Pago online; los modelos impresos solo se envían dentro de Ucrania, 2–4 días hábiles después de imprimir. Si no recuerdas la dirección exacta, basta con el nombre de la cafetería o el parque: el buscador encontrará el lugar.",
    },
  },

  "podarunok-na-novyi-rik-2027-shcho-podaruvaty": {
    de: {
      title: "Geschenk zu Weihnachten und Neujahr 2027: eine persönliche Karte für die Familie",
      description: "Geschenkidee zu Weihnachten und Neujahr 2027: eine 3D-Karte, ein Anhänger oder Magnet mit der Heimatstraße oder -stadt. Anhänger ab ≈4 €, Druckdatei ≈3 €.",
      h1: "Was schenkt man zu Neujahr 2027, wenn die üblichen Geschenke ausgereizt sind?",
      intro: "Jedes Jahr vor den Feiertagen dieselbe Frage: Was schenkt man jemandem, der schon alles hat? Kosmetik, Süßes, noch eine Kerze — das funktioniert, sagt aber nichts über diesen Menschen. Eine Karte oder ein Anhänger mit einer bestimmten Adresse — ein Hof, eine Straße, eine Stadt — ist ein Geschenk, das man mit keinem anderen verwechselt, weil es für einen Menschen und einen Ort gemacht ist.",
      sections: [
        { h2: "Warum eine Karte ein typisches Souvenir schlägt", p: [
          "Eine Karte muss keinen Geschmack erraten — Farbe, Stil, Größe. Sie zeigt einen Ort, den der Beschenkte kennt und sofort erkennt: den Hof, die Kindheitsstraße, die Stadt, in der er jetzt lebt.",
          "Außerdem erklärt sie sich in einem Satz: „Hier bist du geboren“ oder „Hier haben wir uns kennengelernt“ — mehr braucht es nicht.",
        ] },
        { h2: "Formate für jedes Budget", p: [
          "Ein Karten-Anhänger (ab ≈4 €) — kompakt, auf Wunsch mit Namens- oder Datumsgravur.",
          "Ein Kühlschrankmagnet (≈5 €) — eine leichte Variante für entferntere Verwandte oder Kollegen.",
          "Eine 3D-Tischkarte in S, M, L oder XL (≈8–18 €) — das Hauptgeschenk für nahe Menschen, auf Wunsch mit Geländerelief.",
          "Ein Kachelbild für die Wand — für ein großes Familiengeschenk oder eine Wandkarte in einer neuen Wohnung.",
        ] },
        { h2: "In 5 Minuten erstellt", p: [
          "Produkt wählen — Anhänger, Magnet oder 3D-Karte —, Adresse oder Stadt auf der Karte finden, die kostenlose 3D-Vorschau prüfen (1–2 Minuten) und bei Bedarf eine Gravur hinzufügen.",
          "Dann den Druck mit Versand innerhalb der Ukraine bestellen — oder die 3MF-Datei (≈3 €) kaufen und in einem 3D-Druckservice in deiner Nähe drucken lassen.",
          "Die genaue Adresse ist unbekannt? Der Name des Viertels oder ein Orientierungspunkt — Schule, Park, U-Bahn-Station — reicht; die Kartensuche findet den Ort.",
        ] },
        { h2: "Wann bestellen", p: [
          "Druck und Versand innerhalb der Ukraine dauern 2–4 Werktage, vor den Feiertagen steigt aber die Nachfrage — für ein Geschenk unter dem Baum bestellt man am besten bis Mitte Dezember. Die 3MF-Datei bekommst du jederzeit sofort.",
        ] },
        { h2: "Mehrere Geschenke auf einmal", p: [
          "Musst du mehrere Menschen beschenken, bestelle einfach eine Reihe Anhänger oder Magnete mit jeweils eigener Adresse — der Stückpreis bleibt gleich.",
          "Das Material ist Eco PLA, die Kartendaten stammen aus OpenStreetMap: echte Gebäude, Straßen, Wasser und Parks rund um den gewählten Ort, keine allgemeine Illustration.",
          "Unsicher, was zu wem passt? Orientiere dich am Budget: Anhänger oder Magnet für Kollegen und entfernte Verwandte, eine Tischkarte für die engste Familie, ein Kachelbild als gemeinsames Geschenk der ganzen Familie.",
        ] },
      ],
      ctaLabel: "Neujahrskarte erstellen",
      ctaHref: "/create?product=map3d",
      outro: "Zahlung online; gedruckte Modelle versenden wir nur innerhalb der Ukraine. Wird der Druck bis zum Fest knapp, kannst du die kostenlose 3D-Vorschau oder die 3MF-Datei (149 ₴ ≈ 3 €) schon als Teil des Geschenks zeigen.",
    },
    pl: {
      title: "Prezent na święta i Nowy Rok 2027: osobista mapa dla rodziny",
      description: "Pomysł na prezent na Boże Narodzenie i Nowy Rok 2027: mapa 3D, brelok lub magnes z rodzinną ulicą lub miastem. Brelok od ≈4 €, plik do druku ≈3 €.",
      h1: "Co dać na Nowy Rok 2027, gdy zwykłe prezenty już się znudziły",
      intro: "Co roku przed świętami to samo pytanie: co dać komuś, kto ma wszystko? Kosmetyki, słodycze, kolejna świeczka — działa, ale nic nie mówi o tej konkretnej osobie. Mapa lub brelok z konkretnym adresem — podwórkiem, ulicą, miastem — to prezent, którego nie da się pomylić z innym, bo powstaje dla jednej osoby i jednego miejsca.",
      sections: [
        { h2: "Dlaczego mapa wygrywa z typową pamiątką", p: [
          "Mapa nie musi zgadywać gustu — koloru, stylu, rozmiaru. Pokazuje miejsce, które obdarowany zna i rozpozna od razu: rodzinne podwórko, ulicę z dzieciństwa, miasto, w którym mieszka.",
          "Do tego tłumaczy się jednym zdaniem: „tu się urodziłeś” albo „tu się poznaliśmy” — nic więcej nie trzeba dodawać.",
        ] },
        { h2: "Formaty na każdy budżet", p: [
          "Brelok z mapą (od ≈4 €) — kompaktowy prezent, można dodać grawer z imieniem lub datą.",
          "Magnes na lodówkę (≈5 €) — lekka opcja dla dalszej rodziny lub współpracowników.",
          "Mapa 3D na biurko w rozmiarze S, M, L lub XL (≈8–18 €) — główny prezent dla bliskich, opcjonalnie z rzeźbą terenu.",
          "Panel ścienny z płytek — na duży rodzinny prezent albo mapę na ścianę w nowym mieszkaniu.",
        ] },
        { h2: "Jak zrobić to w 5 minut", p: [
          "Wybierz produkt — brelok, magnes lub mapę 3D — znajdź adres lub miasto na mapie, sprawdź darmowy podgląd 3D (1–2 minuty) i w razie potrzeby dodaj grawer.",
          "Potem zamów druk z wysyłką na terenie Ukrainy — albo kup plik 3MF (≈3 €) i zleć druk w lokalnym serwisie druku 3D.",
          "Nie znasz dokładnego adresu? Wystarczy nazwa dzielnicy lub punkt orientacyjny — szkoła, park, stacja metra; wyszukiwarka znajdzie miejsce.",
        ] },
        { h2: "Kiedy zamówić", p: [
          "Druk i wysyłka na terenie Ukrainy trwają 2–4 dni robocze, ale przed świętami rośnie popyt — na prezent pod choinkę najlepiej zamówić do połowy grudnia. Plik 3MF dostajesz od razu, w każdej chwili.",
        ] },
        { h2: "Kilka prezentów naraz", p: [
          "Jeśli chcesz obdarować kilka osób, zamów serię breloków lub magnesów, każdy z innym adresem — cena za sztukę się nie zmienia.",
          "Materiał to Eco PLA, a dane mapy pochodzą z OpenStreetMap: prawdziwe budynki, drogi, woda i parki wokół wybranego miejsca, a nie ogólna ilustracja.",
          "Nie wiesz, co komu pasuje? Kieruj się budżetem: brelok lub magnes dla współpracowników i dalszej rodziny, mapa na biurko dla najbliższych, panel z płytek jako wspólny prezent od całej rodziny.",
        ] },
      ],
      ctaLabel: "Stwórz noworoczną mapę",
      ctaHref: "/create?product=map3d",
      outro: "Płatność online; wydrukowane modele wysyłamy tylko na terenie Ukrainy. Jeśli nie zdążysz z drukiem przed świętami, darmowy podgląd 3D albo plik 3MF (149 ₴ ≈ 3 €) możesz już pokazać jako część prezentu.",
    },
    fr: {
      title: "Cadeau de Noël et du Nouvel An 2027 : une carte personnelle pour la famille",
      description: "Idée cadeau pour Noël et le Nouvel An 2027 : une carte 3D, un porte-clés ou un aimant avec la rue ou la ville natale. Porte-clés dès ≈4 €, fichier à imprimer ≈3 €.",
      h1: "Quoi offrir pour le Nouvel An 2027 quand les cadeaux habituels sont usés ?",
      intro: "Chaque année avant les fêtes, la même question : que offrir à quelqu'un qui a déjà tout ? Cosmétiques, chocolats, une bougie de plus — ça marche, mais ça ne dit rien de la personne. Une carte ou un porte-clés avec une adresse précise — une cour, une rue, une ville — est un cadeau qu'on ne confond avec aucun autre, car il est fait pour une personne et un lieu.",
      sections: [
        { h2: "Pourquoi une carte bat un souvenir classique", p: [
          "Une carte n'a pas à deviner les goûts — couleur, style, taille. Elle montre un lieu que la personne connaît et reconnaît tout de suite : sa cour, la rue de son enfance, la ville où elle vit.",
          "Et elle s'explique en une phrase : « c'est ici que tu es né » ou « c'est ici qu'on s'est rencontrés » — rien à ajouter.",
        ] },
        { h2: "Des formats pour tous les budgets", p: [
          "Un porte-clés carte (dès ≈4 €) — compact, avec gravure d'un prénom ou d'une date en option.",
          "Un aimant de frigo (≈5 €) — une option légère pour la famille éloignée ou les collègues.",
          "Une carte 3D de bureau en taille S, M, L ou XL (≈8–18 €) — le cadeau principal pour un proche, avec relief en option.",
          "Un tableau mural en tuiles — pour un grand cadeau familial ou une carte murale dans un nouvel appartement.",
        ] },
        { h2: "Le faire en 5 minutes", p: [
          "Choisis un produit — porte-clés, aimant ou carte 3D —, trouve l'adresse ou la ville, vérifie l'aperçu 3D gratuit (1 à 2 minutes) et ajoute une gravure si besoin.",
          "Ensuite, commande l'impression livrée en Ukraine — ou achète le fichier 3MF (≈3 €) et fais-le imprimer dans un service d'impression 3D près de chez toi.",
          "Pas d'adresse exacte ? Le nom du quartier ou un repère — une école, un parc, une station de métro — suffit ; la recherche trouvera le lieu.",
        ] },
        { h2: "Quand commander", p: [
          "Impression et livraison en Ukraine : 2–4 jours ouvrés, mais la demande augmente avant les fêtes — pour un cadeau sous le sapin, mieux vaut commander avant la mi-décembre. Le fichier 3MF, lui, s'obtient immédiatement, à tout moment.",
        ] },
        { h2: "Plusieurs cadeaux d'un coup", p: [
          "Pour gâter plusieurs personnes, commande une série de porte-clés ou d'aimants avec une adresse différente pour chacun — le prix unitaire reste le même.",
          "La matière est l'Eco PLA et les données viennent d'OpenStreetMap : vrais bâtiments, routes, eau et parcs autour du lieu choisi, pas une illustration générique.",
          "Tu hésites ? Laisse-toi guider par le budget : porte-clés ou aimant pour les collègues et la famille éloignée, carte de bureau pour les proches, tableau en tuiles comme cadeau commun de toute la famille.",
        ] },
      ],
      ctaLabel: "Créer une carte du Nouvel An",
      ctaHref: "/create?product=map3d",
      outro: "Paiement en ligne ; les modèles imprimés ne sont livrés qu'en Ukraine. Si l'impression n'arrive pas à temps pour les fêtes, l'aperçu 3D gratuit ou le fichier 3MF (149 ₴ ≈ 3 €) peut déjà faire partie du cadeau.",
    },
    es: {
      title: "Regalo de Navidad y Año Nuevo 2027: un mapa personal para la familia",
      description: "Idea de regalo para Navidad y Año Nuevo 2027: un mapa 3D, un llavero o un imán con la calle o la ciudad de siempre. Llavero desde ≈4 €, archivo para imprimir ≈3 €.",
      h1: "Qué regalar en Año Nuevo 2027 cuando los regalos de siempre ya cansan",
      intro: "Cada año antes de las fiestas llega la misma pregunta: qué regalar a quien ya lo tiene todo. Cosmética, dulces, otra vela: funciona, pero no dice nada de esa persona. Un mapa o un llavero con una dirección concreta (un patio, una calle, una ciudad) es un regalo que no se confunde con ningún otro, porque está hecho para una persona y un lugar.",
      sections: [
        { h2: "Por qué un mapa gana a un recuerdo típico", p: [
          "Un mapa no tiene que adivinar gustos, colores, estilos ni tallas. Muestra un lugar que la persona conoce y reconoce al instante: el patio de casa, la calle de su infancia, la ciudad donde vive.",
          "Además se explica en una frase: «aquí naciste» o «aquí nos conocimos». No hace falta añadir nada más.",
        ] },
        { h2: "Formatos para cualquier presupuesto", p: [
          "Un llavero con mapa (desde ≈4 €): compacto, con grabado opcional de nombre o fecha.",
          "Un imán de nevera (≈5 €): una opción ligera para familiares lejanos o compañeros.",
          "Un mapa 3D de escritorio en tamaño S, M, L o XL (≈8–18 €): el regalo principal para alguien cercano, con relieve opcional.",
          "Un panel de pared de baldosas: para un gran regalo familiar o un mapa de pared en un piso nuevo.",
        ] },
        { h2: "Cómo hacerlo en 5 minutos", p: [
          "Elige un producto (llavero, imán o mapa 3D), busca la dirección o la ciudad, revisa la vista previa 3D gratis (1–2 minutos) y añade un grabado si quieres.",
          "Después pide la impresión con envío dentro de Ucrania, o compra el archivo 3MF (≈3 €) y encarga la impresión a un servicio de impresión 3D cercano.",
          "¿No sabes la dirección exacta? Basta con el nombre del barrio o una referencia (un colegio, un parque, una estación de metro): el buscador encontrará el lugar.",
        ] },
        { h2: "Cuándo pedirlo", p: [
          "La impresión y el envío dentro de Ucrania tardan 2–4 días hábiles, pero la demanda sube antes de las fiestas: para un regalo bajo el árbol conviene pedirlo antes de mediados de diciembre. El archivo 3MF lo tienes al momento, cuando quieras.",
        ] },
        { h2: "Varios regalos a la vez", p: [
          "Si tienes que felicitar a varias personas, pide una serie de llaveros o imanes, cada uno con su dirección: el precio por unidad no cambia.",
          "El material es Eco PLA y los datos del mapa vienen de OpenStreetMap: edificios, carreteras, agua y parques reales alrededor del lugar elegido, no una ilustración genérica.",
          "¿Dudas qué encaja con cada uno? Guíate por el presupuesto: llavero o imán para compañeros y familia lejana, mapa de escritorio para los más cercanos y panel de baldosas como regalo común de toda la familia.",
        ] },
      ],
      ctaLabel: "Crear un mapa de Año Nuevo",
      ctaHref: "/create?product=map3d",
      outro: "Pago online; los modelos impresos solo se envían dentro de Ucrania. Si la impresión no llega a tiempo para las fiestas, la vista previa 3D gratis o el archivo 3MF (149 ₴ ≈ 3 €) ya pueden formar parte del regalo.",
    },
  },
};
