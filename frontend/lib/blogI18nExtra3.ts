// Переклади статей блогу de/pl/fr/es — частина 3 (правила — див. lib/blogI18nExtra.ts).
import type { BlogArticleContent } from "@/lib/blog";
import type { BlogExtraLocale } from "@/lib/blogI18nExtra";

export const BLOG_I18N_EXTRA_3: Record<string, Partial<Record<BlogExtraLocale, BlogArticleContent>>> = {
  "shcho-podaruvaty-tatovi-na-den-narodzhennya": {
    de: {
      title: "Geburtstagsgeschenk für Papa: 3D-Karte seiner Heimatstadt",
      description: "Geschenkidee zum Geburtstag des Vaters: eine 3D-Karte der Stadt oder des Dorfes, in dem er aufgewachsen ist. Anhänger ab ≈4 €, Karte ab ≈8 €, Druckdatei ≈3 €.",
      h1: "Papas Geburtstag: ein Geschenk, in dem sein Herkunftsort steckt",
      intro: "Für Eltern ist Schenken schwer — sie „haben alles“, und noch ein Hemd oder Werkzeug begeistert wenig. Erinnerung wirkt: eine 3D-Karte der Stadt oder des Dorfes, in dem Papa aufgewachsen ist, mit der Straße zur Schule und dem Fluss oder Park, den er auch nach Jahren sofort erkennt.",
      sections: [
        { h2: "Warum ein Kindheitsort mehr ist als ein Gegenstand", p: [
          "Ein Gegenstand ist nach einer Woche vergessen, eine Karte des Heimathofs steht jahrelang im Regal — und bringt immer wieder Geschichten darüber, wie es früher aussah.",
          "Es funktioniert in jedem Alter und mit jedem Budget: Entscheidend ist die Genauigkeit des Ortes, nicht der Preis. Selbst wenn Papa längst woanders lebt, holt eine Karte seines Heimathofs die Kindheit besser zurück als jede Karte oder jeder Social-Media-Post.",
        ] },
        { h2: "Welcher Ort, wenn es mehrere gibt", p: [
          "Ist Papa mehrmals umgezogen — Schule, erste Wohnung, die Stadt, in der ihr geboren wurdet —, nimm den Ort, von dem er am häufigsten erzählt. Das ist ein besserer Hinweis als jede Geschmacksumfrage.",
          "Du kannst auch zwei kleinere Karten statt einer machen: das Heimatdorf und die Stadt, in der er die meiste Zeit gelebt hat. Zusammen im Regal zeichnen sie seinen Lebensweg nach.",
        ] },
        { h2: "Ehrlich über Dörfer und Kleinstädte", p: [
          "Die Kartendaten stammen aus OpenStreetMap — Großstädte sind detailliert erfasst, in Dörfern und Kleinstädten sind manchmal weniger Gebäude oder Wege eingetragen, als es tatsächlich gibt.",
          "Prüfe vor der Bestellung die 3D-Vorschau: Wirkt das Dorf zu leer, vergrößere den Radius mit der Umgebung oder nimm eine nahe Stadt, in der Papa gearbeitet oder gelernt hat.",
          "Das ist kein Grund aufzugeben — schon ein paar Gebäude und die Straße dazwischen werden in 3D oft sofort erkannt, weil sich die Erinnerung eher an die Form einer Straße als an die Menge der Details hält.",
        ] },
        { h2: "Drei Formate für jedes Budget", p: [
          "Ein Anhänger mit der Karte seiner Heimatstraße (ab ≈4 €) — jeden Tag in der Tasche, mit dem Geburtsjahr auf der Rückseite.",
          "Ein Magnet (≈5 €) — für die Küche, in der er morgens Kaffee trinkt.",
          "Eine 3D-Karte (ab ≈8 €) — das Hauptgeschenk fürs Regal: Gebäude, Straßen, Park oder Wald in der Nähe, auf Wunsch mit Geländerelief. Für ein großes Familienjubiläum passt ein Kachelbild für die Wand, wenn mehrere Kinder zusammenlegen.",
        ] },
        { h2: "In 5 Minuten erstellt", p: [
          "Produkt wählen — Anhänger, Magnet oder 3D-Karte —, Papas Heimatstadt oder -dorf auf der Karte finden und den Rahmen über die richtige Straße ziehen.",
          "Nach 1–2 Minuten steht eine kostenlose 3D-Vorschau — prüfe, ob genug Details da sind, bevor du bestellst.",
        ] },
        { h2: "Wann es fertig ist", p: [
          "Druck und Versand innerhalb der Ukraine dauern 2–4 Werktage, Zahlung online auf der Website.",
          "Lebt Papa außerhalb der Ukraine, kaufe die 3MF-Datei (≈3 €) und lass sie in einem 3D-Druckservice in seiner Nähe drucken.",
        ] },
      ],
      ctaLabel: "Karte von Papas Heimatstadt erstellen",
      ctaHref: "/create?product=map3d",
      outro: "Unsicher, ob das Dorf in OSM detailliert genug ist? Probier zuerst die kostenlose 3D-Vorschau — sie dauert keine zwei Minuten und kostet nichts, auch wenn du dich am Ende für einen anderen Ort entscheidest.",
    },
    pl: {
      title: "Prezent na urodziny taty: mapa 3D jego rodzinnego miasta",
      description: "Pomysł na prezent urodzinowy dla taty: mapa 3D miasta lub wsi, w której dorastał. Brelok od ≈4 €, mapa od ≈8 €, plik do druku ≈3 €.",
      h1: "Urodziny taty: prezent, w którym jest jego rodzinne miejsce",
      intro: "Rodzicom trudno kupić prezent — „mają wszystko”, a kolejna koszula czy narzędzie nie budzi emocji. Działa pamięć: mapa 3D miasta albo wsi, w której tata dorastał, z ulicą, którą chodził do szkoły, i rzeką lub parkiem, które rozpozna od razu nawet po latach.",
      sections: [
        { h2: "Dlaczego miejsce z dzieciństwa wygrywa z kolejną rzeczą", p: [
          "O rzeczy zapomina się po tygodniu, a mapa rodzinnego podwórka stoi na półce latami — i wciąż wywołuje opowieści o tym, jak to kiedyś wyglądało.",
          "Działa w każdym wieku i przy każdym budżecie: liczy się dokładność miejsca, a nie cena. Nawet jeśli tata mieszka dziś gdzie indziej, mapa rodzinnego podwórka przywołuje dzieciństwo lepiej niż kartka czy post w mediach społecznościowych.",
        ] },
        { h2: "Które miejsce, gdy jest ich kilka", p: [
          "Jeśli tata przeprowadzał się kilka razy — szkoła, pierwsze mieszkanie, miasto, w którym wy się urodziliście — wybierz to, o którym najczęściej opowiada. To lepsza wskazówka niż jakakolwiek ankieta.",
          "Możesz też zrobić dwie mniejsze mapy zamiast jednej: rodzinną wieś osobno i miasto, w którym spędził większość życia, osobno. Razem na półce pokazują drogę jego życia.",
        ] },
        { h2: "Uczciwie o wsiach i małych miastach", p: [
          "Dane mapy pochodzą z OpenStreetMap — duże miasta są opisane szczegółowo, ale na wsiach i w małych miasteczkach bywa mniej budynków i dróg, niż jest w rzeczywistości.",
          "Przed zamówieniem sprawdź podgląd 3D: jeśli wieś wygląda ubogo, poszerz obszar o okolicę albo wybierz pobliskie miasto, gdzie tata pracował lub się uczył.",
          "To nie powód, by rezygnować — nawet kilka budynków i droga między nimi w 3D są często rozpoznawane od razu, bo pamięć trzyma się kształtu ulicy bardziej niż liczby szczegółów.",
        ] },
        { h2: "Trzy formaty na każdy budżet", p: [
          "Brelok z mapą jego rodzinnej ulicy (od ≈4 €) — codziennie w kieszeni, z rokiem urodzenia na odwrocie.",
          "Magnes (≈5 €) — do kuchni, gdzie pije poranną kawę.",
          "Mapa 3D (od ≈8 €) — główny prezent na półkę: budynki, drogi, park lub las w pobliżu, opcjonalnie z rzeźbą terenu. Na rodzinny jubileusz sprawdzi się panel ścienny z płytek, gdy składa się kilkoro dzieci.",
        ] },
        { h2: "Jak zrobić to w 5 minut", p: [
          "Wybierz produkt — brelok, magnes lub mapę 3D — znajdź rodzinne miasto lub wieś taty i przeciągnij ramkę na właściwą ulicę.",
          "Darmowy podgląd 3D powstaje w 1–2 minuty — sprawdź, czy jest dość szczegółów, zanim zamówisz.",
        ] },
        { h2: "Kiedy będzie gotowe", p: [
          "Druk i wysyłka na terenie Ukrainy trwają 2–4 dni robocze, płatność online na stronie.",
          "Jeśli tata mieszka poza Ukrainą, kup plik 3MF (≈3 €) i zleć druk w serwisie druku 3D blisko niego.",
        ] },
      ],
      ctaLabel: "Stwórz mapę rodzinnego miasta taty",
      ctaHref: "/create?product=map3d",
      outro: "Nie wiesz, czy wieś jest dość szczegółowa w OSM? Najpierw wypróbuj darmowy podgląd 3D — trwa niecałe dwie minuty i nic nie kosztuje, nawet jeśli wybierzesz potem inne miejsce.",
    },
    fr: {
      title: "Cadeau d'anniversaire pour papa : la carte 3D de sa ville natale",
      description: "Idée cadeau d'anniversaire pour son père : la carte 3D de la ville ou du village où il a grandi. Porte-clés dès ≈4 €, carte dès ≈8 €, fichier à imprimer ≈3 €.",
      h1: "L'anniversaire de papa : un cadeau qui porte le lieu d'où il vient",
      intro: "Les parents sont difficiles à gâter — ils « ont tout », et une chemise ou un outil de plus ne les émeut guère. La mémoire, elle, fonctionne : la carte 3D de la ville ou du village où papa a grandi, avec la rue qu'il prenait pour aller à l'école et la rivière ou le parc qu'il reconnaît immédiatement, même des années après.",
      sections: [
        { h2: "Pourquoi un lieu d'enfance vaut mieux qu'un objet", p: [
          "Un objet s'oublie en une semaine, la carte de la cour d'enfance reste des années sur l'étagère — et relance sans cesse les histoires sur « comment c'était avant ».",
          "Ça marche à tout âge et pour tout budget : ce qui compte, c'est la justesse du lieu, pas le prix. Même si papa vit ailleurs aujourd'hui, la carte de sa cour d'enfance ramène plus de souvenirs qu'une carte de vœux ou une publication en ligne.",
        ] },
        { h2: "Quel lieu choisir s'il y en a plusieurs", p: [
          "Si papa a déménagé plusieurs fois — école, premier appartement, la ville où vous êtes nés —, choisis celui dont il parle le plus souvent. C'est un indice plus fiable que n'importe quel sondage.",
          "Tu peux aussi faire deux petites cartes au lieu d'une : le village natal d'un côté, la ville où il a passé la plus grande partie de sa vie de l'autre. Ensemble sur une étagère, elles retracent son parcours.",
        ] },
        { h2: "Honnêtement, pour les villages et petites villes", p: [
          "Les données viennent d'OpenStreetMap — les grandes villes sont très détaillées, mais les villages et petites villes ont parfois moins de bâtiments ou de chemins cartographiés qu'en réalité.",
          "Avant de commander, regarde l'aperçu 3D : si le village paraît vide, élargis la zone avec les environs ou choisis une ville proche où papa a travaillé ou étudié.",
          "Pas de quoi renoncer — quelques bâtiments et la route qui les relie, en 3D, sont souvent reconnus immédiatement, car la mémoire retient la forme d'une rue plus que la quantité de détails.",
        ] },
        { h2: "Trois formats pour tous les budgets", p: [
          "Un porte-clés avec la carte de sa rue natale (dès ≈4 €) — dans sa poche chaque jour, avec l'année de naissance gravée au dos.",
          "Un aimant (≈5 €) — pour la cuisine où il prend son café du matin.",
          "Une carte 3D (dès ≈8 €) — le cadeau principal pour l'étagère : bâtiments, routes, parc ou forêt proche, avec relief en option. Pour un grand anniversaire en famille, un tableau mural en tuiles convient si plusieurs enfants se cotisent.",
        ] },
        { h2: "Le faire en 5 minutes", p: [
          "Choisis un produit — porte-clés, aimant ou carte 3D —, trouve la ville ou le village natal de papa et fais glisser le cadre sur la bonne rue.",
          "Un aperçu 3D gratuit se construit en 1 à 2 minutes — vérifie qu'il y a assez de détails avant de commander.",
        ] },
        { h2: "Quand ce sera prêt", p: [
          "Impression et livraison en Ukraine : 2–4 jours ouvrés, paiement en ligne sur le site.",
          "Si papa vit hors d'Ukraine, achète le fichier 3MF (≈3 €) et fais-le imprimer dans un service d'impression 3D près de chez lui.",
        ] },
      ],
      ctaLabel: "Créer la carte de la ville natale de papa",
      ctaHref: "/create?product=map3d",
      outro: "Pas sûr que le village soit assez détaillé dans OSM ? Essaie d'abord l'aperçu 3D gratuit — moins de deux minutes, et rien à payer même si tu choisis finalement un autre lieu.",
    },
    es: {
      title: "Regalo de cumpleaños para papá: un mapa 3D de su ciudad natal",
      description: "Idea de regalo de cumpleaños para tu padre: un mapa 3D de la ciudad o el pueblo donde creció. Llavero desde ≈4 €, mapa desde ≈8 €, archivo para imprimir ≈3 €.",
      h1: "El cumpleaños de papá: un regalo que lleva el lugar de donde viene",
      intro: "A los padres cuesta regalarles: «lo tienen todo», y otra camisa u otra herramienta no emociona mucho. Lo que funciona es la memoria: un mapa 3D de la ciudad o el pueblo donde creció papá, con la calle por la que iba al colegio y el río o el parque que reconoce al instante incluso después de años.",
      sections: [
        { h2: "Por qué un lugar de la infancia gana a otro objeto", p: [
          "Un objeto se olvida en una semana; el mapa del patio de su infancia se queda años en la estantería y sigue provocando historias sobre cómo era todo antes.",
          "Funciona a cualquier edad y con cualquier presupuesto: lo que importa es la precisión del lugar, no el precio. Aunque papá viva ahora en otra ciudad, el mapa de su patio le devuelve la infancia mejor que una tarjeta o una publicación en redes.",
        ] },
        { h2: "Qué lugar elegir si hay varios", p: [
          "Si papá se ha mudado varias veces (colegio, primer piso, la ciudad donde nacisteis vosotros), elige el que más menciona cuando habla. Es una pista mejor que cualquier encuesta de gustos.",
          "También puedes hacer dos mapas pequeños en vez de uno: el pueblo natal por un lado y la ciudad donde pasó la mayor parte de su vida por otro. Juntos en la estantería trazan el camino de su vida.",
        ] },
        { h2: "Con honestidad sobre pueblos y ciudades pequeñas", p: [
          "Los datos vienen de OpenStreetMap: las grandes ciudades están muy detalladas, pero en pueblos y ciudades pequeñas a veces hay menos edificios o caminos cartografiados de los que existen.",
          "Antes de pedir, revisa la vista previa 3D: si el pueblo se ve vacío, amplía el radio con los alrededores o elige una ciudad cercana donde papá trabajó o estudió.",
          "No es motivo para renunciar: unos pocos edificios y la carretera entre ellos, en 3D, suelen reconocerse al instante, porque la memoria se agarra a la forma de una calle más que a la cantidad de detalles.",
        ] },
        { h2: "Tres formatos para cualquier presupuesto", p: [
          "Un llavero con el mapa de su calle natal (desde ≈4 €): cada día en el bolsillo, con el año de nacimiento grabado detrás.",
          "Un imán (≈5 €): para la cocina donde toma el café de la mañana.",
          "Un mapa 3D (desde ≈8 €): el regalo principal para la estantería, con edificios, carreteras, el parque o el bosque cercano y relieve opcional. Para un gran aniversario familiar encaja un panel de pared de baldosas si varios hijos ponen dinero.",
        ] },
        { h2: "Cómo hacerlo en 5 minutos", p: [
          "Elige un producto (llavero, imán o mapa 3D), busca la ciudad o el pueblo natal de papá y arrastra el marco a la calle correcta.",
          "La vista previa 3D gratis se genera en 1–2 minutos: comprueba que haya suficiente detalle antes de pedir.",
        ] },
        { h2: "Cuándo estará listo", p: [
          "La impresión y el envío dentro de Ucrania tardan 2–4 días hábiles, con pago online en la web.",
          "Si papá vive fuera de Ucrania, compra el archivo 3MF (≈3 €) y encarga la impresión a un servicio de impresión 3D cerca de él.",
        ] },
      ],
      ctaLabel: "Crear el mapa de la ciudad natal de papá",
      ctaHref: "/create?product=map3d",
      outro: "¿No sabes si el pueblo está bien detallado en OSM? Prueba primero la vista previa 3D gratis: tarda menos de dos minutos y no cuesta nada aunque al final elijas otro lugar.",
    },
  },

  "podarunok-kolezi-na-zvilnennya-abo-pereyizd": {
    de: {
      title: "Abschiedsgeschenk für einen Kollegen, der geht oder umzieht: eine Stadtkarte",
      description: "Was schenkt man einem Kollegen zum Abschied oder Umzug? Eine 3D-Karte des Büroviertels oder seiner neuen Stadt. Als Team zusammenlegen für Anhänger oder Magnete.",
      h1: "Ein Kollege geht: ein Teamgeschenk, das nicht in der Schublade landet",
      intro: "Eine vom ganzen Büro unterschriebene Karte landet meist in der Schublade. Was wirkt, ist etwas, das an einen Ort gebunden ist: das Viertel, in dem das Büro stand und ihr zusammen Mittag gegessen habt, oder die neue Stadt, in die der Kollege zieht. Auf so ein Geschenk kann sich ein Team auch leichter einigen als auf einen abstrakten Gegenstand.",
      sections: [
        { h2: "Zwei Ideen je nach Situation", p: [
          "Geht der Kollege, bleibt aber in der Stadt — eine 3D-Karte des Büroblocks: dieselbe Straße, das Café an der Ecke, der Park der Mittagspausen.",
          "Zieht er in eine andere Stadt — eine Karte des neuen Viertels oder der Innenstadt, schon vor dem Umzug fertig, dazu ein Anhänger mit einer Abschiedsgravur.",
          "Dazwischen gibt es den Wechsel in eine andere Niederlassung derselben Firma: Dann passen zwei kleine Karten — altes und neues Büro — als symbolische Übergabe.",
        ] },
        { h2: "Gemeinsam zusammenlegen", p: [
          "Statt einer großen Karte „für alle“ kann das Team ein Set finanzieren: für jeden einen Anhänger oder Magneten mit demselben Viertel und als Hauptgeschenk eine 3D-Karte für den, der geht.",
          "Das bleibt günstig: Ein Anhänger kostet ab ≈4 €, ein Magnet ≈5 €, sodass ein Set für 5–10 Personen auch neben einer eigenen 3D-Karte erschwinglich ist.",
        ] },
        { h2: "Was man graviert", p: [
          "Der letzte Bürotag, der Teamname oder einfach das Jahr — zurückhaltend und passend. Persönliche Wünsche gehören auf die Karte, auf das Modell eine neutrale Gravur.",
          "Firmen, die ihre Teams regelmäßig beschenken, können das Format für Firmenbestellungen unter /corporate nutzen — mehrere gleiche Sets auf einmal statt jedes Mal neu.",
        ] },
        { h2: "In 5 Minuten erstellt", p: [
          "Produkt wählen — Anhänger, Magnet oder 3D-Karte —, das Büroviertel oder die neue Stadt des Kollegen finden und den Rahmen über den richtigen Block ziehen.",
          "Nach 1–2 Minuten steht eine kostenlose 3D-Vorschau — teile sie im Team-Chat, damit alle das Ergebnis vor der Bestellung sehen.",
        ] },
        { h2: "Wenn das Team über mehrere Städte verteilt ist", p: [
          "Für ein Remote-Team ohne gemeinsames Büro ist der Bezugspunkt nicht die Firmenadresse, sondern die Stadt, in der der Kollege lebt. Deren Karte bedeutet ihm mehr als ein Büro, das er nie betreten hat.",
          "Leben die Kollegen in verschiedenen Städten, kann jeder einen Anhänger mit seiner eigenen Stadt bestellen und signieren — zusammen ergibt das ein Set „woher wir alle kommen“.",
        ] },
        { h2: "Wann es fertig ist", p: [
          "Druck und Versand innerhalb der Ukraine dauern 2–4 Werktage, Zahlung online. Arbeitet das Team außerhalb der Ukraine, kauft die 3MF-Datei (≈3 €) und lasst sie vor Ort drucken.",
          "Für eine 3D-Karte als Hauptgeschenk bestellt man am besten mindestens eine Woche vor dem letzten Arbeitstag — so bleibt ein Puffer.",
        ] },
      ],
      ctaLabel: "Karte für den Kollegen erstellen",
      ctaHref: "/create?product=map3d",
      outro: "Die neue Adresse ist noch unbekannt? Ein Stadt- oder Viertelname reicht — Details lassen sich später anpassen und die Vorschau kostenlos neu erstellen.",
    },
    pl: {
      title: "Prezent pożegnalny dla współpracownika, który odchodzi lub się przeprowadza: mapa miasta",
      description: "Co dać koledze z pracy, który odchodzi lub się przeprowadza? Mapę 3D dzielnicy biura albo jego nowego miasta. Zrzutka zespołu na breloki lub magnesy.",
      h1: "Kolega odchodzi: prezent od zespołu, który nie wyląduje w szufladzie",
      intro: "Kartka podpisana przez całe biuro zwykle ląduje w szufladzie. Działa za to coś związanego z konkretnym miejscem: dzielnica, w której stało biuro i gdzie jadaliście razem lunch, albo nowe miasto, do którego kolega się przenosi. Na taki prezent zespołowi łatwiej też się zgodzić niż na abstrakcyjną rzecz.",
      sections: [
        { h2: "Dwa pomysły na sytuację", p: [
          "Jeśli kolega odchodzi, ale zostaje w mieście — mapa 3D kwartału biura: ta sama ulica, kawiarnia na rogu, park przerw na lunch.",
          "Jeśli przeprowadza się do innego miasta — mapa nowej dzielnicy lub centrum, gotowa jeszcze przed przeprowadzką, plus brelok z pożegnalnym grawerem.",
          "Jest też przypadek pośredni: przeniesienie do innego oddziału tej samej firmy. Wtedy pasują dwie małe mapy — stare i nowe biuro — jako symboliczne przekazanie.",
        ] },
        { h2: "Opcja zrzutki", p: [
          "Zamiast jednej dużej mapy „od wszystkich” zespół może złożyć się na zestaw: dla każdego brelok lub magnes z tą samą dzielnicą, a jako główny prezent mapa 3D dla odchodzącego.",
          "To budżetowe rozwiązanie: brelok od ≈4 €, magnes ≈5 €, więc zestaw dla 5–10 osób pozostaje przystępny nawet obok osobnej mapy 3D.",
        ] },
        { h2: "Co wygrawerować", p: [
          "Ostatni dzień w biurze, nazwę zespołu albo po prostu rok — powściągliwie i trafnie. Osobiste życzenia zostaw na kartkę, na mapie — neutralny grawer.",
          "Firmy, które regularnie obdarowują zespoły, mogą skorzystać z formatu zamówień firmowych na /corporate — kilka identycznych zestawów naraz zamiast składania zamówienia od zera.",
        ] },
        { h2: "Jak zrobić to w 5 minut", p: [
          "Wybierz produkt — brelok, magnes lub mapę 3D — znajdź dzielnicę biura lub nowe miasto kolegi i przeciągnij ramkę na właściwy kwartał.",
          "Darmowy podgląd 3D powstaje w 1–2 minuty — udostępnij go na czacie zespołu, żeby wszyscy zobaczyli efekt przed zamówieniem.",
        ] },
        { h2: "Gdy zespół jest rozproszony po miastach", p: [
          "Dla zespołu zdalnego bez wspólnego biura punktem odniesienia nie jest adres firmy, lecz miasto, w którym mieszka odchodzący kolega. Mapa tego miasta znaczy dla niego więcej niż biuro, w którym nigdy nie był.",
          "Jeśli koledzy mieszkają w różnych miastach, każdy może zamówić brelok z mapą swojego miasta i go podpisać — razem tworzą zestaw „skąd wszyscy jesteśmy”.",
        ] },
        { h2: "Kiedy będzie gotowe", p: [
          "Druk i wysyłka na terenie Ukrainy trwają 2–4 dni robocze, płatność online. Jeśli zespół pracuje poza Ukrainą, kupcie plik 3MF (≈3 €) i wydrukujcie go na miejscu.",
          "Mapę 3D jako główny prezent warto zamówić co najmniej tydzień przed ostatnim dniem pracy — zostaje zapas.",
        ] },
      ],
      ctaLabel: "Stwórz mapę dla kolegi",
      ctaHref: "/create?product=map3d",
      outro: "Nie znasz jeszcze nowego adresu? Wystarczy nazwa miasta lub dzielnicy — szczegóły można doprecyzować później i bezpłatnie wygenerować podgląd ponownie.",
    },
    fr: {
      title: "Cadeau de départ pour un collègue qui part ou déménage : une carte de ville",
      description: "Quoi offrir à un collègue qui quitte l'entreprise ou déménage ? La carte 3D du quartier du bureau ou de sa nouvelle ville. Cagnotte d'équipe pour des porte-clés ou aimants.",
      h1: "Un collègue s'en va : un cadeau d'équipe qui ne finira pas dans un tiroir",
      intro: "Une carte signée par tout le bureau finit généralement dans un tiroir. Ce qui marche, c'est un objet lié à un lieu précis : le quartier du bureau où vous déjeuniez ensemble, ou la nouvelle ville où part le collègue. Un tel cadeau est aussi plus facile à choisir en équipe qu'un objet abstrait.",
      sections: [
        { h2: "Deux idées selon la situation", p: [
          "Si le collègue part mais reste en ville — une carte 3D du pâté de maisons du bureau : la même rue, le café du coin, le parc des pauses déjeuner.",
          "S'il déménage dans une autre ville — une carte du nouveau quartier ou du centre, prête avant même le déménagement, plus un porte-clés avec un message d'adieu gravé.",
          "Cas intermédiaire : une mutation dans une autre agence de la même entreprise. Deux petites cartes — l'ancien bureau et le nouveau — font une passation symbolique.",
        ] },
        { h2: "La cagnotte", p: [
          "Au lieu d'une grande carte « de la part de tous », l'équipe peut financer un lot : un porte-clés ou un aimant du même quartier pour chacun, et une carte 3D comme cadeau principal pour celui qui part.",
          "C'est abordable : porte-clés dès ≈4 €, aimant ≈5 €, donc un lot pour 5 à 10 personnes reste raisonnable même avec une carte 3D en plus.",
        ] },
        { h2: "Que graver", p: [
          "Le dernier jour au bureau, le nom de l'équipe ou simplement l'année — sobre et juste. Les vœux personnels vont sur la carte de vœux, la gravure reste neutre.",
          "Les entreprises qui offrent régulièrement des cadeaux à leurs équipes peuvent passer par le format commandes d'entreprise sur /corporate — plusieurs lots identiques d'un coup.",
        ] },
        { h2: "Le faire en 5 minutes", p: [
          "Choisis un produit — porte-clés, aimant ou carte 3D —, trouve le quartier du bureau ou la nouvelle ville du collègue et fais glisser le cadre sur le bon pâté de maisons.",
          "Un aperçu 3D gratuit se construit en 1 à 2 minutes — partage-le dans le chat de l'équipe pour que tout le monde voie le résultat avant de commander.",
        ] },
        { h2: "Si l'équipe est répartie dans plusieurs villes", p: [
          "Pour une équipe à distance sans bureau commun, le point de repère n'est pas l'adresse de l'entreprise mais la ville où vit le collègue. Sa carte comptera plus qu'un bureau où il n'a jamais mis les pieds.",
          "Si les collègues vivent dans des villes différentes, chacun peut commander un porte-clés de sa propre ville et le signer — ensemble, ils forment un lot « d'où nous venons tous ».",
        ] },
        { h2: "Quand ce sera prêt", p: [
          "Impression et livraison en Ukraine : 2–4 jours ouvrés, paiement en ligne. Si l'équipe travaille hors d'Ukraine, achetez le fichier 3MF (≈3 €) et faites-le imprimer sur place.",
          "Pour une carte 3D comme cadeau principal, mieux vaut commander au moins une semaine avant le dernier jour — cela laisse une marge.",
        ] },
      ],
      ctaLabel: "Créer une carte pour un collègue",
      ctaHref: "/create?product=map3d",
      outro: "Tu ne connais pas encore la nouvelle adresse ? Un nom de ville ou de quartier suffit — les détails s'ajustent plus tard et l'aperçu se régénère gratuitement.",
    },
    es: {
      title: "Regalo de despedida para un compañero que se va o se muda: un mapa de su ciudad",
      description: "Qué regalar a un compañero que deja la empresa o se muda: un mapa 3D del barrio de la oficina o de su nueva ciudad. Opción de bote de equipo para llaveros o imanes.",
      h1: "Un compañero se va: un regalo de equipo que no acabará en un cajón",
      intro: "Una tarjeta firmada por toda la oficina suele acabar en un cajón. Lo que funciona es algo ligado a un lugar concreto: el barrio donde estaba la oficina y donde comíais juntos, o la nueva ciudad a la que se muda. Además, es más fácil que todo el equipo se ponga de acuerdo en un regalo así que en un objeto abstracto.",
      sections: [
        { h2: "Dos ideas según la situación", p: [
          "Si deja la empresa pero se queda en la ciudad: un mapa 3D de la manzana de la oficina, con la misma calle, la cafetería de la esquina y el parque de las pausas para comer.",
          "Si se muda a otra ciudad: un mapa del nuevo barrio o del centro, listo incluso antes de la mudanza, y un llavero con un mensaje de despedida grabado.",
          "Hay un caso intermedio: un traslado a otra sede de la misma empresa. Entonces encajan dos mapas pequeños, la oficina antigua y la nueva, como un relevo simbólico.",
        ] },
        { h2: "Hacer un bote", p: [
          "En vez de un mapa grande «de parte de todos», el equipo puede financiar un set: un llavero o imán del mismo barrio para cada uno y un mapa 3D como regalo principal para quien se va.",
          "Sale económico: llavero desde ≈4 € e imán ≈5 €, así que un set para 5–10 personas sigue siendo asequible aunque se añada un mapa 3D.",
        ] },
        { h2: "Qué grabar", p: [
          "El último día en la oficina, el nombre del equipo o simplemente el año: sobrio y adecuado. Los deseos personales, en la tarjeta; en el mapa, un grabado neutro.",
          "Las empresas que regalan con frecuencia a sus equipos pueden usar el formato de pedidos corporativos en /corporate: varios sets iguales de una vez.",
        ] },
        { h2: "Cómo hacerlo en 5 minutos", p: [
          "Elige un producto (llavero, imán o mapa 3D), busca el barrio de la oficina o la nueva ciudad de tu compañero y arrastra el marco a la manzana correcta.",
          "La vista previa 3D gratis se genera en 1–2 minutos: compártela en el chat del equipo para que todos vean el resultado antes de pedir.",
        ] },
        { h2: "Si el equipo está repartido por varias ciudades", p: [
          "En un equipo remoto sin oficina común, la referencia no es la dirección de la empresa sino la ciudad donde vive quien se va. El mapa de esa ciudad significará más que una oficina que nunca pisó.",
          "Si los compañeros viven en ciudades distintas, cada uno puede pedir un llavero con su propia ciudad y firmarlo: juntos forman un set de «de dónde somos todos».",
        ] },
        { h2: "Cuándo estará listo", p: [
          "La impresión y el envío dentro de Ucrania tardan 2–4 días hábiles, con pago online. Si el equipo trabaja fuera de Ucrania, comprad el archivo 3MF (≈3 €) e imprimidlo allí.",
          "Para un mapa 3D como regalo principal, conviene pedirlo al menos una semana antes del último día: así queda margen.",
        ] },
      ],
      ctaLabel: "Crear un mapa para un compañero",
      ctaHref: "/create?product=map3d",
      outro: "¿Aún no sabes la nueva dirección? Basta con el nombre de la ciudad o del barrio: los detalles se ajustan después y la vista previa se regenera gratis.",
    },
  },

  "podarunok-molodyatam-na-vesillya": {
    de: {
      title: "Hochzeitsgeschenk für das Brautpaar: 3D-Karte ihrer Stadt oder ihres Kennenlernorts",
      description: "Geschenkidee zur Hochzeit: eine 3D-Karte der Hochzeitsstadt oder des Blocks, in dem sich das Paar kennengelernt hat. Ein passendes Anhänger-Paar mit Datum.",
      h1: "Ein Hochzeitsgeschenk, das nicht zwischen den Umschlägen untergeht",
      intro: "Auf den meisten Hochzeiten schenken Gäste Umschläge oder Standard-Geschirrsets. Herauszustechen ist leicht: Schenke eine Karte von einem Ort, der dem Paar etwas bedeutet — der Block, wo sie sich kennengelernt haben, die Straße des ersten Dates oder die Stadt, in der genau diese Hochzeit stattfindet. So ein Geschenk verwechselt niemand, weil es für die Geschichte genau dieses Paares gemacht ist.",
      sections: [
        { h2: "Zwei Ideen, die funktionieren", p: [
          "Eine 3D-Karte des Kennenlern-Blocks oder der Hochzeitsstadt — echte Gebäude, Straßen und ein Park im Regal der neuen Wohnung. Das Paar erkennt den Ort sofort und erzählt die Geschichte jedem Gast, der fragt.",
          "Ein passendes Anhänger-Paar: einer mit dem Viertel des Bräutigams, einer mit dem der Braut, beide mit dem Hochzeitsdatum auf der Rückseite.",
          "Beides lässt sich kombinieren: die Regalkarte als Hauptgeschenk, das Anhänger-Paar als kleine Ergänzung.",
        ] },
        { h2: "Eine Datumsgravur, die es einzigartig macht", p: [
          "Hochzeitsdatum, Namen oder die Koordinaten des Kennenlernorts fügst du direkt im Konfigurator hinzu — auf einer flachen Karte, einem Magneten oder der Rückseite eines Anhängers.",
          "Die Gravur kann kurz sein — nur das Datum — oder ausführlicher mit beiden Namen; wichtig ist, dass sie auch in Jahren gut lesbar bleibt.",
        ] },
        { h2: "Preise und Formate für jedes Budget", p: [
          "Ein Karten-Anhänger ab ≈4 €; ein Paar sind einfach zwei Stück. Ein Kühlschrankmagnet ≈5 € als kleine Ergänzung. Eine 3D-Tischkarte von ≈8 € (5,5 cm) bis ≈18 € (15 cm), je nachdem, wie viel von der Stadt sie zeigen soll.",
          "Ein Kachelbild für die Wand passt, wenn die Hochzeit groß ist und mehrere Gäste ein gemeinsames Geschenk für die neue Wohnung machen wollen.",
        ] },
        { h2: "In 5 Minuten erstellt", p: [
          "Produkt wählen — Karte oder Anhänger —, den Kennenlernort oder die Hochzeitsstadt finden, die kostenlose 3D-Vorschau bestätigen (1–2 Minuten) und die Datumsgravur hinzufügen.",
          "Dann den Druck mit Versand innerhalb der Ukraine bestellen — oder die 3MF-Datei (≈3 €) herunterladen, wenn du das Geschenk selbst oder vor Ort drucken willst.",
        ] },
        { h2: "Wenn mehrere Gäste zusammenlegen", p: [
          "Ein paar Freunde oder Verwandte können für eine größere Tischkarte (L oder XL) oder ein Kachelbild zusammenlegen — dann wirkt das Geschenk wie ein gemeinsamer Beitrag statt eines Umschlags.",
          "Die Karte entsteht aus OpenStreetMap-Daten: echte Gebäude, Straßen, Wasser und Parks des gewählten Ortes, keine allgemeine Stadtillustration.",
        ] },
      ],
      ctaLabel: "Karte für das Brautpaar erstellen",
      ctaHref: "/create?product=map3d",
      outro: "Zahlung online; gedruckte Modelle versenden wir nur innerhalb der Ukraine, 2–4 Werktage nach dem Druck. Kennst du die genaue Adresse nicht, reicht ein Café-, Park- oder Straßenname — die Kartensuche findet den Ort.",
    },
    pl: {
      title: "Prezent ślubny dla młodej pary: mapa 3D ich miasta lub miejsca poznania",
      description: "Pomysł na prezent ślubny: mapa 3D miasta, w którym odbywa się ślub, albo kwartału, gdzie para się poznała. Para breloków z datą.",
      h1: "Prezent ślubny, który nie zginie wśród kopert",
      intro: "Na większości wesel goście dają koperty albo standardowe zestawy naczyń. Łatwo się wyróżnić: podaruj mapę miejsca, które coś znaczy dla pary — kwartału, gdzie się poznali, ulicy pierwszej randki albo miasta, w którym odbywa się to właśnie wesele. Takiego prezentu nie da się pomylić z innym, bo powstaje dla historii tej konkretnej pary.",
      sections: [
        { h2: "Dwa pomysły, które działają", p: [
          "Mapa 3D kwartału, gdzie się poznali, albo miasta ślubu — prawdziwe budynki, ulice i park na półce w nowym mieszkaniu. Para od razu rozpozna miejsce i opowie historię każdemu gościowi, który zapyta.",
          "Para breloków: jeden z dzielnicą pana młodego, drugi z dzielnicą panny młodej, oba z datą ślubu na odwrocie.",
          "Oba pomysły łatwo połączyć: mapa na półkę jako główny prezent i para breloków jako mniejszy dodatek.",
        ] },
        { h2: "Grawer z datą, który czyni go wyjątkowym", p: [
          "Datę ślubu, imiona lub współrzędne miejsca poznania dodasz bezpośrednio w kreatorze — na płaskiej mapie, magnesie albo odwrocie breloka.",
          "Grawer może być krótki — tylko data — albo pełniejszy, z oboma imionami; ważne, żeby był czytelny także za lata.",
        ] },
        { h2: "Ceny i formaty na każdy budżet", p: [
          "Brelok z mapą od ≈4 €; para to po prostu dwie sztuki. Magnes na lodówkę ≈5 € jako mały dodatek. Mapa 3D na biurko od ≈8 € (5,5 cm) do ≈18 € (15 cm), zależnie od tego, ile miasta ma pokazać.",
          "Panel ścienny z płytek sprawdzi się, gdy wesele jest duże i kilku gości chce złożyć się na wspólny prezent do nowego mieszkania.",
        ] },
        { h2: "Jak zrobić to w 5 minut", p: [
          "Wybierz produkt — mapę lub brelok — znajdź miejsce poznania pary lub miasto ślubu, zatwierdź darmowy podgląd 3D (1–2 minuty) i dodaj grawer z datą.",
          "Potem zamów druk z wysyłką na terenie Ukrainy — albo pobierz plik 3MF (≈3 €), jeśli chcesz wydrukować prezent sam lub na miejscu.",
        ] },
        { h2: "Gdy kilku gości się składa", p: [
          "Kilkoro przyjaciół lub krewnych może złożyć się na większą mapę (L lub XL) albo panel z płytek — wtedy prezent jest wspólnym wkładem, a nie kolejną kopertą.",
          "Mapa powstaje z danych OpenStreetMap: prawdziwe budynki, ulice, woda i parki wybranego miejsca, a nie ogólna ilustracja miasta.",
        ] },
      ],
      ctaLabel: "Stwórz mapę dla młodej pary",
      ctaHref: "/create?product=map3d",
      outro: "Płatność online; wydrukowane modele wysyłamy tylko na terenie Ukrainy, 2–4 dni robocze po druku. Jeśli nie znasz dokładnego adresu, wystarczy nazwa kawiarni, parku lub ulicy — wyszukiwarka znajdzie miejsce.",
    },
    fr: {
      title: "Cadeau de mariage pour les jeunes mariés : la carte 3D de leur ville ou de leur rencontre",
      description: "Idée cadeau de mariage : la carte 3D de la ville du mariage ou du quartier où le couple s'est rencontré. Une paire de porte-clés assortis avec la date.",
      h1: "Un cadeau de mariage qui ne se perdra pas parmi les enveloppes",
      intro: "À la plupart des mariages, les invités offrent des enveloppes ou des services de vaisselle standard. Se démarquer est facile : offre la carte d'un lieu qui compte pour le couple — le quartier de leur rencontre, la rue du premier rendez-vous ou la ville où se déroule ce mariage. Un tel cadeau ne peut être confondu avec aucun autre, car il est fait pour l'histoire de ce couple précis.",
      sections: [
        { h2: "Deux idées qui marchent", p: [
          "Une carte 3D du quartier de leur rencontre ou de la ville du mariage — vrais bâtiments, rues et parc sur une étagère du nouvel appartement. Le couple reconnaîtra le lieu immédiatement et racontera l'histoire aux invités qui demanderont.",
          "Une paire de porte-clés assortis : l'un avec le quartier du marié, l'autre avec celui de la mariée, chacun avec la date du mariage au dos.",
          "Les deux idées se combinent : la carte pour l'étagère en cadeau principal, la paire de porte-clés en complément.",
        ] },
        { h2: "Une date gravée qui le rend unique", p: [
          "La date du mariage, les prénoms ou les coordonnées du lieu de rencontre s'ajoutent directement dans le configurateur — sur une carte plate, un aimant ou au dos d'un porte-clés.",
          "La gravure peut être courte — juste la date — ou plus complète avec les deux prénoms ; l'important est qu'elle reste lisible dans des années.",
        ] },
        { h2: "Prix et formats pour tous les budgets", p: [
          "Porte-clés carte dès ≈4 € ; une paire, c'est simplement deux pièces. Aimant de frigo ≈5 € en petit complément. Carte 3D de bureau de ≈8 € (5,5 cm) à ≈18 € (15 cm), selon la part de ville à montrer.",
          "Un tableau mural en tuiles convient pour un grand mariage où plusieurs invités veulent se regrouper pour un cadeau commun.",
        ] },
        { h2: "Le faire en 5 minutes", p: [
          "Choisis un produit — carte ou porte-clés —, trouve le lieu de rencontre ou la ville du mariage, valide l'aperçu 3D gratuit (1 à 2 minutes) et ajoute la date gravée.",
          "Ensuite, commande l'impression livrée en Ukraine — ou télécharge le fichier 3MF (≈3 €) pour imprimer le cadeau toi-même ou près de chez toi.",
        ] },
        { h2: "Quand plusieurs invités se cotisent", p: [
          "Quelques amis ou proches peuvent se cotiser pour une carte plus grande (L ou XL) ou un tableau en tuiles — le cadeau devient une contribution commune plutôt qu'une enveloppe.",
          "La carte est construite à partir des données OpenStreetMap : les vrais bâtiments, rues, cours d'eau et parcs du lieu choisi, pas une illustration générique.",
        ] },
      ],
      ctaLabel: "Créer une carte pour les jeunes mariés",
      ctaHref: "/create?product=map3d",
      outro: "Paiement en ligne ; les modèles imprimés ne sont livrés qu'en Ukraine, 2–4 jours ouvrés après l'impression. Si tu ne connais pas l'adresse exacte, un nom de café, de parc ou de rue suffit — la recherche trouvera le lieu.",
    },
    es: {
      title: "Regalo de boda para los novios: un mapa 3D de su ciudad o del lugar donde se conocieron",
      description: "Idea de regalo de boda: un mapa 3D de la ciudad de la boda o de la manzana donde la pareja se conoció. Un par de llaveros a juego con la fecha.",
      h1: "Un regalo de boda que no se perderá entre los sobres",
      intro: "En la mayoría de bodas los invitados regalan sobres o juegos de vajilla estándar. Destacar es fácil: regala el mapa de un lugar que signifique algo para la pareja, como la manzana donde se conocieron, la calle de la primera cita o la ciudad donde se celebra esta boda. Un regalo así no se confunde con ningún otro, porque está hecho para la historia de esa pareja.",
      sections: [
        { h2: "Dos ideas que funcionan", p: [
          "Un mapa 3D de la manzana donde se conocieron o de la ciudad de la boda: edificios reales, calles y un parque en una estantería del piso nuevo. La pareja reconocerá el lugar al instante y contará la historia a cada invitado que pregunte.",
          "Un par de llaveros a juego: uno con el barrio del novio y otro con el de la novia, ambos con la fecha de la boda grabada detrás.",
          "Las dos ideas se combinan fácilmente: el mapa para la estantería como regalo principal y el par de llaveros como complemento.",
        ] },
        { h2: "Una fecha grabada que lo hace único", p: [
          "La fecha de la boda, los nombres o las coordenadas del lugar donde se conocieron se añaden directamente en el configurador, en un mapa plano, un imán o el reverso de un llavero.",
          "El grabado puede ser corto, solo la fecha, o más completo con ambos nombres; lo importante es que siga leyéndose bien dentro de años.",
        ] },
        { h2: "Precios y formatos para cualquier presupuesto", p: [
          "Llavero con mapa desde ≈4 €; un par son simplemente dos. Imán de nevera ≈5 € como pequeño complemento. Mapa 3D de escritorio de ≈8 € (5,5 cm) a ≈18 € (15 cm), según cuánta ciudad quieras mostrar.",
          "Un panel de pared de baldosas funciona cuando la boda es grande y varios invitados quieren juntarse para un regalo común para el piso nuevo.",
        ] },
        { h2: "Cómo hacerlo en 5 minutos", p: [
          "Elige un producto (mapa o llavero), busca el lugar donde se conocieron o la ciudad de la boda, confirma la vista previa 3D gratis (1–2 minutos) y añade la fecha grabada.",
          "Después pide la impresión con envío dentro de Ucrania, o descarga el archivo 3MF (≈3 €) si quieres imprimir el regalo tú o cerca de ti.",
        ] },
        { h2: "Cuando varios invitados ponen dinero", p: [
          "Unos cuantos amigos o familiares pueden juntarse para un mapa más grande (L o XL) o un panel de baldosas: así el regalo es una aportación común y no otro sobre.",
          "El mapa se construye con datos de OpenStreetMap: los edificios, calles, agua y parques reales del lugar elegido, no una ilustración genérica de una ciudad.",
        ] },
      ],
      ctaLabel: "Crear un mapa para los novios",
      ctaHref: "/create?product=map3d",
      outro: "Pago online; los modelos impresos solo se envían dentro de Ucrania, 2–4 días hábiles después de imprimir. Si no sabes la dirección exacta, basta con el nombre de una cafetería, un parque o una calle: el buscador encontrará el lugar.",
    },
  },
};
