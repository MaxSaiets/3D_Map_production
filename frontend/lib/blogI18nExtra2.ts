// Переклади статей блогу de/pl/fr/es — частина 2 (правила — див. lib/blogI18nExtra.ts).
import type { BlogArticleContent } from "@/lib/blog";
import type { BlogExtraLocale } from "@/lib/blogI18nExtra";

export const BLOG_I18N_EXTRA_2: Record<string, Partial<Record<BlogExtraLocale, BlogArticleContent>>> = {
  "podarunok-na-novosillya-druzyam": {
    de: {
      title: "Einzugsgeschenk für Freunde: eine Karte des neuen Viertels statt noch einer Topfpflanze",
      description: "Was schenkt man zum Einzug? Eine 3D-Karte des Blocks der neuen Wohnung oder einen Magneten mit dem neuen Viertel. Die erste Deko in der leeren Wohnung.",
      h1: "Ein Einzugsgeschenk, das am ersten Tag zur Deko wird",
      intro: "Eine neue Wohnung hat leere Regale und Wände — und eine Topfpflanze ändert daran nichts. Eine Karte des neuen Viertels mit echten Gebäuden, dem Park und der Straße, in der eure Freunde jetzt wohnen, bekommt schon am ersten Abend ihren Platz im Regal.",
      sections: [
        { h2: "Was man wählt", p: [
          "Ein Magnet mit dem neuen Viertel (≈5 €) — günstig, passend und direkt an den Kühlschrank.",
          "Eine 3D-Karte des Blocks (ab ≈8 €) — fürs Wohnzimmerregal; markiere ihr Haus mit einem roten Einsatz, damit Gäste sehen: „Das sind wir“.",
          "Ein Kachelbild für die Wand — für eine große Einweihung, wenn mehrere Freunde zusammenlegen.",
        ] },
        { h2: "Text auf der Karte", p: [
          "Der Straßenname, das Einzugsdatum oder einfach „Home“. Im Konfigurator mit einem Klick hinzugefügt, gedruckt auf einer flachen Karte oder einem Magneten.",
        ] },
        { h2: "Genaue Adresse unbekannt?", p: [
          "Straße und Stadt reichen — die Kartensuche findet sie, und du ziehst den Rahmen aufs richtige Haus. Die 3D-Vorschau ist kostenlos; bestellt wird erst, wenn du das Ergebnis gesehen hast. Druck mit Versand innerhalb der Ukraine; außerhalb der Ukraine die 3MF-Datei (≈3 €) zum Drucken vor Ort.",
        ] },
      ],
      ctaLabel: "Karte des neuen Viertels erstellen",
      ctaHref: "/podarunok/na-novosillya",
    },
    pl: {
      title: "Prezent na parapetówkę dla przyjaciół: mapa nowej dzielnicy zamiast kolejnego kwiatka",
      description: "Co dać na parapetówkę? Mapę 3D kwartału nowego mieszkania albo magnes z nową dzielnicą. Pierwsza ozdoba w pustym mieszkaniu.",
      h1: "Prezent na parapetówkę, który od pierwszego dnia staje się ozdobą",
      intro: "Nowe mieszkanie ma puste półki i ściany — i kwiatek w doniczce tego nie zmieni. Mapa nowej dzielnicy z prawdziwymi budynkami, parkiem i ulicą, przy której teraz mieszkają przyjaciele, zajmie miejsce na półce już pierwszego wieczoru.",
      sections: [
        { h2: "Co wybrać", p: [
          "Magnes z nową dzielnicą (≈5 €) — niedrogi, trafiony i od razu na lodówkę.",
          "Mapa 3D kwartału (od ≈8 €) — na półkę w salonie; zaznacz ich budynek czerwoną wstawką, żeby goście widzieli „to my”.",
          "Panel ścienny z płytek — na dużą parapetówkę, gdy składa się kilku przyjaciół.",
        ] },
        { h2: "Tekst na mapie", p: [
          "Nazwa ulicy, data przeprowadzki albo po prostu „Home”. Dodajesz jednym kliknięciem w kreatorze, drukowany na płaskiej mapie lub magnesie.",
        ] },
        { h2: "Nie znasz dokładnego adresu?", p: [
          "Wystarczy ulica i miasto — wyszukiwarka mapy je znajdzie, a ramkę przeciągniesz na właściwy budynek. Podgląd 3D jest darmowy; zamawiasz dopiero po zobaczeniu efektu. Druk z wysyłką na terenie Ukrainy; poza Ukrainą — plik 3MF (≈3 €) do druku na miejscu.",
        ] },
      ],
      ctaLabel: "Stwórz mapę nowej dzielnicy",
      ctaHref: "/podarunok/na-novosillya",
    },
    fr: {
      title: "Cadeau de crémaillère pour des amis : la carte du nouveau quartier plutôt qu'une plante verte",
      description: "Quoi offrir pour une crémaillère ? La carte 3D du pâté de maisons du nouvel appartement ou un aimant du nouveau quartier. La première déco d'un logement vide.",
      h1: "Un cadeau de crémaillère qui devient déco dès le premier jour",
      intro: "Un nouvel appartement, ce sont des étagères et des murs vides — et une plante verte n'y change rien. La carte du nouveau quartier avec les vrais bâtiments, le parc et la rue où vivent désormais tes amis trouve sa place sur l'étagère dès le premier soir.",
      sections: [
        { h2: "Que choisir", p: [
          "Un aimant du nouveau quartier (≈5 €) — abordable, bien vu et directement sur le frigo.",
          "Une carte 3D du pâté de maisons (dès ≈8 €) — pour l'étagère du salon ; marque leur immeuble d'un insert rouge pour que les invités voient « c'est nous ».",
          "Un tableau mural en tuiles — pour une grande crémaillère quand plusieurs amis se cotisent.",
        ] },
        { h2: "Un texte sur la carte", p: [
          "Le nom de la rue, la date d'emménagement ou simplement « Home ». Ajouté en un clic dans le configurateur, imprimé sur une carte plate ou un aimant.",
        ] },
        { h2: "Tu ne connais pas l'adresse exacte ?", p: [
          "Une rue et une ville suffisent — la recherche la trouve et tu fais glisser le cadre sur le bon immeuble. L'aperçu 3D est gratuit ; tu commandes après avoir vu le résultat. Impression livrée en Ukraine ; hors d'Ukraine, le fichier 3MF (≈3 €) à imprimer sur place.",
        ] },
      ],
      ctaLabel: "Créer la carte du nouveau quartier",
      ctaHref: "/podarunok/na-novosillya",
    },
    es: {
      title: "Regalo de inauguración de casa para amigos: el mapa del nuevo barrio en vez de otra planta",
      description: "Qué regalar al estrenar casa: un mapa 3D de la manzana del piso nuevo o un imán con el nuevo barrio. La primera decoración de un piso vacío.",
      h1: "Un regalo de inauguración que es decoración desde el primer día",
      intro: "Un piso nuevo tiene estanterías y paredes vacías, y una planta no lo arregla. Un mapa del nuevo barrio con los edificios reales, el parque y la calle donde ahora viven tus amigos ocupa su sitio en la estantería la primera noche.",
      sections: [
        { h2: "Qué elegir", p: [
          "Un imán con el nuevo barrio (≈5 €): barato, acertado y directo a la nevera.",
          "Un mapa 3D de la manzana (desde ≈8 €): para la estantería del salón; marca su edificio con una pieza roja para que las visitas vean «somos nosotros».",
          "Un panel de pared de baldosas: para una gran inauguración en la que ponen dinero varios amigos.",
        ] },
        { h2: "Texto en el mapa", p: [
          "El nombre de la calle, la fecha de la mudanza o simplemente «Home». Se añade con un clic en el configurador y se imprime en un mapa plano o un imán.",
        ] },
        { h2: "¿No sabes la dirección exacta?", p: [
          "Basta con la calle y la ciudad: el buscador la encuentra y arrastras el marco al edificio correcto. La vista previa 3D es gratis; pides solo después de ver el resultado. Impresión con envío dentro de Ucrania; fuera de Ucrania, el archivo 3MF (≈3 €) para imprimir allí.",
        ] },
      ],
      ctaLabel: "Crear el mapa del nuevo barrio",
      ctaHref: "/podarunok/na-novosillya",
    },
  },

  "podarunok-tomu-khto-lyubyt-svoye-misto": {
    de: {
      title: "Ein Geschenk für jemanden, der seine Stadt liebt: 3D-Karte statt Souvenirmagnet",
      description: "Ein originelles Geschenk für Stadtverliebte: eine 3D-Karte des Lieblingsviertels, eine Reliefkarte mit Hügeln oder ein Anhänger mit der eigenen Straße. Für jede Stadt.",
      h1: "Ein Geschenk für jemanden, der seine Stadt liebt",
      intro: "Manche Menschen kennen jeden Hof und jedes Café ihres Viertels — und sind stolz darauf. Ein Magnet vom Souvenirstand überrascht sie nicht. Eine 3D-Karte genau ihres Blocks — mit Gebäuden in echter Höhe, dem Park und dem Fluss — ganz sicher.",
      sections: [
        { h2: "Welche Karte", p: [
          "Die 3D-Stadtkarte ist der Klassiker: Gebäude, Straßen, Parks. Wähle ein Viertel, nicht „die ganze Stadt“: 400–800 m gemischte Straßen erkennt man auf einen Blick.",
          "Eine Reliefkarte — für hügelige Städte. Hänge und Höhenlinien werden Teil des Modells.",
          "Ein Anhänger oder Magnet — wenn es klein, aber persönlich sein soll: die Straße, in der jemand aufgewachsen ist, immer am Schlüsselbund.",
        ] },
        { h2: "Nicht nur Großstädte", p: [
          "Der Konfigurator funktioniert mit jedem Punkt auf der Karte — Kleinstadt, Dorf, Gartensiedlung. Wenn es dort Straßen und Gebäude in OpenStreetMap gibt, gibt es ein Modell.",
        ] },
        { h2: "Wie es in der Hand aussieht", p: [
          "Gedruckt aus umweltfreundlichem Eco PLA, 5,5 bis 15 cm. Vor der Bestellung siehst du eine kostenlose 3D-Vorschau — genau das, was ankommt, nur in Kunststoff. Außerhalb der Ukraine druckst du die 3MF-Datei (≈3 €) vor Ort.",
        ] },
      ],
      ctaLabel: "Karte des Lieblingsviertels erstellen",
      ctaHref: "/create",
      outro: "Wie gedruckte Karten verschiedener Städte aussehen, zeigt die Galerie echter Drucke.",
    },
    pl: {
      title: "Prezent dla kogoś, kto kocha swoje miasto: mapa 3D zamiast magnesu z targu",
      description: "Oryginalny prezent dla miłośnika swojego miasta: mapa 3D ulubionej dzielnicy, mapa z rzeźbą terenu albo brelok z rodzinną ulicą. Dla każdego miasta.",
      h1: "Prezent dla kogoś, kto kocha swoje miasto",
      intro: "Są ludzie, którzy znają każde podwórko i każdą kawiarnię w swojej dzielnicy — i są z tego dumni. Magnes z budki z pamiątkami ich nie zaskoczy. Mapa 3D dokładnie ich kwartału — z budynkami w prawdziwej wysokości, parkiem i rzeką — na pewno tak.",
      sections: [
        { h2: "Jaką mapę wybrać", p: [
          "Mapa 3D miasta to klasyka: budynki, ulice, parki. Wybierz dzielnicę, nie „całe miasto”: 400–800 m różnorodnych ulic rozpoznaje się od razu.",
          "Mapa z rzeźbą terenu — dla pagórkowatych miast. Zbocza i poziomice stają się częścią modelu.",
          "Brelok lub magnes — gdy chcesz czegoś małego, ale osobistego: ulica, na której ktoś dorastał, zawsze przy kluczach.",
        ] },
        { h2: "Nie tylko duże miasta", p: [
          "Kreator działa z każdym punktem na mapie — małe miasteczko, wieś, osiedle działkowe. Jeśli w OpenStreetMap są tam ulice i budynki, będzie i model.",
        ] },
        { h2: "Jak wygląda w dłoni", p: [
          "Drukowane z ekologicznego Eco PLA, od 5,5 do 15 cm. Przed zamówieniem widzisz darmowy podgląd 3D — dokładnie to, co przyjdzie, tylko w plastiku. Poza Ukrainą drukujesz plik 3MF (≈3 €) na miejscu.",
        ] },
      ],
      ctaLabel: "Stwórz mapę ulubionej dzielnicy",
      ctaHref: "/create",
      outro: "Jak wyglądają wydrukowane mapy różnych miast, zobaczysz w galerii prawdziwych wydruków.",
    },
    fr: {
      title: "Un cadeau pour quelqu'un qui aime sa ville : une carte 3D plutôt qu'un aimant souvenir",
      description: "Un cadeau original pour un amoureux de sa ville : la carte 3D de son quartier préféré, une carte en relief avec les collines ou un porte-clés avec sa rue. Pour toute ville.",
      h1: "Un cadeau pour quelqu'un qui aime sa ville",
      intro: "Certaines personnes connaissent chaque cour et chaque café de leur quartier — et en sont fières. Un aimant de boutique de souvenirs ne les surprendra pas. Une carte 3D de leur pâté de maisons précis — bâtiments à leur vraie hauteur, parc et rivière — oui.",
      sections: [
        { h2: "Quelle carte choisir", p: [
          "La carte 3D de ville est le classique : bâtiments, rues, parcs. Choisis un quartier, pas « toute la ville » : 400 à 800 m de rues variées se reconnaissent au premier coup d'œil.",
          "Une carte en relief — pour les villes vallonnées. Pentes et courbes de niveau font partie du modèle.",
          "Un porte-clés ou un aimant — pour un petit cadeau personnel : la rue où quelqu'un a grandi, toujours sur ses clés.",
        ] },
        { h2: "Pas seulement les grandes villes", p: [
          "Le configurateur fonctionne avec n'importe quel point de la carte — petite ville, village, lotissement. S'il y a des rues et des bâtiments dans OpenStreetMap, il y a un modèle.",
        ] },
        { h2: "À quoi ça ressemble en main", p: [
          "Imprimé en Eco PLA écologique, de 5,5 à 15 cm. Avant de commander, tu vois un aperçu 3D gratuit — exactement ce qui arrivera, mais en plastique. Hors d'Ukraine, tu imprimes le fichier 3MF (≈3 €) sur place.",
        ] },
      ],
      ctaLabel: "Créer la carte de son quartier préféré",
      ctaHref: "/create",
      outro: "La galerie des impressions réelles montre à quoi ressemblent les cartes de différentes villes.",
    },
    es: {
      title: "Un regalo para quien ama su ciudad: un mapa 3D en vez de un imán de recuerdo",
      description: "Un regalo original para quien ama su ciudad: un mapa 3D de su barrio favorito, un mapa con relieve y colinas o un llavero con su calle. Para cualquier ciudad.",
      h1: "Un regalo para alguien que ama su ciudad",
      intro: "Hay personas que conocen cada patio y cada café de su barrio, y están orgullosas de ello. Un imán del quiosco de recuerdos no las sorprenderá. Un mapa 3D justo de su manzana, con edificios a su altura real, el parque y el río, sí.",
      sections: [
        { h2: "Qué mapa elegir", p: [
          "El mapa 3D de ciudad es el clásico: edificios, calles, parques. Elige un barrio, no «toda la ciudad»: 400–800 m de calles variadas se reconocen a primera vista.",
          "Un mapa con relieve, para ciudades con colinas. Las laderas y curvas de nivel forman parte del modelo.",
          "Un llavero o un imán, cuando quieres algo pequeño pero personal: la calle donde creció alguien, siempre en las llaves.",
        ] },
        { h2: "No solo grandes ciudades", p: [
          "El configurador funciona con cualquier punto del mapa: un pueblo, una aldea, una urbanización. Si hay calles y edificios en OpenStreetMap, hay modelo.",
        ] },
        { h2: "Cómo se ve en la mano", p: [
          "Impreso en Eco PLA ecológico, de 5,5 a 15 cm. Antes de pedir ves una vista previa 3D gratis: exactamente lo que llegará, pero en plástico. Fuera de Ucrania imprimes el archivo 3MF (≈3 €) allí.",
        ] },
      ],
      ctaLabel: "Crear el mapa de su barrio favorito",
      ctaHref: "/create",
      outro: "En la galería de impresiones reales verás cómo quedan los mapas de distintas ciudades.",
    },
  },

  "podarunok-cholovikovi-na-richnytsyu-vesillya": {
    de: {
      title: "Hochzeitstagsgeschenk für den Mann: 3D-Karte eurer Stadt",
      description: "Geschenkidee zum Hochzeitstag für ihn: eine 3D-Karte des Ortes, an dem ihr euch kennengelernt oder geheiratet habt, oder ein Anhänger mit eingraviertem Datum.",
      h1: "Ein Hochzeitstagsgeschenk für ihn, das nicht nach einer Woche vergessen ist",
      intro: "Eine Krawatte kauft er sich selbst, und ein Massagegutschein landet in der Schublade. Was stattdessen wirkt, ist etwas, in dem euer Ort steckt: der Block, wo ihr euch kennengelernt habt, die Straße am Standesamt oder das Haus, in das ihr zusammen gezogen seid. Eine 3D-Karte dieses Ortes mit echten Gebäuden und ein Datum auf der Rückseite eines Anhängers — ein Geschenk, das auf dem Schreibtisch steht, nicht im Schrank, und keine Erklärung braucht.",
      sections: [
        { h2: "Warum eine Karte und nicht noch etwas von der Wunschliste", p: [
          "Eine Karte kann man nicht fertig kaufen — sie entsteht für bestimmte Koordinaten und existiert nur einmal. Auf dem Bürotisch oder im Regal wird sie sofort zum Anlass, die Geschichte zu erzählen: „Hier haben wir uns kennengelernt.“",
          "Sie funktioniert auch, wenn er „schon alles hat“: Ein Ort, der euch beiden etwas bedeutet, hängt nicht von Kleidergröße oder Gadget-Geschmack ab. Und man verwechselt sie nicht mit einem anderen Geschenk — anders als Parfüm oder noch ein Portemonnaie.",
        ] },
        { h2: "Welchen Ort wählen", p: [
          "Nicht die ganze Innenstadt — am stärksten ist ein konkreter Punkt: der Hof eurer ersten gemeinsamen Wohnung, die Straße mit dem Café vom ersten Date oder der Block am Standesamt.",
          "Seid ihr mehrmals umgezogen, nimm den Ort mit dem meisten Gefühl, nicht den neuesten. Den Rahmen kannst du auf einen einzigen Block verkleinern, damit Details, die er sofort erkennt, auch auf einem kleinen Modell gut sichtbar bleiben.",
        ] },
        { h2: "Drei Varianten für jedes Budget", p: [
          "Ein Anhänger mit der Karte eures Blocks (ab ≈4 €) — jeden Tag dabei, mit Hochzeitsdatum oder Koordinaten auf der Rückseite.",
          "Ein Kühlschrankmagnet (≈5 €) — eine kompakte 6-cm-Karte, eine symbolische, günstige Geste.",
          "Eine 3D-Karte (ab ≈8 €, 5,5–15 cm) — das Hauptgeschenk: Gebäude in echter Höhe, Straßen, der Park oder die Uferpromenade, wo ihr spazieren wart.",
        ] },
        { h2: "Eine Datumsgravur, die es zu eurem macht", p: [
          "Hochzeitsdatum, Kennenlerndatum oder einfach das Jahr — die einfachste und genaueste Gravur. Sie kommt auf eine flache Karte, einen Magneten oder als separate Platte auf das 3D-Modell.",
          "Als gemeinsames Symbol gibt es einen zweiteiligen Herz-Anhänger: Die eine Hälfte ist euer Viertel, die andere der Ort des Kennenlernens. Zusammen ergeben sie ein ganzes Herz, getrennt trägt jeder seine Hälfte.",
        ] },
        { h2: "In 5 Minuten erstellt", p: [
          "Wähle ein Produkt — Anhänger, Magnet oder 3D-Karte. Markiere den Ort auf der Karte: eine Adresse, einen Straßennamen oder das Café, in dem alles begann, und zieh den Rahmen über den richtigen Block.",
          "Nach 1–2 Minuten erscheint eine kostenlose 3D-Vorschau. Passt alles, füge die Gravur hinzu und bestelle den Druck (Versand innerhalb der Ukraine) — oder kaufe die 3MF-Datei (≈3 €) und lass sie vor Ort drucken.",
        ] },
        { h2: "Wann es fertig ist", p: [
          "Druck und Versand dauern 2–4 Werktage nach der Bestellung. Bezahlt wird online direkt auf der Website.",
          "Ist der Hochzeitstag schon morgen, wähle einen Anhänger — er ist am schnellsten gedruckt — oder die Datei für einen 3D-Druckservice in deiner Nähe.",
        ] },
      ],
      ctaLabel: "Karte eures Ortes erstellen",
      ctaHref: "/create?product=map3d",
      outro: "Keine genaue Adresse? Ein Straßen- oder Lokalname reicht — die Kartensuche schlägt den Ort vor, und den Rahmen ziehst du von Hand auf den richtigen Block.",
    },
    pl: {
      title: "Prezent dla męża na rocznicę ślubu: mapa 3D waszego miasta",
      description: "Pomysł na prezent dla niego na rocznicę ślubu: mapa 3D miejsca, gdzie się poznaliście lub pobraliście, albo brelok z wygrawerowaną datą.",
      h1: "Prezent na rocznicę ślubu dla niego, o którym nie zapomni po tygodniu",
      intro: "Krawat kupi sobie sam, a „voucher na masaż” wyląduje w szufladzie. Działa za to rzecz, w której zapisane jest wasze miejsce: kwartał, gdzie się poznaliście, ulica przy urzędzie stanu cywilnego albo budynek, do którego razem się wprowadziliście. Mapa 3D tego miejsca z prawdziwymi budynkami i data na odwrocie breloka — prezent, który stoi na biurku, a nie w szafie, i nie wymaga tłumaczenia.",
      sections: [
        { h2: "Dlaczego mapa, a nie kolejna rzecz z listy życzeń", p: [
          "Mapy nie kupisz gotowej — powstaje dla konkretnych współrzędnych, więc istnieje w jednym egzemplarzu. Na biurku w pracy czy na półce w domu od razu staje się pretekstem do opowieści: „tu się poznaliśmy”.",
          "Działa nawet wtedy, gdy on „ma wszystko”: miejsce ważne dla was dwojga nie zależy od rozmiaru ani gustu w gadżetach. Trudno też pomylić go z cudzym prezentem — w przeciwieństwie do perfum czy kolejnego portfela.",
        ] },
        { h2: "Które miejsce wybrać", p: [
          "Nie trzeba całego centrum — najmocniej działa jeden konkretny punkt: podwórko pierwszego wspólnego mieszkania, ulica z kawiarnią z pierwszej randki albo kwartał przy urzędzie, gdzie był ślub.",
          "Jeśli przeprowadzaliście się kilka razy, wybierz miejsce z największym ładunkiem emocji, a nie najnowsze. Ramkę można zawęzić do jednego kwartału, żeby szczegóły, które od razu rozpozna, były dobrze widoczne nawet na małym modelu.",
        ] },
        { h2: "Trzy opcje na każdy budżet", p: [
          "Brelok z mapą waszego kwartału (od ≈4 €) — noszony codziennie, z datą ślubu lub współrzędnymi na odwrocie.",
          "Magnes na lodówkę (≈5 €) — kompaktowa mapa 6 cm, symboliczny i niedrogi gest.",
          "Mapa 3D (od ≈8 €, 5,5–15 cm) — główny prezent: budynki w prawdziwej wysokości, ulice, park lub bulwar, gdzie spacerowaliście.",
        ] },
        { h2: "Grawer z datą, który czyni go waszym", p: [
          "Data ślubu, data poznania albo po prostu rok — najprostszy i najtrafniejszy grawer. Trafia na płaską mapę, magnes lub osobną płytkę na modelu 3D.",
          "Jako wspólny symbol jest brelok-serce z dwóch połówek: jedna to wasza dzielnica, druga miejsce, gdzie się poznaliście. Razem tworzą całe serce, osobno każde nosi swoją połówkę.",
        ] },
        { h2: "Jak zrobić to w 5 minut", p: [
          "Wybierz produkt — brelok, magnes lub mapę 3D. Zaznacz miejsce na mapie: adres, nazwę ulicy lub kawiarni, gdzie wszystko się zaczęło, i przeciągnij ramkę na właściwy kwartał.",
          "Darmowy podgląd 3D pojawi się po 1–2 minutach. Jeśli wszystko pasuje, dodaj grawer i zamów druk (wysyłka na terenie Ukrainy) — albo kup plik 3MF (≈3 €) i wydrukuj go na miejscu.",
        ] },
        { h2: "Kiedy będzie gotowe", p: [
          "Druk i wysyłka trwają 2–4 dni robocze od złożenia zamówienia. Płatność online, bezpośrednio na stronie.",
          "Jeśli rocznica jest jutro, wybierz brelok — drukuje się najszybciej — albo plik dla lokalnego serwisu druku 3D.",
        ] },
      ],
      ctaLabel: "Stwórz mapę waszego miejsca",
      ctaHref: "/create?product=map3d",
      outro: "Nie masz dokładnego adresu? Wystarczy nazwa ulicy lub lokalu — wyszukiwarka podpowie miejsce, a ramkę przeciągniesz ręcznie na właściwy kwartał.",
    },
    fr: {
      title: "Cadeau d'anniversaire de mariage pour son mari : la carte 3D de votre ville",
      description: "Idée cadeau d'anniversaire de mariage pour lui : la carte 3D du lieu de votre rencontre ou de votre mariage, ou un porte-clés avec la date gravée.",
      h1: "Un cadeau d'anniversaire de mariage pour lui qu'il n'oubliera pas en une semaine",
      intro: "Une cravate, il peut se l'acheter, et un « bon pour un massage » finira dans un tiroir. Ce qui marche, c'est un objet qui contient votre lieu : le pâté de maisons de votre rencontre, la rue de la mairie ou l'immeuble où vous avez emménagé ensemble. La carte 3D de ce lieu avec les vrais bâtiments, et une date gravée au dos d'un porte-clés — un cadeau qui reste sur le bureau, pas dans le placard, et qui n'a pas besoin d'explication.",
      sections: [
        { h2: "Pourquoi une carte plutôt qu'un objet de plus sur la liste", p: [
          "Une carte ne s'achète pas toute faite — elle est créée pour des coordonnées précises et n'existe qu'en un exemplaire. Sur un bureau ou une étagère, elle devient tout de suite l'occasion de raconter : « c'est ici qu'on s'est rencontrés ».",
          "Ça marche même s'il « a déjà tout » : un lieu qui compte pour vous deux ne dépend ni de la taille ni des goûts en gadgets. Impossible aussi de le confondre avec le cadeau de quelqu'un d'autre — contrairement à un parfum ou à un énième portefeuille.",
        ] },
        { h2: "Quel lieu choisir", p: [
          "Pas besoin de tout le centre-ville — le plus fort, c'est un point précis : la cour de votre premier logement commun, la rue du café du premier rendez-vous ou le quartier de la mairie où a eu lieu la cérémonie.",
          "Si vous avez déménagé plusieurs fois, choisis le lieu le plus chargé d'émotion, pas le plus récent. Le cadre peut être réduit à un seul pâté de maisons pour que les détails qu'il reconnaîtra restent bien visibles, même sur un petit modèle.",
        ] },
        { h2: "Trois options pour tous les budgets", p: [
          "Un porte-clés avec la carte de votre quartier (dès ≈4 €) — porté tous les jours, avec la date du mariage ou des coordonnées gravées au dos.",
          "Un aimant de frigo (≈5 €) — une carte compacte de 6 cm, un geste symbolique et abordable.",
          "Une carte 3D (dès ≈8 €, 5,5–15 cm) — le cadeau principal : bâtiments à leur vraie hauteur, rues, le parc ou le quai où vous vous promeniez.",
        ] },
        { h2: "Une date gravée qui le rend unique", p: [
          "La date du mariage, de la rencontre ou simplement l'année — la gravure la plus simple et la plus juste. Elle va sur une carte plate, un aimant ou une plaque à part sur le modèle 3D.",
          "Comme symbole à deux, il existe un porte-clés cœur en deux moitiés : l'une est votre quartier, l'autre le lieu de la rencontre. Ensemble, elles forment un cœur entier ; séparées, chacun porte la sienne.",
        ] },
        { h2: "Le faire en 5 minutes", p: [
          "Choisis un produit — porte-clés, aimant ou carte 3D. Marque le lieu sur la carte : une adresse, un nom de rue ou le café où tout a commencé, et fais glisser le cadre sur le bon pâté de maisons.",
          "Un aperçu 3D gratuit apparaît en 1 à 2 minutes. Si tout va bien, ajoute la gravure et commande l'impression (livraison en Ukraine) — ou achète le fichier 3MF (≈3 €) et fais-le imprimer près de chez toi.",
        ] },
        { h2: "Quand ce sera prêt", p: [
          "Impression et expédition : 2–4 jours ouvrés après la commande. Paiement en ligne, directement sur le site.",
          "Si l'anniversaire est demain, choisis un porte-clés — c'est le plus rapide à imprimer — ou le fichier pour un service d'impression 3D local.",
        ] },
      ],
      ctaLabel: "Créer la carte de votre lieu",
      ctaHref: "/create?product=map3d",
      outro: "Pas d'adresse exacte ? Un nom de rue ou d'établissement suffit — la recherche propose le lieu, et le cadre se déplace à la main sur le bon pâté de maisons.",
    },
    es: {
      title: "Regalo de aniversario de boda para tu marido: un mapa 3D de vuestra ciudad",
      description: "Idea de regalo de aniversario de boda para él: un mapa 3D del lugar donde os conocisteis o casasteis, o un llavero con la fecha grabada.",
      h1: "Un regalo de aniversario de boda para él que no olvidará en una semana",
      intro: "Una corbata se la compra él, y un «vale para un masaje» acabará en un cajón. Lo que funciona es algo que guarda vuestro lugar: la manzana donde os conocisteis, la calle del registro civil o el edificio al que os mudasteis juntos. Un mapa 3D de ese lugar con los edificios reales y una fecha grabada en un llavero: un regalo que se queda en el escritorio, no en el armario, y que no necesita explicación.",
      sections: [
        { h2: "Por qué un mapa y no otra cosa de la lista de deseos", p: [
          "Un mapa no se compra hecho: se crea para unas coordenadas concretas, así que existe en un solo ejemplar. En el escritorio de la oficina o en una estantería se convierte enseguida en la excusa para contar la historia: «aquí nos conocimos».",
          "Funciona incluso si él «ya lo tiene todo»: un lugar importante para los dos no depende de tallas ni de gustos en gadgets. Y es difícil confundirlo con el regalo de otra persona, a diferencia de un perfume o de otra cartera.",
        ] },
        { h2: "Qué lugar elegir", p: [
          "No hace falta todo el centro: lo más potente es un punto concreto, como el patio de vuestra primera casa, la calle de la cafetería de la primera cita o la manzana del registro civil donde fue la ceremonia.",
          "Si os habéis mudado varias veces, elige el lugar con más peso emocional, no el más reciente. El marco se puede reducir a una sola manzana para que los detalles que él reconocerá se vean bien incluso en un modelo pequeño.",
        ] },
        { h2: "Tres opciones para cualquier presupuesto", p: [
          "Un llavero con el mapa de vuestra manzana (desde ≈4 €): lo lleva a diario, con la fecha de la boda o las coordenadas grabadas detrás.",
          "Un imán de nevera (≈5 €): un mapa compacto de 6 cm, un gesto simbólico y barato.",
          "Un mapa 3D (desde ≈8 €, 5,5–15 cm): el regalo principal, con edificios a su altura real, calles y el parque o paseo por donde caminabais.",
        ] },
        { h2: "Una fecha grabada que lo hace vuestro", p: [
          "La fecha de la boda, la de cuando os conocisteis o simplemente el año: el grabado más sencillo y preciso. Va en un mapa plano, un imán o una placa aparte en el modelo 3D.",
          "Como símbolo compartido hay un llavero corazón de dos mitades: una es vuestro barrio y la otra el lugar donde os conocisteis. Juntas forman un corazón entero; por separado, cada uno lleva su mitad.",
        ] },
        { h2: "Cómo hacerlo en 5 minutos", p: [
          "Elige un producto: llavero, imán o mapa 3D. Marca el lugar en el mapa con una dirección, un nombre de calle o la cafetería donde todo empezó, y arrastra el marco a la manzana correcta.",
          "En 1–2 minutos aparece una vista previa 3D gratis. Si todo encaja, añade el grabado y pide la impresión (envío dentro de Ucrania), o compra el archivo 3MF (≈3 €) e imprímelo cerca de ti.",
        ] },
        { h2: "Cuándo estará listo", p: [
          "La impresión y el envío tardan 2–4 días hábiles desde el pedido. El pago es online, en la propia web.",
          "Si el aniversario es mañana, elige un llavero, que es lo más rápido de imprimir, o el archivo para un servicio de impresión 3D cercano.",
        ] },
      ],
      ctaLabel: "Crear el mapa de vuestro lugar",
      ctaHref: "/create?product=map3d",
      outro: "¿No tienes la dirección exacta? Basta con el nombre de la calle o del local: el buscador sugiere el lugar y el marco se arrastra a mano a la manzana correcta.",
    },
  },
};
