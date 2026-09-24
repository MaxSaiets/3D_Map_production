// Переклади статей блогу de/pl/fr/es — частина 5 (правила — див. lib/blogI18nExtra.ts).
// 24.09.2026: 14 лютого — свято міжнародне, тож стаття-мапа-серце індексується всіма мовами.
// Ціни ≈€ за EUR_PER_UAH=0.024 (брелок 170 ₴ ≈4 €, M 490 ₴ ≈12 €, L 630 ₴ ≈15 €, магніт 210 ₴ ≈5 €).
import type { BlogArticleContent } from "@/lib/blog";
import type { BlogExtraLocale } from "@/lib/blogI18nExtra";

export const BLOG_I18N_EXTRA_5: Record<string, Partial<Record<BlogExtraLocale, BlogArticleContent>>> = {
  "podarunok-na-14-lyutoho-mapa-mistsya-znayomstva": {
    de: {
      title: "Valentinstagsgeschenk: eine 3D-Karte in Herzform vom Ort eures Kennenlernens",
      description: "Romantisches Geschenk zum Valentinstag: eine herzförmige 3D-Karte vom Ort eures ersten Dates mit eingraviertem Datum. Druckdatei ≈3 €, Anhänger ab ≈4 €.",
      h1: "Ein Valentinsgeschenk: die Karte des Ortes, an dem alles begann — in Herzform",
      intro: "Zum Valentinstag möchte man etwas schenken, das nur euch beide betrifft. Der Ort des ersten Dates, die Straße eurer ersten gemeinsamen Wohnung, der Park des Antrags — all das wird zu einer 3D-Karte in Herzform. Straßen und Gebäude sind echt, und auf der Rückseite steht euer Datum oder ein paar Worte.",
      sections: [
        { h2: "Eine Karte in Herzform", p: [
          "Im Konfigurator wählst du die Form des Modells: Quadrat, Kreis, Sechseck oder Herz. Der Rahmen auf der Karte zeigt die Kontur sofort — du siehst, welche Straßen im Herz landen.",
          "Größen von 5,5 cm (kleines Andenken) bis 20 cm (Tischmodell). M mit 8 cm kostet ≈12 €, L mit 11 cm ≈15 €.",
        ] },
        { h2: "Weitere Ideen für Paare", p: [
          "Zwei Herz-Anhänger, die sich zu einem zusammenfügen — jeder trägt seine Hälfte (ab ≈4 €).",
          "Die Karte der Stadt, in der ihr euch kennengelernt habt, neben der Karte, in der ihr jetzt lebt.",
          "Ein Magnet mit der Karte eurer ersten gemeinsamen Wohnung (≈5 €).",
        ] },
        { h2: "Was man eingravieren kann", p: [
          "Das Datum des ersten Treffens, Koordinaten, „Hier begann alles“ oder eure Namen. Der Text wird im Konfigurator hinzugefügt.",
          "Unsicher, welcher Ort? Schau dir kostenlose Vorschauen mehrerer Varianten an — jede dauert 1–2 Minuten.",
        ] },
        { h2: "Druck und Lieferung", p: [
          "Gedruckte Modelle versenden wir nur innerhalb der Ukraine, 2–4 Tage nach der Bestellung. Außerhalb der Ukraine kaufst du die 3MF-Datei (≈3 €) und lässt sie auf einem eigenen Drucker oder in einem lokalen 3D-Druckservice drucken.",
        ] },
      ],
      ctaLabel: "Herz-Karte erstellen",
      ctaHref: "/create",
      outro: "Die Vorschau ist kostenlos — du kannst sie zeigen, noch bevor das Modell gedruckt ist.",
    },
    pl: {
      title: "Prezent na Walentynki: mapa 3D w kształcie serca z miejscem poznania",
      description: "Romantyczny prezent na Walentynki: mapa 3D w kształcie serca z miejscem pierwszej randki i wygrawerowaną datą. Plik do druku ≈3 €, breloki od ≈4 €.",
      h1: "Prezent na Walentynki: mapa miejsca, w którym wszystko się zaczęło — w kształcie serca",
      intro: "Na Walentynki chce się podarować coś, co dotyczy tylko was dwojga. Miejsce pierwszej randki, ulica pierwszego wspólnego mieszkania, park, w którym padły oświadczyny — wszystko to może stać się mapą 3D w kształcie serca. Ulice i budynki są prawdziwe, a na odwrocie — wasza data lub kilka słów.",
      sections: [
        { h2: "Mapa w kształcie serca", p: [
          "W kreatorze wybierasz kształt modelu: kwadrat, koło, sześciokąt lub serce. Ramka na mapie od razu pokazuje kontur — widać, które ulice trafią do serca.",
          "Rozmiary od 5,5 cm (mała pamiątka) do 20 cm (model na biurko). M 8 cm kosztuje ≈12 €, L 11 cm — ≈15 €.",
        ] },
        { h2: "Inne pomysły dla pary", p: [
          "Dwa breloki-serca, które łączą się w jedno — każde nosi swoją połówkę (od ≈4 €).",
          "Mapa miasta, w którym się poznaliście, obok mapy miasta, w którym mieszkacie teraz.",
          "Magnes z mapą pierwszego wspólnego mieszkania (≈5 €).",
        ] },
        { h2: "Co wygrawerować", p: [
          "Datę pierwszego spotkania, współrzędne miejsca, „Tu wszystko się zaczęło” albo wasze imiona. Napis dodajesz w kreatorze.",
          "Nie wiesz, które miejsce wybrać? Obejrzyj darmowe podglądy kilku wariantów — każdy trwa 1–2 minuty.",
        ] },
        { h2: "Druk i dostawa", p: [
          "Wydrukowane modele wysyłamy tylko na terenie Ukrainy, 2–4 dni od zamówienia. Poza Ukrainą kup plik 3MF (≈3 €) i wydrukuj go na własnej drukarce lub w lokalnym serwisie druku 3D.",
        ] },
      ],
      ctaLabel: "Stwórz mapę-serce",
      ctaHref: "/create",
      outro: "Podgląd jest darmowy — możesz go pokazać, zanim model zostanie wydrukowany.",
    },
    fr: {
      title: "Cadeau de Saint-Valentin : une carte 3D en forme de cœur du lieu de votre rencontre",
      description: "Un cadeau romantique pour la Saint-Valentin : une carte 3D en forme de cœur du lieu de votre premier rendez-vous, avec la date gravée. Fichier ≈3 €, porte-clés dès ≈4 €.",
      h1: "Un cadeau de Saint-Valentin : la carte du lieu où tout a commencé, en forme de cœur",
      intro: "Pour la Saint-Valentin, on a envie d'offrir quelque chose qui ne concerne que vous deux. Le lieu du premier rendez-vous, la rue de votre premier appartement, le parc de la demande en mariage — tout cela peut devenir une carte 3D en forme de cœur. Les rues et les bâtiments sont réels, et au dos figurent votre date ou quelques mots.",
      sections: [
        { h2: "Une carte en forme de cœur", p: [
          "Dans le configurateur, tu choisis la forme du modèle : carré, cercle, hexagone ou cœur. Le cadre sur la carte montre le contour tout de suite — tu vois quelles rues entrent dans le cœur.",
          "Tailles de 5,5 cm (petit souvenir) à 20 cm (modèle de bureau). M de 8 cm coûte ≈12 €, L de 11 cm ≈15 €.",
        ] },
        { h2: "D'autres idées pour un couple", p: [
          "Deux porte-clés cœur qui s'emboîtent — chacun garde sa moitié (dès ≈4 €).",
          "La carte de la ville de votre rencontre à côté de celle où vous vivez aujourd'hui.",
          "Un magnet avec la carte de votre premier appartement (≈5 €).",
        ] },
        { h2: "Que graver", p: [
          "La date de la rencontre, les coordonnées du lieu, « Ici tout a commencé » ou vos prénoms. Le texte s'ajoute dans le configurateur.",
          "Tu hésites sur le lieu ? Regarde les aperçus gratuits de plusieurs variantes — 1 à 2 minutes chacun.",
        ] },
        { h2: "Impression et livraison", p: [
          "Nous livrons les modèles imprimés uniquement en Ukraine, 2 à 4 jours après la commande. Hors d'Ukraine, achète le fichier 3MF (≈3 €) et fais-le imprimer sur ta propre imprimante ou dans un service d'impression 3D local.",
        ] },
      ],
      ctaLabel: "Créer une carte en cœur",
      ctaHref: "/create",
      outro: "L'aperçu est gratuit — tu peux le montrer avant même que le modèle soit imprimé.",
    },
    es: {
      title: "Regalo de San Valentín: un mapa 3D en forma de corazón del lugar donde os conocisteis",
      description: "Un regalo romántico para San Valentín: un mapa 3D en forma de corazón del lugar de vuestra primera cita, con la fecha grabada. Archivo ≈3 €, llaveros desde ≈4 €.",
      h1: "Un regalo de San Valentín: el mapa del lugar donde todo empezó, en forma de corazón",
      intro: "En San Valentín apetece regalar algo que sea solo de los dos. El lugar de la primera cita, la calle del primer piso compartido, el parque de la pedida — todo eso puede convertirse en un mapa 3D en forma de corazón. Las calles y los edificios son reales, y en el reverso va vuestra fecha o unas palabras.",
      sections: [
        { h2: "Un mapa en forma de corazón", p: [
          "En el configurador eliges la forma del modelo: cuadrado, círculo, hexágono o corazón. El marco del mapa muestra el contorno al instante — ves qué calles quedan dentro del corazón.",
          "Tamaños de 5,5 cm (pequeño recuerdo) a 20 cm (modelo de escritorio). M de 8 cm cuesta ≈12 €, L de 11 cm ≈15 €.",
        ] },
        { h2: "Más ideas para parejas", p: [
          "Dos llaveros corazón que encajan en uno — cada uno lleva su mitad (desde ≈4 €).",
          "El mapa de la ciudad donde os conocisteis junto al de la ciudad donde vivís ahora.",
          "Un imán con el mapa de vuestro primer piso (≈5 €).",
        ] },
        { h2: "Qué grabar", p: [
          "La fecha del primer encuentro, las coordenadas, «Aquí empezó todo» o vuestros nombres. El texto se añade en el configurador.",
          "¿No sabes qué lugar elegir? Mira vistas previas gratuitas de varias opciones — cada una tarda 1–2 minutos.",
        ] },
        { h2: "Impresión y envío", p: [
          "Enviamos los modelos impresos solo dentro de Ucrania, 2–4 días después del pedido. Fuera de Ucrania, compra el archivo 3MF (≈3 €) e imprímelo en tu propia impresora o en un servicio de impresión 3D local.",
        ] },
      ],
      ctaLabel: "Crear un mapa corazón",
      ctaHref: "/create",
      outro: "La vista previa es gratuita — puedes enseñarla antes incluso de que el modelo esté impreso.",
    },
  },
};
