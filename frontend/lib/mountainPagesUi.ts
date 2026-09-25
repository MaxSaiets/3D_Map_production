// Інтерфейс сторінок гір (/gory, /gory/[slug]) шістьма мовами. Контент вершин — lib/mountainPages.ts.
import type { PeakLocale } from "@/lib/mountainPages";

export type PeakUi = {
  crumb: string; // «3D-моделі гір»
  indexTitle: string;
  indexDesc: string;
  indexH1: string;
  indexIntro: string;
  cta: (name: string) => string;
  ctaAny: string;
  facts: { elev: string; where: string; area: string; size: string; file: string; print: string };
  sizeVal: string;
  fileVal: string;
  printVal: string;
  howTitle: string;
  how: string[];
  optsTitle: string;
  opts: string[];
  dataNote: string;
  photoCaption: string;
  others: string;
  m: string; // «м» / «m»
  km: string;
};

export const PEAK_UI: Record<PeakLocale, PeakUi> = {
  uk: {
    crumb: "3D-моделі гір",
    indexTitle: "3D-моделі гір — Говерла, Матергорн, Еверест з реального рельєфу",
    indexDesc: "Друковані 3D-моделі відомих гір зі справжнього рельєфу: Говерла, Петрос, Матергорн, Монблан, Еверест, Фудзі та інші. Або будь-яка точка на карті. Превʼю онлайн.",
    indexH1: "3D-моделі гір з реального рельєфу",
    indexIntro: "Оберіть вершину — конструктор побудує модель зі справжніх висот: лідар 2 м для Швейцарії, супутникова модель рельєфу для решти світу. Можна додати ободок, скельні боки, фігурки скелелаза чи хатини й напис. Немає вашої гори в списку — оберіть будь-яку точку на карті.",
    cta: (n) => `Створити 3D-модель: ${n}`,
    ctaAny: "Обрати будь-яку гору на карті",
    facts: { elev: "Висота", where: "Де", area: "Ділянка моделі", size: "Розмір моделі", file: "Файл для друку", print: "Друк" },
    sizeVal: "12–40 см, більше — плитками",
    fileVal: "3MF, одразу після генерації",
    printVal: "ціна за запитом (залежить від розміру)",
    howTitle: "Як зробити модель",
    how: ["Натисніть кнопку — конструктор гір відкриється з цією вершиною.", "Оберіть розмір, висоту, ободок і боки; миттєве превʼю покаже рельєф зверху.", "Згенеруйте 3D-модель, покрутіть її в браузері й замовте друк або завантажте файл."],
    optsTitle: "Що можна налаштувати",
    opts: ["розмір плити від 12 до 40 см (великі — плитками без щілин)", "підсилення вертикалі, щоб схили читались рельєфніше", "ободок: заокруглений, прямий або без нього", "боки: похилі чи «скельні» з ребрами", "фігурки: людина на вершині, скелелаз на найкрутішій стіні, хатина", "кольоровий супутниковий знімок поверхні"],
    dataNote: "Моделі друкуються без підтримок, герметичні. Фото — ілюстрація вершини, модель будується з даних рельєфу.",
    photoCaption: "Фото вершини",
    others: "Інші гори",
    m: "м",
    km: "км",
  },
  en: {
    crumb: "3D mountain models",
    indexTitle: "3D mountain models — Hoverla, Matterhorn, Everest from real terrain",
    indexDesc: "Printed 3D models of famous mountains from real terrain data: Hoverla, Matterhorn, Mont Blanc, Everest, Fuji and more — or any point on the map. Online preview.",
    indexH1: "3D mountain models from real terrain",
    indexIntro: "Pick a summit and the builder creates a model from real elevations: 2 m lidar for Switzerland, a satellite terrain model for the rest of the world. Add a frame, rocky sides, climber or hut figures and text. Your mountain is not listed? Pick any point on the map.",
    cta: (n) => `Create a 3D model: ${n}`,
    ctaAny: "Pick any mountain on the map",
    facts: { elev: "Elevation", where: "Where", area: "Model area", size: "Model size", file: "Print file", print: "Printing" },
    sizeVal: "12–40 cm, larger as tiles",
    fileVal: "3MF, right after generation",
    printVal: "price on request (depends on size)",
    howTitle: "How to make the model",
    how: ["Press the button — the mountain builder opens with this summit.", "Choose size, height, frame and sides; the instant preview shows the terrain from above.", "Generate the 3D model, spin it in the browser and order a print or download the file."],
    optsTitle: "What you can adjust",
    opts: ["plate size from 12 to 40 cm (large ones as gap-free tiles)", "vertical exaggeration so slopes read more clearly", "frame: rounded, straight or none", "sides: sloped or rocky with ribs", "figures: a person on the summit, a climber on the steepest face, a hut", "colour satellite imagery on the surface"],
    dataNote: "Models print without supports and are watertight. The photo illustrates the summit; the model is built from terrain data. Printed models are delivered within Ukraine only.",
    photoCaption: "Photo of the summit",
    others: "Other mountains",
    m: "m",
    km: "km",
  },
  de: {
    crumb: "3D-Bergmodelle",
    indexTitle: "3D-Bergmodelle — Matterhorn, Eiger, Mont Blanc aus echtem Gelände",
    indexDesc: "Gedruckte 3D-Modelle berühmter Berge aus echten Höhendaten: Matterhorn, Eiger, Mont Blanc, Everest, Fuji u. v. m. — oder jeder Punkt auf der Karte. Online-Vorschau.",
    indexH1: "3D-Bergmodelle aus echtem Relief",
    indexIntro: "Gipfel wählen — der Konfigurator baut das Modell aus echten Höhen: 2-m-Lidar für die Schweiz, ein Satelliten-Geländemodell für die übrige Welt. Mit Rahmen, Felswänden, Kletter- oder Hüttenfiguren und Gravur. Dein Berg fehlt? Wähle einen beliebigen Punkt auf der Karte.",
    cta: (n) => `3D-Modell erstellen: ${n}`,
    ctaAny: "Beliebigen Berg auf der Karte wählen",
    facts: { elev: "Höhe", where: "Lage", area: "Modellausschnitt", size: "Modellgröße", file: "Druckdatei", print: "Druck" },
    sizeVal: "12–40 cm, größer als Kacheln",
    fileVal: "3MF, direkt nach der Erstellung",
    printVal: "Preis auf Anfrage (je nach Größe)",
    howTitle: "So entsteht das Modell",
    how: ["Button drücken — der Berg-Konfigurator öffnet sich mit diesem Gipfel.", "Größe, Höhe, Rahmen und Seiten wählen; die Sofortvorschau zeigt das Relief von oben.", "3D-Modell erzeugen, im Browser drehen und Druck bestellen oder Datei laden."],
    optsTitle: "Einstellbar",
    opts: ["Plattengröße 12 bis 40 cm (große als fugenlose Kacheln)", "Überhöhung, damit Hänge plastischer wirken", "Rahmen: abgerundet, gerade oder ohne", "Seiten: schräg oder felsig mit Rippen", "Figuren: Mensch auf dem Gipfel, Kletterer an der steilsten Wand, Hütte", "farbiges Satellitenbild auf der Oberfläche"],
    dataNote: "Die Modelle drucken ohne Stützen und sind wasserdicht. Das Foto illustriert den Gipfel; das Modell entsteht aus Geländedaten. Gedruckte Modelle versenden wir nur innerhalb der Ukraine.",
    photoCaption: "Foto des Gipfels",
    others: "Weitere Berge",
    m: "m",
    km: "km",
  },
  pl: {
    crumb: "Modele 3D gór",
    indexTitle: "Modele 3D gór — Rysy, Howerla, Matterhorn z prawdziwego terenu",
    indexDesc: "Drukowane modele 3D słynnych gór z prawdziwych danych wysokościowych: Rysy, Howerla, Matterhorn, Mont Blanc, Everest, Fudżi — albo dowolny punkt na mapie. Podgląd online.",
    indexH1: "Modele 3D gór z prawdziwego terenu",
    indexIntro: "Wybierz szczyt — kreator zbuduje model z prawdziwych wysokości: lidar 2 m dla Szwajcarii, satelitarny model terenu dla reszty świata. Dodaj ramkę, skalne boki, figurki wspinacza lub schroniska i napis. Nie ma twojej góry? Wybierz dowolny punkt na mapie.",
    cta: (n) => `Stwórz model 3D: ${n}`,
    ctaAny: "Wybierz dowolną górę na mapie",
    facts: { elev: "Wysokość", where: "Gdzie", area: "Obszar modelu", size: "Rozmiar modelu", file: "Plik do druku", print: "Druk" },
    sizeVal: "12–40 cm, większe z płytek",
    fileVal: "3MF, zaraz po wygenerowaniu",
    printVal: "cena na zapytanie (zależy od rozmiaru)",
    howTitle: "Jak powstaje model",
    how: ["Kliknij przycisk — kreator gór otworzy się z tym szczytem.", "Wybierz rozmiar, wysokość, ramkę i boki; natychmiastowy podgląd pokaże relief z góry.", "Wygeneruj model 3D, obróć go w przeglądarce i zamów druk albo pobierz plik."],
    optsTitle: "Co można ustawić",
    opts: ["rozmiar płyty od 12 do 40 cm (duże z płytek bez szczelin)", "przewyższenie, by stoki były bardziej wyraziste", "ramka: zaokrąglona, prosta lub bez", "boki: pochyłe albo skalne z żebrami", "figurki: człowiek na szczycie, wspinacz na najbardziej stromej ścianie, schronisko", "kolorowe zdjęcie satelitarne na powierzchni"],
    dataNote: "Modele drukują się bez podpór i są szczelne. Zdjęcie ilustruje szczyt; model powstaje z danych terenu. Wydrukowane modele wysyłamy tylko na terenie Ukrainy.",
    photoCaption: "Zdjęcie szczytu",
    others: "Inne góry",
    m: "m",
    km: "km",
  },
  fr: {
    crumb: "Maquettes 3D de montagnes",
    indexTitle: "Maquettes 3D de montagnes — Mont Blanc, Cervin, Everest en relief réel",
    indexDesc: "Maquettes 3D imprimées de montagnes célèbres à partir du relief réel : Mont Blanc, Cervin, Eiger, Everest, Fuji… ou n'importe quel point de la carte. Aperçu en ligne.",
    indexH1: "Maquettes 3D de montagnes en relief réel",
    indexIntro: "Choisissez un sommet : le configurateur construit la maquette à partir d'altitudes réelles — lidar 2 m pour la Suisse, modèle satellite du relief pour le reste du monde. Ajoutez un cadre, des flancs rocheux, des figurines de grimpeur ou de refuge et un texte. Votre montagne manque ? Choisissez n'importe quel point de la carte.",
    cta: (n) => `Créer une maquette 3D : ${n}`,
    ctaAny: "Choisir n'importe quelle montagne sur la carte",
    facts: { elev: "Altitude", where: "Où", area: "Zone de la maquette", size: "Taille", file: "Fichier d'impression", print: "Impression" },
    sizeVal: "12–40 cm, plus grand en tuiles",
    fileVal: "3MF, juste après la génération",
    printVal: "prix sur demande (selon la taille)",
    howTitle: "Comment créer la maquette",
    how: ["Cliquez sur le bouton : le configurateur de montagnes s'ouvre avec ce sommet.", "Choisissez taille, hauteur, cadre et flancs ; l'aperçu instantané montre le relief vu du dessus.", "Générez la maquette 3D, faites-la tourner dans le navigateur et commandez l'impression ou téléchargez le fichier."],
    optsTitle: "Réglages possibles",
    opts: ["taille de 12 à 40 cm (les grandes en tuiles sans jointure)", "exagération verticale pour des pentes plus lisibles", "cadre : arrondi, droit ou sans", "flancs : inclinés ou rocheux à nervures", "figurines : personne au sommet, grimpeur sur la face la plus raide, refuge", "image satellite en couleur sur la surface"],
    dataNote: "Les maquettes s'impriment sans supports et sont étanches. La photo illustre le sommet ; la maquette est construite à partir des données de relief. Les maquettes imprimées sont livrées uniquement en Ukraine.",
    photoCaption: "Photo du sommet",
    others: "Autres montagnes",
    m: "m",
    km: "km",
  },
  es: {
    crumb: "Maquetas 3D de montañas",
    indexTitle: "Maquetas 3D de montañas — Aconcagua, Everest, Mont Blanc con relieve real",
    indexDesc: "Maquetas 3D impresas de montañas famosas con datos de relieve reales: Aconcagua, Everest, Mont Blanc, Cervino, Fuji… o cualquier punto del mapa. Vista previa online.",
    indexH1: "Maquetas 3D de montañas con relieve real",
    indexIntro: "Elige una cumbre y el configurador crea la maqueta con alturas reales: lidar de 2 m para Suiza y un modelo satelital del terreno para el resto del mundo. Añade marco, laterales de roca, figuras de escalador o refugio y un texto. ¿No está tu montaña? Elige cualquier punto del mapa.",
    cta: (n) => `Crear maqueta 3D: ${n}`,
    ctaAny: "Elegir cualquier montaña en el mapa",
    facts: { elev: "Altitud", where: "Dónde", area: "Área de la maqueta", size: "Tamaño", file: "Archivo de impresión", print: "Impresión" },
    sizeVal: "12–40 cm, más grande en baldosas",
    fileVal: "3MF, justo después de generar",
    printVal: "precio a consultar (según tamaño)",
    howTitle: "Cómo se crea la maqueta",
    how: ["Pulsa el botón: el configurador de montañas se abre con esta cumbre.", "Elige tamaño, altura, marco y laterales; la vista previa instantánea muestra el relieve desde arriba.", "Genera la maqueta 3D, gírala en el navegador y pide la impresión o descarga el archivo."],
    optsTitle: "Qué puedes ajustar",
    opts: ["tamaño de 12 a 40 cm (las grandes en baldosas sin juntas)", "exageración vertical para laderas más legibles", "marco: redondeado, recto o sin marco", "laterales: inclinados o de roca con nervios", "figuras: persona en la cima, escalador en la cara más empinada, refugio", "imagen satelital en color sobre la superficie"],
    dataNote: "Las maquetas se imprimen sin soportes y son estancas. La foto ilustra la cumbre; la maqueta se construye con datos de relieve. Las maquetas impresas se envían solo dentro de Ucrania.",
    photoCaption: "Foto de la cumbre",
    others: "Otras montañas",
    m: "m",
    km: "km",
  },
};
