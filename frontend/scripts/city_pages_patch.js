/**
 * Локалізація programmatic city-pages (/maps, /maps/[city]) — 6 мов.
 * {city} підставляється next-intl. Запуск: node scripts/city_pages_patch.js
 */
const fs = require("fs");
const path = require("path");

const D = {
  uk: {
    mapsMeta: {
      title: "3D-мапи міст України — обери своє місто",
      description: "Готові 3D-мапи міст України для друку: Київ, Львів, Одеса, Харків, Дніпро та ще 18 міст. Обери місто — і створи тактильну 3D-модель району або брелок з картою.",
      keywords: "3д мапи міст україни, 3d карта українського міста, мапа міста для друку, брелок з картою українського міста",
    },
    cityPages: {
      title: "3D-мапа міста {city} — друк і брелок з картою",
      description: "Створи 3D-мапу міста {city}: обери район, налаштуй модель і завантаж 3MF/STL для друку — або замов друкований виріб чи брелок з картою міста {city}.",
      h1: "3D-мапа міста {city}",
      p1: "Перетвори {city} на тривимірну тактильну модель: вулиці, будинки з реальними висотами, парки й річки за даними OpenStreetMap. Обери будь-який район міста {city} у конструкторі — модель буде готова за кілька хвилин у форматі 3MF або STL для Bambu Studio чи PrusaSlicer.",
      p2: "Не маєш 3D-принтера? Надрукуємо мапу чи брелок з картою міста {city} з екологічного Eco PLA і доставимо Новою Поштою по Україні або в країни ЄС. Ідеальний подарунок для тих, хто любить своє місто.",
      ctaMap: "Створити 3D-мапу міста {city}",
      ctaKeychain: "Брелок з картою міста {city}",
      others: "Інші міста",
      breadcrumb: "Мапи міст",
    },
    footerMaps: "Мапи міст",
  },
  en: {
    mapsMeta: {
      title: "3D city maps of Ukraine — pick your city",
      description: "Ready 3D city maps of Ukraine for printing: Kyiv, Lviv, Odesa, Kharkiv, Dnipro and 18 more cities. Pick a city and create a tactile 3D district model or a map keychain.",
      keywords: "3d maps of ukrainian cities, 3d city map ukraine, city map for 3d printing, ukrainian city map keychain",
    },
    cityPages: {
      title: "3D map of {city} — print it or get a map keychain",
      description: "Create a 3D map of {city}: pick a district, tune the model and download a 3MF/STL for printing — or order a printed map or a {city} map keychain.",
      h1: "3D map of {city}",
      p1: "Turn {city} into a tactile three-dimensional model: streets, buildings with real heights, parks and rivers from OpenStreetMap data. Pick any district of {city} in the builder — the model is ready in minutes as a 3MF or STL for Bambu Studio or PrusaSlicer.",
      p2: "No 3D printer? We print the map or a {city} map keychain in eco-friendly Eco PLA and ship across Ukraine and to the EU. A perfect gift for anyone who loves their city.",
      ctaMap: "Create a 3D map of {city}",
      ctaKeychain: "{city} map keychain",
      others: "Other cities",
      breadcrumb: "City maps",
    },
    footerMaps: "City maps",
  },
  de: {
    mapsMeta: {
      title: "3D-Stadtkarten der Ukraine — wähle deine Stadt",
      description: "Fertige 3D-Stadtkarten ukrainischer Städte zum Drucken: Kiew, Lemberg, Odessa, Charkiw, Dnipro und 18 weitere. Wähle eine Stadt und erstelle ein taktiles 3D-Modell oder einen Karten-Schlüsselanhänger.",
      keywords: "3d stadtkarten ukraine, 3d karte ukrainische stadt, stadtkarte 3d druck, schlüsselanhänger ukrainische stadt",
    },
    cityPages: {
      title: "3D-Karte von {city} — drucken oder als Schlüsselanhänger",
      description: "Erstelle eine 3D-Karte von {city}: Viertel wählen, Modell anpassen und 3MF/STL herunterladen — oder eine gedruckte Karte bzw. einen {city}-Schlüsselanhänger bestellen.",
      h1: "3D-Karte von {city}",
      p1: "Verwandle {city} in ein taktiles dreidimensionales Modell: Straßen, Gebäude mit echten Höhen, Parks und Flüsse aus OpenStreetMap-Daten. Wähle ein beliebiges Viertel von {city} im Konfigurator — das Modell ist in Minuten fertig, als 3MF oder STL für Bambu Studio oder PrusaSlicer.",
      p2: "Kein 3D-Drucker? Wir drucken die Karte oder den {city}-Schlüsselanhänger in Eco PLA und liefern in die Ukraine und in die EU. Ein perfektes Geschenk für alle, die ihre Stadt lieben.",
      ctaMap: "3D-Karte von {city} erstellen",
      ctaKeychain: "{city}-Karten-Schlüsselanhänger",
      others: "Weitere Städte",
      breadcrumb: "Stadtkarten",
    },
    footerMaps: "Stadtkarten",
  },
  pl: {
    mapsMeta: {
      title: "Mapy 3D miast Ukrainy — wybierz swoje miasto",
      description: "Gotowe mapy 3D ukraińskich miast do druku: Kijów, Lwów, Odessa, Charków, Dniepr i 18 innych. Wybierz miasto i stwórz dotykowy model 3D dzielnicy albo brelok z mapą.",
      keywords: "mapy 3d miast ukrainy, mapa 3d miasta ukraina, mapa miasta do druku 3d, brelok z mapą ukraińskiego miasta",
    },
    cityPages: {
      title: "Mapa 3D miasta {city} — wydrukuj lub zamów brelok",
      description: "Stwórz mapę 3D miasta {city}: wybierz dzielnicę, dostosuj model i pobierz 3MF/STL do druku — albo zamów wydrukowaną mapę lub brelok z mapą miasta {city}.",
      h1: "Mapa 3D miasta {city}",
      p1: "Zamień {city} w dotykowy trójwymiarowy model: ulice, budynki z prawdziwymi wysokościami, parki i rzeki z danych OpenStreetMap. Wybierz dowolną dzielnicę miasta {city} w kreatorze — model będzie gotowy w kilka minut jako 3MF lub STL dla Bambu Studio czy PrusaSlicer.",
      p2: "Nie masz drukarki 3D? Wydrukujemy mapę lub brelok z mapą miasta {city} z ekologicznego Eco PLA i wyślemy po Ukrainie i do UE. Idealny prezent dla każdego, kto kocha swoje miasto.",
      ctaMap: "Stwórz mapę 3D miasta {city}",
      ctaKeychain: "Brelok z mapą miasta {city}",
      others: "Inne miasta",
      breadcrumb: "Mapy miast",
    },
    footerMaps: "Mapy miast",
  },
  fr: {
    mapsMeta: {
      title: "Cartes 3D des villes d'Ukraine — choisissez votre ville",
      description: "Cartes 3D prêtes des villes ukrainiennes à imprimer : Kyiv, Lviv, Odessa, Kharkiv, Dnipro et 18 autres. Choisissez une ville et créez un modèle 3D tactile ou un porte-clés carte.",
      keywords: "cartes 3d villes ukraine, carte 3d ville ukrainienne, carte de ville impression 3d, porte-clés ville ukrainienne",
    },
    cityPages: {
      title: "Carte 3D de {city} — à imprimer ou en porte-clés",
      description: "Créez une carte 3D de {city} : choisissez un quartier, ajustez le modèle et téléchargez le 3MF/STL — ou commandez une carte imprimée ou un porte-clés carte de {city}.",
      h1: "Carte 3D de {city}",
      p1: "Transformez {city} en modèle tridimensionnel tactile : rues, bâtiments avec leurs vraies hauteurs, parcs et rivières issus d'OpenStreetMap. Choisissez n'importe quel quartier de {city} dans le configurateur — le modèle est prêt en quelques minutes en 3MF ou STL pour Bambu Studio ou PrusaSlicer.",
      p2: "Pas d'imprimante 3D ? Nous imprimons la carte ou le porte-clés de {city} en Eco PLA écologique et livrons en Ukraine et dans l'UE. Un cadeau parfait pour qui aime sa ville.",
      ctaMap: "Créer la carte 3D de {city}",
      ctaKeychain: "Porte-clés carte de {city}",
      others: "Autres villes",
      breadcrumb: "Cartes des villes",
    },
    footerMaps: "Cartes des villes",
  },
  es: {
    mapsMeta: {
      title: "Mapas 3D de ciudades de Ucrania — elige tu ciudad",
      description: "Mapas 3D listos de ciudades ucranianas para imprimir: Kyiv, Lviv, Odesa, Kharkiv, Dnipro y 18 más. Elige una ciudad y crea un modelo 3D táctil o un llavero con mapa.",
      keywords: "mapas 3d ciudades ucrania, mapa 3d ciudad ucraniana, mapa de ciudad impresión 3d, llavero ciudad ucraniana",
    },
    cityPages: {
      title: "Mapa 3D de {city} — imprímelo o pide un llavero",
      description: "Crea un mapa 3D de {city}: elige un barrio, ajusta el modelo y descarga el 3MF/STL para imprimir — o pide un mapa impreso o un llavero con el mapa de {city}.",
      h1: "Mapa 3D de {city}",
      p1: "Convierte {city} en un modelo tridimensional táctil: calles, edificios con alturas reales, parques y ríos a partir de datos de OpenStreetMap. Elige cualquier barrio de {city} en el configurador — el modelo está listo en minutos como 3MF o STL para Bambu Studio o PrusaSlicer.",
      p2: "¿No tienes impresora 3D? Imprimimos el mapa o el llavero de {city} en Eco PLA ecológico y enviamos a Ucrania y a la UE. Un regalo perfecto para quien ama su ciudad.",
      ctaMap: "Crear el mapa 3D de {city}",
      ctaKeychain: "Llavero con el mapa de {city}",
      others: "Otras ciudades",
      breadcrumb: "Mapas de ciudades",
    },
    footerMaps: "Mapas de ciudades",
  },
};

for (const [locale, d] of Object.entries(D)) {
  const file = path.join(__dirname, "..", "messages", `${locale}.json`);
  const m = JSON.parse(fs.readFileSync(file, "utf8"));
  m.mapsMeta = d.mapsMeta;
  m.cityPages = d.cityPages;
  if (m.home && m.home.footer) m.home.footer.maps = d.footerMaps;
  fs.writeFileSync(file, JSON.stringify(m, null, 2) + "\n", "utf8");
  console.log(`${locale}: cityPages + mapsMeta + footer.maps`);
}
