/**
 * SEO-патч 2026-06-11: розширені кластери пошукових запитів усіма 6 мовами,
 * топо-брелок у описах, FAQ q7-q9 (топо/доставка ЄС/магніт), новий видимий
 * SEO-блок home.seo. Запуск: node scripts/seo_patch.js (з frontend/).
 * Скрипт ідемпотентний — просто перезаписує цільові ключі.
 */
const fs = require("fs");
const path = require("path");

const D = {
  uk: {
    metaKeywords:
      "3D мапа міста, 3д карта міста, 3D друк мапи, мапа міста на замовлення, брелок з картою міста, іменний брелок на замовлення, топографічна 3D карта, рельєф Карпат 3D, брелок Говерла, магніт з картою міста, подарунок з картою, тактильна мапа, 3MF, STL мапа міста, сувенір з картою, панно карта міста, мапа Києва 3D, мапа Львова 3D, Bambu Studio, PrusaSlicer",
    createKeywords:
      "створити 3д мапу міста, конструктор 3D мапи онлайн, генератор 3d карти, 3d карта району, завантажити 3MF мапу, STL мапа міста скачати, 3d друк мапи міста, мапа з висотами будинків, мапа-магніт на холодильник, мапа Києва для друку, мапа Львова 3D, мапа Одеси 3D",
    keychainsKeywords:
      "брелок з картою міста, брелок мапа, 3d брелок місто, брелок на замовлення з написом, іменний брелок, персональний брелок, брелок з гравіюванням, топо брелок рельєф гір, брелок Говерла Карпати, брелок серце з картою, брелок будиночок, подарунок хлопцю, подарунок дівчині, подарунок на річницю, брелок для пари",
    keychainsDescription:
      "Брелок-мапа твого міста: персональний 3D-брелок із вулицями, будинками та написом — або топо-брелок з рельєфом гір (Говерла, Карпати, Альпи). Створи онлайн і завантаж 3MF або замов друк з Eco PLA.",
    showcaseKeywords:
      "галерея 3д мап міст, приклади 3d друку мапи, надруковані мапи міст, 3д модель міста, приклади брелків з картою, топо брелок приклад, мапа магніт приклад",
    faq: {
      q7: "Чи можна зробити брелок з рельєфом гір замість карти вулиць?",
      a7: "Так, це топо-брелок. Увімкни режим «Рельєф висот» у конструкторі — і на жетоні буде надруковано справжній рельєф місцевості за супутниковими даними висот: Говерла й Карпати, Альпи, фіорди чи морське узбережжя. Вершини й долини читаються пальцями.",
      q8: "Чи доставляєте ви брелки та мапи в Європу?",
      a8: "Так. Друкуємо й надсилаємо по Україні (Нова Пошта, Укрпошта) та в 15 країн ЄС — через Nova Post EU і Meest: Польща, Німеччина, Чехія, Іспанія, Італія, Франція та інші.",
      q9: "Що таке мапа-магніт на холодильник?",
      a9: "Це плаский 3D-друкований магніт ~60 мм з картою твого району: вулиці, парки й річки рельєфом, ззаду — кишеня під неодимовий магніт. Бюджетний персональний сувенір або подарунок з пам'ятним місцем.",
    },
    seo: {
      title: "3D-мапи міст і персональні брелки — що ми робимо",
      p1: "Monadruk перетворює будь-яке місце на Землі на тривимірну модель для 3D-друку: 3D-мапу району з реальними висотами будинків, парками та річками за даними OpenStreetMap, брелок з картою міста з власним написом, топо-брелок з рельєфом гір за супутниковими даними висот або мапу-магніт на холодильник. Файли 3MF і STL відкриваються в Bambu Studio та PrusaSlicer і готові до друку вдома.",
      p2: "Не маєш 3D-принтера — надрукуємо за тебе з екологічного пластику Eco PLA і доставимо по Україні та в країни ЄС. Мапа Києва, Львова, Одеси чи Нью-Йорка, рельєф Говерли чи Альп, іменний брелок на річницю — обери місце на карті, і за кілька хвилин модель готова.",
    },
  },
  en: {
    metaKeywords:
      "3d printed city map, city map 3d print, custom city map, map keychain, personalized map keychain, 3d city map model, stl city map, 3mf map, topographic map 3d print, terrain relief keychain, city map fridge magnet, map wall art, personalized map gift, anniversary map gift, tactile map, Bambu Studio, PrusaSlicer",
    createKeywords:
      "create 3d city map, 3d map maker online, 3d map generator, download 3mf map, stl city map download, 3d print city map, building heights map, city map magnet, custom map of my city, 3d map of London, 3d map of Berlin",
    keychainsKeywords:
      "city map keychain, map keychain, custom 3d keychain, keychain with name, engraved keychain, topographic keychain, mountain relief keychain, heart map keychain, house keychain, gift for him, gift for her, anniversary gift, couple keychain",
    keychainsDescription:
      "A map keychain of your city: a personal 3D keychain with streets, buildings and custom text — or a topographic keychain with real mountain relief (Alps, Carpathians). Create online and download 3MF or order Eco PLA printing.",
    showcaseKeywords:
      "3d printed map gallery, city map print examples, printed city maps, 3d city model examples, map keychain examples, topographic keychain example, map magnet example",
    faq: {
      q7: "Can I get a keychain with mountain relief instead of streets?",
      a7: "Yes — that's the topo keychain. Switch on the “Terrain relief” mode in the builder and the token is printed with the real elevation relief of the place from satellite height data: the Alps, the Carpathians, fjords or a coastline. Peaks and valleys are readable with your fingers.",
      q8: "Do you ship keychains and maps across Europe?",
      a8: "Yes. We print and ship within Ukraine (Nova Poshta, Ukrposhta) and to 15 EU countries via Nova Post EU and Meest: Poland, Germany, Czechia, Spain, Italy, France and more.",
      q9: "What is a city-map fridge magnet?",
      a9: "A flat 3D-printed magnet of about 60 mm with the map of your neighbourhood: streets, parks and rivers in relief, with a pocket for a neodymium magnet on the back. A budget-friendly personal souvenir of a memorable place.",
    },
    seo: {
      title: "3D city maps and personalised keychains — what we make",
      p1: "Monadruk turns any place on Earth into a printable 3D model: a 3D map of a city district with real building heights, parks and rivers from OpenStreetMap data, a city map keychain with your own text, a topographic keychain with real mountain relief from satellite elevation data, or a city-map fridge magnet. The 3MF and STL files open straight in Bambu Studio and PrusaSlicer, ready to print at home.",
      p2: "No 3D printer? We print for you in eco-friendly Eco PLA and ship across Ukraine and the EU. A 3D map of London, Berlin, Kyiv or New York, the relief of the Alps or Hoverla, a personalised anniversary keychain — pick a spot on the map and the model is ready in minutes.",
    },
  },
  de: {
    metaKeywords:
      "3D Stadtkarte, Stadtplan 3D Druck, Stadtkarte drucken, Schlüsselanhänger Stadtkarte, personalisierter Schlüsselanhänger Karte, STL Stadtplan, 3MF Karte, topografische Karte 3D, Reliefkarte Druck, Kühlschrankmagnet Stadtkarte, personalisiertes Geschenk Karte, Jahrestagsgeschenk Karte, taktile Karte, Bambu Studio, PrusaSlicer",
    createKeywords:
      "3D Stadtkarte erstellen, Stadtkarten Generator online, 3MF Karte herunterladen, STL Stadtplan Download, Stadtkarte 3D drucken, Karte mit Gebäudehöhen, Stadtkarte Magnet, eigene Stadtkarte erstellen, 3D Karte Berlin, 3D Karte München",
    keychainsKeywords:
      "Schlüsselanhänger Stadtkarte, Karten Schlüsselanhänger, 3D Schlüsselanhänger personalisiert, Schlüsselanhänger mit Namen, Schlüsselanhänger Gravur, topografischer Schlüsselanhänger, Gebirgsrelief Anhänger, Herz Schlüsselanhänger Karte, Geschenk für ihn, Geschenk für sie, Jahrestagsgeschenk, Pärchen Geschenk",
    keychainsDescription:
      "Ein Karten-Schlüsselanhänger deiner Stadt: persönlicher 3D-Anhänger mit Straßen, Gebäuden und eigenem Text — oder ein Topo-Anhänger mit echtem Gebirgsrelief (Alpen, Karpaten). Online erstellen, 3MF herunterladen oder Druck in Eco PLA bestellen.",
    showcaseKeywords:
      "3D Stadtkarten Galerie, Beispiele Stadtkarte 3D Druck, gedruckte Stadtkarten, 3D Stadtmodell Beispiele, Schlüsselanhänger Karte Beispiele, Topo Anhänger Beispiel, Karten Magnet Beispiel",
    faq: {
      q7: "Gibt es einen Anhänger mit Gebirgsrelief statt Straßenkarte?",
      a7: "Ja — der Topo-Anhänger. Aktiviere im Konfigurator den Modus „Höhenrelief“, und das echte Geländerelief des Ortes wird aus Satelliten-Höhendaten gedruckt: Alpen, Karpaten, Fjorde oder Küstenlinien. Gipfel und Täler sind mit den Fingern fühlbar.",
      q8: "Versendet ihr Anhänger und Karten nach Europa?",
      a8: "Ja. Wir drucken und versenden innerhalb der Ukraine sowie in 15 EU-Länder über Nova Post EU und Meest: Deutschland, Polen, Tschechien, Spanien, Italien, Frankreich und weitere.",
      q9: "Was ist ein Stadtkarten-Kühlschrankmagnet?",
      a9: "Ein flacher 3D-gedruckter Magnet von ca. 60 mm mit der Karte deines Viertels: Straßen, Parks und Flüsse als Relief, hinten eine Tasche für einen Neodym-Magneten. Ein günstiges persönliches Andenken an einen besonderen Ort.",
    },
    seo: {
      title: "3D-Stadtkarten und personalisierte Anhänger — was wir machen",
      p1: "Monadruk verwandelt jeden Ort der Erde in ein druckbares 3D-Modell: eine 3D-Stadtkarte mit echten Gebäudehöhen, Parks und Flüssen aus OpenStreetMap-Daten, einen Karten-Schlüsselanhänger mit eigenem Text, einen topografischen Anhänger mit echtem Gebirgsrelief aus Satelliten-Höhendaten oder einen Stadtkarten-Magneten. Die 3MF- und STL-Dateien öffnen sich direkt in Bambu Studio und PrusaSlicer.",
      p2: "Kein 3D-Drucker? Wir drucken für dich in umweltfreundlichem Eco PLA und liefern in die Ukraine und in die EU. Eine 3D-Karte von Berlin, München, Kyjiw oder New York, das Relief der Alpen, ein personalisierter Anhänger zum Jahrestag — wähle einen Ort auf der Karte, und das Modell ist in Minuten fertig.",
    },
  },
  pl: {
    metaKeywords:
      "mapa miasta 3d, wydruk 3d mapy miasta, mapa miasta do druku 3d, brelok z mapą miasta, personalizowany brelok mapa, mapa stl, mapa 3mf, mapa topograficzna 3d, brelok z rzeźbą terenu, magnes na lodówkę mapa miasta, prezent z mapą, prezent na rocznicę, mapa dotykowa, Bambu Studio, PrusaSlicer",
    createKeywords:
      "stwórz mapę miasta 3d, generator map 3d online, pobierz mapę 3mf, mapa stl do pobrania, druk 3d mapy miasta, mapa z wysokościami budynków, magnes z mapą miasta, własna mapa miasta, mapa 3d Warszawy, mapa 3d Krakowa",
    keychainsKeywords:
      "brelok z mapą miasta, brelok mapa, brelok 3d personalizowany, brelok z imieniem, brelok z grawerem, brelok topograficzny, brelok z rzeźbą gór, brelok serce z mapą, brelok domek, prezent dla niego, prezent dla niej, prezent na rocznicę, brelok dla pary",
    keychainsDescription:
      "Brelok-mapa twojego miasta: osobisty brelok 3D z ulicami, budynkami i własnym napisem — albo brelok topo z prawdziwą rzeźbą gór (Tatry, Alpy, Karpaty). Stwórz online i pobierz 3MF lub zamów druk z Eco PLA.",
    showcaseKeywords:
      "galeria map 3d miast, przykłady druku 3d mapy, wydrukowane mapy miast, model 3d miasta, przykłady breloków z mapą, brelok topo przykład, magnes mapa przykład",
    faq: {
      q7: "Czy można zrobić brelok z rzeźbą gór zamiast mapy ulic?",
      a7: "Tak — to brelok topo. Włącz tryb „Rzeźba terenu” w kreatorze, a na żetonie zostanie wydrukowana prawdziwa rzeźba terenu z satelitarnych danych wysokości: Tatry, Alpy, Karpaty, fiordy czy wybrzeże. Szczyty i doliny czuć pod palcami.",
      q8: "Czy wysyłacie breloki i mapy po Europie?",
      a8: "Tak. Drukujemy i wysyłamy po Ukrainie oraz do 15 krajów UE przez Nova Post EU i Meest: Polska, Niemcy, Czechy, Hiszpania, Włochy, Francja i inne.",
      q9: "Czym jest magnes na lodówkę z mapą miasta?",
      a9: "To płaski magnes 3D ok. 60 mm z mapą twojej okolicy: ulice, parki i rzeki w reliefie, z tyłu kieszeń na magnes neodymowy. Budżetowa, osobista pamiątka wyjątkowego miejsca.",
    },
    seo: {
      title: "Mapy miast 3D i personalizowane breloki — co robimy",
      p1: "Monadruk zamienia dowolne miejsce na Ziemi w model 3D do druku: mapę dzielnicy z prawdziwymi wysokościami budynków, parkami i rzekami z danych OpenStreetMap, brelok z mapą miasta z własnym napisem, brelok topograficzny z rzeźbą gór z satelitarnych danych wysokości albo magnes na lodówkę z mapą. Pliki 3MF i STL otwierają się w Bambu Studio i PrusaSlicer.",
      p2: "Nie masz drukarki 3D? Wydrukujemy za ciebie z ekologicznego Eco PLA i wyślemy po Ukrainie i UE. Mapa 3D Warszawy, Krakowa, Kijowa czy Nowego Jorku, rzeźba Tatr lub Howerli, imienny brelok na rocznicę — wybierz miejsce na mapie, a model będzie gotowy w kilka minut.",
    },
  },
  fr: {
    metaKeywords:
      "carte de ville 3D, impression 3D carte de ville, plan de ville 3D, porte-clés carte de ville, porte-clés personnalisé carte, carte STL, carte 3MF, carte topographique 3D, porte-clés relief, magnet frigo carte de ville, cadeau carte personnalisé, cadeau anniversaire couple, carte tactile, Bambu Studio, PrusaSlicer",
    createKeywords:
      "créer carte de ville 3D, générateur de carte 3D en ligne, télécharger carte 3MF, carte STL à télécharger, imprimer carte de ville 3D, carte avec hauteurs des bâtiments, magnet carte de ville, carte personnalisée de ma ville, carte 3D de Paris, carte 3D de Lyon",
    keychainsKeywords:
      "porte-clés carte de ville, porte-clés carte, porte-clés 3D personnalisé, porte-clés avec prénom, porte-clés gravé, porte-clés topographique, porte-clés relief montagne, porte-clés cœur carte, porte-clés maison, cadeau pour lui, cadeau pour elle, cadeau d'anniversaire, porte-clés couple",
    keychainsDescription:
      "Un porte-clés carte de votre ville : un porte-clés 3D personnel avec rues, bâtiments et texte personnalisé — ou un porte-clés topographique avec le vrai relief des montagnes (Alpes, Carpates). Créez en ligne et téléchargez le 3MF ou commandez l'impression en Eco PLA.",
    showcaseKeywords:
      "galerie cartes 3D de villes, exemples impression 3D carte, cartes de villes imprimées, modèle 3D de ville, exemples porte-clés carte, porte-clés topo exemple, magnet carte exemple",
    faq: {
      q7: "Peut-on avoir un porte-clés avec le relief des montagnes au lieu des rues ?",
      a7: "Oui — c'est le porte-clés topo. Activez le mode « Relief du terrain » dans le configurateur, et le vrai relief du lieu est imprimé à partir des données satellite d'altitude : Alpes, Carpates, fjords ou littoral. Les sommets et vallées se lisent du bout des doigts.",
      q8: "Livrez-vous les porte-clés et cartes en Europe ?",
      a8: "Oui. Nous imprimons et expédions en Ukraine ainsi que dans 15 pays de l'UE via Nova Post EU et Meest : France, Allemagne, Pologne, Espagne, Italie et plus encore.",
      q9: "Qu'est-ce qu'un magnet frigo carte de ville ?",
      a9: "Un magnet plat imprimé en 3D d'environ 60 mm avec la carte de votre quartier : rues, parcs et rivières en relief, avec une poche au dos pour un aimant néodyme. Un souvenir personnel et abordable d'un lieu mémorable.",
    },
    seo: {
      title: "Cartes de villes 3D et porte-clés personnalisés — ce que nous faisons",
      p1: "Monadruk transforme n'importe quel endroit de la Terre en modèle 3D imprimable : une carte 3D de quartier avec les vraies hauteurs des bâtiments, parcs et rivières issues d'OpenStreetMap, un porte-clés carte de ville avec votre texte, un porte-clés topographique avec le vrai relief des montagnes issu des données satellite d'altitude, ou un magnet frigo carte. Les fichiers 3MF et STL s'ouvrent directement dans Bambu Studio et PrusaSlicer.",
      p2: "Pas d'imprimante 3D ? Nous imprimons pour vous en Eco PLA écologique et livrons en Ukraine et dans l'UE. Une carte 3D de Paris, Lyon, Kyiv ou New York, le relief des Alpes, un porte-clés personnalisé pour un anniversaire — choisissez un lieu sur la carte et le modèle est prêt en quelques minutes.",
    },
  },
  es: {
    metaKeywords:
      "mapa 3d de ciudad, impresión 3d mapa ciudad, plano de ciudad 3d, llavero mapa de ciudad, llavero personalizado mapa, mapa stl, mapa 3mf, mapa topográfico 3d, llavero relieve, imán nevera mapa ciudad, regalo mapa personalizado, regalo de aniversario, mapa táctil, Bambu Studio, PrusaSlicer",
    createKeywords:
      "crear mapa 3d de ciudad, generador de mapas 3d online, descargar mapa 3mf, mapa stl para descargar, imprimir mapa ciudad 3d, mapa con alturas de edificios, imán mapa ciudad, mapa personalizado de mi ciudad, mapa 3d de Madrid, mapa 3d de Barcelona",
    keychainsKeywords:
      "llavero mapa de ciudad, llavero mapa, llavero 3d personalizado, llavero con nombre, llavero grabado, llavero topográfico, llavero relieve montaña, llavero corazón mapa, llavero casa, regalo para él, regalo para ella, regalo de aniversario, llavero pareja",
    keychainsDescription:
      "Un llavero-mapa de tu ciudad: un llavero 3D personal con calles, edificios y texto propio — o un llavero topográfico con el relieve real de las montañas (Alpes, Pirineos, Cárpatos). Créalo online y descarga el 3MF o pide la impresión en Eco PLA.",
    showcaseKeywords:
      "galería mapas 3d de ciudades, ejemplos impresión 3d mapa, mapas de ciudades impresos, modelo 3d de ciudad, ejemplos llaveros mapa, llavero topo ejemplo, imán mapa ejemplo",
    faq: {
      q7: "¿Se puede hacer un llavero con relieve de montañas en vez del mapa de calles?",
      a7: "Sí — es el llavero topo. Activa el modo «Relieve del terreno» en el configurador y la ficha se imprime con el relieve real del lugar a partir de datos satelitales de altura: Alpes, Pirineos, Cárpatos, fiordos o costa. Las cumbres y valles se sienten con los dedos.",
      q8: "¿Enviáis llaveros y mapas por Europa?",
      a8: "Sí. Imprimimos y enviamos dentro de Ucrania y a 15 países de la UE a través de Nova Post EU y Meest: España, Alemania, Polonia, Italia, Francia y más.",
      q9: "¿Qué es un imán de nevera con mapa de ciudad?",
      a9: "Un imán plano impreso en 3D de unos 60 mm con el mapa de tu barrio: calles, parques y ríos en relieve, con un hueco trasero para un imán de neodimio. Un recuerdo personal y económico de un lugar especial.",
    },
    seo: {
      title: "Mapas 3D de ciudades y llaveros personalizados — qué hacemos",
      p1: "Monadruk convierte cualquier lugar de la Tierra en un modelo 3D imprimible: un mapa 3D de un barrio con alturas reales de edificios, parques y ríos a partir de datos de OpenStreetMap, un llavero con el mapa de tu ciudad y tu propio texto, un llavero topográfico con el relieve real de las montañas a partir de datos satelitales de altura, o un imán de nevera con mapa. Los archivos 3MF y STL se abren directamente en Bambu Studio y PrusaSlicer.",
      p2: "¿No tienes impresora 3D? Imprimimos por ti en Eco PLA ecológico y enviamos a Ucrania y a la UE. Un mapa 3D de Madrid, Barcelona, Kyiv o Nueva York, el relieve de los Alpes, un llavero personalizado para un aniversario — elige un lugar en el mapa y el modelo está listo en minutos.",
    },
  },
};

for (const [locale, d] of Object.entries(D)) {
  const file = path.join(__dirname, "..", "messages", `${locale}.json`);
  const m = JSON.parse(fs.readFileSync(file, "utf8"));
  m.meta.keywords = d.metaKeywords;
  m.createMeta.keywords = d.createKeywords;
  m.keychainsMeta.keywords = d.keychainsKeywords;
  m.keychainsMeta.description = d.keychainsDescription;
  m.showcaseMeta.keywords = d.showcaseKeywords;
  Object.assign(m.home.faq, d.faq);
  m.home.seo = d.seo;
  fs.writeFileSync(file, JSON.stringify(m, null, 2) + "\n", "utf8");
  console.log(`${locale}: patched (faq q7-q9, seo block, keywords)`);
}
