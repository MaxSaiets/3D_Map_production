// One-off: merge new i18n namespaces into messages/<locale>.json (idempotent).
import fs from "node:fs";
import path from "node:path";

const DIR = path.resolve("messages");

// data[ns][locale] = object
const data = {
  createMeta: {
    uk: { title: "Конструктор 3D-мапи міста — створити онлайн", description: "Створи 3D-мапу будь-якого міста світу онлайн: обери район, налаштуй висоти будинків, парки й річки та завантаж готовий 3MF/STL для друку.", keywords: "створити 3д мапу міста, конструктор 3д мапи, завантажити 3MF мапу, STL мапа міста, 3d друк мапи" },
    en: { title: "3D city map builder — create online", description: "Create a 3D map of any city online: pick a district, tune building, park and river heights and download a ready 3MF/STL for printing.", keywords: "create 3d city map, 3d map builder, download 3MF map, STL city map, 3d print map" },
    de: { title: "3D-Stadtkarten-Konfigurator — online erstellen", description: "Erstelle online eine 3D-Karte jeder Stadt: wähle ein Viertel, passe Gebäude-, Park- und Flusshöhen an und lade eine fertige 3MF/STL zum Drucken herunter.", keywords: "3d stadtkarte erstellen, 3d karten konfigurator, 3MF karte herunterladen, STL stadtkarte" },
    pl: { title: "Kreator mapy 3D miasta — stwórz online", description: "Stwórz online mapę 3D dowolnego miasta: wybierz dzielnicę, dostosuj wysokości budynków, parków i rzek i pobierz gotowy plik 3MF/STL do druku.", keywords: "stwórz mapę 3d miasta, kreator map 3d, pobierz mapę 3MF, mapa STL miasta" },
    fr: { title: "Configurateur de carte 3D — créer en ligne", description: "Crée en ligne une carte 3D de n'importe quelle ville : choisis un quartier, règle les hauteurs des bâtiments, parcs et rivières et télécharge un 3MF/STL prêt à imprimer.", keywords: "créer carte 3d ville, configurateur carte 3d, télécharger carte 3MF, carte STL ville" },
    es: { title: "Configurador de mapa 3D — crear en línea", description: "Crea en línea un mapa 3D de cualquier ciudad: elige un distrito, ajusta alturas de edificios, parques y ríos y descarga un 3MF/STL listo para imprimir.", keywords: "crear mapa 3d ciudad, configurador mapa 3d, descargar mapa 3MF, mapa STL ciudad" },
  },
  keychainsMeta: {
    uk: { title: "Брелок з картою міста — 3D-брелок на замовлення", description: "Брелок-мапа твого міста: персональний 3D-брелок 55×30 мм із вулицями, будинками та написом. Створи онлайн і завантаж 3MF або замов друк з Eco PLA.", keywords: "брелок з картою міста, брелок мапа, 3d брелок місто, брелок на замовлення, персональний брелок" },
    en: { title: "City map keychain — custom 3D keychain", description: "A map keychain of your city: a personal 3D keychain 55×30 mm with streets, buildings and text. Create online and download 3MF or order Eco PLA printing.", keywords: "city map keychain, map keychain, 3d city keychain, custom keychain, personalised keychain" },
    de: { title: "Schlüsselanhänger mit Stadtkarte — individuell", description: "Ein Karten-Anhänger deiner Stadt: ein persönlicher 3D-Anhänger 55×30 mm mit Straßen, Gebäuden und Text. Online erstellen und 3MF laden oder Eco-PLA-Druck bestellen.", keywords: "schlüsselanhänger stadtkarte, karten anhänger, 3d anhänger stadt, individueller anhänger" },
    pl: { title: "Brelok z mapą miasta — spersonalizowany 3D", description: "Brelok z mapą Twojego miasta: osobisty brelok 3D 55×30 mm z ulicami, budynkami i napisem. Stwórz online i pobierz 3MF lub zamów druk z Eco PLA.", keywords: "brelok z mapą miasta, brelok mapa, brelok 3d miasto, spersonalizowany brelok" },
    fr: { title: "Porte-clés carte de ville — 3D personnalisé", description: "Un porte-clés carte de ta ville : un porte-clés 3D personnel 55×30 mm avec rues, bâtiments et texte. Crée en ligne et télécharge le 3MF ou commande l'impression Eco PLA.", keywords: "porte-clés carte ville, porte-clés carte, porte-clés 3d ville, porte-clés personnalisé" },
    es: { title: "Llavero con mapa de ciudad — 3D personalizado", description: "Un llavero con el mapa de tu ciudad: un llavero 3D personal 55×30 mm con calles, edificios y texto. Crea en línea y descarga el 3MF o pide impresión en Eco PLA.", keywords: "llavero mapa ciudad, llavero mapa, llavero 3d ciudad, llavero personalizado" },
  },
  showcaseMeta: {
    uk: { title: "Галерея 3D-мап і брелків міст — приклади друку", description: "Галерея надрукованих 3D-мап та брелків міст України і світу. Покрути моделі у 3D, обери район і замов друк.", keywords: "галерея 3д мап міст, приклади 3d друку мапи, надруковані мапи міст, 3д модель міста" },
    en: { title: "Gallery of 3D city maps and keychains — print examples", description: "A gallery of printed 3D maps and keychains of cities worldwide. Rotate the models in 3D, pick a district and order printing.", keywords: "3d city maps gallery, 3d print map examples, printed city maps, 3d city model" },
    de: { title: "Galerie von 3D-Stadtkarten und Anhängern — Beispiele", description: "Eine Galerie gedruckter 3D-Karten und Anhänger von Städten weltweit. Drehe die Modelle in 3D, wähle ein Viertel und bestelle den Druck.", keywords: "3d stadtkarten galerie, 3d druck karten beispiele, gedruckte stadtkarten, 3d stadtmodell" },
    pl: { title: "Galeria map 3D i breloków miast — przykłady druku", description: "Galeria wydrukowanych map 3D i breloków miast z całego świata. Obróć modele w 3D, wybierz dzielnicę i zamów druk.", keywords: "galeria map 3d miast, przykłady druku 3d map, wydrukowane mapy miast, model 3d miasta" },
    fr: { title: "Galerie de cartes 3D et porte-clés de villes — exemples", description: "Une galerie de cartes 3D et porte-clés imprimés de villes du monde entier. Fais tourner les modèles en 3D, choisis un quartier et commande l'impression.", keywords: "galerie cartes 3d villes, exemples impression 3d carte, cartes de villes imprimées, modèle 3d ville" },
    es: { title: "Galería de mapas 3D y llaveros de ciudades — ejemplos", description: "Una galería de mapas 3D y llaveros impresos de ciudades de todo el mundo. Gira los modelos en 3D, elige un distrito y pide la impresión.", keywords: "galería mapas 3d ciudades, ejemplos impresión 3d mapa, mapas de ciudades impresos, modelo 3d ciudad" },
  },
  privacyMeta: {
    uk: { title: "Політика конфіденційності", description: "Політика конфіденційності Monadruk: як ми обробляємо та захищаємо твої дані." },
    en: { title: "Privacy Policy", description: "Monadruk privacy policy: how we process and protect your data." },
    de: { title: "Datenschutzerklärung", description: "Datenschutzerklärung von Monadruk: wie wir deine Daten verarbeiten und schützen." },
    pl: { title: "Polityka prywatności", description: "Polityka prywatności Monadruk: jak przetwarzamy i chronimy Twoje dane." },
    fr: { title: "Politique de confidentialité", description: "Politique de confidentialité de Monadruk : comment nous traitons et protégeons tes données." },
    es: { title: "Política de privacidad", description: "Política de privacidad de Monadruk: cómo procesamos y protegemos tus datos." },
  },
  termsMeta: {
    uk: { title: "Умови використання", description: "Умови використання сервісу Monadruk." },
    en: { title: "Terms of Service", description: "Terms of service for Monadruk." },
    de: { title: "Nutzungsbedingungen", description: "Nutzungsbedingungen von Monadruk." },
    pl: { title: "Regulamin", description: "Regulamin korzystania z Monadruk." },
    fr: { title: "Conditions d'utilisation", description: "Conditions d'utilisation de Monadruk." },
    es: { title: "Términos de uso", description: "Términos de uso de Monadruk." },
  },
  create: {
    uk: { backHome: "На головну", title: "Конструктор 3D-мапи", keychain: "Брелок", account: "Кабінет" },
    en: { backHome: "Home", title: "3D map builder", keychain: "Keychain", account: "Account" },
    de: { backHome: "Startseite", title: "3D-Karten-Konfigurator", keychain: "Anhänger", account: "Konto" },
    pl: { backHome: "Strona główna", title: "Kreator mapy 3D", keychain: "Brelok", account: "Konto" },
    fr: { backHome: "Accueil", title: "Configurateur de carte 3D", keychain: "Porte-clés", account: "Compte" },
    es: { backHome: "Inicio", title: "Configurador de mapa 3D", keychain: "Llavero", account: "Cuenta" },
  },
  showcase: {
    uk: { back: "На головну", eyebrow: "Галерея", title: "Надруковані мапи й брелки", subtitle: "Реальні моделі з міст України та світу. Обертай у 3D, обери — і замов друк.", all: "Усе", keys: "Брелки", maps: "3D-мапи", viewerDesc: "Точна 3D-мапа: вулиці, будівлі, парки й вода в масштабі. Перетягни, щоб роздивитись з усіх боків.", createKeychain: "Створити брелок", makeMap: "Зробити мапу", rotate3d: "Покрутити в 3D", keychainSize: "Брелок 55×30 мм", district: "3D-район", keyItem: "Брелок-мапа міста", mapItem: "3D-район міста", keyPrice: "від 290 ₴", mapPrice: "від 690 ₴", ctaTitle: "Не знайшов своє місто?", ctaDesc: "Створи мапу будь-якого міста світу за пару хвилин.", ctaButton: "Створити свою мапу", mKey: "Брелок-мапа", mHome: "Брелок «HOME»", mWater: "Брелок з рікою", mBridge: "Брелок з мостами", mBlock: "Район міста", mDistrict: "3D-квартал" },
    en: { back: "Home", eyebrow: "Gallery", title: "Printed maps & keychains", subtitle: "Real models from cities across Ukraine and the world. Rotate in 3D, pick one — and order printing.", all: "All", keys: "Keychains", maps: "3D maps", viewerDesc: "A precise 3D map: streets, buildings, parks and water to scale. Drag to view from any side.", createKeychain: "Create a keychain", makeMap: "Make a map", rotate3d: "Rotate in 3D", keychainSize: "Keychain 55×30 mm", district: "3D district", keyItem: "City map keychain", mapItem: "3D city district", keyPrice: "from €7", mapPrice: "from €17", ctaTitle: "Didn't find your city?", ctaDesc: "Create a map of any city in the world in a couple of minutes.", ctaButton: "Create your map", mKey: "Map keychain", mHome: "“HOME” keychain", mWater: "River keychain", mBridge: "Bridges keychain", mBlock: "City district", mDistrict: "3D block" },
    de: { back: "Startseite", eyebrow: "Galerie", title: "Gedruckte Karten & Anhänger", subtitle: "Echte Modelle aus Städten der Ukraine und der Welt. In 3D drehen, auswählen — und Druck bestellen.", all: "Alle", keys: "Anhänger", maps: "3D-Karten", viewerDesc: "Eine präzise 3D-Karte: Straßen, Gebäude, Parks und Wasser maßstabsgetreu. Ziehen, um von allen Seiten zu sehen.", createKeychain: "Anhänger erstellen", makeMap: "Karte erstellen", rotate3d: "In 3D drehen", keychainSize: "Anhänger 55×30 mm", district: "3D-Viertel", keyItem: "Schlüsselanhänger mit Stadtkarte", mapItem: "3D-Stadtviertel", keyPrice: "ab 7 €", mapPrice: "ab 17 €", ctaTitle: "Deine Stadt nicht gefunden?", ctaDesc: "Erstelle in wenigen Minuten eine Karte jeder Stadt der Welt.", ctaButton: "Karte erstellen", mKey: "Karten-Anhänger", mHome: "„HOME“-Anhänger", mWater: "Fluss-Anhänger", mBridge: "Brücken-Anhänger", mBlock: "Stadtviertel", mDistrict: "3D-Block" },
    pl: { back: "Strona główna", eyebrow: "Galeria", title: "Wydrukowane mapy i breloki", subtitle: "Prawdziwe modele z miast Ukrainy i świata. Obróć w 3D, wybierz — i zamów druk.", all: "Wszystko", keys: "Breloki", maps: "Mapy 3D", viewerDesc: "Dokładna mapa 3D: ulice, budynki, parki i woda w skali. Przeciągnij, aby obejrzeć ze wszystkich stron.", createKeychain: "Stwórz brelok", makeMap: "Zrób mapę", rotate3d: "Obróć w 3D", keychainSize: "Brelok 55×30 mm", district: "Dzielnica 3D", keyItem: "Brelok z mapą miasta", mapItem: "Dzielnica 3D miasta", keyPrice: "od 7 €", mapPrice: "od 17 €", ctaTitle: "Nie znalazłeś swojego miasta?", ctaDesc: "Stwórz mapę dowolnego miasta na świecie w kilka minut.", ctaButton: "Stwórz swoją mapę", mKey: "Brelok-mapa", mHome: "Brelok „HOME”", mWater: "Brelok z rzeką", mBridge: "Brelok z mostami", mBlock: "Dzielnica miasta", mDistrict: "Kwartał 3D" },
    fr: { back: "Accueil", eyebrow: "Galerie", title: "Cartes & porte-clés imprimés", subtitle: "De vrais modèles de villes d'Ukraine et du monde. Tourne en 3D, choisis — et commande l'impression.", all: "Tout", keys: "Porte-clés", maps: "Cartes 3D", viewerDesc: "Une carte 3D précise : rues, bâtiments, parcs et eau à l'échelle. Fais glisser pour voir sous tous les angles.", createKeychain: "Créer un porte-clés", makeMap: "Faire une carte", rotate3d: "Tourner en 3D", keychainSize: "Porte-clés 55×30 mm", district: "Quartier 3D", keyItem: "Porte-clés carte de ville", mapItem: "Quartier 3D", keyPrice: "dès 7 €", mapPrice: "dès 17 €", ctaTitle: "Tu n'as pas trouvé ta ville ?", ctaDesc: "Crée la carte de n'importe quelle ville du monde en quelques minutes.", ctaButton: "Créer ta carte", mKey: "Porte-clés carte", mHome: "Porte-clés « HOME »", mWater: "Porte-clés rivière", mBridge: "Porte-clés ponts", mBlock: "Quartier", mDistrict: "Bloc 3D" },
    es: { back: "Inicio", eyebrow: "Galería", title: "Mapas y llaveros impresos", subtitle: "Modelos reales de ciudades de Ucrania y el mundo. Gira en 3D, elige — y pide la impresión.", all: "Todo", keys: "Llaveros", maps: "Mapas 3D", viewerDesc: "Un mapa 3D preciso: calles, edificios, parques y agua a escala. Arrastra para verlo desde todos los lados.", createKeychain: "Crear un llavero", makeMap: "Hacer un mapa", rotate3d: "Girar en 3D", keychainSize: "Llavero 55×30 mm", district: "Distrito 3D", keyItem: "Llavero con mapa de ciudad", mapItem: "Distrito 3D", keyPrice: "desde 7 €", mapPrice: "desde 17 €", ctaTitle: "¿No encontraste tu ciudad?", ctaDesc: "Crea el mapa de cualquier ciudad del mundo en un par de minutos.", ctaButton: "Crear tu mapa", mKey: "Llavero mapa", mHome: "Llavero «HOME»", mWater: "Llavero río", mBridge: "Llavero puentes", mBlock: "Distrito", mDistrict: "Bloque 3D" },
  },
};

const locales = ["uk", "en", "de", "pl", "fr", "es"];
for (const loc of locales) {
  const file = path.join(DIR, `${loc}.json`);
  const json = JSON.parse(fs.readFileSync(file, "utf8"));
  for (const ns of Object.keys(data)) {
    json[ns] = { ...(json[ns] || {}), ...data[ns][loc] };
  }
  fs.writeFileSync(file, JSON.stringify(json, null, 2) + "\n", "utf8");
  console.log("merged", loc);
}
