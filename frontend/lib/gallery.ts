// ──────────────────────────────────────────────────────────────────────────
// Галерея: ОКРЕМА СТОРІНКА НА КОЖНЕ ФОТО (/foto/[slug]) — SEO під Google
// Картинки + довгий хвіст («брелок серце з картою», «панно з шестикутних
// плиток», «3d мапа району з позначкою дому»…).
//
// Правила контенту (анти-thin/doorway):
//  - описи ЧЕСНІ: лише те, що справді видно на фото. Місто НЕ вигадуємо —
//    на фото районів не підписано, тому пишемо «район», а не «Поділ».
//  - «real» = фото реального друку; «render» = рендер моделі з конструктора
//    (так і підписано на сторінці — не видаємо рендер за фото).
//  - кожен запис має унікальні title/alt/опис; спільні абзаци (матеріал,
//    ціна, терміни) — лише доповнення за типом виробу.
// uk + en повні; інші локалі не генеруються (фокус SEO — Україна, дубль-en
// сторінки de/pl/fr/es Google схлопував — памʼять gsc-indexing-diagnosis).
// ──────────────────────────────────────────────────────────────────────────

export type GalleryKind = "keychain" | "heart" | "group" | "map" | "panno";
export type GalleryLocale = "uk" | "en";

export type GalleryItem = {
  slug: string;
  src: string;
  w: number;
  h: number;
  kind: GalleryKind;
  source: "real" | "render";
  title: Record<GalleryLocale, string>;
  alt: Record<GalleryLocale, string>;
  desc: Record<GalleryLocale, string>;
};

type Raw = Omit<GalleryItem, "title" | "alt" | "desc"> & {
  t: [string, string];
  a: [string, string];
  d: [string, string];
};

const RAW: Raw[] = [
  // ── Реальні фото: брелоки ──
  {
    slug: "kolektsiya-brelokiv-z-kartamy-mist", src: "/real/group-1.webp", w: 825, h: 1100, kind: "group", source: "real",
    t: ["Колекція брелоків з картами міст на дерев'яному столі", "A set of city map keychains on a wooden table"],
    a: ["П'ять 3D-друкованих брелоків з картами районів і написами HOME та LOVE на дерев'яному столі", "Five 3D-printed map keychains with HOME and LOVE lettering on a wooden table"],
    d: [
      "П'ять брелоків різних форм — прямокутник, овал, капсула і серце — поруч, щоб порівняти розміри. На кожному свій район: дороги темною лінією, парки зеленим, будинки об'ємними блоками. Написи HOME і LOVE друкуються рельєфом поверх карти.",
      "Five keychains in different shapes — rectangle, oval, capsule and heart — side by side so you can compare sizes. Each one carries its own neighbourhood: roads in dark lines, parks in green, buildings as raised blocks. The HOME and LOVE words are printed in relief on top of the map.",
    ],
  },
  {
    slug: "breloky-z-kartamy-na-kameni", src: "/real/group-2.webp", w: 825, h: 1100, kind: "group", source: "real",
    t: ["Брелоки з 3D-картою на камені серед зелені", "Map keychains on a stone among leaves"],
    a: ["Дев'ять брелоків з картою району на сірому камені, навколо листя", "Nine neighbourhood map keychains lying on a grey stone surrounded by leaves"],
    d: [
      "Фото на вулиці: дев'ять готових брелоків на камені. Видно, як біла основа і темні дороги читаються навіть при денному світлі, а зелені парки виділяються кольором. Такий набір підходить на весілля чи корпоратив — кожен гість отримує свою адресу.",
      "An outdoor shot: nine finished keychains on a stone. The white base and dark roads stay readable in daylight and the green parks stand out in colour. A set like this works for weddings or team events — every guest gets their own address.",
    ],
  },
  {
    slug: "rizni-formy-brelokiv-z-kartoyu", src: "/real/group-3.webp", w: 825, h: 1100, kind: "group", source: "real",
    t: ["Різні форми брелоків з картою: прямокутник, овал, капсула, жетон", "Keychain shapes with a map: rectangle, oval, capsule, tag"],
    a: ["П'ять брелоків з 3D-картою різних форм на білому тлі з металевими кільцями", "Five 3D map keychains of different shapes with metal rings on a white background"],
    d: [
      "Порівняння форм на білому тлі: прямокутний, овальний, капсула і жетон. Форма впливає на те, скільки кварталів поміститься — прямокутник вміщує найбільше, капсула виглядає найакуратніше на ключах. Кільце і ланцюжок уже в комплекті.",
      "Shapes compared on a white background: rectangle, oval, capsule and tag. The shape decides how many blocks fit — a rectangle holds the most, a capsule looks the neatest on a key ring. Ring and chain are included.",
    ],
  },
  {
    slug: "brelok-sertse-love-z-kartoyu", src: "/real/heart-1.webp", w: 825, h: 1100, kind: "heart", source: "real",
    t: ["Брелок-серце LOVE з картою району", "Heart keychain with a map and the word LOVE"],
    a: ["Брелок у формі серця з 3D-картою вулиць і написом LOVE на дерев'яних дошках", "Heart-shaped keychain with a 3D street map and the word LOVE on wooden boards"],
    d: [
      "Серце з картою місця, важливого для двох: де познайомились, перше спільне житло чи місце освідчення. Написом може бути LOVE, дата або ініціали. Подарунок на річницю чи День закоханих.",
      "A heart with a map of a place that matters to two people: where they met, their first home or where they got engaged. The text can be LOVE, a date or initials. An anniversary or Valentine's gift.",
    ],
  },
  {
    slug: "brelok-sertse-z-kartoyu-rayonu", src: "/real/heart-2.webp", w: 825, h: 1100, kind: "heart", source: "real",
    t: ["Серце з картою району — брелок з темним контуром", "Heart map keychain with a dark outline"],
    a: ["Брелок-серце з вулицями району, темним обідком і ланцюжком на дерев'яному тлі", "Heart keychain with neighbourhood streets, a dark rim and a chain on wood"],
    d: [
      "Той самий формат серця з іншого ракурсу: видно темний обідок, який захищає краї, і висоту будинків над основою. Обідок друкується окремим кольором, тому силует серця чіткий навіть здалеку.",
      "The same heart format from another angle: you can see the dark rim that protects the edges and how far the buildings rise above the base. The rim is printed in a separate colour, so the heart silhouette stays crisp from a distance.",
    ],
  },
  {
    slug: "brelok-ovalnyi-love", src: "/real/key-1.webp", w: 825, h: 1100, kind: "keychain", source: "real",
    t: ["Овальний брелок LOVE з картою і парком", "Oval LOVE keychain with a map and a park"],
    a: ["Овальний 3D-брелок з написом LOVE, дорогами і зеленим парком на світлому тлі", "Oval 3D keychain with the word LOVE, roads and a green park on a light background"],
    d: [
      "Овальний брелок крупним планом: зелений парк у кутку, діагональна вулиця і будинки рельєфом. Напис LOVE стоїть по діагоналі вздовж вулиці — розташування і текст напису обираєте в конструкторі.",
      "An oval keychain up close: a green park in the corner, a diagonal street and raised buildings. The word LOVE runs diagonally along the street — you choose the text and its position in the builder.",
    ],
  },
  {
    slug: "brelok-kapsula-home", src: "/real/key-2.webp", w: 825, h: 1100, kind: "keychain", source: "real",
    t: ["Брелок-капсула HOME з картою дому", "Capsule HOME keychain with a map of home"],
    a: ["Брелок-капсула з написом HOME, будинками кварталу, ланцюжком і кільцем", "Capsule keychain with the word HOME, block buildings, a chain and a ring"],
    d: [
      "Капсула з написом HOME — класика для ключів від нової квартири. На брелоку квартал навколо вашої адреси: видно двори, під'їзні дороги і сквер. Хороший подарунок на новосілля.",
      "A capsule with the word HOME — a classic for the keys to a new flat. It shows the block around your address: courtyards, access roads and a small square. A good housewarming gift.",
    ],
  },
  {
    slug: "brelok-home-pryamokutnyi", src: "/real/key-3.webp", w: 825, h: 1100, kind: "keychain", source: "real",
    t: ["Прямокутний брелок HOME з картою вулиць", "Rectangular HOME keychain with a street map"],
    a: ["Прямокутний брелок з 3D-картою вулиць, написом HOME і зеленою ділянкою парку", "Rectangular keychain with a 3D street map, the word HOME and a green park area"],
    d: [
      "Прямокутна форма вміщує найбільше кварталів — тут кілька вулиць, що перетинаються, і парк праворуч. Напис HOME внизу вздовж краю не закриває саму карту.",
      "The rectangle fits the most blocks — here several crossing streets and a park on the right. The HOME label sits along the bottom edge so it doesn't cover the map itself.",
    ],
  },
  {
    slug: "brelok-home-z-parkom", src: "/real/key-4.webp", w: 825, h: 1100, kind: "keychain", source: "real",
    t: ["Брелок HOME з парком — вигляд під кутом", "HOME keychain with a park — angled view"],
    a: ["Брелок HOME з картою району під кутом, видно висоту будинків і рельєф доріг", "HOME map keychain at an angle showing building heights and raised roads"],
    d: [
      "Той самий прямокутний брелок під кутом: так видно, що будинки справді об'ємні, а дороги трохи підняті над основою. Суцільна пластикова основа тримає форму і в кишені, і на зв'язці ключів.",
      "The same rectangular keychain at an angle: the buildings are truly raised and the roads sit slightly above the base. The solid plastic base keeps its shape in a pocket or on a key ring.",
    ],
  },
  // ── Реальні фото: мапи ──
  {
    slug: "kruhla-3d-mapa-rayonu-zverhu", src: "/real/map-1.webp", w: 1100, h: 825, kind: "map", source: "real",
    t: ["Кругла 3D-мапа району — вигляд зверху", "Round 3D neighbourhood map — top view"],
    a: ["Кругла 3D-друкована мапа району з вулицями, парками і червоною позначкою дому, вид зверху", "Round 3D-printed neighbourhood map with streets, parks and a red home marker, top view"],
    d: [
      "Кругла мапа зверху: вулиці темні, парки й сквери зелені, будинки білі. Червоним виділено один будинок — це позначка «мій дім», її ставите в конструкторі на свою адресу.",
      "A round map from above: streets dark, parks and squares green, buildings white. One building is highlighted in red — the 'my home' marker you place on your address in the builder.",
    ],
  },
  {
    slug: "kruhla-3d-mapa-z-parkom", src: "/real/map-2.webp", w: 825, h: 1100, kind: "map", source: "real",
    t: ["3D-мапа району з парком і бульваром", "3D neighbourhood map with a park and boulevard"],
    a: ["Кругла 3D-мапа з зеленою смугою бульвару, парком по краю і щільними кварталами на столі", "Round 3D map with a green boulevard strip, a park along the edge and dense blocks on a table"],
    d: [
      "Мапа під кутом на дерев'яному столі. Зелена смуга бульвару ділить квартали навпіл, а великий парк іде вздовж краю. Такі деталі беруться з OpenStreetMap — тому мапа збігається з реальним районом.",
      "The map at an angle on a wooden table. A green boulevard splits the blocks in two and a large park runs along the edge. These details come from OpenStreetMap, so the map matches the real neighbourhood.",
    ],
  },
  {
    slug: "3d-mapa-kvartalu-z-poznachkoyu-domu", src: "/real/map-3.webp", w: 825, h: 1100, kind: "map", source: "real",
    t: ["3D-мапа кварталу з позначкою дому", "3D block map with a home marker"],
    a: ["3D-друкована мапа кварталу з червоним будинком-позначкою і висотною забудовою", "3D-printed block map with a red marker building and tall buildings"],
    d: [
      "Квартал з висотками і червоним будинком у центрі — саме так виглядає подарунок «наш дім на карті». Висоту будинків конструктор бере з даних про поверховість, тому багатоповерхівки помітно вищі за сусідів.",
      "A block with tall buildings and a red house in the centre — exactly what an 'our home on the map' gift looks like. Building heights come from floor-count data, so high-rises stand clearly above their neighbours.",
    ],
  },
  {
    slug: "kruhla-3d-mapa-rayonu-na-stoli", src: "/real/map-4.webp", w: 825, h: 1100, kind: "map", source: "real",
    t: ["Кругла 3D-мапа району на робочому столі", "Round 3D neighbourhood map on a desk"],
    a: ["Кругла 3D-мапа міста з білими будинками, темними вулицями і зеленими сквериками", "Round 3D city map with white buildings, dark streets and small green squares"],
    d: [
      "Мапа як настільний сувенір: стоїть на столі без підставки, розмір обираєте від 5,5 до 15 см і більше. Видно, як вулиці утворюють сітку кварталів, а невеликі сквери всередині дворів теж надруковано зеленим.",
      "The map as a desk piece: it stands on a table without a holder, in sizes from 5.5 to 15 cm and up. The streets form a grid of blocks and even small courtyard squares are printed in green.",
    ],
  },
  // ── Реальні фото: панно з плиток ──
  {
    slug: "panno-z-dvokh-shestykutnykh-plytok", src: "/real/panno-1.webp", w: 1100, h: 825, kind: "panno", source: "real",
    t: ["Панно з двох шестикутних плиток з картою міста", "Panel of two hexagonal city map tiles"],
    a: ["Дві шестикутні 3D-плитки з картою міста, з'єднані в панно, з великими зеленими зонами", "Two hexagonal 3D city map tiles joined into a panel with large green areas"],
    d: [
      "Дві шестикутні плитки, які продовжують одна одну: вулиці переходять з плитки на плитку без розриву. Праворуч велика зелена зона. Плитки можна додавати — так мапа росте разом з вашими улюбленими місцями.",
      "Two hexagonal tiles that continue into each other: streets flow across the seam without a break. On the right, a large green area. You can keep adding tiles, so the map grows with your favourite places.",
    ],
  },
  {
    slug: "panno-dvi-plytky-tsentr-mista", src: "/real/panno-2.webp", w: 1100, h: 825, kind: "panno", source: "real",
    t: ["Панно з двох плиток — щільний центр міста", "Two-tile panel — a dense city centre"],
    a: ["Шестикутні плитки з 3D-картою щільного центру міста на дерев'яній поверхні", "Hexagonal tiles with a 3D map of a dense city centre on a wooden surface"],
    d: [
      "Центр міста з дрібною сіткою вулиць. На щільній забудові добре видно, чому ми радимо плитки від 8 см: менше — і провулки зливаються. Зелені зони по краях — набережна і парк.",
      "A city centre with a fine street grid. Dense areas show why we recommend tiles of 8 cm or more: any smaller and the lanes merge. The green areas at the edges are an embankment and a park.",
    ],
  },
  {
    slug: "panno-shestykutni-plytky-zbloku", src: "/real/panno-3.webp", w: 1100, h: 825, kind: "panno", source: "real",
    t: ["Шестикутні плитки з картою — панно під кутом", "Hexagonal map tiles — panel at an angle"],
    a: ["Панно з двох шестикутних 3D-плиток з картою міста під кутом, видно висоту будинків", "Panel of two hexagonal 3D city map tiles at an angle showing building heights"],
    d: [
      "Панно під кутом: видно, що це не плоский принт, а рельєф — будинки різної висоти відкидають тіні. На стіні така мапа змінюється протягом дня разом зі світлом з вікна.",
      "The panel at an angle: it's relief, not a flat print — buildings of different heights cast shadows. On a wall the map changes through the day with the light from the window.",
    ],
  },
  {
    slug: "vertykalne-panno-z-dvokh-plytok", src: "/real/panno-4.webp", w: 825, h: 1100, kind: "panno", source: "real",
    t: ["Вертикальне панно з двох плиток з картою", "Vertical two-tile map panel"],
    a: ["Дві шестикутні плитки з 3D-картою вулиць, складені вертикально на дерев'яному столі", "Two hexagonal 3D street map tiles stacked vertically on a wooden table"],
    d: [
      "Ті самі шестикутники можна скласти вертикально — вийде вузьке панно для простінка чи дверного проходу. Порядок плиток задаєте самі, ми друкуємо кожну з картою саме своєї ділянки.",
      "The same hexagons can be stacked vertically to make a narrow panel for a pier or doorway. You set the tile order, and we print each tile with its own part of the map.",
    ],
  },
  {
    slug: "panno-z-plytok-na-bilomu-tli", src: "/real/panno-5.webp", w: 825, h: 1100, kind: "panno", source: "real",
    t: ["Панно з плиток з картою міста на білому тлі", "City map tile panel on a white background"],
    a: ["Вертикальне панно з двох шестикутних плиток з 3D-картою міста і зеленими парками на білому тлі", "Vertical panel of two hexagonal 3D city map tiles with green parks on white"],
    d: [
      "Панно на білому тлі — так воно виглядатиме на світлій стіні. Зелені парки внизу, щільний центр угорі. Для кріплення достатньо двостороннього скотча або невеликих цвяхів.",
      "The panel on white — this is how it looks on a light wall. Green parks at the bottom, a dense centre at the top. Double-sided tape or small nails are enough to hang it.",
    ],
  },
  // ── Реальні фото крупним планом ──
  {
    slug: "shestykutna-plytka-zblyzka", src: "/showcase/real-1.webp", w: 1200, h: 920, kind: "panno", source: "real",
    t: ["Шестикутна плитка з картою — макрофото", "Hexagonal map tile — close-up"],
    a: ["Крупний план шестикутної 3D-плитки: перехрестя вулиць, білі будинки і зелені дворики", "Close-up of a hexagonal 3D tile: crossing streets, white buildings and green courtyards"],
    d: [
      "Макрофото плитки: видно шари друку і те, наскільки дрібні деталі тримає модель — окремі корпуси будинків, внутрішні дворики, вузькі проїзди між ними.",
      "A macro shot of a tile: you can see the print layers and how much fine detail the model keeps — individual building wings, inner courtyards and the narrow driveways between them.",
    ],
  },
  {
    slug: "plytka-z-relyefom-i-vysotkamy", src: "/showcase/real-2.webp", w: 1200, h: 920, kind: "map", source: "real",
    t: ["3D-мапа з рельєфом і висотками", "3D map with terrain relief and high-rises"],
    a: ["3D-друкована мапа з рельєфом місцевості, висотними будинками, зеленим схилом і дорогою", "3D-printed map with terrain relief, tall buildings, a green slope and a road"],
    d: [
      "Мапа з увімкненим рельєфом: зелений схил піднімається сходинками, дорога огинає пагорб, а висотки стоять на вершині. Рельєф — опція за +85 ₴, найкраще працює для горбистих міст.",
      "A map with terrain relief on: the green slope rises in steps, the road winds around the hill and the high-rises stand on top. Relief is an add-on for +85 UAH and works best for hilly cities.",
    ],
  },
  {
    slug: "3d-mapa-z-relyefom-zblyzka", src: "/showcase/real-3.webp", w: 1200, h: 920, kind: "map", source: "real",
    t: ["Рельєф на 3D-мапі міста — крупний план", "Terrain relief on a 3D city map — close-up"],
    a: ["Крупний план 3D-мапи з рельєфом: пагорб зеленими терасами, серпантин дороги і будинки", "Close-up of a 3D map with relief: green terraced hill, winding road and buildings"],
    d: [
      "Та сама ділянка ближче: тераси рельєфу, серпантин дороги і будинки на різних висотах. Так на мапі впізнаються справжні пагорби міста, а не лише план вулиць.",
      "The same area closer: relief terraces, a winding road and buildings at different heights. This is how the real hills of a city become recognisable, not just its street plan.",
    ],
  },
  {
    slug: "sertse-love-makro", src: "/showcase/real-4.webp", w: 1200, h: 920, kind: "heart", source: "real",
    t: ["Брелок-серце LOVE — макрофото деталей", "LOVE heart keychain — macro detail"],
    a: ["Макрофото брелока-серця з написом LOVE, кільцем і рельєфними вулицями", "Macro photo of a heart keychain with the word LOVE, a ring and raised streets"],
    d: [
      "Макро серця: напис LOVE об'ємний, букви не стираються, бо надруковані разом з основою, а не наклеєні. Кільце кріпиться через отвір у верхній частині серця.",
      "A macro of the heart: the LOVE letters are raised and won't rub off because they're printed together with the base, not glued on. The ring goes through a hole at the top of the heart.",
    ],
  },
  {
    slug: "brelok-home-makro", src: "/showcase/real-5.webp", w: 1200, h: 920, kind: "keychain", source: "real",
    t: ["Брелок HOME — макрофото карти й напису", "HOME keychain — macro of the map and text"],
    a: ["Макрофото брелока-капсули з написом HOME, будинками і зеленими ділянками", "Macro of a capsule keychain with HOME lettering, buildings and green plots"],
    d: [
      "Капсула HOME зблизька: напис чорним рельєфом, поруч будинки і маленькі зелені ділянки. Текст можна замінити на ім'я, назву вулиці чи номер будинку.",
      "The HOME capsule up close: black raised lettering beside buildings and small green plots. You can replace the text with a name, a street name or a house number.",
    ],
  },
  {
    slug: "breloky-home-i-love-zblyzka", src: "/showcase/real-6.webp", w: 1200, h: 920, kind: "group", source: "real",
    t: ["Брелоки HOME і LOVE зблизька", "HOME and LOVE keychains up close"],
    a: ["Кілька 3D-брелоків з написами HOME і LOVE та картами різних районів зблизька", "Several 3D keychains with HOME and LOVE and maps of different neighbourhoods up close"],
    d: [
      "Кілька брелоків поруч: кожен з картою свого району і власним написом. Можна взяти пару «HOME для себе + LOVE у подарунок» — обидва в одному замовленні.",
      "Several keychains together, each with its own neighbourhood map and text. You can take a pair — HOME for yourself and LOVE as a gift — both in one order.",
    ],
  },
  {
    slug: "kruhla-mapa-detali-budynkiv", src: "/showcase/real-7.webp", w: 1200, h: 920, kind: "map", source: "real",
    t: ["Кругла 3D-мапа — деталі будинків і дворів", "Round 3D map — building and courtyard detail"],
    a: ["Кругла 3D-мапа міста під кутом: квартали з дворами, зелений бульвар, червона позначка", "Round 3D city map at an angle: blocks with courtyards, a green boulevard and a red marker"],
    d: [
      "Кругла мапа під кутом: квартали з замкненими дворами, зелений бульвар і червоний будинок-позначка. Видно, що навіть невеликі прибудови й гаражі надруковано окремо.",
      "The round map at an angle: blocks with closed courtyards, a green boulevard and the red marker house. Even small annexes and garages are printed as separate shapes.",
    ],
  },
  {
    slug: "kruhla-mapa-z-chervonym-budynkom", src: "/showcase/real-8.webp", w: 1200, h: 920, kind: "map", source: "real",
    t: ["3D-мапа з червоним будинком — «мій дім»", "3D map with a red 'my home' building"],
    a: ["Кругла 3D-мапа району з червоним будинком-позначкою серед білих кварталів і зелених скверів", "Round 3D neighbourhood map with a red marker building among white blocks and green squares"],
    d: [
      "Червоний будинок на мапі — це ваша адреса. Позначку можна поставити на будь-яку будівлю в межах обраної ділянки: дім, школу, офіс, місце першого побачення.",
      "The red building on the map is your address. You can put the marker on any building inside the chosen area: home, school, office, the place of a first date.",
    ],
  },
  {
    slug: "panno-detali-vulyts", src: "/showcase/real-9.webp", w: 1200, h: 920, kind: "panno", source: "real",
    t: ["Панно з плиток — деталі вулиць", "Tile panel — street detail"],
    a: ["Крупний план панно з шестикутних плиток: вулиці, квартали і зелені зони", "Close-up of a hexagonal tile panel: streets, blocks and green areas"],
    d: [
      "Панно зблизька: стик двох плиток майже непомітний, бо вулиці підігнано одна до одної. Зелені зони ліворуч — парк і промзона з деревами.",
      "The panel up close: the seam between two tiles is barely visible because the streets are matched across it. The green areas on the left are a park and a tree-lined industrial zone.",
    ],
  },
  {
    slug: "panno-shchilna-zabudova", src: "/showcase/real-10.webp", w: 1200, h: 920, kind: "panno", source: "real",
    t: ["Панно зі щільною забудовою старого міста", "Panel with the dense fabric of an old town"],
    a: ["3D-панно з плиток зі щільною забудовою старого міста і вузькими вуличками", "3D tile panel showing the dense fabric of an old town with narrow lanes"],
    d: [
      "Старе місто на панно: будинки впритул один до одного, вузькі вулички і кілька зелених дворів. Саме на таких районах 3D-мапа виглядає найбагатше.",
      "An old town on a panel: houses wall to wall, narrow lanes and a few green courtyards. Districts like this are where a 3D map looks richest.",
    ],
  },
  // ── Рендери моделей з конструктора ──
  {
    slug: "render-brelok-z-napysom", src: "/showcase/keychain-1.webp", w: 1200, h: 920, kind: "keychain", source: "render",
    t: ["Модель брелока з картою і написом — рендер", "Map keychain model with text — render"],
    a: ["Рендер 3D-моделі прямокутного брелока з картою кварталу, написом і кільцем", "Render of a rectangular keychain model with a block map, text and a ring"],
    d: [
      "Так конструктор показує модель до друку: прямокутний брелок з кількома кварталами, звивистими проїздами і написом угорі. Модель можна покрутити, змінити ділянку й текст, а потім замовити друк.",
      "This is how the builder shows a model before printing: a rectangular keychain with several blocks, winding driveways and text at the top. You can rotate it, change the area and text, then order the print.",
    ],
  },
  {
    slug: "render-brelok-rozvylka-vulyts", src: "/showcase/keychain-2.webp", w: 1200, h: 920, kind: "keychain", source: "render",
    t: ["Брелок з розвилкою вулиць — модель", "Keychain with a street fork — model"],
    a: ["Рендер брелока з картою: розвилка вулиць, кругова дорога і будинки", "Keychain render with a map: a fork in the road, a loop road and buildings"],
    d: [
      "Модель брелока, де вулиця розходиться на дві, а праворуч петля під'їзної дороги. Напис розміщено у вільному місці, щоб не перекривати будинки.",
      "A keychain model where the street splits in two with an access loop on the right. The text sits in an empty area so it doesn't cover the buildings.",
    ],
  },
  {
    slug: "render-brelok-kvartal-z-dvorom", src: "/showcase/keychain-3.webp", w: 1200, h: 920, kind: "keychain", source: "render",
    t: ["Брелок з кварталом і внутрішнім двором — модель", "Keychain with a block and inner courtyard — model"],
    a: ["Рендер брелока з картою кварталу з внутрішнім двором і сіткою проїздів", "Keychain render showing a block with an inner courtyard and a grid of driveways"],
    d: [
      "Модель брелока з типовим спальним кварталом: будинки по периметру, зелений двір усередині і сітка проїздів. Так виглядає типовий «домашній» брелок.",
      "A keychain model of a typical residential block: buildings around the edge, a green courtyard inside and a grid of driveways. This is what a typical 'home' keychain looks like.",
    ],
  },
  {
    slug: "render-brelok-kruhova-ploshcha", src: "/showcase/keychain-4.webp", w: 1200, h: 920, kind: "keychain", source: "render",
    t: ["Брелок з круговою площею — модель", "Keychain with a roundabout square — model"],
    a: ["Рендер брелока з картою круглої площі, від якої розходяться проспекти", "Keychain render with a round square and avenues radiating from it"],
    d: [
      "Кругла площа з променями проспектів — одна з найвпізнаваніших форм на карті. На брелоку вона читається навіть без напису.",
      "A round square with avenues radiating out — one of the most recognisable shapes on any map. On a keychain it reads even without a label.",
    ],
  },
  {
    slug: "render-brelok-oval-home-z-vodoyu", src: "/showcase/keychain-5.webp", w: 1200, h: 920, kind: "keychain", source: "render",
    t: ["Овальний брелок HOME з водоймою — модель", "Oval HOME keychain with water — model"],
    a: ["Рендер овального брелока з написом HOME, вулицею і блакитною водоймою", "Render of an oval HOME keychain with a street and a blue body of water"],
    d: [
      "Овальна модель з написом HOME і водоймою праворуч. Вода на моделі друкується нижче рівня землі — на готовому брелоку вона помітна як заглиблення.",
      "An oval model with HOME and water on the right. Water is printed below ground level, so on the finished keychain it shows as a recess.",
    ],
  },
  {
    slug: "render-brelok-z-ozerom", src: "/showcase/keychain-6.webp", w: 1200, h: 920, kind: "keychain", source: "render",
    t: ["Брелок з озером і приватною забудовою — модель", "Keychain with a lake and houses — model"],
    a: ["Рендер брелока з картою: озеро в кутку, вигнута дорога і приватні будинки", "Keychain render with a map: a lake in the corner, a curved road and detached houses"],
    d: [
      "Модель для приміського району: озеро, вигнута дорога і невеликі приватні будинки. Напис крупними літерами займає вільну зелену ділянку.",
      "A model for a suburban area: a lake, a curved road and small detached houses. Large lettering fills the open green area.",
    ],
  },
  {
    slug: "render-brelok-zi-stavkom", src: "/showcase/keychain-7.webp", w: 1200, h: 920, kind: "keychain", source: "render",
    t: ["Брелок зі ставком у парку — модель", "Keychain with a park pond — model"],
    a: ["Рендер брелока з картою: ставок у парку, вулиця і ряд будинків", "Keychain render with a map: a pond in a park, a street and a row of buildings"],
    d: [
      "Ставок у парку та ряд будинків уздовж вулиці. Якщо поруч з домом є вода чи парк — вони стають головною прикметою брелока.",
      "A pond in a park and a row of buildings along the street. If there's water or a park near your home, it becomes the keychain's main landmark.",
    ],
  },
  {
    slug: "render-brelok-perekhrestya", src: "/showcase/keychain-8.webp", w: 1200, h: 920, kind: "keychain", source: "render",
    t: ["Брелок з перехрестям проспектів — модель", "Keychain with a major intersection — model"],
    a: ["Рендер брелока з картою великого перехрестя, будинків і скверів", "Keychain render showing a large intersection, buildings and small squares"],
    d: [
      "Велике перехрестя з розвилками і сквером — гарний вибір, якщо дім стоїть на впізнаваному проспекті. Широкі дороги модель робить помітно ширшими за провулки.",
      "A big intersection with forks and a square — a good pick if home is on a well-known avenue. Main roads are drawn noticeably wider than side lanes.",
    ],
  },
  {
    slug: "render-mapa-khmarochosy", src: "/showcase/map-1.webp", w: 1200, h: 920, kind: "map", source: "render",
    t: ["3D-мапа з висотною забудовою — модель", "3D map with high-rise buildings — model"],
    a: ["Рендер шестикутної 3D-мапи з висотками і вигнутими дорогами", "Render of a hexagonal 3D map with tower blocks and curved roads"],
    d: [
      "Модель нового житлового масиву з висотками. Будинки різної висоти, тому мапа виглядає як справжній макет міста.",
      "A model of a new residential estate with tower blocks. Buildings vary in height, so the map looks like a real architectural model.",
    ],
  },
  {
    slug: "render-mapa-temna-osnova", src: "/showcase/map-2.webp", w: 1200, h: 920, kind: "map", source: "render",
    t: ["3D-мапа району з парком по краю — модель", "3D district map with a park along the edge — model"],
    a: ["Рендер шестикутної 3D-мапи району з перехрестям і парком по краю", "Render of a hexagonal 3D district map with a crossroads and a park along the edge"],
    d: [
      "Модель на темному тлі рендера: сірі будинки, парк зеленим по краю. Так зручно оцінити силуети забудови ще до друку.",
      "The model against a dark render background: grey buildings and a green park along the edge — handy for judging the building silhouettes before printing.",
    ],
  },
  {
    slug: "render-mapa-pryvatnyi-sektor", src: "/showcase/map-3.webp", w: 1200, h: 920, kind: "map", source: "render",
    t: ["3D-мапа приватного сектору з рельєфом — модель", "3D map of a suburb with relief — model"],
    a: ["Рендер 3D-мапи приватного сектору: схил, дорога і окремі будинки", "Render of a 3D suburban map: a slope, a road and detached houses"],
    d: [
      "Приватний сектор на схилі: рідкі будинки, одна головна дорога і багато зелені. На таких ділянках рельєф робить мапу значно цікавішою.",
      "A suburb on a slope: scattered houses, one main road and plenty of green. On areas like this, terrain relief makes the map much more interesting.",
    ],
  },
  {
    slug: "render-mapa-dilovyi-tsentr", src: "/showcase/map-4.webp", w: 1200, h: 920, kind: "map", source: "render",
    t: ["3D-мапа ділового центру з хмарочосами — модель", "3D map of a business district — model"],
    a: ["Рендер шестикутної 3D-мапи ділового центру з хмарочосами і широкими проспектами", "Render of a hexagonal 3D business district map with skyscrapers and wide avenues"],
    d: [
      "Діловий центр: хмарочоси, широкі проспекти і щільна забудова. На мапі 15 см такі райони виглядають найефектніше.",
      "A business district: skyscrapers, wide avenues and dense blocks. On a 15 cm map, areas like this look the most striking.",
    ],
  },
  {
    slug: "render-mapa-sadyba-z-kilcem", src: "/showcase/map-5.webp", w: 1200, h: 920, kind: "map", source: "render",
    t: ["3D-мапа котеджного містечка — модель", "3D map of a cottage neighbourhood — model"],
    a: ["Рендер 3D-мапи котеджного містечка з кільцевою дорогою і вигнутими вулицями", "Render of a 3D cottage-town map with a ring road and curving streets"],
    d: [
      "Котеджне містечко з кільцевою дорогою в центрі. Невисокі будинки рівномірно заповнюють ділянку — хороший варіант для подарунка на новосілля в приватному будинку.",
      "A cottage neighbourhood with a ring road in the middle. Low houses fill the area evenly — a good housewarming gift for a private house.",
    ],
  },
  {
    slug: "render-mapa-tsentr-kvartaly", src: "/showcase/map-6.webp", w: 1200, h: 920, kind: "map", source: "render",
    t: ["3D-мапа історичного центру з кварталами — модель", "3D map of a historic centre — model"],
    a: ["Рендер шестикутної 3D-мапи історичного центру: квартали з дворами і вузькі вулиці", "Render of a hexagonal 3D historic-centre map: blocks with courtyards and narrow streets"],
    d: [
      "Історичний центр із замкненими кварталами і дворами-колодязями. Вузькі вулиці конструктор автоматично розширює до мінімальної друкованої ширини, щоб вони не зникли.",
      "A historic centre with closed blocks and well-like courtyards. The builder automatically widens narrow streets to the minimum printable width so they don't disappear.",
    ],
  },
  {
    slug: "render-mapa-temna-shchilna", src: "/showcase/map-7.webp", w: 1200, h: 920, kind: "map", source: "render",
    t: ["Щільна забудова з діагональною вулицею — модель", "Dense blocks with a diagonal street — model"],
    a: ["Рендер шестикутної 3D-мапи з щільною забудовою і діагональною вулицею", "Render of a hexagonal 3D map with dense buildings and a diagonal street"],
    d: [
      "Щільний район з діагональною вулицею на темному тлі рендера. Діагональ, що розтинає квартали, — прикмета, за якою район легко впізнати.",
      "A dense district with a diagonal street against a dark render background. A diagonal cutting through the blocks is the kind of feature that makes a district easy to recognise.",
    ],
  },
  {
    slug: "render-mapa-stare-misto", src: "/showcase/map-8.webp", w: 1200, h: 920, kind: "map", source: "render",
    t: ["3D-мапа старого міста — модель", "3D map of an old town — model"],
    a: ["Рендер шестикутної 3D-мапи старого міста з площами і нерегулярними вулицями", "Render of a hexagonal 3D old-town map with squares and irregular streets"],
    d: [
      "Старе місто з площами і нерегулярною сіткою вулиць. Площі друкуються окремим рівнем, тому їх видно як відкриті простори між будинками.",
      "An old town with squares and an irregular street grid. Squares are printed on their own level, so they show as open spaces between buildings.",
    ],
  },
  {
    slug: "render-mapa-kvartaly-z-dvoramy", src: "/showcase/map-9.webp", w: 1200, h: 920, kind: "map", source: "render",
    t: ["3D-мапа кварталів з дворами і парком — модель", "3D map of blocks with courtyards and a park — model"],
    a: ["Рендер 3D-мапи кварталів з дворами, парком ліворуч і ставком", "Render of a 3D map of blocks with courtyards, a park on the left and a pond"],
    d: [
      "Квартали з дворами, великий парк зліва і маленький ставок. Центральні ділянки з парком — вдалий вибір для настільної мапи.",
      "Blocks with courtyards, a large park on the left and a small pond. Central areas with a park make a good desk map.",
    ],
  },
  {
    slug: "render-mapa-z-richkoyu", src: "/showcase/map-10.webp", w: 1200, h: 920, kind: "map", source: "render",
    t: ["3D-мапа з річкою і стадіоном — модель", "3D map with a river and a stadium — model"],
    a: ["Рендер шестикутної 3D-мапи з вигнутою річкою, стадіоном і вулицями", "Render of a hexagonal 3D map with a winding river, a stadium and streets"],
    d: [
      "Ділянка з вигнутою річкою і стадіоном. Великі об'єкти — стадіони, вокзали, парки — роблять мапу впізнаваною з першого погляду.",
      "An area with a winding river and a stadium. Large landmarks — stadiums, stations, parks — make a map recognisable at first glance.",
    ],
  },
  {
    slug: "render-mapa-priama-sitka", src: "/showcase/map-11.webp", w: 1200, h: 920, kind: "map", source: "render",
    t: ["3D-мапа з прямою сіткою вулиць — модель", "3D map with a straight street grid — model"],
    a: ["Рендер 3D-мапи з прямокутною сіткою вулиць, невисокими будинками і висоткою", "Render of a 3D map with a rectangular street grid, low houses and one tower"],
    d: [
      "Пряма сітка вулиць, як у багатьох планованих районах: рівні квартали з невисокими будинками і одна висотка на краю.",
      "A straight street grid like in many planned districts: even blocks of low houses and a single tower at the edge.",
    ],
  },
];

export const GALLERY_ITEMS: GalleryItem[] = RAW.map(({ t, a, d, ...rest }) => ({
  ...rest,
  title: { uk: t[0], en: t[1] },
  alt: { uk: a[0], en: a[1] },
  desc: { uk: d[0], en: d[1] },
}));

export const GALLERY_BY_SLUG: Record<string, GalleryItem> = Object.fromEntries(
  GALLERY_ITEMS.map((g) => [g.slug, g]),
);

/** Локалі, для яких генеруються /foto-сторінки. */
export const GALLERY_LOCALES: readonly GalleryLocale[] = ["uk", "en"];

/** Схожі фото: спершу того ж типу, далі решта — детерміновано (SSG-стабільно). */
export function relatedGallery(slug: string, n = 6): GalleryItem[] {
  const self = GALLERY_BY_SLUG[slug];
  if (!self) return [];
  const idx = GALLERY_ITEMS.indexOf(self);
  const rot = [...GALLERY_ITEMS.slice(idx + 1), ...GALLERY_ITEMS.slice(0, idx)];
  const same = rot.filter((g) => g.kind === self.kind || (self.kind === "heart" && g.kind === "keychain"));
  const other = rot.filter((g) => !same.includes(g));
  return [...same, ...other].slice(0, n);
}
