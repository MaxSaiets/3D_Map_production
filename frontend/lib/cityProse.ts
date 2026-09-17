import type { AppLocale } from "@/i18n/routing";
import type { CityFacts } from "@/lib/cityFacts";

/**
 * УНІКАЛЬНИЙ ТЕКСТ ДЛЯ КОЖНОЇ СТОРІНКИ МІСТА (29.07.2026).
 *
 * ПРОБЛЕМА, яку лікуємо: виміряно 78-79% збігу тексту між сторінками міст
 * (454 слова, з них ~32 унікальні) → Google дедуплікує, сторінки лишаються
 * «Crawled/Discovered — currently not indexed».
 *
 * ЧОМУ НЕ РАНТАЙМ-ВИКЛИК LLM: Google із березня 2024 має політику
 * «scaled content abuse» — масова автогенерація сторінок без доданої цінності
 * карається незалежно від того, чим згенеровано. Плюс 64 міста × 6 мов = 384
 * виклики на кожен білд (вартість, латентність, недетермінованість, потреба
 * в ключі). Тому текст будується ДЕТЕРМІНОВАНО з РЕАЛЬНИХ даних міста:
 *
 *   1) обчислювані інсайти, яких немає в жодного іншого міста:
 *      щільність населення, вік, співвідношення площі, «читабельність» рельєфу;
 *   2) 6 різних СТРУКТУР абзацу — вибір за хешем slug (стабільний між білдами);
 *   3) підстановка власних назв (річка/регіон/орієнтир) у різні синтаксичні
 *      позиції, щоб збігалися не речення, а лише термінологія.
 *
 * Результат: кожна сторінка має власний набір чисел, формулювань і акцентів —
 * контент унікальний за суттю, а не переспам синонімами.
 */

export interface CityProseInput {
  slug: string;
  name: string;
  facts: CityFacts;
  locale: AppLocale;
}

/** Стабільний хеш slug → індекс варіанта (однаковий між білдами і локалями). */
function pick(slug: string, variants: number, salt = 0): number {
  let h = 2166136261 ^ salt;
  for (let i = 0; i < slug.length; i++) {
    h ^= slug.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h) % variants;
}

const CURRENT_YEAR = 2026;

type Derived = {
  density: number;        // осіб/км²
  age: number;            // років від заснування/першої згадки
  densityBand: 0 | 1 | 2; // 0 просторе, 1 середнє, 2 щільне
  sizeBand: 0 | 1 | 2;    // 0 невелике, 1 середнє, 2 велике
};

function derive(f: CityFacts): Derived {
  const density = Math.round(f.population / Math.max(f.area_km2, 1));
  const age = CURRENT_YEAR - f.founded;
  const densityBand = density > 4000 ? 2 : density > 1800 ? 1 : 0;
  const sizeBand = f.population > 1000000 ? 2 : f.population > 400000 ? 1 : 0;
  return { density, age, densityBand, sizeBand };
}

/* ── Українська ──────────────────────────────────────────────────────────── */
function uk(i: CityProseInput, d: Derived, nf: Intl.NumberFormat): string[] {
  const { name, facts: f } = i;
  const river = f.river.uk;
  const lm = f.landmark.uk;
  const region = f.oblast.uk;
  const v = pick(i.slug, 6);

  const densityPhrase = [
    `забудова просторова — ${nf.format(d.density)} осіб на км², тож у моделі добре видно окремі квартали й розриви між ними`,
    `щільність ${nf.format(d.density)} осіб на км² дає рівний ритм кварталів — мапа читається як цілісний масив`,
    `щільні ${nf.format(d.density)} осіб на км² перетворюють центр на суцільний рельєф дахів, де вулиці читаються як прорізи`,
  ][d.densityBand];

  const scalePhrase = [
    `За площею ${nf.format(Math.round(f.area_km2))} км² місто вміщується в одну модель без втрати деталей.`,
    `Площа ${nf.format(Math.round(f.area_km2))} км² означає, що для однієї моделі зазвичай беруть район, а не місто цілком.`,
    `На ${nf.format(Math.round(f.area_km2))} км² повне місто в одну плитку не влізе — тому найкраще працює або центр, або панно з кількох частин.`,
  ][d.sizeBand];

  const agePhrase =
    d.age > 900
      ? `${name} має за плечима понад ${Math.floor(d.age / 100) * 100} років — історичне ядро й пізніші райони мають помітно різну геометрію вулиць, і це видно на моделі.`
      : d.age > 400
        ? `Місту близько ${Math.round(d.age / 50) * 50} років, тож регулярні квартали пізніших епох сусідять зі старим нерегулярним центром.`
        : `Порівняно молода забудова (${d.age} років) дає впорядковану сітку вулиць — модель виходить графічною й читабельною.`;

  const A = `Головна водойма — ${river}, а архітектурна візитівка — ${lm}. Саме ці два орієнтири роблять 3D-мапу впізнаваною з першого погляду: русло дає природну діагональ, а домінанта — точку, за яку чіпляється око.`;
  const B = `У тривимірі ${name} впізнають за двома речами: лінією, яку прокладає ${river}, і масою ${lm} у щільній забудові. Все інше — контекст, що тримає ці орієнтири.`;
  const C = `${lm} і ${river} — те, що люди шукають на моделі першим. Решта кварталів працює як фон, який задає масштаб цим орієнтирам.`;

  const orientation = [A, B, C][pick(i.slug, 3, 7)];

  const closing = [
    `Регіон — ${region}. Для друку зазвичай беруть ділянку 1–3 км² навколо центру: у такому масштабі будинки лишаються окремими об'ємами, а не зливаються в суцільну плиту.`,
    `Адміністративно це ${region}. Оптимальна ділянка для однієї моделі — 1–3 км²: далі дрібні вулиці перестають читатись на друку 0.4 мм соплом.`,
    `${region} — саме тут проходять межі, за якими будується модель. Ділянка 1–3 км² дає найкращий баланс між охопленням і деталізацією.`,
  ][pick(i.slug, 3, 13)];

  // 6 різних порядків подачі — структура сторінки теж відрізняється
  const order: string[][] = [
    [orientation, `${scalePhrase} Тут ${densityPhrase}.`, agePhrase, closing],
    [`${agePhrase}`, orientation, `${scalePhrase} При цьому ${densityPhrase}.`, closing],
    [`${scalePhrase}`, orientation, `Крім того, ${densityPhrase}.`, `${agePhrase} ${closing}`],
    [orientation, agePhrase, `${scalePhrase} Додатково: ${densityPhrase}.`, closing],
    [`Тут ${densityPhrase}. ${scalePhrase}`, orientation, agePhrase, closing],
    [`${agePhrase}`, `${scalePhrase} ${orientation}`, `Варто врахувати: ${densityPhrase}.`, closing],
  ];
  return order[v];
}

/* ── Англійська (база для en + фолбек інших локалей) ─────────────────────── */
function en(i: CityProseInput, d: Derived, nf: Intl.NumberFormat): string[] {
  const { name, facts: f } = i;
  const river = f.river.latin;
  const lm = f.landmark.latin;
  const region = f.oblast.latin;
  const v = pick(i.slug, 6);

  const densityPhrase = [
    `the layout is spacious — ${nf.format(d.density)} people per km², so individual blocks and the gaps between them stay legible in the model`,
    `a density of ${nf.format(d.density)} people per km² gives an even block rhythm, and the map reads as one continuous fabric`,
    `at ${nf.format(d.density)} people per km² the centre becomes a solid relief of rooftops, with streets reading as cuts through it`,
  ][d.densityBand];

  const scalePhrase = [
    `At ${nf.format(Math.round(f.area_km2))} km² the city fits into a single model without losing detail.`,
    `With ${nf.format(Math.round(f.area_km2))} km² of area, one model usually covers a district rather than the whole city.`,
    `Spanning ${nf.format(Math.round(f.area_km2))} km², the full city will not fit one tile — the centre alone, or a multi-tile panel, works best.`,
  ][d.sizeBand];

  const agePhrase =
    d.age > 900
      ? `${name} carries more than ${Math.floor(d.age / 100) * 100} years of history — the old core and later districts have visibly different street geometry, and the model shows it.`
      : d.age > 400
        ? `The city is roughly ${Math.round(d.age / 50) * 50} years old, so regular later blocks sit next to an irregular historic centre.`
        : `Relatively young fabric (${d.age} years) means an ordered street grid — the model comes out graphic and easy to read.`;

  const A = `The defining water feature is the ${river}, and the architectural landmark is ${lm}. These two anchors make the 3D map recognisable at a glance: the channel gives a natural diagonal, the landmark gives the eye something to hold.`;
  const B = `In three dimensions ${name} is recognised by two things: the line drawn by the ${river} and the mass of ${lm} within dense blocks. Everything else is context holding those anchors.`;
  const C = `${lm} and the ${river} are what people look for first on the model. The remaining blocks act as a backdrop that sets the scale.`;

  const orientation = [A, B, C][pick(i.slug, 3, 7)];

  const closing = [
    `The region is ${region}. A print usually covers 1–3 km² around the centre: at that scale buildings stay separate volumes instead of merging into a slab.`,
    `Administratively this is ${region}. The sweet spot for a single model is 1–3 km² — beyond that, narrow streets stop resolving at a 0.4 mm nozzle.`,
    `${region} sets the boundaries the model is built from. A 1–3 km² area balances coverage against detail best.`,
  ][pick(i.slug, 3, 13)];

  const order: string[][] = [
    [orientation, `${scalePhrase} Here ${densityPhrase}.`, agePhrase, closing],
    [`${agePhrase}`, orientation, `${scalePhrase} Meanwhile ${densityPhrase}.`, closing],
    [`${scalePhrase}`, orientation, `On top of that, ${densityPhrase}.`, `${agePhrase} ${closing}`],
    [orientation, agePhrase, `${scalePhrase} Also worth noting: ${densityPhrase}.`, closing],
    [`Here ${densityPhrase}. ${scalePhrase}`, orientation, agePhrase, closing],
    [`${agePhrase}`, `${scalePhrase} ${orientation}`, `Worth factoring in: ${densityPhrase}.`, closing],
  ];
  return order[v];
}

/* ── Deutsch ─────────────────────────────────────────────────────────────── */
function de(i: CityProseInput, d: Derived, nf: Intl.NumberFormat): string[] {
  const { name, facts: f } = i;
  const river = f.river.latin;
  const lm = f.landmark.latin;
  const region = f.oblast.latin;
  const v = pick(i.slug, 6);

  const densityPhrase = [
    `die Bebauung ist weitläufig — ${nf.format(d.density)} Einwohner pro km², sodass einzelne Blöcke und die Lücken dazwischen im Modell gut lesbar bleiben`,
    `eine Dichte von ${nf.format(d.density)} Einwohnern pro km² ergibt einen gleichmäßigen Blockrhythmus, und die Karte liest sich als zusammenhängendes Gefüge`,
    `bei ${nf.format(d.density)} Einwohnern pro km² wird das Zentrum zu einem durchgehenden Relief aus Dächern, in dem die Straßen als Einschnitte lesbar sind`,
  ][d.densityBand];

  const scalePhrase = [
    `Mit ${nf.format(Math.round(f.area_km2))} km² passt die Stadt ohne Detailverlust in ein einziges Modell.`,
    `Bei ${nf.format(Math.round(f.area_km2))} km² Fläche deckt ein Modell meist einen Stadtteil ab, nicht die ganze Stadt.`,
    `Auf ${nf.format(Math.round(f.area_km2))} km² passt die ganze Stadt nicht auf eine Kachel — am besten funktioniert das Zentrum allein oder ein Panel aus mehreren Teilen.`,
  ][d.sizeBand];

  const agePhrase =
    d.age > 900
      ? `${name} trägt mehr als ${Math.floor(d.age / 100) * 100} Jahre Geschichte — der alte Kern und die späteren Viertel haben sichtbar unterschiedliche Straßengeometrien, und das Modell zeigt es.`
      : d.age > 400
        ? `Die Stadt ist rund ${Math.round(d.age / 50) * 50} Jahre alt, sodass regelmäßige spätere Blöcke neben einem unregelmäßigen historischen Zentrum liegen.`
        : `Ein vergleichsweise junges Stadtgefüge (${d.age} Jahre) bedeutet ein geordnetes Straßenraster — das Modell wirkt grafisch und ist leicht zu lesen.`;

  const A = `Das prägende Gewässer ist ${river}, das architektonische Wahrzeichen ${lm}. Diese beiden Anker machen die 3D-Karte auf den ersten Blick erkennbar: der Flusslauf gibt eine natürliche Diagonale, das Wahrzeichen einen Haltepunkt fürs Auge.`;
  const B = `In drei Dimensionen erkennt man ${name} an zwei Dingen: an der Linie, die ${river} zieht, und an der Masse von ${lm} zwischen dichten Blöcken. Alles andere ist Kontext, der diese Anker trägt.`;
  const C = `${lm} und ${river} sind das, was man auf dem Modell zuerst sucht. Die übrigen Blöcke wirken als Kulisse, die den Maßstab vorgibt.`;

  const orientation = [A, B, C][pick(i.slug, 3, 7)];

  const closing = [
    `Die Region ist ${region}. Ein Druck deckt meist 1–3 km² rund um das Zentrum ab: in diesem Maßstab bleiben Gebäude eigenständige Volumen, statt zu einer Platte zu verschmelzen.`,
    `Verwaltungstechnisch ist das ${region}. Der ideale Bereich für ein einzelnes Modell liegt bei 1–3 km² — darüber hinaus lösen sich schmale Straßen bei einer 0,4-mm-Düse nicht mehr auf.`,
    `${region} setzt die Grenzen, aus denen das Modell gebaut wird. Eine Fläche von 1–3 km² bringt Abdeckung und Detail am besten ins Gleichgewicht.`,
  ][pick(i.slug, 3, 13)];

  const order: string[][] = [
    [orientation, `${scalePhrase} Hier ist ${densityPhrase}.`, agePhrase, closing],
    [`${agePhrase}`, orientation, `${scalePhrase} Dabei ist ${densityPhrase}.`, closing],
    [`${scalePhrase}`, orientation, `Hinzu kommt: ${densityPhrase}.`, `${agePhrase} ${closing}`],
    [orientation, agePhrase, `${scalePhrase} Ebenfalls erwähnenswert: ${densityPhrase}.`, closing],
    [`Hier ist ${densityPhrase}. ${scalePhrase}`, orientation, agePhrase, closing],
    [`${agePhrase}`, `${scalePhrase} ${orientation}`, `Zu bedenken: ${densityPhrase}.`, closing],
  ];
  return order[v];
}

/* ── Español ─────────────────────────────────────────────────────────────── */
function es(i: CityProseInput, d: Derived, nf: Intl.NumberFormat): string[] {
  const { name, facts: f } = i;
  const river = f.river.latin;
  const lm = f.landmark.latin;
  const region = f.oblast.latin;
  const v = pick(i.slug, 6);

  const densityPhrase = [
    `el trazado es espacioso — ${nf.format(d.density)} habitantes por km², así que las manzanas y los huecos entre ellas se leen bien en el modelo`,
    `una densidad de ${nf.format(d.density)} habitantes por km² da un ritmo de manzanas uniforme, y el mapa se lee como un tejido continuo`,
    `con ${nf.format(d.density)} habitantes por km² el centro se convierte en un relieve continuo de tejados, donde las calles se leen como cortes`,
  ][d.densityBand];

  const scalePhrase = [
    `Con ${nf.format(Math.round(f.area_km2))} km², la ciudad cabe en un solo modelo sin perder detalle.`,
    `Con ${nf.format(Math.round(f.area_km2))} km² de superficie, un modelo suele cubrir un distrito y no la ciudad entera.`,
    `Con ${nf.format(Math.round(f.area_km2))} km², la ciudad completa no cabe en una pieza — funciona mejor solo el centro o un panel de varias piezas.`,
  ][d.sizeBand];

  const agePhrase =
    d.age > 900
      ? `${name} acumula más de ${Math.floor(d.age / 100) * 100} años de historia — el núcleo antiguo y los barrios posteriores tienen una geometría de calles visiblemente distinta, y el modelo lo muestra.`
      : d.age > 400
        ? `La ciudad tiene unos ${Math.round(d.age / 50) * 50} años, así que las manzanas regulares de épocas posteriores conviven con un centro histórico irregular.`
        : `Un tejido relativamente joven (${d.age} años) significa una retícula de calles ordenada — el modelo sale gráfico y fácil de leer.`;

  const A = `El elemento de agua que define la ciudad es ${river}, y el hito arquitectónico, ${lm}. Estos dos anclajes hacen que el mapa 3D se reconozca al instante: el cauce da una diagonal natural y el hito, un punto donde fijar la vista.`;
  const B = `En tres dimensiones, ${name} se reconoce por dos cosas: la línea que traza ${river} y la masa de ${lm} entre manzanas densas. Todo lo demás es contexto que sostiene esos anclajes.`;
  const C = `${lm} y ${river} son lo primero que la gente busca en el modelo. El resto de manzanas actúa como telón de fondo que marca la escala.`;

  const orientation = [A, B, C][pick(i.slug, 3, 7)];

  const closing = [
    `La región es ${region}. Una impresión suele cubrir 1–3 km² alrededor del centro: a esa escala los edificios siguen siendo volúmenes separados en vez de fundirse en una placa.`,
    `Administrativamente es ${region}. El punto óptimo para un solo modelo son 1–3 km² — más allá, las calles estrechas dejan de resolverse con una boquilla de 0,4 mm.`,
    `${region} marca los límites con los que se construye el modelo. Un área de 1–3 km² equilibra mejor cobertura y detalle.`,
  ][pick(i.slug, 3, 13)];

  const order: string[][] = [
    [orientation, `${scalePhrase} Aquí ${densityPhrase}.`, agePhrase, closing],
    [`${agePhrase}`, orientation, `${scalePhrase} Además, ${densityPhrase}.`, closing],
    [`${scalePhrase}`, orientation, `A esto se suma que ${densityPhrase}.`, `${agePhrase} ${closing}`],
    [orientation, agePhrase, `${scalePhrase} Cabe señalar: ${densityPhrase}.`, closing],
    [`Aquí ${densityPhrase}. ${scalePhrase}`, orientation, agePhrase, closing],
    [`${agePhrase}`, `${scalePhrase} ${orientation}`, `Conviene tener en cuenta: ${densityPhrase}.`, closing],
  ];
  return order[v];
}

/* ── Français ────────────────────────────────────────────────────────────── */
function fr(i: CityProseInput, d: Derived, nf: Intl.NumberFormat): string[] {
  const { name, facts: f } = i;
  const river = f.river.latin;
  const lm = f.landmark.latin;
  const region = f.oblast.latin;
  const v = pick(i.slug, 6);

  const densityPhrase = [
    `le tissu urbain est aéré — ${nf.format(d.density)} habitants au km², si bien que les îlots et les vides entre eux restent lisibles sur le modèle`,
    `une densité de ${nf.format(d.density)} habitants au km² donne un rythme d'îlots régulier, et la carte se lit comme un tissu continu`,
    `à ${nf.format(d.density)} habitants au km², le centre devient un relief continu de toits, où les rues se lisent comme des entailles`,
  ][d.densityBand];

  const scalePhrase = [
    `Avec ${nf.format(Math.round(f.area_km2))} km², la ville tient dans un seul modèle sans perte de détail.`,
    `Avec ${nf.format(Math.round(f.area_km2))} km² de superficie, un modèle couvre en général un quartier plutôt que toute la ville.`,
    `Sur ${nf.format(Math.round(f.area_km2))} km², la ville entière ne tient pas sur une seule pièce — le centre seul, ou un panneau en plusieurs pièces, fonctionne le mieux.`,
  ][d.sizeBand];

  const agePhrase =
    d.age > 900
      ? `${name} porte plus de ${Math.floor(d.age / 100) * 100} ans d'histoire — le noyau ancien et les quartiers plus récents ont une géométrie de rues visiblement différente, et le modèle le montre.`
      : d.age > 400
        ? `La ville a environ ${Math.round(d.age / 50) * 50} ans : des îlots réguliers d'époques plus récentes côtoient un centre historique irrégulier.`
        : `Un tissu relativement jeune (${d.age} ans) signifie une trame de rues ordonnée — le modèle en ressort graphique et facile à lire.`;

  const A = `L'élément d'eau qui structure la ville est ${river}, et le repère architectural, ${lm}. Ces deux ancrages rendent la carte 3D reconnaissable au premier coup d'œil : le cours d'eau donne une diagonale naturelle, le monument donne à l'œil un point d'accroche.`;
  const B = `En trois dimensions, ${name} se reconnaît à deux choses : la ligne tracée par ${river} et la masse de ${lm} au milieu d'îlots denses. Tout le reste est un contexte qui porte ces repères.`;
  const C = `${lm} et ${river} sont ce que l'on cherche en premier sur le modèle. Les autres îlots servent de toile de fond qui donne l'échelle.`;

  const orientation = [A, B, C][pick(i.slug, 3, 7)];

  const closing = [
    `La région est ${region}. Une impression couvre généralement 1–3 km² autour du centre : à cette échelle, les bâtiments restent des volumes distincts au lieu de se fondre en une dalle.`,
    `Administrativement, c'est ${region}. La zone idéale pour un seul modèle est de 1–3 km² — au-delà, les rues étroites cessent d'être résolues avec une buse de 0,4 mm.`,
    `${region} fixe les limites à partir desquelles le modèle est construit. Une zone de 1–3 km² équilibre le mieux couverture et détail.`,
  ][pick(i.slug, 3, 13)];

  const order: string[][] = [
    [orientation, `${scalePhrase} Ici, ${densityPhrase}.`, agePhrase, closing],
    [`${agePhrase}`, orientation, `${scalePhrase} Par ailleurs, ${densityPhrase}.`, closing],
    [`${scalePhrase}`, orientation, `À cela s'ajoute que ${densityPhrase}.`, `${agePhrase} ${closing}`],
    [orientation, agePhrase, `${scalePhrase} À noter aussi : ${densityPhrase}.`, closing],
    [`Ici, ${densityPhrase}. ${scalePhrase}`, orientation, agePhrase, closing],
    [`${agePhrase}`, `${scalePhrase} ${orientation}`, `À prendre en compte : ${densityPhrase}.`, closing],
  ];
  return order[v];
}

/* ── Polski ──────────────────────────────────────────────────────────────── */
function pl(i: CityProseInput, d: Derived, nf: Intl.NumberFormat): string[] {
  const { name, facts: f } = i;
  const river = f.river.latin;
  const lm = f.landmark.latin;
  const region = f.oblast.latin;
  const v = pick(i.slug, 6);

  const densityPhrase = [
    `zabudowa jest przestronna — ${nf.format(d.density)} osób na km², więc pojedyncze kwartały i przerwy między nimi pozostają w modelu czytelne`,
    `gęstość ${nf.format(d.density)} osób na km² daje równy rytm kwartałów, a mapa czyta się jak jednolita tkanka`,
    `przy ${nf.format(d.density)} osobach na km² centrum staje się ciągłą rzeźbą dachów, w której ulice czyta się jak nacięcia`,
  ][d.densityBand];

  const scalePhrase = [
    `Przy ${nf.format(Math.round(f.area_km2))} km² miasto mieści się w jednym modelu bez utraty szczegółów.`,
    `Przy ${nf.format(Math.round(f.area_km2))} km² powierzchni jeden model zwykle obejmuje dzielnicę, a nie całe miasto.`,
    `Na ${nf.format(Math.round(f.area_km2))} km² całe miasto nie zmieści się na jednej płytce — najlepiej sprawdza się samo centrum albo panel z kilku części.`,
  ][d.sizeBand];

  const agePhrase =
    d.age > 900
      ? `${name} ma za sobą ponad ${Math.floor(d.age / 100) * 100} lat historii — stare jądro i późniejsze dzielnice mają wyraźnie inną geometrię ulic, i model to pokazuje.`
      : d.age > 400
        ? `Miasto ma około ${Math.round(d.age / 50) * 50} lat, więc regularne kwartały późniejszych epok sąsiadują z nieregularnym historycznym centrum.`
        : `Stosunkowo młoda tkanka (${d.age} lat) oznacza uporządkowaną siatkę ulic — model wychodzi graficzny i łatwy do odczytania.`;

  const A = `Główny akwen to ${river}, a architektoniczna wizytówka — ${lm}. Te dwa punkty odniesienia sprawiają, że mapę 3D rozpoznaje się od pierwszego spojrzenia: koryto daje naturalną przekątną, a dominanta — punkt, o który zaczepia się oko.`;
  const B = `W trzech wymiarach ${name} rozpoznaje się po dwóch rzeczach: linii, którą wyznacza ${river}, i bryle ${lm} wśród gęstej zabudowy. Cała reszta to kontekst, który trzyma te punkty odniesienia.`;
  const C = `${lm} i ${river} to to, czego ludzie szukają na modelu najpierw. Pozostałe kwartały działają jak tło, które nadaje skalę.`;

  const orientation = [A, B, C][pick(i.slug, 3, 7)];

  const closing = [
    `Region to ${region}. Do druku zwykle bierze się obszar 1–3 km² wokół centrum: w tej skali budynki pozostają osobnymi bryłami, zamiast zlewać się w jedną płytę.`,
    `Administracyjnie to ${region}. Optymalny obszar dla jednego modelu to 1–3 km² — powyżej wąskie ulice przestają być czytelne przy dyszy 0,4 mm.`,
    `${region} wyznacza granice, z których budowany jest model. Obszar 1–3 km² najlepiej równoważy zasięg i szczegółowość.`,
  ][pick(i.slug, 3, 13)];

  const order: string[][] = [
    [orientation, `${scalePhrase} Tutaj ${densityPhrase}.`, agePhrase, closing],
    [`${agePhrase}`, orientation, `${scalePhrase} Przy tym ${densityPhrase}.`, closing],
    [`${scalePhrase}`, orientation, `Do tego ${densityPhrase}.`, `${agePhrase} ${closing}`],
    [orientation, agePhrase, `${scalePhrase} Warto też zauważyć: ${densityPhrase}.`, closing],
    [`Tutaj ${densityPhrase}. ${scalePhrase}`, orientation, agePhrase, closing],
    [`${agePhrase}`, `${scalePhrase} ${orientation}`, `Warto uwzględnić: ${densityPhrase}.`, closing],
  ];
  return order[v];
}

/**
 * Абзаци унікального опису міста мовою сторінки. 16.09.2026: раніше de/es/fr/pl
 * отримували англійський текст посеред локалізованої сторінки (віденець із
 * Google бачив «Spanning 415 km²…» під німецьким вступом). Назви річки/памʼятки
 * лишаються латинкою (у даних є лише uk + latin).
 */
export function cityProse(input: CityProseInput): string[] {
  const d = derive(input.facts);
  const nf = new Intl.NumberFormat(input.locale === "uk" ? "uk-UA" : input.locale);
  switch (input.locale) {
    case "uk": return uk(input, d, nf);
    case "de": return de(input, d, nf);
    case "es": return es(input, d, nf);
    case "fr": return fr(input, d, nf);
    case "pl": return pl(input, d, nf);
    default: return en(input, d, nf);
  }
}

/** Обчислені показники для видимого блоку фактів (унікальні числа на сторінку). */
export function cityDerivedFacts(facts: CityFacts) {
  return derive(facts);
}
