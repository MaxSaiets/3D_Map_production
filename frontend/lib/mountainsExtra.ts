import type { AppLocale } from "@/i18n/routing";

/**
 * 07.10.2026: GSC позначив /en/mountains як «Soft 404». Студія гір — ssr:false,
 * тож у серверному HTML лишались заголовок, порожня заглушка й «тестовий режим».
 * Цей текст дає пошуковику змістовну сторінку: для кого, як замовити, питання.
 * Лише факти, які вже є в самій студії (джерела висот, ободок, плитки, фігурки,
 * ціна друку індивідуальна).
 */
export type MountainsExtra = {
  h2who: string;
  who: { h3: string; p: string }[];
  h2how: string;
  how: string[];
  h2faq: string;
  faq: { q: string; a: string }[];
};

export const MOUNTAINS_EXTRA: Record<AppLocale, MountainsExtra> = {
  uk: {
    h2who: "Кому дарують 3D-модель гори",
    who: [
      { h3: "Тим, хто вже зійшов на вершину", p: "Говерла, Піп Іван чи Монблан на полиці нагадують про конкретний день, маршрут і людей поруч. На моделі видно хребет, яким ви йшли, і схил, де ставили намет." },
      { h3: "Альпіністам і туристам, які ще мріють", p: "Модель Евересту, Кіліманджаро чи Матергорну стає ціллю на столі. Фігурки альпіністів і хатинок показують масштаб і додають історії." },
      { h3: "Клубам, гідам і турфірмам", p: "Рельєф району походів допомагає пояснити маршрут групі, а друкована вершина стає подарунком учасникам або призом змагань." },
    ],
    h2how: "Як замовити модель гори",
    how: [
      "Оберіть вершину зі списку або точку на мапі й задайте розмір ділянки.",
      "Налаштуйте ободок, бокові стінки, фігурки альпіністів і хатинок, подивіться 3D-превʼю безкоштовно.",
      "Завантажте файл для друку на своєму принтері або напишіть нам: ціну друку й фарбування рахуємо індивідуально під розмір.",
    ],
    h2faq: "Питання про моделі гір",
    faq: [
      { q: "Звідки беруться висоти?", a: "Для Швейцарії з лідарної зйомки swissALTI3D з кроком 2 метри, для решти світу з глобальної моделі Copernicus GLO-30 з кроком 30 метрів. Рельєф справжній, а не намальований." },
      { q: "Чи можна надрукувати велику модель?", a: "Так. Якщо модель більша за стіл принтера, ми ділимо її на плитки, які складаються без шва в одну пластину." },
      { q: "Чи фігурки в масштабі?", a: "Ні, фігурки декоративні й більші, ніж були б у реальному масштабі, щоб їх було видно. Альпініст стає на найкрутішу стіну біля вершини, а хатинка на пологий схил." },
      { q: "Чи потрібні підтримки під час друку?", a: "Рельєф, бокові стінки, ободок і дно збираються в одне замкнене тіло, яке друкується без підтримок." },
    ],
  },
  en: {
    h2who: "Who a 3D mountain model is for",
    who: [
      { h3: "People who reached the summit", p: "Hoverla, Pip Ivan or Mont Blanc on a shelf recalls a specific day, route and people. The model shows the ridge you walked and the slope where you pitched the tent." },
      { h3: "Climbers and hikers who still dream", p: "A model of Everest, Kilimanjaro or the Matterhorn becomes a goal on the desk. Climber and hut figures show the scale and add a story." },
      { h3: "Clubs, guides and tour operators", p: "Relief of a hiking area helps explain the route to a group, and a printed summit makes a gift for participants or a race prize." },
    ],
    h2how: "How to order a mountain model",
    how: [
      "Pick a summit from the list or a point on the map and set the area size.",
      "Set the rim, side walls, climber and hut figures, and look at the 3D preview for free.",
      "Download the file to print on your own printer, or message us: print and painting price is calculated individually for the size.",
    ],
    h2faq: "Questions about mountain models",
    faq: [
      { q: "Where do the heights come from?", a: "For Switzerland from swissALTI3D lidar at 2 metres, for the rest of the world from the global Copernicus GLO-30 model at 30 metres. The terrain is real, not drawn." },
      { q: "Can a large model be printed?", a: "Yes. If the model is larger than the printer bed, we split it into tiles that fit together seamlessly into one plate." },
      { q: "Are the figures to scale?", a: "No, figures are decorative and larger than at true scale so you can see them. The climber goes on the steepest wall near the summit and the hut on a gentle slope." },
      { q: "Does it need supports?", a: "Terrain, side walls, rim and bottom form one watertight solid that prints without supports." },
    ],
  },
  de: {
    h2who: "Für wen ein 3D-Bergmodell ist",
    who: [
      { h3: "Für alle, die oben waren", p: "Hoverla, Pip Iwan oder Mont Blanc im Regal erinnern an einen bestimmten Tag, die Route und die Menschen dabei. Man sieht den Grat, den man gegangen ist." },
      { h3: "Für Bergsteiger mit Zielen", p: "Ein Modell von Everest, Kilimandscharo oder Matterhorn wird zum Ziel auf dem Schreibtisch. Figuren von Kletterern und Hütten zeigen den Maßstab." },
      { h3: "Für Vereine, Guides und Reiseveranstalter", p: "Das Relief eines Wandergebiets erklärt der Gruppe die Route, ein gedruckter Gipfel ist ein Geschenk oder Preis." },
    ],
    h2how: "So bestellen Sie ein Bergmodell",
    how: [
      "Gipfel aus der Liste oder Punkt auf der Karte wählen und Ausschnitt festlegen.",
      "Rand, Seitenwände und Figuren einstellen, kostenlose 3D-Vorschau ansehen.",
      "Datei für den eigenen Drucker laden oder uns schreiben: Druck und Bemalung kalkulieren wir individuell.",
    ],
    h2faq: "Fragen zu Bergmodellen",
    faq: [
      { q: "Woher kommen die Höhen?", a: "Für die Schweiz aus swissALTI3D-Lidar mit 2 m, weltweit aus Copernicus GLO-30 mit 30 m. Das Gelände ist echt." },
      { q: "Geht auch ein großes Modell?", a: "Ja. Ist das Modell größer als das Druckbett, teilen wir es in nahtlose Kacheln." },
      { q: "Sind die Figuren maßstabsgetreu?", a: "Nein, sie sind dekorativ und größer, damit man sie sieht. Der Kletterer steht an der steilsten Wand nahe dem Gipfel, die Hütte am flachen Hang." },
      { q: "Braucht der Druck Stützen?", a: "Gelände, Seitenwände, Rand und Boden bilden einen geschlossenen Körper, der ohne Stützen druckt." },
    ],
  },
  fr: {
    h2who: "À qui offrir une maquette 3D de montagne",
    who: [
      { h3: "À ceux qui ont atteint le sommet", p: "L'Hoverla, le Pip Ivan ou le Mont Blanc sur une étagère rappellent un jour, un itinéraire et des compagnons. On voit l'arête parcourue." },
      { h3: "Aux alpinistes qui rêvent encore", p: "Une maquette de l'Everest, du Kilimandjaro ou du Cervin devient un objectif sur le bureau. Les figurines donnent l'échelle." },
      { h3: "Aux clubs, guides et agences", p: "Le relief d'une zone de randonnée explique l'itinéraire au groupe, un sommet imprimé devient cadeau ou prix." },
    ],
    h2how: "Comment commander une maquette de montagne",
    how: [
      "Choisissez un sommet dans la liste ou un point sur la carte et la taille de la zone.",
      "Réglez le bord, les flancs et les figurines, regardez l'aperçu 3D gratuit.",
      "Téléchargez le fichier pour votre imprimante ou écrivez-nous : impression et peinture sont chiffrées sur mesure.",
    ],
    h2faq: "Questions sur les maquettes de montagne",
    faq: [
      { q: "D'où viennent les altitudes ?", a: "Pour la Suisse du lidar swissALTI3D à 2 m, ailleurs du modèle mondial Copernicus GLO-30 à 30 m. Le relief est réel." },
      { q: "Peut-on imprimer une grande maquette ?", a: "Oui. Si elle dépasse le plateau, nous la découpons en tuiles qui s'assemblent sans couture." },
      { q: "Les figurines sont-elles à l'échelle ?", a: "Non, elles sont décoratives et plus grandes pour être visibles. Le grimpeur est placé sur la paroi la plus raide, le refuge sur une pente douce." },
      { q: "Faut-il des supports ?", a: "Relief, flancs, bord et fond forment un seul volume fermé qui s'imprime sans supports." },
    ],
  },
  es: {
    h2who: "Para quién es una maqueta 3D de montaña",
    who: [
      { h3: "Para quien llegó a la cima", p: "La Hoverla, el Pip Iván o el Mont Blanc en una estantería recuerdan un día, una ruta y a la gente. Se ve la cresta que recorriste." },
      { h3: "Para montañeros que aún sueñan", p: "Una maqueta del Everest, el Kilimanjaro o el Cervino se convierte en una meta sobre el escritorio. Las figuras dan la escala." },
      { h3: "Para clubes, guías y agencias", p: "El relieve de una zona de senderismo explica la ruta al grupo, y una cima impresa es un regalo o premio." },
    ],
    h2how: "Cómo pedir una maqueta de montaña",
    how: [
      "Elige una cima de la lista o un punto en el mapa y el tamaño de la zona.",
      "Ajusta el borde, los laterales y las figuras, y mira la vista previa 3D gratis.",
      "Descarga el archivo para tu impresora o escríbenos: impresión y pintura se calculan a medida.",
    ],
    h2faq: "Preguntas sobre maquetas de montaña",
    faq: [
      { q: "¿De dónde salen las alturas?", a: "En Suiza del lidar swissALTI3D a 2 m, en el resto del mundo del modelo global Copernicus GLO-30 a 30 m. El relieve es real." },
      { q: "¿Se puede imprimir una maqueta grande?", a: "Sí. Si es mayor que la cama de la impresora, la dividimos en piezas que encajan sin costura." },
      { q: "¿Las figuras están a escala?", a: "No, son decorativas y más grandes para que se vean. El escalador va en la pared más empinada y el refugio en una ladera suave." },
      { q: "¿Necesita soportes?", a: "Relieve, laterales, borde y base forman un solo cuerpo cerrado que se imprime sin soportes." },
    ],
  },
  pl: {
    h2who: "Dla kogo jest model 3D góry",
    who: [
      { h3: "Dla tych, którzy weszli na szczyt", p: "Howerla, Pop Iwan czy Mont Blanc na półce przypominają konkretny dzień, trasę i ludzi. Widać grań, którą szliście." },
      { h3: "Dla wspinaczy, którzy jeszcze marzą", p: "Model Everestu, Kilimandżaro czy Matterhornu staje się celem na biurku. Figurki pokazują skalę." },
      { h3: "Dla klubów, przewodników i biur podróży", p: "Rzeźba terenu pomaga wyjaśnić trasę grupie, a wydrukowany szczyt to prezent lub nagroda." },
    ],
    h2how: "Jak zamówić model góry",
    how: [
      "Wybierz szczyt z listy lub punkt na mapie i wielkość obszaru.",
      "Ustaw obrzeże, boki i figurki, obejrzyj darmowy podgląd 3D.",
      "Pobierz plik do własnej drukarki albo napisz do nas: cenę druku i malowania liczymy indywidualnie.",
    ],
    h2faq: "Pytania o modele gór",
    faq: [
      { q: "Skąd są wysokości?", a: "Dla Szwajcarii z lidaru swissALTI3D co 2 m, dla reszty świata z globalnego modelu Copernicus GLO-30 co 30 m. Rzeźba terenu jest prawdziwa." },
      { q: "Czy można wydrukować duży model?", a: "Tak. Jeśli jest większy niż stół drukarki, dzielimy go na kafle łączące się bez szwu." },
      { q: "Czy figurki są w skali?", a: "Nie, są dekoracyjne i większe, żeby było je widać. Wspinacz stoi na najbardziej stromej ścianie, schronisko na łagodnym stoku." },
      { q: "Czy potrzebne są podpory?", a: "Teren, boki, obrzeże i dno tworzą jedną zamkniętą bryłę, która drukuje się bez podpór." },
    ],
  },
};
