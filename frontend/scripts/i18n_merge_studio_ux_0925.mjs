// UX-спрощення /mountains, /maket і Telegram у шапці (25.09.2026). Запуск: node scripts/i18n_merge_studio_ux_0925.mjs (з frontend/).
import fs from "node:fs"; import path from "node:path";
const DIR = path.resolve("messages");
const L = ["uk", "en", "de", "pl", "fr", "es"];

const mountains = {
  uk: { stepPlace: "Оберіть гору", stepSize: "Розмір", stepLook: "Вигляд", searchLabel: "Знайти гору", searchPlaceholder: "Назва вершини чи місця — Говерла, Ай-Петрі, Татри…",
    searchNothing: "Нічого не знайшли. Спробуйте іншу назву або оберіть точку на мапі.", searching: "Шукаю…", showAll: "Усі вершини ({n})", showLess: "Згорнути",
    pickOnMap: "Обрати точку на мапі", refineOnMap: "Уточнити на мапі", hideMap: "Сховати мапу", cm: "см", sizeCustomCm: "свій, см",
    style_classic: "Класика", style_classic_d: "заокруглений ободок, природні схили", style_rock: "Скеля", style_rock_d: "плаский ободок, скельні стінки", style_bare: "Лише рельєф", style_bare_d: "без ободка, рівний зріз", style_custom: "свої налаштування",
    figuresShort: "Фігурки (необовʼязково)", advanced: "Точні налаштування", advancedHint: "висота, ободок, боки, де стоять фігурки, текстура",
    agentToggle: "Описати словами — AI заповнить форму за вас", summaryRelief: "рельєф ≈{mm} мм", summaryFigures: "фігурок: {n}",
    etaHint: "Зазвичай 20–90 секунд. Модель можна покрутити й замовити друк.", previewLoading: "Завантажую рельєф…", askUs: "Питання чи хочете друк? Напишіть нам у Telegram",
    worldsLink: "Потрібна вигадана гора? «Світи з опису» →" },
  en: { stepPlace: "Choose a mountain", stepSize: "Size", stepLook: "Look", searchLabel: "Find a mountain", searchPlaceholder: "Peak or place name — Matterhorn, Fuji, Tatras…",
    searchNothing: "Nothing found. Try another name or pick a point on the map.", searching: "Searching…", showAll: "All peaks ({n})", showLess: "Show less",
    pickOnMap: "Pick a point on the map", refineOnMap: "Adjust on the map", hideMap: "Hide map", cm: "cm", sizeCustomCm: "custom, cm",
    style_classic: "Classic", style_classic_d: "rounded rim, natural slopes", style_rock: "Rock", style_rock_d: "flat rim, rocky walls", style_bare: "Relief only", style_bare_d: "no rim, clean cut", style_custom: "custom settings",
    figuresShort: "Figures (optional)", advanced: "Fine-tuning", advancedHint: "height, rim, sides, figure placement, texture",
    agentToggle: "Describe it in words — AI fills the form for you", summaryRelief: "relief ≈{mm} mm", summaryFigures: "figures: {n}",
    etaHint: "Usually 20–90 seconds. You can rotate the model and order a print.", previewLoading: "Loading terrain…", askUs: "Questions or want a print? Message us on Telegram",
    worldsLink: "Want an imaginary mountain? “Worlds from a description” →" },
  de: { stepPlace: "Berg wählen", stepSize: "Größe", stepLook: "Aussehen", searchLabel: "Berg suchen", searchPlaceholder: "Gipfel oder Ort — Matterhorn, Zugspitze, Tatra…",
    searchNothing: "Nichts gefunden. Anderen Namen versuchen oder Punkt auf der Karte wählen.", searching: "Suche…", showAll: "Alle Gipfel ({n})", showLess: "Weniger",
    pickOnMap: "Punkt auf der Karte wählen", refineOnMap: "Auf der Karte anpassen", hideMap: "Karte ausblenden", cm: "cm", sizeCustomCm: "eigene, cm",
    style_classic: "Klassisch", style_classic_d: "abgerundeter Rand, natürliche Hänge", style_rock: "Fels", style_rock_d: "flacher Rand, Felswände", style_bare: "Nur Relief", style_bare_d: "ohne Rand, gerader Schnitt", style_custom: "eigene Einstellungen",
    figuresShort: "Figuren (optional)", advanced: "Feineinstellungen", advancedHint: "Höhe, Rand, Seiten, Figurenposition, Textur",
    agentToggle: "In Worten beschreiben — die KI füllt das Formular aus", summaryRelief: "Relief ≈{mm} mm", summaryFigures: "Figuren: {n}",
    etaHint: "Meist 20–90 Sekunden. Modell drehen und Druck bestellen.", previewLoading: "Gelände wird geladen…", askUs: "Fragen oder Druck gewünscht? Schreiben Sie uns auf Telegram",
    worldsLink: "Ein erfundener Berg? „Welten aus Beschreibung“ →" },
  pl: { stepPlace: "Wybierz górę", stepSize: "Rozmiar", stepLook: "Wygląd", searchLabel: "Znajdź górę", searchPlaceholder: "Szczyt lub miejsce — Rysy, Giewont, Tatry…",
    searchNothing: "Nic nie znaleziono. Spróbuj innej nazwy lub wybierz punkt na mapie.", searching: "Szukam…", showAll: "Wszystkie szczyty ({n})", showLess: "Zwiń",
    pickOnMap: "Wybierz punkt na mapie", refineOnMap: "Doprecyzuj na mapie", hideMap: "Ukryj mapę", cm: "cm", sizeCustomCm: "własny, cm",
    style_classic: "Klasyka", style_classic_d: "zaokrąglona ramka, naturalne zbocza", style_rock: "Skała", style_rock_d: "płaska ramka, skalne ściany", style_bare: "Sama rzeźba", style_bare_d: "bez ramki, równe cięcie", style_custom: "własne ustawienia",
    figuresShort: "Figurki (opcjonalnie)", advanced: "Dokładne ustawienia", advancedHint: "wysokość, ramka, boki, położenie figurek, tekstura",
    agentToggle: "Opisz słowami — AI wypełni formularz", summaryRelief: "rzeźba ≈{mm} mm", summaryFigures: "figurki: {n}",
    etaHint: "Zwykle 20–90 sekund. Model można obracać i zamówić druk.", previewLoading: "Wczytuję teren…", askUs: "Pytania lub druk? Napisz do nas na Telegramie",
    worldsLink: "Wymyślona góra? „Światy z opisu” →" },
  fr: { stepPlace: "Choisissez une montagne", stepSize: "Taille", stepLook: "Style", searchLabel: "Trouver une montagne", searchPlaceholder: "Sommet ou lieu — Mont Blanc, Cervin, Pyrénées…",
    searchNothing: "Aucun résultat. Essayez un autre nom ou choisissez un point sur la carte.", searching: "Recherche…", showAll: "Tous les sommets ({n})", showLess: "Réduire",
    pickOnMap: "Choisir un point sur la carte", refineOnMap: "Ajuster sur la carte", hideMap: "Masquer la carte", cm: "cm", sizeCustomCm: "autre, cm",
    style_classic: "Classique", style_classic_d: "bord arrondi, pentes naturelles", style_rock: "Roche", style_rock_d: "bord plat, parois rocheuses", style_bare: "Relief seul", style_bare_d: "sans bord, coupe nette", style_custom: "réglages personnalisés",
    figuresShort: "Figurines (facultatif)", advanced: "Réglages fins", advancedHint: "hauteur, bord, côtés, position des figurines, texture",
    agentToggle: "Décrivez-la avec des mots — l’IA remplit le formulaire", summaryRelief: "relief ≈{mm} mm", summaryFigures: "figurines : {n}",
    etaHint: "En général 20–90 secondes. Faites pivoter le modèle et commandez l’impression.", previewLoading: "Chargement du relief…", askUs: "Des questions ou une impression ? Écrivez-nous sur Telegram",
    worldsLink: "Une montagne imaginaire ? « Mondes à partir d’une description » →" },
  es: { stepPlace: "Elige una montaña", stepSize: "Tamaño", stepLook: "Aspecto", searchLabel: "Buscar montaña", searchPlaceholder: "Cumbre o lugar — Aneto, Teide, Mont Blanc…",
    searchNothing: "Sin resultados. Prueba otro nombre o elige un punto en el mapa.", searching: "Buscando…", showAll: "Todas las cumbres ({n})", showLess: "Ver menos",
    pickOnMap: "Elegir un punto en el mapa", refineOnMap: "Ajustar en el mapa", hideMap: "Ocultar mapa", cm: "cm", sizeCustomCm: "otro, cm",
    style_classic: "Clásico", style_classic_d: "borde redondeado, laderas naturales", style_rock: "Roca", style_rock_d: "borde plano, paredes de roca", style_bare: "Solo relieve", style_bare_d: "sin borde, corte recto", style_custom: "ajustes propios",
    figuresShort: "Figuras (opcional)", advanced: "Ajustes finos", advancedHint: "altura, borde, lados, posición de figuras, textura",
    agentToggle: "Descríbela con palabras — la IA rellena el formulario", summaryRelief: "relieve ≈{mm} mm", summaryFigures: "figuras: {n}",
    etaHint: "Normalmente 20–90 segundos. Puedes girar el modelo y pedir la impresión.", previewLoading: "Cargando el relieve…", askUs: "¿Dudas o quieres imprimirla? Escríbenos por Telegram",
    worldsLink: "¿Una montaña imaginaria? «Mundos desde una descripción» →" },
};

const maket = {
  uk: { stepUpload: "План", stepCheck: "Перевірка", stepModel: "3D-модель", dropHere: "Перетягніть файл сюди, вставте Ctrl+V або", trySample: "Спробувати на прикладі плану",
    sampleNote: "Немає плану під рукою? Подивіться, як це працює, на нашому зразку.", formats: "JPG, PNG, WEBP або PDF",
    scaleNeedTitle: "Перевірте розмір квартири", scaleEasiest: "Найпростіше — впишіть загальну площу з договору:", scaleOr: "або", scaleRulerAlt: "проведіть лінійку по відомому розміру",
    scaleLooksRight: "Розмір виглядає правильно", priceFrom: "Друк макета: {price} ₴", downloadFailed: "Не вдалося завантажити файл. Спробуйте ще раз або напишіть нам.",
    changeScale: "Змінити", uploadOther: "Інший план", askUs: "Складний план? Надішліть його нам у Telegram — допоможемо" },
  en: { stepUpload: "Plan", stepCheck: "Check", stepModel: "3D model", dropHere: "Drop a file here, paste with Ctrl+V, or", trySample: "Try with a sample plan",
    sampleNote: "No plan at hand? See how it works on our sample.", formats: "JPG, PNG, WEBP or PDF",
    scaleNeedTitle: "Check the apartment size", scaleEasiest: "Easiest: enter the total floor area from your contract:", scaleOr: "or", scaleRulerAlt: "draw the ruler along a known dimension",
    scaleLooksRight: "The size looks right", priceFrom: "Printed model: {price} ₴", downloadFailed: "Could not download the file. Try again or message us.",
    changeScale: "Change", uploadOther: "Another plan", askUs: "Tricky plan? Send it to us on Telegram — we’ll help" },
  de: { stepUpload: "Grundriss", stepCheck: "Prüfen", stepModel: "3D-Modell", dropHere: "Datei hierher ziehen, mit Strg+V einfügen oder", trySample: "Mit Beispiel-Grundriss testen",
    sampleNote: "Kein Grundriss zur Hand? Sehen Sie am Beispiel, wie es funktioniert.", formats: "JPG, PNG, WEBP oder PDF",
    scaleNeedTitle: "Wohnungsgröße prüfen", scaleEasiest: "Am einfachsten: Gesamtfläche aus dem Vertrag eingeben:", scaleOr: "oder", scaleRulerAlt: "Lineal an einem bekannten Maß ziehen",
    scaleLooksRight: "Die Größe stimmt", priceFrom: "Gedrucktes Modell: {price} ₴", downloadFailed: "Download fehlgeschlagen. Erneut versuchen oder uns schreiben.",
    changeScale: "Ändern", uploadOther: "Anderer Grundriss", askUs: "Schwieriger Grundriss? Schicken Sie ihn uns auf Telegram — wir helfen" },
  pl: { stepUpload: "Plan", stepCheck: "Sprawdzenie", stepModel: "Model 3D", dropHere: "Przeciągnij plik tutaj, wklej Ctrl+V lub", trySample: "Wypróbuj na przykładowym planie",
    sampleNote: "Nie masz planu pod ręką? Zobacz, jak to działa, na naszym przykładzie.", formats: "JPG, PNG, WEBP lub PDF",
    scaleNeedTitle: "Sprawdź wielkość mieszkania", scaleEasiest: "Najprościej: wpisz powierzchnię z umowy:", scaleOr: "lub", scaleRulerAlt: "przeciągnij linijkę wzdłuż znanego wymiaru",
    scaleLooksRight: "Rozmiar się zgadza", priceFrom: "Wydruk makiety: {price} ₴", downloadFailed: "Nie udało się pobrać pliku. Spróbuj ponownie lub napisz do nas.",
    changeScale: "Zmień", uploadOther: "Inny plan", askUs: "Trudny plan? Wyślij go nam na Telegramie — pomożemy" },
  fr: { stepUpload: "Plan", stepCheck: "Vérification", stepModel: "Modèle 3D", dropHere: "Déposez un fichier ici, collez avec Ctrl+V ou", trySample: "Essayer avec un plan d’exemple",
    sampleNote: "Pas de plan sous la main ? Voyez comment ça marche sur notre exemple.", formats: "JPG, PNG, WEBP ou PDF",
    scaleNeedTitle: "Vérifiez la taille du logement", scaleEasiest: "Le plus simple : saisissez la surface totale du contrat :", scaleOr: "ou", scaleRulerAlt: "tracez la règle sur une cote connue",
    scaleLooksRight: "La taille est correcte", priceFrom: "Maquette imprimée : {price} ₴", downloadFailed: "Échec du téléchargement. Réessayez ou écrivez-nous.",
    changeScale: "Modifier", uploadOther: "Autre plan", askUs: "Plan compliqué ? Envoyez-le-nous sur Telegram — on vous aide" },
  es: { stepUpload: "Plano", stepCheck: "Revisión", stepModel: "Modelo 3D", dropHere: "Arrastra un archivo aquí, pega con Ctrl+V o", trySample: "Probar con un plano de ejemplo",
    sampleNote: "¿No tienes un plano a mano? Mira cómo funciona con nuestro ejemplo.", formats: "JPG, PNG, WEBP o PDF",
    scaleNeedTitle: "Comprueba el tamaño de la vivienda", scaleEasiest: "Lo más fácil: escribe la superficie total del contrato:", scaleOr: "o", scaleRulerAlt: "traza la regla sobre una medida conocida",
    scaleLooksRight: "El tamaño es correcto", priceFrom: "Maqueta impresa: {price} ₴", downloadFailed: "No se pudo descargar el archivo. Inténtalo de nuevo o escríbenos.",
    changeScale: "Cambiar", uploadOther: "Otro plano", askUs: "¿Plano complicado? Envíanoslo por Telegram — te ayudamos" },
};

const nav = {
  uk: { telegram: "Написати в Telegram" }, en: { telegram: "Message us on Telegram" }, de: { telegram: "Auf Telegram schreiben" },
  pl: { telegram: "Napisz na Telegramie" }, fr: { telegram: "Écrire sur Telegram" }, es: { telegram: "Escribir por Telegram" },
};

for (const l of L) {
  const f = path.join(DIR, `${l}.json`);
  const j = JSON.parse(fs.readFileSync(f, "utf8"));
  j.mountains = { ...(j.mountains || {}), ...mountains[l] };
  j.maket = { ...(j.maket || {}), ...maket[l] };
  j.nav = { ...(j.nav || {}), ...nav[l] };
  fs.writeFileSync(f, JSON.stringify(j, null, 2) + "\n", "utf8");
  console.log(l, "ok");
}
