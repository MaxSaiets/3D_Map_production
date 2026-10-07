import type { Metadata } from "next";
import { setRequestLocale } from "next-intl/server";
import { BASE, localeUrl } from "@/i18n/metadata";
import { routing, locales, localeMeta, defaultLocale, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";
import { KEYCHAIN_PRICE_UAH, MAP_MAGNET_PRICE_UAH, mapPriceEur } from "@/lib/mapPrices";
import { CorporateLeadForm } from "@/components/CorporateLeadForm";

/**
 * Лендінг /corporate — B2B: корпоративні подарунки та мерч з мапою.
 * Кластер з аудиту 16.07: «корпоративні подарунки (з логотипом / для команди /
 * партнерам)», «мерч на замовлення», «подарунки для команди». Вищий чек,
 * ніж роздріб (B2B-playbook). Без фейкових знижок: ціни «від», наклад —
 * індивідуально через контакт.
 */
type CorpCopy = {
  title: string; description: string; h1: string; intro: string;
  h2what: string; items: { label: string; desc: string; price: string }[];
  h2cases: string; cases: { h3: string; p: string }[];
  h2how: string; how: string[]; h2faq: string; faq: { q: string; a: string }[];
  cta: string; ctaTry: string;
};

const eur = (uah: number) => mapPriceEur(uah);

const COPY_TRANSLATED: Record<Exclude<AppLocale, "ro">, CorpCopy> = {
  uk: {
    title: "Корпоративні подарунки з мапою — мерч для команди",
    description: `Корпоративні 3D-подарунки на замовлення: брелоки з районом офісу, мапи міста з вашим текстом, магніти для партнерів. Наклади від 10 шт, від ${KEYCHAIN_PRICE_UAH} ₴/шт.`,
    h1: "Корпоративні подарунки з мапою вашого міста",
    intro:
      "Мерч, який не викидають: 3D-брелоки з районом вашого офісу, мапи міста з фірмовим текстом, магніти з локацією події. Кожен виріб — реальна карта місця, яке щось означає для вашої команди чи клієнтів.",
    h2what: "Що можна замовити",
    items: [
      { label: "Брелоки з районом офісу", desc: "жетон 55×30 мм з вулицями навколо вашого офісу + назва компанії на звороті", price: `від ${KEYCHAIN_PRICE_UAH} ₴/шт` },
      { label: "Мапи міста з вашим текстом", desc: "настільні 3D-мапи 8–15 см — подарунок партнерам чи топ-клієнтам", price: "від 490 ₴/шт" },
      { label: "Магніти з локацією", desc: "район офісу, місце конференції чи місто філії", price: `від ${MAP_MAGNET_PRICE_UAH} ₴/шт` },
      { label: "GPX-брелоки для спортивних команд", desc: "маршрут корпоративного забігу чи велопробігу", price: `від ${KEYCHAIN_PRICE_UAH} ₴/шт` },
    ],
    h2cases: "Кому це замовляють",
    cases: [
      { h3: "Команді", p: "Онбординг-набори з брелоком району офісу, річниці компанії, подарунки на новий рік. Кожен співробітник отримує шматочок спільного місця." },
      { h3: "Партнерам і клієнтам", p: "Мапа міста, де відбулась угода чи відкрився спільний проєкт — подарунок, який лишається на столі, а не в шухляді." },
      { h3: "Подіям", p: "Мерч конференції з районом локації, брелоки фінішера для корпоративного забігу з реальним GPX-треком маршруту." },
      { h3: "Релокованим командам", p: "Брелок з рідним містом для колег, які переїхали — емоційний подарунок, що показує турботу." },
    ],
    h2how: "Як замовити наклад",
    how: [
      "Напишіть нам: скільки штук, яке місце (район офісу, місто, маршрут) і який текст.",
      "Зберемо безкоштовне 3D-превʼю виробу — затвердите вигляд до оплати.",
      "Виготовлення: 10–50 шт за 3–7 робочих днів, більші наклади — за домовленістю.",
      "Оплата на ФОП з документами; доставка Новою Поштою на офіс або кожному співробітнику окремо.",
    ],
    h2faq: "Часті запитання",
    faq: [
      { q: "Який мінімальний наклад?", a: "Від 10 штук. Менше — просто замовте поштучно в конструкторі за роздрібною ціною." },
      { q: "Чи можна додати логотип?", a: "На звороті брелока розміщуємо текст: назву компанії, слоган, дату події чи координати. Об'ємний друк дрібних логотипів обмежений технологією — чесно покажемо на превʼю, як виглядатиме саме ваш." },
      { q: "Чи даєте документи для бухгалтерії?", a: "Так — працюємо як ФОП: рахунок, акт виконаних робіт. Оплата на рахунок." },
      { q: "Скільки триває виготовлення накладу?", a: "10–50 шт — 3–7 робочих днів. Понад 50 — узгоджуємо терміни окремо, чесно і до передоплати." },
      { q: "Чи можуть вироби бути різними в одному накладі?", a: "Так — наприклад, кожному співробітнику брелок з ЙОГО районом чи рідним містом. Це наша улюблена корпоративна історія: однаковий формат, персональний зміст." },
    ],
    cta: "Написати нам про наклад",
    ctaTry: "Спробувати конструктор",
  },
  en: {
    title: "Corporate map gifts — team merch made to order",
    description: `Corporate 3D gifts: keychains with your office district, city maps with custom text, magnets for partners. Runs from 10 pcs, from ≈€${eur(KEYCHAIN_PRICE_UAH)}/pc.`,
    h1: "Corporate gifts with the map of your city",
    intro:
      "Merch people don't throw away: 3D keychains with your office district, city maps with company text, magnets with an event location. Every piece is a real map of a place that means something to your team or clients.",
    h2what: "What you can order",
    items: [
      { label: "Office-district keychains", desc: "a 55×30 mm tag with the streets around your office + company name on the back", price: `from ≈€${eur(KEYCHAIN_PRICE_UAH)}/pc` },
      { label: "City maps with your text", desc: "desk 3D maps 8–15 cm — a gift for partners or top clients", price: `from ≈€${eur(350)}/pc` },
      { label: "Location magnets", desc: "office district, conference venue or branch city", price: `from ≈€${eur(MAP_MAGNET_PRICE_UAH)}/pc` },
      { label: "GPX keychains for sports teams", desc: "the route of a corporate run or bike ride", price: `from ≈€${eur(KEYCHAIN_PRICE_UAH)}/pc` },
    ],
    h2cases: "Who orders this",
    cases: [
      { h3: "For the team", p: "Onboarding kits with an office-district keychain, company anniversaries, new-year gifts. Everyone gets a piece of the shared place." },
      { h3: "For partners and clients", p: "A map of the city where the deal happened — a gift that stays on the desk, not in a drawer." },
      { h3: "For events", p: "Conference merch with the venue district, finisher keychains for a corporate run with the real GPX track." },
      { h3: "For relocated teams", p: "A keychain with the home city for colleagues who moved — an emotional gift that shows care." },
    ],
    h2how: "How to order a run",
    how: [
      "Write to us: quantity, place (office district, city, route) and text.",
      "We build a free 3D preview — you approve the look before paying.",
      "Production: 10–50 pcs in 3–7 business days; larger runs by agreement.",
      "Invoice with documents; delivery to the office or to each employee individually.",
    ],
    h2faq: "FAQ",
    faq: [
      { q: "Minimum quantity?", a: "From 10 pieces. For fewer, just order one by one in the builder at the retail price." },
      { q: "Can you add a logo?", a: "We place text on the back: company name, slogan, event date or coordinates. Relief printing of small logos is limited by the technology — we'll honestly show you a preview of how yours would look." },
      { q: "Do you provide accounting documents?", a: "Yes — invoice and act of completed work, payment to a business account." },
      { q: "Production time for a run?", a: "10–50 pcs — 3–7 business days. Over 50 — agreed separately and honestly, before any prepayment." },
      { q: "Can items differ within one run?", a: "Yes — e.g. each employee gets a keychain with THEIR district or home city. Same format, personal content — our favourite corporate story." },
    ],
    cta: "Contact us about a run",
    ctaTry: "Try the builder",
  },
  de: {
    title: "Firmengeschenke mit Stadtkarte — Team-Merch",
    description: `3D-Firmengeschenke: Anhänger mit dem Büro-Viertel, Stadtkarten mit Firmentext, Magnete für Partner. Auflagen ab 10 Stück, ab ≈${eur(KEYCHAIN_PRICE_UAH)} €/Stück.`,
    h1: "Firmengeschenke mit der Karte eurer Stadt",
    intro:
      "Merch, das niemand wegwirft: 3D-Anhänger mit dem Viertel eures Büros, Stadtkarten mit Firmentext, Magnete mit dem Event-Ort. Jedes Stück ist eine echte Karte eines Ortes, der eurem Team oder euren Kunden etwas bedeutet.",
    h2what: "Was bestellt werden kann",
    items: [
      { label: "Anhänger mit Büro-Viertel", desc: "55×30-mm-Anhänger mit den Straßen rund ums Büro + Firmenname auf der Rückseite", price: `ab ≈${eur(KEYCHAIN_PRICE_UAH)} €/St.` },
      { label: "Stadtkarten mit eurem Text", desc: "Tisch-3D-Karten 8–15 cm — für Partner oder Top-Kunden", price: `ab ≈${eur(350)} €/St.` },
      { label: "Standort-Magnete", desc: "Büro-Viertel, Konferenzort oder Filialstadt", price: `ab ≈${eur(MAP_MAGNET_PRICE_UAH)} €/St.` },
      { label: "GPX-Anhänger für Sportteams", desc: "die Route eines Firmenlaufs oder einer Radtour", price: `ab ≈${eur(KEYCHAIN_PRICE_UAH)} €/St.` },
    ],
    h2cases: "Wer das bestellt",
    cases: [
      { h3: "Fürs Team", p: "Onboarding-Sets mit Büro-Viertel-Anhänger, Firmenjubiläen, Neujahrsgeschenke." },
      { h3: "Für Partner und Kunden", p: "Eine Karte der Stadt, in der der Deal zustande kam — bleibt auf dem Schreibtisch statt in der Schublade." },
      { h3: "Für Events", p: "Konferenz-Merch mit dem Veranstaltungsviertel, Finisher-Anhänger mit echtem GPX-Track." },
      { h3: "Für relocierte Teams", p: "Ein Anhänger mit der Heimatstadt für umgezogene Kolleginnen und Kollegen." },
    ],
    h2how: "So läuft eine Auflage",
    how: [
      "Schreibt uns: Stückzahl, Ort (Büro-Viertel, Stadt, Route) und Text.",
      "Wir bauen eine kostenlose 3D-Vorschau — ihr gebt das Design vor der Zahlung frei.",
      "Fertigung: 10–50 Stück in 3–7 Werktagen; größere Auflagen nach Absprache.",
      "Rechnung mit Unterlagen; Lieferung ans Büro oder einzeln an jede Person.",
    ],
    h2faq: "Häufige Fragen",
    faq: [
      { q: "Mindestmenge?", a: "Ab 10 Stück. Für weniger einfach einzeln im Konfigurator zum Einzelpreis bestellen." },
      { q: "Ist ein Logo möglich?", a: "Auf der Rückseite platzieren wir Text: Firmenname, Slogan, Datum oder Koordinaten. Reliefdruck kleiner Logos ist technologisch begrenzt — die Vorschau zeigt ehrlich, wie eures aussähe." },
      { q: "Gibt es Buchhaltungsunterlagen?", a: "Ja — Rechnung und Leistungsnachweis, Zahlung auf Geschäftskonto." },
      { q: "Fertigungszeit?", a: "10–50 Stück — 3–7 Werktage. Über 50 — separat und ehrlich vereinbart, vor jeder Anzahlung." },
      { q: "Dürfen die Stücke unterschiedlich sein?", a: "Ja — z. B. jede Person bekommt einen Anhänger mit IHREM Viertel oder ihrer Heimatstadt. Gleiches Format, persönlicher Inhalt." },
    ],
    cta: "Wegen einer Auflage schreiben",
    ctaTry: "Konfigurator testen",
  },
  pl: {
    title: "Prezenty firmowe z mapą — merch dla zespołu",
    description: `Firmowe prezenty 3D: breloki z dzielnicą biura, mapy miasta z tekstem firmy, magnesy dla partnerów. Nakłady od 10 szt., od ≈${eur(KEYCHAIN_PRICE_UAH)} €/szt.`,
    h1: "Prezenty firmowe z mapą waszego miasta",
    intro:
      "Merch, którego nikt nie wyrzuca: breloki 3D z dzielnicą biura, mapy miasta z firmowym tekstem, magnesy z lokalizacją wydarzenia. Każda sztuka to prawdziwa mapa miejsca, które coś znaczy dla zespołu lub klientów.",
    h2what: "Co można zamówić",
    items: [
      { label: "Breloki z dzielnicą biura", desc: "zawieszka 55×30 mm z ulicami wokół biura + nazwa firmy z tyłu", price: `od ≈${eur(KEYCHAIN_PRICE_UAH)} €/szt.` },
      { label: "Mapy miasta z waszym tekstem", desc: "biurkowe mapy 3D 8–15 cm — prezent dla partnerów", price: `od ≈${eur(350)} €/szt.` },
      { label: "Magnesy z lokalizacją", desc: "dzielnica biura, miejsce konferencji lub miasto oddziału", price: `od ≈${eur(MAP_MAGNET_PRICE_UAH)} €/szt.` },
      { label: "Breloki GPX dla drużyn sportowych", desc: "trasa firmowego biegu lub rajdu rowerowego", price: `od ≈${eur(KEYCHAIN_PRICE_UAH)} €/szt.` },
    ],
    h2cases: "Kto to zamawia",
    cases: [
      { h3: "Dla zespołu", p: "Zestawy onboardingowe z brelokiem dzielnicy biura, rocznice firmy, prezenty noworoczne." },
      { h3: "Dla partnerów i klientów", p: "Mapa miasta, w którym doszło do współpracy — prezent, który zostaje na biurku." },
      { h3: "Na wydarzenia", p: "Merch konferencyjny z dzielnicą lokalizacji, breloki finishera z prawdziwym śladem GPX." },
      { h3: "Dla zespołów po relokacji", p: "Brelok z rodzinnym miastem dla kolegów, którzy się przeprowadzili." },
    ],
    h2how: "Jak zamówić nakład",
    how: [
      "Napiszcie do nas: ile sztuk, jakie miejsce (dzielnica biura, miasto, trasa) i jaki tekst.",
      "Przygotujemy darmowy podgląd 3D — zatwierdzacie wygląd przed płatnością.",
      "Realizacja: 10–50 szt. w 3–7 dni roboczych; większe nakłady do uzgodnienia.",
      "Faktura z dokumentami; dostawa do biura lub do każdego pracownika osobno.",
    ],
    h2faq: "Częste pytania",
    faq: [
      { q: "Minimalny nakład?", a: "Od 10 sztuk. Mniej — po prostu zamówcie pojedynczo w kreatorze po cenie detalicznej." },
      { q: "Czy można dodać logo?", a: "Z tyłu umieszczamy tekst: nazwę firmy, slogan, datę lub współrzędne. Reliefowy druk małych logotypów jest ograniczony technologią — podgląd uczciwie pokaże, jak wyglądałoby wasze." },
      { q: "Czy są dokumenty księgowe?", a: "Tak — faktura i protokół, płatność na konto firmowe." },
      { q: "Czas realizacji nakładu?", a: "10–50 szt. — 3–7 dni roboczych. Powyżej 50 — ustalane osobno, uczciwie i przed przedpłatą." },
      { q: "Czy sztuki mogą się różnić?", a: "Tak — np. każdy pracownik dostaje brelok ze SWOJĄ dzielnicą lub rodzinnym miastem. Ten sam format, osobista treść." },
    ],
    cta: "Napisać w sprawie nakładu",
    ctaTry: "Wypróbować kreator",
  },
  fr: {
    title: "Cadeaux d'entreprise avec carte — merch d'équipe",
    description: `Cadeaux 3D d'entreprise : porte-clés avec le quartier du bureau, cartes de ville avec votre texte, magnets pour partenaires. Séries dès 10 pièces, dès ≈${eur(KEYCHAIN_PRICE_UAH)} €/pièce.`,
    h1: "Cadeaux d'entreprise avec la carte de votre ville",
    intro:
      "Du merch qu'on ne jette pas : porte-clés 3D avec le quartier de votre bureau, cartes de ville avec texte d'entreprise, magnets avec le lieu d'un événement. Chaque pièce est une vraie carte d'un lieu qui compte pour votre équipe ou vos clients.",
    h2what: "Ce qu'on peut commander",
    items: [
      { label: "Porte-clés du quartier du bureau", desc: "plaque 55×30 mm avec les rues autour du bureau + nom de l'entreprise au dos", price: `dès ≈${eur(KEYCHAIN_PRICE_UAH)} €/pc` },
      { label: "Cartes de ville avec votre texte", desc: "cartes 3D de bureau 8–15 cm — cadeau partenaires ou grands clients", price: `dès ≈${eur(350)} €/pc` },
      { label: "Magnets de lieu", desc: "quartier du bureau, lieu de conférence ou ville d'une filiale", price: `dès ≈${eur(MAP_MAGNET_PRICE_UAH)} €/pc` },
      { label: "Porte-clés GPX pour équipes sportives", desc: "le parcours d'une course ou sortie vélo d'entreprise", price: `dès ≈${eur(KEYCHAIN_PRICE_UAH)} €/pc` },
    ],
    h2cases: "Qui commande cela",
    cases: [
      { h3: "Pour l'équipe", p: "Kits d'onboarding avec porte-clés du quartier du bureau, anniversaires d'entreprise, cadeaux de fin d'année." },
      { h3: "Pour partenaires et clients", p: "La carte de la ville où l'affaire s'est conclue — un cadeau qui reste sur le bureau." },
      { h3: "Pour les événements", p: "Merch de conférence avec le quartier du lieu, porte-clés finisher avec la vraie trace GPX." },
      { h3: "Pour les équipes relocalisées", p: "Un porte-clés avec la ville natale pour les collègues qui ont déménagé." },
    ],
    h2how: "Commander une série",
    how: [
      "Écrivez-nous : quantité, lieu (quartier du bureau, ville, parcours) et texte.",
      "Nous préparons un aperçu 3D gratuit — vous validez avant paiement.",
      "Fabrication : 10–50 pièces en 3–7 jours ouvrés ; plus grandes séries sur accord.",
      "Facture avec documents ; livraison au bureau ou à chaque personne individuellement.",
    ],
    h2faq: "Questions fréquentes",
    faq: [
      { q: "Quantité minimale ?", a: "Dès 10 pièces. En dessous, commandez simplement à l'unité dans le configurateur au prix détail." },
      { q: "Peut-on ajouter un logo ?", a: "Au dos, nous plaçons du texte : nom d'entreprise, slogan, date ou coordonnées. L'impression en relief de petits logos est limitée par la technologie — l'aperçu montrera honnêtement le rendu du vôtre." },
      { q: "Des documents comptables ?", a: "Oui — facture et acte, paiement sur compte professionnel." },
      { q: "Délai de fabrication ?", a: "10–50 pièces — 3–7 jours ouvrés. Au-delà de 50 — convenu séparément, honnêtement, avant tout acompte." },
      { q: "Les pièces peuvent-elles différer ?", a: "Oui — p. ex. chaque employé reçoit un porte-clés avec SON quartier ou sa ville natale. Même format, contenu personnel." },
    ],
    cta: "Nous écrire pour une série",
    ctaTry: "Essayer le configurateur",
  },
  es: {
    title: "Regalos corporativos con mapa — merch para equipos",
    description: `Regalos 3D corporativos: llaveros con el distrito de la oficina, mapas de ciudad con su texto, imanes para socios. Tiradas desde 10 uds., desde ≈${eur(KEYCHAIN_PRICE_UAH)} €/ud.`,
    h1: "Regalos corporativos con el mapa de vuestra ciudad",
    intro:
      "Merch que no se tira: llaveros 3D con el barrio de vuestra oficina, mapas de ciudad con texto corporativo, imanes con la ubicación de un evento. Cada pieza es un mapa real de un lugar con significado para el equipo o los clientes.",
    h2what: "Qué se puede pedir",
    items: [
      { label: "Llaveros del barrio de la oficina", desc: "placa 55×30 mm con las calles alrededor de la oficina + nombre de la empresa al dorso", price: `desde ≈${eur(KEYCHAIN_PRICE_UAH)} €/ud.` },
      { label: "Mapas de ciudad con vuestro texto", desc: "mapas 3D de escritorio 8–15 cm — regalo para socios o grandes clientes", price: `desde ≈${eur(350)} €/ud.` },
      { label: "Imanes de ubicación", desc: "barrio de la oficina, sede de conferencia o ciudad de una filial", price: `desde ≈${eur(MAP_MAGNET_PRICE_UAH)} €/ud.` },
      { label: "Llaveros GPX para equipos deportivos", desc: "la ruta de una carrera o salida en bici corporativa", price: `desde ≈${eur(KEYCHAIN_PRICE_UAH)} €/ud.` },
    ],
    h2cases: "Quién lo pide",
    cases: [
      { h3: "Para el equipo", p: "Kits de onboarding con llavero del barrio de la oficina, aniversarios de empresa, regalos de fin de año." },
      { h3: "Para socios y clientes", p: "El mapa de la ciudad donde se cerró el acuerdo — un regalo que se queda en el escritorio." },
      { h3: "Para eventos", p: "Merch de conferencia con el barrio de la sede, llaveros finisher con el track GPX real." },
      { h3: "Para equipos reubicados", p: "Un llavero con la ciudad natal para colegas que se mudaron." },
    ],
    h2how: "Cómo pedir una tirada",
    how: [
      "Escribidnos: cantidad, lugar (barrio de la oficina, ciudad, ruta) y texto.",
      "Preparamos una vista previa 3D gratuita — aprobáis el diseño antes de pagar.",
      "Producción: 10–50 uds. en 3–7 días hábiles; tiradas mayores según acuerdo.",
      "Factura con documentos; entrega en la oficina o a cada persona por separado.",
    ],
    h2faq: "Preguntas frecuentes",
    faq: [
      { q: "¿Cantidad mínima?", a: "Desde 10 piezas. Para menos, pedid simplemente por unidades en el configurador a precio minorista." },
      { q: "¿Se puede añadir un logo?", a: "Al dorso colocamos texto: nombre de la empresa, eslogan, fecha o coordenadas. La impresión en relieve de logos pequeños está limitada por la tecnología — la vista previa mostrará honestamente cómo quedaría el vuestro." },
      { q: "¿Documentos contables?", a: "Sí — factura y acta, pago a cuenta profesional." },
      { q: "¿Plazo de producción?", a: "10–50 uds. — 3–7 días hábiles. Más de 50 — acordado por separado, con honestidad, antes de cualquier anticipo." },
      { q: "¿Pueden variar las piezas?", a: "Sí — p. ej., cada empleado recibe un llavero con SU barrio o ciudad natal. Mismo formato, contenido personal." },
    ],
    cta: "Escribirnos sobre una tirada",
    ctaTry: "Probar el configurador",
  },
};
// ro (07.10.2026): перекладу цього лендінгу ще нема → англійський текст; /ro-версія
// сторінки в noindex і поза sitemap (middleware.ts, RO_INDEXED_PATHS).
const COPY: Record<AppLocale, CorpCopy> = { ...COPY_TRANSLATED, ro: COPY_TRANSLATED.en };

/** 30.09.2026: новорічний сезон корпоративних подарунків. Дати — ті самі, що в
 *  сезонному банері сайту (друк 2–4 дні + доставка); для великих накладів —
 *  чесно просимо писати раніше, без обіцянок, яких не виконаємо. */
const SEASON: Record<AppLocale, { h2: string; p: string; points: string[]; photos: string }> = {
  uk: {
    h2: "Новорічні корпоративні подарунки",
    p: "Брелок з районом офісу чи мапа міста, де працює команда, запамʼятовується краще за чергову чашку з логотипом. Кожен виріб можна зробити персональним: інший район, імʼя чи дата для кожної людини.",
    points: ["До Миколая: заявка до 27 листопада", "До Нового року: заявка до 19 грудня", "Наклад понад 50 штук: напишіть якомога раніше, погодимо термін до оплати"],
    photos: "Справжні вироби, які ми надрукували",
  },
  en: {
    h2: "New Year corporate gifts",
    p: "A keychain with the office district or a map of the city where the team works is remembered longer than another branded mug. Each piece can be personal: a different district, name or date for every person.",
    points: ["For St Nicholas Day: request by 27 November", "For New Year: request by 19 December", "Runs over 50 pieces: write as early as possible, we agree the date before payment"],
    photos: "Real pieces we have printed",
  },
  de: {
    h2: "Firmengeschenke zum Jahreswechsel",
    p: "Ein Anhänger mit dem Büro-Viertel oder eine Karte der Stadt, in der das Team arbeitet, bleibt länger im Gedächtnis als die nächste Logo-Tasse. Jedes Stück kann persönlich sein: anderes Viertel, Name oder Datum pro Person.",
    points: ["Zum Nikolaus: Anfrage bis 27. November", "Zum Neujahr: Anfrage bis 19. Dezember", "Über 50 Stück: bitte früh schreiben, den Termin klären wir vor der Zahlung"],
    photos: "Echte Stücke, die wir gedruckt haben",
  },
  fr: {
    h2: "Cadeaux d'entreprise de fin d'année",
    p: "Un porte-clés avec le quartier du bureau ou une carte de la ville de l'équipe marque plus qu'une énième tasse au logo. Chaque pièce peut être personnelle : quartier, prénom ou date différents pour chacun.",
    points: ["Pour la Saint-Nicolas : demande avant le 27 novembre", "Pour le Nouvel An : demande avant le 19 décembre", "Plus de 50 pièces : écrivez tôt, nous fixons la date avant paiement"],
    photos: "De vraies pièces que nous avons imprimées",
  },
  es: {
    h2: "Regalos corporativos de Año Nuevo",
    p: "Un llavero con el barrio de la oficina o un mapa de la ciudad del equipo se recuerda más que otra taza con logo. Cada pieza puede ser personal: otro barrio, nombre o fecha para cada persona.",
    points: ["Para San Nicolás: pedido antes del 27 de noviembre", "Para Año Nuevo: pedido antes del 19 de diciembre", "Más de 50 piezas: escribe cuanto antes, acordamos la fecha antes del pago"],
    photos: "Piezas reales que hemos impreso",
  },
  pl: {
    h2: "Noworoczne prezenty firmowe",
    p: "Brelok z dzielnicą biura albo mapa miasta, w którym pracuje zespół, zostaje w pamięci dłużej niż kolejny kubek z logo. Każdy wyrób może być osobisty: inna dzielnica, imię lub data dla każdej osoby.",
    points: ["Na Mikołajki: zapytanie do 27 listopada", "Na Nowy Rok: zapytanie do 19 grudnia", "Ponad 50 sztuk: napisz jak najwcześniej, termin ustalimy przed płatnością"],
    photos: "Prawdziwe wyroby, które wydrukowaliśmy",
  },
  ro: {
    h2: "Cadouri corporative de Anul Nou",
    p: "Un breloc cu cartierul biroului sau o hartă a orașului unde lucrează echipa rămâne în memorie mai mult decât încă o cană cu logo. Fiecare piesă poate fi personală: alt cartier, alt nume sau altă dată pentru fiecare om.",
    points: ["Pentru Sfântul Nicolae: cerere până pe 27 noiembrie", "Pentru Anul Nou: cerere până pe 19 decembrie", "Tiraje de peste 50 de bucăți: scrieți cât mai devreme, stabilim data înainte de plată"],
    photos: "Piese reale pe care le-am imprimat",
  },
};

/** 30.09.2026: знижки на наклад ПІДТВЕРДЖЕНІ власником. Відсоток від роздробу;
 *  ціни за штуку рахуються з тих самих констант, що й роздріб (без дрейфу). */
const TIERS: { min: number; max: number | null; pct: number }[] = [
  { min: 10, max: 29, pct: 10 },
  { min: 30, max: 99, pct: 15 },
  { min: 100, max: null, pct: 20 },
];
const RESELLER_PCT = 25;
const off = (uah: number, pct: number) => Math.round((uah * (100 - pct)) / 100);
const TIER_COPY: Record<AppLocale, { h2: string; qty: string; disc: string; kc: string; map: string; reseller: string; note: string; pcs: string; plus: string }> = {
  uk: { h2: "Знижки на наклад", qty: "Кількість", disc: "Знижка", kc: "Брелок", map: "Мапа 8 см", reseller: "Реселерам (для перепродажу)", note: "Знижка діє на весь наклад і на різні вироби в одному замовленні. Ціни за штуку, доставка Новою Поштою окремо.", pcs: "шт", plus: "+" },
  en: { h2: "Volume discounts", qty: "Quantity", disc: "Discount", kc: "Keychain", map: "8 cm map", reseller: "Resellers", note: "The discount applies to the whole run and to mixed items in one order. Prices per piece, Nova Poshta delivery extra.", pcs: "pcs", plus: "+" },
  de: { h2: "Mengenrabatt", qty: "Menge", disc: "Rabatt", kc: "Anhänger", map: "Karte 8 cm", reseller: "Wiederverkäufer", note: "Der Rabatt gilt für die ganze Bestellung, auch bei gemischten Artikeln. Preise pro Stück, Versand extra.", pcs: "Stk.", plus: "+" },
  fr: { h2: "Remises sur quantité", qty: "Quantité", disc: "Remise", kc: "Porte-clés", map: "Carte 8 cm", reseller: "Revendeurs", note: "La remise s'applique à toute la commande, articles mixtes compris. Prix à la pièce, livraison en sus.", pcs: "pcs", plus: "+" },
  es: { h2: "Descuentos por volumen", qty: "Cantidad", disc: "Descuento", kc: "Llavero", map: "Mapa 8 cm", reseller: "Revendedores", note: "El descuento se aplica a todo el pedido, también con artículos mixtos. Precios por unidad, envío aparte.", pcs: "uds.", plus: "+" },
  pl: { h2: "Rabaty ilościowe", qty: "Ilość", disc: "Rabat", kc: "Brelok", map: "Mapa 8 cm", reseller: "Dla resellerów", note: "Rabat obejmuje całe zamówienie, także różne wyroby. Ceny za sztukę, dostawa osobno.", pcs: "szt.", plus: "+" },
  ro: { h2: "Reduceri de volum", qty: "Cantitate", disc: "Reducere", kc: "Breloc", map: "Hartă 8 cm", reseller: "Revânzători", note: "Reducerea se aplică întregului tiraj și articolelor combinate într-o singură comandă. Prețuri pe bucată, livrarea Nova Poshta se plătește separat.", pcs: "buc.", plus: "+" },
};

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  const c = COPY[locale];
  const path = "/corporate";
  const languages: Record<string, string> = {};
  for (const l of locales) languages[localeMeta[l].htmlLang] = localeUrl(l, path);
  languages["x-default"] = localeUrl(defaultLocale, path);
  return {
    title: c.title,
    description: c.description,
    alternates: { canonical: localeUrl(locale, path), languages },
    openGraph: {
      title: c.title, description: c.description, url: localeUrl(locale, path),
      siteName: "Monadruk", type: "website", locale: localeMeta[locale].ogLocale,
      images: [`${BASE}/opengraph-image`],
    },
    twitter: { card: "summary_large_image", title: c.title, description: c.description, images: [`${BASE}/opengraph-image`] },
  };
}

export default async function CorporatePage({ params }: { params: { locale: string } }) {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  setRequestLocale(locale);
  const c = COPY[locale];
  const se = SEASON[locale];
  const tc = TIER_COPY[locale];
  const isUA = locale === "uk";
  const money = (uah: number) => (isUA ? `${uah} ₴` : `≈€${eur(uah)}`);

  const ld = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Service",
        name: c.h1,
        description: c.description,
        provider: { "@type": "Organization", name: "Monadruk", url: BASE },
        areaServed: ["UA"],
        url: localeUrl(locale, "/corporate"),
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale, "/") },
          { "@type": "ListItem", position: 2, name: c.h1, item: localeUrl(locale, "/corporate") },
        ],
      },
      {
        "@type": "FAQPage",
        mainEntity: c.faq.map((f) => ({ "@type": "Question", name: f.q, acceptedAnswer: { "@type": "Answer", text: f.a } })),
      },
    ],
  };

  return (
    <main id="main-content" tabIndex={-1} className="mx-auto max-w-[920px] px-5 py-14 lg:py-20">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      <h1 className="text-[clamp(28px,4vw,46px)] leading-tight">{c.h1}</h1>
      <p className="mt-4 max-w-[680px] text-[15px] leading-relaxed text-ink-2">{c.intro}</p>

      <section className="mt-8 flex flex-wrap gap-3">
        <a href="#b2b-form" className="inline-flex min-h-[44px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-5 py-2.5 text-sm font-semibold text-white transition hover:opacity-90">
          {c.cta}
        </a>
        <Link href="/keychains" className="inline-flex min-h-[44px] items-center justify-center rounded-[22px] border border-line-soft bg-white/80 px-5 py-2.5 text-sm font-semibold text-ink transition hover:border-[var(--accent)]">
          {c.ctaTry}
        </Link>
      </section>

      <section className="mt-12 rounded-[20px] border border-[rgba(142,107,61,0.35)] bg-[rgba(255,247,230,0.7)] px-5 py-5 sm:px-6" data-testid="corp-season">
        <h2 className="text-[20px] font-semibold">{se.h2}</h2>
        <p className="mt-2 max-w-[680px] text-[14.5px] leading-relaxed text-ink-2">{se.p}</p>
        <ul className="mt-3 flex flex-col gap-1.5">
          {se.points.map((pt) => (
            <li key={pt} className="flex gap-2 text-[14.5px] font-semibold text-ink"><span className="text-[var(--bronze,#8E6B3D)]">•</span>{pt}</li>
          ))}
        </ul>
      </section>

      <section className="mt-8">
        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-3">{se.photos}</p>
        <div className="mt-2 grid grid-cols-3 gap-2 sm:grid-cols-6">
          {[5, 6, 1, 7, 2, 9].map((i) => (
            // eslint-disable-next-line @next/next/no-img-element
            <img key={i} src={`/showcase/real-${i}.webp`} alt="" loading="lazy" decoding="async"
              className="aspect-square w-full rounded-[12px] border border-line-soft object-cover" />
          ))}
        </div>
      </section>

      <section className="mt-12">
        <h2 className="text-[20px] font-semibold">{c.h2what}</h2>
        <ul className="mt-4 grid gap-3 sm:grid-cols-2">
          {c.items.map((s) => (
            <li key={s.label} className="rounded-[18px] border border-line-soft bg-white/70 px-5 py-4">
              <p className="text-[15px] font-semibold text-ink">{s.label}</p>
              <p className="mt-1 text-[13.5px] leading-relaxed text-ink-2">{s.desc}</p>
              <p className="mt-2 text-[15px] font-semibold text-[var(--accent-strong)]">{s.price}</p>
            </li>
          ))}
        </ul>
      </section>

      <section className="mt-12" data-testid="corp-tiers">
        <h2 className="text-[20px] font-semibold">{tc.h2}</h2>
        <div className="mt-4 overflow-x-auto rounded-[18px] border border-line-soft bg-white/70">
          <table className="w-full min-w-[420px] text-left text-[14.5px]">
            <thead className="text-[12.5px] uppercase tracking-wide text-ink-3">
              <tr><th className="px-4 py-3">{tc.qty}</th><th className="px-4 py-3">{tc.disc}</th><th className="px-4 py-3">{tc.kc}</th><th className="px-4 py-3">{tc.map}</th></tr>
            </thead>
            <tbody>
              {TIERS.map((t) => (
                <tr key={t.min} className="border-t border-line-soft">
                  <td className="px-4 py-3 font-semibold text-ink">{t.max ? `${t.min}–${t.max}` : `${t.min}${tc.plus}`} {tc.pcs}</td>
                  <td className="px-4 py-3 font-semibold text-[var(--accent-strong)]">−{t.pct}%</td>
                  <td className="px-4 py-3 text-ink-2">{money(off(KEYCHAIN_PRICE_UAH, t.pct))}</td>
                  <td className="px-4 py-3 text-ink-2">{money(off(490, t.pct))}</td>
                </tr>
              ))}
              <tr className="border-t border-line-soft">
                <td className="px-4 py-3 font-semibold text-ink">{tc.reseller}</td>
                <td className="px-4 py-3 font-semibold text-[var(--accent-strong)]">−{RESELLER_PCT}%</td>
                <td className="px-4 py-3 text-ink-2">{money(off(KEYCHAIN_PRICE_UAH, RESELLER_PCT))}</td>
                <td className="px-4 py-3 text-ink-2">{money(off(490, RESELLER_PCT))}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-[13px] leading-relaxed text-ink-3">{tc.note}</p>
      </section>

      <section className="mt-12">
        <h2 className="text-[20px] font-semibold">{c.h2cases}</h2>
        <ul className="mt-4 grid gap-3 sm:grid-cols-2">
          {c.cases.map((s) => (
            <li key={s.h3} className="rounded-[18px] border border-line-soft bg-white/70 px-5 py-4">
              <h3 className="text-[15px] font-semibold text-ink">{s.h3}</h3>
              <p className="mt-1 text-[13.5px] leading-relaxed text-ink-2">{s.p}</p>
            </li>
          ))}
        </ul>
      </section>

      <section className="mt-12 max-w-[680px]">
        <h2 className="text-[20px] font-semibold">{c.h2how}</h2>
        <ol className="mt-4 flex flex-col gap-2.5">
          {c.how.map((step, i) => (
            <li key={i} className="flex gap-3 text-[15px] leading-relaxed text-ink-2">
              <span className="mt-0.5 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[rgba(31,51,40,0.10)] text-[13px] font-bold text-[var(--accent-strong)]">{i + 1}</span>
              {step}
            </li>
          ))}
        </ol>
      </section>

      <section className="mt-12">
        <CorporateLeadForm />
      </section>

      <section className="mt-12 max-w-[680px]">
        <h2 className="text-[20px] font-semibold">{c.h2faq}</h2>
        <dl className="mt-4 flex flex-col gap-4">
          {c.faq.map((f) => (
            <div key={f.q}>
              <dt className="text-[15px] font-semibold text-ink">{f.q}</dt>
              <dd className="mt-1.5 text-[14.5px] leading-relaxed text-ink-2">{f.a}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="mt-10">
        <a href="#b2b-form" className="inline-flex min-h-[44px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-5 py-2.5 text-sm font-semibold text-white transition hover:opacity-90">
          {c.cta}
        </a>
      </section>
    </main>
  );
}
