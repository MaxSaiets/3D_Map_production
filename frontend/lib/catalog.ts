// ──────────────────────────────────────────────────────────────────────────
// ЄДИНЕ ДЖЕРЕЛО вмісту публічної сторінки «Ціни / Каталог» (/prices).
// Ціни беруться з lib/mapPrices.ts (одна таблиця → UI + SEO + ця сторінка не
// «дрейфують»). Тексти — 6 мовами (як lib/legal/content.ts). Сторінка повністю
// серверна, БЕЗ WebGL/квоти — щоб модератор LiqPay і пошуковий бот бачили
// назви + описи + ЦІНИ В ГРИВНЯХ у простому HTML (вимога активації мерчанта).
// ──────────────────────────────────────────────────────────────────────────
import {
  MAP_SIZE_PRICES_UAH,
  MAP_MAGNET_PRICE_UAH,
  KEYCHAIN_PRICE_UAH,
  MAP_RELIEF_ADDON_UAH,
  mapPriceEur,
} from "@/lib/mapPrices";

export type CatalogItem = {
  name: string;
  desc: string;
  uah: number;
  /** "from" → «від N ₴»; "addon" → «+N ₴»; "fixed" (default) → «N ₴». */
  kind?: "from" | "addon" | "fixed";
};
export type CatalogCategory = { title: string; items: CatalogItem[] };

export type CatalogFaqItem = { q: string; a: string };
export type Catalog = {
  metaTitle: string;
  metaDescription: string;
  h1: string;
  intro: string;
  categories: CatalogCategory[];
  notesTitle: string;
  notes: string[];
  sellerTitle: string;
  sellerName: string; // identity label, value from BUSINESS
  docsIntro: string;
  docs: { offer: string; delivery: string; refund: string; contacts: string };
  ctaLabel: string;
  faqTitle: string;
  faq: CatalogFaqItem[];
};

const P = MAP_SIZE_PRICES_UAH;

// Спільні (мовнонезалежні) ціни — щоб не дублювати числа в кожній локалі.
const PR = {
  keychain: KEYCHAIN_PRICE_UAH,
  s: P[55],
  m: P[80],
  l: P[110],
  xl: P[150],
  magnet: MAP_MAGNET_PRICE_UAH,
  relief: MAP_RELIEF_ADDON_UAH,
};

const uk: Catalog = {
  metaTitle: "Ціни на 3D-мапи, магніти та брелоки",
  metaDescription:
    "Актуальні ціни в гривнях: 3D-мапа міста від 250 ₴ (S/M/L/XL), магніт-мапа 150 ₴, брелок-мапа від 120 ₴, рельєф +60 ₴. Друк з Eco PLA, доставка Новою Поштою.",
  h1: "Ціни",
  intro:
    "Ціна вказана за готовий виріб (3D-друк з біопластику Eco PLA). Доставка оплачується окремо за тарифом перевізника. Оплата — карткою Visa / Mastercard онлайн або при отриманні.",
  categories: [
    {
      title: "3D-мапи міст",
      items: [
        { name: "3D-мапа міста — S (≈5,5 см)", desc: "Друкована 3D-модель ділянки міста, ребро ~5,5 см.", uah: PR.s },
        { name: "3D-мапа міста — M (≈8 см)", desc: "Друкована 3D-модель ділянки міста, ребро ~8 см.", uah: PR.m },
        { name: "3D-мапа міста — L (≈11 см)", desc: "Друкована 3D-модель ділянки міста, ребро ~11 см.", uah: PR.l },
        { name: "3D-мапа міста — XL (≈15 см)", desc: "Друкована 3D-модель ділянки міста, ребро ~15 см.", uah: PR.xl },
        { name: "Рельєф місцевості (опція)", desc: "Додаткові висоти ландшафту на будь-якій 3D-мапі.", uah: PR.relief, kind: "addon" },
      ],
    },
    {
      title: "Магніти",
      items: [
        { name: "Магніт-мапа на холодильник (≈6 см)", desc: "Плаский магніт із 3D-мапою ділянки міста.", uah: PR.magnet },
      ],
    },
    {
      title: "Брелоки-мапи",
      items: [
        { name: "Брелок-мапа (3D-друк)", desc: "Брелок із 3D-мапою ділянки міста або маршруту (GPX), Eco PLA.", uah: PR.keychain, kind: "from" },
      ],
    },
    {
      title: "Цифрові файли",
      items: [
        { name: "Файл 3MF / STL для самостійного друку", desc: "Готовий файл моделі. Безкоштовно в межах ліміту акаунта (5 завантажень), далі — за домовленістю.", uah: 0, kind: "from" },
      ],
    },
  ],
  notesTitle: "Умови",
  notes: [
    "Усі ціни — у гривнях (₴), за один виріб.",
    "Доставка — окремо, за тарифом перевізника (Нова Пошта / Укрпошта).",
    "Оплата — карткою Visa / Mastercard онлайн (LiqPay) або при отриманні (накладений платіж).",
    "Вироби виготовляються на індивідуальне замовлення; терміни — 2–4 робочі дні + доставка.",
  ],
  sellerTitle: "Продавець",
  sellerName: "Продавець",
  docsIntro: "Замовлення регулюється договором публічної оферти. Деталі:",
  docs: { offer: "Договір публічної оферти", delivery: "Оплата і доставка", refund: "Повернення та обмін", contacts: "Контакти" },
  ctaLabel: "Створити свою мапу",
  faqTitle: "Часті запитання",
  faq: [
    { q: "Скільки триває виготовлення?", a: "2–4 робочі дні на друк, потім доставка Новою Поштою по Україні." },
    { q: "Чи є знижки для великих замовлень?", a: "Так — для тиражів від 5 однакових виробів (наприклад, корпоративні брелоки) вартість узгоджується окремо, напишіть нам." },
    { q: "Що входить у ціну?", a: "Ціна — за готовий надрукований виріб з Eco PLA. Доставка та рельєф місцевості (+60 ₴) оплачуються окремо." },
    { q: "Чи можна оплатити при отриманні?", a: "Так, крім оплати карткою онлайн через LiqPay доступний накладений платіж при отриманні." },
  ],
};

const en: Catalog = {
  metaTitle: "Prices for 3D city maps, magnets & keychains",
  metaDescription:
    "Current prices in UAH: 3D city map from 250 ₴ (S/M/L/XL), fridge magnet 150 ₴, map keychain from 120 ₴, relief +60 ₴. Eco PLA print, delivery by Nova Poshta.",
  h1: "Prices",
  intro:
    "The price is for the finished item (3D-printed in Eco PLA bioplastic). Delivery is paid separately at the carrier's tariff. Payment by Visa / Mastercard online or on delivery.",
  categories: [
    {
      title: "3D city maps",
      items: [
        { name: "3D city map — S (≈5.5 cm)", desc: "Printed 3D model of a city area, ~5.5 cm edge.", uah: PR.s },
        { name: "3D city map — M (≈8 cm)", desc: "Printed 3D model of a city area, ~8 cm edge.", uah: PR.m },
        { name: "3D city map — L (≈11 cm)", desc: "Printed 3D model of a city area, ~11 cm edge.", uah: PR.l },
        { name: "3D city map — XL (≈15 cm)", desc: "Printed 3D model of a city area, ~15 cm edge.", uah: PR.xl },
        { name: "Terrain relief (option)", desc: "Extra landscape elevation on any 3D map.", uah: PR.relief, kind: "addon" },
      ],
    },
    {
      title: "Magnets",
      items: [
        { name: "Fridge magnet map (≈6 cm)", desc: "Flat magnet with a 3D map of a city area.", uah: PR.magnet },
      ],
    },
    {
      title: "Map keychains",
      items: [
        { name: "Map keychain (3D print)", desc: "Keychain with a 3D map of a city area or a route (GPX), Eco PLA.", uah: PR.keychain, kind: "from" },
      ],
    },
    {
      title: "Digital files",
      items: [
        { name: "3MF / STL file for self-printing", desc: "Ready model file. Free within your account limit (5 downloads), then by arrangement.", uah: 0, kind: "from" },
      ],
    },
  ],
  notesTitle: "Terms",
  notes: [
    "All prices are in Ukrainian hryvnia (₴), per item.",
    "Delivery is charged separately at the carrier's tariff (Nova Poshta / Ukrposhta).",
    "Payment by Visa / Mastercard online (LiqPay) or cash on delivery.",
    "Items are made to order; lead time 1–3 business days plus shipping.",
  ],
  sellerTitle: "Seller",
  sellerName: "Seller",
  docsIntro: "Orders are governed by the public offer agreement. Details:",
  docs: { offer: "Public offer agreement", delivery: "Payment & delivery", refund: "Returns & refunds", contacts: "Contacts" },
  ctaLabel: "Create your map",
  faqTitle: "FAQ",
  faq: [
    { q: "How long does production take?", a: "1–3 business days to print, then delivery across Ukraine." },
    { q: "Are there discounts for bulk orders?", a: "Yes — for runs of 5+ identical items (e.g. corporate keychains) pricing is agreed individually, just message us." },
    { q: "What's included in the price?", a: "The price covers the finished item printed in Eco PLA. Delivery and terrain relief (+≈€1.5) are charged separately." },
    { q: "Can I pay on delivery?", a: "Yes, besides online card payment via LiqPay, cash on delivery is available." },
  ],
};

const de: Catalog = {
  metaTitle: "Preise für 3D-Stadtkarten, Magnete & Schlüsselanhänger",
  metaDescription:
    "Aktuelle Preise in UAH: 3D-Stadtkarte ab 250 ₴ (S/M/L/XL), Kühlschrankmagnet 150 ₴, Karten-Schlüsselanhänger ab 120 ₴, Relief +60 ₴. Eco-PLA-Druck.",
  h1: "Preise",
  intro:
    "Der Preis gilt für das fertige Produkt (3D-Druck aus Eco-PLA-Biokunststoff). Der Versand wird separat zum Tarif des Zustellers berechnet. Zahlung per Visa / Mastercard online oder bei Lieferung.",
  categories: [
    {
      title: "3D-Stadtkarten",
      items: [
        { name: "3D-Stadtkarte — S (≈5,5 cm)", desc: "Gedrucktes 3D-Modell eines Stadtgebiets, Kante ~5,5 cm.", uah: PR.s },
        { name: "3D-Stadtkarte — M (≈8 cm)", desc: "Gedrucktes 3D-Modell eines Stadtgebiets, Kante ~8 cm.", uah: PR.m },
        { name: "3D-Stadtkarte — L (≈11 cm)", desc: "Gedrucktes 3D-Modell eines Stadtgebiets, Kante ~11 cm.", uah: PR.l },
        { name: "3D-Stadtkarte — XL (≈15 cm)", desc: "Gedrucktes 3D-Modell eines Stadtgebiets, Kante ~15 cm.", uah: PR.xl },
        { name: "Geländerelief (Option)", desc: "Zusätzliche Geländehöhen auf jeder 3D-Karte.", uah: PR.relief, kind: "addon" },
      ],
    },
    {
      title: "Magnete",
      items: [
        { name: "Kühlschrankmagnet-Karte (≈6 cm)", desc: "Flacher Magnet mit einer 3D-Karte eines Stadtgebiets.", uah: PR.magnet },
      ],
    },
    {
      title: "Karten-Schlüsselanhänger",
      items: [
        { name: "Karten-Schlüsselanhänger (3D-Druck)", desc: "Anhänger mit 3D-Karte eines Stadtgebiets oder einer Route (GPX), Eco PLA.", uah: PR.keychain, kind: "from" },
      ],
    },
    {
      title: "Digitale Dateien",
      items: [
        { name: "3MF-/STL-Datei zum Selbstdrucken", desc: "Fertige Modelldatei. Kostenlos im Rahmen Ihres Kontolimits (5 Downloads), danach nach Vereinbarung.", uah: 0, kind: "from" },
      ],
    },
  ],
  notesTitle: "Bedingungen",
  notes: [
    "Alle Preise sind in ukrainischen Hrywnja (₴), pro Stück.",
    "Der Versand wird separat zum Tarif des Zustellers berechnet (Nova Poshta / Ukrposhta).",
    "Zahlung per Visa / Mastercard online (LiqPay) oder per Nachnahme.",
    "Die Artikel werden auf Bestellung gefertigt; Bearbeitungszeit 1–3 Werktage zzgl. Versand.",
  ],
  sellerTitle: "Verkäufer",
  sellerName: "Verkäufer",
  docsIntro: "Bestellungen unterliegen dem öffentlichen Angebotsvertrag. Details:",
  docs: { offer: "Öffentlicher Angebotsvertrag", delivery: "Zahlung & Versand", refund: "Rückgabe & Umtausch", contacts: "Kontakte" },
  ctaLabel: "Eigene Karte erstellen",
  faqTitle: "Häufige Fragen",
  faq: [
    { q: "Wie lange dauert die Herstellung?", a: "1–3 Werktage Druckzeit, danach Versand innerhalb der Ukraine." },
    { q: "Gibt es Rabatte für größere Bestellungen?", a: "Ja — bei 5 oder mehr identischen Stücken (z. B. Firmen-Schlüsselanhänger) wird der Preis individuell vereinbart." },
    { q: "Was ist im Preis enthalten?", a: "Der Preis gilt für das fertige Eco-PLA-Produkt. Versand und Geländerelief (+≈1,5 €) werden separat berechnet." },
    { q: "Kann ich bei Lieferung bezahlen?", a: "Ja, neben Online-Zahlung per LiqPay ist auch Nachnahme möglich." },
  ],
};

const es: Catalog = {
  metaTitle: "Precios de mapas 3D, imanes y llaveros",
  metaDescription:
    "Precios actuales en UAH: mapa 3D de ciudad desde 250 ₴ (S/M/L/XL), imán 150 ₴, llavero-mapa desde 120 ₴, relieve +60 ₴. Impresión en Eco PLA.",
  h1: "Precios",
  intro:
    "El precio corresponde al producto terminado (impreso en 3D con bioplástico Eco PLA). El envío se paga aparte según la tarifa del transportista. Pago con Visa / Mastercard en línea o contra entrega.",
  categories: [
    {
      title: "Mapas 3D de ciudades",
      items: [
        { name: "Mapa 3D de ciudad — S (≈5,5 cm)", desc: "Modelo 3D impreso de una zona urbana, borde ~5,5 cm.", uah: PR.s },
        { name: "Mapa 3D de ciudad — M (≈8 cm)", desc: "Modelo 3D impreso de una zona urbana, borde ~8 cm.", uah: PR.m },
        { name: "Mapa 3D de ciudad — L (≈11 cm)", desc: "Modelo 3D impreso de una zona urbana, borde ~11 cm.", uah: PR.l },
        { name: "Mapa 3D de ciudad — XL (≈15 cm)", desc: "Modelo 3D impreso de una zona urbana, borde ~15 cm.", uah: PR.xl },
        { name: "Relieve del terreno (opción)", desc: "Altitudes adicionales del paisaje en cualquier mapa 3D.", uah: PR.relief, kind: "addon" },
      ],
    },
    {
      title: "Imanes",
      items: [
        { name: "Imán-mapa de nevera (≈6 cm)", desc: "Imán plano con un mapa 3D de una zona urbana.", uah: PR.magnet },
      ],
    },
    {
      title: "Llaveros-mapa",
      items: [
        { name: "Llavero-mapa (impresión 3D)", desc: "Llavero con mapa 3D de una zona urbana o una ruta (GPX), Eco PLA.", uah: PR.keychain, kind: "from" },
      ],
    },
    {
      title: "Archivos digitales",
      items: [
        { name: "Archivo 3MF / STL para imprimir tú mismo", desc: "Archivo de modelo listo. Gratis dentro del límite de tu cuenta (5 descargas), luego según acuerdo.", uah: 0, kind: "from" },
      ],
    },
  ],
  notesTitle: "Condiciones",
  notes: [
    "Todos los precios están en grivnas ucranianas (₴), por unidad.",
    "El envío se cobra aparte según la tarifa del transportista (Nova Poshta / Ukrposhta).",
    "Pago con Visa / Mastercard en línea (LiqPay) o contra reembolso.",
    "Los artículos se fabrican por encargo; plazo 1–3 días hábiles más envío.",
  ],
  sellerTitle: "Vendedor",
  sellerName: "Vendedor",
  docsIntro: "Los pedidos se rigen por el contrato de oferta pública. Detalles:",
  docs: { offer: "Contrato de oferta pública", delivery: "Pago y envío", refund: "Devoluciones y cambios", contacts: "Contactos" },
  ctaLabel: "Crea tu mapa",
  faqTitle: "Preguntas frecuentes",
  faq: [
    { q: "¿Cuánto tarda la fabricación?", a: "1–3 días hábiles de impresión, luego envío por Ucrania." },
    { q: "¿Hay descuentos para pedidos grandes?", a: "Sí — para tandas de 5 o más piezas idénticas (por ejemplo, llaveros corporativos) el precio se acuerda por separado." },
    { q: "¿Qué incluye el precio?", a: "El precio corresponde al producto terminado en Eco PLA. El envío y el relieve del terreno (+≈1,5 €) se cobran aparte." },
    { q: "¿Puedo pagar contra entrega?", a: "Sí, además del pago con tarjeta online vía LiqPay, está disponible el pago contra reembolso." },
  ],
};

const fr: Catalog = {
  metaTitle: "Prix des cartes 3D, aimants et porte-clés",
  metaDescription:
    "Prix actuels en UAH : carte 3D de ville dès 250 ₴ (S/M/L/XL), aimant 150 ₴, porte-clés carte dès 120 ₴, relief +60 ₴. Impression en Eco PLA.",
  h1: "Tarifs",
  intro:
    "Le prix concerne le produit fini (imprimé en 3D en bioplastique Eco PLA). La livraison est facturée séparément au tarif du transporteur. Paiement par Visa / Mastercard en ligne ou à la livraison.",
  categories: [
    {
      title: "Cartes 3D de villes",
      items: [
        { name: "Carte 3D de ville — S (≈5,5 cm)", desc: "Modèle 3D imprimé d'une zone urbaine, arête ~5,5 cm.", uah: PR.s },
        { name: "Carte 3D de ville — M (≈8 cm)", desc: "Modèle 3D imprimé d'une zone urbaine, arête ~8 cm.", uah: PR.m },
        { name: "Carte 3D de ville — L (≈11 cm)", desc: "Modèle 3D imprimé d'une zone urbaine, arête ~11 cm.", uah: PR.l },
        { name: "Carte 3D de ville — XL (≈15 cm)", desc: "Modèle 3D imprimé d'une zone urbaine, arête ~15 cm.", uah: PR.xl },
        { name: "Relief du terrain (option)", desc: "Altitudes supplémentaires du paysage sur toute carte 3D.", uah: PR.relief, kind: "addon" },
      ],
    },
    {
      title: "Aimants",
      items: [
        { name: "Aimant-carte de frigo (≈6 cm)", desc: "Aimant plat avec une carte 3D d'une zone urbaine.", uah: PR.magnet },
      ],
    },
    {
      title: "Porte-clés carte",
      items: [
        { name: "Porte-clés carte (impression 3D)", desc: "Porte-clés avec carte 3D d'une zone urbaine ou d'un itinéraire (GPX), Eco PLA.", uah: PR.keychain, kind: "from" },
      ],
    },
    {
      title: "Fichiers numériques",
      items: [
        { name: "Fichier 3MF / STL à imprimer soi-même", desc: "Fichier de modèle prêt. Gratuit dans la limite de votre compte (5 téléchargements), puis sur accord.", uah: 0, kind: "from" },
      ],
    },
  ],
  notesTitle: "Conditions",
  notes: [
    "Tous les prix sont en hryvnia ukrainienne (₴), par article.",
    "La livraison est facturée séparément au tarif du transporteur (Nova Poshta / Ukrposhta ).",
    "Paiement par Visa / Mastercard en ligne (LiqPay) ou à la livraison.",
    "Les articles sont fabriqués sur commande ; délai 1–3 jours ouvrés plus expédition.",
  ],
  sellerTitle: "Vendeur",
  sellerName: "Vendeur",
  docsIntro: "Les commandes sont régies par le contrat d'offre publique. Détails :",
  docs: { offer: "Contrat d'offre publique", delivery: "Paiement et livraison", refund: "Retours et remboursements", contacts: "Contacts" },
  ctaLabel: "Créer votre carte",
  faqTitle: "Questions fréquentes",
  faq: [
    { q: "Combien de temps prend la fabrication ?", a: "1 à 3 jours ouvrés d'impression, puis livraison en Ukraine." },
    { q: "Y a-t-il des remises pour les grandes commandes ?", a: "Oui — pour 5 pièces identiques ou plus (porte-clés d'entreprise par exemple), le prix se négocie séparément." },
    { q: "Qu'est-ce qui est inclus dans le prix ?", a: "Le prix concerne le produit fini en Eco PLA. La livraison et le relief du terrain (+≈1,5 €) sont facturés à part." },
    { q: "Puis-je payer à la livraison ?", a: "Oui, en plus du paiement en ligne par carte via LiqPay, le paiement à la livraison est disponible." },
  ],
};

const pl: Catalog = {
  metaTitle: "Ceny map 3D, magnesów i breloków",
  metaDescription:
    "Aktualne ceny w UAH: mapa 3D miasta od 250 ₴ (S/M/L/XL), magnes 150 ₴, brelok-mapa od 120 ₴, relief +60 ₴. Druk z Eco PLA, dostawa Nową Pocztą.",
  h1: "Cennik",
  intro:
    "Cena dotyczy gotowego produktu (druk 3D z biotworzywa Eco PLA). Dostawa płatna osobno według taryfy przewoźnika. Płatność kartą Visa / Mastercard online lub przy odbiorze.",
  categories: [
    {
      title: "Mapy 3D miast",
      items: [
        { name: "Mapa 3D miasta — S (≈5,5 cm)", desc: "Drukowany model 3D fragmentu miasta, krawędź ~5,5 cm.", uah: PR.s },
        { name: "Mapa 3D miasta — M (≈8 cm)", desc: "Drukowany model 3D fragmentu miasta, krawędź ~8 cm.", uah: PR.m },
        { name: "Mapa 3D miasta — L (≈11 cm)", desc: "Drukowany model 3D fragmentu miasta, krawędź ~11 cm.", uah: PR.l },
        { name: "Mapa 3D miasta — XL (≈15 cm)", desc: "Drukowany model 3D fragmentu miasta, krawędź ~15 cm.", uah: PR.xl },
        { name: "Relief terenu (opcja)", desc: "Dodatkowe wysokości krajobrazu na dowolnej mapie 3D.", uah: PR.relief, kind: "addon" },
      ],
    },
    {
      title: "Magnesy",
      items: [
        { name: "Magnes-mapa na lodówkę (≈6 cm)", desc: "Płaski magnes z mapą 3D fragmentu miasta.", uah: PR.magnet },
      ],
    },
    {
      title: "Breloki-mapy",
      items: [
        { name: "Brelok-mapa (druk 3D)", desc: "Brelok z mapą 3D fragmentu miasta lub trasy (GPX), Eco PLA.", uah: PR.keychain, kind: "from" },
      ],
    },
    {
      title: "Pliki cyfrowe",
      items: [
        { name: "Plik 3MF / STL do samodzielnego druku", desc: "Gotowy plik modelu. Bezpłatnie w ramach limitu konta (5 pobrań), dalej po uzgodnieniu.", uah: 0, kind: "from" },
      ],
    },
  ],
  notesTitle: "Warunki",
  notes: [
    "Wszystkie ceny są w hrywnach ukraińskich (₴), za sztukę.",
    "Dostawa naliczana osobno według taryfy przewoźnika (Nova Poshta / Ukrposhta).",
    "Płatność kartą Visa / Mastercard online (LiqPay) lub za pobraniem.",
    "Produkty wykonywane na zamówienie; czas realizacji 1–3 dni robocze plus wysyłka.",
  ],
  sellerTitle: "Sprzedawca",
  sellerName: "Sprzedawca",
  docsIntro: "Zamówienia reguluje umowa oferty publicznej. Szczegóły:",
  docs: { offer: "Umowa oferty publicznej", delivery: "Płatność i dostawa", refund: "Zwroty i wymiana", contacts: "Kontakt" },
  ctaLabel: "Stwórz swoją mapę",
  faqTitle: "Częste pytania",
  faq: [
    { q: "Ile trwa wykonanie?", a: "1–3 dni robocze druku, potem dostawa po Ukrainie." },
    { q: "Czy są rabaty przy większych zamówieniach?", a: "Tak — przy 5 i więcej identycznych sztukach (np. breloki firmowe) cena ustalana jest indywidualnie." },
    { q: "Co zawiera cena?", a: "Cena dotyczy gotowego produktu z Eco PLA. Dostawa i relief terenu (+≈1,5 €) są płatne osobno." },
    { q: "Czy mogę zapłacić przy odbiorze?", a: "Tak, oprócz płatności kartą online przez LiqPay dostępna jest płatność za pobraniem." },
  ],
};

const CATALOGS: Record<string, Catalog> = { uk, en, de, es, fr, pl };

export function getCatalog(locale: string): Catalog {
  return CATALOGS[locale] ?? uk;
}

// Локалізовані слова цінника (спільні для /prices і price-band на сторінках міст).
export const PRICE_WORDS: Record<string, { from: string; free: string }> = {
  uk: { from: "від", free: "Безкоштовно*" },
  en: { from: "from", free: "Free*" },
  de: { from: "ab", free: "Kostenlos*" },
  es: { from: "desde", free: "Gratis*" },
  fr: { from: "dès", free: "Gratuit*" },
  pl: { from: "od", free: "Bezpłatnie*" },
};

/** Єдине форматування ціни товару: «N ₴» (uk) / «N ₴ · ≈M €» (EU); «+N ₴» (addon);
 *  «від N ₴» (from); «Безкоштовно*» (uah=0). Спільне для /prices і сторінок міст. */
export function formatCatalogPrice(uah: number, kind: string | undefined, locale: string): string {
  const w = PRICE_WORDS[locale] ?? PRICE_WORDS.uk;
  if (uah === 0) return w.free;
  if (kind === "addon") return `+${uah} ₴`;
  const eur = locale !== "uk" ? ` · ≈${mapPriceEur(uah)} €` : "";
  const base = `${uah} ₴${eur}`;
  return kind === "from" ? `${w.from} ${base}` : base;
}
