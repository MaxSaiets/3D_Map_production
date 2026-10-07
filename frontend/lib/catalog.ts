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
  FILE_PRICE_UAH,
  mapPriceEur,
} from "@/lib/mapPrices";
import { SUB_PRICE } from "@/lib/legal/subscription";

export type CatalogItem = {
  name: string;
  desc: string;
  uah: number;
  /** "from" → «від N ₴»; "addon" → «+N ₴»; "monthly" → «N ₴ / міс»; "fixed" (default) → «N ₴». */
  kind?: "from" | "addon" | "fixed" | "monthly";
  /** Ціна в доларах для іноземців (підписка списується в USD, а не за курсом). */
  usd?: number;
  /** Сторінка з деталями (рядок стає посиланням на /prices). */
  href?: string;
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
  /** Плашка Monadruk Pro на /prices (для тих, хто друкує багато або на продаж). */
  pro: { eyebrow: string; title: string; text: string; cta: string };
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
  // 07.10.2026: файл давно платний (149 ₴, FREE_DOWNLOADS=0 на проді), а каталог —
  // і /prices, і ~1700 сторінок міст — досі писав «Безкоштовно* (5 завантажень)».
  file: FILE_PRICE_UAH,
  pro: SUB_PRICE.UAH,
  proUsd: SUB_PRICE.USD,
};

const uk: Catalog = {
  metaTitle: "Ціни на 3D-мапи, магніти та брелоки",
  metaDescription:
    "Актуальні ціни в гривнях: 3D-мапа міста від 350 ₴ (S/M/L/XL), магніт-мапа 210 ₴, брелок-мапа від 170 ₴, рельєф +85 ₴. Друк з Eco PLA, доставка Новою Поштою.",
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
        { name: "Файл 3MF / STL для самостійного друку", desc: "Готовий 3MF з поділом на кольори — відкривається в Bambu Studio чи PrusaSlicer. Для особистого друку; 3D-превʼю перед покупкою безкоштовне.", uah: PR.file },
        { name: "Monadruk Pro — безлім файлів", desc: "Необмежені файли 3MF/STL усіх моделей і комерційна ліцензія: друкуйте й продавайте вироби. Скасування в один клік.", uah: PR.pro, usd: PR.proUsd, kind: "monthly", href: "/pro" },
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
  pro: { eyebrow: "Для 3D-друкарень і продавців", title: "Друкуєте багато або на продаж?", text: "Monadruk Pro — безлім файлів 3MF/STL і комерційна ліцензія на надруковані вироби. Окупається вже з {n}-го файлу на місяць; скасування в один клік.", cta: "Детальніше про Pro" },
  faqTitle: "Часті запитання",
  faq: [
    { q: "Скільки триває виготовлення?", a: "2–4 робочі дні на друк, потім доставка Новою Поштою по Україні." },
    { q: "Чи є знижки для великих замовлень?", a: "Так — для тиражів від 5 однакових виробів (наприклад, корпоративні брелоки) вартість узгоджується окремо, напишіть нам." },
    { q: "Що входить у ціну?", a: "Ціна — за готовий надрукований виріб з Eco PLA. Доставка та рельєф місцевості (+85 ₴) оплачуються окремо." },
    { q: "Чи можна оплатити при отриманні?", a: "Так, крім оплати карткою онлайн через LiqPay доступний накладений платіж при отриманні." },
  ],
};

const en: Catalog = {
  metaTitle: "Prices for 3D city maps, magnets & keychains",
  metaDescription:
    "Current prices in UAH: 3D city map from 350 ₴ (S/M/L/XL), fridge magnet 210 ₴, map keychain from 170 ₴, relief +85 ₴. Eco PLA print, delivery by Nova Poshta.",
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
        { name: "3MF / STL file for self-printing", desc: "Ready colour-split 3MF that opens in Bambu Studio or PrusaSlicer. For personal printing; the 3D preview before buying is free.", uah: PR.file },
        { name: "Monadruk Pro — unlimited files", desc: "Unlimited 3MF/STL files for every model plus a commercial licence: print and sell the items. Cancel in one click.", uah: PR.pro, usd: PR.proUsd, kind: "monthly", href: "/pro" },
      ],
    },
  ],
  notesTitle: "Terms",
  notes: [
    "All prices are in Ukrainian hryvnia (₴), per item.",
    "Delivery is charged separately at the carrier's tariff (Nova Poshta / Ukrposhta).",
    "Payment by Visa / Mastercard online (LiqPay) or cash on delivery.",
    "Items are made to order; lead time 2–4 business days plus shipping.",
  ],
  sellerTitle: "Seller",
  sellerName: "Seller",
  docsIntro: "Orders are governed by the public offer agreement. Details:",
  docs: { offer: "Public offer agreement", delivery: "Payment & delivery", refund: "Returns & refunds", contacts: "Contacts" },
  ctaLabel: "Create your map",
  pro: { eyebrow: "For print shops and sellers", title: "Printing a lot, or printing to sell?", text: "Monadruk Pro gives you unlimited 3MF/STL files and a commercial licence for printed items. It pays off from about the {n}th file a month; cancel in one click.", cta: "Learn about Pro" },
  faqTitle: "FAQ",
  faq: [
    { q: "How long does production take?", a: "2–4 business days to print, then delivery across Ukraine." },
    { q: "Are there discounts for bulk orders?", a: "Yes — for runs of 5+ identical items (e.g. corporate keychains) pricing is agreed individually, just message us." },
    { q: "What's included in the price?", a: "The price covers the finished item printed in Eco PLA. Delivery and terrain relief (+≈€2) are charged separately." },
    { q: "Can I pay on delivery?", a: "Yes, besides online card payment via LiqPay, cash on delivery is available." },
  ],
};

const de: Catalog = {
  metaTitle: "Preise für 3D-Stadtkarten, Magnete & Schlüsselanhänger",
  metaDescription:
    "Aktuelle Preise in UAH: 3D-Stadtkarte ab 350 ₴ (S/M/L/XL), Kühlschrankmagnet 210 ₴, Karten-Schlüsselanhänger ab 170 ₴, Relief +85 ₴. Eco-PLA-Druck.",
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
        { name: "3MF-/STL-Datei zum Selbstdrucken", desc: "Fertige, farbgetrennte 3MF-Datei für Bambu Studio oder PrusaSlicer. Für den privaten Druck; die 3D-Vorschau vor dem Kauf ist kostenlos.", uah: PR.file },
        { name: "Monadruk Pro — unbegrenzte Dateien", desc: "Unbegrenzte 3MF/STL-Dateien aller Modelle und kommerzielle Lizenz: drucken und verkaufen. Kündigung mit einem Klick.", uah: PR.pro, usd: PR.proUsd, kind: "monthly", href: "/pro" },
      ],
    },
  ],
  notesTitle: "Bedingungen",
  notes: [
    "Alle Preise sind in ukrainischen Hrywnja (₴), pro Stück.",
    "Der Versand wird separat zum Tarif des Zustellers berechnet (Nova Poshta / Ukrposhta).",
    "Zahlung per Visa / Mastercard online (LiqPay) oder per Nachnahme.",
    "Die Artikel werden auf Bestellung gefertigt; Bearbeitungszeit 2–4 Werktage zzgl. Versand.",
  ],
  sellerTitle: "Verkäufer",
  sellerName: "Verkäufer",
  docsIntro: "Bestellungen unterliegen dem öffentlichen Angebotsvertrag. Details:",
  docs: { offer: "Öffentlicher Angebotsvertrag", delivery: "Zahlung & Versand", refund: "Rückgabe & Umtausch", contacts: "Kontakte" },
  ctaLabel: "Eigene Karte erstellen",
  pro: { eyebrow: "Für Druckereien und Verkäufer", title: "Drucken Sie viel oder für den Verkauf?", text: "Monadruk Pro: unbegrenzte 3MF/STL-Dateien und kommerzielle Lizenz für gedruckte Stücke. Lohnt sich ab etwa der {n}. Datei im Monat; Kündigung mit einem Klick.", cta: "Mehr über Pro" },
  faqTitle: "Häufige Fragen",
  faq: [
    { q: "Wie lange dauert die Herstellung?", a: "2–4 Werktage Druckzeit, danach Versand innerhalb der Ukraine." },
    { q: "Gibt es Rabatte für größere Bestellungen?", a: "Ja — bei 5 oder mehr identischen Stücken (z. B. Firmen-Schlüsselanhänger) wird der Preis individuell vereinbart." },
    { q: "Was ist im Preis enthalten?", a: "Der Preis gilt für das fertige Eco-PLA-Produkt. Versand und Geländerelief (+≈2 €) werden separat berechnet." },
    { q: "Kann ich bei Lieferung bezahlen?", a: "Ja, neben Online-Zahlung per LiqPay ist auch Nachnahme möglich." },
  ],
};

const es: Catalog = {
  metaTitle: "Precios de mapas 3D, imanes y llaveros",
  metaDescription:
    "Precios actuales en UAH: mapa 3D de ciudad desde 350 ₴ (S/M/L/XL), imán 210 ₴, llavero-mapa desde 170 ₴, relieve +85 ₴. Impresión en Eco PLA.",
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
        { name: "Archivo 3MF / STL para imprimir tú mismo", desc: "3MF listo y separado por colores que se abre en Bambu Studio o PrusaSlicer. Para impresión personal; la vista previa 3D antes de comprar es gratis.", uah: PR.file },
        { name: "Monadruk Pro — archivos ilimitados", desc: "Archivos 3MF/STL ilimitados de todos los modelos y licencia comercial: imprime y vende. Cancelación en un clic.", uah: PR.pro, usd: PR.proUsd, kind: "monthly", href: "/pro" },
      ],
    },
  ],
  notesTitle: "Condiciones",
  notes: [
    "Todos los precios están en grivnas ucranianas (₴), por unidad.",
    "El envío se cobra aparte según la tarifa del transportista (Nova Poshta / Ukrposhta).",
    "Pago con Visa / Mastercard en línea (LiqPay) o contra reembolso.",
    "Los artículos se fabrican por encargo; plazo 2–4 días hábiles más envío.",
  ],
  sellerTitle: "Vendedor",
  sellerName: "Vendedor",
  docsIntro: "Los pedidos se rigen por el contrato de oferta pública. Detalles:",
  docs: { offer: "Contrato de oferta pública", delivery: "Pago y envío", refund: "Devoluciones y cambios", contacts: "Contactos" },
  ctaLabel: "Crea tu mapa",
  pro: { eyebrow: "Para talleres y vendedores", title: "¿Imprimes mucho o para vender?", text: "Monadruk Pro: archivos 3MF/STL ilimitados y licencia comercial para las piezas impresas. Compensa desde unos {n} archivos al mes; cancelación en un clic.", cta: "Más sobre Pro" },
  faqTitle: "Preguntas frecuentes",
  faq: [
    { q: "¿Cuánto tarda la fabricación?", a: "2–4 días hábiles de impresión, luego envío por Ucrania." },
    { q: "¿Hay descuentos para pedidos grandes?", a: "Sí — para tandas de 5 o más piezas idénticas (por ejemplo, llaveros corporativos) el precio se acuerda por separado." },
    { q: "¿Qué incluye el precio?", a: "El precio corresponde al producto terminado en Eco PLA. El envío y el relieve del terreno (+≈2 €) se cobran aparte." },
    { q: "¿Puedo pagar contra entrega?", a: "Sí, además del pago con tarjeta online vía LiqPay, está disponible el pago contra reembolso." },
  ],
};

const fr: Catalog = {
  metaTitle: "Prix des cartes 3D, aimants et porte-clés",
  metaDescription:
    "Prix actuels en UAH : carte 3D de ville dès 350 ₴ (S/M/L/XL), aimant 210 ₴, porte-clés carte dès 170 ₴, relief +85 ₴. Impression en Eco PLA.",
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
        { name: "Fichier 3MF / STL à imprimer soi-même", desc: "3MF prêt, séparé par couleurs, qui s’ouvre dans Bambu Studio ou PrusaSlicer. Pour un usage personnel ; l’aperçu 3D avant achat est gratuit.", uah: PR.file },
        { name: "Monadruk Pro — fichiers illimités", desc: "Fichiers 3MF/STL illimités pour tous les modèles et licence commerciale : imprimez et vendez. Résiliation en un clic.", uah: PR.pro, usd: PR.proUsd, kind: "monthly", href: "/pro" },
      ],
    },
  ],
  notesTitle: "Conditions",
  notes: [
    "Tous les prix sont en hryvnia ukrainienne (₴), par article.",
    "La livraison est facturée séparément au tarif du transporteur (Nova Poshta / Ukrposhta ).",
    "Paiement par Visa / Mastercard en ligne (LiqPay) ou à la livraison.",
    "Les articles sont fabriqués sur commande ; délai 2–4 jours ouvrés plus expédition.",
  ],
  sellerTitle: "Vendeur",
  sellerName: "Vendeur",
  docsIntro: "Les commandes sont régies par le contrat d'offre publique. Détails :",
  docs: { offer: "Contrat d'offre publique", delivery: "Paiement et livraison", refund: "Retours et remboursements", contacts: "Contacts" },
  ctaLabel: "Créer votre carte",
  pro: { eyebrow: "Pour les ateliers et vendeurs", title: "Vous imprimez beaucoup ou pour vendre ?", text: "Monadruk Pro : fichiers 3MF/STL illimités et licence commerciale pour les objets imprimés. Rentable dès environ {n} fichiers par mois ; résiliation en un clic.", cta: "En savoir plus sur Pro" },
  faqTitle: "Questions fréquentes",
  faq: [
    { q: "Combien de temps prend la fabrication ?", a: "1 à 3 jours ouvrés d'impression, puis livraison en Ukraine." },
    { q: "Y a-t-il des remises pour les grandes commandes ?", a: "Oui — pour 5 pièces identiques ou plus (porte-clés d'entreprise par exemple), le prix se négocie séparément." },
    { q: "Qu'est-ce qui est inclus dans le prix ?", a: "Le prix concerne le produit fini en Eco PLA. La livraison et le relief du terrain (+≈2 €) sont facturés à part." },
    { q: "Puis-je payer à la livraison ?", a: "Oui, en plus du paiement en ligne par carte via LiqPay, le paiement à la livraison est disponible." },
  ],
};

const pl: Catalog = {
  metaTitle: "Ceny map 3D, magnesów i breloków",
  metaDescription:
    "Aktualne ceny w UAH: mapa 3D miasta od 350 ₴ (S/M/L/XL), magnes 210 ₴, brelok-mapa od 170 ₴, relief +85 ₴. Druk z Eco PLA, dostawa Nową Pocztą.",
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
        { name: "Plik 3MF / STL do samodzielnego druku", desc: "Gotowy 3MF z podziałem na kolory, otwiera się w Bambu Studio lub PrusaSlicer. Do druku na własny użytek; podgląd 3D przed zakupem jest darmowy.", uah: PR.file },
        { name: "Monadruk Pro — pliki bez limitu", desc: "Pliki 3MF/STL bez limitu dla wszystkich modeli i licencja komercyjna: drukuj i sprzedawaj. Anulowanie jednym kliknięciem.", uah: PR.pro, usd: PR.proUsd, kind: "monthly", href: "/pro" },
      ],
    },
  ],
  notesTitle: "Warunki",
  notes: [
    "Wszystkie ceny są w hrywnach ukraińskich (₴), za sztukę.",
    "Dostawa naliczana osobno według taryfy przewoźnika (Nova Poshta / Ukrposhta).",
    "Płatność kartą Visa / Mastercard online (LiqPay) lub za pobraniem.",
    "Produkty wykonywane na zamówienie; czas realizacji 2–4 dni robocze plus wysyłka.",
  ],
  sellerTitle: "Sprzedawca",
  sellerName: "Sprzedawca",
  docsIntro: "Zamówienia reguluje umowa oferty publicznej. Szczegóły:",
  docs: { offer: "Umowa oferty publicznej", delivery: "Płatność i dostawa", refund: "Zwroty i wymiana", contacts: "Kontakt" },
  ctaLabel: "Stwórz swoją mapę",
  pro: { eyebrow: "Dla drukarni i sprzedawców", title: "Drukujesz dużo albo na sprzedaż?", text: "Monadruk Pro: pliki 3MF/STL bez limitu i licencja komercyjna na wydruki. Opłaca się już od ok. {n} plików miesięcznie; anulowanie jednym kliknięciem.", cta: "Więcej o Pro" },
  faqTitle: "Częste pytania",
  faq: [
    { q: "Ile trwa wykonanie?", a: "2–4 dni robocze druku, potem dostawa po Ukrainie." },
    { q: "Czy są rabaty przy większych zamówieniach?", a: "Tak — przy 5 i więcej identycznych sztukach (np. breloki firmowe) cena ustalana jest indywidualnie." },
    { q: "Co zawiera cena?", a: "Cena dotyczy gotowego produktu z Eco PLA. Dostawa i relief terenu (+≈2 €) są płatne osobno." },
    { q: "Czy mogę zapłacić przy odbiorze?", a: "Tak, oprócz płatności kartą online przez LiqPay dostępna jest płatność za pobraniem." },
  ],
};

const ro: Catalog = {
  metaTitle: "Prețuri pentru hărți 3D ale orașelor, magneți și brelocuri",
  metaDescription:
    "Prețuri actuale în UAH: hartă 3D a orașului de la 350 ₴ (S/M/L/XL), magnet de frigider 210 ₴, breloc cu hartă de la 170 ₴, relief +85 ₴. Imprimare din Eco PLA, livrare prin Nova Poshta.",
  h1: "Prețuri",
  intro:
    "Prețul este pentru produsul finit (imprimat 3D din bioplastic Eco PLA). Livrarea se plătește separat, după tariful curierului. Plata cu Visa / Mastercard online sau la livrare.",
  categories: [
    {
      title: "Hărți 3D ale orașelor",
      items: [
        { name: "Hartă 3D a orașului — S (≈5,5 cm)", desc: "Model 3D imprimat al unei zone din oraș, latura ~5,5 cm.", uah: PR.s },
        { name: "Hartă 3D a orașului — M (≈8 cm)", desc: "Model 3D imprimat al unei zone din oraș, latura ~8 cm.", uah: PR.m },
        { name: "Hartă 3D a orașului — L (≈11 cm)", desc: "Model 3D imprimat al unei zone din oraș, latura ~11 cm.", uah: PR.l },
        { name: "Hartă 3D a orașului — XL (≈15 cm)", desc: "Model 3D imprimat al unei zone din oraș, latura ~15 cm.", uah: PR.xl },
        { name: "Relieful terenului (opțiune)", desc: "Altitudinea suplimentară a peisajului pe orice hartă 3D.", uah: PR.relief, kind: "addon" },
      ],
    },
    {
      title: "Magneți",
      items: [
        { name: "Magnet de frigider cu hartă (≈6 cm)", desc: "Magnet plat cu harta 3D a unei zone din oraș.", uah: PR.magnet },
      ],
    },
    {
      title: "Brelocuri cu hartă",
      items: [
        { name: "Breloc cu hartă (imprimare 3D)", desc: "Breloc cu harta 3D a unei zone din oraș sau cu un traseu (GPX), Eco PLA.", uah: PR.keychain, kind: "from" },
      ],
    },
    {
      title: "Fișiere digitale",
      items: [
        { name: "Fișier 3MF / STL pentru imprimare proprie", desc: "Un 3MF gata făcut, împărțit pe culori, care se deschide în Bambu Studio sau PrusaSlicer. Pentru imprimare personală; previzualizarea 3D înainte de cumpărare este gratuită.", uah: PR.file },
        { name: "Monadruk Pro — fișiere nelimitate", desc: "Fișiere 3MF/STL nelimitate pentru orice model plus licență comercială: imprimă și vinde produsele. Anulare cu un clic.", uah: PR.pro, usd: PR.proUsd, kind: "monthly", href: "/pro" },
      ],
    },
  ],
  notesTitle: "Condiții",
  notes: [
    "Toate prețurile sunt în grivne ucrainene (₴), pe bucată.",
    "Livrarea se taxează separat, după tariful curierului (Nova Poshta / Ukrposhta).",
    "Plata cu Visa / Mastercard online (LiqPay) sau ramburs la livrare.",
    "Produsele se realizează la comandă; termenul de execuție este de 2–4 zile lucrătoare plus livrarea.",
  ],
  sellerTitle: "Vânzător",
  sellerName: "Vânzător",
  docsIntro: "Comenzile sunt reglementate de contractul de ofertă publică. Detalii:",
  docs: { offer: "Contract de ofertă publică", delivery: "Plată și livrare", refund: "Returnări și rambursări", contacts: "Contacte" },
  ctaLabel: "Creează-ți harta",
  pro: { eyebrow: "Pentru ateliere de imprimare și vânzători", title: "Imprimi mult sau imprimi pentru vânzare?", text: "Monadruk Pro îți oferă fișiere 3MF/STL nelimitate și o licență comercială pentru produsele imprimate. Merită de la aproximativ al {n}-lea fișier pe lună; anulare cu un clic.", cta: "Află despre Pro" },
  faqTitle: "Întrebări frecvente",
  faq: [
    { q: "Cât durează realizarea?", a: "2–4 zile lucrătoare pentru imprimare, apoi livrarea în Ucraina." },
    { q: "Există reduceri pentru comenzi mari?", a: "Da — pentru tiraje de 5+ produse identice (de ex. brelocuri corporative) prețul se stabilește individual, scrie-ne." },
    { q: "Ce include prețul?", a: "Prețul acoperă produsul finit imprimat din Eco PLA. Livrarea și relieful terenului (+≈2 €) se plătesc separat." },
    { q: "Pot plăti la livrare?", a: "Da, pe lângă plata online cu cardul prin LiqPay este disponibilă și plata ramburs." },
  ],
};

const CATALOGS: Record<string, Catalog> = { uk, en, de, es, fr, pl, ro };

export function getCatalog(locale: string): Catalog {
  return CATALOGS[locale] ?? uk;
}

// Локалізовані слова цінника (спільні для /prices і price-band на сторінках міст).
export const PRICE_WORDS: Record<string, { from: string; free: string; perMonth: string }> = {
  uk: { from: "від", free: "Безкоштовно*", perMonth: "/ міс" },
  en: { from: "from", free: "Free*", perMonth: "/ mo" },
  de: { from: "ab", free: "Kostenlos*", perMonth: "/ Monat" },
  es: { from: "desde", free: "Gratis*", perMonth: "/ mes" },
  fr: { from: "dès", free: "Gratuit*", perMonth: "/ mois" },
  pl: { from: "od", free: "Bezpłatnie*", perMonth: "/ mies." },
  ro: { from: "de la", free: "Gratuit*", perMonth: "/ lună" },
};

/** Єдине форматування ціни товару: «N ₴» (uk) / «N ₴ · ≈M €» (EU); «+N ₴» (addon);
 *  «від N ₴» (from); «Безкоштовно*» (uah=0). Спільне для /prices і сторінок міст. */
export function formatCatalogPrice(uah: number, kind: string | undefined, locale: string, usd?: number): string {
  const w = PRICE_WORDS[locale] ?? PRICE_WORDS.uk;
  if (uah === 0) return w.free;
  if (kind === "addon") return `+${uah} ₴`;
  if (kind === "monthly") {
    const n = new Intl.NumberFormat(locale === "uk" ? "uk-UA" : locale).format(uah);
    return locale !== "uk" && usd ? `$${usd} ${w.perMonth}` : `${n} ₴ ${w.perMonth}`;
  }
  const eur = locale !== "uk" ? ` · ≈${mapPriceEur(uah)} €` : "";
  const base = `${uah} ₴${eur}`;
  return kind === "from" ? `${w.from} ${base}` : base;
}
