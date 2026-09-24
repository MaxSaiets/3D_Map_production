// Guided /create — уточнення «Частин» (24.09.2026): відмінки кількості плиток (ICU plural),
// замок одиночної плитки переїхав у «Частини» (коротший підпис), розділ 4 без згадки про замок.
import fs from "node:fs"; import path from "node:path";
const DIR = path.resolve("messages");
const scenario = {
  uk: {
    partsTiles: "{n, plural, one {# плитка} few {# плитки} many {# плиток} other {# плитки}}",
    connectors: "Замки по краях плитки",
    connectorsHint: "Щоб пізніше докупити сусідню ділянку й скласти їх в одну велику мапу",
    personalizeHint: "Позначте свій дім або додайте напис.",
  },
  en: {
    partsTiles: "{n, plural, one {# tile} other {# tiles}}",
    connectors: "Locks on the tile edges",
    connectorsHint: "So you can add a neighbouring area later and join them into one big map",
    personalizeHint: "Mark your home or add a label.",
  },
  de: {
    partsTiles: "{n, plural, one {# Kachel} other {# Kacheln}}",
    connectors: "Verbinder an den Kachelrändern",
    connectorsHint: "Damit du später ein Nachbarviertel dazukaufen und zu einer großen Karte verbinden kannst",
    personalizeHint: "Markiere dein Zuhause oder füge einen Schriftzug hinzu.",
  },
  pl: {
    partsTiles: "{n, plural, one {# płytka} few {# płytki} many {# płytek} other {# płytki}}",
    connectors: "Zamki na krawędziach płytki",
    connectorsHint: "Żeby później dokupić sąsiedni obszar i połączyć je w jedną dużą mapę",
    personalizeHint: "Zaznacz swój dom lub dodaj napis.",
  },
  fr: {
    partsTiles: "{n, plural, one {# tuile} other {# tuiles}}",
    connectors: "Attaches sur les bords de la tuile",
    connectorsHint: "Pour ajouter plus tard un quartier voisin et les assembler en une grande carte",
    personalizeHint: "Marquez votre maison ou ajoutez un texte.",
  },
  es: {
    partsTiles: "{n, plural, one {# baldosa} other {# baldosas}}",
    connectors: "Cierres en los bordes de la baldosa",
    connectorsHint: "Para añadir más adelante una zona vecina y unirlas en un mapa grande",
    personalizeHint: "Marca tu casa o añade un texto.",
  },
};
for (const loc of ["uk", "en", "de", "pl", "fr", "es"]) {
  const f = path.join(DIR, `${loc}.json`); const j = JSON.parse(fs.readFileSync(f, "utf8"));
  j.scenario = { ...j.scenario, ...scenario[loc] };
  fs.writeFileSync(f, JSON.stringify(j, null, 2) + "\n", "utf8");
}
console.log("done");
