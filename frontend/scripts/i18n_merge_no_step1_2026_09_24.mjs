// Guided /create і /keychains без кроку 1 (24.09.2026, власник: «перші кроки взагалі не
// подобаються — прибери»): товар/форма перемикаються компактним рядком угорі панелі.
import fs from "node:fs"; import path from "node:path";
const DIR = path.resolve("messages");
const scenario = {
  uk: { productLabel: "Що друкуємо", prodShort_map3d: "Обʼємна", prodShort_relief: "З рельєфом", prodShort_flat: "Плоска", prodShort_magnet: "Магніт" },
  en: { productLabel: "What we print", prodShort_map3d: "3D city", prodShort_relief: "With relief", prodShort_flat: "Flat", prodShort_magnet: "Magnet" },
  de: { productLabel: "Was wir drucken", prodShort_map3d: "3D-Stadt", prodShort_relief: "Mit Relief", prodShort_flat: "Flach", prodShort_magnet: "Magnet" },
  pl: { productLabel: "Co drukujemy", prodShort_map3d: "3D miasto", prodShort_relief: "Z reliefem", prodShort_flat: "Płaska", prodShort_magnet: "Magnes" },
  fr: { productLabel: "Ce qu'on imprime", prodShort_map3d: "Ville 3D", prodShort_relief: "Avec relief", prodShort_flat: "Plate", prodShort_magnet: "Aimant" },
  es: { productLabel: "Qué imprimimos", prodShort_map3d: "Ciudad 3D", prodShort_relief: "Con relieve", prodShort_flat: "Plana", prodShort_magnet: "Imán" },
};
const kcScenario = {
  uk: { kcShapeLabel: "Форма брелока" },
  en: { kcShapeLabel: "Keychain shape" },
  de: { kcShapeLabel: "Form des Anhängers" },
  pl: { kcShapeLabel: "Kształt breloka" },
  fr: { kcShapeLabel: "Forme du porte-clés" },
  es: { kcShapeLabel: "Forma del llavero" },
};
for (const loc of ["uk", "en", "de", "pl", "fr", "es"]) {
  const f = path.join(DIR, `${loc}.json`); const j = JSON.parse(fs.readFileSync(f, "utf8"));
  j.scenario = { ...j.scenario, ...scenario[loc] };
  j.kcScenario = { ...j.kcScenario, ...kcScenario[loc] };
  fs.writeFileSync(f, JSON.stringify(j, null, 2) + "\n", "utf8");
}
console.log("done");
