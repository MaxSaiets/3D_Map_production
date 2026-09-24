// Guided /create — ФОРМА мапи (24.09.2026, власник: «не зрозуміло, не повністю все можна вибрати»):
// секція 2 = «Форма і розмір», 4 форми (квадрат/коло/шестикутник/серце) з контурами.
import fs from "node:fs"; import path from "node:path";
const DIR = path.resolve("messages");
const scenario = {
  uk: {
    sizeTitle: "Форма і розмір",
    sizeHint: "Розмір — ширина готової моделі. Ціна — за готовий друк.",
    shapeLabel: "Форма",
    sizeLabel: "Розмір",
    shapeSquare: "Квадрат",
    shapeCircle: "Коло",
    shapeHexagon: "Шестикутник",
    shapeHeart: "Серце",
    shapeHint: "Контур одразу видно на карті — надрукуємо саме те, що всередині.",
  },
  en: {
    sizeTitle: "Shape and size",
    sizeHint: "Size is the width of the finished model. Price is for the finished print.",
    shapeLabel: "Shape",
    sizeLabel: "Size",
    shapeSquare: "Square",
    shapeCircle: "Circle",
    shapeHexagon: "Hexagon",
    shapeHeart: "Heart",
    shapeHint: "The outline appears on the map right away — we print exactly what is inside.",
  },
  de: {
    sizeTitle: "Form und Größe",
    sizeHint: "Die Größe ist die Breite des fertigen Modells. Preis für den fertigen Druck.",
    shapeLabel: "Form",
    sizeLabel: "Größe",
    shapeSquare: "Quadrat",
    shapeCircle: "Kreis",
    shapeHexagon: "Sechseck",
    shapeHeart: "Herz",
    shapeHint: "Die Kontur erscheint sofort auf der Karte — gedruckt wird genau, was darin liegt.",
  },
  pl: {
    sizeTitle: "Kształt i rozmiar",
    sizeHint: "Rozmiar to szerokość gotowego modelu. Cena za gotowy wydruk.",
    shapeLabel: "Kształt",
    sizeLabel: "Rozmiar",
    shapeSquare: "Kwadrat",
    shapeCircle: "Koło",
    shapeHexagon: "Sześciokąt",
    shapeHeart: "Serce",
    shapeHint: "Kontur od razu widać na mapie — wydrukujemy dokładnie to, co w środku.",
  },
  fr: {
    sizeTitle: "Forme et taille",
    sizeHint: "La taille est la largeur du modèle fini. Prix de l'impression finie.",
    shapeLabel: "Forme",
    sizeLabel: "Taille",
    shapeSquare: "Carré",
    shapeCircle: "Cercle",
    shapeHexagon: "Hexagone",
    shapeHeart: "Cœur",
    shapeHint: "Le contour apparaît aussitôt sur la carte — on imprime exactement ce qu'il contient.",
  },
  es: {
    sizeTitle: "Forma y tamaño",
    sizeHint: "El tamaño es el ancho del modelo terminado. Precio de la impresión terminada.",
    shapeLabel: "Forma",
    sizeLabel: "Tamaño",
    shapeSquare: "Cuadrado",
    shapeCircle: "Círculo",
    shapeHexagon: "Hexágono",
    shapeHeart: "Corazón",
    shapeHint: "El contorno aparece al instante en el mapa: imprimimos exactamente lo que queda dentro.",
  },
};
for (const loc of ["uk", "en", "de", "pl", "fr", "es"]) {
  const f = path.join(DIR, `${loc}.json`); const j = JSON.parse(fs.readFileSync(f, "utf8"));
  j.scenario = { ...j.scenario, ...scenario[loc] };
  fs.writeFileSync(f, JSON.stringify(j, null, 2) + "\n", "utf8");
}
console.log("done");
