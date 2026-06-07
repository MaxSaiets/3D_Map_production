import fs from "node:fs"; import path from "node:path";
const DIR = path.resolve("messages");
const data = { notFound: {
  uk: { title: "Сторінку не знайдено", text: "Можливо, її переміщено або видалено. Повернись на головну й створи свою 3D-мапу.", home: "На головну" },
  en: { title: "Page not found", text: "It may have been moved or removed. Head back home and create your 3D map.", home: "Back home" },
  de: { title: "Seite nicht gefunden", text: "Sie wurde möglicherweise verschoben oder gelöscht. Zurück zur Startseite und erstelle deine 3D-Karte.", home: "Zur Startseite" },
  pl: { title: "Nie znaleziono strony", text: "Mogła zostać przeniesiona lub usunięta. Wróć na stronę główną i stwórz swoją mapę 3D.", home: "Strona główna" },
  fr: { title: "Page introuvable", text: "Elle a peut-être été déplacée ou supprimée. Reviens à l'accueil et crée ta carte 3D.", home: "Accueil" },
  es: { title: "Página no encontrada", text: "Puede que se haya movido o eliminado. Vuelve al inicio y crea tu mapa 3D.", home: "Inicio" },
}};
for (const loc of ["uk","en","de","pl","fr","es"]) { const f=path.join(DIR,`${loc}.json`); const j=JSON.parse(fs.readFileSync(f,"utf8")); for(const ns of Object.keys(data)) j[ns]={...(j[ns]||{}),...data[ns][loc]}; fs.writeFileSync(f,JSON.stringify(j,null,2)+"\n","utf8"); }
console.log("done");
