import fs from "node:fs"; import path from "node:path";
const DIR = path.resolve("messages");
const data = {
  consent: {
    uk: { text: "Ми використовуємо файли cookie для аналітики та покращення сайту.", accept: "Прийняти", decline: "Відхилити", more: "Детальніше" },
    en: { text: "We use cookies for analytics and to improve the site.", accept: "Accept", decline: "Decline", more: "Learn more" },
    de: { text: "Wir verwenden Cookies für Analysen und zur Verbesserung der Website.", accept: "Akzeptieren", decline: "Ablehnen", more: "Mehr erfahren" },
    pl: { text: "Używamy plików cookie do analiz i ulepszania witryny.", accept: "Akceptuję", decline: "Odrzuć", more: "Więcej" },
    fr: { text: "Nous utilisons des cookies pour l'analyse et l'amélioration du site.", accept: "Accepter", decline: "Refuser", more: "En savoir plus" },
    es: { text: "Usamos cookies para análisis y para mejorar el sitio.", accept: "Aceptar", decline: "Rechazar", more: "Más información" },
  },
};
for (const loc of ["uk","en","de","pl","fr","es"]) {
  const f = path.join(DIR, `${loc}.json`); const j = JSON.parse(fs.readFileSync(f,"utf8"));
  for (const ns of Object.keys(data)) j[ns] = { ...(j[ns]||{}), ...data[ns][loc] };
  fs.writeFileSync(f, JSON.stringify(j,null,2)+"\n","utf8"); console.log("merged",loc);
}
