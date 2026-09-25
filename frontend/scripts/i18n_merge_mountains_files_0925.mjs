// Гори: файли через вхід (3 безкоштовно), ціна індивідуальна, ручний розпис (25.09.2026).
// Запуск: node scripts/i18n_merge_mountains_files_0925.mjs (з frontend/).
import fs from "node:fs"; import path from "node:path";
const DIR = path.resolve("messages");
const L = ["uk", "en", "de", "pl", "fr", "es"];

const mountains = {
  uk: { priceTitle: "Ціна індивідуальна", priceIndividual: "Залежить від розміру, фігурок і розпису. Напишіть нам — порахуємо за кілька хвилин.",
    paintTitle: "Можемо розмалювати", paintSub: "вручну акрилом: ліс, скелі, сніг — як у справжньої гори.", wantPaint: "Хочу з розписом",
    msgPaint: "Хочу з ручним розписом.", orderSub: "Ціна індивідуальна: залежить від розміру, фігурок і розпису. Опис і посилання на 3D скопіюються — просто вставте в чат.",
    downloadFreeNote: "Файл для друку — після входу: 3 гори безкоштовно на акаунт.", downloadLogin: "Увійдіть, щоб завантажити — 3 файли гір безкоштовно.",
    downloadLeft: "Готово! Залишилось безкоштовних файлів гір: {n}.", downloadLimit: "Безкоштовні файли гір вичерпано. Напишіть нам — надішлемо файл або надрукуємо:",
    downloadVerify: "Підтвердьте email (лист у пошті), щоб завантажувати файли." },
  en: { priceTitle: "Individual price", priceIndividual: "Depends on size, figures and painting. Message us — we’ll quote in a few minutes.",
    paintTitle: "We can paint it", paintSub: "by hand in acrylic: forest, rock, snow — like the real mountain.", wantPaint: "I want it painted",
    msgPaint: "I’d like it hand-painted.", orderSub: "Individual price: depends on size, figures and painting. The description and 3D link are copied — just paste them into the chat.",
    downloadFreeNote: "Print file after signing in: 3 mountains free per account.", downloadLogin: "Sign in to download — 3 mountain files free.",
    downloadLeft: "Done! Free mountain files left: {n}.", downloadLimit: "You’ve used your free mountain files. Message us — we’ll send the file or print it:",
    downloadVerify: "Confirm your email (check your inbox) to download files." },
  de: { priceTitle: "Individueller Preis", priceIndividual: "Hängt von Größe, Figuren und Bemalung ab. Schreiben Sie uns — Angebot in wenigen Minuten.",
    paintTitle: "Wir bemalen ihn", paintSub: "von Hand mit Acryl: Wald, Fels, Schnee — wie der echte Berg.", wantPaint: "Mit Bemalung",
    msgPaint: "Ich möchte ihn handbemalt.", orderSub: "Individueller Preis: hängt von Größe, Figuren und Bemalung ab. Beschreibung und 3D-Link werden kopiert — einfach in den Chat einfügen.",
    downloadFreeNote: "Druckdatei nach Anmeldung: 3 Berge pro Konto kostenlos.", downloadLogin: "Melden Sie sich an — 3 Berg-Dateien kostenlos.",
    downloadLeft: "Fertig! Verbleibende kostenlose Berg-Dateien: {n}.", downloadLimit: "Kostenlose Berg-Dateien aufgebraucht. Schreiben Sie uns — wir senden die Datei oder drucken:",
    downloadVerify: "Bestätigen Sie Ihre E-Mail (Posteingang prüfen), um Dateien zu laden." },
  pl: { priceTitle: "Cena indywidualna", priceIndividual: "Zależy od rozmiaru, figurek i malowania. Napisz do nas — wycenimy w kilka minut.",
    paintTitle: "Możemy pomalować", paintSub: "ręcznie akrylami: las, skały, śnieg — jak prawdziwa góra.", wantPaint: "Chcę z malowaniem",
    msgPaint: "Chcę ręcznie malowaną.", orderSub: "Cena indywidualna: zależy od rozmiaru, figurek i malowania. Opis i link do 3D zostaną skopiowane — wklej je w czacie.",
    downloadFreeNote: "Plik do druku po zalogowaniu: 3 góry za darmo na konto.", downloadLogin: "Zaloguj się, aby pobrać — 3 pliki gór za darmo.",
    downloadLeft: "Gotowe! Pozostało darmowych plików gór: {n}.", downloadLimit: "Darmowe pliki gór wykorzystane. Napisz do nas — wyślemy plik lub wydrukujemy:",
    downloadVerify: "Potwierdź e-mail (sprawdź skrzynkę), aby pobierać pliki." },
  fr: { priceTitle: "Prix sur mesure", priceIndividual: "Selon la taille, les figurines et la peinture. Écrivez-nous — devis en quelques minutes.",
    paintTitle: "Nous pouvons la peindre", paintSub: "à la main à l’acrylique : forêt, roche, neige — comme la vraie montagne.", wantPaint: "Je la veux peinte",
    msgPaint: "Je la voudrais peinte à la main.", orderSub: "Prix sur mesure : selon la taille, les figurines et la peinture. La description et le lien 3D sont copiés — collez-les dans le chat.",
    downloadFreeNote: "Fichier d’impression après connexion : 3 montagnes gratuites par compte.", downloadLogin: "Connectez-vous pour télécharger — 3 fichiers de montagne gratuits.",
    downloadLeft: "C’est fait ! Fichiers de montagne gratuits restants : {n}.", downloadLimit: "Fichiers gratuits épuisés. Écrivez-nous — nous enverrons le fichier ou l’imprimerons :",
    downloadVerify: "Confirmez votre e-mail (voir la boîte de réception) pour télécharger." },
  es: { priceTitle: "Precio individual", priceIndividual: "Depende del tamaño, las figuras y la pintura. Escríbenos — te damos precio en minutos.",
    paintTitle: "Podemos pintarla", paintSub: "a mano con acrílico: bosque, roca, nieve — como la montaña real.", wantPaint: "La quiero pintada",
    msgPaint: "La quiero pintada a mano.", orderSub: "Precio individual: depende del tamaño, las figuras y la pintura. La descripción y el enlace 3D se copian — pégalos en el chat.",
    downloadFreeNote: "Archivo de impresión tras iniciar sesión: 3 montañas gratis por cuenta.", downloadLogin: "Inicia sesión para descargar — 3 archivos de montaña gratis.",
    downloadLeft: "¡Listo! Archivos de montaña gratis restantes: {n}.", downloadLimit: "Has usado tus archivos gratis. Escríbenos — te enviamos el archivo o lo imprimimos:",
    downloadVerify: "Confirma tu email (revisa la bandeja) para descargar archivos." },
};

const beta = {
  uk: "Реальний рельєф зі swissALTI3D / Copernicus; фігурки сувенірні, не в масштабі. Ціна друку індивідуальна, можемо розмалювати — напишіть нам.",
  en: "Real terrain from swissALTI3D / Copernicus; figures are decorative, not to scale. Print price is individual and we can paint it — message us.",
  de: "Echtes Gelände aus swissALTI3D / Copernicus; Figuren dekorativ, nicht maßstäblich. Druckpreis individuell, Bemalung möglich — schreiben Sie uns.",
  pl: "Prawdziwa rzeźba terenu ze swissALTI3D / Copernicus; figurki ozdobne, nie w skali. Cena druku indywidualna, możemy pomalować — napisz do nas.",
  fr: "Relief réel swissALTI3D / Copernicus ; figurines décoratives, pas à l’échelle. Prix d’impression sur mesure, peinture possible — écrivez-nous.",
  es: "Relieve real de swissALTI3D / Copernicus; figuras decorativas, no a escala. Precio de impresión individual y podemos pintarla — escríbenos.",
};

for (const l of L) {
  const f = path.join(DIR, `${l}.json`);
  const j = JSON.parse(fs.readFileSync(f, "utf8"));
  j.mountains = { ...(j.mountains || {}), ...mountains[l] };
  j.beta = { ...(j.beta || {}), mountains: beta[l] };
  fs.writeFileSync(f, JSON.stringify(j, null, 2) + "\n", "utf8");
  console.log(l, "ok");
}
