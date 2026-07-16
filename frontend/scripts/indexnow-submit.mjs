#!/usr/bin/env node
/**
 * IndexNow submit — миттєво повідомляє Bing/Yandex (та інші IndexNow-двигуни) про
 * URL-и сайту, щоб їх переіндексували без очікування краулу. Ключ підтверджується
 * файлом public/<KEY>.txt (той самий KEY нижче). Запускати ПІСЛЯ деплою:
 *   node scripts/indexnow-submit.mjs
 * (можна в cron раз на день, або в deploy-хук). Безкоштовно, без акаунта.
 *
 * Джерело URL-ів — той самий перелік, що й у app/sitemap.ts (щоб не розходились).
 * Тут дублюємо мінімально (core + programmatic шляхи) — за потреби розширити.
 */
const HOST = "monadruk.com";
const BASE = `https://${HOST}`;
const KEY = "7fd03a9642792143bf04983c18510084"; // == public/<KEY>.txt
const KEY_LOCATION = `${BASE}/${KEY}.txt`;

// Мінімальний перелік найважливіших сторінок (uk, без префікса). Розширюй за потреби —
// або згенеруй із sitemap.xml (fetch(`${BASE}/sitemap.xml`) + парс) у майбутньому.
const CORE = ["", "/create", "/keychains", "/brelok", "/maps", "/podarunok", "/prices", "/showcase", "/blog"];

async function main() {
  const urlList = CORE.map((p) => `${BASE}${p || "/"}`);
  const body = { host: HOST, key: KEY, keyLocation: KEY_LOCATION, urlList };
  const res = await fetch("https://api.indexnow.org/indexnow", {
    method: "POST",
    headers: { "Content-Type": "application/json; charset=utf-8" },
    body: JSON.stringify(body),
  });
  console.log(`IndexNow → ${res.status} ${res.statusText} (${urlList.length} URLs)`);
  if (!res.ok) console.log(await res.text());
}

main().catch((e) => { console.error("IndexNow submit failed:", e); process.exit(1); });
