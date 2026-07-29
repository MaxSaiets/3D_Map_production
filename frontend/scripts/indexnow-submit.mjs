#!/usr/bin/env node
/**
 * IndexNow submit — миттєво повідомляє Bing/Yandex/Seznam/Naver про URL-и сайту,
 * щоб їх переіндексували без очікування краулу. Ключ підтверджується файлом
 * public/<KEY>.txt (той самий KEY нижче). Безкоштовно, без акаунта.
 *
 * ВАЖЛИВО (перевірено 2026-07): Google IndexNow НЕ підтримує — це трафік з
 * інших двигунів. Для Google працюють лише sitemap + внутрішні лінки.
 *
 * 29.07.2026: URL-и тепер беруться з ЖИВОГО sitemap.xml (усі ~800), а не з
 * хардкод-списку на 9 сторінок. Ліміт IndexNow — 10 000 URL за запит, влазить.
 * Запуск: node scripts/indexnow-submit.mjs   (автоматично з deploy/sync.ps1)
 */
const HOST = "monadruk.com";
const BASE = `https://${HOST}`;
const KEY = "7fd03a9642792143bf04983c18510084"; // == public/<KEY>.txt
const KEY_LOCATION = `${BASE}/${KEY}.txt`;
const MAX_URLS = 10000;

// Фолбек, якщо sitemap недоступний (мережа/деплой ще не піднявся).
const CORE = ["", "/create", "/keychains", "/brelok", "/maps", "/podarunok", "/prices", "/showcase", "/blog"];

async function urlsFromSitemap() {
  const res = await fetch(`${BASE}/sitemap.xml`, { headers: { "User-Agent": "monadruk-indexnow" } });
  if (!res.ok) throw new Error(`sitemap ${res.status}`);
  const xml = await res.text();
  const urls = [...xml.matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => m[1].trim());
  // Унікальні + лише свій хост (IndexNow відхилить чужі домени цілим запитом)
  return [...new Set(urls)].filter((u) => u.startsWith(BASE)).slice(0, MAX_URLS);
}

async function main() {
  let urlList;
  try {
    urlList = await urlsFromSitemap();
    console.log(`IndexNow: взято ${urlList.length} URL із sitemap.xml`);
  } catch (e) {
    urlList = CORE.map((p) => `${BASE}${p || "/"}`);
    console.log(`IndexNow: sitemap недоступний (${e.message}) → фолбек ${urlList.length} core-URL`);
  }
  const body = { host: HOST, key: KEY, keyLocation: KEY_LOCATION, urlList };
  const res = await fetch("https://api.indexnow.org/indexnow", {
    method: "POST",
    headers: { "Content-Type": "application/json; charset=utf-8" },
    body: JSON.stringify(body),
  });
  console.log(`IndexNow → ${res.status} ${res.statusText} (${urlList.length} URLs)`);
  // 200/202 = прийнято. 422 = ключ/хост не збігаються. 429 = забагато запитів.
  if (!res.ok) console.log((await res.text()).slice(0, 300));
}

main().catch((e) => { console.error("IndexNow submit failed:", e); process.exit(1); });
