// Merge the "account" namespace into every locale messages file.
// Source of truth: C:\Temp\i18n_account.json  (shape { namespace, locales: { uk, en, ... } })
// Usage: node scripts/i18n_merge_account.mjs
import fs from "node:fs";
import path from "node:path";

const DIR = path.resolve("messages");
const SRC = "C:/Temp/i18n_account.json";
const { namespace, locales } = JSON.parse(fs.readFileSync(SRC, "utf8"));

for (const loc of ["uk", "en", "de", "pl", "fr", "es"]) {
  const f = path.join(DIR, `${loc}.json`);
  const j = JSON.parse(fs.readFileSync(f, "utf8"));
  j[namespace] = { ...(j[namespace] || {}), ...locales[loc] };
  fs.writeFileSync(f, JSON.stringify(j, null, 2) + "\n", "utf8");
  console.log("merged", loc, namespace);
}
