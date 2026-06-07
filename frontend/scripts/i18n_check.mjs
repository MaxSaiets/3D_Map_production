// i18n completeness checker. Compares every locale against the default (uk) and
// reports missing / extra keys. Usage:  node scripts/i18n_check.mjs
// Exit code 1 if any locale is missing keys (handy for CI / pre-deploy).
import fs from "node:fs";
import path from "node:path";

const DIR = path.resolve("messages");
const DEFAULT = "uk";
const locales = ["uk", "en", "de", "pl", "fr", "es"];

const flat = (obj, prefix = "", out = {}) => {
  for (const k of Object.keys(obj)) {
    const v = obj[k];
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v)) flat(v, key, out);
    else out[key] = true;
  }
  return out;
};

const load = (l) => JSON.parse(fs.readFileSync(path.join(DIR, `${l}.json`), "utf8"));
const baseKeys = Object.keys(flat(load(DEFAULT)));
let problems = 0;

for (const l of locales) {
  if (l === DEFAULT) continue;
  const keys = new Set(Object.keys(flat(load(l))));
  const missing = baseKeys.filter((k) => !keys.has(k));
  const extra = [...keys].filter((k) => !baseKeys.includes(k));
  if (missing.length || extra.length) {
    console.log(`\n[${l}]`);
    if (missing.length) { problems += missing.length; console.log(`  MISSING (${missing.length}) — буде показано укр.:`); missing.forEach((k) => console.log(`    - ${k}`)); }
    if (extra.length) console.log(`  EXTRA (${extra.length}) — зайві ключі:\n${extra.map((k) => `    + ${k}`).join("\n")}`);
  } else {
    console.log(`[${l}] OK — ${baseKeys.length} keys`);
  }
}

console.log(problems ? `\n⚠ ${problems} missing translation(s). Вони фолбекнуться на ${DEFAULT}.` : `\n✓ Усі локалі повні (${baseKeys.length} ключів).`);
process.exit(problems ? 1 : 0);
