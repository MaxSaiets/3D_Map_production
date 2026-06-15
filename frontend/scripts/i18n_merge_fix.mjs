// Merges all C:/Temp/i18n_fix_*.json (each: { uk:{ns:{...}}, en:{...}, ... }) into
// messages/<locale>.json via additive deep-merge. Usage: node scripts/i18n_merge_fix.mjs
import fs from "node:fs";
import path from "node:path";

const TEMP = "C:/Temp";
const MSG = path.resolve("messages");
const LOCALES = ["uk", "en", "de", "es", "fr", "pl"];

const deepMerge = (target, src) => {
  for (const k of Object.keys(src)) {
    const v = src[k];
    if (v && typeof v === "object" && !Array.isArray(v)) {
      if (!target[k] || typeof target[k] !== "object" || Array.isArray(target[k])) target[k] = {};
      deepMerge(target[k], v);
    } else {
      target[k] = v;
    }
  }
  return target;
};

const tempFiles = fs.readdirSync(TEMP).filter((f) => /^i18n_fix_.*\.json$/.test(f));
if (!tempFiles.length) { console.error("No i18n_fix_*.json in", TEMP); process.exit(1); }
console.log("Temp files:", tempFiles.join(", "));

// load & validate
const batches = [];
for (const f of tempFiles) {
  const obj = JSON.parse(fs.readFileSync(path.join(TEMP, f), "utf8"));
  const present = LOCALES.filter((l) => obj[l]);
  const missing = LOCALES.filter((l) => !obj[l]);
  // namespaces present (top-level keys of uk)
  const ns = obj.uk ? Object.keys(obj.uk) : [];
  console.log(`  ${f}: locales=[${present}]${missing.length ? " MISSING=[" + missing + "]" : ""} ns=[${ns}]`);
  batches.push({ f, obj });
}

// merge per locale
let added = 0;
const flat = (o, p = "", out = {}) => { for (const k of Object.keys(o)) { const key = p ? p + "." + k : k; const v = o[k]; if (v && typeof v === "object" && !Array.isArray(v)) flat(v, key, out); else out[key] = true; } return out; };
for (const locale of LOCALES) {
  const file = path.join(MSG, `${locale}.json`);
  const msg = JSON.parse(fs.readFileSync(file, "utf8"));
  const before = Object.keys(flat(msg)).length;
  for (const b of batches) {
    if (b.obj[locale]) deepMerge(msg, b.obj[locale]);
    else if (b.obj.uk) deepMerge(msg, b.obj.uk); // fallback: if a locale missing, seed with uk (so no raw keys)
  }
  const after = Object.keys(flat(msg)).length;
  fs.writeFileSync(file, JSON.stringify(msg, null, 2) + "\n", "utf8");
  console.log(`  ${locale}.json: ${before} -> ${after} keys (+${after - before})`);
  if (locale === "uk") added = after - before;
}
console.log(`\nDone. ~${added} new keys per locale.`);
