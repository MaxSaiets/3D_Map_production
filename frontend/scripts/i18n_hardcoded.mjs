// Finds HARDCODED Cyrillic UI text in source (strings/JSX outside comments) that
// won't translate when the locale switches. Usage: node scripts/i18n_hardcoded.mjs
import fs from "node:fs";
import path from "node:path";

const SRC = ["components", "app"];
const files = [];
const walk = (d) => {
  for (const e of fs.readdirSync(d, { withFileTypes: true })) {
    const p = path.join(d, e.name);
    if (e.isDirectory()) { if (!/node_modules|\.next/.test(p)) walk(p); }
    else if (/\.(tsx|jsx)$/.test(e.name)) files.push(p);
  }
};
for (const s of SRC) { const d = path.resolve(s); if (fs.existsSync(d)) walk(d); }

const CYR = /[А-Яа-яІіЇїЄєҐґ]/;
// strip // line comments and /* */ block comments and {/* jsx */} (rough)
const stripComments = (s) =>
  s.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:])\/\/[^\n]*/g, "$1");

const results = [];
for (const f of files) {
  const raw = fs.readFileSync(f, "utf8");
  const code = stripComments(raw);
  const lines = code.split("\n");
  const hits = [];
  lines.forEach((line, i) => {
    if (!CYR.test(line)) return;
    // ignore lines that are clearly only comments left over
    const trimmed = line.trim();
    if (trimmed.startsWith("*") || trimmed.startsWith("//")) return;
    hits.push({ n: i + 1, text: trimmed.slice(0, 90) });
  });
  if (hits.length) results.push({ file: path.relative(process.cwd(), f), count: hits.length, hits });
}

results.sort((a, b) => b.count - a.count);
let total = 0;
console.log("===== HARDCODED CYRILLIC IN SOURCE (will NOT translate) =====\n");
for (const r of results) {
  total += r.count;
  console.log(`${r.file}  (${r.count})`);
  for (const h of r.hits.slice(0, 6)) console.log(`    ${h.n}: ${h.text}`);
  if (r.hits.length > 6) console.log(`    … +${r.hits.length - 6} more`);
  console.log("");
}
console.log(`===== ${results.length} files, ${total} hardcoded Cyrillic lines total =====`);
