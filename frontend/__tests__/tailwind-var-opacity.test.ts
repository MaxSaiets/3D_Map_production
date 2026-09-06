/**
 * L-1 (2026-09-06): `bg-[var(--x)]/95` — Tailwind v3 НЕ вміє накладати модифікатор
 * прозорості на CSS-змінну → клас компілюється у невалідний CSS, і фон стає
 * ПРОЗОРИМ (обчислений колір rgba(0,0,0,0)). Так guided sticky-бар на мобільному
 * не мав фону, і картки розмірів просвічували крізь нього. Ловимо на рівні джерел.
 */
import fs from "fs";
import path from "path";

const ROOT = path.join(__dirname, "..");
const DIRS = ["app", "components"];
const RE = /\b(bg|text|border|ring|from|to|via)-\[var\([^\]]*\)\]\/\d+/g;

function walk(dir: string, out: string[] = []): string[] {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p, out);
    else if (/\.(tsx|ts|jsx|js|css)$/.test(e.name)) out.push(p);
  }
  return out;
}

it("no Tailwind opacity modifier on CSS-variable colours", () => {
  const hits: string[] = [];
  for (const d of DIRS) {
    for (const f of walk(path.join(ROOT, d))) {
      const src = fs.readFileSync(f, "utf8");
      const m = src.match(RE);
      if (m) hits.push(`${path.relative(ROOT, f)}: ${[...new Set(m)].join(", ")}`);
    }
  }
  expect(hits).toEqual([]);
});
