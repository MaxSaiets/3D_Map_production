/**
 * 07.10.2026: `text-white/72`, `bg-white/8` тощо — значення ПОЗА шкалою opacity Tailwind
 * (0, 5, 10 … 95, 100) → клас не генерується взагалі. Так «Зворот» у макеті брелока
 * мав темний текст на темному тлі (невидимий). Довільне значення пишіть як `/[.72]`.
 */
import fs from "fs";
import path from "path";

const ROOT = path.join(__dirname, "..");
const SCALE = new Set([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]);
const RE = /\b(?:bg|text|border|ring|from|to|via|fill|stroke|shadow|outline|divide|placeholder|decoration)-[a-z]+(?:-\d{2,3})?\/(\d+)\b/g;

function walk(dir: string, out: string[] = []): string[] {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p, out);
    else if (/\.(tsx|ts|jsx|js)$/.test(e.name)) out.push(p);
  }
  return out;
}

it("opacity modifiers stay on the Tailwind scale", () => {
  const hits: string[] = [];
  for (const d of ["app", "components"]) {
    for (const f of walk(path.join(ROOT, d))) {
      const src = fs.readFileSync(f, "utf8");
      for (const m of src.matchAll(RE)) {
        if (!SCALE.has(Number(m[1]))) hits.push(`${path.relative(ROOT, f)}: ${m[0]}`);
      }
    }
  }
  expect(hits).toEqual([]);
});
