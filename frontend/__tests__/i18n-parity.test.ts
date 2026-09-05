/**
 * I-3: усі локалі мають ОДНАКОВИЙ набір ключів, як у uk (джерело правди).
 * Причина: 05.09 виявилось, що G-2 додав `scenario.myHomeMarked/shareLink/
 * shareCopied` у 5 локалей, але НЕ в uk — українець побачив би сирий ключ.
 * Лічильник «N×6» цього не ловить, коли розбіжності взаємно компенсуються.
 */
import fs from "fs";
import path from "path";

const DIR = path.join(__dirname, "..", "messages");
const LOCALES = ["uk", "en", "de", "pl", "fr", "es"];

function flat(o: Record<string, unknown>, p = ""): string[] {
  return Object.entries(o).flatMap(([k, v]) =>
    v && typeof v === "object" ? flat(v as Record<string, unknown>, `${p}${k}.`) : [`${p}${k}`],
  );
}

/** Дублікати ключів у JSON парсер мовчки «перекриває» останнім значенням —
 *  06.09 так зникли 15 ключів home.footer.* (другий блок "footer"). Ловимо. */
function findDuplicateKeys(text: string): string[] {
  const dups: string[] = [];
  const stack: Array<Set<string>> = [];
  const pathStack: string[] = [];
  const re = /"((?:[^"\\]|\\.)*)"\s*:\s*(\{)?|(\})/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (m[3]) { stack.pop(); pathStack.pop(); continue; }
    const key = m[1];
    const cur = stack[stack.length - 1];
    if (cur) {
      const full = [...pathStack, key].join(".");
      if (cur.has(key)) dups.push(full);
      cur.add(key);
    }
    if (m[2]) { stack.push(new Set()); pathStack.push(key); }
  }
  return dups;
}

describe("i18n parity", () => {
  for (const l of LOCALES) {
    it(`${l} has no duplicate keys`, () => {
      const text = fs.readFileSync(path.join(DIR, `${l}.json`), "utf8");
      // корінь: відкриваюча дужка без ключа
      expect(findDuplicateKeys("\"root\": " + text)).toEqual([]);
    });
  }
  const keys = Object.fromEntries(
    LOCALES.map((l) => [l, new Set(flat(JSON.parse(fs.readFileSync(path.join(DIR, `${l}.json`), "utf8"))))]),
  );
  for (const l of LOCALES.slice(1)) {
    it(`${l} has exactly the uk key set`, () => {
      const missing = [...keys.uk].filter((k) => !keys[l].has(k));
      const extra = [...keys[l]].filter((k) => !keys.uk.has(k));
      expect({ missing, extra }).toEqual({ missing: [], extra: [] });
    });
  }
});
