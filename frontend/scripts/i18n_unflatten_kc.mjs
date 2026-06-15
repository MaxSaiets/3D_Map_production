// Fix: the "kc" namespace was stored with flat DOTTED keys (kc["tpl.heart46.name"]),
// but next-intl resolves t("tpl.heart46.name") by NESTED traversal (kc.tpl.heart46.name).
// Flat-dotted keys never resolve -> raw keys render. This converts kc (and any other
// namespace that has dotted keys) to proper nested objects, in all locales, losslessly.
import fs from "node:fs";
import path from "node:path";

const DIR = path.resolve("messages");
const LOCALES = ["uk", "en", "de", "es", "fr", "pl"];

const flatLeaves = (o, p = "", out = {}) => {
  for (const k of Object.keys(o)) {
    const key = p ? `${p}.${k}` : k;
    const v = o[k];
    if (v && typeof v === "object" && !Array.isArray(v)) flatLeaves(v, key, out);
    else out[key] = v;
  }
  return out;
};
const setPath = (root, parts, val) => {
  let cur = root;
  for (let i = 0; i < parts.length - 1; i++) {
    const p = parts[i];
    if (typeof cur[p] !== "object" || cur[p] === null || Array.isArray(cur[p])) cur[p] = {};
    cur = cur[p];
  }
  cur[parts[parts.length - 1]] = val;
};
// rebuild a namespace object as fully-nested, splitting any dotted leaf keys
const nestNamespace = (nsObj) => {
  const leaves = flatLeaves(nsObj); // dotted-path -> value (already splits nested + flat-dotted)
  const out = {};
  for (const [k, v] of Object.entries(leaves)) setPath(out, k.split("."), v);
  return out;
};

// which namespaces actually have dotted keys (need fixing)? compute from uk
const uk = JSON.parse(fs.readFileSync(path.join(DIR, "uk.json"), "utf8"));
const needFix = Object.keys(uk).filter((ns) => {
  const o = uk[ns];
  return o && typeof o === "object" && Object.keys(o).some((k) => k.includes("."));
});
console.log("Namespaces with flat-dotted keys (will nest):", needFix.join(", ") || "(none)");

let problems = 0;
for (const l of LOCALES) {
  const f = path.join(DIR, `${l}.json`);
  const m = JSON.parse(fs.readFileSync(f, "utf8"));
  for (const ns of needFix) {
    if (!m[ns]) continue;
    const before = Object.keys(flatLeaves(m[ns])).sort();
    m[ns] = nestNamespace(m[ns]);
    const after = Object.keys(flatLeaves(m[ns])).sort();
    const identical = before.length === after.length && before.every((x, i) => x === after[i]);
    if (!identical) {
      problems++;
      const miss = before.filter((x) => !after.includes(x));
      const extra = after.filter((x) => !before.includes(x));
      console.log(`  ${l}/${ns} MISMATCH: -${miss.slice(0, 4)} +${extra.slice(0, 4)}`);
    }
  }
  fs.writeFileSync(f, JSON.stringify(m, null, 2) + "\n", "utf8");
  console.log(`  ${l}.json: nested [${needFix}] (kc leaves=${Object.keys(flatLeaves(m.kc || {})).length})`);
}
console.log(problems ? `\n⚠ ${problems} mismatches — DO NOT trust output` : "\n✓ All namespaces nested losslessly (every leaf path preserved).");
