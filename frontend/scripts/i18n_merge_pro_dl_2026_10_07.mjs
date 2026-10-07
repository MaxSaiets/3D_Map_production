// Підпис під «Завантажити» для Pro/безліму (07.10.2026). node scripts/i18n_merge_pro_dl_2026_10_07.mjs
import fs from "node:fs"; import path from "node:path";
const T = {
  uk: "Безлім у вашому акаунті — файл без доплати · готується ≈2 хв",
  en: "Unlimited on your account — no extra charge · ready in about 2 min",
  de: "Unbegrenzt in Ihrem Konto — ohne Aufpreis · fertig in ca. 2 Min.",
  es: "Ilimitado en tu cuenta — sin coste extra · listo en unos 2 min",
  fr: "Illimité sur votre compte — sans supplément · prêt en 2 min environ",
  pl: "Bez limitu na Twoim koncie — bez dopłaty · gotowe w ok. 2 min",
};
for (const l of Object.keys(T)) {
  const f = path.resolve("messages", `${l}.json`);
  const j = JSON.parse(fs.readFileSync(f, "utf8"));
  j.scenario = { ...j.scenario, downloadSubUnlimited: T[l] };
  j.kcScenario = { ...j.kcScenario, downloadSubUnlimited: T[l] };
  fs.writeFileSync(f, JSON.stringify(j, null, 2) + "\n", "utf8");
}
console.log("ok");
