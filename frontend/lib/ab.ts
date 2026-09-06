// Легка A/B-інфраструктура: деterмінований варіант A/B на основі visitorId,
// що зберігається в localStorage (переживає сесії, не потребує бекенд-кукі).
// SSR-safe: на сервері немає localStorage — повертаємо стабільний дефолт "A".

const VISITOR_ID_KEY = "mnd_vid";

/** Активні експерименти — єдине джерело правди для abProps() і будь-якого
 *  коду, що хоче знати, які A/B зараз крутяться. */
export const ACTIVE_EXPERIMENTS = ["cta"] as const;
export type ExperimentId = (typeof ACTIVE_EXPERIMENTS)[number];

function randomId(len = 12): string {
  const chars = "abcdefghijklmnopqrstuvwxyz0123456789";
  let out = "";
  for (let i = 0; i < len; i++) out += chars[Math.floor(Math.random() * chars.length)];
  return out;
}

/** Отримати (і за потреби створити) стабільний ідентифікатор відвідувача.
 *  SSR: повертає "" (без localStorage) — виклик getVariant з цим id завжди
 *  дає детермінований, хоч і не персональний, результат. */
export function getVisitorId(): string {
  if (typeof window === "undefined" || typeof window.localStorage === "undefined") return "";
  try {
    let id = window.localStorage.getItem(VISITOR_ID_KEY);
    if (!id) {
      id = randomId(12);
      window.localStorage.setItem(VISITOR_ID_KEY, id);
    }
    return id;
  } catch {
    return "";
  }
}

/** Простий детермінований хеш рядка (FNV-1a-подібний) → 32-бітне беззнакове число. */
function hashString(str: string): number {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** Детермінований варіант A/B для експерименту `exp`: той самий visitorId
 *  завжди дає той самий варіант, ~50/50 розподіл між відвідувачами.
 *  На сервері (SSR) завжди повертає "A" — реальний варіант призначається
 *  після монтування на клієнті (щоб уникнути hydration mismatch). */
export function getVariant(exp: string): "A" | "B" {
  if (typeof window === "undefined") return "A";
  const vid = getVisitorId();
  if (!vid) return "A";
  const h = hashString(`${vid}|${exp}`);
  return h % 2 === 0 ? "A" : "B";
}

/** Пропси для аналітики: {ab_<exp>: "A"|"B"} для КОЖНОГО активного експерименту.
 *  Порожній обʼєкт на сервері (немає стабільного visitorId для SSR-рендеру). */
export function abProps(): Record<string, "A" | "B"> {
  if (typeof window === "undefined") return {};
  const out: Record<string, "A" | "B"> = {};
  for (const exp of ACTIVE_EXPERIMENTS) {
    out[`ab_${exp}`] = getVariant(exp);
  }
  return out;
}
