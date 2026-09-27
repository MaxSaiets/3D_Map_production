import { PAGE_OSM_STATS, type OsmStats } from "@/lib/pageOsmStats";
import type { FaqItem } from "@/lib/cityLanding";

/**
 * 27.09.2026: «Що буде на вашій 3D-моделі» — реальні дані OpenStreetMap для
 * ділянки 800×800 м кожної сторінки (tools/osm_page_stats.py): кількість
 * будівель, поверховість, назви парків/водойм/пам'яток. Це і користь покупцю,
 * і унікальний текст сторінки (різні числа й назви), без вигаданих фактів.
 */

function h(s: string): number {
  let x = 2166136261;
  for (let i = 0; i < s.length; i++) { x ^= s.charCodeAt(i); x = Math.imul(x, 16777619); }
  return Math.abs(x);
}

// Не показуємо: радянську/імперську топоніміку, російськомовні назви, банки й установи.
const BAD_POI = /лен[іi]н|тургенєв|ворошилов|артему$|^артем|ватутін|пушкін|радянськ|визволител|кіров|щорс|чапаєв|дзержинськ|жуков|комсомол|жовтнев|[ыэъё]|\bbank\b|банк|горздрав|приймальна/i;
function cleanList(xs?: string[]): string[] {
  return (xs ?? []).filter((n) => !BAD_POI.test(n) && n.length > 3);
}
function cleanStats(st: OsmStats): OsmStats {
  return { ...st, parks: cleanList(st.parks), water: cleanList(st.water), pois: (st.pois ?? []).filter(([n]) => !BAD_POI.test(n) && n.length > 3) };
}

const KIND_UK: Record<string, string> = { church: "храм", museum: "музей", theatre: "театр", university: "університет", castle: "замок", monument: "памʼятник" };

function lines(st: OsmStats, name: string, isUA: boolean, id: string): string[] {
  const nf = new Intl.NumberFormat(isUA ? "uk-UA" : "en-US");
  const v = h(id) % 2;
  const out: string[] = [];
  const b = st.b;
  const character = st.avgLv
    ? st.avgLv >= 9
      ? isUA ? "висотна забудова — модель виходить обʼємною, з чіткими «вежами»" : "high-rise blocks — the model comes out bold, with clear towers"
      : st.avgLv >= 5
        ? isUA ? "середньоповерхова забудова — будинки добре читаються окремо" : "mid-rise blocks — buildings read clearly one by one"
        : isUA ? "малоповерхова забудова — вулиці й рельєф на моделі видно особливо добре" : "low-rise blocks — streets and terrain show especially well"
    : "";
  const density = b > 600
    ? isUA ? "дуже щільна ділянка — модель вийде насиченою деталями" : "a very dense area — the model will be packed with detail"
    : b < 150
      ? isUA ? "простора ділянка з великими відкритими місцями" : "an open area with plenty of free space"
      : "";
  if (isUA) {
    const lv = st.maxLv ? ` Найвища будівля в кадрі має ${st.maxLv} поверхів${st.avgLv ? `, у середньому — ${nf.format(st.avgLv)}` : ""}.` : "";
    out.push(
      v === 0
        ? `У квадрат 800 × 800 м навколо центру (${name}) потрапляє близько ${nf.format(b)} будівель — кожна з реальною висотою.${lv}`
        : `Близько ${nf.format(b)} будівель — стільки опиняється на мапі 8 см, якщо взяти ділянку 800 × 800 м (${name}).${lv}`,
    );
    const ch = [character, density].filter(Boolean).join("; ");
    if (ch) out.push(`Характер ділянки: ${ch}.`);
    if (st.parks?.length) out.push(`Зелені зони, які друкуються окремим кольором: ${st.parks.join(", ")}.`);
    if (st.water?.length) out.push(`Вода в кадрі: ${st.water.join(", ")} — на моделі вона трохи нижча за землю, тож берег читається рельєфно.`);
    if (st.pois?.length) out.push(`Впізнавані місця всередині ділянки: ${st.pois.slice(0, 6).map(([n, k]) => (KIND_UK[k] ? `${n} (${KIND_UK[k]})` : n)).join(", ")}.`);
  } else {
    const lv = st.maxLv ? ` The tallest building in frame has ${st.maxLv} floors${st.avgLv ? `, ${nf.format(st.avgLv)} on average` : ""}.` : "";
    out.push(`An 800 × 800 m square around the centre of ${name} holds about ${nf.format(b)} buildings, each with its real height.${lv}`);
    const ch = [character, density].filter(Boolean).join("; ");
    if (ch) out.push(`What the area is like: ${ch}.`);
    if (st.parks?.length) out.push(`Green areas printed in a separate colour: ${st.parks.join(", ")}.`);
    if (st.water?.length) out.push(`Water in frame: ${st.water.join(", ")} — slightly lower than the ground, so the shoreline reads in relief.`);
    if (st.pois?.length) out.push(`Recognisable places inside the area: ${st.pois.slice(0, 6).map(([n]) => n).join(", ")}.`);
  }
  return out;
}

/** Питання FAQ саме про цю ділянку (додаються до загального FAQ і в JSON-LD). */
export function osmFaq(id: string, name: string, isUA: boolean): FaqItem[] {
  const raw = PAGE_OSM_STATS[id];
  if (!raw) return [];
  const st = cleanStats(raw);
  const nf = new Intl.NumberFormat(isUA ? "uk-UA" : "en-US");
  const f: FaqItem[] = [];
  if (isUA) {
    f.push({ q: `Скільки будинків буде на 3D-моделі (${name})?`, a: `На мапі 8 см з ділянкою 800 × 800 м — близько ${nf.format(st.b)} будівель${st.maxLv ? `, найвища — ${st.maxLv} поверхів` : ""}. Якщо збільшити розмір моделі чи зсунути рамку, кількість зміниться — превʼю в конструкторі показує точний результат.` });
    const p = st.pois?.[0]?.[0] ?? st.parks?.[0];
    if (p) f.push({ q: `Чи буде на моделі видно: ${p}?`, a: `Так, «${p}» — у межах ділянки, яку ми показуємо для цієї сторінки. Щоб він опинився в центрі мапи, просто посуньте рамку в конструкторі.` });
  } else {
    f.push({ q: `How many buildings will the 3D model of ${name} have?`, a: `An 8 cm map of an 800 × 800 m area holds about ${nf.format(st.b)} buildings${st.maxLv ? `, the tallest with ${st.maxLv} floors` : ""}. Resize or move the frame and the count changes — the builder preview shows the exact result.` });
    const p = st.pois?.[0]?.[0] ?? st.parks?.[0];
    if (p) f.push({ q: `Will ${p} be visible on the model?`, a: `Yes — ${p} lies inside the area shown for this page. Move the frame in the builder to put it right in the centre.` });
  }
  return f;
}

/** Загальні питання — лише 2 з 5, різні для різних сторінок (менше дублю між сторінками). */
export function rotateFaq(all: FaqItem[], id: string, keep = 2): FaqItem[] {
  if (all.length <= keep) return all;
  const start = h(id) % all.length;
  return Array.from({ length: keep }, (_, i) => all[(start + i * 2) % all.length]);
}

export default function OsmModelContents({ id, name, isUA }: { id: string; name: string; isUA: boolean }) {
  const raw = PAGE_OSM_STATS[id];
  if (!raw || !raw.b) return null;
  const st = cleanStats(raw);
  return (
    <section className="mt-10">
      <h2 className="text-[20px] font-semibold">{isUA ? `Що буде на вашій 3D-моделі: ${name}` : `What your 3D model of ${name} will show`}</h2>
      {lines(st, name, isUA, id).map((p, i) => (
        <p key={i} className="mt-3 text-[15px] leading-relaxed text-ink-2">{p}</p>
      ))}
      <p className="mt-2 text-[12.5px] text-ink-3">{isUA ? "Дані: OpenStreetMap, ділянка 800 × 800 м." : "Data: OpenStreetMap, 800 × 800 m area."}</p>
    </section>
  );
}
