import { PAGE_AI_TEXT } from "@/lib/pageAiText";

/**
 * 27.09.2026: унікальний текст сторінки міста/району/вулиці (tools/ai_page_texts.py).
 * ШІ лише переказує перевірені факти (Wikidata, OSM); кожен текст пройшов валідатор
 * (без тире, без кліше, числа з фактів). Немає тексту для сторінки → нічого не рендеримо.
 */
export default function AiPageText({ id, isUA }: { id: string; isUA: boolean }) {
  const t = PAGE_AI_TEXT[id];
  const paras = t ? (isUA ? t.uk : t.en) : null;
  if (!paras?.length) return null;
  return (
    <div className="mt-3 flex flex-col gap-3">
      {paras.map((p, i) => (
        <p key={i} className="text-[15px] leading-relaxed text-ink-2">{p}</p>
      ))}
    </div>
  );
}
