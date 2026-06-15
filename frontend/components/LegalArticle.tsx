import React from "react";
import { Link } from "@/i18n/navigation";
import { BUSINESS, IBAN_DISPLAY } from "@/lib/legal";
import { localeUrl } from "@/i18n/metadata";
import { type AppLocale } from "@/i18n/routing";
import { type LegalDoc, type LegalBlock, LEGAL_LABELS } from "@/lib/legal/content";

const linkCls = "text-forest underline-offset-2 hover:underline";

// BUSINESS.updated зберігається як ISO (2026-06-15) → форматуємо під локаль.
const LOCALE_TAG: Record<string, string> = { uk: "uk-UA", en: "en-US", de: "de-DE", es: "es-ES", fr: "fr-FR", pl: "pl-PL" };
function formatUpdated(locale: string): string {
  try {
    return new Intl.DateTimeFormat(LOCALE_TAG[locale] ?? "uk-UA", { year: "numeric", month: "long", day: "numeric" })
      .format(new Date(`${BUSINESS.updated}T00:00:00`));
  } catch {
    return BUSINESS.updated;
  }
}

// Підстановка токенів {data} та посилань [route:текст] у рядку → React-вузли.
function renderText(text: string, locale: string): React.ReactNode[] {
  const parts = text.split(/(\{[a-zA-Z]+\}|\[[a-z]+:[^\]]+\])/g).filter((p) => p !== "");
  return parts.map((part, i) => {
    // Дата-токени
    if (part.startsWith("{") && part.endsWith("}")) {
      const key = part.slice(1, -1);
      switch (key) {
        case "email": return <a key={i} className={linkCls} href={`mailto:${BUSINESS.email}`}>{BUSINESS.email}</a>;
        case "phone": return <a key={i} className={linkCls} href={`tel:${BUSINESS.phone}`}>{BUSINESS.phoneDisplay}</a>;
        case "domain": return <a key={i} className={linkCls} href={BUSINESS.site}>{BUSINESS.domain}</a>;
        case "site": return <React.Fragment key={i}>{BUSINESS.site}</React.Fragment>;
        case "iban": return <span key={i} className="tabular-nums">{IBAN_DISPLAY}</span>;
        case "ownerFull": return <React.Fragment key={i}>{BUSINESS.ownerFull}</React.Fragment>;
        case "ownerShort": return <React.Fragment key={i}>{BUSINESS.ownerShort}</React.Fragment>;
        case "taxId": return <React.Fragment key={i}>{BUSINESS.taxId}</React.Fragment>;
        case "ved": return <React.Fragment key={i}>{BUSINESS.ved}</React.Fragment>;
        case "storeName": return <React.Fragment key={i}>{BUSINESS.storeName}</React.Fragment>;
        case "storeAddress": return <React.Fragment key={i}>{BUSINESS.storeAddress}</React.Fragment>;
        case "ownerRegAddress": return <React.Fragment key={i}>{BUSINESS.ownerRegAddress}</React.Fragment>;
        case "updated": return <React.Fragment key={i}>{formatUpdated(locale)}</React.Fragment>;
        default: return <React.Fragment key={i}>{part}</React.Fragment>;
      }
    }
    // Посилання [route:видимий текст]
    if (part.startsWith("[") && part.endsWith("]")) {
      const inner = part.slice(1, -1);
      const idx = inner.indexOf(":");
      const route = inner.slice(0, idx);
      const label = inner.slice(idx + 1);
      return <Link key={i} className={linkCls} href={`/${route}`}>{label}</Link>;
    }
    return <React.Fragment key={i}>{part}</React.Fragment>;
  });
}

function Block({ block, locale }: { block: LegalBlock; locale: string }) {
  if ("p" in block) return <p>{renderText(block.p, locale)}</p>;
  if ("ul" in block) {
    return (
      <ul className="list-disc space-y-1.5 pl-5">
        {block.ul.map((item, i) => <li key={i}>{renderText(item, locale)}</li>)}
      </ul>
    );
  }
  // kv — реквізити (мітка: значення)
  return (
    <ul className="space-y-1.5">
      {block.kv.map((row, i) => (
        <li key={i}>{row.k}: {renderText(row.v, locale)}</li>
      ))}
    </ul>
  );
}

export function LegalArticle({ doc, locale, path }: { doc: LegalDoc; locale: string; path?: string }) {
  const updatedLabel = (LEGAL_LABELS[locale] ?? LEGAL_LABELS.uk).updated;
  // BreadcrumbList — Google показує шлях «monadruk.com › <Сторінка>» у видачі.
  const breadcrumb = path
    ? {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "Monadruk", item: localeUrl(locale as AppLocale, "/") },
          { "@type": "ListItem", position: 2, name: doc.title, item: localeUrl(locale as AppLocale, path) },
        ],
      }
    : null;
  return (
    <>
      {breadcrumb && <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumb) }} />}
      <h1 className="mt-4 font-serif text-[clamp(28px,4vw,42px)] text-ink">{doc.title}</h1>
      <p className="mt-2 text-[13px] text-ink-3">{updatedLabel}: {formatUpdated(locale)}</p>
      {doc.intro && (
        <div className="mt-8 space-y-3 text-[15px] leading-relaxed text-ink-2">
          {doc.intro.map((p, i) => <p key={i}>{renderText(p, locale)}</p>)}
        </div>
      )}
      <div className="mt-6 space-y-6 text-[15px] leading-relaxed text-ink-2">
        {doc.sections.map((s, i) => (
          <section key={i}>
            <h2 className="mb-2 font-serif text-xl text-ink">{s.h}</h2>
            <div className="space-y-2">
              {s.blocks.map((b, j) => <Block key={j} block={b} locale={locale} />)}
            </div>
          </section>
        ))}
      </div>
    </>
  );
}
