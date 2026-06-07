# i18n — як додавати мови / сторінки / тексти

Стек: **next-intl** + App Router `app/[locale]/...`.
Мови: `uk` (типова, без префікса) · `en` · `de` · `pl` · `fr` · `es`.
Автовизначення мови — `middleware.ts` (cookie `NEXT_LOCALE` → `Accept-Language`).

## Файли
- `i18n/routing.ts` — список локалей, типова локаль, `localeMeta` (підписи + hreflang/og теги).
- `i18n/request.ts` — завантаження словника для локалі.
- `i18n/navigation.ts` — **locale-aware** `Link`, `useRouter`, `usePathname`. Використовуй ЦЕЙ `Link` для внутрішніх посилань (не `next/link`), щоб мова зберігалась.
- `i18n/metadata.ts` — `pageMetadata()` будує title/description/canonical/**hreflang**/OG однією функцією.
- `messages/<locale>.json` — словники (namespace → ключі).

## Додати НОВУ сторінку (наприклад `/pricing`)
1. Створи `app/[locale]/pricing/page.tsx` (можна `"use client"`).
2. У `messages/*.json` додай namespace `pricing` (тексти) і `pricingMeta` (`title`,`description`,`keywords`).
3. Додай SEO-метадані через `app/[locale]/pricing/layout.tsx`:
   ```tsx
   import { pageMetadata } from "@/i18n/metadata";
   export async function generateMetadata({ params }) {
     return pageMetadata({ locale: params.locale, path: "/pricing", ns: "pricingMeta" });
   }
   export default function L({ children }) { return children; }
   ```
4. У сторінці: `const t = useTranslations("pricing"); ... {t("title")}`.
5. Додай шлях у `app/sitemap.ts` (масив `PATHS`) — hreflang підставиться автоматично.

## Додати НОВУ мову (наприклад `it`)
1. `i18n/routing.ts`: додай `"it"` у `locales` + запис у `localeMeta`.
2. Скопіюй `messages/uk.json` → `messages/it.json` і переклади значення.
Все інше (маршрути, hreflang, sitemap, перемикач) підхопить автоматично.

## Додати/змінити текст
Редагуй `messages/<locale>.json`. У компоненті: `const t = useTranslations("namespace"); {t("key")}`.

## Що ще НЕ перекладено (тексти в коді, не критичні для SEO)
- Глибокі рядки конструктора: `components/ControlPanel.tsx`, `KeychainDesigner.tsx`,
  `SimpleControlPanel.tsx`, `WizardSteps.tsx`, `OrderDialog.tsx`, `ContactWidget.tsx`, `app/[locale]/account`.
- Юридичні тексти `privacy`/`terms` (тіло) — лишені укр. навмисно (метадані локалізовані).
Патерн той самий: винеси рядки в namespace і заміни на `t(...)`.
