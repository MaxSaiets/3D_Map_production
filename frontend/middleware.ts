import createMiddleware from "next-intl/middleware";
import type { NextRequest } from "next/server";
import { routing } from "./i18n/routing";

const intl = createMiddleware(routing);

// Auto-detects locale from the cookie (NEXT_LOCALE) then Accept-Language header,
// and rewrites/redirects accordingly. Default locale (uk) stays unprefixed.
export default function middleware(req: NextRequest) {
  // 410 Gone для вкладених opengraph-image/twitter-image: Next file-convention
  // колись генерував OG-роут для КОЖНОГО сегмента (/pl/maps/kyiv/opengraph-image),
  // під next-intl вони 307→404 і висіли в GSC як «Не знайдено 404». 410 каже
  // Google «зникло назавжди» — викидає з черги швидше за 404. Корінь
  // /opengraph-image (робочі соц-картки) сюди не потрапляє — виключений matcher-ом.
  if (/\/(opengraph-image|twitter-image)$/.test(req.nextUrl.pathname)) {
    return new Response(null, { status: 410 });
  }
  return intl(req);
}

export const config = {
  // Skip API, Next internals, root metadata routes, and any path with a file
  // extension. ВАЖЛИВО: `icon`/`apple-icon`/`opengraph-image`/`manifest` живуть у
  // app/ КОРЕНІ (поза [locale]) і НЕ мають крапки → раніше їх ловив локаль-middleware
  // і вони 404-или (фавікон, Org-логотип, apple-touch-icon не вантажились). Явно
  // виключаємо їх. (sitemap.xml/robots.txt працювали бо мають крапку → .*\..*).
  matcher: ["/((?!api|capture|_next|icon|apple-icon|opengraph-image|twitter-image|manifest|sitemap|robots|favicon|.*\\..*).*)"],
};
