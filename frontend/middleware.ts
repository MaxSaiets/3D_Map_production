import createMiddleware from "next-intl/middleware";
import { routing } from "./i18n/routing";

// Auto-detects locale from the cookie (NEXT_LOCALE) then Accept-Language header,
// and rewrites/redirects accordingly. Default locale (uk) stays unprefixed.
export default createMiddleware(routing);

export const config = {
  // Skip API, Next internals, root metadata routes, and any path with a file
  // extension. ВАЖЛИВО: `icon`/`apple-icon`/`opengraph-image`/`manifest` живуть у
  // app/ КОРЕНІ (поза [locale]) і НЕ мають крапки → раніше їх ловив локаль-middleware
  // і вони 404-или (фавікон, Org-логотип, apple-touch-icon не вантажились). Явно
  // виключаємо їх. (sitemap.xml/robots.txt працювали бо мають крапку → .*\..*).
  matcher: ["/((?!api|capture|_next|icon|apple-icon|opengraph-image|twitter-image|manifest|sitemap|robots|favicon|.*\\..*).*)"],
};
