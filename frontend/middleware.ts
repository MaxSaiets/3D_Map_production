import createMiddleware from "next-intl/middleware";
import { routing } from "./i18n/routing";

// Auto-detects locale from the cookie (NEXT_LOCALE) then Accept-Language header,
// and rewrites/redirects accordingly. Default locale (uk) stays unprefixed.
export default createMiddleware(routing);

export const config = {
  // Skip API, Next internals, and any path with a file extension (assets,
  // models, sitemap.xml, robots.txt, manifest, icons, og-image, etc.).
  matcher: ["/((?!api|capture|_next|.*\\..*).*)"],
};
