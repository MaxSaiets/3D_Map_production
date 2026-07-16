import type { MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        allow: "/",
        // Workspace/account routes have no SEO value and shouldn't be indexed.
        // Wildcard-варіанти покривають локалізовані префікси (/en/account тощо).
        // /*/opengraph-image — Next.js file-convention генерує OG-роут для КОЖНОГО
        // сегмента (/pl/maps/kyiv/opengraph-image тощо), який під next-intl 307→404.
        // Це асети, не сторінки — блокуємо їх краул (корінь /opengraph-image лишається
        // дозволеним для соц-карток). Прибирає 12 «404» з GSC-звіту індексації.
        disallow: ["/api/", "/account", "/admin", "/*/account", "/*/admin", "/capture", "/*/capture", "/share/", "/*/share/", "/*/opengraph-image"],
      },
    ],
    sitemap: "https://monadruk.com/sitemap.xml",
    host: "https://monadruk.com",
  };
}
