import type { MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        allow: "/",
        // Workspace/account routes have no SEO value and shouldn't be indexed.
        // Wildcard-варіанти покривають локалізовані префікси (/en/account тощо).
        // /*/opengraph-image НЕ блокуємо: middleware тепер віддає їм 410 Gone,
        // а щоб Google ПОБАЧИВ 410 і викинув старі URL, краул має бути дозволений
        // (robots-блок ховає статус-код і лишає URL висіти в «Заблоковано robots»).
        disallow: ["/api/", "/account", "/admin", "/*/account", "/*/admin", "/capture", "/*/capture", "/share/", "/*/share/"],
      },
    ],
    sitemap: "https://monadruk.com/sitemap.xml",
    host: "https://monadruk.com",
  };
}
