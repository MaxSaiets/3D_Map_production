import type { MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        allow: "/",
        // Workspace/account routes have no SEO value and shouldn't be indexed.
        // Wildcard-варіанти покривають локалізовані префікси (/en/account тощо).
        disallow: ["/api/", "/account", "/admin", "/*/account", "/*/admin", "/capture", "/*/capture", "/share/", "/*/share/"],
      },
    ],
    sitemap: "https://monadruk.com/sitemap.xml",
    host: "https://monadruk.com",
  };
}
