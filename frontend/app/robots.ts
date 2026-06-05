import type { MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        allow: "/",
        // Workspace/account routes have no SEO value and shouldn't be indexed.
        disallow: ["/api/", "/account", "/admin"],
      },
    ],
    sitemap: "https://monadruk.com/sitemap.xml",
    host: "https://monadruk.com",
  };
}
