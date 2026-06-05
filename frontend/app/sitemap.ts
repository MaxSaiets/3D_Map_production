import type { MetadataRoute } from "next";

export default function sitemap(): MetadataRoute.Sitemap {
  const base = "https://monadruk.com";
  const now = new Date();
  return [
    { url: `${base}/`, lastModified: now, changeFrequency: "weekly", priority: 1.0 },
    { url: `${base}/create`, lastModified: now, changeFrequency: "monthly", priority: 0.9 },
    { url: `${base}/keychains`, lastModified: now, changeFrequency: "monthly", priority: 0.9 },
  ];
}
