import { GALLERY_ITEMS, GALLERY_LOCALES } from "@/lib/gallery";
import { PEAKS, PEAK_LOCALES } from "@/lib/mountainPages";

/**
 * Image sitemap для Google Картинок (Next 14.2 MetadataRoute.Sitemap не вміє
 * <image:image>). Кожна /foto-сторінка (uk) + її фото з підписом; галерея
 * /showcase — усі фото разом. Посилання з robots.ts.
 */
export const dynamic = "force-static";

const BASE = "https://monadruk.com";
const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&apos;");

export function GET() {
  const img = (src: string) => `<image:image><image:loc>${esc(BASE + src)}</image:loc></image:image>`;
  const urls = [
    `<url><loc>${BASE}/showcase</loc>${GALLERY_ITEMS.map((g) => img(g.src)).join("")}</url>`,
    ...GALLERY_LOCALES.flatMap((l) =>
      GALLERY_ITEMS.map((g) => `<url><loc>${esc(`${BASE}${l === "uk" ? "" : `/${l}`}/foto/${g.slug}`)}</loc>${img(g.src)}</url>`),
    ),
    // Сторінки вершин /gory/[slug] — фото вершини (25.09.2026).
    ...PEAK_LOCALES.flatMap((l) =>
      PEAKS.map((p) => `<url><loc>${esc(`${BASE}${l === "uk" ? "" : `/${l}`}/gory/${p.slug}`)}</loc>${img(p.photo)}</url>`),
    ),
  ];
  const xml =
    `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">\n` +
    urls.join("\n") +
    `\n</urlset>\n`;
  return new Response(xml, { headers: { "Content-Type": "application/xml; charset=utf-8" } });
}
