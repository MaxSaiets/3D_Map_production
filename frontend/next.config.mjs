import createNextIntlPlugin from "next-intl/plugin";

const withNextIntl = createNextIntlPlugin("./i18n/request.ts");

// Безпекові заголовки на всі відповіді (security-audit hardening). БЕЗ строгого CSP
// — він легко ламає Firebase/Leaflet/Three/GA і потребує окремого тестування.
// geolocation=(self) ЛИШАЄМО — карта використовує «📍 я тут» (MapSearchBox).
const securityHeaders = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "X-Frame-Options", value: "SAMEORIGIN" },
  { key: "X-DNS-Prefetch-Control", value: "on" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), browsing-topics=(), geolocation=(self)" },
];

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false, // не рекламувати "X-Powered-By: Next.js" (узгоджено з securityHeaders)
  // Zero-downtime деплой: build у ОКРЕМУ теку (NEXT_DIST_DIR=.next-build) поки старий
  // фронт ще обслуговує з .next, потім атомарний swap + restart (~секунди замість ~6хв
  // простою від rm -rf .next). За замовчуванням .next — звичайний build/start не зачеплено.
  distDir: process.env.NEXT_DIST_DIR || ".next",
  // collectBuildTraces (.nft.json) потрібен ЛИШЕ для output:'standalone'/serverless.
  // Тут продакшн = `next start` із повним .next, тож трейси не використовуються.
  // А з кастомним distDir (.next-build для zero-downtime) цей крок РЕГУЛЯРНО падав
  // `Cannot find module next/.../app-page/module.compiled` (відома вада Next 14 при
  // нестандартному distDir) і валив весь білд. Вимкнення прибирає цей крах + пришвидшує білд.
  outputFileTracing: false,
  images: {
    unoptimized: true,
  },
  async headers() {
    return [
      { source: "/:path*", headers: securityHeaders },
      // SEO: API не має потрапляти в індекс (підстраховка поверх robots.txt).
      { source: "/api/:path*", headers: [{ key: "X-Robots-Tag", value: "noindex, nofollow" }] },
    ];
  },
  webpack: (config) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
      crypto: false,
    };
    return config;
  },
};

export default withNextIntl(nextConfig);

