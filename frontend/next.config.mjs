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
  images: {
    unoptimized: true,
  },
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
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

