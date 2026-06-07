import { ImageResponse } from "next/og";
import { getTranslations } from "next-intl/server";
import { routing } from "@/i18n/routing";

export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const alt = "Monadruk — 3D maps & keychains of your city";

export function generateStaticParams() {
  return routing.locales.map((locale) => ({ locale }));
}

export default async function OpengraphImage({ params }: { params: { locale: string } }) {
  const locale = (routing.locales as readonly string[]).includes(params.locale) ? params.locale : routing.defaultLocale;
  const t = await getTranslations({ locale, namespace: "meta" });

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%", height: "100%", display: "flex", flexDirection: "column",
          justifyContent: "space-between", padding: "72px 80px",
          background: "linear-gradient(135deg,#F4EFE4 0%,#E7DDC9 100%)", color: "#1B2A22",
          fontFamily: "Georgia, serif",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16, fontSize: 34, fontWeight: 600 }}>
          <div style={{ width: 40, height: 40, borderRadius: 10, background: "#2E4A3A" }} />
          monadruk
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div style={{ fontSize: 64, fontWeight: 600, lineHeight: 1.05, maxWidth: 920 }}>{t("ogTitle")}</div>
          <div style={{ fontSize: 30, color: "#3c4a42", maxWidth: 900, fontFamily: "Arial, sans-serif" }}>{t("ogDescription")}</div>
        </div>
        <div style={{ display: "flex", gap: 14, fontSize: 24, color: "#2E4A3A", fontFamily: "Arial, sans-serif", fontWeight: 600 }}>
          <span>3MF · STL</span><span>·</span><span>Eco PLA</span><span>·</span><span>monadruk.com</span>
        </div>
      </div>
    ),
    size,
  );
}
