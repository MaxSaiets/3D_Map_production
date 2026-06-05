import { ImageResponse } from "next/og";

export const alt = "Monadruk — 3D-мапи твого міста для друку";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OgImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          background: "linear-gradient(135deg, #2E4A3A 0%, #1B2A22 100%)",
          padding: "72px",
          color: "#F4EFE4",
          fontFamily: "serif",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16, fontSize: 30, letterSpacing: 4 }}>
          <div
            style={{
              width: 44,
              height: 44,
              borderRadius: 12,
              background: "#8E6B3D",
              display: "flex",
            }}
          />
          MONADRUK
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ fontSize: 86, fontWeight: 600, lineHeight: 1.05 }}>Твоє місто.</div>
          <div style={{ fontSize: 86, fontWeight: 600, fontStyle: "italic", color: "#D9C29A", lineHeight: 1.05 }}>
            Виміряне в 3D.
          </div>
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", fontSize: 30 }}>
          <div style={{ opacity: 0.85, maxWidth: 720 }}>
            Тактильні 3D-мапи й брелки твого міста. Завантаж готовий 3MF для друку.
          </div>
          <div style={{ fontSize: 26, color: "#D9C29A" }}>monadruk.com</div>
        </div>
      </div>
    ),
    { ...size },
  );
}
