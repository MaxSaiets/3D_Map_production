import { ImageResponse } from "next/og";

// iOS «Додати на головний екран» / share-картка не використовує manifest-іконку —
// Next з цього файлу авто-додає <link rel="apple-touch-icon">. 180×180 — стандарт Apple.
export const size = { width: 180, height: 180 };
export const contentType = "image/png";

export default function AppleIcon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#2E4A3A",
          color: "#F4EFE4",
          fontSize: 116,
          fontWeight: 700,
          fontFamily: "serif",
          borderRadius: 36,
        }}
      >
        M
      </div>
    ),
    { ...size },
  );
}
