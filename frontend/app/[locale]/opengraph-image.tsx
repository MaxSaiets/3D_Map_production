import { ImageResponse } from "next/og";
import { routing } from "@/i18n/routing";

export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

// OG images are generated outside the next-intl React provider (edge/og context),
// so we cannot use next-intl hooks here. Keep the per-locale copy inline instead.
type OgStrings = {
  alt: string;
  title1: string; // headline line 1
  title2: string; // headline line 2 (accent)
  sub: string; // supporting subline
};

const OG_TEXT: Record<string, OgStrings> = {
  uk: {
    alt: "Monadruk — 3D-мапи й брелки твого міста",
    title1: "Твоє місто.",
    title2: "Виміряне в 3D.",
    sub: "Тактильні 3D-мапи й брелки твого міста. Завантаж готовий 3MF для друку.",
  },
  en: {
    alt: "Monadruk — 3D maps & keychains of your city",
    title1: "Your city.",
    title2: "Measured in 3D.",
    sub: "Tactile 3D maps and keychains of your city. Download a print-ready 3MF.",
  },
  de: {
    alt: "Monadruk — 3D-Karten & Schlüsselanhänger deiner Stadt",
    title1: "Deine Stadt.",
    title2: "In 3D vermessen.",
    sub: "Taktile 3D-Karten und Schlüsselanhänger deiner Stadt. Lade die druckfertige 3MF herunter.",
  },
  es: {
    alt: "Monadruk — mapas 3D y llaveros de tu ciudad",
    title1: "Tu ciudad.",
    title2: "Medida en 3D.",
    sub: "Mapas 3D táctiles y llaveros de tu ciudad. Descarga un 3MF listo para imprimir.",
  },
  fr: {
    alt: "Monadruk — cartes 3D et porte-clés de ta ville",
    title1: "Ta ville.",
    title2: "Mesurée en 3D.",
    sub: "Cartes 3D tactiles et porte-clés de ta ville. Télécharge un 3MF prêt à imprimer.",
  },
  pl: {
    alt: "Monadruk — mapy 3D i breloki Twojego miasta",
    title1: "Twoje miasto.",
    title2: "Zmierzone w 3D.",
    sub: "Dotykowe mapy 3D i breloki Twojego miasta. Pobierz gotowy do druku plik 3MF.",
  },
};

function pickLocale(locale: string): string {
  return (routing.locales as readonly string[]).includes(locale) ? locale : routing.defaultLocale;
}

// `alt` must be a per-locale string; export a sensible English+brand default as the
// module const (Next.js reads `alt` statically) and derive the localized value at render.
export const alt = OG_TEXT.en.alt;

export function generateStaticParams() {
  return routing.locales.map((locale) => ({ locale }));
}

export default async function OpengraphImage({ params }: { params: { locale: string } }) {
  const locale = pickLocale(params.locale);
  const text = OG_TEXT[locale] ?? OG_TEXT.uk;

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
        <div style={{ display: "flex", flexDirection: "column", gap: 4, maxWidth: 920 }}>
          <div style={{ fontSize: 64, fontWeight: 600, lineHeight: 1.05 }}>{text.title1}</div>
          <div style={{ fontSize: 64, fontWeight: 600, lineHeight: 1.05, fontStyle: "italic", color: "#8E6B3D" }}>{text.title2}</div>
          <div style={{ fontSize: 30, color: "#3c4a42", maxWidth: 900, marginTop: 16, fontFamily: "Arial, sans-serif" }}>{text.sub}</div>
        </div>
        <div style={{ display: "flex", gap: 14, fontSize: 24, color: "#2E4A3A", fontFamily: "Arial, sans-serif", fontWeight: 600 }}>
          <span>3MF · STL</span><span>·</span><span>Eco PLA</span><span>·</span><span>monadruk.com</span>
        </div>
      </div>
    ),
    size,
  );
}
