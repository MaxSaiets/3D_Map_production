import { ImageResponse } from "next/og";
import { routing } from "@/i18n/routing";

export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

// Власний OG для /keychains. Ціна локале-залежна (UAH для uk, EUR для решти) —
// як у keychains/layout.tsx offer. Поза next-intl провайдером → копія inline.
type OgStrings = { alt: string; title1: string; title2: string; sub: string };

const OG_TEXT: Record<string, OgStrings> = {
  uk: {
    alt: "Monadruk — брелок-мапа твого міста",
    title1: "Брелок-мапа",
    title2: "твого міста.",
    sub: "Улюблений район на брелку. Від 120 ₴ — друкуємо й доставляємо.",
  },
  en: {
    alt: "Monadruk — map keychain of your city",
    title1: "Map keychain",
    title2: "of your city.",
    sub: "Your favourite district as a keychain. From €3 — we print & ship.",
  },
  de: {
    alt: "Monadruk — Karten-Schlüsselanhänger deiner Stadt",
    title1: "Karten-Anhänger",
    title2: "deiner Stadt.",
    sub: "Dein Lieblingsviertel als Schlüsselanhänger. Ab €3 — Druck & Versand.",
  },
  es: {
    alt: "Monadruk — llavero-mapa de tu ciudad",
    title1: "Llavero-mapa",
    title2: "de tu ciudad.",
    sub: "Tu barrio favorito como llavero. Desde €3 — imprimimos y enviamos.",
  },
  fr: {
    alt: "Monadruk — porte-clés carte de ta ville",
    title1: "Porte-clés carte",
    title2: "de ta ville.",
    sub: "Ton quartier préféré en porte-clés. Dès €3 — impression et livraison.",
  },
  pl: {
    alt: "Monadruk — brelok-mapa Twojego miasta",
    title1: "Brelok-mapa",
    title2: "Twojego miasta.",
    sub: "Twoja ulubiona dzielnica jako brelok. Od €3 — drukujemy i wysyłamy.",
  },
};

function pickLocale(locale: string): string {
  return (routing.locales as readonly string[]).includes(locale) ? locale : routing.defaultLocale;
}

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
