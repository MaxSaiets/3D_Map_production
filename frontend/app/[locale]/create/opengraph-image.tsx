import { ImageResponse } from "next/og";
import { routing } from "@/i18n/routing";

export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

// Власний OG для /create (раніше успадковувався загальний homepage-card, чий текст
// не збігався зі сторінкою-конструктором). Поза next-intl провайдером → копія inline.
type OgStrings = { alt: string; title1: string; title2: string; sub: string };

const OG_TEXT: Record<string, OgStrings> = {
  uk: {
    alt: "Monadruk — конструктор 3D-мап міста",
    title1: "Конструктор",
    title2: "3D-мапи міста.",
    sub: "Обери район, налаштуй модель — завантаж 3MF/STL або замов друк.",
  },
  en: {
    alt: "Monadruk — 3D city map builder",
    title1: "Builder.",
    title2: "3D city maps.",
    sub: "Pick a district, tune the model — download 3MF/STL or order a print.",
  },
  de: {
    alt: "Monadruk — 3D-Stadtkarten-Konfigurator",
    title1: "Konfigurator.",
    title2: "3D-Stadtkarten.",
    sub: "Bezirk wählen, Modell anpassen — 3MF/STL laden oder Druck bestellen.",
  },
  es: {
    alt: "Monadruk — configurador de mapas 3D de ciudad",
    title1: "Configurador.",
    title2: "Mapas 3D de ciudad.",
    sub: "Elige un distrito, ajusta el modelo — descarga 3MF/STL o pide impresión.",
  },
  fr: {
    alt: "Monadruk — configurateur de cartes 3D de ville",
    title1: "Configurateur.",
    title2: "Cartes 3D de ville.",
    sub: "Choisis un quartier, règle le modèle — télécharge 3MF/STL ou commande l'impression.",
  },
  pl: {
    alt: "Monadruk — kreator map 3D miasta",
    title1: "Kreator.",
    title2: "Mapy 3D miasta.",
    sub: "Wybierz dzielnicę, dostrój model — pobierz 3MF/STL lub zamów druk.",
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
