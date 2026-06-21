import { ImageResponse } from "next/og";
import { routing } from "@/i18n/routing";
import { CITY_PAGES, CITY_PAGE_BY_SLUG } from "@/lib/cityPages";
import { cityFacts } from "@/lib/cityFacts";
import type { AppLocale } from "@/i18n/routing";

export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

// Pre-render one OG image per (locale, city) so social shares of a city page
// show that city's name. Runs outside the next-intl React provider (og context),
// so the per-locale copy is kept inline here (cannot use next-intl hooks).
export function generateStaticParams() {
  return CITY_PAGES.map((c) => ({ city: c.slug }));
}

export const dynamicParams = false;

type OgCopy = {
  alt: string;
  kicker: (city: string) => string; // small headline над назвою
  sub: string; // supporting subline
};

const OG_TEXT: Record<string, OgCopy> = {
  uk: {
    alt: "Monadruk — 3D-мапа міста",
    kicker: (c) => `3D-мапа · ${c}`,
    sub: "Тактильна 3D-мапа й брелок твого міста. Готовий 3MF для друку.",
  },
  en: {
    alt: "Monadruk — 3D city map",
    kicker: (c) => `3D map · ${c}`,
    sub: "Tactile 3D map and keychain of your city. Print-ready 3MF.",
  },
  de: {
    alt: "Monadruk — 3D-Stadtkarte",
    kicker: (c) => `3D-Karte · ${c}`,
    sub: "Taktile 3D-Karte und Schlüsselanhänger deiner Stadt. Druckfertige 3MF.",
  },
  es: {
    alt: "Monadruk — mapa 3D de la ciudad",
    kicker: (c) => `Mapa 3D · ${c}`,
    sub: "Mapa 3D táctil y llavero de tu ciudad. 3MF listo para imprimir.",
  },
  fr: {
    alt: "Monadruk — carte 3D de la ville",
    kicker: (c) => `Carte 3D · ${c}`,
    sub: "Carte 3D tactile et porte-clés de ta ville. 3MF prêt à imprimer.",
  },
  pl: {
    alt: "Monadruk — mapa 3D miasta",
    kicker: (c) => `Mapa 3D · ${c}`,
    sub: "Dotykowa mapa 3D i brelok Twojego miasta. Gotowy do druku plik 3MF.",
  },
};

function pickLocale(locale: string): string {
  return (routing.locales as readonly string[]).includes(locale) ? locale : routing.defaultLocale;
}

// `alt` is read statically by Next.js — export an English+brand default.
export const alt = OG_TEXT.en.alt;

export default async function CityOpengraphImage({
  params,
}: {
  params: { locale: string; city: string };
}) {
  const locale = pickLocale(params.locale) as AppLocale;
  const text = OG_TEXT[locale] ?? OG_TEXT.uk;
  const city = CITY_PAGE_BY_SLUG[params.city];
  const cityName = city ? city.names[locale] : "";
  // Візитівка міста (lib/cityFacts): uk-локаль бере .uk, решта — .latin.
  const facts = cityFacts(params.city);
  const landmark = facts ? (locale === "uk" ? facts.landmark.uk : facts.landmark.latin) : "";

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
        <div style={{ display: "flex", flexDirection: "column", gap: 4, maxWidth: 980 }}>
          <div style={{ fontSize: 30, fontWeight: 600, color: "#8E6B3D", fontFamily: "Arial, sans-serif", letterSpacing: 1 }}>
            {text.kicker(cityName)}
          </div>
          <div style={{ fontSize: 88, fontWeight: 600, lineHeight: 1.02, color: "#1B2A22", fontStyle: "italic" }}>
            {cityName}
          </div>
          {landmark ? (
            <div style={{ fontSize: 30, color: "#2E4A3A", marginTop: 10, fontFamily: "Georgia, serif" }}>
              {landmark}
            </div>
          ) : null}
          <div style={{ fontSize: 28, color: "#3c4a42", maxWidth: 900, marginTop: 18, fontFamily: "Arial, sans-serif" }}>
            {text.sub}
          </div>
        </div>
        <div style={{ display: "flex", gap: 14, fontSize: 24, color: "#2E4A3A", fontFamily: "Arial, sans-serif", fontWeight: 600 }}>
          <span>3MF · STL</span><span>·</span><span>Eco PLA</span><span>·</span><span>monadruk.com</span>
        </div>
      </div>
    ),
    size,
  );
}
