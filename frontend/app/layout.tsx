import type { Metadata } from "next";
import { Cormorant_Garamond, Manrope, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { ContactWidget } from "@/components/ContactWidget";
import { AuthProvider } from "@/components/AuthProvider";

// Self-hosted via next/font: no render-blocking external stylesheet, no CLS.
const serif = Cormorant_Garamond({
  subsets: ["latin"], weight: ["400", "500", "600"], style: ["normal", "italic"],
  variable: "--font-serif", display: "swap",
});
const sans = Manrope({
  subsets: ["latin", "cyrillic"], weight: ["300", "400", "500", "600", "700"],
  variable: "--font-sans", display: "swap",
});
const mono = JetBrains_Mono({
  subsets: ["latin"], weight: ["400", "500"], variable: "--font-mono", display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "Monadruk — 3D-мапи твого міста для друку",
    template: "%s · Monadruk",
  },
  description:
    "Обери район свого міста, налаштуй модель і завантаж готовий 3D-файл для друку. Тактильні архітектурні мапи з висотами будинків, парків і річок. Брелки з мапою на замовлення.",
  metadataBase: new URL("https://monadruk.com"),
  applicationName: "Monadruk",
  keywords: [
    "3D мапа", "3D друк", "мапа міста", "3MF", "брелок з мапою", "Київ 3D",
    "тактильна мапа", "сувенір", "Bambu Studio", "PrusaSlicer", "Україна",
  ],
  authors: [{ name: "Monadruk" }],
  creator: "Monadruk",
  alternates: { canonical: "/" },
  verification: process.env.NEXT_PUBLIC_GOOGLE_SITE_VERIFICATION
    ? { google: process.env.NEXT_PUBLIC_GOOGLE_SITE_VERIFICATION }
    : undefined,
  robots: {
    index: true,
    follow: true,
    googleBot: { index: true, follow: true, "max-image-preview": "large" },
  },
  openGraph: {
    title: "Monadruk — 3D-мапи твого міста",
    description: "Перетвори будь-яке місце на Землі у тактильну 3D-мапу для друку.",
    url: "https://monadruk.com",
    siteName: "Monadruk",
    type: "website",
    locale: "uk_UA",
  },
  twitter: {
    card: "summary_large_image",
    title: "Monadruk — 3D-мапи твого міста",
    description: "Перетвори будь-яке місце на Землі у тактильну 3D-мапу для друку.",
  },
};

export const viewport = {
  themeColor: "#2E4A3A",
};

const jsonLd = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "Organization",
      "@id": "https://monadruk.com/#org",
      name: "Monadruk",
      url: "https://monadruk.com",
      logo: "https://monadruk.com/icon",
      image: "https://monadruk.com/opengraph-image",
      description: "Тактильні 3D-мапи та брелки з мапою твого міста для друку.",
      areaServed: "UA",
      sameAs: ["https://t.me/monadruk"],
    },
    {
      "@type": "WebSite",
      "@id": "https://monadruk.com/#website",
      url: "https://monadruk.com",
      name: "Monadruk",
      inLanguage: "uk-UA",
      publisher: { "@id": "https://monadruk.com/#org" },
    },
    {
      "@type": "Service",
      "@id": "https://monadruk.com/#service",
      name: "3D-мапи та брелки міст на замовлення",
      serviceType: "Виготовлення 3D-мап і брелків міст для 3D-друку",
      provider: { "@id": "https://monadruk.com/#org" },
      areaServed: { "@type": "Country", name: "Україна" },
      description:
        "Створення 3D-мапи будь-якого міста світу: завантаження готового 3MF/STL або друк на замовлення з Eco PLA. Персональні брелки-мапи.",
      offers: [
        {
          "@type": "Offer",
          name: "Брелок з картою міста",
          priceCurrency: "UAH",
          price: "290",
          url: "https://monadruk.com/keychains",
          availability: "https://schema.org/InStock",
        },
        {
          "@type": "Offer",
          name: "3D-мапа району міста",
          priceCurrency: "UAH",
          price: "690",
          url: "https://monadruk.com/create",
          availability: "https://schema.org/InStock",
        },
      ],
    },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="uk" className={`${serif.variable} ${sans.variable} ${mono.variable}`}>
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      </head>
      <body className="antialiased">
        <AuthProvider>
          {children}
          <ContactWidget />
        </AuthProvider>
      </body>
    </html>
  );
}
