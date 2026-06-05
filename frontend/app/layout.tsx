import type { Metadata } from "next";
import "./globals.css";
import { ContactWidget } from "@/components/ContactWidget";

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
      description: "Тактильні 3D-мапи та брелки з мапою твого міста для друку.",
      areaServed: "UA",
    },
    {
      "@type": "WebSite",
      "@id": "https://monadruk.com/#website",
      url: "https://monadruk.com",
      name: "Monadruk",
      inLanguage: "uk-UA",
      publisher: { "@id": "https://monadruk.com/#org" },
    },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="uk">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500&family=Manrope:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      </head>
      <body className="antialiased">
        {children}
        <ContactWidget />
      </body>
    </html>
  );
}
