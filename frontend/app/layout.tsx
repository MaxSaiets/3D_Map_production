import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Monadruk — 3D-мапи твого міста для друку",
  description:
    "Обери район свого міста, налаштуй модель і завантаж готовий 3D-файл для друку. Тактильні архітектурні мапи з висотами будинків, парків і річок. Брелки з мапою на замовлення.",
  metadataBase: new URL("https://monadruk.com"),
  openGraph: {
    title: "Monadruk — 3D-мапи твого міста",
    description: "Перетвори будь-яке місце на Землі у тактильну 3D-мапу для друку.",
    type: "website",
    locale: "uk_UA",
  },
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
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}
