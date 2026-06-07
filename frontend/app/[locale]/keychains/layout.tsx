import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Брелок з картою міста — 3D-брелок на замовлення",
  description:
    "Брелок-мапа твого міста: персональний 3D-брелок 55×30 мм із вулицями, будинками та написом. Створи онлайн і завантаж 3MF для друку або замов виготовлення з Eco PLA.",
  keywords: [
    "брелок з картою міста", "брелок мапа", "3d брелок місто", "брелок на замовлення",
    "персональний брелок", "брелок з мапою києва", "брелок 3д друк", "сувенір брелок місто",
  ],
  alternates: { canonical: "/keychains" },
  openGraph: {
    title: "Брелок з картою міста — Monadruk",
    description: "Персональний 3D-брелок із мапою твого міста. Створи онлайн і замов друк.",
    url: "https://monadruk.com/keychains",
    type: "website",
    locale: "uk_UA",
  },
};

const ld = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "Product",
      name: "Брелок з картою міста",
      description:
        "Персональний 3D-брелок-мапа 55×30 мм із вулицями, будинками та власним написом. Друк з Eco PLA або готовий 3MF для самостійного друку.",
      image: "https://monadruk.com/showcase/keychain-5.png",
      brand: { "@type": "Brand", name: "Monadruk" },
      category: "Сувеніри / Брелки",
      offers: {
        "@type": "Offer",
        priceCurrency: "UAH",
        price: "290",
        availability: "https://schema.org/InStock",
        url: "https://monadruk.com/keychains",
      },
    },
    {
      "@type": "BreadcrumbList",
      itemListElement: [
        { "@type": "ListItem", position: 1, name: "Головна", item: "https://monadruk.com" },
        { "@type": "ListItem", position: 2, name: "Брелки", item: "https://monadruk.com/keychains" },
      ],
    },
  ],
};

export default function KeychainsLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(ld) }} />
      {children}
    </>
  );
}
