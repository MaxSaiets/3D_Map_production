import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Конструктор 3D-мапи міста — створити онлайн",
  description:
    "Створи 3D-мапу будь-якого міста світу онлайн: обери район, налаштуй висоти будинків, парки й річки та завантаж готовий 3MF/STL для друку вдома або замов друк.",
  keywords: [
    "створити 3д мапу міста", "конструктор 3д мапи", "3д мапа онлайн",
    "завантажити 3MF мапу міста", "STL мапа міста", "3d друк мапи міста",
    "тривимірна мапа міста", "мапа міста на замовлення",
  ],
  alternates: { canonical: "/create" },
  openGraph: {
    title: "Конструктор 3D-мапи міста — Monadruk",
    description: "Обери район будь-якого міста і завантаж готовий 3D-файл для друку.",
    url: "https://monadruk.com/create",
    type: "website",
    locale: "uk_UA",
  },
};

export default function CreateLayout({ children }: { children: React.ReactNode }) {
  return children;
}
