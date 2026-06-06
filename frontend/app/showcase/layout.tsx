import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Галерея 3D-мап і брелків міст — приклади друку",
  description:
    "Галерея надрукованих 3D-мап та брелків міст України і світу. Покрути моделі у 3D, обери район і замов друк своєї тактильної мапи або брелка-мапи.",
  keywords: [
    "галерея 3д мап міст", "приклади 3d друку мапи", "3д мапа міста фото",
    "надруковані мапи міст", "3д модель міста", "брелок мапа приклади",
  ],
  alternates: { canonical: "/showcase" },
  openGraph: {
    title: "Галерея 3D-мап і брелків — Monadruk",
    description: "Реальні 3D-моделі міст. Покрути у 3D, обери — і замов друк.",
    url: "https://monadruk.com/showcase",
    type: "website",
    locale: "uk_UA",
  },
};

export default function ShowcaseLayout({ children }: { children: React.ReactNode }) {
  return children;
}
