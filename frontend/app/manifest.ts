import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Monadruk — 3D-мапи твого міста",
    short_name: "Monadruk",
    description:
      "Тактильні 3D-мапи твого міста та брелки з мапою для друку. Завантаж готовий 3MF.",
    start_url: "/",
    display: "standalone",
    background_color: "#F4EFE4",
    theme_color: "#2E4A3A",
    lang: "uk",
    icons: [
      { src: "/icon", sizes: "512x512", type: "image/png" },
    ],
  };
}
