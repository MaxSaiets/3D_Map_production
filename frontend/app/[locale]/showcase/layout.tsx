import type { Metadata } from "next";
import { pageMetadata } from "@/i18n/metadata";

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  return pageMetadata({ locale: params.locale, path: "/showcase", ns: "showcaseMeta" });
}

export default function ShowcaseLayout({ children }: { children: React.ReactNode }) {
  return children;
}
