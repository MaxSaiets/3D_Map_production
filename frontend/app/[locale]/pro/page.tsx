import type { Metadata } from "next";
import { pageMetadata } from "@/i18n/metadata";
import { ProSubscription } from "@/components/ProSubscription";

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  return pageMetadata({ locale: params.locale, path: "/pro", ns: "proMeta" });
}

export default function ProPage() {
  return <ProSubscription />;
}
