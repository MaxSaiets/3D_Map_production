import type { Metadata } from "next";
import { Link } from "@/i18n/navigation";
import { pageMetadata } from "@/i18n/metadata";
import { getSubscriptionTerms } from "@/lib/legal/subscription";
import { LegalArticle } from "@/components/LegalArticle";

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  return pageMetadata({ locale: params.locale, path: "/pro-terms", ns: "proTermsMeta" });
}

export default function ProTermsPage({ params }: { params: { locale: string } }) {
  const doc = getSubscriptionTerms(params.locale);
  return (
    <div className="mx-auto max-w-[760px] px-5 py-12 lg:px-8">
      <Link href="/pro" className="text-[13px] font-semibold text-ink-2 hover:text-ink">← Monadruk Pro</Link>
      <LegalArticle doc={doc} locale={params.locale} path="/pro-terms" />
    </div>
  );
}
