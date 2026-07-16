import type { Metadata } from "next";
import { Link } from "@/i18n/navigation";
import { pageMetadata } from "@/i18n/metadata";
import { BUSINESS } from "@/lib/legal";
import { getLegalSet } from "@/lib/legal/content";
import { LegalArticle } from "@/components/LegalArticle";

export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  return pageMetadata({ locale: params.locale, path: "/contacts", ns: "contactsMeta" });
}

const STORE_JSON_LD = {
  "@context": "https://schema.org",
  "@type": "Store",
  // Посилаємось на ГОЛОВНИЙ Organization/Store вузол з layout (${site}/#org),
  // а не плодимо другий безіменний Store того самого бізнесу (SEO-дедуп).
  "@id": `${BUSINESS.site}/#org`,
  name: BUSINESS.storeName,
  url: BUSINESS.site,
  image: `${BUSINESS.site}/opengraph-image`,
  email: BUSINESS.email,
  telephone: BUSINESS.phone,
  priceRange: "₴₴",
  address: {
    "@type": "PostalAddress",
    streetAddress: "вул. Завадського, 38",
    addressLocality: "Хмельницький",
    addressRegion: "Хмельницька область",
    addressCountry: "UA",
  },
  founder: { "@type": "Person", name: "Саєць Максим Володимирович" },
};

export default function ContactsPage({ params }: { params: { locale: string } }) {
  const doc = getLegalSet(params.locale).contacts;
  return (
    <div id="main-content" tabIndex={-1} className="mx-auto max-w-[760px] px-5 py-12 lg:px-8">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(STORE_JSON_LD) }} />
      <Link href="/" className="text-[13px] font-semibold text-ink-2 hover:text-ink">← monadruk</Link>
      <LegalArticle doc={doc} locale={params.locale} path="/contacts" />
    </div>
  );
}
