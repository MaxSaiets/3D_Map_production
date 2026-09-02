import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { ArrowRight, KeyRound, Map, Tag } from "lucide-react";
import { Link } from "@/i18n/navigation";
import { localeUrl } from "@/i18n/metadata";

/**
 * /start — посадкова для link-in-bio (TikTok / Instagram / Pinterest).
 * Власник: «дуже мало людей взагалі знаходять і заходять». Соцмережі — єдине
 * живе джерело (TikTok ~3.5K переглядів/тиждень), а лінк у біо вів на важку
 * головну. Ця сторінка: миттєва, мобільна, ТРИ дії й реальні фото. UTM-мітки з
 * посилання ловить lib/analytics.ts автоматично → у /admin видно, звідки прийшли.
 * noindex: це не контент для пошуку, а вхід із соцмереж (без дублю головної).
 */
export async function generateMetadata({ params }: { params: { locale: string } }): Promise<Metadata> {
  const t = await getTranslations({ locale: params.locale, namespace: "start" });
  const url = localeUrl(params.locale as never, "/start");
  return {
    title: t("title"),
    description: t("line"),
    robots: { index: false, follow: true },
    alternates: { canonical: url },
    openGraph: { title: t("title"), description: t("line"), url, siteName: "Monadruk", type: "website" },
  };
}

const PHOTOS = [1, 2, 3, 4, 5, 6] as const;

export default async function StartPage({ params }: { params: { locale: string } }) {
  setRequestLocale(params.locale);
  const t = await getTranslations({ locale: params.locale, namespace: "start" });
  const btn = "flex min-h-[56px] w-full items-center justify-between gap-3 rounded-full px-6 text-[16px] font-bold transition active:scale-[0.99]";
  return (
    <main className="mx-auto flex min-h-[100dvh] max-w-[520px] flex-col px-5 pb-10 pt-8">
      <div className="flex items-center gap-2 text-[15px] font-semibold tracking-tight text-ink">
        <span aria-hidden className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-forest text-[#F4EFE4]">◈</span>
        monadruk
      </div>

      <h1 className="mt-6 font-serif text-[clamp(30px,8vw,40px)] leading-[1.05] text-ink">{t("title")}</h1>
      <p className="mt-3 text-[15px] leading-relaxed text-ink-2">{t("line")}</p>
      <p className="mt-2 inline-flex items-center gap-1.5 text-[13px] font-semibold text-forest">
        <Tag size={14} /> {t("price")}
      </p>

      <div className="mt-6 grid gap-3" data-testid="start-actions">
        <Link href="/create" className={`${btn} bg-forest text-[#F4EFE4] shadow-lift`}>
          <span className="inline-flex items-center gap-2.5"><Map size={18} /> {t("cta1")}</span>
          <ArrowRight size={18} />
        </Link>
        <Link href="/keychains" className={`${btn} border border-line bg-paper text-ink`}>
          <span className="inline-flex items-center gap-2.5"><KeyRound size={18} /> {t("cta2")}</span>
          <ArrowRight size={18} />
        </Link>
        <Link href="/prices" className={`${btn} border border-line-soft bg-transparent text-ink-2`}>
          <span>{t("cta3")}</span>
          <ArrowRight size={18} />
        </Link>
      </div>

      <div className="mt-8">
        <div className="eyebrow mb-3">{t("photos")}</div>
        <div className="grid grid-cols-3 gap-2">
          {PHOTOS.map((n) => (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              key={n}
              src={`/prints/print-${n}.webp`}
              alt=""
              width={320}
              height={320}
              loading={n <= 3 ? "eager" : "lazy"}
              decoding="async"
              className="aspect-square w-full rounded-[14px] border border-line-soft object-cover"
            />
          ))}
        </div>
      </div>
    </main>
  );
}
