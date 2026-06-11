import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { BASE, localeUrl } from "@/i18n/metadata";
import { routing, localeMeta, defaultLocale, type AppLocale } from "@/i18n/routing";
import { Link } from "@/i18n/navigation";

/**
 * E4: публічна share-сторінка згенерованої моделі. og:image = реальний рендер
 * користувача (/api/og/{taskId}). noindex — це особисті шери, не SEO-контент.
 */
const ID_RE = /^[A-Za-z0-9_-]{8,64}$/;

export async function generateMetadata({
  params,
}: {
  params: { locale: string; taskId: string };
}): Promise<Metadata> {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  const t = await getTranslations({ locale, namespace: "share" });
  const ok = ID_RE.test(params.taskId);
  const title = t("title");
  const description = t("description");
  const ogImage = ok ? `${BASE}/api/og/${params.taskId}` : `${BASE}/opengraph-image`;
  return {
    title,
    description,
    robots: { index: false, follow: true },
    openGraph: {
      title,
      description,
      url: localeUrl(locale, `/share/${params.taskId}`),
      siteName: "Monadruk",
      type: "website",
      locale: localeMeta[locale].ogLocale,
      images: [{ url: ogImage, width: 1200, height: 630 }],
    },
    twitter: { card: "summary_large_image", title, description, images: [ogImage] },
  };
}

export default async function SharePage({
  params,
}: {
  params: { locale: string; taskId: string };
}) {
  const locale = ((routing.locales as readonly string[]).includes(params.locale)
    ? params.locale
    : defaultLocale) as AppLocale;
  setRequestLocale(locale);
  const t = await getTranslations({ locale, namespace: "share" });
  const ok = ID_RE.test(params.taskId);

  return (
    <main className="mx-auto max-w-[720px] px-5 py-14 text-center lg:py-20">
      <div className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--text-secondary)]">Monadruk</div>
      <h1 className="mt-3 text-[clamp(26px,4vw,40px)] leading-tight">{t("title")}</h1>
      <p className="mx-auto mt-4 max-w-[480px] text-[15px] leading-relaxed text-ink-2">{t("description")}</p>
      {ok && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={`/api/og/${params.taskId}`}
          alt={t("imageAlt")}
          className="mx-auto mt-8 w-full max-w-[560px] rounded-[24px] border border-line-soft bg-white/70 shadow-[0_18px_50px_rgba(46,74,58,0.12)]"
        />
      )}
      <div className="mt-9 flex flex-wrap justify-center gap-3">
        <Link
          href="/create"
          className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] bg-[var(--accent-strong)] px-6 py-3 text-sm font-semibold text-white transition hover:opacity-90"
        >
          {t("ctaMap")}
        </Link>
        <Link
          href="/keychains"
          className="inline-flex min-h-[48px] items-center justify-center rounded-[22px] border border-line-soft bg-white/80 px-6 py-3 text-sm font-semibold text-ink transition hover:border-[var(--accent)]"
        >
          {t("ctaKeychain")}
        </Link>
      </div>
    </main>
  );
}
