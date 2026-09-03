"use client";

import { useId, useState } from "react";
import { ArrowRight, Box, User, Menu, X, Globe, KeyRound } from "lucide-react";
import { useTranslations, useLocale } from "next-intl";
import { Link, usePathname, useRouter } from "@/i18n/navigation";
import { locales, localeMeta } from "@/i18n/routing";
import { useAuth } from "@/components/AuthProvider";

// Локалізований лейбл сторінки «Ціни» (тримаємо тут, щоб не чіпати 6 messages-файлів).
const PRICES_LABEL: Record<string, string> = { uk: "Ціни", en: "Prices", de: "Preise", es: "Precios", fr: "Tarifs", pl: "Cennik" };

/* ---------- Language switcher ---------- */
export function LanguageSwitcher({ compact }: { compact?: boolean }) {
  const t = useTranslations("nav");
  const locale = useLocale();
  const pathname = usePathname();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  // useId → унікальний id меню на КОЖЕН інстанс (LanguageSwitcher монтується двічі:
  // десктоп + мобільне меню) → без дублю id / неоднозначного aria-controls.
  const menuId = useId();
  return (
    <div className="relative">
      <button
        type="button"
        aria-label={t("language")}
        aria-expanded={open}
        aria-controls={menuId}
        onClick={() => setOpen((v) => !v)}
        className={`inline-flex min-h-[44px] items-center gap-1.5 rounded-full border border-line text-sm font-semibold text-ink-2 transition hover:border-forest/40 hover:text-ink ${compact ? "px-2.5 py-2" : "px-3 py-2"}`}
      >
        <Globe size={15} />
        <span className="uppercase">{locale}</span>
      </button>
      {open && (
        <>
          {/* Скрим-закривач: НЕ-фокусований div (раніше був aria-hidden <button> —
              порушення «hidden але фокусоване»; меню й так закриється кнопкою-тоглом
              та вибором локалі). */}
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} aria-hidden="true" />
          <ul id={menuId} className="absolute right-0 z-50 mt-2 w-44 overflow-hidden rounded-2xl border border-line bg-paper p-1 shadow-lift">
            {locales.map((l) => (
              <li key={l}>
                <button
                  type="button"
                  onClick={() => { setOpen(false); router.replace(pathname, { locale: l }); }}
                  className={`flex w-full items-center justify-between rounded-xl px-3 py-2.5 text-left text-sm transition hover:bg-bg-2 ${l === locale ? "font-bold text-forest" : "text-ink-2"}`}
                >
                  {localeMeta[l].label}
                  <span className="text-[11px] uppercase text-ink-3">{l}</span>
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

/* ---------- Header ---------- */
type BuilderProps = {
  /** A-1 (2026-09-03): «builder» = один ряд ≤ 56 px для /create і /keychains:
   *  лого · назва сторінки · мова · кабінет · лінк на інший продукт. Без заливних
   *  CTA, що конкурують із флоу, без бургера. */
  variant?: "default" | "builder";
  /** Назва сторінки для builder-варіанту (рендериться як h1 — SEO лишається). */
  title?: string;
  /** Лінк на «інший продукт» (з /create → брелок, з /keychains → мапа). */
  other?: { href: string; label: string; icon?: "keychain" | "map" };
};

export function SiteHeader({ variant = "default", title, other }: BuilderProps = {}) {
  const { user, configured } = useAuth();
  const t = useTranslations("nav");
  const locale = useLocale();
  const pricesLabel = PRICES_LABEL[locale] ?? PRICES_LABEL.uk;
  const [open, setOpen] = useState(false);

  if (variant === "builder") {
    return (
      <header className="sticky top-0 z-50 border-b border-line-soft bg-[rgba(244,239,228,0.9)] backdrop-blur" data-testid="site-header-builder">
        <div className="mx-auto flex max-w-[1760px] items-center gap-2 px-3 py-1.5 sm:px-4 lg:px-6">
          <Link href="/" className="flex min-h-[44px] shrink-0 items-center gap-1.5 font-serif text-lg font-semibold tracking-tight text-ink" aria-label="monadruk">
            <Box size={20} className="text-forest" />
            <span className="hidden sm:inline">monadruk</span>
          </Link>
          <span className="hidden h-5 w-px bg-line sm:block" />
          {title && (
            <h1 className="min-w-0 flex-1 truncate font-title text-[15px] font-semibold text-ink sm:text-base">{title}</h1>
          )}
          {!title && <span className="flex-1" />}
          <div className="flex shrink-0 items-center gap-1.5">
            <LanguageSwitcher compact />
            <Link
              href="/account"
              title={configured && user ? (user.email || user.phoneNumber || t("account")) : t("login")}
              className="inline-flex min-h-[44px] items-center gap-1.5 rounded-full border border-line px-2.5 py-2 text-sm font-semibold text-ink-2 transition hover:border-forest/40 hover:text-ink sm:px-3.5"
            >
              <User size={15} />
              <span className="hidden md:inline">{configured && user ? t("account") : t("login")}</span>
            </Link>
            {other && (
              <Link
                href={other.href}
                className="inline-flex min-h-[44px] items-center gap-1.5 rounded-full px-2.5 py-2 text-sm font-semibold text-forest transition hover:bg-[rgba(46,74,58,0.08)] sm:px-3"
              >
                {other.icon === "keychain" ? <KeyRound size={15} /> : null}
                <span>{other.label}</span>
                <ArrowRight size={14} />
              </Link>
            )}
          </div>
        </div>
      </header>
    );
  }

  return (
    <header className="sticky top-0 z-50 border-b border-line-soft bg-[rgba(244,239,228,0.85)] backdrop-blur">
      <div className="mx-auto flex max-w-[1360px] items-center justify-between px-5 py-4 lg:px-8">
        <Link href="/" className="flex min-w-0 shrink items-center gap-2 truncate font-serif text-xl font-semibold tracking-tight text-ink">
          <Box size={22} className="text-forest" />
          monadruk
        </Link>
        {/* Спрощено: лише чіткі ПУНКТИ ПРИЗНАЧЕННЯ (без home-якорів #how/#templates,
            що захаращували глобальне меню). Галерея · Ціни · Брелоки. */}
        <nav className="hidden items-center gap-8 text-sm text-ink-2 md:flex">
          <Link href="/showcase" className="hover:text-ink">{t("gallery")}</Link>
          <Link href="/prices" className="hover:text-ink">{pricesLabel}</Link>
          <Link href="/keychains" className="hover:text-ink">{t("keychains")}</Link>
          <Link href="/worlds" className="hover:text-ink">{t("worlds")}</Link>
          <Link href="/maket" className="hover:text-ink">{t("maket")}</Link>
        </nav>
        <div className="flex shrink-0 items-center gap-1.5 sm:gap-2.5">
          <div className="hidden sm:block"><LanguageSwitcher /></div>
          <Link
            href="/account"
            className="hidden sm:inline-flex min-h-[44px] items-center gap-1.5 rounded-full border border-line px-3.5 py-2 text-sm font-semibold text-ink-2 transition hover:border-forest/40 hover:text-ink"
            title={configured && user ? (user.email || user.phoneNumber || t("account")) : t("login")}
          >
            <User size={15} />
            <span className="hidden sm:inline">{configured && user ? t("account") : t("login")}</span>
          </Link>
          {/* ДВІ продуктові дії поруч — обидві ВИДНО і на мобільному (раніше
              «Брелок» був похований у бургер-меню, тож з телефона досяжні були
              лише «Карти»). «Брелок» = вторинна обведена пігулка (forest-tint),
              «Карти» = головна заливна CTA. Обидві ≥44px, компактні на 375px
              (px-3, короткі лейбли nav.keychain / nav.mapShort). */}
          <Link
            href="/keychains"
            className="inline-flex min-h-[44px] items-center gap-1.5 rounded-full border border-forest/35 bg-[rgba(46,74,58,0.08)] px-3 py-2 text-sm font-semibold text-forest transition hover:bg-[rgba(46,74,58,0.14)] sm:px-3.5"
          >
            <KeyRound size={15} />
            <span>{t("keychain")}</span>
          </Link>
          <Link
            href="/create"
            className="inline-flex min-h-[44px] items-center gap-1.5 rounded-full bg-forest px-3 py-2.5 text-sm font-bold text-[#F4EFE4] shadow-[0_10px_24px_rgba(46,74,58,0.28)] transition hover:opacity-90 sm:px-5"
            style={{ background: "var(--forest, #2E4A3A)" }}
          >
            <span className="sm:hidden">{t("mapShort")}</span>
            <span className="hidden sm:inline">{t("createMap")}</span>
            <ArrowRight size={15} />
          </Link>
          {/* Mobile menu toggle */}
          <button
            type="button"
            aria-label={open ? t("closeMenu") : t("openMenu")}
            aria-expanded={open}
            aria-controls="mobile-nav"
            onClick={() => setOpen((v) => !v)}
            className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-line text-ink-2 transition hover:border-forest/40 hover:text-ink md:hidden"
          >
            {open ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </div>

      {/* Mobile dropdown nav */}
      {open && (
        <nav id="mobile-nav" className="border-t border-line-soft bg-[rgba(244,239,228,0.98)] px-5 py-3 backdrop-blur md:hidden">
          <ul className="flex flex-col">
            {[
              { href: "/showcase", label: t("gallery") },
              { href: "/prices", label: pricesLabel },
              { href: "/keychains", label: t("keychains") },
              { href: "/worlds", label: t("worlds") },
              { href: "/maket", label: t("maket") },
              { href: "/account", label: configured && user ? t("account") : t("login") },
            ].map((it) => (
              <li key={it.href}>
                <Link
                  href={it.href}
                  onClick={() => setOpen(false)}
                  className="flex min-h-[48px] items-center border-b border-line-soft/60 text-[15px] font-semibold text-ink-2 transition hover:text-ink"
                >
                  {it.label}
                </Link>
              </li>
            ))}
          </ul>
          <div className="mt-3">
            <Link href="/create" onClick={() => setOpen(false)} className="inline-flex min-h-[48px] w-full items-center justify-center gap-1.5 rounded-full bg-forest text-sm font-bold text-[#F4EFE4]" style={{ background: "var(--forest, #2E4A3A)" }}>
              {t("createMap")} <ArrowRight size={15} />
            </Link>
          </div>
          <div className="mt-3 border-t border-line-soft/60 pt-3"><LanguageSwitcher /></div>
        </nav>
      )}
    </header>
  );
}

/* Маршрути, що рендерять власну шапку: головна (SiteHeader сама), білдери
   (/create, /keychains — builder-варіант усередині сторінки), /start (link-in-bio,
   без chrome) і службовий /capture. Решта ~28 сторінок раніше НЕ мали шапки
   взагалі (лише 20-лінковий футер) і перемикача мови — A-1 (2026-09-03). */
const NO_GLOBAL_HEADER = new Set<string>(["/", "/create", "/keychains", "/start"]);
function isCapturePath(pathname: string): boolean {
  return pathname === "/capture" || pathname.startsWith("/capture/");
}

/** Монтується глобально в locale-layout над children. */
export function GlobalHeader() {
  const pathname = usePathname();
  if (NO_GLOBAL_HEADER.has(pathname) || isCapturePath(pathname)) return null;
  return <SiteHeader />;
}

export default SiteHeader;
