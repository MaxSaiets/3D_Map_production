"use client";

import { useEffect } from "react";

/**
 * ROOT error boundary (non-locale). This sits ABOVE the [locale] segment, so it
 * runs OUTSIDE the next-intl provider — it cannot call useTranslations(). There
 * is also no root app/layout.tsx, so we cannot rely on globals.css being present.
 * Everything here is therefore inline-styled and bilingual (English + Ukrainian)
 * so the fallback always looks branded, never raw dev text.
 *
 * Locale-scoped runtime errors are handled by app/[locale]/error.tsx (localized).
 */
export default function RootError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Diagnostics only — never surface raw error.message to users.
    // eslint-disable-next-line no-console
    console.error("Root error boundary:", error);
  }, [error]);

  const paper = "#F4EFE4";
  const ink = "#1B2A22";
  const ink2 = "#3c4a42";
  const forest = "#2E4A3A";

  return (
    <div
      style={{
        minHeight: "100dvh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "64px 20px",
        background: paper,
        color: ink,
        fontFamily:
          "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif",
      }}
    >
      <div style={{ width: "100%", maxWidth: 540, textAlign: "center" }}>
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            fontSize: 12,
            fontWeight: 700,
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: forest,
          }}
        >
          <span
            style={{ width: 8, height: 8, borderRadius: 99, background: forest, display: "inline-block" }}
          />
          monadruk
        </div>

        <h1
          style={{
            margin: "20px 0 0",
            fontSize: "clamp(28px, 5vw, 44px)",
            lineHeight: 1.1,
            fontWeight: 600,
            color: forest,
            fontFamily: "Georgia, 'Times New Roman', serif",
          }}
        >
          Something went wrong
        </h1>
        <p
          style={{
            margin: "6px 0 0",
            fontSize: "clamp(18px, 3.2vw, 24px)",
            lineHeight: 1.2,
            fontWeight: 600,
            color: ink,
            fontStyle: "italic",
            fontFamily: "Georgia, 'Times New Roman', serif",
          }}
        >
          Щось пішло не так
        </p>

        <p style={{ margin: "20px auto 0", maxWidth: 440, fontSize: 15, lineHeight: 1.6, color: ink2 }}>
          We hit an unexpected error. Please try again — your work is safe.
          <br />
          Сталася неочікувана помилка. Спробуйте ще раз — ваші дані в безпеці.
        </p>

        <div
          style={{
            marginTop: 32,
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "center",
            gap: 12,
          }}
        >
          <button
            type="button"
            onClick={reset}
            style={{
              minHeight: 48,
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              borderRadius: 99,
              border: "none",
              cursor: "pointer",
              padding: "12px 24px",
              fontSize: 14,
              fontWeight: 700,
              color: paper,
              background: forest,
              boxShadow: "0 10px 24px rgba(46,74,58,0.28)",
            }}
          >
            Try again · Спробувати ще раз
          </button>
          <a
            href="/"
            style={{
              minHeight: 48,
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              borderRadius: 99,
              border: "1px solid rgba(27,42,34,0.18)",
              padding: "12px 20px",
              fontSize: 14,
              fontWeight: 600,
              color: ink2,
              textDecoration: "none",
            }}
          >
            Home · На головну
          </a>
        </div>

        {error?.digest && (
          <p style={{ marginTop: 32, fontSize: 11, color: "rgba(27,42,34,0.45)", fontFamily: "ui-monospace, monospace" }}>
            {error.digest}
          </p>
        )}
      </div>
    </div>
  );
}
