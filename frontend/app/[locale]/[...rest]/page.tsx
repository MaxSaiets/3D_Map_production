import { notFound } from "next/navigation";

// Any unmatched path under a locale (e.g. /en/foo) renders the localized
// app/[locale]/not-found.tsx instead of Next's default 404.
export default function CatchAll() {
  notFound();
}
