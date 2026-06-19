import type { Metadata } from "next";

// Адмін-панель — НЕ індексувати (до robots.txt Disallow). noindex,nofollow.
export const metadata: Metadata = {
  robots: { index: false, follow: false, googleBot: { index: false, follow: false } },
};

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  return children;
}
