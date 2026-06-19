import type { Metadata } from "next";

// Приватна сторінка (кабінет) — НЕ індексувати (належ-і-suspenders до robots.txt
// Disallow). noindex,nofollow прибирає її з пошуку навіть якщо хтось дасть лінк.
export const metadata: Metadata = {
  robots: { index: false, follow: false, googleBot: { index: false, follow: false } },
};

export default function AccountLayout({ children }: { children: React.ReactNode }) {
  return children;
}
