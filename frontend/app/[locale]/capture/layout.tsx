import type { Metadata } from "next";

// Внутрішній інструмент рендеру (capture) — НЕ індексувати. Belt-and-suspenders
// до robots.txt Disallow: on-page noindex прибирає його з пошуку навіть за прямим
// лінком. Дочірня сторінка — 'use client' (не може мати metadata), тож гейт тут.
export const metadata: Metadata = {
  robots: { index: false, follow: false, googleBot: { index: false, follow: false } },
};

export default function CaptureLayout({ children }: { children: React.ReactNode }) {
  return children;
}
