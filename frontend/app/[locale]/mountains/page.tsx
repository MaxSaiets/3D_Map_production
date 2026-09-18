"use client";

import dynamic from "next/dynamic";
import { BetaBanner } from "@/components/BetaBanner";

// Студія тягне leaflet і three.js — на сервері їй нічого робити.
const MountainStudio = dynamic(() => import("@/components/MountainStudio"), {
  ssr: false,
  loading: () => <div className="mx-auto mt-10 h-[520px] w-full max-w-[1180px] animate-pulse rounded-[28px] bg-[rgba(255,255,255,0.6)]" />,
});

export default function MountainsPage() {
  return (
    <>
      <BetaBanner mode="mountains" />
      <MountainStudio />
    </>
  );
}
