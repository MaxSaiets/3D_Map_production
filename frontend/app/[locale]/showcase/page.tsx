"use client";
export const dynamic = "force-dynamic";

import { useState } from "react";
import dynamicImport from "next/dynamic";
import { ArrowLeft } from "lucide-react";
import { useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import ModelModal, { type ModalModel } from "@/components/ModelModal";

const Model3DViewer = dynamicImport(() => import("@/components/Model3DViewer"), {
  ssr: false,
  loading: () => (
    <div className="flex h-[420px] w-full items-center justify-center text-sm text-ink-3">3D…</div>
  ),
});

const WEB_KEY = ["/models/keychain-fea.glb", "/models/keychain-home.glb", "/models/keychain-water.glb", "/models/keychain-bridge.glb"];
const WEB_MAP = ["/models/map-dense.glb", "/models/map-district.glb"];

export default function ShowcasePage() {
  const t = useTranslations("showcase");
  const [filter, setFilter] = useState<"all" | "key" | "map">("all");
  const [modal, setModal] = useState<ModalModel | null>(null);

  const VIEW_MODELS = [
    { id: "keychain-fea", label: t("mKey"), url: "/models/keychain-fea.glb", kind: "key" as const },
    { id: "keychain-home", label: t("mHome"), url: "/models/keychain-home.glb", kind: "key" as const },
    { id: "keychain-water", label: t("mWater"), url: "/models/keychain-water.glb", kind: "key" as const },
    { id: "keychain-bridge", label: t("mBridge"), url: "/models/keychain-bridge.glb", kind: "key" as const },
    { id: "map-dense", label: t("mBlock"), url: "/models/map-dense.glb", kind: "map" as const },
    { id: "map-district", label: t("mDistrict"), url: "/models/map-district.glb", kind: "map" as const },
  ];
  const [active, setActive] = useState(VIEW_MODELS[0]);

  type Item = { src: string; kind: "key" | "map" };
  const ITEMS: Item[] = [
    ...Array.from({ length: 8 }, (_, i) => ({ src: `/showcase/keychain-${i + 1}.png`, kind: "key" as const })),
    ...Array.from({ length: 11 }, (_, i) => ({ src: `/showcase/map-${i + 1}.png`, kind: "map" as const })),
  ];
  const items = ITEMS.filter((it) => filter === "all" || it.kind === filter);
  const viewModels = VIEW_MODELS.filter((m) => filter === "all" || m.kind === filter);
  let keyN = 0, mapN = 0;
  const openItem = (it: Item) => {
    if (it.kind === "key") setModal({ url: WEB_KEY[(keyN++) % WEB_KEY.length], label: t("keyItem"), kind: "key", price: t("keyPrice") });
    else setModal({ url: WEB_MAP[(mapN++) % WEB_MAP.length], label: t("mapItem"), kind: "map", price: t("mapPrice") });
  };

  return (
    <div className="mx-auto min-h-[100dvh] max-w-[1280px] px-5 py-8 lg:px-8">
      <Link href="/" className="mb-6 inline-flex min-h-[40px] items-center gap-1.5 py-2 text-[13px] font-semibold text-ink-2 hover:text-ink">
        <ArrowLeft size={15} /> {t("back")}
      </Link>

      <div className="text-center">
        <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-ink-3">{t("eyebrow")}</p>
        <h1 className="mt-2 font-serif text-[clamp(30px,4vw,56px)] text-ink">{t("title")}</h1>
        <p className="mx-auto mt-3 max-w-[640px] text-[15px] text-ink-2">{t("subtitle")}</p>
      </div>

      {/* Filter */}
      <div className="mt-7 flex justify-center gap-2">
        {([["all", t("all")], ["key", t("keys")], ["map", t("maps")]] as const).map(([k, lbl]) => (
          <button
            key={k}
            onClick={() => {
              setFilter(k as "all" | "key" | "map");
              const first = VIEW_MODELS.find((m) => k === "all" || m.kind === k);
              if (first) setActive(first);
            }}
            className={`rounded-full border px-5 py-2 text-sm font-semibold transition ${
              filter === k ? "border-forest bg-forest text-white" : "border-line bg-paper text-ink-2 hover:border-forest/40"
            }`}
          >
            {lbl}
          </button>
        ))}
      </div>

      {/* 3D feature */}
      <div className="mt-8 grid items-center gap-8 lg:grid-cols-[1.1fr_1fr]">
        <div
          className="overflow-hidden rounded-[28px] border border-line bg-gradient-to-b from-[#f4efe3] to-[#e7ddc9] shadow-[0_30px_80px_rgba(15,23,42,0.10)]"
          title={t("rotate3d")}
        >
          <Model3DViewer
            url={active.url}
            height={440}
            onActivate={() => setModal({ url: active.url, label: active.label, kind: active.kind })}
          />
        </div>
        <div>
          <h2 className="font-serif text-2xl text-ink">{active.label}</h2>
          <p className="mt-2 text-[14px] text-ink-2">{t("viewerDesc")}</p>
          <div className="mt-5 flex flex-wrap gap-2">
            {(filter === "all" ? VIEW_MODELS : viewModels).map((m) => (
              <button
                key={m.id}
                onClick={() => setActive(m)}
                className={`rounded-full border px-4 py-2 text-sm font-semibold transition ${
                  active.id === m.id ? "border-forest bg-forest text-white" : "border-line bg-paper text-ink-2 hover:border-forest/40"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
          <div className="mt-6 flex gap-3">
            <Link href="/keychains" className="inline-flex min-h-[48px] items-center justify-center rounded-full bg-forest px-6 text-[15px] font-semibold text-white hover:brightness-110">
              {t("createKeychain")} →
            </Link>
            <Link href="/create" className="inline-flex min-h-[48px] items-center justify-center rounded-full border border-line px-6 text-[15px] font-semibold text-ink-2 hover:border-forest/50">
              {t("makeMap")}
            </Link>
          </div>
        </div>
      </div>

      {/* Gallery */}
      <div className="mt-14 grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
        {items.map((it) => (
          <button key={it.src} onClick={() => openItem(it)} className="group overflow-hidden rounded-[20px] border border-line bg-paper text-left" title={t("rotate3d")}>
            <div className="relative aspect-square overflow-hidden">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={it.src} alt={it.kind === "key" ? t("keyItem") : t("mapItem")} loading="lazy" className="h-full w-full object-cover transition duration-500 group-hover:scale-[1.06]" />
              <span className="pointer-events-none absolute inset-0 flex items-center justify-center bg-ink/0 transition group-hover:bg-ink/25">
                <span className="rounded-full bg-white/90 px-3 py-1 text-[11px] font-bold text-ink opacity-0 transition group-hover:opacity-100">{t("rotate3d")} ↻</span>
              </span>
            </div>
            <div className="flex items-center justify-between gap-2 px-3 py-3">
              <div>
                <div className="text-[13px] font-semibold text-ink">{it.kind === "key" ? t("keyItem") : t("mapItem")}</div>
                <div className="text-[11px] text-ink-3">{it.kind === "key" ? t("keychainSize") : t("district")}</div>
              </div>
              <span className="shrink-0 rounded-full bg-forest px-3 py-1.5 text-[11px] font-bold text-white">{it.kind === "key" ? t("keyPrice") : t("mapPrice")}</span>
            </div>
          </button>
        ))}
      </div>

      <div className="mt-14 rounded-[24px] bg-forest px-6 py-10 text-center text-[#F4EFE4]">
        <h3 className="font-serif text-[clamp(22px,3vw,34px)]">{t("ctaTitle")}</h3>
        <p className="mx-auto mt-2 max-w-[520px] text-[14px] opacity-90">{t("ctaDesc")}</p>
        <Link href="/create" className="mt-5 inline-flex min-h-[48px] items-center justify-center rounded-full bg-white px-7 text-[15px] font-semibold text-forest hover:brightness-95">
          {t("ctaButton")} →
        </Link>
      </div>

      <ModelModal model={modal} onClose={() => setModal(null)} />
    </div>
  );
}
