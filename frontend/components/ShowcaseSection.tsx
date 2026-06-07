"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { useTranslations } from "next-intl";
import { Link } from "@/i18n/navigation";
import ModelModal, { type ModalModel } from "@/components/ModelModal";

const Model3DViewer = dynamic(() => import("@/components/Model3DViewer"), {
  ssr: false,
  loading: () => (
    <div className="flex h-[420px] w-full items-center justify-center text-sm text-ink-3">3D…</div>
  ),
});

const MODELS_BASE = [
  { id: "keychain-fea", labelKey: "mKey", url: "/models/keychain-fea.glb", kind: "key" as const },
  { id: "keychain-home", labelKey: "mHome", url: "/models/keychain-home.glb", kind: "key" as const },
  { id: "keychain-water", labelKey: "mWater", url: "/models/keychain-water.glb", kind: "key" as const },
  { id: "keychain-bridge", labelKey: "mBridge", url: "/models/keychain-bridge.glb", kind: "key" as const },
  { id: "map-dense", labelKey: "mBlock", url: "/models/map-dense.glb", kind: "map" as const },
  { id: "map-district", labelKey: "mDistrict", url: "/models/map-district.glb", kind: "map" as const },
];

const WEB_KEY = ["/models/keychain-fea.glb", "/models/keychain-home.glb", "/models/keychain-water.glb", "/models/keychain-bridge.glb"];
const WEB_MAP = ["/models/map-dense.glb", "/models/map-district.glb"];

const KEYCHAINS = Array.from({ length: 8 }, (_, i) => `/showcase/keychain-${i + 1}.png`);
const MAPS = Array.from({ length: 11 }, (_, i) => `/showcase/map-${i + 1}.png`);

type Tile = { src: string; kind: "key" | "map"; idx: number };
const TILES: Tile[] = [
  ...KEYCHAINS.map((src, idx) => ({ src, kind: "key" as const, idx })),
  ...MAPS.map((src, idx) => ({ src, kind: "map" as const, idx })),
];
const ROW_A = TILES.filter((_, i) => i % 2 === 0);
const ROW_B = TILES.filter((_, i) => i % 2 === 1);

function Row({ tiles, dir, onOpen, rotateLabel, keyAlt, mapAlt }: {
  tiles: Tile[]; dir: "left" | "right"; onOpen: (t: Tile) => void;
  rotateLabel: string; keyAlt: string; mapAlt: string;
}) {
  const doubled = [...tiles, ...tiles];
  return (
    <div className={`marquee marquee-${dir}`}>
      <div className="marquee-track">
        {doubled.map((t, i) => (
          <button
            key={`${t.kind}-${t.idx}-${i}`}
            onClick={() => onOpen(t)}
            className="group relative h-[180px] w-[230px] shrink-0 overflow-hidden rounded-[18px] border border-line"
            title={rotateLabel}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={t.src} alt={t.kind === "key" ? keyAlt : mapAlt} loading="lazy"
                 className="h-full w-full object-cover transition duration-500 group-hover:scale-[1.07]" />
            <span className="pointer-events-none absolute inset-0 flex items-center justify-center bg-ink/0 transition group-hover:bg-ink/25">
              <span className="rounded-full bg-white/90 px-3 py-1 text-[11px] font-bold text-ink opacity-0 transition group-hover:opacity-100">
                {rotateLabel} ↻
              </span>
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

export default function ShowcaseSection() {
  const t = useTranslations("showcase");
  const models = MODELS_BASE.map((m) => ({ ...m, label: t(m.labelKey) }));
  const [activeId, setActiveId] = useState(models[0].id);
  const active = models.find((m) => m.id === activeId) || models[0];
  const [modal, setModal] = useState<ModalModel | null>(null);

  const tileToModel = (tile: Tile): ModalModel =>
    tile.kind === "key"
      ? { url: WEB_KEY[tile.idx % WEB_KEY.length], label: t("keyItem"), kind: "key", price: t("keyPrice") }
      : { url: WEB_MAP[tile.idx % WEB_MAP.length], label: t("mapItem"), kind: "map", price: t("mapPrice") };

  return (
    <section id="showcase" className="mx-auto max-w-[1360px] px-5 py-20 lg:px-8 lg:py-24">
      <div className="mb-10 text-center">
        <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-ink-3">{t("hEyebrow")}</p>
        <h2 className="mt-3 text-[clamp(30px,3.6vw,54px)] leading-tight">{t("hTitle")}</h2>
        <p className="mx-auto mt-3 max-w-[620px] text-[15px] text-ink-2">{t("hSubtitle")}</p>
      </div>

      {/* Interactive 3D */}
      <div className="grid items-center gap-8 lg:grid-cols-[1.15fr_1fr]">
        <div
          className="overflow-hidden rounded-[28px] border border-line bg-gradient-to-b from-[#f6f1e6] to-[#ece4d3] shadow-[0_30px_80px_rgba(15,23,42,0.10)]"
          title={t("rotate3d")}
        >
          <Model3DViewer
            url={active.url}
            height={420}
            onActivate={() => setModal({ url: active.url, label: active.label, kind: active.kind })}
          />
        </div>
        <div>
          <h3 className="font-serif text-2xl text-ink">{active.label}</h3>
          <p className="mt-2 text-[14px] text-ink-2">{t("viewerDescHome")}</p>
          <div className="mt-5 flex flex-wrap gap-2">
            {models.map((m) => (
              <button
                key={m.id}
                onClick={() => setActiveId(m.id)}
                className={`rounded-full border px-4 py-2 text-sm font-semibold transition ${
                  active.id === m.id ? "border-forest bg-forest text-white" : "border-line bg-paper text-ink-2 hover:border-forest/40"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
          <Link href="/keychains" className="mt-6 inline-flex min-h-[48px] items-center justify-center gap-2 rounded-full bg-forest px-7 text-[15px] font-semibold text-white transition hover:brightness-110">
            {t("createOwn")} →
          </Link>
        </div>
      </div>

      {/* Dual infinite carousels */}
      <div className="mt-16">
        <div className="mb-5 flex items-end justify-between">
          <h3 className="font-serif text-2xl text-ink">{t("galleryTitle")}</h3>
          <Link href="/showcase" className="text-sm font-semibold text-forest hover:underline">{t("allGallery")} →</Link>
        </div>
        <div className="space-y-3">
          <Row tiles={ROW_A} dir="right" onOpen={(tl) => setModal(tileToModel(tl))} rotateLabel={t("rotate3d")} keyAlt={t("keyItem")} mapAlt={t("mapItem")} />
          <Row tiles={ROW_B} dir="left" onOpen={(tl) => setModal(tileToModel(tl))} rotateLabel={t("rotate3d")} keyAlt={t("keyItem")} mapAlt={t("mapItem")} />
        </div>
      </div>

      <ModelModal model={modal} onClose={() => setModal(null)} />
    </section>
  );
}
