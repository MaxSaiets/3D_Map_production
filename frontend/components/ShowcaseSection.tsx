"use client";

import { useState } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import ModelModal, { type ModalModel } from "@/components/ModelModal";

const Model3DViewer = dynamic(() => import("@/components/Model3DViewer"), {
  ssr: false,
  loading: () => (
    <div className="flex h-[420px] w-full items-center justify-center text-sm text-ink-3">Завантаження 3D…</div>
  ),
});

const MODELS = [
  { id: "keychain-home", label: "Брелок «HOME»", url: "/models/keychain-home.glb", kind: "key" as const },
  { id: "keychain-city", label: "Щільний центр", url: "/models/keychain-city.glb", kind: "key" as const },
  { id: "keychain-water", label: "З рікою", url: "/models/keychain-water.glb", kind: "key" as const },
  { id: "keychain-bridge", label: "З мостами", url: "/models/keychain-bridge.glb", kind: "key" as const },
  { id: "map-district", label: "3D-район", url: "/models/map-district.glb", kind: "map" as const },
  { id: "map-dense", label: "Щільний квартал", url: "/models/map-dense.glb", kind: "map" as const },
];

const WEB_KEY = ["/models/keychain-home.glb", "/models/keychain-city.glb", "/models/keychain-water.glb", "/models/keychain-bridge.glb"];
const WEB_MAP = ["/models/map-district.glb", "/models/map-dense.glb", "/models/map-block.glb"];

const KEYCHAINS = Array.from({ length: 8 }, (_, i) => `/showcase/keychain-${i + 1}.png`);
const MAPS = Array.from({ length: 11 }, (_, i) => `/showcase/map-${i + 1}.png`);

type Tile = { src: string; kind: "key" | "map"; idx: number };
const TILES: Tile[] = [
  ...KEYCHAINS.map((src, idx) => ({ src, kind: "key" as const, idx })),
  ...MAPS.map((src, idx) => ({ src, kind: "map" as const, idx })),
];
const ROW_A = TILES.filter((_, i) => i % 2 === 0);
const ROW_B = TILES.filter((_, i) => i % 2 === 1);

function tileToModel(t: Tile): ModalModel {
  if (t.kind === "key") {
    return { url: WEB_KEY[t.idx % WEB_KEY.length], label: "Брелок-мапа міста", kind: "key", price: "від 290 ₴" };
  }
  return { url: WEB_MAP[t.idx % WEB_MAP.length], label: "3D-район міста", kind: "map", price: "від 690 ₴" };
}

function Row({ tiles, dir, onOpen }: { tiles: Tile[]; dir: "left" | "right"; onOpen: (t: Tile) => void }) {
  const doubled = [...tiles, ...tiles];
  return (
    <div className={`marquee marquee-${dir}`}>
      <div className="marquee-track">
        {doubled.map((t, i) => (
          <button
            key={`${t.kind}-${t.idx}-${i}`}
            onClick={() => onOpen(t)}
            className="group relative h-[180px] w-[230px] shrink-0 overflow-hidden rounded-[18px] border border-line"
            title="Відкрити в 3D"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={t.src} alt={t.kind === "key" ? "Брелок-мапа" : "3D-район"} loading="lazy"
                 className="h-full w-full object-cover transition duration-500 group-hover:scale-[1.07]" />
            <span className="pointer-events-none absolute inset-0 flex items-center justify-center bg-ink/0 transition group-hover:bg-ink/25">
              <span className="rounded-full bg-white/90 px-3 py-1 text-[11px] font-bold text-ink opacity-0 transition group-hover:opacity-100">
                Покрутити в 3D ↻
              </span>
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

export default function ShowcaseSection() {
  const [active, setActive] = useState(MODELS[0]);
  const [modal, setModal] = useState<ModalModel | null>(null);

  return (
    <section id="showcase" className="mx-auto max-w-[1360px] px-5 py-20 lg:px-8 lg:py-24">
      <div className="mb-10 text-center">
        <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-ink-3">Готові моделі</p>
        <h2 className="mt-3 text-[clamp(30px,3.6vw,54px)] leading-tight">Покрути в руках — ще до друку</h2>
        <p className="mx-auto mt-3 max-w-[620px] text-[15px] text-ink-2">
          Реальні моделі, надруковані з твоєї мапи. Обертай у 3D, обери район — і замов.
        </p>
      </div>

      {/* Interactive 3D */}
      <div className="grid items-center gap-8 lg:grid-cols-[1.15fr_1fr]">
        <button
          onClick={() => setModal({ url: active.url, label: active.label, kind: active.kind })}
          className="overflow-hidden rounded-[28px] border border-line bg-gradient-to-b from-[#f6f1e6] to-[#ece4d3] shadow-[0_30px_80px_rgba(15,23,42,0.10)]"
          title="Відкрити на весь екран"
        >
          <Model3DViewer url={active.url} height={420} />
        </button>
        <div>
          <h3 className="font-serif text-2xl text-ink">{active.label}</h3>
          <p className="mt-2 text-[14px] text-ink-2">
            Точна 3D-мапа: вулиці, будівлі, парки й вода в масштабі. Натисни на модель — відкриється на весь екран,
            крути й наближай.
          </p>
          <div className="mt-5 flex flex-wrap gap-2">
            {MODELS.map((m) => (
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
          <Link href="/keychains" className="mt-6 inline-flex min-h-12 items-center justify-center gap-2 rounded-full bg-forest px-7 text-[15px] font-semibold text-white transition hover:brightness-110">
            Створити свій брелок →
          </Link>
        </div>
      </div>

      {/* Dual infinite carousels */}
      <div className="mt-16">
        <div className="mb-5 flex items-end justify-between">
          <h3 className="font-serif text-2xl text-ink">Галерея надрукованих</h3>
          <Link href="/showcase" className="text-sm font-semibold text-forest hover:underline">Вся галерея →</Link>
        </div>
        <div className="space-y-3">
          <Row tiles={ROW_A} dir="right" onOpen={(t) => setModal(tileToModel(t))} />
          <Row tiles={ROW_B} dir="left" onOpen={(t) => setModal(tileToModel(t))} />
        </div>
      </div>

      <ModelModal model={modal} onClose={() => setModal(null)} />
    </section>
  );
}
