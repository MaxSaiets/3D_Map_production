"use client";

import { useState } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";

const Model3DViewer = dynamic(() => import("@/components/Model3DViewer"), {
  ssr: false,
  loading: () => (
    <div className="flex h-[360px] w-full items-center justify-center text-sm text-ink-3">
      Завантаження 3D…
    </div>
  ),
});

const MODELS = [
  { id: "keychain-home", label: "Брелок «HOME»", url: "/models/keychain-home.glb" },
  { id: "keychain-city", label: "Щільний центр", url: "/models/keychain-city.glb" },
  { id: "keychain-water", label: "З рікою", url: "/models/keychain-water.glb" },
  { id: "keychain-bridge", label: "З мостами", url: "/models/keychain-bridge.glb" },
  { id: "map-district", label: "3D-район", url: "/models/map-district.glb" },
  { id: "map-dense", label: "Щільний квартал", url: "/models/map-dense.glb" },
];

// Curated render gallery (transparent PNGs in /public/showcase).
const KEYCHAINS = Array.from({ length: 8 }, (_, i) => `/showcase/keychain-${i + 1}.png`);
const MAPS = Array.from({ length: 13 }, (_, i) => `/showcase/map-${i + 1}.png`);

export default function ShowcaseSection() {
  const [active, setActive] = useState(MODELS[0]);

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
        <div className="overflow-hidden rounded-[28px] border border-line bg-gradient-to-b from-[#f6f1e6] to-[#ece4d3] shadow-[0_30px_80px_rgba(15,23,42,0.10)]">
          <Model3DViewer url={active.url} height={420} />
        </div>
        <div>
          <h3 className="font-serif text-2xl text-ink">{active.label}</h3>
          <p className="mt-2 text-[14px] text-ink-2">
            Брелок-мапа 55×30 мм: вулиці, будівлі, парки й вода — у точному масштабі твого міста.
            Перетягни модель, щоб роздивитись з усіх боків.
          </p>
          <div className="mt-5 flex flex-wrap gap-2">
            {MODELS.map((m) => (
              <button
                key={m.id}
                onClick={() => setActive(m)}
                className={`rounded-full border px-4 py-2 text-sm font-semibold transition ${
                  active.id === m.id
                    ? "border-forest bg-forest text-white"
                    : "border-line bg-paper text-ink-2 hover:border-forest/40"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
          <Link
            href="/keychains"
            className="mt-6 inline-flex min-h-12 items-center justify-center gap-2 rounded-full bg-forest px-7 text-[15px] font-semibold text-white transition hover:brightness-110"
          >
            Створити свій брелок →
          </Link>
        </div>
      </div>

      {/* Photo gallery */}
      <div className="mt-16">
        <div className="mb-5 flex items-end justify-between">
          <h3 className="font-serif text-2xl text-ink">Галерея надрукованих</h3>
          <Link href="/showcase" className="text-sm font-semibold text-forest hover:underline">
            Вся галерея →
          </Link>
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {[...KEYCHAINS, ...MAPS].map((src, i) => {
            const isKey = i < KEYCHAINS.length;
            return (
              <Link
                key={src}
                href={isKey ? "/keychains" : "/create"}
                className="group relative aspect-square overflow-hidden rounded-[20px] border border-line"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={src}
                  alt={isKey ? "Брелок-мапа міста" : "3D-район міста"}
                  loading="lazy"
                  className="h-full w-full object-cover transition duration-500 group-hover:scale-[1.07]"
                />
                <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between bg-gradient-to-t from-ink/70 to-transparent p-3 opacity-0 transition group-hover:opacity-100">
                  <span className="text-[11px] font-semibold text-white">
                    {isKey ? "Брелок 55×30 мм" : "3D-район міста"}
                  </span>
                  <span className="rounded-full bg-white/90 px-2 py-0.5 text-[10px] font-bold text-ink">
                    {isKey ? "від 290 ₴" : "від 690 ₴"}
                  </span>
                </div>
              </Link>
            );
          })}
        </div>
      </div>
    </section>
  );
}
