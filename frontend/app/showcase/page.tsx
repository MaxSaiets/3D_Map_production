"use client";
export const dynamic = "force-dynamic";

import { useState } from "react";
import Link from "next/link";
import dynamicImport from "next/dynamic";
import { ArrowLeft } from "lucide-react";

const Model3DViewer = dynamicImport(() => import("@/components/Model3DViewer"), {
  ssr: false,
  loading: () => (
    <div className="flex h-[420px] w-full items-center justify-center text-sm text-ink-3">Завантаження 3D…</div>
  ),
});

const VIEW_MODELS = [
  { id: "keychain-home", label: "Брелок «HOME»", url: "/models/keychain-home.glb", kind: "key" },
  { id: "keychain-city", label: "Щільний центр", url: "/models/keychain-city.glb", kind: "key" },
  { id: "keychain-water", label: "Брелок з рікою", url: "/models/keychain-water.glb", kind: "key" },
  { id: "keychain-bridge", label: "Брелок з мостами", url: "/models/keychain-bridge.glb", kind: "key" },
  { id: "map-district", label: "3D-район", url: "/models/map-district.glb", kind: "map" },
  { id: "map-dense", label: "Щільний квартал", url: "/models/map-dense.glb", kind: "map" },
];

type Item = { src: string; kind: "key" | "map"; title: string; price: string };
const KEY_TITLES = ["Серце міста", "Старий центр", "Біля річки", "Мости й набережна", "Щільні квартали", "Тихі вулиці", "Парковий район", "Класичний брелок"];
const MAP_TITLES = ["Центральний район", "Біля води", "Історичний квартал", "Діловий центр", "Зелений масив", "Старе місто", "Набережна", "Проспекти", "Площа міста", "Житловий квартал", "Промзона", "Парк і алеї", "Перехрестя"];

const ITEMS: Item[] = [
  ...Array.from({ length: 8 }, (_, i) => ({ src: `/showcase/keychain-${i + 1}.png`, kind: "key" as const, title: KEY_TITLES[i] || "Брелок-мапа", price: "від 290 ₴" })),
  ...Array.from({ length: 13 }, (_, i) => ({ src: `/showcase/map-${i + 1}.png`, kind: "map" as const, title: MAP_TITLES[i] || "3D-район", price: "від 690 ₴" })),
];

export default function ShowcasePage() {
  const [filter, setFilter] = useState<"all" | "key" | "map">("all");
  const [active, setActive] = useState(VIEW_MODELS[0]);
  const items = ITEMS.filter((it) => filter === "all" || it.kind === filter);
  const viewModels = VIEW_MODELS.filter((m) => filter === "all" || m.kind === filter);

  return (
    <div className="mx-auto min-h-[100dvh] max-w-[1280px] px-5 py-8 lg:px-8">
      <Link href="/" className="mb-6 inline-flex items-center gap-1.5 text-[13px] font-semibold text-ink-2 hover:text-ink">
        <ArrowLeft size={15} /> На головну
      </Link>

      <div className="text-center">
        <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-ink-3">Галерея</p>
        <h1 className="mt-2 font-serif text-[clamp(30px,4vw,56px)] text-ink">Надруковані мапи й брелки</h1>
        <p className="mx-auto mt-3 max-w-[640px] text-[15px] text-ink-2">
          Реальні моделі з міст України та світу. Обертай у 3D, обери — і замов друк.
        </p>
      </div>

      {/* Filter */}
      <div className="mt-7 flex justify-center gap-2">
        {([["all", "Усе"], ["key", "Брелки"], ["map", "3D-мапи"]] as const).map(([k, lbl]) => (
          <button
            key={k}
            onClick={() => {
              setFilter(k);
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
        <div className="overflow-hidden rounded-[28px] border border-line bg-gradient-to-b from-[#f4efe3] to-[#e7ddc9] shadow-[0_30px_80px_rgba(15,23,42,0.10)]">
          <Model3DViewer url={active.url} height={440} />
        </div>
        <div>
          <h2 className="font-serif text-2xl text-ink">{active.label}</h2>
          <p className="mt-2 text-[14px] text-ink-2">
            Точна 3D-мапа: вулиці, будівлі, парки й вода в масштабі. Перетягни, щоб роздивитись з усіх боків.
          </p>
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
            <Link href="/keychains" className="inline-flex min-h-12 items-center justify-center rounded-full bg-forest px-6 text-[15px] font-semibold text-white hover:brightness-110">
              Створити брелок →
            </Link>
            <Link href="/create" className="inline-flex min-h-12 items-center justify-center rounded-full border border-line px-6 text-[15px] font-semibold text-ink-2 hover:border-forest/50">
              Зробити мапу
            </Link>
          </div>
        </div>
      </div>

      {/* Gallery */}
      <div className="mt-14 grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
        {items.map((it) => (
          <div key={it.src} className="group overflow-hidden rounded-[20px] border border-line bg-paper">
            <div className="relative aspect-square overflow-hidden">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={it.src} alt={it.title} loading="lazy" className="h-full w-full object-cover transition duration-500 group-hover:scale-[1.06]" />
            </div>
            <div className="flex items-center justify-between gap-2 px-3 py-3">
              <div>
                <div className="text-[13px] font-semibold text-ink">{it.title}</div>
                <div className="text-[11px] text-ink-3">{it.kind === "key" ? "Брелок 55×30 мм" : "3D-район"}</div>
              </div>
              <Link
                href={it.kind === "key" ? "/keychains" : "/create"}
                className="shrink-0 rounded-full bg-forest px-3 py-1.5 text-[11px] font-bold text-white hover:brightness-110"
              >
                {it.price}
              </Link>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-14 rounded-[24px] bg-forest px-6 py-10 text-center text-[#F4EFE4]">
        <h3 className="font-serif text-[clamp(22px,3vw,34px)]">Не знайшов своє місто?</h3>
        <p className="mx-auto mt-2 max-w-[520px] text-[14px] opacity-90">Створи мапу будь-якого міста світу за пару хвилин.</p>
        <Link href="/create" className="mt-5 inline-flex min-h-12 items-center justify-center rounded-full bg-white px-7 text-[15px] font-semibold text-forest hover:brightness-95">
          Створити свою мапу →
        </Link>
      </div>
    </div>
  );
}
