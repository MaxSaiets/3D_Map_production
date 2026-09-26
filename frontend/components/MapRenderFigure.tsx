import { MAP_RENDERS } from "@/lib/mapRenders";

/** URL рендера моделі (або null) — для JSON-LD Product.image та og:image. */
export function mapRenderUrl(id: string, base = ""): string | null {
  return MAP_RENDERS.has(id) ? `${base}/maps-renders/${id}.webp` : null;
}

/**
 * Студійний рендер РЕАЛЬНОЇ згенерованої 3D-моделі для сторінки міста/району
 * (tools/night_city_renders.py: мапа 80 мм, ділянка 800×800 м навколо центру).
 * Чесний підпис — це рендер моделі, не фото друку.
 */
export default function MapRenderFigure({ id, name, isUA }: { id: string; name: string; isUA: boolean }) {
  if (!MAP_RENDERS.has(id)) return null;
  return (
    <figure className="mt-8 overflow-hidden rounded-[20px] border border-line-soft bg-white/60">
      <img
        src={`/maps-renders/${id}.webp`}
        srcSet={`/maps-renders/${id}-400.webp 400w, /maps-renders/${id}.webp 640w`}
        sizes="(max-width: 700px) 100vw, 640px"
        width={640}
        height={480}
        alt={isUA ? `3D-модель: ${name} — макет центру, ділянка 800 × 800 м` : `3D model: ${name} — centre, 800 × 800 m area`}
        className="h-auto w-full"
        decoding="async"
      />
      <figcaption className="px-4 py-2.5 text-[13px] leading-snug text-ink-3">
        {isUA
          ? `Рендер реальної 3D-моделі з нашого конструктора: ${name}, ділянка 800 × 800 м, мапа 8 см. Свою ділянку можна обрати будь-де.`
          : `Render of a real 3D model from our builder: ${name}, 800 × 800 m area, 8 cm map. You can pick any area you like.`}
      </figcaption>
    </figure>
  );
}
