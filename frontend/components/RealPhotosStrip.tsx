/**
 * Смужка СПРАВЖНІХ фото надрукованих виробів (public/showcase/real-*.webp).
 *
 * Навіщо (аналітика 16–30.09.2026): із 16 людей, що запустили генерацію, 6 не
 * дочекались моделі, а з тих, хто дочекався, ніхто не відкрив замовлення. Людина
 * бачить лише рендер і не певна, що отримає фізичну річ такої якості. Реальні
 * фото — під час очікування (зайняти й переконати) і біля кнопки «Замовити».
 * Лише справжні фото наших виробів, без вигаданих відгуків.
 */
const MAP_PHOTOS = [1, 2, 3, 7, 8, 9, 10];
const KEYCHAIN_PHOTOS = [4, 5, 6];

export function RealPhotosStrip({
  kind,
  title,
  compact = false,
  testId,
}: {
  kind: "map" | "keychain";
  title: string;
  compact?: boolean;
  testId?: string;
}) {
  const ids = (kind === "keychain" ? [...KEYCHAIN_PHOTOS, 1, 8] : MAP_PHOTOS).slice(0, compact ? 4 : 6);
  return (
    <figure className="flex flex-col gap-1.5" data-testid={testId}>
      <figcaption className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">{title}</figcaption>
      <div className={`grid gap-1.5 ${compact ? "grid-cols-4" : "grid-cols-3"}`}>
        {ids.map((i) => (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            key={i}
            src={`/showcase/real-${i}.webp`}
            alt=""
            loading="lazy"
            decoding="async"
            className="aspect-square w-full rounded-[10px] border border-[var(--surface-border)] object-cover"
          />
        ))}
      </div>
    </figure>
  );
}
