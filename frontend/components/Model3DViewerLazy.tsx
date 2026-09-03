"use client";

import dynamic from "next/dynamic";
import { useRef, type ComponentProps } from "react";
import { useNearViewport } from "@/components/ViewportRuntime";

/**
 * T-6.5 — three.js на лендінгу тільки коли демо-картка доїхала до екрана.
 *
 * Model3DViewer уже сам НЕ створює WebGL-канвас, поки він за екраном, АЛЕ
 * `next/dynamic` усе одно тягне його чанк (three + drei ≈1.5 МБ у dev) одразу
 * на гідрації. На телефоні герой-демо лежить нижче згину: заміряно — канвасів
 * 0, а 1.5 МБ JS завантажено й розпарсено. Тут чанк не просять узагалі, поки
 * блок не наблизився до вікна; до того на місці вʼюера стоїть ТА САМА
 * постер-заглушка, що й усередині Model3DViewer (без спінера — вʼюер ще не вантажиться).
 */
const Model3DViewer = dynamic(() => import("@/components/Model3DViewer"), { ssr: false });

type Props = ComponentProps<typeof Model3DViewer>;

export default function Model3DViewerLazy(props: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  // 100px < 120px внутрішнього IO у Model3DViewer: коли ми змонтували вʼюер,
  // його власний гейт уже теж спрацьовує — інакше вантажили б чанк даремно.
  const near = useNearViewport(ref, "100px");
  const height = props.height ?? 360;
  return (
    <div ref={ref}>
      {near ? (
        <Model3DViewer {...props} />
      ) : (
        <div className="relative w-full rounded-[inherit]" style={{ height }}>
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            {props.poster ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={props.poster} alt="" aria-hidden className="absolute inset-0 h-full w-full rounded-[inherit] object-cover" />
            ) : (
              <div className="absolute inset-0 animate-pulse bg-gradient-to-br from-black/[0.05] via-transparent to-black/[0.07]" />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
