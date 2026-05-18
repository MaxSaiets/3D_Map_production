"use client";

import { useMemo, useRef, useState } from "react";

export type KeychainBaseShape = "rounded" | "capsule" | "tag" | "octagon";
export type KeychainLoopStyle = "round" | "teardrop" | "slot" | "side-tab";

export type KeychainDesignerConfig = {
  bodyWidthMm: number;
  bodyHeightMm: number;
  cornerRadiusMm: number;
  baseShape: KeychainBaseShape;
  loopStyle: KeychainLoopStyle;
  loopAngleDeg: number;
  loopXMm: number;
  loopYMm: number;
  loopOuterMm: number;
  loopInnerMm: number;
  mapXMm: number;
  mapYMm: number;
  mapWidthMm: number;
  mapHeightMm: number;
  labelXMm: number;
  labelYMm: number;
  labelWidthMm: number;
  labelBandMm: number;
  labelTextHeightMm: number;
  labelStrokeMm: number;
  labelAngleDeg: number;
  rimWidthMm: number;
  rimHeightMm: number;
};

export type KeychainTemplate = {
  id: string;
  name: string;
  description: string;
  design: KeychainDesignerConfig;
};

type DragTarget = "body" | "map-move" | "map-resize" | "loop" | "label";
type DragSession = {
  target: DragTarget;
  start: { x: number; y: number };
  initial: KeychainDesignerConfig;
};

const MIN_MAP_WIDTH_MM = 28;
const MIN_MAP_HEIGHT_MM = 18;
const SNAP_MM = 1.15;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function snapTo(value: number, target: number, threshold = SNAP_MM) {
  return Math.abs(value - target) <= threshold ? target : value;
}

function clampLoopToBody(next: KeychainDesignerConfig) {
  const margin = Math.max(next.loopOuterMm * 0.85, 4);
  next.loopXMm = clamp(next.loopXMm, -margin, next.bodyWidthMm + margin);
  next.loopYMm = clamp(next.loopYMm, -margin, next.bodyHeightMm + margin);
  next.loopXMm = snapTo(next.loopXMm, next.bodyWidthMm / 2);
  next.loopYMm = snapTo(next.loopYMm, next.bodyHeightMm / 2);
  next.loopYMm = snapTo(next.loopYMm, -Math.max(next.loopOuterMm * 0.58, 3.2));
  next.loopYMm = snapTo(next.loopYMm, next.bodyHeightMm + Math.max(next.loopOuterMm * 0.58, 3.2));
  next.loopXMm = snapTo(next.loopXMm, -Math.max(next.loopOuterMm * 0.58, 3.2));
  next.loopXMm = snapTo(next.loopXMm, next.bodyWidthMm + Math.max(next.loopOuterMm * 0.58, 3.2));
  return next;
}

function fitAfterBodyResize(next: KeychainDesignerConfig) {
  const minMapWidthMm = Math.min(MIN_MAP_WIDTH_MM, next.bodyWidthMm);
  const minMapHeightMm = Math.min(MIN_MAP_HEIGHT_MM, next.bodyHeightMm);
  next.mapXMm = clamp(next.mapXMm, 0, Math.max(0, next.bodyWidthMm - minMapWidthMm));
  next.mapYMm = clamp(next.mapYMm, 0, Math.max(0, next.bodyHeightMm - minMapHeightMm));
  next.mapWidthMm = clamp(next.mapWidthMm, minMapWidthMm, Math.max(minMapWidthMm, next.bodyWidthMm - next.mapXMm));
  next.mapHeightMm = clamp(next.mapHeightMm, minMapHeightMm, Math.max(minMapHeightMm, next.bodyHeightMm - next.mapYMm));
  next.labelXMm = clamp(next.labelXMm, 4, Math.max(4, next.bodyWidthMm - 4));
  next.labelYMm = clamp(next.labelYMm, 4, Math.max(4, next.bodyHeightMm - 4));
  next.labelWidthMm = clamp(next.labelWidthMm, 8, next.bodyWidthMm);
  next.labelTextHeightMm = clamp(next.labelTextHeightMm, 2.4, 8.5);
  next.labelStrokeMm = clamp(next.labelStrokeMm, 0.4, 2.0);
  clampLoopToBody(next);
  return next;
}

export const DEFAULT_KEYCHAIN_DESIGN: KeychainDesignerConfig = {
  bodyWidthMm: 35,
  bodyHeightMm: 55,
  cornerRadiusMm: 4.2,
  baseShape: "rounded",
  loopStyle: "round",
  loopAngleDeg: 0,
  loopXMm: 17.5,
  loopYMm: -4,
  loopOuterMm: 6.8,
  loopInnerMm: 3.1,
  mapXMm: 2,
  mapYMm: 3,
  mapWidthMm: 31,
  mapHeightMm: 40,
  labelXMm: 17.5,
  labelYMm: 49.5,
  labelWidthMm: 30,
  labelBandMm: 9.5,
  labelTextHeightMm: 4.2,
  labelStrokeMm: 0.75,
  labelAngleDeg: 0,
  rimWidthMm: 1.2,
  rimHeightMm: 0.45,
};

export const KEYCHAIN_TEMPLATES: KeychainTemplate[] = [
  {
    id: "classic-wide",
    name: "35 x 55",
    description: "Стандартний компактний вертикальний брелок.",
    design: DEFAULT_KEYCHAIN_DESIGN,
  },
  {
    id: "right-loop",
    name: "Side Loop",
    description: "Петля справа, зручно для широкої карти.",
    design: {
      ...DEFAULT_KEYCHAIN_DESIGN,
      bodyWidthMm: 55,
      bodyHeightMm: 35,
      loopXMm: 60.5,
      loopYMm: 17.5,
      loopAngleDeg: 270,
      mapXMm: 3,
      mapYMm: 3,
      mapWidthMm: 49,
      mapHeightMm: 22,
      labelXMm: 27.5,
      labelYMm: 30.5,
      labelWidthMm: 45,
    },
  },
  {
    id: "vertical-tag",
    name: "Vertical",
    description: "Вертикальний брелок з повернутим написом.",
    design: {
      ...DEFAULT_KEYCHAIN_DESIGN,
      bodyWidthMm: 35,
      bodyHeightMm: 55,
      baseShape: "tag",
      loopXMm: 17.5,
      loopYMm: -4,
      mapXMm: 3,
      mapYMm: 4,
      mapWidthMm: 29,
      mapHeightMm: 38,
      labelXMm: 17.5,
      labelYMm: 49,
      labelWidthMm: 28,
      labelAngleDeg: 0,
    },
  },
  {
    id: "soft-capsule",
    name: "Capsule",
    description: "М'яка капсульна форма з slot-вушком.",
    design: {
      ...DEFAULT_KEYCHAIN_DESIGN,
      bodyWidthMm: 55,
      bodyHeightMm: 35,
      baseShape: "capsule",
      loopStyle: "slot",
      loopXMm: 9,
      loopYMm: -3.5,
      mapXMm: 5,
      mapYMm: 4,
      mapWidthMm: 45,
      mapHeightMm: 21,
      labelXMm: 27.5,
      labelYMm: 30,
      labelWidthMm: 44,
    },
  },
];

function bodyPath(value: KeychainDesignerConfig) {
  const w = value.bodyWidthMm;
  const h = value.bodyHeightMm;
  const r = clamp(value.cornerRadiusMm, 0, Math.min(w, h) / 2);
  if (value.baseShape === "capsule") {
    const cr = h / 2;
    return `M ${cr} 0 H ${w - cr} A ${cr} ${cr} 0 0 1 ${w - cr} ${h} H ${cr} A ${cr} ${cr} 0 0 1 ${cr} 0 Z`;
  }
  if (value.baseShape === "tag") {
    const cut = Math.min(w, h) * 0.16;
    return `M ${r} 0 H ${w - cut} L ${w} ${cut} V ${h - r} Q ${w} ${h} ${w - r} ${h} H ${r} Q 0 ${h} 0 ${h - r} V ${r} Q 0 0 ${r} 0 Z`;
  }
  if (value.baseShape === "octagon") {
    const cut = Math.min(w, h) * 0.13;
    return `M ${cut} 0 H ${w - cut} L ${w} ${cut} V ${h - cut} L ${w - cut} ${h} H ${cut} L 0 ${h - cut} V ${cut} Z`;
  }
  return `M ${r} 0 H ${w - r} Q ${w} 0 ${w} ${r} V ${h - r} Q ${w} ${h} ${w - r} ${h} H ${r} Q 0 ${h} 0 ${h - r} V ${r} Q 0 0 ${r} 0 Z`;
}

function LoopPreview({ value }: { value: KeychainDesignerConfig }) {
  const outer = value.loopOuterMm;
  const inner = value.loopInnerMm;
  const tabWidth = Math.max((outer - inner) * 1.35, 2.4);
  const tabHeight = Math.max(outer * 0.95, 5);

  if (value.loopStyle === "slot") {
    return (
      <g transform={`rotate(${value.loopAngleDeg} ${value.loopXMm} ${value.loopYMm})`}>
        <rect x={value.loopXMm - outer * 1.28} y={value.loopYMm - outer * 0.72} width={outer * 2.56} height={outer * 1.44} rx={outer * 0.72} fill="#a6926b" />
        <rect x={value.loopXMm - inner * 1.25} y={value.loopYMm - inner * 0.58} width={inner * 2.5} height={inner * 1.16} rx={inner * 0.58} fill="#050a18" stroke="rgba(255,255,255,0.35)" strokeWidth={0.25} />
        <rect x={value.loopXMm - tabWidth / 2} y={value.loopYMm} width={tabWidth} height={tabHeight} rx={tabWidth / 2} fill="#a6926b" />
      </g>
    );
  }

  if (value.loopStyle === "teardrop") {
    return (
      <g transform={`rotate(${value.loopAngleDeg} ${value.loopXMm} ${value.loopYMm})`}>
        <path d={`M ${value.loopXMm} ${value.loopYMm - outer} C ${value.loopXMm + outer} ${value.loopYMm - outer}, ${value.loopXMm + outer} ${value.loopYMm + outer * 0.45}, ${value.loopXMm} ${value.loopYMm + outer + tabHeight * 0.35} C ${value.loopXMm - outer} ${value.loopYMm + outer * 0.45}, ${value.loopXMm - outer} ${value.loopYMm - outer}, ${value.loopXMm} ${value.loopYMm - outer} Z`} fill="#a6926b" />
        <circle cx={value.loopXMm} cy={value.loopYMm} r={inner} fill="#050a18" stroke="rgba(255,255,255,0.35)" strokeWidth={0.25} />
      </g>
    );
  }

  if (value.loopStyle === "side-tab") {
    return (
      <g transform={`rotate(${value.loopAngleDeg} ${value.loopXMm} ${value.loopYMm})`}>
        <rect x={value.loopXMm - outer} y={value.loopYMm - outer * 0.78} width={outer * 2} height={outer * 1.56} rx={outer * 0.45} fill="#a6926b" />
        <circle cx={value.loopXMm} cy={value.loopYMm} r={inner} fill="#050a18" stroke="rgba(255,255,255,0.35)" strokeWidth={0.25} />
        <rect x={value.loopXMm - tabWidth / 2} y={value.loopYMm} width={tabWidth} height={tabHeight} rx={tabWidth / 2} fill="#a6926b" />
      </g>
    );
  }

  return (
    <g transform={`rotate(${value.loopAngleDeg} ${value.loopXMm} ${value.loopYMm})`}>
      <circle cx={value.loopXMm} cy={value.loopYMm} r={outer} fill="#a6926b" />
      <rect x={value.loopXMm - tabWidth / 2} y={value.loopYMm} width={tabWidth} height={tabHeight} rx={tabWidth / 2} fill="#a6926b" />
      <circle cx={value.loopXMm} cy={value.loopYMm} r={inner} fill="#050a18" stroke="rgba(255,255,255,0.35)" strokeWidth={0.25} />
    </g>
  );
}

function TemplateMiniature({ design, label, active }: { design: KeychainDesignerConfig; label: string; active: boolean }) {
  const pad = Math.max(design.loopOuterMm * 2.2, 12);
  const view = {
    x: -pad,
    y: -pad,
    w: design.bodyWidthMm + pad * 2,
    h: design.bodyHeightMm + pad * 2,
  };
  const clipId = `templateMapClip-${design.bodyWidthMm}-${design.bodyHeightMm}-${design.loopXMm}-${design.loopYMm}`.replace(/\./g, "-");

  return (
    <svg viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`} className="h-20 w-full rounded-[14px] bg-[#050a18]">
      <defs>
        <clipPath id={clipId}>
          <rect
            x={design.mapXMm}
            y={design.mapYMm}
            width={design.mapWidthMm}
            height={design.mapHeightMm}
            rx={Math.min(design.cornerRadiusMm, 3)}
          />
        </clipPath>
      </defs>
      <LoopPreview value={design} />
      <path d={bodyPath(design)} fill="#a6926b" stroke={active ? "#5eead4" : "rgba(255,255,255,0.35)"} strokeWidth={active ? 0.75 : 0.35} />
      {design.rimWidthMm > 0 && (
        <path
          d={bodyPath(design)}
          fill="none"
          stroke="rgba(68,55,32,0.34)"
          strokeWidth={Math.min(design.rimWidthMm * 1.6, 5)}
        />
      )}
      <g clipPath={`url(#${clipId})`}>
        <rect x={design.mapXMm} y={design.mapYMm} width={design.mapWidthMm} height={design.mapHeightMm} fill="#b7ab8e" />
        <path d={`M ${design.mapXMm + design.mapWidthMm * 0.18} ${design.mapYMm - 3} L ${design.mapXMm + design.mapWidthMm * 0.34} ${design.mapYMm + design.mapHeightMm + 4}`} stroke="#101010" strokeWidth={1.1} />
        <path d={`M ${design.mapXMm + design.mapWidthMm * 0.56} ${design.mapYMm - 3} L ${design.mapXMm + design.mapWidthMm * 0.66} ${design.mapYMm + design.mapHeightMm + 4}`} stroke="#101010" strokeWidth={1.1} />
        <path d={`M ${design.mapXMm + design.mapWidthMm * 0.76} ${design.mapYMm + 3} C ${design.mapXMm + design.mapWidthMm * 0.9} ${design.mapYMm + design.mapHeightMm * 0.45}, ${design.mapXMm + design.mapWidthMm * 0.62} ${design.mapYMm + design.mapHeightMm * 0.62}, ${design.mapXMm + design.mapWidthMm * 0.8} ${design.mapYMm + design.mapHeightMm - 2}`} fill="none" stroke="#6fa1c8" strokeWidth={1.5} />
        <rect x={design.mapXMm + design.mapWidthMm * 0.48} y={design.mapYMm + design.mapHeightMm * 0.28} width={design.mapWidthMm * 0.12} height={design.mapHeightMm * 0.1} fill="#d8d8d8" />
        <rect x={design.mapXMm + design.mapWidthMm * 0.22} y={design.mapYMm + design.mapHeightMm * 0.52} width={design.mapWidthMm * 0.14} height={design.mapHeightMm * 0.1} fill="#d8d8d8" />
        <path d={`M ${design.mapXMm + design.mapWidthMm * 0.69} ${design.mapYMm + design.mapHeightMm * 0.5} l 4 1 l -1 4 l -5 1 z`} fill="#3f8a4d" />
      </g>
      <text
        x={design.labelXMm}
        y={design.labelYMm}
        textAnchor="middle"
        dominantBaseline="middle"
        fill="#f1f5f9"
        stroke="rgba(248,250,252,0.45)"
        strokeWidth={Math.max(design.labelStrokeMm * 0.08, 0.04)}
        paintOrder="stroke"
        fontSize={Math.max(design.labelTextHeightMm * 0.7, 2.5)}
        fontWeight={700}
        transform={`rotate(${design.labelAngleDeg} ${design.labelXMm} ${design.labelYMm})`}
      >
        {label || "TEXT"}
      </text>
    </svg>
  );
}

export function KeychainTemplateStrip({
  value,
  label,
  onSelect,
}: {
  value: KeychainDesignerConfig;
  label: string;
  onSelect: (value: KeychainDesignerConfig) => void;
}) {
  return (
    <div className="border-t border-white/10 bg-[#070d1d] px-3 py-3">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-300">Готові шаблони</div>
        <div className="text-[11px] font-medium text-slate-400">tap / click</div>
      </div>
      <div className="flex gap-3 overflow-x-auto pb-1">
        {KEYCHAIN_TEMPLATES.map((template) => {
          const active =
            template.design.baseShape === value.baseShape &&
            template.design.loopStyle === value.loopStyle &&
            Math.round(template.design.bodyWidthMm) === Math.round(value.bodyWidthMm) &&
            Math.round(template.design.bodyHeightMm) === Math.round(value.bodyHeightMm);
          return (
            <button
              key={template.id}
              type="button"
              onClick={() => onSelect(template.design)}
              className={`min-w-[142px] rounded-[18px] border p-2 text-left transition ${
                active
                  ? "border-teal-300 bg-teal-300/10"
                  : "border-white/10 bg-white/[0.04] hover:border-white/25 hover:bg-white/[0.07]"
              }`}
            >
              <TemplateMiniature design={template.design} label={label} active={active} />
              <div className="mt-2 text-xs font-semibold text-white">{template.name}</div>
              <div className="mt-0.5 line-clamp-2 text-[11px] leading-4 text-slate-400">{template.description}</div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function KeychainDesigner({
  value,
  label,
  onChange,
}: {
  value: KeychainDesignerConfig;
  label: string;
  onChange: (value: KeychainDesignerConfig) => void;
}) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragSessionRef = useRef<DragSession | null>(null);
  const dragCleanupRef = useRef<(() => void) | null>(null);
  const [dragTarget, setDragTarget] = useState<DragTarget | null>(null);

  const view = useMemo(() => {
    const pad = Math.max(value.loopOuterMm * 2.4, 14);
    return {
      x: -pad,
      y: -pad,
      w: value.bodyWidthMm + pad * 2,
      h: value.bodyHeightMm + pad * 2,
    };
  }, [value.bodyWidthMm, value.bodyHeightMm, value.loopOuterMm]);

  const pointerToMmFromClient = (clientX: number, clientY: number) => {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const matrix = svg.getScreenCTM();
    if (matrix) {
      const point = svg.createSVGPoint();
      point.x = clientX;
      point.y = clientY;
      const svgPoint = point.matrixTransform(matrix.inverse());
      return { x: svgPoint.x, y: svgPoint.y };
    }
    const rect = svg.getBoundingClientRect();
    return {
      x: view.x + ((clientX - rect.left) / rect.width) * view.w,
      y: view.y + ((clientY - rect.top) / rect.height) * view.h,
    };
  };

  const updateFromClient = (clientX: number, clientY: number) => {
    const session = dragSessionRef.current;
    if (!session) return;
    const point = pointerToMmFromClient(clientX, clientY);
    const dx = point.x - session.start.x;
    const dy = point.y - session.start.y;
    const next = { ...session.initial };

    if (session.target === "body") {
      next.bodyWidthMm = clamp(session.initial.bodyWidthMm + dx, 35, 140);
      next.bodyHeightMm = clamp(session.initial.bodyHeightMm + dy, 26, 96);
      fitAfterBodyResize(next);
    } else if (session.target === "map-move") {
      next.mapXMm = clamp(session.initial.mapXMm + dx, 0, next.bodyWidthMm - next.mapWidthMm);
      next.mapYMm = clamp(session.initial.mapYMm + dy, 0, next.bodyHeightMm - next.mapHeightMm);
      const mapCenterX = snapTo(next.mapXMm + next.mapWidthMm / 2, next.bodyWidthMm / 2);
      const mapCenterY = snapTo(next.mapYMm + next.mapHeightMm / 2, next.bodyHeightMm / 2);
      next.mapXMm = clamp(mapCenterX - next.mapWidthMm / 2, 0, next.bodyWidthMm - next.mapWidthMm);
      next.mapYMm = clamp(mapCenterY - next.mapHeightMm / 2, 0, next.bodyHeightMm - next.mapHeightMm);
    } else if (session.target === "map-resize") {
      next.mapWidthMm = clamp(session.initial.mapWidthMm + dx, Math.min(MIN_MAP_WIDTH_MM, next.bodyWidthMm), next.bodyWidthMm - next.mapXMm);
      next.mapHeightMm = clamp(session.initial.mapHeightMm + dy, Math.min(MIN_MAP_HEIGHT_MM, next.bodyHeightMm), next.bodyHeightMm - next.mapYMm);
    } else if (session.target === "loop") {
      next.loopXMm = session.initial.loopXMm + dx;
      next.loopYMm = session.initial.loopYMm + dy;
      clampLoopToBody(next);
    } else if (session.target === "label") {
      next.labelXMm = clamp(session.initial.labelXMm + dx, 4, next.bodyWidthMm - 4);
      next.labelYMm = clamp(session.initial.labelYMm + dy, 4, next.bodyHeightMm - 4);
      next.labelXMm = snapTo(next.labelXMm, next.bodyWidthMm / 2);
      next.labelYMm = snapTo(next.labelYMm, next.bodyHeightMm - next.labelBandMm / 2);
    }

    onChange(next);
  };

  const updateFromPointer = (event: React.PointerEvent<SVGSVGElement>) => {
    updateFromClient(event.clientX, event.clientY);
  };

  const beginDrag = (event: React.PointerEvent<SVGElement>, target: DragTarget) => {
    event.preventDefault();
    event.stopPropagation();
    dragCleanupRef.current?.();
    const point = pointerToMmFromClient(event.clientX, event.clientY);
    dragSessionRef.current = {
      target,
      start: point,
      initial: { ...value },
    };
    setDragTarget(target);
    try {
      svgRef.current?.setPointerCapture(event.pointerId);
    } catch {
      // Pointer capture is a nicety; dragging still works through the SVG move handler.
    }

    const handleMove = (moveEvent: PointerEvent) => {
      moveEvent.preventDefault();
      updateFromClient(moveEvent.clientX, moveEvent.clientY);
    };
    const handleEnd = () => {
      dragCleanupRef.current?.();
      dragCleanupRef.current = null;
      dragSessionRef.current = null;
      setDragTarget(null);
    };
    window.addEventListener("pointermove", handleMove, { passive: false });
    window.addEventListener("pointerup", handleEnd);
    window.addEventListener("pointercancel", handleEnd);
    dragCleanupRef.current = () => {
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", handleEnd);
      window.removeEventListener("pointercancel", handleEnd);
    };
  };

  return (
    <div className="h-full min-h-[320px] rounded-[24px] bg-[#050a18] p-3">
      <svg
        ref={svgRef}
        data-testid="keychain-designer-svg"
        viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
        preserveAspectRatio="xMidYMin meet"
        className="h-full min-h-[320px] w-full touch-none select-none"
        onPointerMove={updateFromPointer}
        onPointerUp={(event) => {
          dragCleanupRef.current?.();
          dragCleanupRef.current = null;
          dragSessionRef.current = null;
          setDragTarget(null);
          try {
            svgRef.current?.releasePointerCapture(event.pointerId);
          } catch {
            // ignore pointer capture edge cases
          }
        }}
        onPointerCancel={() => {
          dragCleanupRef.current?.();
          dragCleanupRef.current = null;
          dragSessionRef.current = null;
          setDragTarget(null);
        }}
      >
        <defs>
          <pattern id="keychainGrid" width="5" height="5" patternUnits="userSpaceOnUse">
            <path d="M 5 0 L 0 0 0 5" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="0.25" />
          </pattern>
          <clipPath id="keychainMapClip">
            <rect
              x={value.mapXMm}
              y={value.mapYMm}
              width={value.mapWidthMm}
              height={value.mapHeightMm}
              rx={Math.min(value.cornerRadiusMm, 3)}
            />
          </clipPath>
        </defs>

        <rect x={view.x} y={view.y} width={view.w} height={view.h} fill="url(#keychainGrid)" />
        <g pointerEvents="none">
          <path
            d={`M ${value.bodyWidthMm / 2} ${view.y + 2} V ${value.bodyHeightMm + 7}`}
            stroke="rgba(94,234,212,0.34)"
            strokeDasharray="1.5 1.4"
            strokeWidth={0.28}
          />
          <path
            d={`M ${view.x + 2} ${value.bodyHeightMm / 2} H ${value.bodyWidthMm + 7}`}
            stroke="rgba(94,234,212,0.18)"
            strokeDasharray="1.5 1.4"
            strokeWidth={0.24}
          />
          <circle cx={value.bodyWidthMm / 2} cy={value.bodyHeightMm / 2} r={0.55} fill="rgba(94,234,212,0.72)" />
          <text x={value.bodyWidthMm / 2 + 1.2} y={value.bodyHeightMm / 2 - 1.2} fill="rgba(94,234,212,0.8)" fontSize={1.8} fontWeight={700}>
            center
          </text>
          <path
            d={`M ${value.bodyWidthMm / 2} ${value.bodyHeightMm - value.labelBandMm} V ${value.bodyHeightMm}`}
            stroke="rgba(255,255,255,0.18)"
            strokeDasharray="1.4 1.4"
            strokeWidth={0.22}
          />
        </g>
        <LoopPreview value={value} />
        <path d={bodyPath(value)} fill="#a6926b" stroke="rgba(255,255,255,0.42)" strokeWidth={0.35} />
        {value.rimWidthMm > 0 && (
          <path
            d={bodyPath(value)}
            fill="none"
            stroke="rgba(68,55,32,0.32)"
            strokeWidth={Math.min(value.rimWidthMm * 1.6, 5)}
            pointerEvents="none"
          />
        )}
        <rect
          x={value.labelXMm - value.labelWidthMm / 2}
          y={value.labelYMm - value.labelBandMm / 2}
          width={value.labelWidthMm}
          height={value.labelBandMm}
          rx={Math.min(2, value.labelBandMm / 2)}
          fill="rgba(255,255,255,0.04)"
          stroke="rgba(255,255,255,0.28)"
          strokeDasharray="1.4 1.2"
          strokeWidth={0.28}
          transform={`rotate(${value.labelAngleDeg} ${value.labelXMm} ${value.labelYMm})`}
          pointerEvents="none"
        />

        <g clipPath="url(#keychainMapClip)" opacity={0.96}>
          <rect x={value.mapXMm} y={value.mapYMm} width={value.mapWidthMm} height={value.mapHeightMm} fill="#b7ab8e" />
          {Array.from({ length: 10 }).map((_, idx) => {
            const x = value.mapXMm + ((idx * 13) % Math.max(value.mapWidthMm, 1));
            return (
              <path
                key={`road-${idx}`}
                d={`M ${x} ${value.mapYMm - 4} L ${x + 10} ${value.mapYMm + value.mapHeightMm + 6}`}
                stroke="#101010"
                strokeWidth={0.9}
                strokeLinecap="round"
              />
            );
          })}
          {Array.from({ length: 18 }).map((_, idx) => {
            const x = value.mapXMm + 4 + ((idx * 9) % Math.max(value.mapWidthMm - 12, 1));
            const y = value.mapYMm + 4 + ((idx * 7) % Math.max(value.mapHeightMm - 12, 1));
            return <rect key={`b-${idx}`} x={x} y={y} width={5 + (idx % 4)} height={2 + (idx % 3)} fill="#d8d8d8" />;
          })}
          <path
            d={`M ${value.mapXMm + value.mapWidthMm * 0.62} ${value.mapYMm + 3} C ${value.mapXMm + value.mapWidthMm * 0.8} ${value.mapYMm + value.mapHeightMm * 0.35}, ${value.mapXMm + value.mapWidthMm * 0.55} ${value.mapYMm + value.mapHeightMm * 0.58}, ${value.mapXMm + value.mapWidthMm * 0.82} ${value.mapYMm + value.mapHeightMm - 2}`}
            fill="none"
            stroke="#6fa1c8"
            strokeWidth={1.4}
          />
          <path
            d={`M ${value.mapXMm + value.mapWidthMm * 0.68} ${value.mapYMm + value.mapHeightMm * 0.5} l 5 1 l -1 4 l -6 1 z`}
            fill="#3f8a4d"
          />
        </g>

        <rect
          data-testid="map-move-hit"
          x={value.mapXMm}
          y={value.mapYMm}
          width={value.mapWidthMm}
          height={value.mapHeightMm}
          rx={Math.min(value.cornerRadiusMm, 3)}
          fill="transparent"
          stroke="#e9f5ff"
          strokeDasharray="1.8 1.4"
          strokeWidth={0.45}
          className="cursor-move"
          onPointerDown={(event) => beginDrag(event, "map-move")}
        />

        <text
          data-testid="label-move-hit"
          x={value.labelXMm}
          y={value.labelYMm}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="#f1f5f9"
          stroke="rgba(248,250,252,0.58)"
          strokeWidth={Math.max(value.labelStrokeMm * 0.08, 0.04)}
          paintOrder="stroke"
          fontSize={Math.max(value.labelTextHeightMm * 0.72, 2.6)}
          fontWeight={700}
          letterSpacing={0.35}
          transform={`rotate(${value.labelAngleDeg} ${value.labelXMm} ${value.labelYMm})`}
          className="cursor-move"
          onPointerDown={(event) => beginDrag(event, "label")}
        >
          {label || "TEXT"}
        </text>

        <g fill="none" stroke="rgba(255,255,255,0.66)" strokeWidth={0.28}>
          <path d={`M 0 ${-4.2} H ${value.bodyWidthMm}`} />
          <path d={`M ${value.bodyWidthMm + 4.2} 0 V ${value.bodyHeightMm}`} />
          <path d={`M ${value.mapXMm} ${value.mapYMm - 2.8} H ${value.mapXMm + value.mapWidthMm}`} stroke="rgba(45,212,191,0.9)" />
        </g>
        <g fill="#f8fafc" fontSize={2.2} fontWeight={700}>
          <text x={value.bodyWidthMm / 2} y={-5.6} textAnchor="middle">
            {value.bodyWidthMm.toFixed(0)} mm
          </text>
          <text x={value.bodyWidthMm + 7.2} y={value.bodyHeightMm / 2} textAnchor="middle" transform={`rotate(90 ${value.bodyWidthMm + 7.2} ${value.bodyHeightMm / 2})`}>
            {value.bodyHeightMm.toFixed(0)} mm
          </text>
          <text x={value.mapXMm + value.mapWidthMm / 2} y={value.mapYMm - 4.2} textAnchor="middle" fill="#5eead4">
            map {value.mapWidthMm.toFixed(0)} x {value.mapHeightMm.toFixed(0)}
          </text>
          <text x={value.loopXMm} y={value.loopYMm - value.loopOuterMm - 2.2} textAnchor="middle">
            O {value.loopOuterMm.toFixed(1)} / hole {value.loopInnerMm.toFixed(1)}
          </text>
        </g>

        <circle
          data-testid="loop-move-hit"
          cx={value.loopXMm}
          cy={value.loopYMm}
          r={value.loopOuterMm + 1.3}
          fill="transparent"
          stroke="#f8fafc"
          strokeWidth={0.55}
          className="cursor-move"
          onPointerDown={(event) => beginDrag(event, "loop")}
        />
        <rect
          data-testid="map-resize-handle"
          x={value.mapXMm + value.mapWidthMm - 5}
          y={value.mapYMm + value.mapHeightMm - 5}
          width={10}
          height={10}
          rx={1.7}
          fill="#f8fafc"
          stroke="#14b8a6"
          strokeWidth={0.45}
          className="cursor-nwse-resize"
          onPointerDown={(event) => beginDrag(event, "map-resize")}
        />
        <rect
          data-testid="body-resize-handle"
          x={value.bodyWidthMm - 3.8}
          y={value.bodyHeightMm - 3.8}
          width={7.6}
          height={7.6}
          rx={1.1}
          fill="#14b8a6"
          className="cursor-nwse-resize"
          onPointerDown={(event) => beginDrag(event, "body")}
        />
      </svg>
    </div>
  );
}
