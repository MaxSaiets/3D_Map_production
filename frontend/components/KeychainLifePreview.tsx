"use client";

import type { KeychainDesignerConfig } from "@/components/KeychainDesigner";

function bodyRadius(value: KeychainDesignerConfig) {
  if (value.baseShape === "capsule" || value.baseShape === "token") return value.bodyHeightMm / 2;
  if (value.baseShape === "octagon") return 0;
  return Math.min(value.cornerRadiusMm, Math.min(value.bodyWidthMm, value.bodyHeightMm) / 2);
}

export function KeychainLifePreview({
  design,
  label,
}: {
  design: KeychainDesignerConfig;
  label: string;
}) {
  const aspect = design.bodyWidthMm / Math.max(design.bodyHeightMm, 1);
  const plateWidth = aspect >= 1 ? 250 : Math.max(156, 250 * aspect);
  const plateHeight = plateWidth / Math.max(aspect, 0.35);
  const mapLeft = (design.mapXMm / design.bodyWidthMm) * plateWidth;
  const mapTop = (design.mapYMm / design.bodyHeightMm) * plateHeight;
  const mapWidth = (design.mapWidthMm / design.bodyWidthMm) * plateWidth;
  const mapHeight = (design.mapHeightMm / design.bodyHeightMm) * plateHeight;
  const labelLeft = (design.labelXMm / design.bodyWidthMm) * plateWidth;
  const labelTop = (design.labelYMm / design.bodyHeightMm) * plateHeight;
  const loopLeft = (design.loopXMm / design.bodyWidthMm) * plateWidth;
  const loopTop = (design.loopYMm / design.bodyHeightMm) * plateHeight;
  const loopOuter = (design.loopOuterMm / design.bodyWidthMm) * plateWidth;
  const loopInner = (design.loopInnerMm / design.bodyWidthMm) * plateWidth;
  const radius = (bodyRadius(design) / design.bodyWidthMm) * plateWidth;

  const sceneTilt = aspect >= 1 ? -13 : -10;
  const plateSceneX = aspect >= 1 ? 84 : 132;
  const plateSceneY = aspect >= 1 ? 138 : 82;

  return (
    <div className="relative h-full min-h-[360px] overflow-hidden rounded-[22px] bg-[#d8c6a5]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_8%_6%,rgba(255,255,255,0.42),transparent_25%),linear-gradient(135deg,#eadfc8,#b89768_72%)]" />
      <svg viewBox="40 50 300 330" preserveAspectRatio="xMidYMid meet" className="absolute inset-0 h-full w-full">
        <defs>
          <filter id="lifeShadow" x="-30%" y="-30%" width="160%" height="170%">
            <feDropShadow dx="0" dy="20" stdDeviation="13" floodColor="#26180b" floodOpacity="0.36" />
          </filter>
          <filter id="softBlur" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" />
          </filter>
          <linearGradient id="ringMetal" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#f7f7f2" />
            <stop offset="0.42" stopColor="#9ca3af" />
            <stop offset="1" stopColor="#ffffff" />
          </linearGradient>
          <clipPath id="lifeMapClip">
            <rect x={mapLeft} y={mapTop} width={mapWidth} height={mapHeight} rx="5" />
          </clipPath>
          <linearGradient id="plateSide" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#7f6c48" />
            <stop offset="1" stopColor="#463a28" />
          </linearGradient>
        </defs>

        <ellipse cx="250" cy="345" rx="138" ry="34" fill="#3d2d1b" opacity="0.2" filter="url(#softBlur)" />
        <g transform="translate(62 250) rotate(-21)">
          <ellipse cx="0" cy="0" rx="42" ry="24" fill="none" stroke="url(#ringMetal)" strokeWidth="8" />
          {Array.from({ length: 5 }).map((_, index) => (
            <ellipse
              key={`chain-${index}`}
              cx={38 + index * 24}
              cy={12 + index * 4}
              rx="17"
              ry="9"
              fill="none"
              stroke="url(#ringMetal)"
              strokeWidth="7"
              transform={`rotate(${index % 2 ? 68 : -18} ${38 + index * 24} ${12 + index * 4})`}
            />
          ))}
          <path d="M -38 14 C -62 43 -80 72 -89 104" fill="none" stroke="url(#ringMetal)" strokeWidth="8" strokeLinecap="round" />
          <path d="M -96 105 h46 l 10 24 l -27 12 l -3 24 h-18 l -8 -24 h-22 l 7 -24 z" fill="#bec3c8" stroke="#f8fafc" strokeWidth="2" />
        </g>

        <g
          filter="url(#lifeShadow)"
          transform={`translate(${plateSceneX} ${plateSceneY}) rotate(${Number(design.layoutRotationDeg || 0) + sceneTilt} ${plateWidth / 2} ${plateHeight / 2}) skewX(-4)`}
        >
          <rect x="8" y={plateHeight - 3} width={Math.max(plateWidth - 16, 1)} height="14" rx="7" fill="url(#plateSide)" opacity="0.75" />
          {design.baseShape === "octagon" ? (
            <path
              d={`M 18 0 H ${plateWidth - 18} L ${plateWidth} 18 V ${plateHeight - 18} L ${plateWidth - 18} ${plateHeight} H 18 L 0 ${plateHeight - 18} V 18 Z`}
              fill="#a6926b"
              stroke="#d8ccb1"
              strokeWidth="2"
            />
          ) : (
            <rect width={plateWidth} height={plateHeight} rx={radius} fill="#a6926b" stroke="#d8ccb1" strokeWidth="2" />
          )}
          {design.rimWidthMm > 0 ? (
            <rect
              x="4"
              y="4"
              width={Math.max(plateWidth - 8, 1)}
              height={Math.max(plateHeight - 8, 1)}
              rx={Math.max(radius - 4, 0)}
              fill="none"
              stroke="#6d5c3f"
              strokeOpacity="0.45"
              strokeWidth={Math.max(2, design.rimWidthMm * 1.2)}
            />
          ) : null}
          {design.baseShape === "token" ? (
            <g transform={`translate(${loopLeft} ${loopTop})`}>
              <circle r={Math.max(loopInner + 6, loopOuter)} fill="none" stroke="#f8fafc" strokeOpacity="0.58" strokeWidth="2" />
              <circle r={loopInner} fill="#101725" stroke="#f8fafc" strokeOpacity="0.5" strokeWidth="1.4" />
              <path d={`M ${-Math.max(loopInner + 9, loopOuter + 3)} 0 H ${Math.max(loopInner + 9, loopOuter + 3)}`} stroke="#f8fafc" strokeOpacity="0.36" strokeWidth="1" />
              <path d={`M 0 ${-Math.max(loopInner + 9, loopOuter + 3)} V ${Math.max(loopInner + 9, loopOuter + 3)}`} stroke="#f8fafc" strokeOpacity="0.36" strokeWidth="1" />
            </g>
          ) : (
            <g transform={`translate(${loopLeft} ${loopTop}) rotate(${design.loopAngleDeg || 0})`}>
              <circle r={loopOuter} fill="#a6926b" stroke="#d8ccb1" strokeWidth="2" />
              <circle r={loopInner} fill="#101725" stroke="#f8fafc" strokeOpacity="0.4" strokeWidth="1" />
            </g>
          )}
          <g clipPath="url(#lifeMapClip)" transform={`rotate(${design.mapRotationDeg || 0} ${mapLeft + mapWidth / 2} ${mapTop + mapHeight / 2})`}>
            <rect x={mapLeft} y={mapTop} width={mapWidth} height={mapHeight} fill="#b7ab8e" />
            {Array.from({ length: 9 }).map((_, index) => (
              <path
                key={`life-road-${index}`}
                d={`M ${mapLeft + index * (mapWidth / 8)} ${mapTop - 10} L ${mapLeft + 8 + index * (mapWidth / 9)} ${mapTop + mapHeight + 12}`}
                stroke="#101010"
                strokeWidth="3"
                strokeLinecap="round"
              />
            ))}
            <path d={`M ${mapLeft + mapWidth * 0.62} ${mapTop} C ${mapLeft + mapWidth * 0.82} ${mapTop + mapHeight * 0.24}, ${mapLeft + mapWidth * 0.45} ${mapTop + mapHeight * 0.64}, ${mapLeft + mapWidth * 0.8} ${mapTop + mapHeight}`} fill="none" stroke="#74a9d8" strokeWidth="6" />
            <rect x={mapLeft + mapWidth * 0.18} y={mapTop + mapHeight * 0.2} width={mapWidth * 0.16} height={mapHeight * 0.12} fill="#d7d7d2" />
            <rect x={mapLeft + mapWidth * 0.52} y={mapTop + mapHeight * 0.56} width={mapWidth * 0.18} height={mapHeight * 0.12} fill="#d7d7d2" />
            <path d={`M ${mapLeft + mapWidth * 0.72} ${mapTop + mapHeight * 0.45} l 18 4 l -6 16 l -20 1 z`} fill="#32884f" />
          </g>
          <text
            x={labelLeft}
            y={labelTop}
            textAnchor="middle"
            dominantBaseline="middle"
            fill="#f8fafc"
            stroke="#f8fafc"
            strokeWidth={design.labelFontStyle === "wide" ? 1.1 : design.labelFontStyle === "condensed" ? 0.45 : 0.75}
            paintOrder="stroke"
            fontFamily={design.labelFontStyle === "wide" ? "Arial Black, Impact, sans-serif" : design.labelFontStyle === "condensed" ? "Arial Narrow, Bahnschrift, sans-serif" : "monospace"}
            fontSize={Math.max(10, design.labelTextHeightMm * 2.7)}
            fontWeight="800"
            transform={`rotate(${design.labelAngleDeg || 0} ${labelLeft} ${labelTop})`}
          >
            {(label || "CITY").slice(0, 28)}
          </text>
        </g>
      </svg>
      <div className="absolute left-3 top-3 rounded-full border border-black/10 bg-white/70 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-[#5b4a32] backdrop-blur">
        Real-life preview
      </div>
    </div>
  );
}

export function KeychainSlicerPreview({
  design,
  label,
}: {
  design: KeychainDesignerConfig;
  label: string;
}) {
  const layers = [
    { name: "Base", color: "#a6926b", height: "2.0 mm", width: "100%" },
    { name: "Rim", color: "#ffffff", height: `${design.rimHeightMm.toFixed(2)} mm`, width: "96%" },
    { name: "Water", color: "#6fa8dc", height: "0.28 mm", width: "54%" },
    { name: "Parks", color: "#2f8b4b", height: "0.34 mm", width: "62%" },
    { name: "Roads", color: "#111111", height: "0.44 mm", width: "82%" },
    { name: "Buildings", color: "#d8d8d8", height: "0.8-2.2 mm", width: "68%" },
    { name: "Text", color: "#f8fafc", height: `${design.labelTextHeightMm.toFixed(1)} mm`, width: `${Math.max(36, Math.min(94, (label.length || 8) * 5))}%` },
  ];

  return (
    <div className="relative h-full min-h-[360px] overflow-hidden rounded-[22px] bg-[#151923] p-4 text-white">
      <div className="absolute inset-0 bg-[linear-gradient(135deg,rgba(255,255,255,0.06),transparent_38%),radial-gradient(circle_at_20%_12%,rgba(94,234,212,0.16),transparent_26%)]" />
      <div className="relative z-10">
        <div className="rounded-full border border-white/10 bg-white/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-white/70">
          Slicer preview
        </div>
        <h3 className="mt-4 font-title text-xl font-semibold">Порядок друку шарів</h3>
        <p className="mt-2 text-sm leading-6 text-white/68">
          Карта має обрізатись всередині краю, текст і край лишаються окремими шарами для чистого кольору.
        </p>
      </div>
      <div className="relative z-10 mt-5 space-y-3">
        {layers.map((layer, index) => (
          <div key={layer.name} className="grid grid-cols-[76px,1fr,72px] items-center gap-3">
            <div className="text-xs font-semibold text-white/75">{index + 1}. {layer.name}</div>
            <div className="h-7 overflow-hidden rounded-full border border-white/10 bg-black/20 p-1">
              <div
                className="h-full rounded-full shadow-[inset_0_1px_0_rgba(255,255,255,0.3)]"
                style={{ width: layer.width, background: layer.color }}
              />
            </div>
            <div className="text-right text-[11px] font-semibold text-white/62">{layer.height}</div>
          </div>
        ))}
      </div>
      <div className="relative z-10 mt-6 rounded-[18px] border border-white/10 bg-white/[0.06] p-3">
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-white/50">Print note</div>
        <div className="mt-1 text-sm font-semibold text-white/86">
          Мінімальний штрих: 0.4 мм. Текст краще друкувати як останній колірний шар.
        </div>
      </div>
    </div>
  );
}
