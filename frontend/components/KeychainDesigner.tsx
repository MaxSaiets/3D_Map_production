"use client";

import dynamic from "next/dynamic";
import { useMemo, useRef, useState } from "react";
import { useTranslations } from "next-intl";

// Three.js + Overpass fetch — lazy load to avoid SSR + keep designer bundle light
// SVG-LAYER (нативні шляхи у батьківському SVG). НЕ foreignObject — інакше на
// iOS/Safari прев'ю карти вилазить у лівий верхній кут (WebKit bug #23113).
// loading повертає null бо рендеримось усередині <svg> (div там недопустимий).
const LiveCitySvgPaths = dynamic(
  () => import("@/components/LiveCity3D").then((m) => ({ default: m.LiveCitySvgPaths })),
  { ssr: false, loading: () => null },
);

export type KeychainBaseShape = "rounded" | "capsule" | "tag" | "octagon" | "token" | "heart" | "house" | "puzzle-l" | "puzzle-r" | "heart-l" | "heart-r";
export type KeychainLoopStyle = "round" | "teardrop" | "slot" | "side-tab";
export type KeychainLabelFontStyle = "block" | "wide" | "condensed";

export type KeychainDesignerConfig = {
  bodyWidthMm: number;
  bodyHeightMm: number;
  layoutRotationDeg: number;
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
  mapRotationDeg: number;
  labelXMm: number;
  labelYMm: number;
  labelWidthMm: number;
  labelBandMm: number;
  labelTextHeightMm: number;
  labelStrokeMm: number;
  labelFontStyle: KeychainLabelFontStyle;
  labelAngleDeg: number;
  // Опційно: вирізати прямокутну зону карти під написом. За замовчуванням
  // вимкнено — карта суцільна, підіймаються лише літери.
  labelClearBand?: boolean;
  rimWidthMm: number;
  rimHeightMm: number;
};

export type KeychainTemplate = {
  id: string;
  /** i18n keys (namespace "kc") for the template name/description. */
  nameKey: string;
  descKey: string;
  /** Ukrainian fallback (kept for backward-compat with consumers that don't
   *  resolve via next-intl yet). New render sites use nameKey/descKey + t(). */
  name: string;
  description: string;
  design: KeychainDesignerConfig;
};

type DragTarget = "body" | "map-move" | "map-resize" | "loop" | "label" | "label-rotate";
type DragSession = {
  target: DragTarget;
  start: { x: number; y: number };
  initial: KeychainDesignerConfig;
};

const MIN_MAP_WIDTH_MM = 28;
const MIN_MAP_HEIGHT_MM = 18;
const MIN_TOKEN_MAP_WIDTH_MM = 18;
const MIN_TOKEN_MAP_HEIGHT_MM = 8;
const SNAP_MM = 1.15;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function snapTo(value: number, target: number, threshold = SNAP_MM) {
  return Math.abs(value - target) <= threshold ? target : value;
}

function clampLoopToBody(next: KeychainDesignerConfig) {
  if (next.baseShape === "token") {
    const wallMm = Math.max(next.loopInnerMm + 2.0, 3.5);
    next.loopXMm = clamp(next.loopXMm, wallMm, next.bodyWidthMm - wallMm);
    next.loopYMm = clamp(next.loopYMm, wallMm, next.bodyHeightMm - wallMm);
    next.loopXMm = snapTo(next.loopXMm, 4.5, 1.4);
    next.loopYMm = snapTo(next.loopYMm, next.bodyHeightMm / 2);
    return next;
  }

  const attachOffset = Math.max(next.loopOuterMm * 0.58, 3.2);
  const margin = Math.max(attachOffset, 3.8);
  next.loopXMm = clamp(next.loopXMm, -margin, next.bodyWidthMm + margin);
  next.loopYMm = clamp(next.loopYMm, -margin, next.bodyHeightMm + margin);
  next.loopXMm = snapTo(next.loopXMm, next.bodyWidthMm / 2);
  next.loopYMm = snapTo(next.loopYMm, next.bodyHeightMm / 2);
  next.loopYMm = snapTo(next.loopYMm, -attachOffset);
  next.loopYMm = snapTo(next.loopYMm, next.bodyHeightMm + attachOffset);
  next.loopXMm = snapTo(next.loopXMm, -attachOffset);
  next.loopXMm = snapTo(next.loopXMm, next.bodyWidthMm + attachOffset);
  return next;
}

function fitAfterBodyResize(next: KeychainDesignerConfig) {
  const minMapWidthBaseMm = next.baseShape === "token" ? MIN_TOKEN_MAP_WIDTH_MM : MIN_MAP_WIDTH_MM;
  const minMapHeightBaseMm = next.baseShape === "token" ? MIN_TOKEN_MAP_HEIGHT_MM : MIN_MAP_HEIGHT_MM;
  const minMapWidthMm = Math.min(minMapWidthBaseMm, next.bodyWidthMm);
  const minMapHeightMm = Math.min(minMapHeightBaseMm, next.bodyHeightMm);
  next.layoutRotationDeg = ((Math.round((next.layoutRotationDeg || 0) / 90) * 90) % 360 + 360) % 360;
  next.mapXMm = clamp(next.mapXMm, 0, Math.max(0, next.bodyWidthMm - minMapWidthMm));
  next.mapYMm = clamp(next.mapYMm, 0, Math.max(0, next.bodyHeightMm - minMapHeightMm));
  next.mapWidthMm = clamp(next.mapWidthMm, minMapWidthMm, Math.max(minMapWidthMm, next.bodyWidthMm - next.mapXMm));
  next.mapHeightMm = clamp(next.mapHeightMm, minMapHeightMm, Math.max(minMapHeightMm, next.bodyHeightMm - next.mapYMm));
  next.mapRotationDeg = ((Math.round((next.mapRotationDeg || 0) / 15) * 15) % 360 + 360) % 360;
  next.labelXMm = clamp(next.labelXMm, 4, Math.max(4, next.bodyWidthMm - 4));
  next.labelYMm = clamp(next.labelYMm, 4, Math.max(4, next.bodyHeightMm - 4));
  next.labelWidthMm = clamp(next.labelWidthMm, 6, next.bodyWidthMm);
  // Print-safe мінімуми (FDM 0.4мм сопло): висота літер ≥2мм, штрих ≥0.8мм (2× сопла).
  next.labelTextHeightMm = clamp(next.labelTextHeightMm, 2.0, 8.5);
  next.labelStrokeMm = clamp(next.labelStrokeMm, 0.8, 2.0);
  // Смуга напису завжди ≥ висота літер + 1.4мм → бекенд не зменшує текст (WYSIWYG).
  next.labelBandMm = clamp(Math.max(next.labelBandMm, next.labelTextHeightMm + 1.4), 3, 18);
  clampLoopToBody(next);
  return next;
}

export const DEFAULT_KEYCHAIN_DESIGN: KeychainDesignerConfig = {
  bodyWidthMm: 35,
  bodyHeightMm: 55,
  layoutRotationDeg: 0,
  cornerRadiusMm: 4.2,
  baseShape: "rounded",
  loopStyle: "round",
  loopAngleDeg: 0,
  loopXMm: 17.5,
  // loopYMm = 0: центр петлі ТОЧНО на верхній грані тіла → 50% кільця стирчить
  // назовні, 50% у тілі (стандартний вигляд вушка, юзер: «має виглядати на 50%»).
  loopYMm: 0,
  loopOuterMm: 4.0,   // зовнішній радіус петлі (стандарт)
  loopInnerMm: 2.0,   // радіус отвору під кільце (стандарт)
  mapXMm: 0,
  mapYMm: 0,
  mapWidthMm: 35,
  mapHeightMm: 55,
  mapRotationDeg: 0,
  labelXMm: 17.5,
  labelYMm: 51.0,
  labelWidthMm: 30,
  labelBandMm: 5.0,   // мінімальна смуга (висота літер + 1.2мм padding)
  labelTextHeightMm: 3.2,
  labelStrokeMm: 0.9,  // print-safe штрих ≥0.8мм (2× сопла 0.4мм)
  labelFontStyle: "block",
  labelAngleDeg: 0,
  rimWidthMm: 1.2,
  rimHeightMm: 0.45,
};

export const KEYCHAIN_TEMPLATES: KeychainTemplate[] = [
  {
    id: "classic-wide",
    nameKey: "tpl.classicWide.name",
    descKey: "tpl.classicWide.desc",
    name: "35 x 55",
    description: "Стандартний компактний вертикальний брелок.",
    design: DEFAULT_KEYCHAIN_DESIGN,
  },
  {
    id: "token-55",
    nameKey: "tpl.token55.name",
    descKey: "tpl.token55.desc",
    name: "Token 55 x 30",
    description: "Стандартний жетон 55×30 з лівим отвором Ø3 мм і капсульною основою.",
    design: {
      ...DEFAULT_KEYCHAIN_DESIGN,
      bodyWidthMm: 55,
      bodyHeightMm: 30,
      cornerRadiusMm: 15,
      baseShape: "token",
      loopStyle: "round",
      // Петля по центру зверху і ВТОПЛЕНА в тіло (юзер: «посередину»).
      loopXMm: 27.5,
      loopYMm: 5.0,
      loopOuterMm: 2.8,
      loopInnerMm: 1.5,
      mapXMm: 0,
      mapYMm: 0,
      mapWidthMm: 55,
      mapHeightMm: 30,
      mapRotationDeg: 0,
      labelXMm: 32,
      labelYMm: 25.2,
      labelWidthMm: 34,
      labelBandMm: 6,
      labelTextHeightMm: 3.2,
      labelStrokeMm: 0.9,
      rimWidthMm: 0.9,
      rimHeightMm: 0.35,
    },
  },
  {
    id: "heart-46",
    nameKey: "tpl.heart46.name",
    descKey: "tpl.heart46.desc",
    name: "Серце 46 × 42",
    description: "Мапа місця, що в серці — подарунок для двох.",
    design: {
      ...DEFAULT_KEYCHAIN_DESIGN,
      bodyWidthMm: 46,
      bodyHeightMm: 42,
      cornerRadiusMm: 0,
      baseShape: "heart",
      loopStyle: "round",
      loopXMm: 23,
      loopYMm: 1.5,
      loopOuterMm: 4,
      loopInnerMm: 2,
      mapXMm: 0,
      mapYMm: 0,
      mapWidthMm: 46,
      mapHeightMm: 42,
      labelXMm: 23,
      labelYMm: 27,
      labelWidthMm: 22,
      labelBandMm: 5,
      labelTextHeightMm: 3.0,
    },
  },
  {
    id: "house-44",
    nameKey: "tpl.house44.name",
    descKey: "tpl.house44.desc",
    name: "Будиночок 44 × 48",
    description: "Дім — там, де твоя вулиця. Дах з вушком зверху.",
    design: {
      ...DEFAULT_KEYCHAIN_DESIGN,
      bodyWidthMm: 44,
      bodyHeightMm: 48,
      cornerRadiusMm: 0,
      baseShape: "house",
      loopStyle: "round",
      loopXMm: 22,
      loopYMm: 1,
      loopOuterMm: 4,
      loopInnerMm: 2,
      mapXMm: 0,
      mapYMm: 0,
      mapWidthMm: 44,
      mapHeightMm: 48,
      labelXMm: 22,
      labelYMm: 43.5,
      labelWidthMm: 30,
      labelBandMm: 5,
      labelTextHeightMm: 3.2,
    },
  },
  {
    id: "puzzle-left",
    nameKey: "tpl.puzzleLeft.name",
    descKey: "tpl.puzzleLeft.desc",
    name: "Пазл L · 40 × 42",
    description: "Половинка пари: твоє місто. Виступ праворуч зʼєднується з половинкою R.",
    design: {
      ...DEFAULT_KEYCHAIN_DESIGN,
      bodyWidthMm: 40,
      bodyHeightMm: 42,
      cornerRadiusMm: 5,
      baseShape: "puzzle-l",
      loopStyle: "round",
      loopXMm: 20,
      loopYMm: 0,
      mapXMm: 0,
      mapYMm: 0,
      mapWidthMm: 40,
      mapHeightMm: 42,
      labelXMm: 20,
      labelYMm: 37.5,
      labelWidthMm: 30,
      labelBandMm: 6,
      labelTextHeightMm: 3.2,
    },
  },
  {
    id: "puzzle-right",
    nameKey: "tpl.puzzleRight.name",
    descKey: "tpl.puzzleRight.desc",
    name: "Пазл R · 40 × 42",
    description: "Половинка пари: місто близької людини. Паз ліворуч приймає половинку L.",
    design: {
      ...DEFAULT_KEYCHAIN_DESIGN,
      bodyWidthMm: 40,
      bodyHeightMm: 42,
      cornerRadiusMm: 5,
      baseShape: "puzzle-r",
      loopStyle: "round",
      loopXMm: 20,
      loopYMm: 0,
      mapXMm: 0,
      mapYMm: 0,
      mapWidthMm: 40,
      mapHeightMm: 42,
      labelXMm: 20,
      labelYMm: 37.5,
      labelWidthMm: 30,
      labelBandMm: 6,
      labelTextHeightMm: 3.2,
    },
  },
  {
    id: "heart-pair-left",
    nameKey: "tpl.heartPairLeft.name",
    descKey: "tpl.heartPairLeft.desc",
    name: "Серце пари · L · 30 × 44",
    description: "Половинка серця для двох: твоє місто. Замок на грані зʼєднується з половинкою R у повне серце.",
    design: {
      ...DEFAULT_KEYCHAIN_DESIGN,
      bodyWidthMm: 30,
      bodyHeightMm: 44,
      cornerRadiusMm: 0,
      baseShape: "heart-l",
      loopStyle: "round",
      loopXMm: 15,
      loopYMm: 0,
      mapXMm: 0,
      mapYMm: 0,
      mapWidthMm: 30,
      mapHeightMm: 44,
      labelXMm: 15,
      labelYMm: 33,
      labelWidthMm: 16,
      labelBandMm: 5,
      labelTextHeightMm: 2.8,
    },
  },
  {
    id: "heart-pair-right",
    nameKey: "tpl.heartPairRight.name",
    descKey: "tpl.heartPairRight.desc",
    name: "Серце пари · R · 30 × 44",
    description: "Половинка серця для двох: місто близької людини. Паз приймає половинку L.",
    design: {
      ...DEFAULT_KEYCHAIN_DESIGN,
      bodyWidthMm: 30,
      bodyHeightMm: 44,
      cornerRadiusMm: 0,
      baseShape: "heart-r",
      loopStyle: "round",
      loopXMm: 15,
      loopYMm: 0,
      mapXMm: 0,
      mapYMm: 0,
      mapWidthMm: 30,
      mapHeightMm: 44,
      labelXMm: 15,
      labelYMm: 33,
      labelWidthMm: 16,
      labelBandMm: 5,
      labelTextHeightMm: 2.8,
    },
  },
  {
    id: "right-loop",
    nameKey: "tpl.rightLoop.name",
    descKey: "tpl.rightLoop.desc",
    name: "Side Loop",
    description: "Петля справа, зручно для широкої карти.",
    design: {
      ...DEFAULT_KEYCHAIN_DESIGN,
      bodyWidthMm: 55,
      bodyHeightMm: 35,
      loopXMm: 60.5,
      loopYMm: 17.5,
      loopAngleDeg: 270,
      mapXMm: 0,
      mapYMm: 0,
      mapWidthMm: 55,
      mapHeightMm: 35,
      labelXMm: 27.5,
      labelYMm: 30.5,
      labelWidthMm: 45,
    },
  },
  {
    id: "vertical-tag",
    nameKey: "tpl.verticalTag.name",
    descKey: "tpl.verticalTag.desc",
    name: "Vertical",
    description: "Вертикальний брелок з повернутим написом.",
    design: {
      ...DEFAULT_KEYCHAIN_DESIGN,
      bodyWidthMm: 35,
      bodyHeightMm: 55,
      baseShape: "tag",
      loopXMm: 17.5,
      loopYMm: -4,
      mapXMm: 0,
      mapYMm: 0,
      mapWidthMm: 35,
      mapHeightMm: 55,
      labelXMm: 17.5,
      labelYMm: 49,
      labelWidthMm: 28,
      labelAngleDeg: 0,
    },
  },
  {
    id: "soft-capsule",
    nameKey: "tpl.softCapsule.name",
    descKey: "tpl.softCapsule.desc",
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
      mapXMm: 0,
      mapYMm: 0,
      mapWidthMm: 55,
      mapHeightMm: 35,
      labelXMm: 27.5,
      labelYMm: 30,
      labelWidthMm: 44,
    },
  },
];

export function shapePath(
  value: KeychainDesignerConfig,
  {
    x = 0,
    y = 0,
    width = value.bodyWidthMm,
    height = value.bodyHeightMm,
    radius = value.cornerRadiusMm,
  }: { x?: number; y?: number; width?: number; height?: number; radius?: number } = {},
) {
  const w = Math.max(width, 0.1);
  const h = Math.max(height, 0.1);
  const r = clamp(radius, 0, Math.min(w, h) / 2);
  const minX = x;
  const minY = y;
  const maxX = x + w;
  const maxY = y + h;
  if (value.baseShape === "capsule" || value.baseShape === "token") {
    const cr = h / 2;
    return `M ${minX + cr} ${minY} H ${maxX - cr} A ${cr} ${cr} 0 0 1 ${maxX - cr} ${maxY} H ${minX + cr} A ${cr} ${cr} 0 0 1 ${minX + cr} ${minY} Z`;
  }
  if (value.baseShape === "tag") {
    const cut = Math.min(w, h) * 0.16;
    return `M ${minX} ${minY + r} Q ${minX} ${minY} ${minX + r} ${minY} H ${maxX - cut} L ${maxX} ${minY + cut} V ${maxY - r} Q ${maxX} ${maxY} ${maxX - r} ${maxY} H ${minX + r} Q ${minX} ${maxY} ${minX} ${maxY - r} V ${minY + r} Z`;
  }
  if (value.baseShape === "octagon") {
    const cut = Math.min(w, h) * 0.13;
    return `M ${minX + cut} ${minY} H ${maxX - cut} L ${maxX} ${minY + cut} V ${maxY - cut} L ${maxX - cut} ${maxY} H ${minX + cut} L ${minX} ${maxY - cut} V ${minY + cut} Z`;
  }
  if (value.baseShape === "heart") {
    // Контур серця (спільне джерело з містком-ущелиною LoopPreview).
    const pts2 = heartShapePoints(minX, minY, w, h);
    const d = pts2.map(([sx, sy], i) => `${i === 0 ? "M" : "L"} ${sx.toFixed(2)} ${sy.toFixed(2)}`).join(" ");
    return `${d} Z`;
  }
  if (value.baseShape === "house") {
    const roofH = h * 0.38;
    const cx = minX + w / 2;
    return `M ${cx} ${minY} L ${maxX} ${minY + roofH} V ${maxY} H ${minX} V ${minY + roofH} Z`;
  }
  if (value.baseShape === "heart-l" || value.baseShape === "heart-r") {
    const pts2 = heartHalfPoints(minX, minY, w, h, value.baseShape === "heart-l" ? "l" : "r");
    return pts2.map(([sx, sy], i) => `${i === 0 ? "M" : "L"} ${sx.toFixed(2)} ${sy.toFixed(2)}`).join(" ") + " Z";
  }
  if (value.baseShape === "puzzle-l" || value.baseShape === "puzzle-r") {
    // Та сама геометрія, що на беку (_keychain_body_shape puzzle-l/r):
    // knob k=0.13·min(w,h), шийка nw=0.62k, центр головки на 0.95k від грані.
    // Вертикально центрована → y-фліп превʼю не впливає.
    const k = Math.min(w, h) * 0.13;
    const nw = k * 0.62;
    const cy = minY + h / 2;
    const xInt = Math.sqrt(k * k - nw * nw); // зміщення точки входу шийки в коло
    if (value.baseShape === "puzzle-l") {
      const cx0 = maxX + 0.95 * k;
      const xi = cx0 - xInt;
      return (
        `M ${minX + r} ${minY} H ${maxX - r} Q ${maxX} ${minY} ${maxX} ${minY + r} ` +
        `V ${cy - nw} L ${xi.toFixed(2)} ${cy - nw} A ${k} ${k} 0 1 1 ${xi.toFixed(2)} ${cy + nw} L ${maxX} ${cy + nw} ` +
        `V ${maxY - r} Q ${maxX} ${maxY} ${maxX - r} ${maxY} H ${minX + r} Q ${minX} ${maxY} ${minX} ${maxY - r} V ${minY + r} Q ${minX} ${minY} ${minX + r} ${minY} Z`
      );
    }
    const cxn = minX + 0.95 * k;
    const xi = cxn - xInt;
    return (
      `M ${minX + r} ${minY} H ${maxX - r} Q ${maxX} ${minY} ${maxX} ${minY + r} V ${maxY - r} ` +
      `Q ${maxX} ${maxY} ${maxX - r} ${maxY} H ${minX + r} Q ${minX} ${maxY} ${minX} ${maxY - r} ` +
      `V ${cy + nw} L ${xi.toFixed(2)} ${cy + nw} A ${k} ${k} 0 1 0 ${xi.toFixed(2)} ${cy - nw} L ${minX} ${cy - nw} ` +
      `V ${minY + r} Q ${minX} ${minY} ${minX + r} ${minY} Z`
    );
  }
  return `M ${minX + r} ${minY} H ${maxX - r} Q ${maxX} ${minY} ${maxX} ${minY + r} V ${maxY - r} Q ${maxX} ${maxY} ${maxX - r} ${maxY} H ${minX + r} Q ${minX} ${maxY} ${minX} ${maxY - r} V ${minY + r} Q ${minX} ${minY} ${minX + r} ${minY} Z`;
}

/** Заокруглення одного гострого вузла контуру (вістря серця): вершини в
 *  радіусі radius від кінчика → семпли квадратичної Безьє через старий кінчик.
 *  Дзеркало бекендового _round_polygon_tip — превʼю і модель збігаються. */
function roundPolygonTip(pts: Array<[number, number]>, tipIndex: number, radius: number): Array<[number, number]> {
  const n = pts.length;
  if (n < 8 || radius <= 0) return pts;
  const tip = pts[tipIndex];
  const walk = (dir: number) => {
    let dist = 0;
    let i = tipIndex;
    let prev = tip;
    for (let s = 0; s < Math.floor(n / 2); s++) {
      i = (i + dir + n) % n;
      const cur = pts[i];
      dist += Math.hypot(cur[0] - prev[0], cur[1] - prev[1]);
      prev = cur;
      if (dist >= radius) return i;
    }
    return (tipIndex + dir + n) % n;
  };
  // СИМЕТРИЧНО: однакова кількість вузлів з обох боків (інакше асиметрична
  // зазубрина внизу серця). Дзеркало бекендового _round_polygon_tip.
  const ka = (tipIndex - walk(-1) + n) % n;
  const kb = (walk(+1) - tipIndex + n) % n;
  const k = Math.max(1, Math.min(ka, kb));
  const ia = (tipIndex - k + n) % n;
  const ib = (tipIndex + k) % n;
  const a = pts[ia], b = pts[ib];
  const arc: Array<[number, number]> = [];
  const S = 16;
  for (let s = 0; s <= S; s++) {
    const t = s / S;
    arc.push([
      (1 - t) ** 2 * a[0] + 2 * (1 - t) * t * tip[0] + t ** 2 * b[0],
      (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * tip[1] + t ** 2 * b[1],
    ]);
  }
  const out: Array<[number, number]> = [];
  let i = ib;
  while (i !== ia) {
    out.push(pts[i]);
    i = (i + 1) % n;
  }
  out.push(pts[ia]);
  out.push(...arc);
  return out;
}

/** Точки контуру серця (SVG y-вниз: лоби зверху=мала y, вістря знизу) із
 *  заокругленим вістрям — дзеркало бекендового _keychain_body_shape "heart".
 *  Спільне джерело для shapePath і містка-ущелини LoopPreview (превʼю = модель). */
function heartShapePoints(minX: number, minY: number, w: number, h: number): Array<[number, number]> {
  const n = 160; // вища роздільність → гладкий низ (синхрон з беком)
  const raw: Array<[number, number]> = [];
  for (let i = 0; i < n; i++) {
    const t = (2 * Math.PI * i) / n;
    const hx = 16 * Math.sin(t) ** 3;
    const hy = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t);
    raw.push([hx, hy]);
  }
  const xs = raw.map((p) => p[0]);
  const ys = raw.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  let pts2 = raw.map(([px, py]) => [
    minX + ((px - x0) / (x1 - x0)) * w,
    minY + ((y1 - py) / (y1 - y0)) * h, // y-фліп: лоби (py=max) зверху
  ] as [number, number]);
  // Вістря (max sy, бо y-вниз) заокруглюємо — _round_polygon_tip на беку.
  let tipIdx = 0;
  for (let i = 1; i < pts2.length; i++) if (pts2[i][1] > pts2[tipIdx][1]) tipIdx = i;
  pts2 = roundPolygonTip(pts2, tipIdx, Math.min(w, h) * 0.11);
  return pts2;
}

/** Опукла оболонка (monotone chain), CCW. Для містка-ущелини серця у превʼю. */
function convexHull(points: Array<[number, number]>): Array<[number, number]> {
  const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length < 3) return pts;
  const cross = (o: [number, number], a: [number, number], b: [number, number]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Array<[number, number]> = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper: Array<[number, number]> = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

/** Місток-ущелина серця (SVG): заповнює V-розколину між лобами під петлею ТАК САМО,
 *  як бекенд (опукла оболонка петля∪зріз-тіла у вузькій смузі довкола осі петлі) —
 *  кругла петля над відкритою розколиною читалась як булавка-маркер (📍). Малюється
 *  ПІД тілом тим самим кольором → крізь розколину видно заповнення. Превʼю = модель. */
function heartLoopGussetPath(value: KeychainDesignerConfig): string | null {
  if (value.baseShape !== "heart") return null;
  const w = Math.max(value.bodyWidthMm, 0.1);
  const h = Math.max(value.bodyHeightMm, 0.1);
  const pts = heartShapePoints(0, 0, w, h); // тіло малюється у тих самих координатах (x=0,y=0)
  const outer = Math.max(value.loopOuterMm, 1);
  const lx = value.loopXMm, ly = value.loopYMm;
  const fillW = Math.max(outer * 1.4, 2.5);
  const cleftY = 0.239 * h; // дно розколини (hy=5 → 0.239·h від верху, з кривої серця)
  const cand: Array<[number, number]> = [];
  for (const [px, py] of pts) {
    if (Math.abs(px - lx) <= fillW && py <= cleftY + outer * 0.5) cand.push([px, py]);
  }
  if (cand.length < 2) return null;
  for (let i = 0; i < 20; i++) {
    const a = (2 * Math.PI * i) / 20;
    cand.push([lx + outer * Math.cos(a), ly + outer * Math.sin(a)]);
  }
  const hull = convexHull(cand);
  if (hull.length < 3) return null;
  return hull.map(([sx, sy], i) => `${i === 0 ? "M" : "L"} ${sx.toFixed(2)} ${sy.toFixed(2)}`).join(" ") + " Z";
}

/** ПАРА ДЛЯ ЗАКОХАНИХ: контур половинки серця (w×h) із puzzle-замком на грані
 *  розрізу — дзеркало бекендового _keychain_body_shape heart-l/r.
 *  Повне серце будується на подвійній ширині, кліпається по x=cut, прямий
 *  сегмент розрізу замінюється на knob (l, назовні) / notch (r, всередину). */
function heartHalfPoints(minX: number, minY: number, w: number, h: number, side: "l" | "r"): Array<[number, number]> {
  const n = 192;
  const raw: Array<[number, number]> = [];
  for (let i = 0; i < n; i++) {
    const t = (2 * Math.PI * i) / n;
    raw.push([16 * Math.sin(t) ** 3, 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)]);
  }
  const xs = raw.map((p) => p[0]), ys = raw.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const fullMinX = side === "l" ? minX : minX - w; // праву половину зсуваємо в [minX..minX+w]
  let pts = raw.map(([px, py]) => [
    fullMinX + ((px - x0) / (x1 - x0)) * (2 * w),
    minY + ((y1 - py) / (y1 - y0)) * h, // y-вниз СВГ: лоби зверху, вістря знизу
  ] as [number, number]);
  // БЕЗ заокруглення вістря: гостре вістря по центру → кожна половинка сходить у
  // чистий кінчик на шві (заокруглений низ давав 90°-«гачок» біля шва). Дзеркало
  // бекенду, де heart-l/r будує full без _round_polygon_tip.
  const cut = side === "l" ? minX + w : minX;
  const keep = (p: [number, number]) => (side === "l" ? p[0] <= cut + 1e-9 : p[0] >= cut - 1e-9);
  // Sutherland–Hodgman кліп по півплощині x=cut
  const clipped: Array<[number, number]> = [];
  for (let i = 0; i < pts.length; i++) {
    const cur = pts[i];
    const prev = pts[(i - 1 + pts.length) % pts.length];
    const curIn = keep(cur), prevIn = keep(prev);
    if (curIn !== prevIn) {
      const t = (cut - prev[0]) / (cur[0] - prev[0]);
      clipped.push([cut, prev[1] + t * (cur[1] - prev[1])]);
    }
    if (curIn) clipped.push(cur);
  }
  // Прямий сегмент розрізу = між двома сусідніми вершинами з x≈cut
  let i1 = -1;
  for (let i = 0; i < clipped.length; i++) {
    const a = clipped[i], b = clipped[(i + 1) % clipped.length];
    if (Math.abs(a[0] - cut) < 1e-6 && Math.abs(b[0] - cut) < 1e-6 && Math.abs(a[1] - b[1]) > h * 0.2) {
      i1 = i;
      break;
    }
  }
  if (i1 < 0) return clipped; // fallback: без замка
  const A = clipped[i1], B = clipped[(i1 + 1) % clipped.length];
  const yLo = Math.min(A[1], B[1]), yHi = Math.max(A[1], B[1]);
  const elen = yHi - yLo;
  const cy = (yLo + yHi) / 2;
  const dir = Math.sign(B[1] - A[1]) || 1; // напрям обходу грані
  // ЗАМОК = jigsaw-кнопка (головка ШИРША за шийку → справжнє зчеплення в площині).
  // Дзеркало бекендового _keychain_body_shape heart-l/r: коло k=0.14·грані, шийка
  // 0.60k (головка 2k vs шийка 1.2k → ~1.67×), центр зсунутий на 0.95k за грань.
  // L = виступ (без кліренсу), R = паз (+кліренс 0.6% грані). Опукла дуга у +x.
  const kBase = elen * 0.14;
  const cl = side === "r" ? elen * 0.006 : 0;
  const R = kBase + cl;            // радіус головки (+кліренс для паза)
  const NW = kBase * 0.6 + cl;     // півширина шийки
  const kc = cut + kBase * 0.95;   // центр головки за гранню розрізу
  const xj = kc - Math.sqrt(Math.max(R * R - NW * NW, 1e-9)); // стик шийки з колом
  const th1 = Math.atan2(NW, xj - kc); // кут верхнього стику (~+143°)
  const M = 44;
  const arc: Array<[number, number]> = [];
  for (let s = 0; s <= M; s++) {
    const th = th1 - (s / M) * (2 * th1); // велика дуга th1 → 0° → -th1 (опукла у +x)
    arc.push([kc + R * Math.cos(th), cy + R * Math.sin(th)]);
  }
  let lock: Array<[number, number]> = [
    [cut, cy + NW], [xj, cy + NW], ...arc, [xj, cy - NW], [cut, cy - NW],
  ];
  // впорядкувати так, щоб полілінія йшла від A-кінця до B-кінця
  if (Math.sign(lock[lock.length - 1][1] - lock[0][1]) !== dir) lock.reverse();
  lock = [[cut, A[1]], ...lock, [cut, B[1]]];
  const out: Array<[number, number]> = [];
  for (let i = 0; i <= i1; i++) out.push(clipped[i]);
  out.push(...lock);
  for (let i = i1 + 1; i < clipped.length; i++) out.push(clipped[i]);
  return out;
}

function bodyPath(value: KeychainDesignerConfig) {
  return shapePath(value);
}

function innerBodyPath(value: KeychainDesignerConfig) {
  const inset = Math.max(value.rimWidthMm, 0);
  if (inset <= 0) return bodyPath(value);
  const width = Math.max(value.bodyWidthMm - inset * 2, 0.1);
  const height = Math.max(value.bodyHeightMm - inset * 2, 0.1);
  return shapePath(value, {
    x: inset,
    y: inset,
    width,
    height,
    radius: Math.max(value.cornerRadiusMm - inset, 0),
  });
}

function LoopPreview({ value }: { value: KeychainDesignerConfig }) {
  if (value.baseShape === "token") {
    return null;
  }

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

  const gusset = heartLoopGussetPath(value);
  return (
    <>
      {/* Місток-ущелина (серце): заповнює V між лобами → петля не «булавка» */}
      {gusset && <path d={gusset} fill="#a6926b" />}
      <g transform={`rotate(${value.loopAngleDeg} ${value.loopXMm} ${value.loopYMm})`}>
        <circle cx={value.loopXMm} cy={value.loopYMm} r={outer} fill="#a6926b" />
        <rect x={value.loopXMm - tabWidth / 2} y={value.loopYMm} width={tabWidth} height={tabHeight} rx={tabWidth / 2} fill="#a6926b" />
        <circle cx={value.loopXMm} cy={value.loopYMm} r={inner} fill="#050a18" stroke="rgba(255,255,255,0.35)" strokeWidth={0.25} />
      </g>
    </>
  );
}

function TokenHolePreview({ value }: { value: KeychainDesignerConfig }) {
  if (value.baseShape !== "token") {
    return null;
  }

  const inner = Math.max(value.loopInnerMm, 1.5);
  const guideOuter = Math.max(inner + 0.9, 2.35);
  return (
    <g pointerEvents="none">
      <circle
        cx={value.loopXMm}
        cy={value.loopYMm}
        r={guideOuter}
        fill="none"
        stroke="rgba(248,250,252,0.75)"
        strokeWidth={0.28}
      />
      <circle
        cx={value.loopXMm}
        cy={value.loopYMm}
        r={inner}
        fill="#050a18"
        stroke="rgba(248,250,252,0.42)"
        strokeWidth={0.22}
      />
      <path
        d={`M ${value.loopXMm - guideOuter - 1.6} ${value.loopYMm} H ${value.loopXMm + guideOuter + 1.6}`}
        stroke="rgba(248,250,252,0.55)"
        strokeWidth={0.16}
      />
      <path
        d={`M ${value.loopXMm} ${value.loopYMm - guideOuter - 1.6} V ${value.loopYMm + guideOuter + 1.6}`}
        stroke="rgba(248,250,252,0.55)"
        strokeWidth={0.16}
      />
    </g>
  );
}

export function TemplateMiniature({ design, label, active }: { design: KeychainDesignerConfig; label: string; active: boolean }) {
  const pad = Math.max(design.loopOuterMm * 2.2, 12);
  const view = {
    x: -pad,
    y: -pad,
    w: design.bodyWidthMm + pad * 2,
    h: design.bodyHeightMm + pad * 2,
  };
  const clipId = `templateMapClip-${design.bodyWidthMm}-${design.bodyHeightMm}-${design.loopXMm}-${design.loopYMm}`.replace(/\./g, "-");
  const mapCx = design.mapXMm + design.mapWidthMm / 2;
  const mapCy = design.mapYMm + design.mapHeightMm / 2;
  const bodyCx = design.bodyWidthMm / 2;
  const bodyCy = design.bodyHeightMm / 2;

  return (
    <svg viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`} className="h-14 w-full rounded-[12px] bg-[#050a18] sm:h-16 lg:h-12">
      <defs>
        {/* Карту обрізаємо по РЕАЛЬНОМУ контуру тіла (форма), а не по rect —
            інакше серце/пазл/будиночок показували мапу прямокутником. */}
        <clipPath id={clipId}>
          <path d={bodyPath(design)} />
        </clipPath>
      </defs>
      <g transform={`rotate(${design.layoutRotationDeg || 0} ${bodyCx} ${bodyCy})`}>
        <LoopPreview value={design} />
        <path d={bodyPath(design)} fill="#a6926b" stroke={active ? "#5eead4" : "rgba(255,255,255,0.35)"} strokeWidth={active ? 0.75 : 0.35} />
        <TokenHolePreview value={design} />
        {design.rimWidthMm > 0 && (
          <path
            d={bodyPath(design)}
            fill="none"
            stroke="rgba(68,55,32,0.34)"
            strokeWidth={Math.min(design.rimWidthMm * 1.6, 5)}
          />
        )}
        <g clipPath={`url(#${clipId})`} transform={`rotate(${design.mapRotationDeg || 0} ${mapCx} ${mapCy})`}>
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
          fontSize={Math.max(design.labelTextHeightMm / 0.7, 2.5)}
          fontWeight={700}
          fontFamily={design.labelFontStyle === "wide" ? "Arial Black, Impact, sans-serif" : design.labelFontStyle === "condensed" ? "Arial Narrow, Bahnschrift, sans-serif" : "monospace"}
          letterSpacing={design.labelFontStyle === "wide" ? 0.55 : design.labelFontStyle === "condensed" ? 0.05 : 0.28}
          transform={`rotate(${design.labelAngleDeg} ${design.labelXMm} ${design.labelYMm})`}
        >
          {label || "TEXT"}
        </text>
      </g>
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
  const t = useTranslations("kc");
  return (
    <div className="border-t border-white/10 bg-[#070d1d] px-2 py-2 sm:px-3">
      <div className="mb-1.5 flex items-center justify-between gap-3">
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-300">{t("templates.ready")}</div>
        <div className="text-[11px] font-medium text-slate-400">{t("templates.tapClick")}</div>
      </div>
      <div className="flex gap-2 overflow-x-auto pb-1">
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
              className={`min-w-[108px] rounded-[16px] border p-1.5 text-left transition sm:min-w-[118px] ${
                active
                  ? "border-teal-300 bg-teal-300/10"
                  : "border-white/10 bg-white/[0.04] hover:border-white/25 hover:bg-white/[0.07]"
              }`}
            >
              <TemplateMiniature design={template.design} label={label} active={active} />
              <div className="mt-1 text-[11px] font-semibold text-white sm:text-xs">{t(template.nameKey)}</div>
              <div className="mt-0.5 line-clamp-2 text-[11px] leading-4 text-slate-400 lg:hidden">{t(template.descKey)}</div>
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
  mapBounds,
  cropRotationDeg = 0,
  cropPolygon = null,
}: {
  value: KeychainDesignerConfig;
  label: string;
  onChange: (value: KeychainDesignerConfig) => void;
  /** Bounds of the selected area on the main map. When provided, KeychainDesigner
   *  shows a real OSM tile preview inside the map area instead of generic stripes. */
  mapBounds?: { north: number; south: number; east: number; west: number } | null;
  /** Crop rotation from MapSelector (deg). Applied together with design.mapRotationDeg
   *  so the preview always matches what will be on the printed keychain. */
  cropRotationDeg?: number;
  /** 4 кути обернутого rect ([lon, lat]) — preview обрізає по полігону. */
  cropPolygon?: Array<[number, number]> | null;
}) {
  const t = useTranslations("kc");
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragSessionRef = useRef<DragSession | null>(null);
  const dragCleanupRef = useRef<(() => void) | null>(null);
  const [dragTarget, setDragTarget] = useState<DragTarget | null>(null);
  const [previewSide, setPreviewSide] = useState<"front" | "back">("front");
  const mapCx = value.mapXMm + value.mapWidthMm / 2;
  const mapCy = value.mapYMm + value.mapHeightMm / 2;
  const bodyCx = value.bodyWidthMm / 2;
  const bodyCy = value.bodyHeightMm / 2;

  const view = useMemo(() => {
    const pad = Math.max(value.loopOuterMm * 1.55, 8);
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
    const layoutAngle = -((session.initial.layoutRotationDeg || 0) * Math.PI) / 180;
    const localDx = dx * Math.cos(layoutAngle) - dy * Math.sin(layoutAngle);
    const localDy = dx * Math.sin(layoutAngle) + dy * Math.cos(layoutAngle);
    const next = { ...session.initial };

    if (session.target === "body") {
      next.bodyWidthMm = clamp(session.initial.bodyWidthMm + localDx, 35, 140);
      next.bodyHeightMm = clamp(session.initial.bodyHeightMm + localDy, 26, 96);
      fitAfterBodyResize(next);
    } else if (session.target === "map-move") {
      next.mapXMm = clamp(session.initial.mapXMm + localDx, 0, next.bodyWidthMm - next.mapWidthMm);
      next.mapYMm = clamp(session.initial.mapYMm + localDy, 0, next.bodyHeightMm - next.mapHeightMm);
      const mapCenterX = snapTo(next.mapXMm + next.mapWidthMm / 2, next.bodyWidthMm / 2);
      const mapCenterY = snapTo(next.mapYMm + next.mapHeightMm / 2, next.bodyHeightMm / 2);
      next.mapXMm = clamp(mapCenterX - next.mapWidthMm / 2, 0, next.bodyWidthMm - next.mapWidthMm);
      next.mapYMm = clamp(mapCenterY - next.mapHeightMm / 2, 0, next.bodyHeightMm - next.mapHeightMm);
    } else if (session.target === "map-resize") {
      const minMapWidthBaseMm = next.baseShape === "token" ? MIN_TOKEN_MAP_WIDTH_MM : MIN_MAP_WIDTH_MM;
      const minMapHeightBaseMm = next.baseShape === "token" ? MIN_TOKEN_MAP_HEIGHT_MM : MIN_MAP_HEIGHT_MM;
      next.mapWidthMm = clamp(session.initial.mapWidthMm + localDx, Math.min(minMapWidthBaseMm, next.bodyWidthMm), next.bodyWidthMm - next.mapXMm);
      next.mapHeightMm = clamp(session.initial.mapHeightMm + localDy, Math.min(minMapHeightBaseMm, next.bodyHeightMm), next.bodyHeightMm - next.mapYMm);
    } else if (session.target === "loop") {
      next.loopXMm = session.initial.loopXMm + localDx;
      next.loopYMm = session.initial.loopYMm + localDy;
      clampLoopToBody(next);
    } else if (session.target === "label") {
      next.labelXMm = clamp(session.initial.labelXMm + localDx, 4, next.bodyWidthMm - 4);
      next.labelYMm = clamp(session.initial.labelYMm + localDy, 4, next.bodyHeightMm - 4);
      next.labelXMm = snapTo(next.labelXMm, next.bodyWidthMm / 2);
      next.labelYMm = snapTo(next.labelYMm, next.bodyHeightMm - next.labelBandMm / 2);
    } else if (session.target === "label-rotate") {
      // Кут між початковою позицією pointer та центром label, плюс
      // кут між поточною позицією pointer та центром label — різниця = поворот.
      const cx = session.initial.labelXMm;
      const cy = session.initial.labelYMm;
      const startAngle = Math.atan2(session.start.y - cy, session.start.x - cx);
      const currentAngle = Math.atan2(point.y - cy, point.x - cx);
      let newAngle = ((session.initial.labelAngleDeg + (currentAngle - startAngle) * 180 / Math.PI) % 360 + 360) % 360;
      // Snap to 15° increments коли shift не утримується (просто 1° для свободи)
      newAngle = Math.round(newAngle);
      next.labelAngleDeg = newAngle;
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

    // rAF-throttle: pointermove фірить 100+ разів/с, а кожен updateFromClient —
    // це повний ре-рендер SVG-дизайнера. Оновлюємо стан максимум раз на кадр.
    let pendingMove: { x: number; y: number } | null = null;
    let moveRaf: number | null = null;
    const handleMove = (moveEvent: PointerEvent) => {
      moveEvent.preventDefault();
      pendingMove = { x: moveEvent.clientX, y: moveEvent.clientY };
      if (moveRaf == null) {
        moveRaf = requestAnimationFrame(() => {
          moveRaf = null;
          if (pendingMove && dragSessionRef.current) updateFromClient(pendingMove.x, pendingMove.y);
        });
      }
    };
    const handleEnd = () => {
      if (moveRaf != null) { cancelAnimationFrame(moveRaf); moveRaf = null; }
      if (pendingMove && dragSessionRef.current) updateFromClient(pendingMove.x, pendingMove.y);
      pendingMove = null;
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
    <div className="relative h-full min-h-[280px] overflow-hidden rounded-[22px] bg-[#050a18] p-2 sm:min-h-[340px] sm:p-3">
      <div className="pointer-events-none absolute left-3 top-3 z-10 rounded-full border border-white/10 bg-black/30 px-3 py-1.5 text-[11px] font-semibold text-white/75 backdrop-blur">
        {previewSide === "front" ? t("designer.hintFront") : t("designer.hintBack")}
      </div>
      <div className="absolute right-3 top-3 z-20 flex overflow-hidden rounded-full border border-white/15 bg-black/35 p-1 backdrop-blur">
        <button
          type="button"
          onClick={() => setPreviewSide("front")}
          className={`min-h-[32px] rounded-full px-3 text-[11px] font-semibold ${previewSide === "front" ? "bg-white text-[#050a18]" : "text-white/72"}`}
        >
          {t("designer.front")}
        </button>
        <button
          type="button"
          onClick={() => setPreviewSide("back")}
          className={`min-h-[32px] rounded-full px-3 text-[11px] font-semibold ${previewSide === "back" ? "bg-white text-[#050a18]" : "text-white/72"}`}
        >
          {t("designer.back")}
        </button>
      </div>
      <svg
        ref={svgRef}
        data-testid="keychain-designer-svg"
        viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
        preserveAspectRatio="xMidYMid meet"
        className="block h-full min-h-[280px] w-full touch-none select-none sm:min-h-[340px]"
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
          {/* Map area MASK: прямокутник мапи МІНУС label_band МІНУС token hole.
              Так карта НЕ накладається на текст і не показується крізь отвір токена. */}
          <mask id="keychainMapAreaMask">
            <rect
              x={value.mapXMm}
              y={value.mapYMm}
              width={value.mapWidthMm}
              height={value.mapHeightMm}
              fill="white"
            />
            {/* За замовчуванням карту під написом НЕ вирізаємо (текст лежить
                поверх карти, підіймаються лише літери). Прямокутну зону-підкладку
                вмикає лише опційний labelClearBand. */}
            {value.labelClearBand ? (
              <rect
                x={value.labelXMm - value.labelWidthMm / 2}
                y={value.labelYMm - value.labelBandMm / 2}
                width={value.labelWidthMm}
                height={value.labelBandMm}
                fill="black"
                transform={`rotate(${value.labelAngleDeg} ${value.labelXMm} ${value.labelYMm})`}
              />
            ) : null}
            {/* Виключаємо token loop hole */}
            {value.baseShape === "token" && (
              <circle
                cx={value.loopXMm}
                cy={value.loopYMm}
                r={Math.max(value.loopInnerMm, 1.5) + 0.5}
                fill="black"
              />
            )}
          </mask>
          <clipPath id="keychainMapClip">
            <rect
              x={value.mapXMm}
              y={value.mapYMm}
              width={value.mapWidthMm}
              height={value.mapHeightMm}
            />
          </clipPath>
          <clipPath id="keychainInnerBodyClip">
            <path d={innerBodyPath(value)} />
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
        <g transform={`rotate(${value.layoutRotationDeg || 0} ${bodyCx} ${bodyCy})`}>
          <LoopPreview value={value} />
          {/* Token mode: реальний отвір у body через mask (фон видно крізь нього) */}
          {value.baseShape === "token" && (
            <defs>
              <mask id="tokenBodyMask">
                <path d={bodyPath(value)} fill="white" />
                <circle cx={value.loopXMm} cy={value.loopYMm} r={Math.max(value.loopInnerMm, 1.5)} fill="black" />
              </mask>
            </defs>
          )}
          <path
            d={bodyPath(value)}
            fill="#a6926b"
            stroke="rgba(255,255,255,0.42)"
            strokeWidth={0.35}
            mask={value.baseShape === "token" ? "url(#tokenBodyMask)" : undefined}
          />
          <TokenHolePreview value={value} />
          {/* Отвір вушка малюємо ПОВЕРХ тіла, щоб він був видимий навіть коли
              петля перекриває корпус (інакше body перекривав дірку і її «не
              вирізало» у превʼю). Колір = темний фон → читається як наскрізна. */}
          {value.baseShape !== "token" && (
            <g transform={`rotate(${value.loopAngleDeg} ${value.loopXMm} ${value.loopYMm})`} pointerEvents="none">
              {value.loopStyle === "slot" ? (
                <rect
                  x={value.loopXMm - value.loopInnerMm * 1.25}
                  y={value.loopYMm - value.loopInnerMm * 0.58}
                  width={value.loopInnerMm * 2.5}
                  height={value.loopInnerMm * 1.16}
                  rx={value.loopInnerMm * 0.58}
                  fill="#050a18"
                  stroke="rgba(255,255,255,0.4)"
                  strokeWidth={0.25}
                />
              ) : (
                <circle
                  cx={value.loopXMm}
                  cy={value.loopYMm}
                  r={value.loopInnerMm}
                  fill="#050a18"
                  stroke="rgba(255,255,255,0.4)"
                  strokeWidth={0.25}
                />
              )}
            </g>
          )}
          {value.rimWidthMm > 0 && (
            <path
              d={bodyPath(value)}
              fill="none"
              stroke="rgba(68,55,32,0.32)"
              strokeWidth={Math.min(value.rimWidthMm * 1.6, 5)}
              pointerEvents="none"
            />
          )}
          {value.labelClearBand ? (
            <rect
              x={value.labelXMm - value.labelWidthMm / 2}
              y={value.labelYMm - value.labelBandMm / 2}
              width={value.labelWidthMm}
              height={value.labelBandMm}
              rx={Math.min(2, value.labelBandMm / 2)}
              fill={previewSide === "back" ? "rgba(5,10,24,0.16)" : "rgba(255,255,255,0.04)"}
              stroke={previewSide === "back" ? "rgba(94,234,212,0.38)" : "rgba(255,255,255,0.28)"}
              strokeDasharray="1.4 1.2"
              strokeWidth={0.28}
              transform={`rotate(${value.labelAngleDeg} ${value.labelXMm} ${value.labelYMm})`}
              pointerEvents="none"
            />
          ) : null}

          {previewSide === "front" ? (
            /* Preview обрізається по INNER BODY (форма тіла мінус rim).
               Так контент карти не вилазить за межі корпусу — як на готовому
               надрукованому брелку. */
            <g clipPath="url(#keychainInnerBodyClip)">
              <g
                mask="url(#keychainMapAreaMask)"
                opacity={0.98}
                transform={`rotate(${(value.mapRotationDeg || 0) % 360} ${mapCx} ${mapCy})`}
              >
                <rect x={value.mapXMm} y={value.mapYMm} width={value.mapWidthMm} height={value.mapHeightMm} fill="#e8e1cc" />
                {mapBounds ? (
                  /* Реальний перегляд OSM (buildings/roads/water/parks) обраної
                     ділянки як НАТИВНІ SVG-шляхи у спільній мм-системі координат.
                     Раніше було через <foreignObject> → на iOS/Safari прев'ю
                     вилазило у лівий верхній кут (WebKit #23113). Тепер коректно
                     скрізь і успадковує mapRotation/clip/mask батька. */
                  <LiveCitySvgPaths
                    bounds={mapBounds}
                    cropRotationDeg={cropRotationDeg}
                    cropPolygon={cropPolygon}
                    design={{
                      bodyWidthMm: value.bodyWidthMm,
                      bodyHeightMm: value.bodyHeightMm,
                      cornerRadiusMm: value.cornerRadiusMm,
                      mapXMm: value.mapXMm,
                      mapYMm: value.mapYMm,
                      mapWidthMm: value.mapWidthMm,
                      mapHeightMm: value.mapHeightMm,
                      loopXMm: value.loopXMm,
                      loopYMm: value.loopYMm,
                      loopOuterMm: value.loopOuterMm,
                      loopInnerMm: value.loopInnerMm,
                      rimWidthMm: value.rimWidthMm,
                      baseShape: value.baseShape as any,
                    }}
                  />
                ) : (
                  /* Fallback — generic stripes якщо ще не обрано ділянку */
                  <>
                    {Array.from({ length: 8 }).map((_, idx) => (
                      <path
                        key={`stub-road-${idx}`}
                        d={`M ${value.mapXMm + ((idx * 13) % Math.max(value.mapWidthMm, 1))} ${value.mapYMm - 4} L ${value.mapXMm + ((idx * 13) % Math.max(value.mapWidthMm, 1)) + 10} ${value.mapYMm + value.mapHeightMm + 6}`}
                        stroke="#999"
                        strokeWidth={0.6}
                        opacity={0.5}
                        strokeLinecap="round"
                      />
                    ))}
                    <text
                      x={mapCx}
                      y={mapCy}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fill="#6a5d44"
                      fontSize={2.2}
                      fontWeight={700}
                    >
                      {t("designer.pickArea")}
                    </text>
                  </>
                )}
              </g>
            </g>
          ) : (
            <g pointerEvents="none">
              <path d={innerBodyPath(value)} fill="rgba(255,255,255,0.035)" stroke="rgba(94,234,212,0.28)" strokeDasharray="1.8 1.4" strokeWidth={0.28} />
              <text
                x={bodyCx}
                y={Math.max(value.bodyHeightMm - 4, value.bodyHeightMm / 2)}
                textAnchor="middle"
                fill="rgba(248,250,252,0.64)"
                fontSize={2.1}
                fontWeight={800}
              >
                back side / mirrored check
              </text>
            </g>
          )}

          <rect
            data-testid="map-move-hit"
            x={value.mapXMm}
            y={value.mapYMm}
            width={value.mapWidthMm}
            height={value.mapHeightMm}
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
            fill="#141414"
            stroke="rgba(20,20,20,0.18)"
            strokeWidth={Math.max(value.labelStrokeMm * 0.06, 0.03)}
            paintOrder="stroke"
            fontSize={Math.max(value.labelTextHeightMm / 0.7, 2.6)}
            fontWeight={700}
            fontFamily={value.labelFontStyle === "wide" ? "Arial Black, Impact, sans-serif" : value.labelFontStyle === "condensed" ? "Arial Narrow, Bahnschrift, sans-serif" : "monospace"}
            letterSpacing={value.labelFontStyle === "wide" ? 0.55 : value.labelFontStyle === "condensed" ? 0.05 : 0.28}
            transform={`rotate(${value.labelAngleDeg} ${value.labelXMm} ${value.labelYMm})`}
            className="cursor-move"
            onPointerDown={(event) => beginDrag(event, "label")}
          >
            {label || "TEXT"}
          </text>
          {/* Rotate handle: великий кружок з ↻, на 6мм над текстом (далеко щоб
              не плутати з drag-move). Drag по ньому = поворот тексту. */}
          {(() => {
            const handleOffset = Math.max(value.labelBandMm / 2 + 4.5, 5.5);
            const angle = value.labelAngleDeg * Math.PI / 180;
            // напрям "вгору" від тексту з врахуванням поточного кута
            const hx = value.labelXMm + Math.sin(angle) * handleOffset;
            const hy = value.labelYMm - Math.cos(angle) * handleOffset;
            return (
              <g onPointerDown={(event) => beginDrag(event, "label-rotate")} className="cursor-grab" data-testid="label-rotate-hit">
                <line x1={value.labelXMm} y1={value.labelYMm} x2={hx} y2={hy} stroke="rgba(94,234,212,0.55)" strokeWidth={0.35} strokeDasharray="0.8 0.6" />
                <circle cx={hx} cy={hy} r={2.4} fill="#5eead4" stroke="#050a18" strokeWidth={0.35} />
                <text x={hx} y={hy + 0.7} textAnchor="middle" fontSize={2.6} fill="#050a18" fontWeight={900} pointerEvents="none">↻</text>
              </g>
            );
          })()}
        </g>

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
            {value.baseShape === "token" ? `hole Ø${(value.loopInnerMm * 2).toFixed(1)}` : `O ${value.loopOuterMm.toFixed(1)} / hole ${value.loopInnerMm.toFixed(1)}`}
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
          width={11}
          height={11}
          rx={1.7}
          fill="#f8fafc"
          stroke="#14b8a6"
          strokeWidth={0.45}
          className="cursor-nwse-resize"
          onPointerDown={(event) => beginDrag(event, "map-resize")}
        />
        <rect
          data-testid="body-resize-handle"
          x={value.bodyWidthMm - 4.4}
          y={value.bodyHeightMm - 4.4}
          width={8.8}
          height={8.8}
          rx={1.1}
          fill="#14b8a6"
          className="cursor-nwse-resize"
          onPointerDown={(event) => beginDrag(event, "body")}
        />
      </svg>
    </div>
  );
}
