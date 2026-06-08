"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { MapContainer, TileLayer, useMap, ZoomControl } from "react-leaflet";
import L from "leaflet";
import "leaflet-draw";
import { useGenerationStore } from "@/store/generation-store";

// Виправлення іконок Leaflet для Next.js (тільки на клієнті)
if (typeof window !== "undefined") {
  delete (L.Icon.Default.prototype as any)._getIconUrl;
  L.Icon.Default.mergeOptions({
    iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
    iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
    shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
  });
}

// Fixed 1:10000 scale: max real-world zone = model size (mm) * 10 meters.
// 80mm(8cm) -> 800m, +100m per +1cm, up to 200mm(20cm) -> 2000m.
const ZONE_M_PER_MODEL_MM = 10.0;

/** Clamp a bounds to a max side length in meters, keeping its centre. */
function clampBoundsToMaxMeters(bounds: L.LatLngBounds, maxSpanM: number): { bounds: L.LatLngBounds; clamped: boolean } {
  const center = bounds.getCenter();
  const nsM = bounds.getNorth() - bounds.getSouth();
  const ewM = bounds.getEast() - bounds.getWest();
  const latM = Math.abs(nsM) * 111320;
  const lonM = Math.abs(ewM) * 111320 * Math.max(0.05, Math.cos((center.lat * Math.PI) / 180));
  if (latM <= maxSpanM && lonM <= maxSpanM) return { bounds, clamped: false };
  const halfLatDeg = Math.min(latM, maxSpanM) / 2 / 111320;
  const halfLonDeg = Math.min(lonM, maxSpanM) / 2 / (111320 * Math.max(0.05, Math.cos((center.lat * Math.PI) / 180)));
  return {
    bounds: L.latLngBounds(
      L.latLng(center.lat - halfLatDeg, center.lng - halfLonDeg),
      L.latLng(center.lat + halfLatDeg, center.lng + halfLonDeg),
    ),
    clamped: true,
  };
}

function DrawControl() {
  const map = useMap();
  const drawnItemsRef = useRef<L.FeatureGroup>(new L.FeatureGroup());
  const { setSelectedArea, modelSizeMm } = useGenerationStore();

  useEffect(() => {
    if (!map) return;

    map.addLayer(drawnItemsRef.current);

    const drawControl = new L.Control.Draw({
      position: "topright",
      draw: {
        rectangle: {
          shapeOptions: {
            color: "#3388ff",
            weight: 2,
          },
        },
        polygon: {
          shapeOptions: {
            color: "#3388ff",
            weight: 2,
          },
        },
        circle: {
          shapeOptions: {
            color: "#3388ff",
            weight: 2,
          },
        },
        marker: false,
        circlemarker: false,
        polyline: false,
      },
      edit: {
        featureGroup: drawnItemsRef.current,
        remove: true,
      },
    });

    map.addControl(drawControl);

    const maxSpanM = () => Math.max(50, (modelSizeMm || 80) * ZONE_M_PER_MODEL_MM);

    const applyBounds = (layer: any) => {
      if (!("getBounds" in layer) || typeof layer.getBounds !== "function") {
        console.warn("Draw created layer does not support getBounds:", layer);
        return;
      }
      const raw = layer.getBounds() as L.LatLngBounds;
      const { bounds, clamped } = clampBoundsToMaxMeters(raw, maxSpanM());
      if (clamped) {
        // Shrink the visible rectangle to the allowed size and tell the user why.
        if (typeof layer.setBounds === "function") {
          try { layer.setBounds(bounds); } catch { /* polygon/circle: visual stays */ }
        }
        const cm = Math.round((modelSizeMm || 80) / 10);
        const mx = Math.round(maxSpanM());
        try {
          window.dispatchEvent(new CustomEvent("monadruk:toast", {
            detail: { type: "warn", message: `Зона обмежена до ~${mx} м (макс. для моделі ${cm} см, масштаб 1:10000). Для більшої зони збільште розмір моделі.` },
          }));
        } catch { /* no-op */ }
        console.warn(`[zone] clamped to ${mx}m for ${cm}cm model`);
      }
      setSelectedArea(bounds);
    };

    const handleDrawCreated = (e: any) => {
      const layer = e.layer;
      drawnItemsRef.current.addLayer(layer);
      applyBounds(layer);
    };

    const handleDrawEdited = () => {
      const layers = drawnItemsRef.current.getLayers();
      if (layers.length > 0) {
        applyBounds(layers[0] as L.Layer);
      }
    };

    const handleDrawDeleted = () => {
      setSelectedArea(null);
    };

    map.on(L.Draw.Event.CREATED, handleDrawCreated);
    map.on(L.Draw.Event.EDITED, handleDrawEdited);
    map.on(L.Draw.Event.DELETED, handleDrawDeleted);

    return () => {
      map.off(L.Draw.Event.CREATED, handleDrawCreated);
      map.off(L.Draw.Event.EDITED, handleDrawEdited);
      map.off(L.Draw.Event.DELETED, handleDrawDeleted);
      map.removeControl(drawControl);
    };
  }, [map, setSelectedArea, modelSizeMm]);

  return null;
}

type KeychainCropSpec = {
  aspectRatio: number;
  maxMetersPerMm: number;
  /** Комфортний цільовий масштаб для INITIAL розміру crop. Якщо не задано — береться 60% від maxMetersPerMm. */
  targetMetersPerMm?: number;
  mapWidthMm: number;
  mapHeightMm: number;
  /** Форма брелка — впливає на полігон виділення на карті. */
  baseShape?: "rounded" | "capsule" | "tag" | "octagon" | "token" | "circle" | "hexagon" | "heart";
  /** When true, the polygon sent to the backend is the actual (rotated) SHAPE
   *  outline, not the axis-aligned bbox — so the model is cut to that shape
   *  (heart/circle/…). Keychains keep bbox (their base shape is separate). */
  cropToShape?: boolean;
  /** Радіус заокруглення кутів (для visual shape). */
  cornerRadiusMm?: number;
  rotationDeg?: number;
  onRotationChange?: (rotationDeg: number) => void;
  /** Викликається при будь-якій зміні rect (drag/resize/rotate) з 4-ма кутами
   *  rotated rect у форматі [lon, lat]. Backend використовує це щоб обрізати
   *  OSM-дані строго до обраної ділянки (а не axis-aligned bbox). */
  onPolygonChange?: (polygon: Array<[number, number]>) => void;
};

const MAP_CLICK_SUPPRESS_AFTER_DRAG_MS = 900;

function metersPerDegreeLng(lat: number) {
  return 111_320 * Math.max(Math.cos((lat * Math.PI) / 180), 0.18);
}

function boundsFromCenterMeters(center: L.LatLng, widthM: number, heightM: number) {
  const halfLat = (heightM / 2) / 111_320;
  const halfLng = (widthM / 2) / metersPerDegreeLng(center.lat);
  return L.latLngBounds(
    [center.lat - halfLat, center.lng - halfLng],
    [center.lat + halfLat, center.lng + halfLng],
  );
}

function boundsSizeMeters(bounds: L.LatLngBounds) {
  const center = bounds.getCenter();
  return {
    widthM: Math.abs(bounds.getEast() - bounds.getWest()) * metersPerDegreeLng(center.lat),
    heightM: Math.abs(bounds.getNorth() - bounds.getSouth()) * 111_320,
  };
}

function normalizeAngle(angleDeg: number) {
  // Гранулярність 1° (раніше було 5°). Користувач має повний контроль над кутом.
  return ((Math.round(angleDeg) % 360) + 360) % 360;
}

function offsetLatLngMeters(center: L.LatLng, dxM: number, dyM: number) {
  return L.latLng(center.lat + dyM / 111_320, center.lng + dxM / metersPerDegreeLng(center.lat));
}

function localOffsetFromCenterMeters(center: L.LatLng, point: L.LatLng) {
  return {
    x: (point.lng - center.lng) * metersPerDegreeLng(center.lat),
    y: (point.lat - center.lat) * 111_320,
  };
}

/** Генерує полігон форми брелка (token=oval, rounded=rect with rounded corners, etc).
 *  Повертає список точок ВЗДОВЖ ПЕРИМЕТРА у local (dx, dy) метрах від центру.
 *  Для backend і aspect-розрахунків використовується bbox цих точок (axis-aligned). */
function shapeOutlinePoints(widthM: number, heightM: number, shape: string, cornerRadiusFraction: number = 0.15): Array<{x: number; y: number}> {
  const w = widthM, h = heightM;
  const pts: Array<{x: number; y: number}> = [];
  if (shape === "token" || shape === "capsule") {
    // Capsule/oval — дві півкола по краях, прямі сторони
    const r = Math.min(w, h) / 2;  // radius = half of smaller dim
    const straight = Math.abs(w - h);  // довжина прямих сторін
    const isWide = w >= h;
    const N = 24;
    if (isWide) {
      // лівий півкруг (від низу проти годинникової)
      for (let i = 0; i <= N / 2; i++) {
        const a = Math.PI / 2 + (Math.PI * i / (N / 2));
        pts.push({ x: -straight / 2 + Math.cos(a) * r, y: Math.sin(a) * r });
      }
      // правий півкруг
      for (let i = 0; i <= N / 2; i++) {
        const a = -Math.PI / 2 + (Math.PI * i / (N / 2));
        pts.push({ x: straight / 2 + Math.cos(a) * r, y: Math.sin(a) * r });
      }
    } else {
      for (let i = 0; i <= N / 2; i++) {
        const a = 0 + (Math.PI * i / (N / 2));
        pts.push({ x: Math.cos(a) * r, y: straight / 2 + Math.sin(a) * r });
      }
      for (let i = 0; i <= N / 2; i++) {
        const a = Math.PI + (Math.PI * i / (N / 2));
        pts.push({ x: Math.cos(a) * r, y: -straight / 2 + Math.sin(a) * r });
      }
    }
  } else if (shape === "circle") {
    // Perfect circle (radius = half of the smaller dimension)
    const r = Math.min(w, h) / 2;
    const N = 40;
    for (let i = 0; i < N; i++) {
      const a = (2 * Math.PI * i) / N;
      pts.push({ x: Math.cos(a) * r, y: Math.sin(a) * r });
    }
  } else if (shape === "heart") {
    // Classic heart curve, normalised to fit the box (tip pointing down).
    const raw: Array<{ x: number; y: number }> = [];
    const N = 64;
    for (let i = 0; i < N; i++) {
      const t = (2 * Math.PI * i) / N;
      const x = 16 * Math.pow(Math.sin(t), 3);
      const y = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t);
      raw.push({ x, y });
    }
    const xs = raw.map((p) => p.x), ys = raw.map((p) => p.y);
    const minx = Math.min(...xs), maxx = Math.max(...xs), miny = Math.min(...ys), maxy = Math.max(...ys);
    const s = Math.min(w / (maxx - minx), h / (maxy - miny));
    const cx = (minx + maxx) / 2, cy = (miny + maxy) / 2;
    for (const p of raw) pts.push({ x: (p.x - cx) * s, y: (p.y - cy) * s });
  } else if (shape === "hexagon") {
    // Flat-top hexagon inscribed in the box
    const rx = w / 2, ry = h / 2;
    for (let i = 0; i < 6; i++) {
      const a = (Math.PI / 3) * i + Math.PI / 6; // pointy offset
      pts.push({ x: Math.cos(a) * rx, y: Math.sin(a) * ry });
    }
  } else if (shape === "octagon") {
    const r = Math.min(w, h) / 2 * 0.4;  // зрізаний кут
    pts.push({ x: -w / 2 + r, y: -h / 2 });
    pts.push({ x: w / 2 - r, y: -h / 2 });
    pts.push({ x: w / 2, y: -h / 2 + r });
    pts.push({ x: w / 2, y: h / 2 - r });
    pts.push({ x: w / 2 - r, y: h / 2 });
    pts.push({ x: -w / 2 + r, y: h / 2 });
    pts.push({ x: -w / 2, y: h / 2 - r });
    pts.push({ x: -w / 2, y: -h / 2 + r });
  } else {
    // rounded/tag — прямокутник з заокругленими кутами
    const r = Math.min(w, h) * cornerRadiusFraction;
    const N = 6;  // points per corner
    // bottom-left → bottom-right → top-right → top-left → close
    for (let i = 0; i <= N; i++) {
      const a = Math.PI + (Math.PI / 2 * i / N);
      pts.push({ x: -w / 2 + r + Math.cos(a) * r, y: -h / 2 + r + Math.sin(a) * r });
    }
    for (let i = 0; i <= N; i++) {
      const a = -Math.PI / 2 + (Math.PI / 2 * i / N);
      pts.push({ x: w / 2 - r + Math.cos(a) * r, y: -h / 2 + r + Math.sin(a) * r });
    }
    for (let i = 0; i <= N; i++) {
      const a = 0 + (Math.PI / 2 * i / N);
      pts.push({ x: w / 2 - r + Math.cos(a) * r, y: h / 2 - r + Math.sin(a) * r });
    }
    for (let i = 0; i <= N; i++) {
      const a = Math.PI / 2 + (Math.PI / 2 * i / N);
      pts.push({ x: -w / 2 + r + Math.cos(a) * r, y: h / 2 - r + Math.sin(a) * r });
    }
  }
  return pts;
}

function rotatedShapePoints(center: L.LatLng, widthM: number, heightM: number, rotationDeg: number, shape: string, cornerRadiusFraction: number = 0.15): L.LatLng[] {
  const pts = shapeOutlinePoints(widthM, heightM, shape, cornerRadiusFraction);
  const angle = (rotationDeg * Math.PI) / 180;
  const cos = Math.cos(angle), sin = Math.sin(angle);
  return pts.map((p) => offsetLatLngMeters(center, p.x * cos - p.y * sin, p.x * sin + p.y * cos));
}

function rotatedCropCorners(center: L.LatLng, widthM: number, heightM: number, rotationDeg: number) {
  const angle = (rotationDeg * Math.PI) / 180;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const corners = [
    { x: -widthM / 2, y: heightM / 2 },
    { x: widthM / 2, y: heightM / 2 },
    { x: widthM / 2, y: -heightM / 2 },
    { x: -widthM / 2, y: -heightM / 2 },
  ];
  return corners.map((corner) =>
    offsetLatLngMeters(
      center,
      corner.x * cos - corner.y * sin,
      corner.x * sin + corner.y * cos,
    ),
  );
}

function rotatedControlPoint(center: L.LatLng, widthM: number, heightM: number, rotationDeg: number, localX: number, localY: number) {
  const angle = (rotationDeg * Math.PI) / 180;
  return offsetLatLngMeters(
    center,
    localX * Math.cos(angle) - localY * Math.sin(angle),
    localX * Math.sin(angle) + localY * Math.cos(angle),
  );
}

function targetCropMeters(spec: KeychainCropSpec) {
  // Комфортний initial — на 60% від максимуму або як задано в spec.targetMetersPerMm.
  // Це гарантує що при першому відкритті крокова зона "у зеленій зоні" друкованості.
  const aspect = Math.max(spec.aspectRatio, 0.2);
  const target = spec.targetMetersPerMm ?? Math.max(spec.maxMetersPerMm * 0.5, 2.5);
  const byWidth = spec.mapWidthMm * target;
  const byHeight = spec.mapHeightMm * target * aspect;
  const widthM = Math.min(byWidth, byHeight);
  return { widthM, heightM: widthM / aspect };
}

function safeCropMeters(spec: KeychainCropSpec) {
  const aspect = Math.max(spec.aspectRatio, 0.2);
  const safeByWidth = spec.mapWidthMm * spec.maxMetersPerMm;
  const safeByHeight = spec.mapHeightMm * spec.maxMetersPerMm * aspect;
  const widthM = Math.min(safeByWidth, safeByHeight);
  return {
    widthM,
    heightM: widthM / aspect,
  };
}

function KeychainCropOverlay({ spec }: { spec: KeychainCropSpec }) {
  const map = useMap();
  const { selectedArea, setSelectedArea } = useGenerationStore();
  const initialSelectedAreaRef = useRef(selectedArea);
  const shapeRef = useRef<L.Polygon | null>(null);
  // Current shape kind + corner fraction kept in refs so the drag handler
  // (created once in the setup effect) always draws the CURRENT shape instead of
  // the one captured when the effect first ran (which caused moving to revert
  // the figure to the initial rectangle).
  const shapeKindRef = useRef<string>(spec.baseShape || "rounded");
  const cornerFracRef = useRef<number>(0.15);
  const resizeHandleRef = useRef<L.Marker | null>(null);
  const rotateHandleRef = useRef<L.Marker | null>(null);
  const labelRef = useRef<L.Marker | null>(null);
  const currentBoundsRef = useRef<L.LatLngBounds | null>(null);
  const lastDragEndedAtRef = useRef(0);
  const handleInteractionRef = useRef(false);
  const handleInteractionTimerRef = useRef<number | null>(null);
  const dragStateRef = useRef<{
    startPoint: L.Point;
    startCenter: L.LatLng;
    widthM: number;
    heightM: number;
  } | null>(null);

  const safeSize = useMemo(() => safeCropMeters(spec), [spec.aspectRatio, spec.mapHeightMm, spec.mapWidthMm, spec.maxMetersPerMm]);
  const targetSize = useMemo(() => targetCropMeters(spec), [spec.aspectRatio, spec.mapHeightMm, spec.mapWidthMm, spec.maxMetersPerMm, spec.targetMetersPerMm]);
  const northCenter = (bounds: L.LatLngBounds) => L.latLng(bounds.getNorth(), bounds.getCenter().lng);
  const rotationDeg = normalizeAngle(spec.rotationDeg || 0);
  const rotationRef = useRef(rotationDeg);

  useEffect(() => {
    rotationRef.current = rotationDeg;
    const bounds = currentBoundsRef.current;
    const shape = shapeRef.current;
    if (!bounds || !shape) return;
    const center = bounds.getCenter();
    const size = boundsSizeMeters(bounds);
    // Visual shape outline (oval for token, rounded rect etc)
    const shapeKind = spec.baseShape || "rounded";
    const cornerFrac = Math.min(0.45, Math.max(0.0, (spec.cornerRadiusMm || 4) / Math.max(spec.mapWidthMm, spec.mapHeightMm, 1)));
    shapeKindRef.current = shapeKind;
    cornerFracRef.current = cornerFrac;
    const visualPoints = rotatedShapePoints(center, size.widthM, size.heightM, rotationDeg, shapeKind, cornerFrac);
    shape.setLatLngs(visualPoints);
    resizeHandleRef.current?.setLatLng(rotatedControlPoint(center, size.widthM, size.heightM, rotationDeg, size.widthM / 2, -size.heightM / 2));
    rotateHandleRef.current?.setLatLng(rotatedControlPoint(center, size.widthM, size.heightM, rotationDeg, 0, size.heightM / 2 + 42));
    // Polygon for backend: the actual shape outline (cropToShape) or bbox corners.
    const backendPoly = spec.cropToShape
      ? visualPoints.map((p) => [p.lng, p.lat] as [number, number])
      : rotatedCropCorners(center, size.widthM, size.heightM, rotationDeg).map((c) => [c.lng, c.lat] as [number, number]);
    spec.onPolygonChange?.(backendPoly);
  }, [rotationDeg, spec.onPolygonChange, spec.baseShape, spec.cornerRadiusMm, spec.mapWidthMm, spec.mapHeightMm]);

  useEffect(() => {
    if (!map) return;

    const initialSelectedArea = initialSelectedAreaRef.current;
    const existingCenter = initialSelectedArea?.getCenter() ?? map.getCenter();
    // KEYCHAIN-MODE: ЗАВЖДИ починаємо з комфортного targetSize. Попередня зона
    // з іншої сторінки/міста ігнорується — інакше нова мапа відкривається з
    // занадто великою зоною (RED warning) тільки тому що в Zustand store
    // ще лежить bounds від main мапи.
    const aspect = Math.max(spec.aspectRatio, 0.2);
    const existingSize = initialSelectedArea ? boundsSizeMeters(initialSelectedArea) : targetSize;
    // Якщо попередня зона менша за target — поважаємо вибір користувача.
    // Якщо більша за target → знижуємо до target (комфорт за замовчуванням).
    const candidateWidth = Math.min(existingSize.widthM || targetSize.widthM, targetSize.widthM);
    const widthM = Math.max(Math.min(candidateWidth, safeSize.widthM), Math.min(targetSize.widthM, 80));
    const heightM = Math.min(widthM / aspect, safeSize.heightM);
    const initialBounds = boundsFromCenterMeters(existingCenter, widthM, heightM);

    const cropSize = { widthM, heightM };
    const shapeKind = spec.baseShape || "rounded";
    const cornerFrac = Math.min(0.45, Math.max(0.0, (spec.cornerRadiusMm || 4) / Math.max(spec.mapWidthMm, spec.mapHeightMm, 1)));
    shapeKindRef.current = shapeKind;
    cornerFracRef.current = cornerFrac;
    const shape = L.polygon(rotatedShapePoints(existingCenter, cropSize.widthM, cropSize.heightM, rotationRef.current, shapeKind, cornerFrac), {
      color: "#14b8a6",
      weight: 2,
      fillColor: "#14b8a6",
      fillOpacity: 0.14,
      dashArray: "8 6",
      interactive: true,
    }).addTo(map);
    shapeRef.current = shape;
    currentBoundsRef.current = initialBounds;
    setSelectedArea(initialBounds);

    const handleIcon = L.divIcon({
      className: "",
      html: '<div style="width:44px;height:44px;border-radius:14px;background:#14b8a6;border:4px solid white;box-shadow:0 10px 24px rgba(15,23,42,.25);"></div>',
      iconSize: [44, 44],
      iconAnchor: [22, 22],
    });
    const labelIcon = L.divIcon({
      className: "",
      html: '<div style="padding:6px 9px;border-radius:999px;background:rgba(5,10,24,.82);border:1px solid rgba(255,255,255,.3);color:white;font:700 11px/1.1 system-ui;white-space:nowrap;">клік = поставити · ⟳ = крутити</div>',
      iconSize: [172, 28],
      iconAnchor: [86, 36],
    });
    const rotateIcon = L.divIcon({
      className: "",
      html: '<div style="width:42px;height:42px;border-radius:999px;background:#050a18;border:3px solid #5eead4;box-shadow:0 10px 24px rgba(15,23,42,.25);display:grid;place-items:center;color:white;font:900 20px/1 system-ui;">⟳</div>',
      iconSize: [42, 42],
      iconAnchor: [21, 21],
    });

    const handle = L.marker(rotatedControlPoint(existingCenter, cropSize.widthM, cropSize.heightM, rotationRef.current, cropSize.widthM / 2, -cropSize.heightM / 2), {
      icon: handleIcon,
      draggable: true,
      zIndexOffset: 800,
    }).addTo(map);
    resizeHandleRef.current = handle;

    const rotateHandle = L.marker(rotatedControlPoint(existingCenter, cropSize.widthM, cropSize.heightM, rotationRef.current, 0, cropSize.heightM / 2 + 42), {
      icon: rotateIcon,
      draggable: true,
      zIndexOffset: 820,
    }).addTo(map);
    rotateHandleRef.current = rotateHandle;

    const label = L.marker(northCenter(initialBounds), {
      icon: labelIcon,
      interactive: false,
      zIndexOffset: 700,
    }).addTo(map);
    labelRef.current = label;

    const syncDecorations = (bounds: L.LatLngBounds, nextRotationDeg = rotationRef.current) => {
      const center = bounds.getCenter();
      const size = boundsSizeMeters(bounds);
      // Visual: показуємо ПОТОЧНУ форму (через ref, не зафіксовану в замиканні)
      shape.setLatLngs(rotatedShapePoints(center, size.widthM, size.heightM, nextRotationDeg, shapeKindRef.current, cornerFracRef.current));
      resizeHandleRef.current?.setLatLng(rotatedControlPoint(center, size.widthM, size.heightM, nextRotationDeg, size.widthM / 2, -size.heightM / 2));
      rotateHandleRef.current?.setLatLng(rotatedControlPoint(center, size.widthM, size.heightM, nextRotationDeg, 0, size.heightM / 2 + 42));
      labelRef.current?.setLatLng(northCenter(bounds));
    };

    const updateBounds = (bounds: L.LatLngBounds) => {
      currentBoundsRef.current = bounds;
      syncDecorations(bounds);
      setSelectedArea(bounds);
      // Обчислюємо 4 кути ОБЕРНУТОГО прямокутника (lat/lon) і передаємо нагору.
      // Це дозволяє backend'у обрізати OSM строго по обертанню, а не bbox.
      if (spec.onPolygonChange) {
        const center = bounds.getCenter();
        const size = boundsSizeMeters(bounds);
        const poly = spec.cropToShape
          ? rotatedShapePoints(center, size.widthM, size.heightM, rotationRef.current, shapeKindRef.current, cornerFracRef.current).map((p) => [p.lng, p.lat] as [number, number])
          : rotatedCropCorners(center, size.widthM, size.heightM, rotationRef.current).map((c) => [c.lng, c.lat] as [number, number]);
        spec.onPolygonChange(poly);
      }
    };

    const blockMapPlacement = () => {
      lastDragEndedAtRef.current = Date.now();
    };

    const beginHandleInteraction = (event?: L.LeafletEvent) => {
      handleInteractionRef.current = true;
      blockMapPlacement();
      if (handleInteractionTimerRef.current) {
        window.clearTimeout(handleInteractionTimerRef.current);
        handleInteractionTimerRef.current = null;
      }
      if ((event as L.LeafletMouseEvent | undefined)?.originalEvent) {
        L.DomEvent.stop((event as L.LeafletMouseEvent).originalEvent);
      }
      map.dragging.disable();
    };

    const endHandleInteraction = () => {
      blockMapPlacement();
      map.dragging.enable();
      if (handleInteractionTimerRef.current) {
        window.clearTimeout(handleInteractionTimerRef.current);
      }
      handleInteractionTimerRef.current = window.setTimeout(() => {
        handleInteractionRef.current = false;
        handleInteractionTimerRef.current = null;
      }, MAP_CLICK_SUPPRESS_AFTER_DRAG_MS);
    };

    const stopHandleClick = (event?: L.LeafletEvent) => {
      blockMapPlacement();
      if ((event as L.LeafletMouseEvent | undefined)?.originalEvent) {
        L.DomEvent.stop((event as L.LeafletMouseEvent).originalEvent);
      }
    };

    const handleRectangleDown = (event: L.LeafletMouseEvent) => {
      const bounds = currentBoundsRef.current ?? initialBounds;
      const size = boundsSizeMeters(bounds);
      dragStateRef.current = {
        startPoint: map.latLngToContainerPoint(event.latlng),
        startCenter: bounds.getCenter(),
        widthM: size.widthM,
        heightM: size.heightM,
      };
      map.dragging.disable();
      L.DomEvent.stop(event);
    };

    const handleMove = (event: L.LeafletMouseEvent) => {
      const state = dragStateRef.current;
      if (!state) return;
      const point = map.latLngToContainerPoint(event.latlng);
      const startCenterPoint = map.latLngToContainerPoint(state.startCenter);
      const nextCenter = map.containerPointToLatLng([
        startCenterPoint.x + point.x - state.startPoint.x,
        startCenterPoint.y + point.y - state.startPoint.y,
      ]);
      updateBounds(boundsFromCenterMeters(nextCenter, state.widthM, state.heightM));
    };

    const handleEnd = () => {
      if (!dragStateRef.current) return;
      dragStateRef.current = null;
      blockMapPlacement();
      map.dragging.enable();
    };

    const handleResize = () => {
      const current = currentBoundsRef.current ?? initialBounds;
      const center = current.getCenter();
      const corner = handle.getLatLng();
      const offset = localOffsetFromCenterMeters(center, corner);
      const angle = -(rotationRef.current * Math.PI) / 180;
      const localX = offset.x * Math.cos(angle) - offset.y * Math.sin(angle);
      const widthM = Math.min(Math.max(Math.abs(localX) * 2, Math.min(80, safeSize.widthM)), safeSize.widthM);
      updateBounds(boundsFromCenterMeters(center, widthM, widthM / aspect));
    };

    const handleRotate = () => {
      const current = currentBoundsRef.current ?? initialBounds;
      const center = current.getCenter();
      const offset = localOffsetFromCenterMeters(center, rotateHandle.getLatLng());
      const next = normalizeAngle((Math.atan2(offset.y, offset.x) * 180) / Math.PI - 90);
      rotationRef.current = next;
      spec.onRotationChange?.(next);
      syncDecorations(current, next);
      // Emit polygon з новим кутом — без цього preview не оновлюється під час drag.
      const size = boundsSizeMeters(current);
      const corners = rotatedCropCorners(center, size.widthM, size.heightM, next);
      spec.onPolygonChange?.(corners.map((c) => [c.lng, c.lat]));
    };

    const handleMapClick = (event: L.LeafletMouseEvent) => {
      if (handleInteractionRef.current) {
        L.DomEvent.stop(event);
        return;
      }
      if (Date.now() - lastDragEndedAtRef.current < MAP_CLICK_SUPPRESS_AFTER_DRAG_MS) return;
      const current = currentBoundsRef.current ?? initialBounds;
      const size = boundsSizeMeters(current);
      const widthM = Math.min(Math.max(size.widthM, Math.min(80, safeSize.widthM)), safeSize.widthM);
      updateBounds(boundsFromCenterMeters(event.latlng, widthM, widthM / aspect));
    };

    shape.on("mousedown", handleRectangleDown);
    shape.on("touchstart", handleRectangleDown as any);
    map.on("mousemove", handleMove);
    map.on("touchmove", handleMove as any);
    map.on("mouseup", handleEnd);
    map.on("touchend", handleEnd);
    map.on("click", handleMapClick);
    handle.on("mousedown", beginHandleInteraction);
    handle.on("touchstart", beginHandleInteraction);
    handle.on("mouseup", endHandleInteraction);
    handle.on("touchend", endHandleInteraction);
    handle.on("dragstart", beginHandleInteraction);
    handle.on("drag", handleResize);
    handle.on("dragend", endHandleInteraction);
    handle.on("click", stopHandleClick);
    rotateHandle.on("mousedown", beginHandleInteraction);
    rotateHandle.on("touchstart", beginHandleInteraction);
    rotateHandle.on("mouseup", endHandleInteraction);
    rotateHandle.on("touchend", endHandleInteraction);
    rotateHandle.on("dragstart", beginHandleInteraction);
    rotateHandle.on("drag", handleRotate);
    rotateHandle.on("dragend", endHandleInteraction);
    rotateHandle.on("click", stopHandleClick);

    return () => {
      if (handleInteractionTimerRef.current) {
        window.clearTimeout(handleInteractionTimerRef.current);
        handleInteractionTimerRef.current = null;
      }
      shape.off("mousedown", handleRectangleDown);
      shape.off("touchstart", handleRectangleDown as any);
      map.off("mousemove", handleMove);
      map.off("touchmove", handleMove as any);
      map.off("mouseup", handleEnd);
      map.off("touchend", handleEnd);
      map.off("click", handleMapClick);
      handle.off("mousedown", beginHandleInteraction);
      handle.off("touchstart", beginHandleInteraction);
      handle.off("mouseup", endHandleInteraction);
      handle.off("touchend", endHandleInteraction);
      handle.off("dragstart", beginHandleInteraction);
      handle.off("drag", handleResize);
      handle.off("dragend", endHandleInteraction);
      handle.off("click", stopHandleClick);
      rotateHandle.off("mousedown", beginHandleInteraction);
      rotateHandle.off("touchstart", beginHandleInteraction);
      rotateHandle.off("mouseup", endHandleInteraction);
      rotateHandle.off("touchend", endHandleInteraction);
      rotateHandle.off("dragstart", beginHandleInteraction);
      rotateHandle.off("drag", handleRotate);
      rotateHandle.off("dragend", endHandleInteraction);
      rotateHandle.off("click", stopHandleClick);
      shape.remove();
      handle.remove();
      rotateHandle.remove();
      label.remove();
    };
  }, [map, safeSize, setSelectedArea, spec.aspectRatio, spec.onRotationChange]);

  return null;
}


function MapViewUpdater({ center }: { center: [number, number] }) {
  const map = useMap();
  useEffect(() => {
    map.flyTo(center, 13);
  }, [center, map]);
  return null;
}

/** Keeps Leaflet sized correctly when its container changes (mobile tab
 *  switches map↔preview↔settings hide/show the map → stale 0/small size →
 *  grey/half-loaded tiles). ResizeObserver + a couple of delayed nudges fix it. */
function InvalidateOnResize() {
  const map = useMap();
  useEffect(() => {
    const el = map.getContainer();
    const fix = () => map.invalidateSize({ animate: false });
    const ro = new ResizeObserver(fix);
    ro.observe(el);
    const t1 = setTimeout(fix, 150);
    const t2 = setTimeout(fix, 600);
    return () => { ro.disconnect(); clearTimeout(t1); clearTimeout(t2); };
  }, [map]);
  return null;
}

interface MapSelectorProps {
  center?: [number, number];
  keychainCrop?: KeychainCropSpec;
}

export function MapSelector({ center = [50.4501, 30.5234], keychainCrop }: MapSelectorProps) {
  const [tileMode, setTileMode] = useState<"map" | "satellite">("map");
  const { selectedArea } = useGenerationStore();
  const isKeychainCrop = Boolean(keychainCrop);
  const mapInstanceKey = useMemo(
    () => `${center[0].toFixed(5)}:${center[1].toFixed(5)}:${isKeychainCrop ? "keychain" : "draw"}`,
    [center, isKeychainCrop],
  );
  const cropMetrics = useMemo(() => {
    if (!keychainCrop || !selectedArea) return null;
    const size = boundsSizeMeters(selectedArea);
    const metersPerMm = Math.max(
      size.widthM / Math.max(keychainCrop.mapWidthMm, 1),
      size.heightM / Math.max(keychainCrop.mapHeightMm, 1),
    );
    return {
      widthM: size.widthM,
      heightM: size.heightM,
      detailM: metersPerMm * 0.4,
      isSafe: metersPerMm <= keychainCrop.maxMetersPerMm,
    };
  }, [keychainCrop, selectedArea]);

  return (
    <div className="relative h-full w-full" style={{ minHeight: '100%' }}>
      <MapContainer
        key={mapInstanceKey}
        center={center} // Initial center
        zoom={13}
        zoomControl={false}
        style={{ height: "100%", width: "100%", minHeight: "100%" }}
        className="w-full h-full"
      >
        <ZoomControl position="bottomleft" />
        {tileMode === "satellite" ? (
          <TileLayer
            attribution='Tiles &copy; Esri'
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          />
        ) : (
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
        )}
        {keychainCrop ? <KeychainCropOverlay spec={keychainCrop} /> : <DrawControl />}
        <MapViewUpdater center={center} />
        <InvalidateOnResize />
      </MapContainer>
      <div
        className="pointer-events-auto absolute left-2 top-2 flex overflow-hidden rounded-full border border-white/50 bg-[#050a18]/85 p-0.5 shadow-[0_8px_20px_rgba(15,23,42,0.22)] backdrop-blur"
        style={{ zIndex: 10_000 }}
      >
        <button
          type="button"
          onClick={() => setTileMode("map")}
          className={`min-h-[30px] rounded-full px-2.5 text-[11px] font-semibold transition ${tileMode === "map" ? "bg-white text-[#050a18]" : "text-white/80"}`}
        >
          Карта
        </button>
        <button
          type="button"
          onClick={() => setTileMode("satellite")}
          className={`min-h-[30px] rounded-full px-2.5 text-[11px] font-semibold transition ${tileMode === "satellite" ? "bg-white text-[#050a18]" : "text-white/80"}`}
        >
          Супутник
        </button>
      </div>
      {keychainCrop ? (
        <div
          className="pointer-events-auto absolute right-2 top-2 flex items-center overflow-hidden rounded-full border border-white/50 bg-[#050a18]/85 p-0.5 shadow-[0_8px_20px_rgba(15,23,42,0.22)] backdrop-blur"
          style={{ zIndex: 10_000 }}
        >
          <button
            type="button"
            onClick={() => keychainCrop.onRotationChange?.(normalizeAngle((keychainCrop.rotationDeg || 0) - 15))}
            className="min-h-[30px] px-2 text-sm font-black text-white/90 transition hover:bg-white/10"
            aria-label="−15°" title="−15°"
          >
            ↺
          </button>
          <div className="grid min-w-[40px] place-items-center px-0.5 text-[11px] font-bold text-white tabular-nums">
            {normalizeAngle(keychainCrop.rotationDeg || 0)}°
          </div>
          <button
            type="button"
            onClick={() => keychainCrop.onRotationChange?.(normalizeAngle((keychainCrop.rotationDeg || 0) + 15))}
            className="min-h-[30px] px-2 text-sm font-black text-white/90 transition hover:bg-white/10"
            aria-label="+15°" title="+15°"
          >
            ↻
          </button>
        </div>
      ) : null}
      {keychainCrop ? (
        <div
          className="pointer-events-none absolute inset-x-3 bottom-3 grid gap-2 sm:inset-x-auto sm:right-3 sm:w-[280px]"
          style={{ zIndex: 10_000 }}
        >
          <div className="hidden rounded-[18px] border border-white/45 bg-[#050a18]/86 px-3 py-2 text-white shadow-[0_12px_28px_rgba(15,23,42,0.22)] backdrop-blur sm:block">
            <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-white/65">Область друку</div>
            <div className="mt-1 text-xs font-semibold">
              Клік ставить рамку. Бірюзовий квадрат змінює розмір, кругла ручка ⟳ крутить форму.
            </div>
          </div>
          {cropMetrics ? (
            <div className={`rounded-[18px] border px-3 py-2 text-xs font-semibold shadow-[0_12px_28px_rgba(15,23,42,0.22)] backdrop-blur ${
              cropMetrics.isSafe
                ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                : "border-red-200 bg-red-50 text-red-700"
            }`}>
              {Math.round(cropMetrics.widthM)} x {Math.round(cropMetrics.heightM)} м · 0.4 мм = ~{cropMetrics.detailM.toFixed(1)} м
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

