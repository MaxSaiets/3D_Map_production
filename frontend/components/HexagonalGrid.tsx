"use client";

import { useState, useEffect, useRef } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet-draw";

type GBounds = { north: number; south: number; east: number; west: number };

/** Компактна стартова зона сітки (~3км) довкола центру bbox міста — щоб дефолтна
 *  сітка мала ~100 видимих клітин, а не тисячі на весь bbox міста. */
function defaultGridArea(b: GBounds, halfKm = 1.5): GBounds {
  const cLat = (b.north + b.south) / 2;
  const cLon = (b.east + b.west) / 2;
  const dLat = (halfKm * 1000) / 111_320;
  const dLon = (halfKm * 1000) / (111_320 * Math.max(Math.cos((cLat * Math.PI) / 180), 0.2));
  return { north: cLat + dLat, south: cLat - dLat, east: cLon + dLon, west: cLon - dLon };
}

// Lets the user draw a rectangle to choose the AREA the grid fills (instead of
// the whole city). On draw it reports the rectangle's bounds upward.
function GridAreaDraw({ onArea }: { onArea: (b: GBounds) => void }) {
  const map = useMap();
  useEffect(() => {
    if (!map) return;
    const group = new L.FeatureGroup();
    map.addLayer(group);
    const control = new (L as any).Control.Draw({
      position: "topright",
      draw: { rectangle: { shapeOptions: { color: "#0f766e", weight: 2 } },
        polygon: false, circle: false, marker: false, circlemarker: false, polyline: false },
      edit: { featureGroup: group, remove: true },
    });
    map.addControl(control);
    const onCreated = (e: any) => {
      group.clearLayers();
      group.addLayer(e.layer);
      const b = e.layer.getBounds();
      onArea({ north: b.getNorth(), south: b.getSouth(), east: b.getEast(), west: b.getWest() });
    };
    map.on((L as any).Draw.Event.CREATED, onCreated);
    return () => {
      map.off((L as any).Draw.Event.CREATED, onCreated);
      try { map.removeControl(control); map.removeLayer(group); } catch { /* no-op */ }
    };
  }, [map, onArea]);
  return null;
}

// Компонент для автоматичного fitBounds (тільки при першому завантаженні)
function MapBounds({ bounds }: { bounds: { north: number; south: number; east: number; west: number } }) {
  const map = useMap();
  const hasFittedRef = useRef(false);

  useEffect(() => {
    if (bounds && map && !hasFittedRef.current) {
      try {
        map.fitBounds([
          [bounds.south, bounds.west],
          [bounds.north, bounds.east],
        ] as L.LatLngBoundsExpression, {
          padding: [20, 20],
          maxZoom: 15,
        });
        hasFittedRef.current = true; // Виконуємо тільки один раз
      } catch (e) {
        console.error("Помилка fitBounds:", e);
      }
    }
  }, [map, bounds]);
  return null;
}

interface HexagonalGridProps {
  bounds: {
    north: number;
    south: number;
    east: number;
    west: number;
  };
  onZonesSelected: (zones: any[]) => void;
  gridType?: "hexagonal" | "square" | "circle";
  hexSizeM?: number;
  /** Notifies the parent when the user draws/clears the grid area (large zone). */
  onAreaChange?: (area: GBounds | null) => void;
  /** Pre-set the grid area (e.g. when reopening a saved grid from history). */
  initialArea?: GBounds | null;
}

// Стилі для шестикутників
const defaultStyle = {
  color: "#3388ff",
  weight: 1.5,
  opacity: 0.8,
  fillOpacity: 0.15,
};

const selectedStyle = {
  color: "#dc2626",
  weight: 3,
  opacity: 1,
  fillOpacity: 0.6,
  fillColor: "#ef4444",
};

const hoverStyle = {
  color: "#10b981",
  weight: 2.5,
  opacity: 1,
  fillOpacity: 0.3,
  fillColor: "#34d399",
};

export default function HexagonalGrid({
  bounds,
  onZonesSelected,
  gridType: externalGridType = "hexagonal",
  hexSizeM: externalHexSizeM = 300.0,
  onAreaChange,
  initialArea = null,
}: HexagonalGridProps) {
  const normalizeId = (id: any): string => String(id ?? "");
  const [hexGrid, setHexGrid] = useState<any>(null);
  const [selectedZones, setSelectedZones] = useState<Set<string>>(new Set());
  // Ordered selection (so zones can be generated and previewed "one after another")
  const [selectedOrder, setSelectedOrder] = useState<string[]>([]);
  const [hoveredZone, setHoveredZone] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  // User-drawn area (large zone) the grid fills. Falls back to the city bounds.
  const [drawnBounds, setDrawnBounds] = useState<GBounds | null>(null);
  const drawnBoundsRef = useRef<GBounds | null>(null);
  const [isValid, setIsValid] = useState(true);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  // IMPORTANT: Leaflet feature handlers are attached once and keep stale React closures.
  // Use refs as the source of truth for click/hover handlers.
  const hexGridRef = useRef<any>(null);
  const selectedZonesRef = useRef<Set<string>>(new Set());
  const selectedOrderRef = useRef<string[]>([]);
  const hoveredZoneRef = useRef<string | null>(null);
  const onZonesSelectedRef = useRef(onZonesSelected);

  useEffect(() => {
    hexGridRef.current = hexGrid;
  }, [hexGrid]);
  useEffect(() => {
    selectedZonesRef.current = new Set(selectedZones);
  }, [selectedZones]);
  useEffect(() => {
    selectedOrderRef.current = [...selectedOrder];
  }, [selectedOrder]);
  useEffect(() => {
    hoveredZoneRef.current = hoveredZone;
  }, [hoveredZone]);
  useEffect(() => {
    onZonesSelectedRef.current = onZonesSelected;
  }, [onZonesSelected]);

  // Використовуємо зовнішні значення якщо передані, інакше внутрішні
  const [internalGridType, setInternalGridType] = useState<"hexagonal" | "square" | "circle">("hexagonal");
  // "Applied" params — те, що реально використовувалось при генерації
  const [appliedGridType, setAppliedGridType] = useState(externalGridType || "hexagonal");
  const [appliedHexSizeM, setAppliedHexSizeM] = useState(externalHexSizeM || 300.0);

  const gridType = externalGridType || internalGridType;
  const hexSizeM = externalHexSizeM || 300.0;

  // Чи є незастосовані зміни параметрів
  const hasPendingChanges = gridType !== appliedGridType || hexSizeM !== appliedHexSizeM;

  const generateGrid = async () => {
    if (isLoading) return; // Запобігаємо подвійній генерації

    // Запам'ятовуємо параметри які застосовуємо
    setAppliedGridType(gridType);
    setAppliedHexSizeM(hexSizeM);

    setIsLoading(true);
    setHexGrid(null); // Скидаємо попередню сітку

    try {

      const { api } = await import("@/lib/api");

      // Use the user-drawn area when present. Раніше fallback = ВЕСЬ bbox міста
      // (~44км) → тисячі клітин (8807 для Києва), що зливались у суцільну пляму
      // і користувач не розумів, що це окремі клітини. Тепер дефолт — компактна
      // зона ~3км довкола центру міста: ~100 видимих клітин, які легко обирати.
      // Юзер може намалювати власну зону прямокутником.
      const eb = drawnBoundsRef.current || defaultGridArea(bounds);
      if (!eb || eb.north <= eb.south || eb.east <= eb.west) {
        throw new Error(`Невірні координати bounds: north=${eb?.north}, south=${eb?.south}, east=${eb?.east}, west=${eb?.west}`);
      }


      const data = await api.generateHexagonalGrid({
        north: eb.north,
        south: eb.south,
        east: eb.east,
        west: eb.west,
        hex_size_m: hexSizeM,
        grid_type: gridType,
      });



      if (!data.geojson || !data.geojson.features || data.geojson.features.length === 0) {
        throw new Error("Сітка порожня або невалідна");
      }

      // Діагностика першого feature
      if (data.geojson.features.length > 0) {
        const firstFeature = data.geojson.features[0];
        const firstCoords = firstFeature?.geometry?.coordinates?.[0]?.[0];
        // Debug only in dev: console.log("[HexagonalGrid] first hex:", firstFeature?.id);
      }

      setHexGrid(data.geojson);
      setIsValid(data.is_valid);
      setValidationErrors(data.validation_errors || []);
    } catch (error: any) {
      console.error("Помилка генерації сітки:", error);
      const errorMessage = error.response?.data?.detail || error.message || String(error);
      alert("Помилка генерації сітки: " + errorMessage);
      setHexGrid(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleZoneClick = (zoneIdRaw: any) => {
    const zoneId = normalizeId(zoneIdRaw);
    const currentSelected = selectedZonesRef.current;
    const currentOrder = selectedOrderRef.current;

    if (!zoneId) {
      console.error("[HexagonalGrid] zoneId is empty!");
      return;
    }

    const nextSelected = new Set(currentSelected);
    let nextOrder = [...currentOrder];

    // Перемикаємо стан зони
    if (nextSelected.has(zoneId)) {
      nextSelected.delete(zoneId);
      nextOrder = nextOrder.filter((id) => id !== zoneId);

    } else {
      nextSelected.add(zoneId);
      // Add to the end to preserve click order
      if (!nextOrder.includes(zoneId)) nextOrder.push(zoneId);

    }
    // Sync refs immediately (so next click sees updated state even before React renders)
    selectedZonesRef.current = nextSelected;
    selectedOrderRef.current = nextOrder;
    setSelectedZones(nextSelected);
    setSelectedOrder(nextOrder);

    // Оновлюємо список вибраних зон у стабільному порядку (click-order),
    // щоб backend створював задачі у тій же послідовності.
    const featureById = new Map<string, any>();
    for (const f of (hexGridRef.current?.features || [])) {
      const fId = normalizeId(f.id || f.properties?.id);
      if (fId) featureById.set(fId, f);
    }
    const selectedFeatures = nextOrder.map((id) => featureById.get(id)).filter(Boolean);


    onZonesSelectedRef.current(selectedFeatures);
  };

  const handleSelectAll = () => {
    if (!hexGrid || !hexGrid.features) return;
    const all = (hexGrid.features || [])
      .map((f: any) => ({ id: normalizeId(f.id || f.properties?.id), feature: f }))
      .filter((x: any) => !!x.id);
    // Default order: by row/col if present (better UX for "in a row" selections), else original order
    all.sort((a: any, b: any) => {
      const ar = a.feature?.properties?.row;
      const br = b.feature?.properties?.row;
      const ac = a.feature?.properties?.col;
      const bc = b.feature?.properties?.col;
      if (ar != null && br != null && ar !== br) return ar - br;
      if (ac != null && bc != null && ac !== bc) return ac - bc;
      return String(a.id).localeCompare(String(b.id));
    });
    const allZoneIds = new Set<string>(all.map((x: any) => x.id as string));
    selectedZonesRef.current = allZoneIds;
    selectedOrderRef.current = all.map((x: any) => x.id);
    setSelectedZones(allZoneIds);
    setSelectedOrder(all.map((x: any) => x.id));
    onZonesSelectedRef.current(all.map((x: any) => x.feature));

  };

  const handleDeselectAll = () => {
    selectedZonesRef.current = new Set();
    selectedOrderRef.current = [];
    setSelectedZones(new Set());
    setSelectedOrder([]);
    onZonesSelectedRef.current([]);

  };

  const handleZoneHover = (zoneId: string | null) => {
    hoveredZoneRef.current = zoneId;
    setHoveredZone(zoneId);
  };

  const getZoneStyle = (zoneId: string) => {
    const zid = normalizeId(zoneId);
    if (!zid) {
      console.warn("[HexagonalGrid] getZoneStyle called with empty zoneId");
      return defaultStyle;
    }

    const isSelected = selectedZonesRef.current.has(zid);
    const isHovered = hoveredZoneRef.current === zid;

    if (isSelected) {
      return selectedStyle;
    }
    if (isHovered) {
      return hoverStyle;
    }
    return defaultStyle;
  };

  const center: [number, number] = [
    (bounds.north + bounds.south) / 2,
    (bounds.east + bounds.west) / 2,
  ];

  // Автоматично генеруємо сітку при першому відкритті або зміні bounds (міста)
  // НЕ перегенеруємо при зміні gridType/hexSizeM — для цього є кнопка "Застосувати"
  useEffect(() => {
    if (!bounds) return;
    if (bounds.north <= bounds.south || bounds.east <= bounds.west) return;
    if (!hexGrid && !isLoading) {
      const timer = setTimeout(() => { generateGrid(); }, 200);
      return () => clearTimeout(timer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bounds?.north, bounds?.south, bounds?.east, bounds?.west]);

  const handleAreaDraw = (b: GBounds) => {
    drawnBoundsRef.current = b;
    setDrawnBounds(b);
    onAreaChange?.(b);
    setSelectedZones(new Set());
    setSelectedOrder([]);
    setHexGrid(null);
    // regenerate the grid inside the drawn area
    setTimeout(() => generateGrid(), 50);
  };

  const resetArea = () => {
    drawnBoundsRef.current = null;
    setDrawnBounds(null);
    onAreaChange?.(null);
    setHexGrid(null);
    setTimeout(() => generateGrid(), 50);
  };

  // Apply a pre-set area (reopening a saved grid from history) once on mount.
  const appliedInitialAreaRef = useRef(false);
  useEffect(() => {
    if (appliedInitialAreaRef.current || !initialArea) return;
    appliedInitialAreaRef.current = true;
    drawnBoundsRef.current = initialArea;
    setDrawnBounds(initialArea);
    setHexGrid(null);
    setTimeout(() => generateGrid(), 80);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialArea]);

  const zoom = 11; // Оптимальний zoom для Києва

  return (
    <div className="w-full h-full flex flex-col">
      <div className="px-2 py-1.5 bg-white border-b border-gray-200 flex-shrink-0 shadow-sm">
        {isLoading ? (
          <div className="flex items-center gap-1.5 text-[11px]">
            <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-500"></div>
            <span className="text-gray-700">Генерація сітки...</span>
          </div>
        ) : hexGrid ? (
          <div className="space-y-1.5">
            {/* Чітка інструкція: раніше юзер бачив суцільну пляму клітин і не
                розумів, що їх треба КЛІКАТИ. Тепер — крок-за-кроком + легенда. */}
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] leading-4 text-gray-700">
              <span className="font-semibold text-[var(--accent-strong,#0f766e)]">
                {selectedZones.size === 0
                  ? "👆 Клікайте клітинки — вони стануть"
                  : `Обрано ${selectedZones.size} — далі «Згенерувати серію»`}
              </span>
              {selectedZones.size === 0 && (
                <span className="inline-flex items-center gap-1">
                  <span className="inline-block h-3 w-3 rounded-sm border border-red-600 bg-red-400/60" />
                  червоними
                </span>
              )}
              <span className="text-gray-400">·</span>
              <span title="Намалюйте прямокутник інструментом ▢ вгорі праворуч, щоб обмежити сітку своєю зоною">
                ▢ для своєї зони
              </span>
            </div>
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3 text-[11px]">
              <span className="font-medium text-gray-700">
                Клітинок: <span className="text-gray-900 font-semibold">{hexGrid.features.length}</span>
              </span>
              <span className="font-medium text-blue-700">
                Вибрано: <span className="text-blue-800 font-bold">{selectedZones.size}</span>
              </span>
              {selectedZones.size > 0 && (
                <span className="text-green-700 font-semibold">✓ Готово</span>
              )}
              {!isValid && validationErrors.length > 0 && (
                <span className="text-red-600 text-[10px]">
                  ⚠ {validationErrors.length} помилок
                </span>
              )}
            </div>
            <div className="flex items-center gap-1.5">
              {hasPendingChanges && (
                <button
                  onClick={generateGrid}
                  className="px-2 py-0.5 text-[10px] bg-orange-500 text-white rounded hover:bg-orange-600 transition-colors font-semibold"
                  title={`Застосувати нові параметри: ${gridType}, ${hexSizeM}м`}
                >
                  ↻ Застосувати
                </button>
              )}
              <button
                onClick={handleSelectAll}
                className="px-2 py-0.5 text-[10px] bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
                title="Вибрати всі зони"
              >
                Всі
              </button>
              <button
                onClick={handleDeselectAll}
                className="px-2 py-0.5 text-[10px] bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors"
                title="Зняти вибір з усіх зон"
              >
                Очистити
              </button>
              {drawnBounds ? (
                <button
                  onClick={resetArea}
                  className="px-2 py-0.5 text-[10px] bg-teal-600 text-white rounded hover:bg-teal-700 transition-colors"
                  title="Скинути намальовану зону — сітка повернеться до стартової зони в центрі міста"
                >
                  ⤢ Своя зона
                </button>
              ) : (
                <span className="px-2 py-0.5 text-[10px] text-teal-700" title="Намалюйте прямокутник на карті (значок ▢ праворуч), щоб сітка будувалась лише в ньому">
                  ▢ намалюйте зону
                </span>
              )}
            </div>
          </div>
          </div>
        ) : (
          <div className="text-[11px] text-gray-600">Генерація сітки...</div>
        )}
      </div>

      <div className="flex-1 relative min-h-0">
        <MapContainer
          center={center}
          zoom={zoom}
          style={{ height: "100%", width: "100%" }}
          scrollWheelZoom={true}
          whenReady={() => {
            // Карта готова
          }}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          />
          {/* Фітимо карту на ДЕФОЛТНУ зону сітки (~3км), а не весь bbox міста —
              інакше компактна сітка виглядала б крихітною плямою в центрі. */}
          <MapBounds bounds={drawnBounds || initialArea || defaultGridArea(bounds)} />
          <GridAreaDraw onArea={handleAreaDraw} />

          {hexGrid && hexGrid.features && hexGrid.features.length > 0 && (
            <GeoJSON
              // IMPORTANT: Do NOT remount on selection changes, otherwise the map/tiles can "jump back".
              key={`hex-grid-${hexGrid.features.length}-${gridType}-${hexSizeM}`}
              data={hexGrid}
              style={(feature) => {
                const zoneId = normalizeId(feature?.properties?.id || feature?.id);
                if (!zoneId) {
                  console.warn("[HexagonalGrid] Feature without ID in style function:", feature);
                  return defaultStyle;
                }
                const style = getZoneStyle(zoneId);
                return style;
              }}
              onEachFeature={(feature, layer) => {
                const zoneId = normalizeId(feature?.properties?.id || feature?.id);

                if (!zoneId) {
                  console.error("[HexagonalGrid] Feature without ID:", feature);
                  return;
                }



                // Зберігаємо посилання на layer для оновлення стилю
                (layer as any)._hexZoneId = zoneId;

                layer.on({
                  click: (e: L.LeafletMouseEvent) => {
                    e.originalEvent?.stopPropagation?.();
                    e.originalEvent?.preventDefault?.();

                    // Apply immediate visual feedback based on ref state (no stale closures)
                    const willSelect = !selectedZonesRef.current.has(zoneId);
                    handleZoneClick(zoneId);

                    // Оновлюємо стиль після кліку
                    setTimeout(() => {
                      // Use immediate decision first, then fallback to state-driven style
                      (layer as L.Path).setStyle(willSelect ? selectedStyle : defaultStyle);
                      // After state settles, sync with computed style
                      setTimeout(() => (layer as L.Path).setStyle(getZoneStyle(zoneId)), 0);
                    }, 0);
                  },
                  mouseover: () => {
                    handleZoneHover(zoneId);
                    (layer as L.Path).setStyle(hoverStyle);
                  },
                  mouseout: () => {
                    handleZoneHover(null);
                    (layer as L.Path).setStyle(getZoneStyle(zoneId));
                  },
                });

                // Додаємо popup з інформацією (тільки при наведенні, не при кліку)
                const props = feature.properties || {};
                const isSelected = selectedZones.has(zoneId);
                layer.bindTooltip(
                  `<b>Зона ${zoneId}</b><br/>Ряд: ${props.row}, Колонка: ${props.col}<br/>${isSelected ? '<span style="color: red; font-weight: bold;">✓ Вибрано</span>' : '<span style="color: gray;">Клікніть для вибору</span>'}`,
                  {
                    permanent: false,
                    direction: 'top',
                    offset: [0, -10],
                    className: 'zone-tooltip'
                  }
                );
              }}
            />
          )}
        </MapContainer>
      </div>
    </div>
  );
}

