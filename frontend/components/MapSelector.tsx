"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { MapContainer, TileLayer, useMap } from "react-leaflet";
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

function DrawControl() {
  const map = useMap();
  const drawnItemsRef = useRef<L.FeatureGroup>(new L.FeatureGroup());
  const { setSelectedArea } = useGenerationStore();

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

    const handleDrawCreated = (e: any) => {
      const layer = e.layer;
      drawnItemsRef.current.addLayer(layer);

      // Отримуємо bounds обраної області
      if ("getBounds" in (layer as any) && typeof (layer as any).getBounds === "function") {
        const bounds = (layer as L.Rectangle | L.Polygon | L.Circle).getBounds();
        setSelectedArea(bounds);
      } else {
        // На випадок неочікуваних layer типів
        console.warn("Draw created layer does not support getBounds:", layer);
      }
    };

    const handleDrawEdited = () => {
      const layers = drawnItemsRef.current.getLayers();
      if (layers.length > 0) {
        const layer = layers[0] as L.Layer;
        if ("getBounds" in layer) {
          const bounds = (layer as L.Rectangle | L.Polygon | L.Circle).getBounds();
          setSelectedArea(bounds);
        }
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
  }, [map, setSelectedArea]);

  return null;
}

type KeychainCropSpec = {
  aspectRatio: number;
  maxMetersPerMm: number;
  mapWidthMm: number;
  mapHeightMm: number;
};

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
  const rectangleRef = useRef<L.Rectangle | null>(null);
  const resizeHandleRef = useRef<L.Marker | null>(null);
  const labelRef = useRef<L.Marker | null>(null);
  const lastDragEndedAtRef = useRef(0);
  const dragStateRef = useRef<{
    startPoint: L.Point;
    startCenter: L.LatLng;
    widthM: number;
    heightM: number;
  } | null>(null);

  const safeSize = useMemo(() => safeCropMeters(spec), [spec]);
  const northCenter = (bounds: L.LatLngBounds) => L.latLng(bounds.getNorth(), bounds.getCenter().lng);

  useEffect(() => {
    if (!map) return;

    const initialSelectedArea = initialSelectedAreaRef.current;
    const existingCenter = initialSelectedArea?.getCenter() ?? map.getCenter();
    const existingSize = initialSelectedArea ? boundsSizeMeters(initialSelectedArea) : safeSize;
    const aspect = Math.max(spec.aspectRatio, 0.2);
    const unclampedWidth = Math.min(existingSize.widthM || safeSize.widthM, safeSize.widthM);
    const widthM = Math.max(Math.min(unclampedWidth, safeSize.widthM), Math.min(safeSize.widthM, 80));
    const heightM = Math.min(widthM / aspect, safeSize.heightM);
    const initialBounds = boundsFromCenterMeters(existingCenter, widthM, heightM);

    const rectangle = L.rectangle(initialBounds, {
      color: "#14b8a6",
      weight: 2,
      fillColor: "#14b8a6",
      fillOpacity: 0.14,
      dashArray: "8 6",
      interactive: true,
    }).addTo(map);
    rectangleRef.current = rectangle;
    setSelectedArea(initialBounds);

    const handleIcon = L.divIcon({
      className: "",
      html: '<div style="width:30px;height:30px;border-radius:10px;background:#14b8a6;border:3px solid white;box-shadow:0 10px 24px rgba(15,23,42,.25);"></div>',
      iconSize: [30, 30],
      iconAnchor: [15, 15],
    });
    const labelIcon = L.divIcon({
      className: "",
      html: '<div style="padding:6px 9px;border-radius:999px;background:rgba(5,10,24,.82);border:1px solid rgba(255,255,255,.3);color:white;font:700 11px/1.1 system-ui;white-space:nowrap;">клік = поставити</div>',
      iconSize: [116, 28],
      iconAnchor: [58, 36],
    });

    const handle = L.marker(initialBounds.getSouthEast(), {
      icon: handleIcon,
      draggable: true,
      zIndexOffset: 800,
    }).addTo(map);
    resizeHandleRef.current = handle;

    const label = L.marker(northCenter(initialBounds), {
      icon: labelIcon,
      interactive: false,
      zIndexOffset: 700,
    }).addTo(map);
    labelRef.current = label;

    const syncDecorations = (bounds: L.LatLngBounds) => {
      resizeHandleRef.current?.setLatLng(bounds.getSouthEast());
      labelRef.current?.setLatLng(northCenter(bounds));
    };

    const updateBounds = (bounds: L.LatLngBounds) => {
      rectangle.setBounds(bounds);
      syncDecorations(bounds);
      setSelectedArea(bounds);
    };

    const handleRectangleDown = (event: L.LeafletMouseEvent) => {
      const bounds = rectangle.getBounds();
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
      lastDragEndedAtRef.current = Date.now();
      map.dragging.enable();
    };

    const handleResize = () => {
      const current = rectangle.getBounds();
      const center = current.getCenter();
      const corner = handle.getLatLng();
      const dxM = Math.abs(corner.lng - center.lng) * metersPerDegreeLng(center.lat) * 2;
      const widthM = Math.min(Math.max(dxM, Math.min(80, safeSize.widthM)), safeSize.widthM);
      updateBounds(boundsFromCenterMeters(center, widthM, widthM / aspect));
    };

    const handleMapClick = (event: L.LeafletMouseEvent) => {
      if (Date.now() - lastDragEndedAtRef.current < 180) return;
      const current = rectangle.getBounds();
      const size = boundsSizeMeters(current);
      const widthM = Math.min(Math.max(size.widthM, Math.min(80, safeSize.widthM)), safeSize.widthM);
      updateBounds(boundsFromCenterMeters(event.latlng, widthM, widthM / aspect));
    };

    rectangle.on("mousedown", handleRectangleDown);
    rectangle.on("touchstart", handleRectangleDown as any);
    map.on("mousemove", handleMove);
    map.on("touchmove", handleMove as any);
    map.on("mouseup", handleEnd);
    map.on("touchend", handleEnd);
    map.on("click", handleMapClick);
    handle.on("drag", handleResize);

    return () => {
      rectangle.off("mousedown", handleRectangleDown);
      rectangle.off("touchstart", handleRectangleDown as any);
      map.off("mousemove", handleMove);
      map.off("touchmove", handleMove as any);
      map.off("mouseup", handleEnd);
      map.off("touchend", handleEnd);
      map.off("click", handleMapClick);
      handle.off("drag", handleResize);
      rectangle.remove();
      handle.remove();
      label.remove();
    };
  }, [map, safeSize, setSelectedArea, spec.aspectRatio]);

  return null;
}


function MapViewUpdater({ center }: { center: [number, number] }) {
  const map = useMap();
  useEffect(() => {
    map.flyTo(center, 13);
  }, [center, map]);
  return null;
}

interface MapSelectorProps {
  center?: [number, number];
  keychainCrop?: KeychainCropSpec;
}

export function MapSelector({ center = [50.4501, 30.5234], keychainCrop }: MapSelectorProps) {
  const [tileMode, setTileMode] = useState<"map" | "satellite">("map");

  return (
    <div className="relative h-full w-full" style={{ minHeight: '100%' }}>
      <MapContainer
        center={center} // Initial center
        zoom={13}
        style={{ height: "100%", width: "100%", minHeight: "100%" }}
        className="w-full h-full"
      >
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
      </MapContainer>
      <div className="absolute left-3 top-3 z-[500] flex overflow-hidden rounded-full border border-white/50 bg-[#050a18]/85 p-1 shadow-[0_12px_28px_rgba(15,23,42,0.22)] backdrop-blur">
        <button
          type="button"
          onClick={() => setTileMode("map")}
          className={`rounded-full px-3 py-1.5 text-xs font-semibold transition ${tileMode === "map" ? "bg-white text-[#050a18]" : "text-white/80"}`}
        >
          Карта
        </button>
        <button
          type="button"
          onClick={() => setTileMode("satellite")}
          className={`rounded-full px-3 py-1.5 text-xs font-semibold transition ${tileMode === "satellite" ? "bg-white text-[#050a18]" : "text-white/80"}`}
        >
          Супутник
        </button>
      </div>
    </div>
  );
}

