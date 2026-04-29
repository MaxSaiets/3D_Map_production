"use client";

import { useEffect, useRef } from "react";
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

export interface MapSelection {
  bounds: { north: number; south: number; east: number; west: number };
  polygonGeoJson?: any;
}

function boundsToPlain(bounds: L.LatLngBounds): MapSelection["bounds"] {
  return {
    north: bounds.getNorth(),
    south: bounds.getSouth(),
    east: bounds.getEast(),
    west: bounds.getWest(),
  };
}

function DrawControl({
  initialBounds,
  onSelectionChange,
}: {
  initialBounds?: MapSelection["bounds"];
  onSelectionChange?: (selection: MapSelection | null) => void;
}) {
  const map = useMap();
  const drawnItemsRef = useRef<L.FeatureGroup>(new L.FeatureGroup());
  const { setSelectedArea } = useGenerationStore();

  useEffect(() => {
    if (!map) return;

    map.addLayer(drawnItemsRef.current);

    if (initialBounds && drawnItemsRef.current.getLayers().length === 0) {
      const bounds = L.latLngBounds(
        [initialBounds.south, initialBounds.west],
        [initialBounds.north, initialBounds.east],
      );
      const rectangle = L.rectangle(bounds, {
        color: "#c96745",
        weight: 1.5,
        dashArray: "6 4",
        fillColor: "#c96745",
        fillOpacity: 0.08,
      });
      drawnItemsRef.current.addLayer(rectangle);
      setSelectedArea(bounds);
      onSelectionChange?.({ bounds: boundsToPlain(bounds), polygonGeoJson: rectangle.toGeoJSON().geometry });
      map.fitBounds(bounds.pad(0.45));
    }

    const drawControl = new L.Control.Draw({
      position: "topleft",
      draw: {
        rectangle: {
          shapeOptions: {
            color: "#c96745",
            weight: 1.5,
            dashArray: "6 4",
            fillColor: "#c96745",
            fillOpacity: 0.08,
          },
        },
        polygon: {
          shapeOptions: {
            color: "#c96745",
            weight: 1.5,
            fillColor: "#c96745",
            fillOpacity: 0.08,
          },
        },
        circle: {
          shapeOptions: {
            color: "#c96745",
            weight: 1.5,
            fillColor: "#c96745",
            fillOpacity: 0.08,
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
      drawnItemsRef.current.clearLayers();
      drawnItemsRef.current.addLayer(layer);

      // Отримуємо bounds обраної області
      if ("getBounds" in (layer as any) && typeof (layer as any).getBounds === "function") {
        const bounds = (layer as L.Rectangle | L.Polygon | L.Circle).getBounds();
        setSelectedArea(bounds);
        const polygonGeoJson = "toGeoJSON" in layer ? layer.toGeoJSON().geometry : undefined;
        onSelectionChange?.({ bounds: boundsToPlain(bounds), polygonGeoJson });
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
          const polygonGeoJson = "toGeoJSON" in layer ? (layer as any).toGeoJSON().geometry : undefined;
          onSelectionChange?.({ bounds: boundsToPlain(bounds), polygonGeoJson });
        }
      }
    };

    const handleDrawDeleted = () => {
      setSelectedArea(null);
      onSelectionChange?.(null);
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
  }, [map, setSelectedArea, initialBounds, onSelectionChange]);

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
  initialBounds?: MapSelection["bounds"];
  onSelectionChange?: (selection: MapSelection | null) => void;
}

export function MapSelector({ center = [50.4501, 30.5234], initialBounds, onSelectionChange }: MapSelectorProps) {
  return (
    <div className="h-full w-full" style={{ minHeight: "100%" }}>
      <MapContainer
        center={center} // Initial center
        zoom={13}
        style={{ height: "100%", width: "100%", minHeight: "100%" }}
        className="w-full h-full"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <DrawControl initialBounds={initialBounds} onSelectionChange={onSelectionChange} />
        <MapViewUpdater center={center} />
      </MapContainer>
    </div>
  );
}

