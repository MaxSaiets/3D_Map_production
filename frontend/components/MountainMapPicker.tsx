"use client";

import { useEffect, useMemo } from "react";
import { MapContainer, TileLayer, Rectangle, Marker, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

/**
 * Мапа для режиму «Гори»: клік = центр моделі, квадрат = ділянка area_km. Топографічна підкладка
 * (OpenTopoMap, ODbL/CC-BY-SA) — на ній гори читаються, на OSM-стандарті ні. Маркер можна тягнути.
 */
const ICON = L.divIcon({
  className: "",
  html: '<div style="width:14px;height:14px;border-radius:50%;background:#0f766e;border:2px solid #fff;box-shadow:0 0 0 2px rgba(15,118,110,.35)"></div>',
  iconSize: [14, 14], iconAnchor: [7, 7],
});

function metersPerDegLng(lat: number) { return 111_320 * Math.max(0.05, Math.cos((lat * Math.PI) / 180)); }

function squareBounds(lat: number, lon: number, areaKm: number): L.LatLngBoundsExpression {
  const half = (areaKm * 1000) / 2;
  const dLat = half / 111_320; const dLng = half / metersPerDegLng(lat);
  return [[lat - dLat, lon - dLng], [lat + dLat, lon + dLng]];
}

function Events({ onPick }: { onPick: (lat: number, lon: number) => void }) {
  useMapEvents({ click: (e) => onPick(e.latlng.lat, e.latlng.lng) });
  return null;
}

function FlyTo({ lat, lon, areaKm }: { lat: number; lon: number; areaKm: number }) {
  const map = useMap();
  useEffect(() => {
    const b = L.latLngBounds(squareBounds(lat, lon, areaKm) as L.LatLngBoundsLiteral);
    map.fitBounds(b.pad(0.6), { animate: true, maxZoom: 14 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lat, lon]);
  return null;
}

export default function MountainMapPicker({
  lat, lon, areaKm, onPick, height = 360,
}: { lat: number | null; lon: number | null; areaKm: number; onPick: (lat: number, lon: number) => void; height?: number }) {
  const center = useMemo<[number, number]>(() => [lat ?? 48.16, lon ?? 24.5], [lat, lon]);
  return (
    <div className="overflow-hidden rounded-2xl border border-[var(--surface-border)]" style={{ height }} data-testid="mountain-map">
      <MapContainer center={center} zoom={lat != null ? 12 : 6} style={{ height: "100%", width: "100%" }} scrollWheelZoom>
        <TileLayer
          url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
          maxZoom={17}
          attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>, SRTM · © <a href="https://opentopomap.org">OpenTopoMap</a> (CC-BY-SA)'
        />
        <Events onPick={onPick} />
        {lat != null && lon != null && (
          <>
            <FlyTo lat={lat} lon={lon} areaKm={areaKm} />
            <Rectangle bounds={squareBounds(lat, lon, areaKm)} pathOptions={{ color: "#0f766e", weight: 2, fillOpacity: 0.08 }} />
            <Marker position={[lat, lon]} icon={ICON} draggable eventHandlers={{ dragend: (e) => { const p = (e.target as L.Marker).getLatLng(); onPick(p.lat, p.lng); } }} />
          </>
        )}
      </MapContainer>
    </div>
  );
}
