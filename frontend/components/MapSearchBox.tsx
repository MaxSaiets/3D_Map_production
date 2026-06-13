"use client";

import { useEffect, useRef, useState } from "react";
import { Search, Loader2, LocateFixed, X } from "lucide-react";
import { geocodeSearch, reverseGeocode, type GeoResult } from "@/lib/geocode";

/**
 * Пошук локації над картою (обидва конструктори). Знаходить будь-яке місто/
 * адресу через Nominatim і шле подію `monadruk:map-goto` {lat,lon,label}, яку
 * слухають оверлеї карти (фокусують карту + ставлять зону туди).
 * Кнопка 📍 — геолокація браузера.
 */
export function MapSearchBox() {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<GeoResult[]>([]);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [geoBusy, setGeoBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const debounceRef = useRef<number | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const runSearch = (value: string) => {
    setQ(value);
    if (debounceRef.current) window.clearTimeout(debounceRef.current);
    if (value.trim().length < 3) { setResults([]); setOpen(false); return; }
    debounceRef.current = window.setTimeout(async () => {
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      setBusy(true);
      const res = await geocodeSearch(value, ctrl.signal);
      setBusy(false);
      setResults(res);
      setOpen(res.length > 0);
    }, 450);
  };

  const goto = (lat: number, lon: number, label: string) => {
    setOpen(false);
    setQ(label);
    window.dispatchEvent(new CustomEvent("monadruk:map-goto", { detail: { lat, lon, label } }));
  };

  const useMyLocation = () => {
    if (!navigator.geolocation) return;
    setGeoBusy(true);
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const { latitude, longitude } = pos.coords;
        const name = await reverseGeocode(latitude, longitude);
        setGeoBusy(false);
        goto(latitude, longitude, name || "Моє місце");
      },
      () => setGeoBusy(false),
      { enableHighAccuracy: true, timeout: 8000 },
    );
  };

  return (
    <div
      ref={boxRef}
      className="pointer-events-auto absolute left-1/2 top-2 w-[min(360px,calc(100%-110px))] -translate-x-1/2"
      style={{ zIndex: 10_000 }}
      data-testid="map-search"
    >
      <div className="flex items-center gap-1 rounded-full border border-white/50 bg-[#050a18]/85 px-2 py-1 shadow-[0_8px_20px_rgba(15,23,42,0.28)] backdrop-blur">
        <Search size={15} className="ml-1 shrink-0 text-white/70" />
        <input
          value={q}
          onChange={(e) => runSearch(e.target.value)}
          onFocus={() => { if (results.length) setOpen(true); }}
          placeholder="Знайти місто, село чи адресу…"
          className="min-w-0 flex-1 bg-transparent px-1 py-1 text-[13px] font-medium text-white placeholder:text-white/55 outline-none"
        />
        {busy && <Loader2 size={15} className="shrink-0 animate-spin text-white/70" />}
        {q && !busy && (
          <button type="button" onClick={() => { setQ(""); setResults([]); setOpen(false); }} className="shrink-0 text-white/60 hover:text-white" aria-label="Очистити">
            <X size={15} />
          </button>
        )}
        <button
          type="button"
          onClick={useMyLocation}
          className="ml-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-full bg-white/15 text-white transition hover:bg-white/25"
          title="Моє місцеположення"
          aria-label="Моє місцеположення"
        >
          {geoBusy ? <Loader2 size={14} className="animate-spin" /> : <LocateFixed size={14} />}
        </button>
      </div>
      {open && results.length > 0 && (
        <ul className="mt-1 overflow-hidden rounded-2xl border border-white/15 bg-[#0a1020]/95 shadow-[0_18px_40px_rgba(15,23,42,0.4)] backdrop-blur">
          {results.map((r, i) => (
            <li key={i}>
              <button
                type="button"
                onClick={() => goto(r.lat, r.lon, r.label)}
                className="block w-full px-3 py-2 text-left text-[12px] leading-tight text-white/90 transition hover:bg-white/10"
              >
                <span className="font-semibold">{r.label}</span>
                <span className="block truncate text-[11px] text-white/55">{r.full}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
