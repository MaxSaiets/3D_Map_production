"use client";

import { useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import { Search, Loader2, LocateFixed, X } from "lucide-react";
import { geocodeSearch, reverseGeocode, type GeoResult } from "@/lib/geocode";

/**
 * Пошук локації над картою (обидва конструктори). Знаходить будь-яке місто/
 * адресу через Nominatim і шле подію `monadruk:map-goto` {lat,lon,label}, яку
 * слухають оверлеї карти (фокусують карту + ставлять зону туди).
 * Кнопка 📍 — геолокація браузера.
 */
export function MapSearchBox({ variant = "map" }: { variant?: "map" | "panel" } = {}) {
  // panel-варіант (guided-флоу): світла інлайн-пігулка в потоці панелі —
  // без absolute і без темного оверлей-стилю карти.
  const isPanel = variant === "panel";
  const t = useTranslations("search");
  const [q, setQ] = useState("");
  const [results, setResults] = useState<GeoResult[]>([]);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [geoBusy, setGeoBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const debounceRef = useRef<number | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const onDoc = (e: MouseEvent | TouchEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("touchstart", onDoc);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("touchstart", onDoc);
    };
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
        goto(latitude, longitude, name || t("myPlace"));
      },
      () => setGeoBusy(false),
      { enableHighAccuracy: true, timeout: 8000 },
    );
  };

  return (
    <div
      ref={boxRef}
      // Мобільний: власний повноширинний рядок зверху (тоглі/поворот опускаються
      // на ряд нижче — не налазять). Десктоп (sm+): по центру з достатнім
      // резервом обабіч (тоглі ~120px зліва + поворот ~110px справа).
      className={isPanel
        ? "relative w-full"
        : "pointer-events-auto absolute left-2 right-2 top-2 sm:left-1/2 sm:right-auto sm:w-[min(360px,calc(100%-260px))] sm:-translate-x-1/2"}
      style={isPanel ? undefined : { zIndex: 10_000 }}
      data-testid="map-search"
    >
      <div className={isPanel
        ? "flex items-center gap-1 rounded-full px-1 py-0.5"
        : "flex items-center gap-1 rounded-full border border-white/50 bg-[#050a18]/85 px-2 py-1 shadow-[0_8px_20px_rgba(15,23,42,0.28)] backdrop-blur"}>
        <Search size={15} className={isPanel ? "ml-1 shrink-0 text-[var(--accent-strong)]" : "ml-1 shrink-0 text-white/70"} />
        <input
          value={q}
          onChange={(e) => runSearch(e.target.value)}
          onFocus={() => { if (results.length) setOpen(true); }}
          placeholder={t("placeholder")}
          role="combobox"
          aria-expanded={open && results.length > 0}
          aria-controls="map-search-results"
          aria-autocomplete="list"
          aria-label={t("placeholder")}
          className={isPanel
            ? "min-w-0 flex-1 bg-transparent px-1 py-1.5 text-[13px] font-medium text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] outline-none"
            : "min-w-0 flex-1 bg-transparent px-1 py-1 text-[13px] font-medium text-white placeholder:text-white/55 outline-none"}
        />
        {busy && <Loader2 size={15} className={isPanel ? "shrink-0 animate-spin text-[var(--text-secondary)]" : "shrink-0 animate-spin text-white/70"} />}
        {q && !busy && (
          <button type="button" onClick={() => { setQ(""); setResults([]); setOpen(false); }} className={isPanel ? "grid h-9 w-9 shrink-0 place-items-center text-[var(--text-secondary)] hover:text-[var(--text-primary)]" : "grid h-9 w-9 shrink-0 place-items-center text-white/60 hover:text-white"} aria-label={t("clear")}>
            <X size={15} />
          </button>
        )}
        <button
          type="button"
          onClick={useMyLocation}
          className={isPanel
            ? "ml-0.5 grid h-9 w-9 shrink-0 place-items-center rounded-full bg-[rgba(15,118,110,0.1)] text-[var(--accent-strong)] transition hover:bg-[rgba(15,118,110,0.2)]"
            : "ml-0.5 grid h-9 w-9 shrink-0 place-items-center rounded-full bg-white/15 text-white transition hover:bg-white/25"}
          title={t("myLocation")}
          aria-label={t("myLocation")}
        >
          {geoBusy ? <Loader2 size={14} className="animate-spin" /> : <LocateFixed size={14} />}
        </button>
      </div>
      {open && results.length > 0 && (
        <ul id="map-search-results" role="listbox" className={isPanel
          ? "absolute left-0 right-0 z-30 mt-1 overflow-hidden rounded-2xl border border-[var(--surface-border)] bg-white shadow-[0_18px_40px_rgba(15,23,42,0.18)]"
          : "mt-1 overflow-hidden rounded-2xl border border-white/15 bg-[#0a1020]/95 shadow-[0_18px_40px_rgba(15,23,42,0.4)] backdrop-blur"}>
          {results.map((r) => (
            <li key={`${r.lat},${r.lon},${r.label}`} role="option" aria-selected={false}>
              <button
                type="button"
                onClick={() => goto(r.lat, r.lon, r.label)}
                className={isPanel
                  ? "block w-full px-3 py-2 text-left text-[12px] leading-tight text-[var(--text-primary)] transition hover:bg-[rgba(15,118,110,0.08)]"
                  : "block w-full px-3 py-2 text-left text-[12px] leading-tight text-white/90 transition hover:bg-white/10"}
              >
                <span className="font-semibold">{r.label}</span>
                <span className={isPanel ? "block truncate text-[11px] text-[var(--text-secondary)]" : "block truncate text-[11px] text-white/55"}>{r.full}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
