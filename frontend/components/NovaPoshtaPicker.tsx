"use client";

import { useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import { Loader2, MapPin, Search } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

type City = { ref: string; name: string; area: string; region: string };
type Wh = { ref: string; number: string; name: string; short: string };

/**
 * Пошук міста + відділення Нової Пошти у формі замовлення.
 * Бекенд-проксі ховає API-ключ. Якщо ключ не налаштовано (configured:false) —
 * рендеримо звичайні текстові поля (як було), без жодної регресії.
 */
export function NovaPoshtaPicker({
  city,
  branch,
  setCity,
  setBranch,
  inputCls,
}: {
  city: string;
  branch: string;
  setCity: (s: string) => void;
  setBranch: (s: string) => void;
  inputCls: string;
}) {
  const t = useTranslations("order");
  const [configured, setConfigured] = useState<boolean | null>(null);
  const [cityQuery, setCityQuery] = useState(city);
  const [cityResults, setCityResults] = useState<City[]>([]);
  const [cityRef, setCityRef] = useState("");
  const [cityOpen, setCityOpen] = useState(false);
  const [whResults, setWhResults] = useState<Wh[]>([]);
  const [whQuery, setWhQuery] = useState(branch);
  const [whOpen, setWhOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const debCity = useRef<ReturnType<typeof setTimeout> | null>(null);
  const debWh = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let alive = true;
    fetch(`${API_BASE}/api/delivery/np/status`)
      .then((r) => r.json())
      .then((d) => { if (alive) setConfigured(!!d.configured); })
      .catch(() => { if (alive) setConfigured(false); });
    return () => { alive = false; };
  }, []);

  // Пошук міста (debounce 280мс)
  useEffect(() => {
    if (!configured) return;
    if (debCity.current) clearTimeout(debCity.current);
    const q = cityQuery.trim();
    if (q.length < 2 || q === city) { setCityResults([]); return; }
    debCity.current = setTimeout(async () => {
      try {
        setLoading(true);
        const r = await fetch(`${API_BASE}/api/delivery/np/cities?q=${encodeURIComponent(q)}`);
        const d = await r.json();
        setCityResults(Array.isArray(d.items) ? d.items : []);
      } catch { setCityResults([]); } finally { setLoading(false); }
    }, 280);
    return () => { if (debCity.current) clearTimeout(debCity.current); };
  }, [cityQuery, configured, city]);

  const loadWarehouses = async (ref: string, q: string) => {
    if (!ref) return;
    try {
      setLoading(true);
      const r = await fetch(`${API_BASE}/api/delivery/np/warehouses?cityRef=${encodeURIComponent(ref)}&q=${encodeURIComponent(q)}`);
      const d = await r.json();
      setWhResults(Array.isArray(d.items) ? d.items : []);
    } catch { setWhResults([]); } finally { setLoading(false); }
  };

  const pickCity = (c: City) => {
    setCity(c.name); setCityQuery(c.name); setCityRef(c.ref);
    setCityResults([]); setCityOpen(false);
    setBranch(""); setWhQuery(""); setWhResults([]);
    loadWarehouses(c.ref, "");
    setWhOpen(true);
  };

  // Пошук відділення в обраному місті (debounce)
  useEffect(() => {
    if (!configured || !cityRef) return;
    if (debWh.current) clearTimeout(debWh.current);
    const q = whQuery.trim();
    if (q === branch) return;
    debWh.current = setTimeout(() => loadWarehouses(cityRef, q), 280);
    return () => { if (debWh.current) clearTimeout(debWh.current); };
  }, [whQuery, cityRef, configured, branch]);

  // Ключ не налаштовано → звичайні поля (поведінка як раніше, без регресій)
  if (configured === false) {
    return (
      <>
        <input className={inputCls} placeholder={t("phCity")} aria-label={t("phCity")} value={city} onChange={(e) => setCity(e.target.value)} />
        <input className={inputCls} placeholder={t("phNova")} aria-label={t("phNova")} value={branch} onChange={(e) => setBranch(e.target.value)} />
      </>
    );
  }
  // Статус ще вантажиться (≈миттєво) → одне поле міста, щоб не блимало двома
  if (configured === null) {
    return <input className={inputCls} placeholder={t("npCityPh")} aria-label={t("npCityPh")} value={city} onChange={(e) => setCity(e.target.value)} />;
  }

  return (
    <div className="space-y-2">
      {/* Місто */}
      <div className="relative">
        <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-secondary)]"><Search size={15} /></span>
        <input
          className={`${inputCls} pl-9`}
          placeholder={t("npCityPh")}
          aria-label={t("npCityPh")}
          autoComplete="off"
          role="combobox"
          aria-autocomplete="list"
          aria-expanded={cityOpen && cityResults.length > 0}
          aria-controls="np-city-listbox"
          value={cityQuery}
          onChange={(e) => { setCityQuery(e.target.value); setCityOpen(true); setCityRef(""); setBranch(""); setWhQuery(""); }}
          onFocus={(e) => { setCityOpen(true); e.target.scrollIntoView({ block: 'center' }); }}
          onBlur={() => setTimeout(() => setCityOpen(false), 160)}
        />
        {cityOpen && cityResults.length > 0 && (
          <ul id="np-city-listbox" role="listbox" className="absolute z-30 mt-1 max-h-44 w-full overflow-auto rounded-2xl border border-[var(--surface-border)] bg-white shadow-[0_18px_40px_rgba(15,23,42,0.16)]">
            {cityResults.map((c) => (
              <li key={c.ref} role="option" aria-selected={false}>
                <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => pickCity(c)}
                  className="flex w-full items-center gap-2 px-4 py-2.5 text-left text-sm hover:bg-[rgba(15,118,110,0.08)]">
                  <MapPin size={14} className="shrink-0 text-[var(--accent-strong)]" />
                  <span className="truncate"><b className="font-semibold">{c.name}</b>{c.area ? <span className="text-[var(--text-secondary)]"> · {c.area} обл.</span> : null}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Відділення (після вибору міста) */}
      {cityRef && (
        <div className="relative">
          <input
            className={inputCls}
            placeholder={t("npWhPh")}
            aria-label={t("npWhPh")}
            autoComplete="off"
            role="combobox"
            aria-autocomplete="list"
            aria-expanded={whOpen && whResults.length > 0}
            aria-controls="np-wh-listbox"
            value={whQuery}
            onChange={(e) => { setWhQuery(e.target.value); setWhOpen(true); }}
            onFocus={(e) => { setWhOpen(true); if (!whResults.length) loadWarehouses(cityRef, ""); e.target.scrollIntoView({ block: 'center' }); }}
            onBlur={() => setTimeout(() => setWhOpen(false), 160)}
          />
          {whOpen && whResults.length > 0 && (
            <ul id="np-wh-listbox" role="listbox" className="absolute z-30 mt-1 max-h-44 w-full overflow-auto rounded-2xl border border-[var(--surface-border)] bg-white shadow-[0_18px_40px_rgba(15,23,42,0.16)]">
              {whResults.map((w) => (
                <li key={w.ref} role="option" aria-selected={false}>
                  <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => { setBranch(w.name); setWhQuery(w.name); setWhOpen(false); }}
                    className="block w-full px-4 py-2.5 text-left text-sm hover:bg-[rgba(15,118,110,0.08)]">
                    <span className="line-clamp-2">{w.name}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {loading && <p className="flex items-center gap-1.5 px-1 text-[11px] text-[var(--text-secondary)]"><Loader2 size={12} className="animate-spin" /> {t("npLoading")}</p>}
    </div>
  );
}
