"use client";
export const dynamic = "force-dynamic";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import nextDynamic from "next/dynamic";
import { useGenerationStore } from "@/store/generation-store";
import { MAP_TEMPLATES } from "@/lib/templates";

// three.js (Preview3D) — динамічний імпорт, без SSR: важкий client-only рендерер,
// не має сенсу в серверному бандлі цього internal-роуту (той самий патерн, що й
// на / та /worlds). Іменований експорт → `dynamic` тут зайнятий Next.js
// route-config (export const dynamic вище), тому імпорт з аліасом nextDynamic.
const Preview3D = nextDynamic(() => import("@/components/Preview3D").then((m) => m.Preview3D), { ssr: false });

// T-6.6 (security): same shared-secret gate as /create?capture= — this route auto-runs
// a real generation with zero UI, so it must not be reachable by a stray link/crawler.
// If NEXT_PUBLIC_CAPTURE_TOKEN is unset (local dev / tooling not configured), fall back
// to today's ungated behaviour and warn once.
let warnedCaptureTokenMissing = false;
function isCaptureAuthorized(t: string | null): boolean {
  const expected = process.env.NEXT_PUBLIC_CAPTURE_TOKEN;
  if (!expected) {
    if (!warnedCaptureTokenMissing) {
      warnedCaptureTokenMissing = true;
      // eslint-disable-next-line no-console
      console.warn("[capture] NEXT_PUBLIC_CAPTURE_TOKEN не задано — /capture/[id] працює без гейту (dev fallback).");
    }
    return true;
  }
  return t === expected;
}

/**
 * Internal capture route for generating gallery thumbnails through the real
 * site pipeline. /capture/<templateId> auto-runs a preview generation for that
 * district and renders ONLY the clean 3D model (no UI chrome), then sets
 * window.__captureReady = true so an automated screenshot/toDataURL can grab it.
 * Not linked anywhere in the UI. Requires ?t=<NEXT_PUBLIC_CAPTURE_TOKEN>.
 */
export default function CapturePage() {
  const params = useParams();
  const id = String(params?.id || "");
  const { downloadUrl, taskGroupId, setTaskGroup, setActiveTaskId, setGenerating, setSelectedArea, setDownloadUrl, setTaskStatuses, updateProgress } = useGenerationStore();
  // Checked client-side after mount (no window during SSR) — null while pending.
  const [authorized, setAuthorized] = useState<boolean | null>(null);
  useEffect(() => {
    try {
      const t = new URLSearchParams(window.location.search).get("t");
      setAuthorized(isCaptureAuthorized(t));
    } catch {
      setAuthorized(false);
    }
  }, []);

  useEffect(() => {
    if (!authorized) return;
    const tpl = MAP_TEMPLATES.find((t) => t.id === id);
    if (!tpl) return;
    const [lat, lon] = tpl.center;
    const s = tpl.span;
    const lonPad = s / Math.max(Math.cos((lat * Math.PI) / 180), 0.2);
    const north = lat + s, south = lat - s, east = lon + lonPad, west = lon - lonPad;
    (async () => {
      const L = await import("leaflet");
      setSelectedArea(new L.LatLngBounds([south, west], [north, east]) as any);
      const { api } = await import("@/lib/api");
      const req: any = {
        north, south, east, west,
        road_width_multiplier: 0.8, road_height_mm: 0.5, road_embed_mm: 0.3,
        building_min_height: 5.0, building_height_multiplier: 1.8,
        building_foundation_mm: 0.6, building_embed_mm: 0.2,
        water_depth: 2.0, terrain_enabled: false, terrain_z_scale: 1.0,
        terrain_base_thickness_mm: 1.3, terrain_resolution: 180, terrarium_zoom: 15,
        flatten_buildings_on_terrain: false, flatten_roads_on_terrain: false,
        export_format: "3mf", model_size_mm: 80, context_padding_m: 400.0,
        is_ams_mode: false, flat_plate_mode: false, preview_mode: true,
        preview_include_base: true, preview_include_roads: true,
        preview_include_buildings: true, preview_include_water: true, preview_include_parks: true,
      };
      try {
        setGenerating(true);
        const r = await api.generateModel(req);
        setTaskGroup(r.task_id, [r.task_id]);
        setActiveTaskId(r.task_id);
      } catch {/* ignore */}
    })();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, authorized]);

  // This route has no ControlPanel, so it must poll task status itself and feed
  // the store (downloadUrl) that Preview3D reads to load the generated model.
  useEffect(() => {
    if (!authorized || !taskGroupId) return;
    let stop = false;
    const iv = setInterval(async () => {
      try {
        const { api } = await import("@/lib/api");
        const s: any = await api.getStatus(taskGroupId);
        if (stop) return;
        setTaskStatuses({ [s.task_id]: s });
        updateProgress(s.progress, s.message);
        if (s.status === "completed") {
          setDownloadUrl(s.download_url);
          setGenerating(false);
          clearInterval(iv);
        } else if (s.status === "failed" || s.status === "cancelled") {
          setGenerating(false);
          clearInterval(iv);
        }
      } catch {/* ignore */}
    }, 2500);
    return () => { stop = true; clearInterval(iv); };
  }, [authorized, taskGroupId, setDownloadUrl, setGenerating, setTaskStatuses, updateProgress]);

  useEffect(() => {
    if (typeof window === "undefined" || !authorized) return;
    (window as any).__captureReady = Boolean(downloadUrl);
    if (downloadUrl) {
      // give three.js a moment to render the loaded model before flagging ready
      setTimeout(() => document.body.setAttribute("data-capture-ready", "1"), 2500);
    }
  }, [authorized, downloadUrl]);

  if (authorized === false) {
    return (
      <div className="h-[100dvh] w-[100vw] bg-slate-950 flex items-center justify-center text-slate-500 text-sm">
        Not available
      </div>
    );
  }

  return (
    <div className="h-[100dvh] w-[100vw] bg-slate-950">
      <Preview3D capture />
    </div>
  );
}
