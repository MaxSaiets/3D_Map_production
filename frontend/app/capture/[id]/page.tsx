"use client";
export const dynamic = "force-dynamic";

import { useEffect } from "react";
import { useParams } from "next/navigation";
import { Preview3D } from "@/components/Preview3D";
import { useGenerationStore } from "@/store/generation-store";
import { MAP_TEMPLATES } from "@/lib/templates";

/**
 * Internal capture route for generating gallery thumbnails through the real
 * site pipeline. /capture/<templateId> auto-runs a preview generation for that
 * district and renders ONLY the clean 3D model (no UI chrome), then sets
 * window.__captureReady = true so an automated screenshot/toDataURL can grab it.
 * Not linked anywhere in the UI.
 */
export default function CapturePage() {
  const params = useParams();
  const id = String(params?.id || "");
  const { downloadUrl, setTaskGroup, setActiveTaskId, setGenerating, setSelectedArea } = useGenerationStore();

  useEffect(() => {
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
        terrain_base_thickness_mm: 0.3, terrain_resolution: 180, terrarium_zoom: 15,
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
  }, [id]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    (window as any).__captureReady = Boolean(downloadUrl);
    if (downloadUrl) {
      // give three.js a moment to render the loaded model before flagging ready
      setTimeout(() => document.body.setAttribute("data-capture-ready", "1"), 2500);
    }
  }, [downloadUrl]);

  return (
    <div className="h-[100dvh] w-[100vw] bg-slate-950">
      <Preview3D capture />
    </div>
  );
}
