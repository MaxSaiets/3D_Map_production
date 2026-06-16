"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stage, useGLTF } from "@react-three/drei";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import * as THREE from "three";

/** Heavy per-model prep (geometry clone + mirror + normals) done ONCE per URL and
 *  cached at module level — re-mounts reuse geometries via a cheap hierarchy clone.
 *  Re-running this on every mount blocked the main thread for seconds per viewer. */
const preparedSceneCache = new Map<string, THREE.Object3D>();

function prepareScene(scene: THREE.Object3D, mirror: boolean, lieFlat: boolean) {
  const s = scene.clone(true);
  s.traverse((o: any) => {
    if (o.isMesh && o.geometry) {
      o.geometry = o.geometry.clone();
      // Un-mirror trimesh's handedness flip at the GEOMETRY level (a scale on
      // the <primitive>/Stage wrapper gets swallowed by <Stage>).
      if (mirror) o.geometry.scale(-1, 1, 1);
      // City maps come in standing vertical (their long axis is up). Tip them
      // back so the map lies flat with buildings pointing up.
      if (lieFlat) o.geometry.rotateX(-Math.PI / 2);
      // scale()/rotateX() already transform existing normals via the normal
      // matrix; only compute when the GLB ships without them.
      if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
    }
    if (o.isMesh && o.material) {
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach((m: any) => { m.side = THREE.DoubleSide; m.needsUpdate = true; });
    }
  });
  return s;
}

function Model({ url, mirror = true, lieFlat = false, onReady }: { url: string; mirror?: boolean; lieFlat?: boolean; onReady?: () => void }) {
  // Draco-enabled loader (maps are Draco-compressed); decoder from gstatic CDN.
  const { scene } = useGLTF(url, true);
  const fixed = useMemo(() => {
    const key = `${url}|${mirror}|${lieFlat}`;
    let base = preparedSceneCache.get(key);
    if (!base) {
      base = prepareScene(scene, mirror, lieFlat);
      preparedSceneCache.set(key, base);
    }
    // Object3D.clone shares geometries/materials — each mounted viewer gets its
    // own hierarchy (an object can live in one scene at a time) at near-zero cost.
    return base.clone(true);
  }, [scene, mirror, lieFlat, url]);
  useEffect(() => { onReady?.(); }, [onReady]);
  return <primitive object={fixed} />;
}

/** Auto-rotating 3D viewer for a baked GLB. Mounts the WebGL canvas only when
 *  scrolled near the viewport (saves battery/CPU on mobile & speeds first paint). */
export default function Model3DViewer({
  url, height = 360, allowZoom = false, autoRotate = true, label, onActivate, flat,
}: { url: string; height?: number; allowZoom?: boolean; autoRotate?: boolean; label?: string; onActivate?: () => void; flat?: boolean }) {
  const t = useTranslations("viewer3d");
  const ref = useRef<HTMLDivElement | null>(null);
  const down = useRef<{ x: number; y: number } | null>(null);
  const [mounted, setMounted] = useState(false);
  // True once the GLB is decoded and the model is in the scene — until then a
  // skeleton overlay covers the (transparent) canvas so users never see a blank box.
  const [ready, setReady] = useState(false);
  // City maps lie flat (thin axis up) — a low camera shows them edge-on like a
  // vertical slab. Look down at a 3/4 angle instead. Keychains stay front-on.
  // `flat` пропом можна примусово (коли URL не містить /map- — напр. /api/download).
  const isMap = flat ?? /\/map-/.test(url);
  const camPos: [number, number, number] = isMap ? [0, 2.5, 2.1] : [0, 0.6, 2.4];

  useEffect(() => {
    if (mounted || typeof IntersectionObserver === "undefined") { setMounted(true); return; }
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => { if (entries.some((e) => e.isIntersecting)) { setMounted(true); io.disconnect(); } },
      { rootMargin: "120px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [mounted]);

  return (
    <div
      ref={ref}
      style={{ height, width: "100%", cursor: onActivate ? "pointer" : undefined }}
      className="relative touch-none"
      role={onActivate ? "button" : "img"}
      tabIndex={onActivate ? 0 : undefined}
      aria-label={label ? t("modelLabeled", { label }) : t("model")}
      onPointerDown={onActivate ? (e) => { down.current = { x: e.clientX, y: e.clientY }; } : undefined}
      onPointerUp={onActivate ? (e) => {
        const d = down.current; down.current = null;
        if (d && Math.hypot(e.clientX - d.x, e.clientY - d.y) < 8) onActivate();
      } : undefined}
      onKeyDown={onActivate ? (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onActivate(); }
      } : undefined}
    >
      {label && <span className="sr-only">{label}</span>}
      {mounted ? (
        <Canvas
          dpr={[1, 1.5]}
          shadows
          camera={{ fov: 40, position: camPos }}
          gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
        >
          {/* Explicit lights — no remote HDRI (drei's Environment CDN often fails,
              leaving Suspense unresolved → blank canvas). environment={null}. */}
          <ambientLight intensity={0.85} />
          <hemisphereLight args={[0xffffff, 0x8d7a5a, 0.7]} />
          <directionalLight position={[4, 6, 5]} intensity={1.7} castShadow />
          <directionalLight position={[-5, 3, -4]} intensity={0.55} />
          <Suspense fallback={null}>
            <Stage
              intensity={0.4}
              environment={null}
              preset="rembrandt"
              adjustCamera={1.1}
              shadows={{ type: "contact", opacity: 0.3, blur: 2.4, frames: 1 }}
            >
              <Model url={url} lieFlat={isMap} onReady={() => setReady(true)} />
            </Stage>
            <OrbitControls
              autoRotate={autoRotate}
              autoRotateSpeed={1.6}
              enablePan={false}
              enableZoom={allowZoom}
              minPolarAngle={0}
              maxPolarAngle={Math.PI}
            />
          </Suspense>
        </Canvas>
      ) : null}
      {!ready && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div className="absolute inset-0 animate-pulse bg-gradient-to-br from-black/[0.05] via-transparent to-black/[0.07]" />
          <span className="relative inline-flex items-center gap-2 rounded-full bg-white/80 px-3 py-1.5 text-[12px] font-semibold text-ink-3 shadow-sm">
            <svg className="h-3.5 w-3.5 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" opacity=".25" />
              <path d="M22 12a10 10 0 0 0-10-10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
            </svg>
            3D
          </span>
        </div>
      )}
    </div>
  );
}

// Preload the hero models for snappy first paint.
useGLTF.preload("/models/keychain-fea.glb");
