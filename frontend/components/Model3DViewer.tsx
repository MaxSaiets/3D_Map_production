"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stage, useGLTF } from "@react-three/drei";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

function Model({ url, mirror = true, lieFlat = false }: { url: string; mirror?: boolean; lieFlat?: boolean }) {
  // Draco-enabled loader (maps are Draco-compressed); decoder from gstatic CDN.
  const { scene } = useGLTF(url, true);
  const fixed = useMemo(() => {
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
        o.geometry.computeVertexNormals();
      }
      if (o.isMesh && o.material) {
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach((m: any) => { m.side = THREE.DoubleSide; m.needsUpdate = true; });
      }
    });
    return s;
  }, [scene, mirror, lieFlat]);
  return <primitive object={fixed} />;
}

/** Auto-rotating 3D viewer for a baked GLB. Mounts the WebGL canvas only when
 *  scrolled near the viewport (saves battery/CPU on mobile & speeds first paint). */
export default function Model3DViewer({
  url, height = 360, allowZoom = false, autoRotate = true, label, onActivate,
}: { url: string; height?: number; allowZoom?: boolean; autoRotate?: boolean; label?: string; onActivate?: () => void }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const down = useRef<{ x: number; y: number } | null>(null);
  const [mounted, setMounted] = useState(false);
  // City maps lie flat (thin axis up) — a low camera shows them edge-on like a
  // vertical slab. Look down at a 3/4 angle instead. Keychains stay front-on.
  const isMap = /\/map-/.test(url);
  const camPos: [number, number, number] = isMap ? [0, 2.5, 2.1] : [0, 0.6, 2.4];

  useEffect(() => {
    if (mounted || typeof IntersectionObserver === "undefined") { setMounted(true); return; }
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => { if (entries.some((e) => e.isIntersecting)) { setMounted(true); io.disconnect(); } },
      { rootMargin: "250px" },
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
      aria-label={label ? `3D-модель: ${label}` : "Інтерактивна 3D-модель"}
      onPointerDown={onActivate ? (e) => { down.current = { x: e.clientX, y: e.clientY }; } : undefined}
      onPointerUp={onActivate ? (e) => {
        const d = down.current; down.current = null;
        if (d && Math.hypot(e.clientX - d.x, e.clientY - d.y) < 8) onActivate();
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
              shadows={{ type: "contact", opacity: 0.3, blur: 2.4 }}
            >
              <Model url={url} lieFlat={isMap} />
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
      ) : (
        <div className="flex h-full w-full items-center justify-center">
          <span className="inline-flex items-center gap-2 rounded-full bg-white/70 px-3 py-1.5 text-[12px] font-semibold text-ink-3">
            <span className="h-2 w-2 animate-pulse rounded-full bg-forest" /> 3D
          </span>
        </div>
      )}
    </div>
  );
}

// Preload the hero models for snappy first paint.
useGLTF.preload("/models/keychain-fea.glb");
