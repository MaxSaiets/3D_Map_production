"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stage, useGLTF } from "@react-three/drei";
import { Suspense, useMemo } from "react";
import * as THREE from "three";

function Model({ url }: { url: string }) {
  // Draco-enabled loader (maps are Draco-compressed); decoder from gstatic CDN.
  // GLBs are un-mirrored at the geometry level in the bake, so no viewer mirror.
  const { scene } = useGLTF(url, true);
  const fixed = useMemo(() => {
    const s = scene.clone(true);
    s.traverse((o: any) => {
      if (o.isMesh && o.material) {
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach((m: any) => { m.side = THREE.DoubleSide; m.needsUpdate = true; });
      }
    });
    return s;
  }, [scene]);
  return <primitive object={fixed} />;
}

/** Auto-rotating 3D viewer for a baked GLB (oriented, coloured). */
export default function Model3DViewer({
  url, height = 360, allowZoom = false, autoRotate = true,
}: { url: string; height?: number; allowZoom?: boolean; autoRotate?: boolean }) {
  return (
    <div style={{ height, width: "100%" }} className="touch-none">
      <Canvas
        dpr={[1, 2]}
        shadows
        camera={{ fov: 40, position: [0, 0.6, 2.4] }}
        gl={{ antialias: true, alpha: true }}
      >
        <Suspense fallback={null}>
          <Stage
            intensity={0.5}
            environment="city"
            adjustCamera={1.1}
            shadows={{ type: "contact", opacity: 0.35, blur: 2.4 }}
          >
            <Model url={url} />
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
    </div>
  );
}

// Preload the hero model for snappy first paint.
useGLTF.preload("/models/keychain-home.glb");
