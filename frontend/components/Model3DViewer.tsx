"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stage, useGLTF } from "@react-three/drei";
import { Suspense } from "react";

function Model({ url }: { url: string }) {
  // Draco-enabled loader (maps are Draco-compressed); decoder from gstatic CDN.
  const { scene } = useGLTF(url, true);
  return <primitive object={scene} />;
}

/** Auto-rotating 3D viewer for a baked GLB (oriented, coloured). */
export default function Model3DViewer({ url, height = 360 }: { url: string; height?: number }) {
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
            autoRotate
            autoRotateSpeed={2.0}
            enablePan={false}
            enableZoom={false}
            minPolarAngle={Math.PI / 6}
            maxPolarAngle={Math.PI / 2.1}
          />
        </Suspense>
      </Canvas>
    </div>
  );
}

// Preload the hero model for snappy first paint.
useGLTF.preload("/models/keychain-home.glb");
