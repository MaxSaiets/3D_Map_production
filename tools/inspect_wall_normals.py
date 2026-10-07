"""Діагностика «вивернутих граней» готової моделі (3MF/STL/GLB).

Для кожної частини: watertight, відкриті ребра, узгодженість обходу (winding), знак об'єму,
скільки ВЕРТИКАЛЬНИХ граней на периметрі дивляться ВСЕРЕДИНУ (нормаль до центру в XY)
і скільки граней на самому ДНІ дивляться ВГОРУ (мали б униз).

  venv/Scripts/python.exe ../tools/inspect_wall_normals.py path/to/model.3mf [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import trimesh


def analyze(name: str, m: trimesh.Trimesh) -> dict:
    m = m.copy()
    n = m.face_normals
    c = m.triangles_center
    zmin = float(m.bounds[0][2])
    ctr = m.bounds.mean(axis=0)
    # периметр: грані, центр яких у зовнішніх 3 % радіуса bbox у XY
    rxy = np.linalg.norm(c[:, :2] - ctr[:2], axis=1)
    rmax = float(rxy.max()) if len(rxy) else 0.0
    vertical = np.abs(n[:, 2]) < 0.2
    perim = rxy > rmax * 0.97
    radial = (c[:, :2] - ctr[:2])
    radial_n = radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    dot = np.einsum("ij,ij->i", n[:, :2], radial_n)
    walls = vertical & perim
    inward = walls & (dot < -0.3)
    bottom = np.abs(c[:, 2] - zmin) < 1e-3
    bottom_up = bottom & (n[:, 2] > 0.5)
    edges = m.edges_sorted
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "part": name,
        "faces": int(len(m.faces)),
        "watertight": bool(m.is_watertight),
        "winding_consistent": bool(m.is_winding_consistent),
        "open_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "volume": round(float(m.volume), 3) if m.is_watertight else None,
        "perimeter_walls": int(walls.sum()),
        "walls_facing_inward": int(inward.sum()),
        "bottom_faces": int(bottom.sum()),
        "bottom_facing_up": int(bottom_up.sum()),
        "z_range": [round(zmin, 3), round(float(m.bounds[1][2]), 3)],
    }


def load_parts(path: str) -> dict[str, trimesh.Trimesh]:
    obj = trimesh.load(path, force=None)
    if isinstance(obj, trimesh.Scene):
        out = {}
        for node in obj.graph.nodes_geometry:
            tf, gname = obj.graph[node]
            g = obj.geometry[gname]
            if isinstance(g, trimesh.Trimesh):
                out[f"{node}"] = g.copy().apply_transform(tf)
        return out
    return {"mesh": obj}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = [analyze(k, v) for k, v in load_parts(a.path).items()]
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    if a.json:
        json.dump(rows, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
