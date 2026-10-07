"""Компоненти частини Base у 3MF: де дно, чи є «листи» нульової товщини, де стоять будинки.
  venv/Scripts/python.exe ../tools/base_components.py model.3mf"""
import sys
import numpy as np
import trimesh
s = trimesh.load(sys.argv[1])
for node in s.graph.nodes_geometry:
    tf, g = s.graph[node]
    m = s.geometry[g].copy().apply_transform(tf)
    cc = m.split(only_watertight=False)
    flat = [k for k in cc if (k.bounds[1][2] - k.bounds[0][2]) < 1e-3]
    big = sorted(cc, key=lambda x: -len(x.faces))[:3]
    print(f"{node}: faces={len(m.faces)} comps={len(cc)} zero-thickness sheets={len(flat)} "
          f"sheet_faces={sum(len(k.faces) for k in flat)} z={np.round(m.bounds[:,2],3).tolist()}")
    for k in big:
        print("   big comp", len(k.faces), "z", np.round(k.bounds[:, 2], 3).tolist(), "watertight", k.is_watertight)
