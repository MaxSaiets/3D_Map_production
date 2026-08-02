"""Dev-утиліта: рендер мешу в PNG без GPU/вікна (matplotlib Poly3DCollection).

Потрібна саме для перевірки очима — статистики на кшталт is_watertight не
показують, чи справді прорізались вікна й чи не з'їхала основа.
Не імпортується рантаймом; тільки `python -m ml.floorplan.render_debug`.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


def render_mesh(mesh: trimesh.Trimesh, out_path: str, elev: float = 32.0,
                azim: float = -55.0, title: str = "", size: int = 900) -> str:
    verts = mesh.vertices
    faces = mesh.faces
    tris = verts[faces]
    normals = mesh.face_normals

    # СОРТУВАННЯ ЗА ГЛИБИНОЮ. Poly3DCollection не має z-буфера: грані малюються
    # в тому порядку, в якому їх дали, і стіни з дальнього боку перекривають
    # ближні — модель виглядає порваною, хоча геометрія бездоганна. Рахуємо
    # напрямок камери з elev/azim і малюємо від дальніх до ближніх.
    theta, phi = math.radians(elev), math.radians(azim)
    view = np.array([math.cos(theta) * math.cos(phi),
                     math.cos(theta) * math.sin(phi),
                     math.sin(theta)])
    depth = tris.mean(axis=1) @ view
    order = np.argsort(depth)
    tris, normals = tris[order], normals[order]

    # Відкидаємо грані, відвернуті від камери — менше сміття і швидший рендер.
    facing = normals @ view
    keep = facing > -0.02
    tris, normals, facing = tris[keep], normals[keep], facing[keep]

    # Ламбертівський шейдинг — без нього все зливається в силует.
    light = np.array([0.35, -0.55, 0.76])
    light = light / np.linalg.norm(light)
    shade = np.clip(np.abs(normals @ light), 0.0, 1.0) * 0.6 + 0.4
    colors = np.column_stack([shade * 0.93, shade * 0.91, shade * 0.87,
                              np.ones(len(shade))])

    fig = plt.figure(figsize=(size / 100.0, size / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(tris, facecolors=colors, edgecolors=(0.30, 0.30, 0.30, 0.5),
                            linewidths=0.35, zsort="min")
    ax.add_collection3d(coll)

    lo, hi = verts.min(axis=0), verts.max(axis=0)
    center = (lo + hi) / 2.0
    radius = float(np.max(hi - lo)) / 2.0 * 1.05
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, transparent=False, facecolor="white")
    plt.close(fig)
    return out_path


def render_views(mesh: trimesh.Trimesh, out_dir: str, stem: str,
                 views: Optional[Sequence[tuple]] = None, title: str = "") -> list:
    os.makedirs(out_dir, exist_ok=True)
    views = views or [(34.0, -58.0), (90.0, -90.0)]
    out = []
    for i, (elev, azim) in enumerate(views):
        path = os.path.join(out_dir, f"{stem}_v{i}.png")
        render_mesh(mesh, path, elev=elev, azim=azim, title=title)
        out.append(path)
    return out


if __name__ == "__main__":
    import sys

    import numpy as _np

    from ml.floorplan import synth
    from services.floorplan.builder import BuildOptions, build_plan_mesh

    seeds = [int(s) for s in sys.argv[1:]] or [1002, 1005]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_preview3d")
    for seed in seeds:
        spec = synth.generate_layout(_np.random.default_rng(seed))
        plan = synth.layout_to_plan(spec)
        res = build_plan_mesh(plan, BuildOptions(model_size_mm=180.0))
        s = res.stats
        paths = render_views(
            res.mesh, out_dir, f"plan{seed}",
            title=f"seed {seed} | 1:{s['scale_denominator']:.0f} | "
                  f"{s['model_size_mm'][0]:.0f}x{s['model_size_mm'][1]:.0f}x{s['model_size_mm'][2]:.0f} mm",
        )
        print(seed, "->", paths)
