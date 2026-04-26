from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from services.geometry_diagnostics import ensure_valid_geometry, iter_polygons


def _iter_lines(geometry: Any) -> list[Any]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return []
    geom_type = str(getattr(geometry, "geom_type", "") or "")
    if geom_type == "LineString":
        return [geometry]
    if geom_type == "MultiLineString":
        return [geom for geom in getattr(geometry, "geoms", []) if getattr(geom, "geom_type", "") == "LineString"]
    if geom_type == "GeometryCollection":
        lines = []
        for geom in getattr(geometry, "geoms", []):
            lines.extend(_iter_lines(geom))
        return lines
    return []


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    return plt, MplPolygon


def _save_figure(fig: Any, out_png: Path) -> Path:
    plt, _ = _load_matplotlib()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_png


def render_geometry_png(
    geometry: Any,
    out_png: Path,
    *,
    facecolor: str = "black",
    edgecolor: str = "black",
    background: str = "white",
    linewidth: float = 0.2,
) -> Path | None:
    geometry = ensure_valid_geometry(geometry)
    if geometry is None:
        return None

    plt, MplPolygon = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.set_facecolor(background)
    for polygon in iter_polygons(geometry):
        ax.add_patch(
            MplPolygon(
                list(polygon.exterior.coords),
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
            )
        )
        for ring in polygon.interiors:
            ax.add_patch(
                MplPolygon(
                    list(ring.coords),
                    closed=True,
                    facecolor=background,
                    edgecolor=background,
                    linewidth=0.0,
                )
            )
    for line in _iter_lines(geometry):
        xs, ys = line.xy
        ax.plot(xs, ys, color=edgecolor, linewidth=max(linewidth, 0.4))
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    return _save_figure(fig, out_png)


def render_overlay_png(
    layers: Iterable[dict[str, Any]],
    out_png: Path,
    *,
    background: str = "white",
) -> Path | None:
    prepared = []
    for layer in layers:
        geometry = ensure_valid_geometry(layer.get("geometry"))
        if geometry is None:
            continue
        prepared.append(
            {
                "geometry": geometry,
                "facecolor": layer.get("facecolor", "black"),
                "edgecolor": layer.get("edgecolor", layer.get("facecolor", "black")),
                "alpha": float(layer.get("alpha", 0.7)),
                "linewidth": float(layer.get("linewidth", 0.2)),
            }
        )
    if not prepared:
        return None

    plt, MplPolygon = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.set_facecolor(background)
    for layer in prepared:
        for polygon in iter_polygons(layer["geometry"]):
            ax.add_patch(
                MplPolygon(
                    list(polygon.exterior.coords),
                    closed=True,
                    facecolor=layer["facecolor"],
                    edgecolor=layer["edgecolor"],
                    linewidth=layer["linewidth"],
                    alpha=layer["alpha"],
                )
            )
            for ring in polygon.interiors:
                ax.add_patch(
                    MplPolygon(
                        list(ring.coords),
                        closed=True,
                        facecolor=background,
                        edgecolor=background,
                        linewidth=0.0,
                    )
                )
        for line in _iter_lines(layer["geometry"]):
            xs, ys = line.xy
            ax.plot(
                xs,
                ys,
                color=layer["edgecolor"],
                linewidth=max(layer["linewidth"], 0.4),
                alpha=layer["alpha"],
            )
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    return _save_figure(fig, out_png)


def render_mesh_top_png(
    mesh: Any,
    out_png: Path,
    *,
    facecolor: str = "black",
    edgecolor: str = "black",
    face_normal_threshold: float = 0.5,
) -> Path | None:
    import trimesh
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    if mesh is None:
        return None
    if not isinstance(mesh, trimesh.Trimesh):
        try:
            meshes = [item for item in mesh if item is not None and len(item.vertices) > 0]
            mesh = trimesh.util.concatenate(meshes) if meshes else None
        except Exception:
            mesh = None
    if mesh is None or len(mesh.vertices) == 0:
        return None

    face_polygons = []
    for tri in mesh.faces:
        pts = mesh.vertices[tri][:, :2]
        poly = Polygon(pts)
        if poly.is_empty or poly.area <= 1e-9:
            continue
        if not poly.is_valid:
            try:
                poly = poly.buffer(0)
            except Exception:
                continue
        if poly.is_empty or poly.area <= 1e-9:
            continue
        face_polygons.append(poly)
    if not face_polygons:
        return None

    try:
        geometry = unary_union(face_polygons).buffer(0)
    except Exception:
        geometry = unary_union(face_polygons)
    geometry = ensure_valid_geometry(geometry)
    if geometry is None:
        return None

    plt, MplPolygon = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.set_facecolor("white")
    for polygon in iter_polygons(geometry):
        ax.add_patch(
            MplPolygon(
                list(polygon.exterior.coords),
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=0.0,
            )
        )
        for ring in polygon.interiors:
            ax.add_patch(
                MplPolygon(
                    list(ring.coords),
                    closed=True,
                    facecolor="white",
                    edgecolor="white",
                    linewidth=0.0,
                )
            )
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    return _save_figure(fig, out_png)
