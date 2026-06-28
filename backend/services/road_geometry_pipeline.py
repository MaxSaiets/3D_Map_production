from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import osmnx as ox
from geopandas import GeoDataFrame
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry, unary_union

from services.detail_layer_utils import MICRO_REGION_THRESHOLD_MM, model_mm_to_world_m
from services.road_processor import (
    build_road_polygons,
    normalize_drivable_highway_tag,
    normalize_road_mask_for_print,
)
from services.geometry_context import looks_like_projected_meters

# СВІТОВІ КАПИ для доріг. Пороги друкованості (мм моделі) на великих зонах
# (масштаб ≤0.1 мм/м) вибухають у світових метрах: 1.0мм → 10м мін. ширини
# вулиці + заливка проміжків ≤10м → щільний центр зливається у суцільні плями
# (скарга юзера: «обʼєднує дуже багато», чорний Майдан на 1500м-зоні).
# Капи тримають значення в межах реальної міської геометрії.
# 2026-06-14: знижено (скарга «досі зливає, чорний Майдан»). 9м роздувало
# КОЖНУ вулицю до туширини сусідів → unary_union зливав у пляму. 6м×0.1=0.6мм
# ще друкується (>0.4мм сопло), але вулиці з кроком ≥10м лишаються окремими.
# gap-fill 6→3.5 закриває лише міжсмугові щілини, НЕ цілі квартали.
# 2026-06-25: gap-fill 3.5→2.5 (скарга «дороги не так сильно обєднувались») —
# ШИРОКЕ proximity-злиття зменшено; крихкі ТОНКІ рельєфні стінки між дорогами
# тепер заповнює ТОЧКОВИЙ thinness-based fill (_fill_thin_terrain_walls_with_road),
# що ріже дорогою ЛИШЕ непридатно-тонкі місця, а не зближені дороги загалом.
ROAD_MIN_WIDTH_WORLD_CAP_M = 6.0    # макс. роздування вузької вулиці (друковано 0.6мм)
ROAD_GAP_FILL_WORLD_CAP_M = 2.5     # лише вузькі міжсмугові щілини/трикутники перехресть
ROAD_ORPHAN_HOLE_WORLD_CAP_M = 2.0  # дрібні дірки в перехрестях

# Точкова заливка ДОРОГОЮ тонких рельєфних стінок між дорогами: рельєфна стінка
# тонша за це ламається при друці («малі стовби» → пустота). Заповнюємо дорогою
# ЛИШЕ де тонко (morphological opening) → широкі ділянки рельєфу лишаються, дороги
# НЕ зливаються широко. Поріг у мм МОДЕЛІ; світовий кап — щоб на великих зонах не
# роздувало. Юзер може тюнити.
PRINT_SAFE_TERRAIN_WALL_MM = 2.5
ROAD_THIN_WALL_FILL_WORLD_CAP_M = 16.0  # honor 2.5мм-model threshold up to ~1km zones


@dataclass
class RoadGeometryPreparationResult:
    merged_roads_geom: Optional[BaseGeometry]
    merged_roads_geom_local: Optional[BaseGeometry]
    merged_roads_geom_local_raw: Optional[BaseGeometry]
    semantic_centerlines_local: Optional[BaseGeometry] = None


def _build_local_road_edges_subset(
    *,
    G_roads: Any,
    global_center: Any,
    zone_polygon_local: Optional[BaseGeometry],
    zone_prefix: str,
) -> Optional[GeoDataFrame]:
    if G_roads is None or global_center is None or zone_polygon_local is None or getattr(zone_polygon_local, "is_empty", True):
        return None

    try:
        if isinstance(G_roads, GeoDataFrame):
            gdf_edges = G_roads.copy()
        else:
            if not hasattr(G_roads, "edges") or len(G_roads.edges) == 0:
                return None
            gdf_edges = ox.graph_to_gdfs(G_roads, nodes=False)
        if gdf_edges is None or gdf_edges.empty:
            return None

        def to_local(x, y, z=None):
            # Robust CRS normalization:
            # - projected meters (UTM-like): subtract global center directly
            # - lon/lat (WGS84): convert to UTM first, then to local
            if abs(float(x)) <= 180.0 and abs(float(y)) <= 90.0:
                x_utm, y_utm = global_center.to_utm(float(x), float(y))
                x_local, y_local = global_center.to_local(x_utm, y_utm)
            else:
                x_local, y_local = global_center.to_local(float(x), float(y))
            return (x_local, y_local) if z is None else (x_local, y_local, z)

        gdf_local = gdf_edges.copy()
        sample_geom = gdf_local.geometry.iloc[0] if len(gdf_local) else None
        if sample_geom is not None and not getattr(sample_geom, "is_empty", True):
            if looks_like_projected_meters(sample_geom):
                gdf_local["geometry"] = gdf_local.geometry.apply(lambda geom: transform_geometry(to_local, geom))
            else:
                # If data appears unprojected (lon/lat), still normalize to local.
                gdf_local["geometry"] = gdf_local.geometry.apply(lambda geom: transform_geometry(to_local, geom))
        else:
            gdf_local["geometry"] = gdf_local.geometry.apply(lambda geom: transform_geometry(to_local, geom))

        minx, miny, maxx, maxy = zone_polygon_local.bounds
        # Keep a generous apron around the target zone so buffered roads that land
        # on the border still have their correct shape before the final polygon clip.
        clip_window = box(minx - 40.0, miny - 40.0, maxx + 40.0, maxy + 40.0)
        gdf_local = gdf_local[gdf_local.geometry.intersects(clip_window)].copy()
        if gdf_local.empty:
            return None

        print(
            f"[DEBUG] {zone_prefix} prefiltered road edges for local mask: "
            f"{len(gdf_local)}/{len(gdf_edges)} kept"
        )
        return gdf_local
    except Exception as exc:
        print(f"[WARN] {zone_prefix} Failed to prefilter road edges locally: {exc}")
        return None


def _fill_thin_terrain_walls_with_road(
    roads: Optional[BaseGeometry],
    zone_polygon_local: Optional[BaseGeometry],
    scale_factor: Optional[float],
    zone_prefix: str = "",
) -> Optional[BaseGeometry]:
    """Заповнює ДОРОГОЮ тонкі рельєфні стінки між/біля доріг (тонші за друкопридатне).

    Точково через morphological opening — лише там, де рельєф між дорогами тонший за
    PRINT_SAFE_TERRAIN_WALL_MM. Широкі ділянки рельєфу лишаються (дороги НЕ зливаються
    широко). Так зникають крихкі «стовби», що ломаються при друці й лишають пустоту.
    Безпечно: будь-яка помилка → повертаємо вихідні дороги (нічого не ламає).
    """
    try:
        import os
        if os.environ.get("THIN_WALL_FILL", "1") == "0":   # toggle для A/B-перевірки
            return roads
        if (roads is None or getattr(roads, "is_empty", True)
                or zone_polygon_local is None or getattr(zone_polygon_local, "is_empty", True)
                or not scale_factor or float(scale_factor) <= 0):
            return roads
        W = min(
            model_mm_to_world_m(PRINT_SAFE_TERRAIN_WALL_MM, float(scale_factor)),
            ROAD_THIN_WALL_FILL_WORLD_CAP_M,
        )
        if W <= 0.05:
            return roads
        r = W / 2.0
        terrain_gap = zone_polygon_local.difference(roads).buffer(0)
        if getattr(terrain_gap, "is_empty", True):
            return roads
        # opening прибирає смужки тонші за W → різниця = тонкі стінки
        opened = terrain_gap.buffer(-r, join_style=2).buffer(r, join_style=2)
        thin = terrain_gap.difference(opened).buffer(0)
        if getattr(thin, "is_empty", True):
            return roads
        # стінки МІЖ/біля доріг (буфер W*3 щоб впіймати центри стінок і між дальшими
        # дорогами; далекий від доріг тонкий рельєф — парки/край зони — не чіпаємо)
        near = thin.intersection(roads.buffer(W * 3.0, join_style=2)).buffer(0)
        if getattr(near, "is_empty", True):
            return roads
        out = roads.union(near).buffer(0)
        out = out.intersection(zone_polygon_local).buffer(0)
        added = float(getattr(near, "area", 0.0) or 0.0)
        if added > 1e-6:
            try:  # ASCII + guarded: a print/encoding failure must NOT lose the fill
                print(f"[INFO] {zone_prefix} thin-terrain-wall->road fill: +{added:.0f} m2 road "
                      f"(walls <{W:.1f}m world / {PRINT_SAFE_TERRAIN_WALL_MM}mm model)")
            except Exception:
                pass
        return out
    except Exception as exc:
        print(f"[WARN] {zone_prefix} thin-terrain-wall fill failed (non-fatal): {exc}")
        return roads


def _drop_dense_service_clusters(gdf_local, *, cluster_gap_m=9.0, min_cluster_edges=4,
                                 zone_prefix=""):
    """Прибирає service-смуги, що утворюють КЛАСТЕР (паркінг/дворовий лабіринт), лишаючи
    ПООДИНОКІ service-заїзди + УСІ residential та вищі (скелет вулиць). КОРІНЬ
    (верифіковано wf_3b3880fd + OSM-аналіз Вугледара: 153 service vs 19 residential у
    дворах): на масштабі друку (sf<~0.15) service-смуги <~5м одна від одної отримують
    min-width floor ~4.5м → перекриваються → суцільна чорна плита, яку НІЯКИЙ fill/guard
    не розділить (фізика: зазор <0.6мм друку). Прибирання кластерів service ДО
    buffer/union → двір стає чистою відкритою основою з будинками (skeleton residential
    лишається, бо ті рознесені >10м і не зливаються).

    Правило РІВНОМІРНЕ (без density/hull-порогів = БЕЗ «латковості», яку власник
    відкинув): буфер service на cluster_gap_m/2, union; БУДЬ-ЯКИЙ зв'язний blob із
    ≥ min_cluster_edges service-ребер = лабіринт → дропнути ВСІ його service-ребра.
    Поодинокі заїзди (1..min_cluster_edges-1 у blob) ЛИШАЮТЬСЯ. residential+ не чіпає
    взагалі (фільтр лише по _normalized_highway=='service'). ENV: DENSE_SERVICE_PRUNE=0
    вимикає; DENSE_SERVICE_MIN_EDGES перекриває поріг кластера.
    (Прод DuckDB road-таблиця: лише id/highway/bridge/wkt — 'service'-підтегу немає →
    детект геометричний, не за підтегом.)
    """
    try:
        if os.environ.get("DENSE_SERVICE_PRUNE", "1") == "0":
            return gdf_local
        if gdf_local is None or getattr(gdf_local, "empty", True):
            return gdf_local
        if "_normalized_highway" not in getattr(gdf_local, "columns", []):
            return gdf_local
        try:
            min_cluster_edges = int(os.environ.get("DENSE_SERVICE_MIN_EDGES", str(min_cluster_edges)))
        except Exception:
            pass
        svc = gdf_local[gdf_local["_normalized_highway"] == "service"]
        if len(svc) < min_cluster_edges:
            return gdf_local
        items = [(idx, g) for idx, g in zip(svc.index, svc.geometry)
                 if g is not None and not getattr(g, "is_empty", True)]
        if len(items) < min_cluster_edges:
            return gdf_local
        # зв'язні кластери: буфер кожної смуги на пів-зазору, union
        merged = unary_union([g.buffer(cluster_gap_m / 2.0) for _, g in items]).buffer(0)
        blobs = list(getattr(merged, "geoms", [merged]))
        drop_idx = []
        for blob in blobs:
            members = [idx for (idx, g) in items if g.intersects(blob)]
            if len(members) >= min_cluster_edges:
                drop_idx.extend(members)
        if not drop_idx:
            return gdf_local
        keep_mask = ~gdf_local.index.isin(drop_idx)
        print(f"[INFO] {zone_prefix} dropped {len(drop_idx)} clustered service lanes "
              f"({len(svc)} service total, {len(blobs)} cluster blobs) -> open courtyards")
        return gdf_local[keep_mask].copy()
    except Exception as exc:
        print(f"[WARN] {zone_prefix} dense-service prune failed (non-fatal): {exc}")
        return gdf_local


def _fill_small_road_holes(geometry: Optional[BaseGeometry], *, max_hole_area_m2: float) -> Optional[BaseGeometry]:
    """Fill small fully-ENCLOSED interior holes (terrain islands surrounded by road —
    e.g. a pocket between two close roads) below max_hole_area_m2. On THIS raw road
    polygon the islands are topological holes (the full-width road polygons close
    around them), unlike the narrower canonical insert mask where the same pocket
    reads as an open gap and the existing fill misses it. A hole is enclosed by
    definition, so this NEVER welds parallel roads (open gaps are not holes) — only
    road-surrounded pockets are filled. This geom feeds BOTH the inlay mesh and the
    canonical mask, so the fix propagates to the printed model."""
    from shapely.geometry import Polygon as _Poly
    if geometry is None or getattr(geometry, "is_empty", True) or max_hole_area_m2 <= 0:
        return geometry
    polys = [geometry] if getattr(geometry, "geom_type", "") == "Polygon" else list(getattr(geometry, "geoms", []))
    rebuilt = []
    changed = False
    for poly in polys:
        if getattr(poly, "geom_type", "") != "Polygon" or poly.is_empty:
            if not getattr(poly, "is_empty", True):
                rebuilt.append(poly)
            continue
        kept = []
        for ring in poly.interiors:
            try:
                if float(_Poly(ring.coords).area) > float(max_hole_area_m2):
                    kept.append(ring)
                else:
                    changed = True
            except Exception:
                kept.append(ring)
        if len(kept) != len(list(poly.interiors)):
            try:
                rebuilt.append(_Poly(poly.exterior.coords, holes=kept).buffer(0))
            except Exception:
                rebuilt.append(poly)
        else:
            rebuilt.append(poly)
    if not changed or not rebuilt:
        return geometry
    try:
        return unary_union(rebuilt).buffer(0)
    except Exception:
        return rebuilt[0] if len(rebuilt) == 1 else geometry


def prepare_road_geometry(
    *,
    G_roads: Any,
    scale_factor: Optional[float],
    road_width_multiplier_effective: float,
    min_printable_gap_mm: float,
    tiny_feature_threshold_mm: float = MICRO_REGION_THRESHOLD_MM,
    road_gap_fill_threshold_mm: float = 0.0,
    enforce_printable_min_width: bool = False,
    min_gap_fill_floor_mm: float = 0.0,
    global_center: Any,
    zone_polygon_local: Optional[BaseGeometry],
    zone_prefix: str = "",
) -> RoadGeometryPreparationResult:
    merged_roads_geom = None
    merged_roads_geom_local = None
    merged_roads_geom_local_raw = None
    semantic_centerlines_local = None
    min_road_width_for_build = None
    effective_min_width_mm = max(
        float(min_printable_gap_mm or 0.0),
        float(min_gap_fill_floor_mm or 0.0),
        0.5,
    )
    effective_gap_fill_mm = max(
        float(road_gap_fill_threshold_mm or 0.0),
        0.5,  # floor: always fill at least 0.5mm-model gaps (nozzle tolerance)
    )
    effective_tiny_feature_mm = max(float(tiny_feature_threshold_mm or 0.0), effective_min_width_mm)
    if scale_factor and float(scale_factor) > 0:
        try:
            # СВІТОВИЙ КАП: пороги друкованості задані в мм МОДЕЛІ, тож на
            # великих зонах (масштаб ≤0.1мм/м) вони вибухають у світових метрах:
            # 1.0мм = 10м мін. ширини КОЖНОЇ вулиці + заливка проміжків ≤10м →
            # щільний центр (Київ, Майдан) зливався у суцільні чорні плями.
            # Реальна вулиця з тротуарами ~9м — ширше робити безглуздо.
            min_road_width_for_build = min(
                model_mm_to_world_m(effective_min_width_mm, float(scale_factor)),
                ROAD_MIN_WIDTH_WORLD_CAP_M,
            )
        except Exception:
            min_road_width_for_build = None
    local_edges_subset = _build_local_road_edges_subset(
        G_roads=G_roads,
        global_center=global_center,
        zone_polygon_local=zone_polygon_local,
        zone_prefix=zone_prefix,
    )

    try:
        if local_edges_subset is not None and not local_edges_subset.empty:
            local_edges_subset = local_edges_subset.copy()
            if "highway" in local_edges_subset.columns:
                local_edges_subset["_normalized_highway"] = local_edges_subset["highway"].apply(
                    normalize_drivable_highway_tag
                )
                local_edges_subset = local_edges_subset[local_edges_subset["_normalized_highway"].notna()].copy()
            try:
                semantic_parts = [
                    geom
                    for geom in local_edges_subset.geometry.values
                    if geom is not None and not getattr(geom, "is_empty", True)
                ]
                if semantic_parts:
                    semantic_centerlines_local = unary_union(semantic_parts)
                    if zone_polygon_local is not None and not getattr(zone_polygon_local, "is_empty", True):
                        semantic_centerlines_local = semantic_centerlines_local.intersection(zone_polygon_local)
            except Exception:
                semantic_centerlines_local = None

            merged_roads_geom_local = build_road_polygons(
                local_edges_subset,
                width_multiplier=float(road_width_multiplier_effective),
                min_width_m=min_road_width_for_build,
                scale_factor=scale_factor,
            )
            if merged_roads_geom_local is not None and not getattr(merged_roads_geom_local, "is_empty", True):
                if zone_polygon_local is not None:
                    merged_roads_geom_local = merged_roads_geom_local.intersection(zone_polygon_local)
                    # ── Pull roads off the boundary cliff ──────────────────
                    # The base has a vertical side-wall at the zone edge. A road
                    # whose polygon touches that edge gets a full-height vertical
                    # side face there -> a thin tall road "pillar" at the rim
                    # (and it chops the rim terrain into thin slivers between
                    # road-ends and the corner). Clip roads to a slightly-inset
                    # zone so they stop just short of the cliff; the rim stays a
                    # single continuous terrain band. Env: ROAD_EDGE_INSET_MM.
                    # ДЕФОЛТ 0 (вимкнено): власник хоче, щоб дороги ДОХОДИЛИ до краю
                    # моделі. Дорога, зрізана врівень з боковою стінкою основи на краю —
                    # це нормально/очікувано, а не «стовп». Лишається env-перемикач
                    # на випадок, якщо колись знадобиться відсунути від обриву.
                    try:
                        _inset_mm = float(os.environ.get("ROAD_EDGE_INSET_MM", "0.0"))
                    except (TypeError, ValueError):
                        _inset_mm = 0.0
                    if _inset_mm > 0.0:
                        try:
                            _band_m = model_mm_to_world_m(_inset_mm, float(scale_factor))
                            _inset_zone = zone_polygon_local.buffer(-_band_m)
                            if (
                                _inset_zone is not None
                                and not getattr(_inset_zone, "is_empty", True)
                                and _inset_zone.area > 0.0
                            ):
                                merged_roads_geom_local = merged_roads_geom_local.intersection(_inset_zone)
                        except Exception:
                            pass
                try:
                    merged_roads_geom_local = merged_roads_geom_local.buffer(0)
                except Exception:
                    pass

            merged_roads_geom_local_raw = merged_roads_geom_local

            # ── Printable gap-fill ─────────────────────────────────────────
            # Close sub-printable gaps between road polygons NOW, on the raw
            # buffered road mask (before layer-precedence clipping, before the
            # canonical bundle). This is the only safe location:
            #   • Later stages (runtime_canonical_masks, detail_layer_pipeline)
            #     operate on already-processed masks where a CLOSE fills entire
            #     city blocks, not just road endpoint gaps.
            #   • merge_close_road_gaps adds only narrow wedges/strips (guarded
            #     by per-polygon min_dim < 1.25×gap and equiv_width < 1.1×gap),
            #     so legitimate terrain patches between distinct roads are left
            #     intact.
            # Only gap_fill_m is passed (min_feature_m=trim_width_m=0) so
            # normalize_road_mask_for_print only runs merge_close_road_gaps and
            # never deletes road polygons.
            if merged_roads_geom_local is not None and scale_factor and float(scale_factor) > 0:
                # Світові капи (див. ROAD_*_WORLD_CAP_M вище): на великих зонах
                # мм-модельні пороги вибухають і зливають квартали в плями.
                gap_fill_m = min(
                    model_mm_to_world_m(float(effective_gap_fill_mm), float(scale_factor)),
                    ROAD_GAP_FILL_WORLD_CAP_M,
                )
                if gap_fill_m and gap_fill_m > 0:
                    try:
                        min_road_feature_m = min(
                            model_mm_to_world_m(0.5, float(scale_factor)), 4.0,
                        )
                        # orphan_hole fills interior junction holes (triangular gaps at
                        # intersections). Keep it SMALL (0.5mm = 2.5m world) so only
                        # tight junction wedges are filled — not courtyards or medians.
                        orphan_hole_m = min(
                            model_mm_to_world_m(0.5, float(scale_factor)),
                            ROAD_ORPHAN_HOLE_WORLD_CAP_M,
                        )
                        filled = normalize_road_mask_for_print(
                            merged_roads_geom_local,
                            gap_fill_m=float(gap_fill_m),
                            min_feature_m=float(min_road_feature_m),
                            trim_width_m=0.0,
                            orphan_hole_width_m=float(orphan_hole_m),
                            zone_polygon=zone_polygon_local,
                            scale_factor=float(scale_factor) if scale_factor else 0.0,
                        )
                        if filled is not None and not getattr(filled, "is_empty", True):
                            merged_roads_geom_local = filled
                            print(
                                f"[INFO] {zone_prefix} merged_roads: gap-fill "
                                f"{effective_gap_fill_mm:.2f}mm model "
                                f"({gap_fill_m:.2f}m world) applied"
                            )
                    except Exception as exc:
                        print(f"[WARN] {zone_prefix} road gap-fill failed: {exc}")
                # ① Fill small ENCLOSED terrain islands (pockets fully surrounded by
                # road, e.g. between two close roads) up to ROAD_ISLAND_HOLE_MM. Safe:
                # only topological holes are filled (never welds parallel roads), and
                # it runs on the raw road geom that feeds BOTH inlay + canonical mask,
                # so the island is genuinely replaced by road in the print.
                try:
                    _isl_mm = float(os.environ.get("ROAD_ISLAND_HOLE_MM", "3.0"))
                except (TypeError, ValueError):
                    _isl_mm = 3.0
                if _isl_mm > 0 and merged_roads_geom_local is not None and scale_factor and float(scale_factor) > 0:
                    try:
                        _isl_cap = float(model_mm_to_world_m(_isl_mm, float(scale_factor))) ** 2
                        _a0 = float(getattr(merged_roads_geom_local, "area", 0.0) or 0.0)
                        _filled_isl = _fill_small_road_holes(
                            merged_roads_geom_local, max_hole_area_m2=_isl_cap
                        )
                        if _filled_isl is not None and not getattr(_filled_isl, "is_empty", True):
                            _a1 = float(getattr(_filled_isl, "area", 0.0) or 0.0)
                            if _a1 > _a0:
                                merged_roads_geom_local = _filled_isl
                                print(
                                    f"[INFO] {zone_prefix} road-island hole-fill "
                                    f"(<{_isl_mm}mm): +{_a1 - _a0:.1f} m2 road"
                                )
                    except Exception as exc:
                        print(f"[WARN] {zone_prefix} road-island hole-fill failed: {exc}")
                # ВІДКОЧЕНО 2026-06-25: thin-terrain-wall->road fill ЗЛИВАВ дороги в маси
                # + поглинав зелені медіани/смуги (не виключав parks/water; W*3≈54м
                # proximity на 1км-зоні), а стовбів НЕ виправляв (маска не доходить до
                # різу бази, 37-агент review). Прибрано → дороги/зелень як треба.
        # FALLBACK STAGE 1: якщо local-edges branch не дав результату, спробуємо G_roads (повний граф)
        need_fallback_to_global = (
            merged_roads_geom_local is None
            or getattr(merged_roads_geom_local, "is_empty", True)
        )
        if need_fallback_to_global and G_roads is not None and len(G_roads.edges) > 0:
            try:
                merged_roads_geom = build_road_polygons(
                    G_roads,
                    width_multiplier=float(road_width_multiplier_effective),
                    min_width_m=min_road_width_for_build,
                    scale_factor=scale_factor,
                )
                if merged_roads_geom is not None and not getattr(merged_roads_geom, "is_empty", True):
                    print(
                        f"[DEBUG] {zone_prefix} merged_roads_geom (fallback to global graph) created: "
                        f"area={getattr(merged_roads_geom, 'area', 0.0):.2f} m2"
                    )
            except Exception as exc:
                print(f"[WARN] {zone_prefix} G_roads fallback failed: {exc}")

        # FALLBACK STAGE 2: якщо все ще пусто — рятуємо буфером по semantic_centerlines (raw geometry)
        if (
            (merged_roads_geom_local is None or getattr(merged_roads_geom_local, "is_empty", True))
            and (merged_roads_geom is None or getattr(merged_roads_geom, "is_empty", True))
            and semantic_centerlines_local is not None
            and not getattr(semantic_centerlines_local, "is_empty", True)
        ):
            try:
                # Простий буфер по центральних лініях. Ширина за замовч. 3м world
                # (типова вулиця після врахування multiplier).
                fallback_width_m = max(
                    float(min_road_width_for_build) if min_road_width_for_build else 0.0,
                    1.5,
                )
                buffered = semantic_centerlines_local.buffer(fallback_width_m, cap_style=2, join_style=2)
                if buffered is not None and not getattr(buffered, "is_empty", True):
                    if zone_polygon_local is not None:
                        buffered = buffered.intersection(zone_polygon_local).buffer(0)
                    if buffered is not None and not getattr(buffered, "is_empty", True):
                        merged_roads_geom_local = buffered
                        merged_roads_geom_local_raw = buffered
                        print(
                            f"[FALLBACK] {zone_prefix} Roads recovered via centerline buffer "
                            f"({fallback_width_m:.2f}m world): area={buffered.area:.2f} m2"
                        )
            except Exception as exc:
                print(f"[WARN] {zone_prefix} Centerline buffer fallback failed: {exc}")

        if merged_roads_geom is None and merged_roads_geom_local is None:
            print(f"[DEBUG] {zone_prefix} all road sources exhausted, layer will be empty")
    except Exception as exc:
        print(f"[WARN] {zone_prefix} Failed to create merged_roads_geom: {exc}")
        merged_roads_geom = None

    if merged_roads_geom_local is None and merged_roads_geom is not None and global_center is not None:
        try:
            def to_local(x, y, z=None):
                x_local, y_local = global_center.to_local(x, y)
                return (x_local, y_local) if z is None else (x_local, y_local, z)

            merged_roads_geom_local_raw = transform_geometry(to_local, merged_roads_geom)
            if zone_polygon_local is not None:
                merged_roads_geom_local = merged_roads_geom_local_raw.intersection(zone_polygon_local)
                print(
                    f"[DEBUG] {zone_prefix} merged_roads_geom_local created: area={merged_roads_geom_local.area:.2f} m2, empty={merged_roads_geom_local.is_empty}"
                    if merged_roads_geom_local is not None and hasattr(merged_roads_geom_local, "area")
                    else f"[DEBUG] {zone_prefix} merged_roads_geom_local created"
                )
            else:
                merged_roads_geom_local = merged_roads_geom_local_raw
                print(
                    f"[DEBUG] {zone_prefix} merged_roads_geom_local created (no zone clipping): area={merged_roads_geom_local.area:.2f} m2"
                    if merged_roads_geom_local is not None and hasattr(merged_roads_geom_local, "area")
                    else f"[DEBUG] {zone_prefix} merged_roads_geom_local created (no zone clipping)"
                )
        except Exception as exc:
            print(f"[WARN] {zone_prefix} Failed to convert merged_roads_geom to local: {exc}")
            import traceback

            traceback.print_exc()
            merged_roads_geom_local = None
            merged_roads_geom_local_raw = None
    else:
        if merged_roads_geom_local is None:
            if merged_roads_geom is None:
                print(f"[DEBUG] {zone_prefix} merged_roads_geom is None, cannot create merged_roads_geom_local")
            if global_center is None:
                print(f"[DEBUG] {zone_prefix} global_center is None, cannot create merged_roads_geom_local")

    if (
        (merged_roads_geom_local is None or getattr(merged_roads_geom_local, "is_empty", True))
        and merged_roads_geom_local_raw is not None
        and not getattr(merged_roads_geom_local_raw, "is_empty", True)
    ):
        try:
            merged_roads_geom_local = (
                merged_roads_geom_local_raw.intersection(zone_polygon_local).buffer(0)
                if zone_polygon_local is not None
                else merged_roads_geom_local_raw.buffer(0)
            )
            if merged_roads_geom_local is not None and not getattr(merged_roads_geom_local, "is_empty", True):
                print(f"[WARN] {zone_prefix} restored merged_roads_geom_local from raw road mask fallback")
        except Exception:
            merged_roads_geom_local = None

    return RoadGeometryPreparationResult(
        merged_roads_geom=merged_roads_geom,
        merged_roads_geom_local=merged_roads_geom_local,
        merged_roads_geom_local_raw=merged_roads_geom_local_raw,
        semantic_centerlines_local=semantic_centerlines_local,
    )
