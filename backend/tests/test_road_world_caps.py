"""Світові капи доріг (ROAD_*_WORLD_CAP_M): на великих зонах мм-модельні
пороги друкованості вибухали у світових метрах (1.0мм = 10м на масштабі
0.1мм/м) і зливали щільний центр міста у суцільні плями."""
import geopandas as gpd
from shapely.geometry import LineString, box

from services.road_geometry_pipeline import prepare_road_geometry


class _GC:
    def to_local(self, x, y, z=None):
        return (x, y)

    def to_utm(self, lon, lat):
        return (lon, lat)


def _prepare(edges, zone):
    return prepare_road_geometry(
        G_roads=edges,
        scale_factor=0.1,  # зона 1500м на 150мм — кейс зливання
        road_width_multiplier_effective=1.0,
        min_printable_gap_mm=1.0,
        road_gap_fill_threshold_mm=1.0,
        min_gap_fill_floor_mm=0.5,
        global_center=_GC(),
        zone_polygon_local=zone,
    )


def test_parallel_streets_16m_apart_stay_separate_on_large_zone():
    # ДО фіксу: min-width 10м (вулиці впритул) + gap-fill 10м → одна пляма.
    # ПІСЛЯ: ширина капнута 9м (зазор 7м) + gap-fill капнутий 6м → 2 стрічки.
    edges = gpd.GeoDataFrame({
        "highway": ["residential", "residential"],
        "geometry": [LineString([(0, 0), (200, 0)]), LineString([(0, 16), (200, 16)])],
    })
    res = _prepare(edges, box(-50, -50, 250, 70))
    mask = res.merged_roads_geom_local
    assert mask is not None and not mask.is_empty
    parts = list(mask.geoms) if hasattr(mask, "geoms") else [mask]
    assert len(parts) == 2, f"вулиці за 16м злились ({len(parts)} частин) — капи не працюють"


def test_min_road_width_capped_at_street_scale():
    # Одна вулиця: ширина має бути ~9м (кап), а не 10м (1.0мм × 10м/мм)
    edges = gpd.GeoDataFrame({
        "highway": ["residential"],
        "geometry": [LineString([(0, 0), (200, 0)])],
    })
    res = _prepare(edges, box(-50, -50, 250, 50))
    mask = res.merged_roads_geom_local
    assert mask is not None and not mask.is_empty
    miny, maxy = mask.bounds[1], mask.bounds[3]
    width = maxy - miny
    assert 8.0 <= width <= 9.6, f"ширина вулиці {width:.2f}м — очікувався кап ~9м"
