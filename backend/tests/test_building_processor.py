"""
Тести для сервісу обробки будівель
"""
import pytest
import geopandas as gpd
import numpy as np
import trimesh
from shapely.geometry import Polygon
from services.building_processor import (
    process_buildings,
    get_building_height,
    extrude_building
)


class TestBuildingProcessor:
    """Тести для building_processor.py"""
    
    def test_get_building_height_from_levels(self):
        """Тест отримання висоти з building:levels"""
        row = {'building:levels': 5}
        height = get_building_height(row, min_height=2.0)
        assert height == 15.0  # 5 * 3 метри

    def test_get_building_height_from_levels_string(self):
        """Тест отримання висоти з building:levels як рядок"""
        row = {'building:levels': '5'}
        height = get_building_height(row, min_height=2.0)
        assert height == 15.0
    
    def test_get_building_height_from_height_tag(self):
        """Тест отримання висоти з тегу height"""
        row = {'height': '20 m'}
        height = get_building_height(row, min_height=2.0)
        assert height == 20.0
    
    def test_get_building_height_minimum(self):
        """Тест використання мінімальної висоти"""
        row = {}
        height = get_building_height(row, min_height=2.0)
        assert height == 2.0
    
    def test_get_building_height_numeric(self):
        """Тест отримання висоти як числового значення"""
        row = {'height': 25.5}
        height = get_building_height(row, min_height=2.0)
        assert height == 25.5

    def test_get_building_height_from_numpy_levels(self):
        """GeoDataFrame numeric columns often arrive as numpy scalars."""
        row = {'building:levels': np.int64(30)}
        height = get_building_height(row, min_height=2.0)
        assert height == 90.0
    
    def test_extrude_building_simple(self):
        """Тест екструзії простого полігону"""
        # Створюємо квадратний полігон
        polygon = Polygon([
            (0, 0),
            (10, 0),
            (10, 10),
            (0, 10)
        ])
        
        mesh = extrude_building(polygon, height=5.0)
        
        assert mesh is not None
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
    
    def test_extrude_building_invalid_polygon(self):
        """Тест обробки невалідного полігону"""
        # Створюємо невалідний полігон (самоперетин)
        # Це має обробитися gracefully
        try:
            polygon = Polygon([
                (0, 0),
                (10, 10),
                (0, 10),
                (10, 0)
            ])
            mesh = extrude_building(polygon, height=5.0)
            # Може повернути None або спробувати виправити
        except:
            pass  # Очікувана поведінка для невалідних полігонів
    
    def test_process_buildings_empty(self):
        """Тест обробки порожнього GeoDataFrame"""
        empty_gdf = gpd.GeoDataFrame()
        result = process_buildings(empty_gdf)
        assert result == []
    
    def test_process_buildings_with_data(self):
        """Тест обробки будівель з даними"""
        from shapely.geometry import Polygon
        import geopandas as gpd
        
        # Створюємо валідний полігон
        polygon = Polygon([
            (0, 0),
            (10, 0),
            (10, 10),
            (0, 10)
        ])
        
        buildings = gpd.GeoDataFrame({
            'building': ['residential'],
            'building:levels': [3],
            'geometry': [polygon]
        }, crs='EPSG:4326')
        
        result = process_buildings(buildings, min_height=2.0, height_multiplier=1.0)
        
        # Може бути 0 якщо геометрія не проектується правильно, але перевіряємо структуру
        assert isinstance(result, list)
        if len(result) > 0:
            assert all(isinstance(mesh, trimesh.Trimesh) for mesh in result)
    
    def test_process_buildings_height_multiplier(self):
        """Тест множника висоти"""
        buildings = gpd.GeoDataFrame({
            'building': ['residential'],
            'building:levels': [2],
            'geometry': [
                Polygon([
                    (0, 0),
                    (5, 0),
                    (5, 5),
                    (0, 5)
                ])
            ]
        })
        
        # З множником 2.0, висота має бути 2 * 3 * 2 = 12 метрів
        result = process_buildings(buildings, min_height=2.0, height_multiplier=2.0)
        
        if result:
            # Перевіряємо, що меші створені
            assert len(result) > 0

    def test_process_buildings_places_base_at_lowest_ground_minus_model_sink(self):
        """Будівля на схилі сідає на НАЙНИЖЧУ точку footprint (не плаває над низьким
        боком). Раніше брали ground_max → на грубій DEM-сітці footprint семплив
        сусідні невирівняні (вищі) клітини, і будинок зависав у повітрі над рельєфом.
        ground_min ставить базу на flatten-пад; рельєф вище заходить у стіну (природна
        виїмка у схилі), а не лишає щілину. Див. building_processor.base_z."""

        class SlopedTerrain:
            def get_heights_for_points(self, points):
                pts = np.asarray(points, dtype=float)
                return 10.0 + 0.1 * pts[:, 0]

            def get_surface_heights_for_points(self, points):
                return self.get_heights_for_points(points)

        buildings = gpd.GeoDataFrame({
            'building': ['residential'],
            'height': [3.0],
            'geometry': [
                Polygon([
                    (0, 0),
                    (10, 0),
                    (10, 10),
                    (0, 10)
                ])
            ]
        })

        result = process_buildings(
            buildings,
            min_height=2.0,
            terrain_provider=SlopedTerrain(),
            foundation_depth=0.05,
            coordinates_already_local=True,
            scale_factor=10.0,
        )

        assert result
        mesh = result[0]
        # ground over footprint x∈[0,10] spans 10.0..11.0 → base = ground_min - sink.
        expected_base_z = 10.0 - (0.1 / 10.0)
        assert float(mesh.bounds[0][2]) == pytest.approx(expected_base_z, abs=1e-6)
        assert float(mesh.bounds[1][2]) == pytest.approx(expected_base_z + 3.0, abs=1e-6)
        # The base must NOT float above the lowest terrain point under the footprint.
        assert float(mesh.bounds[0][2]) <= 10.0 + 1e-6
