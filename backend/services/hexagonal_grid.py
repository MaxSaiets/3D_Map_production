"""
Р“РµРЅРµСЂР°С†С–СЏ СЃС–С‚РєРё (С€РµСЃС‚РёРєСѓС‚РЅРёРєРё Р°Р±Рѕ РєРІР°РґСЂР°С‚Рё) РґР»СЏ РґС–Р»РµРЅРЅСЏ РєР°СЂС‚Рё РЅР° Р·РѕРЅРё.
Р’РёРєРѕСЂРёСЃС‚РѕРІСѓС” offset coordinates РґР»СЏ С–РґРµР°Р»СЊРЅРѕРіРѕ РїС–РґС…РѕРґСѓ Р±РµР· РіСЂР°РЅРёС†СЊ.
"""
import math
from typing import List, Tuple, Dict
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union


def hexagon_center_to_corner(center_x: float, center_y: float, size: float) -> List[Tuple[float, float]]:
    """
    РЎС‚РІРѕСЂСЋС” РєРѕРѕСЂРґРёРЅР°С‚Рё РІРµСЂС€РёРЅ С€РµСЃС‚РёРєСѓС‚РЅРёРєР° РЅР°РІРєРѕР»Рѕ С†РµРЅС‚СЂСѓ.
    
    Args:
        center_x: X РєРѕРѕСЂРґРёРЅР°С‚Р° С†РµРЅС‚СЂСѓ
        center_y: Y РєРѕРѕСЂРґРёРЅР°С‚Р° С†РµРЅС‚СЂСѓ
        size: Р Р°РґС–СѓСЃ С€РµСЃС‚РёРєСѓС‚РЅРёРєР° (РІС–РґСЃС‚Р°РЅСЊ РІС–Рґ С†РµРЅС‚СЂСѓ РґРѕ РІРµСЂС€РёРЅРё)
    
    Returns:
        РЎРїРёСЃРѕРє РєРѕРѕСЂРґРёРЅР°С‚ РІРµСЂС€РёРЅ (x, y)
    """
    corners = []
    # Р”Р»СЏ flat-top РѕСЂС–С”РЅС‚Р°С†С–С— (РїР»РѕСЃРєР° РІРµСЂС…РЅСЏ РіСЂР°РЅСЊ) - Р·РјС–С‰СѓС”РјРѕ РЅР° -30В°
    # Р¦Рµ Р·Р°Р±РµР·РїРµС‡СѓС” РїСЂР°РІРёР»СЊРЅРµ Р·'С”РґРЅР°РЅРЅСЏ РіСЂР°РЅСЊ РґРѕ РіСЂР°РЅС–
    for i in range(6):
        angle = math.pi / 3 * i - math.pi / 6  # -30В° Р·РјС–С‰РµРЅРЅСЏ РґР»СЏ flat-top
        x = center_x + size * math.cos(angle)
        y = center_y + size * math.sin(angle)
        corners.append((x, y))
    return corners


def generate_square_grid(
    bbox: Tuple[float, float, float, float],
    square_size_m: float = 400.0  # 0.4 РєРј = 400 Рј
) -> List[Dict]:
    """
    Р“РµРЅРµСЂСѓС” РєРІР°РґСЂР°С‚РЅСѓ СЃС–С‚РєСѓ РґР»СЏ Р·Р°РґР°РЅРѕРіРѕ bbox.
    
    Args:
        bbox: Bounding box (minx, miny, maxx, maxy) РІ РјРµС‚СЂР°С…
        square_size_m: Р РѕР·РјС–СЂ РєРІР°РґСЂР°С‚Р° (СЃС‚РѕСЂРѕРЅР°) РІ РјРµС‚СЂР°С…
    
    Returns:
        РЎРїРёСЃРѕРє СЃР»РѕРІРЅРёРєС–РІ Р· С–РЅС„РѕСЂРјР°С†С–С”СЋ РїСЂРѕ РєРѕР¶РµРЅ РєРІР°РґСЂР°С‚
    """
    minx, miny, maxx, maxy = bbox
    
    # РџРµСЂРµРІС–СЂРєР° РІР°Р»С–РґРЅРѕСЃС‚С– bbox
    if maxx <= minx or maxy <= miny:
        raise ValueError(f"РќРµРІС–СЂРЅРёР№ bbox: minx={minx}, maxx={maxx}, miny={miny}, maxy={maxy}")
    
    # Р РѕР·СЂР°С…РѕРІСѓС”РјРѕ РєС–Р»СЊРєС–СЃС‚СЊ РєРІР°РґСЂР°С‚С–РІ
    cols = int(math.ceil((maxx - minx) / square_size_m)) + 1
    rows = int(math.ceil((maxy - miny) / square_size_m)) + 1
    
    print(f"[DEBUG] Square grid: bbox=({minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f})")
    print(f"[DEBUG] Square size: {square_size_m}m")
    print(f"[DEBUG] Grid: cols={cols}, rows={rows}, total cells: {cols * rows}")
    
    squares = []
    for row in range(rows):
        for col in range(cols):
            # Р¦РµРЅС‚СЂ РєРІР°РґСЂР°С‚Р°
            center_x = minx + col * square_size_m + square_size_m / 2.0
            center_y = miny + row * square_size_m + square_size_m / 2.0
            
            # РЎС‚РІРѕСЂСЋС”РјРѕ РєРІР°РґСЂР°С‚
            half_size = square_size_m / 2.0
            square_polygon = box(
                center_x - half_size,
                center_y - half_size,
                center_x + half_size,
                center_y + half_size
            )
            
            # РџРµСЂРµРІС–СЂСЏС”РјРѕ, С‡Рё РєРІР°РґСЂР°С‚ РїРµСЂРµС‚РёРЅР°С”С‚СЊСЃСЏ Р· bbox
            if square_polygon.intersects(box(minx, miny, maxx, maxy)):
                squares.append({
                    'id': f'square_{row}_{col}',
                    'center': (center_x, center_y),
                    'polygon': square_polygon,
                    'row': row,
                    'col': col
                })
    
    print(f"[DEBUG] Р—РіРµРЅРµСЂРѕРІР°РЅРѕ {len(squares)} РєРІР°РґСЂР°С‚С–РІ")
    return squares


def generate_hexagonal_grid(
    bbox: Tuple[float, float, float, float],
    hex_size_m: float = 400.0  # 0.4 РєРј = 400 Рј
) -> List[Dict]:
    """
    Р“РµРЅРµСЂСѓС” РіРµРєСЃР°РіРѕРЅР°Р»СЊРЅСѓ СЃС–С‚РєСѓ РґР»СЏ Р·Р°РґР°РЅРѕРіРѕ bbox.
    
    Args:
        bbox: Bounding box (minx, miny, maxx, maxy) РІ РјРµС‚СЂР°С…
        hex_size_m: Р РѕР·РјС–СЂ С€РµСЃС‚РёРєСѓС‚РЅРёРєР° (СЂР°РґС–СѓСЃ РІС–Рґ С†РµРЅС‚СЂСѓ РґРѕ РІРµСЂС€РёРЅРё) РІ РјРµС‚СЂР°С…
    
    Returns:
        РЎРїРёСЃРѕРє СЃР»РѕРІРЅРёРєС–РІ Р· С–РЅС„РѕСЂРјР°С†С–С”СЋ РїСЂРѕ РєРѕР¶РµРЅ С€РµСЃС‚РёРєСѓС‚РЅРёРє:
        {
            'id': str,
            'center': (x, y),
            'polygon': Polygon,
            'row': int,
            'col': int
        }
    """
    minx, miny, maxx, maxy = bbox
    
    # РџРµСЂРµРІС–СЂРєР° РІР°Р»С–РґРЅРѕСЃС‚С– bbox
    if maxx <= minx or maxy <= miny:
        raise ValueError(f"РќРµРІС–СЂРЅРёР№ bbox: minx={minx}, maxx={maxx}, miny={miny}, maxy={maxy}")
    
    # Р РѕР·РјС–СЂРё С€РµСЃС‚РёРєСѓС‚РЅРёРєР° РґР»СЏ offset coordinates
    # РЁРёСЂРёРЅР° (РіРѕСЂРёР·РѕРЅС‚Р°Р»СЊРЅР° РІС–РґСЃС‚Р°РЅСЊ РјС–Р¶ С†РµРЅС‚СЂР°РјРё): sqrt(3) * size
    # Р’РёСЃРѕС‚Р° (РІРµСЂС‚РёРєР°Р»СЊРЅР° РІС–РґСЃС‚Р°РЅСЊ РјС–Р¶ С†РµРЅС‚СЂР°РјРё): 1.5 * size
    hex_width = math.sqrt(3) * hex_size_m
    hex_height = 1.5 * hex_size_m  # РџСЂР°РІРёР»СЊРЅР° РІРµСЂС‚РёРєР°Р»СЊРЅР° РІС–РґСЃС‚Р°РЅСЊ РґР»СЏ offset
    
    # Р РѕР·СЂР°С…РѕРІСѓС”РјРѕ РєС–Р»СЊРєС–СЃС‚СЊ С€РµСЃС‚РёРєСѓС‚РЅРёРєС–РІ
    # РћРџРўРРњР†Р—РђР¦Р†РЇ: РћР±РјРµР¶СѓС”РјРѕ РєС–Р»СЊРєС–СЃС‚СЊ РґР»СЏ РІРµР»РёРєРёС… РѕР±Р»Р°СЃС‚РµР№
    cols = int(math.ceil((maxx - minx) / hex_width)) + 2  # +2 РґР»СЏ Р·Р°РїР°СЃСѓ
    rows = int(math.ceil((maxy - miny) / hex_height)) + 2  # +2 РґР»СЏ Р·Р°РїР°СЃСѓ
    
    print(f"[DEBUG] Hexagonal grid: bbox=({minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f})")
    print(f"[DEBUG] Hex size: {hex_size_m}m, width: {hex_width:.2f}m, height: {hex_height:.2f}m")
    print(f"[DEBUG] Grid: cols={cols}, rows={rows}, total cells: {cols * rows}")
    
    # РћРџРўРРњР†Р—РђР¦Р†РЇ: РћР±РјРµР¶СѓС”РјРѕ РјР°РєСЃРёРјСѓРј РґР»СЏ РїСЂРѕРґСѓРєС‚РёРІРЅРѕСЃС‚С–
    MAX_HEXAGONS = 50000
    total_candidate_cells = cols * rows
    if total_candidate_cells > MAX_HEXAGONS:
        print(
            f"[WARN] Large hex grid requested: {cols}x{rows} ({total_candidate_cells} candidate cells). "
            "Preserving full bbox coverage; generation may be slower."
        )
    
    hexagons = []
    hex_id = 0
    
    # Р“РµРЅРµСЂСѓС”РјРѕ С€РµСЃС‚РёРєСѓС‚РЅРёРєРё Р· offset coordinates
    for row in range(rows):
        for col in range(cols):
            # Offset coordinates: РЅРµРїР°СЂРЅС– СЂСЏРґРё Р·РјС–С‰РµРЅС– РЅР° РїРѕР»РѕРІРёРЅСѓ С€РёСЂРёРЅРё
            if row % 2 == 0:
                center_x = minx + col * hex_width
            else:
                center_x = minx + col * hex_width + hex_width / 2.0
            
            # Р’РµСЂС‚РёРєР°Р»СЊРЅР° РїРѕР·РёС†С–СЏ: РїСЂРѕСЃС‚Рѕ row * hex_height
            center_y = miny + row * hex_height
            
            # РЎС‚РІРѕСЂСЋС”РјРѕ С€РµСЃС‚РёРєСѓС‚РЅРёРє
            corners = hexagon_center_to_corner(center_x, center_y, hex_size_m)
            polygon = Polygon(corners)
            
            # РџРµСЂРµРІС–СЂСЏС”РјРѕ, С‡Рё С€РµСЃС‚РёРєСѓС‚РЅРёРє РїРµСЂРµС‚РёРЅР°С”С‚СЊСЃСЏ Р· bbox
            bbox_poly = Polygon([
                (minx, miny),
                (maxx, miny),
                (maxx, maxy),
                (minx, maxy)
            ])
            
            if polygon.intersects(bbox_poly):
                hex_id_str = f"hex_{row}_{col}"
                hexagons.append({
                    'id': hex_id_str,
                    'center': (center_x, center_y),
                    'polygon': polygon,
                    'row': row,
                    'col': col,
                    'bounds': polygon.bounds  # (minx, miny, maxx, maxy)
                })
                hex_id += 1
    
    return hexagons


def generate_circular_grid(
    bbox: Tuple[float, float, float, float],
    radius_m: float = 200.0  # Р Р°РґС–СѓСЃ РєСЂСѓРіР° РІ РјРµС‚СЂР°С… (РґС–Р°РјРµС‚СЂ = 2*radius)
) -> List[Dict]:
    """
    Р“РµРЅРµСЂСѓС” СЃС–С‚РєСѓ Р· РєСЂСѓРіРѕРІРёРјРё Р·РѕРЅР°РјРё РґР»СЏ Р·Р°РґР°РЅРѕРіРѕ bbox.
    РљРѕР¶РµРЅ РєСЂСѓРі РјР°С” РѕРґРЅР°РєРѕРІСѓ РїР»РѕС‰Сѓ (СЂС–РІРЅС– Р·РѕРЅРё).
    
    Args:
        bbox: Bounding box (minx, miny, maxx, maxy) РІ РјРµС‚СЂР°С…
        radius_m: Р Р°РґС–СѓСЃ РєСЂСѓРіР° РІ РјРµС‚СЂР°С…
    
    Returns:
        РЎРїРёСЃРѕРє СЃР»РѕРІРЅРёРєС–РІ Р· С–РЅС„РѕСЂРјР°С†С–С”СЋ РїСЂРѕ РєРѕР¶РЅСѓ РєСЂСѓРіР»Сѓ Р·РѕРЅСѓ
    """
    minx, miny, maxx, maxy = bbox
    
    if maxx <= minx or maxy <= miny:
        raise ValueError(f"РќРµРІС–СЂРЅРёР№ bbox: minx={minx}, maxx={maxx}, miny={miny}, maxy={maxy}")
    
    diameter = 2.0 * radius_m
    cols = int(math.ceil((maxx - minx) / diameter)) + 1
    rows = int(math.ceil((maxy - miny) / diameter)) + 1
    
    print(f"[DEBUG] Circular grid: bbox=({minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f})")
    print(f"[DEBUG] Radius: {radius_m}m, diameter: {diameter}m")
    print(f"[DEBUG] Grid: cols={cols}, rows={rows}, total cells: {cols * rows}")
    
    circles = []
    for row in range(rows):
        for col in range(cols):
            center_x = minx + col * diameter + radius_m
            center_y = miny + row * diameter + radius_m
            
            circle_poly = Point(center_x, center_y).buffer(radius_m)
            
            bbox_poly = box(minx, miny, maxx, maxy)
            if circle_poly.intersects(bbox_poly):
                circles.append({
                    'id': f'circle_{row}_{col}',
                    'center': (center_x, center_y),
                    'polygon': circle_poly,
                    'row': row,
                    'col': col,
                    'bounds': circle_poly.bounds
                })
    
    print(f"[DEBUG] Р—РіРµРЅРµСЂРѕРІР°РЅРѕ {len(circles)} РєСЂСѓРіС–РІ")
    return circles


def hexagons_to_geojson(cells: List[Dict], to_wgs84=None) -> Dict:
    """
    РљРѕРЅРІРµСЂС‚СѓС” СЃРїРёСЃРѕРє РєР»С–С‚РёРЅРѕРє (С€РµСЃС‚РёРєСѓС‚РЅРёРєРё Р°Р±Рѕ РєРІР°РґСЂР°С‚Рё) РІ GeoJSON С„РѕСЂРјР°С‚.
    
    Args:
        cells: РЎРїРёСЃРѕРє СЃР»РѕРІРЅРёРєС–РІ Р· С–РЅС„РѕСЂРјР°С†С–С”СЋ РїСЂРѕ РєР»С–С‚РёРЅРєРё (hexagons Р°Р±Рѕ squares)
        to_wgs84: Р¤СѓРЅРєС†С–СЏ РґР»СЏ РєРѕРЅРІРµСЂС‚Р°С†С–С— UTM РєРѕРѕСЂРґРёРЅР°С‚ РІ WGS84 (lat, lon). 
                  РЇРєС‰Рѕ None, РєРѕРѕСЂРґРёРЅР°С‚Рё РІРІР°Р¶Р°СЋС‚СЊСЃСЏ РІР¶Рµ РІ WGS84.
    
    Returns:
        GeoJSON FeatureCollection
    """
    features = []
    for cell_data in cells:
        polygon = cell_data['polygon']
        
        # РљРѕРЅРІРµСЂС‚СѓС”РјРѕ РєРѕРѕСЂРґРёРЅР°С‚Рё Р· UTM РІ lat/lon СЏРєС‰Рѕ РїРѕС‚СЂС–Р±РЅРѕ
        if to_wgs84 is not None:
            # РљРѕРЅРІРµСЂС‚СѓС”РјРѕ РєРѕР¶РЅСѓ С‚РѕС‡РєСѓ
            coords_utm = list(polygon.exterior.coords)
            coords_wgs84 = []
            for x, y in coords_utm:
                try:
                    lon, lat = to_wgs84(x, y)  # to_wgs84 РїРѕРІРµСЂС‚Р°С” (lon, lat)
                    coords_wgs84.append([lon, lat])  # GeoJSON РІРёРєРѕСЂРёСЃС‚РѕРІСѓС” [lon, lat]
                except Exception as e:
                    print(f"[WARN] РџРѕРјРёР»РєР° РєРѕРЅРІРµСЂС‚Р°С†С–С— РєРѕРѕСЂРґРёРЅР°С‚ ({x}, {y}): {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback: РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”РјРѕ РѕСЂРёРіС–РЅР°Р»СЊРЅС– РєРѕРѕСЂРґРёРЅР°С‚Рё (РЅРµРїСЂР°РІРёР»СЊРЅРѕ, Р°Р»Рµ РєСЂР°С‰Рµ РЅС–Р¶ РїРѕРјРёР»РєР°)
                    coords_wgs84.append([x, y])
            coords = [coords_wgs84]
        else:
            # РљРѕРѕСЂРґРёРЅР°С‚Рё РІР¶Рµ РІ РїСЂР°РІРёР»СЊРЅРѕРјСѓ С„РѕСЂРјР°С‚С–
            coords = [[list(p) for p in polygon.exterior.coords]]
        
        feature = {
            'type': 'Feature',
            'id': cell_data['id'],
            'geometry': {
                'type': 'Polygon',
                'coordinates': coords
            },
            'properties': {
                'id': cell_data['id'],
                'row': cell_data['row'],
                'col': cell_data['col'],
                'center': cell_data['center']
            }
        }
        features.append(feature)
    
    return {
        'type': 'FeatureCollection',
        'features': features
    }


def calculate_grid_center_from_geojson(geojson: Dict, to_wgs84=None) -> Tuple[float, float]:
    """
    РћР±С‡РёСЃР»СЋС” РѕРїС‚РёРјР°Р»СЊРЅРёР№ С†РµРЅС‚СЂ РґР»СЏ РІСЃС–С”С— СЃС–С‚РєРё РЅР° РѕСЃРЅРѕРІС– GeoJSON features.
    Р¦Рµ Р·Р°Р±РµР·РїРµС‡СѓС”, С‰Рѕ РІСЃС– РєР»С–С‚РёРЅРєРё РІРёРєРѕСЂРёСЃС‚РѕРІСѓСЋС‚СЊ РѕРґРЅСѓ С‚РѕС‡РєСѓ РІС–РґР»С–РєСѓ (0,0).
    
    Args:
        geojson: GeoJSON FeatureCollection Р· РєР»С–С‚РёРЅРєР°РјРё СЃС–С‚РєРё
        to_wgs84: Р¤СѓРЅРєС†С–СЏ РґР»СЏ РєРѕРЅРІРµСЂС‚Р°С†С–С— РєРѕРѕСЂРґРёРЅР°С‚ РІ WGS84 (СЏРєС‰Рѕ РєРѕРѕСЂРґРёРЅР°С‚Рё РІ UTM)
    
    Returns:
        Tuple (center_lat, center_lon) - С†РµРЅС‚СЂ РІСЃС–С”С— СЃС–С‚РєРё РІ WGS84
    """
    if not geojson or 'features' not in geojson:
        raise ValueError("GeoJSON РЅРµ РјС–СЃС‚РёС‚СЊ features")
    
    features = geojson['features']
    if not features:
        raise ValueError("GeoJSON РЅРµ РјС–СЃС‚РёС‚СЊ Р¶РѕРґРЅРёС… features")
    
    # Р—Р±РёСЂР°С”РјРѕ РІСЃС– РєРѕРѕСЂРґРёРЅР°С‚Рё Р· СѓСЃС–С… features
    all_lons = []
    all_lats = []
    
    for feature in features:
        geometry = feature.get('geometry', {})
        if geometry.get('type') != 'Polygon':
            continue
        
        coordinates = geometry.get('coordinates', [])
        if not coordinates or len(coordinates) == 0:
            continue
        
        # Р—РЅР°С…РѕРґРёРјРѕ РєРѕРѕСЂРґРёРЅР°С‚Рё Р· feature
        all_coords = [coord for ring in coordinates for coord in ring]
        feature_lons = [coord[0] for coord in all_coords]
        feature_lats = [coord[1] for coord in all_coords]
        
        all_lons.extend(feature_lons)
        all_lats.extend(feature_lats)
    
    if len(all_lons) == 0 or len(all_lats) == 0:
        raise ValueError("РќРµ РІРґР°Р»РѕСЃСЏ РѕС‚СЂРёРјР°С‚Рё РєРѕРѕСЂРґРёРЅР°С‚Рё Р· features")
    
    # РћР±С‡РёСЃР»СЋС”РјРѕ С†РµРЅС‚СЂ (СЃРµСЂРµРґРЅС” Р·РЅР°С‡РµРЅРЅСЏ)
    center_lon = (min(all_lons) + max(all_lons)) / 2.0
    center_lat = (min(all_lats) + max(all_lats)) / 2.0
    
    return (center_lat, center_lon)


def validate_hexagonal_grid(hexagons: List[Dict], tolerance: float = 0.01) -> Tuple[bool, List[str]]:
    """
    РџРµСЂРµРІС–СЂСЏС”, С‡Рё РіРµРєСЃР°РіРѕРЅР°Р»СЊРЅР° СЃС–С‚РєР° С–РґРµР°Р»СЊРЅРѕ РїС–РґС…РѕРґРёС‚СЊ Р±РµР· РіСЂР°РЅРёС†СЊ.
    
    Args:
        hexagons: РЎРїРёСЃРѕРє С€РµСЃС‚РёРєСѓС‚РЅРёРєС–РІ
        tolerance: Р”РѕРїСѓСЃРє РґР»СЏ РїРµСЂРµРІС–СЂРєРё (РІ РјРµС‚СЂР°С…)
    
    Returns:
        Tuple (is_valid, list_of_errors)
    """
    errors = []
    
    # РџРµСЂРµРІС–СЂРєР° 1: Р’СЃС– С€РµСЃС‚РёРєСѓС‚РЅРёРєРё РјР°СЋС‚СЊ РѕРґРЅР°РєРѕРІРёР№ СЂРѕР·РјС–СЂ
    if len(hexagons) == 0:
        return False, ["РќРµРјР°С” С€РµСЃС‚РёРєСѓС‚РЅРёРєС–РІ"]
    
    # РџРµСЂРµРІС–СЂРєР° 2: РЁРµСЃС‚РёРєСѓС‚РЅРёРєРё РЅРµ РїРµСЂРµРєСЂРёРІР°СЋС‚СЊСЃСЏ (РєСЂС–Рј СЃСѓСЃС–РґРЅС–С…)
    for i, hex1 in enumerate(hexagons):
        for j, hex2 in enumerate(hexagons[i+1:], start=i+1):
            intersection = hex1['polygon'].intersection(hex2['polygon'])
            if intersection.area > tolerance:
                # РџРµСЂРµРІС–СЂСЏС”РјРѕ, С‡Рё РІРѕРЅРё СЃСѓСЃС–РґРЅС–
                row_diff = abs(hex1['row'] - hex2['row'])
                col_diff = abs(hex1['col'] - hex2['col'])
                
                # РЎСѓСЃС–РґРЅС– С€РµСЃС‚РёРєСѓС‚РЅРёРєРё РјР°СЋС‚СЊ СЂС–Р·РЅРёС†СЋ РІ РєРѕРѕСЂРґРёРЅР°С‚Р°С… <= 1
                if not (row_diff <= 1 and col_diff <= 1):
                    errors.append(f"РЁРµСЃС‚РёРєСѓС‚РЅРёРєРё {hex1['id']} С‚Р° {hex2['id']} РїРµСЂРµРєСЂРёРІР°СЋС‚СЊСЃСЏ")
    
    # РџРµСЂРµРІС–СЂРєР° 3: РЎСѓС†С–Р»СЊРЅРµ РїРѕРєСЂРёС‚С‚СЏ (РїРµСЂРµРІС–СЂСЏС”РјРѕ, С‡Рё РЅРµРјР°С” РІРµР»РёРєРёС… РїСЂРѕРіР°Р»РёРЅ)
    # Р¦Рµ СЃРєР»Р°РґРЅС–С€Рµ, С‚РѕРјСѓ РїСЂРѕСЃС‚Рѕ РїРµСЂРµРІС–СЂСЏС”РјРѕ, С‡Рё С” РґРѕСЃС‚Р°С‚РЅСЊРѕ С€РµСЃС‚РёРєСѓС‚РЅРёРєС–РІ
    
    is_valid = len(errors) == 0
    return is_valid, errors


