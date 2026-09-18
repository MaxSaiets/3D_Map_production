# -*- coding: utf-8 -*-
"""Відомі вершини (пресети) + словник назв для агента.

Координати вершин — з відкритих джерел (Wikipedia/OSM), area_km підібрано так, щоб уся
піраміда/масив помістилися й гора читалася (перевірено на Матергорні: 3,6 км)."""
from __future__ import annotations

PRESETS: list[dict] = [
    {"id": "matterhorn", "lat": 45.97640, "lon": 7.65860, "elev": 4478, "area_km": 3.6,
     "name": {"uk": "Матергорн", "en": "Matterhorn", "de": "Matterhorn", "pl": "Matterhorn", "fr": "Cervin", "es": "Cervino"},
     "country": {"uk": "Швейцарія / Італія", "en": "Switzerland / Italy"}, "aliases": ["matterhorn", "матергорн", "маттерхорн", "cervino", "cervin", "церматт", "zermatt"]},
    {"id": "hoverla", "lat": 48.16000, "lon": 24.50028, "elev": 2061, "area_km": 5.0,
     "name": {"uk": "Говерла", "en": "Hoverla", "de": "Howerla", "pl": "Howerla", "fr": "Hoverla", "es": "Hoverla"},
     "country": {"uk": "Україна, Чорногора", "en": "Ukraine, Chornohora"}, "aliases": ["говерла", "hoverla", "howerla", "чорногор"]},
    {"id": "petros", "lat": 48.17194, "lon": 24.42056, "elev": 2020, "area_km": 4.0,
     "name": {"uk": "Петрос", "en": "Petros", "de": "Petros", "pl": "Petros", "fr": "Petros", "es": "Petros"},
     "country": {"uk": "Україна, Чорногора", "en": "Ukraine, Chornohora"}, "aliases": ["петрос", "petros"]},
    {"id": "pip_ivan", "lat": 48.04861, "lon": 24.62806, "elev": 2028, "area_km": 4.0,
     "name": {"uk": "Піп Іван Чорногірський", "en": "Pip Ivan", "de": "Pip Iwan", "pl": "Pop Iwan", "fr": "Pip Ivan", "es": "Pip Ivan"},
     "country": {"uk": "Україна, Чорногора", "en": "Ukraine, Chornohora"}, "aliases": ["піп іван", "pip ivan", "pop iwan", "обсерватор"]},
    {"id": "mont_blanc", "lat": 45.83262, "lon": 6.86521, "elev": 4808, "area_km": 7.0,
     "name": {"uk": "Монблан", "en": "Mont Blanc", "de": "Mont Blanc", "pl": "Mont Blanc", "fr": "Mont Blanc", "es": "Mont Blanc"},
     "country": {"uk": "Франція / Італія", "en": "France / Italy"}, "aliases": ["монблан", "mont blanc", "montblanc", "monte bianco", "шамоні", "chamonix"]},
    {"id": "eiger", "lat": 46.57750, "lon": 8.00528, "elev": 3967, "area_km": 5.0,
     "name": {"uk": "Айгер", "en": "Eiger", "de": "Eiger", "pl": "Eiger", "fr": "Eiger", "es": "Eiger"},
     "country": {"uk": "Швейцарія", "en": "Switzerland"}, "aliases": ["айгер", "eiger", "ейгер", "гріндельвальд", "grindelwald"]},
    {"id": "everest", "lat": 27.98806, "lon": 86.92528, "elev": 8849, "area_km": 12.0,
     "name": {"uk": "Еверест", "en": "Everest", "de": "Everest", "pl": "Everest", "fr": "Everest", "es": "Everest"},
     "country": {"uk": "Непал / Китай", "en": "Nepal / China"}, "aliases": ["еверест", "everest", "джомолунгма", "сагарматха", "chomolungma", "sagarmatha"]},
    {"id": "k2", "lat": 35.88139, "lon": 76.51333, "elev": 8611, "area_km": 10.0,
     "name": {"uk": "K2 (Чогорі)", "en": "K2", "de": "K2", "pl": "K2", "fr": "K2", "es": "K2"},
     "country": {"uk": "Пакистан / Китай", "en": "Pakistan / China"}, "aliases": ["k2", "к2", "чогорі", "chogori", "годуін-остен"]},
    {"id": "elbrus", "lat": 43.35500, "lon": 42.43917, "elev": 5642, "area_km": 10.0,
     "name": {"uk": "Ельбрус", "en": "Elbrus", "de": "Elbrus", "pl": "Elbrus", "fr": "Elbrouz", "es": "Elbrús"},
     "country": {"uk": "Кавказ", "en": "Caucasus"}, "aliases": ["ельбрус", "elbrus", "эльбрус"]},
    {"id": "fuji", "lat": 35.36083, "lon": 138.72750, "elev": 3776, "area_km": 20.0,
     "name": {"uk": "Фудзі", "en": "Mount Fuji", "de": "Fuji", "pl": "Fudżi", "fr": "Mont Fuji", "es": "Monte Fuji"},
     "country": {"uk": "Японія", "en": "Japan"}, "aliases": ["фудзі", "фудзіяма", "fuji", "fujiyama", "fujisan", "富士山"]},
    {"id": "kilimanjaro", "lat": -3.06583, "lon": 37.35861, "elev": 5895, "area_km": 30.0,
     "name": {"uk": "Кіліманджаро", "en": "Kilimanjaro", "de": "Kilimandscharo", "pl": "Kilimandżaro", "fr": "Kilimandjaro", "es": "Kilimanjaro"},
     "country": {"uk": "Танзанія", "en": "Tanzania"}, "aliases": ["кіліманджаро", "kilimanjaro", "кибо", "kibo"]},
    {"id": "aconcagua", "lat": -32.65333, "lon": -70.01083, "elev": 6961, "area_km": 12.0,
     "name": {"uk": "Аконкагуа", "en": "Aconcagua", "de": "Aconcagua", "pl": "Aconcagua", "fr": "Aconcagua", "es": "Aconcagua"},
     "country": {"uk": "Аргентина", "en": "Argentina"}, "aliases": ["аконкагуа", "aconcagua"]},
    {"id": "denali", "lat": 63.06917, "lon": -151.00694, "elev": 6190, "area_km": 15.0,
     "name": {"uk": "Деналі", "en": "Denali", "de": "Denali", "pl": "Denali", "fr": "Denali", "es": "Denali"},
     "country": {"uk": "США, Аляска", "en": "USA, Alaska"}, "aliases": ["деналі", "denali", "мак-кінлі", "mckinley"]},
    {"id": "tre_cime", "lat": 46.61889, "lon": 12.30306, "elev": 2999, "area_km": 3.0,
     "name": {"uk": "Тре-Чіме-ді-Лаваредо", "en": "Tre Cime di Lavaredo", "de": "Drei Zinnen", "pl": "Tre Cime", "fr": "Tre Cime", "es": "Tre Cime"},
     "country": {"uk": "Італія, Доломіти", "en": "Italy, Dolomites"}, "aliases": ["тре чіме", "tre cime", "drei zinnen", "три зубці", "доломіт", "dolomit"]},
    {"id": "rysy", "lat": 49.17944, "lon": 20.08833, "elev": 2503, "area_km": 4.0,
     "name": {"uk": "Риси (Татри)", "en": "Rysy (Tatras)", "de": "Rysy", "pl": "Rysy", "fr": "Rysy", "es": "Rysy"},
     "country": {"uk": "Польща / Словаччина", "en": "Poland / Slovakia"}, "aliases": ["риси", "rysy", "татр", "tatra", "tatry", "морське око", "morskie oko"]},
    {"id": "olympus", "lat": 40.08556, "lon": 22.35861, "elev": 2917, "area_km": 8.0,
     "name": {"uk": "Олімп", "en": "Mount Olympus", "de": "Olymp", "pl": "Olimp", "fr": "Olympe", "es": "Olimpo"},
     "country": {"uk": "Греція", "en": "Greece"}, "aliases": ["олімп", "olympus", "olymp", "мітікас", "mytikas"]},
    {"id": "ai_petri", "lat": 44.45111, "lon": 34.05639, "elev": 1234, "area_km": 5.0,
     "name": {"uk": "Ай-Петрі", "en": "Ai-Petri", "de": "Ai-Petri", "pl": "Aj-Petri", "fr": "Aï-Petri", "es": "Ai-Petri"},
     "country": {"uk": "Україна, Крим", "en": "Ukraine, Crimea"}, "aliases": ["ай-петрі", "ай петрі", "ai-petri", "ai petri", "крим", "crimea"]},
    {"id": "table_mountain", "lat": -33.96250, "lon": 18.40361, "elev": 1085, "area_km": 6.0,
     "name": {"uk": "Столова гора", "en": "Table Mountain", "de": "Tafelberg", "pl": "Góra Stołowa", "fr": "Montagne de la Table", "es": "Montaña de la Mesa"},
     "country": {"uk": "ПАР, Кейптаун", "en": "South Africa"}, "aliases": ["столова гора", "table mountain", "tafelberg", "кейптаун", "cape town"]},
]


def find_by_alias(text: str):
    t = text.lower()
    best = None
    for p in PRESETS:
        for a in p["aliases"] + [n.lower() for n in p["name"].values()]:
            if a and a in t and (best is None or len(a) > best[0]):
                best = (len(a), p)
    return best[1] if best else None


def public(locale: str = "uk") -> list[dict]:
    out = []
    for p in PRESETS:
        out.append({"id": p["id"], "lat": p["lat"], "lon": p["lon"], "elev": p["elev"], "area_km": p["area_km"],
                    "name": p["name"].get(locale) or p["name"]["en"], "country": p["country"].get(locale) or p["country"]["en"],
                    "photo": f"/mountains/presets/{p['id']}.jpg"})
    return out
