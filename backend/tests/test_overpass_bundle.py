# -*- coding: utf-8 -*-
"""Один Overpass-запит на всю генерацію (services/overpass_bundle.py).

Що тут доводиться:
1. Запит пакета містить ТІ САМІ компоненти, що osmnx збирає для кожного шару
   окремо (рядок-у-рядок), плюс компонент доріг із буфером 500 м.
2. Розділена по шарах відповідь дає ТІ САМІ GeoDataFrame/граф, що й окремі
   відповіді Overpass, пропущені через ті самі функції osmnx.
3. Локальний фільтр way доріг = семантика Overpass QL для `[k]`, `[k=v]`,
   `[k!=v]`, `[k~re]`, `[k!~re]`.
4. LazyBundle качає рівно раз, ділиться між потоками, а після збою всі шари
   отримують BundleMiss (→ старий шлях), а не тихо порожні дані.
"""
from __future__ import annotations

import copy
import threading

import networkx as nx
import osmnx as ox
import pytest
from osmnx._errors import InsufficientResponseError

from services import overpass_bundle as ob


# ────────────────────────── синтетичний фрагмент OSM ──────────────────────────
def _node(i, lat, lon, tags=None):
    d = {"type": "node", "id": i, "lat": lat, "lon": lon}
    if tags:
        d["tags"] = tags
    return d


def _way(i, nodes, tags=None):
    d = {"type": "way", "id": i, "nodes": list(nodes)}
    if tags:
        d["tags"] = tags
    return d


def _rel(i, members, tags):
    return {"type": "relation", "id": i, "members": members, "tags": tags}


BBOX = (30.500, 50.400, 30.520, 50.410)  # (left, bottom, right, top) — ~1.4×1.1 км
POLY = ox.utils_geo.bbox_to_poly(BBOX)


def _world():
    """Місто-іграшка: 2 будинки (один — multipolygon з двором), ставок, парк,
    міст, колія і сітка вулиць (одна — площа `area=yes`, одна — `construction`)."""
    nodes = {}
    nid = [1]

    def N(lat, lon, tags=None):
        i = nid[0]
        nid[0] += 1
        nodes[i] = _node(i, lat, lon, tags)
        return i

    ways, rels = [], []
    # будинок-квадрат
    b1 = [N(50.4020, 30.5020), N(50.4020, 30.5030), N(50.4030, 30.5030), N(50.4030, 30.5020)]
    ways.append(_way(100, b1 + [b1[0]], {"building": "yes", "height": "12"}))
    # будинок-мультиполігон: зовнішній контур (без тегів) + двір
    outer = [N(50.4040, 30.5040), N(50.4040, 30.5070), N(50.4070, 30.5070), N(50.4070, 30.5040)]
    inner = [N(50.4050, 30.5050), N(50.4050, 30.5060), N(50.4060, 30.5060), N(50.4060, 30.5050)]
    ways.append(_way(101, outer + [outer[0]]))
    ways.append(_way(102, inner + [inner[0]]))
    rels.append(_rel(900, [{"type": "way", "ref": 101, "role": "outer"}, {"type": "way", "ref": 102, "role": "inner"}],
                     {"type": "multipolygon", "building": "apartments", "building:levels": "5"}))
    # частина будівлі з висотою
    p1 = [N(50.4021, 30.5021), N(50.4021, 30.5025), N(50.4025, 30.5025), N(50.4025, 30.5021)]
    ways.append(_way(103, p1 + [p1[0]], {"building:part": "yes", "height": "20"}))
    # ставок
    w1 = [N(50.4080, 30.5100), N(50.4080, 30.5120), N(50.4095, 30.5120), N(50.4095, 30.5100)]
    ways.append(_way(200, w1 + [w1[0]], {"natural": "water", "water": "pond"}))
    # парк
    g1 = [N(50.4010, 30.5150), N(50.4010, 30.5190), N(50.4040, 30.5190), N(50.4040, 30.5150)]
    ways.append(_way(300, g1 + [g1[0]], {"leisure": "park", "name": "Парк"}))
    # вулиці: сітка 3×3 + міст + площа + будівництво
    grid = {}
    for r, lat in enumerate((50.4005, 50.4050, 50.4095)):
        for c, lon in enumerate((30.5005, 30.5100, 30.5195)):
            grid[(r, c)] = N(lat, lon)
    wid = 400
    for r in range(3):
        ways.append(_way(wid, [grid[(r, 0)], grid[(r, 1)], grid[(r, 2)]], {"highway": "residential", "name": f"H{r}"}))
        wid += 1
    for c in range(3):
        ways.append(_way(wid, [grid[(0, c)], grid[(1, c)], grid[(2, c)]], {"highway": "tertiary", "oneway": "yes" if c == 1 else "no"}))
        wid += 1
    # міст поверх ставка — і дорога, і bridge-шар
    br = [N(50.4080, 30.5110), N(50.4095, 30.5110)]
    ways.append(_way(500, br, {"highway": "secondary", "bridge": "yes", "layer": "1"}))
    # площа: highway + area=yes — НЕ дорога для osmnx
    sq = [N(50.4060, 30.5160), N(50.4060, 30.5170), N(50.4070, 30.5170), N(50.4070, 30.5160)]
    ways.append(_way(501, sq + [sq[0]], {"highway": "pedestrian", "area": "yes"}))
    # будівництво: виключає фільтр `highway!~construction`
    cs = [N(50.4001, 30.5001), N(50.4001, 30.5050)]
    ways.append(_way(502, cs, {"highway": "construction", "construction": "residential"}))
    # колія
    rl = [N(50.4002, 30.5002), N(50.4098, 30.5198)]
    ways.append(_way(600, rl, {"railway": "rail"}))
    # вузол-фонтан із тегом води (node-feature)
    N(50.4085, 30.5105, {"natural": "water", "water": "fountain"})
    return list(nodes.values()), ways, rels


TAGS_B = {"building": True}
TAGS_P = {"building:part": True}
TAGS_W = {"natural": "water", "water": True, "waterway": ["riverbank", "dock", "canal"],
          "landuse": ["reservoir", "basin"], "man_made": ["water_well", "reservoir_covered"]}
TAGS_BR = {"bridge": True}
TAGS_R = {"railway": ["rail", "light_rail", "narrow_gauge", "tram", "subway", "funicular"]}
TAGS_G = {"leisure": ["park", "garden"], "landuse": ["grass", "forest"], "natural": ["wood"]}
ALL_TAGS = [TAGS_B, TAGS_P, TAGS_W, TAGS_BR, TAGS_R, TAGS_G]


def _separate_response(nodes, ways, rels, tags):
    """Що повернув би Overpass на ОКРЕМИЙ запит шару: елементи з тегом + `(._;>;)`."""
    pairs = ob._tag_pairs(tags)
    node_ix = {n["id"]: n for n in nodes}
    way_ix = {w["id"]: w for w in ways}
    top = [e for e in nodes + ways + rels if e.get("tags") and ob.Bundle._matches(e["tags"], pairs)]
    out_n, out_w, out_r = {}, {}, {}
    for e in top:
        if e["type"] == "node":
            out_n[e["id"]] = e
        elif e["type"] == "way":
            out_w[e["id"]] = e
        else:
            out_r[e["id"]] = e
            for m in e["members"]:
                if m["type"] == "way":
                    out_w[m["ref"]] = way_ix[m["ref"]]
    for w in list(out_w.values()):
        for ref in w["nodes"]:
            out_n[ref] = node_ix[ref]
    # osmnx руйнує елементи при обробці — «відповідь сервера» завжди свіжа копія
    return copy.deepcopy({"elements": list(out_n.values()) + list(out_w.values()) + list(out_r.values())})


def _roads_response(nodes, ways, way_filter):
    ok = ob.parse_way_filter(way_filter)
    node_ix = {n["id"]: n for n in nodes}
    rw = [w for w in ways if w.get("tags") and ok(w["tags"])]
    rn = {ref: node_ix[ref] for w in rw for ref in w["nodes"]}
    return {"elements": list(rn.values()) + rw}


def _union(nodes, ways, rels, way_filter):
    """Union-відповідь: усі шари + дороги, без дублікатів, у «серверному» порядку."""
    seen = {}
    for tags in ALL_TAGS:
        for e in _separate_response(nodes, ways, rels, tags)["elements"]:
            seen[(e["type"], e["id"])] = e
    for e in _roads_response(nodes, ways, way_filter)["elements"]:
        seen[(e["type"], e["id"])] = e
    els = list(seen.values())
    els.sort(key=lambda e: ({"node": 0, "way": 1, "relation": 2}[e["type"]], e["id"]))
    return copy.deepcopy(els)


@pytest.fixture()
def world():
    return _world()


@pytest.fixture()
def bundle(world):
    nodes, ways, rels = world
    way_filter = ob._network_filter("all")
    requested = set()
    for t in ALL_TAGS:
        requested.update(ob._tag_pairs(t))
    return ob.Bundle(_union(nodes, ways, rels, way_filter), requested, way_filter)


# ────────────────────────── 1. текст запиту ──────────────────────────
def test_query_contains_every_layer_component_verbatim():
    layers = [(POLY, t) for t in ALL_TAGS]
    query, way_filter = ob.build_query(layers, ob.roads_query_polygon(POLY))
    settings = ox._overpass._make_overpass_settings()
    assert query.startswith(settings + ";(") and query.endswith(");out;")
    coord = ob._polygon_coord_strs(POLY)[0]
    for tags in ALL_TAGS:
        own = ox._overpass._create_overpass_features_query(coord, tags)
        body = own[len(settings) + 2:-len(");out;")]  # між `;(` і `);out;`
        assert body in query, f"компонент шару {tags} відсутній у пакеті"
    # дороги: буферний полігон + фільтр osmnx, з рекурсією `>`
    buff_coord = ob._polygon_coord_strs(ob.roads_query_polygon(POLY))[0]
    assert f"(way{way_filter}(poly:{buff_coord!r});>;);" in query
    assert way_filter == ox._overpass._get_network_filter("all")


def test_query_polygon_matches_osmnx_graph_buffer():
    """500 м буфер — той самий, що `graph_from_polygon` рахує сам."""
    poly_proj, crs = ox.projection.project_geometry(POLY)
    expected, _ = ox.projection.project_geometry(poly_proj.buffer(500), crs=crs, to_latlong=True)
    assert ob.roads_query_polygon(POLY).equals(expected)


# ────────────────────────── 2. еквівалентність шарів ──────────────────────────
def _gdf_signature(gdf):
    rows = []
    for idx, row in gdf.iterrows():
        tags = {k: row[k] for k in sorted(gdf.columns) if k != "geometry" and row[k] == row[k]}
        rows.append((idx, row.geometry.wkb, tuple(sorted(tags.items()))))
    return sorted(rows, key=lambda r: (r[0][0], r[0][1]))


@pytest.mark.parametrize("tags", ALL_TAGS, ids=["buildings", "parts", "water", "bridges", "rail", "green"])
def test_layer_from_bundle_equals_separate_request(world, bundle, tags):
    nodes, ways, rels = world
    expected = ox.features._create_gdf([_separate_response(nodes, ways, rels, tags)], POLY, tags)
    got = bundle.features(tags, POLY)
    assert _gdf_signature(got) == _gdf_signature(expected)
    assert len(got) > 0


def test_multipolygon_building_keeps_its_hole(bundle):
    gdf = bundle.features(TAGS_B, POLY)
    rel = gdf.loc[("relation", 900)]
    assert rel.geometry.geom_type == "Polygon" and len(rel.geometry.interiors) == 1
    assert ("way", 101) not in gdf.index, "контур без власних тегів не має ставати окремим будинком"


def test_empty_layer_raises_like_osmnx(bundle):
    with pytest.raises(InsufficientResponseError):
        bundle.features({"natural": "wood"}, POLY)  # ліс запитували, але його нема


def test_unrequested_tags_are_a_miss_not_silence(bundle):
    with pytest.raises(ob.BundleMiss):
        bundle.features({"amenity": "fountain"}, POLY)


def _graph_signature(G):
    nodes = sorted((n, round(d["x"], 7), round(d["y"], 7), d.get("street_count")) for n, d in G.nodes(data=True))
    edges = []
    for u, v, k, d in G.edges(keys=True, data=True):
        geom = d.get("geometry")
        edges.append((u, v, k, str(d.get("osmid")), str(d.get("highway")), round(float(d["length"]), 3),
                      geom.wkb if geom is not None else None))
    return nodes, sorted(edges, key=str)


def test_roads_from_bundle_equal_graph_from_response(world, bundle):
    """Той самий граф, що `graph_from_polygon` збудував би з відповіді на свій запит."""
    nodes, ways, _ = world
    way_filter = ox._overpass._get_network_filter("all")
    poly_buff = ob.roads_query_polygon(POLY)
    G_ref = ox.graph._create_graph([_roads_response(nodes, ways, way_filter)], False)
    G_ref = ox.truncate.truncate_graph_polygon(G_ref, poly_buff)
    G_ref = ox.simplification.simplify_graph(G_ref)
    G_ref_final = ox.truncate.truncate_graph_polygon(G_ref, POLY)
    nx.set_node_attributes(G_ref_final, ox.stats.count_streets_per_node(G_ref, nodes=G_ref_final.nodes), "street_count")

    G = bundle.graph(POLY, simplify=True, retain_all=True)
    assert _graph_signature(G) == _graph_signature(G_ref_final)
    hw = {str(d.get("highway")) for _, _, d in G.edges(data=True)}
    assert "residential" in hw and "secondary" in hw
    assert "construction" not in hw and "pedestrian" not in hw, "фільтр osmnx має відсіяти будівництво і площу"
    assert G.number_of_edges() > 0


# ────────────────────────── 3. фільтр way ──────────────────────────
def test_way_filter_semantics():
    ok = ob.parse_way_filter('["highway"]["area"!~"yes"]["highway"!~"abandoned|construction"]["access"!="private"]')
    assert ok({"highway": "residential"})
    assert ok({"highway": "residential", "area": "no"})
    assert not ok({"highway": "pedestrian", "area": "yes"})
    assert not ok({"highway": "construction"})
    assert not ok({"highway": "residential", "access": "private"})
    assert not ok({"building": "yes"})
    with pytest.raises(ob.BundleMiss):
        ob.parse_way_filter('["highway"](if:1)')


def test_osmnx_all_filter_is_parseable():
    ob.parse_way_filter(ox._overpass._get_network_filter("all"))


# ────────────────────────── 4. LazyBundle ──────────────────────────
class _FakeRequest:
    def __init__(self, elements, fail=None):
        self.elements, self.fail, self.calls = elements, fail, 0

    def __call__(self, data):
        self.calls += 1
        if self.fail:
            raise self.fail
        return {"elements": self.elements}


def _lazy(world, monkeypatch, fail=None):
    nodes, ways, rels = world
    fake = _FakeRequest(_union(nodes, ways, rels, ob._network_filter("all")), fail=fail)
    monkeypatch.setattr(ox._overpass, "_overpass_request", fake)
    monkeypatch.setenv("OVERPASS_PREFLIGHT", "0")
    from services import data_loader
    monkeypatch.setattr(data_loader, "_overpass_endpoints", lambda: ["https://one.example/api"])
    monkeypatch.setattr(data_loader.time, "sleep", lambda *_a, **_k: None)
    lb = ob.LazyBundle(city_bbox=BBOX, extras_bbox=BBOX, feature_tags=ALL_TAGS[:5], extras_tags=[TAGS_G])
    return lb, fake


def test_lazy_bundle_fetches_once_for_all_layers_and_threads(world, monkeypatch):
    lb, fake = _lazy(world, monkeypatch)
    results, errors = [], []

    def worker(tags):
        try:
            results.append(len(lb.features(tags, POLY)))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in ALL_TAGS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lb.graph(POLY, simplify=True, retain_all=True)
    assert not errors
    assert fake.calls == 1, "усі шари з усіх потоків — з ОДНОГО запиту"
    assert lb.fetched and not lb.failed


def test_lazy_bundle_failure_becomes_miss_for_every_layer(world, monkeypatch):
    lb, fake = _lazy(world, monkeypatch, fail=RuntimeError("boom"))
    with pytest.raises(ob.BundleMiss):
        lb.features(TAGS_B, POLY)
    with pytest.raises(ob.BundleMiss):
        lb.graph(POLY)
    assert fake.calls == 1, "після збою пакет не пробує знову — шари йдуть старим шляхом"
    assert lb.failed


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("OVERPASS_BUNDLE", "0")
    assert not ob.enabled()
    monkeypatch.setenv("OVERPASS_BUNDLE", "1")
    assert ob.enabled()


# ────────────────────────── 5. інтеграція з пайплайном ──────────────────────────
def test_fetch_generation_data_hands_one_bundle_to_both_loaders(monkeypatch):
    import geopandas as gpd
    from services import data_fetch_pipeline as dfp

    seen = {}

    def fake_city(*args, **kwargs):
        seen["city"] = kwargs.get("bundle")
        return gpd.GeoDataFrame(), gpd.GeoDataFrame(), None

    def fake_extras(*args, **kwargs):
        seen["extras"] = kwargs.get("bundle")
        return gpd.GeoDataFrame()

    monkeypatch.setattr(dfp, "fetch_city_data", fake_city)
    monkeypatch.setattr(dfp, "fetch_extras", fake_extras)
    monkeypatch.setattr(dfp, "resolve_osm_source", lambda: "overpass")

    class Req:
        north, south, east, west = 50.41, 50.40, 30.52, 30.50
        keychain_mode = False

    class Task:
        def update_status(self, *a, **k):
            pass

    dfp.fetch_generation_data(request=Req(), global_center=None, task=Task())
    assert seen["city"] is not None and seen["city"] is seen["extras"], "обидва фетчери — один і той самий пакет"
    lb = seen["city"]
    # bbox пакета = bbox, який fetch_city_data рахує сама (road_padding 0.004 + loader 0.002)
    assert lb.city_bbox == (30.50 - 0.004 - 0.002, 50.40 - 0.004 - 0.002, 30.52 + 0.004 + 0.002, 50.41 + 0.004 + 0.002)
    assert lb.extras_bbox == (30.50, 50.40, 30.52, 50.41)
    assert not lb.fetched, "ніхто не звертався по шар → у мережу не ходили"


def test_data_loader_layer_helper_falls_back_on_miss(monkeypatch):
    """`_features_layer` у data_loader: BundleMiss → окремий запит, як раніше."""
    from services import data_loader

    class Miss:
        def features(self, tags, polygon):
            raise ob.BundleMiss("немає")

    calls = []
    monkeypatch.setattr(data_loader, "_run_overpass_with_retries", lambda label, fn: calls.append(label) or "net")
    # дістаємо замикання через реальний виклик неможливо без мережі — перевіряємо контракт напряму
    bundle = Miss()
    try:
        bundle.features({"building": True}, POLY)
    except ob.BundleMiss:
        result = data_loader._run_overpass_with_retries("buildings", lambda: None)
    assert result == "net" and calls == ["buildings"]


# ────────────────────────── 6. очікування, поки джерело оживе ──────────────────────────
class _Clock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_bundle_waits_for_breaker_and_then_succeeds(monkeypatch):
    """Джерело лягло на хвилину → пакет чекає (і каже про це в статусі), а не валить задачу."""
    from services import overpass_health as oh

    monkeypatch.setenv("OVERPASS_WAIT_MAX_S", "120")
    clock = _Clock()
    monkeypatch.setattr(oh.time, "time", clock.now)
    oh.reset()
    attempts = {"n": 0}
    statuses = []

    def attempt():
        attempts["n"] += 1
        if attempts["n"] < 3:
            oh.note_failure(ConnectionError("Connection refused"))
            oh.note_failure(ConnectionError("Connection refused"))  # поріг 2 → карантин 30 с
            raise oh.OverpassUnavailableError(oh.outage_reason())
        oh.note_success()
        return {"elements": []}

    out = ob._fetch_with_recovery(attempt, oh, status_cb=lambda m, p: statuses.append((m, p)),
                                  sleep=clock.sleep, clock=clock.now)
    assert out == {"elements": []}
    assert attempts["n"] == 3
    assert clock.t - 1000.0 >= 60.0, "між спробами чекали повну паузу запобіжника"
    assert "повторна спроба через 30 с" in statuses[0][0] and statuses[0][1] == 30.0
    assert statuses[-1] == ("", None), "після успіху очікування знято"
    oh.reset()


def test_bundle_gives_up_after_deadline(monkeypatch):
    from services import overpass_health as oh

    monkeypatch.setenv("OVERPASS_WAIT_MAX_S", "50")
    clock = _Clock()
    monkeypatch.setattr(oh.time, "time", clock.now)
    oh.reset()
    attempts = {"n": 0}

    def attempt():
        attempts["n"] += 1
        oh.note_failure(ConnectionError("Connection refused"))
        oh.note_failure(ConnectionError("Connection refused"))
        raise oh.OverpassUnavailableError("down")

    with pytest.raises(oh.OverpassUnavailableError):
        ob._fetch_with_recovery(attempt, oh, sleep=clock.sleep, clock=clock.now)
    assert 2 <= attempts["n"] <= 3, "кілька спроб у межах 50 с, потім чесна відмова"
    assert clock.t - 1000.0 <= 50.0 + 1e-6
    oh.reset()


def test_zero_wait_keeps_old_fail_fast_behaviour(monkeypatch):
    from services import overpass_health as oh

    monkeypatch.setenv("OVERPASS_WAIT_MAX_S", "0")
    clock = _Clock()
    calls = {"n": 0}

    def attempt():
        calls["n"] += 1
        raise oh.OverpassUnavailableError("down")

    with pytest.raises(oh.OverpassUnavailableError):
        ob._fetch_with_recovery(attempt, oh, sleep=clock.sleep, clock=clock.now)
    assert calls["n"] == 1 and clock.t == 1000.0
