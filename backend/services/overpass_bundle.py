# -*- coding: utf-8 -*-
"""Один Overpass-запит на всю генерацію (зони поза ukraine.duckdb).

⭐Навіщо (заміряно на проді, реальні задачі 03.08–15.09.2026):
    закордонне превʼю — медіана 593 с, з них fetch_source 590.9 с (Дюссельдорф
    13.09: будівлі 187 с → мости 7 с → дороги 323 с → залізниця 61 с → …);
    12 із 30 закордонних спроб упали, бо джерело лягло ПОСЕРЕД серії запитів.

Причина — не геометрія (≈28 с), а СІМ послідовних запитів до overpass-api.de:
будівлі, частини будівель, вода, дороги, мости, залізниця, зелень. Кожен зі
своєю паузою rate-limit (osmnx питає /status і чекає вільний слот; слотів на
IP два, а ми шлемо три паралельно), своїм таймаутом 180 с і своїм шансом
попасти на відмову.

Ідея: усі шари — в ОДИН union-запит `( … );out;` — одна пауза, один таймаут,
один шанс упасти. Відповідь ділимо локально по шарах (той самий фільтр тегів,
що й на сервері) і віддаємо в ті самі функції osmnx, які збирають
GeoDataFrame/граф із JSON Overpass. Тож шари виходять ТІ САМІ, що й при семи
окремих запитах — це перевіряє `tests/test_overpass_bundle.py` на записаній
відповіді, а живе порівняння на реальних містах є в `scripts/bench_bundle.py`.

Безпека: коли щось не так (osmnx іншої версії без потрібних хелперів, набір
тегів, якого пакет не запитував, помилка мережі) — `BundleMiss`, і викликач
повертається на старий шлях по шарах. Вимикач: OVERPASS_BUNDLE=0.
"""
from __future__ import annotations

import os
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Iterable, Optional

import networkx as nx
import osmnx as ox
from osmnx._errors import InsufficientResponseError
from shapely.geometry import MultiPolygon, Polygon

__all__ = [
    "BundleMiss",
    "LazyBundle",
    "Bundle",
    "build_query",
    "enabled",
    "parse_way_filter",
    "wait_max_s",
]


class BundleMiss(RuntimeError):
    """Пакет не може віддати цей шар — беріть його окремим запитом, як раніше."""


def enabled() -> bool:
    return (os.getenv("OVERPASS_BUNDLE", "1") or "1").strip().lower() not in ("0", "false", "no")


# ──────────────────────────── теги → компоненти запиту ────────────────────────────
TagValue = "bool | str | list[str]"


def _tag_pairs(tags: dict) -> list[tuple[str, Any]]:
    """Той самий порядок і та сама нормалізація, що в osmnx
    `_create_overpass_features_query`: bool → сам ключ, str → [str], list → як є."""
    if not isinstance(tags, dict):
        raise TypeError("`tags` must be a dict with values of bool, str, or list of str.")
    pairs: list[tuple[str, Any]] = []
    for key, value in tags.items():
        if isinstance(value, bool):
            pairs.append((key, True))
        elif isinstance(value, str):
            pairs.append((key, value))
        elif isinstance(value, list):
            for item in value:
                if not isinstance(item, str):
                    raise TypeError("`tags` values must be bool, str, or list of str.")
                pairs.append((key, item))
        else:
            raise TypeError("`tags` values must be bool, str, or list of str.")
    return pairs


def _feature_components(polygon_coord_str: str, tags: dict) -> str:
    """Рядок-у-рядок те, що osmnx кладе у власний запит шару (без `out;`)."""
    parts: list[str] = []
    for key, value in _tag_pairs(tags):
        if value is True:
            tag_str = f"[{key!r}](poly:{polygon_coord_str!r});(._;>;);"
        else:
            tag_str = f"[{key!r}={value!r}](poly:{polygon_coord_str!r});(._;>;);"
        for kind in ("node", "way", "relation"):
            parts.append(f"({kind}{tag_str});")
    return "".join(parts)


def _polygon_coord_strs(polygon) -> list[str]:
    fn = getattr(ox._overpass, "_make_overpass_polygon_coord_strs", None)
    if fn is None:
        raise BundleMiss("osmnx без _make_overpass_polygon_coord_strs")
    return list(fn(polygon))


def _network_filter(network_type: str) -> str:
    fn = getattr(ox._overpass, "_get_network_filter", None)
    if fn is None:
        raise BundleMiss("osmnx без _get_network_filter")
    return str(fn(network_type))


def _overpass_settings() -> str:
    fn = getattr(ox._overpass, "_make_overpass_settings", None)
    if fn is None:
        raise BundleMiss("osmnx без _make_overpass_settings")
    return str(fn())


def roads_query_polygon(polygon):
    """Буфер 500 м, як у `graph_from_polygon` (Overpass віддає way цілком, якщо
    хоч один вузол у полігоні; буфер потрібен, щоб потім чесно обрізати)."""
    poly_proj, crs_utm = ox.projection.project_geometry(polygon)
    poly_buff, _ = ox.projection.project_geometry(poly_proj.buffer(500), crs=crs_utm, to_latlong=True)
    return poly_buff


def build_query(
    feature_layers: Iterable[tuple[Any, dict]],
    roads_polygon,
    network_type: str = "all",
) -> tuple[str, str]:
    """Збирає один union-запит. Повертає (query, way_filter доріг)."""
    components: list[str] = []
    for polygon, tags in feature_layers:
        for coord_str in _polygon_coord_strs(polygon):
            components.append(_feature_components(coord_str, tags))
    way_filter = _network_filter(network_type)
    if roads_polygon is not None:
        for coord_str in _polygon_coord_strs(roads_polygon):
            components.append(f"(way{way_filter}(poly:{coord_str!r});>;);")
    query = f"{_overpass_settings()};({''.join(components)});out;"
    return query, way_filter


# ──────────────────────────── локальний фільтр way доріг ────────────────────────────
_CLAUSE_RX = re.compile(r'\[\s*"([^"]+)"\s*(?:(!=|=|!~|~)\s*"([^"]*)")?\s*\]')


def parse_way_filter(filter_str: str):
    """`["highway"]["area"!~"yes"]["highway"!~"a|b"]` → предикат tags→bool.

    Семантика Overpass QL: `[k]` — ключ є; `[k=v]` — дорівнює; `[k!=v]` — ключа
    нема АБО інше значення; `[k~re]` — є і збігається; `[k!~re]` — ключа нема
    АБО не збігається. Невідому конструкцію не вгадуємо — BundleMiss.
    """
    clauses = []
    consumed = 0
    for m in _CLAUSE_RX.finditer(filter_str):
        if m.start() != consumed:
            raise BundleMiss(f"незрозумілий фільтр доріг: {filter_str!r}")
        consumed = m.end()
        key, op, val = m.group(1), m.group(2), m.group(3)
        rx = re.compile(val) if op in ("~", "!~") else None
        clauses.append((key, op, val, rx))
    if consumed != len(filter_str) or not clauses:
        raise BundleMiss(f"незрозумілий фільтр доріг: {filter_str!r}")

    def predicate(tags: dict) -> bool:
        for key, op, val, rx in clauses:
            present = key in tags
            if op is None:
                if not present:
                    return False
            elif op == "=":
                if not present or str(tags[key]) != val:
                    return False
            elif op == "!=":
                if present and str(tags[key]) == val:
                    return False
            elif op == "~":
                if not present or rx.search(str(tags[key])) is None:
                    return False
            elif op == "!~":
                if present and rx.search(str(tags[key])) is not None:
                    return False
        return True

    return predicate


# ──────────────────────────── пакет елементів ────────────────────────────
def _copy_element(el: dict) -> dict:
    """Неглибока копія елемента з власним dict тегів (списки nodes/members osmnx
    не змінює — лише забирає ключі з dict)."""
    out = dict(el)
    tags = el.get("tags")
    if tags is not None:
        out["tags"] = dict(tags)
    return out


class Bundle:
    """Розібрана відповідь одного union-запиту: віддає шари так само, як osmnx."""

    def __init__(self, elements: list[dict], requested_pairs: set[tuple[str, Any]], way_filter: str):
        self._nodes: dict[int, dict] = {}
        self._ways: dict[int, dict] = {}
        self._rels: dict[int, dict] = {}
        for el in elements:
            et = el.get("type")
            if et == "node":
                self._nodes[el["id"]] = el
            elif et == "way":
                self._ways[el["id"]] = el
            elif et == "relation":
                self._rels[el["id"]] = el
        self._requested = set(requested_pairs)
        self._road_ok = parse_way_filter(way_filter)
        self.element_count = len(elements)

    # -- допоміжне --
    @staticmethod
    def _matches(tags: dict, pairs: list[tuple[str, Any]]) -> bool:
        for key, value in pairs:
            if key in tags and (value is True or str(tags[key]) == value):
                return True
        return False

    def _closure(self, top: Iterable[dict]) -> list[dict]:
        """Те, що робить `(._;>;)`: члени relation (way+node) і вузли way."""
        picked_nodes: dict[int, dict] = {}
        picked_ways: dict[int, dict] = {}
        picked_rels: dict[int, dict] = {}
        for el in top:
            et = el["type"]
            if et == "node":
                picked_nodes[el["id"]] = el
            elif et == "way":
                picked_ways[el["id"]] = el
            elif et == "relation":
                picked_rels[el["id"]] = el
                for mem in el.get("members", ()):
                    if mem.get("type") == "way":
                        w = self._ways.get(mem.get("ref"))
                        if w is not None:
                            picked_ways[w["id"]] = w
                    elif mem.get("type") == "node":
                        n = self._nodes.get(mem.get("ref"))
                        if n is not None:
                            picked_nodes[n["id"]] = n
        for w in list(picked_ways.values()):
            for ref in w.get("nodes", ()):
                n = self._nodes.get(ref)
                if n is not None:
                    picked_nodes[n["id"]] = n
        return list(picked_nodes.values()) + list(picked_ways.values()) + list(picked_rels.values())

    # -- шари --
    def features(self, tags: dict, polygon):
        """Аналог `ox.features_from_polygon(polygon, tags)` з пакета.

        Кидає той самий `InsufficientResponseError`, що й osmnx, коли шар порожній.
        """
        pairs = _tag_pairs(tags)
        missing = [p for p in pairs if p not in self._requested]
        if missing:
            raise BundleMiss(f"пакет не запитував теги {missing}")
        create_gdf = getattr(ox.features, "_create_gdf", None)
        if create_gdf is None:
            raise BundleMiss("osmnx без features._create_gdf")
        top = [
            el
            for store in (self._nodes, self._ways, self._rels)
            for el in store.values()
            if el.get("tags") and self._matches(el["tags"], pairs)
        ]
        # osmnx `_process_features` РУЙНУЄ елементи (pop type/nodes/tags/members,
        # lon/lat) — кожен шар мусить отримати власні копії, інакше другий шар
        # побачить порожній пакет.
        subset = [_copy_element(el) for el in self._closure(top)]
        return create_gdf([{"elements": subset}], polygon, tags)

    def graph(self, polygon, *, simplify: bool = True, retain_all: bool = False, truncate_by_edge: bool = False):
        """Аналог `ox.graph_from_polygon(polygon, network_type='all', …)` з пакета.

        Кроки — дослівно з osmnx 2.x `graph_from_polygon`: граф із буферного
        полігона → обрізка до буфера → найбільша компонента (якщо треба) →
        спрощення → обрізка до полігона → street_count.
        """
        create_graph = getattr(ox.graph, "_create_graph", None)
        if create_graph is None:
            raise BundleMiss("osmnx без graph._create_graph")
        poly_buff = roads_query_polygon(polygon)
        ways = [w for w in self._ways.values() if w.get("tags") and self._road_ok(w["tags"])]
        subset = self._closure(ways)
        bidirectional = "all" in getattr(ox.settings, "bidirectional_network_types", [])
        G_buff = create_graph([{"elements": subset}], bidirectional)
        G_buff = ox.truncate.truncate_graph_polygon(G_buff, poly_buff, truncate_by_edge=truncate_by_edge)
        if not retain_all:
            G_buff = ox.truncate.largest_component(G_buff, strongly=False)
        if simplify:
            G_buff = ox.simplification.simplify_graph(G_buff)
        G = ox.truncate.truncate_graph_polygon(G_buff, polygon, truncate_by_edge=truncate_by_edge)
        if not retain_all:
            G = ox.truncate.largest_component(G, strongly=False)
        spn = ox.stats.count_streets_per_node(G_buff, nodes=G.nodes)
        nx.set_node_attributes(G, values=spn, name="street_count")
        return G

    def release(self) -> None:
        self._nodes.clear()
        self._ways.clear()
        self._rels.clear()


def _rss_mb() -> float:
    """RSS процесу (Linux); -1, де /proc нема. Для логу: скільки коштує великий пакет."""
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as fh:
            return int(fh.read().split()[1]) * 4096 / 1048576.0
    except Exception:  # noqa: BLE001
        return -1.0


# ──────────────────────────── очікування, поки джерело оживе ────────────────────────────
def wait_max_s() -> float:
    """Скільки максимум чекати на джерело, що лягло (0 = не чекати, як раніше).

    Заміряно 11.09.2026 на проді: обидва перебої overpass-api.de тривали кілька
    хвилин; людина з Відня за цей час натиснула «Створити» сім разів і пішла.
    Два-три повтори з паузою 30 с дешевші за втраченого покупця.
    """
    try:
        return max(0.0, float(os.getenv("OVERPASS_WAIT_MAX_S", "120")))
    except Exception:
        return 120.0


SOURCE_WAIT_MESSAGE = "Джерело карт (OpenStreetMap) тимчасово недоступне — чекаємо, повторна спроба через {s} с"


def _fetch_with_recovery(attempt, overpass_health, *, status_cb=None, sleep=time.sleep, clock=time.time):
    """Повторює `attempt`, поки джерело в карантині запобіжника і ще є час.

    Це ЄДИНЕ місце, де ми чекаємо: пакет — один запит на генерацію. Старий
    шлях по шарах лишається fail-fast (решта шарів тієї ж генерації не мають
    чекати намарно) — його поведінку перевіряють тести запобіжника.

    `status_cb(message, pause_s)`: pause_s — за скільки секунд наступна спроба
    (фронт показує це людині), None — очікування закінчилось.
    """
    deadline = clock() + wait_max_s()
    waited = False
    while True:
        try:
            result = attempt()
            if waited and status_cb is not None:
                try:
                    status_cb("", None)
                except Exception:  # noqa: BLE001
                    pass
            return result
        except overpass_health.OverpassUnavailableError:
            remaining = deadline - clock()
            if remaining <= 0:
                if waited and status_cb is not None:
                    try:
                        status_cb("", None)
                    except Exception:  # noqa: BLE001
                        pass
                raise
            pause = overpass_health.seconds_until_retry() or 5.0
            pause = min(pause, remaining)
            msg = SOURCE_WAIT_MESSAGE.format(s=int(round(pause)))
            print(f"[BUNDLE] {msg} (лишилось {int(remaining)} с очікування)", flush=True)
            waited = True
            if status_cb is not None:
                try:
                    status_cb(msg, pause)
                except Exception:  # noqa: BLE001 — статус лише інформує
                    pass
            sleep(pause)


# ──────────────────────────── лінивий пакет на одну генерацію ────────────────────────────
class LazyBundle:
    """Один пакет на генерацію: качається ПРИ ПЕРШОМУ зверненні будь-якого шару.

    Чому ліниво: `fetch_city_data` спершу дивиться parquet-кеш і DuckDB — якщо
    все є локально, у мережу йти не треба взагалі. А ще `fetch_extras` (зелень)
    бігає паралельним потоком — обидва тягнуть із того самого пакета, перший
    качає, другий чекає на замку.
    """

    def __init__(
        self,
        *,
        city_bbox: tuple[float, float, float, float],
        extras_bbox: Optional[tuple[float, float, float, float]],
        feature_tags: Iterable[dict],
        extras_tags: Iterable[dict] = (),
        network_type: str = "all",
        label: str = "",
        status_cb=None,
    ) -> None:
        # bbox у порядку osmnx 2.x: (left, bottom, right, top) = (west, south, east, north)
        self.city_bbox = tuple(float(v) for v in city_bbox)
        self.extras_bbox = tuple(float(v) for v in extras_bbox) if extras_bbox else None
        self.feature_tags = [dict(t) for t in feature_tags]
        self.extras_tags = [dict(t) for t in extras_tags]
        self.network_type = network_type
        self.label = label
        # status_cb(message, pause_s) — щоб людина бачила, що ми чекаємо на
        # джерело, а не «зависли» (див. _fetch_with_recovery).
        self.status_cb = status_cb
        self._lock = threading.Lock()
        self._bundle: Optional[Bundle] = None
        self._error: Optional[BaseException] = None
        self.fetch_seconds: Optional[float] = None

    # -- один запит --
    def _fetch(self) -> Bundle:
        city_poly = ox.utils_geo.bbox_to_poly(self.city_bbox)
        layers: list[tuple[Any, dict]] = [(city_poly, t) for t in self.feature_tags]
        if self.extras_bbox is not None and self.extras_tags:
            extras_poly = ox.utils_geo.bbox_to_poly(self.extras_bbox)
            layers += [(extras_poly, t) for t in self.extras_tags]
        query, way_filter = build_query(layers, roads_query_polygon(city_poly), self.network_type)
        requested: set[tuple[str, Any]] = set()
        for _, tags in layers:
            requested.update(_tag_pairs(tags))

        request_fn = getattr(ox._overpass, "_overpass_request", None)
        if request_fn is None:
            raise BundleMiss("osmnx без _overpass_request")

        def _once():
            ox.settings.use_cache = False
            return request_fn(OrderedDict(data=query))

        # Той самий обгортач, що й у шарів: список дзеркал, preflight, запобіжник.
        from services.data_loader import _run_overpass_with_retries
        from services import overpass_health

        t0 = time.time()
        rss0 = _rss_mb()
        response = _fetch_with_recovery(
            lambda: _run_overpass_with_retries("bundle", _once),
            overpass_health,
            status_cb=self.status_cb,
        )
        self.fetch_seconds = time.time() - t0
        elements = list(response.get("elements", [])) if isinstance(response, dict) else []
        rss1 = _rss_mb()
        print(
            f"[BUNDLE]{(' ' + self.label) if self.label else ''} один Overpass-запит: "
            f"{len(elements)} елементів за {self.fetch_seconds:.1f} с "
            f"(шарів: {len(layers)} + дороги)"
            + (f"; rss {rss0:.0f} → {rss1:.0f} МБ" if rss0 >= 0 and rss1 >= 0 else ""),
            flush=True,
        )
        return Bundle(elements, requested, way_filter)

    def _ensure(self) -> Bundle:
        if self._bundle is not None:
            return self._bundle
        if self._error is not None:
            raise BundleMiss(f"пакет уже впав: {self._error}") from self._error
        with self._lock:
            if self._bundle is not None:
                return self._bundle
            if self._error is not None:
                raise BundleMiss(f"пакет уже впав: {self._error}") from self._error
            try:
                self._bundle = self._fetch()
            except BundleMiss:
                raise
            except BaseException as exc:  # noqa: BLE001 — будь-який збій = старий шлях
                self._error = exc
                print(f"[BUNDLE] не вдалося ({type(exc).__name__}: {str(exc)[:160]}) → шари окремими запитами", flush=True)
                raise BundleMiss(str(exc)[:200]) from exc
            return self._bundle

    @property
    def fetched(self) -> bool:
        return self._bundle is not None

    @property
    def failed(self) -> bool:
        return self._error is not None

    def features(self, tags: dict, polygon):
        return self._ensure().features(tags, polygon)

    def graph(self, polygon, **kwargs):
        return self._ensure().graph(polygon, **kwargs)

    def release(self) -> None:
        b = self._bundle
        if b is not None:
            b.release()
