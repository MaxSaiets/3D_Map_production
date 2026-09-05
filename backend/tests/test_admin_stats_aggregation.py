"""Unit tests for main._aggregate_analytics — pure aggregation over raw
analytics.jsonl lines, used by /api/admin/stats. Exercises the per-visitor
timeline and the guided-flow "choices" aggregates."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import main as app_main


def _line(event, path="", props=None, visitor="vis1", day=None, ts=None, cc="UA"):
    now = datetime.now(timezone.utc)
    day = day or now.strftime("%Y-%m-%d")
    ts = ts or now.isoformat(timespec="seconds")
    rec = {
        "ts": ts, "day": day, "event": event, "path": path,
        "locale": "uk", "ref": "", "visitor": visitor, "cc": cc,
    }
    if props is not None:
        rec["props"] = {str(k): str(v) for k, v in props.items()}
    return json.dumps(rec, ensure_ascii=False)


def test_basic_totals_and_pageviews():
    lines = [
        _line("pageview", path="/"),
        _line("pageview", path="/create"),
        _line("ping"),
    ]
    agg = app_main._aggregate_analytics(lines, 30)
    assert agg["totals"]["pageviews"] == 2
    # ping doesn't count toward events
    assert agg["totals"]["events"] == 2
    assert agg["totals"]["uniqueVisitors"] == 1


def test_timeline_excludes_ping_and_untargeted_clicks():
    lines = [
        _line("pageview", path="/create"),
        _line("ping"),
        _line("click", props={}),  # no el -> excluded
        _line("click", props={"el": "buy-button"}),  # has el -> included
        _line("guided_step", props={"step": "2", "product": "map"}),
    ]
    agg = app_main._aggregate_analytics(lines, 30)
    tl = agg["recentVisitors"][0]["timeline"]
    events = [item["e"] for item in tl]
    assert "ping" not in events
    assert events.count("click") == 1
    assert "guided_step" in events
    assert "pageview" in events
    click_item = next(i for i in tl if i["e"] == "click")
    assert click_item["p"] == {"el": "buy-button"}


def test_timeline_caps_clicks_at_8_per_visitor():
    lines = [_line("click", props={"el": f"btn{i}"}) for i in range(12)]
    agg = app_main._aggregate_analytics(lines, 30)
    tl = agg["recentVisitors"][0]["timeline"]
    assert sum(1 for i in tl if i["e"] == "click") == 8


def test_timeline_caps_total_at_40():
    lines = [_line("guided_step", props={"step": str(i)}) for i in range(55)]
    agg = app_main._aggregate_analytics(lines, 30)
    tl = agg["recentVisitors"][0]["timeline"]
    assert len(tl) == 40
    # keeps the LAST 40 (most recent), so step "54" survives, step "0" doesn't.
    steps = [i["p"].get("step") for i in tl]
    assert "54" in steps
    assert "0" not in steps


def test_timeline_prop_subset_only_keeps_known_keys():
    lines = [
        _line("guided_generate", props={
            "product": "map", "sizeMm": "150", "place": "kyiv",
            "placePicked": "True", "secretUnknownField": "xxx",
        }),
    ]
    agg = app_main._aggregate_analytics(lines, 30)
    item = agg["recentVisitors"][0]["timeline"][0]
    assert item["e"] == "guided_generate"
    assert "secretUnknownField" not in item["p"]
    assert item["p"]["sizeMm"] == "150"
    assert item["p"]["place"] == "kyiv"


def test_guided_choices_sizes_places():
    lines = [
        _line("guided_size", props={"sizeMm": "150"}),
        _line("guided_generate", props={"sizeMm": "150", "place": "Kyiv", "placePicked": "True"}),
        _line("guided_place", props={"place": "Lviv"}),
        _line("guided_place", props={"place": ""}),  # empty -> skipped
    ]
    agg = app_main._aggregate_analytics(lines, 30)
    choices = agg["guided"]["choices"]
    sizes = dict(choices["sizes"])
    places = dict(choices["places"])
    assert sizes.get("150") == 2
    assert places.get("Kyiv") == 1
    assert places.get("Lviv") == 1
    assert "" not in places


def test_guided_choices_home_share_download_order_results():
    lines = [
        _line("guided_home", props={"action": "mark"}),
        _line("guided_home", props={"action": "unmark"}),
        _line("guided_share", props={}),
        _line("guided_download", props={}),
        _line("download_model", props={}),
        _line("guided_order_click", props={}),
        _line("guided_result", props={"ok": "True"}),
        _line("guided_result", props={"ok": "false"}),
        _line("guided_result", props={}),
    ]
    agg = app_main._aggregate_analytics(lines, 30)
    choices = agg["guided"]["choices"]
    assert choices["homeMarked"] == 1
    assert choices["shares"] == 1
    assert choices["downloads"] == 2
    assert choices["orderClicks"] == 1
    assert choices["results"] == {"ok": 1, "fail": 2}


def test_guided_events_respect_days_cutoff():
    old_day = (datetime.now(timezone.utc) - timedelta(days=100)).strftime("%Y-%m-%d")
    lines = [
        _line("guided_size", props={"sizeMm": "999"}, day=old_day,
              ts=(datetime.now(timezone.utc) - timedelta(days=100)).isoformat(timespec="seconds")),
        _line("guided_size", props={"sizeMm": "150"}),
    ]
    agg = app_main._aggregate_analytics(lines, 30)
    sizes = dict(agg["guided"]["choices"]["sizes"])
    assert "999" not in sizes
    assert sizes.get("150") == 1


def test_multiple_visitors_get_separate_timelines():
    lines = [
        _line("pageview", path="/a", visitor="visA"),
        _line("pageview", path="/b", visitor="visB"),
    ]
    agg = app_main._aggregate_analytics(lines, 30)
    assert len(agg["recentVisitors"]) == 2
    ids = {v["id"] for v in agg["recentVisitors"]}
    assert len(ids) == 2
