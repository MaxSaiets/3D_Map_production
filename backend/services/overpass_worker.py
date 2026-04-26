from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from osmnx._errors import InsufficientResponseError

from services.data_loader import (
    _fetch_buildings_overpass,
    _fetch_roads_overpass,
    _fetch_water_overpass,
)
from services.extras_loader import _fetch_green_overpass
from services.overpass_client import configure_overpass_runtime


_QUERY_TARGETS = {
    "buildings": _fetch_buildings_overpass,
    "water": _fetch_water_overpass,
    "roads": _fetch_roads_overpass,
    "green": _fetch_green_overpass,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--timeout-s", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    status = {"status": "error", "error": "worker_failed"}
    status_path = Path(args.status)
    try:
        query_target = _QUERY_TARGETS[str(args.layer)]
    except KeyError:
        status = {"status": "error", "error": f"unknown_layer:{args.layer}"}
        status_path.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
        return 2

    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    try:
        configure_overpass_runtime(endpoint=args.endpoint, timeout_s=int(args.timeout_s))
        result = query_target(**payload)
        with open(args.result, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        status = {"status": "ok"}
    except InsufficientResponseError as exc:
        status = {"status": "empty", "error": str(exc)}
    except Exception as exc:
        status = {"status": "error", "error": str(exc)}
    finally:
        status_path.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
