from __future__ import annotations

import os
from pathlib import Path


def candidate_pbf_paths() -> list[Path]:
    env_path = os.getenv("OSM_PBF_PATH")
    paths: list[Path] = []
    if env_path:
        paths.append(Path(env_path))

    backend_root = Path(__file__).resolve().parents[1]
    repo_root = backend_root.parent
    paths.extend(
        [
            backend_root / "cache" / "osm" / "ukraine-latest.osm.pbf",
            repo_root / "cache" / "osm" / "ukraine-latest.osm.pbf",
            backend_root / "cache" / "osm" / "geofabrik.osm.pbf",
            repo_root / "cache" / "osm" / "geofabrik.osm.pbf",
        ]
    )
    return paths


def resolve_osm_source() -> str:
    explicit = (os.getenv("OSM_SOURCE") or "").strip().lower()
    if explicit:
        return explicit

    force_api = (os.getenv("OSM_FORCE_API") or "1").strip().lower()
    if force_api in ("1", "true", "yes"):
        return "overpass"

    for candidate in candidate_pbf_paths():
        try:
            if candidate.exists():
                return "pbf"
        except Exception:
            continue

    return "overpass"


def resolve_pbf_path() -> Path | None:
    for candidate in candidate_pbf_paths():
        try:
            if candidate.exists():
                return candidate
        except Exception:
            continue
    return None
