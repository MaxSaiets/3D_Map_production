"""Model-file retention: deletes old generated output so `output/` doesn't grow
unbounded forever. Files tied to a real order get a much longer grace period
(accounting / reprint requests) than plain never-downloaded generations.

Nothing in this module touches the generation/geometry pipeline — it only
walks `output_dir` and deletes stale files by mtime + name pattern.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

MODEL_RETENTION_DAYS = int(os.getenv("MODEL_RETENTION_DAYS", "90"))
RETENTION_KEEP_ORDERED_DAYS = int(os.getenv("RETENTION_KEEP_ORDERED_DAYS", "1095"))

# Filenames/dirs that must NEVER be touched by retention regardless of age —
# private data stores, config, golden baselines, VCS markers.
_PROTECTED_NAMES = {
    "users.json", "orders.jsonl", "orders.jsonl.1",
    "analytics.jsonl", "analytics.jsonl.1",
    "panel_batches.json", ".gitkeep",
}
_PROTECTED_PREFIXES = ("pricing", "golden")

# Model-output naming patterns we're allowed to expire:
#   model_<size...>_<8hex>_<8hex>.{3mf,glb,stl}
#   <uuid>_print_*.{json,ini,zip}   (+ the sibling <uuid>_print_layout_parts/ dir)
#   previews/<uuid-or-8hex-ish-id>.png   (share OG preview)
_RE_MODEL_FILE = re.compile(r"^model_.*\.(3mf|glb|stl)$", re.IGNORECASE)
_RE_PRINT_FILE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"_print_.*\.(json|ini|zip)$"
)
_RE_PRINT_LAYOUT_DIR = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"_print_layout_parts$"
)

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_HEX8_RE = re.compile(r"\b[0-9a-fA-F]{8}\b")


def _is_expirable_name(name: str) -> bool:
    """True when `name` (a file or directory basename directly under output_dir)
    matches a known generated-model pattern we are allowed to age out."""
    if name in _PROTECTED_NAMES:
        return False
    if any(name.lower().startswith(p) for p in _PROTECTED_PREFIXES):
        return False
    if _RE_MODEL_FILE.match(name):
        return True
    if _RE_PRINT_FILE.match(name):
        return True
    if _RE_PRINT_LAYOUT_DIR.match(name):
        return True
    return False


def _is_preview_png(rel_parts) -> bool:
    """True for output/previews/<id>.png entries (share OG images)."""
    return len(rel_parts) == 2 and rel_parts[0] == "previews" and rel_parts[1].lower().endswith(".png")


def _extract_ids_from_line(line: str) -> Set[str]:
    """Be generous: collect every uuid-like token, every bare 8-hex fragment, and
    every model_<...>_<8hex> tail found in the raw JSON line — regardless of
    which field it lives under (task_id, summary.task_id, summary.task_ids, or
    any other string field a caller might have used)."""
    ids: Set[str] = set()
    for m in _UUID_RE.findall(line):
        ids.add(m.lower())
        ids.add(m.replace("-", "").lower())
    for m in _HEX8_RE.findall(line):
        ids.add(m.lower())
    return ids


def _referenced_task_ids(data_dir: Path) -> Set[str]:
    """Every uuid/8-hex fragment referenced anywhere in data_dir/orders.jsonl."""
    orders_path = data_dir / "orders.jsonl"
    ids: Set[str] = set()
    if not orders_path.exists():
        return ids
    try:
        for line in orders_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            # Grab from well-known fields first (documented contract)...
            candidates: list = []
            tid = rec.get("task_id")
            if tid:
                candidates.append(tid)
            summary = rec.get("summary") or {}
            if isinstance(summary, dict):
                if summary.get("task_id"):
                    candidates.append(summary.get("task_id"))
                tids = summary.get("task_ids")
                if isinstance(tids, (list, tuple)):
                    candidates.extend(tids)
            for c in candidates:
                if isinstance(c, str) and c:
                    ids.add(c.lower())
                    ids.add(c.replace("-", "").lower())
            # ...then be generous and scan the whole raw line for uuid/hex tokens,
            # in case the id lives under some other field we didn't anticipate.
            ids |= _extract_ids_from_line(line)
    except Exception:
        pass
    return ids


def _name_matches_any_id(name: str, ids: Set[str]) -> bool:
    if not ids:
        return False
    lname = name.lower()
    for i in ids:
        if i and i in lname:
            return True
    return False


def _entry_size(path: Path) -> int:
    try:
        if path.is_dir():
            total = 0
            for root, _dirs, files in os.walk(path):
                for f in files:
                    try:
                        total += (Path(root) / f).stat().st_size
                    except OSError:
                        pass
            return total
        return path.stat().st_size
    except OSError:
        return 0


def _entry_mtime(path: Path) -> float:
    try:
        if path.is_dir():
            # Use the newest mtime among children (or the dir's own mtime if
            # empty) so an in-progress/just-touched directory isn't prematurely
            # reaped, and a freshly-mkdir'd wrapper doesn't mask old contents.
            newest = None
            for root, _dirs, files in os.walk(path):
                for f in files:
                    try:
                        m = (Path(root) / f).stat().st_mtime
                        newest = m if newest is None else max(newest, m)
                    except OSError:
                        pass
            return newest if newest is not None else path.stat().st_mtime
        return path.stat().st_mtime
    except OSError:
        return time.time()


def run_retention(
    output_dir: Path,
    data_dir: Path,
    now: Optional[float] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Delete expired generated files/dirs under `output_dir`.

    - Entries NOT referenced by any order get `MODEL_RETENTION_DAYS`.
    - Entries referenced by an order (task id found in data_dir/orders.jsonl)
      get `RETENTION_KEEP_ORDERED_DAYS` instead.
    - Protected filenames/prefixes and anything not matching a known model
      output pattern are never touched.

    Returns {"deleted": n, "freed_bytes": b, "kept_ordered": k, "errors": e}.
    """
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)
    now = time.time() if now is None else now

    result = {"deleted": 0, "freed_bytes": 0, "kept_ordered": 0, "errors": 0}

    if MODEL_RETENTION_DAYS <= 0:
        return result

    if not output_dir.exists():
        return result

    ordered_ids = _referenced_task_ids(data_dir)
    retention_secs = MODEL_RETENTION_DAYS * 86400.0
    ordered_retention_secs = RETENTION_KEEP_ORDERED_DAYS * 86400.0

    entries = []
    try:
        entries = list(output_dir.iterdir())
    except OSError:
        return result

    for entry in entries:
        name = entry.name
        try:
            rel_parts = entry.relative_to(output_dir).parts
        except ValueError:
            rel_parts = (name,)

        is_preview = False
        if entry.is_dir() and name == "previews":
            # Walk previews/*.png individually (each is its own share image).
            try:
                for child in entry.iterdir():
                    if not child.is_file() or child.suffix.lower() != ".png":
                        continue
                    _process_entry(
                        child, child.name, ordered_ids, retention_secs,
                        ordered_retention_secs, now, dry_run, result,
                    )
            except OSError:
                result["errors"] += 1
            continue

        if not _is_expirable_name(name):
            continue

        _process_entry(
            entry, name, ordered_ids, retention_secs, ordered_retention_secs,
            now, dry_run, result,
        )

    print(
        f"[RETENTION] deleted={result['deleted']} freed_bytes={result['freed_bytes']} "
        f"kept_ordered={result['kept_ordered']} errors={result['errors']} "
        f"dry_run={dry_run}"
    )
    return result


def _process_entry(
    entry: Path,
    name: str,
    ordered_ids: Set[str],
    retention_secs: float,
    ordered_retention_secs: float,
    now: float,
    dry_run: bool,
    result: Dict[str, Any],
) -> None:
    is_ordered = _name_matches_any_id(name, ordered_ids)
    limit = ordered_retention_secs if is_ordered else retention_secs
    try:
        age = now - _entry_mtime(entry)
    except Exception:
        result["errors"] += 1
        return
    if age <= limit:
        if is_ordered:
            result["kept_ordered"] += 1
        return
    size = _entry_size(entry)
    if dry_run:
        result["deleted"] += 1
        result["freed_bytes"] += size
        return
    try:
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
        result["deleted"] += 1
        result["freed_bytes"] += size
    except Exception:
        result["errors"] += 1
