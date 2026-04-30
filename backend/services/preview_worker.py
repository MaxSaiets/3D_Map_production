from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

from services.site_preview import PREVIEW_JOBS_DIR, build_preview_cache_from_worker_payload


class PreviewWorkerTimeout(TimeoutError):
    pass


def _install_timeout_guard() -> None:
    timeout_seconds = int(os.getenv("PREVIEW_WORKER_TIMEOUT_SECONDS", "1800"))
    if timeout_seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return

    def _handle_timeout(_signum, _frame):
        raise PreviewWorkerTimeout(f"Preview worker exceeded {timeout_seconds}s")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(timeout_seconds)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m services.preview_worker <input.json>")
        return 2

    input_file = Path(sys.argv[1]).resolve()
    payload = json.loads(input_file.read_text(encoding="utf-8"))
    preview_id = str(payload["preview_id"])
    status_file = PREVIEW_JOBS_DIR / f"{preview_id}.status.json"
    started = time.time()
    status_file.write_text(
        json.dumps({"status": "running", "started_at": started, "preview_id": preview_id, "pid": os.getpid()}, ensure_ascii=False),
        encoding="utf-8",
    )
    try:
        _install_timeout_guard()
        result = build_preview_cache_from_worker_payload(payload)
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        status_file.write_text(
            json.dumps(
                {
                    "status": "ready",
                    "preview_id": preview_id,
                    "started_at": started,
                    "finished_at": time.time(),
                    "cache_file": payload.get("cache_file"),
                    "features": {
                        key: len(result.get("layers", {}).get(key, {}).get("features", []))
                        for key in ("roads", "buildings", "water", "parks")
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return 0
    except BaseException as exc:
        status_file.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "preview_id": preview_id,
                    "started_at": started,
                    "finished_at": time.time(),
                    "error": str(exc),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
