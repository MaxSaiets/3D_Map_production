from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from services.firebase_service import FirebaseService


def publish_outputs_to_firebase(
    *,
    task: Any,
    output_file_abs: Path,
    primary_format: str,
) -> None:
    try:
        print("[INFO] Start uploading all files to Firebase...")

        primary_remote = f"3dMap/{output_file_abs.name}"
        primary_url = FirebaseService.upload_file(str(output_file_abs), remote_path=primary_remote)
        if primary_url:
            task.firebase_url = primary_url
            task.firebase_outputs[primary_format] = primary_url
            print(f"[INFO] Main Firebase Cloud link: {primary_url}")

        for fmt, local_path in task.output_files.items():
            if not local_path or not os.path.exists(local_path):
                continue
            if Path(local_path).resolve() == output_file_abs.resolve():
                continue

            remote_path = f"3dMap/{Path(local_path).name}"
            url = FirebaseService.upload_file(local_path, remote_path=remote_path)
            if url:
                task.firebase_outputs[fmt] = url
                print(f"[INFO] Part {fmt} uploaded to Firebase: {url}")

        if task.firebase_url:
            task.message = "Модель та шари готові та завантажені в Firebase!"

            try:
                keep_local = os.environ.get("KEEP_LOCAL_FILES", "false").lower() == "true"
                if not keep_local:
                    for file_path in [output_file_abs] + [Path(path) for path in task.output_files.values() if path]:
                        if file_path.exists():
                            file_path.unlink()
                    print("[INFO] Cleanup: Local temp files deleted.")
                else:
                    print("[INFO] Cleanup skipped (KEEP_LOCAL_FILES=true).")
            except Exception as cleanup_err:
                print(f"[WARN] Cleanup failed: {cleanup_err}")
        else:
            print("[INFO] Firebase upload skipped (not configured or failed).")
    except Exception as exc:
        print(f"[WARN] Firebase upload exception: {exc}")
