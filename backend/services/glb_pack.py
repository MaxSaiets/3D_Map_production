"""Post-export GLB size optimization via gltfpack (meshopt compression).

Pure post-processing on already-exported preview GLB files. Never touches
mesh geometry pipelines. Runs `gltfpack -cc -kn -km -ke` on a finished .glb
and atomically swaps it in only if the result is a valid, smaller,
meshopt-compressed GLB. On any failure the original file is left untouched.

Env vars:
    GLTFPACK_BIN     - explicit path to the gltfpack executable (optional).
    PREVIEW_MESHOPT  - "1" (default) to enable packing, "0" to disable.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

__all__ = ["pack_glb_inplace", "resolve_gltfpack_bin", "is_meshopt_enabled"]

_GLB_MAGIC = b"glTF"


def is_meshopt_enabled() -> bool:
    return os.environ.get("PREVIEW_MESHOPT", "1") != "0"


def resolve_gltfpack_bin() -> Optional[str]:
    """Find the gltfpack executable, or None if not available.

    Resolution order:
      1. GLTFPACK_BIN env var (explicit path)
      2. <repo>/frontend/node_modules/.bin/gltfpack (and .cmd on Windows)
      3. shutil.which("gltfpack") (global install / PATH)
    """
    env_bin = os.environ.get("GLTFPACK_BIN")
    if env_bin:
        p = Path(env_bin)
        if p.exists():
            return str(p)
        # Might just be a bare command name resolvable via PATH.
        found = shutil.which(env_bin)
        if found:
            return found

    try:
        # backend/services/glb_pack.py -> parents[0]=services, [1]=backend, [2]=repo root
        repo_root = Path(__file__).resolve().parents[2]
        bin_dir = repo_root / "frontend" / "node_modules" / ".bin"
        if os.name == "nt":
            # On Windows the extensionless shim is a shebang script that
            # cannot be exec'd directly; prefer the .cmd/.exe wrapper.
            candidates = [bin_dir / "gltfpack.cmd", bin_dir / "gltfpack.exe", bin_dir / "gltfpack"]
        else:
            candidates = [bin_dir / "gltfpack"]
        for c in candidates:
            if c.exists():
                return str(c)
    except Exception:
        pass

    found = shutil.which("gltfpack")
    if found:
        return found

    return None


def _read_chunk_header(data: bytes, offset: int):
    if offset + 8 > len(data):
        return None
    length, ctype = struct.unpack_from("<I4s", data, offset)
    return length, ctype


def _validate_glb_meshopt(data: bytes) -> tuple[bool, Optional[str]]:
    """Validate GLB magic + parse JSON chunk + confirm EXT_meshopt_compression."""
    if len(data) < 12 or data[0:4] != _GLB_MAGIC:
        return False, "not a valid GLB (bad magic)"
    version, total_length = struct.unpack_from("<II", data, 4)
    offset = 12
    hdr = _read_chunk_header(data, offset)
    if hdr is None:
        return False, "GLB missing JSON chunk"
    json_len, json_type = hdr
    if json_type != b"JSON":
        return False, "GLB first chunk is not JSON"
    json_start = offset + 8
    json_bytes = data[json_start:json_start + json_len]
    try:
        doc = json.loads(json_bytes.decode("utf-8"))
    except Exception as exc:
        return False, f"GLB JSON chunk did not parse: {exc}"

    required = doc.get("extensionsRequired") or []
    if "EXT_meshopt_compression" not in required:
        return False, "EXT_meshopt_compression not in extensionsRequired"

    return True, None


def pack_glb_inplace(path: Union[str, Path], timeout_s: float = 40.0) -> dict:
    """Run gltfpack on `path` and atomically replace it if the result is smaller.

    Never raises. On any failure the original file is left untouched and
    the returned dict carries an "error" message.
    """
    t0 = time.monotonic()
    path = Path(path)
    result = {"ok": False, "before": 0, "after": 0, "ms": 0, "error": None}

    try:
        before = path.stat().st_size
    except OSError as exc:
        result["error"] = f"input file not accessible: {exc}"
        print(f"[GLB_PACK] skip: {result['error']}")
        return result
    result["before"] = before

    if not is_meshopt_enabled():
        result["error"] = "disabled via PREVIEW_MESHOPT=0"
        result["ms"] = int((time.monotonic() - t0) * 1000)
        print(f"[GLB_PACK] skip: {result['error']} path={path}")
        return result

    gltfpack_bin = resolve_gltfpack_bin()
    if not gltfpack_bin:
        result["error"] = "gltfpack not found"
        result["ms"] = int((time.monotonic() - t0) * 1000)
        print(f"[GLB_PACK] skip: {result['error']} path={path}")
        return result

    tmp_fd = None
    tmp_path = None
    try:
        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".glb", prefix="glbpack_", dir=str(path.parent))
        os.close(tmp_fd)
        tmp_fd = None
        tmp_path = Path(tmp_name)

        cmd = [gltfpack_bin, "-i", str(path), "-o", str(tmp_path), "-cc", "-kn", "-km", "-ke"]
        # ІНЦИДЕНТ 06.09.2026 (×3): wasm-gltfpack у node на ~3 МБ GLB (Львів M) роздував
        # памʼять, 6.4 ГБ VM ішла у своп, mem_guard вбивав бекенд, а сирота-node далі
        # душив VM → тунель мовчав ~25 хв. Два запобіжники: (1) не пакуємо файли більші за
        # GLB_PACK_MAX_MB (дефолт 2.0 — Одеса 1 МБ проходила, Львів 3 МБ валив);
        # (2) на Linux — RLIMIT_AS для дочірнього процесу (GLB_PACK_MEM_MB, дефолт 1200):
        # при перевищенні gltfpack падає сам, оригінал лишається (Caddy віддасть gzip).
        try:
            _max_mb = float(os.environ.get("GLB_PACK_MAX_MB", "2.0") or 2.0)
        except ValueError:
            _max_mb = 2.0
        if _max_mb > 0 and before > _max_mb * 1048576:
            result["error"] = f"skipped: {before/1048576:.2f} MB > GLB_PACK_MAX_MB={_max_mb}"
            print(f"[GLB_PACK] {result['error']} path={path}")
            return result
        _preexec = None
        if sys.platform.startswith("linux"):
            try:
                _mem_mb = int(os.environ.get("GLB_PACK_MEM_MB", "1200") or 1200)
                import resource as _res
                _lim = _mem_mb * 1048576
                def _preexec():  # noqa: E306
                    try:
                        _res.setrlimit(_res.RLIMIT_AS, (_lim, _lim))
                    except Exception:
                        pass
            except Exception:
                _preexec = None
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout_s,
                shell=False,
                preexec_fn=_preexec,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            result["error"] = f"gltfpack timed out after {timeout_s}s"
            print(f"[GLB_PACK] fail: {result['error']} path={path}")
            return result
        except OSError as exc:
            result["error"] = f"gltfpack failed to start: {exc}"
            print(f"[GLB_PACK] fail: {result['error']} path={path}")
            return result

        if proc.returncode != 0:
            stderr_tail = (proc.stderr or b"").decode("utf-8", "replace")[-500:]
            result["error"] = f"gltfpack exited {proc.returncode}: {stderr_tail}"
            print(f"[GLB_PACK] fail: {result['error']} path={path}")
            return result

        if not tmp_path.exists():
            result["error"] = "gltfpack produced no output file"
            print(f"[GLB_PACK] fail: {result['error']} path={path}")
            return result

        try:
            after = tmp_path.stat().st_size
        except OSError as exc:
            result["error"] = f"output file not accessible: {exc}"
            print(f"[GLB_PACK] fail: {result['error']} path={path}")
            return result

        if after <= 0:
            result["error"] = "output file is empty"
            print(f"[GLB_PACK] fail: {result['error']} path={path}")
            return result

        try:
            data = tmp_path.read_bytes()
        except OSError as exc:
            result["error"] = f"could not read output file: {exc}"
            print(f"[GLB_PACK] fail: {result['error']} path={path}")
            return result

        valid, verr = _validate_glb_meshopt(data)
        if not valid:
            result["error"] = f"output failed validation: {verr}"
            print(f"[GLB_PACK] fail: {result['error']} path={path}")
            return result

        if after >= before:
            result["error"] = f"output not smaller (before={before} after={after})"
            print(f"[GLB_PACK] skip: {result['error']} path={path}")
            return result

        # Atomic replace of the original file.
        os.replace(str(tmp_path), str(path))
        tmp_path = None  # replaced; nothing left to clean up

        result["ok"] = True
        result["after"] = after
        result["ms"] = int((time.monotonic() - t0) * 1000)
        pct = (1 - after / before) * 100 if before else 0.0
        print(
            f"[GLB_PACK] ok: {path} {before} -> {after} bytes "
            f"(-{pct:.1f}%) in {result['ms']}ms"
        )
        return result

    except Exception as exc:  # noqa: BLE001 - never raise from this function
        result["error"] = f"unexpected error: {exc}"
        print(f"[GLB_PACK] fail: {result['error']} path={path}")
        return result

    finally:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
