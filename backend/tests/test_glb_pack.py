"""Tests for services.glb_pack (post-export GLB meshopt packing).

Pure post-processing tests: no real gltfpack binary is required. Fake
shims (Python scripts invoked via the platform's script launcher) stand in
for the real gltfpack executable so tests are hermetic and fast.
"""
import json
import os
import stat
import struct
import sys
from pathlib import Path

import pytest

from services import glb_pack


def _pad4(data: bytes, pad_byte: bytes) -> bytes:
    rem = len(data) % 4
    if rem:
        data = data + pad_byte * (4 - rem)
    return data


def _build_minimal_glb(extensions_required=None) -> bytes:
    """Build a minimal, structurally-valid GLB (JSON chunk only, no BIN)."""
    doc = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "scene": 0,
        "nodes": [{"name": "MapLabel"}],
    }
    if extensions_required:
        doc["extensionsRequired"] = list(extensions_required)
        doc["extensionsUsed"] = list(extensions_required)

    json_bytes = json.dumps(doc).encode("utf-8")
    json_bytes = _pad4(json_bytes, b" ")

    json_chunk_header = struct.pack("<I4s", len(json_bytes), b"JSON")
    header_len = 12
    total_length = header_len + 8 + len(json_bytes)
    glb_header = struct.pack("<4sII", b"glTF", 2, total_length)

    return glb_header + json_chunk_header + json_bytes


def _write_fake_gltfpack(tmp_path: Path, script_body: str) -> str:
    """Write a fake gltfpack executable and return its path.

    On Windows we emit a .cmd shim that calls the current Python
    interpreter on a co-located .py file, so no shell scripting quirks
    are involved.
    """
    py_path = tmp_path / "fake_gltfpack_impl.py"
    py_path.write_text(script_body, encoding="utf-8")

    if os.name == "nt":
        shim_path = tmp_path / "fake_gltfpack.cmd"
        shim_path.write_text(
            f'@echo off\r\n"{sys.executable}" "{py_path}" %*\r\n',
            encoding="utf-8",
        )
        return str(shim_path)
    else:
        shim_path = tmp_path / "fake_gltfpack"
        shim_path.write_text(f"#!/bin/sh\n\"{sys.executable}\" \"{py_path}\" \"$@\"\n", encoding="utf-8")
        shim_path.chmod(shim_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return str(shim_path)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Ensure each test starts from a known env state regardless of the
    # real environment (e.g. a locally-installed gltfpack on PATH).
    monkeypatch.delenv("GLTFPACK_BIN", raising=False)
    monkeypatch.delenv("PREVIEW_MESHOPT", raising=False)
    monkeypatch.setattr(glb_pack.shutil, "which", lambda *_a, **_k: None)


def test_binary_missing_leaves_file_untouched(tmp_path, monkeypatch):
    # Force resolution to fail regardless of whether a real gltfpack
    # happens to be installed in this checkout's frontend/node_modules.
    monkeypatch.setattr(glb_pack, "resolve_gltfpack_bin", lambda: None)

    src = tmp_path / "preview.glb"
    original_bytes = _build_minimal_glb()
    src.write_bytes(original_bytes)

    result = glb_pack.pack_glb_inplace(src)

    assert result["ok"] is False
    assert "gltfpack not found" in result["error"]
    assert src.read_bytes() == original_bytes


def test_non_smaller_output_leaves_file_untouched(tmp_path, monkeypatch):
    # Fake gltfpack that just copies input->output verbatim: no meshopt
    # extension, and same size, so packing must be rejected.
    script = (
        "import sys, shutil\n"
        "args = sys.argv[1:]\n"
        "i = args[args.index('-i') + 1]\n"
        "o = args[args.index('-o') + 1]\n"
        "shutil.copyfile(i, o)\n"
    )
    fake_bin = _write_fake_gltfpack(tmp_path, script)
    monkeypatch.setenv("GLTFPACK_BIN", fake_bin)

    src = tmp_path / "preview.glb"
    original_bytes = _build_minimal_glb()
    src.write_bytes(original_bytes)

    result = glb_pack.pack_glb_inplace(src)

    assert result["ok"] is False
    assert result["error"]
    assert src.read_bytes() == original_bytes


def test_valid_smaller_meshopt_output_replaces_file(tmp_path, monkeypatch):
    packed_bytes = _build_minimal_glb(extensions_required=["EXT_meshopt_compression"])

    script = (
        "import sys\n"
        "args = sys.argv[1:]\n"
        "o = args[args.index('-o') + 1]\n"
        f"data = {packed_bytes!r}\n"
        "with open(o, 'wb') as f:\n"
        "    f.write(data)\n"
    )
    fake_bin = _write_fake_gltfpack(tmp_path, script)
    monkeypatch.setenv("GLTFPACK_BIN", fake_bin)

    src = tmp_path / "preview.glb"
    # Make the "before" file bigger than the fake packed output by padding
    # extra bytes onto an otherwise-valid GLB-shaped file.
    original_bytes = _build_minimal_glb() + b"\x00" * 4096
    src.write_bytes(original_bytes)
    assert len(original_bytes) > len(packed_bytes)

    result = glb_pack.pack_glb_inplace(src)

    assert result["ok"] is True
    assert result["error"] is None
    assert result["before"] == len(original_bytes)
    assert result["after"] == len(packed_bytes)
    assert isinstance(result["ms"], int)
    assert src.read_bytes() == packed_bytes

    # Node names must survive (frontend relies on e.g. "MapLabel").
    replaced = src.read_bytes()
    json_len = struct.unpack_from("<I", replaced, 12)[0]
    doc = json.loads(replaced[20:20 + json_len].decode("utf-8"))
    assert doc["nodes"][0]["name"] == "MapLabel"
    assert "EXT_meshopt_compression" in doc["extensionsRequired"]


def test_preview_meshopt_disabled_is_noop(tmp_path, monkeypatch):
    packed_bytes = _build_minimal_glb(extensions_required=["EXT_meshopt_compression"])
    script = (
        "import sys\n"
        "args = sys.argv[1:]\n"
        "o = args[args.index('-o') + 1]\n"
        f"data = {packed_bytes!r}\n"
        "with open(o, 'wb') as f:\n"
        "    f.write(data)\n"
    )
    fake_bin = _write_fake_gltfpack(tmp_path, script)
    monkeypatch.setenv("GLTFPACK_BIN", fake_bin)
    monkeypatch.setenv("PREVIEW_MESHOPT", "0")

    src = tmp_path / "preview.glb"
    original_bytes = _build_minimal_glb() + b"\x00" * 4096
    src.write_bytes(original_bytes)

    result = glb_pack.pack_glb_inplace(src)

    assert result["ok"] is False
    assert "PREVIEW_MESHOPT" in result["error"] or "disabled" in result["error"]
    assert src.read_bytes() == original_bytes


def test_resolve_gltfpack_bin_prefers_env(tmp_path, monkeypatch):
    fake = tmp_path / "gltfpack_bin_stub"
    fake.write_text("stub", encoding="utf-8")
    monkeypatch.setenv("GLTFPACK_BIN", str(fake))

    resolved = glb_pack.resolve_gltfpack_bin()

    assert resolved == str(fake)


def test_is_meshopt_enabled_default_and_toggle(monkeypatch):
    monkeypatch.delenv("PREVIEW_MESHOPT", raising=False)
    assert glb_pack.is_meshopt_enabled() is True

    monkeypatch.setenv("PREVIEW_MESHOPT", "0")
    assert glb_pack.is_meshopt_enabled() is False

    monkeypatch.setenv("PREVIEW_MESHOPT", "1")
    assert glb_pack.is_meshopt_enabled() is True
