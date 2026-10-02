"""Pure-CPU verification for immutable Phase 2/Phase 3 closeout evidence.

The module deliberately imports no GPU-facing package. A verified hard stop is
BLOCKED, never PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any

BLOCKED_EXIT = 2
_HEX64 = set("0123456789abcdef")


class VerificationError(RuntimeError):
    """Closeout evidence failed closed."""


def _reject_constant(value: str) -> None:
    raise VerificationError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON {path}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise VerificationError("artifact path must be a non-empty string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "\\" in value or "\x00" in value:
        raise VerificationError(f"unsafe artifact path: {value!r}")
    if pure.as_posix() in {"", "."} or pure.as_posix() != value:
        raise VerificationError(f"non-canonical artifact path: {value!r}")
    return value


def _regular_inside(root: Path, relative: str) -> Path:
    path = root / relative
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise VerificationError(f"missing artifact: {relative}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise VerificationError(f"artifact is not a regular non-symlink file: {relative}")
    if not path.resolve(strict=True).is_relative_to(root):
        raise VerificationError(f"artifact escaped root: {relative}")
    return path


def _json_pointer(document: Any, pointer: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise VerificationError(f"invalid JSON pointer: {pointer!r}")
    current = document
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not token.isdigit() or int(token) >= len(current):
                raise VerificationError(f"invalid list pointer: {pointer}")
            current = current[int(token)]
        elif isinstance(current, dict):
            if token not in current:
                raise VerificationError(f"missing JSON pointer: {pointer}")
            current = current[token]
        else:
            raise VerificationError(f"JSON pointer traversed scalar: {pointer}")
    return current


def _validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise VerificationError("closeout manifest must be an object")
    required = {
        "schema_version",
        "status",
        "formal_complete",
        "phase2_complete",
        "artifacts",
        "json_assertions",
        "phase3",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise VerificationError(f"manifest missing fields: {missing}")
    if manifest["schema_version"] != 1:
        raise VerificationError("unsupported closeout manifest version")
    if manifest["status"] != "BLOCKED_PHASE3_TIMEOUT":
        raise VerificationError("closeout status is not the frozen timeout status")
    if manifest["formal_complete"] is not False or manifest["phase2_complete"] is not True:
        raise VerificationError("invalid completion flags")
    if not isinstance(manifest["artifacts"], list) or not manifest["artifacts"]:
        raise VerificationError("artifact list must be non-empty")
    if not isinstance(manifest["json_assertions"], list):
        raise VerificationError("json_assertions must be a list")
    expected_phase3 = {
        "campaign_id": "6beb857c-ef2c-4e0a-a6e6-34abee7f4a13",
        "completed_runs": 7,
        "planned_runs": 30,
        "failed_run_index": 8,
        "timeout_seconds": 7200,
        "automatic_retry": False,
        "gpu1_queried_or_used": False,
    }
    if manifest["phase3"] != expected_phase3:
        raise VerificationError("phase3 hard-stop summary does not match frozen facts")
    return manifest


def verify(root: Path, manifest_path: Path, scope: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise VerificationError(f"root is not a directory: {root}")
    manifest_path = manifest_path.resolve(strict=True)
    if not manifest_path.is_relative_to(root) or manifest_path.is_symlink():
        raise VerificationError("manifest must be a regular file inside root")
    manifest = _validate_manifest(load_json(manifest_path))

    verified: dict[str, dict[str, Any]] = {}
    scopes: dict[str, str] = {}
    for record in manifest["artifacts"]:
        if not isinstance(record, dict) or set(record) != {
            "path", "scope", "sha256", "size_bytes"
        }:
            raise VerificationError("invalid artifact record fields")
        relative = _safe_relative(record["path"])
        if relative in scopes:
            raise VerificationError(f"duplicate artifact record: {relative}")
        artifact_scope = record["scope"]
        if artifact_scope not in {"phase2", "phase3"}:
            raise VerificationError(f"invalid artifact scope: {artifact_scope!r}")
        scopes[relative] = artifact_scope
        expected_hash = record["sha256"]
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or set(expected_hash) - _HEX64
        ):
            raise VerificationError(f"invalid SHA-256 for {relative}")
        expected_size = record["size_bytes"]
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size <= 0
        ):
            raise VerificationError(f"invalid size for {relative}")
        if scope != "all" and artifact_scope != scope:
            continue
        path = _regular_inside(root, relative)
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        if actual_size != expected_size:
            raise VerificationError(f"size mismatch: {relative}")
        if actual_hash != expected_hash:
            raise VerificationError(f"SHA-256 mismatch: {relative}")
        verified[relative] = {"sha256": actual_hash, "size_bytes": actual_size}

    checked = 0
    loaded: dict[str, Any] = {}
    for assertion in manifest["json_assertions"]:
        if not isinstance(assertion, dict) or set(assertion) != {
            "path", "pointer", "expected"
        }:
            raise VerificationError("invalid JSON assertion fields")
        relative = _safe_relative(assertion["path"])
        if relative not in scopes:
            raise VerificationError(f"assertion references untracked artifact: {relative}")
        if scope != "all" and scopes[relative] != scope:
            continue
        if relative not in verified:
            raise VerificationError(f"assertion artifact was not verified: {relative}")
        if relative not in loaded:
            loaded[relative] = load_json(root / relative)
        actual = _json_pointer(loaded[relative], assertion["pointer"])
        expected = assertion["expected"]
        if actual != expected or type(actual) is not type(expected):
            raise VerificationError(
                f"JSON assertion mismatch: {relative}{assertion['pointer']} "
                f"expected={expected!r} actual={actual!r}"
            )
        checked += 1

    selected = sum(
        1 for item in manifest["artifacts"]
        if scope == "all" or item["scope"] == scope
    )
    if len(verified) != selected:
        raise VerificationError("not every selected artifact was verified")
    if checked == 0:
        raise VerificationError("no semantic assertions were checked")
    blocked = scope in {"all", "phase3"}
    return {
        "schema_version": 1,
        "scope": scope,
        "artifact_count": len(verified),
        "assertion_count": checked,
        "formal_complete": False,
        "status": "blocked" if blocked else "pass",
        "blocked_reason": "phase3_run08_timeout_7200s" if blocked else None,
        "gpu_called": False,
        "torch_imported": "torch" in sys.modules,
    }


def _write_no_replace(path: Path, payload: bytes, root: Path) -> None:
    parent = path.parent.resolve(strict=True)
    if not parent.is_relative_to(root):
        raise VerificationError("JSON output parent escaped root")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scope", choices=("phase2", "phase3", "all"), default="all")
    parser.add_argument("--allow-blocked", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    manifest = args.manifest or (root / "evaluation" / "closeout_manifest.json")
    try:
        result = verify(root, manifest, args.scope)
    except VerificationError as exc:
        result = {
            "schema_version": 1,
            "scope": args.scope,
            "status": "fail",
            "error": str(exc),
            "gpu_called": False,
        }
        rc = 1
    else:
        rc = 0 if result["status"] == "pass" or args.allow_blocked else BLOCKED_EXIT
    encoded = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if args.json_out:
        output = args.json_out if args.json_out.is_absolute() else root / args.json_out
        _write_no_replace(output, encoded, root)
    sys.stdout.buffer.write(encoded)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
