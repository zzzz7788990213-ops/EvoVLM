"""Strict, stdlib-only validation of the two frozen model snapshots."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve(strict=True).parent.parent
MANIFEST_ROOT = PROJECT_ROOT / "evaluation/model_snapshots"
MANIFESTS = {
    "Qwen/Qwen3-VL-2B-Instruct": MANIFEST_ROOT / "qwen3_vl_2b_instruct.critical.v1.json",
    "Qwen/Qwen3-VL-2B-Thinking": MANIFEST_ROOT / "qwen3_vl_2b_thinking.critical.v1.json",
}
_MODEL_ENV = {
    "Qwen/Qwen3-VL-2B-Instruct": "EVOLVING_VLM_INSTRUCT_MODEL_PATH",
    "Qwen/Qwen3-VL-2B-Thinking": "EVOLVING_VLM_THINKING_MODEL_PATH",
}
_HASH = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_REQUIRED_RUNTIME = {
    "config.json", "generation_config.json", "preprocessor_config.json",
    "video_preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
    "vocab.json", "merges.txt", "chat_template.json",
}


class SnapshotError(RuntimeError):
    """A critical model snapshot is absent, unsafe, or byte-inconsistent."""


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise SnapshotError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject(value: str) -> object:
    raise SnapshotError(f"non-finite JSON constant: {value}")


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject,
            object_pairs_hook=_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SnapshotError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise SnapshotError(f"{label} is not text")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise SnapshotError(f"unsafe {label}: {value!r}")
    return pure.as_posix()


def _remote_modules(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "auto_map" and isinstance(child, dict):
                for target in child.values():
                    targets = target if isinstance(target, list) else [target]
                    for item in targets:
                        if not isinstance(item, str) or not item:
                            continue
                        module = item.rsplit("--", 1)[-1].split(".", 1)[0]
                        if module and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", module):
                            found.add(module + ".py")
            found.update(_remote_modules(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_remote_modules(child))
    return found


def _actual_layout(root: Path) -> tuple[str, set[str]]:
    index = root / "model.safetensors.index.json"
    weight = root / "model.safetensors"
    if index.exists() or index.is_symlink():
        if index.is_symlink() or not index.is_file():
            raise SnapshotError("weight index is not a regular file")
        document = _load(index)
        weight_map = document.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise SnapshotError("weight index has no non-empty weight_map")
        shards = {_relative(value, "weight shard") for value in weight_map.values()}
        if any("/" in shard or not shard.endswith(".safetensors") for shard in shards):
            raise SnapshotError("weight index shard name is unsafe")
        if weight.exists() or weight.is_symlink():
            raise SnapshotError("snapshot mixes indexed and single-file weights")
        return "indexed_safetensors", {"model.safetensors.index.json", *shards}
    if weight.is_symlink() or not weight.is_file():
        raise SnapshotError("single-file model.safetensors is absent or unsafe")
    return "single_safetensors", {"model.safetensors"}


def _actual_remote_code(root: Path, json_names: set[str]) -> set[str]:
    remote: set[str] = set()
    for name in sorted(json_names):
        remote.update(_remote_modules(_load(root / name)))
    return remote

def _resolve_local_root(
    model_id: str, directory: str, environ: Mapping[str, str]
) -> Path:
    specific = str(environ.get(_MODEL_ENV[model_id], "")).strip()
    generic = str(environ.get("EVOLVING_VLM_MODEL_ROOT", "")).strip()
    if specific:
        candidate = Path(specific)
    elif generic:
        candidate = Path(generic) / directory
    else:
        candidate = PROJECT_ROOT / "models" / directory
        if not candidate.exists() and not candidate.is_symlink():
            raise SnapshotError(
                "fresh reproduction requires local snapshots; set "
                f"{_MODEL_ENV[model_id]} or EVOLVING_VLM_MODEL_ROOT"
            )
    if candidate.is_symlink() or not candidate.is_dir():
        raise SnapshotError(
            f"configured local snapshot is not a physical directory: {candidate}"
        )
    return candidate.resolve(strict=True)


def validate_snapshot(
    model_id: str,
    *,
    environ: Mapping[str, str] | None = None,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    """Resolve and byte-validate one local snapshot without ML-runtime imports."""
    path = MANIFESTS.get(model_id) if manifest_path is None else manifest_path
    if path is None:
        raise SnapshotError(f"unsupported snapshot model: {model_id}")
    document = _load(path)
    required = {
        "schema_version", "model_id", "revision", "processor_revision",
        "model_directory", "weight_layout", "runtime_files",
        "remote_code_files", "files",
    }
    if (
        set(document) != required
        or document.get("schema_version") != "critical-model-snapshot-v1"
    ):
        raise SnapshotError("critical snapshot manifest fields/version differ")
    if document.get("model_id") != model_id:
        raise SnapshotError("critical snapshot model ID differs")
    revision = document.get("revision")
    processor_revision = document.get("processor_revision")
    if (
        not isinstance(revision, str)
        or not _REVISION.fullmatch(revision)
        or processor_revision != revision
    ):
        raise SnapshotError("critical snapshot revision differs")
    directory = document.get("model_directory")
    if (
        not isinstance(directory, str)
        or not re.fullmatch(r"Qwen3-VL-2B-(Instruct|Thinking)", directory)
    ):
        raise SnapshotError("critical snapshot model directory differs")
    files = document.get("files")
    if not isinstance(files, list) or not files:
        raise SnapshotError("critical snapshot files are absent")
    records: dict[str, dict[str, object]] = {}
    for index, item in enumerate(files):
        if (
            not isinstance(item, dict)
            or set(item) != {"relative_path", "role", "size_bytes", "sha256"}
        ):
            raise SnapshotError(f"critical snapshot file fields differ: {index}")
        relative = _relative(item["relative_path"], "snapshot file")
        if relative in records:
            raise SnapshotError(f"duplicate snapshot file: {relative}")
        size = item["size_bytes"]
        digest = item["sha256"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise SnapshotError(f"snapshot file size differs: {relative}")
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise SnapshotError(f"snapshot file hash differs: {relative}")
        if item["role"] not in {
            "weight", "weight_index", "config", "processor", "tokenizer",
            "chat_template", "remote_code",
        }:
            raise SnapshotError(f"snapshot file role differs: {relative}")
        records[relative] = item
    if [item["relative_path"] for item in files] != sorted(records):
        raise SnapshotError("snapshot file records are not canonically ordered")
    runtime_files = document.get("runtime_files")
    remote_files = document.get("remote_code_files")
    if (
        not isinstance(runtime_files, list)
        or runtime_files != sorted(set(runtime_files))
        or not isinstance(remote_files, list)
        or remote_files != sorted(set(remote_files))
    ):
        raise SnapshotError("snapshot runtime/remote-code paths are not canonical")
    runtime_set = {_relative(item, "runtime file") for item in runtime_files}
    remote_set = {_relative(item, "remote-code file") for item in remote_files}
    environment = os.environ if environ is None else environ
    root = _resolve_local_root(model_id, directory, environment)
    layout, weight_files = _actual_layout(root)
    if document.get("weight_layout") != layout:
        raise SnapshotError("snapshot weight layout differs")
    json_files = {name for name in _REQUIRED_RUNTIME if name.endswith(".json")}
    remote_actual = _actual_remote_code(root, json_files)
    expected_runtime = _REQUIRED_RUNTIME | weight_files | remote_actual
    if (
        runtime_set != expected_runtime
        or remote_set != remote_actual
        or set(records) != expected_runtime
    ):
        raise SnapshotError("snapshot runtime-file closure differs")
    verified: list[dict[str, object]] = []
    for relative in sorted(expected_runtime):
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise SnapshotError(f"snapshot file is absent/unsafe: {relative}")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise SnapshotError(f"snapshot file escaped root: {relative}") from exc
        record = records[relative]
        if (
            resolved.stat().st_size != record["size_bytes"]
            or _sha256(resolved) != record["sha256"]
        ):
            raise SnapshotError(f"snapshot file hash/size mismatch: {relative}")
        verified.append(dict(record))
    return {
        "schema_version": document["schema_version"],
        "model_id": model_id,
        "revision": revision,
        "processor_revision": processor_revision,
        "model_directory": directory,
        "model_source": str(root),
        "model_local_files_only": True,
        "weight_layout": layout,
        "manifest_path": path.resolve(strict=True).relative_to(PROJECT_ROOT).as_posix(),
        "manifest_sha256": _sha256(path),
        "manifest_size_bytes": path.stat().st_size,
        "runtime_files": verified,
        "remote_code_files": sorted(remote_actual),
    }


def validate_both_snapshots(
    *, environ: Mapping[str, str] | None = None
) -> dict[str, dict[str, object]]:
    """Fail atomically before inference unless both frozen snapshots validate."""
    return {
        model_id: validate_snapshot(model_id, environ=environ)
        for model_id in sorted(MANIFESTS)
    }
