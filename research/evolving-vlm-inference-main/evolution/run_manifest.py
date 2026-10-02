"""Immutable pre-GPU source/model/data closure for Phase 2 production runs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from .dsl import MODEL_METADATA, MODEL_IDS
from .mutation_catalog import (
    CATALOG_MANIFEST_PATH,
    CATALOG_PATH,
    CATALOG_VERSION,
    load_frozen_catalog,
)
from .process_safety import EVO_ROOT, PROJECT_ROOT, atomic_create_json


RUN_MANIFEST_VERSION = "phase2-production-run-manifest-v1"
GPU_SCOPE = {
    "physical_gpu_index": 0,
    "cuda_visible_devices": "0",
    "visible_device_count": 1,
    "logical_device": "cuda:0",
}
_EXECUTABLE_SUFFIXES = {".py", ".pyc", ".pyo", ".so", ".pth"}
_ROOT_EXECUTABLE_NAMES = {
    "evaluate.py",
    "evolve_instruct.py",
    "evolve_thinking.py",
    "evolved_instruct.py",
    "evolved_thinking.py",
    "manual_instruct.py",
    "manual_thinking.py",
}


class RunManifestError(RuntimeError):
    """A production source closure is absent, stale, or unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(EVO_ROOT), *arguments],
        cwd=str(EVO_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
        check=False,
    )
    if result.returncode != 0:
        raise RunManifestError(
            "git source closure failed: " + result.stderr.strip()[:512]
        )
    return result.stdout.strip()


def source_identity() -> dict[str, str]:
    dirty = _git("status", "--porcelain=v1", "--untracked-files=no")
    if dirty:
        raise RunManifestError("production run requires a clean tracked tree")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "tree": _git("rev-parse", "HEAD^{tree}"),
    }


def _tracked_files() -> set[str]:
    prefix = PROJECT_ROOT.relative_to(EVO_ROOT).as_posix() + "/"
    values = set()
    for line in _git("ls-files", "-z").split("\x00"):
        if line.startswith(prefix):
            values.add(line[len(prefix):])
    return values


def _reject_untracked_importable_source(
    tracked: set[str],
    *,
    allowed_generated: Mapping[str, str] | None = None,
) -> None:
    allowed = dict(allowed_generated or {})
    generated_names = {"evolved_instruct.py", "evolved_thinking.py"}
    for relative, digest in allowed.items():
        if (
            relative not in generated_names
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RunManifestError("generated source allowance is invalid")
    candidates: list[Path] = []
    evolution = PROJECT_ROOT / "evolution"
    for root, directories, files in os.walk(evolution, followlinks=False):
        root_path = Path(root)
        directories[:] = sorted(
            name
            for name in directories
            if not (root_path / name).is_symlink()
        )
        for name in sorted(files):
            path = root_path / name
            if path.suffix in _EXECUTABLE_SUFFIXES:
                candidates.append(path)
    candidates.extend(PROJECT_ROOT / name for name in _ROOT_EXECUTABLE_NAMES)
    rogue: list[str] = []
    accepted_generated: set[str] = set()
    for path in candidates:
        if not path.exists() and not path.is_symlink():
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if path.is_symlink() or not path.is_file():
            rogue.append(relative)
            continue
        if path.suffix in {".pyc", ".pyo", ".so", ".pth"}:
            rogue.append(relative)
            continue
        if relative not in tracked:
            expected = allowed.get(relative)
            if expected is not None and sha256_file(path) == expected:
                accepted_generated.add(relative)
            else:
                rogue.append(relative)
    missing = sorted(set(allowed) - accepted_generated)
    if rogue or missing:
        detail = sorted(rogue)[:32]
        if missing:
            detail.extend(f"missing-or-changed:{item}" for item in missing)
        raise RunManifestError(
            "untracked/importable executable source is forbidden: "
            + repr(detail[:32])
        )


def _regular_internal(path: Path, *, root: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RunManifestError(f"closure path is not a regular file: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise RunManifestError(f"closure path escaped its root: {path}")
    return resolved


def _source_closure(tracked: set[str]) -> list[dict[str, object]]:
    paths = sorted(
        relative
        for relative in tracked
        if (
            (relative.startswith("evolution/") and relative.endswith(".py"))
            or relative in _ROOT_EXECUTABLE_NAMES
            or relative == "requirements.lock.txt"
        )
    )
    required_runtime = {
        "evolution/catalog_provider.py",
        "evolution/local_provider.py",
        "evolution/mutation_catalog.py",
    }
    if not paths or not required_runtime.issubset(paths):
        raise RunManifestError("active runtime source closure is incomplete")
    closure: list[dict[str, object]] = []
    prefix = PROJECT_ROOT.relative_to(EVO_ROOT).as_posix()
    for relative in paths:
        path = _regular_internal(PROJECT_ROOT / relative, root=PROJECT_ROOT)
        git_line = _git(
            "ls-tree",
            "HEAD",
            f"{prefix}/{relative}",
        ).split()
        if len(git_line) < 3 or git_line[1] != "blob":
            raise RunManifestError(f"runtime source is not a tracked blob: {relative}")
        closure.append(
            {
                "path": relative,
                "mode": git_line[0],
                "git_blob": git_line[2],
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return closure


def _model_closure(model_id: str) -> dict[str, object]:
    if model_id not in MODEL_IDS:
        raise RunManifestError("run manifest model is unsupported")
    metadata = MODEL_METADATA[model_id]
    snapshot = EVO_ROOT / "models" / metadata["directory"]
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise RunManifestError("model snapshot is not a physical directory")
    resolved_snapshot = snapshot.resolve(strict=True)
    if not resolved_snapshot.is_relative_to(EVO_ROOT):
        raise RunManifestError("model snapshot escaped Evo")
    files: list[dict[str, object]] = []
    for root, directories, names in os.walk(resolved_snapshot, followlinks=False):
        root_path = Path(root)
        if any((root_path / name).is_symlink() for name in directories):
            raise RunManifestError("model snapshot contains a directory symlink")
        directories[:] = sorted(directories)
        for name in sorted(names):
            path = root_path / name
            regular = _regular_internal(path, root=resolved_snapshot)
            files.append(
                {
                    "path": regular.relative_to(resolved_snapshot).as_posix(),
                    "size_bytes": regular.stat().st_size,
                    "sha256": sha256_file(regular),
                }
            )
    if not files:
        raise RunManifestError("model snapshot closure is empty")
    weight = next(
        (item for item in files if item["path"] == "model.safetensors"),
        None,
    )
    if weight is None or weight["sha256"] != metadata["weight_sha256"]:
        raise RunManifestError("model weight hash does not match locked metadata")
    canonical = json.dumps(
        files,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "model_id": model_id,
        "revision": metadata["revision"],
        "weight_sha256": metadata["weight_sha256"],
        "ordered_files": files,
        "ordered_files_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _manifest_closure(manifest_root: Path) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for name in ("warmup.json", "single.json", "search12.json", "selection32.json"):
        path = _regular_internal(manifest_root / name, root=EVO_ROOT)
        values.append(
            {
                "path": path.relative_to(EVO_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return values


def _catalog_closure() -> dict[str, object]:
    catalog, manifest = load_frozen_catalog()
    catalog_path = _regular_internal(CATALOG_PATH, root=EVO_ROOT)
    manifest_path = _regular_internal(CATALOG_MANIFEST_PATH, root=EVO_ROOT)
    return {
        "catalog_version": CATALOG_VERSION,
        "catalog": {
            "path": catalog_path.relative_to(EVO_ROOT).as_posix(),
            "size_bytes": catalog_path.stat().st_size,
            "sha256": sha256_file(catalog_path),
        },
        "manifest": {
            "path": manifest_path.relative_to(EVO_ROOT).as_posix(),
            "size_bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
        "entry_count": len(catalog["choices"]),
        "manifest_document": manifest,
    }


def create_run_manifest(
    *,
    path: Path,
    campaign_id: str,
    kind: str,
    model_id: str,
    manifest_root: Path,
    versions: Mapping[str, str],
    output_allowlist: Sequence[str],
) -> tuple[dict[str, object], str]:
    tracked = _tracked_files()
    _reject_untracked_importable_source(tracked)
    mutation_model = next(
        model for model in MODEL_IDS if model.endswith("-Instruct")
    )
    models = [mutation_model]
    if model_id != mutation_model:
        models.append(model_id)
    document: dict[str, object] = {
        "run_manifest_version": RUN_MANIFEST_VERSION,
        "execution_class": "production_evolution",
        "campaign_id": campaign_id,
        "kind": kind,
        "source": source_identity(),
        "versions": dict(sorted(versions.items())),
        "source_closure": _source_closure(tracked),
        "models": [_model_closure(value) for value in models],
        "phase2_manifests": _manifest_closure(manifest_root),
        "mutation_catalog": _catalog_closure(),
        "gpu_scope": dict(GPU_SCOPE),
        "network_policy": "offline-zero-network",
        "output_allowlist": sorted(output_allowlist),
    }
    atomic_create_json(path, document)
    return verify_run_manifest(path)


def verify_run_manifest(
    path: Path,
    *,
    expected_sha256: str | None = None,
    allowed_generated_source: Mapping[str, str] | None = None,
) -> tuple[dict[str, object], str]:
    physical = _regular_internal(path, root=EVO_ROOT)
    actual_sha = sha256_file(physical)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise RunManifestError("run manifest SHA-256 mismatch")
    try:
        value = json.loads(physical.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunManifestError("run manifest is not strict JSON") from exc
    if not isinstance(value, dict):
        raise RunManifestError("run manifest is not an object")
    required = {
        "run_manifest_version",
        "execution_class",
        "campaign_id",
        "kind",
        "source",
        "versions",
        "source_closure",
        "models",
        "phase2_manifests",
        "mutation_catalog",
        "gpu_scope",
        "network_policy",
        "output_allowlist",
    }
    if set(value) != required:
        raise RunManifestError("run manifest fields are not exact")
    if (
        value["run_manifest_version"] != RUN_MANIFEST_VERSION
        or value["execution_class"] != "production_evolution"
        or value["source"] != source_identity()
        or value["gpu_scope"] != GPU_SCOPE
        or value["network_policy"] != "offline-zero-network"
    ):
        raise RunManifestError("run manifest fixed policy is stale")
    tracked = _tracked_files()
    _reject_untracked_importable_source(
        tracked,
        allowed_generated=allowed_generated_source,
    )
    if value["source_closure"] != _source_closure(tracked):
        raise RunManifestError("run manifest source closure changed")
    if value["mutation_catalog"] != _catalog_closure():
        raise RunManifestError("run manifest mutation catalog closure changed")
    versions = value["versions"]
    if (
        not isinstance(versions, dict)
        or not versions
        or any(not isinstance(key, str) or not isinstance(item, str) for key, item in versions.items())
    ):
        raise RunManifestError("run manifest version closure is invalid")
    models = value["models"]
    if not isinstance(models, list) or not models or any(
        not isinstance(item, dict) for item in models
    ):
        raise RunManifestError("run manifest model closure is invalid")
    model_ids: list[str] = []
    for item in models:
        model_id = item.get("model_id")
        if not isinstance(model_id, str) or model_id in model_ids:
            raise RunManifestError("run manifest model identity is invalid")
        model_ids.append(model_id)
        if item != _model_closure(model_id):
            raise RunManifestError("run manifest model closure changed")
    manifests = value["phase2_manifests"]
    if not isinstance(manifests, list) or len(manifests) != 4 or any(
        not isinstance(item, dict)
        or set(item) != {"path", "size_bytes", "sha256"}
        for item in manifests
    ):
        raise RunManifestError("run manifest data closure is invalid")
    expected_names = ["warmup.json", "single.json", "search12.json", "selection32.json"]
    actual_names: list[str] = []
    parents: set[str] = set()
    for item in manifests:
        relative_value = item["path"]
        if not isinstance(relative_value, str):
            raise RunManifestError("run manifest data path is invalid")
        relative = Path(relative_value)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise RunManifestError("run manifest data path is unsafe")
        physical_path = _regular_internal(EVO_ROOT / relative, root=EVO_ROOT)
        if (
            item["size_bytes"] != physical_path.stat().st_size
            or item["sha256"] != sha256_file(physical_path)
        ):
            raise RunManifestError("run manifest data closure changed")
        actual_names.append(relative.name)
        parents.add(relative.parent.as_posix())
    if actual_names != expected_names or len(parents) != 1:
        raise RunManifestError("run manifest data ordering is invalid")
    allowlist = value["output_allowlist"]
    if not isinstance(allowlist, list) or allowlist != sorted(set(allowlist)):
        raise RunManifestError("run manifest output allowlist is invalid")
    for entry in allowlist:
        if not isinstance(entry, str):
            raise RunManifestError("run manifest output path is invalid")
        relative = Path(entry)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise RunManifestError("run manifest output path is unsafe")
    if (
        not isinstance(value["campaign_id"], str)
        or not value["campaign_id"]
        or not isinstance(value["kind"], str)
        or not value["kind"]
    ):
        raise RunManifestError("run manifest identity is invalid")
    return value, actual_sha


__all__ = [
    "GPU_SCOPE",
    "RUN_MANIFEST_VERSION",
    "RunManifestError",
    "create_run_manifest",
    "sha256_file",
    "source_identity",
    "verify_run_manifest",
]
