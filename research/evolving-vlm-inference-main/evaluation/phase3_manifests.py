"""Deterministic Phase 3 manifests; ground truth remains parent-owned."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from evaluate import DATA_PATH, IMAGES_PATH, _expected_answer
from descriptive_utils import build_descriptive_quries
from evolution.phase2_manifests import (
    canonical_manifest_documents as phase2_manifest_documents,
)
from evolution.process_safety import EVO_ROOT, atomic_create_json


PROJECT_ROOT = EVO_ROOT / "workspace/evolving-vlm-inference-main"
MANIFEST_ROOT = PROJECT_ROOT / "data/phase3/manifests"
MANIFEST_VERSION = "phase3-fixed-evaluation-v1"
SELECTION_SALT = "cfp04-phase3-speed-fixed-hash-2026-08-24-v1"
NAMES = (
    "single1",
    "dev128",
    "evolution_used",
    "untouched_complement",
    "speed32",
    "warmup1",
)


class Phase3ManifestError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_hash(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _load_source() -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        ground_truth = json.load(handle)
    if not isinstance(ground_truth, dict):
        raise Phase3ManifestError("descriptive ground truth must be an object")
    built = build_descriptive_quries(ground_truth, str(IMAGES_PATH))
    if not isinstance(built, dict) or len(built) < 128:
        raise Phase3ManifestError("descriptive query source is incomplete")
    items: list[tuple[str, dict[str, Any]]] = []
    for key, query in built.items():
        if not isinstance(key, str) or not isinstance(query, dict):
            raise Phase3ManifestError("descriptive query entry is invalid")
        items.append((key, query))
    if len({key for key, _ in items}) != len(items):
        raise Phase3ManifestError("descriptive query keys are not unique")
    return ground_truth, items


def _rank(
    items: Iterable[tuple[str, dict[str, Any]]], role: str
) -> list[tuple[str, dict[str, Any]]]:
    return sorted(
        items,
        key=lambda item: (
            hashlib.sha256(
                f"{SELECTION_SALT}:{role}:{item[0]}".encode("utf-8")
            ).hexdigest(),
            item[0],
        ),
    )


def _one_per_figure(
    ranked: Iterable[tuple[str, dict[str, Any]]], count: int
) -> list[tuple[str, dict[str, Any]]]:
    selected: list[tuple[str, dict[str, Any]]] = []
    figures: set[str] = set()
    for item in ranked:
        figure_id = str(item[1]["figure_id"])
        if figure_id in figures:
            continue
        figures.add(figure_id)
        selected.append(item)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise Phase3ManifestError("cannot allocate figure-isolated manifest")
    return selected


def _document(
    *,
    name: str,
    items: list[tuple[str, dict[str, Any]]],
    ground_truth_sha256: str,
    selection: str,
    extra: dict[str, object],
) -> dict[str, object]:
    keys = [key for key, _ in items]
    figures = [str(query["figure_id"]) for _, query in items]
    return {
        "manifest_version": MANIFEST_VERSION,
        "name": name,
        "sample_count": len(items),
        "selection": selection,
        "selection_salt_sha256": hashlib.sha256(
            SELECTION_SALT.encode("utf-8")
        ).hexdigest(),
        "ground_truth_path": "charxiv/data/descriptive_val.json",
        "ground_truth_sha256": ground_truth_sha256,
        "query_keys": keys,
        "figure_ids": figures,
        "query_key_order_sha256": ordered_hash(keys),
        "figure_order_sha256": ordered_hash(figures),
        **extra,
    }


def canonical_manifest_documents() -> dict[str, dict[str, object]]:
    _, source = _load_source()
    ground_truth_sha256 = sha256_file(DATA_PATH)
    phase2 = phase2_manifest_documents()
    evolution_keys: list[str] = []
    evolution_figures: list[str] = []
    for stage in ("warmup", "single", "search12", "selection32"):
        for key, figure in zip(
            phase2[stage]["query_keys"], phase2[stage]["figure_ids"]
        ):
            if str(key) not in evolution_keys:
                evolution_keys.append(str(key))
            if str(figure) not in evolution_figures:
                evolution_figures.append(str(figure))
    by_key = dict(source)
    evolution_items = [(key, by_key[key]) for key in evolution_keys]
    evolution_figure_set = set(evolution_figures)
    dev128 = source[:128]
    dev_figures = {str(query["figure_id"]) for _, query in dev128}
    untouched = [
        item for item in source
        if str(item[1]["figure_id"]) not in evolution_figure_set
    ]
    dev_untouched_keys = [
        key for key, query in dev128
        if str(query["figure_id"]) not in evolution_figure_set
    ]
    speed_pool = [
        item for item in untouched
        if str(item[1]["figure_id"]) not in dev_figures
    ]
    speed32 = _one_per_figure(_rank(speed_pool, "speed32"), 32)
    speed_figures = {str(query["figure_id"]) for _, query in speed32}
    warmup_pool = [
        item for item in untouched
        if str(item[1]["figure_id"]) not in dev_figures | speed_figures
    ]
    warmup1 = _one_per_figure(_rank(warmup_pool, "warmup1"), 1)
    common = {
        "source_query_count": len(source),
        "phase2_manifest_version": phase2["warmup"]["manifest_version"],
    }
    documents = {
        "single1": _document(
            name="single1", items=dev128[:1],
            ground_truth_sha256=ground_truth_sha256,
            selection="first query of frozen Dev128", extra=common,
        ),
        "dev128": _document(
            name="dev128", items=dev128,
            ground_truth_sha256=ground_truth_sha256,
            selection="first 128 queries in descriptive_val insertion order",
            extra={
                **common,
                "untouched_complement_query_keys": dev_untouched_keys,
                "untouched_complement_query_key_order_sha256": ordered_hash(
                    dev_untouched_keys
                ),
            },
        ),
        "evolution_used": _document(
            name="evolution_used", items=evolution_items,
            ground_truth_sha256=ground_truth_sha256,
            selection="canonical Phase2 warmup+single+Search12+Selection32 union",
            extra=common,
        ),
        "untouched_complement": _document(
            name="untouched_complement", items=untouched,
            ground_truth_sha256=ground_truth_sha256,
            selection="source-order queries whose figure was never used in Phase2",
            extra=common,
        ),
        "speed32": _document(
            name="speed32", items=speed32,
            ground_truth_sha256=ground_truth_sha256,
            selection="fixed salted hash rank from non-Dev128 untouched figures; one per figure",
            extra=common,
        ),
        "warmup1": _document(
            name="warmup1", items=warmup1,
            ground_truth_sha256=ground_truth_sha256,
            selection="fixed salted hash rank excluding Dev128/evolution/speed figures",
            extra=common,
        ),
    }
    if set(documents) != set(NAMES):
        raise AssertionError("Phase3 manifest role set drifted")
    if set(documents["speed32"]["figure_ids"]) & set(
        documents["warmup1"]["figure_ids"]
    ):
        raise AssertionError("speed and warmup figures overlap")
    if len(set(documents["speed32"]["figure_ids"])) != 32:
        raise AssertionError("speed manifest is not figure isolated")
    return documents


def write_manifests(root: Path = MANIFEST_ROOT) -> dict[str, Path]:
    documents = canonical_manifest_documents()
    paths: dict[str, Path] = {}
    for name in NAMES:
        path = root / f"{name}.json"
        atomic_create_json(path, documents[name])
        paths[name] = path
    return paths


def load_manifest(name: str, root: Path = MANIFEST_ROOT) -> dict[str, object]:
    if name not in NAMES:
        raise Phase3ManifestError("unknown Phase3 manifest")
    path = root / f"{name}.json"
    if path.is_symlink() or not path.is_file():
        raise Phase3ManifestError(f"missing physical manifest: {name}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(EVO_ROOT):
        raise Phase3ManifestError("manifest escaped Evo")
    with resolved.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if document != canonical_manifest_documents()[name]:
        raise Phase3ManifestError(f"manifest differs from canonical selection: {name}")
    return document


def load_private_queries(
    name: str, root: Path = MANIFEST_ROOT
) -> list[dict[str, object]]:
    document = load_manifest(name, root)
    ground_truth, source = _load_source()
    by_key = dict(source)
    result: list[dict[str, object]] = []
    query_keys = document.get("query_keys")
    if not isinstance(query_keys, list):
        raise Phase3ManifestError("manifest query keys are invalid")
    for raw_key in query_keys:
        key = str(raw_key)
        query = by_key[key]
        image = Path(str(query["figure_path"]))
        if image.is_symlink() or not image.is_file():
            raise Phase3ManifestError("query image is not a physical file")
        resolved = image.resolve(strict=True)
        if not resolved.is_relative_to(EVO_ROOT):
            raise Phase3ManifestError("query image escaped Evo")
        result.append(
            {
                "query_key": key,
                "figure_id": str(query["figure_id"]),
                "subq_idx": int(query["subq_idx"]),
                "qid": str(query["qid"]),
                "question": str(query["question"]),
                "image_path": str(resolved),
                "image_sha256": sha256_file(resolved),
                "expected_answer": _expected_answer(ground_truth, query),
            }
        )
    return result


__all__ = [
    "MANIFEST_ROOT", "MANIFEST_VERSION", "NAMES", "Phase3ManifestError",
    "canonical_manifest_documents", "load_manifest", "load_private_queries",
    "ordered_hash", "sha256_file", "write_manifests",
]
