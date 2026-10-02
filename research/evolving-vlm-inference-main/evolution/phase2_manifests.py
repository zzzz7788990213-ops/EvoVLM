"""Fixed, figure-isolated Phase 2 query manifests owned by the scoring parent."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from evaluate import DATA_PATH, IMAGES_PATH, _expected_answer
from descriptive_utils import build_descriptive_quries

from .process_safety import EVO_ROOT, atomic_create_json
from .target_worker import PrivateQuery


MANIFEST_VERSION = "phase2-local-search-manifests-v2"
MANIFEST_SALT = "cfp04-phase2-fixed-figure-isolated-v2-2026-08-24"
DEFAULT_MANIFEST_ROOT = EVO_ROOT / "data/phase2_local_search_v2/manifests"
STAGE_COUNTS = {"warmup": 1, "single": 1, "search12": 12, "selection32": 32}


class Phase2ManifestError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source() -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        ground_truth = json.load(handle)
    if not isinstance(ground_truth, dict):
        raise Phase2ManifestError("descriptive ground truth is not an object")
    built = build_descriptive_quries(ground_truth, str(IMAGES_PATH))
    if not isinstance(built, dict) or len(built) < sum(STAGE_COUNTS.values()):
        raise Phase2ManifestError("descriptive query source is incomplete")
    items: list[tuple[str, dict[str, Any]]] = []
    for key, query in built.items():
        if not isinstance(key, str) or not isinstance(query, dict):
            raise Phase2ManifestError("descriptive query entry is invalid")
        items.append((key, query))
    return ground_truth, items


def _ranked_figure_isolated(
    items: list[tuple[str, dict[str, Any]]],
) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    ranked = sorted(
        items,
        key=lambda item: (
            hashlib.sha256(
                f"{MANIFEST_SALT}:{item[0]}".encode("utf-8")
            ).hexdigest(),
            item[0],
        ),
    )
    allocated: dict[str, list[tuple[str, dict[str, Any]]]] = {
        stage: [] for stage in STAGE_COUNTS
    }
    globally_used_figures: set[str] = set()
    cursor = 0
    for stage, count in STAGE_COUNTS.items():
        while len(allocated[stage]) < count and cursor < len(ranked):
            key, query = ranked[cursor]
            cursor += 1
            figure_id = str(query.get("figure_id"))
            if figure_id in globally_used_figures:
                continue
            globally_used_figures.add(figure_id)
            allocated[stage].append((key, query))
        if len(allocated[stage]) != count:
            raise Phase2ManifestError(f"cannot allocate figure-isolated stage {stage}")
    if set(
        str(query["figure_id"])
        for _, query in allocated["search12"]
    ) & set(
        str(query["figure_id"])
        for _, query in allocated["selection32"]
    ):
        raise AssertionError("search and selection figures overlap")
    return allocated


def canonical_manifest_documents() -> dict[str, dict[str, object]]:
    _, items = _load_source()
    allocation = _ranked_figure_isolated(items)
    gt_hash = _sha256_file(DATA_PATH)
    documents: dict[str, dict[str, object]] = {}
    for stage, selected in allocation.items():
        keys = [key for key, _ in selected]
        figures = [str(query["figure_id"]) for _, query in selected]
        documents[stage] = {
            "manifest_version": MANIFEST_VERSION,
            "stage": stage,
            "sample_count": STAGE_COUNTS[stage],
            "selection": "sha256 fixed-salt rank; globally figure isolated",
            "salt_sha256": hashlib.sha256(MANIFEST_SALT.encode("utf-8")).hexdigest(),
            "ground_truth_sha256": gt_hash,
            "query_keys": keys,
            "figure_ids": figures,
            "query_key_order_sha256": hashlib.sha256(
                "\n".join(keys).encode("utf-8")
            ).hexdigest(),
        }
    return documents


def create_manifests(
    root: Path = DEFAULT_MANIFEST_ROOT,
) -> dict[str, Path]:
    documents = canonical_manifest_documents()
    paths: dict[str, Path] = {}
    for stage, document in documents.items():
        path = root / f"{stage}.json"
        atomic_create_json(path, document)
        paths[stage] = path
    return paths


def _load_manifest(stage: str, root: Path) -> dict[str, object]:
    if stage not in STAGE_COUNTS:
        raise Phase2ManifestError("unknown Phase 2 stage")
    path = root / f"{stage}.json"
    if path.is_symlink() or not path.is_file():
        raise Phase2ManifestError(f"missing physical manifest: {stage}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(EVO_ROOT):
        raise Phase2ManifestError("manifest escaped Evo")
    with resolved.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    expected = canonical_manifest_documents()[stage]
    if document != expected:
        raise Phase2ManifestError(f"manifest differs from canonical selection: {stage}")
    return document


def load_private_queries(
    stage: str,
    *,
    root: Path = DEFAULT_MANIFEST_ROOT,
) -> list[PrivateQuery]:
    document = _load_manifest(stage, root)
    ground_truth, items = _load_source()
    by_key = dict(items)
    keys = document["query_keys"]
    if not isinstance(keys, list):
        raise Phase2ManifestError("manifest query_keys is invalid")
    queries: list[PrivateQuery] = []
    for index, key in enumerate(keys):
        if not isinstance(key, str) or key not in by_key:
            raise Phase2ManifestError("manifest query key is unavailable")
        query = by_key[key]
        image = Path(str(query["figure_path"]))
        queries.append(
            PrivateQuery(
                request_id=f"{stage}_{index:03d}",
                query_key=key,
                figure_id=str(query["figure_id"]),
                qid=str(query["qid"]),
                question=str(query["question"]),
                image_path=image,
                expected_answer=_expected_answer(ground_truth, query),
            )
        )
    figures = [query.figure_id for query in queries]
    if len(figures) != len(set(figures)):
        raise Phase2ManifestError("manifest is not figure-isolated")
    return queries


def validate_disjoint_manifests(
    root: Path = DEFAULT_MANIFEST_ROOT,
) -> dict[str, object]:
    stages = {
        stage: load_private_queries(stage, root=root) for stage in STAGE_COUNTS
    }
    search = {query.figure_id for query in stages["search12"]}
    selection = {query.figure_id for query in stages["selection32"]}
    if search & selection:
        raise Phase2ManifestError("search12 and selection32 overlap by figure")
    return {
        "manifest_version": MANIFEST_VERSION,
        "stage_counts": {
            stage: len(queries) for stage, queries in stages.items()
        },
        "all_stages_figure_isolated": all(
            len({query.figure_id for query in queries}) == len(queries)
            for queries in stages.values()
        ),
        "search12_selection32_figure_overlap": sorted(search & selection),
        "ground_truth_sha256": _sha256_file(DATA_PATH),
    }


__all__ = [
    "DEFAULT_MANIFEST_ROOT",
    "MANIFEST_VERSION",
    "Phase2ManifestError",
    "canonical_manifest_documents",
    "create_manifests",
    "load_private_queries",
    "validate_disjoint_manifests",
]
