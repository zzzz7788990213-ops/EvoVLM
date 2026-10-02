#!/usr/bin/env python3
"""Build the canonical Gate 1 v3 provenance manifests and freeze document."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.provenance import (
    FREEZE_PATH,
    GT_PATH,
    IMAGE_MANIFEST_PATH,
    IMAGE_ZIP_PATH,
    MODEL_SPECS,
    PROJECT_ROOT,
    REQUIRED_MODEL_FILES,
    document_sha256,
    evaluation_dependency_closure,
    image_manifest_document,
    local_dependency_closure,
    model_manifest_document,
    sha256_file,
    write_json_atomic,
)

PROTOCOL = PROJECT_ROOT / "evaluation/protocol.json"
SCHEMA = PROJECT_ROOT / "evaluation/result_schema.json"
LOCK = PROJECT_ROOT / "requirements.lock.txt"
MANIFEST_ROOT = PROJECT_ROOT / "evaluation/manifests"


def main() -> int:
    image_document = image_manifest_document()
    write_json_atomic(IMAGE_MANIFEST_PATH, image_document)
    image_manifest_sha = sha256_file(IMAGE_MANIFEST_PATH)

    models = {}
    for model_id, spec in MODEL_SPECS.items():
        document = model_manifest_document(model_id)
        manifest_path = PROJECT_ROOT / spec["manifest"]
        write_json_atomic(manifest_path, document)
        entries = {entry["relative_path"]: entry["sha256"] for entry in document["entries"]}
        models[model_id] = {
            "revision": spec["revision"],
            "snapshot_relative_to_evo": document["snapshot_relative_to_evo"],
            "manifest_path": spec["manifest"],
            "manifest_sha256": sha256_file(manifest_path),
            "required_file_sha256": {
                name: entries[name] for name in REQUIRED_MODEL_FILES
            },
        }

    manifests = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifests[path.stem] = {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(path),
        }

    document = {
        "schema_version": 3,
        "protocol_version": "gate1-exact-match-v3",
        "dataset": {
            "ground_truth_path": GT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "ground_truth_sha256": sha256_file(GT_PATH),
            "image_zip_path_relative_to_evo": IMAGE_ZIP_PATH.relative_to(
                PROJECT_ROOT.parents[1]
            ).as_posix(),
            "image_zip_sha256": sha256_file(IMAGE_ZIP_PATH),
            "image_manifest_path": IMAGE_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "image_manifest_sha256": image_manifest_sha,
            "image_count": image_document["count"],
        },
        "models": models,
        "evaluator_dependency_sha256": evaluation_dependency_closure(),
        "program_dependency_sha256": {
            program: local_dependency_closure(PROJECT_ROOT / f"{program}.py")
            for program in ("baseline_instruct", "baseline_thinking")
        },
        "manifests": manifests,
        "protocol_path": PROTOCOL.relative_to(PROJECT_ROOT).as_posix(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "requirements_lock_path": LOCK.relative_to(PROJECT_ROOT).as_posix(),
        "requirements_lock_sha256": sha256_file(LOCK),
        "result_schema_path": SCHEMA.relative_to(PROJECT_ROOT).as_posix(),
        "result_schema_sha256": sha256_file(SCHEMA),
    }
    document["freeze_id"] = document_sha256(document)
    write_json_atomic(FREEZE_PATH, document)
    print(
        json.dumps(
            {
                "freeze_id": document["freeze_id"],
                "freeze_file_sha256": sha256_file(FREEZE_PATH),
                "ground_truth_sha256": document["dataset"]["ground_truth_sha256"],
                "image_zip_sha256": document["dataset"]["image_zip_sha256"],
                "image_manifest_sha256": image_manifest_sha,
                "model_manifest_sha256": {
                    model_id: value["manifest_sha256"]
                    for model_id, value in models.items()
                },
                "protocol_sha256": document["protocol_sha256"],
                "result_schema_sha256": document["result_schema_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
