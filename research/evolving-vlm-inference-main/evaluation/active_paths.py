"""Single source of truth for the active Gate 1 protocol and artifact paths."""

from __future__ import annotations

ACTIVE_MAJOR = 4
ACTIVE_MINOR = 4
ACTIVE_VERSION_TAG = f"v{ACTIVE_MAJOR}_{ACTIVE_MINOR}"
ACTIVE_VERSION_DOTTED = f"v{ACTIVE_MAJOR}.{ACTIVE_MINOR}"
PROTOCOL_VERSION = f"gate1-exact-match-{ACTIVE_VERSION_DOTTED}"

PROTOCOL_PATH = f"evaluation/protocol_{ACTIVE_VERSION_TAG}.json"
SCHEMA_PATH = f"evaluation/result_schema_{ACTIVE_VERSION_TAG}.json"
FREEZE_PATH = f"evaluation/evaluator_freeze_{ACTIVE_VERSION_TAG}.json"
LOCK_PATH = "requirements.lock.txt"
MANIFEST_ROOT = f"evaluation/manifests/{ACTIVE_VERSION_TAG}"
WARMUP_MANIFEST = f"{MANIFEST_ROOT}/warmup_1.json"
IMAGE_MANIFEST_PATH = (
    f"evaluation/provenance/charxiv_images_manifest_{ACTIVE_VERSION_TAG}.json"
)
MODEL_MANIFEST_PATHS = {
    "Qwen/Qwen3-VL-2B-Instruct": (
        f"evaluation/provenance/qwen3_vl_2b_instruct_manifest_{ACTIVE_VERSION_TAG}.json"
    ),
    "Qwen/Qwen3-VL-2B-Thinking": (
        f"evaluation/provenance/qwen3_vl_2b_thinking_manifest_{ACTIVE_VERSION_TAG}.json"
    ),
}

RESULTS_ROOT = f"results/evaluations/{ACTIVE_VERSION_TAG}"
CAMPAIGN_ROOT = f"{RESULTS_ROOT}/campaigns"
LATEST_PATH = f"{RESULTS_ROOT}/latest_campaign_{ACTIVE_VERSION_TAG}.json"
RECEIPT_PREFIX = f"gate1_{ACTIVE_VERSION_TAG}"
CHECK_PREFIX = f"gate1_{ACTIVE_VERSION_TAG}_check"
OUTPUT_SUFFIX = f"__{ACTIVE_VERSION_TAG}.json"

RUN_SPECS = (
    ("baseline_instruct", "single_1", 1),
    ("baseline_instruct", "smoke_4", 3),
    ("baseline_instruct", "dev_128", 1),
    ("baseline_thinking", "single_1", 1),
    ("baseline_thinking", "smoke_4", 3),
    ("baseline_thinking", "dev_128", 1),
)

ACTIVE_MANIFEST_SPECS = {
    "single_1": (
        "single",
        1,
        "d682c3be7e35f7c298e788f11183bda1f504339363d55b1de6f8368db6ea4dbc",
    ),
    "smoke_4": (
        "smoke",
        4,
        "5e575564283efb2c7066e53fd8eb3da8952f7b1b7e1515e4404011195724923c",
    ),
    "dev_128": (
        "dev",
        128,
        "8971f2eaaf9fff58ea36ddbd7416d22cd84252fece31e2cc55ee4126666cc8d9",
    ),
    "warmup_1": (
        "warmup",
        1,
        "cc96be98e9b467b6bbabd9d4897d7a0d59899922c4372c00b873b3e2667a1e9a",
    ),
}


def canonical_output_filename(program: str, manifest: str, repeats: int) -> str:
    if (program, manifest, repeats) not in RUN_SPECS:
        raise ValueError("run specification is not canonical")
    return f"{program}__{manifest}__r{repeats}{OUTPUT_SUFFIX}"


def canonical_output_filenames() -> tuple[str, ...]:
    return tuple(canonical_output_filename(*spec) for spec in RUN_SPECS)
