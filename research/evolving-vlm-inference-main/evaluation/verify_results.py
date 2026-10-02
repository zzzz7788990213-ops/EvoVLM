#!/usr/bin/env python3
"""Validate a newly generated active six-run Gate 1 campaign from current bytes."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from evaluate import load_charxiv_data, load_manifest
from evaluation.active_paths import MANIFEST_ROOT
from evaluation.active_contract import (
    CAMPAIGN_ROOT as CAMPAIGN_ROOT_RELATIVE,
    LATEST_PATH,
    LOCK_PATH as LOCK_PATH_RELATIVE,
    PROTOCOL_PATH as PROTOCOL_PATH_RELATIVE,
    PROTOCOL_VERSION,
    RUN_SPECS,
    SCHEMA_PATH as SCHEMA_PATH_RELATIVE,
    RunOutcome,
    ValidationOutcome,
    canonical_output_filename,
    canonical_provenance_paths,
    canonical_run_paths,
)
from evaluation.metrics import latency_statistics
from evaluation.gpu_scope import (
    AUTHORIZATION_SCOPE,
    CUDA_VISIBLE_DEVICES,
    DEFAULT_FUTURE_GPU,
    LOGICAL_DEVICE,
    PHYSICAL_GPU_INDEX,
    VISIBLE_DEVICE_COUNT,
)
from evaluation.provenance import (
    EVO_ROOT,
    FREEZE_PATH,
    GT_PATH,
    IMAGE_MANIFEST_PATH,
    IMAGE_ZIP_PATH,
    MODEL_SPECS,
    PROJECT_ROOT,
    REQUIRED_MODEL_FILES,
    active_literal_audit_document,
    assert_file_hash,
    evaluation_dependency_closure,
    image_manifest_document,
    load_freeze,
    load_json_strict,
    local_dependency_closure,
    model_manifest_document,
    sha256_bytes,
    sha256_file,
    verify_manifest_file,
    write_json_atomic,
)
from evaluation.schema_validator import validate as execute_schema

SCHEMA_PATH = PROJECT_ROOT / SCHEMA_PATH_RELATIVE
PROTOCOL_PATH = PROJECT_ROOT / PROTOCOL_PATH_RELATIVE
LOCK_PATH = PROJECT_ROOT / LOCK_PATH_RELATIVE
LATEST = PROJECT_ROOT / LATEST_PATH
SAMPLES_BY_MANIFEST = {"single_1": 1, "smoke_4": 4, "dev_128": 128}
EXPECTED_RUNS = tuple(
    (
        canonical_output_filename(*spec),
        spec[0],
        spec[1],
        SAMPLES_BY_MANIFEST[spec[1]],
        spec[2],
    )
    for spec in RUN_SPECS
)

MODEL_BY_PROGRAM = {
    "baseline_instruct": "Qwen/Qwen3-VL-2B-Instruct",
    "baseline_thinking": "Qwen/Qwen3-VL-2B-Thinking",
}


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def assert_uuid4(value: str, label: str) -> None:
    parsed = UUID(value)
    if parsed.version != 4 or str(parsed) != value:
        raise AssertionError(f"{label} is not canonical UUIDv4")


def _unique_raw_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate raw JSON key: {key}")
        result[key] = value
    return result


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(
            line,
            object_pairs_hook=_unique_raw_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-RFC JSON constant at line {index}: {constant}")
            ),
        )
        if not isinstance(value, dict):
            raise AssertionError(f"raw row {index} is not an object")
        rows.append(value)
    return rows

def validate_evaluator_payload(
    summary: dict[str, Any],
    raw_path: Path,
    *,
    manifest: dict[str, Any],
    repeats: int,
    program: str,
) -> tuple[RunOutcome, dict[str, Any]]:
    """Execute the production schema, typed, and semantic checks in memory."""

    if raw_path.is_symlink() or not raw_path.is_file():
        raise AssertionError("raw artifact is absent or not regular")
    schema = load_json_strict(SCHEMA_PATH)
    execute_schema(summary, schema)
    outcome = RunOutcome.from_mapping(summary)
    if outcome.raw_log != raw_path.relative_to(PROJECT_ROOT).as_posix():
        raise AssertionError("typed outcome raw path mismatch")
    if outcome.raw_log_sha256 != sha256_file(raw_path):
        raise AssertionError("typed outcome raw hash mismatch")
    keys = manifest.get("query_keys")
    if not isinstance(keys, list) or not keys:
        raise AssertionError("semantic manifest has no query keys")
    queries, gt = load_charxiv_data(keys)
    image_entries: dict[str, dict[str, object]] = {}
    for query in queries.values():
        path = Path(query["figure_path"])
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        image_entries[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    validate_generation_configuration(summary, program)
    metrics = validate_raw_and_records(
        summary,
        load_jsonl(raw_path),
        gt=gt,
        queries=queries,
        manifest_keys=keys,
        samples=len(keys),
        repeats=repeats,
        program=program,
        closure={"image_entries": image_entries},
    )
    return outcome, metrics


def current_closure(freeze: dict[str, Any]) -> dict[str, Any]:
    assert_file_hash(GT_PATH, freeze["dataset"]["ground_truth_sha256"], "ground truth")
    assert_file_hash(IMAGE_ZIP_PATH, freeze["dataset"]["image_zip_sha256"], "image ZIP")
    image_document = image_manifest_document()
    image_manifest_hash = verify_manifest_file(IMAGE_MANIFEST_PATH, image_document)
    if image_manifest_hash != freeze["dataset"]["image_manifest_sha256"]:
        raise AssertionError("current image manifest differs from freeze")
    image_entries = {
        item["relative_path"]: item for item in image_document["entries"]
    }

    models = {}
    for model_id, spec in MODEL_SPECS.items():
        actual = model_manifest_document(model_id)
        frozen = freeze["models"][model_id]
        manifest_path = PROJECT_ROOT / frozen["manifest_path"]
        manifest_hash = verify_manifest_file(manifest_path, actual)
        if manifest_hash != frozen["manifest_sha256"]:
            raise AssertionError(f"model manifest changed: {model_id}")
        entries = {item["relative_path"]: item for item in actual["entries"]}
        required = {name: entries[name]["sha256"] for name in REQUIRED_MODEL_FILES}
        if required != frozen["required_file_sha256"]:
            raise AssertionError(f"required model files changed: {model_id}")
        models[model_id] = {
            "manifest_hash": manifest_hash,
            "required": required,
            "entries": entries,
        }

    evaluator = evaluation_dependency_closure()
    if evaluator != freeze["evaluator_dependency_sha256"]:
        raise AssertionError("current evaluator closure differs from freeze")
    program_closures = {}
    for program in MODEL_BY_PROGRAM:
        closure = local_dependency_closure(PROJECT_ROOT / f"{program}.py")
        program_closures[program] = closure
        if closure != freeze["program_dependency_sha256"][program]:
            raise AssertionError(f"current program closure differs: {program}")
    literal_audit = active_literal_audit_document(evaluator, program_closures)
    if literal_audit != freeze["active_literal_audit"]:
        raise AssertionError("active version literal audit differs from freeze")
    assert_file_hash(PROTOCOL_PATH, freeze["protocol_sha256"], "protocol")
    assert_file_hash(LOCK_PATH, freeze["requirements_lock_sha256"], "requirements lock")
    assert_file_hash(SCHEMA_PATH, freeze["result_schema_sha256"], "result schema")
    for manifest in freeze["manifests"].values():
        assert_file_hash(PROJECT_ROOT / manifest["path"], manifest["sha256"], "sample manifest")
    return {
        "image_entries": image_entries,
        "models": models,
        "evaluator": evaluator,
        "freeze_file_sha256": sha256_file(FREEZE_PATH),
    }


def expected_answer(gt: dict[str, Any], query_key: str) -> str:
    figure_id, subq_index = query_key.rsplit("_", 1)
    value = gt[figure_id]["answers"][int(subq_index)]
    if not isinstance(value, str):
        raise AssertionError("current ground-truth answer is not a string")
    return value


def validate_generation_configuration(summary: dict[str, Any], program: str) -> None:
    metadata = summary["program_metadata"]
    model_id = MODEL_BY_PROGRAM[program]
    spec = MODEL_SPECS[model_id]
    if metadata["MODEL_ID"] != model_id:
        raise AssertionError("model ID mismatch")
    if metadata["MODEL_REVISION"] != spec["revision"]:
        raise AssertionError("model revision mismatch")
    if metadata["PROCESSOR_REVISION"] != spec["revision"]:
        raise AssertionError("processor revision mismatch")
    if metadata["DTYPE"] != "bfloat16" or metadata["QUANTIZATION"] != "none":
        raise AssertionError("dtype or quantization changed")
    if metadata["ATTENTION_IMPLEMENTATION"] != "sdpa":
        raise AssertionError("attention implementation changed")
    if metadata["MAX_NEW_TOKENS"] != 128 or metadata["BATCH_SIZE"] != 1:
        raise AssertionError("generation length or batch size changed")
    expected = {
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "min_p": None,
        "typical_p": None,
        "max_new_tokens": 128,
        "use_cache": True,
        "use_model_defaults": False,
    }
    if metadata["EFFECTIVE_GENERATION_CONFIG"] != expected:
        raise AssertionError("effective generation config is not frozen greedy")


def validate_provenance(
    summary: dict[str, Any],
    freeze: dict[str, Any],
    closure: dict[str, Any],
    program: str,
    manifest_name: str,
    raw_path: Path,
) -> None:
    provenance = summary["provenance"]
    model_id = MODEL_BY_PROGRAM[program]
    frozen_model = freeze["models"][model_id]
    expected_pairs = {
        "freeze_id": freeze["freeze_id"],
        "freeze_file_sha256": closure["freeze_file_sha256"],
        **canonical_provenance_paths(model_id),
        "ground_truth_sha256": freeze["dataset"]["ground_truth_sha256"],
        "image_zip_sha256": freeze["dataset"]["image_zip_sha256"],
        "image_manifest_path": freeze["dataset"]["image_manifest_path"],
        "image_manifest_sha256": freeze["dataset"]["image_manifest_sha256"],
        "selected_model_id": model_id,
        "selected_model_revision": frozen_model["revision"],
        "model_manifest_path": frozen_model["manifest_path"],
        "model_manifest_sha256": frozen_model["manifest_sha256"],
        "manifest_sha256": freeze["manifests"][manifest_name]["sha256"],
        "warmup_manifest_sha256": freeze["manifests"]["warmup_1"]["sha256"],
        "protocol_sha256": freeze["protocol_sha256"],
        "requirements_lock_sha256": freeze["requirements_lock_sha256"],
        "result_schema_sha256": freeze["result_schema_sha256"],
        "raw_log_sha256": sha256_file(raw_path),
        "physical_gpu_index": PHYSICAL_GPU_INDEX,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "default_future_gpu": DEFAULT_FUTURE_GPU,
    }
    for key, expected in expected_pairs.items():
        if provenance[key] != expected:
            raise AssertionError(f"provenance mismatch: {key}")
    if provenance["model_required_file_sha256"] != frozen_model["required_file_sha256"]:
        raise AssertionError("required model-file closure mismatch")
    expected_model_snapshots = {
        current_model_id: {
            "revision": current_frozen["revision"],
            "manifest_path": current_frozen["manifest_path"],
            "manifest_sha256": closure["models"][current_model_id]["manifest_hash"],
            "required_file_sha256": closure["models"][current_model_id]["required"],
        }
        for current_model_id, current_frozen in freeze["models"].items()
    }
    if provenance["model_snapshot_manifests"] != expected_model_snapshots:
        raise AssertionError("two-model snapshot provenance closure mismatch")
    actual_weight_hash = closure["models"][model_id]["required"]["model.safetensors"]
    if summary["program_metadata"]["MODEL_WEIGHT_SHA256"] != actual_weight_hash:
        raise AssertionError("program weight hash is not the current model weight hash")
    if provenance["evaluator_dependency_sha256"] != closure["evaluator"]:
        raise AssertionError("evaluator closure mismatch")
    if provenance["program_dependency_sha256"] != freeze["program_dependency_sha256"][program]:
        raise AssertionError("program closure mismatch")
    if provenance["git_state_at_start"]["commit"] == "":
        raise AssertionError("missing git commit")


def validate_raw_and_records(
    summary: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    *,
    gt: dict[str, Any],
    queries: dict[str, dict[str, Any]],
    manifest_keys: list[str],
    samples: int,
    repeats: int,
    program: str,
    closure: dict[str, Any],
) -> dict[str, Any]:
    if len(summary["records"]) != samples:
        raise AssertionError("summary record count differs from manifest")
    record_keys = [record["query_key"] for record in summary["records"]]
    if len(set(record_keys)) != samples:
        raise AssertionError("summary records contain duplicate query keys")
    if record_keys != manifest_keys:
        raise AssertionError("summary record order/content differs from manifest")
    if len(raw_rows) != samples * repeats + 1:
        raise AssertionError("raw row count mismatch")
    warmup = raw_rows[0]
    warmup_required = {
        "phase", "run_id", "runtime_event_id", "query_key", "status",
        "latency_seconds", "image_sha256", "prediction_sha256",
        "generated_token_count", "termination_reason", "program_revision",
        "model_revision", "error",
    }
    if set(warmup) != warmup_required or warmup["phase"] != "warmup":
        raise AssertionError("warmup row schema mismatch")
    measurement_required = {
        "phase", "run_id", "runtime_event_id", "call_index", "repeat",
        "query_key", "program", "program_revision", "model_id",
        "model_revision", "prediction", "prediction_sha256",
        "expected_answer", "match", "latency_seconds", "image_sha256",
        "generated_token_count", "termination_reason", "error",
    }
    measurements = raw_rows[1:]
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in manifest_keys}
    latencies: list[float] = []
    tokens: list[int] = []
    terminations = {"eos": 0, "length": 0, "other": 0}
    expected_sequence = manifest_keys * repeats
    for index, (row, expected_key) in enumerate(zip(measurements, expected_sequence), 1):
        if set(row) != measurement_required or row["phase"] != "measurement":
            raise AssertionError(f"measurement row schema mismatch at {index}")
        if row["call_index"] != index or row["query_key"] != expected_key:
            raise AssertionError("raw order or call index mismatch")
        if row["repeat"] != (index - 1) // samples:
            raise AssertionError("repeat index mismatch")
        if row["run_id"] != summary["run_id"] or row["runtime_event_id"] != summary["runtime_event_id"]:
            raise AssertionError("run/runtime event ID mismatch")
        if row["program"] != program or row["model_id"] != MODEL_BY_PROGRAM[program]:
            raise AssertionError("raw program/model mismatch")
        if row["model_revision"] != summary["program_metadata"]["MODEL_REVISION"]:
            raise AssertionError("raw model revision mismatch")
        expected_program_revision = summary["provenance"][
            "program_dependency_sha256"
        ][f"{program}.py"]
        if row["program_revision"] != expected_program_revision:
            raise AssertionError("raw program revision mismatch")
        if row["error"] is not None:
            raise AssertionError("baseline raw row contains an error")
        latency = row["latency_seconds"]
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
        ):
            raise AssertionError("invalid raw latency")
        prediction = row["prediction"]
        if not isinstance(prediction, str):
            raise AssertionError("successful prediction must be a string")
        if row["prediction_sha256"] != sha256_bytes(prediction.encode("utf-8")):
            raise AssertionError("prediction hash mismatch")
        count = row["generated_token_count"]
        reason = row["termination_reason"]
        if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= 128:
            raise AssertionError("generated token count is invalid")
        if reason not in terminations:
            raise AssertionError("termination reason is invalid")
        if reason == "length" and count != 128:
            raise AssertionError("length termination count mismatch")
        current_expected = expected_answer(gt, expected_key)
        if row["expected_answer"] != current_expected:
            raise AssertionError("raw embedded answer differs from current GT")
        recomputed_match = prediction.strip().lower() == current_expected.lower()
        if row["match"] is not recomputed_match:
            raise AssertionError("exact match differs when recomputed from current GT")
        query = queries[expected_key]
        image_relative = f"charxiv/images/{query['figure_id']}.jpg"
        current_image = closure["image_entries"][image_relative]
        if row["image_sha256"] != current_image["sha256"]:
            raise AssertionError("actual input image hash mismatch")
        grouped[expected_key].append(row)
        latencies.append(latency)
        tokens.append(count)
        terminations[reason] += 1

    correct = 0
    nondeterministic: list[str] = []
    expected_images: dict[str, str] = {}
    for ordinal, (key, aggregate) in enumerate(zip(manifest_keys, summary["records"])):
        rows = grouped[key]
        if len(rows) != repeats:
            raise AssertionError("aggregate repeat count mismatch")
        query = queries[key]
        current_expected = expected_answer(gt, key)
        derived = {
            "query_key": key,
            "figure_id": str(query["figure_id"]),
            "subq_idx": int(query["subq_idx"]),
            "qid": str(query["qid"]),
            "question": query["question"],
            "expected_answer": current_expected,
        }
        for field, value in derived.items():
            if aggregate[field] != value:
                raise AssertionError(f"record {ordinal} derived field mismatch: {field}")
        responses = [row["prediction"] for row in rows]
        expected_arrays = {
            "responses": responses,
            "prediction_sha256": [row["prediction_sha256"] for row in rows],
            "latency_seconds": [row["latency_seconds"] for row in rows],
            "generated_token_count": [row["generated_token_count"] for row in rows],
            "termination_reason": [row["termination_reason"] for row in rows],
        }
        for field, value in expected_arrays.items():
            if aggregate[field] != value:
                raise AssertionError(f"record/raw aggregate mismatch: {field}")
        if aggregate["errors"] != []:
            raise AssertionError("baseline record errors differ from raw")
        if aggregate["image_sha256"] != rows[0]["image_sha256"]:
            raise AssertionError("aggregate image hash differs from raw")
        is_nondeterministic = len(set(responses)) > 1
        if is_nondeterministic:
            nondeterministic.append(key)
        recomputed = responses[0].strip().lower() == current_expected.lower()
        if aggregate["correct"] is not recomputed:
            raise AssertionError("aggregate correctness differs from current GT")
        correct += int(recomputed)
        expected_images[key] = rows[0]["image_sha256"]

    if summary["correct"] != correct or summary["num_evaluated"] != samples:
        raise AssertionError("summary count mismatch")
    if summary["num_calls"] != samples * repeats or summary["repeats"] != repeats:
        raise AssertionError("summary call/repeat mismatch")
    if summary["num_errors"] != 0 or summary["errors"] != [] or summary["error_rate"] != 0.0:
        raise AssertionError("baseline summary contains errors")
    if summary["nondeterministic_query_keys"] != nondeterministic:
        raise AssertionError("nondeterminism summary mismatch")
    expected_status = "pass" if not nondeterministic else "fail"
    if summary["status"] != expected_status:
        raise AssertionError("summary status mismatch")
    if summary["accuracy"] != correct / samples:
        raise AssertionError("accuracy mismatch")
    latency = latency_statistics(latencies)
    if summary["measured_call_seconds"] != latency["measured_call_seconds"]:
        raise AssertionError("measured seconds mismatch")
    if summary["avg_time_per_query"] != latency["mean"]:
        raise AssertionError("mean latency mismatch")
    expected_latency = {
        key: latency[key] for key in ("mean", "median", "p50", "p95", "min", "max")
    }
    if summary["latency_seconds"] != expected_latency:
        raise AssertionError("latency statistics mismatch")
    if summary["throughput_queries_per_second"] != latency["throughput_queries_per_second"]:
        raise AssertionError("throughput mismatch")
    if summary["total_time"] != summary["measurement_wall_seconds"]:
        raise AssertionError("wall total mismatch")
    if summary["wall_throughput_queries_per_second"] != (
        len(latencies) / summary["measurement_wall_seconds"]
    ):
        raise AssertionError("wall throughput mismatch")
    generation = summary["generation_summary"]
    if generation["total_generated_tokens"] != sum(tokens):
        raise AssertionError("generated token total mismatch")
    expected_token_mean = statistics.fmean(tokens) if tokens else 0.0
    if generation["mean_generated_tokens"] != expected_token_mean:
        raise AssertionError("mean generated tokens mismatch")
    if generation["termination_counts"] != terminations:
        raise AssertionError("termination summary mismatch")
    if summary["provenance"]["input_image_sha256"] != expected_images:
        raise AssertionError("input image provenance does not equal full manifest sample set")
    return {
        "accuracy": summary["accuracy"],
        "correct": correct,
        "num_evaluated": samples,
        "num_calls": samples * repeats,
        "avg_time_per_query": summary["avg_time_per_query"],
        "latency_seconds": expected_latency,
        "measured_call_seconds": latency["measured_call_seconds"],
        "throughput_queries_per_second": summary["throughput_queries_per_second"],
        "total_generated_tokens": sum(tokens),
        "mean_generated_tokens": expected_token_mean,
        "termination_counts": terminations,
        "peak_gpu_memory_allocated_bytes": summary[
            "torch_peak_memory_allocated_bytes"
        ],
        "peak_gpu_memory_reserved_bytes": summary[
            "torch_peak_memory_reserved_bytes"
        ],
    }


def validate_run(
    campaign: Path,
    plan_run: dict[str, Any],
    expected: tuple[str, str, str, int, int],
    freeze: dict[str, Any],
    closure: dict[str, Any],
    gt: dict[str, Any],
    schema: dict[str, Any],
    campaign_started_epoch_ns: int,
) -> dict[str, Any]:
    filename, program, manifest_name, samples, repeats = expected
    if (
        plan_run["output_filename"] != filename
        or plan_run["program"] != program
        or plan_run["manifest_name"] != manifest_name
        or plan_run["repeats"] != repeats
    ):
        raise AssertionError("campaign plan semantics mismatch")
    assert_uuid4(plan_run["run_id"], "planned run ID")
    assert_uuid4(plan_run["runtime_event_id"], "planned runtime event ID")
    summary_path = campaign / filename
    if summary_path.is_symlink() or not summary_path.is_file():
        raise AssertionError("planned summary is absent or not regular")
    if summary_path.stat().st_mtime_ns < campaign_started_epoch_ns:
        raise AssertionError("summary predates campaign start")
    summary = load_json_strict(summary_path)
    execute_schema(summary, schema)
    if summary["run_id"] != plan_run["run_id"] or summary["program"] != program:
        raise AssertionError("summary does not belong to planned run")
    if summary["runtime_event_id"] != plan_run["runtime_event_id"]:
        raise AssertionError("runtime event ID differs from the predeclared plan")
    assert_uuid4(summary["runtime_event_id"], "runtime event ID")
    validate_generation_configuration(summary, program)
    paths = canonical_run_paths(
        campaign.name,
        program,
        manifest_name,
        repeats,
        plan_run["run_id"],
        plan_run["runtime_event_id"],
        plan_run["order"],
    )
    plan_path_fields = {
        key: paths[key]
        for key in (
            "output_filename",
            "output_path",
            "runtime_event_path",
            "console_path",
            "start_event_path",
            "end_event_path",
        )
    }
    for key, value in plan_path_fields.items():
        if plan_run[key] != value:
            raise AssertionError(f"campaign plan path differs from builder: {key}")
    if summary["manifest"] != paths["manifest"]:
        raise AssertionError("summary manifest path mismatch")
    if summary["warmup_manifest"] != paths["warmup_manifest"]:
        raise AssertionError("summary warmup path mismatch")

    raw_path = PROJECT_ROOT / summary["raw_log"]
    if summary["raw_log"] != paths["raw_log"] or raw_path != summary_path.with_suffix(".raw.jsonl"):
        raise AssertionError("raw path is not canonical for summary")
    if raw_path.is_symlink() or not raw_path.is_file():
        raise AssertionError("raw log is absent or not regular")
    if raw_path.stat().st_mtime_ns < campaign_started_epoch_ns:
        raise AssertionError("raw log predates campaign start")
    validate_provenance(summary, freeze, closure, program, manifest_name, raw_path)

    provenance = summary["provenance"]
    event_path = PROJECT_ROOT / provenance["runtime_event_path"]
    if (
        provenance["runtime_event_path"] != paths["runtime_event_path"]
        or event_path.parent.resolve(strict=True) != campaign
        or event_path.name != f"runtime_event.{summary['runtime_event_id']}.json"
        or event_path.is_symlink()
        or not event_path.is_file()
    ):
        raise AssertionError("runtime event artifact path is not canonical")
    if event_path.stat().st_mtime_ns < campaign_started_epoch_ns:
        raise AssertionError("runtime event predates campaign")
    if sha256_file(event_path) != provenance["runtime_event_sha256"]:
        raise AssertionError("runtime event artifact hash mismatch")
    event = load_json_strict(event_path)
    model_id = MODEL_BY_PROGRAM[program]
    expected_event = {
        "runtime_event_id": summary["runtime_event_id"],
        "run_id": summary["run_id"],
        "model_id": model_id,
        "model_revision": MODEL_SPECS[model_id]["revision"],
        "processor_revision": MODEL_SPECS[model_id]["revision"],
        "model_weight_sha256": closure["models"][model_id]["required"]["model.safetensors"],
        "dtype": "bfloat16",
        "quantization": "none",
        "attention_implementation": "sdpa",
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "torch_cuda_device_count": VISIBLE_DEVICE_COUNT,
        "logical_device": LOGICAL_DEVICE,
        "physical_gpu_index": PHYSICAL_GPU_INDEX,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "default_future_gpu": DEFAULT_FUTURE_GPU,
    }
    for key, expected_value in expected_event.items():
        if event.get(key) != expected_value:
            raise AssertionError(f"runtime event mismatch: {key}")
    if event.get("effective_generation_config") != summary["program_metadata"][
        "EFFECTIVE_GENERATION_CONFIG"
    ]:
        raise AssertionError("runtime generation configuration mismatch")

    manifest_path = Path(f"{MANIFEST_ROOT}/{manifest_name}.json")
    manifest, manifest_sha = load_manifest(manifest_path)
    if manifest_sha != freeze["manifests"][manifest_name]["sha256"]:
        raise AssertionError("active manifest differs from freeze")
    if len(manifest["query_keys"]) != samples:
        raise AssertionError("manifest sample count differs from run definition")
    queries, reloaded_gt = load_charxiv_data(manifest["query_keys"])
    if reloaded_gt != gt:
        raise AssertionError("ground truth changed within validation")
    metrics = validate_raw_and_records(
        summary,
        load_jsonl(raw_path),
        gt=gt,
        queries=queries,
        manifest_keys=manifest["query_keys"],
        samples=samples,
        repeats=repeats,
        program=program,
        closure=closure,
    )
    return {
        "status": "pass",
        "run_id": summary["run_id"],
        "runtime_event_id": summary["runtime_event_id"],
        "runtime_event_path": event_path.relative_to(PROJECT_ROOT).as_posix(),
        "runtime_event_sha256": sha256_file(event_path),
        "program": program,
        "manifest": manifest_name,
        "repeats": repeats,
        "summary_path": summary_path.relative_to(PROJECT_ROOT).as_posix(),
        "summary_mtime_ns": summary_path.stat().st_mtime_ns,
        "summary_bytes": summary_path.stat().st_size,
        "summary_sha256": sha256_file(summary_path),
        "raw_path": raw_path.relative_to(PROJECT_ROOT).as_posix(),
        "raw_mtime_ns": raw_path.stat().st_mtime_ns,
        "raw_bytes": raw_path.stat().st_size,
        "raw_sha256": sha256_file(raw_path),
        "metrics": metrics,
    }


def resolve_campaign(value: Path | None) -> Path:
    if value is None:
        latest = load_json_strict(LATEST)
        value = PROJECT_ROOT / latest["campaign_path"]
    campaign = value if value.is_absolute() else PROJECT_ROOT / value
    campaign = campaign.resolve()
    root = (PROJECT_ROOT / CAMPAIGN_ROOT_RELATIVE).resolve()
    if not campaign.is_relative_to(root) or campaign.is_symlink() or not campaign.is_dir():
        raise ValueError("campaign directory is outside the fixed active root")
    return campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-path", type=Path)
    parser.add_argument("--probe-expected-sha256")
    parser.add_argument("--probe-label", default="negative probe")
    arguments = parser.parse_args()

    if arguments.probe_path is not None:
        if arguments.probe_expected_sha256 is None:
            raise ValueError("--probe-expected-sha256 is required with --probe-path")
        probe = arguments.probe_path
        probe = probe if probe.is_absolute() else PROJECT_ROOT / probe
        resolved = probe.resolve()
        if (
            not resolved.is_relative_to(EVO_ROOT)
            or probe.is_symlink()
            or not probe.is_file()
        ):
            raise ValueError("probe must be a regular Evo-internal file")
        try:
            assert_file_hash(
                probe,
                arguments.probe_expected_sha256,
                arguments.probe_label,
            )
        except AssertionError as exc:
            print(
                json.dumps(
                    {"probe": arguments.probe_label, "detected": True, "error": str(exc)}
                )
            )
            return 2
        print(json.dumps({"probe": arguments.probe_label, "detected": False}))
        return 0

    campaign = resolve_campaign(arguments.campaign_dir)
    plan_path = arguments.plan or campaign / "campaign_plan.json"
    if arguments.output is not None:
        requested_output = (
            arguments.output
            if arguments.output.is_absolute()
            else PROJECT_ROOT / arguments.output
        )
        if requested_output.exists() or requested_output.is_symlink():
            raise FileExistsError(
                "immutable validation target already exists before validation"
            )
    plan_path = plan_path if plan_path.is_absolute() else PROJECT_ROOT / plan_path
    if plan_path.resolve().parent != campaign:
        raise ValueError("plan must be inside campaign directory")
    plan = load_json_strict(plan_path)
    if plan["mode"] != "regenerate_all_six_once" or len(plan["runs"]) != 6:
        raise AssertionError("plan is not a six-run regeneration campaign")
    if plan.get("execution_eligibility") != "gpu_production":
        raise AssertionError("campaign was not created by a GPU-production-eligible engine")
    if len({run["run_id"] for run in plan["runs"]}) != 6:
        raise AssertionError("planned run IDs are not unique")
    expected_gpu_execution = {
        "physical_gpu_index": PHYSICAL_GPU_INDEX,
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "logical_device": LOGICAL_DEVICE,
        "visible_device_count": VISIBLE_DEVICE_COUNT,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "default_future_gpu": DEFAULT_FUTURE_GPU,
    }
    if plan.get("gpu_execution") != expected_gpu_execution:
        raise AssertionError("campaign GPU authorization metadata mismatch")
    for key in (
        "physical_gpu_index",
        "authorization_scope",
        "default_future_gpu",
    ):
        if plan.get(key) != expected_gpu_execution[key]:
            raise AssertionError(f"campaign GPU authorization field mismatch: {key}")
    preflight = plan.get("launcher_torch_preflight")
    if not isinstance(preflight, dict):
        raise AssertionError("campaign lacks launcher torch preflight")
    if preflight.get("torch_cuda_device_count") != VISIBLE_DEVICE_COUNT:
        raise AssertionError("launcher torch preflight device count mismatch")
    if preflight.get("logical_device") != LOGICAL_DEVICE:
        raise AssertionError("launcher torch preflight logical device mismatch")

    freeze = load_freeze()
    closure = current_closure(freeze)
    plan_sha = sha256_file(plan_path)
    if plan.get("freeze_id") != freeze["freeze_id"]:
        raise AssertionError("plan freeze ID mismatch")
    if plan.get("freeze_file_sha256") != closure["freeze_file_sha256"]:
        raise AssertionError("plan freeze file hash mismatch")
    schema = load_json_strict(SCHEMA_PATH)
    gt = load_json_strict(GT_PATH)
    files = [
        validate_run(
            campaign,
            plan_run,
            expected,
            freeze,
            closure,
            gt,
            schema,
            plan["campaign_started_epoch_ns"],
        )
        for plan_run, expected in zip(plan["runs"], EXPECTED_RUNS)
    ]
    if len({item["runtime_event_id"] for item in files}) != 6:
        raise AssertionError("runtime event IDs are not unique")

    if arguments.output is None:
        checks = campaign / "checks"
        checks.mkdir(mode=0o700, exist_ok=True)
        if checks.is_symlink() or not checks.is_dir():
            raise ValueError("checks directory is unsafe")
        output = checks / f"check.{uuid4()}.json"
    else:
        output = (
            arguments.output
            if arguments.output.is_absolute()
            else PROJECT_ROOT / arguments.output
        )
    output_parent = output.parent.resolve(strict=True)
    if output_parent != campaign and output_parent.parent != campaign:
        raise ValueError("validation output must be canonical or a campaign check")
    evidence = {
        "schema_version": 4,
        "status": "pass",
        "protocol_version": PROTOCOL_VERSION,
        "freeze_id": freeze["freeze_id"],
        "freeze_file_sha256": closure["freeze_file_sha256"],
        "campaign_id": plan["campaign_id"],
        "campaign_started_at_utc": plan["campaign_started_at_utc"],
        "campaign_started_epoch_ns": plan["campaign_started_epoch_ns"],
        "plan_path": plan_path.relative_to(PROJECT_ROOT).as_posix(),
        "plan_sha256": plan_sha,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_executed": True,
        "ground_truth_reloaded_for_exact_match": True,
        "current_files_rehashed": True,
        "physical_gpu_index": PHYSICAL_GPU_INDEX,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "default_future_gpu": DEFAULT_FUTURE_GPU,
        "files": files,
    }
    typed_before = ValidationOutcome.from_mapping(evidence)
    write_json_atomic(output, evidence)
    persisted = load_json_strict(output)
    typed_after = ValidationOutcome.from_mapping(persisted)
    if persisted != evidence or typed_after != typed_before:
        raise AssertionError("validation changed across no-replace persistence")
    print(json.dumps(persisted, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
