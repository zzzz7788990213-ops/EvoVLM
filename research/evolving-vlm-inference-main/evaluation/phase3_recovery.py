"""Phase 3 Recovery-v2: preregistration, atomic journals, resume, and selection."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

from evaluate import import_program, verify_greedy
from evolution.process_safety import EVO_ROOT, ensure_internal_directory
from evaluation.atomic_artifacts import (
    finalize_stream,
    open_text_exclusive,
    write_json_noreplace,
)
from evaluation.phase3_manifests import (
    MANIFEST_ROOT,
    load_manifest,
    load_private_queries,
    sha256_file,
)
from evaluation.phase3_runner import (
    MODEL_META,
    PROGRAM_CONFIG,
    PROGRAM_ORDER,
    validate_measurement_mapping,
)


PROJECT_ROOT = EVO_ROOT / "workspace/evolving-vlm-inference-main"
RECOVERY_ROOT = PROJECT_ROOT / "results/phase3_recovery_v2"
CAMPAIGN_ROOT = RECOVERY_ROOT / "campaigns"
RUNTIME_EVENT_ROOT = PROJECT_ROOT / "results/phase3/runtime_events"
PREREG_ROOT = PROJECT_ROOT / "evaluation/recovery_v2"
PROTOCOL_PATH = PREREG_ROOT / "protocol.json"
CANDIDATE_MANIFEST_PATH = PREREG_ROOT / "candidate_manifest.json"
DATA_MANIFEST_PATH = PREREG_ROOT / "data_manifest.json"
SELECTION_RULES_PATH = PREREG_ROOT / "selection_rules.json"
AUTHORIZATION_PATH = PREREG_ROOT / "recovery_authorization.json"
VENV_PYTHON = EVO_ROOT / ".venv/bin/python"

PROTOCOL_ID = "phase3-recovery-v2"
RESULT_VERSION = "phase3-recovery-program-result-v2"
PLAN_VERSION = "phase3-recovery-campaign-plan-v2"
SUMMARY_VERSION = "phase3-recovery-summary-v2"
COMPLETED_VERSION = "phase3-recovery-completed-v2"
JOURNAL_VERSION = "phase3-recovery-query-journal-v2"
ATTEMPT_VERSION = "phase3-recovery-attempt-v2"
PROGRAM_BUDGET_SECONDS = 14_400.0
BOOTSTRAP_SEED = 20260825
BOOTSTRAP_REPLICATES = 10_000

OPTIMIZED_PROGRAMS = (
    "manual_instruct",
    "manual_thinking",
    "evolved_instruct",
    "evolved_thinking",
)
FORMAL_PROGRAMS = (
    "baseline_instruct",
    *OPTIMIZED_PROGRAMS,
)
TERMINATION_REASONS = ("eos", "length", "other", "stop_text")
JOURNAL_NAME = re.compile(r"^(?P<index>[0-9]{3})\.json$")

V1_CAMPAIGN_ID = "6beb857c-ef2c-4e0a-a6e6-34abee7f4a13"
V1_INCIDENT_RELATIVE = (
    "results/phase3/campaigns/"
    + V1_CAMPAIGN_ID
    + "/incident.08.baseline_thinking.dev128.timeout_7200s.1.json"
)
V1_INCIDENT_SHA256 = (
    "30d2490971fe06f386ecd707da0e10ec03a8ea104c925dd45a8fca148dfef4c7"
)
V1_PLAN_SHA256 = (
    "effe1b6f06a78a3544444248667f9e5c5f941633424299475121cb329f526f567"
)

RECOVERY_CLOSURE = (
    "evaluate.py",
    "evaluation/phase3_recovery.py",
    "evaluation/phase3_manifests.py",
    "evaluation/atomic_artifacts.py",
    "evaluation/gpu_scope.py",
    "charxiv/src/descriptive_utils.py",
    "requirements.lock.txt",
)


class RecoveryError(RuntimeError):
    """Fail-closed Recovery-v2 error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def percentile_type7(values: Sequence[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise RecoveryError("invalid percentile input")
    ordered = sorted(float(item) for item in values)
    if not all(math.isfinite(item) for item in ordered):
        raise RecoveryError("percentile input is nonfinite")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    fraction = position - lower
    if lower + 1 == len(ordered):
        return ordered[lower]
    return ordered[lower] + fraction * (ordered[lower + 1] - ordered[lower])


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise RecoveryError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def load_json(path: Path, *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RecoveryError(f"missing physical JSON artifact: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise RecoveryError(f"JSON artifact escaped root: {path}")
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_strict_pairs,
                parse_constant=lambda raw: (_ for _ in ()).throw(
                    RecoveryError(f"nonfinite JSON constant: {raw}")
                ),
            )
    except json.JSONDecodeError as exc:
        raise RecoveryError(f"malformed JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"JSON artifact is not an object: {path}")
    return value


def expect_equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise RecoveryError(f"{name} differs from frozen value")


def file_record(relative: str) -> dict[str, object]:
    path = PROJECT_ROOT / relative
    if path.is_symlink() or not path.is_file():
        raise RecoveryError(f"unsafe runtime closure file: {relative}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise RecoveryError(f"runtime closure escaped project: {relative}")
    return {
        "path": relative,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def recovery_schedule() -> list[dict[str, object]]:
    schedule: list[dict[str, object]] = []
    for program in PROGRAM_ORDER:
        schedule.append({"program": program, "manifest": "single1", "repeat": 1})
    for program in FORMAL_PROGRAMS:
        schedule.append({"program": program, "manifest": "dev128", "repeat": 1})
    formal = tuple(FORMAL_PROGRAMS)
    for repeat, offset in ((1, 0), (2, 2), (3, 4)):
        order = formal[offset:] + formal[:offset]
        for program in order:
            schedule.append(
                {"program": program, "manifest": "speed32", "repeat": repeat}
            )
    return schedule


def protocol_document() -> dict[str, object]:
    return {
        "protocol_id": PROTOCOL_ID,
        "recovery_of": {
            "campaign_id": V1_CAMPAIGN_ID,
            "campaign_plan_sha256": V1_PLAN_SHA256,
            "baseline_thinking_dev128": {
                "status": "TIMEOUT_CENSORED",
                "timeout_seconds": 7200,
                "accuracy_available": False,
                "partial_predictions_available": False,
                "incident_path": V1_INCIDENT_RELATIVE,
                "incident_sha256": V1_INCIDENT_SHA256,
            },
            "v1_results_must_not_be_mixed": True,
        },
        "schedule": recovery_schedule(),
        "physical_gpu_index": 0,
        "cuda_visible_devices": "0",
        "visible_device_count": 1,
        "logical_device": "cuda:0",
        "program_budget_seconds": int(PROGRAM_BUDGET_SECONDS),
        "infrastructure_restart": {
            "allowed_before_first_measured_query": 1,
            "configuration_must_be_identical": True,
            "failure_evidence_retained": True,
        },
        "journal": {
            "one_atomic_noreplace_file_per_query": True,
            "commit_outside_latency_interval": True,
            "resume_revalidates_and_skips_completed_queries": True,
            "gaps_duplicates_extra_and_partial_files_rejected": True,
        },
        "accuracy": {
            "manifest": "dev128",
            "definition": "prediction.strip().casefold()==expected.strip().casefold()",
            "errors_remain_in_denominator": True,
            "repeats": 1,
        },
        "speed": {
            "manifest": "speed32",
            "repeats": 3,
            "programs": list(FORMAL_PROGRAMS),
            "selection": "median per query across three repeats, then aggregate 32 queries",
            "fixed_rotation": True,
            "warmup_identical": True,
            "load_excluded": True,
            "no_repeat_discarding": True,
        },
        "generation": {
            "greedy": True,
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "batch_size": 1,
            "dtype": "bfloat16",
            "attention": "sdpa",
            "weights_unchanged": True,
        },
        "candidate_timeout": {
            "status": "TIMEOUT",
            "best_eligible": False,
            "continue_other_programs": True,
        },
        "v1_source_results_are_historical_only": True,
    }


def candidate_manifest_document() -> dict[str, object]:
    programs: list[dict[str, object]] = []
    for name in PROGRAM_ORDER:
        kind, family, max_new_tokens, dependencies = PROGRAM_CONFIG[name]
        model = MODEL_META[family]
        programs.append(
            {
                "program": name,
                "kind": kind,
                "family": family,
                "optimized": name in OPTIMIZED_PROGRAMS,
                "source": file_record(f"{name}.py"),
                "dependencies": [file_record(item) for item in dependencies],
                "model_id": model["model_id"],
                "model_revision": model["revision"],
                "processor_revision": model["revision"],
                "weight_sha256": model["weight_sha256"],
                "dtype": "bfloat16",
                "quantization": "none",
                "attention": "sdpa",
                "batch_size": 1,
                "effective_generation": {
                    "do_sample": False,
                    "num_beams": 1,
                    "num_return_sequences": 1,
                    "max_new_tokens": max_new_tokens,
                },
            }
        )
    closure = [file_record(item) for item in RECOVERY_CLOSURE]
    return {
        "manifest_version": "phase3-recovery-candidate-manifest-v2",
        "program_order": list(PROGRAM_ORDER),
        "formal_programs": list(FORMAL_PROGRAMS),
        "optimized_programs": list(OPTIMIZED_PROGRAMS),
        "programs": programs,
        "runtime_closure": closure,
        "runtime_closure_sha256": canonical_hash(closure),
    }


def data_manifest_document() -> dict[str, object]:
    roles = {}
    for name in ("single1", "dev128", "speed32", "warmup1"):
        document = load_manifest(name)
        roles[name] = {
            "path": f"data/phase3/manifests/{name}.json",
            "sha256": sha256_file(MANIFEST_ROOT / f"{name}.json"),
            "sample_count": document["sample_count"],
            "query_key_order_sha256": document["query_key_order_sha256"],
            "figure_order_sha256": document["figure_order_sha256"],
        }
    return {
        "manifest_version": "phase3-recovery-data-manifest-v2",
        "roles": roles,
        "same_dev128_as_v1": True,
        "same_speed32_as_v1": True,
        "same_ground_truth_and_images": True,
    }


def selection_rules_document() -> dict[str, object]:
    return {
        "rules_version": "phase3-recovery-selection-v2",
        "candidates": list(OPTIMIZED_PROGRAMS),
        "baseline": "baseline_instruct",
        "best_accuracy": [
            "highest Dev128 integer correct_count",
            "fewest Dev128 errors",
            "lowest Speed32 aggregate median",
            "fewest generated tokens",
            "lexicographic source SHA256",
        ],
        "best_speed": {
            "floor_correct": "max(baseline_instruct_correct,best_accuracy_correct-4)",
            "eligibility": "Dev128 complete, errors==0, correct_count>=floor_correct",
            "ties": [
                "lowest Speed32 median",
                "lowest Type-7 p95",
                "fewest generated tokens",
                "highest correct_count",
                "source SHA256",
            ],
        },
        "best_overall": {
            "eligibility": "Dev128 complete, errors==0, correct_count>=baseline_instruct",
            "accuracy_component": "correct_count/max_eligible_correct_count",
            "speed_component": "min_eligible_median/candidate_median",
            "score": "harmonic mean",
            "ties": [
                "higher score",
                "higher correct_count",
                "lower median",
                "fewer tokens",
                "source SHA256",
            ],
            "author_defined": True,
        },
        "baseline_thinking_best_eligible": False,
        "timeout_best_eligible": False,
        "selection_uses_only_recovery_v2_results": True,
    }


def authorization_document() -> dict[str, object]:
    return {
        "authorization_version": "phase3-recovery-authorization-v2",
        "authorized_by": "user",
        "reason": "whole-run timeout and absence of per-query journal",
        "observed_run08_accuracy": False,
        "observed_run08_partial_prediction": False,
        "metric_conditioned_retry": False,
        "candidate_programs_modified": False,
        "model_weights_modified": False,
        "ground_truth_or_answers_modified": False,
        "dev128_modified": False,
        "greedy_or_exact_match_modified": False,
        "v1_campaign_preserved": {
            "campaign_id": V1_CAMPAIGN_ID,
            "incident_sha256": V1_INCIDENT_SHA256,
            "completed_result_count": 7,
        },
        "recovery_changes": [
            "per-query atomic no-replace journal",
            "validated resume from next query",
            "14400-second cumulative per-program budget",
            "candidate timeout does not abort unrelated programs",
            "baseline_thinking Dev128 excluded as TIMEOUT_CENSORED",
        ],
    }


PREREG_DOCUMENTS: tuple[tuple[Path, Callable[[], dict[str, object]]], ...] = (
    (PROTOCOL_PATH, protocol_document),
    (CANDIDATE_MANIFEST_PATH, candidate_manifest_document),
    (DATA_MANIFEST_PATH, data_manifest_document),
    (SELECTION_RULES_PATH, selection_rules_document),
    (AUTHORIZATION_PATH, authorization_document),
)


def prepare_preregistration() -> dict[str, object]:
    ensure_internal_directory(PREREG_ROOT)
    hashes: dict[str, str] = {}
    for path, builder in PREREG_DOCUMENTS:
        expected = builder()
        if path.exists() or path.is_symlink():
            expect_equal(load_json(path), expected, path.name)
        else:
            write_json_noreplace(path, expected)
        hashes[path.name] = sha256_file(path)
    return {"protocol_id": PROTOCOL_ID, "sha256": hashes}


def verify_preregistration() -> dict[str, object]:
    hashes: dict[str, str] = {}
    for path, builder in PREREG_DOCUMENTS:
        expect_equal(load_json(path), builder(), path.name)
        hashes[path.name] = sha256_file(path)
    return {"protocol_id": PROTOCOL_ID, "sha256": hashes}


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(EVO_ROOT), *args],
        cwd=EVO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
        check=False,
    )
    if result.returncode != 0:
        raise RecoveryError("git identity check failed: " + result.stderr[:256])
    return result.stdout.strip()


def source_identity() -> dict[str, str]:
    if run_git("status", "--porcelain=v1", "--untracked-files=no"):
        raise RecoveryError("tracked worktree must be clean")
    raw = run_git("status", "--porcelain=v1", "--untracked-files=all")
    unsafe = []
    for line in raw.splitlines():
        relative = line[3:] if len(line) > 3 else ""
        if Path(relative).suffix.lower() in {".py", ".pyc", ".pyo", ".so", ".pth"}:
            unsafe.append(relative)
    if unsafe:
        raise RecoveryError(f"untracked runtime source forbidden: {unsafe[:12]}")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "tree": run_git("rev-parse", "HEAD^{tree}"),
    }


def program_record(program: str) -> dict[str, Any]:
    for record in candidate_manifest_document()["programs"]:
        if isinstance(record, dict) and record.get("program") == program:
            return record
    raise RecoveryError(f"program is not frozen: {program}")


def preflight_gpu_environment(environ: Mapping[str, str] | None = None) -> None:
    values = os.environ if environ is None else environ
    if values.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RecoveryError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if values.get("EVO_GPU_AUTHORIZATION_SCOPE"):
        raise RecoveryError("retired GPU authorization scope is forbidden")
    if values.get("EVO_ALLOW_GPU1") or values.get("EVO_PHYSICAL_GPU_INDEX") not in {None, "0"}:
        raise RecoveryError("nonzero physical GPU authorization is forbidden")


def _safe_recovery_path(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(RECOVERY_ROOT.resolve(strict=False)):
        raise RecoveryError("recovery artifact escaped recovery root")
    return resolved


def journal_filename(index: int) -> str:
    if index < 0 or index > 999:
        raise RecoveryError("journal index outside supported range")
    return f"{index:03d}.json"


def make_journal_document(
    *,
    campaign_id: str,
    run_id: str,
    attempt_id: str,
    program: str,
    record: Mapping[str, object],
    query: Mapping[str, object],
    query_index: int,
    manifest_name: str,
    manifest_sha256: str,
    manifest_order_sha256: str,
    source: Mapping[str, str],
    frozen_program: Mapping[str, object],
    gpu_scope: Mapping[str, object],
) -> dict[str, object]:
    return {
        "journal_version": JOURNAL_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "program": program,
        "source_identity": dict(source),
        "program_source_sha256": frozen_program["source"]["sha256"],
        "model": {
            "model_id": frozen_program["model_id"],
            "model_revision": frozen_program["model_revision"],
            "processor_revision": frozen_program["processor_revision"],
            "weight_sha256": frozen_program["weight_sha256"],
        },
        "manifest": {
            "name": manifest_name,
            "sha256": manifest_sha256,
            "query_key_order_sha256": manifest_order_sha256,
        },
        "query": {
            "index": query_index,
            "query_key": query["query_key"],
            "figure_id": query["figure_id"],
            "subq_idx": query["subq_idx"],
            "qid": query["qid"],
            "question": query["question"],
            "image_sha256": query["image_sha256"],
        },
        "prediction": record["prediction"],
        "prediction_sha256": record["prediction_sha256"],
        "correct": record["correct"],
        "latency_seconds": record["latency_seconds"],
        "measurement": record["measurement"],
        "error_type": record["error_type"],
        "error_message": record["error_message"],
        "gpu_scope": dict(gpu_scope),
    }


def validate_journal_document(
    document: Mapping[str, object],
    *,
    campaign_id: str,
    run_id: str,
    program: str,
    query: Mapping[str, object],
    query_index: int,
    manifest_name: str,
    manifest_sha256: str,
    manifest_order_sha256: str,
    source: Mapping[str, str],
    frozen_program: Mapping[str, object],
) -> dict[str, object]:
    expect_equal(document.get("journal_version"), JOURNAL_VERSION, "journal version")
    expect_equal(document.get("protocol_id"), PROTOCOL_ID, "journal protocol")
    expect_equal(document.get("campaign_id"), campaign_id, "journal campaign")
    expect_equal(document.get("run_id"), run_id, "journal run")
    try:
        uuid.UUID(str(document.get("attempt_id")))
    except ValueError as exc:
        raise RecoveryError("journal attempt ID is invalid") from exc
    expect_equal(document.get("program"), program, "journal program")
    expect_equal(document.get("source_identity"), dict(source), "journal source")
    expect_equal(
        document.get("program_source_sha256"),
        frozen_program["source"]["sha256"],
        "journal program source hash",
    )
    expect_equal(
        document.get("model"),
        {
            "model_id": frozen_program["model_id"],
            "model_revision": frozen_program["model_revision"],
            "processor_revision": frozen_program["processor_revision"],
            "weight_sha256": frozen_program["weight_sha256"],
        },
        "journal model",
    )
    expect_equal(
        document.get("manifest"),
        {
            "name": manifest_name,
            "sha256": manifest_sha256,
            "query_key_order_sha256": manifest_order_sha256,
        },
        "journal manifest",
    )
    expect_equal(
        document.get("query"),
        {
            "index": query_index,
            "query_key": query["query_key"],
            "figure_id": query["figure_id"],
            "subq_idx": query["subq_idx"],
            "qid": query["qid"],
            "question": query["question"],
            "image_sha256": query["image_sha256"],
        },
        "journal query",
    )
    scope = document.get("gpu_scope")
    if not isinstance(scope, dict):
        raise RecoveryError("journal GPU scope is invalid")
    for key, expected in (
        ("physical_gpu_index", 0),
        ("cuda_visible_devices", "0"),
        ("visible_device_count", 1),
        ("logical_device", "cuda:0"),
    ):
        expect_equal(scope.get(key), expected, f"journal GPU {key}")
    prediction = document.get("prediction")
    prediction_sha256 = document.get("prediction_sha256")
    measurement = document.get("measurement")
    error_type = document.get("error_type")
    error_message = document.get("error_message")
    if prediction is None:
        if prediction_sha256 is not None or measurement is not None:
            raise RecoveryError("failed journal carries output or measurement")
        if not isinstance(error_type, str) or not isinstance(error_message, str):
            raise RecoveryError("failed journal error fields are invalid")
        normalized_measurement = None
    elif isinstance(prediction, str):
        expect_equal(prediction_sha256, text_hash(prediction), "prediction hash")
        if error_type is not None or error_message is not None:
            raise RecoveryError("successful journal carries error fields")
        normalized_measurement = validate_measurement_mapping(measurement)
    else:
        raise RecoveryError("journal prediction is neither text nor null")
    correct = (
        isinstance(prediction, str)
        and prediction.strip().casefold()
        == str(query["expected_answer"]).strip().casefold()
    )
    expect_equal(document.get("correct"), correct, "journal correctness")
    latency = document.get("latency_seconds")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        raise RecoveryError("journal latency is not numeric")
    latency_value = float(latency)
    if not math.isfinite(latency_value) or latency_value < 0:
        raise RecoveryError("journal latency is negative or nonfinite")
    return {
        "query_key": query["query_key"],
        "figure_id": query["figure_id"],
        "subq_idx": query["subq_idx"],
        "qid": query["qid"],
        "question": query["question"],
        "image_path": query["image_path"],
        "image_sha256": query["image_sha256"],
        "prediction": prediction,
        "prediction_sha256": prediction_sha256,
        "correct": correct,
        "latency_seconds": latency_value,
        "measurement": normalized_measurement,
        "error_type": error_type,
        "error_message": error_message,
    }


def load_valid_journals(
    journal_dir: Path,
    *,
    campaign_id: str,
    run_id: str,
    program: str,
    queries: Sequence[Mapping[str, object]],
    manifest_name: str,
    manifest_sha256: str,
    manifest_order_sha256: str,
    source: Mapping[str, str],
    frozen_program: Mapping[str, object],
) -> list[dict[str, object]]:
    _safe_recovery_path(journal_dir)
    if not journal_dir.exists():
        return []
    if journal_dir.is_symlink() or not journal_dir.is_dir():
        raise RecoveryError("journal directory is not physical")
    entries = sorted(journal_dir.iterdir(), key=lambda path: path.name)
    indexes: list[int] = []
    for path in entries:
        match = JOURNAL_NAME.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_file():
            raise RecoveryError(f"non-atomic or unexpected journal artifact: {path.name}")
        indexes.append(int(match.group("index")))
    if indexes != list(range(len(indexes))):
        raise RecoveryError("journal indexes contain a gap, duplicate, or reorder")
    if len(indexes) > len(queries):
        raise RecoveryError("journal count exceeds manifest")
    records: list[dict[str, object]] = []
    for index in indexes:
        document = load_json(journal_dir / journal_filename(index), root=RECOVERY_ROOT)
        records.append(
            validate_journal_document(
                document,
                campaign_id=campaign_id,
                run_id=run_id,
                program=program,
                query=queries[index],
                query_index=index,
                manifest_name=manifest_name,
                manifest_sha256=manifest_sha256,
                manifest_order_sha256=manifest_order_sha256,
                source=source,
                frozen_program=frozen_program,
            )
        )
    return records


def measurement_from_module(module: ModuleType) -> dict[str, object]:
    value = getattr(module.vlm_inference, "last_generation_metadata", None)
    if value is None:
        value = getattr(getattr(module, "_RUNTIME", None), "last_generation_metadata", None)
    return validate_measurement_mapping(value)


def summarize_records(
    *,
    program: str,
    manifest_name: str,
    records: Sequence[Mapping[str, object]],
    warmup_seconds: float,
    warmup_metadata: Mapping[str, object],
) -> dict[str, object]:
    if not records:
        raise RecoveryError("cannot summarize an empty formal record set")
    latencies = [float(record["latency_seconds"]) for record in records]
    measured = sum(latencies)
    if measured <= 0:
        raise RecoveryError("formal measured duration must be positive")
    normalized_warmup = validate_measurement_mapping(warmup_metadata)
    generated = sum(
        int(record["measurement"]["generated_token_count"])
        for record in records
        if record["measurement"] is not None
    )
    terminations = {
        reason: sum(
            1
            for record in records
            if record["measurement"] is not None
            and record["measurement"]["termination_reason"] == reason
        )
        for reason in TERMINATION_REASONS
    }
    return {
        "program": program,
        "manifest": manifest_name,
        "sample_count": len(records),
        "correct_count": sum(int(record["correct"]) for record in records),
        "accuracy": sum(int(record["correct"]) for record in records) / len(records),
        "error_count": sum(record["error_type"] is not None for record in records),
        "load_and_warmup_excluded": True,
        "journal_commit_outside_latency": True,
        "warmup_seconds": float(warmup_seconds),
        "warmup_metadata": normalized_warmup,
        "latency": {
            "measured_seconds": measured,
            "mean_seconds": statistics.fmean(latencies),
            "median_seconds": statistics.median(latencies),
            "min_seconds": min(latencies),
            "max_seconds": max(latencies),
            "type7_p95_seconds": percentile_type7(latencies, 0.95),
            "throughput_qps": len(latencies) / measured,
        },
        "generated_token_total": generated,
        "termination_counts": terminations,
        "peak_memory_bytes": max(
            (
                int(record["measurement"]["peak_memory_bytes"])
                for record in records
                if record["measurement"] is not None
            ),
            default=0,
        ),
        "records": [dict(record) for record in records],
    }


def evaluate_with_journals(
    *,
    module: ModuleType,
    cuda: Any,
    program: str,
    manifest_name: str,
    queries: list[dict[str, object]],
    warmup: dict[str, object],
    journal_dir: Path,
    campaign_id: str,
    run_id: str,
    attempt_id: str,
    source: Mapping[str, str],
    frozen_program: Mapping[str, object],
    gpu_scope: Mapping[str, object],
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    manifest_path = MANIFEST_ROOT / f"{manifest_name}.json"
    manifest = load_manifest(manifest_name)
    manifest_sha = sha256_file(manifest_path)
    order_sha = str(manifest["query_key_order_sha256"])
    ensure_internal_directory(journal_dir)
    records = load_valid_journals(
        journal_dir,
        campaign_id=campaign_id,
        run_id=run_id,
        program=program,
        queries=queries,
        manifest_name=manifest_name,
        manifest_sha256=manifest_sha,
        manifest_order_sha256=order_sha,
        source=source,
        frozen_program=frozen_program,
    )
    cuda.synchronize()
    warm_started = clock()
    warm_prediction = module.vlm_inference(
        str(warmup["image_path"]), str(warmup["question"])
    )
    cuda.synchronize()
    warmup_seconds = clock() - warm_started
    if not isinstance(warm_prediction, str):
        raise RecoveryError("warmup prediction is not text")
    warmup_metadata = measurement_from_module(module)
    for index in range(len(records), len(queries)):
        query = queries[index]
        prediction: str | None = None
        measurement: dict[str, object] | None = None
        error_type: str | None = None
        error_message: str | None = None
        cuda.synchronize()
        started = clock()
        try:
            value = module.vlm_inference(
                str(query["image_path"]), str(query["question"])
            )
            if not isinstance(value, str):
                raise TypeError("vlm_inference returned non-text")
            prediction = value
            cuda.synchronize()
            latency = clock() - started
            measurement = measurement_from_module(module)
        except Exception as exc:
            cuda.synchronize()
            latency = clock() - started
            error_type = type(exc).__name__
            error_message = str(exc)[:1024]
        correct = (
            isinstance(prediction, str)
            and prediction.strip().casefold()
            == str(query["expected_answer"]).strip().casefold()
        )
        record = {
            "query_key": query["query_key"],
            "figure_id": query["figure_id"],
            "subq_idx": query["subq_idx"],
            "qid": query["qid"],
            "question": query["question"],
            "image_path": query["image_path"],
            "image_sha256": query["image_sha256"],
            "prediction": prediction,
            "prediction_sha256": text_hash(prediction) if prediction is not None else None,
            "correct": correct,
            "latency_seconds": latency,
            "measurement": measurement,
            "error_type": error_type,
            "error_message": error_message,
        }
        journal = make_journal_document(
            campaign_id=campaign_id,
            run_id=run_id,
            attempt_id=attempt_id,
            program=program,
            record=record,
            query=query,
            query_index=index,
            manifest_name=manifest_name,
            manifest_sha256=manifest_sha,
            manifest_order_sha256=order_sha,
            source=source,
            frozen_program=frozen_program,
            gpu_scope=gpu_scope,
        )
        # The measured interval ended at the CUDA post-sync above. Durable journal
        # publication is deliberately outside that interval.
        write_json_noreplace(journal_dir / journal_filename(index), journal)
        records.append(record)
    if len(records) != len(queries):
        raise RecoveryError("journaled record count differs from manifest")
    return summarize_records(
        program=program,
        manifest_name=manifest_name,
        records=records,
        warmup_seconds=warmup_seconds,
        warmup_metadata=warmup_metadata,
    )


def campaign_directory(campaign_id: str) -> Path:
    try:
        uuid.UUID(campaign_id)
    except ValueError as exc:
        raise RecoveryError("campaign ID is not a UUID") from exc
    return CAMPAIGN_ROOT / campaign_id


def entry_names(index: int, entry: Mapping[str, object]) -> dict[str, str]:
    stem = (
        f"{index:02d}.{entry['program']}.{entry['manifest']}."
        f"r{entry['repeat']}"
    )
    return {
        "result": stem + ".json",
        "journal_dir": stem + ".journals",
        "console_prefix": stem + ".attempt",
        "attempt_prefix": stem + ".attempt",
    }


def campaign_plan_document(
    campaign_id: str,
    *,
    created_at: str | None = None,
    run_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    try:
        uuid.UUID(campaign_id)
    except ValueError as exc:
        raise RecoveryError("campaign ID is not a UUID") from exc
    prereg = verify_preregistration()
    identity = source_identity()
    schedule = []
    raw_schedule = recovery_schedule()
    selected_ids = list(run_ids) if run_ids is not None else [
        str(uuid.uuid4()) for _ in raw_schedule
    ]
    if len(selected_ids) != len(raw_schedule):
        raise RecoveryError("run ID count differs from schedule")
    for index, (raw, run_id) in enumerate(zip(raw_schedule, selected_ids), start=1):
        try:
            uuid.UUID(run_id)
        except ValueError as exc:
            raise RecoveryError("plan run ID is invalid") from exc
        item = dict(raw)
        item.update({"index": index, "run_id": run_id, **entry_names(index, raw)})
        schedule.append(item)
    return {
        "plan_version": PLAN_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": campaign_id,
        "created_at": created_at or utc_now(),
        "source_identity": identity,
        "preregistration": prereg,
        "physical_gpu_index": 0,
        "cuda_visible_devices": "0",
        "visible_device_count": 1,
        "program_budget_seconds": int(PROGRAM_BUDGET_SECONDS),
        "schedule": schedule,
    }


def create_campaign(campaign_id: str | None = None) -> Path:
    ensure_internal_directory(CAMPAIGN_ROOT)
    selected = campaign_id or str(uuid.uuid4())
    directory = campaign_directory(selected)
    directory.mkdir(mode=0o700)
    write_json_noreplace(directory / "campaign_plan.json", campaign_plan_document(selected))
    return directory


def load_campaign_plan(directory: Path) -> dict[str, Any]:
    _safe_recovery_path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise RecoveryError("campaign directory is not physical")
    plan = load_json(directory / "campaign_plan.json", root=RECOVERY_ROOT)
    expect_equal(plan.get("plan_version"), PLAN_VERSION, "plan version")
    expect_equal(plan.get("protocol_id"), PROTOCOL_ID, "plan protocol")
    expect_equal(plan.get("source_identity"), source_identity(), "plan source")
    expect_equal(plan.get("preregistration"), verify_preregistration(), "plan prereg")
    expect_equal(plan.get("physical_gpu_index"), 0, "plan physical GPU")
    expect_equal(plan.get("cuda_visible_devices"), "0", "plan CUDA scope")
    expect_equal(plan.get("visible_device_count"), 1, "plan visible count")
    expect_equal(
        plan.get("program_budget_seconds"),
        int(PROGRAM_BUDGET_SECONDS),
        "plan program budget",
    )
    schedule = plan.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != len(recovery_schedule()):
        raise RecoveryError("plan schedule length differs")
    run_ids = []
    for item in schedule:
        if not isinstance(item, dict) or not isinstance(item.get("run_id"), str):
            raise RecoveryError("plan schedule entry is invalid")
        run_ids.append(item["run_id"])
    expected = campaign_plan_document(
        str(plan["campaign_id"]),
        created_at=str(plan["created_at"]),
        run_ids=run_ids,
    )
    expect_equal(plan, expected, "complete campaign plan")
    return plan


def runtime_scope(torch: Any) -> dict[str, object]:
    if int(torch.cuda.device_count()) != 1 or int(torch.cuda.current_device()) != 0:
        raise RecoveryError("logical CUDA scope must be cuda:0/count1")
    return {
        "physical_gpu_index": 0,
        "cuda_visible_devices": "0",
        "visible_device_count": 1,
        "logical_device": "cuda:0",
        "device_name": str(torch.cuda.get_device_name(0)),
    }


def validate_runtime_event(
    path: Path,
    *,
    run_id: str,
    runtime_event_id: str,
    frozen: Mapping[str, object],
) -> dict[str, Any]:
    expected_path = (
        RUNTIME_EVENT_ROOT
        / run_id
        / f"runtime_event.{runtime_event_id}.json"
    )
    expect_equal(path, expected_path, "runtime event path")
    event = load_json(path, root=PROJECT_ROOT)
    if "protocol_id" in event:
        expect_equal(event.get("protocol_id"), PROTOCOL_ID, "runtime event protocol")
    expect_equal(event.get("run_id"), run_id, "runtime event run")
    expect_equal(event.get("runtime_event_id"), runtime_event_id, "runtime event ID")
    for key, expected in (
        ("model_id", frozen["model_id"]),
        ("model_revision", frozen["model_revision"]),
        ("processor_revision", frozen["processor_revision"]),
        ("model_weight_sha256", frozen["weight_sha256"]),
        ("cuda_visible_devices", "0"),
        ("torch_cuda_device_count", 1),
        ("logical_device", "cuda:0"),
        ("physical_gpu_index", 0),
    ):
        expect_equal(event.get(key), expected, f"runtime event {key}")
    return event

def run_one_gpu(*, directory: Path, index: int, attempt_id: str) -> dict[str, object]:
    preflight_gpu_environment()
    if Path.cwd().resolve(strict=True) != PROJECT_ROOT:
        raise RecoveryError("recovery runner cwd must be exact project root")
    try:
        uuid.UUID(attempt_id)
    except ValueError as exc:
        raise RecoveryError("attempt ID is invalid") from exc
    plan = load_campaign_plan(directory)
    if index < 1 or index > len(plan["schedule"]):
        raise RecoveryError("schedule index is invalid")
    entry = plan["schedule"][index - 1]
    result_path = directory / str(entry["result"])
    if result_path.exists() or result_path.is_symlink():
        raise RecoveryError("result already exists; GPU replay refused")
    source = dict(plan["source_identity"])
    frozen = program_record(str(entry["program"]))
    journal_dir = directory / str(entry["journal_dir"])
    ensure_internal_directory(journal_dir)
    run_id = str(entry["run_id"])
    runtime_event_id = str(uuid.uuid4())
    ensure_internal_directory(RUNTIME_EVENT_ROOT / run_id)
    runtime_event = (
        RUNTIME_EVENT_ROOT / run_id / f"runtime_event.{runtime_event_id}.json"
    )
    os.environ["EVO_PHASE3_MODE"] = "1"
    os.environ["EVO_RUN_ID"] = run_id
    os.environ["EVO_RUNTIME_EVENT_ID"] = runtime_event_id
    os.environ["EVO_RUNTIME_EVENT_PATH"] = str(runtime_event)
    torch = importlib.import_module("torch")
    scope = runtime_scope(torch)
    module = import_program(str(entry["program"]))
    greedy = verify_greedy(module)
    for attribute, key in (
        ("MODEL_ID", "model_id"),
        ("MODEL_REVISION", "model_revision"),
        ("PROCESSOR_REVISION", "processor_revision"),
        ("MODEL_WEIGHT_SHA256", "weight_sha256"),
    ):
        expect_equal(getattr(module, attribute, None), frozen[key], attribute)
    expect_equal(
        greedy,
        {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": frozen["effective_generation"]["max_new_tokens"],
        },
        "effective greedy generation",
    )
    queries = load_private_queries(str(entry["manifest"]))
    warmup = load_private_queries("warmup1")[0]
    started = time.time()
    evaluation = evaluate_with_journals(
        module=module,
        cuda=torch.cuda,
        program=str(entry["program"]),
        manifest_name=str(entry["manifest"]),
        queries=queries,
        warmup=warmup,
        journal_dir=journal_dir,
        campaign_id=str(plan["campaign_id"]),
        run_id=run_id,
        attempt_id=attempt_id,
        source=source,
        frozen_program=frozen,
        gpu_scope=scope,
    )
    if runtime_event.exists() or runtime_event.is_symlink():
        validate_runtime_event(
            runtime_event,
            run_id=run_id,
            runtime_event_id=runtime_event_id,
            frozen=frozen,
        )
    else:
        write_json_noreplace(
            runtime_event,
            {
                "event_version": "phase3-recovery-runtime-event-v2",
                "protocol_id": PROTOCOL_ID,
                "campaign_id": plan["campaign_id"],
                "run_id": run_id,
                "runtime_event_id": runtime_event_id,
                "attempt_id": attempt_id,
                "model_id": frozen["model_id"],
                "model_revision": frozen["model_revision"],
                "processor_revision": frozen["processor_revision"],
                "model_weight_sha256": frozen["weight_sha256"],
                "cuda_visible_devices": scope["cuda_visible_devices"],
                "torch_cuda_device_count": scope["visible_device_count"],
                "logical_device": scope["logical_device"],
                "physical_gpu_index": scope["physical_gpu_index"],
                "program": entry["program"],
                "manifest": entry["manifest"],
                "repeat": entry["repeat"],
                "started_unix": started,
                "completed_unix": time.time(),
            },
        )
    journal_hashes = {
        journal_filename(i): sha256_file(journal_dir / journal_filename(i))
        for i in range(len(queries))
    }
    document = {
        "result_version": RESULT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": plan["campaign_id"],
        "run_id": run_id,
        "runtime_event_id": runtime_event_id,
        "attempt_id": attempt_id,
        "repeat": entry["repeat"],
        "source_identity": source,
        "preregistration": plan["preregistration"],
        "program_record": frozen,
        "manifest_sha256": sha256_file(
            MANIFEST_ROOT / f"{entry['manifest']}.json"
        ),
        "gpu_scope": scope,
        "greedy": greedy,
        "journal_directory": str(journal_dir.relative_to(PROJECT_ROOT)),
        "journal_sha256": journal_hashes,
        "runtime_event_path": str(runtime_event.relative_to(PROJECT_ROOT)),
        "runtime_event_sha256": sha256_file(runtime_event),
        "evaluation": evaluation,
    }
    write_json_noreplace(result_path, document)
    return document


def verify_result_document(
    document: Mapping[str, object],
    *,
    directory: Path,
    plan: Mapping[str, object],
    entry: Mapping[str, object],
) -> dict[str, object]:
    expect_equal(document.get("result_version"), RESULT_VERSION, "result version")
    expect_equal(document.get("protocol_id"), PROTOCOL_ID, "result protocol")
    expect_equal(document.get("campaign_id"), plan["campaign_id"], "result campaign")
    expect_equal(document.get("run_id"), entry["run_id"], "result run")
    expect_equal(document.get("repeat"), entry["repeat"], "result repeat")
    expect_equal(document.get("source_identity"), plan["source_identity"], "result source")
    expect_equal(
        document.get("preregistration"), plan["preregistration"], "result prereg"
    )
    frozen = program_record(str(entry["program"]))
    expect_equal(document.get("program_record"), frozen, "result program record")
    manifest_name = str(entry["manifest"])
    manifest_sha = sha256_file(MANIFEST_ROOT / f"{manifest_name}.json")
    expect_equal(document.get("manifest_sha256"), manifest_sha, "result manifest hash")
    scope = document.get("gpu_scope")
    if not isinstance(scope, dict):
        raise RecoveryError("result GPU scope is invalid")
    for key, expected in (
        ("physical_gpu_index", 0),
        ("cuda_visible_devices", "0"),
        ("visible_device_count", 1),
        ("logical_device", "cuda:0"),
    ):
        expect_equal(scope.get(key), expected, f"result GPU {key}")
    queries = load_private_queries(manifest_name)
    manifest = load_manifest(manifest_name)
    journal_dir = directory / str(entry["journal_dir"])
    expect_equal(
        document.get("journal_directory"),
        str(journal_dir.relative_to(PROJECT_ROOT)),
        "journal directory",
    )
    records = load_valid_journals(
        journal_dir,
        campaign_id=str(plan["campaign_id"]),
        run_id=str(entry["run_id"]),
        program=str(entry["program"]),
        queries=queries,
        manifest_name=manifest_name,
        manifest_sha256=manifest_sha,
        manifest_order_sha256=str(manifest["query_key_order_sha256"]),
        source=plan["source_identity"],
        frozen_program=frozen,
    )
    if len(records) != len(queries):
        raise RecoveryError("result journals do not cover complete manifest")
    expected_hashes = {
        journal_filename(i): sha256_file(journal_dir / journal_filename(i))
        for i in range(len(queries))
    }
    expect_equal(document.get("journal_sha256"), expected_hashes, "journal hashes")
    evaluation = document.get("evaluation")
    if not isinstance(evaluation, dict):
        raise RecoveryError("result evaluation is invalid")
    expected_evaluation = summarize_records(
        program=str(entry["program"]),
        manifest_name=manifest_name,
        records=records,
        warmup_seconds=float(evaluation.get("warmup_seconds", -1)),
        warmup_metadata=evaluation.get("warmup_metadata"),
    )
    expect_equal(evaluation, expected_evaluation, "result evaluation")
    event_relative = document.get("runtime_event_path")
    if not isinstance(event_relative, str):
        raise RecoveryError("runtime event path is invalid")
    runtime_event_id = document.get("runtime_event_id")
    if not isinstance(runtime_event_id, str):
        raise RecoveryError("runtime event ID is invalid")
    try:
        uuid.UUID(runtime_event_id)
    except ValueError as exc:
        raise RecoveryError("runtime event ID is invalid") from exc
    event_path = PROJECT_ROOT / event_relative
    expected_event_path = (
        RUNTIME_EVENT_ROOT
        / str(entry["run_id"])
        / f"runtime_event.{runtime_event_id}.json"
    )
    expect_equal(event_path, expected_event_path, "runtime event path")
    expect_equal(
        document.get("runtime_event_sha256"),
        sha256_file(event_path),
        "runtime event hash",
    )
    validate_runtime_event(
        event_path,
        run_id=str(entry["run_id"]),
        runtime_event_id=runtime_event_id,
        frozen=frozen,
    )
    return dict(document)


def verify_result_path(
    path: Path,
    *,
    directory: Path,
    plan: Mapping[str, object],
    entry: Mapping[str, object],
) -> dict[str, object]:
    return verify_result_document(
        load_json(path, root=RECOVERY_ROOT),
        directory=directory,
        plan=plan,
        entry=entry,
    )


def attempt_paths(directory: Path, entry: Mapping[str, object], attempt: int) -> tuple[Path, Path, Path]:
    prefix = str(entry["attempt_prefix"]) + f".{attempt}"
    return (
        directory / (prefix + ".start.json"),
        directory / (prefix + ".end.json"),
        directory / (str(entry["console_prefix"]) + f".{attempt}.console.log"),
    )


def attempt_documents(directory: Path, plan: Mapping[str, object]) -> list[dict[str, Any]]:
    values = []
    for entry in plan["schedule"]:
        attempt = 1
        while True:
            _, end_path, _ = attempt_paths(directory, entry, attempt)
            if not end_path.exists() and not end_path.is_symlink():
                break
            values.append(load_json(end_path, root=RECOVERY_ROOT))
            attempt += 1
    return values


def program_elapsed(
    directory: Path, plan: Mapping[str, object], program: str
) -> float:
    return sum(
        float(item["elapsed_seconds"])
        for item in attempt_documents(directory, plan)
        if item.get("program") == program
    )


def journal_count(directory: Path, entry: Mapping[str, object]) -> int:
    path = directory / str(entry["journal_dir"])
    if not path.exists():
        return 0
    if path.is_symlink() or not path.is_dir():
        raise RecoveryError("journal directory is unsafe")
    names = [item.name for item in path.iterdir()]
    if any(JOURNAL_NAME.fullmatch(name) is None for name in names):
        raise RecoveryError("journal directory contains non-atomic artifact")
    return len(names)


def run_subprocess_attempt(
    directory: Path,
    *,
    plan: Mapping[str, object],
    entry: Mapping[str, object],
    attempt: int,
    remaining_seconds: float,
) -> tuple[str, int]:
    attempt_id = str(uuid.uuid4())
    start_path, end_path, console_path = attempt_paths(directory, entry, attempt)
    if any(path.exists() or path.is_symlink() for path in (start_path, end_path, console_path)):
        raise RecoveryError("attempt artifact already exists")
    stream = open_text_exclusive(console_path)
    command = [
        str(VENV_PYTHON),
        "-B",
        "-m",
        "evaluation.phase3_recovery",
        "run-one",
        "--campaign-id",
        str(plan["campaign_id"]),
        "--index",
        str(entry["index"]),
        "--attempt-id",
        attempt_id,
    ]
    started_wall = time.time()
    started_mono = time.monotonic()
    process: subprocess.Popen[str] | None = None
    outcome = "INFRA_FAILURE"
    returncode = -1
    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
            start_new_session=True,
        )
        write_json_noreplace(
            start_path,
            {
                "attempt_version": ATTEMPT_VERSION,
                "protocol_id": PROTOCOL_ID,
                "campaign_id": plan["campaign_id"],
                "entry_index": entry["index"],
                "attempt": attempt,
                "attempt_id": attempt_id,
                "program": entry["program"],
                "manifest": entry["manifest"],
                "repeat": entry["repeat"],
                "pid": process.pid,
                "remaining_program_budget_seconds": remaining_seconds,
                "started_unix": started_wall,
            },
        )
        try:
            returncode = process.wait(timeout=remaining_seconds)
            outcome = "COMPLETED" if returncode == 0 else "INFRA_FAILURE"
        except subprocess.TimeoutExpired:
            outcome = "TIMEOUT"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
            returncode = int(process.returncode or -signal.SIGKILL)
    finally:
        finalize_stream(stream, console_path)
    elapsed = min(time.monotonic() - started_mono, remaining_seconds)
    write_json_noreplace(
        end_path,
        {
            "attempt_version": ATTEMPT_VERSION,
            "protocol_id": PROTOCOL_ID,
            "campaign_id": plan["campaign_id"],
            "entry_index": entry["index"],
            "attempt": attempt,
            "attempt_id": attempt_id,
            "program": entry["program"],
            "manifest": entry["manifest"],
            "repeat": entry["repeat"],
            "outcome": outcome,
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "journal_count_after": journal_count(directory, entry),
            "console_path": str(console_path.relative_to(PROJECT_ROOT)),
            "console_sha256": sha256_file(console_path),
            "completed_unix": time.time(),
        },
    )
    return outcome, returncode


def completed_attempt_count(
    directory: Path, entry: Mapping[str, object]
) -> int:
    count = 0
    while True:
        _, end_path, _ = attempt_paths(directory, entry, count + 1)
        if not end_path.exists() and not end_path.is_symlink():
            return count
        load_json(end_path, root=RECOVERY_ROOT)
        count += 1


def run_entry(
    directory: Path,
    *,
    plan: Mapping[str, object],
    entry: Mapping[str, object],
) -> dict[str, object] | None:
    result_path = directory / str(entry["result"])
    if result_path.exists() or result_path.is_symlink():
        return verify_result_path(
            result_path, directory=directory, plan=plan, entry=entry
        )
    program = str(entry["program"])
    elapsed = program_elapsed(directory, plan, program)
    remaining = PROGRAM_BUDGET_SECONDS - elapsed
    if remaining <= 0:
        return None
    attempts = completed_attempt_count(directory, entry)
    initial_journals = journal_count(directory, entry)
    maximum_attempts = 2
    while attempts < maximum_attempts and remaining > 0:
        attempts += 1
        outcome, returncode = run_subprocess_attempt(
            directory,
            plan=plan,
            entry=entry,
            attempt=attempts,
            remaining_seconds=remaining,
        )
        elapsed = program_elapsed(directory, plan, program)
        remaining = PROGRAM_BUDGET_SECONDS - elapsed
        if outcome == "COMPLETED":
            return verify_result_path(
                result_path, directory=directory, plan=plan, entry=entry
            )
        if outcome == "TIMEOUT":
            return None
        current_journals = journal_count(directory, entry)
        if attempts == 1 and initial_journals == 0 and current_journals == 0:
            continue
        if attempts == 1 and current_journals > initial_journals:
            continue
        raise RecoveryError(
            f"formal entry failed rc={returncode} after controlled retry policy"
        )
    return None


def timed_out_programs(directory: Path, plan: Mapping[str, object]) -> set[str]:
    timed_out = set()
    for item in attempt_documents(directory, plan):
        if item.get("outcome") == "TIMEOUT":
            timed_out.add(str(item["program"]))
    for program in FORMAL_PROGRAMS:
        if program_elapsed(directory, plan, program) >= PROGRAM_BUDGET_SECONDS:
            timed_out.add(program)
    return timed_out


def aggregate_results(
    documents: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_key: dict[tuple[str, str, int], Mapping[str, object]] = {}
    for document in documents:
        evaluation = document["evaluation"]
        key = (
            str(evaluation["program"]),
            str(evaluation["manifest"]),
            int(document["repeat"]),
        )
        if key in by_key:
            raise RecoveryError(f"duplicate recovery result: {key}")
        by_key[key] = document
    expected = {
        (str(item["program"]), str(item["manifest"]), int(item["repeat"]))
        for item in recovery_schedule()
    }
    if set(by_key) != expected:
        raise RecoveryError("recovery result set differs from schedule")
    metrics: dict[str, dict[str, object]] = {}
    for program in FORMAL_PROGRAMS:
        dev = by_key[(program, "dev128", 1)]["evaluation"]
        speeds = [
            by_key[(program, "speed32", repeat)]["evaluation"]
            for repeat in (1, 2, 3)
        ]
        order = [str(item["query_key"]) for item in speeds[0]["records"]]
        for repeated in speeds[1:]:
            expect_equal(
                [str(item["query_key"]) for item in repeated["records"]],
                order,
                f"{program} speed order",
            )
        per_query_latency: list[float] = []
        per_query_tokens: list[int] = []
        for index in range(len(order)):
            records = [repeated["records"][index] for repeated in speeds]
            if any(record["measurement"] is None for record in records):
                raise RecoveryError("speed record contains an inference error")
            per_query_latency.append(
                statistics.median(float(record["latency_seconds"]) for record in records)
            )
            per_query_tokens.append(
                int(
                    statistics.median(
                        int(record["measurement"]["generated_token_count"])
                        for record in records
                    )
                )
            )
        measured = sum(per_query_latency)
        metrics[program] = {
            "source_sha256": by_key[(program, "dev128", 1)]["program_record"]["source"]["sha256"],
            "dev128": {
                "sample_count": dev["sample_count"],
                "correct_count": dev["correct_count"],
                "accuracy": dev["accuracy"],
                "error_count": dev["error_count"],
                "generated_token_total": dev["generated_token_total"],
                "termination_counts": dev["termination_counts"],
                "peak_memory_bytes": dev["peak_memory_bytes"],
            },
            "speed32": {
                "repeat_count": 3,
                "sample_count": len(order),
                "query_order": order,
                "per_query_median_latency_seconds": per_query_latency,
                "measured_seconds": measured,
                "mean_seconds": statistics.fmean(per_query_latency),
                "median_seconds": statistics.median(per_query_latency),
                "type7_p95_seconds": percentile_type7(per_query_latency, 0.95),
                "throughput_qps": len(order) / measured,
                "generated_token_median_sum": sum(per_query_tokens),
                "raw_repeat_latency_seconds": [
                    [float(record["latency_seconds"]) for record in repeated["records"]]
                    for repeated in speeds
                ],
            },
        }
    best = select_best(metrics)
    return {
        "metrics": metrics,
        "best": best,
        "baseline_thinking_dev128": {
            "status": "TIMEOUT_CENSORED",
            "timeout_seconds": 7200,
            "accuracy_available": False,
            "source_campaign_id": V1_CAMPAIGN_ID,
        },
    }


def select_best(metrics: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    expected = {"baseline_instruct", *OPTIMIZED_PROGRAMS}
    if set(metrics) != expected:
        raise RecoveryError("selection metrics candidate set differs")
    best_accuracy = min(
        OPTIMIZED_PROGRAMS,
        key=lambda name: (
            -int(metrics[name]["dev128"]["correct_count"]),
            int(metrics[name]["dev128"]["error_count"]),
            float(metrics[name]["speed32"]["median_seconds"]),
            int(metrics[name]["speed32"]["generated_token_median_sum"]),
            str(metrics[name]["source_sha256"]),
        ),
    )
    baseline_correct = int(metrics["baseline_instruct"]["dev128"]["correct_count"])
    floor_correct = max(
        baseline_correct,
        int(metrics[best_accuracy]["dev128"]["correct_count"]) - 4,
    )
    speed_eligible = [
        name
        for name in OPTIMIZED_PROGRAMS
        if int(metrics[name]["dev128"]["error_count"]) == 0
        and int(metrics[name]["dev128"]["correct_count"]) >= floor_correct
    ]
    if not speed_eligible:
        raise RecoveryError("no optimized candidate meets speed floor")
    best_speed = min(
        speed_eligible,
        key=lambda name: (
            float(metrics[name]["speed32"]["median_seconds"]),
            float(metrics[name]["speed32"]["type7_p95_seconds"]),
            int(metrics[name]["speed32"]["generated_token_median_sum"]),
            -int(metrics[name]["dev128"]["correct_count"]),
            str(metrics[name]["source_sha256"]),
        ),
    )
    overall_eligible = [
        name
        for name in OPTIMIZED_PROGRAMS
        if int(metrics[name]["dev128"]["error_count"]) == 0
        and int(metrics[name]["dev128"]["correct_count"]) >= baseline_correct
    ]
    if not overall_eligible:
        raise RecoveryError("no optimized candidate meets overall floor")
    max_correct = max(
        int(metrics[name]["dev128"]["correct_count"]) for name in overall_eligible
    )
    min_latency = min(
        float(metrics[name]["speed32"]["median_seconds"]) for name in overall_eligible
    )
    scores = {}
    for name in overall_eligible:
        accuracy_component = (
            int(metrics[name]["dev128"]["correct_count"]) / max_correct
        )
        speed_component = min_latency / float(
            metrics[name]["speed32"]["median_seconds"]
        )
        scores[name] = (
            2 * accuracy_component * speed_component
            / (accuracy_component + speed_component)
        )
    best_overall = min(
        overall_eligible,
        key=lambda name: (
            -scores[name],
            -int(metrics[name]["dev128"]["correct_count"]),
            float(metrics[name]["speed32"]["median_seconds"]),
            int(metrics[name]["speed32"]["generated_token_median_sum"]),
            str(metrics[name]["source_sha256"]),
        ),
    )
    return {
        "best_accuracy": best_accuracy,
        "best_speed": best_speed,
        "best_overall": best_overall,
        "best_speed_floor_correct": floor_correct,
        "best_speed_eligible": speed_eligible,
        "best_overall_eligible": overall_eligible,
        "overall_scores": scores,
    }


def verify_summary(
    summary: Mapping[str, object],
    *,
    directory: Path,
    plan: Mapping[str, object],
    documents: Sequence[Mapping[str, object]],
) -> None:
    expect_equal(summary.get("summary_version"), SUMMARY_VERSION, "summary version")
    expect_equal(summary.get("protocol_id"), PROTOCOL_ID, "summary protocol")
    expect_equal(summary.get("campaign_id"), plan["campaign_id"], "summary campaign")
    expect_equal(summary.get("source_identity"), plan["source_identity"], "summary source")
    expect_equal(summary.get("aggregation"), aggregate_results(documents), "aggregation")
    expected_hashes = {
        str(entry["result"]): sha256_file(directory / str(entry["result"]))
        for entry in plan["schedule"]
    }
    expect_equal(summary.get("result_sha256"), expected_hashes, "result hashes")


def run_campaign(directory: Path) -> dict[str, object]:
    plan = load_campaign_plan(directory)
    status_path = directory / "campaign_status.json"
    summary_path = directory / "campaign_summary.json"
    completed_path = directory / "completed.json"
    if completed_path.exists() or completed_path.is_symlink():
        return check_campaign(directory)
    if status_path.exists() or status_path.is_symlink():
        status = load_json(status_path, root=RECOVERY_ROOT)
        if status.get("status") == "PARTIAL_TIMEOUT":
            return status
        raise RecoveryError("existing campaign status is invalid")
    documents: list[dict[str, object]] = []
    timeout_programs = timed_out_programs(directory, plan)
    for entry in plan["schedule"]:
        program = str(entry["program"])
        result_path = directory / str(entry["result"])
        if program in timeout_programs:
            continue
        result = run_entry(directory, plan=plan, entry=entry)
        if result is None:
            timeout_programs.add(program)
            continue
        documents.append(result)
    if timeout_programs:
        status = {
            "status_version": "phase3-recovery-status-v2",
            "status": "PARTIAL_TIMEOUT",
            "protocol_id": PROTOCOL_ID,
            "campaign_id": plan["campaign_id"],
            "source_identity": plan["source_identity"],
            "timed_out_programs": sorted(timeout_programs),
            "best_selection_eligible": False,
            "completed_result_count": len(documents),
            "planned_result_count": len(plan["schedule"]),
            "completed_result_sha256": {
                str(entry["result"]): sha256_file(directory / str(entry["result"]))
                for entry in plan["schedule"]
                if (directory / str(entry["result"])).is_file()
            },
            "created_at": utc_now(),
        }
        write_json_noreplace(status_path, status)
        return status
    if len(documents) != len(plan["schedule"]):
        raise RecoveryError("campaign stopped without timeout evidence")
    aggregation = aggregate_results(documents)
    summary = {
        "summary_version": SUMMARY_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": plan["campaign_id"],
        "source_identity": plan["source_identity"],
        "aggregation": aggregation,
        "result_sha256": {
            str(entry["result"]): sha256_file(directory / str(entry["result"]))
            for entry in plan["schedule"]
        },
    }
    write_json_noreplace(summary_path, summary)
    verify_summary(
        summary,
        directory=directory,
        plan=plan,
        documents=documents,
    )
    completed = {
        "completed_version": COMPLETED_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": plan["campaign_id"],
        "source_identity": plan["source_identity"],
        "campaign_plan_sha256": sha256_file(directory / "campaign_plan.json"),
        "campaign_summary_sha256": sha256_file(summary_path),
        "completed_at": utc_now(),
    }
    write_json_noreplace(completed_path, completed)
    return summary


def check_campaign(directory: Path) -> dict[str, object]:
    plan = load_campaign_plan(directory)
    status_path = directory / "campaign_status.json"
    if status_path.exists() or status_path.is_symlink():
        status = load_json(status_path, root=RECOVERY_ROOT)
        expect_equal(status.get("protocol_id"), PROTOCOL_ID, "status protocol")
        expect_equal(status.get("campaign_id"), plan["campaign_id"], "status campaign")
        if status.get("status") != "PARTIAL_TIMEOUT":
            raise RecoveryError("campaign status is not a valid timeout")
        return status
    documents = [
        verify_result_path(
            directory / str(entry["result"]),
            directory=directory,
            plan=plan,
            entry=entry,
        )
        for entry in plan["schedule"]
    ]
    summary = load_json(directory / "campaign_summary.json", root=RECOVERY_ROOT)
    verify_summary(
        summary,
        directory=directory,
        plan=plan,
        documents=documents,
    )
    completed = load_json(directory / "completed.json", root=RECOVERY_ROOT)
    expect_equal(completed.get("completed_version"), COMPLETED_VERSION, "completed version")
    expect_equal(
        completed.get("campaign_plan_sha256"),
        sha256_file(directory / "campaign_plan.json"),
        "completed plan hash",
    )
    expect_equal(
        completed.get("campaign_summary_sha256"),
        sha256_file(directory / "campaign_summary.json"),
        "completed summary hash",
    )
    return {
        "status": "COMPLETE",
        "campaign_id": plan["campaign_id"],
        "result_count": len(documents),
        "summary_sha256": sha256_file(directory / "campaign_summary.json"),
        "completed_sha256": sha256_file(directory / "completed.json"),
        "best": summary["aggregation"]["best"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("check-prereg")
    start = sub.add_parser("start")
    start.add_argument("--campaign-id")
    resume = sub.add_parser("resume")
    resume.add_argument("--campaign-id", required=True)
    check = sub.add_parser("check")
    check.add_argument("--campaign-id", required=True)
    run = sub.add_parser("run-one")
    run.add_argument("--campaign-id", required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--attempt-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            output = prepare_preregistration()
        elif args.command == "check-prereg":
            output = verify_preregistration()
        elif args.command == "start":
            directory = create_campaign(args.campaign_id)
            output = run_campaign(directory)
        elif args.command == "resume":
            output = run_campaign(campaign_directory(args.campaign_id))
        elif args.command == "check":
            output = check_campaign(campaign_directory(args.campaign_id))
        else:
            output = run_one_gpu(
                directory=campaign_directory(args.campaign_id),
                index=args.index,
                attempt_id=args.attempt_id,
            )
    except (
        FileExistsError,
        ImportError,
        KeyError,
        OSError,
        RecoveryError,
        ValueError,
    ) as exc:
        print(f"phase3 recovery failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
