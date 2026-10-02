"""Frozen Phase 3 evaluator and per-program GPU0 runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

from evaluate import import_program, verify_greedy
from evolution.process_safety import EVO_ROOT, atomic_create_json
from evaluation.phase3_manifests import (
    MANIFEST_ROOT,
    canonical_manifest_documents,
    load_manifest,
    load_private_queries,
    sha256_file,
    write_manifests,
)


PROJECT_ROOT = EVO_ROOT / "workspace/evolving-vlm-inference-main"
PROTOCOL_PATH = PROJECT_ROOT / "evaluation/phase3_protocol.json"
CANDIDATE_MANIFEST_PATH = PROJECT_ROOT / "evaluation/candidate_manifest.json"
SELECTION_RULES_PATH = PROJECT_ROOT / "evaluation/selection_rules.json"
RESULT_ROOT = PROJECT_ROOT / "results/phase3"
RUNTIME_EVENT_ROOT = RESULT_ROOT / "runtime_events"
PROTOCOL_ID = "phase3-dev128-speed32-v1"
PROGRAM_ORDER = (
    "baseline_instruct",
    "baseline_thinking",
    "manual_instruct",
    "manual_thinking",
    "evolved_instruct",
    "evolved_thinking",
)
OPTIMIZED_PROGRAMS = (
    "manual_instruct",
    "manual_thinking",
    "evolved_instruct",
    "evolved_thinking",
)
MODEL_META = {
    "instruct": {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "revision": "78448d793a7eb2f7a987a1da76d464384aa1becd",
        "weight_sha256": "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
    },
    "thinking": {
        "model_id": "Qwen/Qwen3-VL-2B-Thinking",
        "revision": "d4f1d9f1b0107e12699c3c83d7baacd07fd0abd3",
        "weight_sha256": "d889ae8b5bffcfce115bc64e47601b1e90ce44b1d9f40ed78e620fd22417d888",
    },
}
PROGRAM_CONFIG = {
    "baseline_instruct": ("baseline", "instruct", 128, ("baseline_common.py",)),
    "baseline_thinking": ("baseline", "thinking", 128, ("baseline_common.py",)),
    "manual_instruct": ("manual", "instruct", 48, ("manual_common.py",)),
    "manual_thinking": ("manual", "thinking", 384, ("manual_common.py",)),
    "evolved_instruct": (
        "evolved", "instruct", 64,
        ("evolution/candidate_runtime.py", "evolution/dsl.py"),
    ),
    "evolved_thinking": (
        "evolved", "thinking", 384,
        ("evolution/candidate_runtime.py", "evolution/dsl.py"),
    ),
}
EVALUATOR_CLOSURE = (
    "evaluate.py",
    "evaluation/phase3_runner.py",
    "evaluation/phase3_campaign.py",
    "evaluation/phase3_manifests.py",
    "evaluation/atomic_artifacts.py",
    "evaluation/gpu_scope.py",
    "requirements.lock.txt",
)
MEASUREMENT_FIELDS = {
    "input_token_count",
    "generated_token_count",
    "termination_reason",
    "preprocessing_seconds",
    "generation_seconds",
    "total_seconds",
    "peak_memory_bytes",
}
TERMINATION_REASONS = {"eos", "length", "stop_text", "other"}


class Phase3Error(RuntimeError):
    pass


def _json_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_record(relative: str) -> dict[str, object]:
    path = PROJECT_ROOT / relative
    if path.is_symlink() or not path.is_file():
        raise Phase3Error(f"runtime closure file is unsafe: {relative}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise Phase3Error(f"runtime closure escaped project: {relative}")
    return {
        "path": relative,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def candidate_manifest_document() -> dict[str, object]:
    programs: list[dict[str, object]] = []
    for name in PROGRAM_ORDER:
        kind, family, max_new_tokens, dependencies = PROGRAM_CONFIG[name]
        model = MODEL_META[family]
        source = _file_record(f"{name}.py")
        programs.append(
            {
                "program": name,
                "kind": kind,
                "family": family,
                "source": source,
                "dependencies": [_file_record(item) for item in dependencies],
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
    closure = [_file_record(item) for item in EVALUATOR_CLOSURE]
    return {
        "manifest_version": "phase3-candidate-manifest-v1",
        "program_order": list(PROGRAM_ORDER),
        "optimized_programs": list(OPTIMIZED_PROGRAMS),
        "programs": programs,
        "evaluator_closure": closure,
        "evaluator_closure_sha256": _json_hash(closure),
    }


def protocol_document() -> dict[str, object]:
    schedule: list[dict[str, object]] = []
    for program in PROGRAM_ORDER:
        schedule.append({"program": program, "manifest": "single1", "repeat": 1})
    for program in PROGRAM_ORDER:
        schedule.append({"program": program, "manifest": "dev128", "repeat": 1})
    for repeat, offset in ((1, 0), (2, 2), (3, 4)):
        order = PROGRAM_ORDER[offset:] + PROGRAM_ORDER[:offset]
        for program in order:
            schedule.append(
                {"program": program, "manifest": "speed32", "repeat": repeat}
            )
    return {
        "protocol_id": PROTOCOL_ID,
        "physical_gpu_index": 0,
        "cuda_visible_devices": "0",
        "visible_device_count": 1,
        "logical_device": "cuda:0",
        "program_order": list(PROGRAM_ORDER),
        "schedule": schedule,
        "accuracy": {
            "manifest": "dev128",
            "repeats": 1,
            "definition": "prediction.strip().casefold()==expected.strip().casefold()",
            "errors_remain_in_denominator": True,
        },
        "speed": {
            "manifest": "speed32",
            "repeats": 3,
            "selection_latency": "per-query median across 3 repeats; aggregate median",
            "aggregate": ["mean", "median", "type7_p95", "qps"],
            "load_time_excluded": True,
            "fixed_excluded_warmup": "warmup1 before each run",
            "repeat_results_never_selected_or_discarded": True,
        },
        "generation": {
            "greedy": True,
            "do_sample": False,
            "num_beams": 1,
            "batch_size": 1,
            "dtype": "bfloat16",
            "attention": "sdpa",
            "quantization": "none",
        },
        "timing": {
            "latency_seconds": "outer wall clock with CUDA pre-sync and post-sync",
            "components": ["preprocessing_seconds", "generation_seconds", "total_seconds"],
            "components_are_diagnostic": True,
            "peak_memory_bytes": "torch.cuda.max_memory_allocated after per-query reset",
        },
        "complement": {
            "definition": "Dev128 queries whose figure was never used in Phase2",
            "report_only": True,
            "cannot_change_selection": True,
        },
        "paired_inference": {
            "comparisons": "each optimized program versus its family baseline",
            "mcnemar": "exact two-sided test on paired Dev128 correctness",
            "bootstrap": "paired accuracy difference with replacement",
            "bootstrap_seed": 20260824,
            "bootstrap_replicates": 10000,
            "bootstrap_interval": "Type-7 percentile 2.5% and 97.5%",
            "cannot_change_selection": True,
        },
    }


def selection_rules_document() -> dict[str, object]:
    return {
        "rules_version": "phase3-best-selection-v1",
        "optimized_programs": list(OPTIMIZED_PROGRAMS),
        "accuracy_uses_integer_correct_count": True,
        "best_accuracy": [
            "highest Dev128 correct_count", "fewest errors",
            "lowest speed32 per-query-median aggregate median",
            "fewest generated tokens", "lexicographic source SHA256",
        ],
        "best_speed": {
            "accuracy_floor": "max(baseline_instruct Dev128 accuracy, best_accuracy-1/32)",
            "eligibility": "Dev128 errors==0 and accuracy>=floor",
            "ties": ["lowest speed median", "lowest type7 p95", "fewest tokens", "highest accuracy", "source SHA256"],
        },
        "best_overall": {
            "eligibility": "Dev128 errors==0 and accuracy>=baseline_instruct",
            "A": "accuracy/max_eligible_accuracy",
            "S": "min_eligible_speed_median/program_speed_median",
            "score": "harmonic_mean(A,S)",
            "ties": ["higher accuracy", "lower latency", "fewer tokens", "source SHA256"],
            "metric_status": "author-defined",
        },
    }


def prepare_preregistration() -> dict[str, object]:
    write_manifests()
    atomic_create_json(CANDIDATE_MANIFEST_PATH, candidate_manifest_document())
    atomic_create_json(PROTOCOL_PATH, protocol_document())
    atomic_create_json(SELECTION_RULES_PATH, selection_rules_document())
    return {
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "candidate_manifest_sha256": sha256_file(CANDIDATE_MANIFEST_PATH),
        "selection_rules_sha256": sha256_file(SELECTION_RULES_PATH),
        "manifests": {
            name: sha256_file(MANIFEST_ROOT / f"{name}.json")
            for name in canonical_manifest_documents()
        },
    }


def _load_exact(path: Path, expected: object) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Phase3Error(f"missing physical preregistration file: {path.name}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value != expected or not isinstance(value, dict):
        raise Phase3Error(f"stale or changed preregistration file: {path.name}")
    return value


def verify_preregistration() -> dict[str, object]:
    _load_exact(PROTOCOL_PATH, protocol_document())
    _load_exact(CANDIDATE_MANIFEST_PATH, candidate_manifest_document())
    _load_exact(SELECTION_RULES_PATH, selection_rules_document())
    for name in canonical_manifest_documents():
        load_manifest(name)
    return {
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "candidate_manifest_sha256": sha256_file(CANDIDATE_MANIFEST_PATH),
        "selection_rules_sha256": sha256_file(SELECTION_RULES_PATH),
    }


def _run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(EVO_ROOT), *args], cwd=EVO_ROOT,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, shell=False, check=False,
    )
    if completed.returncode != 0:
        raise Phase3Error("git identity check failed: " + completed.stderr[:256])
    return completed.stdout.strip()


def source_identity() -> dict[str, str]:
    if _run_git("status", "--porcelain=v1", "--untracked-files=no"):
        raise Phase3Error("tracked worktree must be clean before GPU execution")
    raw = _run_git("status", "--porcelain=v1", "--untracked-files=all")
    unsafe = []
    for line in raw.splitlines():
        relative = line[3:] if len(line) > 3 else ""
        if Path(relative).suffix.lower() in {".py", ".pyc", ".pyo", ".so", ".pth"}:
            unsafe.append(relative)
    if unsafe:
        raise Phase3Error(f"untracked runtime source is forbidden: {unsafe[:16]}")
    return {
        "commit": _run_git("rev-parse", "HEAD"),
        "tree": _run_git("rev-parse", "HEAD^{tree}"),
    }


def percentile_type7(values: Sequence[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("type7 percentile input is invalid")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    fraction = position - lower
    if lower + 1 == len(ordered):
        return ordered[lower]
    return ordered[lower] + fraction * (ordered[lower + 1] - ordered[lower])


def validate_measurement_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not MEASUREMENT_FIELDS.issubset(value):
        raise Phase3Error("program did not expose complete generation metadata")
    for key in ("input_token_count", "generated_token_count", "peak_memory_bytes"):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise Phase3Error(f"invalid measurement count: {key}")
    for key in ("preprocessing_seconds", "generation_seconds", "total_seconds"):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise Phase3Error(f"invalid measurement time: {key}")
        if not math.isfinite(float(item)) or float(item) < 0:
            raise Phase3Error(f"nonfinite measurement time: {key}")
    if value["termination_reason"] not in TERMINATION_REASONS:
        raise Phase3Error("invalid termination reason")
    return {key: value[key] for key in sorted(MEASUREMENT_FIELDS)}


def _measurement_from_module(module: ModuleType) -> dict[str, object]:
    value = getattr(module.vlm_inference, "last_generation_metadata", None)
    if value is None:
        runtime = getattr(module, "_RUNTIME", None)
        value = getattr(runtime, "last_generation_metadata", None)
    return validate_measurement_mapping(value)


def evaluate_loaded_program(
    *,
    module: ModuleType,
    program: str,
    manifest_name: str,
    queries: list[dict[str, object]],
    warmup: dict[str, object],
    cuda: Any,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    cuda.synchronize()
    warm_started = clock()
    warm_prediction = module.vlm_inference(
        str(warmup["image_path"]), str(warmup["question"])
    )
    cuda.synchronize()
    warm_elapsed = clock() - warm_started
    if not isinstance(warm_prediction, str):
        raise Phase3Error("warmup prediction is not text")
    warm_metadata = _measurement_from_module(module)
    records: list[dict[str, object]] = []
    correct_count = 0
    error_count = 0
    for query in queries:
        prediction: str | None = None
        error_type: str | None = None
        error_message: str | None = None
        measurement: dict[str, object] | None = None
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
            measurement = _measurement_from_module(module)
        except Exception as exc:
            cuda.synchronize()
            prediction = None
            latency = clock() - started
            error_count += 1
            error_type = type(exc).__name__
            error_message = str(exc)[:1024]
        expected = str(query["expected_answer"])
        correct = (
            prediction is not None
            and prediction.strip().casefold() == expected.strip().casefold()
        )
        correct_count += int(correct)
        records.append(
            {
                "query_key": query["query_key"],
                "figure_id": query["figure_id"],
                "subq_idx": query["subq_idx"],
                "qid": query["qid"],
                "question": query["question"],
                "image_path": query["image_path"],
                "image_sha256": query["image_sha256"],
                "prediction": prediction,
                "prediction_sha256": (
                    hashlib.sha256(prediction.encode("utf-8")).hexdigest()
                    if prediction is not None else None
                ),
                "correct": correct,
                "latency_seconds": latency,
                "measurement": measurement,
                "error_type": error_type,
                "error_message": error_message,
            }
        )
    latencies = [float(record["latency_seconds"]) for record in records]
    measured_seconds = sum(latencies)
    return {
        "program": program,
        "manifest": manifest_name,
        "sample_count": len(records),
        "correct_count": correct_count,
        "accuracy": correct_count / len(records),
        "error_count": error_count,
        "load_and_warmup_excluded": True,
        "warmup_seconds": warm_elapsed,
        "warmup_metadata": warm_metadata,
        "latency": {
            "measured_seconds": measured_seconds,
            "mean_seconds": statistics.fmean(latencies),
            "median_seconds": statistics.median(latencies),
            "min_seconds": min(latencies),
            "max_seconds": max(latencies),
            "type7_p95_seconds": percentile_type7(latencies, 0.95),
            "throughput_qps": len(latencies) / measured_seconds,
        },
        "generated_token_total": sum(
            int(record["measurement"]["generated_token_count"])
            for record in records if record["measurement"] is not None
        ),
        "termination_counts": {
            reason: sum(
                1 for record in records
                if record["measurement"] is not None
                and record["measurement"]["termination_reason"] == reason
            )
            for reason in sorted(TERMINATION_REASONS)
        },
        "peak_memory_bytes": max(
            [
                int(record["measurement"]["peak_memory_bytes"])
                for record in records if record["measurement"] is not None
            ] or [0]
        ),
        "records": records,
    }


def _program_record(program: str) -> dict[str, object]:
    manifest = candidate_manifest_document()
    for record in manifest["programs"]:
        if isinstance(record, dict) and record.get("program") == program:
            return record
    raise Phase3Error("program is absent from candidate manifest")


def run_program(
    *, program: str, manifest_name: str, repeat: int,
    campaign_id: str, output: Path,
) -> dict[str, object]:
    if program not in PROGRAM_ORDER:
        raise Phase3Error("program is not preregistered")
    if manifest_name not in {"single1", "dev128", "speed32"}:
        raise Phase3Error("run manifest is not formal")
    if repeat != 1 and manifest_name != "speed32":
        raise Phase3Error("only speed32 may repeat")
    if manifest_name == "speed32" and repeat not in {1, 2, 3}:
        raise Phase3Error("speed32 repeat must be 1, 2 or 3")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise Phase3Error("CUDA_VISIBLE_DEVICES must be exactly 0")
    if os.environ.get("EVO_GPU_AUTHORIZATION_SCOPE"):
        raise Phase3Error("retired GPU authorization scope is forbidden")
    if Path.cwd().resolve(strict=True) != PROJECT_ROOT:
        raise Phase3Error("Phase3 runner cwd must be the exact project root")
    prereg = verify_preregistration()
    identity = source_identity()
    run_id = str(uuid.uuid4())
    runtime_event_id = str(uuid.uuid4())
    run_started_unix = time.time()
    RUNTIME_EVENT_ROOT.mkdir(parents=True, exist_ok=True)
    runtime_directory = RUNTIME_EVENT_ROOT / run_id
    runtime_directory.mkdir(mode=0o700)
    runtime_event = runtime_directory / f"runtime_event.{runtime_event_id}.json"
    os.environ["EVO_PHASE3_MODE"] = "1"
    os.environ["EVO_RUN_ID"] = run_id
    os.environ["EVO_RUNTIME_EVENT_ID"] = runtime_event_id
    os.environ["EVO_RUNTIME_EVENT_PATH"] = str(runtime_event)
    torch = importlib.import_module("torch")
    if int(torch.cuda.device_count()) != 1 or int(torch.cuda.current_device()) != 0:
        raise Phase3Error("logical CUDA scope is not exactly cuda:0/count1")
    module = import_program(program)
    greedy = verify_greedy(module)
    expected_program = _program_record(program)
    for attribute, key in (
        ("MODEL_ID", "model_id"), ("MODEL_REVISION", "model_revision"),
        ("PROCESSOR_REVISION", "processor_revision"),
        ("MODEL_WEIGHT_SHA256", "weight_sha256"),
    ):
        if getattr(module, attribute, None) != expected_program[key]:
            raise Phase3Error(f"program metadata mismatch: {attribute}")
    if greedy["max_new_tokens"] != expected_program["effective_generation"]["max_new_tokens"]:
        raise Phase3Error("program max_new_tokens differs from preregistration")
    queries = load_private_queries(manifest_name)
    warmup = load_private_queries("warmup1")[0]
    evaluated = evaluate_loaded_program(
        module=module, program=program, manifest_name=manifest_name,
        queries=queries, warmup=warmup, cuda=torch.cuda,
    )
    if not runtime_event.exists():
        atomic_create_json(
            runtime_event,
            {
                "event_schema": "phase3-runner-runtime-event-v1",
                "protocol_id": PROTOCOL_ID,
                "run_id": run_id,
                "runtime_event_id": runtime_event_id,
                "program": program,
                "manifest": manifest_name,
                "repeat": repeat,
                "source_identity": identity,
                "gpu_scope": {"physical_gpu_index": 0, "cuda_visible_devices": "0", "visible_device_count": 1, "logical_device": "cuda:0"},
                "started_unix": run_started_unix,
                "completed_unix": time.time(),
            },
        )
    document = {
        "result_version": "phase3-program-result-v1",
        "protocol_id": PROTOCOL_ID,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "runtime_event_id": runtime_event_id,
        "repeat": repeat,
        "source_identity": identity,
        "preregistration": prereg,
        "program_record": expected_program,
        "manifest_sha256": sha256_file(MANIFEST_ROOT / f"{manifest_name}.json"),
        "gpu_scope": {
            "physical_gpu_index": 0,
            "cuda_visible_devices": "0",
            "visible_device_count": 1,
            "logical_device": "cuda:0",
            "device_name": str(torch.cuda.get_device_name(0)),
        },
        "greedy": greedy,
        "runtime_event_path": str(runtime_event.relative_to(PROJECT_ROOT)),
        "runtime_event_sha256": (
            sha256_file(runtime_event) if runtime_event.is_file() else None
        ),
        "evaluation": evaluated,
    }
    _reject_nonfinite(document)
    resolved_output = output.resolve(strict=False)
    if not resolved_output.is_relative_to(RESULT_ROOT.resolve(strict=False)):
        raise Phase3Error("result output escaped Phase3 result root")
    atomic_create_json(resolved_output, document)
    return document


def _reject_nonfinite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise Phase3Error("artifact contains NaN or Infinity")
    if isinstance(value, dict):
        for child in value.values():
            _reject_nonfinite(child)
    elif isinstance(value, list):
        for child in value:
            _reject_nonfinite(child)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("check")
    run = subparsers.add_parser("run")
    run.add_argument("--program", choices=PROGRAM_ORDER, required=True)
    run.add_argument("--manifest", choices=("single1", "dev128", "speed32"), required=True)
    run.add_argument("--repeat", type=int, required=True)
    run.add_argument("--campaign-id", required=True)
    run.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_preregistration()
        elif args.command == "check":
            result = verify_preregistration()
        else:
            result = run_program(
                program=args.program, manifest_name=args.manifest,
                repeat=args.repeat, campaign_id=args.campaign_id,
                output=args.output,
            )
    except (Phase3Error, OSError, ValueError, ImportError) as exc:
        print(f"phase3 failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
