"""Immutable Phase 3 campaign orchestration, verification, and aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    CANDIDATE_MANIFEST_PATH,
    OPTIMIZED_PROGRAMS,
    PROGRAM_ORDER,
    PROJECT_ROOT,
    PROTOCOL_ID,
    PROTOCOL_PATH,
    RESULT_ROOT,
    SELECTION_RULES_PATH,
    candidate_manifest_document,
    percentile_type7,
    protocol_document,
    selection_rules_document,
    source_identity,
    validate_measurement_mapping,
    verify_preregistration,
)


CAMPAIGN_ROOT = RESULT_ROOT / "campaigns"
VENV_PYTHON = PROJECT_ROOT.parent.parent / ".venv/bin/python"
RESULT_VERSION = "phase3-program-result-v1"
PLAN_VERSION = "phase3-campaign-plan-v1"
SUMMARY_VERSION = "phase3-campaign-summary-v1"
COMPLETED_VERSION = "phase3-campaign-completed-v1"
RUN_TIMEOUT_SECONDS = 7200
BOOTSTRAP_SEED = 20260824
BOOTSTRAP_REPLICATES = 10000


class Phase3CampaignError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Phase3CampaignError(f"missing physical JSON artifact: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise Phase3CampaignError(f"JSON artifact escaped project: {path}")
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Phase3CampaignError(f"JSON artifact is not an object: {path}")
    return value


def _finite_number(value: object, *, name: str, integer: bool = False) -> float:
    if isinstance(value, bool):
        raise Phase3CampaignError(f"{name} is boolean")
    if integer:
        if not isinstance(value, int) or value < 0:
            raise Phase3CampaignError(f"{name} is not a nonnegative integer")
        return float(value)
    if not isinstance(value, (int, float)):
        raise Phase3CampaignError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise Phase3CampaignError(f"{name} is negative or nonfinite")
    return result


def _expect_equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise Phase3CampaignError(f"{name} differs from frozen value")


def _safe_campaign_directory(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise Phase3CampaignError("campaign directory is not physical")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(CAMPAIGN_ROOT.resolve(strict=True)):
        raise Phase3CampaignError("campaign directory escaped Phase3 root")
    return resolved


def _result_name(index: int, item: Mapping[str, object]) -> str:
    return (
        f"{index:02d}.{item['program']}.{item['manifest']}."
        f"r{item['repeat']}.json"
    )


def _console_name(index: int, item: Mapping[str, object]) -> str:
    return (
        f"{index:02d}.{item['program']}.{item['manifest']}."
        f"r{item['repeat']}.console.log"
    )


def campaign_plan_document(
    campaign_id: str, *, created_at: str | None = None
) -> dict[str, object]:
    try:
        uuid.UUID(campaign_id)
    except ValueError as exc:
        raise Phase3CampaignError("campaign ID must be a UUID") from exc
    prereg = verify_preregistration()
    identity = source_identity()
    schedule = []
    for index, raw in enumerate(protocol_document()["schedule"], start=1):
        item = dict(raw)
        item["index"] = index
        item["result"] = _result_name(index, raw)
        item["console"] = _console_name(index, raw)
        schedule.append(item)
    return {
        "plan_version": PLAN_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": campaign_id,
        "created_at": created_at or _utc_now(),
        "source_identity": identity,
        "preregistration": prereg,
        "physical_gpu_index": 0,
        "cuda_visible_devices": "0",
        "visible_device_count": 1,
        "schedule": schedule,
    }


def create_campaign(campaign_id: str | None = None) -> Path:
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    selected_id = campaign_id or str(uuid.uuid4())
    directory = CAMPAIGN_ROOT / selected_id
    directory.mkdir(mode=0o700)
    plan = campaign_plan_document(selected_id)
    write_json_noreplace(directory / "campaign_plan.json", plan)
    return directory


def load_campaign_plan(directory: Path) -> dict[str, Any]:
    root = _safe_campaign_directory(directory)
    plan = _load_json(root / "campaign_plan.json")
    if plan.get("plan_version") != PLAN_VERSION:
        raise Phase3CampaignError("campaign plan version differs")
    try:
        uuid.UUID(str(plan.get("campaign_id")))
    except ValueError as exc:
        raise Phase3CampaignError("campaign plan UUID is invalid") from exc
    _expect_equal(plan.get("protocol_id"), PROTOCOL_ID, "plan protocol")
    _expect_equal(plan.get("source_identity"), source_identity(), "plan source")
    _expect_equal(
        plan.get("preregistration"), verify_preregistration(), "plan preregistration"
    )
    _expect_equal(plan.get("physical_gpu_index"), 0, "plan physical GPU")
    _expect_equal(plan.get("cuda_visible_devices"), "0", "plan CUDA scope")
    _expect_equal(plan.get("visible_device_count"), 1, "plan visible count")
    expected_schedule = campaign_plan_document(
        str(plan["campaign_id"]), created_at=str(plan["created_at"])
    )["schedule"]
    _expect_equal(plan.get("schedule"), expected_schedule, "plan schedule")
    return plan


def _program_record(program: str) -> dict[str, Any]:
    for record in candidate_manifest_document()["programs"]:
        if isinstance(record, dict) and record.get("program") == program:
            return record
    raise Phase3CampaignError("program missing from candidate manifest")


def _verify_runtime_event(document: Mapping[str, object]) -> None:
    raw_path = document.get("runtime_event_path")
    stated_hash = document.get("runtime_event_sha256")
    if not isinstance(raw_path, str):
        raise Phase3CampaignError("runtime event path is invalid")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise Phase3CampaignError("runtime event path escaped")
    path = PROJECT_ROOT / relative
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise Phase3CampaignError("runtime event is unsafe")
        if not isinstance(stated_hash, str) or sha256_file(path) != stated_hash:
            raise Phase3CampaignError("runtime event hash differs")
    else:
        raise Phase3CampaignError("required runtime event is absent")


def _recomputed_evaluation(
    evaluation: Mapping[str, object],
    *,
    program: str,
    manifest_name: str,
) -> dict[str, object]:
    queries = load_private_queries(manifest_name)
    _expect_equal(evaluation.get("program"), program, "evaluation program")
    _expect_equal(evaluation.get("manifest"), manifest_name, "evaluation manifest")
    _expect_equal(evaluation.get("sample_count"), len(queries), "sample count")
    records = evaluation.get("records")
    if not isinstance(records, list) or len(records) != len(queries):
        raise Phase3CampaignError("records do not cover the complete manifest")
    correct_count = 0
    error_count = 0
    latencies: list[float] = []
    generated_total = 0
    termination_counts = {reason: 0 for reason in ("eos", "length", "other", "stop_text")}
    peak = 0
    seen: set[str] = set()
    canonical_records: list[dict[str, object]] = []
    for index, (record, query) in enumerate(zip(records, queries)):
        if not isinstance(record, dict):
            raise Phase3CampaignError(f"record {index} is not an object")
        key = str(query["query_key"])
        if key in seen:
            raise Phase3CampaignError("duplicate query key in expected manifest")
        seen.add(key)
        for field in (
            "query_key", "figure_id", "subq_idx", "qid", "question",
            "image_path", "image_sha256",
        ):
            _expect_equal(record.get(field), query[field], f"record {index} {field}")
        prediction = record.get("prediction")
        prediction_hash = record.get("prediction_sha256")
        error_type = record.get("error_type")
        error_message = record.get("error_message")
        measurement = record.get("measurement")
        if prediction is None:
            if prediction_hash is not None or measurement is not None:
                raise Phase3CampaignError("failed record carries output or measurement")
            if not isinstance(error_type, str) or not error_type:
                raise Phase3CampaignError("failed record lacks error type")
            if not isinstance(error_message, str):
                raise Phase3CampaignError("failed record lacks error message")
            error_count += 1
            normalized_measurement = None
        elif isinstance(prediction, str):
            _expect_equal(
                prediction_hash, _sha256_text(prediction), "prediction hash"
            )
            if error_type is not None or error_message is not None:
                raise Phase3CampaignError("successful record carries an error")
            normalized_measurement = validate_measurement_mapping(measurement)
            generated_total += int(normalized_measurement["generated_token_count"])
            reason = str(normalized_measurement["termination_reason"])
            termination_counts[reason] += 1
            peak = max(peak, int(normalized_measurement["peak_memory_bytes"]))
        else:
            raise Phase3CampaignError("prediction must be text or null")
        correct = (
            isinstance(prediction, str)
            and prediction.strip().casefold()
            == str(query["expected_answer"]).strip().casefold()
        )
        _expect_equal(record.get("correct"), correct, "record correctness")
        correct_count += int(correct)
        latency = _finite_number(
            record.get("latency_seconds"), name="record latency"
        )
        latencies.append(latency)
        canonical_records.append(
            {
                **{field: query[field] for field in (
                    "query_key", "figure_id", "subq_idx", "qid", "question",
                    "image_path", "image_sha256",
                )},
                "prediction": prediction,
                "prediction_sha256": prediction_hash,
                "correct": correct,
                "latency_seconds": latency,
                "measurement": normalized_measurement,
                "error_type": error_type,
                "error_message": error_message,
            }
        )
    measured = sum(latencies)
    if measured <= 0:
        raise Phase3CampaignError("measured duration must be positive")
    warmup_seconds = _finite_number(
        evaluation.get("warmup_seconds"), name="warmup seconds"
    )
    warmup_metadata = validate_measurement_mapping(
        evaluation.get("warmup_metadata")
    )
    expected_latency = {
        "measured_seconds": measured,
        "mean_seconds": statistics.fmean(latencies),
        "median_seconds": statistics.median(latencies),
        "min_seconds": min(latencies),
        "max_seconds": max(latencies),
        "type7_p95_seconds": percentile_type7(latencies, 0.95),
        "throughput_qps": len(latencies) / measured,
    }
    _expect_equal(evaluation.get("correct_count"), correct_count, "correct count")
    _expect_equal(evaluation.get("accuracy"), correct_count / len(queries), "accuracy")
    _expect_equal(evaluation.get("error_count"), error_count, "error count")
    _expect_equal(
        evaluation.get("load_and_warmup_excluded"), True, "warmup exclusion"
    )
    _expect_equal(evaluation.get("latency"), expected_latency, "latency aggregate")
    _expect_equal(
        evaluation.get("generated_token_total"), generated_total, "token total"
    )
    _expect_equal(
        evaluation.get("termination_counts"),
        {key: termination_counts[key] for key in sorted(termination_counts)},
        "termination summary",
    )
    _expect_equal(evaluation.get("peak_memory_bytes"), peak, "peak memory")
    return {
        "program": program,
        "manifest": manifest_name,
        "sample_count": len(queries),
        "correct_count": correct_count,
        "accuracy": correct_count / len(queries),
        "error_count": error_count,
        "load_and_warmup_excluded": True,
        "warmup_seconds": warmup_seconds,
        "warmup_metadata": warmup_metadata,
        "latency": expected_latency,
        "generated_token_total": generated_total,
        "termination_counts": {
            key: termination_counts[key] for key in sorted(termination_counts)
        },
        "peak_memory_bytes": peak,
        "records": canonical_records,
    }


def verify_result_document(
    document: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    entry: Mapping[str, object],
) -> dict[str, object]:
    _expect_equal(document.get("result_version"), RESULT_VERSION, "result version")
    _expect_equal(document.get("protocol_id"), PROTOCOL_ID, "result protocol")
    _expect_equal(document.get("campaign_id"), plan["campaign_id"], "campaign ID")
    try:
        uuid.UUID(str(document.get("run_id")))
        uuid.UUID(str(document.get("runtime_event_id")))
    except ValueError as exc:
        raise Phase3CampaignError("result event UUID is invalid") from exc
    program = str(entry["program"])
    manifest_name = str(entry["manifest"])
    _expect_equal(document.get("repeat"), entry["repeat"], "repeat")
    _expect_equal(document.get("source_identity"), plan["source_identity"], "source")
    _expect_equal(
        document.get("preregistration"), plan["preregistration"], "preregistration"
    )
    expected_program = _program_record(program)
    _expect_equal(document.get("program_record"), expected_program, "program record")
    _expect_equal(
        document.get("manifest_sha256"),
        sha256_file(MANIFEST_ROOT / f"{manifest_name}.json"),
        "manifest hash",
    )
    scope = document.get("gpu_scope")
    if not isinstance(scope, dict):
        raise Phase3CampaignError("GPU scope is not an object")
    for key, expected in (
        ("physical_gpu_index", 0), ("cuda_visible_devices", "0"),
        ("visible_device_count", 1), ("logical_device", "cuda:0"),
    ):
        _expect_equal(scope.get(key), expected, f"GPU scope {key}")
    if not isinstance(scope.get("device_name"), str) or not scope["device_name"]:
        raise Phase3CampaignError("GPU device name is absent")
    expected_greedy = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": expected_program["effective_generation"]["max_new_tokens"],
    }
    _expect_equal(document.get("greedy"), expected_greedy, "greedy metadata")
    _verify_runtime_event(document)
    evaluation = document.get("evaluation")
    if not isinstance(evaluation, dict):
        raise Phase3CampaignError("evaluation is not an object")
    recomputed = _recomputed_evaluation(
        evaluation, program=program, manifest_name=manifest_name
    )
    _expect_equal(evaluation, recomputed, "complete evaluation")
    return dict(document)


def verify_result_path(
    path: Path, *, plan: Mapping[str, object], entry: Mapping[str, object]
) -> dict[str, object]:
    document = _load_json(path)
    return verify_result_document(document, plan=plan, entry=entry)


def _run_one(
    directory: Path, *, plan: Mapping[str, object], entry: Mapping[str, object]
) -> dict[str, object]:
    result = directory / str(entry["result"])
    if result.is_file() and not result.is_symlink():
        return verify_result_path(result, plan=plan, entry=entry)
    console = directory / str(entry["console"])
    stream = open_text_exclusive(console)
    command = [
        str(VENV_PYTHON), "-B", "-m", "evaluation.phase3_runner", "run",
        "--program", str(entry["program"]),
        "--manifest", str(entry["manifest"]),
        "--repeat", str(entry["repeat"]),
        "--campaign-id", str(plan["campaign_id"]),
        "--output", str(result),
    ]
    process: subprocess.Popen[str] | None = None
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
        returncode = process.wait(timeout=RUN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        if process is not None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
        raise Phase3CampaignError(f"formal run timed out: {entry}") from exc
    finally:
        finalize_stream(stream, console)
    if returncode != 0:
        raise Phase3CampaignError(
            f"formal runner failed rc={returncode}; evidence={console}"
        )
    return verify_result_path(result, plan=plan, entry=entry)


def _exact_mcnemar(first: Sequence[bool], second: Sequence[bool]) -> dict[str, object]:
    b = sum(bool(a) and not bool(c) for a, c in zip(first, second))
    c = sum(not bool(a) and bool(c) for a, c in zip(first, second))
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1))
        p_value = min(1.0, 2.0 * tail / (2 ** discordant))
    return {"first_only": b, "second_only": c, "exact_two_sided_p": p_value}


def _paired_bootstrap(
    first: Sequence[bool], second: Sequence[bool]
) -> dict[str, object]:
    if len(first) != len(second) or not first:
        raise Phase3CampaignError("paired bootstrap inputs differ")
    differences = [int(a) - int(b) for a, b in zip(first, second)]
    rng = random.Random(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        values.append(
            statistics.fmean(
                differences[rng.randrange(len(differences))]
                for _ in range(len(differences))
            )
        )
    return {
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "mean_accuracy_delta": statistics.fmean(differences),
        "type7_ci95": [
            percentile_type7(values, 0.025),
            percentile_type7(values, 0.975),
        ],
    }


def aggregate_verified_results(
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
            raise Phase3CampaignError(f"duplicate formal result: {key}")
        by_key[key] = document
    expected_keys = {
        (str(item["program"]), str(item["manifest"]), int(item["repeat"]))
        for item in protocol_document()["schedule"]
    }
    if set(by_key) != expected_keys:
        raise Phase3CampaignError("formal result set differs from schedule")
    dev_manifest = load_manifest("dev128")
    complement_keys = set(dev_manifest["untouched_complement_query_keys"])
    metrics: dict[str, dict[str, object]] = {}
    for program in PROGRAM_ORDER:
        single = by_key[(program, "single1", 1)]["evaluation"]
        dev = by_key[(program, "dev128", 1)]["evaluation"]
        speed = [
            by_key[(program, "speed32", repeat)]["evaluation"]
            for repeat in (1, 2, 3)
        ]
        query_order = [str(record["query_key"]) for record in speed[0]["records"]]
        for repeated in speed[1:]:
            _expect_equal(
                [str(record["query_key"]) for record in repeated["records"]],
                query_order,
                "speed query order",
            )
        per_query_latency = []
        per_query_tokens = []
        component_medians = {
            key: [] for key in (
                "preprocessing_seconds", "generation_seconds", "total_seconds"
            )
        }
        for index in range(len(query_order)):
            records = [repeated["records"][index] for repeated in speed]
            per_query_latency.append(
                statistics.median(float(record["latency_seconds"]) for record in records)
            )
            measurements = [record["measurement"] for record in records]
            if any(item is None for item in measurements):
                raise Phase3CampaignError("speed result contains an inference error")
            per_query_tokens.append(
                int(statistics.median(
                    int(item["generated_token_count"]) for item in measurements
                ))
            )
            for component in component_medians:
                component_medians[component].append(
                    statistics.median(float(item[component]) for item in measurements)
                )
        speed_measured = sum(per_query_latency)
        if speed_measured <= 0:
            raise Phase3CampaignError("speed aggregate duration is zero")
        dev_records = list(dev["records"])
        complement = [
            record for record in dev_records
            if str(record["query_key"]) in complement_keys
        ]
        if len(complement) != len(complement_keys):
            raise Phase3CampaignError("Dev128 complement coverage differs")
        source_sha = str(by_key[(program, "dev128", 1)]["program_record"]["source"]["sha256"])
        metrics[program] = {
            "source_sha256": source_sha,
            "single": {
                "correct_count": single["correct_count"],
                "error_count": single["error_count"],
            },
            "dev128": {
                "sample_count": dev["sample_count"],
                "correct_count": dev["correct_count"],
                "accuracy": dev["accuracy"],
                "error_count": dev["error_count"],
                "generated_token_total": dev["generated_token_total"],
                "peak_memory_bytes": dev["peak_memory_bytes"],
            },
            "dev128_untouched_complement_report_only": {
                "sample_count": len(complement),
                "correct_count": sum(int(record["correct"]) for record in complement),
                "accuracy": (
                    sum(int(record["correct"]) for record in complement) / len(complement)
                ),
                "error_count": sum(record["error_type"] is not None for record in complement),
            },
            "speed32": {
                "repeat_count": 3,
                "sample_count": len(query_order),
                "query_order": query_order,
                "per_query_median_latency_seconds": per_query_latency,
                "measured_seconds": speed_measured,
                "mean_seconds": statistics.fmean(per_query_latency),
                "median_seconds": statistics.median(per_query_latency),
                "type7_p95_seconds": percentile_type7(per_query_latency, 0.95),
                "throughput_qps": len(per_query_latency) / speed_measured,
                "generated_token_median_sum": sum(per_query_tokens),
                "peak_memory_bytes": max(
                    int(repeated["peak_memory_bytes"]) for repeated in speed
                ),
                "component_per_query_median_mean_seconds": {
                    key: statistics.fmean(values)
                    for key, values in component_medians.items()
                },
                "raw_repeat_latency_seconds": [
                    [float(record["latency_seconds"]) for record in repeated["records"]]
                    for repeated in speed
                ],
            },
        }
    optimized = [metrics[name] for name in OPTIMIZED_PROGRAMS]
    best_accuracy_name = min(
        OPTIMIZED_PROGRAMS,
        key=lambda name: (
            -int(metrics[name]["dev128"]["correct_count"]),
            int(metrics[name]["dev128"]["error_count"]),
            float(metrics[name]["speed32"]["median_seconds"]),
            int(metrics[name]["speed32"]["generated_token_median_sum"]),
            str(metrics[name]["source_sha256"]),
        ),
    )
    baseline_accuracy = float(metrics["baseline_instruct"]["dev128"]["accuracy"])
    best_accuracy = float(metrics[best_accuracy_name]["dev128"]["accuracy"])
    speed_floor = max(baseline_accuracy, best_accuracy - 1 / 32)
    speed_eligible = [
        name for name in OPTIMIZED_PROGRAMS
        if int(metrics[name]["dev128"]["error_count"]) == 0
        and float(metrics[name]["dev128"]["accuracy"]) >= speed_floor
    ]
    if not speed_eligible:
        raise Phase3CampaignError("no optimized program meets best-speed eligibility")
    best_speed_name = min(
        speed_eligible,
        key=lambda name: (
            float(metrics[name]["speed32"]["median_seconds"]),
            float(metrics[name]["speed32"]["type7_p95_seconds"]),
            int(metrics[name]["speed32"]["generated_token_median_sum"]),
            -float(metrics[name]["dev128"]["accuracy"]),
            str(metrics[name]["source_sha256"]),
        ),
    )
    overall_eligible = [
        name for name in OPTIMIZED_PROGRAMS
        if int(metrics[name]["dev128"]["error_count"]) == 0
        and float(metrics[name]["dev128"]["accuracy"]) >= baseline_accuracy
    ]
    if not overall_eligible:
        raise Phase3CampaignError("no optimized program meets overall eligibility")
    max_accuracy = max(float(metrics[name]["dev128"]["accuracy"]) for name in overall_eligible)
    min_latency = min(float(metrics[name]["speed32"]["median_seconds"]) for name in overall_eligible)
    overall_scores = {}
    for name in overall_eligible:
        accuracy_component = float(metrics[name]["dev128"]["accuracy"]) / max_accuracy
        speed_component = min_latency / float(metrics[name]["speed32"]["median_seconds"])
        overall_scores[name] = (
            2 * accuracy_component * speed_component
            / (accuracy_component + speed_component)
        )
    best_overall_name = min(
        overall_eligible,
        key=lambda name: (
            -overall_scores[name],
            -float(metrics[name]["dev128"]["accuracy"]),
            float(metrics[name]["speed32"]["median_seconds"]),
            int(metrics[name]["speed32"]["generated_token_median_sum"]),
            str(metrics[name]["source_sha256"]),
        ),
    )
    paired = {}
    for name in OPTIMIZED_PROGRAMS:
        family_baseline = (
            "baseline_instruct" if name.endswith("instruct") else "baseline_thinking"
        )
        candidate_correct = [
            bool(record["correct"])
            for record in by_key[(name, "dev128", 1)]["evaluation"]["records"]
        ]
        baseline_correct = [
            bool(record["correct"])
            for record in by_key[(family_baseline, "dev128", 1)]["evaluation"]["records"]
        ]
        paired[name] = {
            "baseline": family_baseline,
            "mcnemar": _exact_mcnemar(candidate_correct, baseline_correct),
            "paired_bootstrap": _paired_bootstrap(candidate_correct, baseline_correct),
        }
    return {
        "metrics": metrics,
        "best": {
            "best_accuracy": best_accuracy_name,
            "best_speed": best_speed_name,
            "best_overall": best_overall_name,
            "best_speed_accuracy_floor": speed_floor,
            "overall_scores": overall_scores,
        },
        "paired_inference": paired,
        "selection_rules_sha256": sha256_file(SELECTION_RULES_PATH),
    }


def verify_campaign_summary(
    summary: Mapping[str, object],
    *, plan: Mapping[str, object],
    documents: Sequence[Mapping[str, object]],
) -> None:
    expected = aggregate_verified_results(documents)
    _expect_equal(summary.get("summary_version"), SUMMARY_VERSION, "summary version")
    _expect_equal(summary.get("protocol_id"), PROTOCOL_ID, "summary protocol")
    _expect_equal(summary.get("campaign_id"), plan["campaign_id"], "summary campaign")
    _expect_equal(summary.get("source_identity"), plan["source_identity"], "summary source")
    _expect_equal(summary.get("aggregation"), expected, "summary aggregation")
    hashes = summary.get("result_sha256")
    expected_hashes = {
        str(entry["result"]): sha256_file(
            CAMPAIGN_ROOT / str(plan["campaign_id"]) / str(entry["result"])
        )
        for entry in plan["schedule"]
    }
    _expect_equal(hashes, expected_hashes, "summary result hashes")


def run_campaign(directory: Path) -> dict[str, object]:
    root = _safe_campaign_directory(directory)
    plan = load_campaign_plan(root)
    completed_path = root / "completed.json"
    summary_path = root / "campaign_summary.json"
    documents: list[dict[str, object]] = []
    for entry in plan["schedule"]:
        documents.append(_run_one(root, plan=plan, entry=entry))
    if summary_path.is_file() and not summary_path.is_symlink():
        summary = _load_json(summary_path)
        verify_campaign_summary(summary, plan=plan, documents=documents)
    else:
        aggregation = aggregate_verified_results(documents)
        summary = {
            "summary_version": SUMMARY_VERSION,
            "protocol_id": PROTOCOL_ID,
            "campaign_id": plan["campaign_id"],
            "source_identity": plan["source_identity"],
            "aggregation": aggregation,
            "result_sha256": {
                str(entry["result"]): sha256_file(root / str(entry["result"]))
                for entry in plan["schedule"]
            },
        }
        write_json_noreplace(summary_path, summary)
        verify_campaign_summary(summary, plan=plan, documents=documents)
    completed = {
        "completed_version": COMPLETED_VERSION,
        "campaign_id": plan["campaign_id"],
        "source_identity": plan["source_identity"],
        "campaign_plan_sha256": sha256_file(root / "campaign_plan.json"),
        "campaign_summary_sha256": sha256_file(summary_path),
        "completed_at": _utc_now(),
    }
    if completed_path.is_file() and not completed_path.is_symlink():
        existing = _load_json(completed_path)
        for key in (
            "completed_version", "campaign_id", "source_identity",
            "campaign_plan_sha256", "campaign_summary_sha256",
        ):
            _expect_equal(existing.get(key), completed[key], f"completed {key}")
    else:
        write_json_noreplace(completed_path, completed)
    return summary


def check_campaign(directory: Path) -> dict[str, object]:
    root = _safe_campaign_directory(directory)
    plan = load_campaign_plan(root)
    documents = [
        verify_result_path(root / str(entry["result"]), plan=plan, entry=entry)
        for entry in plan["schedule"]
    ]
    summary = _load_json(root / "campaign_summary.json")
    verify_campaign_summary(summary, plan=plan, documents=documents)
    completed = _load_json(root / "completed.json")
    _expect_equal(completed.get("completed_version"), COMPLETED_VERSION, "completed version")
    _expect_equal(completed.get("campaign_id"), plan["campaign_id"], "completed campaign")
    _expect_equal(
        completed.get("campaign_plan_sha256"),
        sha256_file(root / "campaign_plan.json"),
        "completed plan hash",
    )
    _expect_equal(
        completed.get("campaign_summary_sha256"),
        sha256_file(root / "campaign_summary.json"),
        "completed summary hash",
    )
    return {
        "campaign_id": plan["campaign_id"],
        "result_count": len(documents),
        "summary_sha256": sha256_file(root / "campaign_summary.json"),
        "completed_sha256": sha256_file(root / "completed.json"),
        "best": summary["aggregation"]["best"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--campaign-id")
    resume = subparsers.add_parser("resume")
    resume.add_argument("--campaign-id", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--campaign-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "start":
            directory = create_campaign(args.campaign_id)
            result = run_campaign(directory)
        else:
            directory = CAMPAIGN_ROOT / args.campaign_id
            result = (
                run_campaign(directory)
                if args.command == "resume"
                else check_campaign(directory)
            )
    except (
        FileExistsError, OSError, ValueError, KeyError,
        Phase3CampaignError,
    ) as exc:
        print(f"phase3 campaign failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
