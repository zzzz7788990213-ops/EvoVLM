"""CPU-only finalization of verified Phase3 Recovery-v2 evidence.

The production command verifies either a complete canonical campaign or the sealed
composition of retained single/Dev128 evidence plus an authorized whole-set
Speed32 replacement. Only then may it export byte-identical best programs and
generated paper tables. A hash-complete manifest is published last; later checks
use that sealed manifest and do not replay a GPU workload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PROJECT_NAME = "evolving-vlm-inference-main"
PROTOCOL_ID = "phase3-recovery-v2"
FINAL_MANIFEST_RELATIVE = "evaluation/recovery_v2/final_manifest.json"
FORMAL_TABLE_RELATIVE = "report/generated/formal_results.tex"
NUMBER_TRACE_RELATIVE = "report/generated/report_number_trace.json"
FINAL_MANIFEST_V2 = "phase3-recovery-final-manifest-v2"
FINAL_MANIFEST_REPLACEMENT_V3 = (
    "phase3-recovery-final-manifest-v3-speed32-replacement"
)
REPLACEMENT_SUMMARY_NAME = "speed32_replacement_summary.json"
REPLACEMENT_COMPLETED_NAME = "speed32_replacement_completed.json"
REPLACEMENT_MEASUREMENTS_NAME = "speed32_measurements_complete.json"
REPLACEMENT_MARKER_NAME = "speed32_replacement_marker.json"
CONTAMINATION_AUDIT_RELATIVE = (
    "results/phase3_recovery_v2/incidents/"
    "1fa07427-6b42-4bbb-a9f9-af63742bdfc6.gpu0_external_contention.audit.json"
)
OLD_PLAN_SHA256 = "8d329631caea13add8bb8bfb39a847affc3c5377b4605e703eac7f41ab4fb118"
OLD_SUMMARY_SHA256 = "ca2866b66da4c37426849117d3fea02940522a6a6c389528967bc21372e21d3c"
OLD_COMPLETED_SHA256 = "0651a6751c450b7747397a984a52e479a0e90198fb647835ce8547a743a4658e"
CONTAMINATION_AUDIT_SHA256 = "2f97fbe9539fa1371fe17f43f6d55d9610af1f051037071967885e258daf3c27"
OFFICIAL_STYLE_RELATIVE = "report/neurips_2025.sty"
OFFICIAL_PROVENANCE_RELATIVE = "report/OFFICIAL_TEMPLATE_PROVENANCE.json"
OFFICIAL_STYLE_SHA256 = "67a960f35da6fecb5a0ce0202fa400459f4c8bb6c294f1a3fdd82fbbd39244c2"
BEST_ROLES = ("best_accuracy", "best_speed", "best_overall")
OPTIMIZED = (
    "manual_instruct",
    "manual_thinking",
    "evolved_instruct",
    "evolved_thinking",
)
ALL_PROGRAMS = ("baseline_instruct", "baseline_thinking", *OPTIMIZED)
FORMAL_PROGRAMS = ("baseline_instruct", *OPTIMIZED)


class FinalizeError(RuntimeError):
    """Formal evidence is absent, incomplete, or inconsistent."""


def default_project() -> Path:
    here = Path(__file__).resolve(strict=True).parent.parent
    if here.name != PROJECT_NAME:
        raise FinalizeError("cannot locate exact project root")
    return here


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise FinalizeError(f"non-finite JSON constant: {value}")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FinalizeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalizeError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalizeError(f"JSON root must be an object: {path}")
    return value


def _relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise FinalizeError("artifact path is not canonical")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != value:
        raise FinalizeError(f"unsafe artifact path: {value!r}")
    return value


def _regular_inside(project: Path, relative: str) -> Path:
    relative = _relative(relative)
    path = project / relative
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise FinalizeError(f"missing artifact: {relative}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FinalizeError(f"artifact is not a physical regular file: {relative}")
    if not path.resolve(strict=True).is_relative_to(project):
        raise FinalizeError(f"artifact escaped project: {relative}")
    if metadata.st_size <= 0:
        raise FinalizeError(f"artifact is empty: {relative}")
    return path


def file_record(project: Path, relative: str) -> dict[str, object]:
    path = _regular_inside(project, relative)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalizeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FinalizeError(f"{name} must be finite")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FinalizeError(f"{name} must be a nonnegative integer")
    return value


def recompute_best(metrics: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    best_accuracy = min(
        OPTIMIZED,
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
        name for name in OPTIMIZED
        if int(metrics[name]["dev128"]["error_count"]) == 0
        and int(metrics[name]["dev128"]["correct_count"]) >= floor_correct
    ]
    if not speed_eligible:
        raise FinalizeError("no optimized program meets frozen speed eligibility")
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
        name for name in OPTIMIZED
        if int(metrics[name]["dev128"]["error_count"]) == 0
        and int(metrics[name]["dev128"]["correct_count"]) >= baseline_correct
    ]
    if not overall_eligible:
        raise FinalizeError("no optimized program meets frozen overall eligibility")
    maximum_correct = max(
        int(metrics[name]["dev128"]["correct_count"]) for name in overall_eligible
    )
    minimum_latency = min(
        float(metrics[name]["speed32"]["median_seconds"]) for name in overall_eligible
    )
    scores = {}
    for name in overall_eligible:
        accuracy_component = int(metrics[name]["dev128"]["correct_count"]) / maximum_correct
        speed_component = minimum_latency / float(metrics[name]["speed32"]["median_seconds"])
        scores[name] = 2 * accuracy_component * speed_component / (
            accuracy_component + speed_component
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


def validate_aggregation(aggregation: Mapping[str, object]) -> dict[str, Any]:
    if set(aggregation) != {"metrics", "best", "baseline_thinking_dev128"}:
        raise FinalizeError("formal aggregation fields differ")
    raw_metrics = aggregation["metrics"]
    if not isinstance(raw_metrics, dict) or set(raw_metrics) != set(FORMAL_PROGRAMS):
        raise FinalizeError("formal metric program set differs")
    metrics: dict[str, Any] = {}
    for program in FORMAL_PROGRAMS:
        value = raw_metrics[program]
        if not isinstance(value, dict) or set(value) != {"source_sha256", "dev128", "speed32"}:
            raise FinalizeError(f"invalid metrics for {program}")
        source_hash = value["source_sha256"]
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise FinalizeError(f"invalid source hash for {program}")
        dev = value["dev128"]
        speed = value["speed32"]
        if not isinstance(dev, dict) or not isinstance(speed, dict):
            raise FinalizeError(f"invalid metric groups for {program}")
        sample_count = _nonnegative_int(dev.get("sample_count"), f"{program} samples")
        correct = _nonnegative_int(dev.get("correct_count"), f"{program} correct")
        errors = _nonnegative_int(dev.get("error_count"), f"{program} errors")
        accuracy = _finite_float(dev.get("accuracy"), f"{program} accuracy")
        if sample_count != 128 or correct > sample_count or errors > sample_count:
            raise FinalizeError(f"invalid Dev128 counts for {program}")
        if accuracy != correct / sample_count:
            raise FinalizeError(f"accuracy/count mismatch for {program}")
        if _nonnegative_int(speed.get("repeat_count"), f"{program} repeats") != 3:
            raise FinalizeError(f"speed repeat count differs for {program}")
        if _nonnegative_int(speed.get("sample_count"), f"{program} speed samples") != 32:
            raise FinalizeError(f"speed sample count differs for {program}")
        median = _finite_float(speed.get("median_seconds"), f"{program} median")
        p95 = _finite_float(speed.get("type7_p95_seconds"), f"{program} p95")
        _nonnegative_int(speed.get("generated_token_median_sum"), f"{program} speed tokens")
        _finite_float(speed.get("measured_seconds"), f"{program} measured seconds")
        qps = _finite_float(speed.get("throughput_qps"), f"{program} throughput")
        if median <= 0 or p95 <= 0 or qps <= 0:
            raise FinalizeError(f"nonpositive speed metric for {program}")
        metrics[program] = value
    censored = aggregation["baseline_thinking_dev128"]
    expected_censored = {
        "status": "TIMEOUT_CENSORED",
        "timeout_seconds": 7200,
        "accuracy_available": False,
        "source_campaign_id": "6beb857c-ef2c-4e0a-a6e6-34abee7f4a13",
    }
    if censored != expected_censored:
        raise FinalizeError("baseline-Thinking censoring facts differ")
    best = aggregation["best"]
    if not isinstance(best, dict):
        raise FinalizeError("best selection is not an object")
    for role in BEST_ROLES:
        if best.get(role) not in OPTIMIZED:
            raise FinalizeError(f"invalid {role} selection")
    recomputed = recompute_best(metrics)
    if best != recomputed:
        raise FinalizeError("best selection does not match frozen rules")
    return {"metrics": metrics, "best": best, "baseline_thinking_dev128": censored}


def _tex_program(name: str) -> str:
    return name.replace("_", r"\_")


def render_formal_results(aggregation: Mapping[str, object]) -> bytes:
    checked = validate_aggregation(aggregation)
    metrics = checked["metrics"]
    lines = [
        "% Generated from the semantically verified Recovery-v2 composition; do not edit.",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Recovery-v2 formal validation with authorized whole-set Speed32 replacement. Accuracy is exact match on Dev128; latency is the median of the 32 per-query medians over three steady-state repeats.}",
        r"\label{tab:formal-results}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "Program & Correct/128 & Errors & Median (s) & Type-7 p95 (s) " + r"\\",
        r"\midrule",
    ]
    for program in FORMAL_PROGRAMS:
        dev = metrics[program]["dev128"]
        speed = metrics[program]["speed32"]
        lines.append(
            f"{_tex_program(program)} & {dev['correct_count']}/128 & {dev['error_count']} & "
            f"{float(speed['median_seconds']):.6f} & {float(speed['type7_p95_seconds']):.6f} " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering\small",
            r"\caption{Pre-registered optimized-program selections.}",
            r"\label{tab:best-programs}",
            r"\begin{tabular}{ll}",
            r"\toprule",
            "Role & Selected program " + r"\\",
            r"\midrule",
        ]
    )
    for role in BEST_ROLES:
        lines.append(f"{_tex_program(role)} & {_tex_program(str(checked['best'][role]))} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return ("\n".join(lines)).encode("utf-8")


def build_number_trace(
    *, campaign_id: str, summary: Mapping[str, object], summary_sha256: str
) -> dict[str, object]:
    aggregation = summary.get("aggregation")
    if not isinstance(aggregation, dict):
        raise FinalizeError("campaign summary aggregation is invalid")
    checked = validate_aggregation(aggregation)
    return {
        "trace_version": "phase3-recovery-report-number-trace-v2",
        "protocol_id": PROTOCOL_ID,
        "campaign_id": campaign_id,
        "campaign_summary_sha256": summary_sha256,
        "baseline_thinking": {
            "status": "TIMEOUT_CENSORED",
            "timeout_seconds": 7200,
            "accuracy_available": False,
        },
        "metrics": checked["metrics"],
        "best": checked["best"],
    }


def _encoded_json(document: Mapping[str, object]) -> bytes:
    return (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _ensure_output(path: Path, payload: bytes) -> None:
    from evaluation.atomic_artifacts import write_bytes_noreplace

    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FinalizeError(f"existing final artifact differs: {path.name}")
        return
    write_bytes_noreplace(path, payload, 0o644)


def _walk_records(project: Path, root: Path) -> list[dict[str, object]]:
    if root.is_symlink() or not root.is_dir():
        raise FinalizeError("campaign evidence root is unsafe")
    records = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        base = Path(current)
        for name in directories:
            path = base / name
            if path.is_symlink():
                raise FinalizeError(f"campaign directory symlink: {path}")
        for name in filenames:
            path = base / name
            relative = path.relative_to(project).as_posix()
            records.append(file_record(project, relative))
    return sorted(records, key=lambda item: str(item["path"]))


def finalize(project: Path, campaign_id: str) -> dict[str, object]:
    project = project.resolve(strict=True)
    if project != default_project():
        raise FinalizeError("production finalizer requires the exact project root")
    try:
        uuid.UUID(campaign_id)
    except ValueError as exc:
        raise FinalizeError("campaign ID is not a UUID") from exc
    from evaluation import phase3_recovery as recovery

    directory = recovery.campaign_directory(campaign_id)
    plan_path = directory / "campaign_plan.json"
    replacement_mode = (directory / REPLACEMENT_SUMMARY_NAME).is_file()
    old_directory: Path | None = None
    if replacement_mode:
        from evaluation import phase3_speed_replacement as replacement

        verified = replacement.check_replacement(directory)
        if verified.get("status") != "COMPLETE" or verified.get("result_count") != 15:
            raise FinalizeError("full Speed32 replacement is not formally complete")
        summary_path = directory / REPLACEMENT_SUMMARY_NAME
        completed_path = directory / REPLACEMENT_COMPLETED_NAME
    else:
        verified = recovery.check_campaign(directory)
        if verified.get("status") != "COMPLETE" or verified.get("result_count") != 26:
            raise FinalizeError("Recovery-v2 campaign is not formally complete")
        summary_path = directory / "campaign_summary.json"
        completed_path = directory / "completed.json"

    summary = load_json(summary_path)
    aggregation = summary.get("aggregation")
    if not isinstance(aggregation, dict):
        raise FinalizeError("campaign aggregation is invalid")
    checked = validate_aggregation(aggregation)
    if replacement_mode:
        old_campaign_id = summary.get("old_campaign_id")
        if not isinstance(old_campaign_id, str):
            raise FinalizeError("replacement old campaign ID is invalid")
        try:
            uuid.UUID(old_campaign_id)
        except ValueError as exc:
            raise FinalizeError("replacement old campaign ID is not a UUID") from exc
        old_directory = recovery.campaign_directory(old_campaign_id)
        anchors = {
            "old plan": (old_directory / "campaign_plan.json", OLD_PLAN_SHA256),
            "old summary": (old_directory / "campaign_summary.json", OLD_SUMMARY_SHA256),
            "old completion": (old_directory / "completed.json", OLD_COMPLETED_SHA256),
            "contamination audit": (
                project / CONTAMINATION_AUDIT_RELATIVE,
                CONTAMINATION_AUDIT_SHA256,
            ),
        }
        for name, (anchor_path, expected_hash) in anchors.items():
            relative = anchor_path.relative_to(project).as_posix()
            if sha256_file(_regular_inside(project, relative)) != expected_hash:
                raise FinalizeError(f"{name} hash differs")

    table_payload = render_formal_results(aggregation)
    trace = build_number_trace(
        campaign_id=campaign_id,
        summary=summary,
        summary_sha256=sha256_file(summary_path),
    )
    _ensure_output(project / FORMAL_TABLE_RELATIVE, table_payload)
    _ensure_output(project / NUMBER_TRACE_RELATIVE, _encoded_json(trace))
    for role in BEST_ROLES:
        selected = str(checked["best"][role])
        selected_payload = _regular_inside(project, f"{selected}.py").read_bytes()
        _ensure_output(project / f"{role}.py", selected_payload)

    evidence = _walk_records(project, directory)
    if old_directory is not None:
        evidence.extend(_walk_records(project, old_directory))
    static_artifacts = [
        "evaluation/recovery_v2/protocol.json",
        "evaluation/recovery_v2/candidate_manifest.json",
        "evaluation/recovery_v2/data_manifest.json",
        "evaluation/recovery_v2/selection_rules.json",
        "evaluation/recovery_v2/recovery_authorization.json",
        "evaluation/phase3_recovery.py",
        "evaluation/recovery_finalize.py",
        "data/phase3/manifests/dev128.json",
        "data/phase3/manifests/evolution_used.json",
        "data/phase3/manifests/single1.json",
        "data/phase3/manifests/speed32.json",
        "data/phase3/manifests/untouched_complement.json",
        "data/phase3/manifests/warmup1.json",
        "report/main.tex",
        "report/references.bib",
        OFFICIAL_STYLE_RELATIVE,
        OFFICIAL_PROVENANCE_RELATIVE,
        FORMAL_TABLE_RELATIVE,
        NUMBER_TRACE_RELATIVE,
        "requirements.lock.txt",
        *[f"{name}.py" for name in ALL_PROGRAMS],
        *[f"{role}.py" for role in BEST_ROLES],
    ]
    if replacement_mode:
        static_artifacts.extend(
            [
                "evaluation/phase3_speed_replacement.py",
                "evaluation/recovery_v2/speed32_replacement_authorization.json",
                CONTAMINATION_AUDIT_RELATIVE,
            ]
        )
    for relative in static_artifacts:
        evidence.append(file_record(project, relative))
    unique = {str(item["path"]): item for item in evidence}
    manifest: dict[str, object] = {
        "manifest_version": (
            FINAL_MANIFEST_REPLACEMENT_V3 if replacement_mode else FINAL_MANIFEST_V2
        ),
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "formal_complete": True,
        "campaign_id": campaign_id,
        "source_identity": summary["source_identity"],
        "result_count": 26,
        "campaign_plan": file_record(
            project, plan_path.relative_to(project).as_posix()
        ),
        "campaign_summary": file_record(
            project, summary_path.relative_to(project).as_posix()
        ),
        "campaign_completed": file_record(
            project, completed_path.relative_to(project).as_posix()
        ),
        "official_style_sha256": OFFICIAL_STYLE_SHA256,
        "best": checked["best"],
        "artifacts": [unique[name] for name in sorted(unique)],
    }
    if replacement_mode:
        assert old_directory is not None
        manifest["composition"] = {
            "kind": "full_speed32_replacement",
            "old_campaign_id": old_directory.name,
            "retained_result_count": 11,
            "replacement_speed_result_count": 15,
            "old_speed32_results_used": False,
            "partial_replacement_used": False,
            "retained_campaign_plan": file_record(
                project,
                (old_directory / "campaign_plan.json").relative_to(project).as_posix(),
            ),
            "retained_campaign_summary": file_record(
                project,
                (old_directory / "campaign_summary.json").relative_to(project).as_posix(),
            ),
            "retained_campaign_completed": file_record(
                project,
                (old_directory / "completed.json").relative_to(project).as_posix(),
            ),
            "speed_measurement_completion": file_record(
                project,
                (directory / REPLACEMENT_MEASUREMENTS_NAME).relative_to(project).as_posix(),
            ),
            "replacement_marker": file_record(
                project,
                (directory / REPLACEMENT_MARKER_NAME).relative_to(project).as_posix(),
            ),
            "contamination_audit": file_record(project, CONTAMINATION_AUDIT_RELATIVE),
        }
    manifest_path = project / FINAL_MANIFEST_RELATIVE
    _ensure_output(manifest_path, _encoded_json(manifest))
    return verify_final_manifest(project, manifest_path)

def verify_final_manifest(project: Path, manifest_path: Path | None = None) -> dict[str, object]:
    project = project.resolve(strict=True)
    path = manifest_path or (project / FINAL_MANIFEST_RELATIVE)
    document = load_json(_regular_inside(project, path.relative_to(project).as_posix()))
    version = document.get("manifest_version")
    replacement_mode = version == FINAL_MANIFEST_REPLACEMENT_V3
    if version not in {FINAL_MANIFEST_V2, FINAL_MANIFEST_REPLACEMENT_V3}:
        raise FinalizeError("final manifest version differs")
    expected_fields = {
        "manifest_version", "protocol_id", "status", "formal_complete", "campaign_id",
        "source_identity", "result_count", "campaign_plan", "campaign_summary",
        "campaign_completed", "official_style_sha256", "best", "artifacts",
    }
    if replacement_mode:
        expected_fields.add("composition")
    if set(document) != expected_fields:
        raise FinalizeError("final manifest fields differ")
    if document["protocol_id"] != PROTOCOL_ID or document["status"] != "COMPLETE":
        raise FinalizeError("formal status/protocol differs")
    if document["formal_complete"] is not True or document["result_count"] != 26:
        raise FinalizeError("formal completion facts differ")
    if document["official_style_sha256"] != OFFICIAL_STYLE_SHA256:
        raise FinalizeError("official style hash differs")
    try:
        uuid.UUID(str(document["campaign_id"]))
    except ValueError as exc:
        raise FinalizeError("final campaign ID is invalid") from exc
    identity = document["source_identity"]
    if not isinstance(identity, dict) or set(identity) != {"commit", "tree"}:
        raise FinalizeError("final source identity fields differ")
    if any(
        not isinstance(identity[key], str)
        or re.fullmatch(r"[0-9a-f]{40}", identity[key]) is None
        for key in identity
    ):
        raise FinalizeError("final source identity is invalid")

    records = document["artifacts"]
    if not isinstance(records, list) or not records:
        raise FinalizeError("final artifact list is empty")
    seen: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
            raise FinalizeError("final artifact record fields differ")
        relative = _relative(record["path"])
        if relative in seen:
            raise FinalizeError(f"duplicate final artifact: {relative}")
        digest = record["sha256"]
        size = record["size_bytes"]
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise FinalizeError(f"invalid final artifact metadata: {relative}")
        artifact = _regular_inside(project, relative)
        if artifact.stat().st_size != size or sha256_file(artifact) != digest:
            raise FinalizeError(f"final artifact mismatch: {relative}")
        seen[relative] = record

    def bound_record(value: object, label: str) -> dict[str, object]:
        if not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}:
            raise FinalizeError(f"{label} record fields differ")
        relative = value.get("path")
        if not isinstance(relative, str) or value != seen.get(relative):
            raise FinalizeError(f"{label} record is not in artifact closure")
        return value

    plan_record = bound_record(document["campaign_plan"], "campaign_plan")
    summary_record = bound_record(document["campaign_summary"], "campaign_summary")
    completed_record = bound_record(document["campaign_completed"], "campaign_completed")
    required = {
        OFFICIAL_STYLE_RELATIVE,
        OFFICIAL_PROVENANCE_RELATIVE,
        FORMAL_TABLE_RELATIVE,
        NUMBER_TRACE_RELATIVE,
        *[f"{role}.py" for role in BEST_ROLES],
    }
    if replacement_mode:
        required.update(
            {
                "evaluation/recovery_finalize.py",
                "evaluation/phase3_speed_replacement.py",
                "evaluation/recovery_v2/speed32_replacement_authorization.json",
                CONTAMINATION_AUDIT_RELATIVE,
                "data/phase3/manifests/dev128.json",
                "data/phase3/manifests/single1.json",
                "data/phase3/manifests/speed32.json",
                "data/phase3/manifests/warmup1.json",
            }
        )
    if not required.issubset(seen):
        raise FinalizeError("final artifact set lacks required outputs")

    summary = load_json(project / str(summary_record["path"]))
    if summary.get("source_identity") != identity:
        raise FinalizeError("summary/final source identity differs")
    if replacement_mode:
        composition = document.get("composition")
        expected_composition_fields = {
            "kind", "old_campaign_id", "retained_result_count",
            "replacement_speed_result_count", "old_speed32_results_used",
            "partial_replacement_used", "retained_campaign_plan",
            "retained_campaign_summary", "retained_campaign_completed",
            "speed_measurement_completion", "replacement_marker",
            "contamination_audit",
        }
        if not isinstance(composition, dict) or set(composition) != expected_composition_fields:
            raise FinalizeError("replacement composition fields differ")
        expected_scalars = {
            "kind": "full_speed32_replacement",
            "retained_result_count": 11,
            "replacement_speed_result_count": 15,
            "old_speed32_results_used": False,
            "partial_replacement_used": False,
        }
        for key, expected in expected_scalars.items():
            if composition.get(key) != expected or type(composition.get(key)) is not type(expected):
                raise FinalizeError(f"replacement composition {key} differs")
        old_campaign_id = composition.get("old_campaign_id")
        try:
            uuid.UUID(str(old_campaign_id))
        except ValueError as exc:
            raise FinalizeError("replacement retained campaign ID is invalid") from exc
        retained_plan = bound_record(
            composition["retained_campaign_plan"], "retained_campaign_plan"
        )
        retained_summary = bound_record(
            composition["retained_campaign_summary"], "retained_campaign_summary"
        )
        retained_completed = bound_record(
            composition["retained_campaign_completed"], "retained_campaign_completed"
        )
        measurement_record = bound_record(
            composition["speed_measurement_completion"], "speed_measurement_completion"
        )
        marker_record = bound_record(composition["replacement_marker"], "replacement_marker")
        audit_record = bound_record(composition["contamination_audit"], "contamination_audit")
        fixed_hashes = {
            "retained plan": (retained_plan, OLD_PLAN_SHA256),
            "retained summary": (retained_summary, OLD_SUMMARY_SHA256),
            "retained completion": (retained_completed, OLD_COMPLETED_SHA256),
            "contamination audit": (audit_record, CONTAMINATION_AUDIT_SHA256),
        }
        for label, (record, expected_hash) in fixed_hashes.items():
            if record["sha256"] != expected_hash:
                raise FinalizeError(f"{label} fixed hash differs")

        expected_summary_fields = {
            "summary_version", "protocol_id", "replacement_campaign_id",
            "source_identity", "old_campaign_id", "composition",
            "campaign_plan_sha256", "replacement_marker_sha256",
            "contamination_audit_sha256", "speed_measurement_completion_sha256",
            "retained_result_sha256", "replacement_speed_result_sha256",
            "aggregation",
        }
        if set(summary) != expected_summary_fields:
            raise FinalizeError("replacement summary fields differ")
        if (
            summary["summary_version"] != "phase3-speed32-replacement-summary-v1"
            or summary["protocol_id"] != PROTOCOL_ID
            or summary["replacement_campaign_id"] != document["campaign_id"]
            or summary["old_campaign_id"] != old_campaign_id
            or summary["campaign_plan_sha256"] != plan_record["sha256"]
            or summary["replacement_marker_sha256"] != marker_record["sha256"]
            or summary["contamination_audit_sha256"] != audit_record["sha256"]
            or summary["speed_measurement_completion_sha256"]
            != measurement_record["sha256"]
        ):
            raise FinalizeError("replacement summary hash chain differs")
        expected_summary_composition = {
            "retained_old_indices": list(range(1, 12)),
            "replacement_new_indices": list(range(12, 27)),
            "old_speed32_results_used": False,
            "partial_replacement_used": False,
        }
        if summary["composition"] != expected_summary_composition:
            raise FinalizeError("replacement summary composition differs")

        completed = load_json(project / str(completed_record["path"]))
        expected_completed_fields = {
            "completed_version", "protocol_id", "replacement_campaign_id",
            "source_identity", "campaign_plan_sha256", "replacement_summary_sha256",
            "contamination_audit_sha256", "speed_measurement_completion_sha256",
            "old_speed32_entire_set_qualifying", "replacement_speed_result_count",
            "completed_at_utc",
        }
        if not isinstance(completed, dict) or set(completed) != expected_completed_fields:
            raise FinalizeError("replacement completion fields differ")
        if (
            completed["completed_version"] != "phase3-speed32-replacement-completed-v1"
            or completed["protocol_id"] != PROTOCOL_ID
            or completed["replacement_campaign_id"] != document["campaign_id"]
            or completed["source_identity"] != identity
            or completed["campaign_plan_sha256"] != plan_record["sha256"]
            or completed["replacement_summary_sha256"] != summary_record["sha256"]
            or completed["contamination_audit_sha256"] != audit_record["sha256"]
            or completed["speed_measurement_completion_sha256"]
            != measurement_record["sha256"]
            or completed["old_speed32_entire_set_qualifying"] is not False
            or completed["replacement_speed_result_count"] != 15
            or not isinstance(completed["completed_at_utc"], str)
            or not completed["completed_at_utc"]
        ):
            raise FinalizeError("replacement completion hash chain differs")

        measurement = load_json(project / str(measurement_record["path"]))
        expected_measurement_fields = {
            "measurement_completion_version", "protocol_id", "replacement_campaign_id",
            "source_identity", "campaign_plan_sha256", "replacement_marker_sha256",
            "contamination_audit_sha256", "authorized_indices", "result_sha256",
            "old_speed32_results_used", "partial_replacement_used", "completed_at_utc",
        }
        if set(measurement) != expected_measurement_fields:
            raise FinalizeError("replacement measurement fields differ")
        if (
            measurement["measurement_completion_version"]
            != "phase3-speed32-measurements-complete-v1"
            or measurement["protocol_id"] != PROTOCOL_ID
            or measurement["replacement_campaign_id"] != document["campaign_id"]
            or measurement["source_identity"] != identity
            or measurement["campaign_plan_sha256"] != plan_record["sha256"]
            or measurement["replacement_marker_sha256"] != marker_record["sha256"]
            or measurement["contamination_audit_sha256"] != audit_record["sha256"]
            or measurement["authorized_indices"] != list(range(12, 27))
            or measurement["old_speed32_results_used"] is not False
            or measurement["partial_replacement_used"] is not False
            or not isinstance(measurement["completed_at_utc"], str)
            or not measurement["completed_at_utc"]
        ):
            raise FinalizeError("replacement measurement hash chain differs")

        marker = load_json(project / str(marker_record["path"]))
        marker_expectations = {
            "marker_version": "phase3-speed32-replacement-marker-v1",
            "protocol_id": PROTOCOL_ID,
            "campaign_id": document["campaign_id"],
            "supersedes_speed32_campaign_id": old_campaign_id,
            "retains_old_schedule_indices": list(range(1, 12)),
            "authorized_replacement_indices": list(range(12, 27)),
            "old_speed32_entire_set_qualifying": False,
            "partial_replacement_allowed": False,
            "old_new_speed_mixing_allowed": False,
            "source_identity": identity,
            "campaign_plan_sha256": plan_record["sha256"],
            "contamination_audit_path": CONTAMINATION_AUDIT_RELATIVE,
            "contamination_audit_sha256": audit_record["sha256"],
        }
        for key, expected in marker_expectations.items():
            if marker.get(key) != expected:
                raise FinalizeError(f"replacement marker {key} differs")
        gpu_policy = marker.get("gpu_policy")
        if gpu_policy != {
            "cuda_visible_devices": "0",
            "gpu1_queried_or_used": False,
            "host": "hopper-25",
            "logical_device": "cuda:0",
            "physical_gpu_index": 0,
            "strictly_serial": True,
            "visible_device_count": 1,
        }:
            raise FinalizeError("replacement GPU policy differs")
        authorization_path = marker.get("authorization_path")
        if (
            authorization_path != "evaluation/recovery_v2/speed32_replacement_authorization.json"
            or marker.get("authorization_sha256")
            != seen[authorization_path]["sha256"]
        ):
            raise FinalizeError("replacement authorization binding differs")
        frozen_hashes = marker.get("frozen_hashes")
        frozen_paths = {
            "phase3_recovery": "evaluation/phase3_recovery.py",
            "protocol": "evaluation/recovery_v2/protocol.json",
            "candidate_manifest": "evaluation/recovery_v2/candidate_manifest.json",
            "data_manifest": "evaluation/recovery_v2/data_manifest.json",
            "selection_rules": "evaluation/recovery_v2/selection_rules.json",
            "recovery_authorization": "evaluation/recovery_v2/recovery_authorization.json",
            "speed32_manifest": "data/phase3/manifests/speed32.json",
            "warmup1_manifest": "data/phase3/manifests/warmup1.json",
        }
        if not isinstance(frozen_hashes, dict) or set(frozen_hashes) != set(frozen_paths):
            raise FinalizeError("replacement frozen hash fields differ")
        for key, relative in frozen_paths.items():
            if frozen_hashes[key] != seen[relative]["sha256"]:
                raise FinalizeError(f"replacement frozen hash differs: {key}")

        new_plan = load_json(project / str(plan_record["path"]))
        old_plan = load_json(project / str(retained_plan["path"]))
        new_schedule = new_plan.get("schedule")
        old_schedule = old_plan.get("schedule")
        if (
            new_plan.get("campaign_id") != document["campaign_id"]
            or new_plan.get("source_identity") != identity
            or old_plan.get("campaign_id") != old_campaign_id
            or not isinstance(new_schedule, list)
            or not isinstance(old_schedule, list)
            or len(new_schedule) != 26
            or len(old_schedule) != 26
        ):
            raise FinalizeError("replacement schedule headers differ")
        retained_hashes = summary.get("retained_result_sha256")
        replacement_hashes = summary.get("replacement_speed_result_sha256")
        measurement_hashes = measurement.get("result_sha256")
        if (
            not isinstance(retained_hashes, dict)
            or set(retained_hashes) != {str(i) for i in range(1, 12)}
            or not isinstance(replacement_hashes, dict)
            or set(replacement_hashes) != {str(i) for i in range(12, 27)}
            or measurement_hashes != replacement_hashes
        ):
            raise FinalizeError("replacement result hash maps differ")
        old_prefix = PurePosixPath(str(retained_plan["path"])).parent
        new_prefix = PurePosixPath(str(plan_record["path"])).parent
        selected_paths: list[str] = []
        for index, (old_entry, new_entry) in enumerate(
            zip(old_schedule, new_schedule, strict=True), start=1
        ):
            if not isinstance(old_entry, dict) or not isinstance(new_entry, dict):
                raise FinalizeError("replacement schedule entry is invalid")
            identity_tuple = ("index", "program", "manifest", "repeat")
            if (
                old_entry.get("index") != index
                or new_entry.get("index") != index
                or tuple(old_entry.get(key) for key in identity_tuple)
                != tuple(new_entry.get(key) for key in identity_tuple)
            ):
                raise FinalizeError(f"replacement schedule entry {index} differs")
            chosen_entry = old_entry if index < 12 else new_entry
            result_name = chosen_entry.get("result")
            if (
                not isinstance(result_name, str)
                or PurePosixPath(result_name).name != result_name
            ):
                raise FinalizeError(f"replacement result path {index} is invalid")
            prefix = old_prefix if index < 12 else new_prefix
            relative = (prefix / result_name).as_posix()
            record = seen.get(relative)
            expected_hash = (
                retained_hashes[str(index)]
                if index < 12
                else replacement_hashes[str(index)]
            )
            if record is None or record["sha256"] != expected_hash:
                raise FinalizeError(f"composed formal result {index} differs")
            selected_paths.append(relative)
        if len(selected_paths) != 26 or len(set(selected_paths)) != 26:
            raise FinalizeError("composed formal result set is not exactly 26")
    else:
        campaign_prefix = str(PurePosixPath(str(summary_record["path"])).parent) + "/"
        result_records = {
            name: record
            for name, record in seen.items()
            if name.startswith(campaign_prefix)
            and re.fullmatch(
                r"[0-9]{2}\.[a-z0-9_]+\.(single1|dev128|speed32)\.r[1-3]\.json",
                PurePosixPath(name).name,
            )
        }
        if len(result_records) != 26:
            raise FinalizeError("sealed campaign does not contain exactly 26 results")
        result_hashes = summary.get("result_sha256")
        if not isinstance(result_hashes, dict) or set(result_hashes) != {
            PurePosixPath(name).name for name in result_records
        }:
            raise FinalizeError("summary result hash set differs")
        for name, record in result_records.items():
            if result_hashes[PurePosixPath(name).name] != record["sha256"]:
                raise FinalizeError("summary/result artifact hash differs")

    aggregation = summary.get("aggregation")
    if not isinstance(aggregation, dict):
        raise FinalizeError("sealed aggregation is invalid")
    checked = validate_aggregation(aggregation)
    if checked["best"] != document["best"]:
        raise FinalizeError("sealed best selection differs")
    expected_table = render_formal_results(aggregation)
    if (project / FORMAL_TABLE_RELATIVE).read_bytes() != expected_table:
        raise FinalizeError("generated formal table differs")
    trace = load_json(project / NUMBER_TRACE_RELATIVE)
    expected_trace = build_number_trace(
        campaign_id=str(document["campaign_id"]),
        summary=summary,
        summary_sha256=str(summary_record["sha256"]),
    )
    if trace != expected_trace:
        raise FinalizeError("generated number trace differs")

    for role in BEST_ROLES:
        selected = str(checked["best"][role])
        if (project / f"{role}.py").read_bytes() != (
            project / f"{selected}.py"
        ).read_bytes():
            raise FinalizeError(f"{role} is not an exact selected-program copy")
    if sha256_file(project / OFFICIAL_STYLE_RELATIVE) != OFFICIAL_STYLE_SHA256:
        raise FinalizeError("official NeurIPS style bytes differ")
    provenance = load_json(project / OFFICIAL_PROVENANCE_RELATIVE)
    if provenance != {
        "source_url": "https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip",
        "source_zip_relative": "source/neurips2025_styles.zip",
        "source_zip_sha256":
            "72032130232955b785cd622b7c52d893e3166f30a90e1e6279d0a28e8373beef",
        "source_zip_size_bytes": 188273,
        "style_relative": OFFICIAL_STYLE_RELATIVE,
        "style_sha256": OFFICIAL_STYLE_SHA256,
        "usage":
            "official NeurIPS 2025 preprint style; fallback styles are forbidden",
    }:
        raise FinalizeError("official NeurIPS template provenance differs")
    return {
        "status": "COMPLETE",
        "formal_complete": True,
        "campaign_id": document["campaign_id"],
        "result_count": 26,
        "artifact_count": len(records),
        "manifest_version": version,
        "manifest_sha256": sha256_file(path),
        "best": document["best"],
        "torch_imported": "torch" in sys.modules,
        "gpu_called": False,
    }

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("finalize", "check"))
    parser.add_argument("--campaign-id")
    parser.add_argument("--project", type=Path, default=default_project())
    args = parser.parse_args(argv)
    try:
        if args.command == "finalize":
            if not args.campaign_id:
                raise FinalizeError("--campaign-id is required for finalize")
            result = finalize(args.project, args.campaign_id)
        else:
            result = verify_final_manifest(args.project)
    except (FileExistsError, FinalizeError, KeyError, OSError, ValueError) as exc:
        print(f"Recovery-v2 finalization failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
