"""Strict machine rendering and checking for final fresh-run evidence."""
from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from evaluation.fresh_semantics import (
    FreshSemanticError,
    verify_result_semantics,
)

BEGIN = "<!-- FINAL_FRESH_REPRODUCTION_ACTIVE_BEGIN -->"
END = "<!-- FINAL_FRESH_REPRODUCTION_ACTIVE_END -->"
PENDING = """<!-- FINAL_FRESH_REPRODUCTION_ACTIVE_BEGIN -->
## Active final fresh reproduction

Status: `PENDING_FINAL_FRESH_REPRODUCTION`

No fresh result is active until the frozen dual-GPU run commits a strict receipt.
<!-- FINAL_FRESH_REPRODUCTION_ACTIVE_END -->"""
LABEL = "FINAL_FRESH_REPRODUCTION_ONLY"


class FreshEvidenceError(RuntimeError):
    pass


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise FreshEvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject(value: str) -> object:
    raise FreshEvidenceError(f"non-finite JSON constant: {value}")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject,
        object_pairs_hook=_pairs,
    )
    if not isinstance(value, dict):
        raise FreshEvidenceError("receipt root is not an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FreshEvidenceError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise FreshEvidenceError(f"{label} is not finite/positive")
    return result


def validate_receipt(receipt: Mapping[str, object]) -> None:
    required = {
        "schema_version", "label", "status", "run_id", "source", "plan",
        "formal_manifest", "started_at_utc", "completed_at_utc", "lanes",
        "results", "formal_metrics_affected", "selection_affected",
        "expected_dev128_correct", "best_source_identity", "lane_receipts",
        "orchestrator_logs",
    }
    if set(receipt) != required:
        raise FreshEvidenceError("receipt fields differ")
    if (
        receipt["schema_version"] != "final-fresh-reproduction-receipt-v1"
        or receipt["label"] != LABEL
        or receipt["status"] != "COMPLETE"
        or receipt["formal_metrics_affected"] is not False
        or receipt["selection_affected"] is not False
    ):
        raise FreshEvidenceError("receipt status/semantics differ")
    try:
        if str(uuid.UUID(str(receipt["run_id"]), version=4)) != receipt["run_id"]:
            raise ValueError
    except ValueError as exc:
        raise FreshEvidenceError("run ID is not canonical UUIDv4") from exc
    source = receipt["source"]
    if not isinstance(source, dict) or set(source) != {"commit", "tree"}:
        raise FreshEvidenceError("source identity fields differ")
    if not re.fullmatch(r"[0-9a-f]{40}", str(source["commit"])) or not re.fullmatch(
        r"[0-9a-f]{40}", str(source["tree"])
    ):
        raise FreshEvidenceError("source identity format differs")
    for name in ("plan", "formal_manifest"):
        record = receipt[name]
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
            raise FreshEvidenceError(f"{name} record fields differ")
        if not re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"])):
            raise FreshEvidenceError(f"{name} hash differs")
        _finite(record["size_bytes"], f"{name} size", positive=True)
    for name in ("lane_receipts", "orchestrator_logs"):
        records = receipt[name]
        if not isinstance(records, list) or len(records) != 2:
            raise FreshEvidenceError(f"{name} count differs")
        for record in records:
            if not isinstance(record, dict) or set(record) != {
                "path", "sha256", "size_bytes"
            }:
                raise FreshEvidenceError(f"{name} record fields differ")
            if not re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"])):
                raise FreshEvidenceError(f"{name} hash differs")
            _finite(record["size_bytes"], f"{name} size", positive=True)
    results = receipt["results"]
    if not isinstance(results, list) or len(results) not in {8, 9}:
        raise FreshEvidenceError("fresh result count differs")
    roles: set[tuple[str, str]] = set()
    for item in results:
        required_item = {
            "program", "role", "physical_gpu_index", "sample_count", "correct",
            "num_errors", "total_time", "avg_time_per_query",
            "generated_token_count", "result", "console", "attempt_id",
            "program_source_sha256", "dependency_closure_sha256",
            "model_id", "model_revision", "processor_revision",
            "model_weight_sha256", "input_manifest_sha256", "start", "end",
        }
        if not isinstance(item, dict) or set(item) != required_item:
            raise FreshEvidenceError("fresh result fields differ")
        identity = (str(item["program"]), str(item["role"]))
        if identity in roles:
            raise FreshEvidenceError("duplicate fresh result identity")
        roles.add(identity)
        if item["physical_gpu_index"] not in {0, 1}:
            raise FreshEvidenceError("fresh physical GPU index differs")
        count = item["sample_count"]
        if item["role"] == "dev128" and count != 128:
            raise FreshEvidenceError("Dev128 sample count differs")
        if item["role"] == "single1" and count != 1:
            raise FreshEvidenceError("Single1 sample count differs")
        if item["num_errors"] != 0:
            raise FreshEvidenceError("fresh result contains errors")
        if not isinstance(item["correct"], int) or not 0 <= item["correct"] <= count:
            raise FreshEvidenceError("fresh correct count differs")
        _finite(item["total_time"], "total time", positive=True)
        _finite(item["avg_time_per_query"], "average time", positive=True)
        _finite(item["generated_token_count"], "generated tokens", positive=True)
        for key in (
            "program_source_sha256", "dependency_closure_sha256",
            "model_weight_sha256", "input_manifest_sha256",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", str(item[key])):
                raise FreshEvidenceError(f"{key} differs")
        for record_name in ("result", "console", "start", "end"):
            record = item[record_name]
            if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
                raise FreshEvidenceError(f"{record_name} record fields differ")
            if not re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"])):
                raise FreshEvidenceError(f"{record_name} hash differs")
            _finite(record["size_bytes"], f"{record_name} size", positive=True)
    required_dev = {
        ("baseline_instruct", "dev128"),
        ("manual_instruct", "dev128"),
        ("manual_thinking", "dev128"),
        ("evolved_instruct", "dev128"),
        ("evolved_thinking", "dev128"),
    }
    required_best = {
        ("best_accuracy", "single1"),
        ("best_speed", "single1"),
        ("best_overall", "single1"),
    }
    if not required_dev | required_best <= roles:
        raise FreshEvidenceError("required fresh roles are absent")
    expected = receipt["expected_dev128_correct"]
    actual = {
        item["program"]: item["correct"]
        for item in results if item["role"] == "dev128"
    }
    if expected != actual:
        raise FreshEvidenceError("expected/current Dev128 counts differ")



def _pure_relative(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise FreshEvidenceError(f"{label} path is not text")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise FreshEvidenceError(f"{label} path is unsafe")
    return pure


def _verify_record(
    base: Path,
    root: Path,
    record: object,
    label: str,
) -> Path:
    if not isinstance(record, dict) or set(record) != {
        "path", "sha256", "size_bytes"
    }:
        raise FreshEvidenceError(f"{label} record fields differ")
    pure = _pure_relative(record["path"], label)
    candidate = base.joinpath(*pure.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise FreshEvidenceError(f"{label} artifact is unsafe")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise FreshEvidenceError(f"{label} artifact escaped root") from exc
    if sha256(resolved) != record["sha256"] or resolved.stat().st_size != record["size_bytes"]:
        raise FreshEvidenceError(f"{label} artifact hash/size differs")
    return resolved


def verify_receipt_artifacts(
    root: Path,
    receipt_path: Path,
    receipt: Mapping[str, object],
) -> list[str]:
    """Re-hash every fresh artifact and reject orphan/swap/extra success evidence."""
    validate_receipt(receipt)
    root = root.resolve(strict=True)
    receipt_path = receipt_path.resolve(strict=True)
    try:
        receipt_relative = receipt_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise FreshEvidenceError("active receipt escaped root") from exc
    output = receipt_path.parent
    verified: dict[str, Path] = {receipt_relative: receipt_path}

    def bind(base: Path, record: object, label: str) -> Path:
        path = _verify_record(base, root, record, label)
        relative = path.relative_to(root).as_posix()
        if relative in verified:
            raise FreshEvidenceError(f"duplicate bound artifact: {relative}")
        verified[relative] = path
        return path

    formal_path = bind(root, receipt["formal_manifest"], "formal manifest")
    plan_path = bind(output, receipt["plan"], "fresh plan")
    plan = load(plan_path)
    if plan.get("source") != receipt["source"] or plan.get("formal_manifest") != receipt["formal_manifest"]:
        raise FreshEvidenceError("fresh plan source/formal binding differs")
    if sha256(formal_path) != receipt["formal_manifest"]["sha256"]:
        raise FreshEvidenceError("formal manifest binding differs")

    for index, record in enumerate(receipt["lane_receipts"]):
        bind(output, record, f"lane receipt {index}")
    for index, record in enumerate(receipt["orchestrator_logs"]):
        bind(output, record, f"orchestrator log {index}")

    lane_items: dict[int, dict[tuple[str, str], Mapping[str, object]]] = {0: {}, 1: {}}
    result_documents: dict[tuple[str, str], Mapping[str, object]] = {}
    for item in receipt["results"]:
        assert isinstance(item, dict)
        lane = int(item["physical_gpu_index"])
        identity = (str(item["program"]), str(item["role"]))
        if identity in lane_items[lane]:
            raise FreshEvidenceError("duplicate result in lane")
        lane_items[lane][identity] = item
        bound = {
            name: bind(output, item[name], f"{item['program']} {name}")
            for name in ("result", "console", "start", "end")
        }
        start = load(bound["start"])
        end = load(bound["end"])
        expected = {
            "attempt_id": item["attempt_id"],
            "run_id": receipt["run_id"],
            "program": item["program"],
            "role": item["role"],
            "physical_gpu_index": lane,
        }
        if any(start.get(key) != value or end.get(key) != value for key, value in expected.items()):
            raise FreshEvidenceError(f"fresh attempt identity differs: {item['program']}")
        if start.get("event") != "attempt_started" or end.get("event") != "attempt_ended":
            raise FreshEvidenceError(f"fresh attempt event differs: {item['program']}")
        if end.get("returncode") != 0 or end.get("outcome") != "COMPLETED":
            raise FreshEvidenceError(f"fresh attempt outcome differs: {item['program']}")
        if end.get("result") != item["result"] or end.get("console") != item["console"]:
            raise FreshEvidenceError(f"fresh attempt result/console swap: {item['program']}")
        result = load(bound["result"])
        if result.get("program") != item["program"] or result.get("num_evaluated") != item["sample_count"]:
            raise FreshEvidenceError(f"fresh result identity differs: {item['program']}")
        result_documents[identity] = result

    for lane in (0, 1):
        lane_record = receipt["lane_receipts"][lane]
        lane_path = output.joinpath(*_pure_relative(lane_record["path"], "lane receipt").parts)
        lane_document = load(lane_path)
        if lane_document.get("run_id") != receipt["run_id"] or lane_document.get("lane") != lane:
            raise FreshEvidenceError(f"lane receipt identity differs: {lane}")
        scheduled = plan.get("lanes", {}).get(str(lane))
        if not isinstance(scheduled, list):
            raise FreshEvidenceError(f"lane schedule differs: {lane}")
        try:
            expected_lane = [
                lane_items[lane][(str(task["program"]), str(task["role"]))]
                for task in scheduled
            ]
        except (KeyError, TypeError) as exc:
            raise FreshEvidenceError(f"lane schedule/result binding differs: {lane}") from exc
        if lane_document.get("results") != expected_lane:
            raise FreshEvidenceError(f"lane receipt result order/content differs: {lane}")
        lane_root = lane_path.parent
        allowed = {path for path in verified.values() if path.parent == lane_root}
        actual = {path.resolve(strict=True) for path in lane_root.iterdir() if path.is_file() and not path.is_symlink()}
        if actual != allowed:
            raise FreshEvidenceError(f"orphan or extra successful artifact in lane {lane}")

    input_record = plan.get("input_manifest")
    required_input = {
        "path", "sha256", "size_bytes", "sample_count", "query_order_sha256",
        "independent_regular_file_count",
    }
    if not isinstance(input_record, dict) or set(input_record) != required_input:
        raise FreshEvidenceError("fresh input manifest record fields differ")
    input_path = bind(root, {
        key: input_record[key] for key in ("path", "sha256", "size_bytes")
    }, "fresh input manifest")
    input_manifest = load(input_path)
    records = input_manifest.get("records")
    if (
        not isinstance(records, list)
        or len(records) != 128
        or input_record["sample_count"] != 128
        or input_record["independent_regular_file_count"] != 128
        or input_record["query_order_sha256"] != input_manifest.get("query_order_sha256")
    ):
        raise FreshEvidenceError("fresh input manifest counts/order differ")
    data_path = input_path.parent / "descriptive_val.json"
    if data_path.is_symlink() or not data_path.is_file() or sha256(data_path) != input_manifest.get("packaged_data_sha256"):
        raise FreshEvidenceError("fresh descriptive data differs")
    verified[data_path.relative_to(root).as_posix()] = data_path.resolve(strict=True)
    ground_truth = load(data_path)
    physical: set[tuple[int, int]] = set()
    query_keys: set[str] = set()
    for index, record in enumerate(records):
        expected_fields = {
            "query_key", "relative_path", "sha256", "size_bytes",
            "source_image_sha256",
        }
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise FreshEvidenceError(f"fresh input record fields differ: {index}")
        key = record["query_key"]
        if not isinstance(key, str) or key in query_keys:
            raise FreshEvidenceError("fresh input query key duplicate")
        query_keys.add(key)
        image = bind(input_path.parent / "images", {
            "path": record["relative_path"],
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }, f"fresh image {index}")
        metadata = image.stat()
        inode = (metadata.st_dev, metadata.st_ino)
        if inode in physical:
            raise FreshEvidenceError("fresh input image inode/path is not independent")
        physical.add(inode)
    try:
        for item in receipt["results"]:
            identity = (str(item["program"]), str(item["role"]))
            verify_result_semantics(
                document=result_documents[identity],
                receipt_item=item,
                ground_truth=ground_truth,
                input_records=records,
            )
    except (FreshSemanticError, KeyError) as exc:
        raise FreshEvidenceError(
            f"fresh semantic replay failed: {exc}"
        ) from exc
    return sorted(verified)


def render_active_section(
    receipt: Mapping[str, object], receipt_relative: str, receipt_sha256: str,
) -> str:
    validate_receipt(receipt)
    if not re.fullmatch(r"[0-9a-f]{64}", receipt_sha256):
        raise FreshEvidenceError("receipt hash differs")
    rows = []
    for item in receipt["results"]:
        rows.append(
            "| {program} | {role} | {physical_gpu_index} | {correct}/{sample_count} "
            "| {num_errors} | {total_time:.9f} | {generated_token_count} |".format(
                **item
            )
        )
    source = receipt["source"]
    return "\n".join([
        BEGIN,
        "## Active final fresh reproduction",
        "",
        "Status: `COMPLETE`",
        f"Label: `{LABEL}`",
        f"Run ID: `{receipt['run_id']}`",
        f"Source commit: `{source['commit']}`",
        f"Source tree: `{source['tree']}`",
        f"Receipt: `{receipt_relative}`",
        f"Receipt SHA-256: `{receipt_sha256}`",
        "Formal metrics affected: `false`",
        "Selection affected: `false`",
        "",
        "| Program | Role | GPU | Correct | Errors | Total seconds | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        END,
    ])


def replace_active(document: str, section: str) -> str:
    if document.count(BEGIN) != 1 or document.count(END) != 1:
        raise FreshEvidenceError("active markers must each occur once")
    before, remainder = document.split(BEGIN, 1)
    _, after = remainder.split(END, 1)
    return before + section + after


def verify_active_document(root: Path, stage: str = "final") -> dict[str, object]:
    path = root / "docs/FINAL_PORTABILITY_SMOKE.md"
    text = path.read_text(encoding="utf-8")
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise FreshEvidenceError("active markers must each occur once")
    active = BEGIN + text.split(BEGIN, 1)[1].split(END, 1)[0] + END
    forbidden = ("NaN", "Infinity", "TODO", "null")
    if any(token in active for token in forbidden):
        raise FreshEvidenceError("active evidence contains a forbidden placeholder")
    if "PENDING_FINAL_FRESH_REPRODUCTION" in active:
        if stage != "pre-reproduction" or active != PENDING:
            raise FreshEvidenceError("pending active evidence is not allowed")
        return {"status": "PENDING_FINAL_FRESH_REPRODUCTION"}
    match = re.search(r"^Receipt: `([^`]+)`$", active, re.MULTILINE)
    digest = re.search(r"^Receipt SHA-256: `([0-9a-f]{64})`$", active, re.MULTILINE)
    if match is None or digest is None:
        raise FreshEvidenceError("active receipt pointer is absent")
    pure = _pure_relative(match.group(1), "active receipt")
    root = root.resolve(strict=True)
    receipt_path = root.joinpath(*pure.parts)
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise FreshEvidenceError("active receipt is unsafe")
    receipt_path = receipt_path.resolve(strict=True)
    try:
        receipt_path.relative_to(root)
    except ValueError as exc:
        raise FreshEvidenceError("active receipt escaped root") from exc
    if sha256(receipt_path) != digest.group(1):
        raise FreshEvidenceError("active receipt hash differs")
    receipt = load(receipt_path)
    artifacts = verify_receipt_artifacts(root, receipt_path, receipt)
    expected = render_active_section(receipt, match.group(1), digest.group(1))
    if active != expected:
        raise FreshEvidenceError("active Markdown differs from receipt rendering")
    return {
        "status": "COMPLETE",
        "run_id": receipt["run_id"],
        "receipt_sha256": digest.group(1),
        "receipt_path": match.group(1),
        "result_count": len(receipt["results"]),
        "verified_artifacts": artifacts,
    }
