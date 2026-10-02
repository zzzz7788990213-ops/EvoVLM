"""Independent semantic replay for final fresh result JSON."""
from __future__ import annotations

import math
from collections import Counter
from pathlib import PurePath
from typing import Any, Mapping, Sequence

from charxiv.src.constants import DESCRIPTIVE_RESP_INST

EXACT_MATCH = "prediction.strip().casefold() == expected.strip().casefold()"


class FreshSemanticError(RuntimeError):
    pass


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FreshSemanticError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise FreshSemanticError(f"{label} is not finite/positive")
    return number


def _question(qid: int, subplot: object) -> str:
    template = DESCRIPTIVE_RESP_INST.get(qid)
    if not isinstance(template, str):
        raise FreshSemanticError(f"unknown descriptive qid: {qid}")
    if qid in {18, 19}:
        return template
    if isinstance(subplot, list) and len(subplot) == 2:
        if subplot[0] == 0:
            prefix = "For the current plot, "
        else:
            prefix = (
                f"For the subplot at row {subplot[0]} "
                f"and column {subplot[1]}, "
            )
    elif isinstance(subplot, str):
        prefix = f"For {subplot}, "
    else:
        raise FreshSemanticError("ground-truth subplot location is invalid")
    return template.format(prefix)


def derive_queries(
    ground_truth: Mapping[str, object],
    *,
    limit: int = 128,
) -> list[dict[str, object]]:
    derived: list[dict[str, object]] = []
    for source_key, raw in ground_truth.items():
        if not isinstance(raw, dict):
            raise FreshSemanticError("ground-truth figure is not an object")
        figure_id = raw.get("figure_id")
        qids = raw.get("qids")
        answers = raw.get("answers")
        subplot = raw.get("subplot_loc")
        if (
            isinstance(figure_id, bool)
            or not isinstance(figure_id, int)
            or str(figure_id) != str(source_key)
            or not isinstance(qids, list)
            or not isinstance(answers, list)
            or len(qids) != len(answers)
        ):
            raise FreshSemanticError("ground-truth figure identity differs")
        for subq_idx, (qid, answer) in enumerate(zip(qids, answers)):
            if isinstance(qid, bool) or not isinstance(qid, int):
                raise FreshSemanticError("ground-truth qid is invalid")
            derived.append({
                "query_key": f"{figure_id}_{subq_idx}",
                "figure_id": figure_id,
                "subq_idx": subq_idx,
                "qid": qid,
                "question": _question(qid, subplot),
                "expected_answer": str(answer),
            })
    if len(derived) < limit:
        raise FreshSemanticError("ground truth has fewer than 128 queries")
    return derived[:limit]


def _same_number(actual: object, expected: float, label: str) -> None:
    number = _finite(actual, label)
    if number != expected:
        raise FreshSemanticError(f"{label} differs")


def verify_result_semantics(
    *,
    document: Mapping[str, object],
    receipt_item: Mapping[str, object],
    ground_truth: Mapping[str, object],
    input_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Recompute one result from frozen GT and ZIP-relative image records."""
    all_queries = derive_queries(ground_truth)
    sample_count = receipt_item.get("sample_count")
    if isinstance(sample_count, bool) or sample_count not in {1, 128}:
        raise FreshSemanticError("fresh sample count differs")
    expected_queries = all_queries[:sample_count]
    if len(input_records) != 128:
        raise FreshSemanticError("fresh input record count differs")
    if [record.get("query_key") for record in input_records] != [
        query["query_key"] for query in all_queries
    ]:
        raise FreshSemanticError("fresh input/GT query order differs")
    program = receipt_item.get("program")
    if document.get("program") != program:
        raise FreshSemanticError("fresh result program differs")
    if (
        document.get("num_evaluated") != sample_count
        or document.get("exact_match") != EXACT_MATCH
    ):
        raise FreshSemanticError("fresh result count/scoring differs")
    identity = document.get("program_identity")
    expected_identity = {
        "model_id": receipt_item.get("model_id"),
        "model_revision": receipt_item.get("model_revision"),
        "processor_revision": receipt_item.get("processor_revision"),
        "model_weight_sha256": receipt_item.get("model_weight_sha256"),
    }
    if identity != expected_identity:
        raise FreshSemanticError("fresh result model identity differs")
    records = document.get("records")
    if not isinstance(records, list) or len(records) != sample_count:
        raise FreshSemanticError("fresh raw record count differs")

    correct_count = 0
    error_count = 0
    generated_tokens = 0
    latency_total = 0.0
    terminations: Counter[str] = Counter()
    required_record = {
        "correct", "error", "expected_answer", "figure_id",
        "generation_metadata", "image_path", "latency_seconds",
        "query_key", "question", "response", "subq_idx",
    }
    required_generation = {
        "generated_token_count", "generation_seconds", "input_token_count",
        "peak_memory_bytes", "preprocessing_seconds", "termination_reason",
        "total_seconds",
    }
    for index, (record, query, image) in enumerate(
        zip(records, expected_queries, input_records)
    ):
        if not isinstance(record, dict) or set(record) != required_record:
            raise FreshSemanticError(f"fresh raw record fields differ: {index}")
        for key in ("query_key", "figure_id", "subq_idx", "question"):
            if record.get(key) != query[key]:
                raise FreshSemanticError(f"fresh raw query differs: {index}:{key}")
        if record.get("expected_answer") != query["expected_answer"]:
            raise FreshSemanticError(f"fresh expected answer differs: {index}")
        response = record.get("response")
        error = record.get("error")
        if not isinstance(response, str) or error is not None:
            raise FreshSemanticError(f"fresh response/error differs: {index}")
        expected_correct = (
            response.strip().casefold()
            == str(query["expected_answer"]).strip().casefold()
        )
        if record.get("correct") is not expected_correct:
            raise FreshSemanticError(f"fresh correctness differs: {index}")
        correct_count += int(expected_correct)

        if (
            not isinstance(image, dict)
            or image.get("query_key") != query["query_key"]
            or not isinstance(image.get("relative_path"), str)
        ):
            raise FreshSemanticError(f"fresh input image mapping differs: {index}")
        image_name = PurePath(str(image["relative_path"])).name
        image_path = record.get("image_path")
        if not isinstance(image_path, str) or PurePath(image_path).name != image_name:
            raise FreshSemanticError(f"fresh raw image path differs: {index}")

        metadata = record.get("generation_metadata")
        if (
            not isinstance(metadata, dict)
            or not required_generation <= set(metadata)
        ):
            raise FreshSemanticError(f"fresh generation metadata differs: {index}")
        tokens = metadata.get("generated_token_count")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise FreshSemanticError(f"fresh generated tokens differ: {index}")
        reason = metadata.get("termination_reason")
        if not isinstance(reason, str) or not reason:
            raise FreshSemanticError(f"fresh termination reason differs: {index}")
        for key in (
            "generation_seconds", "preprocessing_seconds", "total_seconds",
            "peak_memory_bytes", "input_token_count",
        ):
            _finite(metadata.get(key), f"fresh {key}:{index}")
        if "prediction" in metadata and metadata["prediction"] != response:
            raise FreshSemanticError(f"fresh metadata prediction differs: {index}")
        image_fields = {
            "input_image_path", "input_image_sha256", "input_image_size_bytes"
        }
        present_image_fields = image_fields & set(metadata)
        if present_image_fields and present_image_fields != image_fields:
            raise FreshSemanticError(f"fresh metadata image fields partial: {index}")
        if present_image_fields:
            if (
                PurePath(str(metadata["input_image_path"])).name != image_name
                or metadata["input_image_sha256"] != image.get("sha256")
                or metadata["input_image_size_bytes"] != image.get("size_bytes")
            ):
                raise FreshSemanticError(f"fresh metadata image identity differs: {index}")
        latency = _finite(
            record.get("latency_seconds"), f"fresh latency:{index}", positive=True
        )
        latency_total += latency
        generated_tokens += tokens
        terminations[reason] += 1

    total_time = _finite(document.get("total_time"), "fresh total time", positive=True)
    if total_time + 1e-9 < latency_total:
        raise FreshSemanticError("fresh total time is below raw latency sum")
    expected_average = total_time / sample_count
    _same_number(document.get("avg_time_per_query"), expected_average, "fresh average")
    expected_accuracy = correct_count / sample_count
    _same_number(document.get("accuracy"), expected_accuracy, "fresh accuracy")
    if (
        document.get("correct") != correct_count
        or document.get("num_errors") != error_count
        or receipt_item.get("correct") != correct_count
        or receipt_item.get("num_errors") != error_count
        or receipt_item.get("generated_token_count") != generated_tokens
    ):
        raise FreshSemanticError("fresh raw-derived counts differ")
    _same_number(receipt_item.get("total_time"), total_time, "receipt total time")
    _same_number(
        receipt_item.get("avg_time_per_query"),
        expected_average,
        "receipt average",
    )
    return {
        "program": program,
        "sample_count": sample_count,
        "correct": correct_count,
        "errors": error_count,
        "generated_token_count": generated_tokens,
        "termination_counts": dict(sorted(terminations.items())),
        "record_latency_sum": latency_total,
        "total_time": total_time,
    }
