"""Strict active fresh-evidence rendering tests."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from pathlib import Path

import pytest

from evaluation import final_fresh_evidence as evidence
from evaluation.fresh_semantics import EXACT_MATCH, derive_queries


def _receipt() -> dict:
    results = []
    for program, role, gpu, count, correct in (
        ("baseline_instruct", "dev128", 0, 128, 51),
        ("manual_instruct", "dev128", 0, 128, 78),
        ("manual_thinking", "dev128", 1, 128, 23),
        ("evolved_instruct", "dev128", 0, 128, 78),
        ("evolved_thinking", "dev128", 1, 128, 44),
        ("best_accuracy", "single1", 0, 1, 1),
        ("best_speed", "single1", 0, 1, 1),
        ("best_overall", "single1", 0, 1, 1),
    ):
        results.append({
            "program": program, "role": role, "physical_gpu_index": gpu,
            "sample_count": count, "correct": correct, "num_errors": 0,
            "total_time": 2.0, "avg_time_per_query": 2.0 / count,
            "generated_token_count": count, "result": {
                "path": f"results/{program}.json", "sha256": "a" * 64,
                "size_bytes": 1,
            }, "console": {
                "path": f"results/{program}.log", "sha256": "b" * 64,
                "size_bytes": 1,
            }, "start": {
                "path": f"results/{program}.start.json", "sha256": "6" * 64,
                "size_bytes": 1,
            }, "end": {
                "path": f"results/{program}.end.json", "sha256": "7" * 64,
                "size_bytes": 1,
            }, "attempt_id": str(uuid.uuid4()),
            "program_source_sha256": "c" * 64,
            "dependency_closure_sha256": "d" * 64,
            "model_id": "Qwen/fixed", "model_revision": "e" * 40,
            "processor_revision": "e" * 40, "model_weight_sha256": "f" * 64,
            "input_manifest_sha256": "1" * 64,
        })
    return {
        "schema_version": "final-fresh-reproduction-receipt-v1",
        "label": evidence.LABEL, "status": "COMPLETE",
        "run_id": str(uuid.uuid4()), "source": {
            "commit": "2" * 40, "tree": "3" * 40,
        }, "plan": {"path": "plan.json", "sha256": "4" * 64, "size_bytes": 1},
        "formal_manifest": {
            "path": "final.json", "sha256": "5" * 64, "size_bytes": 1,
        }, "started_at_utc": "2026-01-01T00:00:00Z",
        "completed_at_utc": "2026-01-01T01:00:00Z",
        "lanes": {"0": [], "1": []}, "results": results,
        "lane_receipts": [
            {"path": f"lane{i}/receipt.json", "sha256": "8" * 64, "size_bytes": 1}
            for i in (0, 1)
        ],
        "orchestrator_logs": [
            {"path": f"lane{i}.orchestrator.log", "sha256": "9" * 64, "size_bytes": 1}
            for i in (0, 1)
        ],
        "formal_metrics_affected": False, "selection_affected": False,
        "expected_dev128_correct": {
            "baseline_instruct": 51, "manual_instruct": 78,
            "manual_thinking": 23, "evolved_instruct": 78,
            "evolved_thinking": 44,
        }, "best_source_identity": {"byte_identical_to": "evolved_instruct"},
    }


def _json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _record(path: Path, base: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _root(tmp_path: Path, receipt: dict) -> Path:
    root = tmp_path / "root"
    (root / "docs").mkdir(parents=True)
    output = root / "results/fresh"
    output.mkdir(parents=True)

    formal = root / "evaluation/recovery_v2/final_manifest.json"
    _json(formal, {"status": "COMPLETE"})
    receipt["formal_manifest"] = _record(formal, root)

    input_root = root / "data/final_fresh"
    images = input_root / "images"
    images.mkdir(parents=True)
    data = input_root / "descriptive_val.json"
    ground_truth = {
        str(figure): {
            "figure_id": figure,
            "num_subplots": 1,
            "subplot_loc": [0, 0],
            "qids": [1, 1, 1, 1],
            "answers": [f"answer-{figure}-{subq}" for subq in range(4)],
        }
        for figure in range(32)
    }
    _json(data, ground_truth)
    queries = derive_queries(evidence.load(data))
    input_records = []
    for index, query in enumerate(queries):
        image = images / f"{index:03d}.jpg"
        image.write_bytes(f"image-{index}".encode())
        record = _record(image, images)
        input_records.append({
            "query_key": query["query_key"], "relative_path": record["path"],
            "sha256": record["sha256"], "size_bytes": record["size_bytes"],
            "source_image_sha256": record["sha256"],
        })
    input_manifest = input_root / "query_images_manifest.json"
    order_sha = hashlib.sha256("\n".join(
        item["query_key"] for item in input_records
    ).encode()).hexdigest()
    _json(input_manifest, {
        "records": input_records, "sample_count": 128,
        "independent_regular_file_count": 128,
        "query_order_sha256": order_sha,
        "packaged_data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    })
    input_record = _record(input_manifest, root)
    input_record.update({
        "sample_count": 128, "query_order_sha256": order_sha,
        "independent_regular_file_count": 128,
    })

    lane_results = {0: [], 1: []}
    lane_schedule = {"0": [], "1": []}
    for item in receipt["results"]:
        lane = item["physical_gpu_index"]
        lane_root = output / f"lane{lane}"
        lane_root.mkdir(exist_ok=True)
        stem = f"{item['program']}.{item['role']}.{item['attempt_id']}"
        result_path = lane_root / f"{stem}.result.json"
        console_path = lane_root / f"{stem}.console.log"
        start_path = lane_root / f"{stem}.start.json"
        end_path = lane_root / f"{stem}.end.json"
        selected_queries = queries[:item["sample_count"]]
        raw_records = []
        for index, (query, image_record) in enumerate(
            zip(selected_queries, input_records)
        ):
            response = (
                query["expected_answer"]
                if index < item["correct"]
                else "deliberately-incorrect"
            )
            latency = 2.0 / item["sample_count"]
            raw_records.append({
                "correct": index < item["correct"],
                "error": None,
                "expected_answer": query["expected_answer"],
                "figure_id": query["figure_id"],
                "generation_metadata": {
                    "generated_token_count": 1,
                    "generation_seconds": latency,
                    "input_token_count": 1,
                    "peak_memory_bytes": 0,
                    "preprocessing_seconds": 0.0,
                    "termination_reason": "eos",
                    "total_seconds": latency,
                },
                "image_path": str(images / image_record["relative_path"]),
                "latency_seconds": latency,
                "query_key": query["query_key"],
                "question": query["question"],
                "response": response,
                "subq_idx": query["subq_idx"],
            })
        _json(result_path, {
            "accuracy": item["correct"] / item["sample_count"],
            "avg_time_per_query": 2.0 / item["sample_count"],
            "correct": item["correct"],
            "exact_match": EXACT_MATCH,
            "num_errors": 0,
            "num_evaluated": item["sample_count"],
            "program": item["program"],
            "program_identity": {
                "model_id": item["model_id"],
                "model_revision": item["model_revision"],
                "processor_revision": item["processor_revision"],
                "model_weight_sha256": item["model_weight_sha256"],
            },
            "records": raw_records,
            "total_time": 2.0,
        })
        console_path.write_text("completed\n")
        common = {
            "attempt_id": item["attempt_id"], "run_id": receipt["run_id"],
            "program": item["program"], "role": item["role"],
            "physical_gpu_index": lane,
        }
        _json(start_path, {"event": "attempt_started", **common})
        item["result"] = _record(result_path, output)
        item["console"] = _record(console_path, output)
        _json(end_path, {
            "event": "attempt_ended", **common, "returncode": 0,
            "outcome": "COMPLETED", "result": item["result"],
            "console": item["console"],
        })
        item["start"] = _record(start_path, output)
        item["end"] = _record(end_path, output)
        lane_results[lane].append(item)
        lane_schedule[str(lane)].append({
            "program": item["program"], "role": item["role"],
            "sample_count": item["sample_count"],
        })
    receipt["lanes"] = lane_schedule
    receipt["lane_receipts"] = []
    receipt["orchestrator_logs"] = []
    for lane in (0, 1):
        lane_receipt = output / f"lane{lane}/receipt.json"
        _json(lane_receipt, {
            "run_id": receipt["run_id"], "lane": lane,
            "results": lane_results[lane],
        })
        receipt["lane_receipts"].append(_record(lane_receipt, output))
        log = output / f"lane{lane}.orchestrator.log"
        log.write_text("lane complete\n")
        receipt["orchestrator_logs"].append(_record(log, output))

    plan = output / "plan.json"
    _json(plan, {
        "source": receipt["source"],
        "formal_manifest": receipt["formal_manifest"],
        "input_manifest": input_record,
        "lanes": lane_schedule,
    })
    receipt["plan"] = _record(plan, output)
    target = output / "receipt.json"
    _json(target, receipt)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    relative = target.relative_to(root).as_posix()
    section = evidence.render_active_section(receipt, relative, digest)
    (root / "docs/FINAL_PORTABILITY_SMOKE.md").write_text(section)
    return root


def test_pending_is_only_accepted_pre_reproduction(tmp_path: Path) -> None:
    root = tmp_path; (root / "docs").mkdir()
    (root / "docs/FINAL_PORTABILITY_SMOKE.md").write_text(evidence.PENDING)
    assert evidence.verify_active_document(root, "pre-reproduction")["status"].startswith("PENDING")
    with pytest.raises(evidence.FreshEvidenceError):
        evidence.verify_active_document(root, "final")


def test_complete_active_markdown_is_exact_receipt_render(tmp_path: Path) -> None:
    root = _root(tmp_path, _receipt())
    assert evidence.verify_active_document(root)["result_count"] == 8


@pytest.mark.parametrize("mutation", ["nan", "missing", "mismatch", "duplicate"])
def test_active_evidence_tamper_fails(tmp_path: Path, mutation: str) -> None:
    receipt = _receipt(); root = _root(tmp_path, receipt)
    doc = root / "docs/FINAL_PORTABILITY_SMOKE.md"
    if mutation == "nan":
        doc.write_text(doc.read_text().replace("2.000000000", "NaN", 1))
    elif mutation == "missing":
        doc.write_text(doc.read_text().replace(f"Run ID: `{receipt['run_id']}`\n", ""))
    elif mutation == "mismatch":
        doc.write_text(doc.read_text().replace("51/128", "52/128"))
    else:
        doc.write_text(doc.read_text() + "\n" + evidence.PENDING)
    with pytest.raises(evidence.FreshEvidenceError):
        evidence.verify_active_document(root)


def test_receipt_nan_and_unknown_field_fail(tmp_path: Path) -> None:
    receipt = _receipt(); receipt["results"][0]["total_time"] = float("nan")
    with pytest.raises(evidence.FreshEvidenceError):
        evidence.validate_receipt(receipt)
    receipt = _receipt(); receipt["extra"] = True
    with pytest.raises(evidence.FreshEvidenceError):
        evidence.validate_receipt(receipt)



def _active_receipt(root: Path) -> tuple[Path, dict]:
    path = root / "results/fresh/receipt.json"
    return path, evidence.load(path)


@pytest.mark.parametrize("artifact", ["start", "end"])
def test_fresh_closure_rejects_deleted_attempt_artifact(
    tmp_path: Path, artifact: str,
) -> None:
    root = _root(tmp_path, _receipt())
    receipt_path, receipt = _active_receipt(root)
    output = receipt_path.parent
    target = output / receipt["results"][0][artifact]["path"]
    target.unlink()
    with pytest.raises(evidence.FreshEvidenceError, match="artifact is unsafe"):
        evidence.verify_receipt_artifacts(root, receipt_path, receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [("returncode", 1), ("outcome", "FAILED")],
)
def test_fresh_closure_rejects_rc_or_outcome_tamper(
    tmp_path: Path, field: str, value: object,
) -> None:
    root = _root(tmp_path, _receipt())
    receipt_path, receipt = _active_receipt(root)
    output = receipt_path.parent
    item = receipt["results"][0]
    end = output / item["end"]["path"]
    document = evidence.load(end)
    document[field] = value
    _json(end, document)
    item["end"] = _record(end, output)
    with pytest.raises(evidence.FreshEvidenceError, match="attempt outcome differs"):
        evidence.verify_receipt_artifacts(root, receipt_path, receipt)


def test_fresh_closure_rejects_result_console_swap(tmp_path: Path) -> None:
    root = _root(tmp_path, _receipt())
    receipt_path, receipt = _active_receipt(root)
    output = receipt_path.parent
    item = receipt["results"][0]
    end = output / item["end"]["path"]
    document = evidence.load(end)
    document["result"], document["console"] = document["console"], document["result"]
    _json(end, document)
    item["end"] = _record(end, output)
    with pytest.raises(evidence.FreshEvidenceError, match="result/console swap"):
        evidence.verify_receipt_artifacts(root, receipt_path, receipt)


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("orphan.bin", b"orphan"),
        ("extra-success.end.json", b'{"outcome":"COMPLETED","returncode":0}\n'),
    ],
)
def test_fresh_closure_rejects_orphan_or_extra_success(
    tmp_path: Path, name: str, payload: bytes,
) -> None:
    root = _root(tmp_path, _receipt())
    receipt_path, receipt = _active_receipt(root)
    (receipt_path.parent / "lane0" / name).write_bytes(payload)
    with pytest.raises(evidence.FreshEvidenceError, match="orphan or extra"):
        evidence.verify_receipt_artifacts(root, receipt_path, receipt)


def test_fresh_closure_rejects_start_end_swap(tmp_path: Path) -> None:
    root = _root(tmp_path, _receipt())
    receipt_path, receipt = _active_receipt(root)
    item = receipt["results"][0]
    item["start"], item["end"] = item["end"], item["start"]
    with pytest.raises(evidence.FreshEvidenceError, match="attempt event differs"):
        evidence.verify_receipt_artifacts(root, receipt_path, receipt)


@pytest.mark.parametrize("mutation", ["answer_and_summaries", "query_and_input"])
def test_fresh_semantic_replay_rejects_coherent_tamper(
    tmp_path: Path, mutation: str,
) -> None:
    root = _root(tmp_path, _receipt())
    receipt_path, receipt = _active_receipt(root)
    output = receipt_path.parent
    plan = evidence.load(output / receipt["plan"]["path"])
    input_manifest = evidence.load(root / plan["input_manifest"]["path"])
    ground_truth = evidence.load(
        (root / plan["input_manifest"]["path"]).parent / "descriptive_val.json"
    )
    item = copy.deepcopy(receipt["results"][0])
    document = evidence.load(output / item["result"]["path"])
    if mutation == "answer_and_summaries":
        record = document["records"][100]
        record["expected_answer"] = "forged"
        record["response"] = "FORGED"
        record["correct"] = True
        document["correct"] += 1
        document["accuracy"] = document["correct"] / document["num_evaluated"]
        item["correct"] += 1
        receipt["expected_dev128_correct"][item["program"]] += 1
    else:
        document["records"][0]["query_key"] = "forged_0"
        input_manifest["records"][0]["query_key"] = "forged_0"
    with pytest.raises(
        evidence.FreshSemanticError,
        match="expected answer differs|input/GT query order differs",
    ):
        evidence.verify_result_semantics(
            document=document,
            receipt_item=item,
            ground_truth=ground_truth,
            input_records=input_manifest["records"],
        )


def test_fresh_input_files_must_have_independent_inodes(tmp_path: Path) -> None:
    root = _root(tmp_path, _receipt())
    receipt_path, receipt = _active_receipt(root)
    output = receipt_path.parent
    input_path = root / "data/final_fresh/query_images_manifest.json"
    input_manifest = evidence.load(input_path)
    images = input_path.parent / "images"
    first = images / input_manifest["records"][0]["relative_path"]
    second = images / input_manifest["records"][1]["relative_path"]
    second.unlink()
    os.link(first, second)
    metadata = second.stat()
    input_manifest["records"][1].update({
        "sha256": evidence.sha256(second), "size_bytes": metadata.st_size,
        "source_image_sha256": evidence.sha256(second),
    })
    _json(input_path, input_manifest)
    plan_path = output / receipt["plan"]["path"]
    plan = evidence.load(plan_path)
    plan["input_manifest"].update(_record(input_path, root))
    _json(plan_path, plan)
    receipt["plan"] = _record(plan_path, output)
    with pytest.raises(evidence.FreshEvidenceError, match="inode/path is not independent"):
        evidence.verify_receipt_artifacts(root, receipt_path, receipt)
