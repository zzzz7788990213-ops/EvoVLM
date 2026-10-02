"""Portable stdlib replay of sealed Recovery-v2 per-query evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from evaluation.phase3_recovery import (
    aggregate_results,
    summarize_records,
    validate_journal_document,
)

PROJECT_ROOT = Path(__file__).resolve(strict=True).parent.parent
FINAL_MANIFEST = PROJECT_ROOT / "evaluation/recovery_v2/final_manifest.json"
AMENDMENT_MANIFEST = PROJECT_ROOT / "evaluation/portability/amendment_manifest.json"
GT_PATH = PROJECT_ROOT / "charxiv/data/descriptive_val.json"
MANIFEST_ROOT = PROJECT_ROOT / "data/phase3/manifests"
OLD_CAMPAIGN_ID = "1fa07427-6b42-4bbb-a9f9-af63742bdfc6"
NEW_CAMPAIGN_ID = "d3a319ae-cdb0-48d4-860c-70f44585e15c"
CAMPAIGN_ROOT = PROJECT_ROOT / "results/phase3_recovery_v2/campaigns"
FORMAL_PROGRAMS = (
    "baseline_instruct",
    "manual_instruct",
    "manual_thinking",
    "evolved_instruct",
    "evolved_thinking",
)
HEX64 = re.compile(r"[0-9a-f]{64}")
PORTABLE_EXECUTABLE_CLOSURE = (
    "evaluate.py",
    "baseline_common.py",
    "baseline_instruct.py",
    "baseline_thinking.py",
    "manual_instruct.py",
    "manual_thinking.py",
    "evolved_instruct.py",
    "evolved_thinking.py",
    "best_accuracy.py",
    "best_speed.py",
    "best_overall.py",
    "evolve_instruct.py",
    "evolve_thinking.py",
    "manual_common.py",
    "model_locator.py",
    "reproduce.sh",
    "submission_checker.py",
    "requirements.lock.txt",
    "charxiv/src/descriptive_utils.py",
    "charxiv/src/constants.py",
    "evaluation/atomic_artifacts.py",
    "evaluation/phase3_manifests.py",
    "evaluation/phase3_recovery.py",
    "evaluation/phase3_runner.py",
    "evaluation/portable_reproduce.py",
    "evaluation/portability_identity.py",
    "evaluation/final_fresh_evidence.py",
    "evaluation/fresh_semantics.py",
    "evaluation/model_snapshot.py",
    "evaluation/model_snapshots/qwen3_vl_2b_instruct.critical.v1.json",
    "evaluation/model_snapshots/qwen3_vl_2b_thinking.critical.v1.json",
    "evaluation/v1_hash_erratum.py",
    "scripts/final_portability_smoke.py",
    "scripts/build_portability_amendment.py",
    "scripts/package_submission.py",
    "scripts/portable_full_inference.py",
    "scripts/build_final_fresh_inputs.py",
    "scripts/final_fresh_reproduction.py",
    "evolution/__init__.py",
    "evolution/candidate_runtime.py",
    "evolution/catalog_provider.py",
    "evolution/core.py",
    "evolution/dsl.py",
    "evolution/local_campaign.py",
    "evolution/local_provider.py",
    "evolution/mutation_catalog.py",
    "evolution/phase2_manifests.py",
    "evolution/process_safety.py",
    "evolution/providers.py",
    "evolution/renderer.py",
    "evolution/restricted.py",
    "evolution/run_manifest.py",
    "evolution/status.py",
    "evolution/target_worker.py",
    "evolution/valid_child_receipt.py",
    "evolution/worker.py",
)
FIELD_PRESENCE_LEGEND = {
    "E": "identity field explicitly present in the artifact",
    "H": "artifact referenced by an explicit path plus SHA-256",
    "L": "identity is only inferable from path/name/time and is not trusted",
    "A": "identity field absent",
}
FORMAL_FIELD_PRESENCE_MATRIX = {
    "campaign_id": {
        "plan": "E", "result": "E", "start": "E", "end": "E",
        "journals": "E", "runtime_event": "A",
    },
    "run_id": {
        "plan": "E", "result": "E", "start": "A", "end": "A",
        "journals": "E", "runtime_event": "E",
    },
    "attempt_id": {
        "plan": "A", "result": "E", "start": "E", "end": "E",
        "journals": "E", "runtime_event": "A",
    },
    "program": {
        "plan": "E", "result": "E", "start": "E", "end": "E",
        "journals": "E", "runtime_event": "A",
    },
    "repeat": {
        "plan": "E", "result": "E", "start": "E", "end": "E",
        "journals": "A", "runtime_event": "A",
    },
    "schedule_index": {
        "plan": "E", "result": "L", "start": "E", "end": "E",
        "journals": "A", "runtime_event": "A",
    },
    "runtime_event_id": {
        "plan": "A", "result": "E", "start": "A", "end": "A",
        "journals": "A", "runtime_event": "E",
    },
    "runtime_artifact": {
        "plan": "A", "result": "H", "start": "A", "end": "A",
        "journals": "A", "runtime_event": "E",
    },
    "journal_artifacts": {
        "plan": "L", "result": "H", "start": "A", "end": "L",
        "journals": "E", "runtime_event": "A",
    },
}


class PortableReplayError(RuntimeError):
    """Sealed evidence or the portability amendment is inconsistent."""


def _duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PortableReplayError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise PortableReplayError(f"non-finite JSON constant: {value}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PortableReplayError(f"cannot read strict JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PortableReplayError(f"JSON root is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_file(relative: str) -> Path:
    if not isinstance(relative, str):
        raise PortableReplayError("artifact path is not text")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise PortableReplayError(f"unsafe artifact path: {relative}")
    candidate = PROJECT_ROOT.joinpath(*pure.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise PortableReplayError(f"artifact is not a regular file: {relative}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise PortableReplayError(f"artifact escaped package: {relative}") from exc
    return resolved


def _hash_index(final: Mapping[str, object]) -> dict[str, dict[str, object]]:
    raw = final.get("artifacts")
    if not isinstance(raw, list) or not raw:
        raise PortableReplayError("formal artifact closure is absent")
    index: dict[str, dict[str, object]] = {}
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size_bytes"}:
            raise PortableReplayError("formal artifact record fields differ")
        path = item["path"]
        digest = item["sha256"]
        size = item["size_bytes"]
        if not isinstance(path, str) or path in index:
            raise PortableReplayError("duplicate or invalid formal artifact path")
        if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
            raise PortableReplayError(f"invalid artifact hash: {path}")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise PortableReplayError(f"invalid artifact size: {path}")
        index[path] = dict(item)
    return index


def verify_amended_closure(final: Mapping[str, object]) -> dict[str, object]:
    amendment = load_json(AMENDMENT_MANIFEST)
    if amendment.get("manifest_version") != "final-portability-amendment-v1":
        raise PortableReplayError("portability amendment version differs")
    index = _hash_index(final)
    if amendment.get("formal_artifact_count") != len(index):
        raise PortableReplayError("formal artifact count differs")
    formal_record = amendment.get("formal_manifest")
    if not isinstance(formal_record, dict) or set(formal_record) != {"path", "sha256", "size_bytes"}:
        raise PortableReplayError("formal manifest record fields differ")
    if formal_record["path"] != FINAL_MANIFEST.relative_to(PROJECT_ROOT).as_posix():
        raise PortableReplayError("formal manifest path differs")
    if (
        formal_record["sha256"] != sha256_file(FINAL_MANIFEST)
        or formal_record["size_bytes"] != FINAL_MANIFEST.stat().st_size
    ):
        raise PortableReplayError("formal manifest hash/size differs")
    raw_changed = amendment.get("changed_artifacts")
    raw_added = amendment.get("added_artifacts")
    raw_closure = amendment.get("portable_executable_closure")
    if not isinstance(raw_changed, list) or not isinstance(raw_added, list) or not isinstance(raw_closure, list):
        raise PortableReplayError("amendment artifact lists are invalid")
    changed: dict[str, dict[str, object]] = {}
    for record in raw_changed:
        expected = {"path", "formal_sha256", "submission_sha256", "size_bytes"}
        if not isinstance(record, dict) or set(record) != expected:
            raise PortableReplayError("changed artifact record fields differ")
        path = str(record["path"])
        if path in changed or path not in index:
            raise PortableReplayError(f"changed artifact is duplicate or unbound: {path}")
        if record["formal_sha256"] != index[path]["sha256"]:
            raise PortableReplayError(f"changed artifact formal hash differs: {path}")
        changed[path] = dict(record)
    verified = 0
    for path, formal in index.items():
        artifact = safe_file(path)
        actual_hash = sha256_file(artifact)
        actual_size = artifact.stat().st_size
        if path in changed:
            expected = changed[path]
            if actual_hash != expected["submission_sha256"] or actual_size != expected["size_bytes"]:
                raise PortableReplayError(f"amended artifact mismatch: {path}")
        elif actual_hash != formal["sha256"] or actual_size != formal["size_bytes"]:
            raise PortableReplayError(f"unregistered formal artifact change: {path}")
        verified += 1
    added_seen: set[str] = set()
    for record in raw_added:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
            raise PortableReplayError("added artifact record fields differ")
        path = str(record["path"])
        if path in added_seen or path in index or path == AMENDMENT_MANIFEST.relative_to(PROJECT_ROOT).as_posix():
            raise PortableReplayError(f"added artifact is duplicate or self-referential: {path}")
        artifact = safe_file(path)
        if sha256_file(artifact) != record["sha256"] or artifact.stat().st_size != record["size_bytes"]:
            raise PortableReplayError(f"added artifact mismatch: {path}")
        added_seen.add(path)
    closure_seen: set[str] = set()
    for record in raw_closure:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
            raise PortableReplayError("portable closure record fields differ")
        path = str(record["path"])
        if path in closure_seen or path not in PORTABLE_EXECUTABLE_CLOSURE:
            raise PortableReplayError(f"portable closure path is duplicate or unknown: {path}")
        artifact = safe_file(path)
        if sha256_file(artifact) != record["sha256"] or artifact.stat().st_size != record["size_bytes"]:
            raise PortableReplayError(f"portable closure artifact mismatch: {path}")
        if path not in index and path not in added_seen:
            raise PortableReplayError(f"portable closure artifact lacks amendment binding: {path}")
        closure_seen.add(path)
    if closure_seen != set(PORTABLE_EXECUTABLE_CLOSURE):
        missing = sorted(set(PORTABLE_EXECUTABLE_CLOSURE) - closure_seen)
        raise PortableReplayError(f"portable executable closure is incomplete: {missing}")
    return {
        "formal_artifacts": verified,
        "changed": len(changed),
        "added": len(added_seen),
        "portable_executable_closure": len(closure_seen),
    }


def _record_hash(record: object, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
        raise PortableReplayError(f"{label} record fields differ")
    path = safe_file(str(record["path"]))
    if sha256_file(path) != record["sha256"] or path.stat().st_size != record["size_bytes"]:
        raise PortableReplayError(f"{label} hash/size differs")
    return path




def _selected_attempt_paths(
    index: Mapping[str, Mapping[str, object]],
    campaign_id: str,
    entry: Mapping[str, object],
) -> dict[str, str]:
    """Enumerate one selected attempt from the formal manifest, never a glob."""
    campaign = f"results/phase3_recovery_v2/campaigns/{campaign_id}/"
    entry_index = int(entry["index"])
    if campaign_id == OLD_CAMPAIGN_ID:
        prefix = campaign + str(entry["attempt_prefix"]) + "."
    elif campaign_id == NEW_CAMPAIGN_ID:
        prefix = campaign + f"replacement.{entry_index}."
    else:
        raise PortableReplayError("selected attempt campaign is unknown")
    matches = {
        suffix: sorted(
            path for path in index
            if path.startswith(prefix) and path.endswith(suffix)
        )
        for suffix in ("start.json", "end.json", "console.log")
    }
    if any(len(paths) != 1 for paths in matches.values()):
        raise PortableReplayError(
            f"selected attempt artifact set differs: {campaign_id}:{entry_index}"
        )
    start, end, console = (
        matches["start.json"][0],
        matches["end.json"][0],
        matches["console.log"][0],
    )
    start_token = start[:-len("start.json")]
    if end != start_token + "end.json" or console != start_token + "console.log":
        raise PortableReplayError("selected attempt artifacts do not share an identity")
    return {"start": start, "end": end, "console": console}


def verify_attempt_closure(
    final: Mapping[str, object],
    entry: Mapping[str, object],
    result: Mapping[str, object],
    campaign_id: str,
) -> dict[str, object]:
    """Cross-bind plan/result/start/end/console/runtime evidence."""
    index = _hash_index(final)
    paths = _selected_attempt_paths(index, campaign_id, entry)
    artifacts = {
        name: _record_hash(index[path], f"attempt {name}")
        for name, path in paths.items()
    }
    start = load_json(artifacts["start"])
    end = load_json(artifacts["end"])
    attempt_id = result.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise PortableReplayError("formal result attempt ID is absent")
    common = {
        "campaign_id": campaign_id,
        "program": entry["program"],
        "repeat": entry["repeat"],
    }
    for key, expected in common.items():
        if start.get(key) != expected or end.get(key) != expected:
            raise PortableReplayError(f"attempt {key} differs")
    start_index = start.get("entry_index", start.get("index"))
    end_index = end.get("entry_index", end.get("index"))
    if start_index != entry["index"] or end_index != entry["index"]:
        raise PortableReplayError("attempt schedule index differs")
    if start.get("attempt_id") != attempt_id or end.get("attempt_id") != attempt_id:
        raise PortableReplayError("attempt ID cross-binding differs")
    if end.get("returncode") != 0:
        raise PortableReplayError("selected attempt did not exit successfully")
    if campaign_id == OLD_CAMPAIGN_ID and end.get("outcome") != "COMPLETED":
        raise PortableReplayError("retained attempt outcome is not COMPLETED")
    console_hash = sha256_file(artifacts["console"])
    if end.get("console_sha256") != console_hash:
        raise PortableReplayError("attempt console hash differs")
    if campaign_id == OLD_CAMPAIGN_ID:
        if end.get("console_path") != paths["console"]:
            raise PortableReplayError("retained attempt console path differs")
        started = float(start.get("started_unix", -1))
        completed = float(end.get("completed_unix", -1))
        closure_level = "legacy_partial_runtime_event_not_in_final_manifest"
    else:
        from datetime import datetime
        try:
            started = datetime.fromisoformat(
                str(start["started_at_utc"]).replace("Z", "+00:00")
            ).timestamp()
            completed = datetime.fromisoformat(
                str(end["completed_at_utc"]).replace("Z", "+00:00")
            ).timestamp()
        except (KeyError, TypeError, ValueError) as exc:
            raise PortableReplayError("replacement attempt timestamps differ") from exc
        closure_level = (
            "legacy_partial_end_omits_result_and_runtime_not_in_final_manifest"
        )
    if not (math.isfinite(started) and math.isfinite(completed) and completed >= started):
        raise PortableReplayError("attempt timestamps are not finite and monotonic")
    runtime_path = result.get("runtime_event_path")
    runtime_hash = result.get("runtime_event_sha256")
    if not isinstance(runtime_path, str):
        raise PortableReplayError("runtime event path is absent")
    runtime = safe_file(runtime_path)
    if sha256_file(runtime) != runtime_hash:
        raise PortableReplayError("runtime event hash differs")
    runtime_doc = load_json(runtime)
    runtime_event_id = result.get("runtime_event_id")
    if (
        not isinstance(runtime_event_id, str)
        or runtime_doc.get("runtime_event_id") != runtime_event_id
        or runtime_doc.get("run_id") != result.get("run_id")
    ):
        raise PortableReplayError("runtime event identity cross-binding differs")
    frozen = result.get("program_record")
    if not isinstance(frozen, dict):
        raise PortableReplayError("runtime program record is absent")
    model_bindings = {
        "model_id": "model_id",
        "model_revision": "model_revision",
        "processor_revision": "processor_revision",
        "weight_sha256": "model_weight_sha256",
        "attention": "attention_implementation",
        "dtype": "dtype",
        "quantization": "quantization",
    }
    runtime_missing: list[str] = []
    for program_key, runtime_key in model_bindings.items():
        if runtime_key not in runtime_doc:
            runtime_missing.append(runtime_key)
            continue
        if frozen.get(program_key) != runtime_doc.get(runtime_key):
            raise PortableReplayError(
                f"runtime model binding differs: {program_key}"
            )
    scope = result.get("gpu_scope")
    if not isinstance(scope, dict):
        raise PortableReplayError("result GPU scope is absent")
    scope_bindings = {
        "physical_gpu_index": "physical_gpu_index",
        "cuda_visible_devices": "cuda_visible_devices",
        "visible_device_count": "torch_cuda_device_count",
        "logical_device": "logical_device",
        "device_name": "device_name",
    }
    runtime_missing_gpu: list[str] = []
    for result_key, runtime_key in scope_bindings.items():
        if runtime_key not in runtime_doc:
            runtime_missing_gpu.append(runtime_key)
            continue
        if scope.get(result_key) != runtime_doc.get(runtime_key):
            raise PortableReplayError(
                f"runtime GPU binding differs: {result_key}"
            )
    runtime_record = (
        dict(index[runtime_path])
        if runtime_path in index
        else {
            "path": runtime_path,
            "sha256": sha256_file(runtime),
            "size_bytes": runtime.stat().st_size,
            "legacy_caveat": "not_enumerated_in_formal_manifest",
        }
    )
    return {
        "campaign_id": campaign_id,
        "entry_index": entry["index"],
        "attempt_id": attempt_id,
        "closure_level": closure_level,
        "start": dict(index[paths["start"]]),
        "end": dict(index[paths["end"]]),
        "console": dict(index[paths["console"]]),
        "runtime_event": runtime_record,
        "runtime_missing_model_fields": runtime_missing,
        "runtime_missing_gpu_fields": runtime_missing_gpu,
    }


def _query_from_record(
    record: Mapping[str, object], ground_truth: Mapping[str, object], index: int,
) -> dict[str, object]:
    key = record.get("query_key")
    figure = str(record.get("figure_id"))
    subq = record.get("subq_idx")
    if not isinstance(key, str) or isinstance(subq, bool) or not isinstance(subq, int):
        raise PortableReplayError("query identity fields are invalid")
    if key != f"{figure}_{subq}":
        raise PortableReplayError(f"query key components differ: {key}")
    gt = ground_truth.get(figure)
    if not isinstance(gt, dict):
        raise PortableReplayError(f"ground truth figure missing: {figure}")
    answers = gt.get("answers")
    qids = gt.get("qids")
    if not isinstance(answers, list) or not isinstance(qids, list) or not 0 <= subq < len(answers) or subq >= len(qids):
        raise PortableReplayError(f"ground truth subquery missing: {key}")
    if str(record.get("qid")) != str(qids[subq]):
        raise PortableReplayError(f"qid differs from ground truth: {key}")
    question = record.get("question")
    image_hash = record.get("image_sha256")
    if not isinstance(question, str) or not question or not isinstance(image_hash, str) or HEX64.fullmatch(image_hash) is None:
        raise PortableReplayError(f"query content is invalid: {key}")
    return {
        "index": index,
        "query_key": key,
        "figure_id": figure,
        "subq_idx": subq,
        "qid": str(qids[subq]),
        "question": question,
        "image_path": record.get("image_path"),
        "image_sha256": image_hash,
        "expected_answer": str(answers[subq]),
    }


def verify_result(
    path: Path,
    expected_hash: str,
    plan: Mapping[str, object],
    entry: Mapping[str, object],
    ground_truth: Mapping[str, object],
    formal_index: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise PortableReplayError(f"formal result is unsafe: {path.name}")
    try:
        relative = path.resolve(strict=True).relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise PortableReplayError("formal result escaped package") from exc
    outer = formal_index.get(relative)
    if (
        not isinstance(outer, dict)
        or outer.get("sha256") != expected_hash
        or outer.get("size_bytes") != path.stat().st_size
    ):
        raise PortableReplayError(f"formal result lacks outer-root binding: {path.name}")
    if sha256_file(path) != expected_hash:
        raise PortableReplayError(f"formal result hash differs: {path.name}")
    document = load_json(path)
    program = str(entry["program"])
    manifest_name = str(entry["manifest"])
    if document.get("campaign_id") != plan.get("campaign_id") or document.get("run_id") != entry.get("run_id"):
        raise PortableReplayError(f"result plan identity differs: {path.name}")
    if document.get("repeat") != entry.get("repeat") or document.get("source_identity") != plan.get("source_identity"):
        raise PortableReplayError(f"result repeat/source differs: {path.name}")
    frozen = document.get("program_record")
    if not isinstance(frozen, dict) or frozen.get("program") != program:
        raise PortableReplayError(f"result program record differs: {path.name}")
    source = frozen.get("source")
    if not isinstance(source, dict) or source.get("path") != f"{program}.py":
        raise PortableReplayError(f"result program source record differs: {path.name}")
    source_path = safe_file(f"{program}.py")
    if sha256_file(source_path) != source.get("sha256"):
        raise PortableReplayError(f"formal wrapper source changed: {program}")
    manifest_path = safe_file(f"data/phase3/manifests/{manifest_name}.json")
    manifest_hash = sha256_file(manifest_path)
    if document.get("manifest_sha256") != manifest_hash:
        raise PortableReplayError(f"result manifest hash differs: {path.name}")
    manifest = load_json(manifest_path)
    query_keys = manifest.get("query_keys")
    if not isinstance(query_keys, list) or len(query_keys) != manifest.get("sample_count") or len(query_keys) != len(set(query_keys)):
        raise PortableReplayError(f"manifest order/count is invalid: {manifest_name}")
    evaluation = document.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("program") != program or evaluation.get("manifest") != manifest_name:
        raise PortableReplayError(f"evaluation identity differs: {path.name}")
    records = evaluation.get("records")
    if not isinstance(records, list) or [item.get("query_key") for item in records if isinstance(item, dict)] != query_keys:
        raise PortableReplayError(f"evaluation record order differs: {path.name}")
    journal_directory = document.get("journal_directory")
    hashes = document.get("journal_sha256")
    if not isinstance(journal_directory, str) or not isinstance(hashes, dict):
        raise PortableReplayError(f"journal binding missing: {path.name}")
    expected_names = [f"{index:03d}.json" for index in range(len(query_keys))]
    if list(sorted(hashes)) != expected_names:
        raise PortableReplayError(f"journal set differs: {path.name}")
    journal_root = PROJECT_ROOT / journal_directory
    normalized = []
    for index, (key, record) in enumerate(zip(query_keys, records)):
        if not isinstance(record, dict):
            raise PortableReplayError(f"record is not an object: {path.name}:{index}")
        query = _query_from_record(record, ground_truth, index)
        if query["query_key"] != key:
            raise PortableReplayError(f"manifest/record key differs: {path.name}:{index}")
        journal_name = expected_names[index]
        journal = journal_root / journal_name
        if journal.is_symlink() or not journal.is_file() or sha256_file(journal) != hashes[journal_name]:
            raise PortableReplayError(f"journal hash differs: {path.name}:{journal_name}")
        journal_relative = f"{journal_directory}/{journal_name}"
        outer_journal = formal_index.get(journal_relative)
        if (
            not isinstance(outer_journal, dict)
            or outer_journal.get("sha256") != hashes[journal_name]
            or outer_journal.get("size_bytes") != journal.stat().st_size
        ):
            raise PortableReplayError(
                f"journal lacks outer-root binding: {path.name}:{journal_name}"
            )
        journal_doc = load_json(journal)
        if journal_doc.get("attempt_id") != document.get("attempt_id"):
            raise PortableReplayError(
                f"journal/result attempt ID differs: {path.name}:{journal_name}"
            )
        normalized.append(
            validate_journal_document(
                journal_doc,
                campaign_id=str(plan["campaign_id"]),
                run_id=str(entry["run_id"]),
                program=program,
                query=query,
                query_index=index,
                manifest_name=manifest_name,
                manifest_sha256=manifest_hash,
                manifest_order_sha256=str(manifest["query_key_order_sha256"]),
                source=plan["source_identity"],
                frozen_program=frozen,
            )
        )
    if normalized != records:
        raise PortableReplayError(f"journal/result records differ: {path.name}")
    expected_evaluation = summarize_records(
        program=program,
        manifest_name=manifest_name,
        records=normalized,
        warmup_seconds=float(evaluation.get("warmup_seconds", -1)),
        warmup_metadata=evaluation.get("warmup_metadata"),
    )
    if expected_evaluation != evaluation:
        raise PortableReplayError(f"evaluation summary is not raw-derived: {path.name}")
    return document


def replay(*, verify_closure: bool = True) -> dict[str, object]:
    final = load_json(FINAL_MANIFEST)
    formal_index = _hash_index(final)
    if final.get("status") != "COMPLETE" or final.get("formal_complete") is not True or final.get("result_count") != 26:
        raise PortableReplayError("formal completion facts differ")
    if final.get("campaign_id") != NEW_CAMPAIGN_ID or final.get("protocol_id") != "phase3-recovery-v2":
        raise PortableReplayError("formal campaign/protocol differs")
    composition = final.get("composition")
    if not isinstance(composition, dict) or composition.get("old_campaign_id") != OLD_CAMPAIGN_ID:
        raise PortableReplayError("replacement composition identity differs")
    if composition.get("old_speed32_results_used") is not False or composition.get("partial_replacement_used") is not False:
        raise PortableReplayError("old/partial Speed32 evidence was used")
    plan_new = load_json(_record_hash(final.get("campaign_plan"), "new plan"))
    summary = load_json(_record_hash(final.get("campaign_summary"), "summary"))
    _record_hash(final.get("campaign_completed"), "completion")
    plan_old = load_json(_record_hash(composition.get("retained_campaign_plan"), "old plan"))
    if plan_old.get("campaign_id") != OLD_CAMPAIGN_ID or plan_new.get("campaign_id") != NEW_CAMPAIGN_ID:
        raise PortableReplayError("campaign plan identity differs")
    if len(plan_old.get("schedule", [])) != 26 or len(plan_new.get("schedule", [])) != 26:
        raise PortableReplayError("campaign schedule length differs")
    retained_hashes = summary.get("retained_result_sha256")
    replacement_hashes = summary.get("replacement_speed_result_sha256")
    if not isinstance(retained_hashes, dict) or set(retained_hashes) != {str(i) for i in range(1, 12)}:
        raise PortableReplayError("retained result hash set differs")
    if not isinstance(replacement_hashes, dict) or set(replacement_hashes) != {str(i) for i in range(12, 27)}:
        raise PortableReplayError("replacement result hash set differs")
    ground_truth = load_json(GT_PATH)
    dev_manifest = load_json(MANIFEST_ROOT / "dev128.json")
    expanded = [
        f"{figure}_{index}"
        for figure, value in ground_truth.items()
        if isinstance(value, dict)
        for index in range(len(value.get("qids", [])))
    ]
    if dev_manifest.get("query_keys") != expanded[:128]:
        raise PortableReplayError("Dev128 is not the first 128 source-order queries")
    documents = []
    attempt_closures = []
    for index in range(1, 27):
        old = index <= 11
        plan = plan_old if old else plan_new
        campaign_id = OLD_CAMPAIGN_ID if old else NEW_CAMPAIGN_ID
        entry = plan["schedule"][index - 1]
        if entry.get("index") != index:
            raise PortableReplayError(f"schedule index differs: {index}")
        expected_hash = (retained_hashes if old else replacement_hashes)[str(index)]
        result_path = CAMPAIGN_ROOT / campaign_id / str(entry["result"])
        document = verify_result(
            result_path, expected_hash, plan, entry, ground_truth, formal_index
        )
        attempt_closures.append(
            verify_attempt_closure(final, entry, document, campaign_id)
        )
        documents.append(document)
    aggregation = aggregate_results(documents)
    if aggregation != summary.get("aggregation"):
        raise PortableReplayError("sealed aggregation differs from raw recomputation")
    if aggregation["best"] != final.get("best"):
        raise PortableReplayError("final best selection differs from raw recomputation")
    from evaluation.v1_hash_erratum import verify as verify_v1_hash_erratum

    closure = verify_amended_closure(final) if verify_closure else {"skipped": True}
    erratum = verify_v1_hash_erratum(PROJECT_ROOT)
    return {
        "status": "COMPLETE",
        "formal_complete": True,
        "result_count": len(documents),
        "metrics": aggregation["metrics"],
        "best": aggregation["best"],
        "baseline_thinking_dev128": aggregation["baseline_thinking_dev128"],
        "closure": closure,
        "v1_hash_erratum": erratum,
        "attempt_cross_binding": {
            "verified": len(attempt_closures),
            "full": sum(
                item["closure_level"] == "full" for item in attempt_closures
            ),
            "legacy_partial": sum(
                item["closure_level"] != "full" for item in attempt_closures
            ),
        },
        "formal_cross_binding": {
            "legend": FIELD_PRESENCE_LEGEND,
            "field_presence_matrix": FORMAL_FIELD_PRESENCE_MATRIX,
            "verified_attempts": len(attempt_closures),
            "outer_bound_results": len(documents),
            "outer_bound_journals": sum(
                len(document["evaluation"]["records"])
                for document in documents
            ),
            "runtime_result_hash_bindings": len(attempt_closures),
            "runtime_model_gpu_attempts_verified": len(attempt_closures),
            "runtime_model_gpu_explicit_field_bindings": sum(
                12
                - len(item["runtime_missing_model_fields"])
                - len(item["runtime_missing_gpu_fields"])
                for item in attempt_closures
            ),
            "runtime_model_gpu_missing_field_occurrences": sum(
                len(item["runtime_missing_model_fields"])
                + len(item["runtime_missing_gpu_fields"])
                for item in attempt_closures
            ),
            "trusted_binding_classes": ["E-E", "E-H"],
            "legacy_caveats": [
                "runtime events omit attempt_id, campaign_id, program, and repeat",
                "start/end events omit run_id",
                "journals omit repeat and schedule index",
                "schedule index in results is path-inferred (L), not explicit",
                "runtime events are E-H bound through outer-root-bound results, but are not directly enumerated by the formal manifest",
            ],
        },
        "formal_manifest_sha256": sha256_file(FINAL_MANIFEST),
    }


def print_table(result: Mapping[str, object]) -> None:
    metrics = result["metrics"]
    print("program              Dev128       errors  Speed32 median   mean      p95       qps")
    for program in FORMAL_PROGRAMS:
        dev = metrics[program]["dev128"]
        speed = metrics[program]["speed32"]
        print(
            f"{program:20s} {dev['correct_count']:3d}/128 "
            f"{dev['error_count']:7d}  {speed['median_seconds']:14.6f} "
            f"{speed['mean_seconds']:9.6f} {speed['type7_p95_seconds']:9.6f} "
            f"{speed['throughput_qps']:.9f}"
        )
    best = result["best"]
    print(
        "best_accuracy={best_accuracy} best_speed={best_speed} "
        "best_overall={best_overall} floor_correct={best_speed_floor_correct}".format(**best)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-only", action="store_true",
        help="development diagnostic: skip current portability artifact closure",
    )
    parser.add_argument("--json", action="store_true", help="print canonical JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = replay(verify_closure=not args.formal_only)
    except (PortableReplayError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"portable evidence replay failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    else:
        print_table(result)
        print(f"formal evidence replay PASS: {result['result_count']}/26 results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
