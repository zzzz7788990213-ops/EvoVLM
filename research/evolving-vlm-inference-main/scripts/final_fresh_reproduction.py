"""One immutable dual-GPU fresh reproduction through the official evaluator."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve(strict=True).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.final_fresh_evidence import (  # noqa: E402
    LABEL,
    validate_receipt,
)
from evaluation.model_snapshot import (  # noqa: E402
    SnapshotError, validate_both_snapshots,
)
from scripts.build_final_fresh_inputs import verify as verify_inputs  # noqa: E402

INPUT_ROOT = PROJECT_ROOT / "data/final_fresh"
FORMAL_MANIFEST = PROJECT_ROOT / "evaluation/recovery_v2/final_manifest.json"
EXPECTED = {
    "baseline_instruct": 51,
    "manual_instruct": 78,
    "manual_thinking": 23,
    "evolved_instruct": 78,
    "evolved_thinking": 44,
}
LANES = {
    0: (
        ("baseline_instruct", "dev128", 128),
        ("manual_instruct", "dev128", 128),
        ("evolved_instruct", "dev128", 128),
        ("best_accuracy", "single1", 1),
        ("best_speed", "single1", 1),
        ("best_overall", "single1", 1),
    ),
    1: (
        ("manual_thinking", "dev128", 128),
        ("evolved_thinking", "dev128", 128),
        ("baseline_thinking", "single1", 1),
    ),
}


class FreshRunError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, base: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise FreshRunError(f"unsafe artifact: {path}")
    return {
        "path": path.resolve(strict=True).relative_to(base.resolve(strict=True)).as_posix(),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _strict(path: Path) -> dict[str, Any]:
    def reject(value: str) -> object:
        raise FreshRunError(f"non-finite JSON: {value}")
    value = json.loads(path.read_text(), parse_constant=reject)
    if not isinstance(value, dict):
        raise FreshRunError("JSON root is not an object")
    return value


def _write(path: Path, value: Mapping[str, object]) -> None:
    payload = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _source_identity() -> dict[str, str]:
    commit = os.environ.get("FINAL_FRESH_SOURCE_COMMIT", "")
    tree = os.environ.get("FINAL_FRESH_SOURCE_TREE", "")
    if not commit and (PROJECT_ROOT / "PACKAGE_MANIFEST.json").is_file():
        package = _strict(PROJECT_ROOT / "PACKAGE_MANIFEST.json")
        source = package.get("source", {})
        if isinstance(source, dict):
            commit, tree = str(source.get("commit", "")), str(source.get("tree", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(
        r"[0-9a-f]{40}", tree
    ):
        raise FreshRunError("frozen source commit/tree are required")
    return {"commit": commit, "tree": tree}


def _local_dependency_records(program: str) -> list[dict[str, object]]:
    queue = [f"{program}.py", "evaluate.py", "charxiv/src/descriptive_utils.py"]
    seen: set[str] = set()
    while queue:
        relative = queue.pop()
        if relative in seen:
            continue
        path = PROJECT_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise FreshRunError(f"missing local dependency: {relative}")
        seen.add(relative)
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                candidates = (
                    PROJECT_ROOT / (name.replace(".", "/") + ".py"),
                    PROJECT_ROOT / name.replace(".", "/") / "__init__.py",
                )
                for candidate in candidates:
                    if candidate.is_file() and not candidate.is_symlink():
                        queue.append(candidate.relative_to(PROJECT_ROOT).as_posix())
                        break
    records = []
    for relative in sorted(seen):
        path = PROJECT_ROOT / relative
        records.append({
            "path": relative, "sha256": _sha(path), "size_bytes": path.stat().st_size
        })
    return records


def _closure_sha(records: Sequence[Mapping[str, object]]) -> str:
    payload = json.dumps(
        list(records), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _program_plan(
    program: str,
    snapshots: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    module = importlib.import_module(program)
    fields = {
        "model_id": getattr(module, "MODEL_ID", None),
        "model_revision": getattr(module, "MODEL_REVISION", None),
        "processor_revision": getattr(module, "PROCESSOR_REVISION", None),
        "model_weight_sha256": getattr(module, "MODEL_WEIGHT_SHA256", None),
    }
    model_id = fields["model_id"]
    if not isinstance(model_id, str) or model_id not in snapshots:
        raise FreshRunError(f"program model identity missing/unsupported: {program}")
    snapshot = snapshots[model_id]
    weights = [
        item for item in snapshot["runtime_files"]
        if isinstance(item, dict) and item.get("role") == "weight"
    ]
    if len(weights) != 1 or weights[0].get("relative_path") != "model.safetensors":
        raise FreshRunError(f"program requires one frozen single-file weight: {program}")
    if (
        fields["model_revision"] != snapshot["revision"]
        or fields["processor_revision"] != snapshot["processor_revision"]
        or fields["model_weight_sha256"] != weights[0]["sha256"]
    ):
        raise FreshRunError(f"program/snapshot identity differs: {program}")
    closure = _local_dependency_records(program)
    return {
        "program": program,
        **fields,
        "model_source": snapshot["model_source"],
        "model_local_files_only": snapshot["model_local_files_only"],
        "model_weight_size_bytes": weights[0]["size_bytes"],
        "model_snapshot_manifest": {
            "path": snapshot["manifest_path"],
            "sha256": snapshot["manifest_sha256"],
            "size_bytes": snapshot["manifest_size_bytes"],
        },
        "program_source_sha256": _sha(PROJECT_ROOT / f"{program}.py"),
        "dependencies": closure,
        "dependency_closure_sha256": _closure_sha(closure),
    }


def build_plan(
    output: Path, source: Mapping[str, str],
    snapshots: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    inputs = verify_inputs(INPUT_ROOT)
    formal = {
        "path": "evaluation/recovery_v2/final_manifest.json",
        "sha256": _sha(FORMAL_MANIFEST),
        "size_bytes": FORMAL_MANIFEST.stat().st_size,
    }
    programs = sorted({task[0] for values in LANES.values() for task in values})
    identities = {
        program: _program_plan(program, snapshots) for program in programs
    }
    snapshot_records = {
        model_id: {
            "manifest": {
                "path": item["manifest_path"], "sha256": item["manifest_sha256"],
                "size_bytes": item["manifest_size_bytes"],
            },
            "revision": item["revision"],
            "processor_revision": item["processor_revision"],
            "weight_layout": item["weight_layout"],
            "runtime_files": item["runtime_files"],
        }
        for model_id, item in sorted(snapshots.items())
    }
    best_hash = identities["evolved_instruct"]["program_source_sha256"]
    for program in ("best_accuracy", "best_speed", "best_overall"):
        if identities[program]["program_source_sha256"] != best_hash:
            raise FreshRunError("best wrapper is not byte-identical to evolved_instruct")
    return {
        "schema_version": "final-fresh-reproduction-plan-v1",
        "label": LABEL,
        "run_id": output.name.rsplit(".", 1)[-1],
        "source": dict(source),
        "formal_manifest": formal,
        "model_snapshots": snapshot_records,
        "input_manifest": {
            "path": "data/final_fresh/query_images_manifest.json",
            "sha256": _sha(INPUT_ROOT / "query_images_manifest.json"),
            "size_bytes": (INPUT_ROOT / "query_images_manifest.json").stat().st_size,
            "sample_count": inputs["sample_count"],
            "query_order_sha256": inputs["query_order_sha256"],
            "independent_regular_file_count": inputs["independent_regular_file_count"],
        },
        "lanes": {
            str(lane): [
                {"program": program, "role": role, "sample_count": count}
                for program, role, count in tasks
            ]
            for lane, tasks in LANES.items()
        },
        "programs": identities,
        "expected_dev128_correct": EXPECTED,
        "formal_metrics_affected": False,
        "selection_affected": False,
        "speed32_executed": False,
        "evolution_executed": False,
        "selection_executed": False,
        "created_at_utc": _now(),
    }


def _expected_query_records(limit: int) -> list[dict[str, object]]:
    manifest = verify_inputs(INPUT_ROOT)
    return list(manifest["records"][:limit])


def validate_result(
    path: Path,
    *,
    program: str,
    role: str,
    count: int,
    physical_gpu: int,
    plan: Mapping[str, object],
) -> dict[str, Any]:
    document = _strict(path)
    evaluate = importlib.import_module("evaluate")
    scope = document.get("gpu_scope")
    if (
        document.get("program") != program
        or document.get("num_evaluated") != count
        or document.get("num_errors") != 0
        or not isinstance(scope, dict)
        or scope.get("physical_gpu_index") != physical_gpu
        or scope.get("cuda_visible_devices") != str(physical_gpu)
        or scope.get("visible_device_count") != 1
        or scope.get("logical_device") != "cuda:0"
    ):
        raise FreshRunError(f"result identity/GPU scope differs: {program}")
    program_plan = plan["programs"][program]
    if document.get("program_identity") != {
        key: program_plan[key]
        for key in ("model_id", "model_revision", "processor_revision", "model_weight_sha256")
    }:
        raise FreshRunError(f"result model identity differs: {program}")
    expected_images = _expected_query_records(count)
    records = document.get("records")
    if not isinstance(records, list) or len(records) != count:
        raise FreshRunError(f"result record count differs: {program}")
    ground_truth, expected_queries = evaluate.load_queries(
        data_path=INPUT_ROOT / "descriptive_val.json",
        images_path=INPUT_ROOT / "images",
        limit=count,
        query_image_manifest=INPUT_ROOT / "query_images_manifest.json",
    )
    correct = 0
    tokens = 0
    for index, ((key, query), image, record) in enumerate(
        zip(expected_queries, expected_images, records)
    ):
        if not isinstance(record, dict) or record.get("query_key") != key:
            raise FreshRunError(f"result query order differs: {program}:{index}")
        expected_answer = evaluate._expected_answer(ground_truth, query)
        response = record.get("response")
        derived = (
            isinstance(response, str)
            and response.strip().casefold() == expected_answer.strip().casefold()
        )
        if record.get("expected_answer") != expected_answer or record.get("correct") is not derived:
            raise FreshRunError(f"result exact-match fields differ: {program}:{index}")
        image_path = Path(str(record.get("image_path")))
        if (
            image_path.is_symlink()
            or not image_path.is_file()
            or _sha(image_path) != image["sha256"]
            or image_path.name != image["relative_path"]
        ):
            raise FreshRunError(f"result image identity differs: {program}:{index}")
        metadata = record.get("generation_metadata")
        if not isinstance(metadata, dict):
            raise FreshRunError(f"generation metadata absent: {program}:{index}")
        token_count = metadata.get("generated_token_count")
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
            raise FreshRunError(f"generated token count differs: {program}:{index}")
        for timing in ("preprocessing_seconds", "generation_seconds", "total_seconds"):
            value = metadata.get(timing)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise FreshRunError(f"generation timing differs: {program}:{index}")
        tokens += token_count
        correct += int(derived)
    if document.get("correct") != correct or document.get("accuracy") != correct / count:
        raise FreshRunError(f"result accuracy is not raw-derived: {program}")
    if role == "dev128" and correct != EXPECTED[program]:
        raise FreshRunError(
            f"Dev128 expected-count mismatch for {program}: {correct} != {EXPECTED[program]}"
        )
    return {"document": document, "correct": correct, "tokens": tokens}


def _lane_environment(output: Path, lane: int) -> dict[str, str]:
    environment = os.environ.copy()
    runtime = output / f"lane{lane}.runtime"
    cache = runtime / "cache"
    locations = {
        "TMPDIR": runtime / "tmp",
        "XDG_CACHE_HOME": cache,
        "HF_HOME": cache / "huggingface",
        "HF_HUB_CACHE": cache / "huggingface/hub",
        "HF_DATASETS_CACHE": cache / "huggingface/datasets",
        "TORCH_HOME": cache / "torch",
        "TORCH_EXTENSIONS_DIR": cache / "torch_extensions",
        "TRITON_CACHE_DIR": cache / "triton",
        "CUDA_CACHE_PATH": cache / "cuda",
        "PYTHONPYCACHEPREFIX": cache / "pycache",
    }
    for key, path in locations.items():
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        environment[key] = str(path)
    environment.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(lane),
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "EVO_NETWORK_DISABLED": "1",
        "PYTHONNOUSERSITE": "1",
    })
    return environment


def run_lane(
    output: Path,
    lane: int,
    *,
    runner: Any = subprocess.run,
) -> dict[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(lane):
        raise FreshRunError("lane process does not own its declared physical GPU")
    plan = _strict(output / "plan.json")
    lane_root = output / f"lane{lane}"
    lane_root.mkdir(mode=0o700, exist_ok=False)
    results = []
    for ordinal, (program, role, count) in enumerate(LANES[lane], 1):
        attempt_id = str(uuid.uuid4())
        stem = f"{ordinal:02d}.{program}.{role}.{attempt_id}"
        result_path = lane_root / f"{stem}.result.json"
        console_path = lane_root / f"{stem}.console.log"
        start_path = lane_root / f"{stem}.start.json"
        end_path = lane_root / f"{stem}.end.json"
        command = [
            sys.executable, "-B", str(PROJECT_ROOT / "evaluate.py"), program,
            "--limit", str(count),
            "--data", str(INPUT_ROOT / "descriptive_val.json"),
            "--images", str(INPUT_ROOT / "images"),
            "--query-images-manifest", str(INPUT_ROOT / "query_images_manifest.json"),
            "--output", str(result_path),
        ]
        _write(start_path, {
            "event": "attempt_started", "attempt_id": attempt_id,
            "run_id": plan["run_id"], "program": program, "role": role,
            "physical_gpu_index": lane, "source": plan["source"],
            "command": command, "started_at_utc": _now(),
        })
        started = time.time()
        with console_path.open("xb") as console:
            completed = runner(
                command, cwd=PROJECT_ROOT, env=os.environ.copy(),
                stdout=console, stderr=subprocess.STDOUT,
                start_new_session=True, check=False,
            )
            console.flush(); os.fsync(console.fileno())
        if completed.returncode != 0:
            _write(end_path, {
                "event": "attempt_ended", "attempt_id": attempt_id,
                "run_id": plan["run_id"], "program": program, "role": role,
                "physical_gpu_index": lane, "returncode": completed.returncode,
                "outcome": "FAILED", "console": _record(console_path, output),
                "completed_at_utc": _now(), "elapsed_seconds": time.time() - started,
            })
            raise FreshRunError(f"official evaluator failed: {program} rc={completed.returncode}")
        verified = validate_result(
            result_path, program=program, role=role, count=count,
            physical_gpu=lane, plan=plan,
        )
        result_record = _record(result_path, output)
        console_record = _record(console_path, output)
        _write(end_path, {
            "event": "attempt_ended", "attempt_id": attempt_id,
            "run_id": plan["run_id"], "program": program, "role": role,
            "physical_gpu_index": lane, "returncode": 0,
            "outcome": "COMPLETED", "result": result_record,
            "console": console_record, "completed_at_utc": _now(),
            "elapsed_seconds": time.time() - started,
        })
        identity = plan["programs"][program]
        document = verified["document"]
        results.append({
            "program": program, "role": role, "physical_gpu_index": lane,
            "sample_count": count, "correct": verified["correct"],
            "num_errors": 0, "total_time": document["total_time"],
            "avg_time_per_query": document["avg_time_per_query"],
            "generated_token_count": verified["tokens"],
            "result": result_record, "console": console_record,
            "start": _record(start_path, output),
            "end": _record(end_path, output),
            "attempt_id": attempt_id,
            "program_source_sha256": identity["program_source_sha256"],
            "dependency_closure_sha256": identity["dependency_closure_sha256"],
            "model_id": identity["model_id"],
            "model_revision": identity["model_revision"],
            "processor_revision": identity["processor_revision"],
            "model_weight_sha256": identity["model_weight_sha256"],
            "input_manifest_sha256": plan["input_manifest"]["sha256"],
        })
    lane_receipt = {
        "schema_version": "final-fresh-lane-receipt-v1",
        "label": LABEL, "run_id": plan["run_id"], "lane": lane,
        "results": results, "completed_at_utc": _now(),
    }
    _write(lane_root / "receipt.json", lane_receipt)
    return lane_receipt


def orchestrate(
    output: Path,
    source: Mapping[str, str],
    snapshots: Mapping[str, Mapping[str, object]],
    *,
    popen: Any = subprocess.Popen,
) -> dict[str, object]:
    if output.exists() or output.is_symlink():
        raise FreshRunError("fresh output already exists")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    started = _now()
    plan = build_plan(output, source, snapshots)
    _write(output / "plan.json", plan)
    processes = []
    logs = []
    try:
        for lane in sorted(LANES):
            environment = _lane_environment(output, lane)
            log = (output / f"lane{lane}.orchestrator.log").open("xb")
            logs.append(log)
            command = [
                sys.executable, "-B", "-m", "scripts.final_fresh_reproduction",
                "--lane", str(lane), "--output", str(output),
            ]
            processes.append((lane, popen(
                command, cwd=PROJECT_ROOT, env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )))
        returncodes = [(lane, process.wait()) for lane, process in processes]
    finally:
        for log in logs:
            log.flush(); os.fsync(log.fileno()); log.close()
    failed = [(lane, rc) for lane, rc in returncodes if rc != 0]
    if failed:
        raise FreshRunError(f"fresh GPU lane failure: {failed}")
    lane_receipts = [_strict(output / f"lane{lane}/receipt.json") for lane in sorted(LANES)]
    results = sorted(
        [item for receipt in lane_receipts for item in receipt["results"]],
        key=lambda item: (
            0 if item["role"] == "dev128" else 1,
            item["program"],
        ),
    )
    plan_record = _record(output / "plan.json", output)
    receipt = {
        "schema_version": "final-fresh-reproduction-receipt-v1",
        "label": LABEL, "status": "COMPLETE", "run_id": plan["run_id"],
        "source": dict(source), "plan": plan_record,
        "formal_manifest": plan["formal_manifest"],
        "started_at_utc": started, "completed_at_utc": _now(),
        "lanes": plan["lanes"], "results": results,
        "lane_receipts": [
            _record(output / f"lane{lane}/receipt.json", output)
            for lane in sorted(LANES)
        ],
        "orchestrator_logs": [
            _record(output / f"lane{lane}.orchestrator.log", output)
            for lane in sorted(LANES)
        ],
        "formal_metrics_affected": False, "selection_affected": False,
        "expected_dev128_correct": EXPECTED,
        "best_source_identity": {
            "byte_identical_to": "evolved_instruct",
            "sha256": plan["programs"]["evolved_instruct"]["program_source_sha256"],
        },
    }
    validate_receipt(receipt)
    _write(output / "receipt.json", receipt)
    return receipt


def _output(value: str | None) -> Path:
    runtime = Path(os.environ.get(
        "EVOLVING_VLM_RUNTIME_ROOT", str(PROJECT_ROOT / ".reproduce_runtime")
    )).resolve(strict=False)
    if value:
        output = Path(value)
        if not output.is_absolute():
            output = runtime / output
    else:
        output = runtime / "fresh" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "." + str(uuid.uuid4())
        )
    output = output.resolve(strict=False)
    if output == runtime or not output.is_relative_to(runtime):
        raise FreshRunError("fresh output escaped managed runtime root")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--lane", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        snapshots = validate_both_snapshots()
        output = _output(args.output)
        source = _source_identity()
        if args.dry_run:
            if output.exists() or output.is_symlink():
                raise FreshRunError("dry-run output already exists")
            plan = build_plan(output, source, snapshots)
            print(json.dumps({
                "status": "DRY_RUN_PASS", "label": LABEL,
                "output": str(output), "run_id": plan["run_id"],
                "lanes": plan["lanes"], "input_manifest": plan["input_manifest"],
            }, sort_keys=True, allow_nan=False))
        elif args.lane is not None:
            print(json.dumps(run_lane(output, args.lane), sort_keys=True, allow_nan=False))
        else:
            print(json.dumps(
                orchestrate(output, source, snapshots), sort_keys=True, allow_nan=False
            ))
    except (FreshRunError, SnapshotError, OSError, ValueError, ImportError) as exc:
        print(f"final fresh reproduction failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
