#!/usr/bin/env python3
"""Create exactly one fresh immutable active six-run Gate 1 GPU0 campaign."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TextIO
from uuid import uuid4

from evaluation.active_paths import (
    ACTIVE_VERSION_TAG,
    RECEIPT_PREFIX,
    canonical_output_filenames,
)
from evaluation.active_contract import (
    CampaignReceipt,
    CAMPAIGN_ROOT as CAMPAIGN_ROOT_RELATIVE,
    LatestCampaign,
    LATEST_PATH,
    PROTOCOL_VERSION,
    RUN_SPECS,
    RawArtifact,
    RunOutcome,
    ValidationOutcome,
    canonical_output_filename,
    canonical_run_paths,
    validate_receipt_validation_link,
)
from evaluation.atomic_artifacts import (
    finalize_stream,
    fsync_parent,
    open_text_exclusive,
    write_json_noreplace,
)
from evaluation.gpu_scope import active_gpu_metadata
from evaluation.provenance import (
    EVO_ROOT,
    FREEZE_PATH,
    PROJECT_ROOT,
    load_freeze,
    load_json_strict,
    sha256_file,
)

PYTHON = EVO_ROOT / ".venv/bin/python"
CAMPAIGN_ROOT = PROJECT_ROOT / CAMPAIGN_ROOT_RELATIVE
LATEST = PROJECT_ROOT / LATEST_PATH
RECEIPT_ROOT = EVO_ROOT / "logs/reproduce"
RUN_DEFINITIONS = tuple(
    (*spec, canonical_output_filename(*spec)) for spec in RUN_SPECS
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def torch_device_preflight() -> dict[str, object]:
    command = [
        str(PYTHON),
        "-c",
        (
            "import json, torch; "
            "c=torch.cuda.device_count(); "
            "assert c == 1, c; "
            "print(json.dumps({'torch_cuda_device_count':c,"
            "'logical_device':'cuda:0','device_name':torch.cuda.get_device_name(0)}))"
        ),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"GPU0 preflight failed with exit {completed.returncode}")
    document = json.loads(completed.stdout.strip().splitlines()[-1])
    if document["torch_cuda_device_count"] != 1:
        raise RuntimeError("GPU0 preflight did not see exactly one logical device")
    return document

class CampaignEngine(Protocol):
    eligibility: str
    def gpu_metadata(self) -> dict[str, object]: ...

    def preflight(self) -> dict[str, object]: ...

    def execute(
        self,
        run: dict[str, object],
        child_env: dict[str, str],
        stream: TextIO,
    ) -> int: ...


class ProductionEngine:
    eligibility = "gpu_production"
    def gpu_metadata(self) -> dict[str, object]:
        return active_gpu_metadata()

    def preflight(self) -> dict[str, object]:
        return torch_device_preflight()

    def execute(
        self,
        run: dict[str, object],
        child_env: dict[str, str],
        stream: TextIO,
    ) -> int:
        completed = subprocess.run(
            list(run["command"]),
            cwd=PROJECT_ROOT,
            env=child_env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return completed.returncode


def artifact(path: Path, started_ns: int | None = None) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise AssertionError(f"missing regular artifact: {path}")
    metadata = path.stat()
    if started_ns is not None and metadata.st_mtime_ns < started_ns:
        raise AssertionError(f"stale artifact predates run: {path}")
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
    }


def assert_safe_evo_destination(path: Path, label: str) -> None:
    if (
        not path.is_absolute()
        or ".." in path.parts
        or not path.is_relative_to(EVO_ROOT)
    ):
        raise ValueError(f"{label} must be lexically confined to Evo")
    current = EVO_ROOT
    root_metadata = os.lstat(current)
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("Evo root is not a physical directory")
    for part in path.relative_to(EVO_ROOT).parts[:-1]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} parent chain is unsafe: {current}")


def run_campaign(
    outputs: list[str],
    *,
    engine: CampaignEngine | None = None,
    latest_path: Path = LATEST,
    receipt_root: Path = RECEIPT_ROOT,
    campaign_id_override: str | None = None,
) -> int:
    expected_outputs = list(canonical_output_filenames())
    if outputs != expected_outputs:
        raise SystemExit("--output must list all six canonical active filenames in order")
    if Path(sys.executable) != PYTHON:
        raise SystemExit(f"must run with project venv: {sys.executable}")
    assert_safe_evo_destination(latest_path, "latest pointer")
    assert_safe_evo_destination(receipt_root / "receipt.json", "receipt root")
    selected_engine = engine if engine is not None else ProductionEngine()
    gpu_execution = selected_engine.gpu_metadata()
    if selected_engine.eligibility != "gpu_production":
        raise PermissionError("campaign engine is not GPU-production eligible")
    if latest_path.exists() or latest_path.is_symlink():
        raise FileExistsError(
            "immutable active latest pointer already exists; replay is intentionally disabled"
        )
    freeze = load_freeze()
    freeze_hash = sha256_file(FREEZE_PATH)
    torch_preflight = selected_engine.preflight()

    started_ns = time.time_ns()
    started_at = utc_now()
    campaign_id = campaign_id_override or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex
    )
    canonical_run_paths(
        campaign_id,
        RUN_SPECS[0][0],
        RUN_SPECS[0][1],
        RUN_SPECS[0][2],
        str(uuid4()),
        str(uuid4()),
        1,
    )
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    if CAMPAIGN_ROOT.is_symlink() or not CAMPAIGN_ROOT.is_dir():
        raise ValueError("campaign root is unsafe")
    campaign = CAMPAIGN_ROOT / campaign_id
    campaign.mkdir(mode=0o700, exist_ok=False)
    fsync_parent(campaign / "placeholder")
    receipt_root.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_root / f"{RECEIPT_PREFIX}_{campaign_id}.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("immutable active receipt target already exists")

    runs: list[dict[str, object]] = []
    for order, (program, manifest, repeats, filename) in enumerate(RUN_DEFINITIONS, 1):
        run_id = str(uuid4())
        runtime_event_id = str(uuid4())
        paths = canonical_run_paths(
            campaign_id,
            program,
            manifest,
            repeats,
            run_id,
            runtime_event_id,
            order,
        )
        if paths["output_filename"] != filename:
            raise AssertionError("shared path builder output mismatch")
        command = [
            str(PYTHON),
            "evaluate.py",
            program,
            "--manifest",
            paths["manifest"],
            "--warmup-manifest",
            paths["warmup_manifest"],
            "--repeats",
            str(repeats),
            "--run-id",
            run_id,
            "--output",
            paths["output_path"],
        ]
        runs.append(
            {
                "order": order,
                "run_id": run_id,
                "runtime_event_id": runtime_event_id,
                "program": program,
                "manifest_name": manifest,
                "repeats": repeats,
                **paths,
                "command": command,
                "planned_before_campaign_start": True,
            }
        )

    plan = {
        "schema_version": 4,
        "protocol_version": PROTOCOL_VERSION,
        "mode": "regenerate_all_six_once",
        "campaign_id": campaign_id,
        "campaign_path": campaign.relative_to(PROJECT_ROOT).as_posix(),
        "campaign_started_at_utc": started_at,
        "campaign_started_epoch_ns": started_ns,
        "freeze_id": freeze["freeze_id"],
        "freeze_file_sha256": freeze_hash,
        "physical_gpu_index": 0,
        "authorization_scope": "physical-gpu0-only",
        "default_future_gpu": 0,
        "execution_eligibility": selected_engine.eligibility,
        "gpu_execution": gpu_execution,
        "launcher_torch_preflight": torch_preflight,
        "runs": runs,
    }
    plan_path = campaign / "campaign_plan.json"
    write_json_noreplace(plan_path, plan)
    plan_hash = sha256_file(plan_path)
    completed_runs: list[dict[str, object]] = []

    try:
        for run in runs:
            run_started_ns = time.time_ns()
            start = {
                "schema_version": 1,
                "event": "run_start",
                "campaign_id": campaign_id,
                "order": run["order"],
                "run_id": run["run_id"],
                "runtime_event_id": run["runtime_event_id"],
                "started_at_utc": utc_now(),
                "started_epoch_ns": run_started_ns,
                "plan_sha256": plan_hash,
                "physical_gpu_index": 0,
                "cuda_visible_devices": "0",
                "visible_device_count": 1,
            }
            start_path = PROJECT_ROOT / str(run["start_event_path"])
            write_json_noreplace(start_path, start)
            console_path = PROJECT_ROOT / str(run["console_path"])
            stream = open_text_exclusive(console_path)
            child_env = os.environ.copy()
            child_env["EVO_RUN_ID"] = str(run["run_id"])
            child_env["EVO_RUNTIME_EVENT_ID"] = str(run["runtime_event_id"])
            child_env["EVO_RUNTIME_EVENT_PATH"] = str(
                PROJECT_ROOT / str(run["runtime_event_path"])
            )
            try:
                returncode = selected_engine.execute(run, child_env, stream)
            finally:
                finalize_stream(stream, console_path)
            if returncode != 0:
                raise RuntimeError(
                    f"evaluation {run['order']} failed with exit {returncode}"
                )
            summary_path = PROJECT_ROOT / str(run["output_path"])
            summary = load_json_strict(summary_path)
            outcome = RunOutcome.from_mapping(summary)
            if outcome.run_id != run["run_id"]:
                raise AssertionError("summary run ID differs from immutable plan")
            if outcome.runtime_event_id != run["runtime_event_id"]:
                raise AssertionError("runtime event ID differs from immutable plan")
            raw_path = PROJECT_ROOT / outcome.raw_log
            raw_artifact = RawArtifact.finalize(raw_path, PROJECT_ROOT)
            if raw_artifact.sha256 != outcome.raw_log_sha256:
                raise AssertionError("planner typed raw hash mismatch")
            if raw_artifact.mtime_ns < run_started_ns:
                raise AssertionError("raw artifact predates run start")
            runtime_path = PROJECT_ROOT / str(run["runtime_event_path"])
            end = {
                "schema_version": 1,
                "event": "run_end",
                "campaign_id": campaign_id,
                "order": run["order"],
                "run_id": run["run_id"],
                "runtime_event_id": run["runtime_event_id"],
                "completed_at_utc": utc_now(),
                "exit_status": returncode,
                "summary": artifact(summary_path, run_started_ns),
                "raw": raw_artifact.as_mapping(),
                "runtime_event": artifact(runtime_path, run_started_ns),
                "console": artifact(console_path, run_started_ns),
                "start_event_sha256": sha256_file(start_path),
                "plan_sha256": plan_hash,
            }
            end_path = PROJECT_ROOT / str(run["end_event_path"])
            write_json_noreplace(end_path, end)
            completed_runs.append(
                {
                    **end,
                    "start_event_path": start_path.relative_to(PROJECT_ROOT).as_posix(),
                    "start_event_sha256": sha256_file(start_path),
                    "end_event_path": end_path.relative_to(PROJECT_ROOT).as_posix(),
                    "end_event_sha256": sha256_file(end_path),
                }
            )

        validation_path = campaign / "validation.run_end.json"
        validation_console = campaign / "validation.run_end.console.log"
        validator_command = [
            str(PYTHON),
            "-m",
            "evaluation.verify_results",
            "--campaign-dir",
            campaign.relative_to(PROJECT_ROOT).as_posix(),
            "--plan",
            plan_path.relative_to(PROJECT_ROOT).as_posix(),
            "--output",
            validation_path.relative_to(PROJECT_ROOT).as_posix(),
        ]
        validation_stream = open_text_exclusive(validation_console)
        validation = subprocess.run(
            validator_command,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            stdout=validation_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        finalize_stream(validation_stream, validation_console)
        if validation.returncode != 0:
            raise RuntimeError(f"active validator failed with exit {validation.returncode}")
        validation_document = load_json_strict(validation_path)
        typed_validation = ValidationOutcome.from_mapping(validation_document)
        if validation_document["status"] != "pass":
            raise AssertionError("active validation did not pass")
        if validation_document["plan_sha256"] != plan_hash:
            raise AssertionError("validation does not anchor the campaign plan")

        receipt = {
            "schema_version": 4,
            "protocol_version": PROTOCOL_VERSION,
            "mode": "regenerate_all_six_once",
            "regenerated": True,
            "status": "pass",
            "campaign_id": campaign_id,
            "campaign_started_at_utc": started_at,
            "campaign_started_epoch_ns": started_ns,
            "completed_at_utc": utc_now(),
            "campaign_path": campaign.relative_to(PROJECT_ROOT).as_posix(),
            "freeze_id": freeze["freeze_id"],
            "freeze_file_sha256": freeze_hash,
            "plan_path": plan_path.relative_to(PROJECT_ROOT).as_posix(),
            "plan_sha256": plan_hash,
            "physical_gpu_index": 0,
            "cuda_visible_devices": "0",
            "visible_device_count": 1,
            "logical_device": "cuda:0",
            "authorization_scope": "physical-gpu0-only",
            "gpu_execution": gpu_execution,
            "launcher_torch_preflight": torch_preflight,
            "runs": completed_runs,
            "validation_path": validation_path.relative_to(PROJECT_ROOT).as_posix(),
            "validation_sha256": sha256_file(validation_path),
            "validation_console_path": validation_console.relative_to(PROJECT_ROOT).as_posix(),
            "validation_console_sha256": sha256_file(validation_console),
            "all_six_paths_and_ids_predeclared": True,
            "all_six_current_hashes_recorded": True,
            "immutable_hash_dag_complete": True,
            "stale_results_cannot_satisfy_campaign": True,
        }
        typed_receipt_before, linked_validation_before = validate_receipt_validation_link(
            receipt,
            validation_document,
            sha256_file(validation_path),
        )
        if linked_validation_before != typed_validation:
            raise AssertionError("precommit receipt/validation typed mismatch")
        write_json_noreplace(receipt_path, receipt)
        persisted_receipt = load_json_strict(receipt_path)
        typed_receipt_after, linked_validation_after = validate_receipt_validation_link(
            persisted_receipt,
            validation_document,
            sha256_file(validation_path),
        )
        if linked_validation_after != typed_validation:
            raise AssertionError("postcommit receipt/validation typed mismatch")
        if persisted_receipt != receipt or typed_receipt_after != typed_receipt_before:
            raise AssertionError("receipt changed across no-replace persistence")
        if typed_validation.campaign_id != campaign_id:
            raise AssertionError("typed validation campaign mismatch")
        if typed_validation.plan_sha256 != plan_hash:
            raise AssertionError("typed validation plan mismatch")
        if typed_receipt_after.campaign_id != campaign_id:
            raise AssertionError("typed receipt campaign mismatch")
        if typed_receipt_after.plan_sha256 != plan_hash:
            raise AssertionError("typed receipt plan mismatch")
        if typed_receipt_after.validation_sha256 != sha256_file(validation_path):
            raise AssertionError("typed receipt validation mismatch")
        latest = {
            "schema_version": 4,
            "protocol_version": PROTOCOL_VERSION,
            "campaign_id": campaign_id,
            "campaign_path": campaign.relative_to(PROJECT_ROOT).as_posix(),
            "freeze_file_sha256": freeze_hash,
            "plan_path": plan_path.relative_to(PROJECT_ROOT).as_posix(),
            "plan_sha256": plan_hash,
            "validation_path": validation_path.relative_to(PROJECT_ROOT).as_posix(),
            "validation_sha256": sha256_file(validation_path),
            "receipt_path_relative_to_evo": receipt_path.relative_to(EVO_ROOT).as_posix(),
            "receipt_sha256": sha256_file(receipt_path),
            "physical_gpu_index": 0,
            "cuda_visible_devices": "0",
            "visible_device_count": 1,
            "logical_device": "cuda:0",
            "updated_at_utc": utc_now(),
        }
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        typed_latest_before = LatestCampaign.from_mapping(latest)
        write_json_noreplace(latest_path, latest)
        persisted_latest = load_json_strict(latest_path)
        typed_latest_after = LatestCampaign.from_mapping(persisted_latest)
        if persisted_latest != latest or typed_latest_after != typed_latest_before:
            raise AssertionError("latest pointer changed across no-replace persistence")
        if typed_latest_after.campaign_id != typed_receipt_after.campaign_id:
            raise AssertionError("typed latest and receipt campaign mismatch")
        if typed_latest_after.plan_sha256 != typed_receipt_after.plan_sha256:
            raise AssertionError("typed latest and receipt plan mismatch")
        if (
            typed_latest_after.validation_sha256
            != typed_receipt_after.validation_sha256
        ):
            raise AssertionError("typed latest and receipt validation mismatch")
        print(json.dumps(persisted_latest, indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        failure = {
            "schema_version": 1,
            "status": "fail",
            "campaign_id": campaign_id,
            "failed_at_utc": utc_now(),
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "plan_path": plan_path.relative_to(PROJECT_ROOT).as_posix(),
            "plan_sha256": plan_hash,
            "completed_runs": completed_runs,
        }
        failure_path = receipt_root / f"{RECEIPT_PREFIX}_failed_{campaign_id}.{uuid4().hex}.json"
        write_json_noreplace(failure_path, failure)
        raise

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-six", action="store_true")
    parser.add_argument("--output", action="append")
    arguments = parser.parse_args()
    if arguments.canonical_six:
        if arguments.output:
            parser.error("--canonical-six cannot be combined with --output")
        outputs = list(canonical_output_filenames())
    elif arguments.output:
        outputs = arguments.output
    else:
        parser.error("--canonical-six is required for production regeneration")
    return run_campaign(outputs)



if __name__ == "__main__":
    raise SystemExit(main())
