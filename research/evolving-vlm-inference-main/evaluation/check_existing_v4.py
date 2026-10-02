#!/usr/bin/env python3
"""Verify the immutable active hash DAG without altering canonical artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from evaluation.active_paths import (
    ACTIVE_VERSION_TAG,
    CAMPAIGN_ROOT,
    CHECK_PREFIX,
    LATEST_PATH,
    PROTOCOL_VERSION,
)
from evaluation.active_contract import (
    CampaignReceipt,
    LatestCampaign,
    ValidationOutcome,
    validate_receipt_validation_link,
)
from evaluation.atomic_artifacts import (
    finalize_stream,
    open_text_exclusive,
    write_json_noreplace,
)
from evaluation.gpu_scope import active_gpu_metadata
from evaluation.provenance import EVO_ROOT, PROJECT_ROOT, load_json_strict, sha256_file

LATEST = PROJECT_ROOT / LATEST_PATH
LOG_ROOT = EVO_ROOT / "logs/reproduce"


def snapshot(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise AssertionError(f"canonical artifact is not a regular file: {path}")
    return {
        "sha256": sha256_file(path),
        "mtime_ns": path.stat().st_mtime_ns,
        "bytes": path.stat().st_size,
    }


def verify_chain(latest_path: Path = LATEST) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    gpu = active_gpu_metadata()
    latest = load_json_strict(latest_path)
    typed_latest = LatestCampaign.from_mapping(latest)
    if (
        latest.get("protocol_version") != PROTOCOL_VERSION
        or latest.get("physical_gpu_index") != 0
        or latest.get("cuda_visible_devices") != "0"
        or latest.get("visible_device_count") != 1
    ):
        raise AssertionError("active latest GPU/protocol metadata mismatch")
    receipt_path = EVO_ROOT / latest["receipt_path_relative_to_evo"]
    if sha256_file(receipt_path) != latest["receipt_sha256"]:
        raise AssertionError("latest receipt hash mismatch")
    receipt = load_json_strict(receipt_path)
    typed_receipt = CampaignReceipt.from_mapping(receipt)
    if receipt["campaign_id"] != latest["campaign_id"]:
        raise AssertionError("receipt/latest campaign mismatch")
    campaign = PROJECT_ROOT / latest["campaign_path"]
    if campaign.resolve().parent != (
        PROJECT_ROOT / CAMPAIGN_ROOT
    ).resolve():
        raise AssertionError("campaign escaped v4 root")
    plan_path = PROJECT_ROOT / latest["plan_path"]
    validation_path = PROJECT_ROOT / latest["validation_path"]
    if sha256_file(plan_path) != latest["plan_sha256"]:
        raise AssertionError("latest plan hash mismatch")
    if receipt["plan_sha256"] != latest["plan_sha256"]:
        raise AssertionError("receipt/latest plan hash mismatch")
    validation_hash = sha256_file(validation_path)
    if validation_hash != latest["validation_sha256"]:
        raise AssertionError("latest canonical validation hash mismatch")
    if receipt["validation_sha256"] != latest["validation_sha256"]:
        raise AssertionError("receipt/latest validation declaration mismatch")
    validation = load_json_strict(validation_path)
    typed_validation = ValidationOutcome.from_mapping(validation)
    linked_receipt, linked_validation = validate_receipt_validation_link(
        receipt,
        validation,
        validation_hash,
    )
    if linked_receipt != typed_receipt or linked_validation != typed_validation:
        raise AssertionError("typed receipt/validation link changed across reload")

    if validation["plan_sha256"] != latest["plan_sha256"]:
        raise AssertionError("validation does not anchor the plan")
    if validation["campaign_id"] != latest["campaign_id"]:
        raise AssertionError("validation/latest campaign mismatch")
    if typed_latest.campaign_id != typed_receipt.campaign_id:
        raise AssertionError("typed latest/receipt campaign mismatch")
    if typed_latest.plan_sha256 != typed_receipt.plan_sha256:
        raise AssertionError("typed latest/receipt plan mismatch")
    if typed_latest.validation_sha256 != typed_receipt.validation_sha256:
        raise AssertionError("typed latest/receipt validation mismatch")
    if typed_validation.campaign_id != typed_latest.campaign_id:
        raise AssertionError("typed validation/latest campaign mismatch")
    if typed_validation.plan_sha256 != typed_latest.plan_sha256:
        raise AssertionError("typed validation/latest plan mismatch")
    validated = {entry["run_id"]: entry for entry in validation["files"]}
    if len(validated) != 6 or len(receipt["runs"]) != 6:
        raise AssertionError("hash DAG does not contain six unique runs")

    paths = [latest_path, receipt_path, plan_path, validation_path]
    validation_console = PROJECT_ROOT / receipt["validation_console_path"]
    if sha256_file(validation_console) != receipt["validation_console_sha256"]:
        raise AssertionError("validation console hash mismatch")
    paths.append(validation_console)
    for run in receipt["runs"]:
        item = validated[run["run_id"]]
        for kind in ("summary", "raw", "runtime_event"):
            info = run[kind]
            path = PROJECT_ROOT / info["path"]
            digest = sha256_file(path)
            if digest != info["sha256"] or digest != item[f"{kind}_sha256"]:
                raise AssertionError(f"{kind} hash chain mismatch")
            paths.append(path)
        console = PROJECT_ROOT / run["console"]["path"]
        if sha256_file(console) != run["console"]["sha256"]:
            raise AssertionError("run console hash mismatch")
        paths.append(console)
        for name in ("start_event", "end_event"):
            path = PROJECT_ROOT / run[f"{name}_path"]
            if sha256_file(path) != run[f"{name}_sha256"]:
                raise AssertionError(f"{name} hash mismatch")
            paths.append(path)
    canonical = {
        path.relative_to(EVO_ROOT).as_posix(): snapshot(path)
        for path in sorted(set(paths))
    }
    return canonical, {
        "latest": latest,
        "receipt": receipt,
        "gpu_execution": gpu,
        "receipt_path": receipt_path.relative_to(EVO_ROOT).as_posix(),
    }


def run_check(
    latest_path: Path = LATEST, log_root: Path = LOG_ROOT
) -> dict[str, object]:
    before, context = verify_chain(latest_path)
    check_id = str(uuid4())
    campaign = PROJECT_ROOT / context["latest"]["campaign_path"]
    checks = campaign / "checks"
    checks.mkdir(mode=0o700, exist_ok=True)
    if checks.is_symlink() or not checks.is_dir():
        raise ValueError("active checks directory is unsafe")
    check_path = checks / f"check.{check_id}.json"
    console_path = checks / f"check.{check_id}.console.log"
    command = [
        sys.executable,
        "-m",
        "evaluation.verify_results",
        "--campaign-dir",
        context["latest"]["campaign_path"],
        "--plan",
        context["latest"]["plan_path"],
        "--output",
        check_path.relative_to(PROJECT_ROOT).as_posix(),
    ]
    stream = open_text_exclusive(console_path)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
    )
    finalize_stream(stream, console_path)
    if completed.returncode != 0:
        raise RuntimeError(f"current active semantic check failed with exit {completed.returncode}")
    check = load_json_strict(check_path)
    if check["status"] != "pass":
        raise AssertionError("current active semantic check did not pass")
    after, post_context = verify_chain(latest_path)
    if before != after:
        raise AssertionError("canonical active hashes or mtimes changed during check")
    receipt = {
        "schema_version": 1,
        "mode": f"check_existing_{ACTIVE_VERSION_TAG}_only",
        "regenerated": False,
        "status": "pass",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "check_id": check_id,
        "campaign_id": context["latest"]["campaign_id"],
        "canonical_immutable_before_after_equal": True,
        "canonical_artifact_count": len(before),
        "canonical_artifacts": before,
        "check_validation_path": check_path.relative_to(EVO_ROOT).as_posix(),
        "check_validation_sha256": sha256_file(check_path),
        "check_console_path": console_path.relative_to(EVO_ROOT).as_posix(),
        "check_console_sha256": sha256_file(console_path),
        "latest_receipt_hash_verified": True,
        "receipt_latest_validation_hash_verified": True,
        "six_raw_summary_runtime_hashes_verified": True,
        "statement": "Existing artifacts were re-hashed; inference was not regenerated.",
    }
    log_root.mkdir(parents=True, exist_ok=True)
    output = log_root / f"{CHECK_PREFIX}_{check_id}.json"
    write_json_noreplace(output, receipt)
    print(
        json.dumps(
            {
                "status": "pass",
                "check_id": check_id,
                "receipt_path": output.relative_to(EVO_ROOT).as_posix(),
                "receipt_sha256": sha256_file(output),
                "canonical_artifacts_unchanged": True,
            },
            sort_keys=True,
        )
    )
    return {
        "status": "pass",
        "check_id": check_id,
        "receipt_path": output.relative_to(EVO_ROOT).as_posix(),
        "receipt_sha256": sha256_file(output),
        "canonical_artifacts_unchanged": True,
    }


def main() -> int:
    run_check()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
