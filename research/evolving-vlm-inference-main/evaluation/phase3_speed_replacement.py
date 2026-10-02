"""Authorized full Speed32 replacement after external GPU0 contention.

This module deliberately leaves the frozen Recovery-v2 evaluator, protocol,
manifests, rotation, warmup, and selection rules byte-identical.  It creates a
new UUID campaign plan with the original 26-entry schedule, authorizes only the
15 Speed32 entries, and never copies old Speed32 evidence into the replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evaluation import phase3_recovery as recovery
from evaluation.atomic_artifacts import (
    finalize_stream,
    open_text_exclusive,
    write_json_noreplace,
)
from evaluation.phase3_manifests import sha256_file


REPLACEMENT_VERSION = "phase3-speed32-replacement-v1"
AUDIT_VERSION = "phase3-speed32-contention-audit-v1"
MARKER_VERSION = "phase3-speed32-replacement-marker-v1"
SUMMARY_VERSION = "phase3-speed32-replacement-summary-v1"
COMPLETED_VERSION = "phase3-speed32-replacement-completed-v1"

OLD_CAMPAIGN_ID = "1fa07427-6b42-4bbb-a9f9-af63742bdfc6"
OLD_PLAN_SHA256 = "8d329631caea13add8bb8bfb39a847affc3c5377b4605e703eac7f41ab4fb118"
OLD_SUMMARY_SHA256 = "ca2866b66da4c37426849117d3fea02940522a6a6c389528967bc21372e21d3c"
OLD_COMPLETED_SHA256 = "0651a6751c450b7747397a984a52e479a0e90198fb647835ce8547a743a4658e"
EXTERNAL_PID = 1298599
EXTERNAL_START_UNIX = 1787644898.0
EXTERNAL_START_UTC = "2026-08-25T08:01:38Z"
AUTHORIZED_INDICES = tuple(range(12, 27))
RETAINED_INDICES = tuple(range(1, 12))
REPLACEMENT_SOURCE_IDENTITY = {
    "commit": "34270de76a497d9b465f5f07ceb0e12e478da276",
    "tree": "48f3331b2cfb83fb24e97ec77b06e68118e02ff4",
}

PROJECT_ROOT = recovery.PROJECT_ROOT
EVO_ROOT = recovery.EVO_ROOT
RECOVERY_ROOT = recovery.RECOVERY_ROOT
CAMPAIGN_ROOT = recovery.CAMPAIGN_ROOT
PREREG_ROOT = recovery.PREREG_ROOT
OLD_CAMPAIGN_ROOT = CAMPAIGN_ROOT / OLD_CAMPAIGN_ID
AUTHORIZATION_PATH = PREREG_ROOT / "speed32_replacement_authorization.json"
INCIDENT_ROOT = RECOVERY_ROOT / "incidents"
AUDIT_PATH = INCIDENT_ROOT / (
    OLD_CAMPAIGN_ID + ".gpu0_external_contention.audit.json"
)
VENV_PYTHON = recovery.VENV_PYTHON

FROZEN_HASHES = {
    "phase3_recovery": (
        PROJECT_ROOT / "evaluation/phase3_recovery.py",
        "5b13f1f6c8543b48192bd0210e8bb1e4446fec531e415707f46b4a7dd5d4afd8",
    ),
    "protocol": (
        PREREG_ROOT / "protocol.json",
        "a251e703c94167df9ff19dd4ff21f9c66297e1caa80d1519a0e0721e711b79dd",
    ),
    "candidate_manifest": (
        PREREG_ROOT / "candidate_manifest.json",
        "7e1f0416cc327d6257e1d235a31f7b48e384b44a8e28eed9a75245d1bbe238f9",
    ),
    "data_manifest": (
        PREREG_ROOT / "data_manifest.json",
        "d9beee3fa956313c9c7d0691c03fdadeba031937b1fc8b25eaf5a29b4bb03cbd",
    ),
    "selection_rules": (
        PREREG_ROOT / "selection_rules.json",
        "fb3a634fa042f3e473053cfa03463b045aaec8a96989627ffebf9d9b55bc52d7",
    ),
    "recovery_authorization": (
        PREREG_ROOT / "recovery_authorization.json",
        "c787e2077f54b421e65211871c54e9bdd2d2bc724451b4566f4465731281871c",
    ),
    "speed32_manifest": (
        PROJECT_ROOT / "data/phase3/manifests/speed32.json",
        "29c3c972089731c12c89c2df2ed5eb81723bf82feda9a77999e255ee10147f87",
    ),
    "warmup1_manifest": (
        PROJECT_ROOT / "data/phase3/manifests/warmup1.json",
        "201bd90ecd2478682e67a422c061c8137db9fff81d8514dc329b7450555307ea",
    ),
}

OLD_SPEED_RESULT_SHA256 = {
    "12.baseline_instruct.speed32.r1.json": "953bdc15832f6f941a2411ce173250041e5710ca53a3e620d0fb7e8d65e8ff6f",
    "13.manual_instruct.speed32.r1.json": "1e69aaa510f0ea73b69cb3de9f36cc6f18b0ff8ea00baceb19dd66fe61bc6625",
    "14.manual_thinking.speed32.r1.json": "f85c3fefe4d7a2bae74eaeaccf557aefdb1eca02adf95362212f369cb380c059",
    "15.evolved_instruct.speed32.r1.json": "ef62cc6ac96062d727d5bb80428ce1f00a84809138e5f0aa4414337b9fe0f431",
    "16.evolved_thinking.speed32.r1.json": "e344e4604bd9927a47c6a05aa2d31368875061f7b6f99d0ed5f26bb5000a0140",
    "17.manual_thinking.speed32.r2.json": "1e696f5d468dad44db2c816e08b860d9bfc269118323edde31d95a4872e2c97b",
    "18.evolved_instruct.speed32.r2.json": "019dacb447f973922cad6ea21581a54f058edb7ef34cc596aa950c926b5bedee",
    "19.evolved_thinking.speed32.r2.json": "79b20413e6346035b892dbe08f92aa6ddbd92607c9fd977c7d9b8cfe19aba761",
    "20.baseline_instruct.speed32.r2.json": "125851faddba08cf5e84bc0c11afa7a2b7f864d4710a0463e5bad983c295f701",
    "21.manual_instruct.speed32.r2.json": "3db0098fa6258ae4559603351cc6a11c530b94a1cb08cb00c27f6a6efe98c023",
    "22.evolved_thinking.speed32.r3.json": "cd6e74cdcc4b98427c2915e8372a95763cd536b7cea64e60e72fc296d25f6709",
    "23.baseline_instruct.speed32.r3.json": "0a190095c709d5eeaa867fb2b99341bb46b832421d7e4d0d99ce7b23d6c68076",
    "24.manual_instruct.speed32.r3.json": "d669b7e8c9f1e1e398452baf1090b09bf8b680fb999e8a808110ef3ee3263b76",
    "25.manual_thinking.speed32.r3.json": "65b15e289ac8994093d0ace124fc5e3d08c7ed48f2404915eb8a5f37d59f762f",
    "26.evolved_instruct.speed32.r3.json": "73b391d0a81ff78d38093f9d56b21a38474c773805638c3cf996d9cde92bf251",
}


class ReplacementError(RuntimeError):
    """Fail-closed replacement error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _expect(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise ReplacementError(f"{name} differs from frozen value")


def _safe_uuid(value: str, name: str) -> str:
    try:
        uuid.UUID(value)
    except ValueError as exc:
        raise ReplacementError(f"{name} is not a UUID") from exc
    return value


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def _record(path: Path, *, root: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ReplacementError(f"unsafe evidence file: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ReplacementError(f"evidence escaped root: {path}")
    stat_result = resolved.stat()
    return {
        "path": resolved.relative_to(root.resolve(strict=True)).as_posix(),
        "size_bytes": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "sha256": sha256_file(resolved),
    }


def _tree_manifest(root: Path) -> list[dict[str, object]]:
    if root.is_symlink() or not root.is_dir():
        raise ReplacementError("old campaign root is unsafe")
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ReplacementError(f"old campaign contains symlink: {path}")
        if path.is_file():
            records.append(_record(path, root=PROJECT_ROOT))
        elif not path.is_dir():
            raise ReplacementError(f"old campaign contains special inode: {path}")
    return records


def verify_frozen_hashes() -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, (path, expected) in FROZEN_HASHES.items():
        actual = sha256_file(path)
        _expect(actual, expected, f"frozen hash {name}")
        observed[name] = actual
    return observed


def verify_authorization() -> dict[str, Any]:
    document = recovery.load_json(AUTHORIZATION_PATH)
    _expect(
        document.get("authorization_version"),
        REPLACEMENT_VERSION,
        "replacement authorization version",
    )
    _expect(document.get("authorized_by"), "user", "replacement authorizer")
    old = document.get("old_campaign")
    if not isinstance(old, dict):
        raise ReplacementError("replacement old campaign record is invalid")
    _expect(old.get("campaign_id"), OLD_CAMPAIGN_ID, "authorized old campaign")
    _expect(old.get("campaign_plan_sha256"), OLD_PLAN_SHA256, "authorized old plan")
    _expect(old.get("campaign_summary_sha256"), OLD_SUMMARY_SHA256, "authorized old summary")
    _expect(old.get("completed_sha256"), OLD_COMPLETED_SHA256, "authorized old completion")
    _expect(old.get("speed32_status"), "NON_QUALIFYING_EXTERNAL_GPU0_CONTENTION", "old speed status")
    scope = document.get("replacement_scope")
    if not isinstance(scope, dict):
        raise ReplacementError("replacement scope is invalid")
    _expect(scope.get("authorized_schedule_indices"), list(AUTHORIZED_INDICES), "authorized indices")
    _expect(scope.get("manifest"), "speed32", "authorized manifest")
    _expect(scope.get("repeats"), 3, "authorized repeats")
    _expect(scope.get("full_replacement_required"), True, "full replacement requirement")
    _expect(scope.get("partial_replacement_forbidden"), True, "partial replacement policy")
    _expect(scope.get("old_and_new_speed_results_must_not_mix"), True, "mixing policy")
    policy = document.get("gpu_policy")
    if not isinstance(policy, dict):
        raise ReplacementError("replacement GPU policy is invalid")
    for key, expected in (
        ("host", "hopper-25"),
        ("physical_gpu_index", 0),
        ("cuda_visible_devices", "0"),
        ("visible_device_count", 1),
        ("logical_device", "cuda:0"),
        ("gpu1_must_not_be_queried_or_used", True),
        ("strictly_serial", True),
        ("continuous_idle_gate_seconds", 120),
        ("idle_poll_seconds", 15),
        ("active_telemetry_seconds", 5),
    ):
        _expect(policy.get(key), expected, f"GPU policy {key}")
    observed = verify_frozen_hashes()
    frozen = document.get("frozen_inputs")
    if not isinstance(frozen, dict):
        raise ReplacementError("replacement frozen input record is invalid")
    authorization_keys = {
        "phase3_recovery": "phase3_recovery_sha256",
        "protocol": "protocol_sha256",
        "candidate_manifest": "candidate_manifest_sha256",
        "data_manifest": "data_manifest_sha256",
        "selection_rules": "selection_rules_sha256",
        "recovery_authorization": "recovery_authorization_sha256",
        "speed32_manifest": "speed32_manifest_sha256",
        "warmup1_manifest": "warmup1_manifest_sha256",
    }
    for observed_key, document_key in authorization_keys.items():
        _expect(
            frozen.get(document_key),
            observed[observed_key],
            f"authorization frozen hash {document_key}",
        )
    _expect(
        frozen.get("runtime_evaluator_and_selection_must_be_byte_identical"),
        True,
        "authorization byte-identity policy",
    )
    return document


def _old_plan_unbound() -> dict[str, Any]:
    _expect(sha256_file(OLD_CAMPAIGN_ROOT / "campaign_plan.json"), OLD_PLAN_SHA256, "old plan hash")
    plan = recovery.load_json(OLD_CAMPAIGN_ROOT / "campaign_plan.json")
    _expect(plan.get("campaign_id"), OLD_CAMPAIGN_ID, "old plan campaign")
    schedule = plan.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != 26:
        raise ReplacementError("old schedule is not the frozen 26-entry schedule")
    return plan


def build_contamination_audit() -> dict[str, object]:
    verify_authorization()
    plan = _old_plan_unbound()
    _expect(sha256_file(OLD_CAMPAIGN_ROOT / "campaign_summary.json"), OLD_SUMMARY_SHA256, "old summary hash")
    _expect(sha256_file(OLD_CAMPAIGN_ROOT / "completed.json"), OLD_COMPLETED_SHA256, "old completion hash")
    summary = recovery.load_json(OLD_CAMPAIGN_ROOT / "campaign_summary.json")
    result_hashes = summary.get("result_sha256")
    if not isinstance(result_hashes, dict) or len(result_hashes) != 26:
        raise ReplacementError("old summary does not anchor 26 results")
    windows: list[dict[str, object]] = []
    for entry in plan["schedule"]:
        index = int(entry["index"])
        if index not in AUTHORIZED_INDICES:
            continue
        stem = str(entry["result"])[:-5]
        start = recovery.load_json(OLD_CAMPAIGN_ROOT / f"{stem}.attempt.1.start.json")
        end = recovery.load_json(OLD_CAMPAIGN_ROOT / f"{stem}.attempt.1.end.json")
        journal_dir = OLD_CAMPAIGN_ROOT / str(entry["journal_dir"])
        journals = sorted(journal_dir.glob("*.json"))
        if len(journals) != 32:
            raise ReplacementError(f"old speed journal count differs at entry {index}")
        mtimes = [item.stat().st_mtime for item in journals]
        before = sum(value < EXTERNAL_START_UNIX for value in mtimes)
        if float(end["completed_unix"]) <= EXTERNAL_START_UNIX:
            overlap = "NO_OVERLAP_WITH_EXTERNAL_PID"
        elif float(start["started_unix"]) >= EXTERNAL_START_UNIX:
            overlap = "FULL_RUN_OVERLAP"
        else:
            overlap = "PARTIAL_RUN_OVERLAP"
        result_path = OLD_CAMPAIGN_ROOT / str(entry["result"])
        _expect(
            sha256_file(result_path),
            result_hashes[str(entry["result"])],
            f"old result hash entry {index}",
        )
        windows.append(
            {
                "entry": index,
                "program": entry["program"],
                "repeat": entry["repeat"],
                "run_start_unix": start["started_unix"],
                "run_start_utc": _iso(float(start["started_unix"])),
                "run_end_unix": end["completed_unix"],
                "run_end_utc": _iso(float(end["completed_unix"])),
                "first_journal_commit_utc": _iso(mtimes[0]),
                "last_journal_commit_utc": _iso(mtimes[-1]),
                "journals_before_external_start": before,
                "journals_at_or_after_external_start": len(mtimes) - before,
                "result_sha256": sha256_file(result_path),
                "overlap": overlap,
            }
        )
    artifacts = _tree_manifest(OLD_CAMPAIGN_ROOT)
    return {
        "audit_version": AUDIT_VERSION,
        "created_at_utc": utc_now(),
        "campaign_id": OLD_CAMPAIGN_ID,
        "campaign_plan_sha256": OLD_PLAN_SHA256,
        "campaign_summary_sha256": OLD_SUMMARY_SHA256,
        "completed_sha256": OLD_COMPLETED_SHA256,
        "result_count": 26,
        "artifact_count": len(artifacts),
        "artifact_size_bytes": sum(int(item["size_bytes"]) for item in artifacts),
        "artifact_manifest": artifacts,
        "artifact_manifest_sha256": recovery.canonical_hash(artifacts),
        "external_process": {
            "pid": EXTERNAL_PID,
            "start_unix": EXTERNAL_START_UNIX,
            "start_utc": EXTERNAL_START_UTC,
            "observed_command_prefix": "python exp/scripts/run_benchmark.py run",
            "terminated_or_signaled_by_project": False,
            "gpu1_queried": False,
        },
        "speed_run_windows": windows,
        "adjudication": {
            "entire_old_speed32_set_qualifying": False,
            "status": "NON_QUALIFYING_EXTERNAL_GPU0_CONTENTION",
            "old_single_and_dev128_valid": True,
            "best_report_package_paused_until_full_replacement": True,
            "partial_replacement_forbidden": True,
            "old_new_speed_mixing_forbidden": True,
        },
        "replacement": {
            "authorized": True,
            "started_at_audit_creation": False,
            "new_uuid_required": True,
            "full_indices": list(AUTHORIZED_INDICES),
        },
    }


def create_contamination_audit() -> dict[str, object]:
    recovery.ensure_internal_directory(INCIDENT_ROOT)
    if AUDIT_PATH.exists() or AUDIT_PATH.is_symlink():
        raise FileExistsError("contamination audit already exists")
    document = build_contamination_audit()
    write_json_noreplace(AUDIT_PATH, document)
    return {
        "path": str(AUDIT_PATH.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(AUDIT_PATH),
        "artifact_count": document["artifact_count"],
    }


def verify_contamination_audit() -> dict[str, Any]:
    document = recovery.load_json(AUDIT_PATH)
    _expect(document.get("audit_version"), AUDIT_VERSION, "audit version")
    _expect(document.get("campaign_id"), OLD_CAMPAIGN_ID, "audit campaign")
    _expect(document.get("campaign_plan_sha256"), OLD_PLAN_SHA256, "audit plan")
    _expect(document.get("campaign_summary_sha256"), OLD_SUMMARY_SHA256, "audit summary")
    _expect(document.get("completed_sha256"), OLD_COMPLETED_SHA256, "audit completion")
    artifacts = _tree_manifest(OLD_CAMPAIGN_ROOT)
    _expect(document.get("artifact_manifest"), artifacts, "old campaign byte manifest")
    _expect(
        document.get("artifact_manifest_sha256"),
        recovery.canonical_hash(artifacts),
        "old campaign manifest hash",
    )
    adjudication = document.get("adjudication")
    if not isinstance(adjudication, dict):
        raise ReplacementError("audit adjudication is invalid")
    _expect(adjudication.get("entire_old_speed32_set_qualifying"), False, "old speed eligibility")
    _expect(adjudication.get("old_single_and_dev128_valid"), True, "old Dev128 status")
    return document


def replacement_marker_document(
    directory: Path, plan: Mapping[str, object]
) -> dict[str, object]:
    return {
        "marker_version": MARKER_VERSION,
        "protocol_id": recovery.PROTOCOL_ID,
        "campaign_id": plan["campaign_id"],
        "source_identity": plan["source_identity"],
        "campaign_plan_sha256": sha256_file(directory / "campaign_plan.json"),
        "authorization_path": str(AUTHORIZATION_PATH.relative_to(PROJECT_ROOT)),
        "authorization_sha256": sha256_file(AUTHORIZATION_PATH),
        "contamination_audit_path": str(AUDIT_PATH.relative_to(PROJECT_ROOT)),
        "contamination_audit_sha256": sha256_file(AUDIT_PATH),
        "supersedes_speed32_campaign_id": OLD_CAMPAIGN_ID,
        "retains_old_schedule_indices": list(RETAINED_INDICES),
        "authorized_replacement_indices": list(AUTHORIZED_INDICES),
        "old_speed32_entire_set_qualifying": False,
        "partial_replacement_allowed": False,
        "old_new_speed_mixing_allowed": False,
        "gpu_policy": {
            "host": "hopper-25",
            "physical_gpu_index": 0,
            "cuda_visible_devices": "0",
            "visible_device_count": 1,
            "logical_device": "cuda:0",
            "gpu1_queried_or_used": False,
            "strictly_serial": True,
        },
        "frozen_hashes": verify_frozen_hashes(),
    }


def create_replacement_campaign(campaign_id: str | None = None) -> Path:
    verify_authorization()
    verify_contamination_audit()
    selected = campaign_id or str(uuid.uuid4())
    _safe_uuid(selected, "replacement campaign ID")
    if selected == OLD_CAMPAIGN_ID:
        raise ReplacementError("replacement campaign must use a new UUID")
    directory = recovery.create_campaign(selected)
    plan = recovery.load_campaign_plan(directory)
    write_json_noreplace(
        directory / "speed32_replacement_marker.json",
        replacement_marker_document(directory, plan),
    )
    return directory


def _load_historical_campaign_plan(directory: Path) -> dict[str, Any]:
    try:
        resolved = directory.resolve(strict=True)
    except OSError as exc:
        raise ReplacementError("replacement campaign path cannot be resolved") from exc
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or resolved.parent != CAMPAIGN_ROOT.resolve(strict=True)
        or resolved.name != directory.name
    ):
        raise ReplacementError("replacement campaign directory is unsafe")
    try:
        uuid.UUID(directory.name)
    except ValueError as exc:
        raise ReplacementError("replacement campaign directory is not a UUID") from exc
    plan = recovery.load_json(directory / "campaign_plan.json", root=RECOVERY_ROOT)
    expected_fields = {
        "plan_version", "protocol_id", "campaign_id", "created_at",
        "source_identity", "preregistration", "physical_gpu_index",
        "cuda_visible_devices", "visible_device_count", "program_budget_seconds",
        "schedule",
    }
    _expect(set(plan), expected_fields, "historical replacement plan fields")
    _expect(plan.get("plan_version"), recovery.PLAN_VERSION, "historical plan version")
    _expect(plan.get("protocol_id"), recovery.PROTOCOL_ID, "historical plan protocol")
    _expect(plan.get("campaign_id"), directory.name, "historical plan campaign")
    _expect(
        plan.get("source_identity"),
        REPLACEMENT_SOURCE_IDENTITY,
        "historical plan source identity",
    )
    _expect(
        plan.get("preregistration"),
        recovery.verify_preregistration(),
        "historical plan preregistration",
    )
    _expect(plan.get("physical_gpu_index"), 0, "historical plan physical GPU")
    _expect(plan.get("cuda_visible_devices"), "0", "historical plan CUDA scope")
    _expect(plan.get("visible_device_count"), 1, "historical plan visible count")
    _expect(
        plan.get("program_budget_seconds"),
        int(recovery.PROGRAM_BUDGET_SECONDS),
        "historical plan program budget",
    )
    if not isinstance(plan.get("created_at"), str) or not plan["created_at"]:
        raise ReplacementError("historical plan creation timestamp is invalid")
    schedule = plan.get("schedule")
    raw_schedule = recovery.recovery_schedule()
    if not isinstance(schedule, list) or len(schedule) != len(raw_schedule):
        raise ReplacementError("historical replacement schedule length differs")
    for index, (item, raw) in enumerate(zip(schedule, raw_schedule, strict=True), start=1):
        if not isinstance(item, dict) or not isinstance(item.get("run_id"), str):
            raise ReplacementError(f"historical schedule entry {index} is invalid")
        try:
            uuid.UUID(item["run_id"])
        except ValueError as exc:
            raise ReplacementError(f"historical run ID {index} is invalid") from exc
        expected = dict(raw)
        expected.update(
            {
                "index": index,
                "run_id": item["run_id"],
                **recovery.entry_names(index, raw),
            }
        )
        _expect(item, expected, f"historical schedule entry {index}")
    return plan


def load_replacement_plan(
    directory: Path, *, historical: bool = False
) -> dict[str, Any]:
    verify_authorization()
    verify_contamination_audit()
    plan = (
        _load_historical_campaign_plan(directory)
        if historical
        else recovery.load_campaign_plan(directory)
    )
    if plan["campaign_id"] == OLD_CAMPAIGN_ID:
        raise ReplacementError("old campaign cannot be used as replacement")
    marker = recovery.load_json(
        directory / "speed32_replacement_marker.json",
        root=RECOVERY_ROOT,
    )
    _expect(marker, replacement_marker_document(directory, plan), "replacement marker")
    old_plan = _old_plan_unbound()
    for index in range(1, 27):
        old_entry = old_plan["schedule"][index - 1]
        new_entry = plan["schedule"][index - 1]
        _expect(
            (new_entry["index"], new_entry["program"], new_entry["manifest"], new_entry["repeat"]),
            (old_entry["index"], old_entry["program"], old_entry["manifest"], old_entry["repeat"]),
            f"frozen schedule entry {index}",
        )
    for index in RETAINED_INDICES:
        entry = plan["schedule"][index - 1]
        for name in ("result", "journal_dir"):
            path = directory / str(entry[name])
            if path.exists() or path.is_symlink():
                raise ReplacementError(
                    f"replacement contains forbidden retained-stage artifact at index {index}"
                )
    return plan


def verified_speed_documents(
    directory: Path, *, require_complete: bool, historical: bool = False
) -> list[dict[str, object]]:
    plan = load_replacement_plan(directory, historical=historical)
    documents: list[dict[str, object]] = []
    missing: list[int] = []
    old_hashes = set(OLD_SPEED_RESULT_SHA256.values())
    for index in AUTHORIZED_INDICES:
        entry = plan["schedule"][index - 1]
        result_path = directory / str(entry["result"])
        if not result_path.exists() and not result_path.is_symlink():
            missing.append(index)
            continue
        document = recovery.verify_result_path(
            result_path,
            directory=directory,
            plan=plan,
            entry=entry,
        )
        if sha256_file(result_path) in old_hashes:
            raise ReplacementError("old Speed32 artifact copied into replacement")
        _expect(document.get("campaign_id"), plan["campaign_id"], "replacement result campaign")
        documents.append(document)
    if require_complete and missing:
        raise ReplacementError(f"partial replacement is forbidden; missing indices {missing}")
    return documents


def _old_retained_documents() -> list[dict[str, object]]:
    plan = _old_plan_unbound()
    summary = recovery.load_json(OLD_CAMPAIGN_ROOT / "campaign_summary.json")
    _expect(sha256_file(OLD_CAMPAIGN_ROOT / "campaign_summary.json"), OLD_SUMMARY_SHA256, "old summary")
    hashes = summary.get("result_sha256")
    if not isinstance(hashes, dict):
        raise ReplacementError("old result hash map is invalid")
    documents: list[dict[str, object]] = []
    for index in RETAINED_INDICES:
        entry = plan["schedule"][index - 1]
        result_path = OLD_CAMPAIGN_ROOT / str(entry["result"])
        _expect(sha256_file(result_path), hashes[str(entry["result"])], f"retained result {index}")
        documents.append(
            recovery.verify_result_path(
                result_path,
                directory=OLD_CAMPAIGN_ROOT,
                plan=plan,
                entry=entry,
            )
        )
    return documents


def build_replacement_summary(
    directory: Path, *, historical: bool = False
) -> dict[str, object]:
    plan = load_replacement_plan(directory, historical=historical)
    old_documents = _old_retained_documents()
    speed_documents = verified_speed_documents(
        directory, require_complete=True, historical=historical
    )
    documents = old_documents + speed_documents
    if len(documents) != 26:
        raise ReplacementError("composed formal result set is not 26 entries")
    aggregation = recovery.aggregate_results(documents)
    return {
        "summary_version": SUMMARY_VERSION,
        "protocol_id": recovery.PROTOCOL_ID,
        "replacement_campaign_id": plan["campaign_id"],
        "source_identity": plan["source_identity"],
        "old_campaign_id": OLD_CAMPAIGN_ID,
        "composition": {
            "retained_old_indices": list(RETAINED_INDICES),
            "replacement_new_indices": list(AUTHORIZED_INDICES),
            "old_speed32_results_used": False,
            "partial_replacement_used": False,
        },
        "campaign_plan_sha256": sha256_file(directory / "campaign_plan.json"),
        "replacement_marker_sha256": sha256_file(directory / "speed32_replacement_marker.json"),
        "contamination_audit_sha256": sha256_file(AUDIT_PATH),
        "speed_measurement_completion_sha256": sha256_file(
            directory / "speed32_measurements_complete.json"
        ),
        "retained_result_sha256": {
            str(index): sha256_file(
                OLD_CAMPAIGN_ROOT
                / str(_old_plan_unbound()["schedule"][index - 1]["result"])
            )
            for index in RETAINED_INDICES
        },
        "replacement_speed_result_sha256": {
            str(index): sha256_file(
                directory / str(plan["schedule"][index - 1]["result"])
            )
            for index in AUTHORIZED_INDICES
        },
        "aggregation": aggregation,
    }


def speed_measurement_document(
    directory: Path, *, completed_at_utc: str | None = None,
    historical: bool = False,
) -> dict[str, object]:
    plan = load_replacement_plan(directory, historical=historical)
    documents = verified_speed_documents(
        directory, require_complete=True, historical=historical
    )
    if len(documents) != len(AUTHORIZED_INDICES):
        raise ReplacementError("complete Speed32 replacement has wrong result count")
    return {
        "measurement_completion_version": "phase3-speed32-measurements-complete-v1",
        "protocol_id": recovery.PROTOCOL_ID,
        "replacement_campaign_id": plan["campaign_id"],
        "source_identity": plan["source_identity"],
        "campaign_plan_sha256": sha256_file(directory / "campaign_plan.json"),
        "replacement_marker_sha256": sha256_file(
            directory / "speed32_replacement_marker.json"
        ),
        "contamination_audit_sha256": sha256_file(AUDIT_PATH),
        "authorized_indices": list(AUTHORIZED_INDICES),
        "result_sha256": {
            str(index): sha256_file(
                directory / str(plan["schedule"][index - 1]["result"])
            )
            for index in AUTHORIZED_INDICES
        },
        "old_speed32_results_used": False,
        "partial_replacement_used": False,
        "completed_at_utc": completed_at_utc or utc_now(),
    }


def record_speed_measurements_complete(directory: Path) -> dict[str, object]:
    path = directory / "speed32_measurements_complete.json"
    if path.exists() or path.is_symlink():
        raise FileExistsError("Speed32 measurement completion already exists")
    document = speed_measurement_document(directory)
    write_json_noreplace(path, document)
    return check_speed_measurements(directory)


def check_speed_measurements(directory: Path) -> dict[str, object]:
    path = directory / "speed32_measurements_complete.json"
    document = recovery.load_json(path, root=RECOVERY_ROOT)
    completed_at = document.get("completed_at_utc")
    if not isinstance(completed_at, str):
        raise ReplacementError("Speed32 completion timestamp is invalid")
    expected = speed_measurement_document(
        directory, completed_at_utc=completed_at, historical=True
    )
    _expect(document, expected, "Speed32 measurement completion")
    return {
        "status": "MEASUREMENTS_COMPLETE",
        "replacement_campaign_id": document["replacement_campaign_id"],
        "result_count": len(AUTHORIZED_INDICES),
        "measurement_completion_sha256": sha256_file(path),
    }


def finalize_replacement(directory: Path) -> dict[str, object]:
    check_speed_measurements(directory)
    summary_path = directory / "speed32_replacement_summary.json"
    completed_path = directory / "speed32_replacement_completed.json"
    if summary_path.exists() or summary_path.is_symlink():
        raise FileExistsError("replacement summary already exists")
    if completed_path.exists() or completed_path.is_symlink():
        raise FileExistsError("replacement completion already exists")
    summary = build_replacement_summary(directory)
    write_json_noreplace(summary_path, summary)
    _expect(recovery.load_json(summary_path, root=RECOVERY_ROOT), summary, "committed replacement summary")
    completed = {
        "completed_version": COMPLETED_VERSION,
        "protocol_id": recovery.PROTOCOL_ID,
        "replacement_campaign_id": summary["replacement_campaign_id"],
        "source_identity": summary["source_identity"],
        "campaign_plan_sha256": summary["campaign_plan_sha256"],
        "replacement_summary_sha256": sha256_file(summary_path),
        "contamination_audit_sha256": summary["contamination_audit_sha256"],
        "speed_measurement_completion_sha256": summary["speed_measurement_completion_sha256"],
        "old_speed32_entire_set_qualifying": False,
        "replacement_speed_result_count": len(AUTHORIZED_INDICES),
        "completed_at_utc": utc_now(),
    }
    write_json_noreplace(completed_path, completed)
    return check_replacement(directory)


def check_replacement(directory: Path) -> dict[str, object]:
    measurement_check = check_speed_measurements(directory)
    expected = build_replacement_summary(directory, historical=True)
    summary_path = directory / "speed32_replacement_summary.json"
    completed_path = directory / "speed32_replacement_completed.json"
    summary = recovery.load_json(summary_path, root=RECOVERY_ROOT)
    _expect(summary, expected, "replacement summary")
    completed = recovery.load_json(completed_path, root=RECOVERY_ROOT)
    _expect(completed.get("completed_version"), COMPLETED_VERSION, "replacement completion version")
    _expect(
        completed.get("replacement_campaign_id"),
        expected["replacement_campaign_id"],
        "replacement completion campaign",
    )
    _expect(
        completed.get("replacement_summary_sha256"),
        sha256_file(summary_path),
        "replacement completion summary hash",
    )
    _expect(
        completed.get("speed_measurement_completion_sha256"),
        measurement_check["measurement_completion_sha256"],
        "replacement completion measurement hash",
    )
    _expect(completed.get("replacement_speed_result_count"), 15, "replacement result count")
    _expect(completed.get("old_speed32_entire_set_qualifying"), False, "old speed eligibility")
    return {
        "status": "COMPLETE",
        "replacement_campaign_id": expected["replacement_campaign_id"],
        "result_count": 15,
        "summary_sha256": sha256_file(summary_path),
        "completed_sha256": sha256_file(completed_path),
        "best": expected["aggregation"]["best"],
    }


def parse_gpu_process_csv(stdout: str) -> list[dict[str, object]]:
    processes: list[dict[str, object]] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        pieces = [piece.strip() for piece in line.split(",", 2)]
        if len(pieces) != 3:
            raise ReplacementError("malformed GPU0 process telemetry")
        try:
            pid = int(pieces[0])
            memory = int(pieces[2])
        except ValueError as exc:
            raise ReplacementError("noninteger GPU0 process telemetry") from exc
        if pid <= 0 or memory < 0:
            raise ReplacementError("invalid GPU0 process telemetry")
        processes.append({"pid": pid, "process_name": pieces[1], "used_memory_mib": memory})
    if len({item["pid"] for item in processes}) != len(processes):
        raise ReplacementError("duplicate GPU0 process telemetry PID")
    return processes


def gpu0_processes() -> list[dict[str, object]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise ReplacementError("GPU0 telemetry command failed: " + result.stderr[:160])
    return parse_gpu_process_csv(result.stdout)


def wait_for_continuous_idle(
    *,
    snapshot: Callable[[], list[dict[str, object]]] = gpu0_processes,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    required_seconds: float = 120.0,
    poll_seconds: float = 15.0,
) -> list[dict[str, object]]:
    if required_seconds <= 0 or poll_seconds <= 0:
        raise ReplacementError("idle gate intervals must be positive")
    records: list[dict[str, object]] = []
    clean_since: float | None = None
    while True:
        observed_at = clock()
        processes = snapshot()
        records.append(
            {
                "observed_at_utc": utc_now(),
                "observed_monotonic": observed_at,
                "processes": processes,
                "clean": not processes,
            }
        )
        if processes:
            clean_since = None
        elif clean_since is None:
            clean_since = observed_at
        elif observed_at - clean_since >= required_seconds:
            return records
        sleeper(poll_seconds)


def record_idle_gate(directory: Path) -> dict[str, object]:
    recovery.preflight_gpu_environment()
    if os.uname().nodename != "hopper-25":
        raise ReplacementError("replacement host must be hopper-25")
    gate_id = str(uuid.uuid4())
    records = wait_for_continuous_idle()
    document = {
        "gate_version": "phase3-speed32-gpu0-idle-gate-v1",
        "gate_id": gate_id,
        "campaign_id": directory.name,
        "physical_gpu_index": 0,
        "cuda_visible_devices": "0",
        "gpu1_queried": False,
        "required_continuous_idle_seconds": 120,
        "poll_seconds": 15,
        "passed": True,
        "records": records,
        "completed_at_utc": utc_now(),
    }
    path = directory / f"gpu0_idle_gate.{gate_id}.json"
    write_json_noreplace(path, document)
    return {"path": path, "sha256": sha256_file(path)}


def foreign_pids(
    processes: Sequence[Mapping[str, object]], allowed_pids: set[int]
) -> list[int]:
    return sorted(
        int(item["pid"])
        for item in processes
        if int(item["pid"]) not in allowed_pids
    )


def _terminate_own_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_speed_entry_with_telemetry(
    directory: Path,
    *,
    plan: Mapping[str, object],
    index: int,
) -> dict[str, object]:
    if index not in AUTHORIZED_INDICES:
        raise ReplacementError("non-Speed32 replacement index refused")
    entry = plan["schedule"][index - 1]
    result_path = directory / str(entry["result"])
    if result_path.exists() or result_path.is_symlink():
        return recovery.verify_result_path(
            result_path,
            directory=directory,
            plan=plan,
            entry=entry,
        )
    boundary = gpu0_processes()
    if boundary:
        raise ReplacementError("GPU0 lost idle status before replacement entry")
    attempt_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    console_path = directory / f"replacement.{index:02d}.{session_id}.console.log"
    telemetry_path = directory / f"replacement.{index:02d}.{session_id}.gpu0.jsonl"
    console = open_text_exclusive(console_path)
    telemetry = open_text_exclusive(telemetry_path)
    env = dict(os.environ)
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "EVO_NETWORK_DISABLED": "1",
        }
    )
    command = [
        str(VENV_PYTHON),
        "-m",
        "evaluation.phase3_recovery",
        "run-one",
        "--campaign-id",
        str(plan["campaign_id"]),
        "--index",
        str(index),
        "--attempt-id",
        attempt_id,
    ]
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=console,
        stderr=subprocess.STDOUT,
        text=True,
        shell=False,
        start_new_session=True,
    )
    start_path = directory / f"replacement.{index:02d}.{session_id}.start.json"
    write_json_noreplace(
        start_path,
        {
            "attempt_version": "phase3-speed32-replacement-attempt-v1",
            "campaign_id": plan["campaign_id"],
            "index": index,
            "program": entry["program"],
            "repeat": entry["repeat"],
            "attempt_id": attempt_id,
            "telemetry_session_id": session_id,
            "child_pid": process.pid,
            "pre_launch_gpu0_processes": boundary,
            "started_at_utc": utc_now(),
            "command": command,
        },
    )
    foreign: list[int] = []
    next_poll = time.monotonic()
    returncode: int | None = None
    try:
        while True:
            now = time.monotonic()
            if now < next_poll:
                time.sleep(next_poll - now)
            processes = gpu0_processes()
            foreign = foreign_pids(processes, {process.pid})
            telemetry.write(
                json.dumps(
                    {
                        "observed_at_utc": utc_now(),
                        "observed_monotonic": time.monotonic(),
                        "physical_gpu_index": 0,
                        "child_pid": process.pid,
                        "processes": processes,
                        "foreign_pids": foreign,
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            telemetry.flush()
            os.fsync(telemetry.fileno())
            if foreign:
                _terminate_own_group(process)
                returncode = process.returncode
                break
            returncode = process.poll()
            if returncode is not None:
                break
            next_poll += 5.0
    finally:
        if process.poll() is None:
            _terminate_own_group(process)
        finalize_stream(console, console_path)
        finalize_stream(telemetry, telemetry_path)
    end_document = {
        "attempt_version": "phase3-speed32-replacement-attempt-v1",
        "campaign_id": plan["campaign_id"],
        "index": index,
        "program": entry["program"],
        "repeat": entry["repeat"],
        "attempt_id": attempt_id,
        "telemetry_session_id": session_id,
        "returncode": returncode,
        "foreign_pids": foreign,
        "console_sha256": sha256_file(console_path),
        "telemetry_sha256": sha256_file(telemetry_path),
        "completed_at_utc": utc_now(),
    }
    write_json_noreplace(
        directory / f"replacement.{index:02d}.{session_id}.end.json",
        end_document,
    )
    if foreign:
        incident = {
            "incident_version": "phase3-speed32-replacement-foreign-pid-v1",
            "campaign_id": plan["campaign_id"],
            "index": index,
            "own_child_pid": process.pid,
            "foreign_pids": foreign,
            "own_process_group_stopped": True,
            "foreign_process_signaled": False,
            "gpu1_queried": False,
            "third_speed_campaign_forbidden": True,
            "telemetry_sha256": sha256_file(telemetry_path),
            "created_at_utc": utc_now(),
        }
        write_json_noreplace(
            directory / f"foreign_gpu0_process.{session_id}.incident.json",
            incident,
        )
        raise ReplacementError(f"foreign GPU0 PID detected: {foreign}")
    if returncode != 0:
        raise ReplacementError(f"replacement run-one index {index} failed rc={returncode}")
    return recovery.verify_result_path(
        result_path,
        directory=directory,
        plan=plan,
        entry=entry,
    )


def run_replacement(directory: Path) -> dict[str, object]:
    recovery.preflight_gpu_environment()
    plan = load_replacement_plan(directory)
    record_idle_gate(directory)
    for index in AUTHORIZED_INDICES:
        run_speed_entry_with_telemetry(directory, plan=plan, index=index)
    verified_speed_documents(directory, require_complete=True)
    return record_speed_measurements_complete(directory)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    create = sub.add_parser("create")
    create.add_argument("--campaign-id")
    run = sub.add_parser("run")
    run.add_argument("--campaign-id", required=True)
    check_speed = sub.add_parser("check-speed")
    check_speed.add_argument("--campaign-id", required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--campaign-id", required=True)
    check = sub.add_parser("check")
    check.add_argument("--campaign-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            output = create_contamination_audit()
        elif args.command == "create":
            directory = create_replacement_campaign(args.campaign_id)
            output = {
                "campaign_id": directory.name,
                "path": str(directory.relative_to(PROJECT_ROOT)),
                "marker_sha256": sha256_file(directory / "speed32_replacement_marker.json"),
            }
        elif args.command == "run":
            output = run_replacement(recovery.campaign_directory(args.campaign_id))
        elif args.command == "check-speed":
            output = check_speed_measurements(
                recovery.campaign_directory(args.campaign_id)
            )
        elif args.command == "finalize":
            output = finalize_replacement(recovery.campaign_directory(args.campaign_id))
        else:
            output = check_replacement(recovery.campaign_directory(args.campaign_id))
    except (
        FileExistsError,
        KeyError,
        OSError,
        ReplacementError,
        recovery.RecoveryError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        print(f"phase3 speed replacement failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
