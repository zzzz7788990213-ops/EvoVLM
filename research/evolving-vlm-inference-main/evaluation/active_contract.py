"""Shared canonical path and evidence builder for the active Gate 1 protocol."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from uuid import UUID

from evaluation.active_paths import (
    CAMPAIGN_ROOT,
    FREEZE_PATH,
    IMAGE_MANIFEST_PATH,
    LATEST_PATH,
    LOCK_PATH,
    MANIFEST_ROOT,
    MODEL_MANIFEST_PATHS,
    PROTOCOL_PATH,
    PROTOCOL_VERSION,
    RUN_SPECS,
    SCHEMA_PATH,
    WARMUP_MANIFEST,
    canonical_output_filename,
)
_CAMPAIGN_ID = re.compile(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{32}")
_HASH = re.compile(r"[0-9a-f]{64}")

SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "run_id",
        "runtime_event_id",
        "started_at_utc",
        "completed_at_utc",
        "program",
        "program_metadata",
        "accuracy",
        "correct",
        "num_evaluated",
        "num_calls",
        "num_errors",
        "error_rate",
        "nondeterministic_query_keys",
        "repeats",
        "manifest",
        "warmup_manifest",
        "warmup_query_keys",
        "warmup_calls",
        "import_seconds",
        "warmup_seconds",
        "model_load_and_warmup_seconds",
        "measurement_includes_model_load",
        "measurement_wall_seconds",
        "measured_call_seconds",
        "total_time",
        "avg_time_per_query",
        "latency_seconds",
        "throughput_queries_per_second",
        "wall_throughput_queries_per_second",
        "torch_peak_memory_allocated_bytes",
        "torch_peak_memory_reserved_bytes",
        "peak_gpu_memory_allocated_bytes",
        "peak_gpu_memory_reserved_bytes",
        "generation_summary",
        "errors",
        "records",
        "status",
        "raw_log",
        "provenance",
    }
)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


@dataclass(frozen=True)
class RawArtifact:
    path: str
    sha256: str
    bytes: int
    mtime_ns: int

    @classmethod
    def finalize(cls, path: Path, project_root: Path) -> "RawArtifact":
        if not path.is_absolute() or not project_root.is_absolute():
            raise ValueError("raw artifact paths must be absolute")
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("raw artifact is not a physical regular file")
        resolved_root = project_root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        if not resolved_path.is_relative_to(resolved_root):
            raise ValueError("raw artifact escaped project root")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        final = os.lstat(path)
        before_state = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        after_state = (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
        if before_state != after_state:
            raise RuntimeError("raw artifact changed while finalizing")
        return cls(
            path=resolved_path.relative_to(resolved_root).as_posix(),
            sha256=digest.hexdigest(),
            bytes=final.st_size,
            mtime_ns=final.st_mtime_ns,
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "mtime_ns": self.mtime_ns,
        }


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    runtime_event_id: str
    status: str
    raw_log: str
    raw_log_sha256: str
    accuracy: float
    avg_time_per_query: float
    throughput_queries_per_second: float

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RunOutcome":
        if set(mapping) != SUMMARY_KEYS:
            missing = sorted(SUMMARY_KEYS - set(mapping))
            extra = sorted(set(mapping) - SUMMARY_KEYS)
            raise ValueError(
                f"summary key contract mismatch; missing={missing}, extra={extra}"
            )
        values = {key: mapping[key] for key in SUMMARY_KEYS}
        if values["schema_version"] != 4:
            raise ValueError("summary schema version mismatch")
        if values["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError("summary protocol version mismatch")
        _uuid4(values["run_id"], "run ID")
        _uuid4(values["runtime_event_id"], "runtime event ID")
        if values["status"] not in {"pass", "fail"}:
            raise ValueError("summary status is invalid")
        if not isinstance(values["raw_log"], str):
            raise TypeError("summary raw_log must be a string")
        provenance = values["provenance"]
        if not isinstance(provenance, dict):
            raise TypeError("summary provenance must be an object")
        raw_hash = provenance.get("raw_log_sha256")
        if not isinstance(raw_hash, str) or _HASH.fullmatch(raw_hash) is None:
            raise ValueError("summary provenance raw log hash is invalid")
        return cls(
            run_id=values["run_id"],
            runtime_event_id=values["runtime_event_id"],
            status=values["status"],
            raw_log=values["raw_log"],
            raw_log_sha256=raw_hash,
            accuracy=_finite_number(values["accuracy"], "accuracy"),
            avg_time_per_query=_finite_number(
                values["avg_time_per_query"], "avg_time_per_query"
            ),
            throughput_queries_per_second=_finite_number(
                values["throughput_queries_per_second"],
                "throughput_queries_per_second",
            ),
        )

    def completion_event(
        self,
        *,
        output_path: str,
        summary_sha256: str,
        raw_artifact: RawArtifact,
    ) -> dict[str, object]:
        if _HASH.fullmatch(summary_sha256) is None:
            raise ValueError("summary SHA-256 is invalid")
        if (
            self.raw_log != raw_artifact.path
            or self.raw_log_sha256 != raw_artifact.sha256
        ):
            raise AssertionError("typed outcome/raw artifact mismatch")
        return {
            "event": "evaluation_complete",
            "run_id": self.run_id,
            "runtime_event_id": self.runtime_event_id,
            "output": output_path,
            "summary_sha256": summary_sha256,
            "raw_log": raw_artifact.path,
            "raw_log_sha256": raw_artifact.sha256,
            "raw_log_bytes": raw_artifact.bytes,
            "raw_log_mtime_ns": raw_artifact.mtime_ns,
            "status": self.status,
            "accuracy": self.accuracy,
            "avg_time_per_query": self.avg_time_per_query,
            "throughput_queries_per_second": self.throughput_queries_per_second,
        }



VALIDATION_KEYS = frozenset(
    {
        "schema_version", "status", "protocol_version", "freeze_id",
        "freeze_file_sha256", "campaign_id", "campaign_started_at_utc",
        "campaign_started_epoch_ns", "plan_path", "plan_sha256",
        "validated_at_utc", "schema_executed",
        "ground_truth_reloaded_for_exact_match", "current_files_rehashed",
        "physical_gpu_index", "authorization_scope", "default_future_gpu",
        "files",
    }
)
VALIDATION_FILE_KEYS = frozenset(
    {
        "status", "run_id", "runtime_event_id", "runtime_event_path",
        "runtime_event_sha256", "program", "manifest", "repeats",
        "summary_path", "summary_mtime_ns", "summary_bytes", "summary_sha256",
        "raw_path", "raw_mtime_ns", "raw_bytes", "raw_sha256", "metrics",
    }
)
RECEIPT_KEYS = frozenset(
    {
        "schema_version", "protocol_version", "mode", "regenerated", "status",
        "campaign_id", "campaign_started_at_utc", "campaign_started_epoch_ns",
        "completed_at_utc", "campaign_path", "freeze_id",
        "freeze_file_sha256", "plan_path", "plan_sha256",
        "physical_gpu_index", "cuda_visible_devices", "visible_device_count",
        "logical_device", "authorization_scope", "gpu_execution",
        "launcher_torch_preflight", "runs", "validation_path",
        "validation_sha256", "validation_console_path",
        "validation_console_sha256", "all_six_paths_and_ids_predeclared",
        "all_six_current_hashes_recorded", "immutable_hash_dag_complete",
        "stale_results_cannot_satisfy_campaign",
    }
)
COMPLETED_RUN_KEYS = frozenset(
    {
        "schema_version", "event", "campaign_id", "order", "run_id",
        "runtime_event_id", "completed_at_utc", "exit_status", "summary",
        "raw", "runtime_event", "console", "start_event_sha256",
        "plan_sha256", "start_event_path", "end_event_path",
        "end_event_sha256",
    }
)
ARTIFACT_KEYS = frozenset({"path", "sha256", "bytes", "mtime_ns"})
LATEST_KEYS = frozenset(
    {
        "schema_version", "protocol_version", "campaign_id", "campaign_path",
        "freeze_file_sha256", "plan_path", "plan_sha256", "validation_path",
        "validation_sha256", "receipt_path_relative_to_evo", "receipt_sha256",
        "physical_gpu_index", "cuda_visible_devices", "visible_device_count",
        "logical_device", "updated_at_utc",
    }
)


def _exact_mapping(
    mapping: Mapping[str, Any], expected: frozenset[str], label: str
) -> dict[str, Any]:
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        raise ValueError(f"{label} key contract mismatch; missing={missing}, extra={extra}")
    return {key: mapping[key] for key in expected}


def _hash_value(value: object, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} is not a SHA-256")
    return value

def _campaign_value(value: object, label: str) -> str:
    if not isinstance(value, str) or _CAMPAIGN_ID.fullmatch(value) is None:
        raise ValueError(f"{label} is not a canonical campaign ID")
    return value


def _relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{label} is not a safe relative path")
    return value


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class ValidationOutcome:
    campaign_id: str
    plan_sha256: str
    files: tuple[tuple[str, str, str, str, str], ...]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ValidationOutcome":
        values = _exact_mapping(mapping, VALIDATION_KEYS, "validation")
        if (
            values["schema_version"] != 4
            or values["status"] != "pass"
            or values["protocol_version"] != PROTOCOL_VERSION
        ):
            raise ValueError("validation identity/status mismatch")
        campaign_id = _campaign_value(values["campaign_id"], "validation campaign")
        _positive_integer(
            values["campaign_started_epoch_ns"], "validation campaign start"
        )
        _hash_value(values["freeze_id"], "validation freeze ID")
        _hash_value(values["freeze_file_sha256"], "validation freeze hash")
        _relative_path(values["plan_path"], "validation plan path")
        if (
            values["schema_executed"] is not True
            or values["ground_truth_reloaded_for_exact_match"] is not True
            or values["current_files_rehashed"] is not True
            or values["physical_gpu_index"] != 0
            or values["authorization_scope"] != "physical-gpu0-only"
            or values["default_future_gpu"] != 0
        ):
            raise ValueError("validation evidence flags/GPU scope mismatch")
        files = values["files"]
        if not isinstance(files, list) or len(files) != len(RUN_SPECS):
            raise ValueError("validation must contain six files")
        typed_files = []
        for order, (item, spec) in enumerate(zip(files, RUN_SPECS), 1):
            fields = _exact_mapping(item, VALIDATION_FILE_KEYS, "validation file")
            _uuid4(fields["run_id"], "validation run ID")
            _uuid4(fields["runtime_event_id"], "validation runtime event ID")
            program, manifest, repeats = spec
            paths = canonical_run_paths(
                campaign_id,
                program,
                manifest,
                repeats,
                fields["run_id"],
                fields["runtime_event_id"],
                order,
            )
            if (
                fields["status"] != "pass"
                or fields["program"] != program
                or fields["manifest"] != manifest
                or fields["repeats"] != repeats
            ):
                raise ValueError("validation file role/status mismatch")
            expected_paths = {
                "runtime_event_path": paths["runtime_event_path"],
                "summary_path": paths["output_path"],
                "raw_path": paths["raw_log"],
            }
            for key, expected in expected_paths.items():
                if fields[key] != expected:
                    raise ValueError(f"validation file path mismatch: {key}")
            for key in ("summary_mtime_ns", "summary_bytes", "raw_mtime_ns", "raw_bytes"):
                _positive_integer(fields[key], f"validation {key}")
            if not isinstance(fields["metrics"], dict) or not fields["metrics"]:
                raise ValueError("validation metrics are absent")
            typed_files.append(
                (
                    fields["run_id"],
                    fields["runtime_event_id"],
                    _hash_value(fields["summary_sha256"], "summary hash"),
                    _hash_value(fields["raw_sha256"], "raw hash"),
                    _hash_value(fields["runtime_event_sha256"], "runtime hash"),
                )
            )
        if len({item[0] for item in typed_files}) != len(RUN_SPECS):
            raise ValueError("validation run IDs are not unique")
        return cls(
            campaign_id=campaign_id,
            plan_sha256=_hash_value(values["plan_sha256"], "validation plan hash"),
            files=tuple(typed_files),
        )


@dataclass(frozen=True)
class CampaignReceipt:
    campaign_id: str
    plan_sha256: str
    validation_sha256: str
    run_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "CampaignReceipt":
        values = _exact_mapping(mapping, RECEIPT_KEYS, "campaign receipt")
        if (
            values["schema_version"] != 4
            or values["protocol_version"] != PROTOCOL_VERSION
            or values["mode"] != "regenerate_all_six_once"
            or values["regenerated"] is not True
            or values["status"] != "pass"
        ):
            raise ValueError("campaign receipt identity/status mismatch")
        campaign_id = _campaign_value(values["campaign_id"], "receipt campaign")
        _positive_integer(values["campaign_started_epoch_ns"], "receipt start")
        for key in ("freeze_id", "freeze_file_sha256", "plan_sha256", "validation_sha256", "validation_console_sha256"):
            _hash_value(values[key], f"receipt {key}")
        for key in ("campaign_path", "plan_path", "validation_path", "validation_console_path"):
            _relative_path(values[key], f"receipt {key}")
        expected_gpu = {
            "physical_gpu_index": 0,
            "cuda_visible_devices": "0",
            "visible_device_count": 1,
            "logical_device": "cuda:0",
            "authorization_scope": "physical-gpu0-only",
        }
        if any(values[key] != expected for key, expected in expected_gpu.items()):
            raise ValueError("campaign receipt GPU scope mismatch")
        for key in ("all_six_paths_and_ids_predeclared", "all_six_current_hashes_recorded", "immutable_hash_dag_complete", "stale_results_cannot_satisfy_campaign"):
            if values[key] is not True:
                raise ValueError(f"campaign receipt evidence flag is false: {key}")
        campaign_path = f"{CAMPAIGN_ROOT}/{campaign_id}"
        if values["campaign_path"] != campaign_path:
            raise ValueError("campaign receipt path mismatch")
        if (
            values["plan_path"] != f"{campaign_path}/campaign_plan.json"
            or values["validation_path"] != f"{campaign_path}/validation.run_end.json"
            or values["validation_console_path"] != f"{campaign_path}/validation.run_end.console.log"
        ):
            raise ValueError("campaign receipt validation/plan path mismatch")
        runs = values["runs"]
        if not isinstance(runs, list) or len(runs) != len(RUN_SPECS):
            raise ValueError("campaign receipt must contain six runs")
        run_ids = []
        for expected_order, item in enumerate(runs, 1):
            fields = _exact_mapping(item, COMPLETED_RUN_KEYS, "completed run")
            if fields["order"] != expected_order or fields["exit_status"] != 0:
                raise ValueError("completed run order/status mismatch")
            _uuid4(fields["run_id"], "receipt run ID")
            _uuid4(fields["runtime_event_id"], "receipt runtime event ID")
            program, manifest, repeats = RUN_SPECS[expected_order - 1]
            paths = canonical_run_paths(
                campaign_id,
                program,
                manifest,
                repeats,
                fields["run_id"],
                fields["runtime_event_id"],
                expected_order,
            )
            if (
                fields["schema_version"] != 1
                or fields["event"] != "run_end"
                or fields["campaign_id"] != campaign_id
            ):
                raise ValueError("completed run identity mismatch")
            expected_artifacts = {
                "summary": paths["output_path"],
                "raw": paths["raw_log"],
                "runtime_event": paths["runtime_event_path"],
                "console": paths["console_path"],
            }
            for key, expected_path in expected_artifacts.items():
                if fields[key].get("path") != expected_path:
                    raise ValueError(f"completed run artifact path mismatch: {key}")
            if (
                fields["start_event_path"] != paths["start_event_path"]
                or fields["end_event_path"] != paths["end_event_path"]
            ):
                raise ValueError("completed run start/end path mismatch")
            if fields["plan_sha256"] != values["plan_sha256"]:
                raise ValueError("completed run plan hash mismatch")
            for name in ("summary", "raw", "runtime_event", "console"):
                artifact = _exact_mapping(fields[name], ARTIFACT_KEYS, f"{name} artifact")
                _hash_value(artifact["sha256"], f"{name} hash")
                _positive_integer(artifact["bytes"], f"{name} artifact bytes")
                _positive_integer(artifact["mtime_ns"], f"{name} artifact mtime")
                _relative_path(artifact["path"], f"{name} artifact path")
            for name in (
                "start_event_sha256", "end_event_sha256", "plan_sha256"
            ):
                _hash_value(fields[name], name)
            run_ids.append(fields["run_id"])
        if len(set(run_ids)) != len(RUN_SPECS):
            raise ValueError("campaign receipt run IDs are not unique")
        return cls(
            campaign_id=campaign_id,
            plan_sha256=_hash_value(values["plan_sha256"], "receipt plan hash"),
            validation_sha256=_hash_value(
                values["validation_sha256"], "receipt validation hash"
            ),
            run_ids=tuple(run_ids),
        )


@dataclass(frozen=True)
class LatestCampaign:
    campaign_id: str
    plan_sha256: str
    validation_sha256: str
    receipt_sha256: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "LatestCampaign":
        values = _exact_mapping(mapping, LATEST_KEYS, "latest campaign")
        if (
            values["schema_version"] != 4
            or values["protocol_version"] != PROTOCOL_VERSION
            or values["physical_gpu_index"] != 0
            or values["cuda_visible_devices"] != "0"
            or values["visible_device_count"] != 1
            or values["logical_device"] != "cuda:0"
        ):
            raise ValueError("latest campaign identity/GPU mismatch")
        campaign_id = _campaign_value(values["campaign_id"], "latest campaign")
        for key in ("campaign_path", "plan_path", "validation_path", "receipt_path_relative_to_evo"):
            _relative_path(values[key], f"latest {key}")
        for key in ("freeze_file_sha256", "plan_sha256", "validation_sha256", "receipt_sha256"):
            _hash_value(values[key], f"latest {key}")
        campaign_path = f"{CAMPAIGN_ROOT}/{campaign_id}"
        if values["campaign_path"] != campaign_path:
            raise ValueError("latest campaign path mismatch")
        if (
            values["plan_path"] != f"{campaign_path}/campaign_plan.json"
            or values["validation_path"] != f"{campaign_path}/validation.run_end.json"
        ):
            raise ValueError("latest plan/validation path mismatch")
        return cls(
            campaign_id=campaign_id,
            plan_sha256=_hash_value(values["plan_sha256"], "latest plan hash"),
            validation_sha256=_hash_value(
                values["validation_sha256"], "latest validation hash"
            ),
            receipt_sha256=_hash_value(
                values["receipt_sha256"], "latest receipt hash"
            ),
        )


def validate_receipt_validation_link(
    receipt: Mapping[str, Any], validation: Mapping[str, Any], validation_sha256: str
) -> tuple[CampaignReceipt, ValidationOutcome]:
    typed_receipt = CampaignReceipt.from_mapping(receipt)
    typed_validation = ValidationOutcome.from_mapping(validation)
    actual_validation_hash = _hash_value(validation_sha256, "linked validation hash")
    if typed_receipt.campaign_id != typed_validation.campaign_id:
        raise AssertionError("receipt/validation campaign mismatch")
    if typed_receipt.plan_sha256 != typed_validation.plan_sha256:
        raise AssertionError("receipt/validation plan mismatch")
    if typed_receipt.validation_sha256 != actual_validation_hash:
        raise AssertionError("receipt validation hash mismatch")
    validation_by_id = {item["run_id"]: item for item in validation["files"]}
    if len(validation_by_id) != len(RUN_SPECS):
        raise AssertionError("validation run IDs are not unique")
    for run in receipt["runs"]:
        item = validation_by_id[run["run_id"]]
        if item["runtime_event_id"] != run["runtime_event_id"]:
            raise AssertionError("receipt/validation runtime event mismatch")
        hash_pairs = (
            (run["summary"]["sha256"], item["summary_sha256"]),
            (run["raw"]["sha256"], item["raw_sha256"]),
            (run["runtime_event"]["sha256"], item["runtime_event_sha256"]),
        )
        if any(receipt_hash != validation_hash for receipt_hash, validation_hash in hash_pairs):
            raise AssertionError("receipt/validation run artifact hash mismatch")
    return typed_receipt, typed_validation
def _uuid4(value: str, label: str) -> None:
    parsed = UUID(value)
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{label} must be canonical UUIDv4")


def canonical_manifest_path(manifest: str) -> str:
    if manifest not in {spec[1] for spec in RUN_SPECS}:
        raise ValueError("manifest role is not canonical")
    return f"{MANIFEST_ROOT}/{manifest}.json"


def canonical_run_paths(
    campaign_id: str,
    program: str,
    manifest: str,
    repeats: int,
    run_id: str,
    runtime_event_id: str,
    order: int,
) -> dict[str, str]:
    if _CAMPAIGN_ID.fullmatch(campaign_id) is None:
        raise ValueError("campaign ID is not canonical")
    if not 1 <= order <= len(RUN_SPECS):
        raise ValueError("run order is not canonical")
    _uuid4(run_id, "run ID")
    _uuid4(runtime_event_id, "runtime event ID")
    filename = canonical_output_filename(program, manifest, repeats)
    stem = filename.removesuffix(".json")
    campaign = f"{CAMPAIGN_ROOT}/{campaign_id}"
    return {
        "campaign_path": campaign,
        "output_filename": filename,
        "output_path": f"{campaign}/{filename}",
        "raw_log": f"{campaign}/{stem}.raw.jsonl",
        "console_path": f"{campaign}/{stem}.console.log",
        "runtime_event_path": f"{campaign}/runtime_event.{runtime_event_id}.json",
        "start_event_path": f"{campaign}/run_start.{order:02d}.{run_id}.json",
        "end_event_path": f"{campaign}/run_end.{order:02d}.{run_id}.json",
        "manifest": canonical_manifest_path(manifest),
        "warmup_manifest": WARMUP_MANIFEST,
    }


def canonical_provenance_paths(model_id: str) -> dict[str, str]:
    try:
        model_manifest = MODEL_MANIFEST_PATHS[model_id]
    except KeyError as exc:
        raise ValueError("model ID is not canonical") from exc
    return {
        "freeze_path": FREEZE_PATH,
        "protocol_path": PROTOCOL_PATH,
        "result_schema_path": SCHEMA_PATH,
        "requirements_lock_path": LOCK_PATH,
        "image_manifest_path": IMAGE_MANIFEST_PATH,
        "model_manifest_path": model_manifest,
    }


def attach_summary_evidence(
    core_result: dict[str, Any],
    provenance: dict[str, Any],
    paths: dict[str, str],
    *,
    raw_log_sha256: str,
    runtime_event_sha256: str,
    input_image_sha256: dict[str, str],
) -> dict[str, Any]:
    if _HASH.fullmatch(raw_log_sha256) is None:
        raise ValueError("raw log SHA-256 is invalid")
    if _HASH.fullmatch(runtime_event_sha256) is None:
        raise ValueError("runtime event SHA-256 is invalid")
    model_id = core_result["program_metadata"]["MODEL_ID"]
    expected_provenance_paths = canonical_provenance_paths(model_id)
    result = copy.deepcopy(core_result)
    evidence = copy.deepcopy(provenance)
    evidence.update(expected_provenance_paths)
    evidence.update(
        {
            "raw_log_sha256": raw_log_sha256,
            "runtime_event_path": paths["runtime_event_path"],
            "runtime_event_sha256": runtime_event_sha256,
            "input_image_sha256": dict(input_image_sha256),
        }
    )
    result.update(
        {
            "manifest": paths["manifest"],
            "warmup_manifest": paths["warmup_manifest"],
            "raw_log": paths["raw_log"],
            "provenance": evidence,
        }
    )
    return result
