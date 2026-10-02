"""Immutable valid-child authorization receipt for formal local evolution."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .candidate_runtime import CANDIDATE_RUNTIME_VERSION
from .catalog_provider import (
    PROVIDER_NAME,
    PROVIDER_VERSION,
    CatalogProviderError,
    validate_catalog_provider_log,
)
from .dsl import (
    DSL_VERSION,
    MODEL_METADATA,
    MODEL_IDS,
    canonical_candidate_source,
    parse_candidate_dsl,
)
from .local_provider import (
    MutationOutcome,
    MutationRequest,
    mutation_focus,
    mutation_focus_fields,
)
from .mutation_catalog import CATALOG_VERSION
from .process_safety import EVO_ROOT, PROJECT_ROOT, atomic_create_json
from .run_manifest import RUN_MANIFEST_VERSION, verify_run_manifest
from .renderer import (
    RENDERER_VERSION,
    rendered_program_diff,
    rendered_program_sha256,
)
from .target_worker import TARGET_WORKER_VERSION


RECEIPT_VERSION = "phase2-valid-child-receipt-v3"
RECEIPT_ROOT = PROJECT_ROOT / "results/evolution/valid_child_receipts"
_GPU0_SCOPE = {
    "physical_gpu_index": 0,
    "cuda_visible_devices": "0",
    "visible_device_count": 1,
    "logical_device": "cuda:0",
}


class ValidChildReceiptError(RuntimeError):
    pass


def _slug(model_id: str) -> str:
    if model_id.endswith("-Instruct"):
        return "instruct"
    if model_id.endswith("-Thinking"):
        return "thinking"
    raise ValidChildReceiptError("receipt model is unsupported")


def receipt_path(model_id: str) -> Path:
    if model_id not in MODEL_IDS:
        raise ValidChildReceiptError("receipt model is unsupported")
    return RECEIPT_ROOT / f"{_slug(model_id)}.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _physical_artifact(path: Path) -> Path:
    if not path.is_absolute() or not path.is_relative_to(EVO_ROOT):
        raise ValidChildReceiptError("receipt artifact escaped Evo")
    if path.is_symlink() or not path.is_file():
        raise ValidChildReceiptError("receipt artifact is not a physical file")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(EVO_ROOT):
        raise ValidChildReceiptError("receipt artifact resolved outside Evo")
    return resolved


def _artifact_meta(path: Path) -> dict[str, object]:
    resolved = _physical_artifact(path)
    stat = resolved.stat()
    return {
        "path": resolved.relative_to(EVO_ROOT).as_posix(),
        "sha256": _sha256_file(resolved),
        "size_bytes": stat.st_size,
    }


def _read_json(path: Path) -> dict[str, object]:
    resolved = _physical_artifact(path)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidChildReceiptError("receipt dependency is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValidChildReceiptError("receipt dependency is not an object")
    return value


def _git_output(*arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(EVO_ROOT), *arguments],
        cwd=str(EVO_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
        check=False,
    )
    if process.returncode != 0:
        raise ValidChildReceiptError(
            "git source identity failed: " + process.stderr.strip()[:512]
        )
    return process.stdout.strip()


def current_source_identity() -> dict[str, str]:
    project_relative = PROJECT_ROOT.relative_to(EVO_ROOT).as_posix()
    dirty = _git_output(
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
    )
    if dirty:
        raise ValidChildReceiptError(
            "formal receipt requires a clean tracked project tree"
        )
    return {
        "commit": _git_output("rev-parse", "HEAD"),
        "tree": _git_output("rev-parse", "HEAD^{tree}"),
        "project_relative_path": project_relative,
    }


def _versions(campaign_version: str) -> dict[str, str]:
    return {
        "receipt": RECEIPT_VERSION,
        "campaign": campaign_version,
        "provider": PROVIDER_VERSION,
        "catalog": CATALOG_VERSION,
        "dsl": DSL_VERSION,
        "renderer": RENDERER_VERSION,
        "candidate_runtime": CANDIDATE_RUNTIME_VERSION,
        "target_worker": TARGET_WORKER_VERSION,
        "run_manifest": RUN_MANIFEST_VERSION,
    }


def _validated_smoke(
    *,
    model_id: str,
    child: Mapping[str, object],
    provider: Mapping[str, object],
    target: Mapping[str, object],
    run_manifest_sha256: str,
) -> dict[str, object]:
    if provider.get("provider_name") != PROVIDER_NAME:
        raise ValidChildReceiptError("provider identity mismatch")
    if provider.get("provider_version") != PROVIDER_VERSION:
        raise ValidChildReceiptError("provider version mismatch")
    if (
        provider.get("execution_class") != "production_evolution"
        or provider.get("run_manifest_sha256") != run_manifest_sha256
    ):
        raise ValidChildReceiptError(
            "provider run manifest identity mismatch"
        )
    if (
        provider.get("network_attempts") != 0
        or provider.get("child_reaped") is not True
        or provider.get("gpu_scope") != _GPU0_SCOPE
    ):
        raise ValidChildReceiptError(
            "provider safety attestation failed"
        )
    mutation_evidence = provider.get("model_evidence")
    provider_model = next(
        model for model in MODEL_IDS if model.endswith("-Instruct")
    )
    if (
        not isinstance(mutation_evidence, dict)
        or mutation_evidence.get("model_id") != provider_model
        or mutation_evidence.get("revision")
        != MODEL_METADATA[provider_model]["revision"]
        or mutation_evidence.get("weight_sha256")
        != MODEL_METADATA[provider_model]["weight_sha256"]
    ):
        raise ValidChildReceiptError(
            "provider model provenance mismatch"
        )
    try:
        validated_log_requests, validated_log_outcomes = (
            validate_catalog_provider_log(provider)
        )
    except CatalogProviderError as exc:
        raise ValidChildReceiptError("catalog provider ledger is invalid") from exc
    requests = provider.get("requests")
    outcomes = provider.get("outcomes")
    if (
        not isinstance(requests, list)
        or len(requests) != 1
        or not isinstance(requests[0], dict)
        or set(requests[0]) != (
            set(MutationRequest.__dataclass_fields__) | {"mutation_focus"}
        )
    ):
        raise ValidChildReceiptError(
            "smoke mutation request is not exact"
        )
    if (
        not isinstance(outcomes, list)
        or len(outcomes) != 1
        or not isinstance(outcomes[0], dict)
        or set(outcomes[0])
        != set(MutationOutcome.__dataclass_fields__)
    ):
        raise ValidChildReceiptError(
            "smoke mutation outcome is not exact"
        )
    request = requests[0]
    request_contract = dict(request)
    recorded_focus = request_contract.pop("mutation_focus")
    try:
        validated_request = MutationRequest(**request_contract)
    except (TypeError, ValueError) as exc:
        raise ValidChildReceiptError(
            "smoke mutation request contract is invalid"
        ) from exc
    if recorded_focus != mutation_focus(validated_request):
        raise ValidChildReceiptError("smoke mutation focus mismatch")
    if validated_log_requests != [validated_request]:
        raise ValidChildReceiptError("smoke request catalog validation mismatch")
    if (
        validated_request.execution_class != "production_evolution"
        or validated_request.run_manifest_sha256 != run_manifest_sha256
    ):
        raise ValidChildReceiptError(
            "smoke request run manifest identity mismatch"
        )
    outcome = outcomes[0]
    if [outcome] != [item.to_mapping() for item in validated_log_outcomes]:
        raise ValidChildReceiptError("smoke outcome catalog validation mismatch")
    if (
        outcome.get("status") != "accepted"
        or outcome.get("request_id") != request.get("request_id")
        or child.get("request_id") != request.get("request_id")
        or child.get("parent_id") != request.get("parent_id")
        or child.get("model_id") != model_id
        or child.get("generation") != 1
    ):
        raise ValidChildReceiptError(
            "smoke mutation identity/status mismatch"
        )
    child_hash = child.get("rendered_sha256")
    canonical_dsl = child.get("canonical_dsl")
    child_source = child.get("rendered_source")
    parent_source = request.get("parent_program_source")
    parent_hash = request.get("parent_program_sha256")
    diff = child.get("unified_diff")
    static_receipt = child.get("static_receipt")
    if (
        not isinstance(child_hash, str)
        or len(child_hash) != 64
        or not isinstance(canonical_dsl, str)
        or not isinstance(child_source, str)
        or not isinstance(parent_source, str)
        or not isinstance(parent_hash, str)
        or len(parent_hash) != 64
        or not isinstance(diff, str)
        or not diff
        or not isinstance(static_receipt, dict)
    ):
        raise ValidChildReceiptError(
            "child mutation evidence scalar is invalid"
        )
    try:
        child_spec = parse_candidate_dsl(
            canonical_dsl,
            expected_model_id=model_id,
        )
    except Exception as exc:
        raise ValidChildReceiptError(
            "child canonical DSL is invalid"
        ) from exc
    parent_spec = parse_candidate_dsl(
        validated_request.parent_dsl,
        expected_model_id=model_id,
    )
    focus_fields = mutation_focus_fields(validated_request)
    focus_differences = {
        field: {
            "parent": getattr(parent_spec, field),
            "child": getattr(child_spec, field),
        }
        for field in focus_fields
        if getattr(parent_spec, field) != getattr(child_spec, field)
    }
    if not focus_differences:
        raise ValidChildReceiptError(
            "child did not change the preregistered mutation focus"
        )
    if (
        canonical_candidate_source(child_spec) != canonical_dsl
        or rendered_program_sha256(child_source) != child_hash
        or rendered_program_sha256(parent_source) != parent_hash
        or rendered_program_diff(parent_source, child_source) != diff
        or outcome.get("canonical_dsl") != canonical_dsl
        or outcome.get("rendered_source") != child_source
        or outcome.get("rendered_sha256") != child_hash
        or static_receipt.get("program_sha256") != child_hash
        or static_receipt.get("renderer_version")
        != RENDERER_VERSION
    ):
        raise ValidChildReceiptError(
            "child diff/static/import validation failed"
        )
    calls = outcome.get("calls")
    proposals = outcome.get("proposals")
    if (
        not isinstance(calls, list)
        or len(calls) != 1
        or not isinstance(calls[0], dict)
        or calls[0].get("attempt") != "catalog_ranking"
        or not isinstance(proposals, list)
        or len(proposals) != 1
        or not isinstance(proposals[0], dict)
    ):
        raise ValidChildReceiptError("catalog provider call ledger is invalid")
    winning = proposals[0]
    if (
        winning.get("request_id") != validated_request.request_id
        or winning.get("generation") != validated_request.generation
        or winning.get("child_index") != validated_request.child_index
        or winning.get("parent_program_sha256") != parent_hash
        or winning.get("focus") != mutation_focus(validated_request)
        or winning.get("child_dsl") != canonical_dsl
        or winning.get("rendered_source") != child_source
        or winning.get("rendered_sha256") != child_hash
        or winning.get("parent_to_child_diff") != diff
        or not isinstance(winning.get("selected_choice_id"), str)
        or not isinstance(winning.get("exact_patch"), dict)
    ):
        raise ValidChildReceiptError("winning catalog proposal identity mismatch")
    changed_outside_focus = [
        field for field in child_spec.__dataclass_fields__
        if field not in focus_fields
        and getattr(parent_spec, field) != getattr(child_spec, field)
    ]
    if changed_outside_focus:
        raise ValidChildReceiptError("catalog patch changed a non-focus field")
    if (
        target.get("worker_version") != TARGET_WORKER_VERSION
        or target.get("network_attempts") != 0
        or target.get("run_manifest_sha256") != run_manifest_sha256
        or target.get("child_reaped") is not True
        or target.get("gpu_scope") != _GPU0_SCOPE
    ):
        raise ValidChildReceiptError(
            "target worker safety attestation failed"
        )
    target_evidence = target.get("model_evidence")
    if (
        not isinstance(target_evidence, dict)
        or target_evidence.get("model_id") != model_id
        or target_evidence.get("revision")
        != MODEL_METADATA[model_id]["revision"]
        or target_evidence.get("weight_sha256")
        != MODEL_METADATA[model_id]["weight_sha256"]
    ):
        raise ValidChildReceiptError(
            "target model provenance mismatch"
        )
    expected_target_source = [
        {
            "candidate_id": child.get("candidate_id"),
            "model_id": model_id,
            "rendered_sha256": child_hash,
            "canonical_dsl_sha256": hashlib.sha256(
                canonical_dsl.encode("utf-8")
            ).hexdigest(),
        }
    ]
    if target.get("candidate_sources") != expected_target_source:
        raise ValidChildReceiptError(
            "target worker did not use the receipt child source identity"
        )
    scores = target.get("scores")
    if not isinstance(scores, list) or len(scores) != 1:
        raise ValidChildReceiptError(
            "target single score cardinality mismatch"
        )
    score = scores[0]
    if (
        not isinstance(score, dict)
        or score.get("candidate_id")
        != child.get("candidate_id")
        or score.get("sample_count") != 1
        or score.get("error_count") != 0
    ):
        raise ValidChildReceiptError(
            "restricted target single did not succeed"
        )
    return {
        "candidate_id": child.get("candidate_id"),
        "parent_id": child.get("parent_id"),
        "request_id": child.get("request_id"),
        "parent_program_sha256": parent_hash,
        "run_manifest_sha256": run_manifest_sha256,
        "mutation_focus": mutation_focus(validated_request),
        "mutation_focus_fields": list(focus_fields),
        "focus_field_differences": focus_differences,
        "child_program_sha256": child_hash,
        "parent_child_diff_sha256": hashlib.sha256(
            diff.encode("utf-8")
        ).hexdigest(),
        "canonical_dsl_sha256": hashlib.sha256(
            canonical_dsl.encode("utf-8")
        ).hexdigest(),
        "static_import_receipt": dict(static_receipt),
        "single": {
            "sample_count": score["sample_count"],
            "correct_count": score["correct_count"],
            "accuracy": score["accuracy"],
            "error_count": score["error_count"],
        },
        "semantic_retry_calls": 0,
        "winning_attempt": "catalog_ranking",
        "provider_call_count": len(calls),
        "selected_choice_id": winning["selected_choice_id"],
    }


def create_valid_child_receipt(
    *,
    model_id: str,
    campaign_version: str,
    campaign_id: str,
    child: Mapping[str, object],
    provider_artifact: Path,
    target_artifact: Path,
    smoke_summary_artifact: Path,
    run_manifest_artifact: Path,
    run_manifest_sha256: str,
) -> dict[str, object]:
    if model_id not in MODEL_IDS:
        raise ValidChildReceiptError("receipt model is unsupported")
    provider = _read_json(provider_artifact)
    target = _read_json(target_artifact)
    summary = _read_json(smoke_summary_artifact)
    manifest, manifest_digest = verify_run_manifest(
        run_manifest_artifact,
        expected_sha256=run_manifest_sha256,
    )
    source = current_source_identity()
    if (
        manifest_digest != run_manifest_sha256
        or manifest.get("source") != {
            "commit": source["commit"],
            "tree": source["tree"],
        }
        or manifest.get("campaign_id") != campaign_id
        or summary.get("run_manifest_sha256") != run_manifest_sha256
        or summary.get("campaign_id") != campaign_id
    ):
        raise ValidChildReceiptError(
            "smoke run manifest chain is inconsistent"
        )
    validation = _validated_smoke(
        model_id=model_id,
        child=child,
        provider=provider,
        target=target,
        run_manifest_sha256=run_manifest_sha256,
    )
    document = {
        "receipt_version": RECEIPT_VERSION,
        "receipt_id": hashlib.sha256(
            (
                f"{campaign_id}\n{run_manifest_sha256}\n"
                f"{validation['child_program_sha256']}"
            ).encode()
        ).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "run_manifest_sha256": run_manifest_sha256,
        "model": {
            "model_id": model_id,
            "revision": MODEL_METADATA[model_id]["revision"],
            "weight_sha256": MODEL_METADATA[model_id]["weight_sha256"],
        },
        "mutation_model": {
            "model_id": provider["model_evidence"]["model_id"],
            "revision": provider["model_evidence"]["revision"],
            "weight_sha256": provider["model_evidence"]["weight_sha256"],
        },
        "versions": _versions(campaign_version),
        "campaign_id": campaign_id,
        "provider_name": PROVIDER_NAME,
        "gpu_scope": _GPU0_SCOPE,
        "network_attempts": {
            "provider": provider["network_attempts"],
            "target": target["network_attempts"],
        },
        "child_validation": validation,
        "artifacts": {
            "run_manifest": _artifact_meta(run_manifest_artifact),
            "provider": _artifact_meta(provider_artifact),
            "target_single": _artifact_meta(target_artifact),
            "smoke_summary": _artifact_meta(smoke_summary_artifact),
        },
    }
    path = receipt_path(model_id)
    atomic_create_json(path, document)
    return verify_valid_child_receipt(model_id, campaign_version=campaign_version)


def verify_valid_child_receipt(
    model_id: str,
    *,
    campaign_version: str,
) -> dict[str, object]:
    path = receipt_path(model_id)
    receipt = _read_json(path)
    expected_fields = {
        "receipt_version",
        "receipt_id",
        "created_at_utc",
        "source",
        "run_manifest_sha256",
        "model",
        "mutation_model",
        "versions",
        "campaign_id",
        "provider_name",
        "gpu_scope",
        "network_attempts",
        "child_validation",
        "artifacts",
    }
    if set(receipt) != expected_fields:
        raise ValidChildReceiptError("receipt fields are not exact")
    if receipt.get("receipt_version") != RECEIPT_VERSION:
        raise ValidChildReceiptError("receipt version mismatch")
    if receipt.get("source") != current_source_identity():
        raise ValidChildReceiptError("receipt commit/tree does not match current source")
    if receipt.get("versions") != _versions(campaign_version):
        raise ValidChildReceiptError("receipt component versions do not match")
    if receipt.get("provider_name") != PROVIDER_NAME:
        raise ValidChildReceiptError("receipt provider identity mismatch")
    if receipt.get("gpu_scope") != _GPU0_SCOPE:
        raise ValidChildReceiptError("receipt GPU scope mismatch")
    if receipt.get("network_attempts") != {"provider": 0, "target": 0}:
        raise ValidChildReceiptError("receipt network attestation mismatch")
    model = receipt.get("model")
    if model != {
        "model_id": model_id,
        "revision": MODEL_METADATA[model_id]["revision"],
        "weight_sha256": MODEL_METADATA[model_id]["weight_sha256"],
    }:
        raise ValidChildReceiptError("receipt target model provenance mismatch")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "run_manifest",
        "provider",
        "target_single",
        "smoke_summary",
    }:
        raise ValidChildReceiptError("receipt artifact map is invalid")
    loaded: dict[str, dict[str, object]] = {}
    for role, metadata in artifacts.items():
        if not isinstance(metadata, dict):
            raise ValidChildReceiptError("receipt artifact metadata is invalid")
        relative = metadata.get("path")
        if not isinstance(relative, str):
            raise ValidChildReceiptError("receipt artifact path is invalid")
        artifact_path = EVO_ROOT / relative
        if _artifact_meta(artifact_path) != metadata:
            raise ValidChildReceiptError(f"receipt artifact changed: {role}")
        loaded[role] = _read_json(artifact_path)
    summary = loaded["smoke_summary"]
    run_manifest_sha256 = receipt.get("run_manifest_sha256")
    if not isinstance(run_manifest_sha256, str) or len(run_manifest_sha256) != 64:
        raise ValidChildReceiptError("receipt run manifest hash is invalid")
    manifest_meta = artifacts["run_manifest"]
    if not isinstance(manifest_meta, dict) or not isinstance(
        manifest_meta.get("path"), str
    ):
        raise ValidChildReceiptError("receipt run manifest metadata is invalid")
    manifest, actual_manifest_sha = verify_run_manifest(
        EVO_ROOT / manifest_meta["path"],
        expected_sha256=run_manifest_sha256,
    )
    source = receipt["source"]
    if (
        actual_manifest_sha != run_manifest_sha256
        or not isinstance(source, dict)
        or manifest.get("source") != {
            "commit": source.get("commit"),
            "tree": source.get("tree"),
        }
        or manifest.get("campaign_id") != receipt.get("campaign_id")
    ):
        raise ValidChildReceiptError("receipt run manifest chain changed")
    if loaded["smoke_summary"].get("run_manifest_sha256") != run_manifest_sha256:
        raise ValidChildReceiptError("smoke summary run manifest mismatch")
    child = summary.get("child")
    if not isinstance(child, dict):
        raise ValidChildReceiptError("receipt smoke summary child is invalid")
    rederived = _validated_smoke(
        model_id=model_id,
        child=child,
        provider=loaded["provider"],
        target=loaded["target_single"],
        run_manifest_sha256=run_manifest_sha256,
    )
    if receipt.get("child_validation") != rederived:
        raise ValidChildReceiptError("receipt child validation changed")
    expected_id = hashlib.sha256(
        (
            f"{receipt['campaign_id']}\n{run_manifest_sha256}\n"
            f"{rederived['child_program_sha256']}"
        ).encode()
    ).hexdigest()
    if receipt.get("receipt_id") != expected_id:
        raise ValidChildReceiptError("receipt ID mismatch")
    return receipt


__all__ = [
    "RECEIPT_VERSION",
    "ValidChildReceiptError",
    "create_valid_child_receipt",
    "current_source_identity",
    "receipt_path",
    "verify_valid_child_receipt",
]
