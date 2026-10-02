"""Two-generation local-Qwen evolution controller with hard GPU budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .candidate_runtime import CANDIDATE_RUNTIME_VERSION
from .core import CHECKPOINT_ROOT, load_checkpoint, write_checkpoint
from .dsl import (
    DSL_VERSION,
    INSTRUCT_MODEL_ID,
    THINKING_MODEL_ID,
    MODEL_IDS,
    canonical_candidate_source,
    parse_candidate_dsl,
    seed_spec,
)
from .catalog_provider import (
    CatalogConstrainedQwenMutationProvider,
    PROVIDER_VERSION,
)
from .local_provider import (
    MutationOutcome,
    MutationRequest,
    mutation_focus,
)
from .mutation_catalog import CATALOG_VERSION
from .phase2_manifests import (
    DEFAULT_MANIFEST_ROOT,
    STAGE_COUNTS,
    create_manifests,
    load_private_queries,
    validate_disjoint_manifests,
)
from .process_safety import (
    EVO_ROOT,
    PROJECT_ROOT,
    atomic_create_json,
    atomic_create_text,
    ensure_internal_directory,
)
from .renderer import (
    import_check_rendered_program,
    RENDERER_VERSION,
    render_candidate_program,
    rendered_program_diff,
    rendered_program_sha256,
    validate_rendered_program,
)
from .run_manifest import (
    RUN_MANIFEST_VERSION,
    create_run_manifest,
    verify_run_manifest,
)
from .target_worker import (
    CandidateScore,
    PrivateQuery,
    RestrictedTargetEvaluator,
    TARGET_WORKER_VERSION,
    TargetCandidate,
)
from .valid_child_receipt import (
    RECEIPT_VERSION,
    create_valid_child_receipt,
    verify_valid_child_receipt,
)


CAMPAIGN_VERSION = "phase2-catalog-qwen-two-generation-v4"
ARTIFACT_ROOT = PROJECT_ROOT / "results/evolution/local_qwen"
GPU_SECONDS_PER_MODEL = 75 * 60
PROVIDER_SMOKE_SECONDS = 45 * 60
MAX_CHILDREN_PER_MODEL = 6
QUALIFICATION_SCHEDULE: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, 1),
    (2, 2),
)
GENERATION_TWO_JOURNAL_VERSION = "phase2-generation2-journal-v1"
FINALIZATION_PLAN_VERSION = "phase2-generation2-finalization-v1"
MANAGED_IMPORT_CHECK_ROOT = (
    EVO_ROOT / ".tmp/local_evolution/import_check"
)


class LocalCampaignError(RuntimeError):
    pass


def _finalization_crash_hook(stage: str) -> None:
    """CPU-test seam; production never configures a crash stage."""
    del stage


def _load_strict_object(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise LocalCampaignError(f"artifact is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocalCampaignError(f"artifact is not strict JSON: {path}") from exc
    if not isinstance(value, dict):
        raise LocalCampaignError(f"artifact is not an object: {path}")
    return value


def _adopt_json(path: Path, document: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        if _load_strict_object(path) != dict(document):
            raise LocalCampaignError(f"existing artifact differs: {path}")
        return
    atomic_create_json(path, document)


def _adopt_text(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise LocalCampaignError(f"text artifact is not regular: {path}")
        if path.read_text(encoding="utf-8") != text:
            raise LocalCampaignError(f"existing text artifact differs: {path}")
        return
    atomic_create_text(path, text)


def _adopt_checkpoint(relative: Path, state: Mapping[str, object]) -> None:
    path = CHECKPOINT_ROOT / relative
    if path.exists() or path.is_symlink():
        if load_checkpoint(relative) != dict(state):
            raise LocalCampaignError(f"existing checkpoint differs: {relative}")
        return
    write_checkpoint(relative, state)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _managed_import_check_directory() -> Path:
    directory = ensure_internal_directory(MANAGED_IMPORT_CHECK_ROOT)
    if not directory.is_relative_to(EVO_ROOT.resolve(strict=True)):
        raise LocalCampaignError("managed import-check directory escaped runtime")
    return directory


def _safe_identifier(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 160
        or not all(character.isalnum() or character in "-_" for character in value)
    ):
        raise LocalCampaignError(f"{label} is not a safe identifier")
    return value



@dataclass(frozen=True)
class LocalCandidate:
    candidate_id: str
    model_id: str
    generation: int
    parent_id: str | None
    request_id: str
    canonical_dsl: str
    rendered_source: str
    rendered_sha256: str
    unified_diff: str
    repair_used: bool
    static_receipt: Mapping[str, object]
    scores: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        _safe_identifier(self.candidate_id, "candidate_id")
        _safe_identifier(self.request_id, "request_id")
        if self.generation == 0:
            if self.parent_id is not None:
                raise ValueError("seed candidate cannot have a parent")
        else:
            _safe_identifier(self.parent_id, "parent_id")
        if not isinstance(self.repair_used, bool):
            raise ValueError("candidate repair_used must be boolean")
        if not isinstance(self.static_receipt, Mapping):
            raise ValueError("candidate static receipt must be a mapping")
        if not isinstance(self.scores, Mapping):
            raise ValueError("candidate scores must be a mapping")
        for stage, score in self.scores.items():
            if stage not in {"single", "search12", "selection32"}:
                raise ValueError("candidate score stage is invalid")
            if not isinstance(score, Mapping) or score.get("stage") != stage:
                raise ValueError("candidate score payload is invalid")
        if self.model_id not in MODEL_IDS:
            raise ValueError("local candidate model is unsupported")
        if self.generation not in {0, 1, 2}:
            raise ValueError("local candidate generation is invalid")
        spec = parse_candidate_dsl(
            self.canonical_dsl,
            expected_model_id=self.model_id,
        )
        if canonical_candidate_source(spec) != self.canonical_dsl:
            raise ValueError("local candidate DSL is not canonical")
        rendered_spec = validate_rendered_program(
            self.rendered_source,
            expected_model_id=self.model_id,
        )
        if rendered_spec != spec:
            raise ValueError("local candidate DSL/rendered source mismatch")
        if rendered_program_sha256(self.rendered_source) != self.rendered_sha256:
            raise ValueError("local candidate rendered hash mismatch")

    def to_mapping(self) -> dict[str, object]:
        value = asdict(self)
        value["static_receipt"] = dict(self.static_receipt)
        value["scores"] = {
            stage: dict(score) for stage, score in self.scores.items()
        }
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "LocalCandidate":
        if set(value) != set(cls.__dataclass_fields__):
            raise LocalCampaignError("checkpoint candidate fields are not exact")
        return cls(**dict(value))


@dataclass(frozen=True)
class CampaignState:
    campaign_id: str
    model_id: str
    generation: int
    gpu_elapsed_seconds: float
    candidates: tuple[LocalCandidate, ...]
    failures: tuple[Mapping[str, object], ...]
    campaign_directory: str
    execution_class: str
    run_manifest_path: str
    run_manifest_sha256: str

    def __post_init__(self) -> None:
        _safe_identifier(self.campaign_id, "campaign_id")
        if self.model_id not in MODEL_IDS:
            raise LocalCampaignError("campaign state model is invalid")
        if self.generation not in {1, 2}:
            raise LocalCampaignError("campaign state generation is invalid")
        if (
            isinstance(self.gpu_elapsed_seconds, bool)
            or not isinstance(self.gpu_elapsed_seconds, (int, float))
            or not math.isfinite(float(self.gpu_elapsed_seconds))
            or float(self.gpu_elapsed_seconds) < 0
        ):
            raise LocalCampaignError("campaign GPU elapsed time is invalid")
        if not isinstance(self.candidates, tuple) or not isinstance(self.failures, tuple):
            raise LocalCampaignError("campaign state collections must be tuples")
        if any(not isinstance(item, Mapping) for item in self.failures):
            raise LocalCampaignError("campaign failure entry is invalid")
        identifiers = [candidate.candidate_id for candidate in self.candidates]
        if len(identifiers) != len(set(identifiers)):
            raise LocalCampaignError("campaign state contains duplicate candidates")
        if any(candidate.model_id != self.model_id for candidate in self.candidates):
            raise LocalCampaignError("campaign state crosses a model population")
        if any(candidate.generation > self.generation for candidate in self.candidates):
            raise LocalCampaignError("campaign state contains a future candidate")
        directory = Path(self.campaign_directory)
        if not directory.is_absolute() or not directory.is_relative_to(ARTIFACT_ROOT):
            raise LocalCampaignError("campaign directory is outside artifact root")
        if self.execution_class not in {
            "production_evolution",
            "synthetic_test",
        }:
            raise LocalCampaignError("campaign execution class is invalid")
        if (
            not isinstance(self.run_manifest_sha256, str)
            or len(self.run_manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.run_manifest_sha256)
        ):
            raise LocalCampaignError("campaign run manifest hash is invalid")
        if self.execution_class == "synthetic_test":
            if self.run_manifest_path != "SYNTHETIC_TEST":
                raise LocalCampaignError("synthetic state has production manifest path")
        else:
            manifest_path = Path(self.run_manifest_path)
            if not manifest_path.is_absolute() or not manifest_path.is_relative_to(ARTIFACT_ROOT):
                raise LocalCampaignError("campaign run manifest path is unsafe")



    def to_mapping(self) -> dict[str, object]:
        return {
            "campaign_version": CAMPAIGN_VERSION,
            "campaign_id": self.campaign_id,
            "model_id": self.model_id,
            "generation": self.generation,
            "gpu_elapsed_seconds": self.gpu_elapsed_seconds,
            "candidates": [candidate.to_mapping() for candidate in self.candidates],
            "failures": [dict(item) for item in self.failures],
            "campaign_directory": self.campaign_directory,
            "execution_class": self.execution_class,
            "run_manifest_path": self.run_manifest_path,
            "run_manifest_sha256": self.run_manifest_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CampaignState":
        expected = {
            "campaign_version",
            "campaign_id",
            "model_id",
            "generation",
            "gpu_elapsed_seconds",
            "candidates",
            "failures",
            "campaign_directory",
            "execution_class",
            "run_manifest_path",
            "run_manifest_sha256",
        }
        if set(value) != expected or value.get("campaign_version") != CAMPAIGN_VERSION:
            raise LocalCampaignError("campaign checkpoint contract mismatch")
        raw_candidates = value.get("candidates")
        raw_failures = value.get("failures")
        if not isinstance(raw_candidates, list) or not isinstance(raw_failures, list):
            raise LocalCampaignError("campaign checkpoint collections are invalid")
        if any(not isinstance(item, dict) for item in raw_candidates):
            raise LocalCampaignError("campaign checkpoint candidate is not an object")
        if any(not isinstance(item, dict) for item in raw_failures):
            raise LocalCampaignError("campaign checkpoint failure is not an object")
        if (
            not isinstance(value["campaign_id"], str)
            or not isinstance(value["model_id"], str)
            or isinstance(value["generation"], bool)
            or not isinstance(value["generation"], int)
            or isinstance(value["gpu_elapsed_seconds"], bool)
            or not isinstance(value["gpu_elapsed_seconds"], (int, float))
            or not isinstance(value["campaign_directory"], str)
            or not isinstance(value["execution_class"], str)
            or not isinstance(value["run_manifest_path"], str)
            or not isinstance(value["run_manifest_sha256"], str)
        ):
            raise LocalCampaignError("campaign checkpoint scalar types are invalid")
        return cls(
            campaign_id=value["campaign_id"],
            model_id=value["model_id"],
            generation=value["generation"],
            gpu_elapsed_seconds=float(value["gpu_elapsed_seconds"]),
            candidates=tuple(
                LocalCandidate.from_mapping(item) for item in raw_candidates
            ),
            failures=tuple(dict(item) for item in raw_failures),
            campaign_directory=value["campaign_directory"],
            execution_class=value["execution_class"],
            run_manifest_path=value["run_manifest_path"],
            run_manifest_sha256=value["run_manifest_sha256"],
        )


def _slug(model_id: str) -> str:
    if model_id == INSTRUCT_MODEL_ID:
        return "instruct"
    if model_id == THINKING_MODEL_ID:
        return "thinking"
    raise LocalCampaignError("unsupported campaign model")


def _campaign_id(model_id: str, kind: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{kind}-{_slug(model_id)}-{timestamp}-{uuid.uuid4().hex}"


def _public_score(score: CandidateScore, stage: str) -> dict[str, object]:
    records = [dict(item) for item in score.private_records]
    preprocessing_total = sum(
        float(item["preprocessing_seconds"]) for item in records
    )
    generation_total = sum(
        float(item["generation_seconds"]) for item in records
    )
    peak_memory = max(
        (int(item["peak_memory_bytes"]) for item in records),
        default=0,
    )
    return {
        "stage": stage,
        "samples": score.sample_count,
        "correct": score.correct_count,
        "accuracy": score.accuracy,
        "errors": score.error_count,
        "total_seconds": score.total_seconds,
        "mean_latency_seconds": score.mean_latency_seconds,
        "total_preprocessing_seconds": preprocessing_total,
        "mean_preprocessing_seconds": preprocessing_total / score.sample_count,
        "total_generation_seconds": generation_total,
        "mean_generation_seconds": generation_total / score.sample_count,
        "peak_memory_bytes": peak_memory,
        "total_generated_tokens": score.total_generated_tokens,
        "termination_counts": dict(score.termination_counts),
        "timing_load_included": False,
        "fixed_warmup_included_in_score": False,
    }


def _score_key(candidate: LocalCandidate, stage: str) -> tuple[float, int, float, str]:
    score = candidate.scores.get(stage)
    if not isinstance(score, Mapping):
        raise LocalCampaignError(f"candidate lacks {stage} score")
    return (
        -float(score["accuracy"]),
        int(score["errors"]),
        float(score["mean_latency_seconds"]),
        candidate.candidate_id,
    )


def _dominates(left: LocalCandidate, right: LocalCandidate, stage: str) -> bool:
    a = left.scores[stage]
    b = right.scores[stage]
    no_worse = (
        float(a["accuracy"]) >= float(b["accuracy"])
        and int(a["errors"]) <= int(b["errors"])
        and float(a["mean_latency_seconds"]) <= float(b["mean_latency_seconds"])
    )
    strict = (
        float(a["accuracy"]) > float(b["accuracy"])
        or int(a["errors"]) < int(b["errors"])
        or float(a["mean_latency_seconds"]) < float(b["mean_latency_seconds"])
    )
    return no_worse and strict


def pareto_order(
    candidates: Sequence[LocalCandidate],
    *,
    stage: str,
) -> list[LocalCandidate]:
    values = list(candidates)
    front = [
        candidate
        for candidate in values
        if not any(
            other.candidate_id != candidate.candidate_id
            and _dominates(other, candidate, stage)
            for other in values
        )
    ]
    front_ids = {candidate.candidate_id for candidate in front}
    remainder = [
        candidate for candidate in values if candidate.candidate_id not in front_ids
    ]
    return sorted(front, key=lambda item: _score_key(item, stage)) + sorted(
        remainder,
        key=lambda item: _score_key(item, stage),
    )


class LocalEvolutionCampaign:
    def __init__(
        self,
        *,
        model_id: str,
        provider: object | None = None,
        evaluator: object | None = None,
        query_loader: Callable[[str], list[PrivateQuery]] | None = None,
        manifest_root: Path = DEFAULT_MANIFEST_ROOT,
        gpu_budget_seconds: float = GPU_SECONDS_PER_MODEL,
        execution_class: str = "production_evolution",
    ) -> None:
        if model_id not in MODEL_IDS:
            raise LocalCampaignError("campaign model is unsupported")
        if not math.isfinite(gpu_budget_seconds) or gpu_budget_seconds <= 0:
            raise ValueError("GPU budget must be finite and positive")
        if execution_class not in {"production_evolution", "synthetic_test"}:
            raise LocalCampaignError("campaign execution class is invalid")
        if execution_class == "production_evolution":
            if provider is None:
                provider = CatalogConstrainedQwenMutationProvider()
            if evaluator is None:
                evaluator = RestrictedTargetEvaluator()
            if type(provider) is not CatalogConstrainedQwenMutationProvider:
                raise LocalCampaignError(
                    "production campaign requires the locked catalog provider"
                )
            if type(evaluator) is not RestrictedTargetEvaluator:
                raise LocalCampaignError(
                    "production campaign requires the restricted target evaluator"
                )
        else:
            if provider is None or evaluator is None:
                raise LocalCampaignError(
                    "synthetic campaign requires explicit fake components"
                )
            if isinstance(
                provider, CatalogConstrainedQwenMutationProvider
            ) or isinstance(
                evaluator, RestrictedTargetEvaluator
            ):
                raise LocalCampaignError(
                    "synthetic campaign cannot use a production component"
                )
            if getattr(provider, "evidence_class", None) != "synthetic_test":
                raise LocalCampaignError("synthetic provider is not test evidence")
            if getattr(evaluator, "evidence_class", None) != "synthetic_test":
                raise LocalCampaignError("synthetic evaluator is not test evidence")
        self.model_id = model_id
        self.provider = provider
        self.evaluator = evaluator
        self.manifest_root = manifest_root
        self.query_loader = query_loader or (
            lambda stage: load_private_queries(stage, root=self.manifest_root)
        )
        self.gpu_budget_seconds = float(gpu_budget_seconds)
        self.gpu_elapsed_seconds = 0.0
        self.campaign_id = ""
        self.campaign_directory = Path()
        self.failures: list[dict[str, object]] = []
        self.execution_class = execution_class
        self.run_manifest_path = "SYNTHETIC_TEST"
        self.run_manifest_sha256 = "0" * 64
        self._run_manifest: Mapping[str, object] | None = None

    @property
    def checkpoint_relative(self) -> Path:
        if not self.campaign_id:
            raise LocalCampaignError("campaign has no ID")
        return Path("phase2_local") / self.campaign_id / "generation1.json"

    @property
    def active_relative(self) -> Path:
        return Path("phase2_local") / _slug(self.model_id) / "active.json"
    @property
    def completion_relative(self) -> Path:
        return Path("phase2_local") / _slug(self.model_id) / "completed.json"

    @property
    def generation_two_journal_path(self) -> Path:
        if not self.campaign_id:
            raise LocalCampaignError("campaign has no ID")
        return self.campaign_directory / "generation2_gpu_journal.json"

    @property
    def finalization_plan_path(self) -> Path:
        if not self.campaign_id:
            raise LocalCampaignError("campaign has no ID")
        return self.campaign_directory / "generation2_finalization_plan.json"

    def _versions(self) -> dict[str, str]:
        return {
            "campaign": CAMPAIGN_VERSION,
            "candidate_runtime": CANDIDATE_RUNTIME_VERSION,
            "catalog": CATALOG_VERSION,
            "dsl": DSL_VERSION,
            "provider": PROVIDER_VERSION,
            "renderer": RENDERER_VERSION,
            "run_manifest": RUN_MANIFEST_VERSION,
            "target_worker": TARGET_WORKER_VERSION,
            "valid_child_receipt": RECEIPT_VERSION,
        }

    def _create_run_manifest(self, *, kind: str) -> None:
        if self.execution_class == "synthetic_test":
            self.run_manifest_path = "SYNTHETIC_TEST"
            self.run_manifest_sha256 = "0" * 64
            self._run_manifest = None
            return
        if not self.campaign_id or not self.campaign_directory.is_dir():
            raise LocalCampaignError("campaign directory must precede run manifest")
        path = self.campaign_directory / "run_manifest.json"
        document, digest = create_run_manifest(
            path=path,
            campaign_id=self.campaign_id,
            kind=kind,
            model_id=self.model_id,
            manifest_root=self.manifest_root,
            versions=self._versions(),
            output_allowlist=[
                self.campaign_directory.relative_to(PROJECT_ROOT).as_posix(),
                "results/evolution/valid_child_receipts",
                "results/evolution/checkpoints/phase2_local",
                f"evolved_{_slug(self.model_id)}.py",
            ],
        )
        self.run_manifest_path = str(path)
        self.run_manifest_sha256 = digest
        self._run_manifest = document

    def _verify_run_manifest(
        self,
        *,
        allowed_generated_source: Mapping[str, str] | None = None,
    ) -> None:
        if self.execution_class == "synthetic_test":
            if (
                self.run_manifest_path != "SYNTHETIC_TEST"
                or self.run_manifest_sha256 != "0" * 64
            ):
                raise LocalCampaignError("synthetic run manifest binding is invalid")
            return
        document, digest = verify_run_manifest(
            Path(self.run_manifest_path),
            expected_sha256=self.run_manifest_sha256,
            allowed_generated_source=allowed_generated_source,
        )
        if document.get("campaign_id") != self.campaign_id:
            raise LocalCampaignError("run manifest campaign identity mismatch")
        if digest != self.run_manifest_sha256:
            raise LocalCampaignError("run manifest digest changed")
    def _finalize_generation_two(
        self,
        plan: Mapping[str, object],
    ) -> CampaignState:
        expected = {
            "finalization_version",
            "campaign_id",
            "model_id",
            "run_manifest_sha256",
            "generation2_checkpoint",
            "final_state",
            "export_name",
            "export_source",
            "export_sha256",
            "summary",
        }
        if set(plan) != expected:
            raise LocalCampaignError("generation-two finalization plan fields are not exact")
        if (
            plan["finalization_version"] != FINALIZATION_PLAN_VERSION
            or plan["campaign_id"] != self.campaign_id
            or plan["model_id"] != self.model_id
            or plan["run_manifest_sha256"] != self.run_manifest_sha256
        ):
            raise LocalCampaignError("generation-two finalization plan identity mismatch")
        final_value = plan["final_state"]
        summary = plan["summary"]
        if not isinstance(final_value, dict) or not isinstance(summary, dict):
            raise LocalCampaignError("generation-two finalization payload is invalid")
        final = CampaignState.from_mapping(final_value)
        if (
            final.generation != 2
            or final.campaign_id != self.campaign_id
            or final.model_id != self.model_id
            or final.run_manifest_sha256 != self.run_manifest_sha256
        ):
            raise LocalCampaignError("generation-two final state identity mismatch")
        export_name = plan["export_name"]
        export_source = plan["export_source"]
        export_sha256 = plan["export_sha256"]
        checkpoint_value = plan["generation2_checkpoint"]
        if (
            not isinstance(export_name, str)
            or export_name != f"evolved_{_slug(self.model_id)}.py"
            or not isinstance(export_source, str)
            or not isinstance(export_sha256, str)
            or hashlib.sha256(export_source.encode("utf-8")).hexdigest()
            != export_sha256
            or not isinstance(checkpoint_value, str)
        ):
            raise LocalCampaignError("generation-two export/checkpoint plan is invalid")
        best_id = summary.get("best_candidate_id")
        matching = [
            candidate
            for candidate in final.candidates
            if candidate.candidate_id == best_id
        ]
        if (
            len(matching) != 1
            or matching[0].generation == 0
            or matching[0].rendered_source != export_source
            or matching[0].rendered_sha256 != export_sha256
            or summary.get("best_program_sha256") != export_sha256
            or summary.get("export_path") != export_name
        ):
            raise LocalCampaignError("generation-two selected source identity mismatch")

        export_path = PROJECT_ROOT / export_name
        _adopt_text(export_path, export_source)
        _finalization_crash_hook("export")

        checkpoint_relative = Path(checkpoint_value)
        _adopt_checkpoint(checkpoint_relative, final.to_mapping())
        _finalization_crash_hook("checkpoint")

        summary_path = self.campaign_directory / "final_summary.json"
        _adopt_json(summary_path, summary)
        _finalization_crash_hook("summary")

        completion = {
            "completion_version": "phase2-generation-completion-v1",
            "campaign_version": CAMPAIGN_VERSION,
            "status": "completed",
            "model_id": self.model_id,
            "campaign_id": self.campaign_id,
            "generation2_checkpoint": checkpoint_relative.as_posix(),
            "final_summary": {
                "path": summary_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _sha256_path(summary_path),
            },
            "export": {
                "path": export_name,
                "sha256": _sha256_path(export_path),
            },
        }
        _adopt_json(CHECKPOINT_ROOT / self.completion_relative, completion)
        return final

    def _recover_generation_two_finalization(self) -> CampaignState | None:
        plan_path = self.finalization_plan_path
        if plan_path.exists() or plan_path.is_symlink():
            plan = _load_strict_object(plan_path)
            return self._finalize_generation_two(plan)
        journal_path = self.generation_two_journal_path
        if journal_path.exists() or journal_path.is_symlink():
            journal = _load_strict_object(journal_path)
            if (
                journal.get("journal_version") != GENERATION_TWO_JOURNAL_VERSION
                or journal.get("campaign_id") != self.campaign_id
                or journal.get("model_id") != self.model_id
                or journal.get("run_manifest_sha256")
                != self.run_manifest_sha256
            ):
                raise LocalCampaignError("generation-two journal identity mismatch")
            raise LocalCampaignError(
                "generation-two GPU work is incomplete or ambiguous; refusing replay"
            )
        return None

    def _require_receipt(self) -> Mapping[str, object]:
        if self.execution_class == "synthetic_test":
            return {
                "execution_class": "synthetic_test",
                "model_id": self.model_id,
                "campaign_version": CAMPAIGN_VERSION,
                "receipt_id": "SYNTHETIC_TEST",
            }
        return verify_valid_child_receipt(
            self.model_id,
            campaign_version=CAMPAIGN_VERSION,
        )


    def _ensure_manifests(self) -> None:
        paths = [self.manifest_root / f"{stage}.json" for stage in STAGE_COUNTS]
        present = [path.is_file() and not path.is_symlink() for path in paths]
        if any(present) and not all(present):
            raise LocalCampaignError("Phase 2 manifests are partially present")
        if not any(present):
            create_manifests(self.manifest_root)
        validate_disjoint_manifests(self.manifest_root)

    def _remaining(self) -> float:
        remaining = self.gpu_budget_seconds - self.gpu_elapsed_seconds
        if remaining <= 0:
            raise LocalCampaignError("per-model GPU wall budget exhausted")
        return remaining

    def _consume(self, component: object) -> None:
        elapsed = getattr(component, "last_child_elapsed_seconds", None)
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
            raise LocalCampaignError("GPU component did not report child elapsed time")
        elapsed = float(elapsed)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise LocalCampaignError("GPU child elapsed time is invalid")
        self.gpu_elapsed_seconds += elapsed
        if self.gpu_elapsed_seconds > self.gpu_budget_seconds:
            raise LocalCampaignError("per-model GPU wall budget exceeded")

    def _seed(self) -> LocalCandidate:
        spec = seed_spec(self.model_id)
        source = render_candidate_program(spec)
        return LocalCandidate(
            candidate_id=f"seed-{_slug(self.model_id)}",
            model_id=self.model_id,
            generation=0,
            parent_id=None,
            request_id="manual_seed",
            canonical_dsl=canonical_candidate_source(spec),
            rendered_source=source,
            rendered_sha256=rendered_program_sha256(source),
            unified_diff="",
            repair_used=False,
            static_receipt={
                "source": f"manual_{_slug(self.model_id)}.py",
                "manual_seed_semantics": True,
                "interface": "vlm_inference(image_path, question)",
            },
            scores={},
        )

    def _requests(
        self,
        *,
        parents: Sequence[LocalCandidate],
        generation: int,
        children_per_parent: int,
        population: Sequence[LocalCandidate] | None = None,
        child_index_offset: int = 0,
    ) -> list[MutationRequest]:
        if (
            isinstance(child_index_offset, bool)
            or not isinstance(child_index_offset, int)
            or child_index_offset < 0
        ):
            raise LocalCampaignError("child index offset is invalid")
        requests: list[MutationRequest] = []
        population_hashes = tuple(
            sorted(
                candidate.rendered_sha256
                for candidate in (population if population is not None else parents)
            )
        )
        index = child_index_offset
        for parent in parents:
            metrics = (
                dict(parent.scores["search12"])
                if "search12" in parent.scores
                else dict(parent.scores["single"])
                if "single" in parent.scores
                else {
                    "stage": "seed",
                    "samples": 0,
                    "accuracy": 0.0,
                    "errors": 0,
                    "mean_latency_seconds": 0.0,
                }
            )
            allowed_metrics = {
                key: metrics[key]
                for key in (
                    "stage",
                    "samples",
                    "accuracy",
                    "errors",
                    "mean_latency_seconds",
                )
            }
            for _ in range(children_per_parent):
                requests.append(
                    MutationRequest(
                        request_id=f"g{generation}_child{index}",
                        generation=generation,
                        child_index=index,
                        parent_id=parent.candidate_id,
                        model_id=self.model_id,
                        parent_dsl=parent.canonical_dsl,
                        parent_program_source=parent.rendered_source,
                        parent_program_sha256=parent.rendered_sha256,
                        parent_metrics=allowed_metrics,
                        run_manifest_sha256=self.run_manifest_sha256,
                        execution_class=self.execution_class,
                        population_program_sha256s=population_hashes,
                    )
                )
                index += 1
        return requests

    def _build_children(
        self,
        *,
        parents: Mapping[str, LocalCandidate],
        requests: Sequence[MutationRequest],
        outcomes: Sequence[MutationOutcome],
        existing_hashes: set[str],
    ) -> list[LocalCandidate]:
        if len(requests) != len(outcomes):
            raise LocalCampaignError("mutation request/outcome cardinality mismatch")
        children: list[LocalCandidate] = []
        temporary = _managed_import_check_directory()
        for request, outcome in zip(requests, outcomes, strict=True):
            if outcome.status != "accepted":
                self.failures.append(
                    {
                        "stage": f"generation{request.generation}_mutation",
                        "request_id": request.request_id,
                        "parent_id": request.parent_id,
                        "error_type": outcome.error_type,
                        "error_message": outcome.error_message,
                    }
                )
                continue
            assert outcome.canonical_dsl is not None
            assert outcome.rendered_source is not None
            assert outcome.rendered_sha256 is not None
            if outcome.rendered_sha256 in existing_hashes:
                self.failures.append(
                    {
                        "stage": f"generation{request.generation}_static",
                        "request_id": request.request_id,
                        "parent_id": request.parent_id,
                        "error_type": "duplicate_program",
                        "error_message": "rendered program duplicates an existing candidate",
                    }
                )
                continue
            parent = parents[request.parent_id]
            try:
                spec = parse_candidate_dsl(
                    outcome.canonical_dsl,
                    expected_model_id=self.model_id,
                )
                if render_candidate_program(spec) != outcome.rendered_source:
                    raise LocalCampaignError("renderer output is not byte-exact")
                receipt = import_check_rendered_program(
                    outcome.rendered_source,
                    temporary_directory=temporary,
                )
            except Exception as exc:
                self.failures.append(
                    {
                        "stage": f"generation{request.generation}_static",
                        "request_id": request.request_id,
                        "parent_id": request.parent_id,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:1024],
                    }
                )
                continue
            material = (
                f"{self.model_id}\n{parent.candidate_id}\n{request.generation}\n"
                f"{request.request_id}\n{outcome.rendered_sha256}"
            )
            candidate_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
            child = LocalCandidate(
                candidate_id=candidate_id,
                model_id=self.model_id,
                generation=request.generation,
                parent_id=parent.candidate_id,
                request_id=request.request_id,
                canonical_dsl=outcome.canonical_dsl,
                rendered_source=outcome.rendered_source,
                rendered_sha256=outcome.rendered_sha256,
                unified_diff=rendered_program_diff(
                    parent.rendered_source,
                    outcome.rendered_source,
                ),
                repair_used=outcome.repair_used,
                static_receipt=receipt,
                scores={},
            )
            children.append(child)
            existing_hashes.add(child.rendered_sha256)
        return children

    def _evaluate(
        self,
        candidates: Sequence[LocalCandidate],
        *,
        stage: str,
        artifact_name: str,
    ) -> list[LocalCandidate]:
        if not candidates:
            return []
        self._verify_run_manifest()
        queries = self.query_loader(stage)
        warmup = self.query_loader("warmup")
        if len(warmup) != 1:
            raise LocalCampaignError("fixed warmup manifest must contain one query")

        target_candidates = [
            TargetCandidate(
                candidate.candidate_id,
                candidate.model_id,
                candidate.canonical_dsl,
                candidate.rendered_source,
                candidate.rendered_sha256,
            )
            for candidate in candidates
        ]
        scores = self.evaluator.evaluate(
            model_id=self.model_id,
            candidates=target_candidates,
            queries=queries,
            warmup_query=warmup[0],
            artifact_path=self.campaign_directory / artifact_name,
            run_manifest_sha256=self.run_manifest_sha256,
            timeout_seconds=min(self._remaining(), 4500.0),
        )
        self._consume(self.evaluator)
        if [score.candidate_id for score in scores] != [
            candidate.candidate_id for candidate in candidates
        ]:
            raise LocalCampaignError("target score order differs from candidates")
        return [
            replace(
                candidate,
                scores={
                    **dict(candidate.scores),
                    stage: _public_score(score, stage),
                },
            )
            for candidate, score in zip(candidates, scores, strict=True)
        ]

    def _mutate(
        self,
        requests: Sequence[MutationRequest],
        *,
        artifact_name: str,
    ) -> list[MutationOutcome]:
        self._verify_run_manifest()
        outcomes = self.provider.mutate_generation(
            requests,
            artifact_path=self.campaign_directory / artifact_name,
            timeout_seconds=min(self._remaining(), 4500.0),
        )
        self._consume(self.provider)
        return list(outcomes)

    def _relocated_completed_state(
        self,
        value: object,
        *,
        campaign_id: str,
        campaign_directory: Path,
    ) -> CampaignState:
        """Validate a sealed state while relocating its two machine-local paths."""

        if not isinstance(value, dict):
            raise LocalCampaignError("completed state is not an object")
        recorded_directory_value = value.get("campaign_directory")
        recorded_manifest_value = value.get("run_manifest_path")
        if not isinstance(recorded_directory_value, str) or not isinstance(
            recorded_manifest_value, str
        ):
            raise LocalCampaignError("completed state path metadata is invalid")
        recorded_directory = Path(recorded_directory_value)
        recorded_manifest = Path(recorded_manifest_value)
        if (
            not recorded_directory.is_absolute()
            or recorded_directory.name != campaign_id
            or ".." in recorded_directory.parts
            or not recorded_manifest.is_absolute()
            or recorded_manifest != recorded_directory / "run_manifest.json"
        ):
            raise LocalCampaignError("completed state path provenance is invalid")
        relocated = dict(value)
        relocated["campaign_directory"] = str(campaign_directory)
        relocated["run_manifest_path"] = str(
            campaign_directory / "run_manifest.json"
        )
        return CampaignState.from_mapping(relocated)

    def _verify_sealed_completed_run_manifest(
        self,
        *,
        campaign_directory: Path,
        expected_export: str,
    ) -> None:
        """Verify archived manifest identity without replaying its old machine closure."""

        manifest_path = campaign_directory / "run_manifest.json"
        document = _load_strict_object(manifest_path)
        expected_fields = {
            "run_manifest_version",
            "execution_class",
            "campaign_id",
            "kind",
            "source",
            "versions",
            "source_closure",
            "models",
            "phase2_manifests",
            "mutation_catalog",
            "gpu_scope",
            "network_policy",
            "output_allowlist",
        }
        if set(document) != expected_fields:
            raise LocalCampaignError("sealed run manifest fields are not exact")
        if (
            document["run_manifest_version"] != RUN_MANIFEST_VERSION
            or document["execution_class"] != "production_evolution"
            or document["campaign_id"] != self.campaign_id
            or document["kind"] != "formal_evolution"
            or _sha256_path(manifest_path) != self.run_manifest_sha256
        ):
            raise LocalCampaignError("sealed run manifest identity mismatch")
        allowlist = document["output_allowlist"]
        if (
            not isinstance(allowlist, list)
            or allowlist != sorted(set(allowlist))
            or expected_export not in allowlist
            or "results/evolution/checkpoints/phase2_local" not in allowlist
        ):
            raise LocalCampaignError("sealed run manifest allowlist mismatch")
        models = document["models"]
        if (
            not isinstance(models, list)
            or not models
            or any(not isinstance(item, dict) for item in models)
            or not any(item.get("model_id") == self.model_id for item in models)
        ):
            raise LocalCampaignError("sealed run manifest model closure mismatch")

    def _load_completed(self) -> CampaignState | None:
        marker_path = CHECKPOINT_ROOT / self.completion_relative
        if not marker_path.exists() and not marker_path.is_symlink():
            return None
        if marker_path.is_symlink() or not marker_path.is_file():
            raise LocalCampaignError("completion marker is not a physical file")
        resolved_marker = marker_path.resolve(strict=True)
        if not resolved_marker.is_relative_to(CHECKPOINT_ROOT.resolve(strict=True)):
            raise LocalCampaignError("completion marker escaped checkpoint root")
        document = _load_strict_object(resolved_marker)
        expected = {
            "completion_version",
            "campaign_version",
            "status",
            "model_id",
            "campaign_id",
            "generation2_checkpoint",
            "final_summary",
            "export",
        }
        if not isinstance(document, dict) or set(document) != expected:
            raise LocalCampaignError("completion marker fields are not exact")
        if (
            document["completion_version"]
            != "phase2-generation-completion-v1"
            or document["campaign_version"] != CAMPAIGN_VERSION
            or document["status"] != "completed"
            or document["model_id"] != self.model_id
        ):
            raise LocalCampaignError("completion marker contract mismatch")
        campaign_id = _safe_identifier(
            document["campaign_id"], "completed campaign_id"
        )
        checkpoint_value = document["generation2_checkpoint"]
        expected_checkpoint = (
            Path("phase2_local") / campaign_id / "generation2.json"
        )
        if (
            not isinstance(checkpoint_value, str)
            or checkpoint_value != expected_checkpoint.as_posix()
        ):
            raise LocalCampaignError("completion checkpoint path is invalid")

        directory = ARTIFACT_ROOT / campaign_id
        if directory.is_symlink() or not directory.is_dir():
            raise LocalCampaignError("completed campaign directory is unavailable")
        campaign_directory = directory.resolve(strict=True)
        if not campaign_directory.is_relative_to(
            ARTIFACT_ROOT.resolve(strict=True)
        ):
            raise LocalCampaignError("completed campaign directory escaped artifacts")

        def checked_project_artifact(
            metadata: object,
            *,
            expected_relative: str | None = None,
        ) -> Path:
            if not isinstance(metadata, dict) or set(metadata) != {
                "path",
                "sha256",
            }:
                raise LocalCampaignError("completion artifact metadata is invalid")
            relative_value = metadata["path"]
            digest = metadata["sha256"]
            if (
                not isinstance(relative_value, str)
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise LocalCampaignError("completion artifact scalar is invalid")
            relative = Path(relative_value)
            if (
                relative.is_absolute()
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
                or (
                    expected_relative is not None
                    and relative.as_posix() != expected_relative
                )
            ):
                raise LocalCampaignError("completion artifact path is unsafe")
            path = PROJECT_ROOT / relative
            if path.is_symlink() or not path.is_file():
                raise LocalCampaignError("completion artifact is unavailable")
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(PROJECT_ROOT.resolve(strict=True)):
                raise LocalCampaignError("completion artifact escaped project")
            if _sha256_path(resolved) != digest:
                raise LocalCampaignError("completion artifact hash mismatch")
            return resolved

        expected_summary = (
            campaign_directory.relative_to(PROJECT_ROOT).as_posix()
            + "/final_summary.json"
        )
        summary_path = checked_project_artifact(
            document["final_summary"],
            expected_relative=expected_summary,
        )
        expected_export = f"evolved_{_slug(self.model_id)}.py"
        export_path = checked_project_artifact(
            document["export"],
            expected_relative=expected_export,
        )
        summary = _load_strict_object(summary_path)
        if (
            summary.get("status") != "completed"
            or summary.get("campaign_id") != campaign_id
            or summary.get("model_id") != self.model_id
            or summary.get("export_path") != expected_export
            or summary.get("best_program_sha256") != _sha256_path(export_path)
        ):
            raise LocalCampaignError("final summary completion evidence mismatch")

        plan_path = campaign_directory / "generation2_finalization_plan.json"
        plan = _load_strict_object(plan_path)
        expected_plan_fields = {
            "finalization_version",
            "campaign_id",
            "model_id",
            "run_manifest_sha256",
            "generation2_checkpoint",
            "final_state",
            "export_name",
            "export_source",
            "export_sha256",
            "summary",
        }
        if set(plan) != expected_plan_fields:
            raise LocalCampaignError(
                "completed finalization plan fields are not exact"
            )
        if (
            plan["finalization_version"] != FINALIZATION_PLAN_VERSION
            or plan["campaign_id"] != campaign_id
            or plan["model_id"] != self.model_id
            or plan["generation2_checkpoint"] != checkpoint_value
            or plan["summary"] != summary
        ):
            raise LocalCampaignError("completed finalization plan identity mismatch")
        sealed_state = plan["final_state"]
        if not isinstance(sealed_state, dict) or any(
            summary.get(key) != value for key, value in sealed_state.items()
        ):
            raise LocalCampaignError(
                "completed final state is not bound by the sealed summary"
            )
        final = self._relocated_completed_state(
            sealed_state,
            campaign_id=campaign_id,
            campaign_directory=campaign_directory,
        )
        if (
            final.generation != 2
            or final.model_id != self.model_id
            or final.campaign_id != campaign_id
            or plan["run_manifest_sha256"] != final.run_manifest_sha256
        ):
            raise LocalCampaignError("completed state identity mismatch")

        export_source = plan["export_source"]
        export_sha256 = plan["export_sha256"]
        try:
            packaged_export_source = export_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise LocalCampaignError("completed export is not UTF-8") from exc
        matching = [
            candidate
            for candidate in final.candidates
            if candidate.candidate_id == summary.get("best_candidate_id")
        ]
        if (
            plan["export_name"] != expected_export
            or not isinstance(export_source, str)
            or not isinstance(export_sha256, str)
            or hashlib.sha256(export_source.encode("utf-8")).hexdigest()
            != export_sha256
            or export_sha256 != _sha256_path(export_path)
            or packaged_export_source != export_source
            or len(matching) != 1
            or matching[0].generation == 0
            or matching[0].rendered_source != export_source
            or matching[0].rendered_sha256 != export_sha256
        ):
            raise LocalCampaignError("completed export identity mismatch")

        try:
            checkpoint_state = load_checkpoint(expected_checkpoint)
        except FileNotFoundError:
            # A packaged finalization plan is the sealed source of truth.  The
            # generation-two checkpoint is an optional resume cache and is not
            # recreated here as if it were an original experiment artifact.
            checkpoint_state = None
        if checkpoint_state is not None:
            checkpoint = self._relocated_completed_state(
                checkpoint_state,
                campaign_id=campaign_id,
                campaign_directory=campaign_directory,
            )
            if checkpoint.to_mapping() != final.to_mapping():
                raise LocalCampaignError(
                    "completed checkpoint differs from finalization plan"
                )

        self.campaign_id = final.campaign_id
        self.campaign_directory = campaign_directory
        self.gpu_elapsed_seconds = final.gpu_elapsed_seconds
        self.failures = [dict(item) for item in final.failures]
        self.execution_class = final.execution_class
        self.run_manifest_path = final.run_manifest_path
        self.run_manifest_sha256 = final.run_manifest_sha256
        self._verify_sealed_completed_run_manifest(
            campaign_directory=campaign_directory,
            expected_export=expected_export,
        )
        return final

    def start_generation_one(self) -> CampaignState:
        completed = self._load_completed()
        if completed is not None:
            return completed
        authorization = self._require_receipt()
        self._ensure_manifests()
        self.campaign_id = _campaign_id(self.model_id, "campaign")
        self.campaign_directory = ensure_internal_directory(
            ARTIFACT_ROOT / self.campaign_id
        )
        self._create_run_manifest(kind="formal_evolution")
        seed = self._seed()
        requests = self._requests(
            parents=[seed],
            generation=1,
            children_per_parent=2,
        )
        outcomes = self._mutate(
            requests,
            artifact_name="generation1_mutations.json",
        )
        children = self._build_children(
            parents={seed.candidate_id: seed},
            requests=requests,
            outcomes=outcomes,
            existing_hashes={seed.rendered_sha256},
        )
        evaluated: list[LocalCandidate] = [seed]
        if children:
            evaluated = self._evaluate(
                [seed, *children],
                stage="single",
                artifact_name="generation1_single.json",
            )
            seed_single = evaluated[0]
            valid_generation_one = [
                candidate
                for candidate in evaluated
                if candidate.generation == 1
                and int(candidate.scores["single"]["errors"]) == 0
            ]
            if valid_generation_one:
                searched = self._evaluate(
                    [seed_single, *valid_generation_one],
                    stage="search12",
                    artifact_name="generation1_search12.json",
                )
                searched_by_id = {
                    candidate.candidate_id: candidate for candidate in searched
                }
                evaluated = [
                    searched_by_id.get(candidate.candidate_id, candidate)
                    for candidate in evaluated
                ]
        state = CampaignState(
            campaign_id=self.campaign_id,
            model_id=self.model_id,
            generation=1,
            gpu_elapsed_seconds=self.gpu_elapsed_seconds,
            candidates=tuple(evaluated),
            failures=tuple(self.failures),
            campaign_directory=str(self.campaign_directory),
            execution_class=self.execution_class,
            run_manifest_path=self.run_manifest_path,
            run_manifest_sha256=self.run_manifest_sha256,
        )
        write_checkpoint(self.checkpoint_relative, state.to_mapping())
        write_checkpoint(
            self.active_relative,
            {
                "campaign_version": CAMPAIGN_VERSION,
                "campaign_id": self.campaign_id,
                "model_id": self.model_id,
                "generation": 1,
                "checkpoint": self.checkpoint_relative.as_posix(),
            },
        )
        atomic_create_json(
            self.campaign_directory / "generation1_summary.json",
            {
                **state.to_mapping(),
                "authorization_receipt_id": authorization.get(
                    "receipt_id", "TEST_ONLY"
                ),
                "next_command": (
                    f"python evolve_{_slug(self.model_id)}.py --resume"
                ),
                "child_attempts": 2,
                "valid_generation1_children": sum(
                    candidate.generation == 1
                    and "single" in candidate.scores
                    and "search12" in candidate.scores
                    and int(candidate.scores["single"]["errors"]) == 0
                    and int(candidate.scores["search12"]["errors"]) == 0
                    for candidate in evaluated
                ),
                "manual_seed_same_single_search_protocol": bool(children),
                "gpu_budget_seconds": self.gpu_budget_seconds,
            },
        )
        return state

    def _load_generation_one(self) -> CampaignState:
        pointer = load_checkpoint(self.active_relative)
        expected = {
            "campaign_version",
            "campaign_id",
            "model_id",
            "generation",
            "checkpoint",
        }
        if set(pointer) != expected:
            raise LocalCampaignError("active checkpoint pointer is invalid")
        if (
            pointer["campaign_version"] != CAMPAIGN_VERSION
            or pointer["model_id"] != self.model_id
            or pointer["generation"] != 1
        ):
            raise LocalCampaignError("active checkpoint is not resumable")
        state = CampaignState.from_mapping(
            load_checkpoint(Path(str(pointer["checkpoint"])))
        )
        if state.model_id != self.model_id or state.generation != 1:
            raise LocalCampaignError(
                "generation-one state model/generation mismatch"
            )
        directory = Path(state.campaign_directory)
        if directory.is_symlink() or not directory.is_dir():
            raise LocalCampaignError("campaign directory is unavailable")
        resolved = directory.resolve(strict=True)
        if not resolved.is_relative_to(ARTIFACT_ROOT.resolve(strict=True)):
            raise LocalCampaignError(
                "campaign directory escaped local artifact root"
            )
        self.campaign_id = state.campaign_id
        self.campaign_directory = resolved
        self.gpu_elapsed_seconds = state.gpu_elapsed_seconds
        self.failures = [dict(item) for item in state.failures]
        self.execution_class = state.execution_class
        self.run_manifest_path = state.run_manifest_path
        self.run_manifest_sha256 = state.run_manifest_sha256
        self._verify_run_manifest()
        self._ensure_manifests()
        return state

    def resume_generation_two(self) -> CampaignState:
        completed = self._load_completed()
        if completed is not None:
            return completed
        state = self._load_generation_one()
        recovered = self._recover_generation_two_finalization()
        if recovered is not None:
            return recovered
        authorization = self._require_receipt()
        seeds = [
            candidate for candidate in state.candidates
            if candidate.generation == 0
        ]
        eligible_generation_one = [
            candidate
            for candidate in state.candidates
            if candidate.generation == 1
            and "single" in candidate.scores
            and "search12" in candidate.scores
            and int(candidate.scores["single"]["errors"]) == 0
            and int(candidate.scores["search12"]["errors"]) == 0
        ]
        if len(seeds) != 1 or not eligible_generation_one:
            raise LocalCampaignError(
                "no valid generation-one child; early stop"
            )
        seed = seeds[0]
        parents = pareto_order(
            eligible_generation_one,
            stage="search12",
        )[:2]
        requests = self._requests(
            parents=parents,
            generation=2,
            children_per_parent=2,
            population=state.candidates,
        )
        if len(requests) not in {2, 4}:
            raise AssertionError(
                "generation two attempt count must follow eligible parents"
            )
        if not any(parent.generation == 1 for parent in parents):
            raise AssertionError(
                "generation two lacks a generation-one parent"
            )
        atomic_create_json(
            self.generation_two_journal_path,
            {
                "journal_version": GENERATION_TWO_JOURNAL_VERSION,
                "state": "planned_before_gpu",
                "campaign_id": self.campaign_id,
                "model_id": self.model_id,
                "run_manifest_sha256": self.run_manifest_sha256,
                "request_ids": [request.request_id for request in requests],
                "parent_ids": [parent.candidate_id for parent in parents],
                "gpu_scope": {
                    "physical_gpu_index": 0,
                    "cuda_visible_devices": "0",
                    "visible_device_count": 1,
                    "logical_device": "cuda:0",
                },
            },
        )
        outcomes = self._mutate(
            requests,
            artifact_name="generation2_mutations.json",
        )
        existing = {
            candidate.rendered_sha256 for candidate in state.candidates
        }
        children = self._build_children(
            parents={parent.candidate_id: parent for parent in parents},
            requests=requests,
            outcomes=outcomes,
            existing_hashes=existing,
        )
        if not children:
            raise LocalCampaignError(
                "generation two produced no valid child; early stop"
            )
        generation_two = self._evaluate(
            children,
            stage="single",
            artifact_name="generation2_single.json",
        )
        valid_generation_two = [
            candidate
            for candidate in generation_two
            if int(candidate.scores["single"]["errors"]) == 0
        ]
        searched_generation_two = (
            self._evaluate(
                valid_generation_two,
                stage="search12",
                artifact_name="generation2_search12.json",
            )
            if valid_generation_two
            else []
        )
        searched_generation_two_by_id = {
            candidate.candidate_id: candidate
            for candidate in searched_generation_two
        }
        generation_two = [
            searched_generation_two_by_id.get(
                candidate.candidate_id, candidate
            )
            for candidate in generation_two
        ]
        all_real_children = [
            candidate
            for candidate in [
                *(
                    candidate for candidate in state.candidates
                    if candidate.generation == 1
                ),
                *generation_two,
            ]
            if "single" in candidate.scores
            and "search12" in candidate.scores
            and int(candidate.scores["single"]["errors"]) == 0
            and int(candidate.scores["search12"]["errors"]) == 0
        ]
        if len(all_real_children) > MAX_CHILDREN_PER_MODEL:
            raise AssertionError("child budget exceeded")
        if not all_real_children:
            raise LocalCampaignError(
                "no error-free child survived search12"
            )
        top_two_real = pareto_order(
            all_real_children,
            stage="search12",
        )[:2]
        selection = self._evaluate(
            [seed, *top_two_real],
            stage="selection32",
            artifact_name="manual_seed_and_top2_selection32.json",
        )
        selected_seed = selection[0]
        selected_real = [
            candidate
            for candidate in selection[1:]
            if int(candidate.scores["selection32"]["errors"]) == 0
        ]
        if not selected_real:
            raise LocalCampaignError(
                "no real mutation child completed selection32"
            )
        best = sorted(
            selected_real,
            key=lambda item: _score_key(item, "selection32"),
        )[0]
        if best.generation == 0:
            raise AssertionError("evolved export cannot be the manual seed")
        export_name = f"evolved_{_slug(self.model_id)}.py"
        export_path = PROJECT_ROOT / export_name

        selected_all = [
            candidate
            for candidate in selection
            if int(candidate.scores["selection32"]["errors"]) == 0
        ]
        pareto_all = [
            candidate.candidate_id
            for candidate in selected_all
            if not any(
                other.candidate_id != candidate.candidate_id
                and _dominates(
                    other,
                    candidate,
                    "selection32",
                )
                for other in selected_all
            )
        ]
        pareto_children = [
            candidate.candidate_id
            for candidate in selected_real
            if not any(
                other.candidate_id != candidate.candidate_id
                and _dominates(
                    other,
                    candidate,
                    "selection32",
                )
                for other in selected_real
            )
        ]
        seed_score = selected_seed.scores["selection32"]
        best_score = best.scores["selection32"]
        accuracy_delta = (
            float(best_score["accuracy"]) - float(seed_score["accuracy"])
        )
        latency_delta = (
            float(best_score["mean_latency_seconds"])
            - float(seed_score["mean_latency_seconds"])
        )
        dominated_by_seed = _dominates(
            selected_seed,
            best,
            "selection32",
        )

        final_candidates_by_id = {
            candidate.candidate_id: candidate
            for candidate in [
                *state.candidates,
                *generation_two,
                *selection,
            ]
        }
        final = CampaignState(
            campaign_id=self.campaign_id,
            model_id=self.model_id,
            generation=2,
            gpu_elapsed_seconds=self.gpu_elapsed_seconds,
            candidates=tuple(final_candidates_by_id.values()),
            failures=tuple(self.failures),
            campaign_directory=str(self.campaign_directory),
            execution_class=self.execution_class,
            run_manifest_path=self.run_manifest_path,
            run_manifest_sha256=self.run_manifest_sha256,
        )
        generation_two_relative = (
            Path("phase2_local")
            / self.campaign_id
            / "generation2.json"
        )
        summary_document = {
            **final.to_mapping(),
            "status": "completed",
            "authorization_receipt_id": authorization.get(
                "receipt_id", "TEST_ONLY"
            ),
            "provider": getattr(self.provider, "name", "unknown"),
            "child_attempts": 2 + len(requests),
            "generation2_attempts": len(requests),
            "generation2_parent_ids": [
                parent.candidate_id for parent in parents
            ],
            "generation2_has_generation1_parent": True,
            "search12_query_count": STAGE_COUNTS["search12"],
            "selection32_query_count": STAGE_COUNTS["selection32"],
            "manual_seed_same_selection_protocol": True,
            "top2_real_child_ids": [
                candidate.candidate_id for candidate in top_two_real
            ],
            "pareto_all": pareto_all,
            "pareto_children": pareto_children,
            "best_candidate_id": best.candidate_id,
            "best_program_sha256": best.rendered_sha256,
            "best_is_real_mutation_child": True,
            "dominated_by_seed": dominated_by_seed,
            "accuracy_delta_vs_manual_seed": accuracy_delta,
            "latency_delta_seconds_vs_manual_seed": latency_delta,
            "claims_improvement_over_seed": (
                accuracy_delta >= 0
                and latency_delta <= 0
                and (accuracy_delta > 0 or latency_delta < 0)
            ),
            "export_path": export_name,
            "gpu_budget_seconds": self.gpu_budget_seconds,
            "gpu_elapsed_seconds": self.gpu_elapsed_seconds,
        }
        finalization_plan = {
            "finalization_version": FINALIZATION_PLAN_VERSION,
            "campaign_id": self.campaign_id,
            "model_id": self.model_id,
            "run_manifest_sha256": self.run_manifest_sha256,
            "generation2_checkpoint": generation_two_relative.as_posix(),
            "final_state": final.to_mapping(),
            "export_name": export_name,
            "export_source": best.rendered_source,
            "export_sha256": best.rendered_sha256,
            "summary": summary_document,
        }
        atomic_create_json(self.finalization_plan_path, finalization_plan)
        return self._finalize_generation_two(finalization_plan)

    def dry_run(self) -> dict[str, object]:
        self._ensure_manifests()
        seed = self._seed()
        receipt = import_check_rendered_program(
            seed.rendered_source,
            temporary_directory=_managed_import_check_directory(),
        )
        return {
            "campaign_version": CAMPAIGN_VERSION,
            "model_id": self.model_id,
            "seed_source": f"manual_{_slug(self.model_id)}.py",
            "seed_program_sha256": seed.rendered_sha256,
            "seed_import_check": receipt,
            "manifests": validate_disjoint_manifests(self.manifest_root),
            "provider": getattr(self.provider, "name", "unknown"),
            "gpu_called": False,
            "production_fake_backend_available": False,
            "budget": {
                "children_max": MAX_CHILDREN_PER_MODEL,
                "generation1_children": 2,
                "generation2_children": 4,
                "gpu_seconds_per_model": self.gpu_budget_seconds,
            },
        }

    def provider_smoke(
        self,
        *,
        qualification_ordinal: int = 1,
    ) -> dict[str, object]:
        if (
            isinstance(qualification_ordinal, bool)
            or not 1 <= qualification_ordinal <= len(QUALIFICATION_SCHEDULE)
        ):
            raise LocalCampaignError("qualification ordinal is invalid")
        generation, child_index_offset = QUALIFICATION_SCHEDULE[
            qualification_ordinal - 1
        ]
        self._ensure_manifests()
        self.gpu_budget_seconds = min(
            self.gpu_budget_seconds,
            float(PROVIDER_SMOKE_SECONDS),
        )
        self.campaign_id = _campaign_id(
            self.model_id,
            "provider-smoke",
        )
        self.campaign_directory = ensure_internal_directory(
            ARTIFACT_ROOT / self.campaign_id
        )
        self._create_run_manifest(kind="provider_smoke")
        seed = self._seed()
        requests = self._requests(
            parents=[seed],
            generation=generation,
            children_per_parent=1,
            child_index_offset=child_index_offset,
        )
        outcomes = self._mutate(
            requests,
            artifact_name="mutation.json",
        )
        children = self._build_children(
            parents={seed.candidate_id: seed},
            requests=requests,
            outcomes=outcomes,
            existing_hashes={seed.rendered_sha256},
        )
        if len(children) != 1:
            raise LocalCampaignError(
                "provider smoke did not yield one valid child"
            )
        evaluated = self._evaluate(
            children,
            stage="single",
            artifact_name="target_single.json",
        )
        child = evaluated[0]
        if int(child.scores["single"]["errors"]) != 0:
            raise LocalCampaignError(
                "provider smoke restricted target single failed"
            )
        result = {
            "campaign_version": CAMPAIGN_VERSION,
            "kind": "provider_smoke",
            "campaign_id": self.campaign_id,
            "model_id": self.model_id,
            "provider": getattr(self.provider, "name", "unknown"),
            "qualification_ordinal": qualification_ordinal,
            "qualification_focus": mutation_focus(requests[0]),
            "run_manifest_path": self.run_manifest_path,
            "run_manifest_sha256": self.run_manifest_sha256,
            "child": child.to_mapping(),
            "gpu_elapsed_seconds": self.gpu_elapsed_seconds,
            "gpu_budget_seconds": self.gpu_budget_seconds,
            "exported": False,
        }
        summary_path = self.campaign_directory / "smoke_summary.json"
        atomic_create_json(summary_path, result)
        if self.execution_class == "synthetic_test":
            receipt = {
                "receipt_id": "SYNTHETIC_TEST",
                "execution_class": "synthetic_test",
            }
        else:
            receipt = create_valid_child_receipt(
                model_id=self.model_id,
                campaign_version=CAMPAIGN_VERSION,
                campaign_id=self.campaign_id,
                child=child.to_mapping(),
                provider_artifact=(
                    self.campaign_directory / "mutation.json"
                ),
                target_artifact=(
                    self.campaign_directory / "target_single.json"
                ),
                smoke_summary_artifact=summary_path,
                run_manifest_artifact=Path(self.run_manifest_path),
                run_manifest_sha256=self.run_manifest_sha256,
            )
        return {
            **result,
            "valid_child_receipt_id": receipt["receipt_id"],
            "valid_child_receipt_created": True,
        }



def cli_main(model_id: str, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one fixed-model local-Qwen evolution campaign."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--provider-smoke", action="store_true")
    parser.add_argument(
        "--qualification-ordinal",
        type=int,
        choices=(1, 2, 3),
        default=None,
    )
    arguments = parser.parse_args(argv)
    if (
        arguments.qualification_ordinal is not None
        and not arguments.provider_smoke
    ):
        parser.error("--qualification-ordinal requires --provider-smoke")
    campaign = LocalEvolutionCampaign(model_id=model_id)
    if arguments.dry_run:
        result = campaign.dry_run()
    elif arguments.provider_smoke:
        result = campaign.provider_smoke(
            qualification_ordinal=arguments.qualification_ordinal or 1
        )
    elif arguments.resume:
        result = campaign.resume_generation_two().to_mapping()
    else:
        result = campaign.start_generation_one().to_mapping()
    print(
        {
            "campaign_version": CAMPAIGN_VERSION,
            "model_id": model_id,
            "generation": result.get("generation", 0),
            "campaign_id": result.get("campaign_id"),
            "gpu_elapsed_seconds": result.get("gpu_elapsed_seconds", 0.0),
        }
    )
    return 0


__all__ = [
    "CAMPAIGN_VERSION",
    "GPU_SECONDS_PER_MODEL",
    "QUALIFICATION_SCHEDULE",
    "LocalCampaignError",
    "LocalCandidate",
    "LocalEvolutionCampaign",
    "pareto_order",
    "cli_main",
]
