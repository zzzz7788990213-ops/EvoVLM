"""Pure-stdlib, provider-neutral evolution data flow.

Gate 1 does not execute untrusted candidates. Production use is fail-closed until
the controller can attest a restricted candidate evaluator.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import secrets
import stat
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from .process_safety import EVO_ROOT as EXPECTED_EVO_ROOT, PROJECT_ROOT


CREDENTIAL_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "QWEAPI_API_KEY",
)
CHECKPOINT_VERSION = 1
CHECKPOINT_ROOT = PROJECT_ROOT / "results/evolution/checkpoints"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    latency_seconds: float
    error_rate: float

    def __post_init__(self) -> None:
        values = (self.accuracy, self.latency_seconds, self.error_rate)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("metrics must be finite")
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError("accuracy must be in [0, 1]")
        if self.latency_seconds <= 0.0:
            raise ValueError("latency_seconds must be positive")
        if not 0.0 <= self.error_rate <= 1.0:
            raise ValueError("error_rate must be in [0, 1]")

    def dominates(self, other: "Metrics") -> bool:
        no_worse = (
            self.accuracy >= other.accuracy
            and self.latency_seconds <= other.latency_seconds
            and self.error_rate <= other.error_rate
        )
        strict = (
            self.accuracy > other.accuracy
            or self.latency_seconds < other.latency_seconds
            or self.error_rate < other.error_rate
        )
        return no_worse and strict


@dataclass(frozen=True)
class Lineage:
    parent_id: str | None
    generation: int
    mutation_id: str
    program_sha256: str
    unified_diff: str

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("generation must be nonnegative")
        if not self.mutation_id:
            raise ValueError("mutation_id is required")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    source: str
    lineage: Lineage
    metrics: Metrics | None = None
    evaluation_error: str | None = None

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("candidate source must be nonempty")
        if _sha256(self.source) != self.lineage.program_sha256:
            raise ValueError("lineage program hash does not match source")
        if self.metrics is not None and self.evaluation_error is not None:
            raise ValueError("metrics and evaluation_error are mutually exclusive")


@dataclass(frozen=True)
class MutationRequest:
    parent_id: str
    source: str
    generation: int
    child_index: int


@dataclass(frozen=True)
class MutationResult:
    source: str
    mutation_id: str
    rationale: str = ""


@runtime_checkable
class MutationProvider(Protocol):
    name: str
    test_only: bool

    def mutate(self, request: MutationRequest) -> MutationResult:
        """Return source text without executing it."""


class MutationUnavailable(RuntimeError):
    pass


def credential_presence(environ: Mapping[str, str] | None = None) -> dict[str, bool]:
    """Inspect presence only; never retain or return credential values."""

    source = os.environ if environ is None else environ
    return {name: bool(source.get(name)) for name in CREDENTIAL_ENV_VARS}


@dataclass(frozen=True)
class NullProvider:
    credential_presence: Mapping[str, bool]
    name: str = "null"
    test_only: bool = False

    def __post_init__(self) -> None:
        sanitized = {
            name: bool(self.credential_presence.get(name, False))
            for name in CREDENTIAL_ENV_VARS
        }
        object.__setattr__(self, "credential_presence", MappingProxyType(sanitized))

    def mutate(self, request: MutationRequest) -> MutationResult:
        del request
        raise MutationUnavailable(
            "No real mutation provider is configured; evolution did not run."
        )


def null_provider_from_environment(
    environ: Mapping[str, str] | None = None,
) -> NullProvider:
    return NullProvider(credential_presence(environ))


class EvaluationCapability(str, Enum):
    TRUSTED_ONLY = "TRUSTED_ONLY"
    RESTRICTED_CANDIDATE = "RESTRICTED_CANDIDATE"


class ExecutionMode(str, Enum):
    PRODUCTION = "PRODUCTION"
    TEST_ONLY = "TEST_ONLY"


class UnsafeCandidateExecution(RuntimeError):
    pass


class FakeProviderForbidden(RuntimeError):
    pass


@dataclass(frozen=True)
class CandidateExecutionPolicy:
    evaluator_capability: EvaluationCapability
    execution_mode: ExecutionMode = ExecutionMode.PRODUCTION

    @property
    def production_allowed(self) -> bool:
        return (
            self.execution_mode is ExecutionMode.PRODUCTION
            and self.evaluator_capability
            is EvaluationCapability.RESTRICTED_CANDIDATE
        )

    def assert_allowed(self, *, provider_test_only: bool) -> None:
        """Fail closed; deliberately has no force/bypass argument."""

        if (
            self.execution_mode is ExecutionMode.PRODUCTION
            and self.evaluator_capability
            is not EvaluationCapability.RESTRICTED_CANDIDATE
        ):
            raise UnsafeCandidateExecution(
                "Production evolution requires a RESTRICTED_CANDIDATE evaluator; "
                "the current evaluator is TRUSTED_ONLY."
            )
        if provider_test_only and self.execution_mode is not ExecutionMode.TEST_ONLY:
            raise FakeProviderForbidden(
                "A test-only mutation provider cannot be used in production."
            )


def _candidate_id(
    source_hash: str,
    parent_id: str | None,
    generation: int,
    mutation_id: str,
) -> str:
    material = "\n".join(
        (parent_id or "ROOT", str(generation), mutation_id, source_hash)
    )
    return _sha256(material)[:24]


def seed_candidate(source: str) -> Candidate:
    source_hash = _sha256(source)
    return Candidate(
        candidate_id=_candidate_id(source_hash, None, 0, "seed"),
        source=source,
        lineage=Lineage(None, 0, "seed", source_hash, ""),
    )


def build_candidate(
    parent: Candidate,
    source: str,
    *,
    generation: int,
    mutation_id: str,
) -> Candidate:
    if generation <= parent.lineage.generation:
        raise ValueError("child generation must exceed parent generation")
    source_hash = _sha256(source)
    unified_diff = "".join(
        difflib.unified_diff(
            parent.source.splitlines(keepends=True),
            source.splitlines(keepends=True),
            fromfile=f"{parent.candidate_id}.py",
            tofile="candidate.py",
        )
    )
    return Candidate(
        candidate_id=_candidate_id(
            source_hash, parent.candidate_id, generation, mutation_id
        ),
        source=source,
        lineage=Lineage(
            parent.candidate_id,
            generation,
            mutation_id,
            source_hash,
            unified_diff,
        ),
    )


def _selection_key(candidate: Candidate) -> tuple[float, float, float, str]:
    if candidate.metrics is None:
        raise ValueError("candidate is not evaluated")
    return (
        -candidate.metrics.accuracy,
        candidate.metrics.latency_seconds,
        candidate.metrics.error_rate,
        candidate.candidate_id,
    )


def pareto_front(candidates: Iterable[Candidate]) -> tuple[Candidate, ...]:
    pool = tuple(
        candidate
        for candidate in candidates
        if candidate.metrics is not None and candidate.evaluation_error is None
    )
    front = tuple(
        candidate
        for candidate in pool
        if not any(
            other.metrics.dominates(candidate.metrics)
            for other in pool
            if other.candidate_id != candidate.candidate_id
        )
    )
    return tuple(sorted(front, key=_selection_key))


def deterministic_parent_selection(
    candidates: Sequence[Candidate],
    count: int,
) -> tuple[Candidate, ...]:
    if count < 1:
        raise ValueError("parent count must be positive")
    eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.metrics is not None and candidate.evaluation_error is None
    )
    if not eligible:
        raise ValueError("no evaluated candidates are available")
    front = pareto_front(eligible)
    front_ids = {candidate.candidate_id for candidate in front}
    remainder = tuple(
        sorted(
            (
                candidate
                for candidate in eligible
                if candidate.candidate_id not in front_ids
            ),
            key=_selection_key,
        )
    )
    ordered = front + remainder
    return ordered[: min(count, len(ordered))]


def export_eligibility(candidate: Candidate) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if candidate.metrics is None:
        reasons.append("missing_metrics")
    if candidate.evaluation_error is not None:
        reasons.append("evaluation_error")
    if candidate.metrics is not None and candidate.metrics.error_rate != 0.0:
        reasons.append("nonzero_error_rate")
    if _sha256(candidate.source) != candidate.lineage.program_sha256:
        reasons.append("program_hash_mismatch")
    return not reasons, tuple(reasons)


def _checkpoint_parts(path: str | Path) -> tuple[str, ...]:
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError("checkpoint path must be relative to the fixed checkpoint root")
    parts = candidate.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError("checkpoint path contains an unsafe component")
    return parts


def _open_child_directory(parent_fd: int, name: str, *, create: bool) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        return os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"checkpoint directory is not a real directory: {name}") from exc


def _open_checkpoint_root(*, create: bool) -> int:
    configured_text = os.environ.get("EVOLVING_VLM_RUNTIME_ROOT") or os.environ.get(
        "EVO_ROOT"
    )
    if configured_text and (
        Path(configured_text).expanduser().resolve(strict=False) != EXPECTED_EVO_ROOT
    ):
        raise ValueError("configured runtime root differs from imported runtime")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(EXPECTED_EVO_ROOT, flags)
    except OSError as exc:
        raise ValueError("Evo root is not a real directory") from exc
    relative = CHECKPOINT_ROOT.relative_to(EXPECTED_EVO_ROOT)
    try:
        for component in relative.parts:
            child = _open_child_directory(descriptor, component, create=create)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _open_checkpoint_parent(
    parts: tuple[str, ...],
    *,
    create: bool,
) -> tuple[int, str]:
    descriptor = _open_checkpoint_root(create=create)
    try:
        for component in parts[:-1]:
            child = _open_child_directory(descriptor, component, create=create)
            os.close(descriptor)
            descriptor = child
        return descriptor, parts[-1]
    except Exception:
        os.close(descriptor)
        raise


def write_checkpoint(path: str | Path, state: Mapping[str, object]) -> None:
    """Atomically write below the fixed checkpoint root without following links."""

    parts = _checkpoint_parts(path)
    parent_fd, name = _open_checkpoint_parent(parts, create=True)
    temporary = f".{name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    document = {"checkpoint_version": CHECKPOINT_VERSION, "state": state}
    try:
        try:
            existing = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ValueError("checkpoint target must be a regular file")
        descriptor = os.open(
            temporary,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                document,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)


def load_checkpoint(path: str | Path) -> dict[str, object]:
    """Load a regular checkpoint below the fixed root without following links."""

    parts = _checkpoint_parts(path)
    parent_fd, name = _open_checkpoint_parent(parts, create=False)
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("checkpoint must be a regular file")
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError("opened checkpoint is not a regular file")
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                descriptor = -1
                document = json.load(handle)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    finally:
        os.close(parent_fd)
    if document.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version")
    state = document.get("state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint state must be an object")
    return state


@dataclass(frozen=True)
class EvolutionConfig:
    generations: int
    parent_count: int = 2
    children_per_parent: int = 2
    max_population: int = 32

    def __post_init__(self) -> None:
        if min(
            self.generations,
            self.parent_count,
            self.children_per_parent,
            self.max_population,
        ) < 1:
            raise ValueError("evolution configuration values must be positive")


@dataclass(frozen=True)
class EvolutionState:
    generation: int
    candidates: tuple[Candidate, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "candidates": [asdict(candidate) for candidate in self.candidates],
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, object]) -> "EvolutionState":
        raw_candidates = document.get("candidates")
        if not isinstance(raw_candidates, list):
            raise ValueError("checkpoint candidates must be a list")
        candidates: list[Candidate] = []
        for raw in raw_candidates:
            if not isinstance(raw, dict):
                raise ValueError("candidate must be an object")
            lineage_raw = raw.get("lineage")
            if not isinstance(lineage_raw, dict):
                raise ValueError("lineage must be an object")
            metrics_raw = raw.get("metrics")
            metrics = Metrics(**metrics_raw) if isinstance(metrics_raw, dict) else None
            error_raw = raw.get("evaluation_error")
            candidates.append(
                Candidate(
                    candidate_id=str(raw["candidate_id"]),
                    source=str(raw["source"]),
                    lineage=Lineage(**lineage_raw),
                    metrics=metrics,
                    evaluation_error=(
                        None if error_raw is None else str(error_raw)
                    ),
                )
            )
        return cls(
            generation=int(document["generation"]),
            candidates=tuple(candidates),
        )


Evaluator = Callable[[str], Metrics]


@dataclass
class EvolutionRunner:
    provider: MutationProvider
    evaluator: Evaluator
    policy: CandidateExecutionPolicy
    checkpoint_path: Path | None = None

    def _evaluate(self, candidate: Candidate) -> Candidate:
        try:
            metrics = self.evaluator(candidate.source)
        except Exception as exc:
            return replace(
                candidate,
                evaluation_error=f"{type(exc).__name__}: {exc}",
            )
        if not isinstance(metrics, Metrics):
            raise TypeError("evaluator must return Metrics")
        return replace(candidate, metrics=metrics)

    def _save(self, state: EvolutionState) -> None:
        if self.checkpoint_path is not None:
            write_checkpoint(self.checkpoint_path, state.to_dict())

    def run(
        self,
        seed_source: str,
        config: EvolutionConfig,
        *,
        resume: bool = False,
    ) -> EvolutionState:
        """Run/resume only after the non-bypassable policy passes."""

        self.policy.assert_allowed(provider_test_only=self.provider.test_only)
        if resume:
            if self.checkpoint_path is None:
                raise ValueError("resume requires checkpoint_path")
            state = EvolutionState.from_dict(
                load_checkpoint(self.checkpoint_path)
            )
        else:
            state = EvolutionState(
                generation=0,
                candidates=(self._evaluate(seed_candidate(seed_source)),),
            )
            self._save(state)

        if state.generation > config.generations:
            raise ValueError("checkpoint is ahead of requested generation")
        while state.generation < config.generations:
            generation = state.generation + 1
            parents = deterministic_parent_selection(
                state.candidates,
                config.parent_count,
            )
            children: list[Candidate] = []
            for parent in parents:
                for child_index in range(config.children_per_parent):
                    mutation = self.provider.mutate(
                        MutationRequest(
                            parent.candidate_id,
                            parent.source,
                            generation,
                            child_index,
                        )
                    )
                    child = build_candidate(
                        parent,
                        mutation.source,
                        generation=generation,
                        mutation_id=mutation.mutation_id,
                    )
                    children.append(self._evaluate(child))
            combined = state.candidates + tuple(children)
            state = EvolutionState(
                generation=generation,
                candidates=deterministic_parent_selection(
                    combined,
                    min(config.max_population, len(combined)),
                ),
            )
            self._save(state)
        return state


CURRENT_EVALUATOR_CAPABILITY = EvaluationCapability.TRUSTED_ONLY
RESTRICTED_WORKER_READY = True
PROVIDER_AVAILABLE = True
PROVIDER_SMOKE_VALID_CHILD = False
LOCAL_PROVIDER_NAME = (
    "LocalQwenMutationProvider/Qwen3-VL-2B-Instruct@78448d793a7eb2f7a987a1da76d464384aa1becd/greedy-bfloat16-sdpa-offline"
)
REAL_EVOLUTION_ALLOWED = False


def build_status(
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    presence = credential_presence(environ)
    authorization: dict[str, dict[str, object]] = {}
    provider_smoke_attempted_by_model = {
        "instruct": True,
        "thinking": False,
    }
    try:
        from .dsl import INSTRUCT_MODEL_ID, THINKING_MODEL_ID
        from .local_campaign import ARTIFACT_ROOT, CAMPAIGN_VERSION
        from .valid_child_receipt import (
            receipt_path,
            verify_valid_child_receipt,
        )

        for slug, model_id in (
            ("instruct", INSTRUCT_MODEL_ID),
            ("thinking", THINKING_MODEL_ID),
        ):
            path = receipt_path(model_id)
            verified = False
            receipt_id = None
            error_type = None
            try:
                receipt = verify_valid_child_receipt(
                    model_id,
                    campaign_version=CAMPAIGN_VERSION,
                )
                verified = True
                receipt_id = receipt.get("receipt_id")
            except Exception as exc:
                error_type = type(exc).__name__
            authorization[slug] = {
                "present": (
                    path.is_file() and not path.is_symlink()
                ),
                "verified": verified,
                "receipt_id": receipt_id,
                "error_type": error_type,
            }
        provider_smoke_attempted_by_model = {
            slug: bool(value["present"])
            or any(
                candidate.is_file() and not candidate.is_symlink()
                for candidate in ARTIFACT_ROOT.glob(
                    f"provider-smoke-{slug}-*/mutation.json"
                )
            )
            for slug, value in authorization.items()
        }
    except Exception as exc:
        authorization = {
            slug: {
                "present": False,
                "verified": False,
                "receipt_id": None,
                "error_type": type(exc).__name__,
            }
            for slug in ("instruct", "thinking")
        }
    per_model_allowed = {
        slug: bool(value["verified"])
        for slug, value in authorization.items()
    }
    blockers = [
        f"valid_child_receipt_not_verified:{slug}"
        for slug, allowed in per_model_allowed.items()
        if not allowed
    ]
    all_models_allowed = all(per_model_allowed.values())
    return {
        "blockers": blockers,
        "candidate_evaluator_capability": (
            EvaluationCapability.RESTRICTED_CANDIDATE.value
        ),
        "credential_presence": presence,
        "provider_available": True,
        "provider_name": LOCAL_PROVIDER_NAME,
        "provider_smoke_attempted": any(
            provider_smoke_attempted_by_model.values()
        ),
        "provider_smoke_attempted_by_model": (
            provider_smoke_attempted_by_model
        ),
        "provider_smoke_executed": any(
            provider_smoke_attempted_by_model.values()
        ),
        "provider_smoke_valid_child": any(
            per_model_allowed.values()
        ),
        "target_single_passed": any(per_model_allowed.values()),
        "target_single_passed_by_model": dict(per_model_allowed),
        "valid_child_receipt_verified": any(
            per_model_allowed.values()
        ),
        "valid_child_receipt_verified_by_model": dict(
            per_model_allowed
        ),
        "valid_child_authorization": authorization,
        "production_model_allowed": per_model_allowed,
        "restricted_worker_ready": True,
        "restricted_worker_mode": (
            "PRODUCTION_TRUSTED_RENDERER_ONLY"
        ),
        "production_evolution_allowed": all_models_allowed,
        "real_evolution_allowed": all_models_allowed,
        "real_evolution_completed": False,
        "legacy_status": {
            "credential_provider_absent": (
                "superseded_by_offline_local_provider"
            ),
            "duplicate_provider_smoke": (
                "superseded_no_valid_child"
            ),
            "test_only_worker": (
                "superseded_by_trusted_renderer_worker"
            ),
        },
        "status": "ready" if all_models_allowed else "blocked",
    }
