"""Provider-neutral, fail-closed evolution framework."""

from .core import (
    Candidate,
    CHECKPOINT_ROOT,
    CandidateExecutionPolicy,
    CURRENT_EVALUATOR_CAPABILITY,
    EvaluationCapability,
    EvolutionConfig,
    EvolutionRunner,
    EvolutionState,
    ExecutionMode,
    Lineage,
    Metrics,
    MutationProvider,
    MutationRequest,
    MutationResult,
    MutationUnavailable,
    NullProvider,
    FakeProviderForbidden,
    UnsafeCandidateExecution,
    build_candidate,
    build_status,
    credential_presence,
    deterministic_parent_selection,
    export_eligibility,
    load_checkpoint,
    null_provider_from_environment,
    pareto_front,
    seed_candidate,
    write_checkpoint,
)
from .providers import ProviderNeutralAdapter, ProviderResponseError
from .worker import (
    CandidateFormatError,
    CandidateSpec,
    DeclarativeCandidateWorker,
    RestrictedExecutionUnavailable,
    WorkerInput,
    parse_candidate_source,
)

__all__ = [name for name in globals() if not name.startswith("_")]
