"""Provider-neutral structured mutation adapter.

Only injected test transports are accepted in this milestone. No credential is
read here and no network client is implemented or invoked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

from .core import MutationRequest, MutationResult, MutationUnavailable
from .worker import parse_candidate_source


class ProviderResponseError(ValueError):
    pass


@runtime_checkable
class MutationTransport(Protocol):
    name: str
    test_only: bool

    def complete(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        pass


@dataclass(frozen=True)
class ProviderNeutralAdapter:
    """Validate a transport response without coupling to a vendor SDK."""

    model: str
    transport: MutationTransport
    max_candidate_bytes: int = 32_768

    @property
    def name(self) -> str:
        return f"provider-neutral:{self.transport.name}"

    @property
    def test_only(self) -> bool:
        return True

    def mutate(self, request: MutationRequest) -> MutationResult:
        if not self.transport.test_only:
            raise MutationUnavailable(
                "production mutation transport is not configured in this milestone"
            )
        payload: dict[str, object] = {
            "schema_version": 1,
            "model": self.model,
            "task": "Return one declarative CANDIDATE assignment.",
            "parent_id": request.parent_id,
            "generation": request.generation,
            "child_index": request.child_index,
            "parent_source": request.source,
            "response_fields": ["candidate_source", "mutation_id", "rationale"],
        }
        response = self.transport.complete(payload)
        if not isinstance(response, Mapping):
            raise ProviderResponseError("transport response must be a mapping")
        if set(response) != {"candidate_source", "mutation_id", "rationale"}:
            raise ProviderResponseError("transport response fields are not exact")
        source = response["candidate_source"]
        mutation_id = response["mutation_id"]
        rationale = response["rationale"]
        if not isinstance(source, str) or not source.strip():
            raise ProviderResponseError("candidate_source must be nonempty text")
        if len(source.encode("utf-8")) > self.max_candidate_bytes:
            raise ProviderResponseError("candidate_source exceeds size limit")
        if "```" in source:
            raise ProviderResponseError("candidate_source must not use markdown fences")
        if not isinstance(mutation_id, str) or not mutation_id.strip():
            raise ProviderResponseError("mutation_id must be nonempty text")
        if len(mutation_id) > 128:
            raise ProviderResponseError("mutation_id is too long")
        if not isinstance(rationale, str) or len(rationale) > 4_096:
            raise ProviderResponseError("rationale is invalid")
        parse_candidate_source(source)
        return MutationResult(
            source=source,
            mutation_id=mutation_id.strip(),
            rationale=rationale,
        )
