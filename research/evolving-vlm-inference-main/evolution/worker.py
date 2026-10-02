"""Declarative candidate worker with no arbitrary Python execution.

Candidates are data-only CANDIDATE assignments parsed by ast.literal_eval.
The milestone backend is test-only; production capability remains TRUSTED_ONLY.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Protocol

from .core import Metrics, UnsafeCandidateExecution


class CandidateFormatError(ValueError):
    pass


class RestrictedExecutionUnavailable(UnsafeCandidateExecution):
    pass


_ALLOWED_FIELDS = {
    "model_family",
    "instruction",
    "question_template",
    "max_new_tokens",
    "answer_mode",
}


@dataclass(frozen=True)
class CandidateSpec:
    model_family: str
    instruction: str
    question_template: str
    max_new_tokens: int
    answer_mode: str

    def __post_init__(self) -> None:
        if self.model_family not in {"instruct", "thinking"}:
            raise CandidateFormatError("model_family must be instruct or thinking")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise CandidateFormatError("instruction must be nonempty text")
        if len(self.instruction) > 4_096 or "\x00" in self.instruction:
            raise CandidateFormatError("instruction is too long or contains NUL")
        if not isinstance(self.question_template, str):
            raise CandidateFormatError("question_template must be text")
        if self.question_template.count("{question}") != 1:
            raise CandidateFormatError("question_template needs exactly one {question}")
        remainder = self.question_template.replace("{question}", "")
        if "{" in remainder or "}" in remainder or len(self.question_template) > 4_096:
            raise CandidateFormatError("question_template has unsafe placeholders")
        if isinstance(self.max_new_tokens, bool) or not isinstance(
            self.max_new_tokens, int
        ):
            raise CandidateFormatError("max_new_tokens must be an integer")
        cap = 48 if self.model_family == "instruct" else 256
        if not 8 <= self.max_new_tokens <= cap:
            raise CandidateFormatError(
                f"max_new_tokens must be between 8 and {cap}"
            )
        if self.answer_mode not in {"direct", "final_tag"}:
            raise CandidateFormatError("answer_mode must be direct or final_tag")

    def render(self, question: str) -> str:
        return self.instruction.strip() + " " + self.question_template.replace(
            "{question}", question.strip()
        )


def parse_candidate_source(source: str) -> CandidateSpec:
    """Parse exactly one CANDIDATE literal assignment; never exec candidate text."""
    if not isinstance(source, str) or not source.strip():
        raise CandidateFormatError("candidate source must be nonempty text")
    if len(source.encode("utf-8")) > 32_768:
        raise CandidateFormatError("candidate source exceeds 32768 bytes")
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, RecursionError, ValueError) as exc:
        raise CandidateFormatError("candidate source is not valid syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise CandidateFormatError("candidate syntax tree is too large")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        raise CandidateFormatError("candidate must contain one assignment only")
    assignment = tree.body[0]
    if (
        len(assignment.targets) != 1
        or not isinstance(assignment.targets[0], ast.Name)
        or assignment.targets[0].id != "CANDIDATE"
    ):
        raise CandidateFormatError("assignment target must be CANDIDATE")
    if not isinstance(assignment.value, ast.Dict):
        raise CandidateFormatError("CANDIDATE must be a literal dictionary")
    literal_keys: list[str] = []
    for key_node in assignment.value.keys:
        if not isinstance(key_node, ast.Constant) or not isinstance(
            key_node.value, str
        ):
            raise CandidateFormatError("CANDIDATE keys must be literal strings")
        literal_keys.append(key_node.value)
    if len(literal_keys) != len(set(literal_keys)):
        raise CandidateFormatError("CANDIDATE contains duplicate keys")
    try:
        document = ast.literal_eval(assignment.value)
    except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
        raise CandidateFormatError("CANDIDATE must be a literal") from exc
    if not isinstance(document, dict) or set(document) != _ALLOWED_FIELDS:
        raise CandidateFormatError("CANDIDATE fields are not exact")
    if not all(isinstance(key, str) for key in document):
        raise CandidateFormatError("CANDIDATE keys must be strings")
    return CandidateSpec(**document)


@dataclass(frozen=True)
class WorkerInput:
    """The complete untrusted-candidate input contract: no key, path, or gold."""

    question: str
    image_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be nonempty text")
        if len(self.question) > 8_192 or "\x00" in self.question:
            raise ValueError("question is too long or contains NUL")
        if not isinstance(self.image_bytes, bytes) or not self.image_bytes:
            raise ValueError("image_bytes must be nonempty bytes")
        if len(self.image_bytes) > 32 * 1024 * 1024:
            raise ValueError("image_bytes exceeds 32 MiB")


@dataclass(frozen=True)
class PreparedCandidate:
    spec: CandidateSpec
    prompt: str
    image_bytes: bytes
    image_sha256: str


class CandidateBackend(Protocol):
    test_only: bool

    def evaluate(self, prepared: PreparedCandidate) -> Metrics:
        pass


@dataclass
class DeclarativeCandidateWorker:
    backend: CandidateBackend
    test_mode: bool = False

    @property
    def capability_report(self) -> dict[str, object]:
        return {
            "candidate_format": "ast_literal_eval_assignment",
            "candidate_python_executed": False,
            "candidate_receives_environment": False,
            "candidate_receives_gold_key_or_path": False,
            "candidate_input_fields": ["question", "image_bytes"],
            "namespace_required": False,
            "production_evaluator_capability": "TRUSTED_ONLY",
            "production_ready": False,
        }

    def prepare(self, source: str, item: WorkerInput) -> PreparedCandidate:
        spec = parse_candidate_source(source)
        prompt = spec.render(item.question)
        return PreparedCandidate(
            spec=spec,
            prompt=prompt,
            image_bytes=bytes(item.image_bytes),
            image_sha256=hashlib.sha256(item.image_bytes).hexdigest(),
        )

    def evaluate(self, source: str, *, question: str, image_bytes: bytes) -> Metrics:
        if not self.test_mode or not self.backend.test_only:
            raise RestrictedExecutionUnavailable(
                "production candidate evaluation remains disabled"
            )
        item = WorkerInput(question=question, image_bytes=image_bytes)
        result = self.backend.evaluate(self.prepare(source, item))
        if not isinstance(result, Metrics):
            raise TypeError("candidate backend must return Metrics")
        return result
