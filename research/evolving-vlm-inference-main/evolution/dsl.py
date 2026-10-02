"""Strict declarative DSL for renderer-generated VLM inference programs."""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping


INSTRUCT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
THINKING_MODEL_ID = "Qwen/Qwen3-VL-2B-Thinking"
DSL_VERSION = "phase2-declarative-v2"
MODEL_IDS = (INSTRUCT_MODEL_ID, THINKING_MODEL_ID)
MODEL_METADATA: dict[str, dict[str, str]] = {
    INSTRUCT_MODEL_ID: {
        "revision": "78448d793a7eb2f7a987a1da76d464384aa1becd",
        "directory": "Qwen3-VL-2B-Instruct",
        "weight_sha256": (
            "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
        ),
    },
    THINKING_MODEL_ID: {
        "revision": "d4f1d9f1b0107e12699c3c83d7baacd07fd0abd3",
        "directory": "Qwen3-VL-2B-Thinking",
        "weight_sha256": (
            "d889ae8b5bffcfce115bc64e47601b1e90ce44b1d9f40ed78e620fd22417d888"
        ),
    },
}

PIXEL_QUANTUM = 28 * 28
DEFAULT_MIN_PIXELS = 256 * PIXEL_QUANTUM
DEFAULT_MAX_PIXELS = 1280 * PIXEL_QUANTUM
MIN_PIXEL_CHOICES = tuple(value * PIXEL_QUANTUM for value in (128, 256, 512))
MAX_PIXEL_CHOICES = tuple(
    value * PIXEL_QUANTUM for value in (1024, 1280, 2048, 4096)
)
PREPROCESSING_OPTIONS = (
    "qwen_default",
    "rgb",
    "autocontrast_rgb",
    "sharpen_rgb",
)
EXTRACTION_STRATEGIES = (
    "final_tag",
    "direct",
    "last_nonempty_line",
    "boxed_or_final",
)
POSTPROCESSING_STRATEGIES = (
    "strip",
    "normalize_whitespace",
    "strip_terminal_punctuation",
    "normalize_numeric",
)
REASONING_MODES = ("direct", "thinking")
FENCE = chr(96) * 3

_ALLOWED_FIELDS = {
    "model_id",
    "system_instruction",
    "prompt_template",
    "reasoning_mode",
    "max_new_tokens",
    "min_pixels",
    "max_pixels",
    "preprocessing_option",
    "extraction_strategy",
    "postprocessing_strategy",
    "dtype",
    "attention",
    "batch_size",
    "do_sample",
    "num_beams",
    "num_return_sequences",
    "use_cache",
}

_FORBIDDEN_TEXT_MARKERS = (
    "query_key",
    "figure_id",
    "subq_idx",
    "expected_answer",
    "ground_truth",
    "sample_id",
    "question_id",
    "/scratch/",
    "file://",
    "http://",
    "https://",
    "ssh_",
    "api_key",
    "subprocess",
    "socket.",
    "__import__",
    "open(",
    "eval(",
    "exec(",
)
_HEX_64 = re.compile(r"[0-9a-f]{64}")


class CandidateDslError(ValueError):
    """Base class for strict candidate rejection."""


class CandidateDslFormatError(CandidateDslError):
    """Syntactic/outer-format error that does not relax semantic checks."""


class CandidateDslSecurityError(CandidateDslError):
    """Semantic or security rejection; never eligible for format repair."""


class RecoverableOuterFormatError(CandidateDslFormatError):
    """One fenced assignment was recoverable for semantic-preserving repair."""

    def __init__(self, message: str, canonical_source: str) -> None:
        super().__init__(message)
        self.canonical_source = canonical_source


@dataclass(frozen=True)
class ProgramSpec:
    model_id: str
    system_instruction: str
    prompt_template: str
    reasoning_mode: str
    max_new_tokens: int
    min_pixels: int
    max_pixels: int
    preprocessing_option: str
    extraction_strategy: str
    postprocessing_strategy: str
    dtype: str
    attention: str
    batch_size: int
    do_sample: bool
    num_beams: int
    num_return_sequences: int
    use_cache: bool

    def __post_init__(self) -> None:
        if self.model_id not in MODEL_IDS:
            raise CandidateDslSecurityError(
                "model_id is not an approved Qwen3-VL 2B model"
            )
        for name, value, limit in (
            ("system_instruction", self.system_instruction, 4096),
            ("prompt_template", self.prompt_template, 4096),
        ):
            if not isinstance(value, str):
                raise CandidateDslSecurityError(f"{name} must be text")
            if name == "prompt_template" and not value.strip():
                raise CandidateDslSecurityError("prompt_template must be nonempty")
            if len(value) > limit or "\x00" in value:
                raise CandidateDslSecurityError(
                    f"{name} is too long or contains NUL"
                )
            lowered = value.casefold()
            marker = next(
                (item for item in _FORBIDDEN_TEXT_MARKERS if item in lowered),
                None,
            )
            if marker is not None:
                raise CandidateDslSecurityError(
                    f"{name} contains forbidden marker: {marker}"
                )
        if self.prompt_template.count("{question}") != 1:
            raise CandidateDslSecurityError(
                "prompt_template must contain exactly one {question}"
            )
        remainder = self.prompt_template.replace("{question}", "")
        if "{" in remainder or "}" in remainder:
            raise CandidateDslSecurityError(
                "prompt_template contains an unapproved placeholder"
            )
        if self.reasoning_mode not in REASONING_MODES:
            raise CandidateDslSecurityError("reasoning_mode is not approved")
        required_reasoning = (
            "direct" if self.model_id == INSTRUCT_MODEL_ID else "thinking"
        )
        if self.reasoning_mode != required_reasoning:
            raise CandidateDslSecurityError(
                "reasoning_mode is fixed by the population model"
            )
        if isinstance(self.max_new_tokens, bool) or not isinstance(
            self.max_new_tokens, int
        ):
            raise CandidateDslSecurityError("max_new_tokens must be an integer")
        lower, upper = (
            (24, 128)
            if self.model_id == INSTRUCT_MODEL_ID
            else (128, 512)
        )
        if not lower <= self.max_new_tokens <= upper:
            raise CandidateDslSecurityError(
                f"max_new_tokens must be in [{lower}, {upper}] for this model"
            )
        if self.min_pixels not in MIN_PIXEL_CHOICES:
            raise CandidateDslSecurityError("min_pixels is not pre-approved")
        if self.max_pixels not in MAX_PIXEL_CHOICES:
            raise CandidateDslSecurityError("max_pixels is not pre-approved")
        if self.min_pixels > self.max_pixels:
            raise CandidateDslSecurityError("min_pixels exceeds max_pixels")
        if self.preprocessing_option not in PREPROCESSING_OPTIONS:
            raise CandidateDslSecurityError(
                "preprocessing_option is not approved"
            )
        if self.preprocessing_option == "qwen_default" and (
            self.min_pixels != DEFAULT_MIN_PIXELS
            or self.max_pixels != DEFAULT_MAX_PIXELS
        ):
            raise CandidateDslSecurityError(
                "qwen_default requires canonical default pixel bounds"
            )
        if self.extraction_strategy not in EXTRACTION_STRATEGIES:
            raise CandidateDslSecurityError(
                "extraction_strategy is not approved"
            )
        if self.postprocessing_strategy not in POSTPROCESSING_STRATEGIES:
            raise CandidateDslSecurityError(
                "postprocessing_strategy is not approved"
            )
        if self.dtype != "bfloat16":
            raise CandidateDslSecurityError("dtype is locked to bfloat16")
        if self.attention != "sdpa":
            raise CandidateDslSecurityError("attention is locked to sdpa")
        if self.batch_size != 1 or isinstance(self.batch_size, bool):
            raise CandidateDslSecurityError("batch_size is locked to 1")
        if self.do_sample is not False:
            raise CandidateDslSecurityError("do_sample is locked to false")
        if self.num_beams != 1 or isinstance(self.num_beams, bool):
            raise CandidateDslSecurityError("num_beams is locked to 1")
        if self.num_return_sequences != 1 or isinstance(
            self.num_return_sequences, bool
        ):
            raise CandidateDslSecurityError(
                "num_return_sequences is locked to 1"
            )
        if self.use_cache is not True:
            raise CandidateDslSecurityError("use_cache is locked to true")

    @property
    def model_revision(self) -> str:
        return MODEL_METADATA[self.model_id]["revision"]

    @property
    def model_directory(self) -> str:
        return MODEL_METADATA[self.model_id]["directory"]

    @property
    def weight_sha256(self) -> str:
        digest = MODEL_METADATA[self.model_id]["weight_sha256"]
        if _HEX_64.fullmatch(digest) is None:
            raise AssertionError("invalid built-in model digest")
        return digest

    @property
    def generation_config(self) -> dict[str, Any]:
        return {
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "max_new_tokens": self.max_new_tokens,
            "use_cache": True,
        }

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def render_prompt(self, question: str) -> str:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be nonempty text")
        if len(question) > 8192 or "\x00" in question:
            raise ValueError("question is too long or contains NUL")
        return self.prompt_template.replace("{question}", question.strip())


def _literal_mapping(source: str) -> Mapping[str, Any]:
    if not isinstance(source, str) or not source.strip():
        raise CandidateDslFormatError("candidate source must be nonempty text")
    if len(source.encode("utf-8")) > 32768:
        raise CandidateDslSecurityError("candidate source exceeds 32768 bytes")
    if FENCE in source:
        raise CandidateDslFormatError("markdown fences are not accepted")
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, RecursionError, ValueError) as exc:
        raise CandidateDslFormatError(
            "candidate source is not valid syntax"
        ) from exc
    if sum(1 for _ in ast.walk(tree)) > 256:
        raise CandidateDslSecurityError("candidate syntax tree is too large")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        raise CandidateDslFormatError(
            "candidate must contain exactly one assignment"
        )
    assignment = tree.body[0]
    if (
        len(assignment.targets) != 1
        or not isinstance(assignment.targets[0], ast.Name)
        or assignment.targets[0].id != "CANDIDATE"
    ):
        raise CandidateDslFormatError("assignment target must be CANDIDATE")
    if not isinstance(assignment.value, ast.Dict):
        raise CandidateDslSecurityError(
            "CANDIDATE must be a literal dictionary"
        )
    keys: list[str] = []
    for node in assignment.value.keys:
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            raise CandidateDslSecurityError(
                "CANDIDATE keys must be literal strings"
            )
        keys.append(node.value)
    if len(keys) != len(set(keys)):
        raise CandidateDslSecurityError("CANDIDATE contains duplicate keys")
    try:
        value = ast.literal_eval(assignment.value)
    except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
        raise CandidateDslSecurityError(
            "CANDIDATE must be literal data"
        ) from exc
    if not isinstance(value, dict):
        raise CandidateDslSecurityError(
            "CANDIDATE must evaluate to a mapping"
        )
    return value


def canonical_candidate_source(spec: ProgramSpec) -> str:
    return "CANDIDATE = " + repr(spec.to_mapping()) + "\n"


def parse_candidate_dsl(
    source: str,
    *,
    expected_model_id: str | None = None,
) -> ProgramSpec:
    try:
        value = _literal_mapping(source)
    except CandidateDslFormatError as strict_error:
        recovered = recover_fenced_candidate(source)
        if recovered is None:
            raise
        spec = parse_candidate_dsl(
            recovered,
            expected_model_id=expected_model_id,
        )
        raise RecoverableOuterFormatError(
            str(strict_error),
            canonical_candidate_source(spec),
        ) from strict_error
    if set(value) != _ALLOWED_FIELDS:
        missing = sorted(_ALLOWED_FIELDS - set(value))
        extra = sorted(set(value) - _ALLOWED_FIELDS)
        raise CandidateDslSecurityError(
            f"CANDIDATE fields are not exact; missing={missing}, extra={extra}"
        )
    if not all(isinstance(key, str) for key in value):
        raise CandidateDslSecurityError("CANDIDATE keys must be strings")
    try:
        spec = ProgramSpec(**value)
    except TypeError as exc:
        raise CandidateDslSecurityError(
            "CANDIDATE field types are invalid"
        ) from exc
    if expected_model_id is not None and spec.model_id != expected_model_id:
        raise CandidateDslSecurityError(
            "mutation attempted to cross the population model boundary"
        )
    return spec


def recover_fenced_candidate(source: str) -> str | None:
    """Recover one complete fenced assignment; never truncation or safety."""

    if not isinstance(source, str) or source.count(FENCE) != 2:
        return None
    escaped = re.escape(FENCE)
    match = re.fullmatch(
        rf"\s*{escaped}(?:python)?\s*\n"
        rf"(.*?)\n{escaped}\s*",
        source,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return None
    inner = match.group(1).strip() + "\n"
    if FENCE in inner:
        return None
    try:
        _literal_mapping(inner)
    except CandidateDslError:
        return None
    return inner


def seed_spec(model_id: str) -> ProgramSpec:
    if model_id == INSTRUCT_MODEL_ID:
        return ProgramSpec(
            model_id=model_id,
            system_instruction="",
            prompt_template=(
                "Answer the chart question concisely. Return only the answer "
                "enclosed in <final> and </final>; do not add an explanation. "
                "Question: {question}"
            ),
            reasoning_mode="direct",
            max_new_tokens=48,
            min_pixels=DEFAULT_MIN_PIXELS,
            max_pixels=DEFAULT_MAX_PIXELS,
            preprocessing_option="qwen_default",
            extraction_strategy="final_tag",
            postprocessing_strategy="strip",
            dtype="bfloat16",
            attention="sdpa",
            batch_size=1,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            use_cache=True,
        )
    if model_id == THINKING_MODEL_ID:
        return ProgramSpec(
            model_id=model_id,
            system_instruction="",
            prompt_template=(
                "Analyze the chart carefully using the model's native reasoning. "
                "Finish with the concise answer enclosed in <final> and </final>. "
                "Question: {question}"
            ),
            reasoning_mode="thinking",
            max_new_tokens=384,
            min_pixels=DEFAULT_MIN_PIXELS,
            max_pixels=DEFAULT_MAX_PIXELS,
            preprocessing_option="qwen_default",
            extraction_strategy="final_tag",
            postprocessing_strategy="strip",
            dtype="bfloat16",
            attention="sdpa",
            batch_size=1,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            use_cache=True,
        )
    raise CandidateDslSecurityError("seed model is not approved")


def mutation_schema_text(model_id: str) -> str:
    if model_id not in MODEL_IDS:
        raise ValueError("unsupported mutation population model")
    token_range = "24..128" if model_id == INSTRUCT_MODEL_ID else "128..512"
    return (
        "Output exactly one Python assignment CANDIDATE = {...} and nothing else. "
        f"model_id is fixed to {model_id!r}. Required exact keys: "
        + ", ".join(sorted(_ALLOWED_FIELDS))
        + ". system_instruction and prompt_template are general chart reasoning "
        "instructions; prompt_template contains exactly one {question}. "
        f"reasoning_mode is fixed to "
        f"{'direct' if model_id == INSTRUCT_MODEL_ID else 'thinking'!r}. "
        "max_new_tokens is an integer in "
        f"{token_range}. min_pixels choices are {MIN_PIXEL_CHOICES}; max_pixels "
        f"choices are {MAX_PIXEL_CHOICES}. preprocessing_option choices are "
        f"{PREPROCESSING_OPTIONS}; qwen_default requires min_pixels="
        f"{DEFAULT_MIN_PIXELS} and max_pixels={DEFAULT_MAX_PIXELS}. "
        "Non-default pixel bounds require non-default preprocessing. "
        "extraction_strategy choices are "
        f"{EXTRACTION_STRATEGIES}. postprocessing_strategy choices are "
        f"{POSTPROCESSING_STRATEGIES}. dtype='bfloat16', attention='sdpa', "
        "batch_size=1, do_sample=False, num_beams=1, "
        "num_return_sequences=1, use_cache=True. Never use sample IDs, query "
        "keys, figure IDs, answers, paths, imports, calls, attributes, shell, "
        "network, or dataset-specific conditions."
    )
