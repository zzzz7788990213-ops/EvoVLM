"""Offline local Qwen mutation provider with a strict declarative output contract."""

from __future__ import annotations

import hashlib
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .candidate_runtime import verify_model_snapshot
from .dsl import (
    INSTRUCT_MODEL_ID,
    MODEL_IDS,
    CandidateDslError,
    CandidateDslFormatError,
    CandidateDslSecurityError,
    RecoverableOuterFormatError,
    canonical_candidate_source,
    mutation_schema_text,
    parse_candidate_dsl,
    seed_spec,
)
from .process_safety import (
    ChildTimeout,
    FrameError,
    atomic_create_json,
    decode_frame,
    encode_frame,
    install_network_guard,
    read_frame,
    run_trusted_child,
    write_frame,
)
from .renderer import (
    render_candidate_program,
    rendered_program_diff,
    rendered_program_sha256,
    validate_rendered_program,
)


PROVIDER_NAME = (
    "LocalQwenMutationProvider/"
    "Qwen3-VL-2B-Instruct@78448d793a7eb2f7a987a1da76d464384aa1becd/"
    "greedy-bfloat16-sdpa-offline"
)
PROVIDER_VERSION = "phase2-local-qwen-mutation-v3"
MAX_MUTATION_OUTPUT_TOKENS = 1024
MAX_REQUESTS_PER_GENERATION = 4
MUTATION_CALL_FIELDS = {
    "attempt",
    "input_token_count",
    "output_token_count",
    "latency_seconds",
    "finish_reason",
    "do_sample",
    "num_beams",
    "num_return_sequences",
    "max_new_tokens",
    "text_only",
    "output_sha256",
    "prompt_sha256",
    "raw_completion_utf8",
    "raw_completion_sha256",
}
MAX_RAW_COMPLETION_BYTES = 65536


class LocalMutationError(RuntimeError):
    pass


@dataclass(frozen=True)
class MutationRequest:
    request_id: str
    generation: int
    child_index: int
    parent_id: str
    model_id: str
    parent_dsl: str
    parent_program_source: str
    parent_program_sha256: str
    parent_metrics: Mapping[str, object]
    run_manifest_sha256: str = "0" * 64
    execution_class: str = "synthetic_test"
    population_program_sha256s: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("request_id", self.request_id),
            ("parent_id", self.parent_id),
        ):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 128
                or not all(character.isalnum() or character in "-_" for character in value)
            ):
                raise ValueError(f"{name} is not a safe identifier")
        if self.model_id not in MODEL_IDS:
            raise ValueError("mutation population model is unsupported")
        if self.generation not in {1, 2}:
            raise ValueError("generation must be 1 or 2")
        population = tuple(self.population_program_sha256s)
        if not population:
            population = (self.parent_program_sha256,)
        if self.parent_program_sha256 not in population:
            raise ValueError("mutation population omits its parent")
        if (
            len(population) > 32
            or len(population) != len(set(population))
            or tuple(sorted(population)) != population
        ):
            raise ValueError("mutation population hashes are not canonical")
        for digest in population:
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("mutation population hash is invalid")
        object.__setattr__(
            self, "population_program_sha256s", population
        )
        if isinstance(self.child_index, bool) or not 0 <= self.child_index < 4:
            raise ValueError("child_index is invalid")
        parent_spec = parse_candidate_dsl(
            self.parent_dsl,
            expected_model_id=self.model_id,
        )
        if canonical_candidate_source(parent_spec) != self.parent_dsl:
            raise ValueError("parent DSL is not canonical")
        validate_rendered_program(
            self.parent_program_source,
            expected_model_id=self.model_id,
        )
        actual = rendered_program_sha256(self.parent_program_source)
        if actual != self.parent_program_sha256:
            raise ValueError("parent rendered program hash mismatch")
        _validate_public_metrics(self.parent_metrics)
        if (
            not isinstance(self.run_manifest_sha256, str)
            or len(self.run_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.run_manifest_sha256
            )
        ):
            raise ValueError("mutation run manifest SHA-256 is invalid")
        if self.execution_class not in {"production_evolution", "synthetic_test"}:
            raise ValueError("mutation execution class is invalid")

    def to_mapping(self) -> dict[str, object]:
        value = asdict(self)
        value["parent_metrics"] = dict(self.parent_metrics)
        return value


@dataclass(frozen=True)
class MutationOutcome:
    request_id: str
    status: str
    error_type: str | None
    error_message: str | None
    canonical_dsl: str | None
    rendered_source: str | None
    rendered_sha256: str | None
    repair_used: bool
    calls: tuple[Mapping[str, object], ...]
    proposals: tuple[Mapping[str, object], ...] = ()

    def to_mapping(self) -> dict[str, object]:
        value = asdict(self)
        value["calls"] = [dict(item) for item in self.calls]
        value["proposals"] = [dict(item) for item in self.proposals]
        return value


def _validate_public_metrics(metrics: Mapping[str, object]) -> None:
    if not isinstance(metrics, Mapping):
        raise ValueError("parent metrics must be a mapping")
    allowed = {"accuracy", "errors", "mean_latency_seconds", "stage", "samples"}
    if set(metrics) - allowed:
        raise ValueError("parent metrics contain private or unapproved fields")
    for key, value in metrics.items():
        if key == "stage":
            if value not in {"seed", "single", "search12", "selection32"}:
                raise ValueError("invalid public metric stage")
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("public metric is not numeric")
        elif not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError("public metric is not finite and nonnegative")


_FOCUS_FIELDS: dict[str, tuple[str, ...]] = {
    "prompt_template": ("prompt_template",),
    "token_budget_and_extraction": ("max_new_tokens", "extraction_strategy"),
    "preprocessing_and_pixel_bounds": (
        "preprocessing_option",
        "min_pixels",
        "max_pixels",
    ),
    "answer_extraction": ("extraction_strategy", "postprocessing_strategy"),
    "system_instruction": ("system_instruction",),
    "token_budget_and_postprocessing": (
        "max_new_tokens",
        "postprocessing_strategy",
    ),
}


def mutation_focus(request: MutationRequest) -> str:
    """Deterministic, preregistered behavioral focus for each child slot."""

    focuses = {
        1: ("prompt_template", "token_budget_and_extraction"),
        2: (
            "preprocessing_and_pixel_bounds",
            "answer_extraction",
            "system_instruction",
            "token_budget_and_postprocessing",
        ),
    }
    return focuses[request.generation][request.child_index]


def mutation_focus_fields(request: MutationRequest) -> tuple[str, ...]:
    return _FOCUS_FIELDS[mutation_focus(request)]


def _retry_constraints(request: MutationRequest) -> str:
    return (
        "Change at least one and only fields in this exact allowed focus set: "
        + repr(mutation_focus_fields(request))
        + ". Every field outside this set MUST remain exactly equal to the parent. "
    )


def _duplicate_retry_prompt(request: MutationRequest) -> str:
    return (
        "The primary proposal duplicated the parent after canonical rendering. "
        "This is the one and only duplicate correction. "
        + _retry_constraints(request)
        + "Do not merely reformat, reorder keys, or change an ineffective field.\n\n"
        + _mutation_prompt(request)
    )


def _off_focus_retry_prompt(request: MutationRequest) -> str:
    return (
        "The primary proposal was legal and different, but it did not change the "
        f"preregistered focus {mutation_focus(request)!r}. This is the one and "
        "only off-focus correction. "
        + _retry_constraints(request)
        + "For prompt_template focus this means prompt_template must change and "
        "all other fields must exactly equal the parent.\n\n"
        + _mutation_prompt(request)
    )


def _mutation_prompt(request: MutationRequest) -> str:
    return (
        "You mutate a declarative vision-language inference configuration. "
        "Improve expected chart-question accuracy and efficiency without using "
        "sample-specific knowledge. The rendered program MUST differ from the "
        "parent by changing at least one runtime-effective field. Make one coherent "
        "change or a small compatible set of changes. The preregistered focus for "
        f"this child is {mutation_focus(request)!r}.\n\nSTRICT SCHEMA:\n"
        + mutation_schema_text(request.model_id)
        + "\n\nPARENT CANONICAL DSL:\n"
        + request.parent_dsl
        + "\nPARENT RENDERED PROGRAM (for review only; output remains DSL):\n"
        + request.parent_program_source
        + "\nAGGREGATE PARENT METRICS ONLY:\n"
        + repr(dict(request.parent_metrics))
        + f"\nGENERATION={request.generation}; CHILD_INDEX={request.child_index}\n"
    )


def _repair_prompt(canonical_source: str) -> str:
    return (
        "Your previous response had only an outer formatting violation. "
        "Reproduce the exact assignment below byte-for-byte, with no markdown, "
        "commentary, or semantic change.\n\n" + canonical_source
    )


def _classify_completion(
    *,
    completion_ids: list[int],
    eos_token_ids: set[int],
    cap: int,
) -> str:
    if completion_ids and completion_ids[-1] in eos_token_ids:
        return "eos"
    if len(completion_ids) >= cap:
        return "length"
    return "other"


def _one_generation_call(
    *,
    model: Any,
    processor: Any,
    torch: Any,
    prompt: str,
    attempt: str,
) -> tuple[str, dict[str, object]]:
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Return only a strict declarative CANDIDATE assignment. "
                        "Never use tools, files, network, code execution, or sample data."
                    ),
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[rendered],
        padding=True,
        return_tensors="pt",
    )
    forbidden_modal_keys = {
        key
        for key in inputs
        if any(marker in str(key).casefold() for marker in ("pixel", "image", "video"))
    }
    if forbidden_modal_keys:
        raise LocalMutationError("mutation processor produced non-text modal inputs")
    inputs = inputs.to("cuda:0")
    input_length = int(inputs.input_ids.shape[-1])
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_MUTATION_OUTPUT_TOKENS,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            use_cache=True,
        )
    torch.cuda.synchronize()
    finished = time.perf_counter()
    completion = generated[0, input_length:]
    completion_ids = [int(item) for item in completion.tolist()]
    values = processor.batch_decode(
        [completion],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if len(values) != 1 or not isinstance(values[0], str):
        raise LocalMutationError("mutation decoder did not return exactly one string")
    encoded_completion = values[0].encode("utf-8")
    if len(encoded_completion) > MAX_RAW_COMPLETION_BYTES:
        raise LocalMutationError("mutation raw completion exceeds evidence bound")
    eos_value = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(eos_value, int):
        eos_ids = {eos_value}
    elif isinstance(eos_value, (list, tuple, set)):
        eos_ids = {int(item) for item in eos_value}
    else:
        eos_ids = set()
    finish_reason = _classify_completion(
        completion_ids=completion_ids,
        eos_token_ids=eos_ids,
        cap=MAX_MUTATION_OUTPUT_TOKENS,
    )
    record = {
        "attempt": attempt,
        "input_token_count": input_length,
        "output_token_count": len(completion_ids),
        "latency_seconds": finished - started,
        "finish_reason": finish_reason,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "max_new_tokens": MAX_MUTATION_OUTPUT_TOKENS,
        "text_only": True,
        "output_sha256": hashlib.sha256(encoded_completion).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "raw_completion_utf8": values[0],
        "raw_completion_sha256": hashlib.sha256(encoded_completion).hexdigest(),
    }
    return values[0], record


def _process_generated_texts(
    *,
    request: MutationRequest,
    primary_text: str,
    primary_call: Mapping[str, object],
    repair_text: str | None = None,
    repair_call: Mapping[str, object] | None = None,
) -> MutationOutcome:
    calls = [dict(primary_call)]
    if primary_call.get("finish_reason") == "length":
        return MutationOutcome(
            request.request_id,
            "rejected",
            "truncated",
            "primary mutation output reached the generation cap",
            None,
            None,
            None,
            False,
            tuple(calls),
        )
    recovered: str | None = None
    try:
        spec = parse_candidate_dsl(
            primary_text,
            expected_model_id=request.model_id,
        )
    except RecoverableOuterFormatError as exc:
        recovered = exc.canonical_source
    except CandidateDslSecurityError as exc:
        return MutationOutcome(
            request.request_id,
            "rejected",
            "security",
            str(exc),
            None,
            None,
            None,
            False,
            tuple(calls),
        )
    except CandidateDslFormatError as exc:
        return MutationOutcome(
            request.request_id,
            "rejected",
            "format_not_recoverable",
            str(exc),
            None,
            None,
            None,
            False,
            tuple(calls),
        )
    else:
        canonical = canonical_candidate_source(spec)
        source = render_candidate_program(spec)
        return MutationOutcome(
            request.request_id,
            "accepted",
            None,
            None,
            canonical,
            source,
            rendered_program_sha256(source),
            False,
            tuple(calls),
        )

    if repair_text is None or repair_call is None:
        return MutationOutcome(
            request.request_id,
            "needs_format_repair",
            "outer_format",
            "one complete fenced assignment requires an exact format repair",
            recovered,
            None,
            None,
            False,
            tuple(calls),
        )
    calls.append(dict(repair_call))
    if repair_call.get("finish_reason") == "length":
        return MutationOutcome(
            request.request_id,
            "rejected",
            "repair_truncated",
            "format repair reached the generation cap",
            None,
            None,
            None,
            True,
            tuple(calls),
        )
    try:
        repaired_spec = parse_candidate_dsl(
            repair_text,
            expected_model_id=request.model_id,
        )
    except CandidateDslError as exc:
        return MutationOutcome(
            request.request_id,
            "rejected",
            "repair_invalid",
            str(exc),
            None,
            None,
            None,
            True,
            tuple(calls),
        )
    repaired = canonical_candidate_source(repaired_spec)
    if repaired != recovered:
        return MutationOutcome(
            request.request_id,
            "rejected",
            "repair_semantic_change",
            "format repair changed declarative semantics",
            None,
            None,
            None,
            True,
            tuple(calls),
        )
    source = render_candidate_program(repaired_spec)
    return MutationOutcome(
        request.request_id,
        "accepted",
        None,
        None,
        repaired,
        source,
        rendered_program_sha256(source),
        True,
        tuple(calls),
    )


def _assert_child_gpu(torch: Any) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise LocalMutationError("mutation child requires CUDA_VISIBLE_DEVICES=0")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise LocalMutationError("mutation child requires HF_HUB_OFFLINE=1")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise LocalMutationError("mutation child requires TRANSFORMERS_OFFLINE=1")
    if int(torch.cuda.device_count()) != 1:
        raise LocalMutationError("mutation child must see exactly one CUDA device")
    if int(torch.cuda.current_device()) != 0:
        raise LocalMutationError("mutation child must use logical cuda:0")


def _load_mutation_model() -> tuple[Any, Any, Any, dict[str, object], dict[str, object]]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    _assert_child_gpu(torch)
    gpu_scope = {
        "physical_gpu_index": 0,
        "cuda_visible_devices": "0",
        "visible_device_count": 1,
        "logical_device": "cuda:0",
    }
    spec = seed_spec(INSTRUCT_MODEL_ID)
    evidence = verify_model_snapshot(spec)
    path = Path(str(evidence["snapshot_path"]))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()
    if str(next(model.parameters()).device) != "cuda:0":
        raise LocalMutationError("mutation model is not on logical cuda:0")
    processor = AutoProcessor.from_pretrained(
        path,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    return model, processor, torch, evidence, gpu_scope


def _strict_requests(document: Mapping[str, object]) -> list[MutationRequest]:
    if set(document) != {"kind", "provider_version", "requests"}:
        raise LocalMutationError("mutation child request fields are not exact")
    if document.get("kind") != "local_mutation_generation":
        raise LocalMutationError("mutation child request kind is invalid")
    if document.get("provider_version") != PROVIDER_VERSION:
        raise LocalMutationError("mutation child provider version mismatch")
    raw_requests = document.get("requests")
    if not isinstance(raw_requests, list) or not 1 <= len(raw_requests) <= MAX_REQUESTS_PER_GENERATION:
        raise LocalMutationError("mutation request count is invalid")
    requests: list[MutationRequest] = []
    for raw in raw_requests:
        if not isinstance(raw, dict):
            raise LocalMutationError("mutation request is not an object")
        expected = set(MutationRequest.__dataclass_fields__)
        if set(raw) != expected:
            raise LocalMutationError("mutation request fields are not exact")
        requests.append(MutationRequest(**raw))
    identifiers = [item.request_id for item in requests]
    if len(identifiers) != len(set(identifiers)):
        raise LocalMutationError("duplicate mutation request_id")
    model_ids = {item.model_id for item in requests}
    generations = {item.generation for item in requests}
    if len(model_ids) != 1 or len(generations) != 1:
        raise LocalMutationError("one child process must serve one population generation")
    return requests


def _focus_changed(
    request: MutationRequest,
    canonical_dsl: str,
) -> bool:
    parent = parse_candidate_dsl(
        request.parent_dsl,
        expected_model_id=request.model_id,
    )
    child = parse_candidate_dsl(
        canonical_dsl,
        expected_model_id=request.model_id,
    )
    fields = mutation_focus_fields(request)
    return any(
        getattr(parent, field) != getattr(child, field)
        for field in fields
    )


def _changed_fields(request: MutationRequest, canonical_dsl: str) -> tuple[str, ...]:
    parent = parse_candidate_dsl(
        request.parent_dsl,
        expected_model_id=request.model_id,
    )
    child = parse_candidate_dsl(
        canonical_dsl,
        expected_model_id=request.model_id,
    )
    return tuple(
        sorted(
            field
            for field in parent.__dataclass_fields__
            if getattr(parent, field) != getattr(child, field)
        )
    )


def _focus_only_change(request: MutationRequest, canonical_dsl: str) -> bool:
    changed = set(_changed_fields(request, canonical_dsl))
    allowed = set(mutation_focus_fields(request))
    return bool(changed) and changed <= allowed


def _proposal_evidence(
    request: MutationRequest,
    *,
    attempt: str,
    raw_text: str,
) -> dict[str, object]:
    raw_sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    canonical: str | None = None
    source: str | None = None
    source_sha: str | None = None
    diff: str | None = None
    changed: tuple[str, ...] = ()
    focus_differences: dict[str, object] = {}
    adherent: bool | None = None
    rejection: str | None = None
    try:
        spec = parse_candidate_dsl(raw_text, expected_model_id=request.model_id)
    except RecoverableOuterFormatError as exc:
        canonical = exc.canonical_source
        rejection = "outer_format"
    except CandidateDslError as exc:
        rejection = f"{type(exc).__name__}: {str(exc)[:1024]}"
    else:
        canonical = canonical_candidate_source(spec)
    if canonical is not None:
        spec = parse_candidate_dsl(canonical, expected_model_id=request.model_id)
        source = render_candidate_program(spec)
        source_sha = rendered_program_sha256(source)
        diff = rendered_program_diff(request.parent_program_source, source)
        changed = _changed_fields(request, canonical)
        parent = parse_candidate_dsl(
            request.parent_dsl,
            expected_model_id=request.model_id,
        )
        for field in mutation_focus_fields(request):
            before = getattr(parent, field)
            after = getattr(spec, field)
            if before != after:
                focus_differences[field] = {
                    "parent": before,
                    "proposal": after,
                }
        adherent = _focus_changed(request, canonical)
        if source_sha == request.parent_program_sha256:
            rejection = "duplicate_parent"
        elif not adherent:
            rejection = "focus_not_changed"
    return {
        "attempt": attempt,
        "request_id": request.request_id,
        "generation": request.generation,
        "child_index": request.child_index,
        "parent_program_sha256": request.parent_program_sha256,
        "raw_completion_utf8": raw_text,
        "raw_completion_sha256": raw_sha,
        "canonical_dsl": canonical,
        "rendered_source": source,
        "rendered_sha256": source_sha,
        "parent_to_proposal_diff": diff,
        "changed_fields": list(changed),
        "focus": mutation_focus(request),
        "focus_fields": list(mutation_focus_fields(request)),
        "focus_field_differences": focus_differences,
        "focus_adherent": adherent,
        "rejection_reason": rejection,
    }


def _attach_proposals(
    outcome: MutationOutcome,
    request: MutationRequest,
    proposals: Sequence[tuple[str, str]],
) -> MutationOutcome:
    return replace(
        outcome,
        proposals=tuple(
            _proposal_evidence(request, attempt=attempt, raw_text=text)
            for attempt, text in proposals
        ),
    )


def _rejected_outcome(
    outcome: MutationOutcome,
    error_type: str,
    error_message: str,
) -> MutationOutcome:
    return MutationOutcome(
        outcome.request_id,
        "rejected",
        error_type,
        error_message,
        None,
        None,
        None,
        outcome.repair_used,
        outcome.calls,
    )


def _mutate_one_request(
    *,
    model: Any,
    processor: Any,
    torch: Any,
    request: MutationRequest,
) -> MutationOutcome:
    primary_text, primary_call = _one_generation_call(
        model=model,
        processor=processor,
        torch=torch,
        prompt=_mutation_prompt(request),
        attempt="primary",
    )
    proposals: list[tuple[str, str]] = [("primary", primary_text)]
    outcome = _process_generated_texts(
        request=request,
        primary_text=primary_text,
        primary_call=primary_call,
    )
    if outcome.status == "needs_format_repair":
        assert outcome.canonical_dsl is not None
        repair_text, repair_call = _one_generation_call(
            model=model,
            processor=processor,
            torch=torch,
            prompt=_repair_prompt(outcome.canonical_dsl),
            attempt="format_repair",
        )
        outcome = _process_generated_texts(
            request=request,
            primary_text=primary_text,
            primary_call=primary_call,
            repair_text=repair_text,
            repair_call=repair_call,
        )
        proposals.append(("format_repair", repair_text))
    if (
        outcome.status == "accepted"
        and outcome.rendered_sha256 == request.parent_program_sha256
    ):
        if outcome.repair_used:
            return _attach_proposals(
                _rejected_outcome(
                    outcome,
                    "duplicate_parent",
                    "format repair consumed the only correction call",
                ),
                request,
                proposals,
            )
        retry_text, retry_call = _one_generation_call(
            model=model,
            processor=processor,
            torch=torch,
            prompt=_duplicate_retry_prompt(request),
            attempt="duplicate_retry",
        )
        proposals.append(("duplicate_retry", retry_text))
        retried = _process_generated_texts(
            request=request,
            primary_text=retry_text,
            primary_call=retry_call,
        )
        combined_calls = (
            dict(primary_call),
            dict(retry_call),
        )
        if (
            retried.status == "accepted"
            and retried.rendered_sha256 != request.parent_program_sha256
            and retried.canonical_dsl is not None
            and _focus_only_change(request, retried.canonical_dsl)
        ):
            return _attach_proposals(
                replace(retried, calls=combined_calls),
                request,
                proposals,
            )
        return _attach_proposals(
            MutationOutcome(
                request.request_id,
                "rejected",
                "duplicate_retry_failed",
                "duplicate retry was invalid, off-focus, or parent-identical",
                None,
                None,
                None,
                False,
                combined_calls,
            ),
            request,
            proposals,
        )
    if (
        outcome.status == "accepted"
        and outcome.canonical_dsl is not None
        and not _focus_changed(request, outcome.canonical_dsl)
    ):
        if outcome.repair_used:
            return _attach_proposals(
                _rejected_outcome(
                    outcome,
                    "focus_not_changed",
                    "format repair consumed the only correction call",
                ),
                request,
                proposals,
            )
        retry_text, retry_call = _one_generation_call(
            model=model,
            processor=processor,
            torch=torch,
            prompt=_off_focus_retry_prompt(request),
            attempt="off_focus_retry",
        )
        proposals.append(("off_focus_retry", retry_text))
        retried = _process_generated_texts(
            request=request,
            primary_text=retry_text,
            primary_call=retry_call,
        )
        combined_calls = (dict(primary_call), dict(retry_call))
        if (
            retried.status == "accepted"
            and retried.rendered_sha256 != request.parent_program_sha256
            and retried.canonical_dsl is not None
            and _focus_only_change(request, retried.canonical_dsl)
        ):
            return _attach_proposals(
                replace(retried, calls=combined_calls),
                request,
                proposals,
            )
        return _attach_proposals(
            MutationOutcome(
                request.request_id,
                "rejected",
                "off_focus_retry_failed",
                "off-focus retry was invalid, duplicate, or changed non-focus fields",
                None,
                None,
                None,
                False,
                combined_calls,
            ),
            request,
            proposals,
        )
    return _attach_proposals(outcome, request, proposals)


def _child_mutation_main() -> int:
    network = install_network_guard()
    try:
        document = read_frame(sys.stdin.buffer)
        if sys.stdin.buffer.read(1) != b"":
            raise FrameError("trailing mutation child input")
        requests = _strict_requests(document)
        model, processor, torch, evidence, gpu_scope = (
            _load_mutation_model()
        )
        outcomes = [
            _mutate_one_request(
                model=model,
                processor=processor,
                torch=torch,
                request=request,
            )
            for request in requests
        ]
        if network.count != 0:
            raise LocalMutationError(
                "mutation child attempted network access"
            )
        reply = {
            "kind": "local_mutation_generation_result",
            "provider_version": PROVIDER_VERSION,
            "provider_name": PROVIDER_NAME,
            "network_attempts": network.count,
            "model_evidence": evidence,
            "gpu_scope": gpu_scope,
            "outcomes": [
                item.to_mapping() for item in outcomes
            ],
        }
        write_frame(sys.stdout.buffer, reply)
        return 0
    except BaseException as exc:
        error = {
            "kind": "local_mutation_generation_error",
            "provider_version": PROVIDER_VERSION,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:2048],
            "network_attempts": network.count,
        }
        try:
            write_frame(sys.stdout.buffer, error)
        except BaseException:
            pass
        return 1


def _validate_child_reply(
    reply: Mapping[str, object],
    requests: Sequence[MutationRequest],
) -> list[MutationOutcome]:
    expected = {
        "kind",
        "provider_version",
        "provider_name",
        "network_attempts",
        "model_evidence",
        "gpu_scope",
        "outcomes",
    }
    if set(reply) != expected:
        raise LocalMutationError("mutation child reply fields are not exact")
    if reply.get("kind") != "local_mutation_generation_result":
        raise LocalMutationError("mutation child returned an error")
    if reply.get("provider_version") != PROVIDER_VERSION:
        raise LocalMutationError("mutation child provider version mismatch")
    if reply.get("provider_name") != PROVIDER_NAME:
        raise LocalMutationError("mutation child provider identity mismatch")
    if reply.get("network_attempts") != 0:
        raise LocalMutationError("mutation child network attempts are nonzero")
    if reply.get("gpu_scope") != {
        "physical_gpu_index": 0,
        "cuda_visible_devices": "0",
        "visible_device_count": 1,
        "logical_device": "cuda:0",
    }:
        raise LocalMutationError("mutation child GPU scope is invalid")
    evidence = reply.get("model_evidence")
    expected_seed = seed_spec(INSTRUCT_MODEL_ID)
    if (
        not isinstance(evidence, dict)
        or evidence.get("model_id") != INSTRUCT_MODEL_ID
        or evidence.get("revision") != expected_seed.model_revision
        or evidence.get("weight_sha256") != expected_seed.weight_sha256
    ):
        raise LocalMutationError("mutation model provenance is invalid")
    raw_outcomes = reply.get("outcomes")
    if not isinstance(raw_outcomes, list) or len(raw_outcomes) != len(requests):
        raise LocalMutationError("mutation outcome cardinality mismatch")
    outcomes: list[MutationOutcome] = []
    expected_fields = set(MutationOutcome.__dataclass_fields__)
    for raw, request in zip(raw_outcomes, requests, strict=True):
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise LocalMutationError("mutation outcome fields are not exact")
        raw = dict(raw)
        calls = raw.pop("calls")
        proposals = raw.pop("proposals")
        if not isinstance(calls, list) or not 1 <= len(calls) <= 2:
            raise LocalMutationError("mutation call log cardinality is invalid")
        for call in calls:
            if not isinstance(call, dict):
                raise LocalMutationError("mutation call log is invalid")
            if set(call) != MUTATION_CALL_FIELDS:
                raise LocalMutationError("mutation call log fields are not exact")
            for count_key in ("input_token_count", "output_token_count"):
                value = call.get(count_key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise LocalMutationError("mutation token count is invalid")
            latency = call.get("latency_seconds")
            if isinstance(latency, bool) or not isinstance(latency, (int, float)):
                raise LocalMutationError("mutation latency is invalid")
            if not math.isfinite(float(latency)) or float(latency) < 0:
                raise LocalMutationError("mutation latency is non-finite")
            if call.get("finish_reason") not in {"eos", "length", "other"}:
                raise LocalMutationError("mutation finish reason is invalid")
            if (
                call.get("do_sample") is not False
                or call.get("num_beams") != 1
                or call.get("num_return_sequences") != 1
                or call.get("max_new_tokens") != MAX_MUTATION_OUTPUT_TOKENS
                or call.get("text_only") is not True
            ):
                raise LocalMutationError("mutation generation locks are invalid")
            output_hash = call.get("output_sha256")
            if (
                not isinstance(output_hash, str)
                or len(output_hash) != 64
                or any(character not in "0123456789abcdef" for character in output_hash)
            ):
                raise LocalMutationError("mutation output hash is invalid")
            raw_text = call.get("raw_completion_utf8")
            raw_sha = call.get("raw_completion_sha256")
            prompt_sha = call.get("prompt_sha256")
            if (
                not isinstance(raw_text, str)
                or len(raw_text.encode("utf-8")) > MAX_RAW_COMPLETION_BYTES
                or raw_sha != hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
                or call.get("output_sha256") != raw_sha
                or not isinstance(prompt_sha, str)
                or len(prompt_sha) != 64
            ):
                raise LocalMutationError("mutation raw proposal evidence is invalid")
        if not isinstance(proposals, list):
            raise LocalMutationError("mutation proposals are not a list")
        outcome = MutationOutcome(
            calls=tuple(calls),
            proposals=tuple(proposals),
            **raw,
        )
        expected_proposals = tuple(
            _proposal_evidence(
                request,
                attempt=str(call["attempt"]),
                raw_text=str(call["raw_completion_utf8"]),
            )
            for call in calls
        )
        if outcome.proposals != expected_proposals:
            raise LocalMutationError(
                "mutation proposal evidence is not reproducible"
            )
        if outcome.request_id != request.request_id:
            raise LocalMutationError("mutation outcome request_id order mismatch")
        if outcome.status not in {"accepted", "rejected"}:
            raise LocalMutationError("mutation outcome terminal status is invalid")
        attempts = [call["attempt"] for call in calls]
        allowed_attempts = {
            ("primary",),
            ("primary", "format_repair"),
            ("primary", "duplicate_retry"),
            ("primary", "off_focus_retry"),
        }
        if tuple(attempts) not in allowed_attempts:
            raise LocalMutationError("mutation call attempt order is invalid")
        if outcome.repair_used != (attempts == ["primary", "format_repair"]):
            raise LocalMutationError("mutation repair accounting is invalid")
        if outcome.status == "accepted":
            if outcome.error_type is not None or outcome.error_message is not None:
                raise LocalMutationError("accepted mutation contains an error")
        elif (
            outcome.canonical_dsl is not None
            or outcome.rendered_source is not None
            or outcome.rendered_sha256 is not None
            or not isinstance(outcome.error_type, str)
            or not isinstance(outcome.error_message, str)
        ):
            raise LocalMutationError("rejected mutation fields are inconsistent")
        if outcome.status == "accepted":
            if (
                outcome.canonical_dsl is None
                or outcome.rendered_source is None
                or outcome.rendered_sha256 is None
            ):
                raise LocalMutationError("accepted mutation is incomplete")
            spec = parse_candidate_dsl(
                outcome.canonical_dsl,
                expected_model_id=request.model_id,
            )
            if canonical_candidate_source(spec) != outcome.canonical_dsl:
                raise LocalMutationError("accepted mutation DSL is not canonical")
            validate_rendered_program(
                outcome.rendered_source,
                expected_model_id=request.model_id,
            )
            if rendered_program_sha256(outcome.rendered_source) != outcome.rendered_sha256:
                raise LocalMutationError("accepted mutation program hash mismatch")
            if outcome.rendered_sha256 == request.parent_program_sha256:
                raise LocalMutationError("accepted mutation duplicates its parent")
            if not _focus_changed(request, outcome.canonical_dsl):
                raise LocalMutationError(
                    "accepted mutation did not change preregistered focus"
                )
        outcomes.append(outcome)
    return outcomes


class LocalQwenMutationProvider:
    """CPU-side launcher. Torch is imported only in the reaped child process."""

    name = PROVIDER_NAME
    version = PROVIDER_VERSION

    evidence_class = "production_evolution"
    def mutate_generation(
        self,
        requests: Sequence[MutationRequest],
        *,
        artifact_path: Path,
        timeout_seconds: float,
    ) -> list[MutationOutcome]:
        values = list(requests)
        if not 1 <= len(values) <= MAX_REQUESTS_PER_GENERATION:
            raise LocalMutationError("generation mutation request count is invalid")
        manifest_hashes = {item.run_manifest_sha256 for item in values}
        if len(manifest_hashes) != 1:
            raise LocalMutationError("generation requests cross run manifests")
        if any(item.execution_class != "production_evolution" for item in values):
            raise LocalMutationError("real provider rejects synthetic execution")
        run_manifest_sha256 = next(iter(manifest_hashes))
        frame = encode_frame(
            {
                "kind": "local_mutation_generation",
                "provider_version": PROVIDER_VERSION,
                "requests": [item.to_mapping() for item in values],
            }
        )
        try:
            child = run_trusted_child(
                module="evolution.local_provider",
                function="_child_mutation_main",
                input_bytes=frame,
                timeout_seconds=timeout_seconds,
            )
        except ChildTimeout as exc:
            atomic_create_json(
                artifact_path,
                {
                    "status": "failed",
                    "provider_name": PROVIDER_NAME,
                    "provider_version": PROVIDER_VERSION,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "child_reaped": True,
                    "requests": [item.request_id for item in values],
                },
            )
            raise LocalMutationError("mutation child timed out and was reaped") from exc
        if not child.reaped:
            atomic_create_json(
                artifact_path,
                {
                    "status": "failed",
                    "provider_name": PROVIDER_NAME,
                    "provider_version": PROVIDER_VERSION,
                    "error_type": "UnreapedChild",
                    "error_message": "mutation child was not reaped",
                    "child_pid": child.pid,
                    "requests": [item.request_id for item in values],
                },
            )
            raise LocalMutationError("mutation child was not reaped")
        self.last_child_elapsed_seconds = child.elapsed_seconds
        reply: dict[str, object]
        try:
            reply = decode_frame(child.stdout)
        except FrameError as exc:
            atomic_create_json(
                artifact_path,
                {
                    "status": "failed",
                    "provider_name": PROVIDER_NAME,
                    "provider_version": PROVIDER_VERSION,
                    "error_type": type(exc).__name__,
                    "error_message": "mutation child stdout is not one strict frame",
                    "child_pid": child.pid,
                    "child_returncode": child.returncode,
                    "child_reaped": child.reaped,
                    "child_elapsed_seconds": child.elapsed_seconds,
                    "stdout_size_bytes": len(child.stdout),
                    "stdout_sha256": hashlib.sha256(child.stdout).hexdigest(),
                    "stderr_size_bytes": len(child.stderr),
                    "stderr_sha256": hashlib.sha256(child.stderr).hexdigest(),
                    "requests": [item.request_id for item in values],
                },
            )
            raise LocalMutationError("mutation child stdout is not one strict frame") from exc
        if child.returncode != 0:
            atomic_create_json(
                artifact_path,
                {
                    "status": "failed",
                    "provider_name": PROVIDER_NAME,
                    "provider_version": PROVIDER_VERSION,
                    "error_type": str(reply.get("error_type", "unknown")),
                    "error_message": str(reply.get("error_message", "no detail")),
                    "network_attempts": reply.get("network_attempts"),
                    "child_pid": child.pid,
                    "child_returncode": child.returncode,
                    "child_reaped": child.reaped,
                    "child_elapsed_seconds": child.elapsed_seconds,
                    "stderr_size_bytes": len(child.stderr),
                    "stderr_sha256": hashlib.sha256(child.stderr).hexdigest(),
                    "requests": [item.request_id for item in values],
                },
            )
            raise LocalMutationError(
                "mutation child failed: "
                + str(reply.get("error_type", "unknown"))
                + ": "
                + str(reply.get("error_message", "no detail"))
            )
        try:
            outcomes = _validate_child_reply(reply, values)
        except Exception as exc:
            atomic_create_json(
                artifact_path,
                {
                    "status": "failed_validation",
                    "provider_name": PROVIDER_NAME,
                    "provider_version": PROVIDER_VERSION,
                    "execution_class": "production_evolution",
                    "run_manifest_sha256": run_manifest_sha256,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:2048],
                    "child_pid": child.pid,
                    "child_returncode": child.returncode,
                    "child_reaped": child.reaped,
                    "child_elapsed_seconds": child.elapsed_seconds,
                    "child_stderr_size_bytes": len(child.stderr),
                    "child_stderr_sha256": hashlib.sha256(child.stderr).hexdigest(),
                    "requests": [item.to_mapping() for item in values],
                    "raw_child_reply": reply,
                },
            )
            raise
        log = {
            "provider_name": PROVIDER_NAME,
            "provider_version": PROVIDER_VERSION,
            "child_pid": child.pid,
            "child_returncode": child.returncode,
            "child_reaped": child.reaped,
            "child_elapsed_seconds": child.elapsed_seconds,
            "child_shell": child.shell,
            "child_start_new_session": child.start_new_session,
            "child_stderr_size_bytes": len(child.stderr),
            "child_stderr_sha256": hashlib.sha256(child.stderr).hexdigest(),
            "child_stderr_utf8": child.stderr.decode("utf-8", errors="replace")[:65536],
            "network_attempts": reply["network_attempts"],
            "model_evidence": reply["model_evidence"],
            "gpu_scope": reply["gpu_scope"],
            "execution_class": "production_evolution",
            "run_manifest_sha256": run_manifest_sha256,
            "requests": [
                {
                    **item.to_mapping(),
                    "mutation_focus": mutation_focus(item),
                }
                for item in values
            ],
            "outcomes": [item.to_mapping() for item in outcomes],
            "total_calls": sum(len(item.calls) for item in outcomes),
            "total_input_tokens": sum(
                int(call["input_token_count"])
                for item in outcomes
                for call in item.calls
            ),
            "total_output_tokens": sum(
                int(call["output_token_count"])
                for item in outcomes
                for call in item.calls
            ),
        }
        atomic_create_json(artifact_path, log)
        return outcomes


__all__ = [
    "LocalQwenMutationProvider",
    "LocalMutationError",
    "MutationOutcome",
    "MutationRequest",
    "PROVIDER_NAME",
    "PROVIDER_VERSION",
    "mutation_focus",
    "mutation_focus_fields",
]
