"""Catalog-constrained offline Qwen mutation provider."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .candidate_runtime import verify_model_snapshot
from .dsl import (
    INSTRUCT_MODEL_ID,
    canonical_candidate_source,
    parse_candidate_dsl,
    seed_spec,
)
from .local_provider import (
    MutationOutcome,
    MutationRequest,
    mutation_focus,
)
from .mutation_catalog import (
    CATALOG_VERSION,
    MutationCatalogError,
    catalog_hashes,
    choices_for,
    load_frozen_catalog,
    rank_choice_ids,
    select_ranked_choice,
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
    "CatalogConstrainedQwenMutationProvider/"
    "Qwen3-VL-2B-Instruct@78448d793a7eb2f7a987a1da76d464384aa1becd/"
    "single-token-argmax-bfloat16-sdpa-offline"
)
PROVIDER_VERSION = "phase2-local-qwen-mutation-v4"
SELECTION_METHOD = "single-token-label-argmax"
LABEL_TEXTS = (" A", " B", " C", " D", " E", " F", " G", " H")
MAX_REQUESTS_PER_GENERATION = 4
MAX_PROMPT_BYTES = 65536
_SCORE_ROW_FIELDS = {
    "ordinal",
    "label_text",
    "choice_id",
    "token_id",
    "score",
    "score_hex",
    "score_float32_le_hex",
}
_CALL_FIELDS = {
    "attempt",
    "selection_method",
    "input_token_count",
    "output_token_count",
    "latency_seconds",
    "finish_reason",
    "do_sample",
    "num_beams",
    "num_return_sequences",
    "text_only",
    "prompt_utf8",
    "prompt_sha256",
    "rendered_chat_sha256",
    "input_token_ids",
    "input_token_ids_sha256",
    "label_scores",
    "ranking",
}
_PROPOSAL_FIELDS = {
    "request_id",
    "generation",
    "child_index",
    "parent_id",
    "parent_program_sha256",
    "population_program_sha256s",
    "catalog_version",
    "catalog_sha256",
    "catalog_manifest_sha256",
    "focus",
    "option_order",
    "selected_choice_id",
    "duplicate_skips",
    "exact_patch",
    "allowed_changed_fields",
    "changed_fields",
    "parent_dsl",
    "child_dsl",
    "rendered_source",
    "rendered_sha256",
    "parent_to_child_diff",
}


class CatalogProviderError(RuntimeError):
    """The production catalog selector failed closed."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _selection_prompt(
    request: MutationRequest,
    choices: Sequence[Mapping[str, object]],
) -> str:
    if len(choices) > len(LABEL_TEXTS):
        raise CatalogProviderError("catalog focus has too many labels")
    rows = []
    for label, choice in zip(LABEL_TEXTS, choices, strict=False):
        rows.append(
            json.dumps(
                {
                    "label": label.strip(),
                    "choice_id": choice["choice_id"],
                    "description": choice["description"],
                    "patch": choice["patch"],
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    prompt = (
        "Choose one general mutation patch for a chart VLM inference program. "
        "The catalog is frozen and trusted code will apply the selected patch. "
        "Use only aggregate parent metrics; never infer or request sample identity, "
        "answers, files, tools, or network. Rank choices for expected accuracy and "
        "efficiency. Return exactly one allowed label.\n"
        f"MODEL={request.model_id}\n"
        f"FOCUS={mutation_focus(request)}\n"
        f"GENERATION={request.generation}\n"
        f"CHILD_INDEX={request.child_index}\n"
        "PARENT_CANONICAL_DSL:\n"
        + request.parent_dsl
        + "AGGREGATE_PARENT_METRICS:\n"
        + json.dumps(
            dict(request.parent_metrics),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\nFROZEN_OPTIONS_IN_ORDER:\n"
        + "\n".join(rows)
        + "\nALLOWED_LABELS="
        + ",".join(label.strip() for label in LABEL_TEXTS[: len(choices)])
        + "\nFINAL_LABEL:"
    )
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise CatalogProviderError("catalog selection prompt exceeds bound")
    return prompt


def _messages(prompt: str) -> list[dict[str, object]]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Select exactly one label for a frozen, general chart-inference "
                        "field patch. Never use tools, files, network, or sample data."
                    ),
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def _exact_label_token_ids(
    tokenizer: Any,
    rendered_chat: str,
    label_texts: Sequence[str],
) -> list[int]:
    base = tokenizer.encode(rendered_chat, add_special_tokens=False)
    if not isinstance(base, list) or not base:
        raise CatalogProviderError("rendered chat tokenization is invalid")
    values: list[int] = []
    for label in label_texts:
        extended = tokenizer.encode(
            rendered_chat + label,
            add_special_tokens=False,
        )
        if (
            not isinstance(extended, list)
            or len(extended) != len(base) + 1
            or extended[:-1] != base
            or isinstance(extended[-1], bool)
            or not isinstance(extended[-1], int)
        ):
            raise CatalogProviderError(
                "label is not one token in the exact chat-ending context"
            )
        values.append(int(extended[-1]))
    if len(values) != len(set(values)):
        raise CatalogProviderError("allowed label token IDs are not distinct")
    return values


def _float32_evidence(value: float) -> tuple[float, str, str]:
    packed = struct.pack("<f", float(value))
    normalized = struct.unpack("<f", packed)[0]
    return normalized, normalized.hex(), packed.hex()


def _load_mutation_model() -> tuple[Any, Any, Any, dict[str, object], dict[str, object]]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise CatalogProviderError("catalog child requires CUDA_VISIBLE_DEVICES=0")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get(
        "TRANSFORMERS_OFFLINE"
    ) != "1":
        raise CatalogProviderError("catalog child requires offline model mode")
    if int(torch.cuda.device_count()) != 1 or int(torch.cuda.current_device()) != 0:
        raise CatalogProviderError("catalog child must see only logical cuda:0")
    evidence = verify_model_snapshot(seed_spec(INSTRUCT_MODEL_ID))
    snapshot = Path(str(evidence["snapshot_path"]))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()
    if str(next(model.parameters()).device) != "cuda:0":
        raise CatalogProviderError("catalog mutation model is not on cuda:0")
    processor = AutoProcessor.from_pretrained(
        snapshot,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    return (
        model,
        processor,
        torch,
        evidence,
        {
            "physical_gpu_index": 0,
            "cuda_visible_devices": "0",
            "visible_device_count": 1,
            "logical_device": "cuda:0",
        },
    )


def _score_one(
    *,
    model: Any,
    processor: Any,
    torch: Any,
    request: MutationRequest,
    choices: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    prompt = _selection_prompt(request, choices)
    rendered = processor.apply_chat_template(
        _messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise CatalogProviderError("processor chat template returned invalid text")
    labels = LABEL_TEXTS[: len(choices)]
    token_ids = _exact_label_token_ids(processor.tokenizer, rendered, labels)
    inputs = processor(
        text=[rendered],
        padding=True,
        return_tensors="pt",
    )
    forbidden = {
        key
        for key in inputs
        if any(marker in str(key).casefold() for marker in ("pixel", "image", "video"))
    }
    if forbidden:
        raise CatalogProviderError("catalog selector received non-text modal inputs")
    input_ids = [int(item) for item in inputs.input_ids[0].tolist()]
    inputs = inputs.to("cuda:0")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)
    torch.cuda.synchronize()
    finished = time.perf_counter()
    final_logits = outputs.logits[0, -1].float().detach().cpu()
    rows: list[dict[str, object]] = []
    scores: dict[str, float] = {}
    for ordinal, (label, choice, token_id) in enumerate(
        zip(labels, choices, token_ids, strict=True)
    ):
        score, score_hex, raw_hex = _float32_evidence(
            float(final_logits[token_id].item())
        )
        choice_id = str(choice["choice_id"])
        scores[choice_id] = score
        rows.append(
            {
                "ordinal": ordinal,
                "label_text": label,
                "choice_id": choice_id,
                "token_id": token_id,
                "score": score,
                "score_hex": score_hex,
                "score_float32_le_hex": raw_hex,
            }
        )
    ranking = rank_choice_ids(choices, scores)
    return {
        "attempt": "catalog_ranking",
        "selection_method": SELECTION_METHOD,
        "input_token_count": len(input_ids),
        "output_token_count": 1,
        "latency_seconds": finished - started,
        "finish_reason": "label_argmax",
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "text_only": True,
        "prompt_utf8": prompt,
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "rendered_chat_sha256": _sha256_bytes(rendered.encode("utf-8")),
        "input_token_ids": input_ids,
        "input_token_ids_sha256": _json_hash(input_ids),
        "label_scores": rows,
        "ranking": ranking,
    }


def _proposal(
    *,
    request: MutationRequest,
    choices: Sequence[Mapping[str, object]],
    call: Mapping[str, object],
    population_extra: Sequence[str],
    catalog_evidence: Mapping[str, str],
) -> tuple[dict[str, object], MutationOutcome]:
    ranking = call["ranking"]
    if not isinstance(ranking, list):
        raise CatalogProviderError("catalog ranking is not a list")
    population = tuple(
        sorted(set(request.population_program_sha256s) | set(population_extra))
    )
    choice, child, skipped = select_ranked_choice(
        parent_dsl=request.parent_dsl,
        model_id=request.model_id,
        choices=choices,
        ranking=ranking,
        population_program_sha256s=population,
    )
    child_dsl = canonical_candidate_source(child)
    source = render_candidate_program(child)
    digest = rendered_program_sha256(source)
    validate_rendered_program(source, expected_model_id=request.model_id)
    parent = parse_candidate_dsl(
        request.parent_dsl,
        expected_model_id=request.model_id,
    )
    changed = sorted(
        field
        for field in parent.__dataclass_fields__
        if getattr(parent, field) != getattr(child, field)
    )
    proposal = {
        "request_id": request.request_id,
        "generation": request.generation,
        "child_index": request.child_index,
        "parent_id": request.parent_id,
        "parent_program_sha256": request.parent_program_sha256,
        "population_program_sha256s": list(population),
        "catalog_version": CATALOG_VERSION,
        "catalog_sha256": catalog_evidence["catalog_sha256"],
        "catalog_manifest_sha256": catalog_evidence[
            "catalog_manifest_sha256"
        ],
        "focus": mutation_focus(request),
        "option_order": [str(item["choice_id"]) for item in choices],
        "selected_choice_id": choice["choice_id"],
        "duplicate_skips": skipped,
        "exact_patch": dict(choice["patch"]),
        "allowed_changed_fields": list(choice["allowed_changed_fields"]),
        "changed_fields": changed,
        "parent_dsl": request.parent_dsl,
        "child_dsl": child_dsl,
        "rendered_source": source,
        "rendered_sha256": digest,
        "parent_to_child_diff": rendered_program_diff(
            request.parent_program_source,
            source,
        ),
    }
    outcome = MutationOutcome(
        request_id=request.request_id,
        status="accepted",
        error_type=None,
        error_message=None,
        canonical_dsl=child_dsl,
        rendered_source=source,
        rendered_sha256=digest,
        repair_used=False,
        calls=(dict(call),),
        proposals=(proposal,),
    )
    return proposal, outcome


def _strict_requests(document: Mapping[str, object]) -> list[MutationRequest]:
    if set(document) != {"kind", "provider_version", "requests"}:
        raise CatalogProviderError("catalog child request fields are not exact")
    if document["kind"] != "catalog_mutation_generation":
        raise CatalogProviderError("catalog child request kind is invalid")
    if document["provider_version"] != PROVIDER_VERSION:
        raise CatalogProviderError("catalog child provider version mismatch")
    raw_requests = document["requests"]
    if not isinstance(raw_requests, list) or not 1 <= len(raw_requests) <= 4:
        raise CatalogProviderError("catalog request count is invalid")
    requests: list[MutationRequest] = []
    expected = set(MutationRequest.__dataclass_fields__)
    for raw in raw_requests:
        if not isinstance(raw, dict) or set(raw) != expected:
            raise CatalogProviderError("catalog mutation request fields are not exact")
        requests.append(MutationRequest(**raw))
    if len({item.request_id for item in requests}) != len(requests):
        raise CatalogProviderError("catalog mutation request IDs are duplicated")
    if len({item.model_id for item in requests}) != 1:
        raise CatalogProviderError("catalog generation crosses model populations")
    return requests


def _child_mutation_main() -> int:
    network = install_network_guard()
    try:
        document = read_frame(sys.stdin.buffer)
        if sys.stdin.buffer.read(1) != b"":
            raise FrameError("trailing catalog child input")
        requests = _strict_requests(document)
        catalog, _manifest = load_frozen_catalog()
        catalog_evidence = catalog_hashes()
        model, processor, torch, model_evidence, gpu_scope = (
            _load_mutation_model()
        )
        outcomes: list[MutationOutcome] = []
        selected_hashes: list[str] = []
        for request in requests:
            choices = choices_for(
                catalog,
                model_id=request.model_id,
                focus=mutation_focus(request),
            )
            call = _score_one(
                model=model,
                processor=processor,
                torch=torch,
                request=request,
                choices=choices,
            )
            _evidence, outcome = _proposal(
                request=request,
                choices=choices,
                call=call,
                population_extra=selected_hashes,
                catalog_evidence=catalog_evidence,
            )
            assert outcome.rendered_sha256 is not None
            selected_hashes.append(outcome.rendered_sha256)
            outcomes.append(outcome)
        if network.count != 0:
            raise CatalogProviderError("catalog child attempted network access")
        write_frame(
            sys.stdout.buffer,
            {
                "kind": "catalog_mutation_generation_result",
                "provider_version": PROVIDER_VERSION,
                "provider_name": PROVIDER_NAME,
                "selection_method": SELECTION_METHOD,
                "network_attempts": network.count,
                "model_evidence": model_evidence,
                "gpu_scope": gpu_scope,
                "catalog_evidence": catalog_evidence,
                "outcomes": [item.to_mapping() for item in outcomes],
            },
        )
        return 0
    except BaseException as exc:
        try:
            write_frame(
                sys.stdout.buffer,
                {
                    "kind": "catalog_mutation_generation_error",
                    "provider_version": PROVIDER_VERSION,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:2048],
                    "network_attempts": network.count,
                },
            )
        except BaseException:
            pass
        return 1


def _validate_score_rows(
    *,
    call: Mapping[str, object],
    choices: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    if set(call) != _CALL_FIELDS:
        raise CatalogProviderError("catalog call fields are not exact")
    if (
        call["attempt"] != "catalog_ranking"
        or call["selection_method"] != SELECTION_METHOD
        or call["output_token_count"] != 1
        or call["finish_reason"] != "label_argmax"
        or call["do_sample"] is not False
        or call["num_beams"] != 1
        or call["num_return_sequences"] != 1
        or call["text_only"] is not True
    ):
        raise CatalogProviderError("catalog call generation locks are invalid")
    for key in ("input_token_count", "output_token_count"):
        if isinstance(call[key], bool) or not isinstance(call[key], int) or call[key] < 0:
            raise CatalogProviderError("catalog token accounting is invalid")
    latency = call["latency_seconds"]
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or float(latency) < 0
    ):
        raise CatalogProviderError("catalog latency is invalid")
    prompt = call["prompt_utf8"]
    if (
        not isinstance(prompt, str)
        or len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES
        or call["prompt_sha256"] != _sha256_bytes(prompt.encode("utf-8"))
    ):
        raise CatalogProviderError("catalog prompt evidence is invalid")
    input_ids = call["input_token_ids"]
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or len(input_ids) != call["input_token_count"]
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in input_ids)
        or call["input_token_ids_sha256"] != _json_hash(input_ids)
    ):
        raise CatalogProviderError("catalog input token evidence is invalid")
    if not isinstance(call["rendered_chat_sha256"], str) or len(
        call["rendered_chat_sha256"]
    ) != 64:
        raise CatalogProviderError("catalog rendered chat hash is invalid")
    rows = call["label_scores"]
    if not isinstance(rows, list) or len(rows) != len(choices):
        raise CatalogProviderError("catalog score cardinality is invalid")
    scores: dict[str, float] = {}
    token_ids: list[int] = []
    for ordinal, (row, choice, label) in enumerate(
        zip(rows, choices, LABEL_TEXTS[: len(choices)], strict=True)
    ):
        if not isinstance(row, dict) or set(row) != _SCORE_ROW_FIELDS:
            raise CatalogProviderError("catalog score row fields are not exact")
        if (
            row["ordinal"] != ordinal
            or row["label_text"] != label
            or row["choice_id"] != choice["choice_id"]
            or isinstance(row["token_id"], bool)
            or not isinstance(row["token_id"], int)
            or row["token_id"] < 0
        ):
            raise CatalogProviderError("catalog score row identity is invalid")
        score = row["score"]
        raw_hex = row["score_float32_le_hex"]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not isinstance(raw_hex, str)
            or len(raw_hex) != 8
        ):
            raise CatalogProviderError("catalog score scalar is invalid")
        try:
            restored = struct.unpack("<f", bytes.fromhex(raw_hex))[0]
        except (ValueError, struct.error) as exc:
            raise CatalogProviderError("catalog raw score bytes are invalid") from exc
        if restored != float(score) or row["score_hex"] != restored.hex():
            raise CatalogProviderError("catalog score representations disagree")
        scores[str(choice["choice_id"])] = restored
        token_ids.append(int(row["token_id"]))
    if len(token_ids) != len(set(token_ids)):
        raise CatalogProviderError("catalog label token IDs are not distinct")
    ranking = rank_choice_ids(choices, scores)
    if call["ranking"] != ranking:
        raise CatalogProviderError("catalog ranking is forged or stale")
    return scores


def _validate_child_reply(
    reply: Mapping[str, object],
    requests: Sequence[MutationRequest],
) -> list[MutationOutcome]:
    expected = {
        "kind",
        "provider_version",
        "provider_name",
        "selection_method",
        "network_attempts",
        "model_evidence",
        "gpu_scope",
        "catalog_evidence",
        "outcomes",
    }
    if set(reply) != expected:
        raise CatalogProviderError("catalog child reply fields are not exact")
    if (
        reply["kind"] != "catalog_mutation_generation_result"
        or reply["provider_version"] != PROVIDER_VERSION
        or reply["provider_name"] != PROVIDER_NAME
        or reply["selection_method"] != SELECTION_METHOD
        or reply["network_attempts"] != 0
    ):
        raise CatalogProviderError("catalog child reply identity is invalid")
    if reply["gpu_scope"] != {
        "physical_gpu_index": 0,
        "cuda_visible_devices": "0",
        "visible_device_count": 1,
        "logical_device": "cuda:0",
    }:
        raise CatalogProviderError("catalog child GPU scope is invalid")
    expected_model = seed_spec(INSTRUCT_MODEL_ID)
    evidence = reply["model_evidence"]
    if (
        not isinstance(evidence, dict)
        or evidence.get("model_id") != INSTRUCT_MODEL_ID
        or evidence.get("revision") != expected_model.model_revision
        or evidence.get("weight_sha256") != expected_model.weight_sha256
    ):
        raise CatalogProviderError("catalog mutation model evidence is invalid")
    if reply["catalog_evidence"] != catalog_hashes():
        raise CatalogProviderError("catalog evidence closure mismatch")
    raw_outcomes = reply["outcomes"]
    if not isinstance(raw_outcomes, list) or len(raw_outcomes) != len(requests):
        raise CatalogProviderError("catalog outcome cardinality mismatch")
    catalog, _manifest = load_frozen_catalog()
    outcomes: list[MutationOutcome] = []
    selected_hashes: list[str] = []
    expected_fields = set(MutationOutcome.__dataclass_fields__)
    for raw, request in zip(raw_outcomes, requests, strict=True):
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise CatalogProviderError("catalog outcome fields are not exact")
        raw_value = dict(raw)
        calls = raw_value.pop("calls")
        proposals = raw_value.pop("proposals")
        if not isinstance(calls, list) or len(calls) != 1:
            raise CatalogProviderError("catalog request must have exactly one call")
        if not isinstance(proposals, list) or len(proposals) != 1:
            raise CatalogProviderError("catalog request must have one proposal")
        call = calls[0]
        proposal = proposals[0]
        if not isinstance(call, dict) or not isinstance(proposal, dict):
            raise CatalogProviderError("catalog call/proposal is not an object")
        choices = choices_for(
            catalog,
            model_id=request.model_id,
            focus=mutation_focus(request),
        )
        _validate_score_rows(call=call, choices=choices)
        if call["prompt_utf8"] != _selection_prompt(request, choices):
            raise CatalogProviderError("catalog prompt is not reproducible")
        if set(proposal) != _PROPOSAL_FIELDS:
            raise CatalogProviderError("catalog proposal fields are not exact")
        expected_choice, expected_child, expected_skips = select_ranked_choice(
            parent_dsl=request.parent_dsl,
            model_id=request.model_id,
            choices=choices,
            ranking=call["ranking"],
            population_program_sha256s=tuple(
                sorted(
                    set(request.population_program_sha256s)
                    | set(selected_hashes)
                )
            ),
        )
        expected_dsl = canonical_candidate_source(expected_child)
        expected_source = render_candidate_program(expected_child)
        expected_sha = rendered_program_sha256(expected_source)
        parent = parse_candidate_dsl(
            request.parent_dsl,
            expected_model_id=request.model_id,
        )
        changed = sorted(
            field
            for field in parent.__dataclass_fields__
            if getattr(parent, field) != getattr(expected_child, field)
        )
        expected_proposal = {
            "request_id": request.request_id,
            "generation": request.generation,
            "child_index": request.child_index,
            "parent_id": request.parent_id,
            "parent_program_sha256": request.parent_program_sha256,
            "population_program_sha256s": list(
                sorted(
                    set(request.population_program_sha256s)
                    | set(selected_hashes)
                )
            ),
            "catalog_version": CATALOG_VERSION,
            "catalog_sha256": catalog_hashes()["catalog_sha256"],
            "catalog_manifest_sha256": catalog_hashes()[
                "catalog_manifest_sha256"
            ],
            "focus": mutation_focus(request),
            "option_order": [str(item["choice_id"]) for item in choices],
            "selected_choice_id": expected_choice["choice_id"],
            "duplicate_skips": expected_skips,
            "exact_patch": dict(expected_choice["patch"]),
            "allowed_changed_fields": list(
                expected_choice["allowed_changed_fields"]
            ),
            "changed_fields": changed,
            "parent_dsl": request.parent_dsl,
            "child_dsl": expected_dsl,
            "rendered_source": expected_source,
            "rendered_sha256": expected_sha,
            "parent_to_child_diff": rendered_program_diff(
                request.parent_program_source,
                expected_source,
            ),
        }
        if proposal != expected_proposal:
            raise CatalogProviderError("catalog proposal evidence is not reproducible")
        outcome = MutationOutcome(
            calls=(dict(call),),
            proposals=(dict(proposal),),
            **raw_value,
        )
        if (
            outcome.request_id != request.request_id
            or outcome.status != "accepted"
            or outcome.error_type is not None
            or outcome.error_message is not None
            or outcome.canonical_dsl != expected_dsl
            or outcome.rendered_source != expected_source
            or outcome.rendered_sha256 != expected_sha
            or outcome.repair_used is not False
        ):
            raise CatalogProviderError("catalog outcome does not match selected patch")
        selected_hashes.append(expected_sha)
        outcomes.append(outcome)
    return outcomes


def validate_catalog_provider_log(
    document: Mapping[str, object],
) -> tuple[list[MutationRequest], list[MutationOutcome]]:
    """Recompute a successful provider ledger from frozen inputs and scores."""
    expected = {
        "provider_name", "provider_version", "selection_method",
        "catalog_evidence", "child_pid", "child_returncode", "child_reaped",
        "child_elapsed_seconds", "child_shell", "child_start_new_session",
        "child_stderr_size_bytes", "child_stderr_sha256", "child_stderr_utf8",
        "network_attempts", "model_evidence", "gpu_scope", "execution_class",
        "run_manifest_sha256", "requests", "outcomes", "total_calls",
        "total_input_tokens", "total_output_tokens",
    }
    if set(document) != expected:
        raise CatalogProviderError("catalog provider log fields are not exact")
    if (
        document["provider_name"] != PROVIDER_NAME
        or document["provider_version"] != PROVIDER_VERSION
        or document["selection_method"] != SELECTION_METHOD
        or document["execution_class"] != "production_evolution"
        or document["network_attempts"] != 0
        or document["child_returncode"] != 0
        or document["child_reaped"] is not True
        or document["child_shell"] is not False
        or document["child_start_new_session"] is not True
    ):
        raise CatalogProviderError("catalog provider log safety identity is invalid")
    request_rows = document["requests"]
    if not isinstance(request_rows, list) or not request_rows:
        raise CatalogProviderError("catalog provider request ledger is empty")
    requests: list[MutationRequest] = []
    expected_request_fields = set(MutationRequest.__dataclass_fields__) | {
        "mutation_focus"
    }
    for row in request_rows:
        if not isinstance(row, dict) or set(row) != expected_request_fields:
            raise CatalogProviderError("catalog provider request fields are not exact")
        contract = dict(row)
        recorded_focus = contract.pop("mutation_focus")
        try:
            request = MutationRequest(**contract)
        except (TypeError, ValueError) as exc:
            raise CatalogProviderError("catalog provider request is invalid") from exc
        if recorded_focus != mutation_focus(request):
            raise CatalogProviderError("catalog provider request focus is forged")
        requests.append(request)
    reply = {
        "kind": "catalog_mutation_generation_result",
        "provider_version": document["provider_version"],
        "provider_name": document["provider_name"],
        "selection_method": document["selection_method"],
        "network_attempts": document["network_attempts"],
        "model_evidence": document["model_evidence"],
        "gpu_scope": document["gpu_scope"],
        "catalog_evidence": document["catalog_evidence"],
        "outcomes": document["outcomes"],
    }
    outcomes = _validate_child_reply(reply, requests)
    if document["outcomes"] != [item.to_mapping() for item in outcomes]:
        raise CatalogProviderError("catalog provider outcome ledger changed")
    total_input = sum(int(item.calls[0]["input_token_count"]) for item in outcomes)
    if (
        document["total_calls"] != len(outcomes)
        or document["total_output_tokens"] != len(outcomes)
        or document["total_input_tokens"] != total_input
    ):
        raise CatalogProviderError("catalog provider aggregate accounting is invalid")
    elapsed = document["child_elapsed_seconds"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0
    ):
        raise CatalogProviderError("catalog provider child timing is invalid")
    return requests, outcomes


def _request_evidence(request: MutationRequest) -> dict[str, object]:
    value = request.to_mapping()
    value["mutation_focus"] = mutation_focus(request)
    return value


class CatalogConstrainedQwenMutationProvider:
    """CPU launcher; the trusted, zero-network child alone imports torch."""

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
            raise CatalogProviderError("catalog generation request count is invalid")
        if any(item.execution_class != "production_evolution" for item in values):
            raise CatalogProviderError("catalog provider rejects synthetic execution")
        manifests = {item.run_manifest_sha256 for item in values}
        if len(manifests) != 1:
            raise CatalogProviderError("catalog requests cross run manifests")
        frame = encode_frame(
            {
                "kind": "catalog_mutation_generation",
                "provider_version": PROVIDER_VERSION,
                "requests": [item.to_mapping() for item in values],
            }
        )
        try:
            child = run_trusted_child(
                module="evolution.catalog_provider",
                function="_child_mutation_main",
                input_bytes=frame,
                timeout_seconds=timeout_seconds,
            )
        except ChildTimeout as exc:
            atomic_create_json(
                artifact_path,
                {
                    "status": "failed_before_reply",
                    "provider_name": PROVIDER_NAME,
                    "provider_version": PROVIDER_VERSION,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "child_reaped": True,
                    "requests": [item.to_mapping() for item in values],
                },
            )
            raise CatalogProviderError("catalog child timed out and was reaped") from exc
        self.last_child_elapsed_seconds = child.elapsed_seconds
        try:
            reply = decode_frame(child.stdout)
        except FrameError as exc:
            atomic_create_json(
                artifact_path,
                {
                    "status": "failed_before_valid_reply",
                    "provider_name": PROVIDER_NAME,
                    "provider_version": PROVIDER_VERSION,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "child_pid": child.pid,
                    "child_returncode": child.returncode,
                    "child_reaped": child.reaped,
                    "child_elapsed_seconds": child.elapsed_seconds,
                    "stdout_sha256": _sha256_bytes(child.stdout),
                    "stderr_sha256": _sha256_bytes(child.stderr),
                    "requests": [item.to_mapping() for item in values],
                },
            )
            raise CatalogProviderError("catalog child stdout is not one frame") from exc
        if child.returncode != 0 or not child.reaped:
            atomic_create_json(
                artifact_path,
                {
                    "status": "failed_before_valid_reply",
                    "provider_name": PROVIDER_NAME,
                    "provider_version": PROVIDER_VERSION,
                    "error_type": str(reply.get("error_type", "unknown")),
                    "error_message": str(reply.get("error_message", "unknown")),
                    "network_attempts": reply.get("network_attempts"),
                    "child_pid": child.pid,
                    "child_returncode": child.returncode,
                    "child_reaped": child.reaped,
                    "child_elapsed_seconds": child.elapsed_seconds,
                    "stderr_sha256": _sha256_bytes(child.stderr),
                    "requests": [item.to_mapping() for item in values],
                },
            )
            raise CatalogProviderError(
                "catalog child failed before a valid selection reply"
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
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:2048],
                    "child_pid": child.pid,
                    "child_returncode": child.returncode,
                    "child_reaped": child.reaped,
                    "child_elapsed_seconds": child.elapsed_seconds,
                    "stderr_sha256": _sha256_bytes(child.stderr),
                    "requests": [item.to_mapping() for item in values],
                    "raw_child_reply": reply,
                },
            )
            raise
        log = {
            "provider_name": PROVIDER_NAME,
            "provider_version": PROVIDER_VERSION,
            "selection_method": SELECTION_METHOD,
            "catalog_evidence": reply["catalog_evidence"],
            "child_pid": child.pid,
            "child_returncode": child.returncode,
            "child_reaped": child.reaped,
            "child_elapsed_seconds": child.elapsed_seconds,
            "child_shell": child.shell,
            "child_start_new_session": child.start_new_session,
            "child_stderr_size_bytes": len(child.stderr),
            "child_stderr_sha256": _sha256_bytes(child.stderr),
            "child_stderr_utf8": child.stderr.decode(
                "utf-8", errors="replace"
            )[:65536],
            "network_attempts": reply["network_attempts"],
            "model_evidence": reply["model_evidence"],
            "gpu_scope": reply["gpu_scope"],
            "execution_class": "production_evolution",
            "run_manifest_sha256": next(iter(manifests)),
            "requests": [_request_evidence(item) for item in values],
            "outcomes": [item.to_mapping() for item in outcomes],
            "total_calls": len(outcomes),
            "total_input_tokens": sum(
                int(item.calls[0]["input_token_count"]) for item in outcomes
            ),
            "total_output_tokens": len(outcomes),
        }
        atomic_create_json(artifact_path, log)
        return outcomes


__all__ = [
    "CatalogConstrainedQwenMutationProvider",
    "CatalogProviderError",
    "LABEL_TEXTS",
    "PROVIDER_NAME",
    "PROVIDER_VERSION",
    "SELECTION_METHOD",
    "validate_catalog_provider_log",
]
