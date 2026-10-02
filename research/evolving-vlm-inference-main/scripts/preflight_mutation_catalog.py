#!/usr/bin/env python3
"""CPU-only exact-context label-token preflight for the frozen mutation catalog."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVO_ROOT = PROJECT_ROOT.parents[1]
if PROJECT_ROOT.is_symlink() or not PROJECT_ROOT.resolve(strict=True).is_relative_to(EVO_ROOT):
    raise SystemExit("unsafe project root")
sys.path.insert(0, str(PROJECT_ROOT))

from evolution.catalog_provider import (  # noqa: E402
    LABEL_TEXTS,
    PROVIDER_NAME,
    PROVIDER_VERSION,
    _exact_label_token_ids,
    _messages,
    _selection_prompt,
)
from evolution.dsl import MODEL_IDS, MODEL_METADATA, canonical_candidate_source, seed_spec  # noqa: E402
from evolution.local_provider import MutationRequest, mutation_focus  # noqa: E402
from evolution.mutation_catalog import catalog_hashes, choices_for, load_frozen_catalog  # noqa: E402
from evolution.process_safety import install_network_guard  # noqa: E402
from evolution.renderer import render_candidate_program, rendered_program_sha256  # noqa: E402


def _request(model_id: str, generation: int, child_index: int) -> MutationRequest:
    parent = seed_spec(model_id)
    source = render_candidate_program(parent)
    return MutationRequest(
        request_id=f"preflight_g{generation}_c{child_index}",
        generation=generation,
        child_index=child_index,
        parent_id="preflight_seed",
        model_id=model_id,
        parent_dsl=canonical_candidate_source(parent),
        parent_program_source=source,
        parent_program_sha256=rendered_program_sha256(source),
        parent_metrics={
            "stage": "seed",
            "samples": 0,
            "accuracy": 0.0,
            "errors": 0,
            "mean_latency_seconds": 0.0,
        },
        run_manifest_sha256="a" * 64,
        execution_class="production_evolution",
    )


def main() -> int:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("preflight requires offline mode")
    network = install_network_guard()
    from transformers import AutoProcessor

    mutation_model = next(item for item in MODEL_IDS if item.endswith("-Instruct"))
    metadata = MODEL_METADATA[mutation_model]
    snapshot = EVO_ROOT / "models" / str(metadata["directory"])
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise RuntimeError("mutation snapshot is not physical")
    processor = AutoProcessor.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    catalog, _ = load_frozen_catalog()
    cases = []
    schedule = ((1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (2, 3))
    for model_id in MODEL_IDS:
        for generation, child_index in schedule:
            request = _request(model_id, generation, child_index)
            focus = mutation_focus(request)
            choices = choices_for(catalog, model_id=model_id, focus=focus)
            prompt = _selection_prompt(request, choices)
            rendered = processor.apply_chat_template(
                _messages(prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
            if not isinstance(rendered, str) or not rendered:
                raise RuntimeError("chat template preflight returned invalid text")
            labels = LABEL_TEXTS[: len(choices)]
            token_ids = _exact_label_token_ids(
                processor.tokenizer,
                rendered,
                labels,
            )
            cases.append(
                {
                    "model_id": model_id,
                    "focus": focus,
                    "labels": list(labels),
                    "token_ids": token_ids,
                    "distinct_single_token": len(token_ids) == len(set(token_ids)),
                }
            )
    if network.count != 0:
        raise RuntimeError("preflight attempted network")
    print(
        json.dumps(
            {
                "preflight_version": "phase2-catalog-label-preflight-v1",
                "provider_name": PROVIDER_NAME,
                "provider_version": PROVIDER_VERSION,
                "mutation_model_id": mutation_model,
                "mutation_model_revision": metadata["revision"],
                "mutation_model_weight_sha256": metadata["weight_sha256"],
                "catalog": catalog_hashes(),
                "network_attempts": network.count,
                "cuda_api_called": False,
                "cases": cases,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
