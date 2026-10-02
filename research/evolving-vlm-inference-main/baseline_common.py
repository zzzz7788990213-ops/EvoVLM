"""Shared immutable naive baseline for Qwen3-VL.

Heavy libraries are deliberately lazy: the official evaluator selects a single
CUDA device before importing a submitted program.  Model discovery is portable
and hash-authenticated through model_locator.
"""
from __future__ import annotations

import copy
import hashlib
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable

from model_locator import (
    ModelLocationError,
    resolve_model_source,
    resolve_readable_image,
)


DTYPE = "bfloat16"
QUANTIZATION = "none"
ATTENTION_IMPLEMENTATION = "sdpa"
MAX_NEW_TOKENS = 128
BATCH_SIZE = 1
EFFECTIVE_GENERATION_CONFIG = {
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "min_p": None,
    "typical_p": None,
    "max_new_tokens": MAX_NEW_TOKENS,
    "use_cache": True,
    "use_model_defaults": False,
}


class BaselineInferenceError(RuntimeError):
    """The fixed baseline could not be loaded or executed safely."""


@dataclass(frozen=True)
class BaselineBackendAdapter:
    """Test-only dependency adapter; production wrappers never expose it."""

    load_model: Callable[[Any], Any]
    load_processor: Callable[[Any], Any]
    process_vision_info: Callable[[list[dict[str, Any]]], tuple[Any, Any]]
    visible_device_count: Callable[[], int]
    device_name: Callable[[int], str]


def _baseline_messages(image_reference: object, question: str) -> list[dict[str, Any]]:
    """Return the exact sealed naive-baseline chat message."""
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_reference},
            {"type": "text", "text": question},
        ],
    }]


def _baseline_processor_kwargs(
    rendered: object, image_inputs: object, video_inputs: object,
) -> dict[str, object]:
    return {
        "text": [rendered],
        "images": image_inputs,
        "videos": video_inputs,
        "padding": True,
        "return_tensors": "pt",
    }


def _assert_single_visible_gpu(torch_module: Any, count: int | None = None) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if re.fullmatch(r"[0-9]+", visible) is None:
        raise BaselineInferenceError(
            "CUDA_VISIBLE_DEVICES must select exactly one physical device"
        )
    actual_count = int(
        torch_module.cuda.device_count() if count is None else count
    )
    if actual_count != 1:
        raise BaselineInferenceError(
            f"expected one visible CUDA device, found {actual_count}"
        )
    if count is None and int(torch_module.cuda.current_device()) != 0:
        raise BaselineInferenceError("expected logical cuda:0")
    return int(visible)


def make_inference(
    model_id: str,
    revision: str,
    directory: str,
    weight_sha256: str,
    *,
    backend_adapter: BaselineBackendAdapter | None = None,
) -> Callable[[str, str], str]:
    model: Any | None = None
    processor: Any | None = None
    generation_config: Any | None = None
    torch_module: Any | None = None
    vision_helper: Any | None = None

    def get_model() -> tuple[Any, Any, Any, Any]:
        nonlocal model, processor, generation_config, torch_module, vision_helper
        if model is not None and processor is not None:
            return model, processor, generation_config, torch_module

        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        torch_module = torch
        count = (
            None
            if backend_adapter is None
            else backend_adapter.visible_device_count()
        )
        physical_index = _assert_single_visible_gpu(torch, count)
        try:
            source = resolve_model_source(
                model_id=model_id,
                revision=revision,
                model_directory=directory,
                expected_weight_sha256=weight_sha256,
            )
        except ModelLocationError as exc:
            raise BaselineInferenceError(str(exc)) from exc

        if backend_adapter is None:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                source.reference,
                dtype=torch.bfloat16,
                device_map={"": 0},
                attn_implementation=ATTENTION_IMPLEMENTATION,
                low_cpu_mem_usage=True,
                **source.pretrained_kwargs,
            )
            processor = AutoProcessor.from_pretrained(
                source.reference,
                use_fast=True,
                **source.pretrained_kwargs,
            )
            vision_helper = process_vision_info
        else:
            model = backend_adapter.load_model(source.reference)
            processor = backend_adapter.load_processor(source.reference)
            vision_helper = backend_adapter.process_vision_info

        model.eval()
        if str(next(model.parameters()).device) != "cuda:0":
            raise BaselineInferenceError("model did not load on logical cuda:0")
        generation_config = copy.deepcopy(model.generation_config)
        for key, value in EFFECTIVE_GENERATION_CONFIG.items():
            if key != "use_model_defaults":
                setattr(generation_config, key, value)
        if generation_config.do_sample or generation_config.num_beams != 1:
            raise BaselineInferenceError("effective generation config is not greedy")
        vlm_inference.model_metadata = {
            "model_id": model_id,
            "model_revision": revision,
            "processor_revision": revision,
            "model_weight_sha256": source.weight_sha256,
            "model_source": source.reference,
            "local_files_only": source.local_files_only,
            "dtype": DTYPE,
            "quantization": QUANTIZATION,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "physical_gpu_index": physical_index,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "visible_device_count": 1,
            "logical_device": "cuda:0",
            "device_name": (
                torch.cuda.get_device_name(0)
                if backend_adapter is None
                else backend_adapter.device_name(0)
            ),
        }
        return model, processor, generation_config, torch

    def vlm_inference(
        image_path: str,
        question: str = "Describe this image in detail.",
    ) -> str:
        nonlocal vision_helper
        current_model, current_processor, current_generation, torch = get_model()
        if backend_adapter is None:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        path = resolve_readable_image(image_path)
        image_bytes = path.read_bytes()
        input_image_sha256 = hashlib.sha256(image_bytes).hexdigest()

        from PIL import Image
        with Image.open(BytesIO(image_bytes)) as opened:
            input_image = opened.copy()
        messages = _baseline_messages(input_image, question)
        rendered = current_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert vision_helper is not None
        image_inputs, video_inputs = vision_helper(messages)
        inputs = current_processor(**_baseline_processor_kwargs(
            rendered, image_inputs, video_inputs
        )).to("cuda:0")
        if backend_adapter is None:
            torch.cuda.synchronize()
        preprocessed_at = time.perf_counter()
        with torch.inference_mode():
            generated_ids = current_model.generate(
                **inputs,
                generation_config=current_generation,
                use_model_defaults=False,
            )
        if backend_adapter is None:
            torch.cuda.synchronize()
        generation_finished = time.perf_counter()
        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = current_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(output) != 1 or not isinstance(output[0], str):
            raise TypeError("decoder did not return exactly one string")
        generated_tokens = generated_ids_trimmed[0].tolist()
        eos_value = current_generation.eos_token_id
        eos_ids = (
            set(eos_value)
            if isinstance(eos_value, (list, tuple, set))
            else {eos_value}
        )
        if generated_tokens and generated_tokens[-1] in eos_ids:
            termination_reason = "eos"
        elif len(generated_tokens) >= MAX_NEW_TOKENS:
            termination_reason = "length"
        else:
            termination_reason = "other"
        finished = time.perf_counter()
        vlm_inference.last_generation_metadata = {
            "input_token_count": int(inputs.input_ids.shape[-1]),
            "generated_token_count": len(generated_tokens),
            "termination_reason": termination_reason,
            "preprocessing_seconds": preprocessed_at - started,
            "generation_seconds": generation_finished - preprocessed_at,
            "total_seconds": finished - started,
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(0))
                if backend_adapter is None else 0
            ),
            "input_image_path": str(path),
            "input_image_size_bytes": len(image_bytes),
            "input_image_sha256": input_image_sha256,
        }
        return output[0].strip()

    vlm_inference.last_generation_metadata = None
    vlm_inference.model_metadata = None
    return vlm_inference
