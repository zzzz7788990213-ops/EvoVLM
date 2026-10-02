"""Shared minimal manual inference implementation for the two required models.

Imports of torch, transformers, PIL, and qwen_vl_utils are deliberately lazy so
CPU contract tests and evaluator argument parsing cannot initialize CUDA.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from model_locator import (
    ModelLocationError,
    resolve_model_source,
    resolve_readable_image,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT / "models"
DTYPE = "bfloat16"
QUANTIZATION = "none"
ATTENTION_IMPLEMENTATION = "sdpa"
FINAL_CLOSE = "</final>"


class ManualInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ManualSpec:
    model_id: str
    revision: str
    model_directory: str
    weight_sha256: str
    mode: str
    max_new_tokens: int

    def __post_init__(self) -> None:
        if self.mode not in {"instruct", "thinking"}:
            raise ValueError("mode must be instruct or thinking")
        expected = 48 if self.mode == "instruct" else 384
        if self.max_new_tokens != expected:
            raise ValueError(f"{self.mode} max_new_tokens must be {expected}")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("revision must be a 40-character Git hash")
        if not re.fullmatch(r"[0-9a-f]{64}", self.weight_sha256):
            raise ValueError("weight_sha256 must be a SHA-256 digest")

    @property
    def effective_generation_config(self) -> dict[str, Any]:
        return {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": self.max_new_tokens,
            "use_cache": True,
            "stop_text": FINAL_CLOSE,
        }


INSTRUCT_SPEC = ManualSpec(
    model_id="Qwen/Qwen3-VL-2B-Instruct",
    revision="78448d793a7eb2f7a987a1da76d464384aa1becd",
    model_directory="Qwen3-VL-2B-Instruct",
    weight_sha256="7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
    mode="instruct",
    max_new_tokens=48,
)

THINKING_SPEC = ManualSpec(
    model_id="Qwen/Qwen3-VL-2B-Thinking",
    revision="d4f1d9f1b0107e12699c3c83d7baacd07fd0abd3",
    model_directory="Qwen3-VL-2B-Thinking",
    weight_sha256="d889ae8b5bffcfce115bc64e47601b1e90ce44b1d9f40ed78e620fd22417d888",
    mode="thinking",
    max_new_tokens=384,
)


_FINAL_PATTERN = re.compile(r"<final>\s*(.*?)\s*</final>", re.IGNORECASE | re.DOTALL)


def build_user_prompt(spec: ManualSpec, question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    if spec.mode == "instruct":
        prefix = (
            "Answer the chart question concisely. Return only the answer enclosed "
            "in <final> and </final>; do not add an explanation. Question: "
        )
    else:
        prefix = (
            "Analyze the chart carefully using the model's native reasoning. "
            "Finish with the concise answer enclosed in <final> and </final>. "
            "Question: "
        )
    return prefix + question.strip()


def _manual_messages(image_reference: object, prompt: str) -> list[dict[str, object]]:
    """Return the exact formal chat message structure."""
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_reference},
            {"type": "text", "text": prompt},
        ],
    }]


def _manual_processor_kwargs(
    rendered: object, image_inputs: object, video_inputs: object,
) -> dict[str, object]:
    """Return the exact formal processor keyword arguments."""
    return {
        "text": [rendered],
        "images": image_inputs,
        "videos": video_inputs,
        "padding": True,
        "return_tensors": "pt",
    }


def _manual_generation_kwargs(
    spec: ManualSpec, stopping_criteria: object,
) -> dict[str, object]:
    """Return the exact formal greedy generation keyword arguments."""
    return {
        "max_new_tokens": spec.max_new_tokens, "do_sample": False,
        "num_beams": 1, "use_cache": True,
        "stopping_criteria": stopping_criteria,
    }


def extract_final_answer(text: str) -> str:
    """Extract one terminal final tag; otherwise preserve a conservative fallback."""
    if not isinstance(text, str):
        raise TypeError("decoded model output must be a string")
    stripped = text.strip()
    matches = list(_FINAL_PATTERN.finditer(stripped))
    if len(matches) == 1:
        match = matches[0]
        answer = match.group(1).strip()
        if answer and not stripped[match.end():].strip():
            return answer
    # A direct untagged answer remains usable. Malformed/multiple tagged text is
    # returned unchanged so the evaluator cannot be tricked by an invented parse.
    return stripped


class ManualBackend(Protocol):
    def generate(self, image_path: str, prompt: str, spec: ManualSpec) -> str:
        pass


class _TransformersBackend:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._processor: Any | None = None
        self.last_generation_metadata: dict[str, Any] | None = None

    @staticmethod
    def _assert_single_visible_gpu(torch: Any) -> None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if re.fullmatch(r"[0-9]+", visible) is None:
            raise ManualInferenceError(
                "CUDA_VISIBLE_DEVICES must select exactly one physical device"
            )
        count = int(torch.cuda.device_count())
        if count != 1:
            raise ManualInferenceError(f"expected one CUDA device, found {count}")
        current = int(torch.cuda.current_device())
        if current != 0:
            raise ManualInferenceError(f"expected logical cuda:0, found cuda:{current}")

    @staticmethod
    def _model_source(spec: ManualSpec):
        try:
            return resolve_model_source(
                model_id=spec.model_id,
                revision=spec.revision,
                model_directory=spec.model_directory,
                expected_weight_sha256=spec.weight_sha256,
            )
        except ModelLocationError as exc:
            raise ManualInferenceError(str(exc)) from exc

    def _load(self, spec: ManualSpec) -> tuple[Any, Any, Any, Any]:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import (
            AutoProcessor,
            Qwen3VLForConditionalGeneration,
            StoppingCriteria,
            StoppingCriteriaList,
        )

        self._assert_single_visible_gpu(torch)
        source = self._model_source(spec)
        if self._model is None:
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                source.reference,
                dtype=torch.bfloat16,
                device_map={"": 0},
                attn_implementation=ATTENTION_IMPLEMENTATION,
                low_cpu_mem_usage=True,
                **source.pretrained_kwargs,
            )
            self._model.eval()
            if str(next(self._model.parameters()).device) != "cuda:0":
                raise ManualInferenceError("model did not load on logical cuda:0")
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                source.reference,
                use_fast=True,
                **source.pretrained_kwargs,
            )
        return (
            self._model,
            self._processor,
            torch,
            (process_vision_info, StoppingCriteria, StoppingCriteriaList),
        )

    def generate(self, image_path: str, prompt: str, spec: ManualSpec) -> str:
        model, processor, torch, helpers = self._load(spec)
        process_vision_info, stopping_base, stopping_list = helpers
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        resolved = resolve_readable_image(image_path)
        messages = _manual_messages(str(resolved), prompt)
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(**_manual_processor_kwargs(
            rendered, image_inputs, video_inputs
        )).to("cuda:0")

        tokenizer = getattr(processor, "tokenizer", processor)
        closing_ids = tokenizer.encode(FINAL_CLOSE, add_special_tokens=False)
        if not closing_ids:
            raise ManualInferenceError("closing final tag did not tokenize")

        class StopOnFinalTag(stopping_base):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                del scores, kwargs
                suffix = input_ids[0, -len(closing_ids):].tolist()
                return suffix == closing_ids

        torch.cuda.synchronize()
        preprocessed_at = time.perf_counter()
        generation_started = preprocessed_at
        with torch.inference_mode():
            generated = model.generate(**inputs, **_manual_generation_kwargs(
                spec,
                stopping_list([StopOnFinalTag()]),
            ))
        torch.cuda.synchronize()
        generation_finished = time.perf_counter()
        trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        decoded = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(decoded) != 1 or not isinstance(decoded[0], str):
            raise TypeError("decoder must return exactly one string")
        completion_ids = trimmed[0].tolist()
        eos_value = getattr(model.generation_config, "eos_token_id", None)
        if isinstance(eos_value, int):
            eos_ids = {eos_value}
        elif isinstance(eos_value, (list, tuple, set)):
            eos_ids = {int(value) for value in eos_value}
        else:
            eos_ids = set()
        if completion_ids[-len(closing_ids):] == closing_ids:
            termination_reason = "stop_text"
        elif completion_ids and completion_ids[-1] in eos_ids:
            termination_reason = "eos"
        elif len(completion_ids) >= spec.max_new_tokens:
            termination_reason = "length"
        else:
            termination_reason = "other"
        finished = time.perf_counter()
        self.last_generation_metadata = {
            "input_token_count": int(inputs.input_ids.shape[-1]),
            "generated_token_count": len(completion_ids),
            "termination_reason": termination_reason,
            "preprocessing_seconds": preprocessed_at - started,
            "generation_seconds": generation_finished - generation_started,
            "total_seconds": finished - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        }
        return decoded[0]


def make_manual_inference(
    spec: ManualSpec,
    *,
    backend_factory: Callable[[], ManualBackend] | None = None,
) -> Callable[[str, str], str]:
    backend: ManualBackend | None = None

    def vlm_inference(image_path: str, question: str) -> str:
        nonlocal backend
        if not isinstance(image_path, str) or not image_path:
            raise ValueError("image_path must be a nonempty string")
        prompt = build_user_prompt(spec, question)
        if backend is None:
            backend = (
                backend_factory() if backend_factory is not None else _TransformersBackend()
            )
        raw_text = backend.generate(image_path, prompt, spec)
        prediction = extract_final_answer(raw_text)
        metadata = getattr(backend, "last_generation_metadata", None)
        if isinstance(metadata, dict):
            vlm_inference.last_generation_metadata = dict(metadata)
        return prediction

    vlm_inference.last_generation_metadata = None
    return vlm_inference
