"""Trusted runtime used only by deterministic renderer-generated programs."""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from model_locator import (
    ModelLocationError,
    ModelSource,
    resolve_model_source,
    resolve_readable_image,
)

from .dsl import ProgramSpec


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = PROJECT_ROOT / "models"
CANDIDATE_RUNTIME_VERSION = "phase2-trusted-target-runtime-v2"
_FINAL_PATTERN = re.compile(
    r"<final>\s*(.*?)\s*</final>",
    flags=re.IGNORECASE | re.DOTALL,
)
_BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}", flags=re.DOTALL)
_NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?$"
)


class CandidateRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class InferenceMeasurement:
    prediction: str
    raw_text: str
    input_token_count: int
    generated_token_count: int
    termination_reason: str
    preprocessing_seconds: float
    generation_seconds: float
    total_seconds: float
    peak_memory_bytes: int

    def __post_init__(self) -> None:
        for value in (
            self.preprocessing_seconds,
            self.generation_seconds,
            self.total_seconds,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("measurement time must be finite and nonnegative")
        for value in (
            self.input_token_count,
            self.generated_token_count,
            self.peak_memory_bytes,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("measurement counts must be nonnegative integers")
        if self.termination_reason not in {
            "eos",
            "length",
            "stop_text",
            "other",
        }:
            raise ValueError("invalid termination reason")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _resolve_model(spec: ProgramSpec) -> ModelSource:
    try:
        return resolve_model_source(
            model_id=spec.model_id,
            revision=spec.model_revision,
            model_directory=spec.model_directory,
            expected_weight_sha256=spec.weight_sha256,
        )
    except ModelLocationError as exc:
        raise CandidateRuntimeError(str(exc)) from exc


def _model_evidence(spec: ProgramSpec, source: ModelSource) -> dict[str, object]:
    return {
        "model_id": spec.model_id,
        "revision": spec.model_revision,
        "weight_sha256": source.weight_sha256,
        "weight_size_bytes": source.weight_size_bytes,
        "snapshot_path": source.reference,
        "local_files_only": source.local_files_only,
    }


def verify_model_snapshot(spec: ProgramSpec) -> dict[str, object]:
    """Return fixed identity evidence; local overrides are hash-authenticated."""

    return _model_evidence(spec, _resolve_model(spec))


def extract_prediction(raw_text: str, spec: ProgramSpec) -> str:
    if not isinstance(raw_text, str):
        raise TypeError("raw model text must be a string")
    stripped = raw_text.strip()
    strategy = spec.extraction_strategy
    prediction = stripped
    final_matches = list(_FINAL_PATTERN.finditer(stripped))
    if strategy == "final_tag":
        if len(final_matches) == 1:
            match = final_matches[0]
            if not stripped[match.end():].strip() and match.group(1).strip():
                prediction = match.group(1).strip()
    elif strategy == "last_nonempty_line":
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        prediction = lines[-1] if lines else ""
    elif strategy == "boxed_or_final":
        if len(final_matches) == 1 and final_matches[0].group(1).strip():
            prediction = final_matches[0].group(1).strip()
        else:
            boxed = _BOXED_PATTERN.findall(stripped)
            prediction = boxed[-1].strip() if boxed else stripped
    elif strategy != "direct":
        raise CandidateRuntimeError("unapproved extraction strategy")

    post = spec.postprocessing_strategy
    if post == "strip":
        return prediction.strip()
    if post == "normalize_whitespace":
        return " ".join(prediction.split())
    if post == "strip_terminal_punctuation":
        return prediction.strip().rstrip(".;")
    if post == "normalize_numeric":
        value = prediction.strip()
        if _NUMBER_PATTERN.fullmatch(value):
            return value.replace(",", "")
        return value
    raise CandidateRuntimeError("unapproved postprocessing strategy")


def _candidate_messages(
    spec: ProgramSpec, image_reference: object, question: str,
) -> list[dict[str, object]]:
    """Return the exact renderer-runtime chat message structure."""
    image_content: dict[str, object] = {
        "type": "image", "image": image_reference,
    }
    if spec.preprocessing_option != "qwen_default":
        image_content.update({
            "min_pixels": spec.min_pixels, "max_pixels": spec.max_pixels,
        })
    messages: list[dict[str, object]] = []
    if spec.system_instruction.strip():
        messages.append({
            "role": "system",
            "content": [{
                "type": "text", "text": spec.system_instruction.strip(),
            }],
        })
    messages.append({
        "role": "user",
        "content": [
            image_content,
            {"type": "text", "text": spec.render_prompt(question)},
        ],
    })
    return messages


def _candidate_processor_kwargs(
    rendered: object, image_inputs: object, video_inputs: object,
) -> dict[str, object]:
    """Return the exact renderer-runtime processor keyword arguments."""
    return {
        "text": [rendered], "images": image_inputs,
        "videos": video_inputs, "padding": True, "return_tensors": "pt",
    }


def _candidate_generation_kwargs(
    spec: ProgramSpec, stopping_criteria: object | None,
) -> dict[str, object]:
    """Return the exact renderer-runtime greedy generation arguments."""
    result: dict[str, object] = {
        "max_new_tokens": spec.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "use_cache": True,
    }
    if stopping_criteria is not None:
        result["stopping_criteria"] = stopping_criteria
    return result


def _assert_single_visible_gpu(torch: Any) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if re.fullmatch(r"[0-9]+", visible) is None:
        raise CandidateRuntimeError(
            "CUDA_VISIBLE_DEVICES must select exactly one physical device"
        )
    count = int(torch.cuda.device_count())
    if count != 1:
        raise CandidateRuntimeError(f"expected one visible CUDA device, found {count}")
    current = int(torch.cuda.current_device())
    if current != 0:
        raise CandidateRuntimeError(f"expected logical cuda:0, found cuda:{current}")
    return int(visible)


class TrustedTargetEngine:
    """One fixed-model GPU engine; all variation comes from validated ProgramSpec."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._helpers: tuple[Any, Any, Any, Any] | None = None
        self.snapshot_evidence: dict[str, object] | None = None
        self.gpu_scope_evidence: dict[str, object] | None = None

    def _load(self, spec: ProgramSpec) -> None:
        if spec.model_id != self.model_id:
            raise CandidateRuntimeError("engine model and candidate model differ")
        if self._model is not None:
            return
        import torch
        from PIL import Image, ImageFilter, ImageOps
        from qwen_vl_utils import process_vision_info
        from transformers import (
            AutoProcessor,
            Qwen3VLForConditionalGeneration,
            StoppingCriteria,
            StoppingCriteriaList,
        )

        physical_gpu_index = _assert_single_visible_gpu(torch)
        self.gpu_scope_evidence = {
            "physical_gpu_index": physical_gpu_index,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "visible_device_count": 1,
            "logical_device": "cuda:0",
        }
        source = _resolve_model(spec)
        evidence = _model_evidence(spec, source)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            source.reference,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            **source.pretrained_kwargs,
        )
        model.eval()
        if str(next(model.parameters()).device) != "cuda:0":
            raise CandidateRuntimeError("target model is not on logical cuda:0")
        processor = AutoProcessor.from_pretrained(
            source.reference,
            use_fast=True,
            **source.pretrained_kwargs,
        )
        self._model = model
        self._processor = processor
        self._torch = torch
        self._helpers = (
            process_vision_info,
            StoppingCriteria,
            StoppingCriteriaList,
            (Image, ImageFilter, ImageOps),
        )
        self.snapshot_evidence = evidence
    def prepare(self, spec: ProgramSpec) -> None:

        """Load and synchronize the fixed snapshot outside measurement."""

        self._load(spec)
        assert self._torch is not None
        self._torch.cuda.synchronize()



    def infer(
        self,
        spec: ProgramSpec,
        *,
        image_bytes: bytes,
        question: str,
    ) -> InferenceMeasurement:
        if spec.model_id != self.model_id:
            raise CandidateRuntimeError("candidate crossed target model boundary")
        if type(image_bytes) is not bytes or not image_bytes:
            raise CandidateRuntimeError("image input must be nonempty bytes")
        if len(image_bytes) > 32 * 1024 * 1024:
            raise CandidateRuntimeError("image input exceeds 32 MiB")
        self.prepare(spec)
        started = time.perf_counter()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None
        assert self._helpers is not None
        model = self._model
        processor = self._processor
        torch = self._torch
        process_vision_info, stopping_base, stopping_list, pil_helpers = self._helpers
        image_type, image_filter, image_ops = pil_helpers

        try:
            image = image_type.open(io.BytesIO(image_bytes))
            image.load()
        except Exception as exc:
            raise CandidateRuntimeError("image bytes are not a valid image") from exc
        if spec.preprocessing_option == "qwen_default":
            prepared_image = image
        elif spec.preprocessing_option == "rgb":
            prepared_image = image.convert("RGB")
        elif spec.preprocessing_option == "autocontrast_rgb":
            prepared_image = image_ops.autocontrast(image.convert("RGB"))
        elif spec.preprocessing_option == "sharpen_rgb":
            prepared_image = image.convert("RGB").filter(image_filter.SHARPEN)
        else:
            raise CandidateRuntimeError("preprocessing option is not approved")

        messages = _candidate_messages(spec, prepared_image, question)
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(**_candidate_processor_kwargs(
            rendered, image_inputs, video_inputs
        )).to("cuda:0")
        preprocessed_at = time.perf_counter()

        tokenizer = getattr(processor, "tokenizer", processor)
        closing_ids = tokenizer.encode("</final>", add_special_tokens=False)

        class StopOnFinalTag(stopping_base):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                del scores, kwargs
                if not closing_ids or input_ids.shape[-1] < len(closing_ids):
                    return False
                return input_ids[0, -len(closing_ids):].tolist() == closing_ids

        stopping = (
            stopping_list([StopOnFinalTag()])
            if spec.extraction_strategy in {"final_tag", "boxed_or_final"}
            and closing_ids
            else None
        )
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        generation_started = time.perf_counter()
        generate_kwargs = _candidate_generation_kwargs(spec, stopping)
        with torch.inference_mode():
            generated = model.generate(**inputs, **generate_kwargs)
        torch.cuda.synchronize()
        generation_finished = time.perf_counter()
        input_length = int(inputs.input_ids.shape[-1])
        completion = generated[0, input_length:]
        generated_count = int(completion.numel())
        decoded_values = processor.batch_decode(
            [completion],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(decoded_values) != 1 or not isinstance(decoded_values[0], str):
            raise CandidateRuntimeError("decoder did not return one string")
        raw_text = decoded_values[0]
        prediction = extract_prediction(raw_text, spec)

        eos_values = getattr(model.generation_config, "eos_token_id", None)
        if isinstance(eos_values, int):
            eos_ids = {eos_values}
        elif isinstance(eos_values, (list, tuple, set)):
            eos_ids = {int(item) for item in eos_values}
        else:
            eos_ids = set()
        completion_ids = completion.tolist()
        if closing_ids and completion_ids[-len(closing_ids):] == closing_ids:
            reason = "stop_text"
        elif completion_ids and completion_ids[-1] in eos_ids:
            reason = "eos"
        elif generated_count >= spec.max_new_tokens:
            reason = "length"
        else:
            reason = "other"
        finished = time.perf_counter()
        return InferenceMeasurement(
            prediction=prediction,
            raw_text=raw_text,
            input_token_count=input_length,
            generated_token_count=generated_count,
            termination_reason=reason,
            preprocessing_seconds=preprocessed_at - started,
            generation_seconds=generation_finished - generation_started,
            total_seconds=finished - started,
            peak_memory_bytes=int(torch.cuda.max_memory_allocated(0)),
        )


class CandidateRuntime:
    """Lazy final-program adapter preserving vlm_inference(image_path, question)."""

    def __init__(self, mapping: Mapping[str, object]) -> None:
        try:
            self.spec = ProgramSpec(**dict(mapping))
        except (TypeError, ValueError) as exc:
            raise CandidateRuntimeError("renderer supplied an invalid spec") from exc
        self._engine: TrustedTargetEngine | None = None
        self.last_generation_metadata: dict[str, object] | None = None

    def infer_bytes(
        self,
        image_bytes: bytes,
        question: str,
    ) -> InferenceMeasurement:
        if self._engine is None:
            self._engine = TrustedTargetEngine(self.spec.model_id)
        return self._engine.infer(
            self.spec,
            image_bytes=image_bytes,
            question=question,
        )

    def vlm_inference(self, image_path: str, question: str) -> str:
        if not isinstance(image_path, str) or not image_path:
            raise CandidateRuntimeError("image_path must be nonempty text")
        resolved = resolve_readable_image(image_path)
        measurement = self.infer_bytes(resolved.read_bytes(), question)
        self.last_generation_metadata = measurement.to_mapping()
        return measurement.prediction
