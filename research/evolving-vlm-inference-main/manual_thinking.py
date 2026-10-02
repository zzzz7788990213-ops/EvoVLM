"""Manual Qwen3-VL-2B-Thinking submission interface."""

from manual_common import (
    ATTENTION_IMPLEMENTATION,
    DTYPE,
    QUANTIZATION,
    THINKING_SPEC,
    make_manual_inference,
)

MODEL_ID = THINKING_SPEC.model_id
MODEL_REVISION = THINKING_SPEC.revision
PROCESSOR_REVISION = THINKING_SPEC.revision
MODEL_WEIGHT_SHA256 = THINKING_SPEC.weight_sha256
EFFECTIVE_GENERATION_CONFIG = THINKING_SPEC.effective_generation_config

vlm_inference = make_manual_inference(THINKING_SPEC)
