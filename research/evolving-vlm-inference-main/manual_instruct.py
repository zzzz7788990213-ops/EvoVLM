"""Manual Qwen3-VL-2B-Instruct submission interface."""

from manual_common import (
    ATTENTION_IMPLEMENTATION,
    DTYPE,
    INSTRUCT_SPEC,
    QUANTIZATION,
    make_manual_inference,
)

MODEL_ID = INSTRUCT_SPEC.model_id
MODEL_REVISION = INSTRUCT_SPEC.revision
PROCESSOR_REVISION = INSTRUCT_SPEC.revision
MODEL_WEIGHT_SHA256 = INSTRUCT_SPEC.weight_sha256
EFFECTIVE_GENERATION_CONFIG = INSTRUCT_SPEC.effective_generation_config

vlm_inference = make_manual_inference(INSTRUCT_SPEC)
