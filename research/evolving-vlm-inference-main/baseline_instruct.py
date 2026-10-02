"""Naive Qwen3-VL-2B-Instruct baseline derived from starting_scripts.py."""
from baseline_common import (
    ATTENTION_IMPLEMENTATION,
    BATCH_SIZE,
    DTYPE,
    EFFECTIVE_GENERATION_CONFIG,
    MAX_NEW_TOKENS,
    QUANTIZATION,
    make_inference,
)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "78448d793a7eb2f7a987a1da76d464384aa1becd"
PROCESSOR_REVISION = MODEL_REVISION
MODEL_WEIGHT_SHA256 = "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"

vlm_inference = make_inference(
    MODEL_ID,
    MODEL_REVISION,
    "Qwen3-VL-2B-Instruct",
    MODEL_WEIGHT_SHA256,
)
