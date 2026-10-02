"""Naive Qwen3-VL-2B-Thinking baseline derived from starting_scripts.py."""
from baseline_common import (
    ATTENTION_IMPLEMENTATION,
    BATCH_SIZE,
    DTYPE,
    EFFECTIVE_GENERATION_CONFIG,
    MAX_NEW_TOKENS,
    QUANTIZATION,
    make_inference,
)

MODEL_ID = "Qwen/Qwen3-VL-2B-Thinking"
MODEL_REVISION = "d4f1d9f1b0107e12699c3c83d7baacd07fd0abd3"
PROCESSOR_REVISION = MODEL_REVISION
MODEL_WEIGHT_SHA256 = "d889ae8b5bffcfce115bc64e47601b1e90ce44b1d9f40ed78e620fd22417d888"

vlm_inference = make_inference(
    MODEL_ID,
    MODEL_REVISION,
    "Qwen3-VL-2B-Thinking",
    MODEL_WEIGHT_SHA256,
)
