"""Deterministic renderer-generated VLM inference program."""

from evolution.candidate_runtime import CandidateRuntime

RENDERER_VERSION = 'phase2-local-renderer-v2'
CANDIDATE_SPEC = {'model_id': 'Qwen/Qwen3-VL-2B-Thinking',
 'system_instruction': '',
 'prompt_template': 'Use only evidence visible in the scientific figure; do not invent '
                    "missing or inapplicable chart elements. Use the model's native "
                    'reasoning, then return only the concise final answer enclosed in '
                    '<final> and </final>. Question: {question}',
 'reasoning_mode': 'thinking',
 'max_new_tokens': 384,
 'min_pixels': 200704,
 'max_pixels': 1003520,
 'preprocessing_option': 'qwen_default',
 'extraction_strategy': 'final_tag',
 'postprocessing_strategy': 'strip_terminal_punctuation',
 'dtype': 'bfloat16',
 'attention': 'sdpa',
 'batch_size': 1,
 'do_sample': False,
 'num_beams': 1,
 'num_return_sequences': 1,
 'use_cache': True}
MODEL_ID = 'Qwen/Qwen3-VL-2B-Thinking'
MODEL_REVISION = 'd4f1d9f1b0107e12699c3c83d7baacd07fd0abd3'
PROCESSOR_REVISION = 'd4f1d9f1b0107e12699c3c83d7baacd07fd0abd3'
MODEL_WEIGHT_SHA256 = 'd889ae8b5bffcfce115bc64e47601b1e90ce44b1d9f40ed78e620fd22417d888'
DTYPE = 'bfloat16'
ATTENTION_IMPLEMENTATION = 'sdpa'
QUANTIZATION = 'none'
EFFECTIVE_GENERATION_CONFIG = {'do_sample': False, 'num_beams': 1, 'num_return_sequences': 1, 'max_new_tokens': 384, 'use_cache': True}

_RUNTIME = CandidateRuntime(CANDIDATE_SPEC)

def vlm_inference(image_path: str, question: str) -> str:
    return _RUNTIME.vlm_inference(image_path, question)
