"""Deterministic renderer-generated VLM inference program."""

from evolution.candidate_runtime import CandidateRuntime

RENDERER_VERSION = 'phase2-local-renderer-v2'
CANDIDATE_SPEC = {'model_id': 'Qwen/Qwen3-VL-2B-Instruct',
 'system_instruction': '',
 'prompt_template': 'Answer the chart question concisely. Return only the answer '
                    'enclosed in <final> and </final>; do not add an explanation. '
                    'Question: {question}',
 'reasoning_mode': 'direct',
 'max_new_tokens': 64,
 'min_pixels': 200704,
 'max_pixels': 1003520,
 'preprocessing_option': 'qwen_default',
 'extraction_strategy': 'boxed_or_final',
 'postprocessing_strategy': 'strip',
 'dtype': 'bfloat16',
 'attention': 'sdpa',
 'batch_size': 1,
 'do_sample': False,
 'num_beams': 1,
 'num_return_sequences': 1,
 'use_cache': True}
MODEL_ID = 'Qwen/Qwen3-VL-2B-Instruct'
MODEL_REVISION = '78448d793a7eb2f7a987a1da76d464384aa1becd'
PROCESSOR_REVISION = '78448d793a7eb2f7a987a1da76d464384aa1becd'
MODEL_WEIGHT_SHA256 = '7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'
DTYPE = 'bfloat16'
ATTENTION_IMPLEMENTATION = 'sdpa'
QUANTIZATION = 'none'
EFFECTIVE_GENERATION_CONFIG = {'do_sample': False, 'num_beams': 1, 'num_return_sequences': 1, 'max_new_tokens': 64, 'use_cache': True}

_RUNTIME = CandidateRuntime(CANDIDATE_SPEC)

def vlm_inference(image_path: str, question: str) -> str:
    return _RUNTIME.vlm_inference(image_path, question)
