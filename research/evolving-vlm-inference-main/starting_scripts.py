# EVOLVE-BLOCK-START

"""Vision Language Model inference for chart QA using Qwen3-VL-2B-Instruct or Thinking."""
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

_model = None
_processor = None
_model_name = "Qwen/Qwen3-VL-2B-Instruct" # or Qwen/Qwen3-VL-2B-Thinking


def _get_model():
    global _model, _processor
    if _model is None:
        _model = AutoModelForVision2Seq.from_pretrained(
            _model_name, torch_dtype=torch.float16, device_map="auto",
        )
        _model.eval()
    if _processor is None:
        _processor = AutoProcessor.from_pretrained(_model_name, use_fast=False)
    return _model, _processor


def vlm_inference(image_path, question="Describe this image in detail."):
    model, processor = _get_model()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    return output_text[0]

# EVOLVE-BLOCK-END
