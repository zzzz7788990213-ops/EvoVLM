# Adapted from https://huggingface.co/google/paligemma-3b-pt-896
# This has support for the PaliGemma model

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch
from tqdm import tqdm

def generate_response(queries, model_path):
    model_id = model_path
    device = "cuda:0"
    dtype = torch.bfloat16

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        revision="bfloat16",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    for k in tqdm(queries):
        image_path = queries[k]['figure_path']
        prompt = queries[k]['question']
        image = Image.open(image_path)
        model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
            decoded = processor.decode(generation, skip_special_tokens=True)
            queries[k]['response'] = decoded
