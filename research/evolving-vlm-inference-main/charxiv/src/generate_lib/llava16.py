# Adapted from https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf and https://huggingface.co/llava-hf/llava-v1.6-34b-hf
# This has support for all Llava 1.6 Mistral 7B and Yi 34B models

import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from tqdm import tqdm
from PIL import Image

def generate_responses(queries, model_path):
    # taken from: https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf
    processor = LlavaNextProcessor.from_pretrained(model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_path, 
        torch_dtype=torch.float16, low_cpu_mem_usage=True).to('cuda:0').eval()
    if 'llava-v1.6-mistral-7b-hf' in model_path:
        max_tokens, prompt_prefix = 1000, "[/INST]"
    elif 'llava-v1.6-34b-hf' in model_path:
        max_tokens, prompt_prefix = 100, "<|im_start|> assistant"
    for k in tqdm(queries):
        image = Image.open(queries[k]["figure_path"])
        if 'llava-v1.6-mistral-7b-hf' in model_path:
            prompt = f"[INST] <image>\n{queries[k]['question']} [/INST]"
        elif 'llava-v1.6-34b-hf' in model_path:
            prompt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{queries[k]['question']}<|im_end|><|im_start|>assistant\n"
        try:
            inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
            output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            response = processor.decode(output[0], skip_special_tokens=True)
            response = response.split(prompt_prefix)[1].strip() # remove the prompt
        except:
            response = "Generation Error"
        queries[k]['response'] = response
