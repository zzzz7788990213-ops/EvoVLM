# Adapted from https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5
# Part of V2.6 implementation is adapted directly from the authors
# This has support for MiniCPM V2 and V2.5, and V2.6

from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from PIL import Image
import torch
import random
import numpy as np
import math

def generate_response(queries, model_path, use_cot=False, random_upsize=False, seed=0):
    if use_cot or random_upsize:
        assert "MiniCPM-V2_6" in model_path, "cot and upsize functionalities are provided by the paper's authors"
    if random_upsize:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # sdpa attn impl for v2.6, default for 2 and 2.5
    if "MiniCPM-V-2_6" in model_path:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    else:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    for k in tqdm(queries):
        query = queries[k]['question']
        image = Image.open(queries[k]["figure_path"]).convert('RGB')
        if model_path.endswith("MiniCPM-V-2"):
            msgs = [{'role': 'user', 'content': query}]
            res, context, _ = model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=False,
                temperature=0.0,
                top_p=1.0,
            )
        # for 2.5
        elif model_path.endswith("MiniCPM-Llama3-V-2_5"):
            msgs = [{'role': 'user', 'content': query}]
            res = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=False,
                temperature=0.0,
                top_p=1.0,
            )
        # for 2.6 (code is adapted from authors directly)
        elif model_path.endswith("MiniCPM-V-2_6"):
            if random_upsize:
                img_width, img_height = image.width, image.height
                if (img_width * img_height) < (1344 * 1344):
                    ratio = math.sqrt((1344 * 1344) / (img_width * img_height))
                    max_img_width = int(img_width * ratio)
                    new_img_width = random.randint(img_width, max_img_width)
                    new_img_height = int(new_img_width / img_width * img_height)
                    image = image.resize((new_img_width, new_img_height))
            system_cot_prompt = '''Based on the following image, please first give your understanding of the following question, then perform careful reasoning, and finally give the final answer.'''
            msgs = [{'role': 'user', 'content': [image, query] if not use_cot else [system_cot_prompt, image, query]}]
            res = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                max_inp_length=8192,
                sampling=False,
                max_new_tokens=2048,
                num_beams=3
            )
        else:
            raise NotImplementedError(f"Model path {model_path} not supported") 
        queries[k]['response'] = res
