# Adapted from https://huggingface.co/THUDM/glm-4v-9b
# This has support for the GLM 4v model

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def generate_response(queries, model_path):
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    for k in tqdm(queries):
        query = queries[k]['question']
        image = Image.open(queries[k]["figure_path"]).convert('RGB')
        inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                            add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                            return_dict=True)  # chat mode
        inputs = inputs.to(device)
        gen_kwargs = {"max_length": 2500, "do_sample": False, "top_k": 1}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            res = tokenizer.decode(outputs[0]).replace(' <|endoftext|>', '')
            queries[k]['response'] = res
