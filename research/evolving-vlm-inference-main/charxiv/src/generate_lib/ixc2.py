# Adapted from https://huggingface.co/internlm/internlm-xcomposer2-4khd-7b and https://huggingface.co/internlm/internlm-xcomposer2-vl-7b
# This has support for all InternLM-XComposer2 and InternLM-XComposer2-4KHD models
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

def generate_response(queries, model_path):
    # taken from: 
    torch.set_grad_enabled(False)
    if '4khd' in model_path:
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    else:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    for k in tqdm(queries):
        query = '<ImageHere>' + queries[k]['question']
        image = queries[k]["figure_path"]
        with torch.cuda.amp.autocast():
            if '4khd' in model_path:
                # set hd_num to 16 for up to 1344^2 support: https://github.com/InternLM/InternLM-XComposer/issues/252#issuecomment-2049507385
                response, _ = model.chat(tokenizer, query=query, image=image, hd_num=16, history=[], do_sample=False)
            else:
                response, _ = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False)
        queries[k]['response'] = response
