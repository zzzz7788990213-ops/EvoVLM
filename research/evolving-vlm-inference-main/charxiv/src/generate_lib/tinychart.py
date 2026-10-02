# Adapted from https://github.com/X-PLUG/mPLUG-DocOwl/blob/main/TinyChart/inference.ipynb
# This has support for the TinyChart model

### HEADER START ###
import os
vlm_codebase = os.environ['VLM_CODEBASE_DIR']

import sys
sys.path.append(vlm_codebase + '/mPLUG-DocOwl/TinyChart')
### HEADER END ###

from tqdm import tqdm
import torch
from PIL import Image
from tinychart.model.builder import load_pretrained_model
from tinychart.mm_utils import get_model_name_from_path
from tinychart.eval.run_tiny_chart import inference_model
from tinychart.eval.eval_metric import parse_model_output, evaluate_cmds

def generate_response(queries, model_path):
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device="cuda" # device="cpu" if running on cpu
    )
    for k in tqdm(queries):
        img_path = queries[k]['figure_path']
        text = queries[k]['question']
        response = inference_model([img_path], text, model, tokenizer, image_processor, context_len, conv_mode="phi", max_new_tokens=1024)
        # print(response)
        try:
            response = evaluate_cmds(parse_model_output(response))
            print('Command successfully executed')
            print(response)
        except Exception as e:
            # if message is NameError: name 'Answer' is not defined, then skip
            if "Error: name 'Answer' is not defined" in str(e):
                response = response
            else:
                print('Error:', e)
                response = response
        response = str(response)
        queries[k]['response'] = response
