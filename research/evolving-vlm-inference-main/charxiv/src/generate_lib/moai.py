# Adapted from https://github.com/ByungKwanLee/MoAI/blob/master/demo.py
# This has support for the MoAI model

import os
vlm_codebase = os.environ['VLM_CODEBASE_DIR']

import sys
sys.path.append(vlm_codebase + '/MoAI')
sys.path.append(vlm_codebase + '/MoAI/moai/sgg"')

from moai.load_moai import prepare_moai
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor

import tqdm
from PIL import Image
import torch

def generate_response(queries, model_path):
    moai_model, moai_processor, seg_model, seg_processor, od_model, od_processor, sgg_model, ocr_model \
    = prepare_moai(moai_path=model_path, bits=4, grad_ckpt=False, lora=False, dtype='fp16')
    for k in tqdm(queries):
        query = queries[k]['question']
        image = Resize(size=(490, 490), antialias=False)(pil_to_tensor(Image.open(queries[k]["figure_path"])))
        moai_inputs = moai_model.demo_process(image=image, 
                                    prompt=query, 
                                    processor=moai_processor,
                                    seg_model=seg_model,
                                    seg_processor=seg_processor,
                                    od_model=od_model,
                                    od_processor=od_processor,
                                    sgg_model=sgg_model,
                                    ocr_model=ocr_model,
                                    device='cuda:0')
        with torch.inference_mode():
            generate_ids = moai_model.generate(**moai_inputs, do_sample=False, temperature=0.0, top_p=1.0, max_new_tokens=256, use_cache=True)
            answer = moai_processor.batch_decode(generate_ids, skip_special_tokens=True)[0].split('[U')[0]
        queries[k]['response'] = answer
