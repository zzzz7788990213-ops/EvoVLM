# Adapted from https://github.com/NVlabs/VILA/blob/main/predict.py
# This has support for the VILA 40B model

import os
vlm_codebase = os.environ['VLM_CODEBASE_DIR']

import sys
sys.path.append(vlm_codebase + '/VILA')

from io import BytesIO
import requests
import torch
from PIL import Image

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                            DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                            IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from tqdm import tqdm

def load_image(image_file):
        image = Image.open(image_file).convert("RGB")
        return image
    
def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def generate_response(queries, model_path):
    disable_torch_init()
    
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, None)
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    for k in tqdm(queries):
        images = load_images([queries[k]['figure_path']])
        qs = queries[k]['question']
        if model.config.mm_use_im_start_end:
            qs = (image_token_se + "\n") * len(images) + qs
        else:
            qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs
        conv_mode = "hermes-2"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[
                    images_tensor,
                ],
                do_sample=False,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        queries[k]['response'] = outputs
