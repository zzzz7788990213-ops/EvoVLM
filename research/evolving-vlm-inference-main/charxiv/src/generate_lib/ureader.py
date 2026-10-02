# adapted from https://github.com/X-PLUG/mPLUG-DocOwl/blob/main/UReader/pipeline/interface.py
# This has support for the UReader model

### HEADER START ###
import os
vlm_codebase = os.environ['VLM_CODEBASE_DIR']

import sys
sys.path.append(vlm_codebase + '/mPLUG-DocOwl/UReader')

UREADER_DIR = os.path.join(vlm_codebase, 'mPLUG-DocOwl/UReader/')
### HEADER END ###

import os
import torch
from sconf import Config
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
import torch
from pipeline.data_utils.processors.builder import build_processors
from pipeline.data_utils.processors import *
from pipeline.utils import add_config_args, set_args
import argparse

from PIL import Image
from tqdm import tqdm

def generate_response(queries, model_path):
    config = Config("{}configs/sft/release.yaml".format(UREADER_DIR))
    args = argparse.ArgumentParser().parse_args([])
    add_config_args(config, args)
    set_args(args)
    model = MplugOwlForConditionalGeneration.from_pretrained(
        model_path,
    )
    model.eval()
    model.cuda()
    model.half()
    image_processor = build_processors(config['valid_processors'])['sft']
    tokenizer = MplugOwlTokenizer.from_pretrained(model_path)
    processor = MplugOwlProcessor(image_processor, tokenizer)

    for k in tqdm(queries):
        image_path = queries[k]['figure_path']
        images = [Image.open(image_path).convert('RGB')]
        question = f"Human: <image>\nHuman: {queries[k]['question']}\nAI: "
        inputs = processor(text=question, images=images, return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = model.generate(**inputs)
        sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        print('model_answer:', sentence)
        queries[k]['response'] = sentence
