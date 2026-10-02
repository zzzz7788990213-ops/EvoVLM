# Adapted from https://github.com/Yuliang-Liu/Monkey/blob/main/demo_textmonkey.py
# This has support for the TextMonkey model

import os
vlm_codebase = os.environ['VLM_CODEBASE_DIR']

import sys
sys.path.append(vlm_codebase + '/Monkey')

import re
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from monkey_model.modeling_textmonkey import TextMonkeyLMHeadModel
from monkey_model.tokenization_qwen import QWenTokenizer
from monkey_model.configuration_monkey import MonkeyConfig
from tqdm import tqdm

def generate_response(queries, model_path):
    device_map = "cuda"
    # Create model
    config = MonkeyConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    model = TextMonkeyLMHeadModel.from_pretrained(model_path,
        config=config,
        device_map=device_map, trust_remote_code=True).eval()
    tokenizer = QWenTokenizer.from_pretrained(model_path,
                                                trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.IMG_TOKEN_SPAN = config.visual["n_queries"]

    for k in tqdm(queries):
        input_image = queries[k]['figure_path']
        input_str = queries[k]['question']
        input_str = f"<img>{input_image}</img> {input_str}"
        input_ids = tokenizer(input_str, return_tensors='pt', padding='longest')

        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids
        
        pred = model.generate(
        input_ids=input_ids.cuda(),
        attention_mask=attention_mask.cuda(),
        do_sample=False,
        num_beams=1,
        max_new_tokens=2048,
        min_new_tokens=1,
        length_penalty=1,
        num_return_sequences=1,
        output_hidden_states=True,
        use_cache=True,
        pad_token_id=tokenizer.eod_id,
        eos_token_id=tokenizer.eod_id,
        )
        response = tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=False).strip()
        image = Image.open(input_image).convert("RGB").resize((1000,1000))
        font = ImageFont.truetype('NimbusRoman-Regular.otf', 22)
        bboxes = re.findall(r'<box>(.*?)</box>', response, re.DOTALL)
        refs = re.findall(r'<ref>(.*?)</ref>', response, re.DOTALL)
        if len(refs)!=0:
            num = min(len(bboxes), len(refs))
        else:
            num = len(bboxes)
        for box_id in range(num):
            bbox = bboxes[box_id]
            matches = re.findall( r"\((\d+),(\d+)\)", bbox)
            draw = ImageDraw.Draw(image)
            point_x = (int(matches[0][0])+int(matches[1][0]))/2
            point_y = (int(matches[0][1])+int(matches[1][1]))/2
            point_size = 8
            point_bbox = (point_x - point_size, point_y - point_size, point_x + point_size, point_y + point_size)
            draw.ellipse(point_bbox, fill=(255, 0, 0))
            if len(refs)!=0:
                text = refs[box_id]
                text_width, text_height = font.getsize(text)
                draw.text((point_x-text_width//2, point_y+8), text, font=font, fill=(255, 0, 0))
        response = tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
        output_str = response
        print(f"Answer: {output_str}")
        queries[k]['response'] = output_str
