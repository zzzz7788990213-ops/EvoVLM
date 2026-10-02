# Adapted from https://github.com/dvlab-research/MGM/blob/main/mgm/serve/cli.py
# This has support for MGM trained on Llama 3 8B and Yi 34B model for the HD version

from mgm.model.builder import load_pretrained_model
from mgm.utils import disable_torch_init
from mgm.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images
from mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mgm.conversation import conv_templates

import os
import torch
from tqdm import tqdm
from PIL import Image

def get_image_input_from_path(image, model, image_processor):
    image = Image.open(image)
    if hasattr(model.config, 'image_size_aux'):
        if not hasattr(image_processor, 'image_size_raw'):
            image_processor.image_size_raw = image_processor.crop_size.copy()
        image_processor.crop_size['height'] = model.config.image_size_aux
        image_processor.crop_size['width'] = model.config.image_size_aux
        image_processor.size['shortest_edge'] = model.config.image_size_aux
    
    image_tensor = process_images([image], image_processor, model.config)[0]
    
    image_grid = getattr(model.config, 'image_grid', 1)
    if hasattr(model.config, 'image_size_aux'):
        raw_shape = [image_processor.image_size_raw['height'] * image_grid, 
                    image_processor.image_size_raw['width'] * image_grid]
        image_tensor_aux = image_tensor
        image_tensor = torch.nn.functional.interpolate(image_tensor[None], 
                                                    size=raw_shape, 
                                                    mode='bilinear', 
                                                    align_corners=False)[0]
    else:
        image_tensor_aux = []

    if image_grid >= 2:            
        raw_image = image_tensor.reshape(3, 
                                        image_grid,
                                        image_processor.image_size_raw['height'],
                                        image_grid,
                                        image_processor.image_size_raw['width'])
        raw_image = raw_image.permute(1, 3, 0, 2, 4)
        raw_image = raw_image.reshape(-1, 3,
                                    image_processor.image_size_raw['height'],
                                    image_processor.image_size_raw['width'])
        
        if getattr(model.config, 'image_global', False):
            global_image = image_tensor
            if len(global_image.shape) == 3:
                global_image = global_image[None]
            global_image = torch.nn.functional.interpolate(global_image, 
                                                    size=[image_processor.image_size_raw['height'],
                                                        image_processor.image_size_raw['width']], 
                                                    mode='bilinear', 
                                                    align_corners=False)
            # [image_crops, image_global]
            raw_image = torch.cat([raw_image, global_image], dim=0)
        image_tensor = raw_image.contiguous()
    
    images = image_tensor[None].to(dtype=model.dtype, device='cuda', non_blocking=True)
    images_aux = image_tensor_aux[None].to(dtype=model.dtype, device='cuda', non_blocking=True) if len(image_tensor_aux)>0 else None
    return images, images_aux, 


def generate_response(queries, model_path):
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,
                                                                           load_8bit=False)
    for k in tqdm(queries):
        query = queries[k]['question']
        if getattr(model.config, 'mm_use_im_start_end', False):
            query = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + query
        else:
            query = DEFAULT_IMAGE_TOKEN + '\n' + query
        if 'MGM-8B-HD' in model_name:
            template_name = "llama_3"
        elif 'MGM-34B-HD' in model_name:
            template_name = "chatml_direct"
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        conv = conv_templates[template_name].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        try:
            images, images_aux = get_image_input_from_path(queries[k]["figure_path"], model, image_processor)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            terminators = tokenizer.eos_token_id
            if template_name == "llama_3":
                terminators = [terminators, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    images_aux=images_aux,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    max_new_tokens=1024,
                    bos_token_id=tokenizer.bos_token_id,  # Begin of sequence token
                    eos_token_id=terminators,  # End of sequence token
                    pad_token_id=tokenizer.pad_token_id,  # Pad token
                    use_cache=True,
                )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            queries[k]['response'] = outputs
        except:
            queries[k]['response'] = "Generation Error"
