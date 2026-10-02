# Adapted from https://github.com/OpenGVLab/ChartAst/blob/main/accessory/single_turn_eval.py
# This has support for the ChartAssistant model

import os
vlm_codebase = os.environ['VLM_CODEBASE_DIR']

import sys
sys.path.append(vlm_codebase + '/ChartAst/accessory')

os.environ['MP'] = '1'
os.environ['WORLD_SIZE'] = '1'

import torch
from tqdm import tqdm
import torch.distributed as dist


sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])
from fairscale.nn.model_parallel import initialize as fs_init
from model.meta import MetaModel
from util.tensor_parallel import load_tensor_parallel_model_list
from util.misc import init_distributed_mode
from PIL import Image

import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from PIL import Image
import os
import torch


class PadToSquare:
    def __init__(self, background_color):
        """
        pad an image to squre (borrowed from LLAVA, thx)
        :param background_color: rgb values for padded pixels, normalized to [0, 1]
        """
        self.bg_color = tuple(int(x * 255) for x in background_color)

    def __call__(self, img: Image.Image):
        width, height = img.size
        if width == height:
            return img
        elif width > height:
            result = Image.new(img.mode, (width, width), self.bg_color)
            result.paste(img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(img.mode, (height, height), self.bg_color)
            result.paste(img, ((height - width) // 2, 0))
            return result

def T_padded_resize(size=448):
    t = transforms.Compose([
        PadToSquare(background_color=(0.48145466, 0.4578275, 0.40821073)),
        transforms.Resize(
            size, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    return t

def generate_response(queries, model_path):
    init_distributed_mode()
    fs_init.initialize_model_parallel(dist.get_world_size())
    model = MetaModel('llama_ens5', model_path + '/params.json', model_path + '/tokenizer.model', with_visual=True)
    print(f"load pretrained from {model_path}")
    load_tensor_parallel_model_list(model, model_path)
    model.bfloat16().cuda()
    max_gen_len = 512
    gen_t = 0.9
    top_p = 0.5

    for k in tqdm(queries):
        question = queries[k]['question']
        img_path = queries[k]['figure_path']

        prompt = f"""Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\nPlease answer my question based on the chart: {question}\n\n### Response:"""

        image = Image.open(img_path).convert('RGB')
        transform_val = T_padded_resize(448)
        image = transform_val(image).unsqueeze(0)
        image = image.cuda()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            response = model.generate([prompt], image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)
        response = response[0].split('###')[0]
        print(response)
        queries[k]['response'] = response
