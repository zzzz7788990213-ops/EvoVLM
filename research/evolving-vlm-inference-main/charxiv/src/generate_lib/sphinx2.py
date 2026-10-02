# Adapted from https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/SPHINX/README.md#single-gpu-inference
# This has support for the SPHINX 2 Llama 13B model

from SPHINX import SPHINXModel
from PIL import Image
from tqdm import tqdm

def generate_response(queries, model_path):
    model = SPHINXModel.from_pretrained(pretrained_path=model_path, with_visual=True)
    for k in tqdm(queries):
        qas = [[queries[k]['question'], None]]
        image = Image.open(queries[k]["figure_path"])
        response = model.generate_response(qas, image, max_gen_len=1024, temperature=0.0, top_p=1, seed=42)
        queries[k]['response'] = response
