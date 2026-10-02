# Adapted from https://huggingface.co/ahmed-masry/chartgemma
# This has support for the ChartGemma model

from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch
from tqdm import tqdm

def generate_response(queries, model_path):
    # Load Model
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for k in tqdm(queries):
        image_path = queries[k]['figure_path']
        input_text = queries[k]['question']

        # Process Inputs
        image = Image.open(image_path).convert('RGB')
        inputs = processor(text=input_text, images=image, return_tensors="pt")
        prompt_length = inputs['input_ids'].shape[1]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        generate_ids = model.generate(**inputs, num_beams=4, max_new_tokens=512)
        output_text = processor.batch_decode(generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        queries[k]['response'] = output_text
