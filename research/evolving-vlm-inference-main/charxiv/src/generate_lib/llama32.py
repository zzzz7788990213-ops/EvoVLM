import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, AutoProcessor

def generate_response(queries, model_path):
    model = MllamaForConditionalGeneration.from_pretrained(model_path,
                                                           torch_dtype=torch.bfloat16,
                                                           device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)

    for k in tqdm(queries):
        query = queries[k]['question']
        image = queries[k]["figure_path"]
        image = Image.open(image).convert('RGB')
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": query}
            ]}
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=1024)
        response = processor.decode(output[0])
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[1].replace("<|eot_id|>", "").strip()
        queries[k]['response'] = response
