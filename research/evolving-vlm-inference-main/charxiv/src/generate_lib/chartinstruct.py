# Adapted from https://huggingface.co/ahmed-masry/ChartInstruct-LLama2, https://huggingface.co/ahmed-masry/ChartInstruct-FlanT5-XL
# This has support for two ChartInstruct models, LLama2 and FlanT5

from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

def generate_response(queries, model_path):
    if "LLama2" in model_path:
        print("Using LLama2 model")
        model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
    elif "FlanT5" in model_path:
        print("Using FlanT5 model")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)
    else:
        raise ValueError(f"Model {model_path} not supported")
    processor = AutoProcessor.from_pretrained(model_path)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for k in tqdm(queries):
        image_path = queries[k]['figure_path']
        input_prompt = queries[k]['question']
        input_prompt = f"<image>\n Question: {input_prompt} Answer: "

        image = Image.open(image_path).convert('RGB')
        inputs = processor(text=input_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # change type if pixel_values in inputs to fp16. 
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        if "LLama2" in model_path:
            prompt_length = inputs['input_ids'].shape[1]
        
        # move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        generate_ids = model.generate(**inputs, num_beams=4, max_new_tokens=512)
        output_text = processor.batch_decode(generate_ids[:, prompt_length:] \
            if 'LLama2' in model_path else generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(output_text)
        queries[k]['response'] = output_text
