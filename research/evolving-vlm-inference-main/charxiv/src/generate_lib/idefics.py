# Adapted from https://huggingface.co/HuggingFaceM4/idefics2-8b
# This has support for all the IDEFICS2/3 models
from transformers.image_utils import load_image
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm

def generate_response(queries, model_path):
    model = AutoModelForVision2Seq.from_pretrained(model_path).to('cuda')
    processor = AutoProcessor.from_pretrained(model_path)
    for k in tqdm(queries):
        query = queries[k]['question']
        image = load_image(queries[k]["figure_path"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{query}"},
                ]
            }  
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        generated_ids = model.generate(**inputs, max_new_tokens=500, do_sample=False)
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)
        response = response[0].split("Assistant: ")[-1] # get the response from the assistant
        queries[k]['response'] = response
