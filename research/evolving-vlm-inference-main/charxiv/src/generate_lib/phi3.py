# Adapted from https://huggingface.co/microsoft/Phi-3-vision-128k-instruct
# This has support for the Phi 3 model

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm

def generate_response(queries, model_path):
    

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, 
                                                 torch_dtype="auto", _attn_implementation='flash_attention_2')
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    for k in tqdm(queries):
        query = queries[k]['question']
        image = queries[k]["figure_path"]
        image = Image.open(image).convert('RGB')
        messages = [{'role': 'user', 'content': f"<|image_1|>\n{query}"}]
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")
        generation_args = {
            "max_new_tokens": 500,
            "temperature": 0.0,
            "do_sample": False,
        }
        generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        queries[k]['response'] = result
