from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
from tqdm import tqdm

def generate_response(queries, model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                device_map="cuda",
                                                trust_remote_code=True,
                                                torch_dtype="auto",
                                                _attn_implementation='flash_attention_2')
    processor = AutoProcessor.from_pretrained(model_path,
                                                trust_remote_code=True,
                                                num_crops=16)
    for k in tqdm(queries):
        query = queries[k]['question']
        image = queries[k]["figure_path"]
        image = Image.open(image).convert('RGB')
        images = [image]
        query = f"<|image_1|>\n{query}"
        messages = [
            {"role": "user", "content": query}
        ]
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")
        generation_args = {
            "max_new_tokens": 1000,
            "temperature": 0.0,
            "do_sample": False
        }
        generate_ids = model.generate(**inputs,
                                    eos_token_id=processor.tokenizer.eos_token_id,
                                    **generation_args)
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)[0]
        print(response)
        queries[k]['response'] = response

