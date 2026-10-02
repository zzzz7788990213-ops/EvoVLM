# Adapted from https://huggingface.co/THUDM/cogagent-vqa-hf
# This has support for the CogAgent model

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm

def generate_response(queries, model_path):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_type = torch.bfloat16
    tokenizer_path, model_path = model_path.split('::')
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            load_in_4bit=False,
            trust_remote_code=True
        ).to('cuda').eval()
        
    for k in tqdm(queries):
        image_path = queries[k]['figure_path']
        image = Image.open(image_path).convert('RGB')
        query = f"Human:{queries[k]['question']}"
        history = []
    
        input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image])
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]],
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

        # add any transformers params here.
        gen_kwargs = {"max_length": 2048,
                    "temperature": 0.9,
                    "do_sample": False}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]
            print("\nCog:", response)
        print('model_answer:', response)
        queries[k]['response'] = response
