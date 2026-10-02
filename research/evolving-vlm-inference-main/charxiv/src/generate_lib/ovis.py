# Adapted from https://huggingface.co/AIDC-AI/Ovis1.5-Llama3-8B
# This has support for the Ovis model series

import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from tqdm import tqdm

def generate_response(queries, model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 multimodal_max_length=8192,
                                                 trust_remote_code=True).cuda()
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    conversation_formatter = model.get_conversation_formatter()

    for k in tqdm(queries):
        query = queries[k]['question']
        image = queries[k]["figure_path"]
        image = Image.open(image).convert('RGB')
        query = f'<image>\n{query}'
        prompt, input_ids = conversation_formatter.format_query(query)
        input_ids = torch.unsqueeze(input_ids, dim=0).to(device=model.device)
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id).to(device=model.device)
        pixel_values = [visual_tokenizer.preprocess_image(image).to(
            dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            response = text_tokenizer.decode(output_ids, skip_special_tokens=True)

        queries[k]['response'] = response
