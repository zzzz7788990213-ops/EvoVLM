# Adapted from https://github.com/vis-nlp/UniChart/blob/main/README.md
# This has support for the UniChart model

from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
from tqdm import tqdm

def generate_response(queries, model_path):
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    processor = DonutProcessor.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for k in tqdm(queries):
        image_path = queries[k]['figure_path']
        input_prompt = queries[k]['question']
        input_prompt = f"<chartqa> {input_prompt} <s_answer>"
        image = Image.open(image_path).convert("RGB")
        decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        pixel_values = processor(image, return_tensors="pt").pixel_values

        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=4,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = sequence.split("<s_answer>")[1].strip()
        queries[k]['response'] = sequence
