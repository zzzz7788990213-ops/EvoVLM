from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageURLChunk, ImageChunk
from mistral_common.protocol.instruct.request import ChatCompletionRequest

from PIL import Image
from tqdm import tqdm

def generate_response(queries, model_path):
    tokenizer = MistralTokenizer.from_file(f"{model_path}/tekken.json")
    model = Transformer.from_folder(model_path)
    for k in tqdm(queries):
        query = queries[k]['question']
        image = queries[k]["figure_path"]
        image = Image.open(image).convert('RGB')
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=[ImageChunk(image=image), TextChunk(text=query)])])
        encoded = tokenizer.encode_chat_completion(completion_request)
        images = encoded.images
        tokens = encoded.tokens
        out_tokens, _ = generate([tokens], model, images=[images], max_tokens=1024, temperature=0., eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
        response = tokenizer.decode(out_tokens[0])
        queries[k]['response'] = response
