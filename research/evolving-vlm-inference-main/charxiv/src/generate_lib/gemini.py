from PIL import Image
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time

def get_client_model(model_path, api_key):
    assert api_key is not None, "API key is required for using Gemini"
    assert model_path is not None, "Model name is required for using Gemini"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_path)
    client = None
    return client, model

def generate_response(image_path, query, model, media_type="image/jpeg", api_key=None, client=None, random_baseline=False):
    # Load from local file
    image = Image.open(image_path)
    response = model.generate_content([image, query], 
    generation_config=genai.types.GenerationConfig(
    candidate_count=1,
    max_output_tokens=1000,
    temperature=0.0,
    top_p=1.0,
    ),
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    )
    return response.text