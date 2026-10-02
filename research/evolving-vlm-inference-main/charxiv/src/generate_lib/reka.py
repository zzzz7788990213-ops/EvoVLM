import reka
import base64
import os

def get_client_model(model_path, api_key):
    assert api_key is not None, "API key is required for using Reka"
    assert model_path is not None, "Model name is required for using Reka"
    os.environ["REKA_API_KEY"] = api_key
    model = model_path
    client = None
    return client, model

def generate_response(image_path, query, model, media_type="image/jpeg", api_key=None, client=None, random_baseline=False):
    reka.API_KEY = api_key
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    response = reka.chat(
        query,
        media_filename=image_path,
        model_name=model,
        request_output_len=1024,
        temperature=0.0,
        runtime_top_p=1.0,
    )
    return response['text']
