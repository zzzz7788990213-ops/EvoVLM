import base64
import requests

def get_client_model(model_path, api_key):
    assert api_key is not None, "API key is required for using GPT"
    assert model_path is not None, "Model name is required for using GPT"
    model = model_path
    client = None
    return client, model

def generate_response(image_path, query, model, media_type="image/jpeg", api_key=None, client=None, random_baseline=False):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }
    if not random_baseline:
        payload = {
        "model": model,
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": query
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 42
        }
    else:
        payload = {
        "model": model,
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": query
                }
            ]
            }
        ],
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 42
        }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response = response.json()
    return response['choices'][0]['message']['content']