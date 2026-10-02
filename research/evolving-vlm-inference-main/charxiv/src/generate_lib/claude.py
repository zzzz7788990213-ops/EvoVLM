import anthropic
import base64
import json

def get_client_model(model_path, api_key):
    assert api_key is not None, "API key is required for using Claude"
    assert model_path is not None, "Model name is required for using Claude"
    client = anthropic.Anthropic(api_key=api_key)
    model = model_path
    return client, model

def generate_response(image_path, query, model, media_type="image/jpeg", api_key=None, client=None, random_baseline=False):
    def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
    image = encode_image(image_path)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.0,
        top_p=1.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image,
                        },
                    },
                    {
                        "type": "text",
                        "text": query,
                    }
                ],
            }
        ],
    )
    message = message.json()
    message = json.loads(message)
    return message['content'][0]['text']