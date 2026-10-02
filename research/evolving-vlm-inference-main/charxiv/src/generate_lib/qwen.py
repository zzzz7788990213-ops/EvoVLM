from dashscope import MultiModalConversation
import dashscope

def get_client_model(model_path, api_key):
    assert api_key is not None, "API key is required for using Qwen"
    assert model_path is not None, "Model name is required for using Qwen"
    model = model_path
    client = None
    return client, model

def generate_response(image_path, query, model, media_type="image/jpeg", api_key=None, client=None, random_baseline=False):
    dashscope.api_key = api_key
    messages = [{
        'role': 'user',
        'content': [
            {
                'image': image_path
            },
            {
                'text': query
            }
        ]
    }]
    response = MultiModalConversation.call(model=model, messages=messages, temperature=0.0, top_p=0.99999, seed=42)
    return response.output['choices'][0]['message']['content'][0]['text']
