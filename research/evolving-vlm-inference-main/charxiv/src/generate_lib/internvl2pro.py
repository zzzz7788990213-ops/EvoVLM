

import requests

def get_client_model(model_path, api_key):
    assert api_key is not None, "API key is required for using GPT"
    assert model_path is not None, "Model name is required for using GPT"
    model = model_path
    client = None
    return client, model

def generate_response(image_path, query, model, media_type="image/jpeg", api_key=None, client=None, random_baseline=False):

    url = "http://101.132.98.120:11005/chat/" 


    file_paths = [
        image_path
    ]
    question = query 

    files = [('files', open(file_path, 'rb')) for file_path in file_paths]
    data = {
        'question': question,
        'api_key': api_key
    }

    try:
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            print("Response:", response.json().get("response", "No response key found in the JSON."))
            return response.json().get("response", "No response key found in the JSON.")
        else:
            print("Error:", response.status_code, response.text)
            return "Error in generating response."
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return "Error in generating response."
