# 📊 CharXiv
🏠[Home](https://charxiv.github.io/) | 🤗[Data](https://huggingface.co/datasets/princeton-nlp/CharXiv) | 🥇[Leaderboard](https://charxiv.github.io/#leaderboard) | 📄[Paper](https://arxiv.org/abs/2406.18521) | *Current Version: v1.0*

This repository contains the code to evaluate models on CharXiv from the paper [CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs](https://arxiv.org/abs/2406.18521).

*🤗 We are first-time evaluation suite builders and this codebase is released for the first-time. We are committed to improving it. If you have any questions, feel free to raise issues and/or submit pull requests for new features of bug fixes.*

### 🔊 An 80-second video teaser of CharXiv

https://github.com/princeton-nlp/CharXiv/assets/59942464/ab9b293b-8fd6-4735-b8b3-0079ee978b61

## 📰 News
**[04/14/2025]** 🚀 CharXiv has been officially adopted into frontier model evaluations for [GPT-4.1](https://openai.com/index/gpt-4-1/), [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), [InternVL2.5](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/), [Doubao-1.5-pro](https://team.doubao.com/zh/special/doubao_1_5_pro), and more!  
**[12/25/2024]** 🚀 We updated the [leaderboard](https://charxiv.github.io/#leaderboard) with the latest models: [o1](https://openai.com/o1/), [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL), [Pixtral](https://mistral.ai/news/pixtral-12b/), [InternVL 2.5](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/), [Llama 3.2 Vision](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/), [NVLM](https://nvlm-project.github.io/), [Molmo](https://molmo.org/), [Llava OneVision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/), [Phi 3.5](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/discover-the-new-multi-lingual-high-quality-phi-3-5-slms/4225280), and more!  
**[10/24/2024]** 🚀 Check out [this paper](https://arxiv.org/abs/2410.18798) which includes a detailed study on 🪜 improving the model performance on CharXiv!  
**[10/10/2024]** 🚀 CharXiv is accepted at [**NeurIPS 2024 Datasets & Benchmarks Track**](https://openreview.net/forum?id=cy8mq7QYae) and NeurIPS 2024 [Multimodal Algorithmic Reasoning Workshop](https://marworkshop.github.io/neurips24/) as a **spotlight** paper.  
**[07/26/2024]** 🚀 Upcoming this week: we'll be releasing scores for [GPT-4o-mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) as well as the largest and most capable open-weight VLM in our benchmark: [InternVL2 LLaMA-3 76B](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B). Alongside scores, we find some [interesting patterns](https://x.com/zwcolin/status/1816948825036071196) in the trend of model improvement with respect to differnet chart understanding benchmarks on X.  
**[07/24/2024]** 🚀 We released the [full evaluation pipeline](https://github.com/princeton-nlp/CharXiv) (i.e., v1.0).  
**[07/23/2024]** 🚀 We released our [evaluation results](https://huggingface.co/datasets/princeton-nlp/CharXiv/tree/main/existing_evaluations) on **all 34 MLLMs** that we have tested so far -- this includes all models' responses to CharXiv's challenging questions, scores graded by GPT-4o, as well as aggregated stats.   
**[07/14/2024]** 🚀 We further evaluated the latest [InternVL Chat V2.0 26B](https://huggingface.co/OpenGVLab/InternVL2-26B) and [Cambrian 34B models](https://huggingface.co/nyu-visionx/cambrian-34b) on CharXiv with some **State-of-the-Art results**. More analysis are [here](https://x.com/zwcolin/status/1812650435808792731).

## 👋 Introduction
Chart understanding plays a pivotal role when applying Multimodal Large Language Models (MLLMs) to real-world tasks such as analyzing scientific papers or financial reports. However, existing datasets often focus on oversimplified and homogeneous charts with template-based questions, leading to an over-optimistic measure of progress. In this work, we propose CharXiv, a comprehensive evaluation suite involving 2,323 natural, challenging, and diverse charts from scientific papers. CharXiv includes two types of questions: (1) descriptive questions about examining basic chart elements and (2) reasoning questions that require synthesizing information across complex visual elements in the chart. To ensure quality, all charts and questions are handpicked, curated, and verified by human experts. Our results reveal a substantial, previously underestimated gap between the reasoning skills of the strongest proprietary model (i.e., GPT-4o), which achieves 47.1% accuracy, and the strongest open-source model (i.e., InternVL Chat V1.5), which achieves 29.2%. All models lag far behind human performance of 80.5%, underscoring weaknesses in the chart understanding capabilities of existing MLLMs. We hope CharXiv facilitates future research on MLLM chart understanding by providing a more realistic and faithful measure of progress.

![Comparison of model performance](https://charxiv.github.io/static/images/comparison.png)

## 🛠️ Evaluate Your Models
### Setup
Download the images:
```
git clone https://github.com/princeton-nlp/CharXiv.git
cd images
wget https://huggingface.co/datasets/princeton-nlp/CharXiv/resolve/main/images.zip
unzip images.zip && rm images.zip
```

<details>
<summary>  (Optional) A short tour for the codebase </summary>
* `data` folder contains all QAs and metadata for images, descriptive questions, and reasoning questions. Answers for the test split are intentionally made to `null` to prevent testing data from leaking into the public.  
* `images` folder contains all images where their identifiers range from 0 to 2399. Note that there are only 2333 images in total and the numberings are **not** consecutive.  
* `results` folder contains all response generation and scoring results.  
* `src` folder contains all python code for CharXiv:  
  * `constants.py` stores all the prompts and mappings from question ids to actual questions.  
  * `descriptive_utils.py` contains all code to build queries for response generation and grading, as well as saving all artifacts for descriptive questions.  
  * `reasoning_utils.py` contains all code to build queries for response generation and grading, as well as saving all artifacts for reasoning questions.  
  * `evaluate.py` is the main function to evaluate model responses against the answer with gpt API calls.  
  * `generate.py` is the main function to loop QAs for model to generate responses.  
  * `get_stats.py` is the main function to print the reasoning and descriptive question statistics.
  * `generate_lib` contains a series of implementations that enable one to generate response on their models.
* `run.sh` is the script to evaluate models


</details>

### Response generation
CharXiv doesn't require any third-party python library when prompting your models to generate responses to the chart-question pairs. Therefore, to set up your model, you should implement the `custom_evaluate` function in `generate.py`. Specifically, this function takes `queries` as the input, which contain all the charts and questions CharXiv uses to evaluate models. It has the following structure:
```js
{
    figure_id:{
        'question': ...<str>
        'figure_path': ...<str>
    },
    ...
    figure_id:{
        'question': ...<str>
        'figure_path': ...<str>
    },
}
```

To run CharXiv for your model, go to `src/generate_lib/` directory and create a new python file e.g.,`custom.py`. You can put whatever auxiliary code (e.g., model split function, image processing function, etc) inside this file, but you need to implement the `generate_response` function which takes in `model_path` and `queries` by default. Inside the function, you load your models and all preprocessing functions, and let your model generate responses. The below script is a simple example of usage. If you need more references, you can look at existing `*.py` files how these functions are implemented for different models.


```py
for k in tqdm(queries):
    query = queries[k]['question'] # This will be a single question with instructions
    image = queries[k]["figure_path"] # This will be the path to the figure associated with the above query
    query, image = preprocess(query, image) #TODO
    response = model.chat(query, image) #TODO
    queries[k]['response'] = response
```

Once you finish implementing the `generate_response` function, go to `src/generate_lib/utils.py` and modify the `get_generate_fn` to include the function you implemented. Say if your model's checkpoint folder is called `custom_1` and you implement the `generate_response` function in the `custom.py` file, then  all you need to do it to add the follwing code:

```py
...
elif model_name in ['custom_1']:
    from .custom import generate_response
...
```

Then, to generate model responses:
```bash
python src/generate.py \
    --model_name $model_name \
    --split $split \
    --mode $mode \
    --model_path $model_path 
```
⚠️**The first three arguments are required and you should not delete them**. It is your responsibility to ensure the correctness and integrity of the evaluation pipeline if you change them. In particular,

* `--model_name` is used as a string identifier to name the resulting file and you can use whatever string except "-", which we reserve to parse file names to get meta information.

* You should either choose `val` or `test` for the `--split` argument. To perform end-to-end in-house evaluations, you should use `val` we do not release answers for `test`.

* Finally, you should either choose `descriptive` or `reasoning` for the `--mode` argument.

✅The last argument i.e., `--model_path` is a custom argument and feel free to delete it, modify it or add more args as you see fit. Correspondingly, you should consider changing the input argument of your `generate_response` function and the function that calls `generate_response` if you change `--model_path`.

🗄️ Once the process ends, you will find a file in `results` folder named:
`gen-<model_name>-<mode>_<split>.json`. This file stores your model's responses.

Note: if you are evaluating a model that is **hosted on cloud and can only be accessed via an API**, the `--model_path` argument will correspond to the name of the model e.g., `gpt-4o-2024-05-13`. Also, in creating the custom file in the `src/generate_lib` directory, you need to implement an additional function i.e., `get_client_model` that takes in the `model_path` argument and the `api_key` argument. In addition, you need to add another `elif` statement in `get_client_fn` inside `src/generate_lib/utils.py` with instructions similar to the above. Specific instructions to implement `get_client_model` function differ by API providers, and examples are provided in `gpt.py`, `gemini.py`, `reka.py`, `claude.py`, and `qwen.py`.
Note: you may see a header like:
```py
### HEADER START ###
import os
vlm_codebase = os.environ['VLM_CODEBASE_DIR']

import sys
sys.path.append(vlm_codebase + '/<codebase>')
### HEADER END ###
```
in existing implementations in `generate_lib`. This becomes convenient when the model needs to be run with some author-provided code, and you want to load/call the model with their code. In this way, you have a local directory which contains their code (i.e., if you are developing a model or cloning a codebase from github), and you append the directory to the system PATH.

### Evaluation

⚠️ Make sure the openai python API library is installed in your evaluation environment. If not, `pip install openai` first.

```bash
python src/evaluate.py \
    --model_name $model_name \
    --split $split \
    --mode $mode \
    --api_key $openai_key
```

The first three arguments are same as the above (response generation), except that you need to provide an openai API key to evaluate your model's responses.

This python script will automatically match the `gen-<model_name>-<mode>_<split>.json` file in the `results` folder and the `<mode>_<split>.json` file in the `data` folder.

🗄️ Once the process ends, you will find a file in results folder named: `scores-<model_name>-<mode>_<split>.json`. This file stores your model's evaluation results graded by LLMs.

Finally, run this:
```bash
python src/get_stats.py \
    --model_name $model_name \
    --split $split 
```
This python script will automatically match the `scores-<model_name>-<mode>_<split>.json` file in the `results` folder to calculate the stats for aggregated scores. Calling this function will automatically generate `stats-<model_name>-<mode>_<split>.json` in the `results` folder.

## 📅 Results from Existing Models
We release full results on the validation set (i.e., generated responses, grading done by LLMs and the aggregated stats) for all models we tested in our [HuggingFace Repo](https://huggingface.co/datasets/princeton-nlp/CharXiv/tree/main/existing_evaluations). If you are interested in doing some fine-grained analysis on these results or calculate some customized metrics, feel free to use them.

## 🏆 Leaderboard

Please refer to our official website by clicking the model name for detailed model information (e.g., huggingface model version or API version).

| Reasoning              | Descriptive                   |
|------------------------|-------------------------------|
| TC = Text-in-Chart     | INEX = Information Extraction |
| TG = Text-in-General   | ENUM = Enumeration            |
| NC = Number-in-Chart   | PATT = Pattern Recognition    |
| NG = Number-in-General | CNTG = Counting               |
|                        | COMP = Compositionality       |


| Metadata                  |                 |                | Reasoning |       |       |       |       | Descriptive |       |       |       |       |       |
| ------------------------- | --------------- | -------------- | --------- | ----- | ----- | ----- | ----- | ----------- | ----- | ----- | ----- | ----- | ----- |
| Model                     | Weight          | Size [V/L] (B) | Overall   | TC    | TG    | NC    | NG    | Overall     | INEX  | ENUM  | PATT  | CNTG  | COMP  |
|🎖️Human                     | N/A             | Unknown        | 80.50     | 77.27 | 77.78 | 84.91 | 83.41 | 92.10       | 91.40 | 91.20 | 95.63 | 93.38 | 92.86 |
|🥇Claude 3.5 Sonnet         | Proprietary     | Unknown        | 60.20     | 61.14 | 78.79 | 63.79 | 46.72 | 84.30       | 82.62 | 88.86 | 90.61 | 90.08 | 48.66 |
|🥈GPT-4o                    | Proprietary     | Unknown        | 47.10     | 50.00 | 61.62 | 47.84 | 34.50 | 84.45       | 82.44 | 89.18 | 90.17 | 85.50 | 59.82 |
|🥉Gemini 1.5 Pro            | Proprietary     | Unknown        | 43.30     | 45.68 | 56.57 | 45.69 | 30.57 | 71.97       | 81.79 | 64.73 | 79.48 | 76.34 | 15.18 |
| InternVL Chat V2.0 Pro    | Proprietary     | Unknown        | 39.80     | 40.00 | 60.61 | 44.40 | 25.76 | 76.83       | 77.11 | 84.67 | 77.07 | 78.88 | 27.23 |
| InternVL Chat V2.0 76B    | Open            | 5.9 / 70       | 38.90     | 40.00 | 59.60 | 42.67 | 24.02 | 75.17       | 77.11 | 78.69 | 76.20 | 79.13 | 32.14 |
| GPT-4V                    | Proprietary     | Unknown        | 37.10     | 38.18 | 57.58 | 37.93 | 25.33 | 79.92       | 78.29 | 85.79 | 88.21 | 80.92 | 41.07 |
| GPT-4o Mini               | Proprietary     | Unknown        | 34.10     | 35.23 | 47.47 | 32.33 | 27.95 | 74.92       | 74.91 | 82.81 | 69.21 | 79.13 | 35.71 |
| Gemini 1.5 Flash          | Proprietary     | Unknown        | 33.90     | 36.36 | 54.55 | 30.60 | 23.58 | \-          | \-    | \-    | \-    | \-    | \-    |
| InternVL Chat V2.0 26B    | Open            | 5.9 / 20       | 33.40     | 33.18 | 51.52 | 41.81 | 17.47 | 62.40       | 71.35 | 61.02 | 55.90 | 67.94 | 6.25  |
| Claude 3 Sonnet           | Proprietary     | Unknown        | 32.20     | 31.59 | 50.51 | 31.47 | 26.20 | 73.65       | 75.74 | 81.92 | 76.64 | 72.26 | 8.48  |
| Claude 3 Haiku            | Proprietary     | Unknown        | 31.80     | 29.77 | 45.45 | 34.48 | 27.07 | 65.08       | 69.87 | 69.98 | 64.85 | 61.83 | 8.04  |
| Phi-3 Vision              | Open            | 0.3 / 4        | 31.60     | 31.36 | 46.46 | 35.78 | 21.40 | 60.48       | 67.62 | 61.18 | 54.59 | 65.39 | 6.25  |
| MiniCPM-V2.6 (Upsize+CoT) | Open            | 0.4 / 8        | 31.00     | 30.00 | 41.41 | 37.93 | 21.40 | 57.05       | 67.85 | 49.56 | 53.49 | 62.85 | 14.29 |
| Claude 3 Opus             | Proprietary     | Unknown        | 30.20     | 26.36 | 50.51 | 33.62 | 25.33 | 71.55       | 75.62 | 73.69 | 73.58 | 70.48 | 26.79 |
| InternVL Chat V1.5        | Open            | 5.9 / 20       | 29.20     | 30.00 | 45.45 | 32.33 | 17.47 | 58.50       | 69.63 | 52.95 | 53.06 | 64.63 | 5.80  |
| GLM 4V 9B                 | Open            | 4.4 / 9        | 29.10     | 30.68 | 42.42 | 33.19 | 16.16 | 57.62       | 67.97 | 61.66 | 43.45 | 45.04 | 8.48  |
| Reka Core                 | Proprietary     | Unknown        | 28.90     | 27.50 | 41.41 | 28.45 | 26.64 | 55.60       | 58.90 | 50.52 | 65.72 | 71.25 | 10.71 |
| Ovis 1.5 Gemma2 9B        | Open            | 0.4 / 9        | 28.40     | 26.14 | 44.44 | 33.19 | 20.96 | 62.60       | 64.29 | 71.75 | 56.33 | 66.16 | 5.80  |
| Ovis 1.5 Llama3 8B        | Open            | 0.4 / 8        | 28.20     | 27.27 | 49.49 | 31.03 | 17.90 | 60.15       | 61.39 | 68.93 | 56.33 | 61.83 | 7.14  |
| Cambrian 34B              | Open            | 1.9 / 34       | 27.30     | 24.55 | 44.44 | 27.59 | 24.89 | 59.73       | 59.31 | 70.94 | 53.28 | 64.63 | 5.36  |
| MiniCPM-V2.6 (Upsize)     | Open            | 0.4 / 8        | 27.10     | 21.59 | 45.45 | 35.34 | 21.40 | 61.62       | 69.28 | 55.93 | 60.48 | 72.01 | 19.64 |
| Reka Flash                | Proprietary     | Unknown        | 26.60     | 26.59 | 39.39 | 30.60 | 17.03 | 56.45       | 61.39 | 48.59 | 69.87 | 72.52 | 7.14  |
| Mini Gemini HD Yi 34B     | Open            | 0.5 / 34       | 25.00     | 26.59 | 43.43 | 27.16 | 11.79 | 52.68       | 53.86 | 55.04 | 65.50 | 53.94 | 2.23  |
| InternLM XComposer2 4KHD  | Open            | 0.3 / 7        | 25.00     | 23.86 | 43.43 | 29.31 | 14.85 | 54.65       | 61.09 | 54.08 | 51.53 | 59.80 | 6.70  |
| MiniCPM-V2.5              | Open            | 0.4 / 8        | 24.90     | 25.23 | 43.43 | 25.43 | 15.72 | 59.27       | 62.28 | 61.90 | 56.77 | 68.96 | 10.27 |
| Qwen VL Max               | Proprietary     | Unknown        | 24.70     | 26.14 | 41.41 | 24.57 | 14.85 | 41.48       | 50.42 | 28.41 | 53.71 | 51.15 | 4.46  |
| VILA 1.5 40B              | Open            | 5.9 / 34       | 24.00     | 21.59 | 41.41 | 25.00 | 20.09 | 38.67       | 42.88 | 29.62 | 51.31 | 50.89 | 9.82  |
| Reka Edge                 | Proprietary     | Unknown        | 23.50     | 20.23 | 32.32 | 30.60 | 18.78 | 33.65       | 36.65 | 28.49 | 34.72 | 52.16 | 4.91  |
| Gemini 1.0 Pro            | Proprietary     | Unknown        | 22.80     | 20.91 | 48.48 | 18.10 | 20.09 | 54.37       | 67.97 | 39.23 | 60.48 | 62.60 | 8.93  |
| LLaVA 1.6 Yi 34B          | Open            | 0.3 / 34       | 22.50     | 20.45 | 37.37 | 23.71 | 18.78 | 51.05       | 46.38 | 63.44 | 56.11 | 51.91 | 5.80  |
| Mini Gemini HD Llama3 8B  | Open            | 0.5 / 8        | 19.00     | 19.77 | 36.36 | 21.12 | 7.86  | 44.42       | 49.41 | 39.23 | 51.09 | 55.98 | 1.79  |
| CogAgent                  | Open            | 4.4 / 7        | 18.80     | 16.82 | 32.32 | 20.69 | 14.85 | 36.30       | 45.14 | 26.80 | 43.23 | 37.15 | 6.70  |
| InternLM XComposer2       | Open            | 0.3 / 7        | 18.70     | 16.14 | 38.38 | 21.98 | 11.79 | 38.75       | 34.10 | 43.58 | 46.72 | 52.93 | 5.80  |
| MiniCPM-V2                | Open            | 0.4 / 2.4      | 18.50     | 17.95 | 33.33 | 19.40 | 12.23 | 35.77       | 39.74 | 36.56 | 26.42 | 44.53 | 5.36  |
| IDEFICS 2                 | Open            | 0.4 / 7        | 18.20     | 15.45 | 35.35 | 17.24 | 17.03 | 32.77       | 36.12 | 27.28 | 40.83 | 43.26 | 3.12  |
| IDEFICS 2 Chatty          | Open            | 0.4 / 7        | 17.80     | 15.45 | 34.34 | 19.83 | 13.10 | 41.55       | 34.88 | 54.56 | 45.63 | 44.27 | 6.70  |
| MoAI                      | Open            | 0.3 / 7        | 17.50     | 9.32  | 36.36 | 21.12 | 21.40 | 28.70       | 31.20 | 21.23 | 39.96 | 40.46 | 7.59  |
| DeepSeek VL               | Open            | 0.5 / 7        | 17.10     | 16.36 | 32.32 | 19.83 | 9.17  | 45.80       | 49.11 | 45.20 | 42.79 | 60.31 | 4.91  |
| DocOwl 1.5 Chat           | Domain-specific | 0.3 / 7        | 17.00     | 14.32 | 34.34 | 15.09 | 16.59 | 37.40       | 36.83 | 49.23 | 36.68 | 22.90 | 3.12  |
| SPHINX V2                 | Open            | 1.9 / 13       | 16.10     | 13.86 | 28.28 | 17.67 | 13.54 | 30.25       | 35.59 | 24.37 | 41.05 | 29.52 | 1.79  |
| Qwen VL Plus              | Proprietary     | Unknown        | 16.00     | 15.45 | 45.45 | 12.07 | 8.30  | 28.93       | 33.33 | 17.92 | 32.10 | 56.23 | 2.23  |
| UReader                   | Domain-specific | 0.3 / 7        | 14.30     | 11.36 | 18.18 | 15.52 | 17.03 | 18.98       | 10.20 | 27.60 | 33.41 | 20.36 | 5.36  |
| ChartLlama                | Domain-specific | 0.3 / 13       | 14.20     | 8.18  | 34.34 | 9.91  | 21.40 | 19.23       | 17.14 | 12.19 | 43.89 | 28.75 | 6.70  |
| LLaVA 1.6 Mistral 7B      | Open            | 0.3 / 7        | 13.90     | 11.36 | 32.32 | 16.81 | 7.86  | 35.40       | 34.70 | 33.98 | 48.91 | 42.49 | 8.48  |
| ChartGemma                | Domain-specific | 0.4 / 2        | 12.50     | 11.59 | 24.24 | 16.81 | 4.80  | 21.30       | 27.58 | 18.97 | 14.19 | 19.59 | 4.46  |
| ChartAssistant            | Domain-specific | 1.9 / 13       | 11.70     | 9.09  | 27.27 | 10.34 | 11.35 | 16.93       | 16.43 | 16.87 | 16.57 | 27.74 | 2.68  |
| ChartInstruct-FlanT5      | Domain-specific | 0.3 / 3        | 11.70     | 7.95  | 32.32 | 9.48  | 12.23 | 15.47       | 11.68 | 17.59 | 15.94 | 29.52 | 6.70  |
| Random (GPT-4o)           | N/A             | Unknown        | 10.80     | 4.32  | 39.39 | 5.60  | 16.16 | 19.85       | 21.65 | 16.71 | 23.80 | 25.70 | 5.36  |
| DocOwl 1.5 Omni           | Domain-specific | 0.3 / 7        | 9.10      | 5.45  | 14.14 | 9.48  | 13.54 | 25.70       | 34.46 | 17.92 | 31.88 | 17.56 | 4.46  |
| ChartInstruct-Llama2      | Domain-specific | 0.3 / 7        | 8.80      | 4.09  | 23.23 | 7.76  | 12.66 | 21.40       | 23.31 | 15.50 | 33.19 | 27.48 | 4.91  |
| TinyChart                 | Domain-specific | 0.4 / 3        | 8.30      | 5.00  | 13.13 | 6.47  | 14.41 | 16.15       | 13.82 | 14.61 | 24.67 | 28.50 | 3.12  |
| UniChart-ChartQA          | Domain-specific | 1.9 / 8        | 5.70      | 3.41  | 6.06  | 3.45  | 12.23 | 19.32       | 9.91  | 38.26 | 12.23 | 19.08 | 0.45  |
| TextMonkey                | Domain-specific | 8-Feb          | 3.90      | 2.50  | 4.04  | 3.02  | 7.42  | 12.45       | 12.16 | 17.92 | 8.73  | 6.36  | 2.68  |

## 📜 License
Our original data contributions (all data except the charts) are distributed under the [CC BY-SA 4.0](data/LICENSE) license. Our code is licensed under [Apache 2.0](LICENSE) license. The copyright of the charts belong to the original authors, where you can find the source in `image_metadata_val.json` and `image_metadata_test.json` under the data folder.


## 🥺 Cite
If you use our work and are inspired by our work, please consider cite us (available soon):
```bibtex
@article{wang2024charxiv,
  title={CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs},
  author={Wang, Zirui and Xia, Mengzhou and He, Luxi and Chen, Howard and Liu, Yitao and Zhu, Richard and Liang, Kaiqu and Wu, Xindi and Liu, Haotian and Malladi, Sadhika and Chevalier, Alexis and Arora, Sanjeev and Chen, Danqi},
  journal={arXiv preprint arXiv:2406.18521},
  year={2024}
}
```
## 🙌 Contributors and Acknowledgement
**📊 [CharXiv](https://charxiv.github.io/) is built by a team consisting of:**  
[Zirui Wang](https://zwcolin.github.io/), [Mengzhou Xia](https://xiamengzhou.github.io/), [Luxi He](https://twitter.com/luxihelucy), [Howard Chen](https://howard50b.github.io/), [Yitao Liu](https://yitaoliu17.com/), [Richard Zhu](https://richardzhu123.github.io/), [Kaiqu Liang](https://kaiquliang.github.io/), [Xindi Wu](https://xindiwu.github.io/), [Haotian Liu](https://hliu.cc/), [Sadhika Malladi](https://www.cs.princeton.edu/~smalladi/), [Alexis Chevalier](https://pli.princeton.edu/people/alexis-chevalier), [Sanjeev Arora](https://www.cs.princeton.edu/~arora/), [Danqi Chen](https://www.cs.princeton.edu/~danqic/)

[Princeton Language and Intelligence, Princeton University](https://pli.princeton.edu/)  
[University of Wisconsin, Madison](https://www.wisc.edu/)  
[The University of Hong Kong](https://www.hku.hk/)

🤗 We adapted part of the [MathVista](https://github.com/lupantech/MathVista)'s codebase in building our evaluation framework, and we greatly appreciate their contributions to the MLLM community.  
🤗 The lyrics in the teaser video are created by [GPT-4o](https://openai.com/index/hello-gpt-4o/) from our abstract, and the music is created by [Suno](https://suno.com/). Video is manually edited using [CapCut](https://www.capcut.com/).

**We’ve received multiple requests for scripts that batch convert vector images to raster images from arXiv source files that we used to construct CharXiv. While the code is not part of this codebase, we are happy to provide them. Please send an email to [Zirui Wang](mailto:zw1300@cs.princeton.edu) for inquiries.**
