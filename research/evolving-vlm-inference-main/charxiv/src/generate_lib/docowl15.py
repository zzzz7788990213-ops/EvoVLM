# Adapted from https://github.com/X-PLUG/mPLUG-DocOwl/blob/main/DocOwl1.5/docowl_infer.py
# This has support for the DocOwl model

### HEADER START ###
import os
vlm_codebase = os.environ['VLM_CODEBASE_DIR']

import sys
sys.path.append(vlm_codebase + '/mPLUG-DocOwl/DocOwl1.5')
### HEADER END ###

from docowl_infer import DocOwlInfer
from tqdm import tqdm
import os

def generate_response(queries, model_path):
    docowl = DocOwlInfer(ckpt_path=model_path, anchors='grid_9', add_global_img=True)
    print('load model from ', model_path)
    # infer the test samples one by one
    for k in tqdm(queries):
        image = queries[k]['figure_path']
        question = queries[k]['question']
        model_answer = docowl.inference(image, question)
        print('model_answer:', model_answer)
        queries[k]['response'] = model_answer
