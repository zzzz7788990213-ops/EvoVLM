import argparse, json
from openai import OpenAI
from tqdm import tqdm
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--gen_prefix', type=str, default='gen-')
    parser.add_argument('--api_key', type=str, required=True)
    args = parser.parse_args()
    
    client = OpenAI(api_key=args.api_key)
    args.input_file = f"data/{args.mode}_{args.split}.json"
    args.resp_file = f"results/{args.gen_prefix}{args.model_name}-{args.mode}_{args.split}.json"
    args.output_file = args.resp_file.replace(args.gen_prefix, "scores-")
    print(f"Output file: {args.output_file}")

    data, response = json.load(open(args.input_file)), json.load(open(args.resp_file))
    mode = 'descriptive' if 'descriptive' in args.resp_file.split('-')[-1] else 'reasoning'

    if mode == 'descriptive':
        from descriptive_utils import preprocess_descriptive_grading_queries, build_descriptive_grading_queries, \
                postprocess_descriptive_grading_queries, get_descriptive_result_gpt
        # group the responses based on the template id instead of figure id
        groups = preprocess_descriptive_grading_queries(data, response)
        # batched evaluation based on number of questions per query (nq_per_query)
        queries = build_descriptive_grading_queries(groups)
        combined_queries = []
        for query in tqdm(queries):
            result = get_descriptive_result_gpt(client, query['grading_query'], len(query['resp_keys']))
            # query contains resp_keys, grading_query, extract_answer and score
            combined_queries.append({**query, **result})
        queries = combined_queries
        # flatten the queries and only keep the necessary fields
        queries = postprocess_descriptive_grading_queries(queries)
    
    elif mode == 'reasoning':
        from reasoning_utils import build_reasoning_grading_queries, get_reasoning_result_gpt
        # dict of figure_id -> {figure_id, grading_query}
        queries = build_reasoning_grading_queries(data, response) 
        for figure_id, query in tqdm(queries.items()):
            ext, scr = get_reasoning_result_gpt(client, query['grading_query'])
            queries[figure_id]['extracted_answer'] = ext
            queries[figure_id]['score'] = scr
            queries[figure_id].pop('grading_query')
    else: raise ValueError("Mode not supported")

    # output the scores
    with open(args.output_file, "w") as f:
        json.dump(queries, f, indent=4)
