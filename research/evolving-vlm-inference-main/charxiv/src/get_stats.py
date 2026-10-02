import json
import argparse
import os

from score_utils import DOMAIN2ABBR, NUM2YEAR, QNUM2QTYPE, \
        NUMSUBPLOTS2SUBPLOTTYPE, D_TEMPLATE, R_TEMPLATE, \
        IDX2ANSTYPE, IDX2SRC

def get_descriptive_scores(scores, dmeta, rmeta, imeta):
    stats = D_TEMPLATE()
    for k, v in dmeta.items():
        num_subplot = dmeta[k]['num_subplots']
        subject = imeta[k]['category']
        year = imeta[k]['year']
        for i in range(4):
            subq_key = f"{k}_{i}"
            score = scores[subq_key]['score']
            qnum = dmeta[k]['qids'][i]

            if score not in [0, 1]:
                stats['N_invalid'].append(1)
                score = 0
            
            stats['N_valid'].append(1)
            stats['Overall Score'].append(score)
            stats['By Category'][QNUM2QTYPE(qnum)].append(score)
            stats['By Subject'][DOMAIN2ABBR[subject]].append(score)
            stats['By Year'][NUM2YEAR[year]].append(score)
            stats['By Subplot'][NUMSUBPLOTS2SUBPLOTTYPE(num_subplot)].append(score)
            stats['By Question'][f'Q{qnum}'].append(score)
    stats['Question Type'] = 'Descriptive'
    return stats

def get_reasoning_scores(scores, dmeta, rmeta, imeta):
    stats = R_TEMPLATE()
    for k, v in rmeta.items():
        num_subplot = dmeta[k]['num_subplots']
        subject = imeta[k]['category']
        year = imeta[k]['year']
        answer_type = rmeta[k]['inst_category']
        source = rmeta[k]['qa_source']
        score = scores[k]['score']
        if score not in [0, 1]:
            stats['N_invalid'].append(1)
            score = 0
        
        stats['N_valid'].append(1)
        stats['Overall Score'].append(score)
        stats['By Answer Type'][IDX2ANSTYPE[answer_type]].append(score)
        stats['By Source'][IDX2SRC[source]].append(score)
        stats['By Subject'][DOMAIN2ABBR[subject]].append(score)
        stats['By Year'][NUM2YEAR[year]].append(score)
        stats['By Subplot'][NUMSUBPLOTS2SUBPLOTTYPE(num_subplot)].append(score)
    stats['Question Type'] = 'Reasoning'
    return stats

def get_stats(stats):
    if len(stats['N_valid']) == 0:
        print("No valid scores")
        return
    for k, v in stats.items():
        # for sub categories
        if isinstance(v, dict):
            for k1, v1 in v.items():
                if len(v1) == 0:
                    print(f"{k1}: No valid scores")
                    stats[k][k1] = 0
                else:
                    stats[k][k1] = round(100 * sum(v1)/len(v1), 2)
        # metadata
        elif k == 'Question Type':
            pass
        # for overall scores
        elif k not in ['N_valid', 'N_invalid']:
            if len(v) == 0:
                print(f"{k}: No valid scores")
                stats[k] = 0
            else:
                stats[k] = round(100 * sum(v)/len(v), 2)
        # for number of valid/invalid scores
        else:
            stats[k] = len(v)
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input/output
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--score_prefix', type=str, default='scores-')
    parser.add_argument('--stats_prefix', type=str, default='stats-')
    args = parser.parse_args()

    descriptive_score_path = f"results/{args.score_prefix}{args.model_name}-descriptive_{args.split}.json"
    reasoning_score_path = f"results/{args.score_prefix}{args.model_name}-reasoning_{args.split}.json"

    image_meta = json.load(open(f"data/image_metadata_{args.split}.json"))
    descriptive_meta = json.load(open(f"data/descriptive_{args.split}.json"))
    reasoning_meta = json.load(open(f"data/reasoning_{args.split}.json"))


    if os.path.exists(reasoning_score_path):
        reasoning_scores = json.load(open(reasoning_score_path))
        reasoning_stats = get_reasoning_scores(reasoning_scores, descriptive_meta, 
                                               reasoning_meta, image_meta)
        reasoning_stats = get_stats(reasoning_stats)
        json.dump(reasoning_stats, open(f"results/{args.stats_prefix}{args.model_name}-reasoning_{args.split}.json", "w"), indent=4)
        print("### Reasoning Stats ###")
        print(json.dumps(reasoning_stats, indent=4))

    if os.path.exists(descriptive_score_path):
        descriptive_scores = json.load(open(descriptive_score_path))
        descriptive_stats = get_descriptive_scores(descriptive_scores, descriptive_meta, 
                                                   reasoning_meta, image_meta)
        descriptive_stats = get_stats(descriptive_stats)

        json.dump(descriptive_stats, open(f"results/{args.stats_prefix}{args.model_name}-descriptive_{args.split}.json", "w"), indent=4)
        print("### Descriptive Stats ###")
        print(json.dumps(descriptive_stats, indent=4))
    
    print("Stats saved to results folder")

