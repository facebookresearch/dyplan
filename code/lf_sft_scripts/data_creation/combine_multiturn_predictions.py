# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import json

STRATEGY2FIELD = {
    "Direct": "direct",
    "CoT": "cot",
    "RAG_Direct": "rag-direct",
    "RAG": "rag",
    "Plan": "selfask"
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=str, required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--test_samples", type=int, default=1000, required=False)
    args = parser.parse_args()
    args.fields = args.fields.split(",")
    return args

def main():
    args = get_args()
    print (f"{args}")

    with open(args.pred_file, 'r') as f:
        all_data = []
        for line in f:
            all_data.append(json.loads(line.strip()))
        preplan_preds = all_data[:args.test_samples]
        strategy_answer_preds = [ {} for _ in range(args.test_samples) ]
        for i in range(args.test_samples):
            for j, field in enumerate(args.fields):
                strategy_answer_preds[i][field] = all_data[(j+1)*args.test_samples + i]

    combined_data = []
    preplan_fail = 1
    for preplan_output, strategy_answer in zip(preplan_preds, strategy_answer_preds):
        if preplan_output["predict"].strip() not in STRATEGY2FIELD:
            preplan_fail += 1
            field = args.fields[-1]
        else:
            field = STRATEGY2FIELD[preplan_output["predict"].strip()]

        combined_data.append({
            "prompt": strategy_answer[field]["prompt"],
            "label": strategy_answer[field]["label"],
            "predict": strategy_answer[field]["predict"],
            "preplan_predict": preplan_output["predict"],
            "preplan_label": preplan_output["label"]
        })
    
    if preplan_fail > 0:
        print ("Preplan failed: %d" % preplan_fail)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as outfile:
        for dt in combined_data:
            outfile.write(json.dumps(dt) + "\n")
    
if __name__ == "__main__":
    main()