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
FIELD2STRATEGY = {v:k for k,v in STRATEGY2FIELD.items()}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--base_file", type=str, required=True)
    parser.add_argument("--save_file", type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    print (f"{args}")

    with open(args.base_file, 'r') as f:
        all_data = []
        for line in f:
            all_data.append(json.loads(line.strip()))

    strategy = FIELD2STRATEGY[args.field]
    combined_data = []
    for dt in all_data:
        dt["label"] = strategy
        dt["predict"] = strategy
        combined_data.append(dt)

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    with open(args.save_file, 'w') as outfile:
        for dt in combined_data:
            outfile.write(json.dumps(dt) + "\n")
    
if __name__ == "__main__":
    main()