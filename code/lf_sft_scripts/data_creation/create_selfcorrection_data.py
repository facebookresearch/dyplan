# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import json

def get_args():
    parser = argparse.ArgumentParser()
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

    combined_data = []
    for dt in all_data:
        if "round1_verify_predict" in dt and dt["round1_verify_predict"] == "No":
            round1_strategy = dt["round1_preplan_predict"]
            dt["label"] = round1_strategy
            dt["predict"] = round1_strategy
        combined_data.append(dt)

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    with open(args.save_file, 'w') as outfile:
        for dt in combined_data:
            outfile.write(json.dumps(dt) + "\n")
    
if __name__ == "__main__":
    main()