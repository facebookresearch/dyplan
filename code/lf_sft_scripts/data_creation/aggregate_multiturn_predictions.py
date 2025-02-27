# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_files", type=str, nargs='+', default=[], required=True)
    parser.add_argument("--turn_names", type=str, nargs='+', default=[], required=True)
    parser.add_argument("--verified_file", type=str, default=None, required=False)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1000, required=False)
    parser.add_argument("--final_file", type=int, default=0, required=False)
    args = parser.parse_args()
    assert len(args.pred_files) == len(args.turn_names)
    return args

def main():
    args = get_args()
    print (f"{args}")

    all_pred_data = []
    for pred_file in args.pred_files:
        with open(pred_file, 'r') as f:
            data = []
            for line in f:
                data.append(json.loads(line.strip()))
            all_pred_data.append(data)
    len_all_preds = [ len(data) for data in all_pred_data ]
    assert args.verified_file or max(len_all_preds) == min(len_all_preds)

    if not args.verified_file:
        combined_data = []
        for idx in range(len(all_pred_data[0])):
            output_dt = all_pred_data[-1][idx]
            output_dt[args.turn_names[-1]] = output_dt["predict"]
            output_dt[args.turn_names[-1].replace("predict", "label")] = output_dt["label"]

            for p_idx in range(len(all_pred_data) - 1):
                output_dt[args.turn_names[p_idx]] = all_pred_data[p_idx][idx]["predict"]
                output_dt[args.turn_names[p_idx].replace("predict", "label")] = all_pred_data[p_idx][idx]["label"]

            combined_data.append(output_dt)
    
    else:
        with open(args.verified_file, 'r') as f:
            verified_data = json.load(f)
        verified_idxs = [ dt[0] for dt in verified_data ]
        assert max(len_all_preds) == len(verified_idxs) + min(len_all_preds)
        assert len(set(len_all_preds)) == 2

        max_length = max(len_all_preds)
        for i, pred_data in enumerate(all_pred_data):
            if len(pred_data) == max_length:
                all_pred_data[i] = [ dt for j, dt in enumerate(pred_data) if j not in verified_idxs ]
        len_all_preds = [ len(data) for data in all_pred_data ]
        assert max(len_all_preds) == min(len_all_preds), len_all_preds

        combined_data = []
        for idx in range(len(all_pred_data[0])):
            output_dt = all_pred_data[-1][idx]
            output_dt[args.turn_names[-1]] = output_dt["predict"]
            output_dt[args.turn_names[-1].replace("predict", "label")] = output_dt["label"]

            for p_idx in range(len(all_pred_data) - 1):
                output_dt[args.turn_names[p_idx]] = all_pred_data[p_idx][idx]["predict"]
                output_dt[args.turn_names[p_idx].replace("predict", "label")] = all_pred_data[p_idx][idx]["label"]

            combined_data.append(output_dt)

        new_combined_data = []
        idx, combined_data_idx = 0, 0
        for dt in verified_data:
            num_add = dt[0] - idx
            idx = dt[0] + 1
            new_combined_data.extend(combined_data[combined_data_idx:combined_data_idx+num_add])
            combined_data_idx = combined_data_idx + num_add
            if args.final_file:
                dt[1]["round1_verify_predict"] = dt[1]["predict"]
                dt[1]["round1_verify_label"] = dt[1]["label"]
                dt[1]["predict"] = dt[1]["round1_strategy_use_predict"]
                dt[1]["label"] = dt[1]["round1_strategy_use_label"]
                del dt[1]["round1_strategy_use_predict"]
                del dt[1]["round1_strategy_use_label"]
            new_combined_data.append(dt[1])
        num_add = args.num_samples - idx
        new_combined_data.extend(combined_data[combined_data_idx:combined_data_idx+num_add])
        assert len(new_combined_data) == args.num_samples
        combined_data = new_combined_data

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as outfile:
        for dt in combined_data:
            outfile.write(json.dumps(dt) + "\n")
    
if __name__ == "__main__":
    main()