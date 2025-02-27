# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from llm_models import StrategyBinaryDecider, StrategyBinaryPostDecider
import os
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=str, nargs='+', default=[], required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_type", type=str, required=True)
    parser.add_argument("--save_file", type=str, required=True)
    parser.add_argument("--is_test", type=int, default=0, required=False)
    parser.add_argument("--foo", type=int, default=0, required=False)
    args = parser.parse_args()
    return args

def create_answer(answer, foo=False):
    if foo:
        return "foo" if answer == "1" else "bar"
    else:
        if answer not in ["1", "0"]:
            import ipdb; ipdb.set_trace()
        return "yes" if answer == "1" else "no"

def main():
    args = get_args()
    print (f"{args}")

    data = []
    if not args.is_test:
        with open(args.data_dir + "/train.tsv", 'r') as f:
            for line in f:
                line_fields = line.strip().split("\t")
                dt = {"question": line_fields[0], "answer": create_answer(line_fields[-1], args.foo)}
                if "reasoning" in args.data_dir:
                    dt["reasoning"] = line_fields[1]
                if "answer" in args.data_dir:
                    dt["prediction"] = line_fields[-2]
                data.append(dt)
        with open(args.data_dir + "/dev.tsv", 'r') as f:
            for line in f:
                line_fields = line.strip().split("\t")
                dt = {"question": line_fields[0], "answer": create_answer(line_fields[-1], args.foo)}
                if "reasoning" in args.data_dir:
                    dt["reasoning"] = line_fields[1]
                if "answer" in args.data_dir:
                    dt["prediction"] = line_fields[-2]
                data.append(dt)
    else:
        with open(args.data_dir + "/test.tsv", 'r') as f:
            for line in f:
                line_fields = line.strip().split("\t")
                dt = {"question": line_fields[0], "answer": create_answer(line_fields[-1], args.foo)}
                if "reasoning" in args.data_dir:
                    dt["reasoning"] = line_fields[1]
                if "answer" in args.data_dir:
                    dt["prediction"] = line_fields[-2]
                data.append(dt)

    if args.data_type == "predecide_binary":
        llm_model = StrategyBinaryDecider(args, no_load_model=True)
    elif args.data_type == "postdecide_binary":
        llm_model = StrategyBinaryPostDecider(args, no_load_model=True)

    output_data = []
    for dt in data:    
        system_prompt, instruction, input = llm_model._get_decomposed_prompt(dt)
        if args.foo:
            instruction = instruction.replace('yes/no', 'foo/bar')
            instruction = instruction.replace('yes', 'foo')
            instruction = instruction.replace("'no'", "'bar'")
        output_data.append({
            "system": system_prompt,
            "instruction": instruction,
            "input": input,
            "output": dt["answer"]
        })

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    with open(args.save_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
if __name__ == "__main__":
    main()