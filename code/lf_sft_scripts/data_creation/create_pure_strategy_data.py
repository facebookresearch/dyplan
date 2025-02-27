# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import json

SYSTEM_PROMPT="You are looking to provide the answer to the question asked using the strategy provided."
INSTRUCTION="Use the following strategy style to provide the answer to the question."
STRATEGY_DESCRIPTIONS = {
    "direct": "Direct - provide the exact answer directly to the question.",
    "cot": "CoT - infer the answer yourself by reasoning step-by-step.",
    "rag-direct": "RAG_Direct - use the external text to better answer the question, but directly answer the question.",
    "rag": "RAG - use the external resources or text to better answer the question along with additional thinking to reason through the external text and your self-knowledge.",
    "selfask": "Plan - break the question into simpler atomic questions, answer them and infer the final answer."
}
INPUT_TEMPLATE = "Question: %s"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_file", type=str, required=True)
    parser.add_argument("--is_test", type=int, default=0, required=False)
    parser.add_argument("--remove_neg", type=int, default=0, required=False)
    parser.add_argument("--start_index", type=int, default=0, required=False)
    parser.add_argument("--num_samples", type=int, default=-1, required=False)
    args = parser.parse_args()
    return args

def get_final_answer(text, force_output=False):
    if "Final answer:" not in text and not force_output:
        return None
    reasoning = text
    try:
        if reasoning.split("Final answer:")[1].count('"') >= 2:
            reasoning = reasoning.split("Final answer:")[0] + "Final answer:" + '"'.join(reasoning.split("Final answer:")[1].split('"')[:2]) + '"'
    except:
        reasoning = reasoning
    return reasoning

def create_single_turn_answer_output(field, dt, force_output=False, rag=False):
    text = dt[field]["reasoning"]
    if rag:
        retrieved_text = "Context: \n" + text.split("Context:")[1].split("\n\nReasoning:")[0].strip()
        reasoning = text.split("\n\nReasoning:")[1].strip()
        return retrieved_text, get_final_answer(reasoning, force_output=force_output)
    else:
        return get_final_answer(text, force_output=force_output)

def load_text(text):
    try:
        return json.loads(text)
    except:
        return text

def main():
    args = get_args()
    print (f"{args}")

    all_data = []
    if not args.is_test:
        with open(args.data_dir + "/train.tsv", 'r') as f:
            for i, line in enumerate(f):
                line_fields = line.strip().split("\t")
                dt = {"question": line_fields[0], args.field: {"answer": line_fields[-1]}}
                if "reasoning" in args.data_dir:
                    dt[args.field]["reasoning"] = load_text(line_fields[1])
                if "answer" in args.data_dir:
                    dt[args.field]["prediction"] = load_text(line_fields[-2])
                
                if len(all_data) > i:
                    assert dt["question"] == all_data[i]["question"], (dt["question"], all_data[i]["question"])
                    all_data[i][args.field] = dt[args.field]
                else:
                    all_data.append(dt)

        train_len = i + 1
        with open(args.data_dir + "/dev.tsv", 'r') as f:
            for i, line in enumerate(f):
                line_fields = line.strip().split("\t")
                dt = {"question": line_fields[0], args.field: {"answer": line_fields[-1]}}
                if "reasoning" in args.data_dir:
                    dt[args.field]["reasoning"] = load_text(line_fields[1])
                if "answer" in args.data_dir:
                    dt[args.field]["prediction"] = load_text(line_fields[-2])
                
                if len(all_data) > train_len + i:
                    assert dt["question"] == all_data[train_len + i]["question"]
                    all_data[train_len + i][args.field] = dt[args.field]
                else:
                    all_data.append(dt)
    else:
        with open(args.data_dir + "/test.tsv", 'r') as f:
            for i, line in enumerate(f):
                line_fields = line.strip().split("\t")
                dt = {"question": line_fields[0], args.field: {"answer": line_fields[-1]}}
                if "reasoning" in args.data_dir:
                    dt[args.field]["reasoning"] = load_text(line_fields[1])
                if "answer" in args.data_dir:
                    dt[args.field]["prediction"] = load_text(line_fields[-2])
                
                if len(all_data) > i:
                    assert dt["question"] == all_data[i]["question"]
                    all_data[i][args.field] = dt[args.field]
                else:
                    all_data.append(dt)
    print ("Processed %d datapoints" % len(all_data))

    if args.start_index != 0:
        all_data = all_data[args.start_index:]
    if args.num_samples > 0:
        all_data = all_data[:args.num_samples]

    output_data = []
    skipped = 0
    for dt in all_data:
        system_prompt, instruction, input = SYSTEM_PROMPT, INSTRUCTION + "\n" + STRATEGY_DESCRIPTIONS[args.field], INPUT_TEMPLATE % dt["question"]
        is_correct = dt[args.field]["answer"] == "1"
        if args.remove_neg and not is_correct and not args.is_test:
            skipped += 1
            continue
        
        retrieved_text = ""
        if args.field == "rag":
            retrieved_text, single_turn_output = create_single_turn_answer_output(args.field, dt, force_output=args.is_test, rag=True)
        else:
            single_turn_output = create_single_turn_answer_output(args.field, dt, force_output=args.is_test, rag=False)
        if single_turn_output is None:
            assert not args.is_test
            skipped += 1
            continue
        output_data.append({
            "system": system_prompt,
            "instruction": instruction,
            "input": input if retrieved_text == "" else retrieved_text + "\n\n" + input,
            "output": single_turn_output
        })     

    print ("Skipped: %d" % skipped)
    print ("Created output data: %d datapoints" % len(output_data))

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    with open(args.save_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
if __name__ == "__main__":
    main()