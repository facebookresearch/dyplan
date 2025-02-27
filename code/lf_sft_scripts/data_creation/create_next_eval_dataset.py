# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from llm_models import E2E_MODEL
import os
import json
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=str, nargs='+', default=[], required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--history", type=int, default=-1, required=True)
    parser.add_argument("--save_file", type=str, required=True)
    parser.add_argument("--verified_save_file", type=str, required=False)
    parser.add_argument("--num_samples", type=int, default=1000, required=False)
    args = parser.parse_args()
    args.data_dirs = [ args.data_dir % field for field in args.fields ]
    assert len(args.fields) == len(args.data_dirs)
    assert args.history >= 0
    return args

def clean_final_answer(text, rag=False):
    if rag:
        retrieved_text = "Context: \n" + text.split("Context:")[1].split("\n\nReasoning:")[0].strip()
        reasoning = text.split("\n\nReasoning:")[1].strip()
        try:
            reasoning = reasoning.split("Final answer:")[0] + "Final answer:" + '"'.join(reasoning.split("Final answer:")[1].split('"')[:2]) + '"'
            return retrieved_text, reasoning
        except:
            return retrieved_text, reasoning
    else:
        try:
            text = text.split("Final answer:")[0] + "Final answer:" + '"'.join(text.split("Final answer:")[1].split('"')[:2]) + '"'
            return text
        except:
            return text

def create_preplan_output(field):
    return field

def load_text(text):
    try:
        return json.loads(text)
    except:
        return text

def main():
    args = get_args()
    print (f"{args}")

    all_data = []
    for field, data_dir in zip(args.fields, args.data_dirs):
        with open(data_dir + "/test.tsv", 'r') as f:
            for i, line in enumerate(f):
                line_fields = line.strip().split("\t")
                dt = {"question": line_fields[0], field: {"answer": line_fields[-1]}}
                if "reasoning" in data_dir:
                    dt[field]["reasoning"] = load_text(line_fields[1])
                if "answer" in data_dir:
                    dt[field]["prediction"] = load_text(line_fields[-2])
                
                if len(all_data) > i:
                    assert dt["question"] == all_data[i]["question"]
                    all_data[i][field] = dt[field]
                else:
                    all_data.append(dt)
        if args.num_samples > 0:
            all_data = all_data[:args.num_samples]
    print ("Processed %d gold datapoints" % len(all_data))

    with open(args.pred_file, 'r') as f:
        pred_data = []
        for line in f:
            pred_data.append(json.loads(line.strip()))
        if args.num_samples > 0:
            pred_data = pred_data[:args.num_samples]
    print ("Processed %d pred datapoints" % len(pred_data))

    llm_model = E2E_MODEL(args, no_load_model=True)

    output_data, verified_data = [], []
    skipped = 0
    for idx, (gold_dt, pred_dt) in enumerate(zip(all_data, pred_data)):
        system_prompt, instruction, input = llm_model._get_decomposed_prompt_preplan(gold_dt)
        pred_strategy = pred_dt["predict"] if args.history == 1 else pred_dt["round1_preplan_predict"]
        if pred_strategy not in llm_model.strategy2field:
            round1_field = args.fields[-1]
            pred_strategy = llm_model.field2strategy[round1_field]
        else:
            round1_field = llm_model.strategy2field[pred_strategy]
        preplan_datapoint = {
            "system": system_prompt,
            "instruction": instruction,
            "input": input,
            "output": pred_strategy
        }

        if args.history >= 1:
            system_prompt, instruction, history = llm_model._get_decomposed_prompt_strategy_use([preplan_datapoint])
            retrieved_input = ""
            if round1_field == "rag":
                retrieved_input, answer_output = clean_final_answer(gold_dt[round1_field]["reasoning"], rag=True)
            else:
                answer_output = clean_final_answer(gold_dt[round1_field]["reasoning"])
            if args.history >= 2:
                answer_output = pred_dt["predict"] if args.history == 2 else pred_dt["round1_strategy_use_predict"]
            strategy_use_datapoint = {
                "system": system_prompt,
                "instruction": instruction,
                "input": retrieved_input,
                "output": answer_output,
                "history": history
            }
            if args.history == 1:
                output_data.append(strategy_use_datapoint)
                continue

            system_prompt, instruction, history = llm_model._get_decomposed_prompt_verify([preplan_datapoint, strategy_use_datapoint])
            if args.history == 2:
                answer_output = "Yes" if gold_dt[round1_field]["answer"] == "1" else "No"
            else:
                if args.history == 3:
                    answer_output = pred_dt["predict"]
                elif "round1_verify_predict" not in pred_dt:
                    answer_output = pred_dt["predict"]
                    assert answer_output == "Yes"
                else:
                    answer_output = pred_dt["round1_verify_predict"]
            verify_datapoint = {
                "system": system_prompt,
                "instruction": instruction,
                "input": "",
                "output": answer_output,
                "history": history
            }
            if args.history == 2:
                output_data.append(verify_datapoint)
                continue

            if answer_output == "Yes":
                verified_data.append((idx, pred_dt))
                continue

            system_prompt, instruction, history = llm_model._get_decomposed_prompt_sec_preplan([preplan_datapoint, strategy_use_datapoint, verify_datapoint], used_fields=[round1_field])
            if args.history == 3:
                correct_fields = [ field for field in args.fields if gold_dt[field]["answer"] == "1" and field != round1_field ]
                if len(correct_fields) == 0:
                    round2_field = [ field for field in args.fields if field != round1_field ][-1]
                else:
                    round2_field = correct_fields[0]
                answer_output = llm_model.field2strategy[round2_field]
            else:
                answer_output = pred_dt["predict"] if args.history == 4 else pred_dt["round2_preplan_predict"]
                if answer_output not in llm_model.strategy2field:
                    round2_field = [ field for field in args.fields if field != round1_field ][-1]
                    answer_output = llm_model.field2strategy[round2_field]
                else:
                    round2_field = llm_model.strategy2field[answer_output]
            sec_preplan_datapoint = {
                "system": system_prompt,
                "instruction": instruction,
                "input": "",
                "output": answer_output,
                "history": history
            }
            if args.history == 3:
                output_data.append(sec_preplan_datapoint)
                continue

            system_prompt, instruction, history = llm_model._get_decomposed_prompt_strategy_use([preplan_datapoint, strategy_use_datapoint, verify_datapoint, sec_preplan_datapoint])
            answer_output = clean_final_answer(gold_dt[round2_field]["reasoning"])
            sec_strategy_datapoint = {
                "system": system_prompt,
                "instruction": instruction,
                "input": "",
                "output": answer_output,
                "history": history
            }
            output_data.append(sec_strategy_datapoint)

    print ("Skipped: %d" % skipped)
    print ("Created output data: %d datapoints" % len(output_data))

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    with open(args.save_file, "w") as f:
        if len(output_data) > 0:
            json.dump(output_data, f, indent=2)
    
    if args.verified_save_file:
        os.makedirs(os.path.dirname(args.verified_save_file), exist_ok=True)
        with open(args.verified_save_file, "w") as f:
            json.dump(verified_data, f, indent=2)
    
if __name__ == "__main__":
    main()