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
    parser.add_argument("--data_dirs", type=str, nargs='+', default=[], required=True)
    parser.add_argument("--save_file", type=str, required=True)
    parser.add_argument("--is_test", type=int, default=0, required=False)
    parser.add_argument("--single_turn", type=int, default=0, required=False)
    parser.add_argument("--include_best", type=int, default=0, required=False)
    parser.add_argument("--include_best_preplan", type=int, default=0, required=False)
    parser.add_argument("--remove_neg", type=int, default=0, required=False)
    parser.add_argument("--remove_neg_postplan", type=int, default=0, required=False)
    parser.add_argument("--sep_neg", type=int, default=0, required=False)
    parser.add_argument("--balance_data", type=int, default=0, required=False)
    parser.add_argument("--verify", type=int, default=0, required=False)
    parser.add_argument("--verify_balance", type=int, default=0, required=False)
    parser.add_argument("--start_index", type=int, default=0, required=False)
    parser.add_argument("--num_samples", type=int, default=-1, required=False)
    parser.add_argument("--high_impact", type=int, default=0, required=False)
    args = parser.parse_args()
    assert len(args.fields) == len(args.data_dirs)
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

def create_single_turn_preplan_answer_output(field, dt, field_name=None, force_output=False):
    output = "Strategy: %s\n" % (field if field_name is None else field_name)
    if "Final answer:" not in dt[field]["reasoning"] and not force_output:
        return None
    reasoning = dt[field]["reasoning"]
    reasoning = clean_final_answer(reasoning)
    output += "Reasoning:\n%s" % reasoning
    return output

def create_preplan_output(field):
    return field

def balance_data(data, field_names, history=0, verify=0):
    if history > 0:
        if verify == 0:
            examples_per_class = { field_name: [ dt for dt in data if "history" in dt and field_name in dt["history"][0][-1] and len(dt["history"]) == history ] for field_name in field_names }
        else:
            examples_per_class = { field_name: [ dt for dt in data if "history" in dt and len(dt["history"]) == history and field_name in dt["output"] ] for field_name in field_names }
    else:
        examples_per_class = { field_name: [ dt for dt in data if "history" not in dt and field_name in dt["output"] ] for field_name in field_names }
    num_examples_per_class = { field_name: len(examples_per_class[field_name]) for field_name in examples_per_class }
    max_examples = max([num_examples_per_class[field_name] for field_name in examples_per_class])
    addn_examples = []
    for field_name in examples_per_class:
        examples_added = max_examples - num_examples_per_class[field_name]
        addn_examples += random.choices(examples_per_class[field_name], k=examples_added)
    print (num_examples_per_class)
    print ("Balanced data. History: %d. Added %d examples" % (history, len(addn_examples)))
    balanced_data = data + addn_examples
    random.shuffle(balanced_data)
    return balanced_data

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
        if not args.is_test:
            with open(data_dir + "/train.tsv", 'r') as f:
                for i, line in enumerate(f):
                    line_fields = line.strip().split("\t")
                    dt = {"question": line_fields[0], field: {"answer": line_fields[-1]}}
                    if "reasoning" in data_dir:
                        dt[field]["reasoning"] = load_text(line_fields[1])
                    if "answer" in data_dir:
                        dt[field]["prediction"] = load_text(line_fields[-2])
                    
                    if len(all_data) > i:
                        assert dt["question"] == all_data[i]["question"], (dt["question"], all_data[i]["question"])
                        all_data[i][field] = dt[field]
                    else:
                        all_data.append(dt)

            train_len = i + 1
            with open(data_dir + "/dev.tsv", 'r') as f:
                for i, line in enumerate(f):
                    line_fields = line.strip().split("\t")
                    dt = {"question": line_fields[0], field: {"answer": line_fields[-1]}}
                    if "reasoning" in data_dir:
                        dt[field]["reasoning"] = load_text(line_fields[1])
                    if "answer" in data_dir:
                        dt[field]["prediction"] = load_text(line_fields[-2])
                    
                    if len(all_data) > train_len + i:
                        assert dt["question"] == all_data[train_len + i]["question"]
                        all_data[train_len + i][field] = dt[field]
                    else:
                        all_data.append(dt)
        else:
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
    print ("Processed %d datapoints" % len(all_data))

    if args.start_index != 0:
        all_data = all_data[args.start_index:]
    if args.num_samples > 0:
        all_data = all_data[:args.num_samples]

    llm_model = E2E_MODEL(args, no_load_model=True)

    output_data = []
    skipped = 0
    preplan_dts, strategy_dts, verify_dts, sec_preplan_dts, sec_strategy_dts = 0, 0, 0, 0, 0
    for dt in all_data:
        preplan_system_prompt, preplan_instruction, input = llm_model._get_decomposed_prompt_preplan(dt)
        correct_fields = [ field for field in args.fields if dt[field]["answer"] == "1" ]
        if args.high_impact and (len(correct_fields) == 0 or len(correct_fields) == args.fields):
            continue

        skip_postplan = 0
        if (args.remove_neg or args.remove_neg_postplan) and len(correct_fields) == 0 and not args.is_test:
            skipped += 1
            if args.remove_neg_postplan:
                skip_postplan = 1
            else:
                continue
        elif args.sep_neg and len(correct_fields) == 0 and not args.is_test:
            correct_fields = ["None"]

        incorrect_fields = [ field for field in args.fields if field not in correct_fields ]

        correct_fields = correct_fields if len(correct_fields) > 0 else [args.fields[-1]]
        if (args.include_best or args.is_test) and len(correct_fields) > 1:    
            correct_fields = [correct_fields[0]]
        
        if len(correct_fields) == 0:
            skipped += 1

        for i, field in enumerate(correct_fields):
            if args.single_turn:
                assert field not in ["rag"], "RAG can't operate in single-turn setting"
                single_turn_output = create_single_turn_preplan_answer_output(field, dt, field_name=llm_model.field2strategy[field], force_output=args.is_test)
                if single_turn_output is None:
                    assert not args.is_test
                    skipped += 1
                    continue
                single_turn_datapoint = {
                    "system": preplan_system_prompt,
                    "instruction": preplan_instruction,
                    "input": input,
                    "output": single_turn_output
                }
                output_data.append(single_turn_datapoint)
                preplan_dts += 1

                if args.verify:
                    if field in incorrect_fields:
                        continue
                    system_prompt, instruction, history = llm_model._get_decomposed_prompt_verify([single_turn_datapoint])
                    answer_output = "Yes"
                    verify_datapoint = {
                        "system": system_prompt,
                        "instruction": instruction,
                        "input": "",
                        "output": answer_output,
                        "history": history
                    }
                    output_data.append(verify_datapoint)
                    verify_dts += 1
            else:
                preplan_output = create_preplan_output(llm_model.field2strategy[field]) if field in llm_model.field2strategy else "None"
                preplan_datapoint = {
                    "system": preplan_system_prompt,
                    "instruction": preplan_instruction,
                    "input": input,
                    "output": preplan_output
                }
                if args.include_best_preplan:
                    if i == 0:
                        output_data.append(preplan_datapoint)
                        preplan_dts += 1
                else:
                    output_data.append(preplan_datapoint)
                    preplan_dts += 1

                if skip_postplan:
                    continue

                system_prompt, instruction, history = llm_model._get_decomposed_prompt_strategy_use([preplan_datapoint])
                retrieved_input = ""
                if field == "rag":
                    retrieved_input, answer_output = clean_final_answer(dt[field]["reasoning"], rag=True)
                else:
                    answer_output = clean_final_answer(dt[field]["reasoning"])
                strategy_use_datapoint = {
                    "system": system_prompt,
                    "instruction": instruction,
                    "input": retrieved_input,
                    "output": answer_output,
                    "history": history
                }
                output_data.append(strategy_use_datapoint)
                strategy_dts += 1

                if args.verify:
                    if field in incorrect_fields:
                        continue
                    system_prompt, instruction, history = llm_model._get_decomposed_prompt_verify([preplan_datapoint, strategy_use_datapoint])
                    answer_output = "Yes"
                    verify_datapoint = {
                        "system": system_prompt,
                        "instruction": instruction,
                        "input": "",
                        "output": answer_output,
                        "history": history
                    }
                    output_data.append(verify_datapoint)
                    verify_dts += 1
        
        if args.verify:
            for i, field in enumerate(incorrect_fields):
                if args.single_turn:
                    single_turn_output = create_single_turn_preplan_answer_output(field, dt, field_name=llm_model.field2strategy[field], force_output=args.is_test)
                    single_turn_datapoint = {
                        "system": preplan_system_prompt,
                        "instruction": preplan_instruction,
                        "input": input,
                        "output": single_turn_output
                    }

                    system_prompt, instruction, history = llm_model._get_decomposed_prompt_verify([single_turn_datapoint])
                    answer_output = "No"
                    verify_datapoint = {
                        "system": system_prompt,
                        "instruction": instruction,
                        "input": "",
                        "output": answer_output,
                        "history": history
                    }
                    output_data.append(verify_datapoint)
                    verify_dts += 1

                else:
                    preplan_output = create_preplan_output(llm_model.field2strategy[field]) if field in llm_model.field2strategy else "None"
                    preplan_datapoint = {
                        "system": preplan_system_prompt,
                        "instruction": preplan_instruction,
                        "input": input,
                        "output": preplan_output
                    }

                    system_prompt, instruction, history = llm_model._get_decomposed_prompt_strategy_use([preplan_datapoint])
                    answer_output = clean_final_answer(dt[field]["reasoning"])
                    strategy_use_datapoint = {
                        "system": system_prompt,
                        "instruction": instruction,
                        "input": "",
                        "output": answer_output,
                        "history": history
                    }

                    system_prompt, instruction, history = llm_model._get_decomposed_prompt_verify([preplan_datapoint, strategy_use_datapoint])
                    answer_output = "No"
                    verify_datapoint = {
                        "system": system_prompt,
                        "instruction": instruction,
                        "input": "",
                        "output": answer_output,
                        "history": history
                    }
                    output_data.append(verify_datapoint)
                    verify_dts += 1
                    
                    if len(incorrect_fields) != len(args.fields):
                        for j, correct_field in enumerate(correct_fields):
                            system_prompt, instruction, history = llm_model._get_decomposed_prompt_sec_preplan([preplan_datapoint, strategy_use_datapoint, verify_datapoint], used_fields=[field])
                            answer_output = create_preplan_output(llm_model.field2strategy[correct_field]) if correct_field in llm_model.field2strategy else "None"
                            sec_preplan_datapoint = {
                                "system": system_prompt,
                                "instruction": instruction,
                                "input": "",
                                "output": answer_output,
                                "history": history
                            }
                            if args.include_best_preplan:
                                if j == 0:
                                    output_data.append(sec_preplan_datapoint)
                                    sec_preplan_dts += 1
                            else:
                                output_data.append(sec_preplan_datapoint)
                                sec_preplan_dts += 1

                            system_prompt, instruction, history = llm_model._get_decomposed_prompt_strategy_use([preplan_datapoint, strategy_use_datapoint, verify_datapoint, sec_preplan_datapoint])
                            retrieved_input = ""
                            if correct_field == "rag":
                                retrieved_input, answer_output = clean_final_answer(dt[correct_field]["reasoning"], rag=True)
                            else:
                                answer_output = clean_final_answer(dt[correct_field]["reasoning"])
                            sec_strategy_datapoint = {
                                "system": system_prompt,
                                "instruction": instruction,
                                "input": retrieved_input,
                                "output": answer_output,
                                "history": history
                            }
                            output_data.append(sec_strategy_datapoint)
                            sec_strategy_dts += 1

    print ("Skipped: %d" % skipped)
    print ("Preplan pts: %d\tStrategy pts: %d\tVerify pts: %d\tSec Preplan pts: %d\tSec Strategy pts: %d" % (preplan_dts, strategy_dts, verify_dts, sec_preplan_dts, sec_strategy_dts))
    print ("Created output data: %d datapoints" % len(output_data))

    if args.balance_data:
        if args.single_turn:
            output_data = balance_data(output_data, [ llm_model.field2strategy[field] for field in args.fields ])
        else:
            if not args.remove_neg_postplan:
                output_data = balance_data(output_data, [ llm_model.field2strategy[field] for field in args.fields ], history=0)
            output_data = balance_data(output_data, [ llm_model.field2strategy[field] for field in args.fields ], history=1)
            if args.verify_balance:
                output_data = balance_data(output_data, [ "Yes", "No" ], history=2, verify=1)
        print ("Created output data: %d datapoints" % len(output_data))

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    with open(args.save_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
if __name__ == "__main__":
    main()