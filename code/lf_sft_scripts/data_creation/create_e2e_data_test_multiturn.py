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
    parser.add_argument("--verify", type=int, default=0, required=False)
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

def balance_data(data, field_names):
    examples_per_class = { field_name: [ dt for dt in data if field_name in dt["output"] ] for field_name in field_names }
    num_examples_per_class = { field_name: len(examples_per_class[field_name]) for field_name in examples_per_class }
    max_examples = max([num_examples_per_class[field_name] for field_name in examples_per_class])
    addn_examples = []
    for field_name in examples_per_class:
        examples_added = max_examples - num_examples_per_class[field_name]
        addn_examples += random.choices(examples_per_class[field_name], k=examples_added)
    print (num_examples_per_class)
    print ("Balanced data. Added %d examples" % (len(addn_examples)))
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

    llm_model = E2E_MODEL(args, no_load_model=True)

    output_data_preplan, output_data_answer_fields, output_data_verify = [], { field:[] for field in args.fields }, { field:[] for field in args.fields }
    for dt in all_data:
        preplan_system_prompt, preplan_instruction, input = llm_model._get_decomposed_prompt_preplan(dt)

        correct_fields = [ field for field in args.fields if dt[field]["answer"] == "1" ]
        incorrect_fields = [ field for field in args.fields if field not in correct_fields ]
        correct_field = correct_fields[0] if len(correct_fields) > 0 else args.fields[-1]
        preplan_output = create_preplan_output(llm_model.field2strategy[correct_field]) if correct_field in llm_model.field2strategy else "None"
        preplan_datapoint = {
            "system": preplan_system_prompt,
            "instruction": preplan_instruction,
            "input": input,
            "output": preplan_output
        }
        output_data_preplan.append(preplan_datapoint)

        for field in args.fields:
            preplan_output = create_preplan_output(llm_model.field2strategy[field]) if field in llm_model.field2strategy else "None"
            preplan_datapoint = {
                "system": preplan_system_prompt,
                "instruction": preplan_instruction,
                "input": input,
                "output": preplan_output
            }

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
            output_data_answer_fields[field].append(strategy_use_datapoint)

            system_prompt, instruction, history = llm_model._get_decomposed_prompt_verify([preplan_datapoint, strategy_use_datapoint])
            answer_output = "Yes" if field not in incorrect_fields else "No"
            verify_datapoint = {
                "system": system_prompt,
                "instruction": instruction,
                "input": "",
                "output": answer_output,
                "history": history
            }
            output_data_verify[field].append(verify_datapoint)

    len_output_data_answer_fields = [ len(output_data_answer_fields[field]) for field in args.fields ]
    len_output_data_verify_fields = [ len(output_data_verify[field]) for field in args.fields ]
    assert min(len_output_data_answer_fields) == max(len_output_data_answer_fields)
    assert min(len_output_data_verify_fields) == max(len_output_data_verify_fields)
    print ("Created output preplan data: %d datapoints" % len(output_data_preplan))
    print ("Created output answer data: %d datapoints" % len_output_data_answer_fields[0])
    print ("Created output verify data: %d datapoints" % len_output_data_verify_fields[0])

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    preplan_filename = args.save_file.split(".json")[0] + "_preplan" + ".json"
    print("Creating file: %s" % preplan_filename)
    with open(preplan_filename, "w") as f:
        json.dump(output_data_preplan, f, indent=2)
    for field in args.fields:
        answer_field_filename = args.save_file.split(".json")[0] + "_answer-" + field + ".json"
        print("Creating file: %s" % answer_field_filename)
        with open(answer_field_filename, "w") as f:
            json.dump(output_data_answer_fields[field], f, indent=2)
        
        if args.verify:
            verify_field_filename = args.save_file.split(".json")[0] + "_verify-" + field + ".json"
            print("Creating file: %s" % verify_field_filename)
            with open(verify_field_filename, "w") as f:
                json.dump(output_data_verify[field], f, indent=2)
    
if __name__ == "__main__":
    main()