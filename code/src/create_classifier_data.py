# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import random

def read_gold(gold_file):
    data = []
    with open(gold_file, 'r') as f:
        data = json.load(f)
    return data

def read_output(output_file):
    data = []
    with open(output_file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def iscorrect(datapoint, metric="F1", threshold=0.5):
    return float(datapoint[metric]) > threshold

def clean_final_answer(text):
    try:
        text = text.split("Final answer:")[0] + "Final answer:" + '"'.join(text.split("Final answer:")[1].split('"')[:2]) + '"'
        return text.strip()
    except:
        return text.strip()

def clean_retrieval(text):
    text_splits = text.split("[R\n")[1].split("\n\n]\n")
    retrieved_text = text_splits[0].split("Context:")[1]
    cot_reasoning = clean_final_answer(text_splits[1])
    return "Context: %s\n\nReasoning: %s" % (retrieved_text, cot_reasoning)

def balance_data(data, fields):
    field_name = fields[0]
    pos_examples = [ dt for dt in data if dt[field_name] == 1]
    neg_examples = [ dt for dt in data if dt[field_name] == 0]
    if len(pos_examples) > len(neg_examples):
        examples_added = len(pos_examples)-len(neg_examples)
        neg_examples += random.choices(neg_examples, k=examples_added)
    elif len(pos_examples) < len(neg_examples):
        examples_added = len(neg_examples)-len(pos_examples)
        pos_examples += random.choices(pos_examples, k=examples_added)
    assert len(neg_examples) == len(pos_examples)
    print ("Balanced data. Added %d examples" % (examples_added))
    balanced_data = pos_examples + neg_examples
    random.shuffle(balanced_data)
    return balanced_data

def write_as_tsv(data, filename, fields, is_binary=False, include_answer=False, include_reasoning=False):
    print ("Creating file: %s" % filename)
    with open(filename, 'w') as f:
        if is_binary:
            assert len(fields) == 1
            field_name = fields[0]
            for dt in data:
                all_text = [dt["question"]]
                if include_reasoning:
                    all_text.append(json.dumps(dt["reasoning"]))
                if include_answer:
                    all_text.append(json.dumps(dt["pred_answer"]))
                all_text.append(str(dt[field_name]))
                f.write("%s\n" % "\t".join(all_text))
        else:
            for dt in data:
                correct_label = [ i for i, field in enumerate(fields) if dt[field] == 1 ]
                if len(correct_label) == 0:
                    correct_label = 0
                else:
                    correct_label = correct_label[0] + 1

                all_text = [dt["question"]]
                for field in fields:
                    if include_reasoning:
                        all_text.append(json.dumps(dt[field + "_reasoning"]))
                    if include_answer:
                        all_text.append(json.dumps(dt[field + "_pred_answer"]))
                all_text.append(str(correct_label))
                f.write("%s\n" % "\t".join(all_text))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=str, nargs='+', required=True)
    parser.add_argument("--data_dir", type=str, required=False)
    parser.add_argument("--details_file", type=str, required=False)
    parser.add_argument("--outputs_file", type=str, required=False)
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--neg_details_files", type=str, nargs='*', default=[], required=False)
    parser.add_argument("--is_binary", type=int, default=1)
    parser.add_argument("--include_reasoning", type=int, default=0)
    parser.add_argument("--include_answer", type=int, default=0)
    parser.add_argument("--dev_split", type=float, default=0.1)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--balanced", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=20000)
    args = parser.parse_args()
    if args.is_binary:
        assert len(args.fields) == 1
    assert args.data_dir or args.details_file
    if args.include_answer or args.include_reasoning:
        assert args.outputs_file or args.data_dir
    return args

def main():
    args = get_args()
    print (f"{args}")

    neg_qids = None
    if len(args.neg_details_files) > 0:
        for filename in args.neg_details_files:
            details = read_output(filename)
            qids = [ dt["qid"] for dt in details if not iscorrect(dt) ]
            if neg_qids == None:
                neg_qids = qids
            else:
                neg_qids = [ qid for qid in qids if qid in neg_qids ]
        print ("Removing %d sentences based on negative details files" % len(neg_qids))
    if neg_qids == None:
        neg_qids = []

    input_data = []
    for field in args.fields:
        if args.details_file:
            details_file = args.details_file % field
        else:
            details_file = args.data_dir + "/%s_all_s1000/details.txt" % field
        details = read_output(details_file)
        details = details[:args.num_samples]

        if args.include_answer or args.include_reasoning:
            if args.outputs_file:
                outputs_file = args.outputs_file % field
            else:
                outputs_file = args.data_dir + "/%s_all_s1000/output.txt" % field
            outputs = read_output(outputs_file)
            assert len(details) == len(outputs)
        
        if len(input_data) == 0:
            input_data = [ {"qid": details[i]["qid"]} for i in range(len(details)) ]

        if args.include_answer or args.include_reasoning:
            for i, (detail, output) in enumerate(zip(details, outputs)):
                assert input_data[i]["qid"] == detail["qid"]
                input_data[i][field] = 1 if iscorrect(detail) else 0
                if args.is_binary:
                    if field == "selfask":
                        input_data[i]["reasoning"] = clean_final_answer(output["prediction"]).strip()
                    elif "rag" in field:
                        input_data[i]["reasoning"] = clean_retrieval(output["debug_prediction"]).strip()
                    else:
                        input_data[i]["reasoning"] = output["prediction"].split("\n")[0].strip()
                    input_data[i]["pred_answer"] = detail["final_pred"].split("\n")[0].strip()
                else:
                    if field == "selfask":
                        input_data[i][field + "_reasoning"] = clean_final_answer(output["prediction"]).strip()
                    elif "rag" in field:
                        input_data[i][field + "_reasoning"] = clean_retrieval(output["debug_prediction"]).strip()
                    else:
                        input_data[i][field + "_reasoning"] = output["prediction"].split("\n")[0].strip()
                    input_data[i][field + "_pred_answer"] = detail["final_pred"].split("\n")[0].strip
        else:
            for i, detail in enumerate(details):
                assert input_data[i]["qid"] == detail["qid"]
                input_data[i][field] = 1 if iscorrect(detail) else 0

    gold_data = read_gold(args.gold_file)
    for i, gold_dt in enumerate(gold_data[:len(input_data)]):
        input_data[i]["question"] = gold_dt["question"].strip()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.dev_split == 0:
        write_as_tsv(input_data, args.save_dir + "/test.tsv", args.fields, is_binary=args.is_binary, include_answer=args.include_answer, include_reasoning=args.include_reasoning)
        with open(args.save_dir + "/labels.txt", 'r') as f:
            train_labels = []
            for line in f:
                train_labels.append(line.strip())
        test_labels = ["none"] + [field for field in args.fields]
        assert train_labels == test_labels
    else:
        if args.shuffle:
            random.shuffle(input_data)
        train_num = int((1-args.dev_split)*len(input_data))
        train_data = input_data[:train_num]
        dev_data = input_data[train_num:]
        
        train_data = [ dt for dt in train_data if dt["qid"] not in neg_qids ]
        if args.balanced and args.is_binary:
            write_as_tsv(balance_data(train_data, args.fields), args.save_dir + "/train.tsv", args.fields, is_binary=args.is_binary, include_answer=args.include_answer, include_reasoning=args.include_reasoning)
        else:
            write_as_tsv(train_data, args.save_dir + "/train.tsv", args.fields, is_binary=args.is_binary, include_answer=args.include_answer, include_reasoning=args.include_reasoning)
        write_as_tsv(dev_data, args.save_dir + "/dev.tsv", args.fields, is_binary=args.is_binary, include_answer=args.include_answer, include_reasoning=args.include_reasoning)
        
        train_labels = ["none"] + [field for field in args.fields]
        with open(args.save_dir + "/labels.txt", 'w') as f:
            f.write("\n".join(train_labels))

if __name__ == "__main__":
    main()
