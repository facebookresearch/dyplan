# This code is taken and modified from - https://github.com/oneal2000/DRAGIN/tree/main
# This source code is licensed under the license found here - https://github.com/oneal2000/DRAGIN/blob/main/LICENSE

import os
import json
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from data import WikiMultiHopQA, HotpotQA, Musique
from transformers import AutoTokenizer, AutoModelForCausalLM
from main import set_seed

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("-s", "--seed", type=int, default=1000)
    tmp = parser.parse_args()
    set_seed(tmp.seed)
    with open(os.path.join(tmp.dir, "config.json"), "r") as f:
        args = json.load(f)
    if "start_index" not in args:
        args["start_index"] = 0
    args = argparse.Namespace(**args)
    args.output_dir = tmp.dir
    return args


def regenerate_answer(cot, tokenizer, model, case, demo, final_answer_str = "the answer is", prompt_answer_str = " So the answer is ", llama3=False):
    # print("##### origin #####")
    # print(cot)
    split_words = ["Question:", "#10000000", "Note:"]
    # split_words = ["Question:", "#10000000", "\n"]
    for word in split_words:
        pos = cot.find(word)
        if pos != -1 and pos > 0:
            cot = cot[:pos]
    if final_answer_str in cot:
        return cot 

    cot += prompt_answer_str
    prompt = "".join([d["case"]+"\n" for d in demo])
    prompt += case + " " + cot
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    input_length = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    if llama3:
        terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]
        outputs = model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            max_new_tokens = 20, 
            eos_token_id = terminators,
            pad_token_id = tokenizer.eos_token_id
        )
    else:
        outputs = model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            max_new_tokens = 20)
    generated_tokens = outputs[:, input_length:]
    text = tokenizer.decode(generated_tokens[0])
    text = cot + text.strip()
    for word in split_words:
        pos = text.find(word)
        if pos != -1:
            text = text[:pos] 
    # print("##### prompt #####")
    # print(prompt)
    # print("##### output #####")
    # print(text)
    # print("##### pred #####")
    return text


def main():
    args = get_args()
    logger.info(f"{args}")
    
    if args.dataset == "2wikimultihopqa":
        if "filename" in args and args.filename != "":\
            data = WikiMultiHopQA(args.data_path, filename=args.filename, start_index=args.start_index)
        else:
            data = WikiMultiHopQA(args.data_path)
    elif args.dataset == "hotpotqa":
        if "filename" in args and args.filename != "":
            data = HotpotQA(args.data_path, filename=args.filename, start_index=args.start_index)
        else:
            data = HotpotQA(args.data_path)
    elif args.dataset == "musique":
        if "filename" in args and args.filename != "":
            data = Musique(args.data_path, filename=args.filename, start_index=args.start_index)
        else:
            data = Musique(args.data_path)
    else:
        raise NotImplementedError
    
    final_answer_str = data.final_answer_str
    prompt_answer_str = data.prompt_answer_str
    data.format(fewshot=args.fewshot)

    dataset = {}
    for i in range(len(data.dataset)):
        t = data.dataset[i]
        dataset[t["qid"]] = [
            t["answer"], 
            t["answer_id"] if "answer_id" in t else None,
            t["case"] if "case" in t else None
        ]

    metrics = ["EM", "F1", "Precision", "Recall"]
    if "use_counter" not in args or args.use_counter:
        count_list = ["retrieve_count", "generate_count", "hallucinated_count", "token_count", "sentence_count"]
        metrics += count_list
    value = [[] for _ in range(len(metrics))]
    with open(os.path.join(args.output_dir, "output.txt"), "r") as fin:
        lines = fin.readlines()
    
    need_generate = args.dataset in ['2wikimultihopqa', "hotpotqa", "musique"] 
    if need_generate:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto",
                                                     trust_remote_code = "falcon" in args.model_name_or_path)
        demo = data.dataset[0]["demo"]
        llama3 = True if "Llama-3" in args.model_name_or_path else False

    pred_out = open(f"{args.output_dir}/details.txt", "w")
    
    for line in tqdm(lines):
        rd = json.loads(line)
        qid = rd["qid"]
        pred = rd["prediction"]
        ground_truth, ground_truth_id, case = dataset[qid]
        if need_generate:
            pred = regenerate_answer(pred, tokenizer, model, case, demo, final_answer_str=final_answer_str, prompt_answer_str=prompt_answer_str, llama3=llama3) 
        pred = data.get_real_prediction(pred)

        em_ret = data.exact_match_score(
            pred, 
            ground_truth, 
            ground_truth_id
        )
        f1_ret = data.f1_score(
            pred, 
            ground_truth, 
            ground_truth_id
        )
        value[0].append(em_ret["correct"])
        for i, k in enumerate(f1_ret.keys()):
            value[i+1].append(f1_ret[k])
        if "use_counter" not in args or args.use_counter:
            for i, k in enumerate(count_list):
                value[i+4].append(rd[k])
        detail = {
            "qid": qid, 
            "final_pred": pred,
            "EM": str(em_ret["correct"]), 
            "F1": str(f1_ret["f1"]) 
        }
        pred_out.write(json.dumps(detail)+"\n")

    ret = []
    for i, metric in enumerate(metrics):
        val = np.array(value[i])
        ret.append([metric, val.mean()])
    df = pd.DataFrame(ret)
    df.to_csv(f"{args.output_dir}/result.tsv", index=False, header=False)


if __name__ == "__main__":
    main()