# This code is taken and modified from - https://github.com/oneal2000/DRAGIN/tree/main
# This source code is licensed under the license found here - https://github.com/oneal2000/DRAGIN/blob/main/LICENSE

import os
import json
import argparse
import logging
import spacy
import torch
import numpy as np
import pandas as pd
import difflib
from tqdm import tqdm
from data import StrategyQA, WikiMultiHopQA, WikiMultiHopQA_PE, HotpotQA, HotpotQA_PE, Bamboogle, Bamboogle_PE, CompositionalCelebrities, CompositionalCelebrities_PE, Musique, Musique_PE, IIRC
from transformers import AutoTokenizer, AutoModelForCausalLM
from main import set_seed

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

class Counter:
    def __init__(self):
        self.retrieve = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        # if len(ids) > 100:
        #     import ipdb; ipdb.set_trace()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self):
        return {
            "retrieve_count": self.retrieve, 
            "token_count": self.token, 
            "sentence_count": self.sentence 
        }

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--pred_dirs", type=str, nargs='+', default=[], required=True)
    parser.add_argument("-s", "--seed", type=int, default=1000)
    args = parser.parse_args()
    set_seed(args.seed)
    return args

def get_strategy(text):
    if "Strategy:" not in text:
        # import ipdb; ipdb.set_trace()
        return None
    strategy = text.split("Strategy:")[1].split("\n")[0].strip()
    return strategy

def get_question(text):
    assert "Question:" in text
    question = text.split("Question:")[1]
    if "assistant\n" in question:
        question = question.split("assistant\n")[0].strip()
    elif "?  " in question:
        question = question.split("?  ")[0]
    else:
        question = question.split("  ")[0]
    return question.strip()

def regenerate_answer(cot, tokenizer, model, case, demo, final_answer_str = "Final answer:", prompt_answer_str = " Final answer: ", llama3=False):
    split_words = ["Question:", "#10000000", "Note:"]
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
    return text

def main():
    args = get_args()
    logger.info(f"{args}")

    if args.dataset == 'strategyqa':
        data = StrategyQA(args.data_path)
    elif args.dataset == "2wikimultihopqa":
        if "filename" in args and args.filename != "":\
            data = WikiMultiHopQA(args.data_path, filename=args.filename, start_index=args.start_index)
        else:
            data = WikiMultiHopQA(args.data_path)
    elif args.dataset == "2wikimultihopqa_pe":
        data = WikiMultiHopQA_PE(args.data_path)
    elif args.dataset == "hotpotqa":
        if "filename" in args and args.filename != "":
            data = HotpotQA(args.data_path, filename=args.filename, start_index=args.start_index)
        else:
            data = HotpotQA(args.data_path)
    elif args.dataset == "hotpotqa_pe":
        if "filename" in args and args.filename != "":
            data = HotpotQA_PE(args.data_path, filename=args.filename)
        else:
            data = HotpotQA_PE(args.data_path)
    elif args.dataset == 'bamboogle':
        data = Bamboogle(args.data_path)
    elif args.dataset == "bamboogle_pe":
        data = Bamboogle_PE(args.data_path)
    elif args.dataset == "compositionalcelebrities":
        data = CompositionalCelebrities(args.data_path)
    elif args.dataset == "compositionalcelebrities_pe":
        data = CompositionalCelebrities_PE(args.data_path)
    elif args.dataset == "musique":
        if "filename" in args and args.filename != "":
            data = Musique(args.data_path, filename=args.filename, start_index=args.start_index)
        else:
            data = Musique(args.data_path)
    elif args.dataset == "musique_pe":
        data = Musique_PE(args.data_path)
    elif args.dataset == 'iirc':
        data = IIRC(args.data_path)
    else:
        raise NotImplementedError
    final_answer_str = data.final_answer_str
    prompt_answer_str = data.prompt_answer_str
    data.format(fewshot=0)

    dataset = {}
    for i in range(len(data.dataset)):
        t = data.dataset[i]
        dataset[t["qid"]] = [
            t["answer"], 
            t["answer_id"] if "answer_id" in t else None,
            t["case"] if "case" in t else None
        ]

    for pred_dir in args.pred_dirs:
        print (pred_dir)
        
        # if os.path.exists(f"{pred_dir}/result.tsv"):
        #     continue

        metrics = ["EM", "F1", "Precision", "Recall", "EM_Orig", "F1_Orig", "Precision_Orig", "Recall_Orig", "EM_Round1", "F1_Round1", "Precision_Round1", "Recall_Round1", "Strategy_Accuracy"]
        metrics += ["retrieve_count", "token_count", "sentence_count"]
        value = [[] for _ in range(len(metrics))]
        analysis_value = {}
        with open(pred_dir + "/generated_predictions.jsonl", "r") as fin:
            lines = fin.readlines()
        if args.dataset == "hotpotqa":
            with open(args.data_path + "/hotpotqa-dev.json", "r") as fin:
                orig_test_data = json.load(fin)
        elif args.dataset == "2wikimultihopqa":
            with open(args.data_path + "/dev.json", "r") as fin:
                orig_test_data = json.load(fin)
        elif args.dataset == "bamboogle":
            with open(args.data_path + "/test.json", "r") as fin:
                orig_test_data = json.load(fin)
        elif args.dataset == "musique":
            with open(args.data_path + "/final_dev.json", "r") as fin:
                orig_test_data = json.load(fin)
        else:
            raise NotImplementedError
        
        need_generate = args.dataset in ['2wikimultihopqa', "2wikimultihopqa_pe", "hotpotqa", "hotpotqa_pe", "hotpotqa_pe2", "iirc", "strategyqa", "bamboogle", "bamboogle_pe", "compositionalcelebrities", "compositionalcelebrities_pe", "musique", "musique_pe"] 
        need_generate = False
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if need_generate:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto",
                                                        trust_remote_code = "falcon" in args.model_name_or_path)
            demo = data.dataset[0]["demo"]
            llama3 = True if "Llama-3" in args.model_name_or_path else False

        pred_out = open(f"{pred_dir}/details.txt", "w")
        unique_strategies, no_final_answer = [], 0
        for i, line in tqdm(enumerate(lines)):
            try:
                rd = json.loads(line)
            except:
                print (line)
                assert False
            orig_pred, orig_strategy = rd["label"], get_strategy(rd["label"]) if "preplan_label" not in rd else rd["preplan_label"] if "round1_preplan_label" not in rd else rd["round1_preplan_label"] if "round2_preplan_label" not in rd else rd["round2_preplan_label"]
            pred, strategy = rd["predict"], rd["round2_preplan_predict"] if "round2_preplan_predict" in rd else rd["round1_preplan_predict"] if "round1_preplan_predict" in rd else rd["preplan_predict"] if "preplan_predict" in rd else get_strategy(rd["predict"])
            if "round2_preplan_predict" in rd:
                round1_pred, round1_strategy = rd["round1_strategy_use_predict"] if "round1_strategy_use_predict" in rd else rd["predict"], rd["round1_preplan_predict"] if "round1_preplan_predict" in rd else None
            else:
                round1_pred, round1_strategy = pred, None
            question = get_question(rd["prompt"])
            unique_strategies = list(set(unique_strategies + [orig_strategy, strategy, round1_strategy]))
            qid = orig_test_data[i]["_id"] if "_id" in orig_test_data[i] else orig_test_data[i]["qid"]
            question_edit_distance = difflib.SequenceMatcher(lambda x: x in " ", orig_test_data[i]["question"].strip(), question).ratio()
            assert orig_test_data[i]["question"].strip() == question or question_edit_distance > 0.98, (orig_test_data[i]["question"], question, question_edit_distance)

            counter = Counter()
            # if len(rd["round1_preplan_predict"]) > 6 or ("round2_preplan_predict" in rd and len(rd["round2_preplan_predict"]) > 6):
            #     import ipdb; ipdb.set_trace()
            for key in rd:
                if "predict" in key:
                    counter.add_generate(rd[key], tokenizer)
            if strategy in [ "RAG" ]:
                counter.retrieve += 1
            if round1_strategy in [ "RAG" ]:
                counter.retrieve += 1

            ground_truth, ground_truth_id, case = dataset[qid]
            if need_generate:
                pred = regenerate_answer(pred, tokenizer, model, case, demo, final_answer_str=final_answer_str, prompt_answer_str=prompt_answer_str, llama3=llama3) 
            pred = data.get_real_prediction(pred)
            orig_pred = data.get_real_prediction(orig_pred)
            round1_pred = data.get_real_prediction(round1_pred)
            if pred == '':
                no_final_answer += 1

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
            em_orig_ret = data.exact_match_score(
                orig_pred, 
                ground_truth, 
                ground_truth_id
            )
            f1_orig_ret = data.f1_score(
                orig_pred, 
                ground_truth, 
                ground_truth_id
            )
            em_round1_ret = data.exact_match_score(
                round1_pred, 
                ground_truth, 
                ground_truth_id
            )
            f1_round1_ret = data.f1_score(
                round1_pred, 
                ground_truth, 
                ground_truth_id
            )
            value[0].append(em_ret["correct"])
            for i, k in enumerate(f1_ret.keys()):
                value[i+1].append(f1_ret[k])
            value[4].append(em_orig_ret["correct"])
            for i, k in enumerate(f1_orig_ret.keys()):
                value[i+5].append(f1_orig_ret[k])
            value[8].append(em_round1_ret["correct"])
            for i, k in enumerate(f1_round1_ret.keys()):
                value[i+9].append(f1_round1_ret[k])
            value[12].append(int(strategy == orig_strategy))
            counts = counter.calc()
            value[13].append(counts["retrieve_count"])
            value[14].append(counts["token_count"])
            value[15].append(counts["sentence_count"])
                
            detail = {
                "qid": qid,
                "orig_pred": orig_pred, 
                "final_pred": pred,
                "EM": str(em_ret["correct"]), 
                "F1": str(f1_ret["f1"]),
                "orig_EM": str(em_orig_ret["correct"]),
                "orig_F1": str(f1_orig_ret["f1"]),
                "strategy_match": str(int(strategy == orig_strategy))
            }
            pred_out.write(json.dumps(detail)+"\n")

            # analysis values        
            if orig_strategy != None:
                met_name = "orig_" + orig_strategy + "_EM"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(em_orig_ret["correct"])
                met_name = "orig_" + orig_strategy + "_F1"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(f1_orig_ret["f1"])
                met_name = "orig_" + orig_strategy + "_Prec"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(f1_orig_ret["precision"])
                met_name = "orig_" + orig_strategy + "_Rec"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(f1_orig_ret["recall"])
                met_name = "strategy_" + orig_strategy + "_recall"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(int(strategy == orig_strategy))
                met_name = "counts_" + orig_strategy
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(1)

            if strategy != None:
                met_name = strategy + "_EM"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(em_ret["correct"])
                met_name = strategy + "_F1"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(f1_ret["f1"])
                met_name = strategy + "_Prec"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(f1_ret["precision"])
                met_name = strategy + "_Rec"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(f1_ret["recall"])
                met_name = "strategy_" + strategy + "_precision"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(int(strategy == orig_strategy))
                met_name = "counts_" + strategy
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(1)

            if orig_strategy != None and strategy != None and orig_strategy == strategy:
                met_name = "strategy_match_orig_EM"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(em_orig_ret["correct"])
                met_name = "strategy_match_orig_F1"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(f1_orig_ret["f1"])
                met_name = "strategy_match_orig_Prec"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(f1_orig_ret["precision"])
                met_name = "strategy_match_orig_Rec"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(f1_orig_ret["recall"])

                met_name = "strategy_match_EM"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(em_ret["correct"])
                met_name = "strategy_match_F1"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(f1_ret["f1"])
                met_name = "strategy_match_Prec"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(f1_ret["precision"])
                met_name = "strategy_match_Rec"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(f1_ret["recall"])

            elif orig_strategy != None and strategy != None:
                met_name = "strategy_notmatch_orig_EM"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(em_orig_ret["correct"])
                met_name = "strategy_notmatch_orig_F1"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(f1_orig_ret["f1"])
                met_name = "strategy_notmatch_orig_Prec"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(f1_orig_ret["precision"])
                met_name = "strategy_notmatch_orig_Rec"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(f1_orig_ret["recall"])

                met_name = "strategy_notmatch_EM"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(em_ret["correct"])
                met_name = "strategy_notmatch_F1"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(f1_ret["f1"])
                met_name = "strategy_notmatch_Prec"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(f1_ret["precision"])
                met_name = "strategy_notmatch_Rec"
                if met_name not in analysis_value:
                    analysis_value[met_name] = []
                analysis_value[met_name].append(f1_ret["recall"])

        ret = []
        ret_verbose = []
        for i, metric in enumerate(metrics):
            val = np.array(value[i])
            ret_verbose.append([metric, val.mean()])
            if "rec" not in metric.lower():
                ret.append([metric, val.mean()])
        for metname in analysis_value:
            mean_value = np.array(analysis_value[metname]).mean()
            ret_verbose.append([metname, mean_value])
            if "rec" not in metname.lower():
                ret.append([metname, mean_value])

        for strategy in unique_strategies:
            if strategy is None:
                continue
            if "strategy_" + strategy + "_precision" in analysis_value and "strategy_" + strategy + "_recall" in analysis_value:
                prec = np.array(analysis_value["strategy_" + strategy + "_precision"]).mean()
                rec = np.array(analysis_value["strategy_" + strategy + "_recall"]).mean()
                f1 = 2*prec*rec/(prec+rec)
                ret.append(["strategy_" + strategy + "_f1", f1])
            else:
                ret.append(["strategy_" + strategy + "_f1", 0.0])
        df = pd.DataFrame(ret)
        df.to_csv(f"{pred_dir}/result.tsv", index=False, header=False)
        df_verbose = pd.DataFrame(ret_verbose)
        df_verbose.to_csv(f"{pred_dir}/result_verbose.tsv", index=False, header=False)

        logger.info(f"No final answer for {no_final_answer}/{len(lines)}")


if __name__ == "__main__":
    main()