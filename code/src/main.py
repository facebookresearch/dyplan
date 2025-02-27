# This code is taken and modified from - https://github.com/oneal2000/DRAGIN/tree/main
# This source code is licensed under the license found here - https://github.com/oneal2000/DRAGIN/blob/main/LICENSE

import os
import json
import argparse
from tqdm import tqdm
from copy import copy
import logging
from data import WikiMultiHopQA, HotpotQA, Musique
from generate import *
import random

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

def set_seed(seed):
    logger.info(f"Setting random seed: {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("-d", "--debug_mode", action='store_true', default=False)
    parser.add_argument("-s", "--seed", type=int, default=1000)
    parser.add_argument("-o", "--output_str", type=str, default="")
    args = parser.parse_args()
    set_seed(args.seed)
    config_path = args.config_path
    debug_mode = args.debug_mode
    seed = args.seed
    output_str = args.output_str
    with open(config_path, "r") as f:
        args = json.load(f)
    if "start_index" not in args:
        args["start_index"] = 0
    args = argparse.Namespace(**args)
    args.config_path = config_path
    args.debug_mode = debug_mode
    args.seed = seed
    args.output_str = output_str
    if "shuffle" not in args:
        args.shuffle = False 
    if "use_counter" not in args:
        args.use_counter = True
    return args


def main():
    args = get_args()
    logger.info(f"{args}")

    # output dir
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    dir_name = os.listdir(args.output_dir)
    if args.output_str == "": 
        for i in range(10000):
            if str(i) not in dir_name:
                args.output_dir = os.path.join(args.output_dir, str(i))
                os.makedirs(args.output_dir)
                break
    else:
        args.output_dir = os.path.join(args.output_dir, args.output_str)
        if args.output_str not in dir_name:
            os.makedirs(args.output_dir)
    logger.info(f"output dir: {args.output_dir}")

    start_idx = 0
    if args.output_str in dir_name:
        output_file = open(os.path.join(args.output_dir, "output.txt"), "r")
        start_idx = len(output_file.readlines())
        output_file = open(os.path.join(args.output_dir, "output.txt"), "a")
        logger.info(f"Loading existing run from {start_idx}")
    else:
        # save config
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(args.__dict__, f, indent=4)
        # create output file
        output_file = open(os.path.join(args.output_dir, "output.txt"), "w")

    # load data
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
    include_cot = True if "include_cot" not in args else args.include_cot
    data.format(fewshot=args.fewshot, include_cot=include_cot)
    data = data.dataset
    if args.shuffle:
        data = data.shuffle()
    if args.sample != -1:
        assert args.sample > start_idx, "Start idx: %d > args.sample: %d" % (start_idx, args.sample)
        samples = min(len(data), args.sample)
        data = data.select(range(samples))
   
    # 根据 method 选择不同的生成策略
    if args.method == "non-retrieval":
        model = BasicRAG(args)
    elif args.method == "single-retrieval":
        model = SingleRAG(args)
    elif args.method == "fix-length-retrieval" or args.method == "fix-sentence-retrieval":
        model = FixLengthRAG(args)
    elif args.method == "token":
        model = TokenRAG(args)
    elif args.method == "entity":
        model = EntityRAG(args)
    elif args.method == "attn_prob" or args.method == "dragin":
        model = AttnWeightRAG(args)
    elif args.method == "selfask":
        model = SelfAsk(args)
    else:
        raise NotImplementedError

    logger.info("start inference")
    for i in tqdm(range(len(data) - start_idx)):
        last_counter = copy(model.counter)
        batch = data[start_idx + i]
        if i == 0:
            logger.info(batch)
        pred, debug_pred = model.inference(batch["question"], batch["demo"], batch["case"], final_ans_str=final_answer_str, debug=args.debug_mode)
            
        if args.debug_mode:
            pred = pred.strip()
            debug_pred = debug_pred.strip()
            ret = {
                "qid": batch["qid"], 
                "prediction": pred,
                "debug_prediction": debug_pred
            }
        else:
            pred = pred.strip()
            ret = {
                "qid": batch["qid"], 
                "prediction": pred,
            }
        if args.use_counter:
            ret.update(model.counter.calc(last_counter))
        output_file.write(json.dumps(ret)+"\n")
        if i % 10 == 0:
            output_file.flush()
    output_file.flush()
    
if __name__ == "__main__":
    main()