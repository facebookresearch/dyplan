# This code is taken and modified from - https://github.com/oneal2000/DRAGIN/tree/main
# This source code is licensed under the license found here - https://github.com/oneal2000/DRAGIN/blob/main/LICENSE

import numpy as np
import logging
import spacy
import torch
from math import exp
from scipy.special import softmax
from retriever import BM25, SGPT
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from selfask import *
from stop_word import StopWordCriteria

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")


class BasicGenerator:
    def __init__(self, model_name_or_path):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_config = AutoConfig.from_pretrained(model_name_or_path,
                    trust_remote_code = "falcon" in model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", 
                    trust_remote_code = "falcon" in model_name_or_path)
        self.llama3 = True if "Llama-3" in model_name_or_path else False
        self.instruct = True if "Instruct" in model_name_or_path else False
        if self.model_config.model_type == "llama" and not self.llama3:
            self.space_token = "▁"
        else:
            self.space_token = self.tokenizer.tokenize(' ')[0]
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.stop_words = ["Question"]

    def apply_chat_template(self, input_text, assistant_keyword=None, force_continue_generation=False):
        messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": ""},
            ]
        if assistant_keyword and assistant_keyword in input_text:
            user_text = assistant_keyword.join(input_text.split(assistant_keyword)[:-1]) + assistant_keyword
            assistant_text = input_text.split(assistant_keyword)[-1].strip()
            if assistant_text.strip() != "":
                messages.append({"role": "assistant", "content": assistant_text})
        else:
            user_text = input_text
        messages[1]["content"] = user_text

        if len(messages) == 2:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            input_text = input_text[:-10]                   # remove last eot_id token to force model to continue generation
            if force_continue_generation:
                input_text += "\n\n"
            return input_text

    def generate(self, input_text, max_length, temperature=None, top_p=None, return_logprobs=False, stop_on_question=False):
        # if self.instruct:
        #     input_text = self.apply_chat_template(input_text, assistant_keyword=assistant_keyword, force_continue_generation=force_continue_generation)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)
        stopping_criteria = [StopWordCriteria(tokenizer=self.tokenizer, prompts=[input_text], stop_words=self.stop_words)]
        sample = True if temperature is not None or top_p is not None else False

        if return_logprobs:
            if self.llama3:
                terminators = [
                                self.tokenizer.eos_token_id,
                                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                            ]
                if stop_on_question:
                    if sample:
                        outputs = self.model.generate(
                            input_ids = input_ids, 
                            attention_mask = attention_mask,
                            max_new_tokens = max_length, 
                            return_dict_in_generate = True, 
                            output_scores = True,
                            eos_token_id = terminators,
                            pad_token_id = self.tokenizer.eos_token_id,
                            temperature = temperature,
                            top_p = top_p,
                            stopping_criteria=stopping_criteria
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids = input_ids, 
                            attention_mask = attention_mask,
                            max_new_tokens = max_length, 
                            return_dict_in_generate = True, 
                            output_scores = True,
                            eos_token_id = terminators,
                            pad_token_id = self.tokenizer.eos_token_id,
                            do_sample=False,
                            stopping_criteria=stopping_criteria
                        )
                else:
                    if sample:
                        outputs = self.model.generate(
                            input_ids = input_ids, 
                            attention_mask = attention_mask,
                            max_new_tokens = max_length, 
                            return_dict_in_generate = True, 
                            output_scores = True,
                            eos_token_id = terminators,
                            pad_token_id = self.tokenizer.eos_token_id,
                            temperature = temperature,
                            top_p = top_p
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids = input_ids, 
                            attention_mask = attention_mask,
                            max_new_tokens = max_length, 
                            return_dict_in_generate = True, 
                            output_scores = True,
                            eos_token_id = terminators,
                            pad_token_id = self.tokenizer.eos_token_id,
                            do_sample=False,
                        )
            else:
                if stop_on_question:
                    if sample:
                        outputs = self.model.generate(
                            input_ids = input_ids, 
                            attention_mask = attention_mask,
                            max_new_tokens = max_length, 
                            return_dict_in_generate = True, 
                            output_scores = True,
                            temperature = temperature,
                            top_p = top_p,
                            stopping_criteria=stopping_criteria
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids = input_ids, 
                            attention_mask = attention_mask,
                            max_new_tokens = max_length, 
                            return_dict_in_generate = True, 
                            output_scores = True,
                            do_sample=False,
                            stopping_criteria=stopping_criteria,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                else:
                    if sample:
                        outputs = self.model.generate(
                            input_ids = input_ids, 
                            attention_mask = attention_mask,
                            max_new_tokens = max_length, 
                            return_dict_in_generate = True, 
                            output_scores = True,
                            temperature = temperature,
                            top_p = top_p,
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids = input_ids, 
                            attention_mask = attention_mask,
                            max_new_tokens = max_length, 
                            return_dict_in_generate = True, 
                            output_scores = True,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            generated_tokens = outputs.sequences[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0]) # text = "".join(tokens)
            tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            return text, tokens, logprobs
        
        else:
            if self.llama3:
                terminators = [
                                self.tokenizer.eos_token_id,
                                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                            ]
                if stop_on_question:
                    if sample:
                        outputs = self.model.generate(
                            input_ids = input_ids, 
                            attention_mask = attention_mask,
                            max_new_tokens = max_length, 
                            eos_token_id=terminators,
                            pad_token_id = self.tokenizer.eos_token_id,
                            temperature = temperature,
                            top_p = top_p,
                            stopping_criteria=stopping_criteria
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids = input_ids, 
                            attention_mask = attention_mask,
                            max_new_tokens = max_length, 
                            eos_token_id=terminators,
                            pad_token_id = self.tokenizer.eos_token_id,
                            do_sample=False,
                            stopping_criteria=stopping_criteria
                        )
                else:
                    if sample:
                        outputs = self.model.generate(
                            input_ids = input_ids, 
                            attention_mask = attention_mask,
                            max_new_tokens = max_length, 
                            eos_token_id=terminators,
                            pad_token_id = self.tokenizer.eos_token_id,
                            temperature = temperature,
                            top_p = top_p
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids = input_ids, 
                            attention_mask = attention_mask,
                            max_new_tokens = max_length, 
                            eos_token_id=terminators,
                            pad_token_id = self.tokenizer.eos_token_id,
                            do_sample=False
                        )
            else:
                if stop_on_question:
                    if sample:
                        outputs = self.model.generate(
                            input_ids = input_ids, 
                            max_new_tokens = max_length, 
                            attention_mask = attention_mask,
                            stopping_criteria=stopping_criteria,
                            temperature = temperature,
                            top_p = top_p
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids = input_ids, 
                            max_new_tokens = max_length, 
                            attention_mask = attention_mask,
                            do_sample=False,
                            stopping_criteria=stopping_criteria
                        )
                else:
                    if sample:
                        outputs = self.model.generate(
                            input_ids = input_ids, 
                            max_new_tokens = max_length, 
                            attention_mask = attention_mask,
                            temperature = temperature,
                            top_p = top_p
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids = input_ids, 
                            max_new_tokens = max_length, 
                            attention_mask = attention_mask,
                            do_sample=False,
                        )
            generated_tokens = outputs[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0])
            return text, None, None
    
    def generate_attn(self, input_text, max_length, solver="max", use_entropy = False, use_logprob = False, stop_on_question=False):
        # if self.instruct:
        #     input_text = self.apply_chat_template(input_text, assistant_keyword=assistant_keyword, force_continue_generation=force_continue_generation)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)
        stopping_criteria = [StopWordCriteria(tokenizer=self.tokenizer, prompts=[input_text], stop_words=self.stop_words)]

        if self.llama3:
            terminators = [
                            self.tokenizer.eos_token_id,
                            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        ]
            if stop_on_question:
                outputs = self.model.generate(
                    input_ids = input_ids, 
                    attention_mask = attention_mask,
                    max_new_tokens = max_length,
                    return_dict_in_generate = True, 
                    output_scores = True,
                    eos_token_id=terminators,
                    pad_token_id = self.tokenizer.eos_token_id,
                    do_sample=False,
                    stopping_criteria=stopping_criteria
                )
            else:
                outputs = self.model.generate(
                    input_ids = input_ids, 
                    attention_mask = attention_mask,
                    max_new_tokens = max_length,
                    return_dict_in_generate = True, 
                    output_scores = True,
                    eos_token_id=terminators,
                    pad_token_id = self.tokenizer.eos_token_id,
                    do_sample=False,
                )
        else:
            if stop_on_question:
                outputs = self.model.generate(
                    input_ids = input_ids, 
                    attention_mask = attention_mask,
                    max_new_tokens = max_length, 
                    return_dict_in_generate = True, 
                    output_scores = True,
                    do_sample=False,
                    stopping_criteria=stopping_criteria
                )
            else:
                outputs = self.model.generate(
                    input_ids = input_ids, 
                    attention_mask = attention_mask,
                    max_new_tokens = max_length, 
                    return_dict_in_generate = True, 
                    output_scores = True,
                    do_sample=False,
                )
        generated_tokens = outputs.sequences[:, input_length:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
        text = self.tokenizer.decode(generated_tokens[0])

        # merge tokens
        range_ = []
        if self.instruct:
            for i, t in enumerate(tokens):
                if i == 0 or t.startswith(self.space_token) or generated_tokens[0][i-1] == 13 or generated_tokens[0][i-1] == 382 \
                or generated_tokens[0][i-1] == 627 or tokens[i-1] == '</s>':
                    range_.append([i, i])
                else:
                    range_[-1][-1] += 1
        
        else:
            for i, t in enumerate(tokens):
                if i == 0 or t.startswith(self.space_token) or generated_tokens[0][i] == 13 or tokens[i-1] == '</s>':
                    range_.append([i, i])
                else:
                    range_[-1][-1] += 1

        # attention
        atten = self.model(generated_tokens, output_attentions=True).attentions[-1][0]
        if solver == "max": 
            mean_atten, _ = torch.max(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
        elif solver == "avg":
            mean_atten = torch.sum(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
            for i in range(mean_atten.shape[0]):
                mean_atten[i] /= (mean_atten.shape[0] - i)
        elif solver == "last_token":
            mean_atten = torch.mean(atten[:, -1], dim=0)
        else:
            raise NotImplementedError
        if mean_atten.shape[0] > 1 and tokens[0] == '</s>':
            mean_atten = mean_atten / sum(mean_atten[1:]).item()
        # mean_atten = mean_atten[tl:tr]
            
        # regular tokens
        seqlist = []
        attns = []
        for r in range_:
            tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(self.space_token, "")
            value = sum(mean_atten[r[0]: r[1]+1]).item()
            seqlist.append(tokenseq)
            attns.append(value)
        seqlist = [ s.replace("Ċ", "\n").replace("<0x0A>", "\n") for s in seqlist ]
        try:
            seqlist = [ s.replace("Ċ", "\n").encode('raw_unicode_escape').decode('utf8') for s in seqlist ]
        except:
            try:
                seqlist = [ s.replace("Ċ", "\n").encode('raw_unicode_escape').decode('latin-1') for s in seqlist ]
            except:
                pass

        # -log prob
        if use_logprob:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            seqlogprobs = []
            for r in range_:
                logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqlogprobs.append(logprobseq)
        else:
            seqlogprobs = None

        # entropy
        if use_entropy:
            tmp = []
            for v in outputs.scores:
                tmp.append(v.cpu())
            softmax_probs = softmax(tmp, axis=-1)
            entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
            entropies = [v[0] for v in entropies]
            seqentropies = []
            for r in range_:
                entropyseq = sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqentropies.append(entropyseq) 
        else:
            seqentropies = None 

        return text, seqlist, attns, seqlogprobs, seqentropies


class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }
         

class BasicRAG:
    def __init__(self, args):
        args = args.__dict__ 
        for k, v in args.items():
            setattr(self, k, v)
        self.generator = BasicGenerator(self.model_name_or_path)
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                # gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                self.retriever = BM25(
                    tokenizer = self.generator.tokenizer, 
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name, 
                    engine = "elasticsearch",
                )
            elif self.retriever_type == "SGPT":
                self.retriever = SGPT(
                    model_name_or_path = self.sgpt_model_name_or_path, 
                    sgpt_encode_file_path = self.sgpt_encode_file_path,
                    passage_file = self.passage_file
                )
            else:
                raise NotImplementedError
        
        self.counter = Counter()
        self.temperature = self.temperature if "temperature" in self.__dict__ and self.temperature != 0 else None
        self.top_p = self.top_p if "top_p" in self.__dict__ else None

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            _docs_ids, docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk, 
                max_query_length = max_query_length,
            )
            return docs[0]
        elif self.retriever_type == "SGPT":
            docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk,
            )
            return docs[0] 
        else:
            raise NotImplementedError
    
    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else "" 
    
    def inference(self, question, demo, case, final_ans_str="the answer is", debug=False):
        # non-retrieval
        assert self.query_formulation == "direct"
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length, temperature=self.temperature, top_p=self.top_p)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        if debug:
            return text, text
        else:
            return text
    

class SingleRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, demo, case, final_ans_str="the answer is", debug=False):
        assert self.query_formulation == "direct"
        docs = self.retrieve(question, topk=self.retrieve_topk)
        # 对 topk 个 passage 生成 prompt
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += "Context:\n"
        doc_prompt = ""
        for i, doc in enumerate(docs):
            doc_prompt += f"[{i+1}] {doc}\n"
        prompt += doc_prompt
        prompt += "Answer in the same format as before.\n"
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        if debug:
            return text, "[R\nQuery:%sContext:%s\n]\n%s" % (question, doc_prompt, text)
        else:
            return text


class FixLengthRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, demo, case, final_ans_str="the answer is", debug=False):
        assert self.query_formulation == "direct"
        query, text, debug_text = question, "", ""
        while True:
            old_len = len(text)
            # query = question
            docs = self.retrieve(query, topk=self.retrieve_topk)
            # 对 topk 个 passage 生成 prompt
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += "Context:\n"
            doc_prompt = ""
            for i, doc in enumerate(docs):
                doc_prompt += f"[{i+1}] {doc}\n"
            prompt += doc_prompt
            prompt += "Answer in the same format as before.\n"
            prompt += case + " " + text
            if self.method == "fix-length-retrieval":
                new_text, _, _ = self.generator.generate(prompt, self.fix_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                text = text.strip() + " " + new_text.strip()
                debug_text = debug_text.strip() + " " + "\n[R\nQuery:%sContext:%s\n]\n%s" % (query, doc_prompt, new_text.strip())
                query = new_text.strip()
            else:
                # fix sentence
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                new_text = new_text.strip()
                sentences = list(nlp(new_text).sents)
                sentences = [str(sent).strip() for sent in sentences]
                if len(sentences) == 0:
                    break
                new_text = sentences[0]
                text = text.strip() + " " + str(new_text)
                debug_text = debug_text.strip() + " " + "\n[R\nQuery:%sContext:%s\n]\n%s" % (query, doc_prompt, str(new_text))
                query = new_text
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or final_ans_str in text:
                break
        if debug:
            return text, debug_text
        else:
            return text


class TokenRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        tid = 0
        for sid, sent in enumerate(sentences):
            pos = 0
            tr = tid
            while tr < len(tokens):
                apr = sent[pos:].find(tokens[tr])
                if apr == -1:
                    break
                pos = apr + len(tokens[tr])
                tr += 1
            probs = [1 - exp(v) for v in logprobs[tid:tr+1]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                # # 这里改成了替换掉最大的那个，而不是所有的
                # max_prob = 0
                # for prob, tok in zip(probs, tokens[tid:tr+1]):
                #     max_prob = max(prob, max_prob)
                for prob, tok in zip(probs, tokens[tid:tr+1]):
                    apr = curr[pos:].find(tok) + pos
                    if prob > self.hallucination_threshold:
                    # if prob == max_prob:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True
            tid = tr + 1
        
        # No hallucination
        return text, None, False
    
    def inference(self, question, demo, case, final_ans_str="the answer is", debug=False):
        # assert self.query_formulation == "direct"
        text, debug_text = "", ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += case + " " + text
            new_text, tokens, logprobs = self.generator.generate(
                prompt, 
                self.generate_max_length, 
                return_logprobs=True
            )
            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            ptext, curr, hallucination = self.modifier(new_text, tokens, logprobs)
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
                debug_text = debug_text.strip() + " " + new_text.strip()
            else:
                if self.query_formulation == "direct":
                    retrieve_question = curr.replace("[xxx]", "")
                elif self.query_formulation == "forward_all":
                    tmp_all = [question, text, ptext]
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                doc_prompt = ""
                for i, doc in enumerate(docs):
                    doc_prompt += f"[{i+1}] {doc}\n"
                prompt += doc_prompt
                prompt += "Answer in the same format as before.\n"
                prompt += case + " " + text + " " + ptext.strip()
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
                debug_text = debug_text.strip() + " " + ptext.strip() + "\n[R\nCurr Tokens: %s\nQuery:%s\nRetrieval:%s]\n" % (curr, retrieve_question, doc_prompt) + new_text.strip()
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or final_ans_str in text:
                break
        if debug:
            return text, debug_text
        else:
            return text, ""
    

class EntityRAG(TokenRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        entity = []
        for sent in sentences:
            doc = nlp(sent)
            li = [ent.text for ent in doc.ents]
            entity.append(li)
        
        belonging = [-1] * len(text)
        pos = 0
        for tid, tok in enumerate(tokens):
            apr = text[pos:].find(tok) + pos
            assert apr != -1
            for j in range(pos, apr+len(tok)):
                belonging[j] = tid
            pos = apr + len(tok)
        
        entity_intv = []
        for sid, sent in enumerate(sentences):
            tmp = []
            pos = text.find(sent)
            for ent in entity[sid]:
                apr = text[pos:].find(ent) + pos
                el = belonging[apr]
                er = belonging[apr + len(ent) - 1]
                tmp.append((el, er))
                pos = apr + len(ent)
            entity_intv.append(tmp)

        entity_prob = []
        for ent_itv_per_sent in entity_intv:
            tmp = []
            for itv in ent_itv_per_sent:
                probs = np.array(logprobs[itv[0]:itv[1]+1])
                p = {
                    "avg": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "first": lambda x: x[0] if len(x) > 0 else 0
                }.get(self.entity_solver, lambda x: 0)(probs)
                tmp.append(p)
            entity_prob.append(tmp)

        for sid in range(len(sentences)):
            if len(entity_prob[sid]) == 0:
                continue
            probs = [1 - exp(v) for v in entity_prob[sid]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated entities in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                for prob, ent in zip(probs, entity[sid]):
                    apr = curr[pos:].find(ent) + pos
                    if prob > self.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(ent):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(ent)
                return prev, curr, True
        # No hallucination
        return text, None, False

    def inference(self, question, demo, case, final_ans_str="the answer is", debug=False):
        return super().inference(question, demo, case, final_ans_str=final_ans_str, debug=debug)


class AttnWeightRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def modifier(self, text, tokens, attentions, weight):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        tid = 0
        for sid, sent in enumerate(sentences):
            tl, tr = tid, tid
            if sid == len(sentences) - 1:
                tl, tr = tid, len(tokens)
            else:
                for i in range(tid + 1, len(tokens)):
                    seq = " ".join(tokens[tl:i])
                    if sent in seq:
                        tr = i
                        break
                if tl == tr:
                    logger.info(f"tl == tr for sentence: {sent} for text: {text} and tokens: {str(tokens)}")
                    # import ipdb; ipdb.set_trace()
                tid = tr
            # value = attenion * (-log prob)
            attns = attentions[tl:tr]
            attns = np.array(attns) / sum(attns)
            value = [attns[i-tl] * weight[i] * (tr-tl) for i in range(tl, tr)] 
            thres = [1 if v > self.hallucination_threshold else 0 for v in value]

            if "check_real_words" in self.__dict__ and self.check_real_words:       # can be pushed out to avoid random retrieval queries
                doc = nlp(sent)
                real_words = set(token.text for token in doc if token.pos_ in 
                    ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                def match(tok):
                    for word in real_words:
                        if word in tok:
                            return True
                    return False
                for i in range(len(thres)):
                    if not match(tokens[tl+i]):
                        thres[i] = 0
                        
            if 1 in thres:
                # hallucinated
                # if "check_real_words" in self.__dict__ and self.check_real_words:
                #     doc = nlp(sent)
                #     real_words = set(token.text for token in doc if token.pos_ in 
                #         ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                #     def match(tok):
                #         for word in real_words:
                #             if word in tok:
                #                 return True
                #         return False
                #     for i in range(len(thres)):
                #         if not match(tokens[tl+i]):
                #             thres[i] = 0                
                
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # curr = " ".join(
                #     [tokens[i] if thres[i] == 0 else "[xxx]" for i in range(len(thres))]
                # )
                return True, prev, tokens[tl:tr], thres
        return False, text, None, None

    def keep_real_words(self, prev_text, curr_tokens, curr_hit):
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt").to(self.generator.model.device)
        input_length = input_ids.shape[1]
        tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])

        atten_tmp = self.generator.model(input_ids, output_attentions=True).attentions[-1][0]

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens_tmp):
            if i == 0 or t.startswith(self.generator.space_token) or input_ids[0][i] == 13:
                range_.append([i, i])
            else:
                range_[-1][-1] += 1
        tokens = []
        for r in range_:
            tokenseq = "".join(tokens_tmp[r[0]: r[1]+1]).replace(self.generator.space_token, "")
            tokens.append(tokenseq)
        tokens = [ t.replace("Ċ", "\n").replace("<0x0A>", "\n") for t in tokens ]
        try:
            tokens = [ t.replace("Ċ", "\n").encode('raw_unicode_escape').decode('utf8') for t in tokens ]
        except:
            try:
                tokens = [ t.replace("Ċ", "\n").encode('raw_unicode_escape').decode('latin-1') for t in tokens ]
            except:
                pass

        # 获取幻觉词对应的 attention
        curr_st = len(tokens) - len(curr_tokens)
        atten_tmp = torch.mean(atten_tmp, dim=0)
        attns = []
        for r in range_:
            # att = torch.zeros(atten_tmp.shape[0], input_length)
            att = torch.zeros(input_length)
            for i in range(r[0], r[1] + 1):
                if i == 0:
                    continue
                v = atten_tmp[i-1][:r[0]] # 上一位的
                v = v / v.sum()
                t = torch.zeros(input_length)
                t[:r[0]] = v
                att += t
            att /= (r[1] - r[0] + 1)
            # merge token for att
            att = torch.tensor([att[rr[0]:rr[1]+1].sum() for rr in range_])
            attns.append(att)
            
        # 计算每个超过阈值的 token 在前文的 attentions
        forward_attns = torch.zeros(len(tokens))
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if curr_hit[i] == 1:
                forward_attns += attns[curr_st + i]
                hit_cnt += 1
        forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        # 分析词性，保留实词对应的 attns
        doc = nlp(all_text)
        real_words = set(token.text for token in doc if token.pos_ in 
                      ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
        
        def match(token):
            for word in real_words:
                if word in token:
                    return True
            return False
        
        real_pairs = []
        for i in range(len(tokens)):
            tok, att = tokens[i], forward_attns[i]
            if i >= curr_st and curr_hit[i - curr_st]:
                continue
            if match(tok):
                real_pairs.append((att, tok, i))
        
        if "retrieve_keep_top_k" in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)
        
        real_pairs = sorted(real_pairs, key = lambda x:x[0], reverse=True)
        real_pairs = real_pairs[:top_k]
        real_pairs = sorted(real_pairs, key = lambda x:x[2])
        return " ".join([x[1] for x in real_pairs])
        
    def inference(self, question, demo, case, final_ans_str="the answer is", debug=False):
        # assert self.query_formulation == "direct"
        # print(question)
        # print("#" * 20)
        text = ""
        debug_text = ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            tmp_li = [case, text]
            prompt += " ".join(s for s in tmp_li if len(s) > 0)
            # print('####', prompt)
            # prompt += case + " " + text
            new_text, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt, 
                self.generate_max_length, 
                # self.attention_solver, 
                use_entropy = self.method == "dragin", 
                use_logprob = self.method == "attn_prob"
            )
            # if self.generator.instruct:
            #     import ipdb; ipdb.set_trace()
            weight = entropies if self.method == "dragin" else [-v for v in logprobs]

            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            hallucination, ptext, curr_tokens, curr_hit =  self.modifier(new_text, tokens, attns, weight)
            
            if not hallucination or final_ans_str in ptext:
                text = text.strip() + " " + new_text.strip()
                debug_text = debug_text.strip() + " " + new_text.strip()
            else:
                forward_all = [question, text, ptext]
                forward_all = " ".join(s for s in forward_all if len(s) > 0)

                def fetch_last_n_tokens(text, num, tokenizer = self.generator.tokenizer):
                    tokens = tokenizer.tokenize(text)
                    if num >= len(tokens):
                        return text
                    last_n_tokens = tokens[-num:]
                    last_n_sentence = ' '.join(last_n_tokens)
                    return last_n_sentence

                if self.query_formulation == "current":
                    retrieve_question = " ".join(curr_tokens)

                elif self.query_formulation == "current_wo_wrong":
                    retrieve_question = " ".join(
                        list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens)))
                    )

                elif self.query_formulation == "forward_all":
                    retrieve_question = forward_all
                
                elif self.query_formulation == "last_sentence":
                    retrieve_question = self.get_last_sentence(forward_all)
                
                elif self.query_formulation == "last_n_tokens":
                    assert "retrieve_keep_top_k" in self.__dict__
                    retrieve_question = fetch_last_n_tokens(
                        forward_all, self.retrieve_keep_top_k)
                
                elif self.query_formulation == "real_words": 
                    retrieve_question = self.keep_real_words(
                        prev_text = question + " " + text + " " + ptext, 
                        curr_tokens = curr_tokens, 
                        curr_hit = curr_hit,
                    ) 
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                doc_prompt = ""
                for i, doc in enumerate(docs):
                    doc_prompt += f"[{i+1}] {doc}\n"
                prompt += doc_prompt
                prompt += "Answer in the same format as before.\n"
                tmp_li = [case, text, ptext.strip()]
                prompt += " ".join(s for s in tmp_li if len(s) > 0)
                # print('#####', prompt)
                # prompt += case + " " + text + " " + ptext.strip()
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                # if self.generator.instruct:
                #     import ipdb; ipdb.set_trace()
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                new_text = self.get_top_sentence(new_text)
                tmp_li = [text.strip(), ptext.strip(), new_text.strip()]
                text = " ".join(s for s in tmp_li if len(s) > 0)
                hall_token = ", ".join([ t for t,v in zip(curr_tokens, curr_hit) if v > 0 ])
                debug_text = debug_text.strip() + " " + ptext.strip() + "\n[R\nCurr Tokens: %s\nHallucinated Token: %s\nQuery:%s\nRetrieval:%s]\n" % (curr_tokens, hall_token, retrieve_question, doc_prompt) + new_text.strip()
                # text = text.strip() + " " + ptext.strip() + " " + new_text.strip()

                # print("### retrieve_question ###")
                # print(retrieve_question)
                # context = "### Context: ###\n"
                # for i, doc in enumerate(docs):
                #     context += f"[{i+1}] {doc}\n" 
                # print(context)
                # print(text)
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or final_ans_str in text:
                break
        # print("#" * 20)
        if debug:
            return text, debug_text
        else:
            return text, ""


class SelfAsk(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
        if self.examples == "original":
            if self.fewshot == 4:
                self.few_shot_examples = SELFASK_ORIG_FEWSHOT4
            elif self.fewshot == 6:
                self.few_shot_examples = SELFASK_ORIG_FEWSHOT6
            elif self.fewshot == 8:
                self.few_shot_examples = SELFASK_ORIG_FEWSHOT8
        else:
            self.few_shot_examples = SELFASK_DRAGIN_FEWSHOT4
        self.case = SELFASK_CASE
        self.stop_on_question = self.stop_on_question if "stop_on_question" in args else False
        
    def inference(self, question, demo, case, final_ans_str="the answer is", debug=False):
        text = ""
        debug_text = ""
        case = self.case + question + "\n"
        while True:
            old_len = len(text)
            prompt = self.few_shot_examples
            tmp_li = [case, text]
            prompt += " ".join(s for s in tmp_li if len(s) > 0)
            if text == "":
                prompt += SELFASK_ORIG_FOLLOWUP
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length, stop_on_question=self.stop_on_question)
                new_text = SELFASK_ORIG_FOLLOWUP + new_text
            else:
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length, stop_on_question=self.stop_on_question)

            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)

            followup_line = [ sent for sent in new_text.split("\n") if sent.strip() != "" ]
            followup_line = followup_line[0] if len(followup_line) > 0 else ""
            if followup_line == "" or final_ans_str in followup_line or not self.do_retrieval:
                text = text.strip() + new_text
                debug_text = debug_text.strip() + new_text
            elif self.do_retrieval:
                retrieve_question = followup_line.split("Follow up:")[1] if "Follow up:" in followup_line else followup_line
                retrieve_question = retrieve_question.strip()

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = self.few_shot_examples
                prompt += "Context:\n"
                doc_prompt = ""
                for i, doc in enumerate(docs):
                    doc_prompt += f"[{i+1}] {doc}\n"
                prompt += doc_prompt
                tmp_li = [case, text]
                prompt += "\n".join(s for s in tmp_li if len(s) > 0)
                prompt += followup_line
                new_text, _, _ = self.generator.generate(prompt + "\n" + SELFASK_ORIG_INT_ANSWER, self.generate_max_length, stop_on_question=self.stop_on_question)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                new_text = self.get_top_sentence(new_text.replace("\n", " ")).replace("</s>", "")
                new_text = SELFASK_ORIG_INT_ANSWER + new_text
                tmp_li = [text.strip(), followup_line.strip(), new_text.strip()]
                text = "\n".join(s for s in tmp_li if len(s) > 0)
                debug_text = debug_text.strip() + "\n" + followup_line.strip() + "\n[R\nQuery:%s\nRetrieval:%s]\n" % (retrieve_question, doc_prompt) + new_text.strip()
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or final_ans_str in text:
                break
        # print("#" * 20)
        if debug:
            return text, debug_text
        else:
            return text, ""

