# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from peft import PeftModel

class LLM:
    def __init__(self, model_name_or_path, lora_name_or_path=None, chat_mode=False, llama3=False):
        print ("Loading model for AutoEval from %s" % model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_config = AutoConfig.from_pretrained(model_name_or_path,
                    trust_remote_code = "falcon" in model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", 
                    trust_remote_code = "falcon" in model_name_or_path)
        if lora_name_or_path:
            self.model = PeftModel.from_pretrained(self.model, lora_name_or_path)
        self.llama3 = True if "Llama-3" in model_name_or_path else False
        self.llama3 = True if llama3 else self.llama3
        self.chat_mode = True if self.llama3 and chat_mode else False
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, input_text, max_length, system_prompt="You are a helpful assistant.", return_probs=False, prob_tokens=[]):
        if self.chat_mode:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
        else:
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        if return_probs:
            if self.llama3:
                terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                outputs = self.model.generate(
                    input_ids = input_ids, 
                    attention_mask = attention_mask,
                    max_new_tokens = max_length, 
                    eos_token_id=terminators,
                    pad_token_id = self.tokenizer.eos_token_id,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            else:
                outputs = self.model.generate(
                    input_ids = input_ids, 
                    max_new_tokens = max_length, 
                    attention_mask = attention_mask,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            if len(prob_tokens) > 0:
                sequence = outputs.sequences[:, :input_length]
                probs = []
                for prob_t in prob_tokens:
                    tokenized_t = self.tokenizer.encode(prob_t, return_tensors="pt").to(self.model.device)[:, 1:]
                    tokenized_t = torch.cat((tokenized_t, outputs.sequences[:, -1:]), 1)
                    sequence_t = torch.cat((sequence, tokenized_t), 1)
                    transition_scores = self.model.compute_transition_scores(
                        sequence_t, outputs.scores, normalize_logits=True
                    )
                    prob_t = torch.exp(transition_scores[0][0]).cpu().numpy()
                    probs.append(prob_t)
                probs = [ p/sum(probs) for p in probs ]
            else:
                transition_scores = self.model.compute_transition_scores(
                        outputs.sequences, outputs.scores, normalize_logits=True
                    )
            outputs = outputs.sequences

        else:
            if self.llama3:
                terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                outputs = self.model.generate(
                    input_ids = input_ids, 
                    attention_mask = attention_mask,
                    max_new_tokens = max_length, 
                    eos_token_id=terminators,
                    pad_token_id = self.tokenizer.eos_token_id,
                    do_sample=False,
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
        if return_probs:
            if len(prob_tokens) > 0:
                return text, prob_tokens, probs
            else:
                tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
                logprobs = transition_scores[0]
                probs = [torch.exp(p).cpu().numpy() for p in logprobs]
                assert len(tokens) == len(probs)
                return text, tokens, probs
        return text
    
class AnswerCoherenceEvaluator:
    def __init__(self, model_name_or_path="", generator=None, chat_mode=False):
        self.model = LLM(model_name_or_path, chat_mode=chat_mode) if generator is None else generator
        self.final_answer_prompt = "Final answer:"
        self.prefix = "Can the final answer be inferred from the reasoning provided for the question? Do not check for correctness, only check for entailment provided the reasoning. Only answer as 'yes' or 'no'."
        self.fewshot = [
            {
                "question": "Are Scott Derrickson and Ed Wood from the same nationality?",
                "reasoning": "Scott Derrickson is an American film director, producer, and screenwriter. Ed Wood was an American film director, producer, and screenwriter. Therefore, both Scott Derrickson and Ed Wood are American.",
                "final_answer": "yes",
                "entailment": "yes"
            },
            {
                "question": "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?",
                "reasoning": "The science fantasy young adult series is \"Wires and Nerve\". The series is told in first person. The series has a set of companion books narrating the stories of enslaved worlds and alien species. The companion books are \"The Hork-Bajir Chronicles\" and \"The Ellimist Chronicles\".",
                "final_answer": "Animorphs",
                "entailment": "no"
            },
            {
                "question": "Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990?",
                "reasoning": "The Force India driver born in 1990 is Sergio P\u00e9rez. Sergio P\u00e9rez has held the podium. Another Mexican Formula One race car driver who has held the podium is Pedro Rodr\u00edguez. Therefore, the answer is Pedro Rodr\u00edguez.",
                "final_answer": "Pedro Rodr\u00edguez",
                "entailment": "yes"
            },
            {
                "question": "What American professional Hawaiian surfer born 18 October 1992 won the Rip Curl Pro Portugal?",
                "reasoning": "The defending champion of the MEO Rip Curl Pro Portugal is the Brazilian surfer Gabriel Medina who won the event in 2017. In 2018, Italo Ferreira, who represents Brazil, is the current champion. Therefore, neither Gabriel Medina nor Italo Ferreira is an American professional Hawaiian surfer born 18 October 1992. The answer is John John Florence.",
                "final_answer": "John John Florence",
                "entailment": "no"
            }
        ]
        self.question_prompt = "Question: %s\nReasoning: %s\nFinal Answer: %s"
        self.fewshot_prompt = "Question: %s\nReasoning: %s\nFinal Answer: %s\nEntailment: %s\n"
        self.fewshot_examples = "\n".join([ self.fewshot_prompt % (fs["question"], fs["reasoning"], fs["final_answer"], fs["entailment"]) for fs in self.fewshot ])
    
    def split_prediction(self, prediction):
        if self.final_answer_prompt in prediction:
            return prediction.split(self.final_answer_prompt)[0]
        return prediction
    
    def parse_output(self, output):
        output = output.strip().lower()
        if "yes" in output:
            return 1
        elif "no" in output:
            return 0
        else:
            return -1
    
    def _get_prompt(self, datapoint, include_fewshot=False):
        question = datapoint["question"]
        final_answer = datapoint["final_pred"]
        reasoning = self.split_prediction(datapoint["prediction"])
        reasoning = reasoning.replace("\n", " ")
        prompt = self.question_prompt % (question, reasoning, final_answer) + "\n\n" + self.prefix + "\n\n"
        if include_fewshot:
            prompt = self.fewshot_examples + prompt
        return prompt
    
    def evaluate(self, datapoint):
        prompt = self._get_prompt(datapoint)
        output = self.model.generate(prompt, 5)
        entails = self.parse_output(output)
        return entails
    
class Verifier:
    def __init__(self, args):
        args = args.__dict__ 
        for k, v in args.items():
            setattr(self, k, v)
        self.model = LLM(self.model_name_or_path, chat_mode=True)
        self.stop_on_question = self.stop_on_question if "stop_on_question" in args else False

        self.final_answer_prompt = "Final answer:"
        self.prefix = "You are a verifier looking to verify the correctness of the answer for the given question based on the provided reasoning. Provide a 'yes' or 'no' answer for the correctness and then provide a corresponding explanation for justification."
        self.fewshot = [
            {
                "question": "Are Scott Derrickson and Ed Wood from the same nationality?",
                "reasoning": "Scott Derrickson is an American film director, producer, and screenwriter. Ed Wood was an American film director, producer, and screenwriter. Therefore, both Scott Derrickson and Ed Wood are American.",
                "final_answer": "yes",
                "correctness": "yes"
            },
            {
                "question": "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?",
                "reasoning": "The science fantasy young adult series is \"Wires and Nerve\". The series is told in first person. The series has a set of companion books narrating the stories of enslaved worlds and alien species. The companion books are \"The Hork-Bajir Chronicles\" and \"The Ellimist Chronicles\".",
                "final_answer": "Animorphs",
                "correctness": "no"
            },
            {
                "question": "Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990?",
                "reasoning": "The Force India driver born in 1990 is Sergio P\u00e9rez. Sergio P\u00e9rez has held the podium. Another Mexican Formula One race car driver who has held the podium is Pedro Rodr\u00edguez. Therefore, the answer is Pedro Rodr\u00edguez.",
                "final_answer": "Pedro Rodr\u00edguez",
                "correctness": "yes"
            },
            {
                "question": "What American professional Hawaiian surfer born 18 October 1992 won the Rip Curl Pro Portugal?",
                "reasoning": "The defending champion of the MEO Rip Curl Pro Portugal is the Brazilian surfer Gabriel Medina who won the event in 2017. In 2018, Italo Ferreira, who represents Brazil, is the current champion. Therefore, neither Gabriel Medina nor Italo Ferreira is an American professional Hawaiian surfer born 18 October 1992. The answer is John John Florence.",
                "final_answer": "John John Florence",
                "correctness": "no"
            }
        ]
        self.question_prompt = "Question: %s\nReasoning: %s\nFinal Answer: %s"
        self.fewshot_prompt = "Question: %s\nReasoning: %s\nFinal Answer: %s\Correctness: %s\n"
        self.fewshot_examples = "\n".join([ self.fewshot_prompt % (fs["question"], fs["reasoning"], fs["final_answer"], fs["correctness"]) for fs in self.fewshot ])

    def split_prediction(self, prediction):
        if self.final_answer_prompt in prediction:
            return prediction.split(self.final_answer_prompt)[0]
        return prediction
    
    def parse_output(self, output):
        output = output.split("\n")[0].strip().lower()
        if "yes" in output:
            return 1
        elif "no" in output:
            return 0
        else:
            return -1
    
    def _get_prompt(self, datapoint, include_fewshot=False):
        question = datapoint["question"]
        final_answer = datapoint["final_pred"]
        reasoning = self.split_prediction(datapoint["prediction"])
        reasoning = reasoning.replace("\n", " ")
        prompt = self.prefix + "\n\n"
        if include_fewshot:
            prompt += self.fewshot_examples + "\n\n"
        prompt += self.question_prompt % (question, reasoning, final_answer) + "\n"
        return prompt

    def verify(self, datapoint):
        prompt = self._get_prompt(datapoint)
        output = self.model.generate(prompt, 100)
        entails = self.parse_output(output)
        return entails, output

class StrategyDecider:
    def __init__(self, args):
        self.model = LLM(args.model_name_or_path, chat_mode=True)
        self.max_length = args.max_length
        self.available_fields = args.fields
        self.num_few_shot = args.few_shot

        self.system_prompt = "You are a planner, looking to choose the strategy that is best suited for answering the given question. You want to maximize the answer correctness as much as possible. Your secondary objective will be to reduce the cost. Prefer to use a higher cost strategy unless you believe that the lower cost strategy can successfully answer the question."
        self.strategy_descriptions = {
            "direct": "Direct - use this strategy when you know the exact answer to the question confidently.",
            "cot": "CoT - use this strategy when you think you can infer the answer yourself by reasoning step-by-step.",
            "rag-direct": "RAG_Direct - use this strategy when you think you want external resources or text to better answer the question, but you can directly answer the question based on the external text.",
            "rag": "RAG - use this strategy when you think you want external resources or text to better answer the question along with additional thinking to reason through the external text and your self-knowledge.",
            "selfask-orig": "Plan - use this strategy when you think the question is too complex and you want to break the question into simpler atomic questions to get to the final answer."
        }
        # self.cost_descriptions = [
        #     "This is the cheapest strategy.",
        #     "This is the second cheapest strategy.",
        #     "This strategy is neither too costly nor too cheap.",
        #     "This is the second expensive strategy.",
        #     "This is the most expensive strategy."
        # ]
        self.cost_descriptions = [
            "Cost: 1.",
            "Cost: 2.",
            "Cost: 3.",
            "Cost: 4.",
            "Cost: 5."
        ]
        self.few_shot_examples_library = {
            "Llama-3-8B": {
                "direct": [
                    "The Bass Rock Lighthouse was next to what Castle?",
                    "Rob Jenkins is an Australian actor, which of his movies, was a 2014 science fiction thriller film?",
                    "Mark Rankin and Cedric Bixler-Zavala, band is what type genre?",
                    "Tata Movus is what type of high-roof vehicle?"
                ],
                "cot": [
                    "The Golden Globe Award winner for best actor from \"Roseanne\" starred along what actress in Gigantic?",
                    "Who achieved fame starring alongside John Richard Schneider in a television series?",
                    "Where was the Hockey League which drafted Niclas Bergfors based?",
                    "Rubi Lira Miranda Palmieri was born in a city whose name honors what Apostle?"
                ],
                "selfask-orig": [
                    "Which of these events came first, the release of the movie Toy Story or the death of marathon runner Stylianos Kyriakides?",
                    "Michael Dobson voiced which character, an adversary of the X-Men, originally depicted as obese?",
                    "The American Aircraft Penetrator was modified from a helicopter produced by what nation's military?",
                    "Chang Ucchin was born in korea during a time that ended with the conclusion of what?"
                ],
                "rag": [
                    "Which tennis player won more Grand Slam titles, Henri Leconte or Jonathan Stark?",
                    "Where did the form of music played by Die Rhöner Säuwäntzt originate?",
                    "The 1988 American comedy film, The Great Outdoors, starred a four-time Academy Award nominee, who received a star on the Hollywood Walk of Fame in what year?",
                    "Who is the director of the 2003 film which has scenes in it filmed at the Quality Cafe in Los Angeles?"
                ]
            },
            "Llama-3-70B": {
                "direct": [
                    "Mark Rankin and Cedric Bixler-Zavala, band is what type genre?",
                    "Which highway that has a junction at Cairo Junction is 1286 miles long?",
                    "What is the busiest Amtrak route that does not stop at either Lortorn Station?",
                    "David Bell was a third baseman for which team?"
                ],
                "cot": [
                    "When a Man Falls in Love, is a 2013 South Korean television series, starring Song Seung-heon, Shin Se-kyung, Chae Jung-an, and which South Korean actor, born on ?",
                    "Which star in The Pokrovsky Gate was also a director and occasional singer?",
                    "For which film did, the composer who scored the music for Interview with the Vampire, win the Academy Award?",
                    "What is the capital of the archipelago where Djoièzi is located?"
                ],
                "rag": [
                    "Which tennis player won more Grand Slam titles, Henri Leconte or Jonathan Stark?",
                    "Where did the form of music played by Die Rhöner Säuwäntzt originate?",
                    "The expert mentor to the celebrities that perform on \"Splash!\" won the 2009 FINA World Championionship in the individual event at what age?",
                    "The Thoen Stone is on display at a museum in what county?"
                ]
            }
        }
        self.strategy2field = {
            "Direct": "direct",
            "CoT": "cot",
            "RAG_Direct": "rag-direct",
            "RAG": "rag",
            "Plan": "selfask-orig"
        }
        self.field2strategy = {v:k for k,v in self.strategy2field.items()}

        self.prefix = self._generate_prefix()
        self.instruction = "Based on your confidence, choose one of the above strategies for the given question. Provide the strategy name and nothing else."
        self.question_prompt = "Question: %s\nStrategy: "
        if self.num_few_shot > 0:
            self.fewshot_prompt = "Question: %s\nStrategy: %s\n"
            few_shot_examples_model = [ k for k in self.few_shot_examples_library if k in args.model_name_or_path ]
            if len(few_shot_examples_model) == 0:
                self.fewshot_examples = ""
                return
            few_shot_examples_model = few_shot_examples_model[0]
            self.fewshot_examples = "\n".join([ self.fewshot_prompt % (self.few_shot_examples_library[few_shot_examples_model][field][i], self._map_field_to_strategy(field)) for i in range(self.num_few_shot) for field in self.available_fields ])
    
    def _map_strategy_to_field(self, strategy):
        if strategy in self.strategy2field:
            return self.strategy2field[strategy]
        else:
            return strategy
    
    def _map_field_to_strategy(self, field):
        return self.field2strategy[field]
    
    def _generate_prefix(self):
        if len(self.available_fields) == 2:
            cost_numbers = [0,-1]
        elif len(self.available_fields) == 3:
            cost_numbers = [0,1,-1]
        elif len(self.available_fields) == 4:
            cost_numbers = [0,1,-2,-1]
        else:
            cost_numbers = [0,1,2,3,4]
        
        prefix = "Available Strategies to answer the question:"
        strategy_descs = [ self.strategy_descriptions[field] for field in self.available_fields ]
        for cost_number in cost_numbers:
            strategy_descs[cost_number] += " %s" % self.cost_descriptions[cost_number]
        for i, strategy_desc in enumerate(strategy_descs):
            prefix += "\n(%d) %s" % (i+1, strategy_desc)
        return prefix
    
    def parse_output(self, output):
        output = output.split("\n")[0].strip().lower()
        output = output.replace("<|eot_id|>", "")
        field = self._map_strategy_to_field(output)
        return field
    
    def _get_prompt(self, datapoint):
        question = datapoint["question"]
        prompt = self.prefix + "\n\n"
        if self.num_few_shot > 0:
            prompt += "Some examples:\n\n" + self.fewshot_examples + "\n"
        prompt += self.instruction + "\n\n"
        prompt += self.question_prompt % (question)
        return prompt

    def strategize(self, datapoint):
        prompt = self._get_prompt(datapoint)
        output = self.model.generate(prompt, self.max_length, system_prompt=self.system_prompt)
        field = self.parse_output(output)
        return field, output
    
class StrategyBinaryDecider:
    def __init__(self, args, no_load_model=False, llama3=False):
        if not no_load_model:
            llama3 = False if "llama3" not in args else args.llama3
            self.model = LLM(args.model_name_or_path, chat_mode=True, llama3=llama3)
            self.max_length = args.max_length
        self.field = args.fields[0]
        self.num_few_shot = args.few_shot if "few_shot" in args else 0

        self.system_prompt = "You are a planner, looking to decide if the suggested strategy can be used to answer the given question. Be conservative, answer yes if you confidently feel that the strategy will work."
        self.strategy_descriptions = {
            "direct": "Direct - use this strategy when you know the exact answer to the question confidently.",
            "cot": "CoT - use this strategy when you think you can infer the answer yourself by reasoning step-by-step.",
            "rag-direct": "RAG_Direct - use this strategy when you think you want external resources or text to better answer the question, but you can directly answer the question based on the external text.",
            "rag": "RAG - use this strategy when you think you want external resources or text to better answer the question along with additional thinking to reason through the external text and your self-knowledge.",
            "selfask-orig": "Plan - use this strategy when you think the question is too complex and you want to break the question into simpler atomic questions to get to the final answer."
        }
        self.few_shot_examples_library = {
            "Llama-3-8B": {
                "direct": [
                    "The Bass Rock Lighthouse was next to what Castle?",
                    "Rob Jenkins is an Australian actor, which of his movies, was a 2014 science fiction thriller film?",
                    "Mark Rankin and Cedric Bixler-Zavala, band is what type genre?",
                    "Tata Movus is what type of high-roof vehicle?"
                ],
                "cot": [
                    "The Golden Globe Award winner for best actor from \"Roseanne\" starred along what actress in Gigantic?",
                    "Who achieved fame starring alongside John Richard Schneider in a television series?",
                    "Where was the Hockey League which drafted Niclas Bergfors based?",
                    "Rubi Lira Miranda Palmieri was born in a city whose name honors what Apostle?"
                ],
                "rag": [
                    "Which tennis player won more Grand Slam titles, Henri Leconte or Jonathan Stark?",
                    "Where did the form of music played by Die Rhöner Säuwäntzt originate?",
                    "The 1988 American comedy film, The Great Outdoors, starred a four-time Academy Award nominee, who received a star on the Hollywood Walk of Fame in what year?",
                    "Who is the director of the 2003 film which has scenes in it filmed at the Quality Cafe in Los Angeles?"
                ]
            },
            "Llama-3-70B": {
                "direct": [
                    "Mark Rankin and Cedric Bixler-Zavala, band is what type genre?",
                    "Which highway that has a junction at Cairo Junction is 1286 miles long?",
                    "What is the busiest Amtrak route that does not stop at either Lortorn Station?",
                    "David Bell was a third baseman for which team?"
                ],
                "cot": [
                    "When a Man Falls in Love, is a 2013 South Korean television series, starring Song Seung-heon, Shin Se-kyung, Chae Jung-an, and which South Korean actor, born on ?",
                    "Which star in The Pokrovsky Gate was also a director and occasional singer?",
                    "For which film did, the composer who scored the music for Interview with the Vampire, win the Academy Award?",
                    "What is the capital of the archipelago where Djoièzi is located?"
                ],
                "rag": [
                    "Which tennis player won more Grand Slam titles, Henri Leconte or Jonathan Stark?",
                    "Where did the form of music played by Die Rhöner Säuwäntzt originate?",
                    "The expert mentor to the celebrities that perform on \"Splash!\" won the 2009 FINA World Championionship in the individual event at what age?",
                    "The Thoen Stone is on display at a museum in what county?"
                ]
            }
        }
        self.strategy2field = {
            "Direct": "direct",
            "CoT": "cot",
            "RAG_Direct": "rag-direct",
            "RAG": "rag",
            "Plan": "selfask-orig"
        }
        self.field2strategy = {v:k for k,v in self.strategy2field.items()}

        self.prefix = self._generate_prefix()
        self.instruction = "Based on your confidence, answer as 'yes' or 'no' if you can use the provided strategy to answer the given question. Be super conservative, analyze well, and output 'yes' only if you are super confident of answering the question with the provided strategy. Else output 'no'. Only output 'yes/no' and nothing else."
        self.question_prompt = "Question: %s\nCan be answered?: "
        if self.num_few_shot > 0:
            self.fewshot_prompt = "Question: %s\nCan be answered?: %s\n"
            few_shot_examples_model = [ k for k in self.few_shot_examples_library if k in args.model_name_or_path ]
            if len(few_shot_examples_model) == 0:
                self.fewshot_examples = ""
                return
            few_shot_examples_model = few_shot_examples_model[0]
            pos_few_examples = [ self.fewshot_prompt % (self.few_shot_examples_library[few_shot_examples_model][self.field][i], "yes") for i in range(self.num_few_shot) ]
            neg_few_examples = [ self.fewshot_prompt % (example, "no") for field in self.few_shot_examples_library[few_shot_examples_model] if field != self.field for example in self.few_shot_examples_library[few_shot_examples_model][field] ]
            self.fewshot_examples = "\n".join([ "%s\n%s" % (p_ex, n_ex) for p_ex, n_ex in zip(pos_few_examples, neg_few_examples) ])
    
    def _generate_prefix(self):
        strategy_desc = self.strategy_descriptions[self.field]
        prefix = "Strategy Description:\n%s" % strategy_desc
        return prefix
    
    def parse_output(self, output):
        output = output.split("\n")[0].strip().lower()
        output = output.replace("<|eot_id|>", "")
        if "yes" in output:
            return 1
        elif "no" in output:
            return 0
        else:
            return -1
    
    def _get_prompt(self, datapoint):
        question = datapoint["question"]
        prompt = self.prefix + "\n\n"
        if self.num_few_shot > 0:
            prompt += "Some examples:\n\n" + self.fewshot_examples + "\n"
        prompt += self.instruction + "\n\n"
        prompt += self.question_prompt % (question)
        return prompt

    def _get_decomposed_prompt(self, datapoint):
        instruction = self.prefix + "\n\n"
        if self.num_few_shot > 0:
            instruction += "Some examples:\n\n" + self.fewshot_examples + "\n"
        instruction += self.instruction + "\n\n"
        question = datapoint["question"]
        input = self.question_prompt % (question)
        return self.system_prompt, instruction, input

    def strategize(self, datapoint, return_probs=False, prob_tokens=[]):
        prompt = self._get_prompt(datapoint)
        if return_probs:
            output, tokens, probs = self.model.generate(prompt, self.max_length, system_prompt=self.system_prompt, return_probs=True, prob_tokens=prob_tokens)
            binary_decision = self.parse_output(output)
            return binary_decision, (output, tokens, probs)
        else:
            output = self.model.generate(prompt, self.max_length, system_prompt=self.system_prompt)
        binary_decision = self.parse_output(output)
        return binary_decision, output

class StrategyBinaryPostDecider:
    def __init__(self, args, no_load_model=False, llama3=False):
        if not no_load_model:
            llama3 = False if "llama3" not in args else args.llama3
            if "lora_name_or_path" in args:
                self.model = LLM(args.model_name_or_path, lora_name_or_path=args.lora_name_or_path, chat_mode=True, llama3=llama3)
            else:
                self.model = LLM(args.model_name_or_path, chat_mode=True, llama3=llama3)
            self.max_length = args.max_length
        self.field = args.fields[0]
        self.num_few_shot = args.few_shot if "few_shot" in args else 0

        self.system_prompt = "You are a verifier, looking to verify if the answer/reason provided by the given strategy for the given question is correct or not. Answer yes only if you are confident that the answer/reason is correct."
        self.strategy_descriptions = {
            "direct": "Direct - use this strategy when you know the exact answer to the question confidently.",
            "cot": "CoT - use this strategy when you think you can infer the answer yourself by reasoning step-by-step.",
            "rag-direct": "RAG_Direct - use this strategy when you think you want external resources or text to better answer the question, but you can directly answer the question based on the external text.",
            "rag": "RAG - use this strategy when you think you want external resources or text to better answer the question along with additional thinking to reason through the external text and your self-knowledge.",
            "selfask-orig": "Plan - use this strategy when you think the question is too complex and you want to break the question into simpler atomic questions to get to the final answer."
        }
        self.strategy2field = {
            "Direct": "direct",
            "CoT": "cot",
            "RAG_Direct": "rag-direct",
            "RAG": "rag",
            "Plan": "selfask-orig"
        }
        self.field2strategy = {v:k for k,v in self.strategy2field.items()}

        self.prefix = self._generate_prefix()
        self.instruction = "Based on your confidence, answer as 'yes' or 'no' if the provided answer/reason correctly and faithfully answers the given question. Take a deep breath, analyze well, and output 'yes' only if you are confident that the answer/reasoning is correct. Else output 'no'. Only output 'yes/no' and nothing else."
        self.question_reasoning_prompt = "Question: %s\nReasoning: %s\nIs correct?: "
        self.question_answer_prompt = "Question: %s\nAnswer: %s\nIs correct?: "
        self.question_reasoning_answer_prompt = "Question: %s\nReasoning: %s\nAnswer: %s\nIs correct?: "
    
    def _generate_prefix(self):
        strategy_desc = self.strategy_descriptions[self.field]
        prefix = "Strategy Description:\n%s" % strategy_desc
        return prefix
    
    def parse_output(self, output):
        output = output.split("\n")[0].strip().lower()
        output = output.replace("<|eot_id|>", "")
        if "yes" in output:
            return 1
        elif "no" in output:
            return 0
        else:
            return -1
    
    def _get_prompt(self, datapoint):
        # prompt = self.prefix + "\n\n"
        prompt = self.instruction + "\n\n"
        if "reasoning" in datapoint and "prediction" in datapoint:
            prompt += self.question_reasoning_answer_prompt % (datapoint["question"], datapoint["reasoning"], datapoint["prediction"])
        elif "reasoning" in datapoint:
            prompt += self.question_reasoning_prompt % (datapoint["question"], datapoint["reasoning"])
        elif "prediction" in datapoint:
            prompt += self.question_answer_prompt % (datapoint["question"], datapoint["prediction"])
        return prompt

    def _get_decomposed_prompt(self, datapoint):
        # instruction = self.prefix + "\n\n"
        instruction = self.instruction + "\n\n"
        if "reasoning" in datapoint and "prediction" in datapoint:
            input = self.question_reasoning_answer_prompt % (datapoint["question"], datapoint["reasoning"], datapoint["prediction"])
        elif "reasoning" in datapoint:
            input = self.question_reasoning_prompt % (datapoint["question"], datapoint["reasoning"])
        elif "prediction" in datapoint:
            input = self.question_answer_prompt % (datapoint["question"], datapoint["prediction"])
        return self.system_prompt, instruction, input

    def strategize(self, datapoint, return_probs=False, prob_tokens=[]):
        prompt = self._get_prompt(datapoint)
        if return_probs:
            output, tokens, probs = self.model.generate(prompt, self.max_length, system_prompt=self.system_prompt, return_probs=True, prob_tokens=prob_tokens)
            binary_decision = self.parse_output(output)
            return binary_decision, (output, tokens, probs)
        else:
            output = self.model.generate(prompt, self.max_length, system_prompt=self.system_prompt)
        binary_decision = self.parse_output(output)
        return binary_decision, output

class E2E_MODEL:
    def __init__(self, args, no_load_model=False, llama3=False):
        if not no_load_model:
            llama3 = False if "llama3" not in args else args.llama3
            if "lora_name_or_path" in args:
                self.model = LLM(args.model_name_or_path, lora_name_or_path=args.lora_name_or_path, chat_mode=True, llama3=llama3)
            else:
                self.model = LLM(args.model_name_or_path, chat_mode=True, llama3=llama3)
            self.max_length = args.max_length
        self.fields = args.fields
        self.num_few_shot = args.few_shot if "few_shot" in args else 0
        self.single_turn = args.single_turn if "single_turn" in args else 0

        self.strategy_descriptions = {
            "direct": "Direct - use this strategy when you know the exact answer to the question confidently.",
            "cot": "CoT - use this strategy when you think you can infer the answer yourself by reasoning step-by-step.",
            "rag-direct": "RAG_Direct - use this strategy when you think you want external resources or text to better answer the question, but you can directly answer the question based on the external text.",
            "rag": "RAG - use this strategy when you think you want external resources or text to better answer the question along with additional thinking to reason through the external text and your self-knowledge.",
            "selfask": "Plan - use this strategy when you think the question is too complex and you want to break the question into simpler atomic questions to get to the final answer."
        }
        self.strategy2field = {
            "Direct": "direct",
            "CoT": "cot",
            "RAG_Direct": "rag-direct",
            "RAG": "rag",
            "Plan": "selfask"
        }
        self.field2strategy = {v:k for k,v in self.strategy2field.items()}

        self.prefix = self._generate_prefix()
        self.system_prompt = "You are required to provide the answer to the given question. Do this by first planning your strategy and then using that strategy to answer the question."
        self.instructions = {
            "single_turn_output": "Based on your confidence, provide the strategy to use. Take a deep breath, analyze well, and output the name of the strategy to use. Then use the strategy style to provide the answer to the question.",
            "multi_turn_preplan": "Based on your confidence, provide the strategy to use. Take a deep breath, analyze well, and output the name of the strategy to use. Only output the strategy name and nothing else.",
            "multi_turn_strategy_use": "Now use the chosen strategy style to answer the question.",
            "multi_turn_verify_binary": "Using your self-knowledge, verify if the answer/reasoning provided for the given question is correct. Answer 'Yes' if the reasoning looks correct, else answer 'No'.",
            "multi_turn_new_preplan": "Based on your confidence, provide a different strategy to use that can answer the question. Take a deep breath, analyze well, and output the name of the strategy to use. Only output the strategy name and nothing else.",
        }
        self.question_prompts = {
            "pre_plan": "Question: %s"
        }
    
    def _generate_prefix(self):
        strategy_desc = [ self.strategy_descriptions[field] for field in self.fields ]
        prefix = "Strategy Description:\n%s" % "\n".join(strategy_desc)
        return prefix
    
    def parse_output(self, output):
        pass
        # output = output.split("\n")[0].strip().lower()
        # output = output.replace("<|eot_id|>", "")
        # if "yes" in output:
        #     return 1
        # elif "no" in output:
        #     return 0
        # else:
        #     return -1
    
    def _get_prompt(self, datapoint):
        prompt = self.prefix + "\n\n"
        prompt += self.instruction + "\n\n"
        question = datapoint["question"]
        prompt += self.question_prompt % (question)
        return prompt

    def _get_decomposed_prompt_preplan(self, datapoint):
        if self.single_turn:
            instruction_template = self.instructions["single_turn_output"]
        else:
            instruction_template = self.instructions["multi_turn_preplan"]

        instruction = self.prefix + "\n\n"
        instruction += instruction_template + "\n\n"
        question = datapoint["question"]
        input = self.question_prompts["pre_plan"] % (question)
        return self.system_prompt, instruction, input

    def _create_history(self, previous_datapoints):
        history = [ [(dt["instruction"] + "\n" + dt["input"]).strip(), dt["output"]] for dt in previous_datapoints ]
        return history

    def _get_decomposed_prompt_strategy_use(self, previous_datapoints):
        instruction = self.instructions["multi_turn_strategy_use"]
        history = self._create_history(previous_datapoints)
        return self.system_prompt, instruction, history

    def _get_decomposed_prompt_verify(self, previous_datapoints):
        instruction = self.instructions["multi_turn_verify_binary"]
        history = self._create_history(previous_datapoints)
        return self.system_prompt, instruction, history
    
    def _get_decomposed_prompt_sec_preplan(self, previous_datapoints, used_fields=[]):
        instruction_template = self.instructions["multi_turn_new_preplan"]
        strategy_desc = [ self.strategy_descriptions[field] for field in self.fields if field not in used_fields ]
        prefix = "Strategy Description:\n%s" % "\n".join(strategy_desc)
        instruction = prefix + "\n\n"
        instruction += instruction_template
        history = self._create_history(previous_datapoints)
        return self.system_prompt, instruction, history

    def strategize(self, datapoint, return_probs=False, prob_tokens=[]):
        prompt = self._get_prompt(datapoint)
        if return_probs:
            output, tokens, probs = self.model.generate(prompt, self.max_length, system_prompt=self.system_prompt, return_probs=True, prob_tokens=prob_tokens)
            binary_decision = self.parse_output(output)
            return binary_decision, (output, tokens, probs)
        else:
            output = self.model.generate(prompt, self.max_length, system_prompt=self.system_prompt)
        binary_decision = self.parse_output(output)
        return binary_decision, output