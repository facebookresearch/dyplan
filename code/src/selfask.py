# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

SELFASK_ORIG_FEWSHOT4_OLD = """Question: Who lived longer, Muhammad Ali or Alan Turing?
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the answer is Muhammad Ali.

Question: When was the founder of craigslist born?
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the answer is December 6, 1952.

Question: Who was the maternal grandfather of George Washington?
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the answer is Joseph Ball 

Question: Are both the directors of Jaws and Casino Royale from the same country? 
Follow up: Who is the director of Jaws? 
Intermediate Answer: The director of Jaws is Steven Spielberg. 
Follow up: Where is Steven Spielberg from? 
Intermediate Answer: The United States. 
Follow up: Who is the director of Casino Royale? 
Intermediate Answer: The director of Casino Royale is Martin Campbell. 
Follow up: Where is Martin Campbell from? 
Intermediate Answer: New Zealand. 
So the answer is no.

"""

SELFASK_ORIG_FEWSHOT4 = """Question: Who lived longer, Muhammad Ali or Alan Turing?
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
Final answer: "Muhammad Ali".

Question: When was the founder of craigslist born?
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
Final answer: "December 6, 1952".

Question: Who was the maternal grandfather of George Washington?
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
Final answer: "Joseph Ball". 

Question: Are both the directors of Jaws and Casino Royale from the same country? 
Follow up: Who is the director of Jaws? 
Intermediate Answer: The director of Jaws is Steven Spielberg. 
Follow up: Where is Steven Spielberg from? 
Intermediate Answer: The United States. 
Follow up: Who is the director of Casino Royale? 
Intermediate Answer: The director of Casino Royale is Martin Campbell. 
Follow up: Where is Martin Campbell from? 
Intermediate Answer: New Zealand. 
Final answer: "no".

"""

SELFASK_ORIG_FEWSHOT6 = """Question: Who lived longer, Muhammad Ali or Alan Turing?
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
Final answer: "Muhammad Ali".

Question: When was the founder of craigslist born?
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
Final answer: "December 6, 1952".

Question: Who was the maternal grandfather of George Washington?
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
Final answer: "Joseph Ball". 

Question: Are both the directors of Jaws and Casino Royale from the same country? 
Follow up: Who is the director of Jaws? 
Intermediate Answer: The director of Jaws is Steven Spielberg. 
Follow up: Where is Steven Spielberg from? 
Intermediate Answer: The United States. 
Follow up: Who is the director of Casino Royale? 
Intermediate Answer: The director of Casino Royale is Martin Campbell. 
Follow up: Where is Martin Campbell from? 
Intermediate Answer: New Zealand. 
Final answer: "no".

Question: Jeremy Theobald and Christopher Nolan share what profession?
Follow up: What is the profession of Jeremy Theobald?
Intermediate Answer: Jeremy Theobald is an actor and producer.
Follow up: What is the profession of Christopher Nolan?
Intermediate Answer: Christopher Nolan is a director, producer, and screenwriter.
Final answer: "producer".

Question: Who was born first? Jan de Bont or Raoul Walsh?
Follow up: When was Jan de Bont born?
Intermediate Answer: Jan de Bont was born on 22 October 1943.
Follow up: When was Raoul Walsh born?
Intermediate Answer: Raoul Walsh was born on March 11, 1887.
Final answer: "Raoul Walsh".

"""

SELFASK_ORIG_FEWSHOT8 = """Question: Who lived longer, Muhammad Ali or Alan Turing?
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
Final answer: "Muhammad Ali".

Question: When was the founder of craigslist born?
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
Final answer: "December 6, 1952".

Question: Who was the maternal grandfather of George Washington?
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
Final answer: "Joseph Ball". 

Question: Are both the directors of Jaws and Casino Royale from the same country? 
Follow up: Who is the director of Jaws? 
Intermediate Answer: The director of Jaws is Steven Spielberg. 
Follow up: Where is Steven Spielberg from? 
Intermediate Answer: The United States. 
Follow up: Who is the director of Casino Royale? 
Intermediate Answer: The director of Casino Royale is Martin Campbell. 
Follow up: Where is Martin Campbell from? 
Intermediate Answer: New Zealand. 
Final answer: "no".

Question: Jeremy Theobald and Christopher Nolan share what profession?
Follow up: What is the profession of Jeremy Theobald?
Intermediate Answer: Jeremy Theobald is an actor and producer.
Follow up: What is the profession of Christopher Nolan?
Intermediate Answer: Christopher Nolan is a director, producer, and screenwriter.
Final answer: "producer".

Question: Who was born first? Jan de Bont or Raoul Walsh?
Follow up: When was Jan de Bont born?
Intermediate Answer: Jan de Bont was born on 22 October 1943.
Follow up: When was Raoul Walsh born?
Intermediate Answer: Raoul Walsh was born on March 11, 1887.
Final answer: "Raoul Walsh".

Question: How many episodes were in the South Korean television series in which Ryu Hye-young played Bo-ra?
Follow up: What was the South Korean television series in which Ryu Hye-young played Bo-ra?
Intermediate Answer: The South Korean television series in which Ryu Hye-young played Bo-ra is Reply 1988.
Follow up: How many episodes were there in Reply 1988?
Intermediate Answer: The number of episodes Reply 1988 has was 20.
Final answer: "20".

Question: Were Lonny and Allure both founded in the 1990s?
Follow up: When was Lonny founded?
Intermediate Answer: Lonny (magazine) was founded in 2009.
Follow up: When was Allure founded?
Intermediate Answer: Allure (magazine) was founded in 1991.
Final answer: "no".

"""

SELFASK_ORIG_FOLLOWUP = "Follow up: "
SELFASK_ORIG_INT_ANSWER = "Intermediate answer: "
SELFASK_DRAGIN_FEWSHOT4 = """
Question: Jeremy Theobald and Christopher Nolan share what profession?
Follow up: What is the profession of Jeremy Theobald?
Intermediate Answer: Jeremy Theobald is an actor and producer.
Follow up: What is the profession of Christopher Nolan?
Intermediate Answer: Christopher Nolan is a director, producer, and screenwriter.
Final answer: "producer".

Question: Who was born first? Jan de Bont or Raoul Walsh?
Follow up: When was Jan de Bont born?
Intermediate Answer: Jan de Bont was born on 22 October 1943.
Follow up: When was Raoul Walsh born?
Intermediate Answer: Raoul Walsh was born on March 11, 1887.
Final answer: "Raoul Walsh".

Question: How many episodes were in the South Korean television series in which Ryu Hye-young played Bo-ra?
Follow up: What was the South Korean television series in which Ryu Hye-young played Bo-ra?
Intermediate Answer: The South Korean television series in which Ryu Hye-young played Bo-ra is Reply 1988.
Follow up: How many episodes were there in Reply 1988?
Intermediate Answer: The number of episodes Reply 1988 has was 20.
Final answer: "20".

Question: Were Lonny and Allure both founded in the 1990s?
Follow up: When was Lonny founded?
Intermediate Answer: Lonny (magazine) was founded in 2009.
Follow up: When was Allure founded?
Intermediate Answer: Allure (magazine) was founded in 1991.
Final answer: "no".

"""
SELFASK_CASE = "Answer the following question by reasoning step-by-step, following the examples above.\nQuestion: "