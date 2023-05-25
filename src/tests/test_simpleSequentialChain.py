from os import path
import sys
sys.path.append(path.dirname(path.abspath(__file__)) + "/../")

from dotenv import load_dotenv
from langchain import PromptTemplate, OpenAI, LLMChain

from Bard_langchain.bard import Bard

load_dotenv()
llm = Bard()

# 質問
question_llm = llm
question_template = """質問: {question}

回答: 段階的に考えてください"""
question_prompt_template = PromptTemplate(input_variables=["question"], template=question_template)
question_chain = LLMChain(llm=question_llm, prompt=question_prompt_template)

# 回答の要約
summarize_llm = llm
summarize_template = """{text}

上記を要約してください:"""
summarize_prompt_template = PromptTemplate(input_variables=["text"], template=summarize_template)
summarize_chain = LLMChain(llm=summarize_llm, prompt=summarize_prompt_template)

# 質問とその回答の要約Chain
from langchain.chains import SimpleSequentialChain
question_answer_summarize_chain = SimpleSequentialChain(chains=[question_chain, summarize_chain], verbose=True)
question_answer_summarize_chain.run("未来の大規模言語モデルはどうなりますか？")
