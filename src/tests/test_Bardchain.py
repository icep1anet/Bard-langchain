from os import path
import sys
sys.path.append(path.dirname(path.abspath(__file__)) + "/../")

from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain

from Bard_langchain.bard import Bard

load_dotenv()
llm = Bard()

template = """質問: {question}

回答: 段階的に考えてください。"""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

question = "関ヶ原の戦いで勝ったのは?"

print(llm_chain.predict(question=question))