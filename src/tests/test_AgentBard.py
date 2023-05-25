from langchain.agents import load_tools
from langchain.agents import initialize_agent

from os import path
import sys
sys.path.append(path.dirname(path.abspath(__file__)) + "/../")

from dotenv import load_dotenv

from Bard_langchain.bard import Bard

load_dotenv()
llm = Bard()
# from langchain.llms import OpenAI

# llm = OpenAI(temperature=0)
tools = load_tools(["wikipedia"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
agent.run("関ヶ原の戦いは西暦何年？")