from os import path
import sys
sys.path.append(path.dirname(path.abspath(__file__)) + "/../")

from dotenv import load_dotenv

from Bard_langchain.bard import Bard

load_dotenv()
llm = Bard()

ans = llm("こんにちは！元気ですか？")
print(ans)
