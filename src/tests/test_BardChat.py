from os import path
import sys
sys.path.append(path.dirname(path.abspath(__file__)) + "/../")

from Bard_langchain.bard import BardChat
from dotenv import load_dotenv

load_dotenv()
chat = BardChat()
resp = chat("最も新しいPixel製品は何？")
print(resp.content)