from os import environ

from bard import Bard
from langchain.schema import (
    HumanMessage,
)
from dotenv import load_dotenv

load_dotenv()

chat = Bard()
resp = chat([HumanMessage(content="最も新しいPixel製品は何？")])