from Bard_langchain.bard import BardChat
from os import environ
from dotenv import load_dotenv

load_dotenv()
llm = BardChat()