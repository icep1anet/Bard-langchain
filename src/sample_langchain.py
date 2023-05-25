from os import environ
from Bard import Chatbot
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.chat_models import ChatOpenAI

load_dotenv()

token = environ.get("BARD_TOKEN")
chatbot = Chatbot(token)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
"""
def read_contents():
    with open('テキストファイルのパス') as f:
    Pixel_data = f.read()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(Pixel_Data)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    qa = VectorDBQA.from_llm(llm=OpenAI(), vectorstore=docsearch)
"""
