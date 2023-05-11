from os import environ
from Bard import Chatbot
from dotenv import load_dotenv

load_dotenv()

token = environ.get("BARD_TOKEN")
chatbot = Chatbot(token)

ans = chatbot.ask("私の好きな食べ物はリンゴです。この情報を覚えておいてください")
print(ans['choices'][0])
ans = chatbot.ask("私の好きな食べ物はなんですか？")
print(ans['choices'][0])
ans = chatbot.ask("あなたは秒間何リクエストまでなら受け付けることができますか？")
print(ans['choices'][0])
