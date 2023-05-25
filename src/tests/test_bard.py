from os import environ
from Bard import Chatbot
from dotenv import load_dotenv

load_dotenv()

token = environ.get("BARD_SESSION_ID")
chatbot = Chatbot(token)

# return is dict
"""
{'content': 'こんにちは！元気です。今日は何かお手伝いできることはありますか？', 
'conversation_id': 'c_aeebcb8edb13a02d', 
'response_id': 'r_aeebcb8edb13a6e4', 
'factualityQueries': [], 
'textQuery': '', 
'choices': 
    [{'id': 'rc_aeebcb8edb13aabc', 'content': ['こんにちは！元気です 。今日は何かお手伝いできることはありますか？']}, 
    {'id': 'rc_aeebcb8edb13ae41', 'content': ['こんにちは！お元気ですか？私は元気です。今日はあなたに何ができるか教えてください。']}, 
    {'id': 'rc_aeebcb8edb13a1c6', 'content': ['こんにちは！お元気ですか？お元気ですか？私は、情報量が多く包括的であるようにトレーニングされた会話型 AI またはチャットボットとしても知られる大規模な言語モデルです。私は大量のテキストデータでトレーニングされており、幅広いプロンプトや質問に応じて、人間のようなテキストを通信および生成することができます。たとえば、事実のトピックの要約を提供したり、ストーリーを作成したりできます。']}]}
"""
ans = chatbot.ask("こんにちは！元気ですか？")
print(ans["content"])
# print(type(ans))
