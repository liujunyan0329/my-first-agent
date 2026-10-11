import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

resp = client.chat.completions.create(
    model = "deepseek-chat",
    messages = [
        {"role":"system","content":"你是一个简洁的助手。"},
        {"role":"user","content":"用一句话介绍一下什么是ai agent。"},
    ],
    temperature = 0.3,
)

print(resp.choices[0].message.content)
print("___")
print("输入 token:",resp.usage.prompt_tokens)
print("输出 token:",resp.usage.completion_tokens)


