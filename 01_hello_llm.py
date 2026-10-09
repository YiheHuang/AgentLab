import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("DEEPSEEK API KEY not found.")

client = OpenAI(
    api_key = api_key,
    base_url = "https://api.deepseek.com"
)

messages = [
    {
        "role":"system",
        "content":"你是一位严格的计算机科学教授。"
    },
    {
        "role":"user",
        "content":"请用了两句话解释什么是 Agent Harness。"
    }
]

response = client.chat.completions.create(
    model = "deepseek-flash",
    messages = messages,
    extra_body = {"thinking":{"type":"disabled"}}
)


answer = response.choices[0].message.content
print(answer)
print(response.model_dump_json(indent=2))