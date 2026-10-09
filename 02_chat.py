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
        "content":"你是一个乐于助人的 AI 助手。"
    }
]

previous_prompt_tokens = None

while True:
    user_input = input('\n你: ').strip()

    if user_input.lower() == "exit":
        print("聊天结束！")
        break

    if user_input.lower() == '/reset':
        messages = messages[:1]
        print("对话已重置！")
        continue
    
    if not user_input:
        continue
    

    
    messages.append({
        'role':'user',
        'content':user_input
    })

    try:
        response = client.chat.completions.create(
            model = 'deepseek-flash',
            messages = messages,
            extra_body = {'thinking':{'type':'disabled'}}
        )
        answer = response.choices[0].message.content

        if not answer:
            raise ValueError("模型未给出文本回答")
    except Exception as error:
        messages.pop()
        print('调用失败',error)
        continue
    
    messages.append({
        'role':'assistant',
        'content':answer
    })

    print('AI: ',answer)

    usage = response.usage

    print('\n----统计----')
    print('本次输入token：',usage.prompt_tokens)
    print("本次输出 Token：", usage.completion_tokens)
    print("本次请求总 Token：", usage.total_tokens)
    print("本次输入携带的消息数：", len(messages) - 1)

    if previous_prompt_tokens is not None:
        growth = usage.prompt_tokens - previous_prompt_tokens
        print("相比上一轮，输入增长：", growth)

    previous_prompt_tokens = usage.prompt_tokens
