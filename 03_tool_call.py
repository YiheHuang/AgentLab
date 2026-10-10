import os
from dotenv import load_dotenv
from openai import OpenAI

import json

def multiply(a: int, b: int) -> int:
    return a * b

def add(a: int, b: int) -> int:
    return a + b

load_dotenv()
api_key=os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError('DEEPSEEK_API_KEY not found')

client = OpenAI(
    api_key = api_key,
    base_url="https://api.deepseek.com"
)



tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "计算两个整数的乘积。",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "第一个整数"
                    },
                    "b": {
                        "type": "integer",
                        "description": "第二个整数"
                    }
                },
                "required": ["a", "b"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "计算两个整数的和。",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "第一个整数"
                    },
                    "b": {
                        "type": "integer",
                        "description": "第二个整数"
                    }
                },
                "required": ["a", "b"],
                "additionalProperties": False
            }
        }
    }
]


messages = [
    {
        "role": "system",
        "content": "你是一个 AI 助手。遇到计算任务时，请使用提供的工具。"
    },
    {
        "role": "user",
        "content": "请计算 123 加 321 的和乘以 456。"
    }
]

response = client.chat.completions.create(
    model="deepseek-flash",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    extra_body={"thinking": {"type": "disabled"}}
)

message = response.choices[0].message

print("结束原因：", response.choices[0].finish_reason)
print("模型返回的消息：")
print(message.model_dump_json(indent=2))

# 确认模型确实请求了工具
if not message.tool_calls:
    raise RuntimeError("模型没有返回工具调用请求")

messages.append(message)


for call in message.tool_calls:
    function_name = call.function.name
    
    try:
        # 1. 检查工具是否在允许执行的范围内
        if function_name != "multiply" and function_name !='add':
            raise ValueError("不允许调用此工具")

        # 2. 将 JSON 字符串解析成 Python 字典
        args = json.loads(call.function.arguments)

         # 3. 校验参数的名称和类型
        if (
            not isinstance(args, dict)
            or set(args.keys()) != {"a", "b"}
            or type(args["a"]) is not int
            or type(args["b"]) is not int
        ):
            raise ValueError("multiply/add 需要两个整数参数 a 和 b")
        
        # 4. 真正执行 Python 函数
        if function_name == 'multiply':
            result = str(multiply(args["a"], args["b"]))
        else:
            result = str(add(args['a'],args['b']))
    
    except (ValueError, TypeError, KeyError) as error:
        result = f"工具执行失败：{error}"

    print("执行工具：", function_name)
    print("执行结果：", result)
    
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "content": result
    })


# 将包含工具调用及执行结果的完整历史再次发送给模型
response2 = client.chat.completions.create(
    model="deepseek-flash",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    extra_body={"thinking": {"type": "disabled"}}
)

final_answer = response2.choices[0].message.content

print("\nAI 最终回答：", final_answer)

print("\n--- Token 统计 ---")
print("第一次请求：", response.usage.total_tokens)
print("第二次请求：", response2.usage.total_tokens)
print(
    "本次任务合计：",
    response.usage.total_tokens + response2.usage.total_tokens
)
