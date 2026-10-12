import json
import os

from dotenv import load_dotenv
from openai import OpenAI

# 1. 初始化 API
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("DEEPSEEK_API_KEY not found.")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

# 2. 本地工具：真正由 Python 执行
def multiply(a: int, b: int) -> int:
    return a * b


def add(a: int, b: int) -> int:
    return a + b

# 3. 工具说明书：提供给模型选择
# JSON Schema 描述输入格式，但 Harness 仍要做运行时校验。
tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "计算两个整数的乘积。",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "第一个整数"},
                    "b": {"type": "integer", "description": "第二个整数"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "计算两个整数的和。",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "第一个整数"},
                    "b": {"type": "integer", "description": "第二个整数"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    },
]

MAX_STEPS = 5         # 单次任务最多调用多少轮 LLM
MAX_TOOL_CALLS = 10   # 单次任务累计最多处理多少个工具请求
MAX_ABS_ARG = 1_000_000


def execute_tool(call) -> str:
    """校验并执行一条工具请求，返回 JSON 格式的 Observation。"""
    name = call.function.name

    try:
        # 工具参数在 API 响应里是 JSON 字符串，不是 Python 字典。
        arguments = json.loads(call.function.arguments)

        if name not in ("multiply", "add"):
            raise ValueError(f"未授权的工具: {name}")
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        if set(arguments.keys()) != {"a", "b"}:
            raise ValueError("工具参数必须且只能包含 a 和 b")

        a, b = arguments["a"], arguments["b"]
        if type(a) is not int or type(b) is not int:
            raise ValueError("参数 a 和 b 必须是整数")
        if abs(a) > MAX_ABS_ARG or abs(b) > MAX_ABS_ARG:
            raise ValueError("参数超出了允许的数值范围")

        if name == "multiply":
            result = multiply(a, b)
        else:
            result = add(a, b)

        return json.dumps({"ok": True, "result": result}, ensure_ascii=False)

    except (ValueError, TypeError) as error:
        # 不让单个参数错误破坏工具消息配对；把错误反馈给模型。
        return json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False)
    

def run_agent(history: list, user_input: str):
    """运行一次用户任务；成功返回新的会话历史，失败返回 None。"""
    working_messages = history.copy()
    working_messages.append({"role": "user", "content": user_input})
    tool_count = 0

    for step in range(1, MAX_STEPS + 1):
        try:
            response = client.chat.completions.create(
                model="deepseek-flash",
                messages=working_messages,
                tools=tools,
                tool_choice="auto",
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as error:
            print(f"[API ERROR] {error}")
            return None

        choice = response.choices[0]
        message = choice.message
        print(f"[LLM] 第 {step} 轮，finish_reason={choice.finish_reason}")

        if message.tool_calls:
            if choice.finish_reason != "tool_calls":
                print("[HALT] 工具调用响应未正常完成")
                return None

            n_calls = len(message.tool_calls)
            if tool_count + n_calls > MAX_TOOL_CALLS:
                print("[HALT] 达到工具调用次数上限")
                return None
            if step == MAX_STEPS:
                print("[HALT] 已到最后一轮，无法再处理工具结果")
                return None
            
            # 一个 assistant 工具调用消息，需要对应每一条 tool 结果。
            working_messages.append(message)
            for call in message.tool_calls:
                print(f"[TOOL] {call.function.name}({call.function.arguments})")
                observation = execute_tool(call)
                print(f"[OBSERVATION] {observation}")
                working_messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": observation,
                })
                tool_count += 1

            continue  # 再次请求 LLM，让它观察工具结果并决定下一步。

        if choice.finish_reason == "stop" and message.content:
            working_messages.append(message)
            print(f"[FINAL] {message.content}")
            return working_messages  
        
        print(f"[HALT] 模型未正常给出最终回答：{choice.finish_reason}")
        return None

    print("[HALT] 达到最大决策轮数")
    return None

# 4. 外层 Session Loop：负责多轮用户对话
messages = [{
    "role": "system",
    "content": (
        "你是一个乐于助人的 AI 助手。"
        "如果用户明确要求使用数学工具，请优先调用工具验证结果。"
        "不要声称未执行过的工具操作已经成功。"
    ),
}]


while True:
    user_input = input("\n你: ").strip()

    if user_input.lower() == "exit":
        print("聊天结束！")
        break
    if user_input.lower() == "/reset":
        messages = messages[:1]
        print("会话已重置！")
        continue
    if not user_input:
        continue

    updated_messages = run_agent(messages, user_input)
    if updated_messages is not None:
        messages = updated_messages
    else:
        print("[SESSION] 本次任务未完成，会话历史未更新。")