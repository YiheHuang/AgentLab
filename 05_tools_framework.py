
"""AgentLab Lesson 05: Tool Registry、Schema 校验与执行权限。

在第四课 ReAct Agent Loop 的基础上，将工具管理从 if/elif 重构成统一注册表。
安装依赖：pip install openai python-dotenv jsonschema
运行方式：python 05_tools_framework.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from jsonschema import Draft202012Validator, ValidationError
from openai import OpenAI

# ---------- 1. 定义工具及执行资源限制 ----------
MAX_STEPS = 6                  # 一次用户任务最多允许 6 次 LLM 决策
MAX_TOOL_CALLS = 10            # 一次用户任务最多处理 10 个工具调用
MAX_ABS_ARG = 1_000_000        # 数学工具参数的绝对值上限
NOTES_PATH = Path(__file__).resolve().with_name("agent_notes.txt")

def multiply(a: int, b: int) -> int:
    """乘法工具：没有外部副作用。"""
    return a * b


def add(a: int, b: int) -> int:
    """加法工具：没有外部副作用。"""
    return a + b


def save_note(text: str) -> str:
    """将笔记写入固定文件。注意：这是真实的、有副作用的操作。"""
    # 路径由 Harness 固定，模型不能通过参数选择任意文件。
    with NOTES_PATH.open("a", encoding="utf-8") as file:
        file.write(text + "\n")
    return f"已保存到 {NOTES_PATH.name}"

# ---------- 2. 注册表：统一管理函数、Schema 和权限策略 ----------
# 每个工具只需注册一次，分派器与 LLM 工具说明都从此处获取。
TOOL_REGISTRY = {}


def register_tool(name: str, description: str, handler, parameters: dict,
                  permission: str = "auto") -> None:
    """注册一个工具；相同名称不能重复注册。"""
    if name in TOOL_REGISTRY:
        raise ValueError(f"工具已注册：{name}")
    if not callable(handler):
        raise TypeError(f"{name} 的 handler 不是可调用对象")
    if permission not in ("auto", "confirm", "deny"):
        raise ValueError(f"未知权限策略：{permission}")

    # 检查“工具说明书”本身是否是合法 JSON Schema。
    Draft202012Validator.check_schema(parameters)
    TOOL_REGISTRY[name] = {
        "description": description,
        "handler": handler,            # Python 函数对象，而不是调用结果
        "parameters": parameters,      # 模型可见的参数 Schema
        "permission": permission,      # auto / confirm / deny
    }

def get_llm_tools() -> list:
    """从注册表生成 DeepSeek 所需的 tools 列表。"""
    result = []
    for name, spec in TOOL_REGISTRY.items():
        if spec["permission"] == "deny":
            continue                   # 禁用工具不对模型公布
        result.append({
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        })
    return result

# 数学工具共用一份参数规则：不重复定义 a、b 的类型与范围。
MATH_PARAMETERS = {
    "type": "object",
    "properties": {
        "a": {"type": "integer", "minimum": -MAX_ABS_ARG, "maximum": MAX_ABS_ARG},
        "b": {"type": "integer", "minimum": -MAX_ABS_ARG, "maximum": MAX_ABS_ARG},
    },
    "required": ["a", "b"],
    "additionalProperties": False,
}

register_tool(
    "multiply", "计算两个整数的乘积。", multiply, MATH_PARAMETERS,
    permission="auto",
)
register_tool(
    "add", "计算两个整数的和。", add, MATH_PARAMETERS,
    permission="auto",
)
register_tool(
    "save_note",
    "在用户明确要求保存笔记时，将一条不超过 200 字的单行笔记保存到本地文件。",
    save_note,
    {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "minLength": 1,
                "maxLength": 200,
                "pattern": r"^[^\r\n]*$",    # 不允许换行，以保持一行一条记录
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    permission="confirm",           # 有副作用的工具须由用户逐次批准
)

# ---------- 3. 工具执行：白名单查找 → 解析校验 → 权限检查 → 执行 ----------
def confirm_tool_action(name: str, arguments: dict) -> bool:
    """真实操作前询问用户：只有输入 yes 才批准。"""
    print(f"[CONFIRM] 工具：{name}")
    print(f"[CONFIRM] 参数：{json.dumps(arguments, ensure_ascii=False)}")
    try:
        answer = input("允许执行？输入 yes 批准，其余输入拒绝：").strip().lower()
    except EOFError:
        return False
    return answer == "yes"

def tool_result(ok: bool, **fields) -> str:
    """Tool Observation 一律是 JSON 字符串，便于模型检查成功/失败。"""
    return json.dumps({"ok": ok, **fields}, ensure_ascii=False)

def execute_tool(call, approval_callback=confirm_tool_action) -> str:
    """只按注册表执行；工具名称与 arguments 都来自不可信的模型输出。"""
    name = call.function.name
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return tool_result(False, error="unknown_tool")
    if spec["permission"] == "deny":
        return tool_result(False, error="permission_denied")

    try:
        arguments = json.loads(call.function.arguments)
        # JSON 可以是数字、数组等，因此还必须检查是否是对象。
        if not isinstance(arguments, dict):
            return tool_result(False, error="arguments_must_be_object")
        # JSON Schema 进行真正的运行时检查（包括类型、必填项和范围）。
        Draft202012Validator(spec["parameters"]).validate(arguments)
    except (json.JSONDecodeError, TypeError, ValidationError):
        return tool_result(False, error="invalid_arguments")

    # 工具“可用”不代表工具“被授权执行”。
    if spec["permission"] == "confirm":
        if not approval_callback(name, arguments):
            return tool_result(False, error="user_rejected")

    try:
        # **arguments 例如将 {"a": 3, "b": 4} 展开为 a=3, b=4。
        value = spec["handler"](**arguments)
        return tool_result(True, result=value)
    except Exception as error:
        # 给模型的结果使用通用错误，避免泄漏本地堆栈及路径等信息。
        print(f"[TOOL ERROR] {name}: {type(error).__name__}")
        return tool_result(False, error="tool_execution_failed")

# ---------- 4. ReAct Loop：在第四课基础上替换工具分派器 ----------
def run_agent(client: OpenAI, history: list, user_input: str):
    """完成一次用户任务，成功返回完整新历史；失败返回 None。"""
    working_messages = history.copy()  # 临时列表，失败时不提交到 Session
    working_messages.append({"role": "user", "content": user_input})
    tool_count = 0

    for step in range(1, MAX_STEPS + 1):
        llm_tools = get_llm_tools()      # 每轮根据注册表生成最新可用工具
        request = {
            "model": "deepseek-flash",
            "messages": working_messages,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if llm_tools:
            request["tools"] = llm_tools
            request["tool_choice"] = "auto"

        try:
            response = client.chat.completions.create(**request)
        except Exception as error:
            print(f"[API ERROR] {type(error).__name__}: {error}")
            return None

        choice = response.choices[0]
        message = choice.message
        print(f"[LLM] 第 {step} 轮，finish_reason={choice.finish_reason}")

        if message.tool_calls:
            if choice.finish_reason != "tool_calls":
                print("[HALT] 工具调用响应不完整")
                return None
            if tool_count + len(message.tool_calls) > MAX_TOOL_CALLS:
                print("[HALT] 达到工具调用次数上限")
                return None
            if step == MAX_STEPS:
                print("[HALT] 无剩余轮次接收工具执行结果")
                return None

            # 先保存 assistant 的完整工具调用消息，随后逐一保存配对的结果。
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
            continue                     # Observation 已加入，继续让模型决策

        if choice.finish_reason == "stop" and message.content:
            working_messages.append(message)
            print(f"[FINAL] {message.content}")
            return working_messages

        print(f"[HALT] 没有收到完整的最终回答：{choice.finish_reason}")
        return None

    print("[HALT] 达到 LLM 轮次上限")
    return None


# ---------- 5. 外层会话：保留第二课的 exit / reset ----------
def main() -> None:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("找不到 DEEPSEEK_API_KEY，请检查 .env")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    messages = [{
        "role": "system",
        "content": (
            "你是一个乐于助人的助手。计算任务优先使用数学工具验证。"
            "只有用户明确要求记录时才使用 save_note。"
            "工具调用的成功与否以返回的 ok 字段为准；用户拒绝时不得声称保存成功。"
        ),
    }]

    print("AgentLab Lesson 05：输入 exit 退出，/reset 清空会话。")
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

        updated = run_agent(client, messages, user_input)
        if updated is not None:
            messages = updated
        else:
            print("[SESSION] 本次任务未完成，未提交对话历史。")


if __name__ == "__main__":
    main()
