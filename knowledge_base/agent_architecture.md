# Agent Harness 架构

Agent Harness 负责维护 messages、调用模型、处理 Tool Call、执行工具并记录 Observation。模型提出 Action，但实际执行由 Harness 控制。

ReAct Agent Loop 持续进行“模型决策 → 工具执行 → 结果观察”，直到模型正常给出最终答案。MAX_STEPS 限制一次任务中的 LLM 决策轮数；MAX_TOOL_CALLS 则单独限制工具调用次数。两者不能互相替代。

Tool Registry 将工具名称、Python 函数、JSON Schema 与权限策略统一管理。模型看见工具不代表自动拥有使用权限。
