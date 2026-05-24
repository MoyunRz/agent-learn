"""
单 Agent 演示 —— ReActAgent 独立运行。

演示内容：
- 注册加法和乘法两个工具
- 让 Agent 解决 "(3+5)*2" 的计算问题
- 预期流程：LLM 先调用 add(3,5) → 获得 8 → 再调用 multiply(8,2) → 获得 16 → 输出 "16"

运行：
    python main.py
"""

import os
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 中的 MINIMAX_API_KEY 等环境变量

from agents import ReActAgent, Tool, tool_registry
from agents.logging_config import setup_logging

# 初始化日志（INFO 级别）
setup_logging()

# ---- 注册工具 ----
# Agent 可以调用这两个函数来完成计算
def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

tool_registry.register(Tool(add, name="add", description="两数相加"))
tool_registry.register(Tool(multiply, name="multiply", description="两数相乘"))

# ---- 创建 Agent ----
agent = ReActAgent(tools=tool_registry)

if __name__ == "__main__":
    # verbose=True 会输出详细的每一步推理和工具调用日志
    result = agent.run("计算 (3+5)*2", verbose=True)
    print(f"\n结果: {result}")

    # 优雅关闭：关闭客户端连接池，释放 daemon 线程
    agent.client.close()
