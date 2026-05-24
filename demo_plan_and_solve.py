"""
Plan-and-Solve Demo —— 两阶段推理演示。

演示内容：
- 先用 P&S 解决数学计算问题
- 再用 P&S 解决需要多步骤的逻辑推理问题

运行：
    python demo_plan_and_solve.py
"""

from dotenv import load_dotenv
import os
import anthropic
from agents.plan_and_solve import PlanAndSolveAgent
from agents.tools import ToolRegistry, Tool
from agents.logging_config import setup_logging

load_dotenv()
setup_logging()

# ---- 工具 ----
def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

def divide(a: int, b: int) -> float:
    return a / b

tools = ToolRegistry()
tools.register(Tool(add, name="add", description="Add two numbers"))
tools.register(Tool(multiply, name="multiply", description="Multiply two numbers"))
tools.register(Tool(divide, name="divide", description="Divide first by second"))

# ---- 客户端 ----
client = anthropic.Anthropic(
    base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic"),
    api_key=os.getenv("MINIMAX_API_KEY", "your-key"),
)

# ---- Plan-and-Solve ----
agent = PlanAndSolveAgent(tools=tools, client=client)

if __name__ == "__main__":
    # ---- Demo 1: 数学计算 ----
    print("=" * 60)
    print("Demo 1: 数学计算")
    print("=" * 60)
    query1 = "计算 100 除以 5，然后乘以 3，再加上 20"
    print(f"Query: {query1}\n")

    plan, answer = agent.run(query1)
    print(f"\n生成的计划:")
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step}")
    print(f"\n最终答案: {answer}")

    # ---- Demo 2: 逻辑推理 ----
    print("\n" + "=" * 60)
    print("Demo 2: 逻辑推理")
    print("=" * 60)
    query2 = "小明有 120 元，买了 3 本书（每本 25 元）和 2 支笔（每支 8 元），还剩多少钱？"
    print(f"Query: {query2}\n")

    plan, answer = agent.run(query2)
    print(f"\n生成的计划:")
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step}")
    print(f"\n最终答案: {answer}")

    client.close()
