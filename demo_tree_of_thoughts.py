"""
Tree of Thoughts Demo —— 思维树广度优先搜索演示。

演示内容：
- Demo 1: 24 点游戏（需要多种策略尝试）
- Demo 2: 逻辑推理问题（需要探索多个思路）

运行：
    python demo_tree_of_thoughts.py
"""

from dotenv import load_dotenv
import os
import anthropic
from agents.tree_of_thoughts import TreeOfThoughts
from agents.tools import ToolRegistry, Tool
from agents.logging_config import setup_logging

load_dotenv()
setup_logging()

# ---- 工具 ----
def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

def subtract(a: int, b: int) -> int:
    return a - b

def divide(a: float, b: float) -> float:
    return a / b

tools = ToolRegistry()
tools.register(Tool(add, name="add", description="a + b"))
tools.register(Tool(multiply, name="multiply", description="a * b"))
tools.register(Tool(subtract, name="subtract", description="a - b"))
tools.register(Tool(divide, name="divide", description="a / b"))

# ---- 客户端 ----
client = anthropic.Anthropic(
    base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic"),
    api_key=os.getenv("MINIMAX_API_KEY", "your-key"),
)

if __name__ == "__main__":
    # ---- Demo 1: 24 点游戏 ----
    print("=" * 60)
    print("Demo 1: 24点游戏")
    print("=" * 60)
    query1 = "24点游戏: 用数字 4, 6, 8, 9 通过加减乘除算出 24，每个数字只能用一次"
    print(f"Problem: {query1}\n")

    # breadth=3 每层保留3个最优思路, candidates=3 每个节点生成3个候选
    tot = TreeOfThoughts(tools=tools, client=client, breadth=3, candidates=3, max_depth=5)
    best_path, best_score = tot.search(query1)

    print(f"\n最佳思维路径 (评分 {best_score}/10):")
    for i, step in enumerate(best_path, 1):
        print(f"  Step {i}: {step}")
    else:
        if not best_path:
            print("  (未找到有效路径)")

    # ---- Demo 2: 逻辑推理 ----
    print("\n" + "=" * 60)
    print("Demo 2: 逻辑推理")
    print("=" * 60)
    query2 = """有 3 个盒子，分别标着 "苹果"、"橙子"、"苹果和橙子"。
    所有标签都贴错了。你只能从一个盒子里拿出一个水果来看。
    如何确定每个盒子实际装的是什么？"""
    print(f"Problem: {query2}\n")

    tot2 = TreeOfThoughts(tools=tools, client=client, breadth=2, candidates=3, max_depth=4, min_score=4.0)
    best_path, best_score = tot2.search(query2)

    print(f"\n最佳思维路径 (评分 {best_score}/10):")
    for i, step in enumerate(best_path, 1):
        print(f"  Step {i}: {step}")
    else:
        if not best_path:
            print("  (未找到有效路径)")

    client.close()
