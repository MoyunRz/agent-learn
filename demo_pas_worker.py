"""
Plan-and-Solve Worker Demo —— Multi-Agent 内嵌 P&S 推理演示。

演示内容：
- 一个复杂查询被 Planner 拆成 3 个 Task（计算 + 编码 + 分析）
- 其中「计算 Task」由 PlanAndSolveWorker 执行，内部再进行 Plan → Solve
- 展示两层计划的嵌套：宏观 Multi-Agent Plan + 微观 P&S Sub-plan

运行：
    python demo_pas_worker.py
"""

from dotenv import load_dotenv
import os
import anthropic
from agents.multi_agent.agents import SupervisorAgent, PlannerAgent, PlanAndSolveWorker, CoderWorker
from agents.multi_agent.agents.worker import ResearchWorker
from agents.multi_agent.shared_state import SharedState
from agents.multi_agent.memory import AgentMemory
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

calc_tools = ToolRegistry()
calc_tools.register(Tool(add, name="add", description="a + b"))
calc_tools.register(Tool(multiply, name="multiply", description="a * b"))
calc_tools.register(Tool(subtract, name="subtract", description="a - b"))
calc_tools.register(Tool(divide, name="divide", description="a / b"))

code_tools = ToolRegistry()
code_tools.register(Tool(
    lambda code: f"执行结果: {eval(code) if 'import' not in code else 'code executed'}",
    name="execute_code", description="Execute Python code and return output"
))

research_tools = ToolRegistry()
research_tools.register(Tool(
    lambda topic: f"关于'{topic}'的分析: 这是一个经济学概念，涉及数值比较和决策分析",
    name="analyze", description="Analyze a topic and return insights"
))

# ---- 客户端 ----
client = anthropic.Anthropic(
    base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic"),
    api_key=os.getenv("MINIMAX_API_KEY", "your-key"),
)

# ---- 规则 ----
system_rules = """约束规则：
1. 计算结果必须是数字
2. 用中文回答
3. 简洁直接，不输出多余内容"""

# ---- Multi-Agent 组建 ----
# 关键：计算 Worker 使用 PlanAndSolveWorker（内部会先规划子步骤再执行）
planner = PlannerAgent(client=client)

supervisor = SupervisorAgent(
    planner=planner,
    workers={
        "calculator": PlanAndSolveWorker(
            tools=calc_tools, client=client,
            system_rules=system_rules,
            max_sub_steps=5,  # 子步骤上限
        ),
        "coder": CoderWorker(
            tools=code_tools, client=client,
            system_rules=system_rules,
        ),
        "researcher": ResearchWorker(
            tools=research_tools, client=client,
            system_rules=system_rules,
        ),
    },
    shared_state=SharedState(),
    memory=AgentMemory(),
    system_rules=system_rules,
)

if __name__ == "__main__":
    query = "综合任务：先计算 (15+25)*3 - 10 的最终值，然后用 Python 代码验证计算结果，最后分析这个结果是否适合作为投资预算"
    print("=" * 60)
    print("Plan-and-Solve Worker Demo")
    print("=" * 60)
    print(f"用户查询: {query}\n")

    # ---- 宏观 Plan ----
    print("=" * 60)
    print("【第 1 层】Planner 生成的宏观执行计划:")
    print("=" * 60)
    macro_plan = planner.run(query)
    for task in macro_plan.tasks:
        print(f"  [{task.id}] type={task.type}")
        print(f"        描述: {task.description}")
        print(f"        依赖: {task.dependencies}")
        print(f"        执行者: {'PlanAndSolveWorker' if task.type == 'calculation' else task.type}")
        print()

    # ---- 执行（Supervisor 调度 Worker，其中计算 Worker 内部触发 P&S） ----
    print("=" * 60)
    print("【第 2 层】Supervisor 调度执行（计算 Task 触发 P&S 子步骤）")
    print("=" * 60)
    result = supervisor.run(query, verbose=True)

    print("\n" + "=" * 60)
    print("最终汇总结果:")
    print("=" * 60)
    print(result)

    client.close()
