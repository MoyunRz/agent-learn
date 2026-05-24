"""
多 Agent 协作演示 —— Supervisor + Planner + Worker 协同工作。

演示内容：
- PlannerAgent 将复杂查询分解为子任务
- CalculatorWorker 处理数学计算（add / multiply）
- CoderWorker 处理代码验证任务
- SupervisorAgent 协调执行顺序（先计算、再验证）

预期流程：
    用户查询 "先计算 15+25，然后乘以 3，最后用代码验证结果"
    → Planner 分解为 3 个 Task：
        task_1 (calculation): 计算 15+25
        task_2 (calculation): 将结果乘以 3（依赖 task_1）
        task_3 (coding): 用 Python 代码验证结果（依赖 task_2）
    → Supervisor 按依赖顺序调度 Worker 执行

运行：
    python main_multi_agent.py
"""

from dotenv import load_dotenv
import os
import anthropic
from agents.multi_agent.agents import SupervisorAgent, PlannerAgent
from agents.multi_agent.agents.worker import CalculatorWorker, CoderWorker
from agents.multi_agent.shared_state import SharedState
from agents.multi_agent.memory import AgentMemory
from agents.tools import ToolRegistry, Tool
from agents.logging_config import setup_logging

load_dotenv()  # 加载 .env 中的 API_KEY
setup_logging()  # 初始化日志系统

# ---- 创建共享的 Anthropic 客户端 ----
client = anthropic.Anthropic(
    base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic"),
    api_key=os.getenv("MINIMAX_API_KEY", "<api key>"),
)

# ---- 配置工具 ----
# 计算 Worker 的工具：加法和乘法
def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

calc_tools = ToolRegistry()
calc_tools.register(Tool(add, name="add", description="Add two numbers"))
calc_tools.register(Tool(multiply, name="multiply", description="Multiply two numbers"))

# 编程 Worker 的工具：执行 Python 代码
code_tools = ToolRegistry()
code_tools.register(Tool(lambda code: f"# Code executed:\n{code}", name="execute_code", description="Execute Python code and return output"))

# ---- 组建多 Agent 系统 ----
# 0. 长期记忆 —— 跨会话持久化，Worker 执行时自动检索历史上下文
memory = AgentMemory(persist_file=".agent_memory.json")

# 0. 约束规则 —— 注入到所有 LLM 调用中的行为规则
system_rules = """你必须严格遵守以下规则：
1. 所有数学计算结果必须是数字，不能是字符串
2. 每次计算完成后，用一句话总结结果
3. 用中文回答
4. 如果之前有相关计算结果，优先引用，不要重复计算
5. 回答要简洁，不要输出多余的解释"""

# 1. Planner —— 负责分解查询
planner = PlannerAgent(client=client)

# 2. Supervisor —— 负责调度执行（Planner + Worker 池 + 共享状态 + 记忆 + 规则）
supervisor = SupervisorAgent(
    planner=planner,
    workers={
        "calculator": CalculatorWorker(tools=calc_tools, client=client, model="MiniMax-M2.7",
                                       system_rules=system_rules),
        "coder": CoderWorker(tools=code_tools, client=client, model="MiniMax-M2.7",
                            system_rules=system_rules),
    },
    shared_state=SharedState(persist_file=".agent_state.json"),
    memory=memory,
    system_rules=system_rules,
)

if __name__ == "__main__":
    query = "先计算 15+25，然后乘以 3，最后用代码验证结果"
    print(f"Query: {query}\n")

    # 一次调用完成：打印计划 + 执行，verbose=True 显示内部日志
    plan = planner.run(query, verbose=True)

    print("\n" + "=" * 60)
    print("AI Planner 生成的执行计划:")
    print("=" * 60)
    for task in plan.tasks:
        print(f"  [{task.id}] type={task.type}")
        print(f"       描述: {task.description}")
        print(f"       依赖: {task.dependencies}")
        print(f"       期望输出: {task.expected_output}")
        print()

    print("=" * 60)
    print("Supervisor 开始执行:")
    print("=" * 60)

    # 注意：这里 supervisor.run 会再次调用 planner
    # 如果要避免重复调用 planner，可以给 supervisor 加 set_plan() 方法
    result = supervisor.run(query, verbose=True)
    print(f"\n最终结果:\n{result}")

    # ---- 演示：二次查询，Worker 自动引用上次记忆 ----
    print("\n" + "=" * 60)
    print("二次查询: 基于之前的结果继续计算")
    print("=" * 60)
    query2 = "把之前的最终结果再除以 10"
    print(f"Query: {query2}\n")

    result2 = supervisor.run(query2, verbose=True)
    print(f"\n最终结果:\n{result2}")

    # ---- 查看已存储的记忆 ----
    print("\n" + "=" * 60)
    print("已存储的长期记忆:")
    print("=" * 60)
    for entry in memory.get_recent(limit=10):
        print(f"  [{entry['key']}] ({entry['tags']}) {entry['value']}")

    # 优雅关闭：显式关闭 client 的链接池，释放 daemon 线程
    client.close()
