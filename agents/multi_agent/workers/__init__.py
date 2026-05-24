"""
Worker 工厂函数 —— 按名称创建对应类型的 WorkerAgent。

用法：
    calc_worker = create_worker("calculator", tools=calc_tools)
    code_worker = create_worker("coder", tools=code_tools)
"""

from agents.tools import ToolRegistry


def create_worker(name: str, tools: ToolRegistry, model: str = "MiniMax-M2.7"):
    """根据名称创建 WorkerAgent。

    支持的 Worker 类型：
        - "calculator"  → CalculatorWorker   (计算/数学)
        - "coder"       → CoderWorker        (编程/代码)
        - "researcher"  → ResearchWorker     (研究/检索)

    参数：
        name: Worker 类型名称
        tools: 该 Worker 使用的工具注册表
        model: LLM 模型名称

    返回：
        对应类型的 WorkerAgent 实例

    异常：
        ValueError: 传入未知的 Worker 类型名称
    """
    from ..agents.worker import CalculatorWorker, CoderWorker, ResearchWorker

    # Worker 名称 → 类的映射
    workers = {
        "calculator": CalculatorWorker,
        "coder": CoderWorker,
        "researcher": ResearchWorker,
    }

    worker_class = workers.get(name)
    if worker_class is None:
        raise ValueError(f"Unknown worker type: {name}. Available: {list(workers.keys())}")

    return worker_class(tools=tools, model=model)
