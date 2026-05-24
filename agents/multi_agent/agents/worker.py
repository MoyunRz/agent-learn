"""
WorkerAgent —— 执行具体任务的 Agent。

架构：
- WorkerAgent: 通用 Worker 基类，使用多重继承融合 ReActAgent 的推理能力和 BaseAgent 的接口
- CalculatorWorker: 擅长数学计算
- CoderWorker: 擅长代码生成和执行
- ResearchWorker: 擅长信息检索和研究

每个 Worker 都是独立的 ReActAgent，拥有自己的工具集。
"""

from agents.react import ReActAgent
from agents.tools import ToolRegistry
from .base import BaseAgent
from ..message import Task


class WorkerAgent(ReActAgent, BaseAgent):
    """通用 Worker —— 从 Supervisor 接收 Task 并执行。

    使用多重继承：
    - ReActAgent: 提供 LLM 推理 + 工具调用能力
    - BaseAgent: 提供统一的 name/run 接口

    参数：
        tools: 此 Worker 可用的工具注册表
        client: 可复用的 Anthropic 客户端（共享可减少线程创建）
        specialization: 专业领域标识（calculation / coding / research）
        model: LLM 模型
        max_steps: 最大推理步数
    """

    def __init__(self, tools: ToolRegistry, client=None, specialization: str = "general",
                 model: str = "MiniMax-M2.7", max_steps: int = 10, system_rules: str = None):
        ReActAgent.__init__(self, tools=tools, client=client, model=model, max_steps=max_steps,
                           system_rules=system_rules)
        BaseAgent.__init__(self, name=f"worker_{specialization}")
        self.specialization = specialization

    def execute(self, task: Task, dep_results: dict[str, str] = None,
                memory=None) -> str:
        """执行一个 Task，注入依赖结果和长期记忆后运行 ReAct 循环。

        参数：
            task: Planner 产出的 Task 对象
            dep_results: 前置依赖任务的执行结果 {task_id: result}
            memory: AgentMemory 实例，用于检索历史上下文
        返回：
            ReAct 循环的最终文本回答
        """
        prompt_parts = [f"Task: {task.description}"]

        # 任务输入
        if task.input_data:
            prompt_parts.append(f"Input: {task.input_data}")

        # 长期记忆上下文（关键词 + 同类型标签两阶段检索）
        if memory is not None:
            mem_text = memory.to_context_text(keyword=task.description, task_type=task.type, limit=3)
            if mem_text:
                prompt_parts.append(mem_text)

        # 前置依赖任务结果
        if dep_results:
            dep_lines = ["\nPrevious task results:"]
            for dep_id, dep_result in dep_results.items():
                dep_lines.append(f"  [{dep_id}]: {dep_result}")
            prompt_parts.append("\n".join(dep_lines))

        prompt = "\n".join(prompt_parts)
        return self.run(prompt)

    def run(self, query: str) -> str:
        """实现 BaseAgent.run() 接口 —— 委托给 ReActAgent.run()。"""
        return ReActAgent.run(self, query)


# ==================== 专业 Worker 子类 ====================

class CalculatorWorker(WorkerAgent):
    """数学计算 Worker —— 擅长调用数学工具执行计算。"""

    def __init__(self, tools: ToolRegistry, client=None, model: str = "MiniMax-M2.7", system_rules: str = None):
        super().__init__(tools=tools, client=client, specialization="calculation", model=model,
                        system_rules=system_rules)


class CoderWorker(WorkerAgent):
    """编程 Worker —— 擅长代码生成、执行和审查。"""

    def __init__(self, tools: ToolRegistry, client=None, model: str = "MiniMax-M2.7", system_rules: str = None):
        super().__init__(tools=tools, client=client, specialization="coding", model=model,
                        system_rules=system_rules)


class ResearchWorker(WorkerAgent):
    """研究 Worker —— 擅长信息检索、分析和研究任务。"""

    def __init__(self, tools: ToolRegistry, client=None, model: str = "MiniMax-M2.7", system_rules: str = None):
        super().__init__(tools=tools, client=client, specialization="research", model=model,
                        system_rules=system_rules)


# ==================== Plan-and-Solve Worker ====================

class PlanAndSolveWorker(WorkerAgent):
    """Plan-and-Solve Worker —— 内置分步推理能力的 Worker。

    与普通 Worker 的区别：
    - 普通 Worker 收到 Task 后直接 ReAct 单链推理
    - P&S Worker 收到 Task 后先分解为子步骤，再按序执行，每步能看到前面所有结果

    适用场景：单个 Task 本身就很复杂，需要"先看清全貌再动手"的情况，
    比如多步数学推导、复杂逻辑推理、需要分阶段处理的编码任务。

    用法：
        pas_worker = PlanAndSolveWorker(tools=calc_tools, client=client)
        pas_worker.execute(task)  # 内部自动 Plan → Solve
    """

    def __init__(self, tools: ToolRegistry, client=None, model: str = "MiniMax-M2.7",
                 max_steps: int = 10, system_rules: str = None,
                 max_sub_steps: int = 5):
        super().__init__(tools=tools, client=client, specialization="plan_and_solve",
                        model=model, max_steps=max_steps, system_rules=system_rules)
        self.max_sub_steps = max_sub_steps  # 子步骤数量上限

    def execute(self, task: Task, dep_results: dict[str, str] = None,
                memory=None) -> str:
        """执行 Task —— 内部使用 Plan-and-Solve 两阶段推理。

        与父类的区别：不直接调用 self.run(prompt)，而是：
        1. 先为这个 Task 生成子步骤计划
        2. 按序执行每个子步骤（累积上下文）
        3. 汇总子步骤结果作为 Task 的最终输出
        """
        from agents.plan_and_solve import PlanAndSolveAgent

        # 构建 P&S 的 query：融合 Task 描述、前置结果、记忆
        query_parts = [task.description]
        if task.input_data:
            query_parts.append(f"输入数据: {task.input_data}")

        if dep_results:
            dep_lines = ["\n前置任务结果:"]
            for dep_id, dep_result in dep_results.items():
                dep_lines.append(f"  [{dep_id}]: {dep_result}")
            query_parts.append("\n".join(dep_lines))

        if memory is not None:
            mem_text = memory.to_context_text(keyword=task.description, task_type=task.type, limit=3)
            if mem_text:
                query_parts.append(mem_text)

        query = "\n".join(query_parts)

        # 使用 PlanAndSolveAgent 执行两阶段推理
        pas = PlanAndSolveAgent(
            tools=self.tools,
            client=self.client,
            model=self.model,
            max_steps=self.max_steps,
            system_rules=self.system_rules,
        )

        import logging
        logger = logging.getLogger("agents")
        logger.info("  [P&S Worker] 对 Task %s 启动 Plan-and-Solve", task.id)

        plan, answer = pas.run(query)
        logger.info("  [P&S Worker] Task %s P&S 完成, %d 个子步骤",
                     task.id, len(plan))

        return answer
