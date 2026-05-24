"""
PlannerAgent —— 任务规划 Agent，负责将复杂用户查询分解为结构化的子任务。

工作流程：
1. 接收用户查询
2. 构造一个详细的 prompt，要求 LLM 输出 JSON 格式的执行计划
3. 通过内置的 ReActAgent 调用 LLM
4. 解析 JSON 响应，构建 ExecutionPlan
5. 如果 JSON 解析失败，回退为单个通用任务
"""

from agents.react import ReActAgent
from agents.tools import ToolRegistry, Tool
from .base import BaseAgent
from ..message import ExecutionPlan, Task
import json
import re


class PlannerAgent(BaseAgent):
    """任务规划 Agent —— 将用户查询分解为有序的子任务列表。

    内部使用 ReActAgent 与 LLM 交互，通过精心设计的 prompt 让 LLM
    输出符合 ExecutionPlan 格式的 JSON。

    参数：
        client: Anthropic SDK 客户端
        model: 使用的 LLM 模型
    """

    def __init__(self, client=None, model="MiniMax-M2.7"):
        super().__init__(name="planner")
        self.client = client
        self.model = model
        self.tools = ToolRegistry()
        self._register_planning_tools()

    def _register_planning_tools(self):
        """注册 Planner 专用的工具。

        这两个工具是「存根」—— 它们不在 LLM 侧被真正调用，
        而是作为 prompt 中的 schema 参考，引导 LLM 理解任务分解的格式。
        """
        self.tools.register(
            Tool(
                func=self._decompose_func,
                name="decompose_query",
                description="Breaks down a complex query into structured sub-tasks",
            )
        )
        self.tools.register(
            Tool(
                func=self._classify_func,
                name="classify_task",
                description="Classifies a task type: calculation, coding, research, general",
            )
        )

    def _decompose_func(self, query: str) -> str:
        """存根函数 —— LLM 不会真正调用它，仅用作 schema 参考。"""
        return json.dumps({"subtasks": []})

    def _classify_func(self, task_description: str) -> str:
        """存根函数 —— LLM 不会真正调用它，仅用作 schema 参考。"""
        return "general"

    def run(self, query: str, verbose: bool = False) -> ExecutionPlan:
        """执行规划 —— 分解查询并返回 ExecutionPlan。

        参数：
            query: 用户的原始查询
            verbose: 是否打印内部 ReActAgent 的详细执行日志
        返回：
            包含子任务列表和依赖关系的 ExecutionPlan
        """
        import logging
        if verbose:
            logging.getLogger("agents").setLevel(logging.DEBUG)

        # 创建内部 ReActAgent 来调用 LLM
        planning_agent = ReActAgent(tools=self.tools, client=self.client, model=self.model)
        planning_prompt = self._build_plan_prompt(query)
        response = planning_agent.run(planning_prompt, verbose=verbose)
        return self._parse_plan_response(response)

    def _build_plan_prompt(self, query: str) -> str:
        """构造规划 prompt —— 要求 LLM 以 JSON 格式返回子任务列表。

        prompt 中明确了 JSON 的字段名和类型，并用示例引导 LLM 输出正确格式。
        """
        return f"""You are a task planner. Analyze the following query and create an execution plan.

Query: {query}

Break down the query into sub-tasks. For each sub-task, specify:
1. id: a unique identifier (e.g., "task_1")
2. type: one of [calculation, coding, research, general]
3. description: what the task does
4. input_data: what information the worker needs (can be empty dict {{}})
5. expected_output: description of expected result
6. dependencies: list of task_ids that must complete before this task (empty list if none)

Return your plan in this JSON format (no other text):
{{
  "tasks": [
    {{"id": "task_1", "type": "calculation", "description": "Calculate 3+5", "input_data": {{}}, "expected_output": "8", "dependencies": []}}
  ]
}}
"""

    def _parse_plan_response(self, llm_response: str) -> ExecutionPlan:
        """解析 LLM 的 JSON 响应，构建 ExecutionPlan。

        使用正则提取 JSON 块（容错处理），解析失败时回退为单个通用任务。
        这保证了即使 LLM 输出格式不完美，系统也能继续运行。
        """
        plan = ExecutionPlan()
        try:
            # 用正则提取第一个完整的 JSON 对象（处理 LLM 多输出额外文字的情况）
            json_match = re.search(r"\{[\s\S]*\}", llm_response)
            if json_match:
                data = json.loads(json_match.group())
                for task_data in data.get("tasks", []):
                    task = Task(
                        id=task_data["id"],
                        type=task_data["type"],
                        description=task_data["description"],
                        input_data=task_data.get("input_data", {}),
                        expected_output=task_data.get("expected_output", ""),
                        dependencies=task_data.get("dependencies", []),
                    )
                    plan.add_task(task, depends_on=task_data.get("dependencies", []))
                return plan
        except (json.JSONDecodeError, KeyError):
            pass  # 解析失败 → 走回退逻辑

        # 回退：将整个 LLM 响应作为单个通用任务
        plan.add_task(Task(
            id="task_1",
            type="general",
            description=llm_response,
            input_data={},
            expected_output="",
            dependencies=[],
        ))
        return plan
