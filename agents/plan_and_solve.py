"""
Plan-and-Solve (P&S) Agent —— 先规划、再求解的两阶段推理策略。

核心思想（出自 Wang et al., 2023）：
- Phase 1 "Plan":   将复杂问题分解为编号步骤列表
- Phase 2 "Solve":  按序执行每一步，前一步的输出作为后一步的上下文

与普通 ReAct 的区别：
- ReAct 是边想边做，LLM 每次只看到当前状态
- P&S 先看清全局再动手，每一步都知道完整的计划和已完成的结果
"""

import re
import time
import logging
from agents.react import ReActAgent
from agents.tools import ToolRegistry
from agents.logging_config import get_logger

logger = get_logger(__name__)


class PlanAndSolveAgent:
    """Plan-and-Solve Agent —— 先规划后执行的推理引擎。

    用法：
        agent = PlanAndSolveAgent(tools=tool_registry)
        plan, answer = agent.run("解方程 3x + 7 = 22")
    """

    def __init__(self, tools: ToolRegistry, client=None, model: str = "MiniMax-M2.7",
                 max_steps: int = 10, system_rules: str = None):
        self.tools = tools
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.system_rules = system_rules

    def run(self, query: str, verbose: bool = False) -> tuple[list[str], str]:
        """执行 P&S 两阶段推理。

        返回:
            (plan_steps, final_answer) — 规划步骤列表和最终答案
        """
        if verbose:
            logging.getLogger("agents").setLevel(logging.DEBUG)

        logger.info("=" * 60)
        logger.info("P&S started | query=%r", query)

        # =========== Phase 2: Solve ===========
        logger.info("--- Phase 1: Plan ---")
        plan = self._generate_plan(query)
        logger.info("Plan: %d steps", len(plan))
        for i, step in enumerate(plan, 1):
            logger.info("  Step %d: %s", i, step)

        # =========== Phase 2: Solve ===========
        logger.info("--- Phase 2: Solve ---")
        answer = self._execute_plan(query, plan)

        logger.info("P&S complete")
        logger.info("=" * 60)
        return plan, answer

    # ---------- Phase 1: 生成计划 ----------

    def _generate_plan(self, query: str) -> list[str]:
        """调用 LLM 生成编号的计划步骤列表。

        不需要工具 —— 纯文本推理即可产出计划。
        """
        prompt = f"""You are a planning assistant. Break down the following problem into a clear,
numbered list of steps. Each step should be a single action or sub-problem.

Problem: {query}

Return ONLY the numbered steps, one per line, like this:
1. [step description]
2. [step description]
...

No extra text before or after the steps."""

        agent = ReActAgent(tools=self.tools, client=self.client, model=self.model,
                          max_steps=3)
        if verbose := logging.getLogger("agents").level <= logging.DEBUG:
            pass  # 跟随全局日志级别
        response = agent.run(prompt)

        # 解析编号步骤（1. / 2. / Step 1: 等多种格式）
        steps = re.findall(r'(?:^|\n)\s*(?:\d+[\.\)]|Step\s*\d+[:\-])\s*(.+)', response)
        if not steps:
            # 回退：按行拆分
            steps = [line.strip() for line in response.strip().split("\n") if line.strip()]
        return [s.strip() for s in steps]

    # ---------- Phase 2: 执行计划 ----------

    def _execute_plan(self, query: str, plan: list[str]) -> str:
        """按序执行每个步骤，累积上下文。

        每一步的 prompt 包含：
        - 原始问题
        - 完整计划
        - 已完成步骤及结果
        - 当前要执行的步骤
        """
        history = []  # [(step_desc, result), ...]

        for i, step_desc in enumerate(plan, 0):
            step_num = i + 1
            logger.info("--- Step %d/%d: %s ---", step_num, len(plan), step_desc)

            # 构建累积上下文的 prompt
            prompt = self._build_step_prompt(query, plan, history, step_num, step_desc)

            agent = ReActAgent(tools=self.tools, client=self.client, model=self.model,
                              max_steps=self.max_steps, system_rules=self.system_rules)
            result = agent.run(prompt)

            history.append((step_desc, result))
            logger.info("Step %d result: %s", step_num, result[:200])

        # 最终汇总
        return self._synthesize(query, plan, history)

    def _build_step_prompt(self, query, plan, history, step_num, step_desc):
        """为当前步骤构建包含完整上下文的 prompt。"""
        # 已完成步骤的汇总
        done_text = ""
        if history:
            done_text = "\nCompleted steps and their results:\n"
            for j, (s, r) in enumerate(history, 1):
                done_text += f"  Step {j} [{s}]: {r}\n"

        # 剩余步骤
        remaining = plan[step_num:]  # 当前步骤之后的
        remaining_text = ""
        if remaining:
            remaining_text = "\nRemaining steps (do NOT execute these yet):\n"
            for j, s_desc in enumerate(remaining, step_num + 1):
                remaining_text += f"  Step {j}: {s_desc}\n"

        return f"""Original problem: {query}

Full plan:
""" + "\n".join(f"  Step {j+1}: {s}" for j, s in enumerate(plan)) + f"""

{done_text}
Now execute ONLY Step {step_num}: {step_desc}

{remaining_text}
Execute this step now. Use any tools available to you. Return the result concisely."""

    def _synthesize(self, query, plan, history):
        """汇总所有步骤结果，生成最终答案。"""
        results_text = ""
        for j, (s, r) in enumerate(history, 1):
            results_text += f"  Step {j} [{s}]: {r}\n"

        prompt = f"""Original problem: {query}

Plan executed:
{results_text}
Based on all the step results above, provide the final answer to the original problem.
Be concise and direct."""

        agent = ReActAgent(tools=self.tools, client=self.client, model=self.model,
                          max_steps=3)
        return agent.run(prompt)
