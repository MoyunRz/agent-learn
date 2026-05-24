"""
Tree of Thoughts (ToT) —— 思维树推理框架。

核心思想（出自 Yao et al., 2023）：
- 不只在一条链上推理，而是同时探索多条思路路径
- 每条路径通过「生成→评估→扩展→回溯」循环迭代
- 用 BFS（广度优先）在思维树中搜索最佳路径

与普通 ReAct 的区别：
- ReAct 是线性链，走一步看一步，容易掉进死胡同
- ToT 是分支树，同时探索 3~5 条路径，选择最优者继续扩展
"""

import time
import logging
from dataclasses import dataclass, field
from agents.react import ReActAgent
from agents.tools import ToolRegistry
from agents.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ThoughtNode:
    """思维树上的一个节点。

    path:       从根节点到当前节点的思维链（文字列表）
    score:      此节点的质量评分（0-10）
    depth:      当前深度（即第几步）
    evaluation: LLM 对这条思维的评估理由
    """
    path: list[str] = field(default_factory=list)     # 从根到当前节点的思考链
    score: float = 0                                     # 评分 0-10
    depth: int = 0                                       # 当前深度
    evaluation: str = ""                                  # 评估理由


class TreeOfThoughts:
    """Tree of Thoughts —— 思维树广度优先搜索。

    参数：
        tools: 工具注册表（传给内部的 ReActAgent）
        breadth: 每层保留的最优思维路径数（BFS 宽度）
        candidates: 每个节点扩展时生成的候选数
        max_depth: 最大搜索深度
        min_score: 保留节点的最低评分

    用法：
        tot = TreeOfThoughts(tools=registry, breadth=3, candidates=3, max_depth=5)
        best_path, score = tot.search("24点游戏: 用 4, 6, 8, 9 算出 24")
    """

    def __init__(self, tools: ToolRegistry, client=None, model: str = "MiniMax-M2.7",
                 breadth: int = 3, candidates: int = 3, max_depth: int = 5,
                 min_score: float = 5.0):
        self.tools = tools
        self.client = client
        self.model = model
        self.breadth = breadth       # BFS 每层保留 K 个最优节点
        self.candidates = candidates  # 每个节点生成 N 个候选扩展
        self.max_depth = max_depth
        self.min_score = min_score

    def search(self, problem: str, verbose: bool = False) -> tuple[list[str], float]:
        """BFS 搜索思维树，返回最佳路径和评分。

        返回：
            (best_path, best_score) — 最优思维链和最终评分
        """
        if verbose:
            logging.getLogger("agents").setLevel(logging.DEBUG)

        logger.info("=" * 60)
        logger.info("ToT search started | problem=%r", problem)
        logger.info("breadth=%d | candidates=%d | max_depth=%d",
                     self.breadth, self.candidates, self.max_depth)

        t_start = time.time()

        # 初始化：根节点为空
        current_layer = [ThoughtNode(path=[], score=10, depth=0)]
        best_solution = None
        best_score = 0

        for depth in range(1, self.max_depth + 1):
            logger.info("--- ToT depth %d/%d | current nodes: %d ---",
                        depth, self.max_depth, len(current_layer))

            # 对当前层每个节点，生成候选扩展
            all_candidates = []
            for node in current_layer:
                candidates = self._generate_thoughts(problem, node, self.candidates)
                all_candidates.extend(candidates)

            logger.info("  Generated %d candidates", len(all_candidates))

            if not all_candidates:
                logger.warning("  No candidates generated, stopping")
                break

            # 评估所有候选
            for node in all_candidates:
                score, evaluation = self._evaluate_thought(problem, node)
                node.score = score
                node.evaluation = evaluation

            # 按评分排序，取前 K 个
            all_candidates.sort(key=lambda n: n.score, reverse=True)
            current_layer = []
            for node in all_candidates:
                if node.score >= self.min_score and len(current_layer) < self.breadth:
                    current_layer.append(node)
                    logger.info("  Keep | score=%.1f depth=%d | %s...",
                                node.score, node.depth,
                                node.path[-1][:80] if node.path else "")

                # 检查是否找到解决方案（高评分的深度节点）
                if node.score > best_score:
                    best_score = node.score
                    best_solution = node

            logger.info("  Layer top scores: %s",
                        [f"{n.score:.1f}" for n in current_layer[:3]])

            if not current_layer:
                logger.info("  No nodes above threshold, stopping")
                break

            # 如果已找到高置信度解答，提前终止
            if best_score >= 9.0 and best_solution and best_solution.depth >= 2:
                logger.info("  High-confidence solution found, early stop")
                break

        elapsed = time.time() - t_start
        logger.info("ToT search complete | best_score=%.1f | duration=%.2fs",
                     best_score, elapsed)
        logger.info("=" * 60)

        if best_solution:
            return best_solution.path, best_solution.score
        return [], 0

    # ---------- 生成候选思维 ----------

    def _generate_thoughts(self, problem: str, parent: ThoughtNode, n: int) -> list[ThoughtNode]:
        """基于父节点路径，生成 n 个候选下一步思考。

        调用 LLM 生成多个不同的推理方向，每个方向成为一个新的 ThoughtNode。
        """
        # 已走过的路径
        path_text = "\n".join(
            f"  Step {i}: {s}" for i, s in enumerate(parent.path, 1)
        ) if parent.path else "  (starting point)"

        prompt = f"""Problem: {problem}

So far, the reasoning path has been:
{path_text}

Now generate {n} DIFFERENT possible next steps. Each step should explore a distinct
direction or strategy. Be creative and cover different angles.

Return exactly {n} candidates, each on a new line, prefixed with "CANDIDATE:" like this:
CANDIDATE: [description of the next reasoning step]
CANDIDATE: [description of a different reasoning step]
...

No other text."""

        agent = ReActAgent(tools=self.tools, client=self.client, model=self.model,
                          max_steps=5)
        response = agent.run(prompt)

        # 解析 CANDIDATE: 行
        import re
        candidates = re.findall(r'CANDIDATE:\s*(.+)', response, re.IGNORECASE)
        if not candidates:
            # 回退：按行取
            candidates = [line.strip() for line in response.strip().split("\n")
                         if line.strip() and not line.strip().startswith("#")]

        nodes = []
        for cand in candidates[:n]:
            new_path = parent.path + [cand.strip()]
            nodes.append(ThoughtNode(
                path=new_path,
                depth=parent.depth + 1,
            ))
        return nodes

    # ---------- 评估思维质量 ----------

    def _evaluate_thought(self, problem: str, node: ThoughtNode) -> tuple[float, str]:
        """评估一条思维路径的质量。

        让 LLM 评判这条路径的：
        - 逻辑性（是否合理）
        - 进展度（是否离答案更近）
        - 完整性（是否可能直接解决）
        """
        path_text = "\n".join(
            f"  Step {i}: {s}" for i, s in enumerate(node.path, 1)
        )

        prompt = f"""Problem: {problem}

Evaluate the following reasoning path:
{path_text}

Score this path on a scale of 0-10 based on:
- Logical correctness (reasonable and valid)
- Progress toward solution (getting closer)
- Completeness (could this solve the problem?)

Return your evaluation in this format:
SCORE: [0-10 number]
REASON: [brief explanation]"""

        agent = ReActAgent(tools=self.tools, client=self.client, model=self.model,
                          max_steps=3)
        response = agent.run(prompt)

        # 解析评分
        import re
        score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 5.0
        score = max(0, min(10, score))  # 限制在 0-10

        reason_match = re.search(r'REASON:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else response[:200]

        return score, reason
