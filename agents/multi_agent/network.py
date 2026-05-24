"""
Multi-Agent Networks —— 去中心化的多 Agent 通信网络。

核心思想：
- 无中央协调者（与 Supervisor 模式不同）
- Agent 之间通过有向边连接，消息沿边传递
- 每个 Agent 有自己的角色（role），从角色视角处理消息
- 支持链式、辩论、广播三种执行模式

与 Supervisor 模式的区别：
- Supervisor: Planner → Supervisor → Workers（星型，中心调度）
- Network: Agent ↔ Agent（P2P，消息驱动）
"""

import time
import logging
from dataclasses import dataclass, field
from collections import deque

from agents.tools import ToolRegistry
from .agents.worker import WorkerAgent
from agents.logging_config import get_logger

logger = get_logger(__name__)


# ==================== NetworkMessage ====================

@dataclass
class NetworkMessage:
    """网络消息 —— 在 AgentNetwork 节点间传递的消息单元。

    字段：
        sender:    发送方 Agent 名称
        receiver:  接收方 Agent 名称（"all" 表示广播给所有邻居）
        content:   消息文本
        round_num: 轮次编号（辩论模式从 0 开始）
        msg_type:  类型标记
          - "query"    初始查询
          - "response" 一般响应
          - "opinion"  观点（辩论模式）
          - "final"    最终答案
        timestamp: 时间戳
    """
    sender: str
    receiver: str
    content: str
    round_num: int = 0
    msg_type: str = "response"
    timestamp: float = field(default_factory=time.time)


# ==================== NetworkAgent ====================

class NetworkAgent(WorkerAgent):
    """网络 Agent 节点 —— 在 AgentNetwork 中以角色身份参与 P2P 协作。

    继承 WorkerAgent，复用：
    - ReActAgent 的 LLM 推理 + 工具调用能力
    - BaseAgent 的命名和接口

    新增：
    - role: 角色描述（如"市场分析师"），注入每次 LLM 调用的 prompt
    - neighbors: 可转发消息的邻居名称列表
    - conversation_history: 当前会话的消息历史

    参数：
        tools: 工具注册表
        client: Anthropic SDK 客户端
        role: 角色描述文本（决定 Agent 的回复视角）
        name: 节点名称（默认从 role 截取）
        model: LLM 模型
        max_steps: 最大推理步数
        system_rules: 系统级约束规则
    """

    def __init__(
        self,
        tools: ToolRegistry,
        client=None,
        role: str = "通用助手",
        name: str | None = None,
        model: str = "MiniMax-M2.7",
        max_steps: int = 10,
        system_rules: str | None = None,
    ):
        # 用 role 的一部分作为 specialization
        spec = role[:12].replace(" ", "_").replace("，", "").replace("、", "")
        super().__init__(
            tools=tools, client=client, specialization=spec,
            model=model, max_steps=max_steps, system_rules=system_rules,
        )
        # 覆盖 BaseAgent 的名称
        self.name = name or f"agent_{spec}"
        self.role = role
        self.neighbors: list[str] = []          # 可转发目标
        self.conversation_history: list[dict] = []  # 会话记录

    # ---------- 核心方法：接收消息 ----------

    def receive(self, message: NetworkMessage) -> str:
        """接收并处理一条网络消息，返回回答文本。

        流程（全部同步，Agent 本身不做任何编排决策）：
        ① 记录消息到 conversation_history（role=发送方名称）
        ② _build_network_prompt() 组装 prompt：
           - 角色描述（你扮演的角色：xxx）
           - 系统约束规则（如有）
           - 最近 5 条对话历史（每条截断至 200 字，防上下文溢出）
           - 当前消息
        ③ 调用父类 ReActAgent.run(prompt) → LLM 推理
           （思考 → 可选工具调用 → 最终文本回答）
        ④ 将回答追加到 conversation_history（role=自己名称）
        ⑤ 返回回答字符串

        关键设计：
        - Agent 完全被动：收到什么消息就回什么，不感知轮次、不决定下一
          步该谁发言。所有编排逻辑在 AgentNetwork 的 run_xxx 方法中。
        - 返回值是纯字符串，调用方（AgentNetwork）用它构建下一轮 prompt
          或存入 all_opinions。

        参数：
            message: 收到的网络消息
        返回：
            Agent 的回答文本（即 LLM 的最终文本输出）
        """
        logger.info("[%s] 收到消息 | from=%s | round=%d | type=%s",
                     self.name, message.sender, message.round_num, message.msg_type)

        # ① 记录接收到的消息到对话历史
        self.conversation_history.append({
            "role": message.sender,
            "content": f"[{message.msg_type}] {message.content}",
        })

        # ②③ 构建 prompt 并调用 LLM 推理
        prompt = self._build_network_prompt(message)
        response = ReActAgent.run(self, prompt)   # 继承自 WorkerAgent → ReActAgent

        # ④ 记录自己的回答到对话历史
        self.conversation_history.append({
            "role": self.name,
            "content": response,
        })

        # ⑤ 返回纯字符串给调用方
        logger.info("[%s] 回应 (%.100s...)", self.name, response)
        return response

    def _build_network_prompt(self, message: NetworkMessage) -> str:
        """构建网络对话 prompt —— 融合角色、规则、历史和当前消息。

        格式：
            [角色描述]
            [系统规则]
            [最近对话历史]
            [当前消息]
        """
        parts = [f"你扮演的角色：{self.role}"]

        # 系统约束规则
        if self.system_rules:
            parts.append(f"\n{self.system_rules}")

        # 最近对话历史（防止上下文溢出，只取最近 5 条）
        if len(self.conversation_history) > 1:
            recent = self.conversation_history[:-1][-5:]  # 排除刚追加的当前消息
            if recent:
                parts.append("\n--- 对话历史 ---")
                for entry in recent:
                    parts.append(f"[{entry['role']}]: {entry['content'][:200]}")

        # 当前消息
        parts.append(f"\n--- 当前消息 ---")
        parts.append(f"来自: {message.sender}")
        parts.append(f"内容: {message.content}")
        parts.append(f"\n请作为「{self.role}」给出你的分析和回答。")

        return "\n".join(parts)

    def has_neighbors(self) -> bool:
        """是否有可转发的邻居。"""
        return len(self.neighbors) > 0

    def clear_history(self):
        """清空当前会话的对话历史。"""
        self.conversation_history.clear()


# ==================== AgentNetwork ====================

class AgentNetwork:
    """AgentNetwork —— 去中心化多 Agent 通信网络的拓扑管理器。

    管理节点注册、有向边连接、以及三种执行模式：
    - run_chain:   链式传递（A→B→C）
    - run_debate:  辩论模式（全连接多轮讨论）
    - run_broadcast: 广播模式（BFS 传播到所有节点）

    用法：
        network = AgentNetwork()
        network.add_agent(analyst)
        network.add_agent(critic)
        network.connect("analyst", "critic")
        result = network.run_chain("analyst", "分析市场趋势")
    """

    def __init__(self):
        self.agents: dict[str, NetworkAgent] = {}       # name → agent
        self.adjacency: dict[str, list[str]] = {}       # name → [neighbor_names]

    # ---------- 拓扑管理 ----------

    def add_agent(self, agent: NetworkAgent) -> str:
        """注册 Agent 节点到网络。

        内部做了三件事：
        ① 以 agent.name 为键存入 self.agents 字典（同名会覆盖）
        ② 在 self.adjacency 中为该节点初始化空的邻接表 []
        ③ 同步 agent.neighbors，让节点自己也知道当前有哪些邻居

        注意：注册后节点是孤立的，需要通过 connect() 建立有向边才能参与
        链式和广播模式。辩论模式不受邻接表影响——它会遍历 self.agents 中
        所有节点，与邻接表无关。

        参数：
            agent: NetworkAgent 实例
        返回：
            agent.name（可用于后续 connect 调用）
        """
        # ① 以 name 为键存入字典（同名节点会覆盖旧节点）
        self.agents[agent.name] = agent
        # ② 初始化该节点的邻接表（空列表，暂无邻居）
        if agent.name not in self.adjacency:
            self.adjacency[agent.name] = []
        # ③ 同步 agent 的 neighbors 列表（此时为空，待 connect 后更新）
        agent.neighbors = list(self.adjacency[agent.name])
        logger.info("网络添加节点: %s (role=%s)", agent.name, agent.role)
        return agent.name

    def connect(self, from_name: str, to_name: str):
        """创建有向边 from_name → to_name。

        同时更新 from_name 节点的 neighbors 列表。
        参数：
            from_name: 发送方节点名称
            to_name: 接收方节点名称
        异常：
            KeyError: 节点名称不存在
        """
        if from_name not in self.agents:
            raise KeyError(f"节点 '{from_name}' 不在网络中")
        if to_name not in self.agents:
            raise KeyError(f"节点 '{to_name}' 不在网络中")

        if to_name not in self.adjacency[from_name]:
            self.adjacency[from_name].append(to_name)
        # 同步 agent 的 neighbors
        self.agents[from_name].neighbors = list(self.adjacency[from_name])
        logger.info("网络连接: %s → %s", from_name, to_name)

    def disconnect(self, from_name: str, to_name: str):
        """移除有向边 from_name → to_name。"""
        if from_name in self.adjacency and to_name in self.adjacency[from_name]:
            self.adjacency[from_name].remove(to_name)
            self.agents[from_name].neighbors = list(self.adjacency[from_name])
            logger.info("网络断开: %s → %s", from_name, to_name)

    # ---------- 执行模式 1: 链式 ----------

    def run_chain(self, start_agent: str, query: str,
                  max_hops: int = 10, verbose: bool = False) -> str:
        """链式执行 —— 查询沿单向链依次传递，每步精炼。

        流程：
        1. 从 start_agent 开始，沿 adjacency 链走到尽头
        2. 第一个节点接收原始 query
        3. 每个节点将自己的 response 转发给下一个邻居
        4. 最后一个节点的 response 作为最终结果

        参数：
            start_agent: 起始节点名称
            query: 用户查询
            max_hops: 最大跳数（防止环）
            verbose: 是否开启详细日志
        返回：
            链式执行的最终结果
        """
        if verbose:
            logging.getLogger("agents").setLevel(logging.DEBUG)

        # 确定传递路径
        path = self._get_chain_path(start_agent)
        if not path:
            raise ValueError(f"节点 '{start_agent}' 不在网络中或没有出边")

        logger.info("=" * 60)
        logger.info("链式执行开始 | 路径: %s", " → ".join(path))
        logger.info("=" * 60)

        # 清空所有节点的历史
        for name in path:
            self.agents[name].clear_history()

        # 第 1 跳：起始节点接收原始查询
        first = self.agents[path[0]]
        msg = NetworkMessage(
            sender="user", receiver=first.name,
            content=query, round_num=0, msg_type="query",
        )
        current_result = first.receive(msg)

        # 后续跳：每个节点接收上一个节点的输出
        for i in range(1, min(len(path), max_hops + 1)):
            sender = self.agents[path[i - 1]]
            receiver = self.agents[path[i]]

            fwd_msg = NetworkMessage(
                sender=sender.name, receiver=receiver.name,
                content=f"上一个节点的分析结果:\n{current_result}",
                round_num=i, msg_type="response",
            )
            current_result = receiver.receive(fwd_msg)

        logger.info("链式执行完成 | 最终节点: %s", path[-1])
        return current_result

    def _get_chain_path(self, start: str) -> list[str]:
        """沿 adjacency 单向链走到底，返回有序节点列表。

        用于 run_chain 确定传递路径。碰到分支时取第一个邻居。
        """
        if start not in self.adjacency:
            return []
        path = [start]
        current = start
        visited = {current}
        while self.adjacency.get(current):
            next_nodes = [n for n in self.adjacency[current] if n not in visited]
            if not next_nodes:
                break
            current = next_nodes[0]
            visited.add(current)
            path.append(current)
        return path

    # ---------- 执行模式 2: 辩论 ----------

    def run_debate(self, query: str, rounds: int = 2,
                   moderator: str | None = None,
                   verbose: bool = False) -> str:
        """辩论模式 —— 所有 Agent 独立发表观点，多轮讨论后由 moderator 汇总。

        === 数据结构 ===

        all_opinions: dict[int, dict[str, str]]
        键 = 轮次号，值 = {agent_name → opinion_text}
        示例：
            all_opinions = {
                0: {"analyst": "远程办公会成为主流...", "critic": "远程办公面临挑战...",
                    "economist": "从经济学角度..."},
                1: {"analyst": "综合考虑critic的担忧...", "critic": "承认技术进步但...",
                    "economist": "成本模型显示..."},
                2: {...},
            }

        === 执行流程 ===

        Round 0 — 初始观点收集:
            对 agent_names 中的每个 Agent 逐一调用 receive(msg_type="query")
            → 返回值存入临时 dict opinions[name] = opinion_text
            → 整个 dict 赋值给 all_opinions[0]

        Round 1..N — 辩论回合:
            对于每一轮 round_num：
            a) 取出上一轮观点 prev_opinions = all_opinions[round_num - 1]
            b) 对每个 Agent：
               - 从 prev_opinions 中取出「其他人的观点」（排除自己）
               - 拼接为带角色标签的 digest：【角色名】观点内容
               - 构造 debate_prompt：原始问题 + 他人观点 + 要求更新
               - 调用 receive()，返回值存入 new_opinions[name]
            c) all_opinions[round_num] = new_opinions

        === 关键设计决策 ===

        ① 编排是程序化的，不是 AI 决定的：
           三个 for 循环（Round 0 → Round 1..N → 汇总）完全硬编码在 run_debate
           中。Agent 不知道自己在第几轮，也不知道谁该下一步发言——这些都是
           AgentNetwork 外部编排的。

        ② 辩论忽略邻接表：
           遍历的是 self.agents 中所有节点（agent_names = list(self.agents.keys())），
           而非邻接表。即使 connect 了某些节点，辩论也会调用全部 Agent。
           adjacency 只对 run_chain 和 run_broadcast 有意义。

        ③ Agent 每轮只看到上一轮的观点，不是实时对话：
           第 1 轮：Agent 看到 Round 0 其他 Agent 的初始观点
           第 2 轮：Agent 看到 Round 1 其他 Agent 更新后的观点
           以此类推——这是"同步回合制"，不是实时多轮对话。

        ④ Moderator 不参与辩论：
           moderator 只出现在最终汇总阶段，之前不在 agent_names 的遍历中。
           注意：如果 moderator 也在 agent_names 中（即也是参与者），它会被
           跳过——汇总时单独调用 receive()，在此之前 conversation_history 为空。

        ⑤ 每个节点唯一：
           self.agents 是 dict[str, NetworkAgent]，以 name 为键。同名节点
           会被覆盖，所以不存在多个同名 analyst。

        ⑥ 防上下文溢出：
           receive() → _build_network_prompt() 只取最近 5 条历史，每条
           截断至 200 字符。

        参数：
            query: 辩论问题
            rounds: 讨论轮次（不含 Round 0，默认 2）
            moderator: 汇总 Agent 的名称（可选，不传则直接拼接所有最终观点）
            verbose: 是否开启详细日志
        返回：
            辩论的最终结果（moderator 的综合结论，或所有观点的拼接文本）
        """
        if verbose:
            logging.getLogger("agents").setLevel(logging.DEBUG)

        # 获取所有注册节点名称列表（dict keys 的顺序就是 Agent 被调用的顺序）
        agent_names = list(self.agents.keys())
        if not agent_names:
            raise ValueError("网络中没有任何 Agent")

        logger.info("=" * 60)
        logger.info("辩论开始 | 问题: %s | 轮次: %d", query, rounds)
        logger.info("参与者: %s", ", ".join(agent_names))
        logger.info("=" * 60)

        # 清空所有 Agent 的对话历史（确保每场辩论从干净状态开始）
        for name in agent_names:
            self.agents[name].clear_history()

        # all_opinions 结构: {轮次号: {agent_name → opinion_text}}
        # 所有轮次的数据都在这里，方便后续追溯每一轮的发言
        all_opinions: dict[int, dict[str, str]] = {}

        # ═══════════════════ Round 0: 初始观点收集 ═══════════════════
        # 每个 Agent 独立收到同一个 query，各自给出初始分析
        round_num = 0
        opinions = {}                         # 本轮临时 dict
        for name in agent_names:              # 按 agent_names 顺序逐一调用
            agent = self.agents[name]
            msg = NetworkMessage(
                sender="user", receiver=name, content=query,
                round_num=round_num, msg_type="query",
            )
            opinion = agent.receive(msg)      # → LLM 推理 → 返回文本
            opinions[name] = opinion          # 存入: "analyst" → "我认为..."

        # 整轮观点存到 all_opinions[0]
        # 例: {0: {"analyst": "...", "critic": "...", "economist": "..."}}
        all_opinions[round_num] = opinions
        logger.info("Round 0 完成 | 收集 %d 个初始观点", len(opinions))

        # ═══════════════════ Round 1..N: 辩论回合 ═══════════════════
        # 每一轮：每个 Agent 看到上一轮其他人的观点，更新自己的观点
        for round_num in range(1, rounds + 1):
            prev_opinions = all_opinions[round_num - 1]   # 取上一轮所有观点
            new_opinions = {}                              # 本轮新观点

            for name in agent_names:          # 每个 Agent 逐一收到辩论 prompt
                agent = self.agents[name]

                # 组装「其他人的观点」— 排除该 Agent 自己上一轮的发言
                others_parts = []
                for n, o in prev_opinions.items():
                    if n != name:             # 排除自己
                        role = self.agents[n].role
                        others_parts.append(f"【{role}】{o}")
                others_digest = "\n\n".join(others_parts)

                # 构造辩论 prompt：告知原始问题 + 他人观点 + 要求更新
                debate_prompt = (
                    f"原始问题: {query}\n\n"
                    f"其他参与者的第 {round_num} 轮观点:\n{others_digest}\n\n"
                    f"请作为「{agent.role}」吸收他人观点中的合理部分，"
                    f"修正你的观点，给出更新后的分析。"
                )
                msg = NetworkMessage(
                    sender="moderator", receiver=name,
                    content=debate_prompt,
                    round_num=round_num, msg_type="opinion",
                )
                new_opinion = agent.receive(msg)   # → LLM 推理 → 返回更新后观点
                new_opinions[name] = new_opinion

            # 本轮观点存入 all_opinions[round_num]
            all_opinions[round_num] = new_opinions
            logger.info("Round %d 完成 | 收集 %d 个观点", round_num, len(new_opinions))

        # ═══════════════════ 最终汇总 ═══════════════════
        final_opinions = all_opinions[rounds]        # 取最后一轮的观点

        if moderator and moderator in self.agents:
            # 情况 A：有 moderator —— 把所有最终观点发给它做综合
            mod = self.agents[moderator]
            parts = []
            for n, o in final_opinions.items():
                parts.append(f"【{self.agents[n].role}（{n}）】\n{o}")
            digest = "\n\n".join(parts)

            final_prompt = (
                f"原始问题: {query}\n\n"
                f"经过 {rounds} 轮辩论后，所有参与者的最终观点:\n\n{digest}\n\n"
                f"请作为「{mod.role}」综合以上所有观点，权衡各方立场，给出最终结论。"
            )
            msg = NetworkMessage(
                sender="moderator", receiver=moderator,
                content=final_prompt,
                round_num=rounds + 1, msg_type="opinion",
            )
            final_result = mod.receive(msg)
            logger.info("辩论完成 | 由 moderator '%s' 汇总", moderator)
            return final_result
        else:
            # 情况 B：无 moderator —— 直接把所有最终观点用分隔线拼起来
            output = []
            for name, opinion in final_opinions.items():
                role = self.agents[name].role
                output.append(f"【{role}（{name}）】\n{opinion}")
            logger.info("辩论完成 | 直接拼接 %d 个最终观点", len(final_opinions))
            return "\n\n" + "=" * 40 + "\n\n".join(output)

    # ---------- 执行模式 3: 广播 ----------

    def run_broadcast(self, start_agent: str, query: str,
                      verbose: bool = False) -> dict[str, str]:
        """广播模式 —— 从起始节点 BFS 遍历所有可达节点。

        流程：
        1. 起始节点接收 query 并处理
        2. 起始节点的 response 转发给所有邻居
        3. 每个邻居处理后转发给各自的邻居（跳过已访问的）
        4. 收集所有节点的 response

        参数：
            start_agent: 广播的起始节点
            query: 用户查询
            verbose: 是否开启详细日志
        返回：
            {agent_name: response_text} 字典
        """
        if verbose:
            logging.getLogger("agents").setLevel(logging.DEBUG)

        if start_agent not in self.agents:
            raise ValueError(f"节点 '{start_agent}' 不在网络中")

        # 清空所有可达节点的历史
        for name in self.agents:
            self.agents[name].clear_history()

        visited: set[str] = set()
        queue: deque[tuple[str, str]] = deque()  # (name, incoming_content)
        results: dict[str, str] = {}

        queue.append((start_agent, query))
        # 记录前驱，用于日志
        predecessor: dict[str, str] = {start_agent: "user"}

        logger.info("=" * 60)
        logger.info("广播开始 | 起点: %s", start_agent)
        logger.info("=" * 60)

        while queue:
            name, content = queue.popleft()
            if name in visited:
                continue
            visited.add(name)

            agent = self.agents[name]
            sender = predecessor.get(name, "user")

            msg = NetworkMessage(
                sender=sender, receiver=name,
                content=content,
                round_num=len(visited) - 1, msg_type="query",
            )
            response = agent.receive(msg)
            results[name] = response

            # 转发给未访问的邻居
            for neighbor in self.adjacency.get(name, []):
                if neighbor not in visited:
                    queue.append((neighbor, response))
                    if neighbor not in predecessor:
                        predecessor[neighbor] = name

        logger.info("广播完成 | 覆盖 %d 个节点", len(results))
        return results
