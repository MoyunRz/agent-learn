"""
Multi-Agent Networks Demo —— 去中心化多 Agent 协作演示。

演示内容：
- Demo 1: 链式网络 —— 分析师 → 评审专家 → 汇总专家，渐进式分析
- Demo 2: 辩论网络 —— 3 个不同视角 Agent 多轮讨论后由 moderator 汇总

运行：
    python demo_network.py
"""

from dotenv import load_dotenv
import os
import anthropic
from agents.multi_agent.network import AgentNetwork, NetworkAgent, NetworkMessage
from agents.tools import ToolRegistry, Tool
from agents.logging_config import setup_logging

load_dotenv()
setup_logging()

# ---- 共享工具 ----
def web_search(query: str) -> str:
    """模拟网络搜索"""
    return f"搜索结果: 关于「{query}」的最新报告显示该领域增长迅速，但存在政策不确定性和供应链风险。"

tools = ToolRegistry()
tools.register(Tool(web_search, name="web_search", description="搜索网络获取最新信息"))

# ---- 共享客户端 ----
client = anthropic.Anthropic(
    base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic"),
    api_key=os.getenv("MINIMAX_API_KEY", "your-key"),
)

system_rules = """约束规则：
1. 用中文回答
2. 基于你的角色视角进行分析
3. 简洁有条理，不超过200字
4. 如果有数据支撑，引用具体数字"""


def build_network():
    """构建一个基础网络，包含 4 个 Agent。"""
    network = AgentNetwork()

    # 分析师 —— 擅长数据驱动分析
    analyst = NetworkAgent(
        tools=tools, client=client,
        role="市场分析师，擅长数据分析和趋势预测",
        name="analyst",
        system_rules=system_rules,
    )
    network.add_agent(analyst)

    # 评审专家 —— 专门找盲点和漏洞
    critic = NetworkAgent(
        tools=tools, client=client,
        role="批判性评审专家，专门发现分析中的盲点、逻辑漏洞和未考虑的变量",
        name="critic",
        system_rules=system_rules,
    )
    network.add_agent(critic)

    # 汇总专家 —— 融合多方观点
    synthesizer = NetworkAgent(
        tools=tools, client=client,
        role="综合汇总专家，擅长融合多方观点形成结构化最终报告",
        name="synthesizer",
        system_rules=system_rules,
    )
    network.add_agent(synthesizer)

    # 经济学者 —— 从成本效益角度分析
    economist = NetworkAgent(
        tools=tools, client=client,
        role="经济学研究员，关注成本效益、生产效率和市场均衡",
        name="economist",
        system_rules=system_rules,
    )
    network.add_agent(economist)

    return network


if __name__ == "__main__":
    # ============================================================
    # Demo 1: 链式网络 —— 三步递进分析
    # ============================================================
    print("=" * 70)
    print("Demo 1: 链式网络 ——— 分析师 → 评审专家 → 汇总专家")
    print("=" * 70)

    network1 = build_network()
    # 链式拓扑: analyst → critic → synthesizer
    network1.connect("analyst", "critic")
    network1.connect("critic", "synthesizer")

    print("\n拓扑结构:")
    print("  用户查询 → analyst(市场分析师) → critic(评审专家) → synthesizer(汇总专家) → 结果")
    print()

    query1 = "分析新能源汽车市场未来3年的发展趋势和投资机会"
    print(f"用户问题: {query1}\n")

    result = network1.run_chain("analyst", query1, verbose=True)

    print("\n" + "=" * 70)
    print("链式网络最终输出:")
    print("=" * 70)
    print(result)

    # ============================================================
    # Demo 2: 辩论网络 —— 多视角碰撞
    # ============================================================
    print("\n\n" + "=" * 70)
    print("Demo 2: 辩论网络 ——— 分析师 vs 经济学家 vs 评审专家")
    print("=" * 70)

    network2 = build_network()

    # 全连接拓扑（每个 Agent 都能看到其他所有人的观点）
    participants = ["analyst", "critic", "economist"]
    for a in participants:
        for b in participants:
            if a != b:
                network2.connect(a, b)

    # synthesizer 作为 moderator（接收所有最终观点做综合）
    # moderator 不在辩论中，所以不需要反向连接

    print("\n拓扑结构:")
    print("  analyst ←→ critic ←→ economist")
    print("    ↑_____________________↑")
    print("    所有最终观点 → synthesizer(汇总)")
    print()

    query2 = "远程办公是否会成为未来主流工作模式？请从技术、管理、经济三个维度分析"
    print(f"辩论问题: {query2}\n")

    result2 = network2.run_debate(
        query=query2,
        rounds=2,            # 2 轮讨论
        moderator="synthesizer",
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("辩论网络最终输出（经 moderator 综合）:")
    print("=" * 70)
    print(result2)

    # ============================================================
    # 清理
    # ============================================================
    client.close()
    print("\n\n执行完毕。")
