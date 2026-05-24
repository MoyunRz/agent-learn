"""
RAG 学习演示 —— 从文档加载到 LLM 生成，逐步展示每阶段的数据形态。

展示内容：
  Demo 1: 纯 RAG 管线（不涉及 LLM）
         文档 → 分块 → 向量化 → 检索，直接看数据
  Demo 2: 用 ReActAgent 调用 RAG 工具
         把 search 注册为工具，让 Agent 自主决定何时检索
  Demo 3: 带/不带 RAG 的对比
         同一问题，对比纯 LLM 和 RAG 增强的回答

运行：
    python demo_rag.py
"""

import os
import anthropic
from dotenv import load_dotenv

from agents.tools import Tool, ToolRegistry
from agents.react import ReActAgent
from agents.rag import SimpleRAG, DocumentLoader, TextChunker, TFIDFEmbedder, VectorStore
from agents.logging_config import setup_logging

load_dotenv()
setup_logging()

# ---- 共享 LLM 客户端 ----
client = anthropic.Anthropic(
    base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic"),
    api_key=os.getenv("MINIMAX_API_KEY", "your-key"),
)

# ═══════════════════════════════════════════════════════════
#  准备知识库
# ═══════════════════════════════════════════════════════════

KNOWLEDGE_TEXTS = [
    # 文档1：技术角度
    """
远程办公技术基础架构包括云计算、协同办公软件、VPN安全接入和视频会议系统。
2024年数据显示，企业级协同软件市场增长47%，Zoom日活用户突破5亿。
5G网络和边缘计算的普及进一步消除了远程办公的带宽瓶颈。
AI辅助工具如自动会议纪要、智能日程安排也显著提升了远程工作效率。
但网络安全仍是挑战：VPN攻击事件在2024年同比增长120%，数据泄露风险不容忽视。
    """,

    # 文档2：管理角度
    """
管理学研究显示，远程办公对员工生产力的影响呈U型曲线：适度远程（每周2-3天）
生产力提升13%，但完全远程可能导致下降5-8%。原因在于团队凝聚力减弱、
隐性知识传递受阻、新人培养周期延长。OKR目标管理和定期线上团建是常见应对策略。
GitLab的完全远程实践表明，完善的文档文化和异步沟通机制至关重要。
微软2024年报告指出，混合办公模式下员工满意度最高，达到82%。
    """,

    # 文档3：经济角度
    """
经济学分析显示，企业采用远程办公可降低运营成本20-35%，主要节省来自：
办公室租金（占节省的45%）、水电物业（15%）、通勤补贴（25%）、其他（15%）。
但同时也增加了远程办公津贴（人均每月1500-3000元）、数字化基础设施投入、
网络安全合规成本等隐性开支。综合计算，企业净节省约15-25%。
对员工而言，远程办公节省的通勤时间和交通费用年均折合约2-4万元。
    """,

    # 文档4：全球趋势
    """
全球远程办公采用率：2020年为42%（疫情峰值），2021年降至35%，2022年28%，
2023年回到32%，2024年稳定在34%。不同行业差异显著：IT行业67%，
金融业28%，制造业8%，医疗业12%。地域差异：北美44%，欧洲38%，亚洲22%。
麦肯锡预测，到2028年全球远程办公率将稳定在35-40%，形成常态化。
    """,

    # 文档5：房地产影响
    """
远程办公对商业地产的影响深远：2024年全球一线城市写字楼空置率攀升至18-25%。
纽约曼哈顿写字楼估值下跌23%，香港中环写字楼估值下跌18%。
但产业园和卫星办公室需求上升，郊区商业综合体投资回报率提升至8-12%。
企业从"集中式总部"转向"分布式办公节点+共享办公空间"模式。
到2028年，预计30%的写字楼将改造为混合用途建筑。
    """,
]


# ═══════════════════════════════════════════════════════════
#  Demo 1: 纯 RAG 管线（逐层看数据）
# ═══════════════════════════════════════════════════════════

def demo1_pipeline_solo():
    """逐步展示 RAG 每层的输入输出，不涉及 LLM。"""
    print("=" * 70)
    print("Demo 1: RAG 管线逐步演示")
    print("=" * 70)

    # 第1步：文档列表
    print("\n[Step 1] 原始文档（5篇知识库文章）")
    print(f"   文档数: {len(KNOWLEDGE_TEXTS)}")
    print(f"   总字数: {sum(len(t) for t in KNOWLEDGE_TEXTS)}")

    # 第2步：分块
    print("\n[Step 2] 文本分块（chunk_size=300, overlap=50）")
    chunker = TextChunker(chunk_size=300, overlap=50)
    docs = [{"source": f"doc_{i+1}", "content": t} for i, t in enumerate(KNOWLEDGE_TEXTS)]
    chunks = chunker.chunk_documents(docs)
    print(f"   产生 {len(chunks)} 个 chunk")
    print(f"   示例 chunk[0]（前100字）: {chunks[0]['content'][:100]}...")
    print(f"   示例 chunk[1]（前100字）: {chunks[1]['content'][:100]}...")
    # 展示overlap效果
    if len(chunks) > 1:
        c0_end = chunks[0]['content'][-50:]
        c1_start = chunks[1]['content'][:50]
        print(f"   chunk[0]尾部: ...{c0_end}")
        print(f"   chunk[1]头部: {c1_start}...")

    # 第3步：向量化（构建词表 + TF-IDF）
    print("\n[Step 3] TF-IDF 向量化")
    embedder = TFIDFEmbedder(max_features=200)
    texts = [c["content"] for c in chunks]
    embedder.fit(texts)
    print(f"   词表大小: {len(embedder.vocabulary)}")
    print(f"   示例词表前10: {list(embedder.vocabulary.keys())[:10]}")
    # 编码示例
    sample_vec = embedder.encode(texts[0])
    nonzero = [(i, round(v, 3)) for i, v in enumerate(sample_vec) if v > 0][:5]
    print(f"   chunk[0] 向量维度: {len(sample_vec)}")
    print(f"   chunk[0] 非零维前5: {nonzero}")

    # 第4步：存储 + 检索
    print("\n[Step 4] 向量检索（余弦相似度）")
    store = VectorStore(embedder)
    store.add_chunks(chunks)
    query = "远程办公能省多少钱？"
    results = store.search(query, top_k=3)
    print(f"   查询: '{query}'")
    print(f"   Top-3 结果:")
    for r in results:
        print(f"   [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")

    return chunks, results


# ═══════════════════════════════════════════════════════════
#  Demo 2: RAG 作为 Tool 供 ReActAgent 调用
# ═══════════════════════════════════════════════════════════

def demo2_agent_with_rag():
    """把 RAG 的 search 注册为工具，让 Agent 在 ReAct 循环中自主检索。"""
    print("\n\n" + "=" * 70)
    print("Demo 2: ReActAgent 自主调用 RAG 工具")
    print("=" * 70)

    # 建 RAG 索引
    rag = SimpleRAG(chunk_size=300, top_k=3)
    rag.load_from_texts(KNOWLEDGE_TEXTS, source="knowledge_base")

    # 把 search 包装为 Tool
    def search_knowledge_base(query: str) -> str:
        """检索远程办公知识库：给定查询，返回相关文档片段。
        适用场景：需要查找数据、统计、研究成果、行业报告等事实性信息时使用。"""
        results = rag.search(query)
        if not results:
            return "未找到相关信息"
        parts = []
        for r in results:
            parts.append(f"[来源: {r['source']} | 相关度: {r['score']}]")
            parts.append(f"{r['content']}")
            parts.append("")
        return "\n".join(parts)

    tools = ToolRegistry()
    tools.register(Tool(search_knowledge_base, name="search_kb",
                         description="检索远程办公知识库，获取事实数据和研究结果"))

    agent = ReActAgent(
        tools=tools,
        client=client,
        model="MiniMax-M2.7",
        max_steps=5,
        system_rules="遇到数据相关问题时，务必调用 search_kb 工具检索知识库。基于检索结果回答，不要凭记忆编造数据。用中文回答。",
    )

    question = "远程办公为企业节省了多少成本？员工个人节省了多少？请引用具体数字。"
    print(f"\n问题: {question}")
    print("\n--- Agent 推理过程 ---")
    answer = agent.run(question)
    print(f"\n--- 最终回答 ---\n{answer}")

    return answer


# ═══════════════════════════════════════════════════════════
#  Demo 3: 对比 —— 相同问题，有/无 RAG
# ═══════════════════════════════════════════════════════════

def demo3_rag_vs_no_rag():
    """同一问题分别问纯 LLM 和 RAG 增强，直观对比。"""
    print("\n\n" + "=" * 70)
    print("Demo 3: 带/不带 RAG 对比")
    print("=" * 70)

    question = "根据最新数据，远程办公的全球采用率是多少？不同行业有什么差异？"

    # --- 无 RAG ---
    print(f"\n[A] 纯 LLM（无检索）")
    print(f"   问题: {question}")
    agent_no_rag = ReActAgent(
        tools=ToolRegistry(),
        client=client,
        model="MiniMax-M2.7",
        system_rules="直接回答。如果不确定，如实说明。",
    )
    answer_no_rag = agent_no_rag.run(question)
    print(f"   回答: {answer_no_rag[:200]}...")

    # --- 有 RAG ---
    print(f"\n[B] RAG 增强（先检索再回答）")
    rag = SimpleRAG(chunk_size=300, top_k=3)
    rag.load_from_texts(KNOWLEDGE_TEXTS, source="knowledge_base")
    context = rag.retrieve_context(question)

    print(f"   检索到的上下文（注入 prompt）:")
    print(f"   {context[:300]}...")

    agent_with_rag = ReActAgent(
        tools=ToolRegistry(),
        client=client,
        model="MiniMax-M2.7",
        system_rules="基于提供的资料回答，引用具体数字和来源。",
    )
    prompt_with_context = f"{context}\n\n问题: {question}"
    answer_with_rag = agent_with_rag.run(prompt_with_context)
    print(f"\n   回答: {answer_with_rag[:200]}...")

    return answer_no_rag, answer_with_rag


# ═══════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Demo 1: 纯管线，不调 LLM，理解数据形态
    demo1_pipeline_solo()

    # Demo 2: RAG 注册为 Tool，ReActAgent 自主调用
    # demo2_agent_with_rag()

    # Demo 3: 同一问题，有无 RAG 的对比
    # demo3_rag_vs_no_rag()

    client.close()
    print("\n\n执行完毕。")
